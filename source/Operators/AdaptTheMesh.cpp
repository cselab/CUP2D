//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "AdaptTheMesh.h"

using namespace cubism;

void AdaptTheMesh::operator()(const double dt)
{
  count ++;

  if ((count-1) % 10 != 0 && count > 20) return;

  sim.startProfiler("AdaptTheMesh");

  // write total divergence and number of blocks to file
  // auto K = computeDivergence(sim);
  // K.run();

  // compute and store vorticity in tmp
  findOmega.run();
  
  //Refine according to chi and omega. Set tmp=inf wherever chi or dchi > 0.
  const std::vector<BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  const size_t Nblocks = tmpInfo.size();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-2,-2, 0}, stenEnd[3] = { 3, 3, 1};
    ScalarLab chilab;
    chilab.prepare(*(sim.chi), stenBeg, stenEnd, 1);
    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
      chilab.load(chiInfo[i], 0);
      auto& __restrict__ TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
      //Loop over block and halo cells and set TMP(0,0) to a value which will cause mesh refinement
      //if any of the cells have:
      // 1. chi > 0 (if bAdaptChiGradient=false)
      // 2. chi > 0 and chi < 0.9 (if bAdaptChiGradient=true)
      // Option 2 is equivalent to grad(chi) != 0
      const int offset = (tmpInfo[i].level == sim.tmp->getlevelMax()-1) ? 2 : 1;
      const double threshold = sim.bAdaptChiGradient ? 0.9 : 1e4;
      for(int y=-offset; y<VectorBlock::sizeY+offset; ++y)
      for(int x=-offset; x<VectorBlock::sizeX+offset; ++x)
      {
        chilab(x,y).s = std::min(chilab(x,y).s,1.0);
        chilab(x,y).s = std::max(chilab(x,y).s,0.0);
        if (chilab(x,y).s > 0.0 && chilab(x,y).s < threshold)
        {
          TMP(0,0).s = 2*sim.Rtol;
          break;
        }
      }
    }
  }


  bool verbose = true;
  bool basic = false;
  tmp_amr ->Tag();
  chi_amr ->TagLike(tmpInfo);
  pres_amr->TagLike(tmpInfo);
  pold_amr->TagLike(tmpInfo);
  vel_amr ->TagLike(tmpInfo);
  vOld_amr->TagLike(tmpInfo);
  tmpV_amr->TagLike(tmpInfo);
  uDef_amr->TagLike(tmpInfo);

  tmp_amr ->Adapt(sim.time,verbose,basic);
  verbose = false;
  chi_amr ->Adapt(sim.time,verbose,basic);
  vel_amr ->Adapt(sim.time,verbose,basic);
  vOld_amr->Adapt(sim.time,verbose,basic);
  pres_amr->Adapt(sim.time,verbose,basic);
  pold_amr->Adapt(sim.time,verbose,basic);
  basic = true;//this means that there's no interpolation of values at refinement
  tmpV_amr->Adapt(sim.time,verbose,basic);
  uDef_amr->Adapt(sim.time,verbose,basic);

  sim.stopProfiler();
}
