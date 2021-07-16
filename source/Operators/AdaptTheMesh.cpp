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

  if ((count-1) % 10 != 0 && count > 10) return;

  sim.startProfiler("AdaptTheMesh");

  // write total divergence and number of blocks to file
  auto K = computeDivergence(sim); 
  K.run();

  // compute and store vorticity in tmp
  findOmega.run();
  
  //Refine according to chi and omega. Set tmp=inf wherever chi or dchi > 0.
  const std::vector<BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  const size_t Nblocks = tmpInfo.size();

  #pragma omp parallel
  {
    //static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    static constexpr int stenBeg[3] = {-3,-3, 0}, stenEnd[3] = { 4, 4, 1};
    ScalarLab chilab;
    chilab.prepare(*(sim.chi), stenBeg, stenEnd, 1);
    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
      chilab.load(chiInfo[i], 0);
      auto& __restrict__ TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
      const double i2h = 0.5/chiInfo[i].h;
      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
      {
        if( sim.bAdaptChiGradient )
        {
          //const double dcdx = i2h*(chilab(x+1,y).s-chilab(x-1,y).s);
          //const double dcdy = i2h*(chilab(x,y+1).s-chilab(x,y-1).s);

          const double dcdx = 2*i2h*((1.0/60.0)*chilab(x+3,y).s + (-0.15)*chilab(x+2,y).s + (0.75)*chilab(x+1,y).s
                                  + (-0.75)*chilab(x-1,y).s + (0.15)*chilab(x-2,y).s + (-1.0/60.0)*chilab(x-3,y).s);
          const double dcdy = 2*i2h*((1.0/60.0)*chilab(x,y+3).s + (-0.15)*chilab(x,y+2).s + (0.75)*chilab(x,y+1).s
                                  + (-0.75)*chilab(x,y-1).s + (0.15)*chilab(x,y-2).s + (-1.0/60.0)*chilab(x,y-3).s);


          const double norm = dcdx*dcdx+dcdy*dcdy;
          if (norm > 0.1) TMP(x,y).s = 1e10;
        }
        else if ( chilab(x,y).s > 0.0 ) TMP(x,y).s = 1e10;
      }
    }
  }

  double Rtol = sim.Rtol;
  double Ctol = sim.Ctol;

  bool verbose = sim.verbose;
  ScalarAMR tmp_amr ( *sim.tmp ,Rtol,Ctol,verbose);//refine according to tmp (vorticity magnitude)
  verbose = false;
  ScalarAMR chi_amr ( *sim.chi ,0.05,0.01,verbose);
  VectorAMR vel_amr ( *sim.vel ,Rtol,Ctol,verbose);
  VectorAMR vOld_amr( *sim.vOld,Rtol,Ctol,verbose);
  ScalarAMR pres_amr( *sim.pres,Rtol,Ctol,verbose);
  ScalarAMR pold_amr( *sim.pold,Rtol,Ctol,verbose);

  MeshAdaptation_basic<VectorGrid,ScalarGrid>tmpV_amr(*sim.tmpV);
  MeshAdaptation_basic<VectorGrid,ScalarGrid>uDef_amr(*sim.uDef);

  tmp_amr .AdaptTheMesh();
  chi_amr .AdaptLikeOther(*sim.tmp);
  vel_amr .AdaptLikeOther(*sim.tmp);
  vOld_amr.AdaptLikeOther(*sim.tmp);
  pres_amr.AdaptLikeOther(*sim.tmp);
  pold_amr.AdaptLikeOther(*sim.tmp);
  uDef_amr.AdaptLikeOther(*sim.tmp);
  tmpV_amr.AdaptLikeOther(*sim.tmp);

  sim.chi ->SortBlocks();
  sim.vel ->SortBlocks();
  sim.vOld->SortBlocks();
  sim.pres->SortBlocks();
  sim.pold->SortBlocks();
  sim.tmpV->SortBlocks();
  sim.tmp ->SortBlocks();
  sim.uDef->SortBlocks();
  sim.stopProfiler();
}
