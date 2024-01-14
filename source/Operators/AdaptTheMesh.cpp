//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "AdaptTheMesh.h"

using namespace cubism;

struct GradChiOnTmp
{
  GradChiOnTmp(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  //const StencilInfo stencil{-2, -2, 0, 3, 3, 1, true, {0}};
  const StencilInfo stencil{-4, -4, 0, 5, 5, 1, true, {0}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  void operator()(ScalarLab & lab, const BlockInfo& info) const
  {
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    if (sim.Qcriterion)
      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
        TMP(x,y).s = std::max(TMP(x,y).s,(Real)0.0);//compress if Q<0

    //Loop over block and halo cells and set TMP(0,0) to a value which will cause mesh refinement
    //if any of the cells have:
    // 1. chi > 0 (if bAdaptChiGradient=false)
    // 2. chi > 0 and chi < 0.9 (if bAdaptChiGradient=true)
    // Option 2 is equivalent to grad(chi) != 0
    //const int offset = (info.level == sim.tmp->getlevelMax()-1) ? 2 : 1;
    const int offset = (info.level == sim.tmp->getlevelMax()-1) ? 4 : 2;
    const Real threshold = sim.bAdaptChiGradient ? 0.9 : 1e4;
    for(int y=-offset; y<VectorBlock::sizeY+offset; ++y)
    for(int x=-offset; x<VectorBlock::sizeX+offset; ++x)
    {
      lab(x,y).s = std::min(lab(x,y).s,(Real)1.0);
      lab(x,y).s = std::max(lab(x,y).s,(Real)0.0);
      if (lab(x,y).s > 0.0 && lab(x,y).s < threshold)
      {
        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
        break;
      }
    }

    #ifdef CUP2D_CYLINDER_REF
    //Hardcoded refinement close the wall, for the high Re cylinder cases.
    //Cylinder center is supposed to be at (1.0,1.0) and its radius is 0.1
    for(int y=0; y<VectorBlock::sizeY; ++y)
    for(int x=0; x<VectorBlock::sizeX; ++x)
    {
      double p[2];
      info.pos(p,x,y);
      p[0] -= 1.0;
      p[1] -= 1.0;
      const double r = p[0]*p[0]+p[1]*p[1];
      if (r>0.1*0.1 && r < 0.11*0.11)
      {
        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
        break;
      }
    }
    #endif
  }
};


void AdaptTheMesh::operator()(const Real dt)
{  
  if (sim.step > 10 && sim.step % sim.AdaptSteps != 0) return;
  adapt();
}

void AdaptTheMesh::adapt()
{
  sim.startProfiler("AdaptTheMesh");

  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();

  // compute vorticity (and use it as refinement criterion) and store it to tmp. 
  if (sim.Qcriterion)
  {
     auto K1 = computeQ(sim);
     K1(0);
  }
  else{
     auto K1 = computeVorticity(sim);
     K1(0);
  }

  // compute grad(chi) and if it's >0 set tmp = infinity
  GradChiOnTmp K2(sim);
  cubism::compute<ScalarLab>(K2,sim.chi);

  tmp_amr ->Tag();
  chi_amr ->TagLike(tmpInfo);
  pres_amr->TagLike(tmpInfo);
  pold_amr->TagLike(tmpInfo);
  vel_amr ->TagLike(tmpInfo);
  vOld_amr->TagLike(tmpInfo);
  tmpV_amr->TagLike(tmpInfo);
  if( sim.smagorinskyCoeff != 0 )
    Cs_amr->TagLike(tmpInfo);

  tmp_amr ->Adapt(sim.time, sim.rank == 0 && !sim.muteAll, false);
  chi_amr ->Adapt(sim.time, false, false);
  vel_amr ->Adapt(sim.time, false, false);
  vOld_amr->Adapt(sim.time, false, false);
  pres_amr->Adapt(sim.time, false, false);
  pold_amr->Adapt(sim.time, false, false);
  tmpV_amr->Adapt(sim.time, false, true);
  if( sim.smagorinskyCoeff != 0 )
    Cs_amr->Adapt(sim.time, false, true);

  sim.stopProfiler();
}
