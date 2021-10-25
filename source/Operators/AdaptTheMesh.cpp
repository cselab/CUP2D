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
  const StencilInfo stencil{-3, -3, 0, 4, 4, 1, true, {0}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  void operator()(ScalarLab & lab, const BlockInfo& info) const
  {
    if (info.level == sim.tmp->getlevelMax()-1) return;
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;

    const double ih = 1.0/info.h;
    const double p3 =  ih/60.0;
    const double p2 = -ih*0.15;
    const double p1 =  ih*0.75;
    for(int y=0; y<VectorBlock::sizeY; ++y)
    for(int x=0; x<VectorBlock::sizeX; ++x)
    {
      if( sim.bAdaptChiGradient )
      {
        double dcdx = p3*lab(x+3,y).s + p2*lab(x+2,y).s + p1*lab(x+1,y).s
                     -p1*lab(x-1,y).s - p2*lab(x-2,y).s - p3*lab(x-3,y).s;
        double dcdy = p3*lab(x,y+3).s + p2*lab(x,y+2).s + p1*lab(x,y+1).s
                     -p1*lab(x,y-1).s - p2*lab(x,y-2).s - p3*lab(x,y-3).s;
        if (tmpInfo[info.blockID].level <= sim.tmp->getlevelMax() - 2)
        {
          dcdx = 0.5*ih*(lab(x+1,y).s-lab(x-1,y).s);
          dcdy = 0.5*ih*(lab(x,y+1).s-lab(x,y-1).s);
        }
        const double norm = dcdx*dcdx+dcdy*dcdy;
        if (norm > 0.1)
        {
          TMP(x,y).s = 1e10;
          return;
        }
      }
      else if ( lab(x,y).s > 0.0 ) TMP(x,y).s = 1e10;
    }
  }
};


void AdaptTheMesh::operator()(const double dt)
{
  if (sim.step > 10 && sim.step % 20 != 0) return;
  //if (sim.step > 10 && sim.step % 5 != 0) return;
  sim.startProfiler("AdaptTheMesh");

  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();

  // compute vorticity (and use it as refinement criterion) and store it to tmp. 
  auto K1 = computeVorticity(sim);
  K1(0);

  // compute grad(chi) and if it's >0 set tmp = infinity
  GradChiOnTmp K2(sim);
  compute<GradChiOnTmp,ScalarGrid,ScalarLab>(K2,*sim.chi,false);

  tmp_amr ->Tag();
  chi_amr ->TagLike(tmpInfo);
  pres_amr->TagLike(tmpInfo);
  pold_amr->TagLike(tmpInfo);
  vel_amr ->TagLike(tmpInfo);
  vOld_amr->TagLike(tmpInfo);
  tmpV_amr->TagLike(tmpInfo);
  uDef_amr->TagLike(tmpInfo);

  tmp_amr ->Adapt(sim.time,sim.rank == 0,false);
  chi_amr ->Adapt(sim.time,false        ,false);
  vel_amr ->Adapt(sim.time,false        ,false);
  vOld_amr->Adapt(sim.time,false        ,false);
  pres_amr->Adapt(sim.time,false        ,false);
  pold_amr->Adapt(sim.time,false        ,false);
  tmpV_amr->Adapt(sim.time,false        ,true );
  uDef_amr->Adapt(sim.time,false        ,true );

  sim.stopProfiler();
}
