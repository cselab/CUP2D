#include "PutSoftObjectsOnGrid.h"
#include "../Shape.h"

using namespace cubism;

static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
struct PutChiOnGrid
{
  PutChiOnGrid(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  void operator()(ScalarLab & lab, const BlockInfo& info) const
  {
    for(const auto& shape : sim.shapes)
    {
      const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
      if(OBLOCK[info.blockID] == nullptr) continue; //obst not in block
      const Real h = info.h;
      const Real h2 = h*h;
      ObstacleBlock& o = * OBLOCK[info.blockID];
      CHI_MAT & __restrict__ X = o.chi;
      const CHI_MAT & __restrict__ sdf = o.dist;
      o.COM_x = 0;
      o.COM_y = 0;
      o.Mass  = 0;
      auto & __restrict__ CHI  = *(ScalarBlock*) chiInfo[info.blockID].ptrBlock;
      for(int iy=0; iy<ScalarBlock::sizeY; iy++)
      for(int ix=0; ix<ScalarBlock::sizeX; ix++)
      {
        #if 0
        X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
        #else //Towers mollified Heaviside
        if (sdf[iy][ix] > +h || sdf[iy][ix] < -h)
        {
          X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
        }
        else
        {
          const Real distPx = lab(ix+1,iy).s;
          const Real distMx = lab(ix-1,iy).s;
          const Real distPy = lab(ix,iy+1).s;
          const Real distMy = lab(ix,iy-1).s;
          const Real IplusX = std::max((Real)0.0,distPx);
          const Real IminuX = std::max((Real)0.0,distMx);
          const Real IplusY = std::max((Real)0.0,distPy);
          const Real IminuY = std::max((Real)0.0,distMy);
          const Real gradIX = IplusX-IminuX;
          const Real gradIY = IplusY-IminuY;
          const Real gradUX = distPx-distMx;
          const Real gradUY = distPy-distMy;
          const Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
          X[iy][ix] = (gradIX*gradUX + gradIY*gradUY)/ gradUSq;
        }
        #endif
        CHI(ix,iy).s = std::max(CHI(ix,iy).s,X[iy][ix]);
        if(X[iy][ix] > 0)
        {
          Real p[2];
          info.pos(p, ix, iy);
          o.COM_x += X[iy][ix] * h2 * (p[0] - shape->centerOfMass[0]);
          o.COM_y += X[iy][ix] * h2 * (p[1] - shape->centerOfMass[1]);
          o.Mass  += X[iy][ix] * h2;
        }
      }
    }
  }
};
void PutSoftObjectsOnGrid::operator()()
{
  sim.startProfiler("PutSoftObjectsGrid");

  putSoftObjectsOnGrid();

  sim.stopProfiler();
}
void PutSoftObjectsOnGrid::putSoftObjectsOnGrid()
{
  const size_t Nblocks = velInfo.size();

  // 1) Clear fields related to obstacle
  Real signal=500;
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    ( (ScalarBlock*)  chiInfo[i].ptrBlock )->clear();
    ( (ScalarBlock*)  tmpInfo[i].ptrBlock )->set(signal);
  }

  // 2) Compute signed dist function (only valid in the region with valid inverse map value)
  for(const auto& shape : sim.shapes)
    shape->create(tmpInfo,signal);
  // 3) compute chi based on signed dist function
  const PutChiOnGrid K(sim);
  cubism::compute<ScalarLab>(K,sim.tmp);
}