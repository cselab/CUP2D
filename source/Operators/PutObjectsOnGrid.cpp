//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "PutObjectsOnGrid.h"
#include "../Shape.h"
#include "../Utils/BufferedLogger.h"

using namespace cubism;

static constexpr double EPS = std::numeric_limits<double>::epsilon();

struct ComputeSurfaceNormals
{
  ComputeSurfaceNormals(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  StencilInfo stencil {-1, -1, 0, 2, 2, 1, false, {0}};
  StencilInfo stencil2{-1, -1, 0, 2, 2, 1, false, {0}};
  void operator()(ScalarLab & labChi, ScalarLab & labSDF, const BlockInfo& infoChi, const BlockInfo& infoSDF) const
  {
    for(const auto& shape : sim.shapes)
    {
      const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
      if(OBLOCK[infoChi.blockID] == nullptr) continue; //obst not in block
      const Real h = infoChi.h;
      ObstacleBlock& o = * OBLOCK[infoChi.blockID];
      const double i2h = 0.5/h;
      const double fac = 0.5*h;
      for(int iy=0; iy<ScalarBlock::sizeY; iy++)
      for(int ix=0; ix<ScalarBlock::sizeX; ix++)
      {
          const double gradHX = labChi(ix+1,iy).s-labChi(ix-1,iy).s;
          const double gradHY = labChi(ix,iy+1).s-labChi(ix,iy-1).s;
          if (gradHX*gradHX + gradHY*gradHY < 1e-12) continue;
          const double gradUX = i2h*(labSDF(ix+1,iy).s-labSDF(ix-1,iy).s);
          const double gradUY = i2h*(labSDF(ix,iy+1).s-labSDF(ix,iy-1).s);
          const double gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
          const double D = fac*(gradHX*gradUX + gradHY*gradUY)/gradUSq;
          if (std::fabs(D) > EPS) o.write(ix, iy, D, gradUX, gradUY);
      }
      o.allocate_surface();
    }
  }
};

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
          const double distPx = lab(ix+1,iy).s;
          const double distMx = lab(ix-1,iy).s;
          const double distPy = lab(ix,iy+1).s;
          const double distMy = lab(ix,iy-1).s;
          const double IplusX = std::max(0.0,distPx);
          const double IminuX = std::max(0.0,distMx);
          const double IplusY = std::max(0.0,distPy);
          const double IminuY = std::max(0.0,distMy);
          const double gradIX = IplusX-IminuX;
          const double gradIY = IplusY-IminuY;
          const double gradUX = distPx-distMx;
          const double gradUY = distPy-distMy;
          const double gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
          X[iy][ix] = (gradIX*gradUX + gradIY*gradUY)/ gradUSq;
        }
        #endif
        CHI(ix,iy).s = std::max(CHI(ix,iy).s,X[iy][ix]);
        if(X[iy][ix] > 0)
        {
          double p[2];
          info.pos(p, ix, iy);
          o.COM_x += X[iy][ix] * h2 * (p[0] - shape->centerOfMass[0]);
          o.COM_y += X[iy][ix] * h2 * (p[1] - shape->centerOfMass[1]);
          o.Mass  += X[iy][ix] * h2;
        }
      }
    }
  }
};

void PutObjectsOnGrid::putObjectVelOnGrid(Shape * const shape) const
{
  const size_t Nblocks = velInfo.size();
  const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    if(OBLOCK[uDefInfo[i].blockID] == nullptr) continue; //obst not in block
    const UDEFMAT & __restrict__ udef = OBLOCK[uDefInfo[i].blockID]->udef;
    const CHI_MAT & __restrict__ chi  = OBLOCK[uDefInfo[i].blockID]->chi;
    auto & __restrict__ UDEF = *(VectorBlock*)uDefInfo[i].ptrBlock; // dest
    const ScalarBlock&__restrict__ CHI  = *(ScalarBlock*) chiInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
    {
      if( chi[iy][ix] < CHI(ix,iy).s) continue;
      Real p[2]; uDefInfo[i].pos(p, ix, iy);
      UDEF(ix, iy).u[0] += udef[iy][ix][0];
      UDEF(ix, iy).u[1] += udef[iy][ix][1];
    }
  }
}

void PutObjectsOnGrid::operator()(const double dt)
{
  sim.startProfiler("PutObjectsGrid");

  const size_t Nblocks = velInfo.size();

  // 1. Clear fields related to obstacle
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    ( (ScalarBlock*)  chiInfo[i].ptrBlock )->clear();
    ( (ScalarBlock*)  tmpInfo[i].ptrBlock )->set(-1);
    ( (VectorBlock*) uDefInfo[i].ptrBlock )->clear();
  }

  // 2. Update object position
  // Update laboratory frame of reference
  int nSum[2] = {0, 0}; double uSum[2] = {0, 0};
  for(Shape * const shape : sim.shapes) 
    shape->updateLabVelocity(nSum, uSum);
  if(nSum[0]>0) {sim.uinfx_old = sim.uinfx; sim.uinfx = uSum[0]/nSum[0];}
  if(nSum[1]>0) {sim.uinfy_old = sim.uinfy; sim.uinfy = uSum[1]/nSum[1];}
  // Update position of object r^{t+1}=r^t+dt*v, \theta^{t+1}=\theta^t+dt*\omega
  for(Shape * const shape : sim.shapes)
  {
    shape->updatePosition(dt);

    // .. and check if shape is outside the simulation domain
    double p[2] = {0,0};
    shape->getCentroid(p);
    const auto& extent = sim.extents;
    if (p[0]<0 || p[0]>extent[0] || p[1]<0 || p[1]>extent[1]) {
      printf("[CUP2D] ABORT: Body out of domain [0,%f]x[0,%f] CM:[%e,%e]\n",
        extent[0], extent[1], p[0], p[1]);
      fflush(0);
      abort();
    }
  }

  // 3) Compute signed dist function and udef
  for(const auto& shape : sim.shapes) 
    shape->create(tmpInfo);

  // 4) Compute chi and shape center of mass
  const PutChiOnGrid K(sim);
  compute<PutChiOnGrid,ScalarGrid,ScalarLab>(K,*sim.tmp,false);
  const ComputeSurfaceNormals K1(sim);
  compute<ComputeSurfaceNormals,ScalarGrid,ScalarLab,ScalarGrid,ScalarLab>(K1,*sim.chi,*sim.tmp);
  for(const auto& shape : sim.shapes)
  {
    double com[3] = {0.0, 0.0, 0.0};
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    #pragma omp parallel for reduction(+ : com[:3])
    for (size_t i=0; i<OBLOCK.size(); i++)
    {
      if(OBLOCK[i] == nullptr) continue;
      com[0] += OBLOCK[i]->Mass;
      com[1] += OBLOCK[i]->COM_x;
      com[2] += OBLOCK[i]->COM_y;
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 3, MPI_DOUBLE,MPI_SUM, sim.chi->getCartComm());
    shape->M = com[0];
    shape->centerOfMass[0] += com[1]/com[0];
    shape->centerOfMass[1] += com[2]/com[0];
  }

  //// 5) remove moments from characteristic function and put on grid U_s
  for(const auto& shape : sim.shapes)
  {
    shape->removeMoments(chiInfo);
    putObjectVelOnGrid(shape);
  }
  sim.stopProfiler();
}