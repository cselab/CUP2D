//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PutObjectsOnGrid.h"
#include "../Shape.h"
#include "../Utils/BufferedLogger.h"

using namespace cubism;

static constexpr double EPS = std::numeric_limits<double>::epsilon();

void PutObjectsOnGrid::putChiOnGrid(Shape * const shape) const
{
  const size_t Nblocks = velInfo.size();

  const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
  double _x=0, _y=0, _m=0;
  #pragma omp parallel reduction(+ : _x, _y, _m)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab distlab; distlab.prepare(*(sim.tmp), stenBeg, stenEnd, 0);

    #pragma omp for schedule(dynamic, 1)
    for (size_t i=0; i < Nblocks; i++)
    {
      if(OBLOCK[chiInfo[i].blockID] == nullptr) continue; //obst not in block

      const Real h = chiInfo[i].h_gridpoint;

      ObstacleBlock& o = * OBLOCK[chiInfo[i].blockID];

      distlab.load(tmpInfo[i], 0); // loads signed distance field with ghosts

      auto & __restrict__ CHI  = *(ScalarBlock*)    chiInfo[i].ptrBlock;
      CHI_MAT & __restrict__ X = o.chi;
      const CHI_MAT & __restrict__ rho = o.rho;
      const CHI_MAT & __restrict__ sdf = o.dist;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
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
          const double distPx = distlab(ix+1,iy).s;
          const double distMx = distlab(ix-1,iy).s;
          const double distPy = distlab(ix,iy+1).s;
          const double distMy = distlab(ix,iy-1).s;
          const double IplusX = std::max(0.0,distPx);
          const double IminuX = std::max(0.0,distMx);
          const double IplusY = std::max(0.0,distPy);
          const double IminuY = std::max(0.0,distMy);
          const double gradIX = IplusX-IminuX;
          const double gradIY = IplusY-IminuY;
          const double gradUX = distPx-distMx;
          const double gradUY = distPy-distMy;
          const double gradUSq = gradUX * gradUX + gradUY * gradUY + EPS;
          X[iy][ix] = (gradIX*gradUX + gradIY*gradUY)/ gradUSq;
        }
        #endif

        // an other partial
        if(X[iy][ix] >= CHI(ix,iy).s)
        {
           CHI(ix,iy).s = X[iy][ix];
        }
        if(X[iy][ix] > 0)
        {
          double p[2]; chiInfo[i].pos(p, ix, iy);
          _x += rho[iy][ix] * X[iy][ix] * h*h * (p[0] - shape->centerOfMass[0]);
          _y += rho[iy][ix] * X[iy][ix] * h*h * (p[1] - shape->centerOfMass[1]);
          _m += rho[iy][ix] * X[iy][ix] * h*h;
        }
      }
    }
  }

  if(_m > EPS) {
    shape->centerOfMass[0] += _x/_m;
    shape->centerOfMass[1] += _y/_m;
    shape->M = _m;
  } else printf("PutObjectsOnGrid _m is too small!\n");

#if 1 //more accurate, uses actual values of mollified Heaviside
  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab chilab; chilab.prepare(*(sim.chi), stenBeg, stenEnd, 0);
    ScalarLab distlab; distlab.prepare(*(sim.tmp), stenBeg, stenEnd, 0);
    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
      if(OBLOCK[chiInfo[i].blockID] == nullptr) continue; //obst not in block

      const Real h = chiInfo[i].h_gridpoint, i2h = 0.5/h, fac = 0.5*h; // fac explained down

      ObstacleBlock& o = * OBLOCK[chiInfo[i].blockID];
      chilab.load (chiInfo[i], 0);
      distlab.load(tmpInfo[i], 0);
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
          const double gradHX = chilab(ix+1,iy).s-chilab(ix-1,iy).s;
          const double gradHY = chilab(ix,iy+1).s-chilab(ix,iy-1).s;
          if (gradHX*gradHX + gradHY*gradHY < 1e-12) continue;
          const double gradUX = i2h*(distlab(ix+1,iy).s-distlab(ix-1,iy).s);
          const double gradUY = i2h*(distlab(ix,iy+1).s-distlab(ix,iy-1).s);
          const double gradUSq = gradUX * gradUX + gradUY * gradUY + EPS;
          const double D = fac*(gradHX*gradUX + gradHY*gradUY)/gradUSq;
          if (std::fabs(D) > EPS) o.write(ix, iy, D, gradUX, gradUY);
      }
    }
  }
#endif

  for (auto & o : OBLOCK) if(o not_eq nullptr) o->allocate_surface();
}

void PutObjectsOnGrid::putObjectVelOnGrid(Shape * const shape) const
{
  const size_t Nblocks = velInfo.size();

  const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
  //const double u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
  //const double Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++)
  {
    if(OBLOCK[uDefInfo[i].blockID] == nullptr) continue; //obst not in block
    //using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
    //using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];

    const UDEFMAT & __restrict__ udef = OBLOCK[uDefInfo[i].blockID]->udef;
    const CHI_MAT & __restrict__ chi  = OBLOCK[uDefInfo[i].blockID]->chi;
    auto & __restrict__ UDEF = *(VectorBlock*)uDefInfo[i].ptrBlock; // dest
    //const ScalarBlock&__restrict__ TMP  = *(ScalarBlock*) tmpInfo[i].ptrBlock;
    const ScalarBlock&__restrict__ CHI  = *(ScalarBlock*) chiInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
    {
      if( chi[iy][ix] < CHI(ix,iy).s) continue;
      Real p[2]; uDefInfo[i].pos(p, ix, iy);
      UDEF(ix, iy).u[0] += udef[iy][ix][0];
      UDEF(ix, iy).u[1] += udef[iy][ix][1];
    }
    //if (TMP(ix,iy).s > -3*h) //( chi[iy][ix] > 0 )
    //{ //plus equal in case of overlapping objects
    //  Real p[2]; uDefInfo[i].pos(p, ix, iy);
    //  UDEF(ix,iy).u[0] += u_s - omega_s*(p[1]-Cy) + udef[iy][ix][0];
    //  UDEF(ix,iy).u[1] += v_s + omega_s*(p[0]-Cx) + udef[iy][ix][1];
    //}
  }
}

void PutObjectsOnGrid::operator()(const double dt)
{
  const size_t Nblocks = velInfo.size();

  sim.startProfiler("PutObjectsGrid");
  //// 0) clear fields related to obstacle
  if(sim.verbose)
    std::cout << "[CUP2D] - clear..." << std::endl;
  // sim.startProfiler("PutObjectsOnGrid - clear");
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++) {
    ( (ScalarBlock*)   chiInfo[i].ptrBlock )->clear();
    ( (ScalarBlock*)   tmpInfo[i].ptrBlock )->set(-1);
    ( (VectorBlock*)  uDefInfo[i].ptrBlock )->clear();
  }
  // sim.stopProfiler();


  //// 1) update objects' position
  if(sim.verbose)
    std::cout << "[CUP2D] - move..." << std::endl;
  // sim.startProfiler("PutObjectsOnGrid - move");
  // 1a) Update laboratory frame of reference
  int nSum[2] = {0, 0}; double uSum[2] = {0, 0};
  for(Shape * const shape : sim.shapes) 
    shape->updateLabVelocity(nSum, uSum);
  if(nSum[0]>0) {sim.uinfx_old = sim.uinfx; sim.uinfx = uSum[0]/nSum[0];}
  if(nSum[1]>0) {sim.uinfy_old = sim.uinfy; sim.uinfy = uSum[1]/nSum[1];}
  // 1b) Update position of object r^{t+1}=r^t+dt*v, \theta^{t+1}=\theta^t+dt*\omega
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
  // sim.stopProfiler();

  //// 2) Compute signed dist function and udef
  if(sim.verbose)
    std::cout << "[CUP2D] - signed dist..." << std::endl;
  // sim.startProfiler("PutObjectsOnGrid - signed dist");
  for(const auto& shape : sim.shapes) 
    shape->create(tmpInfo);
  // sim.stopProfiler();

  //// 3) Compute chi
  if(sim.verbose)
    std::cout << "[CUP2D] - chi..." << std::endl;
  // sim.startProfiler("PutObjectsOnGrid - chi");
  for(const auto& shape : sim.shapes) 
    putChiOnGrid( shape );
  // sim.stopProfiler();

  //// 4) remove moments from characteristic function and put on grid U_s
  if(sim.verbose)
    std::cout << "[CUP2D] - Us..." << std::endl;
  // sim.startProfiler("PutObjectsOnGrid - Us");
  for(const auto& shape : sim.shapes) {
    shape->removeMoments(chiInfo);
    putObjectVelOnGrid(shape);
  }
  sim.stopProfiler();
}
