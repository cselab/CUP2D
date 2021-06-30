//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiff.h"

using namespace cubism;

static inline Real dU_adv_dif(const VectorLab&V, const Real uinf[2], const Real advF, const Real difF, const int ix, const int iy)
{
  const Real u    = V(ix,iy).u[0];
  const Real v    = V(ix,iy).u[1];
  const Real UU   = u + uinf[0];
  const Real VV   = v + uinf[1];

  const Real up1x = V(ix+1,iy).u[0];
  const Real up2x = V(ix+2,iy).u[0];
  const Real um1x = V(ix-1,iy).u[0];
  const Real um2x = V(ix-2,iy).u[0];
  const Real dudx = UU>0 ? (2*up1x + 3*u - 6*um1x + um2x) : (-up2x + 6*up1x - 3*u - 2*um1x);

  const Real up1y = V(ix,iy+1).u[0];
  const Real up2y = V(ix,iy+2).u[0];
  const Real um1y = V(ix,iy-1).u[0];
  const Real um2y = V(ix,iy-2).u[0];
  const Real dudy = VV>0 ? (2*up1y + 3*u - 6*um1y + um2y) : (-up2y + 6*up1y - 3*u - 2*um1y);

  return advF*(UU*dudx+VV*dudy) + difF*(up1x + up1y + um1x + um1y - 4*u);
}

static inline Real dV_adv_dif(const VectorLab&V, const Real uinf[2], const Real advF, const Real difF, const int ix, const int iy)
{
  const Real u    = V(ix,iy).u[0];
  const Real v    = V(ix,iy).u[1];
  const Real UU   = u + uinf[0];
  const Real VV   = v + uinf[1];

  const Real vp1x = V(ix+1,iy).u[1];
  const Real vp2x = V(ix+2,iy).u[1];
  const Real vm1x = V(ix-1,iy).u[1];
  const Real vm2x = V(ix-2,iy).u[1];
  const Real dvdx = UU>0 ? (2*vp1x + 3*v - 6*vm1x + vm2x) : (-vp2x + 6*vp1x - 3*v - 2*vm1x);

  const Real vp1y = V(ix,iy+1).u[1];
  const Real vp2y = V(ix,iy+2).u[1];
  const Real vm1y = V(ix,iy-1).u[1];
  const Real vm2y = V(ix,iy-2).u[1];
  const Real dvdy = VV>0 ? (2*vp1y + 3*v - 6*vm1y + vm2y) : (-vp2y + 6*vp1y - 3*v - 2*vm1y);

  return advF*(UU*dvdx+VV*dvdy) + difF*(vp1x + vp1y + vm1x + vm1y - 4*v);
}

void advDiff::operator()(const double dt)
{
  sim.startProfiler("advDiff");
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;
  const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0;          };
  const auto isE = [&](const BlockInfo&I) { return I.index[0] == sim.bpdx-1; };
  const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0;          };
  const auto isN = [&](const BlockInfo&I) { return I.index[1] == sim.bpdy-1; };

  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  //const Real G[]= {sim.gravity[0],sim.gravity[1]};
  const Real dfac = (sim.nu/h)*(dt/h), afac = -dt/h/6.0;
  const Real fac = std::min((Real)1, sim.uMax_measured * dt / h);
  const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
  const Real fadeW= 1 - fac * std::pow(std::max(UINF[0], (Real)0)/norUinf, 2);
  const Real fadeS= 1 - fac * std::pow(std::max(UINF[1], (Real)0)/norUinf, 2);
  const Real fadeE= 1 - fac * std::pow(std::min(UINF[0], (Real)0)/norUinf, 2);
  const Real fadeN= 1 - fac * std::pow(std::min(UINF[1], (Real)0)/norUinf, 2);
  const auto fade = [&](VectorElement&B,const Real F) { B.u[0]*=F; B.u[1]*=F; };

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-2,-2, 0}, stenEnd[3] = { 3, 3, 1};

    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      if(isW(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(BX-1,iy), fadeW);
      if(isS(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,BY-1), fadeS);
      if(isE(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(EX+1,iy), fadeE);
      if(isN(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,EY+1), fadeN);

      for(int iy=0; iy<BSY; ++iy) for(int ix=0; ix<BSX; ++ix)
      {
        TMP(ix,iy).u[0] = V(ix,iy).u[0] + dU_adv_dif(V,UINF,afac,dfac,ix,iy);
        TMP(ix,iy).u[1] = V(ix,iy).u[1] + dV_adv_dif(V,UINF,afac,dfac,ix,iy);
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
          VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    V.copy(T);
  }

  {
    const std::vector<size_t>& boundaryInfoIDs = sim.boundaryInfoIDs;
    const size_t NboundaryBlocks = boundaryInfoIDs.size();
    ////////////////////////////////////////////////////////////////////////////
    Real IF = 0;
    #pragma omp parallel for schedule(static) reduction(+ : IF)
    for (size_t k=0; k < NboundaryBlocks; k++) {
      const size_t i = boundaryInfoIDs[k];
      const VectorBlock& V = *(VectorBlock*) velInfo[i].ptrBlock;
      if(isW(velInfo[i])) for(int iy=0; iy<BSY; ++iy) IF -= V(BX,iy).u[0];
      if(isE(velInfo[i])) for(int iy=0; iy<BSY; ++iy) IF += V(EX,iy).u[0];
      if(isS(velInfo[i])) for(int ix=0; ix<BSX; ++ix) IF -= V(ix,BY).u[1];
      if(isN(velInfo[i])) for(int ix=0; ix<BSX; ++ix) IF += V(ix,EY).u[1];
    }
    ////////////////////////////////////////////////////////////////////////////
    //const Real corr = IF/std::max(AF, EPS);
    const Real corr = IF/( 2*(BSY*sim.bpdy -0) + 2*(BSX*sim.bpdx -0) );
    //if(sim.verbose) printf("Relative inflow correction %e\n",corr);
    #pragma omp parallel for schedule(static)
    for (size_t k=0; k < NboundaryBlocks; k++) {
      const size_t i = boundaryInfoIDs[k];
      VectorBlock& V = *(VectorBlock*) velInfo[i].ptrBlock;
      if(isW(velInfo[i])) for(int iy=0; iy<BSY; ++iy) V(BX,iy).u[0] += corr;
      if(isE(velInfo[i])) for(int iy=0; iy<BSY; ++iy) V(EX,iy).u[0] -= corr;
      if(isS(velInfo[i])) for(int ix=0; ix<BSX; ++ix) V(ix,BY).u[1] += corr;
      if(isN(velInfo[i])) for(int ix=0; ix<BSX; ++ix) V(ix,EY).u[1] -= corr;
    }
  }
  sim.stopProfiler();
}
