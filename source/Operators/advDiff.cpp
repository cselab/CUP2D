//
//  CubismUP_2D - AMR
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "advDiff.h"

using namespace cubism;

#if 1 // 3rd-order upwind
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

#else //QUICK scheme

static inline Real dU_adv_dif(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  // get grid values
  const Real uppx = V(ix+2, iy).u[0], uppy = V(ix, iy+2).u[0];
  const Real upx  = V(ix+1, iy).u[0], upy  = V(ix, iy+1).u[0];
  const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
  const Real ulx  = V(ix-1, iy).u[0], uly  = V(ix, iy-1).u[0];
  const Real ullx = V(ix-2, iy).u[0], ully = V(ix, iy-2).u[0];

  // advection
  const Real u = ucc+uinf[0];
  const Real dudx  = u > 0 ?           3*upx + 3*ucc - 7*ulx + ullx
                             : -uppx + 7*upx - 3*ucc - 3*ulx        ;
  const Real v = vcc+uinf[1];
  const Real dudy  = v > 0 ?           3*upy + 3*ucc - 7*uly + ully
                             : -uppy + 7*upy - 3*ucc - 3*uly        ;
  const Real dUadv = u * 0.125 * dudx + v * 0.125 * dudy;

  // diffusion
  const Real dUdif = upx + upy + ulx + uly - 4 *ucc;

  return advF * dUadv + difF * dUdif;
}

static inline Real dV_adv_dif(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  const Real vppx = V(ix+2, iy).u[1], vppy = V(ix, iy+2).u[1];
  const Real vpx  = V(ix+1, iy).u[1], vpy  = V(ix, iy+1).u[1];
  const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
  const Real vlx  = V(ix-1, iy).u[1], vly  = V(ix, iy-1).u[1];
  const Real vllx = V(ix-2, iy).u[1], vlly = V(ix, iy-2).u[1];

  // advection
  const Real u = ucc+uinf[0];
  const Real dvdx  = u > 0 ?           3*vpx + 3*vcc - 7*vlx + vllx
                             : -vppx + 7*vpx - 3*vcc - 3*vlx        ;
  const Real v = vcc+uinf[1];
  const Real dvdy  = v > 0 ?           3*vpy + 3*vcc - 7*vly + vlly
                             : -vppy + 7*vpy - 3*vcc - 3*vly        ;
  const Real dVadv = u * 0.125 * dvdx + v * 0.125 * dvdy;

  // diffusion
  const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;

  return advF * dVadv + difF * dVdif;
}
#endif

void advDiff::operator()(const double dt)
{
  sim.startProfiler("advDiff");
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  const size_t Nblocks = velInfo.size();
  const Real UINF[2]= {sim.uinfx, sim.uinfy};
  const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
  const Real fadeW= 1 - std::pow(std::max(UINF[0], (Real)0)/norUinf, 2);
  const Real fadeS= 1 - std::pow(std::max(UINF[1], (Real)0)/norUinf, 2);
  const Real fadeE= 1 - std::pow(std::min(UINF[0], (Real)0)/norUinf, 2);
  const Real fadeN= 1 - std::pow(std::min(UINF[1], (Real)0)/norUinf, 2);

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-2,-2, 0}, stenEnd[3] = { 3, 3, 1};
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);

    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
      int aux = 1<<velInfo[i].level;
      const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0;          };
      const auto isE = [&](const BlockInfo&I) { return I.index[0] == aux*sim.bpdx-1; };
      const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0;          };
      const auto isN = [&](const BlockInfo&I) { return I.index[1] == aux*sim.bpdy-1; };
      const Real h = velInfo[i].h;
      const Real dfac = (sim.nu/h)*(dt/h);
      const Real afac = -dt/h/6.0;
      vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      if(isW(velInfo[i])) 
        for(int iy=-2; iy<VectorBlock::sizeY+3; ++iy)
        for(int ix=-2; ix<0                   ; ++ix)
        {
          V(ix,iy).u[0] *= fadeW;
          V(ix,iy).u[1] *= fadeW;
        }

      if(isS(velInfo[i]))
        for(int iy=-2; iy<0                   ; ++iy) 
        for(int ix=-2; ix<VectorBlock::sizeX+3; ++ix) 
        {
          V(ix,iy).u[0] *= fadeS;
          V(ix,iy).u[1] *= fadeS;
        }

      if(isE(velInfo[i]))
        for(int iy=-2; iy<VectorBlock::sizeY+3; ++iy) 
        for(int ix=VectorBlock::sizeX; ix<VectorBlock::sizeX+3; ++ix) 
        {
          V(ix,iy).u[0] *= fadeE;
          V(ix,iy).u[1] *= fadeE;
        }

      if(isN(velInfo[i]))
        for(int iy=VectorBlock::sizeY; iy<VectorBlock::sizeY+3; ++iy) 
        for(int ix=-2; ix<VectorBlock::sizeX+3; ++ix)
        {
          V(ix,iy).u[0] *= fadeN;
          V(ix,iy).u[1] *= fadeN;
        }

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        TMP(ix,iy).u[0] = V(ix,iy).u[0] + dU_adv_dif(V,UINF,afac,dfac,ix,iy);
        TMP(ix,iy).u[1] = V(ix,iy).u[1] + dV_adv_dif(V,UINF,afac,dfac,ix,iy);
      }
    }
  }

  // Copy TMP to V and store inflow correction
  Real IF = 0.0;
  #pragma omp parallel for reduction(+ : IF)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    V.copy(T);
    const int aux = 1<<velInfo[i].level;
    const bool isW = velInfo[i].index[0] == 0;
    const bool isE = velInfo[i].index[0] == aux*sim.bpdx-1;
    const bool isS = velInfo[i].index[1] == 0;
    const bool isN = velInfo[i].index[1] == aux*sim.bpdy-1;
    const double h = velInfo[i].h;
    if (isW) for(int iy=0; iy<VectorBlock::sizeY; ++iy) IF -= h * V(0,iy).u[0];
    if (isS) for(int ix=0; ix<VectorBlock::sizeX; ++ix) IF -= h * V(ix,0).u[1];
    if (isE) for(int iy=0; iy<VectorBlock::sizeY; ++iy) IF += h * V(VectorBlock::sizeX-1,iy).u[0];
    if (isN) for(int ix=0; ix<VectorBlock::sizeX; ++ix) IF += h * V(ix,VectorBlock::sizeY-1).u[1];
  }

  const double H = sim.getH();//returns smallest grid spacing, at finest refinement level
  const Real corr = IF/H/( 2*VectorBlock::sizeY*sim.bpdy*(1<<(sim.levelMax-1)) 
                         + 2*VectorBlock::sizeX*sim.bpdx*(1<<(sim.levelMax-1)) );

  // Apply correction
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    int aux = 1<<velInfo[i].level;
    const bool isW = velInfo[i].index[0] == 0;
    const bool isE = velInfo[i].index[0] == aux*sim.bpdx-1;
    const bool isS = velInfo[i].index[1] == 0;
    const bool isN = velInfo[i].index[1] == aux*sim.bpdy-1;
    VectorBlock& V = *(VectorBlock*) velInfo[i].ptrBlock;
    if(isW) for(int iy=0; iy<VectorBlock::sizeY; ++iy) V(0,iy).u[0] += corr;
    if(isS) for(int ix=0; ix<VectorBlock::sizeX; ++ix) V(ix,0).u[1] += corr;
    if(isE) for(int iy=0; iy<VectorBlock::sizeY; ++iy) V(VectorBlock::sizeX-1,iy).u[0] -= corr;
    if(isN) for(int ix=0; ix<VectorBlock::sizeX; ++ix) V(ix,VectorBlock::sizeY-1).u[1] -= corr;
  }
  sim.stopProfiler();
}