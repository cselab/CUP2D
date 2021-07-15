//
//  CubismUP_2D - AMR
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
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

  const Real up1y = V(ix,iy+1).u[0];
  const Real up2y = V(ix,iy+2).u[0];
  const Real um1y = V(ix,iy-1).u[0];
  const Real um2y = V(ix,iy-2).u[0];

  const Real dudx = UU>0 ? (2*up1x + 3*u - 6*um1x + um2x) : (-up2x + 6*up1x - 3*u - 2*um1x);
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

  const Real vp1y = V(ix,iy+1).u[1];
  const Real vp2y = V(ix,iy+2).u[1];
  const Real vm1y = V(ix,iy-1).u[1];
  const Real vm2y = V(ix,iy-2).u[1];

  const Real dvdx = UU>0 ? (2*vp1x + 3*v - 6*vm1x + vm2x) : (-vp2x + 6*vp1x - 3*v - 2*vm1x);
  const Real dvdy = VV>0 ? (2*vp1y + 3*v - 6*vm1y + vm2y) : (-vp2y + 6*vp1y - 3*v - 2*vm1y);

  return advF*(UU*dvdx+VV*dvdy) + difF*(vp1x + vp1y + vm1x + vm1y - 4*v);
}

void advDiff::operator()(const double dt)
{
  const double c1 = (sim.Euler || sim.step < 3) ? 1.0 :      (sim.dt_old+0.5*sim.dt)/sim.dt;// 1.5;
  const double c2 = (sim.Euler || sim.step < 3) ? 0.0 : (1.0-(sim.dt_old+0.5*sim.dt)/sim.dt);//-0.5;

  sim.startProfiler("advDiff");
  const size_t Nblocks = velInfo.size();
  const Real UINF[2]= {sim.uinfx, sim.uinfy};
  const Real UINFOLD[2]= {sim.uinfx_old, sim.uinfy_old};

  FluxCorrection<VectorGrid,VectorBlock> Corrector;
  Corrector.prepare(*(sim.tmpV));
  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-2,-2, 0}, stenEnd[3] = { 3, 3, 1};
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 0);
    VectorLab vOldlab; vOldlab.prepare(*(sim.vOld), stenBeg, stenEnd, 0);

    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
      const Real h = velInfo[i].h;
      const Real dfac = sim.nu*dt;
      const Real afac = -dt*h/6.0;
      vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
      vOldlab.load(vOldInfo[i], 0);
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        TMP(ix,iy).u[0] = c1*dU_adv_dif(V,UINF,afac,dfac,ix,iy) + c2*dU_adv_dif(vOldlab,UINFOLD,afac,dfac,ix,iy);
        TMP(ix,iy).u[1] = c1*dV_adv_dif(V,UINF,afac,dfac,ix,iy) + c2*dV_adv_dif(vOldlab,UINFOLD,afac,dfac,ix,iy);
      }

      BlockCase<VectorBlock> * tempCase = (BlockCase<VectorBlock> *)(tmpVInfo[i].auxiliary);
      VectorBlock::ElementType * faceXm = nullptr;
      VectorBlock::ElementType * faceXp = nullptr;
      VectorBlock::ElementType * faceYm = nullptr;
      VectorBlock::ElementType * faceYp = nullptr;
      if (tempCase != nullptr)
      {
        faceXm = tempCase -> storedFace[0] ?  & tempCase -> m_pData[0][0] : nullptr;
        faceXp = tempCase -> storedFace[1] ?  & tempCase -> m_pData[1][0] : nullptr;
        faceYm = tempCase -> storedFace[2] ?  & tempCase -> m_pData[2][0] : nullptr;
        faceYp = tempCase -> storedFace[3] ?  & tempCase -> m_pData[3][0] : nullptr;
      }
      if (faceXm != nullptr)
      {
        int ix = 0;
        for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        {
          faceXm[iy].u[0] = dfac*( c1*(V(ix,iy).u[0] - V(ix-1,iy).u[0]) + c2*(vOldlab(ix,iy).u[0] - vOldlab(ix-1,iy).u[0]));
          faceXm[iy].u[1] = dfac*( c1*(V(ix,iy).u[1] - V(ix-1,iy).u[1]) + c2*(vOldlab(ix,iy).u[1] - vOldlab(ix-1,iy).u[1]));
        }
      }
      if (faceXp != nullptr)
      {
        int ix = VectorBlock::sizeX-1;
        for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        {
          faceXp[iy].u[0] = dfac*( c1*(V(ix,iy).u[0] - V(ix+1,iy).u[0]) + c2*(vOldlab(ix,iy).u[0] - vOldlab(ix+1,iy).u[0]));
          faceXp[iy].u[1] = dfac*( c1*(V(ix,iy).u[1] - V(ix+1,iy).u[1]) + c2*(vOldlab(ix,iy).u[1] - vOldlab(ix+1,iy).u[1]));
        }
      }
      if (faceYm != nullptr)
      {
        int iy = 0;
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          faceYm[ix].u[0] = dfac*( c1*(V(ix,iy).u[0] - V(ix,iy-1).u[0]) + c2*(vOldlab(ix,iy).u[0] - vOldlab(ix,iy-1).u[0]));
          faceYm[ix].u[1] = dfac*( c1*(V(ix,iy).u[1] - V(ix,iy-1).u[1]) + c2*(vOldlab(ix,iy).u[1] - vOldlab(ix,iy-1).u[1]));
        }
      }
      if (faceYp != nullptr)
      {
        int iy = VectorBlock::sizeY-1;
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          faceYp[ix].u[0] = dfac*( c1*(V(ix,iy).u[0] - V(ix,iy+1).u[0]) + c2*(vOldlab(ix,iy).u[0] - vOldlab(ix,iy+1).u[0]));
          faceYp[ix].u[1] = dfac*( c1*(V(ix,iy).u[1] - V(ix,iy+1).u[1]) + c2*(vOldlab(ix,iy).u[1] - vOldlab(ix,iy+1).u[1]));
        }
      }
    }
  }
  Corrector.FillBlockCases();

  // Copy TMP to V and store inflow correction
  Real IF = 0.0;
  #pragma omp parallel for reduction(+ : IF)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    const double ih2 = 1.0/velInfo[i].h/velInfo[i].h;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      V(ix,iy).u[0] += T(ix,iy).u[0]*ih2;
      V(ix,iy).u[1] += T(ix,iy).u[1]*ih2;
    }
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

  const double H = sim.minH;
  const Real corr = IF/H/( 2*VectorBlock::sizeY*sim.bpdy*(1<<(sim.levelMax-1)) 
                         // + 2*VectorBlock::sizeX*sim.bpdx*(1<<(sim.levelMax-1)) );
  // Apply correction
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    const int level = velInfo[i].level;
    const int aux = 1<<level;
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

  #pragma omp parallel for 
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    VectorBlock & __restrict__ Vold  = *(VectorBlock*)  vOldInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      Vold(ix,iy).u[0] = V(ix,iy).u[0];
      Vold(ix,iy).u[1] = V(ix,iy).u[1];
    }
  }

  sim.stopProfiler();
}
