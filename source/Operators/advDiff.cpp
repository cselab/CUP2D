//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
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


void advDiff::step(const int coef)
{
  //For a given velocity V and Vold, compute V = Vold + RHS(V)

  const size_t Nblocks = velInfo.size();
  const Real UINF[2]= {sim.uinfx, sim.uinfy};

  FluxCorrection<VectorGrid,VectorBlock> Corrector;
  Corrector.prepare(*(sim.tmpV));
  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-2,-2, 0}, stenEnd[3] = { 3, 3, 1};
    VectorLab V; 
    V.prepare(*(sim.vel), stenBeg, stenEnd, 0);
    const Real dfac = sim.nu*sim.dt;

    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
      const Real h = velInfo[i].h;
      const Real afac = -sim.dt*h/6.0;
      V.load(velInfo[i], 0);
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        TMP(ix,iy).u[0] = coef*dU_adv_dif(V,UINF,afac,dfac,ix,iy);
        TMP(ix,iy).u[1] = coef*dV_adv_dif(V,UINF,afac,dfac,ix,iy);
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
          faceXm[iy].u[0] = dfac*( coef*(V(ix,iy).u[0] - V(ix-1,iy).u[0]));
          faceXm[iy].u[1] = dfac*( coef*(V(ix,iy).u[1] - V(ix-1,iy).u[1]));
        }
      }
      if (faceXp != nullptr)
      {
        int ix = VectorBlock::sizeX-1;
        for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        {
          faceXp[iy].u[0] = dfac*( coef*(V(ix,iy).u[0] - V(ix+1,iy).u[0]));
          faceXp[iy].u[1] = dfac*( coef*(V(ix,iy).u[1] - V(ix+1,iy).u[1]));
        }
      }
      if (faceYm != nullptr)
      {
        int iy = 0;
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          faceYm[ix].u[0] = dfac*( coef*(V(ix,iy).u[0] - V(ix,iy-1).u[0]));
          faceYm[ix].u[1] = dfac*( coef*(V(ix,iy).u[1] - V(ix,iy-1).u[1]));
        }
      }
      if (faceYp != nullptr)
      {
        int iy = VectorBlock::sizeY-1;
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          faceYp[ix].u[0] = dfac*( coef*(V(ix,iy).u[0] - V(ix,iy+1).u[0]));
          faceYp[ix].u[1] = dfac*( coef*(V(ix,iy).u[1] - V(ix,iy+1).u[1]));
        }
      }
    }
  }
  Corrector.FillBlockCases();

  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ Vold = *(VectorBlock*) vOldInfo[i].ptrBlock;
    const VectorBlock & __restrict__ tmpV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    const double ih2 = 1.0/(velInfo[i].h*velInfo[i].h);
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      V(ix,iy).u[0] = Vold(ix,iy).u[0] + tmpV(ix,iy).u[0]*ih2;
      V(ix,iy).u[1] = Vold(ix,iy).u[1] + tmpV(ix,iy).u[1]*ih2;
    }
  }
}

void advDiff::operator()(const double dt)
{
  sim.startProfiler("advDiff");

  //1.Save u^{n} to dataOld
  #pragma omp parallel for
  for (size_t i=0; i < velInfo.size(); i++)
  {
    VectorBlock & __restrict__ Vold  = *(VectorBlock*) vOldInfo[i].ptrBlock;
    const VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      Vold(ix,iy).u[0] = V(ix,iy).u[0];
      Vold(ix,iy).u[1] = V(ix,iy).u[1];
    }
  }

  // 2. Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
  step(0.5);

  // 3. Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
  step(1.0);

  sim.stopProfiler();
}
