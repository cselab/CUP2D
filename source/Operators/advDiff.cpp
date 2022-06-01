//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "advDiff.h"

using namespace cubism;

#ifdef CUP2D_PRESERVE_SYMMETRY
#define CUP2D_DISABLE_OPTIMIZATIONS __attribute__((optimize("-O1")))
#else
#define CUP2D_DISABLE_OPTIMIZATIONS
#endif

CUP2D_DISABLE_OPTIMIZATIONS
static inline Real weno5_plus(const Real um2, const Real um1, const Real u, const Real up1, const Real up2)
{
  const Real exponent = 2;
  const Real e = 1e-6;
  const Real b1 = 13.0/12.0*pow((um2+u)-2*um1,2)+0.25*pow((um2+3*u)-4*um1,2);
  const Real b2 = 13.0/12.0*pow((um1+up1)-2*u,2)+0.25*pow(um1-up1,2);
  const Real b3 = 13.0/12.0*pow((u+up2)-2*up1,2)+0.25*pow((3*u+up2)-4*up1,2);
  const Real g1 = 0.1;
  const Real g2 = 0.6;
  const Real g3 = 0.3;
  const Real what1 = g1/pow(b1+e,exponent);
  const Real what2 = g2/pow(b2+e,exponent);
  const Real what3 = g3/pow(b3+e,exponent);
  const Real aux = 1.0/((what1+what3)+what2);
  const Real w1 = what1*aux;
  const Real w2 = what2*aux;
  const Real w3 = what3*aux;
  const Real f1 = (11.0/6.0)*u + ( ( 1.0/3.0)*um2- (7.0/6.0)*um1);
  const Real f2 = (5.0 /6.0)*u + ( (-1.0/6.0)*um1+ (1.0/3.0)*up1);
  const Real f3 = (1.0 /3.0)*u + ( (+5.0/6.0)*up1- (1.0/6.0)*up2);
  return (w1*f1+w3*f3)+w2*f2;
}

CUP2D_DISABLE_OPTIMIZATIONS
static inline Real weno5_minus(const Real um2, const Real um1, const Real u, const Real up1, const Real up2)
{
  const Real exponent = 2;
  const Real e = 1e-6;
  const Real b1 = 13.0/12.0*pow((um2+u)-2*um1,2)+0.25*pow((um2+3*u)-4*um1,2);
  const Real b2 = 13.0/12.0*pow((um1+up1)-2*u,2)+0.25*pow(um1-up1,2);
  const Real b3 = 13.0/12.0*pow((u+up2)-2*up1,2)+0.25*pow((3*u+up2)-4*up1,2);
  const Real g1 = 0.3;
  const Real g2 = 0.6;
  const Real g3 = 0.1;
  const Real what1 = g1/pow(b1+e,exponent);
  const Real what2 = g2/pow(b2+e,exponent);
  const Real what3 = g3/pow(b3+e,exponent);
  const Real aux = 1.0/((what1+what3)+what2);
  const Real w1 = what1*aux;
  const Real w2 = what2*aux;
  const Real w3 = what3*aux;
  const Real f1 = ( 1.0/3.0)*u + ( (-1.0/6.0)*um2+ (5.0/6.0)*um1);
  const Real f2 = ( 5.0/6.0)*u + ( ( 1.0/3.0)*um1- (1.0/6.0)*up1);
  const Real f3 = (11.0/6.0)*u + ( (-7.0/6.0)*up1+ (1.0/3.0)*up2);
  return (w1*f1+w3*f3)+w2*f2;
}

static inline Real derivative(const Real U, const Real um3, const Real um2, const Real um1,
                                            const Real u  ,
                                            const Real up1, const Real up2, const Real up3)
{
  Real fp = 0.0;
  Real fm = 0.0;
  if (U > 0)
  {
    fp = weno5_plus (um2,um1,u,up1,up2);
    fm = weno5_plus (um3,um2,um1,u,up1);
  }
  else
  {
    fp = weno5_minus(um1,u,up1,up2,up3);
    fm = weno5_minus(um2,um1,u,up1,up2);
  }
  return (fp-fm);
}

static inline Real dU_adv_dif(const VectorLab&V, const Real uinf[2], const Real advF, const Real difF, const int ix, const int iy)
{
  const Real u    = V(ix,iy).u[0];
  const Real v    = V(ix,iy).u[1];
  const Real UU   = u + uinf[0];
  const Real VV   = v + uinf[1];

  const Real up1x = V(ix+1,iy).u[0];
  const Real up2x = V(ix+2,iy).u[0];
  const Real up3x = V(ix+3,iy).u[0];
  const Real um1x = V(ix-1,iy).u[0];
  const Real um2x = V(ix-2,iy).u[0];
  const Real um3x = V(ix-3,iy).u[0];

  const Real up1y = V(ix,iy+1).u[0];
  const Real up2y = V(ix,iy+2).u[0];
  const Real up3y = V(ix,iy+3).u[0];
  const Real um1y = V(ix,iy-1).u[0];
  const Real um2y = V(ix,iy-2).u[0];
  const Real um3y = V(ix,iy-3).u[0];
  
  const Real dudx = derivative(UU,um3x,um2x,um1x,u,up1x,up2x,up3x);
  const Real dudy = derivative(VV,um3y,um2y,um1y,u,up1y,up2y,up3y);

  return advF*(UU*dudx+VV*dudy) + difF*( ((up1x + um1x) + (up1y  + um1y)) - 4*u);
}
  
static inline Real dV_adv_dif(const VectorLab&V, const Real uinf[2], const Real advF, const Real difF, const int ix, const int iy)
{
  const Real u    = V(ix,iy).u[0];
  const Real v    = V(ix,iy).u[1];
  const Real UU   = u + uinf[0];
  const Real VV   = v + uinf[1];

  const Real vp1x = V(ix+1,iy).u[1];
  const Real vp2x = V(ix+2,iy).u[1];
  const Real vp3x = V(ix+3,iy).u[1];
  const Real vm1x = V(ix-1,iy).u[1];
  const Real vm2x = V(ix-2,iy).u[1];
  const Real vm3x = V(ix-3,iy).u[1];

  const Real vp1y = V(ix,iy+1).u[1];
  const Real vp2y = V(ix,iy+2).u[1];
  const Real vp3y = V(ix,iy+3).u[1];
  const Real vm1y = V(ix,iy-1).u[1];
  const Real vm2y = V(ix,iy-2).u[1];
  const Real vm3y = V(ix,iy-3).u[1];

  const Real dvdx = derivative(UU,vm3x,vm2x,vm1x,v,vp1x,vp2x,vp3x);
  const Real dvdy = derivative(VV,vm3y,vm2y,vm1y,v,vp1y,vp2y,vp3y);

  return advF*(UU*dvdx+VV*dvdy) + difF*( ((vp1x+ vm1x) + (vp1y+ vm1y)) - 4*v);
}


struct KernelAdvectDiffuse
{
  KernelAdvectDiffuse(const SimulationData & s) : sim(s)
  {
    uinf[0] = sim.uinfx;
    uinf[1] = sim.uinfy;
  }
  const SimulationData & sim;
  Real uinf [2];
  const StencilInfo stencil{-3, -3, 0, 4, 4, 1, true, {0,1}};
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  void operator()(VectorLab& lab, const BlockInfo& info) const
  {
    const Real h = info.h;
    const Real dfac = sim.nu*sim.dt;
    const Real afac = -sim.dt*h;
    VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      TMP(ix,iy).u[0] = dU_adv_dif(lab,uinf,afac,dfac,ix,iy);
      TMP(ix,iy).u[1] = dV_adv_dif(lab,uinf,afac,dfac,ix,iy);
    }
    BlockCase<VectorBlock> * tempCase = (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    VectorBlock::ElementType * faceXm = nullptr;
    VectorBlock::ElementType * faceXp = nullptr;
    VectorBlock::ElementType * faceYm = nullptr;
    VectorBlock::ElementType * faceYp = nullptr;

    const Real aux_coef = dfac;

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
        faceXm[iy].u[0] = aux_coef*(lab(ix,iy).u[0] - lab(ix-1,iy).u[0]);
        faceXm[iy].u[1] = aux_coef*(lab(ix,iy).u[1] - lab(ix-1,iy).u[1]);
      }
    }
    if (faceXp != nullptr)
    {
      int ix = VectorBlock::sizeX-1;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXp[iy].u[0] = aux_coef*(lab(ix,iy).u[0] - lab(ix+1,iy).u[0]);
        faceXp[iy].u[1] = aux_coef*(lab(ix,iy).u[1] - lab(ix+1,iy).u[1]);
      }
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYm[ix].u[0] = aux_coef*(lab(ix,iy).u[0] - lab(ix,iy-1).u[0]);
        faceYm[ix].u[1] = aux_coef*(lab(ix,iy).u[1] - lab(ix,iy-1).u[1]);
      }
    }
    if (faceYp != nullptr)
    {
      int iy = VectorBlock::sizeY-1;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYp[ix].u[0] = aux_coef*(lab(ix,iy).u[0] - lab(ix,iy+1).u[0]);
        faceYp[ix].u[1] = aux_coef*(lab(ix,iy).u[1] - lab(ix,iy+1).u[1]);
      }
    }
  }
};


void advDiff::operator()(const Real dt)
{
  sim.startProfiler("advDiff");
  const size_t Nblocks = velInfo.size();
  KernelAdvectDiffuse Step1(sim) ;

  //1.Save u^{n} to dataOld
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
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

  /********************************************************************/
  // 2. Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
  //   2a) Compute 0.5*dt*RHS(u^{n}) and store it to tmpU,tmpV,tmpW
  cubism::compute<VectorLab>(Step1,sim.vel,sim.tmpV);

  //   2b) Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ Vold = *(VectorBlock*) vOldInfo[i].ptrBlock;
    const VectorBlock & __restrict__ tmpV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    const Real ih2 = 1.0/(velInfo[i].h*velInfo[i].h);
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      V(ix,iy).u[0] = Vold(ix,iy).u[0] + (0.5*tmpV(ix,iy).u[0])*ih2;
      V(ix,iy).u[1] = Vold(ix,iy).u[1] + (0.5*tmpV(ix,iy).u[1])*ih2;
    }
  }
  /********************************************************************/

  /********************************************************************/
  // 3. Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
  //   3a) Compute dt*RHS(u^{n+1/2}) and store it to tmpU,tmpV,tmpW
  cubism::compute<VectorLab>(Step1,sim.vel,sim.tmpV);
  //   3b) Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ Vold = *(VectorBlock*) vOldInfo[i].ptrBlock;
    const VectorBlock & __restrict__ tmpV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    const Real ih2 = 1.0/(velInfo[i].h*velInfo[i].h);
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      V(ix,iy).u[0] = Vold(ix,iy).u[0] + tmpV(ix,iy).u[0]*ih2;
      V(ix,iy).u[1] = Vold(ix,iy).u[1] + tmpV(ix,iy).u[1]*ih2;
    }
  }
  /********************************************************************/

  sim.stopProfiler();
}
