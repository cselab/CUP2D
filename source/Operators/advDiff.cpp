//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "advDiff.h"

using namespace cubism;

static inline Real weno5_plus(const Real um2, const Real um1, const Real u, const Real up1, const Real up2)
{
  const double exponent = 2;
  const double e = 1e-6;
  const double b1 = 13.0/12.0*pow(um2-2*um1+u,2)+0.25*pow(um2-4*um1+3*u,2);
  const double b2 = 13.0/12.0*pow(um1-2*u+up1,2)+0.25*pow(um1-up1,2);
  const double b3 = 13.0/12.0*pow(u-2*up1+up2,2)+0.25*pow(3*u-4*up1+up2,2);
  const double g1 = 0.1;
  const double g2 = 0.6;
  const double g3 = 0.3;
  const double what1 = g1/pow(b1+e,exponent);
  const double what2 = g2/pow(b2+e,exponent);
  const double what3 = g3/pow(b3+e,exponent);
  const double w1 = what1/(what1+what2+what3);
  const double w2 = what2/(what1+what2+what3);
  const double w3 = what3/(what1+what2+what3);
  const double f1 =  1.0/3.0*um2-7.0/6.0*um1+11.0/6.0*u;
  const double f2 = -1.0/6.0*um1+5.0/6.0*u  + 1.0/3.0*up1;
  const double f3 =  1.0/3.0*u  +5.0/6.0*up1- 1.0/6.0*up2;
  return w1*f1+w2*f2+w3*f3;
}

static inline Real weno5_minus(const Real um2, const Real um1, const Real u, const Real up1, const Real up2)
{
  const double exponent = 2;
  const double e = 1e-6;
  const double b1 = 13.0/12.0*pow(um2-2*um1+u,2)+0.25*pow(um2-4*um1+3*u,2);
  const double b2 = 13.0/12.0*pow(um1-2*u+up1,2)+0.25*pow(um1-up1,2);
  const double b3 = 13.0/12.0*pow(u-2*up1+up2,2)+0.25*pow(3*u-4*up1+up2,2);
  const double g1 = 0.3;
  const double g2 = 0.6;
  const double g3 = 0.1;
  const double what1 = g1/pow(b1+e,exponent);
  const double what2 = g2/pow(b2+e,exponent);
  const double what3 = g3/pow(b3+e,exponent);
  const double w1 = what1/(what1+what2+what3);
  const double w2 = what2/(what1+what2+what3);
  const double w3 = what3/(what1+what2+what3);
  const double f1 = -1.0/6.0*um2+5.0/6.0*um1+ 1.0/3.0*u;
  const double f2 =  1.0/3.0*um1+5.0/6.0*u  - 1.0/6.0*up1;
  const double f3 = 11.0/6.0*u  -7.0/6.0*up1+ 1.0/3.0*up2;
  return w1*f1+w2*f2+w3*f3;
}

static inline Real derivative(const Real U, const Real um3, const Real um2, const Real um1,
                                            const Real u  ,
                                            const Real up1, const Real up2, const Real up3)
{
  double fp = 0.0;
  double fm = 0.0;
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

  return advF*(UU*dvdx+VV*dvdy) + difF*(vp1x + vp1y + vm1x + vm1y - 4*v);
}


struct KernelAdvectDiffuse
{
  KernelAdvectDiffuse(const SimulationData & s, const double c, const double uinfx, const double uinfy) : sim(s),coef(c)
  {
    uinf[0] = uinfx;
    uinf[1] = uinfy;
  }
  const SimulationData & sim;
  const double coef;
  double uinf [2];
  const StencilInfo stencil{-3, -3, 0, 4, 4, 1, false, {0,1}};
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  void operator()(VectorLab & lab, const BlockInfo& info) const
  {
    const double h = info.h;
    const double dfac = sim.nu*sim.dt;
    const double afac = -sim.dt*h;
    VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      TMP(ix,iy).u[0] += coef*dU_adv_dif(lab,uinf,afac,dfac,ix,iy);
      TMP(ix,iy).u[1] += coef*dV_adv_dif(lab,uinf,afac,dfac,ix,iy);
    }
    BlockCase<VectorBlock> * tempCase = (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
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
        faceXm[iy].u[0] = dfac*coef*(lab(ix,iy).u[0] - lab(ix-1,iy).u[0]);
        faceXm[iy].u[1] = dfac*coef*(lab(ix,iy).u[1] - lab(ix-1,iy).u[1]);
      }
    }
    if (faceXp != nullptr)
    {
      int ix = VectorBlock::sizeX-1;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXp[iy].u[0] = dfac*coef*(lab(ix,iy).u[0] - lab(ix+1,iy).u[0]);
        faceXp[iy].u[1] = dfac*coef*(lab(ix,iy).u[1] - lab(ix+1,iy).u[1]);
      }
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYm[ix].u[0] = dfac*coef*(lab(ix,iy).u[0] - lab(ix,iy-1).u[0]);
        faceYm[ix].u[1] = dfac*coef*(lab(ix,iy).u[1] - lab(ix,iy-1).u[1]);
      }
    }
    if (faceYp != nullptr)
    {
      int iy = VectorBlock::sizeY-1;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYp[ix].u[0] = dfac*coef*(lab(ix,iy).u[0] - lab(ix,iy+1).u[0]);
        faceYp[ix].u[1] = dfac*coef*(lab(ix,iy).u[1] - lab(ix,iy+1).u[1]);
      }
    }
  }
};

void advDiff::operator()(const double dt)
{
  const double c1 = (sim.Euler || sim.step < 3) ? 1.0 :      (sim.dt_old+0.5*sim.dt)/sim.dt;// 1.5;
  const double c2 = (sim.Euler || sim.step < 3) ? 0.0 : (1.0-(sim.dt_old+0.5*sim.dt)/sim.dt);//-0.5;

  sim.startProfiler("advDiff");
  const size_t Nblocks = velInfo.size();
  const Real UINF[2]= {sim.uinfx, sim.uinfy};
  const Real UINFOLD[2]= {sim.uinfx_old, sim.uinfy_old};
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    TMP.clear();
  }

  KernelAdvectDiffuse Step1(sim,c1,UINF   [0],UINF   [1]) ;
  KernelAdvectDiffuse Step2(sim,c2,UINFOLD[0],UINFOLD[1]) ;
  compute<KernelAdvectDiffuse,VectorGrid,VectorLab,VectorGrid>(Step1,*sim.vel ,true,sim.tmpV);
  compute<KernelAdvectDiffuse,VectorGrid,VectorLab,VectorGrid>(Step2,*sim.vOld,true,sim.tmpV);

  // Copy TMP to V
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    VectorBlock & __restrict__ Vold  = *(VectorBlock*) vOldInfo[i].ptrBlock;
    const double ih2 = 1.0/velInfo[i].h/velInfo[i].h;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      V(ix,iy).u[0] += T(ix,iy).u[0]*ih2;
      V(ix,iy).u[1] += T(ix,iy).u[1]*ih2;
      Vold(ix,iy).u[0] = V(ix,iy).u[0];
      Vold(ix,iy).u[1] = V(ix,iy).u[1];
    }
  }
  sim.stopProfiler();
}
