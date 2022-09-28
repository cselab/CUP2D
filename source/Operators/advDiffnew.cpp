//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "advDiff.h"

using namespace cubism;
namespace cubism {

template <typename Lab1,typename Lab2, typename Kernel, typename TGrid1, typename TGrid2>
void compute2(Kernel &&kernel, TGrid1 *g, TGrid2 *f)
{
   	cubism::SynchronizerMPI_AMR<typename TGrid1::Real,TGrid1>& Synch1 = *(g->sync(kernel));
	cubism::SynchronizerMPI_AMR<typename TGrid2::Real,TGrid2>& Synch2 = *(f->sync(kernel));
    std::vector<cubism::BlockInfo*> *inner1 = &Synch1.avail_inner();
	std::vector<cubism::BlockInfo*> *inner2 = &Synch2.avail_inner();
    std::vector<cubism::BlockInfo*> *halo1 = &Synch1.avail_halo();
	std::vector<cubism::BlockInfo*> *halo2 = &Synch2.avail_halo();
    #pragma omp parallel
    {
        Lab1 lab1;
        lab1.prepare(*g, Synch1);

		Lab2 lab2;
		lab2.prepare(*f,Synch2);

        #pragma omp for nowait
		for(size_t i=0;i<inner1->size();i++){
			const cubism::BlockInfo *I1=inner1->at(i);
			const cubism::BlockInfo *I2=inner2->at(i);
			lab1.load(*I1,0);
			lab2.load(*I2,0);
			kernel(lab1,*I1,lab2,*I2);
		}

        #pragma omp barrier

        #pragma omp for nowait
        for (size_t i=0;i<halo1->size();i++)
        {
          const cubism::BlockInfo *I1=halo1->at(i);
		  const cubism::BlockInfo *I2=halo2->at(i);
		  lab1.load(*I1, 0);
		  lab2.load(*I2, 0);
          kernel(lab1, *I1,lab2,*I2);
        }
    }
}
}
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
static inline void SolidDstress(const VectorLab&INVM,const int ix,const int iy,const Real h,Real& stress[4])
{
  const Real i2h=1.0/h/2.0;
  const Real up1x=INVM(ix+1,iy).u[0];
  const Real um1x=INVM(ix-1,iy).u[0];
  const Real up1y=INVM(ix,iy+1).u[0];
  const Real um1y=INVM(ix,iy-1).u[0];
  
  const Real vp1x=INVM(ix+1,iy).u[1];
  const Real vm1x=INVM(ix-1,iy).u[1];
  const Real vp1y=INVM(ix,iy+1).u[1];
  const Real vm1y=INVM(ix,iy-1).u[1];
  
  const Real dudx=(up1x-um1x)*i2h;
  const Real dudy=(up1y-um1y)*i2h;
  const Real dvdx=(vp1x-vm1x)*i2h;
  const Real dvdy=(vp1y-vm1y)*i2h;
  Real denominator=(dvdy*dudx-dudy*dvdx)*(dvdy*dudx-dudy*dvdx);
  if (std::fabs(denominator)<1e-7) std::cout<<"warning======irreversible F\n";
  denominator=1.0/denominator;
  stress[0]=( dudy*dudy+dvdy*dvdy);
  stress[1]=(-dudy*dudx-dvdy*dvdx);
  stress[2]=(-dudy*dudx-dvdx*dvdy);
  stress[3]=( dudx*dudx+dvdx*dvdx);
  Real tr=0.5*(stress[0]+stress[3]);
  stress[0]-=0.5*tr;
  stress[3]-=0.5*tr;
}
/*static inline void DivSolidDstress(const VectorLab&INVM,const int ix,const int iy,const Real h,Real &divS[2])
{
  const Real ih2=1.0/h/h;
  const Real i2h=1.0/h/2.0;
  const Real u=INVM(ix,iy).u[0];
  const Real up1x=INVM(ix+1,iy).u[0];
  const Real up1xp1y=INVM(ix+1,iy+1).u[0];
  const Real um1x=INVM(ix-1,iy).u[0];
  const Real up1xm1y=INVM(ix+1,iy-1).u[0];
  const Real up1y=INVM(ix,iy+1).u[0];
  const Real um1xp1y=INVM(ix-1,iy+1).u[0];
  const Real um1y=INVM(ix,iy-1).u[0];
  const Real um1xm1y=INVM(ix-1,iy-1).u[0];
  
  const Real v=INVM(ix,iy).u[1];
  const Real vp1x=INVM(ix+1,iy).u[1];
  const Real vp1xp1y=INVM(ix+1,iy+1).u[1];
  const Real vm1x=INVM(ix-1,iy).u[1];
  const Real vp1xm1y=INVM(ix+1,iy-1).u[1];
  const Real vp1y=INVM(ix,iy+1).u[1];
  const Real vm1xp1y=INVM(ix-1,iy+1).u[1];
  const Real vm1y=INVM(ix,iy-1).u[1];
  const Real vm1xm1y=INVM(ix-1,iy-1).u[1];
  
  const Real d2udx2=(up1x+um1x-2*u)*ih2;
  const Real d2udy2=(up1y+um1y-2*u)*ih2;
  const Real d2vdx2=(vp1x+vm1x-2*v)*ih2;
  const Real d2vdy2=(vp1y+vm1y-2*v)*ih2;
  const Real d2udxdy=(up1xp1y+um1xm1y-up1xm1y-um1xp1y)*ih2/4.0;
  const Real d2vdxdy=(vp1xp1y+vm1xm1y-vp1xm1y-vm1xp1y)*ih2/4.0;
  const Real dudx=(up1x-um1x)*i2h;
  const Real dudy=(up1y-um1y)*i2h;
  const Real dvdx=(vp1x-vm1x)*i2h;
  const Real dvdy=(vp1y-vm1y)*i2h;
  Real denominator2=(dvdy*dudx-dudy*dvdx)*(dvdy*dudx-dudy*dvdx),denominator3=(dvdy*dudx-dudy*dvdx)*denominator2;
  const Real a0=2*(dudy*dudx+dvdy*dvdx),
             a1=d2vdy2*dudx-d2udy2*dvdx+dvdy*d2udxdy-dudy*d2vdxdy,
             a2=dudx*dudx+dvdx*dvdx-dudy*dudy-dvdy*dvdy,
             a3=-dvdx*d2udxdy+dudx*d2vdxdy+dvdy*d2udx2-dudy*d2vdx2,
             b0=d2udx2+d2udy2,
             b1=d2vdx2+d2vdy2;
  if (std::fabs(denominator)<1e-7) std::cout<<"warning======irreversible F\n";
  denominator2=1.0/denominator2;denominator3=1.0/denominator3;
  divS[0]=-(dudx*b0+dvdx*b1)*denominator2+a0*a1*denominator3+a2*a3*denominator3;
  divS[1]=-(dudy*b0+dvdy*b1)*denominator2-a2*a1*denominator3+a0*a3*denominator3;
}*/
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

  return advF*(UU*dudx+VV*dudy) ;//+ difF*( ((up1x + um1x) + (up1y  + um1y)) - 4*u);
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

  return advF*(UU*dvdx+VV*dvdy);// + difF*( ((vp1x+ vm1x) + (vp1y+ vm1y)) - 4*v);
}

struct FluidStress
{
  FluidStress(const SimulationData& s):sim(s){}
  const SimulationData & sim;
  const StencilInfo stencil{-1,-1,0,2,2,1,true,{0,1}};
  const std::vector<cubism::BlockInfo>& tmpV1Info = sim.tmpV1->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpV2Info = sim.tmpV2->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& EchiInfo  = sim.Echi->getBlocksInfo();
  void operator()(const VectorLab& vellab,const BlockInfo& info){
    const Real i2h=0.5/info.h,nu=sim.nu;
    VectorBlock& __restrict__ TMP1=*(VectorBlock*) tmpV1Info[info.blockID].ptrBlock;
    VectorBlock& __restrict__ TMP2=*(VectorBlock*) tmpV2Info[info.blockID].ptrBlock;
    ScalarBlock& __restrict__ ECHI=*(ScalarBlock*) EchiInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      TMP1(ix,iy).u[0]=(1-ECHI(ix,iy).s)*2*nu*i2h*(vellab(ix+1,iy).u[0]-vellab(ix-1,iy).u[0]);
      TMP1(ix,iy).u[1]=(1-ECHI(ix,iy).s)*nu*i2h*(vellab(ix,iy+1).u[0]-vellab(ix,iy-1).u[0]+vellab(ix+1,iy).u[1]-vellab(ix-1,iy).u[1]);
      TMP2(ix,iy).u[0]=(1-ECHI(ix,iy).s)*TMP1(ix,iy).u[1];
      TMP2(ix,iy).u[1]=(1-ECHI(ix,iy).s)*2*nu*i2h*(vellab(ix,iy+1).u[1]-vellab(ix,iy-1).u[1]);
    }
  }
}
struct SolidStress
{
  SolidStress(const SimulationData& s, const int sid):sim(s),shapeid(sid){}
  const SimulationData & sim;
  const int shapeid;
  const StencilInfo stencil{-1,-1,0,2,2,1,true,{0,1}};
  const std::vector<cubism::BlockInfo>& tmpV1Info = sim.tmpV1->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpV2Info = sim.tmpV2->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& EchiInfo  = sim.Echi->getBlocksInfo();
  void operator()(const VectorLab& invmlab,const BlockInfo& info){
    const std::vector<ObstacleBlock*>& OBLOCK = sim.Eshapes[shapeid]->obstacleBlocks;
    if(OBLOCK[info.blockID] == nullptr) continue; //obst not in block
    ObstacleBlock& o = * OBLOCK[info.blockID];
    CHI_MAT & __restrict__ X = o.chi;
    const Real i2h=0.5/info.h,G=sim.Eshapes[shapeid]->G;
    VectorBlock& __restrict__ TMP1=*(VectorBlock*) tmpV1Info[info.blockID].ptrBlock;
    VectorBlock& __restrict__ TMP2=*(VectorBlock*) tmpV2Info[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      Real stress[4];
      SolidDstress(invmlab,ix,iy,info.h,stress);
      TMP1(ix,iy).u[0]+=stress[0]*G*X[iy][ix];
      TMP1(ix,iy).u[1]+=stress[1]*G*X[iy][ix];
      TMP2(ix,iy).u[0]+=stress[2]*G*X[iy][ix];
      TMP2(ix,iy).u[1]+=stress[3]*G*X[iy][ix];
    }
  }
}
struct DivStress
{
  DivStress(const SimulationData& s):sim(s){}
  const SimulationData& sim;
  const StencilInfo stencil{-1,-1,0,2,2,1,true,{0,1}};
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  void operator()(const VectorLab& lab1,const BlockInfo& info,const VectorLab& lab2,const BlockInfo& info2) const
  {
    const Real i2h=0.5/info.h;
    VectorBlock& __restrict__ TMP=*(VectorBlock*) tmpVInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      TMP(ix,iy).u[0]=sim.dt*i2h*(lab1(ix+1,iy).u[0]-lab1(ix-1,iy).u[0]+lab1(ix,iy+1).u[1]-lab1(ix,iy-1).u[1]);
      TMP(ix,iy).u[1]=sim.dt*i2h*(lab2(ix+1,iy).u[0]-lab2(ix-1,iy).u[0]+lab2(ix,iy+1).u[1]-lab2(ix,iy-1).u[1]);
    }
  }
}
struct KernelAdvectDiffuse
{
  KernelAdvectDiffuse(const SimulationData & s, const Real c, const Real uinfx, const Real uinfy) : sim(s),coef(c)
  {
    uinf[0] = uinfx;
    uinf[1] = uinfy;
  }
  const SimulationData & sim;
  const Real coef;
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
      TMP(ix,iy).u[0] = coef*(TMP(ix,iy).u[0]+dU_adv_dif(lab,uinf,afac,dfac,ix,iy));
      TMP(ix,iy).u[1] = coef*(TMP(ix,iy).u[1]+dV_adv_dif(lab,uinf,afac,dfac,ix,iy));
    }
    BlockCase<VectorBlock> * tempCase = (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    VectorBlock::ElementType * faceXm = nullptr;
    VectorBlock::ElementType * faceXp = nullptr;
    VectorBlock::ElementType * faceYm = nullptr;
    VectorBlock::ElementType * faceYp = nullptr;

    const Real aux_coef = dfac*coef;

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
  const Real UINF[2]= {sim.uinfx, sim.uinfy};

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
  std::cout<<"first step\n";
  FluidStress fs(sim);
  cubism::compute<VectorLab>(fs,sim.vel);
  for(auto shape:sim.Eshapes){
    SolidStress ss(sim,shape->obstacleID);
    cubism::compute<VectorLab>(ss,sim.invms[shape.obstacleID]);
  }
  DivStress Ds(sim);
  compute2<VectorLab,VectorLab>(Ds,sim.tmpV1,sim.tmpV2);
  KernelAdvectDiffuse Step1(sim,0.5,UINF[0],UINF[1]) ;
  cubism::compute<VectorLab>(Step1,sim.vel,sim.tmpV);

  //   2b) Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
  std::cout<<"second step\n";
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

  /********************************************************************/
  // 3. Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
  //   3a) Compute dt*RHS(u^{n+1/2}) and store it to tmpU,tmpV,tmpW
  cubism::compute<VectorLab>(fs,sim.vel);
  for(auto shape:sim.Eshapes){
    SolidStress ss(sim,shape->obstacleID);
    cubism::compute<VectorLab>(ss,sim.invms[shape.obstacleID]);
  }
  compute2<VectorLab,VectorLab>(Ds,sim.tmpV1,sim.tmpV2);
  KernelAdvectDiffuse Step2(sim,1.0,UINF[0],UINF[1]) ;
  cubism::compute<VectorLab>(Step2,sim.vel,sim.tmpV);
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
