//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "PressureSingle.h"
#include "Cubism/FluxCorrection.h"
#include "../Shape.h"

using namespace cubism;

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

//#define EXPL_INTEGRATE_MOM

namespace {

void ComputeJ(const Real * Rc, const Real * R, const Real * N, const Real * I, Real *J)
{
    //Invert I
    const Real m00 = 1.0; //I[0]; //set to these values for 2D!
    const Real m01 = 0.0; //I[3]; //set to these values for 2D!
    const Real m02 = 0.0; //I[4]; //set to these values for 2D!
    const Real m11 = 1.0; //I[1]; //set to these values for 2D!
    const Real m12 = 0.0; //I[5]; //set to these values for 2D!
    const Real m22 = I[5];//I[2]; //set to these values for 2D!
    Real a00 = m22*m11 - m12*m12;
    Real a01 = m02*m12 - m22*m01;
    Real a02 = m01*m12 - m02*m11;
    Real a11 = m22*m00 - m02*m02;
    Real a12 = m01*m02 - m00*m12;
    Real a22 = m00*m11 - m01*m01;
    const Real determinant =  1.0/((m00 * a00) + (m01 * a01) + (m02 * a02));
    a00 *= determinant;
    a01 *= determinant;
    a02 *= determinant;
    a11 *= determinant;
    a12 *= determinant;
    a22 *= determinant;

    const Real aux_0 = ( Rc[1] - R[1] )*N[2] - ( Rc[2] - R[2] )*N[1];
    const Real aux_1 = ( Rc[2] - R[2] )*N[0] - ( Rc[0] - R[0] )*N[2];
    const Real aux_2 = ( Rc[0] - R[0] )*N[1] - ( Rc[1] - R[1] )*N[0];
    J[0] = a00*aux_0 + a01*aux_1 + a02*aux_2;
    J[1] = a01*aux_0 + a11*aux_1 + a12*aux_2;
    J[2] = a02*aux_0 + a12*aux_1 + a22*aux_2;
}


void ElasticCollision (const Real  m1,const Real  m2,
                       const Real *I1,const Real *I2,
                       const Real *v1,const Real *v2,
                       const Real *o1,const Real *o2,
                       Real *hv1,Real *hv2,
                       Real *ho1,Real *ho2,
                       const Real *C1,const Real *C2,
                       const Real  NX,const Real  NY,const Real NZ,
                       const Real  CX,const Real  CY,const Real CZ,
                       Real *vc1,Real *vc2)
{
    const Real e = 1.0; // coefficient of restitution
    const Real N[3] ={NX,NY,NZ};
    const Real C[3] ={CX,CY,CZ};

    const Real k1[3] = { N[0]/m1, N[1]/m1, N[2]/m1};
    const Real k2[3] = {-N[0]/m2,-N[1]/m2,-N[2]/m2};
    Real J1[3];
    Real J2[3]; 
    ComputeJ(C,C1,N,I1,J1);
    ComputeJ(C,C2,N,I2,J2);
    J2[0] = -J2[0];
    J2[1] = -J2[1];
    J2[2] = -J2[2];

    Real u1DEF[3];
    u1DEF[0] = vc1[0] - v1[0] - ( o1[1]*(C[2]-C1[2]) - o1[2]*(C[1]-C1[1]) );
    u1DEF[1] = vc1[1] - v1[1] - ( o1[2]*(C[0]-C1[0]) - o1[0]*(C[2]-C1[2]) );
    u1DEF[2] = vc1[2] - v1[2] - ( o1[0]*(C[1]-C1[1]) - o1[1]*(C[0]-C1[0]) );
    Real u2DEF[3];
    u2DEF[0] = vc2[0] - v2[0] - ( o2[1]*(C[2]-C2[2]) - o2[2]*(C[1]-C2[1]) );
    u2DEF[1] = vc2[1] - v2[1] - ( o2[2]*(C[0]-C2[0]) - o2[0]*(C[2]-C2[2]) );
    u2DEF[2] = vc2[2] - v2[2] - ( o2[0]*(C[1]-C2[1]) - o2[1]*(C[0]-C2[0]) );

    const Real nom = e*( (vc1[0]-vc2[0])*N[0] + 
                           (vc1[1]-vc2[1])*N[1] + 
                           (vc1[2]-vc2[2])*N[2] )
                       + ( (v1[0]-v2[0] + u1DEF[0] - u2DEF[0] )*N[0] + 
                           (v1[1]-v2[1] + u1DEF[1] - u2DEF[1] )*N[1] + 
                           (v1[2]-v2[2] + u1DEF[2] - u2DEF[2] )*N[2] )
                  +( (o1[1]*(C[2]-C1[2]) - o1[2]*(C[1]-C1[1]) )* N[0]+
                     (o1[2]*(C[0]-C1[0]) - o1[0]*(C[2]-C1[2]) )* N[1]+
                     (o1[0]*(C[1]-C1[1]) - o1[1]*(C[0]-C1[0]) )* N[2])
                  -( (o2[1]*(C[2]-C2[2]) - o2[2]*(C[1]-C2[1]) )* N[0]+
                     (o2[2]*(C[0]-C2[0]) - o2[0]*(C[2]-C2[2]) )* N[1]+
                     (o2[0]*(C[1]-C2[1]) - o2[1]*(C[0]-C2[0]) )* N[2]);

    const Real denom = -(1.0/m1+1.0/m2) + 
               +( ( J1[1]*(C[2]-C1[2]) - J1[2]*(C[1]-C1[1]) ) *(-N[0])+
                  ( J1[2]*(C[0]-C1[0]) - J1[0]*(C[2]-C1[2]) ) *(-N[1])+
                  ( J1[0]*(C[1]-C1[1]) - J1[1]*(C[0]-C1[0]) ) *(-N[2]))
               -( ( J2[1]*(C[2]-C2[2]) - J2[2]*(C[1]-C2[1]) ) *(-N[0])+
                  ( J2[2]*(C[0]-C2[0]) - J2[0]*(C[2]-C2[2]) ) *(-N[1])+
                  ( J2[0]*(C[1]-C2[1]) - J2[1]*(C[0]-C2[0]) ) *(-N[2]));
    const Real impulse = nom/(denom+1e-21);
    hv1[0] = v1[0] + k1[0]*impulse;
    hv1[1] = v1[1] + k1[1]*impulse;
    hv1[2] = v1[2] + k1[2]*impulse;
    hv2[0] = v2[0] + k2[0]*impulse;
    hv2[1] = v2[1] + k2[1]*impulse;
    hv2[2] = v2[2] + k2[2]*impulse;
    ho1[0] = o1[0] + J1[0]*impulse;
    ho1[1] = o1[1] + J1[1]*impulse;
    ho1[2] = o1[2] + J1[2]*impulse;
    ho2[0] = o2[0] + J2[0]*impulse;
    ho2[1] = o2[1] + J2[1]*impulse;
    ho2[2] = o2[2] + J2[2]*impulse;
}

}//namespace

struct pressureCorrectionKernel
{
  pressureCorrectionKernel(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  void operator()(ScalarLab & P, const cubism::BlockInfo& info) const
  {
    const Real h = info.h, pFac = -0.5*sim.dt*h;
    VectorBlock&__restrict__ tmpV = *(VectorBlock*)  tmpVInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      tmpV(ix,iy).u[0] = pFac *(P(ix+1,iy).s-P(ix-1,iy).s);
      tmpV(ix,iy).u[1] = pFac *(P(ix,iy+1).s-P(ix,iy-1).s);
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
        faceXm[iy].clear();
        faceXm[iy].u[0] = pFac*(P(ix-1,iy).s+P(ix,iy).s);
      }
    }
    if (faceXp != nullptr)
    {
      int ix = VectorBlock::sizeX-1;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXp[iy].clear();
        faceXp[iy].u[0] = -pFac*(P(ix+1,iy).s+P(ix,iy).s);
      }
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYm[ix].clear();
        faceYm[ix].u[1] = pFac*(P(ix,iy-1).s+P(ix,iy).s);
      }
    }
    if (faceYp != nullptr)
    {
      int iy = VectorBlock::sizeY-1;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYp[ix].clear();
        faceYp[ix].u[1] = -pFac*(P(ix,iy+1).s+P(ix,iy).s);
      }
    }
  }
};

void PressureSingle::pressureCorrection(const Real dt)
{
  const pressureCorrectionKernel K(sim);
  cubism::compute<ScalarLab>(K,sim.pres,sim.tmpV);

  std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  #pragma omp parallel for
  for (size_t i=0; i < velInfo.size(); i++)
  {
      const Real ih2 = 1.0/velInfo[i].h/velInfo[i].h;
      VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;
      VectorBlock&__restrict__   tmpV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        V(ix,iy).u[0] += tmpV(ix,iy).u[0]*ih2;
        V(ix,iy).u[1] += tmpV(ix,iy).u[1]*ih2;
      }
  }
}

void PressureSingle::integrateMomenta(Shape * const shape) const
{
  const size_t Nblocks = velInfo.size();

  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const Real Cx = shape->centerOfMass[0];
  const Real Cy = shape->centerOfMass[1];
  Real PM=0, PJ=0, PX=0, PY=0, UM=0, VM=0, AM=0; //linear momenta

  #pragma omp parallel for reduction(+:PM,PJ,PX,PY,UM,VM,AM)
  for(size_t i=0; i<Nblocks; i++)
  {
    const VectorBlock& __restrict__ VEL = *(VectorBlock*)velInfo[i].ptrBlock;
    const Real hsq = velInfo[i].h*velInfo[i].h;

    if(OBLOCK[velInfo[i].blockID] == nullptr) continue;
    const CHI_MAT & __restrict__ rho = OBLOCK[velInfo[i].blockID]->rho;
    const CHI_MAT & __restrict__ chi = OBLOCK[velInfo[i].blockID]->chi;
    const UDEFMAT & __restrict__ udef = OBLOCK[velInfo[i].blockID]->udef;
    #ifndef EXPL_INTEGRATE_MOM
      const Real lambdt = sim.lambda * sim.dt;
    #endif

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if (chi[iy][ix] <= 0) continue;
      const Real udiff[2] = {
        VEL(ix,iy).u[0] - udef[iy][ix][0], VEL(ix,iy).u[1] - udef[iy][ix][1]
      };
      #ifdef EXPL_INTEGRATE_MOM
        const Real F = hsq * rho[iy][ix] * chi[iy][ix];
      #else
        const Real Xlamdt = chi[iy][ix] * lambdt;
        const Real F = hsq * rho[iy][ix] * Xlamdt / (1 + Xlamdt);
      #endif
      Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      PM += F;
      PJ += F * (p[0]*p[0] + p[1]*p[1]);
      PX += F * p[0];  PY += F * p[1];
      UM += F * udiff[0]; VM += F * udiff[1];
      AM += F * (p[0]*udiff[1] - p[1]*udiff[0]);
    }
  }
  Real quantities[7] = {PM,PJ,PX,PY,UM,VM,AM};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM, sim.chi->getCartComm());
  PM = quantities[0]; 
  PJ = quantities[1]; 
  PX = quantities[2]; 
  PY = quantities[3]; 
  UM = quantities[4]; 
  VM = quantities[5]; 
  AM = quantities[6];

  shape->fluidAngMom = AM; shape->fluidMomX = UM; shape->fluidMomY = VM;
  shape->penalDX=PX; shape->penalDY=PY; shape->penalM=PM; shape->penalJ=PJ;
}

void PressureSingle::penalize(const Real dt) const
{
  std::vector<cubism::BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();

  const size_t Nblocks = velInfo.size();

  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  for (const auto& shape : sim.shapes)
  {
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    const ObstacleBlock*const o = OBLOCK[velInfo[i].blockID];
    if (o == nullptr) continue;

    const Real u_s = shape->u;
    const Real v_s = shape->v;
    const Real omega_s = shape->omega;
    const Real Cx = shape->centerOfMass[0];
    const Real Cy = shape->centerOfMass[1];

    const CHI_MAT & __restrict__ X = o->chi;
    const UDEFMAT & __restrict__ UDEF = o->udef;
    const ScalarBlock& __restrict__ CHI = *(ScalarBlock*)chiInfo[i].ptrBlock;
          VectorBlock& __restrict__   V = *(VectorBlock*)velInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      // What if multiple obstacles share a block? Do not write udef onto
      // grid if CHI stored on the grid is greater than obst's CHI.
      if(CHI(ix,iy).s > X[iy][ix]) continue;
      if(X[iy][ix] <= 0) continue; // no need to do anything

      Real p[2];
      velInfo[i].pos(p, ix, iy);
      p[0] -= Cx;
      p[1] -= Cy;
      #ifndef EXPL_INTEGRATE_MOM
        const Real alpha = 1/(1 + sim.lambda * dt * X[iy][ix]);
      #else
        const Real alpha = 1 - X[iy][ix];
      #endif

      const Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
      const Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
      V(ix,iy).u[0] = alpha*V(ix,iy).u[0] + (1-alpha)*US;
      V(ix,iy).u[1] = alpha*V(ix,iy).u[1] + (1-alpha)*VS;
    }
  }
}

struct updatePressureRHS
{
  // RHS of Poisson equation is div(u) - chi * div(u_def)
  // It is computed here and stored in TMP

  updatePressureRHS(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0,1}};
  cubism::StencilInfo stencil2{-1, -1, 0, 2, 2, 1, false, {0,1}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();

  void operator()(VectorLab & velLab, VectorLab & uDefLab, const cubism::BlockInfo& info, const cubism::BlockInfo& info2) const
  {
    const Real h = info.h;
    const Real facDiv = 0.5*h/sim.dt;
    ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    ScalarBlock& __restrict__ CHI = *(ScalarBlock*) chiInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      TMP(ix, iy).s  =   facDiv                *( (velLab(ix+1,iy).u[0] -  velLab(ix-1,iy).u[0])
                                               +  (velLab(ix,iy+1).u[1] -  velLab(ix,iy-1).u[1]));
      TMP(ix, iy).s += - facDiv * CHI(ix,iy).s *((uDefLab(ix+1,iy).u[0] - uDefLab(ix-1,iy).u[0])
                                               + (uDefLab(ix,iy+1).u[1] - uDefLab(ix,iy-1).u[1]));
    }
    BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    ScalarBlock::ElementType * faceXm = nullptr;
    ScalarBlock::ElementType * faceXp = nullptr;
    ScalarBlock::ElementType * faceYm = nullptr;
    ScalarBlock::ElementType * faceYp = nullptr;
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
        faceXm[iy].s  =  facDiv                *( velLab(ix-1,iy).u[0] +  velLab(ix,iy).u[0]) ;
        faceXm[iy].s += -(facDiv * CHI(ix,iy).s)*(uDefLab(ix-1,iy).u[0] + uDefLab(ix,iy).u[0]) ;
      }
    }
    if (faceXp != nullptr)
    {
      int ix = VectorBlock::sizeX-1;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXp[iy].s  = -facDiv               *( velLab(ix+1,iy).u[0] +  velLab(ix,iy).u[0]);
        faceXp[iy].s -= -(facDiv *CHI(ix,iy).s)*(uDefLab(ix+1,iy).u[0] + uDefLab(ix,iy).u[0]);
      }
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYm[ix].s  =  facDiv               *( velLab(ix,iy-1).u[1] +  velLab(ix,iy).u[1]);
        faceYm[ix].s += -(facDiv *CHI(ix,iy).s)*(uDefLab(ix,iy-1).u[1] + uDefLab(ix,iy).u[1]);
      }
    }
    if (faceYp != nullptr)
    {
      int iy = VectorBlock::sizeY-1;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYp[ix].s  = -facDiv               *( velLab(ix,iy+1).u[1] +  velLab(ix,iy).u[1]);
        faceYp[ix].s -= -(facDiv *CHI(ix,iy).s)*(uDefLab(ix,iy+1).u[1] + uDefLab(ix,iy).u[1]);
      }
    }
  }
};

struct updatePressureRHS1
{
  // RHS of Poisson equation is div(u) - chi * div(u_def)
  // It is computed here and stored in TMP

  updatePressureRHS1(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& poldInfo = sim.pold->getBlocksInfo();

  void operator()(ScalarLab & lab, const cubism::BlockInfo& info) const
  {
    ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      TMP(ix, iy).s  -=  ( ((lab(ix-1,iy).s + lab(ix+1,iy).s) + (lab(ix,iy-1).s + lab(ix,iy+1).s)) - 4.0*lab(ix,iy).s);

    BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    ScalarBlock::ElementType * faceXm = nullptr;
    ScalarBlock::ElementType * faceXp = nullptr;
    ScalarBlock::ElementType * faceYm = nullptr;
    ScalarBlock::ElementType * faceYp = nullptr;
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
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
        faceXm[iy] = lab(ix-1,iy) - lab(ix,iy);
    }
    if (faceXp != nullptr)
    {
      int ix = ScalarBlock::sizeX-1;
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
        faceXp[iy] = lab(ix+1,iy) - lab(ix,iy);
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
        faceYm[ix] = lab(ix,iy-1) - lab(ix,iy);
    }
    if (faceYp != nullptr)
    {
      int iy = ScalarBlock::sizeY-1;
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
        faceYp[ix] = lab(ix,iy+1) - lab(ix,iy);
    }
  }
};

void PressureSingle::preventCollidingObstacles() const
{
    const auto& shapes = sim.shapes;
    const auto & infos  = sim.chi->getBlocksInfo();
    const size_t N = shapes.size();

    struct CollisionInfo // hitter and hittee, symmetry but we do things twice
    {
        Real iM = 0;
        Real iPosX = 0;
        Real iPosY = 0;
        Real iPosZ = 0;
        Real iMomX = 0;
        Real iMomY = 0;
        Real iMomZ = 0;
        Real ivecX = 0;
        Real ivecY = 0;
        Real ivecZ = 0;
        Real jM = 0;
        Real jPosX = 0;
        Real jPosY = 0;
        Real jPosZ = 0;
        Real jMomX = 0;
        Real jMomY = 0;
        Real jMomZ = 0;
        Real jvecX = 0;
        Real jvecY = 0;
        Real jvecZ = 0;
    };
    std::vector<CollisionInfo> collisions(N);

    std::vector <Real> n_vec(3*N,0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<N; ++i)
    for (size_t j=0; j<N; ++j)
    {
        if(i==j) continue;
        auto & coll = collisions[i];

        const auto& iBlocks = shapes[i]->obstacleBlocks;
        const Real iU0      = shapes[i]->u;
        const Real iU1      = shapes[i]->v;
        const Real iU2      = 0; //set to 0 for 2D
        const Real iomega0  = 0; //set to 0 for 2D
        const Real iomega1  = 0; //set to 0 for 2D
        const Real iomega2  = shapes[i]->omega;
        const Real iCx      = shapes[i]->centerOfMass[0];
        const Real iCy      = shapes[i]->centerOfMass[1];
        const Real iCz      = 0; //set to 0 for 2D

        const auto& jBlocks = shapes[j]->obstacleBlocks;
        const Real jU0      = shapes[j]->u;
        const Real jU1      = shapes[j]->v;
        const Real jU2      = 0; //set to 0 for 2D
        const Real jomega0  = 0; //set to 0 for 2D
        const Real jomega1  = 0; //set to 0 for 2D
        const Real jomega2  = shapes[j]->omega;
        const Real jCx      = shapes[j]->centerOfMass[0];
        const Real jCy      = shapes[j]->centerOfMass[1];
        const Real jCz      = 0; //set to 0 for 2D

        assert(iBlocks.size() == jBlocks.size());

        const size_t nBlocks = iBlocks.size();
        for (size_t k=0; k<nBlocks; ++k)
        {
            if ( iBlocks[k] == nullptr || jBlocks[k] == nullptr ) continue;

            const auto & iSDF  = iBlocks[k]->dist;
            const auto & jSDF  = jBlocks[k]->dist;

            const CHI_MAT & iChi  = iBlocks[k]->chi;
            const CHI_MAT & jChi  = jBlocks[k]->chi;

            const UDEFMAT & iUDEF = iBlocks[k]->udef;
            const UDEFMAT & jUDEF = jBlocks[k]->udef;

            for(int iy=0; iy<VectorBlock::sizeY; ++iy)
            for(int ix=0; ix<VectorBlock::sizeX; ++ix)
            {
                if(iChi[iy][ix] <= 0.0 || jChi[iy][ix] <= 0.0 ) continue;

                const auto pos = infos[k].pos<Real>(ix, iy);

                const Real iUr0 = iomega1* (pos[2] - iCz) - iomega2*(pos[1]-iCy);
                const Real iUr1 = iomega2* (pos[0] - iCx) - iomega0*(pos[2]-iCz);
                const Real iUr2 = iomega0* (pos[1] - iCy) - iomega1*(pos[0]-iCx);
                coll.iM    += iChi[iy][ix];
                coll.iPosX += iChi[iy][ix] * pos[0];
                coll.iPosY += iChi[iy][ix] * pos[1];
                coll.iMomX += iChi[iy][ix] * (iU0 + iUr0 + iUDEF[iy][ix][0]);
                coll.iMomY += iChi[iy][ix] * (iU1 + iUr1 + iUDEF[iy][ix][1]);
                coll.iMomZ += iChi[iy][ix] * (iU2 + iUr2);// + iUDEF[iy][ix][2]);//set to 0 for 2D

                const Real jUr0 = jomega1* (pos[2] - jCz) - jomega2*(pos[1]-jCy);
                const Real jUr1 = jomega2* (pos[0] - jCx) - jomega0*(pos[2]-jCz);
                const Real jUr2 = jomega0* (pos[1] - jCy) - jomega1*(pos[0]-jCx);
                coll.jM    += jChi[iy][ix];
                coll.jPosX += jChi[iy][ix] * pos[0];
                coll.jPosY += jChi[iy][ix] * pos[1];
                coll.jMomX += jChi[iy][ix] * (jU0 + jUr0 + jUDEF[iy][ix][0]);
                coll.jMomY += jChi[iy][ix] * (jU1 + jUr1 + jUDEF[iy][ix][1]);
                coll.jMomZ += jChi[iy][ix] * (jU2 + jUr2);// + jUDEF[iy][ix][2]);//set to 0 for 2D
              
                Real dSDFdx_i;
                Real dSDFdx_j;
                if (ix == 0)
                {
                  dSDFdx_i = iSDF[iy][ix+1] - iSDF[iy][ix];
                  dSDFdx_j = jSDF[iy][ix+1] - jSDF[iy][ix];
                }
                else if (ix == VectorBlock::sizeX - 1)
                {
                  dSDFdx_i = iSDF[iy][ix] - iSDF[iy][ix-1];
                  dSDFdx_j = jSDF[iy][ix] - jSDF[iy][ix-1];
                }
                else
                {
                  dSDFdx_i = 0.5*(iSDF[iy][ix+1] - iSDF[iy][ix-1]);
                  dSDFdx_j = 0.5*(jSDF[iy][ix+1] - jSDF[iy][ix-1]);
                }

                Real dSDFdy_i;
                Real dSDFdy_j;
                if (iy == 0)
                {
                  dSDFdy_i = iSDF[iy+1][ix] - iSDF[iy][ix];
                  dSDFdy_j = jSDF[iy+1][ix] - jSDF[iy][ix];
                }
                else if (iy == VectorBlock::sizeY - 1)
                {
                  dSDFdy_i = iSDF[iy][ix] - iSDF[iy-1][ix];
                  dSDFdy_j = jSDF[iy][ix] - jSDF[iy-1][ix];
                }
                else
                {
                  dSDFdy_i = 0.5*(iSDF[iy+1][ix] - iSDF[iy-1][ix]);
                  dSDFdy_j = 0.5*(jSDF[iy+1][ix] - jSDF[iy-1][ix]);
                }

                coll.ivecX += iChi[iy][ix] * dSDFdx_i;
                coll.ivecY += iChi[iy][ix] * dSDFdy_i;

                coll.jvecX += jChi[iy][ix] * dSDFdx_j;
                coll.jvecY += jChi[iy][ix] * dSDFdy_j;
            }
        }
    }

    std::vector<Real> buffer(20*N); //CollisionInfo holds 20 Reals
    for (size_t i = 0 ; i < N ; i++)
    {
        auto & coll = collisions[i];
        buffer[20*i     ] = coll.iM   ;
        buffer[20*i + 1 ] = coll.iPosX;
        buffer[20*i + 2 ] = coll.iPosY;
        buffer[20*i + 3 ] = coll.iPosZ;
        buffer[20*i + 4 ] = coll.iMomX;
        buffer[20*i + 5 ] = coll.iMomY;
        buffer[20*i + 6 ] = coll.iMomZ;
        buffer[20*i + 7 ] = coll.ivecX;
        buffer[20*i + 8 ] = coll.ivecY;
        buffer[20*i + 9 ] = coll.ivecZ;
        buffer[20*i + 10] = coll.jM   ;
        buffer[20*i + 11] = coll.jPosX;
        buffer[20*i + 12] = coll.jPosY;
        buffer[20*i + 13] = coll.jPosZ;
        buffer[20*i + 14] = coll.jMomX;
        buffer[20*i + 15] = coll.jMomY;
        buffer[20*i + 16] = coll.jMomZ;
        buffer[20*i + 17] = coll.jvecX;
        buffer[20*i + 18] = coll.jvecY;
        buffer[20*i + 19] = coll.jvecZ;

    }
    MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), MPI_Real, MPI_SUM, sim.chi->getCartComm());
    for (size_t i = 0 ; i < N ; i++)
    {
        auto & coll = collisions[i];
        coll.iM    = buffer[20*i     ];
        coll.iPosX = buffer[20*i + 1 ];
        coll.iPosY = buffer[20*i + 2 ];
        coll.iPosZ = buffer[20*i + 3 ];
        coll.iMomX = buffer[20*i + 4 ];
        coll.iMomY = buffer[20*i + 5 ];
        coll.iMomZ = buffer[20*i + 6 ];
        coll.ivecX = buffer[20*i + 7 ];
        coll.ivecY = buffer[20*i + 8 ];
        coll.ivecZ = buffer[20*i + 9 ];
        coll.jM    = buffer[20*i + 10];
        coll.jPosX = buffer[20*i + 11];
        coll.jPosY = buffer[20*i + 12];
        coll.jPosZ = buffer[20*i + 13];
        coll.jMomX = buffer[20*i + 14];
        coll.jMomY = buffer[20*i + 15];
        coll.jMomZ = buffer[20*i + 16];
        coll.jvecX = buffer[20*i + 17];
        coll.jvecY = buffer[20*i + 18];
        coll.jvecZ = buffer[20*i + 19];
    }

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<N; ++i)
    for (size_t j=i+1; j<N; ++j)
    {
        if (i==j) continue;
        const Real m1 = shapes[i]->M;
        const Real m2 = shapes[j]->M;
        const Real v1[3]={shapes[i]->u,shapes[i]->v,0.0};
        const Real v2[3]={shapes[j]->u,shapes[j]->v,0.0};
        const Real o1[3]={0,0,shapes[i]->omega};
        const Real o2[3]={0,0,shapes[j]->omega};
        const Real C1[3]={shapes[i]->centerOfMass[0],shapes[i]->centerOfMass[1],0};
        const Real C2[3]={shapes[j]->centerOfMass[0],shapes[j]->centerOfMass[1],0};
        const Real I1[6]={1.0,0,0,0,0,shapes[i]->J};
        const Real I2[6]={1.0,0,0,0,0,shapes[j]->J};

        auto & coll       = collisions[i];
        auto & coll_other = collisions[j];
        // less than one fluid element of overlap: wait to get closer. no hit
        if(coll.iM       < 2.0 || coll.jM       < 2.0) continue; //object i did not collide
        if(coll_other.iM < 2.0 || coll_other.jM < 2.0) continue; //object j did not collide

        if (std::fabs(coll.iPosX/coll.iM  - coll_other.iPosX/coll_other.iM ) > shapes[i]->getCharLength() ||
            std::fabs(coll.iPosY/coll.iM  - coll_other.iPosY/coll_other.iM ) > shapes[i]->getCharLength() )
        {
            continue; // then both objects i and j collided, but not with each other!
        }

        // A collision happened!
        sim.bCollision = true;

        const bool iForced = shapes[i]->bForced;
        const bool jForced = shapes[j]->bForced;
        if (iForced || jForced)
        {
            std::cout << "[CUP2D] WARNING: Forced objects not supported for collision." << std::endl;
            // MPI_Abort(sim.chi->getCartComm(),1);
        }

        Real ho1[3];
        Real ho2[3];
        Real hv1[3];
        Real hv2[3];

        //1. Compute collision normal vector (NX,NY,NZ)
        const Real norm_i = std::sqrt(coll.ivecX*coll.ivecX + coll.ivecY*coll.ivecY + coll.ivecZ*coll.ivecZ);
        const Real norm_j = std::sqrt(coll.jvecX*coll.jvecX + coll.jvecY*coll.jvecY + coll.jvecZ*coll.jvecZ);
        const Real mX = coll.ivecX/norm_i - coll.jvecX/norm_j;
        const Real mY = coll.ivecY/norm_i - coll.jvecY/norm_j;
        const Real mZ = coll.ivecZ/norm_i - coll.jvecZ/norm_j;
        const Real inorm = 1.0/std::sqrt(mX*mX + mY*mY + mZ*mZ);
        const Real NX = mX * inorm;
        const Real NY = mY * inorm;
        const Real NZ = mZ * inorm;

        //If objects are already moving away from each other, don't do anything
        //if( (v2[0]-v1[0])*NX + (v2[1]-v1[1])*NY + (v2[2]-v1[2])*NZ <= 0 ) continue;
        const Real hitVelX = coll.jMomX / coll.jM - coll.iMomX / coll.iM;
        const Real hitVelY = coll.jMomY / coll.jM - coll.iMomY / coll.iM;
        const Real hitVelZ = coll.jMomZ / coll.jM - coll.iMomZ / coll.iM;
        const Real projVel = hitVelX * NX + hitVelY * NY + hitVelZ * NZ;

        /*const*/ Real vc1[3] = {coll.iMomX/coll.iM, coll.iMomY/coll.iM, coll.iMomZ/coll.iM};
        /*const*/ Real vc2[3] = {coll.jMomX/coll.jM, coll.jMomY/coll.jM, coll.jMomZ/coll.jM};


        if(projVel<=0) continue; // vel goes away from collision: no need to bounce

        //2. Compute collision location
        const Real inv_iM = 1.0/coll.iM;
        const Real inv_jM = 1.0/coll.jM;
        const Real iPX = coll.iPosX * inv_iM; // object i collision location
        const Real iPY = coll.iPosY * inv_iM;
        const Real iPZ = coll.iPosZ * inv_iM;
        const Real jPX = coll.jPosX * inv_jM; // object j collision location
        const Real jPY = coll.jPosY * inv_jM;
        const Real jPZ = coll.jPosZ * inv_jM;
        const Real CX = 0.5*(iPX+jPX);
        const Real CY = 0.5*(iPY+jPY);
        const Real CZ = 0.5*(iPZ+jPZ);

        //3. Take care of the collision. Assume elastic collision (kinetic energy is conserved)
        ElasticCollision(m1,m2,I1,I2,v1,v2,o1,o2,hv1,hv2,ho1,ho2,C1,C2,NX,NY,NZ,CX,CY,CZ,vc1,vc2);
        shapes[i]->u = hv1[0];
        shapes[i]->v = hv1[1];
        //shapes[i]->transVel[2] = hv1[2];
        shapes[j]->u = hv2[0];
        shapes[j]->v = hv2[1];
        //shapes[j]->transVel[2] = hv2[2];
        //shapes[i]->angVel[0] = ho1[0];
        //shapes[i]->angVel[1] = ho1[1];
        shapes[i]->omega = ho1[2];
        //shapes[j]->angVel[0] = ho2[0];
        //shapes[j]->angVel[1] = ho2[1];
        shapes[j]->omega = ho2[2];

        if ( sim.rank == 0)
        {
            #pragma omp critical
            {
                std::cout << "Collision between objects " << i << " and " << j << std::endl;
                std::cout << " iM   (0) = " << collisions[i].iM    << " jM   (1) = " << collisions[j].jM << std::endl;
                std::cout << " jM   (0) = " << collisions[i].jM    << " jM   (1) = " << collisions[j].iM << std::endl;
                std::cout << " Normal vector = (" << NX << "," << NY << "," << NZ << std::endl;
                std::cout << " Location      = (" << CX << "," << CY << "," << CZ << std::endl;
                std::cout << " Shape " << i << " before collision u    =(" <<  v1[0] << "," <<  v1[1] << "," <<  v1[2] << ")" << std::endl;
                std::cout << " Shape " << i << " after  collision u    =(" << hv1[0] << "," << hv1[1] << "," << hv1[2] << ")" << std::endl;
                std::cout << " Shape " << j << " before collision u    =(" <<  v2[0] << "," <<  v2[1] << "," <<  v2[2] << ")" << std::endl;
                std::cout << " Shape " << j << " after  collision u    =(" << hv2[0] << "," << hv2[1] << "," << hv2[2] << ")" << std::endl;
                std::cout << " Shape " << i << " before collision omega=(" <<  o1[0] << "," <<  o1[1] << "," <<  o1[2] << ")" << std::endl;
                std::cout << " Shape " << i << " after  collision omega=(" << ho1[0] << "," << ho1[1] << "," << ho1[2] << ")" << std::endl;
                std::cout << " Shape " << j << " before collision omega=(" <<  o2[0] << "," <<  o2[1] << "," <<  o2[2] << ")" << std::endl;
                std::cout << " Shape " << j << " after  collision omega=(" << ho2[0] << "," << ho2[1] << "," << ho2[2] << ")" << std::endl;
            }
        }
    }
}


void PressureSingle::operator()(const Real dt)
{
  sim.startProfiler("Pressure");
  const size_t Nblocks = velInfo.size();

  // update velocity of obstacle
  for(const auto& shape : sim.shapes) {
    integrateMomenta(shape.get());
    shape->updateVelocity(dt);
  }
  // take care if two obstacles collide
  preventCollidingObstacles();

  // apply penalization force
  penalize(dt);

  // compute pressure RHS
  updatePressureRHS K(sim);
  compute<updatePressureRHS,VectorGrid,VectorLab,VectorGrid,VectorLab,ScalarGrid>(K,*sim.vel,*sim.uDef,true,sim.tmp);


  //Add p_old (+dp/dt) to RHS
  const std::vector<cubism::BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& poldInfo = sim.pold->getBlocksInfo();
  
  const size_t size_of_correction = sim.GuessDpDt ? Nblocks*VectorBlock::sizeY*VectorBlock::sizeX : 1;
  std::vector<Real> correction (size_of_correction,0.0);

  //initial guess etc.
  if (sim.GuessDpDt && sim.step > 10)
  {
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__   PRES = *(ScalarBlock*)  presInfo[i].ptrBlock;
      ScalarBlock & __restrict__   POLD = *(ScalarBlock*)  poldInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real dpdt = (2.0*(PRES(ix,iy).s - POLD(ix,iy).s))/(sim.dt_old+ sim.dt_old2);
        const int index = i*VectorBlock::sizeY*VectorBlock::sizeX+iy*VectorBlock::sizeX+ix;
        correction[index] = ((0.5*dpdt)*(sim.dt+sim.dt_old))*.5;
        POLD  (ix,iy).s = PRES (ix,iy).s + correction[index];
        PRES  (ix,iy).s = 0;
      }
    }
  }
  else
  {
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__   PRES = *(ScalarBlock*)  presInfo[i].ptrBlock;
      ScalarBlock & __restrict__   POLD = *(ScalarBlock*)  poldInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        POLD  (ix,iy).s = PRES (ix,iy).s;
        PRES  (ix,iy).s = 0;
      }
    }
  }
  updatePressureRHS1 K1(sim);
  cubism::compute<ScalarLab>(K1,sim.pold,sim.tmp);
  if (sim.GuessDpDt && sim.step > 10)
  {
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__   POLD = *(ScalarBlock*)  poldInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const int index = i*VectorBlock::sizeY*VectorBlock::sizeX+iy*VectorBlock::sizeX+ix;
        POLD(ix,iy).s -= correction[index];
      }
    }
  }

  pressureSolver->solve(sim.tmp, sim.pres);

  if (sim.GuessDpDt && sim.step > 10)
  {
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__   POLD = *(ScalarBlock*)  poldInfo[i].ptrBlock;
      ScalarBlock & __restrict__   PRES = *(ScalarBlock*)  presInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const int index = i*VectorBlock::sizeY*VectorBlock::sizeX+iy*VectorBlock::sizeX+ix;
        PRES(ix,iy).s += POLD(ix,iy).s + correction[index];
      }
    }
  }
  else
  {
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__   POLD = *(ScalarBlock*)  poldInfo[i].ptrBlock;
      ScalarBlock & __restrict__   PRES = *(ScalarBlock*)  presInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        PRES(ix,iy).s += POLD(ix,iy).s;
      }
    }
  }

  // apply pressure correction
  pressureCorrection(dt);

  sim.stopProfiler();
}

PressureSingle::PressureSingle(SimulationData& s) :
  Operator{s},
  pressureSolver{makePoissonSolver(s)}
{ }

PressureSingle::~PressureSingle() = default;
