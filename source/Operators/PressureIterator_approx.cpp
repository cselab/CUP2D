//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureIterator_approx.h"
#include "../Poisson/HYPREdirichletVarRho.h"
#include "../Shape.h"
#include "../Utils/BufferedLogger.h"
#include <stdio.h>

using namespace cubism;

static inline Real DfaceDx(const ScalarLab& B, const int ix, const int iy) {
  return B(ix,iy).s - B(ix-1,iy).s;
}
static inline Real DfaceDy(const ScalarLab& B, const int ix, const int iy) {
  return B(ix,iy).s - B(ix,iy-1).s;
}
static inline Real DcellDx(const ScalarLab& B, const int ix, const int iy) {
  return ( B(ix+1,iy).s - B(ix-1,iy).s ) / 2;
}
static inline Real DcellDy(const ScalarLab& B, const int ix, const int iy) {
  return ( B(ix,iy+1).s - B(ix,iy-1).s ) / 2;
}
static inline Real faceX(const ScalarLab& B, const int ix, const int iy) {
  return ( B(ix,iy).s + B(ix-1,iy).s ) / 2;
}
static inline Real faceY(const ScalarLab& B, const int ix, const int iy) {
  return ( B(ix,iy).s + B(ix,iy-1).s ) / 2;
}
static inline Real laplacian(const ScalarLab&B, const int ix, const int iy) {
  return B(ix+1,iy).s +B(ix-1,iy).s +B(ix,iy+1).s +B(ix,iy-1).s -4*B(ix,iy).s;
}

// FACE CENTERED VECTOR FIELDS (all except udef):
template<int i>
static inline Real Dcell(const VectorLab& B, const int ix, const int iy) {
  if(i==0) return B(ix+1,iy).u[0] - B(ix,iy).u[0];
  else     return B(ix,iy+1).u[1] - B(ix,iy).u[1];
}
template<int i>
static inline Real cell(const VectorLab& B, const int ix, const int iy) {
  if(i==0) return ( B(ix+1,iy).u[0] + B(ix,iy).u[0] ) / 2;
  else     return ( B(ix,iy+1).u[1] + B(ix,iy).u[1] ) / 2;
}
static inline Real div(const VectorLab& B, const int ix, const int iy) {
  return Dcell<0>(B,ix,iy) + Dcell<1>(B,ix,iy);
}

// CELL CENTERED VECTOR FIELDS (udef, which uses CHI field to be computed):
template<int i>
static inline Real faceX(const VectorLab& B, const int ix, const int iy) {
  return ( B(ix,iy).u[i] + B(ix-1,iy).u[i] ) / 2;
}
template<int i>
static inline Real faceY(const VectorLab& B, const int ix, const int iy) {
  return ( B(ix,iy).u[i] + B(ix,iy-1).u[i] ) / 2;
}

template<typename T>
static inline T mean(const T A, const T B) { return (A+B)/2; }
//static inline T mean(const T A, const T B) { return 2*A*B/(A+B); }

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

void PressureVarRho_approx::pressureCorrectionInit(const double dt)
{
  const size_t Nblocks = velInfo.size();

  const Real h = sim.getH(), pFac = -dt/h, dA = h*h;
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>&vPresInfo = sim.vFluid->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  Real intInvRho = 0;
  #pragma omp parallel reduction(+ : intInvRho)
  {
    static constexpr int stenBegV[3] = { 0, 0, 0}, stenEndV[3] = { 2, 2, 1};
    static constexpr int stenBegP[3] = {-1,-1, 0}, stenEndP[3] = { 1, 1, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBegP, stenEndP, 0);
    VectorLab velLab;   velLab.prepare(*(sim.vel),  stenBegV, stenEndV, 0);
    VectorLab uDefLab; uDefLab.prepare(*(sim.uDef), stenBegV, stenEndV, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
       velLab.load( velInfo[i],0); const VectorLab& __restrict__ V   =  velLab;
      uDefLab.load(uDefInfo[i],0); const VectorLab& __restrict__ UDEF= uDefLab;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;
      const ScalarBlock&__restrict__  CHI= *(ScalarBlock*)  chiInfo[i].ptrBlock;
      // returns : pressure-corrected velocity and initial pressure eq RHS
      VectorBlock & __restrict__ vPres =  *(VectorBlock*) vPresInfo[i].ptrBlock;
      ScalarBlock & __restrict__ pRhs  =  *(ScalarBlock*)  pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        vPres(ix,iy).u[0] = V(ix,iy).u[0] + pFac*DfaceDx(P,ix,iy)*IRHO(ix,iy).s;
        vPres(ix,iy).u[1] = V(ix,iy).u[1] + pFac*DfaceDy(P,ix,iy)*IRHO(ix,iy).s;
        pRhs(ix,iy).s = div(V, ix,iy) - CHI(ix,iy).s * div(UDEF, ix,iy);
        intInvRho += dA * IRHO(ix,iy).s;
      }

      ((VectorBlock*) uDefInfo[i].ptrBlock)->clear();
      ((VectorBlock*) tmpVInfo[i].ptrBlock)->clear();
    }
  }
  avgInvRho = intInvRho / (sim.extents[0] * sim.extents[1]);
  //avgInvRho = 1/avgInvRho;
  printf("Average invRho %e\n", avgInvRho);
}

Real PressureVarRho_approx::pressureCorrection(const double dt) const
{
  const size_t Nblocks = velInfo.size();

  const Real h = sim.getH(), pFac = -dt/h;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& vPresInfo= sim.vFluid->getBlocksInfo();

  Real DP = 0, NP = 0;
  #pragma omp parallel reduction(+ : DP, NP)
  {
    static constexpr int stenBegP[3] = {-1,-1, 0}, stenEndP[3] = { 1, 1, 1};
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBegP, stenEndP, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBegP, stenEndP, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
       tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ Pcur =  tmpLab;
      presLab.load(presInfo[i],0); const ScalarLab&__restrict__ P    = presLab;
      VectorBlock & __restrict__    vPres= *(VectorBlock*)vPresInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;
      const VectorBlock&__restrict__    V= *(VectorBlock*)  velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real dUdiv = DfaceDx(P,ix,iy) * (IRHO(ix,iy).s-avgInvRho);
        const Real dVdiv = DfaceDy(P,ix,iy) * (IRHO(ix,iy).s-avgInvRho);
        const Real dUpre = DfaceDx(Pcur,ix,iy) * avgInvRho;
        const Real dVpre = DfaceDy(Pcur,ix,iy) * avgInvRho;
        vPres(ix,iy).u[0] = V(ix,iy).u[0] + pFac * ( dUpre + dUdiv );
        vPres(ix,iy).u[1] = V(ix,iy).u[1] + pFac * ( dVpre + dVdiv );
        DP += std::pow(Pcur(ix,iy).s - P(ix,iy).s, 2);
        NP += std::pow(Pcur(ix,iy).s, 2);
      }
    }
  }

  return std::sqrt(DP / std::max(EPS, NP));
}

void PressureVarRho_approx::integrateMomenta(Shape * const shape) const
{
  const size_t Nblocks = velInfo.size();

  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const std::vector<BlockInfo>& vFluidInfo = sim.vFluid->getBlocksInfo();

  const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
  const double hsq = std::pow(velInfo[0].h_gridpoint, 2);
  double PM=0, PJ=0, PX=0, PY=0, UM=0, VM=0, AM=0; //linear momenta

  #pragma omp parallel reduction(+:PM,PJ,PX,PY,UM,VM,AM)
  {
    static constexpr int stenBeg[3] = {0,0,0}, stenEnd[3] = {2,2,1};
    VectorLab velLab; velLab.prepare(*(sim.vFluid), stenBeg, stenEnd, 0);

    #pragma omp for schedule(dynamic,1)
    for(size_t i=0; i<Nblocks; ++i)
    {
      if(OBLOCK[vFluidInfo[i].blockID] == nullptr) continue;

      velLab.load(vFluidInfo[i],0); const auto& __restrict__ V = velLab;
      const CHI_MAT & __restrict__ rho = OBLOCK[ vFluidInfo[i].blockID ]->rho;
      const CHI_MAT & __restrict__ chi = OBLOCK[ vFluidInfo[i].blockID ]->chi;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        if (chi[iy][ix] <= 0) continue;
        const Real F = hsq * rho[iy][ix] * chi[iy][ix];
        double p[2]; vFluidInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
        PM += F;
        PJ += F * (p[0]*p[0] + p[1]*p[1]);
        PX += F *  p[0];
        PY += F *  p[1];
        UM += F *  cell<0>(V,ix,iy);
        VM += F *  cell<1>(V,ix,iy);
        AM += F * (p[0]*cell<1>(V,ix,iy) - p[1]*cell<0>(V,ix,iy));
      }
    }
  }

  shape->fluidAngMom = AM; shape->fluidMomX = UM; shape->fluidMomY = VM;
  shape->penalDX=PX; shape->penalDY=PY; shape->penalM=PM; shape->penalJ=PJ;
}

Real PressureVarRho_approx::penalize(const Real dt, const int iter) const
{
  const size_t Nblocks = velInfo.size();

  const std::vector<BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& vFluidInfo = sim.vFluid->getBlocksInfo();

  Real MX = 0, MY = 0, DMX = 0, DMY = 0;
  #pragma omp parallel reduction(+ : MX, MY, DMX, DMY)
  {
    static constexpr int stenBeg[3] = {-1,-1,0}, stenEnd[3] = {1,1,1};
    VectorLab udefLab; udefLab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);
    ScalarLab chiLab;   chiLab.prepare(*(sim.chi ), stenBeg, stenEnd, 0);

    #pragma omp for schedule(dynamic, 1)
    for (size_t i=0; i < Nblocks; ++i)
    for (Shape * const shape : sim.shapes)
    {
      const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
      const ObstacleBlock*const o = OBLOCK[velInfo[i].blockID];
      if (o == nullptr) continue;

      const Real US = shape->u, VS = shape->v, WS = shape->omega;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

      const CHI_MAT & __restrict__ X = o->chi;
      const auto & __restrict__   V = *(VectorBlock*) vFluidInfo[i].ptrBlock;
      VectorBlock& __restrict__   F = *(VectorBlock*)   tmpVInfo[i].ptrBlock;
      udefLab.load(uDefInfo[i],0); const auto& __restrict__ UDEF = udefLab;
      chiLab .load( chiInfo[i],0); const auto& __restrict__  CHI =  chiLab;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real CHIX = faceX(CHI, ix,iy), CHIY = faceY(CHI, ix,iy);
        const Real CHIXO = (X[iy][ix]+(ix>0? X[iy][ix-1] : CHI(ix-1,iy).s) )/2;
        const Real CHIYO = (X[iy][ix]+(iy>0? X[iy-1][ix] : CHI(ix,iy-1).s) )/2;
        // What if multiple obstacles share a block? Do not penalize with this
        // obstacle if CHI stored on the grid is greater than obstacle's CHI.
        const Real penX= CHIX>CHIXO? 0 : CHIX/dt, penY= CHIY>CHIYO? 0 : CHIY/dt;
        Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
        const Real oldF[2] = {F(ix,iy).u[0], F(ix,iy).u[1]};

        F(ix,iy).u[0] = penX*(US -WS*p[1] +faceX<0>(UDEF,ix,iy) -V(ix,iy).u[0]);
        F(ix,iy).u[1] = penY*(VS +WS*p[0] +faceY<1>(UDEF,ix,iy) -V(ix,iy).u[1]);
        MX+= std::pow(F(ix,iy).u[0],2); DMX+= std::pow(F(ix,iy).u[0]-oldF[0],2);
        MY+= std::pow(F(ix,iy).u[1],2); DMY+= std::pow(F(ix,iy).u[1]-oldF[1],2);
      }
    }
  }

  // return L2 relative momentum
  return std::sqrt((DMX + DMY) / (EPS + MX + MY));
}

void PressureVarRho_approx::updatePressureRHS(const double dt) const
{
  const size_t Nblocks = velInfo.size();

  const Real h = sim.getH(), rho0 = 1/avgInvRho, facDiv = rho0 * h/dt;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& penlInfo = sim.tmpV->getBlocksInfo();
  Real sumPosPrhs = 0, sumNegPrhs = 0, sumPosFdiv = 0, sumNegFdiv = 0;
  #pragma omp parallel reduction(+ : sumPosPrhs,sumNegPrhs,sumPosFdiv,sumNegFdiv)
  {
    static constexpr int stenBegV[3] = { 0, 0, 0}, stenEndV[3] = { 2, 2, 1};
    static constexpr int stenBegP[3] = {-1,-1, 0}, stenEndP[3] = { 2, 2, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBegP, stenEndP, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBegP, stenEndP, 0);
    VectorLab penlLab; penlLab.prepare(*(sim.tmpV),   stenBegV, stenEndV, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab& __restrict__ IRHO= iRhoLab;
      penlLab.load(penlInfo[i],0); const VectorLab& __restrict__ F   = penlLab;
      ScalarBlock& __restrict__ RHS = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ vRHS =*(ScalarBlock*)pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        #if 1
          const Real rE = (1 - rho0 * (IRHO(ix+1,iy).s + IRHO(ix,iy).s)/2 );
          const Real rW = (1 - rho0 * (IRHO(ix-1,iy).s + IRHO(ix,iy).s)/2 );
          const Real rN = (1 - rho0 * (IRHO(ix,iy+1).s + IRHO(ix,iy).s)/2 );
          const Real rS = (1 - rho0 * (IRHO(ix,iy-1).s + IRHO(ix,iy).s)/2 );
          const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
          const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
          const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
        #else
          const Real lapP = (1 - rho0 * IRHO(ix,iy).s) * laplacian(P,ix,iy);
          const Real dPdRx = DcellDx(P,ix,iy) * DcellDx(IRHO,ix,iy);
          const Real dPdRy = DcellDy(P,ix,iy) * DcellDy(IRHO,ix,iy);
          const Real hatPfac = lapP -rho0*(dPdRx+dPdRy);
        #endif
        RHS(ix,iy).s = hatPfac + facDiv*( vRHS(ix,iy).s + dt * div(F,ix,iy));
        if(     hatPfac < 0) sumNegPrhs -= hatPfac;
        else                 sumPosPrhs += hatPfac;
        if(div(F,ix,iy) < 0) sumNegFdiv -= facDiv * div(F,ix,iy);
        else                 sumPosFdiv += facDiv * div(F,ix,iy);
      }
    }
  }

  const Real sumRhsP = sumPosPrhs-sumNegPrhs, sumRhsF = sumPosFdiv-sumNegFdiv;
  const Real corrDenomP = sumRhsP>0 ? sumPosPrhs : sumNegPrhs;
  const Real corrDenomF = sumRhsF>0 ? sumPosFdiv : sumNegFdiv;
  const Real corrP = sumRhsP / std::max(corrDenomP, EPS);
  const Real corrF = sumRhsF / std::max(corrDenomF, EPS);
  //printf("src terms: pressure sum=%e abs=%e, force  sum=%e abs=%e\n",
  //  sumRhsP, corrDenomP, sumRhsP, corrDenomF);

  #pragma omp parallel
  {
    static constexpr int stenBegV[3] = { 0, 0, 0}, stenEndV[3] = { 2, 2, 1};
    static constexpr int stenBegP[3] = {-1,-1, 0}, stenEndP[3] = { 2, 2, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres),   stenBegP, stenEndP, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBegP, stenEndP, 0);
    VectorLab penlLab; penlLab.prepare(*(sim.tmpV),   stenBegV, stenEndV, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const ScalarLab& __restrict__ P   = presLab;
      iRhoLab.load(iRhoInfo[i],0); const ScalarLab& __restrict__ IRHO= iRhoLab;
      penlLab.load(penlInfo[i],0); const VectorLab& __restrict__ F   = penlLab;
      ScalarBlock& __restrict__ RHS = *(ScalarBlock*)  tmpInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        #if 1
          const Real rE = (1 - rho0 * (IRHO(ix+1,iy).s + IRHO(ix,iy).s)/2 );
          const Real rW = (1 - rho0 * (IRHO(ix-1,iy).s + IRHO(ix,iy).s)/2 );
          const Real rN = (1 - rho0 * (IRHO(ix,iy+1).s + IRHO(ix,iy).s)/2 );
          const Real rS = (1 - rho0 * (IRHO(ix,iy-1).s + IRHO(ix,iy).s)/2 );
          const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
          const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
          const Real hatPfac = rE*dE - rW*dW + rN*dN - rS*dS;
        #else
          const Real lapP = (1 - rho0 * IRHO(ix,iy).s) * laplacian(P,ix,iy);
          const Real dPdRx = DcellDx(P,ix,iy) * DcellDx(IRHO,ix,iy);
          const Real dPdRy = DcellDy(P,ix,iy) * DcellDy(IRHO,ix,iy);
          const Real hatPfac = lapP - rho0 * (dPdRx + dPdRy);
        #endif
        const Real divF = facDiv * dt * div(F,ix,iy);
        if      (hatPfac > 0 and corrP > 0) RHS(ix, iy).s -= corrP * hatPfac;
        else if (hatPfac < 0 and corrP < 0) RHS(ix, iy).s += corrP * hatPfac;
        if      (   divF > 0 and corrF > 0) RHS(ix, iy).s -= corrF *    divF;
        else if (   divF < 0 and corrF < 0) RHS(ix, iy).s += corrF *    divF;
      }
    }
  }
}

void PressureVarRho_approx::finalizePressure(const double dt) const
{
  const size_t Nblocks = velInfo.size();

  const Real h = sim.getH(), pFac = -dt/h;
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1,0}, stenEnd[3] = {1,1,1};
    ScalarLab  tmpLab;  tmpLab.prepare(*(sim.tmp ), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
       tmpLab.load( tmpInfo[i],0); const ScalarLab&__restrict__ Pcur =  tmpLab;
      presLab.load(presInfo[i],0); const ScalarLab&__restrict__ Pold = presLab;
            VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;
      const VectorBlock&__restrict__   F = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      const ScalarBlock&__restrict__ IRHO= *(ScalarBlock*) iRhoInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real dUpre = DfaceDx(Pcur,ix,iy) * avgInvRho;
        const Real dVpre = DfaceDy(Pcur,ix,iy) * avgInvRho;
        const Real dUdiv = DfaceDx(Pold,ix,iy) * (IRHO(ix,iy).s-avgInvRho);
        const Real dVdiv = DfaceDy(Pold,ix,iy) * (IRHO(ix,iy).s-avgInvRho);
        V(ix,iy).u[0] += dt * F(ix,iy).u[0] + pFac * (dUpre + dUdiv);
        V(ix,iy).u[1] += dt * F(ix,iy).u[1] + pFac * (dVpre + dVdiv);
      }
    }
  }
}

void PressureVarRho_approx::operator()(const double dt)
{
  const size_t Nblocks = velInfo.size();

  // first copy velocity before either Pres or Penal onto tmpV
  const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();

  pressureCorrectionInit(dt);

  int iter = 0;
  Real relDF = 1e3, relDP = 1e3;//, oldErr = 1e3;
  bool bConverged = false;
  for(iter = 0; ; iter++)
  {
    sim.startProfiler("Obj_force");
    for(Shape * const shape : sim.shapes)
    {
      // integrate vel in velocity after PP
      integrateMomenta(shape);
      shape->updateVelocity(dt);
    }

     // finally update vel with penalization but without pressure
    relDF = penalize(dt, iter);
    sim.stopProfiler();

    // pressure solver is going to use as RHS = div VEL - \chi div UDEF
    sim.startProfiler("Prhs");
    updatePressureRHS(dt);
    sim.stopProfiler();

    pressureSolver->solve(tmpInfo, tmpInfo);

    sim.startProfiler("PCorrect");
    relDP = pressureCorrection(dt);
    sim.stopProfiler();

    printf("iter:%02d - rel. err penal:%e press:%e\n", iter, relDF, relDP);
    const Real newErr = std::max(relDF, relDP);
    bConverged = newErr<targetRelError || iter>2*oldNsteps  || iter>100;

    if(bConverged)
    {
      sim.startProfiler("PCorrect");
      finalizePressure(dt);
      sim.stopProfiler();
    }

    //size_t nExtra=0, nInter=0;
    #pragma omp parallel for schedule(static) //reduction(+ : nExtra, nInter)
    for (size_t i=0; i < Nblocks; i++)
    {
      const auto& __restrict__ Pnew = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
            auto& __restrict__ Pcur = *(ScalarBlock*) presInfo[i].ptrBlock;
            auto& __restrict__ Pold = *(ScalarBlock*) pOldInfo[i].ptrBlock;
      Pold.copy(Pcur);
      Pcur.copy(Pnew);
    }
    //printf("nInter:%lu nExtra:%lu\n",nInter,nExtra);
    if(bConverged) break;
  }

  oldNsteps = iter+1;
  if(oldNsteps > 30) targetRelError = std::max({relDF,relDP,targetRelError});
  if(oldNsteps > 10) targetRelError *= 1.01;
  if(oldNsteps <= 2) targetRelError *= 0.99;
  targetRelError = std::min((Real)1e-3, std::max((Real)1e-5, targetRelError));

  if(not sim.muteAll)
  {
    std::stringstream ssF; ssF<<sim.path2file<<"/pressureIterStats.dat";
    std::ofstream pfile(ssF.str().c_str(), std::ofstream::app);
    if(sim.step==0) pfile<<"step time dt iter relDF"<<"\n";
    pfile<<sim.step<<" "<<sim.time<<" "<<sim.dt<<" "<<iter<<" "<<relDF<<"\n";
  }
}

PressureVarRho_approx::PressureVarRho_approx(SimulationData& s) :
  Operator(s), pressureSolver( PoissonSolver::makeSolver(s) )  { }

PressureVarRho_approx::~PressureVarRho_approx() {
    delete pressureSolver;
}
