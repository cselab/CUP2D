//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureVarRho_proper.h"
#ifdef AMGX_POISSON
#include "../Poisson/AMGXdirichletVarRho.h"
#else
#include "../Poisson/HYPREdirichletVarRho.h"
#endif

using namespace cubism;
static constexpr double EPS = std::numeric_limits<double>::epsilon();

Real PressureVarRho_proper::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = h/dt;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>&  chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>&  tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& pRhsInfo = sim.pRHS->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const auto isE = [&](const BlockInfo&I) { return I.index[0] == sim.bpdx-1; };
  const auto isN = [&](const BlockInfo&I) { return I.index[1] == sim.bpdy-1; };
  const auto& S = * varRhoSolver;

  Real maxDiffMat = 0;
  #pragma omp parallel reduction(max : maxDiffMat)
  {
    static constexpr int stenBegV[3] = { 0, 0, 0}, stenEndV[3] = { 2, 2, 1};
    static constexpr int stenBeg [3] = {-1,-1, 0}, stenEnd [3] = { 2, 2, 1};
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);
    VectorLab velLab;   velLab.prepare(*(sim.vel), stenBegV, stenEndV, 0);
    VectorLab uDefLab; uDefLab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      const size_t blocki = VectorBlock::sizeX * velInfo[i].index[0];
      const size_t blockj = VectorBlock::sizeY * velInfo[i].index[1];

       velLab.load( velInfo[i], 0); const auto & __restrict__ V   = velLab;
      presLab.load(presInfo[i], 0); const auto & __restrict__ P   = presLab;
      uDefLab.load(uDefInfo[i], 0); const auto & __restrict__ UDEF= uDefLab;
      iRhoLab.load(iRhoInfo[i], 0); const auto & __restrict__ IRHO= iRhoLab;
      const auto& __restrict__ CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;
            auto& __restrict__ TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
            auto& __restrict__ RHS = *(ScalarBlock*) pRhsInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real divUx  =    V(ix+1,iy).u[0] -    V(ix,iy).u[0];
        const Real divVy  =    V(ix,iy+1).u[1] -    V(ix,iy).u[1];
        const Real UDEFW = (UDEF(ix  ,iy).u[0] + UDEF(ix-1,iy).u[0]) / 2;
        const Real UDEFE = (UDEF(ix+1,iy).u[0] + UDEF(ix  ,iy).u[0]) / 2;
        const Real VDEFS = (UDEF(ix,iy  ).u[1] + UDEF(ix,iy-1).u[1]) / 2;
        const Real VDEFN = (UDEF(ix,iy+1).u[1] + UDEF(ix,iy  ).u[1]) / 2;
        const Real divUS = UDEFE - UDEFW + VDEFN - VDEFS;
        TMP(ix, iy).s = facDiv*(divUx+divVy - CHI(ix,iy).s*divUS);

        const Real rE = (IRHO(ix+1,iy).s + IRHO(ix,iy).s)/2;
        const Real rW = (IRHO(ix-1,iy).s + IRHO(ix,iy).s)/2;
        const Real rN = (IRHO(ix,iy+1).s + IRHO(ix,iy).s)/2;
        const Real rS = (IRHO(ix,iy-1).s + IRHO(ix,iy).s)/2;
        const Real dN = P(ix,iy+1).s-P(ix,iy).s, dS = P(ix,iy).s-P(ix,iy-1).s;
        const Real dE = P(ix+1,iy).s-P(ix,iy).s, dW = P(ix,iy).s-P(ix-1,iy).s;
        RHS(ix,iy).s = TMP(ix,iy).s +(1-rE)*dE -(1-rW)*dW +(1-rN)*dN -(1-rS)*dS;
        const size_t idx = blocki + ix, idy = blockj + iy;
        const Real dMat = S.updateMat(idx,idy, -rN -rS -rE -rW, rW, rE, rS, rN);
        maxDiffMat = std::max(maxDiffMat, dMat);
      }


      for(int iy=0; iy<VectorBlock::sizeY && isE(velInfo[i]); ++iy) {
        TMP(VectorBlock::sizeX-1, iy).s = 0;
        RHS(VectorBlock::sizeX-1, iy).s = 0;
      }
      for(int ix=0; ix<VectorBlock::sizeX && isN(velInfo[i]); ++ix) {
        TMP(ix, VectorBlock::sizeY-1).s = 0;
        RHS(ix, VectorBlock::sizeY-1).s = 0;
      }
    }
  }
  return maxDiffMat;
}

void PressureVarRho_proper::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -dt/h;
  const std::vector<BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 1, 1, 1};
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);
    ScalarLab iRhoLab; iRhoLab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      presLab.load(presInfo[i],0); const auto &__restrict__ P = presLab;
      iRhoLab.load(iRhoInfo[i],0); const auto &__restrict__ IRHO = iRhoLab;
      auto& __restrict__ V = *(VectorBlock*) velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        // update vel field after most recent force and pressure response:
        const Real IRHOX = (IRHO(ix,iy).s + IRHO(ix-1,iy).s)/2;
        const Real IRHOY = (IRHO(ix,iy).s + IRHO(ix,iy-1).s)/2;
        V(ix,iy).u[0] += pFac * IRHOX * (P(ix,iy).s - P(ix-1,iy).s);
        V(ix,iy).u[1] += pFac * IRHOY * (P(ix,iy).s - P(ix,iy-1).s);
      }
    }
  }
}

void PressureVarRho_proper::operator()(const double dt)
{
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& rhsInfo  = sim.pRHS->getBlocksInfo();

  if(sim.step < 20) {
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
      auto& __restrict__ Pcur = *(ScalarBlock*) presInfo[i].ptrBlock;
      auto& __restrict__ Pold = *(ScalarBlock*) pOldInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) Pold(ix,iy).s = Pcur(ix,iy).s;
    }
  }

  sim.startProfiler("Prhs");
  const Real maxDiffMat = updatePressureRHS(dt);
  sim.stopProfiler();

  if(sim.step < 20) {
    const Real fac = 1 - sim.step / 20.0;
    unifRhoSolver->solve(rhsInfo, presInfo);
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      const std::vector<BlockInfo>& pOldInfo = sim.pOld->getBlocksInfo();
      auto& __restrict__ Pcur = *(ScalarBlock*) presInfo[i].ptrBlock;
      auto& __restrict__ Pold = *(ScalarBlock*) pOldInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        Pcur(ix,iy).s += fac * (Pold(ix,iy).s - Pcur(ix,iy).s);
    }
  }

  #ifdef HYPREFFT
    const std::vector<BlockInfo>& tmpInfo  = sim.tmp->getBlocksInfo();
    varRhoSolver->bUpdateMat = maxDiffMat > EPS;
    varRhoSolver->solve(tmpInfo, presInfo);
    //pressureSolver->bUpdateMat = false;
  #else
    printf("Class PressureVarRho_proper REQUIRES HYPRE\n");
    fflush(0); abort();
  #endif

  sim.startProfiler("PCorrect");
  pressureCorrection(dt);
  sim.stopProfiler();
}

PressureVarRho_proper::PressureVarRho_proper(SimulationData& s) :
  Operator(s), unifRhoSolver( PoissonSolver::makeSolver(s) ),
#ifdef AMGX_POISSON
  varRhoSolver( new AMGXdirichletVarRho(s) ) { }
#else
  varRhoSolver( new HYPREdirichletVarRho(s) ) { }
#endif

PressureVarRho_proper::~PressureVarRho_proper() {
    delete varRhoSolver;
    delete unifRhoSolver;
}
