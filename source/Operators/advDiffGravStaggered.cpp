//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiffGravStaggered.h"

using namespace cubism;
#define THIRD_ORDER_UPWIND

static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;

#ifdef THIRD_ORDER_UPWIND
  static constexpr int stencilBeg = -2, stencilEnd = 3;
  const int loopBeg = stencilBeg, loopEnd = BSX-1 + stencilBeg;
#else
  static constexpr int stencilBeg = -1, stencilEnd = 2;
  const int loopBeg = stencilBeg, loopEnd = BSY-1 + stencilBeg;
#endif

static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 1, 1, 1};
static constexpr int stenBegV[3] = {stencilBeg, stencilBeg, 0};
static constexpr int stenEndV[3] = {stencilEnd, stencilEnd, 1};

void advDiffGravStaggered::operator()(const double dt)
{
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& iRhoInfo = sim.invRho->getBlocksInfo();
  const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0; };
  const auto isE = [&](const BlockInfo&I) { return I.index[0] == sim.bpdx-1; };
  const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0; };
  const auto isN = [&](const BlockInfo&I) { return I.index[1] == sim.bpdy-1; };

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++) {
    VectorBlock & V = *(VectorBlock*) velInfo[i].ptrBlock;
    if(isE(velInfo[i])) for(int y=0; y<BSY; ++y) V(EX,y).u[1] = V(EX-1,y).u[1];
    if(isN(velInfo[i])) for(int x=0; x<BSX; ++x) V(x,EY).u[0] = V(x,EY-1).u[0];
  }

  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  const Real G[2]= { (Real) dt * sim.gravity[0], (Real) dt * sim.gravity[1] };

  const Real bcfac = std::min((Real)1, sim.uMax_measured * dt / h);
  const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
  const Real fadeW= 1 - bcfac * std::pow(std::max(UINF[0], (Real)0)/norUinf, 2);
  const Real fadeS= 1 - bcfac * std::pow(std::max(UINF[1], (Real)0)/norUinf, 2);
  const Real fadeE= 1 - bcfac * std::pow(std::min(UINF[0], (Real)0)/norUinf, 2);
  const Real fadeN= 1 - bcfac * std::pow(std::min(UINF[1], (Real)0)/norUinf, 2);
  const auto fade = [&](VectorElement&B,const Real F) { B.u[0]*=F; B.u[1]*=F; };

  sim.startProfiler("advDiffGrav");
  #ifdef THIRD_ORDER_UPWIND
    const Real dfac = (sim.nu/h)*(dt/h), afac = -dt/h/6;
  #else
    const Real dfac = (sim.nu/h)*(dt/h), afac = -dt/h/2;
  #endif

  #pragma omp parallel
  {
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBegV, stenEndV, 0);
    ScalarLab rholab; rholab.prepare(*(sim.invRho), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; ++i)
    {
      vellab.load( velInfo[i], 0); auto & __restrict__ V = vellab;
      rholab.load(iRhoInfo[i], 0); auto & __restrict__ IRHO = rholab;
      auto& __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      if (isW(velInfo[i])) for (int iy = loopBeg; iy < loopEnd; ++iy)
      for (int ix = loopBeg; ix < 0; ++ix) fade(V(ix,iy), fadeW);

      if (isE(velInfo[i])) for (int iy = loopBeg; iy < loopEnd; ++iy)
      for (int ix = BSX; ix < loopEnd; ++ix) fade(V(ix,iy), fadeE);

      if (isS(velInfo[i])) for (int iy = loopBeg; iy < 0; ++iy)
      for (int ix = loopBeg; ix < loopEnd; ++ix) fade(V(ix,iy), fadeS);

      if (isN(velInfo[i])) for (int iy = BSY; iy < loopEnd; ++iy)
      for (int ix = loopBeg; ix < loopEnd; ++ix) fade(V(ix,iy), fadeN);

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real gravFacU = G[0] * ( 1 - (IRHO(ix-1,iy).s+IRHO(ix,iy).s)/2 );
        const Real gravFacV = G[1] * ( 1 - (IRHO(ix,iy-1).s+IRHO(ix,iy).s)/2 );

        const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
        const Real up1x = V(ix+1, iy).u[0], up1y = V(ix, iy+1).u[0];
        const Real ul1x = V(ix-1, iy).u[0], ul1y = V(ix, iy-1).u[0];
        const Real vp1x = V(ix+1, iy).u[1], vp1y = V(ix, iy+1).u[1];
        const Real vl1x = V(ix-1, iy).u[1], vl1y = V(ix, iy-1).u[1];
        #ifdef THIRD_ORDER_UPWIND
          const Real up2x = V(ix+2, iy).u[0], up2y = V(ix, iy+2).u[0];
          const Real ul2x = V(ix-2, iy).u[0], ul2y = V(ix, iy-2).u[0];
          const Real vp2x = V(ix+2, iy).u[1], vp2y = V(ix, iy+2).u[1];
          const Real vl2x = V(ix-2, iy).u[1], vl2y = V(ix, iy-2).u[1];
        #endif

        // advection V at x (U) faces and advection U at y (V) faces:
        const Real VadvU = (vp1y + V(ix-1,iy+1).u[1] + vcc + vl1x)/4 + UINF[1];
        const Real UadvV = (up1x + V(ix+1,iy-1).u[0] + ucc + ul1y)/4 + UINF[0];
        const Real UadvU = ucc + UINF[0], VadvV = vcc + UINF[1];

        #ifdef THIRD_ORDER_UPWIND
          const Real dudx = UadvU>0 ?          2*up1x + 3*ucc - 6*ul1x + ul2x
                                    : - up2x + 6*up1x - 3*ucc - 2*ul1x;
          const Real dvdx = UadvV>0 ?          2*vp1x + 3*vcc - 6*vl1x + vl2x
                                    : - vp2x + 6*vp1x - 3*vcc - 2*vl1x;
          const Real dudy = VadvU>0 ?          2*up1y + 3*ucc - 6*ul1y + ul2y
                                    : - up2y + 6*up1y - 3*ucc - 2*ul1y;
          const Real dvdy = VadvV>0 ?          2*vp1y + 3*vcc - 6*vl1y + vl2y
                                    : - vp2y + 6*vp1y - 3*vcc - 2*vl1y;
        #else
          const Real dudx = up1x - ul1x, dudy = up1y - ul1y;
          const Real dvdx = vp1x - vl1x, dvdy = vp1y - vl1y;
        #endif

        const Real dUadv = UadvU * dudx + VadvU * dudy;
        const Real dVadv = UadvV * dvdx + VadvV * dvdy;
        const Real dUdif = up1x + up1y + ul1x + ul1y - 4 * ucc;
        const Real dVdif = vp1x + vp1y + vl1x + vl1y - 4 * vcc;
        TMP(ix,iy).u[0] = V(ix,iy).u[0] + afac*dUadv + dfac*dUdif + gravFacU;
        TMP(ix,iy).u[1] = V(ix,iy).u[1] + afac*dVadv + dfac*dVdif + gravFacV;
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; ++i) {
          VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    V.copy(T);
  }

  if (1)
  {
    ////////////////////////////////////////////////////////////////////////////
    Real IF = 0, AF = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+ : IF, AF)
    for (size_t i=0; i < Nblocks; i++) {
      VectorBlock& V = *(VectorBlock*)  velInfo[i].ptrBlock;
      for(int iy=0; iy<BSY && isW(velInfo[i]); ++iy) {
        IF -= V(BX,iy).u[0]; AF += std::fabs(V(BX,iy).u[0]);
      }
      for(int iy=0; iy<BSY && isE(velInfo[i]); ++iy) {
        IF += V(EX,iy).u[0]; AF += std::fabs(V(EX,iy).u[0]);
      }
      for(int ix=0; ix<BSX && isS(velInfo[i]); ++ix) {
        IF -= V(ix,BY).u[1]; AF += std::fabs(V(ix,BY).u[1]);
      }
      for(int ix=0; ix<BSX && isN(velInfo[i]); ++ix) {
        IF += V(ix,EY).u[1]; AF += std::fabs(V(ix,EY).u[1]);
      }
      if( isN(velInfo[i]) && isW(velInfo[i]) ) {
        IF += V(BX,EY).u[0]; AF -= std::fabs(V(BX,EY).u[0]);
      }
      if( isN(velInfo[i]) && isE(velInfo[i]) ) {
        IF -= V(EX,EY).u[0]; AF -= std::fabs(V(EX,EY).u[0]);
        IF -= V(EX,EY).u[1]; AF -= std::fabs(V(EX,EY).u[1]);
      }
      if( isS(velInfo[i]) && isE(velInfo[i]) ) {
        IF += V(EX,BY).u[1]; AF -= std::fabs(V(EX,BY).u[1]);
      }
    }
    ////////////////////////////////////////////////////////////////////////////
    //const Real corr = IF/std::max(AF, EPS);
    const Real corr = IF/( 2*(BSY*sim.bpdy -1) + 2*(BSX*sim.bpdx -1) );
    //printf("Relative inflow correction %e\n",corr);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i < Nblocks; i++) {
      VectorBlock& V = *(VectorBlock*)  velInfo[i].ptrBlock;
      for(int iy=0; iy<BSY && isW(velInfo[i]); ++iy)
        V(BX,iy).u[0] += corr ;//* std::fabs(V(BX,iy).u[0]);
      for(int iy=0; iy<BSY && isE(velInfo[i]); ++iy)
        V(EX,iy).u[0] -= corr ;//* std::fabs(V(EX,iy).u[0]);
      for(int ix=0; ix<BSX && isS(velInfo[i]); ++ix)
        V(ix,BY).u[1] += corr ;//* std::fabs(V(ix,BY).u[1]);
      for(int ix=0; ix<BSX && isN(velInfo[i]); ++ix)
        V(ix,EY).u[1] -= corr ;//* std::fabs(V(ix,EY).u[1]);
    }
  }

  sim.stopProfiler();
}

/*
  Real MX[2] = {0}, AX[2] = {0}, MY[2] = {0}, AY[2] = {0};
  #pragma omp parallel for schedule(dynamic) reduction(+ : MX[:2], AX[:2], \
                                                           MY[:2], AY[:2])
  for (size_t i=0; i < Nblocks; i++) {
    VectorBlock& V = *(VectorBlock*)  velInfo[i].ptrBlock;
    // handle corners outside of staggered grid, will be overwritten later
    if(isN(velInfo[i]) && isW(velInfo[i])) V(BX,EY).u[0] = 0;
    if(isN(velInfo[i]) && isE(velInfo[i])) { V(EX,EY).u[0]=0; V(EX,EY).u[1]=0; }
    if(isS(velInfo[i]) && isE(velInfo[i])) V(EX,BY).u[1] = 0;

    for(int iy=0; iy<BSY && isW(velInfo[i]); ++iy) {
      MX[0] += V(BX,iy).u[0]; AX[0] += std::fabs( V(BX,iy).u[0] );
      MX[1] += V(BX,iy).u[1]; AX[1] += std::fabs( V(BX,iy).u[1] );
    }
    for(int iy=0; iy<BSY && isE(velInfo[i]); ++iy) {
      MX[0] += V(EX,iy).u[0]; AX[0] += std::fabs( V(EX,iy).u[0] );
      MX[1] += V(EX,iy).u[1]; AX[1] += std::fabs( V(EX,iy).u[1] );
    }
    for(int ix=0; ix<BSX && isS(velInfo[i]); ++ix) {
      MY[0] += V(ix,BY).u[0]; AY[0] += std::fabs( V(ix,BY).u[0] );
      MY[1] += V(ix,BY).u[1]; AY[1] += std::fabs( V(ix,BY).u[1] );
    }
    for(int ix=0; ix<BSX && isN(velInfo[i]); ++ix) {
      MY[0] += V(ix,EY).u[0]; AY[0] += std::fabs( V(ix,EY).u[0] );
      MY[1] += V(ix,EY).u[1]; AY[1] += std::fabs( V(ix,EY).u[1] );
    }
  }
  AX[0] = std::max(EPS, AX[0]); AX[1] = std::max(EPS, AX[1]);
  AY[0] = std::max(EPS, AY[0]); AY[1] = std::max(EPS, AY[1]);
  const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
  const Real fadeW = std::pow( std::max(UINF[0],(Real) 0) / norUinf, 2);
  const Real fadeS = std::pow( std::max(UINF[1],(Real) 0) / norUinf, 2);
  const Real fadeE = std::pow( std::min(UINF[0],(Real) 0) / norUinf, 2);
  const Real fadeN = std::pow( std::min(UINF[1],(Real) 0) / norUinf, 2);
  const Real fadeXW = fadeW * MX[0] / AX[0], fadeYW = fadeW * MX[1] / AX[1];
  const Real fadeXE = fadeE * MX[0] / AX[0], fadeYE = fadeE * MX[1] / AX[1];
  const Real fadeXS = fadeS * MY[0] / AY[0], fadeYS = fadeS * MY[1] / AY[1];
  const Real fadeXN = fadeN * MY[0] / AY[0], fadeYN = fadeN * MY[1] / AY[1];
  printf("fade %e %e - %e %e - %e %e - %e %e\n",fadeXW,fadeYW,fadeXE,fadeYE,fadeXS,fadeYS,fadeXN,fadeYN);
  const auto fade = [&] (VectorElement&B, const Real fadeX, const Real fadeY) {
    B.u[0] -= fadeX * std::fabs(B.u[0]); B.u[1] -= fadeY * std::fabs(B.u[1]);
  };

  for(int iy=-1; iy<=BSY && isW(velInfo[i]); ++iy)
    fade(V(BX-1,iy), fadeXW, fadeYW);

  for(int ix=-1; ix<=BSX && isS(velInfo[i]); ++ix)
    fade(V(ix,BY-1), fadeXS, fadeYS);

  for(int iy=-1; iy<=BSY && isE(velInfo[i]); ++iy)
    fade(V(EX+1,iy), fadeXE, fadeYE);

  for(int ix=-1; ix<=BSX && isN(velInfo[i]); ++ix)
    fade(V(ix,EY+1), fadeXN, fadeYN);
*/
