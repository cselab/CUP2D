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
    const Real upx = V(ix+1, iy).u[0], upy = V(ix, iy+1).u[0];
    const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
    const Real ulx = V(ix-1, iy).u[0], uly = V(ix, iy-1).u[0];
    const Real dUadv = (ucc+uinf[0]) * (upx-ulx) + (vcc+uinf[1]) * (upy-uly);
    const Real dUdif = upx + upy + ulx + uly - 4 *ucc;
    return advF * dUadv + difF * dUdif;
}

static inline Real dV_adv_dif(const VectorLab&V, const Real uinf[2], const Real advF, const Real difF, const int ix, const int iy)
{
    const Real vpx = V(ix+1, iy).u[1], vpy = V(ix, iy+1).u[1];
    const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
    const Real vlx = V(ix-1, iy).u[1], vly = V(ix, iy-1).u[1];
    const Real dVadv = (ucc+uinf[0]) * (vpx-vlx) + (vcc+uinf[1]) * (vpy-vly);
    const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;
    return advF * dVadv + difF * dVdif;
}

void advDiff::operator()(const double dt)
{
    sim.startProfiler("advDiff");

    static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
    static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
    static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;

    const size_t Nblocks = velInfo.size();
    const Real UINF[2]= {sim.uinfx, sim.uinfy};
    const auto fade = [&](VectorElement&B,const Real F) { B.u[0]*=F; B.u[1]*=F; };

    #pragma omp parallel
    {
        static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
        VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);
        #pragma omp for schedule(static)
        for (size_t i=0; i < Nblocks; i++)
        {
            int aux = 1<<velInfo[i].level;
            const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0;          };
            const auto isE = [&](const BlockInfo&I) { return I.index[0] == aux*sim.bpdx-1; };
            const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0;          };
            const auto isN = [&](const BlockInfo&I) { return I.index[1] == aux*sim.bpdy-1; };

            const Real h = velInfo[i].h_gridpoint;
            const Real dfac = (sim.nu/h)*(dt/h), afac = -0.5*dt/h;
            const Real fac = std::min((Real)1, sim.uMax_measured * dt / h);
            const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
            const Real fadeW= 1 - fac * std::pow(std::max(UINF[0], (Real)0)/norUinf, 2);
            const Real fadeS= 1 - fac * std::pow(std::max(UINF[1], (Real)0)/norUinf, 2);
            const Real fadeE= 1 - fac * std::pow(std::min(UINF[0], (Real)0)/norUinf, 2);
            const Real fadeN= 1 - fac * std::pow(std::min(UINF[1], (Real)0)/norUinf, 2);

            vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
            VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

            if(isW(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(BX-1,iy), fadeW);
            if(isS(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,BY-1), fadeS);
            if(isE(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(EX+1,iy), fadeE);
            if(isN(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,EY+1), fadeN);

            for(int iy=0; iy<BSY; ++iy) for(int ix=0; ix<BSX; ++ix)
            {
              TMP(ix,iy).u[0] = V(ix,iy).u[0] + dU_adv_dif(V,UINF,afac,dfac,ix,iy);
              TMP(ix,iy).u[1] = V(ix,iy).u[1] + dV_adv_dif(V,UINF,afac,dfac,ix,iy);
            }
        }
    }

    // Copy TMP to V and store inflow correction
    Real IF = 0.0;
    #pragma omp parallel for schedule(static) reduction(+ : IF)
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
        const double h = velInfo[i].h_gridpoint;
        if (isW) for(int iy=0; iy<BSY; ++iy) IF += - h * V(BX,iy).u[0];
        if (isE) for(int iy=0; iy<BSY; ++iy) IF +=   h * V(EX,iy).u[0];
        if (isS) for(int ix=0; ix<BSX; ++ix) IF += - h * V(ix,BY).u[1];
        if (isN) for(int ix=0; ix<BSX; ++ix) IF +=   h * V(ix,EY).u[1];
    }

    const double H = sim.getH();
    const Real corr = IF/H/( 2*BSY*sim.bpdy*(1<<(sim.levelMax-1)) + 2*BSX*sim.bpdx*(1<<(sim.levelMax-1)) );

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
        if(isW) for(int iy=0; iy<BSY; ++iy) V(BX,iy).u[0] += corr;
        if(isE) for(int iy=0; iy<BSY; ++iy) V(EX,iy).u[0] -= corr;
        if(isS) for(int ix=0; ix<BSX; ++ix) V(ix,BY).u[1] += corr;
        if(isN) for(int ix=0; ix<BSX; ++ix) V(ix,EY).u[1] -= corr;
    }
    sim.stopProfiler();
}

#if 0 //advection-diffusion in conservation form (with flux corrections) - no need to use!

#include "advDiff.h"
#include "Cubism/FluxCorrection.h"  

using namespace cubism;

static inline Real dU(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  const Real u_ip1 = 0.5*(V(ix,iy).u[0] + V(ix+1,iy).u[0]) + uinf[0];
  const Real u_im1 = 0.5*(V(ix,iy).u[0] + V(ix-1,iy).u[0]) + uinf[0];

  const Real u_jp1 = 0.5*(V(ix,iy).u[0] + V(ix,iy+1).u[0]) + uinf[0];
  const Real u_jm1 = 0.5*(V(ix,iy).u[0] + V(ix,iy-1).u[0]) + uinf[0];
  const Real v_jp1 = 0.5*(V(ix,iy).u[1] + V(ix,iy+1).u[1]) + uinf[1];
  const Real v_jm1 = 0.5*(V(ix,iy).u[1] + V(ix,iy-1).u[1]) + uinf[1];

  const Real upx = V(ix+1, iy).u[0], upy = V(ix, iy+1).u[0];
  const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
  const Real ulx = V(ix-1, iy).u[0], uly = V(ix, iy-1).u[0];

  const Real dUadv = (u_ip1*u_ip1 - u_im1*u_im1) + (u_jp1*v_jp1 - u_jm1*v_jm1);
  const Real dUdif = upx + upy + ulx + uly - 4 *ucc;
  return advF * dUadv + difF * dUdif;
}

static inline Real dV(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  const Real v_jp1 = 0.5*(V(ix,iy).u[1] + V(ix,iy+1).u[1]) + uinf[1];
  const Real v_jm1 = 0.5*(V(ix,iy).u[1] + V(ix,iy-1).u[1]) + uinf[1];

  const Real u_ip1 = 0.5*(V(ix,iy).u[0] + V(ix+1,iy).u[0]) + uinf[0];
  const Real u_im1 = 0.5*(V(ix,iy).u[0] + V(ix-1,iy).u[0]) + uinf[0];
  const Real v_ip1 = 0.5*(V(ix,iy).u[1] + V(ix+1,iy).u[1]) + uinf[1];
  const Real v_im1 = 0.5*(V(ix,iy).u[1] + V(ix-1,iy).u[1]) + uinf[1];

  const Real vpx = V(ix+1, iy).u[1], vpy = V(ix, iy+1).u[1];
  const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
  const Real vlx = V(ix-1, iy).u[1], vly = V(ix, iy-1).u[1];

  const Real dVadv = (u_ip1*v_ip1 - u_im1*v_im1) + (v_jp1*v_jp1 - v_jm1*v_jm1);
  const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;
  return advF * dVadv + difF * dVdif;
}

void advDiff::operator()(const double dt)
{
  sim.startProfiler("advDiff");

  cubism::FluxCorrection<VectorGrid,VectorBlock> Corrector;
  Corrector.prepare(*sim.tmpV);

  const size_t Nblocks = velInfo.size();
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;

  const Real UINF[2]= {sim.uinfx, sim.uinfy};
  const Real uinf[2]= {sim.uinfx, sim.uinfy};
  const auto fade = [&](VectorElement&B,const Real F) { B.u[0]*=F; B.u[1]*=F; };

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);

    #pragma omp for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
      int aux = 1<<velInfo[i].level;
      const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0;          };
      const auto isE = [&](const BlockInfo&I) { return I.index[0] == aux*sim.bpdx-1; };
      const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0;          };
      const auto isN = [&](const BlockInfo&I) { return I.index[1] == aux*sim.bpdy-1; };

      const Real h = velInfo[i].h_gridpoint;
      const Real dfac = (sim.nu)*(dt), afac = -dt*h;
      const Real fac = 1.0 ;//std::min((Real)1, sim.uMax_measured * dt / h);
      const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
      const Real fadeW= 1 - fac * std::pow(std::max(UINF[0], (Real)0)/norUinf, 2);
      const Real fadeS= 1 - fac * std::pow(std::max(UINF[1], (Real)0)/norUinf, 2);
      const Real fadeE= 1 - fac * std::pow(std::min(UINF[0], (Real)0)/norUinf, 2);
      const Real fadeN= 1 - fac * std::pow(std::min(UINF[1], (Real)0)/norUinf, 2);

      vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      if(isW(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(BX-1,iy), fadeW);
      if(isS(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,BY-1), fadeS);
      if(isE(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(EX+1,iy), fadeE);
      if(isN(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,EY+1), fadeN);

      for(int iy=0; iy<BSY; ++iy) for(int ix=0; ix<BSX; ++ix)
      {
        TMP(ix,iy).u[0] = dU(V,UINF,afac,dfac,ix,iy);
        TMP(ix,iy).u[1] = dV(V,UINF,afac,dfac,ix,iy);
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
                for(int iy=0; iy<BSY; ++iy)
                {
                  const Real u_ip1 = 0.5*(V(ix,iy).u[0] + V(ix+1,iy).u[0]) + uinf[0];
                  const Real u_im1 = 0.5*(V(ix,iy).u[0] + V(ix-1,iy).u[0]) + uinf[0];
                  const Real v_ip1 = 0.5*(V(ix,iy).u[1] + V(ix+1,iy).u[1]) + uinf[1];
                  const Real v_im1 = 0.5*(V(ix,iy).u[1] + V(ix-1,iy).u[1]) + uinf[1];
                  const Real u_jp1 = 0.5*(V(ix,iy).u[0] + V(ix,iy+1).u[0]) + uinf[0];
                  const Real u_jm1 = 0.5*(V(ix,iy).u[0] + V(ix,iy-1).u[0]) + uinf[0];
                  const Real v_jp1 = 0.5*(V(ix,iy).u[1] + V(ix,iy+1).u[1]) + uinf[1];
                  const Real v_jm1 = 0.5*(V(ix,iy).u[1] + V(ix,iy-1).u[1]) + uinf[1];
                  faceXm[iy] = dfac*(V(ix,iy) - V(ix-1,iy));
                  faceXm[iy].u[0] -= afac*(-u_im1*u_im1);
                  faceXm[iy].u[1] -= afac*(-u_im1*v_im1);
                }
              }
              if (faceXp != nullptr)
              {
                int ix = BSX-1;
                for(int iy=0; iy<BSY; ++iy)
                {
                  const Real u_ip1 = 0.5*(V(ix,iy).u[0] + V(ix+1,iy).u[0]) + uinf[0];
                  const Real u_im1 = 0.5*(V(ix,iy).u[0] + V(ix-1,iy).u[0]) + uinf[0];
                  const Real v_ip1 = 0.5*(V(ix,iy).u[1] + V(ix+1,iy).u[1]) + uinf[1];
                  const Real v_im1 = 0.5*(V(ix,iy).u[1] + V(ix-1,iy).u[1]) + uinf[1];
                  const Real u_jp1 = 0.5*(V(ix,iy).u[0] + V(ix,iy+1).u[0]) + uinf[0];
                  const Real u_jm1 = 0.5*(V(ix,iy).u[0] + V(ix,iy-1).u[0]) + uinf[0];
                  const Real v_jp1 = 0.5*(V(ix,iy).u[1] + V(ix,iy+1).u[1]) + uinf[1];
                  const Real v_jm1 = 0.5*(V(ix,iy).u[1] + V(ix,iy-1).u[1]) + uinf[1];
                  faceXp[iy] = dfac*(V(ix,iy) - V(ix+1,iy));
                  faceXp[iy].u[0] -= afac*(u_ip1*u_ip1);
                  faceXp[iy].u[1] -= afac*(u_ip1*v_ip1);
                }
              }
              if (faceYm != nullptr)
              {
                int iy = 0;
                for(int ix=0; ix<BSX; ++ix)
                {
                  const Real u_ip1 = 0.5*(V(ix,iy).u[0] + V(ix+1,iy).u[0]) + uinf[0];
                  const Real u_im1 = 0.5*(V(ix,iy).u[0] + V(ix-1,iy).u[0]) + uinf[0];
                  const Real v_ip1 = 0.5*(V(ix,iy).u[1] + V(ix+1,iy).u[1]) + uinf[1];
                  const Real v_im1 = 0.5*(V(ix,iy).u[1] + V(ix-1,iy).u[1]) + uinf[1];
                  const Real u_jp1 = 0.5*(V(ix,iy).u[0] + V(ix,iy+1).u[0]) + uinf[0];
                  const Real u_jm1 = 0.5*(V(ix,iy).u[0] + V(ix,iy-1).u[0]) + uinf[0];
                  const Real v_jp1 = 0.5*(V(ix,iy).u[1] + V(ix,iy+1).u[1]) + uinf[1];
                  const Real v_jm1 = 0.5*(V(ix,iy).u[1] + V(ix,iy-1).u[1]) + uinf[1];

                  faceYm[ix] = dfac*(V(ix,iy) - V(ix,iy-1));
                  faceYm[ix].u[0] -= afac*(-u_jm1*v_jm1);
                  faceYm[ix].u[1] -= afac*(-v_jm1*v_jm1);
                }
              }
              if (faceYp != nullptr)
              {
                int iy = BSY-1;
                for(int ix=0; ix<BSX; ++ix)
                {
                  const Real u_ip1 = 0.5*(V(ix,iy).u[0] + V(ix+1,iy).u[0]) + uinf[0];
                  const Real u_im1 = 0.5*(V(ix,iy).u[0] + V(ix-1,iy).u[0]) + uinf[0];
                  const Real v_ip1 = 0.5*(V(ix,iy).u[1] + V(ix+1,iy).u[1]) + uinf[1];
                  const Real v_im1 = 0.5*(V(ix,iy).u[1] + V(ix-1,iy).u[1]) + uinf[1];
                  const Real u_jp1 = 0.5*(V(ix,iy).u[0] + V(ix,iy+1).u[0]) + uinf[0];
                  const Real u_jm1 = 0.5*(V(ix,iy).u[0] + V(ix,iy-1).u[0]) + uinf[0];
                  const Real v_jp1 = 0.5*(V(ix,iy).u[1] + V(ix,iy+1).u[1]) + uinf[1];
                  const Real v_jm1 = 0.5*(V(ix,iy).u[1] + V(ix,iy-1).u[1]) + uinf[1];

                  faceYp[ix] = dfac*(V(ix,iy) - V(ix,iy+1));
                  faceYp[ix].u[0] -= afac*(u_jp1*v_jp1);
                  faceYp[ix].u[1] -= afac*(v_jp1*v_jp1);
                }
              }
    }   
    Corrector.FillBlockCases();
  }

#pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);
    #pragma omp for schedule(runtime)
    for (size_t i=0; i < Nblocks; i++)
    {
      int aux = 1<<velInfo[i].level;
      const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0;          };
      const auto isE = [&](const BlockInfo&I) { return I.index[0] == aux*sim.bpdx-1; };
      const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0;          };
      const auto isN = [&](const BlockInfo&I) { return I.index[1] == aux*sim.bpdy-1; };

      const Real h = velInfo[i].h_gridpoint;
      const Real fac = 1.0;//std::min((Real)1, sim.uMax_measured * dt / h);
      const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
      const Real fadeW= 1 - fac * std::pow(std::max(UINF[0], (Real)0)/norUinf, 2);
      const Real fadeS= 1 - fac * std::pow(std::max(UINF[1], (Real)0)/norUinf, 2);
      const Real fadeE= 1 - fac * std::pow(std::min(UINF[0], (Real)0)/norUinf, 2);
      const Real fadeN= 1 - fac * std::pow(std::min(UINF[1], (Real)0)/norUinf, 2);

      vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      if(isW(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(BX-1,iy), fadeW);
      if(isS(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,BY-1), fadeS);
      if(isE(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(EX+1,iy), fadeE);
      if(isN(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,EY+1), fadeN);

      for(int iy=0; iy<BSY; ++iy) for(int ix=0; ix<BSX; ++ix)
      {
        TMP(ix,iy).u[0] /= (h*h);
        TMP(ix,iy).u[1] /= (h*h);
        TMP(ix,iy).u[0] += V(ix,iy).u[0];
        TMP(ix,iy).u[1] += V(ix,iy).u[1];
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
          VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    V.copy(T);
  }

 {
    ////////////////////////////////////////////////////////////////////////////
    Real IF = 0.0;
    #pragma omp parallel for schedule(static) reduction(+ : IF)
    for (size_t i=0; i < Nblocks; i++)
    {
      const int aux = 1<<velInfo[i].level;
      const bool isW = velInfo[i].index[0] == 0;
      const bool isE = velInfo[i].index[0] == aux*sim.bpdx-1;
      const bool isS = velInfo[i].index[1] == 0;
      const bool isN = velInfo[i].index[1] == aux*sim.bpdy-1;
      const VectorBlock& V = *(VectorBlock*) velInfo[i].ptrBlock;

      const double h = velInfo[i].h_gridpoint;
      
      if (isW) for(int iy=0; iy<BSY; ++iy) IF += - h * V(BX,iy).u[0];
      if (isE) for(int iy=0; iy<BSY; ++iy) IF +=   h * V(EX,iy).u[0];
      if (isS) for(int ix=0; ix<BSX; ++ix) IF += - h * V(ix,BY).u[1];
      if (isN) for(int ix=0; ix<BSX; ++ix) IF +=   h * V(ix,EY).u[1];
    }
    ////////////////////////////////////////////////////////////////////////////
    const double H = sim.getH();
    const Real corr = IF/H/( 2*BSY*sim.bpdy*(1<<(sim.levelMax-1))
                           + 2*BSX*sim.bpdx*(1<<(sim.levelMax-1)) );

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      int aux = 1<<velInfo[i].level;
      const bool isW = velInfo[i].index[0] == 0;
      const bool isE = velInfo[i].index[0] == aux*sim.bpdx-1;
      const bool isS = velInfo[i].index[1] == 0;
      const bool isN = velInfo[i].index[1] == aux*sim.bpdy-1;
      VectorBlock& V = *(VectorBlock*) velInfo[i].ptrBlock;
      if(isW) for(int iy=0; iy<BSY; ++iy) V(BX,iy).u[0] += corr;
      if(isE) for(int iy=0; iy<BSY; ++iy) V(EX,iy).u[0] -= corr;
      if(isS) for(int ix=0; ix<BSX; ++ix) V(ix,BY).u[1] += corr;
      if(isN) for(int ix=0; ix<BSX; ++ix) V(ix,EY).u[1] -= corr;
    }
  }
  sim.stopProfiler();
}
#endif
