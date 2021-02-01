//
//  CubismUP_2D - AMR
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "advDiff.h"

using namespace cubism;

#if 0 //central differences (unstable!)
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
#else // use quick
static inline Real dU_adv_dif(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  // get grid values
  const Real uppx = V(ix+2, iy).u[0], uppy = V(ix, iy+2).u[0];
  const Real upx  = V(ix+1, iy).u[0], upy  = V(ix, iy+1).u[0];
  const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
  const Real ulx  = V(ix-1, iy).u[0], uly  = V(ix, iy-1).u[0];
  const Real ullx = V(ix-2, iy).u[0], ully = V(ix, iy-2).u[0];

  // advection
  const Real u = ucc+uinf[0];
  const Real dudx  = u > 0 ?           3*upx + 3*ucc - 7*ulx + ullx
                             : -uppx + 7*upx - 3*ucc - 3*ulx        ;
  const Real v = vcc+uinf[1];
  const Real dudy  = v > 0 ?           3*upy + 3*ucc - 7*uly + ully
                             : -uppy + 7*upy - 3*ucc - 3*uly        ;
  const Real dUadv = u * 0.125 * dudx + v * 0.125 * dudy;

  // diffusion
  const Real dUdif = upx + upy + ulx + uly - 4 *ucc;

  return advF * dUadv + difF * dUdif;
}

static inline Real dV_adv_dif(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  const Real vppx = V(ix+2, iy).u[1], vppy = V(ix, iy+2).u[1];
  const Real vpx  = V(ix+1, iy).u[1], vpy  = V(ix, iy+1).u[1];
  const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
  const Real vlx  = V(ix-1, iy).u[1], vly  = V(ix, iy-1).u[1];
  const Real vllx = V(ix-2, iy).u[1], vlly = V(ix, iy-2).u[1];

  // advection
  const Real u = ucc+uinf[0];
  const Real dvdx  = u > 0 ?           3*vpx + 3*vcc - 7*vlx + vllx
                             : -vppx + 7*vpx - 3*vcc - 3*vlx        ;
  const Real v = vcc+uinf[1];
  const Real dvdy  = v > 0 ?           3*vpy + 3*vcc - 7*vly + vlly
                             : -vppy + 7*vpy - 3*vcc - 3*vly        ;
  const Real dVadv = u * 0.125 * dvdx + v * 0.125 * dvdy;

  // diffusion
  const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;

  return advF * dVadv + difF * dVdif;
}
#endif

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
        #if 0 // stencil for centered advection
        static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
        #else // for quick
        static constexpr int stenBeg[3] = {-2,-2, 0}, stenEnd[3] = { 3, 3, 1};
        #endif
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
            //const Real dfac = (sim.nu/h)*(dt/h), afac = -0.5*dt/h; //central differences coefficients
            const Real dfac = (sim.nu/h)*(dt/h),  afac = -dt/h/6.0;
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