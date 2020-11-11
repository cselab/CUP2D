//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

/*
Fully implicit advection-diffusion.
First order upwind advection and second order diffusion.
*/

#include "advDiff_implicit_all.h"
#include "../Poisson/LinearSolverDelta.h"

using namespace cubism;

static inline Real RX(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  const Real up1x = V(ix+1, iy).u[0], up1y = V(ix, iy+1).u[0];
  const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
  const Real ul1x = V(ix-1, iy).u[0], ul1y = V(ix, iy-1).u[0];
  #if 1
  const Real dudx = (ucc+uinf[0] > 0) ? ucc - ul1x : up1x - ucc;
  const Real dudy = (vcc+uinf[1] > 0) ? ucc - ul1y : up1y - ucc;
  #else
  const Real up2x = V(ix+2, iy).u[0], up2y = V(ix, iy+2).u[0];
  const Real ul2x = V(ix-2, iy).u[0], ul2y = V(ix, iy-2).u[0];
  const Real vp2x = V(ix+2, iy).u[1], vp2y = V(ix, iy+2).u[1];
  const Real vl2x = V(ix-2, iy).u[1], vl2y = V(ix, iy-2).u[1];
  const Real dudx = (ucc+uinf[0] > 0) ?          (2*up1x + 3*ucc - 6*ul1x + ul2x)/3.0
                                      : (- up2x + 6*up1x - 3*ucc - 2*ul1x)/3.0;
  const Real dudy = (vcc+uinf[1] > 0) ?          (2*up1y + 3*ucc - 6*ul1y + ul2y)/3.0
                                      : (- up2y + 6*up1y - 3*ucc - 2*ul1y)/3.0;
  #endif
  const Real dUadv = (ucc+uinf[0]) * dudx + (vcc+uinf[1]) * dudy;
  const Real dUdif = up1x + up1y + ul1x + ul1y - 4 *ucc;
  return 2.0 * advF * dUadv + difF * dUdif; 
}

static inline Real RY(const VectorLab&V, const Real uinf[2],
  const Real advF, const Real difF, const int ix, const int iy)
{
  const Real vp1x = V(ix+1, iy).u[1], vp1y = V(ix, iy+1).u[1];
  const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
  const Real vl1x = V(ix-1, iy).u[1], vl1y = V(ix, iy-1).u[1];
  #if 1
  const Real dvdx = (ucc+uinf[0] > 0)  ? vcc - vl1x : vp1x - vcc;
  const Real dvdy = (vcc+uinf[1] > 0)  ? vcc - vl1y : vp1y - vcc;
  #else
  const Real up2x = V(ix+2, iy).u[0], up2y = V(ix, iy+2).u[0];
  const Real ul2x = V(ix-2, iy).u[0], ul2y = V(ix, iy-2).u[0];
  const Real vp2x = V(ix+2, iy).u[1], vp2y = V(ix, iy+2).u[1];
  const Real vl2x = V(ix-2, iy).u[1], vl2y = V(ix, iy-2).u[1];
  const Real dvdx = (ucc+uinf[0] > 0)>0 ?          (2*vp1x + 3*vcc - 6*vl1x + vl2x)/3.0
                                        : (- vp2x + 6*vp1x - 3*vcc - 2*vl1x)/3.0;
  const Real dvdy = (vcc+uinf[1] > 0)   ?          (2*vp1y + 3*vcc - 6*vl1y + vl2y)/3.0
                                        : (- vp2y + 6*vp1y - 3*vcc - 2*vl1y)/3.0;
  #endif
  const Real dVadv = (ucc+uinf[0]) * dvdx + (vcc+uinf[1]) * dvdy;
  const Real dVdif = vp1x + vp1y + vl1x + vl1y - 4 * vcc;
  return 2.0 * advF * dVadv + difF * dVdif;
}

void advDiff_implicit_all::explicit_update(double dt)
{
    static const int BSX = VectorBlock::sizeX;
    static const int BSY = VectorBlock::sizeY;

    const Real UINF[2]= {sim.uinfx, sim.uinfy};
    const Real h = sim.getH();
    const Real dfac = -(sim.nu/h)*(dt/h);
    const Real afac = 0.5*dt/h;

    #pragma omp parallel
    {
        static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};  
        static const double t = 0.0; //this value does not matter
        static const bool applybc = true; //bc is zero Neumann (first order)

        VectorLab vellab; 
        vellab.prepare(*(sim.vOld), stenBeg, stenEnd, 1);
  
        #pragma omp for schedule(static)
        for (size_t i=0; i < Nblocks; i++)
        {
            vellab.load(vOldInfo[i],t,applybc); 
            VectorBlock & __restrict__ tmpV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
            VectorBlock & __restrict__ vOld = *(VectorBlock*) vOldInfo[i].ptrBlock;        
        
            for(int iy=0; iy<BSY; ++iy)
            for(int ix=0; ix<BSX; ++ix)
            {
              tmpV(ix,iy).u[0] =  vOld(ix,iy).u[0] - RX(vellab,UINF,afac,dfac,ix,iy); 
              tmpV(ix,iy).u[1] =  vOld(ix,iy).u[1] - RY(vellab,UINF,afac,dfac,ix,iy); 
            }
        }

        #pragma omp for schedule(static)
        for (size_t i=0; i < Nblocks; i++)
        {
            VectorBlock & __restrict__ V  = *(VectorBlock*) velInfo[i].ptrBlock;
            const VectorBlock & __restrict__ T = *(VectorBlock*) tmpVInfo[i].ptrBlock;
            V.copy(T);
        }   
    }
}

void advDiff_implicit_all::operator()(const double dt)
{
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    const VectorBlock & __restrict__ V = *(VectorBlock*) velInfo[i].ptrBlock;
    VectorBlock & __restrict__ Vold = *(VectorBlock*) vOldInfo[i].ptrBlock;
    Vold.copy(V);
  } 
  //We first make an explicit update as an initial guess for the velocities at t = n+1
  explicit_update(dt);

  //VoldInfo : velocities at t = n
  //velInfo  : current estimate for velocities at t = n + 1
  //tmpVinfo : residual vector (RHS for linear systems)

#if 1 //set to 0 and we have explicit advection-diffusion
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;

  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  const Real dfac = -(sim.nu/h)*(dt/h), afac = 0.5*dt/h;

  static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
  static const double t = 0.0; //this value does not matter
  static const bool applybc = true; //bc is zero Neumann (first order)
  
  double norm0 = 0.0;

  for (int iteration = 0 ; iteration < 1000 ; iteration ++)
  {
    double norm_x = 0.0;
    double norm_y = 0.0;

    //1. Pass velocities to solver, so that it computes the LHS
    sim.startProfiler("Compute LHS");
    if (iteration == 0 || iteration % 1 == 0)
    {
      mySolver->cub2LHS(velInfo, UINF[0], UINF[1]);
    } 
    sim.stopProfiler();

    //2. Compute RHS 
    sim.startProfiler("Compute RHS");
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      VectorLab vellab; 
      vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);
      
      vellab.load(velInfo[i], t, applybc);
      
      VectorBlock & __restrict__ VOLD = *(VectorBlock*) vOldInfo[i].ptrBlock;
      VectorBlock & __restrict__ RHS = *(VectorBlock*) tmpVInfo[i].ptrBlock;

      double nx = 0.0;
      double ny = 0.0;
      
      for(int iy=0; iy<BSY; ++iy) 
      for(int ix=0; ix<BSX; ++ix)
      {
        RHS(ix,iy).u[0] = - ( RX(vellab,UINF,afac,dfac,ix,iy) + vellab(ix,iy).u[0] - VOLD(ix,iy).u[0]);
        RHS(ix,iy).u[1] = - ( RY(vellab,UINF,afac,dfac,ix,iy) + vellab(ix,iy).u[1] - VOLD(ix,iy).u[1]);
        nx += RHS(ix,iy).u[0]*RHS(ix,iy).u[0];
        ny += RHS(ix,iy).u[1]*RHS(ix,iy).u[1];
      }

      #pragma omp critical
      {
        norm_x += nx;
        norm_y += ny;
      }
    }
    sim.stopProfiler();

    double norm = ( pow(norm_x,0.5) + pow(norm_y,0.5) ) /Nblocks/BSX/BSY ;
    norm = std::log10(norm + 1e-21);
    if (iteration == 0) norm0 = norm;
    std::cout << "iteration:" << iteration << " norm = " << norm << std::endl;
    if (norm - norm0 < - 3.0 || norm < -15.0) break; //these convergence criteria can probably be relaxed.

    //3. Solve for velocities correction
    sim.startProfiler("Solve systems");
    mySolver->solve(tmpVInfo,true);
    mySolver->solve(tmpVInfo,false);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      VectorBlock & __restrict__ SOL = *(VectorBlock*) tmpVInfo[i].ptrBlock;      
      VectorBlock & __restrict__ V   = *(VectorBlock*) velInfo[i].ptrBlock;
      for(int iy=0; iy<BSY; ++iy) 
      for(int ix=0; ix<BSX; ++ix)
      {
        V(ix,iy).u[0] += SOL(ix,iy).u[0];
        V(ix,iy).u[1] += SOL(ix,iy).u[1];
      }
    }
    sim.stopProfiler();
  }
#endif

}

advDiff_implicit_all::advDiff_implicit_all(SimulationData& s) : Operator(s)
{
    mySolver = new LinearSolverDelta(s); 
}

advDiff_implicit_all::~advDiff_implicit_all() {
  delete mySolver;
}



#if 0
    static inline Real RX1(const VectorLab&V, const Real uinf[2],
      const Real advF, const Real difF, const int ix, const int iy)
    {
      const Real up1x = V(ix+1, iy).u[0], up1y = V(ix, iy+1).u[0];
      const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
      const Real ul1x = V(ix-1, iy).u[0], ul1y = V(ix, iy-1).u[0];
      const Real dudx = (ucc+uinf[0] > 0) ? ucc - ul1x : up1x - ucc;
      const Real dudy = (vcc+uinf[1] > 0) ? ucc - ul1y : up1y - ucc;
      const Real dUadv = (ucc+uinf[0]) * dudx + (vcc+uinf[1]) * dudy;
      const Real dUdif = up1x + up1y + ul1x + ul1y;// for Jacobi
      return 2.0 * advF * dUadv + difF * dUdif; 
    }
    
    static inline Real RY1(const VectorLab&V, const Real uinf[2],
      const Real advF, const Real difF, const int ix, const int iy)
    {
      const Real vp1x = V(ix+1, iy).u[1], vp1y = V(ix, iy+1).u[1];
      const Real ucc  = V(ix  , iy).u[0], vcc  = V(ix, iy  ).u[1];
      const Real vl1x = V(ix-1, iy).u[1], vl1y = V(ix, iy-1).u[1];
      const Real dvdx = (ucc+uinf[0] > 0)  ? vcc - vl1x : vp1x - vcc;
      const Real dvdy = (vcc+uinf[1] > 0)  ? vcc - vl1y : vp1y - vcc;
      const Real dVadv = (ucc+uinf[0]) * dvdx + (vcc+uinf[1]) * dvdy;
      const Real dVdif = vp1x + vp1y + vl1x + vl1y;// for Jacobi;
      return 2.0 * advF * dVadv + difF * dVdif;
    }
    void advDiff_implicit_all::Jacobi(int max_iter,double dt)
    {
      static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
      static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
      static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;
      const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0;          };
      const auto isE = [&](const BlockInfo&I) { return I.index[0] == sim.bpdx-1; };
      const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0;          };
      const auto isN = [&](const BlockInfo&I) { return I.index[1] == sim.bpdy-1; };
    
      const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
      const Real dfac = -(sim.nu/h)*(dt/h), afac = 0.5*dt/h;
      const Real fac = std::min((Real)1, sim.uMax_measured * dt / h);
      const Real norUinf = std::max({std::fabs(UINF[0]), std::fabs(UINF[1]), EPS});
      const Real fadeW= 1 - fac * std::pow(std::max(UINF[0], (Real)0)/norUinf, 2);
      const Real fadeS= 1 - fac * std::pow(std::max(UINF[1], (Real)0)/norUinf, 2);
      const Real fadeE= 1 - fac * std::pow(std::min(UINF[0], (Real)0)/norUinf, 2);
      const Real fadeN= 1 - fac * std::pow(std::min(UINF[1], (Real)0)/norUinf, 2);
      const auto fade = [&](VectorElement&B,const Real F) { B.u[0]*=F; B.u[1]*=F; };
    
      double rel = 0.1;
      double norm = 0.0;
      double norm0 = 0.0;
      for (int iter = 0 ; iter < max_iter; iter ++)
      {
        if (iter == 0) rel = 1.0;
        else           rel = 0.1;
    
        double norm_old = (iter == 0) ? 1000 : norm;
        norm = 0.0;
        #pragma omp parallel
        {
          static constexpr int stenBeg[3] = {-2,-2, 0}, stenEnd[3] = { 3, 3, 1};
      
          VectorLab vellab; 
          vellab.prepare(*(sim.tmpV), stenBeg, stenEnd, 1);
      
          #pragma omp for schedule(static)
          for (size_t i=0; i < Nblocks; i++)
          {
            vellab.load(tmpVInfo[i], 0); 
            VectorLab & __restrict__ V = vellab;
            VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;
            VectorBlock & __restrict__ VEL_OLD = *(VectorBlock*) velInfo[i].ptrBlock;
            
            VectorBlock & __restrict__ VNEW = *(VectorBlock*) vOldInfo[i].ptrBlock;
    
    
            if(isW(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(BX-1,iy), fadeW);
            if(isS(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,BY-1), fadeS);
            if(isE(velInfo[i])) for(int iy=-1; iy<=BSY; ++iy) fade(V(EX+1,iy), fadeE);
            if(isN(velInfo[i])) for(int ix=-1; ix<=BSX; ++ix) fade(V(ix,EY+1), fadeN);
    
            double nx = 0.0;
            double ny = 0.0;
            double dtau = 0.1*dt;
            for(int iy=0; iy<BSY; ++iy) for(int ix=0; ix<BSX; ++ix)
            {
              double rx = ( VEL_OLD(ix,iy).u[0] - RX1(V,UINF,afac,dfac,ix,iy) ) / (1.0 - 4.0*dfac);
              double ry = ( VEL_OLD(ix,iy).u[1] - RY1(V,UINF,afac,dfac,ix,iy) ) / (1.0 - 4.0*dfac);
              double ux = (1.0-rel) * TMP(ix,iy).u[0] + rel * rx;
              double uy = (1.0-rel) * TMP(ix,iy).u[1] + rel * ry;
              nx += (TMP(ix,iy).u[0]-ux)*(TMP(ix,iy).u[0]-ux);
              ny += (TMP(ix,iy).u[1]-uy)*(TMP(ix,iy).u[1]-uy);
              VNEW(ix,iy).u[0] = ux;
              VNEW(ix,iy).u[1] = uy;       
            }
            #pragma omp critical
            {
              norm += nx + ny;
            }
          }
          #pragma omp for schedule(static)
          for (size_t i=0; i < Nblocks; i++)
          {
            VectorBlock & __restrict__ V  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
            const VectorBlock & __restrict__ T  = *(VectorBlock*) vOldInfo[i].ptrBlock;
            V.copy(T);
          }   
        }
        norm = std::log10 ( std::sqrt(norm) + 1e-21);
    
        if (iter == 0) norm0 = norm;
    
        //if (norm > norm_old       ) rel *= 0.9;
        //if (norm < norm_old - 1.0 ) rel *= 1.1;
        //if (rel < 0.01) rel = 0.01;
        if (iter % 100 ==0)std::cout << iter << " norm=" << norm  << " rel="<< rel << std::endl;
        if (norm  - norm0 < -6.0 && iter > 0) break;
        if (norm  < -20.0 && iter > 0) break;
      }
    }
#endif