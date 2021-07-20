//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "PressureSingle.h"
#include "../Poisson/AMRSolver.h"
#include "../Shape.h"

using namespace cubism;

static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

//#define EXPL_INTEGRATE_MOM

void PressureSingle::integrateMomenta(Shape * const shape) const
{
  const size_t Nblocks = velInfo.size();

  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const Real Cx = shape->centerOfMass[0];
  const Real Cy = shape->centerOfMass[1];
  double PM=0, PJ=0, PX=0, PY=0, UM=0, VM=0, AM=0; //linear momenta

  #pragma omp parallel for reduction(+:PM,PJ,PX,PY,UM,VM,AM)
  for(size_t i=0; i<Nblocks; i++)
  {
    const VectorBlock& __restrict__ VEL = *(VectorBlock*)velInfo[i].ptrBlock;
    const double hsq = velInfo[i].h*velInfo[i].h;

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
      double p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      PM += F;
      PJ += F * (p[0]*p[0] + p[1]*p[1]);
      PX += F * p[0];  PY += F * p[1];
      UM += F * udiff[0]; VM += F * udiff[1];
      AM += F * (p[0]*udiff[1] - p[1]*udiff[0]);
    }
  }

  shape->fluidAngMom = AM; shape->fluidMomX = UM; shape->fluidMomY = VM;
  shape->penalDX=PX; shape->penalDY=PY; shape->penalM=PM; shape->penalJ=PJ;
}

void PressureSingle::penalize(const double dt) const
{
  const size_t Nblocks = velInfo.size();

  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  for (Shape * const shape : sim.shapes)
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

void PressureSingle::updatePressureRHS(const double dt) const
{
  // RHS of Poisson equation is div(u) - chi * div(u_def)
  // It is computed here and stored in TMP

  const size_t Nblocks = velInfo.size();
  static constexpr int stenBeg[3] = {-1,-1, 0};
  static constexpr int stenEnd[3] = { 2, 2, 1};

  FluxCorrection<ScalarGrid,ScalarBlock> Corrector;
  Corrector.prepare(*(sim.tmp));
  #pragma omp parallel
  {
    VectorLab velLab;
    VectorLab uDefLab;
    velLab. prepare( *(sim.vel),  stenBeg, stenEnd, 0);
    uDefLab.prepare( *(sim.uDef), stenBeg, stenEnd, 0);

    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
      const Real h = velInfo[i].h;
      const Real facDiv = 0.5*h/dt;
      velLab. load(velInfo [i], 0);
      uDefLab.load(uDefInfo[i], 0);
      ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[i].ptrBlock;
      ScalarBlock& __restrict__ CHI = *(ScalarBlock*) chiInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        TMP(ix, iy).s  =   facDiv                *( velLab(ix+1,iy).u[0] -  velLab(ix-1,iy).u[0] 
                                                 +  velLab(ix,iy+1).u[1] -  velLab(ix,iy-1).u[1]);
        TMP(ix, iy).s += - facDiv * CHI(ix,iy).s *(uDefLab(ix+1,iy).u[0] - uDefLab(ix-1,iy).u[0] 
                                                 + uDefLab(ix,iy+1).u[1] - uDefLab(ix,iy-1).u[1]);
      }

      BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(tmpInfo[i].auxiliary);
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
          faceXm[iy].s += -facDiv * CHI(ix,iy).s *(uDefLab(ix-1,iy).u[0] + uDefLab(ix,iy).u[0]) ;
        }
      }
      if (faceXp != nullptr)
      {
        int ix = VectorBlock::sizeX-1;
        for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        {
          faceXp[iy].s  = -facDiv               *( velLab(ix+1,iy).u[0] +  velLab(ix,iy).u[0]);
          faceXp[iy].s -= -facDiv *CHI(ix,iy).s *(uDefLab(ix+1,iy).u[0] + uDefLab(ix,iy).u[0]);
        }
      }
      if (faceYm != nullptr)
      {
        int iy = 0;
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          faceYm[ix].s  =  facDiv               *( velLab(ix,iy-1).u[1] +  velLab(ix,iy).u[1]);
          faceYm[ix].s += -facDiv *CHI(ix,iy).s *(uDefLab(ix,iy-1).u[1] + uDefLab(ix,iy).u[1]);         
        }
      }
      if (faceYp != nullptr)
      {
        int iy = VectorBlock::sizeY-1;
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
          faceYp[ix].s  = -facDiv               *( velLab(ix,iy+1).u[1] +  velLab(ix,iy).u[1]);
          faceYp[ix].s -= -facDiv *CHI(ix,iy).s *(uDefLab(ix,iy+1).u[1] + uDefLab(ix,iy).u[1]);
        }
      }
    }
  }
  Corrector.FillBlockCases();
}

void PressureSingle::pressureCorrection(const double dt) const
{
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const size_t Nblocks = velInfo.size();
  FluxCorrection<VectorGrid,VectorBlock> Corrector;
  Corrector.prepare(*(sim.tmpV));
  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for
    for (size_t i=0; i < Nblocks; i++)
    {
      const Real h = presInfo[i].h_gridpoint, pFac = -0.5*dt*h;
      //const Real h = presInfo[i].h_gridpoint, pFac = -0.5*dt/h;
      plab.load(presInfo[i], 0); // loads pres field with ghosts
      const ScalarLab  &__restrict__   P = plab; // only this needs ghosts
      VectorBlock&__restrict__   tmpV = *(VectorBlock*)  tmpVInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        tmpV(ix,iy).u[0] = pFac *(P(ix+1,iy).s-P(ix-1,iy).s);
        tmpV(ix,iy).u[1] = pFac *(P(ix,iy+1).s-P(ix,iy-1).s);
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
  }
  Corrector.FillBlockCases();
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
      const Real ih2 = 1.0/presInfo[i].h_gridpoint/presInfo[i].h_gridpoint;
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

void PressureSingle::preventCollidingObstacles() const
{
  const std::vector<Shape*>& shapes = sim.shapes;
  const size_t N = shapes.size();

  struct CollisionInfo // hitter and hittee, symmetry but we do things twice
  {
    Real iM = 0, iPosX = 0, iPosY = 0, iMomX = 0, iMomY = 0;
    Real jM = 0, jPosX = 0, jPosY = 0, jMomX = 0, jMomY = 0;
  };
  std::vector<CollisionInfo> collisions(N);

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i<N; ++i)
  for (size_t j=0; j<N; ++j)
  {
    if(i==j) continue;
    auto & coll = collisions[i];
    const auto& iBlocks = shapes[i]->obstacleBlocks;
    const auto& jBlocks = shapes[j]->obstacleBlocks;
    const Real iUl = shapes[i]->u, iVl = shapes[i]->v, iW = shapes[i]->omega;
    const Real jUl = shapes[j]->u, jVl = shapes[j]->v, jW = shapes[j]->omega;
    const Real iCx =shapes[i]->centerOfMass[0], iCy =shapes[i]->centerOfMass[1];
    const Real jCx =shapes[j]->centerOfMass[0], jCy =shapes[j]->centerOfMass[1];

    assert(iBlocks.size() == jBlocks.size());
    const size_t nBlocks = iBlocks.size();

    for (size_t k=0; k<nBlocks; ++k)
    {
      if ( iBlocks[k] == nullptr || jBlocks[k] == nullptr ) continue;

      const CHI_MAT & iChi  = iBlocks[k]->chi,  & jChi  = jBlocks[k]->chi;
      const UDEFMAT & iUDEF = iBlocks[k]->udef, & jUDEF = jBlocks[k]->udef;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        if(iChi[iy][ix] <= 0 || jChi[iy][ix] <= 0 ) continue;

        const auto pos = velInfo[k].pos<Real>(ix, iy);
        const Real iUr = - iW * (pos[1] - iCy), iVr =   iW * (pos[0] - iCx);
        const Real jUr = - jW * (pos[1] - jCy), jVr =   jW * (pos[0] - jCx);
        coll.iM    += iChi[iy][ix];
        coll.iPosX += iChi[iy][ix] * pos[0];
        coll.iPosY += iChi[iy][ix] * pos[1];
        coll.iMomX += iChi[iy][ix] * (iUl + iUr + iUDEF[iy][ix][0]);
        coll.iMomY += iChi[iy][ix] * (iVl + iVr + iUDEF[iy][ix][1]);
        coll.jM    += jChi[iy][ix];
        coll.jPosX += jChi[iy][ix] * pos[0];
        coll.jPosY += jChi[iy][ix] * pos[1];
        coll.jMomX += jChi[iy][ix] * (jUl + jUr + jUDEF[iy][ix][0]);
        coll.jMomY += jChi[iy][ix] * (jVl + jVr + jUDEF[iy][ix][1]);
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i<N; ++i)
  for (size_t j=0; j<N; ++j)
  {
    if(i==j) continue;
    auto & coll = collisions[i];

    // less than one fluid element of overlap: wait to get closer. no hit
    if(coll.iM < 1 || coll.jM < 1) continue;
    const Real iPX = coll.iPosX / coll.iM, iPY = coll.iPosY / coll.iM;
    const Real iUX = coll.iMomX / coll.iM, iUY = coll.iMomY / coll.iM;
    const Real jPX = coll.jPosX / coll.jM, jPY = coll.jPosY / coll.jM;
    const Real jUX = coll.jMomX / coll.jM, jUY = coll.jMomY / coll.jM;
    const Real CX = (iPX+jPX)/2, CY = (iPY+jPY)/2;
    const Real dirX = iPX - jPX, dirY = iPY - jPY;
    const Real hitVelX = jUX - iUX, hitVelY = jUY - iUY;
    const Real normF = std::max(std::sqrt(dirX*dirX + dirY*dirY), EPS);
    const Real NX = dirX / normF, NY = dirY / normF; // collision normal
    const Real projVel = hitVelX * NX + hitVelY * NY;
    printf("%lu hit %lu in [%f %f] with dir:[%f %f] DU:[%f %f] proj:%f\n",
        i, j, CX, CY, NX, NY, hitVelX, hitVelY, projVel); fflush(0);

    if(projVel<=0) continue; // vel goes away from coll: no need to bounce
    const bool iForcedX = shapes[i]->bForcedx && sim.time<shapes[i]->timeForced;
    const bool iForcedY = shapes[i]->bForcedy && sim.time<shapes[i]->timeForced;
    const bool iForcedA = shapes[i]->bBlockang&& sim.time<shapes[i]->timeForced;
    const bool jForcedX = shapes[j]->bForcedx && sim.time<shapes[j]->timeForced;
    const bool jForcedY = shapes[j]->bForcedy && sim.time<shapes[j]->timeForced;

    const Real iInvMassX = iForcedX? 0 : 1/shapes[i]->M; // forced == inf mass
    const Real iInvMassY = iForcedY? 0 : 1/shapes[i]->M;
    const Real jInvMassX = jForcedX? 0 : 1/shapes[j]->M;
    const Real jInvMassY = jForcedY? 0 : 1/shapes[j]->M;
    const Real iInvJ     = iForcedA? 0 : 1/shapes[i]->J;
    const Real meanMassX = 2 / std::max(iInvMassX + jInvMassX, EPS);
    const Real meanMassY = 2 / std::max(iInvMassY + jInvMassY, EPS);
    // Force_i_bounce _from_j = HARMONIC_MEAN_MASS * (Uj - Ui) / dt
    const Real FXdt = meanMassX * projVel * NX;
    const Real FYdt = meanMassY * projVel * NY;
    shapes[i]->u += FXdt * iInvMassX; // if forced, no update
    shapes[i]->v += FYdt * iInvMassY;
    const Real iCx =shapes[i]->centerOfMass[0], iCy =shapes[i]->centerOfMass[1];
    const Real RcrossF = (CX-iCx) * FYdt - (CY-iCy) * FXdt;
    shapes[i]->omega += iInvJ * RcrossF;
  }
}


bool PressureSingle::detectCollidingObstacles() const
{
  // boolean indicating whether there was a collision
  bool bCollision = false;

  // get shapes present in simulation
  const std::vector<Shape*>& shapes = sim.shapes;
  const size_t N = shapes.size();

  // collisions are symmetric, so only iterate over each pair once
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i<N; ++i)
  for (size_t j=i+1; j<N; ++j)
  {
    // get obstacle blocks for both obstacles
    const auto& iBlocks = shapes[i]->obstacleBlocks;
    const auto& jBlocks = shapes[j]->obstacleBlocks;
    assert(iBlocks.size() == jBlocks.size());

    // iterate over obstacle blocks
    const size_t nBlocks = iBlocks.size();
    for (size_t k=0; k<nBlocks; ++k)
    {
      // If one of the two shapes does not occupy this block, continue
      if ( iBlocks[k] == nullptr || jBlocks[k] == nullptr ) continue;

      // Get characteristic function of candidate blocks
      const CHI_MAT & iChi  = iBlocks[k]->chi,  & jChi  = jBlocks[k]->chi;

      // Iterate over cells in candidate block
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        // If one of the two shapes does not occupy this cell, continue
        if(iChi[iy][ix] <= 0 || jChi[iy][ix] <= 0 ) continue;

        //// collision!
        // get location of collision
        const auto pos = velInfo[k].pos<Real>(ix, iy);

        // set boolean to true and tell user
        bCollision = true;
        printf("[CUP2D] WARNING: %lu hit %lu in [%f %f]\n", i, j, pos[0], pos[1]); fflush(0);
      }
    }
  }
  return bCollision;
}

void PressureSingle::operator()(const double dt)
{
  sim.startProfiler("PressureSingle");

  const std::vector<cubism::BlockInfo>& poldInfo = sim.pold->getBlocksInfo();
  const size_t Nblocks = velInfo.size();
  const int step_extrapolate = 100; //start the extrapolation after 100 steps (after dt ramp up is finished)

  if (sim.step > step_extrapolate)
  {
     #pragma omp parallel for
     for (size_t i=0; i < Nblocks; i++)
     {
        ScalarBlock & __restrict__   POLD = *(ScalarBlock*)  poldInfo[i].ptrBlock;
        ScalarBlock & __restrict__   PRES = *(ScalarBlock*)  presInfo[i].ptrBlock;
        for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        {
           const double dpdt = (PRES(ix,iy).s - POLD(ix,iy).s)/sim.dt_old;
           POLD (ix,iy).s = PRES (ix,iy).s;
           if (sim.step > step_extrapolate + 1)
               PRES(ix,iy).s += dpdt*sim.dt;
        }
     }
  }

  // update velocity of obstacle
  for(Shape * const shape : sim.shapes) {
    integrateMomenta(shape);
    shape->updateVelocity(dt);
  }
  // take care if two obstacles collide
  // preventCollidingObstacles(); (needs to be fixed)
  sim.bCollision = detectCollidingObstacles();

  // apply penalisation force
  penalize(dt);

  // compute pressure RHS
  updatePressureRHS(dt);

  // solve Poisson equation
  pressureSolver->solve();

  // apply pressure correction
  pressureCorrection(dt);

  sim.stopProfiler();
}

//PressureSingle::PressureSingle(SimulationData& s) : Operator(s),
//pressureSolver( PoissonSolver::makeSolver(s) ) { }
PressureSingle::PressureSingle(SimulationData& s) : Operator(s)
{
  pressureSolver = new AMRSolver(s);
}

PressureSingle::~PressureSingle() {
  delete pressureSolver;
}
