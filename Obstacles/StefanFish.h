//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Fish.h"
#include "FishData.h"

class StefanFish: public Fish
{
  const bool bCorrectTrajectory;
  const bool bCorrectPosition;
 public:
  void act(const Real lTact, const std::vector<Real>& a) const;
  Real getLearnTPeriod() const;
  Real getPhase(const Real t) const;

  void resetAll() override;
  StefanFish(SimulationData&s, cubism::ArgumentParser&p, Real C[2]);
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;

  // member functions for state in RL
  std::vector<Real> state( const std::vector<double>& origin ) const;
  std::vector<Real> state3D( ) const;

  // Helpers for state function
  ssize_t holdingBlockID(const std::array<Real,2> pos) const;
  std::array<Real, 2> getShear(const std::array<Real,2> pSurf) const;

  // Old Helpers (here for backward compatibility)
  ssize_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const;
  std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const;
  std::array<Real, 2> getShear(const std::array<Real,2> pSurf, const std::array<Real,2> normSurf, const std::vector<cubism::BlockInfo>& velInfo) const;

  // Helpers to restart simulation
  virtual void saveRestart( FILE * f ) override;
  virtual void loadRestart( FILE * f ) override;
};

class CurvatureFish : public FishData
{
  const Real amplitudeFactor, phaseShift, Tperiod;
 public:
  // PID controller of body curvature:
  Real curv_PID_fac = 0;
  Real curv_PID_dif = 0;
  // exponential averages:
  Real avgDeltaY = 0;
  Real avgDangle = 0;
  Real avgAngVel = 0;
  // stored past action for RL state:
  Real lastTact = 0;
  Real lastCurv = 0;
  Real oldrCurv = 0;
  // quantities needed to correctly control the speed of the midline maneuvers:
  Real periodPIDval = Tperiod;
  Real periodPIDdif = 0;
  bool TperiodPID = false;
  // quantities needed for rl:
  Real time0 = 0;
  Real timeshift = 0;
  // aux quantities for PID controllers:
  Real lastTime = 0;
  Real lastAvel = 0;

  // next scheduler is used to ramp-up the curvature from 0 during first period:
  Schedulers::ParameterSchedulerVector<6> curvatureScheduler;
  // next scheduler is used for midline-bending control points for RL:
  Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;

  // next scheduler is used to ramp-up the period
  Schedulers::ParameterSchedulerScalar periodScheduler;
  Real current_period    = Tperiod;
  Real next_period       = Tperiod;
  Real transition_start  = 0.0;
  Real transition_duration = 0.1*Tperiod;

 protected:
  Real * const rK;
  Real * const vK;
  Real * const rC;
  Real * const vC;
  Real * const rB;
  Real * const vB;

 public:

  CurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
  : FishData(L, _h), amplitudeFactor(_A),  phaseShift(phi),  Tperiod(T), rK(_alloc(Nm)), vK(_alloc(Nm)),
    rC(_alloc(Nm)), vC(_alloc(Nm)), rB(_alloc(Nm)), vB(_alloc(Nm)) 
    {
      _computeWidth();
      writeMidline2File(0, "initialCheck");
    }

  void resetAll() override {
    curv_PID_fac = 0;
    curv_PID_dif = 0;
    avgDeltaY = 0;
    avgDangle = 0;
    avgAngVel = 0;
    lastTact = 0;
    lastCurv = 0;
    oldrCurv = 0;
    periodPIDval = Tperiod;
    periodPIDdif = 0;
    TperiodPID = false;
    time0 = 0;
    timeshift = 0;
    lastTime = 0;
    lastAvel = 0;
    curvatureScheduler.resetAll();
    periodScheduler.resetAll();
    rlBendingScheduler.resetAll();

    FishData::resetAll();
  }

  void correctTrajectory(const Real dtheta, const Real vtheta,
                                         const Real t, const Real dt)
  {
    curv_PID_fac = dtheta;
    curv_PID_dif = vtheta;
  }

  void correctTailPeriod(const Real periodFac,const Real periodVel,
                                        const Real t, const Real dt)
  {
    assert(periodFac>0 && periodFac<2); // would be crazy

    const Real lastArg = (lastTime-time0)/periodPIDval + timeshift;
    time0 = lastTime;
    timeshift = lastArg;
    // so that new arg is only constant (prev arg) + dt / periodPIDval
    // with the new l_Tp:
    periodPIDval = Tperiod * periodFac;
    periodPIDdif = Tperiod * periodVel;
    lastTime = t;
    TperiodPID = true;
  }

  // Execute takes as arguments the current simulation time and the time
  // the RL action should have actually started. This is important for the midline
  // bending because it relies on matching the splines with the half period of
  // the sinusoidal describing the swimming motion (in order to exactly amplify
  // or dampen the undulation). Therefore, for Tp=1, t_rlAction might be K * 0.5
  // while t_current would be K * 0.5 plus a fraction of the timestep. This
  // because the new RL discrete step is detected as soon as t_current>=t_rlAction
  void execute(const Real t_current, const Real t_rlAction,
                              const std::vector<Real>&a)
  {
    assert(t_current >= t_rlAction);
    oldrCurv = lastCurv; // store action
    lastCurv = a[0]; // store action

    rlBendingScheduler.Turn(a[0], t_rlAction);

    if (a.size()>1) // also modify the swimming period
    {
      if (TperiodPID) std::cout << "Warning: PID controller should not be used with RL." << std::endl;
      lastTact = a[1]; // store action
      current_period = periodPIDval;
      next_period = Tperiod * (1 + a[1]);
      transition_start = t_rlAction;
    }
  }

  ~CurvatureFish() override {
    _dealloc(rK); _dealloc(vK); _dealloc(rC); _dealloc(vC);
    _dealloc(rB); _dealloc(vB);
  }

  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override
  {
    const Real sb=.04*length, st=.95*length, wt=.01*length, wh=.04*length;
    if(s<0 or s>L) return 0;
    const Real w = (s<sb ? std::sqrt(2*wh*s -s*s) :
           (s<st ? wh-(wh-wt)*std::pow((s-sb)/(st-sb),1) : // pow(.,2) is 3D
           (wt * (L-s)/(L-st))));
    // std::cout << "s=" << s << ", w=" << w << std::endl;
    assert( w >= 0 );
    return w;
  }
};
