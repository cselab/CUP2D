//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "../ObstacleBlock.h"
#include "../Operator.h"
#include "FishUtilities.h"

struct FishSkin
{
  const size_t Npoints;
  Real * const xSurf;
  Real * const ySurf;
  Real * const normXSurf;
  Real * const normYSurf;
  Real * const midX;
  Real * const midY;
  FishSkin(const FishSkin& c) : Npoints(c.Npoints),
    xSurf(new Real[Npoints]), ySurf(new Real[Npoints])
    , normXSurf(new Real[Npoints-1]), normYSurf(new Real[Npoints-1])
    , midX(new Real[Npoints-1]), midY(new Real[Npoints-1])
    { }

  FishSkin(const size_t N): Npoints(N),
    xSurf(new Real[Npoints]), ySurf(new Real[Npoints])
    , normXSurf(new Real[Npoints-1]), normYSurf(new Real[Npoints-1])
    , midX(new Real[Npoints-1]), midY(new Real[Npoints-1])
    { }

  ~FishSkin() { delete [] xSurf; delete [] ySurf;
      delete [] normXSurf; delete [] normYSurf; delete [] midX; delete [] midY;
  }
};

struct FishData
{
 public:
  const Real length, Tperiod, phaseShift, h;
  const Real waveLength = 1;
  const Real amplitudeFactor;

  // Midline is discretized by more points in first fraction and last fraction:
  const Real fracRefined = 0.1, fracMid = 1 - 2*fracRefined;
  const Real dSmid_tgt = h / std::sqrt(2);
  const Real dSrefine_tgt = 0.125 * h;

  const int Nmid = (int)std::ceil(length * fracMid / dSmid_tgt / 8) * 8;
  const Real dSmid = length * fracMid / Nmid;

  const int Nend = (int)std::ceil( // here we ceil to be safer
    fracRefined * length * 2 / (dSmid + dSrefine_tgt)  / 4) * 4;
  const double dSref = fracRefined * length * 2 / Nend - dSmid;

  const int Nm = Nmid + 2 * Nend + 1; // plus 1 because we contain 0 and L

  Real * const rS; // arclength discretization points
  Real * const rX; // coordinates of midline discretization points
  Real * const rY;
  Real * const vX; // midline discretization velocities
  Real * const vY;
  Real * const norX; // normal vector to the midline discretization points
  Real * const norY;
  Real * const vNorX;
  Real * const vNorY;
  Real * const width;

  Real linMom[2], area, J, angMom; // for diagnostics
  // start and end indices in the arrays where the fish starts and ends (to ignore the extensions when interpolating the shapes)
  FishSkin upperSkin = FishSkin(Nm);
  FishSkin lowerSkin = FishSkin(Nm);
  virtual void resetAll();

 protected:

  template<typename T>
  inline void _rotate2D(const Real Rmatrix2D[2][2], T &x, T &y) const {
    const T p[2] = {x, y};
    x = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    y = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  template<typename T>
  inline void _translateAndRotate2D(const T pos[2], const Real Rmatrix2D[2][2], Real&x, Real&y) const {
    const Real p[2] = { x-pos[0], y-pos[1] };
    x = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    y = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }

  static Real* _alloc(const int N) {
    return new Real[N];
  }
  template<typename T>
  static void _dealloc(T * ptr) {
    if(ptr not_eq nullptr) { delete [] ptr; ptr=nullptr; }
  }

  inline Real _d_ds(const int idx, const Real*const vals, const int maxidx) const {
    if(idx==0) return (vals[idx+1]-vals[idx])/(rS[idx+1]-rS[idx]);
    else if(idx==maxidx-1) return (vals[idx]-vals[idx-1])/(rS[idx]-rS[idx-1]);
    else return ( (vals[idx+1]-vals[idx])/(rS[idx+1]-rS[idx]) +
                  (vals[idx]-vals[idx-1])/(rS[idx]-rS[idx-1]) )/2;
  }
  inline Real _integrationFac1(const int idx) const {
    return 2*width[idx];
  }
  inline Real _integrationFac2(const int idx) const {
    const Real dnorXi = _d_ds(idx, norX, Nm);
    const Real dnorYi = _d_ds(idx, norY, Nm);
    return 2*std::pow(width[idx],3)*(dnorXi*norY[idx] - dnorYi*norX[idx])/3;
  }
  inline Real _integrationFac3(const int idx) const {
    return 2*std::pow(width[idx],3)/3;
  }

  virtual void _computeMidlineNormals() const;

  virtual Real _width(const Real s, const Real L) = 0;

  void _computeWidth() {
    for(int i=0; i<Nm; ++i) width[i] = _width(rS[i], length);
  }

 public:
  FishData(Real L, Real Tp, Real phi, Real _h, const Real A=1);
  virtual ~FishData();

  Real integrateLinearMomentum(Real CoM[2], Real vCoM[2]);
  Real integrateAngularMomentum(Real & angVel);

  void changeToCoMFrameLinear(const Real CoM_internal[2], const Real vCoM_internal[2]) const;
  void changeToCoMFrameAngular(const Real theta_internal, const Real angvel_internal) const;

  void computeSurface() const;
  void surfaceToCOMFrame(const Real theta_internal,
                         const Real CoM_internal[2]) const;
  void surfaceToComputationalFrame(const Real theta_comp,
                                   const double CoM_interpolated[2]) const;
  void computeSkinNormals(const Real theta_comp, const double CoM_comp[3]) const;
  void writeMidline2File(const int step_id, std::string filename);

  virtual void computeMidline(const Real time, const Real dt) = 0;
};

struct AreaSegment
{
  const Real safe_distance;
  const std::pair<int, int> s_range;
  Real w[2], c[2];
  // should be normalized and >=0:
  Real normalI[2] = {(Real)1 , (Real)0};
  Real normalJ[2] = {(Real)0 , (Real)1};
  Real objBoxLabFr[2][2] = {{0,0}, {0,0}};
  Real objBoxObjFr[2][2] = {{0,0}, {0,0}};

  AreaSegment(std::pair<int,int> sr,const Real bb[2][2],const Real safe):
  safe_distance(safe), s_range(sr),
  w{ (bb[0][1]-bb[0][0])/2 + safe, (bb[1][1]-bb[1][0])/2 + safe },
  c{ (bb[0][1]+bb[0][0])/2,        (bb[1][1]+bb[1][0])/2 }
  { assert(w[0]>0); assert(w[1]>0); }

  void changeToComputationalFrame(const double position[2], const Real angle);
  bool isIntersectingWithAABB(const Real start[2],const Real end[2]) const;
};

struct PutFishOnBlocks
{
  const FishData & cfish;
  const Real position[2];
  const Real angle;
  const Real Rmatrix2D[2][2] = {
      {std::cos(angle), -std::sin(angle)},
      {std::sin(angle),  std::cos(angle)}
  };
  static inline Real eulerDistSq2D(const Real a[2], const Real b[2]) {
    return std::pow(a[0]-b[0],2) +std::pow(a[1]-b[1],2);
  }
  void changeVelocityToComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0], x[1]};
    x[0]=Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1]; // rotate (around CoM)
    x[1]=Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  template<typename T>
  void changeToComputationalFrame(T x[2]) const {
    const T p[2] = {x[0], x[1]};
    x[0] = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    x[1] = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
    x[0]+= position[0]; // translate
    x[1]+= position[1];
  }
  template<typename T>
  void changeFromComputationalFrame(T x[2]) const {
    const T p[2] = { x[0]-(T)position[0], x[1]-(T)position[1] };
    // rotate back around CoM
    x[0]=Rmatrix2D[0][0]*p[0] + Rmatrix2D[1][0]*p[1];
    x[1]=Rmatrix2D[0][1]*p[0] + Rmatrix2D[1][1]*p[1];
  }

  PutFishOnBlocks(const FishData& cf, const double pos[2],
    const Real ang): cfish(cf), position{(Real)pos[0],(Real)pos[1]}, angle(ang) { }
  virtual ~PutFishOnBlocks() {}

  void operator()(const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void constructSurface(  const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void constructInternl(  const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void signedDistanceSqrt(const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
};

class CurvatureFish : public FishData
{
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

 protected:
  Real * const rK;
  Real * const vK;
  Real * const rC;
  Real * const vC;
  Real * const rB;
  Real * const vB;

  // next scheduler is used to ramp-up the curvature from 0 during first period:
  Schedulers::ParameterSchedulerVector<6> curvatureScheduler;
  // next scheduler is used for midline-bending control points for RL:
  Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;
 public:

  CurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
  : FishData(L, T, phi, _h, _A),   rK(_alloc(Nm)),vK(_alloc(Nm)),
    rC(_alloc(Nm)),vC(_alloc(Nm)), rB(_alloc(Nm)),vB(_alloc(Nm)) {
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
    timeshift = 0;
    lastTime = 0;
    lastAvel = 0;
    time0 = 0;
    curvatureScheduler.resetAll();
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
                              const std::vector<double>&a)
  {
    assert(t_current >= t_rlAction);
    oldrCurv = lastCurv; // store action
    lastCurv = a[0]; // store action

    rlBendingScheduler.Turn(a[0], t_rlAction);
    //printf("Turning by %g at time %g with period %g.\n",
    //       a[0], t_current, t_rlAction);

    if (a.size()>1) // also modify the swimming period
    {
      lastTact = a[1]; // store action
      // this is arg of sinusoidal before change-of-Tp begins:
      const Real lastArg = (t_rlAction - time0)/periodPIDval + timeshift;
      // make sure change-of-Tp affects only body config for future times:
      timeshift = lastArg; // add phase shift that accounts for current body shape
      time0 = t_rlAction; // shift time axis of undulation
      periodPIDval = Tperiod * (1 + a[1]); // actually change period
      periodPIDdif = 0; // constant periodPIDval between infrequent actions
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
