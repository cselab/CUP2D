#pragma once
#include "Fish.h"
#include "FishData.h"
class StefanFish : public Fish {
  const bool bCorrectTrajectory;
  const bool bCorrectPosition;

public:
  void act(const Real lTact, const std::vector<Real> &a) const;
  Real getLearnTPeriod() const;
  Real getPhase(const Real t) const;
  void resetAll() override;
  StefanFish(SimulationData &s, cubism::ArgumentParser &p, Real C[2]);
  void create(const std::vector<cubism::BlockInfo> &vInfo) override;
  std::vector<Real> state(const std::vector<double> &origin) const;
  std::vector<Real> state3D() const;
  ssize_t holdingBlockID(const std::array<Real, 2> pos) const;
  std::array<Real, 2> getShear(const std::array<Real, 2> pSurf) const;
  ssize_t holdingBlockID(const std::array<Real, 2> pos,
                         const std::vector<cubism::BlockInfo> &velInfo) const;
  std::array<int, 2> safeIdInBlock(const std::array<Real, 2> pos,
                                   const std::array<Real, 2> org,
                                   const Real invh) const;
  std::array<Real, 2>
  getShear(const std::array<Real, 2> pSurf, const std::array<Real, 2> normSurf,
           const std::vector<cubism::BlockInfo> &velInfo) const;
  virtual void saveRestart(FILE *f) override;
  virtual void loadRestart(FILE *f) override;
};
class CurvatureFish : public FishData {
  const Real amplitudeFactor, phaseShift, Tperiod;

public:
  Real curv_PID_fac = 0;
  Real curv_PID_dif = 0;
  Real avgDeltaY = 0;
  Real avgDangle = 0;
  Real avgAngVel = 0;
  Real lastTact = 0;
  Real lastCurv = 0;
  Real oldrCurv = 0;
  Real periodPIDval = Tperiod;
  Real periodPIDdif = 0;
  bool TperiodPID = false;
  Real time0 = 0;
  Real timeshift = 0;
  Real lastTime = 0;
  Real lastAvel = 0;
  Schedulers::ParameterSchedulerVector<6> curvatureScheduler;
  Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;
  Schedulers::ParameterSchedulerScalar periodScheduler;
  Real current_period = Tperiod;
  Real next_period = Tperiod;
  Real transition_start = 0.0;
  Real transition_duration = 0.1 * Tperiod;

protected:
  Real *const rK;
  Real *const vK;
  Real *const rC;
  Real *const vC;
  Real *const rB;
  Real *const vB;

public:
  CurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
      : FishData(L, _h), amplitudeFactor(_A), phaseShift(phi), Tperiod(T),
        rK(_alloc(Nm)), vK(_alloc(Nm)), rC(_alloc(Nm)), vC(_alloc(Nm)),
        rB(_alloc(Nm)), vB(_alloc(Nm)) {
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
  void correctTrajectory(const Real dtheta, const Real vtheta, const Real t,
                         const Real dt) {
    curv_PID_fac = dtheta;
    curv_PID_dif = vtheta;
  }
  void correctTailPeriod(const Real periodFac, const Real periodVel,
                         const Real t, const Real dt) {
    assert(periodFac > 0 && periodFac < 2);
    const Real lastArg = (lastTime - time0) / periodPIDval + timeshift;
    time0 = lastTime;
    timeshift = lastArg;
    periodPIDval = Tperiod * periodFac;
    periodPIDdif = Tperiod * periodVel;
    lastTime = t;
    TperiodPID = true;
  }
  void execute(const Real t_current, const Real t_rlAction,
               const std::vector<Real> &a) {
    assert(t_current >= t_rlAction);
    oldrCurv = lastCurv;
    lastCurv = a[0];
    rlBendingScheduler.Turn(a[0], t_rlAction);
    if (a.size() > 1) {
      if (TperiodPID)
        std::cout << "Warning: PID controller should not be used with RL."
                  << std::endl;
      lastTact = a[1];
      current_period = periodPIDval;
      next_period = Tperiod * (1 + a[1]);
      transition_start = t_rlAction;
    }
  }
  ~CurvatureFish() override {
    _dealloc(rK);
    _dealloc(vK);
    _dealloc(rC);
    _dealloc(vC);
    _dealloc(rB);
    _dealloc(vB);
  }
  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override {
    const Real sb = .04 * length, st = .95 * length, wt = .01 * length,
               wh = .04 * length;
    if (s < 0 or s > L)
      return 0;
    const Real w =
        (s < sb ? std::sqrt(2 * wh * s - s * s)
                : (s < st ? wh - (wh - wt) * std::pow((s - sb) / (st - sb), 1)
                          : (wt * (L - s) / (L - st))));
    assert(w >= 0);
    return w;
  }
};
