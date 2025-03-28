#include <mpi.h>
#include "config.h"
#include "ZebraFish.h"
#include "FishData.h"
#include "FishUtilities.h"
#include <sstream>
using namespace cubism;
class BehaviorCurvatureFish : public FishData {
  const Real Tperiod;

public:
  Real t_next = 0.0;
  Real target[2] = {0.0, 0.0};
  bool act1 = true;
  bool act2 = true;
  bool act3 = true;
  bool act4 = true;
  Real lastBeta = 0;
  Real lastKappa = 0;
  Real lastC = 0;
  Real lastTimingFactor = 0;
  Real oldrBeta = 0;
  Real oldrKappa = 0;
  Real oldrC = 0;
  Real oldrTimingFactor = 0;

protected:
  Real *const rK;
  Real *const vK;
  Real *const rBC;
  Real *const vBC;
  Real *const rUC;
  Real *const vUC;
  Real tauTail;
  Real vTauTail;
  Real alpha;
  Schedulers::ParameterSchedulerVector<6> baselineCurvatureScheduler;
  Schedulers::ParameterSchedulerVector<6> undulatoryCurvatureScheduler;
  Schedulers::ParameterSchedulerScalar tauTailScheduler;

public:
  BehaviorCurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
      : FishData(L, _h), Tperiod(T), rK(_alloc(Nm)), vK(_alloc(Nm)),
        rBC(_alloc(Nm)), vBC(_alloc(Nm)), rUC(_alloc(Nm)), vUC(_alloc(Nm)),
        tauTail(0.0), vTauTail(0.0), alpha(0.0) {
    _computeWidth();
    writeMidline2File(0, "initialCheck");
  }
  ~BehaviorCurvatureFish() override {
    _dealloc(rBC);
    _dealloc(vBC);
    _dealloc(rUC);
    _dealloc(vUC);
    _dealloc(rK);
    _dealloc(vK);
  }
  void resetAll() override {
    t_next = 0.0;
    target[0] = 0.0;
    target[1] = 0.0;
    act1 = true;
    act2 = true;
    act3 = true;
    act4 = true;
    lastBeta = 0;
    lastKappa = 0;
    lastC = 0;
    lastTimingFactor = 0;
    oldrBeta = 0;
    oldrKappa = 0;
    oldrC = 0;
    oldrTimingFactor = 0;
    baselineCurvatureScheduler.resetAll();
    undulatoryCurvatureScheduler.resetAll();
    tauTailScheduler.resetAll();
    FishData::resetAll();
  }
  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override {
    const Real sb = .0862 * length, st = .3448 * length, wt = .0254 * length,
               wh = .0635 * length;
    if (s < 0 or s > L)
      return 0;
    return (s < sb ? wh * std::sqrt(1 - std::pow((sb - s) / sb, 2))
                   : (s < st ? (-2 * (wt - wh) - wt * (st - sb)) *
                                       std::pow((s - sb) / (st - sb), 3) +
                                   (3 * (wt - wh) + wt * (st - sb)) *
                                       std::pow((s - sb) / (st - sb), 2) +
                                   wh
                             : (wt - wt * std::pow((s - st) / (L - st), 2))));
  }
  void burst(const Real t_current, const std::vector<Real> &a) {
    const Real tailPhase = 0.74;
    const Real modulationFactor = a[0];
    const Real timingFactor = a[1];
    const Real curvatureFactor = modulationFactor / this->length;
    const std::array<Real, 6> baselineCurvatureValues = {
        (Real)0.0 * curvatureFactor,  (Real)0.0 * curvatureFactor,
        (Real)-4.0 * curvatureFactor, (Real)-1.0 * curvatureFactor,
        (Real)-1.0 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const std::array<Real, 6> undulatoryCurvatureValues = {
        (Real)0.0 * curvatureFactor,  (Real)0.0 * curvatureFactor,
        (Real)-6.0 * curvatureFactor, (Real)-3.0 * curvatureFactor,
        (Real)-1.5 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const Real actionDuration = (1 - timingFactor) * 0.5 * this->Tperiod / 2 +
                                timingFactor * this->Tperiod / 2;
    this->t_next = t_current + actionDuration;
    const bool useCurrentDerivative = true;
    baselineCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                          baselineCurvatureValues,
                                          useCurrentDerivative);
    undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                            undulatoryCurvatureValues,
                                            useCurrentDerivative);
    tauTailScheduler.transition(t_current, t_current, this->t_next, tailPhase,
                                useCurrentDerivative);
    printf("Performing a burst with timingFactor %f, and modulationFactor %f\n",
           (double)timingFactor, (double)modulationFactor);
    printf("t_next is: %f\n", (double)this->t_next);
  }
  void scoot(const Real t_current, const std::vector<Real> &a) {
    const Real tailPhase = 0.74;
    const Real modulationFactor = a[0];
    const Real timingFactor = a[1];
    const Real curvatureFactor = modulationFactor / this->length;
    const std::array<Real, 6> baselineCurvatureValues = {
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const std::array<Real, 6> undulatoryCurvatureValues = {
        (Real)0.0 * curvatureFactor,     (Real)0.0 * curvatureFactor,
        (Real)2.57136 * curvatureFactor, (Real)3.75425 * curvatureFactor,
        (Real)5.09147 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const Real actionDuration =
        (1 - timingFactor) * 0.5 * this->Tperiod + timingFactor * this->Tperiod;
    this->t_next = t_current + actionDuration;
    const bool useCurrentDerivative = true;
    baselineCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                          baselineCurvatureValues,
                                          useCurrentDerivative);
    undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                            undulatoryCurvatureValues,
                                            useCurrentDerivative);
    tauTailScheduler.transition(t_current, t_current, this->t_next, tailPhase,
                                useCurrentDerivative);
    printf("Performing a scoot with timingFactor %f, and modulationFactor %f\n",
           (double)timingFactor, (double)modulationFactor);
    printf("t_next is: %f\n", (double)this->t_next);
  }
  void coast(const Real t_current, const std::vector<Real> &a) {
    const Real tailPhase = 0.0;
    const Real modulationFactor = a[0];
    const Real timingFactor = a[1];
    const Real curvatureFactor = modulationFactor / this->length;
    const std::array<Real, 6> baselineCurvatureValues = {
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const std::array<Real, 6> undulatoryCurvatureValues = {
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
        (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const Real actionDuration =
        (1 - timingFactor) * 0.5 * this->Tperiod + timingFactor * this->Tperiod;
    this->t_next = t_current + actionDuration;
    const bool useCurrentDerivative = true;
    baselineCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                          baselineCurvatureValues,
                                          useCurrentDerivative);
    undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                            undulatoryCurvatureValues,
                                            useCurrentDerivative);
    tauTailScheduler.transition(t_current, t_current, this->t_next, tailPhase,
                                useCurrentDerivative);
    printf("Performing a coast with timingFactor %f, and modulationFactor %f\n",
           (double)timingFactor, (double)modulationFactor);
    printf("t_next is: %f\n", (double)this->t_next);
  }
  void hybrid(const Real t_current, const std::vector<Real> &a) {
    oldrBeta = lastBeta;
    oldrKappa = lastKappa;
    oldrC = lastC;
    oldrTimingFactor = lastTimingFactor;
    lastBeta = a[0];
    lastKappa = a[1];
    lastC = a[2];
    lastTimingFactor = a[3];
    const Real tailPhase = 0.74;
    const Real baselineCurvatureFactor = lastC * lastBeta / this->length;
    const Real undulatoryCurvatureFactor = 1 / this->length;
    const std::array<Real, 6> baselineCurvatureValues = {
        (Real)0.0 * baselineCurvatureFactor,
        (Real)0.0 * baselineCurvatureFactor,
        (Real)-4.0 * baselineCurvatureFactor,
        (Real)-1.0 * baselineCurvatureFactor,
        (Real)-1.0 * baselineCurvatureFactor,
        (Real)0.0 * baselineCurvatureFactor};
    const std::array<Real, 6> undulatoryCurvatureValuesScoot = {
        (Real)0.0 * undulatoryCurvatureFactor,
        (Real)0.0 * undulatoryCurvatureFactor,
        (Real)2.57136 * undulatoryCurvatureFactor,
        (Real)3.75425 * undulatoryCurvatureFactor,
        (Real)5.09147 * undulatoryCurvatureFactor,
        (Real)0.0 * undulatoryCurvatureFactor};
    const std::array<Real, 6> undulatoryCurvatureValuesBurst = {
        (Real)0.0 * undulatoryCurvatureFactor,
        (Real)0.0 * undulatoryCurvatureFactor,
        (Real)-6.0 * undulatoryCurvatureFactor,
        (Real)-3.0 * undulatoryCurvatureFactor,
        (Real)-1.5 * undulatoryCurvatureFactor,
        (Real)0.0 * undulatoryCurvatureFactor};
    std::array<Real, 6> undulatoryCurvatureValues = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 6; i++) {
      undulatoryCurvatureValues[i] =
          lastC * ((1 - lastKappa) * undulatoryCurvatureValuesScoot[i] +
                   lastKappa * undulatoryCurvatureValuesBurst[i]);
    }
    const Real actionDuration = (1 - lastTimingFactor) * 0.5 * this->Tperiod +
                                lastTimingFactor * this->Tperiod;
    this->t_next = t_current + actionDuration;
    const bool useCurrentDerivative = true;
    baselineCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                          baselineCurvatureValues,
                                          useCurrentDerivative);
    undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                            undulatoryCurvatureValues,
                                            useCurrentDerivative);
    tauTailScheduler.transition(t_current, t_current, this->t_next, tailPhase,
                                useCurrentDerivative);
    printf("Performing a hybrid action with beta %f, kappa %f, c %f\n",
           (double)lastBeta, (double)lastKappa, (double)lastC);
    printf("t_next is: %f\n", (double)this->t_next);
  }
};
void BehaviorCurvatureFish::computeMidline(const Real t, const Real dt) {
  const std::array<Real, 6> curvaturePoints = {(Real)0,
                                               (Real).2 * length,
                                               (Real).5 * length,
                                               (Real).75 * length,
                                               (Real).95 * length,
                                               length};
  const Real phi = 1.11;
  baselineCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rBC, vBC);
  undulatoryCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rUC,
                                           vUC);
  tauTailScheduler.gimmeValues(t, tauTail, vTauTail);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm; ++i) {
    const Real tauS = tauTail * rS[i] / length;
    const Real vTauS = vTauTail * rS[i] / length;
    const Real arg = 2 * M_PI * (t / Tperiod - tauS) + phi;
    const Real vArg = 2 * M_PI / Tperiod - 2 * M_PI * vTauS;
    rK[i] = rBC[i] + rUC[i] * std::sin(arg);
    vK[i] = vBC[i] + rUC[i] * vArg * std::cos(arg) + vUC[i] * std::sin(arg);
    assert(not std::isnan(rK[i]));
    assert(not std::isinf(rK[i]));
    assert(not std::isnan(vK[i]));
    assert(not std::isinf(vK[i]));
  }
  IF2D_Frenet2D::solve(Nm, rS, rK, vK, rX, rY, vX, vY, norX, norY, vNorX,
                       vNorY);
}
ZebraFish::ZebraFish(SimulationData &s, ArgumentParser &p, Real C[2])
    : Fish(s, p, C) {
  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  myFish =
      new BehaviorCurvatureFish(length, Tperiod, phaseShift, sim.minH, ampFac);
  if (sim.rank == 0 && s.verbose)
    printf("[CUP2D] - BehaviorCurvatureFish %d %f %f %f\n", myFish->Nm,
           (double)length, (double)Tperiod, (double)phaseShift);
}
void ZebraFish::resetAll() {
  BehaviorCurvatureFish *const cFish =
      dynamic_cast<BehaviorCurvatureFish *>(myFish);
  if (cFish == nullptr) {
    printf("Someone touched my fish\n");
    abort();
  }
  cFish->resetAll();
  Fish::resetAll();
}
void ZebraFish::create(const std::vector<BlockInfo> &vInfo) {
  BehaviorCurvatureFish *const cFish =
      dynamic_cast<BehaviorCurvatureFish *>(myFish);
  if (cFish == nullptr) {
    printf("Someone touched my fish\n");
    abort();
  }
  Fish::create(vInfo);
}
void ZebraFish::act(const Real t_rlAction, const std::vector<Real> &a) const {
  BehaviorCurvatureFish *const cFish =
      dynamic_cast<BehaviorCurvatureFish *>(myFish);
  cFish->hybrid(sim.time, a);
}
std::vector<Real> ZebraFish::state() const {
  const BehaviorCurvatureFish *const cFish =
      dynamic_cast<BehaviorCurvatureFish *>(myFish);
  std::vector<Real> S(10, 0);
  Real com[2] = {0, 0};
  this->getCenterOfMass(com);
  Real radialDisplacement = this->getRadialDisplacement();
  Real polarAngle = std::atan2(com[1], com[0]);
  S[0] = radialDisplacement / length;
  S[1] = polarAngle;
  S[2] = getOrientation();
  S[3] = getU() * Tperiod / length;
  S[4] = getV() * Tperiod / length;
  S[5] = getW() * Tperiod;
  S[6] = cFish->lastBeta;
  S[7] = cFish->lastKappa;
  S[8] = cFish->lastC;
  S[9] = cFish->lastTimingFactor;
  return S;
}
void ZebraFish::setTarget(Real inTarget[2]) const {
  BehaviorCurvatureFish *const cFish =
      dynamic_cast<BehaviorCurvatureFish *>(myFish);
  cFish->target[0] = inTarget[0];
  cFish->target[1] = inTarget[1];
}
void ZebraFish::getTarget(Real outTarget[2]) const {
  const BehaviorCurvatureFish *const cFish =
      dynamic_cast<BehaviorCurvatureFish *>(myFish);
  outTarget[0] = cFish->target[0];
  outTarget[1] = cFish->target[1];
}
Real ZebraFish::getRadialDisplacement() const {
  Real com[2] = {0, 0};
  this->getCenterOfMass(com);
  Real radialDisplacement = std::sqrt(std::pow((com[0] - this->origC[0]), 2) +
                                      std::pow((com[1] - this->origC[1]), 2));
  return radialDisplacement;
}
Real ZebraFish::getDistanceFromTarget() const {
  Real com[2] = {0.0, 0.0};
  Real target[2] = {0.0, 0.0};
  this->getCenterOfMass(com);
  this->getTarget(target);
  Real distanceFromTarget = std::sqrt(std::pow((com[0] - target[0]), 2) +
                                      std::pow((com[1] - target[1]), 2));
  return distanceFromTarget;
}
Real ZebraFish::getTimeNextAct() const {
  const BehaviorCurvatureFish *const cFish =
      dynamic_cast<BehaviorCurvatureFish *>(myFish);
  return cFish->t_next;
}
