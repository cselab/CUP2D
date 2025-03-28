#include "config.h"
#include "CStartFish.h"
#include "../Utils/BufferedLogger.h"
#include "FishData.h"
#include "FishUtilities.h"
#include <sstream>
using namespace cubism;
class ControlledCurvatureFish : public FishData {
  const Real Tperiod;

public:
  Real lastB3 = 0;
  Real lastB4 = 0;
  Real lastB5 = 0;
  Real lastK3 = 0;
  Real lastK4 = 0;
  Real lastK5 = 0;
  Real lastTau = 0;
  Real lastPhiUndulatory = 0;
  Real lastAlpha = 0;
  Real oldrB3 = 0;
  Real oldrB4 = 0;
  Real oldrB5 = 0;
  Real oldrK3 = 0;
  Real oldrK4 = 0;
  Real oldrK5 = 0;
  Real oldrTau = 0;
  Real oldrAlpha = 0;
  Real oldrPhiUndulatory = 0;
  Real dTprop = 0.0;
  bool firstAction = true;
  Real t_next = 0.0;
  Real target[2] = {0.0, 0.0};
  Real virtualOrigin[2] = {0.5, 0.5};
  Real energyExpended = 0.0;
  Real energyBudget = 0.0;
  Real nextDump = 0.0;
  bool act1 = true;
  bool act2 = true;
  bool act3 = true;
  bool act4 = true;
  bool act5 = true;

protected:
  Real *const rK;
  Real *const vK;
  Real *const rBC;
  Real *const vBC;
  Real *const rUC;
  Real *const vUC;
  Real tauTail;
  Real vTauTail;
  Real phiUndulatory;
  Real vPhiUndulatory;
  Real alpha;
  Schedulers::ParameterSchedulerVector<6> baselineCurvatureScheduler;
  Schedulers::ParameterSchedulerVector<6> undulatoryCurvatureScheduler;
  Schedulers::ParameterSchedulerScalar tauTailScheduler;
  Schedulers::ParameterSchedulerScalar phiScheduler;

public:
  ControlledCurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
      : FishData(L, _h), Tperiod(T), rK(_alloc(Nm)), vK(_alloc(Nm)),
        rBC(_alloc(Nm)), vBC(_alloc(Nm)), rUC(_alloc(Nm)), vUC(_alloc(Nm)),
        tauTail(0.0), vTauTail(0.0), phiUndulatory(0.0), vPhiUndulatory(0.0),
        alpha(0.0) {
    _computeWidth();
    writeMidline2File(0, "initialCheck");
  }
  void resetAll() override {
    lastB3 = 0;
    lastB4 = 0;
    lastB5 = 0;
    lastK3 = 0;
    lastK4 = 0;
    lastK5 = 0;
    lastTau = 0;
    lastPhiUndulatory = 0;
    lastAlpha = 0;
    oldrB3 = 0;
    oldrB4 = 0;
    oldrB5 = 0;
    oldrK3 = 0;
    oldrK4 = 0;
    oldrK5 = 0;
    oldrTau = 0;
    oldrPhiUndulatory = 0;
    oldrAlpha = 0;
    dTprop = 0.0;
    firstAction = true;
    energyExpended = 0.0;
    energyBudget = 0.0;
    nextDump = 0.0;
    t_next = 0.0;
    target[0] = 0.0;
    target[1] = 0.0;
    act1 = true;
    act2 = true;
    act3 = true;
    act4 = true;
    act5 = true;
    baselineCurvatureScheduler.resetAll();
    undulatoryCurvatureScheduler.resetAll();
    tauTailScheduler.resetAll();
    FishData::resetAll();
  }
  void schedule(const Real t_current, const std::vector<Real> &a) {
    oldrB3 = lastB3;
    oldrB4 = lastB4;
    oldrB5 = lastB5;
    oldrK3 = lastK3;
    oldrK4 = lastK4;
    oldrK5 = lastK5;
    oldrTau = lastTau;
    oldrAlpha = lastAlpha;
    oldrPhiUndulatory = lastPhiUndulatory;
    lastB3 = a[0];
    lastB4 = a[1];
    lastB5 = a[2];
    lastK3 = a[3];
    lastK4 = a[4];
    lastK5 = a[5];
    lastTau = a[6];
    lastAlpha = a[7];
    lastPhiUndulatory = a[8];
    Real curvatureFactor = 1.0 / this->length;
    const std::array<Real, 6> baselineCurvatureValues = {
        (Real)0.0 * curvatureFactor,    (Real)0.0 * curvatureFactor,
        (Real)lastB3 * curvatureFactor, (Real)lastB4 * curvatureFactor,
        (Real)lastB5 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const std::array<Real, 6> undulatoryCurvatureValues = {
        (Real)0.0 * curvatureFactor,    (Real)0.0 * curvatureFactor,
        (Real)lastK3 * curvatureFactor, (Real)lastK4 * curvatureFactor,
        (Real)lastK5 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const Real actionDuration =
        (1 - lastAlpha) * 0.5 * this->Tperiod + lastAlpha * this->Tperiod;
    this->t_next = t_current + actionDuration;
    const bool useCurrentDerivative = true;
    baselineCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                          baselineCurvatureValues,
                                          useCurrentDerivative);
    undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                            undulatoryCurvatureValues,
                                            useCurrentDerivative);
    tauTailScheduler.transition(t_current, t_current, this->t_next, lastTau,
                                useCurrentDerivative);
    if (firstAction) {
      printf("FIRST ACTION %f\n", (double)lastPhiUndulatory);
      phiScheduler.transition(t_current, t_current, this->t_next,
                              lastPhiUndulatory, lastPhiUndulatory);
      firstAction = false;
    } else {
      printf("Next action %f\n", (double)lastPhiUndulatory);
      phiScheduler.transition(t_current, t_current, this->t_next,
                              lastPhiUndulatory, useCurrentDerivative);
    }
  }
  void scheduleCStart(const Real t_current, const std::vector<Real> &a) {
    oldrB3 = lastB3;
    oldrB4 = lastB4;
    oldrB5 = lastB5;
    oldrK3 = lastK3;
    oldrK4 = lastK4;
    oldrK5 = lastK5;
    oldrTau = lastTau;
    lastB3 = a[0];
    lastB4 = a[1];
    lastB5 = a[2];
    lastK3 = a[3];
    lastK4 = a[4];
    lastK5 = a[5];
    lastTau = a[6];
    Real curvatureFactor = 1.0 / this->length;
    const std::array<Real, 6> baselineCurvatureValues = {
        (Real)0.0 * curvatureFactor,    (Real)0.0 * curvatureFactor,
        (Real)lastB3 * curvatureFactor, (Real)lastB4 * curvatureFactor,
        (Real)lastB5 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const std::array<Real, 6> undulatoryCurvatureValues = {
        (Real)0.0 * curvatureFactor,    (Real)0.0 * curvatureFactor,
        (Real)lastK3 * curvatureFactor, (Real)lastK4 * curvatureFactor,
        (Real)lastK5 * curvatureFactor, (Real)0.0 * curvatureFactor};
    const Real duration1 = 0.7 * this->Tperiod;
    const Real duration2 = this->Tperiod;
    Real actionDuration = 0.0;
    if (t_current < duration1) {
      actionDuration = duration1;
      this->t_next = t_current + actionDuration;
    } else {
      actionDuration = duration2;
      this->t_next = t_current + actionDuration;
    }
    const bool useCurrentDerivative = true;
    baselineCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                          baselineCurvatureValues,
                                          useCurrentDerivative);
    undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next,
                                            undulatoryCurvatureValues,
                                            useCurrentDerivative);
    tauTailScheduler.transition(t_current, t_current, this->t_next, lastTau,
                                useCurrentDerivative);
    printf("\nAction duration is: %f\n", (double)actionDuration);
    printf("t_next is: %f\n", (double)this->t_next);
    printf("Scheduled a transition between %f and %f to baseline curvatures "
           "%f, %f, %f\n",
           (double)t_current, (double)t_next, (double)lastB3, (double)lastB4,
           (double)lastB5);
    printf("Scheduled a transition between %f and %f to undulatory curvatures "
           "%f, %f, %f\n",
           (double)t_current, (double)t_next, (double)lastK3, (double)lastK4,
           (double)lastK5);
    printf("Scheduled a transition between %f and %f to tau %f\n",
           (double)t_current, (double)t_next, (double)lastTau);
  }
  ~ControlledCurvatureFish() override {
    _dealloc(rBC);
    _dealloc(vBC);
    _dealloc(rUC);
    _dealloc(vUC);
    _dealloc(rK);
    _dealloc(vK);
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
};
void ControlledCurvatureFish::computeMidline(const Real t, const Real dt) {
  const std::array<Real, 6> curvaturePoints = {(Real)0,
                                               (Real).2 * length,
                                               (Real).5 * length,
                                               (Real).75 * length,
                                               (Real).95 * length,
                                               length};
  baselineCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rBC, vBC);
  undulatoryCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rUC,
                                           vUC);
  tauTailScheduler.gimmeValues(t, tauTail, vTauTail);
  phiScheduler.gimmeValues(t, phiUndulatory, vPhiUndulatory);
  const Real curvMax = 2 * M_PI / length;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm; ++i) {
    const Real tauS = tauTail * rS[i] / length;
    const Real vTauS = vTauTail * rS[i] / length;
    const Real arg = 2 * M_PI * (t / Tperiod - tauS) + 2 * M_PI * phiUndulatory;
    const Real vArg =
        2 * M_PI / Tperiod - 2 * M_PI * vTauS + 2 * M_PI * vPhiUndulatory;
    const Real curvCmd = rBC[i] + rUC[i] * std::sin(arg);
    const Real curvCmdVel =
        vBC[i] + rUC[i] * vArg * std::cos(arg) + vUC[i] * std::sin(arg);
    if (std::abs(curvCmd) >= curvMax) {
      rK[i] = curvCmd > 0 ? curvMax : -curvMax;
      vK[i] = 0;
    } else {
      rK[i] = curvCmd;
      vK[i] = curvCmdVel;
    }
    assert(not std::isnan(rK[i]));
    assert(not std::isinf(rK[i]));
    assert(not std::isnan(vK[i]));
    assert(not std::isinf(vK[i]));
  }
  FILE *f1 = fopen("curvature_values.dat", "a+");
  fprintf(f1, "%f  %g  %d\n", (double)t, (double)rK[Nmid], Nmid);
  fclose(f1);
  IF2D_Frenet2D::solve(Nm, rS, rK, vK, rX, rY, vX, vY, norX, norY, vNorX,
                       vNorY);
}
void CStartFish::resetAll() {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  if (cFish == nullptr) {
    printf("Someone touched my fish\n");
    abort();
  }
  cFish->resetAll();
  Fish::resetAll();
}
CStartFish::CStartFish(SimulationData &s, ArgumentParser &p, Real C[2])
    : Fish(s, p, C) {
  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  myFish = new ControlledCurvatureFish(length, Tperiod, phaseShift, sim.minH,
                                       ampFac);
  if (sim.rank == 0 && s.verbose)
    printf("[CUP2D] - ControlledCurvatureFish %d %f %f %f\n", myFish->Nm,
           (double)length, (double)Tperiod, (double)phaseShift);
}
void CStartFish::create(const std::vector<BlockInfo> &vInfo) {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  if (cFish == nullptr) {
    printf("Someone touched my fish\n");
    abort();
  }
  Fish::create(vInfo);
}
void CStartFish::act(const Real t_rlAction, const std::vector<Real> &a) const {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  cFish->schedule(sim.time, a);
}
void CStartFish::actCStart(const Real lTact, const std::vector<Real> &a) const {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  cFish->scheduleCStart(sim.time, a);
}
void CStartFish::setTarget(Real desiredTarget[2]) const {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  cFish->target[0] = desiredTarget[0];
  cFish->target[1] = desiredTarget[1];
}
void CStartFish::getTarget(Real outTarget[2]) const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  outTarget[0] = cFish->target[0];
  outTarget[1] = cFish->target[1];
}
std::vector<Real> CStartFish::stateEscapeTradeoff() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  std::vector<Real> S(26, 0);
  S[0] = this->getRadialDisplacement() / length;
  S[1] = cFish->dTprop / length;
  S[2] = this->getPolarAngle();
  S[3] = cFish->energyBudget - cFish->energyExpended;
  S[4] = getOrientation();
  S[5] = getU() * Tperiod / length;
  S[6] = getV() * Tperiod / length;
  S[7] = getW() * Tperiod;
  S[8] = cFish->lastB3;
  S[9] = cFish->lastB4;
  S[10] = cFish->lastB5;
  S[11] = cFish->lastK3;
  S[12] = cFish->lastK4;
  S[13] = cFish->lastK5;
  S[14] = cFish->lastTau;
  S[15] = cFish->lastAlpha;
  S[16] = cFish->lastPhiUndulatory;
  S[17] = cFish->oldrB3;
  S[18] = cFish->oldrB4;
  S[19] = cFish->oldrB5;
  S[20] = cFish->oldrK3;
  S[21] = cFish->oldrK4;
  S[22] = cFish->oldrK5;
  S[23] = cFish->oldrTau;
  S[24] = cFish->oldrAlpha;
  S[25] = cFish->oldrPhiUndulatory;
  return S;
}
std::vector<Real> CStartFish::stateEscape() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  std::vector<Real> S(25, 0);
  S[0] = this->getRadialDisplacement() / length;
  S[1] = this->getPolarAngle();
  S[2] = cFish->energyExpended;
  S[3] = getOrientation();
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastB3;
  S[8] = cFish->lastB4;
  S[9] = cFish->lastB5;
  S[10] = cFish->lastK3;
  S[11] = cFish->lastK4;
  S[12] = cFish->lastK5;
  S[13] = cFish->lastTau;
  S[14] = cFish->lastAlpha;
  S[15] = cFish->lastPhiUndulatory;
  S[16] = cFish->oldrB3;
  S[17] = cFish->oldrB4;
  S[18] = cFish->oldrB5;
  S[19] = cFish->oldrK3;
  S[20] = cFish->oldrK4;
  S[21] = cFish->oldrK5;
  S[22] = cFish->oldrTau;
  S[23] = cFish->oldrAlpha;
  S[24] = cFish->oldrPhiUndulatory;
  return S;
}
std::vector<Real> CStartFish::stateSequentialEscape() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  std::vector<Real> S(25, 0);
  Real com[2] = {0.0, 0.0};
  this->getCenterOfMass(com);
  bool propulsionForward = com[0] <= this->origC[0];
  Real signedRadialDisplacement = 0.0;
  if (propulsionForward) {
    signedRadialDisplacement = this->getRadialDisplacement() / length;
  } else {
    signedRadialDisplacement = -this->getRadialDisplacement() / length;
  }
  S[0] = signedRadialDisplacement;
  S[1] = this->getPolarAngle();
  S[2] = cFish->energyExpended;
  S[3] = getOrientation();
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastB3;
  S[8] = cFish->lastB4;
  S[9] = cFish->lastB5;
  S[10] = cFish->lastK3;
  S[11] = cFish->lastK4;
  S[12] = cFish->lastK5;
  S[13] = cFish->lastTau;
  S[14] = cFish->lastAlpha;
  S[15] = cFish->lastPhiUndulatory;
  S[16] = cFish->oldrB3;
  S[17] = cFish->oldrB4;
  S[18] = cFish->oldrB5;
  S[19] = cFish->oldrK3;
  S[20] = cFish->oldrK4;
  S[21] = cFish->oldrK5;
  S[22] = cFish->oldrTau;
  S[23] = cFish->oldrAlpha;
  S[24] = cFish->oldrPhiUndulatory;
  return S;
}
std::vector<Real> CStartFish::stateEscapeVariableEnergy() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  std::vector<Real> S(25, 0);
  S[0] = this->getRadialDisplacement() / length;
  S[1] = this->getPolarAngle();
  S[2] = (cFish->energyBudget - cFish->energyExpended);
  S[3] = getOrientation();
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastB3;
  S[8] = cFish->lastB4;
  S[9] = cFish->lastB5;
  S[10] = cFish->lastK3;
  S[11] = cFish->lastK4;
  S[12] = cFish->lastK5;
  S[13] = cFish->lastTau;
  S[14] = cFish->lastAlpha;
  S[15] = cFish->lastPhiUndulatory;
  S[16] = cFish->oldrB3;
  S[17] = cFish->oldrB4;
  S[18] = cFish->oldrB5;
  S[19] = cFish->oldrK3;
  S[20] = cFish->oldrK4;
  S[21] = cFish->oldrK5;
  S[22] = cFish->oldrTau;
  S[23] = cFish->oldrAlpha;
  S[24] = cFish->oldrPhiUndulatory;
  return S;
}
std::vector<Real> CStartFish::stateCStart() const {
  std::vector<Real> S(2, 0);
  S[0] = this->getRadialDisplacement() / length;
  S[1] = this->getPolarAngle();
  return S;
}
std::vector<Real> CStartFish::stateTarget() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  Real com[2] = {0, 0};
  this->getCenterOfMass(com);
  std::vector<Real> S(15, 0);
  S[0] = this->getDistanceFromTarget() / length;
  S[1] = (com[0] - cFish->target[0]) / length;
  S[2] = (com[1] - cFish->target[1]) / length;
  S[3] = getOrientation();
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastB3;
  S[8] = cFish->lastB4;
  S[9] = cFish->lastB5;
  S[10] = cFish->lastK3;
  S[11] = cFish->lastK4;
  S[12] = cFish->lastK5;
  S[13] = cFish->lastTau;
  S[14] = cFish->lastAlpha;
  return S;
}
Real CStartFish::getRadialDisplacement() const {
  Real com[2] = {0, 0};
  this->getCenterOfMass(com);
  Real radialDisplacement = std::sqrt(std::pow((com[0] - this->origC[0]), 2) +
                                      std::pow((com[1] - this->origC[1]), 2));
  return radialDisplacement;
}
Real CStartFish::getDistanceFromTarget() const {
  Real com[2] = {0.0, 0.0};
  Real target[2] = {0.0, 0.0};
  this->getCenterOfMass(com);
  this->getTarget(target);
  Real distanceFromTarget = std::sqrt(std::pow((com[0] - target[0]), 2) +
                                      std::pow((com[1] - target[1]), 2));
  return distanceFromTarget;
}
void CStartFish::setEnergyExpended(const Real energyExpended) {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  cFish->energyExpended = energyExpended;
}
void CStartFish::setDistanceTprop(const Real distanceTprop) {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  Real com[2] = {0.0, 0.0};
  this->getCenterOfMass(com);
  bool propulsionForward = com[0] <= this->origC[0];
  if (propulsionForward) {
    cFish->dTprop = distanceTprop;
  } else {
    cFish->dTprop = -distanceTprop;
  }
}
Real CStartFish::getDistanceTprop() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  return cFish->dTprop;
}
Real CStartFish::getEnergyExpended() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  return cFish->energyExpended;
}
Real CStartFish::getTimeNextAct() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  return cFish->t_next;
}
Real CStartFish::getPolarAngle() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  Real com[2] = {0, 0};
  this->getCenterOfMass(com);
  Real polarAngle = std::atan2(com[1] - cFish->virtualOrigin[1],
                               com[0] - cFish->virtualOrigin[0]);
  return polarAngle;
}
void CStartFish::setVirtualOrigin(const Real vo[2]) {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  cFish->virtualOrigin[0] = vo[0];
  cFish->virtualOrigin[1] = vo[1];
}
void CStartFish::setEnergyBudget(const Real baselineEnergy) {
  ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  cFish->energyBudget = baselineEnergy;
}
Real CStartFish::getEnergyBudget() const {
  const ControlledCurvatureFish *const cFish =
      dynamic_cast<ControlledCurvatureFish *>(myFish);
  return cFish->energyBudget;
}
