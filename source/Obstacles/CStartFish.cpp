//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ioannis Mandralis (ioannima@ethz.ch).
//


#include "CStartFish.h"
#include "FishLibrary.h"
#include "FishUtilities.h"
#include <sstream>

using namespace cubism;

class ControlledCurvatureFish : public FishData
{
public:
    // Last baseline curvature
    Real lastB3 = 0;
    Real lastB4 = 0;
    Real lastB5 = 0;
    // Last undulatory curvature
    Real lastK3 = 0;
    Real lastK4 = 0;
    Real lastK5 = 0;
    // Last tail phase
    Real lastTau = 0;
    // Last alpha
    Real lastAlpha = 0;
    // Older baseline curvature
    Real oldrB3 = 0;
    Real oldrB4 = 0;
    Real oldrB5 = 0;
    // Older undulatory curvature
    Real oldrK3 = 0;
    Real oldrK4 = 0;
    Real oldrK5 = 0;
    // Older tail phase
    Real oldrTau = 0;
    // Older alpha
    Real oldrAlpha = 0;

    // Time for next action
    double t_next = 0.0;

    // Target location
    double target[2] = {0.0, 0.0};

protected:
    // Current curvature and curvature velocity
    Real * const rK;
    Real * const vK;
    // Current baseline curvature and curvature velocity
    Real * const rBC;
    Real * const vBC;
    // Current undulatory curvature and curvature velocity
    Real * const rUC;
    Real * const vUC;
    // Current tail phase and phase velocity
    Real tauTail;
    Real vTauTail;
    // Current alpha
    Real alpha;

    // Schedulers
    Schedulers::ParameterSchedulerVector<6> baselineCurvatureScheduler;
    Schedulers::ParameterSchedulerVector<6> undulatoryCurvatureScheduler;
    Schedulers::ParameterSchedulerScalar tauTailScheduler;

public:

    ControlledCurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
            : FishData(L, T, phi, _h, _A), rK(_alloc(Nm)), vK(_alloc(Nm)),
              rBC(_alloc(Nm)),vBC(_alloc(Nm)), rUC(_alloc(Nm)), vUC(_alloc(Nm)),
              tauTail(0.0), vTauTail(0.0), alpha(0.0) {
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
        lastAlpha = 0;
        oldrB3 = 0;
        oldrB4 = 0;
        oldrB5 = 0;
        oldrK3 = 0;
        oldrK4 = 0;
        oldrK5 = 0;
        oldrTau = 0;
        oldrAlpha = 0;

        t_next = 0.0;

        target[0] = 0.0;
        target[1] = 0.0;

        baselineCurvatureScheduler.resetAll();
        undulatoryCurvatureScheduler.resetAll();
        tauTailScheduler.resetAll();
        FishData::resetAll();
    }

    void schedule(const Real t_current, const std::vector<double>&a)
    {
        // Current time must be later than time at which action should be performed.
        assert(t_current >= t_rlAction);

        // Store last action into the older action placeholder
        oldrB3 = lastB3;
        oldrB4 = lastB4;
        oldrB5 = lastB5;
        oldrK3 = lastK3;
        oldrK4 = lastK4;
        oldrK5 = lastK5;
        oldrTau = lastTau;
        oldrAlpha = lastAlpha;

        // Store the new action into the last action placeholder
        lastB3 = a[0];
        lastB4 = a[1];
        lastB5 = a[2];
        lastK3 = a[3];
        lastK4 = a[4];
        lastK5 = a[5];
        lastTau = a[6];
        lastAlpha = a[7];

        // RL agent should output normalized curvature values as actions.
        double curvatureFactor = 1.0 / this->length;

        // Define the agent-prescribed curvature values
        const std::array<Real ,6> baselineCurvatureValues = {
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)lastB3 * curvatureFactor,
                (Real)lastB4 * curvatureFactor, (Real)lastB5 * curvatureFactor, (Real)0.0 * curvatureFactor
        };
        const std::array<Real ,6> undulatoryCurvatureValues = {
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)lastK3 * curvatureFactor,
                (Real)lastK4  * curvatureFactor, (Real)lastK5 * curvatureFactor, (Real)0.0 * curvatureFactor
        };

        // Using the agent-prescribed action duration get the final time of the prescribed action
        const Real actionDuration = (1 - lastAlpha) * 0.5 * this->Tperiod + lastAlpha * this->Tperiod;
        this->t_next = t_current + actionDuration;

        // Decide whether to use the current derivative for the cubic interpolation
        const bool useCurrentDerivative = true;

        // Act by scheduling a transition at the current time.
        baselineCurvatureScheduler.transition(t_current, t_current, this->t_next, baselineCurvatureValues, useCurrentDerivative);
        undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next, undulatoryCurvatureValues, useCurrentDerivative);
        tauTailScheduler.transition(t_current, t_current, this->t_next, lastTau, useCurrentDerivative);

        printf("Action duration is: %f\n", actionDuration);
        printf("t_next is: %f/n", this->t_next);
        printf("Scheduled a transition between %f and %f to baseline curvatures %f, %f, %f\n", t_current, t_next, lastB3, lastB4, lastB5);
        printf("Scheduled a transition between %f and %f to undulatory curvatures %f, %f, %f\n", t_current, t_next, lastK3, lastK4, lastK5);
        printf("Scheduled a transition between %f and %f to tau %f\n", t_current, t_next, lastTau);

    }

    ~ControlledCurvatureFish() override {
        _dealloc(rBC); _dealloc(vBC); _dealloc(rUC); _dealloc(vUC);
        _dealloc(rK); _dealloc(vK);
    }

    void computeMidline(const Real time, const Real dt) override;
    Real _width(const Real s, const Real L) override
    {
        const Real sb=.0862*length, st=.3448*length, wt=.0254*length, wh=.0635*length;
        if(s<0 or s>L) return 0;
        return (s<sb ? wh * std::sqrt(1 - std::pow((sb - s)/sb, 2)) :
                (s<st ? (-2*(wt-wh)-wt*(st-sb))*std::pow((s-sb)/(st-sb), 3)
                        + (3*(wt-wh)+wt*(st-sb))*std::pow((s-sb)/(st-sb), 2)
                        + wh: (wt - wt * std::pow((s-st)/(L-st), 2))));
    }
};

void ControlledCurvatureFish::computeMidline(const Real t, const Real dt)
{
    // Curvature control points along midline of fish, as in Gazzola et. al.
    const std::array<Real ,6> curvaturePoints = { (Real)0, (Real).2*length,
                                                  (Real).5*length, (Real).75*length, (Real).95*length, length};
//
//    if (t>=0.0 && act1){
//        std::vector<double> a{-3.19, -0.74, -0.44, -5.73, -2.73, -1.09, 0.74, 0.4};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.7* this->Tperiod && act2){
//        std::vector<double> a{0.0, 0.0, 0.0, -5.73, -2.73, -1.09, 0.74, 1.0};
//        this->schedule(t, a);
//        act2=false;
//    }

    const Real phi = 1.11;
//    const Real phi = 0.0;

    // Write values to placeholders
    baselineCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rBC, vBC); // writes to rBC, vBC
    undulatoryCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rUC, vUC); // writes to rUC, vUC
    tauTailScheduler.gimmeValues(t, tauTail, vTauTail); // writes to tauTail and vTauTail

#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {

        const Real tauS = tauTail * rS[i] / length;
        const Real vTauS = vTauTail * rS[i] / length;
        const Real arg = 2 * M_PI * (t/Tperiod - tauS) + phi;
        const Real vArg = 2 * M_PI / Tperiod - 2 * M_PI * vTauS;

        rK[i] = rBC[i] + rUC[i] * std::sin(arg);
        vK[i] = vBC[i] + rUC[i] * vArg * std::cos(arg) + vUC[i] * std::sin(arg);

        assert(not std::isnan(rK[i]));
        assert(not std::isinf(rK[i]));
        assert(not std::isnan(vK[i]));
        assert(not std::isinf(vK[i]));
    }

    // solve frenet to compute midline parameters
    IF2D_Frenet2D::solve(Nm, rS, rK,vK, rX,rY, vX,vY, norX,norY, vNorX,vNorY);
}

void CStartFish::resetAll() {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    cFish->resetAll();
    Fish::resetAll();
}

CStartFish::CStartFish(SimulationData&s, ArgumentParser&p, double C[2]):
        Fish(s,p,C)
{
    const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
    myFish = new ControlledCurvatureFish(length, Tperiod, phaseShift, sim.getH(), ampFac);
    printf("ControlledCurvatureFish %d %f %f %f\n",myFish->Nm, length, Tperiod, phaseShift);
}

void CStartFish::create(const std::vector<BlockInfo>& vInfo)
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    Fish::create(vInfo);
}

void CStartFish::act(const Real t_rlAction, const std::vector<double>& a) const
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->schedule(sim.time, a);
}

void CStartFish::setTarget(double desiredTarget[2]) const
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->target[0] = desiredTarget[0];
    cFish->target[1] = desiredTarget[1];
}

void CStartFish::getTarget(double outTarget[2]) const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    outTarget[0] = cFish->target[0];
    outTarget[1] = cFish->target[1];
}

std::vector<double> CStartFish::stateEscape() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<double> S(14,0);

    double com[2] = {0, 0}; this->getCenterOfMass(com);
    double radialDisplacement = this->getRadialDisplacement();
    double polarAngle = std::atan2(com[1], com[0]);

    S[0] = radialDisplacement / length; // distance from center
    S[1] = polarAngle; // polar angle
    S[2] = getOrientation();
    S[3] = getU() * Tperiod / length;
    S[4] = getV() * Tperiod / length;
    S[5] = getW() * Tperiod;
    S[6] = cFish->lastB3;
    S[7] = cFish->lastB4;
    S[8] = cFish->lastB5;
    S[9] = cFish->lastK3;
    S[10] = cFish->lastK4;
    S[11] = cFish->lastK5;
    S[12] = cFish->lastTau;
    S[13] = cFish->lastAlpha;
    return S;
}

std::vector<double> CStartFish::stateTarget() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    double com[2] = {0, 0}; this->getCenterOfMass(com);

    std::vector<double> S(15,0);

    S[0] = this->getDistanceFromTarget() / length; // normalized distance from target
    S[1] = (com[0] - cFish->target[0]) / length; // relative x position away from target
    S[2] = (com[1] - cFish->target[1]) / length; // relative y position away from target
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

double CStartFish::getRadialDisplacement() const {
    double com[2] = {0, 0};
    this->getCenterOfMass(com);
    double radialDisplacement = std::sqrt(std::pow((com[0] - this->origC[0]), 2) + std::pow((com[1] - this->origC[1]), 2));
    return radialDisplacement;
}

double CStartFish::getDistanceFromTarget() const {
    double com[2] = {0.0, 0.0};
    double target[2] = {0.0, 0.0};
    this->getCenterOfMass(com);
    this->getTarget(target);
    double distanceFromTarget = std::sqrt(std::pow((com[0] - target[0]), 2) + std::pow((com[1] - target[1]), 2));
    return distanceFromTarget;
}

double CStartFish::getTimeNextAct() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->t_next;
}