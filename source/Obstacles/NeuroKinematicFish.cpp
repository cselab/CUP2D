//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ioannis Mandralis (ioannima@ethz.ch).
//


#include "NeuroKinematicFish.h"
#include "FishLibrary.h"
#include "FishUtilities.h"
#include <sstream>


using namespace cubism;

class NeuroFish : public FishData
{
public:
    double t_next = 0.0; // Time for next action
    double target[2] = {0.0, 0.0}; // Target location
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

    Real lastAmplitude = 0;
    Real lastDelay = 0;
    Real lastFireTime = 0;
    Real oldrAmplitude = 0;
    Real oldrDelay = 0;
    Real oldrFireTime = 0;

protected:
    Real * const rK; // Current curvature
    Real * const vK; // Current curvature velocity
    Real * const rBC; // Current baseline curvature
    Real * const vBC; // Current baseline curvature velocity
    Real * const rUC; // Current undulatory curvature
    Real * const vUC; // Current undulatory curvature velocity
    Real * const rMuscSignal;
    Real * const vMuscSignal;
    Real * const spatialDerivativeMuscSignal;
    Real * const spatialDerivativeDMuscSignal;
    Real tauTail; // Current tail phase
    Real vTauTail; // Current tail phase velocity
    Real alpha; // Current alpha
    Schedulers::ParameterSchedulerVector<6> baselineCurvatureScheduler;  // baseline scheduler
    Schedulers::ParameterSchedulerVector<6> undulatoryCurvatureScheduler; // undulation scheduler
    Schedulers::ParameterSchedulerScalar tauTailScheduler; // phase scheduler
    Schedulers::ParameterSchedulerNeuroKinematic<9> neuroKinematicScheduler;

public:
    NeuroFish(Real L, Real T, Real phi, Real _h, Real _A)
            : FishData(L, T, phi, _h, _A), rK(_alloc(Nm)), vK(_alloc(Nm)),
              rBC(_alloc(Nm)),vBC(_alloc(Nm)), rUC(_alloc(Nm)), vUC(_alloc(Nm)),
              rMuscSignal(_alloc(Nm)), vMuscSignal(_alloc(Nm)),
              spatialDerivativeMuscSignal(_alloc(Nm)), spatialDerivativeDMuscSignal(_alloc(Nm)),
              tauTail(0.0), vTauTail(0.0), alpha(0.0) {
        _computeWidth();
        writeMidline2File(0, "initialCheck");
    }
    ~NeuroFish() override {
        _dealloc(rBC); _dealloc(vBC); _dealloc(rUC); _dealloc(vUC);
        _dealloc(rK); _dealloc(vK); _dealloc(rMuscSignal); _dealloc(vMuscSignal);
        _dealloc(spatialDerivativeMuscSignal); _dealloc(spatialDerivativeDMuscSignal);
    }

    void resetAll() override {
        t_next = 0.0;
        target[0] = 0.0; target[1] = 0.0;
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

        lastAmplitude = 0;
        lastDelay = 0;
        lastFireTime = 0;
        oldrAmplitude = 0;
        oldrDelay = 0;
        oldrFireTime = 0;

        baselineCurvatureScheduler.resetAll();
        undulatoryCurvatureScheduler.resetAll();
        tauTailScheduler.resetAll();
        neuroKinematicScheduler.resetAll();
        FishData::resetAll();
    }
    void computeMidline(const Real time, const Real dt) override;
    Real _width(const Real s, const Real L) override
    {
        const Real sb=.055*length, st=.288*length, wt=.0254*length, wh=.0635*length;
        if(s<0 or s>L) return 0;
        return (s<sb ? wh * std::sqrt(1 - std::pow((sb - s)/sb, 2)) :
                (s<st ? (-2*(wt-wh)-wt*(st-sb))*std::pow((s-sb)/(st-sb), 3)
                        + (3*(wt-wh)+wt*(st-sb))*std::pow((s-sb)/(st-sb), 2)
                        + wh: (wt - wt * std::pow((s-st)/(L-st), 2))));
    }

    /*************************** ACTION REPERTOIRE ***********************************/

    void burst(const Real t_current, const std::vector<double> &a) {
        // Current time must be later than time at which action should be performed.
        assert(t_current >= t_rlAction);

        // Fix the phase of the burst. Normally I should deduce the phase required based on the current
        // curvature configuration.
        const double tailPhase = 0.74;

        // Schedule a burst with given modulation and timing factor
        const double modulationFactor = a[0];
        const double timingFactor = a[1];

        // Curvature of the burst is modulated by the modulation factor. Curvatures are normalized.
        const double curvatureFactor = modulationFactor / this->length;

        // Define the curvature values of the burst
        const std::array<Real ,6> baselineCurvatureValues = {
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)-4.0 * curvatureFactor,
                (Real)-1.0 * curvatureFactor, (Real)-1.0 * curvatureFactor, (Real)0.0 * curvatureFactor
        };
        const std::array<Real ,6> undulatoryCurvatureValues = {
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)-6.0 * curvatureFactor,
                (Real)-3.0  * curvatureFactor, (Real)-1.5 * curvatureFactor, (Real)0.0 * curvatureFactor
        };

        // Use the agent-prescribed timing factor to get the final time of the prescribed action
        const Real actionDuration = (1 - timingFactor) * 0.5 * this->Tperiod/2 + timingFactor * this->Tperiod/2;
        this->t_next = t_current + actionDuration;

        // Decide whether to use the current derivative for the cubic interpolation
        const bool useCurrentDerivative = true;

        // Act by scheduling a transition at the current time.
        baselineCurvatureScheduler.transition(t_current, t_current, this->t_next, baselineCurvatureValues, useCurrentDerivative);
        undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next, undulatoryCurvatureValues, useCurrentDerivative);
        tauTailScheduler.transition(t_current, t_current, this->t_next, tailPhase, useCurrentDerivative);

        printf("Performing a burst with timingFactor %f, and modulationFactor %f\n", timingFactor, modulationFactor);
        printf("t_next is: %f\n", this->t_next);
    }

    void scoot(const Real t_current, const std::vector<double> &a) {
        // Current time must be later than time at which action should be performed.
        assert(t_current >= t_rlAction);

        // Fix the phase of the burst. Normally I should deduce the phase required based on the current
        // curvature configuration.
        const double tailPhase = 0.74;

        // Schedule a burst with given modulation and timing factor
        const double modulationFactor = a[0];
        const double timingFactor = a[1];

        // Curvature of the burst is modulated by the modulation factor. Curvatures are normalized.
        const double curvatureFactor = modulationFactor / this->length;

        // Define the curvature values of the burst
        const std::array<Real ,6> baselineCurvatureValues = {
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor
        };
        const std::array<Real ,6> undulatoryCurvatureValues = {
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)2.57136 * curvatureFactor,
                (Real)3.75425 * curvatureFactor, (Real)5.09147 * curvatureFactor, (Real)0.0 * curvatureFactor
        };

        // Use the agent-prescribed timing factor to get the final time of the prescribed action
        const Real actionDuration = (1 - timingFactor) * 0.5 * this->Tperiod + timingFactor * this->Tperiod;
        this->t_next = t_current + actionDuration;

        // Decide whether to use the current derivative for the cubic interpolation
        const bool useCurrentDerivative = true;

        // Act by scheduling a transition at the current time.
        baselineCurvatureScheduler.transition(t_current, t_current, this->t_next, baselineCurvatureValues, useCurrentDerivative);
        undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next, undulatoryCurvatureValues, useCurrentDerivative);
        tauTailScheduler.transition(t_current, t_current, this->t_next, tailPhase, useCurrentDerivative);

        printf("Performing a scoot with timingFactor %f, and modulationFactor %f\n", timingFactor, modulationFactor);
        printf("t_next is: %f\n", this->t_next);
    }

    void coast(const Real t_current, const std::vector<double> &a) {
    // Current time must be later than time at which action should be performed.
    assert(t_current >= t_rlAction);

    // Fix the phase of the burst. Normally I should deduce the phase required based on the current
    // curvature configuration.
    const double tailPhase = 0.0;

    // Schedule a burst with given modulation and timing factor
    const double modulationFactor = a[0];
    const double timingFactor = a[1];

    // Curvature of the burst is modulated by the modulation factor. Curvatures are normalized.
    const double curvatureFactor = modulationFactor / this->length;

    // Define the curvature values of the burst
    const std::array<Real ,6> baselineCurvatureValues = {
            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor
    };
    const std::array<Real ,6> undulatoryCurvatureValues = {
            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor,
            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor
    };

    // Use the agent-prescribed timing factor to get the final time of the prescribed action
    const Real actionDuration = (1 - timingFactor) * 0.5 * this->Tperiod + timingFactor * this->Tperiod;
    this->t_next = t_current + actionDuration;

    // Decide whether to use the current derivative for the cubic interpolation
    const bool useCurrentDerivative = true;

    // Act by scheduling a transition at the current time.
    baselineCurvatureScheduler.transition(t_current, t_current, this->t_next, baselineCurvatureValues, useCurrentDerivative);
    undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next, undulatoryCurvatureValues, useCurrentDerivative);
    tauTailScheduler.transition(t_current, t_current, this->t_next, tailPhase, useCurrentDerivative);

    printf("Performing a coast with timingFactor %f, and modulationFactor %f\n", timingFactor, modulationFactor);
    printf("t_next is: %f\n", this->t_next);
}

    void hybrid(const Real t_current, const std::vector<double> &a)
    {

        // Store last action into the older action placeholder
        oldrBeta = lastBeta;
        oldrKappa = lastKappa;
        oldrC = lastC;
        oldrTimingFactor = lastTimingFactor;

        // Store the new action into the last action placeholder
        lastBeta = a[0];
        lastKappa = a[1];
        lastC = a[2];
        lastTimingFactor = a[3];

        const double tailPhase = 0.74;

        const double baselineCurvatureFactor = lastC * lastBeta / this->length;
        const double undulatoryCurvatureFactor = 1 / this->length;
        const std::array<Real ,6> baselineCurvatureValues = {
                (Real)0.0 * baselineCurvatureFactor, (Real)0.0 * baselineCurvatureFactor, (Real)-4.0 * baselineCurvatureFactor,
                (Real)-1.0 * baselineCurvatureFactor, (Real)-1.0 * baselineCurvatureFactor, (Real)0.0 * baselineCurvatureFactor
        };
        const std::array<Real ,6> undulatoryCurvatureValuesScoot = {
                (Real)0.0 * undulatoryCurvatureFactor, (Real)0.0 * undulatoryCurvatureFactor, (Real)2.57136 * undulatoryCurvatureFactor,
                (Real)3.75425 * undulatoryCurvatureFactor, (Real)5.09147 * undulatoryCurvatureFactor, (Real)0.0 * undulatoryCurvatureFactor
        };
        const std::array<Real ,6> undulatoryCurvatureValuesBurst = {
                (Real)0.0 * undulatoryCurvatureFactor, (Real)0.0 * undulatoryCurvatureFactor, (Real)-6.0 * undulatoryCurvatureFactor,
                (Real)-3.0  * undulatoryCurvatureFactor, (Real)-1.5 * undulatoryCurvatureFactor, (Real)0.0 * undulatoryCurvatureFactor
        };
        std::array<Real, 6> undulatoryCurvatureValues = {0, 0, 0, 0, 0, 0};
        for (int i=0;i<6;i++)
        {
            undulatoryCurvatureValues[i] = lastC * ((1 - lastKappa) * undulatoryCurvatureValuesScoot[i] + lastKappa * undulatoryCurvatureValuesBurst[i]);
        }

        // Use the agent-prescribed timing factor to get the final time of the prescribed action
        const Real actionDuration = (1 - lastTimingFactor) * 0.5 * this->Tperiod + lastTimingFactor * this->Tperiod;
        this->t_next = t_current + actionDuration;
        const bool useCurrentDerivative = true; // Decide whether to use the current derivative for the cubic interpolation

        // Act by scheduling a transition at the current time.
        baselineCurvatureScheduler.transition(t_current, t_current, this->t_next, baselineCurvatureValues, useCurrentDerivative);
        undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next, undulatoryCurvatureValues, useCurrentDerivative);
        tauTailScheduler.transition(t_current, t_current, this->t_next, tailPhase, useCurrentDerivative);
        printf("Performing a hybrid action with beta %f, kappa %f, c %f\n", lastBeta, lastKappa, lastC);
        printf("t_next is: %f\n", this->t_next);
    }

    void spike(const Real t_current, const std::vector<double> &a) {

        // Store last action into the older action placeholder
        oldrAmplitude = lastAmplitude;
        oldrDelay = lastDelay;
        oldrFireTime = lastFireTime;

        // Store the new action into the last action placeholder
        lastAmplitude = a[0];
        lastDelay = a[1];
        lastFireTime = a[2];

        this->t_next = t_current + lastFireTime;

        neuroKinematicScheduler.Spike(t_current, lastAmplitude, lastDelay, lastFireTime);
        printf("Spiking at %f, with amplitude %f, delay %f, fire-time %f\n", t_current, lastAmplitude, lastDelay, lastFireTime);
        printf("t_next is: %f\n", this->t_next);

    }
};

void NeuroFish::computeMidline(const Real t, const Real dt)
{
    // Curvature control points along midline of fish, as in Gazzola et. al.
    const std::array<Real ,9> curvaturePoints = {(Real)0.0, (Real)0.10, (Real)0.20, (Real)0.30, (Real)0.50,
                                                 (Real)0.60, (Real)0.80, (Real)0.90, (Real)1};

    // Define the compliance function (1/wS)
    Real* compliance_fine  = new Real[Nm];
    const int NCompliancePoints = 9;
    std::array<double, 9> criticalSpinePoints = {0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 1};
    std::array<double, 9> compliancePoints = {0.00, 0.16, 0.46, 0.50, 0.60, 0.75, 0.9, 0.85, 0.60};

    IF2D_Interpolation1D::naturalCubicSpline(criticalSpinePoints.data(), compliancePoints.data(), NCompliancePoints, rS, compliance_fine, Nm);

//    if (t>=0.0 && act1){
//        std::vector<double> a{150, 0.04, 0.147}; // a good starting heuristic is = firing time/10
//        neuroKinematicScheduler.Spike(t, a[0], a[1], a[2]);
//        act1=false;
//    }
//    if (t>=0.01 && act2){
//        std::vector<double> a{150, 0.04, 0.147}; // a good starting heuristic is = firing time/10
//        neuroKinematicScheduler.Spike(t, a[0], a[1], a[2]);
//        act2=false;
//    }

    neuroKinematicScheduler.gimmeValues(t,length, curvaturePoints, Nm, rS, rMuscSignal, vMuscSignal, spatialDerivativeMuscSignal, spatialDerivativeDMuscSignal);

#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {
        rK[i] = rMuscSignal[i] * compliance_fine[i] / length;
        vK[i] = vMuscSignal[i] * compliance_fine[i] / length;
        assert(not std::isnan(rK[i]));
        assert(not std::isinf(rK[i]));
        assert(not std::isnan(vK[i]));
        assert(not std::isinf(vK[i]));
    }
    // solve frenet to compute midline parameters
    IF2D_Frenet2D::solve(Nm, rS, rK,vK, rX,rY, vX,vY, norX,norY, vNorX,vNorY);
}

// Core functions
NeuroKinematicFish::NeuroKinematicFish(SimulationData&s, ArgumentParser&p, double C[2]): Fish(s,p,C)
{
    const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
    myFish = new NeuroFish(length, Tperiod, phaseShift, sim.getH(), ampFac);
    printf("NeuroFish %d %f %f %f\n",myFish->Nm, length, Tperiod, phaseShift);
}

void NeuroKinematicFish::resetAll() {
    NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    cFish->resetAll();
    Fish::resetAll();
}

void NeuroKinematicFish::create(const std::vector<BlockInfo>& vInfo)
{
    NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    Fish::create(vInfo);
}

void NeuroKinematicFish::act(const Real t_rlAction, const std::vector<double>& a) const
{
    NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    cFish->spike(sim.time, a);
}

// Functions for state/reward
std::vector<double> NeuroKinematicFish::state() const
{
    const NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    std::vector<double> S(10,0);
    double com[2] = {0, 0}; this->getCenterOfMass(com);

    S[0] = this->getDistanceFromTarget() / length; // normalized distance from target
    S[1] = (com[0] - cFish->target[0]) / length; // relative x position away from target
    S[2] = (com[1] - cFish->target[1]) / length; // relative y position away from target
    S[3] = getOrientation();
    S[4] = getU() * Tperiod / length;
    S[5] = getV() * Tperiod / length;
    S[6] = getW() * Tperiod;
    S[7] = cFish->lastAmplitude;
    S[8] = cFish->lastDelay;
    S[9] = cFish->lastFireTime;
    return S;
}

// Helper functions
void NeuroKinematicFish::setTarget(double inTarget[2]) const
{
    NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    cFish->target[0] = inTarget[0];
    cFish->target[1] = inTarget[1];
}

void NeuroKinematicFish::getTarget(double outTarget[2]) const
{
    const NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    outTarget[0] = cFish->target[0];
    outTarget[1] = cFish->target[1];
}

double NeuroKinematicFish::getRadialDisplacement() const {
    double com[2] = {0, 0};
    this->getCenterOfMass(com);
    double radialDisplacement = std::sqrt(std::pow((com[0] - this->origC[0]), 2) + std::pow((com[1] - this->origC[1]), 2));
    return radialDisplacement;
}

double NeuroKinematicFish::getDistanceFromTarget() const {
    double com[2] = {0.0, 0.0};
    double target[2] = {0.0, 0.0};
    this->getCenterOfMass(com);
    this->getTarget(target);
    double distanceFromTarget = std::sqrt(std::pow((com[0] - target[0]), 2) + std::pow((com[1] - target[1]), 2));
    return distanceFromTarget;
}

double NeuroKinematicFish::getTimeNextAct() const {
    const NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    return cFish->t_next;
}