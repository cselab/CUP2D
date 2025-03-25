//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "NeuroKinematicFish.h"
#include "FishData.h"
#include "FishUtilities.h"
#include <sstream>

using namespace cubism;

class NeuroFish : public FishData
{
    const Real Tperiod;
public:
    Real t_next = 0.0; // Time for next action
    Real target[2] = {0.0, 0.0}; // Target location
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
    Schedulers::ParameterSchedulerNeuroKinematicObject<20> neuroKinematicScheduler;

public:
    NeuroFish(Real L, Real T, Real phi, Real _h, Real _A)
            : FishData(L, _h), Tperiod(T), rK(_alloc(Nm)), vK(_alloc(Nm)),
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

    void burst(const Real t_current, const std::vector<Real> &a) {
        // Current time must be later than time at which action should be performed.
        // (PW) commented to resolve compilation error with config=debug
        // assert(t_current >= t_rlAction);

        // Fix the phase of the burst. Normally I should deduce the phase required based on the current
        // curvature configuration.
        const Real tailPhase = 0.74;

        // Schedule a burst with given modulation and timing factor
        const Real modulationFactor = a[0];
        const Real timingFactor = a[1];

        // Curvature of the burst is modulated by the modulation factor. Curvatures are normalized.
        const Real curvatureFactor = modulationFactor / this->length;

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

        printf("Performing a burst with timingFactor %f, and modulationFactor %f\n", (double)timingFactor, (double)modulationFactor);
        printf("t_next is: %f\n", (double)this->t_next);
    }

    void scoot(const Real t_current, const std::vector<Real> &a) {
        // Current time must be later than time at which action should be performed.
        // (PW) commented to resolve compilation error with config=debug
        // assert(t_current >= t_rlAction);

        // Fix the phase of the burst. Normally I should deduce the phase required based on the current
        // curvature configuration.
        const Real tailPhase = 0.74;

        // Schedule a burst with given modulation and timing factor
        const Real modulationFactor = a[0];
        const Real timingFactor = a[1];

        // Curvature of the burst is modulated by the modulation factor. Curvatures are normalized.
        const Real curvatureFactor = modulationFactor / this->length;

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

        printf("Performing a scoot with timingFactor %f, and modulationFactor %f\n", (double)timingFactor, (double)modulationFactor);
        printf("t_next is: %f\n", (double)this->t_next);
    }

    void coast(const Real t_current, const std::vector<Real> &a) {
    // Current time must be later than time at which action should be performed.
    // (PW) commented to resolve compilation error with config=debug
    // assert(t_current >= t_rlAction);

    // Fix the phase of the burst. Normally I should deduce the phase required based on the current
    // curvature configuration.
    const Real tailPhase = 0.0;

    // Schedule a burst with given modulation and timing factor
    const Real modulationFactor = a[0];
    const Real timingFactor = a[1];

    // Curvature of the burst is modulated by the modulation factor. Curvatures are normalized.
    const Real curvatureFactor = modulationFactor / this->length;

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

    printf("Performing a coast with timingFactor %f, and modulationFactor %f\n", (double)timingFactor, (double)modulationFactor);
    printf("t_next is: %f\n", (double)this->t_next);
}

    void hybrid(const Real t_current, const std::vector<Real> &a)
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

        const Real tailPhase = 0.74;

        const Real baselineCurvatureFactor = lastC * lastBeta / this->length;
        const Real undulatoryCurvatureFactor = 1 / this->length;
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
        printf("Performing a hybrid action with beta %f, kappa %f, c %f\n", (double)lastBeta, (double)lastKappa, (double)lastC);
        printf("t_next is: %f\n", (double)this->t_next);
    }

    void spike(const Real t_current, const std::vector<Real> &a) {

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
        printf("Spiking at %f, with amplitude %f, delay %f, fire-time %f\n", (double)t_current, (double)lastAmplitude, (double)lastDelay, (double)lastFireTime);
        printf("t_next is: %f\n", (double)this->t_next);

    }
};

void NeuroFish::computeMidline(const Real t, const Real dt)
{
//    // Curvature control points along midline of fish, as in Gazzola et. al.
//    const std::array<Real ,9> curvaturePoints = {(Real)0.0, (Real)0.10, (Real)0.20, (Real)0.30, (Real)0.50,
//                                                 (Real)0.60, (Real)0.80, (Real)0.90, (Real)1};

    const std::array<Real ,20> curvaturePoints =
            {(Real)0.0, (Real)0.05263158, (Real)0.10526316, (Real)0.15789474, (Real) 0.21052632, (Real)0.26315789, (Real)0.31578947, (Real)0.36842105, (Real)0.42105263,
             (Real)0.47368421, (Real)0.52631579, (Real)0.57894737, (Real)0.63157895, (Real)0.68421053, (Real)0.73684211, (Real)0.78947368, (Real)0.84210526, (Real)0.89473684,
             (Real)0.94736842, (Real)0.1};

    // Define the compliance function (1/wS)
    Real* compliance_fine  = new Real[Nm];
    const int NCompliancePoints = 10;
//    std::array<Real, 9> criticalSpinePoints = {0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 1};
    std::array<Real, 10> criticalSpinePoints = {0.0, 0.11111111, 0.22222222, 0.33333333, 0.44444444, 0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.0};
    std::array<Real, 10> compliancePoints = {0.00, 0.04, 0.2, 0.4, 0.60, 0.78, 0.9, 0.85, 0.60, 0.20};
//    std::array<Real, 9> compliancePoints = {0.00, 0.16, 0.46, 0.50, 0.60, 0.75, 0.9, 0.85, 0.60};

    IF2D_Interpolation1D::naturalCubicSpline(criticalSpinePoints.data(), compliancePoints.data(), NCompliancePoints, rS, compliance_fine, Nm);

    if (t>=0.0 && act1){
        printf("\n\n\n first action \n\n\n");
        std::vector<Real> a{500, 0.01569, 0.013}; // a good starting heuristic is = firing time/10
        neuroKinematicScheduler.Spike(t, a[0], a[1], a[2]);
        act1=false;
    }
    if (t>=0.3138 && act2){
        printf("\n\n\n second action \n\n\n");
        std::vector<Real> a{-500, 0.01569, 0.013}; // a good starting heuristic is = firing time/10
        neuroKinematicScheduler.Spike(t, a[0], a[1], a[2]);
        act2=false;
    }
    if (t>=0.6276 && act3){
        std::vector<Real> a{300, 0.0157, 0.013}; // a good starting heuristic is = firing time/10
        neuroKinematicScheduler.Spike(t, a[0], a[1], a[2]);
        act3=false;
    }
    if (t>=0.9414 && act4){
        std::vector<Real> a{-300, 0.0157, 0.147}; // a good starting heuristic is = firing time/10
        neuroKinematicScheduler.Spike(t, a[0], a[1], a[2]);
        act4=false;
    }

    neuroKinematicScheduler.gimmeValues(t,length, curvaturePoints, Nm, rS, rMuscSignal, vMuscSignal);

    const Real curvMax = 2*M_PI/length;
#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {
        const Real curvCmd = rMuscSignal[i] * compliance_fine[i] / length;
        const Real curvCmdVel = vMuscSignal[i] * compliance_fine[i] / length;

//        printf("[node %d] curvature %f", i, curvCmd);

        if (curvCmd >= curvMax) {
            rK[i] = curvMax;
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
    // solve frenet to compute midline parameters
    IF2D_Frenet2D::solve(Nm, rS, rK,vK, rX,rY, vX,vY, norX,norY, vNorX,vNorY);
}

// Core functions
NeuroKinematicFish::NeuroKinematicFish(SimulationData&s, ArgumentParser&p, Real C[2]): Fish(s,p,C)
{
    const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
    myFish = new NeuroFish(length, Tperiod, phaseShift, sim.minH, ampFac);
    if( s.verbose ) printf("[CUP2D] - NeuroFish %d %f %f %f\n",myFish->Nm, (double)length, (double)Tperiod, (double)phaseShift);
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

void NeuroKinematicFish::act(const Real t_rlAction, const std::vector<Real>& a) const
{
    NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    cFish->spike(sim.time, a);
}

// Functions for state/reward
std::vector<Real> NeuroKinematicFish::state() const
{
    const NeuroFish* const nFish = dynamic_cast<NeuroFish*>( myFish );
    std::vector<Real> S(12,0);

    S[0] = this->getRadialDisplacement() / length; // distance from center
    S[1] = this->getPolarAngle(); // polar angle
    S[2] = getOrientation();
    S[3] = getU() * Tperiod / length;
    S[4] = getV() * Tperiod / length;
    S[5] = getW() * Tperiod;
    S[6] = nFish->lastAmplitude;
    S[7] = nFish->lastDelay;
    S[8] = nFish->lastFireTime;
    S[9] = nFish->oldrAmplitude;
    S[10] = nFish->oldrDelay;
    S[11] = nFish->oldrFireTime;
    return S;
}

// Helper functions
void NeuroKinematicFish::setTarget(Real inTarget[2]) const
{
    NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    cFish->target[0] = inTarget[0];
    cFish->target[1] = inTarget[1];
}

void NeuroKinematicFish::getTarget(Real outTarget[2]) const
{
    const NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    outTarget[0] = cFish->target[0];
    outTarget[1] = cFish->target[1];
}

Real NeuroKinematicFish::getRadialDisplacement() const {
    Real com[2] = {0, 0};
    this->getCenterOfMass(com);
    Real radialDisplacement = std::sqrt(std::pow((com[0] - this->origC[0]), 2) + std::pow((com[1] - this->origC[1]), 2));
    return radialDisplacement;
}

Real NeuroKinematicFish::getPolarAngle() const {
    Real com[2] = {0, 0}; this->getCenterOfMass(com);
    Real polarAngle = std::atan2(com[1], com[0]);
    return polarAngle;
}

Real NeuroKinematicFish::getDistanceFromTarget() const {
    Real com[2] = {0.0, 0.0};
    Real target[2] = {0.0, 0.0};
    this->getCenterOfMass(com);
    this->getTarget(target);
    Real distanceFromTarget = std::sqrt(std::pow((com[0] - target[0]), 2) + std::pow((com[1] - target[1]), 2));
    return distanceFromTarget;
}

Real NeuroKinematicFish::getTimeNextAct() const {
    const NeuroFish* const cFish = dynamic_cast<NeuroFish*>( myFish );
    return cFish->t_next;
}
