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
    // Preparation boolean
    bool prep = true;
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

    // Prep Prop ratio
    Real prepPropRatio;

    // Alpha and Beta
    Real alpha;
    Real beta;
    Real vAlpha;
    Real vBeta;
    bool act = true;


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
    // Current bending curvature and bending curvature velocity
    Real * const rb;
    Real * const vb;
    // Current tail phase and phase velocity
    Real tauTail;
    Real vTauTail;

    // Schedulers
    Schedulers::ParameterSchedulerVector<6> baselineCurvatureScheduler;
    Schedulers::ParameterSchedulerVector<6> undulatoryCurvatureScheduler;
    Schedulers::ParameterSchedulerScalar tauTailScheduler;
    Schedulers::ParameterSchedulerScalar alphaScheduler;
    Schedulers::ParameterSchedulerScalar betaScheduler;
    Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;


public:

    ControlledCurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
            : FishData(L, T, phi, _h, _A), rBC(_alloc(Nm)),vBC(_alloc(Nm)),
              rUC(_alloc(Nm)), vUC(_alloc(Nm)), tauTail(0.0), vTauTail(0.0),
              rK(_alloc(Nm)), vK(_alloc(Nm)), rb(_alloc(Nm)), vb(_alloc(Nm)),
              prepPropRatio(0.0), alpha(0.0),
              beta(0.0), vAlpha(0.0), vBeta(0.0) {
        _computeWidth();
        writeMidline2File(0, "initialCheck");
    }

    void resetAll() override {
        prep = true;
        prepPropRatio = 0.0;
        alpha = 0.0;
        beta = 0.0;
        vAlpha = 0.0;
        vBeta = 0.0;
        lastB3 = 0;
        lastB4 = 0;
        lastB5 = 0;
        lastK3 = 0;
        lastK4 = 0;
        lastK5 = 0;
        lastTau = 0;
        oldrB3 = 0;
        oldrB4 = 0;
        oldrB5 = 0;
        oldrK3 = 0;
        oldrK4 = 0;
        oldrK5 = 0;
        oldrTau = 0;
        act = true;

        baselineCurvatureScheduler.resetAll();
        undulatoryCurvatureScheduler.resetAll();
        tauTailScheduler.resetAll();
        alphaScheduler.resetAll();
        betaScheduler.resetAll();
        rlBendingScheduler.resetAll();
        FishData::resetAll();
    }

    void execute(const Real t_current, const Real t_rlAction, const std::vector<double>&a)
    {
        assert(t_current >= t_rlAction);

        rlBendingScheduler.Turn(a[0], t_rlAction);
        printf("Turning by %g at time %g with period %g.\n",
               a[0], t_current, t_rlAction);

    }

    void schedule(const Real t_current, const Real t_rlAction, const std::vector<double>&a)
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

        // Store the new action into the last action placeholder
        lastB3 = a[0];
        lastB4 = a[1];
        lastB5 = a[2];
        lastK3 = a[3];
        lastK4 = a[4];
        lastK5 = a[5];
        lastTau = a[6];

        if (t_current > 0.7 * this->Tperiod){
            this->prep = false;
        }

        // RL agent should output normalized curvature values as actions.
        double curvatureFactor = 1.0/this->length;
        const std::array<Real ,6> baselineCurvatureValues = {
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)a[0] * curvatureFactor,
                (Real)a[1] * curvatureFactor, (Real)a[2] * curvatureFactor, (Real)0.0 * curvatureFactor
        };
        const std::array<Real ,6> undulatoryCurvatureValues = {
                (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)a[3] * curvatureFactor,
                (Real)a[4] * curvatureFactor, (Real)a[5] * curvatureFactor, (Real)0.0 * curvatureFactor
        };
        const std::array<Real,6> curvatureZeros = std::array<Real, 6>();

        const bool useCurrentDerivative = false;
        baselineCurvatureScheduler.transition(t_current, 0, 0.7 * this->Tperiod, baselineCurvatureValues, useCurrentDerivative);
        undulatoryCurvatureScheduler.transition(t_current, 0, 0.7 * this->Tperiod, undulatoryCurvatureValues, useCurrentDerivative);
        tauTailScheduler.transition(t_current, 0, 0.7 * this->Tperiod, a[6], useCurrentDerivative);

        baselineCurvatureScheduler.transition(t_current, 0.7 * this->Tperiod, 1.7 * this->Tperiod, curvatureZeros, useCurrentDerivative);
    }

//    void scheduleModulate(const Real t_current, const Real t_rlAction, const std::vector<double>&a)
//    {
//        // Current time must be later than time at which action should be performed.
//        assert(t_current >= t_rlAction);
//        alphaScheduler.transition(t_current, t_rlAction, this->actionDuration, a[0]);
//        betaScheduler.transition(t_current, t_rlAction, this->actionDuration, a[1]);
//    }

    ~ControlledCurvatureFish() override {
        _dealloc(rBC); _dealloc(vBC); _dealloc(rUC); _dealloc(vUC);
        _dealloc(rK); _dealloc(vK); _dealloc(rb); _dealloc(vb);
    }

    void computeMidline(const Real time, const Real dt) override;
//    void computeMidlineModulated(const Real time, const Real dt);
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

//void ControlledCurvatureFish::computeMidlineModulated(const Real t, const Real dt)
//{
//    // Curvature control points along midline of fish, as in Gazzola et. al.
//    const std::array<Real ,6> curvaturePoints = { (Real)0, (Real).2*length,
//                                                  (Real).5*length, (Real).75*length, (Real).95*length, length
//    };
//
//    const std::array<Real,7> bendPoints = {(Real)-.5, (Real)-.25,
//                                           (Real)0,(Real).25, (Real).5, (Real).75, (Real)1};
//
//    // Optimal C-Start parameters identified by Gazzola et. al. (normalized)
//    const double curvatureFactor = 1.0 / length;
//    const std::array<Real ,6> baselineCurvatureValues = {
//            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)-3.19 * curvatureFactor,
//            (Real)-0.74 * curvatureFactor, (Real)-0.44 * curvatureFactor, (Real)0.0 * curvatureFactor
//    };
//    const std::array<Real ,6> undulatoryCurvatureValues = {
//            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)-5.73 * curvatureFactor,
//            (Real)-2.73 * curvatureFactor, (Real)-1.09 * curvatureFactor, (Real)0.0 * curvatureFactor
//    };
//    const Real phi = 1.11;
//    const Real tauTailEnd = 0.74;
//    const Real Tprop = this->Tperiod;
//    const std::array<Real,6> curvatureZeros = std::array<Real, 6>(); // Initial curvature is zero
//
//    // These parameters are always the same, only modulate them with a factor
//    tauTailScheduler.transition(0.0, 0.0, 0.0, tauTailEnd, tauTailEnd);
//    baselineCurvatureScheduler.transition(0.0, 0.0, 0.0, baselineCurvatureValues, baselineCurvatureValues);
//    undulatoryCurvatureScheduler.transition(0.0, 0.0, 0.0, undulatoryCurvatureValues, undulatoryCurvatureValues);
//
//    // Write values to placeholders
//    baselineCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rBC, vBC); // writes to rBC, vBC
//    undulatoryCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rUC, vUC); // writes to rUC, vUC
//    tauTailScheduler.gimmeValues(t, tauTail, vTauTail); // writes to tauTail and vTauTail
//    alphaScheduler.gimmeValuesLinear(t, alpha, vAlpha);
//    betaScheduler.gimmeValuesLinear(t, beta, vBeta);
//
//#pragma omp parallel for schedule(static)
//        for(int i=0; i<Nm; ++i) {
//
//        const Real tauS = tauTail * rS[i] / length;
//        const Real vTauS = vTauTail * rS[i] / length;
//        const Real arg = 2 * M_PI * (t/Tprop - tauS) + phi;
//        const Real vArg = 2 * M_PI / Tprop - 2 * M_PI * vTauS;
//
//        rK[i] = alpha * rBC[i] + beta * rUC[i] * std::sin(arg);
//        vK[i] = alpha * vBC[i] + vAlpha * rBC + beta * rUC[i] * vArg * std::cos(arg) + beta * vUC[i] * std::sin(arg)
//                + vBeta * rUC[i] * std::sin(arg);
//
//        assert(not std::isnan(rK[i]));
//        assert(not std::isinf(rK[i]));
//        assert(not std::isnan(vK[i]));
//        assert(not std::isinf(vK[i]));
//    }
//
//    // solve frenet to compute midline parameters
//    IF2D_Frenet2D::solve(Nm, rS, rK,vK, rX,rY, vX,vY, norX,norY, vNorX,vNorY);
//}


void ControlledCurvatureFish::computeMidline(const Real t, const Real dt)
{
    // Curvature control points along midline of fish, as in Gazzola et. al.
    const std::array<Real ,6> curvaturePoints = { (Real)0, (Real).2*length,
                                                  (Real).5*length, (Real).75*length, (Real).95*length, length
    };
//    const std::array<Real,6> bendPoints = {(Real)-0.75, (Real)-0.5, (Real)-0.25,
//                                           (Real)0.0, (Real).25, (Real)0.5};

//    const std::array<Real, 2> bendPoints = {(Real)-1.0*length, (Real)0.0};

//    const double curvatureFactor = 1.0 / length;
//    const std::array<Real ,6> baselineCurvatureValues = {
//            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)-3.19 * curvatureFactor,
//            (Real)-0.74 * curvatureFactor, (Real)-0.44 * curvatureFactor, (Real)0.0 * curvatureFactor
//    };

    const Real phi = 1.11;

//    const std::vector<double> b = {-5.0/length};
//    const Real t_rlAction = 0.1;
//    Real time0 = 0.0;
//
//    if (t >= t_rlAction && act){
//        rlBendingScheduler.Turn(b[0], t_rlAction);
//        time0 = t_rlAction;
//        act = false;
//    }

    // Write values to placeholders
    baselineCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rBC, vBC); // writes to rBC, vBC
    undulatoryCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rUC, vUC); // writes to rUC, vUC
    tauTailScheduler.gimmeValues(t, tauTail, vTauTail); // writes to tauTail and vTauTail
//    rlBendingScheduler.gimmeValues(t,this->Tperiod,this->length, bendPoints,Nm,rS,rb,vb);

#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {

        const Real tauS = tauTail * rS[i] / length;
        const Real vTauS = vTauTail * rS[i] / length;
        const Real arg = 2 * M_PI * (t/Tperiod - tauS) + phi;
        const Real vArg = 2 * M_PI / Tperiod - 2 * M_PI * vTauS;

        rK[i] = rBC[i] + rUC[i] * std::sin(arg);
        vK[i] = vBC[i] + rUC[i] * vArg * std::cos(arg) + vUC[i] * std::sin(arg);

//        rK[i] = rb[i] + rUC[i] * std::sin(arg);
//        vK[i] = vb[i] + rUC[i] * vArg * std::cos(arg) + vUC[i] * std::sin(arg);

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
    const double DT = sim.dt/Tperiod, time = sim.time;

//    std::vector<double> actions{0.5};
//    this->actSimple(time, actions);

//    const int nActions = this->Tperiod / this->getActionDuration();
//    double tAction = 0.0;
//    for (int i=0;i<nActions;i++){
//        std::vector<double> actions;
//        actions.push_back(-0.319);
//        actions.push_back(-0.074);
//        actions.push_back(-0.044);
//        actions.push_back(-0.573);
//        actions.push_back(-0.273);
//        actions.push_back(-0.109);
//        actions.push_back(0.074);
//        cFish->schedule(time + tAction, tAction, actions);
//        tAction += this->getActionDuration();
//    }

    Fish::create(vInfo);
}

void CStartFish::act(const Real t_rlAction, const std::vector<double>& a) const
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->schedule(sim.time, t_rlAction, a);
}

//void CStartFish::actSimple(const Real t_rlAction, const std::vector<double>& a) const
//{
//    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
//    cFish->prepPropRatio = a[0];
//}

//void CStartFish::actModulate(const Real t_rlAction, const std::vector<double>& a) const
//{
//    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
//    cFish->scheduleModulate(sim.time, a);
//}

void CStartFish::actTurn(const Real t_rlAction, const std::vector<double>& a) const
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->execute(sim.time, t_rlAction, a);
}

double CStartFish::getPrep() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->prep;
}

std::vector<double> CStartFish::state() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<double> S(13,0);

    double length = this->length;
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
    return S;
}

double CStartFish::getRadialDisplacement() const {
    double com[2] = {0, 0};
    this->getCenterOfMass(com);
    double radialDisplacement = std::sqrt(std::pow((com[0] - this->origC[0]), 2) + std::pow((com[1] - this->origC[1]), 2));
    printf("Radial displacement is: %f\n", radialDisplacement);
    return radialDisplacement;
}