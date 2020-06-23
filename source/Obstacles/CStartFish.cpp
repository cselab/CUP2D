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
#include "../Utils/BufferedLogger.h"


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
    // Last phi
    Real lastPhiUndulatory = 0;
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
    // Older phi
    Real oldrPhiUndulatory = 0;

    // Distance travelled at Tprop
    Real dTprop = 0.0;

    // first action
    bool firstAction = true;

    // Time for next action
    double t_next = 0.0;

    // Target location
    double target[2] = {0.0, 0.0};

    // Virtual origin
    double virtualOrigin[2] = {0.5, 0.5};

    // Energy expended
    double energyExpended = 0.0;
    double energyBudget = 0.0;

    // Dump time
    Real nextDump = 0.0;

//    // act bools
    bool act1 = true;
    bool act2 = true;
    bool act3 = true;
    bool act4 = true;
    bool act5 = true;

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
    // Current phi and phi velocity
    Real phiUndulatory;
    Real vPhiUndulatory;
    // Current alpha
    Real alpha;

    // Schedulers
    Schedulers::ParameterSchedulerVector<6> baselineCurvatureScheduler;
    Schedulers::ParameterSchedulerVector<6> undulatoryCurvatureScheduler;
    Schedulers::ParameterSchedulerScalar tauTailScheduler;
    Schedulers::ParameterSchedulerScalar phiScheduler;

public:

    ControlledCurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
            : FishData(L, T, phi, _h, _A), rK(_alloc(Nm)), vK(_alloc(Nm)),
              rBC(_alloc(Nm)),vBC(_alloc(Nm)), rUC(_alloc(Nm)), vUC(_alloc(Nm)),
              tauTail(0.0), vTauTail(0.0), phiUndulatory(0.0), vPhiUndulatory(0.0), alpha(0.0) {
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
        oldrPhiUndulatory = lastPhiUndulatory;

        // Store the new action into the last action placeholder
        lastB3 = a[0];
        lastB4 = a[1];
        lastB5 = a[2];
        lastK3 = a[3];
        lastK4 = a[4];
        lastK5 = a[5];
        lastTau = a[6];
        lastAlpha = a[7];
        lastPhiUndulatory = a[8];

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

        if (firstAction) {
            printf("FIRST ACTION %f\n", lastPhiUndulatory);
            phiScheduler.transition(t_current, t_current, this->t_next, lastPhiUndulatory, lastPhiUndulatory);
            firstAction = false;
        } else {
            printf("Next action %f\n", lastPhiUndulatory);
            phiScheduler.transition(t_current, t_current, this->t_next, lastPhiUndulatory, useCurrentDerivative);
        }

        printf("Action duration is: %f\n", actionDuration);
        printf("Action: {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", lastB3, lastB4, lastB5, lastK3, lastK4, lastK5, lastTau, lastAlpha, lastPhiUndulatory);

        // Save the actions to file
        FILE * f1 = fopen("actions.dat","a+");
        fprintf(f1,"Action: {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", lastB3, lastB4, lastB5, lastK3, lastK4, lastK5, lastTau, lastAlpha, lastPhiUndulatory);
        fclose(f1);

        // Durations
        FILE * f2 = fopen("action_durations.dat","a+");
        fprintf(f2,"Duration: %f\n", actionDuration);
        fclose(f2);


//        printf("Scheduled a transition between %f and %f to tau %f and phi %f\n", t_current, t_next, lastTau, lastPhiUndulatory);
//        printf("Alpha is: %f\n", lastAlpha);
//        printf("Scheduled a transition between %f and %f to baseline curvatures %f, %f, %f\n", t_current, t_next, lastB3, lastB4, lastB5);
//        printf("Scheduled a transition between %f and %f to undulatory curvatures %f, %f, %f\n", t_current, t_next, lastK3, lastK4, lastK5);
//        printf("t_next is: %f/n", this->t_next);
    }

    void scheduleCStart(const Real t_current, const std::vector<double>&a)
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
        const double duration1 = 0.7 * this->Tperiod;
        const double duration2 = this->Tperiod;
        double actionDuration = 0.0;
        if (t_current < duration1){
            actionDuration = duration1;
            this->t_next = t_current + actionDuration;
        } else {
            actionDuration = duration2;
            this->t_next = t_current + actionDuration;
        }

        // Decide whether to use the current derivative for the cubic interpolation
        const bool useCurrentDerivative = true;

        // Act by scheduling a transition at the current time.
        baselineCurvatureScheduler.transition(t_current, t_current, this->t_next, baselineCurvatureValues, useCurrentDerivative);
        undulatoryCurvatureScheduler.transition(t_current, t_current, this->t_next, undulatoryCurvatureValues, useCurrentDerivative);
        tauTailScheduler.transition(t_current, t_current, this->t_next, lastTau, useCurrentDerivative);

        printf("\nAction duration is: %f\n", actionDuration);
        printf("t_next is: %f\n", this->t_next);
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

//    // Parameters of 3D C-start (Gazzola et. al.)
//    if (t>=0.0 && act1){
//        std::vector<double> a{-1.96, -0.46, -0.56, -6.17, -3.71, -1.09, 0.65, 0.4, 0.1321};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.7* this->Tperiod && act2){
//        std::vector<double> a{0, 0, 0, -6.17, -3.71, -1.09, 0.65, 1, 0.1321};
//        this->schedule(t, a);
//        act2=false;
//    }

//    // Reproduces the 2D C-start (Gazzola et. al.)
//    if (t>=0.0 && act1){
//        std::vector<double> a{-3.19, -0.74, -0.44, -5.73, -2.73, -1.09, 0.74, 0.4, 0.176};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.7* this->Tperiod && act2){
//        std::vector<double> a{0, 0, 0, -5.73, -2.73, -1.09, 0.74, 1, 0.176};
//        this->schedule(t, a);
//        act2=false;
//    }

//    // Reproduces the 13(12.7)mJ energy escape
//    if (t>=0.0 && act1){
//        std::vector<double> a{-4.67, -4.09, -2.20, -1.18, -0.95, -1.107, 0.556, 0.1186, 0.565};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.58255* this->Tperiod && act2){
//        std::vector<double> a{-3.41, -3.15, -1.63, -5.45, -3.44, -1.58, 0.776, 0.0847, 0.796};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.58255 + 0.553544) * this->Tperiod && act3){
//        std::vector<double> a{-1.6024, -0.9016, -2.397, -1.356, -1.633, -4.0767, 0.6017, 0.3174, 0.390727};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.58255 + 0.553544 + 0.658692) * this->Tperiod && act4){
//        std::vector<double> a{-1.258, -0.928, -2.5133, -3.56, -2.574, -2.9287, 0.520897, 0.2516, 0.602385};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.58255 + 0.553544 + 0.658692 + 0.6358) * this->Tperiod && act5){
//        std::vector<double> a{-3.04523, -2.983, -2.784, -3.868, -2.648, -2.894, 0.493, 0.3608, 0.481728};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // Reproduces the 7.30mJ energy escape
//    if (t>=0.0 && act1){
//        std::vector<double> a{-4.623, -3.75, -2.034, -1.138, -0.948, -1.374, 0.521658, 0.1651, 0.544885};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.58255* this->Tperiod && act2){
//        std::vector<double> a{-3.082, -3.004, -1.725, -4.696, -2.979, -1.974, 0.23622, 0.1071, 0.756351};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.58255 + 0.553544) * this->Tperiod && act3){
//        std::vector<double> a{-1.6024, -0.9016, -2.397, -1.356, -1.633, -4.0767, 0.6017, 0.3174, 0.390727};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.58255 + 0.553544 + 0.658692) * this->Tperiod && act4){
//        std::vector<double> a{-1.258, -0.928, -2.5133, -3.56, -2.574, -2.9287, 0.520897, 0.2516, 0.602385};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.58255 + 0.553544 + 0.658692 + 0.680396) * this->Tperiod && act5){
//        std::vector<double> a{-3.04523, -2.983, -2.784, -3.868, -2.648, -2.894, 0.493, 0.3608, 0.481728};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // Reproduces the 21.90mJ energy escape
//    if (t>=0.0 && act1){
//        std::vector<double> a{-4.686684, -3.912884, -2.241593, -1.121328, -1.065105, -0.944875, 0.591863, 0.082740, 0.540199};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.318453 && act2){
//        std::vector<double> a{-3.640735, -2.963610, -1.554508, -5.782052, -3.937959, -1.483281, 0.127391, 0.071496, 0.836809};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.318453 + 0.315146) && act3){
//        std::vector<double> a{-1.615101, -0.846640, -2.638458, -1.536289, -1.750644, -3.240543, 0.544093, 0.311599, 0.342352};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.318453 + 0.315146 + 0.385764) && act4){
//        std::vector<double> a{-1.105307, -0.824289, -2.022394, -4.455174, -2.921277, -2.279189, 0.434638, 0.258321, 0.684365};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.318453 + 0.315146 + 0.385764 + 0.370094) && act5){
//        std::vector<double> a{-2.849169, -3.017569, -2.438394, -4.134092, -2.877911, -2.616167, 0.447130, 0.369074, 0.520669};
//        this->schedule(t, a);
//        act5=false;
//    }


    // Write values to placeholders
    baselineCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rBC, vBC); // writes to rBC, vBC
    undulatoryCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rUC, vUC); // writes to rUC, vUC
    tauTailScheduler.gimmeValues(t, tauTail, vTauTail); // writes to tauTail and vTauTail
    phiScheduler.gimmeValues(t, phiUndulatory, vPhiUndulatory);

    const double curvMax = 2*M_PI/length;
#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {

        const Real tauS = tauTail * rS[i] / length;
        const Real vTauS = vTauTail * rS[i] / length;
        const Real arg = 2 * M_PI * (t/Tperiod - tauS) + 2 * M_PI * phiUndulatory;
        const Real vArg = 2 * M_PI / Tperiod - 2 * M_PI * vTauS + 2 * M_PI * vPhiUndulatory;

        const Real curvCmd = rBC[i] + rUC[i] * std::sin(arg);
        const Real curvCmdVel = vBC[i] + rUC[i] * vArg * std::cos(arg) + vUC[i] * std::sin(arg);

        if (std::abs(curvCmd) >= curvMax) {
            rK[i] = curvCmd>0 ? curvMax : -curvMax;
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


//    if (t >= this->nextDump) {
//        // Save the midline curvature values and velocities to recreate spine externally.
//        FILE * f = fopen("midline_coordinates.dat","a+");
//        for(int i=0;i<Nm;++i)
//            fprintf(f,"%d %g %g %g %g %g %g %g %g\n",
//                    i,rS[i],rX[i],rY[i],vX[i],vY[i],
//                    vNorX[i],vNorY[i],width[i]);
//        fclose(f);
//
//        // Save the midpoint curvature to file
//        FILE * f1 = fopen("curvature_values.dat","a+");
//        fprintf(f1,"%f  %g  %d\n", t, rK[Nmid], Nmid);
//        fclose(f1);
//
//        this->nextDump += 0.01; // dump time 0.01
//    }

    // solve Frenet to compute midline parameters
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

void CStartFish::actCStart(const Real lTact, const std::vector<double> &a) const {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->scheduleCStart(sim.time, a);
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

std::vector<double> CStartFish::stateEscapeTradeoff() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<double> S(26,0);

    S[0] = this->getRadialDisplacement() / length; // distance from original position
    S[1] = cFish->dTprop / length;
    S[2] = this->getPolarAngle(); // polar angle from virtual origin
    S[3] = cFish->energyBudget - cFish->energyExpended; // energy expended so far, must be set in RL
    S[4] = getOrientation(); // orientation of fish
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

std::vector<double> CStartFish::stateEscape() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<double> S(25,0);

    S[0] = this->getRadialDisplacement() / length; // distance from original position
    S[1] = this->getPolarAngle(); // polar angle from virtual origin
    S[2] = cFish->energyExpended; // energy expended so far, must be set in RL
    S[3] = getOrientation(); // orientation of fish
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

std::vector<double> CStartFish::stateSequentialEscape() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<double> S(25,0);

    double com[2] = {0.0, 0.0}; this->getCenterOfMass(com);
    bool propulsionForward = com[0] <= this->origC[0];
    double signedRadialDisplacement = 0.0;
    if (propulsionForward) {
        signedRadialDisplacement = this->getRadialDisplacement() / length;
    } else {
        signedRadialDisplacement = -this->getRadialDisplacement() / length;
    }

    S[0] = signedRadialDisplacement; // distance from original position
    S[1] = this->getPolarAngle(); // polar angle from virtual origin
    S[2] = cFish->energyExpended; // energy expended so far, must be set in RL
    S[3] = getOrientation(); // orientation of fish
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

std::vector<double> CStartFish::stateEscapeVariableEnergy() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<double> S(25,0);

    S[0] = this->getRadialDisplacement() / length; // distance from original position
    S[1] = this->getPolarAngle(); // polar angle from virtual origin
    S[2] = cFish->energyBudget - cFish->energyExpended; // energy expended relative to energy budget, must be set in RL
    S[3] = getOrientation(); // orientation of fish
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

std::vector<double> CStartFish::stateCStart() const
{
    std::vector<double> S(2,0);
    S[0] = this->getRadialDisplacement() / length; // distance from center
    S[1] = this->getPolarAngle(); // polar angle
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

void CStartFish::setEnergyExpended(const double energyExpended) {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->energyExpended = energyExpended;
}

void CStartFish::setDistanceTprop(const double distanceTprop) {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    double com[2] = {0.0, 0.0}; this->getCenterOfMass(com);
    bool propulsionForward = com[0] <= this->origC[0];
    if (propulsionForward) {
        cFish->dTprop = distanceTprop;
    } else {
        cFish->dTprop = -distanceTprop;
    }
}

double CStartFish::getDistanceTprop() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->dTprop;
}

double CStartFish::getEnergyExpended() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->energyExpended;
}

double CStartFish::getTimeNextAct() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->t_next;
}

double CStartFish::getPolarAngle() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    double com[2] = {0, 0}; this->getCenterOfMass(com);
    double polarAngle = std::atan2(com[1]- cFish->virtualOrigin[1], com[0]- cFish->virtualOrigin[0]);
    return polarAngle;
}

void CStartFish::setVirtualOrigin(const double vo[2]) {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->virtualOrigin[0] = vo[0];
    cFish->virtualOrigin[1] = vo[1];
}

void CStartFish::setEnergyBudget(const double baselineEnergy) {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->energyBudget = baselineEnergy;
}

double CStartFish::getEnergyBudget() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->energyBudget;
}