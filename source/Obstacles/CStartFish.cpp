//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "CStartFish.h"
#include "FishData.h"
#include "FishUtilities.h"
#include <sstream>
#include "../Utils/BufferedLogger.h"


using namespace cubism;

class ControlledCurvatureFish : public FishData
{
    const Real Tperiod;
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
    Real t_next = 0.0;

    // Target location
    Real target[2] = {0.0, 0.0};

    // Virtual origin
    Real virtualOrigin[2] = {0.5, 0.5};

    // Energy expended
    Real energyExpended = 0.0;
    Real energyBudget = 0.0;

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
            : FishData(L, _h), Tperiod(T), rK(_alloc(Nm)), vK(_alloc(Nm)),
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

    void schedule(const Real t_current, const std::vector<Real>&a)
    {
        // Current time must be later than time at which action should be performed.
        // (PW) commented to resolve compilation error with config=debug
        // assert(t_current >= t_rlAction);

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
        Real curvatureFactor = 1.0 / this->length;

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
            printf("FIRST ACTION %f\n", (double)lastPhiUndulatory);
            phiScheduler.transition(t_current, t_current, this->t_next, lastPhiUndulatory, lastPhiUndulatory);
            firstAction = false;
        } else {
            printf("Next action %f\n", (double)lastPhiUndulatory);
            phiScheduler.transition(t_current, t_current, this->t_next, lastPhiUndulatory, useCurrentDerivative);
        }

//        printf("Action duration is: %f\n", actionDuration);
//        printf("Action: {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", lastB3, lastB4, lastB5, lastK3, lastK4, lastK5, lastTau, lastAlpha, lastPhiUndulatory);
//
//        // Save the actions to file
//        FILE * f1 = fopen("actions.dat","a+");
//        fprintf(f1,"Action: {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", lastB3, lastB4, lastB5, lastK3, lastK4, lastK5, lastTau, lastAlpha, lastPhiUndulatory);
//        fclose(f1);
//
//        // Durations
//        FILE * f2 = fopen("action_durations.dat","a+");
//        fprintf(f2,"Duration: %f\n", actionDuration);
//        fclose(f2);


//        printf("Scheduled a transition between %f and %f to tau %f and phi %f\n", t_current, t_next, lastTau, lastPhiUndulatory);
//        printf("Alpha is: %f\n", lastAlpha);
//        printf("Scheduled a transition between %f and %f to baseline curvatures %f, %f, %f\n", t_current, t_next, lastB3, lastB4, lastB5);
//        printf("Scheduled a transition between %f and %f to undulatory curvatures %f, %f, %f\n", t_current, t_next, lastK3, lastK4, lastK5);
//        printf("t_next is: %f/n", this->t_next);
    }

    void scheduleCStart(const Real t_current, const std::vector<Real>&a)
    {
        // Current time must be later than time at which action should be performed.
        // (PW) commented to resolve compilation error with config=debug
        // assert(t_current >= t_rlAction);

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
        Real curvatureFactor = 1.0 / this->length;

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
        const Real duration1 = 0.7 * this->Tperiod;
        const Real duration2 = this->Tperiod;
        Real actionDuration = 0.0;
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

        printf("\nAction duration is: %f\n", (double)actionDuration);
        printf("t_next is: %f\n", (double)this->t_next);
        printf("Scheduled a transition between %f and %f to baseline curvatures %f, %f, %f\n", (double)t_current, (double)t_next, (double)lastB3, (double)lastB4, (double)lastB5);
        printf("Scheduled a transition between %f and %f to undulatory curvatures %f, %f, %f\n", (double)t_current, (double)t_next, (double)lastK3, (double)lastK4, (double)lastK5);
        printf("Scheduled a transition between %f and %f to tau %f\n", (double)t_current, (double)t_next, (double)lastTau);
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

    //***************************************************************************************************************//

//    // 2.43 (2.43) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.394378, -2.583142, -1.678619, -0.955955, -1.064813, -1.925944, 0.443542, 0.232105, 0.489527};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.362384 && act2){
//        std::vector<Real> a{-2.485280, -2.338090, -2.242998, -3.490950, -2.925883, -3.385765, 0.375721, 0.164584, 0.662246};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=( 0.362384+0.342525 ) && act3){
//        std::vector<Real> a{-1.556780, -1.454230, -3.299322, -0.947089, -1.671784, -4.202114, 0.568120, 0.291511, 0.467864};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=( 0.362384+0.342525  +0.379856 ) && act4){
//        std::vector<Real> a{-1.768546, -1.909876, -3.160618, -3.372401, -3.015401, -3.804584, 0.429210, 0.210001, 0.546459};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=( 0.362384+0.342525  +0.379856  +0.355883 ) && act5){
//        std::vector<Real> a{-2.432865, -2.854739, -3.499387, -2.548434, -2.434035, -3.313599, 0.505750, 0.344945, 0.467138};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // 4.60 (4.46) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.501705, -3.187968, -1.854891, -1.047240, -1.004203, -1.631981, 0.481264, 0.196833, 0.517757};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.352010 && act2){
//        std::vector<Real> a{-2.756316, -2.709547, -1.935662, -4.053762, -2.921679, -2.575511, 0.289008, 0.129697, 0.726307};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=( 0.352010+0.332264 ) && act3){
//        std::vector<Real> a{-1.638049, -1.108426, -2.624121, -1.167477, -1.611290, -4.138656, 0.586663, 0.296912, 0.443818};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=( 0.352010+0.332264  +0.381445 ) && act4){
//        std::vector<Real> a{-1.377286, -1.182907, -2.838169, -3.562582, -2.742909, -3.281147, 0.485458, 0.269470, 0.582267};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=( 0.352010+0.332264  +0.381445  +0.373374 ) && act5){
//        std::vector<Real> a{-2.900758, -2.963751, -3.068763, -3.405851, -2.479450, -3.010067, 0.508307, 0.347674, 0.452785};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // 6.76 (6.52) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.585289, -3.687300, -2.015811, -1.126492, -0.961738, -1.414174, 0.512743, 0.169850, 0.541472};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.344073 && act2){
//        std::vector<Real> a{-3.003033, -2.964232, -1.752716, -4.584373, -2.963824, -2.046727, 0.243175, 0.110393, 0.753060};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=( 0.344073+0.326586 ) && act3){
//        std::vector<Real> a{-1.597985, -0.925536, -2.412679, -1.314756, -1.622079, -4.078839, 0.601689, 0.315822, 0.399234};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=( 0.344073+0.326586  +0.387006 ) && act4){
//        std::vector<Real> a{-1.290073, -0.978738, -2.566674, -3.555598, -2.595279, -2.967997, 0.521687, 0.276598, 0.597619};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=( 0.344073+0.326586  +0.387006  +0.375470 ) && act5){
//        std::vector<Real> a{-3.073839, -3.003205, -2.826751, -3.870952, -2.643668, -2.919011, 0.496321, 0.363866, 0.477231};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // 8.92 (8.63) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.631036, -3.965183, -2.125765, -1.172344, -0.942524, -1.273069, 0.532867, 0.149158, 0.556593};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.337988 && act2){
//        std::vector<Real> a{-3.196996, -3.078392, -1.673658, -4.997335, -3.095449, -1.776184, 0.213846, 0.097741, 0.769822};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=( 0.337988+ 0.322865) && act3){
//        std::vector<Real> a{-1.607076, -0.850258, -2.341231, -1.436818, -1.631797, -3.979139, 0.601293, 0.320120, 0.374928};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=( 0.337988+0.322865  + 0.388271) && act4){
//        std::vector<Real> a{-1.223661, -0.879590, -2.385597, -3.710325, -2.581477, -2.784000, 0.515169, 0.261528, 0.618091};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=( 0.337988+0.322865  + 0.388271 +0.371038 ) && act5){
//        std::vector<Real> a{-3.027528, -2.960710, -2.698337, -3.973508, -2.710399, -2.852488, 0.482810, 0.358837, 0.491252};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // 11.08 (10.86) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.654859, -4.076202, -2.180449, -1.187806, -0.939436, -1.174666, 0.545970, 0.131844, 0.563827};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.332895 && act2){
//        std::vector<Real> a{-3.316472, -3.124565, -1.651551, -5.262369, -3.253724, -1.647494, 0.192940, 0.090957, 0.783653};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=( 0.332895+0.320870 ) && act3){
//        std::vector<Real> a{-1.626424, -0.827613, -2.316539, -1.546340, -1.647769, -3.846736, 0.597161, 0.316748, 0.365782};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=( 0.332895+0.320870  +0.387279 ) && act4){
//        std::vector<Real> a{-1.162392, -0.819409, -2.285829, -3.925021, -2.639759, -2.648756, 0.496513, 0.246885, 0.641182};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.332895 +0.320870  +0.387279  +0.366731 ) && act5){
//        std::vector<Real> a{-2.947716, -2.930931, -2.639388, -3.999925, -2.733923, -2.798701, 0.472590, 0.349387, 0.498743};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // 13.25 (12.89) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.669092, -4.091029, -2.199815, -1.180366, -0.946546, -1.099139, 0.557086, 0.117021, 0.564826};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.328536 && act2){
//        std::vector<Real> a{-3.400822, -3.138930, -1.636164, -5.450703, -3.434716, -1.576357, 0.174830, 0.085556, 0.796697};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=( 0.328536+ 0.319281) && act3){
//        std::vector<Real> a{-1.634065, -0.824726, -2.326143, -1.616123, -1.677073, -3.704563, 0.589777, 0.313482, 0.360904};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.328536+ 0.319281 + 0.386318) && act4){
//        std::vector<Real> a{-1.121264, -0.791190, -2.207700, -4.057422, -2.697201, -2.545713, 0.481230, 0.240465, 0.658457};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=( 0.328536+ 0.319281 + 0.386318 + 0.364843) && act5){
//        std::vector<Real> a{-2.895961, -2.930388, -2.581198, -4.021605, -2.761964, -2.743145, 0.463994, 0.345237, 0.507538};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // 15.41 (15.10) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.678779, -4.057009, -2.205809, -1.160044, -0.962561, -1.041059, 0.568015, 0.104891, 0.561295};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>= 0.324968 && act2){
//        std::vector<Real> a{-3.483177, -3.133075, -1.617716, -5.589856, -3.619439, -1.533727, 0.159126, 0.080504, 0.809013};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.324968 + 0.317795) && act3){
//        std::vector<Real> a{-1.628999, -0.827154, -2.362526, -1.647450, -1.713048, -3.572785, 0.580268, 0.312041, 0.356712};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=( 0.324968+ 0.317795 + 0.385894) && act4){
//        std::vector<Real> a{-1.101130, -0.788941, -2.138788, -4.122668, -2.731504, -2.464712, 0.471397, 0.244024, 0.669718};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=( 0.324968+ 0.317795 + 0.385894 + 0.365889) && act5){
//        std::vector<Real> a{-2.873229, -2.956661, -2.518764, -4.062538, -2.801658, -2.686893, 0.458071, 0.346180, 0.514884};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // 17.57 (17.04) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.685327, -4.005475, -2.212091, -1.138507, -0.988073, -0.997685, 0.578228, 0.095351, 0.555037};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.322162 && act2){
//        std::vector<Real> a{-3.581836, -3.104472, -1.592701, -5.698769, -3.811102, -1.501724, 0.145612, 0.074569, 0.820629};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.322162 + 0.316050) && act3){
//        std::vector<Real> a{-1.624217, -0.838124, -2.453317, -1.617455, -1.742226, -3.457168, 0.566668, 0.313464, 0.351841};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.322162+ 0.316050 + 0.386313) && act4){
//        std::vector<Real> a{-1.102140, -0.801205, -2.078403, -4.132382, -2.744817, -2.417795, 0.465530, 0.254771, 0.673557};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.322162 + 0.316050 + 0.386313 + 0.369050) && act5){
//        std::vector<Real> a{-2.875590, -3.001369, -2.466997, -4.079204, -2.833935, -2.637640, 0.454279, 0.353261, 0.521504};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // 19.74 (19.16) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.688191, -3.954764, -2.223889, -1.124408, -1.022923, -0.966451, 0.586453, 0.088088, 0.547614};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.320026 && act2){
//        std::vector<Real> a{-3.626116, -3.048085, -1.575569, -5.751697, -3.895130, -1.490546, 0.135556, 0.072658, 0.829648};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.320026 + 0.315488) && act3){
//        std::vector<Real> a{-1.615091, -0.841790, -2.542467, -1.584894, -1.749461, -3.345980, 0.556286, 0.313125, 0.346116};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.320026 + 0.315488 + 0.386213) && act4){
//        std::vector<Real> a{-1.099233, -0.811258, -2.045102, -4.300000, -2.824348, -2.344939, 0.450437, 0.257749, 0.679261};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.320026 + 0.315488 + 0.386213 + 0.369926 ) && act5){
//        std::vector<Real> a{-2.852772, -3.007171, -2.454606, -4.112217, -2.857376, -2.629590, 0.450890, 0.363210, 0.520481};
//        this->schedule(t, a);
//        act5=false;
//    }
//
//    // 21.9 (21.31) mJ escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.686684, -3.912884, -2.241593, -1.121328, -1.065105, -0.944875, 0.591863, 0.082740, 0.540199};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>= 0.318453 && act2){
//        std::vector<Real> a{-3.640735, -2.963610, -1.554508, -5.782052, -3.937959, -1.483281, 0.127391, 0.071496, 0.836809};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=( 0.318453 + 0.315146) && act3){
//        std::vector<Real> a{-1.615101, -0.846640, -2.638458, -1.536289, -1.750644, -3.240543, 0.544093, 0.311599, 0.342352};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=( 0.318453 + 0.315146 + 0.385764) && act4){
//        std::vector<Real> a{-1.105307, -0.824289, -2.022394, -4.455174, -2.921277, -2.279189, 0.434638, 0.258321, 0.684365};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=( 0.318453 + 0.315146 + 0.385764 + 0.370094) && act5){
//        std::vector<Real> a{-2.849169, -3.017569, -2.438394, -4.134092, -2.877911, -2.616167, 0.447130, 0.369074, 0.520669};
//        this->schedule(t, a);
//        act5=false;
//    }

    //***************************************************************************************************************//

//    // Parameters of 3D C-start (Gazzola et. al.)
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-1.96, -0.46, -0.56, -6.17, -3.71, -1.09, 0.65, 0.4, 0.1321};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.7* this->Tperiod && act2){
//        std::vector<Real> a{0, 0, 0, -6.17, -3.71, -1.09, 0.65, 1, 0.1321};
//        this->schedule(t, a);
//        act2=false;
//    }

//    // Reproduces the 2D C-start (Gazzola et. al.)
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-3.19, -0.74, -0.44, -5.73, -2.73, -1.09, 0.74, 0.4, 0.176};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.7* this->Tperiod && act2){
//        std::vector<Real> a{0, 0, 0, -5.73, -2.73, -1.09, 0.74, 1, 0.176};
//        this->schedule(t, a);
//        act2=false;
//    }

//    // Reproduces the 13(12.7)mJ energy escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.67, -4.09, -2.20, -1.18, -0.95, -1.107, 0.556, 0.1186, 0.565};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.58255* this->Tperiod && act2){
//        std::vector<Real> a{-3.41, -3.15, -1.63, -5.45, -3.44, -1.58, 0.776, 0.0847, 0.796};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.58255 + 0.553544) * this->Tperiod && act3){
//        std::vector<Real> a{-1.6024, -0.9016, -2.397, -1.356, -1.633, -4.0767, 0.6017, 0.3174, 0.390727};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.58255 + 0.553544 + 0.658692) * this->Tperiod && act4){
//        std::vector<Real> a{-1.258, -0.928, -2.5133, -3.56, -2.574, -2.9287, 0.520897, 0.2516, 0.602385};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.58255 + 0.553544 + 0.658692 + 0.6358) * this->Tperiod && act5){
//        std::vector<Real> a{-3.04523, -2.983, -2.784, -3.868, -2.648, -2.894, 0.493, 0.3608, 0.481728};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // Reproduces the 7.30mJ energy escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.623, -3.75, -2.034, -1.138, -0.948, -1.374, 0.521658, 0.1651, 0.544885};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.58255* this->Tperiod && act2){
//        std::vector<Real> a{-3.082, -3.004, -1.725, -4.696, -2.979, -1.974, 0.23622, 0.1071, 0.756351};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.58255 + 0.553544) * this->Tperiod && act3){
//        std::vector<Real> a{-1.6024, -0.9016, -2.397, -1.356, -1.633, -4.0767, 0.6017, 0.3174, 0.390727};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.58255 + 0.553544 + 0.658692) * this->Tperiod && act4){
//        std::vector<Real> a{-1.258, -0.928, -2.5133, -3.56, -2.574, -2.9287, 0.520897, 0.2516, 0.602385};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.58255 + 0.553544 + 0.658692 + 0.680396) * this->Tperiod && act5){
//        std::vector<Real> a{-3.04523, -2.983, -2.784, -3.868, -2.648, -2.894, 0.493, 0.3608, 0.481728};
//        this->schedule(t, a);
//        act5=false;
//    }

//    // Reproduces the 21.90mJ energy escape
//    if (t>=0.0 && act1){
//        std::vector<Real> a{-4.686684, -3.912884, -2.241593, -1.121328, -1.065105, -0.944875, 0.591863, 0.082740, 0.540199};
//        this->schedule(t, a);
//        act1=false;
//    }
//    if (t>=0.318453 && act2){
//        std::vector<Real> a{-3.640735, -2.963610, -1.554508, -5.782052, -3.937959, -1.483281, 0.127391, 0.071496, 0.836809};
//        this->schedule(t, a);
//        act2=false;
//    }
//    if (t>=(0.318453 + 0.315146) && act3){
//        std::vector<Real> a{-1.615101, -0.846640, -2.638458, -1.536289, -1.750644, -3.240543, 0.544093, 0.311599, 0.342352};
//        this->schedule(t, a);
//        act3=false;
//    }
//    if (t>=(0.318453 + 0.315146 + 0.385764) && act4){
//        std::vector<Real> a{-1.105307, -0.824289, -2.022394, -4.455174, -2.921277, -2.279189, 0.434638, 0.258321, 0.684365};
//        this->schedule(t, a);
//        act4=false;
//    }
//    if (t>=(0.318453 + 0.315146 + 0.385764 + 0.370094) && act5){
//        std::vector<Real> a{-2.849169, -3.017569, -2.438394, -4.134092, -2.877911, -2.616167, 0.447130, 0.369074, 0.520669};
//        this->schedule(t, a);
//        act5=false;
//    }


    // Write values to placeholders
    baselineCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rBC, vBC); // writes to rBC, vBC
    undulatoryCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rUC, vUC); // writes to rUC, vUC
    tauTailScheduler.gimmeValues(t, tauTail, vTauTail); // writes to tauTail and vTauTail
    phiScheduler.gimmeValues(t, phiUndulatory, vPhiUndulatory);

    const Real curvMax = 2*M_PI/length;
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

    // Save the midpoint curvature to file
    FILE * f1 = fopen("curvature_values.dat","a+");
    fprintf(f1,"%f  %g  %d\n", (double)t, (double)rK[Nmid], Nmid);
    fclose(f1);

//    this->nextDump += 0.01; // dump time 0.01

//    if (t >= this->nextDump) {
//        // Save the midline curvature values and velocities to recreate spine externally.
//        FILE * f = fopen("midline_coordinates.dat","a+");
//        for(int i=0;i<Nm;++i)
//            fprintf(f,"%d %g %g %g %g %g %g %g %g\n",
//                    i,rS[i],rX[i],rY[i],vX[i],vY[i],
//                    vNorX[i],vNorY[i],width[i]);
//        fclose(f);
//
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

CStartFish::CStartFish(SimulationData&s, ArgumentParser&p, Real C[2]):
        Fish(s,p,C)
{
    const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
    myFish = new ControlledCurvatureFish(length, Tperiod, phaseShift, sim.minH, ampFac);
    if( sim.rank == 0 && s.verbose ) printf("[CUP2D] - ControlledCurvatureFish %d %f %f %f\n",myFish->Nm, (double)length, (double)Tperiod, (double)phaseShift);
}

void CStartFish::create(const std::vector<BlockInfo>& vInfo)
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    Fish::create(vInfo);
}

void CStartFish::act(const Real t_rlAction, const std::vector<Real>& a) const
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->schedule(sim.time, a);
}

void CStartFish::actCStart(const Real lTact, const std::vector<Real> &a) const {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->scheduleCStart(sim.time, a);
}

void CStartFish::setTarget(Real desiredTarget[2]) const
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->target[0] = desiredTarget[0];
    cFish->target[1] = desiredTarget[1];
}

void CStartFish::getTarget(Real outTarget[2]) const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    outTarget[0] = cFish->target[0];
    outTarget[1] = cFish->target[1];
}

std::vector<Real> CStartFish::stateEscapeTradeoff() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<Real> S(26,0);

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

std::vector<Real> CStartFish::stateEscape() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<Real> S(25,0);

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

std::vector<Real> CStartFish::stateSequentialEscape() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<Real> S(25,0);

    Real com[2] = {0.0, 0.0}; this->getCenterOfMass(com);
    bool propulsionForward = com[0] <= this->origC[0];
    Real signedRadialDisplacement = 0.0;
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

std::vector<Real> CStartFish::stateEscapeVariableEnergy() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<Real> S(25,0);

    S[0] = this->getRadialDisplacement() / length; // distance from original position
    S[1] = this->getPolarAngle(); // polar angle from virtual origin
    S[2] = (cFish->energyBudget - cFish->energyExpended); // energy expended relative to energy budget, must be set in RL
//    S[2] = (cFish->energyBudget - cFish->energyExpended)/0.0073; // energy expended relative to energy budget, must be set in RL
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

std::vector<Real> CStartFish::stateCStart() const
{
    std::vector<Real> S(2,0);
    S[0] = this->getRadialDisplacement() / length; // distance from center
    S[1] = this->getPolarAngle(); // polar angle
    return S;
}

std::vector<Real> CStartFish::stateTarget() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    Real com[2] = {0, 0}; this->getCenterOfMass(com);

    std::vector<Real> S(15,0);

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

Real CStartFish::getRadialDisplacement() const {
    Real com[2] = {0, 0};
    this->getCenterOfMass(com);
    Real radialDisplacement = std::sqrt(std::pow((com[0] - this->origC[0]), 2) + std::pow((com[1] - this->origC[1]), 2));
    return radialDisplacement;
}

Real CStartFish::getDistanceFromTarget() const {
    Real com[2] = {0.0, 0.0};
    Real target[2] = {0.0, 0.0};
    this->getCenterOfMass(com);
    this->getTarget(target);
    Real distanceFromTarget = std::sqrt(std::pow((com[0] - target[0]), 2) + std::pow((com[1] - target[1]), 2));
    return distanceFromTarget;
}

void CStartFish::setEnergyExpended(const Real energyExpended) {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->energyExpended = energyExpended;
}

void CStartFish::setDistanceTprop(const Real distanceTprop) {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    Real com[2] = {0.0, 0.0}; this->getCenterOfMass(com);
    bool propulsionForward = com[0] <= this->origC[0];
    if (propulsionForward) {
        cFish->dTprop = distanceTprop;
    } else {
        cFish->dTprop = -distanceTprop;
    }
}

Real CStartFish::getDistanceTprop() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->dTprop;
}

Real CStartFish::getEnergyExpended() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->energyExpended;
}

Real CStartFish::getTimeNextAct() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->t_next;
}

Real CStartFish::getPolarAngle() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    Real com[2] = {0, 0}; this->getCenterOfMass(com);
    Real polarAngle = std::atan2(com[1]- cFish->virtualOrigin[1], com[0]- cFish->virtualOrigin[0]);
    return polarAngle;
}

void CStartFish::setVirtualOrigin(const Real vo[2]) {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->virtualOrigin[0] = vo[0];
    cFish->virtualOrigin[1] = vo[1];
}

void CStartFish::setEnergyBudget(const Real baselineEnergy) {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->energyBudget = baselineEnergy;
}

Real CStartFish::getEnergyBudget() const {
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->energyBudget;
}
