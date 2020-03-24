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
    Real * const rBC;
    Real * const vBC;
    Real * const rUC;
    Real * const vUC;
    Real * const rB;
    Real * const vB;
    Real tauTail;

    // Scheduler used to ramp-up the curvature from 0 to C-start configuration determined by B and K in Tprep:
    Schedulers::ParameterSchedulerVector<6> baselineCurvatureScheduler;
    // Scheduler used to ramp-up the curvature from 0 to C-start configuration determined by B and K in Tprep:
    Schedulers::ParameterSchedulerVector<6> undulatoryCurvatureScheduler;
    // Scheduler used to ramp-up the tau_tail parameter from 0 to its designated value at Tprep
    Schedulers::ParameterSchedulerScalar tauTailScheduler;
    // Scheduler used for midline bending control points for RL
    Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;
public:

    ControlledCurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
            : FishData(L, T, phi, _h, _A),   rK(_alloc(Nm)),vK(_alloc(Nm)),
              rBC(_alloc(Nm)),vBC(_alloc(Nm)), rUC(_alloc(Nm)), rB(_alloc(Nm)),
              vB(_alloc(Nm)), vUC(_alloc(Nm)), tauTail(0.0) {
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
        baselineCurvatureScheduler.resetAll();
        undulatoryCurvatureScheduler.resetAll();
        tauTailScheduler.resetAll();
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
    void execute(const Real t_current, const Real t_rlAction, const std::vector<double>&a)
    {
        assert(t_current >= t_rlAction);
        oldrCurv = lastCurv; // store action
        lastCurv = a[0]; // store action

        rlBendingScheduler.Turn(a[0], t_rlAction);
        printf("Turning by %g at time %g with period %g.\n",
               a[0], t_current, t_rlAction);

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

    ~ControlledCurvatureFish() override {
        _dealloc(rK); _dealloc(vK); _dealloc(rBC); _dealloc(vBC);
        _dealloc(rUC); _dealloc(vUC); _dealloc(rB); _dealloc(vB);
    }

    void computeMidline(const Real time, const Real dt) override;
    Real _width(const Real s, const Real L) override
    {
        const Real sb=.04*length, st=.95*length, wt=.01*length, wh=.04*length;
        if(s<0 or s>L) return 0;
        return (s<sb ? std::sqrt(2*wh*s -s*s) :
                (s<st ? wh-(wh-wt)*std::pow((s-sb)/(st-sb),1) : // pow(.,2) is 3D
                 (wt * (L-s)/(L-st))));
    }
};

void ControlledCurvatureFish::computeMidline(const Real t, const Real dt)
{
    // Curvature control points along midline of fish, as in Gazzola et. al.
    const std::array<Real ,6> curvaturePoints = { (Real)0, (Real).2*length,
                                                  (Real).5*length, (Real).75*length, (Real).95*length, length
    };

    const std::array<Real,7> bendPoints = {(Real)-.5, (Real)-.25,
                                           (Real)0,(Real).25, (Real).5, (Real).75, (Real)1};

    // Optimal C-Start parameters identified by Gazzola et. al. (already normalized by length)
    // Need to re-normalize to take into account that we have a different fish length.
//    const double lGazzola = 0.0044;
//    const double curvatureFactor = lGazzola / length;
    const double curvatureFactor = 1.5;
    const std::array<Real ,6> baselineCurvatureValues = {
            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)-3.19 * curvatureFactor,
            (Real)-0.74 * curvatureFactor, (Real)-0.44 * curvatureFactor, (Real)0.0 * curvatureFactor
    };
    const std::array<Real ,6> undulatoryCurvatureValues = {
            (Real)0.0 * curvatureFactor, (Real)0.0 * curvatureFactor, (Real)-5.73 * curvatureFactor,
            (Real)-2.73 * curvatureFactor, (Real)-1.09 * curvatureFactor, (Real)0.0 * curvatureFactor
    };
    const Real phi = 1.11;
    const Real tauTailEnd = 0.74;
//    const Real Tprop = 0.044;
//    const Real Tprop = 0.588235;
    const Real Tprop = Tperiod;
    printf("Tprop is %f\n", Tprop);
    const std::array<Real,6> curvatureZeros = std::array<Real, 6>(); // Initial curvature is zero

    // Ratio of preparatory phase to propulsive phase Tprep/Tprop = 0.7.
    const double prep_prop_ratio = 0.7;
    const double phaseOneDuration = prep_prop_ratio * Tprop;
    const double phaseTwoDuration = Tprop;

    // Ramp up tauTail parameter from zero to designated value at Tprep
    tauTailScheduler.transition(t, 0, phaseOneDuration, tauTailEnd);

    // Phase One
    baselineCurvatureScheduler.transition(t, 0, phaseOneDuration, curvatureZeros, baselineCurvatureValues);
    undulatoryCurvatureScheduler.transition(t, 0, phaseOneDuration, curvatureZeros, undulatoryCurvatureValues);

    // Phase Two
    baselineCurvatureScheduler.transition(t, phaseOneDuration, phaseOneDuration + phaseTwoDuration, baselineCurvatureValues, curvatureZeros);
    undulatoryCurvatureScheduler.transition(t, phaseOneDuration, phaseOneDuration + phaseTwoDuration, undulatoryCurvatureValues, undulatoryCurvatureValues);

    // Phase Three
    baselineCurvatureScheduler.transition(t, phaseOneDuration + phaseTwoDuration, phaseOneDuration + 2 * phaseTwoDuration, curvatureZeros, curvatureZeros);
    undulatoryCurvatureScheduler.transition(t, phaseOneDuration + phaseTwoDuration, phaseOneDuration + 2 * phaseTwoDuration, undulatoryCurvatureValues, undulatoryCurvatureValues);

    // Write values to placeholders
    baselineCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rBC, vBC); // writes to rBC, vBC
    undulatoryCurvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rUC, vUC); // writes to rUC, vUC
    tauTailScheduler.gimmeValues(t, tauTail); // writes to tauTail
    rlBendingScheduler.gimmeValues(t,periodPIDval,length, bendPoints,Nm,rS,rB,vB); // not needed here..

    printf("rBC is %f, vBC is %f\n", rBC[2], vBC[2]);
    printf("rUC is %f, vUC is %f\n", rUC[2], vUC[2]);
    printf("tauTail is %f\n", tauTail);

#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {

        const Real tauS = tauTail * rS[i] / length;
        const Real arg = 2 * M_PI * (t/Tprop - tauS) + phi;
        const Real chainKappa = 2 * M_PI / Tprop;

        rK[i] = rBC[i] + rUC[i] * std::sin(arg);
        vK[i] = rUC[i] * chainKappa * std::cos(arg);

        assert(not std::isnan(rK[i]));
        assert(not std::isinf(rK[i]));
        assert(not std::isnan(vK[i]));
        assert(not std::isinf(vK[i]));
    }

    // solve frenet to compute midline parameters
    IF2D_Frenet2D::solve(Nm, rS, rK,vK, rX,rY, vX,vY, norX,norY, vNorX,vNorY);
#if 0
    {
    FILE * f = fopen("stefan_profile","w");
    for(int i=0;i<Nm;++i)
      fprintf(f,"%d %g %g %g %g %g %g %g %g %g\n",
        i,rS[i],rX[i],rY[i],vX[i],vY[i],
        vNorX[i],vNorY[i],width[i],height[i]);
    fclose(f);
   }
#endif
}

void CStartFish::resetAll() {
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    cFish->resetAll();
    Fish::resetAll();
}

CStartFish::CStartFish(SimulationData&s, ArgumentParser&p, double C[2]):
        Fish(s,p,C), bCorrectTrajectory(p("-pid").asInt(0)),
        bCorrectPosition(p("-pidpos").asInt(0))
{
#if 0
    // parse tau
  tau = parser("-tau").asDouble(1.0);
  // parse curvature controlpoint values
  curvature_values[0] = parser("-k1").asDouble(0.82014);
  curvature_values[1] = parser("-k2").asDouble(1.46515);
  curvature_values[2] = parser("-k3").asDouble(2.57136);
  curvature_values[3] = parser("-k4").asDouble(3.75425);
  curvature_values[4] = parser("-k5").asDouble(5.09147);
  curvature_values[5] = parser("-k6").asDouble(5.70449);
  // if nonzero && Learnfreq<0 your fish is gonna keep turning
  baseline_values[0] = parser("-b1").asDouble(0.0);
  baseline_values[1] = parser("-b2").asDouble(0.0);
  baseline_values[2] = parser("-b3").asDouble(0.0);
  baseline_values[3] = parser("-b4").asDouble(0.0);
  baseline_values[4] = parser("-b5").asDouble(0.0);
  baseline_values[5] = parser("-b6").asDouble(0.0);
  // curvature points are distributed by default but can be overridden
  curvature_points[0] = parser("-pk1").asDouble(0.00)*length;
  curvature_points[1] = parser("-pk2").asDouble(0.15)*length;
  curvature_points[2] = parser("-pk3").asDouble(0.40)*length;
  curvature_points[3] = parser("-pk4").asDouble(0.65)*length;
  curvature_points[4] = parser("-pk5").asDouble(0.90)*length;
  curvature_points[5] = parser("-pk6").asDouble(1.00)*length;
  baseline_points[0] = parser("-pb1").asDouble(curvature_points[0]/length)*length;
  baseline_points[1] = parser("-pb2").asDouble(curvature_points[1]/length)*length;
  baseline_points[2] = parser("-pb3").asDouble(curvature_points[2]/length)*length;
  baseline_points[3] = parser("-pb4").asDouble(curvature_points[3]/length)*length;
  baseline_points[4] = parser("-pb5").asDouble(curvature_points[4]/length)*length;
  baseline_points[5] = parser("-pb6").asDouble(curvature_points[5]/length)*length;
  printf("created IF2D_CStartFish: xpos=%3.3f ypos=%3.3f angle=%3.3f L=%3.3f Tp=%3.3f tau=%3.3f phi=%3.3f\n",position[0],position[1],angle,length,Tperiod,tau,phaseShift);
  printf("curvature points: pk1=%3.3f pk2=%3.3f pk3=%3.3f pk4=%3.3f pk5=%3.3f pk6=%3.3f\n",curvature_points[0],curvature_points[1],curvature_points[2],curvature_points[3],curvature_points[4],curvature_points[5]);
  printf("curvature values (normalized to L=1): k1=%3.3f k2=%3.3f k3=%3.3f k4=%3.3f k5=%3.3f k6=%3.3f\n",curvature_values[0],curvature_values[1],curvature_values[2],curvature_values[3],curvature_values[4],curvature_values[5]);
  printf("baseline points: pb1=%3.3f pb2=%3.3f pb3=%3.3f pb4=%3.3f pb5=%3.3f pb6=%3.3f\n",baseline_points[0],baseline_points[1],baseline_points[2],baseline_points[3],baseline_points[4],baseline_points[5]);
  printf("baseline values (normalized to L=1): b1=%3.3f b2=%3.3f b3=%3.3f b4=%3.3f b5=%3.3f b6=%3.3f\n",baseline_values[0],baseline_values[1],baseline_values[2],baseline_values[3],baseline_values[4],baseline_values[5]);
  // make curvature dimensional for this length
  for(int i=0; i<6; ++i) curvature_values[i]/=length;
#endif

    const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
    myFish = new ControlledCurvatureFish(length, Tperiod, phaseShift, sim.getH(), ampFac);
    printf("ControlledCurvatureFish %d %f %f %f\n",myFish->Nm, length, Tperiod, phaseShift);
}

//static inline Real sgn(const Real val) { return (0 < val) - (val < 0); }
void CStartFish::create(const std::vector<BlockInfo>& vInfo)
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    const double DT = sim.dt/Tperiod, time = sim.time;

    // Control pos diffs
    const double   xDiff = (centerOfMass[0] - origC[0])/length;
    const double   yDiff = (centerOfMass[1] - origC[1])/length;
    const double angDiff =  orientation     - origAng;
    const double relU = (u + sim.uinfx) / length;
    const double relV = (v + sim.uinfy) / length;
    const double angVel = omega, lastAngVel = cFish->lastAvel;
    // compute ang vel at t - 1/2 dt such that we have a better derivative:
    const double aVelMidP = (angVel + lastAngVel)*Tperiod/2;
    const double aVelDiff = (angVel - lastAngVel)*Tperiod/sim.dt;
    cFish->lastAvel = angVel; // store for next time

    // derivatives of following 2 exponential averages:
    const double velDAavg = (angDiff-cFish->avgDangle)/Tperiod + DT * angVel;
    const double velDYavg = (  yDiff-cFish->avgDeltaY)/Tperiod + DT * relV;
    const double velAVavg = 10*((aVelMidP-cFish->avgAngVel)/Tperiod +DT*aVelDiff);
    // exponential averages
    cFish->avgDangle = (1.0 -DT) * cFish->avgDangle +    DT * angDiff;
    cFish->avgDeltaY = (1.0 -DT) * cFish->avgDeltaY +    DT *   yDiff;
    // faster average:
    cFish->avgAngVel = (1-10*DT) * cFish->avgAngVel + 10*DT *aVelMidP;
    const double avgDangle = cFish->avgDangle, avgDeltaY = cFish->avgDeltaY;

    // integral (averaged) and proportional absolute DY and their derivative
    const double absPy = std::fabs(yDiff), absIy = std::fabs(avgDeltaY);
    const double velAbsPy =     yDiff>0 ? relV     : -relV;
    const double velAbsIy = avgDeltaY>0 ? velDYavg : -velDYavg;

    if (bCorrectPosition || bCorrectTrajectory)
        assert(origAng<2e-16 && "TODO: rotate pos and vel to fish POV to enable \
                             PID to work even for non-zero angles");

    if (bCorrectPosition && sim.dt>0)
    {
        //If angle is positive: positive curvature only if Dy<0 (must go up)
        //If angle is negative: negative curvature only if Dy>0 (must go down)
        const double IangPdy = (avgDangle *     yDiff < 0)? avgDangle * absPy : 0;
        const double PangIdy = (angDiff   * avgDeltaY < 0)? angDiff   * absIy : 0;
        const double IangIdy = (avgDangle * avgDeltaY < 0)? avgDangle * absIy : 0;

        // derivatives multiplied by 0 when term is inactive later:
        const double velIangPdy = velAbsPy * avgDangle + absPy * velDAavg;
        const double velPangIdy = velAbsIy * angDiff   + absIy * angVel;
        const double velIangIdy = velAbsIy * avgDangle + absIy * velDAavg;

        //zero also the derivatives when appropriate
        const double coefIangPdy = avgDangle *     yDiff < 0 ? 1 : 0;
        const double coefPangIdy = angDiff   * avgDeltaY < 0 ? 1 : 0;
        const double coefIangIdy = avgDangle * avgDeltaY < 0 ? 1 : 0;

        const double valIangPdy = coefIangPdy *    IangPdy;
        const double difIangPdy = coefIangPdy * velIangPdy;
        const double valPangIdy = coefPangIdy *    PangIdy;
        const double difPangIdy = coefPangIdy * velPangIdy;
        const double valIangIdy = coefIangIdy *    IangIdy;
        const double difIangIdy = coefIangIdy * velIangIdy;
        const double periodFac = 1.0 - xDiff;
        const double periodVel =     - relU;

        if(not sim.muteAll) {
            std::ofstream filePID;
            std::stringstream ssF;
            ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
            filePID.open(ssF.str().c_str(), std::ios::app);
            filePID<<time<<" "<<valIangPdy<<" "<<difIangPdy
                   <<" "<<valPangIdy<<" "<<difPangIdy
                   <<" "<<valIangIdy<<" "<<difIangIdy
                   <<" "<<periodFac <<" "<<periodVel <<"\n";
        }
        const double totalTerm = valIangPdy + valPangIdy + valIangIdy;
        const double totalDiff = difIangPdy + difPangIdy + difIangIdy;
        cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
        cFish->correctTailPeriod(periodFac, periodVel, sim.time, sim.dt);
    }
        // if absIy<EPS then we have just one fish that the simulation box follows
        // therefore we control the average angle but not the Y disp (which is 0)
    else if (bCorrectTrajectory && sim.dt>0)
    {
        const double avgAngVel = cFish->avgAngVel, absAngVel = std::fabs(avgAngVel);
        const double absAvelDiff = avgAngVel>0? velAVavg : -velAVavg;
        const Real coefInst = angDiff*avgAngVel>0 ? 0.01 : 1, coefAvg = 0.1;
        const Real termInst = angDiff*absAngVel;
        const Real diffInst = angDiff*absAvelDiff + angVel*absAngVel;
        const double totalTerm = coefInst*termInst + coefAvg*avgDangle;
        const double totalDiff = coefInst*diffInst + coefAvg*velDAavg;

        if(not sim.muteAll) {
            std::ofstream filePID;
            std::stringstream ssF;
            ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
            filePID.open(ssF.str().c_str(), std::ios::app);
            filePID<<time<<" "<<coefInst*termInst<<" "<<coefInst*diffInst
                   <<" "<<coefAvg*avgDangle<<" "<<coefAvg*velDAavg<<"\n";
        }
        cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
    }

    // to debug and check state function, but requires an other obstacle
    //const int indCurrAct = (time + sim.dt)/(Tperiod/2);
    //if(time < indCurrAct*Tperiod/2) state(sim.shapes[0]);

    Fish::create(vInfo);
}

void CStartFish::act(const Real t_rlAction, const std::vector<double>& a) const
{
    ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    cFish->execute(sim.time, t_rlAction, a);
}

double CStartFish::getLearnTPeriod() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    return cFish->periodPIDval;
}

double CStartFish::getPhase(const double t) const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    const double T0 = cFish->time0;
    const double Ts = cFish->timeshift;
    const double Tp = cFish->periodPIDval;
    const double arg  = 2*M_PI*((t-T0)/Tp +Ts) + M_PI*phaseShift;
    const double phase = std::fmod(arg, 2*M_PI);
    return (phase<0) ? 2*M_PI + phase : phase;
}

std::vector<double> CStartFish::state(Shape*const p) const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<double> S(10,0);
    S[0] = ( center[0] - p->center[0] )/ length;
    S[1] = ( center[1] - p->center[1] )/ length;
    S[2] = getOrientation();
    S[3] = getPhase( sim.time );
    S[4] = getU() * Tperiod / length;
    S[5] = getV() * Tperiod / length;
    S[6] = getW() * Tperiod;
    S[7] = cFish->lastTact;
    S[8] = cFish->lastCurv;
    S[9] = cFish->oldrCurv;

#ifndef STEFANS_SENSORS_STATE
    return S;
#else
    S.resize(16);
    const Real h = sim.getH(), invh = 1/h;
    const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

    // function that finds block id of block containing pos (x,y)
    const auto holdingBlockID = [&](const Real x, const Real y)
    {
      const auto getMin = [&]( const BlockInfo&I )
      {
        std::array<Real,2> MIN = I.pos<Real>(0, 0);
        for(int i=0; i<2; ++i)
          MIN[i] -= 0.5 * h; // pos returns cell centers
        return MIN;
      };

      const auto getMax = [&]( const BlockInfo&I )
      {
        std::array<Real,2> MAX = I.pos<Real>(VectorBlock::sizeX-1,
                                             VectorBlock::sizeY-1);
        for(int i=0; i<2; ++i)
          MAX[i] += 0.5 * h; // pos returns cell centers
        return MAX;
      };

      const auto holdsPoint = [&](const std::array<Real,2> MIN, std::array<Real,2> MAX,
                                  const Real X,const Real Y)
      {
        // this may return true for 2 blocks if (X,Y) overlaps with edges
        return X >= MIN[0] && Y >= MIN[1] && X <= MAX[0] && Y <= MAX[1];
      };

      std::vector<std::pair<double, int>> distsBlocks(velInfo.size());
      for(size_t i=0; i<velInfo.size(); ++i)
      {
        std::array<Real,2> MIN = getMin(velInfo[i]);
        std::array<Real,2> MAX = getMax(velInfo[i]);
        if( holdsPoint(MIN, MAX, x, y) )
        {
        // handler to select obstacle block
          const auto& skinBinfo = velInfo[i];
          const auto *const o = obstacleBlocks[skinBinfo.blockID];
          if(o != nullptr ) return (int) i;
        }
        std::array<Real, 4> WENS;
        WENS[0] = MIN[0] - x;
        WENS[1] = x - MAX[0];
        WENS[2] = MIN[1] - y;
        WENS[3] = y - MAX[1];
        const Real dist = *std::max_element(WENS.begin(),WENS.end());
        distsBlocks[i].first = dist;
        distsBlocks[i].second = i;
      }
      std::sort(distsBlocks.begin(), distsBlocks.end());
      std::reverse(distsBlocks.begin(), distsBlocks.end());
      for( auto distBlock: distsBlocks )
      {
        // handler to select obstacle block
          const auto& skinBinfo = velInfo[distBlock.second];
          const auto *const o = obstacleBlocks[skinBinfo.blockID];
          if(o != nullptr ) return (int) distBlock.second;
      }
      printf("ABORT: coordinate could not be associated to obstacle block\n");
      fflush(0); abort();
      return (int) 0;
    };

    // function that is probably unnecessary, unless pos is at block edge
    // then it makes the op stable without increasing complexity in the above
    const auto safeIdInBlock = [&](const std::array<Real,2> pos,
                                   const std::array<Real,2> org)
    {
      const int indx = (int) std::round((pos[0] - org[0])*invh);
      const int indy = (int) std::round((pos[1] - org[1])*invh);
      const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
      const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
      return std::array<int, 2>{{ix, iy}};
    };

    // return fish velocity at a point on the fish skin:
    const auto skinVel = [&](const std::array<Real,2> pSkin)
    {
      const auto& skinBinfo = velInfo[holdingBlockID(pSkin[0], pSkin[1])];
      const auto *const o = obstacleBlocks[skinBinfo.blockID];
      if (o == nullptr) {
        printf("ABORT: skin point is outside allocated obstacle blocks\n");
        fflush(0); abort();
      }
      const std::array<Real,2> oSkin = skinBinfo.pos<Real>(0, 0);
      const std::array<int,2> iSkin = safeIdInBlock(pSkin, oSkin);
      printf("skin pos:[%f %f] -> block org:[%f %f] ind:[%d %d]\n",
        pSkin[0], pSkin[1], oSkin[0], oSkin[1], iSkin[0], iSkin[1]);
      const Real* const udef = o->udef[iSkin[1]][iSkin[0]];
      const Real uSkin = u - omega * (pSkin[1]-centerOfMass[1]) + udef[0];
      const Real vSkin = v + omega * (pSkin[0]-centerOfMass[0]) + udef[1];
      return std::array<Real, 2>{{uSkin, vSkin}};
    };

    // return flow velocity at point of flow sensor:
    const auto sensVel = [&](const std::array<Real,2> pSens)
    {
      const auto& sensBinfo = velInfo[holdingBlockID(pSens[0], pSens[1])];
      const std::array<Real,2> oSens = sensBinfo.pos<Real>(0, 0);
      const std::array<int,2> iSens = safeIdInBlock(pSens, oSens);
      printf("sensor pos:[%f %f] -> block org:[%f %f] ind:[%d %d]\n",
        pSens[0], pSens[1], oSens[0], oSens[1], iSens[0], iSens[1]);
      const VectorBlock& b = * (const VectorBlock*) sensBinfo.ptrBlock;
      return std::array<Real, 2>{{b(iSens[0], iSens[1]).u[0],
                                  b(iSens[0], iSens[1]).u[1]}};
    };

    // side of the head defined by position sb from function _width above ^^^
    int iHeadSide = 0;
    for(int i=0; i<myFish->Nm-1; ++i)
      if( myFish->rS[i] <= 0.04*length && myFish->rS[i+1] > 0.04*length )
        iHeadSide = i;
    assert(iHeadSide>0);

    std::array<Real,2> tipShear, lowShear, topShear;
    { // surface and sensor (shifted by 2h) points of the fish tip
      const auto &DU = myFish->upperSkin, &DL = myFish->lowerSkin;
      // first point of the two skins is the same
      // normal should be almost the same: take the mean
      const std::array<Real,2> pSkin = {DU.xSurf[0], DU.ySurf[0]};
      const Real normX = (DU.normXSurf[0] + DL.normXSurf[0]) / 2;
      const Real normY = (DU.normYSurf[0] + DL.normYSurf[0]) / 2;
      const std::array<Real,2> pSens = {pSkin[0] + h * normX,
                                        pSkin[1] + h * normY};
      const std::array<Real,2> vSens = sensVel(pSens), vSkin = skinVel(pSkin);
      tipShear[0] = (vSens[0] - vSkin[0]) * invh;
      tipShear[1] = (vSens[1] - vSkin[1]) * invh;
    }

    for(int a = 0; a<2; ++a)
    {
      const auto& D = a==0? myFish->upperSkin : myFish->lowerSkin;
      const std::array<Real,2> pSkin = {D.midX[iHeadSide], D.midY[iHeadSide]};
      const Real normX = D.normXSurf[iHeadSide], normY = D.normYSurf[iHeadSide];
      const std::array<Real,2> pSens = {pSkin[0] + h * normX,
                                        pSkin[1] + h * normY};
      const std::array<Real,2> vSens = sensVel(pSens), vSkin = skinVel(pSkin);
      const Real shearX = (vSens[0] - vSkin[0]) * invh;
      const Real shearY = (vSens[1] - vSkin[1]) * invh;
      // now figure out how to rotate it along the fish skin for consistency:
      const Real dX = D.xSurf[iHeadSide+1] - D.xSurf[iHeadSide];
      const Real dY = D.ySurf[iHeadSide+1] - D.ySurf[iHeadSide];
      const Real proj = dX * normX - dY * normY;
      const Real tangX = proj>0?  normX : -normX; // s.t. tang points from head
      const Real tangY = proj>0? -normY :  normY; // to tail, normal outward
      (a==0? topShear[0] : lowShear[0]) = shearX * normX + shearY * normY;
      (a==0? topShear[1] : lowShear[1]) = shearX * tangX + shearY * tangY;
    }

    S[10] = tipShear[0] * Tperiod / length;
    S[11] = tipShear[1] * Tperiod / length;
    S[12] = lowShear[0] * Tperiod / length;
    S[13] = lowShear[1] * Tperiod / length;
    S[14] = topShear[0] * Tperiod / length;
    S[15] = topShear[1] * Tperiod / length;
    printf("shear tip:[%f %f] lower side:[%f %f] upper side:[%f %f]\n",
      S[10],S[11], S[12],S[13], S[14],S[15]);

    return S;
#endif
}

std::vector<double> CStartFish::getCStartState() const
{
    const ControlledCurvatureFish* const cFish = dynamic_cast<ControlledCurvatureFish*>( myFish );
    std::vector<double> S(10,0);

    double charLength = getCharLength();
    double com[2] = {0, 0};
    getLabPosition(com);
    double radialPos = std::sqrt(std::pow(com[0], 2) + std::pow(com[1], 2));
    double polarAngle = std::atan2(com[1], com[0]);

    // no sensor info needed probably

    S[0] = radialPos / charLength; // distance from center
    S[1] = polarAngle; // polar angle
    S[2] = getOrientation();
    S[3] = getPhase( sim.time );
    S[4] = getU() * Tperiod / length;
    S[5] = getV() * Tperiod / length;
    S[6] = getW() * Tperiod;
    S[7] = cFish->lastTact; //proprioception  - fish knows its current shape
    S[8] = cFish->lastCurv; // could have 3 curvature controls instead of 1 e.g
    S[9] = cFish->oldrCurv;
    return S;
}