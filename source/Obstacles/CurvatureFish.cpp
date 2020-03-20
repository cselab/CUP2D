//
// Created by Ioannis Mandralis on 20.03.20.
//

#include "CurvatureFish.h"

using namespace cubism;

void CurvatureFish::computeMidline(const Real t, const Real dt)
{
    const std::array<Real ,6> curvaturePoints = { (Real)0, (Real).15*length,
                                                  (Real).4*length, (Real).65*length, (Real).9*length, length
    };
    const std::array<Real ,6> curvatureValues = {
            (Real)0.82014/length, (Real)1.46515/length, (Real)2.57136/length,
            (Real)3.75425/length, (Real)5.09147/length, (Real)5.70449/length
    };
    const std::array<Real,7> bendPoints = {(Real)-.5, (Real)-.25, (Real)0,
                                           (Real).25, (Real).5, (Real).75, (Real)1};

#if 1 // ramp-up over Tperiod
    const std::array<Real,6> curvatureZeros = std::array<Real, 6>();
    curvatureScheduler.transition(0,0,Tperiod,curvatureZeros ,curvatureValues);
#else // no rampup for debug
    curvatureScheduler.transition(t,0,Tperiod,curvatureValues,curvatureValues);
#endif

    curvatureScheduler.gimmeValues(t,                curvaturePoints,Nm,rS,rC,vC);
    rlBendingScheduler.gimmeValues(t,periodPIDval,length, bendPoints,Nm,rS,rB,vB);

    // next term takes into account the derivative of periodPIDval in darg:
    const Real diffT = TperiodPID? 1 - (t-time0)*periodPIDdif/periodPIDval : 1;
    // time derivative of arg:
    const Real darg = 2*M_PI/periodPIDval * diffT;
    const Real arg0 = 2*M_PI*((t-time0)/periodPIDval +timeshift) +M_PI*phaseShift;

#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {
        const Real arg = arg0 - 2*M_PI*rS[i]/length/waveLength;
        rK[i] = amplitudeFactor* rC[i]*(std::sin(arg)     + rB[i] +curv_PID_fac);
        vK[i] = amplitudeFactor*(vC[i]*(std::sin(arg)     + rB[i] +curv_PID_fac)
                                 +rC[i]*(std::cos(arg)*darg+ vB[i] +curv_PID_dif));
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