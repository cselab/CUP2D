//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Naca.h"
#include "FishData.h"
#include "FishUtilities.h"

using namespace cubism;

class NacaData : public FishData
{
  const Real tRatio;

 public:
  NacaData(Real L, Real _h, Real _tRatio) : FishData(L, _h), tRatio(_tRatio)
  { 
    _computeWidth();
  }

  void computeMidline(const Real time, const Real dt) override
  {
    rX[0] = rY[0] = vX[0] = vY[0] = norX[0] = vNorX[0] = vNorY[0] = 0.0;
    norY[0] = 1.0;
  
    #pragma omp parallel for schedule(static)
    for(int i=1; i<Nm; ++i)
    {
      // only x-coordinate of midline varies, the rest is 0
      const Real dx = std::fabs(rS[i]-rS[i-1]);
      rX[i] = dx;
      rY[i] = vX[i] = vY[i] = norX[i] = vNorX[i] = vNorY[i] = 0.0;
      norY[i] = 1.0;
    }
  
    for(int i=1; i<Nm; ++i)
      rX[i] += rX[i-1];
  }

  Real _width(const Real s, const Real L) override
  {
    const Real a =  0.2969;
    const Real b = -0.1260;
    const Real c = -0.3516;
    const Real d =  0.2843;
    const Real e = -0.1015; // -0.1036 instead of -0.1015 to ensure closing end
    const Real t = tRatio*L; //NACA00{tRatio}
    if(s<0 or s>L) return 0;
    const Real p = s/L;
    const Real w = 5*t*(a*std::sqrt(p) + b*p + c*p*p + d*p*p*p + e*p*p*p*p);
    assert( w >= 0 );
    return w;
  }
};

Naca::Naca(SimulationData&s, ArgumentParser&p, Real C[2]): Fish(s,p,C)
{
  Apitch = p("-Apitch").asDouble(0.0)*M_PI/180; //aplitude of sinusoidal pitch angle
  Fpitch = p("-Fpitch").asDouble(0.0)         ; //frequency
  Mpitch = p("-Mpitch").asDouble(0.0)*M_PI/180; //mean angle
  Fheave = p("-Fheave").asDouble(0.0)         ; //frequency of rowing motion
  Aheave = p("-Aheave").asDouble(0.0)*length  ; //amplitude (NON DIMENSIONAL)
  tAccel = p("-tAccel").asDouble(-1);
  fixedCenterDist = p("-fixedCenterDist").asDouble(0);
  const Real thickness = p("-tRatio").asDouble(0.12);
  myFish = new NacaData(length, sim.minH, thickness);
  if( sim.rank == 0 && sim.verbose) printf("[CUP2D] - NacaData Nm=%d L=%f t=%f A=%f w=%f xvel=%f yvel=%f tAccel=%f fixedCenterDist=%f\n",myFish->Nm, (double)length, (double)thickness, (double)Apitch, (double)Fpitch, (double)forcedu, (double)forcedv, (double)tAccel, (double)fixedCenterDist);
}

void Naca::updateVelocity(Real dt)
{
  const Real omegaAngle = 2*M_PI*Fpitch;
  const Real angle = Mpitch + Apitch*std::sin(omegaAngle*sim.time);
  // angular velocity
  omega = Apitch*omegaAngle*std::cos(omegaAngle*sim.time);

  // heaving motion
  const Real v_heave = -2.0*M_PI*Fheave*Aheave*std::sin(2*M_PI*Fheave*sim.time);
  if( sim.time < tAccel )
  {
    // linear velocity (due to rotation-axis != CoM)
    u = (1.0-sim.time/tAccel)*0.01*forcedu + (sim.time/tAccel)*forcedu - fixedCenterDist*length*omega*std::sin(angle);
    v = (1.0-sim.time/tAccel)*0.01*forcedv + (sim.time/tAccel)*forcedv + fixedCenterDist*length*omega*std::cos(angle) + v_heave;
  }
  else
  {
    // linear velocity (due to rotation-axis != CoM)
    u = forcedu - fixedCenterDist*length*omega*std::sin(angle);
    v = forcedv + fixedCenterDist*length*omega*std::cos(angle) + v_heave;
  }
}

void Naca::updateLabVelocity( int nSum[2], Real uSum[2] )
{
  // heaving motion
  const Real v_heave = -2.0*M_PI*Fheave*Aheave*std::sin(2*M_PI*Fheave*sim.time);

  if(bFixedx)
  {
   (nSum[0])++; 
   if( sim.time < tAccel ) uSum[0] -= (1.0-sim.time/tAccel)*0.01*forcedu + (sim.time/tAccel)*forcedu;
   else                    uSum[0] -=                   forcedu; 
  }
  if(bFixedy)
  {
   (nSum[1])++; 
   if( sim.time < tAccel )uSum[1] -= (1.0-sim.time/tAccel)*0.01*forcedv + (sim.time/tAccel)*forcedv + v_heave;
   else                   uSum[1] -=                   forcedv + v_heave; 
  }
}