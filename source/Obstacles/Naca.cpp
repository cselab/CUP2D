//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
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
  NacaData(Real L, Real _h, Real _tRatio)
  : FishData(L, _h), tRatio(_tRatio) { _computeWidth(); }

  void computeMidline(const Real time, const Real dt) override;
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
    // std::cout << "s=" << s << ", w=" << w << std::endl;
    assert( w >= 0 );
    return w;
  }
};

void NacaData::computeMidline(const Real t, const Real dt)
{
  // TODO: For NACA the midline is static
  rX[0] = rY[0] = vX[0] = vY[0] = norX[0] = vNorX[0] = vNorY[0] = 0.0;
  vNorY[0] = 1.0;

  #pragma omp parallel for schedule(static)
  for(int i=1; i<Nm; ++i) {
    // only x-coordinate of midline varies
    const Real dx = std::fabs(rS[i]-rS[i-1]);
    rX[i] = dx;
    // rest 0
    rY[i] = vX[i] = vY[i] = norX[i] = vNorX[i] = vNorY[i] = 0.0;
    // thus normal is
    norY[i] = 1.0;
  }

  for(int i=1; i<Nm; ++i){
    rX[i] += rX[i-1];
  }
}

Naca::Naca(SimulationData&s, ArgumentParser&p, Real C[2])
  : Fish(s,p,C), Apitch( p("-Apitch").asDouble(0.0)*M_PI/180 ), Fpitch(p("-Fpitch").asDouble(0.0)), tAccel(p("-tAccel").asDouble(-1)), fixedCenterDist(p("-fixedCenterDist").asDouble(0))  {
  const Real tRatio = p("-tRatio").asDouble(0.12);
  myFish = new NacaData(length, sim.minH, tRatio);
  if( sim.rank == 0 && s.verbose  ) printf("[CUP2D] - NacaData Nm=%d L=%f t=%f A=%f w=%f xvel=%f yvel=%f tAccel=%f fixedCenterDist=%f\n",myFish->Nm, (double)length, (double)tRatio, (double)Apitch, (double)Fpitch, (double)forcedu, (double)forcedv, (double)tAccel, (double)fixedCenterDist);
}

void Naca::updateVelocity(Real dt)
{
  const Real omegaAngle = 2*M_PI*Fpitch;
  const Real angle = Apitch*std::sin(omegaAngle*sim.time);
  // angular velocity
  omega = Apitch*omegaAngle*std::cos(omegaAngle*sim.time);
  if( sim.time < tAccel )
  {
    // linear velocity (due to rotation-axis != CoM)
    u = (sim.time/tAccel)*forcedu - fixedCenterDist*length*omega*std::sin(angle);
    v = (sim.time/tAccel)*forcedv + fixedCenterDist*length*omega*std::cos(angle);
  }
  else
  {
    // linear velocity (due to rotation-axis != CoM)
    u = forcedu - fixedCenterDist*length*omega*std::sin(angle);
    v = forcedv + fixedCenterDist*length*omega*std::cos(angle);
  }
  // std::cout << "u=" << u << ", v=" << v << "\n";
}

void Naca::updateLabVelocity( int nSum[2], Real uSum[2] )
{
  if(bFixedx){
   (nSum[0])++; 
   if( sim.time < tAccel )
    uSum[0] -= (sim.time/tAccel)*forcedu;
   else 
    uSum[0] -= forcedu; 
  }
  if(bFixedy){
   (nSum[1])++; 
   if( sim.time < tAccel )
    uSum[1] -= (sim.time/tAccel)*forcedv;
   else 
    uSum[1] -= forcedv; 
  }
}
