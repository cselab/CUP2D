//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Teardrop.h"
#include "FishData.h"
#include "FishUtilities.h"

using namespace cubism;

class TeardropData : public FishData
{
  const Real tRatio;
 public:
  TeardropData(Real L, Real _h, Real _tRatio)
  : FishData(L, _h), tRatio(_tRatio) { _computeWidth(); }

  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override
  {
    // Compute radius of half-circle using given t-ratio
    const Real R = .5*tRatio*L;

    // safety
    if(s<0 or s>L) return 0;

    // compute width
    Real w;
    if( s == 0 )
      w = 0;
    else if(s<=R)
      w = std::sqrt( R*R - (s-R)*(s-R) );
    else
      w = R - R * ( s - R ) / ( L - R );
    // std::cout << "s=" << s << ", w=" << w << std::endl;
    assert( w >= 0 );
    return w;
  }
};

void TeardropData::computeMidline(const Real t, const Real dt)
{
  // TODO: For Teardrop the midline is static
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

  for(int i=1; i<Nm; ++i)
    rX[i] += rX[i-1];
}

Teardrop::Teardrop(SimulationData&s, ArgumentParser&p, Real C[2])
  : Fish(s,p,C), Apitch( p("-Apitch").asDouble(0.0)*M_PI/180 ), Fpitch(p("-Fpitch").asDouble(0.0)), tAccel(p("-tAccel").asDouble(-1)), fixedCenterDist(p("-fixedCenterDist").asDouble(0))  {
  const Real tRatio = p("-tRatio").asDouble(0.12);
  myFish = new TeardropData(length, sim.minH, tRatio);
  if( s.verbose ) printf("[CUP2D] - TeardropData Nm=%d L=%f t=%f A=%f w=%f xvel=%f yvel=%f tAccel=%f fixedCenterDist=%f\n",myFish->Nm, (double)length, (double)tRatio, (double)Apitch, (double)Fpitch, (double)forcedu, (double)forcedv, (double)tAccel, (double)fixedCenterDist);
}

void Teardrop::updateVelocity(Real dt)
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

void Teardrop::updateLabVelocity( int nSum[2], Real uSum[2] )
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
