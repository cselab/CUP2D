//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "../Shape.h"

class activeParticle : public Shape
{
public:
  const double radius;
  Real tStartAccelTransfer;
  Real tStartElliTransfer;

  const double xCenterRotation;
  const double yCenterRotation;
  const double x0;
  const double y0;

  double finalRadiusRotation;
  double finalAngRotation;
  
  const double forcedAccelCirc;

  double forcedOmegaCirc;
  double forcedRadiusMotion;
  double true_anomaly;
  double omegaCirc = forcedOmegaCirc;
  double accCirc = forcedAccelCirc;
  double corrector = 0;

  std::vector<double> lastPos{x0, y0};
  std::vector<double> lastVel{0.0, 0.0};

  bool lastUCM = false;
  bool lastUACM = false;
  bool lastEM = false;

  const Real tAccel;

  double initialRadiusRotation = std::sqrt(std::pow(lastPos[0] - xCenterRotation, 2) + std::pow(lastPos[1] - yCenterRotation, 2));
  double initialAngRotation = forcedOmegaCirc;
  double semimajor_axis = (initialRadiusRotation + finalRadiusRotation)/2;
  double semiminor_axis = std::sqrt(initialRadiusRotation*finalRadiusRotation);
  double eccentricity = std::sqrt(1-std::pow(semiminor_axis/semimajor_axis, 2));
  double semilatus_rectum = semimajor_axis*(1-std::pow(eccentricity, 2));
  double mu = std::pow(forcedOmegaCirc*initialRadiusRotation, 2)/((2/initialRadiusRotation)-(1/semimajor_axis));
  double angMom = std::sqrt(semilatus_rectum*mu);

  Real tTransitElli = M_PI*std::sqrt(std::pow(semimajor_axis, 3)/mu);
  Real tTransitAccel = std::abs(finalAngRotation - initialAngRotation)/forcedAccelCirc;

  activeParticle(SimulationData& s,  cubism::ArgumentParser& p, double C[2] ) :
    Shape(s,p,C), radius( p("-radius").asDouble(0.1) ),
  tStartAccelTransfer( p("-tStartAccelTransfer").asDouble(-1) ),
  tStartElliTransfer( p("-tStartElliTransfer").asDouble(-1) ),
  xCenterRotation( p("-xCenterRotation").asDouble(-1) ), yCenterRotation( p("-yCenterRotation").asDouble(-1) ),
  x0( p("-xpos").asDouble(.5*sim.extents[0])), y0( p("-ypos").asDouble(.5*sim.extents[1])),
  finalRadiusRotation( p("-finalRadiusRotation").asDouble(-1)),
  finalAngRotation( p("-finalAngRotation").asDouble(0)),
  forcedAccelCirc( p("-forcedAccelCirc").asDouble(0)),
  forcedOmegaCirc( p("-forcedOmegaCirc").asDouble(0)),
  tAccel( p("-tAccel").asDouble(-1) ) {  
    printf("Created an Active Particle with: R:%f rho:%f tAccel:%f\n",radius,rhoS,tAccel);
  }
  
  Real getCharLength() const override
  {
      return 2 * radius;
  }
  Real getCharMass() const override { return M_PI * radius * radius; }
  
  void outputSettings(std::ostream &outStream) const override
  {
    outStream << "Active Particle\n";
    outStream << "radius " << radius << std::endl;
  
    Shape::outputSettings(outStream);
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updatePosition(double dt) override;
  void updateVelocity(double dt) override;
  double reward();
  void checkFeasibility();
  void anomalyGivenTime();
};
