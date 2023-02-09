//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Shape.h"

class CylinderNozzle : public Shape
{
  const Real radius;
  std::vector<Real> actuators;
  const int Nactuators;
  const Real actuator_theta;

 public:

  CylinderNozzle(SimulationData& s, cubism::ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ), 
		Nactuators ( p("-Nactuators").asInt(2)), actuator_theta ( p("-actuator_theta").asDouble(10.)*M_PI/180.)
  {
	  actuators.resize(Nactuators,0.);
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  void act(std::vector<Real> action, const int agentID);
  Real reward(const int agentID);
  std::vector<Real> state(const int agentID);
};
