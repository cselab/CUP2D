//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Shape.h"
#include "FishUtilities.h"

class CylinderNozzle : public Shape
{
  const Real radius;
  std::vector<Real> actuators_prev_value;
  std::vector<Real> actuators_next_value;
  const int Nactuators;
  const Real actuator_theta;
  Real fx_integral = 0;
  std::vector < Schedulers::ParameterSchedulerScalar > actuatorSchedulers;
  Real t_change = 0;
  const Real regularizer;
  const Real ccoef;

 public:
  std::vector<Real> actuators;
  CylinderNozzle(SimulationData& s, cubism::ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C),
  radius( p("-radius").asDouble(0.1) ), 
  Nactuators ( p("-Nactuators").asInt(2)),
  actuator_theta ( p("-actuator_theta").asDouble(10.)*M_PI/180.),
  regularizer( p("-regularizer").asDouble(0.0)),
  ccoef( p("-ccoef").asDouble(0.1) )
  {
    actuators.resize(Nactuators,0.);
    actuatorSchedulers.resize(Nactuators);
    actuators_prev_value.resize(Nactuators);
    actuators_next_value.resize(Nactuators);
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void finalize() override;
  void updateVelocity(Real dt) override
  {
     Shape::updateVelocity(dt);
     constexpr Real t1 = 0.25;
     constexpr Real t2 = 0.50;
     omega = (sim.time > t1 && sim.time < t2) ? u*getCharLength()*sin(2*M_PI*(sim.time-t1)/(t2-t1)) : 0.0;
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  void act(std::vector<Real> action, const int agentID);
  Real reward(const int agentID);
  std::vector<Real> state(const int agentID);
};
