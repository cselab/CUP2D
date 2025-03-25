//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Shape.h"
#include "FishUtilities.h"

/*
 * A cylinder with uniformly distributed actuators on its surface.
 * Each actuator can blow/suck fluid, in order to control the 
 * flow.
 */

class CylinderNozzle : public Shape
{
  const Real radius;                       //the cylinder radius
  std::vector<Real> actuators_prev_value;  //the value of each actuator at the previous step
  std::vector<Real> actuators_next_value;  //the value each actuator will obtain next
  const int Nactuators;                    //total number of actuators
  const Real actuator_theta;               //circular arc (in deg) of each actuator
  Real fx_integral = 0;                    //integral of force in x acting on the cylinder for the time it takes to switch from actuators_prev_value to actuators_next_value
  std::vector < Schedulers::ParameterSchedulerScalar > actuatorSchedulers; //object that interpolates in time from actuators_prev_value to actuators_next_value (actuator values change smoothly over time)
  Real t_change = 0; //used to record when actuator values change
  const Real regularizer; //used to regularize the reward in Reinforcement Learning
  const Real ccoef; //max actuation velocity allowed in ccoef * cylinder velocity

 public:
  std::vector<Real> actuators; //current values of actuators

  //class constructor with some default values for the parameters
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

  //this will introduce a small distrurbance in the cylinder velocity, which helps
  //create an assymetric wake with vortex shedding
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
