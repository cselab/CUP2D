//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Fish.h"
#include "Naca.h"
#include "FishUtilities.h"
#include "FishData.h"

/*
 * This is a NACA0012 airfoil equiped with actuators on its surface.
 * The actuators impose a velocity field that is normal to the airfoil's surface
 * (positive for blowing, negative for suction).
 */

class SmartNaca: public Naca
{
  const int Nactuators;   // number of actuators on the airfoil surface
  const Real actuator_ds; // length of each actuator (as a fraction of the airfoil's cord/length: 0 < actuator_ds < 1)

  //If actuator values change, we want the transition to be smooth in time. 
  //When a change happens, we use the 'actuatorSchedulers' object that will 
  //smoothly interpolate from the previous actuation values (actuators_prev_value) 
  //to the next actuation values (actuators_next_value). We also use 't_change'
  //to record when the change of actuation strengths happened.
  std::vector<Real> actuators_prev_value;
  std::vector<Real> actuators_next_value;
  std::vector < Schedulers::ParameterSchedulerScalar > actuatorSchedulers;
  Real t_change = 0;

  Real fx_integral = 0;   //time integral of x-force (drag) of airfoil, used when computing the RL reward
  const Real thickness;   //airfoil thickness parameter
  const Real regularizer; //weight for regularized reward

  Real value1,value2,value3,value4;

 public:
  std::vector<Real> actuators; //current values of the actuators

  //Class constructor
  SmartNaca(SimulationData&s, cubism::ArgumentParser&p, Real C[2]);

  //Called at every timestep to impose actuation velocities.
  void finalize() override;

  //Called when an RL action should be taken, in order to change actuation velocities.
  void act( std::vector<Real> action, const int agentID);

  //Called to compute RL reward.
  Real reward(const int agentID);

  //Called to compute RL state.
  std::vector<Real> state(const int agentID);
};
