#pragma once
#include "Fish.h"
#include "FishData.h"
#include "FishUtilities.h"
#include "Naca.h"
class SmartNaca : public Naca {
  const int Nactuators;
  const Real actuator_ds;
  std::vector<Real> actuators_prev_value;
  std::vector<Real> actuators_next_value;
  std::vector<Schedulers::ParameterSchedulerScalar> actuatorSchedulers;
  Real t_change = 0;
  Real fx_integral = 0;
  const Real thickness;
  const Real regularizer;
  Real value1, value2, value3, value4;

public:
  std::vector<Real> actuators;
  SmartNaca(SimulationData &s, cubism::ArgumentParser &p, Real C[2]);
  void finalize() override;
  void act(std::vector<Real> action, const int agentID);
  Real reward(const int agentID);
  std::vector<Real> state(const int agentID);
};
