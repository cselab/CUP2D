

#pragma once

#include "SimulationData.h"

class Operator {
public:
  SimulationData &sim;
  Operator(SimulationData &s) : sim(s) {}
  virtual ~Operator() {}
  virtual void operator()(const Real dt) = 0;
  virtual std::string getName() = 0;
};
