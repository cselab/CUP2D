

#pragma once

#include "../Operator.h"

class Shape;

class ComputeForces : public Operator {
  const std::vector<cubism::BlockInfo> &presInfo = sim.pres->getBlocksInfo();

public:
  void operator()(const Real dt) override;

  ComputeForces(SimulationData &s);
  ~ComputeForces() {}

  std::string getName() override { return "ComputeForces"; }
};
