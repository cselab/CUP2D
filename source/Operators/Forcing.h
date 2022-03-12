//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"

class Forcing : public Operator
{
protected:
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

 public:
  Forcing(SimulationData& s) : Operator(s) { }

  void operator() (const Real dt) override;

  std::string getName() override
  {
    return "Forcing";
  }
};
