//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

class advDiff : public Operator
{
  const std::vector<cubism::BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& vOldInfo  = sim.vOld->getBlocksInfo();

 public:
  advDiff(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  void step(const int coef);

  std::string getName()
  {
    return "advDiff";
  }
};
