//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

class advDiffSGS : public Operator
{
protected:
  const std::vector<cubism::BlockInfo>& velInfo   = sim.vel->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& vOldInfo  = sim.vOld->getBlocksInfo();

 public:
  advDiffSGS(SimulationData& s) : Operator(s) { }

  void operator() (const Real dt) override;

  std::string getName() override
  {
    return "advDiffSGS";
  }
};
