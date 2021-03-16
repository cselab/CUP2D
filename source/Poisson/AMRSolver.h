//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

class AMRSolver 
{
 protected:
  SimulationData& sim;
 public:
  std::string getName() {
    return "AMRSolver";
  }
  AMRSolver(SimulationData& s);
  cubism::FluxCorrection<ScalarGrid,ScalarBlock> Corrector;
  void solve();
  void Get_LHS (ScalarGrid * lhs, ScalarGrid * x);
  void getZ(){};
};
