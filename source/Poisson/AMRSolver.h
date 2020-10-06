//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#pragma once

#include "../Operator.h"

class AMRSolver 
{
 protected:
  SimulationData& sim;

 public:
  void solve();

  std::string getName() {
    return "AMRSolver";
  }

  AMRSolver(SimulationData& s):sim(s){};
};