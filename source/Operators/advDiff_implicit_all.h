//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//


#pragma once

#include "../Operator.h"

class LinearSolverDelta;

class advDiff_implicit_all : public Operator
{
  const std::vector<cubism::BlockInfo> & tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo> & vOldInfo = sim.vOld->getBlocksInfo();
  
  LinearSolverDelta *  mySolver;

 public:
  advDiff_implicit_all(SimulationData& s);
  ~advDiff_implicit_all();
  
  void operator()(const double dt);

  void explicit_update(double dt);

  //void Jacobi(int max_iter,double dt);

  std::string getName()
  {
    return "advDiff_implicit_all";
  }
};