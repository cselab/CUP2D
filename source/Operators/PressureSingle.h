//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"

class PoissonSolver;
class Shape;

#include "../Poisson/AMRSolver.h"
#include "Cubism/FluxCorrection.h"

class PressureSingle : public Operator
{
  AMRSolver * pressureSolver;

  bool detectCollidingObstacles() const;
  void preventCollidingObstacles() const;
  void pressureCorrection(const double dt);
  void integrateMomenta(Shape * const shape) const;
  void penalize(const double dt) const;

 public:
  void operator()(const double dt);

  PressureSingle(SimulationData& s);
  ~PressureSingle();

  std::string getName() {
    return "PressureSingle";
  }
};
