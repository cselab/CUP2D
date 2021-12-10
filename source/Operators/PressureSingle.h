//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"

class Shape;

#include "../Poisson/Base.h"
#include "Cubism/FluxCorrection.h"

class PressureSingle : public Operator
{
  std::shared_ptr<PoissonSolver> pressureSolver;

  bool detectCollidingObstacles() const;
  void preventCollidingObstacles() const;
  void pressureCorrection(const Real dt);
  void integrateMomenta(Shape * const shape) const;
  void penalize(const Real dt) const;

 public:
  void operator()(const Real dt);

  PressureSingle(SimulationData& s);
  ~PressureSingle();

  std::string getName() {
    return "PressureSingle";
  }
};
