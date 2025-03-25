//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Fish.h"

class Teardrop: public Fish
{
  const Real Apitch; // aplitude of sinusoidal pitch angle
  const Real Fpitch; // frequency
  const Real tAccel; // time to accelerate to target velocity
  const Real fixedCenterDist; // distance s/L from CoM where hydrofoil is fixed

 public:

  Teardrop(SimulationData&s, cubism::ArgumentParser&p, Real C[2]);
  void updateVelocity(Real dt) override;
  void updateLabVelocity( int mSum[2], Real uSum[2] ) override;
};
