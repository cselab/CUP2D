//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Pascal Weber (webepasc@ethz.ch).
//


#pragma once
#include "Fish.h"
class Naca: public Fish
{
  const Real Apitch; // aplitude of sinusoidal pitch angle
  const Real Fpitch; // frequency
  const Real tAccel; // time to accelerate to target velocity
  const Real fixedCenterDist; // distance s/L from CoM where hydrofoil is fixed

 public:

  Naca(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
  void resetAll() override;
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
  void updateLabVelocity( int mSum[2], double uSum[2] ) override;
};
