//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once
#include "Fish.h"
class Naca: public Fish
{
  const Real Apitch; //aplitude of sinusoidal pitch angle
  const Real Fpitch; //frequency
  Real time;
 public:

  Naca(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
  void resetAll() override;
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
};
