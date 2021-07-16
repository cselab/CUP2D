//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once
#include "Fish.h"
class CarlingFish: public Fish
{
 public:

  CarlingFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
  void resetAll() override;
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
};
