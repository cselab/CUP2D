//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Shape.h"

struct FishData;

class Fish: public Shape
{
 public:
  const Real length, Tperiod, phaseShift;
  FishData * myFish = nullptr;
 protected:
  Real area_internal = 0, J_internal = 0;
  Real CoM_internal[2] = {0, 0}, vCoM_internal[2] = {0, 0};
  Real theta_internal = 0, angvel_internal = 0, angvel_internal_prev = 0;

  Fish(SimulationData&s, cubism::ArgumentParser&p, Real C[2]) : Shape(s,p,C),
  length(p("-L").asDouble(0.1)), Tperiod(p("-T").asDouble(1)),
  phaseShift(p("-phi").asDouble(0))  {}
  virtual ~Fish() override;

 public:
  Real getCharLength() const override {
    return length;
  }
  void removeMoments(const std::vector<cubism::BlockInfo>& vInfo) override;
  virtual void resetAll() override;
  virtual void updatePosition(Real dt) override;
  virtual void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  virtual void saveRestart( FILE * f ) override;
  virtual void loadRestart( FILE * f ) override;
};
