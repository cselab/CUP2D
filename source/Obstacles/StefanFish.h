//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Fish.h"

#define STEFANS_SENSORS_STATE

class StefanFish: public Fish
{
  const bool bCorrectTrajectory;
  const bool bCorrectPosition;
 public:
  void act(const Real lTact, const std::vector<double>& a) const;
  double getLearnTPeriod() const;
  double getPhase(const double t) const;

  void resetAll() override;
  StefanFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;

  // member functions for state in RL
  std::vector<double> state(Shape*const p) const;
  std::vector<double> state() const;

  // Helpers for State function
  size_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const;

  std::array<Real, 2> skinVel(const std::array<Real,2> pSkin, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<Real, 2> sensVel(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const;

};
