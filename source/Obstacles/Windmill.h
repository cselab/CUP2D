//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Shape.h"

class Windmill : public Shape
{
  // windmill stuff
  // 3 branches of same length
  const Real semiAxis[2]; 
  const Real smajax = std::max(semiAxis[0], semiAxis[1]);
  const Real sminax = std::min(semiAxis[0], semiAxis[1]);
  const Real windscale = std::sqrt(forcedu*forcedu+forcedv*forcedv);
  const Real lengthscale = getCharLength();
  std::array<Real, 2> target = {0.7, 0.7};
  Real energy = 0;

 public:

  Windmill(SimulationData& s, cubism::ArgumentParser& p, double C[2]):
  Shape(s,p,C), semiAxis{(Real) p("-semiAxisX").asDouble(), (Real) p("-semiAxisY").asDouble()}
  {
  }

  void resetAll() override
  {
    // reset all other variables
    Shape::resetAll();
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
  void updatePosition(double dt) override;

  void setTarget(std::array<Real, 2> target_pos);
  void printVelAtTarget();
  void printRewards(Real r_energy, Real r_flow);

  void printNanRewards(bool energy, Real r);

  void printValues();
  
  void act( double action );
  double reward(std::array<Real, 2> target_vel, Real C, Real D);
  std::vector<double> state();

  // Helpers for reward function
  std::vector<double> average(std::array<Real, 2> pSens) const;
  
  size_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const;

  Real getCharLength() const override
  {
    return semiAxis[0] >= semiAxis[1] ? 2*semiAxis[0] : 2*semiAxis[1];
  }
};