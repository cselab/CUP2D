//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "../Shape.h"

class SmartCylinder : public Shape
{
  const double radius;
 public:
  Real energy;
  Real dist, oldDist;

  SmartCylinder(SimulationData& s, cubism::ArgumentParser& p, double C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ) 
  {}

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
  void updatePosition(double dt) override;

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  
  void act( std::vector<double> action );
  double reward( std::vector<double> target );
  std::vector<double> state( std::vector<double> target );

  // Helpers for State function
  std::array<Real, 2> sensVel(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const;
  
  size_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const;
};
