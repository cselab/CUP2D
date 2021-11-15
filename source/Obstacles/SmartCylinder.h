//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Shape.h"

class SmartCylinder : public Shape
{
  const Real radius;
 public:
  Real energy;
  Real dist, oldDist;

  SmartCylinder(SimulationData& s, cubism::ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ) 
  {}

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(Real dt) override;
  void updatePosition(Real dt) override;
  virtual void resetAll() override
  {
    Shape::resetAll();
    energy = 0; 
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  
  void act( std::vector<Real> action );
  Real reward( std::vector<Real> target );
  std::vector<Real> state( std::vector<Real> target );
};
