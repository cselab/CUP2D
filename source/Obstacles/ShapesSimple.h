//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Shape.h"

class Disk : public Shape
{
  const Real radius;
  const Real tAccel;
 public:
  Disk(SimulationData& s, cubism::ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ),
  tAccel( p("-tAccel").asDouble(-1) )
  {
    if(tAccel > 0) {
      u = 0.0;
      v = 0.0;
    }
  }

  Real getRadius() const { return radius; }

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  Real getCharMass() const override { return M_PI * radius * radius; }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(Real dt) override;
};

class HalfDisk : public Shape
{
 protected:
  const Real radius;
  const Real tAccel;

 public:
  HalfDisk( SimulationData& s, cubism::ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ),
  tAccel( p("-tAccel").asDouble(-1) )
  {
    if(tAccel > 0) {
      u = 0.0;
      v = 0.0;
    }
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  Real getCharMass() const override { return M_PI * radius * radius / 2; }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(Real dt) override;
};

class Ellipse : public Shape
{
 protected:
  const Real semiAxis[2];
  //Characteristic scales:
  const Real majax = std::max(semiAxis[0], semiAxis[1]);
  const Real minax = std::min(semiAxis[0], semiAxis[1]);
  const Real velscale = std::sqrt((rhoS/1-1)*9.81*minax);
  const Real lengthscale = majax, timescale = majax/velscale;
  //const Real torquescale = M_PI/8*pow((a*a-b*b)*velscale,2)/a/b;
  const Real torquescale = M_PI*majax*majax*velscale*velscale;

  Real Torque = 0, old_Torque = 0, old_Dist = 100;
  Real powerOutput = 0, old_powerOutput = 0;

 public:
  Ellipse(SimulationData&s, cubism::ArgumentParser&p, Real C[2]) :
    Shape(s,p,C),
    semiAxis{ (Real) p("-semiAxisX").asDouble(.1),
              (Real) p("-semiAxisY").asDouble(.2) } 
    {}

  Real getCharLength() const  override
  {
    return 2 * std::max(semiAxis[1],semiAxis[0]);
  }
  Real getCharMass() const override { return M_PI * semiAxis[1] * semiAxis[0]; }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
};

class Rectangle : public Shape
{
 protected:
  const Real extentX;
  const Real extentY;

 public:
  Rectangle(SimulationData&s, cubism::ArgumentParser&p, Real C[2]) :
    Shape(s,p,C), extentX( (Real) p("-extentX").asDouble(.2) ), extentY( (Real) p("-extentY").asDouble(.2) ) {}

  Real getCharLength() const override
  {
    return extentX;
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
};

