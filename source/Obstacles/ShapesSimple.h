//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "../Shape.h"

class Disk : public Shape
{
  const double radius;
  const Real tAccel;
  const double xCenterRotation;
  const double yCenterRotation;
  const double x0;
  const double y0;
  const double forcedomegaCirc;
  const double forcedlinCirc;

  double omegaCirc = forcedomegaCirc;
  double linCirc = forcedlinCirc;
  
 public:
  Disk(SimulationData& s, cubism::ArgumentParser& p, double C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ),
  xCenterRotation( p("-xCenterRotation").asDouble(-1) ), yCenterRotation( p("-yCenterRotation").asDouble(-1) ),
  forcedomegaCirc( p("-circVel").asDouble(0)),
  forcedlinCirc( p("-linCircVel").asDouble(0)),
  x0( p("-xpos").asDouble(.5*sim.extents[0])),
  y0( p("-ypos").asDouble(.5*sim.extents[1])),
  tAccel( p("-tAccel").asDouble(-1) ) {
    printf("Created a Disk with: R:%f rho:%f tAccel:%f\n",radius,rhoS,tAccel);
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  Real getCharMass() const override { return M_PI * radius * radius; }

  void outputSettings(std::ostream &outStream) const override
  {
    outStream << "Disk\n";
    outStream << "radius " << radius << std::endl;

    Shape::outputSettings(outStream);
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
  void updatePosition(double dt) override;
};

class HalfDisk : public Shape
{
 protected:
  const double radius;
  const Real tAccel;

 public:
  HalfDisk( SimulationData& s, cubism::ArgumentParser& p, double C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ),
  tAccel( p("-tAccel").asDouble(-1) ) {
    printf("Created a half Disk with: R:%f rho:%f\n",radius,rhoS);
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  Real getCharMass() const override { return M_PI * radius * radius / 2; }

  void outputSettings(std::ostream &outStream) const override
  {
    outStream << "HalfDisk\n";
    outStream << "radius " << radius << std::endl;

    Shape::outputSettings(outStream);
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
};

class Ellipse : public Shape
{
 protected:
  const double semiAxis[2];
  //Characteristic scales:
  const double majax = std::max(semiAxis[0], semiAxis[1]);
  const double minax = std::min(semiAxis[0], semiAxis[1]);
  const Real velscale = std::sqrt((rhoS/1-1)*9.81*minax);
  const Real lengthscale = majax, timescale = majax/velscale;
  //const Real torquescale = M_PI/8*pow((a*a-b*b)*velscale,2)/a/b;
  const Real torquescale = M_PI*majax*majax*velscale*velscale;

  Real Torque = 0, old_Torque = 0, old_Dist = 100;
  Real powerOutput = 0, old_powerOutput = 0;

 public:
  Ellipse(SimulationData&s, cubism::ArgumentParser&p, double C[2]) :
    Shape(s,p,C),
    semiAxis{ (Real) p("-semiAxisX").asDouble(.1),
              (Real) p("-semiAxisY").asDouble(.2) } {
    printf("Created ellipse semiAxis:[%f %f] rhoS:%f a:%f b:%f velscale:%f lengthscale:%f timescale:%f torquescale:%f\n", semiAxis[0], semiAxis[1], rhoS, majax, minax, velscale, lengthscale, timescale, torquescale); fflush(0);
  }

  Real getCharLength() const  override
  {
    return 2 * std::max(semiAxis[1],semiAxis[0]);
  }
  Real getCharMass() const override { return M_PI * semiAxis[1] * semiAxis[0]; }

  void outputSettings(std::ostream &outStream) const override
  {
    outStream << "Ellipse\n";
    outStream << "semiAxisX " << semiAxis[0] << "\n";
    outStream << "semiAxisY " << semiAxis[1] << "\n";

    Shape::outputSettings(outStream);
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
};

class DiskVarDensity : public Shape
{
 protected:
  const double radius;
  const double rhoTop;
  const double rhoBot;

 public:
  DiskVarDensity( SimulationData& s, cubism::ArgumentParser& p, double C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ),
  rhoTop(p("-rhoTop").asDouble(rhoS) ), rhoBot(p("-rhoBot").asDouble(rhoS) ) {
    d_gm[0] = 0;
    // based on weighted average between the centers of mass of half-disks:
    d_gm[1] = -4.*radius/(3.*M_PI) * (rhoTop-rhoBot)/(rhoTop+rhoBot);

    centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
    centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
  }

  Real getCharLength() const  override
  {
    return 2 * radius;
  }
  Real getMinRhoS() const override {
    return std::min(rhoTop, rhoBot);
  }
  Real getCharMass() const override { return M_PI * radius * radius; }
  bool bVariableDensity() const override {
    assert(std::fabs(rhoTop-rhoBot)>std::numeric_limits<Real>::epsilon());
    const bool bTop = std::fabs(rhoTop-1.)>std::numeric_limits<Real>::epsilon();
    const bool bBot = std::fabs(rhoBot-1.)>std::numeric_limits<Real>::epsilon();
    return bTop || bBot;
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;

  void outputSettings(std::ostream &outStream) const override
  {
    outStream << "DiskVarDensity\n";
    outStream << "radius " << radius << "\n";
    outStream << "rhoTop " << rhoTop << "\n";
    outStream << "rhoBot " << rhoBot << "\n";

    Shape::outputSettings(outStream);
  }
};

class EllipseVarDensity : public Shape
{
  protected:
   const double semiAxisX;
   const double semiAxisY;
   const double rhoTop;
   const double rhoBot;

  public:
   EllipseVarDensity(SimulationData&s, cubism::ArgumentParser&p, double C[2] ) :
   Shape(s,p,C),
   semiAxisX( p("-semiAxisX").asDouble(0.1) ),
   semiAxisY( p("-semiAxisY").asDouble(0.1) ),
   rhoTop(p("-rhoTop").asDouble(rhoS) ), rhoBot(p("-rhoBot").asDouble(rhoS) ) {
     d_gm[0] = 0;
     // based on weighted average between the centers of mass of half-disks:
     d_gm[1] = -4.*semiAxisY/(3.*M_PI) * (rhoTop-rhoBot)/(rhoTop+rhoBot);

     centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
     centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
   }

   Real getCharLength() const override {
     return 2 * std::max(semiAxisX, semiAxisY);
   }
   Real getCharMass() const override { return M_PI * semiAxisX * semiAxisY; }
   Real getMinRhoS() const override {
     return std::min(rhoTop, rhoBot);
   }
   bool bVariableDensity() const override {
     assert(std::fabs(rhoTop-rhoBot)>std::numeric_limits<Real>::epsilon());
     const bool bTop= std::fabs(rhoTop-1.)>std::numeric_limits<Real>::epsilon();
     const bool bBot= std::fabs(rhoBot-1.)>std::numeric_limits<Real>::epsilon();
     return bTop || bBot;
   }

   void create(const std::vector<cubism::BlockInfo>& vInfo) override;

   void outputSettings(std::ostream &outStream) const override
   {
     outStream << "Ellipse\n";
     outStream << "semiAxisX " << semiAxisX << "\n";
     outStream << "semiAxisY " << semiAxisY << "\n";
     outStream << "rhoTop " << rhoTop << "\n";
     outStream << "rhoBot " << rhoBot << "\n";

     Shape::outputSettings(outStream);
   }
};
