//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "SimulationData.h"
#include "ObstacleBlock.h"

class Shape
{
 public: // data fields
  SimulationData& sim;
  unsigned obstacleID = 0;
  std::vector<ObstacleBlock*> obstacleBlocks;
  // general quantities
  const Real origC[2], origAng;
  Real center[2]; // for single density, this corresponds to centerOfMass
  Real centerOfMass[2];
  Real d_gm[2] = {0,0}; // distance of center of geometry to center of mass
  Real labCenterOfMass[2] = {0,0};
  Real orientation = origAng;

  const bool bFixed;
  const bool bFixedx;
  const bool bFixedy;
  const bool bForced;
  const bool bForcedx;
  const bool bForcedy;
  const bool bBlockang;
  const Real forcedu;
  const Real forcedv;
  const Real forcedomega;
  const bool bDumpSurface;
  const Real timeForced;
  const int breakSymmetryType;
  const Real breakSymmetryStrength;
  const Real breakSymmetryTime;

  Real M = 0;
  Real J = 0;
  Real u = forcedu; // in lab frame, not sim frame
  Real v = forcedv; // in lab frame, not sim frame
  Real omega = forcedomega;
  Real fluidAngMom = 0;
  Real fluidMomX = 0;
  Real fluidMomY = 0;
  Real penalDX = 0;
  Real penalDY = 0;
  Real penalM = 0;
  Real penalJ = 0;
  Real appliedForceX = 0;
  Real appliedForceY = 0;
  Real appliedTorque = 0;

  Real perimeter=0, forcex=0, forcey=0, forcex_P=0, forcey_P=0;
  Real forcex_V=0, forcey_V=0, torque=0, torque_P=0, torque_V=0;
  Real drag=0, thrust=0, lift=0, circulation=0, Pout=0, PoutNew=0, PoutBnd=0, defPower=0;
  Real defPowerBnd=0, Pthrust=0, Pdrag=0, EffPDef=0, EffPDefBnd=0;

  virtual void resetAll()
  {
    center[0] = origC[0];
    center[1] = origC[1];
    centerOfMass[0] = origC[0];
    centerOfMass[1] = origC[1];
    labCenterOfMass[0] = 0;
    labCenterOfMass[1] = 0;
    orientation = origAng;
    M = 0;
    J = 0;
    u = forcedu;
    v = forcedv;
    omega = forcedomega;
    fluidMomX = 0;
    fluidMomY = 0;
    fluidAngMom = 0;
    appliedForceX = 0;
    appliedForceY = 0;
    appliedTorque = 0;
    d_gm[0] = 0;
    d_gm[1] = 0;
    for(auto & entry : obstacleBlocks) delete entry;
    obstacleBlocks.clear();
  }

 protected:

/*
  inline void rotate(Real p[2]) const
  {
    const Real x = p[0], y = p[1];
    p[0] =  x*std::cos(orientation) + y*std::sin(orientation);
    p[1] = -x*std::sin(orientation) + y*std::cos(orientation);
  }
*/
 public:
  Shape( SimulationData& s, cubism::ArgumentParser& p, Real C[2] );

  virtual ~Shape();

  virtual Real getCharLength() const = 0;
  virtual Real getCharSpeed() const {
    return std::sqrt(forcedu*forcedu + forcedv*forcedv);
  }
  virtual Real getCharMass() const;
  virtual Real getMaxVel() const;

  virtual void create(const std::vector<cubism::BlockInfo>& vInfo) = 0;
  virtual void finalize() {};

  virtual void updateVelocity(Real dt);
  virtual void updatePosition(Real dt);

  void setCentroid(Real C[2])
  {
    this->center[0] = C[0];
    this->center[1] = C[1];
    const Real cost = std::cos(this->orientation);
    const Real sint = std::sin(this->orientation);
    this->centerOfMass[0] = C[0] - cost*this->d_gm[0] + sint*this->d_gm[1];
    this->centerOfMass[1] = C[1] - sint*this->d_gm[0] - cost*this->d_gm[1];
  }

  void setCenterOfMass(Real com[2])
  {
    this->centerOfMass[0] = com[0];
    this->centerOfMass[1] = com[1];
    const Real cost = std::cos(this->orientation);
    const Real sint = std::sin(this->orientation);
    this->center[0] = com[0] + cost*this->d_gm[0] - sint*this->d_gm[1];
    this->center[1] = com[1] + sint*this->d_gm[0] + cost*this->d_gm[1];
  }

  void getCentroid(Real centroid[2]) const
  {
    centroid[0] = this->center[0];
    centroid[1] = this->center[1];
  }

  virtual void getCenterOfMass(Real com[2]) const
  {
    com[0] = this->centerOfMass[0];
    com[1] = this->centerOfMass[1];
  }

  void getLabPosition(Real com[2]) const
  {
    com[0] = this->labCenterOfMass[0];
    com[1] = this->labCenterOfMass[1];
  }

  Real getU() const { return u; }
  Real getV() const { return v; }
  Real getW() const { return omega; }

  Real getOrientation() const
  {
    return this->orientation;
  }
  void setOrientation(const Real angle)
  {
    this->orientation = angle;
  }

  //functions needed for restarting the simulation
  virtual void saveRestart( FILE * f );
  virtual void loadRestart( FILE * f );

  struct Integrals {
    const Real x, y, m, j, u, v, a;
    Integrals(Real _x, Real _y, Real _m, Real _j, Real _u, Real _v, Real _a) :
    x(_x), y(_y), m(_m), j(_j), u(_u), v(_v), a(_a) {}
    Integrals(const Integrals&c) :
      x(c.x), y(c.y), m(c.m), j(c.j), u(c.u), v(c.v), a(c.a) {}
  };

  Integrals integrateObstBlock(const std::vector<cubism::BlockInfo>& vInfo);

  virtual void removeMoments(const std::vector<cubism::BlockInfo>& vInfo);

  virtual void updateLabVelocity( int mSum[2], Real uSum[2] );

  void penalize();

  void diagnostics();

  virtual void computeForces();
};
