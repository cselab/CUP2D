//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Fish.h"
#include "../Utils/BufferedLogger.h"


class Naca: public Fish
{
  /*
   Hydrofoil motion is defined as:

      Rotation:
      a(t) = Mpitch + Apitch*sin(2*pi*Fpitch*t), where:
             a(t)   :pitching angle
             Mpitch :mean pitch angle
             Fpitch :pitching frequency
      omega(t) = da/dt
      Rotation can be defined around a point that is located at a distance of
      d = fixedCenterDist*length for the hydrofoil's center of mass.
      In this case, we add the following velocity to the motion:
        u_rot = - d*omega(t)*sin(a(t))
        v_rot = + d*omega(t)*cos(a(t))

      Heaving motion:
      y(t) = Aheave*cos(2*pi*Fheave*t)
      v(t) = dy/dt = -2.0*pi*Fheave*Aheave*sin(2*pi*Fheave*t)

      It is also possible to add a constant velocity (uforced,vforced) to the motion.
  */
  Real Apitch, Fpitch, Mpitch, Fheave, Aheave;
  Real tAccel; // time to accelerate to target velocity
  Real fixedCenterDist; // distance s/L from CoM where hydrofoil is fixed

 public:
  Naca(SimulationData&s, cubism::ArgumentParser&p, Real C[2]);
  void updateVelocity(Real dt) override;
  void updatePosition(Real dt) override;
  void updateLabVelocity( int mSum[2], Real uSum[2] ) override;
};
