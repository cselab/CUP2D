//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once
#include "Fish.h"
#include "FishData.h"
#include "FishUtilities.h"

class CarlingFish: public Fish
{
 public:
  CarlingFish(SimulationData&s, cubism::ArgumentParser&p, Real C[2]);
  void resetAll() override;
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
};

class AmplitudeFish : public FishData
{
  const Real phaseShift, Tperiod;
 public:
  inline Real midlineLatPos(const Real s, const Real t) const {
    const Real arg = 2*M_PI*(s/length - t/Tperiod + phaseShift);
    return 4./33. *  (s + 0.03125*length)*std::sin(arg);
  }

  inline Real midlineLatVel(const Real s, const Real t) const {
      const Real arg = 2*M_PI*(s/length - t/Tperiod + phaseShift);
      return - 4./33. * (s + 0.03125*length) * (2*M_PI/Tperiod) * std::cos(arg);
  }

  inline Real rampFactorSine(const Real t, const Real T) const {
    return (t<T ? std::sin(0.5*M_PI*t/T) : 1.0);
  }

  inline Real rampFactorVelSine(const Real t, const Real T) const {
    return (t<T ? 0.5*M_PI/T * std::cos(0.5*M_PI*t/T) : 0.0);
  }

  AmplitudeFish(Real L, Real T, Real phi, Real _h)
  : FishData(L, _h),  phaseShift(phi),  Tperiod(T) { _computeWidth(); }

  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override
  {
    const Real sb=.04*length, st=.95*length, wt=.01*length, wh=.04*length;
    if(s<0 or s>L) return 0;
    return (s<sb ? std::sqrt(2*wh*s -s*s) :
           (s<st ? wh-(wh-wt)*std::pow((s-sb)/(st-sb),1) : // pow(.,2) is 3D
           (wt * (L-s)/(L-st))));
  }
};

