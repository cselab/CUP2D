

#pragma once

#include "Fish.h"

class ZebraFish : public Fish {
public:
  ZebraFish(SimulationData &s, cubism::ArgumentParser &p, Real C[2]);
  void resetAll() override;
  void create(const std::vector<cubism::BlockInfo> &vInfo) override;
  void act(const Real lTact, const std::vector<Real> &a) const;

  std::vector<Real> state() const;

  void getTarget(Real outTarget[2]) const;
  void setTarget(Real inTarget[2]) const;
  Real getRadialDisplacement() const;
  Real getDistanceFromTarget() const;
  Real getTimeNextAct() const;
};
