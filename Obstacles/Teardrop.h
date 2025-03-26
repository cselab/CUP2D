

#pragma once

#include "Fish.h"

class Teardrop : public Fish {
  const Real Apitch;
  const Real Fpitch;
  const Real tAccel;
  const Real fixedCenterDist;

public:
  Teardrop(SimulationData &s, cubism::ArgumentParser &p, Real C[2]);
  void updateVelocity(Real dt) override;
  void updateLabVelocity(int mSum[2], Real uSum[2]) override;
};
