#pragma once
#include "Fish.h"
class Naca : public Fish {
  Real Apitch, Fpitch, Mpitch, Fheave, Aheave;
  Real tAccel;
  Real fixedCenterDist;

public:
  Naca(SimulationData &s, cubism::ArgumentParser &p, Real C[2]);
  void updateVelocity(Real dt) override;
  void updatePosition(Real dt) override;
  void updateLabVelocity(int mSum[2], Real uSum[2]) override;
};
