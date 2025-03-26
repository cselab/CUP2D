

#include "CarlingFish.h"

using namespace cubism;

CarlingFish::CarlingFish(SimulationData &s, ArgumentParser &p, Real C[2])
    : Fish(s, p, C) {
  myFish = new AmplitudeFish(length, Tperiod, phaseShift, sim.minH);
  if (s.verbose)
    printf("[CUP2D] - AmplitudeFish %d %f %f %f\n", myFish->Nm, (double)length,
           (double)Tperiod, (double)phaseShift);
}

void CarlingFish::create(const std::vector<BlockInfo> &vInfo) {
  Fish::create(vInfo);
}

void CarlingFish::resetAll() { Fish::resetAll(); }

void AmplitudeFish::computeMidline(const Real t, const Real dt) {
  const Real rampFac = rampFactorSine(t, Tperiod);
  const Real rampFacVel = rampFactorVelSine(t, Tperiod);
  rX[0] = 0.0;
  rY[0] = rampFac * midlineLatPos(rS[0], t);
  vX[0] = 0.0;
  vY[0] =
      rampFac * midlineLatVel(rS[0], t) + rampFacVel * midlineLatPos(rS[0], t);

#pragma omp parallel for schedule(static)
  for (int i = 1; i < Nm; ++i) {
    rY[i] = rampFac * midlineLatPos(rS[i], t);
    vY[i] = rampFac * midlineLatVel(rS[i], t) +
            rampFacVel * midlineLatPos(rS[i], t);
  }

#pragma omp parallel for schedule(static)
  for (int i = 1; i < Nm; ++i) {
    const Real dy = rY[i] - rY[i - 1], ds = rS[i] - rS[i - 1];
    const Real dx = std::sqrt(ds * ds - dy * dy);
    assert(dx > 0);
    const Real dVy = vY[i] - vY[i - 1];
    const Real dVx = -dy / dx * dVy;

    rX[i] = dx;
    vX[i] = dVx;
    norX[i - 1] = -dy / ds;
    norY[i - 1] = dx / ds;
    vNorX[i - 1] = -dVy / ds;
    vNorY[i - 1] = dVx / ds;
  }

  for (int i = 1; i < Nm; ++i) {
    rX[i] += rX[i - 1];
    vX[i] += vX[i - 1];
  }

  norX[Nm - 1] = norX[Nm - 2];
  norY[Nm - 1] = norY[Nm - 2];
  vNorX[Nm - 1] = vNorX[Nm - 2];
  vNorY[Nm - 1] = vNorY[Nm - 2];
}
