#include "SmartCylinder.h"
#include "../Utils/BufferedLogger.h"
#include "ShapeLibrary.h"
using namespace cubism;
void SmartCylinder::create(const std::vector<BlockInfo> &vInfo) {
  const Real h = sim.getH();
  for (auto &entry : obstacleBlocks)
    delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock *>(vInfo.size(), nullptr);
#pragma omp parallel
  {
    FillBlocks_Cylinder kernel(radius, h, center);
#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < vInfo.size(); i++)
      if (kernel.is_touching(vInfo[i])) {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        ScalarBlock &b = *(ScalarBlock *)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID]);
      }
  }
}
void SmartCylinder::updateVelocity(Real dt) {
  Shape::updateVelocity(dt);
  energy += (appliedForceX * u + appliedForceY * v) * dt;
}
void SmartCylinder::updatePosition(Real dt) { Shape::updatePosition(dt); }
void SmartCylinder::act(std::vector<Real> action) {
  if (action.size() != 2) {
    std::cout << "Two actions required, force in X and force in Y.";
    fflush(0);
    abort();
  }
  appliedForceX = action[0];
  appliedForceY = action[1];
}
Real SmartCylinder::reward(std::vector<Real> target) {
  oldDist = dist;
  Real dX = target[0] - centerOfMass[0];
  Real dY = target[1] - centerOfMass[1];
  dist = std::sqrt(dX * dX + dY * dY);
  return oldDist - dist;
}
std::vector<Real> SmartCylinder::state(std::vector<Real> target) {
  std::vector<Real> state(4);
  state[0] = target[0] - centerOfMass[0];
  state[1] = target[1] - centerOfMass[1];
  state[2] = u;
  state[3] = v;
  return state;
}
