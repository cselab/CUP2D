

#include "Waterturbine.h"
#include "ShapeLibrary.h"
#include <cmath>

using namespace cubism;

void Waterturbine::create(const std::vector<BlockInfo> &vInfo) {
  const Real h = sim.getH();
  for (auto &entry : obstacleBlocks)
    delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock *>(vInfo.size(), nullptr);

#pragma omp parallel
  {

    Real center_orig1[2] = {-4 * smajax, 0};

    Real center1[2] = {center[0] + std::cos(orientation) * center_orig1[0] -
                           std::sin(orientation) * center_orig1[1],
                       center[1] + std::sin(orientation) * center_orig1[0] +
                           std::cos(orientation) * center_orig1[1]};

    FillBlocks_Ellipse kernel1(smajax, sminax, h, center1,
                               (orientation + M_PI / 2));

    Real center_orig2[2] = {4 * smajax * std::cos(M_PI / 3),
                            4 * smajax * std::sin(M_PI / 3)};

    Real center2[2] = {center[0] + std::cos(orientation) * center_orig2[0] -
                           std::sin(orientation) * center_orig2[1],
                       center[1] + std::sin(orientation) * center_orig2[0] +
                           std::cos(orientation) * center_orig2[1]};

    FillBlocks_Ellipse kernel2(smajax, sminax, h, center2,
                               (orientation - M_PI / 6));

    Real center_orig3[2] = {4 * smajax * std::cos(M_PI / 3),
                            -(4 * smajax) * std::sin(M_PI / 3)};

    Real center3[2] = {center[0] + std::cos(orientation) * center_orig3[0] -
                           std::sin(orientation) * center_orig3[1],
                       center[1] + std::sin(orientation) * center_orig3[0] +
                           std::cos(orientation) * center_orig3[1]};

    FillBlocks_Ellipse kernel3(smajax, sminax, h, center3,
                               (orientation + M_PI / 6));

#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < vInfo.size(); i++) {
      if (kernel1.is_touching(vInfo[i])) {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear();
      } else if (kernel2.is_touching(vInfo[i])) {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear();
      } else if (kernel3.is_touching(vInfo[i])) {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear();
      }

      ScalarBlock &B = *(ScalarBlock *)vInfo[i].ptrBlock;
      if (obstacleBlocks[vInfo[i].blockID] == nullptr)
        continue;
      kernel1(vInfo[i], B, *obstacleBlocks[vInfo[i].blockID]);
      kernel2(vInfo[i], B, *obstacleBlocks[vInfo[i].blockID]);
      kernel3(vInfo[i], B, *obstacleBlocks[vInfo[i].blockID]);
    }
  }
}
