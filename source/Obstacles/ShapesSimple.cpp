//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "ShapeLibrary.h"
#include "ShapesSimple.h"

using namespace cubism;

void Disk::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h = sim.getH();
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Cylinder kernel(radius, h, center, rhoS);

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID] );
      }
  }
}

void Disk::updateVelocity(Real dt)
{
  Shape::updateVelocity(dt);
  if(tAccel > 0) {
    if(bForcedx && sim.time < tAccel) u = (sim.time/tAccel)*forcedu;
    if(bForcedy && sim.time < tAccel) v = (sim.time/tAccel)*forcedv;
  }
}

void HalfDisk::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h = sim.getH();
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_HalfCylinder kernel(radius, h, center, rhoS, orientation);

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID]);
      }
  }
}

void HalfDisk::updateVelocity(Real dt)
{
  Shape::updateVelocity(dt);
  if(tAccel > 0) {
    if(bForcedx && sim.time < tAccel) u = (sim.time/tAccel)*forcedu;
    if(bForcedy && sim.time < tAccel) v = (sim.time/tAccel)*forcedv;
  }
}

void Ellipse::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h = sim.getH();
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Ellipse kernel(semiAxis[0], semiAxis[1], h, center, orientation, rhoS);

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID]);
      }
  }
}

void Rectangle::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h = sim.getH();
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Rectangle kernel(extentX, extentY, h, center, orientation, rhoS);

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
    if(kernel.is_touching(vInfo[i]))
    {
      assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
      obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
      ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
      kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID]);
    }
  }
}
