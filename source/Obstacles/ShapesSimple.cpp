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
void ElasticDisk::create(const std::vector<BlockInfo>& vInfo,Real signal)
{
  // precondition: setvInfo to a positive number
  const Real h = sim.getH();
  // a) define kernel
  FillBlocks_ElasticDisk kernel(radius,center,h,rhoS);
  // b) use kernel to get sdf
  for(size_t i=0;i<vInfo.size();i++)
  {
    if (obstacleBlocks[vInfo[i].blockID]==nullptr) continue;
    ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
    kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID]);
  }
  // c) reinitialize sdf (warning: vInfo is the info of sim.tmp)
  FastMarching kernel2(sim,obstacleBlocks);
  for(size_t=0;t<10;t++)  cubism::compute<ScalarLab>(kernel2,sim.tmp,sim.tmp,signal);
  // d) use kernel.istouching to reconstruct obstacleblock
  kernel.setextent(0.5*(kernel2.maxx-kernel2.minx),0.5*(kernel2.maxy-kernel2.miny),0.5*(kernel2.maxx+kernel2.minx),0.5*(kernel2.maxy+kernel2.miny));
  for(size_t i=0; i<vInfo.size(); i++)
  {
    if(kernel.is_touching(vInfo[i]))
    {
      if (obstacleBlocks[vInfo[i].blockID] == nullptr)
      {
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
      }
    }
    else
    {
      if (obstacleBlocks[vInfo[i].blockID] != nullptr)
      {
        delete obstacleblock[vInfo[i].blockID];
        obstacleblock[vInfo[i].blockID]=nullptr;
      }
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
