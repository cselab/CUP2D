//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Waterturbine.h"
#include "ShapeLibrary.h"
#include <cmath>

using namespace cubism;

void Waterturbine::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h = sim.getH();
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    // In the case of the waterturbine we have 3 ellipses that are not centered at 0

    // Define the center of blade relative to coordinate origin at T=0, written as {x,y} with standard coordinate system
    // Distance from coordinate origin to blade (6cm) = 4 length of semi major axis (smajax) of ellipse (3cm)
    // Define orientation of ellipse, witten as (orientation + angle of blade) in Fillbock line
    //------------------------------------------------------------------------------------------------------------------------

    // BLADE 1 (left, alinged with flow)
    Real center_orig1[2] = {-4*smajax, 0};
    // center of ellipse 1 wrt to origin
    Real center1[2] = {center[0] + std::cos(orientation) * center_orig1[0] - std::sin(orientation)* center_orig1[1], 
                         center[1] + std::sin(orientation) * center_orig1[0] + std::cos(orientation) * center_orig1[1]};

    FillBlocks_Ellipse kernel1(smajax, sminax, h, center1, (orientation + M_PI/2), rhoS); //

    // BLADE 2 (top right, moving from alinged with flow towards being perpendicular to flow)
    Real center_orig2[2] = {4*smajax*std::cos(M_PI/3), 4*smajax*std::sin(M_PI/3)};
    // center of ellipse 1 wrt to origin
    Real center2[2] = {center[0] + std::cos(orientation) * center_orig2[0] - std::sin(orientation)* center_orig2[1], 
                         center[1] + std::sin(orientation) * center_orig2[0] + std::cos(orientation) * center_orig2[1]};

    FillBlocks_Ellipse kernel2(smajax, sminax, h, center2, (orientation - M_PI/6), rhoS);

    // BLADE 3 (bottom right, moving from being perpendicular to towards being aligned with flow)
    Real center_orig3[2] = {4*smajax*std::cos(M_PI/3), -(4*smajax)*std::sin(M_PI/3)};
    // center of ellipse 1 wrt to origin
    Real center3[2] = {center[0] + std::cos(orientation) * center_orig3[0] - std::sin(orientation)* center_orig3[1], 
                         center[1] + std::sin(orientation) * center_orig3[0] + std::cos(orientation) * center_orig3[1]};

    FillBlocks_Ellipse kernel3(smajax, sminax, h, center3, (orientation + M_PI/6), rhoS);

    // fill blocks for the three ellipses
    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      if(kernel1.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
      else if(kernel2.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
      else if(kernel3.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }

      ScalarBlock& B = *(ScalarBlock*)vInfo[i].ptrBlock;
      if(obstacleBlocks[vInfo[i].blockID] == nullptr) continue;
      kernel1(vInfo[i], B, * obstacleBlocks[vInfo[i].blockID]);
      kernel2(vInfo[i], B, * obstacleBlocks[vInfo[i].blockID]);
      kernel3(vInfo[i], B, * obstacleBlocks[vInfo[i].blockID]);
    }
  }
}

