//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "SmartCylinder.h"
#include "ShapeLibrary.h"
#include "../Utils/BufferedLogger.h"

using namespace cubism;

void SmartCylinder::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
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
        //obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, * obstacleBlocks[vInfo[i].blockID] );
      }
  }
}

void SmartCylinder::updateVelocity(double dt)
{
  Shape::updateVelocity(dt);
  // update energy used
  energy += ( appliedForceX*u + appliedForceY*v ) * dt;
}

void SmartCylinder::updatePosition(double dt)
{
  Shape::updatePosition(dt);
}

void SmartCylinder::act( std::vector<double> action )
{
  if(action.size() != 2){
    std::cout << "Two actions required, force in X and force in Y.";
    fflush(0);
    abort();
  }
  appliedForceX = action[0];
  appliedForceY = action[1];
}

double SmartCylinder::reward( std::vector<double> target )
{
  // set dist to old dist
  oldDist = dist;

  double dX = target[0]-centerOfMass[0];
  double dY = target[1]-centerOfMass[1];

  dist = std::sqrt( dX*dX + dY*dY );

  return oldDist-dist;
}

std::vector<double> SmartCylinder::state( std::vector<double> target )
{
  // intitialize state vector
  std::vector<double> state(4);

  // relative x position
  state[0] = target[0]-centerOfMass[0];
  // relative y position
  state[1] = target[1]-centerOfMass[1];
  // current x-velocity
  state[2] = u;
  // current y-velocity
  state[3] = v;

  return state;
}
