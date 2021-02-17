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

#define NSENSORS 8

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
  std::vector<double> state(4+2*NSENSORS);

  // relative x position
  state[0] = target[0]-centerOfMass[0];
  // relative y position
  state[1] = target[1]-centerOfMass[1];
  // update dist
  dist = std::sqrt( state[0]*state[0] + state[1]*state[1] );
  // current x-velocity
  state[2] = u;
  // current y-velocity
  state[3] = v;

  for(size_t sens = 0; sens<NSENSORS; sens++)
  {
    // equally spaced sensors
    const Real theta = sens * 2 * M_PI / NSENSORS;
    const Real cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    const std::array<Real,2> position{centerOfMass[0] - 1.1 * radius * cosTheta, centerOfMass[1] + 1.1 * radius * sinTheta};

    // Get velInfo
    const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
    // get velocity at sensor location
    std::array<Real,2> vSens = sensVel( position, velInfo );
    // subtract surface velocity from sensor velocity
    state[4+2*sens]   = vSens[0] - u;
    state[4+2*sens+1] = vSens[1] - v;
  }

  return state;
}

/* helpers to compute sensor information */

// return flow velocity at point of flow sensor:
std::array<Real, 2> SmartCylinder::sensVel(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // get blockId
  const size_t blockId = holdingBlockID(pSens, velInfo);

  // get block
  const auto& sensBinfo = velInfo[blockId];

  // get origin of block
  const std::array<Real,2> oSens = sensBinfo.pos<Real>(0, 0);

  // get inverse gridspacing in block
  const Real invh = 1/velInfo[blockId].h_gridpoint;

  // get index for sensor
  const std::array<int,2> iSens = safeIdInBlock(pSens, oSens, invh);

  // get velocity field at point
  const VectorBlock& b = * (const VectorBlock*) sensBinfo.ptrBlock;

  return std::array<Real, 2>{{b(iSens[0], iSens[1]).u[0],
                              b(iSens[0], iSens[1]).u[1]}};
};

// function that finds block id of block containing pos (x,y)
size_t SmartCylinder::holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // std::cout << "pos=(" << pos[0] << ", " << pos[1] << ")\n"; 
  for(size_t i=0; i<velInfo.size(); ++i)
  {
    // get gridspacing in block
    const Real h = velInfo[i].h_gridpoint;

    // compute lower left corner of block
    std::array<Real,2> MIN = velInfo[i].pos<Real>(0, 0);
    for(int j=0; j<2; ++j)
      MIN[j] -= 0.5 * h; // pos returns cell centers

    // compute top right corner of block
    std::array<Real,2> MAX = velInfo[i].pos<Real>(VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    for(int j=0; j<2; ++j)
      MAX[j] += 0.5 * h; // pos returns cell centers

    // check whether point is inside block
    // std::cout << "MIN=(" << MIN[0] << ", " << MIN[1] << "); MAX=(" << MAX[0] << ", " << MAX[1] << ")\n"; 
    if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
    {
      // select obstacle block
      if(obstacleBlocks[i] != nullptr ){
        return i;
      }
    }
  }
  printf("ABORT: coordinate (%g,%g) could not be associated to obstacle block\n", pos[0], pos[1]);
  fflush(0); abort();
  return 0;
};

// function that gives indice of point in block
std::array<int, 2> SmartCylinder::safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const
{
  const int indx = (int) std::round((pos[0] - org[0])*invh);
  const int indy = (int) std::round((pos[1] - org[1])*invh);
  const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
  const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
  return std::array<int, 2>{{ix, iy}};
};
