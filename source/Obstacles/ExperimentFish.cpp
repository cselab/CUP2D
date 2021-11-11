//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "CarlingFish.h"
#include "FishData.h"
#include "FishUtilities.h"

using namespace cubism;

ExperimentFish::ExperimentFish(SimulationData&s, ArgumentParser&p, double C[2])
  : Fish(s,p,C), timeStart(p("-tStart").asDouble()), dtDataset(p("-dtDataset").asDouble()) 
{
  const std::string path = p("-path").asString();
  myFish = new ExperimentDataFish(length, path, sim.minH);
  if( s.verbose ) printf("[CUP2D] - ExperimentDataFish %d %f\n", myFish->Nm, length);
}

void ExperimentFish::create(const std::vector<BlockInfo>& vInfo) {
  // Only start creating the fish after timeStart has passed
  if( sim.time < timeStart )
    return
  Fish::create(vInfo);
}

void ExperimentFish::updatePosition(double dt)
{
  Shape::updatePosition(dt);
}

void ExperimentFish::updateVelocity(double dt)
{
  u = myfish->u;
  v = myfish->v;
  omega = myfish->omega;
}

void ExperimentDataFish::loadCenterOfMass( const std::string path ) {
  std::string filename = path + "Fish2_COM.txt";
  std::ifstream myfile(filename);
  if( myfile.is_open() )
  {
    double tempCoM;
    while( std::getline(myfile,line) )
    {
      std::vector<double> centerOfMass;
      std::istringstream readingStream(line);
      while (readingStream >> tempCoM)
        centerOfMass.push_back(tempCoM);
      centerOfMassData.push_back(centerOfMass);
    }
    myfile.close();
  }
  else{
    cout << "[ExperimentFish] Unable to open center of mass file " << filename << std::endl;
    fflush(0);
    abort();
  }
}

void ExperimentDataFish::loadCenterlines( const std::string path ) {
  std::string filename = path + "Fish2.txt";
  std::ifstream myfile(filename);
  if( myfile.is_open() )
  {
    double tempCenterline;
    while( std::getline(myfile,line) )
    {
      std::vector<double> centerline;
      std::istringstream readingStream(line);
      while (readingStream >> tempCenterline)
        centerline.push_back(tempCenterline);
      centerlineData.push_back(centerline);
    }
    myfile.close();
  }
  else{
    cout << "[ExperimentDataFish] Unable to open centerline file " << filename << std::endl;
    fflush(0);
    abort();
  }
}


void ExperimentDataFish::computeMidline(const Real t, const Real dt)
{
  if( sim.time >= tNext )
  {
    tLast = tNext;
    tNext += dtDataset;
    idxLast = idxNext;
    idxNext++;
    u     = ( centerOfMassData[idxNext][0] - centerOfMassData[idxLast][0] ) / dtDataset;
    u     = ( centerOfMassData[idxNext][1] - centerOfMassData[idxLast][1] ) / dtDataset;
    omega = ( centerOfMassData[idxNext][2] - centerOfMassData[idxLast][2] ) / dtDataset;
    curvatureScheduler.transition(t, tLast, tNext, midlineData[idxLast], midlineData[idxNext]);
  }
  curvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rY, vY);

  rX[0] = 0.0;
  vX[0] = 0.0;

  #pragma omp parallel for schedule(static)
  for(int i=1; i<Nm; ++i) {
    const Real dy = rY[i]-rY[i-1], ds = rS[i]-rS[i-1];
    const Real dx = std::sqrt(ds*ds-dy*dy);
    assert(dx>0);
    const Real dVy = vY[i]-vY[i-1];
    const Real dVx = - dy/dx * dVy; // ds^2 = dx^2+dy^2 -> ddx = -dy/dx*ddy

    rX[i] = dx;
    vX[i] = dVx;
    norX[ i-1] = -dy/ds;
    norY[ i-1] =  dx/ds;
    vNorX[i-1] = -dVy/ds;
    vNorY[i-1] =  dVx/ds;
  }

  for(int i=1; i<Nm; ++i) { 
    rX[i] += rX[i-1]; 
    vX[i] += vX[i-1]; 
  }

  norX[Nm-1] = norX[Nm-2];
  norY[Nm-1] = norY[Nm-2];
  vNorX[Nm-1] = vNorX[Nm-2];
  vNorY[Nm-1] = vNorY[Nm-2];
}