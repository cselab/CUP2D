//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "ExperimentFish.h"
#include "FishData.h"
#include "FishUtilities.h"

using namespace cubism;

ExperimentFish::ExperimentFish(SimulationData&s, ArgumentParser&p, Real C[2])
  : Fish(s,p,C), timeStart( p("-tStart").asDouble() )  
{
  const std::string path = p("-path").asString();
  const Real dtDataset = p("-dtDataset").asDouble();
  myFish = new ExperimentDataFish(length, sim.minH, path, timeStart, dtDataset);
  if( s.verbose ) printf("[CUP2D] - ExperimentDataFish %s %d %f\n", path.c_str(), myFish->Nm, (double)length);
}

void ExperimentFish::updatePosition(Real dt)
{
  Shape::updatePosition(dt);
}

void ExperimentFish::updateVelocity(Real dt)
{
  ExperimentDataFish* const expFish = dynamic_cast<ExperimentDataFish*>( myFish );
  u = expFish->u;
  v = expFish->v;
  omega = expFish->omega;
}

std::vector<std::vector<Real>> ExperimentDataFish::loadFile( const std::string path ) {
  std::vector<std::vector<Real>> data;
  std::ifstream myfile(path);
  if( myfile.is_open() )
  {
    Real temp;
    std::string line;
    while( std::getline(myfile,line) )
    {
      std::vector<Real> lineData;
      std::istringstream readingStream(line);
      while (readingStream >> temp)
        lineData.push_back(temp);
      data.push_back(lineData);
    }
    myfile.close();
  }
  else{
    cout << "[ExperimentFish] Unable to open center of mass file " << path << std::endl;
    fflush(0);
    abort();
  }
  return data;
}

void ExperimentDataFish::computeMidline(const Real t, const Real dt)
{
  // define interpolation points on midline
  const std::array<Real ,6> midlinePoints = { (Real)0, (Real).15*length,
    (Real).4*length, (Real).65*length, (Real).9*length, length
  };
  if( t >= tNext )
  {
    tLast = tNext;
    tNext += dtDataset;
    idxLast = idxNext;
    idxNext++;
    // Only start moving the fish after timeStart has passed
    if( t >= timeStart ) {
      u     = ( centerOfMassData[idxNext][0] - centerOfMassData[idxLast][0] ) / dtDataset;
      v     = ( centerOfMassData[idxNext][1] - centerOfMassData[idxLast][1] ) / dtDataset;
      omega = ( ( centerOfMassData[idxNext][2] - centerOfMassData[idxLast][2] ) / dtDataset ) * ( M_PI / 180 );
    } 
    std::array<Real ,6> lastMidlineValues; 
    std::copy_n(midlineData[idxLast].begin(), 6, lastMidlineValues.begin()); 
    std::array<Real ,6> nextMidlineValues; 
    std::copy_n(midlineData[idxNext].begin(), 6, nextMidlineValues.begin());  
    midlineScheduler.transition(t, tLast, tNext, lastMidlineValues, nextMidlineValues);
  }
  midlineScheduler.gimmeValues(t, midlinePoints, Nm, rS, rY, vY);

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
