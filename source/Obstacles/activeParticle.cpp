//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "ShapeLibrary.h"
#include "activeParticle.h"
#include <iostream>
#include <fstream>

using namespace cubism;

void activeParticle::checkFeasibility()
{
  if(tStartElliTransfer > 0 && tStartAccelTransfer > 0){
    if(tStartElliTransfer < tStartAccelTransfer && tStartElliTransfer + tTransitElli > tStartAccelTransfer) {
      std::cout << "FATAL: insufficient time for transfer!" << std::endl; 
      abort();
    }
    if(tStartAccelTransfer < tStartElliTransfer && tStartAccelTransfer + tTransitAccel > tStartElliTransfer) {
      std::cout << "FATAL: insufficient time for transfer!" << std::endl; 
      abort();}
  }
}

void activeParticle::create(const std::vector<BlockInfo>& vInfo)
{ 
  checkFeasibility();

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
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID] );
      }
  }
}

void activeParticle::updatePosition(double dt)
{
  Shape::updatePosition(dt);
    //Uniform circular motion
    if(bForcedx && bForcedy && xCenterRotation > 0 && yCenterRotation > 0){
      if(sim.time < tStartAccelTransfer || sim.time > tStartAccelTransfer + tTransitAccel || tStartAccelTransfer < 0){
        if(sim.time < tStartElliTransfer || sim.time > tStartElliTransfer + tTransitElli || tStartElliTransfer < 0){
          if(lastUACM || lastElli) lastPos[0] = centerOfMass[0], lastPos[1] = centerOfMass[1], forcedOmegaCirc = omegaCirc, std::cout << lastUACM << lastElli << std::endl;
  
          double forcedRadiusMotion = std::sqrt(std::pow(lastPos[0] - xCenterRotation, 2) + std::pow(lastPos[1] - yCenterRotation, 2));
          double theta_0 = std::atan2(lastPos[1] - yCenterRotation, lastPos[0] - xCenterRotation);
  
          centerOfMass[0] = xCenterRotation + forcedRadiusMotion * std::cos(forcedOmegaCirc*sim.time + theta_0);
          centerOfMass[1] = yCenterRotation + forcedRadiusMotion * std::sin(forcedOmegaCirc*sim.time + theta_0);
    
          lastUCM = true;
          lastUACM = false;
          lastElli = false;
  
          std::cout << "UCM Position" << std::endl;
          std::cout << "tTransitElli = " << tTransitElli << std::endl;
          std::cout << "lastPos = (" << lastPos[0] << "," << lastPos[1] << ")" << std::endl;
          std::cout << "theta_0 = " << theta_0 << std::endl;
          std::cout << "forcedRadiusMotion = " << forcedRadiusMotion << std::endl;
          std::cout << "angCircVel = " << forcedAccelCirc << std::endl;
          std::cout << "forcedAccelCirc = " << forcedAccelCirc << std::endl;
  
          std::ofstream transit;
          transit.open ("transit.csv", std::ios_base::app);
          transit << sim.time << "," << lastPos[0] <<  "," << lastPos[1] << "," << theta_0 << "," << forcedRadiusMotion << "\n";
          transit.close();
        }
      }
    }
    // Uniformly accelerated circular motion
    if(bForcedx && bForcedy && xCenterRotation > 0 && yCenterRotation > 0 && tStartAccelTransfer > 0){
      if(sim.time > tStartAccelTransfer && sim.time < tStartAccelTransfer + tTransitAccel){
      if(lastUCM || lastElli) lastPos[0] = centerOfMass[0], lastPos[1] = centerOfMass[1], forcedOmegaCirc = omegaCirc;

      double forcedRadiusMotion = std::sqrt(std::pow(lastPos[0] - xCenterRotation, 2) + std::pow(lastPos[1] - yCenterRotation, 2));
      double theta_0 = std::atan2(lastPos[1] - yCenterRotation, lastPos[0] - xCenterRotation);  
      
      centerOfMass[0] = xCenterRotation + forcedRadiusMotion * std::cos(0.5*forcedAccelCirc*std::pow(sim.time, 2) + forcedOmegaCirc*sim.time + theta_0);
      centerOfMass[1] = yCenterRotation + forcedRadiusMotion * std::sin(0.5*forcedAccelCirc*std::pow(sim.time, 2) + forcedOmegaCirc*sim.time + theta_0);

      lastUCM = false;
      lastUACM = true;
      lastElli = false;

      std::cout << " " << std::endl;
      std::cout << "UACM Position" << std::endl; 
      std::cout << "LastPos = (" << lastPos[0] << "," << lastPos[1] << ")" << std::endl;
      std::cout << "forcedRadiusMotion = " << forcedRadiusMotion << std::endl;
      }
    }

    // Elliptical motion
    if(bForcedx && bForcedy && xCenterRotation > 0 && yCenterRotation > 0 && tStartElliTransfer > 0){
      if(sim.time > tStartElliTransfer && sim.time < tStartElliTransfer + tTransitElli){
      if(lastUCM || lastUACM) lastPos[0] = centerOfMass[0], lastPos[1] = centerOfMass[1];
      anomalyGivenTime();
      double radiusEllipse = semilatus_rectum/(1+eccentricity*std::cos(true_anomaly));
      double theta_0 = std::atan2(lastPos[1] - yCenterRotation, lastPos[0] - xCenterRotation);  

      centerOfMass[0] = xCenterRotation + radiusEllipse*std::cos(true_anomaly + theta_0);
      centerOfMass[1] = yCenterRotation + radiusEllipse*std::sin(true_anomaly + theta_0); 
  
      lastUCM = false;
      lastUACM = false;
      lastElli = true;;

      std::cout << " " << std::endl;
      std::cout << "Ellipse Position" << std::endl;
      std::cout << "a = " << semimajor_axis << std::endl;
      std::cout << "b = " << semiminor_axis << std::endl;
      std::cout << "e = " << eccentricity << std::endl;
      std::cout << "p = " << semilatus_rectum << std::endl;
      std::cout << "radiusEllipse = " << radiusEllipse << std::endl;
      std::cout << "trueAnomaly = " << true_anomaly*57.3 << std::endl;

      std::ofstream ell;
      ell.open ("ellipsePos.csv", std::ios_base::app);
      ell << sim.time << "," << radiusEllipse <<  "," << true_anomaly << "\n";
      ell.close();
      }
    }
  
    // To be adjusted for bFixed=1
      labCenterOfMass[0] += dt * u;
      labCenterOfMass[1] += dt * v;
  
      orientation += dt*omega;
      orientation = orientation> M_PI ? orientation-2*M_PI : orientation;
      orientation = orientation<-M_PI ? orientation+2*M_PI : orientation;
  
      const double cosang = std::cos(orientation), sinang = std::sin(orientation);
  
      center[0] = centerOfMass[0] + cosang*d_gm[0] - sinang*d_gm[1];
      center[1] = centerOfMass[1] + sinang*d_gm[0] + cosang*d_gm[1];
  
      const Real CX = labCenterOfMass[0], CY = labCenterOfMass[1], t = sim.time;
      const Real cx = centerOfMass[0], cy = centerOfMass[1], angle = orientation;
}  

void activeParticle::updateVelocity(double dt)
{
  Shape::updateVelocity(dt);
  if(tAccel > 0) {

    // Uniform circular motion
    if(bForcedx && bForcedy && xCenterRotation > 0 && yCenterRotation > 0){
      if(sim.time < tStartAccelTransfer || sim.time > tStartAccelTransfer + tTransitAccel || tStartAccelTransfer < 0){
        if(sim.time < tStartElliTransfer || sim.time > tStartElliTransfer + tTransitElli || tStartElliTransfer < 0){
          if(lastUACM || lastElli) lastPos[0] = centerOfMass[0], lastPos[1] = centerOfMass[1], forcedOmegaCirc = omegaCirc;
          
          if(sim.time < tAccel) accelCoef = sim.time/tAccel;
          else if(sim.time > tStartAccelTransfer + tTransitAccel && sim.time - (tStartAccelTransfer + tTransitAccel) < tAccel) accelCoef = (sim.time - (tStartAccelTransfer + tTransitAccel))/tAccel; 
          else if(sim.time > tStartElliTransfer + tTransitElli && sim.time - (tStartElliTransfer + tTransitElli) < tAccel) accelCoef = (sim.time - (tStartElliTransfer + tTransitElli))/tAccel;
          else accelCoef = 1;

          double forcedRadiusMotion = std::sqrt(std::pow(lastPos[0] - xCenterRotation, 2) + std::pow(lastPos[1] - yCenterRotation, 2));
          double theta_0 = std::atan2(lastPos[1] - yCenterRotation, lastPos[0] - xCenterRotation);

          u = accelCoef * (- forcedRadiusMotion*forcedOmegaCirc*std::sin(forcedOmegaCirc*sim.time + theta_0));
          v = accelCoef * (  forcedRadiusMotion*forcedOmegaCirc*std::cos(forcedOmegaCirc*sim.time + theta_0));
          
          lastUCM = true;
          lastUACM = false;
          lastElli = false;
          
          std::cout << " " << std::endl;
          std::cout << "UCM Velocity" << std::endl;
          std::cout << "tStartElliTransfer = " << tStartElliTransfer << std::endl;
          std::cout << "tTransitElli = " << tTransitElli << std::endl;
          std::cout << "finalRadiusRotation = " << finalRadiusRotation << std::endl;
          std::cout << "tStartAccelTransfer = " << tStartAccelTransfer << std::endl;
          std::cout << "tTransitAccel = " << tTransitAccel << std::endl;
          std::cout << "finalAngRotation = " << finalAngRotation << std::endl;
  
          std::cout << "Linear velocity norm = " << std::sqrt(std::pow(u, 2) + std::pow(v, 2)) << std::endl;
          std::cout << "Angular velocity = " << forcedOmegaCirc << std::endl;
          std::cout << "tAccel = " << tAccel << std::endl;
          std::cout << "AccelCoef = " << accelCoef << std::endl;

        }
      }
    }

    // Uniformly accelerated circular motion
    if(bForcedx && bForcedy && xCenterRotation > 0 && yCenterRotation > 0 && tStartAccelTransfer > 0){
      if(sim.time > tStartAccelTransfer && sim.time < tStartAccelTransfer + tTransitAccel){
        if(lastUCM || lastElli) lastPos[0] = centerOfMass[0], lastPos[1] = centerOfMass[1], forcedOmegaCirc = omegaCirc;

        accelCoef = sim.time > tStartAccelTransfer && sim.time - tStartAccelTransfer < tAccel ? (sim.time - tStartAccelTransfer)/tAccel : 1;

        double forcedRadiusMotion = std::sqrt(std::pow(lastPos[0] - xCenterRotation, 2) + std::pow(lastPos[1] - yCenterRotation, 2));
        double theta_0 = std::atan2(lastPos[1] - yCenterRotation, lastPos[0] - xCenterRotation);
        
        omegaCirc += dt*accCirc;
        u = accelCoef * (- forcedRadiusMotion*omegaCirc*std::sin(0.5*forcedAccelCirc*std::pow(sim.time, 2) + forcedOmegaCirc*sim.time + theta_0));
        v = accelCoef * (  forcedRadiusMotion*omegaCirc*std::cos(0.5*forcedAccelCirc*std::pow(sim.time, 2) + forcedOmegaCirc*sim.time + theta_0));
        
        lastUCM = false;
        lastUACM = true;
        lastElli = false;
  
        std::cout << " " << std::endl;
        std::cout << "UACM Velocity" << accelCoef << std::endl;

      }
    }
    
    // Elliptical motion
    if(bForcedx && bForcedy && xCenterRotation > 0 && yCenterRotation > 0 && tStartElliTransfer > 0){
      if(sim.time > tStartElliTransfer && sim.time < tStartElliTransfer + tTransitElli){
        if(lastUCM || lastUACM) lastPos[0] = centerOfMass[0], lastPos[1] = centerOfMass[1], forcedOmegaCirc = omegaCirc;

        accelCoef = sim.time > tStartElliTransfer && sim.time - tStartElliTransfer < tAccel ? (sim.time - tStartElliTransfer)/tAccel : 1;
        double theta_0 = std::atan2(lastPos[1] - yCenterRotation, lastPos[0] - xCenterRotation);

        double radiusEllipse = std::sqrt(std::pow(center[0] - xCenterRotation, 2) + std::pow(center[1] - yCenterRotation, 2));
        double orbital_speed = std::sqrt(mu*(2/radiusEllipse - 1/semimajor_axis));
        double orbital_speed_perp = angMom*(1+eccentricity*std::cos(true_anomaly))/semilatus_rectum;
        double orbital_speed_radial = angMom*eccentricity*std::sin(true_anomaly)/semilatus_rectum;
        double flight_path_angle = std::atan2(orbital_speed_radial, orbital_speed_perp); //should be positive at all times in our case
        
        u = accelCoef * (orbital_speed_radial*std::cos(true_anomaly) - orbital_speed_perp*std::sin(true_anomaly + theta_0));
        v = accelCoef * (orbital_speed_radial*std::sin(true_anomaly) + orbital_speed_perp*std::cos(true_anomaly + theta_0));
  
        lastUCM = false;
        lastUACM = false;
        lastElli = true;
  
        std::ofstream ellVel;
        ellVel.open ("ellipseVel.csv", std::ios_base::app);
        ellVel << sim.time << "," << std::sqrt(std::pow(u, 2) + std::pow(v, 2)) << std::endl << "," << orbital_speed <<  "," << orbital_speed_radial <<  "," << orbital_speed_perp <<  "," << flight_path_angle << "\n";
        ellVel.close();

        std::cout << " " << std::endl;
        std::cout << "Ellipse Velocity" << std::endl;
        

        std::cout << "gamma = " << flight_path_angle*57.3 << std::endl;
        std::cout << "mu = " << mu << std::endl;
        std::cout << "tStartElliTransfer = " << tStartElliTransfer << std::endl;
        std::cout << "tTransitElli = " << tTransitElli << std::endl;
        std::cout << "initialRadius = " << initialRadiusRotation << std::endl;
        std::cout << "finalRadius = " << finalRadiusRotation << std::endl; 
    
        std::cout << "orbital_speed_(norm (u,v)) = " << std::sqrt(std::pow(u, 2) + std::pow(v, 2)) << std::endl;
        std::cout << "orbital_speed_(orbital_formula) = " << orbital_speed << std::endl;
        std::cout << "orbital_speed_(norm(radial, perp)) = " << std::sqrt(std::pow(orbital_speed_perp, 2) + std::pow(orbital_speed_radial, 2)) << std::endl;
        std::cout << "orbital_radial_speed = " << orbital_speed_radial << std::endl;
        std::cout << "orbital_perp_speed = " << orbital_speed_perp << std::endl;
        std::cout << "AccelCoef = " << accelCoef << std::endl;
      }
    }  

  }
}

void activeParticle::anomalyGivenTime() // Iterative solver for the Kepler equation E + e*sin(E) = M_e ;  
{                                       // Ref: Orbital Mechanics for Engineering Students (H.D. Curtis) Chap.3
  double M_e = M_PI*(sim.time - tStartElliTransfer)/tTransitElli;
  double E = M_e < M_PI ? M_e + eccentricity/2 : M_e - eccentricity/2;
  double ratio = 1.0; double tol = std::pow(10, -3);
  while(std::abs(ratio) > tol){
    ratio = (E - eccentricity*std::sin(E) - M_e)/(1 - eccentricity*std::cos(E));
    E = E - ratio;
  }
  double angle = std::atan(std::sqrt((1+eccentricity)/(1-eccentricity))*std::tan(E/2));
  double real_angle = angle > 0 ? angle : angle + M_PI;

  true_anomaly = 2*real_angle; // always between 0 and 180° in our case
}

void computeVorticityCollocated::run() const
{
  const Real invH = 0.5 / sim.getH();
  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  #pragma omp parallel
  {
    static constexpr int stenBeg [3] = {-1,-1, 0}, stenEnd [3] = { 2, 2, 1};
    VectorLab velLab;   velLab.prepare(*(sim.vel), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      velLab.load( velInfo[i], 0); const auto & __restrict__ V   = velLab;
      auto& __restrict__ O = *(ScalarBlock*)  tmpInfo[i].ptrBlock;

      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
      O(x,y).s = invH * (V(x,y-1).u[0]-V(x,y+1).u[0] + V(x+1,y).u[1]-V(x-1,y).u[1]); // adjusted for COLLOCATED grid
    }
  }
}

/*
std::vector<double> activeParticle::getLastPos(double lastUCMVisit, double lastUACMVisit, double lastElliVisit)
{
  std::vector<double> lastPos{x0, y0};
  double mostRecentTime = std::max({lastUCMVisit, lastUACMVisit, lastElliVisit});
  
  if(mostRecentTime == 0.0) return lastPos;
  if(mostRecentTime == lastUCMVisit)  lastPos[0] = lastUCMPosX,  lastPos[1] = lastUCMPosY;
  if(mostRecentTime == lastUACMVisit) lastPos[0] = lastUACMPosX, lastPos[1] = lastUACMPosY;
  if(mostRecentTime == lastElliVisit) lastPos[0] = lastElliPosX, lastPos[1] = lastElliPosY;

  return lastPos;
}
*/

void activeParticle::reward(){
      // get block ID
      // grab vorticity from block id with specified coordinates
}

/*
void activeParticle::getBlockID()
{
// function that finds block id of block containing pos (x,y)
const Real h = sim.getH(), invh = 1/h;
const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

const auto holdingBlockID = [&](const Real x, const Real y)
{
  const auto getMin = [&]( const BlockInfo&I )
  {
    std::array<Real,2> MIN = I.pos<Real>(0, 0);
    for(int i=0; i<2; ++i)
      MIN[i] -= 0.5 * h; // pos returns cell centers
    return MIN;
  };

  const auto getMax = [&]( const BlockInfo&I )
  {
    std::array<Real,2> MAX = I.pos<Real>(VectorBlock::sizeX-1,
                                          VectorBlock::sizeY-1);
    for(int i=0; i<2; ++i)
      MAX[i] += 0.5 * h; // pos returns cell centers
    return MAX;
  };

  const auto holdsPoint = [&](const std::array<Real,2> MIN, std::array<Real,2> MAX,
                              const Real X,const Real Y)
  {
    // this may return true for 2 blocks if (X,Y) overlaps with edges
    return X >= MIN[0] && Y >= MIN[1] && X <= MAX[0] && Y <= MAX[1];
  };

  std::vector<std::pair<double, int>> distsBlocks(velInfo.size());
  for(size_t i=0; i<velInfo.size(); ++i)
  {
    std::array<Real,2> MIN = getMin(velInfo[i]);
    std::array<Real,2> MAX = getMax(velInfo[i]);
    if( holdsPoint(MIN, MAX, x, y) )
    {
    // handler to select obstacle block
      const auto& skinBinfo = velInfo[i];
      const auto *const o = obstacleBlocks[skinBinfo.blockID];
      if(o != nullptr ) return (int) i;
    }
    std::array<Real, 4> WENS;
    WENS[0] = MIN[0] - x;
    WENS[1] = x - MAX[0];
    WENS[2] = MIN[1] - y;
    WENS[3] = y - MAX[1];
    const Real dist = *std::max_element(WENS.begin(),WENS.end());
    distsBlocks[i].first = dist;
    distsBlocks[i].second = i;
  }
  std::sort(distsBlocks.begin(), distsBlocks.end());
  std::reverse(distsBlocks.begin(), distsBlocks.end());
  for( auto distBlock: distsBlocks )
  {
    // handler to select obstacle block
      const auto& skinBinfo = velInfo[distBlock.second];
      const auto *const o = obstacleBlocks[skinBinfo.blockID];
      if(o != nullptr ) return (int) distBlock.second;
  }
  printf("ABORT: coordinate could not be associated to obstacle block\n");
  fflush(0); abort();
  return (int) 0;
};
}
*/
