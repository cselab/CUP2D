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

void activeParticle::create(const std::vector<BlockInfo> &vInfo)
{
  checkFeasibility();
  const Real h = vInfo[0].h_gridpoint;
  for (auto &entry : obstacleBlocks)
    delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock *>(vInfo.size(), nullptr);

#pragma omp parallel
  {
    FillBlocks_Cylinder kernel(radius, h, center, rhoS);

#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < vInfo.size(); i++)
      if (kernel.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        ScalarBlock &b = *(ScalarBlock *)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID]);
      }
  }
}

void activeParticle::updatePosition(double dt)
{
  Shape::updatePosition(dt);

  std::ofstream pos;
  pos.open("position.csv", std::ios_base::app);
  pos << sim.time << "," << centerOfMass[0] << "," << centerOfMass[1] << "\n";
  pos.close();
}

void activeParticle::updateVelocity(double dt)
{
  Shape::updateVelocity(dt);

  if(tAccel > 0 && bForcedx && bForcedy && xCenterRotation > 0 && yCenterRotation > 0)
  {
    // Evaluate motion type
    if(sim.time < tAccel)                                                                    motion = 2;
    else if(tStartUACM > 0 && sim.time > tStartUACM && sim.time < tStartUACM + tTransitUACM) motion = 2;
    else if(tStartEM > 0 && sim.time > tStartEM && sim.time < tStartEM + tTransitEM)         motion = 3;
    else                                                                                     motion = 1;

    // Visit relevant equations of motion
    switch (motion)
    {
    case 1:
    {
      if(lastUACM || lastEM) refreshState();
      if (lastEM) forcedOmegaCirc = std::sqrt(mu * ((2 / finalRadiusRotation) - (1 / semimajor_axis))) / finalRadiusRotation;

      u = -forcedRadiusMotion * forcedOmegaCirc * std::sin(forcedOmegaCirc * (sim.time - corrector) + theta_0);
      v =  forcedRadiusMotion * forcedOmegaCirc * std::cos(forcedOmegaCirc * (sim.time - corrector) + theta_0);

      lastUCM = true;
      lastUACM = false;
      lastEM = false;
      break;
    }
    case 2:
    {
      if(lastUCM || lastEM)
        refreshState();

      if(sim.time < tAccel)
      {
        omegaStartup += dt * (forcedOmegaCirc / tAccel);
        u = -forcedRadiusMotion * omegaStartup * std::sin(0.5 * (forcedOmegaCirc / tAccel) * std::pow(sim.time - corrector, 2) + theta_0);
        v = forcedRadiusMotion * omegaStartup * std::cos(0.5 * (forcedOmegaCirc / tAccel) * std::pow(sim.time - corrector, 2) + theta_0);
      }
      else
      {
        if(forcedAccelCirc > 0)
          omegaCirc += dt * accCirc;
        else
          omegaCirc -= dt * accCirc;
        u = -forcedRadiusMotion * omegaCirc * std::sin(0.5 * forcedAccelCirc * std::pow(sim.time - corrector, 2) + forcedOmegaCirc * (sim.time - corrector) + theta_0);
        v = forcedRadiusMotion * omegaCirc * std::cos(0.5 * forcedAccelCirc * std::pow(sim.time - corrector, 2) + forcedOmegaCirc * (sim.time - corrector) + theta_0);
      }

      lastUCM = false;
      lastUACM = true;
      lastEM = false;
      break;
    }
    case 3:
    {
      if(lastUCM || lastUACM) refreshState();
      anomalyGivenTime();

      double orbital_speed_perp = angMom * (1 + eccentricity * std::cos(true_anomaly)) / semilatus_rectum;
      double orbital_speed_radial = angMom * eccentricity * std::sin(true_anomaly) / semilatus_rectum;

      u = orbital_speed_radial * std::cos(true_anomaly + theta_0) - orbital_speed_perp * std::sin(true_anomaly + theta_0);
      v = orbital_speed_radial * std::sin(true_anomaly + theta_0) + orbital_speed_perp * std::cos(true_anomaly + theta_0);

      lastUCM = false;
      lastUACM = false;
      lastEM = true;
      break;
    }
    default:
    {
      std::cout << "Invalid motion type" << std::endl;
      break;
    }
    }
    std::ofstream vel;
    vel.open("velocity.csv", std::ios_base::app);
    vel << sim.time << "," << u << "," << v << ","
        << "\n";
    vel.close();

    reward();
  }
}

double activeParticle::reward()
{
  double x = xCenterRotation;
  double y = yCenterRotation;
  const Real h = sim.getH();
  const Real invH = 0.5 / h;
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();
  size_t i;
  bool foundBlock = false;
  std::array<Real, 2> MIN, MAX;
  std::vector<std::pair<double, int>> distsBlocks(velInfo.size());

  for (i = 0; i < velInfo.size(); ++i)
  {
    MIN = velInfo[i].pos<Real>(0, 0);
    MAX = velInfo[i].pos<Real>(VectorBlock::sizeX - 1, VectorBlock::sizeY - 1);
    for (int j = 0; j < 2; j++)
    {
      MIN[j] -= 0.5 * h;
      MAX[j] += 0.5 * h;
    }
    if (x >= MIN[0] && y >= MIN[1] && x <= MAX[0] && y <= MAX[1])
    {
      const auto &skinBinfo = velInfo[i];
      const auto *const o = obstacleBlocks[skinBinfo.blockID];
      if (o != nullptr)
      {
        foundBlock = true;
        break;
      }
      std::array<Real, 4> WENS;
      WENS[0] = MIN[0] - x;
      WENS[1] = x - MAX[0];
      WENS[2] = MIN[1] - y;
      WENS[3] = y - MAX[1];
      const Real dist = *std::max_element(WENS.begin(), WENS.end());
      distsBlocks[i].first = dist;
      distsBlocks[i].second = i;
      break;
    }
  }
  if (!foundBlock)
  {
    std::sort(distsBlocks.begin(), distsBlocks.end());
    std::reverse(distsBlocks.begin(), distsBlocks.end());
    for (auto distBlock : distsBlocks)
    {
      // handler to select obstacle block
      const auto &skinBinfo = velInfo[distBlock.second];
      const auto *const o = obstacleBlocks[skinBinfo.blockID];
      if (o != nullptr)
        i = (int)distBlock.second;
    }
  }

  static constexpr int stenBeg[3] = {-1, -1, 0}, stenEnd[3] = {2, 2, 1};
  VectorLab velLab;
  velLab.prepare(*(sim.vel), stenBeg, stenEnd, 0);

  velLab.load(velInfo[i], 0);
  const auto &__restrict__ V = velLab;
  double localx = std::floor(2 * invH * (x - MIN[0]));
  double localy = std::floor(2 * invH * (y - MIN[1]));
  double reward = invH * (V(localx, localy - 1).u[0] - V(localx, localy + 1).u[0] + V(localx + 1, localy).u[1] - V(localx - 1, localy).u[1]); // adjusted for COLLOCATED grid

  std::ofstream vort;
  vort.open("vorticity.csv", std::ios_base::app);
  vort << sim.time << "," << reward << "\n";
  vort.close();

  return reward;
}

void activeParticle::checkFeasibility()
{
  if (tStartEM > 0 && tStartUACM > 0)
  {
    if (tStartUACM > tStartEM && tStartUACM < tStartEM + tTransitEM)
    {
      std::cout << "FATAL: insufficient time for transfer!" << std::endl;
      abort();
    }
    if (tStartEM > tStartUACM && tStartEM < tStartUACM + tTransitUACM)
    {
      std::cout << "FATAL: insufficient time for transfer!" << std::endl;
      abort();
    }
  }
  if (tStartUACM > 0 && tStartUACM < tAccel)
  {
    std::cout << "FATAL: allow particle to come to steady state velocity" << std::endl;
    abort();
  }
  if (tStartEM > 0 && tStartEM < tAccel)
  {
    std::cout << "FATAL: allow particle to come to steady state velocity" << std::endl;
    abort();
  }
}

void activeParticle::anomalyGivenTime() // Iterative solver for the Kepler equation E + e*sin(E) = M_e ;
{                                       // Reference : Orbital Mechanics for Engineering Students (Howard D. Curtis) Chap. 3;
  double M_e = M_PI * (sim.time - tStartEM) / tTransitEM;
  double E = M_e < M_PI ? M_e + eccentricity / 2 : M_e - eccentricity / 2;
  double ratio = 1.0;
  double tol = std::pow(10, -3);
  while (std::abs(ratio) > tol)
  {
    ratio = (E - eccentricity * std::sin(E) - M_e) / (1 - eccentricity * std::cos(E));
    E = E - ratio;
  }
  double angle = std::atan(std::sqrt((1 + eccentricity) / (1 - eccentricity)) * std::tan(E / 2));
  double real_angle = angle > 0 ? angle : angle + M_PI;

  true_anomaly = 2 * real_angle; // always between 0 and 180° in our case
}

void activeParticle::refreshState()
{
  // General
  lastPos[0] = centerOfMass[0];
  lastPos[1] = centerOfMass[1];
  lastVel[0] = u;
  lastVel[1] = v;
  corrector = sim.time;
  theta_0 = std::atan2(lastPos[1] - yCenterRotation, lastPos[0] - xCenterRotation);
  forcedRadiusMotion = std::sqrt(std::pow(lastPos[0] - xCenterRotation, 2) + std::pow(lastPos[1] - yCenterRotation, 2));
  forcedOmegaCirc = std::sqrt(std::pow(lastVel[0], 2) + std::pow(lastVel[1], 2))/forcedRadiusMotion;
  omegaCirc = forcedOmegaCirc;

  // Refresh data for UACM
  initialAngRotation = forcedOmegaCirc;
  if (sim.time + 2 * sim.dt < tStartUACM)
    tTransitUACM = std::abs(finalAngRotation - initialAngRotation) / forcedAccelCirc;

  // Refresh data for EM
  initialRadiusRotation = std::sqrt(std::pow(lastPos[0] - xCenterRotation, 2) + std::pow(lastPos[1] - yCenterRotation, 2));
  semimajor_axis = (initialRadiusRotation + finalRadiusRotation) / 2;
  semiminor_axis = std::sqrt(initialRadiusRotation * finalRadiusRotation);
  eccentricity = std::sqrt(1 - std::pow(semiminor_axis / semimajor_axis, 2));
  semilatus_rectum = semimajor_axis * (1 - std::pow(eccentricity, 2));
  mu = (std::pow(lastVel[0], 2) + std::pow(lastVel[1], 2)) / ((2 / initialRadiusRotation) - (1 / semimajor_axis));
  angMom = std::sqrt(semilatus_rectum * mu);
  if (sim.time + 2 * sim.dt < tStartEM)
    tTransitEM = M_PI * std::sqrt(std::pow(semimajor_axis, 3) / mu);
}
