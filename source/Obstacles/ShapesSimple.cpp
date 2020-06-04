//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "ShapeLibrary.h"
#include "ShapesSimple.h"


using namespace cubism;

void Disk::create(const std::vector<BlockInfo>& vInfo)
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
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, *obstacleBlocks[vInfo[i].blockID] );
      }
  }
}

void Disk::updatePosition(double dt)
{
  Shape::updatePosition(dt);
  if(xCenterRotation > 0 && yCenterRotation > 0){
    double radiusForcedMotion = std::sqrt(std::pow(center[0] - xCenterRotation, 2) + std::pow(center[1] -yCenterRotation, 2));
    double theta_0 = std::atan2(y0 - yCenterRotation, x0 - xCenterRotation);
      if(omegaCirc == 0.0) omegaCirc = linCirc/radiusForcedMotion;
    centerOfMass[0] = xCenterRotation + radiusForcedMotion * std::cos(omegaCirc*sim.time + theta_0);
    centerOfMass[1] = yCenterRotation + radiusForcedMotion * std::sin(omegaCirc*sim.time + theta_0);

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

    if(sim.dt <= 0) return;

    if(sim.verbose)
    printf("CM:[%.02f %.02f] C:[%.02f %.02f] ang:%.02f u:%.05f v:%.05f av:%.03f"
      " M:%.02e J:%.02e\n", cx, cy, center[0], center[1], angle, u, v, omega, M, J);
  }
}

void Disk::updateVelocity(double dt)
{
  #if 0
  {
    std::cout << "Checking against potential flow solution." << std::endl;
    const std::vector<BlockInfo>& velInfo = sim.vel->getBlocksInfo();
    const std::vector<BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
    const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
    const size_t Nblocks = tmpInfo.size();
    const Real uInfX = -forcedu, uInfY = forcedv;
    static constexpr double EPS = std::numeric_limits<double>::epsilon();
    const Real velScaleSq = std::max(forcedu*forcedu +forcedv*forcedv, EPS);
    printf("uinf %f %f \n", uInfX, uInfY);
    #pragma omp parallel for schedule(dynamic, 1)
    for(size_t i=0; i<Nblocks; i++)
    {
      const VectorBlock& __restrict__ VEL = *(VectorBlock*)velInfo[i].ptrBlock;
      VectorBlock& __restrict__ VTMP = *(VectorBlock*)tmpVInfo[i].ptrBlock;
      ScalarBlock& __restrict__ ERR = *(ScalarBlock*)tmpInfo[i].ptrBlock;
      const auto pos = obstacleBlocks[velInfo[i].blockID];
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        double p[2]; velInfo[i].pos(p, ix, iy);
        p[0] -= center[0]; p[1] -= center[1];
        const Real CHI = pos == nullptr? 0 : pos->chi[iy][ix];
        if(CHI > 0.9) { ERR(ix,iy).s = 0; continue; }
        const Real Rsq = p[0]*p[0] + p[1]*p[1], invRstar = radius*radius/Rsq;
        const Real theta = std::atan2(p[1], p[0]);
        const Real cosTh = std::cos(theta), sinTh = std::sin(theta);
        const Real VRtgt =  uInfX * (1-invRstar) * cosTh;
        const Real VTtgt = -uInfX * (1+invRstar) * sinTh;
        const Real VXtgt = cosTh*VRtgt - sinTh*VTtgt;
        const Real VYtgt = sinTh*VRtgt + cosTh*VTtgt;
        const Real VX = VEL(ix,iy).u[0] + uInfX, VY = VEL(ix,iy).u[1] + uInfY;
        const Real VXF = VX / (1 - CHI), VYF = VY / (1 - CHI);
        const Real ERRVX = std::pow(VXtgt-VXF,2) / velScaleSq;
        const Real ERRVY = std::pow(VYtgt-VYF,2) / velScaleSq;
        VTMP(ix,iy).u[0] = VXtgt;
        VTMP(ix,iy).u[1] = VYtgt;
        ERR(ix,iy).s = std::sqrt(ERRVX + ERRVY);
      }
    }
    sim.dumpTmp("PotFlowL2Error");
    sim.dumpTmpV("PotFlowTarget");
  }
  #endif


  Shape::updateVelocity(dt);
  if(tAccel > 0) {
    // Uniform linear motion
    if(bForcedx && sim.time < tAccel && xCenterRotation < 0 && yCenterRotation < 0) u = (sim.time/tAccel)*forcedu;
    if(bForcedy && sim.time < tAccel && xCenterRotation < 0 && yCenterRotation < 0) v = (sim.time/tAccel)*forcedv;

    // Uniform circular motion
    if(bForcedx && xCenterRotation > 0 && yCenterRotation > 0) {
      double accelCoef = sim.time<tAccel ? sim.time/tAccel : 1;
      double radiusForcedMotion = std::sqrt(std::pow(center[0] - xCenterRotation, 2) + std::pow(center[1] - yCenterRotation, 2));
      double theta_0 = std::atan2(y0 - yCenterRotation, x0 - xCenterRotation);
        if(omegaCirc == 0.0) omegaCirc = linCirc/radiusForcedMotion;
      u = accelCoef * (- radiusForcedMotion*omegaCirc*std::sin(omegaCirc*sim.time + theta_0));
    }
    if(bForcedy && xCenterRotation > 0 && yCenterRotation > 0) {
      double accelCoef = sim.time<tAccel ? sim.time/tAccel : 1;
      double radiusForcedMotion = std::sqrt(std::pow(center[0] - xCenterRotation, 2) + std::pow(center[1] - yCenterRotation, 2));
      double theta_0 = std::atan2(y0 - yCenterRotation, x0 - xCenterRotation);
        if(omegaCirc == 0.0) omegaCirc = linCirc/radiusForcedMotion;
      v = accelCoef * (  radiusForcedMotion*omegaCirc*std::cos(omegaCirc*sim.time + theta_0));

      std::cout << "time: " << sim.time << std::endl;
      std::cout << "maxU: " << sim.uMax_measured << std::endl;
      std::cout << "Velocity magnitude: " << std::sqrt(std::pow(u, 2) + std::pow(v, 2)) << std::endl;
    }
  }
}


void HalfDisk::updateVelocity(double dt)
{
  Shape::updateVelocity(dt);
  if(tAccel > 0) {
    if(bForcedx && sim.time < tAccel) u = (sim.time/tAccel)*forcedu;
    if(bForcedy && sim.time < tAccel) v = (sim.time/tAccel)*forcedv;
  }
}

void HalfDisk::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
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

void Ellipse::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
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

void DiskVarDensity::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_VarRhoCylinder kernel(radius, h, center, rhoTop, rhoBot, orientation);

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

void EllipseVarDensity::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_VarRhoEllipse kernel(semiAxisX, semiAxisY, h, center, orientation, rhoTop, rhoBot);

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
