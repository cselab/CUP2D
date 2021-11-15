//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Windmill.h"
#include "ShapeLibrary.h"
#include "../Utils/BufferedLogger.h"
#include <cmath>

using namespace cubism;


//WARNING: NO MPI IMPLEMENTED HERE!!!!


void Windmill::create(const std::vector<BlockInfo>& vInfo)
{
  #if 0
  // windmill stuff
  const Real h = sim.getH();
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    //// in the case of the windmill we have 3 ellipses that are not centered at 0

    // center of ellipse 1 wrt to center of windmill,at T=0
    Real center_orig1[2] = {smajax/2, -(smajax/2)*std::tan(M_PI/6)};
    // center of ellipse 1 wrt to origin
    Real center1[2] = {center[0] + std::cos(orientation) * center_orig1[0] - std::sin(orientation)* center_orig1[1], 
                         center[1] + std::sin(orientation) * center_orig1[0] + std::cos(orientation) * center_orig1[1]};

    FillBlocks_Ellipse kernel1(smajax, sminax, h, center1, (orientation + 2*M_PI / 3), rhoS);

    // center of ellipse 2 wrt to center of windmill,at T=0
    Real center_orig2[2] = {0, smajax/(2*std::cos(M_PI/6))};
    // center of ellipse 1 wrt to origin
    Real center2[2] = {center[0] + std::cos(orientation) * center_orig2[0] - std::sin(orientation)* center_orig2[1], 
                         center[1] + std::sin(orientation) * center_orig2[0] + std::cos(orientation) * center_orig2[1]};

    FillBlocks_Ellipse kernel2(smajax, sminax, h, center2, (orientation + M_PI / 3), rhoS);

    // center of ellipse 3 wrt to center of windmill,at T=0
    Real center_orig3[2] = {-smajax/2, -(smajax/2)*std::tan(M_PI/6)};
    // center of ellipse 1 wrt to origin
    Real center3[2] = {center[0] + std::cos(orientation) * center_orig3[0] - std::sin(orientation)* center_orig3[1], 
                         center[1] + std::sin(orientation) * center_orig3[0] + std::cos(orientation) * center_orig3[1]};

    FillBlocks_Ellipse kernel3(smajax, sminax, h, center3, orientation, rhoS);

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
  #endif
}

void Windmill::updateVelocity(Real dt)
{
  Shape::updateVelocity(dt);
}

void Windmill::updatePosition(Real dt)
{
  Shape::updatePosition(dt);

  // compute the energies as well
  Real r_energy = -std::abs(appliedTorque*omega)*sim.dt;
  energy += r_energy;
}

void Windmill::setTarget(std::array<Real, 2> target_pos)
{
  target = target_pos;
}

void Windmill::printVelAtTarget()
{
  if(not sim.muteAll)
  {
    std::stringstream ssF;
    ssF<<sim.path2file<<"/targetvelocity_"<<obstacleID<<".dat";
    std::stringstream & fout = logger.get_stream(ssF.str());

    // compute average
    std::vector<Real>  avg = average(target);
    Real norm = std::sqrt(avg[0]*avg[0] + avg[1]*avg[1]);

    fout<<sim.time<<" "<<norm<<std::endl;
    fout.flush();
  }
}

void Windmill::printRewards(Real r_energy, Real r_flow)
{
  std::stringstream ssF;
  ssF<<sim.path2file<<"/rewards_"<<obstacleID<<".dat";
  std::stringstream & fout = logger.get_stream(ssF.str());

  fout<<sim.time<<" "<<r_energy<<" "<<r_flow<<std::endl;
  fout.flush();
}

void Windmill::printNanRewards(bool is_energy, Real r)
{
  std::ofstream fout("nan.txt");

  if (is_energy)
  {
    fout<<"Energy produces "<<r<<std::endl;
  } else
  {
    fout<<"Flow produces "<<r<<std::endl;
  }
  fout.close();
}

void Windmill::printValues()
{
  std::stringstream ssF;
  ssF<<sim.path2file<<"/values_"<<obstacleID<<".dat";
  std::stringstream & fout = logger.get_stream(ssF.str());

  fout<<sim.time<<" "<<appliedTorque<<" "<<orientation<<" "<<omega<<std::endl;
  fout.flush();
}




void Windmill::act( Real action )
{
  // dimensionful applied torque from dimensionless action, divide by second squared
  // windscale is around 0.15, lengthscale is around 0.0375, so action is multiplied by around 16
  //appliedTorque = action / ( (lengthscale/windscale) * (lengthscale/windscale) );

  appliedTorque = action;
  printValues();
}

Real Windmill::reward(std::array<Real, 2> target_vel, Real C, Real D)
{
  // first reward is opposite of energy given into the system : r_1 = -torque*angVel*dt
  // Real r_energy = -C * std::abs(appliedTorque*omega)*sim.dt;
  // Real r_energy = -C * appliedTorque*appliedTorque*omega*omega*sim.dt;
  // need characteristic energy
  //r_energy /= (lengthscale*windscale);
  Real r_energy = C*energy;
  
  if (std::isfinite(r_energy) == false)
  {
    printNanRewards(true, r_energy);
  }
  // reset for next time steps
  energy = 0;


  // other reward is diff between target and average of area : r_2^t = C/t\sum_0^t (u(x,y,t)-u^*(x,y,t))^2
  // const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  // compute average
  
  std::vector<Real> avg = average(target);
  // compute norm of difference beween target and average velocity
  //printf("Average, X: %g \nAverage, Y: %g \n", avg[0], avg[1]);
  std::vector<Real> diff = {target_vel[0] - avg[0], target_vel[1] - avg[1]};

  Real r_flow = - D * std::sqrt( diff[0]*diff[0] + diff[1]*diff[1] );

  if (std::isfinite(r_flow) == false)
  {
    printNanRewards(false, r_flow);
  }

  //Real r_flow = - D * std::sqrt( (target_vel[0] - avg[0]) * (target_vel[0] - avg[0]) + (target_vel[1] - avg[1]) * (target_vel[1] - avg[1]) );
  //need characteristic speed
  //r_flow /= windscale;

  // Real r_flow = 0;

  printf("Energy_reward: %f \n Flow_reward: %f \n", (double)r_energy, (double)r_flow);

  printRewards(r_energy, r_flow);
  printVelAtTarget();

  return r_energy + r_flow;
  // return r_flow;
  
  // return C*r_energy;
}


std::vector<Real>  Windmill::state()
{
  // intitialize state vector
  std::vector<Real> state(2);

  // angle
  state[0] = orientation;

  state[1] = omega;
  // angular velocity, dimensionless so multiply by seconds
  //state[1] = omega * (lengthscale/windscale);
  

  return state;
}

/* helpers to compute reward */

// average function
//std::vector<Real> Windmill::average(std::array<Real, 2> pSens) const
std::vector<Real> Windmill::average(std::array<Real, 2> pSens) const
{
  #if 0
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

  // get blockId
  const size_t blockId = holdingBlockID(pSens, velInfo);

  // get block
  const auto& sensBinfo = velInfo[blockId];

  // get origin of block
  const std::array<Real,2> oSens = sensBinfo.pos<Real>(0, 0);

  // get inverse gridspacing in block
  const Real invh = 1/(sensBinfo.h_gridpoint);

  // get index for sensor
  const std::array<int,2> iSens = safeIdInBlock(pSens, oSens, invh);

  // stencil for averaging
  static constexpr int stenBeg[3] = {-5,-5, 0}, stenEnd[3] = { 6, 6, 1};

  VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);
  vellab.load(sensBinfo, 0); VectorLab & __restrict__ V = vellab;

  Real avgX=0.0;
  Real avgY=0.0;

  // average velocity in a cube of 11 points per direction around the point of interest (5 on each side)
  for (ssize_t i = -5; i < 6; ++i)
  {
    for (ssize_t j = -5; j < 6; ++j)
    {
      avgX += V(iSens[0] + i, iSens[1] + j).u[0];
      avgY += V(iSens[0] + i, iSens[1] + j).u[1];
    }
  }

  avgX/=121.0;
  avgY/=121.0;
  #endif
  Real avgX = 0;
  Real avgY = 0;
  //return std::vector<Real> {avgX, avgY};
  return std::vector<Real> {avgX, avgY};
}

// function that finds block id of block containing pos (x,y)
size_t Windmill::holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const
{
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
    if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
    {
      // select block
      return i;
    }
  }
  printf("ABORT: coordinate (%g,%g) could not be associated to block\n", pos[0], pos[1]);
  fflush(0); abort();
  return 0;
};

// function that gives indice of point in block
std::array<int, 2> Windmill::safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const
{
  const int indx = (int) std::round((pos[0] - org[0])*invh);
  const int indy = (int) std::round((pos[1] - org[1])*invh);
  const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
  const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
  return std::array<int, 2>{{ix, iy}};
};
