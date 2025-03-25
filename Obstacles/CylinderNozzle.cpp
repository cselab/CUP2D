//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "CylinderNozzle.h"
#include "ShapeLibrary.h"
#include "../Utils/BufferedLogger.h"
using namespace cubism;

//Create the cylinder object and put the chi function on the grid
void CylinderNozzle::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h = sim.getH();
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Cylinder kernel(radius, h, center);

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, * obstacleBlocks[vInfo[i].blockID] );
      }
  }
}

//This is called after every timestep. It will impose the actuator velocities on the cylinder.
void CylinderNozzle::finalize()
{
  //Step 1: Interpolate between previous values (actuators_prev_value) and next values (actuators_next_value)
  //        in time, with a (hardcoded) transition duration of 0.1.
  const Real transition_duration = 0.1;
  for (size_t idx = 0 ; idx < actuators.size(); idx++)
  {
    Real dummy;
    actuatorSchedulers[idx].transition (sim.time,t_change,t_change+transition_duration,actuators_prev_value[idx],actuators_next_value[idx]);
    actuatorSchedulers[idx].gimmeValues(sim.time,actuators[idx],dummy);
  }

  //Step 2: For the current actuator values, loop over all grid points and impose actuation velocities.
  const Real dtheta = 2*M_PI/Nactuators; //Interval between actuators
  const Real Cx = centerOfMass[0]; //Cylinder center x-coordinate
  const Real Cy = centerOfMass[1]; //Cylinder center y-coordinate
  const Real Uact_max = ccoef * pow(u*u + v*v,0.5); //Max actuation velocity as fraction of total cylinder velocity

  //Loop over blocks of the grid
  const auto & vInfo = sim.vel->getBlocksInfo();
  for(size_t i=0; i<vInfo.size(); i++)
  {
    const auto & info = vInfo[i];
    if(obstacleBlocks[info.blockID] == nullptr) continue; //if this block does not contain the cylinder, move to next block

    //Get array with velocities that will be imposed (imposed actuation velocities will be put here)
    UDEFMAT & __restrict__ UDEF = obstacleBlocks[info.blockID]->udef;

    //Loop over grid points of each block
    for(int iy=0; iy<ScalarBlock::sizeY; iy++)
    for(int ix=0; ix<ScalarBlock::sizeX; ix++)
    {
      //i. Set imposed velocities to zero.
      UDEF[iy][ix][0] = 0.0;
      UDEF[iy][ix][1] = 0.0;

      //ii. Get position (x,y) of the current grid point and make cylinder center the origin of the coordinate system used.
      Real p[2];
      info.pos(p, ix, iy);
      const Real x = p[0]-Cx;
      const Real y = p[1]-Cy;

      //iii. If the current point is more than 2h (h:local grid spacing) distance away from the cylinder surface, do nothing and check next point.
      const Real r = x*x+y*y;
      if (r > (radius+2*info.h)*(radius+2*info.h) || r < (radius-2*info.h)*(radius-2*info.h)) continue;

      //iv. Get polar angle of current point.
      Real theta = atan2(y,x);
      if (theta < 0) theta += 2.*M_PI;

      //v. Find the index of the actuator that is closest to current point
      int idx = round(theta / dtheta); //this is the closest actuator
      if (idx == Nactuators) idx = 0;  //periodic around the cylinder

      //vi. If current point lies with the circular arc of the closest actuator, impose actuation velocity
      const Real theta0 = idx * dtheta; //center of closest actuator
      const Real phi = theta - theta0; //distance (angle) of current point from center of closest actuator
      if ( std::fabs(phi) < 0.5*actuator_theta || (idx == 0 && std::fabs(phi-2*M_PI) < 0.5*actuator_theta))
      {
	//point is within the actuator, impose x- and y-components of actuation velocity
        const Real rr = radius / pow(r,0.5);
        const Real ur = Uact_max*rr*actuators[idx]*cos(M_PI*phi/actuator_theta); //actuation velocity magnitude
        UDEF[iy][ix][0] = ur * cos(theta); //x-component, projected using cylinder normal vector n = [cos(theta),sin(theta)]
        UDEF[iy][ix][1] = ur * sin(theta); //y-component, projected using cylinder normal vector n = [cos(theta),sin(theta)]
      }
    }
  }

  //Step 3: keep track of the x-force integral
  const double cd = forcex / (0.5*u*u*2*radius);
  fx_integral += -std::fabs(cd)*sim.dt;
}

//Called whenever the actuation strengths need to be changed.
void CylinderNozzle::act( std::vector<Real> action, const int agentID)
{
  t_change = sim.time; //keep track of the current time, which is when the smooth transition from current to new values will start.

  //check for invalid inputs
  if(action.size() != actuators.size())
  {
    std::cerr << "action size needs to be equal to actuators\n";
    fflush(0);
    abort();
  }

  //The actuation velocities need to be bounded in [-ccoef*U,ccoef*U], where U:cylinder velocity
  //and they need to have a total mass flux of zero (i.e.: surface integral of actuation velocity = 0)
  //These conditions are equivalent to having actions that are bounded in [-1,+1] and
  //that have a sum of zero; this is iteratively imposed in the following loop:
  bool bounded = false;
  while (bounded == false)
  {
      bounded = true;
      //i. Compute current sum of actions
      Real Q = 0;
      for (size_t i = 0 ; i < action.size() ; i ++)
      {
           Q += action[i];
      }
      Q /= action.size();

      //ii. Substract the computed sum (mean) from all actions and then force them to be in [-1,+1]
      //    Keep doing this until convergence.
      for (size_t i = 0 ; i < action.size() ; i ++)
      {
           action[i] -= Q;
           if (std::fabs(action[i]) > 1.0) bounded = false;
           action[i] = std::max(action[i],-1.0);
           action[i] = std::min(action[i],+1.0);
      }
  }

  // Now that the actions have zero mean and are bounded, set the previous and the next actuation values
  // that will be used for the smooth transition (in 'finalize').
  for (size_t i = 0 ; i < action.size() ; i ++)
  {
    actuators_prev_value[i] = actuators[i];
    actuators_next_value[i] = action   [i];
  }
}

//Computes the reward (for Reinforcement Learning) between actions
Real CylinderNozzle::reward(const int agentID)
{
  Real retval = fx_integral / 0.1; //0.1 is the action times
  fx_integral = 0;
  Real regularizer_sum = 0.0;
  for (size_t idx = 0 ; idx < actuators.size(); idx++)
  {
    regularizer_sum += actuators[idx]*actuators[idx];
  }
  regularizer_sum = pow(regularizer_sum,0.5)/actuators.size(); //O(1)
  const double c = -regularizer;
  return retval + c*regularizer_sum;
}

//Computes the current state (for Reinforcement Learning)
std::vector<Real> CylinderNozzle::state(const int agentID)
{
  std::vector<Real> S;
  const int bins = 16;
  const Real bins_theta = 10*M_PI/180.0;
  const Real dtheta = 2.*M_PI / bins;
  std::vector<int>   n_s (bins,0.0);
  std::vector<Real>  p_s (bins,0.0);
  std::vector<Real>  o_s (bins,0.0);

  //Loop over all blocks that contain part of the cylinder
  for(auto & block : obstacleBlocks) if(block not_eq nullptr)
  {
    for(size_t i=0; i<block->n_surfPoints; i++)
    {
      const Real x = block->x_s[i] - origC[0];
      const Real y = block->y_s[i] - origC[1];
      Real theta = atan2(y,x);
      if (theta < 0) theta += 2.*M_PI;
      int idx = round(theta / dtheta); //this is the closest actuator
      if (idx == bins) idx = 0;  //periodic around the cylinder
      const Real theta0 = idx * dtheta;
      const Real phi = theta - theta0;

      //add up each point's pressure (p) and vorticity (o) values
      if ( std::fabs(phi) < 0.5*bins_theta || (idx == 0 && std::fabs(phi-2*M_PI) < 0.5*bins_theta))
      {
        const Real p = block->p_s[i];
        const Real o = block->omega_s[i];
        n_s[idx] ++;
        p_s[idx] += p;
        o_s[idx] += o;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,n_s.data(),n_s.size(),MPI_INT ,MPI_SUM,sim.comm);
  for (int idx = 0 ; idx < bins; idx++)
  {
    if (n_s[idx] == 0) continue;
    p_s[idx] /= n_s[idx];
    o_s[idx] /= n_s[idx];
  }

  for (int idx = 0 ; idx < bins; idx++) S.push_back(p_s[idx]);
  for (int idx = 0 ; idx < bins; idx++) S.push_back(o_s[idx]);
  MPI_Allreduce(MPI_IN_PLACE,  S.data(),  S.size(),MPI_Real,MPI_SUM,sim.comm);

  //add the forces, the Reynolds number and ccoef to the state
  S.push_back(forcex);
  S.push_back(forcey);
  const Real Re = std::fabs(u)*(2*radius)/sim.nu;
  S.push_back(Re);
  S.push_back(ccoef);

  if (sim.rank ==0 )
    for (size_t i = 0 ; i < S.size() ; i++) std::cout << S[i] << " ";

  return S;
}
