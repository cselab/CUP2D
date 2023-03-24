//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "CylinderNozzle.h"
#include "ShapeLibrary.h"
#include "../Utils/BufferedLogger.h"
using namespace cubism;

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

void CylinderNozzle::finalize()
{
  const Real transition_duration = 0.1;
  for (size_t idx = 0 ; idx < actuators.size(); idx++)
  {
    Real dummy;
    actuatorSchedulers[idx].transition (sim.time,t_change,t_change+transition_duration,actuators_prev_value[idx],actuators_next_value[idx]);
    actuatorSchedulers[idx].gimmeValues(sim.time,actuators[idx],dummy);
  }

  //if (sim.time >= t_action_taken[curr_idx])
  //{
  //  std::vector<Real> a (actuators.size());
  //  for (size_t i = 0 ; i < actuators.size(); i++)
  //    a[i] = action_taken[curr_idx*actuators.size()+i];
  //  act(a,0);
  //  curr_idx++;
  //}

  const auto & vInfo = sim.vel->getBlocksInfo();
  const Real dtheta = 2*M_PI/Nactuators;
  const Real Cx = centerOfMass[0];
  const Real Cy = centerOfMass[1];
  for(size_t i=0; i<vInfo.size(); i++)
  {
    const auto & info = vInfo[i];
    if(obstacleBlocks[info.blockID] == nullptr) continue; //obst not in block
    ObstacleBlock& o = * obstacleBlocks[info.blockID];
    UDEFMAT & __restrict__ UDEF = o.udef;

    for(int iy=0; iy<ScalarBlock::sizeY; iy++)
    for(int ix=0; ix<ScalarBlock::sizeX; ix++)
    {
      UDEF[iy][ix][0] = 0.0;
      UDEF[iy][ix][1] = 0.0;
      Real p[2];
      info.pos(p, ix, iy);
      const Real x = p[0]-Cx;
      const Real y = p[1]-Cy;
      const Real r = x*x+y*y;

      if (r > (radius+2*info.h)*(radius+2*info.h) || r < (radius-2*info.h)*(radius-2*info.h)) continue;

      Real theta = atan2(y,x);
      if (theta < 0) theta += 2.*M_PI;

      int idx = round(theta / dtheta); //this is the closest actuator
      if (idx == Nactuators) idx = 0;  //periodic around the cylinder

      const Real theta0 = idx * dtheta;

      const Real phi = theta - theta0;
      if ( std::fabs(phi) < 0.5*actuator_theta || (idx == 0 && std::fabs(phi-2*M_PI) < 0.5*actuator_theta))
      {
        const Real rr = pow(r,0.5);
        //const Real ur = 0.01*actuators[idx]/rr*cos(M_PI*phi/actuator_theta);
        const Real ur = 0.005*actuators[idx]/rr*cos(M_PI*phi/actuator_theta);
        UDEF[iy][ix][0] = ur * cos(theta);
        UDEF[iy][ix][1] = ur * sin(theta);
      }
    }
  }
  const double cd = forcex / (0.5*u*u*2*radius);
  fx_integral += -std::fabs(cd)*sim.dt;
}

void CylinderNozzle::act( std::vector<Real> action, const int agentID)
{
  t_change = sim.time;
  if(action.size() != actuators.size())
  {
    std::cerr << "action size needs to be equal to actuators\n";
    fflush(0);
    abort();
  }
  Real Q = 0;
  for (size_t i = 0 ; i < action.size() ; i ++)
  {
    actuators_prev_value[i] = actuators[i];
    actuators_next_value[i] = action   [i];
    Q += action[i];
  }
  Q /= actuators.size();
  for (size_t i = 0 ; i < action.size() ; i ++) actuators_next_value[i] -= Q;
}

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

std::vector<Real> CylinderNozzle::state(const int agentID)
{
  std::vector<Real> S;
  const int bins = actuators.size();

  const Real dtheta = 2.*M_PI / bins;
  std::vector<int>   n_s   (bins,0.0);
  std::vector<Real>  p_s   (bins,0.0);
  std::vector<Real> fX_s   (bins,0.0);
  std::vector<Real> fY_s   (bins,0.0);
  for(auto & block : obstacleBlocks) if(block not_eq nullptr)
  {
    for(size_t i=0; i<block->n_surfPoints; i++)
    {
      const Real x     = block->x_s[i] - origC[0];
      const Real y     = block->y_s[i] - origC[1];
      const Real ang   = atan2(y,x);
      const Real theta = ang < 0 ? ang + 2*M_PI : ang;
      const Real p     = block->p_s[i];
      const Real fx    = block->fX_s[i];
      const Real fy    = block->fY_s[i];
      const int idx = theta / dtheta;
      n_s [idx] ++;
      p_s [idx] += p;
      fX_s[idx] += fx;
      fY_s[idx] += fy;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,n_s.data(),n_s.size(),MPI_INT ,MPI_SUM,sim.comm);
  for (int idx = 0 ; idx < bins; idx++)
  {
    p_s [idx] /= n_s[idx];
    fX_s[idx] /= n_s[idx];
    fY_s[idx] /= n_s[idx];
  }

  for (int idx = 0 ; idx < bins; idx++) S.push_back( p_s[idx]);
  for (int idx = 0 ; idx < bins; idx++) S.push_back(fX_s[idx]);
  for (int idx = 0 ; idx < bins; idx++) S.push_back(fY_s[idx]);
  MPI_Allreduce(MPI_IN_PLACE,  S.data(),  S.size(),MPI_Real,MPI_SUM,sim.comm);
  S.push_back(forcex);
  S.push_back(forcey);
  S.push_back(torque);

  if (sim.rank ==0 )
    for (size_t i = 0 ; i < S.size() ; i++) std::cout << S[i] << " ";

  return S;
}
