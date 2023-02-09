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

  const Real dtheta = 2*M_PI/Nactuators;
  const Real actw = actuator_theta*radius;
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
	  const double x = p[0]-Cx;
	  const double y = p[1]-Cy;
	  const double r = x*x+y*y;
	  //if (r < 0.6*0.6*radius*radius) continue;
	  if (r < 0.9*0.9*radius*radius) continue;

	  double theta = atan2(y,x);
	  if (theta < 0) theta += 2.*M_PI;

	  int idx = round(theta / dtheta); //this is the closest actuator
	  if (idx == Nactuators) idx = 0;  //periodic around the cylinder

	  const double theta0 = idx * dtheta;

	  const double phi = theta - theta0;
          if ( std::fabs(phi) < 0.5*actuator_theta || (idx == 0 && std::fabs(phi+2*M_PI) < 0.5*actuator_theta))
          {
	       const double ur = 0.01*actuators[idx]/r*cos(M_PI/actw*phi);
               UDEF[iy][ix][0] = ur * cos(theta);
               UDEF[iy][ix][1] = ur * sin(theta);
          }
      }
  }
}

void CylinderNozzle::act( std::vector<Real> action, const int agentID)
{
  if(action.size() != actuators.size())
  {
    std::cerr << "action size needs to be equal to actuators\n";
    fflush(0);
    abort();
  }
  double Q = 0;
  for (size_t i = 0 ; i < action.size() ; i ++)
  {
    actuators[i] += action[i];
    Q += actuators[i];
  }
  Q /= actuators.size();
  for (size_t i = 0 ; i < action.size() ; i ++) actuators[i] -= Q;
}

Real CylinderNozzle::reward(const int agentID)
{
  return -std::fabs(forcex);
}

std::vector<Real> CylinderNozzle::state(const int agentID)
{
  std::vector<double> S;
  S.push_back(sim.time   );
  S.push_back(forcex     );
  S.push_back(forcey     );
  const int bins = actuators.size();
  const double dtheta = 2.*M_PI / bins;
  std::vector<int>     n_s   (bins,0.0);
  std::vector<double>  p_s   (bins,0.0);
  std::vector<double> fX_s   (bins,0.0);
  std::vector<double> fY_s   (bins,0.0);
  for(auto & block : obstacleBlocks) if(block not_eq nullptr)
  {
    for(size_t i=0; i<block->n_surfPoints; i++)
    {
      const double x     = block->x_s[i] - origC[0];
      const double y     = block->y_s[i] - origC[1];
      const double ang   = atan2(y,x);
      const double theta = ang < 0 ? ang + 2*M_PI : ang;
      const double p     = block->p_s[i];
      const double fx    = block->fX_s[i];
      const double fy    = block->fY_s[i];
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

  if (sim.time < 0.1)
    for (int  i = 0 ; i < S.size() ; i++)S[i]=0;

  return S;
}
