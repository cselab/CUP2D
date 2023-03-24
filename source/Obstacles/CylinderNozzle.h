//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Shape.h"
#include "FishUtilities.h"

class CylinderNozzle : public Shape
{
  const Real radius;
  std::vector<Real> actuators;
  std::vector<Real> actuators_prev_value;
  std::vector<Real> actuators_next_value;
  const int Nactuators;
  const Real actuator_theta;
  Real fx_integral = 0;
  //std::vector<double>   action_taken;
  //std::vector<double> t_action_taken;
  //int curr_idx = 0;
  std::vector < Schedulers::ParameterSchedulerScalar > actuatorSchedulers;
  Real t_change = 0;
  const Real regularizer;

 public:

  CylinderNozzle(SimulationData& s, cubism::ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C),
  radius( p("-radius").asDouble(0.1) ), 
  Nactuators ( p("-Nactuators").asInt(2)),
  actuator_theta ( p("-actuator_theta").asDouble(10.)*M_PI/180.),
  regularizer( p("-regularizer").asDouble(1.0))
  {
    actuators.resize(Nactuators,0.);
    actuatorSchedulers.resize(Nactuators);
    actuators_prev_value.resize(Nactuators);
    actuators_next_value.resize(Nactuators);
    #if 0
    if (sim.rank == 0)
    {
      std::string line;
      ifstream myfile;
      myfile.open ("actions0.txt",std::ios_base::in);

      while (std::getline(myfile, line))
      {
          std::istringstream iss(line);
          std::cout << "--> ";
          double temp;
          iss >> temp;
          t_action_taken.push_back(temp);
          std::cout << temp << " ";
          for( size_t a = 0; a<actuators.size(); a++ )
          {
            iss >> temp;
            std::cout << temp << " ";
            action_taken.push_back(temp);
          }
          std::cout << std::endl;
      }
      myfile.close();
    }
    int n = (int) t_action_taken.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, sim.comm );
    t_action_taken.resize(n);
    action_taken.resize(actuators.size()*n);
    MPI_Bcast(t_action_taken.data(), t_action_taken.size(), MPI_DOUBLE, 0, sim.comm );
    MPI_Bcast(  action_taken.data(),   action_taken.size(), MPI_DOUBLE, 0, sim.comm );
    #endif
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void finalize() override;

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  void act(std::vector<Real> action, const int agentID);
  Real reward(const int agentID);
  std::vector<Real> state(const int agentID);
};
