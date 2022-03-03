//
//  CubismUP_2D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Simulation.h"
#include "Obstacles/StefanFish.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#define NACTIONS 2

/*
 ************************
  HOW TO USE THIS SCRIPT
 ************************
 This script can be used to reproduce a sample (simulation) that was performed with Korali during an RL training.
 It assumes that each agent has the same number of actions available to them (defined as 'NACTIONS').

 To reproduce the run that we want, we need:
  I. The initial conditions for that run.
  II.The actions taken by each agent.

 The initial conditions for the flow field are assumed to be the default ones from CUP2D. The initial conditions
 for the obstacles and the settings for the simulation need to be hardcoded in launch/debugRL.sh
 
 The actions taken by each agent need to be written in (and read from) files.
 The RL training should produce files named actions0.txt, actions1.txt,... (one for each agent).
 Each file contains (assuming NACTIONS=2) t0 a0 b0 t1 a1 b1 ... where a and b are the two actions and t is the 
 time those two actions were taken.

 Once actions*.txt are produced (from the original RL run), they need to be copied in the directory where this executable will run.
 After copying them, use launch/debugRL.sh to launch the run.
*/


std::vector<std::vector<double>> readActions(const int Nagents);

int main(int argc, char **argv)
{
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  Simulation* _environment = new Simulation(argc, argv, MPI_COMM_WORLD);
  _environment->init();

  const int Nagents = _environment->getShapes().size();
  std::vector<std::vector<double>> actions;
  int numActions;// max steps before truncation
  if (rank == 0)
  {
    actions  = readActions(Nagents);
    numActions = actions[0].size()/(NACTIONS+1);
  }
  MPI_Bcast(&numActions, 1, MPI_INT, 0, MPI_COMM_WORLD);

  double t = 0;        // Current time
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action

  for(int a = 0; a < numActions; a++) // Environment loop
  {
    for (int i = 0 ; i < Nagents ; i++)
    {
       StefanFish *agent = dynamic_cast<StefanFish *>(_environment->getShapes()[i].get());
       //const double time = actions[i][a*(NACTIONS+1)  ];
       std::vector<double> action(NACTIONS);
       if (rank == 0)
          for (int j = 1 ; j < NACTIONS+1; j++) action[j-1] = (actions[i][a*(NACTIONS+1)+j]);
       MPI_Bcast(action.data(), NACTIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       agent->act(t, action);
    }
    StefanFish *agent0 = dynamic_cast<StefanFish *>(_environment->getShapes()[0].get());
    dtAct = agent0->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct ) // Run simulation until next action is needed
    {
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;
      _environment->advance(dt);
    }
  }
  delete _environment;
  MPI_Finalize();
}

std::vector<std::vector<double>> readActions(const int Nagents)
{
  std::vector<std::vector<double>> actions(Nagents);
  for (int i = 0 ; i < Nagents ; i++)
  {
    std::fstream myfile("actions"+std::to_string(i)+".txt", std::ios_base::in);
    double a;
    while (myfile >> a)
	actions[i].push_back(a);
  }
  return actions;
}
