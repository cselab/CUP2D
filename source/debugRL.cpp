//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Simulation.h"
#include "Obstacles/StefanFish.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

#define NACTIONS 2

// declarations of functions defined further below
std::vector<double> readIC( std::string filename );
std::vector<std::vector<double>> readActions( std::string filename );
void setInitialConditions( StefanFish *agent, std::vector<double> initialConditions );

int main(int argc, char **argv)
{
  // Initialize Simulation class
  Simulation* _environment = new Simulation(argc, argv);
  _environment->init();

  // Reading Initial Conditions from RL case
  std::string icPath = "XXX/initialCondition.txt";
  auto initialConditions = readIC(icPath);

  // Reading Actions that were performed in RL
  std::string actionsPath = "XXX/actions.txt";
  auto actions = readActions(actionsPath);

  // Obtaining agent
  StefanFish *agent = dynamic_cast<StefanFish *>(_environment->getShapes()[1]);

  // Resetting environment and setting initial conditions
  setInitialConditions(agent, initialConditions);

  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Setting maximum number of steps before truncation
  size_t numActions = actions.size();

  // Variables for time and step conditions
  double t = 0;        // Current time
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action

  // Environment loop
  for( size_t a = 0; a < numActions; a++)
  {
    // Reading new action
    std::vector<double> action = actions[a];

    // Setting action
    agent->act(t, action);

    // Run the simulation until next action is required
    dtAct = agent->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->advance(dt);
    }
  }

  delete _environment;
}

std::vector<double> readIC( std::string filename )
{
  std::string line;
  double tempIC;
  std::vector<double> initialConditions;

  std::ifstream myfile(filename);
  if( myfile.is_open() )
  {
    while( std::getline(myfile,line) )
    {
      std::istringstream readingStream(line);
      while (readingStream >> tempIC)
        initialConditions.push_back(tempIC);
    }
    myfile.close();
  }
  else{
    cout << "[debugRL] Unable to open initialCondition file, setting (0,0.9,1)\n";
    initialConditions.push_back(0.0);
    initialConditions.push_back(0.9);
    initialConditions.push_back(1.0);
  } 

  return initialConditions;
} 

std::vector<std::vector<double>> readActions( std::string filename )
{
  std::string line;
  double tempA;
  std::vector<std::vector<double>> actions;

  std::ifstream myfile(filename);
  if( myfile.is_open() )
  {
    while( std::getline(myfile,line) )
    {
      std::vector<double> action(NACTIONS);
      std::istringstream readingStream(line);
      while (readingStream >> tempA)
        action.push_back(tempA);
      actions.push_back(action);
    }
    myfile.close();
  }
  else{
    cout << "[debugRL] Unable to open actions file, setting (0,0) for 20 actions\n";
    for( size_t i = 0; i<20; i++ )
    {
      std::vector<double> action(2,0.0);
      actions.push_back(action);
    }
  }

  return actions;
}

void setInitialConditions(StefanFish *agent, std::vector<double> initialConditions )
{
  // Initial conditions
  double initialAngle = initialConditions[0];
  std::vector<double> initialPosition{initialConditions[1], initialConditions[2]};

  printf("[debugRL] Initial Condition:\n");
  printf("[debugRL] angle: %f\n", initialAngle);
  printf("[debugRL] x: %f\n", initialPosition[0]);
  printf("[debugRL] y: %f\n", initialPosition[1]);

  // Setting initial position and orientation for the fish
  agent->setCenterOfMass(initialPosition.data());
  agent->setOrientation(initialAngle);
}