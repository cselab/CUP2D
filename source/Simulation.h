//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "SimulationData.h"
#include "Operator.h"

class Profiler;

class Simulation
{
 protected:
  cubism::ArgumentParser parser;
  std::vector<Operator*> pipeline;

  void createShapes();
  void parseRuntime();
  void createPipeline();
  void clearPipeline();
 public:
  SimulationData sim;

  Simulation(int argc, char ** argv);
  ~Simulation();

  void reset();
  void init();
  void startObstacles();
  void simulate();
  double calcMaxTimestep();
  bool advance(const double DT);

  const std::vector<Shape*>& getShapes() { return sim.shapes; }
};
