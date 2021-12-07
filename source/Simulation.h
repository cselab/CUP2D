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
 public:
  SimulationData sim;
  std::vector<Operator*> pipeline;
 protected:
  cubism::ArgumentParser parser;

  void createShapes();
  void parseRuntime();
  // should this stuff be moved? - serialize method will do that
  //void _dumpSettings(ostream& outStream);

public:
  Simulation(int argc, char ** argv, MPI_Comm comm);
  ~Simulation();

  void reset();
  void resetRL();
  void init();
  void startObstacles();
  void simulate();
  Real calcMaxTimestep();
  bool advance(const Real DT);

  const std::vector<std::shared_ptr<Shape>>& getShapes() { return sim.shapes; }
};
