//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "SimulationData.h"
#include "Operator.h"

class Simulation
{
 public:
  SimulationData sim;
  std::vector<std::shared_ptr<Operator>> pipeline;
 protected:
  cubism::ArgumentParser parser;

  void createShapes();
  void parseRuntime();

public:
  Simulation(int argc, char ** argv, MPI_Comm comm);
  ~Simulation();

  /// Find the first operator in the pipeline that matches the given type.
  /// Returns `nullptr` if nothing was found.
  template <typename Op>
  Op *findOperator() const
  {
    for (const auto &ptr : pipeline) {
      Op *out = dynamic_cast<Op *>(ptr.get());
      if (out != nullptr)
        return out;
    }
    return nullptr;
  }

  /// Insert the operator at the end of the pipeline.
  void insertOperator(std::shared_ptr<Operator> op);

  /// Insert an operator after the operator of the given name.
  /// Throws an exception if the name is not found.
  void insertOperatorAfter(std::shared_ptr<Operator> op, const std::string &name);

  void reset();
  void resetRL();
  void init();
  void startObstacles();
  void simulate();
  Real calcMaxTimestep();
  void advance(const Real dt);

  const std::vector<std::shared_ptr<Shape>>& getShapes() { return sim.shapes; }
};
