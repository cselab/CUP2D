#pragma once

#include "../Definitions.h"
#include <memory>

struct SimulationData;

class PoissonSolver {
public:
  virtual ~PoissonSolver() = default;
  virtual void solve(const ScalarGrid *input, ScalarGrid *output) = 0;
};

std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData &s);
