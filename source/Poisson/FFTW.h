#pragma once

#include "Base.h"

class FFTWDirichletImpl;

class FFTWDirichlet : public PoissonSolver {
public:
  FFTWDirichlet(SimulationData& s, Real tol = 0.1);
  ~FFTWDirichlet();

  void solve(const ScalarGrid *input, ScalarGrid *output) override;

private:
  std::unique_ptr<FFTWDirichletImpl> impl_;
};
