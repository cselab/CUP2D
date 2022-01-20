#include "Base.h"
#include "AMRSolver.h"
#include "../SimulationData.h"

std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData& s)
{
  if (s.poissonSolver == "iterative") {
    return std::make_shared<AMRSolver>(s);
  } else {
    throw std::invalid_argument(
        "unrecognized Poisson solver: " + s.poissonSolver);
  }
}
