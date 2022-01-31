#include "Base.h"
#include "AMRSolver.h"
#include "ExpAMRSolver.h"
#include "../SimulationData.h"

std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData& s)
{
  if (s.poissonSolver == "iterative") {
    return std::make_shared<AMRSolver>(s);
  } else if (s.poissonSolver == "cuda_iterative") {
    if (! _DOUBLE_PRECISION_ )
      throw std::runtime_error( 
          "GPU-accelerated Poisson solver must be compiled with double precision!" );
    return std::make_shared<ExpAMRSolver>(s);
  } else {
    throw std::invalid_argument(
        "unrecognized Poisson solver: " + s.poissonSolver);
  }
}
