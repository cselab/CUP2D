#include "Base.h"
#include "AMRSolver.h"
#ifdef GPU_POISSON
#include "ExpAMRSolver.h"
#endif
#include "../SimulationData.h"

std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData& s)
{
  if (s.poissonSolver == "iterative") 
  {
    return std::make_shared<AMRSolver>(s);
  } 
  else if (s.poissonSolver == "cuda_iterative") 
  {
#ifdef GPU_POISSON
    if (! _DOUBLE_PRECISION_ )
      throw std::runtime_error( 
          "Poisson solver: \"" + s.poissonSolver + "\" must be compiled with in double precision mode!" );
    return std::make_shared<ExpAMRSolver>(s);
#else
    throw std::runtime_error(
        "Poisson solver: \"" + s.poissonSolver + "\" must be compiled with the -DGPU_POISSON flag!"); 
#endif
  } 
  else {
    throw std::invalid_argument(
        "Poisson solver: \"" + s.poissonSolver + "\" unrecognized!");
  }
}
