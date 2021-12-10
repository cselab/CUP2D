#include "Base.h"
#include "AMRSolver.h"
#include "FFTW.h"
#include "../SimulationData.h"

std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData& s)
{
  if (s.poissonSolver == "iterative") {
    return std::make_shared<AMRSolver>(s);
  } else if (s.poissonSolver == "fftw_dirichlet") {
    return std::make_shared<FFTWDirichlet>(s, s.fftwPoissonTol);
  } else {
    throw std::invalid_argument("expected 'iterative' or 'fftw_dirichlet', "
                                "got " + s.poissonSolver);
  }
}
