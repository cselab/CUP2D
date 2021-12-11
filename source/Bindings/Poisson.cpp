#include "Common.h"
#include "../Poisson/AMRSolver.h"
#include "../Poisson/FFTW.h"
#include "../SimulationData.h"

namespace cubismup2d {

void bindPoissonSolvers(py::module &m)
{
  using namespace py::literals;
  class_shared<PoissonSolver>(m, "Solver")
    .def("solve", &PoissonSolver::solve, "input"_a, "output"_a);

  class_shared<AMRSolver, PoissonSolver>(m, "AMRSolver")
    .def(py::init<SimulationData &>(), "sim"_a);

  class_shared<FFTWDirichlet, PoissonSolver>(m, "FFTWDirichlet")
    .def(py::init<SimulationData &, Real>(), "sim"_a, "tol"_a);
}

}  // namespace cubismup2d
