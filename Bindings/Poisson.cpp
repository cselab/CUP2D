#include "Common.h"
#include "../Poisson/AMRSolver.h"
#include "../SimulationData.h"

namespace cubismup2d {

void bindPoissonSolvers(py::module &m)
{
  using namespace py::literals;
  class_shared<PoissonSolver>(m, "Solver")
    .def("solve", &PoissonSolver::solve, "input"_a, "output"_a);

  class_shared<AMRSolver, PoissonSolver>(m, "AMRSolver")
    .def(py::init<SimulationData &>(), "data"_a);
}

}  // namespace cubismup2d
