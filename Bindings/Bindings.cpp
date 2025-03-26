#include "Common.h"
#include "Fields.h"
#include <mpi.h>
namespace cubismup2d {
using namespace py::literals;
void bindOperators(py::module &m);
void bindPoissonSolvers(py::module &m);
void bindShapes(py::module &m);
void bindSimulationData(py::module &m);
void bindSimulation(py::module &m);
namespace {
struct CUPMPILoader {
  CUPMPILoader() {
    int flag, provided;
    MPI_Initialized(&flag);
    if (!flag)
      MPI_Init_thread(0, nullptr, MPI_THREAD_MULTIPLE, &provided);
    else
      MPI_Query_thread(&provided);
    if (provided >= MPI_THREAD_MULTIPLE)
      return;
    if (!flag) {
      fprintf(stderr, "Error: MPI does not have the required thread support!\n"
                      "Try setting the following environment variable:\n"
                      "    MPICH_MAX_THREAD_SAFETY=multiple\n");
    } else {
      fprintf(stderr, "Error: MPI does not have or not initialized with the "
                      "required thread support!\n");
    }
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
} cup_mpi_loader;
} // namespace
} // namespace cubismup2d
PYBIND11_MODULE(libcubismup2d, m) {
  using namespace cubismup2d;
  m.doc() = "CubismUP2D solver for incompressible Navier-Stokes";
  m.attr("BLOCK_SIZE") = CUP2D_BLOCK_SIZE;
  bindSimulationData(m);
  bindSimulation(m);
  bindFields(m);
  bindOperators(m);
  bindShapes(m);
  auto poisson = m.def_submodule("poisson");
  bindPoissonSolvers(poisson);
}
