#include "Common.h"
#include "Fields.h"

#include <mpi.h>

namespace cubismup2d {

using namespace py::literals;

// Bindings/Operators.cpp
void bindOperators(py::module &m);

// Bindings/Shapes.cpp
void bindShapes(py::module &m);

// Bindings/Simulation.cpp
void bindSimulationData(py::module &m);
void bindSimulation(py::module &m);

namespace {

/* Ensure that we load highest thread level we need. */
struct CUPMPILoader
{
  CUPMPILoader()
  {
    int flag, provided;
    MPI_Initialized(&flag);
    if (!flag)
      MPI_Init_thread(0, nullptr, MPI_THREAD_MULTIPLE, &provided);
    else
      MPI_Query_thread(&provided);
    if (provided >= MPI_THREAD_MULTIPLE)
      return;
    if (!flag)
      fprintf(stderr, "Error: MPI does not have the required thread support!\n");
    else
      fprintf(stderr, "Error: MPI does not have or not initialized with the required thread support!\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
} cup_mpi_loader;

}  // anonymous namespace

}  // namespace cubismup2d

PYBIND11_MODULE(libcubismup2d, m)
{
  using namespace cubismup2d;
  m.doc() = "CubismUP2D solver for incompressible Navier-Stokes";

  m.attr("BLOCK_SIZE") = CUP2D_BLOCK_SIZE;
#if CUP2D_FFTW
  m.attr("FFTW_ENABLED") = (int)1;
#else
  m.attr("FFTW_ENABLED") = (int)0;
#endif

  bindOperators(m);
  bindSimulationData(m);
  bindSimulation(m);
  bindFields(m);
  bindShapes(m);
}
