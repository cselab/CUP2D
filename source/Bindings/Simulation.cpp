#include "Common.h"
#include "Fields.h"
#include "../Shape.h"
#include "../Simulation.h"

#include <mpi.h>

namespace cubismup2d {

using namespace py::literals;

// Bindings/Shapes.cpp
void bindShapes(py::module &m);

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

/// Operator that stops the simulation when Ctrl-C is pressed in Python.
class SIGINTHandlerOperator : public Operator {
public:
  using Operator::Operator;

  void operator()(double /* dt */) override {
    // https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-properly-handle-ctrl-c-in-long-running-functions
    if (PyErr_CheckSignals() != 0)
      throw py::error_already_set();
  }

  std::string getName() override {
    return "SIGINTHandlerOperator";
  }
};

}  // anonymous namespace

static void bindSimulationData(py::module &m)
{
  py::class_<SimulationData>(m, "SimulationData")
      .def_readonly("CFL", &SimulationData::CFL)
      .def_readonly("extents", &SimulationData::extents)
      .def_readonly("uinfx", &SimulationData::uinfx)
      .def_readonly("uinfy", &SimulationData::uinfy)
      .def_readonly("nsteps", &SimulationData::nsteps);
}

static std::shared_ptr<Simulation> pyCreateSimulation(
      const std::vector<std::string> &argv,
      uintptr_t commPtr)
{
  // https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
  // In Python, pass communicators with `MPI._addressof(comm)`.
  MPI_Comm comm = commPtr ? *(MPI_Comm *)commPtr : MPI_COMM_WORLD;
  std::vector<char *> ptrs(argv.size());
  for (size_t i = 0; i < argv.size(); ++i)
    ptrs[i] = const_cast<char *>(argv[i].data());
  auto sim = std::make_shared<Simulation>((int)ptrs.size(), ptrs.data(), comm);
  sim->pipeline.push_back(new SIGINTHandlerOperator{sim->sim});
  return sim;
}

static void bindSimulation(py::module &m)
{
  py::class_<Simulation, std::shared_ptr<Simulation>>(m, "Simulation")
      .def(py::init(&pyCreateSimulation), "argv"_a, "comm"_a = 0)
      .def_readonly("sim", &Simulation::sim,
                    py::return_value_policy::reference_internal)
      .def("add_shape", [](Simulation *sim, std::shared_ptr<Shape> shape) {
        sim->sim.addShape(std::move(shape));
      }, "shape"_a)
      .def_property_readonly("fields", [](Simulation *sim) {
        return FieldsView{&sim->sim};
      })
      .def("init", &Simulation::init)
      .def("simulate", &Simulation::simulate);
}

}  // namespace cubismup2d

PYBIND11_MODULE(libcubismup2d, m)
{
  using namespace cubismup2d;
  m.doc() = "CubismUP2D solver for incompressible Navier-Stokes";

  m.attr("BLOCK_SIZE") = CUP2D_BLOCK_SIZE;

  bindSimulationData(m);
  bindSimulation(m);
  bindFields(m);
  bindShapes(m);
}
