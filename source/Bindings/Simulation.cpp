#include "Common.h"
#include "../Shape.h"
#include "../Simulation.h"

namespace cubismup2d {

using namespace py::literals;

namespace {

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

void bindSimulationData(py::module &m)
{
  auto pySim = py::class_<SimulationData>(m, "_SimulationData")
      .def_readonly("CFL", &SimulationData::CFL)
      .def_readonly("extents", &SimulationData::extents)
      .def_readonly("uinfx", &SimulationData::uinfx)
      .def_readonly("uinfy", &SimulationData::uinfy)
      .def_readonly("time", &SimulationData::time)
      .def_readonly("step", &SimulationData::step)
      .def_readwrite("_nsteps", &SimulationData::nsteps)
      .def_readwrite("_tend", &SimulationData::endTime);

  // Bind all grids. If updating this, update _FieldsProxy in
  // cubismup2d/simulation.py as well.
  const auto byRef = py::return_value_policy::reference_internal;
  pySim.def_readonly("chi", &SimulationData::chi, byRef);
  pySim.def_readonly("vel", &SimulationData::vel, byRef);
  pySim.def_readonly("vOld", &SimulationData::vOld, byRef);
  pySim.def_readonly("pres", &SimulationData::pres, byRef);
  pySim.def_readonly("tmpV", &SimulationData::tmpV, byRef);
  pySim.def_readonly("tmp", &SimulationData::tmp, byRef);
  pySim.def_readonly("uDef", &SimulationData::uDef, byRef);
  pySim.def_readonly("pold", &SimulationData::pold, byRef);
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
  sim->pipeline.push_back(std::make_shared<SIGINTHandlerOperator>(sim->sim));
  return sim;
}

void bindSimulation(py::module &m)
{
  class_shared<Simulation>(m, "_Simulation")
      .def(py::init(&pyCreateSimulation), "argv"_a, "comm"_a = 0)
      .def_readonly("sim", &Simulation::sim,
                    py::return_value_policy::reference_internal)
      .def("add_shape", [](Simulation *sim, std::shared_ptr<Shape> shape) {
        sim->sim.addShape(std::move(shape));
      }, "shape"_a)
      .def("insert_operator", &Simulation::insertOperator, "op"_a)
      .def("insert_operator", &Simulation::insertOperatorAfter,
           "op"_a, "after"_a)
      .def("init", &Simulation::init)
      .def("simulate", &Simulation::simulate);
}

}  // namespace cubismup2d
