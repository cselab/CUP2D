#include "../Simulation.h"
#include "../Operators/AdaptTheMesh.h"
#include "../Operators/Helpers.h"
#include "../Operators/PutObjectsOnGrid.h"
#include "../Shape.h"
#include "Common.h"

namespace cubismup2d {

using namespace py::literals;

namespace {

class SIGINTHandlerOperator : public Operator {
public:
  using Operator::Operator;

  void operator()(double) override {

    if (PyErr_CheckSignals() != 0)
      throw py::error_already_set();
  }

  std::string getName() override { return "SIGINTHandlerOperator"; }
};

} // namespace

void bindSimulationData(py::module &m) {
  auto pyData =
      py::class_<SimulationData>(m, "SimulationData")
          .def_readonly("CFL", &SimulationData::CFL)
          .def_readonly("extents", &SimulationData::extents)
          .def_readonly("uinfx", &SimulationData::uinfx)
          .def_readonly("uinfy", &SimulationData::uinfy)
          .def_readonly("shapes", &SimulationData::shapes)
          .def_readonly("smagorinskyCoeff", &SimulationData::smagorinskyCoeff)
          .def_readwrite("time", &SimulationData::time)
          .def_readwrite("step", &SimulationData::step)
          .def_readwrite("_nsteps", &SimulationData::nsteps)
          .def_readwrite("_tend", &SimulationData::endTime)
          .def_readwrite("mute_all", &SimulationData::muteAll)
          .def_readwrite("nu", &SimulationData::nu);

  const auto byRef = py::return_value_policy::reference_internal;
  pyData.def_readonly("chi", &SimulationData::chi, byRef);
  pyData.def_readonly("vel", &SimulationData::vel, byRef);
  pyData.def_readonly("vOld", &SimulationData::vOld, byRef);
  pyData.def_readonly("pres", &SimulationData::pres, byRef);
  pyData.def_readonly("tmpV", &SimulationData::tmpV, byRef);
  pyData.def_readonly("tmp", &SimulationData::tmp, byRef);
  pyData.def_readonly("pold", &SimulationData::pold, byRef);
  pyData.def_readonly("Cs", &SimulationData::Cs, byRef);

  pyData.def("dump_chi", &SimulationData::dumpChi, "prefix"_a);
  pyData.def("dump_vel", &SimulationData::dumpVel, "prefix"_a);
  pyData.def("dump_vOld", &SimulationData::dumpVold, "prefix"_a);
  pyData.def("dump_pres", &SimulationData::dumpPres, "prefix"_a);
  pyData.def("dump_tmpV", &SimulationData::dumpTmpV, "prefix"_a);
  pyData.def("dump_tmp", &SimulationData::dumpTmp, "prefix"_a);
  pyData.def("dump_pold", &SimulationData::dumpPold, "prefix"_a);
  pyData.def("dump_all", &SimulationData::dumpAll, "prefix"_a,
             "Compute vorticity (stored in tmp) and dump relevant fields.");
}

static std::shared_ptr<Simulation>
pyCreateSimulation(const std::vector<std::string> &argv, uintptr_t commPtr) {

  MPI_Comm comm = commPtr ? *(MPI_Comm *)commPtr : MPI_COMM_WORLD;
  std::vector<char *> ptrs(argv.size());
  for (size_t i = 0; i < argv.size(); ++i)
    ptrs[i] = const_cast<char *>(argv[i].data());
  auto sim = std::make_shared<Simulation>((int)ptrs.size(), ptrs.data(), comm);
  sim->pipeline.push_back(std::make_shared<SIGINTHandlerOperator>(sim->sim));
  return sim;
}

static void pyAdaptMesh(Simulation &sim) {

  auto *const adapt = sim.findOperator<AdaptTheMesh>();
  auto *const obj = sim.findOperator<PutObjectsOnGrid>();
  if (!adapt)
    throw std::runtime_error("AdaptTheMesh operator not found");
  if (!obj)
    throw std::runtime_error("PutObjectsOnGrid operator not found");
  adapt->adapt();
  obj->putObjectsOnGrid();
}

static void pyComputeVorticity(Simulation &sim) {
  computeVorticity op{sim.sim};
  op(0);
}

void bindSimulation(py::module &m) {
  class_shared<Simulation>(m, "_Simulation")
      .def(py::init(&pyCreateSimulation), "argv"_a, "comm"_a = 0)
      .def_readonly("data", &Simulation::sim,
                    py::return_value_policy::reference_internal)
      .def(
          "add_shape",
          [](Simulation *sim, std::shared_ptr<Shape> shape) {
            sim->sim.addShape(std::move(shape));
          },
          "shape"_a)
      .def("insert_operator", &Simulation::insertOperator, "op"_a)
      .def("insert_operator", &Simulation::insertOperatorAfter, "op"_a,
           "after"_a)
      .def("adapt_mesh", &pyAdaptMesh)
      .def("compute_vorticity", &pyComputeVorticity,
           "compute the vorticity and store it to the tmp field")
      .def("init", &Simulation::init)
      .def("simulate", &Simulation::simulate);
}

} // namespace cubismup2d
