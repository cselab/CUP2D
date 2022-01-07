#include "../Obstacles/ShapesSimple.h"
#include "../Shape.h"
#include "../SimulationData.h"
#include "../Utils/FactoryFileLineParser.h"
#include "Common.h"
#include <sstream>

using namespace pybind11::literals;
namespace py = pybind11;

namespace cubismup2d {

template <typename S>
static std::shared_ptr<S> makeShape(
    SimulationData& s,
    const std::string &argv,
    std::array<Real, 2> C)
{
  std::istringstream stream{argv};
  FactoryFileLineParser ffparser{stream};
  return std::make_shared<S>(s, ffparser, C.data());
}

void bindShapes(py::module &m)
{
  py::class_<Shape, std::shared_ptr<Shape>>(m, "_Shape")
    .def_property_readonly("sim", [](Shape *shape) { return &shape->sim; },
                           py::return_value_policy::reference_internal)
    .def_readonly("id", &Shape::obstacleID)
    .def_readonly("center", &Shape::center)
    .def_readwrite("u", &Shape::u)
    .def_readwrite("v", &Shape::v)
    .def_property_readonly(
        "com",
        [](const Shape& shape) {
          return std::array<Real, 2>{shape.centerOfMass[0], shape.centerOfMass[1]};
        },
        "center of mass");

  py::class_<Disk, Shape, std::shared_ptr<Disk>>(m, "_Disk")
    .def(py::init(&makeShape<Disk>), "sim"_a, "argv"_a, "center"_a)
    .def_property_readonly("r", &Disk::getRadius);
}

}  // namespace cubismup2d
