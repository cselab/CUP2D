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

template <typename T>
static auto bindShape(py::module &m, const char *name)
{
  return class_shared<T, Shape>(m, name)
    .def(py::init(&makeShape<T>), "data"_a, "argv"_a, "center"_a);
}

void bindShapes(py::module &m)
{
  class_shared<Shape>(m, "_Shape")
    .def_property_readonly("data", [](Shape *shape) { return &shape->sim; },
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

  bindShape<Disk>(m, "_Disk")
    .def_property_readonly("r", &Disk::getRadius);
  bindShape<HalfDisk>(m, "_HalfDisk");
  bindShape<Ellipse>(m, "_Ellipse");
  bindShape<Rectangle>(m, "_Rectangle");
}

}  // namespace cubismup2d
