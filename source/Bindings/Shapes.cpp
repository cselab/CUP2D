#include "../Obstacles/ShapesSimple.h"
#include "../Obstacles/StefanFish.h"
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

static auto xyToPair(const Real Shape:: *x, const Real Shape:: *y)
{
  return [x, y](const Shape &shape)
  {
    return std::array<Real, 2>{{shape.*x, shape.*y}};
  };
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
    .def_readonly("com", &Shape::centerOfMass)
    .def_property_readonly("force", xyToPair(&Shape::forcex, &Shape::forcey))
    .def_property_readonly("force_P", xyToPair(&Shape::forcex_P, &Shape::forcey_P))
    .def_property_readonly("force_V", xyToPair(&Shape::forcex_V, &Shape::forcey_V))
    .def_readonly("drag", &Shape::drag)
    .def_readonly("thrust", &Shape::thrust)
    .def_readonly("lift", &Shape::lift);

  bindShape<Disk>(m, "_Disk")
    .def_property_readonly("r", &Disk::getRadius);
  bindShape<HalfDisk>(m, "_HalfDisk");
  bindShape<Ellipse>(m, "_Ellipse");
  bindShape<Rectangle>(m, "_Rectangle");
  bindShape<StefanFish>(m, "_StefanFish")
    .def("act", &StefanFish::act, "Set Action")
    .def("state", &StefanFish::state, "Get State")
    .def_readonly("efficiency", &StefanFish::EffPDefBnd);
}

}  // namespace cubismup2d
