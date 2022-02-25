#include "Common.h"
#include "../Operator.h"
#include "../Operators/AdaptTheMesh.h"

namespace cubismup2d {

class PyOperator : public OperatorBase
{
public:
  PyOperator(std::string name) : name_{std::move(name)} { }

  void operator()(const Real dt) override
  {
    // If this fails, store the object in a permanent variable somewhere. See
    // https://github.com/pybind/pybind11/issues/1546
    // https://github.com/pybind/pybind11/pull/2839
    PYBIND11_OVERRIDE_PURE_NAME(void, OperatorBase, "__call__", operator(), dt);
  }

  std::string getName() override
  {
    return name_;
  }

private:
  std::string name_;
};

void bindOperators(py::module &m)
{
  using namespace py::literals;
  class_shared<OperatorBase, PyOperator>(m, "_OperatorBase")
    .def(py::init<std::string>(), "name"_a)
    .def("__str__", &OperatorBase::getName)
    .def("__repr__", &OperatorBase::getName)
    .def("__call__", &OperatorBase::operator(), "dt"_a);

  class_shared<Operator, OperatorBase>(m, "_SimOperator")
    .def_property_readonly("data", [](Operator *op) { return &op->sim; },
                           py::return_value_policy::reference_internal);

  class_shared<AdaptTheMesh, Operator>(m, "AdaptTheMesh")
    .def("adapt", &AdaptTheMesh::adapt);
}

}  // namespace cubismup2d
