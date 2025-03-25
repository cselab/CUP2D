#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cubismup2d {

namespace py = pybind11;

template <typename T, typename ...Args>
using class_shared = py::class_<T, Args..., std::shared_ptr<T>>;

}  // namespace cubismup2d
