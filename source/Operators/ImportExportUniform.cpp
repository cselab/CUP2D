//
//  CubismUP_2D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "ImportExportUniform.h"
#include <Cubism/ImportExport.hh>

namespace cubismup2d {

namespace {

struct Getter {
  template <typename T>
  T operator()(T x) const { return x; }
};

struct Setter {
  template <typename T>
  void operator()(T &x, T y) const { x = y; }
};

}  // namespace anonymous

void exportToUniformMatrix(ScalarGrid *grid, ScalarElement *out)
{
  cubism::exportGridToUniformMatrix<ScalarLab>(grid, Getter{}, {0}, out);
}

void exportToUniformMatrix(VectorGrid *grid, VectorElement *out)
{
  cubism::exportGridToUniformMatrix<VectorLab>(grid, Getter{}, {0, 1}, out);
}

void exportToUniformMatrixNearestInterpolation(ScalarGrid *grid, ScalarElement *out)
{
  cubism::exportGridToUniformMatrixNearestInterpolation(grid, Getter{}, out);
}

void exportToUniformMatrixNearestInterpolation(VectorGrid *grid, VectorElement *out)
{
  cubism::exportGridToUniformMatrixNearestInterpolation(grid, Getter{}, out);
}

void importFromUniformMatrix(ScalarGrid *grid, const ScalarElement *in)
{
  cubism::importGridFromUniformMatrix(grid, Setter{}, in);
}

void importFromUniformMatrix(VectorGrid *grid, const VectorElement *in)
{
  cubism::importGridFromUniformMatrix(grid, Setter{}, in);
}

}  // namespace cubismup2d
