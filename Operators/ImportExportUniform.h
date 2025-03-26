

#pragma once

#include "../Definitions.h"

namespace cubismup2d {

void exportToUniformMatrix(ScalarGrid *grid, ScalarElement *out);
void exportToUniformMatrix(VectorGrid *grid, VectorElement *out);

void exportToUniformMatrixNearestInterpolation(ScalarGrid *grid,
                                               ScalarElement *out);
void exportToUniformMatrixNearestInterpolation(VectorGrid *grid,
                                               VectorElement *out);

void importFromUniformMatrix(ScalarGrid *grid, const ScalarElement *in);
void importFromUniformMatrix(VectorGrid *grid, const VectorElement *in);

} // namespace cubismup2d
