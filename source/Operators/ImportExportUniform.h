//
//  CubismUP_2D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Definitions.h"

namespace cubismup2d {

/// Interpolate blocks onto the given contiguous matrix of assumed dimension
/// (max grid cells Y, max grid cells X). The interpolation is O(h^2) accurate
/// everywhere except at the boundary between coarse and fine blocks, where the
/// accuracy is O(h).
void exportToUniformMatrix(ScalarGrid *grid, ScalarElement *out);
void exportToUniformMatrix(VectorGrid *grid, VectorElement *out);

void exportToUniformMatrixNearestInterpolation(ScalarGrid *grid, ScalarElement *out);
void exportToUniformMatrixNearestInterpolation(VectorGrid *grid, VectorElement *out);

void importFromUniformMatrix(ScalarGrid *grid, const ScalarElement *in);
void importFromUniformMatrix(VectorGrid *grid, const VectorElement *in);

}  // namespace cubismup2d
