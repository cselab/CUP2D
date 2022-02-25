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
void interpolateGridToUniformMatrix(ScalarGrid *grid, ScalarElement *out);
void interpolateGridToUniformMatrix(VectorGrid *grid, VectorElement *out);

}  // namespace cubismup2d
