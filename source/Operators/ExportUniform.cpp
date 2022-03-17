//
//  CubismUP_2D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "ExportUniform.h"
#include "../Operator.h"

using cubism::BlockInfo;
using cubism::StencilInfo;

namespace cubismup2d {

/*
   Interpolate block onto a uniform matrix with scaling factor of `factor`,
   where `factor` is a power of two larger than or equal to 2.
   The `out` pointer should point to the first element of the submatrix
   corresponding to this block.

   factor == 2
   +-------+
   |   |   |
   | . | . | dy=-1/2    <-- . = output point
   |   |   |
   |---X---|            <-- X = existing grid point
   |   |   |
   | . | . | dy=+1/2
   |   |   |
   +-------+

   factor == 4
   +---------------+
   |   |   |   |   |
   | . | . | . | . |  dy=-3/4
   |   |   |   |   |
   |---+---+---+---|
   |   |   |   |   |
   | . | . | . | . |  dy=-1/4
   |   |   |   |   |
   |---+---X---+---|
   |   |   |   |   |
   | . | . | . | . |  dy=+1/4
   |   |   |   |   |
   |---+---+---+---|
   |   |   |   |   |
   | . | . | . | . |  dy=+3/4
   |   |   |   |   |
   +---------------+
*/
template <typename Block, typename Lab, typename T>
static void interpolateBlockImpl(
    Lab &lab, int factor, int yStride, T * __restrict__ out)
{
  constexpr int BX = Block::sizeX;
  constexpr int BY = Block::sizeY;
  for (int iy = 0; iy < BY; ++iy)
  for (int ix = 0; ix < BX; ++ix)
  {
    const auto dudx = (Real)0.5 * (lab(ix + 1, iy) - lab(ix - 1, iy));
    const auto dudy = (Real)0.5 * (lab(ix, iy + 1) - lab(ix, iy - 1));
    const auto dudx2 = (lab(ix + 1, iy) + lab(ix - 1, iy)) - 2 * lab(ix, iy);
    const auto dudy2 = (lab(ix, iy + 1) + lab(ix, iy - 1)) - 2 * lab(ix, iy);  
    const auto dudxdy = (Real)0.25 * ((lab(ix + 1, iy + 1) + lab(ix - 1, iy - 1))
                                    - (lab(ix + 1, iy - 1) + lab(ix - 1, iy + 1)));

    // This for-loop will be expanded at compile time for factor == 1, 2, 4.
    const Real invF = (Real)0.5 / factor;
    for (int jy = 0; jy < factor; ++jy)
    for (int jx = 0; jx < factor; ++jx)
    {
      const Real dx = (-factor + 1 + 2 * jx) * invF;
      const Real dy = (-factor + 1 + 2 * jy) * invF;
      const auto value = lab(ix, iy)
                       + (dx * dudx + dy * dudy)
                       + (Real)0.5 * (dx * dx * dudx2 + dy * dy * dudy2)
                       + dx * dy * dudxdy;
      out[(factor * iy + jy) * yStride + (factor * ix + jx)] = value;
    }
  }
}

/// Optimize `interpolateBlockImpl` for given `kFactor`.
template <typename Block, int kFactor, typename Lab, typename T>
static void interpolateBlock(Lab &lab, int yStride, T * __restrict__ out)
{
  interpolateBlockImpl<Block>(lab, kFactor, yStride, out);
}

/// Copy block as-is to the output submatrix.
template <typename Block, typename Lab, typename T>
static void copyBlock(Lab &lab, int yStride, T * __restrict__ out)
{
  constexpr int BX = Block::sizeX;
  constexpr int BY = Block::sizeY;
  for (int iy = 0; iy < BY; ++iy)
  for (int ix = 0; ix < BX; ++ix)
  {
    out[iy * yStride + ix] = lab(ix, iy);
  }
}

/// Dispatch to interpolation functions depending on the level.
template <typename Block, typename Lab, typename T>
static void dispatchBlockInterpolation(
    Lab &lab, const BlockInfo &info, int levelMax, int yStride, T *out)
{
  constexpr int BX = Block::sizeX;
  constexpr int BY = Block::sizeY;
  const int dLevel = levelMax - 1 - info.level;
  const int yOffset = info.index[1] * BY * (1 << dLevel);
  const int xOffset = info.index[0] * BX * (1 << dLevel);
  T * const outSubmatrix = out + yOffset * yStride + xOffset;

  switch (dLevel) {
    case 0: copyBlock<Block>(lab, yStride, outSubmatrix); break;
    case 1: interpolateBlock<Block, 2>(lab, yStride, outSubmatrix); break;
    case 2: interpolateBlock<Block, 4>(lab, yStride, outSubmatrix); break;
    case 3: interpolateBlock<Block, 8>(lab, yStride, outSubmatrix); break;
    default:
        assert(dLevel >= 0);
        interpolateBlockImpl<Block>(lab, 1 << dLevel, yStride, outSubmatrix);
  }
}

namespace {

template <typename Block>
struct ExportKernel
{
  StencilInfo stencil;
  int levelMax;
  int yStride;
  typename Block::ElementType *out;

  template <typename Lab>
  void operator()(Lab &lab, const BlockInfo &info) const
  {
    dispatchBlockInterpolation<Block> (lab, info, levelMax, yStride, out);
  }
};

}  // anonymous namespace

template <typename Lab, typename Block, typename Grid, typename T>
static void _interpolateGrid(Grid *grid, T *out, StencilInfo stencil)
{
  const int yStride = Block::sizeX * grid->getMaxMostRefinedBlocks()[0];
  ExportKernel<Block> kernel{stencil, grid->levelMax, yStride, out};
  cubism::compute<Lab>(kernel, grid);
}

void interpolateGridToUniformMatrix(ScalarGrid *grid, ScalarElement *out)
{
  _interpolateGrid<ScalarLab, ScalarBlock>(
      grid, out, {-1, -1, 0, 2, 2, 1, true, {0}});
}

void interpolateGridToUniformMatrix(VectorGrid *grid, VectorElement *out)
{
  _interpolateGrid<VectorLab, VectorBlock>(
      grid, out, {-1, -1, 0, 2, 2, 1, true, {0, 1}});
}

}  // namespace cubismup2d
