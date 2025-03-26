#include "ImportExport.h"
#include "Definitions.h"
#include "StencilInfo.h"

/*
Implementation details.

Strides:
    For simplicity and completeness, we do not hard-code the x-stride to 1 at
    the level of kernels. Instead we pass x-stride (sx) as an argument which is
    at the moment always equal to 1. Clang, gcc and icc all seem to be able to
    optimize the argument out. They are successful only when the strides are
    passed as different arguments and not e.g. as std::array<size_t, 3>.

Vectorization:
    Compilers are not really succesful at vectorizing the innermost loop unless
    the offsets due to y and z loops are computed in advance. Using lab
    prevents us from computing the offset for the input array, but at least we
    can do so for the output array.
*/

namespace cubism
{

namespace ie_detail
{

// Enforce recompilation and optimization for specific N.
template <int N>
struct ConstInt
{
  constexpr operator int() { return N; }
};

/// Copy block as-is to the output submatrix.
template <typename Block, typename Lab, typename T>
static void copyBlock(Lab &lab, size_t sx, size_t sy, size_t sz, T * __restrict__ out)
{
  for (int iz = 0; iz < Block::sizeZ; ++iz)
  for (int iy = 0; iy < Block::sizeY; ++iy)
  {
    // This helps with vectorization.
    T * __restrict__ row = out + iy * sy + iz * sz;
    for (int ix = 0; ix < Block::sizeX; ++ix)
      row[ix * sx] = lab(ix, iy, iz);
  }
}

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
template <typename Block, typename Lab, typename Int, typename T>
static void upscaleBlock2D(
    Lab &lab, Int _factor, size_t sx, size_t sy, T * __restrict__ out)
{
  using Real = typename Block::RealType;
  const int factor = _factor;
  for (int iy = 0; iy < Block::sizeY; ++iy)
  for (int ix = 0; ix < Block::sizeX; ++ix) {
    const auto dudx = (Real)0.5 * (lab(ix + 1, iy) - lab(ix - 1, iy));
    const auto dudy = (Real)0.5 * (lab(ix, iy + 1) - lab(ix, iy - 1));
    const auto dudx2 = (lab(ix + 1, iy) + lab(ix - 1, iy)) - 2 * lab(ix, iy);
    const auto dudy2 = (lab(ix, iy + 1) + lab(ix, iy - 1)) - 2 * lab(ix, iy);
    const auto dudxdy = (Real)0.25 * ((lab(ix + 1, iy + 1) + lab(ix - 1, iy - 1))
                                    - (lab(ix + 1, iy - 1) + lab(ix - 1, iy + 1)));

    // This for-loop will be expanded at compile time for factor == 1, 2, 4.
    const Real invF = (Real)0.5 / factor;
    for (int jy = 0; jy < factor; ++jy)
    for (int jx = 0; jx < factor; ++jx) {
      const Real dx = (-factor + 1 + 2 * jx) * invF;
      const Real dy = (-factor + 1 + 2 * jy) * invF;
      const auto value = lab(ix, iy)
                       + (dx * dudx + dy * dudy)
                       + (Real)0.5 * (dx * dx * dudx2 + dy * dy * dudy2)
                       + dx * dy * dudxdy;
      const size_t idx = (factor * ix + jx) * sx
                       + (factor * iy + jy) * sy;
      out[idx] = value;
    }
  }
}

template <typename Block, typename Lab, typename Int, typename T>
static void upscaleBlock3D(
    Lab &lab, Int _factor, size_t sx, size_t sy, size_t sz, T * __restrict__ out)
{
  using Real = typename Block::RealType;
  const int factor = _factor;
  for (int iz = 0; iz < Block::sizeZ; ++iz)
  for (int iy = 0; iy < Block::sizeY; ++iy)
  for (int ix = 0; ix < Block::sizeX; ++ix) {
    const auto dudx = (Real)0.5 * (lab(ix + 1, iy, iz) - lab(ix - 1, iy, iz));
    const auto dudy = (Real)0.5 * (lab(ix, iy + 1, iz) - lab(ix, iy - 1, iz));
    const auto dudz = (Real)0.5 * (lab(ix, iy, iz + 1) - lab(ix, iy, iz - 1));
    const auto dudx2 = (lab(ix + 1, iy, iz) + lab(ix - 1, iy, iz)) - 2 * lab(ix, iy, iz);
    const auto dudy2 = (lab(ix, iy + 1, iz) + lab(ix, iy - 1, iz)) - 2 * lab(ix, iy, iz);
    const auto dudz2 = (lab(ix, iy, iz + 1) + lab(ix, iy, iz - 1)) - 2 * lab(ix, iy, iz);
    const auto dudxdy = (Real)0.25 * ((lab(ix + 1, iy + 1, iz) + lab(ix - 1, iy - 1, iz))
                                    - (lab(ix + 1, iy - 1, iz) + lab(ix - 1, iy + 1, iz)));
    const auto dudydz = (Real)0.25 * ((lab(ix, iy + 1, iz + 1) + lab(ix, iy - 1, iz - 1))
                                    - (lab(ix, iy + 1, iz - 1) + lab(ix, iy - 1, iz + 1)));
    const auto dudzdx = (Real)0.25 * ((lab(ix + 1, iy, iz + 1) + lab(ix - 1, iy, iz - 1))
                                    - (lab(ix - 1, iy, iz + 1) + lab(ix + 1, iy, iz - 1)));

    // This for-loop will be expanded at compile time for factor == 1, 2, 4.
    const Real invF = (Real)0.5 / factor;
    for (int jz = 0; jz < factor; ++jz)
    for (int jy = 0; jy < factor; ++jy)
    for (int jx = 0; jx < factor; ++jx) {
      const Real dx = (-factor + 1 + 2 * jx) * invF;
      const Real dy = (-factor + 1 + 2 * jy) * invF;
      const Real dz = (-factor + 1 + 2 * jz) * invF;
      const auto value = lab(ix, iy, iz)
                       + (dx * dudx + dy * dudy + dz * dudz)
                       + (Real)0.5 * (dx * dx * dudx2 + dy * dy * dudy2 + dz * dz * dudz2)
                       + (dx * dy * dudxdy + dy * dz * dudydz + dz * dx * dudzdx);
      const size_t idx = (factor * ix + jx) * sx
                       + (factor * iy + jy) * sy
                       + (factor * iz + jz) * sz;
      out[idx] = value;
    }
  }
}

/// Dispatch to interpolation functions depending on the block level.
template <typename Block, typename Lab, typename T>
static void dispatchBlockInterpolation(
    Lab &lab,
    const BlockInfo &info,
    int levelMax,
    size_t sx,
    size_t sy,
    size_t sz,
    T * __restrict__ out)
{
  const int dLevel = levelMax - 1 - info.level;
  const int zOffset = DIMENSION == 3 ? info.index[2] * (Block::sizeZ << dLevel) : 0;
  const int yOffset = info.index[1] * (Block::sizeY << dLevel);
  const int xOffset = info.index[0] * (Block::sizeX << dLevel);
  T * const outSubmatrix = out + xOffset * sx + yOffset * sy + zOffset * sz;

  if constexpr (DIMENSION == 2) {
      switch (dLevel) {
        case 0: copyBlock<Block>(lab, sx, sy, sz, outSubmatrix); break;
        case 1: upscaleBlock2D<Block>(lab, ConstInt<2>{}, sx, sy, outSubmatrix); break;
        case 2: upscaleBlock2D<Block>(lab, ConstInt<4>{}, sx, sy, outSubmatrix); break;
        default:
            assert(dLevel >= 3);
            upscaleBlock2D<Block>(lab, 1 << dLevel, sx, sy, outSubmatrix);
      }
  } else {
      switch (dLevel) {
        case 0: copyBlock<Block>(lab, sx, sy, sz, outSubmatrix); break;
        case 1: upscaleBlock3D<Block>(lab, ConstInt<2>{}, sx, sy, sz, outSubmatrix); break;
        case 2: upscaleBlock3D<Block>(lab, ConstInt<4>{}, sx, sy, sz, outSubmatrix); break;
        default:
            assert(dLevel >= 3);
            upscaleBlock3D<Block>(lab, 1 << dLevel, sx, sy, sz, outSubmatrix);
      }
  }
}

namespace {

template <typename Lab, typename Getter>
struct WrappedLab
{
    Lab &lab;
    Getter getter;
    decltype(auto) operator()(int ix, int iy, int iz = 0) const
    {
        return getter(lab(ix, iy, iz));
    }
};

template <int kDim, typename Grid, typename Getter, typename T>
struct ExportKernel
{
  static_assert(kDim == 2 || kDim == 3);
  using Block = typename Grid::BlockType;

  Getter getter;
  StencilInfo stencil;
  int levelMax;
  size_t yStride;
  size_t zStride;
  T *out;

  ExportKernel(Grid *grid, Getter _getter, std::vector<int> components, T *_out) :
      getter{_getter},
      stencil{-1, -1, kDim == 3 ? -1 : 0,
              2, 2, kDim == 3 ? 2 : 1, true, std::move(components)},
      levelMax{grid->levelMax},
      out{_out}
  {
    const auto numBlocks = grid->getMaxMostRefinedBlocks();
    yStride = Block::sizeX * numBlocks[0];
    zStride = kDim == 3 ? yStride * Block::sizeY * numBlocks[1] : 0;
  }

  template <typename Lab>
  void operator()(Lab &lab, const BlockInfo &info) const
  {
    // Setting xStrides == 1 here should help the compiler optimize it out.
    WrappedLab<Lab, Getter> wrappedLab{lab, getter};
    dispatchBlockInterpolation<Block>(
        wrappedLab, info, levelMax, 1, yStride, kDim == 3 ? zStride : 0, out);
  }
};

}  // anonymous namespace

}  // namespace ie_detail

template <typename Lab, typename Grid, typename Getter, typename T>
void exportGridToUniformMatrix(
    Grid *grid, Getter getter, std::vector<int> components, T * __restrict__ out)
{
  ie_detail::ExportKernel<DIMENSION, Grid, Getter, T> kernel{
	    grid, getter, std::move(components), out};
  compute<Lab>(kernel, grid);
}

template <typename Grid, typename Getter, typename T>
void exportGridToUniformMatrixNearestInterpolation(
    Grid *grid, Getter getter, T * __restrict__ out)
{
  using Block = typename Grid::BlockType;
  const auto C = grid->getMaxMostRefinedCells();
  constexpr int xStride = 1;
  const int yStride = C[0];
  const int zStride = DIMENSION == 3 ? C[0] * C[1] : 0;

#pragma omp parallel for
  for (const BlockInfo &info : grid->getBlocksInfo()) {
    const Block & __restrict__ block = *(Block *)info.ptrBlock;
    const int factor = 1 << ((grid->levelMax - 1) - info.level);

    for (int iz = 0; iz < Block::sizeZ; ++iz)
    for (int iy = 0; iy < Block::sizeY; ++iy)
    for (int ix = 0; ix < Block::sizeX; ++ix) {
      const T value = getter(block(ix, iy, iz));

      const int zOffset = DIMENSION == 3 ? (info.index[2] * Block::sizeZ + iz) * factor : 0;
      const int yOffset = (info.index[1] * Block::sizeY + iy) * factor;
      const int xOffset = (info.index[0] * Block::sizeX + ix) * factor;

      for (int jz = 0; jz < (DIMENSION == 3 ? factor : 1); ++jz)
      for (int jy = 0; jy < factor; ++jy)
      for (int jx = 0; jx < factor; ++jx)
      {
        const int kz = zOffset + jz;
        const int ky = yOffset + jy;
        const int kx = xOffset + jx;
        const int idx = kx * xStride + ky * yStride + kz * zStride;
        assert(idx < C[0] * C[1] * C[2]);
        out[idx] = value;
      }
    }
  }
}


template <typename Grid, typename Setter, typename T>
void importGridFromUniformMatrix(
    Grid *grid, Setter setter, const T * __restrict__ in)
{
  using Block = typename Grid::BlockType;
  constexpr int BX = Block::sizeX;
  constexpr int BY = Block::sizeY;
  constexpr int BZ = Block::sizeZ;
  const auto C = grid->getMaxMostRefinedCells();
  constexpr int xStride = 1;
  const int yStride = C[0];
  const int zStride = DIMENSION == 3 ? C[0] * C[1] : 0;

#pragma omp parallel for
  for (const BlockInfo &info : grid->getBlocksInfo()) {
    Block & __restrict__ block = *(Block *)info.ptrBlock;
    const int factor = 1 << ((grid->levelMax - 1) - info.level);
    const auto avgFactor = (typename Block::ElementType::RealType)1
                         / (factor * factor * (DIMENSION == 3 ? factor : 1));

    for (int iz = 0; iz < BZ; ++iz)
    for (int iy = 0; iy < BY; ++iy)
    for (int ix = 0; ix < BX; ++ix)
    {
      const int zOffset = DIMENSION == 3 ? (info.index[2] * BZ + iz) * factor : 0;
      const int yOffset = (info.index[1] * BY + iy) * factor;
      const int xOffset = (info.index[0] * BX + ix) * factor;

      T sum{};
      for (int jz = 0; jz < (DIMENSION == 3 ? factor : 1); ++jz)
      for (int jy = 0; jy < factor; ++jy)
      for (int jx = 0; jx < factor; ++jx)
      {
        const int kz = zOffset + jz;
        const int ky = yOffset + jy;
        const int kx = xOffset + jx;
        const int idx = kx * xStride + ky * yStride + kz * zStride;
        assert(idx < C[0] * C[1] * C[2]);
        sum += in[idx];
      }

      setter(block(ix, iy, iz), avgFactor * sum);
    }
  }
}

}  // namespace cubism
