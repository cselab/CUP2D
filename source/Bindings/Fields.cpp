#include "Fields.h"
#include "../Definitions.h"
#include "../SimulationData.h"
#include <Cubism/Grid.hh>
#include <pybind11/numpy.h>

// Note: using "field" nomenclature instead of "grid" because "field" sounds
// like it is carrying the data and "grid" only the metadata.

using namespace pybind11::literals;

namespace cubismup2d {
namespace {

constexpr int BS = CUP2D_BLOCK_SIZE;

/** Returns '<' if little-endian is used or '>' otherwise.

    https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_interface__
*/
static char getEndiannessNumpy()
{
    uint16_t x = (uint16_t)'<' | (uint16_t)((uint16_t)'>' << 16);
    char out[2];
    memcpy(out, &x, 2);
    return out[0];
}

/// Return numpy typestr for the type T.
template <typename T>
static std::string floatTypestr()
{
  char str[8];
  str[0] = getEndiannessNumpy();
  str[1] = 'f';
  constexpr int size = (int)sizeof(T);
  if (size < 10) {
    str[2] = '0' + size;
    str[3] = '\0';
  } else if (size < 100) {
    str[2] = '0' + size / 10;
    str[3] = '0' + size % 10;
    str[4] = '\0';
  } else {
    throw std::runtime_error("sizeof(T) too large, not supported");
  }
  return str;
}

/// View to a block, ports its content to a numpy array.
struct BlockView
{
  cubism::BlockInfo *block;
  bool isVector;

  auto str() const
  {
    return py::str("BlockView({}x{} {}, ID={}, ij=({}, {}), level={}, Z={})")
        .format(BS, BS, isVector ? "vector" : "scalar",
                block->blockID, block->index[0], block->index[1],
                block->level, block->Z);
  }

  /// Used by numpy.asarray & co.
  py::dict arrayInterface() const
  {
    // We assume the block has no member variables other than the 2D array.
    return py::dict(
        "shape"_a = isVector ? py::make_tuple(BS, BS, 2)
                             : py::make_tuple(BS, BS),
        "typestr"_a = floatTypestr<Real>(),
        "data"_a = py::make_tuple((uintptr_t)block->ptrBlock,
                                  /* readOnly */ false),
        "version"_a = 3);
  }

  py::tuple ij() const
  {
    assert(block->index[2] == 0);
    return py::make_tuple(block->index[0], block->index[1]);
  }

  int level() const
  {
    return block->level;
  }
};

}  // anonymous namespace

template <typename Grid>
static py::array_t<Real> gridToUniform(const Grid &grid)
{
  static constexpr bool kIsVector = std::is_same_v<Grid, VectorGrid>;
  using T = typename Grid::BlockType::ElementType;
  static_assert(sizeof(T) == (kIsVector ? 2 : 1) * sizeof(Real), "");

  const auto numCells = grid.getMaxMostRefinedCells();
  std::vector<ssize_t> shape(2 + kIsVector);  // (y, x, [channels])
  shape[0] = numCells[1];
  shape[1] = numCells[0];
  if (kIsVector)
    shape[2] = 2;

  py::array_t<Real> out(std::move(shape));
  grid.copyToUniformNoInterpolation(reinterpret_cast<T *>(out.mutable_data()));
  return out;
}

template <typename Grid>
static void gridLoadUniform(Grid *grid, py::array_t<Real> array)
{
  static constexpr bool kIsVector = std::is_same_v<Grid, VectorGrid>;
  using T = typename Grid::BlockType::ElementType;
  static_assert(sizeof(T) == (kIsVector ? 2 : 1) * sizeof(Real), "");

  const auto cells = grid->getMaxMostRefinedCells();
  const bool ok = array.ndim() == (kIsVector ? 3 : 2)
      && array.shape(0) == cells[1]
      && array.shape(1) == cells[0]
      && (!kIsVector || array.shape(2) == 2);
  if (!ok) {
    py::tuple shape(array.ndim());
    for (int i = 0; i < (int)array.ndim(); ++i)
      shape[i] = array.shape(i);
    py::tuple expected = kIsVector ? py::make_tuple(cells[1], cells[0], 2)
                                   : py::make_tuple(cells[1], cells[0]);
    throw py::type_error("expected shape {}, got {}"_s.format(expected, shape));
  }
  grid->copyFromMatrix(reinterpret_cast<const T *>(array.data()));
}

template <typename Grid>
static void bindGrid(py::module &m, const char *name)
{
  py::class_<Grid>(m, name)
    .def("__len__", [](const Grid &grid) { return grid.getBlocksInfo().size(); })
    .def("__getitem__", [](Grid *grid, size_t k) {
      static constexpr bool kIsVector = std::is_same_v<Grid, VectorGrid>;
      return BlockView{&grid->getBlocksInfo().at(k), kIsVector};
    })
    .def("to_uniform", &gridToUniform<Grid>)
    .def("load_uniform", &gridLoadUniform<Grid>);
}

void bindFields(py::module &m)
{
  py::class_<BlockView>(m, "BlockView")
    .def("__repr__", &BlockView::str)
    .def("__str__", &BlockView::str)
    .def_property_readonly("__array_interface__", &BlockView::arrayInterface)
    .def_property_readonly("ij", &BlockView::ij)
    .def_property_readonly("level", &BlockView::level);

  bindGrid<ScalarGrid>(m, "ScalarGrid");
  bindGrid<VectorGrid>(m, "VectorGrid");
}

}  // namespace cubismup2d
