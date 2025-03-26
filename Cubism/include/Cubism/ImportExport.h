#include <vector>

namespace cubism
{

/** Fully refine the AMR grid and store into the given contiguous matrix.

    The output matrix in expected to have xStride == 1,
    yStride == xNumCells and zStride == xNumCells * yNumCells.

    Blocks already at the highest level of refinement are copied without
    interpolation. All other blocks are upscaled using interpolation that is
    quadratic everywhere except at the coarse-fine boundary, where it is
    linear.

    Template arguments:
      Lab: block lab used for the stencil for the upscaling interpolation

    Arguments:
      grid: AMR MPI grid
      getter: Function that takes an element and returns an arbitrary
          arithmetic type T. For example, it can be used to select only one
          element of a vector element. The upscaling interpolation is performed
          after the transformation.
      components: components of the element that need to be exchanged for ghost
          cells
      out: output matrix
*/
template <typename Lab, typename Grid, typename Getter, typename T>
void exportGridToUniformMatrix(
    Grid *grid, Getter getter, std::vector<int> components, T * __restrict__ out);


/** Copy the grid to a uniform grid matrix, scaling where necessary.

    Blocks that are not at the most refined level are scaled with
    no interpolation, i.e. with nearest/constant interpolation.

    See `exportGridToUniformMatrix` for more information. Note that this
    function requires no lab nor the `components` list, since no stencils are
    involved.
*/
template <typename Grid, typename Getter, typename T>
void exportGridToUniformMatrixNearestInterpolation(
    Grid *grid, Getter getter, T * __restrict__ out);


/** Import data from a large contiguous matrix representing the fully refined
    grid.

    The grid structure is left unchanged. The values of cells at level lower
    than maxLevel-1 are computed by averaging the corresponding input cells.

    Arguments:
      grid: the AMR grid
      setter: A function that takes the reference to the cell element and the
          (averaged) input cell value, and updates the cell element in the
          desired fashion. E.g. it can be used to update only one component of
          a vector element.
      in: the input contiguous matrix
*/
template <typename Grid, typename Setter, typename T>
void importGridFromUniformMatrix(
    Grid *grid, Setter setter, const T * __restrict__ in);

}  // namespace cubism
