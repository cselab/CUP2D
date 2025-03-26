#include <vector>

namespace cubism {

template <typename Lab, typename Grid, typename Getter, typename T>
void exportGridToUniformMatrix(Grid *grid, Getter getter,
                               std::vector<int> components,
                               T *__restrict__ out);

template <typename Grid, typename Getter, typename T>
void exportGridToUniformMatrixNearestInterpolation(Grid *grid, Getter getter,
                                                   T *__restrict__ out);

template <typename Grid, typename Setter, typename T>
void importGridFromUniformMatrix(Grid *grid, Setter setter,
                                 const T *__restrict__ in);

} // namespace cubism
