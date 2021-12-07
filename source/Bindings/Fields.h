#include "Common.h"

struct SimulationData;

namespace cubismup2d {

struct FieldsView {
  SimulationData *s;
};

void bindFields(py::module &m);

}  // namespace cubismup2d
