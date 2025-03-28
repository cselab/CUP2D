#include "config.h"
#include "Forcing.h"
using namespace cubism;
void Forcing::operator()(const Real dt) {
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++) {
    VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        const auto pos = velInfo[i].pos<Real>(ix, iy);
        V(ix, iy).u[0] += dt * sim.forcingCoefficient *
                          std::sin(2 * M_PI * sim.forcingWavenumber * pos[1] /
                                   sim.extents[1]);
      }
  }
}
