#include "Helpers.h"
#include "Cubism/HDF5Dumper.h"
using namespace cubism;
void IC::operator()(const Real dt) {
  const std::vector<BlockInfo> &velInfo = sim.vel->getBlocksInfo();
  const std::vector<BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo> &presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo> &poldInfo = sim.pold->getBlocksInfo();
  const std::vector<BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo> &vOldInfo = sim.vOld->getBlocksInfo();
  if (not sim.bRestart) {
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      VectorBlock &VEL = *(VectorBlock *)velInfo[i].ptrBlock;
      VEL.clear();
      ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
      CHI.clear();
      ScalarBlock &PRES = *(ScalarBlock *)presInfo[i].ptrBlock;
      PRES.clear();
      ScalarBlock &POLD = *(ScalarBlock *)poldInfo[i].ptrBlock;
      POLD.clear();
      ScalarBlock &TMP = *(ScalarBlock *)tmpInfo[i].ptrBlock;
      TMP.clear();
      VectorBlock &TMPV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
      TMPV.clear();
      VectorBlock &VOLD = *(VectorBlock *)vOldInfo[i].ptrBlock;
      VOLD.clear();
    }
    if (sim.smagorinskyCoeff != 0) {
      const std::vector<BlockInfo> &CsInfo = sim.Cs->getBlocksInfo();
#pragma omp parallel for
      for (size_t i = 0; i < CsInfo.size(); i++) {
        ScalarBlock &CS = *(ScalarBlock *)CsInfo[i].ptrBlock;
        for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
          for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
            CS(ix, iy).s = sim.smagorinskyCoeff;
          }
      }
    }
  } else {
    assert(0);
  }
}
Real findMaxU::run() const {
  const size_t Nblocks = velInfo.size();
  const Real UINF = sim.uinfx, VINF = sim.uinfy;
  Real U = 0, V = 0, u = 0, v = 0;
#pragma omp parallel for schedule(static) reduction(max : U, V, u, v)
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &VEL = *(VectorBlock *)velInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        U = std::max(U, std::fabs(VEL(ix, iy).u[0] + UINF));
        V = std::max(V, std::fabs(VEL(ix, iy).u[1] + VINF));
        u = std::max(u, std::fabs(VEL(ix, iy).u[0]));
        v = std::max(v, std::fabs(VEL(ix, iy).u[1]));
      }
  }
  Real quantities[4] = {U, V, u, v};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 4, MPI_Real, MPI_MAX,
                sim.chi->getWorldComm());
  U = quantities[0];
  V = quantities[1];
  u = quantities[2];
  v = quantities[3];
  return std::max({U, V, u, v});
}
void ApplyObjVel::operator()(const Real dt) {
  const size_t Nblocks = velInfo.size();
  const std::vector<BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ((VectorBlock *)tmpVInfo[i].ptrBlock)->clear();
  }
  for (const auto &shape : sim.shapes) {
    const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      if (OBLOCK[tmpVInfo[i].blockID] == nullptr)
        continue;
      const UDEFMAT &__restrict__ udef = OBLOCK[tmpVInfo[i].blockID]->udef;
      const CHI_MAT &__restrict__ chi = OBLOCK[tmpVInfo[i].blockID]->chi;
      auto &__restrict__ UDEF = *(VectorBlock *)tmpVInfo[i].ptrBlock;
      const ScalarBlock &__restrict__ CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
      for (int iy = 0; iy < VectorBlock::sizeY; iy++)
        for (int ix = 0; ix < VectorBlock::sizeX; ix++) {
          if (chi[iy][ix] < CHI(ix, iy).s)
            continue;
          Real p[2];
          tmpVInfo[i].pos(p, ix, iy);
          UDEF(ix, iy).u[0] += udef[iy][ix][0];
          UDEF(ix, iy).u[1] += udef[iy][ix][1];
        }
    }
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &UF = *(VectorBlock *)velInfo[i].ptrBlock;
    VectorBlock &US = *(VectorBlock *)tmpVInfo[i].ptrBlock;
    ScalarBlock &X = *(ScalarBlock *)chiInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        UF(ix, iy).u[0] =
            UF(ix, iy).u[0] * (1 - X(ix, iy).s) + US(ix, iy).u[0] * X(ix, iy).s;
        UF(ix, iy).u[1] =
            UF(ix, iy).u[1] * (1 - X(ix, iy).s) + US(ix, iy).u[1] * X(ix, iy).s;
      }
  }
}
