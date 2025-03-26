#pragma once
#include "AMR_SynchronizerMPI.h"
#include "BlockLab.h"
namespace cubism {
template <typename MyBlockLab> class BlockLabMPI : public MyBlockLab {
public:
  using GridType = typename MyBlockLab::GridType;
  using BlockType = typename GridType::BlockType;
  using ElementType = typename BlockType::ElementType;
  using Real = typename ElementType::RealType;

private:
  typedef SynchronizerMPI_AMR<Real, GridType> SynchronizerMPIType;
  SynchronizerMPIType *refSynchronizerMPI;

public:
  virtual void prepare(GridType &grid, const StencilInfo &stencil,
                       const int Istencil_start[3] = default_start,
                       const int Istencil_end[3] = default_end) override {
    auto itSynchronizerMPI = grid.SynchronizerMPIs.find(stencil);
    refSynchronizerMPI = itSynchronizerMPI->second;
    MyBlockLab::prepare(grid, stencil);
  }
  virtual void load(const BlockInfo &info, const Real t = 0,
                    const bool applybc = true) override {
    MyBlockLab::load(info, t, applybc);
    Real *dst = (Real *)&MyBlockLab ::m_cacheBlock->LinAccess(0);
    Real *dst1 = (Real *)&MyBlockLab ::m_CoarsenedBlock->LinAccess(0);
    refSynchronizerMPI->fetch(info, MyBlockLab::m_cacheBlock->getSize(),
                              MyBlockLab::m_CoarsenedBlock->getSize(), dst,
                              dst1);
    if (MyBlockLab::m_refGrid->get_world_size() > 1)
      MyBlockLab::post_load(info, t, applybc);
  }
};
} // namespace cubism
