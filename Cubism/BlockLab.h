#pragma once
#include "BlockInfo.h"
#include "Matrix3D.h"
#include "StencilInfo.h"
#include <array>
#include <cstring>
#include <math.h>
#include <string>
namespace cubism {
#define memcpy2(a, b, c) memcpy((a), (b), (c))
constexpr int default_start[3] = {-1, -1, 0};
constexpr int default_end[3] = {2, 2, 1};
template <typename TGrid,
          template <typename X> class allocator = std::allocator>
class BlockLab {
public:
  using GridType = TGrid;
  using BlockType = typename GridType::BlockType;
  using ElementType = typename BlockType::ElementType;
  using Real = typename ElementType::RealType;

protected:
  Matrix3D<ElementType, allocator> *m_cacheBlock;
  int m_stencilStart[3];
  int m_stencilEnd[3];
  bool istensorial;
  bool use_averages;
  GridType *m_refGrid;
  int NX;
  int NY;
  int NZ;
  std::array<BlockType *, 27> myblocks;
  std::array<int, 27> coarsened_nei_codes;
  int coarsened_nei_codes_size;
  int offset[3];
  Matrix3D<ElementType, allocator> *m_CoarsenedBlock;
  int m_InterpStencilStart[3];
  int m_InterpStencilEnd[3];
  bool coarsened;
  int CoarseBlockSize[3];
  const double d_coef_plus[9] = {-0.09375, 0.4375,   0.15625, 0.15625, -0.5625,
                                 0.90625,  -0.09375, 0.4375,  0.15625};
  const double d_coef_minus[9] = {0.15625, -0.5625, 0.90625, -0.09375, 0.4375,
                                  0.15625, 0.15625, 0.4375,  -0.09375};

public:
  BlockLab()
      : m_cacheBlock(nullptr), m_refGrid(nullptr), m_CoarsenedBlock(nullptr) {
    m_stencilStart[0] = m_stencilStart[1] = m_stencilStart[2] = 0;
    m_stencilEnd[0] = m_stencilEnd[1] = m_stencilEnd[2] = 0;
    m_InterpStencilStart[0] = m_InterpStencilStart[1] =
        m_InterpStencilStart[2] = 0;
    m_InterpStencilEnd[0] = m_InterpStencilEnd[1] = m_InterpStencilEnd[2] = 0;
    CoarseBlockSize[0] = (int)BlockType::sizeX / 2;
    CoarseBlockSize[1] = (int)BlockType::sizeY / 2;
    CoarseBlockSize[2] = (int)BlockType::sizeZ / 2;
    if (CoarseBlockSize[0] == 0)
      CoarseBlockSize[0] = 1;
    if (CoarseBlockSize[1] == 0)
      CoarseBlockSize[1] = 1;
    if (CoarseBlockSize[2] == 0)
      CoarseBlockSize[2] = 1;
  }
  virtual std::string name() const { return "BlockLab"; }
  virtual bool is_xperiodic() { return true; }
  virtual bool is_yperiodic() { return true; }
  virtual bool is_zperiodic() { return true; }
  ~BlockLab() {
    _release(m_cacheBlock);
    _release(m_CoarsenedBlock);
  }
  ElementType &operator()(int ix, int iy = 0, int iz = 0) {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }
  const ElementType &operator()(int ix, int iy = 0, int iz = 0) const {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }
  const ElementType &read(int ix, int iy = 0, int iz = 0) const {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }
  void release() {
    _release(m_cacheBlock);
    _release(m_CoarsenedBlock);
  }
  virtual void prepare(GridType &grid, const StencilInfo &stencil,
                       const int Istencil_start[3] = default_start,
                       const int Istencil_end[3] = default_end) {
    istensorial = stencil.tensorial;
    coarsened = false;
    m_stencilStart[0] = stencil.sx;
    m_stencilStart[1] = stencil.sy;
    m_stencilStart[2] = stencil.sz;
    m_stencilEnd[0] = stencil.ex;
    m_stencilEnd[1] = stencil.ey;
    m_stencilEnd[2] = stencil.ez;
    m_InterpStencilStart[0] = Istencil_start[0];
    m_InterpStencilStart[1] = Istencil_start[1];
    m_InterpStencilStart[2] = Istencil_start[2];
    m_InterpStencilEnd[0] = Istencil_end[0];
    m_InterpStencilEnd[1] = Istencil_end[1];
    m_InterpStencilEnd[2] = Istencil_end[2];
    assert(m_InterpStencilStart[0] <= m_InterpStencilEnd[0]);
    assert(m_InterpStencilStart[1] <= m_InterpStencilEnd[1]);
    assert(m_InterpStencilStart[2] <= m_InterpStencilEnd[2]);
    assert(stencil.sx <= stencil.ex);
    assert(stencil.sy <= stencil.ey);
    assert(stencil.sz <= stencil.ez);
    assert(stencil.sx >= -BlockType::sizeX);
    assert(stencil.sy >= -BlockType::sizeY);
    assert(stencil.sz >= -BlockType::sizeZ);
    assert(stencil.ex < 2 * BlockType::sizeX);
    assert(stencil.ey < 2 * BlockType::sizeY);
    assert(stencil.ez < 2 * BlockType::sizeZ);
    m_refGrid = &grid;
    if (m_cacheBlock == NULL ||
        (int)m_cacheBlock->getSize()[0] !=
            (int)BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1 ||
        (int)m_cacheBlock->getSize()[1] !=
            (int)BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1 ||
        (int)m_cacheBlock->getSize()[2] !=
            (int)BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1) {
      if (m_cacheBlock != NULL)
        _release(m_cacheBlock);
      m_cacheBlock = allocator<Matrix3D<ElementType, allocator>>().allocate(1);
      allocator<Matrix3D<ElementType, allocator>>().construct(m_cacheBlock);
      m_cacheBlock->_Setup(
          BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1,
          BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1,
          BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1);
    }
    offset[0] = (m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0];
    offset[1] = (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1];
    offset[2] = (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2];
    const int e[3] = {(m_stencilEnd[0]) / 2 + 1 + m_InterpStencilEnd[0] - 1,
                      (m_stencilEnd[1]) / 2 + 1 + m_InterpStencilEnd[1] - 1,
                      (m_stencilEnd[2]) / 2 + 1 + m_InterpStencilEnd[2] - 1};
    if (m_CoarsenedBlock == NULL ||
        (int)m_CoarsenedBlock->getSize()[0] !=
            CoarseBlockSize[0] + e[0] - offset[0] - 1 ||
        (int)m_CoarsenedBlock->getSize()[1] !=
            CoarseBlockSize[1] + e[1] - offset[1] - 1 ||
        (int)m_CoarsenedBlock->getSize()[2] !=
            CoarseBlockSize[2] + e[2] - offset[2] - 1) {
      if (m_CoarsenedBlock != NULL)
        _release(m_CoarsenedBlock);
      m_CoarsenedBlock =
          allocator<Matrix3D<ElementType, allocator>>().allocate(1);
      allocator<Matrix3D<ElementType, allocator>>().construct(m_CoarsenedBlock);
      m_CoarsenedBlock->_Setup(CoarseBlockSize[0] + e[0] - offset[0] - 1,
                               CoarseBlockSize[1] + e[1] - offset[1] - 1,
                               CoarseBlockSize[2] + e[2] - offset[2] - 1);
    }
    use_averages = (m_refGrid->FiniteDifferences == false || istensorial ||
                    m_stencilStart[0] < -2 || m_stencilStart[1] < -2 ||
                    m_stencilEnd[0] > 3 || m_stencilEnd[1] > 3);
  }
  virtual void load(const BlockInfo &info, const Real t = 0,
                    const bool applybc = true) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const bool xperiodic = is_xperiodic();
    const bool yperiodic = is_yperiodic();
    const bool zperiodic = is_zperiodic();
    std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    const int aux = 1 << info.level;
    NX = blocksPerDim[0] * aux;
    NY = blocksPerDim[1] * aux;
    NZ = blocksPerDim[2] * aux;
    assert(m_cacheBlock != NULL);
    {
      BlockType &block = *(BlockType *)info.ptrBlock;
      ElementType *ptrSource = &block(0);
#if 0
            for(int iz=0; iz<nZ; iz++)
            for(int iy=0; iy<nY; iy++)
            {
              ElementType * ptrDestination = &m_cacheBlock->Access(0-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
              memcpy2((char *)ptrDestination, (char *)ptrSource, sizeof(ElementType)*nX);
              ptrSource+= nX;
            }
#else
      const int nbytes = sizeof(ElementType) * nX;
      const int _iz0 = -m_stencilStart[2];
      const int _iz1 = _iz0 + nZ;
      const int _iy0 = -m_stencilStart[1];
      const int _iy1 = _iy0 + nY;
      const int m_vSize0 = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
      const int my_ix = -m_stencilStart[0];
#pragma GCC ivdep
      for (int iz = _iz0; iz < _iz1; iz++) {
        const int my_izx = iz * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
        for (int iy = _iy0; iy < _iy1; iy += 4) {
          ElementType *__restrict__ ptrDestination0 =
              &m_cacheBlock->LinAccess(my_izx + (iy)*m_vSize0);
          ElementType *__restrict__ ptrDestination1 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 1) * m_vSize0);
          ElementType *__restrict__ ptrDestination2 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 2) * m_vSize0);
          ElementType *__restrict__ ptrDestination3 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 3) * m_vSize0);
          memcpy2(ptrDestination0, (ptrSource), nbytes);
          memcpy2(ptrDestination1, (ptrSource + nX), nbytes);
          memcpy2(ptrDestination2, (ptrSource + 2 * nX), nbytes);
          memcpy2(ptrDestination3, (ptrSource + 3 * nX), nbytes);
          ptrSource += 4 * nX;
        }
      }
#endif
    }
    {
      coarsened = false;
      const bool xskin = info.index[0] == 0 || info.index[0] == NX - 1;
      const bool yskin = info.index[1] == 0 || info.index[1] == NY - 1;
      const bool zskin = info.index[2] == 0 || info.index[2] == NZ - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      int icodes[DIMENSION == 2 ? 8 : 26];
      int k = 0;
      coarsened_nei_codes_size = 0;
      for (int icode = (DIMENSION == 2 ? 9 : 0);
           icode < (DIMENSION == 2 ? 18 : 27); icode++) {
        myblocks[icode] = nullptr;
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
        if (!xperiodic && code[0] == xskip && xskin)
          continue;
        if (!yperiodic && code[1] == yskip && yskin)
          continue;
        if (!zperiodic && code[2] == zskip && zskin)
          continue;
        const auto &TreeNei =
            m_refGrid->Tree(info.level, info.Znei_(code[0], code[1], code[2]));
        if (TreeNei.Exists()) {
          icodes[k++] = icode;
        } else if (TreeNei.CheckCoarser()) {
          coarsened_nei_codes[coarsened_nei_codes_size++] = icode;
          CoarseFineExchange(info, code);
        }
        if (!istensorial && !use_averages &&
            abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
          continue;
        const int s[3] = {
            code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
            code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
            code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
        const int e[3] = {
            code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
            code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
            code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};
        if (TreeNei.Exists())
          SameLevelExchange(info, code, s, e);
        else if (TreeNei.CheckFiner())
          FineToCoarseExchange(info, code, s, e);
      }
      if (coarsened_nei_codes_size > 0)
        for (int i = 0; i < k; ++i) {
          const int icode = icodes[i];
          const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                               icode / 9 - 1};
          const int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                        (info.index[1] + code[1] + NY) % NY,
                                        (info.index[2] + code[2] + NZ) % NZ};
          if (UseCoarseStencil(info, infoNei_index)) {
            FillCoarseVersion(info, code);
            coarsened = true;
          }
        }
      if (m_refGrid->get_world_size() == 1) {
        post_load(info, t, applybc);
      }
    }
  }

protected:
  void post_load(const BlockInfo &info, const Real t = 0, bool applybc = true) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    if (coarsened) {
#pragma GCC ivdep
      for (int j = 0; j < nY / 2; j++) {
#pragma GCC ivdep
        for (int i = 0; i < nX / 2; i++) {
          if (i > -m_InterpStencilStart[0] &&
              i < nX / 2 - m_InterpStencilEnd[0] &&
              j > -m_InterpStencilStart[1] &&
              j < nY / 2 - m_InterpStencilEnd[1])
            continue;
          const int ix = 2 * i - m_stencilStart[0];
          const int iy = 2 * j - m_stencilStart[1];
          ElementType &coarseElement =
              m_CoarsenedBlock->Access(i - offset[0], j - offset[1], 0);
          coarseElement = AverageDown(m_cacheBlock->Read(ix, iy, 0),
                                      m_cacheBlock->Read(ix + 1, iy, 0),
                                      m_cacheBlock->Read(ix, iy + 1, 0),
                                      m_cacheBlock->Read(ix + 1, iy + 1, 0));
        }
      }
    }
    if (applybc)
      _apply_bc(info, t, true);
    CoarseFineInterpolation(info);
    if (applybc)
      _apply_bc(info, t);
  }
  bool UseCoarseStencil(const BlockInfo &a, const int *b_index) {
    if (a.level == 0 || (!use_averages))
      return false;
    std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    int imin[3];
    int imax[3];
    const int aux = 1 << a.level;
    const bool periodic[3] = {is_xperiodic(), is_yperiodic(), is_zperiodic()};
    const int blocks[3] = {blocksPerDim[0] * aux - 1, blocksPerDim[1] * aux - 1,
                           blocksPerDim[2] * aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (a.index[d] < b_index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b_index[d]) ? 0 : +1;
      if (periodic[d]) {
        if (a.index[d] == 0 && b_index[d] == blocks[d])
          imin[d] = -1;
        if (b_index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b_index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b_index[d] == blocks[d])
          imax[d] = 0;
      }
    }
    for (int itest = 0; itest < coarsened_nei_codes_size; itest++)
      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
        for (int i1 = imin[1]; i1 <= imax[1]; i1++)
          for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
            const int icode_test = (i0 + 1) + 3 * (i1 + 1) + 9 * (i2 + 1);
            if (coarsened_nei_codes[itest] == icode_test)
              return true;
          }
    return false;
  }
  void SameLevelExchange(const BlockInfo &info, const int *const code,
                         const int *const s, const int *const e) {
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9 * (code[2] + 1);
    myblocks[icode] =
        m_refGrid->avail(info.level, info.Znei_(code[0], code[1], code[2]));
    if (myblocks[icode] == nullptr)
      return;
    const BlockType &b = *myblocks[icode];
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int m_vSize0 = m_cacheBlock->getSize(0);
    const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - m_stencilStart[0];
    const int mod = (e[1] - s[1]) % 4;
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int my_izx = (iz - m_stencilStart[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
        ElementType *__restrict__ ptrDest0 = &m_cacheBlock->LinAccess(
            my_izx + (iy - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest1 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 1 - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest2 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 2 - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest3 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 3 - m_stencilStart[1]) * m_vSize0);
        const ElementType *ptrSrc0 =
            &b(s[0] - code[0] * nX, iy - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc1 =
            &b(s[0] - code[0] * nX, iy + 1 - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc2 =
            &b(s[0] - code[0] * nX, iy + 2 - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc3 =
            &b(s[0] - code[0] * nX, iy + 3 - code[1] * nY, iz - code[2] * nZ);
        memcpy2(ptrDest0, ptrSrc0, bytes);
        memcpy2(ptrDest1, ptrSrc1, bytes);
        memcpy2(ptrDest2, ptrSrc2, bytes);
        memcpy2(ptrDest3, ptrSrc3, bytes);
      }
#pragma GCC ivdep
      for (int iy = e[1] - mod; iy < e[1]; iy++) {
        ElementType *__restrict__ ptrDest = &m_cacheBlock->LinAccess(
            my_izx + (iy - m_stencilStart[1]) * m_vSize0);
        const ElementType *ptrSrc =
            &b(s[0] - code[0] * nX, iy - code[1] * nY, iz - code[2] * nZ);
        memcpy2(ptrDest, ptrSrc, bytes);
      }
    }
  }
  ElementType AverageDown(const ElementType &e0, const ElementType &e1,
                          const ElementType &e2, const ElementType &e3) {
    return 0.25 * ((e0 + e3) + (e1 + e2));
  }
  void LI(ElementType &a, ElementType b, ElementType c) {
    auto kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
    auto lambda = (b - c) - kappa;
    a = (4.0 * kappa + 2.0 * lambda) + c;
  }
  void LE(ElementType &a, ElementType b, ElementType c) {
    auto kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
    auto lambda = (b - c) - kappa;
    a = (9.0 * kappa + 3.0 * lambda) + c;
  }
  virtual void TestInterp(ElementType *C[3][3], ElementType &R, int x, int y) {
    const double dx = 0.25 * (2 * x - 1);
    const double dy = 0.25 * (2 * y - 1);
    ElementType dudx = 0.5 * ((*C[2][1]) - (*C[0][1]));
    ElementType dudy = 0.5 * ((*C[1][2]) - (*C[1][0]));
    ElementType dudxdy =
        0.25 * (((*C[0][0]) + (*C[2][2])) - ((*C[2][0]) + (*C[0][2])));
    ElementType dudx2 = ((*C[0][1]) + (*C[2][1])) - 2.0 * (*C[1][1]);
    ElementType dudy2 = ((*C[1][0]) + (*C[1][2])) - 2.0 * (*C[1][1]);
    R = (*C[1][1] + (dx * dudx + dy * dudy)) +
        (((0.5 * dx * dx) * dudx2 + (0.5 * dy * dy) * dudy2) +
         (dx * dy) * dudxdy);
  }
  void FineToCoarseExchange(const BlockInfo &info, const int *const code,
                            const int *const s, const int *const e) {
    const int bytes = (abs(code[0]) * (e[0] - s[0]) +
                       (1 - abs(code[0])) * ((e[0] - s[0]) / 2)) *
                      sizeof(ElementType);
    if (!bytes)
      return;
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int m_vSize0 = m_cacheBlock->getSize(0);
    const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
    const int yStep = (code[1] == 0) ? 2 : 1;
    const int zStep = (code[2] == 0) ? 2 : 1;
    const int mod = ((e[1] - s[1]) / yStep) % 4;
    int Bstep = 1;
    if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
      Bstep = 3;
    else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
      Bstep = 4;
    for (int B = 0; B <= 3; B += Bstep) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      BlockType *b_ptr =
          m_refGrid->avail1(2 * info.index[0] + std::max(code[0], 0) + code[0] +
                                (B % 2) * std::max(0, 1 - abs(code[0])),
                            2 * info.index[1] + std::max(code[1], 0) + code[1] +
                                aux * std::max(0, 1 - abs(code[1])),
                            info.level + 1);
      if (b_ptr == nullptr)
        continue;
      BlockType &b = *b_ptr;
      const int my_ix = abs(code[0]) * (s[0] - m_stencilStart[0]) +
                        (1 - abs(code[0])) * (s[0] - m_stencilStart[0] +
                                              (B % 2) * (e[0] - s[0]) / 2);
      const int XX = s[0] - code[0] * nX + std::min(0, code[0]) * (e[0] - s[0]);
#pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz += zStep) {
        const int ZZ = (abs(code[2]) == 1)
                           ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ
                           : iz;
        const int my_izx =
            (abs(code[2]) * (iz - m_stencilStart[2]) +
             (1 - abs(code[2])) *
                 (iz / 2 - m_stencilStart[2] + (B / 2) * (e[2] - s[2]) / 2)) *
                m_nElemsPerSlice +
            my_ix;
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1] - mod; iy += 4 * yStep) {
          ElementType *__restrict__ ptrDest0 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 0 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 0 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest1 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 1 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 1 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest2 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 2 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 2 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest3 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 3 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 3 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          const int YY0 = (abs(code[1]) == 1)
                              ? 2 * (iy + 0 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 0 * yStep;
          const int YY1 = (abs(code[1]) == 1)
                              ? 2 * (iy + 1 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 1 * yStep;
          const int YY2 = (abs(code[1]) == 1)
                              ? 2 * (iy + 2 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 2 * yStep;
          const int YY3 = (abs(code[1]) == 1)
                              ? 2 * (iy + 3 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 3 * yStep;
          const ElementType *ptrSrc_00 = &b(XX, YY0, ZZ);
          const ElementType *ptrSrc_10 = &b(XX, YY0 + 1, ZZ);
          const ElementType *ptrSrc_01 = &b(XX, YY1, ZZ);
          const ElementType *ptrSrc_11 = &b(XX, YY1 + 1, ZZ);
          const ElementType *ptrSrc_02 = &b(XX, YY2, ZZ);
          const ElementType *ptrSrc_12 = &b(XX, YY2 + 1, ZZ);
          const ElementType *ptrSrc_03 = &b(XX, YY3, ZZ);
          const ElementType *ptrSrc_13 = &b(XX, YY3 + 1, ZZ);
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest0[ee] = AverageDown(
                *(ptrSrc_00 + 2 * ee), *(ptrSrc_10 + 2 * ee),
                *(ptrSrc_00 + 2 * ee + 1), *(ptrSrc_10 + 2 * ee + 1));
            ptrDest1[ee] = AverageDown(
                *(ptrSrc_01 + 2 * ee), *(ptrSrc_11 + 2 * ee),
                *(ptrSrc_01 + 2 * ee + 1), *(ptrSrc_11 + 2 * ee + 1));
            ptrDest2[ee] = AverageDown(
                *(ptrSrc_02 + 2 * ee), *(ptrSrc_12 + 2 * ee),
                *(ptrSrc_02 + 2 * ee + 1), *(ptrSrc_12 + 2 * ee + 1));
            ptrDest3[ee] = AverageDown(
                *(ptrSrc_03 + 2 * ee), *(ptrSrc_13 + 2 * ee),
                *(ptrSrc_03 + 2 * ee + 1), *(ptrSrc_13 + 2 * ee + 1));
          }
        }
#pragma GCC ivdep
        for (int iy = e[1] - mod; iy < e[1]; iy += yStep) {
          ElementType *ptrDest = (ElementType *)&m_cacheBlock->LinAccess(
              my_izx + (abs(code[1]) * (iy - m_stencilStart[1]) +
                        (1 - abs(code[1])) * (iy / 2 - m_stencilStart[1] +
                                              aux * (e[1] - s[1]) / 2)) *
                           m_vSize0);
          const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) +
                                                   std::min(0, code[1]) * nY
                                             : iy;
          const ElementType *ptrSrc_0 = &b(XX, YY, ZZ);
          const ElementType *ptrSrc_1 = &b(XX, YY + 1, ZZ);
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest[ee] =
                AverageDown(*(ptrSrc_0 + 2 * ee), *(ptrSrc_1 + 2 * ee),
                            *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1));
          }
        }
      }
    }
  }
  void CoarseFineExchange(const BlockInfo &info, const int *const code) {
    const int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                  (info.index[1] + code[1] + NY) % NY,
                                  (info.index[2] + code[2] + NZ) % NZ};
    const int infoNei_index_true[3] = {(info.index[0] + code[0]),
                                       (info.index[1] + code[1]),
                                       (info.index[2] + code[2])};
    BlockType *b_ptr = m_refGrid->avail1(
        (infoNei_index[0]) / 2, (infoNei_index[1]) / 2, info.level - 1);
    if (b_ptr == nullptr)
      return;
    const BlockType &b = *b_ptr;
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int s[3] = {
        code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};
    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0])
                                  : CoarseBlockSize[0] + (m_stencilEnd[0]) / 2 +
                                        m_InterpStencilEnd[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1])
                                  : CoarseBlockSize[1] + (m_stencilEnd[1]) / 2 +
                                        m_InterpStencilEnd[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2])
                                  : CoarseBlockSize[2] + (m_stencilEnd[2]) / 2 +
                                        m_InterpStencilEnd[2] - 1};
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int base[3] = {(info.index[0] + code[0]) % 2,
                         (info.index[1] + code[1]) % 2,
                         (info.index[2] + code[2]) % 2};
    int CoarseEdge[3];
    CoarseEdge[0] = (code[0] == 0) ? 0
                    : (((info.index[0] % 2 == 0) &&
                        (infoNei_index_true[0] > info.index[0])) ||
                       ((info.index[0] % 2 == 1) &&
                        (infoNei_index_true[0] < info.index[0])))
                        ? 1
                        : 0;
    CoarseEdge[1] = (code[1] == 0) ? 0
                    : (((info.index[1] % 2 == 0) &&
                        (infoNei_index_true[1] > info.index[1])) ||
                       ((info.index[1] % 2 == 1) &&
                        (infoNei_index_true[1] < info.index[1])))
                        ? 1
                        : 0;
    CoarseEdge[2] = (code[2] == 0) ? 0
                    : (((info.index[2] % 2 == 0) &&
                        (infoNei_index_true[2] > info.index[2])) ||
                       ((info.index[2] % 2 == 1) &&
                        (infoNei_index_true[2] < info.index[2])))
                        ? 1
                        : 0;
    const int start[3] = {
        std::max(code[0], 0) * nX / 2 + (1 - abs(code[0])) * base[0] * nX / 2 -
            code[0] * nX + CoarseEdge[0] * code[0] * nX / 2,
        std::max(code[1], 0) * nY / 2 + (1 - abs(code[1])) * base[1] * nY / 2 -
            code[1] * nY + CoarseEdge[1] * code[1] * nY / 2,
        std::max(code[2], 0) * nZ / 2 + (1 - abs(code[2])) * base[2] * nZ / 2 -
            code[2] * nZ + CoarseEdge[2] * code[2] * nZ / 2};
    const int m_vSize0 = m_CoarsenedBlock->getSize(0);
    const int m_nElemsPerSlice =
        m_CoarsenedBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - offset[0];
    const int mod = (e[1] - s[1]) % 4;
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
        ElementType *__restrict__ ptrDest0 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 0 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest1 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 1 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest2 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 2 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest3 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 3 - offset[1]) * m_vSize0);
        const ElementType *ptrSrc0 =
            &b(s[0] + start[0], iy + 0 + start[1], iz + start[2]);
        const ElementType *ptrSrc1 =
            &b(s[0] + start[0], iy + 1 + start[1], iz + start[2]);
        const ElementType *ptrSrc2 =
            &b(s[0] + start[0], iy + 2 + start[1], iz + start[2]);
        const ElementType *ptrSrc3 =
            &b(s[0] + start[0], iy + 3 + start[1], iz + start[2]);
        memcpy2(ptrDest0, ptrSrc0, bytes);
        memcpy2(ptrDest1, ptrSrc1, bytes);
        memcpy2(ptrDest2, ptrSrc2, bytes);
        memcpy2(ptrDest3, ptrSrc3, bytes);
      }
#pragma GCC ivdep
      for (int iy = e[1] - mod; iy < e[1]; iy++) {
        ElementType *ptrDest =
            &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
        const ElementType *ptrSrc =
            &b(s[0] + start[0], iy + start[1], iz + start[2]);
        memcpy2(ptrDest, ptrSrc, bytes);
      }
    }
  }
  void FillCoarseVersion(const BlockInfo &info, const int *const code) {
    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9 * (code[2] + 1);
    if (myblocks[icode] == nullptr)
      return;
    const BlockType &b = *myblocks[icode];
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int eC[3] = {(m_stencilEnd[0]) / 2 + m_InterpStencilEnd[0],
                       (m_stencilEnd[1]) / 2 + m_InterpStencilEnd[1],
                       (m_stencilEnd[2]) / 2 + m_InterpStencilEnd[2]};
    const int s[3] = {
        code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};
    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0])
                                  : CoarseBlockSize[0] + eC[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1])
                                  : CoarseBlockSize[1] + eC[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2])
                                  : CoarseBlockSize[2] + eC[2] - 1};
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int start[3] = {
        s[0] + std::max(code[0], 0) * CoarseBlockSize[0] - code[0] * nX +
            std::min(0, code[0]) * (e[0] - s[0]),
        s[1] + std::max(code[1], 0) * CoarseBlockSize[1] - code[1] * nY +
            std::min(0, code[1]) * (e[1] - s[1]),
        s[2] + std::max(code[2], 0) * CoarseBlockSize[2] - code[2] * nZ +
            std::min(0, code[2]) * (e[2] - s[2])};
    const int m_vSize0 = m_CoarsenedBlock->getSize(0);
    const int m_nElemsPerSlice =
        m_CoarsenedBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - offset[0];
    const int XX = start[0];
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int ZZ = 2 * (iz - s[2]) + start[2];
      const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1]; iy++) {
        if (code[1] == 0 && code[2] == 0 && iy > -m_InterpStencilStart[1] &&
            iy < nY / 2 - m_InterpStencilEnd[1] &&
            iz > -m_InterpStencilStart[2] &&
            iz < nZ / 2 - m_InterpStencilEnd[2])
          continue;
        ElementType *__restrict__ ptrDest1 =
            &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
        const int YY = 2 * (iy - s[1]) + start[1];
        const ElementType *ptrSrc_0 = (const ElementType *)&b(XX, YY, ZZ);
        const ElementType *ptrSrc_1 = (const ElementType *)&b(XX, YY + 1, ZZ);
#pragma GCC ivdep
        for (int ee = 0; ee < e[0] - s[0]; ee++) {
          ptrDest1[ee] =
              AverageDown(*(ptrSrc_0 + 2 * ee), *(ptrSrc_1 + 2 * ee),
                          *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1));
        }
      }
    }
  }
#ifdef PRESERVE_SYMMETRY
  __attribute__((optimize("-O1")))
#endif
  void
  CoarseFineInterpolation(const BlockInfo &info) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const bool xperiodic = is_xperiodic();
    const bool yperiodic = is_yperiodic();
    const bool zperiodic = is_zperiodic();
    const std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    const int aux = 1 << info.level;
    const bool xskin =
        info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
    const bool yskin =
        info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
    const bool zskin =
        info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
    const int xskip = info.index[0] == 0 ? -1 : 1;
    const int yskip = info.index[1] == 0 ? -1 : 1;
    const int zskip = info.index[2] == 0 ? -1 : 1;
    for (int ii = 0; ii < coarsened_nei_codes_size; ++ii) {
      const int icode = coarsened_nei_codes[ii];
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
      if (code[2] != 0)
        continue;
      if (!xperiodic && code[0] == xskip && xskin)
        continue;
      if (!yperiodic && code[1] == yskip && yskin)
        continue;
      if (!zperiodic && code[2] == zskip && zskin)
        continue;
      if (!istensorial && !use_averages &&
          abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
        continue;
      const int s[3] = {
          code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
          code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
          code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};
      const int sC[3] = {
          code[0] < 1 ? (code[0] < 0 ? ((m_stencilStart[0] - 1) / 2) : 0)
                      : CoarseBlockSize[0],
          code[1] < 1 ? (code[1] < 0 ? ((m_stencilStart[1] - 1) / 2) : 0)
                      : CoarseBlockSize[1],
          code[2] < 1 ? (code[2] < 0 ? ((m_stencilStart[2] - 1) / 2) : 0)
                      : CoarseBlockSize[2]};
      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes)
        continue;
      if (use_averages) {
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1]; iy += 1) {
          const int YY =
              (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
              sC[1];
#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            const int XX =
                (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                sC[0];
            ElementType *Test[3][3];
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++)
                Test[i][j] = &m_CoarsenedBlock->Access(
                    XX - 1 + i - offset[0], YY - 1 + j - offset[1], 0);
            TestInterp(
                Test,
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1], 0),
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2,
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) %
                    2);
          }
        }
      }
      if (m_refGrid->FiniteDifferences && abs(code[0]) + abs(code[1]) == 1) {
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1]; iy += 2) {
          const int YY =
              (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
              sC[1] - offset[1];
          const int y =
              abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
          const int iyp = (abs(iy) % 2 == 1) ? -1 : 1;
          const double dy = 0.25 * (2 * y - 1);
#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 2) {
            const int XX =
                (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                sC[0] - offset[0];
            const int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            const int ixp = (abs(ix) % 2 == 1) ? -1 : 1;
            const double dx = 0.25 * (2 * x - 1);
            if (ix < -2 || iy < -2 || ix > nX + 1 || iy > nY + 1)
              continue;
            if (code[0] != 0) {
              ElementType dudy, dudy2;
              if (YY + offset[1] == 0) {
                dudy = (-0.5 * m_CoarsenedBlock->Access(XX, YY + 2, 0) -
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) +
                       2.0 * m_CoarsenedBlock->Access(XX, YY + 1, 0);
                dudy2 = (m_CoarsenedBlock->Access(XX, YY + 2, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY + 1, 0);
              } else if (YY + offset[1] == CoarseBlockSize[1] - 1) {
                dudy = (0.5 * m_CoarsenedBlock->Access(XX, YY - 2, 0) +
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) -
                       2.0 * m_CoarsenedBlock->Access(XX, YY - 1, 0);
                dudy2 = (m_CoarsenedBlock->Access(XX, YY - 2, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY - 1, 0);
              } else {
                dudy = 0.5 * (m_CoarsenedBlock->Access(XX, YY + 1, 0) -
                              m_CoarsenedBlock->Access(XX, YY - 1, 0));
                dudy2 = (m_CoarsenedBlock->Access(XX, YY + 1, 0) +
                         m_CoarsenedBlock->Access(XX, YY - 1, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY, 0);
              }
              m_cacheBlock->Access(ix - m_stencilStart[0],
                                   iy - m_stencilStart[1], 0) =
                  m_CoarsenedBlock->Access(XX, YY, 0) + dy * dudy +
                  (0.5 * dy * dy) * dudy2;
              if (iy + iyp >= s[1] && iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dy * dudy +
                    (0.5 * dy * dy) * dudy2;
              if (ix + ixp >= s[0] && ix + ixp < e[0])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1], 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) + dy * dudy +
                    (0.5 * dy * dy) * dudy2;
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dy * dudy +
                    (0.5 * dy * dy) * dudy2;
            } else {
              ElementType dudx, dudx2;
              if (XX + offset[0] == 0) {
                dudx = (-0.5 * m_CoarsenedBlock->Access(XX + 2, YY, 0) -
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) +
                       2.0 * m_CoarsenedBlock->Access(XX + 1, YY, 0);
                dudx2 = (m_CoarsenedBlock->Access(XX + 2, YY, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX + 1, YY, 0);
              } else if (XX + offset[0] == CoarseBlockSize[0] - 1) {
                dudx = (0.5 * m_CoarsenedBlock->Access(XX - 2, YY, 0) +
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) -
                       2.0 * m_CoarsenedBlock->Access(XX - 1, YY, 0);
                dudx2 = (m_CoarsenedBlock->Access(XX - 2, YY, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX - 1, YY, 0);
              } else {
                dudx = 0.5 * (m_CoarsenedBlock->Access(XX + 1, YY, 0) -
                              m_CoarsenedBlock->Access(XX - 1, YY, 0));
                dudx2 = (m_CoarsenedBlock->Access(XX + 1, YY, 0) +
                         m_CoarsenedBlock->Access(XX - 1, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY, 0);
              }
              m_cacheBlock->Access(ix - m_stencilStart[0],
                                   iy - m_stencilStart[1], 0) =
                  m_CoarsenedBlock->Access(XX, YY, 0) + dx * dudx +
                  (0.5 * dx * dx) * dudx2;
              if (iy + iyp >= s[1] && iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) + dx * dudx +
                    (0.5 * dx * dx) * dudx2;
              if (ix + ixp >= s[0] && ix + ixp < e[0])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1], 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dx * dudx +
                    (0.5 * dx * dx) * dudx2;
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dx * dudx +
                    (0.5 * dx * dx) * dudx2;
            }
          }
        }
        for (int iy = s[1]; iy < e[1]; iy += 1) {
#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            if (ix < -2 || iy < -2 || ix > nX + 1 || iy > nY + 1)
              continue;
            const int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            const int y =
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
            auto &a = m_cacheBlock->Access(ix - m_stencilStart[0],
                                           iy - m_stencilStart[1], 0);
            if (code[0] == 0 && code[1] == 1) {
              if (y == 0) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 1, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 2, 0);
                LI(a, b, c);
              } else if (y == 1) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 2, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 3, 0);
                LE(a, b, c);
              }
            } else if (code[0] == 0 && code[1] == -1) {
              if (y == 1) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 1, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 2, 0);
                LI(a, b, c);
              } else if (y == 0) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 2, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 3, 0);
                LE(a, b, c);
              }
            } else if (code[1] == 0 && code[0] == 1) {
              if (x == 0) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] - 1,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] - 2,
                                               iy - m_stencilStart[1], 0);
                LI(a, b, c);
              } else if (x == 1) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] - 2,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] - 3,
                                               iy - m_stencilStart[1], 0);
                LE(a, b, c);
              }
            } else if (code[1] == 0 && code[0] == -1) {
              if (x == 1) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] + 1,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] + 2,
                                               iy - m_stencilStart[1], 0);
                LI(a, b, c);
              } else if (x == 0) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] + 2,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] + 3,
                                               iy - m_stencilStart[1], 0);
                LE(a, b, c);
              }
            }
          }
        }
      }
    }
  }
  virtual void _apply_bc(const BlockInfo &info, const Real t = 0,
                         bool coarse = false) {}
  template <typename T> void _release(T *&t) {
    if (t != NULL) {
      allocator<T>().destroy(t);
      allocator<T>().deallocate(t, 1);
    }
    t = NULL;
  }

private:
  BlockLab(const BlockLab &) = delete;
  BlockLab &operator=(const BlockLab &) = delete;
};
} // namespace cubism
