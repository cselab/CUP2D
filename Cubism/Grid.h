#pragma once
#include "BlockInfo.h"
#include "FluxCorrection.h"
#include <algorithm>
#include <unordered_map>
namespace cubism {
struct BlockGroup {
  int i_min[3];
  int i_max[3];
  int level;
  std::vector<long long> Z;
  size_t ID;
  double origin[3];
  double h;
  int NXX;
  int NYY;
  int NZZ;
};
template <typename Block,
          template <typename X> class allocator = std::allocator>
class Grid {
public:
  typedef Block BlockType;
  using ElementType = typename Block::ElementType;
  std::unordered_map<long long, BlockInfo *> BlockInfoAll;
  std::unordered_map<long long, TreePosition> Octree;
  std::vector<BlockInfo> m_vInfo;
  const int NX;
  const int NY;
  const int NZ;
  const double maxextent;
  const int levelMax;
  const int levelStart;
  const bool xperiodic;
  const bool yperiodic;
  const bool zperiodic;
  std::vector<BlockGroup> MyGroups;
  std::vector<long long> level_base;
  bool UpdateFluxCorrection{true};
  bool UpdateGroups{true};
  bool FiniteDifferences{true};
  FluxCorrection<Grid> CorrectorGrid;
  TreePosition &Tree(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = Octree.find(aux);
    if (retval == Octree.end()) {
#pragma omp critical
      {
        const auto retval1 = Octree.find(aux);
        if (retval1 == Octree.end()) {
          TreePosition dum;
          Octree[aux] = dum;
        }
      }
      return Tree(m, n);
    } else {
      return retval->second;
    }
  }
  TreePosition &Tree(BlockInfo &info) { return Tree(info.level, info.Z); }
  TreePosition &Tree(const BlockInfo &info) { return Tree(info.level, info.Z); }
  void _alloc() {
    const int m = levelStart;
    const int TwoPower = 1 << m;
    for (long long n = 0; n < NX * NY * NZ * pow(TwoPower, DIMENSION); n++) {
      Tree(m, n).setrank(0);
      _alloc(m, n);
    }
    if (m - 1 >= 0) {
      for (long long n = 0; n < NX * NY * NZ * pow((1 << (m - 1)), DIMENSION);
           n++)
        Tree(m - 1, n).setCheckFiner();
    }
    if (m + 1 < levelMax) {
      for (long long n = 0; n < NX * NY * NZ * pow((1 << (m + 1)), DIMENSION);
           n++)
        Tree(m + 1, n).setCheckCoarser();
    }
    FillPos();
  }
  void _alloc(const int m, const long long n) {
    allocator<Block> alloc;
    BlockInfo &new_info = getBlockInfoAll(m, n);
    new_info.ptrBlock = alloc.allocate(1);
#pragma omp critical
    { m_vInfo.push_back(new_info); }
    Tree(m, n).setrank(rank());
  }
  void _deallocAll() {
    allocator<Block> alloc;
    for (size_t i = 0; i < m_vInfo.size(); i++) {
      const int m = m_vInfo[i].level;
      const long long n = m_vInfo[i].Z;
      alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
    }
    std::vector<long long> aux;
    for (auto &m : BlockInfoAll)
      aux.push_back(m.first);
    for (size_t i = 0; i < aux.size(); i++) {
      const auto retval = BlockInfoAll.find(aux[i]);
      if (retval != BlockInfoAll.end()) {
        delete retval->second;
      }
    }
    m_vInfo.clear();
    BlockInfoAll.clear();
    Octree.clear();
  }
  void _dealloc(const int m, const long long n) {
    allocator<Block> alloc;
    alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
    for (size_t j = 0; j < m_vInfo.size(); j++) {
      if (m_vInfo[j].level == m && m_vInfo[j].Z == n) {
        m_vInfo.erase(m_vInfo.begin() + j);
        return;
      }
    }
  }
  void dealloc_many(const std::vector<long long> &dealloc_IDs) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      m_vInfo[j].changed2 = false;
    allocator<Block> alloc;
    for (size_t i = 0; i < dealloc_IDs.size(); i++)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        if (m_vInfo[j].blockID_2 == dealloc_IDs[i]) {
          const int m = m_vInfo[j].level;
          const long long n = m_vInfo[j].Z;
          m_vInfo[j].changed2 = true;
          alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
          break;
        }
      }
    m_vInfo.erase(std::remove_if(m_vInfo.begin(), m_vInfo.end(),
                                 [](const BlockInfo &x) { return x.changed2; }),
                  m_vInfo.end());
  }
  void FindBlockInfo(const int m, const long long n, const int m_new,
                     const long long n_new) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      if (m == m_vInfo[j].level && n == m_vInfo[j].Z) {
        BlockInfo &correct_info = getBlockInfoAll(m_new, n_new);
        correct_info.state = Leave;
        m_vInfo[j] = correct_info;
        return;
      }
  }
  virtual void FillPos(bool CopyInfos = true) {
    std::sort(m_vInfo.begin(), m_vInfo.end());
    Octree.reserve(Octree.size() + m_vInfo.size() / 8);
    if (CopyInfos)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        const int m = m_vInfo[j].level;
        const long long n = m_vInfo[j].Z;
        BlockInfo &correct_info = getBlockInfoAll(m, n);
        correct_info.blockID = j;
        m_vInfo[j] = correct_info;
        assert(Tree(m, n).Exists());
      }
    else
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        const int m = m_vInfo[j].level;
        const long long n = m_vInfo[j].Z;
        BlockInfo &correct_info = getBlockInfoAll(m, n);
        correct_info.blockID = j;
        m_vInfo[j].blockID = j;
        m_vInfo[j].state = correct_info.state;
        assert(Tree(m, n).Exists());
      }
  }
  Grid(const unsigned int _NX, const unsigned int _NY = 1,
       const unsigned int _NZ = 1, const double _maxextent = 1,
       const unsigned int _levelStart = 0, const unsigned int _levelMax = 1,
       const bool AllocateBlocks = true, const bool a_xperiodic = true,
       const bool a_yperiodic = true, const bool a_zperiodic = true)
      : NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent), levelMax(_levelMax),
        levelStart(_levelStart), xperiodic(a_xperiodic), yperiodic(a_yperiodic),
        zperiodic(a_zperiodic) {
    BlockInfo dummy;
    const int nx = dummy.blocks_per_dim(0, NX, NY);
    const int ny = dummy.blocks_per_dim(1, NX, NY);
    const int nz = 1;
    const int lvlMax = dummy.levelMax(levelMax);
    for (int m = 0; m < lvlMax; m++) {
      const int TwoPower = 1 << m;
      const long long Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
      if (m == 0)
        level_base.push_back(Ntot);
      if (m > 0)
        level_base.push_back(level_base[m - 1] + Ntot);
    }
    if (AllocateBlocks)
      _alloc();
  }
  virtual ~Grid() { _deallocAll(); }
  virtual Block *avail(const int m, const long long n) {
    return (Block *)getBlockInfoAll(m, n).ptrBlock;
  }
  virtual int rank() const { return 0; }
  virtual void initialize_blocks(const std::vector<long long> &blocksZ,
                                 const std::vector<short int> &blockslevel) {
    _deallocAll();
    for (size_t i = 0; i < blocksZ.size(); i++) {
      const int level = blockslevel[i];
      const long long Z = blocksZ[i];
      _alloc(level, Z);
      Tree(level, Z).setrank(rank());
      int p[2];
      BlockInfo::inverse(Z, level, p[0], p[1]);
      if (level < levelMax - 1)
        for (int j1 = 0; j1 < 2; j1++)
          for (int i1 = 0; i1 < 2; i1++) {
            const long long nc =
                getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
            Tree(level + 1, nc).setCheckCoarser();
          }
      if (level > 0) {
        const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
        Tree(level - 1, nf).setCheckFiner();
      }
    }
    FillPos();
    UpdateFluxCorrection = true;
    UpdateGroups = true;
  }
  long long getZforward(const int level, const int i, const int j) const {
    const int TwoPower = 1 << level;
    const int ix = (i + TwoPower * NX) % (NX * TwoPower);
    const int iy = (j + TwoPower * NY) % (NY * TwoPower);
    return BlockInfo::forward(level, ix, iy);
  }
  Block *avail1(const int ix, const int iy, const int m) {
    const long long n = getZforward(m, ix, iy);
    return avail(m, n);
  }
  Block &operator()(const long long ID) {
    return *(Block *)m_vInfo[ID].ptrBlock;
  }
  std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }
  std::array<int, 3> getMaxMostRefinedBlocks() const {
    return {
        NX << (levelMax - 1),
        NY << (levelMax - 1),
        DIMENSION == 3 ? (NZ << (levelMax - 1)) : 1,
    };
  }
  std::array<int, 3> getMaxMostRefinedCells() const {
    const auto b = getMaxMostRefinedBlocks();
    return {b[0] * Block::sizeX, b[1] * Block::sizeY, b[2] * Block::sizeZ};
  }
  inline int getlevelMax() const { return levelMax; }
  BlockInfo &getBlockInfoAll(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = BlockInfoAll.find(aux);
    if (retval != BlockInfoAll.end()) {
      return *retval->second;
    } else {
#pragma omp critical
      {
        const auto retval1 = BlockInfoAll.find(aux);
        if (retval1 == BlockInfoAll.end()) {
          BlockInfo *dumm = new BlockInfo();
          const int TwoPower = 1 << m;
          const double h0 = (maxextent / std::max(NX * Block::sizeX,
                                                  std::max(NY * Block::sizeY,
                                                           NZ * Block::sizeZ)));
          const double h = h0 / TwoPower;
          double origin[3];
          int i, j, k;
          BlockInfo::inverse(n, m, i, j);
          k = 0;
          origin[0] = i * Block::sizeX * h;
          origin[1] = j * Block::sizeY * h;
          origin[2] = k * Block::sizeZ * h;
          dumm->setup(m, h, origin, n);
          BlockInfoAll[aux] = dumm;
        }
      }
      return getBlockInfoAll(m, n);
    }
  }
  std::vector<BlockInfo> &getBlocksInfo() { return m_vInfo; }
  const std::vector<BlockInfo> &getBlocksInfo() const { return m_vInfo; }
  virtual int get_world_size() const { return 1; }
  virtual void UpdateBoundary(bool clean = false) {}
  void UpdateMyGroups() {
    if (rank() == 0)
      std::cout << "Updating groups..." << std::endl;
    const unsigned int nX = BlockType::sizeX;
    const unsigned int nY = BlockType::sizeY;
    const size_t Ngrids = getBlocksInfo().size();
    const auto &MyInfos = getBlocksInfo();
    UpdateGroups = false;
    MyGroups.clear();
    std::vector<bool> added(MyInfos.size(), false);
    for (unsigned int m = 0; m < Ngrids; m++) {
      const BlockInfo &I = MyInfos[m];
      if (added[I.blockID])
        continue;
      added[I.blockID] = true;
      BlockGroup newGroup;
      newGroup.level = I.level;
      newGroup.h = I.h;
      newGroup.Z.push_back(I.Z);
      const int base[3] = {I.index[0], I.index[1], 0};
      int i_off[4] = {};
      bool ready_[4] = {};
      int d = 0;
      auto blk = getMaxBlocks();
      do {
        if (ready_[d] == false) {
          bool valid = true;
          i_off[d]++;
          const int i0 =
              (d < 2) ? (base[d] - i_off[d]) : (base[d - 2] + i_off[d]);
          const int d0 = (d < 2) ? (d) % 2 : (d - 2) % 2;
          const int d1 = (d < 2) ? (d + 1) % 2 : (d - 2 + 1) % 2;
          for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 2];
               i1++) {
            if (valid == false)
              break;
            if (i0 < 0 || i1 < 0 || i0 >= blk[d0] * (1 << I.level) ||
                i1 >= blk[d1] * (1 << I.level)) {
              valid = false;
              break;
            }
            long long n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1)
                                             : getZforward(I.level, i1, i0);
            if (Tree(I.level, n).rank() != rank()) {
              valid = false;
              break;
            }
            if (added[getBlockInfoAll(I.level, n).blockID] == true) {
              valid = false;
            }
          }
          if (valid == false) {
            i_off[d]--;
            ready_[d] = true;
          } else {
            for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 2];
                 i1++) {
              long long n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1)
                                               : getZforward(I.level, i1, i0);
              newGroup.Z.push_back(n);
              added[getBlockInfoAll(I.level, n).blockID] = true;
            }
          }
        }
        d = (d + 1) % 4;
      } while (ready_[0] == false || ready_[1] == false || ready_[2] == false ||
               ready_[3] == false);
      const int ix_min = base[0] - i_off[0];
      const int iy_min = base[1] - i_off[1];
      const int iz_min = 0;
      const int ix_max = base[0] + i_off[2];
      const int iy_max = base[1] + i_off[3];
      const int iz_max = 0;
      long long n_base = getZforward(I.level, ix_min, iy_min);
      newGroup.i_min[0] = ix_min;
      newGroup.i_min[1] = iy_min;
      newGroup.i_min[2] = iz_min;
      newGroup.i_max[0] = ix_max;
      newGroup.i_max[1] = iy_max;
      newGroup.i_max[2] = iz_max;
      const BlockInfo &info = getBlockInfoAll(I.level, n_base);
      newGroup.origin[0] = info.origin[0];
      newGroup.origin[1] = info.origin[1];
      newGroup.origin[2] = info.origin[2];
      newGroup.NXX = (newGroup.i_max[0] - newGroup.i_min[0] + 1) * nX + 1;
      newGroup.NYY = (newGroup.i_max[1] - newGroup.i_min[1] + 1) * nY + 1;
      newGroup.NZZ = 2;
      MyGroups.push_back(newGroup);
    }
  }
};
} // namespace cubism
