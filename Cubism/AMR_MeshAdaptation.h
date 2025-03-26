#pragma once

#include "AMR_SynchronizerMPI.h"
#include "BlockInfo.h"
#include "LoadBalancer.h"
#include "StencilInfo.h"

namespace cubism {

/**
 * @brief Class responsible for mesh refinement of a GridMPI.
 *
 * This class can label each GridBlock of a GridMPI as requiring
 * refinement/compression/nothing. It can then perform interpolation of points
 * for refinement and averaging down for compression, followed by load balancing
 * of GridBlocks among different processes.
 *
 * The user should use this class through its constructor and through the
 * functions 'Tag', 'TagLike' and 'Adapt'. When constructed, the user provides
 * this class with a Grid and two numbers: the tolerance for refinement (Rtol)
 * and the tolerance for compression (Ctol). By calling 'Tag', GridBlocks with
 * gridpoints that have magnitude() > Rtol will be tagged for refinement and
 * those with magnitude() < Ctol for compression. Alternatively, 'TagLike' can
 * be used to copy the tags of blocks from another already tagged grid to this
 * grid. Once tagged, the user needs to call 'Adapt', which will adapt the mesh,
 * interpolate new points and do the load-balancing of blocks among processes.
 *
 * In order to change the default refinement criterion, a new class can inherit
 * from this class and overload the function 'TagLoadedBlock'. In order to
 * change the default refinement interpolation, a new class can inherit from
 * this class and overload the function 'RefineBlocks'.
 *
 * @tparam TLab  The BlockLab type used for halo cells exchange and boundary
 * condition enforcement, when interpolation of gridpoints happens after mesh
 * refinement.
 */
template <typename TLab> class MeshAdaptation {
protected:
  typedef typename TLab::GridType TGrid;
  typedef typename TGrid::Block BlockType;
  typedef typename TGrid::BlockType::ElementType ElementType;
  typedef typename TGrid::BlockType::ElementType::RealType Real;
  typedef SynchronizerMPI_AMR<Real, TGrid> SynchronizerMPIType;

  StencilInfo stencil;  ///< stencil of +-1 point, needed for 2nd-order
                        ///< refinement interpolation
  bool CallValidStates; ///< will be true when 'Tag' is called and some
                        ///< refinement/compression is needed
  bool boundary_needed; ///< set true to update the boundary blocks of each
                        ///< GridMPI
  LoadBalancer<TGrid> *Balancer; ///< load-balancing of blocks
  TGrid *grid;                   ///< pointer to Grid that will be adapted
  double time; ///< (optional) time of simulation, for time-dependent refinement
               ///< criteria
  bool basic_refinement; ///< set to false if no interpolation is to be
                         ///< performed after refinement
  double tolerance_for_refinement;  ///< compare 'magnitude()' of each gridpoint
                                    ///< to this number, to check if refinement
                                    ///< is needed
  double tolerance_for_compression; ///< compare 'magnitude()' of each gridpoint
                                    ///< to this number, to check if compression
                                    ///< is needed
  std::vector<long long>
      dealloc_IDs; ///< blockIDs for blocks that are deallocated because of mesh
                   ///< refinement/compression

public:
  /**
   * @brief Class constructor.
   *
   * @param g The Grid to be refined/compressed.
   * @param Rtol Tolerance for refinement.
   * @param Ctol Tolerance for compression.
   */
  MeshAdaptation(TGrid &g, double Rtol, double Ctol) {
    grid = &g;

    tolerance_for_refinement = Rtol;
    tolerance_for_compression = Ctol;

    boundary_needed = false;

    constexpr int Gx = 1;
    constexpr int Gy = 1;
    constexpr int Gz = DIMENSION == 3 ? 1 : 0;
    stencil.sx = -Gx;
    stencil.sy = -Gy;
    stencil.sz = -Gz;
    stencil.ex = Gx + 1;
    stencil.ey = Gy + 1;
    stencil.ez = Gz + 1;
    stencil.tensorial = true;
    for (int i = 0; i < ElementType::DIM; i++)
      stencil.selcomponents.push_back(i);

    Balancer = new LoadBalancer<TGrid>(*grid);
  }

  /**
   * @brief Class destructor.
   */
  virtual ~MeshAdaptation() { delete Balancer; }

  /**
   * @brief Tag each block of this grid for refinement/compression based on
   * criterion from 'TagLoadedBlock'.
   * @param t Current time of the simulation; used only for time-dependent
   * boundary conditions.
   */
  void Tag(double t = 0) {
    time = t;
    boundary_needed = true;

    SynchronizerMPI_AMR<Real, TGrid> *Synch = grid->sync(stencil);

    CallValidStates = false;
    bool Reduction = false;
    MPI_Request Reduction_req;
    int tmp;

    std::vector<BlockInfo *> &inner = Synch->avail_inner();
    TagBlocksVector(inner, Reduction, Reduction_req, tmp);

    std::vector<BlockInfo *> &halo = Synch->avail_halo();
    TagBlocksVector(halo, Reduction, Reduction_req, tmp);

    if (!Reduction) {
      tmp = CallValidStates ? 1 : 0;
      Reduction = true;
      MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM,
                     grid->getWorldComm(), &Reduction_req);
    }

    MPI_Wait(&Reduction_req, MPI_STATUS_IGNORE);
    CallValidStates = (tmp > 0);

    grid->boundary = halo;

    if (CallValidStates)
      ValidStates();
  }

  /**
   * @brief Refine/compress the mesh after blocks are tagged.
   * @param t Current time of the simulation; used only for time-dependent
   * boundary conditions.
   * @param verbosity Boolean variable controlling screen output.
   * @param basic Boolean variable; if set to false, no refinement interpolation
   * is performed and blocks are simply allocated (and filled with nothing)
   * after refinement.
   */
  void Adapt(double t = 0, bool verbosity = false, bool basic = false) {
    basic_refinement = basic;
    SynchronizerMPI_AMR<Real, TGrid> *Synch = nullptr;
    if (basic == false) {
      Synch = grid->sync(stencil);
      // TODO: the line below means there's no computation & communication
      // overlap here
      grid->boundary = Synch->avail_halo();
      if (boundary_needed)
        grid->UpdateBoundary();
    }

    int r = 0;
    int c = 0;

    std::vector<int> m_com;
    std::vector<int> m_ref;
    std::vector<long long> n_com;
    std::vector<long long> n_ref;

    std::vector<BlockInfo> &I = grid->getBlocksInfo();

    long long blocks_after = I.size();

    for (auto &info : I) {
      if (info.state == Refine) {
        m_ref.push_back(info.level);
        n_ref.push_back(info.Z);
        blocks_after += (1 << DIMENSION) - 1;
        r++;
      } else if (info.state == Compress && info.index[0] % 2 == 0 &&
                 info.index[1] % 2 == 0 && info.index[2] % 2 == 0) {
        m_com.push_back(info.level);
        n_com.push_back(info.Z);
        c++;
      } else if (info.state == Compress) {
        blocks_after--;
      }
    }
    MPI_Request requests[2];
    int temp[2] = {r, c};
    int result[2];
    int size;
    MPI_Comm_size(grid->getWorldComm(), &size);
    std::vector<long long> block_distribution(size);
    MPI_Iallreduce(&temp, &result, 2, MPI_INT, MPI_SUM, grid->getWorldComm(),
                   &requests[0]);
    MPI_Iallgather(&blocks_after, 1, MPI_LONG_LONG, block_distribution.data(),
                   1, MPI_LONG_LONG, grid->getWorldComm(), &requests[1]);

    dealloc_IDs.clear();

    {
      TLab lab;
      if (Synch != nullptr)
        lab.prepare(*grid, Synch->getstencil());
      for (size_t i = 0; i < m_ref.size(); i++) {
        refine_1(m_ref[i], n_ref[i], lab);
      }
      for (size_t i = 0; i < m_ref.size(); i++) {
        refine_2(m_ref[i], n_ref[i]);
      }
    }
    grid->dealloc_many(dealloc_IDs);

    Balancer->PrepareCompression();

    dealloc_IDs.clear();

    for (size_t i = 0; i < m_com.size(); i++) {
      compress(m_com[i], n_com[i]);
    }

    grid->dealloc_many(dealloc_IDs);

    MPI_Waitall(2, requests, MPI_STATUS_IGNORE);
    if (verbosity) {
      std::cout
          << "==============================================================\n";
      std::cout << " refined:" << result[0] << "   compressed:" << result[1]
                << std::endl;
      std::cout
          << "=============================================================="
          << std::endl;
    }

    Balancer->Balance_Diffusion(verbosity, block_distribution);

    if (result[0] > 0 || result[1] > 0 || Balancer->movedBlocks) {
      grid->UpdateFluxCorrection = true;
      grid->UpdateGroups = true;

      grid->UpdateBlockInfoAll_States(false);

      auto it = grid->SynchronizerMPIs.begin();
      while (it != grid->SynchronizerMPIs.end()) {
        (*it->second)._Setup();
        it++;
      }
    }
  }

  /**
   * @brief Tag each block of this grid for refinement/compression by copying
   * the tags of the given BlockInfos.
   * @param I1 Vector of BlockInfos whose 'state' (refine/compress/leave) will
   * be copied to the BlockInfos of this grid.
   */
  void TagLike(const std::vector<BlockInfo> &I1) {
    std::vector<BlockInfo> &I2 = grid->getBlocksInfo();
    for (size_t i1 = 0; i1 < I2.size(); i1++) {
      BlockInfo &ary0 = I2[i1];
      BlockInfo &info = grid->getBlockInfoAll(ary0.level, ary0.Z);
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++) {
          const long long n = grid->getZforward(info.level, i, j);
          BlockInfo &infoNei = grid->getBlockInfoAll(info.level, n);
          infoNei.state = Leave;
        }
      info.state = Leave;
      ary0.state = Leave;
    }
#pragma omp parallel for
    for (size_t i = 0; i < I1.size(); i++) {
      const BlockInfo &info1 = I1[i];
      BlockInfo &info2 = I2[i];
      BlockInfo &info3 = grid->getBlockInfoAll(info2.level, info2.Z);
      info2.state = info1.state;
      info3.state = info1.state;
      if (info2.state == Compress) {
        const int i2 = 2 * (info2.index[0] / 2);
        const int j2 = 2 * (info2.index[1] / 2);
        const long long n = grid->getZforward(info2.level, i2, j2);
        BlockInfo &infoNei = grid->getBlockInfoAll(info2.level, n);
        infoNei.state = Compress;
      }
    }
  }

protected:
  /**
   * @brief Auxiliary function to tag a vector of blocks
   * @param I Vector of BlockInfos to tag.
   * @param Reduction Boolean that will be set to true if any block is tagged;
   * setting to true will cause a call to 'ValidStates()' after tagging blocks.
   * @param Reduction_req MPI request that will be used if 'Reduction' is true
   * @param tmp Same value as 'Reduction' by this is an integer
   */
  void TagBlocksVector(std::vector<BlockInfo *> &I, bool &Reduction,
                       MPI_Request &Reduction_req, int &tmp) {
    const int levelMax = grid->getlevelMax();
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < I.size(); i++) {
        BlockInfo &info = grid->getBlockInfoAll(I[i]->level, I[i]->Z);

        I[i]->state = TagLoadedBlock(info);

        const bool maxLevel =
            (I[i]->state == Refine) && (I[i]->level == levelMax - 1);
        const bool minLevel = (I[i]->state == Compress) && (I[i]->level == 0);

        if (maxLevel || minLevel)
          I[i]->state = Leave;

        info.state = I[i]->state;
        if (info.state != Leave) {
#pragma omp critical
          {
            CallValidStates = true;
            if (!Reduction) {
              tmp = 1;
              Reduction = true;
              MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM,
                             grid->getWorldComm(), &Reduction_req);
            }
          }
        }
      }
    }
  }

  /**
   * @brief First step of refinement of a block.
   *
   * The new blocks are allocated and the interpolation needed for refinement is
   * perfomed. The parent block will be deallocated in step 2 (refine_2).
   *
   * @param level The refinement level of the block to be refined.
   * @param Z The Z-order index of the block to be refined.
   */
  void refine_1(const int level, const long long Z, TLab &lab) {
    BlockInfo &parent = grid->getBlockInfoAll(level, Z);
    parent.state = Leave;
    if (basic_refinement == false)
      lab.load(parent, time, true);

    const int p[3] = {parent.index[0], parent.index[1], parent.index[2]};

    assert(parent.ptrBlock != NULL);
    assert(level <= grid->getlevelMax() - 1);
    BlockType *Blocks[4];
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
        const long long nc =
            grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
        BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
        Child.state = Leave;
        grid->_alloc(level + 1, nc);
        grid->Tree(level + 1, nc).setCheckCoarser();
        Blocks[j * 2 + i] = (BlockType *)Child.ptrBlock;
      }
    if (basic_refinement == false)
      RefineBlocks(Blocks, lab);
  }

  /**
   * @brief Second step of refinement of a block.
   *
   * After all blocks are refined with refine_1, we can deallocate their parent
   * blocks here.
   *
   * @param level The refinement level of the block to be refined.
   * @param Z The Z-order index of the block to be refined.
   */
  void refine_2(const int level, const long long Z) {
#pragma omp critical
    {
      dealloc_IDs.push_back(grid->getBlockInfoAll(level, Z).blockID_2);
    }

    BlockInfo &parent = grid->getBlockInfoAll(level, Z);
    grid->Tree(parent).setCheckFiner();
    parent.state = Leave;

    int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
        const long long nc =
            grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
        BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
        grid->Tree(Child).setrank(grid->rank());
        if (level + 2 < grid->getlevelMax())
          for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 2; i1++)
              grid->Tree(level + 2, Child.Zchild[i0][i1][1]).setCheckCoarser();
      }
  }

  /**
   * @brief Compress eight blocks.
   *
   * The 'bottom left' block (i,j,k) is provided in the input, via its
   * refinement level and Z-order index. The top right block would be the block
   * (i+1,j+1,k+1).
   *
   * @param level The refinement level of the bottom left block to be refined.
   * @param Z The Z-order index of the bottom left block to be refined.
   */
  void compress(const int level, const long long Z) {
    assert(level > 0);

    BlockInfo &info = grid->getBlockInfoAll(level, Z);

    assert(info.state == Compress);

    BlockType *Blocks[4];
    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        const int blk = J * 2 + I;
        const long long n =
            grid->getZforward(level, info.index[0] + I, info.index[1] + J);
        Blocks[blk] = (BlockType *)(grid->getBlockInfoAll(level, n)).ptrBlock;
      }

    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    const int offsetX[2] = {0, nx / 2};
    const int offsetY[2] = {0, ny / 2};
    if (basic_refinement == false)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          BlockType &b = *Blocks[J * 2 + I];
          for (int j = 0; j < ny; j += 2)
            for (int i = 0; i < nx; i += 2) {
              ElementType average = 0.25 * ((b(i, j, 0) + b(i + 1, j + 1, 0)) +
                                            (b(i + 1, j, 0) + b(i, j + 1, 0)));
              (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], 0) = average;
            }
        }
    const long long np =
        grid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2);
    BlockInfo &parent = grid->getBlockInfoAll(level - 1, np);
    grid->Tree(parent.level, parent.Z).setrank(grid->rank());
    parent.ptrBlock = info.ptrBlock;
    parent.state = Leave;
    if (level - 2 >= 0)
      grid->Tree(level - 2, parent.Zparent).setCheckFiner();

    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        const long long n =
            grid->getZforward(level, info.index[0] + I, info.index[1] + J);
        if (I + J == 0) {
          grid->FindBlockInfo(level, n, level - 1, np);
        } else {
#pragma omp critical
          {
            dealloc_IDs.push_back(grid->getBlockInfoAll(level, n).blockID_2);
          }
        }
        grid->Tree(level, n).setCheckCoarser();
        grid->getBlockInfoAll(level, n).state = Leave;
      }
  }

  /**
   * @brief Make sure adjacent blocks of the to-be-adapted mesh do not differ by
   * more than one refinement level.
   *
   * Given a set of tagged blocks, this function will mark some additional
   * blocks for refinement, to make sure no adjacent blocks differ by more than
   * one refinement level. It will also unmark some blocks from being
   * compressed, if their adjacent blocks do not need compression and/or belong
   * to a finer refinement level.
   */
  void ValidStates() {
    const std::array<int, 3> blocksPerDim = grid->getMaxBlocks();
    const int levelMin = 0;
    const int levelMax = grid->getlevelMax();
    const bool xperiodic = grid->xperiodic;
    const bool yperiodic = grid->yperiodic;
    const bool zperiodic = grid->zperiodic;

    std::vector<BlockInfo> &I = grid->getBlocksInfo();

#pragma omp parallel for
    for (size_t j = 0; j < I.size(); j++) {
      BlockInfo &info = I[j];

      if ((info.state == Refine && info.level == levelMax - 1) ||
          (info.state == Compress && info.level == levelMin)) {
        info.state = Leave;
        (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
      }
      if (info.state != Leave) {
        info.changed2 = true;
        (grid->getBlockInfoAll(info.level, info.Z)).changed2 = info.changed2;
      }
    }

    // 1.Change states of blocks next to finer resolution blocks
    // 2.Change states of blocks next to same resolution blocks
    // 3.Compress a block only if all blocks with the same parent need
    // compression
    bool clean_boundary = true;
    for (int m = levelMax - 1; m >= levelMin; m--) {
      // 1.
      for (size_t j = 0; j < I.size(); j++) {
        BlockInfo &info = I[j];
        if (info.level == m && info.state != Refine &&
            info.level != levelMax - 1) {
          const int TwoPower = 1 << info.level;
          const bool xskin = info.index[0] == 0 ||
                             info.index[0] == blocksPerDim[0] * TwoPower - 1;
          const bool yskin = info.index[1] == 0 ||
                             info.index[1] == blocksPerDim[1] * TwoPower - 1;
          const bool zskin = info.index[2] == 0 ||
                             info.index[2] == blocksPerDim[2] * TwoPower - 1;
          const int xskip = info.index[0] == 0 ? -1 : 1;
          const int yskip = info.index[1] == 0 ? -1 : 1;
          const int zskip = info.index[2] == 0 ? -1 : 1;

          for (int icode = 0; icode < 27; icode++) {
            if (info.state == Refine)
              break;
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                                 (icode / 9) % 3 - 1};
            if (!xperiodic && code[0] == xskip && xskin)
              continue;
            if (!yperiodic && code[1] == yskip && yskin)
              continue;
            if (!zperiodic && code[2] == zskip && zskin)
              continue;
            if (code[2] != 0)
              continue;

            if (grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                    .CheckFiner()) {
              if (info.state == Compress) {
                info.state = Leave;
                (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              }
              // if (info.level == levelMax - 1) break;

              const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]);
              int Bstep = 1; // face
              if (tmp == 2)
                Bstep = 3; // edge
              else if (tmp == 3)
                Bstep = 4; // corner

              // loop over blocks that make up face/edge/corner(respectively 4,2
              // or 1 blocks)
              for (int B = 0; B <= 1; B += Bstep) {
                const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                const int iNei = 2 * info.index[0] + std::max(code[0], 0) +
                                 code[0] +
                                 (B % 2) * std::max(0, 1 - abs(code[0]));
                const int jNei = 2 * info.index[1] + std::max(code[1], 0) +
                                 code[1] + aux * std::max(0, 1 - abs(code[1]));
                const long long zzz = grid->getZforward(m + 1, iNei, jNei);
                BlockInfo &FinerNei = grid->getBlockInfoAll(m + 1, zzz);
                State NeiState = FinerNei.state;
                if (NeiState == Refine) {
                  info.state = Refine;
                  (grid->getBlockInfoAll(info.level, info.Z)).state = Refine;
                  info.changed2 = true;
                  (grid->getBlockInfoAll(info.level, info.Z)).changed2 = true;
                  break;
                }
              }
            }
          }
        }
      }

      grid->UpdateBoundary(clean_boundary);
      clean_boundary = false;
      if (m == levelMin)
        break;

      // 2.
      for (size_t j = 0; j < I.size(); j++) {
        BlockInfo &info = I[j];
        if (info.level == m && info.state == Compress) {
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

          for (int icode = 0; icode < 27; icode++) {
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                                 (icode / 9) % 3 - 1};
            if (!xperiodic && code[0] == xskip && xskin)
              continue;
            if (!yperiodic && code[1] == yskip && yskin)
              continue;
            if (!zperiodic && code[2] == zskip && zskin)
              continue;
            if (code[2] != 0)
              continue;

            BlockInfo &infoNei = grid->getBlockInfoAll(
                info.level, info.Znei_(code[0], code[1], code[2]));
            if (grid->Tree(infoNei).Exists() && infoNei.state == Refine) {
              info.state = Leave;
              (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              break;
            }
          }
        }
      }
    } // m

    // 3.
    for (size_t jjj = 0; jjj < I.size(); jjj++) {
      BlockInfo &info = I[jjj];
      const int m = info.level;
      bool found = false;
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++)
          for (int k = 2 * (info.index[2] / 2);
               k <= 2 * (info.index[2] / 2) + 1; k++) {
            const long long n = grid->getZforward(m, i, j);
            BlockInfo &infoNei = grid->getBlockInfoAll(m, n);
            if (grid->Tree(infoNei).Exists() == false ||
                infoNei.state != Compress) {
              found = true;
              if (info.state == Compress) {
                info.state = Leave;
                (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              }
              break;
            }
          }
      if (found)
        for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
             i++)
          for (int j = 2 * (info.index[1] / 2);
               j <= 2 * (info.index[1] / 2) + 1; j++)
            for (int k = 2 * (info.index[2] / 2);
                 k <= 2 * (info.index[2] / 2) + 1; k++) {
              const long long n = grid->getZforward(m, i, j);
              BlockInfo &infoNei = grid->getBlockInfoAll(m, n);
              if (grid->Tree(infoNei).Exists() && infoNei.state == Compress)
                infoNei.state = Leave;
            }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Virtual functions that can be overwritten by user
  ////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief How cells are interpolated after refinement
   *
   * Default interpolation is a 2nd order Taylor expansion. Can be overidden by
   * a derived class, to enable a custom refinement interpolation.
   *
   * @param B Pointers to the eight new blocks to be interpolated.
   */
  virtual void RefineBlocks(BlockType *B[8], TLab &Lab) {
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;

    int offsetX[2] = {0, nx / 2};
    int offsetY[2] = {0, ny / 2};

    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        BlockType &b = *B[J * 2 + I];
        b.clear();

        for (int j = 0; j < ny; j += 2)
          for (int i = 0; i < nx; i += 2) {
            ElementType dudx =
                0.5 * (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J]) -
                       Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J]));
            ElementType dudy =
                0.5 * (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] + 1) -
                       Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] - 1));

            ElementType dudx2 =
                (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J]) +
                 Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J])) -
                2.0 * Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]);
            ElementType dudy2 =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] + 1) +
                 Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] - 1)) -
                2.0 * Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]);

            ElementType dudxdy =
                0.25 * ((Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J] + 1) +
                         Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J] - 1)) -
                        (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J] - 1) +
                         Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J] + 1)));

            b(i, j, 0) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (-0.25 * dudx - 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) + 0.0625 * dudxdy);
            b(i + 1, j, 0) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (+0.25 * dudx - 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) - 0.0625 * dudxdy);
            b(i, j + 1, 0) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (-0.25 * dudx + 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) - 0.0625 * dudxdy);
            b(i + 1, j + 1, 0) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (+0.25 * dudx + 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) + 0.0625 * dudxdy);
          }
      }
  }

  /**
   * @brief Refinement criterion.
   *
   * Default refinement criterion is to compare the 'magnitude()' of each
   * gridpoint to Rtol and Ctol. Can be overidden by a derived class, to enable
   * a custom refinement criterion.
   *
   * @param info BlockInfo to be tagged.
   */
  virtual State TagLoadedBlock(BlockInfo &info) {
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    BlockType &b = *(BlockType *)info.ptrBlock;

    double Linf = 0.0;
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++) {
        Linf = std::max(Linf, std::fabs(b(i, j).magnitude()));
      }

    if (Linf > tolerance_for_refinement)
      return Refine;
    else if (Linf < tolerance_for_compression)
      return Compress;

    return Leave;
  }
};

} // namespace cubism
