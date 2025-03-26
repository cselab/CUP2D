#pragma once

#include <algorithm>
#include <unordered_map>

#ifdef CUBISM_USE_ONETBB
#include <tbb/concurrent_unordered_map.h>
#endif

#include "BlockInfo.h"
#include "FluxCorrection.h"

namespace cubism
{

/** When dumping a Grid, blocks are grouped into larger rectangular regions
 *  of uniform resolution. These regions (BlockGroups) have blocks with the 
 *  same level and with various Space-filling-curve coordinates Z.
 *  They have NXX x NYY x NZZ grid points, grid spacing h, an origin and a
 *  minimum and maximum index (indices of bottom left and top right blocks).
 */
struct BlockGroup
{
   int i_min[3]; ///< min (i,j,k) index of a block of this group
   int i_max[3]; ///< max (i,j,k) index of a block of this group
   int level; ///< refinement level
   std::vector<long long> Z; ///< Z-order indices of blocks of this group
   size_t ID; ///< unique group number
   double origin[3]; ///< Coordinates (x,y,z) of origin
   double h; ///< Grid spacing of the group
   int NXX; ///< Grid points of the group in the x-direction
   int NYY; ///< Grid points of the group in the y-direction
   int NZZ; ///< Grid points of the group in the z-direction
};


/** Holds the GridBlocks and their meta-data (BlockInfos).
 * This class provides information about the current state of the Octree of blocks in the 
 * simulation. The user can request if a particular block is present in the Octree or if its parent/
 * children block(s) are present instead. This class also provides access to the raw data from the
 * simulation.
 */
template <typename Block, template <typename X> class allocator = std::allocator>
class Grid
{
 public:
   typedef Block BlockType;
   using ElementType = typename Block::ElementType; ///<Blocks hold ElementTypes
   typedef typename Block::RealType Real; ///< Blocks must provide `RealType`.

   #ifdef CUBISM_USE_ONETBB
   tbb::concurrent_unordered_map<long long, BlockInfo *> BlockInfoAll;
   tbb::concurrent_unordered_map<long long, TreePosition> Octree;
   #else

   /** A map from unique BlockInfo IDs to pointers to BlockInfos.
    *  Should be accessed through function 'getBlockInfoAll'. If a Block does not belong to this
    *  rank and it is not adjacent to it, this map should not return something meaningful.
    */
   std::unordered_map<long long, BlockInfo *> BlockInfoAll;

   /** A map from unique BlockInfo IDs to pointers to integers (TreePositions) that encode whether 
    *  a BlockInfo is present in the Octree (and to which rank it belongs to) or not. This is a 
    *  seperate object from BlockInfoAll because all the information we need for some blocks is 
    *  merely whether they exist or not (i.e. we don't need their grid spacing or other meta-data)
    *  held by BlockInfos.
    */
   std::unordered_map<long long, TreePosition> Octree;
   #endif

   /** Meta-data for blocks that belong to this rank.
    *  This vector holds all the BlockInfos for blocks that belong to this rank. When the mesh 
    *  changes, the contents of this vector become outdated and need to be updated. This is done
    *  through the FillPos() function. This array should be used when iterating over the blocks 
    *  owned by a Grid.
    */ 
   std::vector<BlockInfo> m_vInfo;

   const int NX;                     ///< Total # of blocks for level 0 in X-direction
   const int NY;                     ///< Total # of blocks for level 0 in Y-direction
   const int NZ;                     ///< Total # of blocks for level 0 in Z-direction
   const double maxextent;           ///< Maximum domain extent
   const int levelMax;               ///< Maximum refinement level allowed
   const int levelStart;             ///< Initial refinement level
   const bool xperiodic;             ///< grid periodicity in x-direction
   const bool yperiodic;             ///< grid periodicity in y-direction
   const bool zperiodic;             ///< grid periodicity in z-direction
   std::vector<BlockGroup> MyGroups; ///< used for dumping data
   std::vector<long long> level_base;///< auxiliary array used when searching is std::unordered_map
   bool UpdateFluxCorrection{true};  ///< FluxCorrection updates only when grid is refined/compressed
   bool UpdateGroups{true};          ///< (inactive) BlockGroups updated only when this is true
   bool FiniteDifferences{true};     ///< used by BlockLab, to determine what kind of coarse-fine interface interpolation to make.true means that biased stencils will be used to get an O(h^3) approximation
   FluxCorrection<Grid> CorrectorGrid; ///< used for AMR flux-corrections at coarse-fine interfaces

   ///Get the TreePosition of block with Z-order index 'm', at refinement level 'n'.
   TreePosition &Tree(const int m, const long long n)
   {
      /*
       * Return the position in the Octree of a Block at level m and SFC coordinate n.
       */
      const long long aux = level_base[m] + n;
      const auto retval   = Octree.find(aux);
      if (retval == Octree.end())
      {
         #ifndef CUBISM_USE_ONETBB
         #pragma omp critical
         #endif
         {
            const auto retval1 = Octree.find(aux);
            if (retval1 == Octree.end())
            {
               TreePosition dum;
               Octree[aux] = dum;
            }
         }
         return Tree(m, n);
      }
      else
      {
         return retval->second;
      }
   }
   ///Get the TreePosition of block with BlockInfo 'info'.
   TreePosition &Tree(BlockInfo &info) { return Tree(info.level, info.Z); }
   ///Get the TreePosition of block with BlockInfo 'info'.
   TreePosition &Tree(const BlockInfo &info) { return Tree(info.level, info.Z); }

   ///Called in constructor to allocate all blocks at level=levelStart.
   void _alloc()
   {
      const int m        = levelStart;
      const int TwoPower = 1 << m;
      for (long long n = 0; n < NX * NY * NZ * pow(TwoPower, DIMENSION); n++)
      {
         Tree(m, n).setrank(0);
         _alloc(m, n);
      }
      if (m - 1 >= 0)
      {
         for (long long n = 0; n < NX * NY * NZ * pow((1 << (m - 1)), DIMENSION); n++) Tree(m - 1, n).setCheckFiner();
      }
      if (m + 1 < levelMax)
      {
         for (long long n = 0; n < NX * NY * NZ * pow((1 << (m + 1)), DIMENSION); n++) Tree(m + 1, n).setCheckCoarser();
      }
      FillPos();
   }

   ///Called to allocate a block with Z-order index 'm' at refinement level 'n', when the grid is refined.
   void _alloc(const int m, const long long n)
   {
      allocator<Block> alloc;

      BlockInfo & new_info = getBlockInfoAll(m, n);
      new_info.ptrBlock = alloc.allocate(1);
      #pragma omp critical
      {
         m_vInfo.push_back(new_info);
      }
      Tree(m, n).setrank(rank());
   }

   ///Called in destructor to deallocate all blocks.
   void _deallocAll()
   {
      allocator<Block> alloc;
      for (size_t i = 0; i < m_vInfo.size(); i++)
      {
         const int m       = m_vInfo[i].level;
         const long long n = m_vInfo[i].Z;
         alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
      }
      std::vector<long long> aux;
      for (auto & m: BlockInfoAll)
         aux.push_back(m.first);
      for (size_t i = 0 ; i < aux.size() ; i++)
      {
         const auto retval = BlockInfoAll.find(aux[i]);
         if (retval != BlockInfoAll.end())
         {
            delete retval->second;
         }
      }
      m_vInfo.clear();
      BlockInfoAll.clear();
      Octree.clear();
   }

   ///Called to deallocate a block with Z-order index 'm' at refinement level 'n', when the grid is compressed.
   void _dealloc(const int m, const long long n)
   {
      allocator<Block> alloc;
      alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
      for (size_t j = 0; j < m_vInfo.size(); j++)
      {
         if (m_vInfo[j].level == m && m_vInfo[j].Z == n)
         {
            m_vInfo.erase(m_vInfo.begin() + j);
            return;
         }
      }
   }

   ///Called to deallocate many blocks with blockIDs in the vector 'dealloc_IDs'
   void dealloc_many(const std::vector<long long> & dealloc_IDs)
   {
      for (size_t j = 0; j < m_vInfo.size(); j++) m_vInfo[j].changed2 = false;

      allocator<Block> alloc;

      for (size_t i = 0; i < dealloc_IDs.size() ; i++)
      for (size_t j = 0; j < m_vInfo.size()     ; j++)
      {
         if (m_vInfo[j].blockID_2 == dealloc_IDs[i])
         {
            const int m = m_vInfo[j].level;
            const long long n = m_vInfo[j].Z;
            m_vInfo[j].changed2 = true;
            alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
            break;
         }
      }
      //for c++20
      //std::erase_if(m_vInfo, [](BlockInfo & x) { return x.changed2; });
      //for c++17
      m_vInfo.erase(std::remove_if(m_vInfo.begin(),m_vInfo.end(),[](const BlockInfo & x){return x.changed2;}),m_vInfo.end());
   }

   /** Used when Block at level m_new with SFC coordinate n_new is added to the Grid
    *  as a result of compression of Block (m,n). Sets the state of the newly added 
    *  Block. It also replaces BlockInfo(m,n) and Block(m,n) with 
    *  BlockInfo(m_new,n_new) and Block(m_new,n_new).
    * @param m: Refinement level of the GridBlock that is compressed.
    * @param n: Z-order index of the GridBlock that is compressed.
    * @param m_new: Refinement level of the GridBlock that will replace the compressed GridBlock.
    * @param n_new: Z-order index of the GridBlock that will replace the compressed GridBlock.
    */
   void FindBlockInfo(const int m, const long long n, const int m_new, const long long n_new)
   {
      for (size_t j = 0; j < m_vInfo.size(); j++)
         if (m == m_vInfo[j].level && n == m_vInfo[j].Z)
         {
            BlockInfo & correct_info = getBlockInfoAll(m_new, n_new);
            correct_info.state = Leave;
            m_vInfo[j] = correct_info;
            return;
         }
   }

   /** The data in BlockInfoAll is always correct (states, blockIDs etc.), but this 
    * is not the case for m_vInfo, whose content might be outdated
    * after grid refinement/compression or exchange of blocks between different 
    * ranks. This function updates their content.
    * @param CopyInfos: set to true if the correct BlockInfos from BlockInfoAll should be copied to m_vInfo. Otherwise only selected variables are copied.
    */
   virtual void FillPos(bool CopyInfos = true)
   {
      std::sort(m_vInfo.begin(), m_vInfo.end()); //sort according to blockID_2

      #ifndef CUBISM_USE_ONETBB
      //The following will reserve memory for the unordered map.
      //This will result in a thread-safe Tree(m,n) function
      //as Octree will not change size when it is accessed by
      //multiple threads. The number m_vInfo.size()/8 is arbitrary.
      Octree.reserve(Octree.size() + m_vInfo.size()/8);
      #endif

      if (CopyInfos)
         for (size_t j = 0; j < m_vInfo.size(); j++)
         {
            const int m       = m_vInfo[j].level;
            const long long n = m_vInfo[j].Z;
            BlockInfo & correct_info = getBlockInfoAll(m, n);
            correct_info.blockID = j;
            m_vInfo[j]  = correct_info;
            assert(Tree(m, n).Exists());
         }
      else
         for (size_t j = 0; j < m_vInfo.size(); j++)
         {
            const int m       = m_vInfo[j].level;
            const long long n = m_vInfo[j].Z;
            BlockInfo & correct_info = getBlockInfoAll(m, n);
            correct_info.blockID = j;
            m_vInfo[j].blockID = j;
            m_vInfo[j].state = correct_info.state;
            assert(Tree(m, n).Exists());
         }
   }

   /** Constructor.
    * @param _NX: total number of blocks in the x-direction, at the coarsest refinement level.
    * @param _NY: total number of blocks in the y-direction, at the coarsest refinement level.
    * @param _NZ: total number of blocks in the z-direction, at the coarsest refinement level.
    * @param _maxextent: maximum extent of the simulation (largest side of the rectangular domain).
    * @param _levelStart: refinement level where all allocated GridBlocks will be
    * @param _levelMax: maximum refinement level allowed
    * @param AllocateBlocks: true if GridBlocks should be allocated (false if they are allocated by a derived class)
    * @param a_xperiodic: true if the domain is periodic in the x-direction
    * @param a_yperiodic: true if the domain is periodic in the y-direction
    * @param a_zperiodic: true if the domain is periodic in the z-direction
    */
   Grid(const unsigned int _NX, 
        const unsigned int _NY = 1,
        const unsigned int _NZ = 1,
        const double _maxextent = 1,
        const unsigned int _levelStart = 0,
        const unsigned int _levelMax = 1,
        const bool AllocateBlocks = true,
        const bool a_xperiodic = true,
        const bool a_yperiodic = true,
        const bool a_zperiodic = true)
       : NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent), levelMax(_levelMax), levelStart(_levelStart),
         xperiodic(a_xperiodic), yperiodic(a_yperiodic), zperiodic(a_zperiodic)
   {
      BlockInfo dummy;
      #if DIMENSION == 3
      const int nx = dummy.blocks_per_dim(0, NX, NY, NZ);
      const int ny = dummy.blocks_per_dim(1, NX, NY, NZ);
      const int nz = dummy.blocks_per_dim(2, NX, NY, NZ);
      #else
      const int nx = dummy.blocks_per_dim(0, NX, NY);
      const int ny = dummy.blocks_per_dim(1, NX, NY);
      const int nz = 1;
      #endif
      const int lvlMax = dummy.levelMax(levelMax);

      for (int m = 0; m < lvlMax; m++)
      {
         const int TwoPower   = 1 << m;
         const long long Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
         if (m == 0) level_base.push_back(Ntot);
         if (m > 0) level_base.push_back(level_base[m - 1] + Ntot);
      }
      if (AllocateBlocks) _alloc();
   }

   /// Destructor
   virtual ~Grid() { _deallocAll(); }

   /// Returns GridBlock at level 'm' with Z-index 'n'
   virtual Block *avail(const int m, const long long n) { return (Block *)getBlockInfoAll(m, n).ptrBlock; }

   /// Returns MPI ranks of this Grid
   virtual int rank() const { return 0; }

   /**Given two vectors with the SFC coordinate (Z) and the level of each block, this function
   * will erase the current structure of the grid and create a new one, with the given blocks.
   * This is used when reading data from file (possibly to restart) or when initializing the
   * simulation.*/
   virtual void initialize_blocks(const std::vector<long long> & blocksZ, const std::vector<short int> & blockslevel)
   {
      _deallocAll();

      for (size_t i = 0 ; i < blocksZ.size() ; i++)
      {
         const int level   = blockslevel[i];
         const long long Z = blocksZ[i];

         _alloc(level, Z);
         Tree(level, Z).setrank(rank());

         #if DIMENSION == 3
            int p[3];
            BlockInfo::inverse(Z, level, p[0], p[1], p[2]);
            if (level < levelMax - 1)
               for (int k1 = 0; k1 < 2; k1++)
               for (int j1 = 0; j1 < 2; j1++)
               for (int i1 = 0; i1 < 2; i1++)
               {
                  const long long nc = getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1, 2 * p[2] + k1);
                  Tree(level + 1, nc).setCheckCoarser();
               }
            if (level > 0)
            {
               const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
               Tree(level - 1, nf).setCheckFiner();
            }
         #else
            int p[2];
            BlockInfo::inverse(Z, level, p[0], p[1]);
            if (level < levelMax - 1)
               for (int j1 = 0; j1 < 2; j1++)
               for (int i1 = 0; i1 < 2; i1++)
               {
                  const long long nc = getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
                  Tree(level + 1, nc).setCheckCoarser();
               }
            if (level > 0)
            {
               const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
               Tree(level - 1, nf).setCheckFiner();
            }
         #endif
      }
      FillPos();
      UpdateFluxCorrection = true;
      UpdateGroups = true;
   }

   #if DIMENSION == 3
   /// Returns Z-index of GridBlock with indices ijk (ix,iy,iz) at level 'level' 
   long long getZforward(const int level, const int i, const int j, const int k) const
   {
      const int TwoPower = 1 << level;
      const int ix       = (i + TwoPower * NX) % (NX * TwoPower);
      const int iy       = (j + TwoPower * NY) % (NY * TwoPower);
      const int iz       = (k + TwoPower * NZ) % (NZ * TwoPower);
      return BlockInfo::forward(level, ix, iy, iz);
   }

   /// Returns GridBlock with indices ijk (ix,iy,iz) at level 'm' 
   Block *avail1(const int ix, const int iy, const int iz, const int m)
   {
      const long long n = getZforward(m, ix, iy, iz);
      return avail(m, n);
   }

   #else // DIMENSION = 2

   /// Returns Z-index of GridBlock with indices ij (ix,iy) at level 'level' 
   long long getZforward(const int level, const int i, const int j) const
   {
      const int TwoPower = 1 << level;
      const int ix       = (i + TwoPower * NX) % (NX * TwoPower);
      const int iy       = (j + TwoPower * NY) % (NY * TwoPower);
      return BlockInfo::forward(level, ix, iy);
   }

   /// Returns GridBlock with indices ij (ix,iy) at level 'm' 
   Block *avail1(const int ix, const int iy, const int m)
   {
      const long long n = getZforward(m, ix, iy);
      return avail(m, n);
   }

   #endif

   /// Used to iterate though all blocks (ID=0,...,m_vInfo.size()-1)
   Block &operator()(const long long ID)
   {
      return *(Block *)m_vInfo[ID].ptrBlock;
   }

   /// Returns the number of blocks at refinement level 0
   std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }

   /// Returns the number of blocks at refinement level 'levelMax-1'
   std::array<int, 3> getMaxMostRefinedBlocks() const
   {
      return {
       NX << (levelMax - 1),
       NY << (levelMax - 1),
       DIMENSION == 3 ? (NZ << (levelMax - 1)) : 1,
      };
   }

   /// Returns the number of grid points at refinement level 'levelMax-1'
   std::array<int, 3> getMaxMostRefinedCells() const
   {
      const auto b = getMaxMostRefinedBlocks();
      return {b[0] * Block::sizeX, b[1] * Block::sizeY, b[2] * Block::sizeZ};
   }

   /// Returns the maximum refinement level allowed
   inline int getlevelMax() const { return levelMax; }

   /**
    * Access BlockInfo at level m with Space-Filling-Curve coordinate n.
    * If the BlockInfo has not been allocated (not found in the std::unordered_map), 
    * allocate it as well.
    */
   BlockInfo &getBlockInfoAll(const int m, const long long n)
   {
      const long long aux = level_base[m] + n;
      const auto retval   = BlockInfoAll.find(aux);
      if (retval != BlockInfoAll.end())
      {
         return *retval->second;
      }
      else
      {
         #ifndef CUBISM_USE_ONETBB
         #pragma omp critical
         #endif
         {
            const auto retval1 = BlockInfoAll.find(aux);
            if (retval1 == BlockInfoAll.end())
            {
               BlockInfo *dumm = new BlockInfo();
               const int TwoPower    = 1 << m;
               const double h0 = (maxextent / std::max(NX * Block::sizeX, std::max(NY * Block::sizeY, NZ * Block::sizeZ)));
               const double h  = h0 / TwoPower;
               double origin[3];
               int i, j, k;
               #if DIMENSION == 3
               BlockInfo::inverse(n, m, i, j, k);
               #else
               BlockInfo::inverse(n, m, i, j);
               k = 0;
               #endif
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

   /// Returns the vector of BlockInfos of this Grid
   std::vector<BlockInfo> &getBlocksInfo() { return m_vInfo; }

   /// Returns the vector of BlockInfos of this Grid
   const std::vector<BlockInfo> &getBlocksInfo() const { return m_vInfo; }

   /// Returns the total number of MPI processes
   virtual int get_world_size() const { return 1; }

   /// Does nothing for a single rank (no MPI)
   virtual void UpdateBoundary(bool clean=false){}

   /// Used to create BlockGroups, when the Grid is to be dumped to a file
   void UpdateMyGroups()
   {
      /*
       * This function is used before dumping the Grid. It groups adjacent blocks of the same
       * resolution (and owned by the same MPI rank) to BlockGroups that will be dumped as
       * a collection of rectangular uniform grids.
       */

      // if (!UpdateGroups) return; //TODO : does not work for CUP2D
      if (rank() == 0) std::cout << "Updating groups..." << std::endl;

      const unsigned int nX = BlockType::sizeX;
      const unsigned int nY = BlockType::sizeY;
      const size_t Ngrids   = getBlocksInfo().size();
      const auto &MyInfos   = getBlocksInfo();
      UpdateGroups          = false;
      MyGroups.clear();
      std::vector<bool> added(MyInfos.size(), false);

      #if DIMENSION == 3
      const unsigned int nZ = BlockType::sizeZ;
      for (unsigned int m = 0; m < Ngrids; m++)
      {
         const BlockInfo &I = MyInfos[m];

         if (added[I.blockID]) continue;
         added[I.blockID] = true;
         BlockGroup newGroup;

         newGroup.level = I.level;
         newGroup.h     = I.h;
         newGroup.Z.push_back(I.Z);

         const int base[3] = {I.index[0], I.index[1], I.index[2]};
         int i_off[6] = {};
         bool ready_[6] = {};

         int d    = 0;
         auto blk = getMaxBlocks();
         do
         {
            if (ready_[d] == false)
            {
               bool valid = true;
               i_off[d]++;
               const int i0 = (d < 3) ? (base[d] - i_off[d]) : (base[d - 3] + i_off[d]);
               const int d0 = (d < 3) ? (d) % 3 : (d - 3) % 3;
               const int d1 = (d < 3) ? (d + 1) % 3 : (d - 3 + 1) % 3;
               const int d2 = (d < 3) ? (d + 2) % 3 : (d - 3 + 2) % 3;

               for (int i2 = base[d2] - i_off[d2]; i2 <= base[d2] + i_off[d2 + 3]; i2++)
                  for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 3]; i1++)
                  {
                     if (valid == false) break;

                     if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= blk[d0] * (1 << I.level) ||
                         i1 >= blk[d1] * (1 << I.level) || i2 >= blk[d2] * (1 << I.level))
                     {
                        valid = false;
                        break;
                     }
                     long long n;
                     if (d == 0 || d == 3) n = getZforward(I.level, i0, i1, i2);
                     else if (d == 1 || d == 4)
                        n = getZforward(I.level, i2, i0, i1);
                     else /*if (d==2||d==5)*/
                        n = getZforward(I.level, i1, i2, i0);

                     if (Tree(I.level, n).rank() != rank())
                     {
                        valid = false;
                        break;
                     }
                     if (added[getBlockInfoAll(I.level, n).blockID] == true)
                     {
                        valid = false;
                     }
                  }

               if (valid == false)
               {
                  i_off[d]--;
                  ready_[d] = true;
               }
               else
               {
                  for (int i2 = base[d2] - i_off[d2]; i2 <= base[d2] + i_off[d2 + 3]; i2++)
                     for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 3]; i1++)
                     {
                        long long n;
                        if (d == 0 || d == 3) n = getZforward(I.level, i0, i1, i2);
                        else if (d == 1 || d == 4)
                           n = getZforward(I.level, i2, i0, i1);
                        else /*if (d==2||d==5)*/
                           n = getZforward(I.level, i1, i2, i0);
                        newGroup.Z.push_back(n);
                        added[getBlockInfoAll(I.level, n).blockID] = true;
                     }
               }
            }
            d = (d + 1) % 6;
         } while (ready_[0] == false || ready_[1] == false || ready_[2] == false || ready_[3] == false ||
                  ready_[4] == false || ready_[5] == false);

         const int ix_min = base[0] - i_off[0];
         const int iy_min = base[1] - i_off[1];
         const int iz_min = base[2] - i_off[2];
         const int ix_max = base[0] + i_off[3];
         const int iy_max = base[1] + i_off[4];
         const int iz_max = base[2] + i_off[5];

         long long n_base = getZforward(I.level, ix_min, iy_min, iz_min);

         newGroup.i_min[0] = ix_min;
         newGroup.i_min[1] = iy_min;
         newGroup.i_min[2] = iz_min;

         newGroup.i_max[0] = ix_max;
         newGroup.i_max[1] = iy_max;
         newGroup.i_max[2] = iz_max;

         const BlockInfo & info = getBlockInfoAll(I.level, n_base);
         newGroup.origin[0] = info.origin[0];
         newGroup.origin[1] = info.origin[1];
         newGroup.origin[2] = info.origin[2];

         newGroup.NXX = (newGroup.i_max[0] - newGroup.i_min[0] + 1) * nX + 1;
         newGroup.NYY = (newGroup.i_max[1] - newGroup.i_min[1] + 1) * nY + 1;
         newGroup.NZZ = (newGroup.i_max[2] - newGroup.i_min[2] + 1) * nZ + 1;

         MyGroups.push_back(newGroup);
      }
      #else
      for (unsigned int m = 0; m < Ngrids; m++)
      {
         const BlockInfo &I = MyInfos[m];

         if (added[I.blockID]) continue;
         added[I.blockID] = true;
         BlockGroup newGroup;

         newGroup.level = I.level;
         newGroup.h     = I.h;
         newGroup.Z.push_back(I.Z);

         const int base[3] = {I.index[0], I.index[1], 0};  // I.index[2]
         int i_off[4] = {};
         bool ready_[4] = {};

         int d    = 0;
         auto blk = getMaxBlocks();
         do
         {
            if (ready_[d] == false)
            {
               bool valid = true;
               i_off[d]++;
               const int i0 = (d < 2) ? (base[d] - i_off[d]) : (base[d - 2] + i_off[d]);
               const int d0 = (d < 2) ? (d) % 2 : (d - 2) % 2;
               const int d1 = (d < 2) ? (d + 1) % 2 : (d - 2 + 1) % 2;

               for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 2]; i1++)
               {
                  if (valid == false) break;

                  if (i0 < 0 || i1 < 0 || i0 >= blk[d0] * (1 << I.level) || i1 >= blk[d1] * (1 << I.level))
                  {
                     valid = false;
                     break;
                  }
                  long long n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1) : getZforward(I.level, i1, i0);

                  if (Tree(I.level, n).rank() != rank())
                  {
                     valid = false;
                     break;
                  }
                  if (added[getBlockInfoAll(I.level, n).blockID] == true)
                  {
                     valid = false;
                  }
               }

               if (valid == false)
               {
                  i_off[d]--;
                  ready_[d] = true;
               }
               else
               {
                  for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 2]; i1++)
                  {
                     long long n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1) : getZforward(I.level, i1, i0);
                     newGroup.Z.push_back(n);
                     added[getBlockInfoAll(I.level, n).blockID] = true;
                  }
               }
            }
            d = (d + 1) % 4;
         } while (ready_[0] == false || ready_[1] == false || ready_[2] == false || ready_[3] == false);

         const int ix_min = base[0] - i_off[0];
         const int iy_min = base[1] - i_off[1];
         const int iz_min = 0; // base[2] - i_off[2];
         const int ix_max = base[0] + i_off[2];
         const int iy_max = base[1] + i_off[3];
         const int iz_max = 0; // base[2] + i_off[5];

         long long n_base = getZforward(I.level, ix_min, iy_min);

         newGroup.i_min[0] = ix_min;
         newGroup.i_min[1] = iy_min;
         newGroup.i_min[2] = iz_min;

         newGroup.i_max[0] = ix_max;
         newGroup.i_max[1] = iy_max;
         newGroup.i_max[2] = iz_max;

         const BlockInfo & info = getBlockInfoAll(I.level, n_base);
         newGroup.origin[0] = info.origin[0];
         newGroup.origin[1] = info.origin[1];
         newGroup.origin[2] = info.origin[2];

         newGroup.NXX = (newGroup.i_max[0] - newGroup.i_min[0] + 1) * nX + 1;
         newGroup.NYY = (newGroup.i_max[1] - newGroup.i_min[1] + 1) * nY + 1;
         newGroup.NZZ = 2; //(newGroup.i_max[2] - newGroup.i_min[2] + 1)*nZ + 1;

         MyGroups.push_back(newGroup);
      }
      #endif
   }
};

} // namespace cubism
