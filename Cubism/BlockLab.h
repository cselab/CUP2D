#pragma once

#include "BlockInfo.h"
#include "Matrix3D.h"
#include "StencilInfo.h"
#include <cstring>
#include <math.h>
#include <string>
#include <array>

namespace cubism
{
#define memcpy2(a, b, c) memcpy((a), (b), (c))

//default coarse-fine interpolation stencil
#if DIMENSION == 3
    constexpr int default_start [3] = {-1,-1,-1};
    constexpr int default_end   [3] = {2,2,2}; 
#else
    constexpr int default_start [3] = {-1,-1,0};
    constexpr int default_end   [3] = {2,2,1}; 
#endif

/** \brief Copy of a Gridblock plus halo cells.*/
/** This class provides the user a copy of a Gridblock that is extended by a layer of halo cells.
 *  To define one instance of it, the user needs to provide a 'TGrid' type in the template 
 *  parameters. From this, the BlockType and ElementType and inferred, which are the GridBlock
 *  class and Element type stored at each gridpoint of the mesh.
 *  To use a BlockLab, the user first needs to call 'prepare', which will provide the BlockLab with
 *  the stencil of points needed for a particular computation. To get an array of a particular
 *  GridBlock (+halo cells), the user should call 'load' and provide it with the BlockInfo that is
 *  associated with the GridBlock of interest. Once this is done, gridpoints in the GridBlock and
 *  halo cells can be accessed with the (x,y,z) operator. For example, (-1,0,0) would access a 
 *  halo cell in the -x direction.
 *  @tparam TGrid: the kind of Grid/GridMPI halo cells are needed for
 *  @tparam allocator: a class responsible for allocation of memory for this BlockLab
 */
template <typename TGrid, template <typename X> class allocator = std::allocator>
class BlockLab
{
 public:
   using GridType = TGrid; ///< should be a 'Grid', 'GridMPI' or derived class 
   using BlockType = typename GridType::BlockType; ///< GridBlock type used by TGrid
   using ElementType = typename BlockType::ElementType; ///< Element type used by GridBlock type
   using Real = typename ElementType::RealType; ///< Number type used by Element (double/float etc.)

 protected:
   Matrix3D<ElementType, allocator> *m_cacheBlock; ///< working array of GridBlock + halo cells.
   int m_stencilStart[3]; ///< starts of stencil for halo cells
   int m_stencilEnd[3]; ///< ends of stencil fom halo cells
   bool istensorial;///< whether the stencil is tensorial or not (see also StencilInfo struct)
   bool use_averages;///< if true, fine blocks average down their cells to provide halo cells for coarse blocks (2nd order accurate). If false, they perform a 3rd-order accurate interpolation instead (which is the accuracy needed to compute 2nd derivatives).
   GridType *m_refGrid;///< Point to TGrid instance
   int NX;///< GridBlock size in the x-direction.
   int NY;///< GridBlock size in the y-direction.
   int NZ;///< GridBlock size in the z-direction.
   std::array<BlockType *, 27> myblocks;///< Pointers to neighboring blocks of a GridBlock
   std::array<int, 27> coarsened_nei_codes;///< If a neighbor is at a coarser level, store it here
   int coarsened_nei_codes_size;///< Number of coarser neighbors
   int offset[3];///< like m_stencilStart but used when a coarse block sends cells to a finer block
   Matrix3D<ElementType, allocator> *m_CoarsenedBlock;///< coarsened version of given block
   int m_InterpStencilStart[3];///< stencil starts used for refinement (assumed tensorial)
   int m_InterpStencilEnd[3];///< stencil ends used for refinement (assumed tensorial)
   bool coarsened;///< true if block has at least one coarser neighbor
   int CoarseBlockSize[3];///< size of coarsened block (NX/2,NY/2,NZ/2)

   ///Coefficients used with upwind/central stencil of points with 3rd order interpolation of halo cells from fine to coarse blocks
   const double d_coef_plus[9] = {-0.09375, 0.4375,0.15625,  //starting point (+2,+1,0)
                                   0.15625,-0.5625,0.90625,  //last point     (-2,-1,0)
                                  -0.09375, 0.4375,0.15625}; //central point  (-1,0,+1)
   ///Coefficients used with upwind/central stencil of points with 3rd order interpolation of halo cells from fine to coarse blocks
   const double d_coef_minus[9]= { 0.15625,-0.5625, 0.90625, //starting point (+2,+1,0)
                                  -0.09375, 0.4375, 0.15625, //last point     (-2,-1,0)
                                   0.15625, 0.4375,-0.09375};//central point  (-1,0,+1)

 public:
   ///Constructor.
   BlockLab(): m_cacheBlock(nullptr), m_refGrid(nullptr),m_CoarsenedBlock(nullptr)
   {
      m_stencilStart[0] = m_stencilStart[1] = m_stencilStart[2] = 0;
      m_stencilEnd[0] = m_stencilEnd[1] = m_stencilEnd[2] = 0;
      m_InterpStencilStart[0] = m_InterpStencilStart[1] = m_InterpStencilStart[2] = 0;
      m_InterpStencilEnd[0] = m_InterpStencilEnd[1] = m_InterpStencilEnd[2] = 0;

      CoarseBlockSize[0] = (int)BlockType::sizeX / 2;
      CoarseBlockSize[1] = (int)BlockType::sizeY / 2;
      CoarseBlockSize[2] = (int)BlockType::sizeZ / 2;
      if (CoarseBlockSize[0] == 0) CoarseBlockSize[0] = 1;
      if (CoarseBlockSize[1] == 0) CoarseBlockSize[1] = 1;
      if (CoarseBlockSize[2] == 0) CoarseBlockSize[2] = 1;
   }

   ///Return a name for this BlockLab. Useful for derived instances with custom boundary conditions.
   virtual std::string name() const { return "BlockLab"; }

   ///true if boundary conditions are periodic in x-direction
   virtual bool is_xperiodic() { return true; }

   ///true if boundary conditions are periodic in y-direction
   virtual bool is_yperiodic() { return true; }

   ///true if boundary conditions are periodic in z-direction
   virtual bool is_zperiodic() { return true; }

   ///Destructor.
   ~BlockLab()
   {
      _release(m_cacheBlock);
      _release(m_CoarsenedBlock);
   }

   /**
    * Get a single element from the block.
    * stencil_start and stencil_end refer to the values passed in BlockLab::prepare().
    * @param ix: Index in x-direction (stencil_start[0] <= ix < BlockType::sizeX + stencil_end[0] - 1).
    * @param iy: Index in y-direction (stencil_start[1] <= iy < BlockType::sizeY + stencil_end[1] - 1).
    * @param iz: Index in z-direction (stencil_start[2] <= iz < BlockType::sizeZ + stencil_end[2] - 1).
    */
   ElementType &operator()(int ix, int iy = 0, int iz = 0)
   {
      assert(ix - m_stencilStart[0] >= 0 && ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
      assert(iy - m_stencilStart[1] >= 0 && iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
      assert(iz - m_stencilStart[2] >= 0 && iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
      return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1], iz - m_stencilStart[2]);
   }

   /// Just as BlockLab::operator() but const.
   const ElementType &operator()(int ix, int iy = 0, int iz = 0) const
   {
      assert(ix - m_stencilStart[0] >= 0 && ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
      assert(iy - m_stencilStart[1] >= 0 && iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
      assert(iz - m_stencilStart[2] >= 0 && iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
      return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1], iz - m_stencilStart[2]);
   }

   /// Just as BlockLab::operator() but returning a const.
   const ElementType &read(int ix, int iy = 0, int iz = 0) const
   {
      assert(ix - m_stencilStart[0] >= 0 && ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
      assert(iy - m_stencilStart[1] >= 0 && iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
      assert(iz - m_stencilStart[2] >= 0 && iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
      return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1], iz - m_stencilStart[2]);
   }

   /// Deallocate memory (used in destructor).
   void release()
   {
      _release(m_cacheBlock);
      _release(m_CoarsenedBlock);
   }

   /** Prepares the BlockLab for a given 'grid' and stencil of points.
    *  Allocates memory (if not already allocated) for the arrays that will hold the copy of a 
    *  GridBlock plus its halo cells. 
    * @param grid: the Grid/GridMPI with all the GridBlocks that will need halo cells
    * @param stencil: the StencilInfo for the halo cells
    * @param  Istencil_start: the starts of the stencil used for coarse-fine interpolation of halo cells, set to -1 for the default interpolation.
    * @param  Istencil_end: the ends of the stencil used for coarse-fine interpolation of halo cells, set to +2 for the default interpolation.
    */
   virtual void prepare(GridType &grid, const StencilInfo & stencil, const int Istencil_start[3]=default_start, const int Istencil_end[3]=default_end)
   {
      istensorial = stencil.tensorial;
      coarsened   = false;
      m_stencilStart[0] = stencil.sx;
      m_stencilStart[1] = stencil.sy;
      m_stencilStart[2] = stencil.sz;
      m_stencilEnd  [0] = stencil.ex;
      m_stencilEnd  [1] = stencil.ey;
      m_stencilEnd  [2] = stencil.ez;

      m_InterpStencilStart[0] = Istencil_start[0];
      m_InterpStencilStart[1] = Istencil_start[1];
      m_InterpStencilStart[2] = Istencil_start[2];
      m_InterpStencilEnd  [0] = Istencil_end  [0];
      m_InterpStencilEnd  [1] = Istencil_end  [1];
      m_InterpStencilEnd  [2] = Istencil_end  [2];

      assert(m_InterpStencilStart[0] <= m_InterpStencilEnd[0]);
      assert(m_InterpStencilStart[1] <= m_InterpStencilEnd[1]);
      assert(m_InterpStencilStart[2] <= m_InterpStencilEnd[2]);
      assert(stencil.sx              <= stencil.ex           );
      assert(stencil.sy              <= stencil.ey           );
      assert(stencil.sz              <= stencil.ez           );
      assert(stencil.sx              >= -BlockType::sizeX    );
      assert(stencil.sy              >= -BlockType::sizeY    );
      assert(stencil.sz              >= -BlockType::sizeZ    );
      assert(stencil.ex              <  2*BlockType::sizeX   );
      assert(stencil.ey              <  2*BlockType::sizeY   );
      assert(stencil.ez              <  2*BlockType::sizeZ   );

      m_refGrid = &grid;

      if (m_cacheBlock == NULL ||
          (int)m_cacheBlock->getSize()[0] != (int)BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1 ||
          (int)m_cacheBlock->getSize()[1] != (int)BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1 ||
          (int)m_cacheBlock->getSize()[2] != (int)BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1)
      {
         if (m_cacheBlock != NULL) _release(m_cacheBlock);

         m_cacheBlock = allocator<Matrix3D<ElementType, allocator>>().allocate(1);

         allocator<Matrix3D<ElementType, allocator>>().construct(m_cacheBlock);

         m_cacheBlock->_Setup(BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1,
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
          (int)m_CoarsenedBlock->getSize()[0] != CoarseBlockSize[0] + e[0] - offset[0] - 1 ||
          (int)m_CoarsenedBlock->getSize()[1] != CoarseBlockSize[1] + e[1] - offset[1] - 1 ||
          (int)m_CoarsenedBlock->getSize()[2] != CoarseBlockSize[2] + e[2] - offset[2] - 1)
      {
         if (m_CoarsenedBlock != NULL) _release(m_CoarsenedBlock);

         m_CoarsenedBlock = allocator<Matrix3D<ElementType, allocator>>().allocate(1);

         allocator<Matrix3D<ElementType, allocator>>().construct(m_CoarsenedBlock);

         m_CoarsenedBlock->_Setup(CoarseBlockSize[0] + e[0] - offset[0] - 1,
                                  CoarseBlockSize[1] + e[1] - offset[1] - 1,
                                  CoarseBlockSize[2] + e[2] - offset[2] - 1);
      }

      #if DIMENSION == 3
         use_averages = (m_refGrid->FiniteDifferences == false || istensorial
                        || m_stencilStart[0]< -2 || m_stencilStart[1] < -2 || m_stencilStart[2] < -2 
                        || m_stencilEnd  [0]>  3 || m_stencilEnd  [1] >  3 || m_stencilEnd  [2] >  3);
      #else
         use_averages = (m_refGrid->FiniteDifferences == false || istensorial
                        || m_stencilStart[0]< -2 || m_stencilStart[1] < -2 
                        || m_stencilEnd  [0]>  3 || m_stencilEnd  [1] >  3);
      #endif

   }

   /** Provide a prepared BlockLab (working copy of gridpoints+halo cells).
    *  Once called, the user can use the () operators to access the halo cells. For derived
    *  instances of BlockLab, the time 't' can also be provided, in order to enforce time-dependent
    *  boundary conditions.
    * @param info: the BlockInfo for the GridBlock that needs halo cells.
    * @param t: (optional) current time, for time-dependent boundary conditions
    * @param applybc: (optional, default is true) apply boundary conditions or not (assume periodic if not)
    */
   virtual void load(const BlockInfo & info, const Real t = 0, const bool applybc = true)
   {
      const int nX                    = BlockType::sizeX;
      const int nY                    = BlockType::sizeY;
      const int nZ                    = BlockType::sizeZ;
      const bool xperiodic            = is_xperiodic();
      const bool yperiodic            = is_yperiodic();
      const bool zperiodic            = is_zperiodic();

      std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();

      const int aux = 1 << info.level;
      NX      = blocksPerDim[0] * aux; // needed for apply_bc
      NY      = blocksPerDim[1] * aux; // needed for apply_bc
      NZ      = blocksPerDim[2] * aux; // needed for apply_bc

      assert(m_cacheBlock != NULL);

      // 1.load the block into the cache
      {
         BlockType &block            = *(BlockType *)info.ptrBlock;
         ElementType *ptrSource = &block(0);

         #if 0 // original
            for(int iz=0; iz<nZ; iz++)
            for(int iy=0; iy<nY; iy++)
            {
              ElementType * ptrDestination = &m_cacheBlock->Access(0-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
              memcpy2((char *)ptrDestination, (char *)ptrSource, sizeof(ElementType)*nX);
              ptrSource+= nX;
            }
         #else
            const int nbytes = sizeof(ElementType) * nX;
            const int _iz0   = -m_stencilStart[2];
            const int _iz1   = _iz0 + nZ;
            const int _iy0   = -m_stencilStart[1];
            const int _iy1   = _iy0 + nY;
            const int m_vSize0 = m_cacheBlock->getSize(0);
            const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
            const int my_ix = -m_stencilStart[0];
            #pragma GCC ivdep
            for (int iz = _iz0; iz < _iz1; iz++)
            {
               const int my_izx = iz * m_nElemsPerSlice + my_ix;
               #pragma GCC ivdep
               for (int iy = _iy0; iy < _iy1; iy += 4)
               {
                  ElementType * __restrict__ ptrDestination0 = &m_cacheBlock->LinAccess(my_izx + (iy    )*m_vSize0);
                  ElementType * __restrict__ ptrDestination1 = &m_cacheBlock->LinAccess(my_izx + (iy + 1)*m_vSize0);
                  ElementType * __restrict__ ptrDestination2 = &m_cacheBlock->LinAccess(my_izx + (iy + 2)*m_vSize0);
                  ElementType * __restrict__ ptrDestination3 = &m_cacheBlock->LinAccess(my_izx + (iy + 3)*m_vSize0);
                  memcpy2(ptrDestination0, (ptrSource         ), nbytes);
                  memcpy2(ptrDestination1, (ptrSource +     nX), nbytes);
                  memcpy2(ptrDestination2, (ptrSource + 2 * nX), nbytes);
                  memcpy2(ptrDestination3, (ptrSource + 3 * nX), nbytes);
                  ptrSource += 4 * nX;
               }
            }
         #endif
      }

      // 2. put the ghosts into the cache
      {
         coarsened = false;

         const bool xskin = info.index[0] == 0 || info.index[0] == NX - 1;
         const bool yskin = info.index[1] == 0 || info.index[1] == NY - 1;
         const bool zskin = info.index[2] == 0 || info.index[2] == NZ - 1;
         const int xskip  = info.index[0] == 0 ? -1 : 1;
         const int yskip  = info.index[1] == 0 ? -1 : 1;
         const int zskip  = info.index[2] == 0 ? -1 : 1;

         int icodes[DIMENSION == 2 ? 8 : 26];  // Could be uint8_t?
         int k = 0;
         coarsened_nei_codes_size = 0;

         for (int icode = (DIMENSION == 2 ? 9 : 0); icode < (DIMENSION == 2 ? 18 : 27); icode++)
         {
            myblocks[icode] = nullptr;
            if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};

            if (!xperiodic && code[0] == xskip && xskin) continue;
            if (!yperiodic && code[1] == yskip && yskin) continue;
            if (!zperiodic && code[2] == zskip && zskin) continue;

            const auto & TreeNei = m_refGrid->Tree(info.level,info.Znei_(code[0], code[1], code[2]));
            if (TreeNei.Exists())
            {
               icodes[k++] = icode;
            }
            else if (TreeNei.CheckCoarser())
            {
               coarsened_nei_codes[coarsened_nei_codes_size++] = icode;
               CoarseFineExchange(info, code);
            }

            if (!istensorial && !use_averages && abs(code[0]) + abs(code[1]) + abs(code[2]) > 1) continue;

            // s and e correspond to start and end of this lab's cells that are filled by neighbors
            const int s[3] = {code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
                              code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
                              code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};

            const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
                              code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
                              code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};

            if      (TreeNei.Exists()    ) SameLevelExchange   (info, code, s, e);
            else if (TreeNei.CheckFiner()) FineToCoarseExchange(info, code, s, e);
         } // icode = 0,...,26 (3D) or 9,...,17 (2D)
	      if (coarsened_nei_codes_size>0)
            for (int i = 0; i < k; ++i)
            {
               const int icode = icodes[i];
               const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
               const int infoNei_index[3] ={(info.index[0]+code[0]+NX)%NX,
                                            (info.index[1]+code[1]+NY)%NY,
                                            (info.index[2]+code[2]+NZ)%NZ};
	            if (UseCoarseStencil(info, infoNei_index))
	            {
		            FillCoarseVersion(info, code);
		            coarsened = true;
	            }
            }

         if (m_refGrid->get_world_size() == 1)
         {
            post_load(info, t, applybc);
         }
      }
   }

 protected:
   /** Called from 'load', to enforce boundary conditions and coarse-fine interpolation.
    *  To interpolate halo cells from neighboring coarser blocks, the BlockLab first fills a 
    *  coarsened version of the GridBlock that requires the halo cells. This coarsened version is
    *  filled with grid points from the coarse neighbors and with averaged down values of this
    *  GridBlock's gridpoints. Averaging down happens in this function, followed by the 
    *  interpolation. Boundary conditions from derived versions of this class are also enforced.
    *  Default boundary conditions are periodic.
    * @param info: the BlockInfo for the GridBlock that needs halo cells.
    * @param t: (optional) current time, for time-dependent boundary conditions
    * @param applybc: (optional, default is true) apply boundary conditions or not (assume periodic if not)
    */
   void post_load(const BlockInfo &info, const Real t = 0, bool applybc = true)
   {
      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      #if DIMENSION == 3
         const int nZ = BlockType::sizeZ;
         if (coarsened)
         {
            #pragma GCC ivdep
            for (int k = 0; k < nZ / 2; k++)
            {
               #pragma GCC ivdep
               for (int j = 0; j < nY / 2; j++)
               {
                  #pragma GCC ivdep
                  for (int i = 0; i < nX / 2; i++)
                  {
                     if (i > -m_InterpStencilStart[0] && i < nX / 2 - m_InterpStencilEnd[0] &&
                         j > -m_InterpStencilStart[1] && j < nY / 2 - m_InterpStencilEnd[1] &&
                         k > -m_InterpStencilStart[2] && k < nZ / 2 - m_InterpStencilEnd[2]) continue;
                     const int ix = 2 * i - m_stencilStart[0];
                     const int iy = 2 * j - m_stencilStart[1];
                     const int iz = 2 * k - m_stencilStart[2];
                     ElementType &coarseElement = m_CoarsenedBlock->Access(i - offset[0], j - offset[1], k - offset[2]);
                     coarseElement = AverageDown( m_cacheBlock->Read(ix  ,iy  ,iz  ), 
                                                  m_cacheBlock->Read(ix+1,iy  ,iz  ),
                                                  m_cacheBlock->Read(ix  ,iy+1,iz  ),
                                                  m_cacheBlock->Read(ix+1,iy+1,iz  ),
                                                  m_cacheBlock->Read(ix  ,iy  ,iz+1),
                                                  m_cacheBlock->Read(ix+1,iy  ,iz+1),
                                                  m_cacheBlock->Read(ix  ,iy+1,iz+1),
                                                  m_cacheBlock->Read(ix+1,iy+1,iz+1));
                  }
               }
            }
         }
      #else
         if (coarsened)
         {
            #pragma GCC ivdep
            for (int j = 0; j < nY / 2; j++)
            {
               #pragma GCC ivdep
               for (int i = 0; i < nX / 2; i++)
               {
                  if (i > -m_InterpStencilStart[0] && i < nX / 2 - m_InterpStencilEnd[0] &&
                      j > -m_InterpStencilStart[1] && j < nY / 2 - m_InterpStencilEnd[1]) continue;
                  const int ix = 2 * i - m_stencilStart[0];
                  const int iy = 2 * j - m_stencilStart[1];
                  ElementType &coarseElement = m_CoarsenedBlock->Access(i - offset[0], j - offset[1],0);
                  coarseElement = AverageDown( m_cacheBlock->Read(ix  ,iy  ,0),
                                               m_cacheBlock->Read(ix+1,iy  ,0),
                                               m_cacheBlock->Read(ix  ,iy+1,0),
                                               m_cacheBlock->Read(ix+1,iy+1,0));              
               }
            }
         }
      #endif
      if (applybc) _apply_bc(info, t, true); // apply BC to coarse block
      CoarseFineInterpolation(info);
      if (applybc) _apply_bc(info, t); 
   }

   /** Check if blocks on the same refinement level need to exchange averaged down cells.
    *  To perform coarse-fine interpolation, the BlockLab creates a coarsened version of the 
    *  GridBlock that needs halo cells. Filling this coarsened version can require averaged down
    *  values from GridBlocks of the same resolution, which would create a large enough stencil
    *  of coarse values to perform the interpolation. Whether or not this is needed is determined
    *  by this function.
    * @param info: the BlockInfo for the GridBlock that needs halo cells.
    * @param b_index: the (i,j,k) index coordinates of the block that is adjacent to 'info'.
    */  
   bool UseCoarseStencil(const BlockInfo &a, const int *b_index)
   {
      if (a.level == 0|| (!use_averages)) return false;

      std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();

      int imin[3];
      int imax[3];
      const int aux = 1 << a.level;
      const bool periodic [3] = {is_xperiodic(), is_yperiodic(), is_zperiodic()};
      const int  blocks   [3] = {blocksPerDim[0] * aux - 1, blocksPerDim[1] * aux - 1, blocksPerDim[2] * aux - 1};
      for (int d = 0; d < 3; d++)
      {
        imin[d] = (a.index[d] < b_index[d]) ? 0 : -1;
        imax[d] = (a.index[d] > b_index[d]) ? 0 : +1;
        if (periodic[d])
        {
          if (a.index[d] == 0 && b_index[d] == blocks[d]) imin[d] = -1;
          if (b_index[d] == 0 && a.index[d] == blocks[d]) imax[d] = +1;
        }
	else
	{
          if (a.index[d] == 0         && b_index[d] == 0        ) imin[d] =  0;
          if (a.index[d] == blocks[d] && b_index[d] == blocks[d]) imax[d] =  0;
	}
      }

      for (int itest = 0; itest < coarsened_nei_codes_size; itest ++)
      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
      for (int i1 = imin[1]; i1 <= imax[1]; i1++)
      for (int i0 = imin[0]; i0 <= imax[0]; i0++)
      {
       	const int icode_test = (i0+1)+3*(i1+1)+9*(i2+1);
         if (coarsened_nei_codes[itest] == icode_test) return true; 
      }
      return false;
   }

   /** Exchange halo cells for blocks on the same refinement level.
    * @param info: the BlockInfo for the GridBlock that needs halo cells.
    * @param code: pointer to three integers, one for each spatial direction. Possible values of each integer are -1,0,+1, based on the relative position of the neighboring block and 'info'
    * @param s: the starts of the part of 'info' that will be filled
    * @param e: the ends of the part of 'info' that will be filled
    */
   void SameLevelExchange(const BlockInfo &info, const int *const code, const int *const s, const int *const e)
   {
      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes) return;

      const int icode = (code[0]+1)+3*(code[1]+1)+9*(code[2]+1);
      myblocks[icode] = m_refGrid->avail(info.level, info.Znei_(code[0], code[1], code[2]));
      if (myblocks[icode] == nullptr) return;
      const BlockType &b = *myblocks[icode];

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
      const int m_vSize0         = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
      const int my_ix            = s[0] - m_stencilStart[0];
      const int mod = (e[1] - s[1]) % 4;

      #pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz++)
      {
         const int my_izx = (iz - m_stencilStart[2]) * m_nElemsPerSlice + my_ix;
         #pragma GCC ivdep
         for (int iy = s[1]; iy < e[1]-mod; iy += 4)
         {
            ElementType * __restrict__ ptrDest0 = &m_cacheBlock->LinAccess(my_izx + (iy     - m_stencilStart[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest1 = &m_cacheBlock->LinAccess(my_izx + (iy + 1 - m_stencilStart[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest2 = &m_cacheBlock->LinAccess(my_izx + (iy + 2 - m_stencilStart[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest3 = &m_cacheBlock->LinAccess(my_izx + (iy + 3 - m_stencilStart[1]) * m_vSize0);
            const ElementType * ptrSrc0 = &b(s[0] - code[0] * nX, iy     - code[1] * nY, iz - code[2] * nZ);
            const ElementType * ptrSrc1 = &b(s[0] - code[0] * nX, iy + 1 - code[1] * nY, iz - code[2] * nZ);
            const ElementType * ptrSrc2 = &b(s[0] - code[0] * nX, iy + 2 - code[1] * nY, iz - code[2] * nZ);
            const ElementType * ptrSrc3 = &b(s[0] - code[0] * nX, iy + 3 - code[1] * nY, iz - code[2] * nZ);
            memcpy2(ptrDest0, ptrSrc0, bytes);
            memcpy2(ptrDest1, ptrSrc1, bytes);
            memcpy2(ptrDest2, ptrSrc2, bytes);
            memcpy2(ptrDest3, ptrSrc3, bytes);
         }
         #pragma GCC ivdep
         for (int iy = e[1]-mod; iy < e[1]; iy++)
         {
            ElementType * __restrict__ ptrDest = &m_cacheBlock->LinAccess(my_izx + (iy - m_stencilStart[1]) * m_vSize0);
            const ElementType * ptrSrc = &b(s[0] - code[0] * nX, iy - code[1] * nY, iz - code[2] * nZ);
            memcpy2(ptrDest, ptrSrc, bytes);
         }
      }
   }

   #if DIMENSION == 3
   /// Average down eight elements (3D)
   ElementType AverageDown(const ElementType &e0, const ElementType &e1, 
                           const ElementType &e2, const ElementType &e3,
                           const ElementType &e4, const ElementType &e5,
                           const ElementType &e6, const ElementType &e7)
   {
      #ifdef PRESERVE_SYMMETRY
      return ConsistentAverage<ElementType>(e0,e1,e2,e3,e4,e5,e6,e7);
      #else
      return 0.125 * (e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7);
      #endif
   }

   /** Coarse-fine interpolation function, based on interpolation stencil of +-1 point.
    *  This function evaluates a third-order Taylor expansion by using a stencil of +-1 points 
    *  around the coarse grid point that will be replaced by eight finer ones. This function can
    *  be overwritten by derived versions of BlockLab, to enable a custom interpolation. The +-1
    *  points used here come from the 'interpolation stencil' passed to BlockLab.
    *  @param C: pointer to the +-1 points around the coarse point (27 values in total)
    *  @param R: pointer to the eight refined points around the coarse point 
    *  @param x: deprecated parameter, used only in the 2D version of this function 
    *  @param y: deprecated parameter, used only in the 2D version of this function 
    *  @param z: deprecated parameter, used only in the 2D version of this function 
    */
   virtual void TestInterp(ElementType *C[3][3][3], ElementType *R, int x, int y, int z)
   {
      #ifdef PRESERVE_SYMMETRY
      const ElementType dudx   = 0.125*( (*C[2][1][1]) - (*C[0][1][1]) );
      const ElementType dudy   = 0.125*( (*C[1][2][1]) - (*C[1][0][1]) );
      const ElementType dudz   = 0.125*( (*C[1][1][2]) - (*C[1][1][0]) );
      const ElementType dudxdy = 0.015625*(((*C[0][0][1]) + (*C[2][2][1])) - ((*C[2][0][1]) + (*C[0][2][1])));
      const ElementType dudxdz = 0.015625*(((*C[0][1][0]) + (*C[2][1][2])) - ((*C[2][1][0]) + (*C[0][1][2])));
      const ElementType dudydz = 0.015625*(((*C[1][0][0]) + (*C[1][2][2])) - ((*C[1][2][0]) + (*C[1][0][2])));
      const ElementType lap    = *C[1][1][1] + 0.03125* ( ConsistentSum((*C[0][1][1]) + (*C[2][1][1]),(*C[1][0][1]) + (*C[1][2][1]),(*C[1][1][0]) + (*C[1][1][2])) -6.0*(*C[1][1][1]));
      R[0] = lap + ( ConsistentSum((-1.0)*dudx,(-1.0)*dudy,(-1.0)*dudz) + ConsistentSum(       dudxdy,       dudxdz,       dudydz) );
      R[1] = lap + ( ConsistentSum(       dudx,(-1.0)*dudy,(-1.0)*dudz) + ConsistentSum((-1.0)*dudxdy,(-1.0)*dudxdz,       dudydz) );
      R[2] = lap + ( ConsistentSum((-1.0)*dudx,       dudy,(-1.0)*dudz) + ConsistentSum((-1.0)*dudxdy,       dudxdz,(-1.0)*dudydz) );
      R[3] = lap + ( ConsistentSum(       dudx,       dudy,(-1.0)*dudz) + ConsistentSum(       dudxdy,(-1.0)*dudxdz,(-1.0)*dudydz) );
      R[4] = lap + ( ConsistentSum((-1.0)*dudx,(-1.0)*dudy,       dudz) + ConsistentSum(       dudxdy,(-1.0)*dudxdz,(-1.0)*dudydz) );
      R[5] = lap + ( ConsistentSum(       dudx,(-1.0)*dudy,       dudz) + ConsistentSum((-1.0)*dudxdy,       dudxdz,(-1.0)*dudydz) );
      R[6] = lap + ( ConsistentSum((-1.0)*dudx,       dudy,       dudz) + ConsistentSum((-1.0)*dudxdy,(-1.0)*dudxdz,       dudydz) );
      R[7] = lap + ( ConsistentSum(       dudx,       dudy,       dudz) + ConsistentSum(       dudxdy,       dudxdz,       dudydz) );
      #else
      const ElementType dudx   = 0.125*( (*C[2][1][1]) - (*C[0][1][1]) );
      const ElementType dudy   = 0.125*( (*C[1][2][1]) - (*C[1][0][1]) );
      const ElementType dudz   = 0.125*( (*C[1][1][2]) - (*C[1][1][0]) );
      const ElementType dudxdy = 0.015625*((*C[0][0][1]) + (*C[2][2][1]) - (*C[2][0][1]) - (*C[0][2][1]));
      const ElementType dudxdz = 0.015625*((*C[0][1][0]) + (*C[2][1][2]) - (*C[2][1][0]) - (*C[0][1][2]));
      const ElementType dudydz = 0.015625*((*C[1][0][0]) + (*C[1][2][2]) - (*C[1][2][0]) - (*C[1][0][2]));
      const ElementType lap    = *C[1][1][1] + 0.03125*((*C[0][1][1]) + (*C[2][1][1]) + (*C[1][0][1]) + (*C[1][2][1]) + (*C[1][1][0]) + (*C[1][1][2]) + (-6.0)*(*C[1][1][1]));
      R[0] = lap - dudx - dudy - dudz + dudxdy + dudxdz + dudydz;
      R[1] = lap + dudx - dudy - dudz - dudxdy - dudxdz + dudydz;
      R[2] = lap - dudx + dudy - dudz - dudxdy + dudxdz - dudydz;
      R[3] = lap + dudx + dudy - dudz + dudxdy - dudxdz - dudydz;
      R[4] = lap - dudx - dudy + dudz + dudxdy - dudxdz - dudydz;
      R[5] = lap + dudx - dudy + dudz - dudxdy + dudxdz - dudydz;
      R[6] = lap - dudx + dudy + dudz - dudxdy - dudxdz + dudydz;
      R[7] = lap + dudx + dudy + dudz + dudxdy + dudxdz + dudydz;
      #endif
   }
   #else
   /// Average down four elements (2D)
   ElementType AverageDown(const ElementType &e0, const ElementType &e1,
                           const ElementType &e2, const ElementType &e3)
   {
      return 0.25 * ((e0 + e3) + (e1 + e2));
   }  

   /// Auxiliary function for 3rd order coarse-fine interpolation
   void LI(ElementType & a, ElementType b, ElementType c)
   {
      auto kappa = ((4.0/15.0)*a+(6.0/15.0)*c)+(-10.0/15.0)*b;
      auto lambda = (b - c) - kappa;
      a = (4.0*kappa+2.0*lambda)+c;
   }

   /// Auxiliary function for 3rd order coarse-fine interpolation
   void LE(ElementType & a, ElementType b, ElementType c)
   {
      auto kappa = ((4.0/15.0)*a+(6.0/15.0)*c)+(-10.0/15.0)*b;
      auto lambda = (b - c) - kappa;
      a = (9.0*kappa+3.0*lambda)+c;
   }

   /** Coarse-fine interpolation function, based on interpolation stencil of +-1 point.
    *  This function evaluates a third-order Taylor expansion by using a stencil of +-1 points 
    *  around the coarse grid point that will be replaced by eight finer ones. This function can
    *  be overwritten by derived versions of BlockLab, to enable a custom interpolation. The +-1
    *  points used here come from the 'interpolation stencil' passed to BlockLab.
    *  @param C: pointer to the +-1 points around the coarse point (9 values in total)
    *  @param R: pointer to the one refined points around the coarse point 
    *  @param x: delta x of the point to be interpolated (+1 or -1).
    *  @param y: delta y of the point to be interpolated (+1 or -1).
    */
   virtual void TestInterp(ElementType *C[3][3], ElementType &R, int x, int y)
   {
      const double dx = 0.25*(2*x-1);
      const double dy = 0.25*(2*y-1);
      ElementType dudx   = 0.5*( (*C[2][1]) - (*C[0][1]) );
      ElementType dudy   = 0.5*( (*C[1][2]) - (*C[1][0]) );
      ElementType dudxdy = 0.25*(((*C[0][0]) + (*C[2][2])) - ((*C[2][0]) + (*C[0][2])));
      ElementType dudx2  = ((*C[0][1]) + (*C[2][1])) -2.0*(*C[1][1]);
      ElementType dudy2  = ((*C[1][0]) + (*C[1][2])) -2.0*(*C[1][1]);
      R = (*C[1][1] + (dx*dudx + dy*dudy)) + ( ((0.5*dx*dx)*dudx2+(0.5*dy*dy)*dudy2) +(dx*dy)*dudxdy );
   }
   #endif

   /** Exchange halo cells from fine to coarse blocks.
    * @param info: the BlockInfo for the GridBlock that needs halo cells.
    * @param code: pointer to three integers, one for each spatial direction. Possible values of each integer are -1,0,+1, based on the relative position of the neighboring block and 'info'
    * @param s: the starts of the part of 'info' that will be filled
    * @param e: the ends of the part of 'info' that will be filled
    */
   void FineToCoarseExchange(const BlockInfo &info, const int *const code, const int *const s, const int *const e)
   {
      const int bytes = (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)) * sizeof(ElementType);
      if (!bytes) return;

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
      const int m_vSize0         = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
      const int yStep            = (code[1] == 0) ? 2 : 1;
      const int zStep            = (code[2] == 0) ? 2 : 1;
      const int mod              = ((e[1] - s[1]) / yStep) % 4;

      int Bstep = 1;                                                    // face
      if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
      else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3)) Bstep = 4; // corner

      /*
        A corner has one finer block.
        An edge has two finer blocks, corresponding to B=0 and B=3. The block B=0 is the one closer
        to the origin (0,0,0). A face has four finer blocks. They are numbered as follows, depending
        on whether the face lies on the xy- , yz- or xz- plane

        y                                  z                                  z
        ^                                  ^                                  ^
        |                                  |                                  |
        |                                  |                                  |
        |_________________                 |_________________                 |_________________
        |        |        |                |        |        |                |        |        |
        |    2   |   3    |                |    2   |   3    |                |    2   |   3    |
        |________|________|                |________|________|                |________|________|
        |        |        |                |        |        |                |        |        |
        |    0   |    1   |                |    0   |    1   |                |    0   |    1   |
        |________|________|------------->x |________|________|------------->x |________|________|------------->y

      */
      // loop over blocks that make up face/edge/corner (respectively 4,2 or 1 blocks)
      for (int B = 0; B <= 3; B += Bstep)
      {
         const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

         #if DIMENSION == 3
            BlockType *b_ptr = m_refGrid->avail1(2 * info.index[0] + std::max(code[0], 0) + code[0] + (B % 2) * std::max(0, 1 - abs(code[0])),
                                                 2 * info.index[1] + std::max(code[1], 0) + code[1] + aux     * std::max(0, 1 - abs(code[1])),
                                                 2 * info.index[2] + std::max(code[2], 0) + code[2] + (B / 2) * std::max(0, 1 - abs(code[2])),
                                                 info.level + 1);
         #else
            BlockType *b_ptr = m_refGrid->avail1(2 * info.index[0] + std::max(code[0], 0) + code[0] + (B % 2) * std::max(0, 1 - abs(code[0])),
                                                 2 * info.index[1] + std::max(code[1], 0) + code[1] + aux     * std::max(0, 1 - abs(code[1])),
                                                 info.level + 1);
         #endif
         if (b_ptr == nullptr) continue;
         BlockType &b = *b_ptr;

         const int my_ix = abs(code[0]) * (s[0] - m_stencilStart[0]) + (1 - abs(code[0])) * (s[0] - m_stencilStart[0] + (B % 2) * (e[0] - s[0]) / 2);
         const int XX = s[0] - code[0] * nX + std::min(0, code[0]) * (e[0] - s[0]);

         #pragma GCC ivdep
         for (int iz = s[2]; iz < e[2]; iz += zStep)
         {
            const int ZZ = (abs(code[2]) == 1) ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ : iz;
            const int my_izx = (abs(code[2]) * (iz - m_stencilStart[2]) + (1 - abs(code[2])) * (iz / 2 - m_stencilStart[2] + (B / 2) * (e[2] - s[2]) / 2)) * m_nElemsPerSlice + my_ix;

            #pragma GCC ivdep
            for (int iy = s[1]; iy < e[1]-mod; iy += 4 * yStep)
            {
               ElementType * __restrict__ ptrDest0 = &m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy + 0 * yStep - m_stencilStart[1]) + (1 - abs(code[1])) * ((iy + 0 * yStep) / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               ElementType * __restrict__ ptrDest1 = &m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy + 1 * yStep - m_stencilStart[1]) + (1 - abs(code[1])) * ((iy + 1 * yStep) / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               ElementType * __restrict__ ptrDest2 = &m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy + 2 * yStep - m_stencilStart[1]) + (1 - abs(code[1])) * ((iy + 2 * yStep) / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               ElementType * __restrict__ ptrDest3 = &m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy + 3 * yStep - m_stencilStart[1]) + (1 - abs(code[1])) * ((iy + 3 * yStep) / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               const int YY0 = (abs(code[1]) == 1) ? 2 * (iy + 0 * yStep - code[1] * nY) + std::min(0, code[1]) * nY : iy + 0 * yStep;
               const int YY1 = (abs(code[1]) == 1) ? 2 * (iy + 1 * yStep - code[1] * nY) + std::min(0, code[1]) * nY : iy + 1 * yStep;
               const int YY2 = (abs(code[1]) == 1) ? 2 * (iy + 2 * yStep - code[1] * nY) + std::min(0, code[1]) * nY : iy + 2 * yStep;
               const int YY3 = (abs(code[1]) == 1) ? 2 * (iy + 3 * yStep - code[1] * nY) + std::min(0, code[1]) * nY : iy + 3 * yStep;
               #if DIMENSION == 3
                  const ElementType * ptrSrc_00 = &b(XX  ,YY0  ,ZZ  );
                  const ElementType * ptrSrc_10 = &b(XX  ,YY0  ,ZZ+1);
                  const ElementType * ptrSrc_20 = &b(XX  ,YY0+1,ZZ  );
                  const ElementType * ptrSrc_30 = &b(XX  ,YY0+1,ZZ+1);
                  const ElementType * ptrSrc_01 = &b(XX  ,YY1  ,ZZ  );
                  const ElementType * ptrSrc_11 = &b(XX  ,YY1  ,ZZ+1);
                  const ElementType * ptrSrc_21 = &b(XX  ,YY1+1,ZZ  );
                  const ElementType * ptrSrc_31 = &b(XX  ,YY1+1,ZZ+1);
                  const ElementType * ptrSrc_02 = &b(XX  ,YY2  ,ZZ  );
                  const ElementType * ptrSrc_12 = &b(XX  ,YY2  ,ZZ+1);
                  const ElementType * ptrSrc_22 = &b(XX  ,YY2+1,ZZ  );
                  const ElementType * ptrSrc_32 = &b(XX  ,YY2+1,ZZ+1);
                  const ElementType * ptrSrc_03 = &b(XX  ,YY3  ,ZZ  );
                  const ElementType * ptrSrc_13 = &b(XX  ,YY3  ,ZZ+1);
                  const ElementType * ptrSrc_23 = &b(XX  ,YY3+1,ZZ  );
                  const ElementType * ptrSrc_33 = &b(XX  ,YY3+1,ZZ+1);
                  #pragma GCC ivdep
                  for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)); ee++)
                  {
                     ptrDest0[ee] = AverageDown(ptrSrc_00[2*ee],ptrSrc_10[2*ee],ptrSrc_20[2*ee],ptrSrc_30[2*ee],ptrSrc_00[2*ee+1],ptrSrc_10[2*ee+1],ptrSrc_20[2*ee+1],ptrSrc_30[2*ee+1]);
                     ptrDest1[ee] = AverageDown(ptrSrc_01[2*ee],ptrSrc_11[2*ee],ptrSrc_21[2*ee],ptrSrc_31[2*ee],ptrSrc_01[2*ee+1],ptrSrc_11[2*ee+1],ptrSrc_21[2*ee+1],ptrSrc_31[2*ee+1]);
                     ptrDest2[ee] = AverageDown(ptrSrc_02[2*ee],ptrSrc_12[2*ee],ptrSrc_22[2*ee],ptrSrc_32[2*ee],ptrSrc_02[2*ee+1],ptrSrc_12[2*ee+1],ptrSrc_22[2*ee+1],ptrSrc_32[2*ee+1]);
                     ptrDest3[ee] = AverageDown(ptrSrc_03[2*ee],ptrSrc_13[2*ee],ptrSrc_23[2*ee],ptrSrc_33[2*ee],ptrSrc_03[2*ee+1],ptrSrc_13[2*ee+1],ptrSrc_23[2*ee+1],ptrSrc_33[2*ee+1]);
                  }
               #else
                 const ElementType *ptrSrc_00 = &b(XX,YY0  ,ZZ);
                 const ElementType *ptrSrc_10 = &b(XX,YY0+1,ZZ);
                 const ElementType *ptrSrc_01 = &b(XX,YY1  ,ZZ);
                 const ElementType *ptrSrc_11 = &b(XX,YY1+1,ZZ);
                 const ElementType *ptrSrc_02 = &b(XX,YY2  ,ZZ);
                 const ElementType *ptrSrc_12 = &b(XX,YY2+1,ZZ);
                 const ElementType *ptrSrc_03 = &b(XX,YY3  ,ZZ);
                 const ElementType *ptrSrc_13 = &b(XX,YY3+1,ZZ);    
                 #pragma GCC ivdep
                 for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)); ee++)
                 {
                    ptrDest0[ee] = AverageDown(*(ptrSrc_00 + 2 * ee    ), *(ptrSrc_10 + 2 * ee    ),
                                               *(ptrSrc_00 + 2 * ee + 1), *(ptrSrc_10 + 2 * ee + 1));
                    ptrDest1[ee] = AverageDown(*(ptrSrc_01 + 2 * ee    ), *(ptrSrc_11 + 2 * ee    ),
                                               *(ptrSrc_01 + 2 * ee + 1), *(ptrSrc_11 + 2 * ee + 1));
                    ptrDest2[ee] = AverageDown(*(ptrSrc_02 + 2 * ee), *(ptrSrc_12 + 2 * ee),
                                               *(ptrSrc_02 + 2 * ee + 1), *(ptrSrc_12 + 2 * ee + 1));
                    ptrDest3[ee] = AverageDown(*(ptrSrc_03 + 2 * ee), *(ptrSrc_13 + 2 * ee),
                                               *(ptrSrc_03 + 2 * ee + 1), *(ptrSrc_13 + 2 * ee + 1));
                  }
               #endif
            }
            #pragma GCC ivdep
            for (int iy = e[1]-mod; iy < e[1]; iy += yStep)
            {
               ElementType *ptrDest = (ElementType *)&m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy - m_stencilStart[1]) + (1 - abs(code[1])) *(iy / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) + std::min(0, code[1]) * nY : iy;
               #if DIMENSION == 3
                  const ElementType * ptrSrc_0 = &b(XX, YY, ZZ);
                  const ElementType * ptrSrc_1 = &b(XX, YY, ZZ + 1);
                  const ElementType * ptrSrc_2 = &b(XX, YY + 1, ZZ);
                  const ElementType * ptrSrc_3 = &b(XX, YY + 1, ZZ + 1);
                  const ElementType * ptrSrc_0_1 = &b(XX+1, YY, ZZ);
                  const ElementType * ptrSrc_1_1 = &b(XX+1, YY, ZZ + 1);
                  const ElementType * ptrSrc_2_1 = &b(XX+1, YY + 1, ZZ);
                  const ElementType * ptrSrc_3_1 = &b(XX+1, YY + 1, ZZ + 1);
                  // average down elements of block b to send to coarser neighbor
                  #pragma GCC ivdep
                  for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)); ee++)
                  {
                     ptrDest[ee] = AverageDown(ptrSrc_0[2*ee],ptrSrc_1[2*ee],ptrSrc_2[2*ee],ptrSrc_3[2*ee],ptrSrc_0_1[2*ee],ptrSrc_1_1[2*ee],ptrSrc_2_1[2*ee],ptrSrc_3_1[2*ee]);
                  }
               #else
                  const ElementType * ptrSrc_0 = &b(XX, YY    , ZZ);
                  const ElementType * ptrSrc_1 = &b(XX, YY + 1, ZZ);
                  // average down elements of block b to send to coarser neighbor
                  #pragma GCC ivdep
                  for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)); ee++)
                  {
                     ptrDest[ee] = AverageDown(*(ptrSrc_0 + 2 * ee    ), *(ptrSrc_1 + 2 * ee    ),
                                               *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1));
                  }
               #endif
            }
         }
      } // B
   }

   /** Exchange halo cells from coarse to fine blocks.
    * @param info: the BlockInfo for the GridBlock that needs halo cells.
    * @param code: pointer to three integers, one for each spatial direction. Possible values of each integer are -1,0,+1, based on the relative position of the neighboring block and 'info'
    */
   void CoarseFineExchange(const BlockInfo &info, const int *const code)
   {
      // Coarse neighbors send their cells. Those are stored in m_CoarsenedBlock and are later used
      // in function CoarseFineInterpolation to interpolate fine values.

      const int infoNei_index[3] ={(info.index[0]+code[0]+NX)%NX,
                                   (info.index[1]+code[1]+NY)%NY,
                                   (info.index[2]+code[2]+NZ)%NZ};
      const int infoNei_index_true[3] ={(info.index[0]+code[0]),
                                        (info.index[1]+code[1]),
                                        (info.index[2]+code[2])};
      #if DIMENSION == 3
         BlockType *b_ptr = m_refGrid->avail1((infoNei_index[0]) / 2,
                                              (infoNei_index[1]) / 2,
                                              (infoNei_index[2]) / 2, info.level - 1);
      #else
         BlockType *b_ptr = m_refGrid->avail1((infoNei_index[0]) / 2,
                                              (infoNei_index[1]) / 2, info.level - 1);
      #endif

      if (b_ptr == nullptr) return;
      const BlockType &b = *b_ptr;

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;

      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
                        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
                        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};

      const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0]) : CoarseBlockSize[0] + (m_stencilEnd[0]) / 2 + m_InterpStencilEnd[0] - 1,
                        code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1]) : CoarseBlockSize[1] + (m_stencilEnd[1]) / 2 + m_InterpStencilEnd[1] - 1,
                        code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2]) : CoarseBlockSize[2] + (m_stencilEnd[2]) / 2 + m_InterpStencilEnd[2] - 1};

      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes) return;

      const int base[3] = {(info.index[0] + code[0]) % 2, (info.index[1] + code[1]) % 2, (info.index[2] + code[2]) % 2};

      int CoarseEdge[3];
      CoarseEdge[0] = (code[0] == 0) ? 0 : (((info.index[0] % 2 == 0) && (infoNei_index_true[0] > info.index[0])) ||
                                            ((info.index[0] % 2 == 1) && (infoNei_index_true[0] < info.index[0]))) ? 1 : 0;
      CoarseEdge[1] = (code[1] == 0) ? 0 : (((info.index[1] % 2 == 0) && (infoNei_index_true[1] > info.index[1])) ||
                                            ((info.index[1] % 2 == 1) && (infoNei_index_true[1] < info.index[1]))) ? 1 : 0;
      CoarseEdge[2] = (code[2] == 0) ? 0 : (((info.index[2] % 2 == 0) && (infoNei_index_true[2] > info.index[2])) ||
                                            ((info.index[2] % 2 == 1) && (infoNei_index_true[2] < info.index[2]))) ? 1 : 0;

      const int start[3] = {std::max(code[0], 0) * nX / 2 + (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX + CoarseEdge[0] * code[0] * nX / 2,
                            std::max(code[1], 0) * nY / 2 + (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY + CoarseEdge[1] * code[1] * nY / 2,
                            std::max(code[2], 0) * nZ / 2 + (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ + CoarseEdge[2] * code[2] * nZ / 2};

      const int m_vSize0         = m_CoarsenedBlock->getSize(0);
      const int m_nElemsPerSlice = m_CoarsenedBlock->getNumberOfElementsPerSlice();
      const int my_ix            = s[0] - offset[0];
      const int mod              = (e[1] - s[1]) % 4;

      #pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz++)
      {
         const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
         #pragma GCC ivdep
         for (int iy = s[1]; iy < e[1]-mod; iy += 4)
         {
            ElementType * __restrict__ ptrDest0 = &m_CoarsenedBlock->LinAccess(my_izx + (iy + 0 - offset[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest1 = &m_CoarsenedBlock->LinAccess(my_izx + (iy + 1 - offset[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest2 = &m_CoarsenedBlock->LinAccess(my_izx + (iy + 2 - offset[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest3 = &m_CoarsenedBlock->LinAccess(my_izx + (iy + 3 - offset[1]) * m_vSize0);
            const ElementType *ptrSrc0 = &b(s[0] + start[0], iy + 0 + start[1], iz + start[2]);
            const ElementType *ptrSrc1 = &b(s[0] + start[0], iy + 1 + start[1], iz + start[2]);
            const ElementType *ptrSrc2 = &b(s[0] + start[0], iy + 2 + start[1], iz + start[2]);
            const ElementType *ptrSrc3 = &b(s[0] + start[0], iy + 3 + start[1], iz + start[2]);
            memcpy2(ptrDest0, ptrSrc0, bytes);
            memcpy2(ptrDest1, ptrSrc1, bytes);
            memcpy2(ptrDest2, ptrSrc2, bytes);
            memcpy2(ptrDest3, ptrSrc3, bytes);
         }
         #pragma GCC ivdep
         for (int iy = e[1]-mod; iy < e[1]; iy++)
         {
            ElementType *ptrDest = &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
            const ElementType *ptrSrc = &b(s[0] + start[0], iy + start[1], iz + start[2]);
            memcpy2(ptrDest, ptrSrc, bytes);
         }
      }
   }

   /** Fill coarsened version of a block, used for fine-coarse interpolation.
    * Each block will create a coarsened version of itself, with averaged down values. This version
    * is also filled with gridpoints for halo cells that are received from coarser neighbors. It is
    * then used to interpolate fine cells at coarse-fine interfaces.
    * @param info: the BlockInfo for the GridBlock that needs halo cells.
    * @param code: pointer to three integers, one for each spatial direction. Possible values of each integer are -1,0,+1, based on the relative position of the neighboring block and 'info'
    */
   void FillCoarseVersion(const BlockInfo &info, const int *const code)
   {
      // If a neighboring block is on the same level it might need to average down some cells and
      // use them to fill the coarsened version of this block. Those cells are needed to refine the
      // coarsened version and obtain ghosts from coarser neighbors (those cells are inside the
      // interpolation stencil for refinement).

      const int icode = (code[0]+1)+3*(code[1]+1)+9*(code[2]+1);
      if (myblocks[icode] == nullptr) return;
      const BlockType &b = *myblocks[icode];

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;

      const int eC[3] = {(m_stencilEnd[0]) / 2 + m_InterpStencilEnd[0],
                         (m_stencilEnd[1]) / 2 + m_InterpStencilEnd[1],
                         (m_stencilEnd[2]) / 2 + m_InterpStencilEnd[2]};

      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
                        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
                        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};

      const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0]) : CoarseBlockSize[0] + eC[0] - 1,
                        code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1]) : CoarseBlockSize[1] + eC[1] - 1,
                        code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2]) : CoarseBlockSize[2] + eC[2] - 1};

      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes) return;

      const int start[3] = {
          s[0] + std::max(code[0], 0) * CoarseBlockSize[0] - code[0] * nX + std::min(0, code[0]) * (e[0] - s[0]),
          s[1] + std::max(code[1], 0) * CoarseBlockSize[1] - code[1] * nY + std::min(0, code[1]) * (e[1] - s[1]),
          s[2] + std::max(code[2], 0) * CoarseBlockSize[2] - code[2] * nZ + std::min(0, code[2]) * (e[2] - s[2])};

      const int m_vSize0         = m_CoarsenedBlock->getSize(0);
      const int m_nElemsPerSlice = m_CoarsenedBlock->getNumberOfElementsPerSlice();
      const int my_ix            = s[0] - offset[0];
      const int XX               = start[0];

      #pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz++)
      {
         const int ZZ     = 2 * (iz - s[2]) + start[2];
         const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;

         #pragma GCC ivdep
         for (int iy = s[1]; iy < e[1]; iy++)
         {
            if (code[1] == 0 && code[2] == 0 && iy > -m_InterpStencilStart[1] &&
                iy < nY / 2 - m_InterpStencilEnd[1] && iz > -m_InterpStencilStart[2] &&
                iz < nZ / 2 - m_InterpStencilEnd[2])
               continue;

            ElementType * __restrict__ ptrDest1 = &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);

            const int YY = 2 * (iy - s[1]) + start[1];
            #if DIMENSION == 3
               const ElementType *ptrSrc_0 = &b(XX, YY, ZZ);
               const ElementType *ptrSrc_1 = &b(XX, YY, ZZ + 1);
               const ElementType *ptrSrc_2 = &b(XX, YY + 1, ZZ);
               const ElementType *ptrSrc_3 = &b(XX, YY + 1, ZZ + 1);
               // average down elements of block b to send to coarser neighbor
               #pragma GCC ivdep
               for (int ee = 0; ee < e[0] - s[0]; ee++)
               {
                  ptrDest1[ee] = AverageDown(*(ptrSrc_0 + 2 * ee), 
                                             *(ptrSrc_1 + 2 * ee),
                                             *(ptrSrc_2 + 2 * ee), 
                                             *(ptrSrc_3 + 2 * ee),
                                             *(ptrSrc_0 + 2 * ee + 1), 
                                             *(ptrSrc_1 + 2 * ee + 1),
                                             *(ptrSrc_2 + 2 * ee + 1), 
                                             *(ptrSrc_3 + 2 * ee + 1));
               }
            #else
               const ElementType *ptrSrc_0 = (const ElementType *)&b(XX, YY, ZZ);
               const ElementType *ptrSrc_1 = (const ElementType *)&b(XX, YY + 1, ZZ);
               // average down elements of block b to send to coarser neighbor
               #pragma GCC ivdep
               for (int ee = 0; ee < e[0] - s[0]; ee++)
               {
                  ptrDest1[ee] = AverageDown(*(ptrSrc_0 + 2 * ee), 
                                             *(ptrSrc_1 + 2 * ee),
                                             *(ptrSrc_0 + 2 * ee + 1), 
                                             *(ptrSrc_1 + 2 * ee + 1));
               }
            #endif
         }
      }
   }

   /// Perform fine-coarse interpolation, after filling coarsened version of block.
   #ifdef PRESERVE_SYMMETRY
   __attribute__((optimize("-O1")))
   #endif
   void CoarseFineInterpolation(const BlockInfo &info)
   {
      const int nX         = BlockType::sizeX;
      const int nY         = BlockType::sizeY;
      const int nZ         = BlockType::sizeZ;
      const bool xperiodic = is_xperiodic();
      const bool yperiodic = is_yperiodic();
      const bool zperiodic = is_zperiodic();
      const std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
      const int aux    = 1 << info.level;
      const bool xskin = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip  = info.index[0] == 0 ? -1 : 1;
      const int yskip  = info.index[1] == 0 ? -1 : 1;
      const int zskip  = info.index[2] == 0 ? -1 : 1;

      
      for (int ii = 0; ii < coarsened_nei_codes_size; ++ii)
      {
         const int icode = coarsened_nei_codes[ii];
         if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
         const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};

         #if DIMENSION == 2
         if (code[2] != 0) continue;
         #endif

         if (!xperiodic && code[0] == xskip && xskin) continue;
         if (!yperiodic && code[1] == yskip && yskin) continue;
         if (!zperiodic && code[2] == zskip && zskin) continue;
         if (!istensorial && !use_averages && abs(code[0]) + abs(code[1]) + abs(code[2]) > 1) continue;

         // s and e correspond to start and end of this lab's cells that are filled by neighbors
         const int s[3] = {code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
                           code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
                           code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
         const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
                           code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
                           code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};

         const int sC[3] = {
             code[0] < 1 ? (code[0] < 0 ? ((m_stencilStart[0] - 1) / 2) : 0) : CoarseBlockSize[0],
             code[1] < 1 ? (code[1] < 0 ? ((m_stencilStart[1] - 1) / 2) : 0) : CoarseBlockSize[1],
             code[2] < 1 ? (code[2] < 0 ? ((m_stencilStart[2] - 1) / 2) : 0) : CoarseBlockSize[2]};

         const int bytes = (e[0] - s[0]) * sizeof(ElementType);
         if (!bytes) continue;

         #if DIMENSION == 3
            ElementType retval[8];
            if (use_averages)
               for (int iz = s[2]; iz < e[2]; iz += 2)
               {
                  const int ZZ = (iz - s[2] - std::min(0, code[2]) * ((e[2] - s[2]) % 2)) / 2 + sC[2];
                  const int z = abs(iz - s[2] - std::min(0, code[2]) * ((e[2] - s[2]) % 2)) % 2;
                  const int izp = (abs(iz) % 2 == 1) ?  -1 : 1;
                  const int rzp = (izp == 1) ? 1:0;
                  const int rz  = (izp == 1) ? 0:1;

                  #pragma GCC ivdep   
                  for (int iy = s[1]; iy < e[1]; iy += 2)
                  {
                     const int YY = (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 + sC[1];
                     const int y = abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
                     const int iyp = (abs(iy) % 2 == 1) ?  -1 : 1;
                     const int ryp = (iyp == 1) ? 1:0;
                     const int ry  = (iyp == 1) ? 0:1;

                     #pragma GCC ivdep      
                     for (int ix = s[0]; ix < e[0]; ix += 2)
                     {
                        const int XX = (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 + sC[0];
                        const int x = abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
                        const int ixp = (abs(ix) % 2 == 1) ?  -1 : 1;
                        const int rxp = (ixp == 1) ? 1:0;
                        const int rx  = (ixp == 1) ? 0:1;
   
                        ElementType *Test[3][3][3];
                        for (int i = 0; i < 3; i++)
                           for (int j = 0; j < 3; j++)
                              for (int k = 0; k < 3; k++)
                                 Test[i][j][k] = &m_CoarsenedBlock->Access(XX - 1 + i - offset[0], YY - 1 + j - offset[1], ZZ - 1 + k - offset[2]);
   
                        TestInterp(Test,retval,x,y,z);
      
                        if (ix       >= s[0] && ix        < e[0] && iy       >= s[1] && iy        < e[1] && iz       >= s[2] && iz       < e[2])
                          m_cacheBlock->Access(ix     - m_stencilStart[0],iy       - m_stencilStart[1],iz       - m_stencilStart[2]) = retval[ rx  +2*ry  +4*rz ];
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy       >= s[1] && iy        < e[1] && iz       >= s[2] && iz       < e[2])
                          m_cacheBlock->Access(ix+ixp - m_stencilStart[0],iy       - m_stencilStart[1],iz       - m_stencilStart[2]) = retval[ rxp +2*ry  +4*rz ];
                        if (ix       >= s[0] && ix        < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1] && iz       >= s[2] && iz       < e[2])
                          m_cacheBlock->Access(ix     - m_stencilStart[0],iy + iyp - m_stencilStart[1],iz       - m_stencilStart[2]) = retval[ rx  +2*ryp +4*rz ];
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1] && iz       >= s[2] && iz       < e[2])
                          m_cacheBlock->Access(ix+ixp - m_stencilStart[0],iy + iyp - m_stencilStart[1],iz       - m_stencilStart[2]) = retval[ rxp +2*ryp +4*rz ];
                        if (ix       >= s[0] && ix        < e[0] && iy       >= s[1] && iy        < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                          m_cacheBlock->Access(ix     - m_stencilStart[0],iy       - m_stencilStart[1],iz + izp - m_stencilStart[2]) = retval[ rx  +2*ry  +4*rzp];
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy       >= s[1] && iy        < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                          m_cacheBlock->Access(ix+ixp - m_stencilStart[0],iy       - m_stencilStart[1],iz + izp - m_stencilStart[2]) = retval[ rxp +2*ry  +4*rzp];
                        if (ix       >= s[0] && ix        < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                          m_cacheBlock->Access(ix     - m_stencilStart[0],iy + iyp - m_stencilStart[1],iz + izp - m_stencilStart[2]) = retval[ rx  +2*ryp +4*rzp];
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                          m_cacheBlock->Access(ix+ixp - m_stencilStart[0],iy + iyp - m_stencilStart[1],iz + izp - m_stencilStart[2]) = retval[ rxp +2*ryp +4*rzp];
                     }
                  }
               }
            if (m_refGrid->FiniteDifferences && abs(code[0]) + abs(code[1]) + abs(code[2]) == 1) //Correct stencil points +-1 and +-2 at faces
            {
               const int coef_ixyz [3] = {std::min(0, code[0]) * ((e[0] - s[0]) % 2),
                                          std::min(0, code[1]) * ((e[1] - s[1]) % 2),
                                          std::min(0, code[2]) * ((e[2] - s[2]) % 2)}; 
               const int min_iz = std::max(s[2],-2);
               const int min_iy = std::max(s[1],-2);
               const int min_ix = std::max(s[0],-2);
               const int max_iz = std::min(e[2],nZ+2);
               const int max_iy = std::min(e[1],nY+2);
               const int max_ix = std::min(e[0],nX+2);

               for (int iz = min_iz; iz < max_iz; iz ++)
               {
                  const int ZZ  =    (iz - s[2] - coef_ixyz[2])/2 + sC[2] - offset[2];
                  const int z   = abs(iz - s[2] - coef_ixyz[2])%2;
                  const double dz = 0.25*(2*z-1);
                  const double * dz_coef = dz > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
                  const bool zinner = (ZZ+offset[2] != 0) && (ZZ+offset[2] != CoarseBlockSize[2] - 1);
                  const bool zstart = (ZZ+offset[2] == 0);

                  #pragma GCC ivdep
                  for (int iy = min_iy; iy < max_iy; iy ++)
                  {
                     const int YY =    (iy - s[1] - coef_ixyz[1])/2 + sC[1] - offset[1];
                     const int y  = abs(iy - s[1] - coef_ixyz[1])%2;
                     const double dy = 0.25*(2*y-1);
                     const double * dy_coef = dy > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
                     const bool yinner = (YY+offset[1] != 0) && (YY+offset[1] != CoarseBlockSize[1] - 1);
                     const bool ystart = (YY+offset[1] == 0);

                     #pragma GCC ivdep
                     for (int ix = min_ix; ix < max_ix; ix ++)
                     {
                        const int XX =    (ix - s[0] - coef_ixyz[0])/2 + sC[0] - offset[0];
                        const int x  = abs(ix - s[0] - coef_ixyz[0])%2;
                        const double dx = 0.25*(2*x-1);
                        const double * dx_coef = dx > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
                        const bool xinner = (XX+offset[0] != 0) && (XX+offset[0] != CoarseBlockSize[0] - 1);
                        const bool xstart = (XX+offset[0] == 0);

                        auto & a = m_cacheBlock->Access(ix - m_stencilStart[0],iy - m_stencilStart[1],iz - m_stencilStart[2]);
                        if (code[0] != 0) //X-face
                        {
                           ElementType x1D,x2D,mixed;

                           int YP,YM,ZP,ZM;
                           double mixed_coef = 1.0;
                           if (yinner)
                           {
                              x1D = (dy_coef[6]*m_CoarsenedBlock->Access(XX,YY-1,ZZ) +dy_coef[8]*m_CoarsenedBlock->Access(XX,YY+1,ZZ))+ dy_coef[7]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              YP = YY+1;
                              YM = YY-1;
                              mixed_coef *= 0.5;
                           }
                           else if (ystart)
                           {
                              x1D = (dy_coef[0]*m_CoarsenedBlock->Access(XX,YY+2,ZZ) + dy_coef[1]*m_CoarsenedBlock->Access(XX,YY+1,ZZ)) + dy_coef[2]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              YP = YY+1;
                              YM = YY;
                           }
                           else
                           {
                              x1D = (dy_coef[3]*m_CoarsenedBlock->Access(XX,YY-2,ZZ) + dy_coef[4]*m_CoarsenedBlock->Access(XX,YY-1,ZZ)) + dy_coef[5]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              YP = YY;
                              YM = YY-1;
                           }
                           if (zinner)
                           {
                              x2D = (dz_coef[6]*m_CoarsenedBlock->Access(XX,YY,ZZ-1) + dz_coef[8]*m_CoarsenedBlock->Access(XX,YY,ZZ+1))+ dz_coef[7]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              ZP = ZZ+1;
                              ZM = ZZ-1;
                              mixed_coef *= 0.5;
                           }
                           else if (zstart)
                           {
                              x2D = (dz_coef[0]*m_CoarsenedBlock->Access(XX,YY,ZZ+2) + dz_coef[1]*m_CoarsenedBlock->Access(XX,YY,ZZ+1)) + dz_coef[2]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              ZP = ZZ+1;
                              ZM = ZZ;
                           }
                           else
                           {
                              x2D = (dz_coef[3]*m_CoarsenedBlock->Access(XX,YY,ZZ-2) + dz_coef[4]*m_CoarsenedBlock->Access(XX,YY,ZZ-1)) + dz_coef[5]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              ZP = ZZ;
                              ZM = ZZ-1;
                           }
                           mixed = mixed_coef*dy*dz*((m_CoarsenedBlock->Access(XX,YM,ZM)+m_CoarsenedBlock->Access(XX,YP,ZP))-(m_CoarsenedBlock->Access(XX,YP,ZM)+m_CoarsenedBlock->Access(XX,YM,ZP)));
                           a = (x1D + x2D) + mixed;
                        }
                        else if (code[1] != 0) //Y-face
                        {
                           ElementType x1D,x2D,mixed;

                           int XP,XM,ZP,ZM;
                           double mixed_coef = 1.0;
                           if (xinner)
                           {
                              x1D = (dx_coef[6]*m_CoarsenedBlock->Access(XX-1,YY,ZZ)  + dx_coef[8]*m_CoarsenedBlock->Access(XX+1,YY,ZZ)) + dx_coef[7]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              XP = XX+1;
                              XM = XX-1;
                              mixed_coef *= 0.5;
                           }
                           else if (xstart)
                           {
                              x1D = (dx_coef[0]*m_CoarsenedBlock->Access(XX+2,YY,ZZ) + dx_coef[1]*m_CoarsenedBlock->Access(XX+1,YY,ZZ)) + dx_coef[2]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              XP = XX+1;
                              XM = XX;
                           }
                           else
                           {
                              x1D = (dx_coef[3]*m_CoarsenedBlock->Access(XX-2,YY,ZZ) + dx_coef[4]*m_CoarsenedBlock->Access(XX-1,YY,ZZ)) + dx_coef[5]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              XP = XX;
                              XM = XX-1;
                           }
                           if (zinner)
                           {
                              x2D = (dz_coef[6]*m_CoarsenedBlock->Access(XX,YY,ZZ-1) + dz_coef[8]*m_CoarsenedBlock->Access(XX,YY,ZZ+1))+ dz_coef[7]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              ZP = ZZ+1;
                              ZM = ZZ-1;
                              mixed_coef *= 0.5;
                           }
                           else if (zstart)
                           {
                              x2D = (dz_coef[0]*m_CoarsenedBlock->Access(XX,YY,ZZ+2) + dz_coef[1]*m_CoarsenedBlock->Access(XX,YY,ZZ+1)) + dz_coef[2]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              ZP = ZZ+1;
                              ZM = ZZ;
                           }
                           else
                           {
                              x2D = (dz_coef[3]*m_CoarsenedBlock->Access(XX,YY,ZZ-2) + dz_coef[4]*m_CoarsenedBlock->Access(XX,YY,ZZ-1)) + dz_coef[5]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              ZP = ZZ;
                              ZM = ZZ-1;
                           }
                           mixed = mixed_coef*dx*dz*((m_CoarsenedBlock->Access(XM,YY,ZM)+m_CoarsenedBlock->Access(XP,YY,ZP))-(m_CoarsenedBlock->Access(XP,YY,ZM)+m_CoarsenedBlock->Access(XM,YY,ZP)));
                           a = (x1D + x2D) + mixed;
                        }
                        else if (code[2] != 0) //Z-face
                        {
                           ElementType x1D,x2D,mixed;

                           int XP,XM,YP,YM;
                           double mixed_coef = 1.0;
                           if (xinner)
                           {
                              x1D = (dx_coef[6]*m_CoarsenedBlock->Access(XX-1,YY,ZZ)  + dx_coef[8]*m_CoarsenedBlock->Access(XX+1,YY,ZZ)) + dx_coef[7]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              XP = XX+1;
                              XM = XX-1;
                              mixed_coef *= 0.5;
                           }
                           else if (xstart)
                           {
                              x1D = (dx_coef[0]*m_CoarsenedBlock->Access(XX+2,YY,ZZ) + dx_coef[1]*m_CoarsenedBlock->Access(XX+1,YY,ZZ)) + dx_coef[2]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              XP = XX+1;
                              XM = XX;
                           }
                           else
                           {
                              x1D = (dx_coef[3]*m_CoarsenedBlock->Access(XX-2,YY,ZZ) + dx_coef[4]*m_CoarsenedBlock->Access(XX-1,YY,ZZ)) + dx_coef[5]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              XP = XX;
                              XM = XX-1;
                           }
                           if (yinner)
                           {
                              x2D = (dy_coef[6]*m_CoarsenedBlock->Access(XX,YY-1,ZZ) +dy_coef[8]*m_CoarsenedBlock->Access(XX,YY+1,ZZ))+ dy_coef[7]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              YP = YY+1;
                              YM = YY-1;
                              mixed_coef *= 0.5;
                           }
                           else if (ystart)
                           {
                              x2D = (dy_coef[0]*m_CoarsenedBlock->Access(XX,YY+2,ZZ) + dy_coef[1]*m_CoarsenedBlock->Access(XX,YY+1,ZZ)) + dy_coef[2]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              YP = YY+1;
                              YM = YY;
                           }
                           else
                           {
                              x2D = (dy_coef[3]*m_CoarsenedBlock->Access(XX,YY-2,ZZ) + dy_coef[4]*m_CoarsenedBlock->Access(XX,YY-1,ZZ)) + dy_coef[5]*m_CoarsenedBlock->Access(XX,YY,ZZ);
                              YP = YY;
                              YM = YY-1;
                           }

                           mixed = mixed_coef*dx*dy*((m_CoarsenedBlock->Access(XM,YM,ZZ)+m_CoarsenedBlock->Access(XP,YP,ZZ))-(m_CoarsenedBlock->Access(XP,YM,ZZ)+m_CoarsenedBlock->Access(XM,YP,ZZ)));
                           a = (x1D + x2D) + mixed;
                        }

                        const auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] + (-3*code[0]+1)/2 - x*abs(code[0]),
                                                              iy - m_stencilStart[1] + (-3*code[1]+1)/2 - y*abs(code[1]),
                                                              iz - m_stencilStart[2] + (-3*code[2]+1)/2 - z*abs(code[2]));
                        const auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] + (-5*code[0]+1)/2 - x*abs(code[0]),
                                                              iy - m_stencilStart[1] + (-5*code[1]+1)/2 - y*abs(code[1]),
                                                              iz - m_stencilStart[2] + (-5*code[2]+1)/2 - z*abs(code[2]));
                        const int ccc  = code[0] + code[1] + code[2];
                        const int xyz  = abs(code[0])*x+abs(code[1])*y+abs(code[2])*z;

                        if (ccc == 1)     a = (xyz==0)?(1.0/15.0)*(8.0*a+(10.0*b-3.0*c)):(1.0/15.0)*(24.0*a+(-15.0*b+6*c));
                        else /*(ccc=-1)*/ a = (xyz==1)?(1.0/15.0)*(8.0*a+(10.0*b-3.0*c)):(1.0/15.0)*(24.0*a+(-15.0*b+6*c));
                     }
                  }
               }
            }
         #else

            if (use_averages)
            {
               #pragma GCC ivdep
               for (int iy = s[1]; iy < e[1]; iy += 1)
               {
                  const int YY = (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 + sC[1];
                  #pragma GCC ivdep
                  for (int ix = s[0]; ix < e[0]; ix += 1)
                  {
                     const int XX = (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 + sC[0];
                     ElementType *Test[3][3];
                     for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                              Test[i][j] = &m_CoarsenedBlock->Access(XX - 1 + i - offset[0],
                                                                     YY - 1 + j - offset[1],0);
                     TestInterp(Test,m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],0),
                                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2,
                                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2);
                  }
               }
            }
            if (m_refGrid->FiniteDifferences && abs(code[0]) + abs(code[1]) == 1) //Correct stencil points +-1 and +-2 at faces
            {
               #pragma GCC ivdep
               for (int iy = s[1]; iy < e[1]; iy += 2)
               {
                  const int YY = (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 + sC[1]- offset[1];
                  const int y = abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
                  const int iyp = (abs(iy) % 2 == 1) ?  -1 : 1;
                  const double dy = 0.25*(2*y-1);

                  #pragma GCC ivdep
                  for (int ix = s[0]; ix < e[0]; ix += 2)
                  {
                     const int XX = (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 + sC[0]- offset[0];
                     const int x = abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
                     const int ixp = (abs(ix) % 2 == 1) ?  -1 : 1;
                     const double dx = 0.25*(2*x-1);
                     if (ix < -2 || iy < -2 || ix > nX+1 || iy > nY+1) continue;

                     if (code[0] != 0)
                     {
                        ElementType dudy,dudy2;
                        if (YY+offset[1] == 0)
                        {
                           dudy  = (-0.5*m_CoarsenedBlock->Access(XX,YY+2,0) - 1.5*m_CoarsenedBlock->Access(XX,YY,0))+ 2.0*m_CoarsenedBlock->Access(XX,YY+1,0) ;
                           dudy2 = (m_CoarsenedBlock->Access(XX,YY+2,0)+m_CoarsenedBlock->Access(XX,YY,0))-2.0*m_CoarsenedBlock->Access(XX,YY+1,0);
                        }
                        else if (YY+offset[1] == CoarseBlockSize[1] - 1)
                        {
                           dudy  = (0.5*m_CoarsenedBlock->Access(XX,YY-2,0) + 1.5*m_CoarsenedBlock->Access(XX,YY,0) )- 2.0*m_CoarsenedBlock->Access(XX,YY-1,0) ;
                           dudy2 = (m_CoarsenedBlock->Access(XX,YY-2,0)+m_CoarsenedBlock->Access(XX,YY,0))-2.0*m_CoarsenedBlock->Access(XX,YY-1,0);
                        }
                        else
                        {
                           dudy  = 0.5*(m_CoarsenedBlock->Access(XX,YY+1,0)-m_CoarsenedBlock->Access(XX,YY-1,0));
                           dudy2 = (m_CoarsenedBlock->Access(XX,YY+1,0)+m_CoarsenedBlock->Access(XX,YY-1,0))-2.0*m_CoarsenedBlock->Access(XX,YY,0);
                        }
                        m_cacheBlock->Access(ix - m_stencilStart[0]    , iy - m_stencilStart[1]    ,0) = m_CoarsenedBlock->Access(XX,YY,0) + dy*dudy + (0.5*dy*dy)*dudy2;
                        if (iy + iyp >= s[1] && iy + iyp  < e[1]) m_cacheBlock->Access(ix - m_stencilStart[0]    , iy - m_stencilStart[1]+iyp,0) = m_CoarsenedBlock->Access(XX,YY,0) - dy*dudy + (0.5*dy*dy)*dudy2;
                        if (ix + ixp >= s[0] && ix + ixp  < e[0]) m_cacheBlock->Access(ix - m_stencilStart[0]+ixp, iy - m_stencilStart[1]    ,0) = m_CoarsenedBlock->Access(XX,YY,0) + dy*dudy + (0.5*dy*dy)*dudy2;
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1]) m_cacheBlock->Access(ix - m_stencilStart[0]+ixp, iy - m_stencilStart[1]+iyp,0) = m_CoarsenedBlock->Access(XX,YY,0) - dy*dudy + (0.5*dy*dy)*dudy2;
                     }
                     else //if (code[1] != 0)
                     {
                        ElementType dudx,dudx2;
                        if (XX+offset[0] == 0)
                        {
                           dudx  = (-0.5*m_CoarsenedBlock->Access(XX+2,YY,0)- 1.5*m_CoarsenedBlock->Access(XX,YY,0)) + 2.0*m_CoarsenedBlock->Access(XX+1,YY,0) ;
                           dudx2 = (m_CoarsenedBlock->Access(XX+2,YY,0)+m_CoarsenedBlock->Access(XX,YY,0))-2.0*m_CoarsenedBlock->Access(XX+1,YY,0);
                        }
                        else if (XX+offset[0] == CoarseBlockSize[0] - 1)
                        {
                           dudx  = (0.5*m_CoarsenedBlock->Access(XX-2,YY,0)+ 1.5*m_CoarsenedBlock->Access(XX,YY,0)) - 2.0*m_CoarsenedBlock->Access(XX-1,YY,0) ;
                           dudx2 = (m_CoarsenedBlock->Access(XX-2,YY,0)+m_CoarsenedBlock->Access(XX,YY,0))-2.0*m_CoarsenedBlock->Access(XX-1,YY,0);
                        }
                        else
                        {
                           dudx  = 0.5*(m_CoarsenedBlock->Access(XX+1,YY,0)-m_CoarsenedBlock->Access(XX-1,YY,0));
                           dudx2 = (m_CoarsenedBlock->Access(XX+1,YY,0)+m_CoarsenedBlock->Access(XX-1,YY,0))-2.0*m_CoarsenedBlock->Access(XX,YY,0);
                        }
                        m_cacheBlock->Access(ix - m_stencilStart[0]    , iy - m_stencilStart[1]    ,0) = m_CoarsenedBlock->Access(XX,YY,0) + dx*dudx + (0.5*dx*dx)*dudx2;
                        if (iy + iyp >= s[1] && iy + iyp  < e[1]) m_cacheBlock->Access(ix - m_stencilStart[0]    , iy - m_stencilStart[1]+iyp,0) = m_CoarsenedBlock->Access(XX,YY,0) + dx*dudx + (0.5*dx*dx)*dudx2;
                        if (ix + ixp >= s[0] && ix + ixp  < e[0]) m_cacheBlock->Access(ix - m_stencilStart[0]+ixp, iy - m_stencilStart[1]    ,0) = m_CoarsenedBlock->Access(XX,YY,0) - dx*dudx + (0.5*dx*dx)*dudx2;
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1]) m_cacheBlock->Access(ix - m_stencilStart[0]+ixp, iy - m_stencilStart[1]+iyp,0) = m_CoarsenedBlock->Access(XX,YY,0) - dx*dudx + (0.5*dx*dx)*dudx2;
                     }
                  }

               }

               for (int iy = s[1]; iy < e[1]; iy += 1)
               {
               #pragma GCC ivdep
               for (int ix = s[0]; ix < e[0]; ix += 1)
               {
                  if (ix < -2 || iy < -2 || ix > nX+1 || iy > nY+1) continue;
                  const int x = abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
                  const int y = abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;

                  auto & a = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],0);

                  if (code[0] == 0 && code[1] == 1)
                  {
                     if (y==0) //interpolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] - 1,0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] - 2,0);
                        LI(a,b,c);
                     }
                     else if (y==1) //extrapolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] - 2,0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] - 3,0);
                        LE(a,b,c);
                     }
                  }
                  else if (code[0] == 0 && code[1] == -1)
                  {
                     if (y==1) //interpolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] + 1,0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] + 2,0);
                        LI(a,b,c);
                     }
                     else if (y==0) //extrapolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] + 2,0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] + 3,0);
                        LE(a,b,c);
                     }
                  }
                  else if (code[1] == 0 && code[0] == 1)
                  {
                     if (x==0) //interpolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] - 1, iy - m_stencilStart[1],0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] - 2, iy - m_stencilStart[1],0);
                        LI(a,b,c);
                     }
                     else if (x==1) //extrapolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] - 2, iy - m_stencilStart[1],0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] - 3, iy - m_stencilStart[1],0);
                        LE(a,b,c);
                     }
                  }
                  else if (code[1] == 0 && code[0] == -1)
                  {
                     if (x==1) //interpolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] + 1, iy - m_stencilStart[1],0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] + 2, iy - m_stencilStart[1],0);
                        LI(a,b,c);
                     }
                     else if (x==0) //extrapolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] + 2, iy - m_stencilStart[1],0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] + 3, iy - m_stencilStart[1],0);
                        LE(a,b,c);
                     }
                  }
               }
               }
            }
         #endif
      }
   }

   /// Enforce boundary conditions.
   virtual void _apply_bc(const BlockInfo &info, const Real t = 0, bool coarse = false) {}

   /// Deallocate memory.
   template <typename T>
   void _release(T *&t)
   {
      if (t != NULL)
      {
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
