#pragma once

#include <array>
#include <cassert>

#ifndef DIMENSION
#define DIMENSION 3
#endif

#if DIMENSION == 3
  #include "SpaceFillingCurve.h"
#else
  #include "SpaceFillingCurve2D.h"
#endif

namespace cubism
{

enum State : signed char
{
   Leave    = 0,
   Refine   = 1,
   Compress = -1
};

///Single integer used to recognize if a Block exists in the Grid and by which MPI rank it is owned. 
struct TreePosition
{
   int position{-3};
   bool CheckCoarser() const { return position == -2; }
   bool CheckFiner() const { return position == -1; }
   bool Exists() const { return position >= 0; }
   int rank() const { return position; }
   void setrank(const int r) { position = r; }
   void setCheckCoarser() { position = -2; }
   void setCheckFiner() { position = -1; }
};

/** @brief Meta-data for each GridBlock.
 * 
 * This struct holds information such as the grid spacing and the level of refinement of each 
 * GridBlock. It is also used to access the data of the GridBlock through a relevant pointer.
 * Importantly, all blocks are organized in a single Octree/Quadtree, regardless of the number of 
 * fields/variables used in the simulation (and the number of Grids). For this reason, only one 
 * instance of SpaceFillingCurve is needed, which is owned as a static member of the BlockInfo 
 * struct. The functions of the SpaceFillingCurve are also accessed through static functions of 
 * BlockInfo.
 */
struct BlockInfo
{
   long long blockID;        ///< all n BlockInfos owned by one rank have blockID=0,1,...,n-1
   long long blockID_2;      ///< unique index of each BlockInfo, based on its refinement level and Z-order curve index
   long long Z;              ///< Z-order curve index of this block
   long long Znei[3][3][3];  ///< Z-order curve index of 26 neighboring boxes (Znei[1][1][1] = Z)
   long long halo_block_id;  ///< all m blocks at the boundary of a rank are numbered by halo_block_id=0,1,...,m-1
   long long Zparent;        ///< Z-order curve index of parent block (after comression)
   long long Zchild[2][2][2];///< Z-order curve index of blocks that replace this one during refinement
   double h;                 ///< grid spacing
   double origin[3];         ///<(x,y,z) of block's origin
   int index[3];             ///<(i,j,k) coordinates of block at given refinement level
   int level;                ///< refinement level
   void *ptrBlock{nullptr};  ///< Pointer to data stored in user-defined Block
   void *auxiliary;          ///< Pointer to blockcase
   bool changed2;            ///< =true if block will be refined/compressed; used to update State of neighbouring blocks among ranks
   State state;              ///< Refine/Compress/Leave this block

   /// Static function used to initialize static SFC
   static int levelMax(int l = 0)
   {
      static int lmax = l;
      return lmax;
   }

   #if DIMENSION == 3

      /// Static function used to initialize static SFC
      static int blocks_per_dim(int i, int nx = 0, int ny = 0, int nz = 0)
      {
         static int a[3] = {nx, ny, nz};
         return a[i];
      }
   
      /// Pointer to single instance of SFC used
      static SpaceFillingCurve *SFC()
      {
         static SpaceFillingCurve Zcurve(blocks_per_dim(0), blocks_per_dim(1), blocks_per_dim(2), levelMax());
         return &Zcurve;
      }
   
      /// get Z-order index for coordinates (ix,iy,iz) and refinement level
      static long long forward(int level, int ix, int iy, int iz) { return (*SFC()).forward(level, ix, iy, iz); }
   
      /// get unique blockID_2 index from refinement level, Z-order index and coordinates
      static long long Encode(int level, long long Z, int index[3]) { return (*SFC()).Encode(level, Z, index); }
   
      /// get coordinates from refinement level and Z-order index
      static void inverse(long long Z, int l, int &i, int &j, int &k) { (*SFC()).inverse(Z, l, i, j, k); }

   #else

      /// Static function used to initialize static SFC (same as above but in 2D)
      static int blocks_per_dim(int i, int nx = 0, int ny = 0)
      {
         static int a[2] = {nx, ny};
         return a[i];
      }
   
      /// Pointer to single instance of SFC used (same as above but in 2D)
      static SpaceFillingCurve2D *SFC()
      {
         static SpaceFillingCurve2D Zcurve(blocks_per_dim(0), blocks_per_dim(1), levelMax());
         return &Zcurve;
      }
   
      /// get Z-order index for coordinates (ix,iy,iz) and refinement level
      static long long forward(int level, int ix, int iy) { return (*SFC()).forward(level, ix, iy); }
   
      /// get unique blockID_2 index from refinement level, Z-order index and coordinates
      static long long Encode(int level, long long Z, int index[2]) { return (*SFC()).Encode(level, Z, index); }
   
      /// get coordinates from refinement level and Z-order index
      static void inverse(long long Z, int l, int &i, int &j) { (*SFC()).inverse(Z, l, i, j); }

   #endif

   #if DIMENSION == 3
      /// return position (x,y,z) in 3D, given indices of grid point
      template <typename T>
      inline void pos(T p[3], int ix, int iy, int iz) const
      {
         p[0] = origin[0] + h * (ix + 0.5);
         p[1] = origin[1] + h * (iy + 0.5);
         p[2] = origin[2] + h * (iz + 0.5);
      }

      /// return position (x,y,z) in 3D, given indices of grid point
      template <typename T>
      inline std::array<T, 3> pos(int ix, int iy, int iz) const
      {
         std::array<T, 3> result;
         pos(result.data(), ix, iy, iz);
         return result;
      }
   #else
      /// return position (x,y) in 2D, given indices of grid point
      template <typename T>
      inline void pos(T p[2], int ix, int iy) const
      {
         p[0] = origin[0] + h * (ix + 0.5);
         p[1] = origin[1] + h * (iy + 0.5);
      }

      /// return position (x,y) in 2D, given indices of grid point
      template <typename T>
      inline std::array<T, 2> pos(int ix, int iy) const
      {
         std::array<T, 2> result;
         pos(result.data(), ix, iy);
         return result;
      }
   #endif

   /// used to order/sort blocks based on blockID_2, which is only a function of Z and level
   bool operator<(const BlockInfo &other) const { return (blockID_2 < other.blockID_2); }

   /// constructor will do nothing, 'setup' needs to be called instead
   BlockInfo(){};

   /// Provide level, grid spacing, (x,y,z) origin and Z-index to setup/initialize a blockinfo
   void setup(const int a_level, const double a_h, const double a_origin[3], const long long a_Z)
   {
      level     = a_level;
      Z         = a_Z;
      state     = Leave;
      level     = a_level;
      h         = a_h;
      origin[0] = a_origin[0];
      origin[1] = a_origin[1];
      origin[2] = a_origin[2];
      changed2  = true;
      auxiliary = nullptr;

      const int TwoPower = 1 << level;

      //Now we also set the indices of the neighbouring blocks, parent block and child blocks.
      #if DIMENSION == 3
         inverse(Z, level, index[0], index[1], index[2]);

         const int Bmax[3] = {blocks_per_dim(0) * TwoPower, blocks_per_dim(1) * TwoPower, blocks_per_dim(2) * TwoPower};
         for (int i = -1; i < 2; i++)
         for (int j = -1; j < 2; j++)
         for (int k = -1; k < 2; k++)
            Znei[i + 1][j + 1][k + 1] = forward(level, (index[0] + i + Bmax[0]) % Bmax[0], (index[1] + j + Bmax[1]) % Bmax[1], (index[2] + k + Bmax[2]) % Bmax[2]);

         for (int i =  0; i < 2; i++)
         for (int j =  0; j < 2; j++)
         for (int k =  0; k < 2; k++)
            Zchild[i][j][k] = forward(level + 1, 2 * index[0] + i, 2 * index[1] + j, 2 * index[2] + k);

         Zparent = (level == 0) ? 0 : forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0], (index[1] / 2 + Bmax[1]) % Bmax[1], (index[2] / 2 + Bmax[2]) % Bmax[2]);
      #else
         inverse(Z, level, index[0], index[1]);
         index[2] = 0;

         const int Bmax[3] = {blocks_per_dim(0) * TwoPower, blocks_per_dim(1) * TwoPower, 1};
         for (int i = -1; i < 2; i++)
         for (int j = -1; j < 2; j++)
         for (int k = -1; k < 2; k++)
            Znei[i + 1][j + 1][k + 1] = forward(level, (index[0] + i + Bmax[0]) % Bmax[0], (index[1] + j + Bmax[1]) % Bmax[1]);

         for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         for (int k = 0; k < 2; k++)
            Zchild[i][j][k] = forward(level + 1, 2 * index[0] + i, 2 * index[1] + j);

         Zparent = (level == 0) ? 0 : forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0], (index[1] / 2 + Bmax[1]) % Bmax[1]);
      #endif
      blockID_2 = Encode(level, Z, index);
      blockID   = blockID_2;
   }

   /// used for easier access of Znei[][][]
   long long Znei_(const int i, const int j, const int k) const
   {
      assert(abs(i) <= 1);
      assert(abs(j) <= 1);
      assert(abs(k) <= 1);
      return Znei[1 + i][1 + j][1 + k];
   }
};
} // namespace cubism
