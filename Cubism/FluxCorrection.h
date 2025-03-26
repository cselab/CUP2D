#pragma once

#include <map>
#include <algorithm>
#include "BlockInfo.h"

namespace cubism
{

/**
 * @brief Auxiliary struct used to perform flux corrections at coarse-fine block interfaces.
 * 
 * This struct can save the fluxes passing though the six faces of one GridBlock. Each BlockInfo
 * owns a pointer to its own BlockCase. The pointer is not a nullptr only if any of the six faces
 * of the block have a neighboring block at a different refinement level. When a stencil 
 * computation is performed, each block can fill its own BlockCase with the fluxes passing through
 * its faces. Then, the FluxCorrection class will replace the coarse fluxes with the sum of the
 * fine fluxes, which ensures conservation of the quantity whose flux we compute.
 * 
 * @tparam BlockType The user-defined GridBlock
 * @tparam ElementType The type of elements stored by the user-defined GridBlock
 */
template <typename BlockType, typename ElementType = typename BlockType::ElementType>
struct BlockCase
{
   std::vector<std::vector<ElementType>> m_pData; ///< six vectors, one for each face
   unsigned int m_vSize[3]; ///< sizes of the faces (in x,y and z)
   bool storedFace[6]; ///< boolean variables, one for each face (=true if this face needs flux corrections because it is at a coarse-fine interface)
   int level; ///< refinement level of the associated block
   long long Z; ///< Z-order index of the associated block

   ///Constructor.
   BlockCase(bool _storedFace[6], unsigned int nX, unsigned int nY, unsigned int nZ, int _level, long long _Z)
   {
      m_vSize[0] = nX;
      m_vSize[1] = nY;
      m_vSize[2] = nZ;

      storedFace[0] = _storedFace[0];
      storedFace[1] = _storedFace[1];
      storedFace[2] = _storedFace[2];
      storedFace[3] = _storedFace[3];
      storedFace[4] = _storedFace[4];
      storedFace[5] = _storedFace[5];

      m_pData.resize(6);

      for (int d = 0; d < 3; d++)
      {
         int d1 = (d + 1) % 3;
         int d2 = (d + 2) % 3;

         // assume everything is initialized to 0!!!!!!!!
         if (storedFace[2 * d    ]) m_pData[2 * d    ].resize(m_vSize[d1] * m_vSize[d2]);
         if (storedFace[2 * d + 1]) m_pData[2 * d + 1].resize(m_vSize[d1] * m_vSize[d2]);
      }
      level = _level;
      Z     = _Z;
   }

   ~BlockCase() {}
};

/**
 * @brief Performs flux corrections at coarse-fine block interfaces.
 * 
 * This class can replace the coarse fluxes stored at BlockCases with the sum of the
 * fine fluxes (also stored at BlockCases). This ensures conservation of the quantity 
 * whose flux we compute.
 * 
 * @tparam TGrid The user-defined Grid/GridMPI
 * @tparam BlockType The user-defined GridBlock used by TGrid
 */
template <typename TGrid>
class FluxCorrection
{
 public:
   using GridType = TGrid; ///< should be a 'Grid', 'GridMPI' or derived class 
   typedef typename GridType::BlockType BlockType;
   typedef typename BlockType::ElementType ElementType;
   typedef typename ElementType::RealType Real;
   typedef BlockCase<BlockType> Case;
   int rank{0};///< MPI process ID (set to zero here, for a serial implementation)

 protected:
   std::map<std::array<long long, 2>, Case *> MapOfCases; ///< map between BlockCases and BlockInfos (two integers: refinement level and Z-order index)
   TGrid *grid; ///< grid for which we perform the flux corrections
   std::vector<Case> Cases; ///< BlockCases owned by FluxCorrection; BlockInfos have pointers to these (in needed)

   ///Perform flux correction for BlockInfo 'info' in the direction/face specified by 'code'
   void FillCase(BlockInfo & info, const int *const code)
   {
      const int myFace    = abs( code[0]) * std::max(0,  code[0]) + abs( code[1]) * (std::max(0,  code[1]) + 2) + abs( code[2]) * (std::max(0,  code[2]) + 4);
      const int otherFace = abs(-code[0]) * std::max(0, -code[0]) + abs(-code[1]) * (std::max(0, -code[1]) + 2) + abs(-code[2]) * (std::max(0, -code[2]) + 4);

      std::array<long long, 2> temp = {(long long)info.level, info.Z};
      auto search             = MapOfCases.find(temp);

      Case &CoarseCase = (*search->second);
      std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];

      assert(myFace / 2 == otherFace / 2);
      assert(search != MapOfCases.end());
      assert(CoarseCase.Z == info.Z);
      assert(CoarseCase.level == info.level);

      #if DIMENSION == 3
      for (int B = 0; B <= 3; B++) // loop over fine blocks that make up coarse face
      #else
      for (int B = 0; B <= 1; B++) // loop over fine blocks that make up coarse face
      #endif
      {
        const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
        #if DIMENSION == 3
        const long long Z = (*grid).getZforward(info.level + 1,
                  2 * info.index[0] + std::max(code[0], 0) + code[0] +(B % 2) * std::max(0, 1 - abs(code[0])),
                  2 * info.index[1] + std::max(code[1], 0) + code[1] +    aux * std::max(0, 1 - abs(code[1])),
                  2 * info.index[2] + std::max(code[2], 0) + code[2] +(B / 2) * std::max(0, 1 - abs(code[2])));
        #else
        const long long Z = (*grid).getZforward(info.level + 1,
                  2 * info.index[0] + std::max(code[0], 0) + code[0] +(B % 2) * std::max(0, 1 - abs(code[0])),
                  2 * info.index[1] + std::max(code[1], 0) + code[1] +    aux * std::max(0, 1 - abs(code[1])));
        #endif

        const int other_rank = grid->Tree(info.level + 1, Z).rank();
        if (other_rank != rank) continue;
        auto search1 = MapOfCases.find({info.level + 1, Z});

        Case &FineCase                     = (*search1->second);
        std::vector<ElementType> &FineFace = FineCase.m_pData[otherFace];
        const int d   = myFace / 2;
        const int d1  = std::max((d + 1) % 3, (d + 2) % 3);
        const int d2  = std::min((d + 1) % 3, (d + 2) % 3);
        const int N1F = FineCase.m_vSize[d1];
        const int N2F = FineCase.m_vSize[d2];
        const int N1  = N1F;
        const int N2  = N2F;
        int base = 0;
        if      (B == 1)  base = (N2 / 2) + (0) * N2;
        else if (B == 2)  base = (0) + (N1 / 2) * N2;
        else if (B == 3)  base = (N2 / 2) + (N1 / 2) * N2;
        assert(search1 != MapOfCases.end());
        assert(N1F == (int)CoarseCase.m_vSize[d1]);
        assert(N2F == (int)CoarseCase.m_vSize[d2]);
        assert(FineFace.size() == CoarseFace.size());
        #if DIMENSION == 3
          for (int i1 = 0; i1 < N1; i1 += 2)
          for (int i2 = 0; i2 < N2; i2 += 2)
          {
            CoarseFace[base + (i2 / 2) + (i1 / 2) * N2] += FineFace[i2 +  i1    * N2] + FineFace[i2+1 +  i1    * N2] +
                                                           FineFace[i2 + (i1+1) * N2] + FineFace[i2+1 + (i1+1) * N2];
            FineFace[i2     +  i1      * N2].clear();
            FineFace[i2 + 1 +  i1      * N2].clear();
            FineFace[i2     + (i1 + 1) * N2].clear();
            FineFace[i2 + 1 + (i1 + 1) * N2].clear();
          }
        #else
          for (int i2 = 0; i2 < N2; i2 += 2)
          {
            CoarseFace[base + i2/2] += FineFace[i2] + FineFace[i2 + 1];
            FineFace[i2    ].clear();
            FineFace[i2 + 1].clear();
          }
        #endif
      }
   }

 public:
   ///Prepare the FluxCorrection class for a given 'grid' by allocating BlockCases at each coarse-fine interface
   virtual void prepare(TGrid &_grid)
   {
      if (_grid.UpdateFluxCorrection == false) return;
      _grid.UpdateFluxCorrection = false;

      Cases.clear();
      MapOfCases.clear();
      grid = &_grid;
      std::vector<BlockInfo> & B = (*grid).getBlocksInfo();

      std::array<int, 3> blocksPerDim = (*grid).getMaxBlocks();
      std::array<int,6> icode = {1*2 + 3*1 + 9*1, 1*0 + 3*1 + 9*1, 1*1 + 3*2 + 9*1, 1*1 + 3*0 + 9*1, 1*1 + 3*1 + 9*2, 1*1 + 3*1 + 9*0};

      for (auto & info: B)
      {
	      grid->getBlockInfoAll(info.level, info.Z).auxiliary = nullptr;
        const int aux = 1<<info.level;

        const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*aux-1;
        const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*aux-1;
        const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*aux-1;
        const int  xskip = info.index[0]==0 ? -1 : 1;
        const int  yskip = info.index[1]==0 ? -1 : 1;
        const int  zskip = info.index[2]==0 ? -1 : 1;

        bool storeFace[6] = {false,false,false,false,false,false};
        bool stored = false;

        for (int f=0; f<6; f++)
        {
          const int code[3] = { icode[f]%3-1, (icode[f]/3)%3-1, (icode[f]/9)%3-1};
          if (!_grid.xperiodic && code[0] == xskip && xskin) continue;
          if (!_grid.yperiodic && code[1] == yskip && yskin) continue;
          if (!_grid.zperiodic && code[2] == zskip && zskin) continue;
          #if DIMENSION == 2
          if (code[2] != 0) continue;
          #endif

          if (!grid->Tree(info.level,info.Znei_(code[0],code[1],code[2])).Exists())
          {
            storeFace[ abs(code[0]) *  std::max(0,code[0]) + abs(code[1]) * (std::max(0,code[1])+2) + abs(code[2]) * (std::max(0,code[2])+4)  ] = true;
            stored = true;
          }
        }
        if (stored)
        {
          Cases.push_back(Case(storeFace,BlockType::sizeX,BlockType::sizeY,BlockType::sizeZ,info.level,info.Z));
        }
      }
      size_t Cases_index = 0;
      if (Cases.size()>0)
      for (auto &info : B)
      {
        if (Cases_index == Cases.size()) break;
        if (Cases[Cases_index].level == info.level && Cases[Cases_index].Z == info.Z)
        {
          MapOfCases.insert(std::pair<std::array<long long, 2>, Case *>({Cases[Cases_index].level, Cases[Cases_index].Z},&Cases[Cases_index]));
          grid->getBlockInfoAll(Cases[Cases_index].level, Cases[Cases_index].Z).auxiliary = &Cases[Cases_index];
          info.auxiliary = &Cases[Cases_index];
          Cases_index ++;
        }
      }
   }

   ///Go over each coarse-fine interface and perform the flux corrections, assuming the associated BlockCases have been filled with the fluxes by the user
   virtual void FillBlockCases()
   {
      // This assumes that the BlockCases have been filled by the user somehow...
      std::vector<BlockInfo> &B = (*grid).getBlocksInfo();

      std::array<int, 3> blocksPerDim = (*grid).getMaxBlocks();

      std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1, 1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1, 1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};

      #pragma omp parallel for
      for (size_t i = 0 ; i < B.size() ; i++)
      {
        BlockInfo & info = B[i];
        const int aux = 1 << info.level;
        const bool xskin = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
        const bool yskin = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
        const bool zskin = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
        const int  xskip = info.index[0] == 0 ? -1 : 1;
        const int  yskip = info.index[1] == 0 ? -1 : 1;
        const int  zskip = info.index[2] == 0 ? -1 : 1;

        for (int f = 0; f < 6; f++)
        {
          const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1, (icode[f] / 9) % 3 - 1};

          if (!grid->xperiodic && code[0] == xskip && xskin) continue;
          if (!grid->yperiodic && code[1] == yskip && yskin) continue;
          if (!grid->zperiodic && code[2] == zskip && zskin) continue;
          #if DIMENSION == 2
          if (code[2] != 0) continue;
          #endif

          bool checkFiner = grid->Tree(info.level, info.Znei_(code[0], code[1], code[2])).CheckFiner();

          if (checkFiner)
          {
            FillCase(info, code);

            const int myFace = abs(code[0]) * std::max(0, code[0]) + abs(code[1]) * (std::max(0, code[1]) + 2) + abs(code[2]) * (std::max(0, code[2]) + 4);
            std::array<long long, 2> temp = {(long long)info.level, info.Z};
            auto search             = MapOfCases.find(temp);
            assert(search != MapOfCases.end());
            Case &CoarseCase                     = (*search->second);
            std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
            const int d  = myFace / 2;
            const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
            const int N2 = CoarseCase.m_vSize[d2];
            BlockType &block = *(BlockType *)info.ptrBlock;

            #if DIMENSION == 3
              // WARNING: tmp indices are tmp[z][y][x][Flow Quantity]!
              const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
              const int N1 = CoarseCase.m_vSize[d1];
              if (d == 0)
              {
                const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeX - 1;
                for (int i1 = 0; i1 < N1; i1 ++)
                for (int i2 = 0; i2 < N2; i2 ++)
                {
                  block(j,i2,i1) += CoarseFace[i2 + i1 * N2];
                  CoarseFace[i2 + i1 * N2].clear();
                }
              }
              else if (d == 1)
              {
                const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeY - 1;
                for (int i1 = 0; i1 < N1; i1 ++)
                for (int i2 = 0; i2 < N2; i2 ++)
                {
                  block(i2,j,i1) += CoarseFace[i2 + i1 * N2];
                  CoarseFace[i2 + i1 * N2].clear();
                }
              }
              else
              {
                const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeZ - 1;
                for (int i1 = 0; i1 < N1; i1 ++)
                for (int i2 = 0; i2 < N2; i2 ++)
                {
                  block(i2,i1,j) += CoarseFace[i2 + i1 * N2];
                  CoarseFace[i2 + i1 * N2].clear();
                }
              }               
            #else
              assert(d!=2);
              if (d == 0)
              {
                const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeX - 1;
                for (int i2 = 0; i2 < N2; i2 ++)
                {
                  block(j,i2) += CoarseFace[i2];
                  CoarseFace[i2].clear();
                }
              }
              else //if (d == 1)
              {
                const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeY - 1;
                for (int i2 = 0; i2 < N2; i2 ++)
                {
                  block(i2,j) += CoarseFace[i2];
                  CoarseFace[i2].clear();
                }
              }
            #endif
          }
        }
      }
   }

};

} // namespace cubism
