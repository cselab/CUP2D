#pragma once

#include "FluxCorrection.h"

namespace cubism
{

/**
 * @brief Performs flux corrections at coarse-fine block interfaces, with multiple MPI processes.
 * 
 * This class can replace the coarse fluxes stored at BlockCases with the sum of the
 * fine fluxes (also stored at BlockCases). This ensures conservation of the quantity 
 * whose flux we compute.
 * @tparam TFluxCorrection The single-node version from which this class inherits
 */

template <typename TFluxCorrection>
class FluxCorrectionMPI : public TFluxCorrection
{
 public:
   using TGrid = typename TFluxCorrection::GridType;
   typedef typename TFluxCorrection::ElementType ElementType;
   typedef typename TFluxCorrection::Real Real;
   typedef typename TFluxCorrection::BlockType BlockType;
   typedef BlockCase<BlockType> Case;
   int size;
 protected:
   
   ///Auxiliary struct to keep track of coarse-fine interfaces between two different MPI processes
   struct face
   {
      BlockInfo *infos[2]; ///< the two BlockInfos of the interface
      int icode[2]; ///< encodes what face (+x,-x,+y,-y,+z,-z) is shared by the two Blocks
      int offset; ///< offset in the send/recv buffers where the data for this face will be put
      // infos[0] : Fine block
      // infos[1] : Coarse block
      face(BlockInfo &i0, BlockInfo &i1, int a_icode0, int a_icode1)
      {
         infos[0] = &i0;
         infos[1] = &i1;
         icode[0] = a_icode0;
         icode[1] = a_icode1;
      }
      bool operator<(const face &other) const
      {
         if (infos[0]->blockID_2 == other.infos[0]->blockID_2)
         {
            return (icode[0] < other.icode[0]);
         }
         else
         {
            return (infos[0]->blockID_2 < other.infos[0]->blockID_2);
         }
      }
   };

   std::vector<std::vector<Real>> send_buffer; ///< multiple buffers to send to other ranks
   std::vector<std::vector<Real>> recv_buffer; ///< multiple buffers to receive from other ranks
   std::vector<std::vector<face>> send_faces; ///< buffers with 'faces' meta-data to send
   std::vector<std::vector<face>> recv_faces; ///< buffers with 'faces' meta-data to receive

   ///Perform flux correction for face 'F'
   void FillCase(face & F)
   {
      BlockInfo & info  = *F.infos[1];
      const int icode   = F.icode[1];
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};

      const int myFace = abs(code[0]) * std::max(0, code[0]) + abs(code[1]) * (std::max(0, code[1]) + 2) + abs(code[2]) * (std::max(0, code[2]) + 4);
      std::array<long long, 2> temp = {(long long)info.level, info.Z};
      auto search             = TFluxCorrection::MapOfCases.find(temp);
      assert(search != TFluxCorrection::MapOfCases.end());
      Case &CoarseCase                     = (*search->second);
      std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
      #if DIMENSION == 3
      for (int B = 0; B <= 3; B++) // loop over fine blocks that make up coarse face
      #else
      for (int B = 0; B <= 1; B++) // loop over fine blocks that make up coarse face
      #endif
      {
         const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

         #if DIMENSION == 3
            const long long Z = (*TFluxCorrection::grid).getZforward(info.level + 1,
                                     2 * info.index[0] + std::max(code[0], 0) + code[0] + (B % 2) * std::max(0, 1 - abs(code[0])),
                                     2 * info.index[1] + std::max(code[1], 0) + code[1] + aux * std::max(0, 1 - abs(code[1])),
                                     2 * info.index[2] + std::max(code[2], 0) + code[2] + (B / 2) * std::max(0, 1 - abs(code[2])));
         #else
            const long long Z = (*TFluxCorrection::grid).getZforward(info.level + 1,
                            2 * info.index[0] + std::max(code[0], 0) + code[0] + (B % 2) * std::max(0, 1 - abs(code[0])),
                            2 * info.index[1] + std::max(code[1], 0) + code[1] + aux * std::max(0, 1 - abs(code[1])));
         #endif
         if (Z != F.infos[0]->Z) continue;

         const int d  = myFace / 2;
         const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
         const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
         const int N1 = CoarseCase.m_vSize[d1];
         const int N2 = CoarseCase.m_vSize[d2];

         int base = 0; //(B%2)*(N1/2)+ (B/2)*(N2/2)*N1;
         if      (B == 1) base = (N2 / 2) + (0) * N2;
         else if (B == 2) base = (0) + (N1 / 2) * N2;
         else if (B == 3) base = (N2 / 2) + (N1 / 2) * N2;

         int r   = (*TFluxCorrection::grid).Tree(F.infos[0]->level,F.infos[0]->Z).rank();
         int dis = 0;

         #if DIMENSION == 3
            for (int i1 = 0; i1 < N1; i1 += 2)
            for (int i2 = 0; i2 < N2; i2 += 2)
            {
               for (int j = 0; j < ElementType::DIM; j++)
                  CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].member(j) +=recv_buffer[r][F.offset + dis + j];
               dis += ElementType::DIM;
            }
         #else
            for (int i2 = 0; i2 < N2; i2 += 2)
            {
               for (int j = 0; j < ElementType::DIM; j++)
                  CoarseFace[base + (i2 / 2)].member(j) +=recv_buffer[r][F.offset + dis + j];
               dis += ElementType::DIM;
            }
         #endif
      }
   }

   ///Perform flux correction for face 'F' and direction encoded by 'code*' (for data received from other processes)
   void FillCase_2(face & F, int codex, int codey, int codez)
   {
      BlockInfo & info  = *F.infos[1];
      const int icode   = F.icode[1];
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};

      if (abs(code[0]) != codex) return;
      if (abs(code[1]) != codey) return;
      if (abs(code[2]) != codez) return;

      const int myFace = abs(code[0]) * std::max(0, code[0]) + abs(code[1]) * (std::max(0, code[1]) + 2) + abs(code[2]) * (std::max(0, code[2]) + 4);
      std::array<long long, 2> temp = {(long long)info.level, info.Z};
      auto search             = TFluxCorrection::MapOfCases.find(temp);
      assert(search != TFluxCorrection::MapOfCases.end());
      Case &CoarseCase                     = (*search->second);
      std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];

      const int d  = myFace / 2;
      const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      const int N2 = CoarseCase.m_vSize[d2];
      BlockType &block = *(BlockType *)info.ptrBlock;
      #if DIMENSION == 3
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

 public:

   ///Prepare the FluxCorrection class for a given 'grid' by allocating BlockCases at each coarse-fine interface
   virtual void prepare(TGrid &_grid) override
   {
      if (_grid.UpdateFluxCorrection == false) return;
      _grid.UpdateFluxCorrection = false;

      int temprank;
      MPI_Comm_size(_grid.getWorldComm(), &size);
      MPI_Comm_rank(_grid.getWorldComm(), &temprank);
      TFluxCorrection::rank = temprank;


      send_buffer.resize(size);
      recv_buffer.resize(size);
      send_faces.resize(size);
      recv_faces.resize(size);

      for (int r = 0; r < size; r++)
      {
         send_faces[r].clear();
         recv_faces[r].clear();
      }

      std::vector<int> send_buffer_size(size, 0);
      std::vector<int> recv_buffer_size(size, 0);

      const int NC = ElementType::DIM;

      int blocksize[3];
      blocksize[0] = BlockType::sizeX;
      blocksize[1] = BlockType::sizeY;
      blocksize[2] = BlockType::sizeZ;

      TFluxCorrection::Cases.clear();
      TFluxCorrection::MapOfCases.clear();

      TFluxCorrection::grid = &_grid;
      std::vector<BlockInfo> &BB = (*TFluxCorrection::grid).getBlocksInfo();

      std::array<int, 3> blocksPerDim = _grid.getMaxBlocks();

      std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                  1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                  1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};

      for (auto &info : BB)
      {
         (*TFluxCorrection::grid).getBlockInfoAll(info.level, info.Z).auxiliary = nullptr;
         info.auxiliary = nullptr;

         const int aux = 1 << info.level;

         const bool xskin = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
         const bool yskin = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
         const bool zskin = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;

         const int xskip = info.index[0] == 0 ? -1 : 1;
         const int yskip = info.index[1] == 0 ? -1 : 1;
         const int zskip = info.index[2] == 0 ? -1 : 1;

         bool storeFace[6] = {false, false, false, false, false, false};
         bool stored       = false;

         for (int f = 0; f < 6; f++)
         {
            const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1, (icode[f] / 9) % 3 - 1};

            if (!_grid.xperiodic && code[0] == xskip && xskin) continue;
            if (!_grid.yperiodic && code[1] == yskip && yskin) continue;
            if (!_grid.zperiodic && code[2] == zskip && zskin) continue;
            #if DIMENSION == 2
            if (code[2] != 0) continue;
            #endif

            if (! (*TFluxCorrection::grid).Tree(info.level, info.Znei_(code[0], code[1], code[2])).Exists())
            {
               storeFace[abs(code[0]) * std::max(0, code[0]) + abs(code[1]) * (std::max(0, code[1]) + 2) +
                         abs(code[2]) * (std::max(0, code[2]) + 4)] = true;
               stored                                          = true;
            }

            int L[3];
            L[0]  = (code[0] == 0) ? blocksize[0] / 2 : 1;
            L[1]  = (code[1] == 0) ? blocksize[1] / 2 : 1;
            #if DIMENSION == 3
               L[2] = (code[2] == 0) ? blocksize[2] / 2 : 1;
            #else
               L[2] = 1;
            #endif
            int V = L[0] * L[1] * L[2];

            if ( (*TFluxCorrection::grid).Tree(info.level, info.Znei_(code[0], code[1], code[2])).CheckCoarser())
            {
               BlockInfo & infoNei = (*TFluxCorrection::grid).getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
               const long long nCoarse = infoNei.Zparent;
               BlockInfo &infoNeiCoarser = (*TFluxCorrection::grid).getBlockInfoAll(info.level - 1, nCoarse);
               const int infoNeiCoarserrank = (*TFluxCorrection::grid).Tree(info.level - 1, nCoarse).rank();
               {
                  int code2[3] = {-code[0], -code[1], -code[2]};
                  int icode2   = (code2[0] + 1) + (code2[1] + 1) * 3 + (code2[2] + 1) * 9;
                  send_faces[infoNeiCoarserrank].push_back(face(info, infoNeiCoarser, icode[f], icode2));
                  send_buffer_size[infoNeiCoarserrank] += V;
               }
            }
            else if ( (*TFluxCorrection::grid).Tree(info.level, info.Znei_(code[0], code[1], code[2])).CheckFiner())
            {
               BlockInfo & infoNei = (*TFluxCorrection::grid).getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
               int Bstep = 1;                      // face
               #if DIMENSION == 3
               for (int B = 0; B <= 3; B += Bstep) // loop over blocks that make up face
               #else
               for (int B = 0; B <= 1; B += Bstep) // loop over blocks that make up face
               #endif 
               {
                  const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                  const long long nFine  = infoNei.Zchild[std::max(-code[0], 0) + (B % 2) * std::max(0, 1 - abs(code[0]))]
                                                         [std::max(-code[1], 0) + temp    * std::max(0, 1 - abs(code[1]))]
                                                         [std::max(-code[2], 0) + (B / 2) * std::max(0, 1 - abs(code[2]))];
                  const int infoNeiFinerrank = (*TFluxCorrection::grid).Tree(infoNei.level + 1, nFine).rank();
                  {
                     BlockInfo &infoNeiFiner = (*TFluxCorrection::grid).getBlockInfoAll(infoNei.level + 1, nFine);
                     int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
                     recv_faces[infoNeiFinerrank].push_back(face(infoNeiFiner, info, icode2, icode[f]));
                     recv_buffer_size[infoNeiFinerrank] += V;
                  }
               }
            }
         } // icode = 0,...,26

         if (stored)
         {
            TFluxCorrection::Cases.push_back(Case(storeFace, BlockType::sizeX, BlockType::sizeY, BlockType::sizeZ, info.level, info.Z));
         }
      }

      size_t Cases_index = 0;
      if (TFluxCorrection::Cases.size()>0)
      for (auto &info : BB)
      {
         if (Cases_index == TFluxCorrection::Cases.size()) break;
         if (TFluxCorrection::Cases[Cases_index].level == info.level &&
             TFluxCorrection::Cases[Cases_index].Z     == info.Z)
         {
            TFluxCorrection::MapOfCases.insert(std::pair<std::array<long long, 2>, Case *>({TFluxCorrection::Cases[Cases_index].level, TFluxCorrection::Cases[Cases_index].Z},&TFluxCorrection::Cases[Cases_index]));
            TFluxCorrection::grid->getBlockInfoAll(TFluxCorrection::Cases[Cases_index].level, TFluxCorrection::Cases[Cases_index].Z).auxiliary = &TFluxCorrection::Cases[Cases_index];
            info.auxiliary = &TFluxCorrection::Cases[Cases_index];
            Cases_index ++;
         }
      }

      // 2.Sort faces
      for (int r = 0; r < size; r++)
      {
         std::sort(send_faces[r].begin(), send_faces[r].end());
         std::sort(recv_faces[r].begin(), recv_faces[r].end());
      }

      // 3.Define map
      for (int r = 0; r < size; r++)
      {
         send_buffer[r].resize(send_buffer_size[r] * NC);
         recv_buffer[r].resize(recv_buffer_size[r] * NC);

         int offset = 0;
         for (int k = 0; k < (int)recv_faces[r].size(); k++)
         {
            face &f = recv_faces[r][k];

            const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1, (f.icode[1] / 9) % 3 - 1};

            int L[3];
            L[0]  = (code[0] == 0) ? blocksize[0] / 2 : 1;
            L[1]  = (code[1] == 0) ? blocksize[1] / 2 : 1;
            #if DIMENSION == 3
               L[2]  = (code[2] == 0) ? blocksize[2] / 2 : 1;
            #else
               L[2] = 1;
            #endif
            int V = L[0] * L[1] * L[2];

            f.offset = offset;

            offset += V * NC;
         }
      }
   }

   ///Go over each coarse-fine interface and perform the flux corrections, assuming the associated BlockCases have been filled with the fluxes by the user
   virtual void FillBlockCases() override
   {
      auto MPI_real = (sizeof(Real) == sizeof(float) ) ? MPI_FLOAT : ( (sizeof(Real) == sizeof(double)) ? MPI_DOUBLE : MPI_LONG_DOUBLE);

      // This assumes that the BlockCases have been filled by the user somehow...

      // 1.Pack send data
      for (int r = 0; r < size; r++)
      {

         int displacement = 0;
         for (int k = 0; k < (int)send_faces[r].size(); k++)
         {
            face &f = send_faces[r][k];

            BlockInfo &info = *(f.infos[0]);

            auto search = TFluxCorrection::MapOfCases.find({(long long)info.level, info.Z});
            assert(search != TFluxCorrection::MapOfCases.end());

            Case &FineCase = (*search->second);

            int icode         = f.icode[0];
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
            const int myFace = abs(code[0]) * std::max(0, code[0]) + abs(code[1]) * (std::max(0, code[1]) + 2) +
                               abs(code[2]) * (std::max(0, code[2]) + 4);
            std::vector<ElementType> &FineFace = FineCase.m_pData[myFace];

            const int d  = myFace / 2;
            const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
            const int N2 = FineCase.m_vSize[d2];
            #if DIMENSION == 3
            const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
            const int N1 = FineCase.m_vSize[d1];
               for (int i1 = 0; i1 < N1; i1 += 2)
               for (int i2 = 0; i2 < N2; i2 += 2)
               {
                  ElementType avg = ((FineFace[i2 + i1 * N2] + FineFace[i2 + 1 + i1 * N2]) + (FineFace[i2 + (i1 + 1) * N2] + FineFace[i2 + 1 + (i1 + 1) * N2]));
                  for (int j = 0 ; j < ElementType::DIM; j++) send_buffer[r][displacement + j] = avg.member(j);
                  displacement += ElementType::DIM;
                  FineFace[i2 + i1 * N2].clear();
                  FineFace[i2 + 1 + i1 * N2].clear();
                  FineFace[i2 + (i1 + 1) * N2].clear();
                  FineFace[i2 + 1 + (i1 + 1) * N2].clear();
               }
            #else
              for (int i2 = 0; i2 < N2; i2 += 2)
              {
                 ElementType avg = FineFace[i2] + FineFace[i2 + 1];
                 for (int j = 0 ; j < ElementType::DIM; j++) send_buffer[r][displacement + j] = avg.member(j);
                 displacement += ElementType::DIM;
                 FineFace[i2    ].clear();
                 FineFace[i2 + 1].clear();
              }
            #endif
         }
      }

      std::vector<MPI_Request> send_requests;
      std::vector<MPI_Request> recv_requests;

      const int me = TFluxCorrection::rank;
      for (int r = 0; r < size; r++) if (r != me)
      {
         if (recv_buffer[r].size() != 0)
         {
            MPI_Request req{};
            recv_requests.push_back(req);
            MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_real, r, 123456,
                      (*TFluxCorrection::grid).getWorldComm(), &recv_requests.back());
         }
         if (send_buffer[r].size() != 0)
         {
            MPI_Request req{};
            send_requests.push_back(req);
            MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_real, r, 123456,
                      (*TFluxCorrection::grid).getWorldComm(), &send_requests.back());
         }
      }

      MPI_Request me_send_request;
      MPI_Request me_recv_request;
      if (recv_buffer[me].size() != 0)
      {
         MPI_Irecv(&recv_buffer[me][0], recv_buffer[me].size(), MPI_real, me, 123456,
                   (*TFluxCorrection::grid).getWorldComm(), &me_recv_request);
      }
      if (send_buffer[me].size() != 0)
      {
         MPI_Isend(&send_buffer[me][0], send_buffer[me].size(), MPI_real, me, 123456,
                   (*TFluxCorrection::grid).getWorldComm(), &me_send_request);
      }

      if (recv_buffer[me].size() > 0) MPI_Waitall(1, &me_recv_request, MPI_STATUSES_IGNORE);
      if (send_buffer[me].size() > 0) MPI_Waitall(1, &me_send_request, MPI_STATUSES_IGNORE);

      for (int index = 0; index < (int)recv_faces[me].size(); index++)
         FillCase(recv_faces[me][index]);

      if (recv_requests.size() > 0) MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);

      for (int r = 0; r < size; r++) if (r!=me)
         for (int index = 0; index < (int)recv_faces[r].size(); index++)
            FillCase(recv_faces[r][index]);

      //first do x, then y then z. It is done like this to preserve symmetry and not favor any direction
      for (int r = 0; r < size; r++) //if (r!=me)
         for (int index = 0; index < (int)recv_faces[r].size(); index++)
            FillCase_2(recv_faces[r][index],1,0,0);
      for (int r = 0; r < size; r++) //if (r!=me)
         for (int index = 0; index < (int)recv_faces[r].size(); index++)
            FillCase_2(recv_faces[r][index],0,1,0);
      #if DIMENSION == 3
      for (int r = 0; r < size; r++) //if (r!=me)
         for (int index = 0; index < (int)recv_faces[r].size(); index++)
            FillCase_2(recv_faces[r][index],0,0,1);
      #endif

      if (send_requests.size() > 0) MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);
   }

};

} // namespace cubism
