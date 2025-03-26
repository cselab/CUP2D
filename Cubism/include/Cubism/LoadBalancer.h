#pragma once

#include "BlockInfo.h"
#include <algorithm>
#include <cstring>
#include <string>

namespace cubism
{

/** Takes care of load-balancing of Blocks.
 * This class will redistribute Blocks among different MPI ranks for two reasons:
 * 1) Eight (in 3D) or four (in 2D) blocks need to be compressed to one, but they are owned by
 *    different ranks. PrepareCompression() will collect them all to one rank, so that 
 *    they can be compressed.
 * 2) There is a load imbalance after the grid is refined or compressed. If the imbalance is 
 *    not great (load imbalance ratio < 1.1), a 1D-diffusion based scheme is used to redistribute
 *    blocks along the 1D Space-Filling-Curve. Otherwise, all blocks are simply evenly 
 *    redistributed among all ranks.
 * @tparam TGrid: the type of GridMPI to perform load-balancing for
 */
template <typename TGrid>
class LoadBalancer
{
 public:
   typedef typename TGrid::Block BlockType;
   typedef typename TGrid::Block::ElementType ElementType;
   typedef typename TGrid::Block::ElementType::RealType Real;
   bool movedBlocks;///< =true if load-balancing is performed when Balance_Diffusion of Balance_Global is called

 protected:
   TGrid *grid;///< grid where load balancing will be performed

   ///MPI datatype and auxiliary struct used to send/receive blocks among ranks
   MPI_Datatype MPI_BLOCK;
   struct MPI_Block
   {
      long long mn[2]; ///< level and Z-order index of a BlockInfo
      Real data[sizeof(BlockType) / sizeof(Real)]; ///< buffer array of data to send/receive 

      /** Constructor; calls 'prepare'.
       * @param info: BlockInfo for block to be sent/received.
       * @param Fillptr: true if we want the data of the GridBlock to be copied to this MPI_Block.
       */
      MPI_Block(const BlockInfo &info, const bool Fillptr = true)
      {
         prepare(info,Fillptr);
      }

      /** Prepare the MPI_Block with data from a GridBlock.
       * @param info: BlockInfo for block to be sent/received.
       * @param Fillptr: true if we want the data of the GridBlock to be copied to this MPI_Block.
       */
      void prepare(const BlockInfo &info, const bool Fillptr = true)
      {
         mn[0] = info.level;
         mn[1] = info.Z;
         if (Fillptr)
         {
            Real *aux = &((BlockType *)info.ptrBlock)->data[0][0][0].member(0);
            std::memcpy(&data[0], aux, sizeof(BlockType));
         }
      }

      MPI_Block() {}
   };

   ///Allocate a block at a given level and Z-index and fill it with received data
   void AddBlock(const int level, const long long Z, Real * data)
   {
      //1. Allocate the block from the grid
      grid->_alloc(level, Z);

      //2. Fill the block with data received
      BlockInfo &info = grid->getBlockInfoAll(level, Z);
      BlockType *b1 = (BlockType *)info.ptrBlock;
      assert(b1 != NULL);
      Real *a1 = &b1->data[0][0][0].member(0);
      std::memcpy(a1, data, sizeof(BlockType));

      //3. Update status of children and parent block of newly allocated block
      #if DIMENSION == 3
         int p[3];
         BlockInfo::inverse(Z, level, p[0], p[1], p[2]);
         if (level < grid->getlevelMax() - 1)
            for (int k1 = 0; k1 < 2; k1++)
            for (int j1 = 0; j1 < 2; j1++)
            for (int i1 = 0; i1 < 2; i1++)
            {
               const long long nc = grid->getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1, 2 * p[2] + k1);
               grid->Tree(level + 1, nc).setCheckCoarser();
            }
         if (level > 0)
         {
            const long long nf = grid->getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
            grid->Tree(level - 1, nf).setCheckFiner();
         }
      #else
         int p[2];
         BlockInfo::inverse(Z, level, p[0], p[1]);
         if (level < grid->getlevelMax() - 1)
            for (int j1 = 0; j1 < 2; j1++)
            for (int i1 = 0; i1 < 2; i1++)
            {
               const long long nc = grid->getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
               grid->Tree(level + 1, nc).setCheckCoarser();
            }
         if (level > 0)
         {
            const long long nf = grid->getZforward(level - 1, p[0] / 2, p[1] / 2);
            grid->Tree(level - 1, nf).setCheckFiner();
         }
      #endif
   }

 public:
   /// Constructor
   LoadBalancer(TGrid &a_grid)
   {
      grid = &a_grid;
      movedBlocks = false;

      //Create MPI datatype to send/receive blocks (data) + two integers (their level and Z-index)
      int array_of_blocklengths[2]       = {2, sizeof(BlockType) / sizeof(Real)};
      MPI_Aint array_of_displacements[2] = {0, 2 * sizeof(long long)};
      MPI_Datatype array_of_types[2];
      array_of_types[0] = MPI_LONG_LONG;
      if      (sizeof(Real) == sizeof(float)      ) array_of_types[1] = MPI_FLOAT;
      else if (sizeof(Real) == sizeof(double)     ) array_of_types[1] = MPI_DOUBLE;
      else if (sizeof(Real) == sizeof(long double)) array_of_types[1] = MPI_LONG_DOUBLE;
      MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, &MPI_BLOCK);
      MPI_Type_commit(&MPI_BLOCK);
   }

   /// Destructor
   ~LoadBalancer() { MPI_Type_free(&MPI_BLOCK); }

   /// Compression of eight blocks requires all of them to be owned by one rank; this function collects all groups of 8 blocks to be compressed to a single rank.
   void PrepareCompression()
   {
      const int size = grid->get_world_size();
      const int rank = grid->rank();

      std::vector<BlockInfo> &I = grid->getBlocksInfo();
      std::vector<std::vector<MPI_Block>> send_blocks(size);
      std::vector<std::vector<MPI_Block>> recv_blocks(size);

      //Loop over blocks
      for (auto &b : I)
      {
         #if DIMENSION == 3
         const long long nBlock = grid->getZforward(b.level, 2 * (b.index[0] / 2), 2 * (b.index[1] / 2), 2 * (b.index[2] / 2));
         #else
         const long long nBlock = grid->getZforward(b.level, 2 * (b.index[0] / 2), 2 * (b.index[1] / 2));
         #endif

         const BlockInfo &base = grid->getBlockInfoAll(b.level, nBlock);

         //If the 'base' block does not exist, no compression will take place. Continue to next block.
         //By now, if 'base' block is marked for compression it means that the remaining 7 (3, in 2D)
         //blocks will also need compression, so we check if base.state == Compress.
         if (!grid->Tree(base).Exists() || base.state != Compress) continue;

         const BlockInfo &bCopy = grid->getBlockInfoAll(b.level, b.Z);
         const int baserank = grid->Tree(b.level, nBlock).rank();
         const int brank    = grid->Tree(b.level, b.Z).rank();

         //if 'b' is NOT the 'base' block we send it to the rank that owns the 'base' block.
         if (b.Z != nBlock)
         {
            if (baserank != rank && brank == rank)
            {
               send_blocks[baserank].push_back({bCopy});
               grid->Tree(b.level, b.Z).setrank(baserank);
            }
         }
         //if 'b' is the 'base' block we collect the remaining 7 (3, in 2D) blocks that will be compressed with it.
         else
         {
            #if DIMENSION ==3
            for (int k = 0; k < 2; k++)
            #endif
            for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
               #if DIMENSION ==3
               const long long n = grid->getZforward(b.level, b.index[0] + i, b.index[1] + j, b.index[2] + k);
               #else
               const long long n = grid->getZforward(b.level, b.index[0] + i, b.index[1] + j);
               #endif
               if (n == nBlock) continue;
               BlockInfo &temp    = grid->getBlockInfoAll(b.level, n);
               const int temprank = grid->Tree(b.level, n).rank();
               if (temprank != rank)
               {
                  recv_blocks[temprank].push_back({temp, false});
                  grid->Tree(b.level, n).setrank(baserank);
               }
            }
         }
      }

      //1/4 Perform the sends/receives of blocks
      std::vector<MPI_Request> requests;
      for (int r = 0; r < size; r++)
         if (r != rank)
         {
            if (recv_blocks[r].size() != 0)
            {
               MPI_Request req{};
               requests.push_back(req);
               MPI_Irecv(&recv_blocks[r][0], recv_blocks[r].size(), MPI_BLOCK, r, 2468, grid->getWorldComm(), &requests.back());
            }
            if (send_blocks[r].size() != 0)
            {
               MPI_Request req{};
               requests.push_back(req);
               MPI_Isend(&send_blocks[r][0], send_blocks[r].size(), MPI_BLOCK, r, 2468, grid->getWorldComm(), &requests.back());
            }
         }

      //2/4 Do some work while sending/receiving. Here we deallocate the blocks we sent.
      for (int r = 0; r < size; r++)
         for (int i = 0; i < (int)send_blocks[r].size(); i++)
         {
            grid->_dealloc(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]);
            grid->Tree(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]).setCheckCoarser();
         }

      //3/4 Wait for communication to complete
      if (requests.size() != 0)
      {
         movedBlocks = true;
         MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
      }

      //4/4 Allocate the blocks we received and copy data to them.
      for (int r = 0; r < size; r++)
         for (int i = 0; i < (int)recv_blocks[r].size(); i++)
         {
            const int level = (int) recv_blocks[r][i].mn[0];
            const long long Z = recv_blocks[r][i].mn[1];
            grid->_alloc(level, Z);
            BlockInfo & info = grid->getBlockInfoAll(level, Z);
            BlockType *b1  = (BlockType *)info.ptrBlock;
            assert(b1 != NULL);
            Real *a1 = &b1->data[0][0][0].member(0);
            std::memcpy(a1, recv_blocks[r][i].data, sizeof(BlockType));
         }
   }

   /// Redistributes blocks with diffusion algorithm along the 1D Space-Filling Hilbert Curve; block_distribution[i] is the number of blocks owned by rank i, for i=0,...,#of ranks -1
   void Balance_Diffusion(const bool verbose, std::vector<long long> & block_distribution)
   {
      const int size = grid->get_world_size();
      const int rank = grid->rank();

      movedBlocks = false;
      {
         long long max_b = block_distribution[0];
         long long min_b = block_distribution[0];
         for (auto & b : block_distribution)
         {
              max_b = std::max(max_b,b);
              min_b = std::min(min_b,b);
         }
         const double ratio = static_cast<double>(max_b) / min_b;
         if (rank == 0 && verbose)
         {
           std::cout << "Load imbalance ratio = " << ratio << std::endl;
         }
         if (ratio > 1.01 || min_b == 0)
         {
            Balance_Global(block_distribution);
            return;
         }
      }

      const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
      const int left  = (rank == 0       ) ? MPI_PROC_NULL : rank - 1;

      const int my_blocks = grid->getBlocksInfo().size();

      int right_blocks, left_blocks;

      MPI_Request reqs[4];
      MPI_Irecv(&left_blocks , 1, MPI_INT, left , 123, grid->getWorldComm(), &reqs[0]);
      MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, grid->getWorldComm(), &reqs[1]);
      MPI_Isend(&my_blocks   , 1, MPI_INT, left , 456, grid->getWorldComm(), &reqs[2]);
      MPI_Isend(&my_blocks   , 1, MPI_INT, right, 123, grid->getWorldComm(), &reqs[3]);

      MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);

      const int nu         = 4;
      const int flux_left  = (rank == 0       ) ? 0 : (my_blocks - left_blocks ) / nu;
      const int flux_right = (rank == size - 1) ? 0 : (my_blocks - right_blocks) / nu;

      std::vector<BlockInfo> SortedInfos = grid->getBlocksInfo();
      if (flux_right != 0 || flux_left != 0) std::sort(SortedInfos.begin(), SortedInfos.end());

      std::vector<MPI_Block> send_left;
      std::vector<MPI_Block> recv_left;
      std::vector<MPI_Block> send_right;
      std::vector<MPI_Block> recv_right;

      std::vector<MPI_Request> request;

      if (flux_left > 0) // then I will send blocks to my left rank
      {
         send_left.resize(flux_left);
         #pragma omp parallel for schedule(runtime)
         for (int i = 0; i < flux_left; i++) send_left[i].prepare(SortedInfos[i]);
         MPI_Request req{};
         request.push_back(req);
         MPI_Isend(&send_left[0], send_left.size(), MPI_BLOCK, left, 7890, grid->getWorldComm(), &request.back());
      }
      else if (flux_left < 0) // then I will receive blocks from my left rank
      {
         recv_left.resize(abs(flux_left));
         MPI_Request req{};
         request.push_back(req);
         MPI_Irecv(&recv_left[0], recv_left.size(), MPI_BLOCK, left, 4560, grid->getWorldComm(), &request.back());
      }
      if (flux_right > 0) // then I will send blocks to my right rank
      {
         send_right.resize(flux_right);
         #pragma omp parallel for schedule(runtime)
         for (int i = 0; i < flux_right; i++) send_right[i].prepare(SortedInfos[my_blocks - i - 1]);
         MPI_Request req{};
         request.push_back(req);
         MPI_Isend(&send_right[0], send_right.size(), MPI_BLOCK, right, 4560, grid->getWorldComm(), &request.back());
      }
      else if (flux_right < 0) // then I will receive blocks from my right rank
      {
         recv_right.resize(abs(flux_right));
	      MPI_Request req{};
         request.push_back(req);
         MPI_Irecv(&recv_right[0], recv_right.size(), MPI_BLOCK, right, 7890, grid->getWorldComm(), &request.back());
      }

      for (int i = 0; i < flux_right; i++)
      {
         BlockInfo &info = SortedInfos[my_blocks - i - 1];
         grid->_dealloc(info.level, info.Z);
         grid->Tree(info.level, info.Z).setrank(right);
      }

      for (int i = 0; i < flux_left; i++)
      {
         BlockInfo &info = SortedInfos[i];
         grid->_dealloc(info.level, info.Z);
         grid->Tree(info.level, info.Z).setrank(left);
      }

      if (request.size() != 0)
      {
         movedBlocks = true;
         MPI_Waitall(request.size(), &request[0], MPI_STATUSES_IGNORE);
      }
      int temp = movedBlocks ? 1 : 0;
      MPI_Request request_reduction;
      MPI_Iallreduce(MPI_IN_PLACE, &temp, 1, MPI_INT, MPI_SUM, grid->getWorldComm(), &request_reduction);

      for (int i = 0; i < -flux_left ; i++) AddBlock(recv_left [i].mn[0],recv_left [i].mn[1],recv_left [i].data);
      for (int i = 0; i < -flux_right; i++) AddBlock(recv_right[i].mn[0],recv_right[i].mn[1],recv_right[i].data);

      MPI_Wait(&request_reduction, MPI_STATUS_IGNORE);
      movedBlocks = (temp >= 1);
      grid->FillPos();
   }

   /// Redistributes all blocks evenly, along the 1D Space-Filling Hilbert Curve; all_b[i] is the number of blocks owned by rank i, for i=0,...,#of ranks -1
   void Balance_Global(std::vector<long long> & all_b)
   {
      const int size = grid->get_world_size();
      const int rank = grid->rank();

      //Redistribute all blocks evenly, along the 1D Hilbert curve.
      //all_b[i] = # of blocks currently owned by rank i.

      //sort blocks according to Z-index and level on the Hilbert curve.
      std::vector<BlockInfo> SortedInfos = grid->getBlocksInfo();
      std::sort(SortedInfos.begin(), SortedInfos.end());

      //compute the total number of blocks (total_load) and how many blocks each rank should have,
      //for a balanced load distribution
      long long total_load = 0;
      for (int r = 0; r < size; r++) total_load += all_b[r];
      long long my_load = total_load / size;
      if (rank < (total_load % size)) my_load += 1;

      std::vector<long long> index_start(size);
      index_start[0] = 0;
      for (int r = 1; r < size; r++) index_start[r] = index_start[r - 1] + all_b[r - 1];

      long long ideal_index = (total_load / size) * rank;
      ideal_index += (rank < (total_load % size)) ? rank : (total_load % size);

      //now check the actual block distribution and mark the blocks that should not be owned by a
      //particular rank and should instead be sent to another rank.
      std::vector<std::vector<MPI_Block>> send_blocks(size);
      std::vector<std::vector<MPI_Block>> recv_blocks(size);
      for (int r = 0; r < size; r++) if (rank != r)
      {
         { // check if I need to receive blocks
            const long long a1 = ideal_index;
            const long long a2 = ideal_index + my_load - 1;
            const long long b1 = index_start[r];
            const long long b2 = index_start[r] + all_b[r] - 1;
            const long long c1 = std::max(a1, b1);
            const long long c2 = std::min(a2, b2);
            if (c2 - c1 + 1 > 0) recv_blocks[r].resize(c2 - c1 + 1);
         }
         { // check if I need to send blocks
            long long other_ideal_index = (total_load / size) * r;
            other_ideal_index += (r < (total_load % size)) ? r : (total_load % size);
            long long other_load = total_load / size;
            if (r < (total_load % size)) other_load += 1;
            const long long a1 = other_ideal_index;
            const long long a2 = other_ideal_index + other_load - 1;
            const long long b1 = index_start[rank];
            const long long b2 = index_start[rank] + all_b[rank] - 1;
            const long long c1 = std::max(a1, b1);
            const long long c2 = std::min(a2, b2);
            if (c2 - c1 + 1 > 0) send_blocks[r].resize(c2 - c1 + 1);
         }
      }

      //perform the sends and receives of blocks
      int tag = 12345;
      std::vector<MPI_Request> requests;
      for (int r = 0; r < size; r++) if (recv_blocks[r].size() != 0)
      {
         MPI_Request req{};
         requests.push_back(req);
         MPI_Irecv(recv_blocks[r].data(), recv_blocks[r].size(), MPI_BLOCK, r, tag, grid->getWorldComm(), &requests.back());
      }

      long long counter_S = 0;
      long long counter_E = 0;
      for (int r = 0; r < rank; r++) if (send_blocks[r].size() != 0)
      {
         for (size_t i = 0; i < send_blocks[r].size(); i++) send_blocks[r][i].prepare(SortedInfos[counter_S + i]);
         counter_S += send_blocks[r].size();
         MPI_Request req{};
         requests.push_back(req);
         MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r, tag, grid->getWorldComm(), &requests.back());
      }
      for (int r = size-1; r > rank; r--) if (send_blocks[r].size() != 0)
      {
         for (size_t i = 0; i < send_blocks[r].size(); i++) send_blocks[r][i].prepare(SortedInfos[SortedInfos.size() - 1 - (counter_E + i)]);
         counter_E += send_blocks[r].size();
         MPI_Request req{};
         requests.push_back(req);
         MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r, tag, grid->getWorldComm(), &requests.back());
      }

      //no need to wait here, do some work first!
      //MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

      //do some work while sending/receiving, by deallocating the blocks that are being sent
      movedBlocks = true;
      std::vector<long long> deallocIDs;
      counter_S = 0;
      counter_E = 0;
      for (int r = 0; r < size; r++) if (send_blocks[r].size() != 0)
      {
         if (r < rank)
         {
            for (size_t i = 0; i < send_blocks[r].size(); i++)
            {
               BlockInfo &info = SortedInfos[counter_S + i];
               deallocIDs.push_back(info.blockID_2);
               grid->Tree(info.level, info.Z).setrank(r);
            }
            counter_S += send_blocks[r].size();
         }
         else
         {
            for (size_t i = 0; i < send_blocks[r].size(); i++)
            {
               BlockInfo &info = SortedInfos[SortedInfos.size() - 1 - (counter_E + i)];
               deallocIDs.push_back(info.blockID_2);
               grid->Tree(info.level, info.Z).setrank(r);
            }
            counter_E += send_blocks[r].size();
         }
      }
      grid->dealloc_many(deallocIDs);

      //wait for communication
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

      //allocate received blocks
      #pragma omp parallel
      {
         for (int r = 0; r < size; r++) if (recv_blocks[r].size() != 0)
         {
            #pragma omp for
            for (size_t i = 0; i < recv_blocks[r].size(); i++)
               AddBlock(recv_blocks[r][i].mn[0],recv_blocks[r][i].mn[1],recv_blocks[r][i].data);
         }
      }
      grid->FillPos();
   }
};

} // namespace cubism
