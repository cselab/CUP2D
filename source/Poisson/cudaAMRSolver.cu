//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "cudaAMRSolver.cuh"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#include "helper_cuda.h"

using namespace cubism;

cudaAMRSolver::cudaAMRSolver(SimulationData& s):sim(s)
{
  std::cout << "---------------- Calling on cudaAMRSolver() constructor ------------\n";
  // Create CUDA stream for solver and init handles
  checkCudaErrors(cudaStreamCreate(&solver_stream));
  checkCudaErrors(cublasCreate(&cublas_handle)); 
  checkCudaErrors(cusparseCreate(&cusparse_handle)); 
  // Set handles to stream
  checkCudaErrors(cublasSetStream(cublas_handle, solver_stream));
  checkCudaErrors(cusparseSetStream(cusparse_handle, solver_stream));
}

cudaAMRSolver::~cudaAMRSolver()
{
  // Destroy CUDA stream and library handles
  checkCudaErrors(cublasDestroy(cublas_handle)); 
  checkCudaErrors(cusparseDestroy(cusparse_handle)); 
  checkCudaErrors(cudaStreamDestroy(solver_stream));
  
}

void inline cudaAMRSolver::h_cooMatPushBack(
    const double &val, 
    const int &row, 
    const int &col){
  this->h_valA.push_back(val);
  this->h_cooRowA.push_back(row);
  this->h_cooColA.push_back(col);
}

// Prepare linear system for uniform grid
void cudaAMRSolver::unifLinsysPrepHost()
{
  sim.startProfiler("Poisson solver: unifLinsysPrepHost()");

  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

  // Extract number of blocks and cells on grid
  //This returns an array with the blocks that the coarsest possible 
  //mesh would have (i.e. all blocks are at level 0)
  std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();

  //Get a vector of all BlockInfos of the grid we're interested in
  std::vector<cubism::BlockInfo>&  RhsInfo = sim.tmp->getBlocksInfo();
  const size_t Nblocks = RhsInfo.size();
  const size_t N = BSX*BSY*Nblocks;

  // Allocate memory for solution 'x' and RHS vector 'b' on host
  this->h_x.resize(N);
  this->h_b.resize(N);
  // Clear contents from previous call of cudaAMRSolver::solve() and reserve memory 
  // for sparse LHS matrix 'A' (for uniform grid at most 5 elements per row).
  this->h_valA.clear();
  this->h_cooRowA.clear();
  this->h_cooColA.clear();
  this->h_valA.reserve(5 * N);
  this->h_cooRowA.reserve(5 * N);
  this->h_cooColA.reserve(5 * N);

  // No 'parallel for' to avoid accidental reorderings of COO elements during push_back
  for(size_t i=0; i< Nblocks; i++)
  {    
    BlockInfo &rhs_info = RhsInfo[i];
    ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) RhsInfo[i].ptrBlock;
    //1.Check if this is a boundary block
    int aux = 1 << rhs_info.level; // = 2^level
    int MAX_X_BLOCKS = (blocksPerDim[0] - 1)*aux; //this means that if level 0 has blocksPerDim[0] blocks in the x-direction, level rhs.level will have this many blocks
    int MAX_Y_BLOCKS = (blocksPerDim[1] - 1)*aux; //this means that if level 0 has blocksPerDim[1] blocks in the y-direction, level rhs.level will have this many blocks

    //index is the (i,j) coordinates of a block at the current level 
    const bool isWestBoundary  = (rhs_info.index[0] == 0           ); // don't check for west neighbor
    const bool isEastBoundary  = (rhs_info.index[0] == MAX_X_BLOCKS); // don't check for east neighbor
    const bool isSouthBoundary = (rhs_info.index[1] == 0           ); // don't check for south neighbor
    const bool isNorthBoundary = (rhs_info.index[1] == MAX_Y_BLOCKS); // don't check for north neighbor
    if (isWestBoundary) {}//do this
    if (isEastBoundary) {}//do that
    if (isSouthBoundary){}//do sth else
    if (isNorthBoundary){}//do this
    //keep in mind that a block might be on two boundaries (say west and north) at the same time!

    //2.Access the block's neighbors (for the Poisson solve in two dimensions we care about four neighbors in total)
    long long Z_west  = rhs_info.Znei[1-1][1][1];
    long long Z_east  = rhs_info.Znei[1+1][1][1];
    long long Z_south = rhs_info.Znei[1][1-1][1];
    long long Z_north = rhs_info.Znei[1][1+1][1];
    //rhs.Z == rhs.Znei[1][1][1] is true always

    BlockInfo &rhsNei_west  = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_west );
    BlockInfo &rhsNei_east  = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_east );
    BlockInfo &rhsNei_south = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_south);
    BlockInfo &rhsNei_north = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_north);

    //For later: there's a total of three boolean variables:
    // I.   grid->Tree(rhsNei_west).Exists()
    // II.  grid->Tree(rhsNei_west).CheckCoarser()
    // III. grid->Tree(rhsNei_west).CheckFiner()
    // And only one of them is true

    // Add matrix elements associated to interior cells of a block
    for(int iy=1; iy<BSY-1; iy++)
    for(int ix=1; ix<BSX-1; ix++)
    {
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int wn_idx = sfc_idx-1; // west neighbour
      const int en_idx = sfc_idx+1; // east neighbour
      const int sn_idx = i*BSX*BSY+(iy-1)*BSX+ix; // south neighbour 
      const int nn_idx = i*BSX*BSY+(iy+1)*BSX+ix; // north neighbour
      
      // Add diagonal matrix element
      this->h_cooMatPushBack(-4, sfc_idx, sfc_idx);
      // Add matrix element associated to west cell
      this->h_cooMatPushBack(1., sfc_idx, wn_idx);
      // Add matrix element associated to east cell
      this->h_cooMatPushBack(1., sfc_idx, en_idx);
      // Add matrix element associated to south cell
      this->h_cooMatPushBack(1., sfc_idx, sn_idx);
      // Add matrix element associated to north cell
      this->h_cooMatPushBack(1., sfc_idx, nn_idx);
    }

    // Add matrix elements associated to contributions from north/south on western/eastern boundary of block (excl.corners)
    for(int iy=1; iy<BSY-1; iy++)
    for(int ix=0; ix<BSX; ix+=(BSX-1))
    { // The inner loop executes on ix = [0, BSX-1] (western/eastern boundary) where no interaction with
      // neighbouring blocks takes place
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int sn_idx = i*BSX*BSY+(iy-1)*BSX+ix; // south neighbour 
      const int nn_idx = i*BSX*BSY+(iy+1)*BSX+ix; // north neighbour
      // Add matrix element associated to south cell
      this->h_cooMatPushBack(1., sfc_idx, sn_idx);
      // Add matrix element associated to north cell
      this->h_cooMatPushBack(1., sfc_idx, nn_idx);
    }

    // Add matrix elements associated to contributions from east/west cells on western boundary of block (excl. corners)
    for(int iy=1; iy<BSY-1; iy++)
    {
      const int ix = 0;
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int en_idx = sfc_idx+1; // east neighbour

      // Add matrix element associated to east cell
      this->h_cooMatPushBack(1., sfc_idx, en_idx);

      if (isWestBoundary){
        this->h_cooMatPushBack(-3., sfc_idx, sfc_idx);
      }
      else{
        this->h_cooMatPushBack(-4., sfc_idx, sfc_idx);
        if (sim.tmp->Tree(rhsNei_west).Exists())
        {
          //then west neighbor exists and we can safely use rhsNei_west and access the gridpoint-data etc.
          const size_t i_west = rhsNei_west.blockID;
          const int wn_idx = i_west*BSX*BSY+iy*BSX+(BSX-1); // eastmost cell of western block
          this->h_cooMatPushBack(1., sfc_idx, wn_idx);
        }
        else { throw; }
      }
    }

    // Add matrix elements associated to contributions from east/west cells on eastern boundary of block (excl. corners)
    for(int iy=1; iy<BSY-1; iy++)
    {
      const int ix = BSX-1;
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int wn_idx = sfc_idx-1; // west neighbour

      // Add matrix element associated to west cell
      this->h_cooMatPushBack(1., sfc_idx, wn_idx);

      if (isEastBoundary){
        this->h_cooMatPushBack(-3., sfc_idx, sfc_idx);
      }
      else{
        this->h_cooMatPushBack(-4., sfc_idx, sfc_idx);
        if (sim.tmp->Tree(rhsNei_east).Exists())
        {
          //then west neighbor exists and we can safely use rhsNei_west and access the gridpoint-data etc.
          const size_t i_east = rhsNei_east.blockID;
          const int en_idx = i_east*BSX*BSY+iy*BSX+0; // westmost cell of eastern block
          this->h_cooMatPushBack(1., sfc_idx, en_idx);
        }
        else { throw; }
      }
    }

    // Add matrix elements associated to contributions from west/east on southern/northern boundary of block (with corners)
    for(int ix=0; ix<BSX; ix++)
    for(int iy=0; iy<BSY; iy+=(BSY-1))
    { // The inner loop executes on iy = [0, BSX-1] (southern/northern boundary), at ix = [0, BSX-1] interaction
      // with western/eastern block takes place
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;

      // Add matrix element associated to west cell
      if (ix > 0){
        const int wn_idx = sfc_idx-1; // west neighbour
        this->h_cooMatPushBack(1., sfc_idx, wn_idx);
      }
      else{
        if (!isWestBoundary){
          if (sim.tmp->Tree(rhsNei_west).Exists()){
            const size_t i_west = rhsNei_west.blockID;
            const int wn_idx = i_west*BSX*BSY+iy*BSX+(BSX-1); // eastmost cell of western block
            this->h_cooMatPushBack(1., sfc_idx, wn_idx);
          }
          else{ throw; }
        }
      }
      // Add matrix element associated to east cell
      if (ix < BSX - 1){
        const int en_idx = sfc_idx+1; // east neighbour
        this->h_cooMatPushBack(1., sfc_idx, en_idx);
      }
      else {
        if (!isEastBoundary){
          if (sim.tmp->Tree(rhsNei_east).Exists()){
            const size_t i_east = rhsNei_east.blockID;
            const int en_idx = i_east*BSX*BSY+iy*BSX+0; // westmost cell of eastern block
            this->h_cooMatPushBack(1., sfc_idx, en_idx);
          }
          else { throw; }
        }
      }
    }

    // Add matrix elements associated to contributions from north/south cells on southern boundary of block (with corners)
    for(int ix=0; ix<BSX;ix++)
    {
      const int iy = BSY-1;
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int nn_idx = i*BSX*BSY+(iy+1)*BSX+ix; // north neighbour

      // Add matrix element associated to north cell
      this->h_cooMatPushBack(1., sfc_idx, nn_idx);

      if(isSouthBoundary){
        if(isWestBoundary || isEastBoundary)
        { // Two boundary conditions to consider for diagonal element
          this->h_cooMatPushBack(-2., sfc_idx, sfc_idx);
        }
        else
        { // One boundary condition to consider for diagonal element
          this->h_cooMatPushBack(-3., sfc_idx, sfc_idx);
        }
      }
      else{
        this->h_cooMatPushBack(-4., sfc_idx, sfc_idx);
        if (sim.tmp->Tree(rhsNei_south).Exists())
        {
          //then west neighbor exists and we can safely use rhsNei_west and access the gridpoint-data etc.
          const size_t i_south = rhsNei_south.blockID;
          const int sn_idx = i_south*BSX*BSY+(BSY-1)*BSX+ix; // northmost cell of southern block
          this->h_cooMatPushBack(1., sfc_idx, sn_idx);
        }
        else { throw; }
      }
    }

    // Add matrix elements associated to contributions from north/south cells on northern boundary of block (with corners)
    for(int ix=0; ix<BSX;ix++)
    {
      const int iy = BSY-1;
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int sn_idx = i*BSX*BSY+(iy-1)*BSX+ix; // south neighbour

      // Add matrix element associated to south cell
      this->h_cooMatPushBack(1., sfc_idx, sn_idx);

      if (isNorthBoundary){
        if (isWestBoundary || isEastBoundary)
        { // Two boundary conditions to consider for diagonal element
          this->h_cooMatPushBack(-2., sfc_idx, sfc_idx);
        }
        else
        { // One boundary condition to consider for diagonal element
          this->h_cooMatPushBack(-3., sfc_idx, sfc_idx);
        }
      }
      else{
        this->h_cooMatPushBack(-4., sfc_idx, sfc_idx);
        if (sim.tmp->Tree(rhsNei_north).Exists())
        {
          //then west neighbor exists and we can safely use rhsNei_north and access the gridpoint-data etc.
          const size_t i_north = rhsNei_south.blockID;
          const int sn_idx = i_north*BSX*BSY+0*BSX+ix; // southmost cell belonging to north block
          this->h_cooMatPushBack(1., sfc_idx, sn_idx);
        }
        else { throw; }
      }
    }
  }

  // Save params of current linear system
  m_ = N; // rows
  n_ = N; // cols
  nnz_ = h_valA.size(); // non-zero elements

  sim.stopProfiler();
}

void cudaAMRSolver::linsysMemcpyHostToDev(){
  // Host-device exec asynchronous, it may be worth already allocating pinned memory
  // and copying h2h (with cpu code) after async dev memory allocation calls 
  // to speed up h2d transfer down the line
   
  // Allocate device memory for linear system
  checkCudaErrors(cudaMallocAsync(&d_valA, nnz_ * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_cooRowA, nnz_ * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_cooColA, nnz_ * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_x, m_ * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_b, m_ * sizeof(double), solver_stream));

  // Possibly copy to pinned memory here followed by a sync call

  // H2D transfer of linear system
  checkCudaErrors(cudaMemcpyAsync(d_valA, h_valA.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_cooRowA, h_cooRowA.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_cooColA, h_cooColA.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_b, h_b.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  
  // Sort COO storage by row
  // size_t pBufferSizeInBytes;
  // checkCudaErrors(cusparseXcoosort_bufferSizeExt(
  //       cusparse_handle, m_, n_, nnz_, d_cooRowA, d_cooColA, &pBufferSizeInBytes));


}

void cudaAMRSolver::BiCGSTAB(){

}

void cudaAMRSolver::linsysMemcpyDevToHost(){

  // D2H transfer of results to pageable host memory.  This call is blocking, hence, 
  // no need to synchronize and memory deallocation can happen in backgoung
  checkCudaErrors(cudaMemcpyAsync(h_x.data(), d_x, m_ * sizeof(double), cudaMemcpyDeviceToHost, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_valA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_cooRowA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_cooColA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_x, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_b, solver_stream));

  // TODO: write from h_x to
}

void cudaAMRSolver::solve()
{

  cudaDeviceSynchronize();
  std::cout << "--------------------- Calling on cudaAMRSolver.solve() ------------------------ \n";

  this->unifLinsysPrepHost();
  this->linsysMemcpyHostToDev();
  this->BiCGSTAB();
  this->linsysMemcpyDevToHost();
}
