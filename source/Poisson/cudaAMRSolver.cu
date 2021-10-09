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
  checkCudaErrors(cudaStreamCreate(&solver_stream_));
  checkCudaErrors(cublasCreate(&cublas_handle_)); 
  checkCudaErrors(cusparseCreate(&cusparse_handle_)); 
  // Set handles to stream
  checkCudaErrors(cublasSetStream(cublas_handle_, solver_stream_));
  checkCudaErrors(cusparseSetStream(cusparse_handle_, solver_stream_));
}

cudaAMRSolver::~cudaAMRSolver()
{
  // Destroy CUDA stream and library handles
  checkCudaErrors(cublasDestroy(cublas_handle_)); 
  checkCudaErrors(cusparseDestroy(cusparse_handle_)); 
  checkCudaErrors(cudaStreamDestroy(solver_stream_));
  
}

void inline cudaAMRSolver::h_cooMatPushBack(
    const double &val, 
    const int &row, 
    const int &col){
  this->h_cooValA_.push_back(val);
  this->h_cooRowA_.push_back(row);
  this->h_cooColA_.push_back(col);
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
  std::vector<cubism::BlockInfo>&  pInfo = sim.pres->getBlocksInfo();
  const size_t Nblocks = RhsInfo.size();
  const size_t N = BSX*BSY*Nblocks;

  // Allocate memory for solution 'x' and RHS vector 'b' on host
  this->h_x_.resize(N);
  this->h_b_.resize(N);
  // Clear contents from previous call of cudaAMRSolver::solve() and reserve memory 
  // for sparse LHS matrix 'A' (for uniform grid at most 5 elements per row).
  this->h_cooValA_.clear();
  this->h_cooRowA_.clear();
  this->h_cooColA_.clear();
  this->h_cooValA_.reserve(5 * N);
  this->h_cooRowA_.reserve(5 * N);
  this->h_cooColA_.reserve(5 * N);

  // No 'parallel for' to avoid accidental reorderings of COO elements during push_back
  for(size_t i=0; i< Nblocks; i++)
  {    
    BlockInfo &rhs_info = RhsInfo[i];
    ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) RhsInfo[i].ptrBlock;
    ScalarBlock & __restrict__ p  = *(ScalarBlock*) pInfo[i].ptrBlock;

    // Construct RHS and x_0 vectors for linear system
    for(int iy=1; iy<BSY-1; iy++)
    for(int ix=1; ix<BSX-1; ix++)
    {
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      d_b_[sfc_idx] = rhs(ix,iy).s;
      h_x_[sfc_idx] = p(ix,iy).s;
    }

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
  nnz_ = this->h_cooValA_.size(); // non-zero elements

  sim.stopProfiler();
}

void cudaAMRSolver::linsysMemcpyHostToDev(){
  // Host-device exec asynchronous, it may be worth already allocating pinned memory
  // and copying h2h (with cpu code) after async dev memory allocation calls 
  // to speed up h2d transfer down the line
   
  // Allocate device memory for linear system
  checkCudaErrors(cudaMallocAsync(&d_cooValA_, nnz_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMallocAsync(&d_cooValA_sorted_, nnz_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMallocAsync(&d_cooRowA_, nnz_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMallocAsync(&d_cooColA_, nnz_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMallocAsync(&d_x_, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMallocAsync(&d_b_, m_ * sizeof(double), solver_stream_));

  // Possibly copy to pinned memory here followed by a sync call

  // H2D transfer of linear system
  checkCudaErrors(cudaMemcpyAsync(d_cooValA_, h_cooValA_.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_cooRowA_, h_cooRowA_.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_cooColA_, h_cooColA_.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_x_, h_x_.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_b_, h_b_.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  
  // Sort COO storage by row
  // 1. Deduce buffer size necessary for sorting and allocate storage for it
  size_t pBufferSz;
  void* pBuffer;
  checkCudaErrors(cusparseXcoosort_bufferSizeExt(cusparse_handle_, m_, n_, nnz_, d_cooRowA_, d_cooColA_, &pBufferSz));
  checkCudaErrors(cudaMallocAsync(&pBuffer, pBufferSz * sizeof(char), solver_stream_));

  // 2. Set-up permutation vector P to track transformation from un-sorted to sorted list
  int* d_P;
  checkCudaErrors(cudaMallocAsync(&d_P, nnz_ * sizeof(int), solver_stream_));
  checkCudaErrors(cusparseCreateIdentityPermutation(cusparse_handle_, nnz_, d_P));

  // 3. Sort d_cooRowA_ and d_cooCol inplace and apply permutation stored in d_P to d_cooValA_
  checkCudaErrors(cusparseXcoosortByRow(cusparse_handle_, m_, n_, nnz_, d_cooRowA_, d_cooColA_, d_P, pBuffer));
  checkCudaErrors(cusparseDgthr(cusparse_handle_, nnz_, d_cooValA_, d_cooValA_sorted_, d_P, CUSPARSE_INDEX_BASE_ZERO));

  // Free buffers allocated for COO sort
  checkCudaErrors(cudaFreeAsync(pBuffer, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_P, solver_stream_));

}

void cudaAMRSolver::BiCGSTAB()
{
  const double eye = 1.;
  const double nye = -1.;
  const double nil = 0.;

  /*
    This function generally follows notation of the Wikipedia page with several omissions
    to increase variable reuse.  Specifically:
      - d_x_ <-> h, x_i
      - d_b_ <-> r_0, r_i, s
  */

  // Initialize BiCGSTAB scalar parameters
  const double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol * sim.uMax_measured / sim.dt;
  const double max_rel_error = sim.step < 10 ? 0.0 : min(1e-2,sim.PoissonTolRel * sim.uMax_measured / sim.dt );

  // Initialize BiCGSTAB arrays
  double* d_rhat = NULL;
  double* d_p = NULL;
  double* d_nu = NULL;
  double* d_t = NULL;
  // Allocate device memory to arrays
  checkCudaErrors(cudaMallocAsync(&d_rhat, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMallocAsync(&d_p, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMallocAsync(&d_nu, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMallocAsync(&d_t, m_ * sizeof(double), solver_stream_));

  // Initialize variables to evaluate convergence
  double x_diff_norm = 1e50;
  double* d_x_diff = NULL;
  checkCudaErrors(cudaMallocAsync(&d_x_diff, m_ * sizeof(double), solver_stream_));

  // Create descriptors for variables that will pass through cuSPARSE
  cusparseSpMatDescr_t spDescrA;
  cusparseDnVecDescr_t spDescrB;
  cusparseDnVecDescr_t spDescrX0;
  cusparseDnVecDescr_t spDescrP;
  cusparseDnVecDescr_t spDescrNu;
  cusparseDnVecDescr_t spDescrT;
  checkCudaErrors(cusparseCreateCoo(&spDescrA, m_, n_, nnz_, d_cooRowA_, d_cooColA_, d_cooValA_sorted_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrB, m_, d_b_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrX0, m_, d_x_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrP, m_, d_p, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrNu, m_, d_nu, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrT, m_, d_t, CUDA_R_64F));

  // Allocate work buffer for cusparseSpMV
  size_t pBufferSz;
  void* pBuffer;
  checkCudaErrors(cusparseSpMV_bufferSize(
        cusparse_handle_, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &eye, 
        spDescrA, 
        spDescrX0, 
        &nil, 
        spDescrNu, 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        &pBufferSz));
  checkCudaErrors(cudaMallocAsync(&pBuffer, pBufferSz * sizeof(char), solver_stream_));

  // 1. r <- b - A*x_0.  Add bias with cuBLAS like in "NVIDIA_CUDA-11.4_Samples/7_CUDALibraries/conjugateGradient"
  checkCudaErrors(cusparseSpMV( // A*x_0
        cusparse_handle_, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &eye, 
        spDescrA, 
        spDescrX0, 
        &nil, 
        spDescrNu, // Use d_nu as temporary storage for result A*x_0 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        pBuffer)); 
  checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nye, d_nu, 1, d_b_, 1)); // r <- -A*x_0 + b

  // 2. Set r_hat = r
  checkCudaErrors(cublasDcopy(cublas_handle_, m_, d_b_, 1, d_rhat, 1));

  // 3. Set initial values to scalars
  double rho_curr = 1.;
  double rho_prev = 1.;
  double alpha = 1.;
  double omega = 1.;
  double beta = 1.;

  // 4. Set initial values of vectors to zero
  checkCudaErrors(cudaMemsetAsync(d_nu, 0, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMemsetAsync(d_p, 0, m_ * sizeof(double), solver_stream_));

  // 5. Start iterations
  for(size_t k(0); k<1000; k++)
  {
    // 1. rho_i = (r_hat, r)
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat, 1, d_b_, 1, &rho_curr));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); // sync for 2. which happens on host
    
    // 2. beta = (rho_i / rho_{i-1}) * (alpha / omega_{i-1})
    beta = (rho_curr / rho_prev) * (alpha / omega);

    // 3. p_i = r_{i-1} + beta(p_{i-1} - omega_{i-1}*nu_i)
    double nomega = -omega;
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nomega, d_nu, 1, d_p, 1)); // p <- -omega_{i-1}*nu_i + p
    checkCudaErrors(cublasDscal(cublas_handle_, m_, &beta, d_p, 1));            // p <- beta * p
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &eye, d_b_, 1, d_p, 1));    // p <- r_{i-1} + p

    // 4. nu_i = A * p_i 
    checkCudaErrors(cusparseSpMV(
          cusparse_handle_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &eye,
          spDescrA,
          spDescrP,
          &nil,
          spDescrNu,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          pBuffer));

    // 5. alpha = rho_i / (r_hat, nu_i)
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat, 1, d_nu, 1, &alpha)); // alpha <- (r_hat, nu_i)
    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); // sync for host division
    alpha = rho_curr / alpha; // alpha <- rho_i / alpha

    // 6. h = alpha*p_i + x_{i-1}
    checkCudaErrors(cublasDcopy(cublas_handle_, m_, d_x_, 1, d_x_diff, 1)); // copy previous value for future norm calculation
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &alpha, d_p, 1, d_x_, 1));

    // 7. If h accurate enough then quit
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nye, d_x_, 1, d_x_diff, 1));
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_x_diff, 1, &x_diff_norm));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));

    if(x_diff_norm < max_error)
    {
      std::cout << "  [Poisson solver]: Converged after " << k << " iterations.";
      // bConverged = true;
      break;
    }

    // 8. s = -alpha * nu_i + r_{i-1}
    const double nalpha = -alpha;
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nalpha, d_nu, 1, d_b_, 1));

    // 9. t = A * s
    checkCudaErrors(cusparseSpMV(
          cusparse_handle_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &eye,
          spDescrA,
          spDescrB,
          &nil,
          spDescrT,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          pBuffer));
    
    // 10. omega_i = (t,s)/(t,t), variables alpha & beta no longer in use this iter
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_t, 1, d_b_, 1, &alpha)); // alpha <- (t,s)
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_t, 1, &beta));          // beta <- sqrt(t,t)
    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); // sync for host arithmetic
    omega = alpha / (beta * beta);

    // 11. x_i = omega_i * s + h
    checkCudaErrors(cublasDcopy(cublas_handle_, m_, d_x_, 1, d_x_diff, 1)); // copy previous value for future norm calculation
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &omega, d_b_, 1, d_x_, 1));

    // 12. If x_i accurate enough then quit
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nye, d_x_, 1, d_x_diff, 1));
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_x_diff, 1, &x_diff_norm));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));

    if(x_diff_norm < max_error)
    {
      std::cout << "  [Poisson solver]: Converged after " << k << " iterations.";
      // bConverged = true;
      break;
    }

    // 13. r_i = -omega_i * t + s
    nomega = -nomega;
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nomega, d_t, 1, d_b_, 1));

    // Update *_prev values for next iteration
    rho_prev = rho_curr;
  }

  // Synchronization call
  // Cleanup
  checkCudaErrors(cusparseDestroySpMat(spDescrA));
  checkCudaErrors(cusparseDestroyDnVec(spDescrB));
  checkCudaErrors(cusparseDestroyDnVec(spDescrX0));
  checkCudaErrors(cusparseDestroyDnVec(spDescrP));
  checkCudaErrors(cusparseDestroyDnVec(spDescrNu));
  checkCudaErrors(cusparseDestroyDnVec(spDescrT));
  checkCudaErrors(cudaFreeAsync(d_rhat, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_p, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_nu, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_t, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_x_diff, solver_stream_));

}

void cudaAMRSolver::linsysMemcpyDevToHost(){

  // D2H transfer of results to pageable host memory.  This call is blocking, hence, 
  // no need to synchronize and memory deallocation can happen in backgoung
  checkCudaErrors(cudaMemcpyAsync(h_x_.data(), d_x_, m_ * sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_cooValA_, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_cooRowA_, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_cooColA_, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_x_, solver_stream_));
  checkCudaErrors(cudaFreeAsync(d_b_, solver_stream_));

  // Synchronization call to make sure all GPU stuff in finished
  checkCudaErrors(cudaStreamSynchronize(solver_stream_));

  // Write results back into ScalarBloc for pressure
  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

  std::vector<cubism::BlockInfo>&  pInfo = sim.pres->getBlocksInfo();
  const size_t Nblocks = pInfo.size();

  #pragma omp parallel for
  for(size_t i=0; i<Nblocks; i++)
  {
    ScalarBlock & p    = *(ScalarBlock*)  pInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      p(ix,iy).s = h_x_[i*BSX*BSY+iy*BSX+ix];
    }
  }
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
