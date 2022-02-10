#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#include "BiCGSTAB.cuh"

#define _HALO_MSG_ 200

BiCGSTABSolver::BiCGSTABSolver(
    const int &rank,
    const MPI_Comm &m_comm,
    const int &comm_size,
    const int &BSX, 
    const int &BSY, 
    const double* const P_inv)
  : rank_(rank), m_comm_(m_comm), comm_size_(comm_size), BLEN_(BSX*BSY)
{
  std::cout << "---------------- Calling on BiCGSTABSolver() constructor ------------\n";
  // Set-up CUDA streams and handles
  checkCudaErrors(cudaStreamCreate(&solver_stream_));
  checkCudaErrors(cublasCreate(&cublas_handle_)); 
  checkCudaErrors(cusparseCreate(&cusparse_handle_)); 
  // Set handles to stream
  checkCudaErrors(cublasSetStream(cublas_handle_, solver_stream_));
  checkCudaErrors(cusparseSetStream(cusparse_handle_, solver_stream_));
  // Set pointer modes to device
  checkCudaErrors(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
  checkCudaErrors(cusparseSetPointerMode(cusparse_handle_, CUSPARSE_POINTER_MODE_DEVICE));

  // Set constants and allocate memory for scalars
  double h_consts[3] = {1., -1., 0.};
  checkCudaErrors(cudaMalloc(&d_consts_, 3 * sizeof(double)));
  checkCudaErrors(cudaMemcpyAsync(d_consts_, h_consts, 3 * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  d_eye_ = d_consts_;
  d_nye_ = d_consts_ + 1;
  d_nil_ = d_consts_ + 2;
  checkCudaErrors(cudaMalloc(&d_coeffs_, sizeof(BiCGSTABScalars)));
  checkCudaErrors(cudaMallocHost(&h_coeffs_, sizeof(BiCGSTABScalars)));

  // Copy preconditionner
  checkCudaErrors(cudaMalloc(&d_P_inv_, BLEN_ * BLEN_ * sizeof(double)));
  checkCudaErrors(cudaMemcpyAsync(d_P_inv_, P_inv, BLEN_ * BLEN_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
}

BiCGSTABSolver::~BiCGSTABSolver()
{
  std::cout << "---------------- Calling on BiCGSTABSolver() destructor ------------\n";
  // Cleanup after last timestep
  checkCudaErrors(cudaFree(dloc_cooValA_));
  checkCudaErrors(cudaFree(dloc_cooRowA_));
  checkCudaErrors(cudaFree(dloc_cooColA_));
  checkCudaErrors(cudaFree(dbd_cooValA_));
  checkCudaErrors(cudaFree(dbd_cooRowA_));
  checkCudaErrors(cudaFree(dbd_cooColA_));
  checkCudaErrors(cudaFree(d_x_));
  checkCudaErrors(cudaFree(d_r_));
  checkCudaErrors(cusparseDestroySpMat(spDescrLocA_));
  checkCudaErrors(cusparseDestroySpMat(spDescrBdA_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrZ_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrNu_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrT_));
  checkCudaErrors(cudaFree(d_rhat_));
  checkCudaErrors(cudaFree(d_p_));
  checkCudaErrors(cudaFree(d_nu_hd_));
  checkCudaErrors(cudaFree(d_t_hd_));
  checkCudaErrors(cudaFree(d_z_hd_));
  checkCudaErrors(cudaFree(locSpMVBuff_));
  checkCudaErrors(cudaFree(bdSpMVBuff_));

#ifdef BICGSTAB_PROFILER
  std::cout << "  [BiCGSTAB]: total elapsed time: " << pGlob_.elapsed() << " [ms]." << std::endl;
  std::cout << "  [BiCGSTAB]: memory transfers:   " << (pMemcpy_.elapsed()/pGlob_.elapsed())*100. << "%." << std::endl;
  std::cout << "  [BiCGSTAB]: preconditioning:    " << (pPrec_.elapsed()/pGlob_.elapsed())*100. << "%." << std::endl;
  std::cout << "  [BiCGSTAB]: SpVM:               " << (pSpMV_.elapsed()/pGlob_.elapsed())*100. << "%." << std::endl;
#endif

  // Free preconditionner
  checkCudaErrors(cudaFree(d_P_inv_));

  // Free device consants
  checkCudaErrors(cudaFree(d_consts_));
  checkCudaErrors(cudaFree(d_coeffs_));
  checkCudaErrors(cudaFreeHost(h_coeffs_));

  // Destroy CUDA streams and handles
  checkCudaErrors(cublasDestroy(cublas_handle_)); 
  checkCudaErrors(cusparseDestroy(cusparse_handle_)); 
  checkCudaErrors(cudaStreamDestroy(solver_stream_));
}

// --------------------------------- public class methods ------------------------------------

void BiCGSTABSolver::solveWithUpdate(
    std::shared_ptr<LocalSpMatDnVec> LocalLS,
    const double max_error,
    const double max_rel_error,
    const int max_restarts)
{

  this->updateAll(LocalLS);
  this->main(LocalLS->x_.data(), max_error, max_rel_error, max_restarts);
}

void BiCGSTABSolver::solveNoUpdate(
    std::shared_ptr<LocalSpMatDnVec> LocalLS,
    const double max_error,
    const double max_rel_error,
    const int max_restarts)
{
  this->updateVec(LocalLS);
  this->main(LocalLS->x_.data(), max_error, max_rel_error, max_restarts);
}

// --------------------------------- private class methods ------------------------------------

void BiCGSTABSolver::updateAll(std::shared_ptr<LocalSpMatDnVec> LocalLS)
{
  // Update LS metadata
  m_ = LocalLS->m_;
  loc_nnz_ = LocalLS->loc_nnz_ ;
  bd_nnz_ = LocalLS->bd_nnz_ ;
  lower_halo_ = LocalLS->lower_halo_ ;
  upper_halo_ = LocalLS->upper_halo_ ;
  hd_m_ = lower_halo_ + m_ + upper_halo_;


  if (dirty_) // Previous time-step exists so cleanup first
  {
    // Free device memory allocated for linear system from previous time-step
    checkCudaErrors(cudaFree(dloc_cooValA_));
    checkCudaErrors(cudaFree(dloc_cooRowA_));
    checkCudaErrors(cudaFree(dloc_cooColA_));
    // Note that memory for LHS & RHS vectors also reallocated here
    checkCudaErrors(cudaFree(d_x_)); 
    checkCudaErrors(cudaFree(d_r_));
    // Cleanup memory allocated for BiCGSTAB arrays
    checkCudaErrors(cusparseDestroySpMat(spDescrLocA_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrZ_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrNu_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrT_));
    checkCudaErrors(cudaFree(d_rhat_));
    checkCudaErrors(cudaFree(d_p_));
    checkCudaErrors(cudaFree(d_nu_hd_));
    checkCudaErrors(cudaFree(d_t_hd_));
    checkCudaErrors(cudaFree(d_z_hd_));
    checkCudaErrors(cudaFree(locSpMVBuff_));
    if (comm_size_ > 1)
    {
      checkCudaErrors(cudaFree(d_send_buff_pack_idx_));
      checkCudaErrors(cudaFree(d_send_buff_));
      checkCudaErrors(cudaFree(dbd_cooValA_));
      checkCudaErrors(cudaFree(dbd_cooRowA_));
      checkCudaErrors(cudaFree(dbd_cooColA_));
      checkCudaErrors(cusparseDestroySpMat(spDescrBdA_));
      checkCudaErrors(cudaFree(bdSpMVBuff_));
    }
  }
  dirty_ = true;
  
  send_buff_sz_ = LocalLS->send_buff_pack_idx_.size();
  // Allocate device memory for metadata and linear system
  if (comm_size_ > 1)
  {
    checkCudaErrors(cudaMalloc(&d_send_buff_pack_idx_, send_buff_sz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_send_buff_, send_buff_sz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dbd_cooValA_, bd_nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dbd_cooRowA_, bd_nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dbd_cooColA_, bd_nnz_ * sizeof(int)));
  }
  checkCudaErrors(cudaMalloc(&dloc_cooValA_, loc_nnz_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&dloc_cooRowA_, loc_nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&dloc_cooColA_, loc_nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_x_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_r_, m_ * sizeof(double)));

#ifdef BICGSTAB_PROFILER
  pMemcpy_.startProfiler(solver_stream_);
#endif
  // H2D transfer of linear system
  checkCudaErrors(cudaMemcpyAsync(dloc_cooValA_, LocalLS->loc_cooValA_.data(), loc_nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(dloc_cooRowA_, LocalLS->loc_cooRowA_int_.data(), loc_nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(dloc_cooColA_, LocalLS->loc_cooColA_int_.data(), loc_nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
  if (comm_size_ > 1)
  {
    checkCudaErrors(cudaMemcpyAsync(d_send_buff_pack_idx_, LocalLS->send_buff_pack_idx_.data(), send_buff_sz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(dbd_cooValA_, LocalLS->bd_cooValA_.data(), bd_nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(dbd_cooRowA_, LocalLS->bd_cooRowA_int_.data(), bd_nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(dbd_cooColA_, LocalLS->bd_cooColA_int_.data(), bd_nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
  }
#ifdef BICGSTAB_PROFILER
  pMemcpy_.stopProfiler(solver_stream_);
#endif

  // Copy host-side vectors during H2D transfer
  recv_ranks_ = LocalLS->recv_ranks_;
  recv_offset_ = LocalLS->recv_offset_;
  recv_sz_ = LocalLS->recv_sz_;
  send_ranks_ = LocalLS->send_ranks_;
  send_offset_ = LocalLS->send_offset_;
  send_sz_ = LocalLS->send_sz_;

  // Allocate arrays for BiCGSTAB storage
  checkCudaErrors(cudaMalloc(&d_rhat_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_p_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_nu_hd_, hd_m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_t_hd_,  hd_m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_z_hd_,  hd_m_ * sizeof(double)));
  // Ignore halo for local operations
  d_nu_ = &d_nu_hd_[lower_halo_];
  d_t_ = &d_t_hd_[lower_halo_];
  d_z_ = &d_z_hd_[lower_halo_];

  // Create descriptors for variables that will pass through cuSPARSE
  checkCudaErrors(cusparseCreateCoo(&spDescrLocA_, hd_m_, hd_m_, loc_nnz_, dloc_cooRowA_, dloc_cooColA_, dloc_cooValA_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrZ_, hd_m_, d_z_hd_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrNu_, hd_m_, d_nu_hd_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrT_, hd_m_, d_t_hd_, CUDA_R_64F));
  // Allocate work buffer for cusparseSpMV
  checkCudaErrors(cusparseSpMV_bufferSize(
        cusparse_handle_, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        d_eye_, 
        spDescrLocA_, 
        spDescrZ_, 
        d_nil_, 
        spDescrNu_, 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        &locSpMVBuffSz_));
  checkCudaErrors(cudaMalloc(&locSpMVBuff_, locSpMVBuffSz_ * sizeof(char)));
  if (comm_size_ > 1)
  {
    checkCudaErrors(cusparseCreateCoo(&spDescrBdA_, hd_m_, hd_m_, bd_nnz_, dbd_cooRowA_, dbd_cooColA_, dbd_cooValA_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    checkCudaErrors(cusparseSpMV_bufferSize(
          cusparse_handle_, 
          CUSPARSE_OPERATION_NON_TRANSPOSE, 
          d_eye_, 
          spDescrBdA_, 
          spDescrZ_, 
          d_nil_, 
          spDescrNu_, 
          CUDA_R_64F, 
          CUSPARSE_MV_ALG_DEFAULT, 
          &bdSpMVBuffSz_));
    checkCudaErrors(cudaMalloc(&bdSpMVBuff_, bdSpMVBuffSz_ * sizeof(char)));
  }

  this->updateVec(LocalLS);
}

void BiCGSTABSolver::updateVec(std::shared_ptr<LocalSpMatDnVec> LocalLS)
{
#ifdef BICGSTAB_PROFILER
  pMemcpy_.startProfiler(solver_stream_);
#endif
  // Copy RHS LHS vec initial guess, if LS was updated, updateAll reallocates sufficient memory
  // d_z_hd_ with offset to store x0 for A*x0
  checkCudaErrors(cudaMemcpyAsync(&d_z_hd_[lower_halo_], LocalLS->x_.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_r_, LocalLS->b_.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
#ifdef BICGSTAB_PROFILER
  pMemcpy_.stopProfiler(solver_stream_);
#endif
}

__global__ void set_squared(double* const val)
{
  val[0] *= val[0];
}

__global__ void set_amax(double* const dest, const int* const idx, const double* const source)
{
  // 1-based indexing in cublas API
  dest[0] = fabs(source[idx[0]-1]);
}

__global__ void set_negative(double* const dest, double* const source)
{
  dest[0] = -source[0];
}

__global__ void breakdown_update(BiCGSTABScalars* coeffs)
{
  coeffs->rho_prev = 1.;
  coeffs->alpha = 1.;
  coeffs->omega = 1.;
  coeffs->beta = (coeffs->rho_curr / (coeffs->rho_prev + coeffs->eps)) * (coeffs->alpha / (coeffs->omega + coeffs->eps));
}

__global__ void set_beta(BiCGSTABScalars* coeffs)
{
  coeffs->beta = (coeffs->rho_curr / (coeffs->rho_prev + coeffs->eps)) * (coeffs->alpha / (coeffs->omega + coeffs->eps));
}

__global__ void set_alpha(BiCGSTABScalars* coeffs)
{
  coeffs->alpha = coeffs->rho_curr / (coeffs->buff_1 + coeffs->eps);
}

__global__ void set_omega(BiCGSTABScalars* coeffs)
{
  coeffs->omega = coeffs->buff_1 / (coeffs->buff_2 + coeffs->eps);
}

__global__ void set_rho(BiCGSTABScalars* coeffs)
{
  coeffs->rho_prev = coeffs->rho_curr;
}

__global__ void send_buff_pack(
    int buff_sz, 
    const int* const pack_idx, 
    double* const buff, 
    const double* const source)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < buff_sz; i += blockDim.x * gridDim.x)
    buff[i] = source[pack_idx[i]];
}

void BiCGSTABSolver::hd_cusparseSpMV(
  double* d_op_hd,  // operand vec
  cusparseDnVecDescr_t spDescrOp,
  double* d_res_hd, // result vec
  cusparseDnVecDescr_t spDescrRes)
{

//  if (comm_size_ > 1)
//  {
//    send_buff_pack<<<send_buff_sz_/32+1,32, 0, solver_stream_>>>(send_buff_sz_, d_send_buff_pack_idx_, d_send_buff_, d_op_hd);
//    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); // try with events later
//
//    for (size_t i(0); i < send_ranks_.size(); i++)
//    {
//      MPI_Request request;
//      MPI_Isend(&d_send_buff_[send_offset_[i]], send_sz_[i], MPI_INT, send_ranks_[i], _HALO_MSG_, m_comm_, &request);
//    }
//  }

#ifdef BICGSTAB_PROFILER
  pSpMV_.startProfiler(solver_stream_);
#endif
  // A*x for local rows
  checkCudaErrors(cusparseSpMV( 
        cusparse_handle_, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        d_eye_, 
        spDescrLocA_, 
        spDescrOp, 
        d_nil_, 
        spDescrRes, 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        locSpMVBuff_)); 
#ifdef BICGSTAB_PROFILER
  pSpMV_.stopProfiler(solver_stream_);
#endif

  if (comm_size_ > 1)
  {
//    // Schedule receives and wait for them to arrive
//    std::vector<MPI_Request> recv_requests;
//    for (size_t i(0); i < recv_ranks_.size(); i++)
//      MPI_Irecv(&d_op_hd[recv_offset_[i]], recv_sz_[i], MPI_INT, recv_ranks_[i], _HALO_MSG_, m_comm_, &recv_requests[i]);
//    for (size_t i(0); i < recv_ranks_.size(); i++)
//      MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);

#ifdef BICGSTAB_PROFILER
    pSpMV_.startProfiler(solver_stream_);
#endif
    // A*x for rows with halo elements, axpy with local results
    checkCudaErrors(cusparseSpMV( 
          cusparse_handle_, 
          CUSPARSE_OPERATION_NON_TRANSPOSE, 
          d_eye_, 
          spDescrBdA_, 
          spDescrOp, 
          d_eye_, 
          spDescrRes, 
          CUDA_R_64F, 
          CUSPARSE_MV_ALG_DEFAULT, 
          bdSpMVBuff_)); 
#ifdef BICGSTAB_PROFILER
    pSpMV_.stopProfiler(solver_stream_);
#endif
  }
}

// export MPICH_RDMA_ENABLED_CUDA=1
// 
void BiCGSTABSolver::main(
    double* const h_x,
    const double max_error, 
    const double max_rel_error, 
    const int max_restarts)
{
#ifdef BICGSTAB_PROFILER
  pGlob_.startProfiler(solver_stream_);
#endif 

  // Initialize variables to evaluate convergence
  double error = 1e50;
  double error_init = 1e50;
  bool bConverged = false;
  int restarts = 0;

  // 3. Set initial values to scalars
  *h_coeffs_ = {1., 1., 0., 1e-21, 1., 1., 0., 0., 0};
  checkCudaErrors(cudaMemcpyAsync(d_coeffs_, h_coeffs_, sizeof(BiCGSTABScalars), cudaMemcpyHostToDevice, solver_stream_));

  // 1. r <- b - A*x_0.  Add bias with cuBLAS like in "NVIDIA_CUDA-11.4_Samples/7_CUDALibraries/conjugateGradient"
	hd_cusparseSpMV(d_z_hd_, spDescrZ_, d_nu_hd_, spDescrNu_);
  checkCudaErrors(cublasDaxpy(cublas_handle_, m_, d_nye_, d_nu_, 1, d_r_, 1)); // r <- -A*x_0 + b

  // ||A*x_0||_max
  checkCudaErrors(cublasIdamax(cublas_handle_, m_, d_nu_, 1, &(d_coeffs_->amax_idx)));
  set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1), &(d_coeffs_->amax_idx), d_nu_);
  checkCudaErrors(cudaGetLastError());
  // ||b - A*x_0||_max 
  checkCudaErrors(cublasIdamax(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->amax_idx)));
  set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_2), &(d_coeffs_->amax_idx), d_r_);
  checkCudaErrors(cudaGetLastError());
  // buff_1 and buff_2 in contigious memory in BiCGSTABScalars
  checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1), 2*sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
  checkCudaErrors(cudaStreamSynchronize(solver_stream_));

//  MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 2, MPI_DOUBLE, MPI_MAX, m_comm_);

  std::cout << "  [BiCGSTAB]: || A*x_0 || = " << h_coeffs_->buff_1 << std::endl;
  std::cout << "  [BiCGSTAB]: Initial norm: " << h_coeffs_->buff_2 << std::endl;
  // Set initial error
  error = h_coeffs_->buff_2;
  error_init = h_coeffs_->buff_2;

  // 2. Set r_hat = r
  checkCudaErrors(cublasDcopy(cublas_handle_, m_, d_r_, 1, d_rhat_, 1));

  // 4. Set initial values of vectors to zero
  checkCudaErrors(cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_));

  // 5. Start iterations
  const size_t max_iter = 1000;
  for(size_t k(0); k<max_iter; k++)
  {
    // 1. rho_i = (r_hat, r)
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_r_, 1, &(d_coeffs_->rho_curr)));
    
    // Numerical convergence trick
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->buff_1)));
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->buff_2)));
    checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->rho_curr), &(d_coeffs_->rho_curr), 3 * sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); 
    h_coeffs_->buff_1 *= h_coeffs_->buff_1; // get square norm
    h_coeffs_->buff_2 *= h_coeffs_->buff_2;
 //   MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->rho_curr), 3, MPI_DOUBLE, MPI_SUM, m_comm_);
    checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->rho_curr), &(h_coeffs_->rho_curr), sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
    const bool serious_breakdown = h_coeffs_->rho_curr * h_coeffs_->rho_curr < 1e-16 * h_coeffs_->buff_1 * h_coeffs_->buff_2;

    // 2. beta = (rho_i / rho_{i-1}) * (alpha / omega_{i-1})
    set_beta<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
    if(serious_breakdown && max_restarts > 0)
    {
      restarts++;
      if(restarts >= max_restarts){
        break;
      }
      std::cout << "  [BiCGSTAB]: Restart at iteration: " << k << " norm: " << error <<" Initial norm: " << error_init << std::endl;
      checkCudaErrors(cublasDcopy(cublas_handle_, m_, d_r_, 1, d_rhat_, 1));
      checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->rho_curr)));
      checkCudaErrors(cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_));
      checkCudaErrors(cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_));
      breakdown_update<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
      checkCudaErrors(cudaGetLastError());
    }

    // 3. p_i = r_{i-1} + beta(p_{i-1} - omega_{i-1}*nu_i)
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1), &(d_coeffs_->omega));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_nu_, 1, d_p_, 1)); // p <- -omega_{i-1}*nu_i + p
    checkCudaErrors(cublasDscal(cublas_handle_, m_, &(d_coeffs_->beta), d_p_, 1));            // p <- beta * p
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, d_eye_, d_r_, 1, d_p_, 1));    // p <- r_{i-1} + p

    // 4. z <- K_2^{-1} * p_i
#ifdef BICGSTAB_PROFILER
    pPrec_.startProfiler(solver_stream_);
#endif
    checkCudaErrors(cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BLEN_, m_ / BLEN_, BLEN_, d_eye_, d_P_inv_, BLEN_, d_p_, BLEN_, d_nil_, d_z_, BLEN_));
#ifdef BICGSTAB_PROFILER
    pPrec_.stopProfiler(solver_stream_);
#endif

    // 5. nu_i = A * z
	  hd_cusparseSpMV(d_z_hd_, spDescrZ_, d_nu_hd_, spDescrNu_);

    // 6. alpha = rho_i / (r_hat, nu_i)
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_nu_, 1, &(d_coeffs_->buff_1)));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
//    MPI_Allreduce(MPI_IN_PLACE, &(d_coeffs_->buff_1), 1, MPI_DOUBLE, MPI_SUM, m_comm_);
    set_alpha<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());

    // 7. h = alpha*z + x_{i-1}
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->alpha), d_z_, 1, d_x_, 1));

    // 9. s = -alpha * nu_i + r_{i-1}
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1), &(d_coeffs_->alpha));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_nu_, 1, d_r_, 1));

    // 10. z <- K_2^{-1} * s
#ifdef BICGSTAB_PROFILER
    pPrec_.startProfiler(solver_stream_);
#endif
    checkCudaErrors(cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BLEN_, m_ / BLEN_, BLEN_, d_eye_, d_P_inv_, BLEN_, d_r_, BLEN_, d_nil_, d_z_, BLEN_));
#ifdef BICGSTAB_PROFILER
    pPrec_.stopProfiler(solver_stream_);
#endif

    // 11. t = A * z
	  hd_cusparseSpMV(d_z_hd_, spDescrZ_, d_nu_hd_, spDescrT_);
    
    // 12. omega_i = (t,s)/(t,t), variables alpha & beta no longer in use this iter
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_t_, 1, d_r_, 1, &(d_coeffs_->buff_1)));
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_t_, 1, &(d_coeffs_->buff_2)));
    set_squared<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_2));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
//    MPI_Allreduce(MPI_IN_PLACE, &(d_coeffs_->buff_1), 2, MPI_DOUBLE, MPI_SUM, m_comm_);
    set_omega<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());

    // 13. x_i = omega_i * z + h
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->omega), d_z_, 1, d_x_, 1));

    // 15. r_i = -omega_i * t + s
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1), &(d_coeffs_->omega));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_t_, 1, d_r_, 1));

    // If x_i accurate enough then quit
    checkCudaErrors(cublasIdamax(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->amax_idx)));
    set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1), &(d_coeffs_->amax_idx), d_r_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpyAsync(&error, &(d_coeffs_->buff_1), sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, m_comm_);

    if((error <= max_error) || (error / error_init <= max_rel_error))
    // if(x_error <= max_error)
    {
      if (rank_ == 0)
      {
        std::cout << "  [BiCGSTAB]: Converged after " << k << " iterations" << std::endl;;
      }
      bConverged = true;
      break;
    }

    // Update *_prev values for next iteration
    set_rho<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
  }

  if (rank_ == 0)
  {
    if( bConverged )
      std::cout <<  "  [BiCGSTAB] Error norm (relative) = " << error << "/" << max_error 
                << " (" << error/error_init  << "/" << max_rel_error << ")" << std::endl;
    else
      std::cout <<  "  [BiCGSTAB]: Iteration " << max_iter 
                << ". Error norm (relative) = " << error << "/" << max_error 
                << " (" << error/error_init  << "/" << max_rel_error << ")" << std::endl;
  }

#ifdef BICGSTAB_PROFILER
  pMemcpy_.startProfiler(solver_stream_);
#endif
  // Copy result back to host
  checkCudaErrors(cudaMemcpyAsync(h_x, d_x_, m_ * sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
#ifdef BICGSTAB_PROFILER
  pMemcpy_.stopProfiler(solver_stream_);
  pGlob_.stopProfiler(solver_stream_);
#endif
}

