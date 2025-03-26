#include <iostream>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "BiCGSTAB.cuh"

BiCGSTABSolver::BiCGSTABSolver(MPI_Comm m_comm, LocalSpMatDnVec &LocalLS,
                               const int BLEN, const bool bMeanConstraint,
                               const std::vector<double> &P_inv)
    : m_comm_(m_comm), BLEN_(BLEN), bMeanConstraint_(bMeanConstraint),
      LocalLS_(LocalLS), prof_(m_comm) {
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);
  checkCudaErrors(cudaStreamCreate(&solver_stream_));
  checkCudaErrors(cudaStreamCreate(&copy_stream_));
  checkCudaErrors(cudaEventCreate(&sync_event_));
  cublasCreate(&cublas_handle_);
  cusparseCreate(&cusparse_handle_);
  cublasSetStream(cublas_handle_, solver_stream_);
  cusparseSetStream(cusparse_handle_, solver_stream_);
  cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);
  cusparseSetPointerMode(cusparse_handle_, CUSPARSE_POINTER_MODE_DEVICE);
  double h_consts[3] = {1., -1., 0.};
  checkCudaErrors(cudaMalloc(&d_consts_, 3 * sizeof(double)));
  checkCudaErrors(cudaMemcpyAsync(d_consts_, h_consts, 3 * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
  d_eye_ = d_consts_;
  d_nye_ = d_consts_ + 1;
  d_nil_ = d_consts_ + 2;
  checkCudaErrors(cudaMalloc(&d_coeffs_, sizeof(BiCGSTABScalars)));
  checkCudaErrors(cudaMallocHost(&h_coeffs_, sizeof(BiCGSTABScalars)));
  checkCudaErrors(cudaMalloc(&d_P_inv_, BLEN_ * BLEN_ * sizeof(double)));
  checkCudaErrors(cudaMemcpyAsync(d_P_inv_, P_inv.data(),
                                  BLEN_ * BLEN_ * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
}

BiCGSTABSolver::~BiCGSTABSolver() {
  this->freeLast();
  prof_.print("Total");
  checkCudaErrors(cudaFree(d_P_inv_));
  checkCudaErrors(cudaFree(d_consts_));
  checkCudaErrors(cudaFree(d_coeffs_));
  checkCudaErrors(cudaFreeHost(h_coeffs_));
  cublasDestroy(cublas_handle_);
  cusparseDestroy(cusparse_handle_);
  checkCudaErrors(cudaEventDestroy(sync_event_));
  checkCudaErrors(cudaStreamDestroy(copy_stream_));
  checkCudaErrors(cudaStreamDestroy(solver_stream_));
}

void BiCGSTABSolver::solveWithUpdate(const double max_error,
                                     const double max_rel_error,
                                     const int max_restarts) {
  this->updateAll();
  this->main(max_error, max_rel_error, max_restarts);
}

void BiCGSTABSolver::solveNoUpdate(const double max_error,
                                   const double max_rel_error,
                                   const int max_restarts) {
  this->updateVec();
  this->main(max_error, max_rel_error, max_restarts);
}
void BiCGSTABSolver::freeLast() {
  if (dirty_) {
    checkCudaErrors(cudaFree(dloc_cooValA_));
    checkCudaErrors(cudaFree(dloc_cooRowA_));
    checkCudaErrors(cudaFree(dloc_cooColA_));
    checkCudaErrors(cudaFree(d_x_));
    checkCudaErrors(cudaFree(d_x_opt_));
    checkCudaErrors(cudaFree(d_r_));
    checkCudaErrors(cudaFree(d_rhat_));
    checkCudaErrors(cudaFree(d_p_));
    checkCudaErrors(cudaFree(d_nu_));
    checkCudaErrors(cudaFree(d_t_));
    checkCudaErrors(cudaFree(d_z_));
    checkCudaErrors(cudaFree(locSpMVBuff_));
    cusparseDestroySpMat(spDescrLocA_);
    cusparseDestroyDnVec(spDescrNu_);
    cusparseDestroyDnVec(spDescrT_);
    cusparseDestroyDnVec(spDescrLocZ_);
    if (comm_size_ > 1) {
      checkCudaErrors(cudaFree(d_send_pack_idx_));
      checkCudaErrors(cudaFree(d_send_buff_));
      checkCudaErrors(cudaFreeHost(h_send_buff_));
      checkCudaErrors(cudaFreeHost(h_recv_buff_));
      checkCudaErrors(cudaFree(dbd_cooValA_));
      checkCudaErrors(cudaFree(dbd_cooRowA_));
      checkCudaErrors(cudaFree(dbd_cooColA_));
      checkCudaErrors(cudaFree(bdSpMVBuff_));
      cusparseDestroySpMat(spDescrBdA_);
      cusparseDestroyDnVec(spDescrBdZ_);
    }
    if (bMeanConstraint_) {
      checkCudaErrors(cudaFree(d_h2_));
      checkCudaErrors(cudaFree(d_red_));
      checkCudaErrors(cudaFree(d_red_res_));
      checkCudaErrors(cudaFree(d_red_temp_storage_));
    }
  }
  dirty_ = true;
}

void BiCGSTABSolver::updateAll() {
  this->freeLast();
  m_ = LocalLS_.m_;
  halo_ = LocalLS_.halo_;
  hd_m_ = m_ + halo_;
  loc_nnz_ = LocalLS_.loc_nnz_;
  bd_nnz_ = LocalLS_.bd_nnz_;
  send_buff_sz_ = LocalLS_.send_pack_idx_.size();
  const int Nblocks = m_ / BLEN_;
  bMeanRow_ = LocalLS_.bMeanRow_;
  checkCudaErrors(cudaMalloc(&dloc_cooValA_, loc_nnz_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&dloc_cooRowA_, loc_nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&dloc_cooColA_, loc_nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_x_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_opt_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_r_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_rhat_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_p_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_nu_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_t_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_z_, hd_m_ * sizeof(double)));
  if (bMeanConstraint_) {
    checkCudaErrors(cudaMalloc(&d_h2_, Nblocks * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_red_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_red_res_, sizeof(double)));
    d_red_temp_storage_ = NULL;
    red_temp_storage_bytes_ = 0;
    cub::DeviceReduce::Sum<double *, double *>(d_red_temp_storage_,
                                               red_temp_storage_bytes_, d_red_,
                                               d_red_res_, m_, solver_stream_);
    checkCudaErrors(cudaMalloc(&d_red_temp_storage_, red_temp_storage_bytes_));
  }
  if (comm_size_ > 1) {
    checkCudaErrors(cudaMalloc(&d_send_pack_idx_, send_buff_sz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_send_buff_, send_buff_sz_ * sizeof(double)));
    checkCudaErrors(
        cudaMallocHost(&h_send_buff_, send_buff_sz_ * sizeof(double)));
    checkCudaErrors(cudaMallocHost(&h_recv_buff_, halo_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dbd_cooValA_, bd_nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dbd_cooRowA_, bd_nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dbd_cooColA_, bd_nnz_ * sizeof(int)));
  }

  prof_.startProfiler("Memcpy", solver_stream_);
  checkCudaErrors(cudaMemcpyAsync(dloc_cooValA_, LocalLS_.loc_cooValA_.data(),
                                  loc_nnz_ * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(
      dloc_cooRowA_, LocalLS_.loc_cooRowA_int_.data(), loc_nnz_ * sizeof(int),
      cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(
      dloc_cooColA_, LocalLS_.loc_cooColA_int_.data(), loc_nnz_ * sizeof(int),
      cudaMemcpyHostToDevice, solver_stream_));
  if (comm_size_ > 1) {
    checkCudaErrors(cudaMemcpyAsync(
        d_send_pack_idx_, LocalLS_.send_pack_idx_.data(),
        send_buff_sz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(dbd_cooValA_, LocalLS_.bd_cooValA_.data(),
                                    bd_nnz_ * sizeof(double),
                                    cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(
        dbd_cooRowA_, LocalLS_.bd_cooRowA_int_.data(), bd_nnz_ * sizeof(int),
        cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(
        dbd_cooColA_, LocalLS_.bd_cooColA_int_.data(), bd_nnz_ * sizeof(int),
        cudaMemcpyHostToDevice, solver_stream_));
  }
  if (bMeanConstraint_)
    checkCudaErrors(cudaMemcpyAsync(d_h2_, LocalLS_.h2_.data(),
                                    Nblocks * sizeof(double),
                                    cudaMemcpyHostToDevice, solver_stream_));
  prof_.stopProfiler("Memcpy", solver_stream_);
  cusparseCreateCoo(&spDescrLocA_, m_, m_, loc_nnz_, dloc_cooRowA_,
                    dloc_cooColA_, dloc_cooValA_, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateDnVec(&spDescrNu_, m_, d_nu_, CUDA_R_64F);
  cusparseCreateDnVec(&spDescrT_, m_, d_t_, CUDA_R_64F);
  cusparseCreateDnVec(&spDescrLocZ_, m_, d_z_, CUDA_R_64F);
  cusparseSpMV_bufferSize(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          d_eye_, spDescrLocA_, spDescrLocZ_, d_nil_,
                          spDescrNu_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                          &locSpMVBuffSz_);
  checkCudaErrors(cudaMalloc(&locSpMVBuff_, locSpMVBuffSz_ * sizeof(char)));
  if (comm_size_ > 1) {
    cusparseCreateCoo(&spDescrBdA_, m_, hd_m_, bd_nnz_, dbd_cooRowA_,
                      dbd_cooColA_, dbd_cooValA_, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&spDescrBdZ_, hd_m_, d_z_, CUDA_R_64F);
    cusparseSpMV_bufferSize(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            d_eye_, spDescrBdA_, spDescrBdZ_, d_eye_,
                            spDescrNu_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                            &bdSpMVBuffSz_);
    checkCudaErrors(cudaMalloc(&bdSpMVBuff_, bdSpMVBuffSz_ * sizeof(char)));
  }

  this->updateVec();
}

void BiCGSTABSolver::updateVec() {
  prof_.startProfiler("Memcpy", solver_stream_);
  checkCudaErrors(cudaMemcpyAsync(d_x_, LocalLS_.x_.data(), m_ * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_r_, LocalLS_.b_.data(), m_ * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
  prof_.stopProfiler("Memcpy", solver_stream_);
}

__global__ void set_squared(double *const val) { val[0] *= val[0]; }

__global__ void set_amax(double *const dest, const int *const idx,
                         const double *const source) {
  dest[0] = fabs(source[idx[0] - 1]);
}

__global__ void set_negative(double *const dest, double *const source) {
  dest[0] = -source[0];
}

__global__ void breakdown_update(BiCGSTABScalars *coeffs) {
  coeffs->rho_prev = 1.;
  coeffs->alpha = 1.;
  coeffs->omega = 1.;
  coeffs->beta = (coeffs->rho_curr / (coeffs->rho_prev + coeffs->eps)) *
                 (coeffs->alpha / (coeffs->omega + coeffs->eps));
}

__global__ void set_beta(BiCGSTABScalars *coeffs) {
  coeffs->beta = (coeffs->rho_curr / (coeffs->rho_prev + coeffs->eps)) *
                 (coeffs->alpha / (coeffs->omega + coeffs->eps));
}

__global__ void set_alpha(BiCGSTABScalars *coeffs) {
  coeffs->alpha = coeffs->rho_curr / (coeffs->buff_1 + coeffs->eps);
}

__global__ void set_omega(BiCGSTABScalars *coeffs) {
  coeffs->omega = coeffs->buff_1 / (coeffs->buff_2 + coeffs->eps);
}

__global__ void set_rho(BiCGSTABScalars *coeffs) {
  coeffs->rho_prev = coeffs->rho_curr;
}

__global__ void blockDscal(const int m, const int BLEN,
                           const double *__restrict__ const alpha,
                           double *__restrict__ const x) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
       i += blockDim.x * gridDim.x)
    x[i] = alpha[i / BLEN] * x[i];
}

__global__ void send_buff_pack(int buff_sz, const int *const pack_idx,
                               double *const buff, const double *const source) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < buff_sz;
       i += blockDim.x * gridDim.x)
    buff[i] = source[pack_idx[i]];
}

void BiCGSTABSolver::hd_cusparseSpMV(double *d_op_hd, // operand vec
                                     cusparseDnVecDescr_t spDescrLocOp,
                                     cusparseDnVecDescr_t spDescrBdOp,
                                     double *d_res_hd, // result vec
                                     cusparseDnVecDescr_t spDescrRes) {

  const std::vector<int> &recv_ranks = LocalLS_.recv_ranks_;
  const std::vector<int> &recv_offset = LocalLS_.recv_offset_;
  const std::vector<int> &recv_sz = LocalLS_.recv_sz_;
  const std::vector<int> &send_ranks = LocalLS_.send_ranks_;
  const std::vector<int> &send_offset = LocalLS_.send_offset_;
  const std::vector<int> &send_sz = LocalLS_.send_sz_;

  if (comm_size_ > 1) {
    send_buff_pack<<<8 * 56, 32, 0, solver_stream_>>>(
        send_buff_sz_, d_send_pack_idx_, d_send_buff_, d_op_hd);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(
        sync_event_, solver_stream_)); // event to sync up for MPI comm
  }

  prof_.startProfiler("KerSpMV", solver_stream_);
  cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, d_eye_,
               spDescrLocA_, spDescrLocOp, d_nil_, spDescrRes, CUDA_R_64F,
               CUSPARSE_SPMV_ALG_DEFAULT, locSpMVBuff_);
  prof_.stopProfiler("KerSpMV", solver_stream_);

  if (comm_size_ > 1) {
    prof_.startProfiler("HaloComm", copy_stream_);
    checkCudaErrors(cudaStreamWaitEvent(copy_stream_, sync_event_, 0));
    checkCudaErrors(cudaMemcpyAsync(h_send_buff_, d_send_buff_,
                                    send_buff_sz_ * sizeof(double),
                                    cudaMemcpyDeviceToHost, copy_stream_));
    checkCudaErrors(cudaStreamSynchronize(copy_stream_));
    std::vector<MPI_Request> recv_requests(recv_ranks.size());
    for (size_t i(0); i < recv_ranks.size(); i++)
      MPI_Irecv(&h_recv_buff_[recv_offset[i]], recv_sz[i], MPI_DOUBLE,
                recv_ranks[i], 978, m_comm_, &recv_requests[i]);

    std::vector<MPI_Request> send_requests(send_ranks.size());
    for (size_t i(0); i < send_ranks.size(); i++)
      MPI_Isend(&h_send_buff_[send_offset[i]], send_sz[i], MPI_DOUBLE,
                send_ranks[i], 978, m_comm_, &send_requests[i]);

    MPI_Waitall(send_ranks.size(), send_requests.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(recv_ranks.size(), recv_requests.data(), MPI_STATUS_IGNORE);
    prof_.stopProfiler("HaloComm", copy_stream_);
    checkCudaErrors(cudaMemcpyAsync(&d_op_hd[m_], h_recv_buff_,
                                    halo_ * sizeof(double),
                                    cudaMemcpyHostToDevice, solver_stream_));

    prof_.startProfiler("HaloSpMV", solver_stream_);
    cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, d_eye_,
                 spDescrBdA_, spDescrBdOp, d_eye_, spDescrRes, CUDA_R_64F,
                 CUSPARSE_SPMV_ALG_DEFAULT, bdSpMVBuff_);
    prof_.stopProfiler("HaloSpMV", solver_stream_);
  }

  if (bMeanConstraint_) {
    checkCudaErrors(cudaMemcpyAsync(d_red_, d_op_hd, m_ * sizeof(double),
                                    cudaMemcpyDeviceToDevice, solver_stream_));
    blockDscal<<<8 * 56, 128, 0, solver_stream_>>>(m_, BLEN_, d_h2_, d_red_);
    checkCudaErrors(cudaGetLastError());

    cub::DeviceReduce::Sum<double *, double *>(d_red_temp_storage_,
                                               red_temp_storage_bytes_, d_red_,
                                               d_red_res_, m_, solver_stream_);

    double h_red_res;
    checkCudaErrors(cudaMemcpyAsync(&h_red_res, d_red_res_, sizeof(double),
                                    cudaMemcpyDeviceToHost, solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &h_red_res, 1, MPI_DOUBLE, MPI_SUM, m_comm_);

    if (bMeanRow_ >= 0)
      checkCudaErrors(cudaMemcpyAsync(&d_res_hd[bMeanRow_], &h_red_res,
                                      sizeof(double), cudaMemcpyHostToDevice,
                                      solver_stream_));
  }
}

void BiCGSTABSolver::main(const double max_error, const double max_rel_error,
                          const int max_restarts) {
  prof_.startProfiler("Total", solver_stream_);
  double error = 1e50;
  double error_init = 1e50;
  double error_opt = 1e50;
  bool bConverged = false;
  int restarts = 0;
  *h_coeffs_ = {1., 1., 1., 1e-21, 1., 1., 0., 0., 0};
  checkCudaErrors(cudaMemcpyAsync(d_coeffs_, h_coeffs_, sizeof(BiCGSTABScalars),
                                  cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_z_, d_x_, m_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, solver_stream_));
  hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_nu_, spDescrNu_);
  cublasDaxpy(cublas_handle_, m_, d_nye_, d_nu_, 1, d_r_, 1); // r <- -A*x_0 + b
  cublasIdamax(cublas_handle_, m_, d_nu_, 1, &(d_coeffs_->amax_idx));
  set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                        &(d_coeffs_->amax_idx), d_nu_);
  checkCudaErrors(cudaGetLastError());
  cublasIdamax(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->amax_idx));
  set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_2),
                                        &(d_coeffs_->amax_idx), d_r_);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1),
                                  2 * sizeof(double), cudaMemcpyDeviceToHost,
                                  solver_stream_));
  checkCudaErrors(cudaStreamSynchronize(solver_stream_));

  MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 2, MPI_DOUBLE, MPI_MAX,
                m_comm_);

  if (rank_ == 0) {
    std::cout << "  [BiCGSTAB]: || A*x_0 || = " << h_coeffs_->buff_1 << '\n';
    std::cout << "  [BiCGSTAB]: Initial norm: " << h_coeffs_->buff_2 << '\n';
  }
  error = h_coeffs_->buff_2;
  error_init = error;
  error_opt = error;
  checkCudaErrors(cudaMemcpyAsync(d_x_opt_, d_x_, m_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, solver_stream_));

  checkCudaErrors(cudaMemcpyAsync(d_rhat_, d_r_, m_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, solver_stream_));
  checkCudaErrors(
      cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(
      cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_));
  const size_t max_iter = 1000;
  for (size_t k(0); k < max_iter; k++) {
    cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_r_, 1, &(d_coeffs_->rho_curr));
    cublasDnrm2(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->buff_1));
    cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->buff_2));
    checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->rho_curr),
                                    &(d_coeffs_->rho_curr), 3 * sizeof(double),
                                    cudaMemcpyDeviceToHost, solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    h_coeffs_->buff_1 *= h_coeffs_->buff_1;
    h_coeffs_->buff_2 *= h_coeffs_->buff_2;
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->rho_curr), 3, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->rho_curr),
                                    &(h_coeffs_->rho_curr), sizeof(double),
                                    cudaMemcpyHostToDevice, solver_stream_));
    const bool serious_breakdown =
        h_coeffs_->rho_curr * h_coeffs_->rho_curr <
        1e-16 * h_coeffs_->buff_1 * h_coeffs_->buff_2;
    set_beta<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
    if (serious_breakdown && max_restarts > 0) {
      restarts++;
      if (restarts >= max_restarts) {
        break;
      }
      if (rank_ == 0) {
        std::cout << "  [BiCGSTAB]: Restart at iteration: " << k
                  << " norm: " << error << " Initial norm: " << error_init
                  << '\n';
      }
      checkCudaErrors(cudaMemcpyAsync(d_rhat_, d_r_, m_ * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      solver_stream_));
      cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->rho_curr));
      checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->rho_curr),
                                      &(d_coeffs_->rho_curr), sizeof(double),
                                      cudaMemcpyDeviceToHost, solver_stream_));
      checkCudaErrors(cudaStreamSynchronize(solver_stream_));
      h_coeffs_->rho_curr *= h_coeffs_->rho_curr;
      MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->rho_curr), 1, MPI_DOUBLE,
                    MPI_SUM, m_comm_);
      checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->rho_curr),
                                      &(h_coeffs_->rho_curr), sizeof(double),
                                      cudaMemcpyHostToDevice, solver_stream_));
      checkCudaErrors(
          cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_));
      checkCudaErrors(
          cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_));
      breakdown_update<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
      checkCudaErrors(cudaGetLastError());
    }
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->omega));
    checkCudaErrors(cudaGetLastError());
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_nu_, 1, d_p_, 1);
    cublasDscal(cublas_handle_, m_, &(d_coeffs_->beta), d_p_, 1);
    cublasDaxpy(cublas_handle_, m_, d_eye_, d_r_, 1, d_p_, 1);
    prof_.startProfiler("Prec", solver_stream_);
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BLEN_, m_ / BLEN_,
                BLEN_, d_eye_, d_P_inv_, BLEN_, d_p_, BLEN_, d_nil_, d_z_,
                BLEN_);
    prof_.stopProfiler("Prec", solver_stream_);
    hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_nu_, spDescrNu_);
    cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_nu_, 1, &(d_coeffs_->buff_1));
    checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1),
                                    sizeof(double), cudaMemcpyDeviceToHost,
                                    solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 1, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->buff_1), &(h_coeffs_->buff_1),
                                    sizeof(double), cudaMemcpyHostToDevice,
                                    solver_stream_));
    set_alpha<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->alpha), d_z_, 1, d_x_, 1);
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->alpha));
    checkCudaErrors(cudaGetLastError());
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_nu_, 1, d_r_, 1);
    prof_.startProfiler("Prec", solver_stream_);
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BLEN_, m_ / BLEN_,
                BLEN_, d_eye_, d_P_inv_, BLEN_, d_r_, BLEN_, d_nil_, d_z_,
                BLEN_);
    prof_.stopProfiler("Prec", solver_stream_);
    hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_t_, spDescrT_);
    cublasDdot(cublas_handle_, m_, d_t_, 1, d_r_, 1, &(d_coeffs_->buff_1));
    cublasDnrm2(cublas_handle_, m_, d_t_, 1, &(d_coeffs_->buff_2));
    set_squared<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_2));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1),
                                    2 * sizeof(double), cudaMemcpyDeviceToHost,
                                    solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 2, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->buff_1), &(h_coeffs_->buff_1),
                                    2 * sizeof(double), cudaMemcpyHostToDevice,
                                    solver_stream_));
    set_omega<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->omega), d_z_, 1, d_x_, 1);
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->omega));
    checkCudaErrors(cudaGetLastError());
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_t_, 1, d_r_, 1);
    cublasIdamax(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->amax_idx));
    set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                          &(d_coeffs_->amax_idx), d_r_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpyAsync(&error, &(d_coeffs_->buff_1),
                                    sizeof(double), cudaMemcpyDeviceToHost,
                                    solver_stream_));

    checkCudaErrors(cudaMemcpyAsync(h_coeffs_, d_coeffs_,
                                    sizeof(BiCGSTABScalars),
                                    cudaMemcpyDeviceToHost, solver_stream_));

    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, m_comm_);

    if (error < error_opt) {
      error_opt = error;
      checkCudaErrors(cudaMemcpyAsync(d_x_opt_, d_x_, m_ * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      solver_stream_));

      if ((error <= max_error) || (error / error_init <= max_rel_error)) {
        if (rank_ == 0)
          std::cout << "  [BiCGSTAB]: Converged after " << k << " iterations\n";

        bConverged = true;
        break;
      }
    }
    set_rho<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
  }

  if (rank_ == 0) {
    if (bConverged)
      std::cout << "  [BiCGSTAB] Error norm (relative) = " << error_opt << "/"
                << max_error << " (" << error_opt / error_init << "/"
                << max_rel_error << ")\n"
                << std::flush;
    else
      std::cout << "  [BiCGSTAB]: Iteration " << max_iter
                << ". Error norm (relative) = " << error_opt << "/" << max_error
                << " (" << error_opt / error_init << "/" << max_rel_error
                << ")\n"
                << std::flush;
  }

  prof_.startProfiler("Memcpy", solver_stream_);
  checkCudaErrors(cudaMemcpyAsync(LocalLS_.x_.data(), d_x_opt_,
                                  m_ * sizeof(double), cudaMemcpyDeviceToHost,
                                  solver_stream_));
  prof_.stopProfiler("Memcpy", solver_stream_);
  prof_.stopProfiler("Total", solver_stream_);
}
