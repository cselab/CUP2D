#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#include "bicgstab.cuh"

BiCGSTABSolver::BiCGSTABSolver(
    const int BSX, 
    const int BSY, 
    const double* const P_inv)
  : BLEN_(BSX*BSY)
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

  // Copy preconditionner
  checkCudaErrors(cudaMalloc(&d_P_inv_, BLEN_ * BLEN_ * sizeof(double)));
  checkCudaErrors(cudaMemcpyAsync(d_P_inv_, P_inv, BLEN_ * BLEN_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
}

BiCGSTABSolver::~BiCGSTABSolver()
{
  std::cout << "---------------- Calling on BiCGSTABSolver() destructor ------------\n";
  // Cleanup after last timestep
  checkCudaErrors(cudaFree(d_cooValA_));
  checkCudaErrors(cudaFree(d_cooRowA_));
  checkCudaErrors(cudaFree(d_cooColA_));
  checkCudaErrors(cudaFree(d_x_));
  checkCudaErrors(cudaFree(d_r_));
  checkCudaErrors(cusparseDestroySpMat(spDescrA_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrX0_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrZ_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrNu_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrT_));
  checkCudaErrors(cudaFree(d_rhat_));
  checkCudaErrors(cudaFree(d_p_));
  checkCudaErrors(cudaFree(d_nu_));
  checkCudaErrors(cudaFree(d_t_));
  checkCudaErrors(cudaFree(d_z_));
  checkCudaErrors(cudaFree(SpMVBuff_));

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

  // Destroy CUDA streams and handles
  checkCudaErrors(cublasDestroy(cublas_handle_)); 
  checkCudaErrors(cusparseDestroy(cusparse_handle_)); 
  checkCudaErrors(cudaStreamDestroy(solver_stream_));
}

// --------------------------------- public class methods ------------------------------------

void BiCGSTABSolver::solve(
    const int m, // rows
    const int n, // cols
    const int nnz,
    const double* const h_cooValA,
    const int* const h_cooRowA,
    const int* const h_cooColA,
    double* const h_x,
    const double* const h_b,
    const double max_error,
    const double max_rel_error,
    const int max_restarts)
{
  this->updateAll(m, n, nnz, h_cooValA, h_cooRowA, h_cooColA, h_x, h_b);
  this->main(h_x, max_error, max_rel_error, max_restarts);
}

void BiCGSTABSolver::solve(
    double* const h_x,
    const double* const h_b,
    const double max_error,
    const double max_rel_error,
    const int max_restarts)
{
  this->updateVec(h_x, h_b);
  this->main(h_x, max_error, max_rel_error, max_restarts);
}

// --------------------------------- private class methods ------------------------------------

void BiCGSTABSolver::updateAll(
    const int m,
    const int n,
    const int nnz,
    const double* const h_cooValA,
    const int* const h_cooRowA,
    const int* const h_cooColA,
    double* const h_x,
    const double* const h_b)
{
  // Update LS metadata
  m_ = m;
  n_ = n;
  nnz_ = nnz;

  if (dirty_) // Previous time-step exists so cleanup first
  {
    // Free device memory allocated for linear system from previous time-step
    checkCudaErrors(cudaFree(d_cooValA_));
    checkCudaErrors(cudaFree(d_cooRowA_));
    checkCudaErrors(cudaFree(d_cooColA_));
    // Note that memory for LHS & RHS vectors also reallocated here
    checkCudaErrors(cudaFree(d_x_)); 
    checkCudaErrors(cudaFree(d_r_));
    // Cleanup memory allocated for BiCGSTAB arrays
    checkCudaErrors(cusparseDestroySpMat(spDescrA_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrX0_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrZ_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrNu_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrT_));
    checkCudaErrors(cudaFree(d_rhat_));
    checkCudaErrors(cudaFree(d_p_));
    checkCudaErrors(cudaFree(d_nu_));
    checkCudaErrors(cudaFree(d_t_));
    checkCudaErrors(cudaFree(d_z_));
    checkCudaErrors(cudaFree(SpMVBuff_));
  }
  dirty_ = true;

  // Allocate device memory for linear system
  checkCudaErrors(cudaMalloc(&d_cooValA_, nnz_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_cooRowA_, nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_cooColA_, nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_x_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_r_, m_ * sizeof(double)));

#ifdef BICGSTAB_PROFILER
  pMemcpy_.startProfiler(solver_stream_);
#endif
  // H2D transfer of linear system
  checkCudaErrors(cudaMemcpyAsync(d_cooValA_, h_cooValA, nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_cooRowA_, h_cooRowA, nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_cooColA_, h_cooColA, nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
#ifdef BICGSTAB_PROFILER
  pMemcpy_.stopProfiler(solver_stream_);
#endif

  // Allocate arrays for BiCGSTAB storage
  checkCudaErrors(cudaMalloc(&d_rhat_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_p_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_nu_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_t_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_z_, m_ * sizeof(double)));
  // Create descriptors for variables that will pass through cuSPARSE
  checkCudaErrors(cusparseCreateCoo(&spDescrA_, m_, n_, nnz_, d_cooRowA_, d_cooColA_, d_cooValA_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrX0_, m_, d_x_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrZ_, m_, d_z_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrNu_, m_, d_nu_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrT_, m_, d_t_, CUDA_R_64F));
  // Allocate work buffer for cusparseSpMV
  checkCudaErrors(cusparseSpMV_bufferSize(
        cusparse_handle_, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        d_eye_, 
        spDescrA_, 
        spDescrX0_, 
        d_nil_, 
        spDescrNu_, 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        &SpMVBuffSz_));
  checkCudaErrors(cudaMalloc(&SpMVBuff_, SpMVBuffSz_ * sizeof(char)));

  this->updateVec(h_x, h_b);
}

void BiCGSTABSolver::updateVec(
    double* const h_x,
    const double* const h_b)
{
#ifdef BICGSTAB_PROFILER
  pMemcpy_.startProfiler(solver_stream_);
#endif
  // Copy RHS LHS vec initial guess, if LS was updated, updateAll reallocates sufficient memory
  checkCudaErrors(cudaMemcpyAsync(d_x_, h_x, m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_r_, h_b, m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
#ifdef BICGSTAB_PROFILER
  pMemcpy_.stopProfiler(solver_stream_);
#endif
}

__global__ void set_negative(double* const dest, double* const source)
{
  dest[0] = -source[0];
}

__global__ void breakdown_update(BiCGSTABScalars* coeffs)
{
  coeffs->rho_curr *= coeffs->rho_curr;
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
  coeffs->omega = coeffs->buff_1 / (coeffs->buff_2 * coeffs->buff_2 + coeffs->eps);
}

__global__ void set_rho(BiCGSTABScalars* coeffs)
{
  coeffs->rho_prev = coeffs->rho_curr;
}

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
  BiCGSTABScalars h_coeffs = {1., 1., 1., 1., 1., 0., 0., 1e-21};
  checkCudaErrors(cudaMemcpyAsync(d_coeffs_, &h_coeffs, sizeof(BiCGSTABScalars), cudaMemcpyHostToDevice, solver_stream_));

  // 1. r <- b - A*x_0.  Add bias with cuBLAS like in "NVIDIA_CUDA-11.4_Samples/7_CUDALibraries/conjugateGradient"
#ifdef BICGSTAB_PROFILER
  pSpMV_.startProfiler(solver_stream_);
#endif
  checkCudaErrors(cusparseSpMV( // A*x_0
        cusparse_handle_, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        d_eye_, 
        spDescrA_, 
        spDescrX0_, 
        d_nil_, 
        spDescrNu_, // Use d_nu_ as temporary storage for result A*x_0 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        SpMVBuff_)); 
#ifdef BICGSTAB_PROFILER
  pSpMV_.stopProfiler(solver_stream_);
#endif
  checkCudaErrors(cublasDaxpy(cublas_handle_, m_, d_nye_, d_nu_, 1, d_r_, 1)); // r <- -A*x_0 + b

  // Check norm of A*x_0 and get error_init
  checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_nu_, 1, &(d_coeffs_->buff_1)));
  checkCudaErrors(cudaMemcpyAsync(&error, &(d_coeffs_->buff_1), sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
  checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->buff_2)));
  checkCudaErrors(cudaMemcpyAsync(&error_init, &(d_coeffs_->buff_2), sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));

  checkCudaErrors(cudaStreamSynchronize(solver_stream_));
  std::cout << "  [BiCGSTAB]: || A*x_0 || = " << error << std::endl;
  std::cout << "  [BiCGSTAB]: Initial norm: " << error_init << std::endl;
  error = error_init;

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
    
    // 2. beta = (rho_i / rho_{i-1}) * (alpha / omega_{i-1})
    set_beta<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());

    // Numerical convergence trick
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->buff_1)));
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->buff_2)));
    checkCudaErrors(cudaMemcpyAsync(&h_coeffs, d_coeffs_, sizeof(BiCGSTABScalars), cudaMemcpyDeviceToHost, solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); 

    const double cosTheta = h_coeffs.rho_curr / h_coeffs.buff_1 / h_coeffs.buff_2;
    bool serious_breakdown = std::fabs(cosTheta) < 1e-8; 
    if(serious_breakdown && max_restarts > 0)
    {
      restarts++;
      if(restarts >= max_restarts){
        break;
      }
      std::cout << "  [BiCGSTAB]: Restart at iteration: " << k << " norm: " << error <<" Initial norm: " << error_init << std::endl;
      checkCudaErrors(cublasDcopy(cublas_handle_, m_, d_r_, 1, d_rhat_, 1));
      checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->rho_curr)));
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
#ifdef BICGSTAB_PROFILER
    pSpMV_.startProfiler(solver_stream_);
#endif
    checkCudaErrors(cusparseSpMV(
          cusparse_handle_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          d_eye_,
          spDescrA_,
          spDescrZ_,
          d_nil_,
          spDescrNu_,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          SpMVBuff_));
#ifdef BICGSTAB_PROFILER
    pSpMV_.stopProfiler(solver_stream_);
#endif

    // 6. alpha = rho_i / (r_hat, nu_i)
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_nu_, 1, &(d_coeffs_->buff_1)));
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
#ifdef BICGSTAB_PROFILER
    pSpMV_.startProfiler(solver_stream_);
#endif
    checkCudaErrors(cusparseSpMV(
          cusparse_handle_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          d_eye_,
          spDescrA_,
          spDescrZ_,
          d_nil_,
          spDescrT_,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          SpMVBuff_));
#ifdef BICGSTAB_PROFILER
    pSpMV_.stopProfiler(solver_stream_);
#endif
    
    // 12. omega_i = (t,s)/(t,t), variables alpha & beta no longer in use this iter
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_t_, 1, d_r_, 1, &(d_coeffs_->buff_1)));
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_t_, 1, &(d_coeffs_->buff_2)));
    set_omega<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());

    // 13. x_i = omega_i * z + h
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->omega), d_z_, 1, d_x_, 1));

    // 15. r_i = -omega_i * t + s
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1), &(d_coeffs_->omega));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_t_, 1, d_r_, 1));

    // If x_i accurate enough then quit
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->buff_1)));
    checkCudaErrors(cudaMemcpyAsync(&error, &(d_coeffs_->buff_1), sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));

    if((error <= max_error) || (error / error_init <= max_rel_error))
    // if(x_error <= max_error)
    {
      std::cout << "  [BiCGSTAB]: Converged after " << k << " iterations" << std::endl;;
      bConverged = true;
      break;
    }

    // Update *_prev values for next iteration
    set_rho<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
  }

  if( bConverged )
    std::cout <<  "  [BiCGSTAB] Error norm (relative) = " << error << "/" << max_error 
              << " (" << error/error_init  << "/" << max_rel_error << ")" << std::endl;
  else
    std::cout <<  "  [BiCGSTAB]: Iteration " << max_iter 
              << ". Error norm (relative) = " << error << "/" << max_error 
              << " (" << error/error_init  << "/" << max_rel_error << ")" << std::endl;

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

