#include <iostream>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#include "helper_cuda.h"
#include "bicgstab.h"

#ifndef GRID_SIZE
#define GRID_SIZE 2100
#endif

#ifdef BICGSTAB_PROFILER
void startProfiler(cudaEvent_t start, cudaStream_t stream)
{
  checkCudaErrors(cudaEventRecord(start, stream));
}

void stopProfiler(float &total_elapsed_ms, cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream)
{
  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));

  float this_ms = 0.;
  checkCudaErrors(cudaEventElapsedTime(&this_ms, start, stop));
  total_elapsed_ms += this_ms;
}
#endif // BICGSTAB_PROFILER

extern "C" void BiCGSTAB(
    const int m, // rows
    const int n, // cols
    const int nnz, // no. of non-zero elements
    const double* const h_cooValA,
    const int* const h_cooRowA,
    const int* const h_cooColA,
    double* const h_x, // contains initial guess
    const double* const h_b,
    const double max_error,
    const double max_rel_error,
    const int max_restarts) // if max_restarts == 0 defaults to normal BiCGSTAB without tricks
{
  // --------------------------------------------- Set-up streams and handles ---------------------------------------
  cudaStream_t solver_stream;
  cublasHandle_t cublas_handle;
  cusparseHandle_t cusparse_handle;
  checkCudaErrors(cudaStreamCreate(&solver_stream));
  checkCudaErrors(cublasCreate(&cublas_handle)); 
  checkCudaErrors(cusparseCreate(&cusparse_handle)); 
  // Set handles to stream
  checkCudaErrors(cublasSetStream(cublas_handle, solver_stream));
  checkCudaErrors(cusparseSetStream(cusparse_handle, solver_stream));

#ifdef BICGSTAB_PROFILER
  // ------------------------------------------------- Setup profiler -----------------------------------------------
  float elapsed_memcpy = 0.;
  float elapsed_bicgstab = 0.;
  cudaEvent_t start_memcpy; checkCudaErrors(cudaEventCreate(&start_memcpy));
  cudaEvent_t stop_memcpy; checkCudaErrors(cudaEventCreate(&stop_memcpy));
  cudaEvent_t start_bicgstab; checkCudaErrors(cudaEventCreate(&start_bicgstab));
  cudaEvent_t stop_bicgstab; checkCudaErrors(cudaEventCreate(&stop_bicgstab));

  startProfiler(start_bicgstab, solver_stream);
#endif // BICGSTAB_PROFILER

  // ------------------------------------------------- H2D transfer --------------------------------------------------
  // Host-device exec asynchronous, it may be worth already allocating pinned memory
  // and copying h2h (with cpu code) after async dev memory allocation calls 
  // to speed up h2d transfer down the line
   
  // Allocate device memory for linear system
  double* d_cooValA = NULL;
  double* d_cooValA_sorted = NULL;
  int* d_cooRowA = NULL;
  int* d_cooColA = NULL;
  double* d_x = NULL;
  double* d_b = NULL;
  checkCudaErrors(cudaMallocAsync(&d_cooValA, nnz * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_cooValA_sorted, nnz * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_cooRowA, nnz * sizeof(int), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_cooColA, nnz * sizeof(int), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_x, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_b, m * sizeof(double), solver_stream));

  // Possibly copy to pinned memory here followed by a sync call

#ifdef BICGSTAB_PROFILER
  startProfiler(start_memcpy, solver_stream);
#endif
  // H2D transfer of linear system
  checkCudaErrors(cudaMemcpyAsync(d_cooValA, h_cooValA, nnz * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_cooRowA, h_cooRowA, nnz * sizeof(int), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_cooColA, h_cooColA, nnz * sizeof(int), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_x, h_x, m * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_b, h_b, m * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
#ifdef BICGSTAB_PROFILER
  stopProfiler(elapsed_memcpy, start_memcpy, stop_memcpy, solver_stream);
#endif
  
  // Sort COO storage by row
  // 1. Deduce buffer size necessary for sorting and allocate storage for it
  size_t coosortBuffSz;
  void* coosortBuff = NULL;
  checkCudaErrors(cusparseXcoosort_bufferSizeExt(cusparse_handle, m, n, nnz, d_cooRowA, d_cooColA, &coosortBuffSz));
  checkCudaErrors(cudaMallocAsync(&coosortBuff, coosortBuffSz * sizeof(char), solver_stream));

  // 2. Set-up permutation vector P to track transformation from un-sorted to sorted list
  int* d_P = NULL;
  checkCudaErrors(cudaMallocAsync(&d_P, nnz * sizeof(int), solver_stream));
  checkCudaErrors(cusparseCreateIdentityPermutation(cusparse_handle, nnz, d_P));

  // 3. Sort d_cooRowA_ and d_cooCol inplace and apply permutation stored in d_P to d_cooValA_
  checkCudaErrors(cusparseXcoosortByRow(cusparse_handle, m, n, nnz, d_cooRowA, d_cooColA, d_P, coosortBuff));
  checkCudaErrors(cusparseDgthr(cusparse_handle, nnz, d_cooValA, d_cooValA_sorted, d_P, CUSPARSE_INDEX_BASE_ZERO));

  // Free buffers allocated for COO sort
  checkCudaErrors(cudaFreeAsync(coosortBuff, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_P, solver_stream));

  // ---------------------------------------------- BiCGSTAB ----------------------------------------------------------
  const double eye = 1.;
  const double nye = -1.;
  const double nil = 0.;

  /*
    This function generally follows notation of the Wikipedia page with several omissions
    to increase variable reuse.  Specifically:
      - d_x <-> h, x_i
      - d_b <-> r_0, r_i, s
  */

  // Initialize BiCGSTAB arrays and allocate memory
  double* d_rhat = NULL;
  double* d_p = NULL;
  double* d_nu = NULL;
  double* d_t = NULL;
  checkCudaErrors(cudaMallocAsync(&d_rhat, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_p, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_nu, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_t, m * sizeof(double), solver_stream));

  // Initialize variables to evaluate convergence
  double x_error = 1e50;
  double x_error_init = 1e50;
  double* d_xprev = NULL;
  checkCudaErrors(cudaMallocAsync(&d_xprev, m * sizeof(double), solver_stream));

  // Create descriptors for variables that will pass through cuSPARSE
  cusparseSpMatDescr_t spDescrA;
  cusparseDnVecDescr_t spDescrB;
  cusparseDnVecDescr_t spDescrX0;
  cusparseDnVecDescr_t spDescrP;
  cusparseDnVecDescr_t spDescrNu;
  cusparseDnVecDescr_t spDescrT;
  checkCudaErrors(cusparseCreateCoo(&spDescrA, m, n, nnz, d_cooRowA, d_cooColA, d_cooValA_sorted, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrB, m, d_b, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrX0, m, d_x, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrP, m, d_p, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrNu, m, d_nu, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrT, m, d_t, CUDA_R_64F));

  // Allocate work buffer for cusparseSpMV
  size_t SpMVBuffSz;
  void* SpMVBuff;
  checkCudaErrors(cusparseSpMV_bufferSize(
        cusparse_handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &eye, 
        spDescrA, 
        spDescrP, 
        &nil, 
        spDescrNu, 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        &SpMVBuffSz));
  checkCudaErrors(cudaMallocAsync(&SpMVBuff, SpMVBuffSz * sizeof(char), solver_stream));

  // 1. r <- b - A*x_0.  Add bias with cuBLAS like in "NVIDIA_CUDA-11.4_Samples/7_CUDALibraries/conjugateGradient"
  checkCudaErrors(cusparseSpMV( // A*x_0
        cusparse_handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &eye, 
        spDescrA, 
        spDescrX0, 
        &nil, 
        spDescrNu, // Use d_nu as temporary storage for result A*x_0 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        SpMVBuff)); 
  checkCudaErrors(cublasDaxpy(cublas_handle, m, &nye, d_nu, 1, d_b, 1)); // r <- -A*x_0 + b

#ifdef BICGSTAB_PROFILER
  // Check norm of A*x_0
  checkCudaErrors(cublasDnrm2(cublas_handle, m, d_nu, 1, &x_error_init));
  checkCudaErrors(cudaStreamSynchronize(solver_stream));
  std::cout << "  [BiCGSTAB]: || A*x_0 || = " << x_error_init << std::endl;
#endif
  
  // Calculate x_error_init for max_rel_error comparisons
  checkCudaErrors(cublasDcopy(cublas_handle, m, d_x, 1, d_xprev, 1));
  checkCudaErrors(cublasDaxpy(cublas_handle, m, &nye, d_b, 1, d_xprev, 1)); // initial solution guess stored in d_b
  checkCudaErrors(cublasDnrm2(cublas_handle, m, d_xprev, 1, &x_error_init));

  std::cout << "FIRST NORM: " << x_error_init << std::endl;
  // 2. Set r_hat = r
  checkCudaErrors(cublasDcopy(cublas_handle, m, d_b, 1, d_rhat, 1));

  // 3. Set initial values to scalars
  bool bConverged = false;
  int restarts = 0;
  double rho_curr = 1.;
  double rho_prev = 1.;
  double alpha = 1.;
  double omega = 1.;
  double beta = 0.;
  const double eps = 1e-21;

  // 4. Set initial values of vectors to zero
  checkCudaErrors(cudaMemsetAsync(d_nu, 0, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMemsetAsync(d_p, 0, m * sizeof(double), solver_stream));

  // 5. Start iterations
  const size_t max_iter = 1000;
  for(size_t k(0); k<max_iter; k++)
  {
    // 1. rho_i = (r_hat, r)
    checkCudaErrors(cublasDdot(cublas_handle, m, d_rhat, 1, d_b, 1, &rho_curr));
    
    double norm_1 = 0.;
    double norm_2 = 0.;
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_b, 1, &norm_1));
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_rhat, 1, &norm_2));
    checkCudaErrors(cudaStreamSynchronize(solver_stream)); // sync for 2. which happens on host
    // 2. beta = (rho_i / rho_{i-1}) * (alpha / omega_{i-1})
    beta = (rho_curr / (rho_prev+eps)) * (alpha / (omega+eps));

    // Numerical convergence trick
    const double cosTheta = rho_curr / norm_1 / norm_2;
    bool serious_breakdown = std::fabs(cosTheta) < 1e-8; 
    if(serious_breakdown && max_restarts > 0)
    {
      restarts++;
      if(restarts >= max_restarts){
        break;
      }
      std::cout << "  [BiCGSTAB]: Restart at iteration: " << k << " norm: " << x_error <<" Initial norm: " << x_error_init << std::endl;
      checkCudaErrors(cublasDcopy(cublas_handle, m, d_b, 1, d_rhat, 1));
      checkCudaErrors(cublasDnrm2(cublas_handle, m, d_rhat, 1, &rho_curr));
      checkCudaErrors(cudaStreamSynchronize(solver_stream)); 
      rho_curr *= rho_curr;
      rho_prev = 1.;
      alpha = 1.;
      omega = 1.;
      beta = (rho_curr / (rho_prev+eps)) * (alpha / (omega+eps));
    }

    // 3. p_i = r_{i-1} + beta(p_{i-1} - omega_{i-1}*nu_i)
    double nomega = -omega;
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nomega, d_nu, 1, d_p, 1)); // p <- -omega_{i-1}*nu_i + p
    checkCudaErrors(cublasDscal(cublas_handle, m, &beta, d_p, 1));            // p <- beta * p
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &eye, d_b, 1, d_p, 1));    // p <- r_{i-1} + p

    // 4. nu_i = A * p_i 
    checkCudaErrors(cusparseSpMV(
          cusparse_handle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &eye,
          spDescrA,
          spDescrP,
          &nil,
          spDescrNu,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          SpMVBuff));

    // 5. alpha = rho_i / (r_hat, nu_i)
    double alpha_den;
    checkCudaErrors(cublasDdot(cublas_handle, m, d_rhat, 1, d_nu, 1, &alpha_den)); // alpha <- (r_hat, nu_i)
    checkCudaErrors(cudaStreamSynchronize(solver_stream)); // sync for host division
    alpha = rho_curr / (alpha_den+eps); // alpha <- rho_i / alpha

    // 6. h = alpha*p_i + x_{i-1}
    checkCudaErrors(cublasDcopy(cublas_handle, m, d_x, 1, d_xprev, 1)); // copy previous value for future norm calculation
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &alpha, d_p, 1, d_x, 1));

    // 7. If h accurate enough then quit
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nye, d_x, 1, d_xprev, 1));
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_xprev, 1, &x_error));
    checkCudaErrors(cudaStreamSynchronize(solver_stream));

    if((x_error <= max_error) || (x_error / x_error_init <= max_rel_error))
    // if(x_error <= max_error)
    {
      std::cout << "  [BiCGSTAB]: Converged after " << k << " iterations" << std::endl;
      bConverged = true;
      break;
    }

    // 8. s = -alpha * nu_i + r_{i-1}
    const double nalpha = -alpha;
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nalpha, d_nu, 1, d_b, 1));

    // 9. t = A * s
    checkCudaErrors(cusparseSpMV(
          cusparse_handle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &eye,
          spDescrA,
          spDescrB,
          &nil,
          spDescrT,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          SpMVBuff));
    
    // 10. omega_i = (t,s)/(t,t), variables alpha & beta no longer in use this iter
    double omega_num;
    double omega_den;
    checkCudaErrors(cublasDdot(cublas_handle, m, d_t, 1, d_b, 1, &omega_num)); // alpha <- (t,s)
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_t, 1, &omega_den));          // beta <- sqrt(t,t)
    checkCudaErrors(cudaStreamSynchronize(solver_stream)); // sync for host arithmetic
    omega = omega_num / (omega_den * omega_den + eps);

    // 11. x_i = omega_i * s + h
    checkCudaErrors(cublasDcopy(cublas_handle, m, d_x, 1, d_xprev, 1)); // copy previous value for future norm calculation
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &omega, d_b, 1, d_x, 1));

    // 12. If x_i accurate enough then quit
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nye, d_x, 1, d_xprev, 1));
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_xprev, 1, &x_error));
    checkCudaErrors(cudaStreamSynchronize(solver_stream));

    if((x_error <= max_error) || (x_error / x_error_init <= max_rel_error))
    // if(x_error <= max_error)
    {
      std::cout << "  [BiCGSTAB]: Converged after " << k << " iterations" << std::endl;;
      bConverged = true;
      break;
    }

    // 13. r_i = -omega_i * t + s
    nomega = -omega;
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nomega, d_t, 1, d_b, 1));

    // Update *_prev values for next iteration
    rho_prev = rho_curr;
  }

  if( bConverged )
    std::cout <<  "  [BiCGSTAB]: Error norm (relative) = " << x_error << "/" << max_error 
              << " (" << x_error/x_error_init  << "/" << max_rel_error << ")" << std::endl;
  else
    std::cout <<  "  [BiCGSTAB]: Iteration " << max_iter 
              << ". Error norm (relative) = " << x_error << "/" << max_error 
              << " (" << x_error/x_error_init  << "/" << max_rel_error << ")" << std::endl;


#ifdef BICGSTAB_PROFILER
  startProfiler(start_memcpy, solver_stream);
#endif
  // Copy result back to host
  checkCudaErrors(cudaMemcpyAsync(h_x, d_x, m * sizeof(double), cudaMemcpyDeviceToHost, solver_stream));
#ifdef BICGSTAB_PROFILER
  stopProfiler(elapsed_memcpy, start_memcpy, stop_memcpy, solver_stream);
#endif

  // Cleanup memory alocated during BiCGSTAB
  checkCudaErrors(cusparseDestroySpMat(spDescrA));
  checkCudaErrors(cusparseDestroyDnVec(spDescrB));
  checkCudaErrors(cusparseDestroyDnVec(spDescrX0));
  checkCudaErrors(cusparseDestroyDnVec(spDescrP));
  checkCudaErrors(cusparseDestroyDnVec(spDescrNu));
  checkCudaErrors(cusparseDestroyDnVec(spDescrT));
  checkCudaErrors(cudaFreeAsync(d_rhat, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_p, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_nu, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_t, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_xprev, solver_stream));
  checkCudaErrors(cudaFreeAsync(SpMVBuff, solver_stream));


  // ------------------------------------------------------------------------------------------------------------------

  // Free device memory allocated for linear system
  checkCudaErrors(cudaFreeAsync(d_cooValA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_cooValA_sorted, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_cooRowA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_cooColA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_x, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_b, solver_stream));

#ifdef BICGSTAB_PROFILER
  stopProfiler(elapsed_bicgstab, start_bicgstab, stop_bicgstab, solver_stream);
  
  std::cout << "  [BiCGSTAB]: total elapsed time: " << elapsed_bicgstab << " [ms]." << std::endl;
  std::cout << "  [BiCGSTAB]: memory transfers:   " << (elapsed_memcpy/elapsed_bicgstab)*100. << "%." << std::endl;
#endif

  checkCudaErrors(cudaStreamSynchronize(solver_stream));
  // Destroy CUDA stream and library handles
#ifdef BICGSTAB_PROFILER
  checkCudaErrors(cudaEventDestroy(start_memcpy));
  checkCudaErrors(cudaEventDestroy(stop_memcpy));
  checkCudaErrors(cudaEventDestroy(start_bicgstab));
  checkCudaErrors(cudaEventDestroy(stop_bicgstab));
#endif
  checkCudaErrors(cublasDestroy(cublas_handle)); 
  checkCudaErrors(cusparseDestroy(cusparse_handle)); 
  checkCudaErrors(cudaStreamDestroy(solver_stream));
}

/* 
   Kernel to apply the inverse of preconditionner to each block of the grid 
          | Q 0 0  ...  . |
          | 0 Q 0  ...  . |
          | .   .       . |
    x <-- | .     .     . | x
          | .       .   . |
          | .         . 0 |
          | 0 0 0 ... 0 Q |
   where x, y are N-dimensioinal vector and Q (=P_inv) a BxB matrix

  NOTE: this kernel is hardcoded for either BSX or BSY to be a multiple of 8! 
*/
__global__ void preconditionVec(
    const int N,
    const double* const Q,
    double* const x)
    
{
  const int B = blockDim.x;     // B can be inferred from block dimension
  const int rowQ = threadIdx.x * B; // row of x <- Qx assigned to thread

  extern __shared__ double s[]; // establish pointer to shared memory
  double* x_ = s;               // B doubles for x from this block 
  double* Q_ = &x_[B];          // BxB double for K

  for (int i(0); i<B; i++)
  { // Copy Q to Q_ one row at a time
    const int idx = i*B + threadIdx.x;
    Q_[idx] = Q[idx];
  }

  // Grid-stride loop over blocks
  for (int i(blockIdx.x*blockDim.x + threadIdx.x); i<N; i += blockDim.x * gridDim.x)
  { 
    x_[threadIdx.x] = x[i]; // write x to shared mem x_
    __syncthreads();

    double aux = 0.;
    for (int j(0); j<B; j+=8)
    { // Each thread applies a row of preconditionner to x
      aux += Q_[rowQ + j] * x_[j];
      aux += Q_[rowQ + j+1] * x_[j+1];
      aux += Q_[rowQ + j+2] * x_[j+2];
      aux += Q_[rowQ + j+3] * x_[j+3];
      aux += Q_[rowQ + j+4] * x_[j+4];
      aux += Q_[rowQ + j+5] * x_[j+5];
      aux += Q_[rowQ + j+6] * x_[j+6];
      aux += Q_[rowQ + j+7] * x_[j+7];
    }
    x[i] = aux;
    __syncthreads();
  }
}

extern "C" void pBiCGSTAB(
    const int m, // rows
    const int n, // cols
    const int nnz, // no. of non-zero elements
    const double* const h_cooValA,
    const int* const h_cooRowA,
    const int* const h_cooColA,
    double* const h_x, // contains initial guess
    const double* const h_b,
    const int B, // block size BSX * BSY
    const double* const h_P_inv,
    const double max_error,
    const double max_rel_error,
    const int max_restarts) // if max_restarts == 0 defaults to normal BiCGSTAB without tricks
{
  if (B % 8 != 0){
    throw;
  }
  // --------------------------------------------- Set-up streams and handles ---------------------------------------
  cudaStream_t solver_stream;
  cublasHandle_t cublas_handle;
  cusparseHandle_t cusparse_handle;
  checkCudaErrors(cudaStreamCreate(&solver_stream));
  checkCudaErrors(cublasCreate(&cublas_handle)); 
  checkCudaErrors(cusparseCreate(&cusparse_handle)); 
  // Set handles to stream
  checkCudaErrors(cublasSetStream(cublas_handle, solver_stream));
  checkCudaErrors(cusparseSetStream(cusparse_handle, solver_stream));

#ifdef BICGSTAB_PROFILER
  // ------------------------------------------------- Setup profiler -----------------------------------------------
  float elapsed_memcpy = 0.;
  float elapsed_precondition = 0.;
  float elapsed_bicgstab = 0.;
  cudaEvent_t start_memcpy; checkCudaErrors(cudaEventCreate(&start_memcpy));
  cudaEvent_t stop_memcpy; checkCudaErrors(cudaEventCreate(&stop_memcpy));
  cudaEvent_t start_precondition; checkCudaErrors(cudaEventCreate(&start_precondition));
  cudaEvent_t stop_precondition; checkCudaErrors(cudaEventCreate(&stop_precondition));
  cudaEvent_t start_bicgstab; checkCudaErrors(cudaEventCreate(&start_bicgstab));
  cudaEvent_t stop_bicgstab; checkCudaErrors(cudaEventCreate(&stop_bicgstab));

  startProfiler(start_bicgstab, solver_stream);
#endif // BICGSTAB_PROFILER

  // ------------------------------------------------- H2D transfer --------------------------------------------------
  // Host-device exec asynchronous, it may be worth already allocating pinned memory
  // and copying h2h (with cpu code) after async dev memory allocation calls 
  // to speed up h2d transfer down the line
   
  // Allocate device memory for linear system
  double* d_cooValA = NULL;
  double* d_cooValA_sorted = NULL;
  int* d_cooRowA = NULL;
  int* d_cooColA = NULL;
  double* d_x = NULL;
  double* d_b = NULL;
  double* d_P_inv = NULL;
  checkCudaErrors(cudaMallocAsync(&d_cooValA, nnz * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_cooValA_sorted, nnz * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_cooRowA, nnz * sizeof(int), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_cooColA, nnz * sizeof(int), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_x, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_b, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_P_inv, B * B * sizeof(double), solver_stream));

  // Possibly copy to pinned memory here followed by a sync call

#ifdef BICGSTAB_PROFILER
  startProfiler(start_memcpy, solver_stream);
#endif
  // H2D transfer of linear system
  checkCudaErrors(cudaMemcpyAsync(d_cooValA, h_cooValA, nnz * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_cooRowA, h_cooRowA, nnz * sizeof(int), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_cooColA, h_cooColA, nnz * sizeof(int), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_x, h_x, m * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_b, h_b, m * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
  checkCudaErrors(cudaMemcpyAsync(d_P_inv, h_P_inv, B * B * sizeof(double), cudaMemcpyHostToDevice, solver_stream));
#ifdef BICGSTAB_PROFILER
  stopProfiler(elapsed_memcpy, start_memcpy, stop_memcpy, solver_stream);
#endif
  
  // Sort COO storage by row
  // 1. Deduce buffer size necessary for sorting and allocate storage for it
  size_t coosortBuffSz;
  void* coosortBuff =NULL;
  checkCudaErrors(cusparseXcoosort_bufferSizeExt(cusparse_handle, m, n, nnz, d_cooRowA, d_cooColA, &coosortBuffSz));
  checkCudaErrors(cudaMallocAsync(&coosortBuff, coosortBuffSz * sizeof(char), solver_stream));

  // 2. Set-up permutation vector P to track transformation from un-sorted to sorted list
  int* d_P = NULL; // not related to d_P_inv
  checkCudaErrors(cudaMallocAsync(&d_P, nnz * sizeof(int), solver_stream));
  checkCudaErrors(cusparseCreateIdentityPermutation(cusparse_handle, nnz, d_P));

  // 3. Sort d_cooRowA_ and d_cooCol inplace and apply permutation stored in d_P to d_cooValA_
  checkCudaErrors(cusparseXcoosortByRow(cusparse_handle, m, n, nnz, d_cooRowA, d_cooColA, d_P, coosortBuff));
  checkCudaErrors(cusparseDgthr(cusparse_handle, nnz, d_cooValA, d_cooValA_sorted, d_P, CUSPARSE_INDEX_BASE_ZERO));

  // Free buffers allocated for COO sort
  checkCudaErrors(cudaFreeAsync(coosortBuff, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_P, solver_stream));

  // ---------------------------------------------- pBiCGSTAB ----------------------------------------------------------
  const double eye = 1.;
  const double nye = -1.;
  const double nil = 0.;

  /*
    This function generally follows notation of the Wikipedia page with several omissions
    to increase variable reuse.  Specifically:
      - d_x <-> h, x_i
      - d_b <-> r_0, r_i, s
      - d_z <-> y, z
  */
  
  // Define preconditioning kernel launch configuration
  size_t gridSz = std::min(GRID_SIZE, m / B);
  size_t blockSz = B; 
  size_t sharedMemSz = (B + B*B) * sizeof(double); // in bytes!

  // Initialize BiCGSTAB arrays and allocate memory
  double* d_rhat = NULL;
  double* d_p = NULL;
  double* d_nu = NULL;
  double* d_t = NULL;
  double* d_z = NULL;
  checkCudaErrors(cudaMallocAsync(&d_rhat, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_p, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_nu, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_t, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMallocAsync(&d_z, m * sizeof(double), solver_stream));

  // Initialize variables to evaluate convergence
  double x_error = 1e50;
  double x_error_init = 1e50;
  double* d_xprev = NULL;
  checkCudaErrors(cudaMallocAsync(&d_xprev, m * sizeof(double), solver_stream));

  // Create descriptors for variables that will pass through cuSPARSE
  cusparseSpMatDescr_t spDescrA;
  cusparseDnVecDescr_t spDescrX0;
  cusparseDnVecDescr_t spDescrZ;
  cusparseDnVecDescr_t spDescrNu;
  cusparseDnVecDescr_t spDescrT;
  checkCudaErrors(cusparseCreateCoo(&spDescrA, m, n, nnz, d_cooRowA, d_cooColA, d_cooValA_sorted, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrX0, m, d_x, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrZ, m, d_z, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrNu, m, d_nu, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrT, m, d_t, CUDA_R_64F));

  // Allocate work buffer for cusparseSpMV
  size_t SpMVBuffSz;
  void* SpMVBuff;
  checkCudaErrors(cusparseSpMV_bufferSize(
        cusparse_handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &eye, 
        spDescrA, 
        spDescrX0, 
        &nil, 
        spDescrNu, 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        &SpMVBuffSz));
  checkCudaErrors(cudaMallocAsync(&SpMVBuff, SpMVBuffSz * sizeof(char), solver_stream));

  // 1. r <- b - A*x_0.  Add bias with cuBLAS like in "NVIDIA_CUDA-11.4_Samples/7_CUDALibraries/conjugateGradient"
  checkCudaErrors(cusparseSpMV( // A*x_0
        cusparse_handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &eye, 
        spDescrA, 
        spDescrX0, 
        &nil, 
        spDescrNu, // Use d_nu as temporary storage for result A*x_0 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        SpMVBuff)); 
  checkCudaErrors(cublasDaxpy(cublas_handle, m, &nye, d_nu, 1, d_b, 1)); // r <- -A*x_0 + b

#ifdef BICGSTAB_PROFILER
  // Check norm of A*x_0
  checkCudaErrors(cublasDnrm2(cublas_handle, m, d_nu, 1, &x_error_init));
  checkCudaErrors(cudaStreamSynchronize(solver_stream));
  std::cout << "  [pBiCGSTAB]: || A*x_0 || = " << x_error_init << std::endl;
#endif
  
  // Calculate x_error_init for max_rel_error comparisons
  checkCudaErrors(cublasDcopy(cublas_handle, m, d_x, 1, d_xprev, 1));
  checkCudaErrors(cublasDaxpy(cublas_handle, m, &nye, d_b, 1, d_xprev, 1)); // initial solution guess stored in d_b
  checkCudaErrors(cublasDnrm2(cublas_handle, m, d_xprev, 1, &x_error_init));

  // 2. Set r_hat = r
  checkCudaErrors(cublasDcopy(cublas_handle, m, d_b, 1, d_rhat, 1));

  // 3. Set initial values to scalars
  bool bConverged = false;
  int restarts = 0;
  double rho_curr = 1.;
  double rho_prev = 1.;
  double alpha = 1.;
  double omega = 1.;
  double beta = 0.;
  const double eps = 1e-21;

  // 4. Set initial values of vectors to zero
  checkCudaErrors(cudaMemsetAsync(d_nu, 0, m * sizeof(double), solver_stream));
  checkCudaErrors(cudaMemsetAsync(d_p, 0, m * sizeof(double), solver_stream));

  // 5. Start iterations
  const size_t max_iter = 1000;
  for(size_t k(0); k<max_iter; k++)
  {
    // 1. rho_i = (r_hat, r)
    checkCudaErrors(cublasDdot(cublas_handle, m, d_rhat, 1, d_b, 1, &rho_curr));
    
    double norm_1 = 0.;
    double norm_2 = 0.;
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_b, 1, &norm_1));
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_rhat, 1, &norm_2));
    checkCudaErrors(cudaStreamSynchronize(solver_stream)); // sync for 2. which happens on host
    // 2. beta = (rho_i / rho_{i-1}) * (alpha / omega_{i-1})
    beta = (rho_curr / (rho_prev+eps)) * (alpha / (omega+eps));

    // Numerical convergence trick
    const double cosTheta = rho_curr / norm_1 / norm_2;
    bool serious_breakdown = std::fabs(cosTheta) < 1e-8; 
    if(serious_breakdown && max_restarts > 0)
    {
      restarts++;
      if(restarts >= max_restarts){
        break;
      }
      std::cout << "  [pBiCGSTAB]: Restart at iteration: " << k << " norm: " << x_error <<" Initial norm: " << x_error_init << std::endl;
      checkCudaErrors(cublasDcopy(cublas_handle, m, d_b, 1, d_rhat, 1));
      checkCudaErrors(cublasDnrm2(cublas_handle, m, d_rhat, 1, &rho_curr));
      checkCudaErrors(cudaStreamSynchronize(solver_stream)); 
      rho_curr *= rho_curr;
      rho_prev = 1.;
      alpha = 1.;
      omega = 1.;
      beta = (rho_curr / (rho_prev+eps)) * (alpha / (omega+eps));
    }

    // 3. p_i = r_{i-1} + beta(p_{i-1} - omega_{i-1}*nu_i)
    double nomega = -omega;
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nomega, d_nu, 1, d_p, 1)); // p <- -omega_{i-1}*nu_i + p
    checkCudaErrors(cublasDscal(cublas_handle, m, &beta, d_p, 1));            // p <- beta * p
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &eye, d_b, 1, d_p, 1));    // p <- r_{i-1} + p

    // 4. z <- K_2^{-1} * p_i
#ifdef BICGSTAB_PROFILER
    startProfiler(start_precondition, solver_stream);
#endif
    checkCudaErrors(cublasDcopy(cublas_handle, m, d_p, 1, d_z, 1)); // z <- p_i
    preconditionVec<<<gridSz, blockSz, sharedMemSz, solver_stream>>>(m, d_P_inv, d_z); // z <- K_2^{-1}*z
    checkCudaErrors(cudaGetLastError());
#ifdef BICGSTAB_PROFILER
    stopProfiler(elapsed_precondition, start_precondition, stop_precondition, solver_stream);
#endif

    // 5. nu_i = A * z
    checkCudaErrors(cusparseSpMV(
          cusparse_handle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &eye,
          spDescrA,
          spDescrZ,
          &nil,
          spDescrNu,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          SpMVBuff));

    // 6. alpha = rho_i / (r_hat, nu_i)
    double alpha_den;
    checkCudaErrors(cublasDdot(cublas_handle, m, d_rhat, 1, d_nu, 1, &alpha_den)); // alpha <- (r_hat, nu_i)
    checkCudaErrors(cudaStreamSynchronize(solver_stream)); // sync for host division
    alpha = rho_curr / (alpha_den+eps); // alpha <- rho_i / alpha

    // 7. h = alpha*z + x_{i-1}
    checkCudaErrors(cublasDcopy(cublas_handle, m, d_x, 1, d_xprev, 1)); // copy previous value for future norm calculation
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &alpha, d_z, 1, d_x, 1));

    // 8. If h accurate enough then quit
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nye, d_x, 1, d_xprev, 1));
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_xprev, 1, &x_error));
    checkCudaErrors(cudaStreamSynchronize(solver_stream));

    if((x_error <= max_error) || (x_error / x_error_init <= max_rel_error))
    // if(x_error <= max_error)
    {
      std::cout << "  [pBiCGSTAB]: Converged after " << k << " iterations" << std::endl;
      bConverged = true;
      break;
    }

    // 9. s = -alpha * nu_i + r_{i-1}
    const double nalpha = -alpha;
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nalpha, d_nu, 1, d_b, 1));

    // 10. z <- K_2^{-1} * s
#ifdef BICGSTAB_PROFILER
    startProfiler(start_precondition, solver_stream);
#endif
    checkCudaErrors(cublasDcopy(cublas_handle, m, d_b, 1, d_z, 1)); // z <- s
    preconditionVec<<<gridSz, blockSz, sharedMemSz, solver_stream>>>(m, d_P_inv, d_z); // z <- K_2^{-1}*z
    checkCudaErrors(cudaGetLastError());
#ifdef BICGSTAB_PROFILER
    stopProfiler(elapsed_precondition, start_precondition, stop_precondition, solver_stream);
#endif

    // 11. t = A * z
    checkCudaErrors(cusparseSpMV(
          cusparse_handle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &eye,
          spDescrA,
          spDescrZ,
          &nil,
          spDescrT,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          SpMVBuff));
    
    // 12. omega_i = (t,s)/(t,t), variables alpha & beta no longer in use this iter
    double omega_num;
    double omega_den;
    checkCudaErrors(cublasDdot(cublas_handle, m, d_t, 1, d_b, 1, &omega_num)); // alpha <- (t,s)
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_t, 1, &omega_den));          // beta <- sqrt(t,t)
    checkCudaErrors(cudaStreamSynchronize(solver_stream)); // sync for host arithmetic
    omega = omega_num / (omega_den * omega_den + eps);

    // 13. x_i = omega_i * z + h
    checkCudaErrors(cublasDcopy(cublas_handle, m, d_x, 1, d_xprev, 1)); // copy previous value for future norm calculation
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &omega, d_z, 1, d_x, 1));

    // 14. If x_i accurate enough then quit
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nye, d_x, 1, d_xprev, 1));
    checkCudaErrors(cublasDnrm2(cublas_handle, m, d_xprev, 1, &x_error));
    checkCudaErrors(cudaStreamSynchronize(solver_stream));

    if((x_error <= max_error) || (x_error / x_error_init <= max_rel_error))
    // if(x_error <= max_error)
    {
      std::cout << "  [pBiCGSTAB]: Converged after " << k << " iterations" << std::endl;;
      bConverged = true;
      break;
    }

    // 15. r_i = -omega_i * t + s
    nomega = -omega;
    checkCudaErrors(cublasDaxpy(cublas_handle, m, &nomega, d_t, 1, d_b, 1));

    // Update *_prev values for next iteration
    rho_prev = rho_curr;
  }

  if( bConverged )
    std::cout <<  "  [pBiCGSTAB] Error norm (relative) = " << x_error << "/" << max_error 
              << " (" << x_error/x_error_init  << "/" << max_rel_error << ")" << std::endl;
  else
    std::cout <<  "  [pBiCGSTAB]: Iteration " << max_iter 
              << ". Error norm (relative) = " << x_error << "/" << max_error 
              << " (" << x_error/x_error_init  << "/" << max_rel_error << ")" << std::endl;


#ifdef BICGSTAB_PROFILER
  startProfiler(start_memcpy, solver_stream);
#endif
  // Copy result back to host
  checkCudaErrors(cudaMemcpyAsync(h_x, d_x, m * sizeof(double), cudaMemcpyDeviceToHost, solver_stream));
#ifdef BICGSTAB_PROFILER
  stopProfiler(elapsed_memcpy, start_memcpy, stop_memcpy, solver_stream);
#endif

  // Cleanup memory alocated during BiCGSTAB
  checkCudaErrors(cusparseDestroySpMat(spDescrA));
  checkCudaErrors(cusparseDestroyDnVec(spDescrX0));
  checkCudaErrors(cusparseDestroyDnVec(spDescrZ));
  checkCudaErrors(cusparseDestroyDnVec(spDescrNu));
  checkCudaErrors(cusparseDestroyDnVec(spDescrT));
  checkCudaErrors(cudaFreeAsync(d_rhat, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_p, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_nu, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_t, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_z, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_xprev, solver_stream));
  checkCudaErrors(cudaFreeAsync(SpMVBuff, solver_stream));


  // ------------------------------------------------------------------------------------------------------------------

  // Free device memory allocated for linear system
  checkCudaErrors(cudaFreeAsync(d_cooValA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_cooValA_sorted, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_cooRowA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_cooColA, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_x, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_b, solver_stream));
  checkCudaErrors(cudaFreeAsync(d_P_inv, solver_stream));

#ifdef BICGSTAB_PROFILER
  stopProfiler(elapsed_bicgstab, start_bicgstab, stop_bicgstab, solver_stream);
  
  std::cout << "  [pBiCGSTAB]: total elapsed time: " << elapsed_bicgstab << " [ms]." << std::endl;
  std::cout << "  [pBiCGSTAB]: memory transfers:   " << (elapsed_memcpy/elapsed_bicgstab)*100. << "%." << std::endl;
  std::cout << "  [pBiCGSTAB]: preconditioning:    " << (elapsed_precondition/elapsed_bicgstab)*100. << "%." << std::endl;
#endif

  checkCudaErrors(cudaStreamSynchronize(solver_stream));
  // Destroy CUDA stream and library handles
#ifdef BICGSTAB_PROFILER
  checkCudaErrors(cudaEventDestroy(start_memcpy));
  checkCudaErrors(cudaEventDestroy(stop_memcpy));
  checkCudaErrors(cudaEventDestroy(start_precondition));
  checkCudaErrors(cudaEventDestroy(stop_precondition));
  checkCudaErrors(cudaEventDestroy(start_bicgstab));
  checkCudaErrors(cudaEventDestroy(stop_bicgstab));
#endif
  checkCudaErrors(cublasDestroy(cublas_handle)); 
  checkCudaErrors(cusparseDestroy(cusparse_handle)); 
  checkCudaErrors(cudaStreamDestroy(solver_stream));
}
