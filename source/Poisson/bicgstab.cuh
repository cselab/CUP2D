#pragma once

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#ifdef BICGSTAB_PROFILER
#include "../Utils/DeviceProfiler.cuh"
#endif

struct BiCGSTABScalars {
  double rho_curr;
  double rho_prev;
  double alpha;
  double beta;
  double omega;
  double eps;
  double buff_1;
  double buff_2;
  int amax_idx;
};

class BiCGSTABSolver {
public:
  BiCGSTABSolver(
      const int BSX, 
      const int BSY, 
      const double* const P_inv);  
  ~BiCGSTABSolver();  

  // Solve method with update to LHS matrix
  void solve(
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
    const int max_restarts); 

  // Solve method without update to LHS matrix
  void solve(
    double* const h_x,
    const double* const h_b,
    const double max_error,
    const double max_rel_error,
    const int max_restarts); 

private:
  // Method to update LS
  void updateAll(
    const int m,
    const int n,
    const int nnz,
    const double* const h_cooValA,
    const int* const h_cooRowA,
    const int* const h_cooColA,
    double* const h_x,
    const double* const h_b);

  // Method to set RHS and LHS vec initial guess
  void updateVec(
    double* const h_x,
    const double* const h_b);

  // Main BiCGSTAB call
  void main(
    double* const h_x,
    const double max_error, 
    const double max_rel_error, 
    const int restarts);

  cudaStream_t solver_stream_;
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;

  bool dirty_ = false;  // dirty "bit" to set after first call to updateAll

  // Sparse linear system size
  int m_; // rows
  int n_; // cols
  int nnz_; // non-zero elements
  const int BLEN_; // block length (i.e no. of rows in preconditioner)

  // Device-side constants
  double* d_consts_;
  const double* d_eye_;
  const double* d_nye_;
  const double* d_nil_;

  // Device-side varibles for linear system
  BiCGSTABScalars* d_coeffs_;
  double* d_cooValA_;
  int* d_cooRowA_;
  int* d_cooColA_;
  double* d_x_;
  double* d_r_;
  double* d_P_inv_;
  // Device-side intermediate variables for BiCGSTAB
  double* d_rhat_;
  double* d_p_;
  double* d_nu_;
  double* d_t_;
  double* d_z_;
  // Descriptors for variables that will pass through cuSPARSE
  cusparseSpMatDescr_t spDescrA_;
  cusparseDnVecDescr_t spDescrX0_;
  cusparseDnVecDescr_t spDescrZ_;
  cusparseDnVecDescr_t spDescrNu_;
  cusparseDnVecDescr_t spDescrT_;
  // Work buffer for cusparseSpMV
  size_t SpMVBuffSz_;
  void* SpMVBuff_;

#ifdef BICGSTAB_PROFILER
  DeviceProfiler pMemcpy_;
  DeviceProfiler pSpMV_;
  DeviceProfiler pPrec_;
  DeviceProfiler pGlob_;
#endif
};
