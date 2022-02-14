#pragma once

#include <memory>
#include "mpi.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#include "LocalSpMatDnVec.h"
#ifdef BICGSTAB_PROFILER
#include "../Utils/DeviceProfiler.cuh"
#endif

struct BiCGSTABScalars {
  double alpha;
  double beta;
  double omega;
  double eps;
  double rho_prev;
  double rho_curr; // reductions happen along these three, make contigious
  double buff_1;
  double buff_2;
  int amax_idx;
};

class BiCGSTABSolver {
public:
  BiCGSTABSolver(
      MPI_Comm m_comm,
      std::shared_ptr<LocalSpMatDnVec> LocalLS,
      const int &BLEN, 
      const double* const P_inv);  
  ~BiCGSTABSolver();  

  // Solve method with update to LHS matrix
  void solveWithUpdate(
    const double max_error,
    const double max_rel_error,
    const int max_restarts); 

  // Solve method without update to LHS matrix
  void solveNoUpdate(
    const double max_error,
    const double max_rel_error,
    const int max_restarts); 

private:
  // Method to free memory allocated by updateAll
  void freeLast();

  // Method to update LS
  void updateAll();

  // Method to set RHS and LHS vec initial guess
  void updateVec();

  // Main BiCGSTAB call
  void main(
    const double max_error, 
    const double max_rel_error, 
    const int restarts);

  // Haloed SpMV
  void hd_cusparseSpMV(
    double* d_op,  // operand vec
    cusparseDnVecDescr_t spDescrOp,
    double* d_res, // result vec
    cusparseDnVecDescr_t Res);

  cudaStream_t solver_stream_;
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;

  bool dirty_ = false;  // dirty "bit" to set after first call to updateAll

  // Sparse linear system metadata
  int rank_;
  MPI_Comm m_comm_;
  int comm_size_;
  int m_;
  int halo_;
  int loc_nnz_;
  int bd_nnz_;
  int hd_m_; // haloed number of row
  const int BLEN_; // block length (i.e no. of rows in preconditioner)

  // LocalLS to be solved
  std::shared_ptr<LocalSpMatDnVec> LocalLS_;

  // Send/receive rules and buffers
  int send_buff_sz_;
  int* d_send_buff_pack_idx_;
  double* d_send_buff_;
  double* h_send_buff_;
  double* h_recv_buff_;

  // Device-side constants
  double* d_consts_;
  const double* d_eye_;
  const double* d_nye_;
  const double* d_nil_;

  // Device-side varibles for linear system
  BiCGSTABScalars* h_coeffs_;
  BiCGSTABScalars* d_coeffs_;
  double* dloc_cooValA_;
  int* dloc_cooRowA_;
  int* dloc_cooColA_;
  double* dbd_cooValA_;
  int* dbd_cooRowA_;
  int* dbd_cooColA_;
  double* d_x_;
  double* d_r_;
  double* d_P_inv_;
  // Device-side intermediate variables for BiCGSTAB
  double* d_rhat_;
  double* d_p_;
  double* d_nu_; // vecs with halos
  double* d_t_;
  double* d_z_;
  // Descriptors for variables that will pass through cuSPARSE
  cusparseSpMatDescr_t spDescrLocA_;
  cusparseSpMatDescr_t spDescrBdA_;
  cusparseDnVecDescr_t spDescrZ_;
  cusparseDnVecDescr_t spDescrNu_;
  cusparseDnVecDescr_t spDescrT_;
  // Work buffer for cusparseSpMV
  size_t locSpMVBuffSz_;
  void* locSpMVBuff_;
  size_t bdSpMVBuffSz_;
  void* bdSpMVBuff_;

#ifdef BICGSTAB_PROFILER
  DeviceProfiler pMemcpy_;
  DeviceProfiler pSpMV_;
  DeviceProfiler pPrec_;
  DeviceProfiler pGlob_;
#endif
};
