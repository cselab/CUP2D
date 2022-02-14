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
      const int &rank,
      const MPI_Comm &m_comm,
      const int &comm_size,
      const int &BSX, 
      const int &BSY, 
      const double* const P_inv);  
  ~BiCGSTABSolver();  

  // Solve method with update to LHS matrix
  void solveWithUpdate(
    std::shared_ptr<LocalSpMatDnVec> LocalLS,
    const double max_error,
    const double max_rel_error,
    const int max_restarts); 

  // Solve method without update to LHS matrix
  void solveNoUpdate(
    std::shared_ptr<LocalSpMatDnVec> LocalLS,
    const double max_error,
    const double max_rel_error,
    const int max_restarts); 

private:
  // Method to update LS
  void updateAll(std::shared_ptr<LocalSpMatDnVec> LocalLS);

  // Method to set RHS and LHS vec initial guess
  void updateVec(std::shared_ptr<LocalSpMatDnVec> LocalLS);

  // Main BiCGSTAB call
  void main(
    double* const h_x,
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

  // Vectors that contain rules for sending and receiving
  std::vector<int> recv_ranks_;
  std::vector<int> recv_offset_;
  std::vector<int> recv_sz_;

  std::vector<int> send_ranks_;
  std::vector<int> send_offset_;
  std::vector<int> send_sz_;
  int* d_send_buff_pack_idx_;
  double* d_send_buff_;
  int send_buff_sz_;

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
