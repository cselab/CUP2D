#pragma once
#include "../Utils/DeviceProfiler.cuh"
#include "LocalSpMatDnVec.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <memory>
#include <mpi.h>
struct BiCGSTABScalars {
  double alpha;
  double beta;
  double omega;
  double eps;
  double rho_prev;
  double rho_curr;
  double buff_1;
  double buff_2;
  int amax_idx;
};
class BiCGSTABSolver {
public:
  BiCGSTABSolver(MPI_Comm m_comm, LocalSpMatDnVec &LocalLS, const int BLEN,
                 const bool bMeanConstraint, const std::vector<double> &P_inv);
  ~BiCGSTABSolver();
  void solveWithUpdate(const double max_error, const double max_rel_error,
                       const int max_restarts);
  void solveNoUpdate(const double max_error, const double max_rel_error,
                     const int max_restarts);

private:
  void freeLast();
  void updateAll();
  void updateVec();
  void main(const double max_error, const double max_rel_error,
            const int restarts);
  void hd_cusparseSpMV(double *d_op, cusparseDnVecDescr_t spDescrLocOp,
                       cusparseDnVecDescr_t spDescrBdOp, double *d_res,
                       cusparseDnVecDescr_t Res);
  cudaStream_t solver_stream_;
  cudaStream_t copy_stream_;
  cudaEvent_t sync_event_;
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;
  bool dirty_ = false;
  int rank_;
  MPI_Comm m_comm_;
  int comm_size_;
  int m_;
  int halo_;
  int loc_nnz_;
  int bd_nnz_;
  int hd_m_;
  const int BLEN_;
  const bool bMeanConstraint_;
  int bMeanRow_;
  LocalSpMatDnVec &LocalLS_;
  int send_buff_sz_;
  int *d_send_pack_idx_;
  double *d_send_buff_;
  double *h_send_buff_;
  double *h_recv_buff_;
  double *d_consts_;
  const double *d_eye_;
  const double *d_nye_;
  const double *d_nil_;
  BiCGSTABScalars *h_coeffs_;
  BiCGSTABScalars *d_coeffs_;
  double *dloc_cooValA_;
  int *dloc_cooRowA_;
  int *dloc_cooColA_;
  double *dbd_cooValA_;
  int *dbd_cooRowA_;
  int *dbd_cooColA_;
  double *d_x_;
  double *d_x_opt_;
  double *d_r_;
  double *d_P_inv_;
  size_t red_temp_storage_bytes_;
  void *d_red_temp_storage_;
  double *d_red_;
  double *d_red_res_;
  double *d_h2_;
  double *d_rhat_;
  double *d_p_;
  double *d_nu_;
  double *d_t_;
  double *d_z_;
  cusparseSpMatDescr_t spDescrLocA_;
  cusparseSpMatDescr_t spDescrBdA_;
  cusparseDnVecDescr_t spDescrNu_;
  cusparseDnVecDescr_t spDescrT_;
  cusparseDnVecDescr_t spDescrLocZ_;
  cusparseDnVecDescr_t spDescrBdZ_;
  size_t locSpMVBuffSz_;
  void *locSpMVBuff_;
  size_t bdSpMVBuffSz_;
  void *bdSpMVBuff_;
  DeviceProfiler prof_;
};
