//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

#ifdef BICGSTAB_PROFILER
class deviceProfiler
{
  public:
    deviceProfiler();
    ~deviceProfiler();

    void startProfiler(cudaStream_t);
    void stopProfiler(cudaStream_t);
    float elapsed() {return elapsed_; }

  protected:
    float elapsed_;
    cudaEvent_t start_;
    cudaEvent_t stop_;

};
#endif

class AMRSolver 
{
  /*
  Method used to solve Poisson's equation: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
  */
public:
  std::string getName() {
    return "GPU AMRSolver";
  }
  // Constructor and destructor
  AMRSolver(SimulationData& s);
  ~AMRSolver();

  //this object is used to compute the "flux corrections" at the interface between a coarse and fine grid point
  //these corrections are used for the coarse cell
  //for the fine cell, we just interpolate ghost cell values and pretend we're on a uniform grid
  // cubism::FluxCorrection<ScalarGrid,ScalarBlock> Corrector; 

  //main function used to solve Poisson's equation
  void solve();

protected:
  static constexpr int BSX_ = VectorBlock::sizeX;
  static constexpr int BSY_ = VectorBlock::sizeY;
  static constexpr int BSZ_ = BSX_*BSY_;

  static constexpr double eye_ = 1.;
  static constexpr double nye_ = -1.;
  static constexpr double nil_ = 0.;

  //This returns element K_{I1,I2}. It is used when we invert K
  double getA_local(int I1,int I2);

  //this struct contains information such as the currect timestep size, fluid properties and many others
  SimulationData& sim; 

  // Sparse linear system size
  int m_; // rows
  int n_; // cols
  int nnz_; // non-zero elements

  // Method to push back values to coo sparse matrix representaiton
  void cooMatPushBackVal(const double&, const int&, const int&);
  void cooMatPushBackRow(const int &, const std::map<int,double>&);
  // Method to add off-diagonal matrix element associated to cell in 'rhsNei' block
  template<class EdgeIndexer >
  void makeFlux(
      const cubism::BlockInfo &rhs_info,
      const int &ix,
      const int &iy,
      std::map<int,double> &row_map,
      const cubism::BlockInfo &rhsNei,
      const EdgeIndexer &helper) const;
  // Method to construct matrix row for cell on edge of block
  template<class EdgeIndexer>
  void makeEdgeCellRow( // excluding corners
      const cubism::BlockInfo &rhs_info,
      const int &ix,
      const int &iy,
      const bool &isBoundary,
      const cubism::BlockInfo &rhsNei,
      const EdgeIndexer &helper);
  // Method to construct matrix row for cell on corner of block
  template<class EdgeIndexer1, class EdgeIndexer2>
  void makeCornerCellRow(
      const cubism::BlockInfo &rhs_info,
      const int &ix,
      const int &iy,
      const bool &isBoundary1,
      const cubism::BlockInfo &rhsNei_1,
      const EdgeIndexer1 &helper1, 
      const bool &isBoundary2,
      const cubism::BlockInfo &rhsNei_2,
      const EdgeIndexer2 &helper2);
  // Method to compute A and b for the current mesh
  void get_LS();
  // Host-side variables for linear system
  std::vector<double> cooValA_;
  std::vector<int> cooRowA_;
  std::vector<int> csrRowA_;
  std::vector<int> cooColA_;
  std::vector<double> x_;
  std::vector<double> b_;

  // Row major linearly indexed matrix containing inverse preconditioner K_2^{-1}
  std::vector<double> P_inv_; 

  bool virginA_;
  bool updateA_;
  void alloc();
  void BiCGSTAB();
  void zero_mean();
  cudaStream_t solver_stream_;
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;

  // Device-side varibles for linear system
  double* d_cooValA_;
  int* d_csrRowA_;
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
  deviceProfiler pMemcpy_;
  deviceProfiler pSpMV_;
  deviceProfiler pPrec_;
  deviceProfiler pGlob_;
#endif
};
