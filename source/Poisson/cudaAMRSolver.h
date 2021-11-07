//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

class cudaAMRSolver 
{
  /*
  Method used to solve Poisson's equation: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
  */
public:
  std::string getName() {
    return "cudaAMRSolver";
  }
  // Constructor and destructor
  cudaAMRSolver(SimulationData& s);
  ~cudaAMRSolver();

  //this object is used to compute the "flux corrections" at the interface between a coarse and fine grid point
  //these corrections are used for the coarse cell
  //for the fine cell, we just interpolate ghost cell values and pretend we're on a uniform grid
  // cubism::FluxCorrection<ScalarGrid,ScalarBlock> Corrector; 

  //main function used to solve Poisson's equation
  void solve();

protected:
  //This returns element K_{I1,I2}. It is used when we invert K
  double getA_local(int I1,int I2);

  //this struct contains information such as the currect timestep size, fluid properties and many others
  SimulationData& sim; 

  // Sparse linear system size
  int m_; // rows
  int n_; // cols
  int nnz_; // non-zero elements

  // Method to push back values to coo sparse matrix representaiton
  void cooMatPushBack(const double&, const int&, const int&);
  // Method to off-diagonal matrix element associated to 'rhsNei' block
  template<typename F1>
  void neiBlockElement(
    const int &block_idx,
    const int &BSX,
    const int &BSY,
    const int &ix,
    const int &iy,
    const int &sfc_idx,
    double &diag_val,
    cubism::BlockInfo &rhsNei,
    F1 n_func);
  // Method to construct matrix row for cell on edge of block
  template<typename F1, typename F2, typename F3, typename F4>
  void edgeBoundaryCell(
      const int &block_idx,
      const int &BSX,
      const int &BSY,
      const int &ix,
      const int &iy,
      F1 n1_func,
      F2 n2_func,
      F3 n3_func,
      const bool &isBoundary4,
      cubism::BlockInfo &rhsNei_4,
      F4 n4_func);
  // Method to construct matrix row for cell on corner of block
  template<typename F1, typename F2, typename F3, typename F4>
  void cornerBoundaryCell(
    const int &block_idx,
    const int &BSX,
    const int &BSY,
    const int &ix,
    const int &iy,
    F1 n1_func,
    F2 n2_func,
    const bool &isBoundary3,
    cubism::BlockInfo &rhsNei_3,
    F3 n3_func,
    const bool &isBoundary4,
    cubism::BlockInfo &rhsNei_4,
    F4 n4_func);
  // Method to compute A and b for the current mesh
  void Get_LS();
  // Host-side variables for linear system
  std::vector<double> cooValA_;
  std::vector<int> cooRowA_;
  std::vector<int> cooColA_;
  std::vector<double> x_;
  std::vector<double> b_;

  //this stuff below is used for preconditioning the system
  
  //Following Wikipedia's notation, we use the preconditioner K_1 * K_2, where K_1 = I is the identity matrix
  //and K_2 is a block diagonal matrix, i.e.:
  /*
         K 0 ...
   K_2 = 0 K 0 ...
         .
         .
         .
         0 0 0 ... K
    where each K is a small 8x8 matrix that multipli3es each 8x8 block
  */
  //These vectors are used to store the inverse of K
  //We only store the non-zero elements of the inverse
  // std::vector<std::vector<double>> Ld;
  // std::vector <  std::vector <std::vector< std::pair<int,double> > > >L_row;
  // std::vector <  std::vector <std::vector< std::pair<int,double> > > >L_col;

  // Row major linearly indexed matrix containing inverse preconditioner
  std::vector<double> P_inv_; 

};
