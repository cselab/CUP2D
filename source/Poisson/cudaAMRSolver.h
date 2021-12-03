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
  void cooMatPushBackVal(const double&, const int&, const int&);
  void cooMatPushBackRow(const int &, const std::map<int,double>&);
  // Method to add off-diagonal matrix element associated to cell in 'rhsNei' block
  template<class EdgeIndexer >
  void makeFlux(
      const cubism::BlockInfo &rhs_info,
      const int &BSX,
      const int &BSY,
      const int &ix,
      const int &iy,
      std::map<int,double> &row_map,
      const cubism::BlockInfo &rhsNei,
      const EdgeIndexer &helper) const;
  // Method to construct matrix row for cell on edge of block
  template<class EdgeIndexer>
  void makeEdgeCellRow( // excluding corners
      const cubism::BlockInfo &rhs_info,
      const int &BSX,
      const int &BSY,
      const int &ix,
      const int &iy,
      const bool &isBoundary,
      const cubism::BlockInfo &rhsNei,
      const EdgeIndexer &helper);
  // Method to construct matrix row for cell on corner of block
  template<class EdgeIndexer1, class EdgeIndexer2>
  void makeCornerCellRow(
      const cubism::BlockInfo &rhs_info,
      const int &BSX,
      const int &BSY,
      const int &ix,
      const int &iy,
      const bool &isBoundary1,
      const cubism::BlockInfo &rhsNei_1,
      const EdgeIndexer1 &helper1, 
      const bool &isBoundary2,
      const cubism::BlockInfo &rhsNei_2,
      const EdgeIndexer2 &helper2);
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
    where each K is a small (BSX*BSY)x(BSX*BSY) matrix that multipli3es each (BSX*BSY)x(BSX*BSY) block
  */
  // Row major linearly indexed matrix containing inverse preconditioner K_2^{-1}
  std::vector<double> P_inv_; 

};
