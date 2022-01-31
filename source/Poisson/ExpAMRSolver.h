//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"
#include "Base.h"
#include "bicgstab.cuh"

class ExpAMRSolver : public PoissonSolver
{
  /*
  Method used to solve Poisson's equation: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
  */
public:
  std::string getName() {
    // ExpAMRSolver == AMRSolver for explicit linear system
    return "ExpAMRSolver";
  }
  // Constructor and destructor
  ExpAMRSolver(SimulationData& s);
  ~ExpAMRSolver();

  //main function used to solve Poisson's equation
  void solve(
      const ScalarGrid *input, 
      ScalarGrid * const output);

protected:
  //this struct contains information such as the currect timestep size, fluid properties and many others
  SimulationData& sim; 

  // Pointer to solving backend of SpMat DnVec linear system
  BiCGSTABSolver* backend_;

  static constexpr int BSX_ = VectorBlock::sizeX;
  static constexpr int BSY_ = VectorBlock::sizeY;
  static constexpr int BLEN_ = BSX_*BSY_;

  //This returns element K_{I1,I2}. It is used when we invert K
  double getA_local(int I1,int I2);

  // Sparse linear system size
  int m_; // rows
  int n_; // cols
  int nnz_; // non-zero elements

  // Methods to push back values to coo sparse matrix representaiton
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
  void getMat(); // update LHS and RHS after refinement
  void getVec(); // update initial guess and RHS vecs only

  // Host-side variables for linear system
  std::vector<double> cooValA_;
  std::vector<int> cooRowA_;
  std::vector<int> cooColA_;
  std::vector<double> x_;
  std::vector<double> b_;
};
