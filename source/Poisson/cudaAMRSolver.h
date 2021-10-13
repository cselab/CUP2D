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
  //this struct contains information such as the currect timestep size, fluid properties and many others
  SimulationData& sim; 

  // Sparse linear system size
  int m_; // rows
  int n_; // cols
  int nnz_; // non-zero elements

  // Method to push back values to coo sparse matrix representaiton
  void inline h_cooMatPushBack(const double&, const int&, const int&);
  // Method to compute A and b for the current mesh
  void unifLinsysPrepHost();
  // Host-side variables for linear system
  std::vector<double> h_cooValA_;
  std::vector<int> h_cooRowA_;
  std::vector<int> h_cooColA_;
  std::vector<double> h_x_;
  std::vector<double> h_b_;

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

};
