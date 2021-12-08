//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

class AMRSolver 
{
  /*
  Method used to solve Poisson's equation: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
  */
 protected:
  //this struct contains information such as the currect timestep size, fluid properties and many others
  SimulationData& sim; 
 public:
  std::string getName() {
    return "AMRSolver";
  }
  AMRSolver(SimulationData& s);

  //main function used to solve Poisson's equation
  void solve();

  //this is called to compute Ax, where x is the current solution estimate
  void Get_LHS ();

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
  std::vector<std::vector<double>> Ld;
  std::vector <  std::vector <std::vector< std::pair<int,double> > > >L_row;
  std::vector <  std::vector <std::vector< std::pair<int,double> > > >L_col;

  //This returns element K_{I1,I2}. It is used when we invert K
  double getA_local(int I1,int I2);

  //Given a vector z, compute (K_2)^{-1} z, i.e. apply the preconditioner
  void getZ(std::vector<cubism::BlockInfo> & zInfo);
};
