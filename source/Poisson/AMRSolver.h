//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

class AMRSolver 
{
 protected:
  SimulationData& sim;
  size_t iter_min{30};
 public:
  std::string getName() {
    return "AMRSolver";
  }
  AMRSolver(SimulationData& s);
  cubism::FluxCorrection<ScalarGrid,ScalarBlock> Corrector;
  void solve();
  void Get_LHS (ScalarGrid * lhs, ScalarGrid * x);
  void Jacobi (int iter_max);

  std::vector<std::vector<double>> Ld;
  std::vector <  std::vector <std::vector< std::pair<int,double> > > >L_row;
  std::vector <  std::vector <std::vector< std::pair<int,double> > > >L_col;
  double getA(int I1, int I2);
  void getZ(std::vector<cubism::BlockInfo> & zInfo);
  double getA_local(int I1,int I2);
};
