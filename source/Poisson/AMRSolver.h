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

 public:
  void solve();

  std::string getName() {
    return "AMRSolver";
  }

  AMRSolver(SimulationData& s);

  cubism::FluxCorrection<ScalarGrid,ScalarBlock> Corrector;
  
  using bV = std::vector<cubism::BlockInfo>;

  void Update_Vector (bV & aInfo, bV & bInfo, double c, bV & dInfo);
  void Update_Vector1(bV & aInfo, double c, bV & dInfo);
  void Dot_Product(bV & aInfo, bV & bInfo, double & result);
  void Get_LHS (ScalarGrid * lhs, ScalarGrid * x);

  void cub2rhs(const std::vector<cubism::BlockInfo>& BSRC);

  #ifdef PRECOND
  std::vector<std::vector<double>> Ld;
  std::vector <  std::vector <std::vector< std::pair<int,double> > > >L_row;
  std::vector <  std::vector <std::vector< std::pair<int,double> > > >L_col;
  double getA(int I1, int I2);
  void FindZ(std::vector<cubism::BlockInfo> & zInfo,std::vector<cubism::BlockInfo> & rInfo);
  #endif
};
