//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#pragma once

#include "../Operator.h"

#include "HYPRE_struct_ls.h"

class LinearSolverDelta
{
 protected:
  SimulationData& sim;

  // memory buffer for mem transfers to/from solver:
  Real * buffer = nullptr; // rhs in cub2rhs, sol in sol2cub
  double * dbuffer;

  void cub2rhs(const std::vector<cubism::BlockInfo>& BSRC,const bool solveX);
  void sol2cub(const std::vector<cubism::BlockInfo>& BDST,const bool solveX);
  void allocLHS();

  HYPRE_StructGrid     hypre_grid;
  HYPRE_StructStencil  hypre_stencil;

  HYPRE_StructMatrix   hypre_mat_X; //LHS for x-velocities
  HYPRE_StructVector   hypre_rhs_X;
  HYPRE_StructVector   hypre_sol_X;
  HYPRE_StructSolver   hypre_solver_X;

  HYPRE_StructMatrix   hypre_mat_Y; //LHS for y-velocities
  HYPRE_StructVector   hypre_rhs_Y;
  HYPRE_StructVector   hypre_sol_Y;
  HYPRE_StructSolver   hypre_solver_Y;

  double * u;
  double * v;

 public:
  const size_t stride;
  const size_t totNy = sim.vel->getBlocksPerDimension(1) * VectorBlock::sizeY;
  const size_t totNx = sim.vel->getBlocksPerDimension(0) * VectorBlock::sizeX;

  LinearSolverDelta(SimulationData& s);

  void solve(const std::vector<cubism::BlockInfo>& infos, const bool solveX);
  using RowType = double[5];
  RowType * vals_x;
  RowType * vals_y;

  void cub2LHS(const std::vector<cubism::BlockInfo>& BSRC, const Real Uinf, const Real Vinf);

  ~LinearSolverDelta();
};