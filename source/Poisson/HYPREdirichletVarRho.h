//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once
#include "PoissonSolver.h"

#ifdef HYPREFFT
#include "HYPRE_struct_ls.h"
#endif

class HYPREdirichletVarRho : public PoissonSolver
{
  const bool bPeriodic = false;
  const std::string solver;
  #ifdef HYPREFFT
  HYPRE_StructGrid     hypre_grid;
  HYPRE_StructStencil  hypre_stencil;
  HYPRE_StructMatrix   hypre_mat;
  HYPRE_StructVector   hypre_rhs;
  HYPRE_StructVector   hypre_sol;
  HYPRE_StructSolver   hypre_solver;
  HYPRE_StructSolver   hypre_precond;
  #endif
  double * dbuffer;

 public:
  bool bUpdateMat = true;
  using RowType = double[5];
  RowType * matAry = new RowType[totNy*totNx];

  void solve(const std::vector<cubism::BlockInfo>& BSRC,
             const std::vector<cubism::BlockInfo>& BDST) override;

  HYPREdirichletVarRho(SimulationData& s);

  std::string getName() {
    return "hypre";
  }

  Real updateMat(const size_t idx, const size_t idy, Real coefDiag,
                 Real coefW, Real coefE, Real coefS, Real coefN) const
  {
    const size_t ind = stride * idy + idx;
    if(idx == 0) {       // first west col : Neuman BC
      coefDiag += coefW; coefW = 0;
    }
    if(idx == totNx-1) { // first east col : Neuman BC
      coefDiag += coefE; coefE = 0;
    }
    if(idy == 0) {       // first south row : Neuman BC
      coefDiag += coefS; coefS = 0;
    }
    if(idy == totNy-1) { // first north row : Neuman BC
      coefDiag += coefN; coefN = 0;
    }
    const Real dMat = std::pow(matAry[ind][0] - coefDiag, 2)
                    + std::pow(matAry[ind][1] - coefW,    2)
                    + std::pow(matAry[ind][2] - coefE,    2)
                    + std::pow(matAry[ind][3] - coefS,    2)
                    + std::pow(matAry[ind][4] - coefN,    2);
    matAry[ind][0] = coefDiag;
    matAry[ind][1] = coefW; matAry[ind][2] = coefE;
    matAry[ind][3] = coefS; matAry[ind][4] = coefN;
    return dMat;
  }

  ~HYPREdirichletVarRho() override;
};
