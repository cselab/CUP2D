//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once
#include "PoissonSolver.h"

#ifdef AMGX_POISSON
#ifdef AMGX_DYNAMIC_LOADING
#include "amgx_capi.h"
#else
#include "amgx_c.h"
#endif
#endif

class HYPREdirichletVarRho : public PoissonSolver
{
  #ifdef AMGX_POISSON
    AMGX_Mode mode;
    AMGX_resources_handle workspace;
    AMGX_solver_handle solver;
    AMGX_config_handle config;
    AMGX_matrix_handle mat;
    AMGX_vector_handle rhs;
    AMGX_vector_handle sol;
    //status handling
    AMGX_SOLVE_STATUS status;
  #endif
  const bool bPeriodic = false;
  const std::string solver;
  int * row_ptrs = nullptr;
  int64_t * col_indices = nullptr;
  amgx_val_t * dbuffer;

 public:
  bool bUpdateMat = true;
  amgx_val_t * matAry = nullptr;

  Real updateMat(const size_t idx, const size_t idy, Real coefDiag,
                 Real coefW, Real coefE, Real coefS, Real coefN) const
  {
    const size_t ind = stride * idy + idx;
    const int row_ptr = row_ptrs[ind];
    int iW = 1, iC = 2, iE = 3, iN = 4;
    Real dMat = 0;

    if(idy == 0)       { // first south row : Neuman BC
      coefDiag += coefS; coefS = 0;
      iN = 3; iE = 2; iC = 1; iW = 0;
    } else {
      dMat += std::pow(matAry[row_ptr + 0] - coefS, 2);
      matAry[row_ptr + 0] = coefS;
    }

    if(idx == 0)       { // first west col : Neuman BC
      coefDiag += coefW; coefW = 0;
      iN = iE; iE = iC; iC = iW;
    } else {
      dMat += std::pow(matAry[row_ptr + iW] - coefW, 2);
      matAry[row_ptr + iW] = coefW;
    }

    if(idx == totNx-1) { // first east col : Neuman BC
      coefDiag += coefE; coefE = 0;
      iN = iE;
    } else {
      dMat += std::pow(matAry[row_ptr + iE] - coefE, 2);
      matAry[row_ptr + iE] = coefE;
    }

    if(idy == totNy-1) { // first north row : Neuman BC
      coefDiag += coefN; coefN = 0;
    } else {
      dMat += std::pow(matAry[row_ptr + iN] - coefN, 2);
      matAry[row_ptr + iN] = coefN;
    }

    dMat += std::pow(matAry[row_ptr + iC] - coefDiag, 2);
    matAry[row_ptr + iC] = coefDiag;
    return dMat;
  }

  void solve(const std::vector<cubism::BlockInfo>& BSRC,
             const std::vector<cubism::BlockInfo>& BDST) override;

  HYPREdirichletVarRho(SimulationData& s);

  std::string getName() {
    return "hypre";
  }

  ~HYPREdirichletVarRho() override;
};
