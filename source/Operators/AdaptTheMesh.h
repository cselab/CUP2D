//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"

class AdaptTheMesh : public Operator
{
 public:
  ScalarAMR * tmp_amr  = nullptr;
  ScalarAMR * chi_amr  = nullptr;
  ScalarAMR * pres_amr = nullptr;
  ScalarAMR * pold_amr = nullptr;
  VectorAMR * vel_amr  = nullptr;
  VectorAMR * vOld_amr = nullptr;
  VectorAMR * tmpV_amr = nullptr;
  ScalarAMR * Cs_amr   = nullptr;

  AdaptTheMesh(SimulationData& s) : Operator(s)
  {
    tmp_amr  = new ScalarAMR(*sim.tmp ,sim.Rtol,sim.Ctol);
    chi_amr  = new ScalarAMR(*sim.chi ,sim.Rtol,sim.Ctol);
    pres_amr = new ScalarAMR(*sim.pres,sim.Rtol,sim.Ctol);
    pold_amr = new ScalarAMR(*sim.pold,sim.Rtol,sim.Ctol);
    vel_amr  = new VectorAMR(*sim.vel ,sim.Rtol,sim.Ctol);
    vOld_amr = new VectorAMR(*sim.vOld,sim.Rtol,sim.Ctol);
    tmpV_amr = new VectorAMR(*sim.tmpV,sim.Rtol,sim.Ctol);
    if( sim.smagorinskyCoeff != 0 )
      Cs_amr = new ScalarAMR(*sim.Cs,sim.Rtol,sim.Ctol);
  }

  ~AdaptTheMesh()
  {
    if( tmp_amr  not_eq nullptr ) delete tmp_amr ;
    if( chi_amr  not_eq nullptr ) delete chi_amr ;
    if( pres_amr not_eq nullptr ) delete pres_amr;
    if( pold_amr not_eq nullptr ) delete pold_amr;
    if( vel_amr  not_eq nullptr ) delete vel_amr ;
    if( vOld_amr not_eq nullptr ) delete vOld_amr;
    if( tmpV_amr not_eq nullptr ) delete tmpV_amr;
    if( Cs_amr   not_eq nullptr ) delete Cs_amr  ;
  }

  void operator() (const Real dt) override;
  void adapt();

  std::string getName() override
  {
    return "AdaptTheMesh";
  }
};
