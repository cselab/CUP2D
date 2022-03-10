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
  ScalarAMR * tmp_amr;
  ScalarAMR * chi_amr;
  ScalarAMR * pres_amr;
  ScalarAMR * pold_amr;
  VectorAMR * vel_amr;
  VectorAMR * vOld_amr;
  VectorAMR * tmpV_amr;
  VectorAMR * uDef_amr;

  AdaptTheMesh(SimulationData& s) : Operator(s)
  {
    tmp_amr  = new ScalarAMR(*sim.tmp ,sim.Rtol,sim.Ctol);
    chi_amr  = new ScalarAMR(*sim.chi ,sim.Rtol,sim.Ctol);
    pres_amr = new ScalarAMR(*sim.pres,sim.Rtol,sim.Ctol);
    pold_amr = new ScalarAMR(*sim.pold,sim.Rtol,sim.Ctol);
    vel_amr  = new VectorAMR(*sim.vel ,sim.Rtol,sim.Ctol);
    vOld_amr = new VectorAMR(*sim.vOld,sim.Rtol,sim.Ctol);
    tmpV_amr = new VectorAMR(*sim.tmpV,sim.Rtol,sim.Ctol);
    uDef_amr = new VectorAMR(*sim.uDef,sim.Rtol,sim.Ctol);
  }

  ~AdaptTheMesh()
  {
    delete tmp_amr;
    delete chi_amr;
    delete pres_amr;
    delete pold_amr;
    delete vel_amr;
    delete vOld_amr;
    delete tmpV_amr;
    delete uDef_amr;
  }

  void operator() (const Real dt) override;
  void adapt();

  std::string getName() override
  {
    return "AdaptTheMesh";
  }
};
