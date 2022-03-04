//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"

class Shape;

class PutObjectsOnGrid : public Operator
{
protected:
  const std::vector<cubism::BlockInfo>& velInfo   = sim.vel->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();

  void putChiOnGrid(Shape * const shape) const;
  void putObjectVelOnGrid(Shape * const shape) const;

 public:
  using Operator::Operator;

  void operator()(Real dt) override;
  void advanceShapes(Real dt);
  void putObjectsOnGrid();

  std::string getName() override
  {
    return "PutObjectsOnGrid";
  }
};
