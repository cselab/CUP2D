//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "../Operator.h"

class Shape;

class PutObjectsOnGrid : public Operator
{
  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& invRhoInfo   = sim.invRho->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& forceInfo = sim.force->getBlocksInfo();

  void putChiOnGrid(Shape * const shape) const;
  void putObjectVelOnGrid(Shape * const shape) const;

 public:
  PutObjectsOnGrid(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  std::string getName()
  {
    return "PutObjectsOnGrid";
  }
};
