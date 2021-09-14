//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

class findMaxU
{
  SimulationData& sim;
  const std::vector<cubism::BlockInfo> & velInfo = sim.vel->getBlocksInfo();
 public:
  findMaxU(SimulationData& s) : sim(s) { }

  Real run() const;

  std::string getName() const {
    return "findMaxU";
  }
};

class Checker
{
  SimulationData& sim;
  const std::vector<cubism::BlockInfo> & velInfo = sim.vel->getBlocksInfo();
 public:
  Checker(SimulationData& s) : sim(s) { }

  void run(std::string when) const;

  std::string getName() const {
    return "Checker";
  }
};

class IC : public Operator
{
  public:
  IC(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  std::string getName() {
    return "IC";
  }
};

class ApplyObjVel : public Operator
{
  public:
  ApplyObjVel(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  std::string getName() {
    return "ApplyObjVel";
  }
};

struct KernelVorticity
{
  KernelVorticity(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0,1}};
  void operator()(VectorLab & lab, const cubism::BlockInfo& info) const
  {
    const double i2h = 0.5/info.h;
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    for(int y=0; y<VectorBlock::sizeY; ++y)
    for(int x=0; x<VectorBlock::sizeX; ++x)
      TMP(x,y).s = i2h * (lab(x,y-1).u[0]-lab(x,y+1).u[0] + lab(x+1,y).u[1]-lab(x-1,y).u[1]);
  }
};

class computeVorticity : public Operator
{
 public:
  computeVorticity(SimulationData& s) : Operator(s){ }

  void operator()(const double dt)
  {
    const KernelVorticity mykernel(sim);
    compute<KernelVorticity,VectorGrid,VectorLab>(mykernel,*sim.vel,false);
  }

  std::string getName()
  {
    return "computeVorticity";
  }
};