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

  void operator()(const Real dt);

  std::string getName() {
    return "IC";
  }
};

class ApplyObjVel : public Operator
{
  public:
  ApplyObjVel(SimulationData& s) : Operator(s) { }

  void operator()(const Real dt);

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
    const Real i2h = 0.5/info.h;
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    for(int y=0; y<VectorBlock::sizeY; ++y)
    for(int x=0; x<VectorBlock::sizeX; ++x)
      TMP(x,y).s = i2h * ((lab(x,y-1).u[0]-lab(x,y+1).u[0]) + (lab(x+1,y).u[1]-lab(x-1,y).u[1]));
  }
};

class computeVorticity : public Operator
{
 public:
  computeVorticity(SimulationData& s) : Operator(s){ }

  void operator()(const Real dt)
  {
    const KernelVorticity mykernel(sim);
    compute<KernelVorticity,VectorGrid,VectorLab>(mykernel,*sim.vel,false);
    Real maxv = -1e10;
    Real minv = -1e10;
    for (auto & info: sim.tmp->getBlocksInfo())
    {
      auto & TMP = *(ScalarBlock*) info.ptrBlock;
      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
      {
        maxv = std::max(maxv, TMP(x,y).s);
        minv = std::max(minv,-TMP(x,y).s);
      }
    }
    Real buffer[2] = {maxv,minv};
    Real recvbuf[2];
    MPI_Reduce(buffer,recvbuf, 2, MPI_Real, MPI_MAX, 0, sim.chi->getCartComm());
    recvbuf[1]=-recvbuf[1];
    if (sim.rank == 0)
      std::cout << " max(omega)=" << recvbuf[0] << " min(omega)=" << recvbuf[1] << " max(omega)+min(omega)=" << recvbuf[0]+recvbuf[1] << std::endl;
  }

  std::string getName()
  {
    return "computeVorticity";
  }
};
