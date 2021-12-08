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

class computeVorticity
{
  SimulationData& sim;
  const std::vector<cubism::BlockInfo> & velInfo = sim.vel->getBlocksInfo();
 public:
  computeVorticity(SimulationData& s) : sim(s) { }

  void run() const;

  std::string getName() const {
    return "computeVorticity";
  }
};


class computeDivergence
{
  SimulationData& sim;
  const std::vector<cubism::BlockInfo> & velInfo = sim.vel->getBlocksInfo();
 public:
  computeDivergence(SimulationData& s) : sim(s) { }

  void run();

  std::string getName() const {
    return "computeDivergence";
  }
};
