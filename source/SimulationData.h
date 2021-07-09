//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "Definitions.h"
#include "Cubism/Profiler.h"

class Shape;

struct SimulationData
{
  /* parsed parameters */
  /*********************/

  bool bRestart;

  // blocks per dimension
  int bpdx;
  int bpdy;

  // number of levels
  int levelMax;

  // initial level
  int levelStart;

  // refinement/compression tolerance for voriticy magnitude
  double Rtol;
  double Ctol;

  // boolean to switch between refinement according to chi or grad(chi)
  bool bAdaptChiGradient;

  // maximal simulation extent (direction with max(bpd))
  double extent;

  // simulation extents
  std::array<Real,2> extents;

  // timestep / cfl condition
  double dt;
  double CFL;

  // simulation ending parameters
  int nsteps;
  double endTime;

  // penalisation coefficient
  double lambda;

  // constant for explicit penalisation lambda=dlm/dt
  double dlm;

  // kinematic viscosity
  double nu;
  
  // poisson solver parameters
  double PoissonTol;    // absolute error tolerance
  double PoissonTolRel; // relative error tolerance

  // output setting
  int dumpFreq;
  double dumpTime;
  bool verbose;
  bool muteAll;
  std::string path4serialization;
  std::string path2file;

  /*********************/

  // initialize profiler
  cubism::Profiler * profiler = new cubism::Profiler();

  // declare grids
  ScalarGrid * chi   = nullptr;
  VectorGrid * vel   = nullptr;
  VectorGrid * vOld  = nullptr;
  ScalarGrid * pres  = nullptr;
  VectorGrid * tmpV  = nullptr;
  ScalarGrid * tmp   = nullptr;
  VectorGrid * uDef  = nullptr;

  // vector containing obstacles
  std::vector<Shape*> shapes;

  // simulation time
  double time = 0;
  bool Euler = false;

  // simulation step
  int step = 0;

  // velocity of simulation frame of reference
  Real uinfx = 0;
  Real uinfy = 0;
  Real uinfx_old = 0;
  Real uinfy_old = 0;
  double dt_old;

  // largest velocity measured
  double uMax_measured = 0;

  // gravity
  std::array<Real,2> gravity = { (Real) 0.0, (Real) -9.8 };

  // time of next dump
  double nextDumpTime = 0;

  // bools specifying whether we dump or not
  bool _bDump = false;
  bool DumpUniform = false;

  // bool for detecting collisions
  bool bCollision = false;

  void allocateGrid();
  void resetAll();
  bool bDump();
  void registerDump();
  bool bOver() const;

  double minRho() const;
  double maxSpeed() const;
  double maxRelSpeed() const;

  inline double getH() const
  {
    double minH = 1e50;
    auto & infos = vel->getBlocksInfo();
    for (size_t i = 0 ; i< infos.size(); i++)
    {
      minH = std::min(infos[i].h_gridpoint, minH);
    }
    return minH;
  }

  void startProfiler(std::string name);
  void stopProfiler();
  void printResetProfiler();
  ~SimulationData();

  void dumpChi   (std::string name);
  void dumpPres  (std::string name);
  void dumpTmp   (std::string name);
  void dumpVel   (std::string name);
  void dumpUobj  (std::string name);
  void dumpTmpV  (std::string name);
  void dumpAll   (std::string name);
};
