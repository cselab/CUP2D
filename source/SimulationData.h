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

  // maximal simulation extent (direction with max(bpd))
  double extent;

  // simulation extents
  std::array<Real,2> extents;

  // cfl condition
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
  
  // char length for far-field BC
  Real fadeLenX, fadeLenY;

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
  ScalarGrid * pres  = nullptr;
  ScalarGrid * pOld  = nullptr;
  VectorGrid * tmpV  = nullptr;
  ScalarGrid * tmp   = nullptr;
  VectorGrid * uDef  = nullptr;
  DumpGrid   * dump  = nullptr;

  // vector containing obstacles
  std::vector<Shape*> shapes;

  // simulation time
  double time = 0;

  // timestep
  double dt = 0;

  // simulation step
  int step = 0;

  // velocity of simulation frame of reference
  Real uinfx = 0;
  Real uinfy = 0;

  // largest velocity measured
  double uMax_measured = 0;

  // gravity
  std::array<Real,2> gravity = { (Real) 0.0, (Real) -9.8 };

  // time of next dump
  double nextDumpTime = 0;

  // bools specifying whether we dump or not
  bool _bDump = false;

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
    //return vel->getBlocksInfo().front().h_gridpoint; // yikes
  }

  void startProfiler(std::string name);
  void stopProfiler();
  void printResetProfiler();
  ~SimulationData();

  void dumpChi   (std::string name);
  void dumpPres  (std::string name);
  void dumpPrhs  (std::string name);
  void dumpTmp   (std::string name);
  void dumpTmp2  (std::string name);
  void dumpVel   (std::string name);
  void dumpUobj  (std::string name);
  void dumpTmpV  (std::string name);
  void dumpAll   (std::string name);
  void dumpInvRho(std::string name);
  void dumpGlue  (std::string name);
};
