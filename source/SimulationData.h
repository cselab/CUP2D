//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Definitions.h"
#include "Cubism/Profiler.h"

class Shape;

struct SimulationData
{
  // MPI
  MPI_Comm comm;
  int rank;

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
  int maxPoissonRestarts; // maximal number of restarts of Poisson solver
  int maxPoissonIterations; // maximal number of iterations of Poisson solver
  bool bMeanConstraint; // regularizing the poisson equation using the mean

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
  ScalarGrid * pold  = nullptr;

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

  // minimal and maximal gridspacing possible
  double minH;
  double maxH;

  // minimal gridspacing present on grid
  double getH()
  {
    double minHGrid = std::numeric_limits<double>::infinity();
    auto & infos = vel->getBlocksInfo();
    for (size_t i = 0 ; i< infos.size(); i++)
    {
      minHGrid = std::min(infos[i].h_gridpoint, minHGrid);
    }
    MPI_Allreduce(MPI_IN_PLACE, &minHGrid, 1, MPI_DOUBLE, MPI_MIN, comm);
    return minHGrid;
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
