//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Definitions.h"
#include "Cubism/Profiler.h"
#include <memory>

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
  Real Rtol;
  Real Ctol;

  // boolean to switch between vorticity magnitude and Q-criterion refinement
  // the Q-criterion measures the difference between rotation rate and shear rate
  // Q > 0 indicates that there's a vortex
  // Q < 0 indicates a region where viscous forces are stronger
  // if Qcriterion=true, refinement will be done where Q>Rtol (Rtol>0)
  // Generally this results in less refinement, compared to refining according
  // to vorticity magnitude. For the cases this has been tested with there
  // was no loss of accuracy, despite the fact that the mesh that was refined
  // according to Q had about 1/4 of the points the other mesh had.
  bool Qcriterion{false};

  //check for mesh refinement every this many steps
  int AdaptSteps{20};

  // boolean to switch between refinement according to chi or grad(chi)
  bool bAdaptChiGradient;

  // maximal simulation extent (direction with max(bpd))
  Real extent;

  // simulation extents
  std::array<Real,2> extents;

  // timestep / cfl condition
  Real dt;
  Real CFL;
  int rampup{0};

  // simulation ending parameters
  int nsteps;
  Real endTime;

  // penalisation coefficient
  Real lambda;

  // constant for explicit penalisation lambda=dlm/dt
  Real dlm;

  // kinematic viscosity
  Real nu;
  
  // forcing
  bool bForcing;
  Real forcingWavenumber;
  Real forcingCoefficient;

  // Smagorinsky Model
  Real smagorinskyCoeff;

  // Flag for initial conditions
  std::string ic;

  // poisson solver parameters
  std::string poissonSolver;  // for now only "iterative"
  Real PoissonTol;    // absolute error tolerance
  Real PoissonTolRel; // relative error tolerance
  int maxPoissonRestarts; // maximal number of restarts of Poisson solver
  int maxPoissonIterations; // maximal number of iterations of Poisson solver
  int bMeanConstraint; // regularizing the poisson equation using the mean

  // output setting
  int profilerFreq = 0;
  int dumpFreq;
  Real dumpTime;
  bool verbose;
  bool muteAll;
  std::string path4serialization;
  std::string path2file;

  /*********************/

  // initialize profiler
  cubism::Profiler * profiler = new cubism::Profiler();

  // declare grids
  ScalarGrid * chi  = nullptr;
  VectorGrid * vel  = nullptr;
  VectorGrid * vOld = nullptr;
  ScalarGrid * pres = nullptr;
  VectorGrid * tmpV = nullptr;
  ScalarGrid * tmp  = nullptr;
  ScalarGrid * pold = nullptr;
  ScalarGrid * Cs   = nullptr;

  // vector containing obstacles
  std::vector<std::shared_ptr<Shape>> shapes;

  // simulation time
  Real time = 0;

  // simulation step
  int step = 0;

  // velocity of simulation frame of reference
  Real uinfx = 0;
  Real uinfy = 0;
  Real uinfx_old = 0;
  Real uinfy_old = 0;
  Real dt_old = 1e10;//need to initialize to a big value so that restarting does not
  Real dt_old2 = 1e10;//break when these are used in PressureSingle.cpp

  // largest velocity measured
  Real uMax_measured = 0;

  // time of next dump
  Real nextDumpTime = 0;

  // bools specifying whether we dump or not
  bool _bDump = false;
  bool DumpUniform = false;
  bool bDumpCs = false;

  // bool for detecting collisions
  bool bCollision = false;
  std::vector<int> bCollisionID;

  void addShape(std::shared_ptr<Shape> shape);

  void allocateGrid();
  void resetAll();
  bool bDump();
  void registerDump();
  bool bOver() const;

  // minimal and maximal gridspacing possible
  Real minH;
  Real maxH;

  SimulationData();
  SimulationData(const SimulationData &) = delete;
  SimulationData(SimulationData &&) = delete;
  SimulationData& operator=(const SimulationData &) = delete;
  SimulationData& operator=(SimulationData &&) = delete;
  ~SimulationData();

  // minimal gridspacing present on grid
  Real getH()
  {
    Real minHGrid = std::numeric_limits<Real>::infinity();
    auto & infos = vel->getBlocksInfo();
    for (size_t i = 0 ; i< infos.size(); i++)
    {
      minHGrid = std::min((Real)infos[i].h, minHGrid);
    }
    MPI_Allreduce(MPI_IN_PLACE, &minHGrid, 1, MPI_Real, MPI_MIN, comm);
    return minHGrid;
  }

  void startProfiler(std::string name);
  void stopProfiler();
  void printResetProfiler();

  void writeRestartFiles();
  void readRestartFiles();

  void dumpChi  (std::string name);
  void dumpPres (std::string name);
  void dumpTmp  (std::string name);
  void dumpVel  (std::string name);
  void dumpUdef (std::string name);
  void dumpVold (std::string name);
  void dumpPold (std::string name);
  void dumpTmpV (std::string name);
  void dumpCs   (std::string name);
  void dumpAll  (std::string name);
};
