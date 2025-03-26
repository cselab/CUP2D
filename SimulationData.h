

#pragma once

#include "Cubism/Profiler.h"
#include "Definitions.h"
#include <memory>

class Shape;

struct SimulationData {

  MPI_Comm comm;
  int rank;

  bool bRestart;

  int bpdx;
  int bpdy;

  int levelMax;

  int levelStart;

  Real Rtol;
  Real Ctol;

  bool Qcriterion{false};

  int AdaptSteps{20};

  bool bAdaptChiGradient;

  Real extent;

  std::array<Real, 2> extents;

  Real dt;
  Real CFL;
  int rampup{0};

  int nsteps;
  Real endTime;

  Real lambda;

  Real dlm;

  Real nu;

  bool bForcing;
  Real forcingWavenumber;
  Real forcingCoefficient;

  Real smagorinskyCoeff;

  std::string ic;

  std::string poissonSolver;
  Real PoissonTol;
  Real PoissonTolRel;
  int maxPoissonRestarts;
  int maxPoissonIterations;
  int bMeanConstraint;

  int profilerFreq = 0;
  int dumpFreq;
  Real dumpTime;
  bool verbose;
  bool muteAll;
  std::string path4serialization;
  std::string path2file;

  cubism::Profiler *profiler = new cubism::Profiler();

  ScalarGrid *chi = nullptr;
  VectorGrid *vel = nullptr;
  VectorGrid *vOld = nullptr;
  ScalarGrid *pres = nullptr;
  VectorGrid *tmpV = nullptr;
  ScalarGrid *tmp = nullptr;
  ScalarGrid *pold = nullptr;
  ScalarGrid *Cs = nullptr;

  std::vector<std::shared_ptr<Shape>> shapes;

  Real time = 0;

  int step = 0;

  Real uinfx = 0;
  Real uinfy = 0;
  Real uinfx_old = 0;
  Real uinfy_old = 0;
  Real dt_old = 1e10;
  Real dt_old2 = 1e10;

  Real uMax_measured = 0;

  Real nextDumpTime = 0;

  bool _bDump = false;
  bool DumpUniform = false;
  bool bDumpCs = false;

  bool bCollision = false;
  std::vector<int> bCollisionID;

  void addShape(std::shared_ptr<Shape> shape);

  void allocateGrid();
  void resetAll();
  bool bDump();
  void registerDump();
  bool bOver() const;

  Real minH;
  Real maxH;

  SimulationData();
  SimulationData(const SimulationData &) = delete;
  SimulationData(SimulationData &&) = delete;
  SimulationData &operator=(const SimulationData &) = delete;
  SimulationData &operator=(SimulationData &&) = delete;
  ~SimulationData();

  Real getH() {
    Real minHGrid = std::numeric_limits<Real>::infinity();
    auto &infos = vel->getBlocksInfo();
    for (size_t i = 0; i < infos.size(); i++) {
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

  void dumpChi(std::string name);
  void dumpPres(std::string name);
  void dumpTmp(std::string name);
  void dumpVel(std::string name);
  void dumpUdef(std::string name);
  void dumpVold(std::string name);
  void dumpPold(std::string name);
  void dumpTmpV(std::string name);
  void dumpCs(std::string name);
  void dumpAll(std::string name);
};
