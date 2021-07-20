//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Definitions.h"
#include "Shape.h"
#include "Operators/Helpers.h"
#include <Cubism/HDF5Dumper.h>

#include <iomanip>
using namespace cubism;

void SimulationData::resetAll()
{
  for(const auto& shape : shapes) shape->resetAll();
  time = 0;
  step = 0;
  uinfx = 0;
  uinfy = 0;
  nextDumpTime = 0;
  _bDump = false;
  bCollision = false;
}

void SimulationData::allocateGrid()
{
  ScalarLab dummy;
  const bool xperiodic = dummy.is_xperiodic();
  const bool yperiodic = dummy.is_yperiodic();
  const bool zperiodic = dummy.is_zperiodic();

  chi   = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  vel   = new VectorGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  vOld  = new VectorGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  pres  = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  tmpV  = new VectorGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  tmp   = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  uDef  = new VectorGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  pold  = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);

  const std::vector<BlockInfo>& velInfo = vel->getBlocksInfo();

  // Compute extents, assume all blockinfos have same h at the start!!!
  int aux = pow(2,levelStart);
  extents[0] = aux * bpdx * velInfo[0].h_gridpoint * VectorBlock::sizeX;
  extents[1] = aux * bpdy * velInfo[0].h_gridpoint * VectorBlock::sizeY;
  // printf("Extents %e %e (%e)\n", extents[0], extents[1], extent);

  // compute min and max gridspacing for set AMR parameter
  int auxMax = pow(2,levelMax-1);
  minH = extents[0] / (auxMax*bpdx*VectorBlock::sizeX);
  maxH = extents[0] / (bpdx*VectorBlock::sizeX);
}

void SimulationData::dumpChi(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_groups<StreamerScalar, float, ScalarGrid>(*(chi), time,"chi_" + ss.str(), path4serialization);
 if (DumpUniform) 
    DumpHDF5_uniform<StreamerScalar, float, ScalarGrid>(*(chi), time,"chi_" + ss.str(), path4serialization);
}
void SimulationData::dumpPres(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_groups<StreamerScalar, float, ScalarGrid>(*(pres), time,"pres_" + ss.str(), path4serialization);
  if (DumpUniform) 
    DumpHDF5_uniform<StreamerScalar, float, ScalarGrid>(*(pres), time,"pres_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmp(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_groups<StreamerScalar, float, ScalarGrid>(*(tmp), time,"tmp_" + ss.str(), path4serialization);
  if (DumpUniform)
    DumpHDF5_uniform<StreamerScalar, float, ScalarGrid>(*(tmp), time,"tmp_" + ss.str(), path4serialization);
}
void SimulationData::dumpVel(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerVector, float, VectorGrid>(*(vel), time,"vel_" + ss.str(), path4serialization);
  if (DumpUniform)
    DumpHDF5_uniform<StreamerVector, float, VectorGrid>(*(vel), time,"vel_" + ss.str(), path4serialization);
}
void SimulationData::dumpUobj(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerVector, float, VectorGrid>(*(uDef), time,"uobj_" + ss.str(), path4serialization);
  if (DumpUniform)
    DumpHDF5_uniform<StreamerVector, float, VectorGrid>(*(uDef), time,"uobj_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmpV(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerVector, float, VectorGrid>(*(tmpV), time,"tmpV_" + ss.str(), path4serialization);
  if (DumpUniform)
    DumpHDF5_uniform<StreamerVector, float, VectorGrid>(*(tmpV), time,"tmpV_" + ss.str(), path4serialization);
}

void SimulationData::registerDump()
{
  nextDumpTime += dumpTime;
}

double SimulationData::minRho() const
{
  double minR = 1; // fluid is 1
  for(const auto& shape : shapes)
    minR = std::min( (double) shape->getMinRhoS(), minR );
  return minR;
}

double SimulationData::maxSpeed() const
{
  double maxS = 0;
  for(const auto& shape : shapes) {
    maxS = std::max(maxS, (double) shape->getMaxVel() );
  }
  return maxS;
}

double SimulationData::maxRelSpeed() const
{
  double maxS = 0;
  for(const auto& shape : shapes)
    maxS = std::max(maxS, (double) shape->getMaxVel() );
  return maxS;
}

SimulationData::~SimulationData()
{
  #ifndef SMARTIES_APP
    delete profiler;
  #endif
  if(vel not_eq nullptr) delete vel;
  if(chi not_eq nullptr) delete chi;
  if(uDef not_eq nullptr) delete uDef;
  if(pres not_eq nullptr) delete pres;
  if(pold not_eq nullptr) delete pold;
  if(vOld not_eq nullptr) delete vOld;
  if(tmpV not_eq nullptr) delete tmpV;
  if(tmp not_eq nullptr) delete tmp;
  while( not shapes.empty() ) {
    Shape * s = shapes.back();
    if(s not_eq nullptr) delete s;
    shapes.pop_back();
  }
}

bool SimulationData::bOver() const
{
  const bool timeEnd = endTime>0 && time >= endTime;
  const bool stepEnd =  nsteps>0 && step > nsteps;
  return timeEnd || stepEnd;
}

bool SimulationData::bDump()
{
  const bool timeDump = dumpTime>0 && time >= nextDumpTime;
  const bool stepDump = dumpFreq>0 && (step % dumpFreq) == 0;
  _bDump = stepDump || timeDump;
  return _bDump;
}

void SimulationData::startProfiler(std::string name)
{
 #ifndef NDEBUG
  Checker check (*this);
  check.run("before" + name);
 #endif
  profiler->push_start(name);
}

void SimulationData::stopProfiler()
{
    //Checker check (*this);
    //check.run("after" + profiler->currentAgentName());
    profiler->pop_stop();
}

void SimulationData::printResetProfiler()
{
    profiler->printSummary();
    profiler->reset();
}

void SimulationData::dumpAll(std::string name)
{
  startProfiler("Dump");

  // dump vorticity
  const auto K1 = computeVorticity(*this); K1.run();
  dumpTmp (name);
  dumpChi  (name);
  //dumpVel  (name);
  dumpPres(name);
  //dumpUobj (name);
  //dumpTmpV (name); // probably useless
  stopProfiler();
}
