//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
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
  bPing = false;
}

void SimulationData::allocateGrid()
{
  int levelStart = levelMax-1;
  int aux = pow(2,levelStart);

  ScalarLab dummy;
  const bool xperiodic = dummy.is_xperiodic();
  const bool yperiodic = dummy.is_yperiodic();
  const bool zperiodic = dummy.is_zperiodic();

  chi   = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  vel   = new VectorGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  pres  = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  pOld  = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  pRHS  = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  invRho= new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  tmpV  = new VectorGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  tmp   = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  uDef  = new VectorGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  dump  = new DumpGrid  (bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);

  #ifdef PRECOND
  z_cg = new ScalarGrid(bpdx, bpdy, 1, extent,levelStart,levelMax,true,xperiodic,yperiodic,zperiodic);
  #endif

  const std::vector<BlockInfo>& velInfo = vel->getBlocksInfo();

  //assume all blockinfos have same h at the start!!!
  extents[0] = aux * bpdx * velInfo[0].h_gridpoint * VectorBlock::sizeX;
  extents[1] = aux * bpdy * velInfo[0].h_gridpoint * VectorBlock::sizeY;
  // printf("Extents %e %e (%e)\n", extents[0], extents[1], extent);
}

void SimulationData::dumpGlue(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerGlue, float, DumpGrid>(*(dump), time,
    "velChi_" + ss.str(), path4serialization);
}
void SimulationData::dumpChi(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerScalar, float, ScalarGrid>(*(chi), time,
    "chi_" + ss.str(), path4serialization);
}
void SimulationData::dumpPres(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerScalar, float, ScalarGrid>(*(pres), time,
    "pres_" + ss.str(), path4serialization);
}
void SimulationData::dumpPrhs(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerScalar, float, ScalarGrid>(*(pRHS), time,
    "pRHS_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmp(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerScalar, float, ScalarGrid>(*(tmp), time,
    "tmp_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmp2(std::string name) {
  DumpHDF5<StreamerScalar, float, ScalarGrid>(*(tmp), time,
    "tmp_" + name, path4serialization);
}
void SimulationData::dumpVel(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerVector, float, VectorGrid>(*(vel), time,
    "vel_" + ss.str(), path4serialization);
}
void SimulationData::dumpUobj(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerVector, float, VectorGrid>(*(uDef), time,
    "uobj_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmpV(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerVector, float, VectorGrid>(*(tmpV), time,
    "tmpV_" + ss.str(), path4serialization);
}
void SimulationData::dumpInvRho(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerScalar, float, ScalarGrid>(*(invRho), time,
    "invRho_" + ss.str(), path4serialization);
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

void SimulationData::checkVariableDensity()
{
  bVariableDensity = false;
  for(const auto& shape : shapes)
    bVariableDensity = bVariableDensity || shape->bVariableDensity();
  if( verbose ){
    if( bVariableDensity) std::cout << "[CUP2D] Shape with variable density found\n";
    if(!bVariableDensity) std::cout << "[CUP2D] No shape with variable density found\n";
  }
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
  if(pRHS not_eq nullptr) delete pRHS;
  if(tmpV not_eq nullptr) delete tmpV;
  if(invRho not_eq nullptr) delete invRho;
  if(pOld not_eq nullptr) delete pOld;
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
  Checker check (*this);
  check.run("before" + name);
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
  const std::vector<BlockInfo>& chiInfo = chi->getBlocksInfo();
  const std::vector<BlockInfo>& velInfo = vel->getBlocksInfo();
  const std::vector<BlockInfo>& dmpInfo =dump->getBlocksInfo();
  //const auto K1 = computeVorticity(*this); K1.run(); // uncomment to dump vorticity
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < velInfo.size(); i++)
  {
    VectorBlock* VEL = (VectorBlock*) velInfo[i].ptrBlock;
    ScalarBlock* CHI = (ScalarBlock*) chiInfo[i].ptrBlock;
    VelChiGlueBlock& DMP = * (VelChiGlueBlock*) dmpInfo[i].ptrBlock;
    DMP.assign(CHI, VEL);
  }

  // dump vorticity
  const auto K1 = computeVorticity(*this); K1.run();
  dumpTmp (name);

  //dumpChi  (name); // glued together: skip
  //dumpVel  (name); // glued together: skip
  dumpGlue(name);
  dumpPres(name);
  //dumpInvRho(name);
  //dumpUobj (name);
  //dumpForce(name);
  //dumpTmpV (name); // probably useless
  stopProfiler();
}
