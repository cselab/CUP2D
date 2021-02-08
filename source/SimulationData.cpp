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

void SimulationData::allocateGrid()
{
  chi   = new ScalarGrid(bpdx, bpdy, 1, extent);
  vel   = new VectorGrid(bpdx, bpdy, 1, extent);
  pres  = new ScalarGrid(bpdx, bpdy, 1, extent);
  pOld  = new ScalarGrid(bpdx, bpdy, 1, extent);

  pRHS  = new ScalarGrid(bpdx, bpdy, 1, extent);
  invRho= new ScalarGrid(bpdx, bpdy, 1, extent);

  tmpV  = new VectorGrid(bpdx, bpdy, 1, extent);
  vFluid= new VectorGrid(bpdx, bpdy, 1, extent);
  tmp   = new ScalarGrid(bpdx, bpdy, 1, extent);
  uDef  = new VectorGrid(bpdx, bpdy, 1, extent);

  dump  = new DumpGrid(bpdx, bpdy, 1, extent);

  extents[0] = bpdx * vel->getH() * VectorBlock::sizeX;
  extents[1] = bpdy * vel->getH() * VectorBlock::sizeY;
  printf("Extents %e %e (%e)\n", extents[0], extents[1], extent);

  const auto isW = [&](const BlockInfo&I) { return I.index[0] == 0;      };
  const auto isE = [&](const BlockInfo&I) { return I.index[0] == bpdx-1; };
  const auto isS = [&](const BlockInfo&I) { return I.index[1] == 0;      };
  const auto isN = [&](const BlockInfo&I) { return I.index[1] == bpdy-1; };
  const std::vector<BlockInfo>& velInfo = vel->getBlocksInfo();

  for (size_t i=0; i < velInfo.size(); i++)
    if(isW(velInfo[i]) || isE(velInfo[i]) || isS(velInfo[i]) || isN(velInfo[i]))
      boundaryInfoIDs.push_back(i);
}

void SimulationData::writeRestartFiles(){
  // write restart file for field
  FILE * fField = fopen("field.restart", "w");
  if (fField == NULL) {
    printf("Could not write %s. Aborting...\n", "field.restart");
    fflush(0); abort();
  }
  assert(fField != NULL);
  fprintf(fField, "time: %20.20e\n",  time);
  fprintf(fField, "stepid: %d\n",     step);
  fprintf(fField, "uinfx: %20.20e\n", uinfx);
  fprintf(fField, "uinfy: %20.20e\n", uinfy);
  fclose(fField);

  // write restart file for shapes
  for(Shape * const shape : shapes){
    std::stringstream ssR;
    ssR << "shape_" << shape->obstacleID << ".restart";
    FILE * fShape = fopen(ssR.str().c_str(), "w");
    if (fShape == NULL) {
      printf("Could not write %s. Aborting...\n", ssR.str().c_str());
      fflush(0); abort();
    }
    shape->saveRestart( fShape );
    fclose(fShape);
  }
}

void SimulationData::readRestartFiles(){
  // read restart file for field
  FILE * fField = fopen("field.restart", "r");
  if (fField == NULL) {
    printf("Could not read %s. Aborting...\n", "field.restart");
    fflush(0); abort();
  }
  assert(fField != NULL);
  printf("Reading %s...\n", "field.restart");
  bool ret = true;
  ret = ret && 1==fscanf(fField, "time: %le\n",   &time);
  ret = ret && 1==fscanf(fField, "stepid: %d\n",  &step);
  ret = ret && 1==fscanf(fField, "uinfx: %le\n",  &uinfx);
  ret = ret && 1==fscanf(fField, "uinfy: %le\n",  &uinfy);
  fclose(fField);
  if( (not ret) || step<0 || time<0) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0); abort();
  }
  printf("Restarting flow.. time: %le, stepid: %d, uinfx: %le, uinfy: %le\n", time, step, uinfx, uinfy);
  nextDumpTime = time + dumpTime;

  // read restart file for shapes
  for(Shape * const shape : shapes){
    std::stringstream ssR;
    ssR << "shape_" << shape->obstacleID << ".restart";
    FILE * fShape = fopen(ssR.str().c_str(), "r");
    if (fShape == NULL) {
      printf("Could not read %s. Aborting...\n", ssR.str().c_str());
      fflush(0); abort();
    }
    printf("Reading %s...\n", ssR.str().c_str());
    shape->loadRestart( fShape );
    fclose(fShape);
  }
}


void SimulationData::dumpGlue(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<StreamerGlue, Real, DumpGrid>(*(dump), time,
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
  if( bVariableDensity) std::cout << "Using variable density solver\n";
  if(!bVariableDensity) std::cout << "Using constant density solver\n";
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
  if(vFluid not_eq nullptr) delete vFluid;
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
  //std::cout << name << std::endl;
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
  if(bStaggeredGrid)
  {
    const auto K1 = computeVorticity(*this); K1.run();
    dumpTmp (name);
    //const auto K2 = computeDivergence(*this); K2.run();
    //dumpTmp (name+"div");
    dumpPres (name);
    dumpInvRho (name);
  }
  else
  {
    writeRestartFiles();

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
    //dumpChi  (name); // glued together: skip
    //dumpVel  (name); // glued together: skip
    dumpGlue(name);
    dumpPres(name);
    //dumpInvRho(name);
    //dumpTmp (name);  // uncomment to dump vorticity
    //dumpUobj (name);
    //dumpForce(name);
    //dumpTmpV (name); // probably useless
  }
}
