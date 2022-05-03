//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Definitions.h"
#include "Shape.h"
#include "Operators/Helpers.h"
#include <Cubism/HDF5Dumper.h>
#include <Cubism/HDF5Dumper_MPI.h>

#include <iomanip>
using namespace cubism;

void SimulationData::addShape(std::shared_ptr<Shape> shape) {
  shape->obstacleID = (unsigned)shapes.size();
  shapes.push_back(std::move(shape));
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
  bCollision = false;
}

void SimulationData::allocateGrid()
{
  ScalarLab dummy;
  const bool xperiodic = dummy.is_xperiodic();
  const bool yperiodic = dummy.is_yperiodic();
  const bool zperiodic = dummy.is_zperiodic();
  
  chi  = new ScalarGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);
  vel  = new VectorGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);
  vOld = new VectorGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);
  pres = new ScalarGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);
  tmpV = new VectorGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);
  tmp  = new ScalarGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);
  uDef = new VectorGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);
  pold = new ScalarGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);

  // For RL SGS learning
  Cs = new ScalarGrid (1,1,1,bpdx,bpdy,1,extent,levelStart,levelMax,comm,xperiodic,yperiodic,zperiodic);

  const std::vector<BlockInfo>& velInfo = vel->getBlocksInfo();

  if (velInfo.size() == 0)
  {
    std::cout << "You are using too many MPI ranks for the given initial number of blocks.";
    std::cout << "Either increase levelStart or reduce the number of ranks." << std::endl;
    MPI_Abort(chi->getWorldComm(),1);
  }
  // Compute extents, assume all blockinfos have same h at the start!!!
  int aux = pow(2,levelStart);
  extents[0] = aux * bpdx * velInfo[0].h * VectorBlock::sizeX;
  extents[1] = aux * bpdy * velInfo[0].h * VectorBlock::sizeY;
  // printf("Extents %e %e (%e)\n", extents[0], extents[1], extent);

  // compute min and max gridspacing for set AMR parameter
  int auxMax = pow(2,levelMax-1);
  minH = extents[0] / (auxMax*bpdx*VectorBlock::sizeX);
  maxH = extents[0] / (bpdx*VectorBlock::sizeX);
}

void SimulationData::dumpChi(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerScalar,Real, ScalarGrid,ScalarLab>(*chi, time, "chi_" + ss.str(),path4serialization);
}
void SimulationData::dumpPres(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerScalar,Real, ScalarGrid,ScalarLab>(*pres, time, "pres_" + ss.str(),path4serialization);
}
void SimulationData::dumpPold(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerScalar,Real, ScalarGrid,ScalarLab>(*pold, time, "pold_" + ss.str(),path4serialization);
}
void SimulationData::dumpTmp(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerScalar,Real, ScalarGrid,ScalarLab>(*tmp, time, "tmp_" + ss.str(),path4serialization);
}
void SimulationData::dumpVel(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerVector, Real,  VectorGrid, VectorLab>(*(vel), time,"vel_" + ss.str(), path4serialization);
}
void SimulationData::dumpVold(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerVector, Real, VectorGrid, VectorLab>(*(vOld), time,"vOld_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmpV(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerVector, Real, VectorGrid, VectorLab>(*(tmpV), time,"tmpV_" + ss.str(), path4serialization);
}
void SimulationData::dumpUdef(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerVector, Real, VectorGrid, VectorLab>(*(uDef), time,"uDef_" + ss.str(), path4serialization);
}
void SimulationData::dumpCs(std::string name)
{
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerScalar,Real, ScalarGrid,ScalarLab>(*(Cs), time,"Cs_" + ss.str(), path4serialization);
}


void SimulationData::registerDump()
{
  nextDumpTime += dumpTime;
}

SimulationData::SimulationData() = default;

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
  if(Cs not_eq nullptr) delete Cs;
}

bool SimulationData::bOver() const
{
  const bool timeEnd = endTime>0 && time >= endTime;
  const bool stepEnd =  nsteps>0 && step >= nsteps;
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

  writeRestartFiles();

  auto K1 = computeVorticity(*this);
  K1(0);
  dumpTmp (name); //dump vorticity
  dumpChi (name);
  dumpVel (name);
  dumpPres(name);
  //dumpPold(name);
  //dumpUdef(name);
  //dumpTmpV(name);
  //dumpVold(name);
  if( bDumpCs )
    dumpCs(name);
  stopProfiler();
}

void SimulationData::writeRestartFiles()
{
  if (rank != 0) return;

  // write restart file for field
  {
     std::stringstream ssR;
     ssR << path4serialization + "/field.restart";
     FILE * fField = fopen(ssR.str().c_str(), "w");
     if (fField == NULL)
     {
        printf("Could not write %s. Aborting...\n", "field.restart");
        fflush(0); abort();
     }
     assert(fField != NULL);
     fprintf(fField, "time: %20.20e\n",  time);
     fprintf(fField, "stepid: %d\n",     step);
     fprintf(fField, "uinfx: %20.20e\n", uinfx);
     fprintf(fField, "uinfy: %20.20e\n", uinfy);
     fprintf(fField, "dt: %20.20e\n", dt);
     fclose(fField);
  }

  // write restart file for shapes
  {
     for(std::shared_ptr<Shape> shape : shapes)
     {
        std::stringstream ssR;
        ssR << path4serialization + "/shape_" << shape->obstacleID << ".restart";
        FILE * fShape = fopen(ssR.str().c_str(), "w");
        if (fShape == NULL)
	{
           printf("Could not write %s. Aborting...\n", ssR.str().c_str());
           fflush(0); abort();
        }
        shape->saveRestart( fShape );
        fclose(fShape);
     }
  }
}

void SimulationData::readRestartFiles()
{
  // read restart file for field
  FILE * fField = fopen("field.restart", "r");
  if (fField == NULL) {
    printf("Could not read %s. Aborting...\n", "field.restart");
    fflush(0); abort();
  }
  assert(fField != NULL);
  if (rank == 0 && verbose) printf("Reading %s...\n", "field.restart");
  bool ret = true;
  ret = ret && 1==fscanf(fField, "time: %le\n",   &time);
  ret = ret && 1==fscanf(fField, "stepid: %d\n",  &step);
  ret = ret && 1==fscanf(fField, "uinfx: %le\n",  &uinfx);
  ret = ret && 1==fscanf(fField, "uinfy: %le\n",  &uinfy);
  ret = ret && 1==fscanf(fField, "dt: %le\n",  &dt);
  fclose(fField);
  if( (not ret) || step<0 || time<0) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0); abort();
  }
  if (rank == 0 && verbose) printf("Restarting flow.. time: %le, stepid: %d, uinfx: %le, uinfy: %le\n", time, step, uinfx, uinfy);
  nextDumpTime = time + dumpTime;

  // read restart file for shapes
  for(std::shared_ptr<Shape> shape : shapes){
    std::stringstream ssR;
    ssR << "shape_" << shape->obstacleID << ".restart";
    FILE * fShape = fopen(ssR.str().c_str(), "r");
    if (fShape == NULL) {
      printf("Could not read %s. Aborting...\n", ssR.str().c_str());
      fflush(0); abort();
    }
    if (rank == 0 && verbose) printf("Reading %s...\n", ssR.str().c_str());
    shape->loadRestart( fShape );
    fclose(fShape);
  }
}
