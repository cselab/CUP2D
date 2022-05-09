//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Helpers.h"
#include "Cubism/HDF5Dumper_MPI.h"
#include <random>
#include "../Shape.h"
using namespace cubism;

void IC::operator()(const Real dt)
{
  const std::vector<BlockInfo>& chiInfo  = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& poldInfo = sim.pold->getBlocksInfo();
  const std::vector<BlockInfo>& invmInfo = sim.invm->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& tmpInfo  = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& vOldInfo = sim.vOld->getBlocksInfo();
  const std::vector<BlockInfo>& CsInfo   = sim.Cs->getBlocksInfo();
  if( not sim.bRestart )
  {
    #pragma omp parallel for
    for (size_t i=0; i < velInfo.size(); i++) ( (ScalarBlock*)   tmpInfo[i].ptrBlock )->set(-1);
    #pragma omp parallel for
    for(const auto& shape : sim.shapes) 
    	shape->create(tmpInfo);
    #pragma omp parallel for
    for (size_t i=0; i < velInfo.size(); i++)
    {
      VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;  VEL.clear();
      VectorBlock& UDEF= *(VectorBlock*) uDefInfo[i].ptrBlock; UDEF.clear();
      ScalarBlock& CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;  CHI.clear();
      ScalarBlock& PRES= *(ScalarBlock*) presInfo[i].ptrBlock; PRES.clear();
      VectorBlock& INVM= *(VectorBlock*) invmInfo[i].ptrBlock; INVM.clear();
      ScalarBlock& POLD= *(ScalarBlock*) poldInfo[i].ptrBlock; POLD.clear();
      ScalarBlock& TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;  
      VectorBlock& TMPV= *(VectorBlock*) tmpVInfo[i].ptrBlock; TMPV.clear();
      VectorBlock& VOLD= *(VectorBlock*) vOldInfo[i].ptrBlock; VOLD.clear();
      ScalarBlock& CS  = *(ScalarBlock*)   CsInfo[i].ptrBlock;
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
      {
         // Define inverse map only inside solids
	 if(TMP(ix,iy).s>0){
		double p[2];
		invmInfo[i].pos(p, ix, iy);
		INVM(ix, iy).u[0] = p[0];
		INVM(ix, iy).u[1] = p[1];
	}
	CS(ix,iy).s = sim.smagorinskyCoeff;
      }
      TMP.clear();
    }
  }
  else
  {
    // create filename from step
    sim.readRestartFiles();

    std::stringstream ss;
    ss<<"avemaria_"<<std::setfill('0')<<std::setw(7)<<sim.step;

    //The only field that is needed for restarting is velocity. Chi is derived from the files we
    //read for obstacles. Here we also read pres so that the Poisson solver has the same
    //initial guess, which in turn leads to restarted simulations having the exact same result
    //as non-restarted ones (we also read pres because we need to read at least
    //one ScalarGrid, see hack below).
    ReadHDF5_MPI<StreamerVector, Real, VectorGrid>(*(sim.vel ), "vel_"  + ss.str(), sim.path4serialization);
    ReadHDF5_MPI<StreamerScalar, Real, ScalarGrid>(*(sim.pres), "pres_" + ss.str(), sim.path4serialization);

    //hack: need to "read" the other grids too, so that the mesh is the same for every grid.
    //So we read VectorGrids from "vel" and ScalarGrids from "pres". We don't care about the
    //grid point values (those are set to zero below), we only care about the grid structure,
    //i.e. refinement levels etc.
    ReadHDF5_MPI<StreamerScalar, Real, ScalarGrid>(*(sim.pold), "pres_" + ss.str(), sim.path4serialization);
    ReadHDF5_MPI<StreamerScalar, Real, ScalarGrid>(*(sim.chi ), "pres_" + ss.str(), sim.path4serialization);
    ReadHDF5_MPI<StreamerScalar, Real, ScalarGrid>(*(sim.tmp ), "pres_" + ss.str(), sim.path4serialization);
    ReadHDF5_MPI<StreamerVector, Real, VectorGrid>(*(sim.tmpV), "vel_"  + ss.str(), sim.path4serialization);
    ReadHDF5_MPI<StreamerVector, Real, VectorGrid>(*(sim.uDef), "vel_"  + ss.str(), sim.path4serialization);
    ReadHDF5_MPI<StreamerVector, Real, VectorGrid>(*(sim.vOld), "vel_"  + ss.str(), sim.path4serialization);
    #pragma omp parallel for
    for (size_t i=0; i < velInfo.size(); i++)
    {
      ScalarBlock& CHI  = *(ScalarBlock*)  chiInfo[i].ptrBlock;  CHI.clear();
      ScalarBlock& POLD = *(ScalarBlock*) poldInfo[i].ptrBlock; POLD.clear();
      VectorBlock& UDEF = *(VectorBlock*) uDefInfo[i].ptrBlock; UDEF.clear();
      ScalarBlock& TMP  = *(ScalarBlock*)  tmpInfo[i].ptrBlock;  TMP.clear();
      VectorBlock& TMPV = *(VectorBlock*) tmpVInfo[i].ptrBlock; TMPV.clear();
      VectorBlock& VOLD = *(VectorBlock*) vOldInfo[i].ptrBlock; VOLD.clear();
      ScalarBlock& CS  = *(ScalarBlock*)   CsInfo[i].ptrBlock;
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
      {
        CS(ix,iy).s = sim.smagorinskyCoeff;
      }
    }
  }
}

void randomIC::operator()(const Real dt)
{
  const std::vector<BlockInfo>& chiInfo  = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& poldInfo = sim.pold->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
  const std::vector<BlockInfo>& tmpInfo  = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& vOldInfo = sim.vOld->getBlocksInfo();
  const std::vector<BlockInfo>& CsInfo   = sim.Cs->getBlocksInfo();

  #pragma omp parallel
  {
    std::random_device seed;
    std::mt19937 gen(seed());
    std::normal_distribution<Real> dist(0.0, 0.01);

    #pragma omp for
    for (size_t i=0; i < velInfo.size(); i++)
    {
      VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        VEL(ix,iy).u[0] = 0.5+dist(gen);
        VEL(ix,iy).u[1] = 0.5+dist(gen);
      }

      VectorBlock& UDEF= *(VectorBlock*) uDefInfo[i].ptrBlock; UDEF.clear();
      ScalarBlock& CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;  CHI.clear();
      ScalarBlock& PRES= *(ScalarBlock*) presInfo[i].ptrBlock; PRES.clear();
      ScalarBlock& POLD= *(ScalarBlock*) poldInfo[i].ptrBlock; POLD.clear();
      ScalarBlock& TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;  TMP.clear();
      VectorBlock& TMPV= *(VectorBlock*) tmpVInfo[i].ptrBlock; TMPV.clear();
      VectorBlock& VOLD= *(VectorBlock*) vOldInfo[i].ptrBlock; VOLD.clear();
      ScalarBlock& CS  = *(ScalarBlock*)   CsInfo[i].ptrBlock;
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
      {
        CS(ix,iy).s = sim.smagorinskyCoeff;
      }
    }
  }
}

Real findMaxU::run() const
{
  const size_t Nblocks = velInfo.size();

  const Real UINF = sim.uinfx, VINF = sim.uinfy;
  ///*
  #ifdef ZERO_TOTAL_MOM
  Real momX = 0, momY = 0, totM = 0; 
  #pragma omp parallel for schedule(static) reduction(+ : momX, momY, totM)
  for (size_t i=0; i < Nblocks; i++) {
    const Real h = velInfo[i].h;
    const VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      const Real facMom = h*h;
      momX += facMom * VEL(ix,iy).u[0];
      momY += facMom * VEL(ix,iy).u[1];
      totM += facMom;
    }
  }
  Real temp[3] = {momX,momY,totM};
  MPI_Allreduce(MPI_IN_PLACE, temp, 3, MPI_Real, MPI_SUM, sim.chi->getCartComm());
  momX = temp[0];
  momY = temp[1];
  totM = temp[2];
  //printf("Integral of momenta X:%e Y:%e mass:%e\n", momX, momY, totM);
  const Real DU = momX / totM, DV = momY / totM;
  #endif
  //*/
  Real U = 0, V = 0, u = 0, v = 0;
  #pragma omp parallel for schedule(static) reduction(max : U, V, u, v)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      #ifdef ZERO_TOTAL_MOM
        VEL(ix,iy).u[0] -= DU; VEL(ix,iy).u[1] -= DV;
      #endif
      U = std::max( U, std::fabs( VEL(ix,iy).u[0] + UINF ) );
      V = std::max( V, std::fabs( VEL(ix,iy).u[1] + VINF ) );
      u = std::max( u, std::fabs( VEL(ix,iy).u[0] ) );
      v = std::max( v, std::fabs( VEL(ix,iy).u[1] ) );
    }
  }
  Real quantities[4] = {U,V,u,v};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 4, MPI_Real, MPI_MAX, sim.chi->getCartComm());
  U = quantities[0];
  V = quantities[1];
  u = quantities[2];
  v = quantities[3];
  return std::max( { U, V, u, v } );
}

void Checker::run(std::string when) const
{
  return;
  const size_t Nblocks = velInfo.size();

  const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  bool bAbort = false;

  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    ScalarBlock& PRES= *(ScalarBlock*) presInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if(std::isnan( VEL(ix,iy).u[0])) {
        printf("isnan( VEL(ix,iy).u[0]) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isinf( VEL(ix,iy).u[0])) {
        printf("isinf( VEL(ix,iy).u[0]) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isnan( VEL(ix,iy).u[1])) {
        printf("isnan( VEL(ix,iy).u[1]) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isinf( VEL(ix,iy).u[1])) {
        printf("isinf( VEL(ix,iy).u[1]) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isnan(PRES(ix,iy).s   )) {
        printf("isnan(PRES(ix,iy).s   ) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isinf(PRES(ix,iy).s   )) {
        printf("isinf(PRES(ix,iy).s   ) %s\n", when.c_str());
        bAbort = true;
        break;
      }
    }
  }

  if( bAbort )
  {
    std::cout << "[CUP2D] Detected NaN/INF Field Values. Dumping the field and aborting..." << std::endl;
    sim.dumpAll("abort_");
    MPI_Abort(sim.comm,1);
  }
}

void ApplyObjVel::operator()(const Real dt)
{
  const size_t Nblocks = velInfo.size();

  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock& UF = *(VectorBlock*)  velInfo[i].ptrBlock;
    VectorBlock& US = *(VectorBlock*) uDefInfo[i].ptrBlock;
    ScalarBlock& X  = *(ScalarBlock*)  chiInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
     UF(ix,iy).u[0]= UF(ix,iy).u[0] *(1-X(ix,iy).s) +US(ix,iy).u[0] *X(ix,iy).s;
     UF(ix,iy).u[1]= UF(ix,iy).u[1] *(1-X(ix,iy).s) +US(ix,iy).u[1] *X(ix,iy).s;
    }
  }
}
