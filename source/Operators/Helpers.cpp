//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Helpers.h"
#include <gsl/gsl_linalg.h>
#include <random>

using namespace cubism;

void computeVorticity::run() const
{
  const size_t Nblocks = velInfo.size();
  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  #pragma omp parallel
  {
    static constexpr int stenBeg [3] = {-1,-1, 0}, stenEnd [3] = { 2, 2, 1};
    VectorLab velLab;   velLab.prepare(*(sim.vel), stenBeg, stenEnd, false);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      const Real invH = 0.5 / tmpInfo[i].h_gridpoint;
      velLab.load( velInfo[i]);
      const auto & __restrict__ V   = velLab;
      auto& __restrict__ O = *(ScalarBlock*)  tmpInfo[i].ptrBlock;

      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
      {
        O(x,y).s = invH * (V(x,y-1).u[0]-V(x,y+1).u[0] + V(x+1,y).u[1]-V(x-1,y).u[1]);
      }
    }
  }
}

void computeDivergence::run() const
{
  FluxCorrection<ScalarGrid,ScalarBlock> Corrector;
  Corrector.prepare(*(sim.tmp));
  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

  const size_t Nblocks = velInfo.size();

  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  #pragma omp parallel
  {
    static constexpr int stenBeg [3] = {-1,-1,0}, stenEnd [3] = { 2, 2, 1};

    VectorLab velLab;   velLab.prepare(*(sim.vel), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      const Real H = tmpInfo[i].h_gridpoint;
      velLab.load( velInfo[i], 0); const auto & __restrict__ V   = velLab;
      auto& __restrict__ O = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
      auto& __restrict__ CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;

      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
      {
        O(x,y).s = (1.0-CHI(x,y).s)*0.5*H*(V(x+1,y).u[0]-V(x-1,y).u[0] + V(x,y+1).u[1]-V(x,y-1).u[1]);
      }

      BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(tmpInfo[i].auxiliary);
      ScalarBlock::ElementType * faceXm = nullptr;
      ScalarBlock::ElementType * faceXp = nullptr;
      ScalarBlock::ElementType * faceYm = nullptr;
      ScalarBlock::ElementType * faceYp = nullptr;
      if (tempCase != nullptr)
      {
        faceXm = tempCase -> storedFace[0] ?  & tempCase -> m_pData[0][0] : nullptr;
        faceXp = tempCase -> storedFace[1] ?  & tempCase -> m_pData[1][0] : nullptr;
        faceYm = tempCase -> storedFace[2] ?  & tempCase -> m_pData[2][0] : nullptr;
        faceYp = tempCase -> storedFace[3] ?  & tempCase -> m_pData[3][0] : nullptr;
      }
      if (faceXm != nullptr)
      {
        int ix = 0;
        for(int iy=0; iy<BSY; ++iy)
          faceXm[iy].s = (1.0-CHI(ix,iy).s)*0.5*H *(V(ix-1,iy).u[0] + V(ix,iy).u[0]) ;
      }
      if (faceXp != nullptr)
      {
        int ix = BSX-1;
        for(int iy=0; iy<BSY; ++iy)
          faceXp[iy].s = -(1.0-CHI(ix,iy).s)*0.5*H*(V(ix+1,iy).u[0] + V(ix,iy).u[0]);
      }
      if (faceYm != nullptr)
      {
        int iy = 0;
        for(int ix=0; ix<BSX; ++ix)
          faceYm[ix].s = (1.0-CHI(ix,iy).s)*0.5*H* (V(ix,iy-1).u[1] + V(ix,iy).u[1]);
      }
      if (faceYp != nullptr)
      {
        int iy = BSY-1;
        for(int ix=0; ix<BSX; ++ix)
          faceYp[ix].s = -(1.0-CHI(ix,iy).s)*0.5*H* (V(ix,iy+1).u[1] + V(ix,iy).u[1]);
      }
    }
  }

  Corrector.FillBlockCases();


  double sDtot = 0.0;
  for (size_t i=0; i < Nblocks; i++)
  {
    auto& __restrict__ O = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
    for(int y=0; y<VectorBlock::sizeY; ++y)
    for(int x=0; x<VectorBlock::sizeX; ++x)
      sDtot += O(x,y).s;
  }
  // std::cout << "Total div(V)="<<sDtot<<std::endl;
  std::ofstream outfile;
  outfile.open("div.txt", std::ios_base::app);
  outfile << sim.time << " " << sDtot  << " " << Nblocks << "\n";
  outfile.close();
}

void IC::operator()(const double dt)
{

  const std::vector<BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
  const std::vector<BlockInfo>& presInfo  = sim.pres->getBlocksInfo();
  const std::vector<BlockInfo>& uDefInfo  = sim.uDef->getBlocksInfo();
  //const std::vector<BlockInfo>& forceInfo = sim.force->getBlocksInfo();

  const std::vector<BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  const std::vector<BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();

  const size_t Nblocks = velInfo.size();

  #pragma omp parallel
  {
    //std::random_device rd;
    //std::normal_distribution<Real> dist(0, 1e-7);
    //std::mt19937 gen(rd());
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;  VEL.clear();
      VectorBlock& UDEF= *(VectorBlock*) uDefInfo[i].ptrBlock; UDEF.clear();
      ScalarBlock& CHI = *(ScalarBlock*)  chiInfo[i].ptrBlock;  CHI.clear();
      ScalarBlock& PRES= *(ScalarBlock*) presInfo[i].ptrBlock; PRES.clear();
      //VectorBlock&    F= *(VectorBlock*)forceInfo[i].ptrBlock;    F.clear();

      ScalarBlock& TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;  TMP.clear();
      VectorBlock& TMPV= *(VectorBlock*) tmpVInfo[i].ptrBlock; TMPV.clear();
      assert(velInfo[i].blockID ==  uDefInfo[i].blockID);
      assert(velInfo[i].blockID ==   chiInfo[i].blockID);
      assert(velInfo[i].blockID ==  presInfo[i].blockID);
      //assert(velInfo[i].blockID == forceInfo[i].blockID);
      assert(velInfo[i].blockID ==   tmpInfo[i].blockID);
      assert(velInfo[i].blockID ==  tmpVInfo[i].blockID);
      //for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      //for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      //  VEL(ix,iy).u[0] += dist(gen);
      //  VEL(ix,iy).u[1] += dist(gen);
      //}
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
    const Real h = velInfo[i].h_gridpoint;
    const VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      const Real facMom = h*h;
      momX += facMom * VEL(ix,iy).u[0];
      momY += facMom * VEL(ix,iy).u[1];
      totM += facMom;
    }
  }
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
  return std::max( { U, V, u, v } );
}

void Checker::run(std::string when) const
{
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
        // printf("isnan( VEL(ix,iy).u[0]) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isinf( VEL(ix,iy).u[0])) {
        // printf("isinf( VEL(ix,iy).u[0]) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isnan( VEL(ix,iy).u[1])) {
        // printf("isnan( VEL(ix,iy).u[1]) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isinf( VEL(ix,iy).u[1])) {
        // printf("isinf( VEL(ix,iy).u[1]) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isnan(PRES(ix,iy).s   )) {
        // printf("isnan(PRES(ix,iy).s   ) %s\n", when.c_str());
        bAbort = true;
        break;
      }
      if(std::isinf(PRES(ix,iy).s   )) {
        // printf("isinf(PRES(ix,iy).s   ) %s\n", when.c_str());
        bAbort = true;
        break;
      }
    }
  }

  if( bAbort )
  {
    std::cout << "[CUP2D] Detected Errorous Field Values. Dumping the field and aborting..." << std::endl;
    sim.dumpAll("abort_");
    fflush(0); 
    abort();
  }
}

void ApplyObjVel::operator()(const double dt)
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
