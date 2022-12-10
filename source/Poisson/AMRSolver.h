//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Base.h"

#include <Cubism/FluxCorrection.h>

class ComputeLHS : public Operator
{
  struct LHSkernel
  {
    LHSkernel(const SimulationData & ss) : sim(ss) {}
    const SimulationData & sim;
    const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
    const std::vector<cubism::BlockInfo>& lhsInfo = sim.tmp->getBlocksInfo();
    const std::vector<cubism::BlockInfo>& xInfo = sim.pres->getBlocksInfo();
  
    void operator()(ScalarLab & lab, const cubism::BlockInfo& info) const
    {
      ScalarBlock & __restrict__ LHS = *(ScalarBlock*) lhsInfo[info.blockID].ptrBlock;
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
        LHS(ix,iy).s = ( ((lab(ix-1,iy).s + lab(ix+1,iy).s) + (lab(ix,iy-1).s + lab(ix,iy+1).s)) - 4.0*lab(ix,iy).s);
  
      cubism::BlockCase<ScalarBlock> * tempCase = (cubism::BlockCase<ScalarBlock> *)(lhsInfo[info.blockID].auxiliary);
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
        for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
          faceXm[iy] = lab(ix,iy) - lab(ix-1,iy);
      }
      if (faceXp != nullptr)
      {
        int ix = ScalarBlock::sizeX-1;
        for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
          faceXp[iy] = lab(ix,iy) - lab(ix+1,iy);
      }
      if (faceYm != nullptr)
      {
        int iy = 0;
        for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
          faceYm[ix] = lab(ix,iy) - lab(ix,iy-1);
      }
      if (faceYp != nullptr)
      {
        int iy = ScalarBlock::sizeY-1;
        for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
          faceYp[ix] = lab(ix,iy) - lab(ix,iy+1);
      }
    }
  };
  public:
  ComputeLHS(SimulationData & ss) : Operator(ss) { }
  bool isCorner(cubism::BlockInfo & info)
  {
    const bool x = info.index[0] == 0;
    const bool y = info.index[1] == 0;
    return x && y;
  }


  void operator()(const Real dt)
  {
    const LHSkernel K(sim);
    cubism::compute<ScalarLab>(K,sim.pres,sim.tmp);
    if( sim.bMeanConstraint )
    {
      int index = -1;
      Real mean = 0.0;
      std::vector<cubism::BlockInfo>& lhsInfo = sim.tmp->getBlocksInfo();
      const std::vector<cubism::BlockInfo>& xInfo = sim.pres->getBlocksInfo();
      #pragma omp parallel for reduction(+:mean)
      for (size_t i = 0 ; i < lhsInfo.size() ; i++)
      {
       cubism::BlockInfo & info = lhsInfo[i];
       if ( isCorner(info) ) index = i;
       const Real h2 = info.h*info.h;
       ScalarBlock & __restrict__ X   = *(ScalarBlock*) xInfo[info.blockID].ptrBlock;
       for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
       for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
         mean += h2 * X(ix,iy).s;
      }
      MPI_Allreduce(MPI_IN_PLACE,&mean,1,MPI_Real,MPI_SUM,sim.chi->getWorldComm());
      if (index != -1)
      {
       ScalarBlock & __restrict__ LHS = *(ScalarBlock*) lhsInfo[index].ptrBlock;
       LHS(0,0).s = mean;
      }
      /*
      for (size_t i = 0 ; i < lhsInfo.size() ; i++)
      {
         ScalarBlock & __restrict__ LHS = *(ScalarBlock*) lhsInfo[i].ptrBlock;
         const Real h2 = lhsInfo[i].h*lhsInfo[i].h;
         for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
         for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
             LHS(ix,iy).s += mean * h2;
      }
      */
    }
  }
  std::string getName() { return "ComputeLHS"; }
};

class AMRSolver : public PoissonSolver
{
 protected:
  SimulationData& sim;
 public:
  std::string getName() {
    return "AMRSolver";
  }
  AMRSolver(SimulationData& ss);
  void solve(const ScalarGrid *input, ScalarGrid *output) override;
  ComputeLHS Get_LHS;
  std::vector<std::vector<Real>> Ld;
  std::vector <  std::vector <std::vector< std::pair<int,Real> > > >L_row;
  std::vector <  std::vector <std::vector< std::pair<int,Real> > > >L_col;
  void getZ(cubism::BlockInfo & zInfo);
  Real getA_local(const int I1, const int I2);

  std::vector<Real> x   ;
  std::vector<Real> r   ;
  std::vector<Real> p   ; //initialize p = 0
  std::vector<Real> v   ; //initialize v = 0
  std::vector<Real> s   ;
  std::vector<Real> rhat;
  std::vector<Real> x_opt;

  bool isCorner(cubism::BlockInfo & info)
  {
    return Get_LHS.isCorner(info);
  }
};
