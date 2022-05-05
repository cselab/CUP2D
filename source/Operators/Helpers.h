//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"

class findMaxU
{
  SimulationData& sim;
  const std::vector<cubism::BlockInfo> & velInfo = sim.vel->getBlocksInfo();
 public:
  findMaxU(SimulationData& s) : sim(s) { }

  Real run() const;

  std::string getName() const {
    return "findMaxU";
  }
};

class Checker
{
  SimulationData& sim;
  const std::vector<cubism::BlockInfo> & velInfo = sim.vel->getBlocksInfo();
 public:
  Checker(SimulationData& s) : sim(s) { }

  void run(std::string when) const;

  std::string getName() const {
    return "Checker";
  }
};

class IC : public Operator
{
  protected:
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

  public:
  IC(SimulationData& s) : Operator(s) { }

  void operator()(const Real dt);

  std::string getName() {
    return "IC";
  }
};

class gaussianIC : public Operator
{
  protected:
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

  public:
  gaussianIC(SimulationData& s) : Operator(s) { }

  void operator()(const Real dt);

  std::string getName() {
    return "gaussianIC";
  }
};

class randomIC : public Operator
{
  protected:
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

  public:
  randomIC(SimulationData& s) : Operator(s) { }

  void operator()(const Real dt);

  std::string getName() {
    return "randomIC";
  }
};

class ApplyObjVel : public Operator
{
  protected:
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

  public:
  ApplyObjVel(SimulationData& s) : Operator(s) { }

  void operator()(const Real dt);

  std::string getName() {
    return "ApplyObjVel";
  }
};

struct KernelVorticity
{
  KernelVorticity(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0,1}};
  void operator()(VectorLab & lab, const cubism::BlockInfo& info) const
  {
    const Real i2h = 0.5/info.h;
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    for(int y=0; y<VectorBlock::sizeY; ++y)
    for(int x=0; x<VectorBlock::sizeX; ++x)
      TMP(x,y).s = i2h * ((lab(x,y-1).u[0]-lab(x,y+1).u[0]) + (lab(x+1,y).u[1]-lab(x-1,y).u[1]));
  }
};

class computeVorticity : public Operator
{
 public:
  computeVorticity(SimulationData& s) : Operator(s){ }

  void operator()(const Real dt)
  {
    const KernelVorticity mykernel(sim);
    cubism::compute<VectorLab>(mykernel,sim.vel);

    if (!sim.muteAll)
      reportVorticity();
  }

  void reportVorticity() const
  {
    Real maxv = -1e10;
    Real minv = -1e10;
    #pragma omp parallel for reduction(min:minv) reduction(max:maxv)
    for (auto & info: sim.tmp->getBlocksInfo())
    {
      auto & TMP = *(ScalarBlock*) info.ptrBlock;
      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
      {
        maxv = std::max(maxv, TMP(x,y).s);
        minv = std::max(minv,-TMP(x,y).s);
      }
    }
    Real buffer[2] = {maxv,minv};
    Real recvbuf[2];
    MPI_Reduce(buffer,recvbuf, 2, MPI_Real, MPI_MAX, 0, sim.chi->getCartComm());
    recvbuf[1]=-recvbuf[1];
    if (sim.rank == 0)
      std::cout << " max(omega)=" << recvbuf[0] << " min(omega)=" << recvbuf[1] << " max(omega)+min(omega)=" << recvbuf[0]+recvbuf[1] << std::endl;
  }

  std::string getName()
  {
    return "computeVorticity";
  }
};

struct KernelQ
{
  KernelQ(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0,1}};
  void operator()(VectorLab & lab, const cubism::BlockInfo& info) const
  {
    const Real i2h = 0.5/info.h;
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    for(int y=0; y<VectorBlock::sizeY; ++y)
    for(int x=0; x<VectorBlock::sizeX; ++x)
    {
      const Real WZ  = i2h * ((lab(x,y-1).u[0]-lab(x,y+1).u[0]) + (lab(x+1,y).u[1]-lab(x-1,y).u[1]));
      const Real D11 = i2h * (lab(x+1,y).u[0]-lab(x-1,y).u[0]); // shear stresses
      const Real D22 = i2h * (lab(x,y+1).u[1]-lab(x,y-1).u[1]); // shear stresses
      const Real D12 = i2h * ( (lab(x,y+1).u[0]-lab(x,y-1).u[0]) + (lab(x+1,y).u[1]-lab(x-1,y).u[1]) ); // shear stresses
      const Real SS = D11*D11 + D22*D22 + 0.5*(D12*D12);
      TMP(x,y).s = 0.5*( 0.5*(WZ*WZ) - SS );
    }
  }
};

class computeQ : public Operator
{
 public:
  computeQ(SimulationData& s) : Operator(s){ }

  void operator()(const Real dt)
  {
    const KernelQ mykernel(sim);
    cubism::compute<VectorLab>(mykernel,sim.vel);
  }

  std::string getName()
  {
    return "computeQ";
  }
};

struct KernelDivergence
{
  KernelDivergence(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0,1}};
  void operator()(VectorLab & lab, const cubism::BlockInfo& info) const
  {
    const Real h = info.h;
    const Real facDiv = 0.5*h;
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      TMP(ix, iy).s  =   facDiv*( (lab(ix+1,iy).u[0] -  lab(ix-1,iy).u[0]) +  (lab(ix,iy+1).u[1] -  lab(ix,iy-1).u[1]));
    cubism::BlockCase<ScalarBlock> * tempCase = (cubism::BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
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
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        faceXm[iy].s  =  facDiv                *( lab(ix-1,iy).u[0] +  lab(ix,iy).u[0]) ;
    }
    if (faceXp != nullptr)
    {
      int ix = VectorBlock::sizeX-1;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
        faceXp[iy].s  = -facDiv               *( lab(ix+1,iy).u[0] +  lab(ix,iy).u[0]);
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        faceYm[ix].s  =  facDiv               *( lab(ix,iy-1).u[1] +  lab(ix,iy).u[1]);
    }
    if (faceYp != nullptr)
    {
      int iy = VectorBlock::sizeY-1;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
        faceYp[ix].s  = -facDiv               *( lab(ix,iy+1).u[1] +  lab(ix,iy).u[1]);
    }
  }
};

class computeDivergence : public Operator
{
 public:
  computeDivergence(SimulationData& s) : Operator(s){ }

  void operator()(const Real dt)
  {

    const KernelDivergence mykernel(sim);
    cubism::compute<VectorLab>(mykernel,sim.vel,sim.tmp);
    #if 0
    Real total = 0.0;
    Real abs   = 0.0;
    for (auto & info: sim.tmp->getBlocksInfo())
    {
      auto & TMP = *(ScalarBlock*) info.ptrBlock;
      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
      {
        abs   += std::fabs(TMP(x,y).s);
        total += TMP(x,y).s;
      }
    }
    Real sendbuf[2]={total,abs};
    Real recvbuf[2];
    MPI_Reduce(sendbuf, recvbuf, 2, MPI_Real, MPI_SUM, 0, sim.chi->getCartComm());
    if (sim.rank == 0)
    {
      ofstream myfile;
      myfile.open ("div.txt",ios::app);
      myfile << sim.step << " " << total << " " << abs << std::endl;
      myfile.close();
    }
    #endif
  }

  std::string getName()
  {
    return "computeDivergence";
  }
};
