//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "SimulationData.h"

/// Abstract base operator with kernel evaluation static member functions.
class OperatorBase
{
public:
  template <typename Kernel, typename TGrid, typename LabMPI,typename TGrid_corr = TGrid>
  static void compute(const Kernel& kernel, TGrid& grid, const bool applyFluxCorrection = false, TGrid_corr * corrected_grid = nullptr)
  {
    if (applyFluxCorrection)
      corrected_grid->Corrector.prepare(*corrected_grid);

    cubism::SynchronizerMPI_AMR<Real,TGrid>& Synch = * grid.sync(kernel);

    std::vector<cubism::BlockInfo*> & inner = Synch.avail_inner();
    std::vector<cubism::BlockInfo*> *halo;
    #pragma omp parallel
    {
      LabMPI lab;
      lab.prepare(grid, Synch);
      #pragma omp for nowait
      for (const cubism::BlockInfo *I : inner) {
        lab.load(*I, 0);
        kernel(lab, *I);
      }

      #pragma omp master
      halo = &Synch.avail_halo();
      #pragma omp barrier

      lab.prepare(grid, Synch);
      #pragma omp for nowait
      for (const cubism::BlockInfo *I : *halo) {
        lab.load(*I, 0);
        kernel(lab, *I);
      }
    }

    if (applyFluxCorrection)
      corrected_grid->Corrector.FillBlockCases();

    //MPI_Barrier(grid.getCartComm());
  }

  template <typename Kernel, typename TGrid, typename LabMPI, typename TGrid2, typename LabMPI2, typename TGrid_corr = TGrid>
  static void compute(const Kernel& kernel, TGrid& grid, TGrid2& grid2, const bool applyFluxCorrection = false, TGrid_corr * corrected_grid = nullptr)
  {
    if (applyFluxCorrection)
      corrected_grid->Corrector.prepare(*corrected_grid);

    cubism::SynchronizerMPI_AMR<Real,TGrid >& Synch  = * grid .sync(kernel);

    Kernel kernel2 = kernel;
    kernel2.stencil.sx = kernel2.stencil2.sx;
    kernel2.stencil.sy = kernel2.stencil2.sy;
    kernel2.stencil.sz = kernel2.stencil2.sz;
    kernel2.stencil.ex = kernel2.stencil2.ex;
    kernel2.stencil.ey = kernel2.stencil2.ey;
    kernel2.stencil.ez = kernel2.stencil2.ez;
    kernel2.stencil.tensorial = kernel2.stencil2.tensorial;
    kernel2.stencil.selcomponents.clear();
    kernel2.stencil.selcomponents = kernel2.stencil2.selcomponents;


    cubism::SynchronizerMPI_AMR<Real,TGrid2>& Synch2 = * grid2.sync(kernel2);

    const int nthreads = omp_get_max_threads();
    LabMPI * labs = new LabMPI[nthreads];
    LabMPI2 * labs2 = new LabMPI2[nthreads];
    #pragma omp parallel for schedule(static, 1)
    for(int i = 0; i < nthreads; ++i)
    {
      labs[i].prepare(grid, Synch);
      labs2[i].prepare(grid2, Synch2);      
    }

    //MPI_Barrier(grid.getCartComm());

    std::vector<cubism::BlockInfo*> & avail0  = Synch .avail_inner();
    std::vector<cubism::BlockInfo*> & avail02 = Synch2.avail_inner();
    const int Ninner = avail0.size();
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      LabMPI & lab  = labs[tid];
      LabMPI2& lab2 = labs2[tid];
      #pragma omp for schedule(static)
      for(int i=0; i<Ninner; i++) {
        const cubism::BlockInfo &I  = *avail0 [i];
        const cubism::BlockInfo &I2 = *avail02[i];
        lab.load(I, 0);
        lab2.load(I2, 0);
        kernel(lab,lab2,I,I2);
      }
    }
    std::vector<cubism::BlockInfo*> & avail1  = Synch .avail_halo();
    std::vector<cubism::BlockInfo*> & avail12 = Synch2.avail_halo();
    const int Nhalo = avail1.size();
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      LabMPI & lab  = labs [tid];
      LabMPI2& lab2 = labs2[tid];
      #pragma omp for schedule(static)
      for(int i=0; i<Nhalo; i++) {
        const cubism::BlockInfo &I  = *avail1 [i];
        const cubism::BlockInfo &I2 = *avail12[i];
        lab.load(I, 0);
        lab2.load(I2, 0);
        kernel(lab, lab2, I, I2);
      }
    }

    delete [] labs;
    delete [] labs2;
    labs = nullptr;
    labs2 = nullptr;

    if (applyFluxCorrection)
      corrected_grid->Corrector.FillBlockCases();

    //MPI_Barrier(grid.getCartComm());
  }

  virtual ~OperatorBase() = default;
  virtual void operator()(const Real dt) = 0;
  virtual std::string getName() = 0;
};


class Operator : public OperatorBase
{
public:
  SimulationData& sim;

protected:
  const std::vector<cubism::BlockInfo>& velInfo;

public:
  Operator(SimulationData& s) : sim{s}, velInfo{_getVelInfo(s.vel)} { }

private:
  // Circumvent a g++ 9 bug where `s.vel ? ... : throw ...` erroneously
  // triggers the warning "a temporary bound to 'Operator::velInfo' only
  // persists until the constructor exits".
  static inline const std::vector<cubism::BlockInfo> &_getVelInfo(VectorGrid *vel)
  {
    if (!vel)
      throw std::runtime_error("cannot create an operator before initializing fields");
    return vel->getBlocksInfo();
  }
};
