//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

//#include "PoissonSolver.h"
#include "../Poisson/FFTW_freespace.h"
#ifdef HYPREFFT
#include "../Poisson/HYPREdirichlet.h"
#endif
#include "../Poisson/FFTW_dirichlet.h"
#include "../Poisson/FFTW_periodic.h"
#ifdef CUDAFFT
#include "../Poisson/CUDA_all.h"
#endif

using namespace cubism;
static constexpr double EPS = std::numeric_limits<double>::epsilon();

PoissonSolver * PoissonSolver::makeSolver(SimulationData& sim)
{
  #ifdef HYPREFFT
    if (sim.poissonType == "hypre")
      return static_cast<PoissonSolver*>(new HYPREdirichlet(sim));
    else
  #endif

  #ifdef CUDAFFT

    if (sim.poissonType == "periodic")
      return static_cast<PoissonSolver*>(new CUDA_periodic(sim));
    else
    if (sim.poissonType == "freespace")
      return static_cast<PoissonSolver*>(new CUDA_freespace(sim));
    else
    if (sim.poissonType == "cpu_periodic")
      return static_cast<PoissonSolver*>(new FFTW_periodic(sim));
    else
    if (sim.poissonType == "cpu_freespace")
      return static_cast<PoissonSolver*>(new FFTW_freespace(sim));
    else

  #else

    if (sim.poissonType == "periodic")
      return static_cast<PoissonSolver*>(new FFTW_periodic(sim));
    else
    if (sim.poissonType == "freespace")
      return static_cast<PoissonSolver*>(new FFTW_freespace(sim));
    else

  #endif

  // default is dirichlet BC
  return static_cast<PoissonSolver*>(new FFTW_dirichlet(sim));
}

void PoissonSolver::cub2rhs(const std::vector<BlockInfo>& BSRC)
{
  const size_t nBlocks = BSRC.size();
  Real sumRHS = 0, sumABS = 0;

  Real * __restrict__ const dest = buffer;

  #pragma omp parallel for schedule(static) reduction(+ : sumRHS, sumABS)
  for(size_t i=0; i<nBlocks; ++i)
  {
    const BlockInfo& info = BSRC[i];
    const size_t blocki = VectorBlock::sizeX * info.index[0];
    const size_t blockj = VectorBlock::sizeY * info.index[1];
    const ScalarBlock& b = *(ScalarBlock*)info.ptrBlock;
    const size_t blockStart = blocki + stride * blockj;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++) {
      dest[blockStart + ix + stride*iy] = b(ix,iy).s;
      sumABS += std::fabs(b(ix,iy).s);
      sumRHS +=           b(ix,iy).s;
    }
  }

  #if 0
    const Real C = sumRHS/std::max(std::numeric_limits<Real>::epsilon(),sumABS);
    //printf("Relative RHS correction:%e\n", C);
    #pragma omp parallel for schedule(static)
    for (size_t iy = 0; iy < totNy; iy++)
    for (size_t ix = 0; ix < totNx; ix++)
      dest[ix + stride * iy] -=  std::fabs(dest[ix +stride * iy]) * C;
  #elif 0
    const Real C = sumRHS/totNy/totNx;
    //printf("Relative RHS correction:%e\n", C);
    #pragma omp parallel for schedule(static)
    for (size_t iy = 0; iy < totNy; iy++)
    for (size_t ix = 0; ix < totNx; ix++) dest[ix + stride * iy] -= C;
  #else
    const auto& extent = sim.extents;
    const Real fadeLenX = sim.fadeLenX, fadeLenY = sim.fadeLenY;
    const Real invFadeX = 1/std::max(fadeLenX,EPS);
    const Real invFadeY = 1/std::max(fadeLenY,EPS);
    const Real nInnX = totNx*(extent[0] - 2*fadeLenX)/extent[0];
    const Real nInnY = totNy*(extent[1] - 2*fadeLenY)/extent[1];
    const Real nOutX = (totNx-nInnX)/2, nOutY = (totNy-nInnY)/2;
    const Real C = sumRHS/(nInnX*nOutY + nInnY*nOutX + 4*nOutX*nOutY);
    const auto _is_touching = [&] (const BlockInfo& i) {
      Real min_pos[2], max_pos[2]; i.pos(min_pos, 0, 0);
      i.pos(max_pos, VectorBlock::sizeX-1, VectorBlock::sizeY-1);
      const bool touchW = fadeLenX >= min_pos[0];
      const bool touchE = fadeLenX >= extent[0] - max_pos[0];
      const bool touchS = fadeLenY >= min_pos[1];
      const bool touchN = fadeLenY >= extent[1] - max_pos[1];
      return touchN || touchE || touchS || touchW;
    };
    #pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i < nBlocks; i++)
    {
      if( not _is_touching(BSRC[i]) ) continue;
      const size_t blocki = VectorBlock::sizeX * BSRC[i].index[0];
      const size_t blockj = VectorBlock::sizeY * BSRC[i].index[1];
      const size_t start = blocki + stride * blockj;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        Real p[2]; BSRC[i].pos(p, ix, iy);
        const Real yt = invFadeY*std::max(Real(0), fadeLenY -extent[1]+p[1] );
        const Real yb = invFadeY*std::max(Real(0), fadeLenY -p[1] );
        const Real xt = invFadeX*std::max(Real(0), fadeLenX -extent[0]+p[0] );
        const Real xb = invFadeX*std::max(Real(0), fadeLenX -p[0] );
        if(yt*xt>0 || yt*xb>0 || yb*xt>0 || yb*xb>0) {
          buffer[start + ix + stride*iy] -= C;
        } else {
          const Real fadeArg = std::min(std::max({yt, yb, xt, xb}), (Real)1);
          buffer[start + ix + stride*iy] -= C * fadeArg;
        }
      }
    }
  #endif

  #if 1 // ndef NDEBUG
    Real sumRHSpost = 0;
    #pragma omp parallel for schedule(static) reduction(+ : sumRHSpost)
    for(size_t iy = 0; iy < totNy; iy++)
    for(size_t ix = 0; ix < totNx; ix++) sumRHSpost += dest[ix + stride * iy];
    printf("Relative RHS post correction:%e\n", sumRHSpost);
    //assert(sumRHSpost < 10*std::sqrt(std::numeric_limits<Real>::epsilon()));
  #endif
}

void PoissonSolver::sol2cub(const std::vector<BlockInfo>& BDST)
{
  const size_t nBlocks = BDST.size();
  //const Real F = 0.2, A = F * iter / (1 + F * iter);
  //const Real A = iter == 0 ? 0 : 0;//MOMENTUM_FACTOR;
  //if(iter == 0) std::fill(presMom, presMom + totNy * totNx, 0);
  const Real * __restrict__ const sorc = buffer;
  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<nBlocks; ++i)
  {
    const BlockInfo& info = BDST[i];
    const size_t blocki = VectorBlock::sizeX * info.index[0];
    const size_t blockj = VectorBlock::sizeY * info.index[1];
    ScalarBlock& b = *(ScalarBlock*)info.ptrBlock;
    const size_t blockStart = blocki + stride*blockj;
    //const size_t momSt = blocki + totNx*blockj;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++) {
      //const Real DP = sorc[blockStart + ix + stride*iy] - b(ix,iy).s;
      //presMom[momSt + ix + totNx*iy] = A*presMom[momSt + ix + totNx*iy] + DP;
      //b(ix,iy).s = b(ix,iy).s + presMom[momSt + ix + totNx*iy];
      b(ix,iy).s = sorc[blockStart + ix + stride*iy];
    }
  }
}

PoissonSolver::PoissonSolver(SimulationData&s,long p): sim(s), stride(p) {
  std::fill(presMom, presMom + totNy * totNx, 0);
}
