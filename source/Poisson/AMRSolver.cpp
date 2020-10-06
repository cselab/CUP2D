//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "AMRSolver.h"

using namespace cubism;

void AMRSolver::solve()
{
  sim.startProfiler("AMRSolver");
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  const std::vector<cubism::BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp ->getBlocksInfo();

  size_t Nblocks = presInfo.size();
  

  double norm_tot0;
  for (int iter = 0 ; iter < 100000; iter ++)
  {
    double norm_tot = 0.0;
    #pragma omp parallel
    {
      static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
      ScalarLab lab; 
      lab.prepare(*(sim.pres), stenBeg, stenEnd, 1);
  
      #pragma omp for schedule(static)
      for (size_t i=0; i < Nblocks; i++)
      {
        //const Real h = presInfo[i].h_gridpoint;
  
        lab.load(presInfo[i]); 
  
        ScalarBlock & __restrict__ TMP = *(ScalarBlock*) tmpInfo[i].ptrBlock;
        ScalarBlock & __restrict__ P = *(ScalarBlock*) presInfo[i].ptrBlock;
  
        double norm = 0.0;
  
        double rel = 1.0;

        for(int iy=0; iy<BSY; ++iy)
        for(int ix=0; ix<BSX; ++ix)
        {
          double Pnew = (1.-rel)*P(ix,iy).s + 
                        rel* ( ( lab(ix-1,iy).s + lab(ix+1,iy).s + 
                                 lab(ix,iy-1).s + lab(ix,iy+1).s ) - TMP(ix,iy).s )*0.25;       
          norm += std::fabs(Pnew - P(ix,iy).s);
          P(ix,iy).s = Pnew;
        }

        #pragma omp atomic
        norm_tot += norm;
      }
    }

    norm_tot /= (Nblocks*BSX*BSY);

    if (iter == 0) norm_tot0 = norm_tot; 

    if (norm_tot/norm_tot0 < 1e-2 || norm_tot < 1e-6) break;

    if (iter % 2500 == 0) std::cout << iter << " norm=" << norm_tot << " " << norm_tot0 << std::endl; 
  }

  sim.stopProfiler();
}