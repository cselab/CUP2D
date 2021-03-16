//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "AMRSolver.h"

using namespace cubism;

void AMRSolver::Get_LHS (ScalarGrid * lhs, ScalarGrid * x)
{
    static constexpr int BSX = VectorBlock::sizeX;
    static constexpr int BSY = VectorBlock::sizeY;

    //compute A*x and store it into LHS
    //here A corresponds to the discrete Laplacian operator for an AMR mesh

    const std::vector<cubism::BlockInfo>& lhsInfo = lhs->getBlocksInfo();
    const std::vector<cubism::BlockInfo>& xInfo = x->getBlocksInfo();

    #pragma omp parallel
    {
      static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
      ScalarLab lab; 
      lab.prepare(*x, stenBeg, stenEnd, 1);
  
      #pragma omp for schedule(runtime)
      for (size_t i=0; i < xInfo.size(); i++)
      {
        lab.load(xInfo[i]);
        ScalarBlock & __restrict__ LHS = *(ScalarBlock*) lhsInfo[i].ptrBlock;
        for(int iy=0; iy<BSY; ++iy)
        for(int ix=0; ix<BSX; ++ix)
        {
          LHS(ix,iy).s = ( lab(ix-1,iy).s + 
                           lab(ix+1,iy).s + 
                           lab(ix,iy-1).s + 
                           lab(ix,iy+1).s - 4.0*lab(ix,iy).s );
        }

        BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(lhsInfo[i].auxiliary);
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
            faceXm[iy] = lab(ix,iy) - lab(ix-1,iy);
        }
        if (faceXp != nullptr)
        {
          int ix = BSX-1;
          for(int iy=0; iy<BSY; ++iy)
            faceXp[iy] = lab(ix,iy) - lab(ix+1,iy);
        }
        if (faceYm != nullptr)
        {
          int iy = 0;
          for(int ix=0; ix<BSX; ++ix)
            faceYm[ix] = lab(ix,iy) - lab(ix,iy-1);
        }
        if (faceYp != nullptr)
        {
          int iy = BSY-1;
          for(int ix=0; ix<BSX; ++ix)
            faceYp[ix] = lab(ix,iy) - lab(ix,iy+1);
        }
      }
    }
    Corrector.FillBlockCases();
}

AMRSolver::AMRSolver(SimulationData& s):sim(s){}

void AMRSolver::solve()
{
  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

  // The bi-conjugate gradient method needs the following 8 arrays:
  // r: residual vector
  // rhat: (2nd residual vector)
  // p: conjugate vector
  // v: bi-conjugate vector
  // s vector
  // Ax vector (stores the LHS computation)
  // x vector (current solution estimate)
  // z vector (used for preconditioning)

  // There arrays will be represented by the following CUP2D arrays:
  // x    --> sim.pres
  // s    --> sim.chi
  // z    --> sim.pOld
  // Ax   --> sim.tmp
  // p    --> sim.vel[0]
  // v    --> sim.vel[1]
  // r    --> sim.tmpV[0]
  // rhat --> sim.tmpV[1]
  Corrector.prepare(*sim.tmp);

  std::vector<cubism::BlockInfo>&  xInfo = sim.pres->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  sInfo = sim.chi ->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  zInfo = sim.pOld->getBlocksInfo();
  std::vector<cubism::BlockInfo>& AxInfo = sim.tmp ->getBlocksInfo();
  std::vector<cubism::BlockInfo>& pvInfo = sim.vel ->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  rInfo = sim.tmpV->getBlocksInfo();

  // First, we store in SavedFields whatever is contained 
  // in the CUP2D arrays. Then, once the algorithm finishes
  // everything is copied back from SavedFields to CUP2D.
  const size_t Nblocks = xInfo.size();
  std::vector<double> SavedFields( Nblocks*BSX*BSY* 8);
  #pragma omp parallel for
  for(size_t i=0; i< Nblocks; i++)
  {
    ScalarBlock& x0  = *(ScalarBlock*) xInfo[i].ptrBlock;
    ScalarBlock& x1  = *(ScalarBlock*) sInfo[i].ptrBlock;
    ScalarBlock& x2  = *(ScalarBlock*) zInfo[i].ptrBlock;
    ScalarBlock& x3  = *(ScalarBlock*)AxInfo[i].ptrBlock;
    VectorBlock& x45 = *(VectorBlock*)pvInfo[i].ptrBlock;
    VectorBlock& x67 = *(VectorBlock*) rInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8     ] = x0 (ix,iy).s;
      SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 1 ] = x1 (ix,iy).s;
      SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 2 ] = x2 (ix,iy).s;
      SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 3 ] = x3 (ix,iy).s;
      SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 4 ] = x45(ix,iy).u[0];
      SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 5 ] = x45(ix,iy).u[1];
      SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 6 ] = x67(ix,iy).u[0];
      SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 7 ] = x67(ix,iy).u[1];
      x0(ix,iy).s = 0.0;
    }
  }

  //Warning: AxInfo (sim.tmp) initially contains the RHS of the system!


  //1. r = RHS - Ax_0, x_0: initial solution guess
  //2. rhat = r
  //3. Initialize p = 0, v = 0
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
          VectorBlock & __restrict__ r    = *(VectorBlock*)   rInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ rhs  = *(ScalarBlock*)  AxInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
      r(ix,iy).u[0] = rhs(ix,iy).s;
  }

  Get_LHS(sim.tmp,sim.pres); // tmp <-- A*x_{0}
  
  double norm = 0.0; //initial residual norm
  #pragma omp parallel for reduction (+:norm)
  for (size_t i=0; i < Nblocks; i++)
  {
          VectorBlock & __restrict__ r    = *(VectorBlock*)   rInfo[i].ptrBlock;
          VectorBlock & __restrict__ pv   = *(VectorBlock*)  pvInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ lhs  = *(ScalarBlock*)  AxInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      r(ix,iy).u[0] -= lhs(ix,iy).s;
      r(ix,iy).u[1]  = r(ix,iy).u[0];
      pv(ix,iy).u[0] = 0.0;
      pv(ix,iy).u[1] = 0.0;
      norm += r(ix,iy).u[0]*r(ix,iy).u[0];
    }
  }
  const size_t Nsystem = BSX*BSY*Nblocks;
  norm = std::sqrt(norm)/Nsystem;

  std::cout << "-------------->[Poisson] : initial norm = " << norm << std::endl;

  std::vector <Real> x_opt (Nsystem);

  //4. some bi-conjugate gradient parameters
  double rho   = 1.0;
  double alpha = 1.0;
  double omega = 1.0;
  const double eps = 1e-21;
  const double max_error = 1e-13;
  const double max_rel_error = 1e-18;
  double min_norm = 1e50;
  double rho_m1;
  double init_norm=norm;
  bool useXopt = false;
  int iter_opt = 0;

  //5. start iterations
  for (size_t k = 0 ; k < 10000; k++)
  {
    //1. rho_{k} = rhat_0 * rho_{k-1}
    //2. beta = rho_{k} / rho_{k-1} * alpha/omega 
    rho_m1 = rho;
    rho = 0.0;
    #pragma omp parallel for reduction(+:rho)
    for(size_t i=0; i< Nblocks; i++)
    {
      const VectorBlock & __restrict__ r    = *(VectorBlock*) rInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
        rho += r(ix,iy).u[0] * r(ix,iy).u[1];
    }
    double beta = rho / (rho_m1+eps) * alpha / (omega+eps);

    //3. p_{k} = r_{k-1} + beta*(p_{k-1}-omega *v_{k-1})
    //4. z = K_2 ^{-1} p
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
            VectorBlock & __restrict__ pv = *(VectorBlock*) pvInfo[i].ptrBlock;
            ScalarBlock & __restrict__ z  = *(ScalarBlock*)  zInfo[i].ptrBlock;
      const VectorBlock & __restrict__ r  = *(VectorBlock*)  rInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        pv(ix,iy).u[0] = r(ix,iy).u[0] + beta* ( pv(ix,iy).u[0] - omega * pv(ix,iy).u[1]);
        z(ix,iy).s = pv(ix,iy).u[0];
      }
    }
    //getZ();

    //5. v = A z
    //6. alpha = rho_i / (rhat_0,v_i)
    alpha = 0.0;
    Get_LHS(sim.tmp,sim.pOld); // v <-- Az //v stored in AxVector
    #pragma omp parallel for reduction(+:alpha)
    for (size_t i=0; i < Nblocks; i++)
    {
            VectorBlock & __restrict__ pv = *(VectorBlock*) pvInfo[i].ptrBlock;
            ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
      const VectorBlock & __restrict__ r  = *(VectorBlock*)  rInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        pv(ix,iy).u[1] = Ax(ix,iy).s;
        alpha += r(ix,iy).u[1] * pv(ix,iy).u[1];
      }
    }  
    alpha = rho / (alpha + eps);


    //7. x += a z
    //8. 
    //9. s = r_{i-1}-alpha * v_i
    //10. z = K_2^{-1} s
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__ x = *(ScalarBlock*) xInfo[i].ptrBlock;
      ScalarBlock & __restrict__ z = *(ScalarBlock*) zInfo[i].ptrBlock;
      ScalarBlock & __restrict__ s = *(ScalarBlock*) sInfo[i].ptrBlock;
      VectorBlock & __restrict__ r = *(VectorBlock*) rInfo[i].ptrBlock;
      VectorBlock & __restrict__ pv= *(VectorBlock*)pvInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        x(ix,iy).s += alpha * z(ix,iy).s;
        s(ix,iy).s = r(ix,iy).u[0] - alpha * pv(ix,iy).u[1];
        z(ix,iy).s = s(ix,iy).s;
      }
    }


    //getZ();
    Get_LHS(sim.tmp,sim.pOld); // t <-- Az //t stored in AxVector

    //12. omega = ...
    double aux1 = 0;
    double aux2 = 0;
    #pragma omp parallel for reduction (+:aux1,aux2)
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
      ScalarBlock & __restrict__ s = *(ScalarBlock*) sInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        aux1 += Ax(ix,iy).s *  s(ix,iy).s;
        aux2 += Ax(ix,iy).s * Ax(ix,iy).s;
      }
    }
    omega = aux1 / (aux2+eps); 

    //13. x += omega * z
    //14.
    //15. r = s - omega * t
    norm = 0.0; 
    #pragma omp parallel for reduction(+:norm)
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
      ScalarBlock & __restrict__ x = *(ScalarBlock*) xInfo[i].ptrBlock;
      ScalarBlock & __restrict__ z = *(ScalarBlock*) zInfo[i].ptrBlock;
      ScalarBlock & __restrict__ s = *(ScalarBlock*) sInfo[i].ptrBlock;
      VectorBlock & __restrict__ r = *(VectorBlock*) rInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        x(ix,iy).s += omega * z(ix,iy).s;
        r(ix,iy).u[0] = s(ix,iy).s - omega * Ax(ix,iy).s;
        norm+= r(ix,iy).u[0]*r(ix,iy).u[0]; 
      }
    }
    norm = std::sqrt(norm) / Nsystem;

    if (norm < min_norm)
    {
      iter_opt = k;
      min_norm = norm;
      #pragma omp parallel for
      for (size_t i=0; i < Nblocks; i++)
      {
        ScalarBlock & __restrict__ x = *(ScalarBlock*) xInfo[i].ptrBlock;
        for(int iy=0; iy<VectorBlock::sizeY; iy++)
        for(int ix=0; ix<VectorBlock::sizeX; ix++)
        {
          x_opt[i*BSX*BSY + iy*BSX + ix] = x(ix,iy).s;
        }
      }
    }

    if (norm / (init_norm+eps) > 2.0 && k > 10)
    {
      useXopt = true;
      std::cout <<  "XOPT Poisson solver converged after " <<  k << " iterations. Error norm = " << norm << "  iter_opt="<< iter_opt << std::endl;
      break;
    }
    if ( (norm < max_error || norm/init_norm < max_rel_error ) && k > 3 )
    {
      std::cout <<  "Poisson solver converged after " <<  k << " iterations. Error norm = " << norm << std::endl;
      break;
    }
  }//k-loop

  if (useXopt)
  {
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__ x = *(ScalarBlock*) xInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        x(ix,iy).s = x_opt[i*BSX*BSY + iy*BSX + ix];
      }
    }
  }

  double avg = 0;
  for (size_t i=0; i < Nblocks; i++)
  {
    ScalarBlock & __restrict__ x = *(ScalarBlock*) xInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
    {
      avg += x(ix,iy).s;
    }
  }
  avg/=(Nblocks*BSX*BSY);

  #pragma omp parallel for
  for(size_t i=0; i< Nblocks; i++)
  {
    ScalarBlock& x0  = *(ScalarBlock*) xInfo[i].ptrBlock;
    ScalarBlock& x1  = *(ScalarBlock*) sInfo[i].ptrBlock;
    ScalarBlock& x2  = *(ScalarBlock*) zInfo[i].ptrBlock;
    ScalarBlock& x3  = *(ScalarBlock*)AxInfo[i].ptrBlock;
    VectorBlock& x45 = *(VectorBlock*)pvInfo[i].ptrBlock;
    VectorBlock& x67 = *(VectorBlock*) rInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
    {
      //x0 (ix,iy).s    = SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8     ];
      x0 (ix,iy).s    -= avg;
      x1 (ix,iy).s    = SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 1 ];
      x2 (ix,iy).s    = SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 2 ];
      x3 (ix,iy).s    = SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 3 ];
      x45(ix,iy).u[0] = SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 4 ];
      x45(ix,iy).u[1] = SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 5 ];
      x67(ix,iy).u[0] = SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 6 ];
      x67(ix,iy).u[1] = SavedFields[ i*(BSX*BSY*8) + iy*(BSX*8) + ix*8 + 7 ];
    }
  }




}
