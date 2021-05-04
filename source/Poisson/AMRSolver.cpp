//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "AMRSolver.h"

using namespace cubism;


void AMRSolver::getZ(std::vector<BlockInfo> & zInfo)
{
   static constexpr int BSX = VectorBlock::sizeX;
   static constexpr int BSY = VectorBlock::sizeY;
   static constexpr int N   = BSX*BSY;

   #pragma omp parallel
   {
     const int tid = omp_get_thread_num();
     #pragma omp for
     for (size_t i=0; i < zInfo.size(); i++)
     {
       ScalarBlock & __restrict__ z  = *(ScalarBlock*) zInfo[i].ptrBlock;

       //1. z = L^{-1}r
       for (int I = 0; I < N ; I++)
       {
         double rhs = 0.0;
         for (size_t jj = 0 ; jj < L_row[tid][I].size(); jj++)
         {
           const int J = L_row[tid][I][jj].first;
           const int iy = J / BSX;
           const int ix = J % BSX;
           rhs += L_row[tid][I][jj].second*z(ix,iy).s;
         }
         const int iy = I / BSX;
         const int ix = I % BSX;
         z(ix,iy).s = (z(ix,iy).s - rhs)*Ld[tid][I];
       }
       //2. z = L^T{-1}r
       for (int I = N-1; I >= 0 ; I--)
       {
         double rhs = 0.0;
         for (size_t jj = 0 ; jj < L_col[tid][I].size(); jj++)
         {
           const int J = L_col[tid][I][jj].first;
           const int iy = J / BSX;
           const int ix = J % BSX;
           rhs += L_col[tid][I][jj].second*z(ix,iy).s;
         }
         const int iy = I / BSX;
         const int ix = I % BSX;
         z(ix,iy).s = (z(ix,iy).s - rhs) *Ld[tid][I];
       }

       for (int iy=0;iy<BSY;iy++)
         for (int ix=0;ix<BSX;ix++)
           z(ix,iy).s = -z(ix,iy).s;
     }
   }
}

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
  
      #pragma omp for
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

double AMRSolver::getA_local(int I1,int I2)
{
   static constexpr int BSX = VectorBlock::sizeX;
   int j1 = I1 / BSX;
   int i1 = I1 % BSX;
   int j2 = I2 / BSX;
   int i2 = I2 % BSX;
   if (i1==i2 && j1==j2)
   {
     return 4.0;
   }
   else if (abs(i1-i2) + abs(j1-j2) == 1)
   {
     return -1.0;
   }
   else
   {
     return 0.0;
   }
}

AMRSolver::AMRSolver(SimulationData& s):sim(s)
{
   std::vector<std::vector<double>> L;

   int BSX = VectorBlock::sizeX;
   int BSY = VectorBlock::sizeY;
   int N = BSX*BSY;
   L.resize(N);

   for (int i = 0 ; i<N ; i++)
   {
     L[i].resize(i+1);
   }
   for (int i = 0 ; i<N ; i++)
   {
     double s1=0;
     for (int k=0; k<=i-1; k++)
       s1 += L[i][k]*L[i][k];
     L[i][i] = sqrt(getA_local(i,i) - s1);
     for (int j=i+1; j<N; j++)
     {
       double s2 = 0;
       for (int k=0; k<=i-1; k++)
         s2 += L[i][k]*L[j][k];
       L[j][i] = (getA_local(j,i)-s2) / L[i][i];
     }
   }
   L_row.resize(omp_get_max_threads());
   L_col.resize(omp_get_max_threads());
   Ld.resize(omp_get_max_threads());
   #pragma omp parallel
   {
     int tid = omp_get_thread_num();
     L_row[tid].resize(N);
     L_col[tid].resize(N);

     for (int i = 0 ; i<N ; i++)
     {
       Ld[tid].push_back(1.0/L[i][i]);
       for (int j = 0 ; j < i ; j++)
       {
         if ( abs(L[i][j]) > 1e-10 ) L_row[tid][i].push_back({j,L[i][j]});
       }
     }
     for (int j = 0 ; j<N ; j++)
     {
       for (int i = j ; i < N ; i++)
       {
         if ( abs(L[i][j]) > 1e-10 ) L_col[tid][j].push_back({i,L[i][j]});
       }
     }
   }
}

void AMRSolver::Jacobi(int iter_max)
{
  std::vector<cubism::BlockInfo>&  xInfo = sim.pres->getBlocksInfo();
  std::vector<cubism::BlockInfo>& AxInfo = sim.tmp ->getBlocksInfo();
  #pragma omp parallel
  {
    for (int iter = 0 ; iter < iter_max ; iter++)
    {
      //double norm = 0.0;
      static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
      ScalarLab lab;
      lab.prepare(*sim.pres, stenBeg, stenEnd, 1);
      #pragma omp for //reduction (+:norm)
      for (size_t i=0; i < xInfo.size(); i++)
      {
        //const double vol = xInfo[i].h*xInfo[i].h;
        lab.load(xInfo[i]);
        ScalarBlock & __restrict__ x   = *(ScalarBlock*)  xInfo[i].ptrBlock;
        ScalarBlock & __restrict__ rhs = *(ScalarBlock*) AxInfo[i].ptrBlock;
        for(int iy=0; iy<VectorBlock::sizeY; iy++)
        for(int ix=0; ix<VectorBlock::sizeX; ix++)
        {
          const double xnew =.25*(lab(ix+1,iy).s+lab(ix,iy+1).s+
                                  lab(ix-1,iy).s+lab(ix,iy-1).s-rhs(ix,iy).s);
          //norm += std::fabs(xnew-x(ix,iy).s)*vol;
          x(ix,iy).s = xnew;
        }
      }
    }
  }
}

void AMRSolver::solve()
{
  Jacobi(50);
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
  // At the same time, we start the solver as:
  //1. r = RHS - Ax_0, x_0: initial solution guess
  //2. rhat = r
  //3. Initialize p = 0, v = 0
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
          VectorBlock & __restrict__ r    = *(VectorBlock*)   rInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ rhs  = *(ScalarBlock*)  AxInfo[i].ptrBlock;
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
      r(ix,iy).u[0] = rhs(ix,iy).s;
    }
  }

  //Warning: AxInfo (sim.tmp) initially contains the RHS of the system!
  Get_LHS(sim.tmp,sim.pres); // tmp <-- A*x_{0}
  
  double norm = 0.0; //initial residual norm
  double norm_opt = 0.0; //initial residual norm
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
  norm = std::sqrt(norm);

  std::vector <Real> x_opt (Nsystem);

  //4. some bi-conjugate gradient parameters
  double rho   = 1.0;
  double alpha = 1.0;
  double omega = 1.0;
  const double eps = 1e-21;
  const double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol;
  const double max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
  double min_norm = 1e50;
  double rho_m1;
  double init_norm=norm;
  bool useXopt = false;
  int iter_opt = 0;
  bool serious_breakdown = false;

  //5. start iterations
  for (size_t k = 0 ; k < 2000; k++)
  {
    //1. rho_{k} = rhat_0 * rho_{k-1}
    //2. beta = rho_{k} / rho_{k-1} * alpha/omega 
    rho_m1 = rho;
    rho = 0.0;
    double norm_1 = 0.0;
    double norm_2 = 0.0;
    #pragma omp parallel for reduction(+:rho,norm_1,norm_2)
    for(size_t i=0; i< Nblocks; i++)
    {
      const VectorBlock & __restrict__ r    = *(VectorBlock*) rInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        rho += r(ix,iy).u[0] * r(ix,iy).u[1];
        norm_1 += r(ix,iy).u[0] * r(ix,iy).u[0];
        norm_2 += r(ix,iy).u[1] * r(ix,iy).u[1];
      }
    }
    double beta = rho / (rho_m1+eps) * alpha / (omega+eps);

    norm_1 = sqrt(norm_1);
    norm_2 = sqrt(norm_2);
    double cosTheta = rho/norm_1/norm_2; 
    serious_breakdown = std::fabs(cosTheta) < 1e-8;
    if (serious_breakdown)
    {
        beta = 0.0;
        rho = 0.0;
        #pragma omp parallel for reduction(+:rho)
        for(size_t i=0; i< Nblocks; i++)
        {
            VectorBlock & __restrict__ r    = *(VectorBlock*) rInfo[i].ptrBlock;
            for(int iy=0; iy<VectorBlock::sizeY; iy++)
            for(int ix=0; ix<VectorBlock::sizeX; ix++)
            {
                r(ix,iy).u[1] = r(ix,iy).u[0];
                rho += r(ix,iy).u[0]*r(ix,iy).u[1];
            }
        }
        std::cout << "  [Poisson solver]: restart at iteration:" << k << 
                     "  norm:"<< norm <<" init_norm:" << init_norm << std::endl;
    }
    //std::cout << k << " " << norm << std::endl;
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
    getZ(zInfo);

    //5. v = A z
    //6. alpha = rho_i / (rhat_0,v_i)
    alpha = 0.0;
    Get_LHS(sim.tmp,sim.pOld); // v <-- Az //v stored in AxVector

    //7. x += a z
    //8. 
    //9. s = r_{i-1}-alpha * v_i
    //10. z = K_2^{-1} s
    #pragma omp parallel
    {
        double alpha_t = 0.0;
        #pragma omp for
        for (size_t i=0; i < Nblocks; i++)
        {
                VectorBlock & __restrict__ pv = *(VectorBlock*) pvInfo[i].ptrBlock;
                ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
          const VectorBlock & __restrict__ r  = *(VectorBlock*)  rInfo[i].ptrBlock;
          for(int iy=0; iy<VectorBlock::sizeY; iy++)
          for(int ix=0; ix<VectorBlock::sizeX; ix++)
          {
            pv(ix,iy).u[1] = Ax(ix,iy).s;
            alpha_t += r(ix,iy).u[1] * pv(ix,iy).u[1];
          }
        }
        #pragma omp atomic
        alpha += alpha_t;
        
        #pragma omp barrier
        #pragma omp single
        {
            alpha = rho / (alpha + eps);
        }

        #pragma omp for
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
    }

    getZ(zInfo);
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
    norm = std::sqrt(norm);

    if (norm < min_norm)
    {
      norm_opt = norm;
      useXopt = true;
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

    if (norm / (init_norm+eps) > 1000.0 && k > 10)
    {
      useXopt = true;
      std::cout <<  "XOPT Poisson solver converged after " <<  k << " iterations. Error norm = " << norm << "  iter_opt="<< iter_opt << std::endl;
      break;
    }
    if ( (norm < max_error || norm/init_norm < max_rel_error ) && k > iter_min )
    {
      std::cout <<  "Poisson solver converged after " <<  k << " iterations. Error norm = " << norm << "   opt=" << norm_opt << std::endl;
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
  if (iter_min > 1) iter_min --;
}
