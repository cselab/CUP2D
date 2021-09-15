//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

// TODO: check if mean = 0 constraint leads to faster convergence

#include "AMRSolver.h"

using namespace cubism;

void AMRSolver::getZ(std::vector<BlockInfo> & zInfo)
{
  sim.startProfiler("Poisson:getZ");
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
   sim.stopProfiler();
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

AMRSolver::AMRSolver(SimulationData& s):sim(s),Get_LHS(s)
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

void AMRSolver::solve()
{
  const int BSX = VectorBlock::sizeX;
  const int BSY = VectorBlock::sizeY;
  const MPI_Comm m_comm = sim.chi->getCartComm();
  const int rank = sim.rank;

  // The bi-conjugate gradient method needs the following 8 arrays:
  // r: residual vector
  // rhat: (2nd residual vector)
  // p: conjugate vector
  // v: bi-conjugate vector
  // s vector
  // Ax vector (stores the LHS computation)
  // x vector (current solution estimate)
  // z vector (used for preconditioning)

  std::vector<cubism::BlockInfo>& AxInfo = sim.tmp ->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const size_t Nblocks = zInfo.size();
  const size_t N = BSX*BSY*Nblocks;
  std::vector<double> x   (N);
  std::vector<double> r   (N);
  std::vector<double> p   (N,0.0); //initialize p = 0
  std::vector<double> v   (N,0.0); //initialize v = 0
  std::vector<double> s   (N);
  std::vector<double> rhat(N);
  std::vector <Real> x_opt(N);


  //Warning: AxInfo (sim.tmp) initially contains the RHS of the system!
  //Warning:  zInfo (sim.pres) initially contains the initial solution guess x0!

  //1. r = RHS - Ax_0, x_0: initial solution guess
  //   - (1a) We set r = RHS and store x0 in x
  //#pragma omp parallel for
  for(size_t i=0; i< Nblocks; i++)
  {    
    ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) AxInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ z    = *(ScalarBlock*)  zInfo[i].ptrBlock;
    //if (isCorner(AxInfo[i])) rhs(4,4).s = 0.0;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      r[i*BSX*BSY+iy*BSX+ix] = rhs(ix,iy).s;
      x[i*BSX*BSY+iy*BSX+ix] = z  (ix,iy).s;
    }
  }
  //   - (1b) We compute A*x0 (which is stored in AxInfo)
  Get_LHS(0);

  //   - (1c) We substract A*x0 from r. 
  //          Here we also set rhat = r and compute the initial guess error norm.
  double norm = 0.0; //initial residual norm
  double norm_opt = 0.0; //initial residual norm
  double norm_max = 0.0;
  //#pragma omp parallel for reduction (+:norm)
  for (size_t i=0; i < Nblocks; i++)
  {
    const ScalarBlock & __restrict__ lhs  = *(ScalarBlock*)  AxInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      r[i*BSX*BSY+iy*BSX+ix] -= lhs(ix,iy).s;
      rhat[i*BSX*BSY+iy*BSX+ix]  = r[i*BSX*BSY+iy*BSX+ix];
      norm += r[i*BSX*BSY+iy*BSX+ix]*r[i*BSX*BSY+iy*BSX+ix];
      norm_max = std::max(norm_max,std::fabs(r[i*BSX*BSY+iy*BSX+ix]));
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_DOUBLE,MPI_SUM,m_comm);
  MPI_Allreduce(MPI_IN_PLACE,&norm_max,1,MPI_DOUBLE,MPI_MAX,m_comm);
  norm = std::sqrt(norm);
  norm = norm_max;

  //2. Set some bi-conjugate gradient parameters
  double rho   = 1.0;
  double alpha = 1.0;
  double omega = 1.0;
  const double eps = 1e-21;
  const double max_error     = sim.step < 10 ? 0.0 : sim.PoissonTol;//* sim.uMax_measured / sim.dt;
  const double max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;//min(1e-2,sim.PoissonTolRel * sim.uMax_measured / sim.dt );
  double min_norm = 1e50;
  double rho_m1;
  double init_norm=norm;
  bool useXopt = false;
  bool serious_breakdown = false;
  int restarts = 0;
  const int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
  bool bConverged = false;

  //3. start iterations
  size_t k;
  if (rank == 0) std::cout << "  [Poisson solver]: Initial norm: " << init_norm << std::endl;
  const int kmax = 1000;
  for ( k = 0 ; k < kmax; k++)
  {
    //1. rho_{k} = rhat_0 * rho_{k-1}
    //2. beta = rho_{k} / rho_{k-1} * alpha/omega 
    rho_m1 = rho;
    rho = 0.0;
    double norm_1 = 0.0;
    double norm_2 = 0.0;
    norm_max = 0;
    #pragma omp parallel for reduction(+:rho,norm_1,norm_2)
    for(size_t i=0; i< N; i++)
    {
      rho    += r[i] * rhat[i];
      norm_1 += r[i] * r[i];
      norm_2 += rhat[i] * rhat[i];
      #pragma omp critical
      norm_max = std::max(norm_max,std::fabs(r[i]));
    }
    norm = norm_max;
    MPI_Allreduce(MPI_IN_PLACE,&norm_max,1,MPI_DOUBLE,MPI_MAX,m_comm);
    norm = norm_max;
    if (norm < min_norm)
    {
      norm_opt = norm;
      useXopt = true;
      min_norm = norm;
      #pragma omp parallel for
      for (size_t i=0; i < N; i++)
        x_opt[i] = x[i];
    }
    if ( (norm < max_error || norm/init_norm < max_rel_error ) )
    {
      if (rank == 0) std::cout << "  [Poisson solver]: Converged after " << k << " iterations.";
      bConverged = true;
      break;
    }
    double quantities[3] = {rho,norm_1,norm_2};
    MPI_Allreduce(MPI_IN_PLACE,&quantities,3,MPI_DOUBLE,MPI_SUM,m_comm);
    rho = quantities[0]; norm_1 = quantities[1] ; norm_2 = quantities[2];

    double beta = rho / (rho_m1+eps) * alpha / (omega+eps);
    norm_1 = sqrt(norm_1);
    norm_2 = sqrt(norm_2);
    const double cosTheta = rho/norm_1/norm_2; 
    serious_breakdown = std::fabs(cosTheta) < 1e-8;
    if (serious_breakdown)
    {
        restarts ++;
        if (restarts >= max_restarts)
        {
           break;
        }
        if (rank == 0) std::cout << "  [Poisson solver]: Restart at iteration: " << k << " norm: " << norm <<" Initial norm: " << init_norm << std::endl;
        beta = 0.0;
        rho = 0.0;
        #pragma omp parallel for reduction(+:rho)
        for(size_t i=0; i< N; i++)
        {
          rhat[i] = r[i];
          rho += r[i]*rhat[i];
        }
        MPI_Allreduce(MPI_IN_PLACE,&rho,1,MPI_DOUBLE,MPI_SUM,m_comm);
        alpha = 1.;
        omega = 1.;
        rho_m1 = 1.;
        beta = rho / (rho_m1+eps) * alpha / (omega+eps) ;
    }

    //3. p_{k} = r_{k-1} + beta*(p_{k-1}-omega *v_{k-1})
    //4. z = K_2 ^{-1} p
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__ z  = *(ScalarBlock*)  zInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        const int j = i*BSX*BSY+iy*BSX+ix;
        p[j] = r[j] + beta * (p[j] - omega * v[j]);
        z(ix,iy).s = p[j];
      }
    } 
    getZ(zInfo);

    //5. v = A z
    //6. alpha = rho_i / (rhat_0,v_i)
    alpha = 0.0;
    Get_LHS(0);// v <-- Az //v stored in AxVector
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
          const ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
          for(int iy=0; iy<VectorBlock::sizeY; iy++)
          for(int ix=0; ix<VectorBlock::sizeX; ix++)
          {
            const int j = i*BSX*BSY+iy*BSX+ix;
            v[j] = Ax(ix,iy).s;
            alpha_t += rhat[j] * v[j];
          }
        }
        #pragma omp atomic
        alpha += alpha_t;
    }
    MPI_Allreduce(MPI_IN_PLACE,&alpha,1,MPI_DOUBLE,MPI_SUM,m_comm);
    alpha = rho / (alpha + eps);
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__ z = *(ScalarBlock*) zInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        const int j = i*BSX*BSY+iy*BSX+ix;
        x[j] += alpha * z(ix,iy).s;
        s[j] = r[j] - alpha * v[j];
        z(ix,iy).s = s[j];
      }
    }

    getZ(zInfo);
    Get_LHS(0); // t <-- Az //t stored in AxVector
    //12. omega = ...
    double aux1 = 0;
    double aux2 = 0;
    #pragma omp parallel for reduction (+:aux1,aux2)
    for (size_t i=0; i < Nblocks; i++)
    {
      const ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        const int j = i*BSX*BSY+iy*BSX+ix;
        aux1 += Ax(ix,iy).s *  s[j];
        aux2 += Ax(ix,iy).s * Ax(ix,iy).s;
      }
    }
    double temp[2] = {aux1,aux2};
    MPI_Allreduce(MPI_IN_PLACE,&temp,2,MPI_DOUBLE,MPI_SUM,m_comm);
    aux1 = temp[0]; aux2 = temp[1] ;
    omega = aux1 / (aux2+eps); 

    //13. x += omega * z
    //14.
    //15. r = s - omega * t
    for (size_t i=0; i < Nblocks; i++)
    {
      const ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
      const ScalarBlock & __restrict__ z = *(ScalarBlock*) zInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        const int j = i*BSX*BSY+iy*BSX+ix;
        x[j] += omega * z(ix,iy).s;
        r[j] = s[j] - omega * Ax(ix,iy).s;
      }
    }
    //if (rank == 0 && k % 50 == 0)
    //  std::cout << "  iter:" << k << " norm:" << norm << " opt:" << norm_opt << std::endl;
  } //k-loop
  if (rank == 0)
  {
    if( bConverged )
      std::cout <<  " Error norm (relative) = " << norm_opt << "/" << max_error << " (" << norm_opt/init_norm  << "/" << max_rel_error << ")" << std::endl;
    else
      std::cout <<  "  [Poisson solver]: Iteration " << k << ". Error norm (relative) = " << norm_opt << "/" << max_error << " (" << norm_opt/init_norm  << "/" << max_rel_error << ")" << std::endl;
  std::cout << "max = " << norm_max << std::endl;
  }

  if (useXopt)
  {
    #pragma omp parallel for
    for (size_t i=0; i < N; i++)
      x[i] = x_opt[i];
  }
  double avg = 0;
  double avg1 = 0;
  //#pragma omp parallel
  {
     //#pragma omp for reduction (+:avg,avg1)
     //#pragma omp parallel for
     for(size_t i=0; i< Nblocks; i++)
     {
        ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
        const double vv = zInfo[i].h*zInfo[i].h;
        for(int iy=0; iy<VectorBlock::sizeY; iy++)
        for(int ix=0; ix<VectorBlock::sizeX; ix++)
        {
            P(ix,iy).s = x[i*BSX*BSY + iy*BSX + ix];
            avg += P(ix,iy).s * vv;
            avg1 += vv;
        }
     }
     ////#pragma omp barrier
     ////#pragma omp master
     ////{
        double quantities[2] = {avg,avg1};
        MPI_Allreduce(MPI_IN_PLACE,&quantities,2,MPI_DOUBLE,MPI_SUM,m_comm);
        avg = quantities[0]; avg1 = quantities[1] ;
        avg = avg/avg1;
     ////}
     ////#pragma omp for
#if 1
     for(size_t i=0; i< Nblocks; i++)
     {
        ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
        for(int iy=0; iy<VectorBlock::sizeY; iy++)
        for(int ix=0; ix<VectorBlock::sizeX; ix++)
           P(ix,iy).s -= avg;
     }
     //if (rank == 0) std::cout << " Poisson solver avg=" << avg << std::endl;
#endif
  }
}
