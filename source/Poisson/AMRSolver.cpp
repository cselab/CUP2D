//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "AMRSolver.h"

using namespace cubism;

void AMRSolver::getZ(BlockInfo & zInfo)
{
  const int BSX  = VectorBlock::sizeX;
  const int BSY  = VectorBlock::sizeY;
  const int N    = BSX*BSY;
  const int bpdx = sim.chi->getMaxBlocks()[0];
  const int bpdy = sim.chi->getMaxBlocks()[1];
  const int tid  = omp_get_thread_num();

  ScalarBlock & __restrict__ z  = *(ScalarBlock*) zInfo.ptrBlock;

  //This is done to preserve symmetries. All it does is change the order with
  //which the elements of z(x,y) are accessed (x=0,...,N-1 or x=N-1,...0, same for y)
  const Real c11 = z(BSX - 1, BSY - 1).s;
  const Real c10 = z(BSX - 1,       0).s;
  const Real c01 = z(0      , BSY - 1).s;
  const Real c00 = z(0      ,       0).s;
  bool plus_x = zInfo.index[0] > bpdx * (1 << zInfo.level) / 2 - 1;
  bool plus_y = zInfo.index[1] > bpdy * (1 << zInfo.level) / 2 - 1;
  if ( std::fabs(std::fabs(c11 + c10) - std::fabs(c01 + c00)) > 1e-14) plus_x = std::fabs(c11 + c10) > std::fabs(c01 + c00);
  if ( std::fabs(std::fabs(c11 + c01) - std::fabs(c10 + c00)) > 1e-14) plus_y = std::fabs(c11 + c01) > std::fabs(c10 + c00);
  const int base_x = plus_x ? 0 : BSX - 1;
  const int base_y = plus_y ? 0 : BSY - 1;
  const int sign_x = plus_x ? 1 : -1;
  const int sign_y = plus_y ? 1 : -1;

  //1. z = L^{-1}r
  for (int I = 0; I < N ; I++)
  {
      Real rhs = 0.0;
      for (size_t jj = 0 ; jj < L_row[tid][I].size(); jj++)
      {
          const int J = L_row[tid][I][jj].first;
          const int iy = base_y + sign_y * J / BSX;
          const int ix = base_x + sign_x * J % BSX;
          rhs += L_row[tid][I][jj].second*z(ix,iy).s;
      }
      const int iy = base_y + sign_y * I / BSX;
      const int ix = base_x + sign_x * I % BSX;
      z(ix,iy).s = (z(ix,iy).s - rhs)*Ld[tid][I];
  }

  //2. z = L^T{-1}r
  for (int I = N-1; I >= 0 ; I--)
  {
      Real rhs = 0.0;
      for (size_t jj = 0 ; jj < L_col[tid][I].size(); jj++)
      {
          const int J = L_col[tid][I][jj].first;
          const int iy = base_y + sign_y * J / BSX;
          const int ix = base_x + sign_x * J % BSX;
          rhs -= L_col[tid][I][jj].second*z(ix,iy).s;
      }
      const int iy = base_y + sign_y * I / BSX;
      const int ix = base_x + sign_x * I % BSX;
      z(ix,iy).s = (rhs - z(ix,iy).s) *Ld[tid][I];
  }
}

Real AMRSolver::getA_local(const int I1,const int I2)
{
  const int BSX = VectorBlock::sizeX;
  const int j1 = I1 / BSX;
  const int i1 = I1 % BSX;
  const int j2 = I2 / BSX;
  const int i2 = I2 % BSX;
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

AMRSolver::AMRSolver(SimulationData& ss):sim(ss),Get_LHS(ss)
{
  const int BSX = VectorBlock::sizeX;
  const int BSY = VectorBlock::sizeY;
  const int N = BSX*BSY;
  std::vector<std::vector<Real>> L(N);

  for (int i = 0 ; i<N ; i++)
  {
    L[i].resize(i+1);
  }
  for (int i = 0 ; i<N ; i++)
  {
    Real s1=0;
    for (int k=0; k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = sqrt(getA_local(i,i) - s1);
    for (int j=i+1; j<N; j++)
    {
      Real s2 = 0;
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
    const int tid = omp_get_thread_num();
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
      for (int i = j+1 ; i < N ; i++)
      {
        if ( abs(L[i][j]) > 1e-10 ) L_col[tid][j].push_back({i,L[i][j]});
      }
  }
}

void AMRSolver::solve(const ScalarGrid *input, ScalarGrid * const output)
{
  if (input != sim.tmp || output != sim.pres)
    throw std::invalid_argument("AMRSolver hardcoded to sim.tmp and sim.pres for now");

  //MPI stuff & grid size parameters
  //Warning: AxInfo (sim.tmp ) initially contains the RHS of the system!
  //Warning:  zInfo (sim.pres) initially contains the initial solution guess x0!
  auto & AxInfo         = sim.tmp ->getBlocksInfo(); //will store the LHS result
  auto &  zInfo         = sim.pres->getBlocksInfo(); //used for preconditioning
  const size_t Nblocks  = zInfo.size();
  const int BSX         = VectorBlock::sizeX;
  const int BSY         = VectorBlock::sizeY;
  const size_t N        = BSX*BSY*Nblocks;
  const MPI_Comm m_comm = sim.chi->getWorldComm();
  const int rank        = sim.rank;

  //Set some bi-conjugate gradient parameters
  Real rho                 = 1.0;
  Real alpha               = 1.0;
  Real omega               = 1.0;
  const Real eps           = 1e-100;
  const Real max_error     = sim.step < 10 ? 0.0 : sim.PoissonTol;
  const Real max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
  Real min_norm            = 1e50;
  bool useXopt             = false;
  bool serious_breakdown   = false;
  int restarts             = 0;
  const int max_restarts   = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
  bool bConverged          = false;
  Real norm                = 0.0;     //residual norm
  Real norm_opt            = 0.0; //residual norm (for best x)
  Real norm_1              = 0.0;
  Real norm_2              = 0.0;
  Real rho_m1              = rho;
  rho = 0.0;

  // The bi-conjugate gradient arrays:
  x    .resize(N);//     solution vector
  r    .resize(N);//     residual vector
  p    .resize(N);//    conjugate vector (need to set this to zero!)
  v    .resize(N);// bi-conjugate vector (need to set this to zero!)
  s    .resize(N);//            s-vector
  rhat .resize(N);// 2nd residual vector
  x_opt.resize(N);// solution with minimum residual

  //1. r = RHS - Ax_0, x_0: initial solution guess
  //   - (1a) We set r = RHS and store x0 in x
  #pragma omp parallel for
  for(size_t i=0; i< Nblocks; i++)
  {    
    ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) AxInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ z    = *(ScalarBlock*)  zInfo[i].ptrBlock;
    if( sim.bMeanConstraint )
      if (isCorner(AxInfo[i])) rhs(0,0).s = 0.0;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      p[i*BSX*BSY+iy*BSX+ix] = 0.0;
      v[i*BSX*BSY+iy*BSX+ix] = 0.0;
      r[i*BSX*BSY+iy*BSX+ix] = rhs(ix,iy).s;
      x[i*BSX*BSY+iy*BSX+ix] = z  (ix,iy).s;
    }
  }
  //   - (1b) We compute A*x0 (which is stored in AxInfo)
  Get_LHS(0);

  //   - (1c) We substract A*x0 from r. 
  //          Here we also set rhat = r and compute the initial guess error norm.
  #pragma omp parallel for reduction(+:rho,norm_1,norm_2)reduction(max:norm)
  for (size_t i=0; i < Nblocks; i++)
  {
    const ScalarBlock & __restrict__ lhs  = *(ScalarBlock*)  AxInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      const int idx = i*BSX*BSY+iy*BSX+ix;
      r[idx] -= lhs(ix,iy).s;
      rhat[idx]  = r[idx];
      norm = std::max(norm,std::fabs(r[idx]));
      rho    += r[idx] * rhat[idx];
      norm_1 += r[idx] * r[idx];
      norm_2 += rhat[idx] * rhat[idx];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_Real,MPI_MAX,m_comm);
  {
     Real quantities[3] = {rho,norm_1,norm_2};
     MPI_Allreduce(MPI_IN_PLACE,&quantities,3,MPI_Real,MPI_SUM,m_comm);
     rho = quantities[0]; norm_1 = quantities[1] ; norm_2 = quantities[2];
  }
  const Real init_norm=norm;
  if (rank == 0 && !sim.muteAll)
    std::cout << "  [Poisson solver]: Initial norm: " << init_norm << std::endl;


  MPI_Request request_norm;
  Real beta = 0.0;

  //3. start iterations
  int k;
  const int kmax = sim.maxPoissonIterations;
  for ( k = 0 ; k < kmax; k++)
  {
    //3. p_{k} = r_{k-1} + beta*(p_{k-1}-omega *v_{k-1})
    //4. z = K_2 ^{-1} p
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__ z  = *(ScalarBlock*)  zInfo[i].ptrBlock;
      for(int iy=0; iy<BSY; iy++)
      for(int ix=0; ix<BSX; ix++)
      {
        const int j = i*BSX*BSY+iy*BSX+ix;
        p[j] = r[j] + beta * (p[j] - omega * v[j]);
        z(ix,iy).s = p[j];
      }
      getZ(zInfo[i]);
    }

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
      Real alpha_t = 0.0;
      #pragma omp for
      for (size_t i=0; i < Nblocks; i++)
      {
        const ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
        for(int iy=0; iy<BSY; iy++)
        for(int ix=0; ix<BSX; ix++)
        {
          const int j = i*BSX*BSY+iy*BSX+ix;
          v[j] = Ax(ix,iy).s;
          alpha_t += rhat[j] * v[j];
        }
      }
      #pragma omp atomic
      alpha += alpha_t;
    }
    MPI_Allreduce(MPI_IN_PLACE,&alpha,1,MPI_Real,MPI_SUM,m_comm);
    alpha = rho / (alpha + eps);

    if ( k > 0 ) MPI_Waitall(1,&request_norm,MPI_STATUSES_IGNORE);
    if (norm / (init_norm + 1e-21) > 1e10)
    {
      if (rank == 0 && !sim.muteAll)
        std::cout << "   [Poisson solver]: early termination." << std::endl;
      break;
    }
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
      if (rank == 0 && !sim.muteAll)
        std::cout << "  [Poisson solver]: Converged after " << k << " iterations.\n";
      bConverged = true;
      break;
    }

    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      ScalarBlock & __restrict__ z = *(ScalarBlock*) zInfo[i].ptrBlock;
      for(int iy=0; iy<BSY; iy++)
      for(int ix=0; ix<BSX; ix++)
      {
        const int j = i*BSX*BSY+iy*BSX+ix;
        x[j] += alpha * z(ix,iy).s;
        s[j] = r[j] - alpha * v[j];
        z(ix,iy).s = s[j];
      }
      getZ(zInfo[i]);
    }

    Get_LHS(0); // t <-- Az //t stored in AxVector
    //12. omega = ...
    Real aux1 = 0;
    Real aux2 = 0;
    #pragma omp parallel for reduction (+:aux1,aux2)
    for (size_t i=0; i < Nblocks; i++)
    {
      const ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
      for(int iy=0; iy<BSY; iy++)
      for(int ix=0; ix<BSX; ix++)
      {
        const int j = i*BSX*BSY+iy*BSX+ix;
        aux1 += Ax(ix,iy).s *  s[j];
        aux2 += Ax(ix,iy).s * Ax(ix,iy).s;
      }
    }
    Real temp[2] = {aux1,aux2};
    MPI_Allreduce(MPI_IN_PLACE,&temp,2,MPI_Real,MPI_SUM,m_comm);
    aux1 = temp[0]; aux2 = temp[1] ;
    omega = aux1 / (aux2+eps); 

    //13. x += omega * z
    //14.
    //15. r = s - omega * t
    rho_m1 = rho;
    rho = 0.0;
    norm_1 = 0.0;
    norm_2 = 0.0;
    norm = 0;
    #pragma omp parallel for reduction(+:rho,norm_1,norm_2)reduction(max:norm)
    for (size_t i=0; i < Nblocks; i++)
    {
      const ScalarBlock & __restrict__ Ax = *(ScalarBlock*) AxInfo[i].ptrBlock;
      const ScalarBlock & __restrict__ z = *(ScalarBlock*) zInfo[i].ptrBlock;
      for(int iy=0; iy<BSY; iy++)
      for(int ix=0; ix<BSX; ix++)
      {
        const int j = i*BSX*BSY+iy*BSX+ix;
        x[j]   += omega * z(ix,iy).s;
        r[j]    = s[j] - omega * Ax(ix,iy).s;
        rho    += r[j] * rhat[j];
        norm_1 += r[j] * r[j];
        norm_2 += rhat[j] * rhat[j];
        norm = std::max(norm,std::fabs(r[j]));
      }
    }
    MPI_Iallreduce(MPI_IN_PLACE,&norm,1,MPI_Real,MPI_MAX,m_comm,&request_norm);

    Real quantities[3] = {rho,norm_1,norm_2};
    MPI_Allreduce(MPI_IN_PLACE,&quantities,3,MPI_Real,MPI_SUM,m_comm);
    rho = quantities[0]; norm_1 = quantities[1] ; norm_2 = quantities[2];

    beta = rho / (rho_m1+eps) * alpha / (omega+eps);

    //Check if restart should be made. If so, current solution estimate is used as an initial
    //guess and solver starts again.
    serious_breakdown = rho * rho < 1e-16 * norm_1 * norm_2;
    if (serious_breakdown && restarts < max_restarts)
    {
      if ( k > 0 ) MPI_Waitall(1,&request_norm,MPI_STATUSES_IGNORE);

      restarts ++;
      if (rank == 0 && !sim.muteAll)
        std::cout << "  [Poisson solver]: Restart at iteration: " << k << " norm: " << norm <<" Initial norm: " << init_norm << std::endl;
      beta = 0.0;
      rho = 0.0;
      #pragma omp parallel for reduction(+:rho)
      for(size_t i=0; i< N; i++)
      {
        rhat[i] = r[i];
        rho += r[i]*rhat[i];
        p[i] = 0;
        v[i] = 0;
      }
      MPI_Allreduce(MPI_IN_PLACE,&rho,1,MPI_Real,MPI_SUM,m_comm);
      alpha = 1.;
      omega = 1.;
      rho_m1 = 1.;
      beta = rho / (rho_m1+eps) * alpha / (omega+eps) ;
    }

  } //k-loop

  if (rank == 0 && !sim.muteAll)
  {
    if( bConverged )
      std::cout <<  " Error norm (relative) = " << norm_opt << "/" << max_error << " (" << norm_opt/init_norm  << "/" << max_rel_error << ")" << std::endl;
    else
      std::cout <<  "  [Poisson solver]: Iteration " << k << ". Error norm (relative) = " << norm_opt << "/" << max_error << " (" << norm_opt/init_norm  << "/" << max_rel_error << ")" << std::endl;
  }

  if (useXopt)
  {
    #pragma omp parallel for
    for (size_t i=0; i < N; i++)
      x[i] = x_opt[i];
  }

  Real avg = 0;
  Real avg1 = 0;
  #pragma omp parallel for reduction (+:avg,avg1)
  for(size_t i=0; i< Nblocks; i++)
  {
    ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
    const Real vv = zInfo[i].h*zInfo[i].h;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      P(ix,iy).s = x[i*BSX*BSY + iy*BSX + ix];
      avg += P(ix,iy).s * vv;
      avg1 += vv;
    }
  }
  Real quantities[2] = {avg,avg1};
  MPI_Allreduce(MPI_IN_PLACE,&quantities,2,MPI_Real,MPI_SUM,m_comm);
  avg = quantities[0]; avg1 = quantities[1] ;
  avg = avg/avg1;
  #pragma omp parallel for
  for(size_t i=0; i< Nblocks; i++)
  {
    ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
       P(ix,iy).s -= avg;
  }
}
