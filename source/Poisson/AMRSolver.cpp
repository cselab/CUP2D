//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "AMRSolver.h"

using namespace cubism;

void AMRSolver::Update_Vector(std::vector<BlockInfo>& aInfo, std::vector<BlockInfo>& bInfo, double c, std::vector<BlockInfo>& dInfo)
{
  //set a = b + c * d
  #pragma omp parallel for schedule(runtime)
  for(size_t i=0; i< aInfo.size(); i++)
  {
    ScalarBlock& a = *(ScalarBlock*)aInfo[i].ptrBlock;
    ScalarBlock& b = *(ScalarBlock*)bInfo[i].ptrBlock;
    ScalarBlock& d = *(ScalarBlock*)dInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
      a(ix,iy).s = b(ix,iy).s + c * d(ix,iy).s;
  }
}
void AMRSolver::Update_Vector1(std::vector<BlockInfo>& aInfo, double c, std::vector<BlockInfo>& dInfo)
{
  //set a = a + c * d
  #pragma omp parallel for schedule(runtime)
  for(size_t i=0; i< aInfo.size(); i++)
  {
    ScalarBlock& a = *(ScalarBlock*)aInfo[i].ptrBlock;
    ScalarBlock& d = *(ScalarBlock*)dInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
      a(ix,iy).s += c * d(ix,iy).s;
  }
}
void AMRSolver::Dot_Product(std::vector<BlockInfo>& aInfo, std::vector<BlockInfo>& bInfo, double & result)
{
  result = 0.0;
  #pragma omp parallel
  {
    double my_result = 0.0;
    #pragma omp for schedule(runtime)
    for(size_t i=0; i< aInfo.size(); i++)
    {
      ScalarBlock& a = *(ScalarBlock*)aInfo[i].ptrBlock;
      ScalarBlock& b = *(ScalarBlock*)bInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        my_result += a(ix,iy).s * b(ix,iy).s;
      }
    }
    #pragma omp atomic
        result += my_result;
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
      }
    }
}


#ifdef PRECOND

double getA_local(int I1,int I2)
{
  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

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

void AMRSolver::FindZ(std::vector<BlockInfo> & zInfo,std::vector<BlockInfo> & rInfo)
{
  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int N   = BSX*BSY;

  #pragma omp parallel
  { 
    int tid = omp_get_thread_num();
    #pragma omp for schedule(runtime)
    for (size_t i=0; i < zInfo.size(); i++)
    {
  
      ScalarBlock & __restrict__ r  = *(ScalarBlock*) rInfo[i].ptrBlock;
      ScalarBlock & __restrict__ z  = *(ScalarBlock*) zInfo[i].ptrBlock;
  
      //1. z = L^{-1}r
      for (int I = 0; I < N ; I++)
      {
        double rhs = 0.0;
        for (size_t jj = 0 ; jj < L_row[tid][I].size(); jj++)
        {
          int J = L_row[tid][I][jj].first;
          double LIJ = L_row[tid][I][jj].second;
  
          int iy = J / BSX;
          int ix = J % BSX;
          rhs += LIJ*z(ix,iy).s;
        }
        int iy = I / BSX;
        int ix = I % BSX;
        z(ix,iy).s = (r(ix,iy).s - rhs) *Ld[tid][I];//
      }
  
      //2. z = L^T{-1}r
      for (int I = N-1; I >= 0 ; I--)
      {
        double rhs = 0.0;
        for (size_t jj = 0 ; jj < L_col[tid][I].size(); jj++)
        {
          int J = L_col[tid][I][jj].first;
          double LJI = L_col[tid][I][jj].second;

          int iy = J / BSX;
          int ix = J % BSX;
          rhs += LJI*z(ix,iy).s;
        }
        int iy = I / BSX;
        int ix = I % BSX;
        z(ix,iy).s = (z(ix,iy).s - rhs) *Ld[tid][I];
      }

      for (int iy=0;iy<BSY;iy++)
        for (int ix=0;ix<BSX;ix++)
          z(ix,iy).s = -z(ix,iy).s;
    }
  }
}
#else
AMRSolver::AMRSolver(SimulationData& s):sim(s){}
#endif
  
void AMRSolver::solve()
{
  sim.startProfiler("AMRSolver");

  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

  //tmp stores the RHS of the system --> b or the current Ax
  //pres stores current estimate of the solution --> x
  //pold stores conjugate vector --> p 
  //prhs stores residual vector --> r
  //(yes, the names are terrible)
  std::vector<cubism::BlockInfo>& xInfo = sim.pres->getBlocksInfo();
  std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp ->getBlocksInfo();
  std::vector<cubism::BlockInfo>& pInfo = sim.pOld->getBlocksInfo();
  std::vector<cubism::BlockInfo>& rInfo = sim.pRHS->getBlocksInfo();
  
  #ifdef PRECOND
  std::vector<cubism::BlockInfo>& zInfo = sim.z_cg->getBlocksInfo();
  double rk_zk;
  double rkp1_zkp1;
  #endif

  std::vector<cubism::BlockInfo>& SavedSolInfo = sim.tmpV->getBlocksInfo();

  const size_t Nblocks = xInfo.size();
  const size_t Nsystem = Nblocks * BSX * BSY; //total number of unknowns
  const double max_error = 1e-6;

  double err_min = 1e50;
  double err=1;

  double alpha = 1.0;
  double beta  = 0.0;
  double rk_rk;
  double rkp1_rkp1;
  double pAp;

  //Set r_{0} = b
  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
          ScalarBlock & __restrict__ r  = *(ScalarBlock*) rInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ b  = *(ScalarBlock*) tmpInfo[i].ptrBlock;
    r.copy(b);
  }

  Get_LHS(sim.tmp,sim.pres); // tmp <-- A*x_{0}

  Update_Vector(rInfo,rInfo,-alpha,tmpInfo); // r_{0} <-- r_{0} - alpha * tmp

  Dot_Product(rInfo,rInfo,rk_rk);

#ifdef PRECOND
  FindZ(zInfo,rInfo);
  Dot_Product(rInfo,zInfo,rk_zk);
  //p_{0} = r_{0}
  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
          ScalarBlock & __restrict__ p  = *(ScalarBlock*) pInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ z  = *(ScalarBlock*) zInfo[i].ptrBlock;
    p.copy(z);
  }
#else
  //p_{0} = r_{0}
  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
          ScalarBlock & __restrict__ p  = *(ScalarBlock*) pInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ r  = *(ScalarBlock*) rInfo[i].ptrBlock;
    p.copy(r);
  }
#endif
 
  bool flag = false;
  
  int count = 0;

  //for (size_t k = 1; k < Nsystem; k++)  
  for (size_t k = 1; k < 1000; k++)
  {    
    count++;
    err = std::sqrt(rk_rk)/Nsystem;

    if (err < err_min && k >=2)
    {
      flag = true;
      err_min = err;
      #pragma omp parallel for schedule(runtime)
      for(size_t i=0; i< xInfo.size(); i++)
      {
        ScalarBlock& x = *(ScalarBlock*)xInfo[i].ptrBlock;
        VectorBlock& sol = *(VectorBlock*)SavedSolInfo[i].ptrBlock;
        for(int iy=0; iy<VectorBlock::sizeY; iy++)
        for(int ix=0; ix<VectorBlock::sizeX; ix++)
          sol(ix,iy).u[0] = x(ix,iy).s;
      }
    }

    if (err < max_error) break;

    if (  err/(err_min+1e-21) > 3.0 ) break; //error grows, stop iterations!

    Get_LHS(sim.tmp,sim.pOld); // tmp <-- A*p_{k}

    Dot_Product(pInfo,tmpInfo,pAp);

#ifdef PRECOND
    alpha = rk_zk / (pAp + 1e-21);
#else
    alpha = rk_rk / (pAp + 1e-21);
#endif

    Update_Vector1(xInfo, alpha,pInfo);//x_{k+1} <-- x_{k} + alpha * p_{k}
    Update_Vector1(rInfo,-alpha,tmpInfo);//r_{k+1} <-- r_{k} - alpha * tmp 
    Dot_Product(rInfo,rInfo,rkp1_rkp1);
    
#ifdef PRECOND    
    FindZ(zInfo,rInfo);
    Dot_Product(rInfo,zInfo,rkp1_zkp1);
    beta = rkp1_zkp1 / (rk_zk + 1e-21);
    rk_zk = rkp1_zkp1;  
    Update_Vector(pInfo,zInfo,beta,pInfo);//p_{k+1} <-- r_{k+1} + beta * p_{k}
#else
    beta = rkp1_rkp1 / (rk_rk + 1e-21);
    Update_Vector(pInfo,rInfo,beta,pInfo);//p_{k+1} <-- r_{k+1} + beta * p_{k}
#endif
    rk_rk = rkp1_rkp1;
  }


  if (flag)
  {
    #pragma omp parallel for schedule(runtime)
    for(size_t i=0; i< xInfo.size(); i++)
    {
      ScalarBlock& x = *(ScalarBlock*)xInfo[i].ptrBlock;
      VectorBlock& sol = *(VectorBlock*)SavedSolInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
        x(ix,iy).s = sol(ix,iy).u[0]; 
    }
  }
  sim.stopProfiler();
  std::cout << "CG Poisson solver took "<<count << " iterations. Final residual norm = "<< err_min << std::endl;
}