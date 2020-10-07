//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chatzimanolakis (michaich@ethz.ch).
//

#include "AMRSolver.h"
#include "cblas.h"

using namespace cubism;

#if 1 //Conjugate gradient


#if 0
void Update_Vector(std::vector<BlockInfo>& aInfo, std::vector<BlockInfo>& bInfo, double c, std::vector<BlockInfo>& dInfo)
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

void Update_Vector1(std::vector<BlockInfo>& aInfo, double c, std::vector<BlockInfo>& dInfo)
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

void Dot_Product(std::vector<BlockInfo>& aInfo, std::vector<BlockInfo>& bInfo, double & result)
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

void Get_LHS (ScalarGrid * lhs, ScalarGrid * x)
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


  std::vector<cubism::BlockInfo>& DogShit = sim.tmpV->getBlocksInfo();

  size_t Nblocks = xInfo.size();
  size_t Nsystem = Nblocks * BSX * BSY; //total number of unknowns
  double max_error = 1e-8;

  double err_min = 1e50;
  double err=1;


  double alpha = 1.0;
  double beta  = 0.0;
  double rk_rk;
  double rkp1_rkp1;
  double pAp;


  int counter=0;

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

  //p_{0} = r_{0}
  #pragma omp parallel for schedule(runtime)
  for (size_t i=0; i < Nblocks; i++)
  {
          ScalarBlock & __restrict__ p  = *(ScalarBlock*) pInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ r  = *(ScalarBlock*) rInfo[i].ptrBlock;
    p.copy(r);
  }
  
  bool flag = false;
  //for (size_t k = 1; k < Nsystem; k++)
  for (size_t k = 1; k < 100; k++)
  {
    
    err = std::sqrt(rk_rk)/Nsystem;
    
    if (err < err_min && k>=3)
    {
        counter = 0;
        flag = true;
        //std::cout << "min res: " << k << std::endl;
        err_min = err;
        #pragma omp parallel for schedule(runtime)
        for(size_t i=0; i< xInfo.size(); i++)
        {
          ScalarBlock& x = *(ScalarBlock*)xInfo[i].ptrBlock;
          VectorBlock& sol = *(VectorBlock*)DogShit[i].ptrBlock;
          for(int iy=0; iy<VectorBlock::sizeY; iy++)
          for(int ix=0; ix<VectorBlock::sizeX; ix++)
            sol(ix,iy).u[0] = x(ix,iy).s;
        }
    }
    else if (k>=10)
    {
        counter ++;
    }

    if (err < max_error) break;
    //if (k%10 ==0) 
        std::cout << k << " " << err << "   " << pAp << std::endl;

    if (counter == 5) break;

    Get_LHS(sim.tmp,sim.pOld); // tmp <-- A*p_{k}

    Dot_Product(pInfo,tmpInfo,pAp);

    alpha = rk_rk / (pAp + 1e-21);

    //Update_Vector(xInfo,xInfo, alpha,pInfo);//x_{k+1} <-- x_{k} + alpha * p_{k}
    //Update_Vector(rInfo,rInfo,-alpha,tmpInfo);//r_{k+1} <-- r_{k} - alpha * tmp
    Update_Vector1(xInfo, alpha,pInfo);//x_{k+1} <-- x_{k} + alpha * p_{k}
    Update_Vector1(rInfo,-alpha,tmpInfo);//r_{k+1} <-- r_{k} - alpha * tmp 

    Dot_Product(rInfo,rInfo,rkp1_rkp1);
    
    beta = rkp1_rkp1 / (rk_rk + 1e-21);
    rk_rk = rkp1_rkp1;  

    Update_Vector(pInfo,rInfo,beta,pInfo);//p_{k+1} <-- r_{k+1} + beta * p_{k}
  }


  if (flag)
  {
    #pragma omp parallel for schedule(runtime)
    for(size_t i=0; i< xInfo.size(); i++)
    {
      ScalarBlock& x = *(ScalarBlock*)xInfo[i].ptrBlock;
      VectorBlock& sol = *(VectorBlock*)DogShit[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
        x(ix,iy).s = sol(ix,iy).u[0]; 
    }
  }
  sim.stopProfiler();
}


#else
void Update_Vector(std::vector<double>& a, std::vector<double>& b, double c)
{
  #pragma omp parallel for
  for(size_t i=0; i< a.size(); i++)
      a[i] = c * a[i] + b[i];
}

void Update_Vector1(std::vector<double>& a, double c, std::vector<double>& d)
{
  #pragma omp parallel for
  for(size_t i=0; i< a.size(); i++)
  {
      a[i] += c * d[i];
  }
}

void Dot_Product(std::vector<double> & a, std::vector<double>& b, double & result)
{
  result = 0.0;
  #pragma omp parallel for reduction(+: result)
  for(size_t i=0; i< a.size(); i++)
    result += a[i]*b[i];
}

void Get_LHS (ScalarGrid * tmp, std::vector<double> & x, std::vector<double> & Ax)
{
    static constexpr int BSX = VectorBlock::sizeX;
    static constexpr int BSY = VectorBlock::sizeY;

    //compute A*x and store it into LHS
    //here A corresponds to the discrete Laplacian operator for an AMR mesh
    const std::vector<cubism::BlockInfo>& tmpInfo = tmp->getBlocksInfo();

    #pragma omp parallel for
    for (size_t i=0; i < tmpInfo.size(); i++)
    {
      ScalarBlock & __restrict__ TMP  = *(ScalarBlock*) tmpInfo[i].ptrBlock;
      for(int iy=0; iy<BSY; iy++)
      for(int ix=0; ix<BSX; ix++)
      {
        TMP(ix,iy).s = x[ i * (BSY*BSX) + iy*BSX + ix ]; 
      }
    }

    #pragma omp parallel
    {
      static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
      ScalarLab lab; 
      lab.prepare(*tmp, stenBeg, stenEnd, 1);
  
      #pragma omp for schedule(runtime)
      for (size_t i=0; i < tmpInfo.size(); i++)
      {
        lab.load(tmpInfo[i]); 
  
        for(int iy=0; iy<BSY; ++iy)
        for(int ix=0; ix<BSX; ++ix)
        {
          Ax[ i * (BSY*BSX) + iy*BSX + ix ] = ( lab(ix-1,iy).s + 
                                                lab(ix+1,iy).s + 
                                                lab(ix,iy-1).s + 
                                                lab(ix,iy+1).s - 4.0*lab(ix,iy).s );
        }
      }
    }
}

void AMRSolver::solve()
{
  sim.startProfiler("AMRSolver");

  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

  std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  std::vector<cubism::BlockInfo>& presInfo = sim.pres->getBlocksInfo();

  size_t Nblocks = tmpInfo.size();

  const size_t Nsystem = Nblocks * BSX * BSY; //total number of unknowns
  std::vector <double> r (Nsystem);
  std::vector <double> p (Nsystem);
  std::vector <double> x (Nsystem);
  std::vector <double> Ap(Nsystem);
  std::vector <double> x_stored(Nsystem);
 
  double max_error = 1e-8;
  double err_min = 1e50;
  double err=1;
  double alpha = 1.0;
  double beta  = 0.0;
  double rk_rk;
  double rkp1_rkp1;
  double pAp;
  int counter=0;

  //Set r_{0} = b
  for (size_t i=0; i < Nblocks; i++)
  {
    const ScalarBlock & __restrict__ b  = *(ScalarBlock*) tmpInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ pres  = *(ScalarBlock*) presInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      r[ i * (BSY*BSX) + iy*BSX + ix ] = b(ix,iy).s;
      x[ i * (BSY*BSX) + iy*BSX + ix ] = pres(ix,iy).s;
    }
  }

  Get_LHS(sim.tmp,x,Ap); // Ap <-- A*x_{0}
  Update_Vector1(r,-alpha,Ap); // r_{0} <-- r_{0} - alpha * Ap
  Dot_Product(r,r,rk_rk);
  p = r;//p_{0}=r_{0}


  bool flag = false;
  for (size_t k = 1; k < 300; k++)
  {  
    err = std::sqrt(rk_rk)/Nsystem;
    
    if (err < err_min && k>=5)
    {
        counter = 0;
        flag = true;
        err_min = err;
        x_stored = x;
    }
    else if (k>=10)
    {
        counter ++;
    }

    if (err < max_error) break;
    //if (k%10 ==0) std::cout << k << " " << err << "   " << pAp << std::endl;
    if (counter == 5) break;

    sim.startProfiler("LHS");
    Get_LHS(sim.tmp,p,Ap); // A*p_{k}
    sim.stopProfiler();

    Dot_Product(p,Ap,pAp);

    alpha = rk_rk / (pAp + 1e-21);

    Update_Vector1(x, alpha,p );//x_{k+1} <-- x_{k} + alpha * p_{k}
    Update_Vector1(r,-alpha,Ap);//r_{k+1} <-- r_{k} - alpha * Ap 

    Dot_Product(r,r,rkp1_rkp1);
    
    beta = rkp1_rkp1 / (rk_rk + 1e-21);
    rk_rk = rkp1_rkp1;  

    Update_Vector(p,r,beta);//p_{k+1} <-- r_{k+1} + beta * p_{k}
  }

  if (flag)
    x = x_stored; 
  
  for (size_t i=0; i < Nblocks; i++)
  {
    ScalarBlock & __restrict__ pres  = *(ScalarBlock*) presInfo[i].ptrBlock;
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      pres(ix,iy).s = x[ i * (BSY*BSX) + iy*BSX + ix ];
    }
  }

  sim.stopProfiler();
}


#endif


#else //Jacobi

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
        double s = 0.0;
      for (size_t i=0; i < Nblocks; i++)
      {
        ScalarBlock & __restrict__ p  = *(ScalarBlock*) presInfo[i].ptrBlock;
        for(int iy=0; iy<BSY; ++iy)
        for(int ix=0; ix<BSX; ++ix)
           s+= p(ix,iy).s;    
      }
      std::cout << "s=" << s << std::endl;
    
    }


#endif