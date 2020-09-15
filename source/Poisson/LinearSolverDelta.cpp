//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Michalis Chtzimanolakis (michaich@ethz.ch).
//


/*
Linear solver used to solve the linearized system that arises for implicit advection-diffusion.

Given the velocities at time t = n, we wish to find the velocities (u,v) at time t=n+1 so that

R_ij^{x} := ADVECTION_ij^{x}(u,v) + DIFFUSION_ij^{x}(u,v) = 0
R_ij^{y} := ADVECTION_ij^{y}(u,v) + DIFFUSION_ij^{y}(u,v) = 0

where 

  DIFFUSION_ij^{x}(u,v) = 4.0 * u_{i,j} - (u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1})
  DIFFUSION_ij^{y}(u,v) = 4.0 * v_{i,j} - (v_{i+1,j}+v_{i-1,j}+v_{i,j+1}+v_{i,j-1})

  and ADVECTION_ij^{x}(u,v),ADVECTION_ij^{y}(u,v) are 1st (soon 3rd) order upwind advection terms

The non-linear system R_ij^{x}=0, R_ij^{y}=0 is linearized through Newton's method:

  dR/d(u,v) delta(u,v) = - (R_ij^{x},R_ij^{y})  (1)
  
  where dR/d(u,v) is the (approximate) Jacobian of the residuals R_ij^{x},R_ij^{y} with respect to 
  the velocities and delta(u,v) are the velocity corrections used to update the current estimates
  for (u,v).

Here, the linearized system (1) is decoupled in two smaller systems for the corrections du and dv
by assuming the following approximation for the Jacobian 

  dR_ij^{x}/dv = 0 , dR_ij^{y}/du = 0

Zero Neumann boundary conditions are assumed.
*/

#include "LinearSolverDelta.h"

using namespace cubism;

void LinearSolverDelta::allocLHS()
{
    sim.startProfiler("HYPRE SETUP");
    const auto COMM = MPI_COMM_SELF;

    HYPRE_StructHybridCreate(COMM, &hypre_solver_X);
    HYPRE_StructHybridSetTol(hypre_solver_X, 1e-2);
    HYPRE_StructHybridSetPrintLevel(hypre_solver_X, 0);
    HYPRE_StructHybridSetSolverType (hypre_solver_X, 1);

    HYPRE_StructHybridCreate(COMM, &hypre_solver_Y);
    HYPRE_StructHybridSetTol(hypre_solver_Y, 1e-2);
    HYPRE_StructHybridSetPrintLevel(hypre_solver_Y, 0);
    HYPRE_StructHybridSetSolverType (hypre_solver_Y, 1);

    HYPRE_StructHybridSetup(hypre_solver_X, hypre_mat_X, hypre_rhs_X, hypre_sol_X);
    HYPRE_StructHybridSetup(hypre_solver_Y, hypre_mat_Y, hypre_rhs_Y, hypre_sol_Y);
    sim.stopProfiler();
}

void LinearSolverDelta::cub2rhs(const std::vector<BlockInfo>& BSRC, const bool solveX)
{
    const size_t nBlocks = BSRC.size();
    Real * __restrict__ const dest = buffer;
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nBlocks; ++i)
    {
      int comp = solveX ? 0 : 1;
      const BlockInfo& info = BSRC[i];
      const size_t blocki = VectorBlock::sizeX * info.index[0];
      const size_t blockj = VectorBlock::sizeY * info.index[1];
      const VectorBlock& b = *(VectorBlock*)info.ptrBlock;

      const size_t blockStart = blocki + stride * blockj;
  
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
        dest[blockStart + ix + stride*iy] = b(ix,iy).u[comp];
    }
    #ifdef _FLOAT_PRECISION_
      std::copy(buffer, buffer + totNy*totNx, dbuffer);
    #endif
}

void LinearSolverDelta::cub2LHS(const std::vector<BlockInfo>& BSRC, const Real Uinf, const Real Vinf)
{
    const size_t nBlocks = BSRC.size();
    double * __restrict__ const dest_u = u;
    double * __restrict__ const dest_v = v;
    #pragma omp parallel for schedule(static) 
    for(size_t i=0; i<nBlocks; ++i)
    {
      const BlockInfo& info = BSRC[i];
      const size_t blocki = VectorBlock::sizeX * info.index[0];
      const size_t blockj = VectorBlock::sizeY * info.index[1];
      
      const VectorBlock& b = *(VectorBlock*)info.ptrBlock;  
      const size_t blockStart = blocki + stride * blockj;
  
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        dest_u[blockStart + ix + stride*iy] = b(ix,iy).u[0];
        dest_v[blockStart + ix + stride*iy] = b(ix,iy).u[1];
      } 
    }

    const Real h  = sim.getH();
    const Real dt = sim.dt;
    const Real nu = sim.nu;
    const Real dfac = nu*dt/h/h;
    const Real afac = dt/h;

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < totNy; j++)
    for (size_t i = 0; i < totNx; i++) 
    {
      const Real uij   = u[ j*totNx + i ];
      const Real vij   = v[ j*totNx + i ];
  
      const int su = (uij + Uinf > 0) ? 1:-1;
      const int sv = (vij + Vinf > 0) ? 1:-1;
    
      //double cfl = 0.4;
      //double dtau = cfl * h /(1e-10 + std::pow((uij+Uinf)*(uij+Uinf)+(vij+Vinf)*(vij+Vinf),0.5));   
      //double precond1 = 1.0/(dtau);
      double precond1 = 1e-21;
  
      // center
      vals_x[j*totNx + i][0] =  1.0 + 4.0*dfac  
                            + afac*( su*(2.0*uij + Uinf - u[j*totNx + i-su])+sv*(vij+Vinf))+ precond1; 
      // center
      vals_y[j*totNx + i][0] =  1.0 + 4.0*dfac 
                            + afac*( su*(uij+Uinf) +sv*(2.0*vij+Vinf-v[(j-sv)*totNx + i])) + precond1;
  
      vals_x[j*totNx + i][1] = -dfac; // west
      vals_x[j*totNx + i][2] = -dfac; // east
      vals_x[j*totNx + i][3] = -dfac; // south
      vals_x[j*totNx + i][4] = -dfac; // north
  
      vals_y[j*totNx + i][1] = -dfac; // west
      vals_y[j*totNx + i][2] = -dfac; // east
      vals_y[j*totNx + i][3] = -dfac; // south
      vals_y[j*totNx + i][4] = -dfac; // north
  
      if (su == 1)
      {
        vals_x[j*totNx + i][1] -= afac * (uij + Uinf);
        vals_y[j*totNx + i][1] -= afac * (uij + Uinf);
      }
      else
      {
        vals_x[j*totNx + i][2] += afac * (uij + Uinf);
        vals_y[j*totNx + i][2] += afac * (uij + Uinf);
      }
  
      if (sv == 1)
      {
        vals_x[j*totNx + i][3] -= afac * (vij + Vinf);
        vals_y[j*totNx + i][3] -= afac * (vij + Vinf);
      }
      else
      {
        vals_x[j*totNx + i][4] += afac * (vij + Vinf);
        vals_y[j*totNx + i][4] += afac * (vij + Vinf);
      }
    }

    // zero Neumann BC
    {
      #pragma omp parallel for schedule(static)
      for (size_t j = 0; j < totNy; j++)
      {
          double precond1 = 1e-21;
  
          //first west column
          Real uij   = u[j*totNx + 0];
          Real vij   = v[j*totNx + 0];
          int su = (uij + Uinf > 0) ? 1:-1;
          int sv = (vij + Vinf > 0) ? 1:-1; 
          if (su == 1) // (uij+Uinf>0) d/dx = 0
          {
              vals_x[j*totNx + 0][0] =  1.0 + 4.0*dfac + afac*( sv*(vij+Vinf))+ precond1;
              vals_y[j*totNx + 0][0] =  1.0 + 4.0*dfac + afac*( sv*(2.0*vij+Vinf-v[(j-sv)*totNx + 0])) + precond1;
          }
          vals_x[totNx*j +       0][0] -= dfac; // center
          vals_y[totNx*j +       0][0] -= dfac; // center
          vals_x[totNx*j +       0][1]  = 0; // west
          vals_y[totNx*j +       0][1]  = 0; // west
  
          //last east column
          uij   = u[j*totNx + totNx-1];
          vij   = v[j*totNx + totNx-1];
          su = (uij + Uinf > 0) ? 1:-1;
          sv = (vij + Vinf > 0) ? 1:-1; 
          if (su != 1) // (uij+Uinf<0) d/dx = 0
          {
            vals_x[j*totNx + totNx-1][0] =  1.0 + 4.0*dfac + afac*( sv*(vij+Vinf))+ precond1;
            vals_y[j*totNx + totNx-1][0] =  1.0 + 4.0*dfac + afac*( sv*(2.0*vij+Vinf-v[(j-sv)*totNx + totNx-1])) + precond1;
          }
          vals_x[totNx*j + totNx-1][0] -= dfac; // center
          vals_y[totNx*j + totNx-1][0] -= dfac; // center
          vals_x[totNx*j + totNx-1][2]  = 0; // east
          vals_y[totNx*j + totNx-1][2]  = 0; // east
      }
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < totNx; i++)
      {
          double precond1 = 1e-21;
  
          // first south row
          Real uij   = u[ 0*totNx + i ];
          Real vij   = v[ 0*totNx + i ];
          int su = (uij + Uinf > 0) ? 1:-1;
          int sv = (vij + Vinf > 0) ? 1:-1; 
          if (sv == 1) // (vij + V > 0) d/dy=0
          {
              if ( (su == 1 && i == 0) || (su == -1 && i == totNx-1) ) su = 0; //take care of corners
              vals_x[totNx*(0) +i][0] =  1.0 + 4.0*dfac + afac*( su*(2.0*uij + Uinf - u[0*totNx + i-su]) )+ precond1;
              vals_y[totNx*(0) +i][0] =  1.0 + 4.0*dfac + afac*( su*(uij+Uinf) )+ precond1;
          }
          vals_x[totNx*(0)       +i][0] -= dfac; // center
          vals_y[totNx*(0)       +i][0] -= dfac; // center
          vals_x[totNx*(0)       +i][3]  = 0; // south
          vals_y[totNx*(0)       +i][3]  = 0; // south
  
          // last north row
          uij   = u[ (totNy-1)*totNx + i ];
          vij   = v[ (totNy-1)*totNx + i ];
          su = (uij + Uinf > 0) ? 1:-1;
          sv = (vij + Vinf > 0) ? 1:-1; 
          if (sv != 1) // (vij + V < 0) d/dy=0
          {
              if ( (su == 1 && i == 0) || (su == -1 && i == totNx-1) ) su = 0; //take care of corners
              vals_x[totNx*(totNy-1) +i][0] =  1.0 + 4.0*dfac + afac*( su*(2.0*uij + Uinf - u[(totNy-1)*totNx + i-su]))+ precond1;
              vals_y[totNx*(totNy-1) +i][0] =  1.0 + 4.0*dfac + afac*( su*(uij+Uinf) ) + precond1; 
          }
          vals_x[totNx*(totNy-1) +i][0] -= dfac; // center
          vals_y[totNx*(totNy-1) +i][0] -= dfac; // center
          vals_x[totNx*(totNy-1) +i][4]  = 0; // north
          vals_y[totNx*(totNy-1) +i][4]  = 0; // north
      }
    }
  
    HYPRE_Int inds[5] = {0, 1, 2, 3, 4}; // These indices must match to those in the offset array
    HYPRE_Int ilower[2] = {0,0};
    HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};
    double * const linV_x = (double*) vals_x;
    double * const linV_y = (double*) vals_y;
    HYPRE_StructMatrixSetBoxValues(hypre_mat_X, ilower, iupper, 5, inds, linV_x);
    HYPRE_StructMatrixSetBoxValues(hypre_mat_Y, ilower, iupper, 5, inds, linV_y);
  
    HYPRE_StructHybridDestroy(hypre_solver_X);
    HYPRE_StructHybridDestroy(hypre_solver_Y);
    allocLHS();
}

void LinearSolverDelta::solve(const std::vector<BlockInfo>& infos, const bool solveX)
{
    HYPRE_StructMatrix & hypre_mat    = (solveX) ?  hypre_mat_X    : hypre_mat_Y;
    HYPRE_StructSolver & hypre_solver = (solveX) ?  hypre_solver_X : hypre_solver_Y;
    HYPRE_StructVector & hypre_rhs    = (solveX) ?  hypre_rhs_X    : hypre_rhs_Y;
    HYPRE_StructVector & hypre_sol    = (solveX) ?  hypre_sol_X    : hypre_sol_Y;
   
    HYPRE_Int ilower[2] = {0,0};
    HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};
 
    //set up RHS
    cub2rhs(infos,solveX);
    HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, dbuffer);
  
    //set up solution initial guess with Jacobi preconditioner
    auto & vals = (solveX == true) ? vals_x : vals_y; 
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < totNy*totNx; k++)
      dbuffer[k] /= vals[k][0];
  
    HYPRE_StructVectorSetBoxValues(hypre_sol, ilower, iupper, dbuffer);
  
    sim.startProfiler("HYPRE SOLVE");
    HYPRE_StructHybridSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
    sim.stopProfiler();
  
    HYPRE_StructVectorGetBoxValues(hypre_sol, ilower, iupper, dbuffer);
    #ifdef _FLOAT_PRECISION_
      std::copy(dbuffer, dbuffer + totNy*totNx, buffer);
    #endif
    sol2cub(infos,solveX);
}

#define STRIDE s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX

LinearSolverDelta::LinearSolverDelta(SimulationData& s) : sim(s), stride(STRIDE)
{
    buffer = new Real[totNy * totNx];
    u       = new double[totNy * totNx];
    v       = new double[totNy * totNx];
   
    #ifdef _FLOAT_PRECISION_
      dbuffer = new double[totNy * totNx];
    #else
      dbuffer = buffer;
    #endif
    HYPRE_Int ilower[2] = {0,0};
    HYPRE_Int iupper[2] = {(int)totNx-1, (int)totNy-1};
    const auto COMM = MPI_COMM_SELF;
  
    // Grid
    HYPRE_StructGridCreate(COMM, 2, &hypre_grid);
    HYPRE_StructGridSetExtents(hypre_grid, ilower, iupper);
    HYPRE_StructGridAssemble(hypre_grid);
  
    // Stencil
    HYPRE_Int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};
    HYPRE_StructStencilCreate(2, 5, &hypre_stencil);
    for (int j = 0; j < 5; ++j)
      HYPRE_StructStencilSetElement(hypre_stencil, j, offsets[j]);
  
  
    vals_x = new RowType[totNy*totNx]; 
    vals_y = new RowType[totNy*totNx]; 
  
    // LHS Matrix
    HYPRE_StructMatrixCreate(COMM, hypre_grid, hypre_stencil, &hypre_mat_X);
    HYPRE_StructMatrixInitialize(hypre_mat_X);
    HYPRE_StructMatrixAssemble(hypre_mat_X);
    HYPRE_StructMatrixCreate(COMM, hypre_grid, hypre_stencil, &hypre_mat_Y);
    HYPRE_StructMatrixInitialize(hypre_mat_Y);
    HYPRE_StructMatrixAssemble(hypre_mat_Y); 
  
    // RHS vector
    HYPRE_StructVectorCreate(COMM, hypre_grid, &hypre_rhs_X);
    HYPRE_StructVectorInitialize(hypre_rhs_X);
    HYPRE_StructVectorAssemble(hypre_rhs_X);
    HYPRE_StructVectorCreate(COMM, hypre_grid, &hypre_rhs_Y);
    HYPRE_StructVectorInitialize(hypre_rhs_Y);
    HYPRE_StructVectorAssemble(hypre_rhs_Y);
  
    // solution vector
    HYPRE_StructVectorCreate(COMM, hypre_grid, &hypre_sol_X);
    HYPRE_StructVectorInitialize(hypre_sol_X);
    HYPRE_StructVectorAssemble(hypre_sol_X);
    HYPRE_StructVectorCreate(COMM, hypre_grid, &hypre_sol_Y);
    HYPRE_StructVectorInitialize(hypre_sol_Y);
    HYPRE_StructVectorAssemble(hypre_sol_Y);
  
    allocLHS();
}

// let's relinquish STRIDE which was only added for clarity:
#undef STRIDE

LinearSolverDelta::~LinearSolverDelta()
{
  HYPRE_StructHybridDestroy(hypre_solver_X);
  HYPRE_StructHybridDestroy(hypre_solver_Y);

  HYPRE_StructGridDestroy(hypre_grid);
  HYPRE_StructStencilDestroy(hypre_stencil);
  HYPRE_StructMatrixDestroy(hypre_mat_X);
  HYPRE_StructMatrixDestroy(hypre_mat_Y);

  HYPRE_StructVectorDestroy(hypre_rhs_X);
  HYPRE_StructVectorDestroy(hypre_sol_X);

  HYPRE_StructVectorDestroy(hypre_rhs_Y);
  HYPRE_StructVectorDestroy(hypre_sol_Y);

  #ifdef _FLOAT_PRECISION_
    delete [] dbuffer;
  #endif
  delete [] buffer;
  delete [] vals_x;
  delete [] vals_y;
  delete [] u;
  delete [] v;}

void LinearSolverDelta::sol2cub(const std::vector<BlockInfo>& BDST, const bool solveX)
{
    const size_t nBlocks = BDST.size();
    const Real * __restrict__ const sorc = buffer;
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<nBlocks; ++i)
    {
      int comp = solveX ? 0 : 1;
      const BlockInfo& info = BDST[i];
      const size_t blocki = VectorBlock::sizeX * info.index[0];
      const size_t blockj = VectorBlock::sizeY * info.index[1];
      VectorBlock& b = *(VectorBlock*)info.ptrBlock;
      const size_t blockStart = blocki + stride*blockj;
  
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
        b(ix,iy).u[comp] = sorc[blockStart + ix + stride*iy];
    }
}