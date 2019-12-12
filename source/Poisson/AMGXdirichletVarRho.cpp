//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "AMGXdirichletVarRho.h"
#include <algorithm>
#include "mpi.h"
//#include "cuda_runtime.h"

using namespace cubism;

void print_callback(const char *msg, int length)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) { printf("%s", msg); }
}

void AMGXdirichletVarRho::solve(const std::vector<BlockInfo>& BSRC,
                                 const std::vector<BlockInfo>& BDST)
{
  #ifdef AMGX_POISSON

  //static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  const size_t nBlocks = BDST.size();

  sim.startProfiler("AMGX_cub2rhs");
    // pre-hypre solve plan:
    // 1) place initial guess of pressure into vector x
    // 2) in the same compute discrepancy from sum RHS = 0
    // 3) send initial guess for x to hypre
    // 4) correct RHS such that sum RHS = 0 due to boundary conditions
    // 5) give RHS to hypre
    // 6) if user modified matrix, make sure it respects neumann BC
    // 7) if user modified matrix, reassemble it so that hypre updates precond

  if(bUpdateMat) { // 7)
    AMGX_matrix_upload_all(mat, totNy*totNx, nNonZeroInMatrix,
                           1, 1, row_ptrs, col_indices, matAry, NULL);
    // set the connectivity information (for the vector)
    AMGX_vector_bind(sol, mat);
    AMGX_vector_bind(rhs, mat);
    AMGX_solver_setup(solver, mat);
  }

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<nBlocks; ++i) {
    const size_t blocki = VectorBlock::sizeX * BSRC[i].index[0];
    const size_t blockj = VectorBlock::sizeY * BSRC[i].index[1];
    const ScalarBlock& P = *(ScalarBlock*) BDST[i].ptrBlock;
    const size_t blockStart = blocki + stride * blockj;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      dbuffer[blockStart + ix + stride*iy] = P(ix,iy).s; // 1)
  }
  AMGX_vector_upload(sol, totNy*totNx, 1, dbuffer); // 3)

  cub2rhs(BSRC);

  AMGX_vector_upload(rhs, totNy*totNx, 1, dbuffer); // 5)

  sim.stopProfiler();

  #if 0
    char fname[512]; sprintf(fname, "RHS_%06d", sim.step);
    sim.dumpTmp2( std::string(fname) );
  #endif

  sim.startProfiler("AMGX_solve");
  AMGX_solver_solve(solver, rhs, sol);
  //AMGX_SAFE_CALL( AMGX_solver_get_status(solver, &status) );
  sim.stopProfiler();

  sim.startProfiler("AMGX_getBoxV");
  AMGX_vector_download(sol, dbuffer);
  sim.stopProfiler();

  sim.startProfiler("AMGX_sol2cub");

  if(1) { // remove mean pressure
    double avgP = 0;
    const double fac = 1.0 / (totNx * totNy);
    #pragma omp parallel for schedule(static) reduction(+ : avgP)
    for (size_t i = 0; i < totNy*totNx; ++i) avgP += fac * dbuffer[i];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < totNy*totNx; ++i) dbuffer[i] -= avgP;
    //printf("Average pressure:%e\n", avgP);
  }

  #ifdef _FLOAT_PRECISION_
    std::copy(dbuffer, dbuffer + totNy*totNx, buffer);
  #endif
  sol2cub(BDST); // this will read buffer, which, if single prec, is not dbuffer

  sim.stopProfiler();

  #endif
}

#define STRIDE s.vel->getBlocksPerDimension(0) * VectorBlock::sizeX

AMGXdirichletVarRho::AMGXdirichletVarRho(SimulationData& s) :
  PoissonSolver(s, STRIDE) //
{
  buffer = new Real[totNy * totNx];
  #ifdef _FLOAT_PRECISION_
    dbuffer = new double[totNy * totNx];
  #else
    dbuffer = buffer;
  #endif

  #ifdef AMGX_POISSON
    #ifdef AMGX_DYNAMIC_LOADING
      void *lib_handle = lib_handle = amgx_libopen("libamgxsh.so");
      assert(lib_handle not_eq NULL && "ERROR: can not load the library");
    #endif

    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    //AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
    // TODO WHAT IS MODE?
    mode = AMGX_mode_dDDI; // AMGX_mode_dDFI / AMGX_mode_dFFI
    // TODO WHAT IS CONFIG?
    //AMGX_SAFE_CALL(AMGX_config_create(&config, "") );
    AMGX_SAFE_CALL(  AMGX_config_create_from_file(&config, "AMGX_setup.json") );

    if (sizeof(amgx_val_t) == sizeof(double)) {
      assert(AMGX_GET_MODE_VAL(AMGX_MatPrecision,mode) == AMGX_matDouble);
      assert(AMGX_GET_MODE_VAL(AMGX_MatPrecision,mode) == AMGX_matDouble);
    } else {
      assert(AMGX_GET_MODE_VAL(AMGX_MatPrecision,mode) != AMGX_matDouble);
      assert(AMGX_GET_MODE_VAL(AMGX_MatPrecision,mode) != AMGX_matDouble);
    }

    // create resources, matrix, vector and solver
    int device = 0;
    AMGX_SAFE_CALL(AMGX_resources_create(&workspace, config, NULL, 1, &device));
    AMGX_SAFE_CALL(AMGX_matrix_create(&mat, workspace, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&sol, workspace, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&rhs, workspace, mode));
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, workspace, mode, config));

    //generate 3D Poisson matrix, [and rhs & solution]
    // generate the matrix: this routine will create 2D (5 point) discretization of the
    // Poisson operator. The discretization is performed on a the 2D domain consisting
    // of nx and ny points in x- and y-dimension respectively. Each rank processes it's
    // own part of discretization points. Finally, the rhs and solution will be set to
    // a vector of ones and zeros, respectively.
    const size_t nDof = totNy * totNx;
    row_ptrs    = (int *) malloc(sizeof(int) * (nDof + 1));
    col_indices = (int *) malloc(sizeof(int) *  nDof * 5 );
    matAry = (amgx_val_t *) malloc(nDof * 5 * sizeof(amgx_val_t));
    nNonZeroInMatrix = 0;
    for (size_t i = 0; i < nDof; ++i) {
      row_ptrs[i] = nNonZeroInMatrix;
      if ( i > totNy ) {
        col_indices[nNonZeroInMatrix] = i - totNy;
        matAry[nNonZeroInMatrix] =  1;
        nNonZeroInMatrix++;
      }
      if ( i % totNy not_eq 0 ) {
        col_indices[nNonZeroInMatrix] = i - 1;
        matAry[nNonZeroInMatrix] =  1;
        nNonZeroInMatrix++;
      }
      {
        col_indices[nNonZeroInMatrix] = i;
        matAry[nNonZeroInMatrix] = -4;
        nNonZeroInMatrix++;
      }
      if ( (i + 1) % totNy == 0 ) {
        col_indices[nNonZeroInMatrix] = i + 1;
        matAry[nNonZeroInMatrix] =  1;
        nNonZeroInMatrix++;
      }
      if (  i / totNy not_eq (totNx - 1) ) {
        col_indices[nNonZeroInMatrix] = i + totNy;
        matAry[nNonZeroInMatrix] =  1;
        nNonZeroInMatrix++;
      }
      assert(nNonZeroInMatrix <= nDof * 5);
    }
    row_ptrs[nDof] = nNonZeroInMatrix;

    AMGX_SAFE_CALL(AMGX_matrix_upload_all(mat, nDof, nNonZeroInMatrix, 1, 1,
                                          row_ptrs, col_indices, matAry, NULL));

    // set the connectivity information (for the vector)
    AMGX_SAFE_CALL( AMGX_vector_bind(sol, mat) );
    AMGX_SAFE_CALL( AMGX_vector_bind(rhs, mat) );

    // generate the rhs and solution
    std::fill(dbuffer, dbuffer + totNy * totNx, 0);
    AMGX_SAFE_CALL( AMGX_vector_upload(rhs, nDof, 1, dbuffer) );
    std::fill(dbuffer, dbuffer + totNy * totNx, 1);
    AMGX_SAFE_CALL( AMGX_vector_upload(sol, nDof, 1, dbuffer) );
  #endif
}
// let's relinquish STRIDE which was only added for clarity:
#undef STRIDE

AMGXdirichletVarRho::~AMGXdirichletVarRho()
{
  #ifdef AMGX_POISSON
    AMGX_SAFE_CALL( AMGX_solver_destroy(solver) );
    AMGX_SAFE_CALL( AMGX_matrix_destroy(mat) );
    AMGX_SAFE_CALL( AMGX_vector_destroy(sol) );
    AMGX_SAFE_CALL( AMGX_vector_destroy(rhs) );
    AMGX_SAFE_CALL( AMGX_resources_destroy(workspace) );
    // destroy config (need to use AMGX_SAFE_CALL after this point)
    AMGX_SAFE_CALL( AMGX_config_destroy(config) );
    // shutdown and exit
    AMGX_SAFE_CALL( AMGX_finalize_plugins() );
    AMGX_SAFE_CALL( AMGX_finalize() );
    // close the library (if it was dynamically loaded)
    #ifdef AMGX_DYNAMIC_LOADING
      amgx_libclose(lib_handle);
    #endif
  #endif
}
