//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "cudaAMRSolver.cuh"

using namespace cubism;

void cudaAMRSolver::getZ(std::vector<BlockInfo> & zInfo)
{
  sim.startProfiler("Poisson solver: preconditioner");

  cudaDeviceSynchronize();
  std::cout << "Calling on cudaAMRSolver.getZ() \n";

  sim.stopProfiler();
}

void cudaAMRSolver::Get_LHS ()
{
  sim.startProfiler("Poisson solver: LHS");

  cudaDeviceSynchronize();
  std::cout << "Calling on cudaAMRSolver.Get_LHS() \n";

  sim.stopProfiler();
}

double cudaAMRSolver::getA_local(int I1,int I2) //matrix for Poisson's equation on a uniform grid
{
  std::cout << "Calling on cudaAMRSolver.getA_local() \n";
  return 0.;
}

cudaAMRSolver::cudaAMRSolver(SimulationData& s):sim(s)
{
  cudaDeviceSynchronize();
  std::cout << "--------------------- Calling on cudaAMRSolver() constructor -----------------\n";
}

void cudaAMRSolver::solve()
{
  cudaDeviceSynchronize();
  std::cout << "--------------------- Calling on cudaAMRSolver.solve() ------------------------ \n";

}
