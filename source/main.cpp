//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Simulation.h"

#include "mpi.h"
using namespace cubism;

int main(int argc, char **argv)
{
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);

  double time = -MPI_Wtime();

  Simulation* sim = new Simulation(argc, argv, MPI_COMM_WORLD);
  sim->init();
  sim->simulate();
  time += MPI_Wtime();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0)
    std::cout << "Runtime = " << time << std::endl;
  delete sim;
  MPI_Finalize();
  return 0;
}
