//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Simulation.h"
#include "Cubism/ArgumentParser.h"

#include "mpi.h"
using namespace cubism;

int main(int argc, char **argv)
{
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);

  double time = -MPI_Wtime();

  Simulation* sim = new Simulation(argc, argv);
  sim->init();
  sim->simulate();

  time += MPI_Wtime();
  std::cout << "Runtime = " << time << std::endl;
  MPI_Finalize();
  return 0;
}
