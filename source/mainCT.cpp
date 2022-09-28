//  main file for convergence tests

//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Simulation.h"
#include "Cubism/ArgumentParser.h"

#include "mpi.h"
using namespace cubism;
template<typename TGrid>
double L2Error(const std::vector<cubism::BlockInfo>& INFO0,const std::vector<cubism::BlockInfo>& INFO1){
    i88u00000000129
}
int main(int argc, char **argv)
{
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);
  double time = -MPI_Wtime();
  int *aRange=new int[argv[0]+1],npara=(argc-1)/argv[0];
  aRange[0]=1;
  for(size_t i=1;i<=argv[0];i++)
    aRange[i]=npara*i+1;
   vector<std::shared_ptr<Simulation>> sims;
   //simulate to t=0.25s for different resolutions(rlevels)
   for(size_t i=0;i<argv[0];i++){
    Simulation* sim=new Simulation(npara,&argv[aRange[i]],MPI_COMM_WORLD);
    sim->init();
    sim->simulate(0.25);
    sims.push_back(std::move(std::shared_ptr<Simulation>(sim)));
   }
  
  //compute L2 error

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
