#include <mpi.h>
#include "config.h"
#include "Simulation.h"
struct Config cfg;
int main(int argc, char **argv) {
  int threadSafety, rank;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);
  cfg.comm = MPI_COMM_WORLD;
  MPI_Comm_rank(cfg.comm, &cfg.rank);
  MPI_Comm_rank(cfg.comm, &cfg.size);
  Simulation *sim = new Simulation(argc, argv, MPI_COMM_WORLD);
  sim->init();
  sim->simulate();
  delete sim;
  MPI_Finalize();
  return 0;
}
