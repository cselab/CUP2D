#include <mpi.h>
#include "config.h"
#include <fenv.h>
#include <csignal>
#include <iostream>
#include "Simulation.h"
struct Config cfg;

void fpe_handler(int sig) {
  fprintf(stderr, "main: error: floating point exception\n");
  std::abort();
}

int main(int argc, char **argv) {
  int threadSafety, rank;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);
  std::signal(SIGFPE, fpe_handler);
  fesetenv(FE_NOMASK_ENV);
  fedisableexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_DIVBYZERO);
  feenableexcept(FE_INVALID);
  feenableexcept(FE_OVERFLOW);
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
