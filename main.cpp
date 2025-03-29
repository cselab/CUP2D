#include <mpi.h>
#include "config.h"
#include <fenv.h>
#include <csignal>
#include <iostream>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include "Simulation.h"
struct Config cfg;
static void handler(int);

int main(int argc, char **argv) {
  int threadSafety, rank;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);
  cfg.comm = MPI_COMM_WORLD;
  MPI_Comm_rank(cfg.comm, &cfg.rank);
  MPI_Comm_rank(cfg.comm, &cfg.size);
  std::signal(SIGFPE, handler);
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  Simulation *sim = new Simulation(argc, argv, MPI_COMM_WORLD);
  sim->init();
  sim->simulate();
  delete sim;
  MPI_Finalize();
  return 0;
}

static void handler(int sig) {
  void *array[10];
  size_t size, i;
  char **strings;
  size = backtrace(array, 10);
  fprintf(stderr, "%s:%d: error: floating point exception on rank %d\n",
          __FILE__, __LINE__, cfg.rank);
  size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  if (strings != NULL) {
    for (i = 0; i < size; i++)
      fprintf(stderr, "%s\n", strings[i]);
  }
  free(strings);
  exit(1);
}
