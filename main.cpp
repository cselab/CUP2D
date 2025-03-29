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
  std::signal(SIGFPE, handler);
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

static const struct {
  int sig;
  const char *name, *desc;
} FEInfo[] = {
    {FE_DIVBYZERO, "FE_DIVBYZERO", "Division by zero"},
    {FE_INEXACT, "FE_INEXACT", "Inexact result (rounding occurred)"},
    {FE_INVALID, "FE_INVALID", "Invalid operation (e.g., sqrt(-1))"},
    {FE_OVERFLOW, "FE_OVERFLOW", "Overflow (result too large to represent)"},
    {FE_UNDERFLOW, "FE_UNDERFLOW", "Underflow (result too small to represent)"},
};
static void handler(int sig) {
  void *array[10];
  size_t size, i;
  char **strings;
  const char *name = "unknown";
  for (i = 0; i < sizeof FEInfo / sizeof *FEInfo; i++)
    if (FEInfo[i].sig == sig)
      name = FEInfo[i].name;
  size = backtrace(array, 10);
  fprintf(stderr, "%s:%d: error: '%s' floating point exception on rank %d\n",
          __FILE__, __LINE__, name, cfg.rank);
  size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  if (strings != NULL) {
    for (i = 0; i < size; i++)
      fprintf(stderr, "%s\n", strings[i]);
  }
  free(strings);
  exit(1);
}
