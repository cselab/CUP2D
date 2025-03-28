#define OMPI_SKIP_MPICXX 1
#ifdef _FLOAT_PRECISION_
typedef float Real;
#define MPI_Real MPI_FLOAT
#endif
#ifdef _DOUBLE_PRECISION_
typedef double Real;
#define MPI_Real MPI_DOUBLE
#endif
#ifdef _LONG_DOUBLE_PRECISION_
typedef long double Real;
#define MPI_Real MPI_LONG_DOUBLE
#endif

struct Config {
  MPI_Comm comm;
  int rank, size;
};
extern struct Config cfg;
