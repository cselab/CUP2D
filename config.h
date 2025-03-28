#define OMPI_SKIP_MPICXX 1
#ifdef _FLOAT_PRECISION_
using Real = float;
#define MPI_Real MPI_FLOAT
#endif
#ifdef _DOUBLE_PRECISION_
using Real = double;
#define MPI_Real MPI_DOUBLE
#endif
#ifdef _LONG_DOUBLE_PRECISION_
using Real = long double;
#define MPI_Real MPI_LONG_DOUBLE
#endif
