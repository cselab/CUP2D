# CubismUP-2D

Incompressible Flow Solver for Complex Deformable Geometries in 2D.

## Dependencies

CubismUP-2D depends on MPI and [GSL - GNU Scientific
Library](https://www.gnu.org/software/gsl).

## Compilation

For CPU
```
make "CXXFLAGS = -Ofast" "LIBS = `pkg-config --libs gsl`"
```

or
```
module load gcc openmpi gsl
make "CXXFLAGS = -Ofast `gsl-config --cflags`" "LIBS = `gsl-config --libs`"
```

For GPU
```
make "gpu = true" "CXXFLAGS = -Ofast" "LIBS = `pkg-config --libs gsl`"
```

or

```
module load gcc/13 openmpi gsl cuda
make "gpu = true" "CXXFLAGS = -Ofast" "LIBS = `gsl-config --libs`"
```

## Running

```
mpiexec -n 4 sh launch/stefanfish.sh
```
or

```
mpiexec -n 4 sh launch/windmills.sh
```
