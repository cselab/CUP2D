# CubismUP-2D

Incompressible Flow Solver for Complex Deformable Geometries in 2D.

## Dependencies

CubismUP-2D depends on MPI, GSL, and HDF5.

## Compilation

For CPU
```
make "CXXFLAGS = -Ofast" "LIBS = `pkg-config --libs gsl`"
```

or
```
module load gcc openmpi gsl
make "CXXFLAGS = -Ofast" "LIBS = `gsl-config --libs`"
```

For GPU
```
make "gpu = true" "CXXFLAGS = -Ofast" "LIBS = `pkg-config --libs gsl`"
```

or

```
module load gcc/12 openmpi gsl cuda
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
