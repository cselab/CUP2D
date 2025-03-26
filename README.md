# CubismUP-2D

Incompressible Flow Solver for Complex Deformable Geometries in 2D.

## Dependencies

CubismUP-2D depends on MPI, GSL, and HDF5.

## Compilation

For CPU
```
make "CXXFLAGS = -Ofast `pkg-config --cflags hdf5-openmpi`" \
     "LIBS = `pkg-config --libs gsl hdf5-openmpi`"
```

or
```
make \
     "CXXFLAGS = -Ofast `pkg-config --cflags hdf5-openmpi`" \
     "LIBS = `pkg-config --libs gsl hdf5-openmpi`"
```

For GPU
```
make "gpu = true" \
     "CXXFLAGS = -Ofast `pkg-config --cflags hdf5-openmpi`" \
     "LIBS = `pkg-config --libs gsl hdf5-openmpi`"
```

or

```
module load gcc/12 openmpi hdf5 gsl cuda
make "gpu = true" "CXXFLAGS = -Ofast" "LIBS = -lgsl -lgslcblas" "MPICXX = h5c++" -j 4
```

## Running

```
mpiexec -n 4 sh launch/stefanfish.sh
```
or

```
mpiexec -n 4 sh launch/windmills.sh
```
