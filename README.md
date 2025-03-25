# CubismUP-2D

Incompressible Flow Solver for Complex Deformable Geometries in 2D.

## Dependencies

CubismUP-2D depends on MPI, GSL, and HDF5.

## Compilation

```
make 'CXX = mpicxx '"`pkg-config --cflags hdf5-openmpi gsl`" 'LIBS = -fopenmp '"`pkg-config --libs hdf5-openmpi gsl`"
```

or
```
make 'CXX = mpicxx -I/scratch/slitvinov/.grace/include -fopenmp' 'LIBS = -L/scratch/slitvinov/.grace/lib -lhdf5 -lgsl -lgslcblas -Wl,-R/scratch/slitvinov/.grace/lib'
```

Run an example with the following commands, starting from the `build` directory:
```
cd ..
export PYTHONPATH=$(pwd):$(pwd)/build/:$PYTHONPATH
cd cubismup2d/examples/
./rectangle_and_operator.py
```
Output files will be stored in the `output/` directory.

## Running

```
mpiexec -n 4 sh launch/stefanfish.sh
```
or

```
mpiexec -n 4 sh launch/windmills.sh
```
