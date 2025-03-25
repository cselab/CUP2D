# CubismUP-2D

Incompressible Flow Solver for Complex Deformable Geometries in 2D.

## Dependencies

CubismUP-2D depends on MPI, GSL, and HDF5.

## Compilation

```
make 'CXX = mpicxx `pkg-config --cflags --libs hdf5-openmpi gsl`'
```

Run an example with the following commands, starting from the `build` folder:
```
cd ..
export PYTHONPATH=$(pwd):$(pwd)/build/:$PYTHONPATH
cd cubismup2d/examples/
./rectangle_and_operator.py
```
Output files will be stored in the `output/` folder.

## Running

To run a simulation go to the launch directory for some preset cases
