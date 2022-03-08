# CubismUP-2D

Incompressible Flow Solver for Complex Deformable Geometries in 2D.

## Dependencies

CubismUP-2D has the following prerequisite libraries:

- MPI, with the $MPICXX enviroment variable defined.
- GSL, with the $GSL_ROOT environment variable defined.
- HDF5, with the $HDF5_ROOT environment variable defined.

On Piz Daint:
```
module swap PrgEnv-cray PrgEnv-gnu
module load daint-gpu 
moudle load cray-python
module load cray-hdf5 
module load GSL 
module load cdt-cuda
module load craype-accel-nvidia60
export MPICXX=CC
export CXX=CC
export NVCC=nvcc
```

On Euler:
```
env2lmod
module load gcc
module load openmpi
module load hdf5
module load python
module load gsl
export MPICXX=mpic++
```

On Panda/Falcon:
```
module load gnu mpich python hdf5
export GSL_ROOT=/usr
```


## Compilation

With the above dependencies installed and associated environment variables set the code can be compiled by
```
cd makefiles
make -j
```

## Compilation (cmake)

Compile the code using:
```
mkdir -p build
cd build
cmake ..
make
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

In order to run a simulation go to the launch directory for some preset cases
