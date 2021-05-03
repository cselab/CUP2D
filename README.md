# CubismUP-2D

Incompressible Flow Solver for Complex Deformable Geometries in 2D.

## Dependencies

CubismUP-2D has the following prerequisite libraries:

- MPI, with the $MPICXX enviroment variable defined.
- GSL, with the $GSL_ROOT environment variable defined.
- HDF5, with the $HDF5_ROOT environment variable defined.

On Piz Daint:
```
module load daint-gpu GSL cray-hdf5 cray-python
export GSL_ROOT=/apps/dom/UES/jenkins/7.0.UP02/gpu/easybuild/software/GSL/2.5-CrayGNU-20.11
export MPICXX=CC
```

On Euler:
```
env2lmod
module load gcc/6.3.0
module load openmpi/3.0.1
module load hdf5
module load python/3.7.4
module load gsl
export MPICXX=mpic++
```

On Panda/Falcon:
```
module load gnu mpich python fftw hdf5
export GSL_ROOT=/usr
```


## Installation

With the above dependencies installed and associated environment variables set the code can be compiled by
```
cd makefiles
make -j
```

## Running

In order to run a simulation go to the launch directory for some preset cases
