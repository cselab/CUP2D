# CubismUP-2D

Incompressible Flow Solver for Complex Deformable Geometries in 2D.

## Dependencies

CubismUP-2D has the following prerequisite libraries:

- MPI, with the $MPICXX enviroment variable defined.
- HYPRE, with the $HYPRE_ROOT environment variable defined.
- GSL, with the $GSL_ROOT environment variable defined.
- HDF5, with the $HDF5_ROOT environment variable defined.
- FFTW, with the $FFTW_ROOT environment variable defined.

On Piz Daint:
```
module load daint-gpu; 
module swap PrgEnv-cray PrgEnv-gnu;
module load cray-hdf5-parallel cray-fftw cray-petsc cudatoolkit GSL cray-python
export HYPRE_ROOT=/users/novatig/hypre/build
export GSL_ROOT=/apps/daint/UES/jenkins/7.0.UP02/gpu/easybuild/software/GSL/2.5-CrayGNU-20.08
```

On Panda/Falcon:
```
module load gnu mpich python fftw hdf5
export HYPRE_ROOT=/home/novatig/hypre/build
export GSL_ROOT=/usr
```

On Barry:
```
module load gnu/8.2.0 mpich python fftw hdf5
```
Install GSL:
```
wget 'ftp://ftp.gnu.org/gnu/gsl/gsl-2.6.tar.gz'; check
tar -xzvf gsl-2.6.tar.gz ; check
cd gsl-2.6
mkdir -p $HOME/gsl
./configure --prefix=$HOME/gsl; check
make -j; check
make install; check
```
Install Cubism:
```
module load gnu/8.2.0 mpich python fftw hdf5
export LD_LIBRARY_PATH=$HOME/gsl/lib:${LD_LIBRARY_PATH}
export GSL_ROOT=$HOME/gsl
cd makefiles
make -j
```







## Installation

With the above dependencies installed and associated environment variables set the code can be compiled by
```
cd makefiles
make -j
```

## Running

In order to run a simulation go to the launch directory for some preset cases