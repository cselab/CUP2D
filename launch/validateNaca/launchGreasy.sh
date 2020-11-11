TASK="gridSizeValidation"
NUMNODES=4

cat << EOF > daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --job-name=${TASK}
#SBATCH --output=${TASK}_out_%j.txt
#SBATCH --error=${TASK}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=${NUMNODES}
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:0,craynetwork:4

module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel cray-fftw cray-petsc cudatoolkit GSL cray-python
module load GREASY

export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0
export OMP_NUM_THREADS=12

# greasy tasksAngle.txt
# greasy tasksFrequency.txt
greasy tasksGridSize.txt
# greasy tasksDomainSize.txt
EOF

sbatch daint_sbatch
