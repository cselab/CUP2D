if [ $# -lt 1 ] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
RUNNAME=$1

BASEPATH="${SCRATCH}/CUP2D/"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=12

FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ./kolmogorov_flow.py ${FOLDERNAME}
cd ${FOLDERNAME}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s929
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=24:00:00
# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu

srun python kolmogorov_flow.py --N $N -Cs 0.0
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
