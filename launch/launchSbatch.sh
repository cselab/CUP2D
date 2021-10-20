if [ $# -lt 1 ] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
RUNNAME=$1

cd ${FOLDERNAME}
cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s929
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=24:00:00
# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
export OMP_NUM_THREADS=1

srun ./simulation ${OPTIONS} -shapes "${OBJECTS}"
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
