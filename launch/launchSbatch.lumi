if [ $# -lt 1 ] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
RUNNAME=$1

cd ${FOLDERNAME}
cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=$ACCOUNT
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=24:00:00
#SBATCH --partition=standard
# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$TASKS_PER_NODE
#SBATCH --cpus-per-task=$OMP_NUM_THREADS

srun ./simulation ${OPTIONS} -shapes "${OBJECTS}"
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
