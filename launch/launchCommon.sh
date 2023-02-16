HOST=`hostname`
OS_D=`uname`
echo $RUNNAME
if [[ $# -lt 1 && -z "$RUNNAME" ]] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
RUNNAME=$1
fi

unset LSB_AFFINITY_HOSTFILE #euler cluster
export MV2_ENABLE_AFFINITY=0 #MVAPICH

###################################################################################################
if [ ${OS_D} == 'Darwin' ] ; then

BASEPATH="../runs/"
export OMP_NUM_THREADS=`sysctl -n hw.physicalcpu_max`
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log

###################################################################################################
elif [ ${HOST:0:5} == 'daint' ] ; then

BASEPATH="${SCRATCH}/CUP2D/"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
if [ ${PSOLVER:0:4} == 'cuda' ] ; then
  export TASKS_PER_NODE=1
  if [ "${TASKS_PER_NODE}" -gt "1" ] ; then
    export CRAY_CUDA_MPS=1
  fi
  export OMP_NUM_THREADS=$(expr 12 / $TASKS_PER_NODE)
else
  export TASKS_PER_NODE=12
  export OMP_NUM_THREADS=1
fi
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}

# did we allocate a node?
srun hostname &> /dev/null
if [[ "$?" -gt "0" ]] ; then
source launchSbatch.daint
else
cd ${FOLDERNAME}
srun --nodes $SLURM_NNODES --ntasks-per-node=$TASKS_PER_NODE --cpus-per-task=$OMP_NUM_THREADS --threads-per-core=1 simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log
fi

###################################################################################################
elif [ ${HOST:0:3} == 'eu-' ] ; then

BASEPATH="$SCRATCH/CUP2D"
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}
if [ "${RUNLOCAL}" == "true" ] ; then
unset LSB_AFFINITY_HOSTFILE
export MV2_ENABLE_AFFINITY=0
mpirun -n 128 --map-by core:PE=1 ./simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log
else
bsub -J ${RUNNAME} -W 120:00 -n 128 "unset LSB_AFFINITY_HOSTFILE; mpirun -n 128 --map-by core:PE=1 ./simulation ${OPTIONS} -shapes ${OBJECTS}"
fi

###################################################################################################
elif [ ${HOST:0:3} == 'uan' ] ; then

BASEPATH="$SCRATCH/CUP2D"
export OMP_NUM_THREADS=1
export TASKS_PER_NODE=128
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}

source launchSbatch.lumi

###################################################################################################
else

BASEPATH="../runs/"
NCPUSTR=`lscpu | grep "Core"`
export OMP_NUM_THREADS=${NCPUSTR: -3}
echo "Setting nThreads to "${OMP_NUM_THREADS}
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

mpirun -n 1 ./simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log

fi

