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
export OMP_NUM_THREADS=12
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
# AMGXCONF=FGMRES_NOPREC.json
# cp /users/novatig/AMGX/gnu9.1/lib/configs/core/${AMGXCONF} ${FOLDERNAME}/AMGX_setup.json

# did we allocate a node?
srun hostname &> /dev/null
if [[ "$?" -gt "0" ]] ; then
source launchSbatch.sh
else
cd ${FOLDERNAME}
srun -n 1 simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log
fi

###################################################################################################
elif [ ${HOST:0:3} == 'eu-' ] ; then

BASEPATH="$SCRATCH/CUP2D"
export OMP_NUM_THREADS=36 # 128 for other Euler nodes
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}
if [ "${RUNLOCAL}" == "true" ] ; then
./simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log
else
bsub -n ${OMP_NUM_THREADS} -J ${RUNNAME} -W 24:00 -R "select[model==XeonGold_6150]" ./simulation ${OPTIONS} -shapes "${OBJECTS}" # select[model==EPYC_7H12/EPYC_7742] for other Euler nodes
fi

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

