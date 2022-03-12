#!/bin/bash
if [[ $# -lt 1 && -z "$RUNNAME" ]] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
RUNNAME=$1
fi

OPTIONS="-bpdx 2 -bpdy 2 -levelMax 7 -levelStart 4  -Rtol 5 -Ctol 0.01 -extent 1 -CFL 0.45 -poissonTol 1e-8 -poissonTolRel 0 -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1"
OBJECTS="stefanfish L=0.2 T=1 angle=-0.018605 xpos=0.302229 ypos=0.302928 \n
 stefanfish L=0.2 T=1 angle=-0.034845 xpos=0.294406 ypos=0.708361 \n
 stefanfish L=0.2 T=1 angle=0.010831 xpos=0.606730 ypos=0.242243 \n
 stefanfish L=0.2 T=1 angle=0.045902 xpos=0.601398 ypos=0.494894 \n
 stefanfish L=0.2 T=1 angle=0.007808 xpos=0.604442 ypos=0.759843"

BASEPATH="${SCRATCH}"
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/debugRL ${FOLDERNAME}
cd ${FOLDERNAME}

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --job-name="${RUNNAME}"
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
export OMP_NUM_THREADS=1
srun ./debugRL ${OPTIONS} -shapes "${OBJECTS}"
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
