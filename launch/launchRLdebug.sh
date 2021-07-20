#!/bin/bash

if [[ $# -lt 1 && -z "$RUNNAME" ]] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
RUNNAME=$1
fi

# Defaults for Options
BPDX=${BPDX:-2}
BPDY=${BPDY:-1}
LEVELS=${LEVELS:-8}
RTOL=${RTOL-2}
CTOL=${CTOL-1}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.5}
PT=${PT:-1e-5}
PTR=${PTR:-1e-2}
PR=${PR:-0}

# Defaults for follower
LENGTH=${LENGTH:-0.2}
PERIOD=${PERIOD:-1} # 0.2 for NACA
XPOSFOLLOWER=${XPOSFOLLOWER:-0.9}

# Settings for Obstacle
OBSTACLE=${OBSTACLE:-halfDisk}
XPOSLEADER=${XPOSLEADER:-0.6}

if [ "$OBSTACLE" = "halfDisk" ]
then
	echo "###############################"
	echo "setting options for halfDisk"
	# options for halfDisk
	ANGLE=${ANGLE:-20}
	XVEL=${XVEL:-0.15}
	RADIUS=${RADIUS:-0.06}
	# set object string
	OBJECTS="halfDisk radius=$RADIUS angle=$ANGLE xpos=$XPOSLEADER bForced=1 bFixed=1 xvel=$XVEL tAccel=5
stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSFOLLOWER"
	echo $OBJECTS
	# halfDisk Re=1'000 <-> NU=0.000018
	NU=${NU:-0.000018}
elif [ "$OBSTACLE" = "NACA" ]
then
	echo "###############################"
	echo "setting options for NACA"
	# options for NACA
	ANGLE=${ANGLE:-0}
	FPITCH=${FPITCH:-0.715} # corresponds to f=1.43 from experiment
	LEADERLENGTH=${LEADERLENGTH:-0.12}
	VELX=${VELX:-0.15}
	# set object string
	OBJECTS="NACA L=$LEADERLENGTH xpos=$XPOSLEADER angle=$ANGLE fixedCenterDist=0.299412 bFixed=1 xvel=$VELX Apitch=13.15 Fpitch=$FPITCH tAccel=5
stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSFOLLOWER"
	echo $OBJECTS
	# NACA Re=1'000 <-> NU=0.000018
	NU=${NU:-0.000018}
elif [ "$OBSTACLE" = "stefanfish" ]
then
	echo "###############################"
	echo "setting options for stefanfish"
	# set object string
	OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSLEADER bFixed=1 
stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSFOLLOWER"
	echo $OBJECTS
	# stefanfish Re=1'000 <-> NU=0.00001125
	NU=${NU:-0.00004}
fi

echo "----------------------------"
echo "setting simulation options"
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 5  -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -poissonTol $PT -poissonTolRel $PTR -maxPoissonRestarts $PR -bAdaptChiGradient 0 -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 1"
echo $OPTIONS
echo "###############################"

BASEPATH="${SCRATCH}/CUP2D/"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=36
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/debugRL ${FOLDERNAME}
cd ${FOLDERNAME}

srun -n 1 debugRL ${OPTIONS} -shapes "${OBJECTS}" | tee out.log