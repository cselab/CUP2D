#!/bin/bash

if [[ $# -lt 1 && -z "$RUNNAME" ]] ; then
	echo "Usage: ./launchDebugRL.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
RUNNAME=$1
fi

# Defaults for Options
BPDX=${BPDX:-4}
BPDY=${BPDY:-2}
LEVELS=${LEVELS:-7}
RTOL=${RTOL-2}
CTOL=${CTOL-1}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.4}
PT=${PT:-1e-5}
PTR=${PTR:-1e-2}
PR=${PR:-5}

# Defaults for follower
LENGTH=${LENGTH:-0.2}
PERIOD=${PERIOD:-1} # 0.2 for NACA
XPOSFOLLOWER=${XPOSFOLLOWER:-0.9}
PID=${PID:-0}

# L=0.2 stefanfish Re=1'000 <-> NU=0.00004
NU=${NU:-0.00004}

# Settings for Obstacle
OBSTACLE=${OBSTACLE:-halfDisk}
XPOSLEADER=${XPOSLEADER:-0.6}

if [ "$OBSTACLE" = "multitask" ]
then
	echo "###############################"
	echo "no options for multitask"
	NAGENTS=1
	echo "###############################"
else
echo "###############################"
echo "setting simulation options"
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4  -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -poissonTol $PT -poissonTolRel $PTR -maxPoissonRestarts $PR -bAdaptChiGradient 0 -tdump 0.1 -nu $NU -tend 0 -muteAll 1 -verbose 0"
echo $OPTIONS
echo "----------------------------"
fi

if [ "$OBSTACLE" = "halfDisk" ]
then
	echo "setting options for halfDisk"
	# options for halfDisk
	NAGENTS=1
	ANGLE=${ANGLE:-20}
	XVEL=${XVEL:-0.15}
	RADIUS=${RADIUS:-0.06}
	# set object string
	OBJECTS="halfDisk radius=$RADIUS angle=$ANGLE xpos=$XPOSLEADER bForced=1 bFixed=1 xvel=$XVEL tAccel=5
stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSFOLLOWER
"
	echo $OBJECTS
	echo "###############################"
	# halfDisk Re=1'000 <-> NU=0.000018
	# NU=${NU:-0.000018}
elif [ "$OBSTACLE" = "NACA" ]
then
	echo "setting options for NACA"
	# options for NACA
	NAGENTS=1
	ANGLE=${ANGLE:-0}
	FPITCH=${FPITCH:-0.715} # corresponds to f=1.43 from experiment
	LEADERLENGTH=${LEADERLENGTH:-0.12}
	VELX=${VELX:-0.15}
	# set object string
	OBJECTS="NACA L=$LEADERLENGTH xpos=$XPOSLEADER angle=$ANGLE fixedCenterDist=0.299412 bFixed=1 xvel=$VELX Apitch=13.15 Fpitch=$FPITCH tAccel=5
stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSFOLLOWER
"
	echo $OBJECTS
	echo "###############################"
	# NACA Re=1'000 <-> NU=0.000018
	# NU=${NU:-0.000018}
elif [ "$OBSTACLE" = "stefanfish" ]
then
	echo "setting options for stefanfish"
	# options for stefanfish
	NAGENTS=1
	# set object string
	OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSLEADER bFixed=1 pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSFOLLOWER
"
	echo $OBJECTS
	echo "###############################"
elif [ "$OBSTACLE" = "swarm4" ]
then
	echo "setting options for swarm4"
	# options for swarm4
	NAGENTS=3
	# set object string
	### for L=0.2 and extentx=extenty=2, 4 swimmers
	OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00
"
	echo $OBJECTS
	echo "###############################"
elif [ "$OBSTACLE" = "swarm9" ]
then
	echo "setting options for swarm9"
	# options for swarm9
	NAGENTS=8
	# set object string
	### for L=0.2 and extentx=extenty=2, 9 swimmers
	OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=0.80
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.20
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.00
"
	echo $OBJECTS
	echo "###############################"
elif [ "$OBSTACLE" = "swarm16" ]
then
	echo "setting options for swarm16"
	# options for swarm16
	NAGENTS=15
	# set object string
	### for L=0.2 and extentx=extenty=2, 16 swimmers
	OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=0.80
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.20
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.70
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.30
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=0.80
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.00
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.20
stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=2.40 ypos=1.00
"
	echo $OBJECTS
	echo "###############################"
elif [ "$OBSTACLE" = "swarm25" ]
then
	echo "setting options for swarm25"
	# options for swarm25
	NAGENTS=24
	# set object string
	### for L=0.2 and extentx=extenty=2, 25 swimmers
	OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=0.80
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.20
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.70
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.30
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=0.60
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=0.80
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.00
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.20
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.40
stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=0.70
stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=1.30
stefanfish L=$LENGTH T=$PERIOD xpos=2.40 ypos=0.80
stefanfish L=$LENGTH T=$PERIOD xpos=2.40 ypos=1.00
stefanfish L=$LENGTH T=$PERIOD xpos=2.40 ypos=1.20
stefanfish L=$LENGTH T=$PERIOD xpos=2.70 ypos=0.90
stefanfish L=$LENGTH T=$PERIOD xpos=2.70 ypos=1.10
stefanfish L=$LENGTH T=$PERIOD xpos=3.00 ypos=1.00
"
	echo $OBJECTS
	echo "###############################"
fi

# Create runfolder and copy executable
BASEPATH="${SCRATCH}/CUP2D/"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=12
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/debugRL ${FOLDERNAME}

# Run in runfolder
cd ${FOLDERNAME}
srun -n 1 debugRL ${OPTIONS} -shapes "${OBJECTS}" | tee out.log