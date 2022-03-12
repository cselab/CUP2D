#!/bin/bash

# Defaults for Options
BPDX=${BPDX:-4}
BPDY=${BPDY:-2}
LEVELS=${LEVELS:-7}
RTOL=${RTOL-2}
CTOL=${CTOL-1}
EXTENT=${EXTENT:-2}
CFL=${CFL:-0.4}
PT=${PT:-1e-5}
PTR=${PTR:-0}

# Defaults for follower
LENGTH=${LENGTH:-0.2}
PERIOD=${PERIOD:-1} # 0.2 for NACA
XPOSFOLLOWER=${XPOSFOLLOWER:-0.9}
PID=${PID:-1}

# L=0.2 stefanfish Re=1'000 <-> NU=0.00004
NU=${NU:-0.00004}

# Settings for Obstacle
OBSTACLE=${OBSTACLE:-waterturbine}
XPOSLEADER=${XPOSLEADER:-0.6}

if [ "$OBSTACLE" = "multitask" ]
then
    echo "----------------------------"
	echo "no options for multitask"
	NAGENTS=1
	echo "----------------------------"
else
	if [ "$OBSTACLE" = "halfDisk" ]
	then
		echo "----------------------------"
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
		# halfDisk Re=1'000 <-> NU=0.000018
		# NU=${NU:-0.000018}
	elif [ "$OBSTACLE" = "NACA" ]
	then
		echo "----------------------------"
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
		# NACA Re=1'000 <-> NU=0.000018
		# NU=${NU:-0.000018}
	elif [ "$OBSTACLE" = "stefanfish" ]
	then
		echo "----------------------------"
		echo "setting options for stefanfish"
		# options for stefanfish
		NAGENTS=1
		# set object string
		OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSLEADER bFixed=1 pid=$PID
	stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSFOLLOWER
	"
		echo $OBJECTS
	elif [ "$OBSTACLE" = "waterturbine" ]
	then
		echo "----------------------------"
		echo "setting options for waterturbine"
		# options for Waterturbine
		NAGENTS=1 #1 fish
		XVEL=${XVEL:-0.2} #streamwise velocity, 0.25m/s in experiment <-> 0.2 according to settings, scale by 1.25
		ANGVEL=${ANGVEL:--0.79} #angular velocity, in experiment 7.91 rad/s, omega=1.9*xvel/turbineRadius, turbine radius scaled to fish #0.79
		MAAXIS=${MAAXIS:-0.05} #semi-major axis, 0.015m in experiment, fish length in exp 0.06m i.e. scale 1/4; here fish 0.2m therefore majax 0.05
		MIAXIS=${MIAXIS:-0.017} # semi-minor axis, 0.01m ish in experiment, i.e. scale by 1/12 to fish length
		# set object string
		OBJECTS="waterturbine semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOSLEADER bForced=1 bFixed=1 xvel=$XVEL angvel=$ANGVEL tAccel=0
		stefanfish L=$LENGTH T=$PERIOD xpos=$XPOSFOLLOWER
	"
		echo $OBJECTS
	elif [ "$OBSTACLE" = "swarm2" ]
	then
		echo "setting options for swarm2"
		# options for swarm4
		NAGENTS=1
		EXTENT=4
		# set object string
		### for L=0.2 and extentx=extenty=2, 4 swimmers
		OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pid=$PID
	stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.00
	"
		echo $OBJECTS
	elif [ "$OBSTACLE" = "swarm4" ]
	then
		echo "setting options for swarm4"
		# options for swarm4
		NAGENTS=3
		EXTENT=4
		# set object string
		### for L=0.2 and extentx=extenty=2, 4 swimmers
		OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pid=$PID
	stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90
	stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10
	stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00
	"
		echo $OBJECTS
	elif [ "$OBSTACLE" = "swarm9" ]
	then
		echo "----------------------------"
		echo "setting options for swarm9"
		# options for swarm9
		NAGENTS=8
		EXTENT=4
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
	elif [ "$OBSTACLE" = "swarm16" ]
	then
		echo "----------------------------"
		echo "setting options for swarm16"
		# options for swarm16
		NAGENTS=15
		EXTENT=4
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
	elif [ "$OBSTACLE" = "swarm25" ]
	then
		echo "----------------------------"
		echo "setting options for swarm25"
		# options for swarm25
		NAGENTS=24
		EXTENT=4
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
	fi
echo "----------------------------"
echo "setting simulation options"
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4  -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -poissonTol $PT -poissonTolRel $PTR -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0.1 -nu $NU -tend 0 -muteAll 1 -verbose 0"
echo $OPTIONS
echo "----------------------------"
fi

source launchCommon.sh