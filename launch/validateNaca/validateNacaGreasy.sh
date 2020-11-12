#!/bin/bash -l

rm tasks*

## WRITE GREASY TASKS TO VALIDATE ANGLES
# for i in {01..14}
# do
# OPTIONS="-bpdx 48 -bpdy 24 -tdump 0 -nu 0.00001125 -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 1 -extent 1"
# OBJECTS="NACA L=0.075 xpos=0.2 angle=$i fixedCenterDist=0.299412 bFixed=0 xvel=0.15 Apitch=13.15 Fpitch=0 tAccel=1
# "
# RUNDIR="$SCRATCH/CUP2D/naca-angle$i"
# mkdir $RUNDIR
# cp ../../makefiles/simulation $RUNDIR
# cat << EOF >> tasksAngle.txt
# [@ $RUNDIR/ @] ./simulation ${OPTIONS} -shapes "${OBJECTS}"
# EOF
# done

# WRITE GREASY TASKS TO VALIDATE FREQUENCY
# for i in {0.00 1.25 1.43 1.67 2.00 2.5}
# do
# OPTIONS="-bpdx 48 -bpdy 24 -tdump 0 -nu 0.00001125 -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 1 -extent 1"
# OBJECTS="NACA L=0.075 xpos=0.2 angle=0 fixedCenterDist=0.299412 bFixed=0 xvel=0.15 Apitch=13.15 Fpitch=$i tAccel=1
# "
# RUNDIR="$SCRATCH/CUP2D/naca-freq$i"
# mkdir $RUNDIR
# cp ../../makefiles/simulation $RUNDIR
# cat << EOF >> tasksFrequency.txt
# [@ $RUNDIR/ @] ./simulation ${OPTIONS} -shapes "${OBJECTS}"
# EOF
# done

# WRITE GREASY TASK FOR GRIDSIZE
for i in {24,32,48,64,96,128}
do
OPTIONS="-bpdx $i -bpdy $(($i / 2)) -tdump 0 -nu 0.00001125 -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 0 -extent 1"
OBJECTS="NACA L=0.075 xpos=0.2 angle=0 fixedCenterDist=0.299412 bFixed=0 xvel=0.15 Apitch=13.15 Fpitch=1 tAccel=1
"
RUNDIR="$SCRATCH/CUP2D/naca-bpdx$i"
mkdir $RUNDIR
cp ../../makefiles/simulation $RUNDIR
cat << EOF >> tasksGridSize.txt
[@ $RUNDIR/ @] ./simulation ${OPTIONS} -shapes "${OBJECTS}"
EOF
done

# WRITE GREASY TASK FOR DOMAINSIZE
# for i in {1,2,3}
# do
# OPTIONS="-bpdx $(( $i * 48)) -bpdy $(( $i * 24)) -tdump 0 -nu 0.00001125 -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 1 -extent $i"
# OBJECTS="NACA L=0.075 xpos=0.2 angle=0 fixedCenterDist=0.299412 bFixed=0 xvel=0.15 Apitch=13.15 Fpitch=1 tAccel=1
# "
# RUNDIR="$SCRATCH/CUP2D/NACA-extent=$i"
# mkdir $RUNDIR
# cp ../../makefiles/simulation $RUNDIR
# cat << EOF >> tasksDomainSize.txt
# [@ $RUNDIR/ @] ./simulation ${OPTIONS} -shapes "${OBJECTS}"
# EOF
# done
