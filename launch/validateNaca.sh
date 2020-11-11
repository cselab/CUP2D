#!/bin/bash -l

## WRITE GREASY TASKS TO VALIDATE ANGLES
# here resolution, 
for i in {01..14}
do
cat << EOF >> taskAngle.txt
[@ ./ @] export BPDX=48; export BPDY=24; export ANGLE=$i; export FPITCH=0; export EXTENT=1; export XPOS=0.2; ./launchNaca.sh NACA-theta=$i
EOF
done

# WRITE GREASY TASKS TO VALIDATE FREQUENCY
for i in {0.00 1.25 1.43 1.67 2.00 2.5}
do
cat << EOF >> taskFrequency.txt
[@ ./ @] export BPDX=48; export BPDY=24; export ANGLE=0; export FPITCH=$i; export EXTENT=1; export XPOS=0.2; ./launchNaca.sh NACA-freq=$i
EOF
done

# WRITE GREASY TASK FOR GRIDSIZE
for i in {16,24,32,48,64,96,128}
do
cat << EOF >> taskGridSize.txt
[@ ./ @] export BPDX=$i; export BPDY=$(($i / 2)); export ANGLE=13.15; export FPITCH=0; export EXTENT=1; export XPOS=0.2; ./launchNaca.sh NACA-bpdx=$i
EOF
done

# WRITE GREASY TASK FOR DOMAINSIZE
for i in {1,2,3}
do
cat << EOF >> taskDomainSize.txt
[@ ./ @] export BPDX=$(( $i * 48)); export BPDY=$(( $i * 24)); export ANGLE=13.15; export FPITCH=0; export EXTENT=$i; export XPOS=$(( $i / 5 )); ./launchNaca.sh NACA-bpdx=$i
EOF
done
