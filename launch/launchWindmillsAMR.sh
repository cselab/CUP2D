# # Defaults for Options
# BPDX=${BPDX:-3}
# BPDY=${BPDY:-8}
# LEVELS=${LEVELS:-4}
# RTOL=${RTOL:-0.1}
# CTOL=${CTOL:-0.01}
# EXTENT=${EXTENT:-1.4}
# CFL=${CFL:-0.4}
# # Defaults for Objects
# XPOS=${XPOS:-0.2}
# YPOS1=${YPOS:-0.6}
# YPOS2=${YPOS2:-0.8}
# XVEL=${XVEL:-0.15}

# Defaults for Options
BPDX=${BPDX:-3}
BPDY=${BPDY:-4}
LEVELS=${LEVELS:-4}
RTOL=${RTOL:-0.1}
CTOL=${CTOL:-0.01}
EXTENT=${EXTENT:-0.7}
CFL=${CFL:-0.4}
# Defaults for Objects
XPOS=${XPOS:-0.1}
YPOS1=${YPOS:-0.25}
YPOS2=${YPOS2:-0.45}
XVEL=${XVEL:-0.15}

MAAXIS=${MAAXIS:-0.0405}
MIAXIS=${MIAXIS:-0.0135}

#NU=${NU:-0.0001215}
NU=${NU:-0.000243}

PSOLVER="cuda_iterative"

TOP=${TOP:-3}
BOT=${BOT:--2.8}

FREQ1=${FREQ1:-0.25}
FREQ2=${FREQ2:-1.6}


# echo "Vel is $VEL"
# echo "Factor is $FACTOR"
echo "Top is $TOP"
echo "Bottom is $BOT"
echo "Freq1 is $FREQ1"
echo "Freq2 is $FREQ2"


# we now decide to set the -poissonTol value. By default, it is 10^-6. We also set -poissonTolRel to 0



OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.0 -nu $NU -poissonTol 1.0e-3 -tend 60 -muteAll 0 -verbose 1 -poissonTolRel 0 -poissonSolver cuda_iterative"
## two WINDMILLS, constant angular velocity of 4.0hz
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvelmax=$TOP freq=$FREQ1
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvelmax=$BOT freq=$FREQ2
"

## TWO WINDMILLS
# OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
# windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
# "


echo "$OBJECTS"


#-angvel changes angular velocity of object forcedOmega

source launchCommon.sh
