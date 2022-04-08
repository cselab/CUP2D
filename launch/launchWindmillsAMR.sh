# Defaults for Options
BPDX=${BPDX:-3}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-4}
RTOL=${RTOL:-0.1}
CTOL=${CTOL:-0.01}
EXTENT=${EXTENT:-1.4}
CFL=${CFL:-0.22}
# Defaults for Objects
XPOS=${XPOS:-0.2}

YPOS1=${YPOS:-0.6}
YPOS2=${YPOS2:-0.8}

XVEL=${XVEL:-0.15}
#XVEL=${XVEL:-0.3}

MAAXIS=${MAAXIS:-0.0405}
MIAXIS=${MIAXIS:-0.0135}

#NU=${NU:-0.0001215}
NU=${NU:-0.000243}

#OVEL=${OVEL:-12.56} # 4hz
#OVEL=${OVEL:-6.28} # 2hz
#OVEL=${OVEL:-3.14} # 1hz

## ovel is now actually the angular acceleration
# divided by 2 * pi * f gives the max angular velocity
# for 24, max ang vel is around 16

# OVEL=${OVEL:-12}

VEL=${VEL:-12.0}
FACTOR=${FACTOR:--1.0} # varies between -1 and 1 in steps of 0.2
TOP=$VEL
BOTTOM=$(echo $VEL*$FACTOR | bc)


echo "Vel is $VEL"
echo "Factor is $FACTOR"
echo "Top is $TOP"
echo "Bottom is $BOTTOM"


# we now decide to set the -poissonTol value. By default, it is 10^-6. We also set -poissonTolRel to 0



OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.0 -nu $NU -poissonTol 1.0e-2 -tend 200 -muteAll 0 -verbose 1 -poissonTolRel 0"
## two WINDMILLS, constant angular velocity of 4.0hz
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvel=$TOP
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvel=$BOTTOM
"

## TWO WINDMILLS
# OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
# windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
# "

#-angvel changes angular velocity of object forcedOmega

source launchCommon.sh
