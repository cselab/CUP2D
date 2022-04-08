# Defaults for Options
BPDX=${BPDX:-8}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-4}
RTOL=${RTOL:-0.1}
CTOL=${CTOL:-0.01}
EXTENT=${EXTENT:-1.4}
CFL=${CFL:-0.22}
# Defaults for Objects
XPOS=${XPOS:-0.2}

XVEL=${XVEL:-0.15}

MAAXIS=${MAAXIS:-0.0405}
MIAXIS=${MIAXIS:-0.0135}

NU=${NU:-0.000243}

## ovel is now actually the angular acceleration
# divided by 2 * pi * f gives the max angular velocity
# for 24, max ang vel is around 16

# OVEL=${OVEL:-12}

YPOS1=${YPOS1:-0.4}
YPOS2=${YPOS2:-0.6}
YPOS3=${YPOS3:-0.8}
YPOS4=${YPOS4:-1.0}

V=${V:-12.0}
VT=${VT:--12.0}

# # config 1 
# VEL1=$V
# VEL2=$V
# VEL3=$VT
# VEL4=$VT

# # config 2
# VEL1=$V
# VEL2=$VT
# VEL3=$V
# VEL4=$VT

# config 3
VEL1=$V
VEL2=$VT
VEL3=$VT
VEL4=$V


echo "Vel1 is $VEL1"
echo "Vel2 is $VEL2"
echo "Vel3 is $VEL3"
echo "Vel4 is $VEL4"



OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 1"
## two WINDMILLS, constant angular velocity of 4.0hz
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvel=$VEL1
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvel=$VEL2
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS3 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvel=$VEL3
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS4 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvel=$VEL4
"

## TWO WINDMILLS
# OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
# windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
# "

#-angvel changes angular velocity of object forcedOmega

source launchCommon.sh
