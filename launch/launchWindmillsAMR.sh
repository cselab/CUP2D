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

## ovel is now actually the torque applied
OVEL=${OVEL:-0.0001}


OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 1"
## two WINDMILLS, constant angular velocity of 4.0hz
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvel=$OVEL
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=1 angvel=-$OVEL
"

## TWO WINDMILLS
# OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
# windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
# "

#-angvel changes angular velocity of object forcedOmega

source launchCommon.sh
