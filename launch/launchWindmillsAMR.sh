# Defaults for Options
BPDX=${BPDX:-8}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-4}
RTOL=${RTOL:-2}
CTOL=${CTOL:-0.2}
EXTENT=${EXTENT:-1.2}
CFL=${CFL:-0.5}
# Defaults for Objects
XPOS=${XPOS:-0.2}

# YPOS1=${YPOS:-0.2}
# YPOS2=${YPOS2:-0.4}
# YPOS3=${YPOS3:-0.6}
# YPOS4=${YPOS4:-0.8}

YPOS1=${YPOS:-0.3}
YPOS2=${YPOS2:-0.5}
YPOS3=${YPOS3:-0.7}
YPOS4=${YPOS4:-0.9}

XVEL=${XVEL:-0.15}

MAAXIS=${MAAXIS:-0.0405}
MIAXIS=${MIAXIS:-0.0135}

#NU=${NU:-0.0001215}
NU=${NU:-0.0000243}
#NU=${NU:-0.00001215}
#NU=${NU:-0.00000305}


OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 20 -muteAll 0 -verbose 1"
## FOUR WINDMILLS
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS3 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS4 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
"

## TWO WINDMILLS
# OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
# windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
# "

#-angvel changes angular velocity of object

source launchCommon.sh
