# Defaults for Options
BPDX=${BPDX:-8}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-4}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.5}
# Defaults for Objects
XPOS=${XPOS:-0.2}
YPOS1=${YPOS:-0.2}
YPOS2=${YPOS2:-0.4}
YPOS3=${YPOS3:-0.6}
YPOS4=${YPOS4:-0.8}

XVEL=${XVEL:-0.15}

MAAXIS=${MAAXIS:-0.0375}
MIAXIS=${MIAXIS:-0.0125}

NU=${NU:-0.0001125}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 50 -muteAll 0 -verbose 1"
# bForced, tAccel is needed here!
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS3 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS4 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0
"

#-angvel changes angular velocity of object

source launchCommon.sh
