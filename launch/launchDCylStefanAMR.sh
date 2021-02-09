# Defaults for Options
BPDX=${BPDX:-32}
BPDY=${BPDY:-16}
LEVELS=${LEVELS:-3}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.5}
# Defaults for Halfdisk
XPOS0=${XPOS0:-0.2}
ANGLE0=${ANGLE0:-20}
XVEL0=${XVEL0:-0.15}
RADIUS0=${RADIUS0:-0.0375}
# Defaults for Halfdisk
LENGTH1=${XPOS1:-0.2}
PERIOD1=${PERIOD1:-1}
XPOS1=${XPOS1:-0.5}
# Cylinder Re=1'000 <-> 0.00001125; Re=10'000 <-> 0.000001125
NU=${NU:-0.00001125}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 10 -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 1"
# bForced, tAccel is needed here!
OBJECTS="halfDisk radius=$RADIUS0 angle=$ANGLE0 xpos=$XPOS0 bForced=1 bFixed=1 xvel=$XVEL0 tAccel=5
stefanfish L=$LENGTH1 T=$PERIOD1 xpos=$XPOS1"

source launchCommon.sh
