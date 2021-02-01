# Defaults for Options
BPDX=${BPDX:-8}
BPDY=${BPDY:-4}
LEVELS=${LEVELS:-3}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.5}
# Defaults for Objects
XPOS=${XPOS:-0.2}
ANGLE=${ANGLE:-20}
XVEL=${XVEL:-0.15}
RADIUS=${RADIUS:-0.0375}
# Re=1'000 <-> 0.00001125; Re=10'000 <-> 0.000001125
NU=${NU:-0.00001125}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 10 -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 0"
# bForced, tAccel is needed here!
OBJECTS="halfDisk radius=$RADIUS angle=$ANGLE xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=5"

source launchCommon.sh
