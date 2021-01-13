# Defaults for Options
BPDX=${BPDX:-48}
BPDY=${BPDY:-24}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.1}
# Defaults for Objects
XPOS=${XPOS:-0.2}
ANGLE=${ANGLE:-30}
XVEL=${XVEL:-0.15}
RADIUS=${RADIUS:-0.0375}
# Re=1'000 <-> 0.00001125; Re=10'000 <-> 0.000001125
NU=${NU:-0.00001125}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -extent $EXTENT -CFL $CFL  -tdump 0.1 -nu $NU -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 0"
# bForced, tAccel is needed here!
OBJECTS="halfDisk radius=$RADIUS angle=$ANGLE xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=5"

source launchCommon.sh
