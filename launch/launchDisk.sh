# Defaults for Options
BPDX=${BPDX:-128}
BPDY=${BPDY:-64}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.5}
# Defaults for Objects
XPOS=${XPOS:-0.2}
ANGLE=${ANGLE:-0}
XVEL=${XVEL:-0.15}
RADIUS=${RADIUS:-0.0375}
# Re=100 <-> 0.0001125; Re=1'000 <-> 0.00001125; Re=2'500 <-> 0.0000045; Re=5'000 <-> 0.00000225; Re=7'500 <-> 0.0000015; Re=10'000 <-> 0.000001125
NU=${NU:-0.000001125}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -extent $EXTENT -CFL $CFL -tdump 0.001 -nu $NU -tend 50 -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 0"
# bForced, tAccel is needed here!
OBJECTS="disk radius=$RADIUS angle=$ANGLE xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0"

source launchCommon.sh
