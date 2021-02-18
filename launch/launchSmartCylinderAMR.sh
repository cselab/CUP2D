 # Defaults for Options
BPDX=${BPDX:-32}
BPDY=${BPDY:-16}
LEVELS=${LEVELS:-3}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.5}
# Defaults for Objects
XPOS=${XPOS:-0.2}
RADIUS=${RADIUS:-0.06}
NU=${NU:-0.000018}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 10 -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 1"
# bForced, tAccel is needed here!
OBJECTS="smartDisk radius=$RADIUS xpos=$XPOS"

source launchCommon.sh
