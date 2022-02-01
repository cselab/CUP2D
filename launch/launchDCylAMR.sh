#!/bin/bash

# Defaults for Options
BPDX=${BPDX:-4}
BPDY=${BPDY:-2}
LEVELS=${LEVELS:-7}
RTOL=${RTOL-2}
CTOL=${CTOL-1}
EXTENT=${EXTENT:-2}
CFL=${CFL:-0.2}
PT=${PT:-1e-5}
PTR=${PTR:-0}

# Defaults for Objects
XPOS=${XPOS:-0.6}
ANGLE=${ANGLE:-10}
XVEL=${XVEL:-0.15}
RADIUS=${RADIUS:-0.06}

# With swimmer of length 0.2 and period 1, this is Re=1'000
NU=${NU:-0.00004}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4  -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -poissonTol $PT -poissonTolRel $PTR -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0.1 -nu $NU -tend 20 -muteAll 0 -verbose 1"

OBJECTS="halfDisk radius=$RADIUS angle=$ANGLE xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=5"
source launchCommon.sh
