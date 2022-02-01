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
ANGLE=${ANGLE:-0}
# Fpitch for St = 0.1 0.2 0.3 0.4 0.5 gives 0.2723183437 0.5446366874 0.8169550311 1.0892733748 1.3615917185
FPITCH=${FPITCH:-0.715}
LENGTH=${LENGTH:-0.12}
VELX=${VELX:-0.15}

# With swimmer of length 0.2 and period 1, this is Re=1'000
NU=${NU:-0.00004}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4  -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -poissonTol $PT -poissonTolRel $PTR -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0.1 -nu $NU -tend 50 -muteAll 0 -verbose 1"

# COM IS 0.399421, COR 0.1, thus fixedCenterDist=0.299412
OBJECTS="NACA L=$LENGTH xpos=$XPOS angle=$ANGLE fixedCenterDist=0.299412 bFixed=1 xvel=$VELX Apitch=13.15 Fpitch=$FPITCH tAccel=0
"

source launchCommon.sh