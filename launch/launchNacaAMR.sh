# Defaults for Options
BPDX=${BPDX:-16}
BPDY=${BPDY:-16}
LEVELS=${LEVELS:-5}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.2}
# Defaults for Objects
XPOS=${XPOS:-1.2}
ANGLE=${ANGLE:-0}
FPITCH=${FPITCH:-0}
LENGTH=${LENGTH:-0.2}
VELX=${VELX:-0.2}
# Re=1'000 <-> NU=0.00004; Re=10'000 <-> 0.000004
NU=${NU:-0.00004}

# Re=1'000 <-> 0.00001125; Re=10'000 <-> 0.000001125 # 
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 1 -nu $NU -tend 100 -muteAll 0 -verbose 0 -poissonTol 1e-13"
# COM IS 0.399421, COR 0.1, thus fixedCenterDist=0.299412
# xvel is needed here!
OBJECTS="NACA L=$LENGTH xpos=$XPOS angle=$ANGLE fixedCenterDist=0.299412 bFixed=1 xvel=$VELX Apitch=13.15 Fpitch=$FPITCH tAccel=0
"

source launchCommon.sh