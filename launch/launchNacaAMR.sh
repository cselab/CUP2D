# Defaults for Options
BPDX=${BPDX:-8}
BPDY=${BPDY:-4}
LEVELS=${LEVELS:-5}
RTOL=${RTOL-1}
CTOL=${CTOL-0.1}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.8}
# Defaults for Objects
XPOS=${XPOS:-0.2}
ANGLE=${ANGLE:-0}
FPITCH=${FPITCH:-1}
LENGTH=${LENGTH:-0.075}
VELX=${VELX:-0.15}
# Re=100 <-> 0.0001125; Re=1'000 <-> 0.00001125; Re=2'500 <-> 0.0000045; Re=5'000 <-> 0.00000225; Re=7'500 <-> 0.0000015; Re=10'000 <-> 0.000001125
NU=${NU:-0.000001125}

# Re=1'000 <-> 0.00001125; Re=10'000 <-> 0.000001125
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 100 -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 1 "
# COM IS 0.399421, COR 0.1, thus fixedCenterDist=0.299412
# xvel is needed here!
OBJECTS="NACA L=$LENGTH xpos=$XPOS angle=$ANGLE fixedCenterDist=0.299412 bFixed=1 xvel=$VELX Apitch=13.15 Fpitch=$FPITCH tAccel=0
"

source launchCommon.sh