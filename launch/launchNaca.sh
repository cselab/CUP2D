# Defaults for Options
BPDX=${BPDX:-48}
BPDY=${BPDY:-24  }
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.1}

# Defaults for Objects
XPOS=${XPOS:-0.2}
ANGLE=${ANGLE:-0}
FPITCH=${FPITCH:-1}

# Re=1'000 <-> 0.00001125; Re=10'000 <-> 0.000001125
OPTIONS="-bpdx $BPDX -bpdy $BPDY -extent $EXTENT -CFL $CFL -tdump 0.1 -nu 0.00001125 -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 0 "
# COM IS 0.399421, COR 0.1, thus fixedCenterDist=0.299412
# xvel is needed here!
OBJECTS="NACA L=0.075 xpos=$XPOS angle=$ANGLE fixedCenterDist=0.299412 bFixed=1 xvel=0.15 Apitch=13.15 Fpitch=$FPITCH tAccel=0
"

source launchCommon.sh