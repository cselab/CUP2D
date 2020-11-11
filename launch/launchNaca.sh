BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
ANGLE=${ANGLE:-0}
FPITCH=${FPITCH:-1}
XPOS=${XPOS:-0.2}
EXTENT=${EXTENT:-1}

# Re=1'000 <-> 0.00001125; Re=10'000 <-> 0.000001125
OPTIONS="-bpdx $BPDX -bpdy $BPDY -tdump 0 -nu 0.00001125 -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 0 -extent $EXTENT"
# COM OF NACA AIRFOIL IS 0.399421, ROTATION AROUND 0.1, thus fixedCenterDist=0.299412 -> CoR=0.1550882
# Xpos=0.2*D in [0,D]x[0,0.5*D] domain 
OBJECTS="NACA L=0.075 xpos=$XPOS angle=$ANGLE fixedCenterDist=0.299412 bFixed=0 xvel=0.15 Apitch=13.15 Fpitch=$FPITCH tAccel=1
"

source launchCommon.sh
