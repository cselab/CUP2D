BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
ANGLE=${ANGLE:-0}
FPITCH=${FPITCH:-1}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -tdump 0.1 -nu 0.0000225 -tend 50 -poissonType cosine -iterativePensalization 1 -muteAll 0 -extent "
# COM OF NACA AIRFOIL IS 0.399421, ROTATION AROUND 0.1, thus fixedCenterDist=0.299412 -> CoR=0.1550882
# Xpos=0.2 in [0,1]x[0,0.5] domain; for validation Xpos= in [0,15]x[0,7.5]
OBJECTS='NACA L=0.15 xpos=0.2 angle=$ANGLE fixedCenterDist=0.299412 bFixed=0 xvel=0.0 Apitch=13.15 Fpitch=$FPITCH tAccel=1
'

source launchCommon.sh
