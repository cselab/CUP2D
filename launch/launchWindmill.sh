# Defaults for Options
BPDX=${BPDX:-32} # number of blocks in x direction, 8 cells per block in compilation
BPDY=${BPDY:-16}
LEVELS=${LEVELS:-3} # each block can be refined twice, double number of points per refinement. 
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.5}
# Defaults for Objects
XPOS=${XPOS:-0.2}
XVEL=${XVEL:-0.15}
MAAXIS=${MAAXIS:-0.0375}
MIAXIS=${MIAXIS:-0.01}

NU=${NU:-0.0001125}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 10 -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 1"
# bForced, tAccel is needed here!
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS bForced=1 xvel=$XVEL tAccel=0"

#-angvel changes angular velocity of object

source launchCommon.sh
