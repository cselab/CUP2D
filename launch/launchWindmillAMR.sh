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
XVEL=${XVEL:-0.15}
MAAXIS=${MAAXIS:-0.0375}
MIAXIS=${MIAXIS:-0.01}

NU=${NU:-0.0001125}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 10 -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 1"
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS bForced=1 xvel=$XVEL tAccel=0"

source launchCommon.sh
