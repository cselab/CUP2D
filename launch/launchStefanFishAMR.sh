# Defaults for Options
BPDX=${BPDX:-32}
BPDY=${BPDY:-16}
EXTENT=${EXTENT:-1}
LEVELS=${LEVELS:-3}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
CFL=${CFL:-1}
# Defaults for Objects
XPOS=${XPOS:-0.2}
LENGTH=${LENGTH:-0.1}
PERIOD=${PERIOD:-0.5}

# Re=1'000 <-> 0.0000049; Re=10'000 <-> 0.00000049
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu 0.0000049 -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 1"
OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=$XPOS bFixed=1
"

source launchCommon.sh
