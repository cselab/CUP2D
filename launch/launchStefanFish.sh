# Defaults for Options
BPDX=${BPDX:-32}
BPDY=${BPDY:-16}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.1}
# Defaults for Objects
XPOS=${XPOS:-0.2}
LENGTH=${LENGTH:-0.1}
PERIOD=${PERIOD:-1}

# Re=1'000 <-> 0.0000049; Re=10'000 <-> 0.00000049
OPTIONS="-bpdx $BPDX -bpdy $BPDY -extent $EXTENT -CFL $CFL -tdump 0.1 -nu 0.0000049 -tend 50 -poissonType cosine -iterativePensalization 0 -muteAll 0 -verbose 0"
OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=$XPOS bFixed=1
"

source launchCommon.sh
