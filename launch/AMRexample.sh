OPTIONS="-tdump 0.1 -nu 0.0000225 -tend 20 -poissonType cosine -iterativePensalization 0"

#AMR OPTIONS
OPTIONS+=" -bpdx 8 -bpdy 4"
OPTIONS+=" -levelMax 4 -Rtol 0.5 -Ctol 0.1"

OBJECTS='halfDisk radius=0.075 angle=30 xpos=0.2 bForced=1 bFixed=1 xvel=0.15 tAccel=5
stefanfish L=0.12 xpos=0.5 bFixed=0 pidpos=1'
source launchCommon.sh
