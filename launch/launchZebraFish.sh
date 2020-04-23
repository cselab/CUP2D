# ZebraFish launch script

BPDX=${BPDX:-16}
BPDY=${BPDY:-16}
CFL=${CFL:-0.1}

OPTIONS="-bpdx ${BPDX} -bpdy ${BPDY} -tdump 0.03 -nu 0.0001931818182 -tend 4.5 -CFL ${CFL} "
OBJECTS='zebrafish L=0.25 T=0.5882352941 xpos=0.50 ypos=0.50 angle=-98
'

source launchCommon.sh
