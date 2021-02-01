BPDX=${BPDX:-16}
BPDY=${BPDY:-8}

OPTIONS="-bpdx ${BPDX} -bpdy ${BPDY} -tdump 0.02 -nu 0.0003 -tend 80 "
OBJECTS='disk radius=0.03 xpos=0.3 ypos=0.25 bForced=1 bFixed=1 xvel=0.2
'

source launchCommon.sh
