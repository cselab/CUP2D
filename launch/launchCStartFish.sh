BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
CFL=${CFL:-0.1}

OPTIONS="-bpdx ${BPDX} -bpdy ${BPDY} -tdump 0.01 -nu 0.0001236364255 -tend 3 -CFL ${CFL} "
OBJECTS='cstartfish L=0.2 xpos=0.50 pid=0 bFixed=0
'

source launchCommon.sh
