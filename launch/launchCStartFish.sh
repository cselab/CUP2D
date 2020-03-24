BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
CFL=${CFL:-0.1}

OPTIONS="-bpdx ${BPDX} -bpdy ${BPDY} -tdump 0.000748 -nu 0.00165289256 -tend 13 -CFL ${CFL} "
OBJECTS='cstartfish L=0.2 T=0.00748 xpos=0.50 pid=0 bFixed=0
'

source launchCommon.sh
