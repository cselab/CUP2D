BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
CFL=${CFL:-0.1}

OPTIONS="-bpdx ${BPDX} -bpdy ${BPDY} -tdump 0.1 -nu 0.0000056 -tend 13 -CFL ${CFL} "
OBJECTS='stefanfish L=0.2 xpos=0.50 pid=0 bFixed=1
'

source launchCommon.sh
