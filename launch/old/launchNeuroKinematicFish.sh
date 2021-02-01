# Options necessary to replicate C-Start: optimal start of larval fish (Gazzola et. al.).
# To obtain the precise quantities reported in the paper, the resolution must be bpdx 32, bpdy 32.
# All quantities are scaled to satisfy Reynolds number 550.
# The episode length of 1.5882352941 corresponds to [0, Tprep + 2*Tprop]

# AMR version
# BPDX=${BPDX:-8}
# BPDY=${BPDY:-8}
# CFL=${CFL:-0.1}

# OPTIONS="-bpdx ${BPDX} -bpdy ${BPDY} -tdump 0.01 -nu 0.0001931818182 -tend 1.5882352941 -CFL ${CFL} -Rtol 10.0 -Ctol 5.0 -levelMax 3"
# OBJECTS='cstartfish L=0.25 xpos=0.50 pid=0 bFixed=0 angle=-98

BPDX=${BPDX:-16}
BPDY=${BPDY:-16}
CFL=${CFL:-0.1}

OPTIONS="-bpdx ${BPDX} -bpdy ${BPDY} -tdump 0.03 -nu 0.0001931818182 -tend 5.0 -CFL ${CFL} "
OBJECTS='neurokinematicfish L=0.25 T=0.5882352941 xpos=0.50 ypos=0.50 angle=-90
'

source launchCommon.sh
