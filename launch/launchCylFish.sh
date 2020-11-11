OPTIONS="-bpdx 48 -bpdy 24 -tdump 0.1 -nu 0.0000225 -tend 20 -poissonType cosine -iterativePensalization 0"
OBJECTS='halfDisk radius=0.075 angle=30 xpos=0.2 bForced=1 bFixed=1 xvel=0.15 tAccel=5
stefanfish L=0.12 xpos=0.5 bFixed=0 pidpos=1
'

source launchCommon.sh
