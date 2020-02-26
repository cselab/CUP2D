OPTIONS="-bpdx 64 -bpdy 32 -tdump 0.1 -nu 0.000045 -tend 20 -poissonType cosine -iterativePensalization 0"
OBJECTS='NACA L=0.15 xpos=0.25 bFixed=1 bForced=1 xvel=0.3 tAccel=5 Apitch=0.15 Fpitch=2.5
stefanfish L=0.15 xpos=0.6 bFixed=0 pidpos=1
'

source launchCommon.sh
