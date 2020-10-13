OPTIONS="-bpdx 12 -bpdy 6 -levelMax 3 -Rtol 5.0 -Ctol 1.0 -tdump 0.1 -nu 0.000045 -tend 50. -poissonType cosine -iterativePensalization 0"
#OPTIONS="-bpdx 6 -bpdy 3 -levelMax 6 -Rtol 5.0 -Ctol 1.0 -tdump 0.1 -nu 0.000045 -tend 50. -poissonType cosine -iterativePensalization 0"
OBJECTS='NACA L=0.15 xpos=0.25 bFixed=1 bForced=1 xvel=0.0 Apitch=0.15 Fpitch=2.5
stefanfish L=0.125 xpos=0.6 bFixed=0 pidpos=1
'
source launchCommon.sh
