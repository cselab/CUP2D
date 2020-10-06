#OPTIONS="-bpdx 32 -bpdy 16 -dlm 1 -CFL 0.1 -tdump 0.1 -poissonType cosine -nu 0.00005 -tend 50 "
#OPTIONS="-bpdx 24 -bpdy 12 -dlm 1 -CFL 0.2 -tdump 0.1 -poissonType cosine -nu 0.00005 -tend 50 "
OPTIONS="-bpdx 8 -bpdy 4 -dlm 1 -CFL 0.2 -tdump 0.05 -poissonType cosine -nu 0.00005 -tend 2.5 -levelMax 3 -Rtol=0.01 -Ctol=0.001"
OBJECTS='halfDisk radius=0.05 angle=30 xpos=0.1 ypos=0.25 bForced=1 bFixed=1 xvel=0.1 tAccel=5
stefanfish L=0.2 xpos=0.4 bFixed=0 pidpos=1
'
#OBJECTS='halfDisk radius=0.05 angle=0 xpos=0.1 ypos=0.25 bForced=1 bFixed=1 xvel=0.1 tAccel=5
#'

source launchCommon.sh
