# Reynolds 1000 :
NU=${NU:-0.000015625}
# Reynolds 9500 :
#NU=${NU:-0.000001644736842}
OPTIONS="-bpdx 10 -bpdy 10 -tdump 0.01 -nu ${NU} -CFL 0.1 -iterativePenalization 0 -tend 10 -poissonType cosine "
OBJECTS="disk_radius=0.01_xpos=0.6_ypos=0.5_bForced=1_bFixed=0_angvel=1000"
source launchCommon.sh