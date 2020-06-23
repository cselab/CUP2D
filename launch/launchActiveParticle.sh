#NU=pow(linCircVel, 2)/Re;
#t_dump=0.1*T_min
#T_min=2*pi*radiusForcedMotion_min/linCircVel;
#Re=1000
#linCircVel=0.15
NU=${NU:-2.25e-05}
OPTIONS="-bpdx 50 -bpdy 50 -tdump 0.01 -nu ${NU} -CFL 0.05 -iterativePenalization 0 -tend 2.5 -poissonType cosine "
OBJECTS="activeParticle_radius=0.01_bForced=1_bFixed=0_xCenterRotation=0.5_yCenterRotation=0.5_xpos=0.6_ypos=0.5_angCircVel=7_tAccel=0.1_tStartElliTransfer=0.23_finalRadius=0.2_
tStartAccelTransfer=1.5_finalAngRotation=10_forcedAccelCirc=6"
source launchCommon.sh
