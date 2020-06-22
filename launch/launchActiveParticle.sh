#NU=pow(linCircVel, 2)/Re;
#t_dump=0.1*T_min
#T_min=2*pi*radiusForcedMotion_min/linCircVel;
#Re=1000
#linCircVel=0.15
NU=${NU:-2.25e-05}
OPTIONS="-bpdx 20 -bpdy 20 -tdump 0.001 -nu ${NU} -CFL 0.1 -iterativePenalization 0 -tend 3 -poissonType cosine "
OBJECTS="activeParticle_radius=0.01_bForced=1_bFixed=0_xCenterRotation=0.5_yCenterRotation=0.5_xpos=0.6_ypos=0.5_angCircVel=7_tAccel=0.006_tStartElliTransfer=0.01_initialRadius=0.1_finalRadius=0.2"
source launchCommon.sh
