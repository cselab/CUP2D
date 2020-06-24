#NU=pow(linCircVel, 2)/Re;
#t_dump=0.1*T_min
#T_min=2*pi*radiusForcedMotion_min/linCircVel;
#Re=1000
#linCircVel=0.15
NU=${NU:-2.25e-05}
OPTIONS="-bpdx 16 -bpdy 16 -tdump 0.01 -nu ${NU} -CFL 0.05 -iterativePenalization 0 -tend 1.8 -poissonType cosine "
OBJECTS="activeParticle_radius=0.01_bForced=1_bFixed=0_xCenterRotation=0.5_yCenterRotation=0.5_xpos=0.6_ypos=0.5_forcedOmegaCirc=7_tAccel=0.05_tStartElliTransfer=0.10_finalRadiusRotation=0.2_tStartAccelTransfer=1.3_finalAngRotation=10_forcedAccelCirc=6"
source launchCommon.sh
