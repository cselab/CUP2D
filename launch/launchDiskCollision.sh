LEVELS=${LEVELS:-10}
CFL=${CFL:-0.15}
NU=${NU:-0.0004}
# Re=100
PT=${PT:-1e-9}
PTR=${PTR:-1e-5}

OPTIONS="-bpdx 8 -bpdy 4 -levelMax $LEVELS -levelStart 1 -Rtol 10.0 -Ctol 1.0 -extent 4.0 -CFL $CFL -tdump 0.01 -nu $NU -tend 0.25 -muteAll 0 -verbose 1 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1"
OBJECTS="disk radius=0.1 xpos=1.2 bFixed=1 xvel=0.2 tAccel=0
	 disk radius=0.1 xpos=2.0 xvel=-0.2 tAccel=0"
source launchCommon.sh
