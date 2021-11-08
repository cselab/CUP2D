#!/bin/bash
BPDX=${BPDX:-2}
BPDY=${BPDY:-1}
LEVELS=${LEVELS:-8}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.001}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.5}
PT=${PT:-1e-6}
PTR=${PTR:-1e-4}
NU=${NU:-0.00004} #Re=1000

OPTIONS="-bMeanConstraint 1 -bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 5 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -poissonTol $PT -poissonTolRel $PTR -maxPoissonRestarts 100 -bAdaptChiGradient 1 -tdump 0.1 -nu $NU -tend 100 -muteAll 0 -verbose 0"

OBJECTS="
		 stefanfish L=0.2 T=1.0 xpos=1.00 ypos=0.50 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.00 ypos=0.70 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.00 ypos=0.90 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.00 ypos=1.10 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.00 ypos=1.30 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.00 ypos=1.50 bFixed=1 Forced=1 xvel=0.1 

		 stefanfish L=0.2 T=1.0 xpos=1.40 ypos=0.60 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.40 ypos=0.80 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.40 ypos=1.00 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.40 ypos=1.20 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.40 ypos=1.40 bFixed=1 Forced=1 xvel=0.1 

		 stefanfish L=0.2 T=1.0 xpos=1.80 ypos=0.50 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.80 ypos=0.70 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.80 ypos=0.90 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.80 ypos=1.10 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.80 ypos=1.30 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=1.80 ypos=1.50 bFixed=1 Forced=1 xvel=0.1 

		 stefanfish L=0.2 T=1.0 xpos=2.20 ypos=0.60 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.20 ypos=0.80 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.20 ypos=1.00 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.20 ypos=1.20 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.20 ypos=1.40 bFixed=1 Forced=1 xvel=0.1 

		 stefanfish L=0.2 T=1.0 xpos=2.60 ypos=0.50 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.60 ypos=0.70 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.60 ypos=0.90 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.60 ypos=1.10 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.60 ypos=1.30 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=2.60 ypos=1.50 bFixed=1 Forced=1 xvel=0.1 

		 stefanfish L=0.2 T=1.0 xpos=3.00 ypos=0.60 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=3.00 ypos=0.80 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=3.00 ypos=1.00 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=3.00 ypos=1.20 bFixed=1 Forced=1 xvel=0.1 
		 stefanfish L=0.2 T=1.0 xpos=3.00 ypos=1.40 bFixed=1 Forced=1 xvel=0.1 
"

source launchCommon.sh
