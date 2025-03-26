./simulation \
    -bMeanConstraint 1 \
    -bpdx 2 \
    -bpdy 1 \
    -levelMax 8 \
    -levelStart 5 \
    -Rtol 0.1 \
    -Ctol 0.001 \
    -extent 4 \
    -CFL 0.5 \
    -poissonTol 1e-6 \
    -poissonTolRel 1e-4 \
    -maxPoissonRestarts 100 \
    -bAdaptChiGradient 1 \
    -tdump 0.1 \
    -nu 0.00004 \
    -tend 100 \
    -muteAll 0 \
    -verbose 0 \
    -shapes '
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
'
