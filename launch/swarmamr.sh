"${main=./main}" \
    -bpdx 2 \
    -bpdy 1 \
    -levelMax 8 \
    -levelStart 5 \
    -Rtol 2 \
    -Ctol 1 \
    -extent 4 \
    -CFL 0.5 \
    -poissonTol 1e-3 \
    -poissonTolRel 1e-2 \
    -maxPoissonRestarts 0 \
    -bAdaptChiGradient 0 \
    -tdump 0.1 \
    -nu 0.00004 \
    -tend 1 \
    -muteAll 0 \
    -verbose 1 \
    -poissonSolver iterative \
    -shapes 'stefanfish L=0.2 T=1 xpos=0.60 ypos=1.00 bFixed=1 pidpos=0 pid=0
		 stefanfish L=0.2 T=1 xpos=0.90 ypos=0.90 bFixed=1 pidpos=0 pid=0
		 stefanfish L=0.2 T=1 xpos=0.90 ypos=1.10 bFixed=1 pidpos=0 pid=0
		 stefanfish L=0.2 T=1 xpos=1.20 ypos=0.80 bFixed=1 pidpos=0 pid=0
		 stefanfish L=0.2 T=1 xpos=1.20 ypos=1.00 bFixed=1 pidpos=0 pid=0
		 stefanfish L=0.2 T=1 xpos=1.20 ypos=1.20 bFixed=1 pidpos=0 pid=0
		 stefanfish L=0.2 T=1 xpos=1.50 ypos=0.90 bFixed=1 pidpos=0 pid=0
		 stefanfish L=0.2 T=1 xpos=1.50 ypos=1.10 bFixed=1 pidpos=0 pid=0
		 stefanfish L=0.2 T=1 xpos=1.80 ypos=1.00 bFixed=1 pidpos=0 pid=0
'
