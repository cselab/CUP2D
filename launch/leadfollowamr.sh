${simulation="${main=./main}"} \
    -bpdx 4 \
    -bpdy 2 \
    -levelMax 7 \
    -levelStart 4 \
    -Rtol 2 \
    -Ctol 1 \
    -extent 2 \
    -CFL 0.4 \
    -poissonTol 1e-5 \
    -poissonTolRel 0 \
    -bMeanConstraint 1 \
    -bAdaptChiGradient 0 \
    -tdump 0.1 \
    -nu 0.00004 \
    -tend 0 \
    -muteAll 1 \
    -verbose 0 \
    -shapes 'waterturbine semiAxisX=0.05 semiAxisY=0.017 xpos=0.6 bForced=1 bFixed=1 xvel=0.2 angvel=-0.79 tAccel=0
		stefanfish L=0.2 T=1 xpos=0.9
	'
