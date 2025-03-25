../simulation \
    -bpdx 3 \
    -bpdy 4 \
    -CFL 0.4 \
    -Ctol 0.01 \
    -extent 0.7 \
    -levelMax 4 \
    -muteAll 0 \
    -nu 0.000243 \
    -poissonSolver iterative \
    -poissonTol 1.0e-3 \
    -poissonTolRel 0 \
    -Rtol 0.1 \
    -tdump 0.0 \
    -tend 60 \
    -verbose 1 \
    -shapes '
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.1 ypos=0.25 bForced=1 bFixed=1 xvel=0.15 tAccel=0 bBlockAng=1 angvelmax=3 freq=0.25
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.1 ypos=0.45 bForced=1 xvel=0.15 tAccel=0 bBlockAng=1 angvelmax=-2.8 freq=1.6
'

