"${main=./main}" \
    -bpdx 4 \
    -bpdy 4 \
    -CFL 0.4 \
    -Ctol 0.01 \
    -extent 1.0 \
    -levelMax 3 \
    -muteAll 0 \
    -nu 0.000243 \
    -poissonSolver iterative \
    -poissonTol 1.0e-3 \
    -poissonTolRel 0 \
    -Rtol 0.1 \
    -tdump 0.1 \
    -tend 600 \
    -verbose 1 \
    -ic random \
    -shapes '
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.6 ypos=0.6 bBlockAng=1 angvelmax=1 freq=0.25
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.4 ypos=0.4 bBlockAng=1 angvelmax=-1 freq=0.25
'
