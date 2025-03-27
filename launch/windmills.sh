"${main=./main}" \
    -bpdx 4 \
    -bpdy 4 \
    -CFL 0.4 \
    -Ctol 0.01 \
    -extent 1.0 \
    -levelMax 7 \
    -muteAll 0 \
    -nu 0.00000243 \
    -poissonSolver iterative \
    -poissonTol 1.0e-3 \
    -poissonTolRel 0 \
    -Rtol 0.1 \
    -tdump 0.1 \
    -tend 600 \
    -verbose 1 \
    -ic random \
    -shapes '
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.44 ypos=0.38 bBlockAng=1 angvelmax=-1.82018 freq=0.25
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.44 ypos=0.53 bBlockAng=1 angvelmax=-1.92776 freq=0.25
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.44 ypos=0.68 bBlockAng=1 angvelmax=1.54212 freq=0.25
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.59 ypos=0.38 bBlockAng=1 angvelmax=1.98246 freq=0.25
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.59 ypos=0.53 bBlockAng=1 angvelmax=-1.1189 freq=0.25
'
