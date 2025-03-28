${main=./main} \
    -bAdaptChiGradient 0 \
    -bMeanConstraint 1 \
    -bpdx 2 \
    -bpdy 1 \
    -CFL 0.5 \
    -Ctol 1 \
    -extent 4 \
    -levelMax 8 \
    -levelStart 5 \
    -muteAll 0 \
    -nu 0.00004 \
    -poissonSolver iterative \
    -poissonTol 1e-3 \
    -poissonTolRel 1e-2 \
    -Rtol 2.0 \
    -tdump 0.5 \
    -tend 10.0 \
    -verbose 1 \
    -shapes '
  stefanfish angle=0 L=0.2 xpos=1.8 ypos=0.8
  stefanfish angle=180 L=0.2 xpos=1.6 ypos=0.8
'
