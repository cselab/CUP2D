${main=./main} \
    -bAdaptChiGradient 0 \
    -bMeanConstraint 1 \
    -bpdx 4 \
    -bpdy 2 \
    -CFL 0.4 \
    -Ctol 1.0 \
    -extent 2 \
    -levelMax 9 \
    -levelStart 4 \
    -muteAll 0 \
    -nu 0.00004 \
    -poissonSolver iterative \
    -poissonTol 1e-5 \
    -poissonTolRel 0 \
    -Rtol 2.0 \
    -tdump 0.1 \
    -tend 5.0 \
    -verbose 1 \
    -shapes '
stefanfish L=0.2 T=1 xpos=0.6 bFixed=1
'
