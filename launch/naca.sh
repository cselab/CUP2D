./simulation \
    -bpdx 16 \
    -bpdy 8 \
    -CFL 0.45 \
    -Ctol 0.01 \
    -extent 4 \
    -levelMax 6 \
    -levelStart 4 \
    -nu 0.00001 \
    -poissonSolver iterative \
    -poissonTol 1e-10 \
    -poissonTolRel 0 \
    -Rtol 5 \
    -tdump 0.1 \
    -tend 10. \
    -verbose 0 \
    -shapes '
NACA L=0.20 xpos=0.60 ypos=1.00 fixedCenterDist=0.0 bFixed=1 xvel=0.15 yvel=0.0  Apitch=0.0 Fpitch=0.0 Mpitch=0.0 Aheave=0.0 Fheave=0.0 tRatio=1.00
'
