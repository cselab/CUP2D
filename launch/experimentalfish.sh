./simulation \
    -bpdx 16 \
    -bpdy 8 \
    -levelMax 5 \
    -levelStart 4 \
    -Rtol 0.1 \
    -Ctol 0.01 \
    -extent 4 \
    -CFL 0.2 \
    -tdump 0.1 \
    -nu 0.00004 \
    -tend 0.5 \
    -muteAll 0 \
    -verbose 1 \
    -poissonTol 1e-6 \
    -poissonTolRel 1e-4 \
    -bAdaptChiGradient 0 \
    -shapes 'experimentFish L=0.2 xpos=2 ypos=1 timeStart=0.0 dtDataset=0.1 path=/users/pweber/korali/examples/study.cases/CUP2D/ bFixed=0
'
