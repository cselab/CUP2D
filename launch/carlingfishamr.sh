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
    -nu 0.0000056 \
    -tend 50 \
    -muteAll 0 \
    -verbose 0 \
    -poissonTol 1e-6 \
    -poissonTolRel 1e-4 \
    -bAdaptChiGradient 0 \
    -shapes 'carlingfish L=0.2 T=1 xpos=1.2 bFixed=1
'
