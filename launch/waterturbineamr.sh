"${main=./main}" \
    -bpdx 12 \
    -bpdy 3 \
    -levelMax 6 \
    -Rtol 1 \
    -Ctol 0.1 \
    -extent 1.2 \
    -CFL 0.2 \
    -tdump 0.1 \
    -nu 0.000001 \
    -tend 20 \
    -muteAll 0 \
    -verbose 0 \
    -shapes 'waterturbine semiAxisX=0.015 semiAxisY=0.005 xpos=0.6 ypos=0.15 bForced=1 bFixed=1 xvel=0.25 angvel=-7.91 tAccel=0
'
