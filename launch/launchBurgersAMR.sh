# Defaults for Options
BPDX=${BPDX:-64}
BPDY=${BPDY:-64}
LEVELS=${LEVELS:-1}
RTOL=${RTOL:-0.1}
CTOL=${CTOL:-0.01}
EXTENT=${EXTENT:-6.2831853072}
CFL=${CFL:-0.15}
DT=${DT:-0}
PT=${PT:-1e-10}
PTR=${PTR:-0}
NU=${NU:-0.025}
# NU=${NU:-0.00001}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 0 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -dt $DT -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 1 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1 -bBurgers 1 -ic gaussian"

source launchCommon.sh
