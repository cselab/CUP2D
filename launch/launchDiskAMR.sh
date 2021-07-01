# Defaults for Options
BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-4}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0}
DT=${DT:-1e-4}
PT=${PT:-1e-9}
PTR=${PTR:-1e-4}
# Defaults for Objects
XPOS=${XPOS:-1.2}
XVEL=${XVEL:-0.2}
RADIUS=${RADIUS:-0.1}
## to compare against "High-resolution simulations of the flow around an impulsively started cylinder using vortex methods" By P. KOUMOUTSAKOST AND A. LEONARD ##
# Re=40 <-> NU=0.001; Re=200 <-> NU=0.0002; Re=550 <-> NU=0.00007272727273; Re=1000 <-> NU=0.00004; Re=3000 <-> NU=0.00001333333333; Re=9'500 <-> NU=0.000004210526316; Re=10'000 <-> NU=0.000004
#############################################################################################
# Re=100'000 <-> NU=0.0000004; Re=1'000'000 <-> NU=0.00000004
NU=${NU:-0.000004210526316}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 1 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -dt $DT -tdump 0.1 -nu $NU -tend 5 -muteAll 0 -verbose 0 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1"
OBJECTS="disk radius=$RADIUS xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0"

source launchCommon.sh
