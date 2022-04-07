# Defaults for Options
BPDX=${BPDX:-8}
BPDY=${BPDY:-4}
LEVELS=${LEVELS:-10}
RTOL=${RTOL:-5}
CTOL=${CTOL:-0.5}
EXTENT=${EXTENT:-2}
CFL=${CFL:-0.15}
DT=${DT:-0}
PT=${PT:-1e-10}
PTR=${PTR:-0}
# Defaults for Objects
XPOS=${XPOS:-0.5}
XVEL=${XVEL:-0.2}
RADIUS=${RADIUS:-0.1}
#PSOLVER="iterative"
PSOLVER="cuda_iterative"
## to compare against "High-resolution simulations of the flow around an impulsively started cylinder using vortex methods" By P. KOUMOUTSAKOST AND A. LEONARD ##
# Re=40 <-> NU=0.001; Re=200 <-> NU=0.0002; Re=550 <-> NU=0.00007272727273; Re=1000 <-> NU=0.00004; Re=3000 <-> NU=0.00001333333333; Re=9'500 <-> NU=0.000004210526316; Re=10'000 <-> NU=0.000004
#############################################################################################
# Re=100'000 <-> NU=0.0000004; Re=1'000'000 <-> NU=0.00000004
NU=${NU:-0.0000004}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -dt $DT -tdump 0.025 -nu $NU -tend 10. -muteAll 0 -verbose 0 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1 -poissonSolver $PSOLVER"
OBJECTS="disk radius=$RADIUS xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0"

source launchCommon.sh
