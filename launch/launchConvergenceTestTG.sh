# Defaults for Options
BPDX=${BPDX:-4}
BPDY=${BPDY:-4}
TEND=${TEND:-0.25}
RTOL=${RTOL:-5}
CTOL=${CTOL:-0.5}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.1}
DT=${DT:-0}
PT=${PT:-1e-3}
PTR=${PTR:-1e-2}
# Defaults for Objects
XPOS=${XPOS:-0.5}
XVEL=${XVEL:-0}
RADIUS=${RADIUS:-0.2}
PSOLVER="iterative"
IC="TaylorGreen"
#PSOLVER="cuda_iterative"
## to compare against "High-resolution simulations of the flow around an impulsively started cylinder using vortex methods" By P. KOUMOUTSAKOST AND A. LEONARD ##
# Re=40 <-> NU=0.001; Re=200 <-> NU=0.0002; Re=550 <-> NU=0.00007272727273; Re=1000 <-> NU=0.00004; Re=3000 <-> NU=0.00001333333333; Re=9'500 <-> NU=0.000004210526316; Re=10'000 <-> NU=0.000004
#############################################################################################
# Re=100'000 <-> NU=0.0000004; Re=1'000'000 <-> NU=0.00000004
NU=${NU:-0.001}
for LEVELS in {4..10}
do
    FILE="ConvergenceTest$LEVELS"
    OPTIONS="-ic $IC -bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 1 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -dt $DT -tdump 0.025 -nu $NU -tend 0.25 -muteAll 0 -verbose 1 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1 -poissonSolver $PSOLVER"
    OBJECTS="ElasticDisk radius=$RADIUS xpos=$XPOS bForced=0 tAccel=0 G=1 "
    source launchCommon.sh "TestTimeConvergence$CFL"
done
