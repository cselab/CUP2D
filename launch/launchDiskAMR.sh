# Defaults for Options
BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-6}
RTOL=${RTOL:-5}
CTOL=${CTOL:-0.01}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.45}
DT=${DT:-0}
PT=${PT:-1e-10}
PTR=${PTR:-0}
# Defaults for Objects
XPOS=${XPOS:-1.0}
XVEL=${XVEL:-0.2}
RADIUS=${RADIUS:-0.1}
PSOLVER="iterative"
# PSOLVER="cuda_iterative"
## to compare against "High-resolution simulations of the flow around an impulsively started cylinder using vortex methods" By P. KOUMOUTSAKOS AND A. LEONARD ##
# Re=40 <-> NU=0.001; Re=200 <-> NU=0.0002; Re=550 <-> NU=0.00007272727273; Re=1000 <-> NU=0.00004; Re=3000 <-> NU=0.00001333333333; Re=9'500 <-> NU=0.000004210526316
#############################################################################################
# Re=10'000 <-> NU=0.000004; Re=100'000 <-> NU=0.0000004; Re=1'000'000 <-> NU=0.00000004


# Re = 1k
# NU=${NU:-0.00004}
# Re = 100
# NU=${NU:-0.0004}
# Re = 10k -> CHANGE LEVELS!
# NU=${NU:-0.000004}
T=${T:-1.0}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -dt $DT -tdump 0.1 -nu $NU -tend 100. -muteAll 0 -verbose 0 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1 -poissonSolver $PSOLVER"
OBJECTS="disk radius=$RADIUS xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0 breakSymmetryExponent=1.0 breakSymmetryStrength=0.5 breakSymmetryType=0 breakSymmetryTime=$T"

# srun ../makefiles/simulation ${OPTIONS} -shapes "${OBJECTS}"
source launchCommon.sh
