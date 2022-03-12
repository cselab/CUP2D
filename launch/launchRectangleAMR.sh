# Defaults for Options
BPDX=${BPDX:-8}
BPDY=${BPDY:-4}
LEVELS=${LEVELS:-8}
RTOL=${RTOL:-0.1}
CTOL=${CTOL:-0.01}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.15}
DT=${DT:-0}
PT=${PT:-1e-10}
PTR=${PTR:-0}
# Defaults for Objects
XPOS=${XPOS:-0.5}
XVEL=${XVEL:-0.2}
EXTENTX=${EXTENTX:-0.2}
EXTENTY=${EXTENTY:-0.2}
# Re=5000 <-> 0.000004 (R=0.1) or 0.000008 (R=0.2)
NU=${NU:-0.000008}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -dt $DT -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 0 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1"
OBJECTS="rectangle extentX=$EXTENTX extentY=$EXTENTY xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBreakSymmetry=1"

source launchCommon.sh
