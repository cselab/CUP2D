# Defaults for Options
BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-7}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.2}
PT=${PT:-1e-6}
PTR=${PTR:-0}
PR=${PR:-100}
PI=${PI:-10000}
# Defaults for Objects
LENGTH=${LENGTH:-0.2}
TRATIO=${TRATIO:-0.1}
XPOS=${XPOS:-1.2} #3.8
ANGLE=${ANGLE:-0}
RCENTER=${RCENTER:-0.2603} # rotation trough center of circle
VELX=${VELX:-0.2}
APITCH=${APITCH:-7}
FPITCH=${FPITCH:-0.664}

# Re=5'400 <-> NU=0.000007407407407
NU=${NU:-0.000007407407407}

# Re=1'000 <-> 0.00001125; Re=10'000 <-> 0.000001125 # 
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 100 -muteAll 0 -verbose 0 -poissonTol $PT -poissonTolRel $PTR -maxPoissonRestarts $PR -maxPoissonIterations $PI -bAdaptChiGradient 0 -bMeanConstraint 1"
# COM IS 0.3103, COR 0.05, thus fixedCenterDist=0.2603
OBJECTS="teardrop L=$LENGTH tRatio=$TRATIO xpos=$XPOS angle=$ANGLE fixedCenterDist=$RCENTER xvel=$VELX Apitch=$APITCH Fpitch=$FPITCH tAccel=0 bFixed=1
"

source launchCommon.sh