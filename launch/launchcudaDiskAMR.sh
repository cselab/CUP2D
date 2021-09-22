#Flow around a cylinder - settings

# Blocks per direction at refinement level 0
BPDX=${BPDX:-8}
BPDY=${BPDY:-4}

# Number of refinement levels. The number of blocks at a given level is BPDX * 2^{level-1} x BPDY * 2^{level-1}
LEVELS=${LEVELS:-8}

# A block will get refined if |omega| > Rtol and four blocks will get compressed if |omega| < Ctol
RTOL=${RTOL-1.0}
CTOL=${CTOL-0.01}

# Domain size
EXTENT=${EXTENT:-2}

# This controls the timestep and must always be less than 1.0 (higher values mean less accuracy)
CFL=${CFL:-0.1}

# Tolerance for error in Poisson equation
PT=${PT:-1e-10} #tolerance for |Ax-b|
PTR=${PTR:-0} #tolerance for |Ax-b|/|Ax_{0}-b|

XPOS=${XPOS:-0.8} #cylinder position
XVEL=${XVEL:-0.2} #cylinder velocity
RADIUS=${RADIUS:-0.1} #cylinder radius
NU=${NU:-0.00004} #fluid viscosity

TEND=${TEND:-0.25} #final time

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 1 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.05 -nu $NU -tend $TEND -muteAll 0 -verbose 1 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1"
OBJECTS="disk radius=$RADIUS xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0"

source launchcudaCommon.sh
