#Flow around a cylinder - settings

# Blocks per direction at refinement level 0
BPDX=${BPDX:-32} #this means that the uniform mesh is 256x256 points (32 blocks x 8 points each)
BPDY=${BPDY:-32} #if that's slow, you can decrease the blocks to 16 or even 8

# Number of refinement levels. The number of blocks at a given level is BPDX * 2^{level-1} x BPDY * 2^{level-1}
LEVELS=${LEVELS:-1}

# A block will get refined if |omega| > Rtol and four blocks will get compressed if |omega| < Ctol
RTOL=${RTOL-1.0}
CTOL=${CTOL-0.01}

# Domain size
EXTENT=${EXTENT:-1}

# This controls the timestep and must always be less than 1.0 (higher values mean less accuracy)
CFL=${CFL:-0.1}

# Tolerance for error in Poisson equation
PT=${PT:-1e-10} #tolerance for |Ax-b|
PTR=${PTR:-0} #tolerance for |Ax-b|/|Ax_{0}-b|

XPOS=${XPOS:-0.5} #cylinder position
XVEL=${XVEL:-0.2} #cylinder velocity
RADIUS=${RADIUS:-0.1} #cylinder radius
NU=${NU:-0.00004} #fluid viscosity

TEND=${TEND:-0.25} #final time

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 0 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.05 -nu $NU -tend $TEND -muteAll 0 -verbose 1 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1"
OBJECTS="disk radius=$RADIUS xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0"

source cuda_launchCommon.sh
