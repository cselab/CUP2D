#----------------------------------
#Settings for cylinder
#----------------------------------
XPOS=${XPOS:-1.0}      #cylinder center (x-coordinate)
XVEL=${XVEL:-0.2}      #cylinder velocity (x-component)
RADIUS=${RADIUS:-0.1}  #cylinder radius

#------------------------------------------------
#Settings for breaking symmetry for cylinder wake
#------------------------------------------------
BREAK_SYMMETRY_TYPE=${BREAK_SYMMETRY_TYPE:0}
BREAK_SYMMETRY_STRENGTH=${BREAK_SYMMETRY_STRENGTH:0.0}
BREAK_SYMMETRY_TIME=${BREAK_SYMMETRY_TIME:1.0}
#
# These parameters can introduce a perturbation in the 
# cylinder's motion, in order to break symmetric conditions
# in its wake.
# 
# * If BREAK_SYMMETRY_TYPE = 0, no perturbation is introduced.
# * If BREAK_SYMMETRY_TYPE = 1, then we add an angular velocity
#   for BREAK_SYMMETRY_TIME<t<t-BREAK_SYMMETRY_TIME+1, given from:
#    omega(t) = BREAK_SYMMETRY_STRENGTH * |U| * D * sin(2*pi*(t-BREAK_SYMMETRY_TIME))
#   where D is the cylinder diameter and U is the x-component of the cylinder velocity
# * If BREAK_SYMMETRY_TYPE = 1, then we add a y velocity
#   for BREAK_SYMMETRY_TIME<t<t-BREAK_SYMMETRY_TIME+1, given from:
#    v(t) = BREAK_SYMMETRY_STRENGTH * |U| * sin(2*pi*(t-BREAK_SYMMETRY_TIME))
#

#----------------------------------
#Settings for pressure equation 
#----------------------------------
PSOLVER="iterative"       #CPU solver
#PSOLVER="cuda_iterative" #GPU solver
PT=${PT:-1e-10}           #absolute error tolerance
PTR=${PTR:-0}             #relative error tolerance

#----------------------------------
#Settings for simulation domain
#----------------------------------
EXTENT=${EXTENT:-4}    #length of largest side
BPDX=${BPDX:-16}       #number of blocks in x-side, at refinement level = 0
BPDY=${BPDY:-8}        #number of blocks in y-side, at refinement level = 0
#
# Coarsest possible mesh (at refinement level = 0) is a
# (BPDX * BS) x (BPDY * BS) grid, where BS is the number 
# of grid points per block (equal to 8, by default).

#--------------------------------------
#Settings for Adaptive Mesh Refinement
#--------------------------------------
RTOL=${RTOL:-5}                #grid is refined when curl(u) > Rtol (u: fluid velocity)
CTOL=${CTOL:-0.01}             #grid is compressed when curl(u) < Ctol (u: fluid velocity)
LEVELS=${LEVELS:-6}            #maximum number of refinement levels allowed
LEVELSSTART=${LEVELSSTART:-4}  #at t=0 the grid is uniform and at this refinement level. Must be strictly less than LEVELS.

#--------------------------------------
#Other settings
#--------------------------------------
NU=${NU:-0.0004} #fluid viscosity
#The Reynolds number is defined as Re = XVEL * 2 * RADIUS / NU and can be controlled by
#modifying the values of NU. Here are some examples:
# Re=40    <-> NU=0.001
# Re=200   <-> NU=0.0002
# Re=550   <-> NU=0.00007272727273
# Re=1000  <-> NU=0.00004
# Re=3000  <-> NU=0.00001333333333
# Re=9'500 <-> NU=0.000004210526316
T=${T:-1.0}  #at t=T we introduce a small disturbance in the cylinder's velocity. By doing so we break symmetric flow conditions and get vortex shedding.

#--------------------------------------
#Timestep and file saving
#--------------------------------------
TDUMP=${TDUMP:-0.1}   #Save files for t = i*TDUMP, i=0,1,...
TEND=${TEND:-10.}     #Perform simulation until t=TEND
CFL=${CFL:-0.45}      #Courant number: controls timestep size (should not exceed 1.0).
VERBOSE=${VERBOSE:-0} #Set to 1 for more verbose screen output.

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart $LEVELSSTART -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump $TDUMP -nu $NU -tend $TEND -verbose $VERBOSE -poissonTol $PT -poissonTolRel $PTR -poissonSolver $PSOLVER"
OBJECTS="disk radius=$RADIUS xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL breakSymmetryStrength=$BREAK_SYMMETRY_STRENGTH breakSymmetryType=$BREAK_SYMMETRY_TYPE breakSymmetryTime=$BREAK_SYMMETRY_TIME"

source launchCommon.sh
