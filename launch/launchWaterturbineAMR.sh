# Defaults for Options
EXTENT=${EXTENT:-1.2} # largest extent of the domain -> BPD# largest, i.e test section: x=1.2m, y=0.3m
BPDX=${BPDX:-12} #defines the number of blocks (e.g. 8 blocks of 8x8 gridpoints) in x direction, here 8 blocks
BPDY=${BPDY:-3} #defines the number of blocks (of 8x8 gridpoints) in y direction, here 4 blocks
LEVELS=${LEVELS:-6} # AMR grid has levels 0,...,L-1, where L(evels); maximum number of gridpoints is BPD#*2^{L-1}*8, here for x we have 8*2^3*8=512
RTOL=${RTOL:-1} # Tolerance for vorticity magnitude at which a block is refinement
CTOL=${CTOL:-0.1} # Tolerance for vorticity magnitude at which a block is compressed
CFL=${CFL:-0.2} # Courant-Friedrich-Lewis number, restriction for conversion

# Defaults position of the Objects,
XPOS=${XPOS:-0.6} #centre of object placed in streamwise direction
YPOS=${YPOS:-0.15} #centre of object placed laterally placed in the middle of the test section

XVEL=${XVEL:-0.25} #streamwise velocity in m/s? would be 0.25m/s #-0.15
ANGVEL=${ANGVEL:--7.91} #angular velocity of turbine would be 7.91 rad/s; TSR=omega*R <-> omega=1.9*0.25/0.06=7.91 rad/s <-> 7.91*60/2*PI=75.6rpm (rotational speed of turbine) #-0.1

#major and minor axis of the ellipse
MAAXIS=${MAAXIS:-0.015} #blade length is 0.03m #-0.0405 --> MAAXIS seems to be semi-major axis
MIAXIS=${MIAXIS:-0.005} #thickness of blade (?)

# GENERAL: Re=100 <-> 0.0405*2*0.15/100=0.0001215=NU

# Flow condition for harsh-rotating case, with turbine diamter 0.12, bulk velocity 0.25
# Re based on turbine diameter = 30000
# NU=4*0.03*0.25/30000=0.000001
NU=${NU:-0.000001} #kinematic viscosity determined from Reynolds number #-0.0001215

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 20 -muteAll 0 -verbose 0"
## FOUR WINDMILLS
OBJECTS="waterturbine semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS bForced=1 bFixed=1 xvel=$XVEL angvel=$ANGVEL tAccel=0
"

source launchCommon.sh
