#!/bin/bash

#-----------------------------------------
#Settings for airfoil (symmetric NACA0012)
#-----------------------------------------
# Airfoil motion is defined as a combination of an imposed rotation,
# an imposed motion in the y-direction (heaving) and an imposed constant
# velocity (uforced,vforced).
#
#       Rotation:
#       a(t) = Mpitch*(pi/180) + Apitch*(pi/180)*sin(2*pi*Fpitch*t), where:
#              a(t)   :pitching angle
#              Mpitch :mean pitch angle
#              Fpitch :pitching frequency
#       omega(t) = da/dt
#       Rotation can be defined around a point that is located at a distance of
#       d = fixedCenterDist*L for the airfoil's center of mass, where L is
#       the airfoil's cord (length)
#       In this case, we the following velocity is added to the motion:
#         u_rot = - d*omega(t)*sin(a(t))
#         v_rot = + d*omega(t)*cos(a(t))
#
#       Heaving motion:
#       y(t) = Aheave*cos(2*pi*Fheave*t)
#       v(t) = dy/dt = -2.0*pi*Fheave*Aheave*sin(2*pi*Fheave*t)
#
#       It is also possible to add a constant velocity (uforced,vforced) to the motion.
# 
FIXEDCENTERDIST=${FIXEDCENTERDIST:-0.0}
APITCH=${APITCH:-0.0} #this is degrees, not rad
FPITCH=${FPITCH:-0.0}
MPITCH=${MPITCH:-0.0} #this is degrees, not rad
AHEAVE=${AHEAVE:-0.0}
FHEAVE=${FHEAVE:-0.0}
UFORCED=${UFORCED:-0.15}
VFORCED=${VFORCED:-0.0}

LENGTH=${LENGTH:-0.20} #airfoil cord length
XPOS=${XPOS:-0.60} #airfoil position in x
YPOS=${YPOS:-1.00} #airfoil position in y
TRATIO=${YPOS:-0.12} #this number controls airfoil thickness


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
NU=${NU:-0.00001} #fluid viscosity
#The Reynolds number is defined as Re = XVEL * LENGTH / NU and can be controlled by
#modifying the values of NU.

#--------------------------------------
#Timestep and file saving
#--------------------------------------
TDUMP=${TDUMP:-0.1}   #Save files for t = i*TDUMP, i=0,1,...
TEND=${TEND:-10.}     #Perform simulation until t=TEND
CFL=${CFL:-0.45}      #Courant number: controls timestep size (should not exceed 1.0).
VERBOSE=${VERBOSE:-0} #Set to 1 for more verbose screen output.

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart $LEVELSSTART -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump $TDUMP -nu $NU -tend $TEND -verbose $VERBOSE -poissonTol $PT -poissonTolRel $PTR -poissonSolver $PSOLVER"

OBJECTS="NACA L=$LENGTH xpos=$XPOS ypos=$YPOS fixedCenterDist=$FIXEDCENTERDIST bFixed=1 xvel=$UFORCED yvel=$VFORCED  Apitch=$APITCH Fpitch=$FPITCH Mpitch=$MPITCH Aheave=$AHEAVE Fheave=$FHEAVE tRatio=$TRATIO"

source launchCommon.sh
