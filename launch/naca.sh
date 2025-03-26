"${main=./main}" \
    -bpdx 16 \
    -bpdy 8 \
    -CFL 0.45 \
    -Ctol 0.01 \
    -extent 4 \
    -levelMax 6 \
    -levelStart 4 \
    -nu 0.00001 \
    -poissonSolver iterative \
    -poissonTol 1e-10 \
    -poissonTolRel 0 \
    -Rtol 5 \
    -tdump 0.1 \
    -tend 10. \
    -verbose 0 \
    -shapes '
NACA L=0.20 xpos=0.60 ypos=1.00 fixedCenterDist=0.0 bFixed=1 xvel=0.15 yvel=0.0  Apitch=0.0 Fpitch=0.0 Mpitch=0.0 Aheave=0.0 Fheave=0.0 tRatio=1.00
'
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
