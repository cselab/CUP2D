# Defaults for Options
PSOLVER="iterative"
OPTIONS="-BC_y wall -bpdx 8 -bpdy 4 -levelMax 9 -levelStart 4 -Rtol 2.0 -Ctol 0.2 -extent 13.5 -CFL 0.5 -tdump 0.1 -nu 0.00003636363 -tend 200. -muteAll 0 -verbose 1 -poissonTol 0.0 -poissonTolRel 1e-4 -bAdaptChiGradient 1 -poissonSolver $PSOLVER -bMeanConstraint 2"

#OBJECTS="disk radius=0.1 xpos=5.0 ypos=3.375 bForced=1 bFixed=1 xvel=0.2 breakSymmetryExponent=1.0 breakSymmetryStrength=0.5 breakSymmetryType=1 breakSymmetryTime=3.0
#        "

#OBJECTS="disk radius=0.1 xpos=5.0 ypos=3.375 bForced=1 bFixed=1 xvel=0.2 breakSymmetryExponent=1.0 breakSymmetryStrength=0.5 breakSymmetryType=1 breakSymmetryTime=3.0
#         disk radius=0.1 xpos=5.5 ypos=3.54166666667 bForced=1 bFixed=1 xvel=0.2
#         disk radius=0.1 xpos=5.5 ypos=3.20833333333 bForced=1 bFixed=1 xvel=0.2
#        "

#OBJECTS="disk radius=0.1 xpos=5.0 ypos=3.375 bForced=1 bFixed=1 xvel=0.2 breakSymmetryExponent=1.0 breakSymmetryStrength=0.5 breakSymmetryType=1 breakSymmetryTime=3.0
#         disk radius=0.1 xpos=5.5 ypos=3.54166666667 bForced=1 bFixed=1 xvel=0.2
#         disk radius=0.1 xpos=5.5 ypos=3.20833333333 bForced=1 bFixed=1 xvel=0.2
#         disk radius=0.1 xpos=6.0 ypos=3.70833333333 bForced=1 bFixed=1 xvel=0.2
#         disk radius=0.1 xpos=6.0 ypos=3.04166666667 bForced=1 bFixed=1 xvel=0.2
#        "

OBJECTS="disk radius=0.1 xpos=5.0 ypos=3.375 bForced=1 bFixed=1 xvel=0.2 breakSymmetryExponent=1.0 breakSymmetryStrength=0.5 breakSymmetryType=1 breakSymmetryTime=3.0
         disk radius=0.1 xpos=5.5 ypos=3.54166666667 bForced=1 bFixed=1 xvel=0.2
         disk radius=0.1 xpos=5.5 ypos=3.20833333333 bForced=1 bFixed=1 xvel=0.2
         disk radius=0.1 xpos=6.0 ypos=3.70833333333 bForced=1 bFixed=1 xvel=0.2
         disk radius=0.1 xpos=6.0 ypos=3.04166666667 bForced=1 bFixed=1 xvel=0.2
         disk radius=0.1 xpos=6.5 ypos=3.875 bForced=1 bFixed=1 xvel=0.2
         disk radius=0.1 xpos=6.5 ypos=2.875 bForced=1 bFixed=1 xvel=0.2
        "

source launchCommon.sh
