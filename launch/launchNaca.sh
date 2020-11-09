OPTIONS="-bpdx 48 -bpdy 24 -tdump 0.1 -nu 0.0000225 -tend 20 -poissonType cosine -iterativePensalization 0 -muteAll 1"
# COM OF NACA AIRFOIL IS 0.399421, ROTATION AROUND 0.1, thus fixedCenterDist=0.299412 -> CoR=0.1550882
OBJECTS='NACA L=0.15 xpos=0.2 fixedCenterDist=0.299412 bFixed=1 xvel=0.15 Apitch=0.15 Fpitch=1 tAccel=1
'

source launchCommon.sh
