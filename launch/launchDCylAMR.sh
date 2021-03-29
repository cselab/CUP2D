BPDX=${BPDX:-8}
BPDY=${BPDY:-4}
LEVELS=${LEVELS:-7}
RTOL=${RTOL-0.50}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-4.0}
CFL=${CFL:-0.2}
XPOS=${XPOS:-2.0}
ANGLE=${ANGLE:-0}
XVEL=${XVEL:-0.2}
RADIUS=${RADIUS:-0.1}
#NU=${NU:-0.00000421052} #Re9500
#NU=${NU:-0.00001333333} #Re=3000
#NU=${NU:-0.00007272727} #Re=550
NU=${NU:-0.001} #Re=40

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 3 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 1.0 -nu $NU -tend 15. -verbose 1 -lambda 1e6 "
OBJECTS="disk radius=$RADIUS angle=$ANGLE xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0"
source launchCommon.sh
