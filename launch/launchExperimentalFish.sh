# Defaults for Options
BPDX=${BPDX:-16}
BPDY=${BPDY:-8}
LEVELS=${LEVELS:-5}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.2}
PT=${PT:-1e-6}
PTR=${PTR:-1e-4}
# Defaults for Objects
XPOS=${XPOS:-2}
YPOS=${YPOS:-1}
LENGTH=${LENGTH:-0.2}
TIMESTART=${TIMESTART:-0.0}
DTDATA=${DTDATA:-0.1}
DATAPATH=${DATAPATH:-"/users/pweber/korali/examples/study.cases/CUP2D/"}

NU=${NU:-0.00004}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 0.5 -muteAll 0 -verbose 1 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 0"
OBJECTS="experimentFish L=$LENGTH xpos=$XPOS ypos=$YPOS timeStart=$TIMESTART dtDataset=$DTDATA path="${DATAPATH}" bFixed=0
"

source launchCommon.sh
