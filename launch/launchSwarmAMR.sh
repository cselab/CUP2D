# Defaults for Options
BPDX=${BPDX:-4}
BPDY=${BPDY:-3}
LEVELS=${LEVELS:-8}
RTOL=${RTOL-1}
CTOL=${CTOL-0.1}
EXTENT=${EXTENT:-2}
CFL=${CFL:-0.2}
PT=${PT:-1e-6}
PTR=${PTR:-1e-4}

# Defaults for fish
LENGTH=${LENGTH:-0.1}
PERIOD=${PERIOD:-1}
PID=${PID:-0}
PIDPOS=${PIDPOS:-1}

# stefanfish Re=1'000 <-> NU=0.00001
NU=${NU:-0.00001}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 0 -tdump 0.1 -nu $NU -tend 100 -muteAll 0 -verbose 1"

OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.2 ypos=0.25 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.3 ypos=0.20 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.3 ypos=0.30 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.4 ypos=0.25 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.4 ypos=0.35 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.4 ypos=0.15 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.5 ypos=0.20 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.5 ypos=0.30 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.6 ypos=0.25 bFixed=1 pidpos=$PIDPOS pid=$PID
"

source launchCommon.sh
