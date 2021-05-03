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
XPOS=${XPOS:-1.2}
LENGTH=${LENGTH:-0.2}
PERIOD=${PERIOD:-1}

### "Scaling macroscopic aquatic locomotion" from M. Gazzola et al. ###
# Re=10 <-> NU=0.004; Re=100 <-> NU=0.0004; Re=158 <-> NU=0.000253165; Re=251 <-> NU=0.000159363; Re=398 <-> NU=0.000100503; Re=631 <-> NU=0.0000633914; Re=1000 <-> NU=0.00004; Re=1585 <-> NU=0.0000252366; Re=2512 <-> NU=0.0000159236; Re=3981 <-> NU=0.0000100477; Re=6310 <-> NU=0.00000633914; Re=10'000 <-> NU=0.000004; Re=100'000 <-> NU=0.0000004
####################################
### "Simulations of optimized anguilliform swimming" from S. Kern und P. Koumoutsakos
# NU=0.00014 for L=1 and T=1 (! adjust extent !)
# NU=0.0000056 for L=0.2 and T=1
####################################
NU=${NU:-0.0000056}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 4 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 50 -muteAll 0 -verbose 0 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 0"
OBJECTS="carlingfish L=$LENGTH T=$PERIOD xpos=$XPOS bFixed=1
"

source launchCommon.sh
