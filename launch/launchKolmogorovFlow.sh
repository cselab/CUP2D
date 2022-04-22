# Defaults for Options
BPDX=${BPDX:-16}
BPDY=${BPDY:-16}
LEVELS=${LEVELS:-1}
RTOL=${RTOL:-0.1}
CTOL=${CTOL:-0.01}
EXTENT=${EXTENT:-6.2831853072}
CFL=${CFL:-0.15}
DT=${DT:-0}
PT=${PT:-1e-10}
PTR=${PTR:-0}
NU=${NU:-0.05}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 0 -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -dt $DT -tdump 1 -nu $NU -tend 0 -muteAll 0 -verbose 1 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1 -bForcing 1 -forcingCoefficient 4 -forcingWavenumber 4 -ic random -smagorinskyCoeff 0.0"

source launchCommon.sh
