##############
# D-CYLINDER #
##############

### CONVERGENCE ANALYSIS ###

# LAUNCH RUNS FOR GRIDSIZE
for i in 8 16 32 64 128
do
export BPDX=$i; export BPDY=$(($i / 2)); export EXTENT=1; export CFL=0.1; export ANGLE=30; export XPOS=0.2; ./launchCyl.sh Cyl-bpdx=$i
done

# LAUNCH RUNS FOR DOMAINSIZE (at same resolution)
for i in 1 2 3
do
export BPDX=$(( $i * 48)); export BPDY=$(( $i * 24)); export EXTENT=$i; export CFL=0.1; export ANGLE=30; export XPOS=$(( $i / 5 )); ./launchCyl.sh Cyl-extent=$i
done

# LAUNCH RUNS FOR CFL
for i in 0.01 0.05 0.1 0.15 0.2 0.25 0.3
do
export BPDX=48; export BPDY=24; export EXTENT=1; export CFL=$i; export ANGLE=30; export XVEL=0.15; ./launchCyl.sh Cyl-cfl=$i
done

### PHYSICAL PARAMETERS ###

# LAUNCH RUNS FOR ANGLES
for i in {00..30}
do
export BPDX=48; export BPDY=24; export EXTENT=1; export CFL=0.1; export ANGLE=$i; export XVEL=0.15; ./launchCyl.sh Cyl-theta=$i
done

########
# NACA #
########

### CONVERGENCE ANALYSIS ###

# LAUNCH RUNS FOR GRIDSIZE
for i in 8 16 32 64 128
do
export BPDX=$i; export BPDY=$(($i / 2)); export EXTENT=1; export CFL=0.1; export ANGLE=14; export FPITCH=0; export XPOS=0.2; ./launchNaca.sh NACA-bpdx=$i
done

# LAUNCH RUNS FOR DOMAINSIZE (at same resolution)
for i in 1 2 3
do
export BPDX=$(( $i * 48)); export BPDY=$(( $i * 24)); export EXTENT=$i; export CFL=0.1; export ANGLE=14; export FPITCH=0; export XPOS=$(( $i / 5 )); ./launchNaca.sh NACA-extent=$i
done

# LAUNCH RUNS FOR CFL
for i in 0.01 0.05 0.1 0.15 0.2 0.25 0.3
do
export BPDX=48; export BPDY=24; export EXTENT=1; export CFL=$i; export ANGLE=14; export FPITCH=0; export XPOS=0.2; ./launchNaca.sh NACA-cfl=$i
done

### PHYSICAL PARAMETERS ###

# LAUNCH RUNS FOR ANGLES
for i in {00..30}
do
export BPDX=48; export BPDY=24; export EXTENT=1; export CFL=0.1; export ANGLE=$i; export FPITCH=0; export XPOS=0.2; ./launchNaca.sh NACA-theta=$i
done

# LAUNCH RUNS FOR FREQUENCY (~St for fixed Amplitude)
for i in 1.25 1.43 1.67 2.00 2.5
do
export BPDX=48; export BPDY=24; export EXTENT=1; export CFL=0.1; export ANGLE=0; export FPITCH=$i; export XPOS=0.2; ./launchNaca.sh NACA-freq=$i

done

##############
# STEFANFISH #
##############

### CONVERGENCE ANALYSIS ###

# LAUNCH RUNS FOR GRIDSIZE
for i in 8 16 32 64 128
do
export BPDX=$i; export BPDY=$(($i / 2)); export EXTENT=1; export CFL=0.1; export XPOS=0.2; export PERIOD=1; ./launchStefanFish.sh StefanFish-bpdx=$i
done

# LAUNCH RUNS FOR DOMAINSIZE
for i in 1 2 3
do
export BPDX=$(( $i * 48)); export BPDY=$(( $i * 24)); export EXTENT=$i; export CFL=0.1; export XPOS=$(( $i / 5 ));  export PERIOD=1; ./launchStefanFish.sh StefanFish-extent=$i
done

# LAUNCH RUNS FOR CFL
for i in 0.01 0.05 0.1 0.15 0.2 0.25 0.3
do
export BPDX=48; export BPDY=24; export EXTENT=1; export CFL=$i; export XPOS=0.2; export PERIOD=1; ./launchStefanFish.sh StefanFish-cfl=$i
done

### PHYSICAL PARAMETERS ###

# LAUNCH RUNS FOR SWIMMING PERIOD
for i in 
do
export BPDX=48; export BPDY=24; export EXTENT=1; export CFL=$i; export XPOS=0.2; export PERIOD=$i; ./launchStefanFish.sh StefanFish-cfl=$i
done
