#!/bin/bash

#for (( k = 1; k < 201; ++k )); do
#  a=$(( 100*k))
#  nu=$(echo 0.04 \/ $a |bc -l)
#  re=$(printf "%05d" $a)
#  echo $re
#  ./launchDiskAMR.sh Re$re $nu
#done


#Re=20,000
export PT=1e-9
export PTR=1e-5
export CFL=0.1
export NU=0.000002
for (( k = 8; k < 12; ++k )); do
  export LEVELS=$k 
  ./launchDiskMultiple.sh Re020000_$k
done

#Re=50,000
export PT=1e-9
export PTR=1e-5
export CFL=0.075
export NU=0.0000008
for (( k = 8; k < 12; ++k )); do
  export LEVELS=$k 
  ./launchDiskMultiple.sh Re050000_$k
done

#Re=100,000
export PT=1e-10
export PTR=1e-6
export CFL=0.05
export NU=0.0000004
for (( k = 9; k < 13; ++k )); do
  export LEVELS=$k 
  ./launchDiskMultiple.sh Re100000_$k
done
