#!/bin/bash
for (( k = 1; k < 201; ++k )); do
  a=$(( 100*k))
  nu=$(echo 0.04 \/ $a |bc -l)
  re=$(printf "%05d" $a)
  echo $re
  ./launchDiskAMR.sh Re$re $nu
done
