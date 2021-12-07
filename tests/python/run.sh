#!/bin/bash

mkdir -p output/h5
python3 -m unittest "$@"
