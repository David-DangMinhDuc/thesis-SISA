#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
    
if [[ ! -d "containers/${shards}" ]] ; then
    mkdir -p  "containers/${shards}"
    mkdir -p  "containers/${shards}/cache"
    mkdir -p  "containers/${shards}/times"
    mkdir -p  "containers/${shards}/outputs"
    echo 0 > "containers/${shards}/times/null.time"
fi

python distribution.py --shards "${shards}" --distribution uniform --container "${shards}" --dataset face_data/orl/orl_info --label 0

for j in {1..3}; do
    r=$((${j}*${shards}/4))
    python distribution.py --requests "${r}" --distribution uniform --container "${shards}" --dataset face_data/orl/orl_info --label "${r}"
done
