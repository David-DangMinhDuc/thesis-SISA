#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
slices=$2
epochs=$3

for i in $(seq 0 "$((${shards}-1))"); do
    for j in {0..3}; do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/4"
        r=$((${j}*${shards}))
        python sisa.py --model orl_model --train --slices "${slices}" --dataset face_data/orl/orl_info --label "${r}" --epochs "${epochs}" --batch_size 4 --learning_rate 0.0001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}" --stt "${j}"
    done
done
