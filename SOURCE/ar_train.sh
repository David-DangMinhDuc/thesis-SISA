#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
slices=$2
epochs=$3

for i in $(seq 0 "$((${shards}-1))"); do
    for j in {0..15}; do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/16"
        r=$((${j}*${shards}/16))
        python sisa.py --model ar_model --train --slices "${slices}" --dataset face_data/ar/ar_info --label "${r}" --epochs "${epochs}" --batch_size 16 --learning_rate 0.001 --optimizer adam --chkpt_interval 1 --container "${shards}" --shard "${i}" --stt "${j}"
    done
done
