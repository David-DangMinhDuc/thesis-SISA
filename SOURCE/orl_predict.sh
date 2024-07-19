#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

for i in $(seq 0 "$((${shards}-1))"); do
    for j in {0..3}; do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/4"
        r=$((${j}*${shards}))
        python sisa.py --model orl_model --test --dataset face_data/orl/orl_info --label "${j}" --batch_size 4 --container "${shards}" --shard "${i}"
    done
done
