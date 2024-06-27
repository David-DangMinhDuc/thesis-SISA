#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

if [[ ! -f general-report.csv ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time,nb_retrained_points" > general-report.csv
fi

for j in {0..15}; do
    r=$((${j}*${shards}/5))
    acc=$(python aggregation.py --strategy uniform --container "${shards}" --shards "${shards}" --dataset face_data/orl/orl_info --label "${r}")
    cat containers/"${shards}"/times/shard-*:"${r}".time > "containers/${shards}/times/times.tmp"
    time=$(python time.py --container "${shards}" | awk -F ',' '{print $1}')
    cat containers/"${shards}"/shard-*:"${r}".txt > "containers/${shards}/numOfRetrainPoints.tmp"
    numOfRetrainPoints=$(python numOfRetrainPoints.py --container "${shards}")
    echo "${shards},${r},${acc},${time},${numOfRetrainPoints}" >> general-report.csv
done
