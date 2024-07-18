#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
slices=$2
epochs=$3

if [[ ! -d "results" ]] ; then
    mkdir -p  "results"
    mkdir -p  "results/orl"
    mkdir -p  "results/ar"
fi

if [[ ! -f general-report.csv ]]; then
    echo "nb_shards,nb_slices,nb_epochs,nb_requests,accuracy,retraining_time,nb_retrained_points" > results/orl/report_a_part_of_"${shards}"_shards_"${slices}"_slices_"${epochs}"_epochs.csv
fi

for j in {0..3}; do
    r=$((${j}*${shards}))
    shard_idx=0
    acc=$(python aggregation.py --strategy uniform --container "${shards}" --shards "${shards}" --dataset face_data/orl/orl_info --baseline "${shard_idx}" --label "${r}")
    cat containers/"${shards}"/times/shard-*:"${r}".time > "containers/${shards}/times/times.tmp"
    time=$(python time.py --container "${shards}" | awk -F ',' '{print $1}')
    cat containers/"${shards}"/shard-*:"${r}"_"${j}".txt > "containers/${shards}/numOfRetrainPoints.tmp"
    numOfRetrainPoints=$(python numOfRetrainPoints.py --container "${shards}")
    echo "${shards},${slices},${epochs},${r},${acc},${time},${numOfRetrainPoints}" >> results/orl/report_a_part_of_"${shards}"_shards_"${slices}"_slices_"${epochs}"_epochs.csv
done
