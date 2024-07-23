import numpy as np
import json
import os

import argparse

"""
Tập tin này thể hiện quá trình phân chia dữ liệu thành s phân đoạn và khởi tạo các yêu cầu loại bỏ một cách ngẫu nhiên 
dựa trên phân phối xác suất cụ thể. 
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--shards",
    default=None,
    type=int,
    help="Split the dataset in the given number of shards in an optimized manner (PLS-GAP partitionning) according to the given distribution, create the corresponding splitfile",
)
parser.add_argument(
    "--requests",
    default=None,
    type=int,
    help="Generate the given number of unlearning requests according to the given distribution and apply them directly to the splitfile",
)
parser.add_argument(
    "--distribution",
    default="uniform",
    help="Assumed distribution when used with --shards, sampling distribution when used with --requests. Use 'reset' to reset requestfile, default uniform",
)
parser.add_argument("--container", default="default", help="Name of the container")
parser.add_argument(
    "--dataset",
    default="face_data/orl/orl_info",
    help="Location of the facial dataset's infomation file, default datasets/purchase/face_data/orl/orl_info",
)
parser.add_argument("--label", default="latest", help="Label, default latest")
args = parser.parse_args()

# Load dataset metadata.
with open(args.dataset) as f:
    face_data_info = json.loads(f.read())

if args.shards != None:
    # If distribution is uniform, split without optimizing.
    if args.distribution == "uniform":
        partition = np.split(
            np.arange(0, face_data_info["nb_train"]),
            [
                t * (face_data_info["nb_train"] // args.shards)
                for t in range(1, args.shards)
            ],
        )
        
        np.save("containers/{}/splitfile.npy".format(args.container), np.array(partition, dtype=object), allow_pickle=True)
        requests = np.array([[] for _ in range(args.shards)])

        np.save(
            "containers/{}/requestfile:{}.npy".format(args.container, args.label),
            requests,
        )

    # Else run PLS-GAP algorithm to find a low cost split.
    else:

        def mass(index):
            if args.distribution.split(":")[0] == "exponential":
                lbd = (
                    float(args.distribution.split(":")[1])
                    if len(args.distribution.split(":")) > 1
                    else -np.log(0.05) / face_data_info["nb_train"]
                )
                return np.exp(-lbd * index) - np.exp(-lbd * (index + 1))
            if args.distribution.split(":")[0] == "pareto":
                a = (
                    float(args.distribution.split(":")[1])
                    if len(args.distribution.split(":")) > 1
                    else 1.16
                )
                return a / ((index + 1) ** (a + 1))

        if args.shards != None:
            # Initialize queue and partition.
            weights = mass(np.arange(0, face_data_info["nb_train"]))
            indices = np.argsort(weights)
            queue = np.array([weights[indices], np.ones(weights.shape)]).transpose()
            partition = [np.array([index]) for index in indices]

            # Put all points in the top queue.
            bottom_queue = queue.shape[0]  # pylint: disable=unsubscriptable-object
            lim = (
                int(float(args.algo.split(":")[1]) * face_data_info["nb_train"])
                if len(args.algo.split(":")) > 1
                else int(0.01 * face_data_info["nb_train"])
            )

            for _ in range(face_data_info["nb_train"] - args.shards):
                # Fetch top 2 clusters and merge them.
                w1 = queue[0]
                w2 = queue[1]

                l1 = partition[0]
                l2 = partition[1]

                partition = partition[2:]
                queue = queue[2:]
                bottom_queue -= 2

                merged_weight = w1 + w2

                # If merged cluster is smaller in number of points than the limit, insert it in top queue.
                if merged_weight[1] < lim:
                    # Top queue is ordered first by number of points (weight[1]) and second by cost (weight[0]).
                    offset_array = np.where(queue[:bottom_queue, 1] >= merged_weight[1])
                    limit_array = np.where(queue[:bottom_queue, 1] > merged_weight[1])
                    offset = (
                        offset_array[0][0]
                        if offset_array[0].shape[0] > 0
                        else bottom_queue
                    )
                    limit = (
                        limit_array[0][0]
                        if limit_array[0].shape[0] > 0
                        else bottom_queue
                    )
                    position_array = np.where(
                        queue[offset:limit][:, 0] >= merged_weight[0]
                    )
                    position = (
                        position_array[0][0]
                        if position_array[0].shape[0] > 0
                        else bottom_queue
                    )
                    bottom_queue += 1

                # Otherwise insert it in the bottom queue.
                else:
                    # Bottom queue is ordered by cost only.
                    position_array = np.where(
                        queue[bottom_queue:][:, 0] >= merged_weight[0]
                    )
                    position = (
                        position_array[0][0]
                        if position_array[0].shape[0] > 0
                        else queue.shape[0]
                    )

                # Actual insertion.
                queue = np.insert(queue, position, merged_weight, axis=0)
                partition = (
                    partition[:position]
                    + [np.concatenate((l1, l2))]
                    + partition[position:]
                )

            # Generate splitfile and empty request file.
            np.save("containers/{}/splitfile.npy".format(args.container), np.array(partition, dtype=object), allow_pickle=True)
            requests = np.array([[] for _ in range(partition.shape[0])])
            np.save(
                "containers/{}/requestfile:{}.npy".format(args.container, args.label),
                requests,
            )

if args.requests != None:
    if args.distribution == "reset":
        requests = np.array([[] for _ in range(partition.shape[0])])
        np.save(
            "containers/{}/requestfile:{}.npy".format(args.container, args.label),
            requests,
        )
    else:
        # Load splitfile.
        partition = np.load(
            "containers/{}/splitfile.npy".format(args.container), allow_pickle=True
        )

        # Randomly select points to be removed with given distribution at the dataset scale.
        if args.distribution.split(":")[0] == "exponential":
            lbd = (
                float(args.distribution.split(":")[1])
                if len(args.distribution.split(":")) > 1
                else -np.log(0.05) / face_data_info["nb_train"]
            )
            all_requests = np.random.exponential(1 / lbd, (args.requests,))
        if args.distribution.split(":")[0] == "pareto":
            a = (
                float(args.distribution.split(":")[1])
                if len(args.distribution.split(":")) > 1
                else 1.16
            )
            all_requests = np.random.pareto(a, (args.requests,))
        else:
            all_requests = np.random.randint(0, face_data_info["nb_train"], args.requests)
        
        requests = []
        # Divide up the new requests among the shards.
        for shard in range(partition.shape[0]):
            requests.append(np.intersect1d(partition[shard], all_requests))

        # Update requestfile.
        np.save(
            "containers/{}/requestfile:{}.npy".format(args.container, args.label),
            np.array(requests, dtype=object),
        )
