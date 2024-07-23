import numpy as np
import json
import os
import importlib

import argparse

"""
Mục đích của tập tin này là thu thập lại tham số đầu ra sau quá trình thử nghiệm mô hình trên tập dữ liệu thử nghiệm, sau đó chuyển các tham số đó thành các véc-tơ dự đoán và tính trung bình toàn bộ các véc-tơ dự đoán đó. Cuối cùng, chúng tôi chọn ra nhãn (đối tượng) có giá trị trung bình lớn nhất và dựa vào đó tính độ chính xác tổng thể của phương pháp đề xuất
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--strategy", default="uniform", help="Voting strategy, default uniform"
)
parser.add_argument("--container", help="Name of the container")
parser.add_argument("--shards", type=int, default=1, help="Number of shards, default 1")
parser.add_argument(
    "--dataset",
    default="face_data/orl/orl_info",
    help="Location of the facial dataset's infomation file, default face_data/orl/orl_info",
)
parser.add_argument(
    "--baseline", type=int, help="Use only the specified shard (lone shard baseline)"
)
parser.add_argument("--label", default="latest", help="Label, default latest")
args = parser.parse_args()

# Load dataset metadata.
with open(args.dataset) as f:
    face_data_info = json.loads(f.read())
dataloader = importlib.import_module(
    ".".join(args.dataset.split("/")[:-1] + [face_data_info["dataloader"]])
)

# Output files used for the vote.
if args.baseline != None:
    filenames = ["shard-{}:{}.npy".format(args.baseline, args.label)]
else:
    filenames = ["shard-{}:{}.npy".format(i, args.label) for i in range(args.shards)]

# Concatenate output files.
outputs = []
for filename in filenames:
    outputs.append(
        np.load(
            os.path.join("containers/{}/outputs".format(args.container), filename),
            allow_pickle=True,
        )
    )
outputs = np.array(outputs)
#print("Outputs: ")
#print(outputs)
#print()

# Compute weight vector based on given strategy.
if args.strategy == "uniform":
    weights = (
        1 / outputs.shape[0] * np.ones((outputs.shape[0],))
    )  # pylint: disable=unsubscriptable-object
    #print("Weights: ")
    #print(weights)
    #print()
elif args.strategy.startswith("models:"):
    models = np.array(args.strategy.split(":")[1].split(",")).astype(int)
    weights = np.zeros((outputs.shape[0],))  # pylint: disable=unsubscriptable-object
    weights[models] = 1 / models.shape[0]  # pylint: disable=unsubscriptable-object
elif args.strategy == "proportional":
    split = np.load(
        "containers/{}/splitfile.npy".format(args.container), allow_pickle=True
    )
    weights = np.array([shard.shape[0] for shard in split])

# Tensor contraction of outputs and weights (on the shard dimension).
votes = np.argmax(
    np.tensordot(weights.reshape(1, weights.shape[0]), outputs, axes=1), axis=2
).reshape(
    (outputs.shape[1],)
)  # pylint: disable=unsubscriptable-object
#print("Votes: ")
#print(votes)
#print()
# Load labels.
_, labels = dataloader.load(np.arange(face_data_info["nb_test"]), method="test")

# Compute and print accuracy.
accuracy = (
    np.where(votes == labels)[0].shape[0] / outputs.shape[1]
)  # pylint: disable=unsubscriptable-object
print(accuracy)
