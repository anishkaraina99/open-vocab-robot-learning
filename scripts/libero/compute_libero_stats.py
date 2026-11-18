
import os
import json
import h5py
import numpy as np

DATASET ="/Users/anabellafalk/cs6758/LIBERO/libero/datasets/libero_object"
SPLITS = "configs/libero_dataset_splits.json"
OUTPUT = "configs/libero_robot_state_stats.json"

with open(SPLITS, "r") as f:
    splits = json.load(f)

count = 0
total = np.zeros(9, dtype=np.float64)
total_sq = np.zeros(9, dtype=np.float64)

# Group training demonstrations by task
demos_by_task = {}

for demo_path in splits["train"]:
    task_name, demo_name = demo_path.split("/", 1)
    demos_by_task.setdefault(task_name, []).append(demo_name)

for task_name, demo_names in demos_by_task.items():
    dataset_path = os.path.join(
        DATASET,
        f"{task_name}_demo.hdf5"
    )

    with h5py.File(dataset_path, "r") as f:
        for demo_name in demo_names:
            states = f["data"][demo_name]["robot_states"][:]

            total += states.sum(axis=0)
            total_sq += (states ** 2).sum(axis=0)
            count += len(states)

mean = total / count
variance = np.maximum(total_sq / count - mean ** 2, 0)
std = np.sqrt(variance)

# Avoid dividing by extremely small values
std = np.maximum(std, 1e-3)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w") as f:
    json.dump(
        {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": count,
        },
        f,
        indent=2,
    )

print("Samples:", count)
print("Mean:", np.round(mean, 4))
print("Std:", np.round(std, 4))
print("Saved:", OUTPUT)
