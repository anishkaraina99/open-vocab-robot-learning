
import json
import h5py
import numpy as np

DATASET = "data/processed/can_rgb.hdf5"
SPLITS = "configs/dataset_splits.json"
OUTPUT = "configs/robot_state_stats.json"

with open(SPLITS, "r") as f:
    splits = json.load(f)

count = 0
total = np.zeros(16, dtype=np.float64)
total_sq = np.zeros(16, dtype=np.float64)

with h5py.File(DATASET, "r") as f:
    for demo_name in splits["train"]:
        states = f["data"][demo_name]["robot_state"][:]

        total += states.sum(axis=0)
        total_sq += (states ** 2).sum(axis=0)
        count += len(states)

mean = total / count
variance = np.maximum(total_sq / count - mean ** 2, 0)
std = np.sqrt(variance)

# Avoid dividing by extremely small values
std = np.maximum(std, 1e-3)

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
