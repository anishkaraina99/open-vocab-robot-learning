
import os
import json
import h5py
import numpy as np
from collections import defaultdict

DATASET_DIR = (
    "/Users/anabellafalk/cs6758/LIBERO/libero/datasets/libero_object"
)
OUTPUT = "configs/libero_dataset_splits.json"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


rng = np.random.default_rng(SEED)

splits = {
    "seed": SEED,
    "train": [],
    "val": [],
    "test": [],
    "task_counts": {}
}

total_demos = 0

# Each HDF5 file corresponds to one task
dataset_files = sorted(
    filename
    for filename in os.listdir(DATASET_DIR)
    if filename.endswith("_demo.hdf5")
)

print("Number of task files:", len(dataset_files))

for filename in dataset_files:

    task_name = filename.removesuffix("_demo.hdf5")
    filepath = os.path.join(DATASET_DIR, filename)

    with h5py.File(filepath, "r") as f:

        demos = sorted(
            f["data"].keys(),
            key=lambda x: int(x.split("_")[1])
        )

        n = len(demos)
        total_demos += n

        # Shuffle demonstrations within this task
        demos = np.array(demos)
        rng.shuffle(demos)

        # Calculate split sizes
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        train_demos = demos[:n_train]
        val_demos = demos[n_train:n_train + n_val]
        test_demos = demos[n_train + n_val:]

        # Store task and demonstration name together
        splits["train"].extend(
            [f"{task_name}/{demo}" for demo in train_demos]
        )
        splits["val"].extend(
            [f"{task_name}/{demo}" for demo in val_demos]
        )
        splits["test"].extend(
            [f"{task_name}/{demo}" for demo in test_demos]
        )

        splits["task_counts"][task_name] = {
            "train": len(train_demos),
            "val": len(val_demos),
            "test": len(test_demos),
            "total": n
        }

        print(
            f"{task_name}: {n} demos | "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

# Shuffle combined splits
for split in ["train", "val", "test"]:
    rng.shuffle(splits[split])

# Save split manifest
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w") as out:
    json.dump(splits, out, indent=2)

# Print overall statistics
print("\nTotal demonstrations:", total_demos)

for split in ["train", "val", "test"]:

    total_frames = 0

    # Count frames across all task files
    for demo_path in splits[split]:
        task_name, demo_name = demo_path.split("/", 1)

        filepath = os.path.join(
            DATASET_DIR,
            task_name + "_demo.hdf5"
        )

        with h5py.File(filepath, "r") as f:
            total_frames += len(f["data"][demo_name]["actions"])

    print(
        f"{split}: {len(splits[split])} demos, "
        f"{total_frames} frames"
    )

print("\nSaved splits to:", OUTPUT)
