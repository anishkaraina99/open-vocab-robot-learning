
import os
import json
import h5py
import numpy as np

DATASET = "data/processed/can_rgb.hdf5"
OUTPUT = "configs/dataset_splits.json"

SEED = 42

with h5py.File(DATASET, "r") as f:
    demos = sorted(
        f["data"].keys(),
        key=lambda x: int(x.split("_")[1])
    )

    print("Total demonstrations:", len(demos))

    assert len(demos) == 200, "Expected 200 demonstrations"

    rng = np.random.default_rng(SEED)
    rng.shuffle(demos)

    train = demos[:160]
    val = demos[160:180]
    test = demos[180:200]

    splits = {
        "seed": SEED,
        "train": train,
        "val": val,
        "test": test,
    }

    os.makedirs("configs", exist_ok=True)

    with open(OUTPUT, "w") as out:
        json.dump(splits, out, indent=2)

    for name in ["train", "val", "test"]:
        frames = sum(
            len(f["data"][d]["actions"])
            for d in splits[name]
        )
        print(f"{name}: {len(splits[name])} demos, {frames} frames")

print("Saved splits to:", OUTPUT)
