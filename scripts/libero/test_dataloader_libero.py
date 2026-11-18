import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, "src/policies")

from dataset_libero import LIBERODataset

DATASET_DIR = (
    "/Users/anabellafalk/cs6758/LIBERO/libero/datasets/libero_object"
)

SPLITS_PATH = "configs/libero_dataset_splits.json"

train_dataset = LIBERODataset(
    dataset_dir=DATASET_DIR,
    splits_path=SPLITS_PATH,
    split="train",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(train_loader))

print("Image:", batch["image"].shape)
print("Robot state:", batch["robot_state"].shape)
print("Action:", batch["action"].shape)
print("Instructions:", batch["instruction"][:3])


# Inspect the first image in the batch
image = batch["image"][0]

plt.figure(figsize=(8, 6))
plt.imshow(image.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("LIBERO Agent View")
plt.show()