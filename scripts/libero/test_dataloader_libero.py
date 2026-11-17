import sys
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