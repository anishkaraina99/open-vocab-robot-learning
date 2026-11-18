
import sys
import numpy as np
import torch
import h5py

sys.path.insert(0, "src/policies")
from bc_model import BehaviorCloningPolicy

DATASET = "data/processed/can_rgb.hdf5"
CHECKPOINT = "checkpoints/rgb_full/best.pt"

device = torch.device("cpu")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)

model = BehaviorCloningPolicy(pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with h5py.File(DATASET, "r") as f:
    demo = f["data/demo_0"]

    images = demo["images"][:]
    states = demo["robot_state"][:]
    actions = demo["actions"][:]

expert_gripper = actions[:, 6]
predicted_gripper = []

with torch.no_grad():
    for t in range(len(actions)):
        image = (
            torch.from_numpy(images[t].copy())
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0) / 255.0
        )

        state = (
            torch.from_numpy(states[t])
            .float()
            .unsqueeze(0)
        )

        prediction = model(image, state)
        predicted_gripper.append(
            prediction[0, 6].item()
        )

predicted_gripper = np.array(predicted_gripper)

print("\n=== EXPERT GRIPPER ===")
print("Min:", expert_gripper.min())
print("Max:", expert_gripper.max())
print("Mean:", expert_gripper.mean())
print("Unique values:", np.unique(expert_gripper))

print("\n=== PREDICTED GRIPPER ===")
print("Min:", predicted_gripper.min())
print("Max:", predicted_gripper.max())
print("Mean:", predicted_gripper.mean())

print("\n=== STEP-BY-STEP COMPARISON ===")
for t in range(0, len(actions), 10):
    print(
        f"Step {t:3d} | "
        f"Expert: {expert_gripper[t]:+.3f} | "
        f"Predicted: {predicted_gripper[t]:+.3f}"
    )
