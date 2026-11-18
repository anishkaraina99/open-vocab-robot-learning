
import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "src/policies")

from dataset import RobotDataset
from bc_model import BehaviorCloningPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/rgb_full/best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Evaluation device:", device)

    test_dataset = RobotDataset(
        "data/processed/can_rgb.hdf5",
        "configs/dataset_splits.json",
        split="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Test samples:", len(test_dataset))

    # Load checkpoint
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    # Avoid downloading pretrained weights again
    model = BehaviorCloningPolicy(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.MSELoss(reduction="sum")

    total_squared_error = 0.0
    total_elements = 0
    per_action_squared_error = torch.zeros(7)
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            states = batch["robot_state"].to(device)
            actions = batch["action"].to(device)

            predictions = model(images, states)

            total_squared_error += criterion(
                predictions, actions
            ).item()

            total_elements += actions.numel()

            per_action_squared_error += (
                (predictions - actions) ** 2
            ).sum(dim=0).cpu()

            total_samples += actions.shape[0]

            if (batch_idx + 1) % 25 == 0:
                print(
                    f"Evaluated {total_samples}/{len(test_dataset)} samples",
                    flush=True,
                )

    test_mse = total_squared_error / total_elements
    per_action_mse = (
        per_action_squared_error / total_samples
    ).tolist()

    results = {
        "checkpoint": args.checkpoint,
        "test_samples": total_samples,
        "test_mse": test_mse,
        "per_action_mse": per_action_mse,
    }

    os.makedirs("results/rgb_full", exist_ok=True)

    with open(
        "results/rgb_full/test_results.json", "w"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nOffline evaluation complete!")
    print(f"Test MSE: {test_mse:.6f}")
    print("Per-action MSE:", per_action_mse)
    print("Saved: results/rgb_full/test_results.json")


if __name__ == "__main__":
    main()
