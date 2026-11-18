
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/policies"))

from dataset_libero import LIBERODataset
from lc_model import LanguageConditionedPolicy
from encode_text_embeddings import encode_instructions


DATASET = "data/libero/libero_object"

def get_instruction(task_name):
    """Convert LIBERO task filename into a natural-language instruction."""
    instruction = task_name.replace("_", " ")
    return instruction.capitalize() + "."


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/lc_baseline/best.pt",
    )

    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument(
        "--run-name",
        type=str,
        default="lc_baseline",
    )

    parser.add_argument("--dataset-dir", default=DATASET)
    args = parser.parse_args()

    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Evaluation device:", device)

    # Dataset paths
    dataset_dir = (
        args.dataset_dir
    )

    splits_path = "configs/libero_dataset_splits.json"

    # Load test dataset
    test_dataset = LIBERODataset(
        dataset_dir,
        splits_path,
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

    # Initialize model with matching architecture
    model = LanguageConditionedPolicy(
        robot_state_dim=9,
        language_dim=512,
        action_dim=7,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load robot-state normalization from checkpoint buffers.
    # These are already included in model.state_dict().

    # Get unique task names from the split manifest
    with open(splits_path, "r") as f:
        splits = json.load(f)

    task_names = sorted({
        demo_path.split("/", 1)[0]
        for split in ["train", "val", "test"]
        for demo_path in splits[split]
    })

    # Precompute CLIP embeddings for all task instructions
    print("Encoding task instructions with CLIP...")

    instructions = [
        get_instruction(task_name)
        for task_name in task_names
    ]

    embeddings = encode_instructions(instructions)

    language_embeddings = {
        task_name: embedding.cpu()
        for task_name, embedding in zip(task_names, embeddings)
    }

    print("Encoded", len(language_embeddings), "task instructions")

    criterion = nn.MSELoss(reduction="sum")

    # Evaluation metrics
    total_squared_error = 0.0
    total_elements = 0
    per_action_squared_error = torch.zeros(7)
    total_samples = 0

    # Evaluation loop
    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            images = batch["image"].to(device)
            states = batch["robot_state"].to(device)
            actions = batch["action"].to(device)

            # Retrieve CLIP embedding for each task
            language = torch.stack([
                language_embeddings[task_name]
                for task_name in batch["task_name"]
            ]).to(device)

            predictions = model(
                images,
                states,
                language,
            )

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
                    f"Evaluated {total_samples}/"
                    f"{len(test_dataset)} samples",
                    flush=True,
                )

    # Calculate metrics
    test_mse = total_squared_error / total_elements

    per_action_mse = (
        per_action_squared_error / total_samples
    ).tolist()

    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "test_samples": total_samples,
        "test_mse": test_mse,
        "per_action_mse": per_action_mse,
        "tasks": task_names,
    }

    results_dir = os.path.join(
        "results", args.run_name
    )

    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(
        results_dir, "test_results.json"
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    test_dataset.close()

    print("\nOffline evaluation complete!")
    print(f"Test MSE: {test_mse:.6f}")
    print("Per-action MSE:", per_action_mse)
    print("Saved:", results_path)


if __name__ == "__main__":
    main()
