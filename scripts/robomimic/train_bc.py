
import os
import sys
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "src/policies")

from dataset import RobotDataset
from bc_model import BehaviorCloningPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="rgb_baseline")
    args = parser.parse_args()

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Training device:", device)

    # Output directories
    checkpoint_dir = os.path.join("checkpoints", args.run_name)
    results_dir = os.path.join("results", args.run_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load datasets
    train_dataset = RobotDataset(
        "data/processed/can_rgb.hdf5",
        "configs/dataset_splits.json",
        split="train",
    )

    val_dataset = RobotDataset(
        "data/processed/can_rgb.hdf5",
        "configs/dataset_splits.json",
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # Load normalization statistics
    with open("configs/robot_state_stats.json", "r") as f:
        stats = json.load(f)

    # Initialize pretrained model
    model = BehaviorCloningPolicy(pretrained=True).to(device)

    model.set_state_normalization(
        stats["mean"],
        stats["std"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.MSELoss()

    # Save experiment configuration
    config = {
        "run_name": args.run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": seed,
        "device": str(device),
        "pretrained": True,
        "max_batches": args.max_batches,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    with open(
        os.path.join(results_dir, "config.json"), "w"
    ) as f:
        json.dump(config, f, indent=2)

    history = []
    best_val = float("inf")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if (
                args.max_batches is not None
                and batch_idx >= args.max_batches
            ):
                break

            images = batch["image"].to(device)
            states = batch["robot_state"].to(device)
            actions = batch["action"].to(device)

            predicted = model(images, states)
            loss = criterion(predicted, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = images.shape[0]
            train_loss += loss.item() * n
            train_count += n

        avg_train = train_loss / train_count

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if (
                    args.max_batches is not None
                    and batch_idx >= args.max_batches
                ):
                    break

                images = batch["image"].to(device)
                states = batch["robot_state"].to(device)
                actions = batch["action"].to(device)

                predicted = model(images, states)
                loss = criterion(predicted, actions)

                n = images.shape[0]
                val_loss += loss.item() * n
                val_count += n

        avg_val = val_loss / val_count

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train MSE: {avg_train:.6f} | "
            f"Val MSE: {avg_val:.6f}",
            flush=True,
        )

        # Record results
        history.append({
            "epoch": epoch + 1,
            "train_mse": avg_train,
            "val_mse": avg_val,
        })

        # Save best model
        if avg_val < best_val:
            best_val = avg_val

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_mse": avg_val,
                    "config": config,
                },
                os.path.join(checkpoint_dir, "best.pt"),
            )

            print(
                f"  Saved new best checkpoint "
                f"(Val MSE: {best_val:.6f})",
                flush=True,
            )

        # Save history after each epoch
        with open(
            os.path.join(results_dir, "history.json"), "w"
        ) as f:
            json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print("Best validation MSE:", best_val)
    print(
        "Best checkpoint:",
        os.path.join(checkpoint_dir, "best.pt"),
    )
    print(
        "Training history:",
        os.path.join(results_dir, "history.json"),
    )


if __name__ == "__main__":
    main()
