
import os
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.policies.sam_model import SAMBehaviorCloningPolicy
from src.policies.sam_dataset import SAMRobotDataset

from src.policies.dataset import RobotDataset
from src.policies.bc_model import BehaviorCloningPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="rgb_baseline")
    parser.add_argument("--dataset", default="data/processed/can_rgb.hdf5")
    parser.add_argument("--splits", default="configs/dataset_splits.json")
    parser.add_argument("--stats", default="configs/robot_state_stats.json")
    parser.add_argument("--mask-cache", help="Use the SAM policy with a complete mask cache")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline smoke tests only")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("epochs, batch-size, and max-batches must be positive")

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print("Training device:", device)

    # Output directories
    checkpoint_dir = os.path.join("checkpoints", args.run_name)
    results_dir = os.path.join("results", args.run_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load datasets
    dataset_cls = SAMRobotDataset if args.mask_cache else RobotDataset
    dataset_kwargs = {"cache_path": args.mask_cache} if args.mask_cache else {}
    train_dataset = dataset_cls(
        args.dataset,
        args.splits,
        **dataset_kwargs,
        split="train",
    )

    val_dataset = dataset_cls(
        args.dataset,
        args.splits,
        **dataset_kwargs,
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
    with open(args.stats, "r") as f:
        stats = json.load(f)

    # Initialize pretrained model
    model_cls = SAMBehaviorCloningPolicy if args.mask_cache else BehaviorCloningPolicy
    example = train_dataset[0]
    train_dataset.close()  # Workers must open their own HDF5 handles.
    model = model_cls(robot_state_dim=example["robot_state"].numel(),
                      action_dim=example["action"].numel(),
                      pretrained=not args.no_pretrained).to(device)

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
        "pretrained": not args.no_pretrained,
        "policy_type": "sam" if args.mask_cache else "rgb",
        "dataset": args.dataset,
        "splits": args.splits,
        "stats": args.stats,
        "mask_cache": args.mask_cache,
        "language_dim": 0,
        "robot_state_dim": example["robot_state"].numel(),
        "action_dim": example["action"].numel(),
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

            predicted = (
                model(images, states, batch["masks"].to(device))
                if args.mask_cache else model(images, states)
            )
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

                predicted = (
                    model(images, states, batch["masks"].to(device))
                    if args.mask_cache else model(images, states)
                )
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
