import os
import json

import torch
import numpy as np
import h5py

from torch.utils.data import Dataset

try:
    from .libero_images import policy_image
except ImportError:  # Baseline scripts import this module directly.
    from libero_images import policy_image


class LIBERODataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        splits_path,
        split="train",
    ):
        """
        Args:
            dataset_dir: Directory containing task-specific HDF5 files.
            splits_path: Path to dataset_splits.json.
            split: "train", "val", or "test".
        """

        with open(splits_path, "r") as f:
            splits = json.load(f)

        self.dataset_dir = dataset_dir
        self.samples = []
        self._files = {}

        # Build frame-level index from demonstration-level splits.
        for demo_path in splits[split]:

            task_name, demo_name = demo_path.split("/", 1)

            hdf5_path = os.path.join(
                dataset_dir,
                task_name + "_demo.hdf5"
            )

            if not os.path.exists(hdf5_path):
                raise FileNotFoundError(hdf5_path)

            with h5py.File(hdf5_path, "r") as f:
                num_steps = len(
                    f["data"][demo_name]["actions"]
                )

            for t in range(num_steps):
                self.samples.append(
                    (task_name, demo_name, t)
                )

        print(
            f"{split}: {len(self.samples)} frames "
            f"from {len(splits[split])} demonstrations"
        )

    def __len__(self):
        return len(self.samples)

    def _get_file(self, task_name):
        """Open each task's HDF5 file lazily."""

        if task_name not in self._files:
            hdf5_path = os.path.join(
                self.dataset_dir,
                task_name + "_demo.hdf5"
            )

            self._files[task_name] = h5py.File(
                hdf5_path, "r"
            )

        return self._files[task_name]

    @staticmethod
    def _get_instruction(task_name):
        instruction = " ".join(task_name.split("_"))
        return instruction.capitalize() + "."

    def __getitem__(self, idx):

        task_name, demo_name, t = self.samples[idx]

        f = self._get_file(task_name)
        demo = f["data"][demo_name]
        obs = demo["obs"]

        # Third-person camera image: (H, W, C)
        image = np.array(obs["agentview_rgb"][t], copy=True)
        # Rotate 180 degrees
        image = policy_image(image)
        # Convert to PyTorch tensor: (C, H, W), normalized to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Precomputed robot state
        robot_state = torch.from_numpy(
            np.array(demo["robot_states"][t], copy=True)
        ).float()

        # Task instruction
        instruction = self._get_instruction(task_name)

        # Continuous action
        action = torch.from_numpy(
            np.array(demo["actions"][t], copy=True)
        ).float()

        return {
            "image": image,
            "robot_state": robot_state,
            "instruction": instruction,
            "action": action,
            "task_name": task_name,
            "demo_name": demo_name,
        }

    def close(self):
        """Close all open HDF5 files."""

        for f in self._files.values():
            if f.id.valid:
                f.close()

        self._files.clear()

    def __getstate__(self):
        """Avoid sharing HDF5 handles between DataLoader workers."""

        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    
