
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class RobotDataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        splits_path,
        split="train",
    ):
        with open(splits_path, "r") as f:
            splits = json.load(f)

        self.hdf5_path = hdf5_path
        self.samples = []
        self._file = None

        with h5py.File(hdf5_path, "r") as f:
            for demo_name in splits[split]:
                num_steps = len(f["data"][demo_name]["actions"])

                for t in range(num_steps):
                    self.samples.append((demo_name, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

        demo_name, t = self.samples[idx]
        demo = self._file["data"][demo_name]

        image = demo["images"][t]
        robot_state = demo["robot_state"][t]
        action = demo["actions"][t]

        # Convert image from HWC to CHW and scale to [0, 1].
        image = torch.from_numpy(
            np.array(image, copy=True)
        ).permute(2, 0, 1).float() / 255.0

        robot_state = torch.from_numpy(
            np.array(robot_state, copy=True)
        ).float()

        action = torch.from_numpy(
            np.array(action, copy=True)
        ).float()

        return {
            "image": image,
            "robot_state": robot_state,
            "action": action,
        }

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
