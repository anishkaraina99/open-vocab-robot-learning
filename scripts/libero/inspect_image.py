import h5py
import os

DATASET_PATH = "/Users/anabellafalk/cs6758/LIBERO/libero/datasets/libero_object"

# Find HDF5 files
hdf5_files = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith((".hdf5", ".h5")):
            hdf5_files.append(os.path.join(root, file))

print(f"Found {len(hdf5_files)} HDF5 files")
print(hdf5_files[:5])

demo_path = hdf5_files[0]

with h5py.File(demo_path, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(name)

    f.visititems(print_structure)

import matplotlib.pyplot as plt

with h5py.File(demo_path, "r") as f:
    image = f["data/demo_0/obs/agentview_rgb"][0]

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title("LIBERO Agent View")
plt.show()

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

import numpy as np
import matplotlib.pyplot as plt

with h5py.File(demo_path, "r") as f:
    images = f["data/demo_0/obs/agentview_rgb"]

    num_frames = len(images)
    indices = np.linspace(0, num_frames - 1, 8, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for ax, idx in zip(axes.flatten(), indices):
        ax.imshow(images[idx])
        ax.set_title(f"Frame {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


