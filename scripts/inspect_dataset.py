
import h5py
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Path to the HDF5 dataset")
args = parser.parse_args()

with h5py.File(args.dataset, "r") as f:

    print("\n========== DATASET METADATA ==========")

    data = f["data"]

    if "env_args" in data.attrs:
        env_args = data.attrs["env_args"]

        if isinstance(env_args, bytes):
            env_args = env_args.decode("utf-8")

        print(json.dumps(json.loads(env_args), indent=2))
    else:
        print("No environment metadata found")

    demos = sorted(
        [key for key in data.keys() if key.startswith("demo_")]
    )

    print("\n========== DEMONSTRATIONS ==========")
    print("Number of demonstrations:", len(demos))

    if not demos:
        raise RuntimeError("No demonstrations found")

    demo = data[demos[0]]

    print("\n========== FIRST DEMONSTRATION ==========")
    print("Demonstration:", demos[0])

    def print_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")

    demo.visititems(print_dataset)

    if "actions" in demo:
        print("\nAction dimension:", demo["actions"].shape[-1])
        print("Trajectory length:", len(demo["actions"]))

    if "model_file" in demo.attrs:
        print("Model XML available: Yes")
    else:
        print("Model XML available: No")
