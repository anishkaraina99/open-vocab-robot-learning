
import os
import h5py
import numpy as np
import robosuite as suite
from PIL import Image

DATASET = "data/robomimic/v1.5/can/ph/low_dim_v15.hdf5"
OUTPUT_DIR = "data/demo_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with h5py.File(DATASET, "r") as f:
    demo = f["data/demo_0"]
    states = demo["states"][:]
    model_xml = demo.attrs["model_file"]

print("Loaded", len(states), "saved states")

env = suite.make(
    env_name="PickPlaceCan",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=256,
    camera_widths=256,
    reward_shaping=False,
)

env.reset()

# Restore the exact model used during demonstration collection
env.reset_from_xml_string(model_xml)
env.sim.reset()

for i, state in enumerate(states):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()

    # Render the camera image for this saved state
    image = env.sim.render(
        camera_name="agentview",
        width=256,
        height=256,
    )

    image = np.flipud(image)

    Image.fromarray(image).save(
        os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
    )

    if i % 20 == 0:
        print(f"Rendered frame {i}/{len(states)}")

env.close()

print("\nDone! Frames saved to:", OUTPUT_DIR)
