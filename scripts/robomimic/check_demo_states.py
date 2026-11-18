
import h5py
import numpy as np
import robosuite as suite

DATASET = "data/robomimic/v1.5/can/ph/low_dim_v15.hdf5"

with h5py.File(DATASET, "r") as f:
    demo = f["data/demo_0"]

    states = demo["states"][:]
    actions = demo["actions"][:]

    print("Saved states:", states.shape)
    print("Saved actions:", actions.shape)
    print("First action:", actions[0])

    # Inspect the XML model saved with the demonstration
    model_xml = demo.attrs.get("model_file")

    print("Model XML available:", model_xml is not None)

    if model_xml is not None:
        print("Model XML length:", len(model_xml))

    # Inspect the saved states for invalid values
    print("States finite:", np.isfinite(states).all())
    print("Actions finite:", np.isfinite(actions).all())

print("\nCreating Panda PickPlaceCan environment...")

env = suite.make(
    env_name="PickPlaceCan",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

env.reset()

current_state = env.sim.get_state().flatten()

print("Current simulator state shape:", current_state.shape)
print("Dataset state shape:", states[0].shape)

print(
    "State dimensions match:",
    current_state.shape == states[0].shape
)

env.close()
