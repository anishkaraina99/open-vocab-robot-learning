
import json
import h5py
import numpy as np
import robosuite as suite

DATASET = "data/robomimic/v1.5/can/ph/low_dim_v15.hdf5"

KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_gripper_qpos",
]

with h5py.File(DATASET, "r") as f:
    env_args = json.loads(f["data"].attrs["env_args"])
    demo = f["data/demo_0"]

    kwargs = dict(env_args["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    env = suite.make(
        env_name=env_args["env_name"],
        **kwargs,
    )

    env.reset()
    env.reset_from_xml_string(demo.attrs["model_file"])
    env.sim.reset()

    for t in [0, 1, 20, 50, 100]:
        env.sim.set_state_from_flattened(demo["states"][t])
        env.sim.forward()

        obs = env._get_observations(force_update=True)

        print(f"\n=== TIMESTEP {t} ===")

        for key in KEYS:
            predicted = np.asarray(obs[key])
            expected = np.asarray(demo["obs"][key][t])

            error = np.linalg.norm(predicted - expected)

            print(f"{key}: L2 error = {error:.6f}")

    env.close()
