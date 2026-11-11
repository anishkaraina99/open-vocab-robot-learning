
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

    states = demo["states"][:]
    actions = demo["actions"][:]
    model_xml = demo.attrs["model_file"]

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
    env.reset_from_xml_string(model_xml)
    env.sim.reset()

    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    # Reset controller targets to the restored robot configuration.
    env.robots[0].reset(deterministic=True)

    obs, reward, done, info = env.step(actions[0])

    print("\n=== OBSERVATION COMPARISON ===")

    for key in KEYS:
        predicted = np.asarray(obs[key])
        expected = np.asarray(demo["next_obs"][key][0])

        print(f"\n{key}")
        print("Predicted:", np.round(predicted, 5))
        print("Expected: ", np.round(expected, 5))
        print("L2 error:", np.linalg.norm(predicted - expected))

    env.close()
