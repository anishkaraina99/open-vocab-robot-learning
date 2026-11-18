
import os
import json
import argparse

import h5py
import numpy as np
import cv2
import robosuite as suite


DATASET = "data/robomimic/v1.5/can/ph/low_dim_v15.hdf5"

COMPARE_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_joint_pos",
    "robot0_gripper_qpos",
]


def render_frame(env, size=256):
    image = env.sim.render(
        camera_name="agentview",
        width=size,
        height=size,
    )
    return np.flipud(image).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--output",
        default="results/rgb_full/expert_replay.mp4",
    )
    args = parser.parse_args()

    with h5py.File(DATASET, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demo = f["data"][args.demo]

        states = demo["states"][:]
        actions = demo["actions"][:]
        model_xml = demo.attrs["model_file"]

        expected_next = {
            key: demo["next_obs"][key][:]
            for key in COMPARE_KEYS
        }

    kwargs = dict(env_args["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )

    env = suite.make(
        env_name=env_args["env_name"],
        **kwargs,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (args.size, args.size),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create video writer")

    success = False

    try:
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # Match the controller initialization used in
        # our earlier single-action replay test.
        env.robots[0].reset(deterministic=True)

        print("Replaying:", args.demo)
        print("Number of expert actions:", len(actions))

        for t, action in enumerate(actions):
            frame = render_frame(env, args.size)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            obs, reward, done, info = env.step(action)

            if (t + 1) % 10 == 0 or t == 0:
                print(f"\nStep {t + 1}")

                for key in COMPARE_KEYS:
                    predicted = np.asarray(obs[key])
                    expected = expected_next[key][t]

                    error = np.linalg.norm(predicted - expected)
                    print(f"  {key}: L2 error = {error:.5f}")

            if env._check_success():
                success = True

        # Include final state in the video.
        frame = render_frame(env, args.size)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    finally:
        writer.release()
        env.close()

    print("\n=== EXPERT REPLAY RESULTS ===")
    print("Success reached:", success)
    print("Video:", args.output)


if __name__ == "__main__":
    main()
