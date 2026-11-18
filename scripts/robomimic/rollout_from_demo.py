
import os
import sys
import json
import argparse

import cv2
import h5py
import numpy as np
import torch
import robosuite as suite

sys.path.insert(0, "src/policies")
from bc_model import BehaviorCloningPolicy


DATASET = "data/robomimic/v1.5/can/ph/low_dim_v15.hdf5"

ROBOT_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
]


def get_robot_state(obs):
    return np.concatenate([
        np.asarray(obs[key]).reshape(-1)
        for key in ROBOT_KEYS
    ]).astype(np.float32)


def render_image(env, size=128):
    image = env.sim.render(
        camera_name="agentview",
        width=size,
        height=size,
    )
    return np.flipud(image).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rgb_full/best.pt",
    )
    parser.add_argument(
        "--output",
        default="results/rgb_full/rollout_from_demo.mp4",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model = BehaviorCloningPolicy(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with h5py.File(DATASET, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demo = f["data"][args.demo]
        model_xml = demo.attrs["model_file"]
        initial_state = demo["states"][0].copy()
        expert_actions = demo["actions"][:]

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
        (256, 256),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create video")

    gripper_log = []
    success = False

    try:
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        env.robots[0].reset(deterministic=True)

        # Read observations after restoring the initial state.
        obs = env._get_observations(force_update=True)

        print("Starting policy rollout from:", args.demo)
        print("Expert trajectory length:", len(expert_actions))

        for t in range(args.horizon):
            # Same 128x128 image as training.
            image = render_image(env, 128)
            robot_state = get_robot_state(obs)

            image_tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0) / 255.0
            )

            state_tensor = (
                torch.from_numpy(robot_state)
                .float()
                .unsqueeze(0)
            )

            with torch.no_grad():
                action = model(
                    image_tensor,
                    state_tensor,
                ).squeeze(0).numpy()

            action = np.clip(action, -1.0, 1.0)
            gripper_log.append(float(action[6]))

            # Higher-resolution video, same camera view.
            frame = render_image(env, 256)
            writer.write(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )

            obs, reward, done, info = env.step(action)

            if t % 10 == 0:
                print(
                    f"Step {t:3d} | "
                    f"Gripper: {action[6]:+.3f} | "
                    f"Position action: "
                    f"{np.round(action[:3], 3)}",
                    flush=True,
                )

            if env._check_success():
                success = True
                print(f"\nSUCCESS at step {t + 1}!")
                break

        print("\n=== RESULTS ===")
        print("Success:", success)
        print("Gripper min:", min(gripper_log))
        print("Gripper max:", max(gripper_log))
        print("Gripper sign switches:", sum(
            np.sign(gripper_log[i]) != np.sign(gripper_log[i - 1])
            for i in range(1, len(gripper_log))
        ))

    finally:
        writer.release()
        env.close()

    print("Video saved:", args.output)


if __name__ == "__main__":
    main()
