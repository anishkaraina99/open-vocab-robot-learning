
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
    return np.concatenate(
        [np.asarray(obs[key]).reshape(-1) for key in ROBOT_KEYS]
    ).astype(np.float32)


def get_image(env, size=128):
    image = env.sim.render(
        camera_name="agentview",
        width=size,
        height=size,
    )
    return np.flipud(image).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rgb_full/best.pt",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        default="results/rgb_full/rollouts",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model = BehaviorCloningPolicy(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", args.checkpoint)
    print("Checkpoint epoch:", checkpoint["epoch"])

    # Reconstruct the expert environment configuration.
    with h5py.File(DATASET, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )

    env = suite.make(
        env_name=env_args["env_name"],
        **env_kwargs,
    )

    results = []

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            frames = []
            success = False

            print(f"\nStarting episode {episode + 1}")

            for step in range(args.horizon):
                image = get_image(env, args.size)
                robot_state = get_robot_state(obs)

                image_tensor = torch.from_numpy(
                    image.copy()
                ).permute(2, 0, 1).float()

                image_tensor = (
                    image_tensor.unsqueeze(0).to(device) / 255.0
                )

                state_tensor = torch.from_numpy(
                    robot_state
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    action = model(
                        image_tensor,
                        state_tensor,
                    ).squeeze(0).cpu().numpy()

                action = np.clip(action, -1.0, 1.0)

                obs, reward, done, info = env.step(action)

                frames.append(image)

                # Check RoboSuite task completion.
                if env._check_success():
                    success = True
                    print(f"Success at step {step + 1}")
                    break

                if (step + 1) % 50 == 0:
                    print(
                        f"Step {step + 1}/{args.horizon} | "
                        f"Action: {np.round(action, 3)}",
                        flush=True,
                    )

            # Save rollout video.
            video_path = os.path.join(
                args.output_dir,
                f"episode_{episode + 1}.mp4",
            )

            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20,
                (args.size, args.size),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not create video: {video_path}"
                )

            for frame in frames:
                writer.write(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )

            writer.release()

            results.append({
                "episode": episode + 1,
                "success": bool(success),
                "steps": len(frames),
                "video": video_path,
            })

            print(
                f"Episode {episode + 1}: "
                f"{'SUCCESS' if success else 'FAILED'}"
            )
            print("Video saved:", video_path)

    finally:
        env.close()

    success_rate = np.mean(
        [result["success"] for result in results]
    ) * 100

    summary = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "success_rate": float(success_rate),
        "results": results,
    }

    summary_path = os.path.join(
        args.output_dir,
        "summary.json",
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nClosed-loop evaluation complete!")
    print(f"Success rate: {success_rate:.1f}%")
    print("Results saved:", summary_path)


if __name__ == "__main__":
    main()
