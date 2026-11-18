
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


SOURCE = "data/robomimic/v1.5/can/ph/low_dim_v15.hdf5"
SPLITS = "configs/dataset_splits.json"

ROBOT_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
]


def get_test_demos():
    with open(SPLITS) as f:
        splits = json.load(f)

    # Support a plain list or a common nested split format.
    test = splits.get("test")
    if test is None and "splits" in splits:
        test = splits["splits"]["test"]

    if isinstance(test, dict):
        test = test.get("demos", test.get("demo_names"))

    if not isinstance(test, list) or not test:
        raise ValueError(
            "Cannot find test demo list in dataset_splits.json"
        )

    return test


def robot_state(obs):
    return np.concatenate([
        np.asarray(obs[key]).reshape(-1)
        for key in ROBOT_KEYS
    ]).astype(np.float32)


def render(env, size):
    return np.flipud(env.sim.render(
        camera_name="agentview",
        width=size,
        height=size,
    )).copy()


def predict_action(model, image, state):
    image_tensor = (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0) / 255.0
    )

    state_tensor = (
        torch.from_numpy(state.copy())
        .float()
        .unsqueeze(0)
    )

    with torch.inference_mode():
        action = model(
            image_tensor,
            state_tensor,
        )[0].cpu().numpy()

    return np.clip(action, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rgb_full/best.pt",
    )
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-size", type=int, default=256)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="results/rgb_full/test_rollouts",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model = BehaviorCloningPolicy(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    demos = get_test_demos()

    if args.limit is not None:
        demos = demos[:args.limit]

    with h5py.File(SOURCE, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

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

    results = []

    try:
        with h5py.File(SOURCE, "r") as f:
            for index, demo_name in enumerate(demos):
                demo = f["data"][demo_name]

                model_xml = demo.attrs["model_file"]
                initial_state = demo["states"][0].copy()

                print(
                    f"\n[{index + 1}/{len(demos)}] "
                    f"Evaluating {demo_name}",
                    flush=True,
                )

                # Match the successful single-demo rollout.
                env.reset()
                env.reset_from_xml_string(model_xml)
                env.sim.reset()
                env.sim.set_state_from_flattened(initial_state)
                env.sim.forward()
                env.robots[0].reset(deterministic=True)

                obs = env._get_observations(
                    force_update=True
                )

                writer = None
                video_path = None

                if not args.no_video:
                    video_path = os.path.join(
                        args.output_dir,
                        f"{demo_name}.mp4",
                    )
                    writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20,
                        (args.video_size, args.video_size),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Cannot create {video_path}"
                        )

                success = False
                gripper_values = []
                steps_taken = 0

                try:
                    for t in range(args.horizon):
                        # Model always sees 128x128 images.
                        image = render(env, 128)
                        state = robot_state(obs)

                        action = predict_action(
                            model, image, state
                        )

                        gripper_values.append(
                            float(action[6])
                        )

                        if writer is not None:
                            frame = render(
                                env, args.video_size
                            )
                            writer.write(
                                cv2.cvtColor(
                                    frame,
                                    cv2.COLOR_RGB2BGR,
                                )
                            )

                        obs, reward, done, info = env.step(
                            action
                        )
                        steps_taken = t + 1

                        if env._check_success():
                            success = True
                            break
                finally:
                    if writer is not None:
                        writer.release()

                switches = sum(
                    np.sign(gripper_values[i])
                    != np.sign(gripper_values[i - 1])
                    for i in range(1, len(gripper_values))
                )

                result = {
                    "demo": demo_name,
                    "success": bool(success),
                    "steps": steps_taken,
                    "gripper_switches": int(switches),
                    "video": video_path,
                }

                results.append(result)

                print(
                    f"  {'SUCCESS' if success else 'FAILED'} "
                    f"| Steps: {steps_taken} "
                    f"| Gripper switches: {switches}",
                    flush=True,
                )

                # Save incrementally in case the run stops.
                with open(
                    os.path.join(
                        args.output_dir, "results.json"
                    ),
                    "w",
                ) as out:
                    json.dump(results, out, indent=2)

    finally:
        env.close()

    successes = [
        r for r in results if r["success"]
    ]

    success_rate = (
        len(successes) / len(results)
        if results else 0.0
    )

    avg_success_steps = (
        float(np.mean([
            r["steps"] for r in successes
        ]))
        if successes else None
    )

    summary = {
        "checkpoint": args.checkpoint,
        "split": "test",
        "horizon": args.horizon,
        "num_episodes": len(results),
        "num_successes": len(successes),
        "success_rate": success_rate,
        "avg_steps_successful": avg_success_steps,
        "episodes": results,
    }

    path = os.path.join(
        args.output_dir, "summary.json"
    )

    with open(path, "w") as out:
        json.dump(summary, out, indent=2)

    print("\n=== FINAL RESULTS ===")
    print(
        f"Successes: {len(successes)}/{len(results)}"
    )
    print(f"Success rate: {success_rate:.1%}")
    print(
        "Average steps (successful episodes):",
        avg_success_steps,
    )
    print("Saved:", path)


if __name__ == "__main__":
    main()
