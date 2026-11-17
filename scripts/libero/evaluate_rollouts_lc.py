"""Closed-loop LIBERO evaluation from held-out demonstration initial states."""
import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, "src/policies")

from lc_model import LanguageConditionedPolicy
from encode_text_embeddings import encode_instructions


def instruction_for(task_name):
    # Must stay identical to train_lc.py / evaluate_lc.py.
    return task_name.replace("_", " ").capitalize() + "."


def read_test_demos(split_file):
    with open(split_file) as f:
        split = json.load(f)
    demos = split.get("test", [])
    if not demos:
        raise ValueError(f"No test demonstrations in {split_file}")
    return [(p.split("/", 1)[0], p.split("/", 1)[1]) for p in demos]


def robot_state_from_obs(obs):
    # Reconstruct the live 9D robot-state vector from simulator observations.
    # This is intended to match the dataset's robot_states representation:
    # [gripper_qpos(2), eef_position(3), eef_quaternion(4)].
    # The first rollout action uses demo["robot_states"][0] directly to ensure
    # the initial policy input exactly matches the stored training representation.
    state = np.concatenate([
        np.asarray(obs["robot0_gripper_qpos"]).reshape(-1),
        np.asarray(obs["robot0_eef_pos"]).reshape(-1),
        np.asarray(obs["robot0_eef_quat"]).reshape(-1),
    ]).astype(np.float32)
    if state.shape != (9,):
        raise ValueError(f"Expected 9D robot state, got {state.shape}")
    return state


def image_from_obs(obs, rotate_180=True):
    image = np.asarray(obs["agentview_image"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 agentview image, got {image.shape}")
    if rotate_180:
        image = np.rot90(image, 2)
    image = np.ascontiguousarray(image)
    if image.shape[:2] != (128, 128):
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    return image


def policy_action(model, image, state, language, device):
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        action = model(image, state, language)[0].cpu().numpy().astype(np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid policy action: {action}")
    return np.clip(action, -1.0, 1.0)


def check_success(env):
    if hasattr(env, "check_success"):
        return bool(env.check_success())
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    raise AttributeError("LIBERO env has no check_success() method")


def rollout_one(env, hdf5_path, demo_name, model, language, device,
                horizon, video_path, rotate_180):
    with h5py.File(hdf5_path, "r") as f:
        demo_key = f"data/{demo_name}"
        if demo_key not in f:
            raise KeyError(f"Missing {demo_key} in {hdf5_path}")
        demo = f[demo_key]
        if "states" not in demo or len(demo["states"]) == 0:
            raise KeyError(f"{demo_key} has no simulator states")
        if "robot_states" not in demo or len(demo["robot_states"]) == 0:
            raise KeyError(f"{demo_key} has no robot_states for input validation")
        initial_state = np.asarray(demo["states"][0]).copy()
        reference_robot_state = np.asarray(demo["robot_states"][0]).astype(np.float32)

    # Reconstruct the held-out demo's initial state; do not replay its actions.
    env.reset()
    obs = env.set_init_state(initial_state)
    observed_robot_state = robot_state_from_obs(obs)
    if reference_robot_state.shape != (9,):
        raise ValueError(
            f"Expected demo robot_states[0] to be 9D, got {reference_robot_state.shape}"
        )
    state_max_abs_error = float(
        np.max(np.abs(observed_robot_state - reference_robot_state))
    )
    if state_max_abs_error > 1e-3:
        print(
            "WARNING: simulator state at set_init_state differs from the "
            f"stored demo robot_states[0] (max abs error={state_max_abs_error:.6g}). "
            "Check state ordering, quaternion convention, and demo/environment versions.",
            flush=True,
        )

    writer = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 20, (128, 128))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video {video_path}")

    success, steps = False, 0
    try:
        for t in range(horizon):
            image = image_from_obs(obs, rotate_180)

            # At t=0, use the exact precomputed state representation that
            # was supplied to the policy during training. After the first
            # environment step, construct the current state from live
            # simulator observations (the policy cannot use future demo data).
            if t == 0:
                state = reference_robot_state.copy()
            else:
                state = robot_state_from_obs(obs)

            action = policy_action(model, image, state, language, device)
            if writer is not None:
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            obs, reward, done, info = env.step(action.tolist())
            steps = t + 1
            if check_success(env):
                success = True
                break
            if done:
                break
    finally:
        if writer is not None:
            writer.release()

    return {"demo": demo_name, "success": bool(success), "steps": steps,
            "initial_state_max_abs_error": state_max_abs_error,
            "video": str(video_path) if video_path else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/lc_baseline/best.pt")
    parser.add_argument("--dataset-dir", default=os.environ.get("LIBERO_DATASET_DIR", "datasets/libero_object"))
    parser.add_argument("--splits", default="configs/libero_dataset_splits.json")
    parser.add_argument("--task-name", default=None,
                        help="One task from the test split; default is the first sorted task")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Evaluate all tasks represented in the test split")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum held-out demos per task")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--rotate-image-180", action="store_true",
                        help="Rotate simulator images 180 degrees if needed to match training images")
    parser.add_argument("--output-dir", default="results/lc_baseline/test_rollouts")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.horizon < 1:
        parser.error("--horizon must be positive")

    dataset_dir, split_file = Path(args.dataset_dir), Path(args.splits)
    checkpoint_path, output_dir = Path(args.checkpoint), Path(args.output_dir)
    for path, label in [(dataset_dir, "dataset directory"),
                        (split_file, "split manifest"),
                        (checkpoint_path, "checkpoint")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print("Evaluation device:", device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = LanguageConditionedPolicy(9, 512, 7, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_demos = read_test_demos(split_file)
    available_tasks = sorted({task for task, _ in test_demos})
    if args.task_name:
        if args.task_name not in available_tasks:
            raise ValueError(f"{args.task_name} is not in test split. Tasks: {available_tasks}")
        tasks = [args.task_name]
    else:
        tasks = available_tasks if args.all_tasks else [available_tasks[0]]

    # Use the same task-name-derived text as training, not task.language.
    encoded = encode_instructions([instruction_for(t) for t in tasks]).detach().cpu()
    language = {t: encoded[i:i+1].to(device) for i, t in enumerate(tasks)}

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task_lookup = {suite.get_task(i).name: suite.get_task(i) for i in range(suite.n_tasks)}

    all_results = []
    rotate_180 = args.rotate_image_180
    for task_name in tasks:
        if task_name not in task_lookup:
            raise KeyError(f"{task_name} is not a task in libero_object")
        task = task_lookup[task_name]
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl.is_file():
            raise FileNotFoundError(f"BDDL file not found: {bddl}")
        hdf5_path = dataset_dir / f"{task_name}_demo.hdf5"
        if not hdf5_path.is_file():
            raise FileNotFoundError(f"Task HDF5 not found: {hdf5_path}")
        demos = [demo for name, demo in test_demos if name == task_name]
        if args.limit is not None:
            demos = demos[:args.limit]

        print(f"\n{task_name}: evaluating {len(demos)} held-out demos", flush=True)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
        env.seed(args.seed)
        try:
            for i, demo_name in enumerate(demos):
                print(f"  [{i+1}/{len(demos)}] {demo_name}", flush=True)
                video = None if args.no_video else output_dir / task_name / f"{demo_name}.mp4"
                result = rollout_one(env, hdf5_path, demo_name, model,
                                     language[task_name], device, args.horizon,
                                     video, rotate_180)
                result["task_name"] = task_name
                all_results.append(result)
                print(f"    {'SUCCESS' if result['success'] else 'FAILED'} ({result['steps']} steps)", flush=True)
                with open(output_dir / "episodes.json", "w") as f:
                    json.dump(all_results, f, indent=2)
        finally:
            env.close()

    per_task = {}
    for task_name in tasks:
        rows = [r for r in all_results if r["task_name"] == task_name]
        wins = [r for r in rows if r["success"]]
        per_task[task_name] = {
            "episodes": len(rows), "successes": len(wins),
            "success_rate": len(wins) / len(rows) if rows else 0.0,
            "average_success_steps": float(np.mean([r["steps"] for r in wins])) if wins else None,
        }
    successes = sum(r["success"] for r in all_results)
    summary = {
        "checkpoint": str(checkpoint_path), "dataset_dir": str(dataset_dir),
        "split": "test", "initial_state_source": "held-out demo states[0]",
        "benchmark": "libero_object", "tasks": per_task,
        "num_episodes": len(all_results), "num_successes": int(successes),
        "success_rate": successes / len(all_results) if all_results else 0.0,
        "horizon": args.horizon, "image_rotation_180": rotate_180,
        "episodes": all_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOverall success: {successes}/{len(all_results)} ({summary['success_rate']:.1%})")
    print("Saved:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
