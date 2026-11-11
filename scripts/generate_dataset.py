
import os
import json
import argparse
import h5py
import numpy as np
import robosuite as suite

INPUT = "data/robomimic/v1.5/can/ph/low_dim_v15.hdf5"
OUTPUT = "data/processed/can_rgb.hdf5"

ROBOT_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    with h5py.File(INPUT, "r") as source:
        demos = sorted(
            [k for k in source["data"] if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1])
        )

        if args.limit is not None:
            demos = demos[:args.limit]

        env_args = json.loads(source["data"].attrs["env_args"])
        env_kwargs = dict(env_args["env_kwargs"])

        # Override only rendering-related settings.
        env_kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=args.size,
            camera_widths=args.size,
        )

        print("Creating environment...")
        env = suite.make(
            env_name=env_args["env_name"],
            **env_kwargs,
        )

        env.reset()

        with h5py.File(OUTPUT, "w") as dest:
            dest.attrs["image_size"] = args.size
            dest.attrs["robot_keys"] = json.dumps(ROBOT_KEYS)
            dest.attrs["source"] = INPUT

            output_group = dest.create_group("data")

            for demo_idx, demo_name in enumerate(demos):
                demo = source["data"][demo_name]
                states = demo["states"][:]
                actions = demo["actions"][:].astype(np.float32)

                model_xml = demo.attrs["model_file"]

                # Restore original MuJoCo model and initial state.
                env.reset_from_xml_string(model_xml)
                env.sim.reset()

                num_steps = len(actions)
                images = np.empty(
                    (num_steps, args.size, args.size, 3),
                    dtype=np.uint8,
                )

                robot_state = np.concatenate(
                    [demo["obs"][key][:] for key in ROBOT_KEYS],
                    axis=1,
                ).astype(np.float32)

                for t in range(num_steps):
                    env.sim.set_state_from_flattened(states[t])
                    env.sim.forward()

                    image = env.sim.render(
                        camera_name="agentview",
                        width=args.size,
                        height=args.size,
                    )
                    images[t] = np.flipud(image)

                group = output_group.create_group(demo_name)

                group.create_dataset(
                    "images",
                    data=images,
                    compression="gzip",
                    compression_opts=4,
                )
                group.create_dataset("robot_state", data=robot_state)
                group.create_dataset("actions", data=actions)

                print(
                    f"[{demo_idx + 1}/{len(demos)}] "
                    f"{demo_name}: {num_steps} frames"
                )

            dest.attrs["num_demos"] = len(demos)

        env.close()

    print("\nDataset generated successfully!")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
