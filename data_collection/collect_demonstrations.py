import random
import numpy as np

# Ignore warnings
import logging
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

import robosuite as suite
from robosuite import load_composite_controller_config

# ---------------------------------------
# Configuration
# ---------------------------------------

NUM_SUCCESSFUL_EPISODES = 1#300
MAX_ATTEMPTS = 1000
MAX_STEPS_PER_EPISODE = 3000

OBJECTS = [
    "milk",
    "bread",
    "cereal",
    "can",
]

CAMERA = "agentview"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

OUTPUT_PATH = "data/pickplace_demonstrations.hdf5"

OPEN = -1.0
CLOSED = 1.0

def get_target_positions(env):
    # Define desired object placement positions
    target_positions = {
        "milk": env.target_bin_placements[env.object_to_id["milk"]].copy(),
        "bread": env.target_bin_placements[env.object_to_id["bread"]].copy(),
        "cereal": env.target_bin_placements[env.object_to_id["cereal"]].copy(),
        "can": env.target_bin_placements[env.object_to_id["can"]].copy(),
    }

    # Adjust z to position gripper above object bin position
    # for pos in target_positions.values():
    #     pos[2] += 0.05

    return target_positions

def object_in_correct_bin(env, object_name):
    return env.objects_in_bins[env.object_to_id[object_name]] == 1

def compute_control_action(obs, target_position, gripper_action):
    eef_pos = obs["robot0_eef_pos"]

    position_error = target_position - eef_pos

    # OSC_POSE translational input is normalized to [-1, 1]
    translation = np.clip(position_error / 0.05, -1.0, 1.0)

    rotation = np.zeros(3)

    return np.concatenate([
        translation,
        rotation,
        [gripper_action]
    ])

def reached_target(obs, target_pose):
    # Check whether the end effector reached its target
    eef_pos = obs["robot0_eef_pos"]
    distance = np.linalg.norm(target_pose - eef_pos)

    # 1cm tolerance
    return distance < 0.01


def main():

    # Load default Panda-compatible controller
    controller_config = load_composite_controller_config(
        controller="BASIC"
    )

    # ---------------------------------------
    # 1. Initialize environment
    # ---------------------------------------

    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,###
        has_offscreen_renderer=False, ###
        use_camera_obs=False, ###
        camera_names=CAMERA,
        camera_heights=IMAGE_HEIGHT,
        camera_widths=IMAGE_WIDTH,
        single_object_mode=0,
    )

    try:
        # initialize_hdf5_file(OUTPUT_PATH)

        successful_episodes = 0
        attempts = 0

        # ---------------------------------------
        # 2. Generate demonstrations
        # ---------------------------------------

        while successful_episodes < NUM_SUCCESSFUL_EPISODES \
            and attempts < MAX_ATTEMPTS:

            attempts += 1

            steps = 0

            # Reset environment and randomize object placements
            obs = env.reset()

            target_positions = get_target_positions(env)
            # randomize_object_positions(env)
            # randomize_object_orientations(env)

            # Advance simulation if necessary after randomization
            # env.sim.forward()

            # Store data for this episode
            images = []
            robot_states = []
            actions = []
            instructions = []

            episode_success = True

            # Randomize order in which objects are picked
            object_order = random.sample(OBJECTS, len(OBJECTS))

            # ---------------------------------------
            # 3. Pick and place each object
            # ---------------------------------------

            for object_name in object_order:

                # Retrieve privileged simulator information
                object_pos = obs.get(f"{object_name.capitalize()}_pos")
                target_pos = target_positions.get(object_name)

                # Create language instruction
                instruction = f"Pick up the {object_name} and place it in its bin"

                # Move through grasping and placement stages
                stages = [
                    ("APPROACH", object_pos + [0, 0, 0.10], OPEN),
                    ("DESCEND",  object_pos + [0, 0, 0.02], OPEN),
                    ("GRASP",    object_pos + [0, 0, 0.02], CLOSED),
                    ("LIFT",     object_pos + [0, 0, 0.15], CLOSED),
                    ("TRANSFER", target_pos + [0, 0, 0.15], CLOSED),
                    ("LOWER",    target_pos + [0, 0, 0.05], CLOSED),
                    ("RELEASE",  target_pos + [0, 0, 0.05], OPEN),
                    ("RETREAT",  target_pos + [0, 0, 0.15], OPEN),
                ]

                for stage, target_pose, gripper in stages:

                    # Move toward target pose using OSC controller
                    while True:

                        action = compute_control_action(
                            obs,
                            target_pose,
                            gripper
                        )

                        # # Record observation BEFORE taking action
                        # images.append(obs["agentview_image"])
                        # robot_states.append(obs["robot0_proprio-state"])
                        # actions.append(action)
                        # instructions.append(instruction)

                        # Execute action
                        obs, reward, done, info = env.step(action)

                        env.render()

                        if reached_target(obs, target_pose):
                            break

                        steps += 1
                        if done or steps >= MAX_STEPS_PER_EPISODE:
                            episode_success = False
                            break

                    if not episode_success:
                        break

                if not episode_success:
                    break

                # Verify that the object was actually placed correctly
                if not object_in_correct_bin(env, object_name):
                    episode_success = False
                    break

            # ---------------------------------------
            # 4. Validate and save episode
            # ---------------------------------------

            if episode_success: #and all_objects_in_correct_bins(env):

                # save_episode(
                #     file_path=OUTPUT_PATH,
                #     episode_id=successful_episodes,
                #     images=images,
                #     robot_states=robot_states,
                #     actions=actions,
                #     instructions=instructions,
                #     metadata={
                #         "object_order": object_order,
                #         "success": True,
                #     }
                # )

                successful_episodes += 1
                print("Saved episode:", successful_episodes)

            else:
                print("Discarding failed episode:", attempts)

    finally:
        env.close()

if __name__ == "__main__":
    main()