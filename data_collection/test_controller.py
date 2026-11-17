
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config


def main():

    # --------------------------------------------------
    # 1. Load the controller configuration
    # --------------------------------------------------

    controller_config = load_composite_controller_config(
        controller="BASIC"
    )

    # --------------------------------------------------
    # 2. Create the RoboSuite environment
    # --------------------------------------------------

    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        controller_configs=controller_config,

        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,

        control_freq=20,
        horizon=1000,

        single_object_mode=0,
    )

    try:

        # --------------------------------------------------
        # 3. Reset the environment
        # --------------------------------------------------

        obs = env.reset()

        print("\nEnvironment initialized!")
        print("Action dimension:", env.action_dim)
        print("Action bounds:", env.action_spec)

        print("\nController action information:")
        env.robots[0].print_action_info()

        # --------------------------------------------------
        # 4. Test end-effector movement
        # --------------------------------------------------

        print("\nTesting upward movement...")

        initial_pos = env.sim.data.site(
            "gripper0_right_grip_site"
        ).xpos.copy()

        print("Initial end-effector position:", initial_pos)

        for _ in range(100):

            action = np.zeros(env.action_dim)

            # Move the end effector upward
            action[2] = 0.5

            obs, reward, done, info = env.step(action)

            env.render()

        final_pos = env.sim.data.site(
            "gripper0_right_grip_site"
        ).xpos.copy()

        print("Final end-effector position:", final_pos)
        print("Position change:", final_pos - initial_pos)

        # --------------------------------------------------
        # 5. Test gripper movement
        # --------------------------------------------------

        print("\nTesting gripper...")

        # Close the gripper
        print("Closing gripper...")

        for _ in range(50):

            action = np.zeros(env.action_dim)
            action[-1] = -1.0

            obs, reward, done, info = env.step(action)
            env.render()

        # Open the gripper
        print("Opening gripper...")

        for _ in range(50):

            action = np.zeros(env.action_dim)
            action[-1] = 1.0

            obs, reward, done, info = env.step(action)
            env.render()

        print("\nController test completed!")

    finally:
        env.close()


c
