
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config


def main():
    controller_config = load_composite_controller_config(
        controller="BASIC"
    )

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
    
    # Environment information
    print(type(env))
    print(env.action_spec)

    # Robot information
    robot = env.robots[0]
    print(robot.name)
    print(robot.robot_model)

    # Available observation keys
    obs = env.reset()
    print(obs.keys())

    # Object positions
    print("Milk:", obs.get("Milk_pos"))
    print("Bread:", obs.get("Bread_pos"))
    print("Cereal:", obs.get("Cereal_pos"))
    print("Can:", obs.get("Can_pos"))

    # # Get bin names
    # for i in range(env.sim.model.nbody):
    #     print(env.sim.model.body_id2name(i))

    # print([attr for attr in dir(env) if "bin" in attr.lower() or "object" in attr.lower()])
    print(env.target_bin_placements)
    print(env.object_to_id)
    print(env.not_in_bin)
    print(env.objects_in_bins)

if __name__ == "__main__":
    main()