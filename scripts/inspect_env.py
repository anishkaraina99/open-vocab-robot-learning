
import robosuite as suite

env = suite.make(
    env_name="PickPlaceCan",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

obs = env.reset()

print("RoboSuite version:", suite.__version__)
print("Environment:", type(env).__name__)
print("Robot:", env.robots[0].name)
print("Action dimension:", env.action_dim)
print("Action bounds:", env.action_spec)
print("Control frequency:", env.control_freq)
print("Robot state shape:", obs["robot0_proprio-state"].shape)
print("Object state shape:", obs["object-state"].shape)

env.close()
