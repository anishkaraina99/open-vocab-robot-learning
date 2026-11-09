
import os

import numpy as np
import robosuite as suite
from PIL import Image

# 1. Initialize the simulation
env = suite.make(
    env_name="PickPlaceCan",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=512,
    camera_widths=512,
    reward_shaping=False,
)

# 2. Reset to a new episode
obs = env.reset()

# 3. Inspect available observations
print("\n=== OBSERVATIONS ===")
for key, value in obs.items():
    print(f"{key}: {np.shape(value)}")

# 4. Inspect the action space
print("\n=== ACTION SPACE ===")
print("Action dimension:", env.action_dim)
print("Action bounds:", env.action_spec)

# 5. Save the RGB image
os.makedirs("data/images", exist_ok=True)

image = obs["agentview_image"]
image = np.flipud(image)

Image.fromarray(image).save("data/images/pickplace.png")
print("\nSaved image: data/images/pickplace.png")

# 6. Run a short simulation with neutral actions
action = np.zeros(env.action_dim)

for step in range(100):
    obs, reward, done, info = env.step(action)

    if step % 20 == 0:
        print(f"Step {step}: reward={reward}")

env.close()
print("\nSimulation completed successfully!")
