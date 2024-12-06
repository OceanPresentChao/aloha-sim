import numpy as np
import os
from os import path
import imageio
from aloha_gym import make_aloha_env


from scripts.visualize_episodes import save_videos


output_path = path.join(path.dirname(__file__), "./output/transfer_cube")

if not path.exists(output_path):
    print(f"create dir {output_path}")
    os.makedirs(output_path)


env = make_aloha_env(task_name="transfer_cube")
observation, info = env.reset()
frames = []
obs_frames = []


imageio.imsave(f"{output_path}/init.jpg", env.render())

for _ in range(500):
    action = env.action_space.sample()
    # print(f"sample action:{len(action)}")
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    frames.append(image)
    obs_frames.append(observation["images"])

    if terminated or truncated:
        observation = env.reset()

env.close()
imageio.mimsave(f"{output_path}/collaborator.gif", np.stack(frames), fps=10)
save_videos(obs_frames, 0.1, f"{output_path}/observation.mp4")
