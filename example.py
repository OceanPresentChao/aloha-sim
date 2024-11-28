import numpy as np
from os import path
import gymnasium as gym
import imageio

from rdt_gym import aloha_env
from scripts.visualize_episode import save_videos
from rdt_gym.task.task_none import NoneTask

gym.register(
    id="aloha",
    entry_point=aloha_env.AlohaEnv,
)

output_path = path.join(path.dirname(__file__), "./data")
task = NoneTask()
env = gym.make("aloha", task=task, max_episode_steps=30)

observation, info = env.reset()
frames = []
obs_frames = []


imageio.imsave(f"{output_path}/init.jpg", env.render())

for _ in range(100):
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
