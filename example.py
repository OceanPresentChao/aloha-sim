import numpy as np
from os import path
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio
from rdt_gym import aloha_env

gym.register(
    id="aloha",
    entry_point=aloha_env.AlohaEnv,
)

model_path = path.join(path.dirname(__file__), "./aloha_xml/scene.xml")
output_path = path.join(path.dirname(__file__), "./data")

env = gym.make("aloha", xml_file=model_path, max_episode_steps=30)

observation, info = env.reset()
frames = []

imageio.imsave(f"{output_path}/test.jpg", env.render())

for _ in range(100):
    action = env.action_space.sample()
    # print(f"sample action:{len(action)}")
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation = env.reset()

env.close()
imageio.mimsave(f"{output_path}/example.gif", np.stack(frames), fps=10)
