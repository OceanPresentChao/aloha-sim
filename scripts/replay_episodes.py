import os
import h5py
import argparse
import matplotlib.pyplot as plt

from aloha_gym import make_aloha_env
from constants import DT

import IPython

e = IPython.embed


def main(args):
    """
    replay episode in sim
    """
    dataset_dir = args["dataset_dir"]
    episode_idx = args["episode_idx"]
    dataset_name = f"episode_{episode_idx}"

    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        # actions = root["/action"][()]

    env = make_aloha_env(task_name="transfer_cube")
    env.reset()
    print("start replay")
    plt.ion()
    ax = plt.subplot()
    plt_img = ax.imshow(env.render())

    for action in qpos:
        env.step(action)
        plt_img.set_data(env.render())
        plt.pause(DT)

    print("replay finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", action="store", type=str, help="Dataset dir.", required=True
    )
    parser.add_argument(
        "--episode_idx", action="store", type=int, help="Episode index.", required=False
    )
    main(vars(parser.parse_args()))
