import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import argparse
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from tqdm import tqdm
import h5py


from constants import (
    SIM_TASK_CONFIGS,
    MASTER_START_ARM_POSE,
    MASTER_GRIPPER_JOINT_OPEN,
    DT,
)
from aloha_gym import make_aloha_env, AlohaEnv
from robot_utils import (
    torque_off,
    torque_on,
    move_arms,
    move_grippers,
    get_arm_gripper_positions,
)
from sim_utils import get_action

import IPython

e = IPython.embed


def opening_ceremony(master_bot_left, master_bot_right, env: AlohaEnv):
    """Move all 4 robots to a pose where it is easy to start demonstration"""
    # reboot gripper motors, and set operating modes for all motors
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")

    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")

    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_joint = MASTER_START_ARM_POSE[:6]
    move_arms(
        [master_bot_left, master_bot_right],
        [start_arm_joint] * 2,
        move_time=1.5,
    )
    # move grippers to starting position
    move_grippers(
        [master_bot_left, master_bot_right],
        [MASTER_GRIPPER_JOINT_OPEN, MASTER_GRIPPER_JOINT_OPEN] * 2,
        move_time=1,
    )

    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print("Close the gripper to start")
    close_thresh = -0.3
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        # sync in sim
        env_action = get_action(master_bot_left, master_bot_right)
        env.step(env_action)
        if (abs(gripper_pos_left - close_thresh) < 0.03) and (
            abs(gripper_pos_right - close_thresh) < 0.03
        ):
            pressed = True
        time.sleep(DT / 10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print("Started!")


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args["task_name"]
    overwrite = args["overwrite"]
    task_config = SIM_TASK_CONFIGS[args["task_name"]]
    dataset_dir = task_config["dataset_dir"]
    max_timesteps = task_config["episode_len"]
    render_cam = "teleoperator_pov"
    task_camera_names = task_config["camera_names"]
    episode_idx = args["episode_idx"]

    assert episode_idx is not None
    dataset_name = f"episode_{episode_idx}"

    dataset_path = os.path.join(dataset_dir, dataset_name) + ".hdf5"
    if os.path.isfile(dataset_path) and not overwrite:
        print(
            f"Dataset already exist at \n{dataset_path}\nHint: set overwrite to True."
        )
        exit()

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    master_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="master_left",
        init_node=True,
    )
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="master_right",
        init_node=False,
    )

    env = make_aloha_env(task_name=task_name, render_cam=render_cam)
    env.reset()

    opening_ceremony(master_bot_left, master_bot_right, env)

    # Data collection
    obs = env.get_obs()
    actions = []
    episode = [obs]
    actual_dt_history = []

    plt.ion()
    ax = plt.subplot()
    plt_img = ax.imshow(env.render())

    for t in tqdm(range(max_timesteps)):
        t0 = time.time()  #
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time()  #
        obs, reward, terminated, truncated, info = env.step(action)
        t2 = time.time()  #

        plt_img.set_data(env.render())
        plt.pause(DT)

        episode.append(obs)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])
    plt.close()

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 42:
        print(f"episode is not healthy, freq_mean:{freq_mean}")
        return False

    # len(action): max_timesteps, len(episode): max_timesteps + 1
    save_episode(episode, action, task_camera_names, dataset_path, True)

    print(f"Saved to {dataset_dir}")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(
        f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}"
    )
    return freq_mean


def save_episode(
    episode, action, camera_names: list[str], dataset_path: str, sim: bool
):
    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'

    action                  (14,)         'float64'
    """
    assert len(episode) > 1

    # len(action): max_timesteps, len(episode): max_timesteps + 1
    if len(action) > 0:
        assert len(action) == len(episode) - 1

    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/action": [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    data_dict["/action"] = action
    for obs in episode:
        data_dict["/observations/qpos"].append(obs["qpos"])
        data_dict["/observations/qvel"].append(obs["qvel"])
        if "effort" in obs:
            data_dict["/observations/qvel"].append(obs["effort"])

        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                obs["images"][cam_name]
            )

    # HDF5
    assert str(dataset_path).endswith(".hdf5")

    t0 = time.time()
    with h5py.File(dataset_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = sim
        root.attrs["max_timesteps"] = len(action)
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            _ = image.create_dataset(
                cam_name,
                (len(episode), 480, 640, 3),
                dtype="uint8",
                chunks=(1, 480, 640, 3),
            )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        _ = obs.create_dataset("qpos", (len(episode), 14))
        _ = obs.create_dataset("qvel", (len(episode), 14))
        _ = obs.create_dataset("effort", (len(episode), 14))
        _ = root.create_dataset("action", (len(episode), 14))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f"Saving: {time.time() - t0:.1f} secs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="dataset saving dir",
        required=True,
    )
    parser.add_argument(
        "--num_episodes", action="store", type=int, help="num_episodes", required=False
    )
    parser.add_argument("--onscreen_render", action="store_true")

    main(vars(parser.parse_args()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", action="store", type=str, help="Task name.", required=True
    )
    parser.add_argument(
        "--episode_idx",
        action="store",
        type=int,
        help="Episode index.",
        default=None,
        required=False,
    )
    main(vars(parser.parse_args()))
    # debug()
