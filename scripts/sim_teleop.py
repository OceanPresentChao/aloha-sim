import matplotlib.pyplot as plt
import numpy as np

from aloha_gym import make_aloha_env
from constants import (
    UNNORMALIZE_PUPPET_GRIPPER_QPOS,
    NORMALIZE_MASTER_GRIPPER_JOINT,
)


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_joint = master_bot_left.dxl.joint_states.position[7]
    right_gripper_joint = master_bot_right.dxl.joint_states.position[7]
    left_gripper_qpos = UNNORMALIZE_PUPPET_GRIPPER_QPOS(
        NORMALIZE_MASTER_GRIPPER_JOINT(left_gripper_joint)
    )
    right_gripper_qpos = UNNORMALIZE_PUPPET_GRIPPER_QPOS(
        NORMALIZE_MASTER_GRIPPER_JOINT(right_gripper_joint)
    )
    print(
        "left_gripper_joint", left_gripper_joint, "left_gripper_qpos", left_gripper_qpos
    )
    action[6] = left_gripper_qpos
    action[7 + 6] = right_gripper_qpos
    return action


def sim_teleop():
    """Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work."""
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    # source of data
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

    env = make_aloha_env(task_name="transfer_cube")
    obs = env.reset()
    episode = [obs]
    actions = []
    plt.ion()
    ax = plt.subplot()
    plt_img = ax.imshow(env.render())

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        obs, reward, terminated, truncated, info = env.step(action)

        episode.append(obs)
        actions.append(action)

        plt_img.set_data(env.render())
        plt.pause(0.02)
    plt.close()


if __name__ == "__main__":
    sim_teleop()
