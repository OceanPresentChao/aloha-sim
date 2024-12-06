import numpy as np

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
