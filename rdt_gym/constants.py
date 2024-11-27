### Simulation envs fixed constants
DT = 0.02

# 16 JOINTS
JOINTS = [
    # absolute joint position
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_angle",
    "left_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "left_arm_gripper/left_finger",
    "left_arm_gripper/right_finger",
    # absolute joint position
    "right_arm_waist",
    "right_arm_shoulder",
    "right_arm_elbow",
    "right_arm_forearm_roll",
    "right_arm_wrist_angle",
    "right_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "right_arm_gripper/left_finger",
    "right_arm_gripper/right_finger",
]

# 14 ACTUATORS. Two Fingers of each gripper are controlled by one actuator
ACTUATORS = [
    # position and quaternion for end effector
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_angle",
    "left_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "left_arm_gripper",
    "right_arm_waist",
    "right_arm_shoulder",
    "right_arm_elbow",
    "right_arm_forearm_roll",
    "right_arm_wrist_angle",
    "right_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "right_arm_gripper",
]


# In Mujoco Simulation. The sign of qops in two fingers are same, it is opposite to the aloha hardware
PUPPET_GRIPPER_QPOS_OPEN = 0.035
PUPPET_GRIPPER_QPOS_CLOSE = 0.0084

PUPPET_GRIPPER_QPOS_MID = (PUPPET_GRIPPER_QPOS_OPEN + PUPPET_GRIPPER_QPOS_CLOSE) / 2

# initial joints of aloha. len = 16. last two joint of gripper are right and left finger respectively
# same as keyframe_ctrl.xml
INITIAL_ARM_QPOS = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    PUPPET_GRIPPER_QPOS_OPEN,
    PUPPET_GRIPPER_QPOS_OPEN,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    PUPPET_GRIPPER_QPOS_OPEN,
    PUPPET_GRIPPER_QPOS_OPEN,
]

############################ Helper functions ############################


def NORMALIZE_PUPPET_GRIPPER_QPOS(qpos):
    return (qpos - PUPPET_GRIPPER_QPOS_CLOSE) / (
        PUPPET_GRIPPER_QPOS_OPEN - PUPPET_GRIPPER_QPOS_CLOSE
    )


# x [0, 1]
def UNNORMALIZE_PUPPET_GRIPPER_QPOS(x):
    return (
        x * (PUPPET_GRIPPER_QPOS_OPEN - PUPPET_GRIPPER_QPOS_CLOSE)
        + PUPPET_GRIPPER_QPOS_CLOSE
    )
