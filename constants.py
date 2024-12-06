from os import path

### Task parameters
DATA_DIR = "<put your data dir here>"
SIM_TASK_CONFIGS = {
    "sim_transfer_cube": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
    },
}


XML_DIR = path.join(path.dirname(__file__), "../aloha_xml")

### Simulation envs fixed constants
DT = 0.02

# 16 JOINTS
ALOHA_JOINTS = [
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
ALOHA_ACTUATORS = [
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

INITIAL_CUBES_QPOS = [
    # red cube
    0,
    0,
    0.02,
    1,
    0,
    0,
    0,
    # green cube
    0.2,
    0.2,
    0.02,
    1,
    0,
    0,
    0,
    # blue cube
    0.2,
    -0.2,
    0.02,
    1,
    0,
    0,
    0,
    # yellow cube
    -0.2,
    0.2,
    0.02,
    1,
    0,
    0,
    0,
    # purple cube
    -0.2,
    -0.2,
    0.02,
    1,
    0,
    0,
    0,
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


# Gripper joint limits (qpos[6])
# They are different according to your real robot
MASTER_GRIPPER_JOINT_OPEN = 0.85136
MASTER_GRIPPER_JOINT_CLOSE = -0.056757
PUPPET_GRIPPER_JOINT_OPEN = -1.477224
PUPPET_GRIPPER_JOINT_CLOSE = -2.584758

PUPPET_START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    PUPPET_GRIPPER_JOINT_OPEN,
    -PUPPET_GRIPPER_JOINT_OPEN,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    PUPPET_GRIPPER_JOINT_OPEN,
    -PUPPET_GRIPPER_JOINT_OPEN,
]

MASTER_START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0.0,
    MASTER_GRIPPER_JOINT_OPEN,
    -MASTER_GRIPPER_JOINT_OPEN,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0.0,
    MASTER_GRIPPER_JOINT_OPEN,
    -MASTER_GRIPPER_JOINT_OPEN,
]


def NORMALIZE_MASTER_GRIPPER_JOINT(joint):
    return (joint - MASTER_GRIPPER_JOINT_CLOSE) / (
        MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE
    )
