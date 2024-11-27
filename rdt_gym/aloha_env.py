import gymnasium
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import math
import mujoco

from .constants import (
    UNNORMALIZE_PUPPET_GRIPPER_QPOS,
    NORMALIZE_PUPPET_GRIPPER_QPOS,
    DT,
    INITIAL_ARM_QPOS,
    JOINTS,
    ACTUATORS,
)


class AlohaEnv(MujocoEnv):
    def __init__(
        self,
        xml_file: str = "scene.xml",
        frame_skip: int = 5,
        observation_width: int = 640,
        observation_height: int = 480,
        render_mode: str = "rgb_array",
        **kwargs,
    ):
        self.observation_width = observation_width
        self.observation_height = observation_height
        ALOHA_OBS_IMAGE = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, 3),
            dtype=np.uint8,
        )
        self.observation_space = gymnasium.spaces.Dict(
            {
                "qpos": gymnasium.spaces.Box(
                    low=-math.pi,
                    high=math.pi,
                    shape=(len(ACTUATORS), 1),
                    dtype=np.float64,
                ),
                "qvel": gymnasium.spaces.Box(
                    low=-math.inf,
                    high=math.inf,
                    shape=(len(ACTUATORS),),
                    dtype=np.float64,
                ),
                "images": gymnasium.spaces.Dict(
                    {
                        "top": ALOHA_OBS_IMAGE,
                        "low": ALOHA_OBS_IMAGE,
                        "left_wrist": ALOHA_OBS_IMAGE,
                        "right_wrist": ALOHA_OBS_IMAGE,
                    }
                ),
            }
        )
        self.action_space = gymnasium.spaces.Box(
            low=-math.pi, high=math.pi, shape=(len(ACTUATORS),), dtype=np.float64
        )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(100),
        }
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=self.observation_space,
            render_mode=render_mode,
            width=self.observation_width,
            height=self.observation_height,
            **kwargs,
        )

        assert self.render_mode in self.metadata["render_modes"]

    def get_qpos(self):
        # len: len(JOINTS)
        qpos_raw = self.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [left_qpos_raw[6]]
        right_gripper_qpos = [right_qpos_raw[6]]
        qpos = np.concatenate(
            [left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos]
        )
        return qpos

    def get_qvel(self):
        # len: len(JOINTS)
        qvel_raw = self.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [left_qvel_raw[6]]
        right_gripper_qvel = [right_qvel_raw[6]]
        qvel = np.concatenate(
            [left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel]
        )
        return qvel

    def get_image(
        self,
        camera_name: str,
    ):
        assert camera_name in [
            "wrist_cam_left",
            "wrist_cam_right",
            "overhead_cam",
            "worms_eye_cam",
            "teleoperator_pov",
            "collaborator_pov",
        ]
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        image = self.mujoco_renderer._get_viewer(
            render_mode=self.render_mode,
        ).render(
            camera_id=camera_id,
            render_mode=self.render_mode,
        )
        return image

    def get_obs(self):
        obs = {}
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["images"] = {}
        obs["images"]["top"] = self.get_image("overhead_cam")
        obs["images"]["low"] = self.get_image("worms_eye_cam")
        obs["images"]["left_wrist"] = self.get_image("wrist_cam_left")
        obs["images"]["right_wrist"] = self.get_image("wrist_cam_right")
        return obs

    def get_reward(self):
        return 0

    def render(self):
        return self.get_image("collaborator_pov")

    def reset_model(self):
        # key_ctrl = self.model.key_ctrl[0 * self.model.nu : (0 + 1) * self.model.nu]
        init_qpos = np.array(INITIAL_ARM_QPOS, dtype=np.float64)
        init_qvel = np.zeros(shape=(len(JOINTS),), dtype=np.float64)
        self.set_state(init_qpos, init_qvel)
        return self.get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self.get_obs()
        reward = self.get_reward()
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
