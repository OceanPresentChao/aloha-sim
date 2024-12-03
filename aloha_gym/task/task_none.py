from __future__ import annotations  # 允许在类型注解中使用类名字符串
from .base import AlohaTask
from os import path
import numpy as np
from typing import TYPE_CHECKING

from ..constants import XML_DIR, INITIAL_ARM_QPOS

if TYPE_CHECKING:
    from ..aloha_env import AlohaEnv  # 仅在类型检查时导入


class NoneTask(AlohaTask):
    def get_initial_state(self):
        init_qpos = np.concatenate([INITIAL_ARM_QPOS], dtype=np.float64)
        init_qvel = np.zeros(shape=(len(INITIAL_ARM_QPOS),), dtype=np.float64)
        return init_qpos, init_qvel

    def get_reward(self, env: AlohaEnv):
        return 0

    def get_xml_file(self):
        return path.join(XML_DIR, "scene.xml")
