from __future__ import annotations  # 允许在类型注解中使用类名字符串
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..aloha_env import AlohaEnv  # 仅在类型检查时导入


class AlohaTask(ABC):
    # used by MujocoEnv.set_state(init_qpos, init_qvel)
    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_reward(self, env: AlohaEnv):
        pass

    @abstractmethod
    def get_xml_file(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_task_name() -> str:
        pass
