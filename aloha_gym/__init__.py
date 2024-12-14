import gymnasium as gym


from .aloha_env import AlohaEnv, SIM_CAMS
from .task.task_none import NoneTask
from .task.task_transfer_cube import TransferCubeTask


def make_aloha_env(task_name: str, **kwargs) -> AlohaEnv:
    assert task_name in [NoneTask.get_task_name(), TransferCubeTask.get_task_name()]
    gym.register(
        id="aloha",
        entry_point=AlohaEnv,
    )
    if task_name == NoneTask.get_task_name():
        task = NoneTask()
        env = gym.make("aloha", task=task)
        return env
    elif task_name == TransferCubeTask.get_task_name():
        task = TransferCubeTask()
        env = gym.make("aloha", task=task, **kwargs)
        return env
    else:
        raise NotImplementedError()
