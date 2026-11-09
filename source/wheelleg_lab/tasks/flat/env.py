"""V14--6 平地环境接口。用户在具体子类中实现策略输入、输出和任务规则。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import DirectRLEnv

from .env_cfg import WheelbipeFlatEnvCfg

if TYPE_CHECKING:
    from torch import Tensor


# ABC 禁止实例化尚未实现接口的环境，防止骨架被当成可训练任务。
class WheelbipeFlatEnv(DirectRLEnv, ABC):
    cfg: WheelbipeFlatEnvCfg

    def _setup_scene(self) -> None:
        # Lab 3 已按 scene 配置创建资产；这里只取实例，避免再次创建或克隆机器人。
        self.robot = self.scene["robot"]

    @abstractmethod
    def _get_observations(self) -> dict[str, Tensor]:
        """用户设计：读取机器人状态并组装观测，返回以 policy 为键的批量张量。

        向量观测的第一维是环境数；字段顺序、坐标系、单位和缩放由你定义。
        """
        raise NotImplementedError

    @abstractmethod
    def _pre_physics_step(self, actions: Tensor) -> None:
        """用户设计：每个策略步读取动作，将它转换成控制目标并缓存。

        先决定动作各维的含义，再决定缩放、限幅和控制模式；这里不写入关节命令。
        """
        raise NotImplementedError

    @abstractmethod
    def _apply_action(self) -> None:
        """用户设计：在物理子步中把缓存目标写入机器人。

        框架在此后提交命令并推进物理；这里不自行步进仿真。
        """
        raise NotImplementedError

    @abstractmethod
    def _get_rewards(self) -> Tensor:
        """后续任务设计：每个环境返回一个奖励值。"""
        raise NotImplementedError

    @abstractmethod
    def _get_dones(self) -> tuple[Tensor, Tensor]:
        """后续任务设计：分别返回任务终止与超时，避免混淆两者。"""
        raise NotImplementedError

    @abstractmethod
    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """后续任务设计：只重置指定环境，并调用基类处理场景和回合计数。"""
        super()._reset_idx(env_ids)
