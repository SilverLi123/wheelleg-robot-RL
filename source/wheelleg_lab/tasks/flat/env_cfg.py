"""平地任务的场景和配置；参考 SCUTRobotLab 把配置与每步逻辑分开。"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from wheelleg_lab.assets import ROBOT_USD


@configclass
class WheelbipeFlatSceneCfg(InteractiveSceneCfg):
    # 全部环境共享一块地面；机器人则放到各自的环境命名空间。
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        # 这里用 robot-only 交付资产。它不带原生气弹簧驱动，后续需验证施力方案。
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD)),
        # 动作设计确定了主动关节与控制模式后，再填写对应执行器。
        actuators=MISSING,
    )


@configclass
class WheelbipeFlatEnvCfg(DirectRLEnvCfg):
    # 第一阶段只检查一台机器人。先按独立实例处理闭环，不启用物理复制优化。
    scene: WheelbipeFlatSceneCfg = WheelbipeFlatSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=False
    )

    # 不沿用参考机器人的观测/动作维数和物理周期。
    # action_space、observation_space、decimation、episode_length_s 沿用父类的必填声明。
    # sim.dt 是物理步长（秒）；策略周期是 sim.dt * decimation。
    sim: sim_utils.SimulationCfg = MISSING
