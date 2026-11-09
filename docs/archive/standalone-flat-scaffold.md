# wheelleg_lab

> 历史说明：记录之前从零搭建的平地任务。当前分支的入口见 [根 README](../../README.md)；下方的开发顺序和验证状态仅代表当时。

从基础开始学习和开发 Wheelbipe V14--6 的 Isaac Lab 仿真与强化学习项目。主项目按 SCUTRobotLab 的结构搭建自己的平地任务，Observation 和 Action 由用户设计。队友的平地 PPO 保留在本机的 `training/flat_baseline/` 作对照，该目录不随 Git 上传，主项目不依赖它。

## 目录用途

```text
wheelleg_lab/
├── assets/robot/v14_6/  # 原样复制的 V14--6 USD 模型及用途说明
├── scripts/
│   └── open_robot.py   # 在已启动的 Isaac Sim 中打开模型，保持时间线停止
├── source/wheelleg_lab/
│   ├── __init__.py     # Python 包入口，导入时不启动仿真
│   ├── assets.py       # 根据文件位置计算模型路径，无物理参数
│   └── tasks/
│       └── flat/
│           ├── env_cfg.py # 平地、机器人资产和待填写的训练配置
│           └── env.py     # 用户实现观测和动作的环境接口
├── scenes/             # 空；之后保存自己创建的仿真场景
├── logs/               # 空；之后存放自己的运行记录
├── AGENT.md            # 边学边做、每步 review 的协作约定
└── TASKS.md            # 当前小目标与学习路线
```

## 参考仓库，采用哪些部分

参考：[scutrobotlab/wheeled-legged_RL 的目录说明](https://github.com/scutrobotlab/wheeled-legged_RL#project-structure)。

| 参考仓库的职责 | 本项目当前安排 |
| --- | --- |
| `agent_world`：机器人资产等 | USD 放 `assets/`，文件位置统一写在 `wheelleg_lab/assets.py` |
| `agent_tasks`：任务与环境 | `wheelleg_lab/tasks/flat/`，沿用 `env_cfg.py` / `env.py` 的分工 |
| `scripts`：运行入口 | 当前只有 `scripts/open_robot.py` |
| `agent_rl`：算法扩展 | 后续按需使用已有标准 RSL-RL PPO，当前不复制算法 |

这一步只借鉴职责分工，不复制完整训练系统。参考仓库文档面向 Isaac Sim 4.5/5.1、Python 3.11；我们的目标是已调查的共享 Sim 6.2 / Lab 3.0 环境，不照搬其安装限制或可编辑安装命令。

## 自己的平地任务从哪里开始

先看 `source/wheelleg_lab/tasks/flat/env_cfg.py`，再看同目录的 `env.py`。配置声明一块平地和一台 V14--6，使用 robot-only USD。执行器、物理步长、控制频率和空间维度仍待填写。该 USD 不含原生气弹簧驱动，之后要单独验证气弹簧施力和复位姿态。

| 你要设计的部分 | 接口 | 要先定下来的内容 |
| --- | --- | --- |
| Observation | `_get_observations()` | 策略看哪些状态、顺序、坐标系、单位和归一化 |
| Action | `_pre_physics_step(actions)` | 每个动作代表什么、控制哪些关节、如何缩放和限幅 |
| 执行 Action | `_apply_action()` | 在物理子步写入哪种目标（位置、速度或力矩），用什么执行器 |

`WheelbipeFlatEnv` 是抽象类。之后在自己的具体子类里实现这些接口，并补齐奖励、终止和复位。当前不注册任务或接训练入口，因此不会误跑空观测、零动作或占位奖励。

## 模型加载脚本的 review 顺序

1. `source/wheelleg_lab/__init__.py`：理解 Python 包入口，确认没有启动或注册副作用。
2. `source/wheelleg_lab/assets.py`：理解 `Path(__file__)`、项目根目录与两种 USD；这里没有电机参数。
3. `scripts/open_robot.py`：理解临时导入路径、`async def` / `await`、先暂停再加载，以及 `finally` 保持暂停。

输入是本地 USD 文件，主要输出是已打开的场景及控制台元信息。默认使用带地面与原生气弹簧的检查场景；脚本不叠加外部弹簧、不发送电机指令、不调用物理步进或保存场景。能打开模型不等于已验证其平衡能力。

## 最小检查方式

在 **Mac 本地终端**运行，只检查文件路径，不需要安装 Isaac Sim：

```bash
cd /Users/silver/RL/wheelleg_lab
PYTHONPATH=source python3 -B -c 'from wheelleg_lab.assets import ROBOT_USD, INSPECTION_SCENE_USD; print(ROBOT_USD.is_file(), INSPECTION_SCENE_USD.is_file())'
```

预期输出 `True True`。这只是文件存在检查，不读取 USD 物理结构或运行机器人。

**下面是 review 通过且手动同步之后的运行方式，本轮尚未执行。** 在服务器本人已启动的 Isaac Sim 的 **Script Editor** 中运行：

```python
import runpy
runpy.run_path(
    "/data/wheeled-legged/projects/wheelleg_lab/scripts/open_robot.py",
    run_name="__main__",
)
```

该操作会替换当前打开的场景，运行前先处理本人当前场景未保存的改动。预期看到 V14--6 检查场景，时间线停止；控制台输出根节点、单位（1 场景单位等于 1 米）与上轴 Z。可以观察模型，但本步先不点击播放。遇到加载异常，不能当作通过。

不要直接用普通 Python 执行 `open_robot.py` 来启动应用：它会明确提示需要已有 Isaac Sim。脚本通过 `source` 路径加载本项目，不向共享环境安装包。

## 复用什么

- 继续复用服务器现有 Python 环境：`/data/isaac620-migration-20261103/env620/bin/python`，不重新安装 Isaac Sim、Isaac Lab 或 RSL-RL。
- 主模型为 V14--6，名义总质量约 20.66189 kg。模型来自队友交付，不在整理时修改结构、惯量、关节或气弹簧参数。
- 完整队友材料保存在项目外：`/Users/silver/RL/references/teammate_v14_6_20261108_224209/`。其中有源码、配套 profiles、权重、评测报告、URDF/网格和 V14--5 XML。
- 需要某个功能时先读懂对应参考，再决定是否复用；不直接继承整套复杂训练任务。

## 当前边界

本轮只编写和检查本地骨架。服务器 `/data/wheeled-legged/projects/wheelleg_lab` 仍保留上一次完整复用包，尚未同步成此最小结构。服务器修改由用户执行，后续按需手动同步，不配置自动同步。

本轮检查通过：自有源码语法、任务包导入不加载 Isaac Lab 或 Gymnasium、机器人 USD 路径存在。另只读核对了服务器 Lab 3 源码的场景创建顺序与动作钩子调用顺序。此前的模型加载入口已单独检查过路径解析与错误提示。

没有启动 Isaac Sim、物理步进、策略回放或训练。异步打开场景、真实时间线状态和控制台 USD 元信息仍待在真实应用中验证；本地检查不能替代它们。交付方已有报告保存在参考快照内，不当作本项目重新测试的结果。

本轮目录变更及完整快照哈希记录：`/Users/silver/RL/docs/reorganizations/2026-11-08-minimal-workspace.json`。更早的 V14--5 开发目录备份仍在 `/Users/silver/RL/backups/wheelleg_lab_before_20261108_224209/`。

## 下一小步

先 review 自己的 `env_cfg.py` 与 `env.py`。用户确认 Observation、Action 设计后，再接上执行器并验证单机器人环境。队友的训练基线可以独立使用，不是自有任务的启动入口。
