# Wheelbipe V14--6 队友代码阅读快照

这是 pi 上 `/data/wheelbipe-unified-parallel-20261108` 的只读代码快照，供学习和讨论。保留原代码的目录关系、注释和许可证。复制时保存的训练配置来自 `model_1500.pt` 同一轮运行，随附的评测样例来自 `model_1400.pt`，导出的观测/动作契约来自 `model_724.pt`。这三个编号对应不同证据。

| 目录或文件 | 从这里看什么 |
| --- | --- |
| [训练入口](project/scripts/rsl_rl/train.py) | 创建环境、包装为 RSL-RL 接口、调用 Runner |
| [训练参数](examples/training_params/env.yaml) 与 [PPO 参数](examples/training_params/agent.yaml) | 该轮真正保存的配置 |
| [观测和动作契约](examples/export_724/input_contract.json) | 35 维观测的切片、六轴动作顺序及缩放 |
| [统一地形环境](project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py) | 地形、指令采样、复位与训练统计如何接入 |
| [V14--6 环境](project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/env.py) 与 [坐标转换](project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/coordinates.py) | 按关节名映射、腿目标限幅、气弹簧施力、策略坐标与 PhysX 坐标 |
| [父类环境](project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py) | 实际拼接观测、解码动作和写入关节命令的位置 |
| [评测样例](examples/evaluation/summary.json) | 五类地形的指令、速度误差和失败计数 |
| [来源清单](SNAPSHOT.json) | 每个复制文件的服务器路径和 SHA256 |

建议第一轮只追一帧：从 Actor 的观测拼接开始，跟到 6 维输出分别写入位置和速度目标。第二轮可以用评测样例理解坡面失败，再追终止判定、复位和奖励。

可在网页版 ChatGPT 中粘贴：

> 带我读这个仓库的 `references/teammate-unified-2026-11-09/`。从这个目录的 README 开始，每次只讲一条调用链。第一轮请按真实源码带我追一帧：35 维 Observation 的字段从哪里读，6 维 Action 如何写到四腿与两轮。引用文件路径和函数名，区分保存的训练配置和较早的导出契约。讲完留一个理解检查，先不要改代码。

这是阅读副本。权重、USD、重置数据、训练日志、缓存、运行环境和视频没有上传；保留的脚本和配置可能引用 pi 上的原始路径，不宜从此目录直接运行。
