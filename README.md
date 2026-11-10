# Wheelbipe V14-6：复杂地形微调

本分支直接以队友的统一地形项目作为开发基线。当前先用已验证的平地策略 `flat_seed42 / model_2000.pt` 复现原表现，再做轻坡上行、下行和平地对照。新的平地仿真仍待运行。

| 入口 | 职责 |
| --- | --- |
| [project/source/](project/source/) | 主源码：`agent_tasks` 定义任务，`agent_world` 定义机器人和执行器，`agent_rl` 提供 RL 接口和算法扩展。 |
| [project/scripts/rsl_rl/train.py](project/scripts/rsl_rl/train.py) | RSL-RL 训练入口。 |
| [profiles.json](profiles.json) | 队友统一地形的实验预设，按原路径供任务读取。 |
| [tools/](tools/) | 启动、评测、监控和恢复工具。[平地复现脚本](tools/reproduce_v14_6_flat.sh)从 Mac 调用 pi 上的原评测。 |
| [docs/v14-6-flat-baseline.md](docs/v14-6-flat-baseline.md) | 平地基线来源、复现命令和验收标准。 |
| [docs/upstream/](docs/upstream/) | 队友历次实验的配置、评估与来源记录。里面的 `plan.json` 是队友的旧运行计划。 |

做坡面适配时，先看 [unified_terrain.py](project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_terrain.py) 如何生成地形，再看 [unified_environment.py](project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py) 如何接入复位、终止和课程。观测与动作可对照[导出契约](docs/upstream/examples/export_724/input_contract.json)；它对应 724 号权重，不是平地 2000 号权重的验证记录。

训练用 USD、重置姿态库和 checkpoint 没有纳入 Git。运行统一地形代码前，需要在自己的服务器检出目录配齐同版本资产和权重，再建立自己的运行计划。当前平地复现继续读取 pi 上的原始训练项目和权重，结果写入个人日志目录。

当前步骤见[任务清单](TASKS.md)。旧独立项目保存在 [main 分支](https://github.com/SilverLi123/wheelleg-robot-RL/tree/main)。
