# Wheelbipe V14-6：复杂地形微调

当前分支以队友已验证的平地策略 **`flat_seed42 / model_2000.pt`** 为起点。当前小目标是复现平地运动，之后再用同一份冻结权重测轻坡上行、下行和平地，为 PPO 微调建立对照。**已重算历史评测，新的平地仿真待运行。**

| 想看什么 | 从这里打开 |
| --- | --- |
| 平地基线为什么选 2000、如何复现 | [平地复现说明](docs/v14-6-flat-baseline.md) → [启动脚本](scripts/reproduce_v14_6_flat.sh) |
| 队友的 Observation、Action 和训练流程 | [队友源码阅读指南](references/teammate-unified-2026-11-09/README.md) → [35 维观测与 6 维动作的导出样例](references/teammate-unified-2026-11-09/examples/export_724/input_contract.json) |
| 地形、复位和课程如何接入 | [统一地形环境](references/teammate-unified-2026-11-09/project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py) |
| 自己之前写的平地任务 | [旧骨架说明](docs/archive/standalone-flat-scaffold.md) → [环境配置](source/wheelleg_lab/tasks/flat/env_cfg.py) / [环境接口](source/wheelleg_lab/tasks/flat/env.py) |
| 看机器人模型 | [V14-6 资产说明](assets/robot/v14_6/README.md) → [场景检查脚本](scripts/open_robot.py) |

```text
wheelleg_lab/
├── docs/                                  # 当前复现说明；archive/ 存旧开发说明
├── scripts/                               # 平地复现、场景检查两个入口
├── references/teammate-unified-2026-11-09/  # 队友统一地形的源码阅读快照
├── source/wheelleg_lab/                    # 之前从零搭建的骨架，暂停开发
└── assets/robot/v14_6/                     # 仓库里的 V14-6 检查用资产
```

平地复现使用 pi 上的原训练项目、训练资产和权重，输出到服务器个人日志目录。仓库中的队友快照用于读代码；本地检查用 USD 与旧平地骨架都不参与这次复现。原主工作区的 `training/flat_baseline/` 是旧的本地对照包，不随 Git 分支检出。

下一步见[任务清单](TASKS.md)，协作约定见 [AGENT.md](AGENT.md)。
