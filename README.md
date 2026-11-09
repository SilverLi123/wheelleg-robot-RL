# Wheelbipe V14-6：复杂地形微调

当前分支以队友已验证的平地策略 **`flat_seed42 / model_2000.pt`** 为起点。当前小目标是复现平地运动，之后再用同一份冻结权重测轻坡上行、下行和平地，为 PPO 微调建立对照。**已重算历史评测，新的平地仿真待运行。**

| 想看什么 | 从这里打开 |
| --- | --- |
| 平地基线为什么选 2000、如何复现 | [平地复现说明](docs/v14-6-flat-baseline.md) → [启动脚本](scripts/reproduce_v14_6_flat.sh) |
| 队友的 Observation、Action 和训练流程 | [队友源码阅读指南](references/teammate-unified-2026-11-09/README.md) → [35 维观测与 6 维动作的导出样例](references/teammate-unified-2026-11-09/examples/export_724/input_contract.json) |
| 地形、复位和课程如何接入 | [统一地形环境](references/teammate-unified-2026-11-09/project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py) |

```text
wheelleg_lab/
├── docs/                                  # 平地基线和复现说明
├── scripts/                               # 平地复现入口
└── references/teammate-unified-2026-11-09/  # 队友统一地形的源码阅读快照
```

平地复现使用 pi 上的原训练项目、训练资产和权重，输出到服务器个人日志目录。仓库中的队友快照用于读代码；从零搭建的独立项目保存在 [main 分支](https://github.com/SilverLi123/wheelleg-robot-RL/tree/main)。

下一步见[任务清单](TASKS.md)，协作约定见 [AGENT.md](AGENT.md)。
