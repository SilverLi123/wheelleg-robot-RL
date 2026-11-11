# Wheelbipe V14-6：复杂地形微调

本分支直接以队友的统一地形项目作为开发基线。平地数值复现已通过，并通过冻结策略录像检查了站立、前进、后退和转向。

2026-11-11 已在平地以及 6%/10%/14%/18% 上下坡组成的混合地形上完成一次 60 次 PPO 微调。256 个环境共训练 368,640 timesteps，使用 GPU 3，耗时 15 分 44 秒。三种子评估显示新模型能稳定通过全部 12 秒地形探测，但速度跟踪和横向偏移没有稳定超过原 baseline，因此最终推荐仍使用原 `model_50.pt`。完整数据见[混合坡度实验报告](reports/mixed_slopes_20261111/README.md)。

| 入口 | 职责 |
| --- | --- |
| [project/source/](project/source/) | 主源码：`agent_tasks` 定义任务，`agent_world` 定义机器人和执行器，`agent_rl` 提供 RL 接口和算法扩展。 |
| [project/scripts/rsl_rl/train.py](project/scripts/rsl_rl/train.py) | RSL-RL 训练入口。 |
| [profiles.json](profiles.json) | 队友统一地形的实验预设，按原路径供任务读取。 |
| [tools/](tools/) | 启动、评测、监控和恢复工具。[平地复现脚本](tools/reproduce_v14_6_flat.sh)从 Mac 调用 pi 上的原评测。 |
| [docs/v14-6-flat-baseline.md](docs/v14-6-flat-baseline.md) | 平地基线来源、复现命令和验收标准。 |
| [docs/upstream/](docs/upstream/) | 队友历次实验的配置、评估与来源记录。里面的 `plan.json` 是队友的旧运行计划。 |
| [reports/mixed_slopes_20261111/](reports/mixed_slopes_20261111/) | 本次混合坡度训练、三种子评估、曲线和模型选择证据。 |

做坡面适配时，先看 [unified_terrain.py](project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_terrain.py) 如何生成地形，再看 [unified_environment.py](project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py) 如何接入复位、终止和课程。观测与动作可对照[导出契约](docs/upstream/examples/export_724/input_contract.json)；它对应 724 号权重，不是平地 2000 号权重的验证记录。

训练用 USD、重置姿态库和 checkpoint 没有纳入 Git。运行统一地形代码前，需要在自己的服务器检出目录配齐同版本资产和权重，再建立自己的运行计划。当前平地复现继续读取 pi 上的原始训练项目和权重，结果写入个人日志目录。

当前步骤见[任务清单](TASKS.md)。旧独立项目保存在 [main 分支](https://github.com/SilverLi123/wheelleg-robot-RL/tree/main)。

轻坡评估使用 [light_slope.py](tools/light_slope.py) 调用原有录像逻辑。把它和同目录的 `record.py` 放在 pi 上自己的检出目录；从该目录执行 `python3 tools/light_slope.py up 6`（`up` 可换为 `flat` 或 `down`，`6` 换成当前空闲 GPU）。每次只跑一组，结果保存在个人日志目录。三组都以 +0.50 m/s、零转向、静止起步回放 6 秒。

`policy.mp4` 用来观察运动，`summary.json` 记录速度误差、重置与世界坐标俯仰范围。Isaac Lab 3.0 的 `root_state_w` 按 `(x, y, z, w)` 存储四元数；计算函数按 `(w, x, y, z)` 接收，读取完整 root state 时会显式重排。`initial_pose_status` 检查机器人是否面向世界 `+X`，`direction_status` 检查第一次尝试是否实际向 `+X` 前进。坡面组另记录沿预计路线测出的坡度和实际地面高差，并通过 `slope_status` 判断是否走过预期方向。

只有 `status`、`initial_pose_status` 和 `slope_status` 均满足预期时才判定通过。若 `first_attempt_forward_m` 为负，立即停止，不得继续用互换 `up/down` 标签的方式掩盖朝向错误。DR（域随机化）保持开启；三组使用相同 profile、seed、checkpoint、指令和步数，但 DR 的随机量会随地形交互继续演化，因此结果是可比较的重复实验，不是严格的配对实验。

混合坡度训练与评估直接复用：

```bash
bash tools/train_mixed_slopes.sh train <空闲GPU编号>
bash tools/eval_mixed_slopes.sh <空闲GPU编号> <checkpoint> 600 <seed>
python3 tools/summarize_mixed_slopes_eval.py --raw-dir <原始报告目录> --output-dir <输出目录>
bash tools/record_mixed_slopes.sh <空闲GPU编号> <checkpoint> <输出目录> 800
```

长训练应从服务器 `tmux` 会话中执行。脚本会检查显存、GPU 利用率和可选的 checkpoint SHA-256，并把缓存、临时文件和日志写入个人工作区。
