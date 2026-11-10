# Wheelbipe V14-6：复杂地形微调

本分支直接以队友的统一地形项目作为开发基线。当前先用已验证的平地策略 `flat_seed42 / model_2000.pt` 复现原表现，再做轻坡上行、下行和平地对照。新的平地仿真仍待运行。

## 日常命令

从仓库根目录使用 [tools/wheelleg.sh](tools/wheelleg.sh)。把 `--dry-run` 放在子命令之前可先查看实际转发的命令；它不连接服务器、不检查资产，也不启动仿真。

| 入口 | 在哪里执行 | 调用什么 |
| --- | --- | --- |
| `baseline` | Mac | [平地复现脚本](tools/reproduce_v14_6_flat.sh)，固定原权重、配置与指令。加 `--check` 只做只读预检。 |
| `train` | pi 的个人检出目录 | [train.py](project/scripts/rsl_rl/train.py)，从指定 checkpoint 初始化 PPO 微调。迭代数、任务和实验配置显式指定。 |
| `eval` | pi 的个人检出目录 | [validate.py](tools/validate.py)，输出地形和速度评估的 `summary.json`。 |
| `view video` | pi 的个人检出目录 | [record.py](tools/record.py)，输出 `policy.mp4` 与回放结果。 |
| `view interactive` | 当前暂不可用 | 旧 Own 回放入口要求 Sim 5.1，本分支使用 6.2。该命令只提示，不启动仿真。 |

在 **Mac** 上先检查计划使用的 GPU，例如 4：

```bash
bash tools/wheelleg.sh baseline 4 --check
bash tools/wheelleg.sh baseline 4
```

其余命令需先把本分支放到 **pi 的个人目录**，配齐训练 USD、复位姿态库，以及 checkpoint 和同级 `params/`。下面是命令模板：先替换两个权重路径、确认 GPU 空闲，并按该次实验确认 stage 和积分子步。示例不代表已验证平地权重可以直接迁移到统一地形。

```bash
FLAT=/absolute/path/to/model_2000.pt
UNIFIED=/absolute/path/to/matching_unified_checkpoint.pt
export WHEELBIPE_GPU=4 WHEELBIPE_STAGE=3 WHEELBIPE_INTEGRATION_SUBSTEPS=2

WHEELBIPE_EXPERIMENT=unified_locomotion bash tools/wheelleg.sh train \
  --task Robotics-Wheelbipe-Own-Explore-v0 --checkpoint "$FLAT" --checkpoint_architecture \
  --num_envs 8 --max_iterations 1 --seed 42 --device cuda:0 --viz none --run_name slope_smoke

bash tools/wheelleg.sh eval --root "$PWD" --profile unified_locomotion \
  --checkpoint "$UNIFIED" --output "$PWD/runs/unified_eval" \
  --level 0 --seed 71903 --seconds 8 --device cuda:0 --viz none

bash tools/wheelleg.sh view video --profile unified_locomotion \
  --checkpoint "$UNIFIED" --output "$PWD/runs/unified_video" \
  --terrain-column 0 --terrain-level 0 --steps 300 --seed 71903 --device cuda:0

bash tools/wheelleg.sh view interactive
```

统一入口沿用原启动器和底层兼容性校验。启动前还会检查 checkpoint 的配套文件和资产哈希；评估与录制要求配套的统一地形 profile 和非归一化 ELU 策略。平地原权重的原条件评估请用 `baseline`。微调允许选择新的 profile，能否加载仍由训练入口检查；只有恢复同一实验才显式使用 `--resume_training`。统一入口不会代选权重、种子、地形或训练资源。

## 源码索引

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
