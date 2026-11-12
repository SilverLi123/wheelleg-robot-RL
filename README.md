# Wheelbipe V14-6：复杂地形微调

基于队友稳定平地 PPO 策略，在 Isaac Sim 6.2 / Isaac Lab 3 / RSL-RL 中开发复杂地形能力，暂不实机部署。当前开发进度见 [TASKS](TASKS.md)，工具职责见 [tools/README](tools/README.md)。

## 当前状态与来源

- 平地 `model_2000.pt` 数值复现和动作录像已有完成记录。纵向混合坡度试验及能力退化结论见 [2026-11-11 报告](reports/mixed_slopes_20261111/README.md)。
- 原随机地图的接缝、高度奖励、课程和连续路线失败见 [地形核查报告](reports/random_terrain_20261112/README.md)。它是历史证据，不表示整图已通过。
- 后续三角面高度修复、4096×40 训练和候选录像发生在另一独立 clone `/Users/silver/RL/wheelleg_lab` 与 pi 个人目录，**当前 IDE 检出尚未合入这些源码修复**。本轮只读核实了服务器结果，来源、哈希及条件见 [维护审计](reports/repo_maintenance_20261112/README.md)。先核对同步，再复现实验。
- `model_39.pt` 是已训练候选，不能标成最佳：已有九地块 27/27 结果来自一个 seed 下三个 repeat，训练物理为 400 Hz、评估为 200 Hz；尚需同条件能力保留与整图验证。`model_30` 的现有平均横移还略小于 `model_39`。
- 本地 MP4 在 `/Users/silver/RL/wheelleg_lab/videos/final39_showcase_20261112/`；后续默认交付 MP4。二维 HTML 视频壳不提供自由三维相机。

## 四个日常入口

从仓库根目录运行 `bash tools/wheelleg.sh ...`。`--dry-run` 放在子命令前，仅显示转发命令，不连接服务器或启动仿真；不代表兼容性预检。

| 入口 | 执行地点 | 实际调用与前提 |
| --- | --- | --- |
| `baseline GPU [--check]` | Mac | `reproduce_v14_6_flat.sh`，SSH 校验队友原 `model_2000.pt` 配套配置；`--check` 只读。不加该参数会回放。 |
| `train ...` | pi 个人检出 | `run.sh` → `project/scripts/rsl_rl/train.py`。显式指定 task、checkpoint、profile、预算与设备。 |
| `eval ...` | pi 个人检出 | `run.sh` → `tools/validate.py`。用于原 unified 五家族速度矩阵，**不是**随机图九地块评估。 |
| `view video ...` | pi 个人检出 | `run.sh` → `tools/record.py --record`。输出 `policy.mp4`、`review.html`、rollout、summary；配套 profile/资产须匹配。 |

`view interactive` 当前明确返回未支持，不启动仿真。`run.sh` 只设导入路径、缓存和设备，不自动保证 GPU 空闲。共享 GPU 在实际启动前需核对利用率、显存和 compute PID。

在 **Mac** 上查看四条路由，无运行副作用：

```bash
bash tools/wheelleg.sh --dry-run baseline 6 --check
bash tools/wheelleg.sh --dry-run train --task Robotics-Wheelbipe-Own-Explore-v0 --checkpoint /absolute/model.pt --max_iterations 1
bash tools/wheelleg.sh --dry-run eval --root /absolute/personal-checkout --profile unified_locomotion --checkpoint /absolute/model.pt --output /absolute/personal-checkout/runs/eval
bash tools/wheelleg.sh --dry-run view video --profile unified_locomotion --checkpoint /absolute/model.pt --output /absolute/personal-checkout/runs/video --steps 300
```

实际运行使用共享 `/data/isaac620-migration-20261103/env620/bin/python` 与个人源码 PYTHONPATH，不在共享环境重复安装项目。Own checkpoint 需要同级 `params/launch.json`、`env.yaml`、`agent.yaml` 和匹配 USD/reset bank；详情由 `checkpoint_utils.py` 等现有校验实现。

**已知 CLI 前提：** 当前 baseline 包装脚本保留旧 `--headless` 参数，而共享 Lab 的参数由实际 launcher 注册；使用前核对该版本支持的选项。本轮不修改运行脚本，也不宣称静态路由检查已验证真实 baseline 执行。

## 实验专用入口

| 任务 | 已有工具 |
| --- | --- |
| 随机连片地形训练/冒烟 | `tools/train_random_terrain.sh {train,smoke} GPU`；默认 4096×120，运行前显式确定有限预算与初始化权重。 |
| 随机图九地块冻结评估 | `tools/eval_random_terrain.py`，显式设 profile/checkpoint/output/seconds/seed/repeats/speed。 |
| 随机图连续路线录像 | `tools/record.py --random-terrain-route --record`；验收需同时检查 `route.completed`、reset 和物理结果。 |
| 纵向混合坡度训练/评估/录像 | `train_mixed_slopes.sh`、`eval_mixed_slopes.sh`、`record_mixed_slopes.sh`。 |
| 固定轻坡诊断 | `tools/light_slope.py {flat,up,down} GPU`；原平地权重、+0.50 m/s、300 策略步。 |

随机地形路线默认使用低空跟随相机；整图视角可显式传入 `--camera-mode overview`。现有 [video_review.py](tools/video_review.py) 会生成二维 `review.html`，也可用 `python3 tools/video_review.py <录像目录>` 补生成；它只缩放已录制画面，不提供自由三维相机，默认交付仍使用 MP4。录像工具更新已随本次推送合入，三角面地形修复仍需单独同步。

同 profile 也可能有不同物理 dt：`WHEELBIPE_INTEGRATION_SUBSTEPS` 默认 1，训练脚本设为 2。比较实验须看落盘配置。恢复同一优化器用原有 `--resume_training`，新分布微调用 fresh optimizer；不能混称。

长训练可在服务器 tmux 中存活，但持续自动监控需另有兼容的后台监控进程、心跳与有限重试。旧 `tools/supervise.py` 依赖历史 unified plan 和目录布局，当前随机图尚未适配；本轮没有启动训练或部署监控。

## 文件地图

```text
.agents/skills/wheelleg-repo-maintenance/  仓库专属维护 Skill
AGENTS.md / AGENT.md                      项目通用规则与授权边界
TASKS.md                                 技术进度、下一验收与学习 Review
profiles.json                            现有实验配置，保持与 project 相对布局
project/source/{agent_tasks,agent_world,agent_rl}/
                                         任务/地形、机器人资产、RL 适配
project/scripts/rsl_rl/                   训练、checkpoint 与回放主入口
tools/                                   操作薄入口、实验工具、轻量回归测试
docs/                                    当前操作说明；upstream/ 为历史来源
reports/                                 实验报告与选取的小型证据
runs/ logs/ videos/                       个人产物目录，默认 Git 忽略
```

Own→V14→V13→25_v3 的实现继承链仍在使用；不要按版本名删除。`tools/environment.py`、`sampling.py`、`terrain.py` 是旧安装器输入，与包内运行模块角色不同。完整工具索引和历史说明见 [tools/README](tools/README.md)。

## 维护与验证

结构整理、脚本去重、实验归档、文档同步及训练监控流程维护时，使用 [wheelleg-repo-maintenance](.agents/skills/wheelleg-repo-maintenance/SKILL.md)。例如：

> 使用 $wheelleg-repo-maintenance 整理本周实验与过时文档，先检查依赖，完成低风险修改并报告验证结果。

本机可运行的轻量检查（所需 NumPy/Torch/trimesh 覆盖以实际输出为准）：

```bash
python3 -B tools/test_light_slope.py -v
python3 -B tools/test_random_terrain.py
python3 -B tools/test_terrain_curriculum.py
python3 -B tools/test_video_review.py
```

Mac 缺少 Isaac runtime 时用 AST/静态检查，不直接 import 仿真入口。模型、缓存、视频和原始大日志留个人目录；reports 保留必要小型证据与哈希。历史系统说明见 [docs/upstream](docs/upstream/)，旧独立项目保存在 [main 分支](https://github.com/SilverLi123/wheelleg-robot-RL/tree/main)。
