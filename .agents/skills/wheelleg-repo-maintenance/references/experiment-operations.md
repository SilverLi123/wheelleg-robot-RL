# 实验来源、资源与持续监控

仅在维护实验产物、启动方式或已授权训练时读取。本 Skill 不授权训练、远端改动、调参或扩大预算。

## 用已有清单补缺，不再建一套系统

优先沿用 checkpoint 同级 `params/launch.json`、`env.yaml`、`agent.yaml`，以及 `training_updates.jsonl`、TensorBoard 和 diagnostics。`write_launch_manifest` 已记录源码/资产哈希、import 来源和版本，但未替代完整 Git 来源。

重要实验至少能回答：

- 个人检出绝对路径、branch、commit、dirty diff（包括未跟踪运行文件，必要时保存 patch/哈希）；服务器副本若没有 Git 明确记为无 Git，补传输清单与源码哈希。
- 初始化 checkpoint 路径/SHA256、checkpoint 旁的三个 params 文件；恢复优化器还是 fresh fine-tune；如存在课程 sidecar 保留它。
- task、profile 名及**实际内容/哈希**、USD/reset bank/机制文件哈希、Observation/Action 契约、运行包版本及实际 import 路径。
- seed、地图 layout seed、repeat 数、命令、起始条件、DR、物理 dt、decimation、策略 dt、环境数、训练迭代/总转移、LR schedule/clip/noise、保存间隔。
- GPU UUID/进程归属、tmux 会话、启动/结束时间与退出码、原始日志/结果目录、评估分布及结论限制。

小实验可直接在已有 report 记这些字段；只有现有清单缺少时，才在个人 `runs/<实验>/` 补一个小型 launch/运行记录，不要求复制所有日志进 Git。

## 启动前与恢复

1. 在 pi `ssh ubuntu@10.14.3.245 -p 6102` 只读核对可用空间、`nvidia-smi` 中显存/利用率/**compute PID**，确认没有队友占卡。优先按 UUID 绑定设备；空闲快照不是资源锁。
2. 训练与依赖都来自个人检出；共享 `/data/isaac620-migration-20261103/env620` 只复用，不安装/升级或切换 editable 项目。不改队友 `/data/wheelbipe-v14-6-training-20261108`。
3. 明确有限预算、评估节点、停止条件和允许调整的变量；授权已有明确预算就直接执行，不逐文件审批。用户未指定时提出一次可复核的小预算，不默认采用脚本的 4096×120。
4. 预检 profile/资产/输入语义，检查目标环境的真实 CLI 参数；`AppLauncher` 版本不同，不能把 `--headless`/`--viz` 当普遍支持。不要靠删除契约检查或随便 `--allow-profile-override` 跑通；覆盖须证明仅是批准的测试条件并记录差异。
5. 精确续训用现有 `--resume_training` 校验模型、优化器与迭代；新的训练分布用显式 fine-tune。不同物理积分、profile 或奖励不得冒充精确续训。

## 持续监控必须真实运行

`tmux` 让训练存活，**不等于自动监控**；终端手动 `tail` 只算本轮观察。

- 先调查 `tools/supervise.py`、`tools/status.py`、`project/scripts/rsl_rl/training_recovery.py` 的兼容性。旧 supervisor 固定 unified profile、依赖根 `plan.json`、curriculum sidecar，并且已有自动重试循环没有统一最大重试预算；不能直接给随机地形套用或复制历史 plan。
- 当前实验没有兼容监控器时，明确报告“仅训练后台运行，持续监控尚未配置”。在本次训练授权内如需实现，优先最小适配既有工具；必要的专用 watcher 放该次忽略的 `runs/<实验>/`，长期复用才提升为工具。
- 监控进程应在服务器独立 tmux/服务中运行，并保存自己的 PID、训练 PID/进程组、GPU UUID、start time/命令、心跳 `updated_at`、最新迭代/日志时间、退出码和操作记录。控制前重新确认 PID 身份，防止 PID 复用；禁止 `pkill python` 等广泛终止。
- 检查 Reward、loss、LR、KL（有记录才报告）、gradient、噪声、更新审计、失败重置/闭链误差与 GPU/内存。训练成功率分母和窗口要读实现，不能把日志数字直接当独立评估成功率。
- 启动耗时与日志停滞分开处理：4096 环境历史初始化约 10.5 分钟，只作为量级线索；结合 CPU、内存、日志时间和声明的超时判断，不宣称 CPU 活跃就一定未卡死。
- 非有限动作/损失、OOM、反复物理故障、超预算或资源冲突触发有记录的停止。只停止自己启动且身份确认的任务；拒绝抢卡，不因他人加入就擅自杀其进程。
- 重试必须有限次/有限总时长，只改授权变量；保留失败输出并用新目录运行，不热改正在训练的源码，不在无限恢复循环里消耗 GPU。扩大资源/目标或更改算法契约须先确认。
- 用替身子进程在临时目录测试心跳、正常退出、失败停止和重试上限后，才能宣称监控机制可用。关闭本机后心跳仍更新才是断连存活证据；本轮未做断连实验则明确写未验证。

## 评估与选型

- 原平地 `model_2000.pt`、原 downhill pilot `model_50.pt`、随机图新 `model_39.pt` 属于不同运行；裸文件名不能确定身份。结果必须带哈希和来源目录。
- `domain_randomization.py` 从 `WHEELBIPE_INTEGRATION_SUBSTEPS` 读取细化因子，默认 1；训练 shell 设置 2。比较基线和候选须核对落盘 dt，不因同 profile 就认为评估物理一致。
- `eval_random_terrain.py --repeats 3 --seed S` 是一个 seed 下每种地块三个实例；多 seed 需实际分别执行并保存。相同 seed 的 DR 与重置抽样也不自动构成严格配对。
- 保存原始 PASS/FAIL，但解释含义：单块无 reset、机械闭链通过、路线到达、速度/轨迹达标是不同判据。连续路线还要核对 `route.completed`、进度与物理失败，不能以拼接视频替代。
- 同时看速度 MAE、带符号/绝对横移、推进距离、横移/米、接触/净空和能力保留。样本量、时长、命令及地形比例不同的结果不混成一个排名。减速换稳定性可作为权衡，不能叫全面优胜。
- 模型选择应有同条件 baseline、预先说明的接受线和目标指标；训练 reward 变高或最后一份文件都不自动是最佳。迭代编号可能 0 开始，列出现有文件并查 metadata，不凭 `max_iterations=40` 推断存在 `model_40.pt`。
- 交付 checkpoint 同时交付同级 params、哈希、选择依据和局限。大文件留个人运行目录；reports 留小型摘要和确切来源。视频交付 MP4；只有用户明确需要真实三维回放时另作设计。
