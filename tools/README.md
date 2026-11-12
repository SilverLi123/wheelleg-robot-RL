# 工具索引

从仓库根目录使用本页路径。四个日常入口和通用用法只在 [README](../README.md) 维护；本页区分工具职责与运行前提。`--dry-run` 只打印转发，不能代替 GPU、资产、CLI 或 profile 检查。

## 日常入口与实验专用工具

| 工具 | 职责与范围 |
| --- | --- |
| `wheelleg.sh` | baseline / train / eval / view video 的薄转发；interactive 暂明确不可用。 |
| `run.sh` | 设置个人源码 PYTHONPATH 与缓存路径后执行 Python；本身不检查 GPU 空闲。 |
| `reproduce_v14_6_flat.sh` | SSH 读取队友原 `model_2000.pt` 并校验；`--check` 不仿真。含旧 `--headless` 参数，需按共享版本核对后再实跑，详见维护报告。 |
| `train_random_terrain.sh` | 随机整图训练/冒烟；空闲卡检查、flock、独立 runs；默认 4096×120，运行前显式确定本轮预算。 |
| `eval_random_terrain.py` | 固定九地块起点、指定 seed/repeats/时长的冻结策略评估；不等于整图路线完成。 |
| `train_mixed_slopes.sh` / `eval_mixed_slopes.sh` / `mixed_slopes_eval.py` | 原纵向混合坡度训练与评测；不要和随机整图矩阵混用。 |
| `validate.py` | 原 unified 五家族与有符号速度评估，`wheelleg.sh eval` 实际指向这里。 |
| `record.py` / `record_mixed_slopes.sh` | 冻结策略 MP4、rollout、summary；前者可选单块或连续路线，后者组织指定坡度案例。 |
| `light_slope.py` | 固定平地权重的 flat/up/down 轻坡诊断，保持专门的参数与方向检查。 |
| `summarize_mixed_slopes_eval.py` / `assemble_mixed_slopes_video.py` | 对已有结果做汇总/视频处理；不训练。 |
| `test_light_slope.py` / `test_mixed_slopes.py` / `test_random_terrain.py` / `test_terrain_curriculum.py` | 相应现有轻量回归测试；缺少 NumPy/Torch/trimesh 时按实际输出报告 SKIP。 |

## 安装模板与历史操作：保留但不当作日常命令

| 工具 | 保留原因与使用边界 |
| --- | --- |
| `environment.py` / `sampling.py` / `terrain.py` | `configure.py` 的复制输入，对应包内 unified_* 运行模块。sampling 调查时完全相同，另外两份已分化；不能直接删模板或用模板覆盖新运行代码。 |
| `configure.py` | 用模板在特定旧项目副本打补丁、写 profile 与 plan；不是当前源码更新器。 |
| `install_recovery.py` / `install_mixed_slopes.py` | 特定历史迁移补丁；静态确认锚点、目标与授权后才考虑使用。 |
| `supervise.py` / `status.py` | 旧 unified plan/training/sidecar 布局的真实调度与状态工具；不直接适用于当前随机地形 runs。现有恢复循环还需要总预算/重试上限。 |
| `verify_recovery.py` | 旧 supervisor/recovery 的替身故障测试，仍有价值；需 Torch，不能把测试通过解释成随机图监控已适配。 |
| `benchmark_parallel.py` / `reward_ablation.py` | 有训练副作用，依赖旧初始化目录/profile/日志布局；仅在明确授权实验时核对后使用。 |
| `install_*` / `finalize_handoff.py` | 迁移/交接；后者有进程信号和跨目录写入，不能作为通用清理器直接运行。 |

原上游操作说明完整保存在 [2026-11-08 工具历史记录](../docs/upstream/tooling-20261108.md)。上游快照、plan 与恢复记录见 `docs/upstream/`；它们提供复现来源，不授权当前训练。

需要整理、新增或归档工具时使用 [wheelleg-repo-maintenance](../.agents/skills/wheelleg-repo-maintenance/SKILL.md)。训练配置、观测/动作和物理修改仍由具体算法任务决定。
