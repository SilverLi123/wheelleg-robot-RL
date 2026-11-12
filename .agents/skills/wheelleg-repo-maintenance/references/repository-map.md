# 仓库地图与维护陷阱

所有路径相对 `git rev-parse --show-toplevel`；本图是导航线索，每次以实际代码为准。2026-11-12 调查时存在 IDE worktree 与 `/Users/silver/RL/wheelleg_lab` 独立 clone，二者相同分支名不代表内容相同。

## 入口调用链

| 入口 | 真实行为 | 维护注意点 |
| --- | --- | --- |
| `tools/wheelleg.sh baseline GPU [--check]` | SSH 到 pi，`reproduce_v14_6_flat.sh` 读取队友原 `model_2000.pt` 与资产并做哈希校验 | `--check` 不仿真；不加此参数会执行。`--dry-run` 只打印，不能证明输入兼容。 |
| `wheelleg.sh train ...` | `tools/run.sh` → `project/scripts/rsl_rl/train.py` | 通用转发不替用户选择 profile/GPU/预算；`run.sh` 本身没有空闲 GPU 检查。 |
| `wheelleg.sh eval ...` | `run.sh` → `tools/validate.py` | 原 unified 五家族×有符号速度矩阵；不是随机九地块评估器。 |
| `wheelleg.sh view video ...` | `run.sh` → `tools/record.py --record` | MP4 与 rollout/summary；旧检出相机/HTML 支持不同，先看实际参数。 |
| `tools/train_random_terrain.sh {smoke,train} GPU` | GPU 检查、flock、隔离输出 → 同一 `train.py` | 默认 4096×120；参数来自环境变量。stage=2、积分细化=2、种子/保存间隔须实读。 |
| `tools/eval_random_terrain.py` | 每种地块固定起点/朝向，`--seconds`、`--repeats`、`--seed`、`--speed` | repeats 是同一 seed 下多个实例；无 reset 判据不能代替整图/净空验收。 |
| `tools/train_mixed_slopes.sh` / `eval_mixed_slopes.sh` | 纵向坡度家族训练 / `mixed_slopes_eval.py` | 与随机整图不同实验；不能仅因名称相近合并。 |
| `record_mixed_slopes.sh` / `light_slope.py` | 按指定列或固定轻坡条件调用 `record.py` | 后者锁定平地权重与 300 步诊断条件，不能随便改默认值。 |

`view interactive` 当前统一入口明确拒绝启动，不把它改成“成功”提示。视频默认交付 MP4；用户已拒绝没有真实三维视角能力的 HTML 视频壳。

## 运行依赖不是按年份淘汰

```text
agent_tasks/__init__.py
  imports wheelbipe_V14 and wheelbipe_own → gym.register(... strings ...)
Own-Explore-v0 → experiment.WheelbipeOwnExperimentEnv
  (UnifiedLocomotionMixin, DomainRandomizationMixin, WheelbipeOwnEnv)
WheelbipeOwnEnv → WheelbipeV14Env → WheelbipeV13Env
               → Wheelbipe25V3Env → Isaac Lab DirectRLEnv
```

配置链不完全同上：OwnFlatCfg → V14FlatCfg → 25v3FlatCfg → ReduceSpring/BaseCfg。
`agent_world` 资产模块、`agent_tasks.manager.mdp`、状态机和继承类的辅助模块也要保留。
`project/scripts/rsl_rl/train.py` 同时导入 `rsl_rl.runners` 与 `agent_rl.rsl_rl.runners`；后者适配/扩展不能仅凭包同名判重复。

保持 `tools/run.sh` 中四个 `PYTHONPATH` 根：

```text
project/source/agent_tasks
project/source/agent_world
project/source/agent_rl
project
```

它们既支持 `agent_*` 也支持 `scripts.*`。导入来源由 `params/launch.json` 的 `imports` 与 `project_root` 记录；别用共享环境 editable install 切换项目。
`experiment.py` 通过 `Path(__file__).resolve().parents[7] / 'profiles.json'` 找配置。迁移目录要检查这类隐式关系。
资产入口 `project/source/agent_world/agent_world/assets/wheelbipe_own.py` 读取同级 `usd_files/wheelbipe_own/` 的 mechanism、reachability、USD 和 reset bank。

## 表面重复与历史工具

| 文件 | 已确认角色 | 建议 |
| --- | --- | --- |
| `tools/sampling.py` ↔ `.../wheelbipe_own/unified_sampling.py` | 调查时字节相同；`configure.py` 会将前者复制成后者 | 先保留安装输入。若以后退出安装器，要一起证明调用失效，再去重。 |
| `tools/environment.py` / `terrain.py` | 旧安装模板，复制为 `unified_environment.py` / `unified_terrain.py` | 已分化，覆盖会丢修复；不要以 tools 模板替换运行模块。 |
| `configure.py` / `install_recovery.py` / `install_mixed_slopes.py` | 按特定源码锚点打补丁/重写配置 | 历史或迁移用途，不是日常更新；不可为了测试直接运行。 |
| `supervise.py` / `status.py` | 旧 unified `plan.json`、`training/`、curriculum sidecar 与日志布局 | 真实后台调度实现，但不兼容当前随机地形 `runs/train_*` 的前提未补齐。 |
| `finalize_handoff.py` | 特定历史交接，含进程信号、复制与写状态 | 有副作用，先静态阅读；不是通用清理命令。 |
| `project/scripts/own_robot/preflight.py` 与 `preflight620.py` | 前者服务旧运行入口，后者是当前 Own 训练启动检查 | 不互相覆盖；检查实际调用和版本要求。 |
| `project/scripts/utils/velocity_trace_html.py` | V14 环境直接 import 的遥测构建功能 | 用户拒绝视频 HTML 不等于可删除这个模块。 |

## 审计命令与证据

```bash
git ls-files tools project/scripts project/source reports docs
rg -n 'unified_sampling|sampling.py|copyfile|entry_point|PROFILE_FILE|parents\[' tools project
rg -n 'class .*Env|from .*wheelbipe' project/source/agent_tasks/agent_tasks/direct/wheelbipe
git diff --check
```

文本搜索后检查 AST 的 import、类基类及字符串路径。零命中不证明运行时没有生成引用；有需要时在已授权的隔离短任务中核实实际 import 路径。
