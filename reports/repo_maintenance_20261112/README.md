# 仓库维护、专属 Skill 与实验来源审计（2026-11-12）

本轮执行 `skill-creator` 创建并验证仓库专属 Skill，再实际应用到文档和文件管理。未启动仿真或训练，未修改服务器文件、共享依赖、模型、PPO、Observation、Action、Reward、物理和运行源码。远端仅检查已有源码和评估结果。

## 工作树与事实边界

- 修改位置：当前 IDE 工作树 `/Users/silver/.codex/worktrees/v14-6-terrain-finetune/wheelleg_lab`，开始时 HEAD `88c4957812aa84d87b83c7d234c8ce00d965a27d`，branch `v14-6-terrain-finetune`，原状态干净。
- 另一独立 clone `/Users/silver/RL/wheelleg_lab` 的 HEAD 是 `d3b0655`，有未提交地形/测试/README 改动和障碍地形文件；它不是当前 Git common-dir 的另一个 worktree。本轮不覆盖或同步它。
- pi 个人副本为 `/data/wheeled-legged/projects/v14_6_random_terrain_20261112`。只读核实的源码哈希与本地检出信息见 [evidence.json](evidence.json)。
- 当前 IDE 的 `random_terrain_map.py` SHA256 为 `000b631a…`；另一 clone 与服务器已修复版为 `56ed33f9…`。`record.py` 也不同。因此旧检出的几何测试 PASS 不能证明服务器三角面高度修复已合入，也不能以 Git HEAD 单独代表训练源码。

## 真实结构与主要问题

| 问题 | 代码证据 | 本轮处理 |
| --- | --- | --- |
| 工具说明仍在指挥历史 unified 实验 | 旧 `tools/README.md` 描述历史绝对路径、plan 和暂停状态 | 原文存入 `docs/upstream/tooling-20261108.md`，工具目录改为当前职责索引。 |
| TASKS 和授权规则过时 | 原任务仍是首次平地复现；原 AGENT 每文件/小步需停，服务器仅只读 | 分开技术进度和用户学习 Review；支持用户已授权的连续开发与有限训练，不扩大授权。 |
| 看似旧版却仍在运行链 | OwnEnv→V14Env→V13Env→25V3Env；配置 Own→V14→25v3；`agent_tasks/__init__.py` 触发任务注册 | 全部保留，写入 Skill 依赖地图；没有按文件名删模块。 |
| 看似重复却有安装用途 | `configure.py:16–17` 复制 tools 下 environment/sampling/terrain 为包内 unified_*；sampling 完全相同，另外两份已分化 | 标注安装输入，保留；不覆盖较新的运行修复。 |
| 通用 eval 与随机图 eval 容易混用 | `wheelleg.sh eval` 实际调用 `validate.py` 的五家族速度矩阵 | README/工具索引明确九地块须用 `eval_random_terrain.py`；不改实验矩阵。 |
| 监控器前提不符 | `supervise.py` 依赖根 plan、固定 unified profile、curriculum sidecar、training 日志；恢复循环没有总重试上限 | 文档标为历史专用，不部署；Skill 要求有身份、心跳、退出状态与有限重试的真实后台机制。 |
| 大产物容易误入 Git | 已忽略 runs/logs/pt，但没有 videos/mp4/pth/ckpt 等保护 | 补忽略规则，不移除已有报告的小型证据。 |
| baseline CLI 有旧参数风险 | `reproduce_v14_6_flat.sh` 保留 `--headless`；共享 Lab 的实际参数由 sim_launcher 注册 | 文档显式标注，留待目标 runtime 参数核对。源码/原条件校验不在维护中悄悄改动。 |

包导入与任务链：`tools/wheelleg.sh → tools/run.sh → project/scripts/rsl_rl/train.py → import agent_tasks → gym.register` 的字符串 entry point → `wheelbipe_own.experiment`。PYTHONPATH 同时包括三个 agent 包根及 project，使 `agent_*`、`scripts.*` 都可找到。`experiment.py` 用 `parents[7]` 找根 profiles；资产模块以相对目录读取 mechanism/reachability/USD/reset bank。这些隐式路径是大迁移风险，已纳入 Skill 的迁移核查项。

## 既有随机地形训练结果的准确表述

本轮从 pi 原始 `summary.json` 及对应 `env.yaml` 只读汇总；不是重新仿真。详见 evidence 中每份报告路径、SHA256 和 checkpoint SHA256。

统一条件：seed=20261112、每地块 repeats=3、九地块各12秒、+0.65 m/s；落盘物理 dt=0.005秒、decimation=4，策略50 Hz。**这不是三个独立 seed。** 训练使用400 Hz，与评估不同，需后续同条件复核。

| 权重 | 原报告 | 速度 MAE（m/s） | 平均绝对横移（m） |
| --- | --- | ---: | ---: |
| 原 downhill model_50 | PASS，27/27 | 0.0150 | 0.4920 |
| 新 model_10 | PASS，27/27 | 0.0136 | 0.5374 |
| 新 model_20 | PASS，27/27 | 0.0290 | 0.4823 |
| 新 model_30 | PASS，27/27 | 0.0425 | 0.4497 |
| 新 model_39 | PASS，27/27 | 0.0496 | 0.4543 |

原来“model_39 平均横移最低”的说法不成立，model_30 略低；二者均有速度代价。PASS 主要是单块回放无失败，不是完整连续路线、泛化或综合最优的证明。当前统一标为候选，不替换原 baseline。

既有交付目录 `/Users/silver/RL/wheelleg_lab/runs/delivery_terrain_tri_fix_4096_40_20261112/` 保存了 `model_39.pt` 和 params，哈希以 `25030a3f…` 开头；其 metadata 不是本轮生成或改写。该清单中的 `final_policy_steps=900` 来自周期健康采样，40×24 的实际采集预算是960个策略步、4096×960=3,932,160个环境转移；不把健康采样数当末次训练总步数。

## Skill 创建与实际试用

- 使用已安装 `/Users/silver/.codex/skills/.system/skill-creator/SKILL.md`，本地文件 SHA256 `6656e54755638e8efcf275a472b9672eaa8a9a1b9e59dc210e275b03b59e1e66`；该文件未声明语义版本。
- 对照用户提供的 [skill-creator 目录页](https://www.skills.sh/openai/codex/skill-creator)，并按 [OpenAI Skills 文档](https://developers.openai.com/codex/skills) 的仓库发现约定存放在 `.agents/skills/`；不重复安装已存在的系统 Skill。
- 使用其 `scripts/init_skill.py wheelleg-repo-maintenance --path .agents/skills --resources references --interface ...` 初始化，而非只手写 AGENTS。
- 按 `references/openai_yaml.md` 生成 UI 元数据，保留默认隐式选择；没有复制外部大模板或新建一串维护脚本。
- `SKILL.md` 包含范围选择、依赖调查、文件归属、文档/实验判断和验证交付；两个参考分别承载真实仓库关系与实验/持续监控要求。
- 独立只读 forward-testing：评估者先只看 frontmatter 判断五条请求，再读取 Skill 与必要代码完成行为审查，未给予预期结论。详见下表。
- 试用后修正一处路径歧义：`training_recovery.py` 明确指向 `project/scripts/rsl_rl/training_recovery.py`，没有继续堆叠通用规则。
- 实际维护后的独立复核发现 `/artifacts/` 不覆盖当前 artifacts 符号链接；补回根路径 `/artifacts`，把真实链接加入忽略回归验证，并在 Skill 里记录这个文件管理陷阱。
- 随后实际读取/应用 Skill 完成本轮低风险维护：README、TASKS、工具说明、历史说明归属、AGENT/AGENTS、gitignore。原运行入口全部保留。

| 行为案例 | 观察结果 |
| --- | --- |
| 删除 V13/25_v3/sampling“旧文件” | 应触发；找到实际继承/复制源，拒绝直接删除。 |
| 仅凭 model_39、27/27、40次声明归档 | 应触发；当前检出缺源证据时不认证；识别旧27/27属于原model_50，要求权重/params/哈希和原报告。主代理再只读取得真实服务器证据。 |
| 4096×40、关机后监控，审查旧 supervisor | 应触发；发现布局/profile/sidecar与无限恢复边界，要求真实心跳和有限机制；未假装已部署。 |
| 解释 projected_gravity 三维 | 不触发维护；直接解释观测，避免全量扫描。 |
| 迁移 tools/environment.py | 应触发；区分运行模块与安装复制输入，发现相对导入和安装器路径，先交影响分析。 |

这属于独立场景模拟与实际维护试用，**不是平台自动路由端到端测试**。当前新 Skill 可按路径显式读取；未来会话的自动发现按 `.agents/skills/` 元数据和 AGENTS 路由。Skill 本身不启动进程、不批准训练。

## 验证

可机器读取的实际命令、退出码、输出和边界记入 [validation.json](validation.json)。包括官方 quick_validate、YAML/引用、Python AST、Shell 语法、动态注册的无仿真替身检查、四入口 dry-run、现有轻量测试、忽略规则及运行源码哈希不变检查。

校验临时环境仅在本机临时目录，使用 PyYAML 6.0.3 和 NumPy 2.0.2；项目 Python 与 env620 未安装或升级依赖。NumPy 测试不等于 Torch/CUDA 分支运行，trimesh/Isaac 缺失部分按 SKIP/未执行记录。

## 本轮有意保留的工作

- 不同步两个 clone，也不覆盖服务器；把未提交三角面修复合入当前树需要明确来源及单独回归。
- 不移动/删除继承模块、安装模板、supervisor 或重写四入口；它们仍有调用或实验条件差异。
- 不修原 baseline CLI、不重测仿真、不重新排名最佳模型；报告静态前提和同条件评估缺口。
- 不把历史 plan 复制到根，不开启无预算自动恢复，不部署新监控服务。
- 不修改另一 clone 的交付清单/视频，不将大 checkpoint 或视频提交 Git，也不自动提交/推送本轮维护。

后续优先解决当前源码与训练来源一致，再统一400 Hz对照和验收线；技术进度与学习 Review 见根 TASKS。
