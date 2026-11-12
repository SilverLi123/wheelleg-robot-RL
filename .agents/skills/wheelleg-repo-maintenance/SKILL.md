---
name: wheelleg-repo-maintenance
description: "维护 Wheelbipe V14-6 Isaac Lab/RSL-RL 仓库：处理脚本冗余、文件归属、依赖安全迁移、实验归档、文档同步和已授权训练的运行管理。适用于仓库整理或实验可复现性、监控流程维护；普通算法讨论或单点代码修改无需全量审计。"
---

# Wheelleg Repository Maintenance

让 V14-6 平地基线与复杂地形微调可理解、可运行、可追溯。在用户授权范围内连续完成相关修改、验证和报告；本 Skill 本身不授予服务器写入或训练权限。

## 选择范围

- 先读根目录 `AGENTS.md`、`AGENT.md`、`TASKS.md`，以当前用户任务覆盖过时约定。
- 用 `pwd`、`git status --short --branch`、`git rev-parse --show-toplevel --git-common-dir`、`git worktree list --porcelain` 确认本次检出。另一路径的独立 clone 不会出现在 worktree 列表；若对话/报告出现另一检出，另行比较其 HEAD 和状态。不同 clone、服务器副本和未提交修改分别记录，不静默覆盖或跨树同步。
- 单个工具/文档问题：只看该入口、调用方和配套报告。入口变化、清理继承模块、多处状态冲突或用户要求全面整理时，才扩展审计到 `tools/`、`project/scripts/`、`project/source/`、`reports/`、`docs/`。
- 先按[仓库地图](references/repository-map.md)定位实现；涉及 checkpoint、训练或监控时再读[实验运行管理](references/experiment-operations.md)。

## 调查后做决策

1. 找实际入口：`rg -n '目标符号|脚本名|profile名' tools project/scripts project/source profiles.json README.md TASKS.md docs reports`。沿 Shell 转发、Python import/继承、任务注册字符串、路径计算追踪；无 Isaac 的本机优先 AST/文本分析，不能通过 import 启动仿真。
2. 判断重复：比较内容哈希、AST、调用方、参数默认值、环境变量、输出目录和安装/生成角色。字节相同也可能是安装输入；名字相近但实验矩阵不同的评估器不是同义入口。空 `__init__.py` 也可能承担包结构角色。
3. 每个候选给出“保留 / 文档标注 / 整合 / 归档”的依据与验证办法。保留 V14→V13→25_v3 运行继承链。没有直接引用只是证据之一，不足以证明可删。
4. 先完成确定的低风险修正：文档路由、历史状态标注、忽略大产物、纠正重复说明。若整合入口，旧调用保留兼容转发，参数与退出码不得悄悄改变。大规模迁移、关键模块删除、覆盖模型先交付影响分析与方案供用户确认。

## 新文件与归档

- 机器人任务逻辑放 `project/source/agent_tasks/.../wheelbipe_own/`；资产/坐标与执行器放 `agent_world`；算法适配放 `agent_rl`。维护任务不顺便改 PPO、Observation、Action、Reward 或物理。
- 日常操作薄入口放 `tools/`；上游训练/回放主入口保留 `project/scripts/rsl_rl/`。开发前查已有相邻工具；几行转发够用时不新造框架。
- 现有轻量回归测试留在 `tools/test_*.py`，与被测行为配套。临时探针优先放忽略的 `runs/<实验>/`；有复用价值再提升为工具，别按日期复制一串入口。
- `profiles.json` 与 `project/` 保持现有相对布局。当前 profile 加参数会影响契约；新实验优先用显式新 profile，仍需算法任务授权。
- 正式结论放 `reports/<主题或实验>/`，包含小型证据或其哈希/路径；设计与当前操作说明放 `docs/`。一次性安装、补丁、交接脚本先在 `tools/README.md` 标注用途和输入前提；证实不再被调用、保留复现替代方案后再按任务归档。
- checkpoint、视频、TensorBoard、缓存和原始日志放个人 `runs/`、`logs/`、`videos/`，默认不进 Git；小型且明确选择的证据可保留在 reports。检查 `git ls-files`，忽略规则不会自动移除已跟踪文件。调整忽略规则要用实际路径验证，`/artifacts/` 不匹配同名符号链接，应保留 `/artifacts` 的覆盖。

## 文档同步与实验结论

- README 只给当前状态、日常入口和报告链接；`tools/README.md` 负责工具职责及旧工具前提；TASKS 记录实际进度与下一验收目标。历史报告保留原条件和失败证据，用后续报告链接纠正，不重写历史结果。
- 区分“执行完成”“物理检查通过”“策略任务成功”“用户已 Review”。27 次单块无 reset 不代表整图完成；三次 repeat 不代表三个独立 seed；新 checkpoint 未经同条件比较只能标候选。
- 操作入口、文件位置、授权方式或模型选择结论改变时更新相应文档。普通实现细节无需重复写三份；报告保存事实，README/TASKS 链接事实。

## 验证与交付

- 变动前记录受影响源码、profile 的哈希/差异；变动后核实不在范围内的运行文件未变。
- 文件迁移检查：import（含相对导入与包 `__init__`）、`gym.register`/Hydra 字符串、Shell、`Path.parents[...]`、配置/安装复制源、测试、文档相对链接。静态检查不能证明动态运行无误。
- 按改动运行 `bash -n`、Python AST/隔离编译、四入口 `--dry-run`、相关现有轻量测试和 `git diff --check`。只有涉及运行契约且已获授权时才做短仿真；维护不默认触发 baseline/训练。明确写出依赖缺失导致的 SKIP 与未覆盖层。
- Skill 本身变更用已安装 skill-creator 的 `quick_validate.py`，再用真实维护请求做行为试验；结构通过不等于自动触发已验证。
- 最终给出实际修改、命令与结果、保留依赖、未处理风险和下一步；宣称持续监控前必须有可检查的后台进程、心跳和退出状态。具体见实验参考。
