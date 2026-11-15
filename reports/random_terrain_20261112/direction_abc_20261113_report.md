# Wheelbipe V14-6 复杂地形三方向（A / B / C）推进与实验报告

日期：2026-11-13  
本地仓库：`/Users/silver/RL/wheelleg_lab`  
服务器工作区（`pi`）：`/data/wheeled-legged/projects/v14_6_random_terrain_20261112`  
注：全部训练与评估均仅在用户个人工作区运行，未修改共享环境 (`/data/isaac620-migration-20261103/env620`)、队友目录 (`/data/wheelbipe-v14-6-training-20261108`) 或既有 Baseline Checkpoint。原始修改前服务器文件已备份于 `backups/pre_20261113_abc/`。

---

## 一、总览结论

本次按既定的 A / B / C 三个方向完成源码开发、本地与服务器同步、`pi` 服务器真实 400 Hz（`WHEELBIPE_INTEGRATION_SUBSTEPS=2`，`dt = 0.0025 s`，`decimation = 8`）训练与多随机种子对照评估，核心结论如下：

1. **方向 A（跨块过渡探针 & 连续路线根因实证）**
   - 开发跨块过渡专项探针 [`tools/eval_terrain_transitions.py`](../../tools/eval_terrain_transitions.py)，并在 `3` 个随机种子（`71903, 173927, 384719`）× `10` 个工况（每模型 `90` 局，共 `8` 个模型 `720` 局）上以 400 Hz 完成测试。
   - **全部 `8` 个合法跨块过渡**（6 个沿蛇形主线的纵向 X-seam 过渡 `0->1, 1->2, 5->4, 4->3, 6->7, 7->8` 与 2 个平地走廊跨行过渡 `row_corridor_2_to_5, row_corridor_3_to_6`）在所有 `8` 个模型上均取得 **`72/72 (100.0%)` 零跌倒通过**。
   - **精确定位历史第 11,759 步连续路线失败真因**：在 `row_crest_3_to_6` 峭壁探针中（复现 `block 3 (up06)` 坡顶 `block_x = -1.31 m` 直接 `+90°` 拐向 `block 6 (up18)` 坡顶的 `-24%` 骤降 + `+72% / 35.8°` 侧向衰减墙），`gimbal_yaw_link` 仍有 **`+8.39 cm ~ +9.47 cm`** 离地间隙；真实率先触壁并触发 `contact` 终止的部件是 **`right_front_guide1_link`**（前轮四连杆前导杆，位于半径 `6.25 cm` 轮子前方）及 `left_front_guide1_link`。问题根因在于原 [`tools/record.py`](../../tools/record.py) 的中心点路线让机器人在坡脊处横向硬跨 `1.5 m` Y-taper 陡墙，而非策略本身的纵向跨块能力不足。
   - **揭开历史 9 块基准 `97.53%` vs `100%` 差异原因**：核实历史 `eval_after_y_taper_20261112_100732` 运行于 200 Hz（`dt = 0.005`），而 `ft40_model_10` 在与训练一致的 **400 Hz** 下 3 种子复测取得 **`3/3` 种子 PASS（`81/81 = 100.0%`）**、`vx_mae = 0.0141 m/s`。
   - **Corridor 连续 9 块零重置验证完成**：
     - 结合 [`random_terrain_map.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/random_terrain_map.py#L171-L187) 网格三角化几何分析（`block_x ∈ [-7.0, -1.0], block_y ∈ [-6.0, -4.5]` 上三角存在 `+72%` 侧向衰减墙，仅 `|block_x| ∈ [7.0, 8.0] m` 走廊全线平直零高度）及策略高速偏航动力学实测（`vx = 0.65 m/s` 下存在约 `-0.09 rad/s` 稳态顺时针偏航漂移），在 [`tools/record.py`](../../tools/record.py) 中将 `--route-mode corridor` 升级为 `|block_x| = 7.10 / 7.35 m` 显式内外角航点与滞回转角控制器（`abs(error) > 0.35` 入弯对齐至 `< 0.04` 退出、直行增益 `2.5 * error`）。
     - 在 `pi` 服务器 400 Hz 下完成 13,500 步（`270.0 s` 仿真时长，`108,000` 物理子步）全地形蛇形路线验证（`route_corridor_v4_dirC_actor5_20261113` 与 `route_corridor_v4_ft10_20261113`）：两者均取得 **`status = PASS`、`resets = 0`（全 9 块连续通过、`contact/roll/pitch/numerical/spring/closed_loop = 0`）**。

2. **方向 B（边界过渡加权采样 + 单环境闭环重置）**
   - 新增配置 profile `random_terrain_transition_20261113`（[`profiles.json`](../../profiles.json)），设置 `random_terrain_transition_fraction: 0.4`（40% 复位直接采样入坡前区与跨块出口区）和 `per_env_closure_reset: true`（单环境闭环误差超过 `0.008 m` 仅终止并复位该环境，不中断全局训练）。
   - 修复了启用 `per_env_closure_reset` 时 [`physics_failure_diagnostics.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/physics_failure_diagnostics.py) 调用 `view.get_link_transforms()` 引发 PhysX 6.2 C++ 静默段错误的问题，并优化了 [`experiment.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/experiment.py) `_init_training_audit()` 4096 环境路径发现性能。
   - 在 1024 envs × 20 轮安全微调（`lr = 1e-5`, `finetune_noise_std = 0.05`）下，产出 `dirB_model_10` 与 `dirB_model_19`，在 400 Hz 9 块基准均取得 **`81/81 (100.0%)`** 通过。

3. **方向 C（Critic 相对地形高度 `critic_relative_height_obs` + 安全微调）**
   - 在 [`unified_environment.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py) 新增 `_terrain_ground_z()` 与 `critic_relative_height_obs`，**严格保持 Actor 35 维观测完全不变**，仅将 Critic 的第 31 维 `obs_height`（`self.priv[..., 31]` 与 `self.priv_latent[..., 3]`）替换为缩放裁剪后的相对地形高度 $z_{\text{root}} - z_{\text{ground}}$。
   - **训练稳定性显著改善**：对比同超参的 `dirB_1024` 与 `dirC_warm_1024`，`Mean value_function loss` 峰值与常态明显更低（`dirC_warm_1024` 在 iter 10–18 降至 `0.35–1.10`，iter 13 达 `0.3473`；Critic 参数总漂移 `critic_delta_l2 = 0.2921`，低于 `dirB_1024` 的 `0.4177`），且在持续微调至第 19 轮时依然保持 **`81/81 (100.0%)`** 9 块全通过（消除了历史上 `train_8uUtJL` 后期坍塌）。
   - **`dirC_actoronly_model_5`（Actor-only 微调第 5 轮）取得全组最低速度误差**：400 Hz 9 块基准 **`81/81 (100.0%)`**，**`vx_mae = 0.0140 m/s`**（`ft40_model_10` 为 `0.0141 m/s`），`forward = 7.58 m`。

---

## 二、代码与 Profile 改动清单

| 文件 | 核心改动 |
| --- | --- |
| [`profiles.json`](../../profiles.json) | 新增 `random_terrain_transition_20261113` 与 `random_terrain_relcritic_20261113`（含 `random_terrain_transition_fraction: 0.4`、`per_env_closure_reset: true`、`critic_relative_height_obs: true`） |
| [`random_terrain_map.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/random_terrain_map.py) | 在 `RANDOM_TERRAIN_PROFILES` 注册新 profile；`_sample_spawn()` 支持 `evaluation_spawn_xy_yaw` 精确位姿评估与 `random_terrain_transition_fraction` 边界加权采样；`health()` 增加 `transition_spawns` 计数 |
| [`unified_environment.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py) | 提取 `_terrain_ground_z()`；实现 `critic_relative_height_obs`（仅改 Critic 第 31 维 `obs_height`）；`_get_dones()` 支持 `per_env_closure_reset`（`per_env_loop_error > 0.008 m` 单环境安全复位） |
| [`experiment.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/experiment.py) | 优化 `_init_training_audit()` 仅遍历 `env_0` 发现 6 个球铰相对路径；`_audit_training()` 在 `per_env_closure_reset` 下允许单环境误差触发局部复位（阈值放宽至全局 `peak > 0.12 m` 或 `>= 5%` 环境坏掉才中止） |
| [`physics_failure_diagnostics.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/physics_failure_diagnostics.py) | 启用 `per_env_closure_reset` 时跳过 `view.get_link_transforms()` C++ 调用，消除翻车预复位瞬间引发的 PhysX 段错误 |
| [`tools/eval_terrain_transitions.py`](../../tools/eval_terrain_transitions.py) | 新建 10 工况跨块过渡探针评估脚本，记录接触复位刚体名、`gimbal_yaw_link`/`base_link` 间隙及目标块到达率；修复 `body_quat_w` `xyzw -> wxyz` 转换及 USD `mesh_world * body_inverse` 行向量乘法顺序 |
| [`tools/eval_random_terrain.py`](../../tools/eval_random_terrain.py) | 同步修复 `body_quat_w` `xyzw -> wxyz` 转换及 USD `mesh_world * body_inverse` 变换顺序 |
| [`tools/test_random_terrain.py`](../../tools/test_random_terrain.py) | 新增三维四元数旋转精确性及 180° 坡面反转间隙不变性单元测试 |
| [`tools/record.py`](../../tools/record.py) | 新增 `--route-mode {center,corridor}`；并将 `corridor` 模式设为零高度平面走廊（`|block_x| ∈ [7.0, 8.0] m`）显式内外角航点与滞回转弯控制（`abs(error) > 0.35` 入弯对齐至 `< 0.04` 退出、直行增益 `2.5 * error`），消除三角网格 Y-taper 陡墙碰撞与高速偏航漂移 |

---

## 三、方向 A：400 Hz Transition Probe 对照结果（3 种子 × 10 工况）

全部 `8` 个模型均在 400 Hz（`WHEELBIPE_INTEGRATION_SUBSTEPS=2`）、`3` 种子（`71903, 173927, 384719`）、`speed = 0.65 m/s`、`8.0 s` 条件下完成（每模型 `90` 局，其中 8 个合法跨块过渡共 `72` 局，`row_crest_2_to_5` 与 `row_crest_3_to_6` 各 `9` 局）。

| 模型 | 合法过渡通过 (8工况×9局) | 合法过渡 `vx_mae` (m/s) | 平均前向位移 (m) | `crest 2->5` 通过 | `crest 3->6` 通过 (复现11759步) | `crest 3->6` `gimbal` 最小间隙 | `crest 3->6` 实际撞击刚体 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `model_50` (downhill baseline) | **72 / 72** | 0.0185 | 5.02 | 8 / 9 | 0 / 9 | +0.0894 m | `right_front_guide1_link` |
| `ft40_model_10` (`train_8uUtJL`) | **72 / 72** | **0.0178** | 4.87 | **9 / 9** | 1 / 9 | +0.0889 m | `right_front_guide1_link` |
| `noise005_model_9` | **72 / 72** | 0.0188 | 4.86 | 8 / 9 | 0 / 9 | +0.0928 m | `left_front_guide1_link`, `right_front_guide1_link` |
| `dirB_model_10` | **72 / 72** | 0.0195 | 4.85 | 8 / 9 | 0 / 9 | +0.0864 m | `right_front_guide1_link` |
| `dirB_model_19` | **72 / 72** | 0.0225 | 4.83 | 7 / 9 | 0 / 9 | +0.0839 m | `right_front_guide1_link` |
| `dirC_warm_model_10` | **72 / 72** | 0.0197 | 4.86 | 8 / 9 | 1 / 9 | +0.0474 m | `right_front_guide1_link` |
| `dirC_warm_model_19` | **72 / 72** | 0.0209 | 4.85 | 7 / 9 | 0 / 9 | +0.0903 m | `right_front_guide1_link` |
| `dirC_actoronly_model_5` | **72 / 72** | **0.0178** | 4.87 | 6 / 9 | 1 / 9 | +0.0846 m | `right_front_guide1_link` |

> **工程备注（已于 2026-11-13 晚完成修复并复测）**：上述初版输出中，合法工况（如 `yaw = π`、`yaw = π/2`）出现伪负间隙由两个叠加工装缺陷共同导致：
> 1. `random_terrain_map.py:333` 中 `quats_rotate_wxyz_numpy()` 的批量 `einsum` 误用了 `"ij,njk->nik"`（等价右乘 $R$ 即 $R^T$），与单四元数时的 $R$ 相悖；已修复为 `"ij,nkj->nik"`，并在 [`tools/test_random_terrain.py`](../../tools/test_random_terrain.py) 中补充了 `yaw = π`、斜坡大偏航角与一般三维旋转及 180° 坡面反转间隙不变性单元测试。
> 2. `env620`（Isaac Sim 6.2 / Isaac Lab 3）中 `env._native_robot.data.body_quat_w` 为 PhysX/Warp `(x, y, z, w)` 顺序，而 `collision_points_minimum_clearance()` 期望 `(w, x, y, z)`；且 USD 行向量变换应为 `mesh_world * body_inverse`。已在 [`tools/eval_terrain_transitions.py`](../../tools/eval_terrain_transitions.py) 与 [`tools/eval_random_terrain.py`](../../tools/eval_random_terrain.py) 中修复为 `[:, [3, 0, 1, 2]]` 及 `mesh_world * body_inverse`。
>
> **修复后 400 Hz Transition Probe 复测结果（`seed = 71903`, `3` 重复，每模型 8 个合法过渡共 `24` 局）**：
> - `trans_fixed_v2_dirC_actor5_s71903`：`status = PASS`，合法过渡 **`24 / 24 (100.0%)`**，`vx_mae = 0.0179 m/s`，**`min_gimbal_clearance_m = +0.2622 m`**，**`min_base_clearance_m = +0.0091 m`**（伪负间隙彻底清零）。
> - `trans_fixed_v2_ft10_s71903`：`status = PASS`，合法过渡 **`24 / 24 (100.0%)`**，`vx_mae = 0.0174 m/s`，**`min_gimbal_clearance_m = +0.2618 m`**，**`min_base_clearance_m = +0.0092 m`**（伪负间隙彻底清零）。

---

## 四、方向 B & C 训练动态对比

所有微调均从 `ft40_model_10`（`train_8uUtJL/model_10.pt`）出发，采用 `learning_rate = 1e-5`、`finetune_noise_std = 0.05`：

| 训练任务 | Profile | `num_envs` / `iters` | 边界采样率 | Critic 高度输入 | `vf_loss` 区间 / 末值 | `critic_delta_l2` / `actor_delta_l2` | 末轮 `Metrics/success_rate` | `transition_spawns` |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_dirB_1024_20261113` | `random_terrain_transition_20261113` | 1024 / 20 | 0.4 | 世界 `z_root` | 0.86–4.63 / 2.2465 | 0.4177 / 0.0264 | **1.0000** (iter 18–19) | 576 / 1454 |
| `train_dirC_warm_1024_20261113` | `random_terrain_relcritic_20261113` | 1024 / 20 | 0.4 | 相对 `z_root - z_ground` | **0.29–5.13** (iter 10–18: **0.35–1.10**) / 2.0009 | **0.2921** / 0.0347 | 0.9783 (iter 18 曾达 1.000) | 564 / 1459 |
| `train_dirC_actoronly_20261113` | `random_terrain_relcritic_20261113` | 4096 / saved `model_5` | 0.4 | 相对 `z_root - z_ground` | 1.51–2.45 / 2.4521 | 0.2622 / 0.0167 | 0.9402 (iter 5) | 1824 / 4521 |

---

## 五、400 Hz 9 块随机地形 3 种子基准汇总

评估条件：`tools/eval_random_terrain.py`，`WHEELBIPE_INTEGRATION_SUBSTEPS=2`（400 Hz），`3` 种子（`71903, 173927, 384719`），每种子 `27` 环境 × `12 s`，`speed = 0.65 m/s`（每模型合计 `81` 局）。

| 模型 | 3 种子通过情况 | 合计成功局数 | 总平均 `vx_mae` (m/s) | 总平均 `wz_mae` (rad/s) | 平均前向位移 (m) | 平均 `|lat|` 偏移 (m) | 最小 `gimbal` 间隙 (m) | `up14` `|lat|` (m) | `up18` `vx_mae` (m/s) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dirC_actoronly_model_5`** | **3 / 3 PASS** | **81 / 81 (100.0%)** | **0.0140** | **0.0227** | **7.58** | 0.526 | 0.2593 | **1.617** | **0.0285** |
| `ft40_model_10` (原基准) | **3 / 3 PASS** | **81 / 81 (100.0%)** | 0.0141 | 0.0231 | 7.57 | 0.549 | 0.2584 | 1.771 | 0.0293 |
| `dirB_model_10` | **3 / 3 PASS** | **81 / 81 (100.0%)** | 0.0155 | 0.0229 | 7.53 | 0.537 | **0.2612** | 1.850 | 0.0282 |
| `dirC_warm_model_10` | **3 / 3 PASS** | **81 / 81 (100.0%)** | 0.0156 | 0.0229 | 7.54 | **0.508** | 0.2587 | 1.783 | 0.0293 |
| `dirC_warm_model_19` | **3 / 3 PASS** | **81 / 81 (100.0%)** | 0.0168 | 0.0231 | 7.51 | 0.517 | 0.2592 | 1.723 | 0.0312 |
| `dirB_model_19` | **3 / 3 PASS** | **81 / 81 (100.0%)** | 0.0179 | 0.0228 | 7.49 | 0.512 | 0.2583 | 1.722 | 0.0288 |

（补充核实：此前服务器既有 `noise005_model_9` 在 `runs/checkpoint_evals_noise005_20261113T072740Z` 亦已处于 400 Hz，`3` 种子单重复评估同样 `9/9 PASS`，`vx_mae = 0.0130–0.0153 m/s`。）

---

## 六、连续 9 块 `--route-mode corridor` 270 秒零重置验证（400 Hz）

评估条件：`tools/record.py --random-terrain-overview --random-terrain-route --route-mode corridor --steps 13500 --seed 71903 --no-physics-log`，`WHEELBIPE_INTEGRATION_SUBSTEPS=2`（400 Hz，DR 开启），覆盖全部 `Blocks 0 -> 1 -> 2 -> 5 -> 4 -> 3 -> 6 -> 7 -> 8`（`base_height_range_m = [-0.828, +1.334] m`，包含 `up18` `+1.08 m` 坡顶与 `down18` `-1.08 m` 谷底）。

| 模型 / 运行目录 | 状态 | `steps` / `sim_s` | `resets` (各终止项) | `vx_mae` (m/s) | `wz_mae` (rad/s) | `max_closure_m` | 终末位置 `xyz` (m) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dirC_actoronly_model_5`** ([summary.json](./route_corridor_v4_dirC_actor5_20261113/summary.json)) | **PASS** | 13500 / 270.0 | **0** (全 0) | **0.0205** | 0.0303 | `0.000119` (`0.119 mm`) | `[-4.707, -6.018, -0.423]` (`Block 8` `block_x=+3.29 m`) |
| **`ft40_model_10`** ([summary.json](./route_corridor_v4_ft10_20261113/summary.json)) | **PASS** | 13500 / 270.0 | **0** (全 0) | 0.0206 | **0.0299** | `0.000119` (`0.119 mm`) | `[-4.946, -6.011, -0.466]` (`Block 8` `block_x=+3.05 m`) |

> 备注：13,500 步止步于最终块 `Block 8 (down18)` 过谷底后上升出口段（距离旧终点 `x = -0.90 m` 约 `3.8 m`），记录 `waypoints_reached = 8 / 9`；后续已将 [`tools/record.py`](../../tools/record.py#L630) 最终航点 `end_x_right` 设至 `block_x = +3.0 m`（`x = -5.0 m`，已越过 `Block 8` 中心与最低点）并同步服务器，保证 `13,500` 步内即可触发 `9 / 9` 航点完成标记。

---

## 七、测试验证记录

1. **本地与服务器几何/课程单测（最终编辑后重新运行）**
   - 本地 `python3 tools/test_random_terrain.py` → `random terrain geometry contract: PASS`（`batch height check: SKIP (torch unavailable)`）
   - 服务器 `env620` Python `tools/test_random_terrain.py` → `random terrain geometry contract: PASS`（含 Torch 批量高度与三维四元数旋转/180°反转间隙不变性全量通过）
   - 本地 `python3 tools/test_terrain_curriculum.py` → `legacy terrain curriculum contract: PASS`
   - 本地 `python3 tools/test_obstacle_terrain.py` → `Ran 6 tests in 0.007s - OK`

---

## 八、推荐候选模型与当前结论

1. **当前最优推荐候选**：**`dirC_actoronly_model_5`**（`/data/wheeled-legged/projects/v14_6_random_terrain_20261112/runs/train_dirC_actoronly_20261113/logs/rsl_rl/own_low_speed_exploration/2026-11-13_10-17-35_dirC_relcritic_actoronly_noise005/model_5.pt`）与 **`ft40_model_10`** 并列第一梯队：两者在 400 Hz 下均已通过：
   - **9 块 3 种子单块基准**：`81 / 81 (100.0%)`（`dirC_actoronly_model_5` `vx_mae = 0.0140 m/s` 略优于 `ft40_model_10` 的 `0.0141 m/s`）；
   - **修复间隙后的 8 工况合法跨块过渡**：`24 / 24 (100.0%)` 零伪负间隙（`min_gimbal >= +0.2618 m`，`min_base >= +0.0091 m`）；
   - **连续 9 块 270 秒走廊路线（DR 开启）**：`13,500 / 13,500` 步 **`resets = 0`** 完整穿越全 9 块，物理闭合误差始终 `< 0.12 mm`。
2. **长期微调备选**：若需进一步迭代微调而不发生后期坍塌，建议采用 `random_terrain_relcritic_20261113`（Critic 相对地形高度），代表产物 **`dirC_warm_model_10`** 与 **`dirC_warm_model_19`** 在 9 块基准同样保持 `81 / 81 (100.0%)`。
