# Wheelbipe V14-6 方向 D（偏航收敛 + 混合楼梯/可攀爬台阶自适应）技术报告

日期：2026-11-14  
本地仓库：`/Users/silver/RL/wheelleg_lab`  
服务器工作区（`pi`）：`/data/wheeled-legged/projects/v14_6_random_terrain_20261112`  
约束遵守：全部训练与评估均以 `WHEELBIPE_INTEGRATION_SUBSTEPS=2`（400 Hz 物理仿真，`dt = 0.0025 s`，`decimation = 8`，`policy_dt = 0.02 s`）仅在用户个人工作区执行，未修改 `/data/isaac620-migration-20261103/env620`、`/data/wheelbipe-v14-6-training-20261108` 或既有基线检查点。

---

## 一、总览与推荐冠军模型

本轮围绕多地形统一适应目标完成了四阶段实验：**Phase 1（既有 400 Hz 策略在楼梯/台阶上的零样本探针）→ Track D1（偏航与斜入过渡强化微调）→ Track D2（3×3 随机坡面 + 下台阶 + 可攀爬低台阶混合训练）→ 全基准 400 Hz 交叉验证与 270 s 连续 Corridor 路线验收**。

### 核心结论

1. **推荐首选多地形冠军模型：`dirD2_model_5`**
   - 检查点（服务器 `pi`）：`logs/rsl_rl/own_low_speed_exploration/2026-11-13_17-36-05_dirD2_stairs_4096_15/model_5.pt`
   - `sha256`：`ca108ded071f2d58ee075304f4805418643645984167be0e489f026e83c0e6fd`
   - 全基准表现（400 Hz）：
     - **3-Seed 3×3 随机坡面（81 局）**：`81 / 81 (100.0%)` PASS，取得全部 8 个对照模型中**最低速度跟踪误差（`vx_mae = 0.0133 m/s`）**与**最高平均前向推进（`fwd = 7.584 m`）**，`wz_mae = 0.0229 rad/s`，`|lat| = 0.515 m`（优于 `dirC_actor5` 的 `0.526 m` 与 `ft40_10` 的 `0.549 m`），`min_gimbal = 0.2585 m`。
     - **跨块过渡（`s71903`，24 合法过渡）**：`24 / 24 (100.0%)` PASS，取得全组**最低 `vx_mae = 0.01650 m/s`**（较 `dirC_actor5` 的 `0.01787 m/s` 下降 `7.7%`）、最高前向推进（`4.943 m`）与最高最小云台离地间隙（`0.2634 m`）。
     - **270 s 连续 Corridor 路线（`13,500` 步，`seed=71903`）**：**`PASS`，`resets = 0`，`9 / 9` 航点全部完成（`completed = True`）**——相较 `ft40_10` 与 `dirC_actor5` 在 270 s 内仅达 `8 / 9`（`completed = False`），取得**全组最低路线偏航误差 `wz_mae = 0.02767 rad/s`**（较 `dirC_actor5` 的 `0.03031 rad/s` 降低 `8.7%`）及 `vx_mae = 0.02051 m/s`。
     - **3-Seed Canonical 障碍评估（54 局）**：下台阶 `stairs_down_{30,60,90}mm` **`27 / 27 (100.0%)` 生存并 `27 / 27 (100.0%)` 到达着陆台**；在最严苛的 `stairs_down_90mm` 上将最小机身离地间隙 `min_base` 从 `dirC_actor5` 的 `0.0028 m` 提升 3.25× 至 **`0.0091 m`**；在 24 局 Extended 障碍（含 `platform_up_{25,40}mm`）上取得全组最高的平台总生存（`7 / 15`）并恢复 `platform_up_25mm` 登顶能力（`1 / 3` 达台，`platform_up_40mm` `2 / 3` 保持直立未倾覆）。

2. **备选低横移模型：`dirD2_model_10`**
   - 检查点（服务器 `pi`）：`logs/rsl_rl/own_low_speed_exploration/2026-11-13_17-36-05_dirD2_stairs_4096_15/model_10.pt`
   - `sha256`：`5e89e6f99312d0e1a51fc204a96c5e74eab6e62e85baa285c8df9472b5a2cef0`
   - 在 81 局随机坡面上取得全组**最低平均横向偏移（`|lat| = 0.492 m`）**，同时保持 `81 / 81` 坡面、`24 / 24` transition、`9 / 9` 270 s corridor 全完赛（`wz_mae = 0.02795 rad/s`）与 `27 / 27` 下台阶全达台（`stairs_down_90mm` `vx_mae = 0.0780 m/s`，`min_base = 0.0094 m`）。

3. **重要发现与物理边界确认**
   - **Track D1 直接解锁 270 s Corridor 全航点完赛**：收紧偏航跟踪奖励（`ang_vel_z_sigma: 0.09 -> 0.045`，`track_ang_vel_z: 2.0 -> 3.0`）并引入 `±0.45 rad` 斜入边界采样后，所有 Track D 检查点（`dirD1_m10`, `dirD1_m11`, `dirD2_m5`, `dirD2_m10`, `dirD2_m14`）全部从 `8 / 9` 跃升为 **`9 / 9` 完赛（`resets = 0`）**。
   - **下台阶与垂直上台阶的能力分野**：轮腿当前 35 维纯本体感知（盲走）Actor 已具备 **`100%` 下台阶通行能力（`30/60/90 mm`）**；但面对无倒角的 `90°` 垂直上台阶，受轮半径 $R = 0.0625\text{ m}$ 与前导杆碰撞包络及**Actor 缺乏前视地形高度输入**的物理/观测双重限制，`≥ 60 mm` 台阶无法由单纯轮速推动爬升，`25–40 mm` 台阶在盲走双轮异步触阶时会因滚转扰动（`causes=['roll']`）产生较大失稳风险（仅 `1 / 3` 稳定登顶 `25 mm`）。

---

## 二、代码、环境与 Profile 变更清单

| 文件 | 本轮核心变更 |
| --- | --- |
| [`profiles.json`](../../profiles.json) | 新增 `random_terrain_yaw045_20261113`（Track D1）与 `random_terrain_stairs_20261113`（Track D2）。D1 将 `track_ang_vel_z=3.0`、`ang_vel_z_sigma=0.045`、`random_terrain_transition_yaw_rad=0.45`；D2 进一步配置 `random_terrain_stairs_lanes=true`、`random_terrain_start_fraction=0.2`、`random_terrain_transition_fraction=0.3`、`random_terrain_stairs_fraction=0.3`、`random_terrain_transition_yaw_rad=0.35`、`lin_vel_z=-1.5`、`flight_penalty=-2.0`。 |
| [`random_terrain_map.py`](../../project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/random_terrain_map.py) | 注册 D1/D2 配置读取；新增 `_load_obstacle_terrain()`、`_stair_lane_specs()`、`_stairs_lane_heights()`；将 4 条附加障碍训练跑道定义为 `STAIRS_LANE_KINDS = ("stairs_down", "stairs_down", "platform_up", "platform_up")`、`STAIRS_LANE_STEP_HEIGHTS_M = (0.035, 0.070, 0.025, 0.040)`；支持 `random_terrain_transition_yaw_rad` 与 `random_terrain_stairs_fraction` 采样并将 `stairs_spawns` 写入 `health()`。 |
| [`tools/eval_obstacle_terrain.py`](../../tools/eval_obstacle_terrain.py) | 新建 400 Hz 障碍专项评估脚本，支持 `--suite {canonical,extended}`：`canonical` 评估 6 跑道（`stairs_down_{30,60,90}mm` + `platform_up_{60,120,180}mm`），`extended` 追加 `platform_up_{25,40}mm` 共 8 跑道；自动统计 `reached_landing_count`、`failure_causes`、接触碰撞刚体、及 `min_gimbal/min_base` 间隙。 |
| [`tools/test_random_terrain.py`](../../tools/test_random_terrain.py) | 新增混合地形中 2 条 `stairs_down` 与 2 条 `platform_up` 跑道的几何高度采样回归测试；在本地与服务器 `env620` 均运行 `PASS`。 |

---

## 三、Phase 1：既有策略 400 Hz 障碍零样本基准

在开始 Track D1/D2 训练前，对两份已验证的 400 Hz 基线模型 `ft40_model_10`（`sha256: 94b2f591...`）和 `dirC_actoronly_model_5`（`sha256: 0d9eaa26...`）进行了系统化零样本障碍基准测试（结果归档于 [`phase1_obstacle_20261113/`](./phase1_obstacle_20261113/) 与 [`phase1_obstacle_extended_20261113/`](./phase1_obstacle_extended_20261113/)）。

### 3.1 Canonical 6 跑道（3 种子 `71903, 173927, 384719`，每模型 54 局）

| 工况 | `ft40_model_10` 生存 / 达台 | `ft40_model_10` `vx_mae` / `min_base` | `dirC_actor5` 生存 / 达台 | `dirC_actor5` `vx_mae` / `min_base` |
| --- | :---: | :---: | :---: | :---: |
| `stairs_down_30mm` | **9 / 9** · **9 / 9** | `0.0149` / `0.0233 m` | **9 / 9** · **9 / 9** | `0.0166` / `0.0238 m` |
| `stairs_down_60mm` | **9 / 9** · **9 / 9** | `0.0223` / `0.0302 m` | **9 / 9** · **9 / 9** | `0.0251` / `0.0382 m` |
| `stairs_down_90mm` | **9 / 9** · **9 / 9** | `0.0789` / `0.0093 m` | **9 / 9** · **9 / 9** | `0.0751` / **`0.0028 m`** |
| `platform_up_60mm` | 2 / 9 · 0 / 9 | `0.2975` / `0.0068 m` | 2 / 9 · 0 / 9 | `0.3057` / `0.0075 m` |
| `platform_up_120mm` | 0 / 9 · 0 / 9 | `0.2427` / `0.0003 m` | 0 / 9 · 0 / 9 | `0.2781` / `-0.0001 m` |
| `platform_up_180mm` | 0 / 9 · 0 / 9 | `0.2179` / `0.0205 m` | 0 / 9 · 0 / 9 | `0.2069` / `0.0330 m` |

### 3.2 Extended 8 跑道探针（`seed=71903`，追加 `25mm` 与 `40mm` 低台阶，24 局）

- **`dirC_actoronly_model_5`**：`platform_up_25mm` 生存 `1 / 3`、达台 `1 / 3`（2 次失败原因为 `roll`）；`platform_up_40mm` 生存 `2 / 3`、达台 `0 / 3`（1 次 `roll`）；`60mm` `2 / 3`（`contact/roll` `base_link`）；`120mm` `0 / 3`（`base_link`, `left_buttom_guide1_link`, `left_front_guide1_link`, `left_front_guide3_link`）。
- **`ft40_model_10`**：`platform_up_25mm` `1 / 3` 达台 `1 / 3`（`roll`）；`platform_up_40mm` `1 / 3` 达台 `1 / 3`（`roll`）；`60mm` `2 / 3`（`left_front_guide1_link`）。
- **结论**：`25–40 mm` 对 `62.5 mm` 半径轮胎**物理可攀爬**，但既有模型稳定达台率仅为 `1/3`；而 `60 mm` 接近轮半径极限，`120/180 mm` 垂直立面必然由车轮与前/底连杆（`front_guide*`/`buttom_guide*`/`base_link`）阻挡卡死。

---

## 四、Track D1（`dirD1_yaw045_4096_12`）与 Track D2（`dirD2_stairs_4096_15`）训练审计

两轮微调均从已有收敛策略出发，采用**4096 并行环境、400 Hz（`dt=0.0025 s`）、`lr=1e-5`、`--actor_only_finetune --finetune_noise_std 0.05`、`--audit_updates`** 安全微调方案。

| 训练阶段 | Profile & 父检查点 | 环境与迭代 | 训练健康度（`training_health.json`） | PPO 更新轨迹（`training_updates.jsonl`） |
| --- | --- | --- | --- | --- |
| **Track D1** (`dirD1_yaw045_4096_12`) | `random_terrain_yaw045_20261113`<br>父：`dirC_actoronly_model_5` | `4096` envs<br>`12` iters | `max_loop_error = 0.001148 m`<br>`peak_initial_tire_penetration = 0.0 m`<br>`transition_spawns = 2003 / 5016 (39.9%)` | `Mean reward`: `1.45 -> 42.31`（iter 11/12，`success_rate=0.9761`）<br>`value_loss`: `1.84 -> 2.20`<br>终轮累计 `actor_delta_l2 = 0.0230`、`critic_delta_l2 = 0.5314` |
| **Track D2** (`dirD2_stairs_4096_15`) | `random_terrain_stairs_20261113`<br>父：`dirD1_model_11` | `4096` envs<br>`15` iters | `max_loop_error = 0.003970 m`<br>`latest_loop_error = 0.000127 m`<br>`peak_initial_tire_penetration = 0.0 m`<br>`stairs_spawns = 1684 / 5571 (30.2%)`<br>`transition_spawns = 1645 / 5571 (29.5%)`<br>`run_up_spawns = 1148 / 5571 (20.6%)` | `Mean reward`: `1.37 -> 47.60`（iter 14/15，`success_rate=0.9558`，iter 13 峰值 `0.9727`）<br>`value_loss`: `2.04 -> 3.63`<br>终轮累计 `actor_delta_l2 = 0.0247`、`critic_delta_l2 = 0.6835` |

> **D2 父检查点选型说明（Track D1 对比结果）**：`dirD1_model_5/10/11` 均在 3 种子 3×3 坡面取得 `81/81` PASS 且在 24 个合法跨块过渡取得 `24/24` PASS；其中 `dirD1_model_11`（`sha256: f498abcf...`）取得 D1 最低横向偏移 `|lat| = 0.494 m`（`dirD1_m10` `0.532 m`、`dirD1_m5` `0.562 m`、`dirC` `0.526 m`）和最低 `wz_mae = 0.0225 rad/s`，并在 `stairs_down_90mm` 表现最佳（canonical `vx_mae = 0.0786`、extended `min_base = 0.0210 m`），故被选定为 Track D2 的父检查点。

---

## 五、全模型 400 Hz 交叉基准对照结果

### 5.1 3×3 随机坡面（3 种子 = 81 局）、Transition（24 合法 + 6 峭壁）与 270 s Corridor 连续路线

> 注：坡面与 Transition 数据读取自 `eval_random_terrain.py` 与 `eval_terrain_transitions.py`（均采用纯 3×3 坡面地图以确保公平对比）；Corridor 数据来自 `tools/record.py --random-terrain-route --route-mode corridor --steps 13500 --seed 71903`（270.0 s 仿真，`108,000` 物理子步，均满足零重置与 `< 0.14 mm` 闭环误差）。

| 模型 | 3-Seed 坡面 (81局) | 坡面 `vx_mae` | 坡面 `wz_mae` | 坡面 `\|lat\|` | 坡面 `min_gimbal` | Transition (24合法) `vx_mae` / `min_base` | 270 s Corridor 航点完赛 (`resets`) | Corridor `vx_mae` / `wz_mae` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ft40_model_10` | **81 / 81** | 0.0141 | 0.0231 | 0.549 m | 0.2584 m | 24/24 · `0.01743` / `0.0092 m` | 8 / 9 (`False`) · 0 | `0.02055` / `0.02987` |
| `dirC_actoronly_model_5` | **81 / 81** | 0.0140 | 0.0227 | 0.526 m | 0.2593 m | 24/24 · `0.01787` / `0.0091 m` | 8 / 9 (`False`) · 0 | **`0.02045`** / `0.03031` |
| `dirD1_model_5` | **81 / 81** | 0.0137 | 0.0231 | 0.562 m | **0.2594 m** | 24/24 · `0.01699` / `0.0091 m` | — | — |
| `dirD1_model_10` | **81 / 81** | 0.0138 | 0.0229 | 0.532 m | 0.2588 m | 24/24 · `0.01680` / `0.0088 m` | **9 / 9 (`True`)** · 0 | `0.02059` / `0.02860` |
| `dirD1_model_11` | **81 / 81** | 0.0138 | **0.0225** | 0.494 m | 0.2580 m | 24/24 · `0.01732` / `0.0086 m` | **9 / 9 (`True`)** · 0 | `0.02066` / `0.02885` |
| **`dirD2_model_5`** | **81 / 81** | **0.0133** | 0.0229 | 0.515 m | 0.2585 m | 24/24 · **`0.01650`** / `0.0090 m` | **9 / 9 (`True`)** · 0 | `0.02051` / **`0.02767`** |
| **`dirD2_model_10`** | **81 / 81** | 0.0141 | 0.0227 | **0.492 m** | 0.2588 m | 24/24 · `0.01695` / `0.0088 m` | **9 / 9 (`True`)** · 0 | `0.02099` / `0.02795` |
| `dirD2_model_14` | **81 / 81** | 0.0141 | 0.0227 | 0.505 m | 0.2579 m | 24/24 · `0.01743` / `0.0086 m` | **9 / 9 (`True`)** · 0 | `0.02084` / `0.02801` |

**Corridor 关键对比分析**：
- Track D1 与 Track D2 的全数测试检查点均在 `270.0 s` 内跑完 `9 / 9` 全部航点（`resets = 0`），而 `ft40_10` 与 `dirC_actor5` 均因坡间弧线与入弯滞后略大只到达 `8 / 9`（尽管也零跌倒）。
- **`dirD2_model_5`** 在 Corridor 上取得全部模型最低偏航率跟踪误差 **`0.02767 rad/s`**（相比 `dirC_actor5` 改善 `8.7%`），在 `dirD2_m14` 上 `max_closure_m` 低至 `0.1160 mm`。
- 在 Transition Probe 的 `crest_wall_probes_summary`（故意设置的坡脊 `90°` 硬拐急墙探针）中，所有模型的失败数仍在 `3/6`–`4/6` 区间，证实该探针代表几何撞壁而非合法路线能力缺陷。

### 5.2 Track D2 3-Seed Canonical 障碍基准（54 局/模型）与 Extended 1-Seed（24 局/模型）

| 模型 | Canonical 下台阶总计 (`30/60/90 mm`, 27局) | `stairs_down_30mm` (`vx_mae` / `min_base`) | `stairs_down_90mm` (`vx_mae` / `min_base`) | Canonical 台阶总生存 (`60/120/180 mm`, 27局) | Extended 台阶总生存 (`25/40/60/120/180 mm`, 15局) | `platform_up_25mm` (`s71903`) 生存 / 达台 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ft40_model_10` | **27/27** 生存 · **27/27** 达台 | `0.0149` / `0.0233 m` | `0.0789` / `0.0093 m` | 2 / 27 (reach 0) | 5 / 15 | 1 / 3 · **1 / 3** |
| `dirC_actoronly_model_5` | **27/27** 生存 · **27/27** 达台 | `0.0166` / `0.0238 m` | **`0.0751`** / `0.0028 m` | 2 / 27 (reach 0) | 6 / 15 | 1 / 3 · **1 / 3** |
| `dirD1_model_10` (`s71903` 9/15局) | (9/9) | — | — | (1/9) | 3 / 15 | 0 / 3 · 0 / 3 |
| `dirD1_model_11` (`s71903` 9/15局) | (9/9) | — | — | (2/9) | 5 / 15 | 0 / 3 · 0 / 3 |
| **`dirD2_model_5`** | **27/27** 生存 · **27/27** 达台 | **`0.0129`** / `0.0455 m` | `0.0909` / **`0.0091 m`** | **5 / 27** (reach 0) | **7 / 15** | 1 / 3 · **1 / 3** |
| **`dirD2_model_10`** | **27/27** 生存 · **27/27** 达台 | `0.0146` / **`0.0464 m`** | `0.0780` / **`0.0094 m`** | 3 / 27 (reach 0) | 4 / 15 | 1 / 3 · **1 / 3** |
| `dirD2_model_14` | **27/27** 生存 · **27/27** 达台 | `0.0134` / `0.0460 m` | `0.1047` / **`0.0099 m`** | 3 / 27 (reach 0) | 5 / 15 | 1 / 3 · **1 / 3** |

---

## 六、失败机理定位与后续方向（方向 E）建议

结合 [`dirD2_stairs_20261113/m5_obs_ext_s71903/summary.json`](./dirD2_stairs_20261113/m5_obs_ext_s71903/summary.json) 与 `m10` 的逐局坐标和碰撞刚体记录，可明确给出当前**唯一尚未彻底攻克的瓶颈（垂直上台阶）**的机理证据：

1. **盲走策略在 `platform_up_25mm / 40mm` 上的失稳机制（`roll` 为主）**
   - 在 `platform_up_25mm` 上，`dirD2_model_5` 的 `env=19` 成功跑完台阶并达台（`max_x = 6.692 m`, `reach=True`），但 `env=3, 11` 分别在首级台阶沿附近（`max_x = 3.22 m`）以 `causes=['roll']`、`bodies=[]`（无导杆撞壁）翻倒；`dirD2_model_10` 的 `env=19` 爬上第一级后在 `max_x = 4.161 m` 以 `roll` 翻倒。
   - **原因**：Domain Randomization 下左右轮刚度/阻尼/摩擦与触阶相位略微不一致，而 Actor 没有前向高度扫面观测（35 维纯本体），只能在轮子已被 `90°` 阶沿顶阻产生滚转力矩后才被动响应。
2. **在 `platform_up_40mm / 60mm` 上的顶死与硬件碰撞边界**
   - 在 `platform_up_40mm` 上，`dirD2_model_5` 的 `env=12, 20` 均保持直立无跌倒但阻滞于阶前 `x = 3.033–3.039 m`（`reach=False`）；在 `platform_up_60mm` 上，`env=5, 13` 阻滞于 `x = 3.021–3.026 m`，而 `env=21` 则触发 `['left_front_guide1_link', 'right_front_guide1_link']` `contact` 终止。
   - 物理原因在于静态轮胎中心高度等于轮半径 $R = 0.0625\text{ m}$：当台阶高度 $h \ge 40\text{ mm}$ 且无预抬腿/重心前移爆发策略时，纯轮扭矩将车轮持续推向垂直接触面产生向下锁死分力，而 $h \ge 60\text{ mm}$ 时前四连杆导杆 `front_guide1_link` 会直接撞上阶沿。

### 下一步（方向 E）最值得推进的设计

1. **为 Actor 引入低维前向地形先验（Exteroceptive Height-Scan / Foothold Preview）**：当前 Critic 已通过 `critic_relative_height_obs` 使用局部地形高度信息；若要可靠攀爬 `25–60 mm` 台阶与复杂离散障碍，下一步应先将前置极简高程扫描（如机器人前方 `0.05–0.60 m` 的 1D/3-ray 相对地面高度）接入新 Actor（从头训练或蒸馏+微调），使策略能在触阶前主动收腿抬轮或调高机身。
2. **验证前导杆与轮心包络的物理越障极限**：对于 `platform_up >= 120 mm`（$\approx 2 R$），任何纯轮滚动作在不抬离轮地接触（跳跃/单腿跨越）的情况下受连杆干涉均不可行，应将离散墙类课程上限控制在物理可越区间或显式接入跳跃门控动作。

---

## 七、本地归档目录与同步路径

所有训练日志、健康检查与 400 Hz 评估 JSON 已完整同步至本地：
- Phase 1 障碍基准：[`phase1_obstacle_20261113/`](./phase1_obstacle_20261113/) 与 [`phase1_obstacle_extended_20261113/`](./phase1_obstacle_extended_20261113/)
- Track D1 训练与评估：[`dirD1_yaw045_train/`](./dirD1_yaw045_train/) 与 [`dirD1_yaw045_20261113/`](./dirD1_yaw045_20261113/)
- Track D2 训练与评估：[`dirD2_stairs_train/`](./dirD2_stairs_train/) 与 [`dirD2_stairs_20261113/`](./dirD2_stairs_20261113/)
- 270 s Corridor 连续路线基线与 D1/D2 对比：[`corridor_v4_baselines/`](./corridor_v4_baselines/) 与 [`corridor_20261113/`](./corridor_20261113/)
