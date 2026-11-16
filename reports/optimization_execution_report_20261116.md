# Wheelbipe V14-6 Stage 4 模型优化与交付报告（2026-11-16）

> **物理与接口约束（全程保持不变）**：
> - `35-dim Actor / 78-dim Critic / 6-dim Action`，`[256, 128, 64]` ELU MLP；
> - V14-6 USD（`20.66 kg`），显式气弹簧 `400–580 N`，`cylinder_wheels: true`，Domain Randomization 开启；
> - `400 Hz` 仿真控制步率（`WHEELBIPE_INTEGRATION_SUBSTEPS=2`，`dt = 0.0025 s`，`decimation = 8`）；
> - 闭环几何约束：`max_loop_error_m < 0.002 m`（全部通过局）。
> - 未改动共享环境 `env620`（`/data/isaac620-migration-20261103/env620`）、GPU 4 与 GPU 6（`env51`）。

---

## 一、交付结论（Stage 4 Champion）

本次正式交付 **Stage 4 高速全地形机动 / 急停急转冠军检查点 `dirI3_m30`**；原 Stage 3 低速走廊基线 **`dirG1_m20`** 继续冻结存档，互不覆盖。

| 角色定位 | 检查点标识 | SHA256 | 服务器路径与定位 |
| :--- | :--- | :--- | :--- |
| **Stage 4 交付主冠军** | **`dirI3_m30`** | `fe467c9e21fc16ae68ce2317087c249d46cebb2792e62e7039eea8d6f0cd2f74` | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/runs/train_dirI3_fromH8m39_4096_60/logs/rsl_rl/own_low_speed_exploration/2026-11-16_05-41-14_dirI3_fromH8m39_4096_60/model_30.pt` |
| **Stage 4 零崩溃同门对照** | `dirI6_m20` | `9db68c7a20445334a5d79d3ef4e9dd7e1401793ede6d4f0840247dc0dc98f6eb` | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/runs/train_dirI6_fromI3m30_4096_35/logs/rsl_rl/own_low_speed_exploration/2026-11-16_08-12-31_dirI6_fromI3m30_4096_35/model_20.pt` |
| **Stage 4 单项急停/下坡对照** | `dirI4_m10` | `823882a4d8debbf0a21680b99eb3cf6dc62bd45e4f766d2aa95f9a07adf9dd77` | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/runs/train_dirI4_fromI3m40_4096_50/logs/rsl_rl/own_low_speed_exploration/2026-11-16_06-40-31_dirI4_fromI3m40_4096_50/model_10.pt` |
| **Stage 3 冻结低速走廊基线** | `dirG1_m20` | `a327c331c08701aebe048f061d0608eeb39a48f5fba992bb37a714ae65691a0e` | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/runs/train_dirG1_fromE2_steep_agile_4096_25/logs/rsl_rl/own_low_speed_exploration/2026-11-14_14-08-36_dirG1_fromE2_steep_agile_4096_25/model_20.pt` |

**选型依据（`dirI3_m30` 胜出原因）**：
1. 在 3 种子 `highspeed`（`121/135 = 89.6%`）与 3 种子 `standard`（`118/126 = 93.7%`）总通过率上与 `dirI6_m20` 持平，但 **`<25°` 干净高速通过数更高**（`115/135 = 85.2%` vs `111/135`）。
2. 在 `flat_v200_sturn075`（`s71903` 取得 `3/3` `<25°` 干净通过）和 `slopes_v135` 速度跟踪误差（`vx_mae = 0.090–0.105` vs `dirI6_m20` 的 `0.145`）上领先。
3. 已完整跑通全部 9 个评测套件的 3 种子验证（`slopes_v065`、`transitions`、`canonical down-stairs`、`slopes_v135`、`steep_v065`、`yaw080`、`yaw080_sturn`、`standard`、`highspeed`）。

---

## 二、问题定位与 Stage 4 工程修复（`dirI3`–`dirI6`）

1. **`2.0 m/s` 阶跃过冲与前倾栽头诊断（`diag_dirI3_m40_flat_v200`）**：
   - 在 `0 -> 2.00 m/s` 瞬时阶跃指令下，策略产生 `-18.8°` 机身俯仰前倾（前腿触及 `-0.30 rad` 限位），实际前向速度冲过头至 `2.51–2.54 m/s`（`undesired_contact = -0.04`）。
   - 代码核查 [`env.py`](file:///Users/silver/RL/wheelleg_lab/project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py#L3743) 确认二次速度跟踪项公式为 `rew_track_lin_vel_xy_square = torch.square(lin_vel_err * self.cfg.lin_vel_xy_square_sigma)`（`sigma` 位于乘法分子而非分母），导致 `0.35–0.40` 的配置把超速二次惩罚削弱了约 `10×`（`-0.25..-0.32 * err^2`）。
2. **数值崩溃根因与平滑地形限定修复（`dirI4` / `dirI5` → `dirI6`，提交 `7ed982e`）**：
   - `dirI4` 在第 22 轮出现 `FloatingPointError` 与 105 次闭环峰值崩溃，`dirI5` 在第 2 轮崩溃，原因是 `boost_command_high_speed` 对零速出生在楼梯、地形过渡区及 `±30%` 陡坡上的环境无差别施加了 `35%` 比例的 `[1.55–2.05 m/s]` 高速指令。
   - 提交 [`7ed982e`](file:///Users/silver/RL/wheelleg_lab/project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/random_terrain_map.py) 修改 `random_terrain_map.py`、`experiment.py`、`profiles.json`（新增 `random_terrain_stage4_dirI5_20261116` 与 `random_terrain_stage4_dirI6_20261116`）以及 `tools/test_stage3_curriculum.py`，将高速 boost 掩码限制在平滑地形块上，随后 `dirI6`（从 `dirI3_m30` 续训，4096 envs，35 iters）实现 **35/35 零崩溃完成（`exit_code: 0`）**，`cause_numerical = 0.0000`、`cause_closed_loop = 0.0000`，`Mean reward` 峰值达 `+71.38`。

---

## 三、全量多随机种子对比结果

### 3.1 核心套件对比（`dirI3_m30` vs `dirI6_m20` vs `dirI4_m10` vs `dirG1_m20`）

| 评测套件 | `dirG1_m20`（Stage 3 基线） | **`dirI3_m30`（Stage 4 交付主冠军）** | `dirI6_m20`（平滑 boost 同门） | `dirI4_m10`（参考） |
| :--- | :---: | :--- | :--- | :--- |
| **`slopes_v065`** (`0.65 m/s`) | — | **3 种子 `81/81 PASS`** (`27/27 × 3`，`vx_mae=0.102–0.107`，`maxR=3.76–6.39°`，`maxP=4.08–4.94°`，`min_gimbal=0.256–0.264 m`) | `s71903: 27/27 PASS` (`maxR=3.64°`, `maxP=4.17°`, `vx_mae=0.105`) | `s71903: 27/27` (`maxR=10.90°`) |
| **`transitions`** | — | **`PASS`** (`valid_transitions_summary: 8/8 cases, 24/24 episodes, 0 fail`, `vx_mae=0.104`, `min_gimbal=0.257 m`; `crest_probes`: 2 cases / 6 eps, 4 fails) | `24/24 PASS` | `PASS` |
| **`canonical down-stairs`** (30/60/90 mm) | — | **3 种子 `27/27 PASS (0 fail)`** (`9/9 + 9/9 + 9/9`; `stairs_max_abs_pitch_deg=16.08–36.75°`) | `s71903: 9/9 PASS` (`stairs_max_abs_pitch_deg=20.07°`) | `s71903: 8/9` |
| **`slopes_v135`** (`1.35 m/s`) | `70/81 (86.4%)` | **3 种子 `79/81 (97.5%)`** (`26/27 + 27/27 + 26/27`，`vx_mae=0.090–0.105`) | `s71903: 26/27` (`vx_mae=0.145`) | `s71903: 26/27` |
| **`steep_v065`** (`±10/18/24/30%`) | `113/135 (83.7%)` | **3 种子 `75/81 (92.6%)`** (`24/27 + 24/27 + 27/27 PASS`) | `s71903: 24/27` (`vx_mae=0.125`, `maxP=6.1°`) | — |
| **`yaw080`** (`0.80 rad/s`) | `59/81 (72.8%)` | **3 种子 `57/81 (70.4%)`** (`19/27 + 18/27 + 20/27`) | — | — |
| **`yaw080_sturn`** (`0.80 rad/s` S 弯) | `62/81 (76.5%)` | **3 种子 `73/81 (90.1%)`** (`23/27 + 24/27 + 26/27`) | `s71903: 24/27` | — |
| **`standard`** (`14×3×3 = 126` 局) | — | **3 种子 `118/126 (93.7%)`**（`clean<25°: 112/126 = 88.9%`；分种子 `39/42, 39/42, 40/42`） | **3 种子 `118/126 (93.7%)`**（`clean: 113/126`；分种子 `38/42, 39/42, 41/42`） | `s71903: 41/42 clean` (`down10_v065_yaw080: 3/3`) |
| **`highspeed`** (`15×3×3 = 135` 局) | — | **3 种子 `121/135 (89.6%)`**（**`clean<25°: 115/135 = 85.2%`**；分种子 `41/45, 38/45, 42/45`） | **3 种子 `121/135 (89.6%)`**（`clean: 111/135`；分种子 `42/45, 38/45, 41/45`） | `s71903: 41/45` (`36 clean`; `flat_estop_200: 3/3`) |

### 3.2 `dirI3_m30` `standard` / `highspeed` 细分场景表现（3 种子 9 局口径）

- **`standard` 与 `highspeed` 全胜（`9/9` clean `<25°`）场景**：
  - `0.5–1.5 m/s` 全部直行项与原地/行进转向项（`yaw 0.5 / 1.0 / 1.5 rad/s`）；
  - `flat_v100`、`flat_v150`、`up10_v150`、`down10_v150`、`down18_v150`、`up18_v120`；
  - `flat_v150_yaw100`（`1.50 m/s + 1.00 rad/s`）、`up10_v065_yaw080`、`down10_v120_sturn080`（下坡 S 弯）；
  - **`flat_estop_150`**（`1.50 -> 0 m/s` 急停：`stop_d = 0.51–0.57 m`, `hold = 0.013–0.024`）；
  - **`down10_estop_150`**（`-10%` 下坡急停：`stop_d = 0.43–0.54 m`, `hold = 0.017–0.032`）；
  - **`flat_reversal_150_n080`**（`+1.50 -> -0.80 m/s` 急反向）。
- **部分失败 / 边界项（3 种子 9 局）**：
  - `flat_v150_sturn`: `8/9 clean`；
  - `up18_v150`: `6/9 (5 clean)`；
  - `flat_estop_200` (`2.00 -> 0 m/s`): `7/9 (5 clean)`；
  - `flat_v200` (`2.00 m/s`): `5/9 (3 clean)`；
  - `flat_v200_sturn075`: `5/9 (4 clean)`（其中 `s71903` 为 `3/3` clean）；
  - `down10_v065_yaw080`: `5/9 (1 clean)`（对应单项可参考 `dirI4_m10` 在 `s71903` 的 `3/3`）。

---

## 四、Stage 4 Showcase 实机视频与关键帧校验

全部 4 条视频均由 `dirI3_m30` 录制并经关键帧目视检查确认机器人姿态端正、地形清晰且 `resets=0`、`closure <= 0.125 mm`：

| 视频文件 | 大小 | 实测指标 | 代表关键帧 |
| :--- | :---: | :--- | :--- |
| [dirI3_m30_flat_v160_yaw080_10s.mp4](file:///Users/silver/RL/wheelleg_lab/videos/stage4_showcase_20261116/dirI3_m30_flat_v160_yaw080_10s.mp4) | `6.52 MB` | `PASS`, `resets=0`, `closure=0.097 mm`, `vx=1.55`, `vx_err=0.067` | [frame_0250.png](file:///Users/silver/RL/wheelleg_lab/videos/stage4_showcase_20261116/keyframes/dirI3_m30_flat_v160_yaw080_10s/frame_0250.png) |
| [dirI3_m30_row0_v150_24s.mp4](file:///Users/silver/RL/wheelleg_lab/videos/stage4_showcase_20261116/dirI3_m30_row0_v150_24s.mp4) | `14.56 MB` | `PASS`, `resets=0`, `closure=0.111 mm`, `vx=1.43`, `vx_err=0.067` | [frame_0750.png](file:///Users/silver/RL/wheelleg_lab/videos/stage4_showcase_20261116/keyframes/dirI3_m30_row0_v150_24s/frame_0750.png) |
| [dirI3_m30_sequence_15s.mp4](file:///Users/silver/RL/wheelleg_lab/videos/stage4_showcase_20261116/dirI3_m30_sequence_15s.mp4) | `7.75 MB` | `PASS`, `resets=0`, `closure=0.125 mm`, `vx=0.57`, `vx_err=0.111` (`5s straight -> 5s turn -> 5s reverse`) | [frame_0500.png](file:///Users/silver/RL/wheelleg_lab/videos/stage4_showcase_20261116/keyframes/dirI3_m30_sequence_15s/frame_0500.png) |
| [dirI3_m30_stairs70mm.mp4](file:///Users/silver/RL/wheelleg_lab/videos/stage4_showcase_20261116/dirI3_m30_stairs70mm.mp4) | `4.23 MB` | `PASS`, `resets=0`, `closure=0.119 mm`, `vx=0.56`, `vx_err=0.101` (`random_terrain_stage4_dirI3_20261116` 混合楼梯) | [frame_0260.png](file:///Users/silver/RL/wheelleg_lab/videos/stage4_showcase_20261116/keyframes/dirI3_m30_stairs70mm/frame_0260.png) |

---

## 五、能力边界与下一步范围说明

1. **`2.0 m/s` 阶跃边界**：`0–1.60 m/s` 全地形高速、转向、S 弯、急停与急反向已达全场景 `9/9` 稳态；`0 -> 2.00 m/s` 无坡度斜坡/直接阶跃指令在最差 DR 采样（延迟/低力矩/外推扰动）下仍存在约 `4/9` 失败风险（前倾 `~-18°` 与 `2.3–2.5 m/s` 过冲），建议实车上层规划加入 `vx` 加速度斜坡限制或保留在 `<= 1.60 m/s` 全自主区间。
2. **270 s 单环境走廊评测**：由于单环境 270 s 走廊评估实测耗时长约 `108 min`（`6487.8 s` wall clock），Stage 4 未重复运行该长跑评测；`dirG1_m20` 继续作为已通过 270 s 走廊验证的低速对照基线保留，不得将 Stage 3 走廊成绩直接套用于 Stage 4。
3. **Stage 5 后续方向**：本阶段明确未纳入的 `up-platform`（`25 / 40 mm` 主动跳/爬上台阶）留待 Stage 5 专项推进；同时 `lin_vel_xy_square_sigma` 乘法分子公式与 `down10_v065_yaw080` 下坡急转稳定度可作为后续精调切入点。

---

## 六、检查点台账与仓库追踪状态

- 检查点 SHA256 台账已更新至 [checkpoints_sha256.txt](file:///Users/silver/RL/wheelleg_lab/logs/checkpoints_sha256.txt)（含 `dirH7_m15`、`dirH8_m39`、`dirI1a_m20`、`dirI3_m30`、`dirI3_m40`、`dirI3_m50`、`dirI4_m10`、`dirI4_m20`、`dirI6_m20`、`dirI6_m30`、`dirI6_m34`）；因 `.gitignore` 规则，[`logs/`](file:///Users/silver/RL/wheelleg_lab/.gitignore#L5) 与 [`/videos/`](file:///Users/silver/RL/wheelleg_lab/.gitignore#L23) 仅本地保留，不纳入 Git 提交。
