# Stage 3 (`dirH1`) 训练与全量门禁回归报告 (2026-11-15)

## 1. 训练摘要 (`dirH1_curriculum_4096_140`)

- **Profile**: `random_terrain_stage3_curriculum_20261115`
- **起点**: `dirG1_m20` (`sha256: a327c331c08701aebe048f061d0608eeb39a48f5fba992bb37a714ae65691a0e`)
- **物理与接口**: `35D Actor / 78D Critic / 6D Action`，V14-6 USD (`20.66189 kg`)，`400–580 N` 气弹簧，`cylinder_wheels: true`，DR ON，`400 Hz` (`WHEELBIPE_INTEGRATION_SUBSTEPS=2`, `physics_dt=0.0025 s`, `policy_dt=0.02 s`)
- **优化超参**: `schedule=fixed`, `lr=2e-5`, `clip_param=0.12`, `entropy_coef=0.0006`, `init_noise_std=0.12`, `4096 envs`
- **进程与峰值**: 实际完成 `iter 0..107`（保存 `model_0 / 20 / 40 / 60 / 80 / 100.pt`），在 `iter 108` 因 `FloatingPointError: Nonfinite native/policy joint state` 中止。训练峰值位于 **`iter 60`**（`Train/mean_reward = 118.218`, `Train/mean_episode_length = 724.85`, `Metrics/success_rate = 0.770`）；`iter 80–107` 全量课程叠加（`2.0 m/s` + `±1.5 rad/s` + `45%` 陡坡）后 `cause_roll` 与 `cause_contact` 上升。

## 2. Wave 1 Checkpoint 扫描 (`dirH1_m20..m100`, `seed=71903`)

| Checkpoint | `steep_v065` (±30%, 81 eps) | `speed_sweep` (42 eps) | `flat_v050` 实测均速 | `max_loop` (`steep`) |
|---|---:|---:|---:|---:|
| `dirG1_m20` (基线) | `69/81 (85.2%)` | `31/42` | `0.48 m/s` | `—` |
| `dirH1_m20` | `68/81 (84.0%)` | **`37/42`** | `0.47 m/s` | `0.00117 m` |
| `dirH1_m40` | `69/81 (85.2%)` | `34/42` | `0.46 m/s` | `0.00111 m` |
| **`dirH1_m60`** | **`74/81 (91.4%)`** | `34/42` | `0.43 m/s` | **`0.00018 m`** |
| `dirH1_m80` | `72/81 (88.9%)` | `34/42` | `0.41 m/s` | `0.00044 m` |
| `dirH1_m100` | `68/81 (84.0%)` | `34/42` | `0.38 m/s` | `0.00047 m` |

## 3. Wave 2 `dirH1_m60` 全量门禁对比 `dirG1_m20`

| 评测门禁 | `dirG1_m20` 基线 | `dirH1_m60` 实测 | 结论 |
|---|---:|---:|---|
| Gate 1 常规坡面 `±18%` (`v=0.65`, 3 seeds) | `81/81 (100%)` | **`81/81 (100%) PASS`** (`vx_mae=0.0531–0.0590`, `max_loop<=0.00010 m`) | 低速不跌，但 `vx_mae` 变松 |
| Gate 2 合法过渡 (`seed=71903`) | `24/24 (100%)` | **`24/24 (100%) PASS`** (`row_crest_2_to_5` `2/3 -> 3/3`, `max_loop=0.00024 m`) | **PASS** |
| Gate 3 270 s 走廊 (`seed=71903`) | `9/9 航点, resets=0` | `resets=0`, `max_closure=0.000136 m`, **`waypoints=6/9` (`completed=False`)** | 因低速偏慢未走完全程，**保留 `dirG1_m20` 为冻结主基线** |
| Gate 4 Canonical 障碍 (3 seeds) | 下台阶 `27/27` | 下台阶 **`27/27`** (`0` fail, `plat=2/27`, `max_loop<=0.00030 m`) | 持平 |
| `slopes_v100` (`±18%`, `1.0 m/s`, 3 seeds) | `—` | **`79/81 (97.5%)`** (`26/27 + 27/27 + 26/27`) | 良好 |
| `steep_v065` (`±30%`, 3 seeds, 135 eps) | `113/135 (83.7%)` | **`119/135 (88.1%)`** (`74/81 + 24/27 + 21/27`) | **+4.4% 提升** |
| `yaw080` (`seed=71903`) | `21/27 (77.8%)` | `21/27 (77.8%)` | 持平，仍为 P0 瓶颈 |
| `yaw080_sturn` (`seed=71903`) | `20/27 (74.1%)` | **`23/27 (85.2%)`** | **+11.1% 提升** |
| `obstacles_ext` (`seed=71903`, 24 eps) | 下台阶 `9/9`, `p40=2/3` | 下台阶 `9/9`, `p40=3/3` (未登台), `p60=2/3`, `max_loop=0.000091 m` | 抗跌倒改善 |

## 4. 根因定位与下一轮 (`dirH2`) 改动依据

1. **低速偏慢 + 高速超调**：`lin_vel_xy_sigma = 0.22`（`dirG1` 为 `0.09`）放宽了速度跟踪梯度，导致 `flat_v050` 降至 `0.43 m/s`（拖慢走廊 `6/9` 未走完），同时 `1.5 m/s` 指令实际冲至 `1.88–2.40 m/s` 引发平地与急停翻车。
2. **无向心包络的高速急转**：`2.0 m/s × ±1.5 rad/s` 同步叠加在 `45%` 陡坡上（`|vx * wz|` 达 `3.0 m/s²`）直接触发 `iter 80–107` 翻滚与 `iter 108` 数值异常。
3. **`up14/up18` 侧偏**：同一批受静态 DR 偏置影响的 env 在爬陡坡时产生 `-0.13～-0.35 rad/s` 偏航漂移，持续 `10–11 s` 后从侧缘滑出。

下一轮 `dirH2` 将保持 `35/78/6` 与 `400 Hz` 不变，收紧 `lin_vel_xy_sigma` 至 `0.11`，并在课程采样加入向心包络约束（`|vx * wz| <= 1.5`），从 `dirH1_m60` 续训验证。
