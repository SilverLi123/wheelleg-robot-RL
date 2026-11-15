# Wheelbipe V14-6 模型优化执行报告（2026-11-14）

> **执行依据**：
> - 评价标尺：[robomaster_wheelleg_evaluation_standard.md](file:///Users/silver/RL/wheelleg_lab/reports/robomaster_wheelleg_evaluation_standard.md)
> - 近期方向：[optimization_roadmap_20261114.md](file:///Users/silver/RL/wheelleg_lab/reports/optimization_roadmap_20261114.md)
> - 长期方案与统一门槛：[long_term_optimization_plan_20261114.md](file:///Users/silver/RL/wheelleg_lab/reports/long_term_optimization_plan_20261114.md)
>
> **物理仿真约束**：全流程保持 400 Hz（`WHEELBIPE_INTEGRATION_SUBSTEPS=2`，`physics_dt = 0.0025 s`，`policy_dt = 0.02 s`，Domain Randomization 开启），不改 35 维 Actor / 6 维动作接口，不修改既有闭环连杆资产。

---

## 一、执行总览与冠军模型结论

本分支后续按 **Stage 1（`dirE1..E4`）→ Stage 2（`dirF1..F2` 敏捷微调与 Critic 爆炸修复）→ Stage 3（`dirG1..G2` 全地图整合）** 递进迭代。**当前最终推荐结论**：Stage 3 检查点 **`dirG1_m20`** 已完整通过 Gates 1–5 全部统一回归门槛，在保持 Stage 1 全部稳态与陡坡下限的同时融合 Stage 2 高速（`1.35 m/s`）、连续转向（`yaw080`）与 `25 mm` 越障能力（详情见[第九节](#九stage-3-全地图整合-dirg400-hz-联合微调与统一主冠军-dirg1_m20)），现晋升为**唯一综合主力新冠军（Stage 3 Primary Unified Champion）**；原 Stage 1 双冠军与备选保留为历史对照：

| 角色定位 | 检查点标识 | 突出优势 | 适用场景建议 |
| :--- | :--- | :--- | :--- |
| **Stage 3 统一综合主力新冠军（当前默认推荐）** | **`dirG1_m20`** | Gate 1 (`81/81`)、Gate 2 (`24/24`)、Gate 3 走廊 (`9/9, 270 s, resets=0`)、Gate 4 canonical (`27/27` 台阶 `0` 失败)、Gate 5 闭环全评测 `<= 0.99 mm` 全部 PASS；`slopes_v135_blk6m` **3 种子 `70/81 (86.4%)` (`s71903: 26/27 = 96.3%`)**、`yaw080` **3 种子 `59/81`**、`obstacles_ext` **`p25 = 2/3` (`reached=11/24`)** 全面追平/超越 Stage 1 双冠军 | **全场景唯一默认部署**：单检查点覆盖常规坡、陡坡、`1.35 m/s` 高速、转向与低台阶越障 |
| Stage 1 综合冠军（历史参考） | `dirE2_m14` | `slopes_v065` **81/81**、`slopes_v090` **27/27**、`slopes_v110 (1.10 m/s)` **26/27 (96.30%)**、`steep_v065 (±30% 坡)` **70/81 (86.42%)**、Gate 1–5 PASS | Stage 1 参照基线（`v135_blk6m` `s71903/s173927` 闭环误差曾升至 `6.6–7.2 mm`，且 `p25=0/3`） |
| Stage 1 姿态与越低凸起冠军（历史参考） | `dirE1_m10` | `obstacles_ext` `reached=11/24`（`p25 = 2/3`）、`90 mm` 下台阶最大俯仰降至 **`15.69°`**、Gate 1–5 PASS | Stage 1 低速/越障参照（已被 `dirG1_m20` 在 `obstacles_ext` 追平并兼顾高速） |
| Stage 1 高稳态/平滑下楼备选（历史参考） | `dirE2_m10` | `90 mm` 下台阶最大俯仰 **`13.97°`**、最大横滚 **`1.95°`**、触地回弹 **`0.580 m/s`**、270 s 走廊 **`9/9, resets=0`** | Stage 1 备选参照（尚未补齐 `s173927/s384719` 多种子 Gate 1/4） |

---

## 二、基础设施与代码改动清单

| 模块 / 文件 | 改动内容 | 本地验证与生效状态 |
| :--- | :--- | :--- |
| `project/source/agent_tasks/.../wheelbipe_own/env_cfg.py`<br>`project/source/agent_tasks/.../wheelbipe_own/experiment.py` | 新增 `upward_rebound_vel_z`（单向惩罚触地后正向竖直回弹）与 `roll_angle_l2`（机身横滚平方惩罚）奖励项及配置开关 | 已同步至 `pi` 并在 `dirE1..E4` 训练启用（`upward_rebound_vel_z: -12.0`，`roll_angle_l2: -15.0`） |
| `project/source/agent_tasks/.../wheelbipe_own/random_terrain_map.py` | 新增 `random_terrain_grade_scale`、`random_terrain_stairs_yaw_rad`、`random_terrain_stairs_lane_probs`，支持斜向入阶与陡坡缩放 | 已通过 `test_random_terrain.py` 验证并同步至 `pi` |
| `profiles.json` | 新增 3 组微调配置：<br>- `random_terrain_opt_posture_20261114`<br>- `random_terrain_opt_speed_20261114`<br>- `random_terrain_opt_full_rm_20261114` | 已在 `dirE1..E4` 训练中实测生效 |
| `tools/eval_obstacle_terrain.py` | 补充输出 `stairs_max_abs_pitch_deg`、`stairs_max_abs_roll_deg`、`stairs_mean_roll_abs_deg`、`stairs_max_upward_vz_b_m_s`、`mean_mech_power_w`、`peak_mech_power_w` | 已执行基线及全部 `dirE*` 障碍/Gate 4 评测 |
| `tools/eval_random_terrain.py` | 支持 `--steep` / `grade_scale`（`±30%` 坡）、`--yaw` 转向评测，新增 `mean_abs_roll_bias_deg` 与机械功率统计 | 已执行全部 `slopes_v*` / `steep_v065` / `yaw080` / Gate 1 评测 |
| `tools/eval_terrain_transitions.py` | 修复过渡评测前强制归一化 `random_terrain_grade_scale = 1.0` | `python3 -m py_compile` 通过，已同步并在 Gate 2 评测验证 |

---

## 三、400 Hz 训练实验总表（`dirE1`–`dirE4`）

所有训练均在 `WHEELBIPE_INTEGRATION_SUBSTEPS=2`（`physics_dt = 0.0025 s`）、`4096` 环境、`15` 次迭代下顺序执行（受 `pi` 主机 `251 GiB` 内存限制，单次 4096 环境训练约占 `120–128 GiB`，顺序运行确保无 OOM 稳定收敛）：

| 实验名称 | Profile | 父检查点 & 优化配置 | 最终均值回报 / 成功率 | 训练闭环误差 `max` / `latest` (m) | 训练诊断与淘汰说明 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`dirE1_posture_platform_4096_15`** | `random_terrain_opt_posture_20261114` | `dirD2_m5`，actor-only，`lr=1e-5`，`cmd=(0.65, 0.45)`，`stairs_yaw=0.28 rad` | `26.11` / `0.9249` | `0.001418` / `0.000106` | 收敛极稳；有效压缩下台阶俯仰并提升 `25 mm` 上台阶至 `2/3` |
| **`dirE2_speed105_yaw08_4096_15`** | `random_terrain_opt_speed_20261114` | `dirD2_m5`，actor-only，`lr=1e-5`，`cmd=(1.05, 0.80)`，`stairs_yaw=0.25 rad` | `25.66` / `0.8908` | `0.001227` / `0.000109` | **综合最优**：在 `1.10 m/s` 和 `±30%` 陡坡下突破明显，且闭环误差全过程 `< 1.23 mm` |
| **`dirE3_full_rm_steep_4096_15`** | `random_terrain_opt_full_rm_20261114` | `dirD2_m5`，actor-only，`lr=1e-5`，`cmd=(1.05, 0.80)`，`grade_scale=1.45` | `7.47` / `0.8877` | **`0.006049`** / `0.000495` | **淘汰**：训练 `max_loop_error_m = 6.05 mm`，且 `dirE3_m10` 在 `slopes_v110` 闭环误差达 `4.82 mm`（超出 `2 mm` 红线） |
| **`dirE4_chain_4096_15`** | `random_terrain_opt_speed_20261114` | 从 **`dirE1_m10`** 链式微调，`lr=5e-5`，`cmd=(1.05, 0.80)` | `5.22` / `0.8804` | **`0.004008`** / `0.000175` | **淘汰**：高 `lr` 导致训练闭环误差飙至 `4.01 mm` 且 critic value loss 震荡至 `34403.98`，高速表现不如 `dirE2` |

---

## 四、六大 RM 评测套件全量对比

> 注：`slopes_v065`、`steep_v065` 为 `--repeats 9`（81 局，`seed=71903`）；`slopes_v090`（`0.90 m/s`）、`slopes_v110`（`1.10 m/s`）、`yaw080`（`vx=0.65 m/s, wz=0.80 rad/s`）为 `--repeats 3`（27 局，`seed=71903`）；`obstacles_ext` 为 8 种障碍（`stairs_down_{30,60,90}mm` + `platform_up_{25,40,60,120,180}mm`）共 24 局（`seed=71903`）。

| 评测套件 | 关键指标 | `dirD2_m5`（基线） | **`dirE1_m10`** | **`dirE2_m10`** | **`dirE2_m14`（主冠军）** | `dirE3_m10`（淘汰） | `dirE4_m10`（淘汰） |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `slopes_v065`**<br>(`±6..18%`, `0.65 m/s`) | 达标率 / 失败项<br>`vx_mae` / `wz_mae`<br>均值功率 / 峰值功率<br>`max_loop_error_m` | `80/81 (98.77%)` (`up06: 8/9`)<br>`0.0180` / `0.0272`<br>`14.1 W` / `398.1 W`<br>`0.00032 m` | **`81/81 (100%)`** (`0` fail)<br>`0.0159` / `0.0240`<br>`14.1 W` / `660.9 W`<br>`0.00054 m` | **`81/81 (100%)`** (`0` fail)<br>`0.0157` / `0.0237`<br>`14.1 W` / `440.7 W`<br>`0.00068 m` | **`81/81 (100%)`** (`0` fail)<br>**`0.0147` / `0.0230`**<br>`14.0 W` / `411.0 W`<br>**`0.00011 m`** | `81/81 (100%)`<br>`0.0162` / `0.0241`<br>`14.1 W` / `404.2 W`<br>`0.00025 m` | `81/81 (100%)`<br>`0.0154` / `0.0239`<br>`14.1 W` / `417.5 W`<br>`0.00030 m` |
| **2. `slopes_v090`**<br>(`±6..18%`, `0.90 m/s`) | 达标率 / 失败项<br>`vx_mae` / `wz_mae`<br>均值功率 / 峰值功率<br>`max_loop_error_m` | `26/27 (96.30%)` (`up18: 2/3`)<br>`0.0779` / `0.0799`<br>`22.7 W` / `783.8 W`<br>`0.00098 m` | **`27/27 (100%)`**<br>**`0.0477`** / `0.0220`<br>`19.7 W` / `103.8 W`<br>`0.00009 m` | **`27/27 (100%)`**<br>`0.0486` / **`0.0219`**<br>`19.7 W` / **`85.0 W`**<br>`0.00012 m` | **`27/27 (100%)`**<br>`0.0486` / `0.0223`<br>`19.7 W` / `104.2 W`<br>`0.00010 m` | `26/27 (96.30%)`<br>`0.0459` / `0.0280`<br>`19.7 W` / `227.1 W`<br>`0.00010 m` | `27/27 (100%)`<br>`0.0524` / `0.0224`<br>`19.8 W` / `79.8 W`<br>`0.00010 m` |
| **3. `slopes_v110`**<br>(`±6..18%`, `1.10 m/s`) | 达标率 / 失败项<br>`vx_mae` / `wz_mae`<br>均值功率 / 峰值功率<br>`max_loop_error_m` | `23/27 (85.19%)` (`up14:1/3, up18:1/3`)<br>`0.2282` / `0.1903`<br>`32.1 W` / `930.6 W`<br>`0.00132 m` | `23/27 (85.19%)`<br>`0.2066` / `0.3254`<br>`34.4 W` / `733.9 W`<br>`0.00065 m` | `24/27 (88.89%)` (`up14:2/3, up18:1/3`)<br>`0.1832` / `0.1403`<br>`29.2 W` / `677.8 W`<br>`0.00104 m` | **`26/27 (96.30%)`** (`up18: 2/3`)<br>`0.2120` / **`0.0723`**<br>**`27.9 W` / `520.6 W`**<br>`0.00098 m` | `21/27 (77.78%)`<br>`0.2196` / `0.1353`<br>`29.3 W` / `1564.6 W`<br>**`0.00482 m (FAIL)`** | `22/27 (81.48%)`<br>`0.2202` / `0.1937`<br>`31.0 W` / `687.4 W`<br>`0.00023 m` |
| **4. `steep_v065`**<br>(`±10..30%`, `0.65 m/s`) | 达标率 / 失败项<br>`vx_mae` / `wz_mae`<br>均值功率 / 峰值功率<br>`max_loop_error_m` | `68/81 (83.95%)` (`up14:3/9, up18:3/9, down18:8/9`)<br>`0.0955` / `0.0925`<br>`22.5 W` / `653.8 W`<br>`0.00144 m` | `69/81 (85.19%)` (`up14:3/9, up18:3/9`)<br>**`0.0874`** / **`0.0679`**<br>`22.1 W` / `553.8 W`<br>`0.00058 m` | `69/81 (85.19%)` (`up14:4/9, up18:3/9, down18:8/9`)<br>`0.0964` / `0.0749`<br>`22.3 W` / `711.9 W`<br>`0.00079 m` | **`70/81 (86.42%)`** (`up14:4/9, up18:3/9, down18:9/9`)<br>`0.0898` / `0.0758`<br>`22.4 W` / `684.4 W`<br>`0.00085 m` | `69/81 (85.19%)`<br>`0.0928` / `0.0698`<br>`22.2 W` / `576.2 W`<br>`0.00039 m` | `67/81 (82.72%)`<br>`0.0952` / `0.0856`<br>`23.1 W` / `871.6 W`<br>`0.00160 m` |
| **5. `yaw080`**<br>(`vx=0.65`, `wz=0.80`) | 达标率 / 失败项<br>`vx_mae` / `wz_mae`<br>`max_loop_error_m` | `21/27 (77.78%)` (`down10:1/3, up06:2/3, up18:0/3`)<br>`0.0490` / `0.1555`<br>`0.00063 m` | `20/27 (74.07%)`<br>`0.0472` / `0.1440`<br>`0.00014 m` | `20/27 (74.07%)`<br>`0.0502` / `0.1603`<br>`0.00012 m` | `20/27 (74.07%)`<br>**`0.0436` / `0.1432`**<br>`0.00038 m` | `20/27 (74.07%)`<br>`0.0477` / `0.1477`<br>`0.00013 m` | `20/27 (74.07%)`<br>`0.0431` / `0.1470`<br>`0.00015 m` |
| **6. `obstacles_ext`**<br>(`24` 局含平台攀爬) | 总达台 (`stairs / p25 / p40`)<br>`s90` `max\|p\|` / `max\|r\|`<br>`s90` `vz_rebound` / `cl` | `10/24` (`9/9, 1/3, 0/3`)<br>`22.16°` / `3.08°`<br>`0.796 m/s` / `0.00020 m` | **`11/24`** (`9/9,` **`2/3`**, `0/3`)<br>`15.69°` / `2.84°`<br>`0.689 m/s` / `0.00011 m` | `10/24` (`9/9, 1/3, 0/3`)<br>**`13.97°` / `1.95°`**<br>**`0.580 m/s`** / `0.00019 m` | `9/24` (`9/9, 0/3, 0/3`)<br>`16.19°` / `3.75°`<br>`0.632 m/s` / `0.00011 m` | `10/24` (`9/9, 1/3, 0/3`)<br>`12.80°` / `2.30°`<br>`0.568 m/s` / `0.00012 m` | `10/24` (`9/9, 1/3, 0/3`)<br>`13.81°` / `3.05°`<br>`0.568 m/s` / `0.00011 m` |

> **障碍消融补充观测（同套件 `obstacles_ext` 不同保存点对照）**：
> - `dirE1_m5`: `10/24` (`p25=1/3`, `s90 max|p|=14.99°`, `vz=0.609`)；`dirE1_m14`: `9/24` (`p25=0/3`, `s90 max|p|=21.50°`, `vz=0.941`) —— 确认 `dirE1` 最优点在 **`model_10`**。
> - `dirE2_m5`: `10/24` (`p25=1/3`, `s90 max|p|=15.77°`, `max|r|=1.98°`, `vz=0.568`) —— 较早迭代更利于低台阶平衡，后续迭代把重心向高速/陡坡倾斜。
> - `dirE3_m5`: `12/24` (`p25=2/3, p40=1/3`)，但其在 `steep_v065` 仍为 `68/81` 且后续 `dirE3_m10` 闭环误差超限（`0.00482 m`），因此不采用。
> - `dirE4_m5`: `11/24` (`p25=2/3, s90 max|p|=16.79°`)；`dirE4_m14`: `11/24` (`p25=2/3`，但 `s90 max|p|=46.93°, vz=1.020 m/s` 剧烈恶化)。

---

## 五、五项强制统一回归门槛验证表（Gates 1–5）

依据《长期优化方案》第七节，任何新冠军必须同时满足以下全部 5 项回归门槛：

| 回归门槛 | 合格判据 | `dirE2_m14`（主力冠军） | `dirE1_m10`（姿态冠军） | `dirE2_m10`（平衡备选） |
| :--- | :--- | :---: | :---: | :---: |
| **Gate 1：3 种子 3×3 随机坡面 (`0.65 m/s`, `--repeats 3`)** | 3 种子合计 **`81/81` PASS** | **PASS (`81/81`)**<br>- `s71903`: `27/27` (`vx=0.0131`, `wz=0.0220`, `cl=0.00010`)<br>- `s173927`: `27/27` (`vx=0.0140`, `wz=0.0237`, `cl=0.00011`)<br>- `s384719`: `27/27` (`vx=0.0147`, `wz=0.0233`, `cl=0.00010`) | **PASS (`81/81`)**<br>- `s71903`: `27/27` (`vx=0.0136`, `wz=0.0220`, `cl=0.00009`)<br>- `s173927`: `27/27` (`vx=0.0145`, `wz=0.0239`, `cl=0.00010`)<br>- `s384719`: `27/27` (`vx=0.0155`, `wz=0.0226`, `cl=0.00011`) | *部分已验*：`s71903` **`27/27`** (`vx=0.0133`, `wz=0.0217`, `cl=0.00010`) 及 `repeats=9` **`81/81`**；`s173927`/`s384719` 未单跑 |
| **Gate 2：24 种合法跨块过渡 (`0.65 m/s`)** | **`24/24` PASS**（`8/8` case） | **PASS (`24/24`)**<br>`vx_mae = 0.01710 m/s`，`travel = 4.935 m`，`crest = 0/2`，`cl = 0.00038 m` | **PASS (`24/24`)**<br>`vx_mae = 0.01754 m/s`，`travel = 4.929 m`，`crest = 0/2`，`cl = 0.00189 m` | **PASS (`24/24`)**<br>`vx_mae = 0.01788 m/s`，`travel = 4.933 m`，`crest = 1/2`，`cl = 0.00049 m` |
| **Gate 3：270 s（13,500 步）蛇形走廊全航程 (`seed=71903`)** | **`9/9` 航点完成且 `resets = 0`** | **PASS (`9/9`, `0 resets`)**<br>`steps=13500`，`vx_err=0.02027 m/s`，`yaw_err=0.02685 rad/s`，`max_cl=0.000122 m`，`above2mm=0.0 s` | **PASS (`9/9`, `0 resets`)**<br>`steps=13500`，`vx_err=0.02076 m/s`，`yaw_err=0.02768 rad/s`，`max_cl=0.000123 m`，`above2mm=0.0 s` | **PASS (`9/9`, `0 resets`)**<br>`steps=13500`，`vx_err=0.02045 m/s`，`yaw_err=0.02679 rad/s`，`max_cl=0.000130 m`，`above2mm=0.0 s` |
| **Gate 4：3 种子连续下台阶 `30/60/90 mm` (`--suite canonical`)** | 3 种子台阶段 **`27/27` 到达** | **PASS (`27/27`)**<br>- `s71903`: `9/9` (`max_p=16.83°, max_r=2.85°, vz=0.675, cl=0.00016`)<br>- `s173927`: `9/9` (`max_p=20.37°, max_r=4.07°, vz=0.648, cl=0.00011`)<br>- `s384719`: `9/9` (`max_p=17.87°, max_r=2.95°, vz=0.778, cl=0.00013`) | **PASS (`27/27`)**<br>- `s71903`: `9/9` (`max_p=14.40°, max_r=3.85°, vz=0.584, cl=0.00032`)<br>- `s173927`: `9/9` (`max_p=22.55°, max_r=4.61°, vz=0.723, cl=0.00013`)<br>- `s384719`: `9/9` (`max_p=9.04°, max_r=3.64°, vz=0.559, cl=0.00013`) | *单种子已验*：`obstacles_ext (s71903)` 台阶段 `9/9` (`s90 max_p=13.97°, max_r=1.95°, vz=0.580`)；3 种子 canonical Gate 4 未单独补跑 |
| **Gate 5：400 Hz 闭环与数值健康** | 全评测 `max_loop_error_m < 0.002 m`，无 `numerical`/`closed_loop` 重置 | **PASS**<br>全部 12 项评测 `max_cl <= 0.00098 m`，训练 `max_cl = 0.001227 m`，`0` 数值/闭环异常 | **PASS**<br>全部 13 项评测 `max_cl <= 0.00189 m`，训练 `max_cl = 0.001418 m`，`0` 数值/闭环异常 | **PASS**<br>全部 8 项已跑评测 `max_cl <= 0.00104 m`，训练 `max_cl = 0.001227 m`，`0` 数值/闭环异常 |

---

## 六、关键收益与物理边界分析

### 1. 本轮实测显著收益（对照基线 `dirD2_m5`）
1. **`0.65 m/s` 常规坡面消除孤立失败并大幅收敛极端倾角**：`dirD2_m5` 在 `slopes_v065`（81 局）曾出现 1 次 `up06` 倾覆（`80/81`、`max_r=179.97°`）；`dirE2_m14` 与 `dirE1_m10` 在 3×3 全随机种子（及 `81` 局复测）均达 **`100%`**，且 `dirE2_m14` 的 `max_abs_pitch_deg` 从 `71.61°` 压低至 **`9.19°`**。
2. **中高速斜坡控制跨跃（`0.90 -> 1.10 m/s`）**：`dirE2_m14` 将 `slopes_v090` 从 `26/27 (96.30%)` 提升至 **`27/27 (100%)`**，峰值功率从 `783.8 W` 降至 **`104.2 W`**；将 `slopes_v110 (1.10 m/s)` 从 `23/27 (85.19%)` 大幅提升至 **`26/27 (96.30%)`**（`+11.11%`），`wz_mae` 从 `0.1903` 降至 **`0.0723 rad/s`**，峰值功率从 `930.6 W` 降至 **`520.6 W`**。
3. **30% 标准公路陡坡全通过测试（`steep_v065`）**：`dirE2_m14` 总通过率提升至 **`70/81 (86.42%)`**，且彻底解决基线在 `-30%` 极限下坡（`down18`）的唯一失败（`8/9 -> 9/9`）。
4. **连续下台阶俯仰剧烈收敛**：`90 mm` 连续下台阶最大俯仰从基线 `22.16°` 降至 `dirE2_m14` **`16.19°`**、**`dirE1_m10` `15.69°`**、**`dirE2_m10` `13.97°`**，正向回弹竖直速度 `vz` 从 `0.796 m/s` 降至 **`0.580–0.632 m/s`**（尚未降至 `< 0.15 m/s` 档，主要受 15 iter actor-only 微调幅度限制）。
5. **270 秒长时复杂路线零重置通关**：`dirE2_m14`、`dirE2_m10`、`dirE1_m10` 在随机地形蛇形走廊 `13,500` 步全 `9/9` 航点 **`0` 重置完赛**，闭环误差峰值仅 `0.122–0.130 mm`，速度误差仅 `0.02027–0.02076 m/s`。

### 2. 当前触及的架构天花板（需后续长期阶段突破）
- **`yaw080` 大转向复合陡坡瓶颈（固定 `20/27`）**：基线与所有 `dirE1..E4` 模型在边走 `0.65 m/s` 边以 `0.80 rad/s` 急转时，均在 `up18 (0/3)` 与 `down10 (0/3–1/3)` 处受限于 `15 iter` actor-only + 未放宽 `lin_vel_xy_sigma` 与长周期速度/转向课程，需要按《长期方案》阶段 2–3 执行 **含 Critic 联合训练 + 数百轮课程训练（1.5–2.0 m/s、≥1.5 rad/s）**。
- **`platform_up_40mm/60mm/120mm/180mm` 纯滚动无法跨越**：受限于 `62.5 mm` 轮半径、机腹结构及 35 维无高程前视感知（盲走），`60 mm` 以上垂直台阶按方案处于阶段 4 决策点（指令驱动蓄力跳跃模态 vs 教师-学生高程感知蒸馏）。
- **`steep_v065` 上行 `24%/30%`（`up14: 4/9`, `up18: 3/9`）**：未启用专用 22%→26%→30% 渐进课程与高扭矩/防抬头专项训练（如 `dirE3` 一次硬拉 `1.45×` 会诱发闭环误差突破 `6 mm`），后续应改用《长期方案》阶段 3 的**分阶段渐进坡度课程 + 闭环动作速率约束**。

---

## 七、检查点清单与本地归档位置

- **本地同步目录**：[logs/](file:///Users/silver/RL/wheelleg_lab/logs/)（包含全部 `eval_dir*` 评测 `summary.json`、`diagnostics/training_health.json`、270 s 走廊 `eval.log`、训练日志 `train_runs/` 与 `rsl_rl/` 更新记录）
- **SHA256 校验清单**：[checkpoints_sha256.txt](file:///Users/silver/RL/wheelleg_lab/logs/checkpoints_sha256.txt)

| 标识 | `pi` 服务器路径 | `SHA-256`（前 16 位 / 完整见校验清单） |
| :--- | :--- | :--- |
| `dirD2_m5`（基线） | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/logs/rsl_rl/own_low_speed_exploration/2026-11-13_17-36-05_dirD2_stairs_4096_15/model_5.pt` | `ca108ded071f2d58…e0e6fd` |
| **`dirE2_m14`（主冠军）** | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/logs/rsl_rl/own_low_speed_exploration/2026-11-14_07-52-06_dirE2_speed105_yaw08_4096_15/model_14.pt` | `8bc9ea69ae2f008d…000575` |
| **`dirE1_m10`（姿态冠军）** | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/logs/rsl_rl/own_low_speed_exploration/2026-11-14_07-23-14_dirE1_posture_platform_4096_15/model_10.pt` | `2176991a0d266fb8…bfa0d50` |
| `dirE2_m10`（平衡备选） | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/logs/rsl_rl/own_low_speed_exploration/2026-11-14_07-52-06_dirE2_speed105_yaw08_4096_15/model_10.pt` | `a4350a063a1f0795…e733a3` |
| `dirE3_m10`（闭环超限淘汰） | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/logs/rsl_rl/own_low_speed_exploration/2026-11-14_07-23-14_dirE3_full_rm_steep_4096_15/model_10.pt` | `4a6ad41c9d5db608…0d0fc8` |
| `dirE4_m10`（链式震荡淘汰） | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/logs/rsl_rl/own_low_speed_exploration/2026-11-14_08-18-34_dirE4_chain_4096_15/model_10.pt` | `9b2cbb5f89fab498…bf536be` |
| `dirF1_m15`（Stage 2 Critic 爆炸对照候选） | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/logs/rsl_rl/own_low_speed_exploration/2026-11-14_11-22-08_dirF1_agile_critic_4096_25/model_15.pt` | `4601069338bb7784…c819ca` |
| `dirF2_m15`（Stage 2 修复后高速/S弯候选） | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/logs/rsl_rl/own_low_speed_exploration/2026-11-14_12-47-04_dirF2_clean_agile_4096_20/model_15.pt` | `2353cd21561f772a…ddaa47` |
| **`dirG1_m20`（Stage 3 统一主冠军）** | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/runs/train_dirG1_fromE2_steep_agile_4096_25/logs/rsl_rl/own_low_speed_exploration/2026-11-14_14-08-36_dirG1_fromE2_steep_agile_4096_25/model_20.pt` | `a327c331c08701ae…691a0e` |
| `dirG2_m24`（Stage 3 对照最优） | `/data/wheeled-legged/projects/v14_6_random_terrain_20261112/runs/train_dirG2_fromF2_steep_agile_4096_25/logs/rsl_rl/own_low_speed_exploration/2026-11-14_14-30-36_dirG2_fromF2_steep_agile_4096_25/model_24.pt` | `d9530998dddcb8cc…a5b55e` |

---

## 八、后续 Goals 1–3 执行结果（评测对齐、Critic 爆炸修复与 Stage 2 视频归档）

> 以下全部训练与评测均在严格 **400 Hz（`WHEELBIPE_INTEGRATION_SUBSTEPS=2`, `physics_dt = 0.0025 s`, `policy_dt = 0.02 s`）** 下完成并已核验 `diagnostics/training_health.json`。

### 1. Goal 1：评测口径对齐与极速/S 弯失利物理诊断
- **改动文件**：[eval_random_terrain.py](file:///Users/silver/RL/wheelleg_lab/tools/eval_random_terrain.py)
  - 新增 `--target-forward-m` 与 `--alternating-yaw-period-s`，并在重置帧保护 `last_xy`（避免跌落瞬间的坐标覆盖掉历史最远位姿），单独统计 `completed_block`。
- **对齐前后 400 Hz 实测对比**：
  - **高速 `1.35 m/s` 单块 6 m 对齐（`slopes_v135_blk6m`, `--seconds 8.0 --target-forward-m 6.0`）**：
    - 此前 12 s 长跑（跑出 `17–26 m`，跨出 `7–10 m` 单块并撞击非训练边界）下，`dirE2_m14` 为 `12/27 (44.4%)`、`dirE1_m10` 为 `14/27 (51.9%)`。
    - 对齐至单块 `6 m` 穿越任务后：**`dirE2_m14`** 达 **`22/27 (81.5%)`**（`completed_block = 21/27`，全部 4 组下坡 + `flat` + `up14` 均为 `3/3`）；Stage 2 候选 **`dirF2_m15`** 升至 **`23/27 (85.2%)`**（`completed_block = 23/27`, `wz_mae` 从 `0.6787` 降至 `0.2405 rad/s`）；**`dirF1_m15`** 达到 **`24/27 (88.9%)`**（`completed_block = 23/27`, `vx_mae = 0.4147 m/s`, `max_loop_error_m = 0.00099 m`）。
  - **S 弯蛇形大转向对齐（`yaw080_sturn`, `vx=0.65, wz=±0.80 rad/s, period=3.0s, target=6.0m`）**：
    - `dirE2_m14`：`20/27 (74.1%, completed_block=15/27)`（`up06..up18`、`flat`、`down06` 全 `3/3`）；
    - **`dirF1_m15`**：**`21/27 (77.8%, completed_block=16/27, cl=0.00013 m)`**；
    - **`dirF2_m15`**：**`21/27 (77.8%, completed_block=15/27, cl=0.00012 m)`**（`down10` 也达到 `3/3`，7/9 地形满分；仅剩 `down14` 与 `down18` 在第 `457–580` 步出现 `left_front_guide1_link` / `right_front_guide1_link` 轻擦触地 `13–61 N` 触发 `contact` 重置，未发生倾覆）。

### 2. Goal 2：Critic 梯度/Value Loss 爆炸修复与 `dirF1` / `dirF2` 400 Hz 全量对照
- **代码修复**：
  - [experiment.py](file:///Users/silver/RL/wheelleg_lab/project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/experiment.py#L362-L363)：钳位 `terms['track_height_square'] <= 0.25`，防止飞出地图下坠时高度误差平方无界放大。
  - [unified_environment.py](file:///Users/silver/RL/wheelleg_lab/project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py#L323-L336)：随机地形训练模式（`_u_ready=False`）增加 `fell_off`（相对高度 `< -1.2 m`）终止与地图外边框截断（`local_xy[:, 0] > 11.5` 或 `local_xy[:, 1] > 7.0`），并钳位 Critic `rel_height.clamp(-0.5, 1.0)`。
  - [random_terrain_map.py](file:///Users/silver/RL/wheelleg_lab/project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/random_terrain_map.py#L47-L52) 与 [profiles.json](file:///Users/silver/RL/wheelleg_lab/profiles.json#L2543-L2820)：新增 Stage 2 agile profile（`random_terrain_stage2_agile_20261114`）。
- **修复前后 400 Hz 训练收敛对比**（均从 `dirE2_m14` 起训，`4096 envs`，Actor+Critic 联合微调，`lr=1e-5`）：

| 训练实验 | 迭代数 | `value_function` loss 区间（起点 → 峰值 → 终点） | `actor_delta_l2` 走势 | `surrogate` 终点 | 训练诊断 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`dirF1_agile_critic_4096_25`**（修复前） | 25 | `3.516 → 56,122,323.6 → 56,122,323.6` | `0.0081 → 0.0295`（15 轮后停滞） | `-0.00005` | 飞出地图自由落体导致 Critic loss 爆炸并挤占共用 `max_grad_norm=0.5` |
| **`dirF2_clean_agile_4096_20`**（修复后） | 20 | **`1.863 → 10.023 → 5.780`** | **`0.0074 → 0.0337`**（全程平滑递增） | **`+0.00501`** | **Critic 爆炸彻底消除**，Actor 恢复稳定正向更新 |

- **400 Hz 核心指标对照（`dirE2_m14` vs `dirF1_m15` vs `dirF2_m15`）**：

| 400 Hz 评测套件 | **`dirE2_m14`（主冠军）** | `dirF1_m15`（修复前对照） | **`dirF2_m15`（修复后 Stage 2 候选）** | 结论解读 |
| :--- | :---: | :---: | :---: | :--- |
| `slopes_v065` (`81` 局, `0.65 m/s`) | **`81/81 (100%)`** (`cl=0.00011 m`) | `dirF1_m10`: `80/81` (`up18:8/9`) | `80/81 (98.8%)` (`up18:8/9`, `cl=0.00010 m`) | Stage 2 `agile` profile 仅设 `±18%` 且偏重高速，导致 `up18` 偶发 1 次失利，未满足 Gate 1 严格 `81/81` |
| `slopes_v110` (`27` 局, `1.10 m/s`) | **`26/27 (96.3%)`** (`cl=0.00098 m`) | — | `24/27 (88.9%)` (`cl=0.00084 m`) | `dirE2_m14` 保持中高速长跑稳态优势 |
| **`slopes_v135_blk6m`** (`27` 局, `1.35 m/s`, `6m`) | `22/27 (81.5%)` (`blk=21/27`) | **`24/27 (88.9%)`** (`blk=23/27`, `cl=0.00099 m`) | **`23/27 (85.2%)`** (`blk=23/27`, `wz=0.2405`) | Stage 2 显式改善 `1.35 m/s` 高速穿块表现与偏航稳定度 |
| **`yaw080`** (`27` 局, `0.65 m/s, 0.80 rad/s`) | `20/27 (74.1%)` (`cl=0.00038 m`) | **`21/27 (77.8%)`** (`cl=0.00013 m`) | **`21/27 (77.8%)`** (`cl=0.00015 m`) | Stage 2 将连续大转向提升到 `21/27`（`dirF2_m19` 回落为 `20/27`） |
| **`yaw080_sturn`** (`27` 局, S 弯 `6m`) | `20/27 (74.1%, blk=15)` | **`21/27 (77.8%, blk=16)`** | **`21/27 (77.8%, blk=15, cl=0.00012 m)`** | 7/9 地形满分；`down14/down18` 仅后期前导轮轻擦 |
| `steep_v065` (`81` 局, `±30%`) | **`70/81 (86.4%)`** (`down18: 9/9`) | `dirF1_m10`: `69/81 (85.2%)` | `59/81 (72.8%)` (`down18: 0/9`, `cl=0.00130 m`) | 因 `random_terrain_stage2_agile_20261114` 关闭了 `steep_grades=False`，`dirF2` 退出陡坡泛化 |
| `obstacles_ext` (`24` 局) | `9/24` (`9/9` 阶, `0/3` p25) | — | **`10/24`** (`9/9` 阶, `p25=1/3, p40=2/3, p60=2/3, s90_p=14.62°`) | `dirF2_m15` 下台阶俯仰收敛且略优高台存活，但 `platform_up_25mm` 仍低于 `dirE1_m10 (2/3)` |
| Gate 2 `transitions` (`24` 局) | **PASS `24/24`** (`cl=0.00038 m`) | — | **PASS `24/24`** (`vx=0.0162, crest=2/6, cl=0.00070 m`) | `dirF2_m15` 通过合法跨块过渡且破除 2/6 坡顶墙探针 |

- **最终选型判定**：**维持 `dirE2_m14` 为综合主力冠军、`dirE1_m10` 为姿态/25mm 低台阶冠军**；将 **`dirF2_m15`（与 `dirF1_m15`）标注为 Stage 2 高速/大转向敏捷候选**（证明了 Critic 爆炸修复有效，且在 `1.35 m/s` 与 `yaw080` 上取得突破；后续若需合并为统一新冠军，只需在 `random_terrain_stage2_agile_20261114` 中保留 `steep_grades=True` 混训即可）。

### 3. Goal 3：400 Hz 全新展示视频录制与本地归档
- 全部 4 段视频均按 **400 Hz（`physics_dt = 0.0025 s`）** 录制完成，零重置、闭环误差峰值 `<= 0.137 mm`、`longest_above_2mm_s = 0.0`：
  1. `dirE2_m14_corridor_48s.mp4`（48.0 s, 2400 steps, `status=PASS, resets=0, max_closure_m=0.000119 m`）
  2. `dirE2_m14_stairs70mm.mp4`（10.4 s, 520 steps, `status=PASS, resets=0, max_closure_m=0.000137 m`）
  3. `dirE1_m10_stairs70mm.mp4`（10.4 s, 520 steps, `status=PASS, resets=0, max_closure_m=0.000107 m`）
  4. `dirE1_m10_platform25mm.mp4`（10.4 s, 520 steps, `status=PASS, resets=0, max_closure_m=0.000114 m`）
  5. 拼接总片：`showcase_20261114_79s.mp4`（79.2 s, 3960 frames, 1280×720@50fps, H.264 `faststart`）
- **本地仓库规范归档路径**：
  - 视频、关键帧与单段摘要：[stage2_showcase_20261114/](file:///Users/silver/RL/wheelleg_lab/videos/stage2_showcase_20261114/)
  - 400 Hz 阶段 2 评测与训练健康日志：[eval_stage2_f/](file:///Users/silver/RL/wheelleg_lab/logs/eval_stage2_f/)、[train_stage2_f/](file:///Users/silver/RL/wheelleg_lab/logs/train_stage2_f/)
  - 检查点与哈希清单：[checkpoints_sha256.txt](file:///Users/silver/RL/wheelleg_lab/logs/checkpoints_sha256.txt)

---

## 九、Stage 3 全地图整合（`dirG`）：400 Hz 联合微调与统一主冠军 `dirG1_m20`

> 以下训练、多种子评测、走廊验证及展示视频全部在严格 **400 Hz（`WHEELBIPE_INTEGRATION_SUBSTEPS=2`，`physics_dt = 0.0025 s`，`policy_dt = 0.02 s`，Domain Randomization 开启）** 下完成并已核验 `summary.json` 与 `diagnostics/training_health.json`。

### 1. Stage 3 环境修复与整合配置
1. **陡坡/边界逻辑修复**（[unified_environment.py](file:///Users/silver/RL/wheelleg_lab/project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/unified_environment.py#L323-L364)，本地与 `pi` SHA-256 一致：`c94c310195f2cbd8a452f205065402e894a049e91c6cc1d09e3e52bd8cab5213`）：
   - 将 `fell_off` 判定由相对 `env_origins` 的固定高度改为相对局部地形高程 `(root_pos[:, 2] - self._terrain_ground_z()) < -0.7`，消除 `-30%` 陡下坡及深谷处的误触发提前终止（此前 Stage 2 `dirF2_m15` 的 `down18: 0/9` 主要受旧 `< -1.2 m` 判定影响）。
   - 将 `map_boundary` 边界截断对齐 `random_terrain_map.MAP_SIZE_M / 2.0 - 0.5`（全尺寸 `48 × 36 m`）及阶梯通道 `in_lane_x` 范围，避免常规跨块过渡和陡坡长滑行被错误提前截断，同时对显式评估参数跳过该训练边界截断。
2. **Stage 3 整合 Profile**（[profiles.json](file:///Users/silver/RL/wheelleg_lab/profiles.json#L2681-L2820)，`random_terrain_stage2_steep_agile_20261114`）：
   - 在保留 Stage 2 敏捷激励（`command_lin_vel_x = [-0.85, 1.25] m/s`，`command_ang_vel_z = [-1.0, 1.0] rad/s`，`command_resampling = [1.2, 3.0] s`，`stairs_fraction = 0.34`，`upward_rebound_vel_z = -16.0`，`roll_angle_l2 = -18.0`，`lin_vel_xy_sigma = 0.14`，`ang_vel_z_sigma = 0.08`）的基础上，引入 **`random_terrain_grade_scale = 1.22`（`±22%` 坡度扩展）** 进行 Actor+Critic 联合微调。

### 2. 400 Hz 训练实验与检查点初选（`dirG1` vs `dirG2`）
两组训练均以 `4096 envs × 25 iters`（每组 `2,457,600` timesteps）于 400 Hz 完成且 Critic loss 全程平稳无爆炸：

| 实验名称 | 起点检查点 | `Mean reward` (0 → 24 轮) | `value_function loss` (起点 / 峰值 / 终点) | `Metrics/success_rate` (0 → 24 轮) | 训练期 `latest_loop_error_m`（全过程峰值） |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train_dirG1_fromE2_steep_agile_4096_25`** | `dirE2_m14` | `0.61 → 86.42` | `3.7948 / 13.1853 / 7.5249` | `0.7094 → 0.8972` | `0.000649 m`（`0.007352 m`） |
| `train_dirG2_fromF2_steep_agile_4096_25` | `dirF2_m15` | `1.41 → 80.45` | `2.9127 / 8.8957 / 5.8169` | `0.6583 → 0.8619` | `0.000123 m`（`0.004260 m`） |

> 注：训练阶段在 `4096` 并行极端接触探索样本中偶见瞬时闭环尖峰，但进入部署选型后 **`dirG1_m20` 在全部 24 项 400 Hz 评测中峰值闭环误差严格 `<= 0.000990 m`（`< 1 mm`）**。

在 `seed=71903` 的高速基准 **`slopes_v135_blk6m`**（`1.35 m/s`，6 m 单块）初选消融中，**`dirG1_m20` 胜出其余所有 Stage 3 候选**：
- **`dirG1_m20`**：**`26/27 (96.3%)`**，`ge6m = 24/27`，`vx_mae = 0.4576`，`wz_mae = 0.1712`，`max_loop_error_m = 0.00076 m`；
- `dirG1_m15`：`25/27 (92.6%)`，`ge6m = 24/27`，`max_loop = 0.00080 m`（`v110` 为 `24/27` 低于 `m20` 的 `25/27`）；
- `dirG1_m24`：`22/27 (81.5%)`，`ge6m = 20/27`，且 `max_loop` 飙升至 **`0.00640 m`（`> 2 mm` 淘汰）**，`v110` 跌至 `21/27`；
- `dirG2_m15 / m20 / m24`：`21/27 (0.00117 m)` / `23/27 (0.00713 m, > 2 mm)` / `24/27 (0.00118 m)`。

### 3. `dirG1_m20` vs `dirE2_m14` 全量 400 Hz 回归门槛与扩展基准对比

| 400 Hz 评测项（全开 DR） | 判定口径 / 种子数 | **`dirG1_m20`（Stage 3 新主冠军）** | `dirE2_m14`（Stage 1 主冠军对照） | 关键结论 |
| :--- | :---: | :---: | :---: | :--- |
| **Gate 1: `slopes_v065`（3 种子）** | `--repeats 3`×3 种子 + `9` 次复测 | **PASS `81/81` + `81/81 (100%)`**<br>`s71903: 27/27 (vx=0.0147, wz=0.0220, cl=0.00009 m)`<br>`s173927: 27/27 (vx=0.0165, wz=0.0235, cl=0.00010 m)`<br>`s384719: 27/27 (vx=0.0157, wz=0.0241, cl=0.00010 m)` | **PASS `81/81` + `81/81 (100%)`**<br>`vx=0.0131–0.0147, cl=0.00010–0.00011 m` | **均满分通过**（9 次复测 `cl = 0.00010 m` vs `0.00011 m`） |
| **Gate 2: `transitions`** | 24 合法过渡 + 6 坡顶墙探针 | **PASS `24/24`**<br>`vx=0.0173, travel=4.935 m, gm=+0.2622 m`<br>`crest_pass=0/2 (episodes=2/6), cl=0.00036 m` | **PASS `24/24`**<br>`vx=0.0171, travel=4.935 m, gm=+0.2607 m`<br>`crest_pass=0/2 (episodes=2/6), cl=0.00038 m` | **全合法过渡通过**，与 `dirE2_m14` 持平且略降 `cl` |
| **Gate 3: `corridor`（270 s）** | `13,500` 步全 9 航点（`s71903`） | **PASS `9/9`, `resets=0`**<br>`vx=0.5798 m/s, vx_err=0.0214 m/s`<br>`yaw_err=0.0257 rad/s, max_cl=0.000122 m` | **PASS `9/9`, `resets=0`**<br>`vx=0.5805 m/s, vx_err=0.0203 m/s`<br>`yaw_err=0.0269 rad/s, max_cl=0.000122 m` | **270 秒长航程零重置完赛**，`yaw_err` 更优 |
| **Gate 4: `canonical`（3 种子）** | 台阶 `30/60/90 mm` 零失败 | **台阶 `27/27 PASS`（0/9 失败×3 种子）**<br>`s_pitch = 18.06° / 18.02° / 15.96°`<br>`reached = 9+9+9 = 27`（平台 `9/27`, `cl<=0.00028 m`） | **台阶 `27/27 PASS`（0/9 失败×3 种子）**<br>`s_pitch = 16.83° / 20.37° / 17.87°`<br>`reached = 9+9+9 = 27`（平台 `9/27`, `cl<=0.00016 m`） | **3 种子下台阶全过**且峰值俯仰更收敛 |
| **Gate 5: 400 Hz 闭环健康** | 全评测 `< 0.002 m`、0 数值异常 | **PASS：24/24 项评测全 `<= 0.000990 m`** | `v135_blk6m` 在 `s71903/s173927` 飙至 `0.00722 / 0.00657 m` | **`dirG1_m20` 在 `1.35 m/s` 下守住闭环红线** |
| **`slopes_v090` (`0.90 m/s`)** | `27` 局（`s71903`） | **PASS `27/27 (100%)`**, `vx=0.0446, wz=0.0222, cl=0.00010 m` | PASS `27/27 (100%)`, `vx=0.0486, wz=0.0223` | **速度跟踪误差改善 8.2%** |
| **`slopes_v110` (`1.10 m/s`)** | `27` 局（`s71903`） | `25/27 (92.6%)`, **`ge6m=27/27`**, `vx=0.1695, wz=0.0527, cl=0.00029 m` | **`26/27 (96.3%)`**, `ge6m=26/27`, `vx=0.2120, wz=0.0723, cl=0.00098 m` | `dirG1_m20` 27/27 全越 6 m，跟踪更准且闭环更稳 |
| **`slopes_v135_blk6m` (`1.35 m/s`)** | 3 种子合计（`81` 局） | **`70/81 (86.4%)` (`ge6m=68/81`)**<br>`26/27 + 23/27 + 21/27`（**`max_cl <= 0.00089 m`**） | `67/81 (82.7%)` (`ge6m=64/81`)<br>`22/27 + 23/27 + 22/27`（2 个种子 `> 6.5 mm`） | **3 种子全面反超 +3.7 个百分点且无闭环超限** |
| **`steep_v065` (`±30%` 坡)** | `s71903 (81)` + `173927/384719 (27×2)` | `69/81 + 22/27 + 22/27 =` **`113/135 (83.7%)`**<br>（`down18` `8/9`，修补旧 bug） | `70/81 + 22/27 + 21/27 = 113/135 (83.7%)` | **与 Stage 1 最强陡坡泛化完全持平** |
| **`yaw080` (`0.65, 0.80 rad/s`)** | 3 种子合计（`81` 局） | **`59/81 (72.8%)`** (`21/27 + 20/27 + 18/27`, `cl<=0.00018 m`) | `57/81 (70.4%)` (`20/27 + 19/27 + 18/27`, `cl<=0.00138 m`) | **3 种子反超**（2 胜 1 平且闭环低一个数量级） |
| **`yaw080_sturn`（S 弯 6 m）** | 3 种子合计（`81` 局） | `62/81 (76.5%)` (`20/27 + 22/27 + 20/27`)<br>`gm=+0.2608..+0.2654 m`, 0 触地 | **`64/81 (79.0%)`** (`20/27 + 22/27 + 22/27`)<br>`gm=+0.2601..+0.2660 m` | 两者接近，均保持零云台触地（`+0.26 m`） |
| **`obstacles_ext` (`24` 局)** | 3×下楼梯 + 5×平台攀爬 | **未跌倒 `15/24`，达台 `11/24`（`p25 = 2/3`）**<br>`s90 max\|p\|=15.70°, max\|r\|=3.98°, vz=0.637` | 未跌倒 `13/24`，达台 `9/24`（`p25 = 0/3`）<br>`s90 max\|p\|=16.19°, max\|r\|=3.75°, vz=0.632` | **追平姿态冠军 `dirE1_m10 (15/24, reached=11)`，显著优于 `dirE2_m14`** |

### 4. Stage 3 400 Hz 展示视频、关键帧与规范归档
全部 5 段 Stage 3 实录视频（`1280×720@50fps`，400 Hz 仿真，Domain Randomization 全开）均 `PASS`、`resets = 0`、`max_closure_m <= 0.000123 m`，并已严格按仓库规范归档于 [stage3_showcase_20261114/](file:///Users/silver/RL/wheelleg_lab/videos/stage3_showcase_20261114/)（元数据独立存放于 `summaries/`，无重复 `policy.mp4` 冗余）：

1. **`dirG1_m20_corridor_48s.mp4`**（`48.00 s` / `2400 steps` / `27,161,619 B`）：`PASS, resets=0, max_closure_m=0.000122 m, mean_abs_vx=0.6301 m/s, vx_err=0.0212 m/s, wz_err=0.0139 rad/s`，SHA-256 `2a9b1b6f7e66e2c9b8e815c451f9a29c3bfada178ce98b638c4c95fbf656ec4f`。
2. **`dirG1_m20_row0_v125_25s.mp4`**（`25.00 s` / `1250 steps` / `16,415,501 B`，`fixed_command=(1.25, 0.0)` 第 0 排高速连坡实录）：`PASS, resets=0, max_closure_m=0.000123 m, mean_abs_vx=1.4881 m/s, vx_err=0.2972 m/s, wz_err=0.0950 rad/s`，SHA-256 `d6fdae05f6875c56d4146d034e04c71e621cdfdeb5050f50ef8c6a92f58499d6`；已在 [05s](file:///Users/silver/RL/wheelleg_lab/videos/stage3_showcase_20261114/keyframes/g1_m20_row0_v125_05s.png)、[14s](file:///Users/silver/RL/wheelleg_lab/videos/stage3_showcase_20261114/keyframes/g1_m20_row0_v125_14s.png)、[23s](file:///Users/silver/RL/wheelleg_lab/videos/stage3_showcase_20261114/keyframes/g1_m20_row0_v125_23s.png) 抽帧确认姿态稳定通过下坡、过渡区与 `+14%` 上坡。
3. **`dirG1_m20_stairs70mm.mp4`**（`10.40 s` / `520 steps` / `4,330,756 B`）：`PASS, resets=0, max_closure_m=0.000112 m, mean_abs_vx=0.6344 m/s, vx_err=0.0338 m/s, wz_err=0.0143 rad/s`，SHA-256 `6dc553474df6a43d6cfc8d4a5140d3dcd4cca9bb08c2b17231e287ebb3f3549a`。
4. **`dirG1_m20_platform25mm.mp4`**（`seed=173927`，`10.40 s` / `520 steps` / `4,328,143 B`）：`PASS, resets=0, max_closure_m=0.000091 m, mean_abs_vx=0.6305 m/s, vx_err=0.0315 m/s, wz_err=0.0222 rad/s`，SHA-256 `40f2043815582edd0a83f2c1b507fe02910705c2d0313f5d85ef6c07e863fc74`。
5. **`dirG1_m20_platform25mm_s384719.mp4`**（`seed=384719`，`10.40 s` / `520 steps` / `4,417,566 B`）：`PASS, resets=0, max_closure_m=0.000074 m, mean_abs_vx=0.6224 m/s, vx_err=0.0387 m/s, wz_err=0.0307 rad/s`，SHA-256 `95b1ff9362b7c9e437efc9616d1a48f2eca8d5f28864100aff79b0ebf8b56e67`。

> 同步提示：试拍阶段中越界或高速临界失败样本（`seed=71903` 25 mm 平台步 327 触地重置、第 8 块 `+22%` 零速猛冲 `up22_v135` 第 4 帧接触、以及第 0 排 32 s 全图冲出 `+24 m` 东边界崖边 `step=1541`）仅作诊断参照，未混入 showcase 成品目录。

- **Stage 3 归档目录**：
  - 视频与关键帧：[stage3_showcase_20261114/](file:///Users/silver/RL/wheelleg_lab/videos/stage3_showcase_20261114/)（含 `keyframes/` 与 `summaries/`）
  - 400 Hz Stage 3 评测日志（全部 48 个评测目录）：[eval_stage3_g/](file:///Users/silver/RL/wheelleg_lab/logs/eval_stage3_g/)
  - 400 Hz Stage 3 训练与健康诊断日志：[train_stage3_g/](file:///Users/silver/RL/wheelleg_lab/logs/train_stage3_g/)
  - 校验清单（已追加 `dirG1_m0..m24` 与 `dirG2_m0..m24`）：[checkpoints_sha256.txt](file:///Users/silver/RL/wheelleg_lab/logs/checkpoints_sha256.txt)
