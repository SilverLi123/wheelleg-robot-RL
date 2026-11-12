# Wheelbipe V14-6 随机混合地形核查

日期：2026-11-12

本报告只使用个人服务器目录：

```text
/data/wheeled-legged/projects/v14_6_random_terrain_20261112
```

没有修改队友的训练目录、共享环境或依赖。

## 地图结构

当前地图为 3 行 × 3 列的连续地图，每块 16 m × 12 m。

```text
down10  up10    up14
up06    down14  flat
up18    down06  down18
```

每块沿前进方向包含 2 m 平地、6 m 坡面、2 m 坡顶或坡底、6 m 回坡和 2 m 平地。地图固定种子，每次生成相同排布。训练时可随机选择地块、地图内位置和初始朝向；评估时固定在地块前端的零高度起步点。

## 已确认的问题

### 1. 原地图横向没有接平

坡面的高度原先只随地图 `x` 坐标变化，没有随地块 `y` 边缘衰减。相邻行中，一个地块的坡顶或坡底会直接接触另一地块的零高度边缘。导出的实际碰撞网格测得最大高差为 1.44 m。

旧九块路线在约第 4,556 个策略步发生 roll、pitch 和 spring working range 同时终止。旧录像进程最终在第 4,963 步由手动中断结束，`summary.json` 为 `ERROR`，并记录 1 次 reset。

证据：

```text
runs/route_all9_baseline_20261112_085918/summary.json
runs/route_all9_baseline_20261112_085918/rollout.jsonl
```

修复方法是在每块最后 1.5 m 的 `y` 边缘把完整坡高平滑衰减到零。解析高度查询和三角网格共用同一个衰减函数。

修复后的纯几何结果：

```text
原解析行边界 high/low 高差: 1.440000 m
新解析行边界跨边界高差:   0.000000960 m
实际导出网格行边界顶点数: 108
实际导出网格行边界 max |z|: 0 m
```

测试：

```bash
python3 tools/test_random_terrain.py -v
```

结果：

```text
random terrain geometry contract: PASS
```

### 2. 外边框坐标错位

随机地图回调已经把地图居中，Isaac Lab 又执行一次居中变换；原 `border_width=2.0` 还会在每个地图副本内侧插入零高度平条，与本任务使用的完整地图坐标不一致。

当前地图使用 `border_width=0.0`。地图四条外边本身已经回到零高度，因此重复地图副本也不会形成新的边界台阶。

### 3. 高度奖励曾使用世界海拔

继承的 V14-6 策略使用绝对根节点高度。直接把这个值作为 `track_height` 奖励参考时，机器人在同一坡面正常前进，也会因为世界坐标 `z` 升高而被当成腿长错误。

修复只改变奖励参考：

```text
奖励参考高度 = 根节点绝对高度 - 当前地块地面世界高度
```

Observation 仍是原来的绝对高度信号，Action、物理和 checkpoint 输入契约不变。四个随机地形 profile 均显式启用 `terrain_relative_height_reward`。

真实 Isaac Sim 5 步运行预检结果：

```text
status: PASS
max_reference_error_m: 0.0
observed_height_m: 0.268146961917
ground_height_world_m: 0.0
reward_reference_m: 0.268146961917
```

证据：

```text
runs/height_reward_runtime_smoke_20261112_103000/summary.json
```

### 4. 旧课程读取了清零后的回合时长

旧 `_custom_reset_random` 在 `DirectRLEnv` 清除 `episode_length_buf` 后读取回合时长，因此“2 秒内失败则降级”的判断无法得到真实回合年龄。旧逻辑还只检查二维位移是否超过 2 m，所以后退和横移也可能升级。

修复现在在复位前保存：

- 本回合真实时长；
- 回合开始位置；
- 回合开始朝向；
- 朝向坐标系中的前向位移；
- 是否由物理失败终止。

升级条件改为：没有物理失败、回合至少 3 s、前向位移至少 2 m。2 s 内物理失败才会降级。

测试：

```bash
python3 tools/test_terrain_curriculum.py
```

结果：

```text
legacy terrain curriculum contract: PASS
```

旧 `mixed_slopes_20261111` profile 的真实 Isaac Sim 两步构造与复位预检同样通过：

```text
status: PASS
steps: 2
physics_steps: 8
resets: 0
max_closure_m: 0.000202062
```

证据：

```text
runs/curriculum_runtime_smoke_20261112_103500/summary.json
```

## 修复后验证

### 6,000 步无渲染探针

```text
status: PASS
steps: 6000
physics_steps: 24000
resets: 0
sim_seconds: 120
route waypoints reached: 4 / 9
max_closure_m: 0.001164
```

该探针安全穿过旧地图第 4,556 步的失败位置。

证据：

```text
runs/route_seam_probe_20261112_093028/summary.json
```

### 27 项单块评估

原始 `model_50.pt` 在 9 种地形上各运行 3 次、每次 12 s，结果 `27/27` 成功，无 reset。

| 地形 | 成功次数 | 平均 `vx` MAE (m/s) | 平均绝对横向偏移 (m) |
| --- | ---: | ---: | ---: |
| down10 | 3/3 | 0.0166 | 0.169 |
| up10 | 3/3 | 0.0117 | 0.344 |
| up14 | 3/3 | 0.0146 | 1.899 |
| up06 | 3/3 | 0.0093 | 0.447 |
| down14 | 3/3 | 0.0125 | 0.260 |
| flat | 3/3 | 0.0116 | 0.121 |
| up18 | 3/3 | 0.0245 | 1.145 |
| down06 | 3/3 | 0.0146 | 0.363 |
| down18 | 3/3 | 0.0160 | 0.0891 |

该表使用固定零高度起步点，速度指令为 0.65 m/s。横向偏移仍是需要继续优化的指标，尤其在 14% 和 18% 上坡。

证据：

```text
runs/eval_after_y_taper_20261112_100732/summary.json
```

### 连续九块冻结策略回放（未通过）

使用同一 `model_50.pt`、随机地形 profile 和蛇形路线录制连续九块视频。机器人成功走完前两行，并在第 11,778 个策略步进入最后一行第 6 块时触发一次 `contact` 终止：

```text
终止位置（世界坐标）: (-41.3097, -11.9700, 0.2826) m
终止位置（地图局部坐标）: (-17.3097, 6.0300) m
终止类型: contact
roll / pitch / numerical / spring working range: 0
终止前三步机身 x 速度: 0.0299, 0.0125, -0.0004 m/s
```

这是从第 3 块上坡进入最后一行第 6 块上坡时的横向换行位置。终止前机器人明显减速，随后出现非轮部件接触，符合“冻结平地策略未主动抬高机架，碰到上坡过渡”的表现。本次日志只确认了终止类别，没有逐 link 记录具体接触刚体，因此不能据此认定某一个连杆先碰地。

该失败不等同于地图重新出现了垂直断层：

- 解析几何和导出碰撞网格仍证明行边界高度为零；
- 这 12,669 步中没有 roll、pitch、数值发散或气弹簧工作范围终止；
- 最大机械闭合误差仍为 `0.001164 m`，低于 `0.005 m` 验收线。

录像在 reset 后被手动停止以保留失败现场，最终 `summary.json` 为 `ERROR`。证据只用于说明冻结策略在连续换行上坡处仍有能力缺口，不能作为九块连续路线 PASS：

```text
runs/route_all9_seamfixed_20261112_100359/summary.json
runs/route_all9_seamfixed_20261112_100359/rollout.jsonl
runs/route_all9_seamfixed_20261112_100359/policy.mp4
```

## Baseline 选择

当前推荐继续使用原始 `model_50.pt`：

```text
/data/wheeled-legged/projects/v14_6_downhill_pilot_8KMxPj/runs/
train_udMln2/logs/rsl_rl/wheelbipe_v14_6_dr_20261108/
2026-11-11_06-29-46_downhill_pilot/model_50.pt
```

SHA-256：

```text
dd55308419fbd04eecaf213588dd1249a09ec7eba86073d8338d0b06b46c3a61
```

修复边界后、最终 27 项评估前的三模型对照：

| 模型 | 27 项 pass | 平均 `vx` MAE (m/s) |
| --- | ---: | ---: |
| 原 `model_50.pt` | 27/27 | 0.0152 |
| near-eval `model_80.pt` | 27/27 | 0.0715 |
| straight `model_50.pt` | 27/27 | 0.0425 |

两个微调候选在该评测中都能完成地形，但速度跟踪明显差于原模型。因此当前不能把“能走完”解释为“已经优于 baseline”。

## 尚未完成

- 九块连续路线录像未通过：冻结策略在第 11,778 步发生一次非轮部件接触，因此没有 `route.completed=true`。该段应纳入后续训练和课程起点，不能把当前冻结策略当作连续地形通过样本。
- 本次核查没有启动正式 PPO 训练。地图接缝、高度奖励和课程逻辑已经完成静态与运行预检；正式训练前仍需用户 Review 本次连续路线失败与训练配置。
- 27 项评估只使用一个指令速度（0.65 m/s）和固定起步条件；不能代替多速度、多种子和随机复位的泛化测试。

