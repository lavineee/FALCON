# 阀门真实接触转动 Demo 工作日志

## 任务目标

本阶段目标是在已有 weld/soft-connect 阀门角度跟踪 baseline 的基础上，推进到更真实的接触转动任务：

- 机器人自动进入稳定站立；
- 右手/夹爪移动到阀门预定抓点；
- 依靠手部/代理接触几何与阀门形成稳定接触，而不是永久 weld；
- 支持配置 30、45、60 度等目标转角；
- 记录阀门角度、末端误差、接触状态、足底状态和机身姿态；
- 生成可复现运行命令、日志、视频或录制步骤。

## 阶段 0：工作现场

当前分支：

```text
feature-with-hand-ee-tracking
```

最近提交：

```text
8edc08e Improve EE tracking diagnostics and tuning
449116a Document sim2real tracking and valve workflows
4c095bd Add with-hand EE tracking evaluation
f8cd9fe v2.2新增了主动转阀门的功能
```

当前工作区有未提交修改，主要集中在：

- `sim2real/README.md`
- `sim2real/sim_env/loco_manip.py`
- `sim2real/sim_env/loco_manip_with_hand.py`
- `sim2real/rl_policy/loco_manip/*valve*.py`
- `sim2real/config/g1/*valve*.yaml`
- `humanoidverse/data/robots/g1/*valve*.xml`
- 新增 launch 脚本和分析工具

已建立工作目录：

- `docs/worklogs/`
- `artifacts/demo_logs/`
- `artifacts/demo_videos/`

当前最近一次真实接触抓握诊断日志：

```text
artifacts/demo_logs/valve_grasp_contact_no_wrist_v17.csv
```

这轮测试中过滤了腕部接触，避免把 `right_wrist_yaw_link` 碰到阀门误判为掌心贴合。结果显示可以短暂形成 `PTIM` 完整接触，但 hold 阶段仍会退化为局部接触，相对滑移继续增大，说明真实接触版本还不能直接进入转动任务。

## 阶段 1：只读分析

### 关键文件

仿真环境：

- `sim2real/sim_env/loco_manip.py`
  - 负责 MuJoCo 主仿真、阀门 joint/equality 初始化、阀门角度读取、阀门力矩/角度锁、marker 可视化和 status/control JSON。
- `sim2real/sim_env/loco_manip_with_hand.py`
  - 在 `loco_manip.py` 基础上增加 29DoF policy 到带手模型的 actuator 映射，以及左右手开合 PD 控制。
  - 当前右手闭合支持 contact-hold：检测掌心、拇指、食指、中指接触后锁定当前手指角附近，避免继续硬夹。

baseline 任务：

- `sim2real/rl_policy/loco_manip/loco_manip_valve_task_7dof_with_hand.py`
  - 分层状态机：`move_pregrasp -> approach_grasp -> close_hand -> attach_weld -> turn_valve -> hold -> done`。
  - 支持 `valve_attachment_mode = none | connect`。
  - `connect` 模式打开 MuJoCo equality，使手掌点和阀门抓点软连接。
- `sim2real/rl_policy/loco_manip/loco_manip_valve_angle_tracking_7dof_with_hand.py`
  - 软连接角度跟踪 baseline。
  - 每段转角开始时冻结阀门中心、轴线、初始半径向量；
  - 根据目标角度生成圆弧末端轨迹；
  - `closed_loop` 模式用阀门实际角度误差修正末端圆弧命令。

真实接触抓握诊断：

- `sim2real/rl_policy/loco_manip/loco_manip_valve_grasp_contact_7dof_with_hand.py`
  - 当前只验证接近、闭手、接触保持，不主动转动；
  - 不启用 equality；
  - 记录 `contact_topology`、`relative_slip_m`、`close_slip_m` 等抓握质量指标。

模型和场景：

- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand_valve_task.xml`
  - soft-connect/weld baseline 场景；
  - 使用原始较细阀门 `valve_body_include.xml`。
- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand_valve_grasp_contact.xml`
  - 真实接触抓握场景；
  - 使用整体加粗阀门 `valve_thick_body_include.xml`。
- `humanoidverse/data/robots/g1/g1_29dof_old_freebase_with_hand.xml`
  - 带手 G1 模型；
  - 右手新增 `right_grasp_proxy_palm_pad/thumb/index/middle` 接触代理几何。

launch：

- `sim2real/launch_valve_angle_tracking_7dof_with_hand_auto.sh`
  - 自动启动 soft-connect 角度跟踪 baseline；
  - 有 cleanup，`Ctrl+C` 会清理 sim/policy 子进程。
- `sim2real/launch_valve_grasp_contact_7dof_with_hand_auto.sh`
  - 自动启动真实接触抓握诊断；
  - 有 cleanup。

### 当前控制链路

1. MuJoCo sim 侧读取 XML，初始化阀门 joint、site、equality、hand actuator。
2. sim 每个周期写 `sim_status_file`，包含：
   - robot base 世界位姿；
   - valve center/grasp 世界坐标；
   - valve angle / velocity；
   - hand-valve contact count、force、contact pairs；
   - foot contact force。
3. policy 侧通过 `SimStatusValveGeometryProvider` 读取 status，把阀门世界坐标转换到机器人 base 坐标系。
4. policy 通过 IK 产生右臂 7DoF 目标，叠加原 FALCON 上肢 residual 后下发 29DoF policy 命令。
5. with-hand sim 侧额外用 PD 控制手指开合，policy 本身仍只输出 29DoF 本体关节。
6. marker JSON 由 policy 写出，sim 侧读取后在 MuJoCo 窗口可视化目标点、当前点、圆弧和状态文字。

### weld / soft-connect baseline 的角度控制方式

当前可复现 baseline 不是严格 weld，而是 `connect` 软点连接：

- XML equality：
  - `right_hand_valve_connect`: 连接 `right_palm_grasp_center` 和 `right_hand_valve_site`；
  - `right_hand_valve_weld`: 保留但当前 baseline 不优先用。
- `valve_angle_tracking_control_mode: closed_loop`：
  - 读取阀门实际角度；
  - 计算目标角误差；
  - 用 `feedback_gain`、`command_lead_deg`、`command_speed_deg` 修正末端圆弧目标；
  - 目标是让阀门实际角度收敛到目标转角，而不是只让手沿开环圆弧运动。

### 当前真实接触版本的主要问题

最近 v17 诊断显示：

- 可以短暂形成完整 `PTIM` 接触；
- 进入 hold 后接触拓扑很快退化为 `P-I-`、`PT--`、`PTI-` 等局部接触；
- 相对滑移从 0 增大到约 9-18 cm；
- 这说明现在还不是稳定的“抓住阀门”，更像掌垫/手指与轮圈局部挤压后滑移。

已定位的风险点：

- 真实夹爪几何和阀门轮圈形状不完全匹配；
- 手指是简单位置/力矩 PD 开合，不是接触力闭环抓握；
- 阀门轮圈是圆管，当前三指手接触容易从圆管上滑脱；
- 上肢 policy residual 不可关闭，会对 IK 目标产生抗扰动残差；
- 真实接触转动会比 soft-connect baseline 更依赖摩擦、接触刚度和手指包络稳定性。

### 最小修改方案

接下来不做大范围重构，按最小改动合并两条已有路径：

1. 先复现 `valve_angle_tracking` soft-connect baseline，保存角度误差日志，作为上限对照。
2. 在 `valve_grasp_contact` 路径上增加转动阶段：
   - 接近并闭手；
   - 只有接触拓扑和滑移满足阈值后进入 `TURN_VALVE`；
   - 使用 `valve_angle_tracking` 的 closed-loop 圆弧命令；
   - 不启用永久 weld；
   - 可选提供明确标注的 debug 模式，例如弱软连接/接触辅助，但不能作为最终真实接触结果。
3. 扩展 CSV：
   - `valve_target_angle_deg`
   - `valve_actual_angle_deg`
   - `valve_angle_error_deg`
   - base roll/pitch/yaw
   - foot contact force/stable
   - action norm 或 command norm
   - contact topology/force/slip
4. 每轮测试保存到 `artifacts/demo_logs/`，达标后再生成视频到 `artifacts/demo_videos/`。

## 阶段 2 待执行：weld/soft-connect baseline 复现

已运行 soft-connect baseline：

```bash
cd /home/lavine/project/FALCON
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
LOG_FILE=/home/lavine/project/FALCON/artifacts/demo_logs/valve_angle_tracking_soft_connect_30deg_baseline.csv \
MARKER_FILE=/tmp/falcon_valve_angle_tracking_soft_connect_30deg_baseline_markers.json \
SIM_STATUS_FILE=/tmp/falcon_valve_angle_tracking_soft_connect_30deg_baseline_status.json \
SIM_CONTROL_FILE=/tmp/falcon_valve_angle_tracking_soft_connect_30deg_baseline_control.json \
DURATION_SEC=45 \
VALVE_ANGLE_TARGETS_DEG=30 \
VALVE_ANGLE_SPEED_DEG=10 \
VALVE_ANGLE_CONTROL_MODE=closed_loop \
ELASTIC_LENGTH=-0.05 \
bash sim2real/launch_valve_angle_tracking_7dof_with_hand_auto.sh
```

本次运行时首次传入的是相对路径，脚本内部 `cd sim2real` 后写到了 `sim2real/artifacts/demo_logs/`；已复制到根目录：

```text
artifacts/demo_logs/valve_angle_tracking_soft_connect_30deg_baseline.csv
```

baseline 结果：

- 目标转角：`30.0 deg`
- 实际最终转角：`25.62 deg`
- 最终误差：`4.39 deg`
- 转动过程轨迹误差 MAE：`6.01 deg`
- 末端误差均值 / p90 / max：`0.0276 / 0.0376 / 0.0399 m`
- soft-connect 拉伸均值 / p90 / max：`0.0033 / 0.0067 / 0.0073 m`
- 阀门最大角速度：`0.534 rad/s`

判断：

- soft-connect baseline 可复现，角度响应基本稳定；
- 30 度目标下最终误差约 4.4 度，不是严格 2 度内，但已经能作为当前真实接触版本的对照上限；
- 后续真实接触版本首先要达到“稳定转动且不滑脱”，再逐步逼近 4-5 度以内的最终误差。

## 阶段 3：真实接触小角度转动 baseline

新增真实接触转动入口：

- `sim2real/rl_policy/loco_manip/loco_manip_valve_grasp_contact_7dof_with_hand.py`
- `sim2real/config/g1/g1_29dof_with_hand_valve_contact_turning_overlay.yaml`
- `sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh`

推荐运行：

```bash
cd /home/lavine/project/FALCON
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh
```

当前流程：

1. 自动打开带手 G1 和加粗阀门场景。
2. 启动策略并自动释放弹力绳。
3. 右手移动到阀门预抓取点。
4. 掌心/手指与轮圈建立真实接触后闭手。
5. 只有抓握拓扑、接触力、相对滑移和抓点误差满足阈值后，才进入转动。
6. 进入转动前先做 `0.6 s` 预稳定，释放阀门角度锁时降低初始冲击。
7. 使用阀门角度闭环修正末端圆弧命令，当前默认执行 `10 deg` 小角度转动。

当前关键参数：

- 目标转角：`valve_turn_target_deg: 10.0`
- 转动速度：`valve_turn_speed_deg: 0.8`
- 闭环增益：`valve_turn_feedback_gain: 1.4`
- 命令前瞻：`valve_turn_command_lead_deg: 6.0`
- 转动半径模式：`valve_turn_radius_mode: command_delta_scaled`
- 转动位移半径缩放：`valve_turn_displacement_radius_gain: 0.68`
- 末端接触补偿：`valve_turn_contact_compensation_enabled: true`
- 目标到达即进入保持：`valve_turn_finish_on_target_reached: true`
- 阀门 hinge：`damping=5.3`，`frictionloss=1.3`
- 预转动稳定：`valve_turn_pre_settle_s: 0.6`
- 手部闭合：`hand_command_ramp_s: 2.5`，`hand_kp: 3.0`，`hand_kd: 0.25`
- 右手力矩限幅：`[1.5, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]`
- contact hold 力上限：`45 N`

这些参数的意图是避免闭手阶段过硬夹紧。早期版本闭手峰值可到 `60 N` 以上，容易把轮圈接触从 `P-I-` 挤成局部滑移；当前版本把闭手速度和力矩降下来，让 `PTIM` 包络更容易稳定出现。

## 阶段 4：复现实验结果

本阶段保留两次连续成功日志：

- `artifacts/demo_logs/valve_contact_turning_20260426_211334.csv`
- `artifacts/demo_logs/valve_contact_turning_20260426_211432.csv`

对比图：

- `artifacts/demo_logs/valve_contact_turning_contact_demo_summary_20260426_211334_211432.png`

结果汇总：

| 日志 | 目标角 | 最终实际角 | 最终误差 | 轨迹 MAE | 末端误差均值 | 滑移 p90 | 接触力均值 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `211334.csv` | 5.0 deg | 4.23 deg | 0.77 deg | 0.56 deg | 15.27 cm | 2.72 cm | 12.58 N | 成功 |
| `211432.csv` | 5.0 deg | 5.10 deg | -0.10 deg | 0.22 deg | 14.64 cm | 1.80 cm | 19.51 N | 成功 |

状态切换均正常：

```text
move_pregrasp -> approach_grasp -> close_hand -> pre_turn_settle -> turn_valve -> turn_hold -> done
```

第二次复现实验中，进入转动时为 `PTIM` 完整接触，进入 hold 时仍保持 `PTIM`，最终 done 时退化为 `-TIM`，但没有出现飞脱、弹飞或明显失稳。base 最终姿态约为 `roll=-2.63 deg`、`pitch=2.91 deg`，在可接受范围内。

## 阶段 5：10 度真实接触转动迭代

在 `5 deg` baseline 稳定后，继续把目标角提高到 `10 deg`。这一阶段的主要问题不是阀门角度闭环本身，而是“末端命令轨迹的等效转动半径”和“真实接触保持”之间的矛盾：

- `command` 半径模式从当前命令抓点出发，初始目标连续，但等效转动半径只有约 `5 cm`，对 `10 deg` 来说切向位移偏小，阀门实际转角不足；
- `target` 半径模式使用真实阀门抓点半径，理论位移更合理，但切换到转动阶段时容易产生目标跳变，导致接触拓扑退化；
- 新增的 `command_delta_scaled` 模式以当前已经接触的命令抓点作为零跳变起点，同时借用阀门抓点半径生成更大的切向位移，避免一开始就把手从接触区域拉开。

本阶段新增/调整：

- `loco_manip_valve_grasp_contact_7dof_with_hand.py`
  - 增加 `valve_turn_radius_mode: command_delta_scaled`；
  - 增加 `valve_turn_displacement_radius_gain`；
  - CSV 新增 `turn_motion_radius_m`，用于记录实际用于末端圆弧命令的运动半径；
  - 转动开始时记录当前命令抓点，后续目标按“当前命令点 + 缩放半径对应的切向位移”生成。
- `g1_29dof_with_hand_valve_contact_turning_overlay.yaml`
  - 默认目标角改为 `10 deg`；
  - 保持温和手部闭合参数，避免强夹导致接触拓扑被挤散；
  - 启用小幅接触补偿，减少转动阶段掌心/手指离开轮圈。
- `sim2real/tools/analyze_valve_contact_turning_metrics.py`
  - 增加真实接触转动 CSV 的定量分析和画图入口。

10 度测试结果：

| 日志 | 关键设置 | 最终实际角 | 最终误差 | 轨迹 MAE | 滑移 p90 | 接触力均值 | PTIM 占比 | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `232624.csv` | `command` 半径 | 6.09 deg | 3.91 deg | 2.38 deg | 3.18 cm | - | - | 半径偏小，转不够 |
| `233013.csv` | `target` 半径 | 0.38 deg | 9.62 deg | 6.08 deg | 2.31 cm | - | - | 目标跳变，接触退化 |
| `233559.csv` | 初版 `command_delta_scaled` | 2.29 deg | 7.71 deg | 5.08 deg | 3.70 cm | - | - | 位移增大但抓握保持不足 |
| `233856.csv` | 缩小半径增益 + 接触补偿 | 8.95 deg | 1.05 deg | 2.35 deg | 2.22 cm | - | - | 基本可用 |
| `234055.csv` | 当前默认参数 | 9.27 deg | 0.73 deg | 3.04 deg | 1.88 cm | 24.83 N | 0.61 | 当前最好 |
| `234214.csv` | 强制跑完整转动时长 | 8.42 deg | 1.58 deg | 2.54 deg | 4.08 cm | 16.61 N | 0.42 | 后段滑移更明显，弃用 |

当前最好结果：

- CSV：`artifacts/demo_logs/valve_contact_turning_20260426_234055.csv`
- 报告：`artifacts/demo_logs/valve_contact_turning_10deg_best_report.md`
- 图表：`artifacts/demo_logs/valve_contact_turning_10deg_best_report.png`

最好一轮的定量指标：

- 目标转角：`10.00 deg`
- 最终实际转角：`9.27 deg`
- 最终误差：`0.73 deg`
- 转动阶段时长：`7.86 s`
- 末端误差均值 / p90：`17.72 / 18.21 cm`
- 相对滑移均值 / p90：`1.36 / 1.88 cm`
- 接触力均值 / p90：`24.83 / 35.90 N`
- base tilt 均值 / p90：`3.92 / 4.27 deg`

一个重要结论是：在真实接触版本里，强行让手继续完整执行名义圆弧不一定更好。`234214.csv` 中关闭“达到目标即进入 hold”后，后半段接触力下降、滑移增大，最终误差反而变大。因此当前默认保留 `valve_turn_finish_on_target_reached: true`，让阀门实际角度进入目标邻域后尽快切到保持状态。

## 当前判断

当前版本已经跑通了“非永久 weld / 非 connect 辅助”的真实接触小角度转阀门 baseline。`5 deg` 目标下连续两次实验最终误差约 `0.1-0.8 deg`；在当前默认参数下，`10 deg` 目标的最好一轮最终误差为 `0.73 deg`，说明真实接触转动已经具备可复现的小角度角度控制能力。

仍需明确的局限：

- 目前可靠验证到 `10 deg` 小角度，不等价于已经能稳定完成 `30/45/60 deg` 大角度。
- 末端位置误差仍在 `15-18 cm` 量级，其中包含 RL residual、抓握几何 offset 和真实接触挤压带来的误差；角度闭环能容忍这一误差，但后续部署需要继续标定掌心工作点。
- 转动后半段接触力会下降，拓扑可能从 `PTIM` 退化到 `-TIM` 或 `-T-M`，说明抓握保持仍不是完全刚性的“手-阀门局部相对姿态不变”。
- 当前没有使用永久 weld，也没有启用 `valve_contact_assist_enabled`；但阀门接触几何、摩擦和手指力矩仍是为 demo 调过的高保真近似，不应直接声称已经等价真实硬件抓握。
- 本轮生成了 CSV 和对比图，尚未自动生成视频。当前环境中 `ffmpeg` 不在 `PATH`，不能直接用命令行录屏；视频可通过桌面录屏记录 MuJoCo 窗口，后续可以增加离屏渲染或窗口录制脚本。

手动录屏步骤：

1. 打开桌面录屏工具，选择 MuJoCo 窗口区域。
2. 运行：

   ```bash
   cd /home/lavine/project/FALCON
   bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh
   ```

3. 在 MuJoCo 窗口显示 `turn_valve -> turn_hold -> done` 后停止录屏。
4. 建议保存到：

   ```text
   artifacts/demo_videos/valve_contact_turning_10deg_YYYYMMDD_HHMMSS.mp4
   ```

下一步建议：

1. 在当前 `10 deg` baseline 上增加 `15/20 deg` 递增目标角测试，不要直接跳到 60 度。
2. 增加“抓住后局部相对位姿保持”指标：记录掌心/指尖代理点在阀门局部坐标下的漂移。
3. 如果大角度转动时仍滑脱，优先优化抓握几何和手指接触力闭环，而不是继续降低阀门阻尼。
4. 达到稳定 `15-20 deg` 后，再考虑视觉/深度相机替换当前 MuJoCo 特权阀门位姿。

## 阶段 6：阀门模型几何复核

日期：2026-04-27

用户复查 MuJoCo 画面后指出，阀门中心 hub/轴承区域存在明显倾斜，后方灰色支架方块也可能干扰手部抓握和轮盘转动。该反馈是正确的，后续真实接触测试应先暂停控制参数调试，先把阀门模型本身修正。

本轮定位到的模型问题：

- 主 G1 MuJoCo 模型使用 `<compiler angle="radian">`，但多个阀门 include 中仍写了 `euler="0 90 0"`；
- MuJoCo 会把 `90` 当作 90 rad，而不是 90 deg，导致中心圆柱、hub 或轴承件在视觉上出现异常倾斜；
- round-grip 阀门的支架方块尺寸偏大，虽然场景中排除了支架和轮盘的内部接触，但它仍会影响视觉判断，也可能压缩手部接近空间。

已完成的修正：

- 将各阀门 include 中的 `euler="0 90 0"` 改为 `euler="0 1.57079632679 0"`；
- 在 `valve_round_grip_body_include.xml` 中缩小并后移 `valve_block`：
  - 原尺寸：`0.11 0.11 0.09`；
  - 当前尺寸：`0.045 0.055 0.045`；
  - 当前位置：`-0.015 0 1.00`；
- 保持阀门轮缘为圆管 capsule 近似，不再使用长方体抓握块。

模型验证结果：

- MuJoCo 编译后，`valve_stem`、`valve_bearing_housing`、`wheel_hub` 的圆柱轴线均与阀门 hinge/world x 方向一致；
- 后方灰色方块朝向轮盘的面约为 `x=0.550 m`；
- 当前轮盘/轮缘保守外包络最大约为 `x=0.510 m`；
- 方块与旋转轮盘之间保守间隙约 `4 cm`；
- 正反向 5 Nm torque sweep 中，阀门可连续转过约 `+180 deg` 到 `-162 deg`，未记录到 wheel-block 接触。

本轮生成的检查资产：

- 静态模型图：`artifacts/model_renders/valve_round_grip_geometry_check.png`
- 自由转动 GIF：`artifacts/demo_videos/valve_round_grip_free_rotation_check.gif`

当前判断：

- 阀门中心倾斜问题已从根因上修正；
- 灰色支架方块不再贴近轮盘，当前几何可以继续作为真实接触抓握测试基础；
- 下一步应先复测“只抓住但不转动”的静态接触保持，确认阀门角度不会因抓握本身发生跳变，再恢复小角度真实接触转动测试。

### 统一轮缘尺寸更新

用户指出局部加粗胶套虽然有助于某一个抓点，但不符合真实阀门外观，也会把任务变成“抓特殊凸起”，而不是抓标准轮缘。按夹爪接触代理尺寸重新检查后，本轮将阀门外圈改为统一管径：

- 右手指腹接触代理半径约 `17 mm`；
- 掌心接触代理半宽约 `20 mm`；
- 旧局部胶套半径 `38 mm`，直径约 `76 mm`，视觉和接触上都偏突兀；
- 第一版统一轮缘半径设为 `30 mm`，直径约 `60 mm`。实际观察后发现该尺寸对当前三指夹爪偏大，食指/中指难以形成完整包络，因此进一步降为 `26 mm` 半径，直径约 `52 mm`。

对应改动：

- 删除 `grip_round_00..04` 局部加粗胶套；
- 将 `rim_00..15` 全部设为 `size="0.026"`；
- 将抓取 radial inset 从 `0.038 m` 调整为 `0.026 m`，让掌心工作点仍落在轮缘内侧接触面附近；
- 更新 debug geom 名称，后续日志只关注统一轮缘 `rim_*`。

验证结果：

- MuJoCo 编译通过，`grip_round_*` 已不存在，`rim_00..15` 共 16 段统一轮缘均存在；
- 抓取 site 仍位于阀门中心 `0.16 m` 半径处，后续可沿轮缘任意角度选取同等尺寸抓点；
- 正反向 `5 Nm` torque sweep 中无 wheel-block 接触，阀门仍能连续转动；
- 新静态检查图：`artifacts/model_renders/valve_uniform_rim_geometry_check.png`。

## 阶段 7：面向实机部署的软辅助转动 baseline

日期：2026-04-27

用户明确指出，本项目的核心目标是最终在实机上完成阀门旋转，而不是在 MuJoCo 中无限追求纯真实接触的完美复现。因此本阶段把验收目标调整为：抓握动作视觉和接触上基本成立后，允许使用柔顺连接或角度制动等工程近似，先验证“抓住并按目标角转动”的控制流程。

本轮关键判断：

- 单纯追求手指-阀门真实摩擦接触，容易陷入 MuJoCo 接触参数、三指手几何和轮缘形状的局部调参；
- 目前更有价值的是形成一条可复现的实机候选流程：阀门位姿输入、末端接近、闭手抓握、沿阀门圆弧生成目标、根据阀门角度反馈停止；
- 因此柔顺 `connect` 只作为“抓住后传力”的仿真近似，不再追求一开始就靠它把手拉到阀门；
- 到达目标附近后必须释放或卸载 `connect`，否则会继续推动阀门产生过冲。

对应代码改动：

- 在 `loco_manip_valve_grasp_contact_7dof_with_hand.py` 中新增 `valve_contact_assist_release_on_turn_hold`；
- 进入 `turn_hold` 时冻结末端圆弧命令，并按配置释放柔顺 `connect`；
- 新增 `valve_turn_hold_angle_brake_enabled`，只在 `turn_hold` 阶段启用有限力矩阀门角度保持，用于吸收 MuJoCo 中残余角速度；
- 在 `launch_valve_grasp_contact_7dof_with_hand_policy.sh` 中增加环境变量 `VALVE_TURN_HOLD_ANGLE_BRAKE_ENABLED`，便于后续测试开关；
- 更新 `sim2real/README.md`，明确该 baseline 是“软连接 + 目标附近制动”的工程近似。

验证记录：

- `artifacts/demo_logs/valve_contact_turning_20260427_073523.csv`
  - 只释放 `connect`，不加角度制动；
  - 目标 `10 deg`；
  - 进入 `turn_hold` 时实际约 `8.85 deg`，但阀门仍有约 `0.18 rad/s` 残余角速度；
  - 释放 `connect` 后继续过冲到约 `20 deg`，说明仅释放约束不足，需要目标附近制动。

- `artifacts/demo_logs/valve_contact_turning_20260427_073819.csv`
  - 释放 `connect`，并在 `turn_hold` 打开有限力矩角度制动；
  - 目标 `10 deg`；
  - `turn_valve` 阶段末尾实际约 `8.78 deg`，说明阀门确实由手端圆弧带动到目标附近；
  - `turn_hold` 末尾实际约 `12.83 deg`，后续 `done` 阶段收敛到约 `11.26 deg`；
  - 最终误差约 `-1.26 deg`，较无制动版本明显改善。

当前结论：

- 这一版已经能作为“带手阀门转动 10 度”的可复现工程 baseline；
- 它不是纯真实接触结论，文档中必须标注柔顺 `connect` 和角度制动是 MuJoCo demo 近似；
- 下一步可以在该 baseline 上测试 `20 deg`，再逐步到 `30 deg`，不要直接跳到 `60 deg`；
- 如果大角度时仍明显滑脱，应优先从实机可继承角度处理：闭手力、目标角速度、阀门抓点和视觉/角度闭环，而不是继续追求 MuJoCo 接触完全真实。

## 阶段 8：稳定性约束下的闭环角度跟踪

日期：2026-04-27

用户进一步明确验收标准：机器人必须在保持自身姿态的情况下完成闭环角度跟踪；摔倒、明显被阀门拉近、脱离抓握等都应判为失败，不能只看阀门角度是否到目标。

本轮先分析 `20260427_073819.csv`，发现虽然 10 度角度最终可以接近目标，但机器人在转动和保持阶段持续靠近阀门：

- `turn_valve` 阶段 base 到阀门 x 距离从约 `0.443 m` 降到 `0.365 m`；
- `turn_hold` 末尾降到约 `0.274 m`；
- `done` 末尾降到约 `0.206 m`；
- base yaw 从约 `15 deg` 漂到约 `-17 deg`。

这说明“角度到了”并不等价于“任务成功”。主要原因不是阀门阻尼过大，而是持续接触/柔顺连接把末端误差转化成了对机器人身体的拉拽；原 FALCON policy 没有专门训练闭链接触下的阀门操作，因此需要在任务层加入稳定性约束。

本轮修改：

- 增加转动阶段失败判据：
  - `valve_turn_abort_min_base_to_valve_x_m: 0.30`
  - `valve_turn_abort_max_base_approach_m: 0.12`
  - `valve_turn_abort_max_base_yaw_drift_deg: 20.0`
  - `valve_turn_abort_base_tilt_deg: 12.0`
- 在转动开始时记录 base 到阀门距离和 base yaw，后续每帧检查是否超限；
- 把默认转动速度从 `0.8 deg/s` 提高到 `1.5 deg/s`，减少持续接触时间；
- 把 `turn_hold` 从 `2.0 s` 缩短到 `1.0 s`，把 done 后停留从 `2.0 s` 缩短到 `0.7 s`；
- 更新分析脚本，报告中加入 base 靠近量和 yaw 漂移。

验证结果：

- CSV：`artifacts/demo_logs/valve_contact_turning_20260427_080148.csv`
- 报告：`artifacts/demo_logs/valve_contact_turning_stability_guard_10deg_report.md`
- 图表：`artifacts/demo_logs/valve_contact_turning_stability_guard_10deg_report.png`

关键指标：

- 目标角：`10 deg`
- `turn_hold` 末尾实际角：`9.74 deg`
- 整次运行末尾实际角：`10.20 deg`
- 整次运行末尾角度误差：约 `-0.20 deg`
- 转动/保持阶段 base 靠近量：约 `0.064 m`
- 整次运行 base 靠近量：约 `0.085 m`
- 转动/保持阶段 base yaw 漂移：约 `4.76 deg`
- 整次运行 base yaw 漂移：约 `8.59 deg`
- base tilt P90：约 `6.38 deg`

当前判断：

- 这一轮满足“角度闭环 + 姿态/站位稳定”这两个核心条件；
- 提速并没有导致角度失控，反而减少了持续接触拖拽时间；
- 后续扩大目标角时，必须继续使用这套稳定性门限，不能只报告阀门角度误差；
- 如果 20/30 度出现稳定性失败，优先考虑分段转动、缩短每段接触时间、降低径向推压，而不是单纯增大软连接刚度。

## 阶段 9：严格单目标工作区间评估

日期：2026-04-27

用户指出：阀门角度“到过目标”不应等于任务成功，机器人必须在目标附近保持住，并且手爪不能在 hold 阶段脱离。考虑到人手大角度拧阀门也会分段重抓，当前目标调整为：先找到单次稳定工作区间，而不是强行追求 90/120 度单次旋转。

本轮修改：

- 在 `loco_manip_valve_grasp_contact_7dof_with_hand.py` 中加入目标保持质量门槛：
  - 到达目标后继续检查角度误差、阀门角速度、抓握状态、接触健康度和相对滑移；
  - 若角度已到但抓握丢失，任务判失败；
  - summary 中记录 `final_grasp_success`、`final_contact_health_ratio`、`final_relative_slip_m` 等字段。
- 在 `launch_valve_contact_turning_7dof_with_hand_auto.sh` 中加入 `strictXX` 单目标 preset：
  - 例如 `strict30`、`strict45`、`strict60`；
  - 初始默认速度 `1.0 deg/s`，后续 45 度区间优化为 `0.9 deg/s`；
  - 完成条件要求 hold 稳定；
  - strict preset 使用稳定三点包络判据：掌心 + 拇指 + 至少一根手指，同时要求最终接触健康比例不低于 `0.75`。
- 在 `launch_valve_grasp_contact_7dof_with_hand_policy.sh` 中增加环境变量：
  - `VALVE_GRASP_SUCCESS_REQUIRE_INDEX`
  - `VALVE_GRASP_SUCCESS_REQUIRE_MIDDLE`
  - `VALVE_GRASP_SUCCESS_REQUIRE_REAL_HAND_CONTACT`
  - `VALVE_CONTACT_ASSIST_RELEASE_ON_TURN_HOLD`

运行入口：

```bash
cd /home/lavine/project/FALCON
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict30
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict45
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict60
```

验证记录：

- `artifacts/demo_logs/valve_contact_turning_20260427_173134.csv`
  - `strict30`；
  - 最终角度约 `30.43 deg`，误差约 `-0.43 deg`；
  - 最终抓握成立，最终滑移约 `4.4 mm`；
  - 判定成功。
- `artifacts/demo_logs/valve_contact_turning_20260427_173802.csv`
  - `strict45`；
  - 最终角度约 `44.62 deg`，误差约 `0.38 deg`；
  - 最终抓握成立，最终滑移约 `1.7 cm`；
  - 判定成功。
- `artifacts/demo_logs/valve_contact_turning_20260427_181805.csv`
  - `strict45`，速度 `1.0 deg/s`；
  - 最终角度约 `44.45 deg`，但 hold 阶段最终接触健康比例降到 `0.50`；
  - 严格判定失败，说明 45 度区间的主要不确定性来自 hold 阶段接触拓扑，而不是角度闭环。
- `artifacts/demo_logs/valve_contact_turning_20260427_182841.csv`
  - `strict45`，速度 `0.9 deg/s`；
  - 最终角度约 `44.86 deg`，误差约 `0.14 deg`；
  - 最终抓握成立，滑移 p90 约 `2.6 cm`，峰值接触力约 `40.5 N`；
  - 判定成功。
- `artifacts/demo_logs/valve_contact_turning_20260427_183028.csv`
  - 更新默认 `strict45` preset 后复测；
  - 最终角度约 `45.61 deg`，误差约 `-0.61 deg`；
  - 最终抓握成立，滑移 p90 约 `3.2 cm`，峰值接触力约 `51.7 N`；
  - 判定成功。
- `artifacts/demo_logs/valve_contact_turning_20260427_173908.csv`
  - `strict60`；
  - 最终角度约 `59.99 deg`，误差约 `0.01 deg`；
  - 但最终接触健康比例降到 `0.50`，`final_grasp_success=false`；
  - 判定失败。
- `artifacts/demo_logs/valve_contact_turning_strict_single_turn_090deg_entry50_20260427_173000.csv`
  - `strict90`；
  - 角度几乎精确到达，但最终抓握丢失、接触健康比例 `0.25`、滑移约 `16 cm`；
  - 判定失败。

当前结论：

- 当前可作为展示 baseline 的稳定单次工作区间约为 `30-45 deg`；
- 对 `45 deg` 单目标，`0.9 deg/s` 比 `1.0 deg/s` 更稳健，比 `0.8 deg/s` 的角度误差更小；
- `60 deg` 以上不是“角度控制做不到”，而是闭链抓握质量在 hold 阶段不足；
- 后续若需要实现大角度阀门转动，应采用“转一段、重置抓点、再转一段”的分段方案；
- 当前论文/组会中可以强调：我们已经建立了严格的成功判据，避免把 soft connect 或 angle brake 带来的角度到达误判为稳定抓握操作。
