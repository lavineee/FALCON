# FALCON sim2real / sim2sim 使用说明

本目录负责 FALCON 的 MuJoCo sim2sim、真实机器人 sim2real 部署，以及本项目新增的末端跟踪、带手模型和阀门交互验证流程。原始 FALCON 的控制链路仍然保持为核心：MuJoCo 端模拟真实机器人的状态发布和命令接收，policy 端通过同一套 SDK 桥接接口读取状态并下发控制命令。

<table>
  <tr>
    <td style="text-align: center;">
      <img src="../assets/deploy.png" style="width: 99%;"/>
    </td>
  </tr>
</table>

## 当前目录的作用

- 原始部署链路：支持 Unitree G1 / H1 / H1-2 和 Booster T1 的 sim2sim / sim2real。
- FALCON loco-manipulation：运行下肢 locomotion 和上肢 manipulation 组合策略。
- 末端跟踪评测：验证右臂末端在机器人 base 坐标系下对随机目标点的跟踪能力。
- 7DoF 手臂扩展：在原 4DoF 上肢 IK 的基础上加入手腕三个自由度，先做位置跟踪验证。
- 带手模型验证：在不修改 29DoF ONNX 策略输出的前提下，增加手部模型和独立开合控制器。
- 阀门交互 demo：保留阀门场景、weld 固连、阀门主动扭矩和主动旋转策略接口，用于后续持续接触任务。

所有命令默认在 `sim2real/` 目录下执行：

```bash
cd /home/lavine/project/FALCON/sim2real
```

本机推荐显式指定 Python：

```bash
export PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python
```

如果你的 shell 已经能直接找到正确环境里的 `python`，也可以不设置 `PYTHON_BIN`。

## 环境准备

已有 `fcreal` 环境时可以跳过创建步骤。

```bash
conda create -n fcreal python=3.10
conda activate fcreal
conda install pinocchio=3.2.0 -c conda-forge
cd /home/lavine/project/FALCON/sim2real
pip install -r requirements.txt
```

Unitree G1 部署需要安装 `unitree_sdk2_python`：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

Booster T1 部署需要 Booster SDK。原始 Booster SDK 没有完整的状态发布和命令接收，这里沿用 FALCON 原文档推荐的 fork：

```bash
git clone https://github.com/hang0610/booster_robotics_sdk
cd booster_robotics_sdk
pip install pybind11 pybind11-stubgen
mkdir build
cd build
cmake .. -DBUILD_PYTHON_BINDING=on
make
sudo make install
```

## 常用配置项

不同实验主要通过 yaml 配置切换。常见字段如下：

```yaml
ROBOT_TYPE: "g1_29dof"
ROBOT_SCENE: "../humanoidverse/data/robots/g1/scene_g1_29dof_freebase.xml"
ASSET_ROOT: "../humanoidverse/data/robots/g1"
DOMAIN_ID: 0
INTERFACE: "lo"
SDK_TYPE: "unitree"
MOTOR_TYPE: "serial"
USE_JOYSTICK: 0
```

需要注意：

- sim2sim 一般使用 `INTERFACE: "lo"`，macOS 上可能是 `lo0`。
- sim2real 要把 `INTERFACE` 改成连接真实机器人的网卡，例如 `eth0`、`enp...`。
- 带手跟踪使用 overlay 配置 `config/g1/g1_29dof_with_hand_tracking_overlay.yaml`，它会覆盖场景、IK 资产路径、手部关节和手部 PD 参数。
- 当前 ONNX 策略仍然只输出原始 29DoF 本体控制量；带手模型多出来的手部 actuator 由单独的 hand controller 控制。

## 基础 FALCON sim2sim

sim2sim 需要先启动 MuJoCo 环境，再启动 policy。sim2real 只启动 policy。

### G1 29DoF Locomotion

```bash
${PYTHON_BIN:-python} sim_env/base_sim.py \
  --config=config/g1/g1_29dof.yaml
```

另开一个终端：

```bash
${PYTHON_BIN:-python} rl_policy/dec_loco/dec_loco.py \
  --config=config/g1/g1_29dof.yaml \
  --model_path=models/dec_loco/g1_29dof.onnx
```

### G1 29DoF FALCON

```bash
${PYTHON_BIN:-python} sim_env/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml
```

另开一个终端：

```bash
${PYTHON_BIN:-python} rl_policy/loco_manip/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

### T1 29DoF FALCON

```bash
${PYTHON_BIN:-python} sim_env/loco_manip.py \
  --config=config/t1/t1_29dof_falcon.yaml
```

另开一个终端：

```bash
${PYTHON_BIN:-python} rl_policy/loco_manip/loco_manip.py \
  --config=config/t1/t1_29dof_falcon.yaml \
  --model_path=models/falcon/t1_29dof.onnx
```

## MuJoCo 和 Policy 按键

MuJoCo 窗口按键：

- `7`：缩短弹力绳，增大上拉力。
- `8`：放长弹力绳，减小上拉力。
- `9`：打开或关闭弹力绳。
- `space`：切换弹力绳力估计显示。
- `backspace`：重置仿真。
- `J`：切换右手和阀门的 weld 固连。
- `K`：切换阀门主动扭矩。
- `U`：阀门扭矩加 1。
- `I`：阀门扭矩减 1。
- `O`：阀门扭矩清零。

Policy 终端按键：

- `]`：开始使用 policy 输出。
- `o`：停止 policy 输出，并把 action 置零。
- `=`：切换站立 / 踏步状态。
- `w` / `s`：增减 x 方向线速度。
- `a` / `d`：增减 y 方向线速度。
- `q` / `e`：增减 z 方向角速度。
- `z`：速度置零。
- `1` / `2`：在策略支持时调整 base 高度。
- `5` / `6`：以 0.01 为步长调整 kp scale。
- `4` / `7`：以 0.1 为步长调整 kp scale。
- `0`：重置 kp scale 为 1.0。

阀门主动旋转策略额外按键：

- `l`：按正方向启动一次阀门旋转轨迹。
- `;`：按反方向启动一次阀门旋转轨迹。

## 末端跟踪评测

末端跟踪评测用于回答一个前置问题：在不真正抓阀门的情况下，右臂末端是否能稳定跟踪 base 坐标系下给定的目标点。评测脚本会每 5 秒随机采样一个目标点，写入 IK 目标，并在 MuJoCo 中画出目标点和当前末端点。

可视化约定：

- 绿色点：采样得到的目标点。
- 红色点：当前末端代理点。
- 两点距离：记录到 CSV 的位置跟踪误差。

默认采样范围：

```bash
X_RANGE=0.20,0.42
Y_RANGE=-0.26,-0.04
Z_RANGE=-0.02,0.24
```

这些范围是机器人 base 坐标系下的右手可达区域，难度比最初版本更大，但仍然避免了明显不可达目标。

### 4DoF 假手跟踪

4DoF 版本复用早期无手腕 IK，适合和原始 FALCON 上肢控制方式对比。它跟踪的是一个虚拟末端点，不应作为最终阀门交互的末端定义。

一条命令自动运行：

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_auto.sh
```

需要分开观察 sim 和 policy 时：

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_sim.sh
```

另开一个终端：

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
AUTO_START_POLICY=1 \
DURATION_SEC=60 \
./launch_ee_tracking_policy.sh
```

默认输出：

- `/tmp/falcon_ee_tracking_markers.json`
- `/tmp/falcon_ee_tracking_metrics.csv`

### 7DoF 假手跟踪

7DoF 版本把右臂肩、肘、腕全部纳入 IK，只评测末端位置，不评测姿态。这是后续阀门轨迹跟踪更合理的基线。

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_auto.sh
```

默认输出：

- `/tmp/falcon_ee_tracking_7dof_markers.json`
- `/tmp/falcon_ee_tracking_7dof_metrics.csv`

### 7DoF 带手模型跟踪

这是当前最推荐的评测入口。它使用带真实手部几何和手部关节的 MuJoCo 模型，但仍然保持 FALCON 原始 29DoF 本体策略不变。右手在目标切换时打开，误差小于阈值后闭合，暂时只作为接近和开合逻辑验证，不进行真实抓取。

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_with_hand_auto.sh
```

默认输出：

- `/tmp/falcon_ee_tracking_7dof_with_hand_markers.json`
- `/tmp/falcon_ee_tracking_7dof_with_hand_metrics.csv`

使用当前跟踪优化配置时：

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
EXTRA_OVERLAY_CONFIG=config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml \
./launch_ee_tracking_7dof_with_hand_auto.sh
```

常用环境变量：

- `DURATION_SEC`：policy 侧运行总时长。
- `SAMPLE_PERIOD_SEC`：目标点重采样周期，默认 5 秒。
- `SEED`：随机采样种子。
- `X_RANGE` / `Y_RANGE` / `Z_RANGE`：目标点采样范围，格式为 `min,max`。
- `ELASTIC_LENGTH`：初始弹力绳长度，当前稳定值是 `-0.05`。
- `ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC`：policy 启动后多久松绳，默认 1 秒。
- `TRACKING_START_DELAY_SEC`：松绳稳定后多久开始随机目标测试。
- `HAND_CLOSE_ERROR_M`：带手版本中，右手闭合的误差阈值。
- `EXTRA_OVERLAY_CONFIG`：额外 yaml overlay。带手跟踪优化推荐使用 `config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml`。
- `TRACKING_IK_SPEED_FACTOR` / `TRACKING_IK_TRANS_WEIGHT` / `TRACKING_IK_REG_WEIGHT` / `TRACKING_IK_SMOOTH_WEIGHT`：临时覆盖 IK 调参。
- `TRACKING_UPPER_TAU_FF_SCALE` / `TRACKING_UPPER_TAU_FF_CLIP`：临时覆盖上肢前馈力矩比例和裁剪。
- `TRACKING_WRIST_KP_SCALE` / `TRACKING_WRIST_KD_SCALE`：临时缩放左右 wrist roll/pitch/yaw 的 PD 增益。

当前自动脚本的基本流程是：

1. 启动 MuJoCo。
2. 设置合适的弹力绳初始长度。
3. 启动 policy。
4. 自动触发 `]` 进入 policy 控制。
5. policy 稳定约 1 秒后关闭弹力绳。
6. 稳定后开始随机目标点跟踪。

### 定量结果和图表

跟踪数据会写入 CSV，可以用于画误差曲线、每目标点收敛统计、P90 / P95 分位误差和稳态误差分布。本地生成过的评测图和摘要默认放在 `sim2real/eval_outputs/`，该目录用于本地分析，默认不作为代码提交内容。

当前推荐分析脚本：

```bash
cd /home/lavine/project/FALCON
/home/lavine/miniconda3/envs/fcreal/bin/python \
  sim2real/tools/analyze_ee_tracking_metrics.py \
  /tmp/falcon_ee_tracking_7dof_with_hand_metrics.csv \
  --steady_after_s=2.0 \
  --report sim2real/eval_outputs/ee_tracking_with_hand_report.md \
  --plot sim2real/eval_outputs/ee_tracking_with_hand_report.png \
  --title 带手末端跟踪评测
```

CSV 中新增的误差分解字段：

- `ik_error_m`：目标点到 IK 参考末端点的距离。
- `servo_error_m`：实际末端点到 IK 参考末端点的距离。
- `current_speed_mps`：实际末端点速度，用于判断稳态晃动。

已记录的典型结果：

- 4DoF 假手长时间测试：整体均值约 2.92 cm，稳态均值约 1.33 cm，稳态 P95 约 2.00 cm。
- 7DoF 假手测试：整体均值约 3.25 cm，稳态均值约 1.62 cm，稳态 P95 约 2.27 cm。
- 7DoF 放大范围测试：整体均值约 4.19 cm，稳态均值约 1.77 cm，稳态 P95 约 2.58 cm。
- 7DoF 带手短回归：均值约 4.56 cm，RMS 约 5.54 cm，最大误差约 18.25 cm，最大值主要来自目标切换瞬态。
- 7DoF 带手基线定量评测：overall mean 约 5.10 cm，steady mean 约 2.48 cm，steady P95 约 4.08 cm。
- 7DoF 带手优化评测：overall mean 约 3.61 cm，steady mean 约 2.03 cm，steady P95 约 2.64 cm；稳态末端速度 P95 从 0.035 m/s 降到 0.005 m/s。

这里的 P90 / P95 表示误差分布的 90% / 95% 分位数。例如 P95 为 2.5 cm，表示 95% 的统计样本误差不超过 2.5 cm。

更完整的误差来源分析、复现实验命令和优化参数解释见 `EE_TRACKING_OPTIMIZATION_REPORT.md`。

## 带手模型说明

当前带手方案没有直接替换成外部完整 G1 模型，而是在 FALCON 原始 MuJoCo 模型上增加手部结构。这样做是为了尽量保持策略训练时依赖的动力学假设：

- 原始 29DoF 本体关节、freebase actuator、接触参数和场景设定保持一致。
- 手部 actuator 追加在原始 actuator 之后，避免改变 policy 输出到本体关节的映射。
- policy 输出仍然只写入原 29DoF actuator。
- 手部关节由 `LocoManipWithHandSimulator` 中的简单 PD 控制器独立控制。
- IK 使用仓库内的带手 URDF 和 mesh，不依赖外部 `unitree_ros` 路径。

主要文件：

- `sim2real/sim_env/loco_manip_with_hand.py`
- `sim2real/utils/arm_ik/robot_arm_ik_with_hand.py`
- `sim2real/config/g1/g1_29dof_with_hand_tracking_overlay.yaml`
- `humanoidverse/data/robots/g1/g1_29dof_old_freebase_with_hand.xml`
- `humanoidverse/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf`
- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand_tracking.xml`
- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand.xml`

末端定义上，7DoF 版本的 Pinocchio `R_ee` 和 MuJoCo `right_EE_frame` 已经按相同 offset 对齐。4DoF 版本的 `R_ee` 是不含腕关节时的虚拟末端点，因此只能用于简化对比，不适合作为最终阀门抓取点。

## 阀门交互与主动旋转 demo

阀门 demo 的目标是验证持续接触、受约束末端运动和全身平衡耦合。当前场景中已经包含：

- 阀门几何和 `valve_hinge` 关节。
- `right_hand_valve_anchor` 和 `valve_center_site`。
- 手-阀门 weld 固连，可运行时按 `J` 开关。
- 通过 `qfrc_applied[valve_dof]` 注入阀门主动扭矩。
- 阀门扭矩、角度、角速度的实时曲线窗口。
- `loco_manip_valve_turn.py` 中的主动单次旋转轨迹接口。

基础阀门场景使用：

- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase.xml`
- `humanoidverse/data/robots/g1/valve_body_include.xml`
- `sim2real/rl_policy/loco_manip/loco_manip_valve_turn.py`

带手阀门场景已经预留：

- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand.xml`

当前建议的阀门手动启动方式：

```bash
${PYTHON_BIN:-python} sim_env/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml
```

另开一个终端：

```bash
${PYTHON_BIN:-python} rl_policy/loco_manip/loco_manip_valve_turn.py \
  --config=config/g1/g1_29dof_falcon.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

推荐流程：

1. 先让机器人在 MuJoCo 中稳定站立。
2. 在 policy 终端按 `]` 启动策略。
3. 在 MuJoCo 中按 `9` 松开弹力绳。
4. 需要固连阀门时按 `J`。
5. 需要环境侧主动扭矩时按 `K`，并用 `U` / `I` / `O` 调整。
6. 需要策略侧主动旋转轨迹时，在 policy 终端按 `l` 或 `;`。

## 带手阀门角度闭环跟踪 baseline

这个测试用于验证“接触建立之后，机器人能否按照给定角度带动阀门转动”。当前它不是手指真实抓握，而是使用 MuJoCo equality `connect` 做软点连接：连接右掌心抓取点和阀门上的固定抓取点。这个设置能降低硬 `weld` 对控制结果的干扰，但仍然没有模拟摩擦、滑移、力闭合和手指包络。

控制逻辑：

1. 机器人接近阀门预抓取点。
2. 右手闭合。
3. 打开软 `connect`。
4. 根据阀门中心、轴线和当前 hinge angle 生成圆弧末端目标。
5. 用阀门角度闭环修正圆弧命令：`参考角 - 实际角` 越大，末端命令点越向目标方向提前。
6. 通过 7DoF IK 和原 FALCON policy 执行。

MuJoCo 可视化约定：

- 绿色点：当前末端目标点。
- 红色点：当前右掌心代理点。
- 蓝色点：阀门中心。
- 紫色线：本段最终目标角度方向。
- 黄色线和黄色圆弧：闭环命令角度方向和轨迹。
- 红色线：阀门实际角度方向。
- 窗口内文字：`target / ref / cmd / actual` 角度。

### 一条命令自动运行

推荐先用下面这条命令复现 1 分钟左右的多段正负角度测试：

```bash
cd /home/lavine/project/FALCON/sim2real

PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=70 \
ELASTIC_LENGTH=-0.05 \
VALVE_ANGLE_TARGETS_DEG=20,-60,35,-45,55,-30,40,-50 \
VALVE_ANGLE_SPEED_DEG=8 \
VALVE_ANGLE_HOLD_SEC=1.5 \
VALVE_ANGLE_CONTROL_MODE=closed_loop \
LOG_FILE=/tmp/falcon_valve_angle_tracking_closed_loop_1min.csv \
MARKER_FILE=/tmp/falcon_valve_angle_tracking_closed_loop_1min_markers.json \
SIM_STATUS_FILE=/tmp/falcon_valve_angle_tracking_closed_loop_1min_status.json \
SIM_CONTROL_FILE=/tmp/falcon_valve_angle_tracking_closed_loop_1min_control.json \
./launch_valve_angle_tracking_7dof_with_hand_auto.sh
```

自动脚本做的事情：

1. 启动带手阀门 MuJoCo 场景。
2. 设置弹力绳初始长度 `ELASTIC_LENGTH`。
3. 启动 policy 并自动进入 policy 控制。
4. policy 启动后延迟关闭弹力绳。
5. 进入靠近、闭手、软连接、角度闭环转动流程。
6. 写出 CSV、marker、status 和 control 文件。

按 `Ctrl+C` 退出时，auto 脚本会同时清理 MuJoCo 和 policy 后台进程。

默认输出：

- `/tmp/falcon_valve_angle_tracking_metrics.csv`
- `/tmp/falcon_valve_angle_tracking_markers.json`
- `/tmp/falcon_valve_angle_tracking_status.json`
- `/tmp/falcon_valve_angle_tracking_control.json`

如果按上面复现命令运行，输出会改成：

- `/tmp/falcon_valve_angle_tracking_closed_loop_1min.csv`
- `/tmp/falcon_valve_angle_tracking_closed_loop_1min_markers.json`
- `/tmp/falcon_valve_angle_tracking_closed_loop_1min_status.json`
- `/tmp/falcon_valve_angle_tracking_closed_loop_1min_control.json`

### 分开启动 sim 和 policy

如果需要手动观察或调试，可以分两个终端运行。

终端 1：

```bash
cd /home/lavine/project/FALCON/sim2real

PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
ELASTIC_LENGTH=-0.05 \
./launch_valve_angle_tracking_7dof_with_hand_sim.sh
```

终端 2：

```bash
cd /home/lavine/project/FALCON/sim2real

PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
AUTO_START_POLICY=1 \
DURATION_SEC=70 \
VALVE_ANGLE_TARGETS_DEG=20,-60,35,-45,55,-30,40,-50 \
VALVE_ANGLE_SPEED_DEG=8 \
VALVE_ANGLE_CONTROL_MODE=closed_loop \
./launch_valve_angle_tracking_7dof_with_hand_policy.sh
```

### 常用参数

运行流程参数：

- `DURATION_SEC`：policy 总运行时长。多段角度目标建议留足时间，否则最后一段可能被截断。
- `ELASTIC_LENGTH`：初始弹力绳长度。当前稳定值是 `-0.05`。
- `ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC`：policy 启动后多久松绳，默认 1 秒。
- `TRACKING_START_DELAY_SEC`：松绳后多久开始任务状态机，默认 2 秒。
- `SIM_STARTUP_WAIT_SEC`：auto 脚本启动 sim 后等待多久再启动 policy。
- `SEED`：随机角度目标采样种子。

角度目标参数：

- `VALVE_ANGLE_TARGETS_DEG`：显式角度序列，逗号分隔，例如 `20,-60,35`。设置它时不会随机采样。
- `VALVE_ANGLE_RANDOM_COUNT`：未设置显式目标时，随机目标段数。
- `VALVE_ANGLE_MIN_ABS_DEG` / `VALVE_ANGLE_MAX_ABS_DEG`：随机目标的绝对角度范围。
- `VALVE_ANGLE_SPEED_DEG`：参考角速度，单位 deg/s。
- `VALVE_ANGLE_HOLD_SEC`：每段到达目标后的保持时间。

角度闭环参数：

- `VALVE_ANGLE_CONTROL_MODE`：`closed_loop` 或 `timed`。`closed_loop` 会用阀门实际角度反馈修正命令；`timed` 是原来的时间开环圆弧。
- `VALVE_ANGLE_FEEDBACK_GAIN`：角度误差反馈增益，默认 `0.8`。增大后追赶更积极，但可能增加末端摆动。
- `VALVE_ANGLE_COMMAND_LEAD_DEG`：闭环命令相对参考角允许提前的最大角度，默认 `20 deg`。
- `VALVE_ANGLE_COMMAND_SPEED_DEG`：闭环命令角本身的最大变化速度，默认 `18 deg/s`。
- `VALVE_ANGLE_EXTRA_SETTLE_SEC`：闭环模式下每段额外允许的收敛时间，默认 `2 s`。
- `VALVE_ANGLE_TOLERANCE_DEG`：判断角度接近目标的阈值，默认 `2 deg`。

阀门和连接参数：

- `VALVE_ATTACHMENT_MODE`：`connect` 或 `none`。角度跟踪 baseline 默认使用 `connect`。
- `VALVE_JOINT_DAMPING`：阀门 hinge damping。
- `VALVE_JOINT_FRICTIONLOSS`：阀门 hinge frictionloss。

IK 和上肢执行参数：

- `TRACKING_IK_SPEED_FACTOR`
- `TRACKING_IK_TRANS_WEIGHT`
- `TRACKING_IK_REG_WEIGHT`
- `TRACKING_IK_SMOOTH_WEIGHT`
- `TRACKING_IK_FILTER_WEIGHTS`
- `TRACKING_UPPER_TAU_FF_SCALE`
- `TRACKING_UPPER_TAU_FF_CLIP`
- `TRACKING_WRIST_KP_SCALE`
- `TRACKING_WRIST_KD_SCALE`

这些参数沿用末端跟踪优化流程。一般先不要同时大幅改多个参数，建议一次只改角度闭环或阀门动力学中的一类。

### 角度跟踪结果分析

分析 CSV 并生成报告和曲线：

```bash
cd /home/lavine/project/FALCON

/home/lavine/miniconda3/envs/fcreal/bin/python \
  sim2real/tools/analyze_valve_angle_tracking_metrics.py \
  /tmp/falcon_valve_angle_tracking_closed_loop_1min.csv \
  --report sim2real/eval_outputs/valve_angle_tracking_closed_loop_1min_report.md \
  --plot sim2real/eval_outputs/valve_angle_tracking_closed_loop_1min_plot.png
```

CSV 关键字段：

- `segment_target_delta_deg`：本段目标转角。
- `desired_delta_deg`：参考轨迹角度。
- `command_delta_deg`：闭环后实际发送给末端圆弧目标的命令角。
- `actual_delta_deg`：阀门实际转角。
- `angle_error_deg`：参考角和实际角的误差。
- `final_angle_error_deg`：本段最终目标角和实际角的误差。
- `ee_error_m`：右掌心代理点到当前末端目标点的距离。
- `connect_stretch_m`：软连接两端的距离，可理解为软连接被拉开的程度。

### 组会图表和视频素材

新增脚本 `sim2real/tools/make_valve_group_meeting_assets.py` 可以从角度跟踪 CSV 生成组会用材料，包括方法示意图、总览图、MP4、GIF 和中文汇报文档。

```bash
cd /home/lavine/project/FALCON

MPLCONFIGDIR=/tmp/matplotlib \
/home/lavine/miniconda3/envs/fcreal/bin/python \
  sim2real/tools/make_valve_group_meeting_assets.py \
  --csv /tmp/falcon_valve_angle_tracking_closed_loop_1min.csv \
  --output_dir sim2real/eval_outputs/group_meeting_valve_closed_loop \
  --gif_frames 80
```

输出目录：

- `sim2real/eval_outputs/group_meeting_valve_closed_loop/group_meeting_valve_closed_loop_report.md`
- `sim2real/eval_outputs/group_meeting_valve_closed_loop/valve_closed_loop_method.png`
- `sim2real/eval_outputs/group_meeting_valve_closed_loop/valve_closed_loop_summary.png`
- `sim2real/eval_outputs/group_meeting_valve_closed_loop/valve_closed_loop_tracking_animation.mp4`
- `sim2real/eval_outputs/group_meeting_valve_closed_loop/valve_closed_loop_tracking_animation.gif`

当前本地没有飞书连接器或飞书 token，不能直接上传飞书。可以把上面这些文件手动上传到飞书文档；如果之后配置了飞书开放平台 token 或 webhook，再改成自动发布。

## 带手阀门真实接触抓握 baseline

这个测试用于验证右手三指夹爪能否在不使用 `connect/weld` 的情况下，通过真实接触夹住阀门轮盘。它使用新的整体加粗阀门场景：

- `humanoidverse/data/robots/g1/valve_thick_body_include.xml`
- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand_valve_grasp_contact.xml`

新版阀门保留原来的轮盘、辐条、中心 hub 和 hinge 拓扑，不再添加局部抓取件。轮圈半径增大到 `0.16 m`，轮圈管径增大到约 `48 mm`，辐条和 hub 也同步加粗，目的是先验证“手指真实接触能不能夹住一个合理粗细的阀门轮盘”，而不是被过细轮圈卡住。

policy 入口 `loco_manip_valve_grasp_contact_7dof_with_hand.py` 是独立状态机，不复用旧版 valve demo / valve task 的转动逻辑。

推荐自动运行：

```bash
cd /home/lavine/project/FALCON/sim2real

PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=35 \
ELASTIC_LENGTH=-0.05 \
LOG_FILE=/tmp/falcon_valve_grasp_contact_metrics.csv \
MARKER_FILE=/tmp/falcon_valve_grasp_contact_markers.json \
SIM_STATUS_FILE=/tmp/falcon_valve_grasp_contact_status.json \
SIM_CONTROL_FILE=/tmp/falcon_valve_grasp_contact_control.json \
./launch_valve_grasp_contact_7dof_with_hand_auto.sh
```

这个脚本会启动 MuJoCo、自动启动 policy、松开弹力绳，然后执行：

1. 移动到预抓取点。
2. 启用右手姿态约束，让手的前向轴对准阀门轴，手内 y 轴对准阀门半径方向。
3. 接近加粗轮圈上的抓取点。
4. 关闭右手。
5. 保持抓握并记录真实接触。

MuJoCo 可视化：

- 默认使用 `ee_marker_visual_mode: grasp_minimal`，只显示绿色目标点、红色当前末端点、黄色抓握工作点和状态文字。
- 如需检查几何定义，可在 `g1_29dof_with_hand_valve_grasp_contact_overlay.yaml` 中临时改成 `ee_marker_visual_mode: grasp_debug`，此时会额外显示阀门中心线、抓握坐标轴和虚拟指尖目标。
- 默认相机也写在同一个 overlay 的 `viewer_camera` 下；如果打开窗口后视角仍不顺手，优先微调 `lookat`、`distance`、`azimuth`、`elevation`。

CSV 关键字段：

- `contact_count`：右手手指和阀门轮盘之间的接触点数量。
- `contact_normal_force`：这些接触点的法向力总和。
- `grasp_success`：是否满足 `contact_count >= valve_grasp_success_min_contacts` 且 `contact_normal_force >= valve_grasp_success_min_force_n`。
- `grasp_error_m`：右手代理点到阀门抓取 site 的距离。
- `attach_enabled`：应为 `0`，表示没有使用 equality 连接。

常用参数：

- `HAND_KP` / `HAND_KD`：手指开合 PD 增益。
- `VALVE_PREGRASP_OFFSET_M`：预抓取点沿阀门轴线外移距离。
- `VALVE_PREGRASP_ERROR_M`：预抓取阶段的误差阈值。
- `VALVE_PREGRASP_MAX_S`：预抓取最多等待时间；若末端存在稳定误差，会自动进入接近阶段。
- `VALVE_GRASP_ERROR_M`：进入闭手阶段的距离阈值。
- `VALVE_CLOSE_WAIT_S`：闭手后等待接触建立的时间。
- `VALVE_HOLD_S`：抓住后保持观察的时间。
- `VALVE_GRASP_SUCCESS_MIN_CONTACTS`：判断抓握成功所需最少接触点。
- `VALVE_GRASP_SUCCESS_MIN_FORCE_N`：判断抓握成功所需最小接触法向力。
- `VALVE_GRASP_TARGET_BIAS_BASE_M`：抓取目标点在 base 坐标系下的微调量，格式如 `0.0,0.0,0.01`。
- `VALVE_GRASP_POINT_OFFSET_EE_M`：闭手后轮圈中心相对 `R_ee` 的期望位置，格式如 `-0.03,0.06,0.0`。默认值来自当前三指手闭合几何的粗略标定。

当前抓握 baseline 默认冻结第一次读到的轮盘抓取点，不跟随阀门被轻微碰撞后的被动转动。这是为了先验证“接近同一个物理抓取点并闭手”的能力，后续进入主动转动阶段再切回阀门角度闭环。

如果这个 baseline 能稳定产生真实接触，下一步再在同一场景里加入小角度真实接触转动，不再依赖 `connect`。

旧版 `README_valve_demo.md` 中提到过 `launch_valve_demo_auto_v2.sh`，但当前仓库没有保留这个脚本。后续如果要恢复阀门自动化，建议参考现有 `launch_ee_tracking_*_auto.sh` 的结构重新写一份，不要直接依赖旧文档中的脚本名。

## 带手阀门接触转动 baseline

这个测试在真实接触抓握 baseline 上加入小角度阀门转动，不使用永久 `weld`。当前默认策略是：先要求掌心/手指和阀门形成真实接触，再启用柔顺 `connect` 作为“抓住后的传力近似”；到目标角附近进入 `turn_hold` 后释放 `connect`，并开启有限力矩角度制动，避免 MuJoCo 中阀门靠残余速度继续过冲。

这不是为了在 MuJoCo 里完美复现真实摩擦抓握，而是为了验证后续实机需要的控制链路：接近预设抓点、闭合夹爪、沿阀门圆弧生成末端目标、用阀门角度反馈停止在目标附近。当前默认目标为 `10 deg`。

推荐自动运行：

```bash
cd /home/lavine/project/FALCON
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh
```

默认输出：

- CSV 日志：`artifacts/demo_logs/valve_contact_turning_YYYYMMDD_HHMMSS.csv`
- MuJoCo marker：`/tmp/falcon_valve_contact_turning_markers_YYYYMMDD_HHMMSS.json`

默认流程：

1. 自动打开带手 G1 和加粗阀门场景。
2. 自动启动 policy、释放弹力绳。
3. 右手移动到预抓取点并接近轮圈抓点。
4. 掌心/手指接触后缓慢闭合右手。
5. 抓握拓扑、接触力、相对滑移和抓点误差满足阈值后，打开柔顺 `connect` 辅助传力。
6. 进入转动前进行短暂预稳定，避免释放阀门角度锁时出现角速度跳变。
7. 用阀门角度闭环修正末端圆弧命令，完成小角度转动。
8. 到目标附近后释放 `connect`，并用有限力矩角度制动吸收残余角速度。

当前默认参数位于：

```text
sim2real/config/g1/g1_29dof_with_hand_valve_contact_turning_overlay.yaml
```

常用参数：

- `valve_turn_target_deg`：目标转角，当前默认 `10.0`。
- `valve_turn_speed_deg`：参考角速度，当前默认 `1.5 deg/s`。这个速度比早期 `0.8 deg/s` 更快，目的是缩短持续接触时间，减少身体被阀门慢慢拉近。
- `valve_turn_feedback_gain`：阀门角度闭环增益。
- `valve_turn_command_lead_deg`：允许末端命令相对参考角提前的最大角度。
- `valve_turn_command_speed_deg`：闭环命令角最大变化速度，当前默认 `5.0 deg/s`。
- `valve_turn_radius_mode`：末端圆弧命令半径模式，当前默认 `target`，按阀门真实抓点半径生成圆弧。
- `valve_turn_tolerance_deg`：实际角进入目标邻域的阈值，当前默认 `1.2 deg`。
- `valve_turn_abort_min_base_to_valve_x_m`：base 到阀门过近时判失败，当前默认 `0.30 m`。
- `valve_turn_abort_max_base_approach_m`：转动开始后 base 被拉近过多时判失败，当前默认 `0.12 m`。
- `valve_turn_abort_max_base_yaw_drift_deg`：转动开始后 base yaw 漂移过大时判失败，当前默认 `20 deg`。
- `valve_turn_contact_compensation_enabled`：是否启用小幅接触保持补偿，当前默认开启。
- `valve_turn_finish_on_target_reached`：实际阀门角进入目标邻域后是否提前进入 hold，当前默认开启。
- `valve_contact_assist_enabled`：真实接触后是否启用柔顺 `connect` 传力辅助，当前默认开启。
- `valve_contact_assist_release_on_turn_hold`：进入目标保持阶段是否释放 `connect`，当前默认开启。
- `valve_turn_hold_angle_brake_enabled`：进入目标保持阶段是否启用有限力矩角度制动，当前默认开启。该项是 MuJoCo demo 近似，实机部署时应由视觉/编码器角度反馈和末端停止保持逻辑替代。
- `valve_turn_pre_settle_s`：进入转动前的短暂稳定时间。
- `valve_joint_damping` / `valve_joint_frictionloss`：阀门 hinge 阻尼和静摩擦。
- `hand_command_ramp_s`：右手闭合斜坡时间。
- `hand_kp` / `hand_kd` / `right_hand_effort_limits`：手指开合 PD 和力矩限幅。
- `hand_contact_hold_force_cap_n`：contact-hold 后的法向力上限。

临时覆盖目标角的例子：

```bash
cd /home/lavine/project/FALCON
VALVE_TURN_TARGET_DEG=20 \
DURATION_SEC=55 \
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh
```

严格单目标工作区间测试：

```bash
cd /home/lavine/project/FALCON
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict30
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict45
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict60
```

`strictXX` preset 不修改默认展示序列，只用于评估单次转动能力边界。它会启用：

- 单目标转角 `XX deg`，当前支持 `10/20/30/45/60/75/90/120`。
- 较慢参考速度 `0.9 deg/s`。这是当前 45 度区间内角度精度和抓握滑移之间较好的折中。
- 到达目标后继续保持，要求角度误差、阀门角速度和抓握质量同时稳定。
- 稳定三点包络判据：掌心 + 拇指 + 至少一根手指；不强制食指和中指同时持续接触。
- 最终接触健康比例不低于 `0.75`，相对滑移不超过 `0.12 m`。
- 默认仍在 `turn_hold` 阶段释放柔顺 `connect`，用真实接触和 angle brake 检查最终保持质量；如需诊断可用 `VALVE_CONTACT_ASSIST_RELEASE_ON_TURN_HOLD=0` 临时覆盖。

CSV 关键字段：

- `turn_reference_delta_deg`：参考转角。
- `turn_command_delta_deg`：闭环修正后的末端圆弧命令角。
- `turn_actual_delta_deg`：阀门实际转角。
- `turn_angle_error_deg`：目标或参考角与实际角的误差。
- `turn_motion_radius_m`：本轮转动实际用于生成末端切向位移的运动半径。
- `contact_topology`：`P/T/I/M` 分别表示掌心、拇指、食指、中指是否与阀门接触。
- `contact_normal_force`：手-阀门接触法向力总和。
- `relative_slip_m`：抓住后，抓点在阀门局部坐标系下的相对滑移。
- `base_roll_deg` / `base_pitch_deg` / `base_yaw_deg`：机身姿态稳定性指标。
- `base_to_valve_x` / `base_to_valve_y`：base 到阀门中心的相对站位，用于判断机器人是否被阀门拉近或拉偏。

生成定量报告和图表：

```bash
cd /home/lavine/project/FALCON
MPLCONFIGDIR=/tmp/matplotlib /home/lavine/miniconda3/envs/fcreal/bin/python \
  sim2real/tools/analyze_valve_contact_turning_metrics.py \
  artifacts/demo_logs/valve_contact_turning_YYYYMMDD_HHMMSS.csv \
  --report artifacts/demo_logs/valve_contact_turning_report.md \
  --plot artifacts/demo_logs/valve_contact_turning_report.png
```

当前已复现的结果：

- `artifacts/demo_logs/valve_contact_turning_20260426_211334.csv`：目标 `5 deg`，最终 `4.23 deg`，最终误差 `0.77 deg`。
- `artifacts/demo_logs/valve_contact_turning_20260426_211432.csv`：目标 `5 deg`，最终 `5.10 deg`，最终误差 `-0.10 deg`。
- `artifacts/demo_logs/valve_contact_turning_20260426_234055.csv`：目标 `10 deg`，最终 `9.27 deg`，最终误差 `0.73 deg`。
- `artifacts/demo_logs/valve_contact_turning_20260427_073819.csv`：目标 `10 deg`，使用柔顺 `connect` + hold 阶段角度制动。`turn_hold` 末尾实际约 `12.83 deg`，后续 `done` 阶段收敛到约 `11.26 deg`，最终误差约 `-1.26 deg`。
- `artifacts/demo_logs/valve_contact_turning_20260427_080148.csv`：目标 `10 deg`，加入姿态/站位失败判据并把速度提高到 `1.5 deg/s`。`turn_hold` 末尾实际约 `9.74 deg`，整次运行末尾实际约 `10.20 deg`；转动/保持阶段 base 靠近约 `6.4 cm`，yaw 漂移约 `4.8 deg`，未触发稳定性失败判据。
- `artifacts/demo_logs/valve_contact_turning_20260427_173134.csv`：`strict30`，目标 `30 deg`，最终约 `30.43 deg`，误差约 `-0.43 deg`，最终抓握成立，最终滑移约 `4.4 mm`。
- `artifacts/demo_logs/valve_contact_turning_20260427_173802.csv`：`strict45`，目标 `45 deg`，最终约 `44.62 deg`，误差约 `0.38 deg`，最终抓握成立，最终滑移约 `1.7 cm`。
- `artifacts/demo_logs/valve_contact_turning_20260427_182841.csv`：`strict45`，速度 `0.9 deg/s`，目标 `45 deg`，最终约 `44.86 deg`，误差约 `0.14 deg`，最终抓握成立，滑移 p90 约 `2.6 cm`。
- `artifacts/demo_logs/valve_contact_turning_20260427_183028.csv`：更新默认 `strict45` preset 后复测，目标 `45 deg`，最终约 `45.61 deg`，误差约 `-0.61 deg`，最终抓握成立，滑移 p90 约 `3.2 cm`。
- `artifacts/demo_logs/valve_contact_turning_20260427_173908.csv`：`strict60`，目标角度可到达，但最终接触健康降到 `0.50` 且 `final_grasp_success=false`，按严格保持标准判失败。
- `artifacts/demo_logs/valve_contact_turning_strict_single_turn_090deg_entry50_20260427_173000.csv`：`strict90`，角度可到达，但最终抓握丢失、滑移约 `16 cm`，判失败。
- 稳定性报告：`artifacts/demo_logs/valve_contact_turning_stability_guard_10deg_report.md`。
- 稳定性图表：`artifacts/demo_logs/valve_contact_turning_stability_guard_10deg_report.png`。
- 对比图：`artifacts/demo_logs/valve_contact_turning_contact_demo_summary_20260426_211334_211432.png`。
- 10 度报告：`artifacts/demo_logs/valve_contact_turning_10deg_best_report.md`。
- 10 度图表：`artifacts/demo_logs/valve_contact_turning_10deg_best_report.png`。
- 阶段性报告：`docs/worklogs/valve_turning_contact_demo.md`。

当前局限：

- 当前严格单次工作区间初步验证到 `45 deg`；`60 deg` 和 `90 deg` 虽然角度能到，但 hold 阶段抓握质量不足，不应作为稳定 demo。
- 受手臂构型和闭链约束影响，单次转动不宜盲目追求大角度；更大的阀门角度应采用“转一段、重置抓点、再转一段”的分段策略。
- 转动后半段接触力可能下降，接触拓扑会从 `PTIM` 退化为局部接触。
- 当前把 base 靠近和 yaw 漂移记录为质量指标，只有明显摔倒、撞阀门、手脱离或接触/滑移失控才判失败；以后扩大目标角时，仍应同时报告角度误差、base 漂移和抓握健康度。
- 没有使用永久 `weld`，但当前默认启用了柔顺 `connect` 和 hold 阶段角度制动。它们是仿真 demo 的工程近似，不应表述为纯真实接触抓握。
- 手指接触几何、摩擦和阀门物理参数仍经过 demo 调整，后续需要继续向真实硬件参数收敛。
- 当前环境没有可直接调用的 `ffmpeg`，本轮没有自动生成视频；需要录视频时先用桌面录屏工具录 MuJoCo 窗口，保存到 `artifacts/demo_videos/`。

## sim2real 注意事项

真实机器人部署前必须先完成 sim2sim 验证。建议遵守以下顺序：

1. 在 MuJoCo 中确认策略能稳定站立、行走和执行上肢目标。
2. 检查 `INTERFACE`、`DOMAIN_ID`、SDK 类型和机器人型号是否正确。
3. 从保守的 kp / kd 或 scale 设置开始。
4. 确认机器人脚已经稳定接触地面后再启用 policy。
5. 保持急停可用：键盘 `o` 可停止 policy action，手柄急停按键按原 FALCON 协议执行。

需要实时 onboard 推理时，纯 Python 的 `unitree_sdk2_python` 可能无法保证 Jetson Orin 上的实时性。FALCON 原作者建议使用 C++ 后端加 pybinding 的 `unitree_sdk2` 方案。

## 视觉模块接入准备

当前阀门任务仍使用 MuJoCo 提供的特权信息读取阀门中心、轴线、抓点和角度。下一阶段计划在独立分支中接入仿真视觉模块，目标不是立即替换整条控制链路，而是先把“感知输出”做成和现有 privileged 接口一致的数据结构，再逐步替换来源。

建议新分支名称：

```bash
feature-valve-vision-module
```

第一阶段建议保持以下边界：

- 不改动已跑通的阀门接触转动 baseline。
- 不改动现有阀门 XML 和手部控制器。
- 新增视觉模块时，先输出阀门 `center / axis / grasp_point / angle` 的估计值。
- 控制侧继续消费统一的阀门状态接口，避免把视觉算法和 IK / valve turning 状态机直接耦合。
- 仿真阶段可以先使用 MuJoCo 内置相机或渲染图像做验证，但所有使用 privileged ground truth 的地方需要明确标注。

推荐接入顺序：

1. 只读梳理当前阀门状态来源，确认哪些字段来自 MuJoCo site、body、joint。
2. 新增视觉数据结构和日志，不改变控制输入。
3. 新增仿真相机读取和图像保存脚本，先离线验证阀门检测。
4. 将视觉估计结果写入与现有 `SIM_STATUS_FILE` 类似的中间状态文件。
5. 在受控 demo 中切换阀门状态来源：`privileged` / `vision_debug` / `vision`。

相关准备记录见 `docs/worklogs/vision_module_branch_prep_2026-05-01.md`。

## 2026-05-01 阀门视觉与转动验证工作日志

本日工作围绕“让视觉几何先安全接管 approach，并用阀门角度完成度而不是 MuJoCo 严格接触拓扑作为主任务指标”展开。默认 GT baseline 仍保持为保守路径；所有新策略都通过本地 overlay 或显式配置启用，不作为全局默认。

### 已完成的接口和视觉链路

- Provider 层加入 `valve_geometry_source: gt / vision_overlay_debug / vision_overlay`，默认仍为 `gt`，不包 overlay、不写 debug、不附加控制字段。
- `vision_overlay_debug` 只做误差对比，返回控制器的仍是纯 GT geom；`vision_overlay` 只有在 `vision_override_control=true` 时才可能接管。
- 新增 `vision_override_angle=false` 默认保护，视觉角度只进入 debug 字段，不覆盖 turn closed-loop 使用的 `valve_angle / valve_vel`。
- 视觉抓点语义收敛为 `vision_grasp_point_semantics: raw_site`，让原控制器继续执行 `valve_grasp_radial_inset_m=0.026`；默认不使用视觉输出的 `grasp_R_base`，即 `vision_use_grasp_R_base=false`。
- close latch 修正为冻结 approach 阶段最近一次实际用于控制的视觉 geom，并支持 `vision_close_latch_frame: world`，避免 base frame 过期导致 close-entry 几何漂移。
- 仿真 RGB-D 彩色弱标识视觉模块已经能输出 `/tmp/falcon_valve_vision_status.json`，包含 `pose_valid / grasp_valid / grasp_latched / angle_valid / plane_aux_valid`。新增 `plane_aux` marker 后，连续 debug 中 pose/axis 稳定性明显提升。

### 抓取判据和任务成功标准的收敛

- `_grasp_success()` 语义保持不变，仍表示 MuJoCo 内部严格接触拓扑诊断，例如 `PTIM`。
- 实验发现 vision-latched 几何下常出现 `-TIM`、`P-IM`、`PT--` 等非完整拓扑，但仍可能带动阀门转动；因此 strict PTIM 不再适合作为流程唯一阻塞条件。
- 保留 `strict_grasp_success`、`contact_topology`、`real_contact_topology`、palm/thumb/index/middle force/count 作为 debug 字段。
- 当前推荐实验评价改为 turn-validation：close 阶段负责闭手和建立接触，不再因 strict PTIM timeout 直接判死；最终任务成功由阀门实际角度误差和安全指标判断。
- 本地 turn-validation overlay 使用：

```yaml
valve_enter_hold_on_close_timeout: true
valve_turn_entry_require_success: false
valve_turn_entry_min_contact_count: 1
valve_turn_entry_min_contact_force_n: 1.0
valve_turn_entry_max_contact_force_n: 200.0
valve_turn_entry_max_slip_m: 0.30
valve_turn_completion_require_grasp: false
valve_turn_completion_min_contact_health_ratio: 0.0
valve_turn_completion_timeout_is_failure: false
valve_angle_success_tolerance_deg: 2.0
```

其中 `valve_turn_completion_max_slip_m: 0.0` 在当前代码中表示禁用 completion 阶段 slip 质量门控；turn 过程仍保留 `valve_turn_abort_slip_m` 等安全中止条件。

### 接触力诊断和 contact compensation 结论

- 对 GT baseline 做 contact-only / turn 诊断后确认：机器人转阀门时身体靠近面板、挪脚不是视觉问题；GT 下也存在明显 axis/radial 非切向接触力。
- 旧的 base-frame 3D contact compensation 会把 axis/radial/tangent 误差一起硬补偿，是机器人被非切向力拖向阀门的重要来源。
- 完全关闭 compensation 会导致转动失败，说明补偿本身仍有必要。
- 当前最有用的实验配置是只保留阀门切向补偿：

```yaml
valve_contact_compensation_frame: "valve"
valve_contact_compensation_axis_scale: 0.0
valve_contact_compensation_radial_scale: 0.0
valve_contact_compensation_tangent_scale: 1.0
```

这组配置显著降低 foot displacement 和 base-to-valve approach，同时仍能驱动阀门转动。它目前作为“contact-turning 推荐实验配置”，但尚未写入全局默认。

### 今日关键短测结果

GT geometry + turn-validation + tangent-only compensation，`+10 deg` 五次短测：

- `entered_turn_rate = 5/5`
- `angle_success_rate = 5/5`，成功定义为 `abs(final_valve_angle_deg - target_deg) <= 2 deg`
- 平均 final angle error 约 `0.43 deg`，最大约 `0.89 deg`
- 平均 foot displacement 约 `0.0086 m`，最大约 `0.0107 m`
- 平均 base approach 约 `0.0154 m`，最大约 `0.0220 m`
- run 03 / run 05 最终 `strict_grasp_success=0` 仍完成角度任务，说明 strict PTIM 应作为 debug 指标，而不是流程阻塞条件。

Vision overlay approach + turn-validation + tangent-only compensation，`+10 deg` 五条有效 vision-latched 样本：

- `entered_turn_rate = 5/5`
- `angle_success_rate = 3/5`
- 平均 final angle error 约 `1.74 deg`，最大约 `2.97 deg`
- 平均 foot displacement 约 `0.0070 m`，最大约 `0.0138 m`
- 平均 base approach 约 `0.0110 m`，最大约 `0.0184 m`
- 平均 slip 约 `0.073 m`，最大约 `0.083 m`
- `geometry_source_used` 分布包含 approach 阶段 `vision_overlay` 和 close/turn 阶段 `vision_latched`；`latched_geometry_source=vision_overlay`。
- 另有一条样本因视觉质量门控失败全程 fallback 到 GT，不计入 vision-latched 统计；主要原因包括 `vision_invalid`、`bad_quality:plane_fit_error_m`、`bad_quality:spoke_num_points`。

这说明当前主线已经基本闭合：视觉几何可以接管 approach，close 不再被 strict PTIM 卡死，tangent-only compensation 能降低非切向拖拽，任务主指标可以转向阀门角度完成度。剩余问题主要是 turn angle tracking / 接触驱动力一致性，而不是视觉接口、marker 或 strict grasp gate。

### 今日遇到的问题和处理方法

1. `vision_overlay_debug` 只能对比、不能影响控制。
   - 问题：如果 debug 模式把视觉 `center/grasp/axis/grasp_R/wheel_xmat/valve_angle` 混入返回 geom，会悄悄改变控制器行为。
   - 处理：`valve_geometry_source=gt` 仍直接走 `SimStatusGraspGeometryProvider`；`vision_overlay_debug` 只返回 GT geom 加不影响控制的 debug metadata；只有 `vision_overlay + vision_override_control=true` 才允许 overlay geom 进入控制。

2. 视觉角度不能提前接管 turn closed-loop。
   - 问题：彩色辐条角度还只是 debug 估计，直接覆盖 `valve_angle/valve_vel` 会污染 turn 反馈。
   - 处理：加入 `vision_override_angle=false` 默认保护；视觉角度写入 `vision_valve_angle / vision_valve_vel / vision_angle_valid`，控制用角度仍来自 GT。

3. green grasp marker 被右手遮挡后，pose/axis 估计会被拖垮。
   - 问题：原始 plane fitting 依赖 center + spoke + grasp，close/pre-turn 阶段 grasp marker 被手遮挡，导致 `grasp_valid` 下降并影响平面法向。
   - 处理：新增不参与抓取的 `plane_aux` 视觉贴片，把有效性拆成 `pose_valid / grasp_valid / angle_valid / plane_aux_valid`；plane fit 改为优先使用 center + spoke + plane_aux，grasp 可见时才参与。grasp 点支持 last-valid latch，只用于 debug status 连续性。

4. plane_aux 贴片不能悬空，也不能和原有颜色冲突。
   - 问题：最初新增的辅助贴片位置和粉色辐条前表面没有完全对齐，并且颜色可能与 MuJoCo 地面视觉上混淆。
   - 处理：改成模仿粉色辐条的薄片式 marker，贴在相邻辐条片上，保持 visual-only，不改变质量、惯量、碰撞、摩擦或 weld/contact 行为；颜色选择高饱和且与已有 red/yellow/cyan/magenta/green 区分。

5. vision approach 成功后，close latch 一度拿到 GT 而不是最后一帧 vision geom。
   - 问题：`vision_close_latch_source=current_control` 如果在 close transition frame 重新读 provider，当前状态已经不允许实时 vision，于是 fallback 到 GT，导致 `latched_geometry_source=gt`，无法验证 vision-latched close。
   - 处理：缓存 `_last_accepted_vision_control_geom`，close latch 优先冻结 approach 阶段最近一次真正用于控制的 vision geom，并记录 `latched_from_last_accepted_vision_control_geom`、`last_accepted_vision_control_age_s` 等字段。

6. base-frame latch 会过期。
   - 问题：机器人 base 在 approach/close 期间仍会移动，旧的 base-frame latched geom 到 close-entry 时可能与 GT effective grasp 相差接近 `9.5 cm`、姿态差约 `32 deg`。
   - 处理：实现 `vision_close_latch_frame=world`。接受 vision geom 时同时保存 world-frame 几何，close latch 时用当前 `world_to_base` 转回 base frame。修正后 close-entry delta 收敛到毫米级和约 `3 deg`。

7. vision grasp 点语义和 GT baseline 不一致。
   - 问题：green marker 更接近原始 `right_hand_valve_site`，但 vision JSON 曾写 `grasp_base_is_effective=true`，导致 `_valve_grasp_point_base()` 跳过 `26 mm` radial inset，close 发生 `P-IM` 缺 thumb。
   - 处理：加入 `vision_grasp_point_semantics: raw_site`，Provider 强制 `grasp_base_is_effective=false`，复用原控制器 radial inset 和 EE offset；默认 `vision_use_grasp_R_base=false`，姿态也由原 `_right_grasp_rotation_base()` 计算。

8. strict PTIM close gate 过于敏感。
   - 问题：vision-latched 几何已经能产生接触和力，但只要缺 palm/thumb/index/middle 中任一角色，就会 `close_stage_timeout`，拿不到后续“能不能实际转动阀门”的数据。
   - 处理：先实现并测试 `functional_probe`，证明 `-TIM` 非严格拓扑也能小角度带动阀门；随后进一步收敛到 turn-validation：strict PTIM 保留为 debug，不再作为进入 turn 的唯一硬门槛。

9. functional probe 候选条件一开始太保守。
   - 问题：G2 中 `-TIM` 接触、force 约 `7.999 N`、close progress 约 `0.64`，但候选 gate 因 force/progress 边界没有启动 probe。
   - 处理：将 allowed topology 和 block reason 显式日志化，放宽 functional 模式下的候选阈值；验证到 `-TIM` 可以通过 probe 并继续完成 `+10 deg`。后续不再把 probe 作为主流程，只保留这个结论支撑 turn-validation。

10. 机器人转阀门时会被拖向面板、脚步移动。
    - 问题：最初怀疑是视觉或 functional_probe，但 GT baseline contact-only 不主动转阀门时也出现 foot displacement 和很大的 axis/radial 非切向力。
    - 处理：新增 foot displacement、base-to-valve distance、valve axis/radial/tangent basis、EE error 分解、hand-valve force 分解等日志。诊断确认旧 base-frame 3D contact compensation 是重要注入源。

11. 不能简单关闭 compensation。
    - 问题：`valve_contact_tracking_compensation_enabled=false` 虽然减少某些硬推目标，但转动容易失败，说明补偿仍是驱动接触转动的一部分。
    - 处理：把 compensation 分解到 valve frame，只保留 tangent 分量，关闭 axis/radial 分量。C1 配置在 GT 下保持 `+10 deg` 角度任务可完成，同时显著降低 foot displacement 和 base approach。

12. 近 3 cm / 近 5 cm 站位不是直接答案。
    - 问题：让机器人相对阀门更近会改善可达性的直觉并不可靠；near3/near5 可能让 pre-turn force 上限更容易触发，身体更贴近面板。
    - 处理：保持默认场景 XML 不变，只通过本地测试 XML/overlay 做对照；当前没有把近距离站位写成默认。

13. `MUJOCO_GL=egl` 下 vision renderer 会失败。
    - 问题：EGL 完整 run 中出现 `Failed to create MuJoCo valve vision renderer: Failed to make the EGL context current`，provider 读到 stale vision status 后全程 fallback GT，容易把 GT 结果误当 vision 结果。
    - 处理：本机 vision overlay 验证改用 `MUJOCO_GL=glfw`；每条 run 后检查 CSV 中 `geometry_source_used` 和 `latched_geometry_source`，只有包含 `vision_overlay / vision_latched` 的样本才计入 vision-latched 统计。

14. vision overlay turn-validation 还不是 5/5。
    - 问题：有效 vision-latched 五条样本都进入 turn，但只有 `3/5` 达到 `±2 deg`；失败分别表现为 overshoot 到约 `12.41 deg` 和 under-turn 到约 `7.03 deg`。
    - 处理：不再回头改 marker、PTIM 或 grasp_R；当前判断剩余瓶颈在 turn angle tracking / 接触驱动力一致性，需要后续围绕 closed-loop 参数、接触力、阀门阻尼/摩擦和切向目标生成继续分析。

### 当前推荐实验入口

本地 overlay 文件：

- `config/local/turn_validation_no_strict_grasp.yaml`：关闭 strict grasp 对 close/turn entry/completion 的阻断，保留 strict topology 作为 debug。
- `config/local/vision_turn_validation_approach.yaml`：启用 vision overlay approach，关闭 vision angle override。
- `config/local/diag_comp_C1_tangent_only.yaml`：只保留阀门切向 contact compensation。

运行 vision overlay + turn-validation 的示例：

```bash
cd /home/lavine/project/FALCON/sim2real
MUJOCO_GL=glfw \
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
RUN_ID=vision_turn_validation_glfw_10deg_01 \
DURATION_SEC=80 \
VALVE_TURN_TARGET_SEQUENCE_DEG=10 \
VALVE_TURN_SPEED_DEG=5 \
CONTACT_TURNING_OVERLAY_CONFIG=config/local/diag_comp_C1_tangent_only.yaml:config/local/turn_validation_no_strict_grasp.yaml:config/local/vision_turn_validation_approach.yaml \
LOG_FILE=/tmp/falcon_valve_vision_turn_validation_glfw_10deg_01.csv \
MARKER_FILE=/tmp/falcon_valve_vision_turn_validation_glfw_10deg_01_markers.json \
SIM_STATUS_FILE=/tmp/falcon_valve_vision_turn_validation_glfw_10deg_01_status.json \
SIM_CONTROL_FILE=/tmp/falcon_valve_vision_turn_validation_glfw_10deg_01_control.json \
timeout 150s bash launch_valve_contact_turning_7dof_with_hand_auto.sh auto
```

注意：当前机器上 `MUJOCO_GL=egl` 下 vision renderer 曾出现 EGL context 创建失败，导致 provider 读到 stale vision status 并 fallback 到 GT。进行 vision overlay 验证时应优先使用 `MUJOCO_GL=glfw`，并检查 CSV 中 `geometry_source_used` 是否包含 `vision_overlay / vision_latched`。

### 下一步建议

- 不再优先追 `PTIM`、thumb contact 或 functional probe；这些保留为诊断标签。
- 优先分析 vision-latched 样本中 overshoot / under-turn 的原因：turn closed-loop 参数、接触驱动力、阀门阻尼/摩擦、角度保持和切向目标生成。
- 继续把最终阀门角度误差、接触力、slip、EE error、foot displacement、base approach 作为任务质量指标。
- 在考虑实机或真实相机前，保持 `vision_override_angle=false`；视觉角度仍只做 debug 对比。

## 相关文档

- `WORKLOG_ee_tracking_with_hand.md`：本阶段末端跟踪和带手模型工作的中文记录。
- `docs/worklogs/vision_module_branch_prep_2026-05-01.md`：视觉模块新分支准备记录。
- 旧版 `README_valve_demo.md` 已合并到本 README，仓库内不再单独维护。

## 文档维护约定

后续每新增一个可直接运行的 `.py` 实验脚本或 launch 脚本，都需要同步更新本 README，至少写清楚：

- 脚本入口和推荐运行命令。
- 必要输入文件、模型和配置 overlay。
- 常用环境变量或命令行参数。
- 默认输出文件位置。
- 适用场景和不适用边界。
- 如果有分析脚本，写清楚如何从 CSV 生成报告、图和视频。

## 致谢

本目录基于以下开源项目和原始 FALCON 部署链路扩展：

- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [booster_robotics_python](https://github.com/BoosterRobotics/booster_robotics_sdk)
