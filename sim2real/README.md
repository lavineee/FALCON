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

旧版 `README_valve_demo.md` 中提到过 `launch_valve_demo_auto_v2.sh`，但当前仓库没有保留这个脚本。后续如果要恢复阀门自动化，建议参考现有 `launch_ee_tracking_*_auto.sh` 的结构重新写一份，不要直接依赖旧文档中的脚本名。

## sim2real 注意事项

真实机器人部署前必须先完成 sim2sim 验证。建议遵守以下顺序：

1. 在 MuJoCo 中确认策略能稳定站立、行走和执行上肢目标。
2. 检查 `INTERFACE`、`DOMAIN_ID`、SDK 类型和机器人型号是否正确。
3. 从保守的 kp / kd 或 scale 设置开始。
4. 确认机器人脚已经稳定接触地面后再启用 policy。
5. 保持急停可用：键盘 `o` 可停止 policy action，手柄急停按键按原 FALCON 协议执行。

需要实时 onboard 推理时，纯 Python 的 `unitree_sdk2_python` 可能无法保证 Jetson Orin 上的实时性。FALCON 原作者建议使用 C++ 后端加 pybinding 的 `unitree_sdk2` 方案。

## 相关文档

- `WORKLOG_ee_tracking_with_hand.md`：本阶段末端跟踪和带手模型工作的中文记录。
- `README_valve_demo.md`：阀门 demo 旧文档入口，当前已合并到本 README。

## 致谢

本目录基于以下开源项目和原始 FALCON 部署链路扩展：

- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [booster_robotics_python](https://github.com/BoosterRobotics/booster_robotics_sdk)
