# 末端跟踪与带手模型阶段工作日志

日期：2026-04-25

## 工作背景

本阶段的目标是在进入“抓住阀门并持续转动”之前，先验证右臂末端是否能够可靠跟踪给定目标点。因为后续阀门任务的本质是末端沿受约束圆周轨迹运动，如果自由空间中的末端点位跟踪本身不稳定，直接做抓取和转动会把问题混在一起，难以判断误差来源。

这一阶段刻意不修改已经训练好的 29DoF ONNX 策略。策略仍然只控制原始本体关节；如果模型中增加手部关节，手部开合由额外的简单控制器处理。

## 主要改动

1. 增加右臂末端跟踪评测脚本。
   - 在机器人 base 坐标系下随机采样右臂末端目标点。
   - 默认每 5 秒重采样一次。
   - 记录目标点、当前末端点和跟踪误差。
   - 在 MuJoCo 中可视化目标点和当前末端代理点。

2. 增加 7DoF 右臂跟踪版本。
   - 将 IK 从原来的 4 个手臂自由度扩展到右臂 7 个自由度。
   - 当前只评测末端位置，不评测末端姿态。
   - 7DoF 版本使用手掌 / 抓取中心附近的末端代理点，不再使用旧的假手板中心作为最终定义。

3. 自动化 MuJoCo 测试流程。
   - 自动启动 MuJoCo。
   - 自动启动 policy。
   - policy 启动后延迟松开弹力绳。
   - 稳定后开始随机点跟踪测试。
   - 避免每次都手动按 `]` 和 `9`。

4. 增加 FALCON 兼容的带手 MuJoCo 模型。
   - 保留原 FALCON 29DoF 本体模型、freebase actuator、地面摩擦和场景假设。
   - 在原模型后追加真实手部 body、joint 和 actuator。
   - 14 个手部 actuator 追加在原始本体 actuator 之后。
   - policy 控制通过关节 / actuator 名称显式映射，避免 actuator 数量变化导致写错控制量。

5. 增加简单手部控制器。
   - ONNX 策略仍然只输出 29DoF 控制量。
   - 手部关节单独由 PD torque 控制。
   - 跟踪测试中，右手在新目标出现时打开；当末端误差低于配置阈值后闭合。

6. 移除对外部 `unitree_ros` 路径的运行依赖。
   - 将带手 URDF 和缺失 mesh 资产复制到本仓库。
   - 带手 IK 使用仓库内资产路径。
   - 路径解析逻辑支持从当前仓库定位 URDF 和 mesh。

## 重要文件

- `sim2real/sim_env/loco_manip.py`
  - 增加 EE marker 可视化。
  - 支持自动设置弹力绳长度和自动松绳。
  - `matplotlib` 缺失时可关闭实时曲线，不影响仿真。
  - 对无阀门场景做容错，避免 tracking-only 场景启动失败。

- `sim2real/sim_env/loco_manip_with_hand.py`
  - 针对带手模型增加 name-mapped 29DoF policy bridge。
  - 增加手部 PD 控制器。
  - 对手部关节做运行时稳定化设置。

- `sim2real/rl_policy/loco_manip/loco_manip_ee_tracking_test.py`
  - 4DoF 假手末端跟踪评测脚本。

- `sim2real/rl_policy/loco_manip/loco_manip_ee_tracking_7dof_test.py`
  - 7DoF 假手位置跟踪评测脚本。

- `sim2real/rl_policy/loco_manip/loco_manip_ee_tracking_7dof_with_hand_test.py`
  - 7DoF 带手位置跟踪评测脚本。
  - 包含右手开合状态写入逻辑。

- `sim2real/utils/arm_ik/robot_arm_ik_with_hand.py`
  - 带手模型 IK 包装器。
  - 锁定下肢和手指关节，保持左右 7DoF 手臂可动。
  - 增加 `L_ee` 和 `R_ee` frame，用于手掌 / 抓取中心末端定义。

- `humanoidverse/data/robots/g1/g1_29dof_old_freebase_with_hand.xml`
  - FALCON 兼容的带手 MuJoCo 机器人模型。

- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand_tracking.xml`
  - 不含阀门的带手跟踪测试场景。

- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand.xml`
  - 带阀门的带手场景，后续用于抓取和转动验证。

- `sim2real/config/g1/g1_29dof_with_hand_tracking_overlay.yaml`
  - 带手跟踪测试 overlay 配置。
  - 定义带手场景、IK 资产路径、本体关节映射、手部关节、手部开合目标和 PD 参数。

## 当前推荐测试命令

推荐优先运行 7DoF 带手末端跟踪：

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_with_hand_auto.sh
```

该脚本会自动完成：

1. 启动 MuJoCo。
2. 设置弹力绳初始长度。
3. 启动 policy。
4. 自动进入 policy 控制。
5. policy 稳定后松开弹力绳。
6. 开始随机目标跟踪。

## 启动脚本

所有命令都应在 `sim2real/` 目录下执行。当前机器上建议显式指定：

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python
```

### 4DoF 假手跟踪

自动运行：

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_auto.sh
```

拆分终端：

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_sim.sh
```

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
AUTO_START_POLICY=1 \
DURATION_SEC=60 \
./launch_ee_tracking_policy.sh
```

默认输出：

- `/tmp/falcon_ee_tracking_markers.json`
- `/tmp/falcon_ee_tracking_metrics.csv`

### 7DoF 假手跟踪

自动运行：

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_auto.sh
```

拆分终端：

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_sim.sh
```

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
AUTO_START_POLICY=1 \
DURATION_SEC=60 \
./launch_ee_tracking_7dof_policy.sh
```

默认输出：

- `/tmp/falcon_ee_tracking_7dof_markers.json`
- `/tmp/falcon_ee_tracking_7dof_metrics.csv`

### 7DoF 带手跟踪

自动运行：

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_with_hand_auto.sh
```

拆分终端：

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_with_hand_sim.sh
```

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
AUTO_START_POLICY=1 \
DURATION_SEC=60 \
./launch_ee_tracking_7dof_with_hand_policy.sh
```

默认输出：

- `/tmp/falcon_ee_tracking_7dof_with_hand_markers.json`
- `/tmp/falcon_ee_tracking_7dof_with_hand_metrics.csv`

## 常用参数

- `DURATION_SEC`：总运行时间，单位秒。
- `SAMPLE_PERIOD_SEC`：目标点重采样周期，默认 5 秒。
- `SEED`：随机采样种子。
- `X_RANGE` / `Y_RANGE` / `Z_RANGE`：base 坐标系下目标采样范围，格式为 `min,max`。
- `ELASTIC_LENGTH`：初始弹力绳长度，当前稳定值是 `-0.05`。
- `ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC`：policy 启动后多久松绳，默认 1 秒。
- `TRACKING_START_DELAY_SEC`：松绳后多久开始目标点测试。
- `HAND_CLOSE_ERROR_M`：带手版本中右手闭合阈值。

## 可视化和数据含义

- 绿色球：当前目标点。
- 红色球：当前末端代理点。
- CSV 中记录的误差：绿色球与红色球之间的欧氏距离。
- P90 / P95：误差分布的 90% / 95% 分位数。例如 P95 为 2.5 cm，表示 95% 的样本误差不超过 2.5 cm。

4DoF 和 7DoF 的末端定义不同：

- 4DoF 版本的 `R_ee` 是无手腕 IK 下的虚拟末端点，可用于早期简化测试。
- 7DoF 版本的 Pinocchio `R_ee` 与 MuJoCo `right_EE_frame` 已按相同 offset 对齐，适合作为后续阀门任务的末端位置定义。

## 当前验证结果

仓库本地资产回归已经跑通：

- MuJoCo 带手场景可以加载。
- IK 可以从仓库本地 URDF 和 mesh 加载。
- policy 可以自动启动。
- policy 启动后弹力绳可以自动释放。
- 手部开合命令可以执行。
- 右臂末端跟踪 CSV 可以生成。

已记录的典型结果：

- 4DoF 假手长时间测试：整体均值约 2.92 cm，稳态均值约 1.33 cm，稳态 P95 约 2.00 cm。
- 7DoF 假手测试：整体均值约 3.25 cm，稳态均值约 1.62 cm，稳态 P95 约 2.27 cm。
- 7DoF 放大范围测试：整体均值约 4.19 cm，稳态均值约 1.77 cm，稳态 P95 约 2.58 cm。
- 7DoF 带手短回归：均值约 4.56 cm，RMS 约 5.54 cm，最大误差约 18.25 cm，末端误差峰值主要来自目标切换瞬态。

## 模型选择记录

曾评估过直接引入 NVlabs GR00T-WholeBodyControl / SONIC 中的 G1 模型。该模型族有参考价值，但存在几个风险：

- mesh 资产依赖 Git LFS，不适合直接作为当前仓库的稳定依赖。
- XML / actuator / contact 设置不一定与 FALCON 已训练策略一致。
- 完整替换模型可能改变本体质量、碰撞、关节阻尼、自由基座和 actuator 顺序。

当前决策：

- 不直接用 SONIC 或 Unitree 完整模型替换 FALCON 原始模型。
- 保留 FALCON 兼容本体。
- 只把外部模型作为手部结构和几何设计参考。
- 手部 actuator 追加在原始 29DoF actuator 之后，策略控制映射用名称显式绑定。

这个选择更适合当前方案，因为 FALCON 策略对 actuator 顺序、contact / friction、freebase actuator 和关节布局都比较敏感。

## Git 记录

本阶段改动已放在独立分支：

```bash
feature-with-hand-ee-tracking
```

已提交并推送的阶段性提交：

```bash
4c095bd Add with-hand EE tracking evaluation
```

后续建议继续在该分支上完善带手末端跟踪、阀门抓取和阀门旋转，不要直接推到 `main`。
