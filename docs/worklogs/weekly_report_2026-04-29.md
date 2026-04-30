# 本周组会周报：FALCON 6D 扰动训练与阀门转动仿真进展

日期：2026-04-29  
当前分支：`feature-6d-wrench-training`  
当前最近提交：`59489a6 Add 6D wrench warm-start training path`

## 1. 本周工作目标

本周工作围绕毕业设计中的两条主线推进：

1. **训练层扩展**：在 FALCON 原有 3D 末端外力课程学习基础上，继续推进 6D generalized wrench 扰动训练路径，为后续受力/受力矩任务提供训练基础。
2. **任务验证层扩展**：从“末端能否跟踪目标点”逐步推进到“带手模型能否接近、抓住并转动阀门”，重点验证持续接触下的末端轨迹执行、阀门角度闭环和机器人全身稳定性。

本周的核心目标不是追求 MuJoCo 中完全真实的三指摩擦抓握，而是形成一条**可复现、可量化、可展示、未来可向实机部署继承**的阀门转动 demo 工作流。

## 2. 当前代码状态

### 2.1 Git 状态

当前分支：

```text
feature-6d-wrench-training
```

最近提交：

```text
59489a6 Add 6D wrench warm-start training path
6ee2d55 Add contact valve turning demo workflow
8edc08e Improve EE tracking diagnostics and tuning
449116a Document sim2real tracking and valve workflows
4c095bd Add with-hand EE tracking evaluation
```

当前仍有未提交修改，主要集中在阀门接触转动 demo 和文档：

```text
M docs/worklogs/valve_turning_contact_demo.md
M sim2real/README.md
M sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh
M sim2real/launch_valve_grasp_contact_7dof_with_hand_policy.sh
M sim2real/rl_policy/loco_manip/loco_manip_valve_grasp_contact_7dof_with_hand.py
M sim2real/tools/analyze_valve_contact_turning_metrics.py
M test
?? tools/motion_browser/
```

### 2.2 本周主要修改模块

1. **6D wrench 训练路径**
   - 新增 6D wrench 训练环境、观测配置和 warm-start 工作流。
   - 新增远程服务器训练同步脚本和训练文档。
   - 目标是让训练层支持更一般的 6D 广义扰动，而不是只考虑末端 3D force。

2. **末端跟踪评测**
   - 完成 4DoF、7DoF、7DoF-with-hand 三类末端跟踪测试入口。
   - 支持在机器人 base 坐标系下随机采样右手目标点。
   - 在 MuJoCo 中可视化目标点、当前末端点和误差。
   - 通过 CSV 记录末端位置误差，作为阀门轨迹执行前的前置验证。

3. **带手模型与手部控制**
   - 引入带手 G1 模型。
   - 原 ONNX policy 仍只输出 29DoF 本体关节。
   - 多出来的手部 actuator 由独立 hand controller 控制。
   - 当前手部控制简化为打开/闭合状态，不做复杂手指规划。

4. **阀门模型与接触几何**
   - 重新检查阀门 XML 的几何、碰撞、摩擦、hinge 阻尼和支架干涉。
   - 修正 MuJoCo XML 中 `angle="radian"` 下误用 `euler="0 90 0"` 导致的阀门中心圆柱倾斜问题。
   - 将阀门轮缘设计为更适合三指夹爪接触的统一圆管轮缘。
   - 缩小并后移后方灰色支架块，减少手部抓取过程中的不必要碰撞。

5. **阀门接触转动 baseline**
   - 建立从接近、闭手、接触判断、转动、目标保持到完成的状态机。
   - 当前 demo 采用“真实接触 + soft connect 辅助 + 阀门角度闭环 + hold 阶段 angle brake”的工程近似方案。
   - 增加严格单目标测试 preset，例如 `strict30`、`strict45`、`strict60`。
   - 增加 contact health logging 和 per-segment summary，便于定量评估每段转动质量。

## 3. 本周已完成内容

### 3.1 6D wrench 训练路径

本周提交 `59489a6` 中加入了 6D wrench warm-start 训练路径，主要内容包括：

- 新增 6D wrench 环境实现；
- 新增 6D wrench 观测历史配置；
- 新增 6D wrench PPO / multi-agent 训练配置；
- 新增服务器训练工作流文档；
- 新增远程同步、训练启动和结果拉取脚本；
- 新增训练稳定性和审计文档。

这部分工作对应毕业设计第一条主线：把 FALCON 原始的 3D 末端外力扰动扩展到更一般的 6D wrench 扰动，为后续包含力矩扰动和持续接触的任务提供训练基础。

### 3.2 末端跟踪能力验证

在正式做阀门抓取前，本周先验证右手末端能否稳定跟踪目标点。

完成内容：

- 基于 `loco_manip.py` 派生出末端跟踪测试脚本。
- 支持每 5 秒在合理工作空间内随机采样目标点。
- 在 MuJoCo 中显示目标点和当前末端点。
- 对 4DoF IK、7DoF IK 和带手 7DoF 模型分别做了测试。
- 明确了 4DoF 版本只是简化对比，最终阀门任务应使用包含腕部的 7DoF 版本。

结论：

- 仅位置跟踪场景下，4DoF 和 7DoF 肉眼效果都可用。
- 7DoF 版本更适合后续阀门任务，因为真实转阀门需要手腕参与。
- 当前末端定义和 MuJoCo 可视化点已经基本对齐，但实机部署前仍需要重新标定掌心/夹爪工作点。

### 3.3 带手模型和手部控制器

本周把原先的假手/末端代理进一步推进到带手模型。

关键处理：

- 保持原 29DoF policy 不变。
- 带手模型新增的手指 actuator 不由 ONNX policy 输出。
- 在 MuJoCo 侧增加独立 hand controller，负责打开和闭合。
- 手部控制暂时采用工程简化：接近时打开，到达抓点后闭合。

这使得后续任务可以从“末端点跟踪”过渡到“手和阀门几何发生接触”。

### 3.4 阀门模型重建与几何检查

本周多次发现控制效果异常并不完全来自 policy，而是来自阀门模型本身，包括：

- 阀门中心圆柱倾斜；
- 支架块过大，可能压缩手部接近空间；
- 局部加粗抓取块不自然，不利于解释为真实阀门；
- 原始轮缘过细，三指夹爪不容易形成稳定包络；
- 某些碰撞设置导致手指穿过阀门或接触不充分。

解决方法：

- 修正阀门 XML 中的弧度/角度错误；
- 删除不合理的局部长方体抓取块；
- 将外圈设计为统一尺寸的圆管轮缘；
- 适当增加轮缘接触面摩擦和接触稳定性；
- 缩小灰色支架块，降低非任务接触风险；
- 增加/检查阀门自由转动测试，确认轮盘不会和支架发生明显干涉。

当前判断：

- 阀门模型已经比早期版本更适合作为 demo 基础。
- 但它仍然是为了当前三指夹爪和 MuJoCo 接触稳定性做过工程设计的模型，后续实机应根据真实阀门尺寸重新标定。

### 3.5 阀门接触转动 baseline

本周建立了新的阀门转动 baseline。

当前状态机大致为：

```text
move_pregrasp
-> approach_grasp
-> close_hand
-> pre_turn_settle
-> turn_valve
-> turn_hold
-> done / failed
```

当前方案：

- 机器人先移动到阀门预抓点。
- 手掌和手指与阀门轮缘形成接触。
- 抓握满足一定接触条件后，启用柔顺 `connect` 辅助传力。
- 沿阀门圆周生成末端目标轨迹。
- 使用阀门实际角度做闭环修正。
- 到达目标附近后进入保持阶段。
- 保持阶段使用有限力矩 angle brake 吸收残余角速度。

需要强调：

当前方案不是 pure contact grasp，也不是永久 weld。它是介于真实接触和工程辅助之间的 MuJoCo demo baseline。这样做的原因是：当前研究重点是验证“阀门位姿输入 -> 末端接近 -> 闭手 -> 沿约束圆弧运动 -> 角度闭环停止”的控制流程，而不是在 MuJoCo 中无限追求三指摩擦抓握的完全真实复现。

## 4. 实验结果汇总

### 4.1 soft-connect 角度跟踪 baseline

早期 soft-connect baseline 用于作为上限参考。

示例日志：

```text
artifacts/demo_logs/valve_angle_tracking_soft_connect_30deg_baseline.csv
```

结果：

- 目标角：30 deg
- 最终实际角：25.62 deg
- 最终误差：4.38 deg
- 末端误差均值：约 2.8 cm
- soft-connect 拉伸均值：约 3.3 mm

结论：

- soft-connect 可以稳定传力，但角度精度并非天然完美。
- 它适合作为对照 baseline，但不能直接等价于真实抓握。

### 4.2 真实接触小角度测试

早期不依赖永久 weld/connect 的小角度测试：

| 日志 | 目标角 | 实际角 | 误差 | 判断 |
| --- | ---: | ---: | ---: | --- |
| `20260426_211334.csv` | 5 deg | 4.23 deg | 0.77 deg | 成功 |
| `20260426_211432.csv` | 5 deg | 5.10 deg | -0.10 deg | 成功 |
| `20260426_234055.csv` | 10 deg | 9.27 deg | 0.73 deg | 当前最好小角度结果 |

结论：

- 小角度转动证明控制链路可行。
- 但纯真实接触容易在保持阶段出现接触拓扑退化和滑移。

### 4.3 稳定性约束下的 10 度测试

示例日志：

```text
artifacts/demo_logs/valve_contact_turning_20260427_080148.csv
```

结果：

- 目标角：10 deg
- `turn_hold` 末尾实际角：约 9.74 deg
- 整次运行末尾实际角：约 10.20 deg
- 转动/保持阶段 base 靠近：约 6.4 cm
- base yaw 漂移：约 4.8 deg
- 未触发稳定性失败判据

结论：

- 加入稳定性监测后，不再只看阀门角度是否到达。
- 角度闭环和身体稳定性可以同时记录和评估。

### 4.4 严格单目标工作区间评估

本周重点评估了单次转动能力边界。严格标准要求：角度达到目标后，还要保持住，并且最终抓握不能丢失。

| 目标 | 代表日志 | 实际角 | 最终误差 | 抓握 | 判断 |
| ---: | --- | ---: | ---: | --- | --- |
| 30 deg | `20260427_173134` | 30.43 deg | -0.43 deg | 成立 | 成功 |
| 45 deg | `20260427_173802` | 44.62 deg | 0.38 deg | 成立 | 成功 |
| 45 deg | `20260427_182841` | 44.86 deg | 0.14 deg | 成立 | 成功 |
| 45 deg | `20260427_183028` | 45.61 deg | -0.61 deg | 成立 | 成功 |
| 60 deg | `20260427_173908` | 59.99 deg | 0.01 deg | 丢失 | 失败 |
| 90 deg | `strict_single_turn_090deg_entry50` | 约 89.5 deg | 小 | 丢失 | 失败 |

本周最有展示价值的结果：

- `strict45` 单目标转动可以稳定展示；
- 45 deg 目标多次运行最终误差在约 0.1-0.6 deg 量级；
- 60/90 deg 不是角度闭环做不到，而是手臂构型接近极限，hold 阶段抓握质量下降；
- 因此当前稳定单次工作区间建议先定位为 30-45 deg。

## 5. 本周遇到的主要难点与解决方法

### 难点 1：末端点和真实手掌抓取点定义不一致

问题：

- 早期末端跟踪的点是 wrist/假手附近的代理点。
- 带手模型引入后，真正用于抓阀门的点应更接近掌心内侧。
- 如果工作点定义不准，即使 IK 到位，手指和阀门也可能没有正确接触。

解决：

- 在 MuJoCo 中可视化目标点和实际末端点；
- 逐步校准掌心/抓取点 offset；
- 当前将阀门抓点通过 MuJoCo site 读取，再转到机器人 base 坐标系下作为 IK 目标。

### 难点 2：三指夹爪真实接触不稳定

问题：

- 手指容易套在阀门边缘上滑动；
- 摩擦不足或接触几何不匹配会导致局部相对姿态漂移；
- 位置控制式闭手无法像真实手一样主动调节接触力。

解决：

- 改进阀门轮缘几何，使其更适合当前夹爪包络；
- 增加接触健康指标，例如掌心、拇指、食指、中指接触状态；
- 记录相对滑移和接触力，而不是只看角度；
- 在 demo baseline 中允许柔顺 `connect` 作为抓住后传力近似。

### 难点 3：阀门模型本身影响实验判断

问题：

- 阀门中心圆柱出现异常倾斜；
- 后方支架块可能干扰抓取；
- 轮缘局部加粗不自然，容易让 demo 变成抓特殊凸起。

解决：

- 修正 XML 中弧度/角度错误；
- 改成统一圆管轮缘；
- 缩小后方支架；
- 做阀门自由转动检查，确认模型可以正常旋转。

### 难点 4：角度到达不等于任务成功

问题：

- 60/90 deg 测试中阀门角度可以到达，但手已经脱离或抓握质量很差；
- 如果只用角度误差判成功，会误判任务质量。

解决：

- 引入 strict 单目标 preset；
- 到达目标后要求继续保持；
- 同时检查角度误差、阀门角速度、抓握状态、接触健康比例、相对滑移和 base 姿态。

### 难点 5：大角度单次转动受手臂构型限制

问题：

- 单次 90 deg 时，右臂接近直臂，后续运动空间不足；
- 人类实际转阀门也通常会分段重抓，而不是单次连续转很大角度。

解决：

- 将当前目标收敛到 30-45 deg 稳定单次工作区间；
- 对更大角度任务，计划采用“转一段、松开/重置抓点、再转一段”的分段策略。

## 6. 当前运行入口与参数

### 6.1 推荐展示命令

45 deg 正向展示：

```bash
cd /home/lavine/project/FALCON
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict45
```

45 deg 反向展示：

```bash
cd /home/lavine/project/FALCON
VALVE_TURN_TARGET_SEQUENCE_DEG=-45 \
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict45
```

调节转动速度：

```bash
VALVE_TURN_SPEED_DEG=1.2 \
bash sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh strict45
```

当前 45 deg 展示推荐速度约为 `0.9-1.2 deg/s`。速度太慢会增加持续接触时间，机器人可能更容易被阀门拉近；速度太快会增加接触冲击和滑移风险。

### 6.2 关键配置文件

```text
sim2real/config/g1/g1_29dof_with_hand_valve_contact_turning_overlay.yaml
sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh
sim2real/rl_policy/loco_manip/loco_manip_valve_grasp_contact_7dof_with_hand.py
humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand_valve_ribbed_round_grip_contact.xml
humanoidverse/data/robots/g1/valve_ribbed_round_grip_body_include.xml
```

### 6.3 关键日志字段

当前 CSV 和 summary 已经记录：

- 阀门目标角、实际角、角度误差；
- 阀门角速度；
- 末端位置误差；
- 手-阀门接触数量；
- 掌心/拇指/食指/中指接触状态；
- 接触力；
- 相对滑移；
- soft connect 是否启用；
- angle brake torque；
- base roll/pitch/yaw；
- base 靠近量；
- base yaw 漂移；
- 每段任务结果和失败原因。

## 7. 当前结论

1. 右手末端位置跟踪能力已经满足阀门轨迹执行前置验证需求。
2. 7DoF 带手模型路径已经跑通，手部 actuator 与原 29DoF policy 解耦。
3. 阀门模型经过几何修正后，已经能支撑当前接触转动 demo。
4. 当前已形成一个可展示的 soft-contact assisted valve turning baseline。
5. 在严格成功判据下，当前稳定单次工作区间建议定位为 30-45 deg。
6. 60/90 deg 单次转动不宜作为当前 demo，因为虽然角度能到，但抓握保持质量不足。
7. 后续大角度转动更合理的技术路线是分段重抓，而不是强行单次转很大角度。

## 8. 待解决问题

1. **抓握真实度**
   - 当前仍依赖 soft connect 和 angle brake 作为 MuJoCo 工程近似；
   - 真实三指摩擦抓握还不能完全稳定保持局部相对姿态。

2. **大角度转动**
   - 单次 60 deg 以上会出现抓握质量下降；
   - 需要设计分段转动和重抓策略。

3. **视觉接口**
   - 当前阀门中心、抓点和角度主要来自 MuJoCo 特权信息；
   - 后续需要用 XML 中的 `head_camera` 或外部深度相机模拟视觉输入；
   - 实机部署还需要相机内参、外参、深度对齐和时间同步。

4. **实机参数迁移**
   - 当前阀门尺寸、摩擦、阻尼和接触参数经过仿真调试；
   - 实机阀门的半径、阻尼、摩擦、抓点位置和手部实际闭合能力需要重新测量。

5. **身体稳定性**
   - base 适度靠近和 yaw 调整可以接受；
   - 但应继续记录并优化 base 漂移，避免长期接触任务中身体姿态逐渐恶化。

## 9. 下周计划

### 9.1 展示稳固化

目标：

- 保持当前 45 deg demo 不被破坏；
- 保存正向 45 deg 和反向 45 deg 的高质量演示视频；
- 确认每次 demo 的日志、summary 和视频可以对应。

计划：

- 固定一个展示 preset；
- 减少不必要可视化，只保留目标角、实际角、状态文字和关键 marker；
- 保持默认相机视角在机器人右后方俯视，便于观察右手操作。

### 9.2 45 deg 工作区间优化

目标：

- 提升 45 deg 单目标任务的成功率和平滑度；
- 尽量缩短执行时间；
- 减少最终保持阶段的滑移。

计划：

- 小范围测试 `VALVE_TURN_SPEED_DEG=0.9/1.0/1.2`；
- 对比角度误差、滑移、接触力峰值和 base 漂移；
- 不同时修改阀门模型、接触参数和状态机，避免调参失控。

### 9.3 分段转动方案预研

目标：

- 为 90/120 deg 阀门转动建立合理方案。

计划：

- 设计“转 30-45 deg -> 保持 -> 松开 -> 重置抓点 -> 再转”的状态机草案；
- 暂时先在仿真中验证流程，不急于追求最终大角度成功率；
- 分段策略应服务于实机部署，而不是只服务 MuJoCo demo。

### 9.4 视觉接口准备

目标：

- 从 MuJoCo 特权信息过渡到仿真相机输入。

计划：

- 使用当前 XML 中的 `head_camera` 渲染 RGB/depth；
- 设计 `VisionGraspGeometryProvider`，输出与当前 MuJoCo 真值 provider 相同的数据结构；
- 用 MuJoCo 真值作为 ground truth，评估视觉估计误差；
- 后续再迁移到真实深度相机。

## 10. 适合 PPT 的讲述结构

建议 PPT 按以下 8 页组织：

1. **本周目标**
   - 6D wrench 训练路径
   - 阀门持续接触 demo

2. **整体技术路线**
   - FALCON 双智能体控制
   - 6D 扰动建模
   - MuJoCo 阀门交互验证

3. **末端跟踪前置验证**
   - 4DoF vs 7DoF
   - 带手模型
   - 目标点可视化和误差记录

4. **阀门模型修正**
   - 原模型问题
   - 统一圆管轮缘
   - 支架避碰
   - 接触参数检查

5. **阀门转动控制流程**
   - approach
   - close hand
   - contact check
   - closed-loop turn
   - hold / brake

6. **实验结果**
   - 5/10 deg 小角度结果
   - 30/45 deg strict 成功
   - 60/90 deg 失败原因

7. **难点与解决**
   - 真实接触滑移
   - 角度到达不等于成功
   - 大角度受手臂构型限制
   - soft connect 的工程近似定位

8. **下周计划**
   - 45 deg 展示稳固化
   - 速度和平滑度优化
   - 分段重抓方案
   - 视觉接口准备

## 11. 可以在组会上强调的结论

1. 本周不是只做了一个 demo，而是建立了从末端跟踪、带手模型、阀门模型、接触抓握、角度闭环到定量评测的一整套验证链路。
2. 当前结果表明：在 30-45 deg 单次工作区间内，机器人可以在保持基本稳定的情况下完成阀门角度跟踪。
3. 当前失败最主要不是角度闭环本身，而是大角度下抓握保持和手臂工作空间不足。
4. 对真实机器人任务来说，分段重抓比单次大角度旋转更符合实际操作方式。
5. 下一步重点不是继续盲目调参，而是把 45 deg demo 做稳、做快、做可复现，并准备视觉输入接口。

