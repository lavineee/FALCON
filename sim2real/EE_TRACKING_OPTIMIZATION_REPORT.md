# 带手末端位置跟踪优化报告

日期：2026-04-25

## 目标

本阶段目标是改善 7DoF 带手模型下右手末端位置跟踪的晃动和稳态误差，并建立一套可复现的定量评测流程。评测仍然只关注自由空间中的末端位置跟踪，不建立手和阀门的 weld，也不要求姿态跟踪。

需要保留的控制约束：

- 29DoF ONNX policy 不重新训练。
- 手部 actuator 继续由独立开合控制器处理。
- 上肢 RL residual 继续叠加在 IK 目标上。这个 residual 是为了抗扰动和平衡协同设计的，不能为了末端跟踪误差直接移除。

## 当前误差来源判断

误差可以拆成三部分：

- `IK error`：目标点到 IK 参考末端点的距离。它反映目标插值、IK 目标权重、平滑项和目标可达性。
- `servo error`：实际末端点到 IK 参考末端点的距离。它反映关节 PD、力矩前馈、模型阻尼、手部质量、policy residual 叠加以及仿真动力学。
- `total error`：目标点到实际末端点的距离，也就是最终跟踪误差。

基线评测显示，稳态误差主要来自 `servo error`，而不是单纯 IK 不收敛。优化后 `IK error` 已经降到毫米级，但 `servo error` 仍然约 2 cm。这说明后续继续压低误差时，主要矛盾已经从 IK 转向“实际关节是否贴住参考”和“residual 是否引入必要但不可避免的偏置”。

## 已完成的工程改动

1. 增加误差分解日志。
   - CSV 从只记录 `error_m` 扩展为同时记录 `target_age_s`、`current_speed_mps`、`ik_error_m`、`servo_error_m`。
   - 每个目标点窗口的日志现在会打印 `mean/rms/max/final/ik_mean/servo_mean/speed_p95`。

2. 增加可复用评测分析脚本。
   - 新增 `sim2real/tools/analyze_ee_tracking_metrics.py`。
   - 支持总体误差、稳态误差、final error、IK/servo 分解和末端速度 P95。
   - 支持输出 Markdown 摘要和 PNG 曲线图。

3. 修正 IK 调参入口。
   - 原 7DoF 测试脚本虽然设置了 `self.speed_factor=0.05`，但并没有真正写入 `upper_body_controller.speed_factor`。
   - 现在 `tracking_ik_speed_factor` 会直接作用到 IK controller。
   - 支持通过配置调整 `tracking_ik_translational_weight`、`tracking_ik_regularization_weight`、`tracking_ik_smooth_weight`、`tracking_ik_filter_weights` 和 `tracking_ik_include_rotation`。

4. 增加保守的上肢力矩前馈。
   - IK 已经能计算 `sol_tauff`，此前 7DoF 跟踪测试直接丢弃了它。
   - 现在可以通过 `tracking_upper_tau_ff_scale` 和 `tracking_upper_tau_ff_clip` 使用小比例前馈，减轻 PD 在手部质量和重力下的稳态负担。
   - 当前 tuned 配置使用 `scale=0.20`、`clip=2.5`，避免前馈过强并和 policy residual 打架。

5. 增加按关节名缩放 PD 增益。
   - 新增 `tracking_motor_kp_scale_by_name` 和 `tracking_motor_kd_scale_by_name`。
   - 当前 tuned 配置只对左右 wrist roll/pitch/yaw 缩放，避免全身增益整体变化影响 policy 分布。
   - 原始腕部 Kp/Kd 很低，而 7DoF 末端点位于 wrist yaw 之后，腕关节误差会被末端 offset 放大。

6. 增加 tuned overlay 和统一启动入口。
   - 新增 `sim2real/config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml`。
   - `launch_ee_tracking_7dof_with_hand_policy.sh` 和 `launch_ee_tracking_7dof_with_hand_sim.sh` 都支持 `EXTRA_OVERLAY_CONFIG`。
   - 这样同一份 overlay 可以同时作用于 policy 侧 IK / 增益和 sim 侧阻尼 / 摩擦参数。

7. 修复 control file 写入覆盖问题。
   - 手部开合状态写入 control file 时会先合并已有 JSON，再更新自己的字段。
   - 避免右手开合状态把 `policy_started`、弹力绳释放等自动流程字段覆盖掉。

## 评测配置

两组评测都使用同一目标范围、同一随机种子和同一自动化流程：

```bash
cd /home/lavine/project/FALCON/sim2real
```

基线：

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=45 \
SEED=42 \
ELASTIC_LENGTH=-0.05 \
TRACKING_IK_SPEED_FACTOR=0.02 \
LOG_FILE=/tmp/falcon_ee_tracking_with_hand_baseline_metrics.csv \
MARKER_FILE=/tmp/falcon_ee_tracking_with_hand_baseline_markers.json \
SIM_STATUS_FILE=/tmp/falcon_sim_status_with_hand_baseline.json \
SIM_CONTROL_FILE=/tmp/falcon_sim_control_with_hand_baseline.json \
./launch_ee_tracking_7dof_with_hand_auto.sh
```

优化版：

```bash
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=45 \
SEED=42 \
ELASTIC_LENGTH=-0.05 \
EXTRA_OVERLAY_CONFIG=config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml \
LOG_FILE=/tmp/falcon_ee_tracking_with_hand_tuned_full_metrics.csv \
MARKER_FILE=/tmp/falcon_ee_tracking_with_hand_tuned_full_markers.json \
SIM_STATUS_FILE=/tmp/falcon_sim_status_with_hand_tuned_full.json \
SIM_CONTROL_FILE=/tmp/falcon_sim_control_with_hand_tuned_full.json \
./launch_ee_tracking_7dof_with_hand_auto.sh
```

分析命令：

```bash
cd /home/lavine/project/FALCON
/home/lavine/miniconda3/envs/fcreal/bin/python \
  sim2real/tools/analyze_ee_tracking_metrics.py \
  /tmp/falcon_ee_tracking_with_hand_baseline_metrics.csv \
  --steady_after_s=2.0 \
  --report sim2real/eval_outputs/ee_tracking_with_hand_baseline_report.md \
  --plot sim2real/eval_outputs/ee_tracking_with_hand_baseline_report.png \
  --title 带手末端跟踪基线评测

/home/lavine/miniconda3/envs/fcreal/bin/python \
  sim2real/tools/analyze_ee_tracking_metrics.py \
  /tmp/falcon_ee_tracking_with_hand_tuned_full_metrics.csv \
  --steady_after_s=2.0 \
  --report sim2real/eval_outputs/ee_tracking_with_hand_tuned_full_report.md \
  --plot sim2real/eval_outputs/ee_tracking_with_hand_tuned_full_report.png \
  --title 带手末端跟踪优化评测
```

## 结果

稳态窗口定义为每个目标点切换后 2 秒之后。单位为 cm，速度单位为 m/s。

| 指标 | 基线 | 优化版 | 变化 |
| --- | ---: | ---: | ---: |
| overall mean | 5.10 | 3.61 | -29.1% |
| overall RMS | 6.90 | 5.31 | -23.0% |
| overall P90 | 11.71 | 7.73 | -34.0% |
| overall P95 | 15.36 | 12.45 | -19.0% |
| steady mean | 2.48 | 2.03 | -18.1% |
| steady RMS | 2.65 | 2.10 | -20.7% |
| steady P90 | 3.51 | 2.61 | -25.7% |
| steady P95 | 4.08 | 2.64 | -35.1% |
| steady max | 5.41 | 2.84 | -47.4% |
| final median | 2.27 | 2.15 | -5.2% |
| IK steady mean | 0.84 | 0.08 | -89.9% |
| servo steady mean | 2.13 | 2.01 | -5.8% |
| steady speed P95 | 0.035 | 0.005 | -85.1% |

结论：

- 目标切换后的瞬态误差明显下降，overall mean、P90、P95 均改善。
- 稳态误差也有改善，尤其是 steady P95 从 4.08 cm 降到 2.64 cm，说明晃动和偶发偏大误差被压下来了。
- IK 误差已经从 0.84 cm 降到 0.08 cm，IK 侧基本不再是主要瓶颈。
- servo error 只从 2.13 cm 降到 2.01 cm，说明当前剩余稳态误差主要来自实际关节跟随、手部模型动力学和上肢 residual 叠加。
- steady speed P95 从 0.035 m/s 降到 0.005 m/s，说明末端稳态晃动显著减小。

## 参数解释

当前 tuned overlay：

```yaml
tracking_ik_speed_factor: 0.04
tracking_ik_filter_weights: [0.50, 0.25, 0.15, 0.10]
tracking_ik_translational_weight: 80.0
tracking_ik_regularization_weight: 0.015
tracking_ik_smooth_weight: 0.08
tracking_ik_include_rotation: false
tracking_upper_tau_ff_scale: 0.20
tracking_upper_tau_ff_clip: 2.5
policy_joint_damping: 0.08
wrist_joint_frictionloss: 0.15
```

关节增益只对左右手腕三个关节做缩放：

- wrist Kp scale：2.0
- wrist Kd scale：3.0

这些值是当前仿真验证用的保守参数，不应直接视为真机最终参数。真机上尤其要重新评估腕关节增益、力矩前馈和 residual 的交互。

## 后续建议

1. 保持上肢 residual，但在评测中继续记录 `servo_error_m`。
   - residual 是抗扰动能力的一部分，不能直接取消。
   - 但在论文和实验中应该明确说明：纯末端位置精度和抗扰动 residual 存在权衡。

2. 若要继续降低稳态误差，优先做 joint-level 跟随诊断。
   - 记录右臂 7 个关节的 `q_ref - q_actual`。
   - 区分肩肘误差和腕关节误差。
   - 判断剩余 2 cm 是否主要来自 wrist、shoulder/elbow，还是 residual 带来的系统性偏置。

3. 继续优化时不要盲目提高全臂 Kp。
   - 全身 policy 对原动力学比较敏感。
   - 当前更合理的方向是只调 wrist、小比例前馈、目标滤波和仿真阻尼。

4. 阀门任务前建议增加一条圆轨迹评测。
   - 先不 weld 阀门，让末端沿 base 坐标系或阀门坐标系圆周目标运动。
   - 记录径向误差、切向相位误差和末端速度。
   - 这比随机点更接近后续拧阀门需求。

