# G1 实机视觉接入阶段汇报简版

日期：2026-05-13

## 1. 本阶段目标

近期工作的主线是把阀门任务从 MuJoCo privileged 几何信息，逐步迁移到真实 G1 头部 D435I 的视觉输出。当前先不追求直接实机转阀门，而是先打通：

```text
G1 头部相机 -> 阀门/marker 几何估计 -> UDP 发送 -> 控制 PC 状态文件 -> real_vision provider -> approach / close / hold 状态机
```

控制侧统一消费 `center_base / grasp_base / axis_base / valid` 等字段，使仿真视觉、实机视觉和原始 GT 几何共用同一套阀门抓取状态机。

## 2. 已完成工作

### 2.1 仿真视觉接口和安全切换

- 增加 `valve_geometry_source: gt / vision_overlay_debug / vision_overlay / real_vision`，默认仍走 `gt`，避免影响原有 baseline。
- `vision_overlay_debug` 只做视觉与 GT 对比，不进入控制；只有 `vision_overlay + vision_override_control=true` 才允许视觉几何接管。
- 默认保持 `vision_override_angle=false`，视觉角度只进入 debug 字段，不提前替换转阀门闭环角度。
- close 阶段改为冻结 approach 阶段最近一次真正用于控制的视觉几何，避免 close transition 时 fallback 到 GT。
- 引入 `vision_close_latch_frame=world`，解决 base frame 移动后 latched 几何过期的问题。

### 2.2 仿真 RGB-D 弱标识视觉验证

- 在 MuJoCo 中基于 `head_camera` 渲染 RGB-D，使用彩色 marker 输出 `/tmp/falcon_valve_vision_status.json`。
- 输出字段包括 `pose_valid / grasp_valid / angle_valid / plane_aux_valid`，并保留中心、抓点、轴线、角度和质量指标。
- 为解决手遮挡 green grasp marker 后平面估计不稳的问题，新增不参与抓取的 `plane_aux` marker，pose/axis 连续性明显改善。
- 视觉抓点语义统一为 `raw_site`，继续复用原控制器的 radial inset 和 EE offset，避免视觉输出和 GT baseline 语义不一致。

### 2.3 视觉接管 approach 的短测结果

评估标准从严格接触拓扑 `PTIM`，收敛为“能否进入 turn、最终角度误差、base/foot 稳定性、slip 和接触力”等任务指标。

- GT geometry + tangent-only compensation，`+10 deg` 五次短测：
  - `entered_turn_rate = 5/5`
  - `angle_success_rate = 5/5`
  - 平均最终角度误差约 `0.43 deg`，最大约 `0.89 deg`
  - 平均 foot displacement 约 `0.0086 m`
- Vision overlay approach + tangent-only compensation，`+10 deg` 五条有效 vision-latched 样本：
  - `entered_turn_rate = 5/5`
  - `angle_success_rate = 3/5`
  - 平均最终角度误差约 `1.74 deg`，最大约 `2.97 deg`
  - 平均 base approach 约 `0.011 m`

结论：视觉几何已经可以安全接管 approach，并能进入后续 close/turn 验证；剩余主要问题更像是 turn angle tracking、接触驱动力一致性和阀门参数，而不是视觉接口本身。

### 2.4 G1 实机视觉节点

- 在 G1 本机 ROS1 Noetic 环境部署 D435I 视觉节点，使用系统 Python 3.8 和 `cv2.aruco`。
- 输入 topic：
  - `/camera/color/image_raw`
  - `/camera/aligned_depth_to_color/image_raw`
  - `/camera/color/camera_info`
- 当前使用 ArUco `DICT_4X4_50`、`id=0`、边长 `0.0765 m`。
- 节点以约 `15 Hz` 通过 UDP 发到控制 PC `192.168.123.222:5055`。
- 输出低维 JSON：`center_base / grasp_base / axis_base / quality / valid`。
- 增加标定工具链：
  - 采集 `T_camera_tag`
  - 人工填写 `T_base_tag`
  - 求解 `T_base_camera`
  - live sanity check 验证 base 坐标方向和误差

当前 `T_base_camera` 是基于当前 G1 头部 D435I 的粗标定，迁移到新机器人或真实接触前必须重新检查。

### 2.5 控制 PC real_vision 链路和安全联调

- 增加 `RealVisionGraspGeometryProvider`，读取 `/tmp/falcon_real_valve_vision_status.json`，转换成原状态机消费的几何字典。
- 增加 real runner：`run_valve_real_with_hand.py`，复用阀门抓取状态机，但关闭 MuJoCo-only 的 weld、angle lock、contact truth 和 sim completion。
- Phase 1 配置中禁用真实转阀门，只做 approach / close / hold。
- 真实 G1 上已验证：
  - G1 D435I 视觉 UDP 包能到控制 PC，`raw_valid=true`
  - 策略端能读取 `real_vision` 几何
  - 安全目标 override 后，状态机走通 `move_pregrasp -> approach_grasp -> close_hand -> hold_grasp -> done`
  - 空抓测试最终 EE error 约 `2.9 cm`
  - `close_stage_timeout` 在无实物接触的 air-grasp 测试中是预期现象
- 同步打通 Inspire 右手 DDS 控制链路，实机 runner 中使用 `inspire_dds` 后端；当前 G1 需要 `MODE_MACHINE=6` 才能正常响应控制。

## 3. 当前状态

目前已经完成从“视觉估计结果”到“真实 G1 控制状态机”的端到端闭环雏形：

```text
真实相机检测 marker
  -> UDP JSON
  -> 控制 PC 原子写状态文件
  -> real_vision provider
  -> 右手目标点 approach
  -> Inspire 手爪 close / hold / open
```

但当前实机结果仍属于安全链路验证：

- 真实视觉 raw packet 有效，但控制实际消费的是安全前方目标 override。
- 使用的是 ArUco marker，不是无标识阀门检测。
- 外参仍是粗标定，不能直接做真实接触。
- 实机转阀门尚未开启，视觉角度也尚未参与控制闭环。
- 当前没有真实阀门接触力/角度反馈，只验证了 approach 和手爪命令链路。

## 4. 主要风险和问题

- 相机外参误差会直接变成抓点误差，几厘米误差就可能导致空夹或碰撞。
- 真实 G1 base/腰部姿态变化后，固定 `T_base_camera` 可能不再准确；后续可能需要实时外参或姿态补偿。
- 当前 marker-based 方案适合作为第一阶段工程入口，但还不是通用阀门视觉。
- 实机 close/contact 阶段缺少可靠接触反馈，不能用 MuJoCo 的 strict topology 逻辑直接类比。
- 仿真里 vision-latched 能进入 turn，但 `+10 deg` 仍只有 `3/5` 达到 `±2 deg`，后续需要继续分析转动闭环和接触驱动力。

## 5. 下一步计划

1. 重新做 G1 D435I 外参标定：采集 6-10 个 marker 位姿，完成 live sanity check，把误差压到可接受范围。
2. 先取消安全 target override，只做真实视觉 approach，到达后停止，不闭手、不接触实物。
3. 固定真实阀门/marker 几何偏移，验证 `center_base / grasp_base / axis_base` 在多次运行中的稳定性。
4. 在人工监控和限幅下做真实 close/hold，不启用 turn。
5. 接入真实阀门角度来源或视觉角度验证，再逐步尝试小角度转动。

一句话总结：本阶段已经把 G1 实机视觉从“能看到 marker”推进到“能进入控制状态机并驱动真实机器人安全 approach/手爪链路”，下一步的关键是外参精标定和从安全目标过渡到真实目标。
