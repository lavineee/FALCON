# FALCON 6D Wrench Training Audit

日期：2026-04-27

本报告只审计训练代码和方案设计，不修改训练代码、不启动训练、不触碰当前 MuJoCo 阀门 demo。

## 1. 当前训练入口地图

### 训练入口

- `humanoidverse/train_agent.py`
  - `main(config)`
  - Hydra 入口：`@hydra.main(config_path="config", config_name="base", version_base="1.1")`
  - 主要流程：加载 Hydra config -> `pre_process_config(config)` 计算 observation 维度 -> `instantiate(config.env)` 创建 env -> `instantiate(config.algo)` 创建 PPO -> `algo.setup()` -> 可选 `algo.load()` -> `algo.learn()`。
  - IsaacGym 会在入口中先 `import isaacgym`，再 import torch。

- `humanoidverse/eval_agent.py`
  - `main(override_config)`
  - Hydra 入口：`@hydra.main(config_path="config", config_name="base_eval")`
  - 主要流程：从 checkpoint 旁边读取 `config.yaml`，merge eval overrides，创建 env/algo，加载 checkpoint，调用 `export_multi_agent_decouple_policy_as_onnx()` 导出 ONNX，然后进入 `algo.evaluate_policy()`。

### 训练命令示例

README 中 G1 force-only baseline：

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
+robot=g1/g1_29dof_waist_fakehand \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma \
num_envs=4096 \
project_name=g1_29dof_falcon \
experiment_name=g1_29dof_falcon \
+opt=wandb \
obs.add_noise=True \
env.config.fix_upper_body_prob=0.3 \
robot.dof_effort_limit_scale=0.9 \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.0001
```

已有 force+torque 尝试配置的组合形态：

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_torque_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
+robot=g1/g1_29dof_waist_fakehand \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_diff_force_torque_history_wolinvel_ma \
num_envs=4096
```

注意：上面的 force+torque 配置默认 `env.config.apply_torque_in_physics_step=False`，因此不是实际施加 torque 的训练。

### 当前可用 Hydra config group

- `algo`
  - `ppo_decoupled_wbc_ma`
- `env`
  - `base_task`
  - `legged_base`
  - `decoupled_locomotion_stand_height_waist_wbc_ma_diff_force`
  - `decoupled_locomotion_stand_height_waist_wbc_ma_diff_force_torque`
- `exp`
  - `base_exp`
  - `legged_base`
  - `decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env`
  - `decoupled_locomotion_stand_height_waist_wbc_diff_force_torque_ma_ppo_ma_env`
- `obs`
  - `legged_obs`
  - `dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma`
  - `dec_loco/g1_29dof_obs_diff_force_torque_history_wolinvel_ma`
  - `dec_loco/t1_29dof_obs_diff_force_history_wolinvel_ma`
  - `dec_loco/t1_29dof_obs_diff_force_torque_history_wolinvel_ma`
- `rewards`
  - `dec_loco/reward_dec_loco_stand_height_ma_diff_force`
- `robot`
  - `g1/g1_29dof_waist_fakehand`
  - `t1/t1_29dof_waist_wrist`
- `domain_rand`
  - `domain_rand_rl_gym`
  - `NO_domain_rand`
- `simulator`
  - `isaacgym`
  - `isaacsim`
  - `mujoco`
  - `genesis`
- `terrain`
  - `terrain_base`
  - `terrain_locomotion`
  - `terrain_locomotion_plane`
- `opt`
  - `wandb`
  - `record`

## 2. PPO / dual-agent 结构

### 关键文件

- `humanoidverse/config/algo/ppo_decoupled_wbc_ma.yaml`
  - `_target_`: `humanoidverse.agents.decouple.ppo_decoupled_wbc_ma.PPOMultiActorCritic`
  - 定义 `actor_lower_body`、`actor_upper_body` 和共享结构的 `critic`。

- `humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py`
  - `PPOMultiActorCritic`
  - `self.keys = robot.body_keys`，G1 默认为 `["lower_body", "upper_body"]`。
  - 为每个 key 创建独立 actor 和独立 critic。
  - rollout 时两个 actor 都读取同一个 `obs_dict["actor_obs"]`。
  - 每个 critic 都读取同一个 `obs_dict["critic_obs"]`。
  - env step 前把 lower/upper action 拼接：
    `torch.cat([actions[key] for key in self.keys], dim=1)`。

### G1 action 拆分

文件：`humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml`

- `actions_dim: 29`
- `lower_body_actions_dim: 15`
- `upper_body_actions_dim: 14`
- lower-body agent 输出 15 维：
  - 左右腿 12 DoF
  - waist yaw/roll/pitch 3 DoF
- upper-body agent 输出 14 维：
  - 左臂 7 DoF
  - 右臂 7 DoF
- policy action 总维度仍是 29。

### observation 共享情况

文件：

- `humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml`
- `humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_torque_history_wolinvel_ma.yaml`
- `humanoidverse/envs/legged_base_task/legged_robot_base_ma.py`

actor obs 包含 whole-body proprioception：

- `base_ang_vel`
- `projected_gravity`
- locomotion / stand / waist / base-height commands
- `ref_upper_dof_pos`
- `dof_pos` 29
- `dof_vel` 29
- `actions` 29

G1 actor obs 单帧维度为 115，history length 为 5，因此 ONNX actor input 为 575。force+torque 配置没有改变 actor obs。

critic obs force-only 单帧维度为 128，force+torque 单帧维度为 134。critic history length 为 1。

实现细节：训练和 sim2real 的 observation 拼接都会对 obs 名称做 `sorted(...)`，因此运行时顺序不是 YAML 中书写顺序，而是字母序。这个行为训练和部署一致，但新增 obs 字段时必须同步所有维度和导出/部署配置。

## 3. 当前 FALCON 训练逻辑

### upper-body target 是什么

当前训练不是直接追 EE pose，而是追 upper-body joint reference。

关键文件：`humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py`

- `_pre_compute_observations_callback()` 从 motion library 读取 `motion_res["dof_pos"]`。
- `self.ref_upper_dof_pos = ref_joint_pos[:, self.upper_dof_indices]`。
- 若 `residual_upper_body_action=True`，`_compute_torques()` 会把 upper-body actor action 当作残差，叠加到 `ref_upper_dof_pos - default_dof_pos` 上。

结论：upper-body target 是 joint target，不是 EE pose。EE 只在 reward 中有 acceleration penalty，当前没有 EE pose tracking reward。

### actor / critic observation 区别

- actor 不含真实 external force 或 torque。
- critic force-only 含 `left_ee_apply_force`、`right_ee_apply_force`。
- critic force+torque 配置额外含 `left_ee_apply_torque`、`right_ee_apply_torque`。
- force/torque obs 返回的是旋转到 base frame 的向量；实际 IsaacGym 施加张量是 ENV_SPACE。

### reward 约束项

reward 配置：`humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml`

root stability / base 姿态：

- `penalty_orientation`
- `penalty_torso_orientation`
- `penalty_ang_vel_xy`
- `penalty_ang_vel_xy_torso`
- `tracking_stance_base_height`
- `tracking_walk_base_height`
- `base_height`
- `termination`

lower-body locomotion：

- `tracking_lin_vel_x`
- `tracking_lin_vel_y`
- `tracking_ang_vel`
- `feet_air_time`
- `penalty_feet_height`
- `penalty_feet_swing_height`
- `feet_heading_alignment`
- `penalty_contact`
- `penalty_contact_no_vel`
- `penalty_stance_root`
- `penalty_stance_tap_feet`
- `penalty_stance_symmetry`
- `penalty_ankle_roll`
- `penalty_negative_knee_joint`

upper-body joint tracking：

- `tracking_upper_body_dofs`

EE 相关：

- `penalty_ee_lin_acc`
- `penalty_ee_ang_acc`

torque / action smoothness：

- `penalty_lower_body_torques`
- `penalty_upper_body_torques`
- `penalty_lower_body_action_rate`
- `penalty_upper_body_action_rate`
- `penalty_lower_body_dof_acc`
- `penalty_upper_body_dof_acc`

joint limit / torque limit：

- `limits_lower_body_dof_pos`
- `limits_upper_body_dof_pos`
- `limits_upper_body_dof_vel`
- `limits_upper_body_torque`
- config 中有 lower-body vel/torque limit 项但当前 reward group 注释掉了部分 lower-body 项。

fall termination：

- `terminate_by_gravity`
- `terminate_by_low_height`
- `terminate_when_motion_end`
- 可选 `terminate_when_close_to_dof_pos_limit`
- 可选 `terminate_when_close_to_dof_vel_limit`
- 可选 `terminate_when_close_to_torque_limit`
- 可选 `terminate_when_low_upper_dof_tracking`

### domain randomization

配置：`humanoidverse/config/domain_rand/domain_rand_rl_gym.yaml`

- `push_robots`: 默认 false，若开启会随机设置 root xy velocity。
- `randomize_link_mass`: true，`link_mass_range: [0.9, 1.2]`。
- `randomize_pd_gain`: true，`kp_range/kd_range: [0.9, 1.1]`。
- `randomize_friction`: true，`friction_range: [0.25, 1.25]`。
- `randomize_base_mass`: true，`added_mass_range: [-1., 3.]`。
- `randomize_torque_rfi`: false，若开启会给 dof torque 加 RFI。
- `randomize_ctrl_delay`: true，delay step `[0, 1]`，lower/upper body 都启用。

## 4. 当前 force disturbance 实现

### 关键文件

- `humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml`
- `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.py`
- `humanoidverse/simulator/isaacgym/isaacgym.py`
- `humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml`

### force config 字段

- `max_force_estimation`
- `update_apply_force_phase`
- `use_lpf`
- `force_filter_alpha`
- `zero_tapping_xy_force`
- `zero_force_prob`
- `random_force_prob`
- `apply_force_x_range`
- `apply_force_y_range`
- `apply_force_z_range`
- `apply_force_pos_ratio_range`
- `lower_body_force_compensation`
- `apply_force_in_physics_step`
- `apply_force_compensation_in_physics_step`
- `randomize_force_duration`
- `only_apply_z_force_when_walking`
- `only_apply_resistance_when_walking`
- reward curriculum:
  - `force_scale_curriculum`
  - `force_scale_initial_scale`
  - `force_scale_up/down_threshold`
  - `force_scale_up/down`
  - `force_scale_min/max`

### 施加位置和坐标系

- body/link：`robot.force_control.left_hand_link` 和 `right_hand_link`，G1 当前是 `left_rubber_hand` / `right_rubber_hand`。
- 真实施加 API：`IsaacGym.apply_rigid_body_force_at_pos_tensor()`
- 底层 API：`gym.apply_rigid_body_force_at_pos_tensors(..., gymapi.ENV_SPACE)`
- 施加点：`apply_force_pos_tensor`，在 `_pre_compute_observations_callback()` 中由 hand link 到 extended fake hand point 的插值位置决定。
- force 方向/大小：
  - 若 `max_force_estimation=True`，使用 hand link Jacobian 估计 arm joint torque limit 下每轴可行 force bound。
  - 再乘 `apply_force_scale` curriculum。
  - 最后 clip 到 `apply_force_{x,y,z}_range`。
  - walking 时可只保留 z force 或把 xy 投影成反向行走阻力。
- 坐标系：
  - 施加张量是 ENV_SPACE。
  - critic 中记录的 `left_ee_apply_force` / `right_ee_apply_force` 是 `quat_rotate_inverse(base_quat, force)`，即 base frame。

### force curriculum 和滤波

- force phase 是 triangular / piecewise ramp：`phase = abs(remainder(ts, 2.0) - 1.0)`。
- 每个 env 的 duration 在 `randomize_force_duration` 中采样，单位是 policy step。
- `use_lpf=True` 时只 low-pass filter min/max bound，不是直接 filter 最终 applied force。
- `force_scale_curriculum` 根据 episode length 上下调 `apply_force_scale`。

### 记录到 privileged obs

force-only critic obs 包含 left/right EE applied force，共 6 维；actor 不含 force。

## 5. 当前 torque disturbance 实现和问题

### 写在哪里

- 配置：
  - `humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force_torque.yaml`
  - `humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_torque_history_wolinvel_ma.yaml`
  - T1 同名 obs 配置也有 torque obs。
- 环境：
  - `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force_torque.py`
- simulator API：
  - `humanoidverse/simulator/isaacgym/isaacgym.py::apply_rigid_body_torque_tensor`

### 是真正 external torque 吗

代码层面可以是真正 external torque：

- `apply_torque_tensor` shape 为 `(num_envs, num_bodies, 3)`。
- `_apply_force_in_physics_step()` 中若 `apply_torque_in_physics_step=True`，调用 `self.simulator.apply_rigid_body_torque_tensor(self.apply_torque_tensor)`。
- IsaacGym wrapper 优先使用 `apply_rigid_body_torque_tensors`，否则使用 `apply_rigid_body_force_tensors(zero_force, torque, ENV_SPACE)`。

但当前 torque env config 默认：

```yaml
apply_torque_in_physics_step: False
```

并且 `_physics_step()` 中如果该字段为 false，会执行：

```python
self.left_ee_apply_torque.zero_()
self.right_ee_apply_torque.zero_()
self.apply_torque_tensor.zero_()
```

因此默认 force+torque 配置不会实际施加 torque，critic 也看到 zero torque。这是当前实现最核心的问题。

### torque 施加到哪里

- body/link：同 force，`left_rubber_hand` / `right_rubber_hand`。
- 不是 root，不是 object，不是 joint torque。
- 若启用，是 rigid body external torque。

### 单位和坐标系

- 配置名是 `apply_torque_x/y/z_range`，代码注释称 torque / moment，但没有系统性写明单位。按 IsaacGym rigid body torque API 应理解为 N*m。
- 施加 API 使用 `gymapi.ENV_SPACE`，所以 applied torque 是 env/global frame。
- critic obs 中 torque 被旋转到 base frame。

### torque 采样和 curriculum

已有尝试：

- torque 使用和 force 相同的 phase 变量。
- 有独立 `apply_torque_scale` 初始化逻辑。
- 有 `_update_torque_scale_curriculum()`。

问题：

- reward config `reward_dec_loco_stand_height_ma_diff_force.yaml` 没有定义 `torque_scale_curriculum`、`torque_scale_initial_scale`、`torque_scale_up_threshold` 等字段。除非命令行手动加完整字段，否则 torque curriculum 不会开启。
- torque duration 共享 `randomize_force_duration`，没有独立 duration / hold / resample 配置。
- `random_torque_prob` 读取了 config，但后续没有被使用。
- `left_torque_xyz_scale` / `right_torque_xyz_scale` 初始化了，但主要计算使用的是共享 `torque_xyz_scale`，左右独立 scale 未发挥作用。
- torque feasible bound 使用 angular Jacobian 的逐轴倒数近似，和 force 逻辑一样是非常粗的 axis-wise bound；没有处理 6D wrench 同时作用时的联合 torque saturation。
- torque 采样最后又加了 `torch.rand(...)` 的正偏置噪声，会让本应对称的 torque 采样出现偏置。
- `use_lpf` 滤的是 torque min/max bound，而不是最终 applied torque，仍可能出现每次 phase / bound 改变时的突变。

### 是否进入 privileged observation

force+torque obs 配置把 `left_ee_apply_torque` / `right_ee_apply_torque` 放进 critic obs，actor 不含 torque。

但默认不施加 torque 时，env 会清零这些值，所以 critic 实际看到 0。

### 日志记录

- PPO 只记录 `Env/apply_force_scale`，没有记录 `apply_torque_scale`。
- 没有记录 applied force norm / torque norm、per-axis torque、wrench enable mask、fall under wrench 等指标。
- debug visualization 有 torque line，但只在 viewer/debug 里有效，不是训练日志。

### 为什么之前 torque 训练效果可能不好

不是简单“参数不合适”，主要来自代码结构：

1. 默认 torque 根本没有施加。`apply_torque_in_physics_step=False` 且 `_physics_step()` 会清零 torque tensor，导致所谓 torque 配置退化为 force-only。
2. reward config 缺 torque curriculum 字段。开启 torque 施加后，如果没有手动补齐 curriculum，可能从 scale=1 和 `[-20,20] N*m` 直接开始，过猛。
3. torque 与 force 没有 6D 联合限幅。force bound 和 torque bound 分别估算，叠加时可能超过 wrist/arm/waist torque 能力。
4. torque 低通不是对最终 wrench 做 filter。实际扰动仍可能出现高频/不连续变化。
5. applied torque 没有训练日志。训练不稳定时无法区分 torque 太大、force 太大、动作饱和、摔倒过多还是 obs 维度/scale 问题。
6. 没有 EE orientation/wrist stability reward。外部 moment 主要破坏手腕姿态和上肢角速度，但 reward 仍以 joint position tracking + EE acceleration penalty 为主，对 moment robustness 的学习信号不够直接。
7. 施加坐标系和 critic 坐标系不同。施加为 ENV_SPACE，critic 记录为 base frame，这是合理选择，但报告/配置没有明确，容易误调。

## 6. with-hand 模型训练可行性

### 当前三个模型

1. 原始 G1 29DoF 训练/部署模型
   - URDF：`humanoidverse/data/robots/g1/g1_29dof.urdf`
   - 29 revolute joints。
   - fixed `left_rubber_hand` / `right_rubber_hand`。

2. 当前 IsaacGym 训练实际使用模型
   - Hydra robot config：`humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml`
   - `robot.asset.robot_type: g1_29dof_fakehand`
   - IsaacGym asset：`humanoidverse/data/robots/g1/g1_29dof_fakehand.urdf`
   - 29 revolute joints。
   - `left_hand_palm_joint` / `right_hand_palm_joint` 是 fixed 且带 `dont_collapse="true"`，用于保留 `left_rubber_hand` / `right_rubber_hand` body。
   - 这是当前训练最稳妥的 fake hand 模型。

3. 当前 MuJoCo demo with-hand 模型
   - scene：`humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand_tracking.xml` 等。
   - include：`humanoidverse/data/robots/g1/g1_29dof_old_freebase_with_hand.xml`
   - IK URDF：`humanoidverse/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf`
   - MJCF：29 policy joints + 14 finger joints + free base，motor 数为 49（6 base motors + 43 joint motors）。
   - sim2real overlay 通过 `policy_motor_joint_names` 只把 29 个 policy joints 暴露给 ONNX policy，finger joints 由单独 PD open/close 控制。

### with-hand 模型能否直接用于训练

不能直接替换当前 IsaacGym 训练 asset。

原因：

- `BaseTask._load_assets()` assert `num_dof == robot.actions_dim`。
- `IsaacGym.load_assets()` assert：
  - asset dof count 等于 `len(robot.dof_names)`
  - asset body count 等于 `len(robot.body_names)`
  - asset dof names 等于 config dof names
  - asset body names 等于 config body names
- `g1_29dof_with_hand_rev_1_0.urdf` 有 43 revolute joints，当前 config 只有 29 action/dof。
- 直接使用 with-hand URDF 会导致 action dim、obs dim、PD gain、dof limit、reward index、ONNX export、sim2real mapping 全部需要重做。

### action / observation dim 是否会变化

- 保持 `g1_29dof_fakehand.urdf`：不变，action dim 29，actor obs 575。
- 直接训练 `g1_29dof_with_hand_rev_1_0.urdf`：会变成 43 DoF，除非额外实现“finger joints 由非-policy PD 控制且不进入 policy action”的训练环境分流。当前 BaseTask 不支持这种分流。

### hand actuator 是否应该进入 policy

第一阶段/最小实现不建议进入 policy。

理由：

- 当前训练目标是底层 whole-body / upper-body tracker 的 6D wrench robustness，不是训练抓握策略。
- finger action 加入 policy 会破坏现有 29D ONNX / sim2real / MuJoCo demo 对齐。
- MuJoCo with-hand demo 已经证明 29D policy + hand PD 控制链路可运行。

### 如果手指不进 policy，是否可以存在

可以，但建议分阶段：

- 最小方案：训练继续使用 `g1_29dof_fakehand.urdf` 的 `left_rubber_hand` / `right_rubber_hand`，不引入 finger joints。
- 后续方案：新增 IsaacGym 专用 with-hand-fixed/finger-PD env，finger 不进 policy，但这需要较多环境分流代码，不是最小改动。

### 推荐 EE link/body

训练最小改动：继续使用 `left_rubber_hand` / `right_rubber_hand`。

原因：

- 当前训练 config、body_names、force_control、Jacobian、reward、privileged obs 全都围绕这两个 body。
- `g1_29dof_fakehand.urdf` 保留这两个 body，action dim 仍为 29。
- sim2real with-hand overlay 虽然 `left_hand_link_name/right_hand_link_name` 用 wrist yaw link，并用 local offset 表示掌心/抓取中心，但 ONNX policy 仍是 29D。

如果后续要更接近阀门接触点，建议新增“disturbance application body/point”和“tracking body/point”分离：

- policy/training EE body：`right_rubber_hand`
- disturbance application point：从 wrist yaw 或 rubber hand 到 palm/grasp center 的 local/world offset
- sim2real IK EE：`R_ee` frame，由 `G1_29_WithHandArmIK` 在 `right_wrist_yaw_joint` 上加 local offset。

不建议第一步改成真实 `right_hand_palm_link`，因为当前 IsaacGym 训练 asset 没有这个 body。

### 改 EE link 会影响的代码

- robot config：
  - `robot.force_control.left_hand_link/right_hand_link`
  - `robot.body_names`
  - `robot.motion.motion_tracking_link`
  - `robot.motion.extend_config`
- env：
  - hand link index
  - Jacobian 索引
  - force/torque tensor body index
  - EE acceleration reward
  - debug visualization
- reward：
  - upper-body/EE acceleration
  - 未来 EE pose/orientation tracking
- sim2real：
  - ONNX input/output dim
  - `policy_motor_joint_names`
  - IK frame `L_ee/R_ee`
  - with-hand bridge policy joint maps
  - MuJoCo actuator mapping

## 7. 6D wrench curriculum 设计

目标：训练底层 tracker 在 EE 6D external wrench 下维持 root stability、lower-body balance、upper-body joint/EE tracking、action smoothness 和 torque safety。不是直接训练转阀门任务。

### 推荐新增 config 字段

建议新增独立 config group，而不是覆盖 force-only baseline：

- `humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.yaml`
- `humanoidverse/config/obs/dec_loco/g1_29dof_obs_6d_wrench_history_wolinvel_ma.yaml`
- `humanoidverse/config/exp/decoupled_locomotion_stand_height_waist_wbc_6d_wrench_ma_ppo_ma_env.yaml`

建议字段：

```yaml
env:
  config:
    wrench:
      enabled: true
      mode: force_torque        # none | force_only | torque_only | force_torque
      apply_in_physics_step: true
      apply_links: ["left_rubber_hand", "right_rubber_hand"]
      force_range_n:
        x: [-40.0, 40.0]
        y: [-40.0, 40.0]
        z: [-50.0, 5.0]
      torque_range_nm:
        x: [-2.0, 2.0]
        y: [-2.0, 2.0]
        z: [-2.0, 2.0]
      force_scale_initial: 0.1
      force_scale_min: 0.0
      force_scale_max: 1.0
      torque_scale_initial: 0.0
      torque_scale_min: 0.0
      torque_scale_max: 1.0
      torque_scale_start_after_force_scale: 0.3
      resample_duration_steps: [150, 250]
      piecewise_constant: true
      use_lpf: true
      lpf_alpha: 0.05
      zero_force_prob: [0.25, 0.25, 0.25]
      zero_torque_prob: [0.5, 0.5, 0.5]
      random_application_point: false
      application_point_ratio_range: [0.0, 1.0]
      only_apply_z_force_when_walking: false
      only_apply_resistance_when_walking: true
      only_apply_z_torque_when_walking: false
      torque_limit_aware_bound: true
      joint_limit_margin: 0.8
      max_fall_rate_guard: 0.7
```

为了兼容旧字段，可以暂时桥接：

- `apply_force_in_physics_step`
- `apply_torque_in_physics_step`
- `apply_force_x/y/z_range`
- `apply_torque_x/y/z_range`

但建议新代码内部以 `wrench.*` 为主，旧字段只在 force-only env 中保留。

### Hydra 参数和 ablation

支持命令行覆盖：

```bash
env.config.wrench.mode=none
env.config.wrench.mode=force_only
env.config.wrench.mode=torque_only
env.config.wrench.mode=force_torque
env.config.wrench.torque_scale_initial=0.0
env.config.wrench.torque_range_nm.x=[-1.0,1.0]
env.config.wrench.torque_range_nm.y=[-1.0,1.0]
env.config.wrench.torque_range_nm.z=[-1.5,1.5]
```

四个 ablation：

- no wrench：force=0, torque=0
- force-only：torque disabled，保持原 baseline 行为
- torque-only：force disabled，只施加 moment
- force+torque：同时施加 6D wrench

### external wrench 在哪里采样

建议集中在新 env 类中，例如：

- `_init_wrench_settings()`
- `_resample_wrench_settings(env_ids)`
- `_update_wrench_phase()`
- `_sample_ee_wrench()`

不要继续把 torque 代码复制在 force env 旁边。当前 force 和 torque 文件重复大量代码，建议保留 force-only env 不动，新建 6D wrench env 统一实现。

### external wrench 在哪里施加

IsaacGym 最直接路径：

- force：`apply_rigid_body_force_at_pos_tensor(force_tensor, pos_tensor)`
- torque：`apply_rigid_body_torque_tensor(torque_tensor)`
- 坐标系统一使用 `ENV_SPACE`，并在 config/report 中明确。
- privileged obs 使用 base frame：
  - `left_ee_apply_force_base`
  - `right_ee_apply_force_base`
  - `left_ee_apply_torque_base`
  - `right_ee_apply_torque_base`

### privileged observation 如何扩展

actor：

- 默认不加入真实 wrench。
- 保持 actor obs 维度 575，便于 ONNX/sim2real 对齐。

critic：

- force-only：保留 left/right force 6 维。
- 6D wrench：left/right force+torque 共 12 维。
- 增加 obs scales：
  - force scale 建议 `0.05` 或 `0.1`
  - torque scale 建议 `0.1` 到 `0.2`，取决于 torque range

### reward 是否需要新增

建议先最小新增，不改 force-only reward：

1. EE orientation / wrist stability
   - 如果当前训练没有 EE orientation reference，可以先惩罚 EE angular velocity / angular acceleration，并保留 joint tracking。
   - 后续若能从 motion/fixed target 得到 wrist orientation reference，再加 `tracking_ee_orientation`。

2. torque saturation / wrist saturation
   - 强化 `limits_upper_body_torque` 或新增 wrist-only torque saturation penalty。

3. action smoothness
   - 保持并可略增强 `penalty_upper_body_action_rate`、`penalty_upper_body_dof_acc`。

4. fall guard
   - 不建议一开始把 termination 设得过严；先用日志观察 fall rate。

### curriculum scale 如何更新

建议拆 force 和 torque：

- force scale 复用现有 episode-length curriculum。
- torque scale 独立，并延迟启动：
  - 初始 `0.0` 或 `0.05`
  - 当 force scale 或 average episode length 达到阈值后开始上调
  - torque 每次上调更慢，例如 `0.01`
- 可加入 fall-rate guard：
  - 如果最近 N 个 episode fall rate 过高，则暂停增加或下调 torque scale。

### 稳定性保护

- torque range 第一阶段建议小很多：例如 `[-1, 1]` 或 `[-2, 2] N*m`，不是当前 `[-20, 20] N*m`。
- 默认 `use_lpf=True`，并对最终 wrench 做 low-pass，而不是只滤 bound。
- piecewise-constant wrench：在 duration 内保持目标 wrench，再用 low-pass 平滑过渡。
- 6D joint-limit-aware bound：用组合 wrench 的 `J^T wrench` 检查 arm/wrist/waist torque margin，而不是 force/torque 分别逐轴估计。
- NaN guard：检查 sampled wrench、reward、obs、torques 是否 finite。
- 日志必须记录 force/torque norm 和 saturation。

### 如何保证 force-only baseline 不破坏

- 不修改 `decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.py`。
- 不修改原 force-only config。
- 新增 6D wrench env/config/obs/exp。
- 若需要复用 force 代码，先抽 helper，但第一版更建议复制并整理成新类，避免影响旧 baseline。
- smoke test 中必须验证 `wrench.mode=force_only` 且 torque=0 时 actor obs/action dim 与旧配置一致。

## 8. 需要修改的文件清单（后续实现阶段）

最小实现建议新增/修改：

- 新增 `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.py`
  - 统一 force+torque sampling/apply/logging。
- 新增 `humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.yaml`
  - 新 6D wrench env config。
- 新增 `humanoidverse/config/obs/dec_loco/g1_29dof_obs_6d_wrench_history_wolinvel_ma.yaml`
  - critic privileged obs 扩展 12D wrench。
- 新增 `humanoidverse/config/exp/decoupled_locomotion_stand_height_waist_wbc_6d_wrench_ma_ppo_ma_env.yaml`
  - 新 experiment shortcut。
- 修改 `humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml`
  - 只新增可选 torque curriculum / optional reward fields，默认不改变旧行为；或新增单独 reward config 更安全。
- 修改 `humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py`
  - 增加 `apply_torque_scale`、force/torque norm 等日志。
- 可能修改 `humanoidverse/simulator/isaacgym/isaacgym.py`
  - 保留 torque API wrapper；如果目标 IsaacGym build 不支持 torque API，需要实现 force-couple fallback。

不建议第一轮修改：

- `sim2real/*valve*`
- `sim2real/launch_valve_contact_turning_7dof_with_hand_auto.sh`
- 现有 ONNX / model / logs / checkpoints
- 现有 force-only env/config

## 9. 最小实现路线

1. 新建 6D wrench env 和 config，保持旧 force-only 文件不动。
2. 先实现 no-wrench / force-only / torque-only / force+torque 四种 mode，但默认 no torque。
3. torque 施加默认小范围、小 scale、final wrench low-pass。
4. critic obs 加 12D privileged wrench，actor obs 不变。
5. PPO 日志补齐 force/torque norm、scale、fall/reset、torque saturation。
6. 通过 smoke test 后，只跑极短 short training，不启动长训练。
7. 保留 ONNX actor input dim 575，确保 sim2real policy 仍能加载。

## 10. 风险点

- 直接用 `g1_29dof_with_hand_rev_1_0.urdf` 会把 DoF 从 29 变 43，当前训练环境会 assert fail。
- 当前 torque config 默认不施加 torque，容易误以为已训练 6D wrench。
- 当前 torque range `[-20,20] N*m` 对 5 N*m wrist joints 可能过大。
- force/torque 独立限幅不是 6D 联合限幅，可能超过 joint effort margin。
- actor obs 不变是部署友好的，但 critic-only wrench 对学习的帮助需要验证。
- 改 EE body 会连带影响 Jacobian、reward、motion extend body、sim2real IK frame 和 demo contact geometry。
- current observation 拼接按 `sorted(obs_config)`，新增字段改变 critic 输入顺序时必须只影响训练 critic，不影响 exported actor。
- `random_torque_prob` 当前未生效，不能依赖它做 ablation。
- torque API 是否在当前 IsaacGym build 存在需要 smoke test 验证。

## 11. Smoke test 命令

不启动长训练的静态/短步建议：

配置 compose / 维度检查：

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_6d_wrench_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
+robot=g1/g1_29dof_waist_fakehand \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_6d_wrench_history_wolinvel_ma \
num_envs=4 \
headless=True \
use_wandb=False \
algo.config.num_learning_iterations=0
```

100-1000 step smoke test 需要新增一个专用脚本或 Hydra flag，避免误触长训练。验收内容：

- env instantiate 成功。
- action dim = 29。
- actor obs = 575。
- critic obs = force-only 128 或 6D wrench 134。
- policy forward 无报错。
- reward / obs / torques all finite。
- no-wrench 行为与旧 force-only force=0 一致。
- torque 非零时 IsaacGym body 上有 applied torque tensor。
- 100-1000 step 不崩溃。

短训练测试：

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_6d_wrench_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
+robot=g1/g1_29dof_waist_fakehand \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_6d_wrench_history_wolinvel_ma \
num_envs=256 \
headless=True \
use_wandb=False \
algo.config.num_learning_iterations=5 \
env.config.wrench.mode=force_torque \
env.config.wrench.torque_scale_initial=0.0
```

注意：以上是后续实现后的命令模板；当前仓库还没有 `*_6d_wrench*` config。

## 12. 训练验收标准

### Smoke test

- 新 config 可 instantiate env。
- observation/action dim 正确。
- policy forward 无报错。
- reward 无 NaN。
- external wrench 能正确记录。
- force=0, torque=0 时行为与原配置一致。
- torque 非零时能看到 EE body 上存在扰动力矩。
- 100-1000 step 不崩溃。

### Short training test

- 训练能启动。
- loss/reward 不立即 NaN。
- 初期 fall rate 不高到无法学习。
- force scale 和 torque scale 按预期变化。
- 日志能区分 force 与 torque 大小。
- critic privileged obs 维度正确。

### Policy evaluation

固定 upper-body target，施加不同 wrench，比对：

- upper-body tracking error
- EE position error
- EE orientation error
- root tilt
- foot slip
- fall rate
- action smoothness
- joint torque saturation

对比对象：原 FALCON force-only policy。

### MuJoCo valve transfer evaluation

1. 导出 ONNX。
2. 替换 MuJoCo valve demo policy。
3. 测试 `+10,-20,+30`。
4. 测试单段 `+30` / `+45`。
5. 对比原 policy：
   - 阀门角度误差
   - EEF error
   - base approach
   - base yaw drift
   - contact force
   - soft connect 依赖程度
   - angle brake 依赖程度
   - 是否更少失稳

## 13. 后续给 ChatGPT 顾问判断的问题

1. 对 G1 29D fake-hand tracker，末端 moment 初始范围建议从 `[-1,1] N*m` 还是 `[-2,2] N*m` 起步？
2. 6D wrench 联合限幅应采用简单 `J^T wrench` scale-down，还是对每个 env 解一个小型 box-constrained projection？
3. torque disturbance 应使用 env/world frame 采样，还是 body/local hand frame 采样后转换到 ENV_SPACE？
4. 如果 actor 不看 wrench，仅 critic 看 privileged wrench，是否足以学到 robust behavior，还是应加入历史 proprioception 的隐式估计辅助？
5. EE orientation tracking 是否应在第一版加入，还是先只加 wrist angular velocity/acceleration penalty？
6. with-hand IsaacGym 训练是否值得做 finger-PD 分流，还是继续保持 fake-hand 训练 + MuJoCo with-hand transfer？
7. valve transfer 的目标是否应该只看稳定性和跟踪鲁棒性，还是需要加入接触反力矩统计作为 policy selection 指标？
