# 2026-04-28 6D Wrench Training 工作日报

## 今日目标

围绕 FALCON 训练代码进入 6D wrench robust policy 微调阶段，重点完成：

- 确认可用的 trained FALCON actor checkpoint；
- 实现并验证 actor-only warm-start；
- 完成 256 env 小规模 warm-start fine-tune 验证；
- 为后续服务器训练准备同步和 tmux 工作流；
- 保持 MuJoCo valve demo、原 force-only env/config、reward、ONNX 不受影响。

## 代码与配置整理

新增独立 6D wrench training path：

- `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.py`
- `humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.yaml`
- `humanoidverse/config/obs/dec_loco/g1_29dof_obs_6d_wrench_history_wolinvel_ma.yaml`
- `humanoidverse/config/exp/decoupled_locomotion_stand_height_waist_wbc_6d_wrench_ma_ppo_ma_env.yaml`

关键约束保持：

- robot: `g1/g1_29dof_waist_fakehand`
- actor obs: 575
- critic obs: 134
- action dim: 29
- EE body: `left_rubber_hand`, `right_rubber_hand`
- actor 不接收真实 wrench
- critic privileged obs 记录 force/torque wrench
- 不改原 force-only env/config
- 不改 reward
- 不改 MuJoCo valve demo

实现 actor-only warm-start loader：

- 修改 `humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py`
- 只加载 `actor_model_state_dict.lower_body`、`actor_model_state_dict.upper_body` 和 actor `std`
- 不加载旧 critic
- 不加载旧 optimizer
- 自动报告 loaded / skipped / missing / unexpected keys

保留并修正 curriculum：

- `curriculum_mode: adaptive | adaptive_floor | fixed`
- `fixed_force_scale: 0.10`
- `fixed_torque_scale: 0.05`
- `force_scale_min_after_warmup: 0.05`
- `torque_scale_min_after_warmup: 0.03`
- `disable_force_down_until_iterations: 100`
- `disable_torque_down_until_iterations: 100`

## Checkpoint 审计

确认 `/home/lavine/project/FALCON/model_10000.pt` 是有效 trained FALCON actor checkpoint：

- checkpoint `iter = 10000`
- optimizer step = 200000
- lower actor shape: `575 -> 512 -> 256 -> 128 -> 15`
- upper actor shape: `575 -> 512 -> 256 -> 128 -> 14`
- actor `std` shape 匹配
- critic 第一层为 `(512, 128)`，与当前 6D critic `(512, 134)` 不匹配，因此跳过 critic 是正确选择

对比早先的 `logs/g1_29dof_falcon_local/.../model_0.pt`：

- 目录名包含 `smoke_test`
- checkpoint `iter = 0`
- TensorBoard 只有少量 step
- rollout 与 random actor 差异不明显
- 不适合作为 warm-start checkpoint

## Warm-Start Smoke

`model_10000.pt` actor-only warm-start smoke：

- 16 env / 64 step / no learning
- mode: `force_torque`
- actor obs: 575
- critic obs: 134
- action dim: 29
- lower actor loaded: 9/9
- upper actor loaded: 9/9
- critic loaded: false
- optimizer loaded: false
- force/torque 均非零
- nonfinite = 0

## 256 Env / 20-It 对比

配置：

- mode: `force_torque`
- curriculum: `adaptive_floor`
- seed: 1
- num_envs: 256
- iterations: 20
- no ONNX export
- no checkpoint save

| metric | warm-start | from-scratch |
|---|---:|---:|
| nonfinite | 0 | 0 |
| reset_rate | 0.00505 | 0.02507 |
| episode_length_mean | 305.40 | 42.58 |
| reward_mean | 60.86 | -23.19 |
| force_norm_mean / max | 3.3709 / 7.9503 | 1.9531 / 5.7570 |
| torque_norm_mean / max | 0.0797 / 0.1729 | 0.0463 / 0.1288 |
| force_scale_final | 0.1136 | 0.1061 |
| torque_scale_final | 0.0566 | 0.0530 |
| upper_body_torque_saturation_ratio | 0.00216 | 0.04333 |
| tracking_upper_body_dofs | 1.8430 | 0.0441 |

结论：

- warm-start 明显降低 reset rate；
- episode length 和 reward 大幅提升；
- force/torque 均持续非零；
- upper-body torque saturation 没有恶化，反而显著降低。

## 256 Env / 100-It Warm-Start

配置：

- checkpoint: `/home/lavine/project/FALCON/model_10000.pt`
- mode: `force_torque`
- curriculum: `adaptive_floor`
- seed: 1
- num_envs: 256
- iterations: 100
- no ONNX export
- no checkpoint save

最终结果：

| metric | value |
|---|---:|
| nonfinite | 0 |
| reset_rate | 0.00439 |
| episode_length_mean | 250.27 |
| reward_mean | 44.61 |
| force_norm_mean / max | 4.6936 / 11.2153 |
| torque_norm_mean / max | 0.1068 / 0.2480 |
| force_scale_final | 0.1580 |
| torque_scale_final | 0.0789 |
| upper_body_torque_saturation_ratio | 0.00202 |
| tracking_upper_body_dofs | 1.4923 |
| wrench_scale_down_fraction | 0.0000 |
| wrench_scale_factor_mean / min | 1.0000 / 1.0000 |

趋势：

| signal | first20 | mid40-60 | last20 | final |
|---|---:|---:|---:|---:|
| force_norm_mean | 3.1546 | 3.9281 | 4.4987 | 4.6936 |
| torque_norm_mean | 0.0750 | 0.0913 | 0.1042 | 0.1068 |
| force_scale | 0.1082 | 0.1307 | 0.1524 | 0.1580 |
| torque_scale | 0.0540 | 0.0652 | 0.0761 | 0.0789 |
| reset_rate | 0.00328 | 0.00403 | 0.00396 | 0.00439 |
| reward_mean | 29.65 | 44.97 | 44.82 | 44.61 |
| upper torque saturation | 0.00171 | 0.00188 | 0.00207 | 0.00202 |

结论：

- 100-it warm-start 稳定性通过；
- force/torque 没有 collapse，随 curriculum 稳定增长；
- reset_rate 很低；
- torque saturation 非常可控；
- scale-down 没有触发。

## 服务器训练工作流

新增服务器训练辅助文档和工具：

- `docs/workflows/server_training_workflow.md`
- `tools/server_training/remote_pull_branch`
- `tools/server_training/run_6d_wrench_warmstart_tmux`
- `tools/server_training/sync_artifacts_from_server`
- `tools/server_training/sync_worktree_to_server`

当前推荐工作流：

1. 本地 Codex 改代码和检查 diff；
2. 需要时直接 `rsync` 当前工作树到服务器；
3. 服务器用 tmux 手动启动训练；
4. logs/checkpoints 不进 git；
5. 训练结果用 `rsync` 拉回本地分析。

服务器暂不可用，因此今日未启动服务器训练。

## 当前风险

- 当前还没有导出新的 ONNX；
- 当前还没有做 MuJoCo valve transfer evaluation；
- 训练仍基于 fakehand 29DoF，不包含 with-hand 43DoF；
- 6D wrench 仍使用 conservative post-scale-down，不是 direction-alpha feasible sampling；
- 100-it 仍属于小规模稳定性验证，不是正式长训。

## 下一步建议

服务器恢复前：

- 整理并提交当前训练主线代码；
- 不继续扩大本地训练规模；
- 如需增加置信度，可做 seed=2 的 256 env / 100-it warm-start 小验证。

服务器恢复后：

1. 同步当前工作树到服务器；
2. 运行 `256 env / 300-it` 或 `512 env / 300-it` warm-start；
3. 检查 force/torque 是否持续非零、reset_rate、reward、tracking、torque saturation；
4. 通过后再考虑更长训练；
5. ONNX export 和 MuJoCo valve transfer evaluation 另开阶段审批。
