# 6D Wrench Curriculum Stability Worklog

Date: 2026-04-27

## Scope

This worklog tracks the isolated 6D wrench training path for the G1 29DoF
fakehand policy. The MuJoCo valve demo, original force-only FALCON env/config,
reward config, existing checkpoints, and existing ONNX exports were not
modified.

Current invariant:

- robot: `g1/g1_29dof_waist_fakehand`
- actor obs history dim: 575
- critic obs dim: 134
- action dim: 29
- EE links: `left_rubber_hand`, `right_rubber_hand`
- wrench implementation: conservative post-scale-down, not direction-alpha feasible sampling

## 5-Iteration Smoke

Initial 5-iteration checks showed the 6D wrench path could instantiate and train
briefly without NaN. The first torque curriculum version collapsed too quickly:
`torque_norm_mean` fell to about `0.0049 Nm`, which made the run effectively
force-dominant / near-no-torque.

After adding torque warmup/floor:

- `torque_scale_floor_during_warmup: 0.03`
- `torque_curriculum_warmup_iterations: 20`
- `torque_curriculum_allow_down_after_iterations: 20`
- `torque_scale_down: 0.002`
- `torque_scale_up: 0.005`

the 5-iteration ablation kept torque alive. `torque_only` and `force_torque`
both showed nonzero applied torque, with no nonfinite scalars.

## 64 Env / 20-It

The first 64-env / 20-it sweep showed torque stayed alive, but force collapsed
to zero in both `force_only` and `force_torque`; `force_torque` was therefore
closer to torque-only late in training.

After adding force warmup/floor:

- `force_scale_floor_during_warmup: 0.05`
- `force_curriculum_warmup_iterations: 20`
- `force_curriculum_allow_down_after_iterations: 20`
- `force_scale_down: 0.005`
- `force_scale_up: 0.01`

the 64-env / 20-it rerun passed:

- `force_only`: `force_norm_mean = 1.8910`, `force_scale_final = 0.1058`, `nonfinite = 0`
- `force_torque`: `force_norm_mean = 1.8820`, `torque_norm_mean = 0.0447`,
  `force_scale_final = 0.1059`, `torque_scale_final = 0.0530`, `nonfinite = 0`
- wrench scale-down did not trigger: fraction `0`, factor mean/min `1/1`

## 256 Env / 20-It

An original force-only compose check was run first to confirm the old config was
not broken:

- original force-only env target:
  `LeggedRobotDecoupledLocomotionStanceHeightWBCForce`
- actor obs: 575
- critic obs: 128
- action dim: 29

Then two 256-env / 20-it groups were run with `seed=1`.

`force_only`:

- `nonfinite = 0`
- `reset_rate = 0.0238`
- `episode_length_mean = 42.58`
- `reward_mean = -23.31`
- `force_norm_mean/max = 1.9682 / 6.2614`
- `torque_norm_mean/max = 0 / 0`
- `force_scale_final = 0.1058`
- `upper_body_torque_saturation_ratio = 0.0433`

`force_torque`:

- `nonfinite = 0`
- `reset_rate = 0.0251`
- `episode_length_mean = 42.58`
- `reward_mean = -23.19`
- `force_norm_mean/max = 1.9531 / 5.7570`
- `torque_norm_mean/max = 0.0463 / 0.1288`
- `force_scale_final = 0.1058`
- `torque_scale_final = 0.0530`
- `upper_body_torque_saturation_ratio = 0.0433`

The 20-it logs were archived at:

`artifacts/server_runs/6d_wrench_256env_20it_seed1.tar.gz`

## 256 Env / 100-It Collapse Diagnosis

A 256-env / 100-it `force_torque` run was numerically stable but did not pass
long-horizon 6D wrench validation because both curriculum scales collapsed:

- `nonfinite = 0`
- final `reset_rate = 0.0270`
- final `upper_body_torque_saturation_ratio = 0.0339`
- final `force_norm_mean/max = 0 / 0`
- final `torque_norm_mean/max = 0 / 0`
- force scale reached zero around iteration 63
- torque scale reached zero around iteration 75

Root cause: `_update_wrench_scale_curriculum()` used per-env
`episode_length_buf` at reset time. Episodes shorter than
`force_scale_down_threshold = 200` decremented the scale. The warmup floors only
protected the first 20 curriculum iterations, so after warmup a steady stream of
early resets drove force and torque to zero. The logic did not use reward or a
global reset-rate average.

## Fixed / Adaptive-Floor 100-It Correction

Two curriculum modes were added to the isolated 6D wrench env/config:

- `curriculum_mode: fixed`
- `curriculum_mode: adaptive`

Additional protection:

- `fixed_force_scale: 0.10`
- `fixed_torque_scale: 0.05`
- `force_scale_min_after_warmup: 0.05`
- `torque_scale_min_after_warmup: 0.03`
- `disable_force_down_until_iterations: 100`
- `disable_torque_down_until_iterations: 100`

`fixed`, 256 env / 100 it:

- `nonfinite = 0`
- final `reset_rate = 0.0278`
- final `reward_mean = -15.58`
- force norm first/mid/last/final: `1.842 / 1.888 / 1.836 / 1.887`
- torque norm first/mid/last/final: `0.0434 / 0.0443 / 0.0434 / 0.0437`
- final force/torque scale: `0.1000 / 0.0500`
- final `upper_body_torque_saturation_ratio = 0.0298`
- scale-down fraction `0`, factor mean/min `1/1`

`adaptive_floor`, 256 env / 100 it:

- `nonfinite = 0`
- final `reset_rate = 0.0285`
- final `reward_mean = -17.05`
- force norm first/mid/last/final: `1.963 / 1.934 / 1.774 / 1.795`
- torque norm first/mid/last/final: `0.0460 / 0.0453 / 0.0423 / 0.0410`
- final force/torque scale: `0.1057 / 0.0530`
- final `upper_body_torque_saturation_ratio = 0.0337`
- scale-down fraction `0`, factor mean/min `1/1`

## Current Conclusion

The 6D wrench curriculum path is stable enough to leave curriculum-collapse
debugging and enter the actor-only warm-start stage.

The next step is to initialize only `actor_lower_body` and `actor_upper_body`
from the original FALCON actor checkpoint while leaving the 6D critic and
optimizers freshly initialized.
