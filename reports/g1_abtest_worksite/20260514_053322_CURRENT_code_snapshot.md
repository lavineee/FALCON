# G1 A/B Test Worksite Snapshot

## 1. Metadata

- Record time: 2026-05-14 05:33:22 Asia/Shanghai
- Operator: Codex；本次只记录代码现场，未执行实机。
- Robot tag: CURRENT
- Robot role: code_worksite
- Test type: code_snapshot
- Notes: 当前代码现场记录，用于之后和 023556 成功 G1、022306 失败 G1 的实机测试结果对比。
- Physical setup:
  - battery: unknown；本次只记录代码现场，未执行实机。
  - rope state: not applicable；本次只记录代码现场，未执行实机。
  - floor: unknown；本次只记录代码现场，未执行实机。
  - robot posture: not applicable；本次只记录代码现场，未执行实机。
  - valve contact: not applicable；本次只记录代码现场，未执行实机。
  - hand connected: unknown；本次只记录代码现场，未连接 G1。
  - RealSense connected: unknown；本次只记录代码现场，未检查 RealSense。
- Safety status:
  - remote controller available: unknown；本次只记录代码现场，未执行实机。
  - damping tested: not applicable；本次只记录代码现场，未执行 damping 测试。
  - emergency operator: unknown；本次只记录代码现场，未执行实机。
  - no-contact test only: yes；本次只记录代码现场，未执行实机，未连接 G1。

## 2. Exact command under test

记录当前标准实机部署命令：

```bash
python rl_policy/loco_manip/loco_manip.py \
  --config=config/g1/g1_29dof_falcon_real.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

说明：该命令只作为本次代码现场的待测部署命令记录。本次没有运行该命令，没有启动策略，也没有连接机器人。

## 3. Repository state

- Repository root: `/home/lavine/project/FALCON`
- Branch: `feature-valve-vision-module`
- HEAD: `0538761880f38157e5345b0e667843fb436fdeb1`
- Git status source: `git status --short`
- Git diff stat source: `git diff --stat`

说明：以下状态是在创建本记录文件前读取的代码现场。创建本记录后，`reports/g1_abtest_worksite/` 下会新增记录文件，这是本次记录动作本身带来的文档变化。

```text
 M humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.yaml
 M humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.py
 M sim2real/config/g1/g1_29dof_with_hand_valve_grasp_contact_overlay.yaml
 M sim2real/rl_policy/loco_manip/loco_manip.py
 M sim2real/rl_policy/loco_manip/loco_manip_ee_tracking_7dof_with_hand_test.py
 M sim2real/rl_policy/loco_manip/loco_manip_ee_tracking_test.py
 M sim2real/rl_policy/loco_manip/loco_manip_valve_grasp_contact_7dof_with_hand.py
 M sim2real/sim_env/loco_manip.py
?? docs/worklogs/g1_real_vision_group_report_2026-05-13.md
?? sim2real/REAL_G1_MIGRATION_NOTES.md
?? sim2real/config/g1/g1_29dof_falcon_real.yaml
?? sim2real/config/g1/g1_29dof_with_hand_valve_real_overlay.yaml
?? sim2real/config/g1/g1_29dof_with_hand_valve_real_vision_overlay.yaml
?? sim2real/config/local/diag_turn_exit_fast.yaml
?? sim2real/config/local/diag_turn_exit_short_hold.yaml
?? sim2real/config/local/diag_turn_extra_settle_4s.yaml
?? sim2real/config/local/diag_turn_hold_actual_exit.yaml
?? sim2real/config/local/diag_turn_radius_actual_only.yaml
?? sim2real/config/local/diag_turn_radius_command_delta_scaled.yaml
?? sim2real/config/local/diag_turn_radius_command_delta_scaled_gain07.yaml
?? sim2real/config/local/diag_turn_tolerance_0p6.yaml
?? sim2real/config/local/diag_turn_tolerance_0p9.yaml
?? sim2real/config/local/g1_29dof_dec_loco_real.yaml
?? sim2real/config/local/g1_29dof_falcon_real.yaml
?? sim2real/config/local/real_link_test_force_close.yaml
?? sim2real/rl_policy/loco_manip/loco_manip_right_hand_circle_test.py
?? sim2real/rl_policy/loco_manip/run_valve_real_with_hand.py
?? sim2real/tools/test_inspire_hand_open_close.py
?? sim2real/utils/hand_controller.py
?? sim2real/utils/valve_target_receiver.py
?? sim2real/vision/valve_vision_ros1/
?? tools/eval_checkpoint_isaacgym.py
```

```text
 ...motion_stand_height_waist_wbc_ma_6d_wrench.yaml |  40 +++
 ...comotion_stand_height_waist_wbc_ma_6d_wrench.py | 391 +++++++++++++++++++--
 ...9dof_with_hand_valve_grasp_contact_overlay.yaml |  16 +
 sim2real/rl_policy/loco_manip/loco_manip.py        |   4 +-
 .../loco_manip_ee_tracking_7dof_with_hand_test.py  |  58 ++-
 .../loco_manip/loco_manip_ee_tracking_test.py      | 359 ++++++++++++++++++-
 ...oco_manip_valve_grasp_contact_7dof_with_hand.py | 225 +++++++++++-
 sim2real/sim_env/loco_manip.py                     |   5 +-
 8 files changed, 1043 insertions(+), 55 deletions(-)
```

## 4. Deployment-relevant paths and hashes

说明：以下路径按待测命令从 `sim2real` 目录运行时的相对路径对应到仓库中的实际文件。hash 使用 `git hash-object` 读取文件内容得到，属于 Git blob hash，不是 `sha256sum`。

- Policy entry path: `sim2real/rl_policy/loco_manip/loco_manip.py`
  - git hash-object: `b572eb7ba567c566eb5ed48593e34011051f700f`
- Config path: `sim2real/config/g1/g1_29dof_falcon_real.yaml`
  - git hash-object: `43add51ac52e7939452efefc05715a2a872256eb`
- Model path: `sim2real/models/falcon/g1_29dof.onnx`
  - git hash-object: `0448f4dedae3c1c12b70b16e7bf84324516ad68f`

## 5. Execution constraints for this record

- 未运行 Python 工具脚本。
- 未修改任何已有控制代码。
- 未连接 G1。
- 未发 DDS。
- 未启动机器人策略。
- 未调用 `MotionSwitcher.ReleaseMode()`。
- 未向 `rt/lowcmd` 发布。
- 未创建自动化 logger 脚本。
- 本次只记录代码现场，未执行实机。

## 6. Read-only commands used

```bash
date +%Y%m%d_%H%M%S
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse --show-toplevel
find reports -maxdepth 2 -type f
find sim2real/rl_policy/loco_manip -maxdepth 1 -type f -name loco_manip.py
find sim2real/config/g1 -maxdepth 1 -type f -name g1_29dof_falcon_real.yaml
find sim2real/models -maxdepth 3 -type f -name g1_29dof.onnx
git hash-object sim2real/rl_policy/loco_manip/loco_manip.py
git hash-object sim2real/config/g1/g1_29dof_falcon_real.yaml
git hash-object sim2real/models/falcon/g1_29dof.onnx
git diff --name-only
git diff --stat
```
