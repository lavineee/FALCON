# G1 A/B Test Worksite Snapshot

## 1. Metadata

- Record time: 2026-05-14 06:04:22 Asia/Shanghai (CST)
- Operator: 用户实机操作；Codex 只做记录整理。
- Robot tag: unknown / previous_failed_g1；用户描述为“另一台之前部署不了的 G1”，疑似对应之前失败机，具体编号待确认。
- Robot role: previously_failed_now_success
- Test type: real_policy_deploy_success_user_reported
- Notes: 用户报告：当前已经连接，并且在另一台之前部署不了的 G1 上成功部署了策略；代码没有改动，按正常步骤运行后成功。本记录用于之后和 023556 成功 G1、022306 失败 G1 以及当前代码现场记录对比。
- Physical setup:
  - battery: unknown；用户未提供电量信息。
  - rope state: unknown；用户未提供保护绳状态。
  - floor: unknown；用户未提供地面条件。
  - robot posture: unknown；用户未提供机器人起始姿态。
  - valve contact: unknown；用户未说明是否发生阀门接触。
  - hand connected: unknown；用户只说明已连接并成功部署策略，未明确说明手部连接状态。
  - RealSense connected: unknown；用户未说明 RealSense 连接状态。
- Safety status:
  - remote controller available: unknown；用户未提供遥控器状态。
  - damping tested: unknown；用户未说明是否执行 damping 测试。
  - emergency operator: unknown；用户未提供现场急停人员信息。
  - no-contact test only: unknown；用户未说明本次是否只做无接触测试。Codex 本次没有执行实机，只记录用户口述结果。

## 2. Exact command under test

记录当前标准实机部署命令：

```bash
python rl_policy/loco_manip/loco_manip.py \
  --config=config/g1/g1_29dof_falcon_real.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

说明：用户描述为“什么也没改，就按正常步骤运行”。本记录按当前标准实机部署命令记录；Codex 没有运行该命令，没有启动策略，没有连接机器人。

## 3. Repository state

- Repository root: `/home/lavine/project/FALCON`
- Branch: `feature-valve-vision-module`
- HEAD: `0538761880f38157e5345b0e667843fb436fdeb1`
- Git status source: `git status --short`
- Git diff stat source: `git diff --stat`

说明：以下状态是在创建本记录文件前读取的代码现场。此时上一份 `reports/g1_abtest_worksite/` 记录已经存在，所以 `git status --short` 中出现 `?? reports/`。

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
?? reports/
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

- Codex 未运行 Python 工具脚本。
- Codex 未修改任何已有控制代码。
- Codex 未连接 G1。
- Codex 未发 DDS。
- Codex 未启动机器人策略。
- Codex 未调用 `MotionSwitcher.ReleaseMode()`。
- Codex 未向 `rt/lowcmd` 发布。
- Codex 未创建自动化 logger 脚本。
- 本记录只整理用户口述的成功部署事实和当前代码现场。

## 6. Read-only commands used

```bash
date +%Y%m%d_%H%M%S
date '+%Y-%m-%d %H:%M:%S %Z'
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse --show-toplevel
git diff --stat
git hash-object sim2real/rl_policy/loco_manip/loco_manip.py
git hash-object sim2real/config/g1/g1_29dof_falcon_real.yaml
git hash-object sim2real/models/falcon/g1_29dof.onnx
find reports/g1_abtest_worksite -maxdepth 1 -type f
find sim2real/models -maxdepth 3 -type f -name g1_29dof.onnx
```

## 7. Deployment observation

- Result: success；用户报告策略已经在此前部署不了的 G1 上成功部署。
- Code change before success: none reported；用户明确说明“什么也没改”。
- Operation path: normal procedure；用户说明“就按正常步骤运行”。
- Important interpretation: 这条记录提示失败原因可能不只来自当前仓库代码差异，也可能来自当时的启动顺序、机器人状态、连接状态、DDS 环境、网络接口、权限、终端环境、上电状态、Motion mode 状态或其他现场条件。这里先只记录现象，不做控制逻辑改动。
