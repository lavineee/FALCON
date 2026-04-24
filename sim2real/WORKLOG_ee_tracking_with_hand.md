# EE Tracking and With-Hand Model Work Log

Date: 2026-04-25

## Background

This stage focused on preparing a reliable end-effector tracking evaluation before moving to valve grasping and turning. The goal was to verify that the right arm can track randomized target points in the robot base frame, first with the existing fake-hand model and then with a real hand model.

The work intentionally avoided changing the trained 29DoF ONNX policy. The policy still controls the original body joints, while any added hand joints are controlled by a separate simple open/close controller.

## Main Changes

1. Added right end-effector tracking evaluation scripts.
   - Randomly samples right EE targets in a bounded base-frame workspace.
   - Resamples every 5 seconds.
   - Logs target/current positions and tracking error.
   - Visualizes target and current EE proxy points in MuJoCo.

2. Added 7DoF right arm tracking.
   - Extended the IK from 4 arm DoFs to all 7 right arm DoFs.
   - Kept the test position-only for now; EE orientation is not evaluated.
   - The measured point is a palm/grasp-center proxy rather than the old fake hand plate.

3. Automated the MuJoCo test flow.
   - Starts MuJoCo.
   - Starts the policy automatically.
   - Releases the elastic band after policy startup.
   - Starts random target evaluation after stabilization.

4. Added a FALCON-compatible with-hand MuJoCo model.
   - Preserves the original FALCON 29DoF body model, freebase actuators, floor friction, and scene assumptions.
   - Adds real hand body/joint/actuator structure.
   - Appends 14 hand actuators after the original policy-controlled actuators.
   - Keeps policy control mapped explicitly by joint/actuator name.

5. Added a simple hand controller.
   - The ONNX policy still outputs only 29DoF commands.
   - Hand joints are controlled separately by PD torque.
   - For tracking tests, the right hand opens when a new target is sampled and closes once the EE error is below the configured threshold.

6. Removed dependency on the external `unitree_ros` model path.
   - Copied the with-hand URDF and missing mesh assets into this repository.
   - Updated with-hand IK asset paths to use repository-local assets.
   - Added more robust path resolution in the with-hand IK helper.

## Important Files

- `sim2real/sim_env/loco_manip.py`
  - Added EE marker visualization.
  - Added automatic elastic-band length/release support.
  - Made live plot optional when matplotlib is missing.
  - Made valve/weld lookup tolerant of scenes without a valve.

- `sim2real/sim_env/loco_manip_with_hand.py`
  - Name-mapped 29DoF policy bridge for a larger MuJoCo actuator set.
  - Separate hand PD controller.
  - Runtime stabilization for hand joints.

- `sim2real/rl_policy/loco_manip/loco_manip_ee_tracking_test.py`
  - Base 4DoF EE tracking evaluation.

- `sim2real/rl_policy/loco_manip/loco_manip_ee_tracking_7dof_test.py`
  - 7DoF position-only EE tracking evaluation.

- `sim2real/rl_policy/loco_manip/loco_manip_ee_tracking_7dof_with_hand_test.py`
  - 7DoF position-only EE tracking with the real hand model and hand open/close logic.

- `sim2real/utils/arm_ik/robot_arm_ik_with_hand.py`
  - With-hand IK wrapper.
  - Locks lower body and finger joints, keeps both 7DoF arms active.
  - Adds `L_ee` and `R_ee` frames using palm/grasp-center offsets.

- `humanoidverse/data/robots/g1/g1_29dof_old_freebase_with_hand.xml`
  - FALCON-compatible robot model with real hand joints appended.

- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand_tracking.xml`
  - Tracking-only scene without the valve.

- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase_with_hand.xml`
  - Valve-capable scene for later grasp/turn experiments.

- `sim2real/config/g1/g1_29dof_with_hand_tracking_overlay.yaml`
  - Overlay config for the with-hand tracking test.

## Current Test Command

Preferred current test, 7DoF right-arm tracking with the real hand model:

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_with_hand_auto.sh
```

## Launch Scripts and Usage

All commands should be run from `sim2real/`. The `PYTHON_BIN` variable is recommended in this environment because the default `python` executable may not exist.

### 4DoF Fake-Hand Tracking

Automatic one-command run:

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_auto.sh
```

Split terminals, if manual inspection is needed:

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

Default outputs:

- `/tmp/falcon_ee_tracking_markers.json`
- `/tmp/falcon_ee_tracking_metrics.csv`

### 7DoF Fake-Hand Tracking

Automatic one-command run:

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_auto.sh
```

Split terminals:

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

Default outputs:

- `/tmp/falcon_ee_tracking_7dof_markers.json`
- `/tmp/falcon_ee_tracking_7dof_metrics.csv`

### 7DoF With-Hand Tracking

Automatic one-command run:

```bash
cd /home/lavine/project/FALCON/sim2real
PYTHON_BIN=/home/lavine/miniconda3/envs/fcreal/bin/python \
DURATION_SEC=60 \
ELASTIC_LENGTH=-0.05 \
./launch_ee_tracking_7dof_with_hand_auto.sh
```

Split terminals:

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

Default outputs:

- `/tmp/falcon_ee_tracking_7dof_with_hand_markers.json`
- `/tmp/falcon_ee_tracking_7dof_with_hand_metrics.csv`

Useful environment variables:

- `DURATION_SEC`: total policy-side run time in seconds.
- `SAMPLE_PERIOD_SEC`: target resampling period; default is 5 seconds.
- `SEED`: random target seed.
- `X_RANGE`, `Y_RANGE`, `Z_RANGE`: base-frame target sampling ranges, formatted as `min,max`.
- `ELASTIC_LENGTH`: initial elastic-band length. The current stable value is `-0.05`.
- `ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC`: release delay after policy startup; default is 1 second.
- `TRACKING_START_DELAY_SEC`: delay before random target sampling starts; default is 2 seconds for current automatic runs.
- `HAND_CLOSE_ERROR_M`: with-hand only; right hand closes when the tracking error is below this threshold.

Visualization:

- Green sphere: sampled target point in the robot base frame.
- Red sphere: current EE proxy point.
- Their distance is the logged position-tracking error.

## Current Verification

The latest repository-local asset regression ran successfully:

- MuJoCo scene loads.
- IK loads from repository-local URDF/meshes.
- Policy starts automatically.
- Elastic band releases after policy startup.
- Hand open/close commands execute.
- Right EE tracking logs are generated.

Short regression result:

- Mean error: about 4.56 cm
- RMS error: about 5.54 cm
- Maximum error: about 18.25 cm, mainly target-switch transient
- Final error: about 2.71 cm

Earlier 20-second regression result:

- Mean error: about 4.29 cm
- RMS error: about 5.52 cm
- P95 error: about 12.29 cm
- Maximum error: about 18.68 cm
- Final error: about 1.08 cm

## Model Selection Decision

We compared the idea of using the SONIC G1 model from NVlabs GR00T-WholeBodyControl. The repository contains many useful G1 variants, but the mesh assets are Git LFS files and the overall model family is still not guaranteed to match FALCON's trained MuJoCo setup.

The current decision is:

- Do not replace the full FALCON robot model with SONIC/Unitree XML.
- Keep the FALCON-compatible base model.
- Use external models only as references for hand subtree design if needed.

This is the safer approach because the trained FALCON policy is sensitive to actuator ordering, freebase actuators, contact/friction parameters, scene assumptions, and joint layout.

## Git Recommendation

Do not push this directly to `main`.

Use a new branch, for example:

```bash
git switch -c feature/with-hand-ee-tracking
```

Reason:

- This stage contains new model assets, new simulator paths, new evaluation scripts, and workflow automation.
- Some launch scripts are currently ignored by `.gitignore` because of the global `*.sh` rule.
- We should explicitly decide which generated evaluation outputs to track and which to leave local.

Recommended commit scope:

- Track source/config/model files required to reproduce the test.
- Force-add the launch scripts if they should be part of the workflow.
- Do not commit `.codex`.
- Commit evaluation summaries only if we want the repository to preserve benchmark artifacts.
