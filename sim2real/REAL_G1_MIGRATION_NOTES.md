# Real G1 Migration Notes

Last validated: 2026-05-13 on the current G1 local PC.

This document records the working real-hardware setup for the
vision -> approach -> Inspire-hand grasp link test. It is meant as a migration
checklist for moving to another G1 robot.

## Topology

- Control PC repository: `/home/lavine/project/FALCON`
- Control PC G1 network interface: `enp0s31f6`
- Control PC G1-network IP: `192.168.123.222/24`
- Current G1 local PC user: `unitree`
- Current G1 local PC hostname: `ubuntu`
- Current G1 local PC interface: `eth0`
- Current G1 local PC IP: `192.168.123.164/24`
- DDS domain/channel: `0`
- Vision UDP target: `192.168.123.222:5055`

Do not store robot passwords in the repository. The current password was
provided interactively during testing.

## Control PC Code State

Important files in this repo:

- `sim2real/config/g1/g1_29dof_falcon_real.yaml`
- `sim2real/config/g1/g1_29dof_with_hand_valve_real_overlay.yaml`
- `sim2real/config/local/real_link_test_force_close.yaml`
- `sim2real/rl_policy/loco_manip/run_valve_real_with_hand.py`
- `sim2real/tools/test_inspire_hand_open_close.py`
- `sim2real/utils/hand_controller.py`
- `sim2real/vision/valve_vision_ros1/`

Key real-G1 setting:

```yaml
# sim2real/config/g1/g1_29dof_falcon_real.yaml
UNITREE_LEGGED_CONST:
  MODE_MACHINE: 6
```

`MODE_MACHINE=6` was required for this G1 to move under the real runner. With
`MODE_MACHINE=5`, the policy sent commands but the robot did not move.

Real hand overlay:

```yaml
# sim2real/config/g1/g1_29dof_with_hand_valve_real_overlay.yaml
runtime_mode: real
hand_backend: inspire_dds
hand_role: real_right_hand
inspire_hand_side: right
inspire_dds_channel: 0
inspire_network_interface: enp0s31f6
inspire_open_raw: [900, 900, 900, 900, 900, 600]
inspire_close_raw: [350, 350, 350, 350, 420, 600]
valve_geometry_source: real_vision
real_vision_status_file: /tmp/falcon_real_valve_vision_status.json
```

Real link-test force-close overlay:

```yaml
# sim2real/config/local/real_link_test_force_close.yaml
valve_approach_max_s: 5.0
valve_close_wait_s: 0.8
valve_close_max_s: 1.5
valve_enter_hold_on_close_timeout: true
valve_hold_s: 2.0
valve_stop_after_done_s: 1.0
```

Use the force-close overlay only for safe air-grasp link tests where the target
is overridden to a reachable point in front of the robot. For real object
contact tests, remove it or tune the close/contact logic.

## G1 Vision Installation

Installed path on the current G1 local PC:

```text
/home/unitree/falcon_valve_vision/
  valve_vision_node.py
  config/valve_vision_g1.yaml
```

The node must run with system Python, not the conda Python:

```bash
/usr/bin/python3 --version
# Python 3.8.10

/usr/bin/python3 - <<'PY'
import cv2
print(cv2.__version__)
print(hasattr(cv2, "aruco"))
PY
# 4.2.0
# True
```

The current conda `python3` on G1 is Python 3.13 and was not used for the vision
node.

Vision config currently used:

```yaml
topics:
  color: "/camera/color/image_raw"
  depth: "/camera/aligned_depth_to_color/image_raw"
  camera_info: "/camera/color/camera_info"

transport:
  udp_host: "192.168.123.222"
  udp_port: 5055

publish_hz: 15.0
publish_invalid: true
depth_scale_m: 0.001
use_depth_center: false
max_depth_age_s: 0.20

marker:
  dictionary: "DICT_4X4_50"
  id: 0
  size_m: 0.0765
  center_offset_marker_m: [0.0, 0.0, 0.0]
  grasp_offset_marker_m: [0.0, -0.10, 0.0]
  axis_marker: [0.0, 0.0, 1.0]

T_base_camera:
  translation_m: [0.065556607, 0.037311661, 0.454167315]
  rotation_matrix:
    - [0.032664000, -0.981054000, 0.190960000]
    - [-0.997941000, -0.042566000, -0.047983000]
    - [0.055203000, -0.189000000, -0.980424000]
```

When migrating to another robot, re-check `T_base_camera`. It is a rough
calibration for the current G1 head D435I and should not be assumed exact on a
new robot.

## G1 Vision Startup

Current tmux sessions:

```text
falcon_roscore
falcon_realsense
falcon_valve_vision
```

Start commands:

```bash
tmux new-session -d -s falcon_roscore \
  'source /opt/ros/noetic/setup.bash; roscore'

tmux new-session -d -s falcon_realsense \
  'source /opt/ros/noetic/setup.bash; roslaunch realsense2_camera rs_camera.launch align_depth:=true color_width:=640 color_height:=480 depth_width:=640 depth_height:=480 color_fps:=30 depth_fps:=30'

tmux new-session -d -s falcon_valve_vision \
  'source /opt/ros/noetic/setup.bash; /usr/bin/python3 /home/unitree/falcon_valve_vision/valve_vision_node.py --config /home/unitree/falcon_valve_vision/config/valve_vision_g1.yaml'
```

Check ROS topics:

```bash
rostopic list | grep -E '^/camera/(color|aligned_depth_to_color)' | sort
```

Expected important topics:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/aligned_depth_to_color/image_raw
/camera/aligned_depth_to_color/camera_info
```

`rs-enumerate-devices -s` can report busy while the ROS RealSense node owns the
camera. Use it before starting the ROS node if you need the RealSense serial.
The current D435I serial observed earlier was `243622070699`.

## Inspire Hand Installation

The current working single-hand bridge is the DFX Inspire service, not the
Unitree `G1-edu-ftp-hand-version` high-level package.

Installed path on the current G1 local PC:

```text
/home/unitree/DFX_inspire_service/
/home/unitree/DFX_inspire_service/build/inspire_h1
/home/unitree/DFX_inspire_service/build/inspire_g1
/home/unitree/DFX_inspire_service_codex.tgz
```

The archive `/home/unitree/DFX_inspire_service_codex.tgz` can be used as a
source snapshot for migration if the new G1 does not already have this package.

Relevant bridge behavior:

- Publishes/subscribes DDS namespace `inspire` by default.
- Command topic: `rt/inspire/cmd`
- State topic: `rt/inspire/state`
- Message type: `unitree_go::msg::dds_::MotorCmds_` / `MotorStates_`
- The 12 command entries are:
  `[right_pinky, right_ring, right_middle, right_index, right_thumb_bend, right_thumb_rotation, left_pinky, left_ring, left_middle, left_index, left_thumb_bend, left_thumb_rotation]`

For the current single-hand setup, use `inspire_h1`, not `inspire_g1`.
`inspire_g1` expects two hard-coded serial ports and was not the working path.

Current working hand serial:

```text
/dev/ttyUSB2
```

On the current G1, the FTDI board enumerated as:

```text
0403:6011 Future Technology Devices International, Ltd FT4232H Quad HS USB-UART/FIFO IC
/dev/ttyUSB0
/dev/ttyUSB1
/dev/ttyUSB2
/dev/ttyUSB3
```

`/dev/ttyUSB2` was identified by checking `rt/inspire/state`: it returned real
q values near `[0.918, 0.90, 0.90, 0.90, 0.896, 0.587]`. `/dev/ttyUSB0` and
`/dev/ttyUSB1` produced all-zero q and rapidly increasing `lost`, indicating
no valid hand-state communication.

Known hardware issue on the current hand: `q[0]` / pinky did not move in an
isolated pinky-only test. Other fingers moved. Treat this as a current-hand
hardware/channel issue, not as a software migration requirement.

## Inspire Hand Startup

Current tmux session:

```text
falcon_inspire_h1
```

Start command:

```bash
tmux new-session -d -s falcon_inspire_h1 \
  'cd /home/unitree/DFX_inspire_service/build && sudo ./inspire_h1 -s /dev/ttyUSB2 --network eth0 > /tmp/falcon_inspire_h1.log 2>&1'
```

If passwordless sudo is not configured, run interactively or use the G1's local
operator workflow. Do not store the sudo password in scripts committed to the
repo.

Check process and log:

```bash
tmux ls
ps -ef | grep -Ei 'inspire_h1|DFX|inspire' | grep -v grep
cat /tmp/falcon_inspire_h1.log
```

Check serial devices:

```bash
find /dev -maxdepth 1 \( -name 'ttyUSB*' -o -name 'ttyACM*' \) -ls
lsusb | grep -E '0403|FTDI|6011|Serial|Bus'
```

If the FTDI device appears and disappears, check power, hub, and USB
enumeration. Earlier failure mode:

```text
USB disconnect
device descriptor read/all, error -71
```

## Hand Test Commands

From the control PC:

```bash
/home/lavine/miniconda3/envs/fcreal/bin/python \
  sim2real/tools/test_inspire_hand_open_close.py \
  --config sim2real/config/g1/g1_29dof_with_hand_valve_grasp_contact_overlay.yaml \
  --overlay sim2real/config/g1/g1_29dof_with_hand_valve_real_overlay.yaml \
  --backend inspire_dds \
  --pause_sec 1.0 \
  --final_wait_sec 0.5
```

Useful state check from the control PC:

```python
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

ChannelFactoryInitialize(0, "enp0s31f6")
# Subscribe to rt/inspire/state and verify q changes and lost does not only
# monotonically grow without valid q.
```

## Safe Vision Override Link Test

For migration bring-up, use the same safe target override before trusting the
new robot's camera calibration. The temporary control-PC UDP receiver listens
on port `5055`, accepts real packets from G1 vision, and writes:

```json
{
  "valid": true,
  "source": "g1_d435i_ros1_marker_overridden_safe_front_target",
  "center_base": [0.36, -0.10, -0.10],
  "grasp_base": [0.36, -0.19, -0.10],
  "axis_base": [-1.0, 0.0, 0.0]
}
```

The latest validated run received real G1 vision UDP packets with
`raw_valid=true`, but the policy consumed the overridden safe front target.

Validated real runner command from `sim2real/`:

```bash
timeout 55s /home/lavine/miniconda3/envs/fcreal/bin/python \
  rl_policy/loco_manip/run_valve_real_with_hand.py \
  --duration_s 22 \
  --log_file /tmp/falcon_valve_real_with_hand_visual_override_inspire_h1_metrics.csv \
  --marker_file /tmp/falcon_valve_real_with_hand_visual_override_inspire_h1_markers.json \
  --config config/g1/g1_29dof_falcon_real.yaml \
  --config config/g1/g1_29dof_with_hand_tracking_overlay.yaml \
  --config config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml \
  --config config/g1/g1_29dof_with_hand_valve_grasp_contact_overlay.yaml \
  --config config/g1/g1_29dof_with_hand_valve_real_overlay.yaml \
  --config config/local/real_link_test_force_close.yaml
```

Expected state progression:

```text
move_pregrasp -> approach_grasp -> close_hand -> hold_grasp -> done
```

Expected hand commands in log:

```text
[HAND] inspire_dds right hand open
[HAND] inspire_dds right hand close
[HAND] inspire_dds right hand hold
[HAND] inspire_dds right hand open
```

`close_stage_timeout` is expected during this air-grasp safe-target test because
there is no real contact closure.

Validated output files from the latest run:

```text
/tmp/falcon_valve_real_with_hand_visual_override_inspire_h1_metrics.csv
/tmp/falcon_valve_real_with_hand_visual_override_inspire_h1_markers.json
```

## Migration Checklist For New G1

1. Put the new G1 on the same network or update all IPs.
   - New G1 `eth0` IP may differ from `192.168.123.164`.
   - Update `/home/unitree/falcon_valve_vision/config/valve_vision_g1.yaml`
     `transport.udp_host` to the control PC IP.
2. Copy/install `/home/unitree/falcon_valve_vision`.
   - Run the node with `/usr/bin/python3`, not conda Python.
   - Verify `cv2.aruco` exists.
3. Start ROS1 Noetic, RealSense, and vision tmux sessions.
   - Verify `/camera/color/image_raw` and aligned depth topics exist.
   - Verify control PC receives UDP packets on port `5055`.
4. Copy/install `/home/unitree/DFX_inspire_service`.
   - Use `/home/unitree/DFX_inspire_service_codex.tgz` if needed.
   - Build `inspire_h1` if binaries are missing.
5. Find the real hand serial port.
   - Confirm FTDI `0403:6011` enumerates.
   - Try `/dev/ttyUSB0` to `/dev/ttyUSB3`.
   - Pick the port whose `rt/inspire/state` has real q values.
   - On the current G1, that was `/dev/ttyUSB2`.
6. Start `inspire_h1` on the selected serial port with `--network eth0`.
7. From the control PC, run the hand open/close test.
8. Confirm `MODE_MACHINE=6` on the control PC config.
9. Run the safe vision-override link test.
10. Only after safe bring-up, remove the target override and validate real
    tag-pose approach with the new camera calibration.

