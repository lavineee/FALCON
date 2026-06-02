# Valve Ring Perception Local Debug Archive

Archive time: 2026-06-02 16:26:00 CST

## Current Status

This archive records the current local D435i valve perception module state before further tuning.

Conclusion:

- Front-facing valve views are basically working.
- Physical radius estimation is accurate in near-frontal views.
- Overhead / steeply tilted views are still weak.
- In those tilted views, candidate geometry can still be visually plausible, but the final radius estimate can drift significantly.

## Validated Mode

Current working mode:

```bash
/home/lavine/miniconda3/envs/rs/bin/python sim2real/vision/valve_ring_perception/tools/run_realsense_online_estimation.py \
  --mode local_debug \
  --config sim2real/vision/valve_ring_perception/config/shape_only.json \
  --output-dir artifacts/valve_ring_perception/local_debug \
  --write-latest-result
```

This mode uses local PC direct D435i access through `pyrealsense2`.

It does not use:

- ROS
- G1 head camera
- UDP
- policy/control code
- `/tmp/falcon_real_valve_vision_status.json`, unless `--write-falcon-status` is explicitly passed

Coordinate convention in this mode:

- `frame = camera/local`
- `T_base_cam = identity(local_debug)`
- `center_cam`, `radius_m`, and `axis_cam` are the values to inspect
- `center_base` / `axis_base` in JSON are camera-local aliases in this mode, not real G1 base-frame values

## Latest Local Debug Snapshot

Latest result file:

```text
artifacts/valve_ring_perception/local_debug/latest_result.json
```

Observed latest values:

```text
valid: True
frame: camera/local
source: valve_ring_perception_local_debug
temporal_status: UNSTABLE
current_valid: True
radius_m: 0.1260293622131485
center_cam: [-0.04570187708959654, -0.07736586705318847, 0.6766938073358952]
axis_cam: [-0.09393288734774197, 0.14966618968863155, -0.984264519495955]
```

Note: `temporal_status = UNSTABLE` does not necessarily mean the current geometry is invalid. It means the temporal latch has not accepted a stable sequence yet. The raw frame-level estimate is still valid here.

## Offline Smoke Test

Offline evaluation output:

```text
artifacts/valve_ring_perception/g1_capture/local_debug_eval/summary.json
```

Observed summary:

```text
success_rate: 1.0
radius_mean: 0.12149510004595301
radius_std: 0.0
center_std: [0.0, 0.0, 0.0]
axis_std: 0.0
failure_reason_counts: {}
```

This was a single-frame smoke test, so std values are not meaningful for long-run stability. It only confirms that the current pipeline can run end to end on an RGB-D frame and produce a valid estimate.

## Current Visual Debug Semantics

The overlay currently visualizes:

- image-plane ellipse candidate
- selected candidate mask
- outer envelope samples
- metric 3D fitted circle projected back to image
- fitted center projected back to image
- local camera-frame center / axis / radius text

Important distinction:

- The magenta ellipse is the image-plane candidate.
- The cyan curve is the projected 3D fitted circle.
- The final `radius_m` is from 3D valve-plane circle fitting, not from the image ellipse axes.

## Known Weakness

The current pipeline is not yet robust for overhead or strongly tilted camera views.

Observed / expected failure pattern:

- The valve outer contour becomes a strong ellipse in the image.
- Shape candidate generation may still find a visually reasonable ellipse.
- Depth sampling and valve-plane fitting become more sensitive to missing depth, specular black surfaces, spokes, center boss, and background leakage.
- Outer-envelope sampling may not preserve the true handwheel rim evenly enough.
- Circle fitting in the valve plane may then produce a noticeably biased `radius_m`.

Practical implication:

- Use current module mainly for near-frontal local debugging.
- Do not treat tilted-view radius estimates as reliable yet.
- Do not feed this directly into sim2real grasp/control until tilted-view radius stability is improved and validated.

## Active Configuration

Current main config:

```text
sim2real/vision/valve_ring_perception/config/shape_only.json
```

Key settings:

```text
candidate_sources: ["shape"]
shape_hough_enabled: true
shape_min_axis_px: 140
shape_max_aspect_ratio: 2.8
radius_range_m: [0.08, 0.35]
hard_min_angular_coverage: 0.25
max_circle_rmse_m: 0.025
temporal_stable_frames_required: 3
```

## Recommended Next Work

Before using this for G1 sim2real:

- Add explicit grasp-point proposal visualization.
- Improve tilted-view handling.
- Validate radius stability across camera pitch/yaw angles.
- Record local sequences with `--record-dir` and evaluate with `sim2real/vision/valve_ring_perception/tools/evaluate_offline_sequence.py`.
- Add per-view-angle summary, especially radius error / std under overhead views.

Suggested local sequence recording command:

```bash
/home/lavine/miniconda3/envs/rs/bin/python sim2real/vision/valve_ring_perception/tools/run_realsense_online_estimation.py \
  --mode local_debug \
  --config sim2real/vision/valve_ring_perception/config/shape_only.json \
  --output-dir artifacts/valve_ring_perception/local_debug \
  --record-dir data/valve_local/session_001 \
  --record-every-n 1 \
  --write-latest-result
```

Suggested offline evaluation command:

```bash
/home/lavine/miniconda3/envs/rs/bin/python sim2real/vision/valve_ring_perception/tools/evaluate_offline_sequence.py \
  data/valve_local/session_001 \
  --config sim2real/vision/valve_ring_perception/config/shape_only.json \
  --frame-label camera/local \
  --source valve_ring_perception_local_debug
```
