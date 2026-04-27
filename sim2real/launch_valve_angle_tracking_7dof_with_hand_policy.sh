#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-config/g1/g1_29dof_falcon.yaml}"
OVERLAY_CONFIG="${OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_tracking_overlay.yaml}"
TUNED_OVERLAY_CONFIG="${TUNED_OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml}"
VALVE_ANGLE_OVERLAY_CONFIG="${VALVE_ANGLE_OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_valve_angle_tracking_overlay.yaml}"
EXTRA_OVERLAY_CONFIG="${EXTRA_OVERLAY_CONFIG:-}"
MERGED_CONFIG="${MERGED_CONFIG:-/tmp/falcon_g1_29dof_with_hand_valve_angle_tracking.yaml}"
MODEL_PATH="${MODEL_PATH:-models/falcon/g1_29dof.onnx}"
MARKER_FILE="${MARKER_FILE:-/tmp/falcon_valve_angle_tracking_markers.json}"
LOG_FILE="${LOG_FILE:-/tmp/falcon_valve_angle_tracking_metrics.csv}"
SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_valve_angle_tracking_status.json}"
SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_valve_angle_tracking_control.json}"
AUTO_WORKFLOW="${AUTO_WORKFLOW:-0}"
AUTO_START_POLICY="${AUTO_START_POLICY:-0}"
TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC="${TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC:-0.5}"
TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC:-2.0}"
ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC:-}"
DURATION_SEC="${DURATION_SEC:-}"
SEED="${SEED:-1}"

export TUNED_OVERLAY_CONFIG
export VALVE_ANGLE_OVERLAY_CONFIG
export EXTRA_OVERLAY_CONFIG
export TRACKING_IK_SPEED_FACTOR="${TRACKING_IK_SPEED_FACTOR:-}"
export TRACKING_IK_TRANS_WEIGHT="${TRACKING_IK_TRANS_WEIGHT:-}"
export TRACKING_IK_REG_WEIGHT="${TRACKING_IK_REG_WEIGHT:-}"
export TRACKING_IK_SMOOTH_WEIGHT="${TRACKING_IK_SMOOTH_WEIGHT:-}"
export TRACKING_IK_FILTER_WEIGHTS="${TRACKING_IK_FILTER_WEIGHTS:-}"
export TRACKING_UPPER_TAU_FF_SCALE="${TRACKING_UPPER_TAU_FF_SCALE:-}"
export TRACKING_UPPER_TAU_FF_CLIP="${TRACKING_UPPER_TAU_FF_CLIP:-}"
export TRACKING_WRIST_KP_SCALE="${TRACKING_WRIST_KP_SCALE:-}"
export TRACKING_WRIST_KD_SCALE="${TRACKING_WRIST_KD_SCALE:-}"
export VALVE_ATTACHMENT_MODE="${VALVE_ATTACHMENT_MODE:-}"
export VALVE_JOINT_DAMPING="${VALVE_JOINT_DAMPING:-}"
export VALVE_JOINT_FRICTIONLOSS="${VALVE_JOINT_FRICTIONLOSS:-}"
export VALVE_ANGLE_TARGETS_DEG="${VALVE_ANGLE_TARGETS_DEG:-}"
export VALVE_ANGLE_RANDOM_COUNT="${VALVE_ANGLE_RANDOM_COUNT:-}"
export VALVE_ANGLE_MIN_ABS_DEG="${VALVE_ANGLE_MIN_ABS_DEG:-}"
export VALVE_ANGLE_MAX_ABS_DEG="${VALVE_ANGLE_MAX_ABS_DEG:-}"
export VALVE_ANGLE_SPEED_DEG="${VALVE_ANGLE_SPEED_DEG:-}"
export VALVE_ANGLE_HOLD_SEC="${VALVE_ANGLE_HOLD_SEC:-}"
export VALVE_ANGLE_CONTROL_MODE="${VALVE_ANGLE_CONTROL_MODE:-}"
export VALVE_ANGLE_FEEDBACK_GAIN="${VALVE_ANGLE_FEEDBACK_GAIN:-}"
export VALVE_ANGLE_COMMAND_LEAD_DEG="${VALVE_ANGLE_COMMAND_LEAD_DEG:-}"
export VALVE_ANGLE_COMMAND_SPEED_DEG="${VALVE_ANGLE_COMMAND_SPEED_DEG:-}"
export VALVE_ANGLE_EXTRA_SETTLE_SEC="${VALVE_ANGLE_EXTRA_SETTLE_SEC:-}"
export VALVE_ANGLE_TOLERANCE_DEG="${VALVE_ANGLE_TOLERANCE_DEG:-}"

"${PYTHON_BIN}" - <<PY
import os
import yaml

with open("${BASE_CONFIG}") as file:
    config = yaml.safe_load(file)
with open("${OVERLAY_CONFIG}") as file:
    config.update(yaml.safe_load(file) or {})

overlay_paths = [
    os.environ["TUNED_OVERLAY_CONFIG"],
    os.environ["VALVE_ANGLE_OVERLAY_CONFIG"],
]
overlay_paths.extend(
    item.strip()
    for item in os.environ.get("EXTRA_OVERLAY_CONFIG", "").replace(",", ":").split(":")
    if item.strip()
)
for overlay_path in overlay_paths:
    with open(overlay_path) as file:
        config.update(yaml.safe_load(file) or {})

float_env_overrides = {
    "tracking_ik_speed_factor": "TRACKING_IK_SPEED_FACTOR",
    "tracking_ik_translational_weight": "TRACKING_IK_TRANS_WEIGHT",
    "tracking_ik_regularization_weight": "TRACKING_IK_REG_WEIGHT",
    "tracking_ik_smooth_weight": "TRACKING_IK_SMOOTH_WEIGHT",
    "tracking_upper_tau_ff_scale": "TRACKING_UPPER_TAU_FF_SCALE",
    "tracking_upper_tau_ff_clip": "TRACKING_UPPER_TAU_FF_CLIP",
    "valve_joint_damping": "VALVE_JOINT_DAMPING",
    "valve_joint_frictionloss": "VALVE_JOINT_FRICTIONLOSS",
    "valve_angle_tracking_min_abs_deg": "VALVE_ANGLE_MIN_ABS_DEG",
    "valve_angle_tracking_max_abs_deg": "VALVE_ANGLE_MAX_ABS_DEG",
    "valve_angle_tracking_speed_deg": "VALVE_ANGLE_SPEED_DEG",
    "valve_angle_tracking_hold_s": "VALVE_ANGLE_HOLD_SEC",
    "valve_angle_tracking_feedback_gain": "VALVE_ANGLE_FEEDBACK_GAIN",
    "valve_angle_tracking_command_lead_deg": "VALVE_ANGLE_COMMAND_LEAD_DEG",
    "valve_angle_tracking_command_speed_deg": "VALVE_ANGLE_COMMAND_SPEED_DEG",
    "valve_angle_tracking_extra_settle_s": "VALVE_ANGLE_EXTRA_SETTLE_SEC",
    "valve_angle_tracking_tolerance_deg": "VALVE_ANGLE_TOLERANCE_DEG",
}
for config_key, env_key in float_env_overrides.items():
    value = os.environ.get(env_key, "")
    if value:
        config[config_key] = float(value)

int_env_overrides = {
    "valve_angle_tracking_random_count": "VALVE_ANGLE_RANDOM_COUNT",
}
for config_key, env_key in int_env_overrides.items():
    value = os.environ.get(env_key, "")
    if value:
        config[config_key] = int(value)

value = os.environ.get("VALVE_ATTACHMENT_MODE", "")
if value:
    config["valve_attachment_mode"] = value

value = os.environ.get("VALVE_ANGLE_CONTROL_MODE", "")
if value:
    config["valve_angle_tracking_control_mode"] = value

targets = os.environ.get("VALVE_ANGLE_TARGETS_DEG", "")
if targets:
    config["valve_angle_tracking_targets_deg"] = [
        float(item.strip()) for item in targets.split(",") if item.strip()
    ]

filter_weights = os.environ.get("TRACKING_IK_FILTER_WEIGHTS", "")
if filter_weights:
    config["tracking_ik_filter_weights"] = [float(item) for item in filter_weights.split(",")]

for env_key, config_key in (
    ("TRACKING_WRIST_KP_SCALE", "tracking_motor_kp_scale_by_name"),
    ("TRACKING_WRIST_KD_SCALE", "tracking_motor_kd_scale_by_name"),
):
    value = os.environ.get(env_key, "")
    if value:
        config.setdefault(config_key, {})
        for joint_name in (
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ):
            config[config_key][joint_name] = float(value)

with open("${MERGED_CONFIG}", "w") as file:
    yaml.safe_dump(config, file, sort_keys=False)
PY

ARGS=()
if [[ "${AUTO_WORKFLOW}" == "1" ]]; then
  ARGS+=(--auto_workflow)
fi
if [[ "${AUTO_START_POLICY}" == "1" ]]; then
  ARGS+=(--auto_start_policy)
fi
if [[ -n "${DURATION_SEC}" ]]; then
  ARGS+=(--duration_s="${DURATION_SEC}")
fi
if [[ -n "${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC}" ]]; then
  ARGS+=(--elastic_release_delay_after_policy_start_s="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC}")
fi

"${PYTHON_BIN}" -u rl_policy/loco_manip/loco_manip_valve_angle_tracking_7dof_with_hand.py \
  --config="${MERGED_CONFIG}" \
  --model_path="${MODEL_PATH}" \
  --marker_file="${MARKER_FILE}" \
  --log_file="${LOG_FILE}" \
  --sim_status_file="${SIM_STATUS_FILE}" \
  --sim_control_file="${SIM_CONTROL_FILE}" \
  --tracking_delay_after_elastic_off_s="${TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC}" \
  --tracking_start_delay_s="${TRACKING_START_DELAY_SEC}" \
  --seed="${SEED}" \
  "${ARGS[@]}"
