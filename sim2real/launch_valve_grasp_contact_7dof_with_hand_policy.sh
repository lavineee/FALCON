#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-config/g1/g1_29dof_falcon.yaml}"
OVERLAY_CONFIG="${OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_tracking_overlay.yaml}"
TUNED_OVERLAY_CONFIG="${TUNED_OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml}"
GRASP_OVERLAY_CONFIG="${GRASP_OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_valve_grasp_contact_overlay.yaml}"
EXTRA_OVERLAY_CONFIG="${EXTRA_OVERLAY_CONFIG:-}"
MERGED_CONFIG="${MERGED_CONFIG:-/tmp/falcon_g1_29dof_with_hand_valve_grasp_contact.yaml}"
MODEL_PATH="${MODEL_PATH:-models/falcon/g1_29dof.onnx}"
MARKER_FILE="${MARKER_FILE:-/tmp/falcon_valve_grasp_contact_markers.json}"
LOG_FILE="${LOG_FILE:-/tmp/falcon_valve_grasp_contact_metrics.csv}"
SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_valve_grasp_contact_status.json}"
SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_valve_grasp_contact_control.json}"
AUTO_WORKFLOW="${AUTO_WORKFLOW:-0}"
AUTO_START_POLICY="${AUTO_START_POLICY:-0}"
TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC="${TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC:-0.5}"
TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC:-2.0}"
ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC:-}"
DURATION_SEC="${DURATION_SEC:-}"

export TUNED_OVERLAY_CONFIG
export GRASP_OVERLAY_CONFIG
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
export HAND_KP="${HAND_KP:-}"
export HAND_KD="${HAND_KD:-}"
export HAND_COMMAND_RAMP_S="${HAND_COMMAND_RAMP_S:-}"
export VALVE_PREGRASP_OFFSET_M="${VALVE_PREGRASP_OFFSET_M:-}"
export VALVE_PREGRASP_ERROR_M="${VALVE_PREGRASP_ERROR_M:-}"
export VALVE_PREGRASP_MAX_S="${VALVE_PREGRASP_MAX_S:-}"
export VALVE_APPROACH_MAX_S="${VALVE_APPROACH_MAX_S:-}"
export VALVE_GRASP_ERROR_M="${VALVE_GRASP_ERROR_M:-}"
export VALVE_CLOSE_WAIT_S="${VALVE_CLOSE_WAIT_S:-}"
export VALVE_CLOSE_MAX_S="${VALVE_CLOSE_MAX_S:-}"
export VALVE_HOLD_ENTRY_MIN_SUCCESS_S="${VALVE_HOLD_ENTRY_MIN_SUCCESS_S:-}"
export VALVE_HOLD_ENTRY_MAX_ABS_VALVE_VEL="${VALVE_HOLD_ENTRY_MAX_ABS_VALVE_VEL:-}"
export VALVE_HOLD_S="${VALVE_HOLD_S:-}"
export VALVE_HOLD_LATCH_MODE="${VALVE_HOLD_LATCH_MODE:-}"
export VALVE_GRASP_SUCCESS_MIN_CONTACTS="${VALVE_GRASP_SUCCESS_MIN_CONTACTS:-}"
export VALVE_GRASP_SUCCESS_MIN_FORCE_N="${VALVE_GRASP_SUCCESS_MIN_FORCE_N:-}"
export VALVE_GRASP_SUCCESS_REQUIRE_THUMB="${VALVE_GRASP_SUCCESS_REQUIRE_THUMB:-}"
export VALVE_GRASP_SUCCESS_REQUIRE_FINGER="${VALVE_GRASP_SUCCESS_REQUIRE_FINGER:-}"
export VALVE_REQUIRE_PALM_CONTACT_TO_CLOSE="${VALVE_REQUIRE_PALM_CONTACT_TO_CLOSE:-}"
export VALVE_PALM_CLOSE_MIN_FORCE_N="${VALVE_PALM_CLOSE_MIN_FORCE_N:-}"
export VALVE_PALM_CLOSE_MAX_ERROR_M="${VALVE_PALM_CLOSE_MAX_ERROR_M:-}"
export VALVE_GRASP_FORWARD_AXIS_SIGN="${VALVE_GRASP_FORWARD_AXIS_SIGN:-}"
export VALVE_GRASP_RADIAL_AXIS_SIGN="${VALVE_GRASP_RADIAL_AXIS_SIGN:-}"
export VALVE_CLOSE_ON_CONTACT="${VALVE_CLOSE_ON_CONTACT:-}"
export VALVE_GRASP_TARGET_BIAS_BASE_M="${VALVE_GRASP_TARGET_BIAS_BASE_M:-}"
export VALVE_GRASP_POINT_OFFSET_EE_M="${VALVE_GRASP_POINT_OFFSET_EE_M:-}"
export VALVE_GRASP_RADIAL_INSET_M="${VALVE_GRASP_RADIAL_INSET_M:-}"
export VALVE_ENABLE_TURN="${VALVE_ENABLE_TURN:-}"
export VALVE_TURN_TARGET_DEG="${VALVE_TURN_TARGET_DEG:-}"
export VALVE_TURN_TARGET_SEQUENCE_DEG="${VALVE_TURN_TARGET_SEQUENCE_DEG:-}"
export VALVE_TURN_SPEED_DEG="${VALVE_TURN_SPEED_DEG:-}"
export VALVE_TURN_CONTROL_MODE="${VALVE_TURN_CONTROL_MODE:-}"
export VALVE_TURN_FEEDBACK_GAIN="${VALVE_TURN_FEEDBACK_GAIN:-}"
export VALVE_TURN_COMMAND_LEAD_DEG="${VALVE_TURN_COMMAND_LEAD_DEG:-}"
export VALVE_TURN_COMMAND_SPEED_DEG="${VALVE_TURN_COMMAND_SPEED_DEG:-}"
export VALVE_TURN_EXTRA_SETTLE_SEC="${VALVE_TURN_EXTRA_SETTLE_SEC:-}"
export VALVE_TURN_TOLERANCE_DEG="${VALVE_TURN_TOLERANCE_DEG:-}"
export VALVE_TURN_HOLD_SEC="${VALVE_TURN_HOLD_SEC:-}"
export VALVE_TURN_HOLD_ANGLE_BRAKE_ENABLED="${VALVE_TURN_HOLD_ANGLE_BRAKE_ENABLED:-}"
export VALVE_TURN_ENTRY_MIN_HOLD_SEC="${VALVE_TURN_ENTRY_MIN_HOLD_SEC:-}"
export VALVE_TURN_ENTRY_MAX_SLIP_M="${VALVE_TURN_ENTRY_MAX_SLIP_M:-}"
export VALVE_TURN_ENTRY_REQUIRE_SUCCESS="${VALVE_TURN_ENTRY_REQUIRE_SUCCESS:-}"
export VALVE_TURN_ABORT_BASE_TILT_DEG="${VALVE_TURN_ABORT_BASE_TILT_DEG:-}"
export VALVE_TURN_ABORT_MIN_BASE_TO_VALVE_X_M="${VALVE_TURN_ABORT_MIN_BASE_TO_VALVE_X_M:-}"
export VALVE_TURN_ABORT_MAX_BASE_APPROACH_M="${VALVE_TURN_ABORT_MAX_BASE_APPROACH_M:-}"
export VALVE_TURN_ABORT_MAX_BASE_YAW_DRIFT_DEG="${VALVE_TURN_ABORT_MAX_BASE_YAW_DRIFT_DEG:-}"
export VALVE_TURN_RADIUS_MODE="${VALVE_TURN_RADIUS_MODE:-}"
export VALVE_TURN_ROTATE_GRASP_ORIENTATION="${VALVE_TURN_ROTATE_GRASP_ORIENTATION:-}"
export VALVE_JOINT_DAMPING="${VALVE_JOINT_DAMPING:-}"
export VALVE_JOINT_FRICTIONLOSS="${VALVE_JOINT_FRICTIONLOSS:-}"
export VALVE_ANGLE_LOCK_ENABLED="${VALVE_ANGLE_LOCK_ENABLED:-}"

"${PYTHON_BIN}" - <<PY
import os
import yaml

with open("${BASE_CONFIG}") as file:
    config = yaml.safe_load(file)
with open("${OVERLAY_CONFIG}") as file:
    config.update(yaml.safe_load(file) or {})

overlay_paths = [
    os.environ["TUNED_OVERLAY_CONFIG"],
    os.environ["GRASP_OVERLAY_CONFIG"],
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
    "hand_kp": "HAND_KP",
    "hand_kd": "HAND_KD",
    "hand_command_ramp_s": "HAND_COMMAND_RAMP_S",
    "valve_pregrasp_offset_m": "VALVE_PREGRASP_OFFSET_M",
    "valve_pregrasp_error_m": "VALVE_PREGRASP_ERROR_M",
    "valve_pregrasp_max_s": "VALVE_PREGRASP_MAX_S",
    "valve_approach_max_s": "VALVE_APPROACH_MAX_S",
    "valve_grasp_error_m": "VALVE_GRASP_ERROR_M",
    "valve_grasp_radial_inset_m": "VALVE_GRASP_RADIAL_INSET_M",
    "valve_close_wait_s": "VALVE_CLOSE_WAIT_S",
    "valve_close_max_s": "VALVE_CLOSE_MAX_S",
    "valve_hold_entry_min_success_s": "VALVE_HOLD_ENTRY_MIN_SUCCESS_S",
    "valve_hold_entry_max_abs_valve_vel": "VALVE_HOLD_ENTRY_MAX_ABS_VALVE_VEL",
    "valve_hold_s": "VALVE_HOLD_S",
    "valve_grasp_success_min_force_n": "VALVE_GRASP_SUCCESS_MIN_FORCE_N",
    "valve_palm_close_min_force_n": "VALVE_PALM_CLOSE_MIN_FORCE_N",
    "valve_palm_close_max_error_m": "VALVE_PALM_CLOSE_MAX_ERROR_M",
    "valve_grasp_forward_axis_sign": "VALVE_GRASP_FORWARD_AXIS_SIGN",
    "valve_grasp_radial_axis_sign": "VALVE_GRASP_RADIAL_AXIS_SIGN",
    "valve_turn_target_deg": "VALVE_TURN_TARGET_DEG",
    "valve_turn_speed_deg": "VALVE_TURN_SPEED_DEG",
    "valve_turn_feedback_gain": "VALVE_TURN_FEEDBACK_GAIN",
    "valve_turn_command_lead_deg": "VALVE_TURN_COMMAND_LEAD_DEG",
    "valve_turn_command_speed_deg": "VALVE_TURN_COMMAND_SPEED_DEG",
    "valve_turn_extra_settle_s": "VALVE_TURN_EXTRA_SETTLE_SEC",
    "valve_turn_tolerance_deg": "VALVE_TURN_TOLERANCE_DEG",
    "valve_turn_hold_s": "VALVE_TURN_HOLD_SEC",
    "valve_turn_entry_min_hold_s": "VALVE_TURN_ENTRY_MIN_HOLD_SEC",
    "valve_turn_entry_max_slip_m": "VALVE_TURN_ENTRY_MAX_SLIP_M",
    "valve_turn_abort_base_tilt_deg": "VALVE_TURN_ABORT_BASE_TILT_DEG",
    "valve_turn_abort_min_base_to_valve_x_m": "VALVE_TURN_ABORT_MIN_BASE_TO_VALVE_X_M",
    "valve_turn_abort_max_base_approach_m": "VALVE_TURN_ABORT_MAX_BASE_APPROACH_M",
    "valve_turn_abort_max_base_yaw_drift_deg": "VALVE_TURN_ABORT_MAX_BASE_YAW_DRIFT_DEG",
    "valve_joint_damping": "VALVE_JOINT_DAMPING",
    "valve_joint_frictionloss": "VALVE_JOINT_FRICTIONLOSS",
}
for config_key, env_key in float_env_overrides.items():
    value = os.environ.get(env_key, "")
    if value:
        config[config_key] = float(value)

value = os.environ.get("VALVE_GRASP_SUCCESS_MIN_CONTACTS", "")
if value:
    config["valve_grasp_success_min_contacts"] = int(value)

value = os.environ.get("VALVE_HOLD_LATCH_MODE", "")
if value:
    config["valve_hold_latch_mode"] = value

value = os.environ.get("VALVE_TURN_CONTROL_MODE", "")
if value:
    config["valve_turn_control_mode"] = value

value = os.environ.get("VALVE_TURN_TARGET_SEQUENCE_DEG", "")
if value:
    config["valve_turn_target_sequence_deg"] = value

value = os.environ.get("VALVE_TURN_RADIUS_MODE", "")
if value:
    config["valve_turn_radius_mode"] = value

for env_key, config_key in (
    ("VALVE_GRASP_SUCCESS_REQUIRE_THUMB", "valve_grasp_success_require_thumb"),
    ("VALVE_GRASP_SUCCESS_REQUIRE_FINGER", "valve_grasp_success_require_finger"),
    ("VALVE_REQUIRE_PALM_CONTACT_TO_CLOSE", "valve_require_palm_contact_to_close"),
    ("VALVE_CLOSE_ON_CONTACT", "valve_close_on_contact"),
    ("VALVE_ENABLE_TURN", "valve_enable_turn"),
    ("VALVE_TURN_ENTRY_REQUIRE_SUCCESS", "valve_turn_entry_require_success"),
    ("VALVE_TURN_ROTATE_GRASP_ORIENTATION", "valve_turn_rotate_grasp_orientation"),
    ("VALVE_TURN_HOLD_ANGLE_BRAKE_ENABLED", "valve_turn_hold_angle_brake_enabled"),
    ("VALVE_ANGLE_LOCK_ENABLED", "valve_angle_lock_enabled"),
):
    value = os.environ.get(env_key, "")
    if value:
        config[config_key] = value.lower() in ("1", "true", "yes", "on")

bias = os.environ.get("VALVE_GRASP_TARGET_BIAS_BASE_M", "")
if bias:
    config["valve_grasp_target_bias_base_m"] = [float(item) for item in bias.split(",")]

offset = os.environ.get("VALVE_GRASP_POINT_OFFSET_EE_M", "")
if offset:
    config["valve_grasp_point_offset_ee_m"] = [float(item) for item in offset.split(",")]

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

"${PYTHON_BIN}" -u rl_policy/loco_manip/loco_manip_valve_grasp_contact_7dof_with_hand.py \
  --config="${MERGED_CONFIG}" \
  --model_path="${MODEL_PATH}" \
  --marker_file="${MARKER_FILE}" \
  --log_file="${LOG_FILE}" \
  --sim_status_file="${SIM_STATUS_FILE}" \
  --sim_control_file="${SIM_CONTROL_FILE}" \
  --tracking_delay_after_elastic_off_s="${TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC}" \
  --tracking_start_delay_s="${TRACKING_START_DELAY_SEC}" \
  "${ARGS[@]}"
