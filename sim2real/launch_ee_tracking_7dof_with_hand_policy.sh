#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-config/g1/g1_29dof_falcon.yaml}"
OVERLAY_CONFIG="${OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_tracking_overlay.yaml}"
MERGED_CONFIG="${MERGED_CONFIG:-/tmp/falcon_g1_29dof_with_hand_tracking.yaml}"
MODEL_PATH="${MODEL_PATH:-models/falcon/g1_29dof.onnx}"
MARKER_FILE="${MARKER_FILE:-/tmp/falcon_ee_tracking_7dof_with_hand_markers.json}"
LOG_FILE="${LOG_FILE:-/tmp/falcon_ee_tracking_7dof_with_hand_metrics.csv}"
SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_sim_status_7dof_with_hand.json}"
SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_sim_control_7dof_with_hand.json}"
SAMPLE_PERIOD_SEC="${SAMPLE_PERIOD_SEC:-5.0}"
X_RANGE="${X_RANGE:-0.20,0.42}"
Y_RANGE="${Y_RANGE:--0.26,-0.04}"
Z_RANGE="${Z_RANGE:--0.02,0.24}"
SEED="${SEED:-1}"
AUTO_WORKFLOW="${AUTO_WORKFLOW:-0}"
AUTO_START_POLICY="${AUTO_START_POLICY:-0}"
TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC="${TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC:-0.5}"
TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC:-2.0}"
ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC:-}"
DURATION_SEC="${DURATION_SEC:-}"
HAND_CLOSE_ERROR_M="${HAND_CLOSE_ERROR_M:-}"

"${PYTHON_BIN}" - <<PY
import yaml

with open("${BASE_CONFIG}") as file:
    config = yaml.safe_load(file)
with open("${OVERLAY_CONFIG}") as file:
    overlay = yaml.safe_load(file)
config.update(overlay)
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
if [[ -n "${HAND_CLOSE_ERROR_M}" ]]; then
  ARGS+=(--hand_close_error_m="${HAND_CLOSE_ERROR_M}")
fi

"${PYTHON_BIN}" -u rl_policy/loco_manip/loco_manip_ee_tracking_7dof_with_hand_test.py \
  --config="${MERGED_CONFIG}" \
  --model_path="${MODEL_PATH}" \
  --marker_file="${MARKER_FILE}" \
  --log_file="${LOG_FILE}" \
  --sim_status_file="${SIM_STATUS_FILE}" \
  --sim_control_file="${SIM_CONTROL_FILE}" \
  --sample_period_s="${SAMPLE_PERIOD_SEC}" \
  --x_range="${X_RANGE}" \
  --y_range="${Y_RANGE}" \
  --z_range="${Z_RANGE}" \
  --seed="${SEED}" \
  --tracking_delay_after_elastic_off_s="${TRACKING_DELAY_AFTER_ELASTIC_OFF_SEC}" \
  --tracking_start_delay_s="${TRACKING_START_DELAY_SEC}" \
  "${ARGS[@]}"
