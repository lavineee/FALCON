#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PROJECT_ROOT="$(cd .. && pwd)"
MODE="${1:-auto}"
MODEL_ONLY_DURATION_SEC="${2:-${DURATION_SEC:-25}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/home/lavine/miniconda3/envs/fcreal/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
CONTACT_TURNING_OVERLAY_CONFIG="${CONTACT_TURNING_OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_valve_contact_turning_overlay.yaml}"
MARKER_FILE="${MARKER_FILE:-/tmp/falcon_valve_contact_turning_markers_${RUN_ID}.json}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/artifacts/demo_logs/valve_contact_turning_${RUN_ID}.csv}"
SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_valve_contact_turning_status_${RUN_ID}.json}"
SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_valve_contact_turning_control_${RUN_ID}.json}"
SIM_STARTUP_WAIT_SEC="${SIM_STARTUP_WAIT_SEC:-1.5}"
ELASTIC_LENGTH="${ELASTIC_LENGTH:--0.15}"
ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC:-1.2}"
# 接触转阀门会在策略启动后立刻移动右臂。这里给下肢策略多留一点站稳时间，
# 避免有效抓取测试被初始倒地或大倾角污染。
TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC:-3.2}"
DURATION_SEC="${DURATION_SEC:-75}"
VALVE_TURN_TARGET_SEQUENCE_DEG="${VALVE_TURN_TARGET_SEQUENCE_DEG:-10,-20,30}"
# 序列展示里 base 靠近/yaw 调整属于正常全身补偿，只记录，不作为失败条件。
# 失败主要看是否摔倒、撞上阀门、手松开或接触异常。
VALVE_TURN_ABORT_MAX_BASE_APPROACH_M="${VALVE_TURN_ABORT_MAX_BASE_APPROACH_M:-0.0}"
VALVE_TURN_ABORT_MIN_BASE_TO_VALVE_X_M="${VALVE_TURN_ABORT_MIN_BASE_TO_VALVE_X_M:-0.0}"
VALVE_TURN_ABORT_MAX_BASE_YAW_DRIFT_DEG="${VALVE_TURN_ABORT_MAX_BASE_YAW_DRIFT_DEG:-0.0}"
VALVE_TURN_ABORT_BASE_TILT_DEG="${VALVE_TURN_ABORT_BASE_TILT_DEG:-25.0}"

rm -f "${MARKER_FILE}" "${SIM_STATUS_FILE}" "${SIM_CONTROL_FILE}"

start_background() {
  if command -v setsid >/dev/null 2>&1; then
    setsid env "$@" &
  else
    env "$@" &
  fi
  STARTED_PID="$!"
}

stop_background() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi
  kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  sleep 0.2
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_background "${POLICY_PID:-}"
  stop_background "${SIM_PID:-}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "${LOG_FILE}")"

start_background \
  MPLCONFIGDIR="${MPLCONFIGDIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  ELASTIC_LENGTH="${ELASTIC_LENGTH}" \
  EXTRA_OVERLAY_CONFIG="${CONTACT_TURNING_OVERLAY_CONFIG}" \
  MARKER_FILE="${MARKER_FILE}" \
  SIM_STATUS_FILE="${SIM_STATUS_FILE}" \
  SIM_CONTROL_FILE="${SIM_CONTROL_FILE}" \
  ./launch_valve_grasp_contact_7dof_with_hand_sim.sh
SIM_PID="${STARTED_PID}"

if [[ "${MODE}" == "--model-only" || "${MODE}" == "model-only" ]]; then
  echo "[VALVE_CONTACT_TURN] model-only viewer for ${MODEL_ONLY_DURATION_SEC}s"
  sleep "${MODEL_ONLY_DURATION_SEC}"
  exit 0
fi

sleep "${SIM_STARTUP_WAIT_SEC}"

start_background \
  MPLCONFIGDIR="${MPLCONFIGDIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  AUTO_WORKFLOW=0 \
  AUTO_START_POLICY=1 \
  ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC}" \
  TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC}" \
  DURATION_SEC="${DURATION_SEC}" \
  VALVE_TURN_TARGET_SEQUENCE_DEG="${VALVE_TURN_TARGET_SEQUENCE_DEG}" \
  VALVE_TURN_ABORT_BASE_TILT_DEG="${VALVE_TURN_ABORT_BASE_TILT_DEG}" \
  VALVE_TURN_ABORT_MAX_BASE_APPROACH_M="${VALVE_TURN_ABORT_MAX_BASE_APPROACH_M}" \
  VALVE_TURN_ABORT_MIN_BASE_TO_VALVE_X_M="${VALVE_TURN_ABORT_MIN_BASE_TO_VALVE_X_M}" \
  VALVE_TURN_ABORT_MAX_BASE_YAW_DRIFT_DEG="${VALVE_TURN_ABORT_MAX_BASE_YAW_DRIFT_DEG}" \
  EXTRA_OVERLAY_CONFIG="${CONTACT_TURNING_OVERLAY_CONFIG}" \
  MARKER_FILE="${MARKER_FILE}" \
  LOG_FILE="${LOG_FILE}" \
  SIM_STATUS_FILE="${SIM_STATUS_FILE}" \
  SIM_CONTROL_FILE="${SIM_CONTROL_FILE}" \
  ./launch_valve_grasp_contact_7dof_with_hand_policy.sh
POLICY_PID="${STARTED_PID}"

echo "[VALVE_CONTACT_TURN] log=${LOG_FILE}"
echo "[VALVE_CONTACT_TURN] marker=${MARKER_FILE}"
wait "${POLICY_PID}"
POLICY_PID=""
