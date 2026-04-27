#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MARKER_FILE="${MARKER_FILE:-/tmp/falcon_valve_grasp_contact_markers.json}"
LOG_FILE="${LOG_FILE:-/tmp/falcon_valve_grasp_contact_metrics.csv}"
SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_valve_grasp_contact_status.json}"
SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_valve_grasp_contact_control.json}"
SIM_STARTUP_WAIT_SEC="${SIM_STARTUP_WAIT_SEC:-0.5}"
ELASTIC_LENGTH="${ELASTIC_LENGTH:--0.05}"
ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC:-1.0}"
TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC:-2.0}"
DURATION_SEC="${DURATION_SEC:-35}"

rm -f "${MARKER_FILE}" "${LOG_FILE}" "${SIM_STATUS_FILE}" "${SIM_CONTROL_FILE}"

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

start_background \
  ELASTIC_LENGTH="${ELASTIC_LENGTH}" \
  MARKER_FILE="${MARKER_FILE}" \
  SIM_STATUS_FILE="${SIM_STATUS_FILE}" \
  SIM_CONTROL_FILE="${SIM_CONTROL_FILE}" \
  ./launch_valve_grasp_contact_7dof_with_hand_sim.sh
SIM_PID="${STARTED_PID}"

sleep "${SIM_STARTUP_WAIT_SEC}"

start_background \
  AUTO_WORKFLOW=0 \
  AUTO_START_POLICY=1 \
  ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC}" \
  TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC}" \
  DURATION_SEC="${DURATION_SEC}" \
  MARKER_FILE="${MARKER_FILE}" \
  LOG_FILE="${LOG_FILE}" \
  SIM_STATUS_FILE="${SIM_STATUS_FILE}" \
  SIM_CONTROL_FILE="${SIM_CONTROL_FILE}" \
  ./launch_valve_grasp_contact_7dof_with_hand_policy.sh
POLICY_PID="${STARTED_PID}"
wait "${POLICY_PID}"
POLICY_PID=""
