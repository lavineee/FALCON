#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MARKER_FILE="${MARKER_FILE:-/tmp/falcon_valve_task_markers.json}"
LOG_FILE="${LOG_FILE:-/tmp/falcon_valve_task_metrics.csv}"
SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_valve_task_status.json}"
SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_valve_task_control.json}"
SIM_STARTUP_WAIT_SEC="${SIM_STARTUP_WAIT_SEC:-0.5}"
ELASTIC_LENGTH="${ELASTIC_LENGTH:--0.05}"
ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC:-1.0}"
TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC:-2.0}"
DURATION_SEC="${DURATION_SEC:-70}"

rm -f "${MARKER_FILE}" "${LOG_FILE}" "${SIM_STATUS_FILE}" "${SIM_CONTROL_FILE}"

cleanup() {
  if [[ -n "${SIM_PID:-}" ]]; then
    kill "${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ELASTIC_LENGTH="${ELASTIC_LENGTH}" \
MARKER_FILE="${MARKER_FILE}" \
SIM_STATUS_FILE="${SIM_STATUS_FILE}" \
SIM_CONTROL_FILE="${SIM_CONTROL_FILE}" \
./launch_valve_task_7dof_with_hand_sim.sh &
SIM_PID=$!

sleep "${SIM_STARTUP_WAIT_SEC}"

AUTO_WORKFLOW=0 \
AUTO_START_POLICY=1 \
ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC="${ELASTIC_RELEASE_DELAY_AFTER_POLICY_START_SEC}" \
TRACKING_START_DELAY_SEC="${TRACKING_START_DELAY_SEC}" \
DURATION_SEC="${DURATION_SEC}" \
MARKER_FILE="${MARKER_FILE}" \
LOG_FILE="${LOG_FILE}" \
SIM_STATUS_FILE="${SIM_STATUS_FILE}" \
SIM_CONTROL_FILE="${SIM_CONTROL_FILE}" \
./launch_valve_task_7dof_with_hand_policy.sh
