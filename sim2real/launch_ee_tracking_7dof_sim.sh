#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MARKER_FILE="${MARKER_FILE:-/tmp/falcon_ee_tracking_7dof_markers.json}"
export SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_sim_control_7dof.json}"
export SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_sim_status_7dof.json}"
export ELASTIC_LENGTH="${ELASTIC_LENGTH:--0.05}"

exec ./launch_ee_tracking_sim.sh
