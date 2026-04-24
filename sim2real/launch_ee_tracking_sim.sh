#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG_PATH="${CONFIG_PATH:-config/g1/g1_29dof_falcon.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROBOT_SCENE="${ROBOT_SCENE:-../humanoidverse/data/robots/g1/scene_g1_29dof_freebase_camera.xml}"
MARKER_FILE="${MARKER_FILE:-/tmp/falcon_ee_tracking_markers.json}"
SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_sim_status.json}"
SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_sim_control.json}"
ELASTIC_LENGTH="${ELASTIC_LENGTH:--0.05}"
LOG_INTERVAL_SEC="${LOG_INTERVAL_SEC:-5.0}"
DISABLE_ELASTIC_AFTER_SEC="${DISABLE_ELASTIC_AFTER_SEC:-}"

"${PYTHON_BIN}" -u - <<PY
import sys
import yaml

sys.path.append("../")

from sim2real.sim_env.loco_manip import LocoManipSimulator

with open("${CONFIG_PATH}") as file:
    config = yaml.safe_load(file)

config.update({
    "ROBOT_SCENE": "${ROBOT_SCENE}",
    "enable_live_plot": False,
    "auto_elastic_length": float("${ELASTIC_LENGTH}"),
    "auto_log_interval_s": float("${LOG_INTERVAL_SEC}"),
    "sim_status_file": "${SIM_STATUS_FILE}",
    "sim_control_file": "${SIM_CONTROL_FILE}",
    "ee_marker_file": "${MARKER_FILE}",
})

if "${DISABLE_ELASTIC_AFTER_SEC}":
    config["auto_disable_elastic_after_s"] = float("${DISABLE_ELASTIC_AFTER_SEC}")

simulation = LocoManipSimulator(config)
simulation.sim_thread.start()
simulation.sim_thread.join()
PY
