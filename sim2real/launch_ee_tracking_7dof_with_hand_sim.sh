#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-config/g1/g1_29dof_falcon.yaml}"
OVERLAY_CONFIG="${OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_tracking_overlay.yaml}"
EXTRA_OVERLAY_CONFIG="${EXTRA_OVERLAY_CONFIG:-}"
MARKER_FILE="${MARKER_FILE:-/tmp/falcon_ee_tracking_7dof_with_hand_markers.json}"
SIM_CONTROL_FILE="${SIM_CONTROL_FILE:-/tmp/falcon_sim_control_7dof_with_hand.json}"
SIM_STATUS_FILE="${SIM_STATUS_FILE:-/tmp/falcon_sim_status_7dof_with_hand.json}"
ELASTIC_LENGTH="${ELASTIC_LENGTH:--0.05}"
LOG_INTERVAL_SEC="${LOG_INTERVAL_SEC:-5.0}"
DISABLE_ELASTIC_AFTER_SEC="${DISABLE_ELASTIC_AFTER_SEC:-}"

export EXTRA_OVERLAY_CONFIG

"${PYTHON_BIN}" -u - <<PY
import os
import sys
import yaml

sys.path.append("../")

from sim2real.sim_env.loco_manip_with_hand import LocoManipWithHandSimulator

with open("${BASE_CONFIG}") as file:
    config = yaml.safe_load(file)
with open("${OVERLAY_CONFIG}") as file:
    overlay = yaml.safe_load(file)
config.update(overlay or {})
extra_overlay_configs = [
    item.strip()
    for item in os.environ.get("EXTRA_OVERLAY_CONFIG", "").replace(",", ":").split(":")
    if item.strip()
]
for overlay_path in extra_overlay_configs:
    with open(overlay_path) as file:
        config.update(yaml.safe_load(file) or {})

config.update({
    "enable_live_plot": False,
    "auto_elastic_length": float("${ELASTIC_LENGTH}"),
    "auto_log_interval_s": float("${LOG_INTERVAL_SEC}"),
    "sim_status_file": "${SIM_STATUS_FILE}",
    "sim_control_file": "${SIM_CONTROL_FILE}",
    "ee_marker_file": "${MARKER_FILE}",
})

if "${DISABLE_ELASTIC_AFTER_SEC}":
    config["auto_disable_elastic_after_s"] = float("${DISABLE_ELASTIC_AFTER_SEC}")

simulation = LocoManipWithHandSimulator(config)
simulation.sim_thread.start()
simulation.sim_thread.join()
PY
