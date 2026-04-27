#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TUNED_OVERLAY_CONFIG="${TUNED_OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml}"
GRASP_OVERLAY_CONFIG="${GRASP_OVERLAY_CONFIG:-config/g1/g1_29dof_with_hand_valve_grasp_contact_overlay.yaml}"
USER_EXTRA_OVERLAY_CONFIG="${EXTRA_OVERLAY_CONFIG:-}"

if [[ -n "${USER_EXTRA_OVERLAY_CONFIG}" ]]; then
  EXTRA_OVERLAY_CONFIG="${TUNED_OVERLAY_CONFIG}:${GRASP_OVERLAY_CONFIG}:${USER_EXTRA_OVERLAY_CONFIG}"
else
  EXTRA_OVERLAY_CONFIG="${TUNED_OVERLAY_CONFIG}:${GRASP_OVERLAY_CONFIG}"
fi

export EXTRA_OVERLAY_CONFIG
exec ./launch_ee_tracking_7dof_with_hand_sim.sh
