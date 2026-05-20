#!/usr/bin/env python3
"""Thin real-G1 valve approach runner with an external right-hand DDS backend."""

import argparse
import sys
from pathlib import Path

import yaml

THIS_FILE = Path(__file__).resolve()
SIM2REAL_ROOT = THIS_FILE.parents[2]
REPO_ROOT = SIM2REAL_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.rl_policy.loco_manip.loco_manip_ee_tracking_test import apply_named_motor_gain_scales
from sim2real.rl_policy.loco_manip.loco_manip_valve_grasp_contact_7dof_with_hand import (
    LocoManipValveGraspContact7DofWithHandPolicy,
)

DEFAULT_CONFIGS = [
    "config/g1/g1_29dof_falcon_real.yaml",
    "config/g1/g1_29dof_with_hand_tracking_overlay.yaml",
    "config/g1/g1_29dof_with_hand_tracking_tuned_overlay.yaml",
    "config/g1/g1_29dof_with_hand_valve_grasp_contact_overlay.yaml",
    "config/g1/g1_29dof_with_hand_valve_real_overlay.yaml",
]

SIM_ONLY_FALSE_KEYS = (
    "valve_contact_assist_enabled",
    "valve_pre_turn_angle_lock_enabled",
    "valve_pre_turn_angle_hold_enabled",
    "valve_angle_lock_enabled",
    "valve_angle_hold_enabled",
    "valve_contact_tracking_compensation_enabled",
    "valve_close_on_contact",
    "valve_require_palm_contact_to_close",
    "valve_close_require_real_hand_contact",
)

REAL_REQUIRED_FALSE_KEYS = (
    "use_sim_status_file",
    "use_sim_control_file",
    "use_mujoco_weld",
    "use_sim_contact_force",
    "use_sim_valve_angle_truth",
    "use_sim_completion_flag",
)


def _resolve_path(path):
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    for root in (Path.cwd(), SIM2REAL_ROOT, REPO_ROOT):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return SIM2REAL_ROOT / candidate


def _load_config(paths):
    config = {}
    for path in paths:
        resolved = _resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"config file not found: {path} -> {resolved}")
        with open(resolved) as file:
            config.update(yaml.safe_load(file) or {})
    return config


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _apply_real_safety_overrides(config):
    config["disable_keyboard_listener"] = True
    config["runtime_mode"] = "real"
    config["valve_freeze_grasp_world_after_start"] = False
    config["valve_latch_grasp_frame_on_close"] = bool(config.get("real_vision_freeze_after_close", True))
    config["vision_close_latch_source"] = "current_control"
    config["vision_close_latch_frame"] = "base"
    config["real_vision_fail_fast_required_fields"] = True
    config["inspire_initialize_channel_factory"] = False
    for key in SIM_ONLY_FALSE_KEYS:
        config[key] = False
    return config


def _validate_real_config(config):
    errors = []
    if str(config.get("runtime_mode", "")).strip().lower() != "real":
        errors.append("runtime_mode must be real")
    if str(config.get("hand_backend", "")).strip().lower() != "inspire_dds":
        errors.append("hand_backend must be inspire_dds")
    if str(config.get("inspire_hand_side", "right")).strip().lower() != "right":
        errors.append("only inspire_hand_side=right is supported by this runner")
    if str(config.get("valve_geometry_source", "")).strip().lower() not in ("real_vision", "vision_json"):
        errors.append("valve_geometry_source must be real_vision")
    if _as_bool(config.get("valve_enable_turn", False)):
        errors.append("valve_enable_turn must be false for phase-1 real approach")
    if _as_bool(config.get("allow_manual_geometry_fallback", False)):
        errors.append("allow_manual_geometry_fallback must be false for default real runs")
    for key in REAL_REQUIRED_FALSE_KEYS:
        if _as_bool(config.get(key, False)):
            errors.append(f"{key} must be false")
    if errors:
        raise ValueError("invalid real valve config:\n  - " + "\n  - ".join(errors))


def _print_real_summary(config, check_only=False):
    print("[VALVE_REAL] runtime_mode: real")
    print(f"[VALVE_REAL] PC DDS interface: {config.get('INTERFACE', '<default>')}")
    print(f"[VALVE_REAL] DDS channel: {config.get('DOMAIN_ID', '<default>')}")
    print(f"[VALVE_REAL] hand backend: {config.get('hand_backend')}")
    print("[VALVE_REAL] right hand channel only: cmd[0:6]")
    print(f"[VALVE_REAL] geometry source: {config.get('valve_geometry_source')}")
    print(f"[VALVE_REAL] vision status file: {config.get('real_vision_status_file')}")
    print("[VALVE_REAL] turn disabled")
    print("[VALVE_REAL] sim-only truth disabled")
    if check_only:
        print("[VALVE_REAL] config check only; policy was not started")


class LocoManipValveRealWithHandPolicy(LocoManipValveGraspContact7DofWithHandPolicy):
    """Real runner policy shim: reuse the valve state machine, disable sim-only hooks."""

    def _warn_real_noop_once(self, key, message):
        warned = getattr(self, "_real_noop_warnings", set())
        if key in warned:
            return
        warned.add(key)
        self._real_noop_warnings = warned
        self.logger.warning(message)

    def _write_attach_state(self, enabled):
        if enabled:
            self._warn_real_noop_once("attach", "[VALVE_REAL] sim weld/attach is disabled on real runner")

    def _set_contact_assist(self, enabled):
        if enabled:
            self._warn_real_noop_once(
                "contact_assist",
                "[VALVE_REAL] sim contact assist is disabled on real runner",
            )
        self._contact_assist_active = False

    def _write_valve_angle_control(self, lock_enabled, hold_enabled, target=None):
        if lock_enabled or hold_enabled:
            self._warn_real_noop_once(
                "angle_control",
                "[VALVE_REAL] sim valve angle lock/hold is disabled on real runner",
            )

    def _write_valve_angle_lock(self, enabled, target=None):
        self._write_valve_angle_control(bool(enabled), False, target)

    def _write_valve_angle_hold(self, enabled, target=None):
        self._write_valve_angle_control(False, bool(enabled), target)


def main():
    parser = argparse.ArgumentParser(description="Run real valve approach with external right hand")
    parser.add_argument("--config", action="append", default=None, help="YAML config/overlay path; may repeat")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--marker_file", default="/tmp/falcon_valve_real_with_hand_markers.json")
    parser.add_argument("--log_file", default="/tmp/falcon_valve_real_with_hand_metrics.csv")
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument("--tracking_start_delay_s", type=float, default=0.0)
    parser.add_argument("--tracking_delay_after_elastic_off_s", type=float, default=0.0)
    parser.add_argument("--check_config_only", action="store_true")
    args = parser.parse_args()

    config_paths = args.config or DEFAULT_CONFIGS
    config = _apply_real_safety_overrides(_load_config(config_paths))
    _validate_real_config(config)
    _print_real_summary(config, check_only=args.check_config_only)
    if args.check_config_only:
        return

    apply_named_motor_gain_scales(config)
    model_path = args.model_path or config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path or config model_path")

    policy = LocoManipValveRealWithHandPolicy(
        config,
        model_path,
        marker_file=args.marker_file,
        log_file=args.log_file,
        sample_period_s=999.0,
        seed=1,
        duration_s=args.duration_s,
        manual_start=False,
        auto_workflow=False,
        sim_status_file=None,
        sim_control_file=None,
        status_timeout_s=0.0,
        tracking_delay_after_elastic_off_s=args.tracking_delay_after_elastic_off_s,
        tracking_start_delay_s=args.tracking_start_delay_s,
        elastic_release_delay_after_policy_start_s=None,
    )
    policy.run()


if __name__ == "__main__":
    main()
