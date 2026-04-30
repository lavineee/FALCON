import argparse
import json
import os
import sys
import time

import numpy as np
import yaml
from termcolor import colored

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.loco_manip.loco_manip_ee_tracking_7dof_with_hand_test import (
    LocoManipEETracking7DofWithHandTestPolicy,
)
from sim2real.rl_policy.loco_manip.loco_manip_ee_tracking_test import (
    apply_named_motor_gain_scales,
)
from sim2real.rl_policy.loco_manip.loco_manip_valve_task_7dof_with_hand import (
    _axis_angle_to_rot,
)


def _normalize_or_none(v, eps=1e-8):
    v = np.asarray(v, dtype=float).reshape(3)
    norm = np.linalg.norm(v)
    if norm < eps:
        return None
    return v / norm


def _base_rpy_deg(geom):
    base_xmat = geom.get("base_xmat_world", None) if geom else None
    if base_xmat is None:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return _rpy_deg_from_xmat(base_xmat)


def _rpy_deg_from_xmat(base_xmat):
    rot = np.asarray(base_xmat, dtype=float).reshape(3, 3)
    sy = np.hypot(rot[0, 0], rot[1, 0])
    if sy < 1e-8:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    else:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def _angle_diff_deg(a, b):
    """Return the shortest signed difference a-b in degrees."""
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _parse_float_sequence(value):
    """Parse a YAML/env sequence like [10, -20, 30] or '10,-20,30'."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "off", "false"):
        return []
    text = text.strip("[]()")
    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    return [float(part) for part in parts if part]


def _csv_token(value):
    """Keep diagnostic strings one-column safe without quoting every CSV row manually."""
    return str(value).replace(",", ";").replace("\n", " ").replace("\r", " ")


class SimStatusGraspGeometryProvider:
    """从 MuJoCo status 读取抓握几何；后续实机可替换成视觉估计输出。"""

    def __init__(self, status_file, timeout_s=2.0):
        self.status_file = status_file
        self.timeout_s = float(timeout_s)

    def read(self):
        if not self.status_file or not os.path.exists(self.status_file):
            return None
        try:
            with open(self.status_file, "r") as file:
                status = json.load(file)
        except Exception:
            return None

        if time.time() - float(status.get("timestamp", 0.0)) > self.timeout_s:
            return None
        required = ("base_pos_world", "base_xmat_world", "valve_center_world", "valve_grasp_world")
        if any(key not in status for key in required):
            return None

        base_pos = np.asarray(status["base_pos_world"], dtype=float).reshape(3)
        base_xmat = np.asarray(status["base_xmat_world"], dtype=float).reshape(3, 3)
        world_to_base = base_xmat.T
        center_world = np.asarray(status["valve_center_world"], dtype=float).reshape(3)
        grasp_world = np.asarray(status["valve_grasp_world"], dtype=float).reshape(3)
        hand_proxy_world = status.get("right_contact_grasp_proxy_world", None)
        if hand_proxy_world is not None:
            hand_proxy_world = np.asarray(hand_proxy_world, dtype=float).reshape(3)
        axis_world = np.asarray(status.get("valve_axis_world", [1.0, 0.0, 0.0]), dtype=float).reshape(3)
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-9)
        axis_base = world_to_base @ axis_world
        axis_base = axis_base / (np.linalg.norm(axis_base) + 1e-9)

        return {
            "timestamp": float(status["timestamp"]),
            "base_pos_world": base_pos,
            "base_xmat_world": base_xmat,
            "world_to_base": world_to_base,
            "center_world": center_world,
            "grasp_world": grasp_world,
            "hand_proxy_world": hand_proxy_world,
            "axis_world": axis_world,
            "wheel_pos_world": np.asarray(status.get("valve_wheel_world", center_world), dtype=float).reshape(3),
            "wheel_xmat_world": np.asarray(status.get("valve_wheel_xmat_world", np.eye(3)), dtype=float).reshape(3, 3),
            "center_base": world_to_base @ (center_world - base_pos),
            "grasp_base": world_to_base @ (grasp_world - base_pos),
            "axis_base": axis_base,
            "valve_angle": float(status.get("valve_angle", 0.0)),
            "valve_vel": float(status.get("valve_vel", 0.0)),
            "valve_angle_hold_enabled": bool(status.get("valve_angle_hold_enabled", False)),
            "valve_angle_hold_target": float(status.get("valve_angle_hold_target", 0.0)),
            "valve_angle_hold_tau": float(status.get("valve_angle_hold_tau", 0.0)),
            "valve_angle_lock_enabled": bool(status.get("valve_angle_lock_enabled", False)),
            "attach_enabled": bool(status.get("attach_enabled", False)),
            "contact_count": int(status.get("right_hand_valve_contact_count", 0)),
            "contact_normal_force": float(status.get("right_hand_valve_contact_normal_force", 0.0)),
            "contact_pairs": status.get("right_hand_valve_contact_pairs", []),
            "real_contact_count": int(
                status.get(
                    "right_hand_valve_real_contact_count",
                    status.get("right_hand_valve_contact_count", 0),
                )
            ),
            "real_contact_normal_force": float(
                status.get(
                    "right_hand_valve_real_contact_normal_force",
                    status.get("right_hand_valve_contact_normal_force", 0.0),
                )
            ),
            "real_contact_pairs": status.get(
                "right_hand_valve_real_contact_pairs",
                status.get("right_hand_valve_contact_pairs", []),
            ),
            "right_hand_state": status.get("right_hand_state", "unknown"),
            "right_hand_q": status.get("right_hand_q", [np.nan] * 7),
            "right_hand_target_q": status.get("right_hand_target_q", [np.nan] * 7),
            "right_hand_torque_cmd": status.get("right_hand_torque_cmd", [np.nan] * 7),
            "right_hand_contact_hold_latched": bool(status.get("right_hand_contact_hold_latched", False)),
            "right_hand_close_progress_thumb": float(status.get("right_hand_close_progress_thumb", np.nan)),
            "right_hand_close_progress_finger": float(status.get("right_hand_close_progress_finger", np.nan)),
            "right_hand_target_close_progress_thumb": float(
                status.get("right_hand_target_close_progress_thumb", np.nan)
            ),
            "right_hand_target_close_progress_finger": float(
                status.get("right_hand_target_close_progress_finger", np.nan)
            ),
            "debug_contact_pairs": status.get("debug_contact_pairs", []),
            "debug_geom_world": status.get("debug_geom_world", {}),
            "left_foot_force": float(status.get("left_foot_force", 0.0)),
            "right_foot_force": float(status.get("right_foot_force", 0.0)),
            "feet_contact_stable": bool(status.get("feet_contact_stable", False)),
        }


class LocoManipValveGraspContact7DofWithHandPolicy(LocoManipEETracking7DofWithHandTestPolicy):
    """独立真实接触抓握 baseline，不复用旧 valve_task 状态机。"""

    WAIT_GEOMETRY = "wait_geometry"
    WAIT_TASK_POSE = "wait_task_pose"
    MOVE_PREGRASP = "move_pregrasp"
    APPROACH_GRASP = "approach_grasp"
    CLOSE_HAND = "close_hand"
    HOLD_GRASP = "hold_grasp"
    PRE_TURN_SETTLE = "pre_turn_settle"
    TURN_VALVE = "turn_valve"
    TURN_HOLD = "turn_hold"
    DONE = "done"
    FAILED = "failed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_provider = SimStatusGraspGeometryProvider(
            self.sim_status_file,
            timeout_s=self.status_timeout_s,
        )
        self.pregrasp_offset_m = float(self.config.get("valve_pregrasp_offset_m", 0.13))
        self.pregrasp_axis = str(self.config.get("valve_pregrasp_axis", "valve_axis")).lower()
        self.pregrasp_error_m = float(self.config.get("valve_pregrasp_error_m", 0.045))
        self.pregrasp_max_s = float(self.config.get("valve_pregrasp_max_s", 3.0))
        self.approach_max_s = float(self.config.get("valve_approach_max_s", 2.5))
        self.grasp_error_m = float(self.config.get("valve_grasp_error_m", 0.024))
        self.close_wait_s = float(self.config.get("valve_close_wait_s", 1.4))
        self.hold_s = float(self.config.get("valve_hold_s", 5.0))
        self.task_start_delay_s = float(self.config.get("valve_task_start_delay_s", 0.5))
        self.stop_after_done_s = float(self.config.get("valve_stop_after_done_s", 2.0))
        self.grasp_target_bias_base = np.asarray(
            self.config.get("valve_grasp_target_bias_base_m", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(3)
        self.grasp_radial_inset_m = float(self.config.get("valve_grasp_radial_inset_m", 0.0))
        self.grasp_point_offset_ee = np.asarray(
            self.config.get("valve_grasp_point_offset_ee_m", [-0.03, 0.06, 0.0]),
            dtype=float,
        ).reshape(3)
        self.align_grasp_orientation = bool(self.config.get("valve_align_grasp_orientation", True))
        self.grasp_forward_axis_sign = float(self.config.get("valve_grasp_forward_axis_sign", -1.0))
        self.grasp_radial_axis = str(self.config.get("valve_grasp_radial_axis", "y")).lower()
        self.grasp_radial_axis_sign = float(self.config.get("valve_grasp_radial_axis_sign", 1.0))
        self.freeze_grasp_world_after_start = bool(self.config.get("valve_freeze_grasp_world_after_start", True))
        self.latch_grasp_frame_on_close = bool(self.config.get("valve_latch_grasp_frame_on_close", True))
        self.hold_target_mode = str(self.config.get("valve_hold_target_mode", "valve_local")).lower()
        self.hold_latch_mode = str(self.config.get("valve_hold_latch_mode", "desired")).lower()
        self.hold_actual_to_desired_blend = float(
            np.clip(self.config.get("valve_hold_actual_to_desired_blend", 0.0), 0.0, 1.0)
        )
        self.close_max_s = float(self.config.get("valve_close_max_s", max(self.close_wait_s, 4.0)))
        self.hold_entry_min_success_s = float(self.config.get("valve_hold_entry_min_success_s", 0.0))
        self.hold_entry_max_abs_valve_vel = float(self.config.get("valve_hold_entry_max_abs_valve_vel", float("inf")))
        self.enter_hold_on_close_timeout = bool(self.config.get("valve_enter_hold_on_close_timeout", True))
        self.success_min_contacts = int(self.config.get("valve_grasp_success_min_contacts", 2))
        self.success_min_force = float(self.config.get("valve_grasp_success_min_force_n", 0.8))
        self.success_require_thumb = bool(self.config.get("valve_grasp_success_require_thumb", True))
        self.success_require_finger = bool(self.config.get("valve_grasp_success_require_finger", True))
        self.success_require_index = bool(self.config.get("valve_grasp_success_require_index", False))
        self.success_require_middle = bool(self.config.get("valve_grasp_success_require_middle", False))
        self.success_require_real_hand_contact = bool(
            self.config.get("valve_grasp_success_require_real_hand_contact", False)
        )
        self.close_require_real_hand_contact = bool(
            self.config.get("valve_close_require_real_hand_contact", self.success_require_real_hand_contact)
        )
        self.require_palm_contact_to_close = bool(self.config.get("valve_require_palm_contact_to_close", True))
        self.palm_close_min_force = float(self.config.get("valve_palm_close_min_force_n", 0.5))
        self.palm_close_max_error = float(self.config.get("valve_palm_close_max_error_m", 0.10))
        self.allow_finger_contact_close = bool(self.config.get("valve_allow_finger_contact_close", False))
        self.finger_close_min_contacts = int(self.config.get("valve_finger_close_min_contacts", 2))
        self.finger_close_min_force = float(self.config.get("valve_finger_close_min_force_n", 4.0))
        self.finger_close_max_error = float(self.config.get("valve_finger_close_max_error_m", 0.10))
        self.finger_close_max_attach_error = float(
            self.config.get("valve_finger_close_max_attach_site_error_m", 0.09)
        )
        self.finger_close_require_index_middle = bool(
            self.config.get("valve_finger_close_require_index_middle", True)
        )
        self.close_on_contact = bool(self.config.get("valve_close_on_contact", True))
        self.contact_close_min_contacts = int(self.config.get("valve_contact_close_min_contacts", 1))
        self.contact_close_min_force = float(self.config.get("valve_contact_close_min_force_n", 0.5))
        self.approach_tracking_compensation_gain = float(
            self.config.get("valve_approach_tracking_compensation_gain", 0.0)
        )
        self.approach_tracking_compensation_max_m = abs(
            float(self.config.get("valve_approach_tracking_compensation_max_m", 0.0))
        )
        # 接近阶段的 wrist/fake-EE 误差不一定等于真实掌垫误差。
        # 这里额外用 MuJoCo palm proxy site 闭环，把掌心内壁推到阀门接触面。
        self.approach_proxy_compensation_gain = float(
            self.config.get("valve_approach_proxy_compensation_gain", 0.0)
        )
        self.approach_proxy_compensation_max_m = abs(
            float(self.config.get("valve_approach_proxy_compensation_max_m", 0.0))
        )
        self.contact_tracking_compensation_gain = float(
            self.config.get("valve_contact_tracking_compensation_gain", 0.0)
        )
        self.contact_tracking_compensation_max_m = abs(
            float(self.config.get("valve_contact_tracking_compensation_max_m", 0.0))
        )
        self.target_rate_limit_mps = abs(float(self.config.get("valve_target_rate_limit_mps", 0.0)))

        self.enable_turn = bool(self.config.get("valve_enable_turn", False))
        self.turn_target_deg = float(self.config.get("valve_turn_target_deg", 0.0))
        self.turn_target_sequence_deg = _parse_float_sequence(
            self.config.get("valve_turn_target_sequence_deg", None)
        )
        self._turn_sequence_index = 0
        if self.turn_target_sequence_deg:
            # 序列按“分段增量”解释：10,-20,30 表示三段相对转角命令。
            self.turn_target_deg = float(self.turn_target_sequence_deg[0])
        self.turn_speed_deg = abs(float(self.config.get("valve_turn_speed_deg", 2.0)))
        if self.turn_speed_deg <= 1e-6:
            self.turn_speed_deg = 2.0
        self.turn_control_mode = str(self.config.get("valve_turn_control_mode", "closed_loop")).strip().lower()
        if self.turn_control_mode not in ("timed", "closed_loop"):
            self.logger.warning(
                colored(
                    f"[VALVE_GRASP] unsupported valve_turn_control_mode={self.turn_control_mode}, "
                    "using closed_loop",
                    "yellow",
                )
            )
            self.turn_control_mode = "closed_loop"
        self.turn_feedback_gain = float(self.config.get("valve_turn_feedback_gain", 0.6))
        self.turn_command_lead_deg = abs(float(self.config.get("valve_turn_command_lead_deg", 10.0)))
        default_turn_command_speed = max(1.8 * self.turn_speed_deg, self.turn_speed_deg + 2.0)
        self.turn_command_speed_deg = abs(
            float(self.config.get("valve_turn_command_speed_deg", default_turn_command_speed))
        )
        if self.turn_command_speed_deg <= 1e-6:
            self.turn_command_speed_deg = default_turn_command_speed
        self.turn_extra_settle_s = float(self.config.get("valve_turn_extra_settle_s", 2.0))
        self.turn_tolerance_deg = abs(float(self.config.get("valve_turn_tolerance_deg", 3.0)))
        self.turn_finish_on_target_reached = bool(
            self.config.get("valve_turn_finish_on_target_reached", False)
        )
        self.turn_min_s = float(self.config.get("valve_turn_min_s", 0.5))
        self.turn_hold_s = float(self.config.get("valve_turn_hold_s", self.hold_s))
        self.turn_hold_command_mode = str(
            self.config.get("valve_turn_hold_command_mode", "current")
        ).strip().lower()
        if self.turn_hold_command_mode not in ("current", "target", "actual"):
            self.logger.warning(
                colored(
                    "[VALVE_GRASP] unsupported valve_turn_hold_command_mode="
                    f"{self.turn_hold_command_mode}, using current",
                    "yellow",
                )
            )
            self.turn_hold_command_mode = "current"
        self.turn_segment_settle_enabled = bool(
            self.config.get("valve_turn_segment_settle_enabled", False)
        )
        self.turn_segment_settle_error_deg = abs(
            float(self.config.get("valve_turn_segment_settle_error_deg", 1.5))
        )
        self.turn_segment_settle_vel_deg_s = abs(
            float(self.config.get("valve_turn_segment_settle_vel_deg_s", 1.0))
        )
        self.turn_segment_settle_s = max(
            0.0,
            float(self.config.get("valve_turn_segment_settle_s", 0.8)),
        )
        default_segment_settle_max_s = max(self.turn_hold_s + 2.0, self.turn_hold_s)
        self.turn_segment_settle_max_s = max(
            self.turn_hold_s,
            float(self.config.get("valve_turn_segment_settle_max_s", default_segment_settle_max_s)),
        )
        self.turn_completion_require_grasp = bool(
            self.config.get("valve_turn_completion_require_grasp", self.turn_segment_settle_enabled)
        )
        self.turn_completion_min_contact_health_ratio = float(
            self.config.get("valve_turn_completion_min_contact_health_ratio", 0.5)
        )
        self.turn_completion_max_slip_m = float(
            self.config.get("valve_turn_completion_max_slip_m", 0.12)
        )
        self.turn_completion_max_grasp_error_m = float(
            self.config.get("valve_turn_completion_max_grasp_error_m", float("inf"))
        )
        self.turn_completion_timeout_is_failure = bool(
            self.config.get("valve_turn_completion_timeout_is_failure", self.turn_segment_settle_enabled)
        )
        self.turn_entry_min_hold_s = float(self.config.get("valve_turn_entry_min_hold_s", 0.4))
        self.turn_entry_max_slip_m = float(self.config.get("valve_turn_entry_max_slip_m", 0.045))
        self.turn_entry_require_success = bool(self.config.get("valve_turn_entry_require_success", True))
        self.turn_entry_max_attach_site_error_m = float(
            self.config.get("valve_turn_entry_max_attach_site_error_m", float("inf"))
        )
        self.turn_entry_success_memory_s = float(
            self.config.get("valve_turn_entry_success_memory_s", 0.0)
        )
        self.turn_entry_min_contact_count = int(
            self.config.get("valve_turn_entry_min_contact_count", self.success_min_contacts)
        )
        self.turn_entry_min_contact_force = float(
            self.config.get("valve_turn_entry_min_contact_force_n", self.success_min_force)
        )
        self.turn_entry_max_contact_force = float(
            self.config.get("valve_turn_entry_max_contact_force_n", float("inf"))
        )
        self.turn_start_on_hold_entry = bool(self.config.get("valve_turn_start_on_hold_entry", False))
        self.turn_pre_settle_s = float(self.config.get("valve_turn_pre_settle_s", 0.0))
        self.turn_pre_settle_max_s = float(
            self.config.get("valve_turn_pre_settle_max_s", max(self.turn_pre_settle_s + 0.5, 1.0))
        )
        self.turn_pre_settle_max_abs_valve_vel = float(
            self.config.get("valve_turn_pre_settle_max_abs_valve_vel", 1.0)
        )
        self.turn_abort_slip_m = float(self.config.get("valve_turn_abort_slip_m", 0.20))
        self.turn_abort_contact_force_n = float(self.config.get("valve_turn_abort_contact_force_n", 600.0))
        self.turn_abort_base_tilt_deg = float(self.config.get("valve_turn_abort_base_tilt_deg", 25.0))
        self.turn_abort_valve_vel_rad_s = float(self.config.get("valve_turn_abort_valve_vel_rad_s", 0.0))
        self.turn_abort_min_base_to_valve_x_m = float(
            self.config.get("valve_turn_abort_min_base_to_valve_x_m", 0.0)
        )
        self.turn_abort_max_base_approach_m = float(
            self.config.get("valve_turn_abort_max_base_approach_m", 0.0)
        )
        self.turn_abort_max_base_yaw_drift_deg = float(
            self.config.get("valve_turn_abort_max_base_yaw_drift_deg", 0.0)
        )
        self.turn_abort_require_feet_contact_stable = bool(
            self.config.get("valve_turn_abort_require_feet_contact_stable", False)
        )
        self.turn_abort_min_total_foot_force_n = float(
            self.config.get("valve_turn_abort_min_total_foot_force_n", 0.0)
        )
        self.turn_radius_mode = str(self.config.get("valve_turn_radius_mode", "actual")).strip().lower()
        if self.turn_radius_mode not in ("actual", "target", "command", "command_delta_scaled"):
            self.turn_radius_mode = "actual"
        self.turn_displacement_radius_gain = float(
            self.config.get("valve_turn_displacement_radius_gain", 1.0)
        )
        self.turn_displacement_radius_m = float(
            self.config.get("valve_turn_displacement_radius_m", 0.0)
        )
        self.turn_rotate_grasp_orientation = bool(self.config.get("valve_turn_rotate_grasp_orientation", True))
        self.turn_trajectory_sign = 1.0 if float(self.config.get("valve_turn_trajectory_sign", 1.0)) >= 0.0 else -1.0
        self.turn_contact_compensation_enabled = bool(
            self.config.get("valve_turn_contact_compensation_enabled", True)
        )
        self.preflight_enabled = bool(self.config.get("valve_preflight_enabled", False))
        self.preflight_min_wait_s = float(self.config.get("valve_preflight_min_wait_s", 0.0))
        self.preflight_required_stable_s = float(
            self.config.get("valve_preflight_required_stable_s", 0.0)
        )
        self.preflight_max_wait_s = float(self.config.get("valve_preflight_max_wait_s", 0.0))
        self.preflight_max_base_tilt_deg = float(
            self.config.get("valve_preflight_max_base_tilt_deg", 8.0)
        )
        self.preflight_base_z_range = tuple(
            float(v) for v in self.config.get("valve_preflight_base_z_range_m", [0.65, 0.95])
        )
        self.preflight_center_dx_range = tuple(
            float(v) for v in self.config.get("valve_preflight_center_dx_range_m", [0.25, 0.65])
        )
        self.preflight_center_abs_y_max = float(
            self.config.get("valve_preflight_center_abs_y_max_m", 0.20)
        )
        self.preflight_min_total_foot_force = float(
            self.config.get("valve_preflight_min_total_foot_force_n", 0.0)
        )
        self.task_pose_gate_enabled = bool(
            self.config.get("valve_task_pose_gate_enabled", self.preflight_enabled)
        )
        self.task_pose_gate_required_stable_s = float(
            self.config.get("valve_task_pose_gate_required_stable_s", 0.0)
        )
        self.task_pose_gate_max_wait_s = float(
            self.config.get("valve_task_pose_gate_max_wait_s", 0.0)
        )
        self.task_pose_gate_max_base_tilt_deg = float(
            self.config.get("valve_task_pose_gate_max_base_tilt_deg", self.preflight_max_base_tilt_deg)
        )
        self.task_pose_gate_base_z_range = tuple(
            float(v)
            for v in self.config.get(
                "valve_task_pose_gate_base_z_range_m",
                list(self.preflight_base_z_range),
            )
        )
        self.task_pose_gate_center_dx_range = tuple(
            float(v)
            for v in self.config.get(
                "valve_task_pose_gate_center_dx_range_m",
                list(self.preflight_center_dx_range),
            )
        )
        self.task_pose_gate_center_abs_y_max = float(
            self.config.get(
                "valve_task_pose_gate_center_abs_y_max_m",
                self.preflight_center_abs_y_max,
            )
        )
        yaw_range = self.config.get("valve_task_pose_gate_yaw_range_deg", None)
        self.task_pose_gate_yaw_range = (
            None if yaw_range is None else tuple(float(v) for v in yaw_range)
        )
        self.pre_turn_angle_lock_enabled = bool(
            self.config.get("valve_pre_turn_angle_lock_enabled", self.enable_turn)
        )
        self.pre_turn_angle_hold_enabled = bool(self.config.get("valve_pre_turn_angle_hold_enabled", False))
        self.contact_assist_enabled = bool(self.config.get("valve_contact_assist_enabled", False))
        self.contact_assist_release_on_done = bool(
            self.config.get("valve_contact_assist_release_on_done", False)
        )
        self.contact_assist_release_on_turn_hold = bool(
            self.config.get("valve_contact_assist_release_on_turn_hold", False)
        )
        self.turn_hold_angle_brake_enabled = bool(
            self.config.get("valve_turn_hold_angle_brake_enabled", False)
        )
        self.contact_assist_lock_settle_s = float(
            self.config.get("valve_contact_assist_lock_settle_s", 0.0)
        )
        self.contact_assist_max_site_error_m = float(
            self.config.get("valve_contact_assist_max_site_error_m", float("inf"))
        )
        self.grasp_tip_offsets_ee = {
            "thumb": np.asarray(
                self.config.get("valve_grasp_thumb_tip_offset_ee_m", [-0.089, 0.090, 0.005]),
                dtype=float,
            ).reshape(3),
            "index": np.asarray(
                self.config.get("valve_grasp_index_tip_offset_ee_m", [-0.052, 0.070, 0.029]),
                dtype=float,
            ).reshape(3),
            "middle": np.asarray(
                self.config.get("valve_grasp_middle_tip_offset_ee_m", [-0.052, 0.070, -0.029]),
                dtype=float,
            ).reshape(3),
        }

        self.task_state = self.WAIT_GEOMETRY
        self._state_enter_t = time.perf_counter()
        root, ext = os.path.splitext(self.log_file)
        self.summary_file = f"{root}_summary.jsonl" if root else f"{self.log_file}_summary.jsonl"
        self._task_ready_t = None
        self._latest_geom = None
        self._current_target_base = np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z], dtype=float)
        self._close_command_sent = False
        self._success_reported = False
        self._latest_raw_geom = None
        self._frozen_world_geom = None
        self._close_latched_world_geom = None
        self._close_latched_grasp_local = None
        self._hold_grasp_local = None
        self._close_success_started_t = None
        self._last_grasp_success_t = None
        self._turn_started = False
        self._turn_start_angle = 0.0
        self._turn_nominal_duration_s = 0.0
        self._turn_duration_s = 0.0
        self._turn_center_world = None
        self._turn_axis_world = None
        self._turn_r0_world = None
        self._turn_start_point_world = None
        self._turn_displacement_r_world = None
        self._turn_R0_world = None
        self._turn_reference_delta_deg = 0.0
        self._turn_command_delta_deg = 0.0
        self._turn_hold_command_delta_deg = None
        self._turn_hold_settle_ok_started_t = None
        self._turn_hold_quality_reason = ""
        self._turn_final_error_deg = self.turn_target_deg
        self._turn_sequence_zero_angle = None
        self._turn_cmd_deg = 0.0
        self._last_turn_update_t = None
        self._turn_start_base_to_valve_x_m = None
        self._turn_start_base_yaw_deg = None
        self._turn_rows = []
        self._turn_segment_summaries = []
        self._overall_summary_written = False
        self._failure_reason = ""
        self._abort_reason = ""
        self._current_command_grasp_base = None
        self._slip_grasp_local = None
        self._slip_palm_R_local = None
        self._contact_assist_active = False
        self._contact_assist_released = False
        self._contact_assist_blocked_reported = False
        self._pre_turn_lock_released = False
        self._pre_turn_unlock_t = None
        self._preflight_ok_started_t = None
        self._task_pose_gate_enter_t = None
        self._task_pose_gate_ok_started_t = None
        self._next_task_pose_log_t = 0.0
        self._rate_limited_target_base = None
        self._last_target_update_t = None

        self._write_attach_state(False)
        self._write_hand_state("open")
        self._reset_grasp_log()

    def _reset_grasp_log(self):
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as file:
            file.write(
                "time_s,state,target_x,target_y,target_z,current_x,current_y,current_z,"
                "ee_error_m,ik_error_m,servo_error_m,grasp_error_m,"
                "valve_angle,valve_vel,contact_count,contact_normal_force,"
                "contact_topology,real_contact_count,real_contact_normal_force,"
                "real_contact_topology,grasp_success,attach_enabled,relative_slip_m,"
                "close_slip_m,relative_slip_axial_m,relative_slip_radial_m,"
                "relative_slip_tangent_m,relative_orientation_slip_deg,"
                "raw_target_drift_m,attach_site_error_m,"
                "attach_site_error_world_x,attach_site_error_world_y,attach_site_error_world_z,"
                "attach_site_error_base_x,attach_site_error_base_y,attach_site_error_base_z,"
                "valve_angle_hold_enabled,valve_angle_lock_enabled,"
                "valve_angle_hold_tau,"
                "turn_target_delta_deg,turn_reference_delta_deg,turn_command_delta_deg,"
                "turn_actual_delta_deg,turn_angle_error_deg,turn_final_error_deg,turn_motion_radius_m,"
                "base_roll_deg,base_pitch_deg,base_yaw_deg,"
                "base_to_valve_x,base_to_valve_y,base_to_valve_z,"
                "left_foot_force,right_foot_force,"
                "feet_contact_stable,action_norm,"
                "right_hand_state,right_thumb_q0,right_thumb_q1,right_thumb_q2,"
                "right_thumb_target_q0,right_thumb_target_q1,right_thumb_target_q2,"
                "right_thumb_tau0,right_thumb_tau1,right_thumb_tau2,"
                "right_hand_hold_latched,right_thumb_close_progress,right_finger_close_progress,"
                "right_thumb_target_close_progress,right_finger_target_close_progress,"
                "contact_pairs,real_contact_pairs,debug_contact_pairs,"
                "task_state,segment_id,segment_count,target_delta_deg,target_cumulative_deg,"
                "valve_angle_deg,valve_target_angle_deg,valve_angle_error_deg,valve_vel_deg_s,"
                "right_eef_pos_error_m,right_eef_orientation_error_deg,"
                "hand_valve_contact_count,finger1_valve_contact_count,"
                "finger2_valve_contact_count,finger3_valve_contact_count,"
                "contact_health_score,contact_health_ratio,"
                "soft_connect_active,soft_connect_active_ratio,"
                "contact_force_n,contact_force_peak_n,"
                "angle_brake_enabled,angle_brake_torque,angle_brake_peak_torque,"
                "base_approach_m,base_yaw_drift_deg,base_tilt_deg,max_base_tilt_deg,"
                "segment_result,failure_reason,abort_reason\n"
            )
        with open(self.summary_file, "w") as file:
            file.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "log_file": self.log_file,
                        "sequence_target_deg": (
                            list(self.turn_target_sequence_deg)
                            if self.turn_target_sequence_deg
                            else [float(self.turn_target_deg)]
                        ),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    def _enter_state(self, state):
        if self.task_state == state:
            return
        self.task_state = state
        self._state_enter_t = time.perf_counter()
        if state == self.FAILED:
            if not self._failure_reason:
                self._failure_reason = self._abort_reason or "entered_failed_state"
            self._summarize_turn(result="failed", failure_reason=self._failure_reason)
            self._write_overall_summary(overall_result="failed")
        if state == self.CLOSE_HAND:
            self._close_success_started_t = None
        if state == self.TURN_HOLD:
            # 到目标角附近后应停止追加圆弧运动；实机上对应“保持当前手端位置”。
            if self._turn_hold_command_delta_deg is None:
                self._turn_hold_command_delta_deg = float(self._turn_command_delta_deg)
            if self.contact_assist_release_on_turn_hold and self._contact_assist_active:
                # 软连接只作为抓住后传力近似；进入目标保持阶段后释放，避免继续把阀门推过头。
                self._set_contact_assist(False)
            if self.turn_hold_angle_brake_enabled and self._turn_started:
                # MuJoCo 里抓握/摩擦不足时，目标附近需要一个有限力矩制动项吸收阀门残余角速度。
                hold_target = self._turn_start_angle + float(np.deg2rad(self.turn_target_deg))
                self._write_valve_angle_hold(True, hold_target)
        self.logger.info(colored(f"[VALVE_GRASP] state -> {state}", "cyan"))
        if state == self.DONE:
            self._write_overall_summary(overall_result="done")

    def _set_failure_reason(self, reason, abort=False):
        reason = _csv_token(reason or "unknown")
        if abort:
            self._abort_reason = reason
        if not self._failure_reason:
            self._failure_reason = reason

    def _write_summary_record(self, record):
        summary_dir = os.path.dirname(self.summary_file)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(self.summary_file, "a") as file:
            file.write(json.dumps(record, ensure_ascii=True, allow_nan=True) + "\n")

    def _write_attach_state(self, enabled):
        self._write_control_file(
            {
                "weld_enabled": bool(enabled),
                "attach_enabled": bool(enabled),
                "weld_timestamp": time.time(),
                "timestamp": time.time(),
            }
        )

    def _set_contact_assist(self, enabled):
        """启用柔顺 connect 抓握辅助；它只在真实接触判据通过后参与保持抓点位置。"""
        enabled = bool(enabled)
        if enabled and not self.contact_assist_enabled:
            return
        if self._contact_assist_active == enabled:
            return
        self._write_attach_state(enabled)
        self._contact_assist_active = enabled
        if enabled:
            self._contact_assist_released = False
            self.logger.info(colored("[VALVE_GRASP] compliant contact assist ON", "cyan"))
        else:
            self._contact_assist_released = True
            self.logger.info(colored("[VALVE_GRASP] compliant contact assist OFF", "cyan"))

    def _maybe_enable_contact_assist(self, geom):
        """只有真实 connect site 已经靠近阀门抓点时，才允许打开辅助约束。"""
        if not self.contact_assist_enabled or self._contact_assist_active:
            return False
        site_error = self._attach_site_error_m(geom)
        if np.isfinite(site_error) and site_error > self.contact_assist_max_site_error_m:
            if not self._contact_assist_blocked_reported:
                self._contact_assist_blocked_reported = True
                self.logger.warning(
                    colored(
                        "[VALVE_GRASP] contact assist skipped: "
                        f"site_error={site_error:.4f}m > {self.contact_assist_max_site_error_m:.4f}m",
                        "yellow",
                    )
                )
            return False
        self._set_contact_assist(True)
        return self._contact_assist_active

    def _write_valve_angle_lock(self, enabled, target=None):
        self._write_valve_angle_control(bool(enabled), False, target)

    def _write_valve_angle_hold(self, enabled, target=None):
        self._write_valve_angle_control(False, bool(enabled), target)

    def _write_valve_angle_control(self, lock_enabled, hold_enabled, target=None):
        payload = {
            "valve_angle_lock_enabled": bool(lock_enabled),
            "valve_angle_hold_enabled": bool(hold_enabled),
            "valve_angle_control_timestamp": time.time(),
            "timestamp": time.time(),
        }
        if target is not None:
            payload["valve_angle_hold_target"] = float(target)
        self._write_control_file(payload)

    def _task_pose_status(self, geom):
        if geom is None:
            return False, "no grasp geometry", {}
        base_pos = np.asarray(geom.get("base_pos_world", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        center = np.asarray(geom.get("center_world", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        base_to_valve = center - base_pos
        roll, pitch, yaw = _base_rpy_deg(geom)
        base_tilt = max(abs(float(roll)), abs(float(pitch)))

        z_min, z_max = self.task_pose_gate_base_z_range
        dx_min, dx_max = self.task_pose_gate_center_dx_range
        checks = [
            (np.all(np.isfinite(base_pos)), "base pose not finite"),
            (base_tilt <= self.task_pose_gate_max_base_tilt_deg, f"base tilt {base_tilt:.1f}deg too large"),
            (z_min <= base_pos[2] <= z_max, f"base z {base_pos[2]:.3f}m out of range"),
            (dx_min <= base_to_valve[0] <= dx_max, f"valve dx {base_to_valve[0]:.3f}m out of range"),
            (
                abs(base_to_valve[1]) <= self.task_pose_gate_center_abs_y_max,
                f"valve dy {base_to_valve[1]:.3f}m too large",
            ),
        ]
        if self.task_pose_gate_yaw_range is not None:
            yaw_min, yaw_max = self.task_pose_gate_yaw_range
            checks.append((yaw_min <= yaw <= yaw_max, f"base yaw {yaw:.1f}deg out of range"))

        metrics = {
            "tilt": base_tilt,
            "yaw": float(yaw),
            "base_z": float(base_pos[2]),
            "dx": float(base_to_valve[0]),
            "dy": float(base_to_valve[1]),
        }
        for ok, reason in checks:
            if not ok:
                return False, reason, metrics
        return True, "ok", metrics

    def _task_pose_ready(self, geom):
        """policy 启动并松绳后，再确认一次任务站位，避免歪站姿污染抓握评测。"""
        if not self.task_pose_gate_enabled:
            return True

        now = time.perf_counter()
        if self._task_pose_gate_enter_t is None:
            self._task_pose_gate_enter_t = now

        ok, reason, metrics = self._task_pose_status(geom)
        if ok:
            if self._task_pose_gate_ok_started_t is None:
                self._task_pose_gate_ok_started_t = now
                self.logger.info(colored("[VALVE_GRASP] task pose candidate is valid", "cyan"))
            if now - self._task_pose_gate_ok_started_t >= self.task_pose_gate_required_stable_s:
                self.logger.info(
                    colored(
                        "[VALVE_GRASP] task pose passed: "
                        f"tilt={metrics['tilt']:.1f}deg, yaw={metrics['yaw']:.1f}deg, "
                        f"dx={metrics['dx']:.3f}m, dy={metrics['dy']:.3f}m, "
                        f"base_z={metrics['base_z']:.3f}m",
                        "green",
                    )
                )
                return True
        else:
            self._task_pose_gate_ok_started_t = None

        elapsed = now - self._task_pose_gate_enter_t
        if now >= self._next_task_pose_log_t:
            self._next_task_pose_log_t = now + 1.0
            metric_text = ""
            if metrics:
                metric_text = (
                    f" tilt={metrics['tilt']:.1f}deg yaw={metrics['yaw']:.1f}deg"
                    f" dx={metrics['dx']:.3f}m dy={metrics['dy']:.3f}m"
                    f" z={metrics['base_z']:.3f}m"
                )
            self.logger.info(colored(f"[VALVE_GRASP] waiting task pose: {reason}{metric_text}", "yellow"))

        if self.task_pose_gate_max_wait_s > 0.0 and elapsed >= self.task_pose_gate_max_wait_s:
            self.logger.warning(colored(f"[VALVE_GRASP] task pose gate failed: {reason}", "yellow"))
            self._enter_state(self.FAILED)
        return False

    def _approach_axis_base(self, geom):
        axis = _normalize_or_none(geom["axis_base"])
        if axis is None:
            return np.array([-1.0, 0.0, 0.0], dtype=float)
        toward_robot = -np.asarray(geom["center_base"], dtype=float)
        if np.dot(axis, toward_robot) < 0.0:
            axis = -axis
        return axis

    def _valve_grasp_point_base(self, geom):
        grasp = np.asarray(geom["grasp_base"], dtype=float)
        if bool(geom.get("grasp_base_is_effective", False)):
            return grasp + self.grasp_target_bias_base
        center = np.asarray(geom["center_base"], dtype=float)
        axis = _normalize_or_none(geom["axis_base"])
        radial = grasp - center
        if axis is not None:
            radial = radial - np.dot(radial, axis) * axis
        radial_axis = _normalize_or_none(radial)
        if radial_axis is not None and self.grasp_radial_inset_m != 0.0:
            grasp = grasp - self.grasp_radial_inset_m * radial_axis
        return grasp + self.grasp_target_bias_base

    def _ee_grasp_target_base(self, geom):
        # 阀门轮圈中心应落在闭合手指形成的夹持区域，而不是直接落在 palm site。
        grasp_point = self._valve_grasp_point_base(geom)
        R = self._right_grasp_rotation_base(geom)
        return grasp_point - R @ self.grasp_point_offset_ee

    def _pregrasp_target_base(self, geom):
        return self._ee_grasp_target_base(geom) + self._pregrasp_retreat_axis_base(geom) * self.pregrasp_offset_m

    def _pregrasp_retreat_axis_base(self, geom):
        """预接近点相对最终抓点的退让方向。

        valve_axis 是旧逻辑：从阀门正前方插入，容易让开放指尖先碰轮缘。
        palm_normal 用掌心法向退让，最终接近时掌心内侧先贴住轮缘，更符合真实抓握动作。
        """
        if self.pregrasp_axis in ("palm_normal", "palm", "palm_y"):
            R = self._right_grasp_rotation_base(geom)
            return -R[:, 1]
        return self._approach_axis_base(geom)

    def _control_geom(self, raw_geom):
        if self._close_latched_world_geom is not None:
            return self._close_latched_geom(raw_geom)

        if not self.freeze_grasp_world_after_start:
            return raw_geom

        if self._frozen_world_geom is None:
            self._frozen_world_geom = {
                "center_world": np.asarray(raw_geom["center_world"], dtype=float).copy(),
                "grasp_world": np.asarray(raw_geom["grasp_world"], dtype=float).copy(),
                "axis_world": np.asarray(raw_geom["axis_world"], dtype=float).copy(),
            }

        # 固定同一个轮圈抓取点，避免轻微碰撞把阀门带转后目标跟着跑。
        geom = dict(raw_geom)
        world_to_base = np.asarray(raw_geom["world_to_base"], dtype=float).reshape(3, 3)
        base_pos = np.asarray(raw_geom["base_pos_world"], dtype=float).reshape(3)
        center_world = self._frozen_world_geom["center_world"]
        grasp_world = self._frozen_world_geom["grasp_world"]
        axis_world = self._frozen_world_geom["axis_world"]
        axis_base = world_to_base @ axis_world
        axis_base = axis_base / (np.linalg.norm(axis_base) + 1e-9)
        geom["center_base"] = world_to_base @ (center_world - base_pos)
        geom["grasp_base"] = world_to_base @ (grasp_world - base_pos)
        geom["axis_base"] = axis_base
        return geom

    def _close_latched_geom(self, raw_geom):
        """闭手后固定同一个抓取框架，避免接触扰动导致目标点跟着阀门 site 漂移。"""
        latched = self._close_latched_world_geom
        geom = dict(raw_geom)
        world_to_base = np.asarray(raw_geom["world_to_base"], dtype=float).reshape(3, 3)
        base_pos = np.asarray(raw_geom["base_pos_world"], dtype=float).reshape(3)
        center_world = np.asarray(latched["center_world"], dtype=float).reshape(3)
        grasp_world = np.asarray(latched["grasp_world"], dtype=float).reshape(3)
        axis_world = np.asarray(latched["axis_world"], dtype=float).reshape(3)
        R_world = np.asarray(latched["grasp_R_world"], dtype=float).reshape(3, 3)

        axis_base = world_to_base @ axis_world
        axis_base = axis_base / (np.linalg.norm(axis_base) + 1e-9)
        geom["center_base"] = world_to_base @ (center_world - base_pos)
        geom["grasp_base"] = world_to_base @ (grasp_world - base_pos)
        # grasp_world 已经是应用过 radial_inset 的控制抓点，后续不能再次内缩。
        geom["grasp_base_is_effective"] = True
        geom["axis_base"] = axis_base
        geom["grasp_R_base"] = world_to_base @ R_world
        geom["grasp_frame_latched"] = True
        return geom

    def _point_base_to_world(self, geom, point_base):
        base_pos = np.asarray(geom["base_pos_world"], dtype=float).reshape(3)
        base_to_world = np.asarray(geom["world_to_base"], dtype=float).reshape(3, 3).T
        return base_pos + base_to_world @ np.asarray(point_base, dtype=float).reshape(3)

    def _point_world_to_base(self, geom, point_world):
        base_pos = np.asarray(geom["base_pos_world"], dtype=float).reshape(3)
        world_to_base = np.asarray(geom["world_to_base"], dtype=float).reshape(3, 3)
        return world_to_base @ (np.asarray(point_world, dtype=float).reshape(3) - base_pos)

    def _world_to_valve_local(self, geom, point_world):
        wheel_pos = np.asarray(geom.get("wheel_pos_world", geom["center_world"]), dtype=float).reshape(3)
        wheel_R = np.asarray(geom.get("wheel_xmat_world", np.eye(3)), dtype=float).reshape(3, 3)
        return wheel_R.T @ (np.asarray(point_world, dtype=float).reshape(3) - wheel_pos)

    def _rotation_world_to_valve_local(self, geom, R_world):
        wheel_R = np.asarray(geom.get("wheel_xmat_world", np.eye(3)), dtype=float).reshape(3, 3)
        return wheel_R.T @ np.asarray(R_world, dtype=float).reshape(3, 3)

    def _palm_rotation_valve_local(self, geom):
        debug_geom_world = geom.get("debug_geom_world", {}) if geom else {}
        palm_pose = debug_geom_world.get("right_grasp_proxy_palm_pad", None)
        if palm_pose is None or "xmat" not in palm_pose:
            return None
        return self._rotation_world_to_valve_local(geom, palm_pose["xmat"])

    def _valve_local_to_base(self, geom, point_local):
        wheel_pos = np.asarray(geom.get("wheel_pos_world", geom["center_world"]), dtype=float).reshape(3)
        wheel_R = np.asarray(geom.get("wheel_xmat_world", np.eye(3)), dtype=float).reshape(3, 3)
        point_world = wheel_pos + wheel_R @ np.asarray(point_local, dtype=float).reshape(3)
        return self._point_world_to_base(geom, point_world)

    def _current_grasp_proxy_base(self, geom, robot_state_data):
        # 带手模型里优先使用 MuJoCo 真实 site；没有该字段时才退回 wrist+offset 的估计。
        hand_proxy_world = geom.get("hand_proxy_world", None) if geom else None
        if hand_proxy_world is not None:
            return self._point_world_to_base(geom, hand_proxy_world)
        current_ee = self._current_right_fake_ee_base(robot_state_data)
        return current_ee + self._right_grasp_rotation_base(geom) @ self.grasp_point_offset_ee

    def _latch_hold_grasp_local(self, geom, robot_state_data):
        if self.hold_target_mode != "valve_local":
            return
        # desired: 锁定预设轮缘点；actual: 锁定闭合稳定后的实际掌心代理点。
        # 真接触抓握优先用 actual，避免 HOLD_GRASP 切换瞬间把手从已夹住的位置拉开。
        actual_base = self._current_grasp_proxy_base(geom, robot_state_data)
        actual_world = self._point_base_to_world(geom, actual_base)
        # 控制目标可以使用预设抓点，但滑移评估必须以“已经真实夹住的位置”为零点；
        # 否则预设 site 与实际接触包络之间的固定偏差会被误判成滑移。
        self._slip_grasp_local = self._world_to_valve_local(geom, actual_world)
        self._slip_palm_R_local = self._palm_rotation_valve_local(geom)
        desired_base = self._valve_grasp_point_base(geom)
        if self.hold_latch_mode == "actual":
            # 纯 desired 会把已经夹住的点拖回预设 site，纯 actual 又可能丢掉掌心贴合。
            # 这里允许在二者之间做小比例混合，优先保持真实接触，同时轻微回拉到预设掌心抓点。
            grasp_point_base = actual_base + self.hold_actual_to_desired_blend * (desired_base - actual_base)
        else:
            grasp_point_base = desired_base
        grasp_point_world = self._point_base_to_world(geom, grasp_point_base)
        self._hold_grasp_local = self._world_to_valve_local(geom, grasp_point_world)
        self.logger.info(
            colored(
                f"[VALVE_GRASP] hold target latched in valve-local frame ({self.hold_latch_mode})",
                "cyan",
            )
        )

    def _hold_tracking_geom(self, geom):
        if self.hold_target_mode != "valve_local" or self._hold_grasp_local is None:
            return geom
        hold_geom = dict(geom)
        hold_geom["grasp_base"] = self._valve_local_to_base(geom, self._hold_grasp_local)
        hold_geom["grasp_base_is_effective"] = True
        return hold_geom

    def _relative_grasp_slip_m(self, geom, robot_state_data):
        slip_reference = self._slip_grasp_local if self._slip_grasp_local is not None else self._hold_grasp_local
        if slip_reference is None:
            return np.nan
        grasp_proxy_world = self._point_base_to_world(
            geom,
            self._current_grasp_proxy_base(geom, robot_state_data),
        )
        current_local = self._world_to_valve_local(geom, grasp_proxy_world)
        return float(np.linalg.norm(current_local - slip_reference))

    def _relative_grasp_slip_components_m(self, geom, robot_state_data):
        """把抓点滑移拆到阀门轴向、轮缘径向和轮缘切向，定位主要滑移方向。"""
        slip_reference = self._slip_grasp_local if self._slip_grasp_local is not None else self._hold_grasp_local
        if slip_reference is None:
            return np.full(3, np.nan, dtype=float)

        grasp_proxy_world = self._point_base_to_world(
            geom,
            self._current_grasp_proxy_base(geom, robot_state_data),
        )
        current_local = self._world_to_valve_local(geom, grasp_proxy_world)
        delta = current_local - slip_reference

        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        radial = np.asarray(slip_reference, dtype=float).reshape(3).copy()
        radial[0] = 0.0
        radial = _normalize_or_none(radial)
        if radial is None:
            radial = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent = _normalize_or_none(np.cross(axis, radial))
        if tangent is None:
            tangent = np.array([0.0, 0.0, 1.0], dtype=float)
        return np.array(
            [
                float(np.dot(delta, axis)),
                float(np.dot(delta, radial)),
                float(np.dot(delta, tangent)),
            ],
            dtype=float,
        )

    def _relative_grasp_orientation_slip_deg(self, geom):
        if self._slip_palm_R_local is None:
            return np.nan
        current_R_local = self._palm_rotation_valve_local(geom)
        if current_R_local is None:
            return np.nan
        R_err = self._slip_palm_R_local.T @ current_R_local
        cos_angle = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.rad2deg(np.arccos(cos_angle)))

    def _close_relative_slip_m(self, geom, robot_state_data):
        if self._close_latched_grasp_local is None:
            return np.nan
        grasp_proxy_world = self._point_base_to_world(
            geom,
            self._current_grasp_proxy_base(geom, robot_state_data),
        )
        current_local = self._world_to_valve_local(geom, grasp_proxy_world)
        return float(np.linalg.norm(current_local - self._close_latched_grasp_local))

    def _attach_site_error_m(self, geom):
        """真实 connect site 和阀门抓点的距离，用来诊断启用约束时的初始残差。"""
        vec_world = self._attach_site_error_world(geom)
        if vec_world is None:
            return np.nan
        return float(np.linalg.norm(vec_world))

    def _attach_site_error_world(self, geom):
        """返回 hand proxy 相对阀门抓点的 world-frame 误差向量。"""
        hand_proxy_world = geom.get("hand_proxy_world", None) if geom else None
        valve_grasp_world = geom.get("grasp_world", None) if geom else None
        if hand_proxy_world is None or valve_grasp_world is None:
            return None
        return (
            np.asarray(hand_proxy_world, dtype=float).reshape(3)
            - np.asarray(valve_grasp_world, dtype=float).reshape(3)
        )

    def _attach_site_error_base(self, geom):
        """返回 hand proxy 相对阀门抓点的 base-frame 误差向量，便于判断抓取偏差方向。"""
        vec_world = self._attach_site_error_world(geom)
        if vec_world is None or geom is None or "world_to_base" not in geom:
            return np.full(3, np.nan, dtype=float)
        world_to_base = np.asarray(geom["world_to_base"], dtype=float).reshape(3, 3)
        return world_to_base @ vec_world

    def _raw_target_drift_m(self):
        if self._close_latched_world_geom is None or self._latest_raw_geom is None:
            return np.nan
        raw_grasp_base = self._valve_grasp_point_base(self._latest_raw_geom)
        raw_grasp_world = self._point_base_to_world(self._latest_raw_geom, raw_grasp_base)
        latched_grasp_world = np.asarray(self._close_latched_world_geom["grasp_world"], dtype=float).reshape(3)
        return float(np.linalg.norm(raw_grasp_world - latched_grasp_world))

    def _latch_close_grasp_frame(self, geom):
        if not self.latch_grasp_frame_on_close or self._close_latched_world_geom is not None:
            return self._control_geom(geom)

        base_to_world = np.asarray(geom["world_to_base"], dtype=float).reshape(3, 3).T
        center_base = np.asarray(geom["center_base"], dtype=float).reshape(3)
        grasp_base = self._valve_grasp_point_base(geom)
        axis_base = np.asarray(geom["axis_base"], dtype=float).reshape(3)
        axis_world = base_to_world @ axis_base
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-9)
        R_base = self._right_grasp_rotation_base(geom)

        grasp_world = self._point_base_to_world(geom, grasp_base)
        self._close_latched_world_geom = {
            "center_world": self._point_base_to_world(geom, center_base),
            "grasp_world": grasp_world,
            "axis_world": axis_world,
            "grasp_R_world": base_to_world @ R_base,
        }
        self._close_latched_grasp_local = self._world_to_valve_local(geom, grasp_world)
        self.logger.info(colored("[VALVE_GRASP] close grasp frame latched", "cyan"))
        return self._close_latched_geom(geom)

    def _right_grasp_rotation_base(self, geom):
        if geom is not None and "grasp_R_base" in geom:
            return np.asarray(geom["grasp_R_base"], dtype=float).reshape(3, 3)

        x_axis = _normalize_or_none(self.grasp_forward_axis_sign * self._approach_axis_base(geom))
        if x_axis is None:
            return self.EE_right_R
        radial = self._valve_grasp_point_base(geom) - np.asarray(geom["center_base"], dtype=float)
        radial = radial - np.dot(radial, x_axis) * x_axis
        radial_axis = _normalize_or_none(self.grasp_radial_axis_sign * radial)
        if radial_axis is None:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
            radial_axis = _normalize_or_none(fallback - np.dot(fallback, x_axis) * x_axis)
        if radial_axis is None:
            return self.EE_right_R

        if self.grasp_radial_axis == "z":
            z_axis = radial_axis
            y_axis = _normalize_or_none(np.cross(z_axis, x_axis))
            if y_axis is None:
                return self.EE_right_R
            z_axis = _normalize_or_none(np.cross(x_axis, y_axis))
        else:
            y_axis = radial_axis
            z_axis = _normalize_or_none(np.cross(x_axis, y_axis))
            if z_axis is None:
                return self.EE_right_R
            y_axis = _normalize_or_none(np.cross(z_axis, x_axis))

        if y_axis is None or z_axis is None:
            return self.EE_right_R
        return np.column_stack((x_axis, y_axis, z_axis))

    def _set_grasp_target(self, target_base, geom):
        target = np.asarray(target_base, dtype=float).reshape(3)
        if self.target_rate_limit_mps > 0.0:
            now = time.perf_counter()
            if self._rate_limited_target_base is None:
                self._rate_limited_target_base = self._current_target_base.copy()
                self._last_target_update_t = now
            dt = np.clip(now - self._last_target_update_t, 1e-3, 0.2)
            delta = target - self._rate_limited_target_base
            max_step = self.target_rate_limit_mps * dt
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_step:
                target = self._rate_limited_target_base + delta * (max_step / (delta_norm + 1e-9))
            self._rate_limited_target_base = target.copy()
            self._last_target_update_t = now

        self._current_target_base = target
        self._current_command_grasp_base = None
        if self.align_grasp_orientation and geom is not None:
            self.EE_right_R = self._right_grasp_rotation_base(geom)
        self._set_right_target(self._current_target_base)

    def _approach_compensated_target_base(self, target_base, robot_state_data, geom=None):
        """接近阶段抵消 wrist 和掌心代理点的稳态欠跟踪，避免还没碰到轮缘就一直等待。"""
        target = np.asarray(target_base, dtype=float).reshape(3)
        if self.approach_tracking_compensation_gain <= 0.0 or self.approach_tracking_compensation_max_m <= 0.0:
            compensated = target
        else:
            current = self._current_right_fake_ee_base(robot_state_data)
            compensation = self.approach_tracking_compensation_gain * (target - current)
            compensation_norm = float(np.linalg.norm(compensation))
            if compensation_norm > self.approach_tracking_compensation_max_m:
                compensation *= self.approach_tracking_compensation_max_m / (compensation_norm + 1e-9)
            compensated = target + compensation

        if (
            geom is None
            or self.approach_proxy_compensation_gain <= 0.0
            or self.approach_proxy_compensation_max_m <= 0.0
        ):
            return compensated

        desired_proxy = self._valve_grasp_point_base(geom)
        current_proxy = self._current_grasp_proxy_base(geom, robot_state_data)
        proxy_compensation = self.approach_proxy_compensation_gain * (desired_proxy - current_proxy)
        proxy_norm = float(np.linalg.norm(proxy_compensation))
        if proxy_norm > self.approach_proxy_compensation_max_m:
            proxy_compensation *= self.approach_proxy_compensation_max_m / (proxy_norm + 1e-9)
        return compensated + proxy_compensation

    def _contact_compensated_ee_target_base(
        self,
        target_base,
        geom,
        robot_state_data,
        desired_grasp_base=None,
    ):
        """接触后用实际抓握代理点闭环，抵消手指接触把手腕顶开的偏差。"""
        target = np.asarray(target_base, dtype=float).reshape(3)
        if (
            self.contact_tracking_compensation_gain <= 0.0
            or self.contact_tracking_compensation_max_m <= 0.0
            or geom is None
            or robot_state_data is None
        ):
            return target

        if desired_grasp_base is None:
            desired = self._valve_grasp_point_base(geom)
        else:
            desired = np.asarray(desired_grasp_base, dtype=float).reshape(3)
        current_proxy = self._current_grasp_proxy_base(geom, robot_state_data)
        compensation = self.contact_tracking_compensation_gain * (desired - current_proxy)
        compensation_norm = float(np.linalg.norm(compensation))
        if compensation_norm > self.contact_tracking_compensation_max_m:
            compensation *= self.contact_tracking_compensation_max_m / (compensation_norm + 1e-9)
        return target + compensation

    def _command_geom_for_grasp_base(self, geom, grasp_base):
        command_geom = dict(geom)
        command_geom["grasp_base"] = np.asarray(grasp_base, dtype=float).reshape(3)
        command_geom["grasp_base_is_effective"] = True
        if self.turn_rotate_grasp_orientation:
            # 转阀门时让手腕姿态随轮缘半径方向缓慢旋转，保持夹爪包络关系。
            command_geom.pop("grasp_R_base", None)
            command_geom.pop("grasp_frame_latched", None)
        return command_geom

    def _set_command_grasp_target(
        self,
        geom,
        command_grasp_base,
        robot_state_data=None,
        compensate=False,
        command_R_base=None,
    ):
        command_grasp_base = np.asarray(command_grasp_base, dtype=float).reshape(3)
        command_geom = self._command_geom_for_grasp_base(geom, command_grasp_base)
        if command_R_base is None:
            R = self._right_grasp_rotation_base(command_geom)
        else:
            R = np.asarray(command_R_base, dtype=float).reshape(3, 3)
            command_geom["grasp_R_base"] = R
            command_geom["grasp_frame_latched"] = True
        self._current_command_grasp_base = command_grasp_base.copy()
        self._current_target_base = command_grasp_base - R @ self.grasp_point_offset_ee
        if compensate:
            self._current_target_base = self._contact_compensated_ee_target_base(
                self._current_target_base,
                command_geom,
                robot_state_data,
                desired_grasp_base=command_grasp_base,
            )
        if self.align_grasp_orientation:
            self.EE_right_R = R
        self._set_right_target(self._current_target_base)
        return command_geom

    def _current_error(self, robot_state_data):
        current = self._current_right_fake_ee_base(robot_state_data)
        commanded = self._last_commanded_right_ee_base()
        error = float(np.linalg.norm(current - self._current_target_base))
        ik_error = float(np.linalg.norm(commanded - self._current_target_base)) if np.all(np.isfinite(commanded)) else np.nan
        servo_error = float(np.linalg.norm(current - commanded)) if np.all(np.isfinite(commanded)) else np.nan
        return current, commanded, error, ik_error, servo_error

    def _physical_grasp_error(self, geom, robot_state_data):
        current_grasp_proxy = self._current_grasp_proxy_base(geom, robot_state_data)
        return float(np.linalg.norm(current_grasp_proxy - self._valve_grasp_point_base(geom)))

    def _turn_actual_delta_deg(self, geom):
        valve_angle = float(geom.get("valve_angle", 0.0))
        return float(np.rad2deg(valve_angle - self._turn_start_angle))

    def _turn_absolute_delta_deg(self, geom):
        valve_angle = float(geom.get("valve_angle", 0.0))
        if self._turn_sequence_zero_angle is None:
            return self._turn_actual_delta_deg(geom)
        return float(np.rad2deg(valve_angle - self._turn_sequence_zero_angle))

    def _turn_sequence_cumulative_target_deg(self):
        if not self.turn_target_sequence_deg:
            return float(self.turn_target_deg)
        return float(sum(self.turn_target_sequence_deg[: self._turn_sequence_index + 1]))

    def _has_next_turn_sequence_target(self):
        return bool(
            self.turn_target_sequence_deg
            and self._turn_sequence_index + 1 < len(self.turn_target_sequence_deg)
        )

    def _start_next_turn_sequence_segment(self, geom, robot_state_data):
        if not self._has_next_turn_sequence_target():
            return False
        self._turn_sequence_index += 1
        self.turn_target_deg = float(self.turn_target_sequence_deg[self._turn_sequence_index])
        self._turn_final_error_deg = self.turn_target_deg
        self._write_valve_angle_hold(False)
        self._maybe_enable_contact_assist(geom)
        self.logger.info(
            colored(
                f"[VALVE_GRASP] next turn segment {self._turn_sequence_index + 1}/"
                f"{len(self.turn_target_sequence_deg)}: segment={self.turn_target_deg:+.1f}deg "
                f"cumulative_target={self._turn_sequence_cumulative_target_deg():+.1f}deg",
                "green",
            )
        )
        self._start_turn_segment(geom, robot_state_data)
        return True

    def _base_to_valve_world(self, geom):
        try:
            return np.asarray(geom["center_world"], dtype=float) - np.asarray(
                geom["base_pos_world"], dtype=float
            )
        except Exception:
            return np.full(3, np.nan, dtype=float)

    def _turn_planned_delta_deg(self, elapsed_s):
        if self._turn_nominal_duration_s <= 1e-6:
            return self.turn_target_deg
        progress = min(1.0, max(0.0, elapsed_s / self._turn_nominal_duration_s))
        return float(self.turn_target_deg * progress)

    def _clamp_turn_command_delta(self, delta_deg):
        low = min(0.0, self.turn_target_deg) - self.turn_command_lead_deg
        high = max(0.0, self.turn_target_deg) + self.turn_command_lead_deg
        return float(np.clip(delta_deg, low, high))

    def _project_turn_radius_world(self, point_world):
        radius = np.asarray(point_world, dtype=float).reshape(3) - self._turn_center_world
        return radius - np.dot(radius, self._turn_axis_world) * self._turn_axis_world

    def _turn_rotation_world_for_delta(self, delta_deg):
        # MuJoCo hinge 正方向和手端沿轮缘运动方向可能相反；用显式符号保证目标角定义清楚。
        return _axis_angle_to_rot(
            self._turn_axis_world,
            np.deg2rad(self.turn_trajectory_sign * delta_deg),
        )

    def _turn_target_world_for_delta(self, delta_deg):
        rot = self._turn_rotation_world_for_delta(delta_deg)
        if self._turn_start_point_world is not None and self._turn_displacement_r_world is not None:
            # 从当前已建立接触的掌心代理点出发，只按阀门半径生成切向圆弧位移。
            # 这样不会在进入 turn_valve 的第一帧把手从已夹住的位置拉到预设 site。
            return self._turn_start_point_world + rot @ self._turn_displacement_r_world - self._turn_displacement_r_world
        return self._turn_center_world + rot @ self._turn_r0_world

    def _turn_motion_radius_m(self):
        if self._turn_displacement_r_world is not None:
            return float(np.linalg.norm(self._turn_displacement_r_world))
        if self._turn_r0_world is not None:
            return float(np.linalg.norm(self._turn_r0_world))
        return np.nan

    def _set_turn_delta_target(self, geom, delta_deg, robot_state_data=None):
        self._turn_cmd_deg = float(delta_deg)
        self._turn_command_delta_deg = self._turn_cmd_deg
        rot = self._turn_rotation_world_for_delta(delta_deg)
        command_world = self._turn_target_world_for_delta(delta_deg)
        command_base = self._point_world_to_base(geom, command_world)
        command_R_base = None
        if self._turn_R0_world is not None:
            command_R_world = rot @ self._turn_R0_world
            world_to_base = np.asarray(geom["world_to_base"], dtype=float).reshape(3, 3)
            command_R_base = world_to_base @ command_R_world
        self._set_command_grasp_target(
            geom,
            command_base,
            robot_state_data=robot_state_data,
            compensate=self.turn_contact_compensation_enabled,
            command_R_base=command_R_base,
        )

    def _start_turn_segment(self, geom, robot_state_data):
        if self.pre_turn_angle_lock_enabled or self.pre_turn_angle_hold_enabled:
            self._write_valve_angle_control(False, False)
        self._turn_started = True
        self._turn_start_angle = float(geom.get("valve_angle", 0.0))
        if self._turn_sequence_zero_angle is None:
            self._turn_sequence_zero_angle = self._turn_start_angle
        self._turn_nominal_duration_s = max(1.0, abs(self.turn_target_deg) / self.turn_speed_deg)
        self._turn_duration_s = self._turn_nominal_duration_s
        if self.turn_control_mode == "closed_loop":
            self._turn_duration_s += max(0.0, self.turn_extra_settle_s)

        self._turn_center_world = np.asarray(geom["center_world"], dtype=float).copy()
        self._turn_axis_world = np.asarray(geom["axis_world"], dtype=float).copy()
        self._turn_axis_world = self._turn_axis_world / (np.linalg.norm(self._turn_axis_world) + 1e-9)
        R0_base = self._right_grasp_rotation_base(geom)
        command_start_base = self._current_target_base + R0_base @ self.grasp_point_offset_ee
        if self.turn_radius_mode in ("command", "command_delta_scaled"):
            # 从当前已经稳定接触的命令位姿起步，避免进入 turn_valve 时目标点跳变。
            r0_base = command_start_base
        elif self.turn_radius_mode == "actual":
            r0_base = self._current_grasp_proxy_base(geom, robot_state_data)
        else:
            r0_base = self._valve_grasp_point_base(geom)
        base_to_world = np.asarray(geom["world_to_base"], dtype=float).reshape(3, 3).T
        self._turn_R0_world = base_to_world @ R0_base
        start_point_world = self._point_base_to_world(geom, r0_base)
        r0_world = self._project_turn_radius_world(start_point_world)
        if np.linalg.norm(r0_world) < 1e-4:
            r0_world = self._project_turn_radius_world(np.asarray(geom["grasp_world"], dtype=float))
        self._turn_r0_world = r0_world
        self._turn_start_point_world = None
        self._turn_displacement_r_world = None
        if self.turn_radius_mode == "command_delta_scaled":
            target_radius = self._project_turn_radius_world(np.asarray(geom["grasp_world"], dtype=float))
            target_radius_norm = float(np.linalg.norm(target_radius))
            if self.turn_displacement_radius_m > 1e-4:
                displacement_radius_norm = self.turn_displacement_radius_m
            else:
                displacement_radius_norm = target_radius_norm * max(0.0, self.turn_displacement_radius_gain)
            direction = _normalize_or_none(r0_world)
            if direction is None:
                direction = _normalize_or_none(target_radius)
            if direction is not None and displacement_radius_norm > 1e-4:
                self._turn_start_point_world = start_point_world.copy()
                self._turn_displacement_r_world = direction * displacement_radius_norm

        self._turn_reference_delta_deg = 0.0
        self._turn_command_delta_deg = 0.0
        self._turn_hold_command_delta_deg = None
        self._turn_hold_settle_ok_started_t = None
        self._turn_final_error_deg = self.turn_target_deg
        self._turn_cmd_deg = 0.0
        self._last_turn_update_t = None
        self._turn_start_base_to_valve_x_m = float(self._base_to_valve_world(geom)[0])
        _, _, start_yaw = _base_rpy_deg(geom)
        self._turn_start_base_yaw_deg = float(start_yaw)
        self._turn_rows = []
        self._set_turn_delta_target(geom, 0.0, robot_state_data=robot_state_data)
        self.logger.info(
            colored(
                f"[VALVE_GRASP] start contact turn target={self.turn_target_deg:+.1f}deg "
                f"seq={self._turn_sequence_index + 1}/"
                f"{max(1, len(self.turn_target_sequence_deg))} "
                f"speed={self.turn_speed_deg:.1f}deg/s "
                f"radius={np.linalg.norm(self._turn_r0_world):.3f}m "
                f"motion_radius={self._turn_motion_radius_m():.3f}m "
                f"radius_mode={self.turn_radius_mode} "
                f"mode={self.turn_control_mode} "
                f"segment_settle={'on' if self.turn_segment_settle_enabled else 'off'}",
                "green",
            )
        )
        self._enter_state(self.TURN_VALVE)

    def _turn_completion_quality(self, geom, robot_state_data):
        """目标角保持阶段的抓握质量门槛；避免手已经脱开但角度恰好到位。"""
        reasons = []
        if self.turn_completion_require_grasp and not self._grasp_success(geom):
            reasons.append("grasp_lost")

        _, _, contact_health_ratio = self._contact_health(geom)
        if contact_health_ratio < self.turn_completion_min_contact_health_ratio:
            reasons.append(
                "contact_health="
                f"{contact_health_ratio:.2f}<{self.turn_completion_min_contact_health_ratio:.2f}"
            )

        relative_slip = self._relative_grasp_slip_m(geom, robot_state_data)
        if (
            self.turn_completion_max_slip_m > 0.0
            and np.isfinite(relative_slip)
            and relative_slip > self.turn_completion_max_slip_m
        ):
            reasons.append(f"slip={relative_slip:.4f}>{self.turn_completion_max_slip_m:.4f}m")

        grasp_error = self._physical_grasp_error(geom, robot_state_data)
        if (
            np.isfinite(self.turn_completion_max_grasp_error_m)
            and self.turn_completion_max_grasp_error_m > 0.0
            and np.isfinite(grasp_error)
            and grasp_error > self.turn_completion_max_grasp_error_m
        ):
            reasons.append(
                f"grasp_error={grasp_error:.4f}>{self.turn_completion_max_grasp_error_m:.4f}m"
            )

        return len(reasons) == 0, "|".join(reasons)

    def _turn_hold_ready_for_next_segment(self, geom, robot_state_data):
        elapsed = time.perf_counter() - self._state_enter_t
        if not self.turn_segment_settle_enabled:
            if elapsed < self.turn_hold_s:
                return False, None
            quality_ok, quality_reason = self._turn_completion_quality(geom, robot_state_data)
            self._turn_hold_quality_reason = quality_reason
            if self.turn_completion_timeout_is_failure and not quality_ok:
                return False, f"turn_hold_quality_failed:{quality_reason or 'unknown'}"
            return True, None

        actual_delta = self._turn_actual_delta_deg(geom) if self._turn_started else 0.0
        final_error = abs(float(self.turn_target_deg - actual_delta))
        valve_vel_deg_s = abs(float(np.rad2deg(float(geom.get("valve_vel", 0.0)))))
        quality_ok, quality_reason = self._turn_completion_quality(geom, robot_state_data)
        self._turn_hold_quality_reason = quality_reason
        stable_now = (
            final_error <= self.turn_segment_settle_error_deg
            and valve_vel_deg_s <= self.turn_segment_settle_vel_deg_s
            and quality_ok
        )
        now = time.perf_counter()
        if stable_now:
            if self._turn_hold_settle_ok_started_t is None:
                self._turn_hold_settle_ok_started_t = now
        else:
            self._turn_hold_settle_ok_started_t = None

        stable_s = (
            0.0
            if self._turn_hold_settle_ok_started_t is None
            else now - self._turn_hold_settle_ok_started_t
        )
        if elapsed >= self.turn_hold_s and stable_s >= self.turn_segment_settle_s:
            return True, None
        if elapsed >= self.turn_segment_settle_max_s:
            if self.turn_completion_timeout_is_failure:
                reason = (
                    f"turn_hold_settle_timeout:error={final_error:.2f}deg "
                    f"vel={valve_vel_deg_s:.2f}deg_s"
                )
                if quality_reason:
                    reason += f" {quality_reason}"
                return False, reason
            return True, None
        return False, None

    def _begin_pre_turn_settle(self, geom):
        # connect 刚打开时可能仍有几厘米几何残差；先保持阀门硬锁，让手臂和接触约束卸载。
        self._pre_turn_lock_released = False
        self._pre_turn_unlock_t = None
        if self._contact_assist_active and self.contact_assist_lock_settle_s > 0.0:
            self._write_valve_angle_lock(True, geom.get("valve_angle", 0.0))
        else:
            self._write_valve_angle_hold(True, geom.get("valve_angle", 0.0))
            self._pre_turn_lock_released = True
            self._pre_turn_unlock_t = time.perf_counter()
        self._enter_state(self.PRE_TURN_SETTLE)

    def _start_turn_or_settle(self, geom, robot_state_data):
        if self.turn_pre_settle_s > 0.0 and self.pre_turn_angle_lock_enabled:
            self._begin_pre_turn_settle(geom)
        else:
            self._start_turn_segment(geom, robot_state_data)

    def _turn_entry_ready(self, geom, robot_state_data, check_min_hold_time=True):
        if not self.enable_turn or abs(self.turn_target_deg) <= 1e-6:
            return False
        if check_min_hold_time and time.perf_counter() - self._state_enter_t < self.turn_entry_min_hold_s:
            return False
        status = self._turn_entry_status(geom, robot_state_data)
        return bool(
            status["slip_ok"]
            and status["success_ok"]
            and status["attach_ok"]
            and status["contact_force_max_ok"]
        )

    def _turn_entry_status(self, geom, robot_state_data):
        """汇总进入转动阶段的判据，便于日志定位是哪一项卡住。"""
        now = time.perf_counter()
        relative_slip = self._relative_grasp_slip_m(geom, robot_state_data)
        slip_ok = np.isfinite(relative_slip) and relative_slip <= self.turn_entry_max_slip_m
        success_now = self._grasp_success(geom)
        success_age_s = (
            float("inf")
            if self._last_grasp_success_t is None
            else max(0.0, now - self._last_grasp_success_t)
        )
        recent_success_ok = success_age_s <= self.turn_entry_success_memory_s
        contact_count = int(geom.get("contact_count", 0))
        normal_force = float(geom.get("contact_normal_force", 0.0))
        contact_force_min_ok = (
            contact_count >= self.turn_entry_min_contact_count
            and normal_force >= self.turn_entry_min_contact_force
        )
        contact_force_max_ok = (
            not np.isfinite(self.turn_entry_max_contact_force)
            or normal_force <= self.turn_entry_max_contact_force
        )
        contact_ok = contact_force_min_ok and contact_force_max_ok
        # MuJoCo 真实接触会在 palm/thumb/finger 之间抖动；曾经稳定抓住且当前接触力仍足够时，
        # 允许进入转动，避免被单帧接触拓扑丢失阻塞。
        success_ok = (
            success_now
            or not self.turn_entry_require_success
            or (recent_success_ok and contact_ok)
        )
        attach_site_error = self._attach_site_error_m(geom)
        attach_ok = (
            not np.isfinite(attach_site_error)
            or attach_site_error <= self.turn_entry_max_attach_site_error_m
        )
        return {
            "slip": float(relative_slip),
            "slip_ok": bool(slip_ok),
            "success_now": bool(success_now),
            "success_age_s": float(success_age_s),
            "recent_success_ok": bool(recent_success_ok),
            "contact_count": contact_count,
            "normal_force": normal_force,
            "contact_ok": bool(contact_ok),
            "contact_force_min_ok": bool(contact_force_min_ok),
            "contact_force_max_ok": bool(contact_force_max_ok),
            "success_ok": bool(success_ok),
            "attach_site_error": float(attach_site_error),
            "attach_ok": bool(attach_ok),
        }

    def _format_turn_entry_status(self, status):
        success_age = status["success_age_s"]
        success_age_text = "inf" if not np.isfinite(success_age) else f"{success_age:.2f}s"
        return (
            f"success={int(status['success_now'])}, recent_success={int(status['recent_success_ok'])}"
            f"({success_age_text}), contacts={status['contact_count']}, "
            f"force={status['normal_force']:.2f}N/"
            f"{self.turn_entry_min_contact_force:.2f}-{self.turn_entry_max_contact_force:.2f}N, "
            f"slip={status['slip']:.4f}m/"
            f"{self.turn_entry_max_slip_m:.4f}m, attach_err={status['attach_site_error']:.4f}m/"
            f"{self.turn_entry_max_attach_site_error_m:.4f}m"
        )

    def _turn_abort_reason(self, geom, robot_state_data):
        valve_vel = abs(float(geom.get("valve_vel", 0.0)))
        if self.turn_abort_valve_vel_rad_s > 0.0 and valve_vel > self.turn_abort_valve_vel_rad_s:
            return f"valve_vel={valve_vel:.2f}rad/s > {self.turn_abort_valve_vel_rad_s:.2f}rad/s"
        relative_slip = self._relative_grasp_slip_m(geom, robot_state_data)
        if np.isfinite(relative_slip) and relative_slip > self.turn_abort_slip_m:
            return f"slip={relative_slip:.4f}m > {self.turn_abort_slip_m:.4f}m"
        normal_force = float(geom.get("contact_normal_force", 0.0))
        if self.turn_abort_contact_force_n > 0.0 and normal_force > self.turn_abort_contact_force_n:
            return f"contact_force={normal_force:.1f}N > {self.turn_abort_contact_force_n:.1f}N"
        base_roll, base_pitch, _ = _base_rpy_deg(geom)
        base_tilt = max(abs(base_roll), abs(base_pitch))
        if self.turn_abort_base_tilt_deg > 0.0 and base_tilt > self.turn_abort_base_tilt_deg:
            return f"base_tilt={base_tilt:.1f}deg > {self.turn_abort_base_tilt_deg:.1f}deg"
        base_to_valve = self._base_to_valve_world(geom)
        base_to_valve_x = float(base_to_valve[0])
        if (
            self.turn_abort_min_base_to_valve_x_m > 0.0
            and np.isfinite(base_to_valve_x)
            and base_to_valve_x < self.turn_abort_min_base_to_valve_x_m
        ):
            return (
                f"base_to_valve_x={base_to_valve_x:.3f}m "
                f"< {self.turn_abort_min_base_to_valve_x_m:.3f}m"
            )
        if (
            self.turn_abort_max_base_approach_m > 0.0
            and self._turn_start_base_to_valve_x_m is not None
            and np.isfinite(base_to_valve_x)
            and np.isfinite(self._turn_start_base_to_valve_x_m)
        ):
            approach = self._turn_start_base_to_valve_x_m - base_to_valve_x
            if approach > self.turn_abort_max_base_approach_m:
                return (
                    f"base_approach={approach:.3f}m "
                    f"> {self.turn_abort_max_base_approach_m:.3f}m"
                )
        if (
            self.turn_abort_max_base_yaw_drift_deg > 0.0
            and self._turn_start_base_yaw_deg is not None
        ):
            _, _, base_yaw = _base_rpy_deg(geom)
            if np.isfinite(base_yaw) and np.isfinite(self._turn_start_base_yaw_deg):
                yaw_drift = abs(_angle_diff_deg(base_yaw, self._turn_start_base_yaw_deg))
                if yaw_drift > self.turn_abort_max_base_yaw_drift_deg:
                    return (
                        f"base_yaw_drift={yaw_drift:.1f}deg "
                        f"> {self.turn_abort_max_base_yaw_drift_deg:.1f}deg"
                    )
        if self.turn_abort_require_feet_contact_stable and not bool(geom.get("feet_contact_stable", False)):
            return "feet_contact_stable=0"
        if self.turn_abort_min_total_foot_force_n > 0.0:
            total_foot_force = float(geom.get("left_foot_force", 0.0)) + float(
                geom.get("right_foot_force", 0.0)
            )
            if total_foot_force < self.turn_abort_min_total_foot_force_n:
                return (
                    f"total_foot_force={total_foot_force:.1f}N "
                    f"< {self.turn_abort_min_total_foot_force_n:.1f}N"
                )
        return None

    def _update_turn_target(self, geom, robot_state_data=None, hold_reference=False):
        if self._turn_center_world is None or self._turn_axis_world is None or self._turn_r0_world is None:
            return
        now = time.perf_counter()
        elapsed = now - self._state_enter_t
        planned_elapsed = self._turn_nominal_duration_s if hold_reference else elapsed
        desired_delta = self._turn_planned_delta_deg(planned_elapsed)
        actual_delta = self._turn_actual_delta_deg(geom)
        self._turn_final_error_deg = self.turn_target_deg - actual_delta

        if hold_reference or elapsed >= self._turn_nominal_duration_s:
            reference_delta = self.turn_target_deg
        else:
            reference_delta = desired_delta
        self._turn_reference_delta_deg = float(reference_delta)

        if self.turn_control_mode == "closed_loop":
            feedback = np.clip(
                self.turn_feedback_gain * (reference_delta - actual_delta),
                -self.turn_command_lead_deg,
                self.turn_command_lead_deg,
            )
            raw_command_delta = self._clamp_turn_command_delta(desired_delta + feedback)
            if self._last_turn_update_t is None:
                command_delta = raw_command_delta
            else:
                dt = np.clip(now - self._last_turn_update_t, 1e-3, 0.2)
                max_step = self.turn_command_speed_deg * dt
                step = np.clip(raw_command_delta - self._turn_command_delta_deg, -max_step, max_step)
                command_delta = self._turn_command_delta_deg + step
            self._last_turn_update_t = now
            command_delta = self._clamp_turn_command_delta(command_delta)
        else:
            command_delta = reference_delta

        self._set_turn_delta_target(geom, command_delta, robot_state_data=robot_state_data)

        turn_time_done = elapsed >= self._turn_nominal_duration_s
        turn_timed_out = elapsed >= self._turn_duration_s
        target_reached = abs(self._turn_final_error_deg) <= self.turn_tolerance_deg
        early_target_reached = (
            self.turn_finish_on_target_reached
            and elapsed >= self.turn_min_s
            and target_reached
        )
        if not hold_reference and (early_target_reached or (turn_time_done and (target_reached or turn_timed_out))):
            self._prepare_turn_hold_command(geom)
            self._enter_state(self.TURN_HOLD)

    def _prepare_turn_hold_command(self, geom):
        """选择进入目标保持阶段时的手端圆弧命令，避免把闭环 lead 当作最终目标。"""
        if self.turn_hold_command_mode == "target":
            hold_delta = self.turn_target_deg
        elif self.turn_hold_command_mode == "actual":
            hold_delta = self._turn_actual_delta_deg(geom)
        else:
            hold_delta = self._turn_command_delta_deg
        self._turn_hold_command_delta_deg = self._clamp_turn_command_delta(float(hold_delta))

    def _hold_turn_command_target(self, geom, robot_state_data=None):
        """保持进入 turn_hold 时的手端命令，避免到达目标后继续推阀门。"""
        if self._turn_hold_command_delta_deg is None:
            self._turn_hold_command_delta_deg = float(self._turn_command_delta_deg)
        actual_delta = self._turn_actual_delta_deg(geom)
        self._turn_reference_delta_deg = float(self.turn_target_deg)
        self._turn_final_error_deg = float(self.turn_target_deg - actual_delta)
        self._set_turn_delta_target(
            geom,
            float(self._turn_hold_command_delta_deg),
            robot_state_data=robot_state_data,
        )

    def _summarize_turn(self, result="completed", failure_reason=""):
        if not self._turn_rows:
            return
        rows = list(self._turn_rows)

        def vals(key):
            return np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)

        def finite_vals(key):
            arr = vals(key)
            return arr[np.isfinite(arr)]

        def mean_or_nan(key):
            arr = finite_vals(key)
            return float(np.mean(arr)) if arr.size else np.nan

        def p90_or_nan(key):
            arr = finite_vals(key)
            return float(np.nanpercentile(arr, 90)) if arr.size else np.nan

        def max_or_nan(key, abs_value=False):
            arr = finite_vals(key)
            if not arr.size:
                return np.nan
            if abs_value:
                arr = np.abs(arr)
            return float(np.max(arr))

        def last_or_nan(key):
            if not rows:
                return np.nan
            try:
                value = float(rows[-1].get(key, np.nan))
            except (TypeError, ValueError):
                return np.nan
            return value if np.isfinite(value) else np.nan

        angle_error = vals("turn_angle_error_deg")
        final_error = vals("turn_final_error_deg")
        actual_delta = vals("turn_actual_delta_deg")
        segment_summary = {
            "type": "segment",
            "trial_id": os.path.splitext(os.path.basename(self.log_file))[0],
            "segment_id": int(self._turn_sequence_index + 1),
            "segment_count": int(max(1, len(self.turn_target_sequence_deg))),
            "target_delta_deg": float(self.turn_target_deg),
            "target_cumulative_deg": float(self._turn_sequence_cumulative_target_deg()),
            "actual_delta_deg": float(actual_delta[-1]) if actual_delta.size else np.nan,
            "final_error_deg": float(final_error[-1]) if final_error.size else np.nan,
            "trajectory_mae_deg": float(np.mean(np.abs(angle_error))) if angle_error.size else np.nan,
            "eef_mean_error_m": mean_or_nan("right_eef_pos_error_m"),
            "slip_p90_m": p90_or_nan("relative_slip_m"),
            "final_relative_slip_m": last_or_nan("relative_slip_m"),
            "grasp_success_ratio": mean_or_nan("grasp_success"),
            "final_grasp_success": bool(last_or_nan("grasp_success") >= 0.5),
            "contact_health_ratio": mean_or_nan("contact_health_ratio"),
            "final_contact_health_ratio": last_or_nan("contact_health_ratio"),
            "soft_connect_active_ratio": mean_or_nan("soft_connect_active"),
            "contact_force_mean_n": mean_or_nan("contact_force_n"),
            "contact_force_peak_n": max_or_nan("contact_force_n"),
            "final_valve_vel_deg_s": last_or_nan("valve_vel_deg_s"),
            "angle_brake_peak_torque": max_or_nan("angle_brake_torque", abs_value=True),
            "max_tilt_deg": max_or_nan("base_tilt_deg"),
            "base_approach_m": max_or_nan("base_approach_m"),
            "base_yaw_drift_deg": max_or_nan("base_yaw_drift_deg"),
            "completion_quality_reason": _csv_token(self._turn_hold_quality_reason),
            "result": result,
            "failure_reason": _csv_token(failure_reason or self._failure_reason),
        }
        self._turn_segment_summaries.append(segment_summary)
        self._write_summary_record(segment_summary)
        self.logger.info(
            colored(
                f"[VALVE_GRASP] contact turn summary target={self.turn_target_deg:+.1f}deg "
                f"actual_final={segment_summary['actual_delta_deg']:+.1f}deg "
                f"final_error={segment_summary['final_error_deg']:+.2f}deg "
                f"traj_mae={segment_summary['trajectory_mae_deg']:.2f}deg "
                f"ee_mean={segment_summary['eef_mean_error_m']:.4f}m "
                f"slip_p90={segment_summary['slip_p90_m']:.4f}m "
                f"final_slip={segment_summary['final_relative_slip_m']:.4f}m "
                f"contact_health={segment_summary['contact_health_ratio']:.2f} "
                f"final_grasp={int(segment_summary['final_grasp_success'])} "
                f"soft_ratio={segment_summary['soft_connect_active_ratio']:.2f} "
                f"force_peak={segment_summary['contact_force_peak_n']:.1f}N "
                f"brake_peak={segment_summary['angle_brake_peak_torque']:.2f}",
                "cyan",
            )
        )
        self._turn_rows = []

    def _write_overall_summary(self, overall_result="done"):
        if self._overall_summary_written:
            return
        if not self._turn_segment_summaries:
            return
        self._overall_summary_written = True
        sequence_target = (
            float(sum(self.turn_target_sequence_deg))
            if self.turn_target_sequence_deg
            else float(self.turn_target_deg)
        )
        sequence_actual = float(sum(row.get("actual_delta_deg", 0.0) for row in self._turn_segment_summaries))
        sequence_error = float(sequence_target - sequence_actual)
        worst = max(
            self._turn_segment_summaries,
            key=lambda row: abs(float(row.get("final_error_deg", 0.0))),
        )
        summary = {
            "type": "overall",
            "trial_id": os.path.splitext(os.path.basename(self.log_file))[0],
            "sequence_target_deg": sequence_target,
            "sequence_actual_deg": sequence_actual,
            "sequence_final_error_deg": sequence_error,
            "total_duration_s": (
                0.0 if self._test_start_t is None else float(time.perf_counter() - self._test_start_t)
            ),
            "overall_result": overall_result,
            "worst_segment_id": int(worst.get("segment_id", -1)),
            "max_contact_force_n": max(
                float(row.get("contact_force_peak_n", 0.0)) for row in self._turn_segment_summaries
            ),
            "max_relative_slip_m": max(
                float(row.get("slip_p90_m", 0.0)) for row in self._turn_segment_summaries
            ),
            "max_final_relative_slip_m": max(
                float(row.get("final_relative_slip_m", 0.0)) for row in self._turn_segment_summaries
            ),
            "min_grasp_success_ratio": min(
                float(row.get("grasp_success_ratio", 0.0)) for row in self._turn_segment_summaries
            ),
            "min_final_contact_health_ratio": min(
                float(row.get("final_contact_health_ratio", 0.0)) for row in self._turn_segment_summaries
            ),
            "max_base_tilt_deg": max(float(row.get("max_tilt_deg", 0.0)) for row in self._turn_segment_summaries),
            "max_base_approach_m": max(
                float(row.get("base_approach_m", 0.0)) for row in self._turn_segment_summaries
            ),
            "max_base_yaw_drift_deg": max(
                float(row.get("base_yaw_drift_deg", 0.0)) for row in self._turn_segment_summaries
            ),
            "failure_reason": _csv_token(self._failure_reason),
        }
        self._write_summary_record(summary)
        self.logger.info(
            colored(
                f"[VALVE_GRASP] sequence summary target={sequence_target:+.1f}deg "
                f"actual={sequence_actual:+.1f}deg error={sequence_error:+.2f}deg "
                f"result={overall_result} worst_segment={summary['worst_segment_id']}",
                "cyan",
            )
        )

    def _turn_point_base_for_delta(self, geom, delta_deg):
        if (
            not self._turn_started
            or self._turn_center_world is None
            or self._turn_axis_world is None
            or self._turn_r0_world is None
        ):
            return None
        return self._point_world_to_base(geom, self._turn_target_world_for_delta(delta_deg))

    def _turn_arc_points_base(self, geom, delta_deg, max_points=18):
        if self._turn_point_base_for_delta(geom, 0.0) is None:
            return []
        point_count = int(np.clip(np.ceil(abs(delta_deg) / 6.0) + 2, 2, max_points))
        return [
            self._turn_point_base_for_delta(geom, sample_delta).tolist()
            for sample_delta in np.linspace(0.0, float(delta_deg), point_count)
        ]

    def _contact_groups(self, geom, real_only=False):
        groups = {"palm": False, "thumb": False, "index": False, "middle": False}
        pair_key = "real_contact_pairs" if real_only else "contact_pairs"
        for pair in geom.get(pair_key, []) if geom else []:
            pair = str(pair)
            if "right_rubber_hand" in pair or "right_hand_palm" in pair or "right_grasp_proxy_palm" in pair:
                groups["palm"] = True
            if "right_hand_thumb" in pair or "right_grasp_proxy_thumb" in pair:
                groups["thumb"] = True
            if "right_hand_index" in pair or "right_grasp_proxy_index" in pair:
                groups["index"] = True
            if "right_hand_middle" in pair or "right_grasp_proxy_middle" in pair:
                groups["middle"] = True
        return groups

    def _contact_group_counts(self, geom, real_only=True):
        counts = {"palm": 0, "thumb": 0, "index": 0, "middle": 0}
        pair_key = "real_contact_pairs" if real_only else "contact_pairs"
        for pair in geom.get(pair_key, []) if geom else []:
            pair = str(pair)
            if "right_rubber_hand" in pair or "right_hand_palm" in pair or "right_grasp_proxy_palm" in pair:
                counts["palm"] += 1
            if "right_hand_thumb" in pair or "right_grasp_proxy_thumb" in pair:
                counts["thumb"] += 1
            if "right_hand_index" in pair or "right_grasp_proxy_index" in pair:
                counts["index"] += 1
            if "right_hand_middle" in pair or "right_grasp_proxy_middle" in pair:
                counts["middle"] += 1
        return counts

    def _contact_health(self, geom):
        # 第一轮只做可观测性：健康度按真实掌心/三指是否都有接触来打分。
        counts = self._contact_group_counts(geom, real_only=True)
        score = sum(1 for key in ("palm", "thumb", "index", "middle") if counts[key] > 0)
        ratio = score / 4.0
        return counts, float(score), float(ratio)

    def _grasp_success(self, geom):
        groups = self._contact_groups(geom, real_only=self.success_require_real_hand_contact)
        has_thumb = groups["thumb"] or not self.success_require_thumb
        has_finger = groups["index"] or groups["middle"] or not self.success_require_finger
        has_index = groups["index"] or not self.success_require_index
        has_middle = groups["middle"] or not self.success_require_middle
        contact_count_key = "real_contact_count" if self.success_require_real_hand_contact else "contact_count"
        force_key = (
            "real_contact_normal_force"
            if self.success_require_real_hand_contact
            else "contact_normal_force"
        )
        return (
            int(geom.get(contact_count_key, 0)) >= self.success_min_contacts
            and float(geom.get(force_key, 0.0)) >= self.success_min_force
            and groups["palm"]
            and has_thumb
            and has_finger
            and has_index
            and has_middle
        )

    def _ready_to_close(self, geom, grasp_error):
        groups = self._contact_groups(geom, real_only=self.close_require_real_hand_contact)
        count_key = "real_contact_count" if self.close_require_real_hand_contact else "contact_count"
        force_key = (
            "real_contact_normal_force"
            if self.close_require_real_hand_contact
            else "contact_normal_force"
        )
        normal_force = float(geom.get(force_key, 0.0))
        # 真抓握阶段先等掌心/腕掌区域贴上阀门，再闭合手指，避免提前空夹。
        if self.require_palm_contact_to_close:
            palm_ready = (
                groups["palm"]
                and normal_force >= self.palm_close_min_force
                and grasp_error <= self.palm_close_max_error
            )
            if palm_ready:
                return True
            # 可抓握阀门上指尖可能先卡入局部抓握套；允许近距离双指接触后闭手，
            # 但最终成功仍要求 palm/thumb/index/middle 完整接触。
            if self.allow_finger_contact_close:
                if self.finger_close_require_index_middle:
                    finger_ready = groups["index"] and groups["middle"]
                else:
                    finger_ready = groups["index"] or groups["middle"]
                attach_site_error = self._attach_site_error_m(geom)
                attach_ok = (
                    not np.isfinite(attach_site_error)
                    or attach_site_error <= self.finger_close_max_attach_error
                )
                return (
                    finger_ready
                    and int(geom.get(count_key, 0)) >= self.finger_close_min_contacts
                    and normal_force >= self.finger_close_min_force
                    and grasp_error <= self.finger_close_max_error
                    and attach_ok
                )
            return False
        if grasp_error <= self.grasp_error_m:
            return True
        if not self.close_on_contact:
            return False
        return (
            int(geom.get(count_key, 0)) >= self.contact_close_min_contacts
            and normal_force >= self.contact_close_min_force
        )

    def _update_task_state(self, robot_state_data):
        raw_geom = self.geometry_provider.read()
        if raw_geom is not None:
            self._latest_raw_geom = raw_geom
            geom = self._control_geom(raw_geom)
            self._latest_geom = geom
        else:
            geom = self._latest_geom

        if geom is None:
            self._enter_state(self.WAIT_GEOMETRY)
            return

        if self._grasp_success(geom):
            self._last_grasp_success_t = time.perf_counter()

        if self.task_state == self.WAIT_GEOMETRY:
            self._write_attach_state(False)
            if self.pre_turn_angle_lock_enabled:
                self._write_valve_angle_lock(True, geom.get("valve_angle", 0.0))
            elif self.pre_turn_angle_hold_enabled:
                self._write_valve_angle_hold(True, geom.get("valve_angle", 0.0))
            self._write_hand_state("open")
            if not self._task_pose_ready(geom):
                if self.task_state != self.FAILED:
                    self._enter_state(self.WAIT_TASK_POSE)
                return
            self._set_grasp_target(self._pregrasp_target_base(geom), geom)
            self._task_ready_t = time.perf_counter()
            self._enter_state(self.MOVE_PREGRASP)
            return

        if self.task_state == self.WAIT_TASK_POSE:
            self._write_attach_state(False)
            if self.pre_turn_angle_lock_enabled:
                self._write_valve_angle_lock(True, geom.get("valve_angle", 0.0))
            elif self.pre_turn_angle_hold_enabled:
                self._write_valve_angle_hold(True, geom.get("valve_angle", 0.0))
            self._write_hand_state("open")
            if not self._task_pose_ready(geom):
                return
            if self.task_state == self.FAILED:
                return
            self._set_grasp_target(self._pregrasp_target_base(geom), geom)
            self._task_ready_t = time.perf_counter()
            self._enter_state(self.MOVE_PREGRASP)
            return

        if self._task_ready_t is not None and time.perf_counter() - self._task_ready_t < self.task_start_delay_s:
            self._set_grasp_target(self._pregrasp_target_base(geom), geom)
            return

        if self.task_state == self.MOVE_PREGRASP:
            self._set_grasp_target(self._pregrasp_target_base(geom), geom)
            _, _, error, _, _ = self._current_error(robot_state_data)
            if error <= self.pregrasp_error_m or time.perf_counter() - self._state_enter_t >= self.pregrasp_max_s:
                self._enter_state(self.APPROACH_GRASP)
            return

        if self.task_state == self.APPROACH_GRASP:
            target_base = self._approach_compensated_target_base(
                self._ee_grasp_target_base(geom),
                robot_state_data,
                geom,
            )
            self._set_grasp_target(target_base, geom)
            grasp_error = self._physical_grasp_error(geom, robot_state_data)
            if self._ready_to_close(geom, grasp_error) or (
                self.approach_max_s > 0.0 and time.perf_counter() - self._state_enter_t >= self.approach_max_s
            ):
                geom = self._latch_close_grasp_frame(geom)
                self._latest_geom = geom
                self._set_grasp_target(self._ee_grasp_target_base(geom), geom)
                self._enter_state(self.CLOSE_HAND)
            return

        if self.task_state == self.CLOSE_HAND:
            close_target = self._contact_compensated_ee_target_base(
                self._ee_grasp_target_base(geom),
                geom,
                robot_state_data,
            )
            self._set_grasp_target(close_target, geom)
            if not self._close_command_sent:
                self._write_hand_state("close")
                self._close_command_sent = True
                self.logger.info(colored("[VALVE_GRASP] close right hand", "cyan"))

            now = time.perf_counter()
            close_elapsed = now - self._state_enter_t
            if self._grasp_success(geom):
                if self._close_success_started_t is None:
                    self._close_success_started_t = now
                    self.logger.info(colored("[VALVE_GRASP] stable contact candidate", "cyan"))
            else:
                self._close_success_started_t = None

            success_elapsed = 0.0 if self._close_success_started_t is None else now - self._close_success_started_t
            valve_vel_ok = abs(float(geom.get("valve_vel", 0.0))) <= self.hold_entry_max_abs_valve_vel
            ready_for_hold = (
                close_elapsed >= self.close_wait_s
                and success_elapsed >= self.hold_entry_min_success_s
                and valve_vel_ok
            )
            close_timed_out = close_elapsed >= self.close_max_s
            if ready_for_hold or close_timed_out:
                if close_timed_out and not ready_for_hold:
                    if not self.enter_hold_on_close_timeout:
                        self.logger.warning(
                            colored(
                                "[VALVE_GRASP] close stage timed out without valid grasp; marking failed",
                                "yellow",
                            )
                        )
                        self._set_failure_reason("close_stage_timeout")
                        self._enter_state(self.FAILED)
                        return
                    self.logger.warning(
                        colored(
                            "[VALVE_GRASP] close stage timed out before fully settled; entering hold for diagnosis",
                            "yellow",
                        )
                    )
                self._latch_hold_grasp_local(geom, robot_state_data)
                # 真实接触已经形成后，才可选启用柔顺 connect，避免一开始就用约束“拉”到阀门。
                self._maybe_enable_contact_assist(geom)
                if (
                    self.turn_start_on_hold_entry
                    and self.enable_turn
                    and abs(self.turn_target_deg) > 1e-6
                    and self._turn_entry_ready(geom, robot_state_data)
                ):
                    # 接触包络刚形成时最稳定；真实接触转动 baseline 直接利用这个窗口进入转动。
                    self._start_turn_or_settle(geom, robot_state_data)
                else:
                    self._enter_state(self.HOLD_GRASP)
            return

        if self.task_state == self.HOLD_GRASP:
            hold_geom = self._hold_tracking_geom(geom)
            hold_target = self._contact_compensated_ee_target_base(
                self._ee_grasp_target_base(hold_geom),
                hold_geom,
                robot_state_data,
            )
            self._set_grasp_target(hold_target, hold_geom)
            self._maybe_enable_contact_assist(geom)
            if self._grasp_success(geom) and not self._success_reported:
                self._success_reported = True
                self.logger.info(
                    colored(
                        f"[VALVE_GRASP] success: contacts={geom.get('contact_count', 0)}, "
                        f"normal_force={geom.get('contact_normal_force', 0.0):.2f}N",
                        "green",
                    )
                )
            hold_elapsed = time.perf_counter() - self._state_enter_t
            if self._turn_entry_ready(geom, robot_state_data):
                self._start_turn_or_settle(geom, robot_state_data)
                return
            if hold_elapsed >= self.hold_s:
                if self.enable_turn and abs(self.turn_target_deg) > 1e-6:
                    status = self._turn_entry_status(geom, robot_state_data)
                    self.logger.warning(
                        colored(
                            "[VALVE_GRASP] turn entry failed: "
                            f"{self._format_turn_entry_status(status)}",
                            "yellow",
                        )
                    )
                    self._set_failure_reason(
                        "turn_entry_failed:" + self._format_turn_entry_status(status)
                    )
                    self._enter_state(self.FAILED)
                else:
                    self._enter_state(self.DONE)
            return

        if self.task_state == self.PRE_TURN_SETTLE:
            hold_geom = self._hold_tracking_geom(geom)
            hold_target = self._contact_compensated_ee_target_base(
                self._ee_grasp_target_base(hold_geom),
                hold_geom,
                robot_state_data,
            )
            self._set_grasp_target(hold_target, hold_geom)
            now = time.perf_counter()
            if not self._pre_turn_lock_released:
                abort_reason = self._turn_abort_reason(geom, robot_state_data)
                if abort_reason is not None:
                    self.logger.warning(colored(f"[VALVE_GRASP] pre-turn aborted: {abort_reason}", "yellow"))
                    self._set_failure_reason(abort_reason, abort=True)
                    self._set_contact_assist(False)
                    self._enter_state(self.FAILED)
                    return
                if now - self._state_enter_t < self.contact_assist_lock_settle_s:
                    return
                self._write_valve_angle_hold(True, geom.get("valve_angle", 0.0))
                self._pre_turn_lock_released = True
                self._pre_turn_unlock_t = now
                self.logger.info(colored("[VALVE_GRASP] pre-turn valve lock -> soft hold", "cyan"))
                return

            unlock_t = self._pre_turn_unlock_t if self._pre_turn_unlock_t is not None else self._state_enter_t
            settle_elapsed = now - unlock_t
            abort_reason = self._turn_abort_reason(geom, robot_state_data)
            if abort_reason is not None:
                self.logger.warning(colored(f"[VALVE_GRASP] pre-turn aborted: {abort_reason}", "yellow"))
                self._set_failure_reason(abort_reason, abort=True)
                self._set_contact_assist(False)
                self._enter_state(self.FAILED)
                return
            valve_vel_ok = abs(float(geom.get("valve_vel", 0.0))) <= self.turn_pre_settle_max_abs_valve_vel
            turn_entry_ok = self._turn_entry_ready(geom, robot_state_data, check_min_hold_time=False)
            if settle_elapsed >= self.turn_pre_settle_s and valve_vel_ok:
                if turn_entry_ok:
                    self._start_turn_segment(geom, robot_state_data)
                else:
                    status = self._turn_entry_status(geom, robot_state_data)
                    self.logger.warning(
                        colored(
                            "[VALVE_GRASP] pre-turn settle rejected: "
                            f"{self._format_turn_entry_status(status)}",
                            "yellow",
                        )
                    )
                    self._set_failure_reason(
                        "pre_turn_settle_rejected:" + self._format_turn_entry_status(status)
                    )
                    self._enter_state(self.FAILED)
                return
            if settle_elapsed >= self.turn_pre_settle_max_s:
                self.logger.warning(
                    colored(
                        "[VALVE_GRASP] pre-turn settle timed out before valid turn entry",
                        "yellow",
                    )
                )
                self._set_failure_reason("pre_turn_settle_timeout")
                self._enter_state(self.FAILED)
            return

        if self.task_state == self.TURN_VALVE:
            abort_reason = self._turn_abort_reason(geom, robot_state_data)
            if abort_reason is not None:
                self.logger.warning(colored(f"[VALVE_GRASP] turn aborted: {abort_reason}", "yellow"))
                self._set_failure_reason(abort_reason, abort=True)
                self._set_contact_assist(False)
                self._enter_state(self.FAILED)
                return
            self._update_turn_target(geom, robot_state_data=robot_state_data)
            return

        if self.task_state == self.TURN_HOLD:
            abort_reason = self._turn_abort_reason(geom, robot_state_data)
            if abort_reason is not None:
                self.logger.warning(colored(f"[VALVE_GRASP] turn hold aborted: {abort_reason}", "yellow"))
                self._set_failure_reason(abort_reason, abort=True)
                self._set_contact_assist(False)
                self._enter_state(self.FAILED)
                return
            self._hold_turn_command_target(geom, robot_state_data=robot_state_data)
            ready, quality_failure = self._turn_hold_ready_for_next_segment(geom, robot_state_data)
            if quality_failure is not None:
                self.logger.warning(colored(f"[VALVE_GRASP] turn hold quality failed: {quality_failure}", "yellow"))
                self._set_failure_reason(quality_failure, abort=True)
                self._set_contact_assist(False)
                self._enter_state(self.FAILED)
                return
            if ready:
                self._summarize_turn()
                if self._start_next_turn_sequence_segment(geom, robot_state_data):
                    return
                self._enter_state(self.DONE)
            return

        if self.task_state == self.DONE:
            if self.contact_assist_release_on_done and self._contact_assist_active:
                self._set_contact_assist(False)
            if self._turn_started:
                self._hold_turn_command_target(geom, robot_state_data=robot_state_data)
            else:
                hold_geom = self._hold_tracking_geom(geom)
                hold_target = self._contact_compensated_ee_target_base(
                    self._ee_grasp_target_base(hold_geom),
                    hold_geom,
                    robot_state_data,
                )
                self._set_grasp_target(hold_target, hold_geom)
            if time.perf_counter() - self._state_enter_t >= self.stop_after_done_s:
                raise KeyboardInterrupt

        if self.task_state == self.FAILED:
            if self.contact_assist_release_on_done and self._contact_assist_active:
                self._set_contact_assist(False)
            if self._turn_started:
                self._hold_turn_command_target(geom, robot_state_data=robot_state_data)
            else:
                failed_target = self._contact_compensated_ee_target_base(
                    self._ee_grasp_target_base(geom),
                    geom,
                    robot_state_data,
                )
                self._set_grasp_target(failed_target, geom)
            if time.perf_counter() - self._state_enter_t >= self.stop_after_done_s:
                raise KeyboardInterrupt

    def _record_grasp_sample(self, robot_state_data):
        current, _, error, ik_error, servo_error = self._current_error(robot_state_data)
        geom = self._latest_geom or {}
        grasp_error = (
            self._physical_grasp_error(geom, robot_state_data)
            if "grasp_base" in geom
            else np.nan
        )
        valve_angle = float(geom.get("valve_angle", 0.0))
        valve_vel = float(geom.get("valve_vel", 0.0))
        valve_hold_enabled = bool(geom.get("valve_angle_hold_enabled", False))
        valve_lock_enabled = bool(geom.get("valve_angle_lock_enabled", False))
        valve_hold_tau = float(geom.get("valve_angle_hold_tau", 0.0))
        contact_count = int(geom.get("contact_count", 0))
        normal_force = float(geom.get("contact_normal_force", 0.0))
        success = self._grasp_success(geom)
        attach_enabled = bool(geom.get("attach_enabled", False))
        relative_slip = (
            self._relative_grasp_slip_m(geom, robot_state_data)
            if "grasp_base" in geom
            else np.nan
        )
        close_slip = (
            self._close_relative_slip_m(geom, robot_state_data)
            if "grasp_base" in geom
            else np.nan
        )
        relative_slip_components = (
            self._relative_grasp_slip_components_m(geom, robot_state_data)
            if "grasp_base" in geom
            else np.full(3, np.nan, dtype=float)
        )
        relative_orientation_slip_deg = self._relative_grasp_orientation_slip_deg(geom)
        raw_target_drift = self._raw_target_drift_m()
        attach_site_error = self._attach_site_error_m(geom)
        attach_site_error_world = self._attach_site_error_world(geom)
        if attach_site_error_world is None:
            attach_site_error_world = np.full(3, np.nan, dtype=float)
        attach_site_error_base = self._attach_site_error_base(geom)
        groups = self._contact_groups(geom)
        topo_code = (
            ("P" if groups["palm"] else "-")
            + ("T" if groups["thumb"] else "-")
            + ("I" if groups["index"] else "-")
            + ("M" if groups["middle"] else "-")
        )
        real_groups = self._contact_groups(geom, real_only=True)
        real_topo_code = (
            ("P" if real_groups["palm"] else "-")
            + ("T" if real_groups["thumb"] else "-")
            + ("I" if real_groups["index"] else "-")
            + ("M" if real_groups["middle"] else "-")
        )
        if self._turn_started:
            turn_actual_delta = self._turn_actual_delta_deg(geom)
            turn_reference_delta = float(self._turn_reference_delta_deg)
            turn_command_delta = float(self._turn_command_delta_deg)
            turn_angle_error = turn_reference_delta - turn_actual_delta
            turn_final_error = self.turn_target_deg - turn_actual_delta
            turn_motion_radius = self._turn_motion_radius_m()
        else:
            turn_actual_delta = 0.0
            turn_reference_delta = 0.0
            turn_command_delta = 0.0
            turn_angle_error = 0.0
            turn_final_error = self.turn_target_deg
            turn_motion_radius = np.nan

        base_roll, base_pitch, base_yaw = _base_rpy_deg(geom)
        if "base_pos_world" in geom and "center_world" in geom:
            base_to_valve = np.asarray(geom["center_world"], dtype=float) - np.asarray(
                geom["base_pos_world"], dtype=float
            )
        else:
            base_to_valve = np.full(3, np.nan, dtype=float)
        left_foot_force = float(geom.get("left_foot_force", 0.0))
        right_foot_force = float(geom.get("right_foot_force", 0.0))
        feet_contact_stable = bool(geom.get("feet_contact_stable", False))
        base_tilt_deg = float(max(abs(base_roll), abs(base_pitch)))
        if self._turn_start_base_to_valve_x_m is not None and np.isfinite(base_to_valve[0]):
            base_approach_m = float(self._turn_start_base_to_valve_x_m - base_to_valve[0])
        else:
            base_approach_m = np.nan
        if self._turn_start_base_yaw_deg is not None and np.isfinite(base_yaw):
            base_yaw_drift_deg = abs(_angle_diff_deg(base_yaw, self._turn_start_base_yaw_deg))
        else:
            base_yaw_drift_deg = np.nan
        action_norm = float(np.linalg.norm(getattr(self, "last_policy_action", np.zeros(1))))
        right_hand_q = np.asarray(geom.get("right_hand_q", [np.nan] * 7), dtype=float)
        right_hand_target_q = np.asarray(geom.get("right_hand_target_q", [np.nan] * 7), dtype=float)
        right_hand_torque = np.asarray(geom.get("right_hand_torque_cmd", [np.nan] * 7), dtype=float)
        right_hand_state = str(geom.get("right_hand_state", "unknown")).replace(",", ";")
        contact_group_counts, contact_health_score, contact_health_ratio = self._contact_health(geom)
        soft_connect_active = bool(self._contact_assist_active or attach_enabled)
        angle_brake_enabled = bool(
            self.turn_hold_angle_brake_enabled
            and valve_hold_enabled
            and self.task_state in (self.TURN_HOLD, self.DONE, self.FAILED)
        )
        angle_brake_torque = float(valve_hold_tau if angle_brake_enabled else 0.0)
        if self._turn_started:
            valve_angle_deg = float(np.rad2deg(valve_angle))
            valve_target_angle_deg = float(np.rad2deg(self._turn_start_angle) + turn_reference_delta)
            valve_angle_error_deg = float(turn_angle_error)
        else:
            valve_angle_deg = float(np.rad2deg(valve_angle))
            valve_target_angle_deg = np.nan
            valve_angle_error_deg = np.nan
        valve_vel_deg_s = float(np.rad2deg(valve_vel))

        current_turn_metric = {
            "turn_angle_error_deg": float(turn_angle_error),
            "turn_final_error_deg": float(turn_final_error),
            "right_eef_pos_error_m": float(error),
            "relative_slip_m": float(relative_slip),
            "turn_actual_delta_deg": float(turn_actual_delta),
            "grasp_success": 1.0 if success else 0.0,
            "contact_health_ratio": float(contact_health_ratio),
            "soft_connect_active": 1.0 if soft_connect_active else 0.0,
            "contact_force_n": float(geom.get("real_contact_normal_force", normal_force)),
            "valve_vel_deg_s": float(valve_vel_deg_s),
            "angle_brake_torque": float(angle_brake_torque),
            "base_tilt_deg": float(base_tilt_deg),
            "base_approach_m": float(base_approach_m),
            "base_yaw_drift_deg": float(base_yaw_drift_deg),
        }
        segment_rows_for_running_stats = list(self._turn_rows)
        if self.task_state in (self.TURN_VALVE, self.TURN_HOLD) and self._turn_started:
            segment_rows_for_running_stats.append(current_turn_metric)
        if segment_rows_for_running_stats:
            soft_connect_active_ratio = float(
                np.mean([row["soft_connect_active"] for row in segment_rows_for_running_stats])
            )
            contact_force_peak_n = float(
                np.nanmax([row["contact_force_n"] for row in segment_rows_for_running_stats])
            )
            angle_brake_peak_torque = float(
                np.nanmax(np.abs([row["angle_brake_torque"] for row in segment_rows_for_running_stats]))
            )
            max_base_tilt_deg = float(
                np.nanmax([row["base_tilt_deg"] for row in segment_rows_for_running_stats])
            )
        else:
            soft_connect_active_ratio = 1.0 if soft_connect_active else 0.0
            contact_force_peak_n = float(geom.get("real_contact_normal_force", normal_force))
            angle_brake_peak_torque = abs(angle_brake_torque)
            max_base_tilt_deg = base_tilt_deg
        if self.task_state in (self.TURN_VALVE, self.TURN_HOLD) and self._turn_started:
            self._turn_rows.append(current_turn_metric)

        segment_result = "failed" if self.task_state == self.FAILED else ("done" if self.task_state == self.DONE else "running")
        contact_pairs = "|".join(
            str(pair).replace(",", ";") for pair in geom.get("contact_pairs", [])
        )
        real_contact_pairs = "|".join(
            str(pair).replace(",", ";") for pair in geom.get("real_contact_pairs", [])
        )
        debug_contact_pairs = "|".join(
            str(pair).replace(",", ";") for pair in geom.get("debug_contact_pairs", [])
        )

        t = 0.0 if self._test_start_t is None else time.perf_counter() - self._test_start_t
        with open(self.log_file, "a") as file:
            file.write(
                f"{t:.6f},{self.task_state},"
                f"{self._current_target_base[0]:.6f},{self._current_target_base[1]:.6f},{self._current_target_base[2]:.6f},"
                f"{current[0]:.6f},{current[1]:.6f},{current[2]:.6f},"
                f"{error:.6f},{ik_error:.6f},{servo_error:.6f},{grasp_error:.6f},"
                f"{valve_angle:.6f},{valve_vel:.6f},"
                f"{contact_count},{normal_force:.6f},{topo_code},"
                f"{int(geom.get('real_contact_count', contact_count))},"
                f"{float(geom.get('real_contact_normal_force', normal_force)):.6f},"
                f"{real_topo_code},{int(success)},{int(attach_enabled)},"
                f"{relative_slip:.6f},{close_slip:.6f},"
                f"{relative_slip_components[0]:.6f},{relative_slip_components[1]:.6f},"
                f"{relative_slip_components[2]:.6f},{relative_orientation_slip_deg:.6f},"
                f"{raw_target_drift:.6f},"
                f"{attach_site_error:.6f},"
                f"{attach_site_error_world[0]:.6f},{attach_site_error_world[1]:.6f},{attach_site_error_world[2]:.6f},"
                f"{attach_site_error_base[0]:.6f},{attach_site_error_base[1]:.6f},{attach_site_error_base[2]:.6f},"
                f"{int(valve_hold_enabled)},{int(valve_lock_enabled)},{valve_hold_tau:.6f},"
                f"{self.turn_target_deg:.6f},{turn_reference_delta:.6f},{turn_command_delta:.6f},"
                f"{turn_actual_delta:.6f},{turn_angle_error:.6f},{turn_final_error:.6f},{turn_motion_radius:.6f},"
                f"{base_roll:.6f},{base_pitch:.6f},{base_yaw:.6f},"
                f"{base_to_valve[0]:.6f},{base_to_valve[1]:.6f},{base_to_valve[2]:.6f},"
                f"{left_foot_force:.6f},{right_foot_force:.6f},{int(feet_contact_stable)},"
                f"{action_norm:.6f},"
                f"{right_hand_state},"
                f"{right_hand_q[0]:.6f},{right_hand_q[1]:.6f},{right_hand_q[2]:.6f},"
                f"{right_hand_target_q[0]:.6f},{right_hand_target_q[1]:.6f},{right_hand_target_q[2]:.6f},"
                f"{right_hand_torque[0]:.6f},{right_hand_torque[1]:.6f},{right_hand_torque[2]:.6f},"
                f"{int(geom.get('right_hand_contact_hold_latched', False))},"
                f"{float(geom.get('right_hand_close_progress_thumb', np.nan)):.6f},"
                f"{float(geom.get('right_hand_close_progress_finger', np.nan)):.6f},"
                f"{float(geom.get('right_hand_target_close_progress_thumb', np.nan)):.6f},"
                f"{float(geom.get('right_hand_target_close_progress_finger', np.nan)):.6f},"
                f"{contact_pairs},{real_contact_pairs},{debug_contact_pairs},"
                f"{self.task_state},{self._turn_sequence_index + 1},"
                f"{max(1, len(self.turn_target_sequence_deg))},"
                f"{self.turn_target_deg:.6f},{self._turn_sequence_cumulative_target_deg():.6f},"
                f"{valve_angle_deg:.6f},{valve_target_angle_deg:.6f},{valve_angle_error_deg:.6f},"
                f"{valve_vel_deg_s:.6f},{error:.6f},unavailable,"
                f"{int(geom.get('real_contact_count', contact_count))},"
                f"{contact_group_counts['thumb']},{contact_group_counts['index']},"
                f"{contact_group_counts['middle']},{contact_health_score:.6f},"
                f"{contact_health_ratio:.6f},{int(soft_connect_active)},"
                f"{soft_connect_active_ratio:.6f},"
                f"{float(geom.get('real_contact_normal_force', normal_force)):.6f},"
                f"{contact_force_peak_n:.6f},{int(angle_brake_enabled)},"
                f"{angle_brake_torque:.6f},{angle_brake_peak_torque:.6f},"
                f"{base_approach_m:.6f},{base_yaw_drift_deg:.6f},"
                f"{base_tilt_deg:.6f},{max_base_tilt_deg:.6f},"
                f"{segment_result},{_csv_token(self._failure_reason)},"
                f"{_csv_token(self._abort_reason)}\n"
            )
        self._write_grasp_marker_file(
            self._current_target_base,
            current,
            error,
            geom,
            contact_count,
            normal_force,
            success,
            topo_code,
        )

    def _write_grasp_marker_file(self, target, current, error, geom, contact_count, normal_force, success, topo_code):
        payload = {
            "target_right_base": np.asarray(target, dtype=float).reshape(3).tolist(),
            "current_right_base": np.asarray(current, dtype=float).reshape(3).tolist(),
            "error_m": float(error),
            "target_id": int(self._target_id),
            "timestamp": time.time(),
        }
        if geom and "center_base" in geom and "grasp_base" in geom:
            center = np.asarray(geom["center_base"], dtype=float).reshape(3)
            grasp = self._valve_grasp_point_base(geom)
            R = self._right_grasp_rotation_base(geom)
            command_grasp = (
                self._current_command_grasp_base
                if self._current_command_grasp_base is not None
                else np.asarray(target) + R @ self.grasp_point_offset_ee
            )
            payload.update(
                {
                    "valve_center_base": center.tolist(),
                    "valve_actual_grasp_base": grasp.tolist(),
                    "valve_command_target_base": np.asarray(command_grasp, dtype=float).reshape(3).tolist(),
                    "right_grasp_thumb_tip_target_base": (
                        np.asarray(target) + R @ self.grasp_tip_offsets_ee["thumb"]
                    ).tolist(),
                    "right_grasp_index_tip_target_base": (
                        np.asarray(target) + R @ self.grasp_tip_offsets_ee["index"]
                    ).tolist(),
                    "right_grasp_middle_tip_target_base": (
                        np.asarray(target) + R @ self.grasp_tip_offsets_ee["middle"]
                    ).tolist(),
                    "valve_label_base": (grasp + np.array([0.0, 0.0, 0.12], dtype=float)).tolist(),
                    "valve_angle_label": (
                        f"contact {contact_count} | F {normal_force:.1f} N | "
                        f"{topo_code} | {'grasp ok' if success else 'grasp wait'}"
                    ),
                }
            )
            if self._turn_started:
                actual_delta = self._turn_actual_delta_deg(geom)
                absolute_delta = self._turn_absolute_delta_deg(geom)
                cumulative_target = self._turn_sequence_cumulative_target_deg()
                final_target_base = self._turn_point_base_for_delta(geom, self.turn_target_deg)
                command_target_base = self._turn_point_base_for_delta(geom, self._turn_cmd_deg)
                payload.update(
                    {
                        "valve_arc_points_base": self._turn_arc_points_base(geom, self._turn_cmd_deg),
                        "valve_desired_delta_deg": float(self._turn_reference_delta_deg),
                        "valve_command_delta_deg": float(self._turn_cmd_deg),
                        "valve_actual_delta_deg": float(actual_delta),
                        "valve_segment_target_delta_deg": float(self.turn_target_deg),
                        "valve_cumulative_target_delta_deg": float(cumulative_target),
                        "valve_absolute_delta_deg": float(absolute_delta),
                        "valve_label_base": (center + np.array([0.0, 0.0, 0.22], dtype=float)).tolist(),
                        "valve_angle_label": (
                            f"seg {self._turn_sequence_index + 1}/"
                            f"{max(1, len(self.turn_target_sequence_deg))} "
                            f"target {self.turn_target_deg:+.1f} deg | "
                            f"ref {self._turn_reference_delta_deg:+.1f} | "
                            f"cmd {self._turn_cmd_deg:+.1f} | actual {actual_delta:+.1f} | "
                            f"sum {absolute_delta:+.1f}/{cumulative_target:+.1f} | "
                            f"{topo_code} F {normal_force:.1f}N"
                        ),
                    }
                )
                if final_target_base is not None:
                    payload["valve_final_target_base"] = final_target_base.tolist()
                if command_target_base is not None:
                    payload["valve_command_target_base"] = command_target_base.tolist()
            axis_len = 0.08
            payload.update(
                {
                    "right_grasp_x_axis_end_base": (np.asarray(target) + axis_len * R[:, 0]).tolist(),
                    "right_grasp_y_axis_end_base": (np.asarray(target) + axis_len * R[:, 1]).tolist(),
                    "right_grasp_z_axis_end_base": (np.asarray(target) + axis_len * R[:, 2]).tolist(),
                }
            )

        tmp_file = f"{self.marker_file}.tmp"
        with open(tmp_file, "w") as file:
            json.dump(payload, file)
        os.replace(tmp_file, self.marker_file)

    def _preflight_status(self, status, wall_elapsed):
        if status is None:
            return False, "no fresh sim status", {}
        if wall_elapsed < self.preflight_min_wait_s:
            return False, "waiting minimum startup time", {}

        base_pos = np.asarray(status.get("base_pos_world", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        center = np.asarray(status.get("valve_center_world", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        base_to_valve = center - base_pos
        roll, pitch, yaw = _rpy_deg_from_xmat(status.get("base_xmat_world", np.eye(3)))
        base_tilt = max(abs(float(roll)), abs(float(pitch)))
        left_foot = float(status.get("left_foot_force", 0.0))
        right_foot = float(status.get("right_foot_force", 0.0))
        total_foot = left_foot + right_foot

        z_min, z_max = self.preflight_base_z_range
        dx_min, dx_max = self.preflight_center_dx_range
        checks = [
            (np.all(np.isfinite(base_pos)), "base pose not finite"),
            (base_tilt <= self.preflight_max_base_tilt_deg, f"base tilt {base_tilt:.1f}deg too large"),
            (z_min <= base_pos[2] <= z_max, f"base z {base_pos[2]:.3f}m out of range"),
            (dx_min <= base_to_valve[0] <= dx_max, f"valve dx {base_to_valve[0]:.3f}m out of range"),
            (abs(base_to_valve[1]) <= self.preflight_center_abs_y_max, f"valve dy {base_to_valve[1]:.3f}m too large"),
            (total_foot >= self.preflight_min_total_foot_force, f"total foot force {total_foot:.1f}N too small"),
        ]
        metrics = {
            "tilt": base_tilt,
            "yaw": float(yaw),
            "base_z": float(base_pos[2]),
            "dx": float(base_to_valve[0]),
            "dy": float(base_to_valve[1]),
            "total_foot": total_foot,
        }
        for ok, reason in checks:
            if not ok:
                return False, reason, metrics
        return True, "ok", metrics

    def _preflight_ready_for_policy(self, wall_elapsed):
        if not self.preflight_enabled:
            return True

        status = self._read_sim_status()
        ok, reason, metrics = self._preflight_status(status, wall_elapsed)
        now = time.perf_counter()
        if ok:
            if self._preflight_ok_started_t is None:
                self._preflight_ok_started_t = now
                self.logger.info(colored("[VALVE_GRASP] preflight candidate is valid", "cyan"))
            if now - self._preflight_ok_started_t >= self.preflight_required_stable_s:
                self.logger.info(
                    colored(
                        "[VALVE_GRASP] preflight passed: "
                        f"tilt={metrics['tilt']:.1f}deg, "
                        f"dx={metrics['dx']:.3f}m, dy={metrics['dy']:.3f}m, "
                        f"base_z={metrics['base_z']:.3f}m, foot_total={metrics['total_foot']:.1f}N",
                        "green",
                    )
                )
                return True
        else:
            self._preflight_ok_started_t = None

        if wall_elapsed >= self._next_start_log_t:
            self._next_start_log_t = wall_elapsed + 1.0
            metric_text = ""
            if metrics:
                metric_text = (
                    f" tilt={metrics['tilt']:.1f}deg"
                    f" dx={metrics['dx']:.3f}m dy={metrics['dy']:.3f}m"
                    f" z={metrics['base_z']:.3f}m foot={metrics['total_foot']:.1f}N"
                )
            self.logger.info(colored(f"[VALVE_GRASP] waiting preflight: {reason}{metric_text}", "yellow"))

        if self.preflight_max_wait_s > 0.0 and wall_elapsed >= self.preflight_max_wait_s:
            self.logger.warning(colored(f"[VALVE_GRASP] bad initial pose: {reason}", "yellow"))
            raise KeyboardInterrupt
        return False

    def policy_action(self):
        wall_elapsed = time.perf_counter() - self._wall_start_t
        if self.duration_s is not None and wall_elapsed >= self.duration_s:
            raise KeyboardInterrupt

        robot_state_data = self.state_processor.robot_state_data
        if robot_state_data is None:
            if wall_elapsed >= self._next_wait_log_t:
                self._next_wait_log_t = wall_elapsed + 2.0
                self.logger.info(colored("[VALVE_GRASP] waiting for sim lowstate...", "yellow"))
            return

        if self.auto_workflow:
            ready = self._auto_workflow_ready_for_tracking(wall_elapsed)
            if ready and self._test_start_t is None:
                self._test_start_t = time.perf_counter()
            if ready:
                self._update_task_state(robot_state_data)
            self._send_7dof_policy_action()
            if ready:
                self._record_grasp_sample(robot_state_data)
            return

        if self.manual_start and not self.use_policy_action:
            if wall_elapsed >= self._next_start_log_t:
                self._next_start_log_t = wall_elapsed + 2.0
                self.logger.info(colored("[VALVE_GRASP] waiting for manual ] policy start...", "yellow"))
            return

        if self.manual_start and not self._started_policy:
            self._started_policy = True
            self._policy_start_wall_t = time.perf_counter()

        if not self.manual_start and not self._started_policy:
            if not self._preflight_ready_for_policy(wall_elapsed):
                return
            self._handle_start_policy()
            self._started_policy = True
            self._policy_start_wall_t = time.perf_counter()
            self._write_control_file(
                {
                    "policy_started": True,
                    "timestamp": time.time(),
                    "policy_start_wall_time": self._policy_start_wall_t,
                    "disable_elastic_after_policy_start_s": self.elastic_release_delay_after_policy_start_s,
                }
            )

        if self._test_start_t is None:
            self._test_start_t = time.perf_counter()
        if (
            self._policy_start_wall_t is not None
            and time.perf_counter() - self._policy_start_wall_t < self.tracking_start_delay_s
        ):
            self._send_7dof_policy_action()
            return
        self._update_task_state(robot_state_data)
        self._send_7dof_policy_action()
        self._record_grasp_sample(robot_state_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7-DoF with-hand true-contact valve grasp baseline")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--marker_file", type=str, default="/tmp/falcon_valve_grasp_contact_markers.json")
    parser.add_argument("--log_file", type=str, default="/tmp/falcon_valve_grasp_contact_metrics.csv")
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument("--auto_start_policy", action="store_true")
    parser.add_argument("--auto_workflow", action="store_true")
    parser.add_argument("--sim_status_file", type=str, default="/tmp/falcon_valve_grasp_contact_status.json")
    parser.add_argument("--sim_control_file", type=str, default="/tmp/falcon_valve_grasp_contact_control.json")
    parser.add_argument("--status_timeout_s", type=float, default=2.0)
    parser.add_argument("--tracking_delay_after_elastic_off_s", type=float, default=0.5)
    parser.add_argument("--tracking_start_delay_s", type=float, default=0.0)
    parser.add_argument("--elastic_release_delay_after_policy_start_s", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    apply_named_motor_gain_scales(config)

    config["disable_keyboard_listener"] = bool(args.auto_start_policy or args.auto_workflow)
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path or config model_path")

    policy = LocoManipValveGraspContact7DofWithHandPolicy(
        config,
        model_path,
        marker_file=args.marker_file,
        log_file=args.log_file,
        sample_period_s=999.0,
        seed=1,
        duration_s=args.duration_s,
        manual_start=not (args.auto_start_policy or args.auto_workflow),
        auto_workflow=args.auto_workflow,
        sim_status_file=args.sim_status_file,
        sim_control_file=args.sim_control_file,
        status_timeout_s=args.status_timeout_s,
        tracking_delay_after_elastic_off_s=args.tracking_delay_after_elastic_off_s,
        tracking_start_delay_s=args.tracking_start_delay_s,
        elastic_release_delay_after_policy_start_s=args.elastic_release_delay_after_policy_start_s,
    )
    policy.run()
