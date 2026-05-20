import argparse
import json
import os
import sys
import time

import numpy as np
import pinocchio as pin
import yaml
from termcolor import colored

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy


def _parse_float_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def apply_named_motor_gain_scales(config):
    """Apply optional named motor gain scales before Robot/CommandSender init."""
    dof_names = list(config.get("dof_names", []))
    for config_key, gain_key in (
        ("tracking_motor_kp_scale_by_name", "MOTOR_KP"),
        ("tracking_motor_kd_scale_by_name", "MOTOR_KD"),
    ):
        scales = config.get(config_key, {}) or {}
        if not scales:
            continue
        gains = list(config[gain_key])
        for joint_name, scale in scales.items():
            if joint_name not in dof_names:
                raise ValueError(f"{config_key} references unknown joint: {joint_name}")
            gains[dof_names.index(joint_name)] = float(gains[dof_names.index(joint_name)]) * float(scale)
        config[gain_key] = gains


class LocoManipEETrackingTestPolicy(LocoManipPolicy):
    """右手末端位置跟踪测试。

    目标点、当前末端点和误差都定义在机器人 base 坐标系下，方便和后续
    阀门轨迹任务共用同一套末端定义。
    """

    def __init__(
        self,
        config,
        model_path,
        rl_rate=50,
        policy_action_scale=0.25,
        marker_file="/tmp/falcon_ee_tracking_markers.json",
        log_file="/tmp/falcon_ee_tracking_metrics.csv",
        sample_period_s=5.0,
        x_range=(0.22, 0.38),
        y_range=(-0.22, -0.08),
        z_range=(0.00, 0.18),
        seed=1,
        duration_s=None,
        manual_start=True,
        auto_workflow=False,
        sim_status_file="/tmp/falcon_sim_status.json",
        sim_control_file="/tmp/falcon_sim_control.json",
        status_timeout_s=2.0,
        tracking_delay_after_elastic_off_s=0.5,
        tracking_start_delay_s=0.0,
        elastic_release_delay_after_policy_start_s=None,
        safe_start_init_s=5.0,
        safe_start_blend_s=3.0,
        safe_start_target_hold_s=2.0,
        safe_start_use_init=False,
        max_joint_rate_rad_s=0.4,
        policy_delta_warn_rad=0.4,
        policy_delta_abort_rad=1.0,
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        self.marker_file = marker_file
        self.log_file = log_file
        self.sample_period_s = float(sample_period_s)
        self.ranges = {
            "x": tuple(float(v) for v in x_range),
            "y": tuple(float(v) for v in y_range),
            "z": tuple(float(v) for v in z_range),
        }
        self.rng = np.random.default_rng(seed)

        self.disable_keyboard_listener = bool(config.get("disable_keyboard_listener", True))
        self._started_policy = False
        self._test_start_t = None
        self._wall_start_t = time.perf_counter()
        self.duration_s = None if duration_s is None else float(duration_s)
        self.manual_start = bool(manual_start)
        self.auto_workflow = bool(auto_workflow)
        self.sim_status_file = sim_status_file
        self.sim_control_file = sim_control_file
        self.status_timeout_s = float(status_timeout_s)
        self.tracking_delay_after_elastic_off_s = float(tracking_delay_after_elastic_off_s)
        self.tracking_start_delay_s = float(tracking_start_delay_s)
        self.elastic_release_delay_after_policy_start_s = (
            None
            if elastic_release_delay_after_policy_start_s is None
            else float(elastic_release_delay_after_policy_start_s)
        )
        self.safe_start_init_s = max(0.0, float(safe_start_init_s))
        self.safe_start_blend_s = max(0.0, float(safe_start_blend_s))
        self.safe_start_target_hold_s = max(0.0, float(safe_start_target_hold_s))
        self.safe_start_use_init = bool(safe_start_use_init)
        self.max_joint_rate_rad_s = max(0.0, float(max_joint_rate_rad_s))
        self.policy_delta_warn_rad = max(0.0, float(policy_delta_warn_rad))
        self.policy_delta_abort_rad = max(0.0, float(policy_delta_abort_rad))
        self.safe_start_enabled = (
            bool(config.get("enable_safe_start", True))
            and not self.auto_workflow
            and (self.safe_start_init_s > 0.0 or self.safe_start_blend_s > 0.0)
        )
        self.safe_start_keys = {"]", "u"}
        self._safe_start_active = False
        self._safe_start_t = None
        self._safe_start_blend_t = None
        self._safe_start_blend_from_q = None
        self._safe_start_policy_delta_checked = False
        self._next_safe_start_log_t = 0.0
        self._last_sent_q_target = None
        self._last_sent_q_target_t = None
        self._policy_start_wall_t = None
        self._elastic_off_wall_t = None
        self._tracking_enabled = not self.auto_workflow
        self._next_wait_log_t = 0.0
        self._next_start_log_t = 0.0
        self._last_sample_t = None
        self._target_id = -1
        self._window_errors = []
        self._window_ik_errors = []
        self._window_servo_errors = []
        self._window_current_speeds = []
        self._last_record_perf_t = None
        self._last_current_right_ee = None
        self._last_upper_body_qpos = None
        self._last_upper_body_tauff = None

        # 4DoF 版本只取肩肘关节；7DoF 子类会覆盖为包含手腕的完整上肢链。
        self.arm_reduced_joint_indices = [0, 1, 2, 3, 7, 8, 9, 10]
        self.full_arm_joint_indices = [
            self.dof_names.index("left_shoulder_pitch_joint"),
            self.dof_names.index("left_shoulder_roll_joint"),
            self.dof_names.index("left_shoulder_yaw_joint"),
            self.dof_names.index("left_elbow_joint"),
            self.dof_names.index("right_shoulder_pitch_joint"),
            self.dof_names.index("right_shoulder_roll_joint"),
            self.dof_names.index("right_shoulder_yaw_joint"),
            self.dof_names.index("right_elbow_joint"),
        ]
        self.right_ee_frame_id = self.upper_body_controller.reduced_robot.model.getFrameId("R_ee")

        marker_dir = os.path.dirname(self.marker_file)
        log_dir = os.path.dirname(self.log_file)
        if marker_dir:
            os.makedirs(marker_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as file:
            file.write(
                "time_s,target_id,target_x,target_y,target_z,"
                "current_x,current_y,current_z,error_m,"
                "target_age_s,current_speed_mps,"
                "ik_x,ik_y,ik_z,ik_error_m,servo_error_m\n"
            )
        if self.sim_control_file:
            self._write_control_file({"policy_started": False, "timestamp": time.time()})

    def handle_keyboard_button(self, keycode):
        if keycode in self.safe_start_keys and self.manual_start and self.safe_start_enabled:
            self._begin_safe_start()
            return
        if keycode == "o":
            self._reset_safe_start()
        super().handle_keyboard_button(keycode)

    def _set_command_sender_active(self):
        if hasattr(self.command_sender, "no_action"):
            self.command_sender.no_action = 0

    def _reset_safe_start(self):
        self._safe_start_active = False
        self._safe_start_t = None
        self._safe_start_blend_t = None
        self._safe_start_blend_from_q = None
        self._safe_start_policy_delta_checked = False
        self._next_safe_start_log_t = 0.0

    def _begin_safe_start(self):
        if self._safe_start_active:
            return
        if self._started_policy or self.use_policy_action:
            self.logger.info(colored("[EE_TRACK] policy is already active", "yellow"))
            return
        self._safe_start_active = True
        self._safe_start_t = time.perf_counter()
        self._safe_start_blend_t = None
        self._safe_start_blend_from_q = None
        self._safe_start_policy_delta_checked = False
        self._next_safe_start_log_t = 0.0
        self._started_policy = False
        self._policy_start_wall_t = None
        self.use_policy_action = False
        self.get_ready_state = self.safe_start_use_init
        self.init_count = 0
        self._set_command_sender_active()
        init_mode = "default init" if self.safe_start_use_init else "hold current"
        self.logger.info(
            colored(
                "[EE_TRACK] safe start: "
                f"{init_mode} {self.safe_start_init_s:.1f}s, "
                f"policy blend {self.safe_start_blend_s:.1f}s",
                "green",
            )
        )

    def _compute_loco_manip_q_target(self, robot_state_data, mode):
        if self.upper_body_controller:
            upper_body_qpos, _ = self.upper_body_controller.get_q_tau(
                self.waypoints_left[0],
                self.waypoints_right[0],
                self.EE_efrc_L,
                self.EE_efrc_R,
            )
            self._last_upper_body_qpos = np.asarray(upper_body_qpos, dtype=float).reshape(-1)
            arm_reduced_joint_indices = [0, 1, 2, 3, 7, 8, 9, 10]
            for i, idx in enumerate(arm_reduced_joint_indices):
                self.ref_upper_dof_pos[0, idx] = upper_body_qpos[i]
            wrist_joint_indices = [19, 20, 21, 26, 27, 28]
            for idx in wrist_joint_indices:
                self.ref_upper_dof_pos[0, idx - 15] = 0.0

        scaled_policy_action = self.rl_inference(robot_state_data)
        if mode == "init":
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif mode == "hold":
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        elif mode == "policy":
            q_target = scaled_policy_action + self.default_dof_angles
        else:
            raise ValueError(f"Unsupported q target mode: {mode}")

        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)
        return q_target[0]

    def _send_q_target(self, robot_state_data, q_target):
        q_target = self._limit_q_target_step(np.asarray(q_target, dtype=float).copy())
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        self.command_sender.send_command(
            q_target,
            cmd_dq,
            cmd_tau,
            robot_state_data[0, 7 : 7 + self.num_dofs],
        )

    def _limit_q_target_step(self, q_target):
        now = time.perf_counter()
        if (
            self.max_joint_rate_rad_s <= 0.0
            or self._last_sent_q_target is None
            or self._last_sent_q_target_t is None
        ):
            self._last_sent_q_target = q_target.copy()
            self._last_sent_q_target_t = now
            return q_target

        dt = max(1e-3, min(now - self._last_sent_q_target_t, 0.1))
        max_step = self.max_joint_rate_rad_s * dt
        step = np.clip(q_target - self._last_sent_q_target, -max_step, max_step)
        limited = self._last_sent_q_target + step
        self._last_sent_q_target = limited.copy()
        self._last_sent_q_target_t = now
        return limited

    def _format_top_joint_deltas(self, delta, top_k=8):
        order = np.argsort(np.abs(delta))[::-1][:top_k]
        parts = []
        for idx in order:
            name = self.dof_names[idx] if self.dof_names and idx < len(self.dof_names) else str(idx)
            parts.append(f"{name}={delta[idx]:+.3f}")
        return ", ".join(parts)

    def _inspect_policy_delta(self, robot_state_data, q_policy):
        current_q = robot_state_data[0, 7 : 7 + self.num_dofs]
        delta = np.asarray(q_policy, dtype=float) - current_q
        max_abs = float(np.max(np.abs(delta)))
        top = self._format_top_joint_deltas(delta)
        level = "red" if self.policy_delta_abort_rad > 0.0 and max_abs > self.policy_delta_abort_rad else "yellow"
        self.logger.info(
            colored(
                f"[EE_TRACK] policy-current max_delta={max_abs:.3f}rad top: {top}",
                level,
            )
        )
        if self.policy_delta_abort_rad > 0.0 and max_abs > self.policy_delta_abort_rad:
            self.logger.error(
                colored(
                    f"[EE_TRACK] abort policy blend: max_delta {max_abs:.3f}rad "
                    f"> abort {self.policy_delta_abort_rad:.3f}rad",
                    "red",
                )
            )
            return False
        if self.policy_delta_warn_rad > 0.0 and max_abs > self.policy_delta_warn_rad:
            self.logger.warning(
                colored(
                    f"[EE_TRACK] large policy delta {max_abs:.3f}rad "
                    f"> warn {self.policy_delta_warn_rad:.3f}rad; rate limiter remains active",
                    "yellow",
                )
            )
        return True

    def _abort_safe_start_to_hold(self, robot_state_data):
        current_q = robot_state_data[0, 7 : 7 + self.num_dofs].copy()
        self._send_q_target(robot_state_data, current_q)
        self._reset_safe_start()
        self.use_policy_action = False
        self.get_ready_state = False
        if hasattr(self.command_sender, "no_action"):
            self.command_sender.no_action = 1

    def _send_loco_manip_policy_action(self, robot_state_data):
        if self.get_ready_state:
            mode = "init"
        elif not self.use_policy_action:
            mode = "hold"
        else:
            mode = "policy"
        q_target = self._compute_loco_manip_q_target(robot_state_data, mode)
        self._send_q_target(robot_state_data, q_target)

    def _prime_tracking_after_safe_start(self, now):
        self._test_start_t = now
        hold_s = min(self.safe_start_target_hold_s, self.sample_period_s)
        self._last_sample_t = now - max(0.0, self.sample_period_s - hold_s)
        self._target_id = 0
        self._window_errors = []
        self._window_ik_errors = []
        self._window_servo_errors = []
        self._window_current_speeds = []
        self._last_record_perf_t = None
        self._last_current_right_ee = None
        target = np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z], dtype=float)
        self.logger.info(
            colored(
                f"[EE_TRACK] safe start complete; hold initial target "
                f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) "
                f"for {hold_s:.1f}s before sampling",
                "green",
            )
        )

    def _run_safe_start(self, robot_state_data):
        now = time.perf_counter()
        elapsed = now - self._safe_start_t

        if elapsed < self.safe_start_init_s:
            self.get_ready_state = self.safe_start_use_init
            self.use_policy_action = False
            q_mode = "init" if self.safe_start_use_init else "hold"
            q_target = self._compute_loco_manip_q_target(robot_state_data, q_mode)
            self._send_q_target(robot_state_data, q_target)
            if now >= self._next_safe_start_log_t:
                self._next_safe_start_log_t = now + 1.0
                init_mode = "default init" if self.safe_start_use_init else "hold current"
                self.logger.info(
                    colored(
                        f"[EE_TRACK] safe start {init_mode} "
                        f"{elapsed:.1f}/{self.safe_start_init_s:.1f}s",
                        "yellow",
                    )
                )
            return

        if self._safe_start_blend_t is None:
            self._safe_start_blend_t = now
            self._safe_start_blend_from_q = robot_state_data[0, 7 : 7 + self.num_dofs].copy()
            self.get_ready_state = False
            self.use_policy_action = True
            self._set_command_sender_active()
            self.logger.info(colored("[EE_TRACK] safe start blending into policy", "yellow"))

        q_policy = self._compute_loco_manip_q_target(robot_state_data, "policy")
        if not self._safe_start_policy_delta_checked:
            self._safe_start_policy_delta_checked = True
            if not self._inspect_policy_delta(robot_state_data, q_policy):
                self._abort_safe_start_to_hold(robot_state_data)
                return

        blend_elapsed = now - self._safe_start_blend_t
        if self.safe_start_blend_s > 0.0:
            alpha = min(1.0, max(0.0, blend_elapsed / self.safe_start_blend_s))
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            q_target = (1.0 - smooth_alpha) * self._safe_start_blend_from_q + smooth_alpha * q_policy
        else:
            alpha = 1.0
            q_target = q_policy

        self._send_q_target(robot_state_data, q_target)
        if now >= self._next_safe_start_log_t and alpha < 1.0:
            self._next_safe_start_log_t = now + 1.0
            self.logger.info(
                colored(
                    f"[EE_TRACK] safe start policy blend "
                    f"{blend_elapsed:.1f}/{self.safe_start_blend_s:.1f}s",
                    "yellow",
                )
            )

        if alpha >= 1.0:
            self._safe_start_active = False
            self._started_policy = True
            self._policy_start_wall_t = now
            self.get_ready_state = False
            self.use_policy_action = True
            self._prime_tracking_after_safe_start(now)
            self._write_control_file(
                {
                    "policy_started": True,
                    "timestamp": time.time(),
                    "policy_start_wall_time": self._policy_start_wall_t,
                    "disable_elastic_after_policy_start_s": self.elastic_release_delay_after_policy_start_s,
                    "safe_start_init_s": self.safe_start_init_s,
                    "safe_start_blend_s": self.safe_start_blend_s,
                    "safe_start_use_init": self.safe_start_use_init,
                    "max_joint_rate_rad_s": self.max_joint_rate_rad_s,
                    "policy_delta_abort_rad": self.policy_delta_abort_rad,
                }
            )

    def _read_sim_status(self):
        if not self.sim_status_file or not os.path.exists(self.sim_status_file):
            return None
        try:
            with open(self.sim_status_file, "r") as file:
                status = json.load(file)
        except Exception:
            return None
        if time.time() - float(status.get("timestamp", 0.0)) > self.status_timeout_s:
            return None
        return status

    def _write_control_file(self, payload):
        if not self.sim_control_file:
            return
        control_dir = os.path.dirname(self.sim_control_file)
        if control_dir:
            os.makedirs(control_dir, exist_ok=True)
        merged = {}
        if os.path.exists(self.sim_control_file):
            try:
                with open(self.sim_control_file, "r") as file:
                    merged = json.load(file)
            except Exception:
                merged = {}
        merged.update(payload)
        tmp_file = f"{self.sim_control_file}.tmp"
        with open(tmp_file, "w") as file:
            json.dump(merged, file)
        os.replace(tmp_file, self.sim_control_file)

    def _sample_right_target(self):
        return np.array(
            [
                self.rng.uniform(*self.ranges["x"]),
                self.rng.uniform(*self.ranges["y"]),
                self.rng.uniform(*self.ranges["z"]),
            ],
            dtype=float,
        )

    def _set_right_target(self, target):
        self.EE_right_x = float(target[0])
        self.EE_right_y = float(target[1])
        self.EE_right_z = float(target[2])
        self.update_waypoints()

    def _summarize_window(self):
        # 每个目标点保持一段时间后汇总一次，避免只看瞬时误差误判跟踪质量。
        if not self._window_errors:
            return
        errors = np.asarray(self._window_errors, dtype=float)
        ik_errors = np.asarray(self._window_ik_errors, dtype=float)
        servo_errors = np.asarray(self._window_servo_errors, dtype=float)
        speeds = np.asarray(self._window_current_speeds, dtype=float)
        def finite_mean(values):
            values = values[np.isfinite(values)]
            return float(np.mean(values)) if values.size else float("nan")
        def finite_percentile(values, percentile):
            values = values[np.isfinite(values)]
            return float(np.percentile(values, percentile)) if values.size else float("nan")
        self.logger.info(
            colored(
                f"[EE_TRACK] target={self._target_id} "
                f"mean={errors.mean():.4f}m rms={np.sqrt(np.mean(errors ** 2)):.4f}m "
                f"max={errors.max():.4f}m final={errors[-1]:.4f}m "
                f"ik_mean={finite_mean(ik_errors):.4f}m servo_mean={finite_mean(servo_errors):.4f}m "
                f"speed_p95={finite_percentile(speeds, 95):.3f}m/s",
                "cyan",
            )
        )
        self._window_errors = []
        self._window_ik_errors = []
        self._window_servo_errors = []
        self._window_current_speeds = []

    def _maybe_resample_target(self):
        now = time.perf_counter()
        if self._test_start_t is None:
            self._test_start_t = now
            self._last_sample_t = now - self.sample_period_s

        if now - self._last_sample_t < self.sample_period_s:
            return

        self._summarize_window()
        self._target_id += 1
        target = self._sample_right_target()
        self._set_right_target(target)
        self._last_sample_t = now
        self.logger.info(
            colored(
                f"[EE_TRACK] new target={self._target_id} "
                f"right_base=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})",
                "green",
            )
        )

    def _current_right_fake_ee_base(self, robot_state_data):
        # 这里的 fake_ee 是 Pinocchio 中的 R_ee 代理点，不一定等同于真实手掌接触点。
        full_q = robot_state_data[0, 7 : 7 + self.num_dofs]
        reduced_q = full_q[self.full_arm_joint_indices]
        return self._right_ee_from_upper_q(reduced_q)

    def _right_ee_from_upper_q(self, upper_q):
        model = self.upper_body_controller.reduced_robot.model
        data = self.upper_body_controller.data
        pin.framesForwardKinematics(model, data, np.asarray(upper_q, dtype=float).reshape(-1))
        pin.updateFramePlacements(model, data)
        return np.array(data.oMf[self.right_ee_frame_id].translation, dtype=float).reshape(3)

    def _last_commanded_right_ee_base(self):
        if self._last_upper_body_qpos is None:
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        try:
            return self._right_ee_from_upper_q(self._last_upper_body_qpos)
        except Exception:
            return np.array([np.nan, np.nan, np.nan], dtype=float)

    def _write_marker_file(self, target, current, error):
        payload = {
            "target_right_base": target.tolist(),
            "current_right_base": current.tolist(),
            "error_m": float(error),
            "target_id": int(self._target_id),
            "timestamp": time.time(),
        }
        tmp_file = f"{self.marker_file}.tmp"
        with open(tmp_file, "w") as file:
            json.dump(payload, file)
        os.replace(tmp_file, self.marker_file)

    def _record_error(self, robot_state_data):
        target = np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z], dtype=float)
        current = self._current_right_fake_ee_base(robot_state_data)
        commanded = self._last_commanded_right_ee_base()
        # error 是最终可见误差；ik_error 和 servo_error 分别用于区分 IK 解算误差和关节执行误差。
        error = float(np.linalg.norm(current - target))
        ik_error = float(np.linalg.norm(commanded - target)) if np.all(np.isfinite(commanded)) else np.nan
        servo_error = float(np.linalg.norm(current - commanded)) if np.all(np.isfinite(commanded)) else np.nan

        now = time.perf_counter()
        if self._last_record_perf_t is None or self._last_current_right_ee is None:
            current_speed = np.nan
        else:
            dt = max(now - self._last_record_perf_t, 1e-6)
            current_speed = float(np.linalg.norm(current - self._last_current_right_ee) / dt)
        self._last_record_perf_t = now
        self._last_current_right_ee = current.copy()
        target_age_s = 0.0 if self._last_sample_t is None else now - self._last_sample_t

        self._window_errors.append(error)
        self._window_ik_errors.append(ik_error)
        self._window_servo_errors.append(servo_error)
        self._window_current_speeds.append(current_speed)

        t = 0.0 if self._test_start_t is None else time.perf_counter() - self._test_start_t
        with open(self.log_file, "a") as file:
            file.write(
                f"{t:.6f},{self._target_id},"
                f"{target[0]:.6f},{target[1]:.6f},{target[2]:.6f},"
                f"{current[0]:.6f},{current[1]:.6f},{current[2]:.6f},"
                f"{error:.6f},{target_age_s:.6f},{current_speed:.6f},"
                f"{commanded[0]:.6f},{commanded[1]:.6f},{commanded[2]:.6f},"
                f"{ik_error:.6f},{servo_error:.6f}\n"
            )
        self._write_marker_file(target, current, error)

    def _start_policy_for_auto_workflow(self):
        if self._started_policy:
            return
        # 自动流程等价于手动按下 ]：先让 policy 接管，再由 sim 端按控制文件延迟松绳。
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
        self.logger.info(colored("[EE_TRACK] auto workflow: policy started", "green"))

    def _auto_workflow_ready_for_tracking(self, wall_elapsed):
        status = self._read_sim_status()
        if status is None:
            if wall_elapsed >= self._next_wait_log_t:
                self._next_wait_log_t = wall_elapsed + 2.0
                self.logger.info(colored("[EE_TRACK] waiting for fresh sim status...", "yellow"))
            return False

        if not self._started_policy:
            if status.get("feet_contact_stable", False):
                self._start_policy_for_auto_workflow()
            elif wall_elapsed >= self._next_start_log_t:
                self._next_start_log_t = wall_elapsed + 2.0
                self.logger.info(
                    colored(
                        "[EE_TRACK] waiting for stable foot contact "
                        f"L={status.get('left_foot_force', 0.0):.1f}N "
                        f"R={status.get('right_foot_force', 0.0):.1f}N "
                        f"elastic_len={status.get('elastic_length', 0.0):.3f}",
                        "yellow",
                    )
                )
            return False

        # 松绳后再开始采样目标，避免把吊绳释放瞬态算进末端跟踪误差。
        if self._tracking_enabled:
            return True

        if not status.get("elastic_enabled", True):
            if self._elastic_off_wall_t is None:
                self._elastic_off_wall_t = time.perf_counter()
                self.logger.info(colored("[EE_TRACK] auto workflow: elastic band is off", "green"))
            if time.perf_counter() - self._elastic_off_wall_t >= self.tracking_delay_after_elastic_off_s:
                self._tracking_enabled = True
                self._test_start_t = None
                self._last_sample_t = None
                self.logger.info(colored("[EE_TRACK] auto workflow: start random target sampling", "green"))
                return True
        elif wall_elapsed >= self._next_start_log_t:
            self._next_start_log_t = wall_elapsed + 2.0
            self.logger.info(colored("[EE_TRACK] waiting for elastic band release...", "yellow"))

        return False

    def policy_action(self):
        wall_elapsed = time.perf_counter() - self._wall_start_t
        if self.duration_s is not None and wall_elapsed >= self.duration_s:
            raise KeyboardInterrupt

        robot_state_data = self.state_processor.robot_state_data
        if robot_state_data is None:
            if wall_elapsed >= self._next_wait_log_t:
                self._next_wait_log_t = wall_elapsed + 2.0
                self.logger.info(colored("[EE_TRACK] waiting for sim lowstate...", "yellow"))
            return

        if self.auto_workflow:
            ready_for_tracking = self._auto_workflow_ready_for_tracking(wall_elapsed)
            self._send_loco_manip_policy_action(robot_state_data)
            if not ready_for_tracking:
                return
            self._maybe_resample_target()
            self._record_error(robot_state_data)
            return

        if self._safe_start_active:
            self._run_safe_start(robot_state_data)
            return

        if self.manual_start and not self.use_policy_action:
            if wall_elapsed >= self._next_start_log_t:
                self._next_start_log_t = wall_elapsed + 2.0
                if self.safe_start_enabled:
                    self.logger.info(colored("[EE_TRACK] waiting for manual ]/u safe policy start...", "yellow"))
                else:
                    self.logger.info(colored("[EE_TRACK] waiting for manual ] policy start...", "yellow"))
            return

        if self.manual_start and not self._started_policy:
            self._started_policy = True
            self._policy_start_wall_t = time.perf_counter()

        if not self.manual_start and not self._started_policy:
            if self.safe_start_enabled:
                self._begin_safe_start()
                self._run_safe_start(robot_state_data)
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

        self._send_loco_manip_policy_action(robot_state_data)
        if (
            self._policy_start_wall_t is not None
            and time.perf_counter() - self._policy_start_wall_t < self.tracking_start_delay_s
        ):
            return
        self._maybe_resample_target()
        self._record_error(robot_state_data)

    def run(self):
        try:
            super().run()
        finally:
            self._summarize_window()
            self.logger.info(colored(f"[EE_TRACK] csv={self.log_file}", "cyan"))
            self.logger.info(colored(f"[EE_TRACK] marker_file={self.marker_file}", "cyan"))


def _parse_range(text):
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be formatted as low,high")
    return tuple(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Right EE tracking test for loco-manip policy")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--marker_file", type=str, default="/tmp/falcon_ee_tracking_markers.json")
    parser.add_argument("--log_file", type=str, default="/tmp/falcon_ee_tracking_metrics.csv")
    parser.add_argument("--sample_period_s", type=float, default=5.0)
    parser.add_argument("--x_range", type=_parse_range, default=(0.20, 0.42))
    parser.add_argument("--y_range", type=_parse_range, default=(-0.26, -0.04))
    parser.add_argument("--z_range", type=_parse_range, default=(-0.02, 0.24))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument(
        "--auto_start_policy",
        action="store_true",
        help="Start policy automatically instead of waiting for manual ] key.",
    )
    parser.add_argument(
        "--auto_workflow",
        action="store_true",
        help="Wait for sim foot contact, start policy, wait for elastic release, then sample targets.",
    )
    parser.add_argument("--sim_status_file", type=str, default="/tmp/falcon_sim_status.json")
    parser.add_argument("--sim_control_file", type=str, default="/tmp/falcon_sim_control.json")
    parser.add_argument("--status_timeout_s", type=float, default=2.0)
    parser.add_argument("--tracking_delay_after_elastic_off_s", type=float, default=0.5)
    parser.add_argument("--tracking_start_delay_s", type=float, default=0.0)
    parser.add_argument("--elastic_release_delay_after_policy_start_s", type=float, default=None)
    parser.add_argument(
        "--safe_start_init_s",
        type=float,
        default=5.0,
        help="After manual ], hold current joints for this many seconds before policy blend.",
    )
    parser.add_argument(
        "--safe_start_blend_s",
        type=float,
        default=3.0,
        help="Blend from current joints to policy target for this many seconds after safe-start init.",
    )
    parser.add_argument(
        "--safe_start_target_hold_s",
        type=float,
        default=2.0,
        help="Hold the initial EE target for this many seconds after safe start before sampling a new target.",
    )
    parser.add_argument(
        "--safe_start_use_init",
        action="store_true",
        help="During safe-start warmup, move toward default init pose instead of holding current joints.",
    )
    parser.add_argument(
        "--max_joint_rate_rad_s",
        type=float,
        default=0.4,
        help="Limit commanded joint target changes while this test is running. Set <=0 to disable.",
    )
    parser.add_argument(
        "--policy_delta_warn_rad",
        type=float,
        default=0.4,
        help="Warn when the first policy target differs from current joints by more than this many radians.",
    )
    parser.add_argument(
        "--policy_delta_abort_rad",
        type=float,
        default=1.0,
        help="Abort safe-start before policy blend if first policy target differs by more than this many radians. Set <=0 to disable.",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    apply_named_motor_gain_scales(config)

    config["disable_keyboard_listener"] = bool(args.auto_start_policy or args.auto_workflow)
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path or config model_path")

    policy = LocoManipEETrackingTestPolicy(
        config,
        model_path,
        marker_file=args.marker_file,
        log_file=args.log_file,
        sample_period_s=args.sample_period_s,
        x_range=args.x_range,
        y_range=args.y_range,
        z_range=args.z_range,
        seed=args.seed,
        duration_s=args.duration_s,
        manual_start=not (args.auto_start_policy or args.auto_workflow),
        auto_workflow=args.auto_workflow,
        sim_status_file=args.sim_status_file,
        sim_control_file=args.sim_control_file,
        status_timeout_s=args.status_timeout_s,
        tracking_delay_after_elastic_off_s=args.tracking_delay_after_elastic_off_s,
        tracking_start_delay_s=args.tracking_start_delay_s,
        elastic_release_delay_after_policy_start_s=args.elastic_release_delay_after_policy_start_s,
        safe_start_init_s=args.safe_start_init_s,
        safe_start_blend_s=args.safe_start_blend_s,
        safe_start_target_hold_s=args.safe_start_target_hold_s,
        safe_start_use_init=args.safe_start_use_init,
        max_joint_rate_rad_s=args.max_joint_rate_rad_s,
        policy_delta_warn_rad=args.policy_delta_warn_rad,
        policy_delta_abort_rad=args.policy_delta_abort_rad,
    )
    policy.run()
