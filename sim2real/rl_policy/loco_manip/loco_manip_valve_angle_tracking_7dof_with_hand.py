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

from sim2real.rl_policy.loco_manip.loco_manip_ee_tracking_test import (
    apply_named_motor_gain_scales,
)
from sim2real.rl_policy.loco_manip.loco_manip_valve_task_7dof_with_hand import (
    LocoManipValveTask7DofWithHandPolicy,
    _axis_angle_to_rot,
)


def _parse_float_sequence(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _base_yaw_deg(geom):
    base_xmat = geom.get("base_xmat_world", None)
    if base_xmat is None:
        return np.nan
    rot = np.asarray(base_xmat, dtype=float).reshape(3, 3)
    return float(np.rad2deg(np.arctan2(rot[1, 0], rot[0, 0])))


class LocoManipValveAngleTracking7DofWithHandPolicy(LocoManipValveTask7DofWithHandPolicy):
    """阀门随机/指定角度跟踪 baseline，使用软点连接而不模拟真实抓握。"""

    SEGMENT_HOLD = "segment_hold"

    def _reset_valve_log(self):
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as file:
            file.write(
                "time_s,state,segment_id,segment_target_delta_deg,"
                "desired_delta_deg,command_delta_deg,actual_delta_deg,"
                "angle_error_deg,final_angle_error_deg,"
                "desired_angle_deg,command_angle_deg,actual_angle_deg,valve_vel,"
                "target_x,target_y,target_z,current_x,current_y,current_z,"
                "ee_error_m,ik_error_m,servo_error_m,connect_stretch_m,"
                "base_x,base_y,base_z,base_yaw_deg,attach_enabled\n"
            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.angle_targets_deg = self._build_angle_targets()
        self.angle_speed_deg = abs(float(self.config.get("valve_angle_tracking_speed_deg", 10.0)))
        if self.angle_speed_deg <= 1e-6:
            self.angle_speed_deg = 10.0
        self.angle_hold_s = float(self.config.get("valve_angle_tracking_hold_s", self.hold_s))
        self.min_segment_s = float(self.config.get("valve_angle_tracking_min_segment_s", 1.0))
        # closed_loop 会根据阀门实际角度修正末端圆弧命令；timed 只按时间走开环轨迹。
        self.control_mode = str(self.config.get("valve_angle_tracking_control_mode", "closed_loop")).strip().lower()
        if self.control_mode not in ("timed", "closed_loop"):
            self.logger.warning(
                colored(
                    f"[VALVE_ANGLE] unsupported valve_angle_tracking_control_mode={self.control_mode}, "
                    "using closed_loop",
                    "yellow",
                )
            )
            self.control_mode = "closed_loop"
        self.feedback_gain = float(self.config.get("valve_angle_tracking_feedback_gain", 0.8))
        self.command_lead_deg = abs(float(self.config.get("valve_angle_tracking_command_lead_deg", 20.0)))
        default_command_speed = max(1.8 * self.angle_speed_deg, self.angle_speed_deg + 4.0)
        self.command_speed_deg = abs(
            float(self.config.get("valve_angle_tracking_command_speed_deg", default_command_speed))
        )
        if self.command_speed_deg <= 1e-6:
            self.command_speed_deg = default_command_speed
        self.extra_settle_s = float(self.config.get("valve_angle_tracking_extra_settle_s", 2.0))
        self.angle_tolerance_deg = abs(float(self.config.get("valve_angle_tracking_tolerance_deg", 2.0)))

        self._segment_id = -1
        self._segment_target_delta_deg = 0.0
        self._segment_start_angle = 0.0
        self._segment_nominal_duration_s = self.min_segment_s
        self._segment_duration_s = self.min_segment_s
        self._segment_center_world = None
        self._segment_axis_world = None
        self._segment_r0_world = None
        self._segment_rows = []
        self._done_summarized = False
        self._segment_reference_delta_deg = 0.0
        self._segment_command_delta_deg = 0.0
        self._segment_final_error_deg = 0.0
        self._last_turn_update_t = None

        self.logger.info(
            colored(
                "[VALVE_ANGLE] targets_deg="
                f"{', '.join(f'{item:.1f}' for item in self.angle_targets_deg)} "
                f"speed={self.angle_speed_deg:.1f}deg/s "
                f"mode={self.control_mode} "
                f"attachment={self.attachment_mode}",
                "cyan",
            )
        )

    def _build_angle_targets(self):
        explicit_targets = _parse_float_sequence(self.config.get("valve_angle_tracking_targets_deg", None))
        if explicit_targets:
            return explicit_targets

        count = int(self.config.get("valve_angle_tracking_random_count", 4))
        min_abs = abs(float(self.config.get("valve_angle_tracking_min_abs_deg", 20.0)))
        max_abs = abs(float(self.config.get("valve_angle_tracking_max_abs_deg", 60.0)))
        if max_abs < min_abs:
            min_abs, max_abs = max_abs, min_abs
        magnitudes = self.rng.uniform(min_abs, max_abs, size=max(count, 1))
        signs = self.rng.choice([-1.0, 1.0], size=max(count, 1))
        return [float(sign * magnitude) for sign, magnitude in zip(signs, magnitudes)]

    def _target_world_for_delta(self, delta_deg):
        # 在阀门平面内绕轴旋转初始抓取半径，得到对应角度的世界系目标点。
        rot = _axis_angle_to_rot(self._segment_axis_world, np.deg2rad(delta_deg))
        return self._segment_center_world + rot @ self._segment_r0_world

    def _set_delta_target(self, geom, delta_deg):
        self._turn_cmd_deg = float(delta_deg)
        self._segment_command_delta_deg = self._turn_cmd_deg
        target_world = self._target_world_for_delta(delta_deg)
        self._set_task_target(self.geometry_provider.world_to_base(geom, target_world))

    def _actual_delta_deg(self, geom):
        valve_angle = float(geom.get("valve_angle", 0.0))
        return float(np.rad2deg(valve_angle - self._segment_start_angle))

    def _planned_delta_deg(self, elapsed_s):
        if self._segment_nominal_duration_s <= 1e-6:
            return self._segment_target_delta_deg
        progress = min(1.0, max(0.0, elapsed_s / self._segment_nominal_duration_s))
        return float(self._segment_target_delta_deg * progress)

    def _clamp_command_delta(self, delta_deg):
        low = min(0.0, self._segment_target_delta_deg) - self.command_lead_deg
        high = max(0.0, self._segment_target_delta_deg) + self.command_lead_deg
        return float(np.clip(delta_deg, low, high))

    def _start_next_segment(self, geom):
        self._summarize_segment()
        self._segment_id += 1
        if self._segment_id >= len(self.angle_targets_deg):
            self._enter_state(self.DONE)
            return

        self._segment_target_delta_deg = float(self.angle_targets_deg[self._segment_id])
        self._segment_start_angle = float(geom.get("valve_angle", 0.0))
        self._segment_nominal_duration_s = max(
            self.min_segment_s,
            abs(self._segment_target_delta_deg) / self.angle_speed_deg,
        )
        self._segment_duration_s = self._segment_nominal_duration_s
        if self.control_mode == "closed_loop":
            self._segment_duration_s += max(0.0, self.extra_settle_s)
        self._segment_center_world = np.asarray(geom["center_world"], dtype=float).copy()
        self._segment_axis_world = np.asarray(geom["axis_world"], dtype=float).copy()
        self._segment_axis_world = self._segment_axis_world / (np.linalg.norm(self._segment_axis_world) + 1e-9)
        # 每段开始时重新捕获阀门几何，目标点随后每 tick 转回当前 base 坐标系。
        self._segment_r0_world = np.asarray(geom["grasp_world"], dtype=float) - self._segment_center_world
        self._segment_rows = []
        self._segment_reference_delta_deg = 0.0
        self._segment_command_delta_deg = 0.0
        self._segment_final_error_deg = self._segment_target_delta_deg
        self._last_turn_update_t = None
        self._set_delta_target(geom, 0.0)
        self.logger.info(
            colored(
                f"[VALVE_ANGLE] segment={self._segment_id} "
                f"target_delta={self._segment_target_delta_deg:.1f}deg "
                f"duration={self._segment_duration_s:.2f}s "
                f"radius={np.linalg.norm(self._segment_r0_world):.3f}m",
                "green",
            )
        )
        self._enter_state(self.TURN_VALVE)

    def _summarize_segment(self):
        if not self._segment_rows:
            return
        rows = np.asarray(self._segment_rows, dtype=float)
        # columns: trajectory_error, final_error, ee_error, stretch, actual_delta
        trajectory_error = rows[:, 0]
        final_error = rows[:, 1]
        ee_error = rows[:, 2]
        stretch = rows[:, 3]
        actual_delta = rows[:, 4]
        self.logger.info(
            colored(
                f"[VALVE_ANGLE] segment={self._segment_id} "
                f"target={self._segment_target_delta_deg:.1f}deg "
                f"actual_final={actual_delta[-1]:.1f}deg "
                f"final_error={final_error[-1]:.2f}deg "
                f"traj_mae={np.mean(np.abs(trajectory_error)):.2f}deg "
                f"ee_mean={np.mean(ee_error):.4f}m "
                f"stretch_mean={np.mean(stretch):.4f}m",
                "cyan",
            )
        )
        self._segment_rows = []

    def _update_turn_target(self, geom, hold_reference=False):
        now = time.perf_counter()
        elapsed = now - self._state_enter_t
        planned_elapsed = self._segment_nominal_duration_s if hold_reference else elapsed
        desired_delta = self._planned_delta_deg(planned_elapsed)
        actual_delta = self._actual_delta_deg(geom)
        self._segment_final_error_deg = self._segment_target_delta_deg - actual_delta

        # reference 是希望阀门达到的角度；command 是发给末端圆弧的提前/滞后目标。
        if hold_reference or elapsed >= self._segment_nominal_duration_s:
            reference_delta = self._segment_target_delta_deg
        else:
            reference_delta = desired_delta
        self._segment_reference_delta_deg = float(reference_delta)

        if self.control_mode == "closed_loop":
            feedback = np.clip(
                self.feedback_gain * (reference_delta - actual_delta),
                -self.command_lead_deg,
                self.command_lead_deg,
            )
            raw_command_delta = self._clamp_command_delta(desired_delta + feedback)
            if self._last_turn_update_t is None:
                command_delta = raw_command_delta
            else:
                dt = np.clip(now - self._last_turn_update_t, 1e-3, 0.2)
                max_step = self.command_speed_deg * dt
                step = np.clip(raw_command_delta - self._segment_command_delta_deg, -max_step, max_step)
                command_delta = self._segment_command_delta_deg + step
            self._last_turn_update_t = now
            command_delta = self._clamp_command_delta(command_delta)
        else:
            command_delta = reference_delta

        self._set_delta_target(geom, command_delta)

        turn_time_done = elapsed >= self._segment_nominal_duration_s
        segment_timed_out = elapsed >= self._segment_duration_s
        target_reached = abs(self._segment_final_error_deg) <= self.angle_tolerance_deg
        if not hold_reference and turn_time_done and (target_reached or segment_timed_out):
            self._enter_state(self.SEGMENT_HOLD)

    def _update_task_state(self, robot_state_data):
        geom = self.geometry_provider.read()
        if geom is not None:
            self._latest_geom = geom
        else:
            geom = self._latest_geom

        if geom is None:
            self._enter_state(self.WAIT_GEOMETRY)
            return

        if self.task_state == self.WAIT_GEOMETRY:
            self._write_weld_state(False)
            self._write_hand_state("open")
            self._set_task_target(self._pregrasp_target_base(geom))
            self._task_ready_t = time.perf_counter()
            self._enter_state(self.MOVE_PREGRASP)
            return

        if self._task_ready_t is not None and time.perf_counter() - self._task_ready_t < self.task_start_delay_s:
            self._set_task_target(self._pregrasp_target_base(geom))
            return

        if self.task_state == self.MOVE_PREGRASP:
            self._set_task_target(self._pregrasp_target_base(geom))
            _, _, error, _, _ = self._current_error(robot_state_data)
            if error <= self.pregrasp_error_m:
                self._enter_state(self.APPROACH_GRASP)
            return

        if self.task_state == self.APPROACH_GRASP:
            self._set_task_target(self._biased_grasp_target_base(geom))
            error = self._physical_grasp_error(geom, robot_state_data)
            if error <= self.grasp_error_m:
                self._enter_state(self.CLOSE_HAND)
            return

        if self.task_state == self.CLOSE_HAND:
            self._set_task_target(self._biased_grasp_target_base(geom))
            error = self._physical_grasp_error(geom, robot_state_data)
            if not self._close_command_sent:
                self._write_hand_state("close")
                self._close_command_sent = True
                self.logger.info(colored(f"[VALVE_ANGLE] close hand at grasp_error={error:.4f}m", "cyan"))
            if time.perf_counter() - self._state_enter_t >= self.close_wait_s:
                self._enter_state(self.ATTACH_WELD)
            return

        if self.task_state == self.ATTACH_WELD:
            if not self._weld_command_sent:
                self._attach_hold_target_base = self._current_right_fake_ee_base(robot_state_data).copy()
                self._set_task_target(self._attach_hold_target_base)
                self._write_weld_state(self.use_hard_attachment)
                self._weld_command_sent = True
                if self.use_hard_attachment:
                    self.logger.info(colored("[VALVE_ANGLE] soft point connect ON", "cyan"))
                else:
                    self.logger.info(colored("[VALVE_ANGLE] attachment disabled", "cyan"))
            else:
                self._set_task_target(
                    self._attach_hold_target_base
                    if self._attach_hold_target_base is not None
                    else self._biased_grasp_target_base(geom)
                )
            if time.perf_counter() - self._state_enter_t >= self.weld_settle_s:
                self._start_next_segment(geom)
            return

        if self.task_state == self.TURN_VALVE:
            self._update_turn_target(geom)
            return

        if self.task_state == self.SEGMENT_HOLD:
            self._update_turn_target(geom, hold_reference=True)
            if time.perf_counter() - self._state_enter_t >= self.angle_hold_s:
                self._start_next_segment(geom)
            return

        if self.task_state == self.DONE:
            if not self._done_summarized:
                self._summarize_segment()
                self._done_summarized = True
            if self.use_hard_attachment and self.release_on_done and not self._done_release_sent:
                self._write_weld_state(False)
                self._done_release_sent = True
                self.logger.info(colored("[VALVE_ANGLE] soft point connect OFF", "cyan"))
            self._set_task_target(self._current_target_base)
            if self.stop_after_done_s is not None and time.perf_counter() - self._state_enter_t >= self.stop_after_done_s:
                raise KeyboardInterrupt

    def _record_valve_sample(self, robot_state_data):
        # 同时记录轨迹误差和最终目标误差，便于区分“过程跟不上”和“最终没到位”。
        current, _, ee_error, ik_error, servo_error = self._current_error(robot_state_data)
        geom = self._latest_geom or {}
        valve_angle = float(geom.get("valve_angle", 0.0))
        valve_vel = float(geom.get("valve_vel", 0.0))
        if self._segment_id < 0:
            actual_delta_deg = 0.0
            desired_delta_deg = 0.0
            command_delta_deg = 0.0
            angle_error_deg = 0.0
            final_angle_error_deg = 0.0
            desired_angle_deg = float(np.rad2deg(valve_angle))
            command_angle_deg = desired_angle_deg
        else:
            actual_delta_deg = self._actual_delta_deg(geom)
            desired_delta_deg = float(self._segment_reference_delta_deg)
            command_delta_deg = float(self._turn_cmd_deg)
            angle_error_deg = desired_delta_deg - actual_delta_deg
            final_angle_error_deg = self._segment_target_delta_deg - actual_delta_deg
            desired_angle_deg = float(np.rad2deg(self._segment_start_angle) + desired_delta_deg)
            command_angle_deg = float(np.rad2deg(self._segment_start_angle) + command_delta_deg)
        actual_angle_deg = float(np.rad2deg(valve_angle))
        connect_stretch = (
            float(np.linalg.norm(current - self._grasp_target_base(geom)))
            if "grasp_base" in geom
            else np.nan
        )
        attach_enabled = bool(geom.get("weld_enabled", False))
        base_pos = np.asarray(geom.get("base_pos_world", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        base_yaw = _base_yaw_deg(geom)

        if self.task_state in (self.TURN_VALVE, self.SEGMENT_HOLD) and self._segment_id >= 0:
            self._segment_rows.append(
                [angle_error_deg, final_angle_error_deg, ee_error, connect_stretch, actual_delta_deg]
            )

        t = 0.0 if self._test_start_t is None else time.perf_counter() - self._test_start_t
        with open(self.log_file, "a") as file:
            file.write(
                f"{t:.6f},{self.task_state},{self._segment_id},"
                f"{self._segment_target_delta_deg:.6f},{desired_delta_deg:.6f},{command_delta_deg:.6f},"
                f"{actual_delta_deg:.6f},{angle_error_deg:.6f},{final_angle_error_deg:.6f},"
                f"{desired_angle_deg:.6f},{command_angle_deg:.6f},{actual_angle_deg:.6f},{valve_vel:.6f},"
                f"{self._current_target_base[0]:.6f},{self._current_target_base[1]:.6f},{self._current_target_base[2]:.6f},"
                f"{current[0]:.6f},{current[1]:.6f},{current[2]:.6f},"
                f"{ee_error:.6f},{ik_error:.6f},{servo_error:.6f},{connect_stretch:.6f},"
                f"{base_pos[0]:.6f},{base_pos[1]:.6f},{base_pos[2]:.6f},{base_yaw:.6f},"
                f"{int(attach_enabled)}\n"
            )
        self._write_marker_file(self._current_target_base, current, ee_error)

    def _arc_points_base(self, geom, delta_deg, max_points=18):
        if (
            geom is None
            or self._segment_center_world is None
            or self._segment_axis_world is None
            or self._segment_r0_world is None
        ):
            return []
        point_count = int(np.clip(np.ceil(abs(delta_deg) / 6.0) + 2, 2, max_points))
        return [
            self.geometry_provider.world_to_base(geom, self._target_world_for_delta(sample_delta)).tolist()
            for sample_delta in np.linspace(0.0, float(delta_deg), point_count)
        ]

    def _point_base_for_delta(self, geom, delta_deg):
        if (
            geom is None
            or self._segment_center_world is None
            or self._segment_axis_world is None
            or self._segment_r0_world is None
        ):
            return None
        return self.geometry_provider.world_to_base(geom, self._target_world_for_delta(delta_deg))

    def _write_marker_file(self, target, current, error):
        # marker JSON 由 MuJoCo 侧读取，用来画目标点、圆弧和角度文字。
        payload = {
            "target_right_base": np.asarray(target, dtype=float).reshape(3).tolist(),
            "current_right_base": np.asarray(current, dtype=float).reshape(3).tolist(),
            "error_m": float(error),
            "target_id": int(self._target_id),
            "timestamp": time.time(),
        }

        geom = self._latest_geom
        if geom is not None and "center_base" in geom and self._segment_id >= 0:
            actual_delta = self._actual_delta_deg(geom)
            final_target_base = self._point_base_for_delta(geom, self._segment_target_delta_deg)
            command_target_base = self._point_base_for_delta(geom, self._turn_cmd_deg)
            payload.update(
                {
                    "valve_center_base": np.asarray(geom["center_base"], dtype=float).reshape(3).tolist(),
                    "valve_actual_grasp_base": np.asarray(geom["grasp_base"], dtype=float).reshape(3).tolist(),
                    "valve_arc_points_base": self._arc_points_base(geom, self._turn_cmd_deg),
                    "valve_desired_delta_deg": float(self._segment_reference_delta_deg),
                    "valve_command_delta_deg": float(self._turn_cmd_deg),
                    "valve_actual_delta_deg": float(actual_delta),
                    "valve_segment_target_delta_deg": float(self._segment_target_delta_deg),
                    "valve_label_base": (
                        np.asarray(geom["center_base"], dtype=float).reshape(3)
                        + np.array([0.0, 0.0, 0.22], dtype=float)
                    ).tolist(),
                    "valve_angle_label": (
                        f"target {self._segment_target_delta_deg:+.1f} deg | "
                        f"ref {self._segment_reference_delta_deg:+.1f} | "
                        f"cmd {self._turn_cmd_deg:+.1f} | "
                        f"actual {actual_delta:+.1f}"
                    ),
                }
            )
            if final_target_base is not None:
                payload["valve_final_target_base"] = final_target_base.tolist()
            if command_target_base is not None:
                payload["valve_command_target_base"] = command_target_base.tolist()

        tmp_file = f"{self.marker_file}.tmp"
        with open(tmp_file, "w") as file:
            json.dump(payload, file)
        os.replace(tmp_file, self.marker_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soft-connect random valve-angle tracking baseline")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--marker_file", type=str, default="/tmp/falcon_valve_angle_tracking_markers.json")
    parser.add_argument("--log_file", type=str, default="/tmp/falcon_valve_angle_tracking_metrics.csv")
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--auto_start_policy", action="store_true")
    parser.add_argument("--auto_workflow", action="store_true")
    parser.add_argument("--sim_status_file", type=str, default="/tmp/falcon_valve_angle_tracking_status.json")
    parser.add_argument("--sim_control_file", type=str, default="/tmp/falcon_valve_angle_tracking_control.json")
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

    policy = LocoManipValveAngleTracking7DofWithHandPolicy(
        config,
        model_path,
        marker_file=args.marker_file,
        log_file=args.log_file,
        sample_period_s=999.0,
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
    )
    policy.run()
