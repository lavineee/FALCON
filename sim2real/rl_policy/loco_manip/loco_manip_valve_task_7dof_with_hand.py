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


def _skew(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def _axis_angle_to_rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    k = _skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _normalize_or_none(v, eps=1e-8):
    v = np.asarray(v, dtype=float).reshape(3)
    norm = np.linalg.norm(v)
    if norm < eps:
        return None
    return v / norm


class SimStatusValveGeometryProvider:
    """从 MuJoCo status 文件读取阀门位姿。

    实机部署时可以替换成深度相机/估计器，只要输出同名字段即可。
    """

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
        center_world = np.asarray(status["valve_center_world"], dtype=float).reshape(3)
        grasp_world = np.asarray(status["valve_grasp_world"], dtype=float).reshape(3)
        axis_world = np.asarray(status.get("valve_axis_world", [1.0, 0.0, 0.0]), dtype=float).reshape(3)
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-9)

        # 将仿真世界坐标下的阀门几何量转到当前机器人 base 坐标系。
        world_to_base = base_xmat.T
        axis_base = world_to_base @ axis_world
        axis_base = axis_base / (np.linalg.norm(axis_base) + 1e-9)

        return {
            "timestamp": float(status["timestamp"]),
            "base_pos_world": base_pos,
            "base_xmat_world": base_xmat,
            "center_world": center_world,
            "grasp_world": grasp_world,
            "axis_world": axis_world,
            "center_base": world_to_base @ (center_world - base_pos),
            "grasp_base": world_to_base @ (grasp_world - base_pos),
            "axis_base": axis_base,
            "valve_angle": float(status.get("valve_angle", 0.0)),
            "valve_vel": float(status.get("valve_vel", 0.0)),
            "weld_enabled": bool(status.get("weld_enabled", False)),
            "right_hand_valve_contact_count": int(status.get("right_hand_valve_contact_count", 0)),
            "right_hand_valve_contact_normal_force": float(
                status.get("right_hand_valve_contact_normal_force", 0.0)
            ),
            "right_hand_valve_contact_pairs": status.get("right_hand_valve_contact_pairs", []),
        }

    @staticmethod
    def world_to_base(geom, pos_world):
        pos_world = np.asarray(pos_world, dtype=float).reshape(3)
        return geom["base_xmat_world"].T @ (pos_world - geom["base_pos_world"])


class LocoManipValveTask7DofWithHandPolicy(LocoManipEETracking7DofWithHandTestPolicy):
    """带手 7DoF 阀门任务：预接近、闭手、连接、转动的分层状态机。"""

    WAIT_GEOMETRY = "wait_geometry"
    MOVE_PREGRASP = "move_pregrasp"
    APPROACH_GRASP = "approach_grasp"
    CLOSE_HAND = "close_hand"
    ATTACH_WELD = "attach_weld"
    TURN_VALVE = "turn_valve"
    HOLD = "hold"
    DONE = "done"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_provider = SimStatusValveGeometryProvider(
            self.sim_status_file,
            timeout_s=self.status_timeout_s,
        )

        self.pregrasp_offset_m = float(self.config.get("valve_pregrasp_offset_m", 0.08))
        self.pregrasp_error_m = float(self.config.get("valve_pregrasp_error_m", 0.05))
        self.grasp_error_m = float(self.config.get("valve_grasp_error_m", 0.035))
        self.grasp_target_bias_base = np.asarray(
            self.config.get("valve_grasp_target_bias_base_m", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(3)
        self.close_wait_s = float(self.config.get("valve_close_wait_s", 0.8))
        self.weld_settle_s = float(self.config.get("valve_weld_settle_s", 0.8))
        self.hold_s = float(self.config.get("valve_hold_s", 2.0))
        self.task_start_delay_s = float(self.config.get("valve_task_start_delay_s", 0.5))
        self.hold_current_on_attach = bool(self.config.get("valve_hold_current_on_attach", True))
        self.enable_turn = bool(self.config.get("valve_enable_turn", True))
        self.turn_target_deg = float(self.config.get("valve_turn_target_deg", 20.0))
        self.turn_speed_deg = float(self.config.get("valve_turn_speed_deg", 8.0))
        self.align_grasp_orientation = bool(self.config.get("valve_align_grasp_orientation", False))
        self.grasp_forward_axis_sign = float(self.config.get("valve_grasp_forward_axis_sign", -1.0))
        # connect 是软点连接；none 用于只验证靠近/闭手，不施加阀门连接约束。
        self.attachment_mode = str(self.config.get("valve_attachment_mode", "connect")).strip().lower()
        if self.attachment_mode not in ("none", "connect"):
            self.logger.warning(
                colored(
                    f"[VALVE_TASK] unsupported valve_attachment_mode={self.attachment_mode}, using none",
                    "yellow",
                )
            )
            self.attachment_mode = "none"
        self.use_hard_attachment = self.attachment_mode == "connect"
        self.turn_actual_tolerance_deg = float(self.config.get("valve_turn_actual_tolerance_deg", 2.0))
        self.release_on_done = bool(self.config.get("valve_release_on_done", True))
        stop_after_done_s = self.config.get("valve_stop_after_done_s", 1.0)
        self.stop_after_done_s = None if stop_after_done_s is None else float(stop_after_done_s)

        self.task_state = self.WAIT_GEOMETRY
        self._state_enter_t = time.perf_counter()
        self._task_ready_t = None
        self._latest_geom = None
        self._current_target_base = np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z], dtype=float)
        self._turn_center_world = None
        self._turn_axis_world = None
        self._turn_r0_world = None
        self._turn_start_valve_angle = None
        self._turn_cmd_deg = 0.0
        self._hold_target_base = None
        self._weld_command_sent = False
        self._close_command_sent = False
        self._done_release_sent = False
        self._attach_hold_target_base = None

        self._write_weld_state(False)
        self._write_hand_state("open")
        self._reset_valve_log()

    def _reset_valve_log(self):
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as file:
            file.write(
                "time_s,state,target_x,target_y,target_z,current_x,current_y,current_z,"
                "error_m,ik_error_m,servo_error_m,grasp_error_m,valve_angle,valve_vel,"
                "turn_cmd_deg,valve_delta_deg,weld_enabled,hard_attachment_enabled\n"
            )

    def _enter_state(self, state):
        if self.task_state == state:
            return
        self.task_state = state
        self._state_enter_t = time.perf_counter()
        self.logger.info(colored(f"[VALVE_TASK] state -> {state}", "cyan"))

    def _write_weld_state(self, enabled):
        self._write_control_file(
            {
                "weld_enabled": bool(enabled),
                "weld_timestamp": time.time(),
                "timestamp": time.time(),
            }
        )

    def _approach_axis_base(self, geom):
        axis = np.asarray(geom["axis_base"], dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        # 选择朝向机器人一侧的阀门轴向，预抓取点就沿这个方向外移。
        toward_robot = -np.asarray(geom["center_base"], dtype=float)
        if np.dot(axis, toward_robot) < 0.0:
            axis = -axis
        return axis

    def _pregrasp_target_base(self, geom):
        return self._biased_grasp_target_base(geom) + self._approach_axis_base(geom) * self.pregrasp_offset_m

    def _grasp_target_base(self, geom):
        return np.asarray(geom["grasp_base"], dtype=float)

    def _biased_grasp_target_base(self, geom):
        return self._grasp_target_base(geom) + self.grasp_target_bias_base

    def _set_task_target(self, target_base):
        self._current_target_base = np.asarray(target_base, dtype=float).reshape(3)
        self._set_right_target(self._current_target_base)

    def _right_grasp_rotation_base(self, geom):
        x_axis = _normalize_or_none(self.grasp_forward_axis_sign * self._approach_axis_base(geom))
        if x_axis is None:
            return self.EE_right_R

        radial = self._biased_grasp_target_base(geom) - np.asarray(geom["center_base"], dtype=float)
        radial = radial - np.dot(radial, x_axis) * x_axis
        z_axis = _normalize_or_none(radial)
        if z_axis is None:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
            z_axis = _normalize_or_none(fallback - np.dot(fallback, x_axis) * x_axis)
        if z_axis is None:
            return self.EE_right_R

        y_axis = _normalize_or_none(np.cross(z_axis, x_axis))
        if y_axis is None:
            return self.EE_right_R
        z_axis = _normalize_or_none(np.cross(x_axis, y_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    def _set_valve_target(self, target_base, geom):
        if self.align_grasp_orientation and geom is not None:
            # 抓握阶段让手的前向轴对准阀门轴，手内 z 轴对准阀门半径方向。
            self.EE_right_R = self._right_grasp_rotation_base(geom)
        self._set_task_target(target_base)

    def _physical_grasp_error(self, geom, robot_state_data):
        current = self._current_right_fake_ee_base(robot_state_data)
        return float(np.linalg.norm(current - self._grasp_target_base(geom)))

    def _current_error(self, robot_state_data):
        current = self._current_right_fake_ee_base(robot_state_data)
        commanded = self._last_commanded_right_ee_base()
        error = float(np.linalg.norm(current - self._current_target_base))
        ik_error = float(np.linalg.norm(commanded - self._current_target_base)) if np.all(np.isfinite(commanded)) else np.nan
        servo_error = float(np.linalg.norm(current - commanded)) if np.all(np.isfinite(commanded)) else np.nan
        return current, commanded, error, ik_error, servo_error

    def _capture_turn_geometry(self, geom):
        # 转动开始时冻结阀门中心、轴线和半径向量，用它生成后续圆弧轨迹。
        self._turn_center_world = np.asarray(geom["center_world"], dtype=float).copy()
        self._turn_axis_world = np.asarray(geom["axis_world"], dtype=float).copy()
        self._turn_axis_world = self._turn_axis_world / (np.linalg.norm(self._turn_axis_world) + 1e-9)
        self._turn_r0_world = np.asarray(geom["grasp_world"], dtype=float) - self._turn_center_world
        self._turn_start_valve_angle = float(geom.get("valve_angle", 0.0))
        self._turn_cmd_deg = 0.0
        self.logger.info(
            colored(
                f"[VALVE_TASK] captured turn geometry radius={np.linalg.norm(self._turn_r0_world):.3f}m",
                "cyan",
            )
        )

    def _valve_delta_deg(self, geom):
        if self._turn_start_valve_angle is None:
            return 0.0
        return float(np.rad2deg(float(geom.get("valve_angle", 0.0)) - self._turn_start_valve_angle))

    def _actual_turn_reached(self, geom):
        target_abs = abs(self.turn_target_deg)
        if target_abs <= 1e-6:
            return True
        direction = 1.0 if self.turn_target_deg >= 0.0 else -1.0
        return direction * self._valve_delta_deg(geom) >= target_abs - self.turn_actual_tolerance_deg

    def _update_turn_target(self, geom):
        elapsed = time.perf_counter() - self._state_enter_t
        target_deg = abs(self.turn_target_deg)
        direction = 1.0 if self.turn_target_deg >= 0.0 else -1.0
        cmd_abs_deg = min(target_deg, elapsed * abs(self.turn_speed_deg))
        self._turn_cmd_deg = direction * cmd_abs_deg
        angle = np.deg2rad(self._turn_cmd_deg)
        rot = _axis_angle_to_rot(self._turn_axis_world, angle)
        target_world = self._turn_center_world + rot @ self._turn_r0_world
        self._set_valve_target(self.geometry_provider.world_to_base(geom, target_world), geom)
        if self._actual_turn_reached(geom) or cmd_abs_deg >= target_deg:
            self._hold_target_base = self._grasp_target_base(geom)
            self._enter_state(self.HOLD)

    def _update_task_state(self, robot_state_data):
        # 状态机每个 policy tick 都用最新 base 位姿重算目标，减少机身轻微移动带来的坐标漂移。
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
            self._set_valve_target(self._pregrasp_target_base(geom), geom)
            self._task_ready_t = time.perf_counter()
            self._enter_state(self.MOVE_PREGRASP)
            return

        if self._task_ready_t is not None and time.perf_counter() - self._task_ready_t < self.task_start_delay_s:
            self._set_valve_target(self._pregrasp_target_base(geom), geom)
            return

        if self.task_state == self.MOVE_PREGRASP:
            self._set_valve_target(self._pregrasp_target_base(geom), geom)
            _, _, error, _, _ = self._current_error(robot_state_data)
            if error <= self.pregrasp_error_m:
                self._enter_state(self.APPROACH_GRASP)
            return

        if self.task_state == self.APPROACH_GRASP:
            self._set_valve_target(self._biased_grasp_target_base(geom), geom)
            error = self._physical_grasp_error(geom, robot_state_data)
            if error <= self.grasp_error_m:
                self._enter_state(self.CLOSE_HAND)
            return

        if self.task_state == self.CLOSE_HAND:
            self._set_valve_target(self._biased_grasp_target_base(geom), geom)
            error = self._physical_grasp_error(geom, robot_state_data)
            if not self._close_command_sent:
                self._write_hand_state("close")
                self._close_command_sent = True
                self.logger.info(colored(f"[VALVE_TASK] close hand at error={error:.4f}m", "cyan"))
            if time.perf_counter() - self._state_enter_t >= self.close_wait_s:
                self._enter_state(self.ATTACH_WELD)
            return

        if self.task_state == self.ATTACH_WELD:
            if not self._weld_command_sent:
                # connect baseline 打开连接前保持当前手位；真实接触抓握可继续压向抓取点。
                if self.hold_current_on_attach:
                    self._attach_hold_target_base = self._current_right_fake_ee_base(robot_state_data).copy()
                else:
                    self._attach_hold_target_base = self._biased_grasp_target_base(geom)
                self._set_valve_target(self._attach_hold_target_base, geom)
                self._write_weld_state(self.use_hard_attachment)
                self._weld_command_sent = True
                if self.use_hard_attachment:
                    self.logger.info(colored("[VALVE_TASK] request hard connect ON", "cyan"))
                else:
                    self.logger.info(colored("[VALVE_TASK] hard attachment disabled", "cyan"))
            else:
                self._set_valve_target(
                    self._attach_hold_target_base
                    if self._attach_hold_target_base is not None
                    else self._biased_grasp_target_base(geom),
                    geom,
                )
            if time.perf_counter() - self._state_enter_t >= self.weld_settle_s:
                self._capture_turn_geometry(geom)
                self._enter_state(self.TURN_VALVE if self.enable_turn else self.HOLD)
            return

        if self.task_state == self.TURN_VALVE:
            self._update_turn_target(geom)
            return

        if self.task_state == self.HOLD:
            if self._hold_target_base is not None:
                self._set_valve_target(self._hold_target_base, geom)
            else:
                self._set_valve_target(self._current_target_base, geom)
            if time.perf_counter() - self._state_enter_t >= self.hold_s:
                self._enter_state(self.DONE)
            return

        if self.task_state == self.DONE:
            if self.use_hard_attachment and self.release_on_done and not self._done_release_sent:
                self._write_weld_state(False)
                self._done_release_sent = True
                self.logger.info(colored("[VALVE_TASK] release attachment after done", "cyan"))
            self._set_valve_target(self._current_target_base, geom)
            if self.stop_after_done_s is not None and time.perf_counter() - self._state_enter_t >= self.stop_after_done_s:
                raise KeyboardInterrupt

    def _record_valve_sample(self, robot_state_data):
        current, _, error, ik_error, servo_error = self._current_error(robot_state_data)
        geom = self._latest_geom or {}
        grasp_error = (
            float(np.linalg.norm(current - self._grasp_target_base(geom)))
            if "grasp_base" in geom
            else np.nan
        )
        valve_angle = float(geom.get("valve_angle", 0.0))
        valve_vel = float(geom.get("valve_vel", 0.0))
        valve_delta_deg = self._valve_delta_deg(geom)
        weld_enabled = bool(geom.get("weld_enabled", False))
        t = 0.0 if self._test_start_t is None else time.perf_counter() - self._test_start_t
        with open(self.log_file, "a") as file:
            file.write(
                f"{t:.6f},{self.task_state},"
                f"{self._current_target_base[0]:.6f},{self._current_target_base[1]:.6f},{self._current_target_base[2]:.6f},"
                f"{current[0]:.6f},{current[1]:.6f},{current[2]:.6f},"
                f"{error:.6f},{ik_error:.6f},{servo_error:.6f},{grasp_error:.6f},"
                f"{valve_angle:.6f},{valve_vel:.6f},"
                f"{self._turn_cmd_deg:.6f},{valve_delta_deg:.6f},{int(weld_enabled)},{int(self.use_hard_attachment)}\n"
            )
        self._write_marker_file(self._current_target_base, current, error)

    def policy_action(self):
        wall_elapsed = time.perf_counter() - self._wall_start_t
        if self.duration_s is not None and wall_elapsed >= self.duration_s:
            raise KeyboardInterrupt

        robot_state_data = self.state_processor.robot_state_data
        if robot_state_data is None:
            if wall_elapsed >= self._next_wait_log_t:
                self._next_wait_log_t = wall_elapsed + 2.0
                self.logger.info(colored("[VALVE_TASK] waiting for sim lowstate...", "yellow"))
            return

        if self.auto_workflow:
            ready = self._auto_workflow_ready_for_tracking(wall_elapsed)
            if ready and self._test_start_t is None:
                self._test_start_t = time.perf_counter()
            if ready:
                self._update_task_state(robot_state_data)
            self._send_7dof_policy_action()
            if ready:
                self._record_valve_sample(robot_state_data)
            return

        if self.manual_start and not self.use_policy_action:
            if wall_elapsed >= self._next_start_log_t:
                self._next_start_log_t = wall_elapsed + 2.0
                self.logger.info(colored("[VALVE_TASK] waiting for manual ] policy start...", "yellow"))
            return

        if self.manual_start and not self._started_policy:
            self._started_policy = True
            self._policy_start_wall_t = time.perf_counter()

        if not self.manual_start and not self._started_policy:
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
        self._record_valve_sample(robot_state_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7-DoF with-hand valve approach/grasp/turn task")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--marker_file", type=str, default="/tmp/falcon_valve_task_markers.json")
    parser.add_argument("--log_file", type=str, default="/tmp/falcon_valve_task_metrics.csv")
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument("--auto_start_policy", action="store_true")
    parser.add_argument("--auto_workflow", action="store_true")
    parser.add_argument("--sim_status_file", type=str, default="/tmp/falcon_valve_task_status.json")
    parser.add_argument("--sim_control_file", type=str, default="/tmp/falcon_valve_task_control.json")
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

    policy = LocoManipValveTask7DofWithHandPolicy(
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
