import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy


class RightHandCircleKinematicsPolicy(LocoManipPolicy):
    """Right hand end-effector circle test in robot-base coordinates."""

    def __init__(
        self,
        *args,
        circle_center_base,
        circle_radius_m,
        circle_plane,
        approach_sec,
        hold_sec,
        period_sec,
        revolutions,
        start_angle_deg,
        direction,
        ik_speed_factor,
        arm_q_filter_alpha,
        arm_max_vel_rad_s,
        log_file: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.circle_center_base = np.asarray(circle_center_base, dtype=np.float64)
        self.circle_radius_m = float(circle_radius_m)
        self.circle_plane = circle_plane.lower()
        self.approach_sec = max(float(approach_sec), 0.0)
        self.hold_sec = max(float(hold_sec), 0.0)
        self.period_sec = max(float(period_sec), 1e-6)
        self.revolutions = max(float(revolutions), 0.0)
        self.start_angle_rad = math.radians(float(start_angle_deg))
        self.direction = 1.0 if float(direction) >= 0.0 else -1.0
        self.ik_speed_factor = float(ik_speed_factor)
        self.arm_q_filter_alpha = float(arm_q_filter_alpha)
        self.arm_max_vel_rad_s = float(arm_max_vel_rad_s)

        if self.circle_plane not in {"yz", "xz", "xy"}:
            raise ValueError("circle_plane must be one of: yz, xz, xy")
        if self.circle_radius_m <= 0.0:
            raise ValueError("circle_radius_m must be positive")

        self._circle_state = "idle"
        self._circle_state_t0 = 0.0
        self._circle_t0 = 0.0
        self._approach_from = self._current_target()
        self._last_print_t = 0.0
        self._smoothed_q_target = None
        self._smoothed_q_t = None
        self.right_arm_ik_dof_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]
        self.right_arm_ik_dof_indices = [
            self.dof_names.index(name) for name in self.right_arm_ik_dof_names if name in self.dof_names
        ]

        if self.upper_body_controller is not None:
            self.upper_body_controller.speed_factor = self.ik_speed_factor

        self.log_file = log_file
        self._log_fp = None
        if self.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
            self._log_fp = open(self.log_file, "w", encoding="utf-8")
            self._log_fp.write("time_sec,state,target_x,target_y,target_z,theta_rad\n")
            self._log_fp.flush()

        logger.info(
            "Right hand circle test ready: center_base={} radius={:.3f}m plane={} period={:.1f}s revs={:.2f}",
            np.round(self.circle_center_base, 4).tolist(),
            self.circle_radius_m,
            self.circle_plane,
            self.period_sec,
            self.revolutions,
        )
        logger.info(
            "Smoothing: ik_speed_factor={:.3f} arm_q_filter_alpha={:.3f} arm_max_vel={:.3f}rad/s",
            self.ik_speed_factor,
            self.arm_q_filter_alpha,
            self.arm_max_vel_rad_s,
        )
        logger.info("Keys: i=hold current, ]/u=enable current-joint hold, l=start right-arm circle, ;=pause, o=stop")

    def _current_target(self):
        return np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z], dtype=np.float64)

    def _circle_point(self, theta_rad):
        c = self.circle_center_base
        r = self.circle_radius_m

        if self.circle_plane == "yz":
            return np.array([c[0], c[1] + r * math.cos(theta_rad), c[2] + r * math.sin(theta_rad)])
        if self.circle_plane == "xz":
            return np.array([c[0] + r * math.cos(theta_rad), c[1], c[2] + r * math.sin(theta_rad)])
        return np.array([c[0] + r * math.cos(theta_rad), c[1] + r * math.sin(theta_rad), c[2]])

    def _set_right_target(self, target):
        self.EE_right_x = float(target[0])
        self.EE_right_y = float(target[1])
        self.EE_right_z = float(target[2])
        self.update_waypoints()

    def _start_circle(self):
        if not self.use_policy_action:
            logger.warning("Policy is not enabled yet. Press ] first, then press l to start the circle.")
            return

        self._approach_from = self._current_target()
        now = time.perf_counter()
        self._circle_state = "approach"
        self._circle_state_t0 = now
        self._circle_t0 = now
        logger.info(
            "Starting right hand circle: approach from {} to {}",
            np.round(self._approach_from, 4).tolist(),
            np.round(self._circle_point(self.start_angle_rad), 4).tolist(),
        )

    def _pause_circle(self):
        if self._circle_state != "idle":
            logger.info("Circle paused at target {}", np.round(self._current_target(), 4).tolist())
        self._circle_state = "idle"
        self._smoothed_q_target = None
        self._smoothed_q_t = None

    def _hold_current_pose(self):
        self.get_ready_state = False
        self.use_policy_action = False
        self.init_count = 0
        self._pause_circle()
        if hasattr(self.command_sender, "no_action"):
            self.command_sender.no_action = 0
        logger.info("Current-pose hold armed. Press ] to enable hold, then l to start the right-arm circle.")

    def _record_target(self, state, target, theta_rad):
        now = time.time()
        if self._log_fp:
            self._log_fp.write(
                f"{now:.6f},{state},{target[0]:.6f},{target[1]:.6f},{target[2]:.6f},{theta_rad:.6f}\n"
            )
            self._log_fp.flush()

        perf_now = time.perf_counter()
        if perf_now - self._last_print_t > 0.5:
            self._last_print_t = perf_now
            logger.info(
                "circle_state={} right_target_base=[{:.3f}, {:.3f}, {:.3f}]",
                state,
                target[0],
                target[1],
                target[2],
            )

    def _update_circle_target(self):
        if self._circle_state == "idle":
            return

        now = time.perf_counter()
        start_point = self._circle_point(self.start_angle_rad)
        theta = self.start_angle_rad

        if self._circle_state == "approach":
            if self.approach_sec <= 0.0:
                alpha = 1.0
            else:
                alpha = min((now - self._circle_state_t0) / self.approach_sec, 1.0)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target = (1.0 - alpha) * self._approach_from + alpha * start_point
            if alpha >= 1.0:
                self._circle_state = "hold"
                self._circle_state_t0 = now
                logger.info("Approach done; holding circle start point")

        elif self._circle_state == "hold":
            target = start_point
            if now - self._circle_state_t0 >= self.hold_sec:
                self._circle_state = "circle"
                self._circle_state_t0 = now
                logger.info("Circle motion started")

        elif self._circle_state == "circle":
            total_sec = self.period_sec * self.revolutions
            if total_sec <= 0.0:
                progress = 1.0
            else:
                progress = min((now - self._circle_state_t0) / total_sec, 1.0)
            theta = self.start_angle_rad + self.direction * 2.0 * math.pi * self.revolutions * progress
            target = self._circle_point(theta)
            if progress >= 1.0:
                self._circle_state = "done"
                self._circle_state_t0 = now
                logger.info("Circle motion done; holding final target")

        else:
            target = self._circle_point(self.start_angle_rad + self.direction * 2.0 * math.pi * self.revolutions)
            theta = self.start_angle_rad + self.direction * 2.0 * math.pi * self.revolutions

        self._set_right_target(target)
        self._record_target(self._circle_state, target, theta)

    def _smooth_arm_q_target(self, q_target, robot_state_data):
        if not self.use_policy_action or self._circle_state == "idle":
            self._smoothed_q_target = None
            self._smoothed_q_t = None
            return q_target

        indices = getattr(self, "upper_dof_indices", None)
        if not indices:
            return q_target

        alpha = np.clip(self.arm_q_filter_alpha, 0.0, 1.0)
        max_vel = max(self.arm_max_vel_rad_s, 0.0)
        if alpha >= 1.0 and max_vel <= 0.0:
            return q_target

        now = time.perf_counter()
        q_smooth = q_target.copy()
        if self._smoothed_q_target is None:
            self._smoothed_q_target = robot_state_data[:, 7 : 7 + self.num_dofs].copy()
            self._smoothed_q_t = now

        dt = max(now - self._smoothed_q_t, 1e-3)
        prev = self._smoothed_q_target.copy()

        desired = q_target.copy()
        desired[:, indices] = prev[:, indices] + alpha * (q_target[:, indices] - prev[:, indices])

        if max_vel > 0.0:
            max_step = max_vel * dt
            delta = np.clip(desired[:, indices] - prev[:, indices], -max_step, max_step)
            desired[:, indices] = prev[:, indices] + delta

        q_smooth[:, indices] = desired[:, indices]
        self._smoothed_q_target = q_smooth.copy()
        self._smoothed_q_t = now
        return q_smooth

    def handle_keyboard_button(self, keycode):
        if keycode == "i":
            self._hold_current_pose()
            return
        if keycode == "u":
            self._handle_start_policy()
            return

        super().handle_keyboard_button(keycode)

        if keycode == "l":
            self._start_circle()
        elif keycode == ";":
            self._pause_circle()
        elif keycode in {"o", "p"}:
            self._pause_circle()

    def policy_action(self):
        self._update_circle_target()
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)

        robot_state_data = self.state_processor.robot_state_data
        if robot_state_data is None:
            return

        q_target = robot_state_data[:, 7 : 7 + self.num_dofs].copy()

        if self.use_policy_action and self._circle_state != "idle" and self.upper_body_controller:
            upper_body_qpos, _ = self.upper_body_controller.get_q_tau(
                self.waypoints_left[0],
                self.waypoints_right[0],
                self.EE_efrc_L,
                self.EE_efrc_R,
            )
            right_q = np.asarray(upper_body_qpos[4:8], dtype=np.float64)
            if len(self.right_arm_ik_dof_indices) != len(right_q):
                raise RuntimeError(
                    f"right_arm_ik_dof_indices length {len(self.right_arm_ik_dof_indices)} "
                    f"does not match IK right arm q length {len(right_q)}"
                )
            q_target[0, self.right_arm_ik_dof_indices] = right_q

        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        q_target = self._smooth_arm_q_target(q_target, robot_state_data)
        cmd_q = q_target[0]
        self._sync_command_mode_from_state()
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs])

    def close(self):
        if self._log_fp:
            self._log_fp.close()
            self._log_fp = None

    def run(self):
        try:
            return super().run()
        finally:
            self.close()


def _parse_center(value):
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("center must be formatted as x,y,z")
    return parts


def _load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        repo_relative = REPO_ROOT / "sim2real" / config_path
        if repo_relative.exists():
            path = repo_relative

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="Right hand pure kinematic circle test for G1 sim2real.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/local/g1_29dof_falcon_real.yaml",
        help="Policy config path, relative to sim2real/ when launched from sim2real.",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Override ONNX model path.")
    parser.add_argument(
        "--center_base",
        type=_parse_center,
        default=_parse_center("0.35,-0.20,0.20"),
        help="Circle center in robot base frame as x,y,z meters.",
    )
    parser.add_argument("--radius_m", type=float, default=0.20, help="Circle radius in meters.")
    parser.add_argument(
        "--plane",
        type=str,
        default="yz",
        choices=["yz", "xz", "xy"],
        help="Circle plane in robot base frame. yz keeps x fixed, like turning a vertical valve in front of the robot.",
    )
    parser.add_argument("--approach_sec", type=float, default=6.0, help="Time to move from current target to circle start.")
    parser.add_argument("--hold_sec", type=float, default=1.0, help="Hold time at circle start before drawing.")
    parser.add_argument("--period_sec", type=float, default=20.0, help="Seconds per full circle.")
    parser.add_argument("--revolutions", type=float, default=1.0, help="Number of circles to draw.")
    parser.add_argument("--start_angle_deg", type=float, default=-90.0, help="Start angle on the circle.")
    parser.add_argument(
        "--direction",
        type=float,
        default=1.0,
        help="Positive for counter-clockwise in the selected plane, negative for clockwise.",
    )
    parser.add_argument(
        "--ik_speed_factor",
        type=float,
        default=0.015,
        help="End-effector interpolation factor inside the arm IK. Smaller is smoother but laggier.",
    )
    parser.add_argument(
        "--arm_q_filter_alpha",
        type=float,
        default=0.12,
        help="Upper-body joint target low-pass alpha. Smaller is smoother but laggier; 1 disables low-pass.",
    )
    parser.add_argument(
        "--arm_max_vel_rad_s",
        type=float,
        default=0.45,
        help="Upper-body joint target velocity limit in rad/s. Use 0 to disable.",
    )
    parser.add_argument("--log_file", type=str, default=None, help="Optional CSV log path for commanded targets.")

    args = parser.parse_args()
    config = _load_config(args.config)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path or config model_path")

    policy = RightHandCircleKinematicsPolicy(
        config=config,
        model_path=model_path,
        circle_center_base=args.center_base,
        circle_radius_m=args.radius_m,
        circle_plane=args.plane,
        approach_sec=args.approach_sec,
        hold_sec=args.hold_sec,
        period_sec=args.period_sec,
        revolutions=args.revolutions,
        start_angle_deg=args.start_angle_deg,
        direction=args.direction,
        ik_speed_factor=args.ik_speed_factor,
        arm_q_filter_alpha=args.arm_q_filter_alpha,
        arm_max_vel_rad_s=args.arm_max_vel_rad_s,
        log_file=args.log_file,
    )
    policy.run()


if __name__ == "__main__":
    main()
