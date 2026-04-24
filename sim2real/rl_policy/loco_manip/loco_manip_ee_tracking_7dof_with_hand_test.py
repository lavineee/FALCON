import argparse
import sys
import time

import numpy as np
import yaml
from termcolor import colored

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.loco_manip.loco_manip_ee_tracking_7dof_test import (
    LocoManipEETracking7DofTestPolicy,
    _parse_range,
)
from sim2real.utils.arm_ik.robot_arm_ik_with_hand import G1_29_WithHandArmIK


class LocoManipEETracking7DofWithHandTestPolicy(LocoManipEETracking7DofTestPolicy):
    """Right palm/grasp-center position tracking test on the with-hand model."""

    def _init_keyboard_handler(self):
        if self.config.get("disable_keyboard_listener", False):
            self.use_joystick = False
            self.logger.info("Keyboard listener disabled")
            return
        super()._init_keyboard_handler()

    def init_upper_body_controller(self):
        if self.config["ROBOT_TYPE"] != "g1_29dof":
            self.logger.error("Unsupported robot type: %s", self.config["ROBOT_TYPE"])
            return

        self.upper_body_controller = G1_29_WithHandArmIK(
            Unit_Test=False,
            Visualization=False,
            robot_config=self.config,
        )
        self.upper_body_controller.opti.minimize(
            50 * self.upper_body_controller.translational_cost
            + 0.02 * self.upper_body_controller.regularization_cost
            + 0.1 * self.upper_body_controller.smooth_cost
        )

        self.waypoint_index = 0
        self.speed_factor = 0.05
        self.base_z_offset = 0.8
        self.degrees = 0
        self.theta = np.radians(self.degrees)
        self.EE_left_R = np.array(
            [
                [np.cos(-self.theta), -np.sin(-self.theta), 0],
                [np.sin(-self.theta), np.cos(-self.theta), 0],
                [0, 0, 1],
            ]
        )
        self.EE_right_R = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta), 0],
                [np.sin(self.theta), np.cos(self.theta), 0],
                [0, 0, 1],
            ]
        )
        self.EE_left_x = 0.30
        self.EE_right_x = 0.30
        self.EE_left_y = 0.13
        self.EE_right_y = -0.13
        self.EE_left_z = 0.08
        self.EE_right_z = 0.08
        self.update_waypoints()
        self.EE_efrc_L = np.array([0, 0, 0, 0, 0, 0])
        self.EE_efrc_R = np.array([0, 0, 0, 0, 0, 0])
        self.upper_body_controller.set_initial_poses(
            self.waypoints_left[0].translation,
            self.waypoints_right[0].translation,
            self.waypoints_left[0].rotation,
            self.waypoints_right[0].rotation,
        )

    def __init__(self, *args, hand_close_error_m=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hand_close_error_m = float(
            self.config.get("hand_close_error_m", 0.035) if hand_close_error_m is None else hand_close_error_m
        )
        self._right_hand_closed_for_target = False
        self._write_hand_state("open")

    def _write_hand_state(self, state):
        payload = {
            "right_hand_state": state,
            "timestamp": time.time(),
            "target_id": int(self._target_id),
        }
        self._write_control_file(payload)

    def _maybe_resample_target(self):
        old_target_id = self._target_id
        super()._maybe_resample_target()
        if self._target_id != old_target_id:
            self._right_hand_closed_for_target = False
            self._write_hand_state("open")
            self.logger.info(colored("[EE_TRACK_HAND] right hand open for approach", "cyan"))

    def _record_error(self, robot_state_data):
        super()._record_error(robot_state_data)
        if self._right_hand_closed_for_target or not self._window_errors:
            return
        if self._window_errors[-1] <= self.hand_close_error_m:
            self._right_hand_closed_for_target = True
            self._write_hand_state("close")
            self.logger.info(
                colored(
                    f"[EE_TRACK_HAND] right hand close, error={self._window_errors[-1]:.4f}m",
                    "cyan",
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Right palm/grasp-center 7-DoF tracking test with hands")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--marker_file", type=str, default="/tmp/falcon_ee_tracking_7dof_with_hand_markers.json")
    parser.add_argument("--log_file", type=str, default="/tmp/falcon_ee_tracking_7dof_with_hand_metrics.csv")
    parser.add_argument("--sample_period_s", type=float, default=5.0)
    parser.add_argument("--x_range", type=_parse_range, default=(0.20, 0.42))
    parser.add_argument("--y_range", type=_parse_range, default=(-0.26, -0.04))
    parser.add_argument("--z_range", type=_parse_range, default=(-0.02, 0.24))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument("--auto_start_policy", action="store_true")
    parser.add_argument("--auto_workflow", action="store_true")
    parser.add_argument("--sim_status_file", type=str, default="/tmp/falcon_sim_status_7dof_with_hand.json")
    parser.add_argument("--sim_control_file", type=str, default="/tmp/falcon_sim_control_7dof_with_hand.json")
    parser.add_argument("--status_timeout_s", type=float, default=2.0)
    parser.add_argument("--tracking_delay_after_elastic_off_s", type=float, default=0.5)
    parser.add_argument("--tracking_start_delay_s", type=float, default=0.0)
    parser.add_argument("--elastic_release_delay_after_policy_start_s", type=float, default=None)
    parser.add_argument("--hand_close_error_m", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    config["disable_keyboard_listener"] = bool(args.auto_start_policy or args.auto_workflow)
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path or config model_path")

    policy = LocoManipEETracking7DofWithHandTestPolicy(
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
        hand_close_error_m=args.hand_close_error_m,
    )
    policy.run()
