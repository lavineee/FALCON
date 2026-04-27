import argparse
import sys
import time

import numpy as np
import pinocchio as pin
import yaml
from termcolor import colored

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.loco_manip.loco_manip_ee_tracking_test import (
    LocoManipEETrackingTestPolicy,
    _parse_float_list,
    _parse_range,
    apply_named_motor_gain_scales,
)
from sim2real.utils.arm_ik.robot_arm_ik import G1_29_ArmIK
from sim2real.utils.arm_ik.weighted_moving_filter import WeightedMovingFilter


class LocoManipEETracking7DofTestPolicy(LocoManipEETrackingTestPolicy):
    """使用右臂 7DoF 做末端位置跟踪测试。"""

    def _configure_tracking_ik(self):
        controller = self.upper_body_controller
        translational_weight = float(self.config.get("tracking_ik_translational_weight", 50.0))
        regularization_weight = float(self.config.get("tracking_ik_regularization_weight", 0.02))
        smooth_weight = float(self.config.get("tracking_ik_smooth_weight", 0.1))
        include_rotation = bool(self.config.get("tracking_ik_include_rotation", False))

        # 当前评测只关心末端位置误差；姿态项默认关闭，避免给手腕引入额外约束。
        objective = (
            translational_weight * controller.translational_cost
            + regularization_weight * controller.regularization_cost
            + smooth_weight * controller.smooth_cost
        )
        if include_rotation:
            objective += controller.rotation_cost
        controller.opti.minimize(objective)

        controller.speed_factor = float(self.config.get("tracking_ik_speed_factor", 0.05))
        filter_weights = _parse_float_list(self.config.get("tracking_ik_filter_weights", None))
        if filter_weights:
            filter_weights = np.asarray(filter_weights, dtype=float)
            filter_weights = filter_weights / np.sum(filter_weights)
            controller.smooth_filter = WeightedMovingFilter(
                filter_weights,
                controller.reduced_robot.model.nq,
            )

        self.logger.info(
            colored(
                "[EE_TRACK_7DOF] IK tuning: "
                f"speed={controller.speed_factor:.3f}, "
                f"w_trans={translational_weight:.1f}, "
                f"w_reg={regularization_weight:.3f}, "
                f"w_smooth={smooth_weight:.3f}, "
                f"rotation={include_rotation}",
                "cyan",
            )
        )

    def init_upper_body_controller(self):
        if self.config["ROBOT_TYPE"] != "g1_29dof":
            self.logger.error("Unsupported robot type: %s", self.config["ROBOT_TYPE"])
            return

        self.upper_body_controller = G1_29_ArmIK(
            Unit_Test=False, Visualization=False, robot_config=self.config
        )
        self._configure_tracking_ik()

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 7DoF 版本使用完整右臂链，包括手腕三个关节。
        self.full_upper_joint_indices = [
            self.dof_names.index(name) for name in self.config["dof_names_upper_body"]
        ]
        self.right_ee_frame_id = self.upper_body_controller.reduced_robot.model.getFrameId("R_ee")
        self.upper_tau_ff_scale = float(self.config.get("tracking_upper_tau_ff_scale", 0.0))
        self.upper_tau_ff_clip = self.config.get("tracking_upper_tau_ff_clip", None)
        self.upper_tau_ff_clip = None if self.upper_tau_ff_clip is None else float(self.upper_tau_ff_clip)
        if self.upper_tau_ff_scale:
            self.logger.info(
                colored(
                    f"[EE_TRACK_7DOF] upper feedforward tau scale={self.upper_tau_ff_scale:.3f}, "
                    f"clip={self.upper_tau_ff_clip}",
                    "cyan",
                )
            )

    def _send_7dof_policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)

        robot_state_data = self.state_processor.robot_state_data
        if self.upper_body_controller:
            upper_body_qpos, upper_body_tauff = self.upper_body_controller.get_q_tau(
                self.waypoints_left[0],
                self.waypoints_right[0],
                self.EE_efrc_L,
                self.EE_efrc_R,
            )
            self._last_upper_body_qpos = np.asarray(upper_body_qpos[: self.num_upper_dofs], dtype=float)
            self._last_upper_body_tauff = np.asarray(upper_body_tauff[: self.num_upper_dofs], dtype=float)
            self.ref_upper_dof_pos[0, :] = self._last_upper_body_qpos

        # 下肢和躯干仍由原 policy 输出；上肢参考由 IK 写入 ref_upper_dof_pos。
        scaled_policy_action = self.rl_inference(robot_state_data)
        if self.get_ready_state:
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            q_target = scaled_policy_action + self.default_dof_angles

        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(
                q_target[0],
                self.motor_pos_lower_limit_list,
                self.motor_pos_upper_limit_list,
            )

        if self.upper_tau_ff_scale and self._last_upper_body_tauff is not None:
            # 只给上肢叠加小比例 IK 前馈力矩，避免影响下肢平衡策略。
            tau_ff = self.upper_tau_ff_scale * self._last_upper_body_tauff
            if self.upper_tau_ff_clip is not None:
                tau_ff = np.clip(tau_ff, -self.upper_tau_ff_clip, self.upper_tau_ff_clip)
            cmd_tau[self.upper_dof_indices] = tau_ff

        cmd_q = q_target[0]
        self.command_sender.send_command(
            cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs]
        )

    def policy_action(self):
        wall_elapsed = time.perf_counter() - self._wall_start_t
        if self.duration_s is not None and wall_elapsed >= self.duration_s:
            raise KeyboardInterrupt

        robot_state_data = self.state_processor.robot_state_data
        if robot_state_data is None:
            if wall_elapsed >= self._next_wait_log_t:
                self._next_wait_log_t = wall_elapsed + 2.0
                self.logger.info(colored("[EE_TRACK_7DOF] waiting for sim lowstate...", "yellow"))
            return

        if self.auto_workflow:
            ready_for_tracking = self._auto_workflow_ready_for_tracking(wall_elapsed)
            self._send_7dof_policy_action()
            if not ready_for_tracking:
                return
            self._maybe_resample_target()
            self._record_error(robot_state_data)
            return

        if self.manual_start and not self.use_policy_action:
            if wall_elapsed >= self._next_start_log_t:
                self._next_start_log_t = wall_elapsed + 2.0
                self.logger.info(colored("[EE_TRACK_7DOF] waiting for manual ] policy start...", "yellow"))
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

        self._send_7dof_policy_action()
        if (
            self._policy_start_wall_t is not None
            and time.perf_counter() - self._policy_start_wall_t < self.tracking_start_delay_s
        ):
            return
        self._maybe_resample_target()
        self._record_error(robot_state_data)

    def _current_right_fake_ee_base(self, robot_state_data):
        full_q = robot_state_data[0, 7 : 7 + self.num_dofs]
        upper_q = full_q[self.full_upper_joint_indices]
        model = self.upper_body_controller.reduced_robot.model
        data = self.upper_body_controller.data
        pin.framesForwardKinematics(model, data, upper_q)
        pin.updateFramePlacements(model, data)
        return np.array(data.oMf[self.right_ee_frame_id].translation, dtype=float).reshape(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Right hand-pad 7-DoF tracking test")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--marker_file", type=str, default="/tmp/falcon_ee_tracking_7dof_markers.json")
    parser.add_argument("--log_file", type=str, default="/tmp/falcon_ee_tracking_7dof_metrics.csv")
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
    parser.add_argument("--sim_status_file", type=str, default="/tmp/falcon_sim_status_7dof.json")
    parser.add_argument("--sim_control_file", type=str, default="/tmp/falcon_sim_control_7dof.json")
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

    policy = LocoManipEETracking7DofTestPolicy(
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
    )
    policy.run()
