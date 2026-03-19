import sys
import argparse
import yaml

import numpy as np
import pinocchio as pin
from termcolor import colored


sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from sim2real.utils.arm_ik.robot_arm_ik_g1_23dof import G1_29_ArmIK_NoWrists


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def _axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    K = _skew(axis)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class LocoManipValveTurnPolicy(DecLocomotionPolicy):
    def __init__(self, config, model_path, rl_rate=50, policy_action_scale=0.25):
        # 本类自己保存时间步，不依赖基类是否存 rl_rate
        self.turn_dt = 1.0 / float(rl_rate)

        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        self.residual_upper_body_action = self.config.get("residual_upper_body_action", False)

        self.upper_body_controller = None

        # ===== right-hand EE target =====
        self.right_degrees = 0.0
        self.left_degrees = 0.0

        # ===== task config =====
        self.turn_target_deg = float(self.config.get("valve_turn_target_deg", 60.0))
        self.turn_speed_deg = float(self.config.get("valve_turn_speed_deg", 20.0))

        # ===== active valve turning state =====
        self.turn_mode_enable = False
        self.turn_cmd_delta_deg = 0.0
        self.turn_direction = 1.0
        self.turn_done = False

        # 第一版近似：阀门轴在 IK 局部目标系中取 +x
        self.turn_axis_local = np.array([1.0, 0.0, 0.0], dtype=float)

        # weld 建立、且手已对准后记录
        self.turn_geom_valid = False
        self.turn_center_local = None
        self.turn_r0_local = None
        self.turn_right_R0 = None

        if self.config.get("use_upper_body_controller", False):
            self.init_upper_body_controller()

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_policy_action
        current_obs_dict["command_base_height"] = self.base_height_command
        return current_obs_dict

    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action[:, self.upper_dof_indices] += (
                self.ref_upper_dof_pos - self.default_dof_angles[self.upper_dof_indices]
            )

        return scaled_policy_action

    def init_upper_body_controller(self):
        if self.config["ROBOT_TYPE"] == "g1_29dof":
            self.upper_body_controller = G1_29_ArmIK_NoWrists(
                Unit_Test=False,
                Visualization=False,
                robot_config=self.config,
            )
        else:
            self.logger.error("Unsupported robot type: %s", self.config["ROBOT_TYPE"])

        theta_l = np.radians(self.left_degrees)
        theta_r = np.radians(self.right_degrees)

        self.EE_left_R = np.array(
            [
                [np.cos(theta_l), -np.sin(theta_l), 0],
                [np.sin(theta_l), np.cos(theta_l), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        self.EE_right_R = np.array(
            [
                [np.cos(theta_r), -np.sin(theta_r), 0],
                [np.sin(theta_r), np.cos(theta_r), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        # 保持和你当前 loco_manip 一致的默认末端初值
        self.EE_left_x = 0.30
        self.EE_left_y = 0.13
        self.EE_left_z = 0.08

        self.EE_right_x = 0.30
        self.EE_right_y = -0.13
        self.EE_right_z = 0.08

        self.update_waypoints()

        self.EE_efrc_L = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        self.EE_efrc_R = np.array([0, 0, 0, 0, 0, 0], dtype=float)

        self.upper_body_controller.set_initial_poses(
            self.waypoints_left[0].translation,
            self.waypoints_right[0].translation,
            self.waypoints_left[0].rotation,
            self.waypoints_right[0].rotation,
        )

    def update_waypoints(self):
        self.waypoints_left = [
            pin.SE3(
                self.EE_left_R.astype(np.float64),
                np.array([self.EE_left_x, self.EE_left_y, self.EE_left_z], dtype=np.float64),
            )
        ]
        self.waypoints_right = [
            pin.SE3(
                self.EE_right_R.astype(np.float64),
                np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z], dtype=np.float64),
            )
        ]

    def capture_turn_geometry(self):
        """
        第一版近似：
        - 当前右手已经通过前序流程对准阀门锚点
        - 阀门轴在 IK 局部目标系中近似为 +x
        - 使用 XML 中 right_hand_valve_anchor 相对 valve_center_site 的偏移
          r0 = [0, 0.084, -0.112]
        """
        p0 = np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z], dtype=float)
        R0 = self.EE_right_R.copy()

        r0 = np.array([0.0, 0.084, -0.112], dtype=float)
        c0 = p0 - r0

        self.turn_center_local = c0
        self.turn_r0_local = r0
        self.turn_right_R0 = R0
        self.turn_geom_valid = True

        self.logger.info(
            colored(
                f"[VALVE TURN] capture geometry: center={self.turn_center_local}, r0={self.turn_r0_local}",
                "cyan",
            )
        )

    def start_single_turn(self, direction=1.0):
        self.turn_mode_enable = True
        self.turn_done = False
        self.turn_direction = float(np.sign(direction)) if direction != 0 else 1.0
        self.turn_cmd_delta_deg = 0.0
        self.capture_turn_geometry()

        self.logger.info(
            colored(
                f"[VALVE TURN] start: target={self.turn_target_deg:.1f} deg, "
                f"speed={self.turn_speed_deg:.1f} deg/s, dir={self.turn_direction:+.0f}",
                "cyan",
            )
        )

    def update_active_turn_target(self):
        if not self.turn_mode_enable or not self.turn_geom_valid:
            return

        target_deg = self.turn_target_deg * self.turn_direction
        step_deg = self.turn_speed_deg * self.turn_dt

        err = target_deg - self.turn_cmd_delta_deg
        if abs(err) <= step_deg:
            self.turn_cmd_delta_deg = target_deg
            self.turn_mode_enable = False
            self.turn_done = True
        else:
            self.turn_cmd_delta_deg += np.sign(err) * step_deg

        angle = np.deg2rad(self.turn_cmd_delta_deg)
        R_delta = _axis_angle_to_rot(self.turn_axis_local, angle)

        # 位置和姿态都按固连点轨迹更新
        p_des = self.turn_center_local + R_delta @ self.turn_r0_local
        R_des = R_delta @ self.turn_right_R0

        self.EE_right_x = float(p_des[0])
        self.EE_right_y = float(p_des[1])
        self.EE_right_z = float(p_des[2])
        self.EE_right_R = R_des

        self.right_degrees = self.turn_cmd_delta_deg
        self.update_waypoints()

        if self.turn_done:
            self.logger.info(
                colored(
                    f"[VALVE TURN] finished at {self.turn_cmd_delta_deg:.1f} deg",
                    "green",
                )
            )

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)

        robot_state_data = self.state_processor.robot_state_data

        if self.upper_body_controller:
            self.update_active_turn_target()

            upper_body_qpos, _ = self.upper_body_controller.get_q_tau(
                self.waypoints_left[0],
                self.waypoints_right[0],
                self.EE_efrc_L,
                self.EE_efrc_R,
            )

            arm_reduced_joint_indices = [0, 1, 2, 3, 7, 8, 9, 10]
            for i, idx in enumerate(arm_reduced_joint_indices):
                self.ref_upper_dof_pos[0, idx] = upper_body_qpos[i]

            wrist_joint_indices = [19, 20, 21, 26, 27, 28]
            for idx in wrist_joint_indices:
                self.ref_upper_dof_pos[0, idx - 15] = 0.0

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

        cmd_q = q_target[0]
        self.command_sender.send_command(
            cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs]
        )

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)

        # 保留 locomotion / base 的必要控制
        if keycode == ",":
            self.waist_dofs_command[:, 0] -= 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
            return

        if keycode == ".":
            self.waist_dofs_command[:, 0] += 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
            return

        if keycode == "m":
            self.lin_vel_command[:, 0] = -1.0
            self.logger.info(colored(f"lin_vel_command: {self.lin_vel_command}", "green"))
            return

        if keycode in ["1", "2"]:
            self._handle_base_height_control(keycode)
            return

        # ===== valve turn task keys =====
        if keycode == "l":
            self.start_single_turn(direction=+1.0)
            return

        if keycode == ";":
            self.start_single_turn(direction=-1.0)
            return

    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)

        if cur_key in ["B+up", "B+down"]:
            self._handle_joystick_base_height_control(cur_key)

        if cur_key == "Y+up":
            self.waist_dofs_command[:, 2] -= 0.1
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "Y+down":
            self.waist_dofs_command[:, 2] += 0.1
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+left":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+right":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+up":
            self.waist_dofs_command[:, 2] -= 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+down":
            self.waist_dofs_command[:, 2] += 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "A+B":
            self.command_sender.kp_level = 1.0
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

    def _handle_base_height_control(self, keycode):
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1

    def _handle_joystick_base_height_control(self, cur_key):
        if cur_key == "B+up":
            self.base_height_command[0, 0] += 0.1
        elif cur_key == "B+down":
            self.base_height_command[0, 0] -= 0.1

    def _print_control_status(self):
        super()._print_control_status()
        print(f"Base height command: {self.base_height_command}")
        print(f"Waist dofs command: {self.waist_dofs_command}")
        print(f"Valve turn target deg: {self.turn_target_deg}")
        print(f"Valve turn speed deg/s: {self.turn_speed_deg}")
        print(f"Valve turn enable: {self.turn_mode_enable}")
        print(f"Valve turn current deg: {self.turn_cmd_delta_deg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valve-turn task policy")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = LocoManipValveTurnPolicy(
        config=config,
        model_path=model_path,
        rl_rate=50,
        policy_action_scale=0.25,
    )
    policy.run()