import argparse
import json
import os
import sys
import time

import glfw
import mujoco
import numpy as np
import yaml

sys.path.append("../")
sys.path.append("./sim2real")

from sim2real.sim_env.loco_manip import LocoManipSimulator
from sim2real.utils.math import quat_rotate_numpy
from sim2real.utils.sdk2py_bridge.unitree.unitree_sdk2py_bridge import UnitreeSdk2Bridge


class NameMappedUnitreeSdk2Bridge(UnitreeSdk2Bridge):
    """Unitree bridge that exposes only the 29 policy joints from a larger MuJoCo model."""

    def __init__(self, mj_model, mj_data, robot_config):
        self.policy_joint_names = list(robot_config.get("policy_motor_joint_names", robot_config["dof_names"]))
        super().__init__(mj_model, mj_data, robot_config)
        if self.use_sensor:
            raise ValueError("The with-hand name-mapped bridge currently expects USE_SENSOR=False.")

        self.num_motor = len(self.policy_joint_names)
        self.torques = np.zeros(self.num_motor)
        self.torque_limit = np.asarray(self.robot.MOTOR_EFFORT_LIMIT_LIST, dtype=float)
        self._build_policy_joint_maps()

    def _joint_id(self, joint_name):
        joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        return joint_id

    def _actuator_id_for_joint(self, joint_name, joint_id):
        actuator_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if actuator_id != -1 and int(self.mj_model.actuator_trnid[actuator_id, 0]) == joint_id:
            return actuator_id
        for i in range(self.mj_model.nu):
            if int(self.mj_model.actuator_trnid[i, 0]) == joint_id:
                return i
        raise ValueError(f"MuJoCo actuator not found for joint: {joint_name}")

    def _build_policy_joint_maps(self):
        self.policy_joint_ids = np.array([self._joint_id(name) for name in self.policy_joint_names], dtype=int)
        self.policy_qpos_adrs = self.mj_model.jnt_qposadr[self.policy_joint_ids].astype(int)
        self.policy_qvel_adrs = self.mj_model.jnt_dofadr[self.policy_joint_ids].astype(int)
        self.policy_actuator_ids = np.array(
            [
                self._actuator_id_for_joint(name, joint_id)
                for name, joint_id in zip(self.policy_joint_names, self.policy_joint_ids)
            ],
            dtype=int,
        )
        print("[WITH_HAND] policy joint -> actuator map")
        for i, (name, actuator_id) in enumerate(zip(self.policy_joint_names, self.policy_actuator_ids)):
            print(f"  policy[{i:02d}] {name} -> ctrl[{actuator_id}]")

    def PublishLowState(self):
        if self.mj_data is None:
            return

        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        qacc = self.mj_data.qacc
        actuator_force = self.mj_data.actuator_force
        imu = self.low_state.imu_state
        motor_state = self.low_state.motor_state

        for i in range(self.num_motor):
            qadr = self.policy_qpos_adrs[i]
            dadr = self.policy_qvel_adrs[i]
            actuator_id = self.policy_actuator_ids[i]
            m = motor_state[i]
            m.q = qpos[qadr]
            m.dq = qvel[dadr]
            m.ddq = qacc[dadr]
            m.tau_est = actuator_force[actuator_id]

        imu.quaternion[:] = qpos[3:7]
        imu.gyroscope[:] = qvel[3:6]
        imu.accelerometer[:] = qacc[0:3]

        self.low_state.tick = int(self.mj_data.time * 1e3)
        self.low_state.crc = self.crc.Crc(self.low_state)
        self.low_state_puber.Write(self.low_state)


class LocoManipWithHandSimulator(LocoManipSimulator):
    """Loco-manip simulator for the 43-actuator G1 with hands.

    The RL policy remains 29-DoF. The extra hand actuators are driven by a
    simple open/close PD controller.
    """

    def init_robot_bridge(self):
        self.robot_bridge = NameMappedUnitreeSdk2Bridge(self.mj_model, self.mj_data, self.config)
        if self.config["USE_JOYSTICK"]:
            if sys.platform == "linux":
                self.robot_bridge.SetupJoystick(
                    device_id=self.config["JOYSTICK_DEVICE"],
                    js_type=self.config["JOYSTICK_TYPE"],
                )
            else:
                self.logger.warning("Joystick is not supported on Windows or MacOS.")
        self.policy_actuator_ids = self.robot_bridge.policy_actuator_ids
        self.policy_qpos_adrs = self.robot_bridge.policy_qpos_adrs
        self.policy_qvel_adrs = self.robot_bridge.policy_qvel_adrs
        self._stabilize_policy_joint_model()
        self._init_hand_controller()

    def _stabilize_policy_joint_model(self):
        joint_damping = float(self.config.get("policy_joint_damping", 0.05))
        joint_armature = float(self.config.get("policy_joint_armature", 0.01))
        joint_frictionloss = float(self.config.get("policy_joint_frictionloss", 0.2))
        wrist_frictionloss = float(self.config.get("wrist_joint_frictionloss", 0.1))
        for name, dof_adr in zip(self.robot_bridge.policy_joint_names, self.policy_qvel_adrs):
            self.mj_model.dof_damping[dof_adr] = max(self.mj_model.dof_damping[dof_adr], joint_damping)
            self.mj_model.dof_armature[dof_adr] = max(self.mj_model.dof_armature[dof_adr], joint_armature)
            frictionloss = wrist_frictionloss if "wrist" in name else joint_frictionloss
            self.mj_model.dof_frictionloss[dof_adr] = max(self.mj_model.dof_frictionloss[dof_adr], frictionloss)
        print(
            f"[WITH_HAND] policy joint stabilization: damping={joint_damping:.3f}, "
            f"armature={joint_armature:.3f}, friction={joint_frictionloss:.3f}"
        )

    def _resolve_joint_actuator_maps(self, joint_names):
        joint_ids = []
        qpos_adrs = []
        qvel_adrs = []
        actuator_ids = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"MuJoCo joint not found: {name}")
            actuator_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id == -1:
                for i in range(self.mj_model.nu):
                    if int(self.mj_model.actuator_trnid[i, 0]) == joint_id:
                        actuator_id = i
                        break
            if actuator_id == -1:
                raise ValueError(f"MuJoCo actuator not found for joint: {name}")
            joint_ids.append(joint_id)
            qpos_adrs.append(int(self.mj_model.jnt_qposadr[joint_id]))
            qvel_adrs.append(int(self.mj_model.jnt_dofadr[joint_id]))
            actuator_ids.append(actuator_id)
        return {
            "joint_names": list(joint_names),
            "joint_ids": np.asarray(joint_ids, dtype=int),
            "qpos_adrs": np.asarray(qpos_adrs, dtype=int),
            "qvel_adrs": np.asarray(qvel_adrs, dtype=int),
            "actuator_ids": np.asarray(actuator_ids, dtype=int),
        }

    def _init_hand_controller(self):
        self.left_hand_joint_names = list(self.config.get("left_hand_joint_names", []))
        self.right_hand_joint_names = list(self.config.get("right_hand_joint_names", []))
        self.left_hand_map = self._resolve_joint_actuator_maps(self.left_hand_joint_names)
        self.right_hand_map = self._resolve_joint_actuator_maps(self.right_hand_joint_names)

        self.left_hand_open_q = np.asarray(self.config.get("left_hand_open_q", np.zeros(7)), dtype=float)
        self.left_hand_close_q = np.asarray(self.config.get("left_hand_close_q", np.zeros(7)), dtype=float)
        self.right_hand_open_q = np.asarray(self.config.get("right_hand_open_q", np.zeros(7)), dtype=float)
        self.right_hand_close_q = np.asarray(self.config.get("right_hand_close_q", np.zeros(7)), dtype=float)
        self.left_hand_effort_limits = np.asarray(self.config.get("left_hand_effort_limits", np.ones(7)), dtype=float)
        self.right_hand_effort_limits = np.asarray(self.config.get("right_hand_effort_limits", np.ones(7)), dtype=float)

        self.hand_kp = float(self.config.get("hand_kp", 3.0))
        self.hand_kd = float(self.config.get("hand_kd", 0.08))
        self.left_hand_state = self.config.get("left_hand_default_state", "open")
        self.right_hand_state = self.config.get("right_hand_default_state", "open")
        self._last_hand_control_timestamp = 0.0
        self._stabilize_hand_model()
        print(
            f"[WITH_HAND] hand controller: left={self.left_hand_state}, right={self.right_hand_state}, "
            f"kp={self.hand_kp:.2f}, kd={self.hand_kd:.2f}"
        )

    def _stabilize_hand_model(self):
        hand_dof_adrs = np.unique(
            np.concatenate((self.left_hand_map["qvel_adrs"], self.right_hand_map["qvel_adrs"]))
        )
        hand_damping = float(self.config.get("hand_joint_damping", 0.2))
        hand_armature = float(self.config.get("hand_joint_armature", 0.003))
        for dof_adr in hand_dof_adrs:
            self.mj_model.dof_damping[dof_adr] = max(self.mj_model.dof_damping[dof_adr], hand_damping)
            self.mj_model.dof_armature[dof_adr] = max(self.mj_model.dof_armature[dof_adr], hand_armature)

        disabled_geoms = 0
        if self.config.get("disable_hand_finger_collision_for_tracking", False):
            for geom_id in range(self.mj_model.ngeom):
                body_id = int(self.mj_model.geom_bodyid[geom_id])
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name and "_hand_" in body_name:
                    self.mj_model.geom_contype[geom_id] = 0
                    self.mj_model.geom_conaffinity[geom_id] = 0
                    disabled_geoms += 1
        print(
            f"[WITH_HAND] hand stabilization: damping={hand_damping:.3f}, "
            f"armature={hand_armature:.4f}, disabled_finger_geoms={disabled_geoms}"
        )

    def _target_for_state(self, side, state):
        if side == "left":
            return self.left_hand_close_q if state == "close" else self.left_hand_open_q
        return self.right_hand_close_q if state == "close" else self.right_hand_open_q

    def _compute_hand_torque(self, hand_map, target_q, effort_limits):
        q = self.mj_data.qpos[hand_map["qpos_adrs"]]
        dq = self.mj_data.qvel[hand_map["qvel_adrs"]]
        torque = self.hand_kp * (target_q - q) - self.hand_kd * dq
        return np.clip(torque, -effort_limits, effort_limits)

    def _maybe_update_hand_state_from_control_file(self):
        if not self.sim_control_file or not os.path.exists(self.sim_control_file):
            return
        try:
            with open(self.sim_control_file, "r") as file:
                control = json.load(file)
        except Exception:
            return

        timestamp = float(control.get("timestamp", 0.0))
        if timestamp <= self._last_hand_control_timestamp:
            return

        updated = False
        right_state = control.get("right_hand_state", None)
        left_state = control.get("left_hand_state", None)
        if right_state in ("open", "close") and right_state != self.right_hand_state:
            self.right_hand_state = right_state
            updated = True
        if left_state in ("open", "close") and left_state != self.left_hand_state:
            self.left_hand_state = left_state
            updated = True
        if updated:
            print(f"[WITH_HAND] hand state: left={self.left_hand_state}, right={self.right_hand_state}")

        self._last_hand_control_timestamp = timestamp

    def MujocoKeyCallback(self, key):
        super().MujocoKeyCallback(key)
        if key == glfw.KEY_H:
            self.right_hand_state = "close" if self.right_hand_state == "open" else "open"
            print(f"[WITH_HAND] right hand {self.right_hand_state}")

    def compute_torques(self):
        if self.robot_bridge.low_cmd:
            motor_cmd = list(self.robot_bridge.low_cmd.motor_cmd)
            try:
                for i in range(self.robot_bridge.num_motor):
                    self.torques[i] = (
                        motor_cmd[i].tau
                        + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[self.policy_qpos_adrs[i]])
                        + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[self.policy_qvel_adrs[i]])
                    )
            except Exception as exc:
                self.logger.error(f"Failed to compute mapped torque for policy motor {i}: {exc}")
        self.torques = np.clip(self.torques, -self.robot_bridge.torque_limit, self.robot_bridge.torque_limit)

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()

        self._maybe_disable_elastic_from_control_file()
        self._maybe_update_hand_state_from_control_file()

        if (
            self.config["ENABLE_ELASTIC_BAND"]
            and self.elastic_band is not None
            and self.auto_disable_elastic_after_s is not None
            and not self._auto_elastic_disabled
            and self.t >= float(self.auto_disable_elastic_after_s)
        ):
            self.elastic_band.enable = False
            self._auto_elastic_disabled = True
            if hasattr(self, "band_attached_link"):
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = 0.0
            print("[AUTO] elastic band OFF")

        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3],
                    self.mj_data.qvel[:3],
                )
            else:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = 0.0

        if self.config["ENABLE_ELASTIC_BAND"] and self.elastic_band.estimate:
            self.EE_xfrc = self.elastic_band.apply_force
            base_target_axis = np.array([-1.0, 0.0, 0.0])
            body_quat = self.mj_data.qpos[3:7]
            force_in_global = quat_rotate_numpy(np.array([body_quat]), base_target_axis * self.EE_xfrc)
            self.mj_data.xfrc_applied[self.mj_model.body(self.left_hand_link_name).id, 0:3] = force_in_global
            self.mj_data.xfrc_applied[self.mj_model.body(self.right_hand_link_name).id, 0:3] = force_in_global

        self.compute_torques()
        ctrl = np.zeros(self.mj_model.nu)
        ctrl[self.policy_actuator_ids] = self.torques
        ctrl[self.left_hand_map["actuator_ids"]] = self._compute_hand_torque(
            self.left_hand_map,
            self._target_for_state("left", self.left_hand_state),
            self.left_hand_effort_limits,
        )
        ctrl[self.right_hand_map["actuator_ids"]] = self._compute_hand_torque(
            self.right_hand_map,
            self._target_for_state("right", self.right_hand_state),
            self.right_hand_effort_limits,
        )
        self.mj_data.ctrl[:] = ctrl

        self.mj_data.qfrc_applied[:] = 0.0
        if self.valve_enable and self.valve_dof != -1:
            self.mj_data.qfrc_applied[self.valve_dof] = self.valve_tau

        mujoco.mj_step(self.mj_model, self.mj_data)

        torque_now = 0.0
        valve_angle = 0.0
        valve_vel = 0.0
        if self.valve_dof != -1:
            torque_now = float(self.mj_data.qfrc_applied[self.valve_dof])
            valve_vel = float(self.mj_data.qvel[self.valve_dof])
        if self.valve_qadr != -1:
            valve_angle = float(self.mj_data.qpos[self.valve_qadr])
        if self.enable_live_plot:
            self.push_live_plot(self.t, torque_now, valve_angle, valve_vel)

        self.t += self.sim_dt
        self._draw_ee_tracking_markers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 with-hand loco-manip simulator")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml", help="config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = LocoManipWithHandSimulator(config)
    simulation.sim_thread.start()
