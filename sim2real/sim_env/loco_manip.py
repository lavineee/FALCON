import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)
import sys
import argparse
import json
import os
import time
from collections import deque
from multiprocessing import Process, Queue
from queue import Empty

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import yaml
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

sys.path.append("../")
sys.path.append("./sim2real")

from sim2real.sim_env.base_sim import BaseSimulator
from sim2real.utils.math import quat_rotate_numpy
from sim2real.utils.sdk2py_bridge import ElasticBand


class LocoManipSimulator(BaseSimulator):
    def __init__(self, config):
        # must be defined before super().__init__(), because init_scene() is called there
        self.EE_xfrc = 0
        self.t = 0.0

        self.left_hand_link_name = config.get("left_hand_link_name", "left_hand_link")
        self.right_hand_link_name = config.get("right_hand_link_name", "right_hand_link")

        # valve control
        self.valve_enable = False
        self.valve_tau = 0.0
        self.valve_tau_step = 1.0
        self.valve_tau_limit = 100.0

        # weld control
        self.weld_name = "right_hand_valve_weld"
        self.weld_enable = False
        self.weld_id = -1

        # valve ids
        self.valve_joint_name = "valve_hinge"
        self.valve_jnt_id = -1
        self.valve_dof = -1
        self.valve_qadr = -1

        # live plot
        self.enable_live_plot = bool(config.get("enable_live_plot", True)) and plt is not None
        self.plot_maxlen = 400
        self.plot_push_every = 1
        self.plot_counter = 0
        self.plot_queue = None
        self.plot_process = None

        self.auto_elastic_length = config.get("auto_elastic_length", None)
        self.auto_disable_elastic_after_s = config.get("auto_disable_elastic_after_s", None)
        self._auto_elastic_disabled = False
        self.sim_control_file = config.get("sim_control_file", None)
        self._last_control_timestamp = 0.0
        self.ee_marker_file = config.get("ee_marker_file", None)
        self.ee_marker_radius = float(config.get("ee_marker_radius", 0.025))

        super().__init__(config)

        if self.enable_live_plot:
            self.start_live_plot_process()

    @staticmethod
    def live_plot_worker(plot_queue, plot_maxlen):
        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        fig.canvas.manager.set_window_title("Valve Live Signals")

        hist_t = deque(maxlen=plot_maxlen)
        hist_tau = deque(maxlen=plot_maxlen)
        hist_q = deque(maxlen=plot_maxlen)
        hist_qd = deque(maxlen=plot_maxlen)

        line_tau, = axes[0].plot([], [], linewidth=2)
        line_q, = axes[1].plot([], [], linewidth=2)
        line_qd, = axes[2].plot([], [], linewidth=2)

        axes[0].set_ylabel("Torque (N·m)")
        axes[1].set_ylabel("Angle (rad)")
        axes[2].set_ylabel("Vel (rad/s)")
        axes[2].set_xlabel("Time (s)")

        axes[0].set_title("Valve Torque Cmd")
        axes[1].set_title("Valve Angle")
        axes[2].set_title("Valve Vel")

        for ax in axes:
            ax.grid(True)

        fig.tight_layout()
        fig.show()

        running = True
        while running:
            try:
                item = plot_queue.get(timeout=0.05)
                if item is None:
                    running = False
                    break

                t_now, torque_now, valve_angle, valve_vel = item
                hist_t.append(t_now)
                hist_tau.append(torque_now)
                hist_q.append(valve_angle)
                hist_qd.append(valve_vel)

                t = list(hist_t)
                tau = list(hist_tau)
                q = list(hist_q)
                qd = list(hist_qd)

                line_tau.set_data(t, tau)
                line_q.set_data(t, q)
                line_qd.set_data(t, qd)

                for ax in axes:
                    ax.relim()
                    ax.autoscale_view()

                fig.canvas.draw_idle()
                plt.pause(0.001)

            except Empty:
                plt.pause(0.001)

        plt.close(fig)

    def start_live_plot_process(self):
        self.plot_queue = Queue(maxsize=200)
        self.plot_process = Process(
            target=LocoManipSimulator.live_plot_worker,
            args=(self.plot_queue, self.plot_maxlen),
            daemon=True,
        )
        self.plot_process.start()

    def push_live_plot(self, t_now, torque_now, valve_angle, valve_vel):
        self.plot_counter += 1
        if self.plot_counter % self.plot_push_every != 0:
            return

        if self.plot_queue is None:
            return

        item = (t_now, torque_now, valve_angle, valve_vel)

        try:
            if self.plot_queue.full():
                try:
                    self.plot_queue.get_nowait()
                except Exception:
                    pass
            self.plot_queue.put_nowait(item)
        except Exception:
            pass

    def MujocoKeyCallback(self, key):
        # keep elastic band behavior
        if self.config["ENABLE_ELASTIC_BAND"] and self.elastic_band is not None:
            self.elastic_band.MujuocoKeyCallback(key)

        # J: toggle weld
        if key == glfw.KEY_J and self.weld_id != -1:
            self.weld_enable = not self.weld_enable
            state = 1 if self.weld_enable else 0

            self.mj_model.eq_active0[self.weld_id] = state
            if hasattr(self.mj_data, "eq_active"):
                self.mj_data.eq_active[self.weld_id] = state

            mujoco.mj_forward(self.mj_model, self.mj_data)
            print(f"[WELD] {'ON' if self.weld_enable else 'OFF'}")

        # K: toggle valve torque
        if key == glfw.KEY_K:
            self.valve_enable = not self.valve_enable
            print(f"[VALVE] {'ON' if self.valve_enable else 'OFF'}  tau={self.valve_tau:.2f}")

        # U: increase torque
        if key == glfw.KEY_U:
            self.valve_tau = min(self.valve_tau + self.valve_tau_step, self.valve_tau_limit)
            print(f"[VALVE] tau={self.valve_tau:.2f}")

        # I: decrease torque
        if key == glfw.KEY_I:
            self.valve_tau = max(self.valve_tau - self.valve_tau_step, -self.valve_tau_limit)
            print(f"[VALVE] tau={self.valve_tau:.2f}")

        # O: zero torque
        if key == glfw.KEY_O:
            self.valve_tau = 0.0
            print(f"[VALVE] tau={self.valve_tau:.2f}")

    def init_scene(self):
        # similar to BaseSimulator.init_scene(), but use our own key callback
        print(self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

        base_body_name = self.config.get("BASE_BODY_NAME", "pelvis")
        self.base_id = self.mj_model.body(base_body_name).id

        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            if self.auto_elastic_length is not None:
                self.elastic_band.length = float(self.auto_elastic_length)
                print(f"[AUTO] elastic length={self.elastic_band.length:.3f}")
            band_attached_link_name = self.config.get("BAND_ATTACHED_LINK", "torso_link")
            self.band_attached_link = self.mj_model.body(band_attached_link_name).id
        else:
            self.elastic_band = None

        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, key_callback=self.MujocoKeyCallback
        )

        NUM_FEET_SENSORS = 8
        self.ffss_idx = len(self.mj_data.sensordata) - NUM_FEET_SENSORS * 3
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        # valve ids
        self.valve_jnt_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, self.valve_joint_name
        )
        if self.valve_jnt_id != -1:
            self.valve_dof = self.mj_model.jnt_dofadr[self.valve_jnt_id]
            self.valve_qadr = self.mj_model.jnt_qposadr[self.valve_jnt_id]
        print(
            f"[VALVE] joint={self.valve_joint_name}, jnt_id={self.valve_jnt_id}, "
            f"dof={self.valve_dof}, qadr={self.valve_qadr}"
        )

        # weld id
        self.weld_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, self.weld_name
        )
        print(f"[WELD] name={self.weld_name}, id={self.weld_id}")

        # force initial OFF
        if self.weld_id != -1:
            self.mj_model.eq_active0[self.weld_id] = 0
            if hasattr(self.mj_data, "eq_active"):
                self.mj_data.eq_active[self.weld_id] = 0

    def _base_to_world(self, pos_base):
        pos_base = np.asarray(pos_base, dtype=float)
        base_pos = self.mj_data.xpos[self.base_id].copy()
        base_rot = self.mj_data.xmat[self.base_id].reshape(3, 3).copy()
        return base_pos + base_rot @ pos_base

    def _draw_ee_tracking_markers(self):
        if not self.ee_marker_file or not hasattr(self.viewer, "user_scn"):
            return
        if not os.path.exists(self.ee_marker_file):
            return
        try:
            with open(self.ee_marker_file, "r") as file:
                marker_data = json.load(file)
            target_base = marker_data.get("target_right_base")
            current_base = marker_data.get("current_right_base")
            if target_base is None or current_base is None:
                return

            geoms = self.viewer.user_scn.geoms
            self.viewer.user_scn.ngeom = 0
            for pos_base, rgba in (
                (target_base, np.array([0.0, 0.8, 0.1, 0.75], dtype=float)),
                (current_base, np.array([0.9, 0.1, 0.1, 0.75], dtype=float)),
            ):
                geom_id = self.viewer.user_scn.ngeom
                if geom_id >= len(geoms):
                    break
                mujoco.mjv_initGeom(
                    geoms[geom_id],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([self.ee_marker_radius, 0.0, 0.0], dtype=float),
                    self._base_to_world(pos_base),
                    np.eye(3).reshape(-1),
                    rgba,
                )
                self.viewer.user_scn.ngeom += 1
        except Exception as exc:
            if not hasattr(self, "_ee_marker_warned"):
                print(f"[EE_MARKER] disabled after read/draw error: {exc}")
                self._ee_marker_warned = True

    def _maybe_disable_elastic_from_control_file(self):
        if (
            not self.config["ENABLE_ELASTIC_BAND"]
            or self.elastic_band is None
            or self._auto_elastic_disabled
            or not self.sim_control_file
            or not os.path.exists(self.sim_control_file)
        ):
            return
        try:
            with open(self.sim_control_file, "r") as file:
                control = json.load(file)
        except Exception:
            return

        timestamp = float(control.get("timestamp", 0.0))
        if timestamp <= self._last_control_timestamp:
            return
        if not control.get("policy_started", False):
            self._last_control_timestamp = timestamp
            return

        delay_value = control.get("disable_elastic_after_policy_start_s", None)
        if delay_value is None:
            return
        delay_s = float(delay_value)
        if delay_s < 0.0:
            return
        if time.time() - timestamp < delay_s:
            return

        self.elastic_band.enable = False
        self._auto_elastic_disabled = True
        self._last_control_timestamp = timestamp
        if hasattr(self, "band_attached_link"):
            self.mj_data.xfrc_applied[self.band_attached_link, :3] = 0.0
        print(f"[AUTO] elastic band OFF after policy start + {delay_s:.2f}s")

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()

        self._maybe_disable_elastic_from_control_file()

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
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
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
        if self.robot_bridge.free_base:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques

        # clear external generalized force
        self.mj_data.qfrc_applied[:] = 0.0

        # apply valve torque only when enabled
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
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = LocoManipSimulator(config)
    simulation.sim_thread.start()

#v1.1 不带画图    
# import sys
# import argparse

# import glfw
# import mujoco
# import mujoco.viewer
# import numpy as np
# import yaml

# sys.path.append("../")
# sys.path.append("./sim2real")

# from sim2real.sim_env.base_sim import BaseSimulator
# from sim2real.utils.math import quat_rotate_numpy
# from sim2real.utils.sdk2py_bridge import ElasticBand


# class LocoManipSimulator(BaseSimulator):
#     def __init__(self, config):
#         # must be defined before super().__init__(), because init_scene() is called there
#         self.EE_xfrc = 0
#         self.t = 0.0

#         self.left_hand_link_name = config.get("left_hand_link_name", "left_hand_link")
#         self.right_hand_link_name = config.get("right_hand_link_name", "right_hand_link")

#         # valve control
#         self.valve_enable = False
#         self.valve_tau = 0.0
#         self.valve_tau_step = 1.0
#         self.valve_tau_limit = 100

#         # weld control
#         self.weld_name = "right_hand_valve_weld"
#         self.weld_enable = False
#         self.weld_id = -1

#         # valve ids
#         self.valve_joint_name = "valve_hinge"
#         self.valve_jnt_id = -1
#         self.valve_dof = -1
#         self.valve_qadr = -1

#         super().__init__(config)

#     def MujocoKeyCallback(self, key):
#         # keep elastic band behavior
#         if self.config["ENABLE_ELASTIC_BAND"] and self.elastic_band is not None:
#             self.elastic_band.MujuocoKeyCallback(key)

#         # J: toggle weld
#         if key == glfw.KEY_J and self.weld_id != -1:
#             self.weld_enable = not self.weld_enable
#             state = 1 if self.weld_enable else 0

#             self.mj_model.eq_active0[self.weld_id] = state
#             if hasattr(self.mj_data, "eq_active"):
#                 self.mj_data.eq_active[self.weld_id] = state

#             mujoco.mj_forward(self.mj_model, self.mj_data)
#             print(f"[WELD] {'ON' if self.weld_enable else 'OFF'}")

#         # K: toggle valve torque
#         if key == glfw.KEY_K:
#             self.valve_enable = not self.valve_enable
#             print(f"[VALVE] {'ON' if self.valve_enable else 'OFF'}  tau={self.valve_tau:.2f}")

#         # U: increase torque
#         if key == glfw.KEY_U:
#             self.valve_tau = min(self.valve_tau + self.valve_tau_step, self.valve_tau_limit)
#             print(f"[VALVE] tau={self.valve_tau:.2f}")

#         # I: decrease torque
#         if key == glfw.KEY_I:
#             self.valve_tau = max(self.valve_tau - self.valve_tau_step, -self.valve_tau_limit)
#             print(f"[VALVE] tau={self.valve_tau:.2f}")

#         # O: zero torque
#         if key == glfw.KEY_O:
#             self.valve_tau = 0.0
#             print(f"[VALVE] tau={self.valve_tau:.2f}")

#     def init_scene(self):
#         # similar to BaseSimulator.init_scene(), but use our own key callback
#         print(self.config["ROBOT_SCENE"])
#         self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
#         self.mj_data = mujoco.MjData(self.mj_model)
#         self.mj_model.opt.timestep = self.sim_dt

#         base_body_name = self.config.get("BASE_BODY_NAME", "pelvis")
#         self.base_id = self.mj_model.body(base_body_name).id

#         if self.config["ENABLE_ELASTIC_BAND"]:
#             self.elastic_band = ElasticBand()
#             band_attached_link_name = self.config.get("BAND_ATTACHED_LINK", "torso_link")
#             self.band_attached_link = self.mj_model.body(band_attached_link_name).id
#         else:
#             self.elastic_band = None

#         self.viewer = mujoco.viewer.launch_passive(
#             self.mj_model, self.mj_data, key_callback=self.MujocoKeyCallback
#         )

#         NUM_FEET_SENSORS = 8
#         self.ffss_idx = len(self.mj_data.sensordata) - NUM_FEET_SENSORS * 3
#         self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
#         self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

#         # valve ids
#         self.valve_jnt_id = mujoco.mj_name2id(
#             self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, self.valve_joint_name
#         )
#         self.valve_dof = self.mj_model.jnt_dofadr[self.valve_jnt_id]
#         self.valve_qadr = self.mj_model.jnt_qposadr[self.valve_jnt_id]
#         print(
#             f"[VALVE] joint={self.valve_joint_name}, jnt_id={self.valve_jnt_id}, "
#             f"dof={self.valve_dof}, qadr={self.valve_qadr}"
#         )

#         # weld id
#         self.weld_id = mujoco.mj_name2id(
#             self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, self.weld_name
#         )
#         print(f"[WELD] name={self.weld_name}, id={self.weld_id}")

#         # force initial OFF
#         if self.weld_id != -1:
#             self.mj_model.eq_active0[self.weld_id] = 0
#             if hasattr(self.mj_data, "eq_active"):
#                 self.mj_data.eq_active[self.weld_id] = 0

#     def sim_step(self):
#         self.robot_bridge.PublishLowState()
#         if self.robot_bridge.joystick:
#             self.robot_bridge.PublishWirelessController()

#         if self.config["ENABLE_ELASTIC_BAND"]:
#             if self.elastic_band.enable:
#                 self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
#                     self.mj_data.qpos[:3], self.mj_data.qvel[:3]
#                 )

#         if self.config["ENABLE_ELASTIC_BAND"] and self.elastic_band.estimate:
#             self.EE_xfrc = self.elastic_band.apply_force
#             base_target_axis = np.array([-1.0, 0.0, 0.0])
#             body_quat = self.mj_data.qpos[3:7]
#             force_in_global = quat_rotate_numpy(np.array([body_quat]), base_target_axis * self.EE_xfrc)
#             self.mj_data.xfrc_applied[self.mj_model.body(self.left_hand_link_name).id, 0:3] = force_in_global
#             self.mj_data.xfrc_applied[self.mj_model.body(self.right_hand_link_name).id, 0:3] = force_in_global

#         self.compute_torques()
#         if self.robot_bridge.free_base:
#             self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
#         else:
#             self.mj_data.ctrl = self.torques

#         # clear external generalized force
#         self.mj_data.qfrc_applied[:] = 0.0

#         # apply valve torque only when enabled
#         if self.valve_enable and self.valve_dof != -1:
#             self.mj_data.qfrc_applied[self.valve_dof] = self.valve_tau

#         mujoco.mj_step(self.mj_model, self.mj_data)
#         self.t += self.sim_dt


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Robot")
#     parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
#     args = parser.parse_args()

#     with open(args.config) as file:
#         config = yaml.safe_load(file)

#     simulation = LocoManipSimulator(config)
#     simulation.sim_thread.start()
