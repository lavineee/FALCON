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
        self.valve_angle_hold_enabled = bool(config.get("valve_angle_hold_enabled", False))
        self.valve_angle_hold_kp = float(config.get("valve_angle_hold_kp", 0.0))
        self.valve_angle_hold_kd = float(config.get("valve_angle_hold_kd", 0.0))
        self.valve_angle_hold_torque_limit = float(config.get("valve_angle_hold_torque_limit", 0.0))
        self.valve_angle_hold_target = config.get("valve_angle_hold_target", None)
        self.valve_angle_hold_tau = 0.0
        self.valve_angle_lock_enabled = bool(config.get("valve_angle_lock_enabled", False))

        # valve hand attachment control. Keep the old "weld" field names for
        # compatibility with earlier control files, but the equality can also
        # be a softer connect constraint.
        self.weld_name = config.get(
            "valve_attach_equality_name",
            config.get("weld_name", "right_hand_valve_weld"),
        )
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
        self.sim_status_file = config.get("sim_status_file", None)
        self.sim_status_write_interval_s = float(config.get("sim_status_write_interval_s", 0.05))
        self._last_status_write_t = -1e9
        self.sim_control_file = config.get("sim_control_file", None)
        self._last_control_timestamp = 0.0
        self._last_weld_control_timestamp = 0.0
        self._last_valve_angle_control_timestamp = 0.0
        self.ee_marker_file = config.get("ee_marker_file", None)
        self.ee_marker_radius = float(config.get("ee_marker_radius", 0.025))
        self.ee_marker_visual_mode = str(config.get("ee_marker_visual_mode", "full")).lower()
        self.viewer_camera = config.get("viewer_camera", None)

        self.valve_center_site_name = config.get("valve_center_site_name", "valve_center_site")
        self.valve_grasp_site_name = config.get("valve_grasp_site_name", "right_hand_valve_site")
        self.valve_wheel_body_name = config.get("valve_wheel_body_name", "valve_wheel")
        self.right_contact_grasp_proxy_site_name = config.get(
            "right_contact_grasp_proxy_site_name", "right_contact_grasp_proxy_site"
        )
        self.valve_center_site_id = -1
        self.valve_grasp_site_id = -1
        self.right_contact_grasp_proxy_site_id = -1
        self.valve_wheel_body_id = -1
        self.right_hand_contact_body_prefixes = tuple(
            config.get("right_hand_contact_body_prefixes", ["right_hand_"])
        )
        self.right_hand_contact_body_names = set(
            config.get("right_hand_contact_body_names", [])
        )

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

        # J: toggle valve hand attachment
        if key == glfw.KEY_J and self.weld_id != -1:
            self._set_weld_enabled(not self.weld_enable)

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
        self._apply_viewer_camera()

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
            valve_damping = self.config.get("valve_joint_damping", None)
            valve_frictionloss = self.config.get("valve_joint_frictionloss", None)
            if valve_damping is not None:
                self.mj_model.dof_damping[self.valve_dof] = float(valve_damping)
            if valve_frictionloss is not None:
                self.mj_model.dof_frictionloss[self.valve_dof] = float(valve_frictionloss)
        print(
            f"[VALVE] joint={self.valve_joint_name}, jnt_id={self.valve_jnt_id}, "
            f"dof={self.valve_dof}, qadr={self.valve_qadr}"
        )
        if self.valve_dof != -1:
            print(
                f"[VALVE] damping={self.mj_model.dof_damping[self.valve_dof]:.4f}, "
                f"frictionloss={self.mj_model.dof_frictionloss[self.valve_dof]:.4f}"
            )
            if self.valve_angle_hold_target is None:
                self.valve_angle_hold_target = float(self.mj_data.qpos[self.valve_qadr])
            else:
                self.valve_angle_hold_target = float(self.valve_angle_hold_target)
            print(
                f"[VALVE_HOLD] enabled={self.valve_angle_hold_enabled}, "
                f"target={self.valve_angle_hold_target:.4f}, "
                f"kp={self.valve_angle_hold_kp:.2f}, kd={self.valve_angle_hold_kd:.2f}, "
                f"limit={self.valve_angle_hold_torque_limit:.2f}"
            )
            print(f"[VALVE_LOCK] enabled={self.valve_angle_lock_enabled}")

        self.valve_center_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.valve_center_site_name
        )
        self.valve_grasp_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.valve_grasp_site_name
        )
        self.right_contact_grasp_proxy_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.right_contact_grasp_proxy_site_name
        )
        self.valve_wheel_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.valve_wheel_body_name
        )
        print(
            f"[VALVE] center_site={self.valve_center_site_id}, "
            f"grasp_site={self.valve_grasp_site_id}, "
            f"hand_proxy_site={self.right_contact_grasp_proxy_site_id}, "
            f"wheel_body={self.valve_wheel_body_id}"
        )
        self._apply_valve_contact_tuning()

        # valve hand attachment equality id
        self.weld_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, self.weld_name
        )
        print(f"[ATTACH] name={self.weld_name}, id={self.weld_id}")

        # force initial OFF
        if self.weld_id != -1:
            self.mj_model.eq_active0[self.weld_id] = 0
            if hasattr(self.mj_data, "eq_active"):
                self.mj_data.eq_active[self.weld_id] = 0

    def _set_weld_enabled(self, enabled):
        if self.weld_id == -1:
            return False
        enabled = bool(enabled)
        if self.weld_enable == enabled:
            return True
        self.weld_enable = enabled
        state = 1 if enabled else 0
        self.mj_model.eq_active0[self.weld_id] = state
        if hasattr(self.mj_data, "eq_active"):
            self.mj_data.eq_active[self.weld_id] = state
        mujoco.mj_forward(self.mj_model, self.mj_data)
        print(f"[ATTACH] {self.weld_name} {'ON' if self.weld_enable else 'OFF'}")
        return True

    def _apply_geom_contact_params(self, geom_id, friction=None, condim=None, solref=None, solimp=None):
        if friction is not None:
            self.mj_model.geom_friction[geom_id, :] = np.asarray(friction, dtype=float).reshape(3)
        if condim is not None:
            self.mj_model.geom_condim[geom_id] = int(condim)
        if solref is not None:
            self.mj_model.geom_solref[geom_id, :] = np.asarray(solref, dtype=float).reshape(2)
        if solimp is not None:
            self.mj_model.geom_solimp[geom_id, :] = np.asarray(solimp, dtype=float).reshape(5)

    def _apply_valve_contact_tuning(self):
        if self.valve_wheel_body_id == -1:
            return
        friction = self.config.get("valve_wheel_contact_friction", None)
        condim = self.config.get("valve_wheel_contact_condim", None)
        solref = self.config.get("valve_wheel_contact_solref", None)
        solimp = self.config.get("valve_wheel_contact_solimp", None)
        if friction is None and condim is None and solref is None and solimp is None:
            return

        tuned = 0
        for geom_id in range(self.mj_model.ngeom):
            body_id = int(self.mj_model.geom_bodyid[geom_id])
            if not self._body_is_descendant(body_id, self.valve_wheel_body_id):
                continue
            if self.mj_model.geom_contype[geom_id] == 0 and self.mj_model.geom_conaffinity[geom_id] == 0:
                continue
            self._apply_geom_contact_params(geom_id, friction, condim, solref, solimp)
            tuned += 1
        print(f"[VALVE] tuned wheel contact geoms={tuned}")

    def _compute_valve_angle_hold_torque(self):
        if (
            self.valve_angle_lock_enabled
            or not self.valve_angle_hold_enabled
            or self.valve_qadr == -1
            or self.valve_dof == -1
            or self.valve_angle_hold_torque_limit <= 0.0
        ):
            self.valve_angle_hold_tau = 0.0
            return 0.0

        q = float(self.mj_data.qpos[self.valve_qadr])
        dq = float(self.mj_data.qvel[self.valve_dof])
        target = float(self.valve_angle_hold_target)
        tau = self.valve_angle_hold_kp * (target - q) - self.valve_angle_hold_kd * dq
        tau = float(np.clip(tau, -self.valve_angle_hold_torque_limit, self.valve_angle_hold_torque_limit))
        self.valve_angle_hold_tau = tau
        return tau

    def _apply_valve_generalized_force(self):
        if self.valve_dof == -1:
            return
        tau = self.valve_tau if self.valve_enable else 0.0
        tau += self._compute_valve_angle_hold_torque()
        self.mj_data.qfrc_applied[self.valve_dof] = tau

    def _apply_valve_angle_lock(self):
        if not self.valve_angle_lock_enabled or self.valve_qadr == -1 or self.valve_dof == -1:
            return
        self.mj_data.qpos[self.valve_qadr] = float(self.valve_angle_hold_target)
        self.mj_data.qvel[self.valve_dof] = 0.0

    def _base_to_world(self, pos_base):
        pos_base = np.asarray(pos_base, dtype=float)
        base_pos = self.mj_data.xpos[self.base_id].copy()
        base_rot = self.mj_data.xmat[self.base_id].reshape(3, 3).copy()
        return base_pos + base_rot @ pos_base

    def _apply_viewer_camera(self):
        if not self.viewer_camera or not hasattr(self, "viewer"):
            return
        try:
            cam = self.viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            if "lookat" in self.viewer_camera:
                cam.lookat[:] = np.asarray(self.viewer_camera["lookat"], dtype=float).reshape(3)
            if "distance" in self.viewer_camera:
                cam.distance = float(self.viewer_camera["distance"])
            if "azimuth" in self.viewer_camera:
                cam.azimuth = float(self.viewer_camera["azimuth"])
            if "elevation" in self.viewer_camera:
                cam.elevation = float(self.viewer_camera["elevation"])
        except Exception as exc:
            print(f"[VIEWER_CAMERA] failed to apply configured camera: {exc}")

    def _draw_ee_tracking_markers(self):
        if not self.ee_marker_file or not hasattr(self.viewer, "user_scn"):
            return
        if not os.path.exists(self.ee_marker_file):
            self.viewer.user_scn.ngeom = 0
            return
        try:
            with open(self.ee_marker_file, "r") as file:
                marker_data = json.load(file)
            target_base = marker_data.get("target_right_base")
            current_base = marker_data.get("current_right_base")

            geoms = self.viewer.user_scn.geoms
            self.viewer.user_scn.ngeom = 0
            if self.ee_marker_visual_mode == "none":
                return
            if target_base is None or current_base is None:
                return

            full_visual = self.ee_marker_visual_mode == "full"
            grasp_debug_visual = self.ee_marker_visual_mode in ("full", "grasp_debug")
            valve_arc_visual = self.ee_marker_visual_mode in ("full", "valve_angle")
            grasp_minimal_visual = self.ee_marker_visual_mode in ("grasp_minimal", "grasp")

            def next_geom():
                geom_id = self.viewer.user_scn.ngeom
                if geom_id >= len(geoms):
                    return None
                self.viewer.user_scn.ngeom += 1
                return geoms[geom_id]

            def add_sphere(pos_base, rgba, radius=None):
                geom = next_geom()
                if geom is None:
                    return
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([self.ee_marker_radius if radius is None else radius, 0.0, 0.0], dtype=float),
                    self._base_to_world(pos_base),
                    np.eye(3).reshape(-1),
                    rgba,
                )

            def add_line(from_base, to_base, rgba, width=3.0):
                geom = next_geom()
                if geom is None:
                    return
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3, dtype=float),
                    np.zeros(3, dtype=float),
                    np.eye(3).reshape(-1),
                    rgba,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    width,
                    self._base_to_world(from_base),
                    self._base_to_world(to_base),
                )

            def add_label(pos_base, text, rgba):
                geom = next_geom()
                if geom is None:
                    return
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LABEL,
                    np.zeros(3, dtype=float),
                    self._base_to_world(pos_base),
                    np.eye(3).reshape(-1),
                    rgba,
                )
                geom.label = str(text)[:120]

            add_sphere(target_base, np.array([0.0, 0.8, 0.1, 0.75], dtype=float))
            add_sphere(current_base, np.array([0.9, 0.1, 0.1, 0.75], dtype=float))

            center_base = marker_data.get("valve_center_base")
            actual_grasp_base = marker_data.get("valve_actual_grasp_base")
            command_target_base = marker_data.get("valve_command_target_base")
            final_target_base = marker_data.get("valve_final_target_base")
            arc_points_base = marker_data.get("valve_arc_points_base") or []

            if full_visual and center_base is not None:
                add_sphere(center_base, np.array([0.1, 0.35, 1.0, 0.85], dtype=float), radius=0.012)
            if grasp_debug_visual and center_base is not None and actual_grasp_base is not None:
                add_line(center_base, actual_grasp_base, np.array([0.9, 0.15, 0.05, 0.85], dtype=float), width=4.0)
                add_sphere(actual_grasp_base, np.array([0.9, 0.15, 0.05, 0.85], dtype=float), radius=0.014)
            if valve_arc_visual and center_base is not None and final_target_base is not None:
                add_line(center_base, final_target_base, np.array([0.55, 0.2, 1.0, 0.65], dtype=float), width=2.0)
                add_sphere(final_target_base, np.array([0.55, 0.2, 1.0, 0.75], dtype=float), radius=0.014)
            if grasp_debug_visual and center_base is not None and command_target_base is not None:
                add_line(center_base, command_target_base, np.array([0.95, 0.8, 0.05, 0.85], dtype=float), width=4.0)
                add_sphere(command_target_base, np.array([0.95, 0.8, 0.05, 0.85], dtype=float), radius=0.018)
            elif grasp_minimal_visual and command_target_base is not None:
                # 抓握实验默认只显示实际需要看的工作点，避免虚拟指尖/坐标轴遮挡接触关系。
                add_sphere(command_target_base, np.array([0.95, 0.8, 0.05, 0.85], dtype=float), radius=0.018)

            thumb_tip_base = marker_data.get("right_grasp_thumb_tip_target_base")
            index_tip_base = marker_data.get("right_grasp_index_tip_target_base")
            middle_tip_base = marker_data.get("right_grasp_middle_tip_target_base")
            if grasp_debug_visual and command_target_base is not None and thumb_tip_base is not None:
                add_line(command_target_base, thumb_tip_base, np.array([0.0, 0.9, 0.9, 0.85], dtype=float), width=2.0)
                add_sphere(thumb_tip_base, np.array([0.0, 0.9, 0.9, 0.85], dtype=float), radius=0.010)
            if grasp_debug_visual and command_target_base is not None and index_tip_base is not None:
                add_line(command_target_base, index_tip_base, np.array([1.0, 0.35, 0.9, 0.85], dtype=float), width=2.0)
                add_sphere(index_tip_base, np.array([1.0, 0.35, 0.9, 0.85], dtype=float), radius=0.010)
            if grasp_debug_visual and command_target_base is not None and middle_tip_base is not None:
                add_line(command_target_base, middle_tip_base, np.array([1.0, 0.35, 0.9, 0.85], dtype=float), width=2.0)
                add_sphere(middle_tip_base, np.array([1.0, 0.35, 0.9, 0.85], dtype=float), radius=0.010)

            x_axis_end = marker_data.get("right_grasp_x_axis_end_base")
            y_axis_end = marker_data.get("right_grasp_y_axis_end_base")
            z_axis_end = marker_data.get("right_grasp_z_axis_end_base")
            if grasp_debug_visual and x_axis_end is not None:
                add_line(target_base, x_axis_end, np.array([1.0, 0.1, 0.1, 0.9], dtype=float), width=3.0)
            if grasp_debug_visual and y_axis_end is not None:
                add_line(target_base, y_axis_end, np.array([0.1, 0.9, 0.1, 0.9], dtype=float), width=3.0)
            if grasp_debug_visual and z_axis_end is not None:
                add_line(target_base, z_axis_end, np.array([0.1, 0.3, 1.0, 0.9], dtype=float), width=3.0)

            if valve_arc_visual:
                for start, end in zip(arc_points_base[:-1], arc_points_base[1:]):
                    add_line(start, end, np.array([1.0, 0.72, 0.0, 0.85], dtype=float), width=3.0)
                for arc_point in arc_points_base[::2]:
                    add_sphere(arc_point, np.array([1.0, 0.72, 0.0, 0.75], dtype=float), radius=0.007)

            label = marker_data.get("valve_angle_label")
            label_base = marker_data.get("valve_label_base")
            if label and label_base is not None:
                add_label(label_base, label, np.array([1.0, 0.95, 0.25, 1.0], dtype=float))
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

    def _maybe_update_weld_from_control_file(self):
        if not self.sim_control_file or not os.path.exists(self.sim_control_file):
            return
        try:
            with open(self.sim_control_file, "r") as file:
                control = json.load(file)
        except Exception:
            return

        if (
            "weld_enabled" not in control
            and "attach_enabled" not in control
            and self.weld_name not in control
            and "right_hand_valve_weld" not in control
        ):
            return
        timestamp = float(control.get("weld_timestamp", control.get("timestamp", 0.0)))
        if timestamp <= self._last_weld_control_timestamp:
            return

        enabled = control.get(
            "weld_enabled",
            control.get("attach_enabled", control.get(self.weld_name, control.get("right_hand_valve_weld", False))),
        )
        self._set_weld_enabled(bool(enabled))
        self._last_weld_control_timestamp = timestamp

    def _maybe_update_valve_angle_control_from_control_file(self):
        if not self.sim_control_file or not os.path.exists(self.sim_control_file):
            return
        try:
            with open(self.sim_control_file, "r") as file:
                control = json.load(file)
        except Exception:
            return

        keys = (
            "valve_angle_lock_enabled",
            "valve_angle_hold_enabled",
            "valve_angle_hold_target",
        )
        if not any(key in control for key in keys):
            return
        timestamp = float(control.get("valve_angle_control_timestamp", control.get("timestamp", 0.0)))
        if timestamp <= self._last_valve_angle_control_timestamp:
            return

        if "valve_angle_lock_enabled" in control:
            self.valve_angle_lock_enabled = bool(control["valve_angle_lock_enabled"])
        if "valve_angle_hold_enabled" in control:
            self.valve_angle_hold_enabled = bool(control["valve_angle_hold_enabled"])
        if "valve_angle_hold_target" in control:
            self.valve_angle_hold_target = float(control["valve_angle_hold_target"])

        self._last_valve_angle_control_timestamp = timestamp
        print(
            "[VALVE_ANGLE_CONTROL] "
            f"lock={self.valve_angle_lock_enabled}, "
            f"hold={self.valve_angle_hold_enabled}, "
            f"target={float(self.valve_angle_hold_target):.4f}"
        )

    def _site_world(self, site_id):
        if site_id == -1:
            return None
        return np.array(self.mj_data.site_xpos[site_id], dtype=float).reshape(3)

    def _current_foot_contact_status(self):
        left_force = 0.0
        right_force = 0.0
        if hasattr(self, "ffss_idx") and self.ffss_idx >= 0:
            sensor_tail = np.asarray(self.mj_data.sensordata[self.ffss_idx:], dtype=float)
            if sensor_tail.size >= 24:
                foot_forces = sensor_tail[:24].reshape(8, 3)
                norms = np.linalg.norm(foot_forces, axis=1)
                left_force = float(np.sum(norms[:4]))
                right_force = float(np.sum(norms[4:]))
        threshold = float(self.config.get("foot_contact_force_threshold", 20.0))
        return left_force, right_force, bool(left_force > threshold and right_force > threshold)

    def _body_is_descendant(self, body_id, root_id):
        if root_id == -1:
            return False
        while body_id > 0:
            if body_id == root_id:
                return True
            body_id = int(self.mj_model.body_parentid[body_id])
        return body_id == root_id

    def _geom_body_name(self, geom_id):
        body_id = int(self.mj_model.geom_bodyid[geom_id])
        return mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""

    def _geom_label(self, geom_id):
        geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return geom_name or self._geom_body_name(geom_id)

    def _is_right_hand_contact_geom(self, geom_id):
        body_name = self._geom_body_name(geom_id)
        return (
            body_name in self.right_hand_contact_body_names
            or any(body_name.startswith(prefix) for prefix in self.right_hand_contact_body_prefixes)
        )

    def _is_valve_wheel_contact_geom(self, geom_id):
        body_id = int(self.mj_model.geom_bodyid[geom_id])
        return self._body_is_descendant(body_id, self.valve_wheel_body_id)

    def _right_hand_valve_contact_summary(self, include_proxy=True):
        contact_count = 0
        normal_force = 0.0
        contact_pairs = []
        for contact_id in range(int(self.mj_data.ncon)):
            contact = self.mj_data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            geom1_is_hand = self._is_right_hand_contact_geom(geom1)
            geom2_is_hand = self._is_right_hand_contact_geom(geom2)
            geom1_is_valve = self._is_valve_wheel_contact_geom(geom1)
            geom2_is_valve = self._is_valve_wheel_contact_geom(geom2)
            if not ((geom1_is_hand and geom2_is_valve) or (geom2_is_hand and geom1_is_valve)):
                continue

            hand_geom = geom1 if geom1_is_hand else geom2
            if not include_proxy and self._geom_label(hand_geom).startswith("right_grasp_proxy_"):
                continue

            contact_count += 1
            force = np.zeros(6, dtype=float)
            try:
                mujoco.mj_contactForce(self.mj_model, self.mj_data, contact_id, force)
                normal_force += max(0.0, float(force[0]))
            except Exception:
                pass

            if len(contact_pairs) < 8:
                if geom1_is_hand:
                    contact_pairs.append(f"{self._geom_label(geom1)}:{self._geom_label(geom2)}")
                else:
                    contact_pairs.append(f"{self._geom_label(geom2)}:{self._geom_label(geom1)}")
        return contact_count, normal_force, contact_pairs

    def _debug_contact_pairs(self):
        """记录和抓握相关的原始 MuJoCo 接触对，便于区分未接触和接触判据未命中。"""
        tokens = tuple(
            self.config.get(
                "contact_debug_name_tokens",
                [
                    "right_grasp_proxy",
                    "right_hand_",
                    "right_rubber_hand",
                    "grasp_sleeve",
                    "rim_",
                    "spoke_",
                    "wheel_",
                ],
            )
        )
        pairs = []
        for contact_id in range(int(self.mj_data.ncon)):
            contact = self.mj_data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            label1 = self._geom_label(geom1)
            label2 = self._geom_label(geom2)
            pair_label = f"{label1}:{label2}"
            if tokens and not any(token in pair_label for token in tokens):
                continue

            force = np.zeros(6, dtype=float)
            normal_force = 0.0
            try:
                mujoco.mj_contactForce(self.mj_model, self.mj_data, contact_id, force)
                normal_force = max(0.0, float(force[0]))
            except Exception:
                pass
            pairs.append(
                f"{pair_label}:dist={float(contact.dist):.5f}:fn={normal_force:.2f}"
            )
            if len(pairs) >= 24:
                break
        return pairs

    def _named_geom_world_pose(self, geom_name):
        geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            return None
        body_id = int(self.mj_model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        return {
            "pos": np.array(self.mj_data.geom_xpos[geom_id], dtype=float).reshape(3).tolist(),
            "xmat": np.array(self.mj_data.geom_xmat[geom_id], dtype=float).reshape(3, 3).tolist(),
            "body": body_name,
            "contype": int(self.mj_model.geom_contype[geom_id]),
            "conaffinity": int(self.mj_model.geom_conaffinity[geom_id]),
        }

    def _write_sim_status(self):
        if not self.sim_status_file:
            return
        if self.t - self._last_status_write_t < self.sim_status_write_interval_s:
            return
        self._last_status_write_t = self.t

        status_dir = os.path.dirname(self.sim_status_file)
        if status_dir:
            os.makedirs(status_dir, exist_ok=True)

        left_force, right_force, feet_contact_stable = self._current_foot_contact_status()
        base_pos = np.array(self.mj_data.xpos[self.base_id], dtype=float).reshape(3)
        base_xmat = np.array(self.mj_data.xmat[self.base_id], dtype=float).reshape(3, 3)

        valve_angle = 0.0
        valve_vel = 0.0
        if self.valve_qadr != -1:
            valve_angle = float(self.mj_data.qpos[self.valve_qadr])
        if self.valve_dof != -1:
            valve_vel = float(self.mj_data.qvel[self.valve_dof])
        contact_count, contact_normal_force, contact_pairs = self._right_hand_valve_contact_summary()
        real_contact_count, real_contact_normal_force, real_contact_pairs = (
            self._right_hand_valve_contact_summary(include_proxy=False)
        )

        payload = {
            "timestamp": time.time(),
            "sim_time": float(self.t),
            "base_pos_world": base_pos.tolist(),
            "base_xmat_world": base_xmat.tolist(),
            "left_foot_force": left_force,
            "right_foot_force": right_force,
            "feet_contact_stable": feet_contact_stable,
            "elastic_enabled": bool(
                self.config["ENABLE_ELASTIC_BAND"]
                and self.elastic_band is not None
                and self.elastic_band.enable
            ),
            "elastic_length": float(self.elastic_band.length) if self.elastic_band is not None else 0.0,
            "attach_name": self.weld_name,
            "attach_enabled": bool(self.weld_enable),
            "weld_enabled": bool(self.weld_enable),
            "valve_angle": valve_angle,
            "valve_vel": valve_vel,
            "valve_angle_hold_enabled": bool(self.valve_angle_hold_enabled),
            "valve_angle_hold_target": (
                float(self.valve_angle_hold_target) if self.valve_angle_hold_target is not None else 0.0
            ),
            "valve_angle_hold_tau": float(self.valve_angle_hold_tau),
            "valve_angle_lock_enabled": bool(self.valve_angle_lock_enabled),
            "right_hand_valve_contact_count": contact_count,
            "right_hand_valve_contact_normal_force": contact_normal_force,
            "right_hand_valve_contact_pairs": contact_pairs,
            "right_hand_valve_real_contact_count": real_contact_count,
            "right_hand_valve_real_contact_normal_force": real_contact_normal_force,
            "right_hand_valve_real_contact_pairs": real_contact_pairs,
            "debug_ncon": int(self.mj_data.ncon),
            "debug_contact_pairs": self._debug_contact_pairs(),
        }
        extra_payload_fn = getattr(self, "_sim_status_extra_payload", None)
        if callable(extra_payload_fn):
            try:
                payload.update(extra_payload_fn())
            except Exception as exc:
                self.logger.warning(f"Failed to add extra sim status payload: {exc}")

        debug_geom_names = self.config.get(
            "sim_status_debug_geom_names",
            [
                "right_grasp_proxy_palm_pad",
                "right_grasp_proxy_thumb_pad",
                "right_grasp_proxy_index_pad",
                "right_grasp_proxy_middle_pad",
                "grasp_sleeve",
                "grasp_sleeve_stop_plus",
                "grasp_sleeve_stop_minus",
            ],
        )
        debug_geom_world = {}
        for geom_name in debug_geom_names:
            pose = self._named_geom_world_pose(str(geom_name))
            if pose is not None:
                debug_geom_world[str(geom_name)] = pose
        if debug_geom_world:
            payload["debug_geom_world"] = debug_geom_world

        center_world = self._site_world(self.valve_center_site_id)
        grasp_world = self._site_world(self.valve_grasp_site_id)
        hand_proxy_world = self._site_world(self.right_contact_grasp_proxy_site_id)
        if center_world is not None:
            payload["valve_center_world"] = center_world.tolist()
        if grasp_world is not None:
            payload["valve_grasp_world"] = grasp_world.tolist()
        if hand_proxy_world is not None:
            payload["right_contact_grasp_proxy_world"] = hand_proxy_world.tolist()
        if self.valve_wheel_body_id != -1:
            wheel_xmat = np.array(self.mj_data.xmat[self.valve_wheel_body_id], dtype=float).reshape(3, 3)
            wheel_pos = np.array(self.mj_data.xpos[self.valve_wheel_body_id], dtype=float).reshape(3)
            payload["valve_wheel_world"] = wheel_pos.tolist()
            payload["valve_wheel_xmat_world"] = wheel_xmat.tolist()
            payload["valve_axis_world"] = wheel_xmat[:, 0].tolist()

        tmp_file = f"{self.sim_status_file}.tmp"
        with open(tmp_file, "w") as file:
            json.dump(payload, file)
        os.replace(tmp_file, self.sim_status_file)

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()

        self._maybe_disable_elastic_from_control_file()
        self._maybe_update_weld_from_control_file()
        self._maybe_update_valve_angle_control_from_control_file()

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

        self._apply_valve_generalized_force()
        self._apply_valve_angle_lock()

        mujoco.mj_step(self.mj_model, self.mj_data)
        self._apply_valve_angle_lock()

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
        self._write_sim_status()


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
