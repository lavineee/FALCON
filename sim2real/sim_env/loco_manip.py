import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)
import sys
import argparse
import csv
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

        # live plot is opt-in to keep deployment/sim runs from opening an extra
        # matplotlib process unless explicitly requested.
        self.enable_live_plot = bool(config.get("enable_live_plot", False)) and plt is not None
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
        self.valve_vision_enable = bool(config.get("valve_vision_enable", False))
        self.valve_vision_mode = str(config.get("valve_vision_mode", "sim_depth_color_debug"))
        self.valve_vision_camera_name = str(config.get("valve_vision_camera_name", "head_camera"))
        self.valve_vision_status_file = config.get(
            "valve_vision_status_file",
            config.get("vision_status_file", "/tmp/falcon_valve_vision_status.json"),
        )
        self.valve_vision_metrics_file = config.get(
            "valve_vision_metrics_file",
            "/tmp/falcon_valve_vision_metrics.csv",
        )
        self.valve_vision_debug_dir = config.get("valve_vision_debug_dir", "/tmp")
        self.valve_vision_hz = float(config.get("valve_vision_hz", 10.0))
        self.valve_vision_debug_image_hz = float(config.get("valve_vision_debug_image_hz", 2.0))
        self.valve_vision_width = int(config.get("valve_vision_width", 640))
        self.valve_vision_height = int(config.get("valve_vision_height", 480))
        self._valve_vision_renderer = None
        self._valve_vision_ready = False
        self._valve_vision_renderer_failed = False
        self._valve_vision_camera_id = -1
        self._valve_vision_last_update_t = -1e9
        self._valve_vision_last_step_t = None
        self._valve_vision_last_debug_image_t = -1e9
        self._valve_vision_state = {}
        self._valve_vision_gt_angle0 = None
        self._valve_vision_theta_vis0 = None
        self._valve_vision_metrics_header_written = False
        self._valve_vision_estimate_fn = None
        self._valve_vision_atomic_write_json_fn = None
        self._valve_vision_project_fn = None
        self._valve_vision_backproject_pixel_fn = None
        self._valve_vision_transform_points_fn = None
        self._valve_vision_sample_depth_fn = None
        self._valve_vision_intrinsics_fn = None
        self._valve_vision_wrap_to_pi_fn = None

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
        self.left_foot_contact_site_names = tuple(
            config.get(
                "left_foot_contact_site_names",
                [
                    "left_foot_contact_1",
                    "left_foot_contact_2",
                    "left_foot_contact_3",
                    "left_foot_contact_4",
                ],
            )
        )
        self.right_foot_contact_site_names = tuple(
            config.get(
                "right_foot_contact_site_names",
                [
                    "right_foot_contact_1",
                    "right_foot_contact_2",
                    "right_foot_contact_3",
                    "right_foot_contact_4",
                ],
            )
        )
        self.left_foot_contact_site_ids = []
        self.right_foot_contact_site_ids = []

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
        self.left_foot_contact_site_ids = self._resolve_site_ids(self.left_foot_contact_site_names)
        self.right_foot_contact_site_ids = self._resolve_site_ids(self.right_foot_contact_site_names)
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
        self._init_valve_vision()

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

    def _resolve_site_ids(self, site_names):
        ids = []
        for name in site_names:
            site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, str(name))
            if site_id != -1:
                ids.append(site_id)
        return ids

    def _site_world(self, site_id):
        if site_id == -1:
            return None
        return np.array(self.mj_data.site_xpos[site_id], dtype=float).reshape(3)

    def _site_centroid_world(self, site_ids):
        points = []
        for site_id in site_ids:
            if site_id != -1:
                points.append(np.array(self.mj_data.site_xpos[site_id], dtype=float).reshape(3))
        if not points:
            return None
        return np.mean(np.vstack(points), axis=0)

    def _root_velocity_world(self):
        qvel = np.asarray(self.mj_data.qvel, dtype=float).reshape(-1)
        lin = np.full(3, np.nan, dtype=float)
        ang = np.full(3, np.nan, dtype=float)
        if qvel.size >= 3:
            lin = qvel[:3].copy()
        if qvel.size >= 6:
            ang = qvel[3:6].copy()
        return lin, ang

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

    def _right_hand_contact_role(self, geom_id):
        label = self._geom_label(geom_id)
        body_name = self._geom_body_name(geom_id)
        token = f"{label}:{body_name}"
        if "right_rubber_hand" in token or "right_hand_palm" in token or "right_grasp_proxy_palm" in token:
            return "palm"
        if "right_hand_thumb" in token or "right_grasp_proxy_thumb" in token:
            return "thumb"
        if "right_hand_index" in token or "right_grasp_proxy_index" in token:
            return "index"
        if "right_hand_middle" in token or "right_grasp_proxy_middle" in token:
            return "middle"
        return "other"

    def _right_hand_valve_contact_role_summary(self, include_proxy=True):
        counts = {"palm": 0, "thumb": 0, "index": 0, "middle": 0, "other": 0}
        forces = {"palm": 0.0, "thumb": 0.0, "index": 0.0, "middle": 0.0, "other": 0.0}
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

            role = self._right_hand_contact_role(hand_geom)
            counts[role] = counts.get(role, 0) + 1
            force = np.zeros(6, dtype=float)
            try:
                mujoco.mj_contactForce(self.mj_model, self.mj_data, contact_id, force)
                forces[role] = forces.get(role, 0.0) + max(0.0, float(force[0]))
            except Exception:
                pass
        return counts, forces

    def _valve_contact_basis_world(self):
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if self.valve_wheel_body_id != -1:
            wheel_xmat = np.array(self.mj_data.xmat[self.valve_wheel_body_id], dtype=float).reshape(3, 3)
            axis = wheel_xmat[:, 0].copy()
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm

        center = self._site_world(self.valve_center_site_id)
        grasp = self._site_world(self.valve_grasp_site_id)
        radial = None
        if center is not None and grasp is not None:
            radial = grasp - center
            radial = radial - np.dot(radial, axis) * axis
            radial_norm = np.linalg.norm(radial)
            if radial_norm > 1e-8:
                radial = radial / radial_norm
        if radial is None:
            radial = np.array([0.0, 1.0, 0.0], dtype=float)

        tangent = np.cross(axis, radial)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-8:
            tangent = tangent / tangent_norm
        else:
            tangent = np.array([0.0, 0.0, 1.0], dtype=float)
        return axis, radial, tangent

    @staticmethod
    def _empty_contact_force_bucket():
        return {
            "count": 0,
            "normal_force": 0.0,
            "force_world": np.zeros(3, dtype=float),
            "axis": 0.0,
            "radial": 0.0,
            "tangent": 0.0,
            "axis_abs": 0.0,
            "radial_abs": 0.0,
            "tangent_abs": 0.0,
        }

    def _right_hand_valve_contact_force_decomposition(self):
        axis, radial, tangent = self._valve_contact_basis_world()
        buckets = {
            "total": self._empty_contact_force_bucket(),
            "real": self._empty_contact_force_bucket(),
            "proxy": self._empty_contact_force_bucket(),
        }

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
            is_proxy = self._geom_label(hand_geom).startswith("right_grasp_proxy_")
            force = np.zeros(6, dtype=float)
            try:
                mujoco.mj_contactForce(self.mj_model, self.mj_data, contact_id, force)
                contact_frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
                # MuJoCo stores contact-frame axes as world-frame rows. The
                # returned force is contact-frame 6D force/torque; this is our
                # best-effort force on the hand, with sign adjusted by geom order.
                force_world_geom2 = contact_frame.T @ force[:3]
                force_world_hand = force_world_geom2 if geom2_is_hand else -force_world_geom2
                normal_force = max(0.0, float(force[0]))
            except Exception:
                force_world_hand = np.zeros(3, dtype=float)
                normal_force = 0.0

            for bucket_name in ("total", "proxy" if is_proxy else "real"):
                bucket = buckets[bucket_name]
                bucket["count"] += 1
                bucket["normal_force"] += normal_force
                bucket["force_world"] += force_world_hand
                axis_force = float(np.dot(force_world_hand, axis))
                radial_force = float(np.dot(force_world_hand, radial))
                tangent_force = float(np.dot(force_world_hand, tangent))
                bucket["axis"] += axis_force
                bucket["radial"] += radial_force
                bucket["tangent"] += tangent_force
                bucket["axis_abs"] += abs(axis_force)
                bucket["radial_abs"] += abs(radial_force)
                bucket["tangent_abs"] += abs(tangent_force)

        return buckets

    @staticmethod
    def _flatten_contact_force_bucket(prefix, bucket):
        tangent_abs = float(bucket["tangent_abs"])
        denom = max(tangent_abs, 1e-6)
        return {
            f"{prefix}_count": int(bucket["count"]),
            f"{prefix}_normal_force": float(bucket["normal_force"]),
            f"{prefix}_force_world": np.asarray(bucket["force_world"], dtype=float).reshape(3).tolist(),
            f"{prefix}_force_axis": float(bucket["axis"]),
            f"{prefix}_force_radial": float(bucket["radial"]),
            f"{prefix}_force_tangent": float(bucket["tangent"]),
            f"{prefix}_force_axis_abs": float(bucket["axis_abs"]),
            f"{prefix}_force_radial_abs": float(bucket["radial_abs"]),
            f"{prefix}_force_tangent_abs": tangent_abs,
            f"{prefix}_force_axis_abs_over_tangent": float(bucket["axis_abs"]) / denom,
            f"{prefix}_force_radial_abs_over_tangent": float(bucket["radial_abs"]) / denom,
        }

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

    def _init_valve_vision(self):
        if not self.valve_vision_enable:
            return
        if self.valve_vision_mode != "sim_depth_color_debug":
            self.logger.warning(
                f"Unsupported valve_vision_mode={self.valve_vision_mode}; vision output disabled."
            )
            return
        try:
            from sim2real.vision.valve_color_geometry import (
                atomic_write_json,
                backproject_pixel_to_point,
                camera_intrinsics_from_fovy,
                estimate_valve_geometry_from_rgbd,
                project_points_to_image,
                sample_depth_near,
                transform_points,
                wrap_to_pi,
            )
        except Exception as exc:
            self.logger.warning(f"Failed to import valve color vision helpers: {exc}")
            return

        camera_id = mujoco.mj_name2id(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.valve_vision_camera_name,
        )
        if camera_id == -1:
            self.logger.warning(
                f"valve_vision_camera_name={self.valve_vision_camera_name} not found; vision output disabled."
            )
            return
        self._valve_vision_camera_id = camera_id
        self._valve_vision_estimate_fn = estimate_valve_geometry_from_rgbd
        self._valve_vision_atomic_write_json_fn = atomic_write_json
        self._valve_vision_project_fn = project_points_to_image
        self._valve_vision_backproject_pixel_fn = backproject_pixel_to_point
        self._valve_vision_transform_points_fn = transform_points
        self._valve_vision_sample_depth_fn = sample_depth_near
        self._valve_vision_intrinsics_fn = camera_intrinsics_from_fovy
        self._valve_vision_wrap_to_pi_fn = wrap_to_pi
        self._valve_vision_metrics_header_written = bool(
            self.valve_vision_metrics_file and os.path.exists(self.valve_vision_metrics_file)
        )
        self._valve_vision_ready = True
        print(
            "[VALVE_VISION] sim_depth_color_debug enabled: "
            f"camera={self.valve_vision_camera_name}, "
            f"size={self.valve_vision_width}x{self.valve_vision_height}, "
            f"hz={self.valve_vision_hz:.1f}"
        )

    def _ensure_valve_vision_renderer(self):
        if self._valve_vision_renderer is not None:
            return True
        if self._valve_vision_renderer_failed:
            return False
        try:
            self._valve_vision_renderer = mujoco.Renderer(
                self.mj_model,
                height=self.valve_vision_height,
                width=self.valve_vision_width,
            )
            return True
        except Exception as exc:
            self._valve_vision_renderer_failed = True
            self.logger.warning(f"Failed to create MuJoCo valve vision renderer: {exc}")
            return False

    def _valve_vision_hsv_defaults(self):
        return {
            "center": [[80, 80, 80], [100, 255, 255]],   # cyan
            "spoke": [[140, 80, 80], [170, 255, 255]],   # magenta
            "grasp": [[45, 80, 80], [80, 255, 255]],     # green
            "plane_aux": [[110, 180, 180], [125, 255, 255]], # saturated blue
        }

    def _valve_vision_config(self):
        return {
            "hsv_ranges": self.config.get("valve_vision_hsv_ranges", self._valve_vision_hsv_defaults()),
            "min_mask_area_px": int(self.config.get("valve_vision_min_mask_area_px", 50)),
            "min_points": int(self.config.get("valve_vision_min_points", 30)),
            "min_aux_points": int(self.config.get("valve_vision_min_aux_points", 50)),
            "use_plane_aux": bool(self.config.get("valve_vision_use_plane_aux", True)),
            "max_plane_fit_error_m": float(self.config.get("valve_vision_max_plane_fit_error_m", 0.02)),
            "depth_min_m": float(self.config.get("valve_vision_depth_min_m", 0.02)),
            "depth_max_m": float(self.config.get("valve_vision_depth_max_m", 5.0)),
            "depth_cluster_tolerance_m": float(self.config.get("valve_vision_depth_cluster_tolerance_m", 0.08)),
            "marker_axis_offset_m": float(self.config.get("valve_vision_marker_axis_offset_m", 0.0325)),
            "max_points_per_mask": int(self.config.get("valve_vision_max_points_per_mask", 6000)),
            "morph_open_iters": int(self.config.get("valve_vision_morph_open_iters", 1)),
            "morph_close_iters": int(self.config.get("valve_vision_morph_close_iters", 1)),
            "morph_kernel_px": int(self.config.get("valve_vision_morph_kernel_px", 3)),
            "grasp_forward_axis_sign": float(self.config.get("valve_grasp_forward_axis_sign", -1.0)),
            "grasp_radial_axis": str(self.config.get("valve_grasp_radial_axis", "y")),
            "grasp_radial_axis_sign": float(self.config.get("valve_grasp_radial_axis_sign", -1.0)),
        }

    def _valve_vision_intrinsics(self):
        fovy = float(self.mj_model.cam_fovy[self._valve_vision_camera_id])
        if not np.isfinite(fovy) or fovy <= 0.0:
            fovy = 45.0
        return self._valve_vision_intrinsics_fn(self.valve_vision_width, self.valve_vision_height, fovy)

    def _valve_vision_T_base_cam(self):
        cam_pos_world = np.array(self.mj_data.cam_xpos[self._valve_vision_camera_id], dtype=float).reshape(3)
        cam_R_world = np.array(self.mj_data.cam_xmat[self._valve_vision_camera_id], dtype=float).reshape(3, 3)
        base_pos_world = np.array(self.mj_data.xpos[self.base_id], dtype=float).reshape(3)
        base_R_world = np.array(self.mj_data.xmat[self.base_id], dtype=float).reshape(3, 3)
        world_to_base = base_R_world.T
        T_base_cam = np.eye(4, dtype=float)
        T_base_cam[:3, :3] = world_to_base @ cam_R_world
        T_base_cam[:3, 3] = world_to_base @ (cam_pos_world - base_pos_world)
        return T_base_cam

    def _render_valve_vision_rgbd(self):
        renderer = self._valve_vision_renderer
        renderer.disable_depth_rendering()
        renderer.update_scene(self.mj_data, camera=self.valve_vision_camera_name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(self.mj_data, camera=self.valve_vision_camera_name)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        return rgb, depth

    def _valve_vision_gt_base_geometry(self):
        base_pos_world = np.array(self.mj_data.xpos[self.base_id], dtype=float).reshape(3)
        base_R_world = np.array(self.mj_data.xmat[self.base_id], dtype=float).reshape(3, 3)
        world_to_base = base_R_world.T
        center_world = self._site_world(self.valve_center_site_id)
        grasp_world = self._site_world(self.valve_grasp_site_id)
        if center_world is None or grasp_world is None:
            return None
        geom = {
            "center_base": world_to_base @ (center_world - base_pos_world),
            "grasp_base": world_to_base @ (grasp_world - base_pos_world),
            "valve_angle": 0.0,
        }
        if self.valve_wheel_body_id != -1:
            wheel_R_world = np.array(self.mj_data.xmat[self.valve_wheel_body_id], dtype=float).reshape(3, 3)
            geom["axis_base"] = world_to_base @ wheel_R_world[:, 0]
        if self.valve_qadr != -1:
            geom["valve_angle"] = float(self.mj_data.qpos[self.valve_qadr])
        return geom

    def _valve_vision_self_check(self, depth, K, T_base_cam):
        gt = self._valve_vision_gt_base_geometry()
        check = {
            "ok": False,
            "max_error_m": None,
            "threshold_m": float(self.config.get("valve_vision_self_check_max_error_m", 0.08)),
            "points": {},
            "marker_points": {},
        }
        if gt is None:
            check["reason"] = "missing_gt_sites"
            return check
        core_errors = []
        marker_errors = []

        def check_points(names, points_base, output):
            pixels, _, visible = self._valve_vision_project_fn(points_base, K, T_base_cam)
            local_errors = []
            for idx, name in enumerate(names):
                item = {
                    "pixel": pixels[idx].tolist(),
                    "visible": bool(visible[idx]),
                    "depth_m": None,
                    "error_m": None,
                }
                if visible[idx]:
                    sampled_depth = self._valve_vision_sample_depth_fn(
                        depth,
                        pixels[idx],
                        radius=int(self.config.get("valve_vision_self_check_depth_radius_px", 4)),
                    )
                    if sampled_depth is not None:
                        point_cam = self._valve_vision_backproject_pixel_fn(pixels[idx], sampled_depth, K)
                        point_base = self._valve_vision_transform_points_fn(T_base_cam, point_cam.reshape(1, 3))[0]
                        error = float(np.linalg.norm(point_base - points_base[idx]))
                        item["depth_m"] = float(sampled_depth)
                        item["reprojected_base"] = point_base.tolist()
                        item["gt_base"] = points_base[idx].tolist()
                        item["error_m"] = error
                        local_errors.append(error)
                output[name] = item
            return local_errors

        names = ("center", "grasp")
        points_base = np.vstack([gt[f"{name}_base"] for name in names])
        core_errors.extend(check_points(names, points_base, check["points"]))

        marker_points = {}
        marker_names = {
            "center_marker": "vision_center_cyan_marker",
            "spoke_marker": "vision_spoke_magenta_marker",
            "grasp_marker": "vision_grasp_green_marker",
            "plane_aux_marker": "vision_plane_aux_blue_marker",
        }
        base_pos_world = np.array(self.mj_data.xpos[self.base_id], dtype=float).reshape(3)
        world_to_base = np.array(self.mj_data.xmat[self.base_id], dtype=float).reshape(3, 3).T
        for key, geom_name in marker_names.items():
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id != -1:
                geom_world = np.array(self.mj_data.geom_xpos[geom_id], dtype=float).reshape(3)
                marker_points[key] = world_to_base @ (geom_world - base_pos_world)
        if marker_points:
            marker_keys = tuple(marker_points.keys())
            marker_base = np.vstack([marker_points[key] for key in marker_keys])
            marker_errors.extend(check_points(marker_keys, marker_base, check["marker_points"]))

        center_error = check["points"].get("center", {}).get("error_m", None)
        if center_error is None:
            check["reason"] = "missing_projected_depth"
            return check
        max_error = float(center_error)
        check["max_error_m"] = max_error
        check["max_core_error_m"] = float(max(core_errors)) if core_errors else max_error
        check["max_marker_error_m"] = float(max(marker_errors)) if marker_errors else None
        all_errors = core_errors + marker_errors
        check["max_all_error_m"] = float(max(all_errors)) if all_errors else max_error
        check["ok"] = bool(max_error <= check["threshold_m"])
        if not check["ok"]:
            check["reason"] = "self_check_error_too_large"
        return check

    def _valve_vision_debug_gt(self, status):
        gt = self._valve_vision_gt_base_geometry()
        if gt is None:
            return None
        debug = {
            "grasp_latched": bool(status.get("grasp_latched", False)),
        }
        if status.get("pose_valid", status.get("valid", False)):
            if "center_base" in status:
                center = np.asarray(status.get("center_base", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
                debug["center_error_m"] = float(np.linalg.norm(center - gt["center_base"]))
            else:
                debug["center_error_m"] = None
            if "grasp_base" in status:
                grasp = np.asarray(status.get("grasp_base", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
                debug["grasp_error_m"] = float(np.linalg.norm(grasp - gt["grasp_base"]))
            else:
                debug["grasp_error_m"] = None
            axis = np.asarray(status.get("axis_base", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            gt_axis = np.asarray(gt.get("axis_base", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            axis_norm = float(np.linalg.norm(axis))
            gt_axis_norm = float(np.linalg.norm(gt_axis))
            if axis_norm > 1e-9 and gt_axis_norm > 1e-9:
                dot = abs(float(np.dot(axis / axis_norm, gt_axis / gt_axis_norm)))
                dot = max(-1.0, min(1.0, dot))
                debug["axis_angle_error_deg"] = float(np.degrees(np.arccos(dot)))
            else:
                debug["axis_angle_error_deg"] = None
            if status.get("angle_valid", False):
                theta_vis = float(status.get("valve_angle", 0.0))
                theta_vis_raw = status.get("quality", {}).get("theta_vis_raw", theta_vis)
                if self._valve_vision_theta_vis0 is None:
                    self._valve_vision_theta_vis0 = float(theta_vis_raw if theta_vis_raw is not None else theta_vis)
                    self._valve_vision_gt_angle0 = float(gt["valve_angle"])
                if theta_vis_raw is not None:
                    theta_vis_delta = self._valve_vision_wrap_to_pi_fn(float(theta_vis_raw) - self._valve_vision_theta_vis0)
                else:
                    theta_vis_delta = self._valve_vision_wrap_to_pi_fn(theta_vis)
                theta_gt_delta = self._valve_vision_wrap_to_pi_fn(float(gt["valve_angle"]) - self._valve_vision_gt_angle0)
                angle_error = self._valve_vision_wrap_to_pi_fn(theta_vis_delta - theta_gt_delta)
                debug["theta_vis_delta"] = float(theta_vis_delta)
                debug["theta_gt_delta"] = float(theta_gt_delta)
                debug["angle_error_deg"] = float(np.degrees(angle_error))
            else:
                debug["theta_vis_delta"] = None
                debug["theta_gt_delta"] = None
                debug["angle_error_deg"] = None
        else:
            debug["center_error_m"] = None
            debug["grasp_error_m"] = None
            debug["axis_angle_error_deg"] = None
            debug["theta_vis_delta"] = None
            debug["theta_gt_delta"] = None
            debug["angle_error_deg"] = None
        return debug

    def _invalid_valve_vision_status(self, reason, self_check=None):
        quality = {}
        if self_check is not None:
            quality["self_check"] = self_check
        return {
            "valid": False,
            "pose_valid": False,
            "grasp_valid": False,
            "grasp_latched": False,
            "plane_aux_valid": False,
            "timestamp": time.time(),
            "frame": "base",
            "source": "sim_depth_color",
            "reason": reason,
            "angle_valid": False,
            "quality": quality,
        }

    def _append_valve_vision_metrics(self, status):
        if not self.valve_vision_metrics_file:
            return
        metrics_dir = os.path.dirname(self.valve_vision_metrics_file)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        debug_gt = status.get("debug_gt", {}) or {}
        quality = status.get("quality", {}) or {}
        self_check = quality.get("self_check", {}) or {}
        row = {
            "wall_time": float(status.get("timestamp", time.time())),
            "sim_time": float(self.t),
            "valid": bool(status.get("valid", False)),
            "pose_valid": bool(status.get("pose_valid", False)),
            "grasp_valid": bool(status.get("grasp_valid", False)),
            "grasp_latched": bool(status.get("grasp_latched", False)),
            "plane_aux_valid": bool(status.get("plane_aux_valid", False)),
            "reason": status.get("reason", ""),
            "center_error_m": debug_gt.get("center_error_m", None),
            "grasp_error_m": debug_gt.get("grasp_error_m", None),
            "axis_angle_error_deg": debug_gt.get("axis_angle_error_deg", None),
            "angle_error_deg": debug_gt.get("angle_error_deg", None),
            "angle_valid": bool(status.get("angle_valid", False)),
            "center_num_points": quality.get("center_num_points", None),
            "spoke_num_points": quality.get("spoke_num_points", None),
            "grasp_num_points": quality.get("grasp_num_points", None),
            "plane_aux_num_points": quality.get("plane_aux_num_points", None),
            "mask_area_center_px": quality.get("mask_area_center_px", None),
            "mask_area_spoke_px": quality.get("mask_area_spoke_px", None),
            "mask_area_grasp_px": quality.get("mask_area_grasp_px", None),
            "mask_area_plane_aux_px": quality.get("mask_area_plane_aux_px", None),
            "plane_fit_error_m": quality.get("plane_fit_error_m", None),
            "self_check_ok": self_check.get("ok", None),
            "self_check_max_error_m": self_check.get("max_error_m", None),
            "self_check_max_core_error_m": self_check.get("max_core_error_m", None),
            "self_check_max_marker_error_m": self_check.get("max_marker_error_m", None),
            "self_check_max_all_error_m": self_check.get("max_all_error_m", None),
            "self_check_center_error_m": (
                self_check.get("points", {}).get("center", {}).get("error_m", None)
            ),
            "self_check_grasp_error_m": (
                self_check.get("points", {}).get("grasp", {}).get("error_m", None)
            ),
            "self_check_center_marker_error_m": (
                self_check.get("marker_points", {}).get("center_marker", {}).get("error_m", None)
            ),
            "self_check_spoke_marker_error_m": (
                self_check.get("marker_points", {}).get("spoke_marker", {}).get("error_m", None)
            ),
            "self_check_grasp_marker_error_m": (
                self_check.get("marker_points", {}).get("grasp_marker", {}).get("error_m", None)
            ),
            "plane_aux_self_check_error_m": (
                self_check.get("marker_points", {}).get("plane_aux_marker", {}).get("error_m", None)
            ),
        }
        fieldnames = list(row.keys())
        write_header = not self._valve_vision_metrics_header_written or not os.path.exists(
            self.valve_vision_metrics_file
        )
        with open(self.valve_vision_metrics_file, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._valve_vision_metrics_header_written = True
            writer.writerow(row)

    def _write_valve_vision_debug_images(self, rgb, depth, debug, status, self_check, K, T_base_cam):
        if self.valve_vision_debug_image_hz <= 0.0:
            return
        interval = 1.0 / self.valve_vision_debug_image_hz
        if self.t - self._valve_vision_last_debug_image_t < interval:
            return
        self._valve_vision_last_debug_image_t = self.t
        try:
            import cv2
        except Exception as exc:
            self.logger.warning(f"Failed to import cv2 for valve vision debug images: {exc}")
            return
        os.makedirs(self.valve_vision_debug_dir, exist_ok=True)
        overlay = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        mask_img = np.zeros_like(overlay)
        colors = {
            "center": (255, 255, 0),
            "spoke": (255, 0, 255),
            "grasp": (0, 255, 0),
            "plane_aux": (255, 0, 0),
        }
        masks = (debug or {}).get("masks", {}) if debug is not None else {}
        for name, mask in masks.items():
            mask = np.asarray(mask, dtype=np.uint8)
            color = colors.get(name, (255, 255, 255))
            mask_img[mask > 0] = color
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
        if status.get("pose_valid", status.get("valid", False)):
            points = []
            labels = []
            grasp_label = "G*" if status.get("grasp_latched", False) else "G"
            for key, label in (("center_base", "C"), ("grasp_base", grasp_label), ("plane_aux_base", "A")):
                if key in status:
                    points.append(status[key])
                    labels.append(label)
            if points:
                pixels, _, visible = self._valve_vision_project_fn(np.asarray(points, dtype=float), K, T_base_cam)
                for pixel, is_visible, label in zip(pixels, visible, labels):
                    if not is_visible or not np.all(np.isfinite(pixel)):
                        continue
                    u, v = int(round(pixel[0])), int(round(pixel[1]))
                    cv2.circle(overlay, (u, v), 5, (255, 255, 255), -1)
                    cv2.putText(overlay, label, (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            arrow_specs = []
            if "center_base" in status and "spoke_dir_base" in status:
                center = np.asarray(status["center_base"], dtype=float)
                spoke = np.asarray(status["spoke_dir_base"], dtype=float)
                arrow_specs.append((center, center + 0.08 * spoke, "S", (255, 255, 255)))
            if "center_base" in status and "axis_base" in status:
                center = np.asarray(status["center_base"], dtype=float)
                axis = np.asarray(status["axis_base"], dtype=float)
                arrow_specs.append((center, center + 0.06 * axis, "X", (255, 255, 255)))
            for start, end, label, color in arrow_specs:
                pixels, _, visible = self._valve_vision_project_fn(np.vstack((start, end)), K, T_base_cam)
                if not bool(visible[0]) or not bool(visible[1]) or not np.all(np.isfinite(pixels)):
                    continue
                p0 = (int(round(pixels[0, 0])), int(round(pixels[0, 1])))
                p1 = (int(round(pixels[1, 0])), int(round(pixels[1, 1])))
                cv2.arrowedLine(overlay, p0, p1, color, 2, tipLength=0.25)
                cv2.putText(overlay, label, (p1[0] + 6, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for name, item in (self_check or {}).get("points", {}).items():
            pixel = item.get("pixel", None)
            if pixel is None:
                continue
            u, v = int(round(pixel[0])), int(round(pixel[1]))
            cv2.drawMarker(overlay, (u, v), (240, 240, 240), markerType=cv2.MARKER_CROSS, markerSize=12)
            cv2.putText(overlay, f"GT {name}", (u + 5, v + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
        depth_arr = np.asarray(depth, dtype=float)
        valid_depth = depth_arr[np.isfinite(depth_arr) & (depth_arr > 0.0)]
        if valid_depth.size > 0:
            d_min = float(np.percentile(valid_depth, 2.0))
            d_max = float(np.percentile(valid_depth, 98.0))
            denom = max(1e-6, d_max - d_min)
            depth_norm = np.clip((depth_arr - d_min) / denom, 0.0, 1.0)
            depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
        else:
            depth_color = np.zeros_like(overlay)
        cv2.imwrite(os.path.join(self.valve_vision_debug_dir, "falcon_valve_rgb_overlay.png"), overlay)
        cv2.imwrite(os.path.join(self.valve_vision_debug_dir, "falcon_valve_color_masks.png"), mask_img)
        cv2.imwrite(os.path.join(self.valve_vision_debug_dir, "falcon_valve_depth_debug.png"), depth_color)

    def _maybe_update_valve_vision(self):
        if not self.valve_vision_enable or not self._valve_vision_ready:
            return
        if not self._ensure_valve_vision_renderer():
            return
        if self._valve_vision_last_step_t is not None and abs(self.t - self._valve_vision_last_step_t) < 1e-12:
            return
        self._valve_vision_last_step_t = float(self.t)
        if self.valve_vision_hz <= 0.0:
            return
        if self.t - self._valve_vision_last_update_t < 1.0 / self.valve_vision_hz:
            return
        self._valve_vision_last_update_t = float(self.t)

        status = None
        debug = None
        self_check = None
        try:
            rgb, depth = self._render_valve_vision_rgbd()
            K = self._valve_vision_intrinsics()
            T_base_cam = self._valve_vision_T_base_cam()
            self_check = self._valve_vision_self_check(depth, K, T_base_cam)
            if not self_check.get("ok", False):
                status = self._invalid_valve_vision_status("camera_geometry_self_check_failed", self_check)
            else:
                result = self._valve_vision_estimate_fn(
                    rgb,
                    depth,
                    K,
                    T_base_cam,
                    self._valve_vision_config(),
                    prev_state=self._valve_vision_state,
                )
                status = result["status"]
                self._valve_vision_state = result.get("state", self._valve_vision_state)
                debug = result.get("debug", None)
                status.setdefault("quality", {})
                status["quality"]["self_check"] = self_check
                status["camera_name"] = self.valve_vision_camera_name
                status["image_width"] = self.valve_vision_width
                status["image_height"] = self.valve_vision_height
            debug_gt = self._valve_vision_debug_gt(status)
            if debug_gt is not None:
                status["debug_gt"] = debug_gt
            self._valve_vision_atomic_write_json_fn(self.valve_vision_status_file, status)
            self._append_valve_vision_metrics(status)
            self._write_valve_vision_debug_images(rgb, depth, debug, status, self_check, K, T_base_cam)
        except Exception as exc:
            self.logger.warning(f"Valve vision update failed: {exc}")
            if self._valve_vision_atomic_write_json_fn is not None and self.valve_vision_status_file:
                status = self._invalid_valve_vision_status(f"exception:{type(exc).__name__}", self_check)
                try:
                    self._valve_vision_atomic_write_json_fn(self.valve_vision_status_file, status)
                    self._append_valve_vision_metrics(status)
                except Exception:
                    pass

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
        root_lin_vel, root_ang_vel = self._root_velocity_world()
        left_foot_pos = self._site_centroid_world(self.left_foot_contact_site_ids)
        right_foot_pos = self._site_centroid_world(self.right_foot_contact_site_ids)
        if left_foot_pos is None:
            left_foot_pos = np.full(3, np.nan, dtype=float)
        if right_foot_pos is None:
            right_foot_pos = np.full(3, np.nan, dtype=float)

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
        contact_role_counts, contact_role_forces = self._right_hand_valve_contact_role_summary()
        real_contact_role_counts, real_contact_role_forces = (
            self._right_hand_valve_contact_role_summary(include_proxy=False)
        )
        contact_force_buckets = self._right_hand_valve_contact_force_decomposition()

        payload = {
            "timestamp": time.time(),
            "sim_time": float(self.t),
            "base_pos_world": base_pos.tolist(),
            "base_xmat_world": base_xmat.tolist(),
            "root_lin_vel_world": root_lin_vel.tolist(),
            "root_ang_vel_world": root_ang_vel.tolist(),
            "left_foot_pos_world": left_foot_pos.tolist(),
            "right_foot_pos_world": right_foot_pos.tolist(),
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
            "right_hand_valve_contact_role_counts": contact_role_counts,
            "right_hand_valve_contact_role_forces": contact_role_forces,
            "right_hand_valve_real_contact_role_counts": real_contact_role_counts,
            "right_hand_valve_real_contact_role_forces": real_contact_role_forces,
            "debug_ncon": int(self.mj_data.ncon),
            "debug_contact_pairs": self._debug_contact_pairs(),
        }
        payload.update(
            self._flatten_contact_force_bucket(
                "right_hand_valve_contact", contact_force_buckets["total"]
            )
        )
        payload.update(
            self._flatten_contact_force_bucket(
                "right_hand_valve_real_contact", contact_force_buckets["real"]
            )
        )
        payload.update(
            self._flatten_contact_force_bucket(
                "right_hand_valve_proxy_contact", contact_force_buckets["proxy"]
            )
        )
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
        self._maybe_update_valve_vision()


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
