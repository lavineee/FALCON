#!/usr/bin/env python3
"""ROS1 D435I valve marker detector that publishes low-dimensional JSON."""

import argparse
import json
import math
import os
import socket
import time

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


def _normalize(vec, eps=1e-9):
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return None
    return arr / norm


def _rpy_deg_to_rot(rpy_deg):
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=float).reshape(3))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _load_transform(config):
    transform = config.get("T_base_camera", {}) or {}
    translation = np.asarray(transform.get("translation_m", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    if "rotation_matrix" in transform:
        rotation = np.asarray(transform["rotation_matrix"], dtype=float).reshape(3, 3)
    else:
        rotation = _rpy_deg_to_rot(transform.get("rotation_rpy_deg", [0.0, 0.0, 0.0]))
    return rotation, translation


def _aruco_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was built without cv2.aruco")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco/AprilTag dictionary: {name}")
    dict_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def _detect_markers(gray, dictionary):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def backproject_pixel_to_point(u, v, z, camera_matrix):
    """RealSense optical frame: x right, y down, z forward."""
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    return np.array([(float(u) - cx) * z / fx, (float(v) - cy) * z / fy, z], dtype=float)


def _depth_to_meters(depth_value, encoding, depth_scale):
    value = float(depth_value)
    if not np.isfinite(value) or value <= 0.0:
        return None
    if encoding in ("16UC1", "mono16"):
        return value * float(depth_scale)
    return value


class ValveVisionNode:
    def __init__(self, config):
        self.config = config
        self.bridge = CvBridge()
        self.depth_image = None
        self.depth_encoding = ""
        self.depth_stamp = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_publish_t = 0.0

        transport = config.get("transport", {}) or {}
        self.udp_host = str(transport.get("udp_host", "192.168.123.100"))
        self.udp_port = int(transport.get("udp_port", 5055))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        marker = config.get("marker", {}) or {}
        self.dictionary = _aruco_dictionary(str(marker.get("dictionary", "DICT_4X4_50")))
        self.marker_id = marker.get("id", None)
        self.marker_id = None if self.marker_id is None else int(self.marker_id)
        self.marker_size_m = float(marker.get("size_m", 0.08))
        self.center_offset_marker = np.asarray(
            marker.get("center_offset_marker_m", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(3)
        self.grasp_offset_marker = np.asarray(
            marker.get("grasp_offset_marker_m", [0.0, -0.10, 0.0]),
            dtype=float,
        ).reshape(3)
        self.axis_marker = _normalize(marker.get("axis_marker", [0.0, 0.0, 1.0]))
        if self.axis_marker is None:
            raise ValueError("marker.axis_marker must be a nonzero 3-vector")

        self.R_base_camera, self.t_base_camera = _load_transform(config)
        self.publish_hz = float(config.get("publish_hz", 15.0))
        self.depth_scale = float(config.get("depth_scale_m", 0.001))
        self.use_depth_center = bool(config.get("use_depth_center", False))
        self.max_depth_age_s = float(config.get("max_depth_age_s", 0.20))
        self.publish_invalid = bool(config.get("publish_invalid", True))

        topics = config.get("topics", {}) or {}
        rospy.Subscriber(
            str(topics.get("camera_info", "/camera/color/camera_info")),
            CameraInfo,
            self._camera_info_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            str(topics.get("depth", "/camera/aligned_depth_to_color/image_raw")),
            Image,
            self._depth_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            str(topics.get("color", "/camera/color/image_raw")),
            Image,
            self._color_cb,
            queue_size=1,
        )

    def _camera_info_cb(self, msg):
        self.camera_matrix = np.asarray(msg.K, dtype=float).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.D, dtype=float).reshape(-1)

    def _depth_cb(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_encoding = msg.encoding
            self.depth_stamp = msg.header.stamp.to_sec()
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "depth conversion failed: %s", exc)

    def _color_cb(self, msg):
        now = time.time()
        if self.publish_hz > 0.0 and now - self.last_publish_t < 1.0 / self.publish_hz:
            return
        self.last_publish_t = now

        if self.camera_matrix is None:
            rospy.logwarn_throttle(1.0, "waiting for camera_info")
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "color conversion failed: %s", exc)
            return

        try:
            payload = self._estimate_from_color(bgr, msg.header.stamp.to_sec())
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "valve marker estimate failed: %s", exc)
            payload = self._invalid_payload("estimate_failed")
        if payload is not None:
            self._send(payload)

    def _estimate_from_color(self, bgr, stamp):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = _detect_markers(gray, self.dictionary)
        if ids is None or len(ids) == 0:
            return self._invalid_payload("marker_missing")

        ids_flat = ids.reshape(-1)
        if self.marker_id is None:
            marker_index = 0
        else:
            matches = np.where(ids_flat == self.marker_id)[0]
            if len(matches) == 0:
                return self._invalid_payload("configured_marker_missing")
            marker_index = int(matches[0])

        selected_corners = [corners[marker_index]]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            selected_corners,
            self.marker_size_m,
            self.camera_matrix,
            self.dist_coeffs,
        )
        rvec = np.asarray(rvecs[0], dtype=float).reshape(3)
        marker_center_cam = np.asarray(tvecs[0], dtype=float).reshape(3)
        R_cam_marker, _ = cv2.Rodrigues(rvec)

        valid_depth_points = 0
        depth_center_cam = None
        if self.use_depth_center:
            depth_center_cam, valid_depth_points = self._depth_center_from_marker(selected_corners[0], stamp)
            if depth_center_cam is not None:
                marker_center_cam = depth_center_cam

        center_cam = marker_center_cam + R_cam_marker @ self.center_offset_marker
        grasp_cam = marker_center_cam + R_cam_marker @ self.grasp_offset_marker
        axis_cam = _normalize(R_cam_marker @ self.axis_marker)
        if axis_cam is None:
            return self._invalid_payload("axis_invalid")

        center_base = self.R_base_camera @ center_cam + self.t_base_camera
        grasp_base = self.R_base_camera @ grasp_cam + self.t_base_camera
        axis_base = _normalize(self.R_base_camera @ axis_cam)
        if axis_base is None:
            return self._invalid_payload("axis_invalid")

        quality = {
            "center_num_points": int(max(4, valid_depth_points)),
            "spoke_num_points": 0,
            "grasp_num_points": int(max(4, valid_depth_points)),
            "plane_aux_num_points": 0,
            "plane_fit_error_m": 0.0,
            "marker_id": int(ids_flat[marker_index]),
            "marker_size_m": float(self.marker_size_m),
            "valid_depth_points": int(valid_depth_points),
        }
        return {
            "timestamp": time.time(),
            "image_timestamp": float(stamp),
            "frame": "base",
            "source": "g1_d435i_ros1_marker",
            "valid": True,
            "pose_valid": True,
            "grasp_valid": True,
            "grasp_latched": False,
            "plane_aux_valid": False,
            "center_base": center_base.tolist(),
            "grasp_base": grasp_base.tolist(),
            "axis_base": axis_base.tolist(),
            "quality": quality,
            "angle_valid": False,
            "valve_angle": None,
            "valve_vel": None,
        }

    def _depth_center_from_marker(self, marker_corners, stamp):
        if self.depth_image is None or self.depth_stamp is None:
            return None, 0
        if abs(float(stamp) - float(self.depth_stamp)) > self.max_depth_age_s:
            return None, 0

        pts = np.asarray(marker_corners, dtype=float).reshape(-1, 2)
        u_min, v_min = np.floor(np.min(pts, axis=0)).astype(int)
        u_max, v_max = np.ceil(np.max(pts, axis=0)).astype(int)
        h, w = self.depth_image.shape[:2]
        u_min, u_max = np.clip([u_min, u_max], 0, w - 1)
        v_min, v_max = np.clip([v_min, v_max], 0, h - 1)
        if u_max <= u_min or v_max <= v_min:
            return None, 0

        patch = self.depth_image[v_min : v_max + 1, u_min : u_max + 1]
        depths = []
        for value in patch.reshape(-1):
            z = _depth_to_meters(value, self.depth_encoding, self.depth_scale)
            if z is not None:
                depths.append(z)
        if not depths:
            return None, 0
        z = float(np.median(depths))
        center = np.mean(pts, axis=0)
        return backproject_pixel_to_point(center[0], center[1], z, self.camera_matrix), len(depths)

    def _invalid_payload(self, reason):
        if not self.publish_invalid:
            return None
        return {
            "timestamp": time.time(),
            "frame": "base",
            "source": "g1_d435i_ros1_marker",
            "valid": False,
            "pose_valid": False,
            "grasp_valid": False,
            "reason": reason,
            "angle_valid": False,
        }

    def _send(self, payload):
        data = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.sock.sendto(data, (self.udp_host, self.udp_port))
        if payload.get("valid", False):
            rospy.loginfo_throttle(
                1.0,
                "valve target sent center=%s grasp=%s axis=%s",
                np.round(payload["center_base"], 4).tolist(),
                np.round(payload["grasp_base"], 4).tolist(),
                np.round(payload["axis_base"], 4).tolist(),
            )


def main():
    parser = argparse.ArgumentParser(description="G1 ROS1 marker-based valve vision node")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "valve_vision_g1.yaml"),
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file) or {}

    rospy.init_node("falcon_valve_vision_node")
    ValveVisionNode(config)
    rospy.spin()


if __name__ == "__main__":
    main()
