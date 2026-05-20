#!/usr/bin/env python3
"""Live sanity check for a calibrated T_base_camera on ROS1."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Optional, Tuple

import numpy as np


try:
    from .calibration_utils import (
        aruco_dictionary,
        detect_aruco_markers,
        estimate_tag_pose,
        load_extrinsic,
        median_depth_near_pixel,
        reprojection_error_px,
        select_marker,
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from calibration_utils import (  # type: ignore
        aruco_dictionary,
        detect_aruco_markers,
        estimate_tag_pose,
        load_extrinsic,
        median_depth_near_pixel,
        reprojection_error_px,
        select_marker,
    )


def import_ros1():
    try:
        import cv2  # type: ignore
        import rospy  # type: ignore
        from cv_bridge import CvBridge  # type: ignore
        from sensor_msgs.msg import CameraInfo, Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "This live check must run on the ROS1 Noetic RealSense host with rospy, "
            "cv_bridge, sensor_msgs, and OpenCV available."
        ) from exc
    return cv2, rospy, CvBridge, CameraInfo, Image


def build_parser() -> argparse.ArgumentParser:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Check T_base_camera live with an ArUco marker.")
    parser.add_argument("--tag-id", type=int, required=True)
    parser.add_argument("--tag-size-m", type=float, required=True)
    parser.add_argument(
        "--extrinsic",
        default=os.path.join(script_dir, "output", "T_base_camera.yaml"),
        help="YAML/JSON file containing T_base_camera.",
    )
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--depth-scale-m", type=float, default=0.001)
    parser.add_argument("--depth-window-px", type=int, default=7)
    parser.add_argument("--max-depth-age-s", type=float, default=0.25)
    parser.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    parser.add_argument("--print-hz", type=float, default=5.0)
    return parser


class LiveChecker:
    def __init__(self, args: argparse.Namespace) -> None:
        cv2, rospy, CvBridge, CameraInfo, Image = import_ros1()
        self.cv2 = cv2
        self.rospy = rospy
        self.bridge = CvBridge()
        self.args = args
        self.t_base_camera = load_extrinsic(args.extrinsic)
        self.r_base_camera = self.t_base_camera[:3, :3]
        self.p_base_camera = self.t_base_camera[:3, 3]
        aruco_dictionary(args.aruco_dictionary)

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.depth_encoding = ""
        self.depth_stamp: Optional[float] = None
        self.last_print_t = 0.0

        self._warn_about_missing_topics()
        rospy.Subscriber(args.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        if not args.no_depth:
            rospy.Subscriber(args.depth_topic, Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(args.image_topic, Image, self._image_cb, queue_size=1)

        rospy.loginfo("Loaded T_base_camera from %s", args.extrinsic)
        rospy.loginfo("Sign check:")
        rospy.loginfo("  move tag toward robot front: p_base.x should increase")
        rospy.loginfo("  move tag toward robot left:  p_base.y should increase")
        rospy.loginfo("  move tag upward:             p_base.z should increase")

    def _warn_about_missing_topics(self) -> None:
        try:
            published = {name for name, _ in self.rospy.get_published_topics()}
        except Exception as exc:
            self.rospy.logwarn("could not query published topics: %s", exc)
            return
        for topic in (self.args.image_topic, self.args.camera_info_topic):
            if topic not in published:
                self.rospy.logwarn("topic is not currently published: %s", topic)
        if not self.args.no_depth and self.args.depth_topic not in published:
            self.rospy.logwarn("depth topic is not currently published: %s", self.args.depth_topic)

    def _camera_info_cb(self, msg: Any) -> None:
        self.camera_matrix = np.asarray(msg.K, dtype=float).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.D, dtype=float).reshape(-1)

    def _depth_cb(self, msg: Any) -> None:
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_encoding = str(msg.encoding)
            self.depth_stamp = float(msg.header.stamp.to_sec())
        except Exception as exc:
            self.rospy.logwarn_throttle(1.0, "depth conversion failed: %s", exc)

    def _image_cb(self, msg: Any) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.rospy.logwarn_throttle(1.0, "camera_info not received yet: %s", self.args.camera_info_topic)
            return
        now = time.time()
        if self.args.print_hz > 0.0 and now - self.last_print_t < 1.0 / float(self.args.print_hz):
            return
        self.last_print_t = now

        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            stamp = float(msg.header.stamp.to_sec())
            p_cam, p_base, depth_m, tag_id, reproj = self._estimate(image_bgr, stamp)
        except LookupError as exc:
            self.rospy.logwarn_throttle(1.0, "%s", exc)
            return
        except Exception as exc:
            self.rospy.logwarn_throttle(1.0, "live check estimate failed: %s", exc)
            return

        depth_text = "null" if depth_m is None else f"{depth_m:.3f}"
        self.rospy.loginfo(
            "tag_id=%d p_cam=[%.3f %.3f %.3f] p_base=[%.3f %.3f %.3f] depth_m=%s reproj=%.3fpx",
            tag_id,
            p_cam[0],
            p_cam[1],
            p_cam[2],
            p_base[0],
            p_base[1],
            p_base[2],
            depth_text,
            reproj,
        )

    def _estimate(self, image_bgr: np.ndarray, stamp: float) -> Tuple[np.ndarray, np.ndarray, Optional[float], int, float]:
        gray = self.cv2.cvtColor(image_bgr, self.cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_aruco_markers(gray, self.args.aruco_dictionary)
        _, detected_id, selected_corners = select_marker(corners, ids, self.args.tag_id)
        rvec, tvec = estimate_tag_pose(
            selected_corners,
            self.args.tag_size_m,
            self.camera_matrix,
            self.dist_coeffs,
        )
        reproj = reprojection_error_px(
            selected_corners,
            self.args.tag_size_m,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        p_cam = np.asarray(tvec, dtype=float).reshape(3)
        p_base = self.r_base_camera @ p_cam + self.p_base_camera
        depth_m = self._depth_for_tag(selected_corners, stamp)
        return p_cam, p_base, depth_m, int(detected_id), float(reproj)

    def _depth_for_tag(self, corners_px: np.ndarray, stamp: float) -> Optional[float]:
        if self.args.no_depth:
            return None
        if self.depth_image is None or self.depth_stamp is None:
            self.rospy.logwarn_throttle(2.0, "depth image not received yet: %s", self.args.depth_topic)
            return None
        if abs(float(stamp) - float(self.depth_stamp)) > float(self.args.max_depth_age_s):
            self.rospy.logwarn_throttle(
                2.0,
                "depth image too old: age=%.3fs",
                abs(float(stamp) - float(self.depth_stamp)),
            )
            return None
        center_uv = np.mean(np.asarray(corners_px, dtype=float).reshape(4, 2), axis=0)
        depth_m, _ = median_depth_near_pixel(
            self.depth_image,
            self.depth_encoding,
            center_uv,
            depth_scale_m=self.args.depth_scale_m,
            window_px=self.args.depth_window_px,
        )
        if depth_m is None:
            self.rospy.logwarn_throttle(2.0, "depth invalid near tag center")
        return depth_m


def main() -> int:
    args = build_parser().parse_args()
    try:
        _, rospy, _, _, _ = import_ros1()
        rospy.init_node("check_T_base_camera_live", anonymous=True)
        LiveChecker(args)
        rospy.spin()
    except Exception as exc:
        print(f"T_base_camera live check failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
