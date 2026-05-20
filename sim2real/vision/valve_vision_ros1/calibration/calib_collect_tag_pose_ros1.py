#!/usr/bin/env python3
"""Collect ArUco T_camera_tag samples on a ROS1 Noetic RealSense host."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import threading
import time
from typing import Any, List, Optional, Tuple

import numpy as np


try:
    from .calibration_utils import (
        append_sample,
        aruco_dictionary,
        average_rotation_matrices,
        cv2_rodrigues,
        detect_aruco_markers,
        estimate_tag_pose,
        load_sample_file,
        median_depth_near_pixel,
        reprojection_error_px,
        rotation_error_deg,
        select_marker,
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from calibration_utils import (  # type: ignore
        append_sample,
        aruco_dictionary,
        average_rotation_matrices,
        cv2_rodrigues,
        detect_aruco_markers,
        estimate_tag_pose,
        load_sample_file,
        median_depth_near_pixel,
        reprojection_error_px,
        rotation_error_deg,
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
            "This collector must run on the ROS1 Noetic RealSense host with rospy, "
            "cv_bridge, sensor_msgs, and OpenCV available."
        ) from exc
    return cv2, rospy, CvBridge, CameraInfo, Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect T_camera_tag samples from a ROS1 color stream.")
    parser.add_argument("--tag-id", type=int, required=True, help="ArUco marker id to collect.")
    parser.add_argument("--tag-size-m", type=float, required=True, help="Marker side length in meters.")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "T_camera_tag_samples.jsonl"),
        help="Output .jsonl/.json/.yaml file.",
    )
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--no-depth", action="store_true", help="Do not subscribe to aligned depth.")
    parser.add_argument("--depth-scale-m", type=float, default=0.001, help="Scale for uint16 depth images.")
    parser.add_argument("--depth-window-px", type=int, default=7, help="Median depth window around tag center.")
    parser.add_argument("--max-depth-age-s", type=float, default=0.25)
    parser.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    parser.add_argument(
        "--auto-save-stable",
        action="store_true",
        help="Automatically save when the recent pose window is stable.",
    )
    parser.add_argument("--stable-frame-count", type=int, default=12)
    parser.add_argument("--stable-translation-std-m", type=float, default=0.003)
    parser.add_argument("--stable-rotation-std-deg", type=float, default=1.0)
    parser.add_argument("--min-save-interval-s", type=float, default=2.0)
    return parser


class TagPoseCollector:
    def __init__(self, args: argparse.Namespace) -> None:
        cv2, rospy, CvBridge, CameraInfo, Image = import_ros1()
        self.cv2 = cv2
        self.rospy = rospy
        self.bridge = CvBridge()
        self.args = args

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.depth_encoding = ""
        self.depth_stamp: Optional[float] = None
        self.last_sample: Optional[dict] = None
        self.lock = threading.Lock()
        self.existing_samples: List[dict] = self._load_existing_samples(args.output)
        self.next_sample_id = len(self.existing_samples)
        self.recent_poses: List[Tuple[np.ndarray, np.ndarray]] = []
        self.last_auto_save_t = 0.0

        aruco_dictionary(args.aruco_dictionary)
        self._warn_about_missing_topics()
        rospy.Subscriber(args.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        if not args.no_depth:
            rospy.Subscriber(args.depth_topic, Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(args.image_topic, Image, self._image_cb, queue_size=1)

    def _load_existing_samples(self, path: str) -> List[dict]:
        if not os.path.exists(path):
            return []
        try:
            samples = load_sample_file(path)
            return [dict(item) for item in samples]
        except Exception as exc:
            self.rospy.logwarn("could not read existing output samples from %s: %s", path, exc)
            return []

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
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.rospy.logwarn_throttle(1.0, "color conversion failed: %s", exc)
            return

        stamp = float(msg.header.stamp.to_sec())
        try:
            sample = self._estimate_sample(image_bgr, stamp)
        except LookupError as exc:
            self.rospy.logwarn_throttle(1.0, "%s", exc)
            return
        except Exception as exc:
            self.rospy.logwarn_throttle(1.0, "tag pose estimation failed: %s", exc)
            return

        with self.lock:
            self.last_sample = sample
        self._update_stability(sample)
        self._print_detection(sample)
        if self.args.auto_save_stable and self._is_stable_now():
            now = time.time()
            if now - self.last_auto_save_t >= float(self.args.min_save_interval_s):
                self.last_auto_save_t = now
                self.save_latest("stable")

    def _estimate_sample(self, image_bgr: np.ndarray, stamp: float) -> dict:
        gray = self.cv2.cvtColor(image_bgr, self.cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_aruco_markers(gray, self.args.aruco_dictionary)
        _, detected_id, selected_corners = select_marker(corners, ids, self.args.tag_id)

        rvec, tvec = estimate_tag_pose(
            selected_corners,
            self.args.tag_size_m,
            self.camera_matrix,
            self.dist_coeffs,
        )
        rotation = cv2_rodrigues(rvec)
        reproj = reprojection_error_px(
            selected_corners,
            self.args.tag_size_m,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        depth_median_m, _ = self._depth_for_tag(selected_corners, stamp)
        depth_error = None if depth_median_m is None else abs(float(depth_median_m) - float(tvec[2]))

        return {
            "stamp": float(stamp),
            "tag_id": int(detected_id),
            "tag_size_m": float(self.args.tag_size_m),
            "T_camera_tag": {
                "translation_m": tvec.tolist(),
                "rotation_matrix": rotation.tolist(),
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
            },
            "corners_px": selected_corners.tolist(),
            "reprojection_error_px": float(reproj),
            "depth_median_m": None if depth_median_m is None else float(depth_median_m),
            "depth_vs_pnp_error_m": None if depth_error is None else float(depth_error),
            "image_topic": self.args.image_topic,
            "camera_info_topic": self.args.camera_info_topic,
        }

    def _depth_for_tag(self, corners_px: np.ndarray, stamp: float) -> Tuple[Optional[float], int]:
        if self.args.no_depth:
            return None, 0
        if self.depth_image is None or self.depth_stamp is None:
            self.rospy.logwarn_throttle(2.0, "depth image not received yet: %s", self.args.depth_topic)
            return None, 0
        if abs(float(stamp) - float(self.depth_stamp)) > float(self.args.max_depth_age_s):
            self.rospy.logwarn_throttle(
                2.0,
                "depth image too old: age=%.3fs",
                abs(float(stamp) - float(self.depth_stamp)),
            )
            return None, 0
        center_uv = np.mean(np.asarray(corners_px, dtype=float).reshape(4, 2), axis=0)
        depth_m, count = median_depth_near_pixel(
            self.depth_image,
            self.depth_encoding,
            center_uv,
            depth_scale_m=self.args.depth_scale_m,
            window_px=self.args.depth_window_px,
        )
        if depth_m is None:
            self.rospy.logwarn_throttle(2.0, "depth invalid near tag center")
        return depth_m, count

    def _update_stability(self, sample: dict) -> None:
        tvec = np.asarray(sample["T_camera_tag"]["translation_m"], dtype=float).reshape(3)
        rotation = np.asarray(sample["T_camera_tag"]["rotation_matrix"], dtype=float).reshape(3, 3)
        self.recent_poses.append((tvec, rotation))
        max_count = max(1, int(self.args.stable_frame_count))
        if len(self.recent_poses) > max_count:
            self.recent_poses = self.recent_poses[-max_count:]

    def _is_stable_now(self) -> bool:
        count = int(self.args.stable_frame_count)
        if len(self.recent_poses) < count:
            return False
        translations = np.asarray([pose[0] for pose in self.recent_poses], dtype=float)
        translation_std = float(np.max(np.std(translations, axis=0)))
        mean_rotation = average_rotation_matrices([pose[1] for pose in self.recent_poses])
        rotation_errors = [rotation_error_deg(pose[1], mean_rotation) for pose in self.recent_poses]
        rotation_std = float(np.max(rotation_errors))
        return (
            translation_std <= float(self.args.stable_translation_std_m)
            and rotation_std <= float(self.args.stable_rotation_std_deg)
        )

    def _print_detection(self, sample: dict) -> None:
        tvec = sample["T_camera_tag"]["translation_m"]
        depth = sample["depth_median_m"]
        depth_text = "null" if depth is None else f"{depth:.3f}"
        self.rospy.loginfo_throttle(
            1.0,
            "tag_id=%d t_camera_tag=[%.3f %.3f %.3f] reproj=%.3fpx depth=%s",
            sample["tag_id"],
            tvec[0],
            tvec[1],
            tvec[2],
            sample["reprojection_error_px"],
            depth_text,
        )

    def save_latest(self, reason: str) -> bool:
        with self.lock:
            if self.last_sample is None:
                self.rospy.logwarn("no tag sample available yet; cannot save")
                return False
            sample = copy.deepcopy(self.last_sample)
            sample["sample_id"] = int(self.next_sample_id)
            sample["save_reason"] = str(reason)
            sample["save_time"] = float(time.time())
            append_sample(self.args.output, sample, self.existing_samples)
            self.next_sample_id += 1
        self.rospy.loginfo("saved sample_id=%d to %s (%s)", sample["sample_id"], self.args.output, reason)
        return True


def stdin_save_loop(collector: TagPoseCollector) -> None:
    rospy = collector.rospy
    print("Press Enter to save the latest detected tag pose sample.")
    while not rospy.is_shutdown():
        line = sys.stdin.readline()
        if line == "":
            time.sleep(0.2)
            continue
        collector.save_latest("enter")


def main() -> int:
    args = build_parser().parse_args()
    try:
        _, rospy, _, _, _ = import_ros1()
        rospy.init_node("calib_collect_tag_pose_ros1", anonymous=True)
        collector = TagPoseCollector(args)
        thread = threading.Thread(target=stdin_save_loop, args=(collector,), daemon=True)
        thread.start()
        rospy.loginfo("collecting tag pose samples; output=%s", args.output)
        rospy.spin()
    except Exception as exc:
        print(f"tag pose collection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
