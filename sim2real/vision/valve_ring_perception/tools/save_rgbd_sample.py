#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_transform(path: str | None, allow_identity: bool = False) -> np.ndarray:
    if path is None:
        if allow_identity:
            return np.eye(4, dtype=float)
        raise ValueError("pass --T-base-cam-json so each sample contains T_base_cam")
    data = _load_json(path)
    if isinstance(data, list):
        return np.asarray(data, dtype=float).reshape(4, 4)
    if "T_base_cam" in data:
        return np.asarray(data["T_base_cam"], dtype=float).reshape(4, 4)
    if "rotation_matrix" in data and "translation_m" in data:
        out = np.eye(4, dtype=float)
        out[:3, :3] = np.asarray(data["rotation_matrix"], dtype=float).reshape(3, 3)
        out[:3, 3] = np.asarray(data["translation_m"], dtype=float).reshape(3)
        return out
    raise ValueError("unsupported T_base_cam JSON; expected 4x4, T_base_cam, or rotation_matrix+translation_m")


def _stamp_to_sec(stamp: Any) -> float:
    if stamp is None:
        return time.time()
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return float(stamp)


def _frame_path(output_dir: str, index: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"frame_{index:06d}.npz")


def _depth_to_storage(
    depth: np.ndarray,
    encoding: str,
    depth_scale_m: float,
    depth_output: str,
) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(depth)
    raw_scale = 1.0
    if encoding in ("16UC1", "mono16") or np.issubdtype(arr.dtype, np.integer):
        raw_scale = float(depth_scale_m)
    if depth_output == "raw":
        return arr.copy(), raw_scale, raw_scale
    if raw_scale != 1.0:
        return arr.astype(np.float32) * raw_scale, 1.0, raw_scale
    return arr.astype(np.float32), 1.0, raw_scale


def _save_npz(
    path: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    timestamp: float,
    depth_scale_m: float,
    raw_depth_scale_m: float,
    metadata: dict[str, Any],
) -> None:
    np.savez_compressed(
        path,
        rgb=np.asarray(rgb, dtype=np.uint8),
        depth=depth,
        K=np.asarray(K, dtype=float).reshape(3, 3),
        T_base_cam=np.asarray(T_base_cam, dtype=float).reshape(4, 4),
        timestamp=float(timestamp),
        depth_scale_m=float(depth_scale_m),
        raw_depth_scale_m=float(raw_depth_scale_m),
        **metadata,
    )


class RosRgbdRecorder:
    def __init__(self, args: argparse.Namespace, T_base_cam: np.ndarray):
        try:
            import rospy  # type: ignore
            from cv_bridge import CvBridge  # type: ignore
            from sensor_msgs.msg import CameraInfo, Image  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ROS1 capture requires rospy, cv_bridge, and sensor_msgs in the active environment") from exc

        self.rospy = rospy
        self.Image = Image
        self.CameraInfo = CameraInfo
        self.bridge = CvBridge()
        self.args = args
        self.T_base_cam = T_base_cam
        self.condition = threading.Condition()
        self.rgb: np.ndarray | None = None
        self.rgb_stamp: float | None = None
        self.depth: np.ndarray | None = None
        self.depth_encoding = ""
        self.depth_stamp: float | None = None
        self.K: np.ndarray | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.last_saved_stamp: float | None = None

    def start(self) -> None:
        self.rospy.init_node("save_rgbd_sample", anonymous=True)
        self.rospy.Subscriber(self.args.camera_info_topic, self.CameraInfo, self._camera_info_cb, queue_size=1)
        self.rospy.Subscriber(self.args.depth_topic, self.Image, self._depth_cb, queue_size=1)
        self.rospy.Subscriber(self.args.rgb_topic, self.Image, self._rgb_cb, queue_size=1)

    def _camera_info_cb(self, msg: Any) -> None:
        with self.condition:
            self.K = np.asarray(msg.K, dtype=float).reshape(3, 3)
            self.width = int(msg.width)
            self.height = int(msg.height)
            self.condition.notify_all()

    def _depth_cb(self, msg: Any) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.rospy.logwarn_throttle(1.0, "depth conversion failed: %s", exc)
            return
        with self.condition:
            self.depth = np.asarray(depth).copy()
            self.depth_encoding = str(msg.encoding)
            self.depth_stamp = _stamp_to_sec(msg.header.stamp)
            self.condition.notify_all()

    def _rgb_cb(self, msg: Any) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            self.rospy.logwarn_throttle(1.0, "rgb conversion failed: %s", exc)
            return
        with self.condition:
            self.rgb = np.asarray(rgb, dtype=np.uint8).copy()
            self.rgb_stamp = _stamp_to_sec(msg.header.stamp)
            self.condition.notify_all()

    def _ready_locked(self) -> bool:
        if self.rgb is None or self.depth is None or self.K is None or self.rgb_stamp is None or self.depth_stamp is None:
            return False
        if self.last_saved_stamp is not None and self.rgb_stamp <= self.last_saved_stamp:
            return False
        return abs(self.rgb_stamp - self.depth_stamp) <= float(self.args.max_depth_age_s)

    def wait_sample(self, timeout_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
        deadline = time.time() + float(timeout_s)
        with self.condition:
            while not self.rospy.is_shutdown():
                if self._ready_locked():
                    assert self.rgb is not None and self.depth is not None and self.K is not None and self.rgb_stamp is not None
                    self.last_saved_stamp = self.rgb_stamp
                    return (
                        self.rgb.copy(),
                        self.depth.copy(),
                        self.K.copy(),
                        float(self.rgb_stamp),
                        str(self.depth_encoding),
                    )
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for synchronized RGB/depth/camera_info")
                self.condition.wait(timeout=min(0.1, remaining))
        raise RuntimeError("ROS shutdown while waiting for sample")

    def run(self) -> None:
        self.start()
        saved = 0
        next_index = int(self.args.start_index)
        while not self.rospy.is_shutdown() and saved < int(self.args.num_frames):
            rgb, depth_raw, K, timestamp, depth_encoding = self.wait_sample(self.args.timeout_s)
            depth, stored_depth_scale, raw_depth_scale = _depth_to_storage(
                depth_raw,
                depth_encoding,
                self.args.depth_scale_m,
                self.args.depth_output,
            )
            path = _frame_path(self.args.output_dir, next_index)
            _save_npz(
                path,
                rgb,
                depth,
                K,
                self.T_base_cam,
                timestamp,
                stored_depth_scale,
                raw_depth_scale,
                {
                    "depth_encoding": depth_encoding,
                    "rgb_topic": self.args.rgb_topic,
                    "depth_topic": self.args.depth_topic,
                    "camera_info_topic": self.args.camera_info_topic,
                },
            )
            print(f"saved {path} timestamp={timestamp:.6f} depth_scale_m={stored_depth_scale}")
            saved += 1
            next_index += 1
            if self.args.interval_s > 0.0 and saved < int(self.args.num_frames):
                self.rospy.sleep(float(self.args.interval_s))


def _run_realsense(args: argparse.Namespace, T_base_cam: np.ndarray) -> None:
    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RealSense backend requires pyrealsense2 in the active environment") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    sensor_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    raw_depth_scale = float(args.depth_scale_m if args.depth_scale_m is not None else sensor_scale)

    try:
        for _ in range(max(0, int(args.warmup_frames))):
            align.process(pipeline.wait_for_frames())
        for i in range(int(args.num_frames)):
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("missing RealSense color or depth frame")
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=float)
            rgb = np.asanyarray(color_frame.get_data()).copy()
            depth_raw = np.asanyarray(depth_frame.get_data()).copy()
            depth, stored_depth_scale, saved_raw_scale = _depth_to_storage(depth_raw, "16UC1", raw_depth_scale, args.depth_output)
            frame_index = int(args.start_index) + i
            path = _frame_path(args.output_dir, frame_index)
            _save_npz(
                path,
                rgb,
                depth,
                K,
                T_base_cam,
                time.time(),
                stored_depth_scale,
                saved_raw_scale,
                {
                    "backend": "realsense",
                    "width": int(intr.width),
                    "height": int(intr.height),
                },
            )
            print(f"saved {path} depth_scale_m={stored_depth_scale}")
            if args.interval_s > 0.0 and i + 1 < int(args.num_frames):
                time.sleep(float(args.interval_s))
    finally:
        pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Save aligned D435i RGB-D frames as frame_XXXXXX.npz.")
    parser.add_argument("--backend", choices=("ros1", "realsense"), default="ros1")
    parser.add_argument("--output-dir", default="artifacts/valve_ring_perception/samples")
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--interval-s", type=float, default=0.0)
    parser.add_argument("--T-base-cam-json", help="4x4 transform JSON, or rotation_matrix+translation_m JSON")
    parser.add_argument("--allow-identity-T-base-cam", action="store_true")
    parser.add_argument("--depth-scale-m", type=float, default=0.001)
    parser.add_argument("--depth-output", choices=("meters", "raw"), default="meters")

    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--max-depth-age-s", type=float, default=0.20)
    parser.add_argument("--timeout-s", type=float, default=10.0)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()

    T_base_cam = _load_transform(args.T_base_cam_json, allow_identity=args.allow_identity_T_base_cam)
    if args.backend == "ros1":
        RosRgbdRecorder(args, T_base_cam).run()
    else:
        _run_realsense(args, T_base_cam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
