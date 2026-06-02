#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.vision.valve_ring_perception import CameraIntrinsics, ValveRingEstimator, ValveRingEstimatorConfig
from sim2real.vision.valve_ring_perception.status_writer import atomic_write_json
from sim2real.vision.valve_ring_perception.temporal_filter import TemporalEstimateFilter
from sim2real.vision.valve_ring_perception.visualization import render_debug_overlay


def _load_config_data(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config file not found: {path}. Omit --config to use defaults, "
            "or pass an existing estimator config JSON/YAML."
        )
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON or install pyyaml") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return data or {}


def _load_transform_from_data(data: Any) -> np.ndarray:
    if isinstance(data, list):
        return np.asarray(data, dtype=float).reshape(4, 4)
    if isinstance(data, dict):
        if "T_base_cam" in data:
            return _load_transform_from_data(data["T_base_cam"])
        if "T_base_camera" in data:
            return _load_transform_from_data(data["T_base_camera"])
        if "rotation_matrix" in data and "translation_m" in data:
            out = np.eye(4, dtype=float)
            out[:3, :3] = np.asarray(data["rotation_matrix"], dtype=float).reshape(3, 3)
            out[:3, 3] = np.asarray(data["translation_m"], dtype=float).reshape(3)
            return out
    raise ValueError("unsupported transform format; expected 4x4, T_base_cam, or rotation_matrix+translation_m")


def _load_transform(path: str | None, config_data: dict[str, Any]) -> tuple[np.ndarray, str]:
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"T_base_cam file not found: {path}. Omit --T-base-cam-json to use identity "
                "for camera-frame testing, or pass an existing calibration JSON."
            )
        with open(path, "r", encoding="utf-8") as file:
            return _load_transform_from_data(json.load(file)), path
    for key in ("T_base_cam", "T_base_camera"):
        if key in config_data:
            return _load_transform_from_data({key: config_data[key]}), f"config:{key}"
    print("warning: no T_base_cam provided; using identity transform for camera-frame testing")
    return np.eye(4, dtype=float), "identity"


def _next_frame_index(output_dir: Path) -> int:
    existing = sorted(output_dir.glob("frame_*.npz"))
    if not existing:
        return 0
    max_index = -1
    for path in existing:
        try:
            max_index = max(max_index, int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max_index + 1


def _save_frame(
    output_dir: Path,
    frame_index: int,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    timestamp: float,
    raw_depth_scale_m: float,
    coordinate_frame: str,
    source: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frame_{frame_index:06d}.npz"
    depth_arr = np.asarray(depth_m, dtype=np.float32)
    np.savez_compressed(
        path,
        rgb=np.asarray(rgb, dtype=np.uint8),
        depth=depth_arr,
        depth_m=depth_arr,
        K=np.asarray(K, dtype=float).reshape(3, 3),
        T_base_cam=np.asarray(T_base_cam, dtype=float).reshape(4, 4),
        timestamp=float(timestamp),
        depth_scale_m=1.0,
        raw_depth_scale_m=float(raw_depth_scale_m),
        frame=np.asarray(coordinate_frame),
        source=np.asarray(source),
    )
    return path


def _set_estimate_context(estimate: Any, coordinate_frame: str, source: str) -> Any:
    estimate.frame = coordinate_frame
    estimate.source = source
    return estimate


def _estimate_payload(
    estimate: Any,
    coordinate_frame: str,
    source: str,
    mode: str,
    transform_source: str,
    temporal_status: str | None = None,
    temporal_accepted: bool | None = None,
    temporal_stale: bool | None = None,
    current_valid: bool | None = None,
    candidate_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = estimate.to_dict()
    payload["frame"] = coordinate_frame
    payload["coordinate_frame"] = coordinate_frame
    payload["source"] = source
    payload["mode"] = mode
    payload["T_base_cam_source"] = transform_source
    if "camera" in coordinate_frame or "local" in coordinate_frame:
        payload["center_cam"] = payload.get("center_base")
        payload["axis_cam"] = payload.get("axis_base")
        payload["local_debug_note"] = "center_base/axis_base contain camera-local coordinates in local_debug mode"
    if temporal_status is not None:
        payload["temporal_status"] = temporal_status
    if temporal_accepted is not None:
        payload["temporal_accepted"] = bool(temporal_accepted)
    if temporal_stale is not None:
        payload["temporal_stale"] = bool(temporal_stale)
        payload["stale_display"] = bool(temporal_stale)
    if current_valid is not None:
        payload["current_valid"] = bool(current_valid)
    if candidate_summaries is not None:
        payload["candidate_summaries"] = candidate_summaries
    return payload


def _draw_fallback_text(cv2: Any, image_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    y = 24
    for line in lines:
        cv2.putText(image_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return image_bgr


def _candidate_summaries_for_json(debug: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in debug.get("candidate_summaries", []) or []:
        summary: dict[str, Any] = {}
        for key, value in item.items():
            if key == "outer_contour_pixels":
                continue
            if isinstance(value, np.generic):
                summary[key] = value.item()
            else:
                summary[key] = value
        out.append(summary)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run online D435i valve ring estimation without ROS or control.")
    parser.add_argument("--mode", choices=("local_debug", "calibrated"), default="local_debug")
    parser.add_argument("--local-debug", action="store_true", help="Alias for --mode local_debug")
    parser.add_argument("--config", help="Estimator config JSON/YAML; may also contain T_base_cam")
    parser.add_argument("--T-base-cam-json", help="JSON with T_base_cam, T_base_camera, or rotation_matrix+translation_m")
    parser.add_argument("--output-dir", default="artifacts/valve_ring_perception/online_test")
    parser.add_argument("--record-dir", help="Directory for automatic/manual frame_XXXXXX.npz recordings")
    parser.add_argument("--record-every-n", type=int, default=0, help="Automatically save every N frames when --record-dir is set")
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--write-latest-result", action="store_true")
    parser.add_argument("--write-falcon-status", action="store_true")
    parser.add_argument("--falcon-status-path", default="/tmp/falcon_real_valve_vision_status.json")
    parser.add_argument("--window-name", default="valve_ring_realsense")
    args = parser.parse_args()
    if args.local_debug:
        args.mode = "local_debug"

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("run_realsense_online_estimation.py requires OpenCV for live display") from exc
    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError as exc:
        raise RuntimeError("run_realsense_online_estimation.py requires pyrealsense2") from exc

    try:
        config_data = _load_config_data(args.config)
        estimator_config = ValveRingEstimatorConfig.from_dict(config_data.get("estimator", config_data))
        estimator_config.depth_scale_m = 1.0
        has_config_transform = any(key in config_data for key in ("T_base_cam", "T_base_camera"))
        if args.mode == "local_debug" and args.T_base_cam_json is None and not has_config_transform:
            T_base_cam = np.eye(4, dtype=float)
            transform_source = "identity(local_debug)"
        else:
            T_base_cam, transform_source = _load_transform(args.T_base_cam_json, config_data)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    estimator = ValveRingEstimator(estimator_config)
    temporal_filter = TemporalEstimateFilter(estimator_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_dir = Path(args.record_dir) if args.record_dir else output_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    next_index = _next_frame_index(record_dir)
    coordinate_frame = "camera/local" if args.mode == "local_debug" else "base"
    estimate_source = f"valve_ring_perception_{args.mode}"

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.rgb8, args.fps)
    rs_config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    raw_depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
    print(f"RealSense started; mode={args.mode}; frame={coordinate_frame}; raw_depth_scale_m={raw_depth_scale_m}")
    print(f"T_base_cam source: {transform_source}")
    print(f"record_dir: {record_dir}")
    print("keys: s save current frame, r reset latch, l latch current estimate, q quit")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    last_rgb: np.ndarray | None = None
    last_depth_m: np.ndarray | None = None
    last_K: np.ndarray | None = None
    last_timestamp: float | None = None
    last_raw_estimate: Any | None = None
    last_raw_segmentation: Any | None = None
    last_raw_debug: dict[str, Any] | None = None
    frame_counter = 0

    try:
        for _ in range(max(0, int(args.warmup_frames))):
            align.process(pipeline.wait_for_frames())

        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            timestamp = time.time()
            rgb = np.asanyarray(color_frame.get_data()).copy()
            depth_raw = np.asanyarray(depth_frame.get_data()).copy()
            depth_m = depth_raw.astype(np.float32) * raw_depth_scale_m
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=float)
            camera = CameraIntrinsics.from_matrix(K, width=int(intr.width), height=int(intr.height))

            try:
                result = estimator.estimate_with_debug(rgb, depth_m, camera, T_base_cam, image_timestamp=timestamp)
                _set_estimate_context(result.estimate, coordinate_frame, estimate_source)
                filtered = temporal_filter.update(result.estimate, result.segmentation, result.debug, now=timestamp)
                estimate = _set_estimate_context(filtered.estimate, coordinate_frame, estimate_source)
                overlay_debug = dict(filtered.debug)
                overlay_debug["coordinate_frame"] = coordinate_frame
                overlay_debug["mode"] = args.mode
                if filtered.segmentation is not None:
                    overlay_rgb = render_debug_overlay(rgb, filtered.segmentation, estimate, overlay_debug, camera, T_base_cam)
                    display = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                else:
                    reasons = ",".join(estimate.quality.failure_reasons)
                    display = _draw_fallback_text(
                        cv2,
                        rgb,
                        [f"mode {args.mode}", f"frame {coordinate_frame}", f"valid {int(estimate.valid)}", filtered.status, reasons[:80]],
                    )
                if args.write_latest_result or args.write_falcon_status:
                    payload = _estimate_payload(
                        estimate,
                        coordinate_frame,
                        estimate_source,
                        args.mode,
                        transform_source,
                        temporal_status=filtered.status,
                        temporal_accepted=bool(filtered.accepted),
                        temporal_stale=bool(filtered.stale),
                        current_valid=bool(result.estimate.valid),
                        candidate_summaries=_candidate_summaries_for_json(overlay_debug),
                    )
                if args.write_latest_result:
                    atomic_write_json(str(output_dir / "latest_result.json"), payload)
                if args.write_falcon_status:
                    atomic_write_json(args.falcon_status_path, payload)
                last_raw_estimate = result.estimate
                last_raw_segmentation = result.segmentation
                last_raw_debug = dict(result.debug)
            except Exception as exc:
                display = _draw_fallback_text(cv2, rgb, [f"estimate failed: {exc}"])

            cv2.imshow(args.window_name, display)
            last_rgb = rgb
            last_depth_m = depth_m
            last_K = K
            last_timestamp = timestamp
            frame_counter += 1

            if args.record_dir and args.record_every_n > 0 and frame_counter % int(args.record_every_n) == 0:
                path = _save_frame(
                    record_dir,
                    next_index,
                    rgb,
                    depth_m,
                    K,
                    T_base_cam,
                    timestamp,
                    raw_depth_scale_m,
                    coordinate_frame,
                    estimate_source,
                )
                print(f"recorded {path}")
                next_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                temporal_filter.reset()
                print("temporal latch reset")
                continue
            if key == ord("l"):
                if last_raw_estimate is None:
                    print("no estimate available to latch yet")
                    continue
                latched = temporal_filter.latch_current(
                    last_raw_estimate,
                    last_raw_segmentation,
                    last_raw_debug or {},
                    now=last_timestamp,
                )
                print("latched current estimate" if latched else "current estimate is invalid; latch skipped")
                continue
            if key == ord("s"):
                if last_rgb is None or last_depth_m is None or last_K is None or last_timestamp is None:
                    print("no frame available to save yet")
                    continue
                path = _save_frame(
                    record_dir,
                    next_index,
                    last_rgb,
                    last_depth_m,
                    last_K,
                    T_base_cam,
                    last_timestamp,
                    raw_depth_scale_m,
                    coordinate_frame,
                    estimate_source,
                )
                print(f"saved {path}")
                next_index += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
