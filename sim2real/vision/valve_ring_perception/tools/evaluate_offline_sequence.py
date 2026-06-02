#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.vision.valve_ring_perception import CameraIntrinsics, ValveRingEstimator, ValveRingEstimatorConfig
from sim2real.vision.valve_ring_perception.plane_fit import normalize
from sim2real.vision.valve_ring_perception.status_writer import atomic_write_json
from sim2real.vision.valve_ring_perception.visualization import draw_debug_overlay


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _scalar(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _sample_has(sample: Any, key: str) -> bool:
    return key in set(sample.keys())


def _sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if not _sample_has(sample, key):
        return default
    value = sample[key]
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def _load_intrinsics(sample: Any, fallback_path: str | None) -> CameraIntrinsics:
    keys = set(sample.keys())
    if "K" in keys:
        return CameraIntrinsics.from_matrix(sample["K"], width=_sample_get(sample, "width"), height=_sample_get(sample, "height"))
    if {"fx", "fy", "cx", "cy"}.issubset(keys):
        return CameraIntrinsics(float(sample["fx"]), float(sample["fy"]), float(sample["cx"]), float(sample["cy"]))
    if fallback_path:
        return CameraIntrinsics.from_mapping(_load_json(fallback_path))
    raise ValueError("sample has no K/fx/fy/cx/cy and no --intrinsics-json was provided")


def _load_transform(sample: Any, fallback_path: str | None, allow_identity: bool) -> np.ndarray:
    if "T_base_cam" in sample:
        return np.asarray(sample["T_base_cam"], dtype=float).reshape(4, 4)
    if allow_identity:
        return np.eye(4, dtype=float)
    if not fallback_path:
        raise ValueError("sample has no T_base_cam and no --T-base-cam-json was provided")
    data = _load_json(fallback_path)
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


def _load_depth(sample: Any, requested_key: str | None) -> tuple[np.ndarray, str]:
    if requested_key:
        return sample[requested_key], requested_key
    for key in ("depth_m", "depth"):
        if _sample_has(sample, key):
            return sample[key], key
    raise ValueError("sample has no depth_m or depth array")


def _load_frame_label(sample: Any, requested_label: str | None) -> str:
    if requested_label:
        return requested_label
    if _sample_has(sample, "frame"):
        return str(_sample_get(sample, "frame"))
    return "base"


def _payload_for_estimate(estimate: Any, coordinate_frame: str, source: str) -> dict[str, Any]:
    payload = estimate.to_dict()
    payload["frame"] = coordinate_frame
    payload["coordinate_frame"] = coordinate_frame
    payload["source"] = source
    if "camera" in coordinate_frame or "local" in coordinate_frame:
        payload["center_cam"] = payload.get("center_base")
        payload["axis_cam"] = payload.get("axis_base")
        payload["local_debug_note"] = "center_base/axis_base contain camera-local coordinates in local_debug mode"
    return payload


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _estimate_frame(
    frame_path: Path,
    output_dir: Path,
    base_config: ValveRingEstimatorConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_output = output_dir / frame_path.stem
    frame_output.mkdir(parents=True, exist_ok=True)
    result_path = frame_output / "result.json"
    overlay_path = frame_output / "debug_overlay.png"

    try:
        sample = np.load(frame_path)
        rgb = sample[args.rgb_key]
        depth, depth_key = _load_depth(sample, args.depth_key)
        intrinsics = _load_intrinsics(sample, args.intrinsics_json)
        T_base_cam = _load_transform(sample, args.T_base_cam_json, args.allow_identity_T_base_cam)
        coordinate_frame = _load_frame_label(sample, args.frame_label)
        source = args.source or str(_sample_get(sample, "source", "valve_ring_perception_offline"))
        config = copy.deepcopy(base_config)
        sample_depth_scale = _scalar(sample["depth_scale_m"], None) if "depth_scale_m" in sample else None
        if args.depth_scale_m is not None:
            config.depth_scale_m = float(args.depth_scale_m)
        elif sample_depth_scale is not None:
            config.depth_scale_m = float(sample_depth_scale)
        timestamp = _scalar(sample["timestamp"], None) if "timestamp" in sample else None

        estimator = ValveRingEstimator(config)
        result = estimator.estimate_with_debug(
            rgb,
            depth,
            intrinsics,
            T_base_cam,
            image_timestamp=timestamp,
        )
        estimate = result.estimate
        estimate.frame = coordinate_frame
        estimate.source = source
        debug = dict(result.debug)
        debug["coordinate_frame"] = coordinate_frame
        debug["mode"] = "offline_sequence"
        atomic_write_json(str(result_path), _payload_for_estimate(estimate, coordinate_frame, source))
        if result.segmentation is not None:
            draw_debug_overlay(rgb, result.segmentation, estimate, debug, intrinsics, T_base_cam, str(overlay_path))
        quality = estimate.quality.to_dict()
        return {
            "frame": frame_path.name,
            "frame_path": str(frame_path),
            "output_dir": str(frame_output),
            "coordinate_frame": coordinate_frame,
            "depth_key": depth_key,
            "valid": bool(estimate.valid),
            "failure_reasons": ";".join(quality["failure_reasons"]),
            "center_x": None if estimate.center_base is None else float(estimate.center_base[0]),
            "center_y": None if estimate.center_base is None else float(estimate.center_base[1]),
            "center_z": None if estimate.center_base is None else float(estimate.center_base[2]),
            "radius_m": estimate.radius_m,
            "axis_x": None if estimate.axis_base is None else float(estimate.axis_base[0]),
            "axis_y": None if estimate.axis_base is None else float(estimate.axis_base[1]),
            "axis_z": None if estimate.axis_base is None else float(estimate.axis_base[2]),
            "plane_rmse_m": quality["plane_rmse_m"],
            "circle_rmse_m": quality["circle_rmse_m"],
            "plane_inlier_ratio": quality["plane_inlier_ratio"],
            "circle_inlier_ratio": quality["circle_inlier_ratio"],
            "num_mask_points": quality["num_mask_points"],
            "num_outer_points": quality["num_outer_points"],
            "result_json": str(result_path),
            "debug_overlay": str(overlay_path),
        }
    except Exception as exc:
        payload = {
            "valid": False,
            "frame": frame_path.name,
            "error": str(exc),
            "quality": {"valid": False, "failure_reasons": ["exception"]},
        }
        atomic_write_json(str(result_path), payload)
        return {
            "frame": frame_path.name,
            "frame_path": str(frame_path),
            "output_dir": str(frame_output),
            "coordinate_frame": "",
            "depth_key": "",
            "valid": False,
            "failure_reasons": f"exception:{exc}",
            "center_x": None,
            "center_y": None,
            "center_z": None,
            "radius_m": None,
            "axis_x": None,
            "axis_y": None,
            "axis_z": None,
            "plane_rmse_m": None,
            "circle_rmse_m": None,
            "plane_inlier_ratio": None,
            "circle_inlier_ratio": None,
            "num_mask_points": None,
            "num_outer_points": None,
            "result_json": str(result_path),
            "debug_overlay": "",
        }


def _axis_stats(axes: np.ndarray) -> dict[str, Any]:
    if axes.shape[0] == 0:
        return {"axis_mean_base": None, "axis_angular_mean_deg": None, "axis_angular_std_deg": None}
    aligned = axes.copy()
    ref = aligned[0]
    for i in range(aligned.shape[0]):
        if float(np.dot(aligned[i], ref)) < 0.0:
            aligned[i] = -aligned[i]
    mean_axis = normalize(np.mean(aligned, axis=0))
    if mean_axis is None:
        return {"axis_mean_base": None, "axis_angular_mean_deg": None, "axis_angular_std_deg": None}
    dots = np.clip(aligned @ mean_axis, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    return {
        "axis_mean_base": [float(v) for v in mean_axis],
        "axis_angular_mean_deg": float(np.mean(angles_deg)),
        "axis_angular_std_deg": float(np.std(angles_deg)),
    }


def _summary(rows: list[dict[str, Any]], data_dir: Path, output_dir: Path, pattern: str) -> dict[str, Any]:
    total = len(rows)
    valid_rows = [row for row in rows if row["valid"]]
    centers = np.array([[row["center_x"], row["center_y"], row["center_z"]] for row in valid_rows], dtype=float)
    radii = np.array([row["radius_m"] for row in valid_rows], dtype=float)
    axes = np.array([[row["axis_x"], row["axis_y"], row["axis_z"]] for row in valid_rows], dtype=float)
    plane_rmse = np.array([row["plane_rmse_m"] for row in valid_rows if row["plane_rmse_m"] is not None], dtype=float)
    circle_rmse = np.array([row["circle_rmse_m"] for row in valid_rows if row["circle_rmse_m"] is not None], dtype=float)

    reason_counts: Counter[str] = Counter()
    for row in rows:
        if row["valid"]:
            continue
        for reason in str(row["failure_reasons"]).split(";"):
            if reason:
                reason_counts[reason] += 1

    summary: dict[str, Any] = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "pattern": pattern,
        "total_frames": int(total),
        "valid_frames": int(len(valid_rows)),
        "failed_frames": int(total - len(valid_rows)),
        "success_rate": float(len(valid_rows) / total) if total else 0.0,
        "failure_reason_counts": dict(sorted(reason_counts.items())),
    }
    if valid_rows:
        center_std = np.std(centers, axis=0)
        axis_summary = _axis_stats(axes)
        summary.update(
            {
                "center_mean_base": [float(v) for v in np.mean(centers, axis=0)],
                "center_std_base": [float(v) for v in center_std],
                "center_std_norm_m": float(np.linalg.norm(center_std)),
                "radius_mean_m": float(np.mean(radii)),
                "radius_std_m": float(np.std(radii)),
                "center_mean": [float(v) for v in np.mean(centers, axis=0)],
                "center_std": [float(v) for v in center_std],
                "radius_mean": float(np.mean(radii)),
                "radius_std": float(np.std(radii)),
                "plane_rmse_mean_m": float(np.mean(plane_rmse)) if plane_rmse.size else None,
                "plane_rmse_std_m": float(np.std(plane_rmse)) if plane_rmse.size else None,
                "circle_rmse_mean_m": float(np.mean(circle_rmse)) if circle_rmse.size else None,
                "circle_rmse_std_m": float(np.std(circle_rmse)) if circle_rmse.size else None,
            }
        )
        summary.update(axis_summary)
        summary["axis_mean"] = axis_summary["axis_mean_base"]
        summary["axis_std"] = axis_summary["axis_angular_std_deg"]
    else:
        summary.update(
            {
                "center_mean_base": None,
                "center_std_base": None,
                "center_std_norm_m": None,
                "radius_mean_m": None,
                "radius_std_m": None,
                "center_mean": None,
                "center_std": None,
                "radius_mean": None,
                "radius_std": None,
                "axis_mean_base": None,
                "axis_angular_mean_deg": None,
                "axis_angular_std_deg": None,
                "axis_mean": None,
                "axis_std": None,
                "plane_rmse_mean_m": None,
                "plane_rmse_std_m": None,
                "circle_rmse_mean_m": None,
                "circle_rmse_std_m": None,
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate valve ring perception on frame_*.npz sequences.")
    parser.add_argument("data_dir", help="Directory containing frame_*.npz files")
    parser.add_argument("--pattern", default="frame_*.npz")
    parser.add_argument("--output-dir", help="Directory for per-frame outputs and summary")
    parser.add_argument("--config", help="Estimator config JSON/YAML")
    parser.add_argument("--depth-scale-m", type=float, help="Override all sample depth_scale_m values")
    parser.add_argument("--rgb-key", default="rgb")
    parser.add_argument("--depth-key", help="Depth array key. Defaults to auto: depth_m, then depth.")
    parser.add_argument("--intrinsics-json", help="Fallback intrinsics JSON if a frame lacks K")
    parser.add_argument("--T-base-cam-json", help="Fallback T_base_cam JSON if a frame lacks T_base_cam")
    parser.add_argument("--allow-identity-T-base-cam", action="store_true")
    parser.add_argument("--frame-label", help="Override output frame label, e.g. camera/local")
    parser.add_argument("--source", help="Override result source")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "valve_ring_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(data_dir.glob(args.pattern))
    if not frames:
        raise SystemExit(f"no frames matched {data_dir / args.pattern}")

    base_config = ValveRingEstimatorConfig.from_file(args.config) if args.config else ValveRingEstimatorConfig()
    rows = [_estimate_frame(frame_path, output_dir, base_config, args) for frame_path in frames]

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "frame",
        "coordinate_frame",
        "depth_key",
        "valid",
        "failure_reasons",
        "center_x",
        "center_y",
        "center_z",
        "radius_m",
        "axis_x",
        "axis_y",
        "axis_z",
        "plane_rmse_m",
        "circle_rmse_m",
        "plane_inlier_ratio",
        "circle_inlier_ratio",
        "num_mask_points",
        "num_outer_points",
        "result_json",
        "debug_overlay",
    ]
    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in fieldnames})

    summary = _summary(rows, data_dir, output_dir, args.pattern)
    summary_path = output_dir / "summary.json"
    atomic_write_json(str(summary_path), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"metrics_csv: {metrics_path}")
    print(f"summary_json: {summary_path}")
    return 0 if summary["valid_frames"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
