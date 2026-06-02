#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.vision.valve_ring_perception import CameraIntrinsics, ValveRingEstimator, ValveRingEstimatorConfig


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_array(path: str, key: str | None = None) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if key is None:
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(f"{path} has keys {keys}; pass an explicit key")
            key = keys[0]
        return data[key]
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"Reading image files requires OpenCV; use .npy/.npz instead for {path}") from exc
    flags = cv2.IMREAD_UNCHANGED
    arr = cv2.imread(path, flags)
    if arr is None:
        raise RuntimeError(f"failed to read {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def _load_intrinsics(path: str | None, sample: Any, args: argparse.Namespace) -> CameraIntrinsics:
    if path is not None:
        return CameraIntrinsics.from_mapping(_load_json(path))
    if sample is not None:
        keys = set(sample.keys())
        if "K" in keys:
            return CameraIntrinsics.from_matrix(sample["K"], width=sample.get("width"), height=sample.get("height"))
        if {"fx", "fy", "cx", "cy"}.issubset(keys):
            return CameraIntrinsics(float(sample["fx"]), float(sample["fy"]), float(sample["cx"]), float(sample["cy"]))
    if None not in (args.fx, args.fy, args.cx, args.cy):
        return CameraIntrinsics(args.fx, args.fy, args.cx, args.cy)
    raise ValueError("intrinsics required: pass --intrinsics-json, sample K/fx/fy/cx/cy, or --fx --fy --cx --cy")


def _load_transform(path: str | None, sample: Any) -> np.ndarray:
    if path is not None:
        if path.endswith(".npy"):
            return np.load(path).reshape(4, 4)
        data = _load_json(path)
    elif sample is not None and "T_base_cam" in sample:
        return np.asarray(sample["T_base_cam"], dtype=float).reshape(4, 4)
    else:
        raise ValueError("T_base_cam required: pass --T-base-cam-json or include T_base_cam in --sample")

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline valve ring RGB-D estimation.")
    parser.add_argument("--sample", help="NPZ with rgb, depth, optional K and T_base_cam")
    parser.add_argument("--rgb", help="RGB image path (.npy/.npz, or image with OpenCV installed)")
    parser.add_argument("--depth", help="Depth image path (.npy/.npz, or image with OpenCV installed)")
    parser.add_argument("--rgb-key", default="rgb")
    parser.add_argument("--depth-key", default="depth")
    parser.add_argument("--intrinsics-json")
    parser.add_argument("--T-base-cam-json")
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--config", help="Estimator config JSON/YAML")
    parser.add_argument("--depth-scale-m", type=float, help="Override config depth scale")
    parser.add_argument("--output-dir", default="artifacts/valve_ring_perception/latest")
    args = parser.parse_args()

    sample = None
    if args.sample:
        sample = np.load(args.sample)
        rgb = sample[args.rgb_key]
        depth = sample[args.depth_key]
    else:
        if not args.rgb or not args.depth:
            parser.error("pass --sample or both --rgb and --depth")
        rgb = _load_array(args.rgb, args.rgb_key if args.rgb.endswith(".npz") else None)
        depth = _load_array(args.depth, args.depth_key if args.depth.endswith(".npz") else None)

    config = ValveRingEstimatorConfig.from_file(args.config) if args.config else ValveRingEstimatorConfig()
    if args.depth_scale_m is not None:
        config.depth_scale_m = float(args.depth_scale_m)
    elif sample is not None and "depth_scale_m" in sample:
        config.depth_scale_m = float(np.asarray(sample["depth_scale_m"]).reshape(-1)[0])
    intrinsics = _load_intrinsics(args.intrinsics_json, sample, args)
    T_base_cam = _load_transform(args.T_base_cam_json, sample)

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "result.json")
    overlay_path = os.path.join(args.output_dir, "debug_overlay.png")
    estimator = ValveRingEstimator(config)
    estimate = estimator.estimate_and_write(rgb, depth, intrinsics, T_base_cam, result_path, overlay_path)
    print(json.dumps(estimate.to_dict(), indent=2, sort_keys=True))
    print(f"result_json: {result_path}")
    print(f"debug_overlay: {overlay_path}")
    return 0 if estimate.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
