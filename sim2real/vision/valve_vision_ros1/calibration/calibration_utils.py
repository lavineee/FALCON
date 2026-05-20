#!/usr/bin/env python3
"""Shared utilities for ArUco based T_base_camera calibration.

Transform naming follows the convention T_parent_child:
    p_parent = R_parent_child @ p_child + t_parent_child
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def as_float_array(value: Any, shape: Optional[Tuple[int, ...]] = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if shape is not None:
        try:
            arr = arr.reshape(shape)
        except ValueError as exc:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def rpy_deg_to_rot(rpy_deg: Sequence[float]) -> np.ndarray:
    """Roll/pitch/yaw degrees using R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    roll, pitch, yaw = np.deg2rad(as_float_array(rpy_deg, (3,), "rotation_rpy_deg"))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def rot_to_rpy_deg(rotation: np.ndarray) -> List[float]:
    """Inverse of rpy_deg_to_rot for non-pathological rotations."""

    rotation = orthonormalize_rotation(rotation)
    sy = float(-rotation[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
    return [float(v) for v in np.rad2deg([roll, pitch, yaw])]


def rodrigues_to_rot(rvec: Sequence[float]) -> np.ndarray:
    rvec_arr = as_float_array(rvec, (3,), "rvec")
    theta = float(np.linalg.norm(rvec_arr))
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    axis = rvec_arr / theta
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3, dtype=float) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def orthonormalize_rotation(rotation: Any) -> np.ndarray:
    rot = as_float_array(rotation, (3, 3), "rotation_matrix")
    u, _, vt = np.linalg.svd(rot)
    fixed = u @ vt
    if np.linalg.det(fixed) < 0.0:
        u[:, -1] *= -1.0
        fixed = u @ vt
    return fixed


def make_transform(rotation: Any, translation: Any) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = orthonormalize_rotation(rotation)
    transform[:3, 3] = as_float_array(translation, (3,), "translation_m")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = as_float_array(transform, (4, 4), "transform")
    inv = np.eye(4, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def compose_transforms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return as_float_array(left, (4, 4), "left_transform") @ as_float_array(
        right,
        (4, 4),
        "right_transform",
    )


def parse_transform(transform_map: Mapping[str, Any], name: str) -> np.ndarray:
    if not isinstance(transform_map, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if "translation_m" in transform_map:
        translation = transform_map["translation_m"]
    elif "tvec" in transform_map:
        translation = transform_map["tvec"]
    else:
        raise ValueError(f"{name} is missing translation_m or tvec")

    if "rotation_matrix" in transform_map:
        rotation = orthonormalize_rotation(transform_map["rotation_matrix"])
    elif "rotation_rpy_deg" in transform_map:
        rotation = rpy_deg_to_rot(transform_map["rotation_rpy_deg"])
    elif "rvec" in transform_map:
        rotation = rodrigues_to_rot(transform_map["rvec"])
    else:
        raise ValueError(f"{name} is missing rotation_matrix, rotation_rpy_deg, or rvec")
    return make_transform(rotation, translation)


def transform_to_mapping(transform: np.ndarray) -> Dict[str, Any]:
    transform = as_float_array(transform, (4, 4), "transform")
    rotation = orthonormalize_rotation(transform[:3, :3])
    translation = transform[:3, 3]
    return {
        "translation_m": to_builtin_list(translation),
        "rotation_matrix": to_builtin_list(rotation),
        "rotation_rpy_deg": to_builtin_list(rot_to_rpy_deg(rotation)),
    }


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = orthonormalize_rotation(rotation_a) @ orthonormalize_rotation(rotation_b).T
    value = (float(np.trace(delta)) - 1.0) * 0.5
    value = max(-1.0, min(1.0, value))
    return float(math.degrees(math.acos(value)))


def rot_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    rotation = orthonormalize_rotation(rotation)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diag = np.diag(rotation)
        axis = int(np.argmax(diag))
        if axis == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def quat_wxyz_to_rot(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = as_float_array(quat, (4,), "quaternion_wxyz")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def average_rotation_matrices(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("at least one rotation is required")
    mats = np.asarray([orthonormalize_rotation(rot) for rot in rotations], dtype=float)
    if len(mats) == 1:
        return mats[0]

    try:
        from scipy.spatial.transform import Rotation as SciPyRotation  # type: ignore

        return SciPyRotation.from_matrix(mats).mean().as_matrix()
    except Exception:
        accum = np.zeros((4, 4), dtype=float)
        for mat in mats:
            quat = rot_to_quat_wxyz(mat)
            accum += np.outer(quat, quat)
        eigvals, eigvecs = np.linalg.eigh(accum)
        quat = eigvecs[:, int(np.argmax(eigvals))]
        if quat[0] < 0.0:
            quat *= -1.0
        return quat_wxyz_to_rot(quat)


def aggregate_translations(translations: Sequence[np.ndarray], method: str) -> np.ndarray:
    if not translations:
        raise ValueError("at least one translation is required")
    arr = np.asarray([as_float_array(t, (3,), "translation") for t in translations], dtype=float)
    if method == "median":
        return np.median(arr, axis=0)
    if method == "mean":
        return np.mean(arr, axis=0)
    raise ValueError(f"unknown translation aggregate: {method}")


def aggregate_base_camera(candidates: Sequence[np.ndarray], translation_aggregate: str) -> np.ndarray:
    if not candidates:
        raise ValueError("at least one T_base_camera candidate is required")
    rotations = [candidate[:3, :3] for candidate in candidates]
    translations = [candidate[:3, 3] for candidate in candidates]
    return make_transform(
        average_rotation_matrices(rotations),
        aggregate_translations(translations, translation_aggregate),
    )


def residual_for_pair(t_base_camera: np.ndarray, t_camera_tag: np.ndarray, t_base_tag: np.ndarray) -> Dict[str, Any]:
    pred = compose_transforms(t_base_camera, t_camera_tag)
    return {
        "translation_error_m": float(np.linalg.norm(pred[:3, 3] - t_base_tag[:3, 3])),
        "rotation_error_deg": rotation_error_deg(pred[:3, :3], t_base_tag[:3, :3]),
        "T_base_tag_pred": transform_to_mapping(pred),
    }


def summarize_residuals(residuals: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not residuals:
        return {
            "translation_mean_error_m": float("nan"),
            "translation_max_error_m": float("nan"),
            "rotation_mean_error_deg": float("nan"),
            "rotation_max_error_deg": float("nan"),
        }
    translation_errors = np.asarray([float(item["translation_error_m"]) for item in residuals], dtype=float)
    rotation_errors = np.asarray([float(item["rotation_error_deg"]) for item in residuals], dtype=float)
    return {
        "translation_mean_error_m": float(np.mean(translation_errors)),
        "translation_max_error_m": float(np.max(translation_errors)),
        "rotation_mean_error_deg": float(np.mean(rotation_errors)),
        "rotation_max_error_deg": float(np.max(rotation_errors)),
    }


def normalize_sample_records(samples: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample[{index}] must be a mapping")
        sample_id = sample.get("sample_id", index)
        transform_map = sample.get("T_camera_tag")
        if transform_map is None:
            raise ValueError(f"sample {sample_id} is missing T_camera_tag")
        records.append(
            {
                "sample_id": sample_id,
                "sample_index": index,
                "tag_id": sample.get("tag_id"),
                "stamp": sample.get("stamp"),
                "T_camera_tag": parse_transform(transform_map, f"sample {sample_id}.T_camera_tag"),
                "raw": sample,
            }
        )
    return records


def normalize_base_tag_records(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise ValueError("base tag pose file must contain a mapping")
    samples = config.get("samples")
    if not isinstance(samples, Sequence):
        raise ValueError("base tag pose file must contain a 'samples' list")
    records: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(samples):
        if not isinstance(item, Mapping):
            raise ValueError(f"base tag sample[{index}] must be a mapping")
        sample_id = item.get("sample_id", index)
        transform_map = item.get("T_base_tag")
        if transform_map is None:
            raise ValueError(f"base tag sample {sample_id} is missing T_base_tag")
        records[str(sample_id)] = {
            "sample_id": sample_id,
            "sample_index": index,
            "T_base_tag": parse_transform(transform_map, f"base sample {sample_id}.T_base_tag"),
            "raw": item,
        }
    return records


def solve_base_camera_from_records(
    sample_records: Sequence[Mapping[str, Any]],
    base_tag_records: Mapping[str, Mapping[str, Any]],
    translation_aggregate: str = "median",
    max_translation_error_m: Optional[float] = 0.05,
    max_rotation_error_deg: Optional[float] = 5.0,
    min_samples: int = 1,
) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    missing_base_ids: List[Any] = []
    for sample in sample_records:
        sample_id = sample["sample_id"]
        base_record = base_tag_records.get(str(sample_id))
        if base_record is None:
            missing_base_ids.append(sample_id)
            continue
        t_camera_tag = as_float_array(sample["T_camera_tag"], (4, 4), "T_camera_tag")
        t_base_tag = as_float_array(base_record["T_base_tag"], (4, 4), "T_base_tag")
        candidate = compose_transforms(t_base_tag, invert_transform(t_camera_tag))
        pairs.append(
            {
                "sample_id": sample_id,
                "sample_index": sample.get("sample_index"),
                "T_camera_tag": t_camera_tag,
                "T_base_tag": t_base_tag,
                "T_base_camera_candidate": candidate,
            }
        )

    if len(pairs) < min_samples:
        raise ValueError(
            f"not enough matched samples: matched={len(pairs)}, required={min_samples}, "
            f"missing_base_ids={missing_base_ids}"
        )

    initial = aggregate_base_camera(
        [pair["T_base_camera_candidate"] for pair in pairs],
        translation_aggregate,
    )
    initial_residuals = []
    for pair in pairs:
        residual = residual_for_pair(initial, pair["T_camera_tag"], pair["T_base_tag"])
        residual.update({"sample_id": pair["sample_id"], "sample_index": pair.get("sample_index")})
        initial_residuals.append(residual)

    inlier_ids = set()
    outlier_ids = []
    for residual in initial_residuals:
        ok_translation = (
            max_translation_error_m is None
            or float(residual["translation_error_m"]) <= float(max_translation_error_m)
        )
        ok_rotation = (
            max_rotation_error_deg is None
            or float(residual["rotation_error_deg"]) <= float(max_rotation_error_deg)
        )
        if ok_translation and ok_rotation:
            inlier_ids.add(str(residual["sample_id"]))
        else:
            outlier_ids.append(residual["sample_id"])

    if len(inlier_ids) < min_samples:
        raise ValueError(
            f"outlier rejection left too few samples: inliers={len(inlier_ids)}, "
            f"required={min_samples}, outliers={outlier_ids}"
        )

    inlier_pairs = [pair for pair in pairs if str(pair["sample_id"]) in inlier_ids]
    final = aggregate_base_camera(
        [pair["T_base_camera_candidate"] for pair in inlier_pairs],
        translation_aggregate,
    )

    all_final_residuals = []
    used_final_residuals = []
    for pair in pairs:
        residual = residual_for_pair(final, pair["T_camera_tag"], pair["T_base_tag"])
        used = str(pair["sample_id"]) in inlier_ids
        residual.update(
            {
                "sample_id": pair["sample_id"],
                "sample_index": pair.get("sample_index"),
                "used": bool(used),
            }
        )
        all_final_residuals.append(residual)
        if used:
            used_final_residuals.append(residual)

    summary = summarize_residuals(used_final_residuals)
    summary["num_samples"] = int(len(used_final_residuals))

    return {
        "T_base_camera": final,
        "num_samples_matched": len(pairs),
        "num_samples_used": len(inlier_pairs),
        "missing_base_ids": missing_base_ids,
        "outlier_sample_ids": outlier_ids,
        "initial_residual_samples": initial_residuals,
        "residual_samples": all_final_residuals,
        "summary": summary,
    }


def require_cv2_aruco():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "OpenCV is required for ArUco detection. Install python3-opencv with contrib "
            "modules or opencv-contrib-python in the ROS environment."
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is unavailable. Install OpenCV contrib modules "
            "(python3-opencv with aruco support or opencv-contrib-python)."
        )
    return cv2


def aruco_dictionary(dictionary_name: str):
    cv2 = require_cv2_aruco()
    name = str(dictionary_name)
    if not hasattr(cv2.aruco, name) and not name.startswith("DICT_"):
        name = f"DICT_{name}"
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def aruco_detector_parameters():
    cv2 = require_cv2_aruco()
    if hasattr(cv2.aruco, "DetectorParameters"):
        try:
            return cv2.aruco.DetectorParameters()
        except Exception:
            pass
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    raise RuntimeError("OpenCV ArUco detector parameters API is unavailable")


def detect_aruco_markers(gray_image: np.ndarray, dictionary_name: str = "DICT_4X4_50"):
    cv2 = require_cv2_aruco()
    dictionary = aruco_dictionary(dictionary_name)
    parameters = aruco_detector_parameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(gray_image)
    return cv2.aruco.detectMarkers(gray_image, dictionary, parameters=parameters)


def marker_object_points(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def select_marker(
    corners: Sequence[Any],
    ids: Optional[np.ndarray],
    tag_id: Optional[int],
) -> Tuple[int, int, np.ndarray]:
    if ids is None or len(ids) == 0:
        raise LookupError("tag not detected")
    ids_flat = np.asarray(ids, dtype=int).reshape(-1)
    if tag_id is None:
        index = 0
    else:
        matches = np.where(ids_flat == int(tag_id))[0]
        if len(matches) == 0:
            raise LookupError(f"tag id {tag_id} not detected; visible ids={ids_flat.tolist()}")
        index = int(matches[0])
    return int(index), int(ids_flat[index]), np.asarray(corners[index], dtype=float).reshape(4, 2)


def estimate_tag_pose(
    marker_corners_px: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cv2 = require_cv2_aruco()
    image_points = as_float_array(marker_corners_px, (4, 2), "marker_corners_px").astype(np.float32)
    camera_matrix = as_float_array(camera_matrix, (3, 3), "camera_matrix")
    dist_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(-1)

    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [image_points.reshape(1, 4, 2)],
            float(tag_size_m),
            camera_matrix,
            dist_coeffs,
        )
        return np.asarray(rvecs[0], dtype=float).reshape(3), np.asarray(tvecs[0], dtype=float).reshape(3)

    object_points = marker_object_points(tag_size_m)
    flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", getattr(cv2, "SOLVEPNP_ITERATIVE", 0))
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=flags)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed for detected tag")
    return np.asarray(rvec, dtype=float).reshape(3), np.asarray(tvec, dtype=float).reshape(3)


def cv2_rodrigues(rvec: Sequence[float]) -> np.ndarray:
    cv2 = require_cv2_aruco()
    rotation, _ = cv2.Rodrigues(as_float_array(rvec, (3,), "rvec"))
    return orthonormalize_rotation(rotation)


def reprojection_error_px(
    marker_corners_px: np.ndarray,
    tag_size_m: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    cv2 = require_cv2_aruco()
    object_points = marker_object_points(tag_size_m)
    projected, _ = cv2.projectPoints(
        object_points,
        as_float_array(rvec, (3,), "rvec"),
        as_float_array(tvec, (3,), "tvec"),
        as_float_array(camera_matrix, (3, 3), "camera_matrix"),
        np.asarray(dist_coeffs, dtype=float).reshape(-1),
    )
    projected = np.asarray(projected, dtype=float).reshape(4, 2)
    observed = as_float_array(marker_corners_px, (4, 2), "marker_corners_px")
    return float(np.mean(np.linalg.norm(projected - observed, axis=1)))


def depth_value_to_meters(value: Any, encoding: str, depth_scale_m: float) -> Optional[float]:
    depth_value = float(value)
    if not np.isfinite(depth_value) or depth_value <= 0.0:
        return None
    if str(encoding) in ("16UC1", "mono16"):
        return depth_value * float(depth_scale_m)
    return depth_value


def median_depth_near_pixel(
    depth_image: Optional[np.ndarray],
    depth_encoding: str,
    center_uv: Sequence[float],
    depth_scale_m: float = 0.001,
    window_px: int = 7,
) -> Tuple[Optional[float], int]:
    if depth_image is None:
        return None, 0
    image = np.asarray(depth_image)
    if image.ndim < 2:
        return None, 0
    u, v = as_float_array(center_uv, (2,), "center_uv")
    radius = max(0, int(window_px) // 2)
    cx = int(round(float(u)))
    cy = int(round(float(v)))
    y0 = max(0, cy - radius)
    y1 = min(image.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(image.shape[1], cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return None, 0
    patch = image[y0:y1, x0:x1]
    depths = []
    for value in np.asarray(patch).reshape(-1):
        depth_m = depth_value_to_meters(value, depth_encoding, depth_scale_m)
        if depth_m is not None:
            depths.append(depth_m)
    if not depths:
        return None, 0
    return float(np.median(np.asarray(depths, dtype=float))), int(len(depths))


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return value


def to_builtin_list(value: Any) -> List[Any]:
    return to_builtin(value)


def load_yaml_or_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".json", ".jsonl"):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"PyYAML is required to read YAML file {path}. Install python3-yaml/PyYAML "
            "or provide an equivalent JSON file."
        ) from exc
    return yaml.safe_load(text)


def load_sample_file(path: str) -> List[Mapping[str, Any]]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".jsonl":
        samples = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    samples.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        return samples

    loaded = load_yaml_or_json(path)
    if isinstance(loaded, Mapping) and "samples" in loaded:
        loaded = loaded["samples"]
    if not isinstance(loaded, list):
        raise ValueError(f"sample file {path} must contain a list or a mapping with samples")
    return loaded


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "null"
        return f"{value:.10g}"
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(to_builtin(value))


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float, np.integer, np.floating))


def _is_scalar_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_scalar(item) for item in value)


def _minimal_yaml_lines(value: Any, indent: int = 0) -> List[str]:
    value = to_builtin(value)
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: List[str] = []
        for key, item in value.items():
            if _is_scalar(item):
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
            elif _is_scalar_list(item):
                inner = ", ".join(_format_scalar(elem) for elem in item)
                lines.append(f"{prefix}{key}: [{inner}]")
            else:
                lines.append(f"{prefix}{key}:")
                lines.extend(_minimal_yaml_lines(item, indent + 2))
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{prefix}- {_format_scalar(item)}")
            elif _is_scalar_list(item):
                inner = ", ".join(_format_scalar(elem) for elem in item)
                lines.append(f"{prefix}- [{inner}]")
            else:
                lines.append(f"{prefix}-")
                lines.extend(_minimal_yaml_lines(item, indent + 2))
        return lines
    return [f"{prefix}{_format_scalar(value)}"]


def dump_yaml_file(data: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = to_builtin(data)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    except Exception:
        text = "\n".join(_minimal_yaml_lines(data)) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def append_sample(path: str, sample: Mapping[str, Any], existing_samples: List[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    suffix = os.path.splitext(path)[1].lower()
    sample_builtin = to_builtin(sample)
    if suffix == ".jsonl":
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample_builtin, sort_keys=True) + "\n")
        return

    existing_samples.append(sample_builtin)
    if suffix == ".json":
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"samples": existing_samples}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return

    dump_yaml_file({"samples": existing_samples}, path)


def load_extrinsic(path: str) -> np.ndarray:
    config = load_yaml_or_json(path)
    if not isinstance(config, Mapping):
        raise ValueError(f"extrinsic file {path} must contain a mapping")
    transform_map = config.get("T_base_camera")
    if transform_map is None:
        camera_extrinsic = config.get("camera_extrinsic")
        if isinstance(camera_extrinsic, Mapping):
            transform_map = camera_extrinsic.get("T_base_camera")
    if transform_map is None:
        raise ValueError(f"extrinsic file {path} is missing T_base_camera")
    return parse_transform(transform_map, "T_base_camera")


def output_mapping_for_solution(
    t_base_camera: np.ndarray,
    num_samples_used: int,
    residual_summary: Mapping[str, Any],
    residual_samples: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    transform = transform_to_mapping(t_base_camera)
    output: Dict[str, Any] = {
        "T_base_camera": {
            "parent_frame": "base",
            "child_frame": "camera_color_optical_frame",
            "translation_m": transform["translation_m"],
            "rotation_matrix": transform["rotation_matrix"],
            "rotation_rpy_deg": transform["rotation_rpy_deg"],
            "source": "aruco_known_base_tag",
            "num_samples_used": int(num_samples_used),
            "residual": dict(residual_summary),
        },
        "camera_extrinsic": {
            "T_base_camera": {
                "translation_m": transform["translation_m"],
                "rotation_matrix": transform["rotation_matrix"],
            }
        },
    }
    if residual_samples is not None:
        output["residual_samples"] = [
            {
                "sample_id": item.get("sample_id"),
                "sample_index": item.get("sample_index"),
                "used": bool(item.get("used", False)),
                "translation_error_m": item.get("translation_error_m"),
                "rotation_error_deg": item.get("rotation_error_deg"),
            }
            for item in residual_samples
        ]
    return output


def format_transform_block(name: str, transform: np.ndarray) -> str:
    mapping = transform_to_mapping(transform)
    lines = [f"{name}:"]
    lines.append(f"  translation_m: {json.dumps(mapping['translation_m'])}")
    lines.append("  rotation_matrix:")
    for row in mapping["rotation_matrix"]:
        lines.append(f"    - {json.dumps(row)}")
    lines.append(f"  rotation_rpy_deg: {json.dumps(mapping['rotation_rpy_deg'])}")
    return "\n".join(lines)
