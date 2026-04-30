import json
import math
import os
import time

import cv2
import numpy as np


def atomic_write_json(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def wrap_to_pi(angle):
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _normalize_or_none(vec, eps=1e-9):
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < eps:
        return None
    return arr / norm


def _as_hsv_ranges(hsv_range):
    if hsv_range is None:
        return []
    arr = np.asarray(hsv_range, dtype=int)
    if arr.shape == (2, 3):
        return [(arr[0], arr[1])]
    if arr.ndim == 2 and arr.shape[1] == 6:
        return [(row[:3], row[3:]) for row in arr]
    if arr.ndim == 3 and arr.shape[1:] == (2, 3):
        return [(item[0], item[1]) for item in arr]
    if arr.size == 6:
        flat = arr.reshape(6)
        return [(flat[:3], flat[3:])]
    raise ValueError(f"Unsupported HSV range shape: {arr.shape}")


def segment_hsv(rgb, hsv_range):
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in _as_hsv_ranges(hsv_range):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8)))
    return mask


def clean_mask(mask, open_iters=1, close_iters=1, kernel_size=3):
    mask = np.asarray(mask, dtype=np.uint8)
    if kernel_size <= 1:
        return mask
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    if open_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))
    if close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))
    return mask


def extract_largest_component(mask, min_area_px=1):
    mask = np.asarray(mask, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return np.zeros_like(mask), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    area = int(stats[best, cv2.CC_STAT_AREA])
    if area < int(min_area_px):
        return np.zeros_like(mask), area
    return (labels == best).astype(np.uint8) * 255, area


def backproject_mask_to_points(
    mask,
    depth,
    K,
    depth_min=0.02,
    depth_max=5.0,
    max_points=6000,
    depth_cluster_tolerance_m=0.08,
):
    mask = np.asarray(mask) > 0
    depth = np.asarray(depth, dtype=float)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    valid = mask & np.isfinite(depth) & (depth > float(depth_min)) & (depth < float(depth_max))
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 2), dtype=float)
    if max_points is not None and xs.size > int(max_points):
        indices = np.linspace(0, xs.size - 1, int(max_points)).astype(int)
        xs = xs[indices]
        ys = ys[indices]
    z_forward = depth[ys, xs]
    if depth_cluster_tolerance_m is not None and z_forward.size >= 8:
        depth_median = float(np.median(z_forward))
        in_cluster = np.abs(z_forward - depth_median) <= float(depth_cluster_tolerance_m)
        xs = xs[in_cluster]
        ys = ys[in_cluster]
        z_forward = z_forward[in_cluster]
        if xs.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 2), dtype=float)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x_local = (xs.astype(float) - cx) / fx * z_forward
    y_local = -(ys.astype(float) - cy) / fy * z_forward
    z_local = -z_forward
    points_cam = np.column_stack((x_local, y_local, z_local))
    pixels = np.column_stack((xs.astype(float), ys.astype(float)))
    return points_cam, pixels


def backproject_pixel_to_point(pixel, depth_value, K):
    u, v = np.asarray(pixel, dtype=float).reshape(2)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    z_forward = float(depth_value)
    return np.array(
        [
            (u - K[0, 2]) / K[0, 0] * z_forward,
            -(v - K[1, 2]) / K[1, 1] * z_forward,
            -z_forward,
        ],
        dtype=float,
    )


def transform_points(T_base_cam, points_cam):
    points_cam = np.asarray(points_cam, dtype=float)
    if points_cam.size == 0:
        return np.zeros((0, 3), dtype=float)
    T_base_cam = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
    return points_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]


def robust_point_center(points, trim_fraction=0.2):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if points.shape[0] == 0:
        return None
    center = np.median(points, axis=0)
    if points.shape[0] < 8 or trim_fraction <= 0.0:
        return center
    distances = np.linalg.norm(points - center, axis=1)
    keep_count = max(4, int(math.ceil(points.shape[0] * (1.0 - float(trim_fraction)))))
    keep_indices = np.argsort(distances)[:keep_count]
    return np.mean(points[keep_indices], axis=0)


def fit_plane_svd(points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if points.shape[0] < 3:
        return None, None, float("nan")
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = _normalize_or_none(vt[-1])
    if normal is None:
        return None, centroid, float("nan")
    residuals = centered @ normal
    return normal, centroid, float(np.sqrt(np.mean(residuals * residuals)))


def orthonormalize_rotation(R):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    u, _, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def estimate_spoke_direction(points_base, center_base, axis_base):
    points_base = np.asarray(points_base, dtype=float).reshape(-1, 3)
    center_base = np.asarray(center_base, dtype=float).reshape(3)
    axis_base = np.asarray(axis_base, dtype=float).reshape(3)
    if points_base.shape[0] == 0:
        return None
    rel = points_base - center_base
    rel = rel - np.outer(rel @ axis_base, axis_base)
    direction = np.mean(rel, axis=0)
    direction = direction - np.dot(direction, axis_base) * axis_base
    direction = _normalize_or_none(direction)
    if direction is not None:
        return direction
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    direction = vt[0]
    if np.dot(np.mean(rel, axis=0), direction) < 0.0:
        direction = -direction
    return _normalize_or_none(direction)


def construct_wheel_R_base(axis_base, spoke_dir_base):
    x_axis = _normalize_or_none(axis_base)
    if x_axis is None:
        return None
    y_axis = np.asarray(spoke_dir_base, dtype=float).reshape(3)
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = _normalize_or_none(y_axis)
    if y_axis is None:
        return None
    z_axis = _normalize_or_none(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    y_axis = _normalize_or_none(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None
    return orthonormalize_rotation(np.column_stack((x_axis, y_axis, z_axis)))


def construct_grasp_R_base(
    axis_base,
    center_base,
    grasp_base,
    forward_axis_sign=-1.0,
    radial_axis="y",
    radial_axis_sign=-1.0,
):
    x_axis = _normalize_or_none(float(forward_axis_sign) * np.asarray(axis_base, dtype=float).reshape(3))
    if x_axis is None:
        return None
    radial = np.asarray(grasp_base, dtype=float).reshape(3) - np.asarray(center_base, dtype=float).reshape(3)
    radial = radial - np.dot(radial, x_axis) * x_axis
    radial = _normalize_or_none(float(radial_axis_sign) * radial)
    if radial is None:
        return None
    if str(radial_axis).lower() == "z":
        z_axis = radial
        y_axis = _normalize_or_none(np.cross(z_axis, x_axis))
        if y_axis is None:
            return None
        z_axis = _normalize_or_none(np.cross(x_axis, y_axis))
    else:
        y_axis = radial
        z_axis = _normalize_or_none(np.cross(x_axis, y_axis))
        if z_axis is None:
            return None
        y_axis = _normalize_or_none(np.cross(z_axis, x_axis))
    if y_axis is None or z_axis is None:
        return None
    return orthonormalize_rotation(np.column_stack((x_axis, y_axis, z_axis)))


def _plane_basis(axis_base):
    axis = _normalize_or_none(axis_base)
    if axis is None:
        return None, None
    ref = np.array([0.0, 1.0, 0.0], dtype=float)
    ref = ref - np.dot(ref, axis) * axis
    ref = _normalize_or_none(ref)
    if ref is None:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        ref = _normalize_or_none(ref - np.dot(ref, axis) * axis)
    if ref is None:
        return None, None
    tangent = _normalize_or_none(np.cross(axis, ref))
    return ref, tangent


def _signed_angle_about_axis(reference_dir, current_dir, axis_base):
    reference_dir = _normalize_or_none(reference_dir)
    current_dir = _normalize_or_none(current_dir)
    axis = _normalize_or_none(axis_base)
    if reference_dir is None or current_dir is None or axis is None:
        return None
    reference_dir = _normalize_or_none(reference_dir - np.dot(reference_dir, axis) * axis)
    current_dir = _normalize_or_none(current_dir - np.dot(current_dir, axis) * axis)
    if reference_dir is None or current_dir is None:
        return None
    return math.atan2(float(np.dot(axis, np.cross(reference_dir, current_dir))), float(np.dot(reference_dir, current_dir)))


def _raw_spoke_theta(spoke_dir_base, axis_base):
    basis0, basis1 = _plane_basis(axis_base)
    if basis0 is None or basis1 is None:
        return None
    spoke = _normalize_or_none(spoke_dir_base)
    if spoke is None:
        return None
    return math.atan2(float(np.dot(spoke, basis1)), float(np.dot(spoke, basis0)))


def estimate_valve_angle(spoke_dir_base, axis_base, state, timestamp):
    state = dict(state or {})
    spoke = _normalize_or_none(spoke_dir_base)
    if spoke is None:
        return False, 0.0, 0.0, state, None
    raw_theta = _raw_spoke_theta(spoke, axis_base)
    if state.get("angle_reference_spoke_dir") is None:
        state["angle_reference_spoke_dir"] = spoke.tolist()
        state["last_angle"] = 0.0
        state["last_timestamp"] = float(timestamp)
        state["last_spoke_dir"] = spoke.tolist()
        return True, 0.0, 0.0, state, raw_theta
    reference = np.asarray(state["angle_reference_spoke_dir"], dtype=float).reshape(3)
    signed = _signed_angle_about_axis(reference, spoke, axis_base)
    if signed is None:
        return False, float(state.get("last_angle", 0.0)), 0.0, state, raw_theta
    last_angle = float(state.get("last_angle", signed))
    angle = last_angle + wrap_to_pi(signed - last_angle)
    last_timestamp = float(state.get("last_timestamp", timestamp))
    dt = max(1e-6, float(timestamp) - last_timestamp)
    vel = (angle - last_angle) / dt
    state["last_angle"] = float(angle)
    state["last_timestamp"] = float(timestamp)
    state["last_spoke_dir"] = spoke.tolist()
    return True, float(angle), float(vel), state, raw_theta


def camera_intrinsics_from_fovy(width, height, fovy_deg):
    fovy = math.radians(float(fovy_deg))
    fy = 0.5 * float(height) / math.tan(0.5 * fovy)
    fx = fy
    return np.array(
        [
            [fx, 0.0, 0.5 * (float(width) - 1.0)],
            [0.0, fy, 0.5 * (float(height) - 1.0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def project_points_to_image(points_base, K, T_base_cam):
    points_base = np.asarray(points_base, dtype=float).reshape(-1, 3)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    T_base_cam = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
    R_cam_base = T_base_cam[:3, :3].T
    t_base = T_base_cam[:3, 3]
    points_cam = (points_base - t_base) @ R_cam_base.T
    z_forward = -points_cam[:, 2]
    pixels = np.full((points_base.shape[0], 2), np.nan, dtype=float)
    valid = z_forward > 1e-6
    pixels[valid, 0] = K[0, 0] * points_cam[valid, 0] / z_forward[valid] + K[0, 2]
    pixels[valid, 1] = -K[1, 1] * points_cam[valid, 1] / z_forward[valid] + K[1, 2]
    return pixels, z_forward, valid


def sample_depth_near(depth, pixel, radius=3):
    depth = np.asarray(depth, dtype=float)
    h, w = depth.shape[:2]
    u, v = np.asarray(pixel, dtype=float).reshape(2)
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    cx = int(round(u))
    cy = int(round(v))
    x0 = max(0, cx - int(radius))
    x1 = min(w, cx + int(radius) + 1)
    y0 = max(0, cy - int(radius))
    y1 = min(h, cy + int(radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.nanmedian(valid))


def estimate_valve_geometry_from_rgbd(rgb, depth, K, T_base_cam, config, prev_state=None):
    timestamp = time.time()
    state = dict(prev_state or {})
    min_area = int(config.get("min_mask_area_px", 50))
    min_points = int(config.get("min_points", 30))
    min_aux_points = int(config.get("min_aux_points", min_points))
    use_plane_aux = bool(config.get("use_plane_aux", True))
    depth_min = float(config.get("depth_min_m", 0.02))
    depth_max = float(config.get("depth_max_m", 5.0))
    depth_cluster_tolerance = float(config.get("depth_cluster_tolerance_m", 0.08))
    max_plane_error = float(config.get("max_plane_fit_error_m", 0.02))
    max_points = int(config.get("max_points_per_mask", 6000))

    hsv_ranges = config.get("hsv_ranges", {})
    masks = {}
    areas = {}
    points = {}
    pixels = {}
    for name in ("center", "spoke", "grasp", "plane_aux"):
        if name == "plane_aux" and not use_plane_aux:
            shape = np.asarray(depth).shape[:2]
            masks[name] = np.zeros(shape, dtype=np.uint8)
            areas[name] = 0
            pixels[name] = np.zeros((0, 2), dtype=float)
            points[name] = np.zeros((0, 3), dtype=float)
            continue
        raw_mask = segment_hsv(rgb, hsv_ranges.get(name))
        clean = clean_mask(
            raw_mask,
            open_iters=int(config.get("morph_open_iters", 1)),
            close_iters=int(config.get("morph_close_iters", 1)),
            kernel_size=int(config.get("morph_kernel_px", 3)),
        )
        mask, area = extract_largest_component(clean, min_area)
        cam_points, pix = backproject_mask_to_points(
            mask,
            depth,
            K,
            depth_min=depth_min,
            depth_max=depth_max,
            max_points=max_points,
            depth_cluster_tolerance_m=depth_cluster_tolerance,
        )
        masks[name] = mask
        areas[name] = int(area)
        pixels[name] = pix
        points[name] = transform_points(T_base_cam, cam_points)

    quality = {
        "mask_area_center_px": areas["center"],
        "mask_area_spoke_px": areas["spoke"],
        "mask_area_grasp_px": areas["grasp"],
        "mask_area_plane_aux_px": areas["plane_aux"],
        "center_num_points": int(points["center"].shape[0]),
        "spoke_num_points": int(points["spoke"].shape[0]),
        "grasp_num_points": int(points["grasp"].shape[0]),
        "plane_aux_num_points": int(points["plane_aux"].shape[0]),
    }

    center_valid = points["center"].shape[0] >= min_points
    spoke_valid = points["spoke"].shape[0] >= min_points
    grasp_valid = points["grasp"].shape[0] >= min_points
    plane_aux_valid = use_plane_aux and points["plane_aux"].shape[0] >= min_aux_points

    center_marker_base = robust_point_center(
        points["center"],
        trim_fraction=float(config.get("center_trim_fraction", 0.2)),
    ) if center_valid else None
    grasp_marker_base = robust_point_center(
        points["grasp"],
        trim_fraction=float(config.get("grasp_trim_fraction", 0.2)),
    ) if grasp_valid else None
    plane_aux_marker_base = robust_point_center(
        points["plane_aux"],
        trim_fraction=float(config.get("plane_aux_trim_fraction", 0.2)),
    ) if plane_aux_valid else None

    debug = {
        "masks": masks,
        "pixels": pixels,
        "points": points,
    }

    reasons = []
    if not center_valid:
        reasons.append("insufficient_center_points")
    if not spoke_valid:
        reasons.append("insufficient_spoke_points")

    plane_clouds = []
    if center_valid:
        plane_clouds.append(points["center"])
    if spoke_valid:
        plane_clouds.append(points["spoke"])
    if plane_aux_valid:
        plane_clouds.append(points["plane_aux"])
    if grasp_valid:
        plane_clouds.append(points["grasp"])

    has_plane_side_support = bool(plane_aux_valid or grasp_valid)
    if not has_plane_side_support:
        reasons.append("insufficient_plane_support")

    if center_marker_base is None:
        reasons.append("center_estimation_failed")
    if len(plane_clouds) >= 3:
        plane_points = np.vstack(plane_clouds)
    else:
        plane_points = np.zeros((0, 3), dtype=float)
    axis_base, _, plane_error = fit_plane_svd(plane_points)
    quality["plane_fit_error_m"] = float(plane_error)
    if axis_base is None or not np.isfinite(plane_error):
        reasons.append("plane_fit_failed")
    elif plane_error > max_plane_error:
        reasons.append("plane_fit_error_too_large")

    if axis_base is not None and center_marker_base is not None:
        cam_pos_base = np.asarray(T_base_cam, dtype=float).reshape(4, 4)[:3, 3]
        center_to_cam = cam_pos_base - center_marker_base
        if np.dot(axis_base, center_to_cam) < 0.0:
            axis_base = -axis_base

    marker_axis_offset = float(config.get("marker_axis_offset_m", 0.0))
    quality["marker_axis_offset_m"] = marker_axis_offset
    if center_marker_base is not None:
        quality["center_marker_base"] = center_marker_base.tolist()
    if grasp_marker_base is not None:
        quality["grasp_marker_base"] = grasp_marker_base.tolist()
    if plane_aux_marker_base is not None:
        quality["plane_aux_marker_base"] = plane_aux_marker_base.tolist()

    center_base = None
    plane_aux_base = None
    current_grasp_base = None
    if axis_base is not None:
        if center_marker_base is not None:
            center_base = center_marker_base - marker_axis_offset * axis_base
            debug["center_base"] = center_base
        if plane_aux_marker_base is not None:
            plane_aux_base = plane_aux_marker_base - marker_axis_offset * axis_base
            debug["plane_aux_base"] = plane_aux_base
        if grasp_marker_base is not None:
            current_grasp_base = grasp_marker_base - marker_axis_offset * axis_base

    grasp_latched = False
    grasp_base = None
    grasp_invalid_reason = None
    if grasp_valid and current_grasp_base is not None:
        grasp_base = current_grasp_base
        state["last_valid_grasp_base"] = grasp_base.tolist()
        state["last_valid_grasp_timestamp"] = float(timestamp)
    elif state.get("last_valid_grasp_base") is not None:
        try:
            grasp_base = np.asarray(state["last_valid_grasp_base"], dtype=float).reshape(3)
            grasp_latched = True
            grasp_invalid_reason = "marker_occluded_or_missing"
        except Exception:
            grasp_base = None
            grasp_invalid_reason = "marker_occluded_or_missing"
    else:
        grasp_invalid_reason = "marker_occluded_or_missing"

    if grasp_base is not None:
        debug["grasp_base"] = grasp_base
    if grasp_invalid_reason is not None and not grasp_valid:
        quality["grasp_invalid_reason"] = grasp_invalid_reason

    spoke_dir_base = None
    if spoke_valid and center_base is not None and axis_base is not None:
        spoke_dir_base = estimate_spoke_direction(points["spoke"], center_base, axis_base)
    if spoke_dir_base is None:
        reasons.append("spoke_direction_failed")
    else:
        debug["spoke_dir_base"] = spoke_dir_base

    wheel_R_base = None
    if axis_base is not None and spoke_dir_base is not None:
        wheel_R_base = construct_wheel_R_base(axis_base, spoke_dir_base)
    if wheel_R_base is None:
        reasons.append("wheel_rotation_construction_failed")

    pose_valid = (
        center_valid
        and spoke_valid
        and has_plane_side_support
        and center_base is not None
        and axis_base is not None
        and spoke_dir_base is not None
        and wheel_R_base is not None
        and np.isfinite(plane_error)
        and plane_error <= max_plane_error
    )

    grasp_R_base = None
    if pose_valid and grasp_base is not None:
        grasp_R_base = construct_grasp_R_base(
            axis_base,
            center_base,
            grasp_base,
            forward_axis_sign=float(config.get("grasp_forward_axis_sign", -1.0)),
            radial_axis=str(config.get("grasp_radial_axis", "y")),
            radial_axis_sign=float(config.get("grasp_radial_axis_sign", -1.0)),
        )
    if grasp_R_base is None and grasp_base is not None:
        reasons.append("grasp_rotation_construction_failed")

    angle_valid = False
    valve_angle = 0.0
    valve_vel = 0.0
    theta_vis_raw = None
    if pose_valid and spoke_dir_base is not None:
        angle_valid, valve_angle, valve_vel, state, theta_vis_raw = estimate_valve_angle(
            spoke_dir_base,
            axis_base,
            state,
            timestamp,
        )
    quality["theta_vis_raw"] = None if theta_vis_raw is None else float(theta_vis_raw)

    valid = bool(pose_valid and (grasp_valid or grasp_latched) and grasp_R_base is not None)
    if not valid and grasp_base is None:
        reasons.append("missing_current_or_latched_grasp")

    status = {
        "valid": bool(valid),
        "pose_valid": bool(pose_valid),
        "grasp_valid": bool(grasp_valid),
        "grasp_latched": bool(grasp_latched),
        "plane_aux_valid": bool(plane_aux_valid),
        "timestamp": timestamp,
        "frame": "base",
        "source": "sim_depth_color",
        "grasp_base_is_effective": True,
        "angle_valid": bool(angle_valid),
        "valve_angle": float(valve_angle),
        "valve_vel": float(valve_vel),
        "quality": quality,
    }
    if center_base is not None:
        status["center_base"] = center_base.tolist()
    if axis_base is not None:
        status["axis_base"] = axis_base.tolist()
        debug["axis_base"] = axis_base
    if grasp_base is not None:
        status["grasp_base"] = grasp_base.tolist()
    if grasp_R_base is not None:
        status["grasp_R_base"] = grasp_R_base.tolist()
    if center_base is not None:
        status["wheel_pos_base"] = center_base.tolist()
    if wheel_R_base is not None:
        status["wheel_R_base"] = wheel_R_base.tolist()
    if spoke_dir_base is not None:
        status["spoke_dir_base"] = spoke_dir_base.tolist()
    if plane_aux_base is not None:
        status["plane_aux_base"] = plane_aux_base.tolist()
    if grasp_invalid_reason is not None and not grasp_valid:
        status["grasp_invalid_reason"] = grasp_invalid_reason
    if reasons:
        status["reason"] = ",".join(dict.fromkeys(reasons))
    return {"status": status, "state": state, "debug": debug}
