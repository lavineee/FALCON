from __future__ import annotations

import numpy as np

from .camera_model import CameraIntrinsics, unproject_pixels


def depth_to_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    pixels: np.ndarray | None = None,
    depth_scale_m: float = 1.0,
    depth_min_m: float = 0.0,
    depth_max_m: float = float("inf"),
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    depth_arr = np.asarray(depth, dtype=float) * float(depth_scale_m)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth must have shape HxW, got {depth_arr.shape}")

    if pixels is None:
        if mask is None:
            valid = np.ones(depth_arr.shape, dtype=bool)
        else:
            valid = np.asarray(mask) > 0
        valid &= np.isfinite(depth_arr) & (depth_arr > 0.0) & (depth_arr >= float(depth_min_m)) & (depth_arr <= float(depth_max_m))
        ys, xs = np.nonzero(valid)
        uv = np.column_stack((xs.astype(float), ys.astype(float)))
        z = depth_arr[ys, xs]
    else:
        uv_all = np.asarray(pixels, dtype=float).reshape(-1, 2)
        xs = np.rint(uv_all[:, 0]).astype(int)
        ys = np.rint(uv_all[:, 1]).astype(int)
        inside = (ys >= 0) & (ys < depth_arr.shape[0]) & (xs >= 0) & (xs < depth_arr.shape[1])
        z_all = np.full(uv_all.shape[0], np.nan, dtype=float)
        z_all[inside] = depth_arr[ys[inside], xs[inside]]
        valid = inside & np.isfinite(z_all) & (z_all > 0.0) & (z_all >= float(depth_min_m)) & (z_all <= float(depth_max_m))
        uv = uv_all[valid]
        z = z_all[valid]

    if max_points is not None and uv.shape[0] > int(max_points):
        idx = np.linspace(0, uv.shape[0] - 1, int(max_points)).astype(int)
        uv = uv[idx]
        z = z[idx]

    if uv.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 2), dtype=float)
    return unproject_pixels(uv, z, intrinsics), uv


def transform_points(T_dst_src: np.ndarray, points_src: np.ndarray) -> np.ndarray:
    points = np.asarray(points_src, dtype=float).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    transform = np.asarray(T_dst_src, dtype=float).reshape(4, 4)
    return points @ transform[:3, :3].T + transform[:3, 3]


def invert_transform(T_dst_src: np.ndarray) -> np.ndarray:
    transform = np.asarray(T_dst_src, dtype=float).reshape(4, 4)
    out = np.eye(4, dtype=float)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out
