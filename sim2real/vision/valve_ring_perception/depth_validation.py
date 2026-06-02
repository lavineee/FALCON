from __future__ import annotations

from typing import Any

import numpy as np

from .plane_fit import PlaneFitResult, ransac_plane


def _valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=float)
    return np.isfinite(arr) & (arr > 0.0)


def compute_valid_depth_ratio(mask: np.ndarray, depth: np.ndarray) -> float:
    candidate = np.asarray(mask) > 0
    total = int(np.count_nonzero(candidate))
    if total == 0:
        return 0.0
    valid = candidate & _valid_depth_mask(depth)
    return float(np.count_nonzero(valid) / total)


def sample_contour_depth(contour_pixels: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(contour_pixels, dtype=float).reshape(-1, 2)
    arr = np.asarray(depth, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=bool)
    xs = np.rint(pts[:, 0]).astype(int)
    ys = np.rint(pts[:, 1]).astype(int)
    inside = (ys >= 0) & (ys < arr.shape[0]) & (xs >= 0) & (xs < arr.shape[1])
    values = np.full(pts.shape[0], np.nan, dtype=float)
    values[inside] = arr[ys[inside], xs[inside]]
    valid = inside & np.isfinite(values) & (values > 0.0)
    return values, valid


def compute_contour_depth_jump_ratio(
    contour_pixels: np.ndarray,
    depth: np.ndarray,
    jump_threshold_m: float,
) -> float:
    pts = np.asarray(contour_pixels, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 1.0
    center = np.nanmedian(pts, axis=0)
    angles = np.mod(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]), 2.0 * np.pi)
    order = np.argsort(angles)
    values, valid = sample_contour_depth(pts[order], depth)
    valid_indices = np.nonzero(valid)[0]
    if valid_indices.size < 2:
        return 1.0
    values = values[valid_indices]
    jumps = np.abs(np.diff(np.r_[values, values[0]])) > float(jump_threshold_m)
    return float(np.count_nonzero(jumps) / jumps.size)


def fit_candidate_plane(points_3d: np.ndarray, threshold_m: float = 0.01, iterations: int = 250, min_inliers: int = 3, random_seed: int | None = None) -> PlaneFitResult:
    return ransac_plane(
        points_3d,
        threshold_m=threshold_m,
        iterations=iterations,
        min_inliers=min_inliers,
        random_seed=random_seed,
    )


def compute_plane_residual_stats(points_3d: np.ndarray, plane: PlaneFitResult) -> dict[str, float]:
    pts = np.asarray(points_3d, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return {
            "plane_rmse_m": float("inf"),
            "plane_median_abs_m": float("inf"),
            "plane_p95_abs_m": float("inf"),
            "plane_inlier_ratio": 0.0,
        }
    residuals = np.abs((pts - plane.point.reshape(1, 3)) @ plane.normal.reshape(3))
    return {
        "plane_rmse_m": float(np.sqrt(np.mean(residuals * residuals))),
        "plane_median_abs_m": float(np.median(residuals)),
        "plane_p95_abs_m": float(np.quantile(residuals, 0.95)),
        "plane_inlier_ratio": float(np.count_nonzero(residuals <= plane.threshold_m) / max(1, pts.shape[0])),
    }


def compute_depth_quality(
    candidate: Any,
    depth: np.ndarray,
    points_3d: np.ndarray | None = None,
    plane: PlaneFitResult | None = None,
    jump_threshold_m: float = 0.05,
) -> dict[str, float]:
    quality = {
        "valid_depth_ratio": compute_valid_depth_ratio(candidate.candidate_mask, depth),
        "contour_depth_jump_ratio": compute_contour_depth_jump_ratio(
            candidate.outer_contour_pixels,
            depth,
            jump_threshold_m,
        ),
    }
    if points_3d is not None and plane is not None:
        quality.update(compute_plane_residual_stats(points_3d, plane))
    return quality
