from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlaneFitResult:
    normal: np.ndarray
    point: np.ndarray
    inlier_mask: np.ndarray
    rmse_m: float
    inlier_ratio: float
    threshold_m: float


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < eps:
        return None
    return arr / norm


def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("at least 3 points are required to fit a plane")
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vt[-1])
    if normal is None:
        raise ValueError("degenerate plane fit")
    residuals = centered @ normal
    return normal, centroid, float(np.sqrt(np.mean(residuals * residuals)))


def _plane_from_three(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    a, b, c = points
    normal = normalize(np.cross(b - a, c - a))
    if normal is None:
        return None
    return normal, a


def ransac_plane(
    points: np.ndarray,
    threshold_m: float = 0.01,
    iterations: int = 250,
    min_inliers: int = 3,
    random_seed: int | None = None,
) -> PlaneFitResult:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("at least 3 points are required for plane RANSAC")
    rng = np.random.default_rng(random_seed)
    best_mask: np.ndarray | None = None
    best_count = -1
    best_rmse = float("inf")

    for _ in range(int(iterations)):
        sample_idx = rng.choice(pts.shape[0], size=3, replace=False)
        candidate = _plane_from_three(pts[sample_idx])
        if candidate is None:
            continue
        normal, point = candidate
        distances = np.abs((pts - point) @ normal)
        mask = distances <= float(threshold_m)
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        rmse = float(np.sqrt(np.mean(distances[mask] ** 2)))
        if count > best_count or (count == best_count and rmse < best_rmse):
            best_count = count
            best_rmse = rmse
            best_mask = mask

    if best_mask is None or best_count < int(min_inliers):
        normal, point, rmse = fit_plane_svd(pts)
        distances = np.abs((pts - point) @ normal)
        best_mask = distances <= float(threshold_m)
    normal, point, rmse = fit_plane_svd(pts[best_mask])
    distances = np.abs((pts - point) @ normal)
    inlier_mask = distances <= float(threshold_m)
    if np.count_nonzero(inlier_mask) >= 3:
        normal, point, rmse = fit_plane_svd(pts[inlier_mask])
        distances = np.abs((pts - point) @ normal)
        inlier_mask = distances <= float(threshold_m)
        rmse = float(np.sqrt(np.mean(distances[inlier_mask] ** 2)))
    ratio = float(np.count_nonzero(inlier_mask) / max(1, pts.shape[0]))
    return PlaneFitResult(
        normal=normal,
        point=point,
        inlier_mask=inlier_mask,
        rmse_m=rmse,
        inlier_ratio=ratio,
        threshold_m=float(threshold_m),
    )
