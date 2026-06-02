from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plane_fit import normalize


@dataclass
class CircleFitResult:
    center_2d: np.ndarray
    radius_m: float
    inlier_mask: np.ndarray
    rmse_m: float
    inlier_ratio: float
    threshold_m: float


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normalize(normal)
    if n is None:
        raise ValueError("plane normal must be nonzero")
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n))) > 0.85:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = ref - np.dot(ref, n) * n
    u = normalize(u)
    if u is None:
        raise ValueError("failed to construct plane basis")
    v = normalize(np.cross(n, u))
    if v is None:
        raise ValueError("failed to construct plane basis")
    return u, v


def project_points_to_plane(
    points: np.ndarray,
    plane_point: np.ndarray,
    normal: np.ndarray,
    basis_u: np.ndarray | None = None,
    basis_v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    p0 = np.asarray(plane_point, dtype=float).reshape(3)
    n = normalize(normal)
    if n is None:
        raise ValueError("plane normal must be nonzero")
    if basis_u is None or basis_v is None:
        basis_u, basis_v = plane_basis(n)
    rel = pts - p0
    projected = pts - np.outer(rel @ n, n)
    rel_proj = projected - p0
    points_2d = np.column_stack((rel_proj @ basis_u, rel_proj @ basis_v))
    return points_2d, projected, basis_u, basis_v


def fit_circle_2d(points_2d: np.ndarray) -> tuple[np.ndarray, float, float]:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("at least 3 points are required to fit a circle")
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack((x, y, np.ones_like(x)))
    b = -(x * x + y * y)
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = np.array([-0.5 * coef[0], -0.5 * coef[1]], dtype=float)
    radius_sq = float(center @ center - coef[2])
    if radius_sq <= 0.0 or not np.isfinite(radius_sq):
        raise ValueError("degenerate circle fit")
    radius = float(np.sqrt(radius_sq))
    residuals = np.linalg.norm(pts - center, axis=1) - radius
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    return center, radius, rmse


def circle_from_three(points_2d: np.ndarray) -> tuple[np.ndarray, float] | None:
    pts = np.asarray(points_2d, dtype=float).reshape(3, 2)
    a = np.column_stack((2.0 * pts, np.ones(3)))
    b = np.sum(pts * pts, axis=1)
    det = float(np.linalg.det(a))
    if abs(det) < 1e-12:
        return None
    sol = np.linalg.solve(a, b)
    center = sol[:2]
    radius_sq = float(sol[2] + center @ center)
    if radius_sq <= 0.0 or not np.isfinite(radius_sq):
        return None
    return center, float(np.sqrt(radius_sq))


def ransac_circle_2d(
    points_2d: np.ndarray,
    threshold_m: float = 0.015,
    iterations: int = 500,
    min_inliers: int = 3,
    radius_range_m: tuple[float, float] | None = None,
    random_seed: int | None = None,
) -> CircleFitResult:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("at least 3 points are required for circle RANSAC")
    rng = np.random.default_rng(random_seed)
    best_mask: np.ndarray | None = None
    best_count = -1
    best_rmse = float("inf")
    min_radius = -float("inf")
    max_radius = float("inf")
    if radius_range_m is not None:
        min_radius, max_radius = (float(radius_range_m[0]), float(radius_range_m[1]))

    for _ in range(int(iterations)):
        sample_idx = rng.choice(pts.shape[0], size=3, replace=False)
        candidate = circle_from_three(pts[sample_idx])
        if candidate is None:
            continue
        center, radius = candidate
        if radius < min_radius or radius > max_radius:
            continue
        residuals = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
        mask = residuals <= float(threshold_m)
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        rmse = float(np.sqrt(np.mean(residuals[mask] ** 2)))
        if count > best_count or (count == best_count and rmse < best_rmse):
            best_mask = mask
            best_count = count
            best_rmse = rmse

    if best_mask is None or best_count < int(min_inliers):
        center, radius, _ = fit_circle_2d(pts)
        residuals = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
        best_mask = residuals <= float(threshold_m)

    if np.count_nonzero(best_mask) >= 3:
        center, radius, _ = fit_circle_2d(pts[best_mask])
    else:
        center, radius, _ = fit_circle_2d(pts)
    residuals = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
    inlier_mask = residuals <= float(threshold_m)
    if np.count_nonzero(inlier_mask) >= 3:
        center, radius, _ = fit_circle_2d(pts[inlier_mask])
        residuals = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
        inlier_mask = residuals <= float(threshold_m)
    rmse = float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))) if np.any(inlier_mask) else float("inf")
    return CircleFitResult(
        center_2d=center,
        radius_m=float(radius),
        inlier_mask=inlier_mask,
        rmse_m=rmse,
        inlier_ratio=float(np.count_nonzero(inlier_mask) / max(1, pts.shape[0])),
        threshold_m=float(threshold_m),
    )


def unproject_plane_point(plane_point: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray, point_2d: np.ndarray) -> np.ndarray:
    p0 = np.asarray(plane_point, dtype=float).reshape(3)
    uv = np.asarray(point_2d, dtype=float).reshape(2)
    return p0 + uv[0] * np.asarray(basis_u, dtype=float).reshape(3) + uv[1] * np.asarray(basis_v, dtype=float).reshape(3)
