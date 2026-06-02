from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OuterEnvelopeResult:
    points_2d: np.ndarray
    rough_center_2d: np.ndarray
    selected_indices: np.ndarray
    rejected_indices: np.ndarray
    angular_coverage_ratio: float
    occupied_bins: int
    total_bins: int
    radius_upper_bound_m: float | None = None


def estimate_rough_center_2d(points_2d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        raise ValueError("at least one point is required for rough center estimation")
    lo = np.quantile(pts, 0.05, axis=0)
    hi = np.quantile(pts, 0.95, axis=0)
    center = 0.5 * (lo + hi)
    if not np.all(np.isfinite(center)):
        center = np.median(pts, axis=0)
    return center.astype(float)


def extract_outer_envelope_points(
    points_2d: np.ndarray,
    rough_center_2d: np.ndarray | None = None,
    angle_bins: int = 72,
    points_per_bin: int = 3,
    top_fraction: float = 0.20,
    min_points_per_bin: int = 1,
    outlier_mad_scale: float = 6.0,
    outlier_min_margin_m: float = 0.03,
) -> OuterEnvelopeResult:
    """Select angular-bin outer envelope points from a handwheel point set.

    Spokes and center bosses may dominate the raw mask point count, so final
    circle fitting should not use all projected mask points. This function bins
    points by angle around a coarse center and keeps only the farthest samples
    per occupied bin.
    """

    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        raise ValueError("at least one point is required for outer envelope extraction")
    bins = max(8, int(angle_bins))
    keep_per_bin = max(1, int(points_per_bin))
    top_frac = min(1.0, max(0.01, float(top_fraction)))
    min_per_bin = max(1, int(min_points_per_bin))
    center = estimate_rough_center_2d(pts) if rough_center_2d is None else np.asarray(rough_center_2d, dtype=float).reshape(2)

    rel = pts - center
    radii = np.linalg.norm(rel, axis=1)
    finite = np.isfinite(radii) & (radii > 1e-9)
    angles = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
    bin_indices = np.floor(angles / (2.0 * np.pi) * bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    selected: list[int] = []
    occupied = 0
    for bin_id in range(bins):
        in_bin = np.nonzero(finite & (bin_indices == bin_id))[0]
        if in_bin.size < min_per_bin:
            continue
        threshold = float(np.quantile(radii[in_bin], 1.0 - top_frac))
        top = in_bin[radii[in_bin] >= threshold]
        order = np.argsort(radii[top])
        selected.extend(top[order[-keep_per_bin:]].tolist())

    if selected:
        selected_indices = np.array(sorted(set(selected)), dtype=int)
        selected_radii = radii[selected_indices]
        median_radius = float(np.median(selected_radii))
        mad = float(np.median(np.abs(selected_radii - median_radius)))
        robust_sigma = 1.4826 * mad
        upper_margin = max(float(outlier_min_margin_m), float(outlier_mad_scale) * robust_sigma)
        radius_upper = median_radius + upper_margin
        keep = selected_radii <= radius_upper
        selected_indices = selected_indices[keep]
    else:
        selected_indices = np.zeros(0, dtype=int)
        radius_upper = None
    if selected_indices.size:
        occupied = int(np.unique(bin_indices[selected_indices]).size)
    else:
        occupied = 0
    selected_mask = np.zeros(pts.shape[0], dtype=bool)
    selected_mask[selected_indices] = True
    rejected_indices = np.nonzero(~selected_mask)[0].astype(int)
    return OuterEnvelopeResult(
        points_2d=pts[selected_indices],
        rough_center_2d=center,
        selected_indices=selected_indices,
        rejected_indices=rejected_indices,
        angular_coverage_ratio=float(occupied / bins),
        occupied_bins=int(occupied),
        total_bins=int(bins),
        radius_upper_bound_m=None if radius_upper is None else float(radius_upper),
    )
