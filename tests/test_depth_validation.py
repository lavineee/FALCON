import numpy as np

from sim2real.vision.valve_ring_perception.depth_validation import (
    compute_contour_depth_jump_ratio,
    compute_valid_depth_ratio,
    fit_candidate_plane,
    compute_plane_residual_stats,
)


def test_valid_depth_ratio_counts_only_positive_finite_depth():
    mask = np.array([[255, 255, 0], [0, 255, 255]], dtype=np.uint8)
    depth = np.array([[1.0, 0.0, 1.0], [np.nan, 2.0, 3.0]], dtype=float)
    assert compute_valid_depth_ratio(mask, depth) == 0.75


def test_contour_depth_jump_ratio_detects_discontinuities():
    depth = np.ones((20, 20), dtype=float)
    depth[:, 10:] = 1.2
    contour = np.array([[5, 5], [9, 5], [12, 5], [15, 5], [15, 10], [5, 10]], dtype=float)
    ratio = compute_contour_depth_jump_ratio(contour, depth, jump_threshold_m=0.05)
    assert ratio > 0.0


def test_plane_residual_stats_for_flat_points():
    yy, xx = np.mgrid[0:10, 0:10]
    points = np.column_stack((xx.ravel() * 0.01, yy.ravel() * 0.01, np.ones(xx.size)))
    plane = fit_candidate_plane(points, threshold_m=0.002, random_seed=1)
    stats = compute_plane_residual_stats(points, plane)
    assert stats["plane_rmse_m"] < 1e-9
    assert stats["plane_inlier_ratio"] == 1.0
