import numpy as np

from sim2real.vision.valve_ring_perception import CameraIntrinsics, ValveRingEstimator, ValveRingEstimatorConfig


def _synthetic_fronto_parallel_ring():
    height, width = 120, 160
    intr = CameraIntrinsics(fx=120.0, fy=120.0, cx=80.0, cy=60.0, width=width, height=height)
    center = np.array([0.0, 0.0, 1.0])
    radius_outer = 0.22
    radius_inner = 0.17
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=float)
    yy, xx = np.mgrid[0:height, 0:width]
    x = (xx - intr.cx) / intr.fx
    y = (yy - intr.cy) / intr.fy
    radial = np.sqrt(x * x + y * y)
    ring = (radial >= radius_inner) & (radial <= radius_outer)
    rgb[ring] = np.array([255, 0, 0], dtype=np.uint8)
    depth[ring] = center[2]
    return rgb, depth, intr, center, radius_outer


def test_synthetic_estimator_outputs_center_radius_axis():
    rgb, depth, intr, center, radius = _synthetic_fronto_parallel_ring()
    cfg = ValveRingEstimatorConfig(
        depth_min_m=0.5,
        depth_max_m=2.0,
        mask_open_iters=0,
        mask_close_iters=0,
        min_circle_inlier_ratio=0.35,
    )
    estimator = ValveRingEstimator(cfg)
    estimate = estimator.estimate(rgb, depth, intr, np.eye(4))

    assert estimate.valid, estimate.quality.failure_reasons
    np.testing.assert_allclose(estimate.center_base, center, atol=0.01)
    assert abs(estimate.radius_m - radius) < 0.012
    np.testing.assert_allclose(estimate.axis_base, [0.0, 0.0, -1.0], atol=1e-6)


def _synthetic_handwheel_with_spokes_and_outlier_handle():
    height, width = 160, 200
    intr = CameraIntrinsics(fx=150.0, fy=150.0, cx=100.0, cy=80.0, width=width, height=height)
    center = np.array([0.0, 0.0, 1.0])
    radius_outer = 0.24
    radius_inner = 0.20
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=float)
    yy, xx = np.mgrid[0:height, 0:width]
    x = (xx - intr.cx) / intr.fx
    y = (yy - intr.cy) / intr.fy
    radial = np.sqrt(x * x + y * y)
    outer_ring = (radial >= radius_inner) & (radial <= radius_outer)
    hub = radial <= 0.045
    spoke_x = (np.abs(y) <= 0.018) & (np.abs(x) <= 0.19)
    spoke_y = (np.abs(x) <= 0.018) & (np.abs(y) <= 0.19)
    handwheel = outer_ring | hub | spoke_x | spoke_y
    rgb[handwheel] = np.array([255, 0, 0], dtype=np.uint8)
    depth[handwheel] = center[2]

    handle = (x >= 0.23) & (x <= 0.33) & (np.abs(y) <= 0.02)
    rgb[handle] = np.array([255, 0, 0], dtype=np.uint8)
    depth[handle] = 0.82
    return rgb, depth, intr, center, radius_outer


def test_synthetic_handwheel_uses_outer_envelope_not_spoke_average():
    rgb, depth, intr, center, radius = _synthetic_handwheel_with_spokes_and_outlier_handle()
    cfg = ValveRingEstimatorConfig(
        depth_min_m=0.5,
        depth_max_m=2.0,
        mask_open_iters=0,
        mask_close_iters=0,
        min_circle_inlier_ratio=0.35,
        min_angular_coverage_ratio=0.7,
    )
    estimator = ValveRingEstimator(cfg)
    estimate = estimator.estimate(rgb, depth, intr, np.eye(4))

    assert estimate.valid, estimate.quality.failure_reasons
    np.testing.assert_allclose(estimate.center_base, center, atol=0.012)
    assert abs(estimate.radius_m - radius) < 0.015
    assert estimate.quality.envelope_num_points >= cfg.min_envelope_points
    assert estimate.quality.angular_coverage_ratio > 0.9
    assert estimate.quality.radius_range_valid
