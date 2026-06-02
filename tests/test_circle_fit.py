import numpy as np

from sim2real.vision.valve_ring_perception.circle_fit import ransac_circle_2d


def test_ransac_circle_recovers_center_radius_with_outliers():
    rng = np.random.default_rng(8)
    center = np.array([0.12, -0.04])
    radius = 0.23
    angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    points += rng.normal(scale=0.001, size=points.shape)
    outliers = rng.uniform(-0.5, 0.5, size=(40, 2))
    all_points = np.vstack((points, outliers))

    result = ransac_circle_2d(
        all_points,
        threshold_m=0.006,
        iterations=300,
        radius_range_m=(0.1, 0.4),
        random_seed=2,
    )

    np.testing.assert_allclose(result.center_2d, center, atol=0.003)
    assert abs(result.radius_m - radius) < 0.003
    assert result.rmse_m < 0.0025
    assert result.inlier_ratio > 0.75
