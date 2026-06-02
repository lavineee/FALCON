import numpy as np

from sim2real.vision.valve_ring_perception.plane_fit import ransac_plane


def test_ransac_plane_recovers_noisy_plane_with_outliers():
    rng = np.random.default_rng(4)
    normal = np.array([0.2, -0.3, 0.9327379])
    normal = normal / np.linalg.norm(normal)
    center = np.array([0.4, -0.2, 1.1])
    u = np.array([1.0, 0.0, 0.0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    coeff = rng.uniform(-0.3, 0.3, size=(200, 2))
    points = center + coeff[:, :1] * u + coeff[:, 1:] * v
    points += rng.normal(scale=0.001, size=points.shape)
    outliers = rng.uniform(-1.0, 1.0, size=(30, 3))
    all_points = np.vstack((points, outliers))

    result = ransac_plane(all_points, threshold_m=0.006, iterations=200, random_seed=7)

    assert abs(float(np.dot(result.normal, normal))) > 0.999
    assert result.rmse_m < 0.0025
    assert result.inlier_ratio > 0.80
