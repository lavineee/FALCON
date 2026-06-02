import numpy as np

from sim2real.vision.valve_ring_perception.outer_envelope import extract_outer_envelope_points


def test_outer_envelope_prefers_outer_points_over_spokes_and_center():
    angles = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    outer = 0.2 * np.column_stack((np.cos(angles), np.sin(angles)))
    center_blob = 0.03 * np.column_stack((np.cos(angles[:80]), np.sin(angles[:80])))
    spoke_x = np.column_stack((np.linspace(-0.18, 0.18, 100), np.zeros(100)))
    spoke_y = np.column_stack((np.zeros(100), np.linspace(-0.18, 0.18, 100)))
    points = np.vstack((outer, center_blob, spoke_x, spoke_y))

    result = extract_outer_envelope_points(points, angle_bins=72, points_per_bin=2)

    assert result.angular_coverage_ratio > 0.95
    assert result.points_2d.shape[0] >= 100
    radii = np.linalg.norm(result.points_2d - result.rough_center_2d, axis=1)
    assert np.median(radii) > 0.18
