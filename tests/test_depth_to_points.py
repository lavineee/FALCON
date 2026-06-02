import numpy as np

from sim2real.vision.valve_ring_perception.camera_model import CameraIntrinsics
from sim2real.vision.valve_ring_perception.pointcloud import depth_to_points


def test_depth_to_points_realsense_optical_frame():
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.0, cy=1.0)
    depth = np.full((3, 3), 2.0, dtype=float)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255
    mask[1, 2] = 255
    mask[0, 1] = 255

    points, pixels = depth_to_points(depth, intr, mask=mask)

    by_pixel = {tuple(pixel.astype(int)): point for point, pixel in zip(points, pixels)}
    np.testing.assert_allclose(by_pixel[(1, 1)], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(by_pixel[(2, 1)], [0.02, 0.0, 2.0])
    np.testing.assert_allclose(by_pixel[(1, 0)], [0.0, -0.02, 2.0])


def test_depth_to_points_filters_invalid_depth():
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    depth = np.array([[0.0, 1.0], [np.nan, 5.0]], dtype=float)
    points, pixels = depth_to_points(depth, intr, depth_min_m=0.5, depth_max_m=2.0)
    assert points.shape == (1, 3)
    np.testing.assert_allclose(pixels[0], [1.0, 0.0])
