from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for an aligned RGB/depth image.

    The camera frame follows the RealSense optical convention: x right, y down,
    z forward. Depth values are metric distance along camera +z after scaling.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_matrix(cls, matrix: Any, width: int | None = None, height: int | None = None) -> "CameraIntrinsics":
        k = np.asarray(matrix, dtype=float).reshape(3, 3)
        return cls(
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            width=None if width is None else int(width),
            height=None if height is None else int(height),
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        if "K" in data:
            return cls.from_matrix(data["K"], width=data.get("width"), height=data.get("height"))
        if "camera_matrix" in data:
            return cls.from_matrix(data["camera_matrix"], width=data.get("width"), height=data.get("height"))
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=None if data.get("width") is None else int(data["width"]),
            height=None if data.get("height") is None else int(data["height"]),
        )

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )


def coerce_intrinsics(value: CameraIntrinsics | dict[str, Any] | np.ndarray) -> CameraIntrinsics:
    if isinstance(value, CameraIntrinsics):
        return value
    if isinstance(value, dict):
        return CameraIntrinsics.from_mapping(value)
    return CameraIntrinsics.from_matrix(value)


def project_points(points_cam: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    z = points[:, 2]
    valid = np.isfinite(z) & (z > 1e-9)
    pixels = np.full((points.shape[0], 2), np.nan, dtype=float)
    pixels[valid, 0] = intrinsics.fx * points[valid, 0] / z[valid] + intrinsics.cx
    pixels[valid, 1] = intrinsics.fy * points[valid, 1] / z[valid] + intrinsics.cy
    return pixels, valid


def unproject_pixels(
    pixels: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    z = np.asarray(depth_m, dtype=float).reshape(-1)
    x = (uv[:, 0] - intrinsics.cx) * z / intrinsics.fx
    y = (uv[:, 1] - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z))
