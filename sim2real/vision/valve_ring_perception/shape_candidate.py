from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ValveRingEstimatorConfig
from .segmentation import external_contour_mask


@dataclass
class HandwheelCandidate:
    candidate_mask: np.ndarray
    outer_contour_mask: np.ndarray
    outer_contour_pixels: np.ndarray
    roi_xywh: tuple[int, int, int, int]
    area_px: int
    source: str
    shape_score: float = 0.0
    ellipse: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def handwheel_mask(self) -> np.ndarray:
        return self.candidate_mask

    @property
    def mask(self) -> np.ndarray:
        return self.candidate_mask


def _roi_bounds(shape: tuple[int, int], roi_xywh: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    h, w = shape
    if roi_xywh is None:
        return 0, 0, w, h
    x, y, rw, rh = (int(v) for v in roi_xywh)
    x = max(0, min(w, x))
    y = max(0, min(h, y))
    rw = max(0, min(w - x, rw))
    rh = max(0, min(h - y, rh))
    return x, y, rw, rh


def _ellipse_fit_error_px(contour_xy: np.ndarray, center: tuple[float, float], axes: tuple[float, float], angle_deg: float) -> float:
    pts = np.asarray(contour_xy, dtype=float).reshape(-1, 2)
    major = max(float(axes[0]), float(axes[1]))
    minor = max(1e-6, min(float(axes[0]), float(axes[1])))
    theta = -np.deg2rad(float(angle_deg))
    c = np.cos(theta)
    s = np.sin(theta)
    rel = pts - np.asarray(center, dtype=float).reshape(2)
    x = c * rel[:, 0] - s * rel[:, 1]
    y = s * rel[:, 0] + c * rel[:, 1]
    norm_radius = np.sqrt((x / (0.5 * major)) ** 2 + (y / (0.5 * minor)) ** 2)
    avg_radius = 0.25 * (major + minor)
    return float(np.mean(np.abs(norm_radius - 1.0)) * avg_radius)


def _angular_coverage(contour_xy: np.ndarray, center: tuple[float, float], bins: int = 36) -> float:
    pts = np.asarray(contour_xy, dtype=float).reshape(-1, 2)
    rel = pts - np.asarray(center, dtype=float).reshape(2)
    radii = np.linalg.norm(rel, axis=1)
    valid = radii > 1e-6
    if not np.any(valid):
        return 0.0
    angles = np.mod(np.arctan2(rel[valid, 1], rel[valid, 0]), 2.0 * np.pi)
    idx = np.floor(angles / (2.0 * np.pi) * int(bins)).astype(int)
    idx = np.clip(idx, 0, int(bins) - 1)
    return float(np.unique(idx).size / int(bins))


def _sample_ellipse_perimeter(
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
    num_points: int,
) -> np.ndarray:
    cx, cy = center
    axis_a, axis_b = axes
    a = 0.5 * float(axis_a)
    b = 0.5 * float(axis_b)
    theta = np.linspace(0.0, 2.0 * np.pi, max(16, int(num_points)), endpoint=False)
    local = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
    rot = np.deg2rad(float(angle_deg))
    c = np.cos(rot)
    s = np.sin(rot)
    x = c * local[:, 0] - s * local[:, 1] + float(cx)
    y = s * local[:, 0] + c * local[:, 1] + float(cy)
    return np.column_stack((x, y))


def _normalize_ellipse(
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    cx, cy = float(center[0]), float(center[1])
    axis_a, axis_b = float(axes[0]), float(axes[1])
    angle = float(angle_deg)
    if axis_b > axis_a:
        axis_a, axis_b = axis_b, axis_a
        angle += 90.0
    angle = angle % 180.0
    return (cx, cy), (axis_a, axis_b), angle


def _ellipse_band_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
    thickness_px: float,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = float(center[0]), float(center[1])
    major, minor = float(axes[0]), max(1e-6, float(axes[1]))
    theta = -np.deg2rad(float(angle_deg))
    c = np.cos(theta)
    s = np.sin(theta)
    rel_x = xx.astype(float) - cx
    rel_y = yy.astype(float) - cy
    x = c * rel_x - s * rel_y
    y = s * rel_x + c * rel_y
    norm_radius = np.sqrt((x / (0.5 * major)) ** 2 + (y / (0.5 * minor)) ** 2)
    avg_radius = max(1.0, 0.25 * (major + minor))
    return np.abs(norm_radius - 1.0) * avg_radius <= float(thickness_px)


def _ellipse_metadata(
    center_abs: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
    fit_error_px: float,
    aspect_ratio: float,
    compactness: float,
    coverage: float,
    area_px: float,
    contour_points: int,
    generator: str,
) -> dict[str, Any]:
    major, minor = float(axes[0]), float(axes[1])
    return {
        "center_px": [float(center_abs[0]), float(center_abs[1])],
        "axes_px": [major, minor],
        "angle_deg": float(angle_deg),
        "ellipse_center_px": [float(center_abs[0]), float(center_abs[1])],
        "major_axis_px": major,
        "minor_axis_px": minor,
        "ellipse_angle_deg": float(angle_deg),
        "fit_error_px": float(fit_error_px),
        "aspect_ratio": float(aspect_ratio),
        "compactness": float(compactness),
        "angular_coverage_ratio": float(coverage),
        "contour_area_px": float(area_px),
        "contour_points": int(contour_points),
        "generator": generator,
    }


def _score_shape(
    aspect_ratio: float,
    fit_error_px: float,
    angular_coverage_ratio: float,
    compactness: float,
    config: ValveRingEstimatorConfig,
) -> float:
    aspect_score = max(0.0, 1.0 - (float(aspect_ratio) - 1.0) / max(1e-6, config.shape_max_aspect_ratio - 1.0))
    fit_score = max(0.0, 1.0 - float(fit_error_px) / max(1e-6, config.shape_max_ellipse_fit_error_px))
    compact_score = min(1.0, max(0.0, float(compactness)))
    return float(0.35 * angular_coverage_ratio + 0.30 * aspect_score + 0.25 * fit_score + 0.10 * compact_score)


def _hough_ellipse_candidates(
    cv2: Any,
    gray: np.ndarray,
    edges: np.ndarray,
    image_shape: tuple[int, int],
    roi_xywh: tuple[int, int, int, int],
    config: ValveRingEstimatorConfig,
) -> list[HandwheelCandidate]:
    if not bool(config.shape_hough_enabled):
        return []

    h, w = image_shape
    x0, y0, rw, rh = roi_xywh
    max_axis = float(config.shape_max_axis_px) if config.shape_max_axis_px > 0 else float(max(rw, rh))
    min_radius = max(4, int(round(0.5 * float(config.shape_min_axis_px))))
    max_radius = max(min_radius + 1, int(round(0.5 * max_axis)))
    min_dist = float(config.shape_hough_min_dist_px)
    if min_dist <= 0.0:
        min_dist = max(16.0, 0.25 * min(rw, rh))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=max(1.0, float(config.shape_hough_dp)),
        minDist=min_dist,
        param1=float(config.shape_canny_threshold2),
        param2=float(config.shape_hough_param2),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    yy, xx = np.mgrid[0:rh, 0:rw]
    edge_pixels = np.asarray(edges) > 0
    thickness = max(2.0, float(config.shape_hough_band_thickness_px))
    out: list[HandwheelCandidate] = []
    for circle in np.asarray(circles[0], dtype=float)[: max(1, int(config.shape_hough_max_candidates))]:
        cx, cy, radius = (float(v) for v in circle[:3])
        if not np.isfinite(radius) or radius <= 1.0:
            continue
        diameter = 2.0 * radius
        if diameter < float(config.shape_min_axis_px) or diameter > max_axis:
            continue
        circle_dist = np.sqrt((xx.astype(float) - cx) ** 2 + (yy.astype(float) - cy) ** 2)
        circle_band = np.abs(circle_dist - radius) <= max(thickness, 0.18 * radius)
        ys, xs = np.nonzero(edge_pixels & circle_band)
        if xs.size < max(5, int(config.shape_min_contour_points)):
            continue
        edge_xy = np.column_stack((xs.astype(float), ys.astype(float)))
        fitted = cv2.fitEllipse(edge_xy.reshape(-1, 1, 2).astype(np.float32))
        (ecx, ecy), (axis_a, axis_b), angle_deg = fitted
        (ecx, ecy), (major, minor), angle_deg = _normalize_ellipse((ecx, ecy), (axis_a, axis_b), angle_deg)
        if major < float(config.shape_min_axis_px) or minor < float(config.shape_min_axis_px):
            continue
        if major > max_axis:
            continue
        aspect = major / max(1e-6, minor)
        if aspect > float(config.shape_max_aspect_ratio):
            continue
        fit_error = _ellipse_fit_error_px(edge_xy, (ecx, ecy), (major, minor), angle_deg)
        if fit_error > max(float(config.shape_max_ellipse_fit_error_px) * 2.0, thickness):
            continue
        ellipse_band = _ellipse_band_mask((rh, rw), (ecx, ecy), (major, minor), angle_deg, thickness)
        candidate_roi = (edge_pixels & ellipse_band).astype(np.uint8) * 255
        area_px = int(np.count_nonzero(candidate_roi))
        if area_px < int(config.min_mask_area_px):
            continue
        ys, xs = np.nonzero(candidate_roi > 0)
        ellipse_edge_xy = np.column_stack((xs.astype(float), ys.astype(float)))
        coverage = _angular_coverage(ellipse_edge_xy, (ecx, ecy), bins=36)
        if coverage < max(0.15, 0.6 * float(config.shape_min_angular_coverage_ratio)):
            continue
        band_area = max(1.0, np.pi * (major + minor) * (2.0 * thickness + 1.0))
        edge_density = min(1.0, float(area_px) / band_area * 4.0)
        axis_score = min(1.0, major / max(1.0, 0.70 * min(rw, rh)))
        score = float(0.55 * coverage + 0.25 * edge_density + 0.20 * axis_score)

        candidate_mask = np.zeros((h, w), dtype=np.uint8)
        candidate_mask[y0 : y0 + rh, x0 : x0 + rw] = candidate_roi
        center_abs = (float(ecx + x0), float(ecy + y0))
        sampled = _sample_ellipse_perimeter(center_abs, (major, minor), angle_deg, config.ellipse_sample_points)
        if sampled.shape[0] > int(config.max_outer_points):
            idx = np.linspace(0, sampled.shape[0] - 1, int(config.max_outer_points)).astype(int)
            sampled = sampled[idx]
        out.append(
            HandwheelCandidate(
                candidate_mask=candidate_mask,
                outer_contour_mask=candidate_mask.copy(),
                outer_contour_pixels=sampled,
                roi_xywh=(x0, y0, rw, rh),
                area_px=int(np.count_nonzero(candidate_mask)),
                source="shape",
                shape_score=score,
                ellipse=_ellipse_metadata(
                    center_abs,
                    (major, minor),
                    angle_deg,
                    fit_error,
                    aspect,
                    edge_density,
                    coverage,
                    area_px,
                    area_px,
                    "hough_ellipse",
                ),
                metadata={"edge_area_px": area_px, "generator": "hough_ellipse", "radius_px": float(radius)},
            )
        )
    return out


def generate_shape_candidates(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    config: ValveRingEstimatorConfig,
) -> list[HandwheelCandidate]:
    """Generate handwheel candidates from grayscale edge/ellipse evidence.

    Candidate masks are sparse edge/contour masks, not filled annuli. Interior
    holes are therefore not artificially filled and used as depth support.
    """

    try:
        import cv2  # type: ignore
    except ImportError:
        return []

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {image.shape}")
    h, w = image.shape[:2]
    x0, y0, rw, rh = _roi_bounds((h, w), config.roi_xywh)
    if rw <= 0 or rh <= 0:
        return []

    roi_rgb = image[y0 : y0 + rh, x0 : x0 + rw]
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    if config.shape_use_clahe:
        grid = max(1, int(config.shape_clahe_tile_grid_size))
        clahe = cv2.createCLAHE(clipLimit=float(config.shape_clahe_clip_limit), tileGridSize=(grid, grid))
        gray = clahe.apply(gray)
    kernel = int(config.shape_blur_kernel_size)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = cv2.Canny(gray, int(config.shape_canny_threshold1), int(config.shape_canny_threshold2))
    if config.shape_edge_dilate_iters > 0:
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=int(config.shape_edge_dilate_iters))

    contours_info = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    max_area = int(config.shape_max_area_px) if config.shape_max_area_px > 0 else int(0.95 * rw * rh)
    max_axis = float(config.shape_max_axis_px) if config.shape_max_axis_px > 0 else float(max(rw, rh))
    candidates: list[HandwheelCandidate] = _hough_ellipse_candidates(
        cv2,
        gray,
        edges,
        (h, w),
        (x0, y0, rw, rh),
        config,
    )

    for contour in contours:
        if contour.shape[0] < int(config.shape_min_contour_points) or contour.shape[0] < 5:
            continue
        area = float(abs(cv2.contourArea(contour)))
        if area < float(config.shape_min_area_px) or area > float(max_area):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        compactness = float(4.0 * np.pi * area / (perimeter * perimeter))
        if compactness < float(config.shape_min_compactness):
            continue
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis_a, axis_b), angle_deg = ellipse
        (cx, cy), (major, minor), angle_deg = _normalize_ellipse((cx, cy), (axis_a, axis_b), angle_deg)
        minor = max(1e-6, minor)
        if major < float(config.shape_min_axis_px) or minor < float(config.shape_min_axis_px):
            continue
        if major > max_axis:
            continue
        aspect = major / minor
        if aspect > float(config.shape_max_aspect_ratio):
            continue
        contour_xy = contour.reshape(-1, 2).astype(float)
        fit_error = _ellipse_fit_error_px(contour_xy, (cx, cy), (major, minor), angle_deg)
        if fit_error > float(config.shape_max_ellipse_fit_error_px):
            continue
        coverage = _angular_coverage(contour_xy, (cx, cy), bins=36)
        if coverage < float(config.shape_min_angular_coverage_ratio):
            continue

        ellipse_band = _ellipse_band_mask(
            (rh, rw),
            (float(cx), float(cy)),
            (float(major), float(minor)),
            float(angle_deg),
            max(1, int(config.shape_mask_contour_thickness_px)),
        )
        candidate_roi = (np.asarray(edges) > 0) & ellipse_band
        candidate_roi = candidate_roi.astype(np.uint8) * 255
        thickness = max(1, int(config.shape_mask_contour_thickness_px))
        cv2.drawContours(candidate_roi, [contour], -1, 255, thickness=thickness)
        candidate_mask = np.zeros((h, w), dtype=np.uint8)
        candidate_mask[y0 : y0 + rh, x0 : x0 + rw] = candidate_roi
        candidate_area = int(np.count_nonzero(candidate_mask))
        if candidate_area < int(config.min_mask_area_px):
            continue
        center_abs = (float(cx + x0), float(cy + y0))
        outer_pixels = _sample_ellipse_perimeter(center_abs, (major, minor), angle_deg, config.ellipse_sample_points)
        outer_mask = external_contour_mask(candidate_mask)
        if outer_pixels.shape[0] > int(config.max_outer_points):
            idx = np.linspace(0, outer_pixels.shape[0] - 1, int(config.max_outer_points)).astype(int)
            outer_pixels = outer_pixels[idx]
        score = _score_shape(aspect, fit_error, coverage, compactness, config)
        candidates.append(
            HandwheelCandidate(
                candidate_mask=candidate_mask,
                outer_contour_mask=outer_mask,
                outer_contour_pixels=outer_pixels,
                roi_xywh=(x0, y0, rw, rh),
                area_px=candidate_area,
                source="shape",
                shape_score=score,
                ellipse=_ellipse_metadata(
                    center_abs,
                    (major, minor),
                    angle_deg,
                    fit_error,
                    aspect,
                    compactness,
                    coverage,
                    area,
                    int(contour.shape[0]),
                    "fit_ellipse",
                ),
                metadata={"edge_area_px": candidate_area, "generator": "fit_ellipse"},
            )
        )

    candidates.sort(key=lambda item: item.shape_score, reverse=True)
    return candidates[: max(1, int(config.shape_max_candidates))]


def candidate_from_hsv_segmentation(segmentation: Any, score: float = 0.5) -> HandwheelCandidate:
    return HandwheelCandidate(
        candidate_mask=np.asarray(segmentation.handwheel_mask, dtype=np.uint8),
        outer_contour_mask=np.asarray(segmentation.outer_contour_mask, dtype=np.uint8),
        outer_contour_pixels=np.asarray(segmentation.outer_contour_pixels, dtype=float).reshape(-1, 2),
        roi_xywh=tuple(int(v) for v in segmentation.roi_xywh),
        area_px=int(segmentation.area_px),
        source="hsv",
        shape_score=float(score),
        ellipse=None,
        metadata={},
    )
