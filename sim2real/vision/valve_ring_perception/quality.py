from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ValveRingEstimatorConfig


@dataclass
class ValveRingQuality:
    image_valid: bool = False
    metric_valid: bool = False
    num_mask_points: int = 0
    num_outer_points: int = 0
    valid_depth_ratio: float = 0.0
    contour_depth_jump_ratio: float = 1.0
    valid_depth_score: float = 0.0
    depth_jump_score: float = 0.0
    plane_score: float = 0.0
    circle_score: float = 0.0
    angular_coverage_score: float = 0.0
    final_geometry_score: float = 0.0
    envelope_num_points: int = 0
    angular_coverage_ratio: float = 0.0
    angular_coverage_occupied_bins: int = 0
    angular_coverage_total_bins: int = 0
    plane_inliers: int = 0
    circle_inliers: int = 0
    plane_rmse_m: float | None = None
    circle_rmse_m: float | None = None
    envelope_circle_rmse_m: float | None = None
    plane_inlier_ratio: float = 0.0
    circle_inlier_ratio: float = 0.0
    radius_m: float | None = None
    radius_source: str | None = None
    radius_range_m: tuple[float, float] | None = None
    radius_range_valid: bool = False
    valid: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "image_valid": bool(self.image_valid),
            "metric_valid": bool(self.metric_valid),
            "failure_reasons": list(self.failure_reasons),
            "num_mask_points": int(self.num_mask_points),
            "num_outer_points": int(self.num_outer_points),
            "valid_depth_ratio": float(self.valid_depth_ratio),
            "contour_depth_jump_ratio": float(self.contour_depth_jump_ratio),
            "valid_depth_score": float(self.valid_depth_score),
            "depth_jump_score": float(self.depth_jump_score),
            "plane_score": float(self.plane_score),
            "circle_score": float(self.circle_score),
            "angular_coverage_score": float(self.angular_coverage_score),
            "final_geometry_score": float(self.final_geometry_score),
            "envelope_num_points": int(self.envelope_num_points),
            "angular_coverage_ratio": float(self.angular_coverage_ratio),
            "angular_coverage_occupied_bins": int(self.angular_coverage_occupied_bins),
            "angular_coverage_total_bins": int(self.angular_coverage_total_bins),
            "plane_inliers": int(self.plane_inliers),
            "circle_inliers": int(self.circle_inliers),
            "plane_rmse_m": None if self.plane_rmse_m is None else float(self.plane_rmse_m),
            "circle_rmse_m": None if self.circle_rmse_m is None else float(self.circle_rmse_m),
            "envelope_circle_rmse_m": None if self.envelope_circle_rmse_m is None else float(self.envelope_circle_rmse_m),
            "plane_inlier_ratio": float(self.plane_inlier_ratio),
            "circle_inlier_ratio": float(self.circle_inlier_ratio),
            "radius_m": None if self.radius_m is None else float(self.radius_m),
            "radius_source": self.radius_source,
            "radius_range_m": None if self.radius_range_m is None else [float(v) for v in self.radius_range_m],
            "radius_range_valid": bool(self.radius_range_valid),
        }


def evaluate_quality(quality: ValveRingQuality, config: ValveRingEstimatorConfig) -> ValveRingQuality:
    reasons: list[str] = []
    if quality.num_mask_points < config.min_mask_points:
        reasons.append("too_few_mask_depth_points")
    if quality.valid_depth_ratio < config.hard_min_valid_depth_ratio:
        reasons.append("valid_depth_ratio_too_low")
    if quality.num_outer_points < config.min_outer_points:
        reasons.append("too_few_outer_depth_points")
    if quality.envelope_num_points < config.min_envelope_points:
        reasons.append("too_few_envelope_points")
    if quality.angular_coverage_ratio < config.hard_min_angular_coverage:
        reasons.append("angular_coverage_too_low")
    if quality.circle_inliers < config.min_circle_inliers:
        reasons.append("too_few_circle_inliers")
    if quality.circle_rmse_m is None or quality.circle_rmse_m > config.hard_max_circle_rmse_m:
        reasons.append("circle_rmse_too_high")
    if quality.radius_m is None:
        reasons.append("radius_missing")
        quality.radius_range_valid = False
    else:
        rmin, rmax = config.radius_range_m
        quality.radius_range_valid = bool(float(rmin) <= quality.radius_m <= float(rmax))
        if not quality.radius_range_valid:
            reasons.append("radius_out_of_range")
    quality.radius_range_m = config.radius_range_m
    quality.failure_reasons = reasons
    quality.metric_valid = len(reasons) == 0
    quality.valid = bool(quality.metric_valid)
    return quality
