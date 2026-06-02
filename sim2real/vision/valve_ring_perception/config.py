from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import json


HSVRange = tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass
class ValveRingEstimatorConfig:
    """Tunable parameters for the offline valve ring estimator."""

    roi_xywh: tuple[int, int, int, int] | None = None
    candidate_sources: list[str] = field(default_factory=lambda: ["shape", "hsv"])
    hsv_ranges: list[HSVRange] = field(
        default_factory=lambda: [
            ((0, 60, 40), (15, 255, 255)),
            ((165, 60, 40), (179, 255, 255)),
        ]
    )
    depth_scale_m: float = 1.0
    depth_min_m: float = 0.15
    depth_max_m: float = 4.0
    min_valid_depth_ratio: float = 0.40
    max_contour_depth_jump_ratio: float = 0.35
    depth_jump_threshold_m: float = 0.05
    depth_validation_enabled: bool = False
    depth_validation_use_for_candidate_selection: bool = False
    depth_validation_use_for_hard_gate: bool = False
    hard_min_valid_depth_ratio: float = 0.25
    hard_max_circle_rmse_m: float = 0.05
    hard_min_angular_coverage: float = 0.25
    scoring_normalize_scores: bool = True
    scoring_weight_shape: float = 0.20
    scoring_weight_valid_depth: float = 0.10
    scoring_weight_depth_jump: float = 0.05
    scoring_weight_plane: float = 0.15
    scoring_weight_circle: float = 0.30
    scoring_weight_coverage: float = 0.20
    temporal_filter_enabled: bool = True
    temporal_max_center_jump_m: float = 0.05
    temporal_max_radius_jump_m: float = 0.02
    temporal_max_axis_jump_deg: float = 15.0
    temporal_stable_frames_required: int = 3
    temporal_latch_last_valid_sec: float = 0.5
    mask_open_iters: int = 1
    mask_close_iters: int = 1
    mask_kernel_size: int = 3
    keep_largest_component: bool = True
    min_mask_area_px: int = 80
    max_plane_points: int = 6000
    max_outer_points: int = 2500
    envelope_angle_bins: int = 72
    angular_coverage_bins: int = 180
    envelope_points_per_bin: int = 3
    envelope_top_fraction: float = 0.20
    envelope_min_points_per_bin: int = 1
    envelope_outlier_mad_scale: float = 6.0
    envelope_outlier_min_margin_m: float = 0.03
    min_envelope_points: int = 40
    min_angular_coverage_ratio: float = 0.35
    shape_use_clahe: bool = True
    shape_clahe_clip_limit: float = 2.0
    shape_clahe_tile_grid_size: int = 8
    shape_blur_kernel_size: int = 5
    shape_canny_threshold1: int = 50
    shape_canny_threshold2: int = 150
    shape_edge_dilate_iters: int = 1
    shape_mask_contour_thickness_px: int = 5
    shape_hough_enabled: bool = True
    shape_hough_dp: float = 1.2
    shape_hough_min_dist_px: float = 80.0
    shape_hough_param2: float = 24.0
    shape_hough_band_thickness_px: int = 12
    shape_hough_max_candidates: int = 8
    shape_min_area_px: int = 400
    shape_max_area_px: int = 0
    shape_min_axis_px: float = 20.0
    shape_max_axis_px: float = 0.0
    shape_max_aspect_ratio: float = 2.8
    shape_min_contour_points: int = 20
    shape_max_ellipse_fit_error_px: float = 8.0
    shape_min_compactness: float = 0.05
    shape_min_angular_coverage_ratio: float = 0.45
    shape_max_candidates: int = 8
    ellipse_band_thickness_px: int = 7
    ellipse_sample_points: int = 720
    plane_ransac_iterations: int = 250
    plane_inlier_threshold_m: float = 0.01
    circle_ransac_iterations: int = 500
    circle_inlier_threshold_m: float = 0.015
    min_mask_points: int = 80
    min_outer_points: int = 40
    min_plane_inliers: int = 60
    min_circle_inliers: int = 24
    max_plane_rmse_m: float = 0.012
    max_circle_rmse_m: float = 0.015
    min_plane_inlier_ratio: float = 0.50
    min_circle_inlier_ratio: float = 0.45
    radius_range_m: tuple[float, float] = (0.03, 0.60)
    random_seed: int = 13

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ValveRingEstimatorConfig":
        if not data:
            return cls()
        data = dict(data)
        depth_validation = data.pop("depth_validation", None) or {}
        if depth_validation:
            mapping = {
                "enabled": "depth_validation_enabled",
                "use_for_candidate_selection": "depth_validation_use_for_candidate_selection",
                "use_for_hard_gate": "depth_validation_use_for_hard_gate",
                "min_valid_depth_ratio": "min_valid_depth_ratio",
                "max_contour_depth_jump_ratio": "max_contour_depth_jump_ratio",
                "depth_jump_threshold_m": "depth_jump_threshold_m",
            }
            for key, target in mapping.items():
                if key in depth_validation:
                    data[target] = depth_validation[key]
        geometry_quality = data.pop("geometry_quality", None) or {}
        if geometry_quality:
            mapping = {
                "max_plane_rmse_m": "max_plane_rmse_m",
                "min_plane_inlier_ratio": "min_plane_inlier_ratio",
                "max_circle_rmse_m": "max_circle_rmse_m",
                "min_circle_inlier_ratio": "min_circle_inlier_ratio",
                "min_angular_coverage": "min_angular_coverage_ratio",
                "min_angular_coverage_ratio": "min_angular_coverage_ratio",
            }
            for key, target in mapping.items():
                if key in geometry_quality:
                    data[target] = geometry_quality[key]
        hard_gates = data.pop("hard_gates", None) or {}
        if hard_gates:
            mapping = {
                "min_valid_depth_ratio": "hard_min_valid_depth_ratio",
                "hard_min_valid_depth_ratio": "hard_min_valid_depth_ratio",
                "hard_max_circle_rmse_m": "hard_max_circle_rmse_m",
                "hard_min_angular_coverage": "hard_min_angular_coverage",
            }
            for key, target in mapping.items():
                if key in hard_gates:
                    data[target] = hard_gates[key]
        scoring = data.pop("scoring", None) or {}
        if scoring:
            if "normalize_scores" in scoring:
                data["scoring_normalize_scores"] = scoring["normalize_scores"]
            weights = scoring.get("weights", {}) or {}
            mapping = {
                "shape": "scoring_weight_shape",
                "valid_depth": "scoring_weight_valid_depth",
                "depth_jump": "scoring_weight_depth_jump",
                "plane": "scoring_weight_plane",
                "circle": "scoring_weight_circle",
                "coverage": "scoring_weight_coverage",
            }
            for key, target in mapping.items():
                if key in weights:
                    data[target] = weights[key]
        temporal = data.pop("temporal_filter", None) or {}
        if temporal:
            mapping = {
                "enabled": "temporal_filter_enabled",
                "max_center_jump_m": "temporal_max_center_jump_m",
                "max_radius_jump_m": "temporal_max_radius_jump_m",
                "max_axis_jump_deg": "temporal_max_axis_jump_deg",
                "stable_frames_required": "temporal_stable_frames_required",
                "latch_last_valid_sec": "temporal_latch_last_valid_sec",
            }
            for key, target in mapping.items():
                if key in temporal:
                    data[target] = temporal[key]
        names = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in names}
        cfg = cls(**kwargs)
        if cfg.roi_xywh is not None:
            cfg.roi_xywh = tuple(int(v) for v in cfg.roi_xywh)  # type: ignore[assignment]
        cfg.candidate_sources = [str(v).lower() for v in cfg.candidate_sources]
        cfg.radius_range_m = tuple(float(v) for v in cfg.radius_range_m)  # type: ignore[assignment]
        cfg.hsv_ranges = [
            (tuple(int(v) for v in lower), tuple(int(v) for v in upper))  # type: ignore[list-item]
            for lower, upper in cfg.hsv_ranges
        ]
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "ValveRingEstimatorConfig":
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("YAML config requires PyYAML; use JSON or install pyyaml") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data or {})

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for item in fields(self):
            data[item.name] = getattr(self, item.name)
        return data
