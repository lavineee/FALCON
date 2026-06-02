from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .camera_model import CameraIntrinsics
from .circle_fit import project_points_to_plane, ransac_circle_2d, unproject_plane_point
from .config import ValveRingEstimatorConfig
from .depth_validation import (
    compute_contour_depth_jump_ratio,
    compute_plane_residual_stats,
    compute_valid_depth_ratio,
    fit_candidate_plane,
)
from .outer_envelope import extract_outer_envelope_points
from .plane_fit import normalize
from .pointcloud import depth_to_points, transform_points
from .quality import ValveRingQuality, evaluate_quality
from .segmentation import segment_handwheel
from .shape_candidate import HandwheelCandidate, candidate_from_hsv_segmentation, generate_shape_candidates


@dataclass
class CandidateEvaluation:
    candidate: HandwheelCandidate
    quality: ValveRingQuality
    score: float
    center_base: np.ndarray | None = None
    radius_m: float | None = None
    axis_base: np.ndarray | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return bool(self.quality.valid)


@dataclass
class CandidateFusionResult:
    evaluations: list[CandidateEvaluation]
    selected: CandidateEvaluation | None
    failure_reasons: list[str]


def generate_candidates(rgb: np.ndarray, depth: np.ndarray, config: ValveRingEstimatorConfig) -> list[HandwheelCandidate]:
    sources = [str(source).lower() for source in config.candidate_sources]
    candidates: list[HandwheelCandidate] = []
    if "shape" in sources:
        candidates.extend(generate_shape_candidates(rgb, depth, config))
    if "hsv" in sources:
        segmentation = segment_handwheel(rgb, config)
        if segmentation.area_px >= config.min_mask_area_px:
            candidates.append(candidate_from_hsv_segmentation(segmentation))
    return candidates


def _invalid_evaluation(candidate: HandwheelCandidate, reason: str, quality: ValveRingQuality | None = None) -> CandidateEvaluation:
    q = quality or ValveRingQuality(radius_range_m=None)
    q.failure_reasons = list(dict.fromkeys(q.failure_reasons + [reason]))
    q.valid = False
    return CandidateEvaluation(candidate=candidate, quality=q, score=float(q.final_geometry_score))


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _score_inverse(value: float | None, max_good_value: float) -> float:
    if value is None:
        return 0.0
    return _clamp01(1.0 - float(value) / max(1e-9, float(max_good_value)))


def _circle_angular_coverage(
    points_2d: np.ndarray,
    center_2d: np.ndarray,
    num_bins: int,
) -> tuple[float, int, int]:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    bins = max(8, int(num_bins))
    if pts.shape[0] == 0:
        return 0.0, 0, bins
    center = np.asarray(center_2d, dtype=float).reshape(2)
    rel = pts - center
    radii = np.linalg.norm(rel, axis=1)
    valid = np.isfinite(radii) & (radii > 1e-9)
    if not np.any(valid):
        return 0.0, 0, bins
    angles = np.mod(np.arctan2(rel[valid, 1], rel[valid, 0]), 2.0 * np.pi)
    bin_indices = np.floor(angles / (2.0 * np.pi) * bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    occupied = int(np.unique(bin_indices).size)
    return float(occupied / bins), occupied, bins


def _populate_normalized_scores(quality: ValveRingQuality, candidate: HandwheelCandidate, config: ValveRingEstimatorConfig) -> None:
    quality.valid_depth_score = _clamp01(quality.valid_depth_ratio)
    quality.depth_jump_score = _score_inverse(quality.contour_depth_jump_ratio, config.max_contour_depth_jump_ratio)
    plane_rmse_score = _score_inverse(quality.plane_rmse_m, config.max_plane_rmse_m)
    plane_inlier_score = _clamp01(quality.plane_inlier_ratio)
    quality.plane_score = _clamp01(0.6 * plane_rmse_score + 0.4 * plane_inlier_score)
    circle_rmse_score = _score_inverse(quality.circle_rmse_m, config.max_circle_rmse_m)
    circle_inlier_score = _clamp01(quality.circle_inlier_ratio)
    quality.circle_score = _clamp01(0.65 * circle_rmse_score + 0.35 * circle_inlier_score)
    quality.angular_coverage_score = _clamp01(quality.angular_coverage_ratio)
    weights = [
        config.scoring_weight_shape,
        config.scoring_weight_valid_depth,
        config.scoring_weight_depth_jump,
        config.scoring_weight_plane,
        config.scoring_weight_circle,
        config.scoring_weight_coverage,
    ]
    shape_score = _clamp01(candidate.shape_score)
    weighted = (
        config.scoring_weight_shape * shape_score
        + config.scoring_weight_valid_depth * quality.valid_depth_score
        + config.scoring_weight_depth_jump * quality.depth_jump_score
        + config.scoring_weight_plane * quality.plane_score
        + config.scoring_weight_circle * quality.circle_score
        + config.scoring_weight_coverage * quality.angular_coverage_score
    )
    total = max(1e-9, float(sum(weights)))
    quality.final_geometry_score = _clamp01(weighted / total)


def evaluate_candidate_geometry(
    candidate: HandwheelCandidate,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_base_cam: np.ndarray,
    config: ValveRingEstimatorConfig,
    random_seed_offset: int = 0,
) -> CandidateEvaluation:
    quality = ValveRingQuality(
        num_mask_points=0,
        num_outer_points=0,
        envelope_num_points=0,
        radius_range_m=config.radius_range_m,
    )
    if candidate.area_px < config.min_mask_area_px:
        quality.num_mask_points = int(candidate.area_px)
        return _invalid_evaluation(candidate, "candidate_area_too_small", quality)
    quality.image_valid = True
    quality.final_geometry_score = _clamp01(candidate.shape_score)

    depth_m = np.asarray(depth, dtype=float) * float(config.depth_scale_m)
    depth_valid_for_validation = np.where(
        np.isfinite(depth_m)
        & (depth_m >= float(config.depth_min_m))
        & (depth_m <= float(config.depth_max_m)),
        depth_m,
        0.0,
    )
    quality.valid_depth_ratio = compute_valid_depth_ratio(candidate.candidate_mask, depth_valid_for_validation)
    quality.contour_depth_jump_ratio = compute_contour_depth_jump_ratio(
        candidate.outer_contour_pixels,
        depth_valid_for_validation,
        config.depth_jump_threshold_m,
    )
    if config.depth_validation_use_for_hard_gate and quality.valid_depth_ratio < config.hard_min_valid_depth_ratio:
        return _invalid_evaluation(candidate, "valid_depth_ratio_too_low", quality)

    points_cam, pixels = depth_to_points(
        depth,
        intrinsics,
        mask=candidate.candidate_mask,
        depth_scale_m=config.depth_scale_m,
        depth_min_m=config.depth_min_m,
        depth_max_m=config.depth_max_m,
        max_points=config.max_plane_points,
    )
    quality.num_mask_points = int(points_cam.shape[0])
    if points_cam.shape[0] < max(3, int(config.min_mask_points)):
        quality.failure_reasons = ["metric_not_enough_depth_points"]
        return CandidateEvaluation(candidate=candidate, quality=quality, score=quality.final_geometry_score)
    points_base = transform_points(T_base_cam, points_cam)

    try:
        plane = fit_candidate_plane(
            points_base,
            threshold_m=config.plane_inlier_threshold_m,
            iterations=config.plane_ransac_iterations,
            min_inliers=config.min_plane_inliers,
            random_seed=config.random_seed + random_seed_offset,
        )
    except ValueError as exc:
        quality.failure_reasons = [f"metric_plane_fit_failed:{exc}"]
        return CandidateEvaluation(candidate=candidate, quality=quality, score=quality.final_geometry_score)

    quality.plane_inliers = int(np.count_nonzero(plane.inlier_mask))
    residual_stats = compute_plane_residual_stats(points_base, plane)
    quality.plane_rmse_m = float(plane.rmse_m)
    quality.plane_inlier_ratio = float(plane.inlier_ratio)
    handwheel_plane_points_base = points_base[plane.inlier_mask]
    quality.num_outer_points = int(handwheel_plane_points_base.shape[0])
    if handwheel_plane_points_base.shape[0] < 3:
        quality.failure_reasons = ["metric_not_enough_plane_inlier_points"]
        return CandidateEvaluation(candidate=candidate, quality=quality, score=quality.final_geometry_score)

    points_2d, projected_handwheel_base, basis_u, basis_v = project_points_to_plane(
        handwheel_plane_points_base,
        plane.point,
        plane.normal,
    )
    try:
        envelope = extract_outer_envelope_points(
            points_2d,
            angle_bins=config.envelope_angle_bins,
            points_per_bin=config.envelope_points_per_bin,
            top_fraction=config.envelope_top_fraction,
            min_points_per_bin=config.envelope_min_points_per_bin,
            outlier_mad_scale=config.envelope_outlier_mad_scale,
            outlier_min_margin_m=config.envelope_outlier_min_margin_m,
        )
    except ValueError as exc:
        quality.failure_reasons = [f"metric_outer_envelope_failed:{exc}"]
        return CandidateEvaluation(candidate=candidate, quality=quality, score=quality.final_geometry_score)
    quality.envelope_num_points = int(envelope.points_2d.shape[0])
    if envelope.points_2d.shape[0] < 3:
        quality.failure_reasons = ["metric_not_enough_outer_envelope_points"]
        return CandidateEvaluation(candidate=candidate, quality=quality, score=quality.final_geometry_score)

    try:
        circle = ransac_circle_2d(
            envelope.points_2d,
            threshold_m=config.circle_inlier_threshold_m,
            iterations=config.circle_ransac_iterations,
            min_inliers=config.min_circle_inliers,
            radius_range_m=config.radius_range_m,
            random_seed=config.random_seed + 1000 + random_seed_offset,
        )
    except ValueError as exc:
        quality.failure_reasons = [f"metric_circle_fit_failed:{exc}"]
        return CandidateEvaluation(candidate=candidate, quality=quality, score=quality.final_geometry_score)

    center_base = unproject_plane_point(plane.point, basis_u, basis_v, circle.center_2d)
    axis_base = normalize(plane.normal)
    if axis_base is None:
        quality.failure_reasons = ["metric_axis_invalid"]
        return CandidateEvaluation(candidate=candidate, quality=quality, score=quality.final_geometry_score)
    camera_origin_base = T_base_cam[:3, 3]
    if float(np.dot(axis_base, camera_origin_base - center_base)) < 0.0:
        axis_base = -axis_base

    quality.circle_inliers = int(np.count_nonzero(circle.inlier_mask))
    quality.circle_rmse_m = float(circle.rmse_m)
    quality.envelope_circle_rmse_m = float(circle.rmse_m)
    quality.circle_inlier_ratio = float(circle.inlier_ratio)
    quality.radius_m = float(circle.radius_m)
    quality.radius_source = "metric_3d_circle"
    circle_coverage_points = envelope.points_2d[circle.inlier_mask] if np.any(circle.inlier_mask) else envelope.points_2d
    coverage, occupied_bins, total_bins = _circle_angular_coverage(
        circle_coverage_points,
        circle.center_2d,
        config.angular_coverage_bins,
    )
    quality.radius_range_valid = bool(config.radius_range_m[0] <= quality.radius_m <= config.radius_range_m[1])
    quality.angular_coverage_ratio = float(coverage)
    quality.angular_coverage_occupied_bins = int(occupied_bins)
    quality.angular_coverage_total_bins = int(total_bins)
    evaluate_quality(quality, config)
    quality.metric_valid = bool(quality.valid)
    if config.scoring_normalize_scores:
        _populate_normalized_scores(quality, candidate, config)
    score = float(quality.final_geometry_score)
    debug = {
        "mask_pixels": pixels,
        "handwheel_pixels": pixels,
        "plane_point": plane.point,
        "plane_normal": plane.normal,
        "basis_u": basis_u,
        "basis_v": basis_v,
        "e1_base": basis_u,
        "e2_base": basis_v,
        "projected_handwheel_base": projected_handwheel_base,
        "projected_envelope_base": projected_handwheel_base[envelope.selected_indices],
        "projected_rejected_base": projected_handwheel_base[envelope.rejected_indices],
        "outer_envelope_points_2d": envelope.points_2d,
        "outer_envelope_rough_center_2d": envelope.rough_center_2d,
        "outer_envelope_coverage_ratio": envelope.angular_coverage_ratio,
        "outer_envelope_occupied_bins": envelope.occupied_bins,
        "outer_envelope_total_bins": envelope.total_bins,
        "circle_angular_coverage_ratio": quality.angular_coverage_ratio,
        "circle_angular_coverage_occupied_bins": occupied_bins,
        "circle_angular_coverage_total_bins": total_bins,
        "outer_envelope_radius_upper_bound_m": envelope.radius_upper_bound_m,
        "circle_center_2d": circle.center_2d,
        "radius_source": quality.radius_source,
        "image_ellipse": candidate.ellipse,
        "candidate_source": candidate.source,
        "candidate_shape_score": candidate.shape_score,
        "valid_depth_ratio": quality.valid_depth_ratio,
        "contour_depth_jump_ratio": quality.contour_depth_jump_ratio,
        "plane_all_points_rmse_m": float(residual_stats["plane_rmse_m"]),
        "plane_all_points_p95_abs_m": float(residual_stats["plane_p95_abs_m"]),
        "final_geometry_score": score,
    }
    return CandidateEvaluation(
        candidate=candidate,
        quality=quality,
        score=score,
        center_base=center_base,
        radius_m=float(circle.radius_m),
        axis_base=axis_base,
        debug=debug,
    )


def fuse_candidates(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_base_cam: np.ndarray,
    config: ValveRingEstimatorConfig,
) -> CandidateFusionResult:
    candidates = generate_candidates(rgb, depth, config)
    if not candidates:
        return CandidateFusionResult(evaluations=[], selected=None, failure_reasons=["no_candidates"])

    evaluations = [
        evaluate_candidate_geometry(candidate, depth, intrinsics, T_base_cam, config, random_seed_offset=i * 17)
        for i, candidate in enumerate(candidates)
    ]
    valid = [item for item in evaluations if item.quality.metric_valid]
    if valid:
        selected = max(valid, key=lambda item: item.score)
        return CandidateFusionResult(evaluations=evaluations, selected=selected, failure_reasons=[])
    image_valid = [item for item in evaluations if item.quality.image_valid]
    reasons: list[str] = []
    for item in evaluations:
        reasons.extend(item.quality.failure_reasons)
    if image_valid:
        selected = max(image_valid, key=lambda item: item.score)
        return CandidateFusionResult(
            evaluations=evaluations,
            selected=selected,
            failure_reasons=list(dict.fromkeys(reasons or ["no_metric_valid_candidates"])),
        )
    selected = max(evaluations, key=lambda item: item.score)
    return CandidateFusionResult(
        evaluations=evaluations,
        selected=None,
        failure_reasons=list(dict.fromkeys(reasons or ["no_valid_candidates"])),
    )


def candidate_summaries(evaluations: list[CandidateEvaluation], selected: CandidateEvaluation | None) -> list[dict[str, Any]]:
    selected_id = id(selected) if selected is not None else None
    summaries: list[dict[str, Any]] = []
    for index, evaluation in enumerate(evaluations):
        candidate = evaluation.candidate
        summaries.append(
            {
                "index": index,
                "source": candidate.source,
                "generator": candidate.metadata.get("generator"),
                "selected": id(evaluation) == selected_id,
                "score": float(evaluation.score),
                "valid": bool(evaluation.valid),
                "image_valid": bool(evaluation.quality.image_valid),
                "metric_valid": bool(evaluation.quality.metric_valid),
                "ellipse": candidate.ellipse,
                "shape_score": float(candidate.shape_score),
                "valid_depth_ratio": float(evaluation.quality.valid_depth_ratio),
                "contour_depth_jump_ratio": float(evaluation.quality.contour_depth_jump_ratio),
                "plane_rmse_m": None if evaluation.quality.plane_rmse_m is None else float(evaluation.quality.plane_rmse_m),
                "plane_inlier_ratio": float(evaluation.quality.plane_inlier_ratio),
                "circle_rmse_m": None if evaluation.quality.circle_rmse_m is None else float(evaluation.quality.circle_rmse_m),
                "circle_inlier_ratio": float(evaluation.quality.circle_inlier_ratio),
                "angular_coverage_ratio": float(evaluation.quality.angular_coverage_ratio),
                "angular_coverage_occupied_bins": int(evaluation.quality.angular_coverage_occupied_bins),
                "angular_coverage_total_bins": int(evaluation.quality.angular_coverage_total_bins),
                "final_geometry_score": float(evaluation.quality.final_geometry_score),
                "radius_m": None if evaluation.quality.radius_m is None else float(evaluation.quality.radius_m),
                "radius_source": evaluation.quality.radius_source,
                "area_px": int(candidate.area_px),
                "outer_contour_pixels": candidate.outer_contour_pixels,
                "failure_reasons": list(evaluation.quality.failure_reasons),
            }
        )
    return summaries
