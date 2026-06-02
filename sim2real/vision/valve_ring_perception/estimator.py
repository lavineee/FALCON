from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import time

import numpy as np

from .camera_model import CameraIntrinsics, coerce_intrinsics
from .candidate_fusion import candidate_summaries, fuse_candidates
from .config import ValveRingEstimatorConfig
from .quality import ValveRingQuality
from .shape_candidate import HandwheelCandidate
from .status_writer import atomic_write_json
from .visualization import draw_debug_overlay


@dataclass
class ValveRingEstimate:
    valid: bool
    center_base: np.ndarray | None
    radius_m: float | None
    axis_base: np.ndarray | None
    quality: ValveRingQuality
    frame: str = "base"
    source: str = "valve_ring_perception_offline"
    timestamp: float = field(default_factory=time.time)
    image_timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "image_timestamp": None if self.image_timestamp is None else float(self.image_timestamp),
            "frame": self.frame,
            "source": self.source,
            "valid": bool(self.valid),
            "center_base": None if self.center_base is None else [float(v) for v in self.center_base],
            "radius_m": None if self.radius_m is None else float(self.radius_m),
            "axis_base": None if self.axis_base is None else [float(v) for v in self.axis_base],
            "quality": self.quality.to_dict(),
        }


@dataclass
class EstimationResult:
    estimate: ValveRingEstimate
    segmentation: HandwheelCandidate | None
    debug: dict[str, Any]


class ValveRingEstimator:
    def __init__(self, config: ValveRingEstimatorConfig | None = None):
        self.config = config or ValveRingEstimatorConfig()

    def _invalid(
        self,
        reason: str,
        quality: ValveRingQuality | None = None,
        segmentation: HandwheelCandidate | None = None,
        debug: dict[str, Any] | None = None,
        image_timestamp: float | None = None,
    ) -> EstimationResult:
        q = quality or ValveRingQuality()
        q.failure_reasons = list(dict.fromkeys(q.failure_reasons + [reason]))
        q.valid = False
        return EstimationResult(
            estimate=ValveRingEstimate(
                valid=False,
                center_base=None,
                radius_m=None,
                axis_base=None,
                quality=q,
                image_timestamp=image_timestamp,
            ),
            segmentation=segmentation,
            debug=debug or {},
        )

    def estimate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics | dict[str, Any] | np.ndarray,
        T_base_cam: np.ndarray,
        image_timestamp: float | None = None,
    ) -> ValveRingEstimate:
        return self.estimate_with_debug(rgb, depth, intrinsics, T_base_cam, image_timestamp).estimate

    def estimate_with_debug(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics | dict[str, Any] | np.ndarray,
        T_base_cam: np.ndarray,
        image_timestamp: float | None = None,
    ) -> EstimationResult:
        cfg = self.config
        camera = coerce_intrinsics(intrinsics)
        transform = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
        fusion = fuse_candidates(np.asarray(rgb), np.asarray(depth), camera, transform, cfg)
        if fusion.selected is None:
            quality = ValveRingQuality(radius_range_m=cfg.radius_range_m)
            quality.failure_reasons = fusion.failure_reasons
            quality.valid = False
            debug = {
                "candidate_summaries": candidate_summaries(fusion.evaluations, None),
                "failure_reasons": fusion.failure_reasons,
            }
            return self._invalid("no_valid_candidate", quality, None, debug, image_timestamp=image_timestamp)

        selected = fusion.selected
        quality = selected.quality
        estimate = ValveRingEstimate(
            valid=quality.valid,
            center_base=selected.center_base,
            radius_m=None if selected.radius_m is None else float(selected.radius_m),
            axis_base=selected.axis_base,
            quality=quality,
            image_timestamp=image_timestamp,
        )
        debug = dict(selected.debug)
        debug["candidate_summaries"] = candidate_summaries(fusion.evaluations, selected)
        debug["selected_candidate_source"] = selected.candidate.source
        debug["selected_candidate_score"] = selected.score
        return EstimationResult(estimate=estimate, segmentation=selected.candidate, debug=debug)

    def estimate_and_write(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics | dict[str, Any] | np.ndarray,
        T_base_cam: np.ndarray,
        result_json_path: str,
        debug_overlay_path: str | None = None,
        image_timestamp: float | None = None,
    ) -> ValveRingEstimate:
        result = self.estimate_with_debug(rgb, depth, intrinsics, T_base_cam, image_timestamp)
        atomic_write_json(result_json_path, result.estimate.to_dict())
        if debug_overlay_path is not None and result.segmentation is not None:
            draw_debug_overlay(
                rgb,
                result.segmentation,
                result.estimate,
                result.debug,
                coerce_intrinsics(intrinsics),
                np.asarray(T_base_cam, dtype=float).reshape(4, 4),
                debug_overlay_path,
            )
        return result.estimate
