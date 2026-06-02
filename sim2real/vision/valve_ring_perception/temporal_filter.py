from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import copy
import math
import time

import numpy as np

from .config import ValveRingEstimatorConfig


@dataclass
class TemporalFilterOutput:
    estimate: Any
    segmentation: Any
    debug: dict[str, Any]
    status: str
    accepted: bool
    stale: bool


class TemporalEstimateFilter:
    def __init__(self, config: ValveRingEstimatorConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.last_accepted_estimate: Any | None = None
        self.last_accepted_segmentation: Any | None = None
        self.last_accepted_debug: dict[str, Any] | None = None
        self.last_accepted_time: float | None = None
        self.pending_estimate: Any | None = None
        self.pending_count = 0

    def latch_current(self, estimate: Any, segmentation: Any, debug: dict[str, Any], now: float | None = None) -> bool:
        if not bool(getattr(estimate, "valid", False)):
            return False
        self.last_accepted_estimate = copy.deepcopy(estimate)
        self.last_accepted_segmentation = segmentation
        self.last_accepted_debug = dict(debug)
        self.last_accepted_time = time.time() if now is None else float(now)
        self.pending_estimate = copy.deepcopy(estimate)
        self.pending_count = max(1, int(self.config.temporal_stable_frames_required))
        return True

    @staticmethod
    def _axis_angle_deg(a: Any, b: Any) -> float:
        va = np.asarray(a, dtype=float).reshape(3)
        vb = np.asarray(b, dtype=float).reshape(3)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na < 1e-9 or nb < 1e-9:
            return float("inf")
        va = va / na
        vb = vb / nb
        # Axis sign is physically ambiguous, so compare the smaller signed angle.
        dot = abs(float(np.dot(va, vb)))
        return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))

    def _is_close(self, estimate: Any, reference: Any) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if estimate.center_base is None or reference.center_base is None:
            reasons.append("center_missing")
        else:
            center_jump = float(np.linalg.norm(np.asarray(estimate.center_base) - np.asarray(reference.center_base)))
            if center_jump > self.config.temporal_max_center_jump_m:
                reasons.append(f"center_jump:{center_jump:.3f}")
        if estimate.radius_m is None or reference.radius_m is None:
            reasons.append("radius_missing")
        else:
            radius_jump = abs(float(estimate.radius_m) - float(reference.radius_m))
            if radius_jump > self.config.temporal_max_radius_jump_m:
                reasons.append(f"radius_jump:{radius_jump:.3f}")
        if estimate.axis_base is None or reference.axis_base is None:
            reasons.append("axis_missing")
        else:
            axis_jump = self._axis_angle_deg(estimate.axis_base, reference.axis_base)
            if axis_jump > self.config.temporal_max_axis_jump_deg:
                reasons.append(f"axis_jump:{axis_jump:.1f}")
        return len(reasons) == 0, reasons

    def _annotate(self, debug: dict[str, Any], status: str, accepted: bool, reasons: list[str] | None = None) -> dict[str, Any]:
        out = dict(debug)
        out["temporal_status"] = status
        out["temporal_accepted"] = bool(accepted)
        out["temporal_pending_count"] = int(self.pending_count)
        if reasons:
            out["temporal_reasons"] = reasons
        return out

    def update(self, estimate: Any, segmentation: Any, debug: dict[str, Any], now: float | None = None) -> TemporalFilterOutput:
        if not self.config.temporal_filter_enabled:
            status = "STABLE" if bool(estimate.valid) else "UNSTABLE"
            return TemporalFilterOutput(estimate, segmentation, self._annotate(debug, status, bool(estimate.valid)), status, bool(estimate.valid), False)

        timestamp = time.time() if now is None else float(now)
        if not bool(estimate.valid):
            if (
                self.last_accepted_estimate is not None
                and self.last_accepted_time is not None
                and timestamp - self.last_accepted_time <= self.config.temporal_latch_last_valid_sec
            ):
                stale_debug = self._annotate(self.last_accepted_debug or {}, "STALE", False, list(estimate.quality.failure_reasons))
                return TemporalFilterOutput(
                    copy.deepcopy(self.last_accepted_estimate),
                    self.last_accepted_segmentation,
                    stale_debug,
                    "STALE",
                    False,
                    True,
                )
            self.pending_count = 0
            self.pending_estimate = None
            return TemporalFilterOutput(estimate, segmentation, self._annotate(debug, "UNSTABLE", False, list(estimate.quality.failure_reasons)), "UNSTABLE", False, False)

        reference = self.last_accepted_estimate if self.last_accepted_estimate is not None else self.pending_estimate
        close = True
        reasons: list[str] = []
        if reference is not None:
            close, reasons = self._is_close(estimate, reference)
        if close:
            self.pending_count += 1
        else:
            self.pending_count = 1
        self.pending_estimate = copy.deepcopy(estimate)

        if self.pending_count >= max(1, int(self.config.temporal_stable_frames_required)):
            self.last_accepted_estimate = copy.deepcopy(estimate)
            self.last_accepted_segmentation = segmentation
            self.last_accepted_debug = dict(debug)
            self.last_accepted_time = timestamp
            stable_debug = self._annotate(debug, "STABLE", True)
            return TemporalFilterOutput(estimate, segmentation, stable_debug, "STABLE", True, False)

        unstable_debug = self._annotate(debug, "UNSTABLE", False, reasons)
        return TemporalFilterOutput(estimate, segmentation, unstable_debug, "UNSTABLE", False, False)
