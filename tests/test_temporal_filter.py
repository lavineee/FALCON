from types import SimpleNamespace

import numpy as np

from sim2real.vision.valve_ring_perception.config import ValveRingEstimatorConfig
from sim2real.vision.valve_ring_perception.temporal_filter import TemporalEstimateFilter


def _estimate(center, radius=0.1, axis=(0.0, 0.0, -1.0), valid=True):
    return SimpleNamespace(
        valid=valid,
        center_base=np.asarray(center, dtype=float),
        radius_m=radius,
        axis_base=np.asarray(axis, dtype=float),
        quality=SimpleNamespace(failure_reasons=[]),
    )


def test_temporal_filter_requires_stable_frames_then_latches():
    cfg = ValveRingEstimatorConfig(temporal_stable_frames_required=3)
    filt = TemporalEstimateFilter(cfg)
    first = _estimate([0.0, 0.0, 1.0])

    assert filt.update(first, None, {}, now=0.0).status == "UNSTABLE"
    assert filt.update(first, None, {}, now=0.1).status == "UNSTABLE"
    assert filt.update(first, None, {}, now=0.2).status == "STABLE"

    failed = _estimate([0.0, 0.0, 0.0], valid=False)
    out = filt.update(failed, None, {}, now=0.3)
    assert out.status == "STALE"
    np.testing.assert_allclose(out.estimate.center_base, first.center_base)
