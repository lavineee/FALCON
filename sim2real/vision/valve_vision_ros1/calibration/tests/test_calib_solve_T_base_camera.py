#!/usr/bin/env python3
"""Offline self-test for the T_base_camera solver.

Run from this directory or the repository root:
    python3 sim2real/vision/valve_vision_ros1/calibration/tests/test_calib_solve_T_base_camera.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np


CALIB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CALIB_DIR)

from calibration_utils import (  # noqa: E402
    compose_transforms,
    make_transform,
    normalize_base_tag_records,
    normalize_sample_records,
    rotation_error_deg,
    rpy_deg_to_rot,
    solve_base_camera_from_records,
)


def transform_mapping(transform: np.ndarray):
    return {
        "translation_m": transform[:3, 3].tolist(),
        "rotation_matrix": transform[:3, :3].tolist(),
    }


class SolveTBaseCameraTest(unittest.TestCase):
    def _synthetic_problem(self, include_outlier: bool = False):
        t_base_camera = make_transform(rpy_deg_to_rot([6.0, -14.0, 31.0]), [0.18, -0.07, 0.82])
        tag_poses_camera = [
            make_transform(rpy_deg_to_rot([0.0, 0.0, 0.0]), [0.42, -0.04, 0.70]),
            make_transform(rpy_deg_to_rot([3.0, -2.0, 6.0]), [0.55, 0.05, 0.82]),
            make_transform(rpy_deg_to_rot([-4.0, 5.0, -8.0]), [0.47, -0.11, 0.95]),
            make_transform(rpy_deg_to_rot([7.0, 2.0, 11.0]), [0.62, 0.02, 1.05]),
        ]

        raw_samples = []
        raw_base_samples = []
        for sample_id, t_camera_tag in enumerate(tag_poses_camera):
            t_base_tag = compose_transforms(t_base_camera, t_camera_tag)
            raw_samples.append({"sample_id": sample_id, "T_camera_tag": transform_mapping(t_camera_tag)})
            raw_base_samples.append({"sample_id": sample_id, "T_base_tag": transform_mapping(t_base_tag)})

        if include_outlier:
            outlier_id = len(raw_samples)
            t_camera_tag = make_transform(rpy_deg_to_rot([2.0, 1.0, -3.0]), [0.51, 0.07, 0.87])
            t_base_tag = compose_transforms(t_base_camera, t_camera_tag)
            t_base_tag[:3, 3] += np.array([0.35, -0.22, 0.18], dtype=float)
            raw_samples.append({"sample_id": outlier_id, "T_camera_tag": transform_mapping(t_camera_tag)})
            raw_base_samples.append({"sample_id": outlier_id, "T_base_tag": transform_mapping(t_base_tag)})

        return t_base_camera, raw_samples, {"samples": raw_base_samples}

    def test_solver_recovers_known_transform(self):
        expected, raw_samples, raw_base_config = self._synthetic_problem()
        result = solve_base_camera_from_records(
            normalize_sample_records(raw_samples),
            normalize_base_tag_records(raw_base_config),
            translation_aggregate="median",
            max_translation_error_m=0.01,
            max_rotation_error_deg=0.5,
            min_samples=4,
        )
        actual = result["T_base_camera"]
        self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 1e-9)
        self.assertLess(rotation_error_deg(actual[:3, :3], expected[:3, :3]), 1e-9)
        self.assertEqual(result["num_samples_used"], 4)
        self.assertEqual(result["summary"]["num_samples"], 4)

    def test_solver_rejects_translation_outlier(self):
        expected, raw_samples, raw_base_config = self._synthetic_problem(include_outlier=True)
        result = solve_base_camera_from_records(
            normalize_sample_records(raw_samples),
            normalize_base_tag_records(raw_base_config),
            translation_aggregate="median",
            max_translation_error_m=0.05,
            max_rotation_error_deg=1.0,
            min_samples=4,
        )
        actual = result["T_base_camera"]
        self.assertIn(4, result["outlier_sample_ids"])
        self.assertEqual(result["num_samples_used"], 4)
        self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 1e-9)
        self.assertLess(rotation_error_deg(actual[:3, :3], expected[:3, :3]), 1e-9)


if __name__ == "__main__":
    unittest.main()
