#!/usr/bin/env python3
"""Solve camera_color_optical_frame -> FALCON base extrinsic from tag samples."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict


try:
    from .calibration_utils import (
        dump_yaml_file,
        format_transform_block,
        load_sample_file,
        load_yaml_or_json,
        normalize_base_tag_records,
        normalize_sample_records,
        output_mapping_for_solution,
        solve_base_camera_from_records,
        to_builtin,
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from calibration_utils import (  # type: ignore
        dump_yaml_file,
        format_transform_block,
        load_sample_file,
        load_yaml_or_json,
        normalize_base_tag_records,
        normalize_sample_records,
        output_mapping_for_solution,
        solve_base_camera_from_records,
        to_builtin,
    )


def build_parser() -> argparse.ArgumentParser:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Compute T_base_camera from collected T_camera_tag samples and manually "
            "measured T_base_tag poses."
        )
    )
    parser.add_argument("--samples", required=True, help="Input JSONL/JSON/YAML T_camera_tag samples.")
    parser.add_argument(
        "--base-tag-poses",
        default=os.path.join(script_dir, "config", "base_tag_poses.yaml"),
        help="YAML/JSON file containing manually measured T_base_tag poses.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "output", "T_base_camera.yaml"),
        help="Output YAML path.",
    )
    parser.add_argument(
        "--translation-aggregate",
        choices=("median", "mean"),
        default="median",
        help="How to fuse per-sample T_base_camera translations.",
    )
    parser.add_argument(
        "--max-translation-error-m",
        type=float,
        default=0.05,
        help="Reject samples whose initial translation residual exceeds this value. Use <=0 to disable.",
    )
    parser.add_argument(
        "--max-rotation-error-deg",
        type=float,
        default=5.0,
        help="Reject samples whose initial rotation residual exceeds this value. Use <=0 to disable.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum number of matched/inlier samples required.",
    )
    return parser


def solve_from_files(args: argparse.Namespace) -> Dict[str, Any]:
    raw_samples = load_sample_file(args.samples)
    if len(raw_samples) < int(args.min_samples):
        raise ValueError(
            f"sample count is too small: got {len(raw_samples)}, required at least {args.min_samples}"
        )
    sample_records = normalize_sample_records(raw_samples)

    base_config = load_yaml_or_json(args.base_tag_poses)
    base_tag_records = normalize_base_tag_records(base_config)

    max_translation_error_m = (
        None if args.max_translation_error_m is not None and args.max_translation_error_m <= 0.0 else args.max_translation_error_m
    )
    max_rotation_error_deg = (
        None if args.max_rotation_error_deg is not None and args.max_rotation_error_deg <= 0.0 else args.max_rotation_error_deg
    )

    result = solve_base_camera_from_records(
        sample_records,
        base_tag_records,
        translation_aggregate=args.translation_aggregate,
        max_translation_error_m=max_translation_error_m,
        max_rotation_error_deg=max_rotation_error_deg,
        min_samples=int(args.min_samples),
    )
    output = output_mapping_for_solution(
        result["T_base_camera"],
        result["num_samples_used"],
        result["summary"],
        result["residual_samples"],
    )
    output["solver"] = {
        "samples": os.path.abspath(args.samples),
        "base_tag_poses": os.path.abspath(args.base_tag_poses),
        "translation_aggregate": args.translation_aggregate,
        "max_translation_error_m": max_translation_error_m,
        "max_rotation_error_deg": max_rotation_error_deg,
        "num_samples_matched": result["num_samples_matched"],
        "missing_base_ids": result["missing_base_ids"],
        "outlier_sample_ids": result["outlier_sample_ids"],
    }
    dump_yaml_file(output, args.output)
    return {"result": result, "output": output}


def print_solution(solved: Dict[str, Any], output_path: str) -> None:
    result = solved["result"]
    output = solved["output"]
    print(format_transform_block("T_base_camera", result["T_base_camera"]))
    print()
    print("residual summary:")
    for key, value in result["summary"].items():
        print(f"  {key}: {value:.6g}")
    print()
    print("per-sample residuals:")
    for residual in result["residual_samples"]:
        used = "used" if residual.get("used") else "rejected"
        print(
            "  sample_id={sample_id} {used}: translation_error_m={t:.6g}, "
            "rotation_error_deg={r:.6g}".format(
                sample_id=residual.get("sample_id"),
                used=used,
                t=float(residual["translation_error_m"]),
                r=float(residual["rotation_error_deg"]),
            )
        )
    if result["missing_base_ids"]:
        print(f"missing base-tag poses for sample ids: {result['missing_base_ids']}")
    if result["outlier_sample_ids"]:
        print(f"outlier sample ids rejected: {result['outlier_sample_ids']}")

    print()
    print(f"wrote: {output_path}")
    print()
    print("copy this snippet into valve_vision_g1.yaml if desired:")
    snippet = {"camera_extrinsic": output["camera_extrinsic"]}
    for line in _snippet_lines(to_builtin(snippet)):
        print(line)


def _snippet_lines(mapping: Dict[str, Any], indent: int = 0):
    spaces = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            yield f"{spaces}{key}:"
            yield from _snippet_lines(value, indent + 2)
        elif isinstance(value, list) and value and all(isinstance(item, list) for item in value):
            yield f"{spaces}{key}:"
            for row in value:
                yield f"{spaces}  - {row}"
        else:
            yield f"{spaces}{key}: {value}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        solved = solve_from_files(args)
    except Exception as exc:
        print(f"calibration solve failed: {exc}", file=sys.stderr)
        return 2
    print_solution(solved, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
