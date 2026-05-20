#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.utils.hand_controller import (
    DEFAULT_INSPIRE_CLOSE_RAW,
    DEFAULT_INSPIRE_OPEN_RAW,
    create_hand_controller,
    inspire_right_hand_q12,
    normalize_inspire_q,
)


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return REPO_ROOT / path


def _load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file) or {}


def load_config(config_path, overlay_paths):
    config = _load_yaml(_resolve_path(config_path))
    for overlay_path in overlay_paths:
        config.update(_load_yaml(_resolve_path(overlay_path)))
    return config


def make_mock_writer(path):
    def writer(payload):
        control_path = Path(path)
        if control_path.parent:
            control_path.parent.mkdir(parents=True, exist_ok=True)
        merged = {}
        if control_path.exists():
            try:
                with open(control_path, "r") as file:
                    merged = json.load(file)
            except Exception:
                merged = {}
        merged.update(payload)
        tmp_path = control_path.with_suffix(control_path.suffix + ".tmp")
        with open(tmp_path, "w") as file:
            json.dump(merged, file)
        os.replace(tmp_path, control_path)

    return writer


def print_q12(label, q12):
    formatted = ", ".join(f"{value:.3f}" for value in q12)
    print(f"{label}: [{formatted}]")


def main():
    parser = argparse.ArgumentParser(description="Right-hand RH56DF3 Inspire DDS open/close test")
    parser.add_argument(
        "--config",
        default="sim2real/config/g1/g1_29dof_with_hand_valve_grasp_contact_overlay.yaml",
        help="YAML config or merged config path.",
    )
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Additional YAML overlay. Can be passed multiple times.",
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "inspire_dds"),
        default=None,
        help="Override hand_backend from config.",
    )
    parser.add_argument("--inspire_dds_channel", type=int, default=None)
    parser.add_argument("--inspire_network_interface", default=None)
    parser.add_argument("--pause_sec", type=float, default=2.0)
    parser.add_argument("--final_wait_sec", type=float, default=1.0)
    parser.add_argument(
        "--sim_control_file",
        default="/tmp/falcon_inspire_hand_mock_control.json",
        help="Mock backend output path.",
    )
    parser.add_argument(
        "--no_initialize_dds",
        action="store_true",
        help="Skip ChannelFactoryInitialize for already-initialized DDS processes.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overlay)
    if args.backend is not None:
        config["hand_backend"] = args.backend
    if args.inspire_dds_channel is not None:
        config["inspire_dds_channel"] = args.inspire_dds_channel
    if args.inspire_network_interface is not None:
        config["inspire_network_interface"] = args.inspire_network_interface or None

    open_q = normalize_inspire_q(
        config.get("inspire_open_raw", DEFAULT_INSPIRE_OPEN_RAW),
        DEFAULT_INSPIRE_OPEN_RAW,
        "inspire_open_raw",
    )
    close_q = normalize_inspire_q(
        config.get("inspire_close_raw", DEFAULT_INSPIRE_CLOSE_RAW),
        DEFAULT_INSPIRE_CLOSE_RAW,
        "inspire_close_raw",
    )
    open_q12 = inspire_right_hand_q12(open_q)
    close_q12 = inspire_right_hand_q12(close_q)

    backend = str(config.get("hand_backend", "mock")).strip().lower()
    print(f"hand_backend={backend}")
    print("right hand order=[pinky, ring, middle, index, thumb_bend, thumb_rotation]")
    print_q12("open q12 target", open_q12)
    print_q12("close q12 target", close_q12)

    controller = create_hand_controller(
        config,
        sim_control_writer=make_mock_writer(args.sim_control_file),
        extra_payload_fn=lambda: {"source": "sim2real/tools/test_inspire_hand_open_close.py"},
        initialize_dds=(backend == "inspire_dds" and not args.no_initialize_dds),
    )

    completed = False
    try:
        print("command: open")
        controller.open(force=True)
        print_q12("sent open q12", getattr(controller, "last_command_q12", open_q12) or open_q12)
        time.sleep(args.pause_sec)

        print("command: close")
        controller.close(force=True)
        print_q12("sent close q12", getattr(controller, "last_command_q12", close_q12) or close_q12)
        time.sleep(args.pause_sec)

        print("command: open")
        controller.open(force=True)
        print_q12("sent open q12", getattr(controller, "last_command_q12", open_q12) or open_q12)
        time.sleep(max(0.0, args.final_wait_sec))
        completed = True
    finally:
        if not completed:
            try:
                controller.open(force=True)
                time.sleep(0.1)
            except Exception as exc:
                print(f"failed to send safety open: {exc}", file=sys.stderr)
        controller.shutdown()


if __name__ == "__main__":
    main()
