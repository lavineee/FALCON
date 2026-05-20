#!/usr/bin/env python3
"""Receive UDP valve target JSON and optionally atomically mirror it to disk."""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

import argparse
import json
import socket
import tempfile
import time


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".valve_target_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(payload, file, ensure_ascii=True, allow_nan=False, sort_keys=True)
            file.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _format_target(payload):
    if not payload.get("valid", False):
        return f"invalid reason={payload.get('reason', '')}"
    center = payload.get("center_base")
    grasp = payload.get("grasp_base")
    axis = payload.get("axis_base")
    return f"center_base={center} grasp_base={grasp} axis_base={axis}"


def main():
    parser = argparse.ArgumentParser(description="UDP valve target JSON receiver")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument(
        "--output-file",
        default="",
        help="If set, atomically writes the latest JSON for RealVisionGraspGeometryProvider.",
    )
    parser.add_argument(
        "--write-valid-only",
        action="store_true",
        help="Print all packets but only mirror valid targets to --output-file.",
    )
    parser.add_argument("--once", action="store_true", help="Receive one packet and exit.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-packet printing.")
    parser.add_argument("--recv-buffer", type=int, default=65535)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, args.port))
    print(f"[valve_target_receiver] listening on {args.bind_host}:{args.port}")
    if args.output_file:
        print(f"[valve_target_receiver] mirroring latest target to {args.output_file}")

    while True:
        data, addr = sock.recvfrom(args.recv_buffer)
        received_timestamp = time.time()
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception as exc:
            print(f"[valve_target_receiver] bad JSON from {addr}: {exc}")
            continue
        if not isinstance(payload, dict):
            print(f"[valve_target_receiver] non-object JSON from {addr}")
            continue

        payload["receiver_timestamp"] = received_timestamp
        payload["receiver_addr"] = f"{addr[0]}:{addr[1]}"
        if not args.quiet:
            print(
                f"[valve_target_receiver] {time.strftime('%H:%M:%S')} "
                f"from={payload['receiver_addr']} {_format_target(payload)}"
            )
        if args.output_file and (not args.write_valid_only or bool(payload.get("valid", False))):
            _atomic_write_json(args.output_file, payload)
        if args.once:
            break


if __name__ == "__main__":
    main()
