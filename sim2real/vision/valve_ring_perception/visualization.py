from __future__ import annotations

import os
import struct
import zlib
from typing import Any

import numpy as np

from .camera_model import CameraIntrinsics, project_points
from .circle_fit import plane_basis
from .pointcloud import invert_transform, transform_points


_FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    ".": ["000", "000", "000", "000", "000", "110", "110"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    ":": ["000", "110", "110", "000", "110", "110", "000"],
    "/": ["001", "001", "010", "010", "100", "100", "000"],
    "%": ["1001", "0010", "0010", "0100", "0100", "1001", "0000"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "100", "100"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["111", "100", "100", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["111", "100", "100", "101", "101", "101", "111"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["111", "001", "001", "001", "101", "101", "111"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["1001", "1111", "1111", "1001", "1001", "1001", "1001"],
    "N": ["1001", "1101", "1101", "1011", "1011", "1001", "1001"],
    "O": ["111", "101", "101", "101", "101", "101", "111"],
    "P": ["110", "101", "101", "110", "100", "100", "100"],
    "R": ["110", "101", "101", "110", "101", "101", "101"],
    "S": ["111", "100", "100", "111", "001", "001", "111"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "Z": ["111", "001", "010", "010", "100", "100", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
}


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def save_png_rgb(path: str, rgb: np.ndarray) -> None:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG writer expects HxWx3 RGB uint8 image")
    h, w = image.shape[:2]
    rows = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    payload += _chunk(b"IDAT", zlib.compress(rows, level=6))
    payload += _chunk(b"IEND", b"")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as file:
        file.write(payload)
    os.replace(tmp_path, path)


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> None:
    idx = np.asarray(mask) > 0
    if np.any(idx):
        image[idx] = np.clip((1.0 - alpha) * image[idx] + alpha * np.asarray(color), 0, 255).astype(np.uint8)


def _draw_point(image: np.ndarray, x: int, y: int, color: tuple[int, int, int], radius: int = 2) -> None:
    h, w = image.shape[:2]
    for yy in range(max(0, y - radius), min(h, y + radius + 1)):
        for xx in range(max(0, x - radius), min(w, x + radius + 1)):
            if (xx - x) * (xx - x) + (yy - y) * (yy - y) <= radius * radius:
                image[yy, xx] = color


def _draw_line(image: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], color: tuple[int, int, int]) -> None:
    x0, y0 = (int(round(v)) for v in p0)
    x1, y1 = (int(round(v)) for v in p1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    h, w = image.shape[:2]
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_rect(image: np.ndarray, roi: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, w, h = roi
    _draw_line(image, (x, y), (x + w - 1, y), color)
    _draw_line(image, (x + w - 1, y), (x + w - 1, y + h - 1), color)
    _draw_line(image, (x + w - 1, y + h - 1), (x, y + h - 1), color)
    _draw_line(image, (x, y + h - 1), (x, y), color)


def _draw_polyline(image: np.ndarray, pixels: np.ndarray, valid: np.ndarray, color: tuple[int, int, int], closed: bool = True) -> None:
    pts = np.asarray(pixels, dtype=float).reshape(-1, 2)
    ok = np.asarray(valid, dtype=bool).reshape(-1)
    if pts.shape[0] < 2:
        return
    count = pts.shape[0] if closed else pts.shape[0] - 1
    for i in range(count):
        j = (i + 1) % pts.shape[0]
        if bool(ok[i]) and bool(ok[j]) and np.all(np.isfinite(pts[i])) and np.all(np.isfinite(pts[j])):
            _draw_line(image, (pts[i, 0], pts[i, 1]), (pts[j, 0], pts[j, 1]), color)


def _draw_image_ellipse_axes(image: np.ndarray, ellipse: dict[str, Any], color: tuple[int, int, int]) -> None:
    center = ellipse.get("ellipse_center_px", ellipse.get("center_px"))
    major = ellipse.get("major_axis_px")
    minor = ellipse.get("minor_axis_px")
    if major is None or minor is None:
        axes = ellipse.get("axes_px")
        if axes is None:
            return
        major, minor = float(axes[0]), float(axes[1])
    angle = float(ellipse.get("ellipse_angle_deg", ellipse.get("angle_deg", 0.0)))
    if center is None:
        return
    cx, cy = (float(center[0]), float(center[1]))
    theta = np.deg2rad(angle)
    major_vec = np.array([np.cos(theta), np.sin(theta)]) * (0.5 * float(major))
    minor_vec = np.array([-np.sin(theta), np.cos(theta)]) * (0.5 * float(minor))
    _draw_line(image, (cx - major_vec[0], cy - major_vec[1]), (cx + major_vec[0], cy + major_vec[1]), color)
    _draw_line(image, (cx - minor_vec[0], cy - minor_vec[1]), (cx + minor_vec[0], cy + minor_vec[1]), color)


def _draw_char(image: np.ndarray, x: int, y: int, ch: str, color: tuple[int, int, int], scale: int = 1) -> int:
    glyph = _FONT.get(ch.upper(), _FONT[" "])
    for gy, row in enumerate(glyph):
        for gx, value in enumerate(row):
            if value == "1":
                image[y + gy * scale : y + (gy + 1) * scale, x + gx * scale : x + (gx + 1) * scale] = color
    return len(glyph[0]) * scale + scale


def _draw_text(image: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 1) -> None:
    cursor = x
    for ch in text.upper():
        cursor += _draw_char(image, cursor, y, ch, color, scale)


def _vec_lines(prefix: str, vec: Any, scale: float = 1.0) -> list[str]:
    if vec is None:
        return [f"{prefix} NA"]
    arr = np.asarray(vec, dtype=float).reshape(3) * float(scale)
    return [
        f"{prefix}X {arr[0]:.3f}",
        f"{prefix}Y {arr[1]:.3f}",
        f"{prefix}Z {arr[2]:.3f}",
    ]


def _project_base_points(points_base: np.ndarray, T_base_cam: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    T_cam_base = invert_transform(T_base_cam)
    points_cam = transform_points(T_cam_base, points_base)
    return project_points(points_cam, intrinsics)


def project_3d_circle_to_image(
    center_base: np.ndarray,
    radius_m: float,
    axis_base: np.ndarray,
    e1_base: np.ndarray | None,
    e2_base: np.ndarray | None,
    T_cam_base: np.ndarray,
    K: np.ndarray,
    num_points: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis_base, dtype=float).reshape(3)
    if e1_base is None or e2_base is None:
        e1_base, e2_base = plane_basis(axis)
    e1 = np.asarray(e1_base, dtype=float).reshape(3)
    e2 = np.asarray(e2_base, dtype=float).reshape(3)
    center = np.asarray(center_base, dtype=float).reshape(3)
    angles = np.linspace(0.0, 2.0 * np.pi, max(12, int(num_points)), endpoint=False)
    points_base = center + float(radius_m) * (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)
    points_cam = transform_points(T_cam_base, points_base)
    k = np.asarray(K, dtype=float).reshape(3, 3)
    z = points_cam[:, 2]
    valid = np.isfinite(z) & (z > 1e-9)
    pixels = np.full((points_cam.shape[0], 2), np.nan, dtype=float)
    pixels[valid, 0] = k[0, 0] * points_cam[valid, 0] / z[valid] + k[0, 2]
    pixels[valid, 1] = k[1, 1] * points_cam[valid, 1] / z[valid] + k[1, 2]
    return pixels, valid


def draw_debug_overlay(
    rgb: np.ndarray,
    segmentation: Any,
    estimate: Any,
    debug: dict[str, Any],
    intrinsics: CameraIntrinsics,
    T_base_cam: np.ndarray,
    output_path: str,
) -> None:
    image = render_debug_overlay(rgb, segmentation, estimate, debug, intrinsics, T_base_cam)
    save_png_rgb(output_path, image)


def render_debug_overlay(
    rgb: np.ndarray,
    segmentation: Any,
    estimate: Any,
    debug: dict[str, Any],
    intrinsics: CameraIntrinsics,
    T_base_cam: np.ndarray,
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8).copy()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3")

    _blend_mask(image, segmentation.mask, (0, 180, 70), alpha=0.30)
    for item in debug.get("candidate_summaries", []):
        if item.get("generator") in {"hough_circle", "hough_ellipse"} and not item.get("selected"):
            continue
        pixels = item.get("outer_contour_pixels")
        if pixels is None:
            continue
        pts = np.asarray(pixels, dtype=float).reshape(-1, 2)
        if pts.shape[0] > 800:
            pts = pts[np.linspace(0, pts.shape[0] - 1, 800).astype(int)]
        if item.get("selected"):
            color = (255, 0, 255)
        elif item.get("source") == "shape":
            color = (0, 160, 255)
        else:
            color = (0, 255, 120)
        for px in pts:
            if np.all(np.isfinite(px)):
                _draw_point(image, int(round(px[0])), int(round(px[1])), color, radius=1)
        if item.get("selected") and item.get("ellipse"):
            _draw_image_ellipse_axes(image, item["ellipse"], (255, 0, 255))
    ys, xs = np.nonzero(segmentation.outer_contour_mask > 0)
    for x, y in zip(xs, ys):
        image[y, x] = (255, 180, 0)
    _draw_rect(image, segmentation.roi_xywh, (255, 255, 0))

    rejected_base = debug.get("projected_rejected_base")
    if rejected_base is not None:
        rejected_base = np.asarray(rejected_base, dtype=float).reshape(-1, 3)
        if rejected_base.shape[0] > 1200:
            sample_idx = np.linspace(0, rejected_base.shape[0] - 1, 1200).astype(int)
            rejected_base = rejected_base[sample_idx]
        rejected_px, rejected_valid = _project_base_points(rejected_base, T_base_cam, intrinsics)
        for px, ok in zip(rejected_px, rejected_valid):
            if bool(ok) and np.all(np.isfinite(px)):
                _draw_point(image, int(round(px[0])), int(round(px[1])), (80, 120, 255), radius=1)

    envelope_base = debug.get("projected_envelope_base")
    if envelope_base is not None:
        envelope_px, envelope_valid = _project_base_points(np.asarray(envelope_base, dtype=float).reshape(-1, 3), T_base_cam, intrinsics)
        for px, ok in zip(envelope_px, envelope_valid):
            if bool(ok) and np.all(np.isfinite(px)):
                _draw_point(image, int(round(px[0])), int(round(px[1])), (255, 255, 0), radius=2)

    if estimate.center_base is not None and estimate.axis_base is not None and estimate.radius_m is not None:
        center = np.asarray(estimate.center_base, dtype=float).reshape(3)
        axis = np.asarray(estimate.axis_base, dtype=float).reshape(3)
        u = debug.get("basis_u")
        v = debug.get("basis_v")
        if u is None or v is None:
            u, v = plane_basis(axis)
        circle_px, circle_valid = project_3d_circle_to_image(
            center,
            float(estimate.radius_m),
            axis,
            np.asarray(u, dtype=float).reshape(3),
            np.asarray(v, dtype=float).reshape(3),
            invert_transform(T_base_cam),
            intrinsics.as_matrix(),
            num_points=180,
        )
        _draw_polyline(image, circle_px, circle_valid, (0, 220, 255), closed=True)
        center_px, center_valid = _project_base_points(center.reshape(1, 3), T_base_cam, intrinsics)
        if bool(center_valid[0]) and np.all(np.isfinite(center_px[0])):
            cx, cy = (int(round(vv)) for vv in center_px[0])
            _draw_line(image, (cx - 6, cy), (cx + 6, cy), (255, 255, 255))
            _draw_line(image, (cx, cy - 6), (cx, cy + 6), (255, 255, 255))
            _draw_point(image, cx, cy, (255, 0, 255), radius=2)

    quality = estimate.quality.to_dict()
    radius_source = str(quality.get("radius_source") or debug.get("radius_source") or "NA").replace("_", " ")
    frame_label = str(debug.get("coordinate_frame") or getattr(estimate, "frame", "base")).replace("_", " ").replace("/", " ")
    local_frame = "camera" in frame_label.lower() or "local" in frame_label.lower()
    candidate_lines = []
    for item in debug.get("candidate_summaries", [])[:5]:
        reason = ""
        if item.get("failure_reasons"):
            reason = str(item["failure_reasons"][0]).replace("_", " ")[:8]
        source = str(item.get("source", "na"))[:3]
        generator = str(item.get("generator") or "")[:3]
        if generator:
            source = generator
        bins = f"{int(item.get('angular_coverage_occupied_bins', 0))}/{int(item.get('angular_coverage_total_bins', 0))}"
        circle = item.get("circle_rmse_m")
        circle_txt = "NA" if circle is None else f"{float(circle):.3f}"
        candidate_lines.append(f"{source} {float(item.get('score', 0.0)):.2f} {bins} {circle_txt} {reason}")
    lines = [
        f"VALID {int(estimate.valid)}",
        f"FRAME {frame_label[:12]}",
        f"IMG {int(quality.get('image_valid', False))} MET {int(quality.get('metric_valid', False))}",
        f"SRC {str(debug.get('selected_candidate_source', 'NA'))[:5]}",
        f"STATE {str(debug.get('temporal_status', 'RAW'))[:8]}",
        f"VDP {quality.get('valid_depth_ratio', 0.0):.2f}",
        f"DJP {quality.get('contour_depth_jump_ratio', 0.0):.2f}",
        f"MASK {quality['num_mask_points']}",
        f"ENV {quality.get('envelope_num_points', 0)}",
        f"COV {quality.get('angular_coverage_ratio', 0.0):.2f}",
        f"BIN {quality.get('angular_coverage_occupied_bins', 0)}/{quality.get('angular_coverage_total_bins', 0)}",
        f"PLANE {quality['plane_rmse_m'] or 0.0:.4f}",
        f"PIN {quality.get('plane_inlier_ratio', 0.0):.2f}",
        f"CIRCLE {quality.get('envelope_circle_rmse_m') or quality['circle_rmse_m'] or 0.0:.4f}",
        f"CIN {quality['circle_inlier_ratio']:.2f}",
        f"SCR {quality.get('final_geometry_score', float(debug.get('selected_candidate_score', debug.get('final_geometry_score', 0.0)))):.2f}",
        f"R {quality['radius_m'] or 0.0:.3f}M",
        f"RSRC {radius_source[:12]}",
    ]
    center_prefix = "CAM C" if local_frame else "C"
    axis_prefix = "CAM A" if local_frame else "A"
    lines.extend(_vec_lines(center_prefix, estimate.center_base))
    lines.extend(_vec_lines(axis_prefix, estimate.axis_base))
    failures = quality.get("failure_reasons", [])
    temporal_reasons = debug.get("temporal_reasons", [])
    if failures:
        lines.append(f"FAIL {str(failures[0]).replace('_', ' ')[:18]}")
    elif temporal_reasons:
        lines.append(f"FAIL {str(temporal_reasons[0]).replace('_', ' ')[:18]}")
    lines.extend(candidate_lines)
    panel_h = min(image.shape[0], 6 + 11 * len(lines))
    panel_w = min(image.shape[1], 220)
    image[:panel_h, :panel_w] = (0.45 * image[:panel_h, :panel_w]).astype(np.uint8)
    for i, line in enumerate(lines):
        _draw_text(image, 6, 6 + 11 * i, line, (255, 255, 255), scale=1)

    return image
