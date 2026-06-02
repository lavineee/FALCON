from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import HSVRange, ValveRingEstimatorConfig


@dataclass
class SegmentationResult:
    handwheel_mask: np.ndarray
    outer_contour_mask: np.ndarray
    outer_contour_pixels: np.ndarray
    roi_xywh: tuple[int, int, int, int]
    area_px: int

    @property
    def mask(self) -> np.ndarray:
        return self.handwheel_mask


def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    arr = image.astype(float) / 255.0
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-12
    idx = nonzero & (maxc == r)
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = nonzero & (maxc == g)
    hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = nonzero & (maxc == b)
    hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    hue = hue * 30.0

    sat = np.zeros_like(maxc)
    sat[maxc > 1e-12] = delta[maxc > 1e-12] / maxc[maxc > 1e-12]
    hsv = np.empty_like(image, dtype=np.uint8)
    hsv[..., 0] = np.clip(np.rint(hue), 0, 179).astype(np.uint8)
    hsv[..., 1] = np.clip(np.rint(sat * 255.0), 0, 255).astype(np.uint8)
    hsv[..., 2] = np.clip(np.rint(maxc * 255.0), 0, 255).astype(np.uint8)
    return hsv


def _in_hsv_range(hsv: np.ndarray, lower: Iterable[int], upper: Iterable[int]) -> np.ndarray:
    lo = np.asarray(tuple(lower), dtype=np.uint8).reshape(3)
    hi = np.asarray(tuple(upper), dtype=np.uint8).reshape(3)
    if int(lo[0]) <= int(hi[0]):
        hue_ok = (hsv[..., 0] >= lo[0]) & (hsv[..., 0] <= hi[0])
    else:
        hue_ok = (hsv[..., 0] >= lo[0]) | (hsv[..., 0] <= hi[0])
    return hue_ok & (hsv[..., 1] >= lo[1]) & (hsv[..., 1] <= hi[1]) & (hsv[..., 2] >= lo[2]) & (hsv[..., 2] <= hi[2])


def segment_hsv(rgb: np.ndarray, hsv_ranges: list[HSVRange], roi_xywh: tuple[int, int, int, int] | None = None) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {image.shape}")
    h, w = image.shape[:2]
    if roi_xywh is None:
        x0, y0, rw, rh = 0, 0, w, h
    else:
        x0, y0, rw, rh = (int(v) for v in roi_xywh)
        x0 = max(0, min(w, x0))
        y0 = max(0, min(h, y0))
        rw = max(0, min(w - x0, rw))
        rh = max(0, min(h - y0, rh))
    hsv = _rgb_to_hsv_numpy(image[y0 : y0 + rh, x0 : x0 + rw])
    roi_mask = np.zeros((rh, rw), dtype=bool)
    for lower, upper in hsv_ranges:
        roi_mask |= _in_hsv_range(hsv, lower, upper)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0 : y0 + rh, x0 : x0 + rw] = roi_mask.astype(np.uint8) * 255
    return mask


def _morph(mask: np.ndarray, kernel_size: int, iterations: int, op: str) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if kernel_size <= 1 or iterations <= 0:
        return out
    pad = int(kernel_size) // 2
    for _ in range(int(iterations)):
        padded = np.pad(out, pad, mode="constant", constant_values=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
        if op == "erode":
            out = np.all(windows, axis=(-2, -1))
        elif op == "dilate":
            out = np.any(windows, axis=(-2, -1))
        else:
            raise ValueError(f"unknown morphology op {op!r}")
    return out


def clean_mask(mask: np.ndarray, open_iters: int = 1, close_iters: int = 1, kernel_size: int = 3) -> np.ndarray:
    binary = np.asarray(mask) > 0
    if open_iters > 0:
        binary = _morph(_morph(binary, kernel_size, open_iters, "erode"), kernel_size, open_iters, "dilate")
    if close_iters > 0:
        binary = _morph(_morph(binary, kernel_size, close_iters, "dilate"), kernel_size, close_iters, "erode")
    return binary.astype(np.uint8) * 255


def largest_connected_component(mask: np.ndarray, min_area_px: int = 1) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask) > 0
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best_pixels: list[tuple[int, int]] = []

    for sy, sx in np.argwhere(binary):
        if visited[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        visited[sy, sx] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) > len(best_pixels):
            best_pixels = pixels

    area = len(best_pixels)
    out = np.zeros_like(binary, dtype=np.uint8)
    if area >= int(min_area_px):
        yy, xx = np.asarray(best_pixels, dtype=int).T
        out[yy, xx] = 255
    return out, int(area)


def external_contour_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    h, w = binary.shape
    background = ~binary
    external_bg = np.zeros_like(binary, dtype=bool)
    stack: list[tuple[int, int]] = []

    for x in range(w):
        if background[0, x]:
            stack.append((0, x))
        if background[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if background[y, 0]:
            stack.append((y, 0))
        if background[y, w - 1]:
            stack.append((y, w - 1))

    for y, x in stack:
        external_bg[y, x] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w and background[ny, nx] and not external_bg[ny, nx]:
                external_bg[ny, nx] = True
                stack.append((ny, nx))

    padded_bg = np.pad(external_bg, 1, mode="constant", constant_values=False)
    neighbor_external = np.zeros_like(binary, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbor_external |= padded_bg[dy : dy + h, dx : dx + w]
    return (binary & neighbor_external).astype(np.uint8) * 255


def contour_pixels_from_mask(contour_mask: np.ndarray, max_points: int | None = None) -> np.ndarray:
    ys, xs = np.nonzero(np.asarray(contour_mask) > 0)
    pixels = np.column_stack((xs.astype(float), ys.astype(float)))
    if max_points is not None and pixels.shape[0] > int(max_points):
        idx = np.linspace(0, pixels.shape[0] - 1, int(max_points)).astype(int)
        pixels = pixels[idx]
    return pixels


def segment_handwheel(rgb: np.ndarray, config: ValveRingEstimatorConfig) -> SegmentationResult:
    h, w = np.asarray(rgb).shape[:2]
    roi = config.roi_xywh or (0, 0, w, h)
    mask = segment_hsv(rgb, config.hsv_ranges, roi)
    mask = clean_mask(mask, config.mask_open_iters, config.mask_close_iters, config.mask_kernel_size)
    if config.keep_largest_component:
        mask, area = largest_connected_component(mask, config.min_mask_area_px)
    else:
        area = int(np.count_nonzero(mask))
    outer = external_contour_mask(mask)
    pixels = contour_pixels_from_mask(outer, config.max_outer_points)
    return SegmentationResult(handwheel_mask=mask, outer_contour_mask=outer, outer_contour_pixels=pixels, roi_xywh=roi, area_px=area)


def segment_valve_ring(rgb: np.ndarray, config: ValveRingEstimatorConfig) -> SegmentationResult:
    return segment_handwheel(rgb, config)
