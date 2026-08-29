#!/usr/bin/env python3
"""Shared Steger centreline extraction for line-laser calibration and reconstruction."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks


@dataclass(frozen=True)
class StegerSettings:
    sigma_px: float = 1.5
    max_offset_px: float = 0.75
    min_normal_y: float = 0.5
    min_response_ratio: float = 0.0005
    background_window_px: int = 31
    background_percentile: float = 20.0
    min_prominence_ratio: float = 0.010
    profile_smoothing_sigma_px: float = 0.8
    sensor_max_value: float | None = None
    scan_axis: str = "column"


@dataclass(frozen=True)
class StegerExtraction:
    u_px: np.ndarray
    v_px: np.ndarray
    valid: np.ndarray
    corrected_signal: np.ndarray
    metadata: dict[str, float | str | None]

    @property
    def pixels(self) -> np.ndarray:
        indices = np.flatnonzero(self.valid)
        if indices.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([self.u_px[indices], self.v_px[indices]])


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return result


def unit_interval(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return result


def positive_odd_int(value: str) -> int:
    result = int(value)
    if result <= 0 or result % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return result


def sensor_full_scale(gray: np.ndarray, sensor_max_value: float | None = None) -> float:
    if sensor_max_value is not None:
        return float(sensor_max_value)
    if gray.dtype == np.uint8:
        return 255.0
    if np.issubdtype(gray.dtype, np.integer):
        maximum = int(np.max(gray))
        return 4095.0 if maximum <= 4095 else float(np.iinfo(gray.dtype).max)
    return float(np.nanmax(gray))


def derivative_images(
    signal: np.ndarray, sigma_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(signal, dtype=np.float32)
    kwargs = {"sigma": sigma_px, "mode": "nearest"}
    gx = gaussian_filter(source, order=(0, 1), **kwargs)
    gy = gaussian_filter(source, order=(1, 0), **kwargs)
    gxx = gaussian_filter(source, order=(0, 2), **kwargs)
    gxy = gaussian_filter(source, order=(1, 1), **kwargs)
    gyy = gaussian_filter(source, order=(2, 0), **kwargs)
    return gx, gy, gxx, gxy, gyy


def steger_point(
    column: int,
    row: int,
    derivatives: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float, float, float, float] | None:
    gx, gy, gxx, gxy, gyy = derivatives
    hessian = np.asarray(
        [[gxx[row, column], gxy[row, column]], [gxy[row, column], gyy[row, column]]],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalue = float(eigenvalues[0])
    if not np.isfinite(eigenvalue) or eigenvalue >= -np.finfo(float).eps:
        return None
    normal = eigenvectors[:, 0]
    gradient = np.asarray([gx[row, column], gy[row, column]], dtype=np.float64)
    offset = -float(gradient @ normal) / eigenvalue
    if not np.isfinite(offset):
        return None
    x = float(column + offset * normal[0])
    y = float(row + offset * normal[1])
    return x, y, -eigenvalue, offset, float(normal[0]), float(normal[1])


def _steger_points_for_peaks(
    columns: np.ndarray,
    rows: np.ndarray,
    derivatives: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of the per-column 2x2 Hessian eigensolve."""
    gx, gy, gxx, gxy, gyy = derivatives
    a = gxx[rows, columns].astype(np.float64, copy=False)
    b = gxy[rows, columns].astype(np.float64, copy=False)
    c = gyy[rows, columns].astype(np.float64, copy=False)
    half_trace = 0.5 * (a + c)
    radius = np.hypot(0.5 * (a - c), b)
    eigenvalue = half_trace - radius
    eigenvalue_max = half_trace + radius
    separation = eigenvalue_max - eigenvalue
    usable = separation > np.finfo(np.float64).eps

    projector_xx = np.zeros_like(eigenvalue)
    projector_xy = np.zeros_like(eigenvalue)
    projector_yy = np.zeros_like(eigenvalue)
    projector_xx[usable] = (
        eigenvalue_max[usable] - a[usable]
    ) / separation[usable]
    projector_xy[usable] = -b[usable] / separation[usable]
    projector_yy[usable] = (
        eigenvalue_max[usable] - c[usable]
    ) / separation[usable]

    sampled_gx = gx[rows, columns]
    sampled_gy = gy[rows, columns]
    projected_x = projector_xx * sampled_gx + projector_xy * sampled_gy
    projected_y = projector_xy * sampled_gx + projector_yy * sampled_gy
    usable &= np.isfinite(eigenvalue) & (eigenvalue < -np.finfo(np.float64).eps)
    displacement_x = np.full_like(eigenvalue, np.nan)
    displacement_y = np.full_like(eigenvalue, np.nan)
    displacement_x[usable] = -projected_x[usable] / eigenvalue[usable]
    displacement_y[usable] = -projected_y[usable] / eigenvalue[usable]
    return (
        displacement_x,
        displacement_y,
        -eigenvalue,
        np.hypot(displacement_x, displacement_y),
        np.sqrt(np.clip(projector_yy, 0.0, 1.0)),
    )


def extract_steger_columns(gray: np.ndarray, settings: StegerSettings) -> StegerExtraction:
    image = gray.astype(np.float32, copy=False)
    full_scale = sensor_full_scale(gray, settings.sensor_max_value)
    background = percentile_filter(
        image,
        percentile=float(settings.background_percentile),
        size=(int(settings.background_window_px), 1),
        mode="nearest",
    )
    corrected = np.maximum(image - background, 0.0)
    peak_signal = (
        gaussian_filter1d(
            corrected,
            sigma=float(settings.profile_smoothing_sigma_px),
            axis=0,
            mode="nearest",
        )
        if settings.profile_smoothing_sigma_px > 0.0
        else corrected
    )
    derivatives = derivative_images(corrected, settings.sigma_px)
    height, width = gray.shape
    min_prominence = settings.min_prominence_ratio * full_scale
    min_response = settings.min_response_ratio * full_scale

    u_px = np.arange(width, dtype=np.float64)
    v_px = np.full(width, np.nan, dtype=np.float64)
    valid = np.zeros(width, dtype=bool)
    reject_counts = {
        "rejected_prominence": 0,
        "rejected_hessian": 0,
        "rejected_steger_response": 0,
        "rejected_steger_offset": 0,
        "rejected_steger_orientation": 0,
        "rejected_image_bounds": 0,
    }
    selected_columns: list[int] = []
    selected_rows: list[int] = []
    for column in range(width):
        profile = peak_signal[:, column]
        peaks, _properties = find_peaks(profile, prominence=min_prominence)
        if peaks.size == 0:
            reject_counts["rejected_prominence"] += 1
            continue

        selected = int(np.argmax(profile[peaks]))
        selected_columns.append(column)
        selected_rows.append(int(peaks[selected]))

    columns = np.asarray(selected_columns, dtype=np.intp)
    rows = np.asarray(selected_rows, dtype=np.intp)
    if columns.size:
        dx, dy, responses, offsets, normal_y_abs = _steger_points_for_peaks(
            columns, rows, derivatives
        )
    else:
        dx = dy = responses = offsets = normal_y_abs = np.empty(0)

    for index, (column, peak) in enumerate(zip(columns, rows, strict=True)):
        if not np.isfinite(dx[index]) or not np.isfinite(dy[index]):
            reject_counts["rejected_hessian"] += 1
            continue

        x = float(column + dx[index])
        y = float(peak + dy[index])
        if responses[index] < min_response:
            reject_counts["rejected_steger_response"] += 1
        elif offsets[index] > settings.max_offset_px:
            reject_counts["rejected_steger_offset"] += 1
        elif normal_y_abs[index] < settings.min_normal_y:
            reject_counts["rejected_steger_orientation"] += 1
        elif not (0.0 <= x < width and 0.0 <= y < height):
            reject_counts["rejected_image_bounds"] += 1
        else:
            u_px[column] = x
            v_px[column] = y
            valid[column] = True

    metadata: dict[str, float | str | None] = {
        "method": "steger_2d_shared",
        **{key: (float(value) if value is not None else None) for key, value in asdict(settings).items()},
        "min_prominence_dn": float(min_prominence),
        "min_response": float(min_response),
        "valid_column_count": float(np.count_nonzero(valid)),
        **{key: float(value) for key, value in reject_counts.items()},
    }
    return StegerExtraction(
        u_px=u_px,
        v_px=v_px,
        valid=valid,
        corrected_signal=corrected,
        metadata=metadata,
    )


def continuity_filter_columns(
    u_px: np.ndarray,
    v_px: np.ndarray,
    valid: np.ndarray,
    window: int,
    max_deviation_px: float,
) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("continuity window must be odd")
    result = valid.copy()
    radius = window // 2
    for index in np.flatnonzero(valid):
        left = max(0, index - radius)
        right = min(len(valid), index + radius + 1)
        neighbours = v_px[left:right][valid[left:right]]
        if neighbours.size >= 3 and abs(v_px[index] - np.median(neighbours)) > max_deviation_px:
            result[index] = False
    return result


def points_from_valid_columns(
    u_px: np.ndarray,
    v_px: np.ndarray,
    valid: np.ndarray,
    max_column_gap: float,
    max_vertical_jump: float,
    min_columns: int,
) -> tuple[np.ndarray, dict[str, float]]:
    indices = np.flatnonzero(valid & np.isfinite(v_px))
    if indices.size == 0:
        return np.empty((0, 2), dtype=np.float64), {
            "candidate_point_count": 0.0,
            "raw_segment_count": 0.0,
            "accepted_segment_count": 0.0,
            "extracted_point_count": 0.0,
        }

    candidates = np.column_stack([u_px[indices], v_px[indices]]).astype(np.float64)
    order = np.argsort(candidates[:, 0], kind="mergesort")
    candidates = candidates[order]
    breaks = np.where(
        (np.diff(candidates[:, 0]) > max_column_gap)
        | (np.abs(np.diff(candidates[:, 1])) > max_vertical_jump)
    )[0] + 1
    raw_segments = np.split(np.arange(len(candidates)), breaks)
    accepted = [segment for segment in raw_segments if len(segment) >= min_columns]
    points = (
        np.concatenate([candidates[segment] for segment in accepted], axis=0)
        if accepted
        else np.empty((0, 2), dtype=np.float64)
    )
    return points, {
        "candidate_point_count": float(len(candidates)),
        "raw_segment_count": float(len(raw_segments)),
        "accepted_segment_count": float(len(accepted)),
        "extracted_point_count": float(len(points)),
    }


def line_ransac_filter_undistorted(
    pixels: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    threshold_px: float,
    iterations: int = 500,
    min_inlier_ratio: float = 0.5,
    seed: int = 20260722,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(pixels) < 2:
        return np.zeros(len(pixels), dtype=bool), np.full(len(pixels), np.inf)
    ideal = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix,
    ).reshape(-1, 2)
    generator = np.random.default_rng(seed)
    best_mask = np.zeros(len(ideal), dtype=bool)
    best_median = math.inf
    for _ in range(iterations):
        first, second = generator.choice(len(ideal), size=2, replace=False)
        direction = ideal[second] - ideal[first]
        length = float(np.linalg.norm(direction))
        if length <= np.finfo(float).eps:
            continue
        direction /= length
        residual = np.abs(
            direction[0] * (ideal[:, 1] - ideal[first, 1])
            - direction[1] * (ideal[:, 0] - ideal[first, 0])
        )
        mask = residual <= threshold_px
        count = int(np.count_nonzero(mask))
        median = float(np.median(residual[mask])) if count else math.inf
        if count > np.count_nonzero(best_mask) or (
            count == np.count_nonzero(best_mask) and median < best_median
        ):
            best_mask = mask
            best_median = median

    minimum = max(2, int(math.ceil(min_inlier_ratio * len(ideal))))
    if int(np.count_nonzero(best_mask)) < minimum:
        return np.zeros(len(pixels), dtype=bool), np.full(len(pixels), np.inf)

    centred = ideal[best_mask] - np.mean(ideal[best_mask], axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    direction = right[0]
    centre = np.mean(ideal[best_mask], axis=0)
    residual = np.abs(
        direction[0] * (ideal[:, 1] - centre[1])
        - direction[1] * (ideal[:, 0] - centre[0])
    )
    return residual <= threshold_px, residual


def signal_to_u8(signal: np.ndarray, full_scale: float | None = None) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if full_scale is None:
        full_scale = float(np.nanmax(values)) if values.size else 0.0
    if full_scale <= 0.0 or not np.isfinite(full_scale):
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(values * 255.0 / full_scale, 0.0, 255.0).astype(np.uint8)


def add_steger_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--steger-sigma-px",
        type=positive_float,
        default=StegerSettings.sigma_px,
        help="2-D Gaussian derivative scale in pixels; default 1.2.",
    )
    parser.add_argument(
        "--steger-max-offset-px",
        type=positive_float,
        default=StegerSettings.max_offset_px,
        help="Maximum Steger subpixel shift along the ridge normal; default 0.75.",
    )
    parser.add_argument(
        "--steger-min-normal-y",
        type=unit_interval,
        default=StegerSettings.min_normal_y,
        help="Minimum absolute vertical component of the ridge normal; default 0.5.",
    )
    parser.add_argument(
        "--steger-min-response-ratio",
        type=nonnegative_float,
        default=StegerSettings.min_response_ratio,
        help="Minimum negative-curvature response as a fraction of full scale; default 0.0005.",
    )
    parser.add_argument(
        "--steger-background-window-px",
        type=positive_odd_int,
        default=StegerSettings.background_window_px,
        help="Vertical percentile-filter window for local background removal; default 31.",
    )
    parser.add_argument(
        "--steger-background-percentile",
        type=unit_interval_or_percent,
        default=StegerSettings.background_percentile,
        help="Background percentile in [0,100]; default 20.",
    )
    parser.add_argument(
        "--steger-min-prominence-ratio",
        type=nonnegative_float,
        default=StegerSettings.min_prominence_ratio,
        help="Minimum column-peak prominence as a fraction of full scale; default 0.010.",
    )
    parser.add_argument(
        "--steger-profile-smoothing-sigma-px",
        type=nonnegative_float,
        default=StegerSettings.profile_smoothing_sigma_px,
        help="Vertical smoothing sigma before peak search; default 0.8.",
    )
    parser.add_argument(
        "--steger-sensor-max-value",
        type=positive_float,
        default=None,
        help="Sensor full-scale value. If omitted, infer from image dtype.",
    )


def unit_interval_or_percent(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise argparse.ArgumentTypeError("must be in [0, 100]")
    return result


def settings_from_args(args: argparse.Namespace) -> StegerSettings:
    background_percentile = getattr(
        args,
        "steger_background_percentile",
        getattr(args, "background_percentile", StegerSettings.background_percentile),
    )
    sensor_max_value = getattr(
        args,
        "steger_sensor_max_value",
        getattr(args, "sensor_max_value", StegerSettings.sensor_max_value),
    )
    return StegerSettings(
        sigma_px=float(args.steger_sigma_px),
        max_offset_px=float(args.steger_max_offset_px),
        min_normal_y=float(args.steger_min_normal_y),
        min_response_ratio=float(args.steger_min_response_ratio),
        background_window_px=int(args.steger_background_window_px),
        background_percentile=float(background_percentile),
        min_prominence_ratio=float(args.steger_min_prominence_ratio),
        profile_smoothing_sigma_px=float(args.steger_profile_smoothing_sigma_px),
        sensor_max_value=None if sensor_max_value is None else float(sensor_max_value),
    )


def settings_metadata(settings: StegerSettings, post_filters: list[str]) -> dict[str, Any]:
    return {
        "method": "steger_realtime",
        "sigma": float(settings.sigma_px),
        "threshold": 30.0,
        "deriv_thresh": 0.5,
        "roi_margin": 120,
        "roi_max_height": 512,
        "scan_axis": getattr(settings, "scan_axis", "column"),
        "post_filters": post_filters,
    }


# 兼容旧的 ``laser.steger_laser_center`` API；实际中心定位统一委托给
# 随测量工具发布的 ``laser.realtime_steger``，避免维护第二套 Hessian 公式。
def _load_realtime_steger():
    from . import realtime_steger

    return realtime_steger


def extract_steger_columns(gray: np.ndarray, settings: StegerSettings):
    options = {
        "sigma": float(settings.sigma_px),
        "threshold": 30.0,
        "deriv_thresh": 0.5,
        "roi_margin": 120,
        "roi_max_height": 512,
        "scan_axis": getattr(settings, "scan_axis", "column"),
    }
    return _load_realtime_steger().extract_steger_columns(gray, options)
