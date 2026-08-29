#!/usr/bin/env python3
"""Daheng board-reference-only working-distance audit.

The production Steger, camera-ray reconstruction, circular-cone C0 solver,
Session extrinsic semantics, and board polygon helper are reused from the
measurement tool. This wrapper only defines the board-reference selection,
explicit gauge-block exclusion, statistics, and audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT.parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from correction.stage_a_height_scale import CorrectionConfig  # noqa: E402
from laser.realtime_steger import points_from_valid_columns  # noqa: E402
from measurement.board_mask import _points_inside_convex_polygon  # noqa: E402
from online.models import CapturedFrame  # noqa: E402
from online.pipeline import FramePipeline  # noqa: E402
from utils.image_io import load_grayscale_image  # noqa: E402
from working_distance_audit import (  # noqa: E402
    _finite_float,
    _int_value,
    _json_safe,
    _load_session_contract,
    _reconstruct_c0_point,
    _reference_plane_signed_distance,
    _sha256,
    _stats,
)


DEFAULT_RECORDING = (
    TOOL_ROOT
    / "output_daheng_0811"
    / "online_recordings"
    / "0827下午热漂_2000"
    / "recording_20260827_160100"
)
DEFAULT_SESSION = TOOL_ROOT / "output_daheng_0811" / "session_ground_calibration.json"
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_C0_MODEL = (
    TOOL_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
)
DEFAULT_ROI_MANIFEST = (
    REPO_ROOT
    / "projects"
    / "daheng"
    / "analysis"
    / "thermal_a2_0827"
    / "thermal_a2_run_manifest.json"
)
DEFAULT_ANALYSIS_MANIFEST = (
    REPO_ROOT
    / "projects"
    / "daheng"
    / "analysis"
    / "thermal_a3_cold_hot_0827"
    / "thermal_a3_cold_hot_run_manifest.json"
)

RIGHT_WINDOW_COLUMNS = 60
RANGE_ENDPOINT_WINDOW_COLUMNS = 60
CONTINUITY_MAX_COLUMN_GAP = 2
CONTINUITY_MAX_VERTICAL_JUMP = 14.0
CONTINUITY_MIN_COLUMNS = 42
BOARD_PLANE_SANITY_LIMIT_MM = 2.0

CSV_FIELDS = [
    "frame",
    "camera_frame_number",
    "host_timestamp_ns",
    "image_sha256",
    "WD_reference_mm",
    "board_reference_u",
    "board_reference_v",
    "board_reference_X_mm",
    "board_reference_Y_mm",
    "board_reference_Z_mm",
    "board_reference_Z_depth_mm",
    "R_board_reference_mm",
    "board_reference_plane_signed_distance_mm",
    "board_reference_plane_distance_mm",
    "board_reference_candidate_point_count",
    "board_reference_raw_segment_count",
    "board_reference_accepted_segment_count",
    "board_reference_point_count",
    "board_reference_laser_pixel_count",
    "board_reference_u_min_px",
    "board_reference_u_max_px",
    "board_reference_v_min_px",
    "board_reference_v_max_px",
    "board_reference_x_g_mm",
    "board_reference_y_g_mm",
    "board_reference_z_g_mm",
    "board_reference_x_g_min_mm",
    "board_reference_x_g_max_mm",
    "board_reference_y_g_min_mm",
    "board_reference_y_g_max_mm",
    "board_reference_z_g_min_mm",
    "board_reference_z_g_max_mm",
    "board_range_point_count",
    "board_range_top_u",
    "board_range_top_v",
    "board_range_top_X_mm",
    "board_range_top_Y_mm",
    "board_range_top_Z_mm",
    "board_range_top_x_g_mm",
    "board_range_top_y_g_mm",
    "board_range_top_z_g_mm",
    "board_range_bottom_u",
    "board_range_bottom_v",
    "board_range_bottom_X_mm",
    "board_range_bottom_Y_mm",
    "board_range_bottom_Z_mm",
    "board_range_bottom_x_g_mm",
    "board_range_bottom_y_g_mm",
    "board_range_bottom_z_g_mm",
    "board_range_delta_x_g_mm",
    "board_range_delta_y_g_mm",
    "board_range_delta_z_g_mm",
    "board_range_y_g_span_mm",
    "board_range_ground_xy_width_mm",
    "board_range_ground_3d_width_mm",
    "board_range_top_window_count",
    "board_range_bottom_window_count",
    "board_polygon_point_count",
    "board_polygon_laser_pixel_count",
    "excluded_gauge_point_count",
    "excluded_nonbaseline_board_point_count",
    "steger_point_count",
    "c0_valid_point_count",
    "reconstruction_filtered",
    "status",
]

COVERAGE_CSV_FIELDS = [
    "frame",
    "camera_frame_number",
    "sample_index",
    "u_px",
    "v_px",
    "X_camera_mm",
    "Y_camera_mm",
    "Z_camera_mm",
    "x_g_mm",
    "y_g_mm",
    "z_g_mm",
    "is_representative_window",
]


@dataclass(frozen=True, slots=True)
class ContinuousSelection:
    points: np.ndarray
    source_indices: np.ndarray
    candidate_point_count: int
    raw_segment_count: int
    accepted_segment_count: int


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def _write_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=COVERAGE_CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(row.get(field))
                    for field in COVERAGE_CSV_FIELDS
                }
            )


def _interval(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element interval")
    lower = _finite_float(value[0], f"{name}[0]")
    upper = _finite_float(value[1], f"{name}[1]")
    if lower > upper:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return lower, upper


def _load_roi_protocol(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ROI protocol manifest does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    rois = document.get("rois")
    if not isinstance(rois, dict):
        raise ValueError("ROI protocol manifest has no rois mapping")
    baseline_intervals: list[tuple[float, float]] = []
    gauge_intervals: list[tuple[float, float]] = []
    objects: dict[str, Any] = {}
    for object_id in ("upper", "middle", "lower"):
        item = rois.get(object_id)
        if not isinstance(item, dict):
            raise ValueError(f"ROI protocol is missing rois.{object_id}")
        before = _interval(item.get("baseline_before"), f"rois.{object_id}.baseline_before")
        height = _interval(item.get("height"), f"rois.{object_id}.height")
        after = _interval(item.get("baseline_after"), f"rois.{object_id}.baseline_after")
        baseline_intervals.extend((before, after))
        gauge_intervals.append(height)
        objects[object_id] = {
            "object_id": item.get("object_id", object_id),
            "position": item.get("position"),
            "nominal_height_mm": item.get("nominal_height_mm"),
            "baseline_before": before,
            "height_excluded": height,
            "baseline_after": after,
        }
    return {
        "path": path.resolve(),
        "sha256": _sha256(path),
        "objects": objects,
        "baseline_intervals_v": baseline_intervals,
        "gauge_intervals_v": gauge_intervals,
    }


def _load_analysis_relation(path: Path, recording_id: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": path.resolve(),
            "exists": False,
            "recording_listed": None,
            "hot_session_sha256": None,
        }
    document = json.loads(path.read_text(encoding="utf-8"))
    provenance = document.get("provenance") or {}
    terminal_ids = provenance.get("terminal_recording_ids") or []
    return {
        "path": path.resolve(),
        "exists": True,
        "sha256": _sha256(path),
        "recording_listed": recording_id in terminal_ids,
        "hot_session_sha256": provenance.get("hot_session_sha256"),
        "terminal_recording_ids": terminal_ids,
    }


def _effective_pipeline(config_path: Path, c0_model_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    base_config = load_app_config(config_path)
    if str(base_config.extraction_method).lower() != "steger":
        raise ValueError("Daheng board audit requires extraction.method=steger")
    scan_axis = str(base_config.extraction_options.get("scan_axis", "column")).lower()
    if scan_axis not in {"column", "row"}:
        raise ValueError(f"unsupported Steger scan_axis={scan_axis!r}")
    effective_config = replace(
        base_config,
        correction=CorrectionConfig(),
        reconstruction=replace(
            base_config.reconstruction,
            enable_laser_ray_correction=False,
        ),
    )
    pipeline = FramePipeline(effective_config)
    c0_calibration = load_calibration_files(
        intrinsics=base_config.calibration.intrinsics,
        laser_plane=c0_model_path,
        extrinsics=base_config.calibration.extrinsics,
        ground_u_compensation=None,
        laser_ray_correction=None,
    )
    laser_model = c0_calibration.get("laser_model")
    if not isinstance(laser_model, dict) or laser_model.get("model_type") != "circular_cone":
        raise ValueError("explicit Daheng C0 model is not circular_cone")
    # FramePipeline remains the production frame runner; only its in-memory
    # calibration snapshot is replaced by the existing Daheng C0 model.
    pipeline.package = replace(
        pipeline.package,
        package_id=f"{pipeline.package.package_id}-circular-cone-c0-audit",
        calibration=c0_calibration,
    )
    metadata = {
        "scan_axis": scan_axis,
        "base_package_id": pipeline.package.package_id.removesuffix(
            "-circular-cone-c0-audit"
        ),
        "base_manifest_sha256": pipeline.package.manifest_sha256,
        "c0_model": laser_model,
        "c0_model_path": c0_model_path.resolve(),
        "c0_model_sha256": _sha256(c0_model_path),
        "base_config_extraction_method": base_config.extraction_method,
        "base_config_correction_mode": str(base_config.correction.mode),
        "base_config_c1_enabled": bool(
            getattr(base_config.reconstruction, "enable_laser_ray_correction", False)
        ),
    }
    return pipeline, effective_config, metadata


def _interval_mask(values: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.zeros(len(values), dtype=bool)
    for lower, upper in intervals:
        mask |= (values >= lower) & (values <= upper)
    return mask


def _match_point_indices(
    candidates: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Map production continuity output back to its source-point rows."""
    source = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(selected, dtype=np.float64).reshape(-1, 2)
    used = np.zeros(len(source), dtype=bool)
    indices: list[int] = []
    for point in target:
        matches = np.flatnonzero(
            (~used)
            & np.isclose(source[:, 0], point[0], rtol=0.0, atol=1.0e-7)
            & np.isclose(source[:, 1], point[1], rtol=0.0, atol=1.0e-7)
        )
        if not len(matches):
            raise ValueError("continuity output cannot be aligned to source pixels")
        index = int(matches[0])
        used[index] = True
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _select_continuous_scan_points(
    points: np.ndarray,
    scan_axis: str,
) -> ContinuousSelection | None:
    candidates = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(candidates):
        return None
    valid = np.isfinite(candidates).all(axis=1)
    if scan_axis == "row":
        along = candidates[:, 1]
        lateral = candidates[:, 0]
    else:
        along = candidates[:, 0]
        lateral = candidates[:, 1]
    accepted, metadata = points_from_valid_columns(
        along,
        lateral,
        valid,
        CONTINUITY_MAX_COLUMN_GAP,
        CONTINUITY_MAX_VERTICAL_JUMP,
        CONTINUITY_MIN_COLUMNS,
    )
    if not len(accepted):
        return None
    accepted_original = (
        accepted[:, [1, 0]] if scan_axis == "row" else accepted
    )
    source_indices = _match_point_indices(candidates, accepted_original)
    if scan_axis == "row":
        accepted = accepted[:, [1, 0]]
    return ContinuousSelection(
        points=np.ascontiguousarray(accepted, dtype=np.float64),
        source_indices=np.ascontiguousarray(source_indices, dtype=np.int64),
        candidate_point_count=int(metadata["candidate_point_count"]),
        raw_segment_count=int(metadata["raw_segment_count"]),
        accepted_segment_count=int(metadata["accepted_segment_count"]),
    )


def _representative_pixel(selection: ContinuousSelection) -> tuple[np.ndarray, np.ndarray]:
    # The board-reference region is represented by the median of all retained
    # continuous baseline samples, never by one extreme pixel.
    window = selection.points[-min(RIGHT_WINDOW_COLUMNS, len(selection.points)) :]
    all_region_median = np.median(selection.points, axis=0)
    return np.ascontiguousarray(all_region_median, dtype=np.float64), np.ascontiguousarray(
        window, dtype=np.float64
    )


def _normalized_bgr(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image)
    if gray.dtype == np.uint8:
        gray8 = gray
    else:
        gray8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)


def _draw_overlay(
    path: Path,
    visual: dict[str, Any],
    session: Any,
    row: dict[str, Any],
    roi_protocol: dict[str, Any],
) -> None:
    canvas = _normalized_bgr(visual["image"])
    offset = np.array([visual["offset_x"], visual["offset_y"]], dtype=np.float64)
    height, width = canvas.shape[:2]

    bands = canvas.copy()
    for lower, upper in roi_protocol["gauge_intervals_v"]:
        y0 = max(0, int(round(lower - offset[1])))
        y1 = min(height - 1, int(round(upper - offset[1])))
        if y0 <= y1:
            cv2.rectangle(bands, (0, y0), (width - 1, y1), (0, 0, 190), -1)
    for lower, upper in roi_protocol["baseline_intervals_v"]:
        y0 = max(0, int(round(lower - offset[1])))
        y1 = min(height - 1, int(round(upper - offset[1])))
        if y0 <= y1:
            cv2.rectangle(bands, (0, y0), (width - 1, y1), (0, 130, 0), 1)
    cv2.addWeighted(bands, 0.24, canvas, 0.76, 0.0, canvas)

    polygon = np.asarray(session.polygon_full_uv, dtype=np.float64) - offset
    polygon_i = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [polygon_i], True, (0, 220, 255), 2, cv2.LINE_AA)

    valid_pixels = np.asarray(visual["valid_pixels"], dtype=np.float64)
    if len(valid_pixels):
        sampled = valid_pixels[:: max(1, len(valid_pixels) // 1800)] - offset
        for u, v in np.round(sampled).astype(np.int32):
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(canvas, (int(u), int(v)), 1, (150, 150, 150), -1)

    board_pixels = np.asarray(visual["board_pixels"], dtype=np.float64)
    if len(board_pixels):
        sampled = board_pixels[:: max(1, len(board_pixels) // 1500)] - offset
        for u, v in np.round(sampled).astype(np.int32):
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(canvas, (int(u), int(v)), 1, (255, 180, 0), -1)

    baseline_pixels = np.asarray(visual["baseline_pixels"], dtype=np.float64)
    if len(baseline_pixels):
        sampled = baseline_pixels[:: max(1, len(baseline_pixels) // 1500)] - offset
        for u, v in np.round(sampled).astype(np.int32):
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(canvas, (int(u), int(v)), 1, (0, 255, 0), -1)

    rep = np.asarray(visual["representative_pixel"], dtype=np.float64) - offset
    rep_i = tuple(np.round(rep).astype(np.int32).tolist())
    if 0 <= rep_i[0] < width and 0 <= rep_i[1] < height:
        cv2.drawMarker(
            canvas,
            rep_i,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )

    for key, color in (
        ("range_top_pixel", (255, 0, 255)),
        ("range_bottom_pixel", (0, 165, 255)),
    ):
        endpoint = np.asarray(visual[key], dtype=np.float64) - offset
        endpoint_i = tuple(np.round(endpoint).astype(np.int32).tolist())
        if 0 <= endpoint_i[0] < width and 0 <= endpoint_i[1] < height:
            cv2.drawMarker(
                canvas,
                endpoint_i,
                color,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=26,
                thickness=2,
            )

    lines = [
        f"Daheng board-only | {row['frame']} | full pixel coordinates",
        f"WD_reference = {float(row['WD_reference_mm']):.3f} mm",
        f"board reference C0 range = {float(row['R_board_reference_mm']):.3f} mm",
        f"representative = ({float(row['board_reference_u']):.1f}, {float(row['board_reference_v']):.1f})",
        f"x_g/y_g = ({float(row['board_reference_x_g_mm']):.2f}, {float(row['board_reference_y_g_mm']):.2f}) mm",
        f"range y_g span = {float(row['board_range_y_g_span_mm']):.2f} mm | ground XY = {float(row['board_range_ground_xy_width_mm']):.2f} mm",
        f"magenta/orange = range top/bottom; y_g = {float(row['board_range_top_y_g_mm']):.2f} / {float(row['board_range_bottom_y_g_mm']):.2f} mm",
        "yellow=board polygon, green=retained baseline centers, red bands=gauge excluded",
    ]
    for index, text in enumerate(lines):
        y = 24 + index * 24
        cv2.putText(
            canvas,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (25, 25, 25),
            1,
            lineType=cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("failed to encode board-only overlay")
    encoded.tofile(str(path))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _numeric_stats(values: list[Any]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
    ]
    if not finite:
        return {
            "valid_count": 0,
            "median": None,
            "mean": None,
            "std": None,
            "p05": None,
            "p95": None,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "valid_count": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def _row_numeric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return _numeric_stats([row.get(key) for row in rows])


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    stats = summary["results"]["stats"]
    coverage = summary["coverage"]
    coverage_frame = coverage["per_frame_stats"]
    coverage_pooled = coverage["pooled_sample_stats"]
    board_range = summary["board_range"]["per_frame_stats"]
    warnings = summary["warnings"]
    wd = stats["WD_reference_mm"]
    board = stats["R_board_reference_mm"]
    plane = stats["board_reference_plane_distance_mm"]
    objects = summary["selection_protocol"]["objects"]
    lines = [
        "# WD-1 | Daheng board-reference-only working-distance audit",
        "",
        f"- Generated at: {summary['generated_at_utc']}",
        f"- Recording: {summary['recording']['path']}",
        f"- Processed frames: {summary['recording']['frame_count']}",
        f"- Valid board-reference frames: {summary['quality']['valid_board_reference_frame_count']}",
        "",
        "## Conclusion",
        "",
        (
            f"当前大恒视觉系统相机光学中心至棋盘格 Session 基准平面的法向工作距离为 "
            f"**{_fmt(wd['median_mm'], 1)} mm**。仅在棋盘格基准区域内、排除上/中/下量块 "
            f"height 区间后，circular-cone C0 代表点距相机光心的空间斜距为 "
            f"**{_fmt(board['median_mm'], 1)} mm**。"
        ),
        "",
        (
            f"棋盘格内保留的生产 Steger 激光中心线覆盖为每帧中位数 "
            f"**{_fmt(coverage_frame['centerline_pixel_count']['median'], 0)} 个亚像素中心点**，"
            f"合计 **{coverage['centerline_sample_count_total']} 个中心点**；"
            f"对应的代表 ground 坐标中位数为 "
            f"**x_g={_fmt(coverage_frame['representative_x_g_mm']['median'], 2)} mm，"
            f"y_g={_fmt(coverage_frame['representative_y_g_mm']['median'], 2)} mm**。"
        ),
        "",
        (
            f"按图中上下红色标记定义的棋盘格激光有效范围，其 `y_g` 轴跨度中位数为 "
            f"**{_fmt(board_range['y_g_span_mm']['median'], 2)} mm**；"
            f"ground XY 端点距离中位数为 **{_fmt(board_range['ground_xy_width_mm']['median'], 2)} mm**。"
        ),
        "",
        "本次不计算整幅图右端距离，也不计算量块区域；`WD_reference` 才是系统工作距离，"
        "棋盘基准区 C0 斜距仅作实际观测位置的辅助描述。",
        "",
        "## Statistics (mm)",
        "",
        "| Metric | median | mean | std | P05 | P95 | valid frame count |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| WD_reference | {_fmt(wd['median_mm'])} | {_fmt(wd['mean_mm'])} | "
            f"{_fmt(wd['std_mm'])} | {_fmt(wd['p05_mm'])} | {_fmt(wd['p95_mm'])} | "
            f"{wd['valid_frame_count']} |"
        ),
        (
            f"| R_board_reference (C0) | {_fmt(board['median_mm'])} | {_fmt(board['mean_mm'])} | "
            f"{_fmt(board['std_mm'])} | {_fmt(board['p05_mm'])} | {_fmt(board['p95_mm'])} | "
            f"{board['valid_frame_count']} |"
        ),
        "",
        "## Board laser-center pixel coverage and ground coordinates",
        "",
        "这里的‘覆盖像素’定义为生产 Steger 提取的激光中心线亚像素采样点，不是原始亮条纹的面积像素数。",
        "每个采样点在 `board_laser_coverage.csv` 中与 `(u_px, v_px)`、相机 XYZ 及 ground `(x_g, y_g, z_g)` 逐行对应；单位均为 mm（像素坐标除外）。",
        "ground 坐标沿用 `FrameResult.points_ground[:, 0:3]` 的列定义，即 `(x_g, y_g, z_g)`；未用棋盘格平面外推。",
        (
            f"每帧保留中心线像素包围盒中位数为 "
            f"u=[{_fmt(coverage_frame['u_min_px']['median'], 2)}, {_fmt(coverage_frame['u_max_px']['median'], 2)}]，"
            f"v=[{_fmt(coverage_frame['v_min_px']['median'], 0)}, {_fmt(coverage_frame['v_max_px']['median'], 0)}]。"
        ),
        "",
        "| Coverage metric | median | mean | std | P05 | P95 | valid count |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| retained centerline samples / frame | {_fmt(coverage_frame['centerline_pixel_count']['median'], 0)} | "
            f"{_fmt(coverage_frame['centerline_pixel_count']['mean'], 2)} | "
            f"{_fmt(coverage_frame['centerline_pixel_count']['std'], 2)} | "
            f"{_fmt(coverage_frame['centerline_pixel_count']['p05'], 0)} | "
            f"{_fmt(coverage_frame['centerline_pixel_count']['p95'], 0)} | "
            f"{coverage_frame['centerline_pixel_count']['valid_count']} frames |"
        ),
        (
            f"| representative x_g (mm) | {_fmt(coverage_frame['representative_x_g_mm']['median'], 2)} | "
            f"{_fmt(coverage_frame['representative_x_g_mm']['mean'], 2)} | "
            f"{_fmt(coverage_frame['representative_x_g_mm']['std'], 2)} | "
            f"{_fmt(coverage_frame['representative_x_g_mm']['p05'], 2)} | "
            f"{_fmt(coverage_frame['representative_x_g_mm']['p95'], 2)} | "
            f"{coverage_frame['representative_x_g_mm']['valid_count']} frames |"
        ),
        (
            f"| representative y_g (mm) | {_fmt(coverage_frame['representative_y_g_mm']['median'], 2)} | "
            f"{_fmt(coverage_frame['representative_y_g_mm']['mean'], 2)} | "
            f"{_fmt(coverage_frame['representative_y_g_mm']['std'], 2)} | "
            f"{_fmt(coverage_frame['representative_y_g_mm']['p05'], 2)} | "
            f"{_fmt(coverage_frame['representative_y_g_mm']['p95'], 2)} | "
            f"{coverage_frame['representative_y_g_mm']['valid_count']} frames |"
        ),
        (
            f"| pooled x_g samples (mm) | {_fmt(coverage_pooled['x_g_mm']['median'], 2)} | "
            f"{_fmt(coverage_pooled['x_g_mm']['mean'], 2)} | "
            f"{_fmt(coverage_pooled['x_g_mm']['std'], 2)} | "
            f"{_fmt(coverage_pooled['x_g_mm']['p05'], 2)} | "
            f"{_fmt(coverage_pooled['x_g_mm']['p95'], 2)} | "
            f"{coverage_pooled['x_g_mm']['valid_count']} samples |"
        ),
        (
            f"| pooled y_g samples (mm) | {_fmt(coverage_pooled['y_g_mm']['median'], 2)} | "
            f"{_fmt(coverage_pooled['y_g_mm']['mean'], 2)} | "
            f"{_fmt(coverage_pooled['y_g_mm']['std'], 2)} | "
            f"{_fmt(coverage_pooled['y_g_mm']['p05'], 2)} | "
            f"{_fmt(coverage_pooled['y_g_mm']['p95'], 2)} | "
            f"{coverage_pooled['y_g_mm']['valid_count']} samples |"
        ),
        "",
        "## Full board laser range (the red-marked top-to-bottom span)",
        "",
        "上下端点均取排除量块高度段后连续激光中心线窗口的中位数；`y_g span=abs(bottom_y_g-top_y_g)`，ground XY distance 同时考虑 x_g 方向变化。",
        "",
        "| Range metric | median | mean | std | P05 | P95 | valid frame count |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| top y_g (mm) | {_fmt(board_range['top_y_g_mm']['median'], 2)} | {_fmt(board_range['top_y_g_mm']['mean'], 2)} | "
            f"{_fmt(board_range['top_y_g_mm']['std'], 2)} | {_fmt(board_range['top_y_g_mm']['p05'], 2)} | "
            f"{_fmt(board_range['top_y_g_mm']['p95'], 2)} | {board_range['top_y_g_mm']['valid_count']} |"
        ),
        (
            f"| bottom y_g (mm) | {_fmt(board_range['bottom_y_g_mm']['median'], 2)} | {_fmt(board_range['bottom_y_g_mm']['mean'], 2)} | "
            f"{_fmt(board_range['bottom_y_g_mm']['std'], 2)} | {_fmt(board_range['bottom_y_g_mm']['p05'], 2)} | "
            f"{_fmt(board_range['bottom_y_g_mm']['p95'], 2)} | {board_range['bottom_y_g_mm']['valid_count']} |"
        ),
        (
            f"| y_g span (mm) | {_fmt(board_range['y_g_span_mm']['median'], 2)} | {_fmt(board_range['y_g_span_mm']['mean'], 2)} | "
            f"{_fmt(board_range['y_g_span_mm']['std'], 2)} | {_fmt(board_range['y_g_span_mm']['p05'], 2)} | "
            f"{_fmt(board_range['y_g_span_mm']['p95'], 2)} | {board_range['y_g_span_mm']['valid_count']} |"
        ),
        (
            f"| ground XY endpoint width (mm) | {_fmt(board_range['ground_xy_width_mm']['median'], 2)} | {_fmt(board_range['ground_xy_width_mm']['mean'], 2)} | "
            f"{_fmt(board_range['ground_xy_width_mm']['std'], 2)} | {_fmt(board_range['ground_xy_width_mm']['p05'], 2)} | "
            f"{_fmt(board_range['ground_xy_width_mm']['p95'], 2)} | {board_range['ground_xy_width_mm']['valid_count']} |"
        ),
        (
            f"| ground 3D endpoint width (mm) | {_fmt(board_range['ground_3d_width_mm']['median'], 2)} | {_fmt(board_range['ground_3d_width_mm']['mean'], 2)} | "
            f"{_fmt(board_range['ground_3d_width_mm']['std'], 2)} | {_fmt(board_range['ground_3d_width_mm']['p05'], 2)} | "
            f"{_fmt(board_range['ground_3d_width_mm']['p95'], 2)} | {board_range['ground_3d_width_mm']['valid_count']} |"
        ),
        "",
        "## Board-reference selection and gauge exclusion",
        "",
        "- Session `session_ground_reference.support.polygon_full_uv` is the board physical polygon.",
        "- Only `baseline_before` and `baseline_after` samples from the frozen ROI protocol are retained.",
        "- The three `height` intervals are excluded before continuous-segment selection.",
        "- The representative pixel is the median of all retained continuous baseline samples; no single extreme pixel is used.",
        "",
        "| Object | baseline before | excluded height | baseline after |",
        "|---|---:|---:|---:|",
    ]
    for object_id in ("upper", "middle", "lower"):
        item = objects[object_id]
        lines.append(
            f"| {object_id} | {item['baseline_before']} | {item['height_excluded']} | {item['baseline_after']} |"
        )
    lines.extend(
        [
            "",
            "## Geometry and protocol",
            "",
            "1. Existing production `FramePipeline.run_frame` and Steger extractor were used.",
            "2. Existing undistorted camera-ray + circular-cone C0 reconstruction was used for the representative point.",
            "3. C1, H1, and H-B2 were disabled for this audit; the current Daheng online default is Quadratic+C1/H1 and is recorded as a protocol difference.",
            "4. No reference-plane intersection was used to recover the laser point; the reference plane is used only for `WD_reference` and the board sanity check.",
            "",
            "## Reference-plane sanity check",
            "",
            (
                f"Board-reference C0 point absolute normal distance to the Session plane: "
                f"median={_fmt(plane['median_mm'])} mm, mean={_fmt(plane['mean_mm'])} mm, "
                f"std={_fmt(plane['std_mm'])} mm, P05/P95={_fmt(plane['p05_mm'])}/{_fmt(plane['p95_mm'])} mm."
            ),
            (
                f"{summary['quality']['board_plane_within_sanity_limit_count']}/"
                f"{summary['quality']['board_plane_valid_count']} valid board-reference points "
                f"are within the {BOARD_PLANE_SANITY_LIMIT_MM:.1f} mm diagnostic threshold."
            ),
            "",
            "## Provenance and relation note",
            "",
            f"- User-selected Session JSON: {summary['session']['path']}",
            f"- Session JSON SHA-256: {summary['session']['sha256']}",
            f"- Daheng config: {summary['provenance']['config_path']}",
            f"- Base calibration manifest SHA-256: {summary['provenance']['base_manifest_sha256']}",
            f"- Explicit Daheng C0 model: {summary['provenance']['c0_model_path']}",
            f"- Explicit Daheng C0 model SHA-256: {summary['provenance']['c0_model_sha256']}",
            f"- ROI protocol manifest: {summary['selection_protocol']['path']}",
            f"- Numeric results were newly computed from the {summary['recording']['frame_count']} target PNGs.",
            "- Reused implementations: production Steger, FramePipeline, camera ray/C0 reconstruction, Session extrinsic semantics, board polygon helper, and continuity helper.",
            "",
            "## Output files",
            "",
            "- working_distance_frames.csv",
            "- working_distance_summary.json",
            "- working_distance_report.md",
            "- working_distance_overlay.png",
            "- board_laser_coverage.csv",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    recording_dir: Path,
    session_path: Path,
    config_path: Path,
    c0_model_path: Path,
    roi_manifest_path: Path,
    analysis_manifest_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    recording_dir = recording_dir.resolve()
    session_path = session_path.resolve()
    config_path = config_path.resolve()
    c0_model_path = c0_model_path.resolve()
    roi_manifest_path = roi_manifest_path.resolve()
    analysis_manifest_path = analysis_manifest_path.resolve()
    if not recording_dir.is_dir():
        raise FileNotFoundError(f"recording directory does not exist: {recording_dir}")
    if output_dir is None:
        output_dir = recording_dir / "working_distance_audit"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _load_session_contract(session_path)
    roi_protocol = _load_roi_protocol(roi_manifest_path)
    relation = _load_analysis_relation(analysis_manifest_path, recording_dir.name)
    pipeline, effective_config, c0_metadata = _effective_pipeline(config_path, c0_model_path)
    pipeline.apply_session_ground_extrinsic(
        session.R_camera_to_ground,
        session.t_camera_to_ground_mm,
        generation=session.ground_extrinsic_generation,
    )
    calibration = pipeline.calibration_for_reconstruction()
    laser_model = calibration.get("laser_model")
    if not isinstance(laser_model, dict) or laser_model.get("model_type") != "circular_cone":
        raise ValueError("effective reconstruction model is not circular_cone")

    frames_csv = recording_dir / "frames.csv"
    if not frames_csv.is_file():
        raise FileNotFoundError(f"frames.csv does not exist: {frames_csv}")
    with frames_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    if not frame_rows:
        raise ValueError("frames.csv has no frame rows")
    required = {
        "filename",
        "camera_frame_number",
        "camera_timestamp_ticks",
        "host_timestamp_ns",
        "host_monotonic_ns",
        "offset_x",
        "offset_y",
        "width",
        "height",
    }
    missing = required - set(frame_rows[0])
    if missing:
        raise ValueError(f"frames.csv is missing fields: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    visuals: dict[str, dict[str, Any]] = {}
    observed_exposure: set[float] = set()
    observed_roi: set[tuple[int, int, int, int]] = set()

    for source_row in frame_rows:
        filename = source_row["filename"]
        row: dict[str, Any] = {
            "frame": filename,
            "camera_frame_number": _int_value(
                source_row["camera_frame_number"], "camera_frame_number"
            ),
            "host_timestamp_ns": _int_value(
                source_row["host_timestamp_ns"], "host_timestamp_ns"
            ),
            "WD_reference_mm": session.WD_reference_mm,
            "status": "pending",
        }
        image_path = recording_dir / filename
        try:
            row["image_sha256"] = _sha256(image_path)
            image = load_grayscale_image(image_path)
            expected_shape = (
                _int_value(source_row["height"], "height"),
                _int_value(source_row["width"], "width"),
            )
            if image.shape != expected_shape:
                raise ValueError(f"image shape {image.shape} != frames.csv {expected_shape}")
            offset_x = _int_value(source_row["offset_x"], "offset_x")
            offset_y = _int_value(source_row["offset_y"], "offset_y")
            observed_roi.add((offset_x, offset_y, expected_shape[1], expected_shape[0]))
            if source_row.get("exposure_us") not in (None, ""):
                observed_exposure.add(_finite_float(source_row["exposure_us"], "exposure_us"))

            captured = CapturedFrame(
                image=image,
                camera_frame_number=row["camera_frame_number"],
                camera_timestamp_ticks=_int_value(
                    source_row["camera_timestamp_ticks"], "camera_timestamp_ticks"
                ),
                host_timestamp_ns=row["host_timestamp_ns"],
                host_monotonic_ns=_int_value(
                    source_row["host_monotonic_ns"], "host_monotonic_ns"
                ),
                offset_x=offset_x,
                offset_y=offset_y,
            )
            result = pipeline.run_frame(captured)
            centers = np.asarray(result.centers_uv_full, dtype=np.float64).reshape(-1, 2)
            valid_pixels = np.asarray(result.pixels_uv, dtype=np.float64).reshape(-1, 2)
            c0_points = np.asarray(result.points_camera, dtype=np.float64).reshape(-1, 3)
            ground_points = np.asarray(result.points_ground, dtype=np.float64).reshape(-1, 3)
            if len(valid_pixels) != len(c0_points) or len(valid_pixels) != len(ground_points):
                raise ValueError("pixels_uv, camera points, and ground points are not aligned")
            row["steger_point_count"] = int(len(centers))
            row["c0_valid_point_count"] = int(len(c0_points))
            row["reconstruction_filtered"] = dict(result.filtered)

            polygon_mask = _points_inside_convex_polygon(
                valid_pixels, session.polygon_full_uv
            )
            board_pixels = valid_pixels[polygon_mask]
            board_camera_points = c0_points[polygon_mask]
            board_ground_points = ground_points[polygon_mask]
            base_mask = _interval_mask(
                board_pixels[:, 1], roi_protocol["baseline_intervals_v"]
            )
            gauge_mask = _interval_mask(
                board_pixels[:, 1], roi_protocol["gauge_intervals_v"]
            )
            baseline_pixels = board_pixels[base_mask]
            baseline_camera_points = board_camera_points[base_mask]
            baseline_ground_points = board_ground_points[base_mask]
            range_mask = ~gauge_mask
            range_pixels = board_pixels[range_mask]
            row["board_polygon_point_count"] = int(len(board_pixels))
            row["board_polygon_laser_pixel_count"] = int(len(board_pixels))
            row["excluded_gauge_point_count"] = int(np.count_nonzero(gauge_mask))
            row["excluded_nonbaseline_board_point_count"] = int(
                len(board_pixels) - len(baseline_pixels)
            )

            selection = _select_continuous_scan_points(
                baseline_pixels,
                str(c0_metadata["scan_axis"]),
            )
            if selection is None:
                raise ValueError("no continuous board baseline segment")
            row["board_reference_candidate_point_count"] = selection.candidate_point_count
            row["board_reference_raw_segment_count"] = selection.raw_segment_count
            row["board_reference_accepted_segment_count"] = selection.accepted_segment_count
            row["board_reference_point_count"] = int(len(selection.points))
            row["board_reference_laser_pixel_count"] = int(len(selection.points))

            selected_indices = selection.source_indices
            selected_camera_points = baseline_camera_points[selected_indices]
            selected_ground_points = baseline_ground_points[selected_indices]
            selected_pixels = baseline_pixels[selected_indices]
            if len(selected_pixels) != len(selection.points):
                raise ValueError("selected board pixels and continuity output are not aligned")

            representative_pixel, representative_window = _representative_pixel(selection)
            representative_point = _reconstruct_c0_point(
                representative_pixel,
                calibration,
                effective_config.reconstruction,
            )
            representative_ground = (
                session.R_camera_to_ground @ representative_point
                + session.t_camera_to_ground_mm
            )
            row["board_reference_u"] = float(representative_pixel[0])
            row["board_reference_v"] = float(representative_pixel[1])
            row["board_reference_X_mm"] = float(representative_point[0])
            row["board_reference_Y_mm"] = float(representative_point[1])
            row["board_reference_Z_mm"] = float(representative_point[2])
            row["board_reference_Z_depth_mm"] = float(representative_point[2])
            row["R_board_reference_mm"] = float(np.linalg.norm(representative_point))
            row["board_reference_x_g_mm"] = float(representative_ground[0])
            row["board_reference_y_g_mm"] = float(representative_ground[1])
            row["board_reference_z_g_mm"] = float(representative_ground[2])
            row["board_reference_u_min_px"] = float(np.min(selected_pixels[:, 0]))
            row["board_reference_u_max_px"] = float(np.max(selected_pixels[:, 0]))
            row["board_reference_v_min_px"] = float(np.min(selected_pixels[:, 1]))
            row["board_reference_v_max_px"] = float(np.max(selected_pixels[:, 1]))
            row["board_reference_x_g_min_mm"] = float(np.min(selected_ground_points[:, 0]))
            row["board_reference_x_g_max_mm"] = float(np.max(selected_ground_points[:, 0]))
            row["board_reference_y_g_min_mm"] = float(np.min(selected_ground_points[:, 1]))
            row["board_reference_y_g_max_mm"] = float(np.max(selected_ground_points[:, 1]))
            row["board_reference_z_g_min_mm"] = float(np.min(selected_ground_points[:, 2]))
            row["board_reference_z_g_max_mm"] = float(np.max(selected_ground_points[:, 2]))

            if str(c0_metadata["scan_axis"]) != "row":
                raise ValueError("board top/bottom range requires Daheng scan_axis=row")
            range_selection = _select_continuous_scan_points(
                range_pixels,
                str(c0_metadata["scan_axis"]),
            )
            if range_selection is None:
                raise ValueError("no continuous board range segment outside gauge heights")
            range_window_count = min(
                RANGE_ENDPOINT_WINDOW_COLUMNS,
                len(range_selection.points),
            )
            top_pixel = np.ascontiguousarray(
                np.median(range_selection.points[:range_window_count], axis=0),
                dtype=np.float64,
            )
            bottom_pixel = np.ascontiguousarray(
                np.median(range_selection.points[-range_window_count:], axis=0),
                dtype=np.float64,
            )
            top_point = _reconstruct_c0_point(
                top_pixel,
                calibration,
                effective_config.reconstruction,
            )
            bottom_point = _reconstruct_c0_point(
                bottom_pixel,
                calibration,
                effective_config.reconstruction,
            )
            top_ground = (
                session.R_camera_to_ground @ top_point
                + session.t_camera_to_ground_mm
            )
            bottom_ground = (
                session.R_camera_to_ground @ bottom_point
                + session.t_camera_to_ground_mm
            )
            delta_ground = bottom_ground - top_ground
            row["board_range_point_count"] = int(len(range_selection.points))
            row["board_range_top_u"] = float(top_pixel[0])
            row["board_range_top_v"] = float(top_pixel[1])
            row["board_range_top_X_mm"] = float(top_point[0])
            row["board_range_top_Y_mm"] = float(top_point[1])
            row["board_range_top_Z_mm"] = float(top_point[2])
            row["board_range_top_x_g_mm"] = float(top_ground[0])
            row["board_range_top_y_g_mm"] = float(top_ground[1])
            row["board_range_top_z_g_mm"] = float(top_ground[2])
            row["board_range_bottom_u"] = float(bottom_pixel[0])
            row["board_range_bottom_v"] = float(bottom_pixel[1])
            row["board_range_bottom_X_mm"] = float(bottom_point[0])
            row["board_range_bottom_Y_mm"] = float(bottom_point[1])
            row["board_range_bottom_Z_mm"] = float(bottom_point[2])
            row["board_range_bottom_x_g_mm"] = float(bottom_ground[0])
            row["board_range_bottom_y_g_mm"] = float(bottom_ground[1])
            row["board_range_bottom_z_g_mm"] = float(bottom_ground[2])
            row["board_range_delta_x_g_mm"] = float(delta_ground[0])
            row["board_range_delta_y_g_mm"] = float(delta_ground[1])
            row["board_range_delta_z_g_mm"] = float(delta_ground[2])
            row["board_range_y_g_span_mm"] = float(abs(delta_ground[1]))
            row["board_range_ground_xy_width_mm"] = float(
                np.linalg.norm(delta_ground[:2])
            )
            row["board_range_ground_3d_width_mm"] = float(
                np.linalg.norm(delta_ground)
            )
            row["board_range_top_window_count"] = range_window_count
            row["board_range_bottom_window_count"] = range_window_count
            signed_distance = _reference_plane_signed_distance(
                representative_point, session
            )
            row["board_reference_plane_signed_distance_mm"] = signed_distance
            row["board_reference_plane_distance_mm"] = abs(signed_distance)
            row["status"] = "ok"
            window_count = min(RIGHT_WINDOW_COLUMNS, len(selection.points))
            window_flags = np.zeros(len(selection.points), dtype=bool)
            window_flags[-window_count:] = True
            for sample_index, (pixel, camera_point, ground_point) in enumerate(
                zip(selected_pixels, selected_camera_points, selected_ground_points)
            ):
                coverage_rows.append(
                    {
                        "frame": filename,
                        "camera_frame_number": row["camera_frame_number"],
                        "sample_index": sample_index,
                        "u_px": float(pixel[0]),
                        "v_px": float(pixel[1]),
                        "X_camera_mm": float(camera_point[0]),
                        "Y_camera_mm": float(camera_point[1]),
                        "Z_camera_mm": float(camera_point[2]),
                        "x_g_mm": float(ground_point[0]),
                        "y_g_mm": float(ground_point[1]),
                        "z_g_mm": float(ground_point[2]),
                        "is_representative_window": bool(window_flags[sample_index]),
                    }
                )
            visuals[filename] = {
                "image": image,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "valid_pixels": valid_pixels,
                "board_pixels": board_pixels,
                "baseline_pixels": selected_pixels,
                "representative_pixel": representative_pixel,
                "representative_window": representative_window,
                "range_top_pixel": top_pixel,
                "range_bottom_pixel": bottom_pixel,
            }
        except Exception as error:
            row["status"] = f"frame_error:{type(error).__name__}:{error}"
        rows.append(row)

    status_counts = Counter(str(row["status"]) for row in rows)
    valid_rows = [row for row in rows if row["status"] == "ok"]
    plane_rows = [
        row
        for row in valid_rows
        if isinstance(row.get("board_reference_plane_distance_mm"), (int, float))
        and np.isfinite(float(row["board_reference_plane_distance_mm"]))
    ]
    within_limit = sum(
        float(row["board_reference_plane_distance_mm"]) <= BOARD_PLANE_SANITY_LIMIT_MM
        for row in plane_rows
    )

    warnings: list[str] = []
    camera_config = getattr(effective_config, "camera", None)
    config_exposure = (
        None if camera_config is None else float(getattr(camera_config, "exposure_us", np.nan))
    )
    if (
        config_exposure is not None
        and np.isfinite(config_exposure)
        and observed_exposure
        and any(abs(value - config_exposure) > 1.0e-9 for value in observed_exposure)
    ):
        warnings.append(
            f"recording exposure_us={sorted(observed_exposure)} differs from config "
            f"camera.exposure_us={config_exposure}; recorded metadata was retained."
        )
    if c0_metadata["base_config_correction_mode"].lower() != "none" or c0_metadata["base_config_c1_enabled"]:
        warnings.append(
            "Current Daheng online config defaults to Quadratic+C1/H1; this audit explicitly "
            "used the existing Daheng circular-cone C0 model with correction none and C1 disabled."
        )
    session_runtime_status = str(session.document.get("session_ground_reference_status") or "")
    if session_runtime_status != "VALID":
        warnings.append(
            f"Session JSON session_ground_reference_status={session_runtime_status!r}; "
            "the valid session_extrinsic and stored board polygon were used."
        )
    if relation.get("recording_listed") is True and relation.get("hot_session_sha256"):
        if relation["hot_session_sha256"] != session.sha256:
            warnings.append(
                "Existing thermal-A3 provenance lists this recording but references a different "
                f"hot Session SHA-256 ({relation['hot_session_sha256']}); calculation follows "
                "the Session JSON explicitly supplied by the user."
            )
    if len(valid_rows) != len(rows):
        warnings.append(
            f"{len(rows) - len(valid_rows)} frame(s) did not produce a valid board-reference representative; see CSV status."
        )

    overlay_frame: str | None = None
    if valid_rows:
        median_range = float(np.median([row["R_board_reference_mm"] for row in valid_rows]))
        overlay_frame = min(
            valid_rows,
            key=lambda row: abs(float(row["R_board_reference_mm"]) - median_range),
        )["frame"]

    coverage_frame_stats = {
        "centerline_pixel_count": _row_numeric_stats(
            valid_rows, "board_reference_laser_pixel_count"
        ),
        "board_polygon_pixel_count": _row_numeric_stats(
            valid_rows, "board_polygon_laser_pixel_count"
        ),
        "u_min_px": _row_numeric_stats(valid_rows, "board_reference_u_min_px"),
        "u_max_px": _row_numeric_stats(valid_rows, "board_reference_u_max_px"),
        "v_min_px": _row_numeric_stats(valid_rows, "board_reference_v_min_px"),
        "v_max_px": _row_numeric_stats(valid_rows, "board_reference_v_max_px"),
        "representative_x_g_mm": _row_numeric_stats(
            valid_rows, "board_reference_x_g_mm"
        ),
        "representative_y_g_mm": _row_numeric_stats(
            valid_rows, "board_reference_y_g_mm"
        ),
        "representative_z_g_mm": _row_numeric_stats(
            valid_rows, "board_reference_z_g_mm"
        ),
    }
    coverage_pooled_stats = {
        key: _numeric_stats([row[key] for row in coverage_rows])
        for key in ("u_px", "v_px", "x_g_mm", "y_g_mm", "z_g_mm")
    }
    board_range_stats = {
        "top_v_px": _row_numeric_stats(valid_rows, "board_range_top_v"),
        "bottom_v_px": _row_numeric_stats(valid_rows, "board_range_bottom_v"),
        "top_x_g_mm": _row_numeric_stats(valid_rows, "board_range_top_x_g_mm"),
        "top_y_g_mm": _row_numeric_stats(valid_rows, "board_range_top_y_g_mm"),
        "top_z_g_mm": _row_numeric_stats(valid_rows, "board_range_top_z_g_mm"),
        "bottom_x_g_mm": _row_numeric_stats(
            valid_rows, "board_range_bottom_x_g_mm"
        ),
        "bottom_y_g_mm": _row_numeric_stats(
            valid_rows, "board_range_bottom_y_g_mm"
        ),
        "bottom_z_g_mm": _row_numeric_stats(
            valid_rows, "board_range_bottom_z_g_mm"
        ),
        "delta_x_g_mm": _row_numeric_stats(valid_rows, "board_range_delta_x_g_mm"),
        "delta_y_g_mm": _row_numeric_stats(valid_rows, "board_range_delta_y_g_mm"),
        "delta_z_g_mm": _row_numeric_stats(valid_rows, "board_range_delta_z_g_mm"),
        "y_g_span_mm": _row_numeric_stats(valid_rows, "board_range_y_g_span_mm"),
        "ground_xy_width_mm": _row_numeric_stats(
            valid_rows, "board_range_ground_xy_width_mm"
        ),
        "ground_3d_width_mm": _row_numeric_stats(
            valid_rows, "board_range_ground_3d_width_mm"
        ),
    }

    summary: dict[str, Any] = {
        "schema_version": 2,
        "task": "WD-1-Daheng-board-only",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recording": {
            "path": recording_dir,
            "frames_csv": frames_csv.resolve(),
            "frames_csv_sha256": _sha256(frames_csv),
            "frame_count": len(frame_rows),
            "observed_exposure_us": sorted(observed_exposure),
            "observed_roi": [list(item) for item in sorted(observed_roi)],
        },
        "session": {
            "path": session.path,
            "sha256": session.sha256,
            "saved_at_utc": session.document.get("saved_at_utc"),
            "top_status": session.document.get("status"),
            "top_valid": session.document.get("valid"),
            "session_ground_reference_status": session_runtime_status,
            "session_ground_reference_nested_status": (
                session.document.get("session_ground_reference", {}).get("status")
            ),
            "ground_extrinsic_generation": session.ground_extrinsic_generation,
            "session_extrinsic": {
                "R_camera_to_ground": session.R_camera_to_ground,
                "t_camera_to_ground_mm": session.t_camera_to_ground_mm,
            },
            "board_mask": {
                "source": session.document.get("session_ground_reference", {})
                .get("support", {})
                .get("source"),
                "mask_mode": session.document.get("session_ground_reference", {})
                .get("support", {})
                .get("mask_mode"),
                "polygon_full_uv": session.polygon_full_uv,
            },
        },
        "reference_plane": {
            "equation_camera": "R_camera_to_ground[2] dot P_camera + t_camera_to_ground_mm[2] = 0",
            "normal_camera": session.plane_normal_camera,
            "normal_norm": session.plane_normal_norm,
            "D_mm": float(session.t_camera_to_ground_mm[2]),
            "WD_reference_mm": session.WD_reference_mm,
        },
        "selection_protocol": {
            "path": roi_protocol["path"],
            "sha256": roi_protocol["sha256"],
            "objects": roi_protocol["objects"],
            "baseline_intervals_v": roi_protocol["baseline_intervals_v"],
            "gauge_intervals_v_excluded": roi_protocol["gauge_intervals_v"],
            "board_polygon_intersection": True,
        },
        "relation": relation,
        "provenance": {
            "config_path": config_path,
            "config_sha256": _sha256(config_path),
            "base_manifest_sha256": c0_metadata["base_manifest_sha256"],
            "base_package_id": c0_metadata["base_package_id"],
            "c0_model_path": c0_metadata["c0_model_path"],
            "c0_model_sha256": c0_metadata["c0_model_sha256"],
            "laser_model_type": laser_model.get("model_type"),
            "laser_model": laser_model,
            "correction_mode_used": "none",
            "c1_used": False,
            "h1_used": False,
            "hb2_used": False,
            "image_right_calculated": False,
            "reused_numeric_artifacts": [],
            "reused_implementations": [
                "FramePipeline.run_frame",
                "extract_laser_center / production Steger backend",
                "reconstruct_uv_to_ground",
                "points_from_valid_columns",
                "measurement.board_mask._points_inside_convex_polygon",
            ],
            "new_computations": [
                f"{len(frame_rows)} target PNG frame extractions",
                "board polygon and frozen baseline-interval intersection",
                "explicit gauge-height interval exclusion",
                "board-reference median-pixel C0 reconstructions and statistics",
                "per-pixel board Steger centerline coverage and ground-coordinate table",
                "full board top/bottom endpoint windows and ground-width statistics",
            ],
        },
        "method": {
            "scan_axis": c0_metadata["scan_axis"],
            "continuity_max_column_gap": CONTINUITY_MAX_COLUMN_GAP,
            "continuity_max_vertical_jump_px": CONTINUITY_MAX_VERTICAL_JUMP,
            "continuity_min_segment_columns": CONTINUITY_MIN_COLUMNS,
            "representative_window_columns": RIGHT_WINDOW_COLUMNS,
            "range_endpoint_window_columns": RANGE_ENDPOINT_WINDOW_COLUMNS,
            "representative_definition": "median of all retained continuous board-baseline samples; trailing window retained for overlay traceability",
            "range_endpoint_definition": "median pixel in the first/last accepted continuous board-centerline windows ordered by v; gauge height intervals excluded",
            "board_mask_coordinates": "full-sensor (u,v)",
            "range_formula": "sqrt(X_camera^2 + Y_camera^2 + Z_camera^2)",
            "ground_coordinate_definition": "(x_g,y_g,z_g) = FrameResult.points_ground[:,0:3] = R_camera_to_ground @ (X_camera,Y_camera,Z_camera) + t_camera_to_ground_mm",
            "pixel_coverage_definition": "production Steger subpixel centerline samples retained inside the Session board polygon and six baseline intervals; gauge height intervals excluded",
        },
        "coverage": {
            "definition": "Steger centerline samples, not raw bright-stripe area pixels",
            "region": "Session board polygon intersected with frozen board baseline intervals; three gauge height intervals excluded",
            "centerline_sample_count_total": len(coverage_rows),
            "per_frame_stats": coverage_frame_stats,
            "pooled_sample_stats": coverage_pooled_stats,
        },
        "board_range": {
            "definition": "top/bottom endpoints of the continuous Steger centerline inside the Session board polygon, after excluding gauge height intervals",
            "endpoint_window_columns": RANGE_ENDPOINT_WINDOW_COLUMNS,
            "width_interpretation": "abs(delta_y_g) is the y_g-axis span; ground_xy_width is the perspective-robust in-plane endpoint distance",
            "per_frame_stats": board_range_stats,
        },
        "results": {
            "stats": {
                "WD_reference_mm": _stats(rows, "WD_reference_mm"),
                "R_board_reference_mm": _stats(rows, "R_board_reference_mm"),
                "board_reference_plane_distance_mm": _stats(
                    rows, "board_reference_plane_distance_mm"
                ),
                "board_reference_Z_depth_mm": _stats(rows, "board_reference_Z_depth_mm"),
            }
        },
        "quality": {
            "valid_board_reference_frame_count": len(valid_rows),
            "status_counts": dict(status_counts),
            "board_plane_valid_count": len(plane_rows),
            "board_plane_within_sanity_limit_count": within_limit,
        },
        "warnings": warnings,
        "overlay": {
            "frame": overlay_frame,
            "coordinate_space": "source ROI image; polygon and labels use full pixel coordinates",
        },
    }

    csv_path = output_dir / "working_distance_frames.csv"
    summary_path = output_dir / "working_distance_summary.json"
    report_path = output_dir / "working_distance_report.md"
    overlay_path = output_dir / "working_distance_overlay.png"
    coverage_path = output_dir / "board_laser_coverage.csv"
    _write_csv(csv_path, rows)
    _write_coverage_csv(coverage_path, coverage_rows)
    if overlay_frame is not None and overlay_frame in visuals:
        _draw_overlay(
            overlay_path,
            visuals[overlay_frame],
            session,
            next(row for row in rows if row["frame"] == overlay_frame),
            roi_protocol,
        )
    else:
        overlay_path.write_bytes(b"")
        summary["warnings"].append("No valid board-reference frame was available for overlay rendering.")
    summary["artifacts"] = {
        "working_distance_frames_csv": csv_path.resolve(),
        "working_distance_summary_json": summary_path.resolve(),
        "working_distance_report_md": report_path.resolve(),
        "working_distance_overlay_png": overlay_path.resolve(),
        "board_laser_coverage_csv": coverage_path.resolve(),
    }
    _write_json(summary_path, summary)
    _write_report(report_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "frame_count": len(rows),
                "valid_board_reference_frame_count": len(valid_rows),
                "stats": summary["results"]["stats"],
                "coverage": summary["coverage"],
                "warnings": summary["warnings"],
            },
            ensure_ascii=False,
        )
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--c0-model", type=Path, default=DEFAULT_C0_MODEL)
    parser.add_argument("--roi-manifest", type=Path, default=DEFAULT_ROI_MANIFEST)
    parser.add_argument(
        "--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_audit(
        args.recording,
        args.session,
        args.config,
        args.c0_model,
        args.roi_manifest,
        args.analysis_manifest,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
