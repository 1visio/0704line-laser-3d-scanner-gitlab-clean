#!/usr/bin/env python3
"""WD-1: audit camera working distance and right-edge laser ranges.

This script delegates image processing and geometry to the existing production
pipeline. It only adds audit-specific segment selection, endpoint windows,
statistics, and report rendering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
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
from laser.realtime_steger import points_from_valid_columns  # noqa: E402
from measurement.board_mask import _points_inside_convex_polygon  # noqa: E402
from online.models import CapturedFrame  # noqa: E402
from online.pipeline import FramePipeline  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402
from utils.image_io import load_grayscale_image  # noqa: E402


DEFAULT_RECORDING = (
    TOOL_ROOT
    / "output_haikang_0828"
    / "online_recordings"
    / "recording_20260829_104617"
)
DEFAULT_SESSION = (
    TOOL_ROOT / "output_haikang_0828" / "session_ground_calibration.json"
)
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_haikang_0828.yaml"

RIGHT_WINDOW_COLUMNS = 60
BOARD_RANGE_ENDPOINT_WINDOW_COLUMNS = 60
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
    "board_right_u",
    "board_right_v",
    "board_right_X_mm",
    "board_right_Y_mm",
    "board_right_Z_mm",
    "board_right_Z_depth_mm",
    "R_board_right_mm",
    "board_right_reference_plane_signed_distance_mm",
    "board_right_reference_plane_distance_mm",
    "board_range_segment_length",
    "board_range_start_u",
    "board_range_start_v",
    "board_range_start_X_mm",
    "board_range_start_Y_mm",
    "board_range_start_Z_mm",
    "board_range_start_x_g_mm",
    "board_range_start_y_g_mm",
    "board_range_start_z_g_mm",
    "board_range_end_u",
    "board_range_end_v",
    "board_range_end_X_mm",
    "board_range_end_Y_mm",
    "board_range_end_Z_mm",
    "board_range_end_x_g_mm",
    "board_range_end_y_g_mm",
    "board_range_end_z_g_mm",
    "board_range_delta_x_g_mm",
    "board_range_delta_y_g_mm",
    "board_range_delta_z_g_mm",
    "board_range_y_g_span_mm",
    "board_range_ground_xy_width_mm",
    "board_range_ground_3d_width_mm",
    "board_range_start_window_count",
    "board_range_end_window_count",
    "board_right_segment_start_u",
    "board_right_segment_end_u",
    "board_right_segment_length",
    "board_right_window_count",
    "image_right_u",
    "image_right_v",
    "image_right_X_mm",
    "image_right_Y_mm",
    "image_right_Z_mm",
    "image_right_Z_depth_mm",
    "R_image_right_mm",
    "image_right_segment_start_u",
    "image_right_segment_end_u",
    "image_right_segment_length",
    "image_right_window_count",
    "steger_point_count",
    "c0_valid_point_count",
    "board_mask_point_count",
    "image_candidate_point_count",
    "image_raw_segment_count",
    "image_accepted_segment_count",
    "board_candidate_point_count",
    "board_raw_segment_count",
    "board_accepted_segment_count",
    "reconstruction_filtered",
    "status",
]


@dataclass(frozen=True, slots=True)
class SessionContract:
    document: dict[str, Any]
    path: Path
    sha256: str
    R_camera_to_ground: np.ndarray
    t_camera_to_ground_mm: np.ndarray
    plane_normal_camera: np.ndarray
    plane_normal_norm: float
    WD_reference_mm: float
    ground_extrinsic_generation: int
    polygon_full_uv: np.ndarray


@dataclass(frozen=True, slots=True)
class SegmentSelection:
    points: np.ndarray
    candidate_point_count: int
    raw_segment_count: int
    accepted_segment_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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
            writer.writerow(
                {field: _csv_value(row.get(field)) for field in CSV_FIELDS}
            )


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _int_value(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    return parsed


def _load_session_contract(path: Path) -> SessionContract:
    if not path.is_file():
        raise FileNotFoundError(f"Session JSON does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "VALID" or document.get("valid") is not True:
        raise ValueError("Session JSON top-level status/valid is not VALID/true")

    reference = document.get("session_ground_reference") or {}
    if reference.get("status") != "VALID":
        raise ValueError("session_ground_reference.status is not VALID")
    support = reference.get("support") or {}
    polygon = np.asarray(support.get("polygon_full_uv"), dtype=np.float64)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError(
            "Session JSON support.polygon_full_uv must be a finite 4x2 polygon"
        )

    extrinsic = document.get("session_extrinsic") or {}
    rotation = np.asarray(
        extrinsic.get("R_camera_to_ground"), dtype=np.float64
    )
    translation = np.asarray(
        extrinsic.get("t_camera_to_ground_mm"), dtype=np.float64
    ).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Session JSON session_extrinsic R/t has invalid shape")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Session JSON session_extrinsic contains non-finite values")

    transform = np.asarray(extrinsic.get("T_ground_from_camera"), dtype=np.float64)
    if transform.shape == (4, 4):
        if not np.allclose(transform[:3, :3], rotation, atol=1.0e-10):
            raise ValueError("Session JSON R disagrees with T_ground_from_camera")
        if not np.allclose(transform[:3, 3], translation, atol=1.0e-10):
            raise ValueError("Session JSON t disagrees with T_ground_from_camera")

    plane_normal = np.ascontiguousarray(rotation[2], dtype=np.float64)
    plane_normal_norm = float(np.linalg.norm(plane_normal))
    if plane_normal_norm <= np.finfo(np.float64).eps:
        raise ValueError("Session reference-plane normal is degenerate")

    runtime = document.get("runtime") or {}
    generation_value = runtime.get(
        "ground_extrinsic_generation",
        reference.get("ground_extrinsic_generation"),
    )
    generation = _int_value(generation_value, "ground_extrinsic_generation")
    if generation < 0:
        raise ValueError("ground_extrinsic_generation must be non-negative")

    wd_reference = abs(float(translation[2])) / plane_normal_norm
    return SessionContract(
        document=document,
        path=path.resolve(),
        sha256=_sha256(path),
        R_camera_to_ground=np.ascontiguousarray(rotation),
        t_camera_to_ground_mm=np.ascontiguousarray(translation),
        plane_normal_camera=plane_normal,
        plane_normal_norm=plane_normal_norm,
        WD_reference_mm=wd_reference,
        ground_extrinsic_generation=generation,
        polygon_full_uv=np.ascontiguousarray(polygon),
    )


def _read_frame_rows(frames_csv: Path) -> list[dict[str, str]]:
    if not frames_csv.is_file():
        raise FileNotFoundError(f"frames.csv does not exist: {frames_csv}")
    with frames_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("frames.csv has no frame rows")
    required = {
        "filename",
        "camera_frame_number",
        "host_timestamp_ns",
        "offset_x",
        "offset_y",
        "width",
        "height",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"frames.csv is missing fields: {sorted(missing)}")
    return rows


def _select_rightmost_segment(
    points: np.ndarray,
    *,
    max_column_gap: int = CONTINUITY_MAX_COLUMN_GAP,
    max_vertical_jump: float = CONTINUITY_MAX_VERTICAL_JUMP,
    min_columns: int = CONTINUITY_MIN_COLUMNS,
) -> SegmentSelection | None:
    candidates = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(candidates):
        return None
    valid = np.isfinite(candidates).all(axis=1)
    accepted, metadata = points_from_valid_columns(
        candidates[:, 0],
        candidates[:, 1],
        valid,
        max_column_gap,
        max_vertical_jump,
        min_columns,
    )
    if not len(accepted):
        return None

    breaks = np.where(
        (np.diff(accepted[:, 0]) > max_column_gap)
        | (np.abs(np.diff(accepted[:, 1])) > max_vertical_jump)
    )[0] + 1
    segments = [
        accepted[index]
        for index in np.split(np.arange(len(accepted)), breaks)
        if len(index) >= min_columns
    ]
    if not segments:
        return None
    selected = max(
        segments,
        key=lambda segment: (float(segment[-1, 0]), len(segment)),
    )
    return SegmentSelection(
        points=np.ascontiguousarray(selected, dtype=np.float64),
        candidate_point_count=int(metadata["candidate_point_count"]),
        raw_segment_count=int(metadata["raw_segment_count"]),
        accepted_segment_count=int(metadata["accepted_segment_count"]),
    )


def _representative_pixel(
    selection: SegmentSelection,
    window_columns: int = RIGHT_WINDOW_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    window = selection.points[-min(window_columns, len(selection.points)) :]
    return (
        np.ascontiguousarray(np.median(window, axis=0), dtype=np.float64),
        np.ascontiguousarray(window, dtype=np.float64),
    )


def _reference_plane_signed_distance(
    point_camera: np.ndarray,
    session: SessionContract,
) -> float:
    point = np.asarray(point_camera, dtype=np.float64).reshape(3)
    signed = (
        float(point @ session.plane_normal_camera)
        + float(session.t_camera_to_ground_mm[2])
    ) / session.plane_normal_norm
    return signed


def _reconstruct_c0_point(
    pixel_uv: np.ndarray,
    calibration: dict[str, object],
    reconstruction_params: Any,
) -> np.ndarray:
    reconstruction = reconstruct_uv_to_ground(
        np.asarray(pixel_uv, dtype=np.float64).reshape(1, 2),
        calibration,
        reconstruction_params,
    )
    if len(reconstruction.points_camera) != 1:
        raise ValueError(
            "representative pixel did not produce one valid C0 reconstruction"
        )
    points_c0 = reconstruction.points_camera_c0
    if points_c0 is None or len(points_c0) != 1:
        raise ValueError("representative reconstruction did not retain C0 point")
    return np.ascontiguousarray(np.asarray(points_c0[0], dtype=np.float64))


def _fill_endpoint(
    row: dict[str, Any],
    label: str,
    selection: SegmentSelection | None,
    calibration: dict[str, object],
    reconstruction_params: Any,
    session: SessionContract,
) -> dict[str, Any] | None:
    if selection is None:
        return None
    pixel, window = _representative_pixel(selection)
    point = _reconstruct_c0_point(pixel, calibration, reconstruction_params)
    range_mm = float(np.linalg.norm(point))
    row[f"{label}_u"] = float(pixel[0])
    row[f"{label}_v"] = float(pixel[1])
    row[f"{label}_X_mm"] = float(point[0])
    row[f"{label}_Y_mm"] = float(point[1])
    row[f"{label}_Z_mm"] = float(point[2])
    row[f"{label}_Z_depth_mm"] = float(point[2])
    row[f"R_{label}_mm"] = range_mm
    row[f"{label}_segment_start_u"] = float(selection.points[0, 0])
    row[f"{label}_segment_end_u"] = float(selection.points[-1, 0])
    row[f"{label}_segment_length"] = int(len(selection.points))
    row[f"{label}_window_count"] = int(len(window))
    row[f"{label}_candidate_point_count"] = int(selection.candidate_point_count)
    row[f"{label}_raw_segment_count"] = int(selection.raw_segment_count)
    row[f"{label}_accepted_segment_count"] = int(selection.accepted_segment_count)
    if label == "board_right":
        signed_distance = _reference_plane_signed_distance(point, session)
        row["board_right_reference_plane_signed_distance_mm"] = signed_distance
        row["board_right_reference_plane_distance_mm"] = abs(signed_distance)
    return {
        "pixel": pixel,
        "window": window,
        "point_camera": point,
        "selection": selection,
    }


def _ground_point(point_camera: np.ndarray, session: SessionContract) -> np.ndarray:
    point = np.asarray(point_camera, dtype=np.float64).reshape(3)
    return np.ascontiguousarray(
        session.R_camera_to_ground @ point + session.t_camera_to_ground_mm,
        dtype=np.float64,
    )


def _fill_board_range(
    row: dict[str, Any],
    selection: SegmentSelection | None,
    calibration: dict[str, object],
    reconstruction_params: Any,
    session: SessionContract,
) -> dict[str, Any] | None:
    """Fill both endpoints of the board-mask laser span using robust windows."""
    if selection is None:
        return None
    window_count = min(
        BOARD_RANGE_ENDPOINT_WINDOW_COLUMNS,
        len(selection.points),
    )
    start_pixel = np.ascontiguousarray(
        np.median(selection.points[:window_count], axis=0),
        dtype=np.float64,
    )
    end_pixel = np.ascontiguousarray(
        np.median(selection.points[-window_count:], axis=0),
        dtype=np.float64,
    )
    start_point = _reconstruct_c0_point(
        start_pixel,
        calibration,
        reconstruction_params,
    )
    end_point = _reconstruct_c0_point(
        end_pixel,
        calibration,
        reconstruction_params,
    )
    start_ground = _ground_point(start_point, session)
    end_ground = _ground_point(end_point, session)
    delta_ground = end_ground - start_ground

    row["board_range_segment_length"] = int(len(selection.points))
    row["board_range_start_u"] = float(start_pixel[0])
    row["board_range_start_v"] = float(start_pixel[1])
    row["board_range_start_X_mm"] = float(start_point[0])
    row["board_range_start_Y_mm"] = float(start_point[1])
    row["board_range_start_Z_mm"] = float(start_point[2])
    row["board_range_start_x_g_mm"] = float(start_ground[0])
    row["board_range_start_y_g_mm"] = float(start_ground[1])
    row["board_range_start_z_g_mm"] = float(start_ground[2])
    row["board_range_end_u"] = float(end_pixel[0])
    row["board_range_end_v"] = float(end_pixel[1])
    row["board_range_end_X_mm"] = float(end_point[0])
    row["board_range_end_Y_mm"] = float(end_point[1])
    row["board_range_end_Z_mm"] = float(end_point[2])
    row["board_range_end_x_g_mm"] = float(end_ground[0])
    row["board_range_end_y_g_mm"] = float(end_ground[1])
    row["board_range_end_z_g_mm"] = float(end_ground[2])
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
    row["board_range_start_window_count"] = window_count
    row["board_range_end_window_count"] = window_count
    return {
        "start_pixel": start_pixel,
        "end_pixel": end_pixel,
        "start_point_camera": start_point,
        "end_point_camera": end_point,
        "start_point_ground": start_ground,
        "end_point_ground": end_ground,
        "selection": selection,
    }


def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
        and isinstance(row.get(key), (int, float, np.integer, np.floating))
        and np.isfinite(float(row[key]))
    ]
    if not values:
        return {
            "valid_frame_count": 0,
            "median_mm": None,
            "mean_mm": None,
            "std_mm": None,
            "p05_mm": None,
            "p95_mm": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "valid_frame_count": int(len(array)),
        "median_mm": float(np.median(array)),
        "mean_mm": float(np.mean(array)),
        "std_mm": float(np.std(array, ddof=0)),
        "p05_mm": float(np.percentile(array, 5)),
        "p95_mm": float(np.percentile(array, 95)),
    }


def _display_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    low, high = np.percentile(array.astype(np.float32), [1.0, 99.8])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.clip(array, 0, 255).astype(np.uint8)
    scaled = (array.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _draw_local_points(
    canvas: np.ndarray,
    points_full: np.ndarray,
    offset_x: int,
    offset_y: int,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    height, width = canvas.shape[:2]
    local = np.asarray(points_full, dtype=np.float64).copy()
    if not len(local):
        return
    local[:, 0] -= offset_x
    local[:, 1] -= offset_y
    for u, v in local:
        x = int(round(float(u)))
        y = int(round(float(v)))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_marker(
    canvas: np.ndarray,
    pixel_full: np.ndarray | None,
    offset_x: int,
    offset_y: int,
    color: tuple[int, int, int],
) -> None:
    if pixel_full is None:
        return
    x = int(round(float(pixel_full[0] - offset_x)))
    y = int(round(float(pixel_full[1] - offset_y)))
    cv2.drawMarker(
        canvas,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
        thickness=3,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(canvas, (x, y), 8, color, 2, lineType=cv2.LINE_AA)


def _write_overlay(
    path: Path,
    visual: dict[str, Any],
    session: SessionContract,
    row: dict[str, Any],
) -> None:
    image = visual["image"]
    offset_x = int(visual["offset_x"])
    offset_y = int(visual["offset_y"])
    canvas = cv2.cvtColor(_display_image(image), cv2.COLOR_GRAY2BGR)

    _draw_local_points(
        canvas,
        visual["valid_pixels"],
        offset_x,
        offset_y,
        (70, 210, 70),
        1,
    )
    polygon_local = np.asarray(session.polygon_full_uv, dtype=np.float64).copy()
    polygon_local[:, 0] -= offset_x
    polygon_local[:, 1] -= offset_y
    cv2.polylines(
        canvas,
        [np.round(polygon_local).astype(np.int32)],
        True,
        (0, 220, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    for key, color in (
        ("board_segment", (0, 80, 255)),
        ("image_segment", (255, 50, 190)),
    ):
        segment = visual.get(key)
        if segment is not None:
            _draw_local_points(canvas, segment, offset_x, offset_y, color, 2)

    board_endpoint = visual.get("board_endpoint")
    image_endpoint = visual.get("image_endpoint")
    _draw_marker(
        canvas,
        None if board_endpoint is None else board_endpoint["pixel"],
        offset_x,
        offset_y,
        (0, 80, 255),
    )
    board_range = visual.get("board_range")
    _draw_marker(
        canvas,
        None if board_range is None else board_range["start_pixel"],
        offset_x,
        offset_y,
        (255, 255, 0),
    )
    _draw_marker(
        canvas,
        None if image_endpoint is None else image_endpoint["pixel"],
        offset_x,
        offset_y,
        (255, 50, 190),
    )

    frame_name = str(row["frame"])
    lines = [
        f"WD-1 overlay | {frame_name} | full pixel coords",
        f"WD_reference = {row['WD_reference_mm']:.3f} mm",
        (
            f"board right = ({row.get('board_right_u', float('nan')):.2f}, "
            f"{row.get('board_right_v', float('nan')):.2f}) | "
            f"R = {row.get('R_board_right_mm', float('nan')):.3f} mm"
        ),
        (
            f"image right = ({row.get('image_right_u', float('nan')):.2f}, "
            f"{row.get('image_right_v', float('nan')):.2f}) | "
            f"R = {row.get('R_image_right_mm', float('nan')):.3f} mm"
        ),
        (
            f"board span y_g = {row.get('board_range_y_g_span_mm', float('nan')):.3f} mm | "
            f"ground XY = {row.get('board_range_ground_xy_width_mm', float('nan')):.3f} mm"
        ),
        (
            f"cyan/orange markers = board range start/end | "
            f"y_g={row.get('board_range_start_y_g_mm', float('nan')):.2f}/"
            f"{row.get('board_range_end_y_g_mm', float('nan')):.2f} mm"
        ),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (12, 22 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (12, 22 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("failed to encode overlay PNG")
    encoded.tofile(str(path))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _write_report(
    path: Path,
    summary: dict[str, Any],
) -> None:
    results = summary["results"]["stats"]
    warnings = summary["warnings"]
    wd = results["WD_reference_mm"]["median_mm"]
    board = results["R_board_right_mm"]["median_mm"]
    image = results["R_image_right_mm"]["median_mm"]
    board_plane = results["board_right_reference_plane_distance_mm"]
    board_range = summary["board_range"]["stats"]
    lines = [
        "# WD-1 | Haikang working distance and right-edge laser range audit",
        "",
        f"- Generated at: {summary['generated_at_utc']}",
        f"- Recording: {summary['recording']['path']}",
        f"- Processed frames: {summary['recording']['frame_count']}",
        f"- Complete two-endpoint frames: {summary['quality']['complete_frame_count']}",
        "",
        "## Conclusion",
        "",
        (
            f"当前海康视觉系统相机光学中心至棋盘格 Session 基准平面的法向工作距离为 "
            f"**{_fmt(wd, 1)} mm**。棋盘格有效区域最右侧激光测量点距相机光心 "
            f"**{_fmt(board, 1)} mm**；当前整幅有效激光视场最右侧实际测量点经 "
            f"circular-cone C0 恢复后距相机光心 **{_fmt(image, 1)} mm**。"
        ),
        "",
        (
            f"按棋盘格 mask 内连续激光段的两端定义，海康当前可观测棋盘格范围的 "
            f"`y_g` 跨度中位数为 **{_fmt(board_range['y_g_span_mm']['median_mm'], 2)} mm**，"
            f"ground XY 端点距离中位数为 **{_fmt(board_range['ground_xy_width_mm']['median_mm'], 2)} mm**。"
        ),
        "",
        "WD_reference 作为系统工作距离；R_board_right 和 R_image_right "
        "用于描述测量视场边缘的实际观测空间斜距。",
        "",
        "## Statistics (mm)",
        "",
        "| Metric | median | mean | std | P05 | P95 | valid frame count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("WD_reference", "WD_reference_mm"),
        ("R_board_right", "R_board_right_mm"),
        ("R_image_right", "R_image_right_mm"),
    ):
        stat = results[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt(stat["median_mm"]),
                    _fmt(stat["mean_mm"]),
                    _fmt(stat["std_mm"]),
                    _fmt(stat["p05_mm"]),
                    _fmt(stat["p95_mm"]),
                    str(stat["valid_frame_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Board-mask physical range",
            "",
            "棋盘格范围端点取 board mask 内连续激光段的起点/终点；每个端点使用 60 个有效列的中位数，再经现有 circular-cone C0 和 Session ground 外参恢复。",
            "`y_g span=abs(end_y_g-start_y_g)`；ground XY endpoint distance 同时包含 x_g 方向变化，更适合作为实际物理宽度。",
            "",
            "| Metric | median | mean | std | P05 | P95 | valid frame count |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| start y_g (mm) | {_fmt(board_range['start_y_g_mm']['median_mm'], 2)} | {_fmt(board_range['start_y_g_mm']['mean_mm'], 2)} | "
                f"{_fmt(board_range['start_y_g_mm']['std_mm'], 2)} | {_fmt(board_range['start_y_g_mm']['p05_mm'], 2)} | "
                f"{_fmt(board_range['start_y_g_mm']['p95_mm'], 2)} | {board_range['start_y_g_mm']['valid_frame_count']} |"
            ),
            (
                f"| end y_g (mm) | {_fmt(board_range['end_y_g_mm']['median_mm'], 2)} | {_fmt(board_range['end_y_g_mm']['mean_mm'], 2)} | "
                f"{_fmt(board_range['end_y_g_mm']['std_mm'], 2)} | {_fmt(board_range['end_y_g_mm']['p05_mm'], 2)} | "
                f"{_fmt(board_range['end_y_g_mm']['p95_mm'], 2)} | {board_range['end_y_g_mm']['valid_frame_count']} |"
            ),
            (
                f"| y_g span (mm) | {_fmt(board_range['y_g_span_mm']['median_mm'], 2)} | {_fmt(board_range['y_g_span_mm']['mean_mm'], 2)} | "
                f"{_fmt(board_range['y_g_span_mm']['std_mm'], 2)} | {_fmt(board_range['y_g_span_mm']['p05_mm'], 2)} | "
                f"{_fmt(board_range['y_g_span_mm']['p95_mm'], 2)} | {board_range['y_g_span_mm']['valid_frame_count']} |"
            ),
            (
                f"| ground XY endpoint width (mm) | {_fmt(board_range['ground_xy_width_mm']['median_mm'], 2)} | {_fmt(board_range['ground_xy_width_mm']['mean_mm'], 2)} | "
                f"{_fmt(board_range['ground_xy_width_mm']['std_mm'], 2)} | {_fmt(board_range['ground_xy_width_mm']['p05_mm'], 2)} | "
                f"{_fmt(board_range['ground_xy_width_mm']['p95_mm'], 2)} | {board_range['ground_xy_width_mm']['valid_frame_count']} |"
            ),
            (
                f"| ground 3D endpoint width (mm) | {_fmt(board_range['ground_3d_width_mm']['median_mm'], 2)} | {_fmt(board_range['ground_3d_width_mm']['mean_mm'], 2)} | "
                f"{_fmt(board_range['ground_3d_width_mm']['std_mm'], 2)} | {_fmt(board_range['ground_3d_width_mm']['p05_mm'], 2)} | "
                f"{_fmt(board_range['ground_3d_width_mm']['p95_mm'], 2)} | {board_range['ground_3d_width_mm']['valid_frame_count']} |"
            ),
            "",
            "## Geometry and call chain",
            "",
            "1. Original PNGs use the existing FramePipeline.run_frame production Steger path.",
            "2. Existing reconstruct_uv_to_ground performs undistorted camera-ray and "
            "circular-cone C0 intersection and returns camera XYZ. C1, H1, and H-B2 "
            "were not used.",
            "3. Existing continuity rules were used: max column gap 2, maximum vertical "
            "jump 14 px, and minimum segment length 42.",
            f"4. The last {RIGHT_WINDOW_COLUMNS} valid columns of the selected rightmost "
            "segment are represented by median (u,v), then reconstructed by the same C0 path.",
            "5. The board mask is the full-sensor polygon stored in "
            "session_ground_reference.support.polygon_full_uv.",
            "6. R_image_right uses C0 3D reconstruction only; no reference-plane ray "
            "intersection or plane extrapolation outside the board mask was used.",
            "",
            "## WD_reference definition",
            "",
            "The Session JSON session_extrinsic defines the reference plane in camera "
            "coordinates as R_camera_to_ground[2] dot P_camera + "
            "t_camera_to_ground_mm[2] = 0. The optical-center normal distance is "
            "abs(t_z) / norm(R_camera_to_ground[2]).",
            f"Computed value: {_fmt(wd, 6)} mm.",
            "",
            "## Board-plane sanity check",
            "",
            (
                "Absolute normal distance of the board-right C0 representative to the "
                f"reference plane: median={_fmt(board_plane['median_mm'])} mm, "
                f"mean={_fmt(board_plane['mean_mm'])} mm, "
                f"std={_fmt(board_plane['std_mm'])} mm, "
                f"P05/P95={_fmt(board_plane['p05_mm'])}/"
                f"{_fmt(board_plane['p95_mm'])} mm."
            ),
            (
                f"{summary['quality']['board_plane_within_sanity_limit_count']}/"
                f"{summary['quality']['board_plane_valid_count']} valid board-right "
                f"points are within the {BOARD_PLANE_SANITY_LIMIT_MM:.1f} mm sanity "
                "threshold. The threshold is diagnostic only and does not select points."
            ),
            "",
            "## Provenance and reuse audit",
            "",
            f"- Session JSON: {summary['session']['path']}",
            f"- Session JSON SHA-256: {summary['session']['sha256']}",
            f"- Haikang config SHA-256: {summary['provenance']['config_sha256']}",
            f"- Calibration manifest SHA-256: {summary['provenance']['manifest_sha256']}",
            f"- Recording frames.csv SHA-256: {summary['recording']['frames_csv_sha256']}",
            "- Numeric results were newly computed from the 20 target PNGs; no numeric "
            "result from another recording was reused.",
            "- Reused implementations are the production Steger, camera-ray/C0 "
            "reconstruction, Session extrinsic semantics, board polygon mask, and "
            "existing continuity helper.",
            "",
            "## Output files",
            "",
            "- working_distance_frames.csv",
            "- working_distance_summary.json",
            "- working_distance_report.md",
            (
                "- working_distance_overlay.png"
                + (
                    f" (representative frame: {summary['overlay']['frame']})"
                    if summary["overlay"]["frame"]
                    else ""
                )
            ),
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Session JSON note",
            "",
            (
                "The top-level status and session_extrinsic are usable, while "
                "session_ground_reference_status is "
                f"{summary['session']['session_ground_reference_status']}. "
                "WD-1 uses session_extrinsic and the stored polygon for this audit "
                "and does not apply the stale linear ground correction to camera XYZ."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    recording_dir: Path,
    session_path: Path,
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    recording_dir = recording_dir.resolve()
    session_path = session_path.resolve()
    config_path = config_path.resolve()
    if not recording_dir.is_dir():
        raise FileNotFoundError(f"recording directory does not exist: {recording_dir}")
    if output_dir is None:
        output_dir = recording_dir / "working_distance_audit"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _load_session_contract(session_path)
    config = load_app_config(config_path)
    if str(config.extraction_method).lower() != "steger":
        raise ValueError("WD-1 requires the current Haikang production method=steger")
    if str(config.correction.mode).lower() != "none":
        raise ValueError("WD-1 requires correction.mode=none")

    pipeline = FramePipeline(config)
    pipeline.apply_session_ground_extrinsic(
        session.R_camera_to_ground,
        session.t_camera_to_ground_mm,
        generation=session.ground_extrinsic_generation,
    )
    calibration = pipeline.calibration_for_reconstruction()
    laser_model = calibration.get("laser_model")
    if not isinstance(laser_model, dict) or laser_model.get("model_type") != "circular_cone":
        raise ValueError("WD-1 requires calibration laser_model.model_type=circular_cone")
    if bool(getattr(config.reconstruction, "enable_laser_ray_correction", False)):
        raise ValueError("WD-1 forbids laser ray correction/C1")

    frames_csv = recording_dir / "frames.csv"
    frame_rows = _read_frame_rows(frames_csv)
    rows: list[dict[str, Any]] = []
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
                raise ValueError(
                    f"image shape {image.shape} != frames.csv {expected_shape}"
                )
            offset_x = _int_value(source_row["offset_x"], "offset_x")
            offset_y = _int_value(source_row["offset_y"], "offset_y")
            observed_roi.add(
                (offset_x, offset_y, expected_shape[1], expected_shape[0])
            )
            exposure_raw = source_row.get("exposure_us")
            if exposure_raw not in (None, ""):
                observed_exposure.add(
                    _finite_float(exposure_raw, "exposure_us")
                )

            captured = CapturedFrame(
                image=image,
                camera_frame_number=row["camera_frame_number"],
                camera_timestamp_ticks=(
                    None
                    if source_row.get("camera_timestamp_ticks") in (None, "")
                    else _int_value(
                        source_row["camera_timestamp_ticks"],
                        "camera_timestamp_ticks",
                    )
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
            # FrameResult exposes points_camera, while ReconstructionResult
            # retains points_camera_c0 internally. C1 is explicitly disabled
            # above, so the production FrameResult camera points are C0 points.
            c0_points = np.asarray(result.points_camera, dtype=np.float64).reshape(-1, 3)
            if len(valid_pixels) != len(c0_points):
                raise ValueError("pixels_uv and C0 camera points are not aligned")

            row["steger_point_count"] = int(len(centers))
            row["c0_valid_point_count"] = int(len(c0_points))
            row["reconstruction_filtered"] = dict(result.filtered)

            image_selection = _select_rightmost_segment(valid_pixels)
            board_mask = _points_inside_convex_polygon(
                valid_pixels, session.polygon_full_uv
            )
            board_pixels = valid_pixels[board_mask]
            row["board_mask_point_count"] = int(len(board_pixels))
            board_selection = _select_rightmost_segment(board_pixels)

            if image_selection is not None:
                row["image_candidate_point_count"] = int(image_selection.candidate_point_count)
                row["image_raw_segment_count"] = int(image_selection.raw_segment_count)
                row["image_accepted_segment_count"] = int(image_selection.accepted_segment_count)
            if board_selection is not None:
                row["board_candidate_point_count"] = int(board_selection.candidate_point_count)
                row["board_raw_segment_count"] = int(board_selection.raw_segment_count)
                row["board_accepted_segment_count"] = int(board_selection.accepted_segment_count)

            board_endpoint = _fill_endpoint(
                row,
                "board_right",
                board_selection,
                calibration,
                config.reconstruction,
                session,
            )
            board_range = _fill_board_range(
                row,
                board_selection,
                calibration,
                config.reconstruction,
                session,
            )
            image_endpoint = _fill_endpoint(
                row,
                "image_right",
                image_selection,
                calibration,
                config.reconstruction,
                session,
            )
            errors: list[str] = []
            if image_selection is None:
                errors.append("image_right_no_continuous_segment")
            if board_selection is None:
                errors.append("board_right_no_continuous_segment")
            if board_range is None:
                errors.append("board_range_no_continuous_segment")
            if image_selection is not None and image_endpoint is None:
                errors.append("image_right_c0_reconstruction_invalid")
            if board_selection is not None and board_endpoint is None:
                errors.append("board_right_c0_reconstruction_invalid")
            if board_selection is not None and board_range is None:
                errors.append("board_range_c0_reconstruction_invalid")
            row["status"] = "ok" if not errors else ";".join(errors)
            visuals[filename] = {
                "image": image,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "valid_pixels": valid_pixels,
                "board_segment": (
                    None
                    if board_selection is None
                    else board_selection.points
                ),
                "image_segment": (
                    None
                    if image_selection is None
                    else image_selection.points
                ),
                "board_endpoint": board_endpoint,
                "board_range": board_range,
                "image_endpoint": image_endpoint,
            }
        except Exception as error:
            row["status"] = f"frame_error:{type(error).__name__}:{error}"
        rows.append(row)

    status_counts = Counter(str(row["status"]) for row in rows)
    complete_rows = [row for row in rows if row["status"] == "ok"]
    board_plane_rows = [
        row
        for row in rows
        if isinstance(
            row.get("board_right_reference_plane_distance_mm"),
            (int, float, np.integer, np.floating),
        )
        and np.isfinite(float(row["board_right_reference_plane_distance_mm"]))
    ]
    within_limit = sum(
        float(row["board_right_reference_plane_distance_mm"])
        <= BOARD_PLANE_SANITY_LIMIT_MM
        for row in board_plane_rows
    )

    camera_config = getattr(config, "camera", None)
    config_exposure = (
        None
        if camera_config is None
        else float(getattr(camera_config, "exposure_us", np.nan))
    )
    warnings: list[str] = []
    if (
        config_exposure is not None
        and np.isfinite(config_exposure)
        and observed_exposure
        and any(abs(value - config_exposure) > 1.0e-9 for value in observed_exposure)
    ):
        warnings.append(
            "recording frames.csv exposure_us="
            f"{sorted(observed_exposure)} differs from config camera.exposure_us="
            f"{config_exposure}; recorded PNG metadata was retained."
        )
    session_runtime_status = str(
        session.document.get("session_ground_reference_status") or ""
    )
    if session_runtime_status != "VALID":
        warnings.append(
            "Session JSON session_ground_reference_status="
            f"{session_runtime_status!r}; WD-1 used session_extrinsic and stored "
            "polygon only, and did not apply the stale linear ground correction."
        )
    if len(complete_rows) != len(rows):
        warnings.append(
            f"{len(rows) - len(complete_rows)} frame(s) did not produce both "
            "right-edge representatives; see CSV status."
        )

    overlay_frame: str | None = None
    if complete_rows:
        median_image_range = float(
            np.median([row["R_image_right_mm"] for row in complete_rows])
        )
        overlay_frame = min(
            complete_rows,
            key=lambda row: abs(
                float(row["R_image_right_mm"]) - median_image_range
            ),
        )["frame"]
    elif visuals:
        overlay_frame = next(iter(visuals))

    board_range_stats = {
        "start_u_px": _stats(rows, "board_range_start_u"),
        "start_v_px": _stats(rows, "board_range_start_v"),
        "end_u_px": _stats(rows, "board_range_end_u"),
        "end_v_px": _stats(rows, "board_range_end_v"),
        "start_x_g_mm": _stats(rows, "board_range_start_x_g_mm"),
        "start_y_g_mm": _stats(rows, "board_range_start_y_g_mm"),
        "start_z_g_mm": _stats(rows, "board_range_start_z_g_mm"),
        "end_x_g_mm": _stats(rows, "board_range_end_x_g_mm"),
        "end_y_g_mm": _stats(rows, "board_range_end_y_g_mm"),
        "end_z_g_mm": _stats(rows, "board_range_end_z_g_mm"),
        "delta_x_g_mm": _stats(rows, "board_range_delta_x_g_mm"),
        "delta_y_g_mm": _stats(rows, "board_range_delta_y_g_mm"),
        "delta_z_g_mm": _stats(rows, "board_range_delta_z_g_mm"),
        "y_g_span_mm": _stats(rows, "board_range_y_g_span_mm"),
        "ground_xy_width_mm": _stats(rows, "board_range_ground_xy_width_mm"),
        "ground_3d_width_mm": _stats(rows, "board_range_ground_3d_width_mm"),
    }

    summary: dict[str, Any] = {
        "schema_version": 2,
        "task": "WD-1",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recording": {
            "path": str(recording_dir),
            "frames_csv": str(frames_csv.resolve()),
            "frames_csv_sha256": _sha256(frames_csv),
            "frame_count": len(frame_rows),
            "observed_exposure_us": sorted(observed_exposure),
            "observed_roi": [list(item) for item in sorted(observed_roi)],
        },
        "session": {
            "path": str(session.path),
            "sha256": session.sha256,
            "saved_at_utc": session.document.get("saved_at_utc"),
            "top_status": session.document.get("status"),
            "top_valid": session.document.get("valid"),
            "session_ground_reference_status": session_runtime_status,
            "session_ground_reference_nested_status": (
                session.document.get("session_ground_reference", {}).get("status")
            ),
            "ground_extrinsic_generation": session.ground_extrinsic_generation,
            "session_generation": (
                session.document.get("frame", {}).get("session_generation")
            ),
            "session_extrinsic": {
                "R_camera_to_ground": session.R_camera_to_ground,
                "t_camera_to_ground_mm": session.t_camera_to_ground_mm,
            },
            "board_mask": {
                "source": (
                    session.document.get("session_ground_reference", {})
                    .get("support", {})
                    .get("source")
                ),
                "mask_mode": (
                    session.document.get("session_ground_reference", {})
                    .get("support", {})
                    .get("mask_mode")
                ),
                "polygon_full_uv": session.polygon_full_uv,
            },
        },
        "reference_plane": {
            "equation_camera": (
                "R_camera_to_ground[2] dot P_camera + "
                "t_camera_to_ground_mm[2] = 0"
            ),
            "normal_camera": session.plane_normal_camera,
            "normal_norm": session.plane_normal_norm,
            "D_mm": float(session.t_camera_to_ground_mm[2]),
            "WD_reference_mm": session.WD_reference_mm,
        },
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "manifest_sha256": pipeline.package.manifest_sha256,
            "calibration_package_id": pipeline.package.package_id,
            "algorithm_config_sha256": pipeline.algorithm_config_sha256,
            "laser_model_type": laser_model.get("model_type"),
            "laser_model": laser_model,
            "correction_mode": str(config.correction.mode),
            "c1_used": False,
            "h1_used": False,
            "hb2_used": False,
            "image_right_plane_intersection_used": False,
            "reused_numeric_artifacts": [],
            "reused_implementations": [
                "FramePipeline.run_frame",
                "extract_laser_center / production Steger backend",
                "reconstruct_uv_to_ground",
                "points_from_valid_columns",
                "measurement.board_mask._points_inside_convex_polygon",
            ],
            "new_computations": [
                "20 target PNG frame extractions",
                "rightmost continuous-segment selection",
                "60-column robust endpoint representatives",
                "C0 endpoint reconstructions and range statistics",
                "board-mask start/end endpoint windows and ground-width statistics",
            ],
        },
        "method": {
            "continuity_max_column_gap": CONTINUITY_MAX_COLUMN_GAP,
            "continuity_max_vertical_jump_px": CONTINUITY_MAX_VERTICAL_JUMP,
            "continuity_min_segment_columns": CONTINUITY_MIN_COLUMNS,
            "right_endpoint_window_columns": RIGHT_WINDOW_COLUMNS,
            "board_range_endpoint_window_columns": BOARD_RANGE_ENDPOINT_WINDOW_COLUMNS,
            "board_mask_coordinates": "full-sensor (u,v)",
            "range_formula": "sqrt(X_camera^2 + Y_camera^2 + Z_camera^2)",
        },
        "board_range": {
            "definition": "start/end endpoints of the selected continuous laser segment inside the Session board polygon",
            "endpoint_window_columns": BOARD_RANGE_ENDPOINT_WINDOW_COLUMNS,
            "width_interpretation": "abs(delta_y_g) is the y_g-axis span; ground_xy_width is the in-plane endpoint distance",
            "stats": board_range_stats,
        },
        "results": {
            "stats": {
                "WD_reference_mm": _stats(rows, "WD_reference_mm"),
                "R_board_right_mm": _stats(rows, "R_board_right_mm"),
                "R_image_right_mm": _stats(rows, "R_image_right_mm"),
                "board_right_reference_plane_distance_mm": _stats(
                    rows, "board_right_reference_plane_distance_mm"
                ),
                "image_right_Z_depth_mm": _stats(rows, "image_right_Z_depth_mm"),
                "board_right_Z_depth_mm": _stats(rows, "board_right_Z_depth_mm"),
            }
        },
        "quality": {
            "complete_frame_count": len(complete_rows),
            "status_counts": dict(status_counts),
            "board_plane_valid_count": len(board_plane_rows),
            "board_plane_within_sanity_limit_count": within_limit,
        },
        "warnings": warnings,
        "overlay": {
            "frame": overlay_frame,
            "coordinate_space": "source ROI image; polygon and endpoint labels use full pixel coordinates",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "working_distance_frames.csv"
    summary_path = output_dir / "working_distance_summary.json"
    report_path = output_dir / "working_distance_report.md"
    overlay_path = output_dir / "working_distance_overlay.png"
    _write_csv(csv_path, rows)
    if overlay_frame is not None and overlay_frame in visuals:
        _write_overlay(
            overlay_path,
            visuals[overlay_frame],
            session,
            next(row for row in rows if row["frame"] == overlay_frame),
        )
    else:
        overlay_path.write_bytes(b"")
        summary["warnings"].append("No frame was available for overlay rendering.")
    summary["artifacts"] = {
        "working_distance_frames_csv": str(csv_path.resolve()),
        "working_distance_summary_json": str(summary_path.resolve()),
        "working_distance_report_md": str(report_path.resolve()),
        "working_distance_overlay_png": str(overlay_path.resolve()),
    }
    _write_json(summary_path, summary)
    _write_report(report_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "frame_count": len(rows),
                "complete_frame_count": len(complete_rows),
                "stats": summary["results"]["stats"],
                "board_range": summary["board_range"],
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
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_audit(args.recording, args.session, args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
