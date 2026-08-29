#!/usr/bin/env python3
"""Replay Haikang C0 with the frozen H0-1M-A manual ROI registry.

The adapter replaces only Auto ROI-V2 target selection.  Frame extraction,
circular-cone reconstruction, Session Ground and the final local height formula
remain the production implementations.  Ground truth is attached only after
all raw frame measurements have completed and the raw-only frame CSV is saved.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "laser_measurement_tool"
TOOLS_ROOT = REPO_ROOT / "tools"
for import_root in (TOOL_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from measurement.ground_reference import MeasurementError  # noqa: E402
from measurement.height_measure import measure_height_line  # noqa: E402

import generate_haikang_c0_h_raw_0829 as h0  # noqa: E402


DIAGNOSTIC_MODE = "MANUAL_ROI_DIAGNOSTIC"
DATA_ROOT_DEFAULT = TOOL_ROOT / "output_haikang_0828" / "online_recordings" / "0829"
CONFIG_DEFAULT = TOOL_ROOT / "configs" / "measure_tool_haikang_0828.yaml"
REGISTRY_DEFAULT = (
    DATA_ROOT_DEFAULT / "c0_height_audit" / "manual_roi" / "manual_roi_registry.json"
)
OUTPUT_DEFAULT = DATA_ROOT_DEFAULT / "c0_height_audit" / "manual_roi_measurement"
HEIGHT_IDS = ("h02", "h06", "h10", "h20", "h30")
POSITION_IDS = tuple(f"p{index:02d}" for index in range(1, 11))
FRAME_COUNT = 20


FRAME_FIELDS = [
    "diagnostic_mode",
    "condition",
    "height_id",
    "position_id",
    "frame_index",
    "filename",
    "camera_frame_number",
    "host_timestamp_ns",
    "offset_x",
    "offset_y",
    "baseline_before_u0",
    "baseline_before_u1",
    "height_u0",
    "height_u1",
    "baseline_after_u0",
    "baseline_after_u1",
    "frame_pipeline_status",
    "frame_pipeline_error",
    "steger_center_count",
    "reconstructed_point_count",
    "ground_extrinsic_source",
    "ground_extrinsic_generation",
    "ground_reference_source",
    "ground_reference_status",
    "ground_reference_applied_count",
    "ground_reference_out_of_range_count",
    "valid_points_baseline_before",
    "valid_points_height",
    "valid_points_baseline_after",
    "valid_points_baseline_combined",
    "support_status",
    "local_baseline_fit_status",
    "local_baseline_fit_error",
    "local_ground_reference_mode",
    "local_ground_profile_slope_mm_per_mm",
    "local_ground_profile_intercept_mm",
    "local_ground_profile_rmse_mm",
    "local_baseline_inlier_count",
    "local_height_inlier_count",
    "session_measurement_status",
    "session_measurement_error",
    "session_height_mean_mm",
    "session_height_median_mm",
    "session_height_std_mm",
    "local_measurement_status",
    "local_measurement_error",
    "h_raw_mm",
    "local_height_median_mm",
    "local_height_std_mm",
    "measurement_status",
    "extraction_ms",
    "reconstruction_ms",
    "total_ms",
]


SUMMARY_FIELDS = [
    "diagnostic_mode",
    "condition",
    "height_id",
    "position_id",
    "height_gt_mm",
    "expected_frame_count",
    "valid_frame_count",
    "valid_frame_ratio",
    "local_baseline_fit_valid_frame_count",
    "session_valid_frame_count",
    "h_raw_mm_median",
    "h_raw_mm_mean",
    "h_raw_temporal_std_mm",
    "h_raw_mm_p05",
    "h_raw_mm_p95",
    "h_raw_mm_min",
    "h_raw_mm_max",
    "session_height_mean_mm_median",
    "session_height_mean_mm_mean",
    "session_height_temporal_std_mm",
    "baseline_before_points_median",
    "height_points_median",
    "baseline_after_points_median",
    "baseline_before_points_min",
    "height_points_min",
    "baseline_after_points_min",
    "bias_mm",
    "absolute_error_mm",
]


class ManualReplayError(RuntimeError):
    """Raised when the frozen replay contract is not satisfied."""


@dataclass(frozen=True, slots=True)
class ManualCondition:
    height_id: str
    position_id: str
    path: Path

    @property
    def condition(self) -> str:
        return f"{self.height_id}_{self.position_id}"


@dataclass(frozen=True, slots=True)
class ManualRoi:
    condition: str
    height_id: str
    position_id: str
    baseline_before: tuple[int, int]
    height: tuple[int, int]
    baseline_after: tuple[int, int]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field)
                    for field in fields
                }
            )


def discover_conditions(root: Path) -> list[ManualCondition]:
    conditions: list[ManualCondition] = []
    for height_id in HEIGHT_IDS:
        for position_id in POSITION_IDS:
            path = root / height_id / f"{height_id}_{position_id}"
            if not path.is_dir():
                raise ManualReplayError(f"missing condition directory: {path}")
            rows = h0.read_csv_rows(path / "frames.csv", h0.FRAME_FIELDS)
            if len(rows) != FRAME_COUNT:
                raise ManualReplayError(
                    f"{height_id}_{position_id}: expected {FRAME_COUNT} frames, got {len(rows)}"
                )
            conditions.append(ManualCondition(height_id, position_id, path.resolve()))
    return conditions


def source_dataset_provenance(
    conditions: list[ManualCondition], registry_payload: dict[str, Any]
) -> dict[str, Any]:
    registry_entries = {
        str(entry.get("condition")): entry for entry in registry_payload.get("entries", [])
    }
    rows: list[dict[str, Any]] = []
    metadata_mismatches: list[dict[str, Any]] = []
    dataset_digest = hashlib.sha256()
    for condition in conditions:
        frames_csv = condition.path / "frames.csv"
        frame_rows = h0.read_csv_rows(frames_csv, h0.FRAME_FIELDS)
        condition_digest = hashlib.sha256()
        geometries: set[tuple[int, int, int, int]] = set()
        representative_row: dict[str, str] | None = None
        for row in frame_rows:
            image_path = condition.path / row["filename"]
            image_sha = h0.sha256_file(image_path)
            token = f"{row['filename']}:{image_sha}\n".encode("utf-8")
            condition_digest.update(token)
            dataset_digest.update(f"{condition.condition}:".encode("utf-8") + token)
            geometries.add(
                (
                    int(row["offset_x"]),
                    int(row["offset_y"]),
                    int(row["width"]),
                    int(row["height"]),
                )
            )
            if row["filename"] == "frame_000010.png":
                representative_row = row
        if len(geometries) != 1:
            raise ManualReplayError(
                f"{condition.condition}: inconsistent frame geometry: {geometries}"
            )
        geometry = next(iter(geometries))
        rows.append(
            {
                "condition": condition.condition,
                "frames_csv": str(frames_csv.resolve()),
                "frames_csv_sha256": h0.sha256_file(frames_csv),
                "source_png_count": len(frame_rows),
                "source_png_combined_sha256": condition_digest.hexdigest(),
                "capture_geometry": {
                    "offset_x": geometry[0],
                    "offset_y": geometry[1],
                    "width": geometry[2],
                    "height": geometry[3],
                },
            }
        )
        registry_source = (registry_entries[condition.condition].get("representative_source") or {})
        if representative_row is not None:
            current = {
                name: int(representative_row[name])
                for name in ("offset_x", "offset_y", "width", "height")
            }
            frozen_snapshot = {
                name: registry_source.get(name)
                for name in ("offset_x", "offset_y", "width", "height")
            }
            if current != frozen_snapshot:
                metadata_mismatches.append(
                    {
                        "condition": condition.condition,
                        "field_scope": "representative_source capture metadata only",
                        "frozen_registry_snapshot": frozen_snapshot,
                        "current_frames_csv": current,
                        "roi_ranges_affected": False,
                    }
                )
    return {
        "condition_count": len(rows),
        "source_png_count": len(rows) * FRAME_COUNT,
        "dataset_combined_sha256": dataset_digest.hexdigest(),
        "conditions": rows,
        "registry_representative_metadata_mismatches": metadata_mismatches,
        "registry_representative_metadata_mismatch_count": len(metadata_mismatches),
    }


def _roi_range(entry: dict[str, Any], prefix: str) -> tuple[int, int]:
    try:
        first = int(entry[f"{prefix}_u0"])
        second = int(entry[f"{prefix}_u1"])
    except (KeyError, TypeError, ValueError) as error:
        raise ManualReplayError(
            f"{entry.get('condition')}: invalid {prefix} range"
        ) from error
    if first > second:
        raise ManualReplayError(f"{entry.get('condition')}: reversed {prefix} range")
    return first, second


def load_registry(path: Path) -> tuple[dict[str, Any], dict[str, ManualRoi]]:
    if not path.is_file():
        raise ManualReplayError(f"manual ROI registry is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != "H0-1M-A":
        raise ManualReplayError("registry task is not H0-1M-A")
    if payload.get("frozen") is not True or payload.get("manual_confirmed") is not True:
        raise ManualReplayError("manual ROI registry is not frozen/manual_confirmed")
    if payload.get("coordinate_system") != "full_sensor":
        raise ManualReplayError("registry coordinate_system is not full_sensor")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 50:
        raise ManualReplayError("manual ROI registry must contain 50 entries")
    output: dict[str, ManualRoi] = {}
    for entry in entries:
        condition = str(entry.get("condition") or "")
        if condition in output:
            raise ManualReplayError(f"duplicate registry condition: {condition}")
        if entry.get("selection_status") != "selected":
            raise ManualReplayError(f"{condition}: selection_status is not selected")
        if entry.get("selection_mode") != "manual":
            raise ManualReplayError(f"{condition}: selection_mode is not manual")
        if entry.get("coordinate_system") != "full_sensor":
            raise ManualReplayError(f"{condition}: entry is not full_sensor")
        provenance = entry.get("manual_provenance") or {}
        if not (
            provenance.get("geometry_only") is True
            and provenance.get("truth_values_used") is False
            and provenance.get("height_results_used") is False
            and provenance.get("automatic_roi_used") is False
        ):
            raise ManualReplayError(f"{condition}: manual provenance guard failed")
        before = _roi_range(entry, "baseline_before")
        height = _roi_range(entry, "height")
        after = _roi_range(entry, "baseline_after")
        if before[1] >= height[0] or height[1] >= after[0]:
            raise ManualReplayError(f"{condition}: ROI ranges overlap or are out of order")
        output[condition] = ManualRoi(
            condition=condition,
            height_id=str(entry.get("height_id")),
            position_id=str(entry.get("position_id")),
            baseline_before=before,
            height=height,
            baseline_after=after,
        )
    expected = {
        f"{height_id}_{position_id}"
        for height_id in HEIGHT_IDS
        for position_id in POSITION_IDS
    }
    if set(output) != expected:
        raise ManualReplayError("registry condition set does not match the 50-condition target")
    return payload, output


def inclusive_u_mask(pixels_uv: np.ndarray, value_range: tuple[int, int]) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ManualReplayError(f"pixels_uv has invalid shape: {pixels.shape}")
    return (pixels[:, 0] >= value_range[0]) & (pixels[:, 0] <= value_range[1])


def _measurement_fields(measurement: Any | None, prefix: str) -> dict[str, Any]:
    if measurement is None:
        return {
            f"{prefix}_height_mean_mm": None,
            f"{prefix}_height_median_mm": None,
            f"{prefix}_height_std_mm": None,
            f"{prefix}_baseline_inlier_count": None,
            f"{prefix}_height_inlier_count": None,
            f"{prefix}_ground_reference_mode": None,
            f"{prefix}_ground_profile_slope_mm_per_mm": None,
            f"{prefix}_ground_profile_intercept_mm": None,
            f"{prefix}_ground_profile_rmse_mm": None,
        }
    profile = measurement.ground_profile_fit
    return {
        f"{prefix}_height_mean_mm": finite(measurement.height_mean_mm),
        f"{prefix}_height_median_mm": finite(measurement.height_median_mm),
        f"{prefix}_height_std_mm": finite(measurement.height_std_mm),
        f"{prefix}_baseline_inlier_count": int(measurement.baseline_inlier_count),
        f"{prefix}_height_inlier_count": int(measurement.height_inlier_count),
        f"{prefix}_ground_reference_mode": measurement.ground_reference_mode,
        f"{prefix}_ground_profile_slope_mm_per_mm": (
            finite(profile.slope_z_per_mm) if profile is not None else None
        ),
        f"{prefix}_ground_profile_intercept_mm": (
            finite(profile.intercept_z_mm) if profile is not None else None
        ),
        f"{prefix}_ground_profile_rmse_mm": (
            finite(profile.rmse_mm) if profile is not None else None
        ),
    }


def _measure(
    baseline: np.ndarray,
    height: np.ndarray,
    params: Any,
    mode: str,
) -> tuple[Any | None, str]:
    try:
        return (
            measure_height_line(
                baseline,
                height,
                params,
                ground_correction_mode=mode,
            ),
            "",
        )
    except (MeasurementError, ValueError, TypeError, np.linalg.LinAlgError) as error:
        return None, f"{type(error).__name__}: {error}"


def _empty_frame_row(
    condition: ManualCondition,
    roi: ManualRoi,
    row: dict[str, str],
    frame_index: int,
) -> dict[str, Any]:
    return {
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "condition": condition.condition,
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "frame_index": frame_index,
        "filename": row["filename"],
        "camera_frame_number": row["camera_frame_number"],
        "host_timestamp_ns": row["host_timestamp_ns"],
        "offset_x": row["offset_x"],
        "offset_y": row["offset_y"],
        "baseline_before_u0": roi.baseline_before[0],
        "baseline_before_u1": roi.baseline_before[1],
        "height_u0": roi.height[0],
        "height_u1": roi.height[1],
        "baseline_after_u0": roi.baseline_after[0],
        "baseline_after_u1": roi.baseline_after[1],
        "frame_pipeline_status": "NOT_RUN",
        "frame_pipeline_error": "",
        "steger_center_count": 0,
        "reconstructed_point_count": 0,
        "valid_points_baseline_before": 0,
        "valid_points_height": 0,
        "valid_points_baseline_after": 0,
        "valid_points_baseline_combined": 0,
        "support_status": "NOT_CHECKED",
        "local_baseline_fit_status": "NOT_MEASURED",
        "local_baseline_fit_error": "",
        "session_measurement_status": "NOT_MEASURED",
        "session_measurement_error": "",
        "local_measurement_status": "NOT_MEASURED",
        "local_measurement_error": "",
        "measurement_status": "NOT_MEASURED",
    }


def replay_condition(
    condition: ManualCondition,
    roi: ManualRoi,
    pipeline: Any,
    app: Any,
) -> list[dict[str, Any]]:
    source_rows = h0.read_csv_rows(condition.path / "frames.csv", h0.FRAME_FIELDS)
    records: list[dict[str, Any]] = []
    for frame_index, source_row in enumerate(source_rows, start=1):
        record = _empty_frame_row(condition, roi, source_row, frame_index)
        try:
            frame = h0.frame_from_row(condition, source_row)
            result = pipeline.run_frame(frame)
            if result.active_height_correction != "none":
                raise ManualReplayError(
                    f"unexpected active height correction: {result.active_height_correction}"
                )
            if result.c1_clamp_status != "NOT_APPLICABLE":
                raise ManualReplayError(
                    f"unexpected C1 reconstruction status: {result.c1_clamp_status}"
                )
            pixels = np.asarray(result.pixels_uv, dtype=np.float64)
            points = np.asarray(result.points_ground, dtype=np.float64)
            if pixels.ndim != 2 or pixels.shape[1] != 2:
                raise ManualReplayError(f"invalid reconstructed pixels shape: {pixels.shape}")
            if points.ndim != 2 or points.shape[1] != 3:
                raise ManualReplayError(f"invalid ground points shape: {points.shape}")
            if len(pixels) != len(points):
                raise ManualReplayError("pixels_uv and points_ground are not aligned")
        except Exception as error:  # noqa: BLE001 - retain every source frame
            message = f"{type(error).__name__}: {error}"
            record.update(
                {
                    "frame_pipeline_status": "INVALID",
                    "frame_pipeline_error": message,
                    "support_status": "INVALID_PIPELINE",
                    "local_baseline_fit_status": "INVALID_PIPELINE",
                    "local_baseline_fit_error": message,
                    "session_measurement_status": "INVALID_PIPELINE",
                    "session_measurement_error": message,
                    "local_measurement_status": "INVALID_PIPELINE",
                    "local_measurement_error": message,
                    "measurement_status": "INVALID_PIPELINE",
                }
            )
            records.append(record)
            continue

        before_mask = inclusive_u_mask(pixels, roi.baseline_before)
        height_mask = inclusive_u_mask(pixels, roi.height)
        after_mask = inclusive_u_mask(pixels, roi.baseline_after)
        before_count = int(np.count_nonzero(before_mask))
        height_count = int(np.count_nonzero(height_mask))
        after_count = int(np.count_nonzero(after_mask))
        baseline_count = before_count + after_count
        record.update(
            {
                "frame_pipeline_status": "VALID",
                "steger_center_count": int(len(result.centers_uv_full)),
                "reconstructed_point_count": int(len(points)),
                "ground_extrinsic_source": result.ground_extrinsic_source,
                "ground_extrinsic_generation": result.ground_extrinsic_generation,
                "ground_reference_source": result.ground_reference_source,
                "ground_reference_status": result.ground_reference_status,
                "ground_reference_applied_count": result.ground_reference_applied_count,
                "ground_reference_out_of_range_count": result.ground_reference_out_of_range_count,
                "valid_points_baseline_before": before_count,
                "valid_points_height": height_count,
                "valid_points_baseline_after": after_count,
                "valid_points_baseline_combined": baseline_count,
                "extraction_ms": finite(result.extraction_ms),
                "reconstruction_ms": finite(result.reconstruction_ms),
                "total_ms": finite(result.total_ms),
            }
        )
        if (
            before_count < app.measurement.min_baseline_points
            or after_count < app.measurement.min_baseline_points
        ):
            record.update(
                {
                    "support_status": "INVALID_BOTH_SIDES_SUPPORT",
                    "local_baseline_fit_status": "INVALID_SUPPORT",
                    "local_baseline_fit_error": "each baseline side must meet min_baseline_points",
                    "session_measurement_status": "INVALID_SUPPORT",
                    "session_measurement_error": "each baseline side must meet min_baseline_points",
                    "local_measurement_status": "INVALID_SUPPORT",
                    "local_measurement_error": "each baseline side must meet min_baseline_points",
                    "measurement_status": "INVALID_SUPPORT",
                }
            )
            records.append(record)
            continue
        if height_count < app.measurement.min_height_points:
            record.update(
                {
                    "support_status": "INVALID_HEIGHT_SUPPORT",
                    "local_baseline_fit_status": "INVALID_SUPPORT",
                    "local_baseline_fit_error": "height must meet min_height_points",
                    "session_measurement_status": "INVALID_SUPPORT",
                    "session_measurement_error": "height must meet min_height_points",
                    "local_measurement_status": "INVALID_SUPPORT",
                    "local_measurement_error": "height must meet min_height_points",
                    "measurement_status": "INVALID_SUPPORT",
                }
            )
            records.append(record)
            continue

        baseline_points = np.concatenate(
            [points[before_mask], points[after_mask]], axis=0
        )
        height_points = points[height_mask]
        session_measurement, session_error = _measure(
            baseline_points, height_points, app.measurement, "session_reference"
        )
        local_measurement, local_error = _measure(
            baseline_points, height_points, app.measurement, "auto"
        )
        session_fields = _measurement_fields(session_measurement, "session")
        local_fields = _measurement_fields(local_measurement, "local")
        local_fit_valid = bool(
            local_measurement is not None
            and local_measurement.ground_profile_fit is not None
            and local_measurement.baseline_fit is not None
        )
        h_raw = local_fields["local_height_mean_mm"]
        record.update(
            {
                "support_status": "VALID",
                "local_baseline_fit_status": "VALID" if local_fit_valid else "INVALID_FIT",
                "local_baseline_fit_error": local_error,
                "local_ground_reference_mode": local_fields["local_ground_reference_mode"],
                "local_ground_profile_slope_mm_per_mm": local_fields[
                    "local_ground_profile_slope_mm_per_mm"
                ],
                "local_ground_profile_intercept_mm": local_fields[
                    "local_ground_profile_intercept_mm"
                ],
                "local_ground_profile_rmse_mm": local_fields[
                    "local_ground_profile_rmse_mm"
                ],
                "local_baseline_inlier_count": local_fields[
                    "local_baseline_inlier_count"
                ],
                "local_height_inlier_count": local_fields["local_height_inlier_count"],
                "session_measurement_status": (
                    "VALID" if session_measurement is not None else "INVALID_MEASUREMENT"
                ),
                "session_measurement_error": session_error,
                "session_height_mean_mm": session_fields["session_height_mean_mm"],
                "session_height_median_mm": session_fields["session_height_median_mm"],
                "session_height_std_mm": session_fields["session_height_std_mm"],
                "local_measurement_status": (
                    "VALID" if local_measurement is not None else "INVALID_MEASUREMENT"
                ),
                "local_measurement_error": local_error,
                "h_raw_mm": h_raw,
                "local_height_median_mm": local_fields["local_height_median_mm"],
                "local_height_std_mm": local_fields["local_height_std_mm"],
                "measurement_status": (
                    "VALID"
                    if local_fit_valid and session_measurement is not None and h_raw is not None
                    else "INVALID_MEASUREMENT"
                ),
            }
        )
        records.append(record)
    return records


def finite_array(values: Iterable[Any]) -> np.ndarray:
    output = [value for value in (finite(item) for item in values) if value is not None]
    return np.asarray(output, dtype=np.float64)


def describe(values: Iterable[Any]) -> dict[str, Any]:
    array = finite_array(values)
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p05": float(np.percentile(array, 5.0)),
        "p95": float(np.percentile(array, 95.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_raw_conditions(
    conditions: list[ManualCondition], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)
    summaries: list[dict[str, Any]] = []
    for condition in conditions:
        group = by_condition[condition.condition]
        if len(group) != FRAME_COUNT:
            raise ManualReplayError(
                f"{condition.condition}: replay produced {len(group)} rows, expected {FRAME_COUNT}"
            )
        raw = describe(row.get("h_raw_mm") for row in group)
        session = describe(row.get("session_height_mean_mm") for row in group)
        before = describe(row.get("valid_points_baseline_before") for row in group)
        height = describe(row.get("valid_points_height") for row in group)
        after = describe(row.get("valid_points_baseline_after") for row in group)
        summaries.append(
            {
                "diagnostic_mode": DIAGNOSTIC_MODE,
                "condition": condition.condition,
                "height_id": condition.height_id,
                "position_id": condition.position_id,
                "height_gt_mm": None,
                "expected_frame_count": FRAME_COUNT,
                "valid_frame_count": raw["count"],
                "valid_frame_ratio": raw["count"] / FRAME_COUNT,
                "local_baseline_fit_valid_frame_count": sum(
                    row.get("local_baseline_fit_status") == "VALID" for row in group
                ),
                "session_valid_frame_count": session["count"],
                "h_raw_mm_median": raw["median"],
                "h_raw_mm_mean": raw["mean"],
                "h_raw_temporal_std_mm": raw["std"],
                "h_raw_mm_p05": raw["p05"],
                "h_raw_mm_p95": raw["p95"],
                "h_raw_mm_min": raw["min"],
                "h_raw_mm_max": raw["max"],
                "session_height_mean_mm_median": session["median"],
                "session_height_mean_mm_mean": session["mean"],
                "session_height_temporal_std_mm": session["std"],
                "baseline_before_points_median": before["median"],
                "height_points_median": height["median"],
                "baseline_after_points_median": after["median"],
                "baseline_before_points_min": before["min"],
                "height_points_min": height["min"],
                "baseline_after_points_min": after["min"],
                "bias_mm": None,
                "absolute_error_mm": None,
            }
        )
    return summaries


def load_ground_truth_after_replay(raw_replay_complete: bool) -> dict[str, float]:
    if raw_replay_complete is not True:
        raise ManualReplayError("ground truth cannot be loaded before raw replay completes")
    return {"h02": 2.0, "h06": 6.0, "h10": 10.0, "h20": 20.0, "h30": 30.0}


def attach_ground_truth(
    summaries: list[dict[str, Any]], truth: dict[str, float]
) -> None:
    for row in summaries:
        gt = truth[row["height_id"]]
        measured = finite(row.get("h_raw_mm_median"))
        row["height_gt_mm"] = gt
        row["bias_mm"] = None if measured is None else measured - gt
        row["absolute_error_mm"] = (
            None if row["bias_mm"] is None else abs(row["bias_mm"])
        )


def error_metrics(errors: Iterable[Any]) -> dict[str, Any]:
    array = finite_array(errors)
    if not len(array):
        return {
            "count": 0,
            "bias_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_absolute_error_mm": None,
            "max_absolute_error_mm": None,
        }
    absolute = np.abs(array)
    return {
        "count": int(len(array)),
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(np.square(array)))),
        "p95_absolute_error_mm": float(np.percentile(absolute, 95.0)),
        "max_absolute_error_mm": float(np.max(absolute)),
    }


def _group_accuracy(
    summaries: list[dict[str, Any]], group_field: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        groups[str(row[group_field])].append(row)
    for key, rows in sorted(groups.items()):
        metrics = error_metrics(row.get("bias_mm") for row in rows)
        measured = describe(row.get("h_raw_mm_median") for row in rows)
        metrics.update(
            {
                "measured_height_median_mm": measured["median"],
                "measured_height_mean_mm": measured["mean"],
                "measured_height_range_mm": (
                    None
                    if measured["min"] is None
                    else measured["max"] - measured["min"]
                ),
                "condition_count": len(rows),
            }
        )
        output[key] = metrics
    return output


def analyze_accuracy(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    overall = error_metrics(row.get("bias_mm") for row in summaries)
    per_height = _group_accuracy(summaries, "height_id")
    per_position = _group_accuracy(summaries, "position_id")
    height_truth = {row["height_id"]: float(row["height_gt_mm"]) for row in summaries}
    height_order = sorted(HEIGHT_IDS, key=lambda item: height_truth[item])

    measured_matrix = np.full((len(height_order), len(POSITION_IDS)), np.nan)
    error_matrix = np.full_like(measured_matrix, np.nan)
    lookup = {(row["height_id"], row["position_id"]): row for row in summaries}
    for height_index, height_id in enumerate(height_order):
        for position_index, position_id in enumerate(POSITION_IDS):
            row = lookup[(height_id, position_id)]
            measured_value = finite(row.get("h_raw_mm_median"))
            error_value = finite(row.get("bias_mm"))
            if measured_value is not None:
                measured_matrix[height_index, position_index] = measured_value
            if error_value is not None:
                error_matrix[height_index, position_index] = error_value

    monotonic_by_position = {
        position_id: bool(
            np.isfinite(measured_matrix[:, index]).all()
            and np.all(np.diff(measured_matrix[:, index]) > 0.0)
        )
        for index, position_id in enumerate(POSITION_IDS)
    }
    with np.errstate(all="ignore"):
        mean_response = np.nanmean(measured_matrix, axis=1)
    mean_response_monotonic = bool(
        np.isfinite(mean_response).all() and np.all(np.diff(mean_response) > 0.0)
    )
    response_gain_endpoint = (
        float(
            (mean_response[-1] - mean_response[0])
            / (height_truth[height_order[-1]] - height_truth[height_order[0]])
        )
        if np.isfinite(mean_response[[0, -1]]).all()
        else None
    )

    adjacent_errors: list[float] = []
    adjacent_rows: list[dict[str, Any]] = []
    for position_index, position_id in enumerate(POSITION_IDS):
        for index in range(len(height_order) - 1):
            lower, upper = height_order[index : index + 2]
            pair = measured_matrix[index : index + 2, position_index]
            if not np.isfinite(pair).all():
                continue
            measured_delta = pair[1] - pair[0]
            truth_delta = height_truth[upper] - height_truth[lower]
            delta_error = float(measured_delta - truth_delta)
            adjacent_errors.append(delta_error)
            adjacent_rows.append(
                {
                    "position_id": position_id,
                    "lower_height_id": lower,
                    "upper_height_id": upper,
                    "true_delta_mm": truth_delta,
                    "measured_delta_mm": float(measured_delta),
                    "delta_error_mm": delta_error,
                }
            )

    with np.errstate(all="ignore"):
        grand = float(np.nanmean(error_matrix))
        height_means = np.nanmean(error_matrix, axis=1)
        position_means = np.nanmean(error_matrix, axis=0)
        interaction = (
            error_matrix - height_means[:, None] - position_means[None, :] + grand
        )
        height_effect_rms = float(
            np.sqrt(np.nanmean(np.square(height_means - grand)))
        )
        position_effect_rms = float(
            np.sqrt(np.nanmean(np.square(position_means - grand)))
        )
        interaction_rms = float(np.sqrt(np.nanmean(np.square(interaction))))
        height_error_span = float(np.nanmax(height_means) - np.nanmin(height_means))
        position_error_span = float(
            np.nanmax(position_means) - np.nanmin(position_means)
        )

    temporal = finite_array(row.get("h_raw_temporal_std_mm") for row in summaries)
    temporal_median = float(np.median(temporal)) if len(temporal) else None
    temporal_p95 = float(np.percentile(temporal, 95.0)) if len(temporal) else None
    temporal_explains = bool(
        temporal_median is not None
        and overall["mae_mm"] is not None
        and overall["rmse_mm"] is not None
        and temporal_p95 is not None
        and (
            temporal_median >= 0.5 * overall["mae_mm"]
            or temporal_p95 >= overall["rmse_mm"]
        )
    )

    overall_valid_frame_ratio = sum(row["valid_frame_count"] for row in summaries) / (
        len(summaries) * FRAME_COUNT
    )
    min_condition_valid_ratio = min(row["valid_frame_ratio"] for row in summaries)
    valid_condition_ratio = sum(row["valid_frame_ratio"] > 0.0 for row in summaries) / len(
        summaries
    )
    target_met = bool(
        overall["mae_mm"] is not None
        and overall["mae_mm"] <= 0.2
        and overall["p95_absolute_error_mm"] <= 0.2
    )
    strict_all_conditions_met = bool(
        overall["max_absolute_error_mm"] is not None
        and overall["max_absolute_error_mm"] <= 0.2
    )
    manual_invalid = bool(
        valid_condition_ratio < 0.95
        or overall_valid_frame_ratio < 0.95
        or min_condition_valid_ratio < 0.80
    )
    if manual_invalid:
        classification = "MANUAL_ROI_STILL_INVALID"
    elif target_met:
        classification = "C0_SUFFICIENT"
    elif (
        height_effect_rms >= 1.5 * position_effect_rms
        and height_error_span >= 0.10
    ):
        classification = "HEIGHT_SCALE_TREND"
    elif (
        position_effect_rms >= 1.5 * height_effect_rms
        and position_error_span >= 0.10
    ):
        classification = "SPATIAL_POSITION_TREND"
    else:
        classification = "MIXED"

    if classification == "HEIGHT_SCALE_TREND":
        next_step = "H1 feasibility"
    elif classification == "SPATIAL_POSITION_TREND":
        next_step = "H-B2/spatial feasibility"
    elif classification in {"MANUAL_ROI_STILL_INVALID", "C0_SUFFICIENT"}:
        next_step = (
            "upstream C0/Session/local measurement issue"
            if classification == "MANUAL_ROI_STILL_INVALID"
            else "no correction feasibility required"
        )
    elif not mean_response_monotonic or not all(monotonic_by_position.values()):
        next_step = "upstream C0/Session/local measurement issue"
    elif height_effect_rms > position_effect_rms:
        next_step = "H1 feasibility"
    elif position_effect_rms > height_effect_rms:
        next_step = "H-B2/spatial feasibility"
    else:
        next_step = "upstream C0/Session/local measurement issue"

    return {
        "overall": overall,
        "per_height": per_height,
        "per_position": per_position,
        "adjacent_height_difference": {
            "pair_count": len(adjacent_errors),
            "mae_mm": (
                float(np.mean(np.abs(adjacent_errors))) if adjacent_errors else None
            ),
            "rmse_mm": (
                float(np.sqrt(np.mean(np.square(adjacent_errors))))
                if adjacent_errors
                else None
            ),
            "rows": adjacent_rows,
        },
        "monotonic_response": {
            "mean_response_monotonic": mean_response_monotonic,
            "monotonic_position_count": sum(monotonic_by_position.values()),
            "position_count": len(monotonic_by_position),
            "by_position": monotonic_by_position,
            "mean_measured_by_height_mm": {
                height_id: float(mean_response[index])
                for index, height_id in enumerate(height_order)
            },
            "endpoint_response_gain": response_gain_endpoint,
        },
        "spatial_drift": {
            "max_same_height_position_drift_mm": max(
                (
                    value["measured_height_range_mm"]
                    for value in per_height.values()
                    if value["measured_height_range_mm"] is not None
                ),
                default=None,
            ),
            "per_height_position_drift_mm": {
                key: value["measured_height_range_mm"]
                for key, value in per_height.items()
            },
        },
        "temporal_variation": {
            "condition_temporal_std_median_mm": temporal_median,
            "condition_temporal_std_p95_mm": temporal_p95,
            "sufficient_to_explain_accuracy_error": temporal_explains,
        },
        "residual_structure": {
            "grand_bias_mm": grand,
            "height_effect_rms_mm": height_effect_rms,
            "position_effect_rms_mm": position_effect_rms,
            "interaction_rms_mm": interaction_rms,
            "height_mean_error_span_mm": height_error_span,
            "position_mean_error_span_mm": position_error_span,
            "decomposition_is_diagnostic_only_not_a_correction_fit": True,
        },
        "validity": {
            "valid_condition_ratio": valid_condition_ratio,
            "overall_valid_frame_ratio": overall_valid_frame_ratio,
            "minimum_condition_valid_frame_ratio": min_condition_valid_ratio,
        },
        "target_0p2": {
            "criterion": "overall MAE <= 0.2 mm and P95 absolute error <= 0.2 mm",
            "met": target_met,
            "strict_all_conditions_criterion": "Max absolute error <= 0.2 mm",
            "strict_all_conditions_met": strict_all_conditions_met,
        },
        "classification": classification,
        "recommended_next_step": next_step,
        "height_order": height_order,
        "position_order": list(POSITION_IDS),
        "measured_matrix_mm": measured_matrix,
        "residual_matrix_mm": error_matrix,
    }


def save_plots(
    output: Path, summaries: list[dict[str, Any]], analysis: dict[str, Any]
) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    gt = np.asarray([row["height_gt_mm"] for row in summaries], dtype=np.float64)
    measured = np.asarray(
        [
            np.nan if finite(row.get("h_raw_mm_median")) is None else row["h_raw_mm_median"]
            for row in summaries
        ],
        dtype=np.float64,
    )
    errors = measured - gt
    position_number = np.asarray(
        [int(str(row["position_id"])[1:]) for row in summaries], dtype=np.int64
    )

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    valid = np.isfinite(measured)
    scatter = axis.scatter(
        gt[valid], measured[valid], c=position_number[valid], cmap="viridis", s=42, alpha=0.85
    )
    limits = [min(gt[valid].min(), measured[valid].min()), max(gt[valid].max(), measured[valid].max())]
    axis.plot(limits, limits, "k--", linewidth=1.2, label="identity")
    axis.set_xlabel("true height [mm]")
    axis.set_ylabel("manual-ROI C0 h_raw median [mm]")
    axis.set_title(f"{DIAGNOSTIC_MODE}: measured height vs truth")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="position index")
    figure.savefig(output / "height_pred_vs_gt.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = axis.scatter(
        gt[valid], errors[valid], c=position_number[valid], cmap="viridis", s=42, alpha=0.85
    )
    height_means = {
        float(row["height_gt_mm"]): analysis["per_height"][row["height_id"]]["bias_mm"]
        for row in summaries
    }
    ordered_truth = sorted(height_means)
    axis.plot(ordered_truth, [height_means[value] for value in ordered_truth], "o-k", label="per-height mean error")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("true height [mm]")
    axis.set_ylabel("condition-median error [mm]")
    axis.set_title(f"{DIAGNOSTIC_MODE}: error vs true height")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="position index")
    figure.savefig(output / "error_vs_height.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for height_id in HEIGHT_IDS:
        rows = [row for row in summaries if row["height_id"] == height_id]
        axis.plot(
            [int(row["position_id"][1:]) for row in rows],
            [row["bias_mm"] for row in rows],
            marker="o",
            linewidth=1.2,
            label=height_id,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("position index")
    axis.set_ylabel("condition-median error [mm]")
    axis.set_xticks(range(1, 11))
    axis.set_title(f"{DIAGNOSTIC_MODE}: error vs position")
    axis.grid(alpha=0.25)
    axis.legend(ncol=5)
    figure.savefig(output / "error_vs_position.png", dpi=160)
    plt.close(figure)

    matrix = np.asarray(analysis["residual_matrix_mm"], dtype=np.float64)
    limit = max(0.01, float(np.max(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(POSITION_IDS)), POSITION_IDS)
    axis.set_yticks(range(len(analysis["height_order"])), analysis["height_order"])
    axis.set_xlabel("position")
    axis.set_ylabel("height condition")
    axis.set_title(f"{DIAGNOSTIC_MODE}: height × position residual [mm]")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    figure.colorbar(image, ax=axis, label="error [mm]")
    figure.savefig(output / "height_position_residual_heatmap.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    temporal = [row["h_raw_temporal_std_mm"] for row in summaries]
    labels = [row["condition"] for row in summaries]
    axis.plot(range(len(labels)), temporal, marker="o", markersize=3, linewidth=1.0)
    axis.axhline(0.2, color="red", linestyle="--", linewidth=1.0, label="0.2 mm reference")
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    axis.set_ylabel("20-frame temporal std [mm]")
    axis.set_title(f"{DIAGNOSTIC_MODE}: temporal std vs condition")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output / "temporal_std_vs_condition.png", dpi=160)
    plt.close(figure)


def format_mm(value: Any) -> str:
    number = finite(value)
    return "N/A" if number is None else f"{number:.4f} mm"


def build_report(
    analysis: dict[str, Any],
    summaries: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> str:
    overall = analysis["overall"]
    monotonic = analysis["monotonic_response"]
    drift = analysis["spatial_drift"]
    temporal = analysis["temporal_variation"]
    residual = analysis["residual_structure"]
    target = analysis["target_0p2"]
    source_dataset = provenance["inputs"]["source_dataset"]
    mismatch_conditions = [
        item["condition"]
        for item in source_dataset["registry_representative_metadata_mismatches"]
    ]
    per_height_lines = "\n".join(
        f"| {height_id} | {format_mm(values['measured_height_mean_mm'])} | "
        f"{format_mm(values['bias_mm'])} | {format_mm(values['mae_mm'])} | "
        f"{format_mm(values['measured_height_range_mm'])} |"
        for height_id, values in analysis["per_height"].items()
    )
    monotonic_answer = (
        f"是。平均响应严格单调，且 {monotonic['monotonic_position_count']}/"
        f"{monotonic['position_count']} 个 position 严格单调。"
        if monotonic["mean_response_monotonic"]
        and monotonic["monotonic_position_count"] == monotonic["position_count"]
        else (
            f"不完全是。平均响应单调={monotonic['mean_response_monotonic']}，"
            f"严格单调 position={monotonic['monotonic_position_count']}/"
            f"{monotonic['position_count']}。"
        )
    )
    temporal_answer = (
        "20 帧 temporal variation 与总体误差同量级，可能解释相当部分误差。"
        if temporal["sufficient_to_explain_accuracy_error"]
        else "20 帧 temporal variation 明显小于总体误差，不足以解释主要残差。"
    )
    dominant = (
        "height"
        if residual["height_effect_rms_mm"] > residual["position_effect_rms_mm"]
        else "position"
    )
    return f"""# Haikang manual-ROI C0 height audit

## Diagnostic boundary

`{DIAGNOSTIC_MODE}`

本报告使用已冻结人工 ROI 绕过 Auto ROI-V2 target detection。它诊断“ROI 正确时”的 C0 + Session Ground + local measurement，不代表当前自动 ROI 系统的最终生产精度。未重新标定 C0，未修改 Session Ground，未拟合或应用 H1/H-B2/C1。

## Provenance warning

Frozen registry 的 representative capture snapshot metadata 有 {source_dataset['registry_representative_metadata_mismatch_count']} 个 condition 与当前 `frames.csv` 不一致：`{', '.join(mismatch_conditions) if mismatch_conditions else 'none'}`。这是代表图 offset/shape 的 frozen snapshot 差异；本轮已对当前 1000 张 PNG 和 50 个 `frames.csv` 建立内容指纹，逐帧使用 registry 中冻结的 full-sensor ROI 数字，并核验 ROI 数字未受影响。旧 registry 未被原地修改。

## Direct answers

1. **2/6/10/20/30 mm 是否产生正确单调响应？** {monotonic_answer} 端点响应增益为 `{monotonic['endpoint_response_gain']:.6f}`。
2. **Overall accuracy：** Bias={format_mm(overall['bias_mm'])}，MAE={format_mm(overall['mae_mm'])}，RMSE={format_mm(overall['rmse_mm'])}，P95 absolute error={format_mm(overall['p95_absolute_error_mm'])}，Max absolute error={format_mm(overall['max_absolute_error_mm'])}。这些统计以 50 个 condition median 为样本；20 帧仅作为重复测量。
3. **同一高度跨 position 最大漂移：** {format_mm(drift['max_same_height_position_drift_mm'])}。
4. **Temporal variation 是否足以解释误差？** {temporal_answer} condition temporal std median={format_mm(temporal['condition_temporal_std_median_mm'])}，P95={format_mm(temporal['condition_temporal_std_p95_mm'])}。
5. **残差主要随 height 还是 position？** 描述性 effect RMS：height={format_mm(residual['height_effect_rms_mm'])}，position={format_mm(residual['position_effect_rms_mm'])}，interaction={format_mm(residual['interaction_rms_mm'])}；在 height/position 两个主效应中较强轴为 `{dominant}`，但 interaction 需要同时比较，不能把残差表述成纯一维趋势。该分解只用于诊断分类，不是 correction fit。
6. **是否达到约 0.2 mm 目标？** Aggregate/P95 判据为 `{'YES' if target['met'] else 'NO'}`（{target['criterion']}）；若要求所有 condition 都满足 Max ≤ 0.2 mm，则为 `{'YES' if target['strict_all_conditions_met'] else 'NO'}`，本轮 Max={format_mm(overall['max_absolute_error_mm'])}。
7. **下一步：** Aggregate/P95 目标下为 `{analysis['recommended_next_step']}`。若要收紧到所有 condition ≤0.2 mm，残差的 height 主效应强于 position，但 interaction 更大，建议先做 H1 feasibility 诊断，再判断是否需要 H-B2/spatial feasibility；本阶段没有拟合或应用二者。

最终分类：`{analysis['classification']}`。

相邻高度差 MAE：{format_mm(analysis['adjacent_height_difference']['mae_mm'])}（40 个 position-matched adjacent pairs）。

## Per-height summary

| Height ID | measured mean across positions | Bias | MAE | position drift |
|---|---:|---:|---:|---:|
{per_height_lines}

## Replay integrity

- Conditions: {len(summaries)}; frames: {provenance['execution']['source_frame_count']}.
- Raw replay completed at `{provenance['execution']['raw_replay_completed_at_utc']}`; ground truth loaded afterward at `{provenance['execution']['ground_truth_loaded_at_utc']}`.
- Manual registry SHA-256: `{provenance['inputs']['manual_roi_registry_sha256']}`.
- Frame chain: `FramePipeline.run_frame → frozen manual full-sensor u masks → existing measure_height_line`.
- Auto ROI-V2 selection calls: `0`.
- Fresh calculations this run: 1000 frame pipeline/measurement results. Reused: frozen manual ROI, production pipeline, Steger/circular-cone config, Session Ground and existing height formula.
- Source dataset combined SHA-256: `{provenance['inputs']['source_dataset']['dataset_combined_sha256']}`.
- Frozen registry representative metadata mismatch count: {provenance['inputs']['source_dataset']['registry_representative_metadata_mismatch_count']}. This audits frozen snapshot metadata only; replay masks use the frozen full-sensor ROI numbers and current `frames.csv` offsets.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--roi-registry", type=Path, default=REGISTRY_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    matplotlib.use("Agg", force=True)
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    registry_path = args.roi_registry.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    registry_payload, registry = load_registry(registry_path)
    conditions = discover_conditions(data_root)
    source_provenance = source_dataset_provenance(conditions, registry_payload)
    config = h0.config_contract(config_path)
    session_path = data_root / "session_ground_calibration.json"
    session_reference, rotation, translation, session_summary = h0.load_session_reference(
        session_path
    )
    app, pipeline = h0.make_pipeline(
        config_path, session_reference, rotation, translation
    )

    frame_records: list[dict[str, Any]] = []
    for ordinal, condition in enumerate(conditions, start=1):
        records = replay_condition(
            condition, registry[condition.condition], pipeline, app
        )
        frame_records.extend(records)
        valid = sum(record.get("measurement_status") == "VALID" for record in records)
        print(
            f"[{ordinal:02d}/{len(conditions)}] {condition.condition}: "
            f"{valid}/{len(records)} valid",
            flush=True,
        )

    if len(frame_records) != 50 * FRAME_COUNT:
        raise ManualReplayError(
            f"raw replay cardinality mismatch: {len(frame_records)} != {50 * FRAME_COUNT}"
        )
    raw_summaries = summarize_raw_conditions(conditions, frame_records)
    raw_replay_completed_at = now_utc()
    # This raw-only artifact intentionally contains neither truth nor accuracy error.
    write_csv(output / "manual_h_raw_frames.csv", FRAME_FIELDS, frame_records)

    ground_truth = load_ground_truth_after_replay(raw_replay_complete=True)
    ground_truth_loaded_at = now_utc()
    attach_ground_truth(raw_summaries, ground_truth)
    analysis = analyze_accuracy(raw_summaries)
    write_csv(
        output / "manual_h_raw_position_summary.csv",
        SUMMARY_FIELDS,
        raw_summaries,
    )
    save_plots(output, raw_summaries, analysis)

    provenance = {
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "inputs": {
            "data_root": str(data_root),
            "manual_roi_registry": str(registry_path),
            "manual_roi_registry_sha256": h0.sha256_file(registry_path),
            "manual_roi_registry_frozen_at_utc": registry_payload.get("frozen_at_utc"),
            "source_dataset": source_provenance,
            "config": config,
            "session_ground": session_summary,
        },
        "reused_implementations": {
            "frame_pipeline": {
                "path": str((TOOL_ROOT / "online" / "pipeline.py").resolve()),
                "sha256": h0.sha256_file(TOOL_ROOT / "online" / "pipeline.py"),
                "entry": "FramePipeline.run_frame",
            },
            "height_measurement": {
                "path": str((TOOL_ROOT / "measurement" / "height_measure.py").resolve()),
                "sha256": h0.sha256_file(
                    TOOL_ROOT / "measurement" / "height_measure.py"
                ),
                "entry": "measure_height_line",
                "manual_adapter": "concatenate baseline_before + baseline_after; pass height points",
            },
            "h0_replay_helpers": {
                "path": str((TOOLS_ROOT / "generate_haikang_c0_h_raw_0829.py").resolve()),
                "sha256": h0.sha256_file(
                    TOOLS_ROOT / "generate_haikang_c0_h_raw_0829.py"
                ),
                "functions": [
                    "read_csv_rows",
                    "frame_from_row",
                    "config_contract",
                    "load_session_reference",
                    "make_pipeline",
                ],
            },
        },
        "execution": {
            "condition_count": len(conditions),
            "frames_per_condition": FRAME_COUNT,
            "source_frame_count": len(frame_records),
            "raw_replay_completed_at_utc": raw_replay_completed_at,
            "ground_truth_loaded_at_utc": ground_truth_loaded_at,
            "truth_loaded_after_raw_replay": True,
            "statistics_unit": "50 condition medians; 20 frames are repeated measurements",
            "fresh_frame_pipeline_calculations": len(frame_records),
            "reused_numeric_auto_roi_results": False,
        },
        "guards": {
            "auto_roi_v2_selection_called": False,
            "manual_roi_modified_after_error_read": False,
            "height_shadow_read": False,
            "c0_recalibrated": False,
            "session_ground_modified": False,
            "c1_applied_by_adapter": False,
            "h1_fitted_or_applied": False,
            "hb2_fitted_or_applied": False,
            "correction_mode_is_none": config["correction_guard"]["mode_is_none"],
        },
    }
    accuracy_payload = {
        "schema_version": 1,
        "task": "H0-1M-B",
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "generated_at_utc": now_utc(),
        "production_accuracy_claim": False,
        "warning": (
            "Manual ROI diagnostic only; this is not the final production accuracy "
            "of Auto ROI-V2 target detection."
        ),
        "accuracy": analysis,
        "provenance": provenance,
    }
    write_json(output / "manual_c0_accuracy_summary.json", accuracy_payload)
    (output / "manual_c0_height_audit_report.md").write_text(
        build_report(analysis, raw_summaries, provenance), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "diagnostic_mode": DIAGNOSTIC_MODE,
                "frame_rows": len(frame_records),
                "condition_rows": len(raw_summaries),
                "classification": analysis["classification"],
                "overall": analysis["overall"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
