#!/usr/bin/env python3
"""Generate raw Haikang heights for the 2026-08-29 H0-1 audit.

This is a thin adapter around the existing runtime, Auto ROI V2 and height
measurement implementations.  It does not implement a second height formula.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "laser_measurement_tool"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from app_config import load_app_config  # noqa: E402
from measurement.ground_reference import MeasurementError, SessionGroundReference  # noqa: E402
from measurement.height_measure import measure_height_line  # noqa: E402
from online.models import CapturedFrame  # noqa: E402
from online.pipeline import FramePipeline  # noqa: E402
from utils.image_io import load_grayscale_image  # noqa: E402

import auto_roi_v2_session01 as roi_v2  # noqa: E402
import thermal_a2a_roi_v2 as roi_v2_wrapper  # noqa: E402


DATA_ROOT_DEFAULT = TOOL_ROOT / "output_haikang_0828" / "online_recordings" / "0829"
CONFIG_DEFAULT = TOOL_ROOT / "configs" / "measure_tool_haikang_0828.yaml"
TARGET_HEIGHT_MM = {
    "h02": 2.0,
    "h06": 6.0,
    "h10": 10.0,
    "h20": 20.0,
    "h30": 30.0,
}
A3_SUMMARY_FILENAME = "sigma_region_diagnostic_summary.json"
HEIGHT_DIR_RE = re.compile(r"^h\d+$")
POSITION_DIR_RE = re.compile(r"^(h\d+)_p(\d+)$")
FRAME_FIELDS = [
    "filename",
    "camera_frame_number",
    "camera_timestamp_ticks",
    "host_timestamp_ns",
    "host_monotonic_ns",
    "frame_gap",
    "exposure_us",
    "gain_db",
    "pixel_format",
    "offset_x",
    "offset_y",
    "width",
    "height",
]


class AuditError(RuntimeError):
    """Raised when the source contract cannot be audited safely."""


@dataclass(frozen=True, slots=True)
class Condition:
    height_id: str
    height_gt_mm: float
    position_id: str
    position_number: int
    path: Path

    @property
    def condition_id(self) -> str:
        return f"{self.height_id}_{self.position_id}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT_DEFAULT / "c0_height_audit" / "measurement",
    )
    return parser.parse_args(argv)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def parse_int(value: Any, name: str, *, allow_blank: bool = False) -> int | None:
    text = str(value or "").strip()
    if not text and allow_blank:
        return None
    if not text:
        raise AuditError(f"{name} is blank")
    number = finite(text)
    if number is None or number != int(number):
        raise AuditError(f"{name} is not an integer: {value!r}")
    return int(number)


def parse_float(value: Any, name: str, *, allow_blank: bool = False) -> float | None:
    text = str(value or "").strip()
    if not text and allow_blank:
        return None
    if not text:
        raise AuditError(f"{name} is blank")
    number = finite(text)
    if number is None:
        raise AuditError(f"{name} is not finite: {value!r}")
    return number


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return finite(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prior_a3_spatial_audit(root: Path) -> dict[str, Any]:
    """Load the prior A-3 spatial artifact for risk tagging only.

    This artifact is not used to select or modify an ROI.  Its three equal
    width regions define the previously reported right-side morphology area;
    the result is carried as provenance so a downstream audit cannot mistake
    an upstream spatial risk for a height compensation residual.
    """
    summary_path = (root.parent / "sigma_region_diagnostic" / A3_SUMMARY_FILENAME).resolve()
    result: dict[str, Any] = {
        "available": False,
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
        "classification": None,
        "right_u_full_range_px": None,
        "region_definition": None,
    }
    if not summary_path.is_file():
        return result
    try:
        document = json.loads(summary_path.read_text(encoding="utf-8"))
        roi = document.get("input", {}).get("roi") or {}
        left = int(roi.get("left") or roi["configured_left"])
        right_boundary = roi.get("right")
        width_value = roi.get("width", roi.get("configured_width"))
        if width_value is None and right_boundary is not None:
            width_value = int(right_boundary) - left
        width = int(width_value)
        if width <= 0:
            raise ValueError("A-3 ROI width must be positive")
        right_start = left + int(math.ceil((2.0 * width) / 3.0))
        right_end = (
            int(right_boundary) - 1
            if right_boundary is not None
            else left + width - 1
        )
        result.update(
            {
                "available": True,
                "classification": document.get("classification")
                or (document.get("diagnosis") or {}).get("classification"),
                "a3_roi": {
                    "left": left,
                    "top": int(roi.get("top") or roi["configured_top"]),
                    "width": width,
                    "height": int(
                        roi.get("height", roi.get("configured_height"))
                    ),
                },
                "right_u_full_range_px": [right_start, right_end],
                "region_definition": (
                    "third equal-width region of the prior A-3 ROI, inclusive; "
                    "derived from input.roi rather than from height data"
                ),
            }
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result["load_error"] = f"{type(error).__name__}:{error}"
    return result


def read_csv_rows(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing source table: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != expected_fields:
            raise AuditError(
                f"schema mismatch in {path}: expected {expected_fields}, got {header}"
            )
        rows: list[dict[str, str]] = []
        for values in reader:
            if not values or all(not str(value).strip() for value in values):
                continue
            if len(values) != len(header):
                raise AuditError(f"column count mismatch in {path}")
            rows.append(
                {field: str(values[index]).strip() for index, field in enumerate(header)}
            )
    if not rows:
        raise AuditError(f"no data rows in {path}")
    return rows


def discover_conditions(root: Path) -> tuple[list[Condition], dict[str, Any]]:
    if not root.is_dir():
        raise AuditError(f"input directory does not exist: {root}")
    conditions: list[Condition] = []
    discovered_heights: list[str] = []
    excluded_heights: list[dict[str, Any]] = []
    for height_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not HEIGHT_DIR_RE.fullmatch(height_dir.name):
            continue
        if height_dir.name not in TARGET_HEIGHT_MM:
            excluded_heights.append(
                {"height_id": height_dir.name, "reason": "outside H0-1 target set"}
            )
            continue
        discovered_heights.append(height_dir.name)
        for position_dir in sorted(path for path in height_dir.iterdir() if path.is_dir()):
            match = POSITION_DIR_RE.fullmatch(position_dir.name)
            if not match or match.group(1) != height_dir.name:
                continue
            number = int(match.group(2))
            conditions.append(
                Condition(
                    height_id=height_dir.name,
                    height_gt_mm=TARGET_HEIGHT_MM[height_dir.name],
                    position_id=f"p{number:02d}",
                    position_number=number,
                    path=position_dir,
                )
            )
    if not conditions:
        raise AuditError(f"no target height×position directories under {root}")
    conditions.sort(key=lambda item: (item.height_gt_mm, item.position_number))
    return conditions, {
        "discovered_height_ids": discovered_heights,
        "excluded_height_directories": excluded_heights,
        "discovered_position_ids": sorted({item.position_id for item in conditions}),
        "condition_count": len(conditions),
    }


def load_session_reference(
    session_path: Path,
) -> tuple[SessionGroundReference, np.ndarray, np.ndarray, dict[str, Any]]:
    if not session_path.is_file():
        raise AuditError(f"missing Session Ground file: {session_path}")
    document = json.loads(session_path.read_text(encoding="utf-8"))
    reference_data = document.get("session_ground_reference") or {}
    extrinsic_data = document.get("session_extrinsic") or {}
    if document.get("status") != "VALID" or not bool(document.get("valid", False)):
        raise AuditError("0829 Session Ground is not VALID")
    if reference_data.get("status") != "VALID":
        raise AuditError("0829 session_ground_reference.status is not VALID")

    def required_float(key: str) -> float:
        value = finite(reference_data.get(key))
        if value is None:
            raise AuditError(f"Session Ground field is missing/non-finite: {key}")
        return value

    origin = np.asarray(reference_data.get("origin_xy"), dtype=np.float64)
    direction = np.asarray(reference_data.get("direction_xy"), dtype=np.float64)
    valid_s = reference_data.get("valid_s_range_mm")
    if origin.shape != (2,) or direction.shape != (2,):
        raise AuditError("Session Ground origin/direction must each have two values")
    if not isinstance(valid_s, list) or len(valid_s) != 2:
        raise AuditError("Session Ground valid_s_range_mm must have two values")

    generation = parse_int(
        reference_data.get("ground_extrinsic_generation"),
        "ground_extrinsic_generation",
        allow_blank=True,
    )
    support = reference_data.get("support") or {}
    support_source = (
        reference_data.get("support_source") or support.get("source") or "unknown"
    )
    reference = SessionGroundReference(
        origin_xy=origin,
        direction_xy=direction,
        slope_z_per_mm=required_float("slope_z_per_mm"),
        intercept_z_mm=required_float("intercept_z_mm"),
        rmse_mm=required_float("rmse_mm"),
        valid_s_range_mm=(float(valid_s[0]), float(valid_s[1])),
        status=str(reference_data.get("status")),
        source=str(reference_data.get("source") or "session_laser_ground"),
        point_count=int(reference_data.get("point_count") or 0),
        inlier_count=int(reference_data.get("inlier_count") or 0),
        support_source=str(support_source),
        active_ground_extrinsic_source=reference_data.get(
            "active_ground_extrinsic_source"
        ),
        ground_extrinsic_generation=generation,
        frame_host_monotonic_ns=parse_int(
            reference_data.get("frame_host_monotonic_ns"),
            "frame_host_monotonic_ns",
            allow_blank=True,
        ),
        mask_inset_mm=finite(reference_data.get("mask_inset_mm")),
        support_metadata=support,
        coordinate=reference_data.get("coordinate"),
        coordinate_units=reference_data.get("coordinate_units"),
        coordinate_formula=reference_data.get("coordinate_formula"),
        frozen_json_path=reference_data.get("frozen_json_path"),
        frozen_json_sha256=reference_data.get("frozen_json_sha256"),
        frozen_schema_version=parse_int(
            reference_data.get("frozen_schema_version"),
            "frozen_schema_version",
            allow_blank=True,
        ),
        fit_pose_ids=tuple(reference_data.get("fit_pose_ids") or ()),
    )
    rotation = np.asarray(extrinsic_data.get("R_camera_to_ground"), dtype=np.float64)
    translation = np.asarray(extrinsic_data.get("t_camera_to_ground_mm"), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape not in {(3,), (3, 1)}:
        raise AuditError("Session Ground extrinsic R/t has an invalid shape")
    translation = translation.reshape(3)
    top_frame = document.get("frame") or {}
    summary = {
        "path": str(session_path.resolve()),
        "sha256": sha256_file(session_path),
        "schema_version": document.get("schema_version"),
        "status": document.get("status"),
        "valid": document.get("valid"),
        "source": reference.source,
        "support_source": reference.support_source,
        "active_ground_extrinsic_source": reference.active_ground_extrinsic_source,
        "ground_extrinsic_generation": reference.ground_extrinsic_generation,
        "session_generation": top_frame.get("session_generation"),
        "origin_xy": origin,
        "direction_xy": direction,
        "slope_z_per_mm": reference.slope_z_per_mm,
        "intercept_z_mm": reference.intercept_z_mm,
        "rmse_mm": reference.rmse_mm,
        "valid_s_range_mm": reference.valid_s_range_mm,
        "point_count": reference.point_count,
        "inlier_count": reference.inlier_count,
        "support": support,
    }
    return reference, rotation, translation, summary


def config_contract(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        raw = yaml.safe_load(text) or {}
    except Exception as error:
        raise AuditError(f"cannot parse Haikang config: {error}") from error
    correction = raw.get("correction") or {}
    reconstruction = raw.get("reconstruction") or {}
    if correction.get("mode") != "none":
        raise AuditError("Haikang config correction.mode is not none")
    if correction.get("stage_a_height_scale_enabled") is not False:
        raise AuditError("Haikang config enables stage-A height scale")
    if correction.get("hb2_height_correction_config") is not None:
        raise AuditError("Haikang config contains an H-B2 correction config")
    if reconstruction.get("image_roi_polygon") is not None:
        raise AuditError("Haikang config has an image ROI polygon not audited here")
    return {
        "path": str(config_path.resolve()),
        "sha256": sha256_file(config_path),
        "system": raw.get("system"),
        "camera": raw.get("camera"),
        "calibration": raw.get("calibration"),
        "correction": correction,
        "extraction": raw.get("extraction"),
        "reconstruction": reconstruction,
        "measurement": raw.get("measurement"),
        "correction_guard": {
            "mode_is_none": True,
            "stage_a_disabled": True,
            "hb2_config_absent": True,
            "c1_not_enabled_in_config": True,
            "ground_u_compensation_absent": (
                (raw.get("calibration") or {}).get("ground_u_compensation") is None
            ),
        },
    }


def make_pipeline(
    config_path: Path,
    reference: SessionGroundReference,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[Any, FramePipeline]:
    app = load_app_config(config_path)
    pipeline = FramePipeline(app, system="mvs")
    pipeline.apply_session_ground_extrinsic(
        rotation,
        translation,
        generation=reference.ground_extrinsic_generation,
    )
    pipeline.apply_session_ground_reference(reference)
    return app, pipeline


def frame_from_row(condition: Condition, row: dict[str, str]) -> CapturedFrame:
    image_path = (condition.path / row["filename"]).resolve()
    try:
        image_path.relative_to(condition.path.resolve())
    except ValueError as error:
        raise AuditError(f"frame filename escapes position directory: {row['filename']}") from error
    image = load_grayscale_image(image_path)
    width = parse_int(row["width"], "width")
    height = parse_int(row["height"], "height")
    if tuple(image.shape) != (height, width):
        raise AuditError(
            f"image shape mismatch for {image_path}: {image.shape} != {(height, width)}"
        )
    return CapturedFrame(
        image=image,
        camera_frame_number=int(parse_int(row["camera_frame_number"], "camera_frame_number")),
        camera_timestamp_ticks=parse_int(
            row["camera_timestamp_ticks"],
            "camera_timestamp_ticks",
            allow_blank=True,
        ),
        host_timestamp_ns=int(parse_int(row["host_timestamp_ns"], "host_timestamp_ns")),
        host_monotonic_ns=int(parse_int(row["host_monotonic_ns"], "host_monotonic_ns")),
        offset_x=int(parse_int(row["offset_x"], "offset_x")),
        offset_y=int(parse_int(row["offset_y"], "offset_y")),
    )


def axis_adapter(centers_uv_full: np.ndarray) -> np.ndarray:
    """Map Haikang column scan to the ROI-V2 row-scan coordinate contract."""
    values = np.asarray(centers_uv_full, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise AuditError(f"unexpected center shape: {values.shape}")
    # ROI-V2 expects (u(v), v).  Haikang supplies (u, v(u)).
    return np.ascontiguousarray(values[:, [1, 0]], dtype=np.float64)


def inclusive_mask(
    pixels_uv: np.ndarray | None,
    interval: list[int] | None,
) -> np.ndarray:
    if pixels_uv is None:
        return np.zeros(0, dtype=bool)
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise AuditError(f"unexpected reconstructed pixel shape: {pixels.shape}")
    if not interval or len(interval) != 2:
        return np.zeros(len(pixels), dtype=bool)
    low, high = float(interval[0]), float(interval[1])
    return np.isfinite(pixels[:, 0]) & (pixels[:, 0] >= low) & (pixels[:, 0] <= high)


def interval_overlap_px(
    interval: Any,
    reference_interval: Any,
) -> int:
    """Return inclusive integer-pixel overlap without changing either interval."""
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return 0
    if not isinstance(reference_interval, (list, tuple)) or len(reference_interval) != 2:
        return 0
    try:
        low = max(int(math.ceil(float(interval[0]))), int(math.ceil(float(reference_interval[0]))))
        high = min(int(math.floor(float(interval[1]))), int(math.floor(float(reference_interval[1]))))
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, high - low + 1)


def a3_overlap_fields(
    roi: dict[str, Any],
    a3_audit: dict[str, Any],
) -> dict[str, Any]:
    right_range = a3_audit.get("right_u_full_range_px")
    height_overlap = interval_overlap_px(
        roi.get("height_u_full_range_px"), right_range
    )
    baseline_before_overlap = interval_overlap_px(
        roi.get("baseline_before_u_full_range_px"), right_range
    )
    baseline_after_overlap = interval_overlap_px(
        roi.get("baseline_after_u_full_range_px"), right_range
    )
    baseline_overlap = baseline_before_overlap + baseline_after_overlap
    any_overlap = bool(
        a3_audit.get("available") and (height_overlap > 0 or baseline_overlap > 0)
    )
    return {
        "a3_right_morphology_region_available": bool(a3_audit.get("available")),
        "a3_right_morphology_region_u_full_range_px": right_range,
        "a3_right_height_roi_overlap_px": height_overlap,
        "a3_right_baseline_before_overlap_px": baseline_before_overlap,
        "a3_right_baseline_after_overlap_px": baseline_after_overlap,
        "a3_right_baseline_overlap_px": baseline_overlap,
        "a3_right_morphology_region_overlap": any_overlap,
        "a3_spatial_risk_reason": (
            "A3_RIGHT_STRIPE_MORPHOLOGY_REGION_OVERLAP" if any_overlap else None
        ),
    }


def h_raw_quality_fields(
    roi: dict[str, Any],
    h_raw_available: bool,
) -> dict[str, Any]:
    """Classify usability without changing the raw measurement value."""
    a3_risk = bool(roi.get("a3_right_morphology_region_overlap"))
    roi_uncertain = roi.get("status") != "PASS"
    if not h_raw_available:
        quality = "NOT_AVAILABLE"
    elif roi_uncertain and a3_risk:
        quality = "ROI_V2_UNCERTAIN_AND_A3_RIGHT_RISK"
    elif roi_uncertain:
        quality = "ROI_V2_UNCERTAIN"
    elif a3_risk:
        quality = "UPSTREAM_A3_RIGHT_RISK"
    else:
        quality = "ELIGIBLE"
    eligible = bool(h_raw_available and not roi_uncertain and not a3_risk)
    return {
        "h_raw_reliability": quality,
        "h_raw_reliable": eligible,
        "h_raw_eligible": eligible,
    }


def optional_measure(
    baseline_ground: np.ndarray,
    height_ground: np.ndarray,
    app: Any,
    mode: str,
) -> tuple[Any | None, str | None]:
    if len(height_ground) < int(app.measurement.min_height_points):
        return None, "HEIGHT_POINTS_INSUFFICIENT"
    if mode == "auto" and len(baseline_ground) < int(
        app.measurement.min_baseline_points
    ):
        return None, "BASELINE_POINTS_INSUFFICIENT"
    try:
        return (
            measure_height_line(
                baseline_ground,
                height_ground,
                app.measurement,
                ground_correction_mode=mode,
            ),
            None,
        )
    except (MeasurementError, ValueError, FloatingPointError) as error:
        return None, f"{type(error).__name__}:{error}"


def measurement_fields(measurement: Any | None, prefix: str) -> dict[str, Any]:
    fields = {
        f"{prefix}_measurement_status": "NOT_MEASURED",
        f"{prefix}_ground_reference_mode": None,
        f"{prefix}_height_mean_mm": None,
        f"{prefix}_height_median_mm": None,
        f"{prefix}_height_std_mm": None,
        f"{prefix}_ground_baseline_zg_mm": None,
        f"{prefix}_ground_noise_sigma_mm": None,
        f"{prefix}_baseline_point_count": 0,
        f"{prefix}_baseline_inlier_count": 0,
        f"{prefix}_height_point_count": 0,
        f"{prefix}_height_inlier_count": 0,
        f"{prefix}_local_ground_fit_slope_mm_per_mm": None,
        f"{prefix}_local_ground_fit_intercept_mm": None,
        f"{prefix}_local_ground_fit_rmse_mm": None,
    }
    if measurement is None:
        return fields
    fields.update(
        {
            f"{prefix}_measurement_status": "VALID",
            f"{prefix}_ground_reference_mode": measurement.ground_reference_mode,
            f"{prefix}_height_mean_mm": float(measurement.height_mean_mm),
            f"{prefix}_height_median_mm": float(measurement.height_median_mm),
            f"{prefix}_height_std_mm": float(measurement.height_std_mm),
            f"{prefix}_ground_baseline_zg_mm": float(
                measurement.ground_baseline_zg_mm
            ),
            f"{prefix}_ground_noise_sigma_mm": finite(
                measurement.ground_noise_sigma_mm
            ),
            f"{prefix}_baseline_point_count": int(measurement.baseline_point_count),
            f"{prefix}_baseline_inlier_count": int(measurement.baseline_inlier_count),
            f"{prefix}_height_point_count": int(measurement.height_point_count),
            f"{prefix}_height_inlier_count": int(measurement.height_inlier_count),
        }
    )
    fit = measurement.ground_profile_fit
    if fit is not None:
        fields.update(
            {
                f"{prefix}_local_ground_fit_slope_mm_per_mm": float(
                    fit.slope_z_per_mm
                ),
                f"{prefix}_local_ground_fit_intercept_mm": float(
                    fit.intercept_z_mm
                ),
                f"{prefix}_local_ground_fit_rmse_mm": float(fit.rmse_mm),
            }
        )
    return fields


def roi_payload(
    assessment: dict[str, Any],
    median_scan: np.ndarray | None,
) -> dict[str, Any]:
    height_range = assessment.get("height_v_range") or []
    baseline_ranges = list(assessment.get("baseline_v_ranges") or [[], []])
    baseline_ranges = (baseline_ranges + [[], []])[:2]
    edge_data = assessment.get("detected_edges") or {}
    return {
        "method": "auto_roi_v2",
        "axis_adapter": "swap_full_uv_to_v_scan",
        "source_coordinates": "Haikang full-sensor (u,v)",
        "detector_coordinates": "ROI-V2 synthetic (u'=v,v'=u)",
        "mask_axis": "u_full_px",
        "status": assessment.get("auto_qc_status", "FAIL"),
        "reasons": assessment.get("auto_qc_reasons", []),
        "height_u_full_range_px": height_range,
        "baseline_before_u_full_range_px": baseline_ranges[0],
        "baseline_after_u_full_range_px": baseline_ranges[1],
        "edge1_u_full_px": edge_data.get("v_edge_1"),
        "edge2_u_full_px": edge_data.get("v_edge_2"),
        "object_width_px": assessment.get("object_width_px"),
        "height_interior_width_px": assessment.get("height_interior_width_px"),
        "transition_exclusion_margin_px": assessment.get(
            "transition_exclusion_margin_px"
        ),
        "step_amplitude_cross_axis_px": assessment.get("plateau_delta_u_px"),
        "candidate_pair_score": assessment.get("selected_pair_score"),
        "candidate_pair_count": len(assessment.get("all_edge_pairs") or []),
        "repeat_support": assessment.get("repeat_support") or {},
        "median_centerline_point_count": (
            int(len(median_scan)) if median_scan is not None else 0
        ),
        "geometry_only": True,
        "height_or_3d_inputs_used": False,
    }


def failed_frame(
    condition: Condition,
    row: dict[str, str],
    *,
    status: str,
    error: str,
    roi: dict[str, Any] | None = None,
    a3_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_roi = dict(roi or {})
    if not effective_roi.get("a3_right_morphology_region_available"):
        effective_roi.update(a3_overlap_fields(effective_roi, a3_audit or {}))
    return {
        "height_gt_mm": condition.height_gt_mm,
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "condition_id": condition.condition_id,
        "frame": parse_int(row.get("camera_frame_number"), "camera_frame_number"),
        "filename": row.get("filename"),
        "host_timestamp_ns": parse_int(row.get("host_timestamp_ns"), "host_timestamp_ns", allow_blank=True),
        "exposure_us": parse_float(row.get("exposure_us"), "exposure_us", allow_blank=True),
        "gain_db": parse_float(row.get("gain_db"), "gain_db", allow_blank=True),
        "pixel_format": row.get("pixel_format"),
        "offset_x": parse_int(row.get("offset_x"), "offset_x", allow_blank=True),
        "offset_y": parse_int(row.get("offset_y"), "offset_y", allow_blank=True),
        "image_width": parse_int(row.get("width"), "width", allow_blank=True),
        "image_height": parse_int(row.get("height"), "height", allow_blank=True),
        "roi_v2": effective_roi,
        "roi_v2_status": effective_roi.get("status", "NOT_RUN"),
        "a3_right_morphology_region_overlap": effective_roi.get(
            "a3_right_morphology_region_overlap", False
        ),
        "a3_right_height_roi_overlap_px": effective_roi.get(
            "a3_right_height_roi_overlap_px", 0
        ),
        "a3_right_baseline_overlap_px": effective_roi.get(
            "a3_right_baseline_overlap_px", 0
        ),
        "center_point_count": 0,
        "valid_points": 0,
        "session_ground_status": "NOT_RUN",
        "session_ground_source": None,
        "session_ground_generation": None,
        "session_ground_applied_count": 0,
        "session_ground_out_of_range_count": 0,
        "session_ground_valid_s_range_mm": None,
        "height_roi_point_count": 0,
        "baseline_before_point_count": 0,
        "baseline_after_point_count": 0,
        "baseline_point_count": 0,
        "height_raw_mm": None,
        "extraction_status": status,
        "extraction_error": error,
        "q1_c0": None,
        "q2_c0": None,
        "q2_in_domain": None,
        **h_raw_quality_fields(effective_roi, False),
        **measurement_fields(None, "session"),
        **measurement_fields(None, "local"),
    }


def run_condition(
    condition: Condition,
    pipeline: FramePipeline,
    app: Any,
    a3_audit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    source_rows = read_csv_rows(condition.path / "frames.csv", FRAME_FIELDS)
    results: list[Any | None] = []
    scan_arrays: list[np.ndarray] = []
    errors: dict[int, str] = {}
    for index, row in enumerate(source_rows):
        try:
            result = pipeline.run_frame(frame_from_row(condition, row))
            results.append(result)
            scan_arrays.append(axis_adapter(result.centers_uv_full))
        except Exception as error:
            results.append(None)
            scan_arrays.append(np.empty((0, 2), dtype=np.float64))
            errors[index] = f"{type(error).__name__}:{error}"

    median_scan: np.ndarray | None = None
    assessment: dict[str, Any] = {
        "condition_id": condition.condition_id,
        "height_label": condition.height_id,
        "position_id": condition.position_id,
        "auto_qc_status": "FAIL",
        "auto_qc_reasons": [],
        "all_edge_pairs": [],
        "repeat_support": {},
    }
    try:
        median_scan = roi_v2_wrapper.median_centerline(scan_arrays)
        assessment = roi_v2.assess_condition(
            condition.condition_id,
            median_scan,
            scan_arrays,
            {},
        )
    except Exception as error:
        assessment["auto_qc_reasons"] = [
            f"ROI_V2_ERROR:{type(error).__name__}:{error}"
        ]

    roi = roi_payload(assessment, median_scan)
    ranges = [
        roi.get("height_u_full_range_px"),
        roi.get("baseline_before_u_full_range_px"),
        roi.get("baseline_after_u_full_range_px"),
    ]
    has_ranges = bool(ranges[0]) and all(
        isinstance(item, list) and len(item) == 2 for item in ranges[1:]
    )
    if not has_ranges:
        roi["status"] = "FAIL"
        roi["reasons"] = list(roi["reasons"]) + [
            "roi_v2_candidate_ranges_unavailable"
        ]
    roi.update(a3_overlap_fields(roi, a3_audit))

    height_range = roi.get("height_u_full_range_px") or []
    before_range = roi.get("baseline_before_u_full_range_px") or []
    after_range = roi.get("baseline_after_u_full_range_px") or []
    frame_records: list[dict[str, Any]] = []

    for index, (row, result) in enumerate(zip(source_rows, results)):
        if result is None:
            frame_records.append(
                failed_frame(
                    condition,
                    row,
                    status="FRAME_PIPELINE_ERROR",
                    error=errors.get(index, "unknown frame error"),
                    roi=roi,
                    a3_audit=a3_audit,
                )
            )
            continue
        pixels = (
            np.empty((0, 2), dtype=np.float64)
            if result.pixels_uv is None
            else np.asarray(result.pixels_uv, dtype=np.float64)
        )
        points = np.asarray(result.points_ground, dtype=np.float64)
        if len(pixels) != len(points):
            frame_records.append(
                failed_frame(
                    condition,
                    row,
                    status="ALIGNMENT_ERROR",
                    error=f"pixels_points_length_mismatch:{len(pixels)}!={len(points)}",
                    roi=roi,
                    a3_audit=a3_audit,
                )
            )
            continue

        before_mask = inclusive_mask(pixels, before_range) if has_ranges else np.zeros(
            len(pixels), dtype=bool
        )
        height_mask = inclusive_mask(pixels, height_range) if has_ranges else np.zeros(
            len(pixels), dtype=bool
        )
        after_mask = inclusive_mask(pixels, after_range) if has_ranges else np.zeros(
            len(pixels), dtype=bool
        )
        baseline_mask = before_mask | after_mask
        before_ground = points[before_mask]
        height_ground = points[height_mask]
        after_ground = points[after_mask]
        baseline_ground = points[baseline_mask]

        if has_ranges:
            session_measurement, session_error = optional_measure(
                baseline_ground, height_ground, app, "session_reference"
            )
            local_measurement, local_error = optional_measure(
                baseline_ground, height_ground, app, "auto"
            )
        else:
            session_measurement, session_error = None, "ROI_V2_UNAVAILABLE"
            local_measurement, local_error = None, "ROI_V2_UNAVAILABLE"

        if local_measurement is not None:
            h_raw = float(local_measurement.height_mean_mm)
            status = "PASS" if roi["status"] == "PASS" else "VALID_ROI_V2_UNCERTAIN"
        else:
            h_raw = None
            status = (
                "ROI_V2_UNCERTAIN_MEASUREMENT_INVALID"
                if roi["status"] != "PASS"
                else "LOCAL_MEASUREMENT_INVALID"
            )
        measurement_errors = [item for item in [session_error, local_error] if item]
        frame_records.append(
            {
                "height_gt_mm": condition.height_gt_mm,
                "height_id": condition.height_id,
                "position_id": condition.position_id,
                "condition_id": condition.condition_id,
                "frame": int(result.frame.camera_frame_number),
                "filename": row["filename"],
                "host_timestamp_ns": int(result.frame.host_timestamp_ns),
                "exposure_us": parse_float(row["exposure_us"], "exposure_us"),
                "gain_db": parse_float(row["gain_db"], "gain_db"),
                "pixel_format": row["pixel_format"],
                "offset_x": int(result.frame.offset_x),
                "offset_y": int(result.frame.offset_y),
                "image_width": parse_int(row["width"], "width"),
                "image_height": parse_int(row["height"], "height"),
                "roi_v2": roi,
                "roi_v2_status": roi["status"],
                "a3_right_morphology_region_overlap": roi.get(
                    "a3_right_morphology_region_overlap", False
                ),
                "a3_right_height_roi_overlap_px": roi.get(
                    "a3_right_height_roi_overlap_px", 0
                ),
                "a3_right_baseline_overlap_px": roi.get(
                    "a3_right_baseline_overlap_px", 0
                ),
                "center_point_count": int(len(result.centers_uv_full)),
                "valid_points": int(len(points)),
                "session_ground_status": result.ground_reference_status,
                "session_ground_source": result.ground_reference_source,
                "session_ground_generation": result.ground_extrinsic_generation,
                "session_ground_applied_count": int(result.ground_reference_applied_count),
                "session_ground_out_of_range_count": int(
                    result.ground_reference_out_of_range_count
                ),
                "session_ground_valid_s_range_mm": result.ground_reference_valid_s_range_mm,
                "height_roi_point_count": int(len(height_ground)),
                "baseline_before_point_count": int(len(before_ground)),
                "baseline_after_point_count": int(len(after_ground)),
                "baseline_point_count": int(len(baseline_ground)),
                "height_raw_mm": h_raw,
                "extraction_status": status,
                "extraction_error": ";".join(measurement_errors),
                "q1_c0": finite(result.q1),
                "q2_c0": finite(result.q2),
                "q2_in_domain": result.q2_in_domain,
                **h_raw_quality_fields(roi, h_raw is not None),
                **measurement_fields(session_measurement, "session"),
                **measurement_fields(local_measurement, "local"),
            }
        )

    summary = summarize_condition(condition, source_rows, frame_records, roi)
    registry = registry_row_for(condition, source_rows, frame_records, roi, median_scan)
    return summary, frame_records, registry


def values(records: Iterable[dict[str, Any]], field: str) -> np.ndarray:
    result = [finite(record.get(field)) for record in records]
    return np.asarray([item for item in result if item is not None], dtype=np.float64)


def stats(array: np.ndarray) -> dict[str, Any]:
    if len(array) == 0:
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
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_condition(
    condition: Condition,
    source_rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    roi: dict[str, Any],
) -> dict[str, Any]:
    raw_stats = stats(values(records, "height_raw_mm"))
    session_stats = stats(values(records, "session_height_mean_mm"))
    status_counts = Counter(record.get("extraction_status") for record in records)
    ground_counts = Counter(record.get("session_ground_status") for record in records)
    quality_counts = Counter(record.get("h_raw_reliability") for record in records)
    valid_count = int(raw_stats["count"])
    reliable_count = sum(bool(record.get("h_raw_reliable")) for record in records)
    return {
        "height_gt_mm": condition.height_gt_mm,
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "condition_id": condition.condition_id,
        "source_path": str(condition.path.resolve()),
        "frame_count": len(records),
        "source_frame_row_count": len(source_rows),
        "expected_frame_count": 20,
        "roi_v2_status": roi.get("status"),
        "roi_v2_reliable": roi.get("status") == "PASS",
        "roi_v2_reasons": roi.get("reasons", []),
        "roi_v2_height_u_full_range_px": roi.get("height_u_full_range_px"),
        "roi_v2_baseline_before_u_full_range_px": roi.get(
            "baseline_before_u_full_range_px"
        ),
        "roi_v2_baseline_after_u_full_range_px": roi.get(
            "baseline_after_u_full_range_px"
        ),
        "roi_v2_valid_center_rows": roi.get("median_centerline_point_count"),
        "roi_v2_candidate_pair_count": roi.get("candidate_pair_count", 0),
        "a3_right_morphology_region_overlap": bool(
            roi.get("a3_right_morphology_region_overlap")
        ),
        "a3_right_height_roi_overlap_px": roi.get(
            "a3_right_height_roi_overlap_px", 0
        ),
        "a3_right_baseline_overlap_px": roi.get(
            "a3_right_baseline_overlap_px", 0
        ),
        "h_raw_reliability": (
            quality_counts.most_common(1)[0][0] if quality_counts else "NOT_AVAILABLE"
        ),
        "h_raw_reliable_frame_count": reliable_count,
        "h_raw_reliable_frame_ratio": (
            reliable_count / len(records) if records else 0.0
        ),
        "h_raw_eligible": bool(records) and reliable_count == len(records),
        "h_raw_reliability_counts": dict(quality_counts),
        "h_raw_definition": "local_measurement.height_mean_mm",
        "h_raw_mm_median": raw_stats["median"],
        "h_raw_mm_mean": raw_stats["mean"],
        "h_raw_temporal_std_mm": raw_stats["std"],
        "h_raw_mm_p05": raw_stats["p05"],
        "h_raw_mm_p95": raw_stats["p95"],
        "h_raw_mm_min": raw_stats["min"],
        "h_raw_mm_max": raw_stats["max"],
        "h_raw_valid_frame_count": valid_count,
        "h_raw_valid_frame_ratio": valid_count / len(records) if records else 0.0,
        "h_raw_error_median_mm": (
            raw_stats["median"] - condition.height_gt_mm
            if raw_stats["median"] is not None
            else None
        ),
        "session_h_raw_mm_median": session_stats["median"],
        "session_h_raw_mm_mean": session_stats["mean"],
        "session_h_raw_temporal_std_mm": session_stats["std"],
        "session_valid_frame_count": int(session_stats["count"]),
        "height_roi_point_count_median": stats(
            values(records, "height_roi_point_count")
        )["median"],
        "height_roi_point_count_p05": stats(
            values(records, "height_roi_point_count")
        )["p05"],
        "height_roi_point_count_p95": stats(
            values(records, "height_roi_point_count")
        )["p95"],
        "height_inlier_count_median": stats(
            values(records, "local_height_inlier_count")
        )["median"],
        "baseline_point_count_median": stats(
            values(records, "baseline_point_count")
        )["median"],
        "valid_points_median": stats(values(records, "valid_points"))["median"],
        "valid_points_p05": stats(values(records, "valid_points"))["p05"],
        "valid_points_p95": stats(values(records, "valid_points"))["p95"],
        "session_ground_status_counts": dict(ground_counts),
        "session_ground_generations": sorted(
            {
                int(value)
                for value in (
                    record.get("session_ground_generation") for record in records
                )
                if value is not None
            }
        ),
        "extraction_status_counts": dict(status_counts),
        "local_measurement_valid_frame_count": sum(
            record.get("local_measurement_status") == "VALID" for record in records
        ),
        "local_measurement_error_counts": dict(
            Counter(
                record.get("extraction_error")
                for record in records
                if record.get("extraction_error")
            )
        ),
    }


def registry_row_for(
    condition: Condition,
    source_rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    roi: dict[str, Any],
    median_scan: np.ndarray | None,
) -> dict[str, Any]:
    support = roi.get("repeat_support") or {}
    height_support = support.get("height") or {}
    before_support = support.get("baseline_before") or {}
    after_support = support.get("baseline_after") or {}

    def endpoint(key: str, index: int) -> Any:
        value = roi.get(key) or [None, None]
        return value[index]

    return {
        "height_id": condition.height_id,
        "height_gt_mm": condition.height_gt_mm,
        "position_id": condition.position_id,
        "condition_id": condition.condition_id,
        "source_path": str(condition.path.resolve()),
        "source_frame_count": len(source_rows),
        "pipeline_success_frame_count": sum(
            record.get("session_ground_status") != "NOT_RUN" for record in records
        ),
        "axis_adapter": roi.get("axis_adapter"),
        "detector_coordinates": roi.get("detector_coordinates"),
        "mask_axis": roi.get("mask_axis"),
        "roi_v2_status": roi.get("status"),
        "roi_v2_reasons": roi.get("reasons"),
        "a3_right_morphology_region_overlap": bool(
            roi.get("a3_right_morphology_region_overlap")
        ),
        "a3_right_height_roi_overlap_px": roi.get(
            "a3_right_height_roi_overlap_px", 0
        ),
        "a3_right_baseline_overlap_px": roi.get(
            "a3_right_baseline_overlap_px", 0
        ),
        "height_u_full_start_px": endpoint("height_u_full_range_px", 0),
        "height_u_full_end_px": endpoint("height_u_full_range_px", 1),
        "baseline_before_u_full_start_px": endpoint(
            "baseline_before_u_full_range_px", 0
        ),
        "baseline_before_u_full_end_px": endpoint(
            "baseline_before_u_full_range_px", 1
        ),
        "baseline_after_u_full_start_px": endpoint(
            "baseline_after_u_full_range_px", 0
        ),
        "baseline_after_u_full_end_px": endpoint(
            "baseline_after_u_full_range_px", 1
        ),
        "edge1_u_full_px": roi.get("edge1_u_full_px"),
        "edge2_u_full_px": roi.get("edge2_u_full_px"),
        "object_width_px": roi.get("object_width_px"),
        "height_interior_width_px": roi.get("height_interior_width_px"),
        "transition_exclusion_margin_px": roi.get(
            "transition_exclusion_margin_px"
        ),
        "step_amplitude_cross_axis_px": roi.get("step_amplitude_cross_axis_px"),
        "candidate_pair_score": roi.get("candidate_pair_score"),
        "candidate_pair_count": roi.get("candidate_pair_count"),
        "median_centerline_point_count": (
            int(len(median_scan)) if median_scan is not None else 0
        ),
        "height_support_min_points": height_support.get("min_points"),
        "height_support_median_points": height_support.get("median_points"),
        "height_support_ok": height_support.get("support_ok"),
        "baseline_before_support_min_points": before_support.get("min_points"),
        "baseline_before_support_ok": before_support.get("support_ok"),
        "baseline_after_support_min_points": after_support.get("min_points"),
        "baseline_after_support_ok": after_support.get("support_ok"),
        "valid_points_median": stats(values(records, "valid_points"))["median"],
        "height_roi_points_median": stats(
            values(records, "height_roi_point_count")
        )["median"],
        "local_measurement_valid_frame_count": sum(
            record.get("local_measurement_status") == "VALID" for record in records
        ),
        "h_raw_valid_frame_count": sum(
            finite(record.get("height_raw_mm")) is not None for record in records
        ),
        "h_raw_reliable_frame_count": sum(
            bool(record.get("h_raw_reliable")) for record in records
        ),
        "h_raw_eligible": bool(records)
        and all(bool(record.get("h_raw_eligible")) for record in records),
        "extraction_status": (
            "PASS"
            if all(record.get("extraction_status") == "PASS" for record in records)
            else "REVIEW"
        ),
    }


FRAME_CSV_FIELDS = [
    "height_gt_mm",
    "height_id",
    "position_id",
    "condition_id",
    "frame",
    "filename",
    "host_timestamp_ns",
    "exposure_us",
    "gain_db",
    "pixel_format",
    "offset_x",
    "offset_y",
    "image_width",
    "image_height",
    "roi_v2",
    "roi_v2_status",
    "a3_right_morphology_region_overlap",
    "a3_right_height_roi_overlap_px",
    "a3_right_baseline_overlap_px",
    "center_point_count",
    "valid_points",
    "session_ground_status",
    "session_ground_source",
    "session_ground_generation",
    "session_ground_applied_count",
    "session_ground_out_of_range_count",
    "session_ground_valid_s_range_mm",
    "height_roi_point_count",
    "baseline_before_point_count",
    "baseline_after_point_count",
    "baseline_point_count",
    "session_measurement_status",
    "session_ground_reference_mode",
    "session_height_mean_mm",
    "session_height_median_mm",
    "session_height_std_mm",
    "session_ground_baseline_zg_mm",
    "session_ground_noise_sigma_mm",
    "session_baseline_point_count",
    "session_baseline_inlier_count",
    "session_height_point_count",
    "session_height_inlier_count",
    "local_measurement_status",
    "local_ground_reference_mode",
    "local_height_mean_mm",
    "local_height_median_mm",
    "local_height_std_mm",
    "local_ground_baseline_zg_mm",
    "local_ground_noise_sigma_mm",
    "local_baseline_point_count",
    "local_baseline_inlier_count",
    "local_height_point_count",
    "local_height_inlier_count",
    "local_local_ground_fit_slope_mm_per_mm",
    "local_local_ground_fit_intercept_mm",
    "local_local_ground_fit_rmse_mm",
    "height_raw_mm",
    "q1_c0",
    "q2_c0",
    "q2_in_domain",
    "h_raw_reliability",
    "h_raw_reliable",
    "h_raw_eligible",
    "extraction_status",
    "extraction_error",
]

SUMMARY_CSV_FIELDS = [
    "height_gt_mm",
    "height_id",
    "position_id",
    "condition_id",
    "source_path",
    "frame_count",
    "source_frame_row_count",
    "expected_frame_count",
    "roi_v2_status",
    "roi_v2_reliable",
    "roi_v2_reasons",
    "roi_v2_height_u_full_range_px",
    "roi_v2_baseline_before_u_full_range_px",
    "roi_v2_baseline_after_u_full_range_px",
    "roi_v2_valid_center_rows",
    "roi_v2_candidate_pair_count",
    "a3_right_morphology_region_overlap",
    "a3_right_height_roi_overlap_px",
    "a3_right_baseline_overlap_px",
    "h_raw_reliability",
    "h_raw_reliable_frame_count",
    "h_raw_reliable_frame_ratio",
    "h_raw_eligible",
    "h_raw_reliability_counts",
    "h_raw_definition",
    "h_raw_mm_median",
    "h_raw_mm_mean",
    "h_raw_temporal_std_mm",
    "h_raw_mm_p05",
    "h_raw_mm_p95",
    "h_raw_mm_min",
    "h_raw_mm_max",
    "h_raw_valid_frame_count",
    "h_raw_valid_frame_ratio",
    "h_raw_error_median_mm",
    "session_h_raw_mm_median",
    "session_h_raw_mm_mean",
    "session_h_raw_temporal_std_mm",
    "session_valid_frame_count",
    "height_roi_point_count_median",
    "height_roi_point_count_p05",
    "height_roi_point_count_p95",
    "height_inlier_count_median",
    "baseline_point_count_median",
    "valid_points_median",
    "valid_points_p05",
    "valid_points_p95",
    "session_ground_status_counts",
    "session_ground_generations",
    "extraction_status_counts",
    "local_measurement_valid_frame_count",
    "local_measurement_error_counts",
]

REGISTRY_CSV_FIELDS = [
    "height_id",
    "height_gt_mm",
    "position_id",
    "condition_id",
    "source_path",
    "source_frame_count",
    "pipeline_success_frame_count",
    "axis_adapter",
    "detector_coordinates",
    "mask_axis",
    "roi_v2_status",
    "roi_v2_reasons",
    "a3_right_morphology_region_overlap",
    "a3_right_height_roi_overlap_px",
    "a3_right_baseline_overlap_px",
    "height_u_full_start_px",
    "height_u_full_end_px",
    "baseline_before_u_full_start_px",
    "baseline_before_u_full_end_px",
    "baseline_after_u_full_start_px",
    "baseline_after_u_full_end_px",
    "edge1_u_full_px",
    "edge2_u_full_px",
    "object_width_px",
    "height_interior_width_px",
    "transition_exclusion_margin_px",
    "step_amplitude_cross_axis_px",
    "candidate_pair_score",
    "candidate_pair_count",
    "median_centerline_point_count",
    "height_support_min_points",
    "height_support_median_points",
    "height_support_ok",
    "baseline_before_support_min_points",
    "baseline_before_support_ok",
    "baseline_after_support_min_points",
    "baseline_after_support_ok",
    "valid_points_median",
    "height_roi_points_median",
    "local_measurement_valid_frame_count",
    "h_raw_valid_frame_count",
    "h_raw_reliable_frame_count",
    "h_raw_eligible",
    "extraction_status",
]


def build_provenance(
    *,
    root: Path,
    config_summary: dict[str, Any],
    session_summary: dict[str, Any],
    discovery: dict[str, Any],
    conditions: list[Condition],
    summaries: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    a3_audit: dict[str, Any],
) -> dict[str, Any]:
    manifest_value = (config_summary.get("calibration") or {}).get("manifest")
    manifest_path = (
        (Path(config_summary["path"]).parent / str(manifest_value)).resolve()
        if manifest_value
        else None
    )
    daheng_parameters = (
        REPO_ROOT
        / "reports"
        / "experiments"
        / "daheng_0822"
        / "session01_roi_freeze"
        / "auto_roi_v2_parameters.json"
    )
    roi_status = Counter(row.get("roi_v2_status") for row in summaries)
    raw_valid = sum(int(row.get("h_raw_valid_frame_count") or 0) for row in summaries)
    local_valid = sum(
        int(row.get("local_measurement_valid_frame_count") or 0)
        for row in summaries
    )
    reliable_valid = sum(
        int(row.get("h_raw_reliable_frame_count") or 0) for row in summaries
    )
    eligible_conditions = sum(bool(row.get("h_raw_eligible")) for row in summaries)
    a3_risk_conditions = sum(
        bool(row.get("a3_right_morphology_region_overlap")) for row in summaries
    )
    return {
        "schema_version": 1,
        "task": "H0-1",
        "purpose": "Generate Haikang C0 raw heights from original PNGs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "root": str(root.resolve()),
            "session_ground_file": str(
                (root / "session_ground_calibration.json").resolve()
            ),
            "height_truth_source": "directory name only",
            "target_height_mm_by_id": TARGET_HEIGHT_MM,
            "height_shadow_used_as_input": False,
            "height_shadow_read": False,
            "frames_csv_schema": FRAME_FIELDS,
            "condition_count": len(conditions),
            "source_frame_count": sum(int(row["frame_count"]) for row in summaries),
            "discovery": discovery,
        },
        "reused_implementations": {
            "runtime_pipeline": {
                "path": str((TOOL_ROOT / "online" / "pipeline.py").resolve()),
                "sha256": sha256_file(TOOL_ROOT / "online" / "pipeline.py"),
                "entry": "FramePipeline.run_frame",
                "semantic_chain": [
                    "extract_laser_center (Steger)",
                    "reconstruct_uv_to_ground (Haikang circular-cone C0)",
                    "SessionGroundReference.apply_to_points",
                ],
            },
            "roi_v2": {
                "detector_path": str(
                    (REPO_ROOT / "tools" / "auto_roi_v2_session01.py").resolve()
                ),
                "detector_sha256": sha256_file(
                    REPO_ROOT / "tools" / "auto_roi_v2_session01.py"
                ),
                "centerline_wrapper_path": str(
                    (REPO_ROOT / "tools" / "thermal_a2a_roi_v2.py").resolve()
                ),
                "centerline_wrapper_sha256": sha256_file(
                    REPO_ROOT / "tools" / "thermal_a2a_roi_v2.py"
                ),
                "functions": [
                    "thermal_a2a_roi_v2.median_centerline",
                    "integer_profile",
                    "build_edge_pairs",
                    "assess_condition",
                    "support_stats",
                ],
                "parameters": roi_v2.PARAMETERS,
                "adapter": (
                    "Haikang column scan is mapped to the ROI-V2 row-scan "
                    "contract by swapping full-sensor (u,v) to (u'=v,v'=u); "
                    "measurement masks use original u."
                ),
                "height_or_3d_inputs_used": False,
            },
            "height_measurement": {
                "path": str((TOOL_ROOT / "measurement" / "height_measure.py").resolve()),
                "sha256": sha256_file(TOOL_ROOT / "measurement" / "height_measure.py"),
                "function": "measure_height_line",
                "session_mode": "session_reference",
                "local_mode": "auto",
                "final_h_raw": "local_measurement.height_mean_mm",
            },
            "daheng_reference_only": {
                "roi_v2_parameters_path": str(daheng_parameters.resolve()),
                "roi_v2_parameters_sha256": sha256_file(daheng_parameters),
                "used_numeric_daheng_roi": False,
                "used_daheng_calibration": False,
                "used_daheng_ground": False,
            },
        },
        "haikang_inputs": {
            "config": config_summary,
            "session_ground": session_summary,
            "prior_a3_spatial_audit": a3_audit,
            "calibration_manifest": (
                {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                }
                if manifest_path is not None
                else None
            ),
        },
        "h_raw_definition": {
            "value": "local_measurement.height_mean_mm",
            "formula_source": "existing measure_height_line",
            "input_points": "Session-Ground leveled points selected by Auto ROI V2",
            "local_reference": (
                "measure_height_line(..., ground_correction_mode='auto') "
                "fits baseline_roi_profile and subtracts predicted local ground"
            ),
            "session_reference_measurement_also_saved": True,
            "fallback_to_session_measurement": False,
            "raw_value_is_preserved_when_ineligible": True,
            "h_raw_eligible_rule": (
                "finite local value AND roi_v2_status=PASS AND no overlap with "
                "the prior A-3 right morphology region"
            ),
        },
        "execution_guards": {
            "production_config_modified": False,
            "c0_modified": False,
            "c1_applied": False,
            "h1_applied": False,
            "hb2_applied": False,
            "new_height_or_residual_fit": False,
            "height_truth_used_for_roi": False,
            "same_frame_reused_as_independent_position": False,
        },
        "result_inventory": {
            "roi_v2_status_counts": dict(roi_status),
            "roi_v2_strict_pass_count": sum(
                row.get("roi_v2_status") == "PASS" for row in summaries
            ),
            "roi_v2_candidate_count": sum(
                row.get("roi_v2_status") in {"PASS", "UNCERTAIN"}
                for row in summaries
            ),
            "local_measurement_valid_frame_count": local_valid,
            "h_raw_finite_frame_count": raw_valid,
            "h_raw_reliable_frame_count": reliable_valid,
            "h_raw_finite_position_count": sum(
                (row.get("h_raw_valid_frame_count") or 0) > 0
                for row in summaries
            ),
            "h_raw_eligible_condition_count": eligible_conditions,
            "a3_right_risk_condition_count": a3_risk_conditions,
            "registry_rows": len(registry_rows),
        },
        "outputs": {
            "directory": str((root / "c0_height_audit" / "measurement").resolve()),
            "h_raw_frames": "h_raw_frames.csv",
            "h_raw_position_summary": "h_raw_position_summary.csv",
            "roi_v2_registry": "roi_v2_registry.csv",
            "measurement_provenance": "measurement_provenance.json",
            "report": "h_raw_reconstruction_report.md",
        },
    }


def build_report(
    *,
    root: Path,
    config_summary: dict[str, Any],
    session_summary: dict[str, Any],
    discovery: dict[str, Any],
    summaries: list[dict[str, Any]],
    frame_count: int,
    a3_audit: dict[str, Any],
) -> str:
    roi_pass = [row for row in summaries if row.get("roi_v2_status") == "PASS"]
    roi_candidate = [
        row for row in summaries if row.get("roi_v2_status") in {"PASS", "UNCERTAIN"}
    ]
    raw_valid = [row for row in summaries if (row.get("h_raw_valid_frame_count") or 0) > 0]
    raw_eligible = [row for row in summaries if bool(row.get("h_raw_eligible"))]
    a3_risk = [
        row for row in summaries if bool(row.get("a3_right_morphology_region_overlap"))
    ]
    unreliable = [
        row["condition_id"]
        for row in summaries
        if not bool(row.get("h_raw_eligible"))
        or int(row.get("h_raw_valid_frame_count") or 0) != int(row.get("frame_count") or 0)
    ]
    generations = sorted(
        {
            generation
            for row in summaries
            for generation in (row.get("session_ground_generations") or [])
        }
    )
    ground_statuses = Counter()
    for row in summaries:
        ground_statuses.update(row.get("session_ground_status_counts") or {})

    lines = [
        "# H0-1 | Haikang C0 h_raw reconstruction report",
        "",
        f"Data root: {root.resolve()}",
        f"Condition count: {len(summaries)}; frame records: {frame_count}",
        f"Ground truth source: directory names only; mapping: {json.dumps(TARGET_HEIGHT_MM, ensure_ascii=False)}",
        "",
        "## 1. Actual reused call chain",
        "",
        "PNG + frames.csv -> FramePipeline.run_frame() -> Steger -> "
        "Haikang circular-cone C0 -> Session Ground -> Auto ROI V2 -> "
        "measure_height_line(session_reference) + measure_height_line(auto) -> "
        "local_measurement.height_mean_mm as h_raw",
        "",
        "The requested h_raw_mm is exactly the existing local measurement "
        "height_mean_mm.  The Session-reference measurement is also retained "
        "per frame as session_height_mean_mm.  Local mode receives the already "
        "Session-Ground-leveled points and does not apply a second Session Ground.",
        "",
        "## 2. Reused implementation and adapter",
        "",
        "1. online/pipeline.py: FramePipeline.run_frame is the only frame "
        "entry for Steger, C0 reconstruction and Session Ground application.",
        "2. measurement/height_measure.py: measure_height_line is called "
        "directly in session_reference and auto modes; its height formula "
        "was not copied.",
        "3. tools/thermal_a2a_roi_v2.py: median_centerline; tools/auto_roi_v2_session01.py: "
        "integer_profile, build_edge_pairs, assess_condition and support_stats "
        "are called directly.",
        "4. The validation reference pattern is the existing "
        "tools/validate_session01_a13b_v2_multireference.py flow: reconstruct once, "
        "apply Ground, mask ROI and call the existing measurement modes.",
        "5. Haikang is a column scan while the ROI-V2 detector contract is "
        "row-scan.  The only adapter is coordinate swap (u,v) to (u'=v,v'=u) "
        "for ROI detection; selected intervals are mapped back to original "
        "full-sensor u for point masks.",
        "6. Daheng ROI numbers, calibration, Ground and old height values were "
        "not reused.  The Daheng parameters file is provenance only.",
        "",
        "## 3. ROI-V2 coverage",
        "",
        f"Automatically discovered positions: {len(discovery.get('discovered_position_ids') or [])}; "
        f"IDs: {', '.join(discovery.get('discovered_position_ids') or [])}",
        f"Auto ROI candidate ranges generated: {len(roi_candidate)}/{len(summaries)}",
        f"Strict Auto ROI-V2 PASS: {len(roi_pass)}/{len(summaries)}",
        f"At least one valid h_raw frame: {len(raw_valid)}/{len(summaries)}",
        f"h_raw eligible after audit gates: {len(raw_eligible)}/{len(summaries)}",
        "No fixed p01-p10 ROI was supplied by hand.  Per-condition ranges, "
        "support, candidate status and point counts are in roi_v2_registry.csv.",
        f"Positions/conditions not fully reliable: {', '.join(unreliable) if unreliable else 'none'}",
        "The existing ROI-V2 selector's first candidate is retained.  Non-selected "
        "candidates were not chosen by comparing h_raw with directory truth.",
        "",
        "## 4. Session Ground",
        "",
        f"Reference status={session_summary.get('status')}, source={session_summary.get('source')}, "
        f"support={session_summary.get('support_source')}",
        f"Applied Ground generations observed: {generations}",
        f"Per-frame Ground status counts: {dict(ground_statuses)}",
        f"Reference rmse={session_summary.get('rmse_mm')} mm; "
        f"valid S range={session_summary.get('valid_s_range_mm')} mm",
        "All selected conditions use the 0829 root Session Ground reference. "
        "The separate h50 reference is not used.",
        "",
        "If partial_out_of_valid_s_domain appears, that is the existing "
        "SessionGroundReference behavior: out-of-domain points are not silently "
        "extrapolated and the count is retained in the frame CSV.",
        "",
        "## 5. local measurement and h_raw",
        "",
        "Session branch: measure_height_line with session_reference, saved for "
        "comparison and state tracing.",
        "Local branch: measure_height_line with auto, fitting the existing "
        "baseline_roi_profile from both ROI-V2 baseline intervals.  Its "
        "height_mean_mm is the only value exported as h_raw_mm.",
        "height_shadow.csv was not read and cannot influence ROI or h_raw.",
        "C1, H1 and H-B2 were not called or applied.  No calibration or "
        "production configuration was changed.",
        "",
        "## 6. A-3 spatial risk gate",
        "",
        f"Prior A-3 artifact: {a3_audit.get('path')}",
        f"A-3 classification: {a3_audit.get('classification')}; "
        f"derived right morphology range (inclusive full u): "
        f"{a3_audit.get('right_u_full_range_px')}",
        f"Selected ROI overlaps that right-side region in {len(a3_risk)}/{len(summaries)} "
        "conditions.  This is a diagnostic upstream-risk flag only; it does not "
        "change ROI selection or h_raw.",
        "A finite h_raw_mm is retained as the raw local-measurement diagnostic. "
        "Downstream C0 accuracy calculations must filter h_raw_eligible=true; "
        "the other rows are not cleared for accuracy claims.",
        "",
        "## 7. Per-condition repeat summary",
        "",
        "| condition | ROI | A-3 risk | eligible | h_raw valid | median mm | temporal std mm | local valid |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        median = "" if row.get("h_raw_mm_median") is None else f"{row['h_raw_mm_median']:.6f}"
        temporal = (
            "" if row.get("h_raw_temporal_std_mm") is None
            else f"{row['h_raw_temporal_std_mm']:.6f}"
        )
        lines.append(
            f"| {row['condition_id']} | {row.get('roi_v2_status')} | "
            f"{'yes' if row.get('a3_right_morphology_region_overlap') else 'no'} | "
            f"{'yes' if row.get('h_raw_eligible') else 'no'} | "
            f"{row.get('h_raw_valid_frame_count')}/{row.get('frame_count')} | "
            f"{median} | {temporal} | {row.get('local_measurement_valid_frame_count')} |"
        )
    lines += [
        "",
        "## 8. Reproducibility",
        "",
        f"Haikang config: {config_summary.get('path')}",
        f"Haikang config SHA-256: {config_summary.get('sha256')}",
        f"Session Ground file: {session_summary.get('path')}",
        f"Session Ground SHA-256: {session_summary.get('sha256')}",
        "The generated files are confined to c0_height_audit/measurement/.",
        "Twenty frames per condition are repeated measurements, not independent "
        "position samples.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.input_dir.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions, discovery = discover_conditions(root)
    config_summary = config_contract(config_path)
    a3_audit = load_prior_a3_spatial_audit(root)
    reference, rotation, translation, session_summary = load_session_reference(
        root / "session_ground_calibration.json"
    )
    app, pipeline = make_pipeline(config_path, reference, rotation, translation)

    summaries: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for condition in conditions:
        summary, frame_records, registry_row = run_condition(
            condition, pipeline, app, a3_audit
        )
        summaries.append(summary)
        frames.extend(frame_records)
        registry.append(registry_row)

    write_csv(output_dir / "h_raw_frames.csv", FRAME_CSV_FIELDS, frames)
    write_csv(output_dir / "h_raw_position_summary.csv", SUMMARY_CSV_FIELDS, summaries)
    write_csv(output_dir / "roi_v2_registry.csv", REGISTRY_CSV_FIELDS, registry)
    provenance = build_provenance(
        root=root,
        config_summary=config_summary,
        session_summary=session_summary,
        discovery=discovery,
        conditions=conditions,
        summaries=summaries,
        registry_rows=registry,
        a3_audit=a3_audit,
    )
    write_json(output_dir / "measurement_provenance.json", provenance)
    (output_dir / "h_raw_reconstruction_report.md").write_text(
        build_report(
            root=root,
            config_summary=config_summary,
            session_summary=session_summary,
            discovery=discovery,
            summaries=summaries,
            frame_count=len(frames),
            a3_audit=a3_audit,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "condition_count": len(summaries),
                "frame_count": len(frames),
                "roi_status_counts": dict(
                    Counter(row.get("roi_v2_status") for row in summaries)
                ),
                "h_raw_valid_frames": sum(
                    int(row.get("h_raw_valid_frame_count") or 0)
                    for row in summaries
                ),
                "h_raw_reliable_frames": sum(
                    int(row.get("h_raw_reliable_frame_count") or 0)
                    for row in summaries
                ),
                "h_raw_eligible_conditions": sum(
                    bool(row.get("h_raw_eligible")) for row in summaries
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
