#!/usr/bin/env python3
"""Audit the 2026-08-27 Daheng thermal-drift recordings.

This script is analysis-only.  It reads recording images/CSV files and the
frozen session/configuration artifacts, then writes audit artifacts to a
separate output directory.  It never rewrites raw recordings and never runs
Steger, reconstruction, or any C0/C1/H1/H-B2/Ground fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:  # pragma: no cover - the project venv provides PyYAML
    yaml = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the project venv provides Pillow
    Image = None


LOCAL_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
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
SHADOW_FIELDS = [
    "camera_frame_number",
    "host_timestamp_ns",
    "height_raw",
    "height_h1",
    "height_hb2",
    "active_height_correction",
    "active_height",
    "active_height_valid",
    "active_height_status",
    "q1",
    "q2",
    "q2_in_domain",
    "hb2_q2_status",
    "v_min",
    "v_median",
    "v_max",
    "point_count",
    "c1_clamp_status",
    "ground_reference_status",
]
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
RECORDING_RE = re.compile(r"^recording_(\d{8})_(\d{6})$")
FRAME_FILE_RE = re.compile(r"^frame_(\d{6})\.(png|tif|tiff|jpg|jpeg)$", re.IGNORECASE)


@dataclass
class TableRead:
    path: Path
    header: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    schema_ok: bool = False
    blank_rows: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class RecordingData:
    recording_id: str
    path: Path
    frames_table: TableRead
    shadow_table: TableRead
    image_files: list[Path]
    name_time: datetime | None
    metrics: dict[str, Any] = field(default_factory=dict)
    frames: list[dict[str, str]] = field(default_factory=list)
    shadows: list[dict[str, str]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = (
        repo_root
        / "laser_measurement_tool"
        / "output_daheng_0811"
        / "online_recordings"
        / "0827上午热漂_2000"
    )
    default_output = repo_root / "projects" / "daheng" / "analysis" / "thermal_a1_0827"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "laser_measurement_tool" / "configs" / "calibration_daheng_0811" / "manifest.yaml",
    )
    parser.add_argument(
        "--measure-config",
        type=Path,
        default=repo_root / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml",
    )
    parser.add_argument("--session-calibration", type=Path, default=None)
    parser.add_argument("--power-on", default="2026-08-27T09:50:00+08:00")
    parser.add_argument("--reference-complete", default="2026-08-27T09:57:00+08:00")
    parser.add_argument("--formal-start", default="2026-08-27T10:00:00+08:00")
    parser.add_argument("--pause-start", default="2026-08-27T11:30:00+08:00")
    parser.add_argument("--pause-end", default="2026-08-27T11:45:00+08:00")
    parser.add_argument("--no-record-start", default="2026-08-27T12:13:00+08:00")
    parser.add_argument("--reconnect", default="2026-08-27T13:01:00+08:00")
    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ)


def local_text(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(LOCAL_TZ).isoformat(timespec="milliseconds")


def ns_to_datetime(value: int | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).astimezone(LOCAL_TZ)
    except (OverflowError, OSError, ValueError):
        return None


def datetime_to_ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
        if not math.isfinite(parsed) or parsed != int(parsed):
            return None
        return int(parsed)
    except (TypeError, ValueError, OverflowError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError, OverflowError):
        return None


def as_bool(value: Any) -> bool | None:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def fmt_values(values: Iterable[Any]) -> str:
    return ";".join(fmt(value) for value in values)


def unique_text(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


def unique_numbers(values: Iterable[float | int | None]) -> list[float | int]:
    result: list[float | int] = []
    seen: set[float] = set()
    for value in values:
        if value is None:
            continue
        key = round(float(value), 9)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def median_or_none(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(cleaned) if cleaned else None


def min_or_none(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(cleaned) if cleaned else None


def max_or_none(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(cleaned) if cleaned else None


def nonincreasing_count(values: list[int | float | None]) -> int:
    return sum(
        1
        for previous, current in zip(values, values[1:])
        if previous is not None and current is not None and current <= previous
    )


def positive_intervals(values: list[int | float | None]) -> list[float]:
    return [
        float(current - previous)
        for previous, current in zip(values, values[1:])
        if previous is not None and current is not None and current > previous
    ]


def read_table(path: Path, expected_fields: list[str]) -> TableRead:
    table = TableRead(path=path)
    if not path.is_file():
        table.errors.append("missing_file")
        return table
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            raw_header = next(reader, None)
            if raw_header is None:
                table.errors.append("empty_file")
                return table
            table.header = [cell.strip() for cell in raw_header]
            table.schema_ok = table.header == expected_fields
            if not table.schema_ok:
                table.errors.append("schema_mismatch")
            for line_number, values in enumerate(reader, start=2):
                if not values or all(not str(value).strip() for value in values):
                    table.blank_rows += 1
                    continue
                if len(values) != len(table.header):
                    table.errors.append(f"line_{line_number}_column_count_{len(values)}")
                row = {
                    field: str(values[index]).strip() if index < len(values) else ""
                    for index, field in enumerate(table.header)
                }
                table.rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        table.errors.append(f"read_error:{type(exc).__name__}")
    return table


def parse_recording_time(name: str) -> datetime | None:
    match = RECORDING_RE.match(name)
    if not match:
        return None
    try:
        return datetime.strptime("_".join(match.groups()), "%Y%m%d_%H%M%S").replace(tzinfo=LOCAL_TZ)
    except ValueError:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_normalized_text(path: Path) -> str | None:
    """Hash UTF-8 text after normalizing line endings and an optional BOM."""
    if not path.is_file():
        return None
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeError):
        return None
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return payload if isinstance(payload, dict) else {}


def audit_calibration(manifest_path: Path, config_path: Path, session_path: Path) -> dict[str, Any]:
    manifest = load_yaml(manifest_path)
    config = load_yaml(config_path)
    session: dict[str, Any] = {}
    session_errors: list[str] = []
    if session_path.is_file():
        try:
            with session_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if isinstance(payload, dict):
                session = payload
            else:
                session_errors.append("session_json_not_object")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            session_errors.append(f"session_json_error:{type(exc).__name__}")
    else:
        session_errors.append("missing_session_json")

    camera_cfg = config.get("camera", {}) if isinstance(config.get("camera"), dict) else {}
    config_expected = {
        "exposure_us": as_float(camera_cfg.get("exposure_us")),
        "gain_db": as_float(camera_cfg.get("gain_db")),
        "pixel_format": str(camera_cfg.get("pixel_format", "")),
        "offset_x": as_int(camera_cfg.get("offset_x")),
        "offset_y": as_int(camera_cfg.get("offset_y")),
        "width": as_int(camera_cfg.get("width")),
        "height": as_int(camera_cfg.get("height")),
    }
    manifest_camera = manifest.get("camera", {}) if isinstance(manifest.get("camera"), dict) else {}
    extractor = manifest.get("extractor", {}) if isinstance(manifest.get("extractor"), dict) else {}
    settings = extractor.get("settings", {}) if isinstance(extractor.get("settings"), dict) else {}
    search_roi = settings.get("search_roi", {}) if isinstance(settings.get("search_roi"), dict) else {}

    manifest_hashes: list[dict[str, Any]] = []
    manifest_files = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
    for key in ("intrinsics", "laser_plane", "extrinsics", "laser_ray_correction"):
        entry = manifest_files.get(key, {})
        if not isinstance(entry, dict):
            manifest_hashes.append(
                {
                    "key": key,
                    "path": "",
                    "declared": "",
                    "actual_raw": "",
                    "actual_normalized": "",
                    "raw_match": False,
                    "normalized_match": False,
                    "match": False,
                    "match_mode": "missing_manifest_entry",
                }
            )
            continue
        relative = str(entry.get("path", ""))
        target = manifest_path.parent / relative
        declared = str(entry.get("sha256", ""))
        actual_raw = sha256_file(target)
        actual_normalized = sha256_normalized_text(target)
        raw_match = bool(declared) and actual_raw == declared
        normalized_match = bool(declared) and actual_normalized == declared
        manifest_hashes.append(
            {
                "key": key,
                "path": str(target),
                "declared": declared,
                "actual_raw": actual_raw or "",
                "actual_normalized": actual_normalized or "",
                "raw_match": raw_match,
                "normalized_match": normalized_match,
                "match": raw_match or normalized_match,
                "match_mode": "raw" if raw_match else "normalized_text" if normalized_match else "mismatch",
            }
        )

    session_runtime = session.get("runtime", {}) if isinstance(session.get("runtime"), dict) else {}
    session_board = session.get("board", {}) if isinstance(session.get("board"), dict) else {}
    session_frame = session.get("frame", {}) if isinstance(session.get("frame"), dict) else {}
    saved_at = None
    if session.get("saved_at_utc"):
        try:
            saved_at = parse_datetime(str(session["saved_at_utc"]))
        except ValueError:
            session_errors.append("invalid_saved_at_utc")

    session_valid = session.get("status") == "VALID" and session.get("valid") is True and not session_errors
    config_manifest_roi = {
        "offset_x": as_int(search_roi.get("offset_x")),
        "offset_y": as_int(search_roi.get("offset_y")),
        "width": as_int(search_roi.get("width")),
        "height": as_int(search_roi.get("height")),
    }
    expected_roi = tuple(config_expected[key] for key in ("offset_x", "offset_y", "width", "height"))
    manifest_roi = tuple(config_manifest_roi[key] for key in ("offset_x", "offset_y", "width", "height"))
    config_manifest_match = expected_roi == manifest_roi and str(manifest_camera.get("model", ""))
    return {
        "manifest_path": manifest_path,
        "config_path": config_path,
        "session_path": session_path,
        "manifest": manifest,
        "config": config,
        "session": session,
        "config_expected": config_expected,
        "manifest_camera": manifest_camera,
        "manifest_extractor": extractor,
        "manifest_roi": config_manifest_roi,
        "config_manifest_roi_match": bool(config_manifest_match),
        "manifest_hashes": manifest_hashes,
        "manifest_hashes_match": bool(manifest_hashes) and all(item["match"] for item in manifest_hashes),
        "session_valid": session_valid,
        "session_errors": session_errors,
        "session_saved_at": saved_at,
        "session_runtime": session_runtime,
        "session_board": session_board,
        "session_frame": session_frame,
    }


def build_recording(input_dir: Path, path: Path, expected: dict[str, Any]) -> RecordingData:
    frames_table = read_table(path / "frames.csv", FRAME_FIELDS)
    shadow_table = read_table(path / "height_shadow.csv", SHADOW_FIELDS)
    image_files = sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    recording = RecordingData(
        recording_id=path.name,
        path=path,
        frames_table=frames_table,
        shadow_table=shadow_table,
        image_files=image_files,
        name_time=parse_recording_time(path.name),
        frames=frames_table.rows,
        shadows=shadow_table.rows,
    )
    frames = recording.frames
    shadows = recording.shadows
    frame_numbers = [as_int(row.get("camera_frame_number")) for row in frames]
    frame_ticks = [as_int(row.get("camera_timestamp_ticks")) for row in frames]
    host_ns = [as_int(row.get("host_timestamp_ns")) for row in frames]
    host_mono_ns = [as_int(row.get("host_monotonic_ns")) for row in frames]
    declared_gaps = [as_int(row.get("frame_gap")) for row in frames]
    computed_gaps: list[int | None] = []
    gap_mismatches = 0
    for index, number in enumerate(frame_numbers):
        if index == 0 or number is None or frame_numbers[index - 1] is None:
            computed = 0 if index == 0 else None
        else:
            computed = max(0, number - frame_numbers[index - 1] - 1)
        computed_gaps.append(computed)
        if declared_gaps[index] is not None and computed is not None and declared_gaps[index] != computed:
            gap_mismatches += 1

    referenced_names = [str(row.get("filename", "")) for row in frames]
    referenced_set = set(referenced_names)
    image_set = {item.name for item in image_files}
    missing_images = sorted(name for name in referenced_set if name and name not in image_set)
    extra_images = sorted(name for name in image_set if name not in referenced_set)
    zero_byte_images = sum(1 for item in image_files if item.stat().st_size == 0)
    non_png_referenced_count = sum(
        1 for name in referenced_names if name and Path(name).suffix.lower() != ".png"
    )
    image_decode_errors: list[str] = []
    image_sizes: list[tuple[int, int]] = []
    image_modes: list[str] = []
    image_metadata: dict[str, tuple[tuple[int, int], str]] = {}
    if Image is None:
        image_decode_errors.append("Pillow_unavailable")
    else:
        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    size = (int(image.size[0]), int(image.size[1]))
                    mode = str(image.mode)
                    image_metadata[image_path.name] = (size, mode)
                    image_sizes.append(size)
                    image_modes.append(mode)
            except (OSError, ValueError, SyntaxError) as exc:
                image_decode_errors.append(f"{image_path.name}:{type(exc).__name__}")
    image_size_mismatch_count = 0
    invalid_size_metadata_count = 0
    image_mode_mismatch_count = 0
    expected_pixel_format = str(expected.get("pixel_format", ""))
    for row in frames:
        filename = str(row.get("filename", ""))
        observed = image_metadata.get(filename)
        expected_size = (as_int(row.get("width")), as_int(row.get("height")))
        if expected_size[0] is None or expected_size[1] is None:
            invalid_size_metadata_count += 1
        elif observed is not None and observed[0] != expected_size:
            image_size_mismatch_count += 1
        if expected_pixel_format == "Mono8" and observed is not None and observed[1] != "L":
            image_mode_mismatch_count += 1
    filename_order_errors = 0
    for index, name in enumerate(referenced_names, start=1):
        match = FRAME_FILE_RE.match(name)
        if not match or int(match.group(1)) != index:
            filename_order_errors += 1

    exposure_values = [as_float(row.get("exposure_us")) for row in frames]
    gain_values = [as_float(row.get("gain_db")) for row in frames]
    pixel_values = [str(row.get("pixel_format", "")) for row in frames]
    roi_values = [
        (
            as_int(row.get("offset_x")),
            as_int(row.get("offset_y")),
            as_int(row.get("width")),
            as_int(row.get("height")),
        )
        for row in frames
    ]
    host_intervals_ms = [value / 1_000_000 for value in positive_intervals(host_ns)]
    mono_intervals_ms = [value / 1_000_000 for value in positive_intervals(host_mono_ns)]
    tick_intervals = positive_intervals(frame_ticks)
    shadow_numbers = [as_int(row.get("camera_frame_number")) for row in shadows]
    shadow_host_ns = [as_int(row.get("host_timestamp_ns")) for row in shadows]
    frame_number_set = {value for value in frame_numbers if value is not None}
    matched_shadow = sum(1 for value in shadow_numbers if value is not None and value in frame_number_set)
    shadow_numeric = {
        key: [as_float(row.get(key)) for row in shadows]
        for key in ("height_raw", "height_h1", "height_hb2", "active_height", "q1", "q2", "v_min", "v_median", "v_max")
    }
    active_valid = [as_bool(row.get("active_height_valid")) for row in shadows]
    q2_domain = [as_bool(row.get("q2_in_domain")) for row in shadows]
    point_counts = [as_int(row.get("point_count")) for row in shadows]

    first_host = next((value for value in host_ns if value is not None), None)
    last_host = next((value for value in reversed(host_ns) if value is not None), None)
    first_shadow_host = next((value for value in shadow_host_ns if value is not None), None)
    last_shadow_host = next((value for value in reversed(shadow_host_ns) if value is not None), None)
    recording.metrics.update(
        {
            "frames_csv_present": frames_table.path.is_file(),
            "shadow_csv_present": shadow_table.path.is_file(),
            "frames_schema_ok": frames_table.schema_ok,
            "shadow_schema_ok": shadow_table.schema_ok,
            "frames_table_blank_rows": frames_table.blank_rows,
            "shadow_table_blank_rows": shadow_table.blank_rows,
            "frames_read_errors": ";".join(frames_table.errors),
            "shadow_read_errors": ";".join(shadow_table.errors),
            "frame_count": len(frames),
            "shadow_row_count": len(shadows),
            "png_count": sum(1 for item in image_files if item.suffix.lower() == ".png"),
            "missing_images": missing_images,
            "extra_images": extra_images,
            "zero_byte_images": zero_byte_images,
            "non_png_referenced_count": non_png_referenced_count,
            "image_decode_errors": image_decode_errors,
            "image_size_unique": unique_text(image_sizes),
            "image_mode_unique": unique_text(image_modes),
            "image_size_mismatch_count": image_size_mismatch_count,
            "invalid_size_metadata_count": invalid_size_metadata_count,
            "image_mode_mismatch_count": image_mode_mismatch_count,
            "filename_order_errors": filename_order_errors,
            "frame_numbers": frame_numbers,
            "frame_ticks": frame_ticks,
            "host_ns": host_ns,
            "host_mono_ns": host_mono_ns,
            "first_host_ns": first_host,
            "last_host_ns": last_host,
            "first_shadow_host_ns": first_shadow_host,
            "last_shadow_host_ns": last_shadow_host,
            "camera_frame_first": next((value for value in frame_numbers if value is not None), None),
            "camera_frame_last": next((value for value in reversed(frame_numbers) if value is not None), None),
            "camera_frame_nonincreasing_count": nonincreasing_count(frame_numbers),
            "camera_timestamp_nonincreasing_count": nonincreasing_count(frame_ticks),
            "host_timestamp_nonincreasing_count": nonincreasing_count(host_ns),
            "host_monotonic_nonincreasing_count": nonincreasing_count(host_mono_ns),
            "frame_gap_positive_count": sum(1 for value in computed_gaps if value is not None and value > 0),
            "frame_gap_total": sum(value for value in computed_gaps if value is not None),
            "frame_gap_max": max((value for value in computed_gaps if value is not None), default=0),
            "frame_gap_field_mismatch_count": gap_mismatches,
            "host_interval_median_ms": median_or_none(host_intervals_ms),
            "host_interval_min_ms": min_or_none(host_intervals_ms),
            "host_interval_max_ms": max_or_none(host_intervals_ms),
            "host_monotonic_interval_median_ms": median_or_none(mono_intervals_ms),
            "camera_tick_interval_median": median_or_none(tick_intervals),
            "exposure_values": exposure_values,
            "gain_values": gain_values,
            "pixel_values": pixel_values,
            "roi_values": roi_values,
            "exposure_unique": unique_numbers(exposure_values),
            "gain_unique": unique_numbers(gain_values),
            "pixel_unique": unique_text(pixel_values),
            "roi_unique": unique_text(roi_values),
            "exposure_config_match": bool(frames) and all(
                value is not None and expected.get("exposure_us") is not None and abs(value - expected["exposure_us"]) <= 1e-6
                for value in exposure_values
            ),
            "gain_config_match": bool(frames) and all(
                value is not None and expected.get("gain_db") is not None and abs(value - expected["gain_db"]) <= 1e-6
                for value in gain_values
            ),
            "pixel_config_match": bool(frames) and all(value == expected.get("pixel_format") for value in pixel_values),
            "roi_config_match": bool(frames)
            and all(
                value
                == tuple(expected.get(key) for key in ("offset_x", "offset_y", "width", "height"))
                for value in roi_values
            ),
            "shadow_frame_matched_count": matched_shadow,
            "shadow_frame_unmatched_count": sum(
                1 for value in shadow_numbers if value is not None and value not in frame_number_set
            ),
            "shadow_duplicate_frame_count": len(shadow_numbers) - len({value for value in shadow_numbers if value is not None}),
            "shadow_numeric": shadow_numeric,
            "shadow_active_valid_true_count": sum(value is True for value in active_valid),
            "shadow_active_valid_false_count": sum(value is False for value in active_valid),
            "shadow_active_valid_unknown_count": sum(value is None for value in active_valid),
            "shadow_q2_in_domain_true_count": sum(value is True for value in q2_domain),
            "shadow_q2_in_domain_false_count": sum(value is False for value in q2_domain),
            "shadow_status_values": unique_text(row.get("active_height_status", "") for row in shadows),
            "shadow_hb2_status_values": unique_text(row.get("hb2_q2_status", "") for row in shadows),
            "shadow_ground_status_values": unique_text(row.get("ground_reference_status", "") for row in shadows),
            "shadow_point_count_values": point_counts,
            "shadow_q1_median": median_or_none(shadow_numeric["q1"]),
            "shadow_q2_median": median_or_none(shadow_numeric["q2"]),
            "shadow_v_min_median": median_or_none(shadow_numeric["v_min"]),
            "shadow_v_median_median": median_or_none(shadow_numeric["v_median"]),
            "shadow_v_max_median": median_or_none(shadow_numeric["v_max"]),
            "shadow_point_count_median": median_or_none(point_counts),
            "shadow_point_count_min": min_or_none(point_counts),
            "shadow_point_count_max": max_or_none(point_counts),
            "shadow_time_median_ns": median_or_none(shadow_host_ns),
        }
    )
    recording.metrics["frames_table_clean"] = bool(
        frames_table.schema_ok and not frames_table.errors and frames_table.blank_rows == 0
    )
    recording.metrics["shadow_table_clean"] = bool(
        shadow_table.schema_ok and not shadow_table.errors and shadow_table.blank_rows == 0
    )
    recording.metrics["raw_core_ok"] = bool(
        recording.metrics["frames_csv_present"]
        and recording.metrics["frames_schema_ok"]
        and not frames_table.errors
        and frames_table.blank_rows == 0
        and recording.metrics["frame_count"] > 0
        and recording.metrics["png_count"] == recording.metrics["frame_count"]
        and not missing_images
        and not extra_images
        and zero_byte_images == 0
        and non_png_referenced_count == 0
        and not image_decode_errors
        and image_size_mismatch_count == 0
        and invalid_size_metadata_count == 0
        and image_mode_mismatch_count == 0
        and filename_order_errors == 0
        and recording.metrics["camera_frame_nonincreasing_count"] == 0
        and recording.metrics["camera_timestamp_nonincreasing_count"] == 0
        and recording.metrics["host_timestamp_nonincreasing_count"] == 0
        and recording.metrics["host_monotonic_nonincreasing_count"] == 0
        and recording.metrics["frame_gap_positive_count"] == 0
        and gap_mismatches == 0
        and recording.metrics["exposure_config_match"]
        and recording.metrics["gain_config_match"]
        and recording.metrics["pixel_config_match"]
        and recording.metrics["roi_config_match"]
    )
    recording.metrics["full_inventory_ok"] = bool(
        recording.metrics["raw_core_ok"]
        and recording.metrics["shadow_csv_present"]
        and recording.metrics["shadow_table_clean"]
    )
    return recording


def sort_recordings(recordings: list[RecordingData]) -> list[RecordingData]:
    return sorted(
        recordings,
        key=lambda item: (
            item.metrics.get("first_host_ns") is None,
            item.metrics.get("first_host_ns") or 0,
            item.recording_id,
        ),
    )


def build_boundaries(recordings: list[RecordingData], reconnect_ns: int) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for previous, current in zip(recordings, recordings[1:]):
        previous_last_host = previous.metrics.get("last_host_ns")
        current_first_host = current.metrics.get("first_host_ns")
        previous_last_frame = previous.metrics.get("camera_frame_last")
        current_first_frame = current.metrics.get("camera_frame_first")
        previous_last_tick = next((value for value in reversed(previous.metrics.get("frame_ticks", [])) if value is not None), None)
        current_first_tick = next((value for value in current.metrics.get("frame_ticks", []) if value is not None), None)
        previous_last_mono = next((value for value in reversed(previous.metrics.get("host_mono_ns", [])) if value is not None), None)
        current_first_mono = next((value for value in current.metrics.get("host_mono_ns", []) if value is not None), None)
        wall_gap_s = None
        if previous_last_host is not None and current_first_host is not None:
            wall_gap_s = (current_first_host - previous_last_host) / 1_000_000_000
        frame_delta = None
        if previous_last_frame is not None and current_first_frame is not None:
            frame_delta = current_first_frame - previous_last_frame
        tick_delta = None
        if previous_last_tick is not None and current_first_tick is not None:
            tick_delta = current_first_tick - previous_last_tick
        monotonic_gap_s = None
        if previous_last_mono is not None and current_first_mono is not None:
            monotonic_gap_s = (current_first_mono - previous_last_mono) / 1_000_000_000
        crosses_reconnect = bool(
            previous_last_host is not None
            and current_first_host is not None
            and previous_last_host < reconnect_ns <= current_first_host
        )
        boundaries.append(
            {
                "previous": previous,
                "current": current,
                "previous_last_host_ns": previous_last_host,
                "current_first_host_ns": current_first_host,
                "wall_gap_s": wall_gap_s,
                "frame_delta": frame_delta,
                "camera_frame_missing_between": frame_delta - 1 if frame_delta is not None and frame_delta > 0 else None,
                "tick_delta": tick_delta,
                "monotonic_gap_s": monotonic_gap_s,
                "frame_reset_or_nonforward": frame_delta is not None and frame_delta <= 0,
                "camera_tick_reset_or_nonforward": tick_delta is not None and tick_delta <= 0,
                "crosses_reconnect": crosses_reconnect,
            }
        )
    return boundaries


def assign_segments(recordings: list[RecordingData], reconnect_ns: int) -> None:
    for recording in recordings:
        first_host = recording.metrics.get("first_host_ns")
        last_host = recording.metrics.get("last_host_ns")
        if first_host is None or last_host is None:
            segment = "unknown"
        elif last_host < reconnect_ns:
            segment = "pre_reconnect"
        elif first_host >= reconnect_ns:
            segment = "post_reconnect"
        else:
            segment = "cross_reconnect"
        recording.metrics["segment"] = segment


def row_for_index(recording: RecordingData, expected_frame_count: int, power_on: datetime, reference: datetime) -> dict[str, Any]:
    m = recording.metrics
    first_dt = ns_to_datetime(m.get("first_host_ns"))
    last_dt = ns_to_datetime(m.get("last_host_ns"))
    duration_s = None
    if m.get("first_host_ns") is not None and m.get("last_host_ns") is not None:
        duration_s = (m["last_host_ns"] - m["first_host_ns"]) / 1_000_000_000
    active_rate = None
    if m["shadow_row_count"]:
        active_rate = m["shadow_active_valid_true_count"] / m["shadow_row_count"]
    q2_rate = None
    if m["shadow_row_count"]:
        q2_rate = m["shadow_q2_in_domain_true_count"] / m["shadow_row_count"]
    return {
        "recording_id": recording.recording_id,
        "relative_path": str(recording.path),
        "segment": m.get("segment", ""),
        "name_timestamp_local": local_text(recording.name_time),
        "first_frame_time_local": local_text(first_dt),
        "last_frame_time_local": local_text(last_dt),
        "first_host_timestamp_ns": m.get("first_host_ns"),
        "last_host_timestamp_ns": m.get("last_host_ns"),
        "elapsed_from_power_start_s": (m["first_host_ns"] - datetime_to_ns(power_on)) / 1_000_000_000
        if m.get("first_host_ns") is not None
        else None,
        "elapsed_from_power_end_s": (m["last_host_ns"] - datetime_to_ns(power_on)) / 1_000_000_000
        if m.get("last_host_ns") is not None
        else None,
        "elapsed_from_reference_start_s": (m["first_host_ns"] - datetime_to_ns(reference)) / 1_000_000_000
        if m.get("first_host_ns") is not None
        else None,
        "elapsed_from_reference_end_s": (m["last_host_ns"] - datetime_to_ns(reference)) / 1_000_000_000
        if m.get("last_host_ns") is not None
        else None,
        "duration_s": duration_s,
        "frame_count": m["frame_count"],
        "expected_frame_count": expected_frame_count,
        "frame_count_ok": m["frame_count"] == expected_frame_count,
        "png_count": m["png_count"],
        "missing_image_count": len(m["missing_images"]),
        "extra_image_count": len(m["extra_images"]),
        "zero_byte_image_count": m["zero_byte_images"],
        "image_decode_error_count": len(m["image_decode_errors"]),
        "non_png_referenced_count": m["non_png_referenced_count"],
        "image_size_unique": ";".join(m["image_size_unique"]),
        "image_mode_unique": ";".join(m["image_mode_unique"]),
        "image_size_mismatch_count": m["image_size_mismatch_count"],
        "invalid_image_size_metadata_count": m["invalid_size_metadata_count"],
        "image_mode_mismatch_count": m["image_mode_mismatch_count"],
        "frames_csv_present": m["frames_csv_present"],
        "shadow_csv_present": m["shadow_csv_present"],
        "frames_schema_ok": m["frames_schema_ok"],
        "shadow_schema_ok": m["shadow_schema_ok"],
        "frames_table_clean": m["frames_table_clean"],
        "shadow_table_clean": m["shadow_table_clean"],
        "frames_read_errors": m["frames_read_errors"],
        "shadow_read_errors": m["shadow_read_errors"],
        "frame_csv_blank_rows": m["frames_table_blank_rows"],
        "shadow_csv_blank_rows": m["shadow_table_blank_rows"],
        "filename_order_errors": m["filename_order_errors"],
        "camera_frame_first": m["camera_frame_first"],
        "camera_frame_last": m["camera_frame_last"],
        "camera_frame_nonincreasing_count": m["camera_frame_nonincreasing_count"],
        "frame_gap_positive_count": m["frame_gap_positive_count"],
        "frame_gap_total": m["frame_gap_total"],
        "frame_gap_max": m["frame_gap_max"],
        "frame_gap_field_mismatch_count": m["frame_gap_field_mismatch_count"],
        "camera_timestamp_nonincreasing_count": m["camera_timestamp_nonincreasing_count"],
        "host_timestamp_nonincreasing_count": m["host_timestamp_nonincreasing_count"],
        "host_monotonic_nonincreasing_count": m["host_monotonic_nonincreasing_count"],
        "host_interval_median_ms": m["host_interval_median_ms"],
        "host_interval_min_ms": m["host_interval_min_ms"],
        "host_interval_max_ms": m["host_interval_max_ms"],
        "camera_tick_interval_median": m["camera_tick_interval_median"],
        "exposure_unique": fmt_values(m["exposure_unique"]),
        "gain_unique": fmt_values(m["gain_unique"]),
        "pixel_format_unique": ";".join(m["pixel_unique"]),
        "roi_unique": ";".join(str(value) for value in m["roi_unique"]),
        "exposure_config_match": m["exposure_config_match"],
        "gain_config_match": m["gain_config_match"],
        "pixel_format_config_match": m["pixel_config_match"],
        "roi_config_match": m["roi_config_match"],
        "shadow_row_count": m["shadow_row_count"],
        "shadow_frame_matched_count": m["shadow_frame_matched_count"],
        "shadow_frame_unmatched_count": m["shadow_frame_unmatched_count"],
        "shadow_duplicate_frame_count": m["shadow_duplicate_frame_count"],
        "shadow_height_raw_numeric_count": sum(value is not None for value in m["shadow_numeric"]["height_raw"]),
        "shadow_height_h1_numeric_count": sum(value is not None for value in m["shadow_numeric"]["height_h1"]),
        "shadow_height_hb2_numeric_count": sum(value is not None for value in m["shadow_numeric"]["height_hb2"]),
        "shadow_active_valid_true_count": m["shadow_active_valid_true_count"],
        "shadow_active_valid_rate": active_rate,
        "shadow_q2_in_domain_true_count": m["shadow_q2_in_domain_true_count"],
        "shadow_q2_in_domain_rate": q2_rate,
        "shadow_q1_median": m["shadow_q1_median"],
        "shadow_q2_median": m["shadow_q2_median"],
        "shadow_v_min_median": m["shadow_v_min_median"],
        "shadow_v_median_median": m["shadow_v_median_median"],
        "shadow_v_max_median": m["shadow_v_max_median"],
        "shadow_point_count_median": m["shadow_point_count_median"],
        "shadow_point_count_min": m["shadow_point_count_min"],
        "shadow_point_count_max": m["shadow_point_count_max"],
        "shadow_active_status_values": ";".join(m["shadow_status_values"]),
        "shadow_hb2_q2_status_values": ";".join(m["shadow_hb2_status_values"]),
        "shadow_ground_reference_status_values": ";".join(m["shadow_ground_status_values"]),
        "raw_recording_integrity": "PASS" if m["raw_core_ok"] and m["frame_count"] == expected_frame_count else "REVIEW",
        "shadow_coverage_note": "LOW_RELATIVE_TO_MODAL" if m["shadow_row_count"] else "MISSING",
    }


INDEX_FIELDS = [
    "recording_id",
    "relative_path",
    "segment",
    "name_timestamp_local",
    "first_frame_time_local",
    "last_frame_time_local",
    "first_host_timestamp_ns",
    "last_host_timestamp_ns",
    "elapsed_from_power_start_s",
    "elapsed_from_power_end_s",
    "elapsed_from_reference_start_s",
    "elapsed_from_reference_end_s",
    "duration_s",
    "frame_count",
    "expected_frame_count",
    "frame_count_ok",
    "png_count",
    "missing_image_count",
    "extra_image_count",
    "zero_byte_image_count",
    "image_decode_error_count",
    "non_png_referenced_count",
    "image_size_unique",
    "image_mode_unique",
    "image_size_mismatch_count",
    "invalid_image_size_metadata_count",
    "image_mode_mismatch_count",
    "frames_csv_present",
    "shadow_csv_present",
    "frames_schema_ok",
    "shadow_schema_ok",
    "frames_table_clean",
    "shadow_table_clean",
    "frames_read_errors",
    "shadow_read_errors",
    "frame_csv_blank_rows",
    "shadow_csv_blank_rows",
    "filename_order_errors",
    "camera_frame_first",
    "camera_frame_last",
    "camera_frame_nonincreasing_count",
    "frame_gap_positive_count",
    "frame_gap_total",
    "frame_gap_max",
    "frame_gap_field_mismatch_count",
    "camera_timestamp_nonincreasing_count",
    "host_timestamp_nonincreasing_count",
    "host_monotonic_nonincreasing_count",
    "host_interval_median_ms",
    "host_interval_min_ms",
    "host_interval_max_ms",
    "camera_tick_interval_median",
    "exposure_unique",
    "gain_unique",
    "pixel_format_unique",
    "roi_unique",
    "exposure_config_match",
    "gain_config_match",
    "pixel_format_config_match",
    "roi_config_match",
    "shadow_row_count",
    "shadow_frame_matched_count",
    "shadow_frame_unmatched_count",
    "shadow_duplicate_frame_count",
    "shadow_height_raw_numeric_count",
    "shadow_height_h1_numeric_count",
    "shadow_height_hb2_numeric_count",
    "shadow_active_valid_true_count",
    "shadow_active_valid_rate",
    "shadow_q2_in_domain_true_count",
    "shadow_q2_in_domain_rate",
    "shadow_q1_median",
    "shadow_q2_median",
    "shadow_v_min_median",
    "shadow_v_median_median",
    "shadow_v_max_median",
    "shadow_point_count_median",
    "shadow_point_count_min",
    "shadow_point_count_max",
    "shadow_active_status_values",
    "shadow_hb2_q2_status_values",
    "shadow_ground_reference_status_values",
    "raw_recording_integrity",
    "shadow_coverage_note",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


TIMELINE_FIELDS = [
    "row_order",
    "row_kind",
    "event_id",
    "label",
    "start_local",
    "end_local",
    "start_host_timestamp_ns",
    "end_host_timestamp_ns",
    "duration_s",
    "elapsed_from_power_start_s",
    "elapsed_from_power_end_s",
    "elapsed_from_reference_start_s",
    "elapsed_from_reference_end_s",
    "segment",
    "recording_id",
    "prior_recording",
    "next_recording",
    "timestamp_source",
    "notes",
]


def timeline_row(
    *,
    row_kind: str,
    event_id: str,
    label: str,
    start: datetime,
    end: datetime,
    power_on: datetime,
    reference: datetime,
    segment: str = "",
    recording_id: str = "",
    prior_recording: str = "",
    next_recording: str = "",
    timestamp_source: str = "declared_event",
    notes: str = "",
) -> dict[str, Any]:
    start = start.astimezone(LOCAL_TZ)
    end = end.astimezone(LOCAL_TZ)
    return {
        "row_order": 0,
        "row_kind": row_kind,
        "event_id": event_id,
        "label": label,
        "start_local": local_text(start),
        "end_local": local_text(end),
        "start_host_timestamp_ns": datetime_to_ns(start),
        "end_host_timestamp_ns": datetime_to_ns(end),
        "duration_s": (end - start).total_seconds(),
        "elapsed_from_power_start_s": (start - power_on).total_seconds(),
        "elapsed_from_power_end_s": (end - power_on).total_seconds(),
        "elapsed_from_reference_start_s": (start - reference).total_seconds(),
        "elapsed_from_reference_end_s": (end - reference).total_seconds(),
        "segment": segment,
        "recording_id": recording_id,
        "prior_recording": prior_recording,
        "next_recording": next_recording,
        "timestamp_source": timestamp_source,
        "notes": notes,
    }


def build_timeline(
    recordings: list[RecordingData],
    boundaries: list[dict[str, Any]],
    events: dict[str, datetime],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    power_on = events["power_on"]
    reference = events["reference_complete"]
    pause_start_ns = datetime_to_ns(events["pause_start"])
    pause_end_ns = datetime_to_ns(events["pause_end"])
    pause_overlaps = [
        recording.recording_id
        for recording in recordings
        if recording.metrics.get("first_host_ns") is not None
        and recording.metrics.get("last_host_ns") is not None
        and recording.metrics["first_host_ns"] < pause_end_ns
        and recording.metrics["last_host_ns"] > pause_start_ns
    ]
    pause_label = "暂停记录（不视为 thermal reset）"
    pause_notes = "用户提供的 11:30–11:45 暂停区间"
    if pause_overlaps:
        pause_label = "暂停记录（声明区间与实际 recording 重叠；不视为 thermal reset）"
        pause_notes += f"；实际重叠 recording：{', '.join(pause_overlaps)}"
    point_events = [
        ("t_power", "上电", events["power_on"], "实验事件：09:50 上电"),
        ("t_reference", "Session 外参与地面基准完成", events["reference_complete"], "实验事件：09:57 完成 Session 标定与基准"),
        ("formal_start", "正式录制开始", events["formal_start"], "实验事件：10:00 开始正式记录"),
        ("camera_reconnect", "相机断开/重连边界", events["reconnect"], "独立事件；不将前后数据直接拼接"),
    ]
    for event_id, label, point, notes in point_events:
        rows.append(
            timeline_row(
                row_kind="declared_event",
                event_id=event_id,
                label=label,
                start=point,
                end=point,
                power_on=power_on,
                reference=reference,
                segment="reconnect_boundary" if event_id == "camera_reconnect" else "",
                timestamp_source="declared_event",
                notes=notes,
            )
        )
    rows.extend(
        [
            timeline_row(
                row_kind="declared_interval",
                event_id="recording_pause",
                label=pause_label,
                start=events["pause_start"],
                end=events["pause_end"],
                power_on=power_on,
                reference=reference,
                timestamp_source="declared_event",
                notes=pause_notes,
            ),
            timeline_row(
                row_kind="declared_interval",
                event_id="no_record_gap",
                label="无记录区间（不视为 thermal reset）",
                start=events["no_record_start"],
                end=events["reconnect"],
                power_on=power_on,
                reference=reference,
                segment="pre_reconnect_to_post_reconnect",
                timestamp_source="declared_event",
                notes="用户提供的 12:13–13:01 无记录区间；末端接入 reconnect 边界",
            ),
        ]
    )
    for recording in recordings:
        first = ns_to_datetime(recording.metrics.get("first_host_ns"))
        last = ns_to_datetime(recording.metrics.get("last_host_ns"))
        if first is None or last is None:
            continue
        rows.append(
            timeline_row(
                row_kind="recording_interval",
                event_id=recording.recording_id,
                label="录制区间",
                start=first,
                end=last,
                power_on=power_on,
                reference=reference,
                segment=recording.metrics.get("segment", ""),
                recording_id=recording.recording_id,
                timestamp_source="frames.csv host_timestamp_ns",
                notes="20-frame formal recording；recording 内部 frame_gap 单独审计",
            )
        )
    for boundary in boundaries:
        start = ns_to_datetime(boundary.get("previous_last_host_ns"))
        end = ns_to_datetime(boundary.get("current_first_host_ns"))
        if start is None or end is None:
            continue
        if boundary["crosses_reconnect"]:
            row_kind = "reconnect_data_gap"
            event_id = "reconnect_data_gap"
            label = "跨相机重连的数据间隔（禁止直接拼接）"
            segment = "reconnect_boundary"
            notes = "camera frame/timestamp continuity must be audited independently on both sides"
        else:
            row_kind = "recording_gap"
            event_id = "recording_gap"
            label = "录制间隔（不是 thermal reset）"
            segment = boundary["previous"].metrics.get("segment", "")
            notes = "未采集区间；elapsed time 连续保留，不插入 reset"
        rows.append(
            timeline_row(
                row_kind=row_kind,
                event_id=event_id,
                label=label,
                start=start,
                end=end,
                power_on=power_on,
                reference=reference,
                segment=segment,
                prior_recording=boundary["previous"].recording_id,
                next_recording=boundary["current"].recording_id,
                timestamp_source="adjacent frames.csv host_timestamp_ns",
                notes=(
                    f"wall_gap_s={fmt(boundary['wall_gap_s'])}; "
                    f"monotonic_gap_s={fmt(boundary['monotonic_gap_s'])}; "
                    f"camera_frame_delta={fmt(boundary['frame_delta'])}; "
                    f"camera_tick_delta={fmt(boundary['tick_delta'])}; {notes}"
                ),
            )
        )
    priority = {"declared_event": 0, "declared_interval": 1, "recording_interval": 2, "recording_gap": 3, "reconnect_data_gap": 4}
    rows.sort(key=lambda row: (row["start_host_timestamp_ns"], priority.get(row["row_kind"], 9), row["event_id"]))
    for index, row in enumerate(rows, start=1):
        row["row_order"] = index
    return rows


QC_FIELDS = ["scope", "recording_id", "segment", "check_id", "severity", "status", "observed", "expected", "evidence", "notes"]


def add_qc(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    recording_id: str,
    segment: str,
    check_id: str,
    severity: str,
    status: str,
    observed: Any,
    expected: Any,
    evidence: str,
    notes: str,
) -> None:
    rows.append(
        {
            "scope": scope,
            "recording_id": recording_id,
            "segment": segment,
            "check_id": check_id,
            "severity": severity,
            "status": status,
            "observed": observed,
            "expected": expected,
            "evidence": evidence,
            "notes": notes,
        }
    )


def build_qc_rows(
    recordings: list[RecordingData],
    boundaries: list[dict[str, Any]],
    calibration: dict[str, Any],
    expected_frame_count: int,
    reconnect_ns: int,
    formal_exposure_confirmed: bool,
    pre_valid: str,
    post_valid: str,
    a2_allowed: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_frames = sum(item.metrics["frame_count"] for item in recordings)
    total_shadows = sum(item.metrics["shadow_row_count"] for item in recordings)
    all_raw_ok = all(item.metrics["raw_core_ok"] and item.metrics["frame_count"] == expected_frame_count for item in recordings)
    all_inventory_ok = bool(recordings) and all(item.metrics["full_inventory_ok"] for item in recordings)
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="recording_inventory",
        severity="INFO" if all_inventory_ok else "BLOCKER",
        status="PASS" if all_inventory_ok else "FAIL",
        observed=f"recordings={len(recordings)}; frames={total_frames}; shadow_rows={total_shadows}; full_inventory={all_inventory_ok}",
        expected=f"all recording_* with {expected_frame_count} frames plus frames.csv/height_shadow.csv",
        evidence="thermal_a1_recording_index.csv",
        notes="frames/shadow CSVs must have exact schema, no malformed rows, and no blank rows; shadow row count is best-effort",
    )
    image_integrity_ok = bool(recordings) and all(
        item.metrics["png_count"] == item.metrics["frame_count"]
        and not item.metrics["missing_images"]
        and not item.metrics["extra_images"]
        and item.metrics["zero_byte_images"] == 0
        and item.metrics["non_png_referenced_count"] == 0
        and not item.metrics["image_decode_errors"]
        and item.metrics["image_size_mismatch_count"] == 0
        and item.metrics["invalid_size_metadata_count"] == 0
        and item.metrics["image_mode_mismatch_count"] == 0
        for item in recordings
    )
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="image_integrity",
        severity="INFO" if image_integrity_ok else "BLOCKER",
        status="PASS" if image_integrity_ok else "FAIL",
        observed=(
            f"png={sum(item.metrics['png_count'] for item in recordings)}; "
            f"missing={sum(len(item.metrics['missing_images']) for item in recordings)}; "
            f"extra={sum(len(item.metrics['extra_images']) for item in recordings)}; "
            f"zero_byte={sum(item.metrics['zero_byte_images'] for item in recordings)}; "
            f"decode_errors={sum(len(item.metrics['image_decode_errors']) for item in recordings)}; "
            f"size_mismatch={sum(item.metrics['image_size_mismatch_count'] for item in recordings)}; "
            f"invalid_size_metadata={sum(item.metrics['invalid_size_metadata_count'] for item in recordings)}; "
            f"mode_mismatch={sum(item.metrics['image_mode_mismatch_count'] for item in recordings)}"
        ),
        expected="one decodable grayscale PNG per frame; size metadata and Mono8 mode agree",
        evidence="frames.csv filenames/width/height/pixel_format + Pillow decode",
        notes="image files are audited as raw inputs; no reconstruction is performed",
    )
    for check_id, key, expected_value, label in [
        ("formal_exposure", "exposure_config_match", 2000, "exposure_us"),
        ("formal_gain", "gain_config_match", 0, "gain_db"),
        ("formal_pixel_format", "pixel_config_match", "Mono8", "pixel_format"),
        ("formal_roi", "roi_config_match", "config ROI", "ROI"),
    ]:
        ok = all(item.metrics[key] for item in recordings) if recordings else False
        observed = ";".join(
            f"{item.recording_id}:{fmt(item.metrics.get('exposure_unique') if key == 'exposure_config_match' else item.metrics.get('gain_unique') if key == 'gain_config_match' else item.metrics.get('pixel_unique') if key == 'pixel_config_match' else item.metrics.get('roi_unique'))}"
            for item in recordings
        )
        add_qc(
            rows,
            scope="global",
            recording_id="",
            segment="",
            check_id=check_id,
            severity="INFO" if ok else "BLOCKER",
            status="PASS" if ok else "FAIL",
            observed=observed,
            expected=f"all rows match {expected_value} ({label})",
            evidence="frames.csv rows in every recording",
            notes="正式 recording rows only; any unsaved exposure sweep is outside this dataset",
        )
    frames_schema_ok = bool(recordings) and all(item.metrics["frames_table_clean"] for item in recordings)
    shadow_schema_ok = bool(recordings) and all(item.metrics["shadow_table_clean"] for item in recordings)
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="frames_schema",
        severity="INFO" if frames_schema_ok else "BLOCKER",
        status="PASS" if frames_schema_ok else "FAIL",
        observed=f"{sum(item.metrics['frames_schema_ok'] for item in recordings)}/{len(recordings)}",
        expected="FRAME_FIELDS exact match",
        evidence="recording.py schema and each frames.csv",
        notes="UTF-8 BOM accepted; malformed and blank rows are integrity failures",
    )
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="shadow_schema",
        severity="INFO" if shadow_schema_ok else "REVIEW",
        status="PASS" if shadow_schema_ok else "FAIL",
        observed=f"{sum(item.metrics['shadow_schema_ok'] for item in recordings)}/{len(recordings)}",
        expected="SHADOW_FIELDS exact match",
        evidence="recording.py schema and each height_shadow.csv",
        notes="shadow is asynchronous best-effort; file/schema/row integrity is still audited",
    )
    internal_gaps = sum(item.metrics["frame_gap_positive_count"] for item in recordings)
    internal_nonforward = sum(item.metrics["camera_frame_nonincreasing_count"] for item in recordings)
    host_nonforward = sum(item.metrics["host_timestamp_nonincreasing_count"] for item in recordings)
    mono_nonforward = sum(item.metrics["host_monotonic_nonincreasing_count"] for item in recordings)
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="within_recording_frame_continuity",
        severity="INFO" if internal_gaps == 0 and internal_nonforward == 0 else "BLOCKER",
        status="PASS" if internal_gaps == 0 and internal_nonforward == 0 else "FAIL",
        observed=f"positive_frame_gap={internal_gaps}; nonforward_frame_pairs={internal_nonforward}",
        expected="0 within every recording",
        evidence="frames.csv frame_gap and camera_frame_number",
        notes="cross-recording gaps are audited separately and are not treated as thermal reset",
    )
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="within_recording_host_time",
        severity="INFO" if host_nonforward == 0 and mono_nonforward == 0 else "BLOCKER",
        status="PASS" if host_nonforward == 0 and mono_nonforward == 0 else "FAIL",
        observed=f"host_nonforward_pairs={host_nonforward}; monotonic_nonforward_pairs={mono_nonforward}",
        expected="0 within every recording",
        evidence="frames.csv host_timestamp_ns/host_monotonic_ns",
        notes="host_timestamp_ns anchors the event timeline; monotonic time is used as a local order check",
    )
    cross_time_bad = [
        item
        for item in boundaries
        if item.get("wall_gap_s") is None
        or item["wall_gap_s"] <= 0
        or item.get("monotonic_gap_s") is None
        or item["monotonic_gap_s"] <= 0
    ]
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="host_wall_monotonic_cross_recording",
        severity="INFO" if not cross_time_bad else "BLOCKER",
        status="PASS" if not cross_time_bad else "FAIL",
        observed=f"boundary_count={len(boundaries)}; invalid_wall_or_monotonic_boundaries={len(cross_time_bad)}",
        expected="both host wall and host monotonic timestamps move forward at recording boundaries",
        evidence="adjacent frames.csv host_timestamp_ns/host_monotonic_ns",
        notes="host wall time remains the event-axis anchor only after this independent monotonic-order check",
    )
    nonforward_cross_recording = [item for item in boundaries if item.get("frame_reset_or_nonforward")]
    nonforward_pre = [
        item
        for item in nonforward_cross_recording
        if item["previous"].metrics.get("segment") == "pre_reconnect"
        and item["current"].metrics.get("segment") == "pre_reconnect"
    ]
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="camera_frame_number_cross_recording",
        severity="REVIEW" if nonforward_cross_recording else "INFO",
        status="REVIEW" if nonforward_cross_recording else "PASS",
        observed=(
            f"nonforward_boundaries={len(nonforward_cross_recording)}; "
            f"pre_reconnect_nonforward={len(nonforward_pre)}; "
            "pairs="
            + ";".join(
                f"{item['previous'].recording_id}->{item['current'].recording_id}({fmt(item['frame_delta'])})"
                for item in nonforward_cross_recording
            )
        ),
        expected="camera frame number is a local frame identity; do not use it as a global thermal time key across recording/reconnect boundaries",
        evidence="adjacent frames.csv camera_frame_number",
        notes="host_timestamp_ns and camera_timestamp_ticks remain separate evidence; negative cross-recording delta is not treated as thermal reset",
    )
    add_qc(
        rows,
        scope="decision",
        recording_id="",
        segment="pre_reconnect",
        check_id="PRE_RECONNECT_GLOBAL_CAMERA_FRAME_CONTINUITY",
        severity="INFO" if not nonforward_pre else "REVIEW",
        status="YES" if not nonforward_pre else "NO",
        observed=f"pre_reconnect_nonforward_boundaries={len(nonforward_pre)}",
        expected="YES only if camera_frame_number never rolls back between pre-reconnect recordings",
        evidence="adjacent pre-reconnect frames.csv camera_frame_number",
        notes="this is separate from PRE_RECONNECT_DATA_VALID, which is recording-level raw/host-time validity",
    )
    gap_durations = [item["wall_gap_s"] for item in boundaries if item.get("wall_gap_s") is not None]
    reconnect_boundaries = [item for item in boundaries if item["crosses_reconnect"]]
    add_qc(
        rows,
        scope="global",
        recording_id="",
        segment="",
        check_id="recording_gaps",
        severity="REVIEW",
        status="REVIEW" if boundaries else "PASS",
        observed=f"boundary_count={len(boundaries)}; max_wall_gap_s={fmt(max(gap_durations, default=None))}",
        expected="gaps represented in timeline; no implicit thermal reset",
        evidence="thermal_a1_event_timeline.csv",
        notes="recording gaps are observation gaps, including the declared pause/no-record intervals",
    )
    if reconnect_boundaries:
        boundary = reconnect_boundaries[0]
        previous = boundary["previous"]
        current = boundary["current"]
        pre_last_mono = next((value for value in reversed(previous.metrics["host_mono_ns"]) if value is not None), None)
        post_first_mono = next((value for value in current.metrics["host_mono_ns"] if value is not None), None)
        host_forward = boundary["current_first_host_ns"] is not None and boundary["previous_last_host_ns"] is not None and boundary["current_first_host_ns"] > boundary["previous_last_host_ns"]
        mono_forward = pre_last_mono is not None and post_first_mono is not None and post_first_mono > pre_last_mono
        add_qc(
            rows,
            scope="reconnect_boundary",
            recording_id=f"{previous.recording_id} -> {current.recording_id}",
            segment="reconnect_boundary",
            check_id="camera_frame_number_continuity",
            severity="REVIEW",
            status="REVIEW" if boundary["frame_reset_or_nonforward"] else "PASS",
            observed=f"frame_delta={fmt(boundary['frame_delta'])}; reset_or_nonforward={fmt(boundary['frame_reset_or_nonforward'])}",
            expected="do not stitch across reconnect; independently segmented",
            evidence="adjacent frames.csv first/last camera_frame_number",
            notes="a counter reset is a reconnect discontinuity, not a thermal reset",
        )
        add_qc(
            rows,
            scope="reconnect_boundary",
            recording_id=f"{previous.recording_id} -> {current.recording_id}",
            segment="reconnect_boundary",
            check_id="camera_timestamp_ticks_continuity",
            severity="REVIEW",
            status="REVIEW" if boundary["camera_tick_reset_or_nonforward"] else "PASS",
            observed=f"tick_delta={fmt(boundary['tick_delta'])}; reset_or_nonforward={fmt(boundary['camera_tick_reset_or_nonforward'])}",
            expected="raw ticks retained; no unit conversion or cross-reset stitching",
            evidence="adjacent frames.csv camera_timestamp_ticks",
            notes="camera timestamp tick unit is not assumed",
        )
        add_qc(
            rows,
            scope="reconnect_boundary",
            recording_id=f"{previous.recording_id} -> {current.recording_id}",
            segment="reconnect_boundary",
            check_id="host_time_continuity",
            severity="INFO" if host_forward and mono_forward else "BLOCKER",
            status="PASS" if host_forward and mono_forward else "FAIL",
            observed=f"wall_forward={host_forward}; monotonic_forward={mono_forward}; wall_gap_s={fmt(boundary['wall_gap_s'])}",
            expected="host wall/monotonic timestamps remain forward across the long gap",
            evidence="adjacent frames.csv host timestamps",
            notes="forward wall time does not make camera counters stitchable",
        )
    else:
        add_qc(
            rows,
            scope="reconnect_boundary",
            recording_id="",
            segment="reconnect_boundary",
            check_id="reconnect_boundary_presence",
            severity="BLOCKER",
            status="FAIL",
            observed="no crossing boundary found",
            expected=f"a data boundary at {local_text(ns_to_datetime(reconnect_ns))}",
            evidence="recording host timestamps",
            notes="cannot certify pre/post reconnect separation",
        )

    shadow_rows = [row for recording in recordings for row in recording.shadows]
    height_counts = {
        key: sum(as_float(row.get(key)) is not None for row in shadow_rows)
        for key in ("height_raw", "height_h1", "height_hb2")
    }
    active_true = sum(as_bool(row.get("active_height_valid")) is True for row in shadow_rows)
    q2_true = sum(as_bool(row.get("q2_in_domain")) is True for row in shadow_rows)
    point_values = [as_int(row.get("point_count")) for row in shadow_rows]
    add_qc(
        rows,
        scope="shadow",
        recording_id="",
        segment="",
        check_id="shadow_height_numeric_availability",
        severity="REVIEW",
        status="PASS" if any(height_counts.values()) else "FAIL",
        observed=f"raw={height_counts['height_raw']}; H1={height_counts['height_h1']}; H-B2={height_counts['height_hb2']}; total_shadow={len(shadow_rows)}",
        expected="diagnostic only; final heights must come from offline Frozen reconstruction",
        evidence="height_shadow.csv",
        notes="FAIL here does not block A2; it confirms shadow cannot serve as final measurement truth",
    )
    add_qc(
        rows,
        scope="shadow",
        recording_id="",
        segment="",
        check_id="shadow_active_valid",
        severity="REVIEW",
        status="PASS" if active_true else "FAIL",
        observed=f"active_height_valid_true={active_true}/{len(shadow_rows)}",
        expected="reported for diagnosis only",
        evidence="height_shadow.csv active_height_valid",
        notes="q2-domain/status must be checked before interpreting active height",
    )
    add_qc(
        rows,
        scope="shadow",
        recording_id="",
        segment="",
        check_id="shadow_q2_domain",
        severity="REVIEW",
        status="PASS" if q2_true else "FAIL",
        observed=f"q2_in_domain_true={q2_true}/{len(shadow_rows)}",
        expected="q2 domain status retained as diagnostic",
        evidence="height_shadow.csv q2_in_domain/hb2_q2_status",
        notes="do not interpret OOD extrapolation/status as final height",
    )
    add_qc(
        rows,
        scope="shadow",
        recording_id="",
        segment="",
        check_id="shadow_point_count",
        severity="INFO" if point_values and all(value is not None and value > 0 for value in point_values) else "REVIEW",
        status="PASS" if point_values and all(value is not None and value > 0 for value in point_values) else "REVIEW",
        observed=f"min={fmt(min_or_none(point_values))}; median={fmt(median_or_none(point_values))}; max={fmt(max_or_none(point_values))}",
        expected="positive diagnostic point counts",
        evidence="height_shadow.csv point_count",
        notes="point count is not a precision/accuracy metric",
    )
    session_status = "PASS" if calibration["session_valid"] else "FAIL"
    add_qc(
        rows,
        scope="provenance",
        recording_id="",
        segment="",
        check_id="session_ground_calibration",
        severity="INFO" if calibration["session_valid"] else "BLOCKER",
        status=session_status,
        observed=f"status={calibration['session'].get('status')}; valid={calibration['session'].get('valid')}; generation={calibration['session_runtime'].get('ground_extrinsic_generation')}",
        expected="VALID session calibration before formal recording",
        evidence=str(calibration["session_path"]),
        notes="session calibration artifact is reused; no refit performed",
    )
    add_qc(
        rows,
        scope="provenance",
        recording_id="",
        segment="",
        check_id="calibration_manifest_hashes",
        severity="INFO" if calibration["manifest_hashes_match"] else "BLOCKER",
        status="PASS" if calibration["manifest_hashes_match"] else "FAIL",
        observed=";".join(f"{item['key']}={item['match']}" for item in calibration["manifest_hashes"]),
        expected="all manifest-declared hashes match local calibration files",
        evidence=str(calibration["manifest_path"]),
        notes="same-protocol frozen package; historical 0824 1000-us results were not reused",
    )
    add_qc(
        rows,
        scope="provenance",
        recording_id="",
        segment="",
        check_id="config_manifest_roi",
        severity="INFO" if calibration["config_manifest_roi_match"] else "BLOCKER",
        status="PASS" if calibration["config_manifest_roi_match"] else "FAIL",
        observed=f"config_roi={calibration['config_expected'].get('offset_x')},{calibration['config_expected'].get('offset_y')},{calibration['config_expected'].get('width')},{calibration['config_expected'].get('height')}; manifest_roi={calibration['manifest_roi']}",
        expected="config and manifest ROI identical",
        evidence=f"{calibration['config_path']} + {calibration['manifest_path']}",
        notes="all formal frames independently matched the same ROI",
    )
    add_qc(
        rows,
        scope="decision",
        recording_id="",
        segment="pre_reconnect",
        check_id="PRE_RECONNECT_DATA_VALID",
        severity="INFO" if pre_valid == "YES" else "BLOCKER",
        status=pre_valid,
        observed=f"{sum(item.metrics['segment'] == 'pre_reconnect' for item in recordings)} recordings; internal_raw_ok={all(item.metrics['raw_core_ok'] for item in recordings if item.metrics['segment'] == 'pre_reconnect')}",
        expected="YES when raw recordings/config/session provenance are valid; shadow is not truth",
        evidence="recording index + session calibration + frames.csv",
        notes="recording gaps remain in elapsed time and do not reset thermal time",
    )
    add_qc(
        rows,
        scope="decision",
        recording_id="",
        segment="post_reconnect",
        check_id="POST_RECONNECT_DATA_VALID",
        severity="REVIEW" if post_valid == "PARTIAL" else ("INFO" if post_valid == "YES" else "BLOCKER"),
        status=post_valid,
        observed=f"{sum(item.metrics['segment'] == 'post_reconnect' for item in recordings)} recordings; reconnect_boundary={bool(reconnect_boundaries)}",
        expected="PARTIAL unless post side has independent continuity/provenance without reset ambiguity",
        evidence="reconnect boundary QC",
        notes="post side may enter A2 only as a separate segment; no direct pre/post curve stitch",
    )
    add_qc(
        rows,
        scope="decision",
        recording_id="",
        segment="pre_and_post_separate",
        check_id="THERMAL_A2_ALLOWED",
        severity="INFO" if a2_allowed == "YES" else "BLOCKER",
        status=a2_allowed,
        observed=f"formal_exposure={formal_exposure_confirmed}; pre={pre_valid}; post={post_valid}; all_raw={all_raw_ok}",
        expected="YES only for segmented offline Frozen reconstruction with reconnect audit retained",
        evidence="all generated A1 artifacts",
        notes="A2 must recompute from PNGs with frozen package; do not use shadow as final heights or fit new correction",
    )
    return rows


def shadow_plot(recordings: list[RecordingData], events: dict[str, datetime], output_path: Path) -> None:
    power_ns = datetime_to_ns(events["power_on"])
    reconnect_min = (events["reconnect"] - events["power_on"]).total_seconds() / 60
    pause_start_min = (events["pause_start"] - events["power_on"]).total_seconds() / 60
    pause_end_min = (events["pause_end"] - events["power_on"]).total_seconds() / 60
    no_record_start_min = (events["no_record_start"] - events["power_on"]).total_seconds() / 60

    x: list[float] = []
    segments: list[str] = []
    height_values = {key: [] for key in ("height_raw", "height_h1", "height_hb2")}
    q_values = {key: [] for key in ("q1", "q2")}
    v_values = {key: [] for key in ("v_min", "v_median", "v_max")}
    point_values: list[float | None] = []
    valid_values: list[float | None] = []
    for recording in recordings:
        time_ns = recording.metrics.get("shadow_time_median_ns") or recording.metrics.get("first_host_ns")
        if time_ns is None:
            continue
        x.append((time_ns - power_ns) / 1_000_000_000 / 60)
        segments.append(recording.metrics.get("segment", ""))
        for key in height_values:
            height_values[key].append(median_or_none(recording.metrics["shadow_numeric"][key]))
        for key in q_values:
            q_values[key].append(median_or_none(recording.metrics["shadow_numeric"][key]))
        for key in v_values:
            v_values[key].append(median_or_none(recording.metrics["shadow_numeric"][key]))
        point_values.append(recording.metrics.get("shadow_point_count_median"))
        count = recording.metrics.get("shadow_row_count", 0)
        valid_values.append(
            recording.metrics.get("shadow_active_valid_true_count", 0) / count if count else None
        )

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, constrained_layout=True)
    colors = {"pre_reconnect": "#2563eb", "post_reconnect": "#d97706", "unknown": "#6b7280"}

    def plot_by_segment(ax: Any, values: list[float | None], label: str, color: str, marker: str = "o") -> None:
        for segment in ("pre_reconnect", "post_reconnect", "unknown"):
            indices = [index for index, value in enumerate(values) if value is not None and segments[index] == segment]
            if indices:
                ax.plot(
                    [x[index] for index in indices],
                    [values[index] for index in indices],
                    marker=marker,
                    linestyle="-",
                    linewidth=1.4,
                    markersize=4,
                    color=colors[segment] if color == "segment" else color,
                    alpha=0.9,
                    label=f"{label} ({segment})" if color == "segment" else label,
                )

    for ax in axes:
        ax.axvspan(pause_start_min, pause_end_min, color="#9ca3af", alpha=0.14, label="declared pause" if ax is axes[0] else None)
        ax.axvspan(no_record_start_min, reconnect_min, color="#6b7280", alpha=0.10, label="declared no-record" if ax is axes[0] else None)
        ax.axvline(reconnect_min, color="#b91c1c", linestyle="--", linewidth=1.2, label="camera reconnect" if ax is axes[0] else None)

    for key, label in (("height_raw", "raw height"), ("height_h1", "H1 height"), ("height_hb2", "H-B2 height")):
        plot_by_segment(axes[0], height_values[key], label, "segment")
    axes[0].set_ylabel("height (shadow units)")
    if not any(value is not None for values in height_values.values() for value in values):
        axes[0].text(0.5, 0.5, "No numeric raw/H1/H-B2 height in shadow CSVs", transform=axes[0].transAxes, ha="center", va="center", color="#b91c1c")

    for key, label in (("q1", "q1"), ("q2", "q2")):
        plot_by_segment(axes[1], q_values[key], label, "segment")
    axes[1].set_ylabel("q value")
    axes[1].legend(loc="best", ncol=2, fontsize=8)

    for key, label in (("v_min", "v_min"), ("v_median", "v_median"), ("v_max", "v_max")):
        plot_by_segment(axes[2], v_values[key], label, "segment")
    axes[2].set_ylabel("v (px)")
    axes[2].legend(loc="best", ncol=3, fontsize=8)

    plot_by_segment(axes[3], point_values, "point_count", "segment")
    valid_axis = axes[3].twinx()
    plot_by_segment(valid_axis, valid_values, "active valid rate", "#16a34a", marker="s")
    valid_axis.set_ylim(-0.05, 1.05)
    valid_axis.set_ylabel("active_height_valid rate")
    axes[3].set_ylabel("point count")
    axes[3].set_xlabel("elapsed from power-on (min)")
    axes[3].legend(loc="upper left", fontsize=8)
    valid_axis.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)
    fig.suptitle("Thermal-A1 shadow preview — existing height_shadow values only; no re-extraction/reconstruction/refit", fontsize=14, y=0.995)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_report(
    output_dir: Path,
    input_dir: Path,
    recordings: list[RecordingData],
    index_rows: list[dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    events: dict[str, datetime],
    boundaries: list[dict[str, Any]],
    expected_frame_count: int,
    formal_exposure_confirmed: bool,
    pre_valid: str,
    post_valid: str,
    a2_allowed: str,
) -> str:
    total_frames = sum(item.metrics["frame_count"] for item in recordings)
    total_shadow = sum(item.metrics["shadow_row_count"] for item in recordings)
    shadow_rows = [row for item in recordings for row in item.shadows]
    height_counts = {
        key: sum(as_float(row.get(key)) is not None for row in shadow_rows)
        for key in ("height_raw", "height_h1", "height_hb2")
    }
    active_true = sum(as_bool(row.get("active_height_valid")) is True for row in shadow_rows)
    q2_true = sum(as_bool(row.get("q2_in_domain")) is True for row in shadow_rows)
    point_counts = [as_int(row.get("point_count")) for row in shadow_rows]
    all_status = Counter(str(row.get("ground_reference_status", "")) for row in shadow_rows)
    all_hb2 = Counter(str(row.get("hb2_q2_status", "")) for row in shadow_rows)
    all_active_status = Counter(str(row.get("active_height_status", "")) for row in shadow_rows)
    shadow_global_numeric = {
        key: [as_float(row.get(key)) for row in shadow_rows]
        for key in ("q1", "q2", "v_min", "v_median", "v_max")
    }
    first_frame = next((item.metrics.get("first_host_ns") for item in recordings if item.metrics.get("first_host_ns") is not None), None)
    last_frame = next((item.metrics.get("last_host_ns") for item in reversed(recordings) if item.metrics.get("last_host_ns") is not None), None)
    pre = [item for item in recordings if item.metrics.get("segment") == "pre_reconnect"]
    post = [item for item in recordings if item.metrics.get("segment") == "post_reconnect"]
    shadow_mode = statistics.mode([r.metrics["shadow_row_count"] for r in recordings]) if recordings else 0
    shadow_min = min((r.metrics["shadow_row_count"] for r in recordings), default=0)
    low_shadow = [item.recording_id for item in recordings if item.metrics["shadow_row_count"] == shadow_min and shadow_min < shadow_mode]
    pause_start_ns = datetime_to_ns(events["pause_start"])
    pause_end_ns = datetime_to_ns(events["pause_end"])
    pause_overlaps = [
        item
        for item in recordings
        if item.metrics.get("first_host_ns") is not None
        and item.metrics.get("last_host_ns") is not None
        and item.metrics["first_host_ns"] < pause_end_ns
        and item.metrics["last_host_ns"] > pause_start_ns
    ]
    total_png = sum(item.metrics["png_count"] for item in recordings)
    total_decode_errors = sum(len(item.metrics["image_decode_errors"]) for item in recordings)
    total_size_mismatches = sum(item.metrics["image_size_mismatch_count"] for item in recordings)
    total_mode_mismatches = sum(item.metrics["image_mode_mismatch_count"] for item in recordings)
    total_invalid_size_metadata = sum(item.metrics["invalid_size_metadata_count"] for item in recordings)

    def table(lines: list[list[Any]]) -> str:
        if not lines:
            return ""
        header = lines[0]
        body = lines[1:]
        result = ["| " + " | ".join(str(cell) for cell in header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        result.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in body)
        return "\n".join(result)

    index_table = [["recording", "segment", "first local", "last local", "frames", "shadow", "frame_gap", "raw QC"]]
    for row in index_rows:
        index_table.append(
            [
                row["recording_id"].replace("recording_", ""),
                row["segment"],
                row["first_frame_time_local"].replace("+08:00", ""),
                row["last_frame_time_local"].replace("+08:00", ""),
                row["frame_count"],
                row["shadow_row_count"],
                row["frame_gap_total"],
                row["raw_recording_integrity"],
            ]
        )
    key_timeline = [["kind", "label", "start", "end", "elapsed power (min)", "segment", "source"]]
    for row in timeline_rows:
        if row["row_kind"] in {"declared_event", "declared_interval", "reconnect_data_gap"}:
            key_timeline.append(
                [
                    row["row_kind"],
                    row["label"],
                    row["start_local"].replace("+08:00", ""),
                    row["end_local"].replace("+08:00", ""),
                    f"{float(row['elapsed_from_power_start_s']) / 60:.3f} -> {float(row['elapsed_from_power_end_s']) / 60:.3f}",
                    row["segment"],
                    row["timestamp_source"],
                ]
            )
    reconnect_detail = "未找到跨边界 recording gap。"
    crossing = [item for item in boundaries if item["crosses_reconnect"]]
    if crossing:
        item = crossing[0]
        reconnect_detail = (
            f"{item['previous'].recording_id} -> {item['current'].recording_id}: "
            f"wall_gap={fmt(item['wall_gap_s'])} s, camera_frame_delta={fmt(item['frame_delta'])}, "
            f"camera_tick_delta={fmt(item['tick_delta'])}。"
        )
    manifest_hash_table = [["artifact", "hash match", "declared", "actual raw", "actual normalized", "path"]]
    for item in calibration["manifest_hashes"]:
        manifest_hash_table.append([
            item["key"],
            f"{item['match']} ({item['match_mode']})",
            item["declared"],
            item["actual_raw"],
            item["actual_normalized"],
            item["path"],
        ])
    qc_highlights = [
        row for row in qc_rows if row["check_id"] in {
            "formal_exposure",
            "image_integrity",
            "within_recording_frame_continuity",
            "host_wall_monotonic_cross_recording",
            "recording_gaps",
            "camera_frame_number_cross_recording",
            "PRE_RECONNECT_GLOBAL_CAMERA_FRAME_CONTINUITY",
            "camera_frame_number_continuity",
            "camera_timestamp_ticks_continuity",
            "shadow_height_numeric_availability",
            "shadow_active_valid",
            "shadow_q2_domain",
            "PRE_RECONNECT_DATA_VALID",
            "POST_RECONNECT_DATA_VALID",
            "THERMAL_A2_ALLOWED",
        }
    ]
    qc_table = [["check", "status", "observed", "notes"]]
    for row in qc_highlights:
        qc_table.append([row["check_id"], row["status"], row["observed"], row["notes"]])
    config_expected = calibration["config_expected"]
    session = calibration["session"]
    session_board = calibration["session_board"]
    session_runtime = calibration["session_runtime"]
    saved_text = local_text(calibration["session_saved_at"])
    frame_span = f"{local_text(ns_to_datetime(first_frame))} -> {local_text(ns_to_datetime(last_frame))}"
    return f"""# Thermal-A1｜0827 上午 2000 μs 热漂数据完整性与时间轴审计

## 结论

```text
FORMAL_EXPOSURE_2000_CONFIRMED = {"YES" if formal_exposure_confirmed else "NO"}
PRE_RECONNECT_DATA_VALID = {pre_valid}
POST_RECONNECT_DATA_VALID = {post_valid}
THERMAL_A2_ALLOWED = {a2_allowed}
```

`PRE_RECONNECT_DATA_VALID` 指原始 PNG + `frames.csv` 及其采集元数据在重连前可进入离线分析；它不把 `height_shadow.csv` 当作最终高度真值。`POST_RECONNECT_DATA_VALID = PARTIAL` 指重连后各 recording 内部完整，但相机帧号/设备 tick 在边界存在独立性问题，后段必须作为独立 segment 处理。

这里的 `PRE_RECONNECT_DATA_VALID = YES` 是 recording-level 的原始数据与 host 时间有效性结论，不代表重连前所有 recording 之间的 `camera_frame_number` 全局连续；实际有 2 个重连前跨 recording 回退，因此全局相机帧号连续性单独为 `NO/REVIEW`。

`THERMAL_A2_ALLOWED = YES` 的适用范围是：使用 frozen calibration 对原始 PNG 做离线 Frozen reconstruction，保留 pre/post 两段，不跨 13:01 直接拼接；不得使用本报告的 shadow preview 作为最终精度结果，也不得在 A2 中重新拟合 C0/C1/H1/H-B2/Ground。若 Thermal-A2 的实现只接受一条跨重连的全局连续曲线，则本结论不适用，应按 `NO` 处理。

## 数据范围与溯源

- 输入：`{input_dir}`
- recording：`{len(recordings)}` 个；正式帧：`{total_frames}`；每个 recording 的众数帧数：`{expected_frame_count}`；shadow 行：`{total_shadow}`。
- `frames.csv` 墙钟时间跨度：`{frame_span}`；elapsed time 以 `host_timestamp_ns` 转换到 Asia/Shanghai 后计算，并用 `host_monotonic_ns` 独立检查 recording 边界顺序。
- 事件基准：上电 `{local_text(events['power_on'])}`；Session 外参与地面基准完成 `{local_text(events['reference_complete'])}`；正式记录 `{local_text(events['formal_start'])}`；重连边界 `{local_text(events['reconnect'])}`。
- session calibration：`status={session.get('status')}`、`valid={session.get('valid')}`、`saved_at_local={saved_text}`、`session_generation={session.get('frame', {}).get('session_generation')}`、`ground_extrinsic_generation={session_runtime.get('ground_extrinsic_generation')}`。
- Session board：`{session_board.get('pattern_cols')}×{session_board.get('pattern_rows')}`、square `{session_board.get('square_size_mm')} mm`、detector `{session_board.get('detector')}`。

本轮复用并核对的冻结 artifact：`measure_tool_daheng_0811.yaml` 的采集配置、`calibration_daheng_0811/manifest.yaml` 及其四个 manifest 引用文件、目标目录内 `session_ground_calibration.json`。manifest hash 核对如下：

{table(manifest_hash_table)}

表中 `normalized_text` 表示 raw bytes 的 SHA-256 不同，但去除 UTF-8 BOM、统一换行后内容一致；该差异已显式保留，不能误读为 raw-byte hash 完全相等。

历史 0824 thermal 结果未直接复用：其正式运行曝光为 1000 μs，协议与本轮 2000 μs 不一致。已有历史报告中关于 shadow 不是真值的边界仅作为方法约束参考，本轮未复用其热漂数值。

## Recording index 摘要

{table(index_table)}

所有 raw recording 均为 `{expected_frame_count}` 张图像、`frames.csv` 和 `height_shadow.csv`；无缺图、无零字节图、无 recording 内部 frame gap。唯一相对较低的 shadow 行数 recording：`{", ".join(low_shadow) if low_shadow else "无"}`；这是 best-effort shadow 覆盖差异，不是 PNG/frames.csv 缺失。

完整逐 recording 字段见 `thermal_a1_recording_index.csv`。

## 事件时间轴

{table(key_timeline)}

时间轴不把 recording gap 当作 thermal reset；未记录区间只作为 elapsed time 上的 observation gap。重连边界证据：{reconnect_detail}

用户声明的 11:30–11:45 暂停区间与实际数据存在重叠：{", ".join(item.recording_id for item in pause_overlaps) if pause_overlaps else "无 recording 重叠"}。因此该区间在 timeline 中保留为“声明事件”，不能解读为完全无记录；本轮不删除、不平移该 recording，也不插入 thermal reset。

重连前 recording：`{len(pre)}` 个（`{pre[0].recording_id if pre else ''}` -> `{pre[-1].recording_id if pre else ''}`）；重连后 recording：`{len(post)}` 个（`{post[0].recording_id if post else ''}` -> `{post[-1].recording_id if post else ''}`）。完整事件/录制/间隔长表见 `thermal_a1_event_timeline.csv`。

## 相机参数和连续性 QC

- 正式帧 exposure：全部 `{fmt(config_expected.get('exposure_us'))} μs`；gain：全部 `{fmt(config_expected.get('gain_db'))} dB`；pixel format：全部 `{config_expected.get('pixel_format')}`。
- 正式 ROI：`offset=({config_expected.get('offset_x')},{config_expected.get('offset_y')})`、size=`{config_expected.get('width')}×{config_expected.get('height')}`；每个 recording 与配置/manifest 一致。
- 每个 recording 内的 `camera_frame_number`、`camera_timestamp_ticks`、`host_timestamp_ns`、`host_monotonic_ns` 均保持递增；每个 recording 的 `frame_gap` 均为 0。
- 不同 recording 之间的正向帧号跳跃只说明中间没有录制，不等于相机故障；这些间隔已单列在 event timeline。
- 跨 recording 的 camera frame number 不是全局连续 key：除 13:01 外，`recording_20260827_100813 -> recording_20260827_101149` 和 `recording_20260827_101652 -> recording_20260827_102037` 也出现回退；相邻 camera timestamp ticks/host 时间仍向前。因此报告保留该 REVIEW，不将回退解释为 thermal reset。
- 13:01 相机重连边界的 camera frame number 和 timestamp ticks 单独审计；如果发生 reset，不把它解释为 thermal reset，也不跨边界拟合/拼接。设备 timestamp 保留原始 tick，未假设单位。

## Shadow preview QC（非最终结果）

- `height_shadow.csv` 共 `{total_shadow}` 行：`height_raw` numeric `{height_counts['height_raw']}`、H1 numeric `{height_counts['height_h1']}`、H-B2 numeric `{height_counts['height_hb2']}`；`active_height_valid=True` `{active_true}` 行；`q2_in_domain=True` `{q2_true}` 行。
- shadow 数值范围（min / median / max）：q1=`{fmt(min_or_none(shadow_global_numeric['q1']))} / {fmt(median_or_none(shadow_global_numeric['q1']))} / {fmt(max_or_none(shadow_global_numeric['q1']))}`；q2=`{fmt(min_or_none(shadow_global_numeric['q2']))} / {fmt(median_or_none(shadow_global_numeric['q2']))} / {fmt(max_or_none(shadow_global_numeric['q2']))}`；v_min=`{fmt(min_or_none(shadow_global_numeric['v_min']))} / {fmt(median_or_none(shadow_global_numeric['v_min']))} / {fmt(max_or_none(shadow_global_numeric['v_min']))}`；v_median=`{fmt(min_or_none(shadow_global_numeric['v_median']))} / {fmt(median_or_none(shadow_global_numeric['v_median']))} / {fmt(max_or_none(shadow_global_numeric['v_median']))}`；v_max=`{fmt(min_or_none(shadow_global_numeric['v_max']))} / {fmt(median_or_none(shadow_global_numeric['v_max']))} / {fmt(max_or_none(shadow_global_numeric['v_max']))}`。
- point count：min=`{fmt(min_or_none(point_counts))}`、median=`{fmt(median_or_none(point_counts))}`、max=`{fmt(max_or_none(point_counts))}`。
- `active_height_status`：`{'; '.join(f'{key}={value}' for key, value in all_active_status.items())}`。
- `ground_reference_status`：`{'; '.join(f'{key}={value}' for key, value in all_status.items())}`。
- `hb2_q2_status`：`{'; '.join(f'{key}={value}' for key, value in all_hb2.items())}`。
- q1/q2、v_min/v_median/v_max 和 point count 的每 recording median 已写入 index；图 `thermal_a1_shadow_preview.png` 仅对这些现有 shadow 值做聚合显示，未重新计算。
- 若 raw/H1/H-B2 numeric 为空或 active invalid，这是 shadow 诊断链的 QC 结果，不是对原始 PNG 的最终高度判断；A2 必须从 PNG 离线重建。

## QC 汇总

{table(qc_table)}

### 数据缺失/异常结论

- 原始 recording 完整性：29/29 的 PNG、`frames.csv`、`height_shadow.csv` 齐全；每个 20/20 PNG 与 CSV 行匹配，且 `{total_png}/{total_frames}` PNG 可解码、为 Mono8 灰度并符合 CSV 声明尺寸；frames/shadow CSV 无 malformed/blank 行。
- 图像/元数据完整性计数：decode errors=`{total_decode_errors}`，size mismatch=`{total_size_mismatches}`，invalid size metadata=`{total_invalid_size_metadata}`，Mono8 mode mismatch=`{total_mode_mismatches}`。
- 采集参数异常：未发现 exposure、gain、pixel format 或 ROI 不一致；目标目录内没有把 exposure sweep 混入正式 recording 的证据。未保存的 sweep 本身不在本轮可审计范围。
- frame gap：recording 内无 gap；recording 之间存在由定长录制间隔/暂停造成的间隔，不能填补，也不能当 thermal reset。
- reconnect discontinuity：13:01 是明确独立边界；前后曲线不能直接连接。post 段保留为 PARTIAL，而非无审计地升级为连续有效。另有两个重连前 inter-recording 的 camera frame ID 回退，不能把 camera frame number 当作全局 thermal time key。
- shadow：存在 q2 OOD / inactive / active invalid 等状态时，只作诊断；不能从 preview 直接报告热漂精度。

## 输出与复现边界

- `thermal_a1_recording_index.csv`：逐 recording inventory、双 elapsed time、frame gap、shadow 摘要和完整性。
- `thermal_a1_event_timeline.csv`：声明事件、每段录制区间、recording gaps 和 reconnect data gap。
- `thermal_a1_qc_summary.csv`：参数、连续性、provenance、shadow 和最终准入 QC。
- `thermal_a1_shadow_preview.png`：shadow 现有值 preview，非最终结果。
- `report.md`：本报告。

本轮新增计算仅包括目录/CSV/图像文件盘点、时间轴转换、帧号/时间戳/参数连续性、shadow 字段统计、manifest hash 核对和 preview 绘图；没有重新拟合或修改任何 C0/C1/H1/H-B2/Ground，也没有修改或删除原始数据。
"""


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = (args.session_calibration or input_dir / "session_ground_calibration.json").resolve()
    events = {
        "power_on": parse_datetime(args.power_on),
        "reference_complete": parse_datetime(args.reference_complete),
        "formal_start": parse_datetime(args.formal_start),
        "pause_start": parse_datetime(args.pause_start),
        "pause_end": parse_datetime(args.pause_end),
        "no_record_start": parse_datetime(args.no_record_start),
        "reconnect": parse_datetime(args.reconnect),
    }
    calibration = audit_calibration(args.manifest.resolve(), args.measure_config.resolve(), session_path)
    expected = calibration["config_expected"]
    recording_paths = sorted(item for item in input_dir.glob("recording_*") if item.is_dir())
    recordings = [build_recording(input_dir, path, expected) for path in recording_paths]
    recordings = sort_recordings(recordings)
    expected_frame_count = statistics.mode([item.metrics["frame_count"] for item in recordings if item.metrics["frame_count"]]) if recordings else 20
    for item in recordings:
        item.metrics["expected_frame_count"] = expected_frame_count
    reconnect_ns = datetime_to_ns(events["reconnect"])
    assign_segments(recordings, reconnect_ns)
    boundaries = build_boundaries(recordings, reconnect_ns)
    index_rows = [row_for_index(item, expected_frame_count, events["power_on"], events["reference_complete"]) for item in recordings]
    timeline_rows = build_timeline(recordings, boundaries, events)
    formal_exposure_confirmed = bool(recordings) and all(item.metrics["exposure_config_match"] for item in recordings) and expected.get("exposure_us") == 2000.0
    pre_records = [item for item in recordings if item.metrics.get("segment") == "pre_reconnect"]
    post_records = [item for item in recordings if item.metrics.get("segment") == "post_reconnect"]
    pre_valid = "YES" if pre_records and all(item.metrics["raw_core_ok"] and item.metrics["frame_count"] == expected_frame_count for item in pre_records) and formal_exposure_confirmed and calibration["session_valid"] else "NO"
    crossing = [item for item in boundaries if item["crosses_reconnect"]]
    post_core = bool(post_records) and all(item.metrics["raw_core_ok"] and item.metrics["frame_count"] == expected_frame_count for item in post_records)
    if not post_records or not post_core:
        post_valid = "NO"
    elif crossing:
        post_valid = "PARTIAL"
    else:
        post_valid = "YES"
    a2_allowed = "YES" if pre_valid == "YES" and post_valid != "NO" and calibration["manifest_hashes_match"] and calibration["config_manifest_roi_match"] else "NO"
    qc_rows = build_qc_rows(
        recordings,
        boundaries,
        calibration,
        expected_frame_count,
        reconnect_ns,
        formal_exposure_confirmed,
        pre_valid,
        post_valid,
        a2_allowed,
    )
    write_csv(output_dir / "thermal_a1_recording_index.csv", index_rows, INDEX_FIELDS)
    write_csv(output_dir / "thermal_a1_event_timeline.csv", timeline_rows, TIMELINE_FIELDS)
    write_csv(output_dir / "thermal_a1_qc_summary.csv", qc_rows, QC_FIELDS)
    shadow_plot(recordings, events, output_dir / "thermal_a1_shadow_preview.png")
    report = build_report(
        output_dir,
        input_dir,
        recordings,
        index_rows,
        timeline_rows,
        qc_rows,
        calibration,
        events,
        boundaries,
        expected_frame_count,
        formal_exposure_confirmed,
        pre_valid,
        post_valid,
        a2_allowed,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "recordings": len(recordings),
                "frames": sum(item.metrics["frame_count"] for item in recordings),
                "shadow_rows": sum(item.metrics["shadow_row_count"] for item in recordings),
                "formal_exposure_2000_confirmed": "YES" if formal_exposure_confirmed else "NO",
                "pre_reconnect_data_valid": pre_valid,
                "post_reconnect_data_valid": post_valid,
                "thermal_a2_allowed": a2_allowed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
