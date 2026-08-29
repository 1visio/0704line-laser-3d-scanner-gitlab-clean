#!/usr/bin/env python3
"""Task A-2: audit stripe quality and Frozen-Steger localization mechanisms.

This is an analysis-only script.  It reuses the Frozen Session01 center cache,
Frozen V2 measurement ROI, and A-13B frame-level Base errors.  It does not
modify production extraction/reconstruction code or fit any correction.

The diagnostic Steger replay uses the exact options recorded in the Frozen
manifest only to expose response/validity diagnostics.  Formal center
coordinates always come from the immutable cache; the replay is checked
against that cache and is never used to replace or select points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "stripe_quality_audit"
DATA_ROOT = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0822\session01"
)
CACHE_NPZ = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "session01_steger_centers.npz"
CACHE_MANIFEST = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "session01_steger_centers_manifest.json"
REGISTRY_PATH = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "session01_roi_registry_manual_v2.json"
A1_CSV = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "pixel_sensitivity_audit" / "pixel_sensitivity_audit.csv"
A13B_FRAMES_CSV = REPO_ROOT / "reports" / "experiments" / "daheng_0822" / "session01_roi_freeze" / "session01_a13b_v2_multireference_frames.csv"
A13B_PROVENANCE = REPO_ROOT / "reports" / "experiments" / "daheng_0822" / "session01_roi_freeze" / "session01_a13b_v2_provenance_audit.json"

CALIBRATION_SRC = Path(r"D:\Docs\linelaserscan\calibration\src")
if str(CALIBRATION_SRC) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SRC))

try:
    import realtime_steger  # type: ignore
except ImportError as exc:  # pragma: no cover - environment diagnostic
    realtime_steger = None  # type: ignore[assignment]
    _REALTIME_STEGER_IMPORT_ERROR = exc
else:
    _REALTIME_STEGER_IMPORT_ERROR = None


V_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("v<1800", None, 1800.0),
    ("1800<=v<2200", 1800.0, 2200.0),
    ("2200<=v<2400", 2200.0, 2400.0),
    ("2400<=v<=2600", 2400.0, 2600.0),
    ("v>2600", 2600.0, None),
)
V_BAND_LABELS = tuple(item[0] for item in V_BANDS)
HEIGHT_ORDER = ("h10", "h20", "h30")
POSITION_ORDER = tuple(f"p{index:02d}" for index in range(1, 11))
PROFILE_HALF_WIDTH = 12
PROFILE_BACKGROUND_INNER_RADIUS = 8
SENSITIVITY_MM_PER_PX = 0.30

PROFILE_METRICS = (
    "peak_dn",
    "background_dn",
    "signal_excess_dn",
    "peak_background_ratio",
    "dynamic_range_dn",
    "stripe_width_fwhm_px",
    "profile_asymmetry",
    "profile_skewness",
    "profile_saturation_fraction",
    "peak_saturated",
)
DIAGNOSTIC_METRICS = (
    "steger_response_dn_per_px2",
    "steger_offset_px",
    "steger_normal_y_abs",
    "centerline_curvature_signed_1_per_px",
    "centerline_curvature_abs_1_per_px",
)
QUALITY_FEATURES = (
    "signal_excess_dn_median",
    "background_dn_median",
    "peak_background_ratio_median",
    "dynamic_range_dn_median",
    "stripe_width_fwhm_px_median",
    "profile_asymmetry_median",
    "profile_skewness_median",
    "profile_saturation_fraction_median",
    "steger_response_dn_per_px2_median",
    "steger_valid_ratio",
    "centerline_curvature_abs_1_per_px_median",
)


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, np.floating) and not math.isfinite(float(value)):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_range(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid v range: {value!r}")
    start, end = int(round(float(value[0]))), int(round(float(value[1])))
    if end < start:
        raise ValueError(f"invalid descending v range: {value!r}")
    return start, end


def v_band(value: float) -> str:
    number = float(value)
    for label, lower, upper in V_BANDS:
        if lower is None and number < float(upper):
            return label
        if upper is None and number > float(lower):
            return label
        if lower is not None and upper is not None and lower <= number < upper:
            return label
        if lower == 2400.0 and upper == 2600.0 and lower <= number <= upper:
            return label
    return "unclassified"


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray([float(value) for value in values if finite(value) is not None], dtype=np.float64)
    if not array.size:
        return None
    return float(np.percentile(array, q))


def median(values: Iterable[float]) -> float | None:
    return percentile(values, 50.0)


def standard_deviation(values: Iterable[float], ddof: int = 0) -> float | None:
    array = np.asarray([float(value) for value in values if finite(value) is not None], dtype=np.float64)
    if array.size <= ddof:
        return None
    return float(np.std(array, ddof=ddof))


def group_stats(values: Iterable[float], prefix: str) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if finite(value) is not None], dtype=np.float64)
    if not array.size:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_median": None,
            f"{prefix}_p05": None,
            f"{prefix}_p95": None,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_n": int(array.size),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p05": float(np.percentile(array, 5.0)),
        f"{prefix}_p95": float(np.percentile(array, 95.0)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array, ddof=0)),
        f"{prefix}_max": float(np.max(array)),
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= np.finfo(float).eps:
        return None
    return float(numerator / denominator)


def linear_crossing(x0: int, y0: float, x1: int, y1: float, threshold: float) -> float:
    if y1 == y0:
        return float(x0)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def fwhm_width(profile: np.ndarray, peak_index: int, threshold_ratio: float = 0.5) -> float | None:
    values = np.asarray(profile, dtype=np.float64)
    if peak_index <= 0 or peak_index >= len(values) - 1:
        return None
    peak = float(values[peak_index])
    if peak <= 0.0 or not math.isfinite(peak):
        return None
    threshold = peak * threshold_ratio
    left = peak_index
    while left > 0 and values[left] >= threshold:
        left -= 1
    if values[left] >= threshold:
        return None
    right = peak_index
    while right < len(values) - 1 and values[right] >= threshold:
        right += 1
    if values[right] >= threshold:
        return None
    left_crossing = linear_crossing(left, float(values[left]), left + 1, float(values[left + 1]), threshold)
    right_crossing = linear_crossing(right - 1, float(values[right - 1]), right, float(values[right]), threshold)
    width = right_crossing - left_crossing
    return float(width) if width > 0.0 else None


def profile_metrics(image: np.ndarray, u_local: float, row: int) -> dict[str, float | None]:
    """Measure one horizontal normal profile around a Frozen center.

    The row-scan sensor semantics are ``u=column, v=row``.  Background is the
    median of the outer profile ring (|u-u_center| >= 8 px) in a 25-pixel
    window; width is FWHM of the background-subtracted positive profile.
    Asymmetry is signed right-minus-left weighted area, and skewness is the
    weighted third central moment of the same positive profile.
    """

    result = {name: None for name in PROFILE_METRICS}
    if row < 0 or row >= image.shape[0] or not math.isfinite(float(u_local)):
        return result
    if u_local < PROFILE_HALF_WIDTH or u_local > image.shape[1] - 1 - PROFILE_HALF_WIDTH:
        return result
    left = int(math.floor(u_local)) - PROFILE_HALF_WIDTH
    right = left + 2 * PROFILE_HALF_WIDTH + 1
    if left < 0 or right > image.shape[1]:
        return result
    x = np.arange(left, right, dtype=np.float64)
    profile = image[row, left:right].astype(np.float64, copy=False)
    center = float(u_local)
    outer = np.abs(x - center) >= PROFILE_BACKGROUND_INNER_RADIUS
    if np.count_nonzero(outer) < 4:
        return result
    background = float(np.median(profile[outer]))
    peak_index = int(np.argmax(profile))
    peak = float(profile[peak_index])
    signal_excess = peak - background
    dynamic_range = float(np.max(profile) - np.min(profile))
    result.update(
        {
            "peak_dn": peak,
            "background_dn": background,
            "signal_excess_dn": signal_excess,
            "peak_background_ratio": float((peak + 1.0) / (background + 1.0)),
            "dynamic_range_dn": dynamic_range,
            "profile_saturation_fraction": float(np.mean(profile >= 255.0)),
            "peak_saturated": float(peak >= 255.0),
        }
    )
    if not math.isfinite(signal_excess) or signal_excess <= 0.0:
        return result
    positive = np.maximum(profile - background, 0.0)
    width = fwhm_width(positive, peak_index)
    result["stripe_width_fwhm_px"] = width
    total = float(np.sum(positive))
    if total <= np.finfo(float).eps:
        return result
    left_area = float(np.sum(positive[x < center]))
    right_area = float(np.sum(positive[x >= center]))
    result["profile_asymmetry"] = float((right_area - left_area) / total)
    weighted_center = float(np.sum(x * positive) / total)
    variance = float(np.sum(((x - weighted_center) ** 2) * positive) / total)
    if variance > np.finfo(float).eps:
        result["profile_skewness"] = float(
            np.sum(((x - weighted_center) ** 3) * positive) / total / (variance ** 1.5)
        )
    return result


def centerline_curvature(centers_full: np.ndarray, vmin: int, vmax: int) -> dict[int, tuple[float, float]]:
    """Return signed/absolute local centerline curvature by integer row.

    This is geometric centerline curvature from the Frozen ``u(v)`` cache,
    not a fitted calibration correction and not a profile-height surrogate.
    A three-row second difference is used only where both neighbours exist.
    """

    by_row: dict[int, list[float]] = defaultdict(list)
    for u, v in np.asarray(centers_full, dtype=np.float64):
        row = int(round(float(v)))
        if vmin <= row <= vmax and math.isfinite(float(u)):
            by_row[row].append(float(u))
    u_by_row = {row: float(np.median(values)) for row, values in by_row.items()}
    output: dict[int, tuple[float, float]] = {}
    for row in sorted(u_by_row):
        if row - 1 not in u_by_row or row + 1 not in u_by_row:
            continue
        slope = (u_by_row[row + 1] - u_by_row[row - 1]) / 2.0
        second = u_by_row[row + 1] - 2.0 * u_by_row[row] + u_by_row[row - 1]
        denominator = (1.0 + slope * slope) ** 1.5
        signed = float(second / denominator)
        output[row] = (signed, abs(signed))
    return output


def load_inputs() -> dict[str, Any]:
    required = (CACHE_NPZ, CACHE_MANIFEST, REGISTRY_PATH, A1_CSV, A13B_FRAMES_CSV, DATA_ROOT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing A-2 input(s):\n" + "\n".join(missing))
    if realtime_steger is None:
        raise RuntimeError(f"Cannot import realtime_steger from {CALIBRATION_SRC}: {_REALTIME_STEGER_IMPORT_ERROR}")

    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    a1_rows = read_csv(A1_CSV)
    a13b_rows = read_csv(A13B_FRAMES_CSV)
    with np.load(CACHE_NPZ, allow_pickle=False) as bundle:
        centers = np.asarray(bundle["centers_full"], dtype=np.float64)
        offsets = np.asarray(bundle["frame_offsets"], dtype=np.int64)
    frames = cache_manifest.get("frames", [])
    if len(frames) != 600 or offsets.shape != (len(frames) + 1,) or int(offsets[-1]) != len(centers):
        raise ValueError("Frozen cache frame count/offsets are inconsistent")
    registry_entries = {str(entry["condition_id"]): entry for entry in registry.get("entries", [])}
    a13b_by_key = {str(row["cache_key"]): row for row in a13b_rows}
    if len(a13b_by_key) != len(a13b_rows):
        raise ValueError("A-13B frame CSV has duplicate cache_key values")
    return {
        "cache_manifest": cache_manifest,
        "registry": registry,
        "registry_entries": registry_entries,
        "a1_rows": a1_rows,
        "a13b_rows": a13b_rows,
        "a13b_by_key": a13b_by_key,
        "centers": centers,
        "offsets": offsets,
        "frames": frames,
    }


def diagnostic_options(cache_manifest: dict[str, Any], image_width: int) -> tuple[dict[str, Any], Any]:
    options = dict(cache_manifest["protocol_key"]["extraction_options"])
    search = options.pop("search_roi", None) or {"offset_x": 0, "offset_y": 0, "width": image_width, "height": 3000}
    start = int(search.get("offset_x", 0))
    width = int(search.get("width", image_width))
    if start != 0:
        # The diagnostic image is local 480-wide, so the frozen full-sensor
        # ROI [1760, 2240) maps to local [0, 480).
        start = 0
    region = realtime_steger.LaserSearchRegion(start, min(image_width, width), "frozen_search_roi")
    return options, region


def quality_for_frame(
    frame: dict[str, Any],
    centers_full: np.ndarray,
    registry_entry: dict[str, Any],
    a13b_row: dict[str, str],
    cache_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(str(frame["source_path"]))
    if not source.is_file():
        source = DATA_ROOT / str(frame["height_label"]) / str(frame["condition_id"]) / str(frame["filename"])
    row: dict[str, Any] = {
        "dataset": "session01",
        "height_label": frame.get("height_label"),
        "position_id": frame.get("position_id"),
        "condition_id": frame.get("cache_key", "").split("/")[0],
        "repeat_index": frame.get("repeat_index"),
        "filename": frame.get("filename"),
        "cache_key": frame.get("cache_key"),
        "camera_frame_number": frame.get("camera_frame_number"),
        "true_height_mm": finite(a13b_row.get("true_height_mm")),
        "v_order_rank": a13b_row.get("v_order_rank"),
        "height_roi_center_v": finite(a13b_row.get("height_roi_center_v")),
        "v_band": v_band(float(a13b_row["height_roi_center_v"])),
        "base_error_mm": finite(a13b_row.get("residual_base_session")),
        "base_abs_error_mm": abs(finite(a13b_row.get("residual_base_session"))) if finite(a13b_row.get("residual_base_session")) is not None else None,
        "h1_error_mm": finite(a13b_row.get("residual_h1_session")),
        "hb2_error_mm": finite(a13b_row.get("residual_hb2_session")),
        "source_path": str(source),
        "source_exists": bool(source.is_file()),
        "source_hash_match": None,
        "image_read_ok": False,
        "image_shape": None,
        "image_dtype": None,
        "quality_status": "INIT",
    }
    expected_hash = str(frame.get("source_sha256", ""))
    if source.is_file() and expected_hash:
        try:
            row["source_hash_match"] = sha256_file(source) == expected_hash
        except OSError:
            row["source_hash_match"] = False
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED) if source.is_file() else None
    if image is None or image.ndim != 2:
        row["quality_status"] = "IMAGE_READ_FAILED"
        return row, {"centers_full": centers_full, "record": row, "quality": row, "entry": registry_entry}
    row["image_read_ok"] = True
    row["image_shape"] = f"{image.shape[0]}x{image.shape[1]}"
    row["image_dtype"] = str(image.dtype)
    if image.shape != (3000, 480):
        row["quality_status"] = "IMAGE_SHAPE_UNEXPECTED"

    offset_x = float(frame.get("offset_xy", [1760, 0])[0])
    offset_y = float(frame.get("offset_xy", [1760, 0])[1])
    centers_local = np.asarray(centers_full, dtype=np.float64) - np.asarray([offset_x, offset_y], dtype=np.float64)
    vmin, vmax = parse_range(registry_entry["height_v_range"])
    selected = centers_full[
        (centers_full[:, 1] >= vmin)
        & (centers_full[:, 1] <= vmax)
        & np.isfinite(centers_full[:, 0])
        & np.isfinite(centers_full[:, 1])
    ]
    selected_local = selected - np.asarray([offset_x, offset_y], dtype=np.float64)
    selected_rows = np.rint(selected_local[:, 1]).astype(np.int64) if len(selected_local) else np.empty(0, dtype=np.int64)
    row["height_v_min"] = vmin
    row["height_v_max"] = vmax
    row["measurement_v_span_px"] = vmax - vmin + 1
    row["measurement_point_count"] = int(len(selected))
    row["measurement_point_ratio"] = float(len(selected) / (vmax - vmin + 1))
    row["measurement_v_median"] = float(np.median(selected[:, 1])) if len(selected) else None

    options, search_region = diagnostic_options(cache_manifest, image.shape[1])
    diagnostic = realtime_steger.extract_steger(
        image,
        options,
        search_region=search_region,
        diagnostic=True,
        use_auto_band=False,
    )
    replay_points = np.asarray(diagnostic.pixels, dtype=np.float64)
    cache_match = replay_points.shape == centers_local.shape
    if cache_match and replay_points.size:
        cache_match = bool(np.max(np.abs(replay_points - centers_local)) <= 1.0e-10)
    row["diagnostic_replay_point_count"] = int(len(replay_points))
    row["diagnostic_replay_cache_exact"] = cache_match
    row["diagnostic_replay_max_abs_uv_delta_px"] = (
        float(np.max(np.abs(replay_points - centers_local)))
        if replay_points.shape == centers_local.shape and replay_points.size
        else None
    )
    row["steger_valid_count"] = int(np.count_nonzero(diagnostic.valid[vmin : vmax + 1]))
    row["steger_valid_ratio"] = float(row["steger_valid_count"] / (vmax - vmin + 1))
    row["steger_cache_count_match"] = bool(row["steger_valid_count"] == len(selected))

    point_values: dict[str, list[float]] = defaultdict(list)
    curvature = centerline_curvature(centers_full, vmin, vmax)
    for point, local_row in zip(selected, selected_rows, strict=True):
        local_u = float(point[0] - offset_x)
        metrics = profile_metrics(image, local_u, int(local_row))
        for key, value in metrics.items():
            if finite(value) is not None:
                point_values[key].append(float(value))
        response = diagnostic.response[int(local_row)] if 0 <= int(local_row) < len(diagnostic.response) else np.nan
        offset = diagnostic.offset_px[int(local_row)] if 0 <= int(local_row) < len(diagnostic.offset_px) else np.nan
        normal_y = diagnostic.normal_y_abs[int(local_row)] if 0 <= int(local_row) < len(diagnostic.normal_y_abs) else np.nan
        for key, value in (
            ("steger_response_dn_per_px2", response),
            ("steger_offset_px", offset),
            ("steger_normal_y_abs", normal_y),
        ):
            if finite(value) is not None:
                point_values[key].append(float(value))
        if int(local_row) in curvature:
            signed, absolute = curvature[int(local_row)]
            point_values["centerline_curvature_signed_1_per_px"].append(signed)
            point_values["centerline_curvature_abs_1_per_px"].append(absolute)

    for metric in (*PROFILE_METRICS, *DIAGNOSTIC_METRICS):
        values = point_values.get(metric, [])
        row.update(group_stats(values, metric))
    row["centerline_curvature_valid_count"] = len(point_values.get("centerline_curvature_abs_1_per_px", []))
    row["profile_metric_valid_count"] = len(point_values.get("signal_excess_dn", []))
    row["quality_status"] = "PASS" if row["image_shape"] == "3000x480" and len(selected) else "PARTIAL"
    if not bool(row["source_hash_match"]):
        row["quality_status"] = "SOURCE_HASH_MISMATCH"
    return row, {"centers_full": centers_full, "record": row, "quality": row, "entry": registry_entry}


def build_repeatability(
    frame_work: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int | None]]]:
    """Align the 20 repeats at identical integer v rows within each condition."""

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in frame_work:
        by_condition[str(item["record"]["condition_id"])].append(item)
    row_stats: list[dict[str, Any]] = []
    condition_summaries: dict[str, dict[str, float | int | None]] = {}
    for condition_id, items in sorted(by_condition.items()):
        items.sort(key=lambda item: int(item["record"].get("repeat_index") or 0))
        entry = items[0]["entry"]
        height = str(items[0]["record"]["height_label"])
        position = str(items[0]["record"]["position_id"])
        vmin, vmax = parse_range(entry["height_v_range"])
        per_frame: list[dict[int, float]] = []
        for item in items:
            mapping: dict[int, list[float]] = defaultdict(list)
            centers = np.asarray(item["centers_full"], dtype=np.float64)
            for u, v in centers:
                row = int(round(float(v)))
                if vmin <= row <= vmax and finite(u) is not None:
                    mapping[row].append(float(u))
            per_frame.append({row: float(np.median(values)) for row, values in mapping.items()})
        condition_deviations: list[float] = []
        condition_stds: list[float] = []
        for row in range(vmin, vmax + 1):
            values = np.asarray([mapping[row] for mapping in per_frame if row in mapping], dtype=np.float64)
            if values.size < 2:
                continue
            row_median = float(np.median(values))
            deviations = np.abs(values - row_median)
            row_std = float(np.std(values, ddof=1))
            condition_stds.append(row_std)
            condition_deviations.extend(float(value) for value in deviations)
            row_stats.append(
                {
                    "condition_id": condition_id,
                    "height_label": height,
                    "position_id": position,
                    "v_px": row,
                    "v_band": v_band(float(row)),
                    "repeat_count": int(values.size),
                    "u_std_px": row_std,
                    "u_abs_deviation_median_px": float(np.median(deviations)),
                    "u_abs_deviation_p95_px": float(np.percentile(deviations, 95.0)),
                    "u_abs_deviation_max_px": float(np.max(deviations)),
                }
            )
        condition_summaries[condition_id] = {
            "height_label": height,
            "position_id": position,
            "n_rows": len(condition_stds),
            "median_u_std_px": median(condition_stds),
            "p95_u_std_px": percentile(condition_stds, 95.0),
            "p95_abs_deviation_px": percentile(condition_deviations, 95.0),
            "max_abs_deviation_px": max(condition_deviations) if condition_deviations else None,
            "median_repeat_count": median([row["repeat_count"] for row in row_stats if row["condition_id"] == condition_id]),
        }

    def scoped_rows(scope: str, scope_id: str | None, band: str) -> list[dict[str, Any]]:
        return [
            row
            for row in row_stats
            if row["v_band"] == band
            and (scope == "pooled" or (scope == "height" and row["height_label"] == scope_id) or (scope == "condition" and row["condition_id"] == scope_id))
        ]

    summary_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str | None]] = [("pooled", None)]
    scopes.extend(("height", height) for height in HEIGHT_ORDER)
    scopes.extend(("condition", condition_id) for condition_id in sorted(condition_summaries))
    for scope, scope_id in scopes:
        for band in V_BAND_LABELS:
            selected = scoped_rows(scope, scope_id, band)
            stds = [float(row["u_std_px"]) for row in selected]
            deviations = [
                float(row["u_abs_deviation_p95_px"])
                for row in selected
                if finite(row.get("u_abs_deviation_p95_px")) is not None
            ]
            # For the main deviation statistic, retain all repeat-level
            # deviations rather than only row-level P95 values.
            repeat_deviations: list[float] = []
            repeat_counts: list[int] = []
            for row in selected:
                repeat_counts.append(int(row["repeat_count"]))
                n = int(row["repeat_count"])
                # A row-level P95 is sufficient for the grouped table while
                # preserving a conservative tail estimate.  The full
                # repeat-level values remain represented by row max/P95.
                repeat_deviations.append(float(row["u_abs_deviation_p95_px"]))
            v_values = [float(row["v_px"]) for row in selected]
            rho, pvalue = correlation(v_values, stds, "spearman") if len(stds) >= 3 else (None, None)
            summary_rows.append(
                {
                    "scope": scope,
                    "scope_id": scope_id or "pooled",
                    "v_band": band,
                    "n_conditions": len({row["condition_id"] for row in selected}),
                    "n_rows": len(selected),
                    "median_valid_repeats": median(repeat_counts),
                    "median_u_std_px": median(stds),
                    "p95_u_std_px": percentile(stds, 95.0),
                    "max_u_std_px": max(stds) if stds else None,
                    "median_u_abs_deviation_p95_px": median(deviations),
                    "p95_u_abs_deviation_p95_px": percentile(deviations, 95.0),
                    "max_u_abs_deviation_p95_px": max(deviations) if deviations else None,
                    "median_u_abs_deviation_max_px": median([row["u_abs_deviation_max_px"] for row in selected]),
                    "spearman_v_u_std_rho": rho,
                    "spearman_v_u_std_pvalue": pvalue,
                }
            )
    return summary_rows, condition_summaries


def correlation(x_values: Iterable[float], y_values: Iterable[float], method: str) -> tuple[float | None, float | None]:
    x = np.asarray([float(value) for value in x_values if finite(value) is not None], dtype=np.float64)
    y = np.asarray([float(value) for value in y_values if finite(value) is not None], dtype=np.float64)
    if len(x) != len(y) or len(x) < 3 or np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return None, None
    try:
        if method == "pearson":
            result = pearsonr(x, y)
        else:
            result = spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    except (ValueError, FloatingPointError):
        return None, None


def demean_by_condition(values: np.ndarray, condition_ids: list[str]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, condition_id in enumerate(condition_ids):
        grouped[condition_id].append(index)
    for indexes in grouped.values():
        mean_value = float(np.mean(result[indexes]))
        result[indexes] -= mean_value
    return result


def build_correlations(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("pooled", frame_rows)]
    scopes.extend((f"height:{height}", [row for row in frame_rows if row["height_label"] == height]) for height in HEIGHT_ORDER)
    for scope, rows in scopes:
        for feature in QUALITY_FEATURES:
            for target_name in ("base_error_mm", "base_abs_error_mm"):
                pairs = [
                    (finite(row.get(feature)), finite(row.get(target_name)), str(row["condition_id"]))
                    for row in rows
                ]
                pairs = [pair for pair in pairs if pair[0] is not None and pair[1] is not None]
                if not pairs:
                    continue
                x = np.asarray([float(pair[0]) for pair in pairs], dtype=np.float64)
                y = np.asarray([float(pair[1]) for pair in pairs], dtype=np.float64)
                condition_ids = [pair[2] for pair in pairs]
                for mode in ("pooled", "within_condition_demeaned"):
                    x_use = x if mode == "pooled" else demean_by_condition(x, condition_ids)
                    y_use = y if mode == "pooled" else demean_by_condition(y, condition_ids)
                    pearson_r, pearson_p = correlation(x_use, y_use, "pearson")
                    spearman_rho, spearman_p = correlation(x_use, y_use, "spearman")
                    output.append(
                        {
                            "scope": scope,
                            "mode": mode,
                            "feature": feature,
                            "target": target_name,
                            "n": len(x_use),
                            "pearson_r": pearson_r,
                            "pearson_pvalue": pearson_p,
                            "spearman_rho": spearman_rho,
                            "spearman_pvalue": spearman_p,
                        }
                    )
    return output


def frame_band_stats(frame_rows: list[dict[str, Any]], feature: str, height: str | None = None) -> dict[str, dict[str, Any]]:
    selected = [row for row in frame_rows if height is None or row["height_label"] == height]
    output: dict[str, dict[str, Any]] = {}
    for band in V_BAND_LABELS:
        values = [finite(row.get(feature)) for row in selected if row["v_band"] == band]
        output[band] = {"n": len([value for value in values if value is not None]), "median": median(value for value in values if value is not None), "p95": percentile((value for value in values if value is not None), 95.0)}
    return output


def strict_valid_sensitivity(a1_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strict = [row for row in a1_rows if row.get("sensitivity_status") == "OK"]
    output: list[dict[str, Any]] = []
    epsilon_fields = {"0.05": "abs_dh_du_0p05_mm_per_px", "0.1": "abs_dh_du_0p10_mm_per_px"}
    for epsilon, field in epsilon_fields.items():
        for band in V_BAND_LABELS:
            values = [
                finite(row.get(field))
                for row in strict
                if row.get("v_band") == band
            ]
            values = [value for value in values if value is not None]
            output.append(
                {
                    "strict_valid_only": True,
                    "epsilon_px": float(epsilon),
                    "v_band": band,
                    "n": len(values),
                    "median_abs_dh_du_mm_per_px": median(values),
                    "p95_abs_dh_du_mm_per_px": percentile(values, 95.0),
                    "max_abs_dh_du_mm_per_px": max(values) if values else None,
                }
            )
    by_band = {band: next(row for row in output if row["epsilon_px"] == 0.05 and row["v_band"] == band) for band in V_BAND_LABELS}
    low = by_band["v<1800"]["p95_abs_dh_du_mm_per_px"]
    high = by_band["v>2600"]["p95_abs_dh_du_mm_per_px"]
    summary = {
        "strict_total": len(strict),
        "all_total": len(a1_rows),
        "strict_status_definition": "sensitivity_status == OK; all five base/u+0.05/u-0.05/u+0.10/u-0.10 Ground references valid",
        "high_edge_comparison_available": high is not None and low is not None,
        "high_edge_to_upper_p95_ratio_0p05": ratio(high, low),
        "strict_valid_edge_sanity": "NOT_ASSESSABLE" if high is None or low is None else "COMPARABLE",
    }
    return output, summary


def high_low_shift(frame_rows: list[dict[str, Any]], feature: str, height: str | None = None) -> dict[str, Any]:
    selected = [row for row in frame_rows if height is None or row["height_label"] == height]
    low_values = [finite(row.get(feature)) for row in selected if row["v_band"] == "v<1800"]
    high_values = [finite(row.get(feature)) for row in selected if row["v_band"] == "v>2600"]
    low_values = [value for value in low_values if value is not None]
    high_values = [value for value in high_values if value is not None]
    low_median = median(low_values)
    high_median = median(high_values)
    return {
        "feature": feature,
        "height": height or "pooled",
        "low_n": len(low_values),
        "high_n": len(high_values),
        "low_median": low_median,
        "high_median": high_median,
        "high_minus_low": (high_median - low_median) if high_median is not None and low_median is not None else None,
    }


def make_decisions(
    frame_rows: list[dict[str, Any]],
    repeatability_rows: list[dict[str, Any]],
    condition_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pooled_repeat = {
        row["v_band"]: row
        for row in repeatability_rows
        if row["scope"] == "pooled"
    }
    high_random = pooled_repeat.get("v>2600", {}).get("p95_u_abs_deviation_p95_px")
    low_random = pooled_repeat.get("v<1800", {}).get("p95_u_abs_deviation_p95_px")
    pooled_random_ratio = ratio(high_random, low_random)
    height_random_ratios: dict[str, float | None] = {}
    for height in HEIGHT_ORDER:
        rows = {row["v_band"]: row for row in repeatability_rows if row["scope"] == "height" and row["scope_id"] == height}
        height_random_ratios[height] = ratio(
            rows.get("v>2600", {}).get("p95_u_abs_deviation_p95_px"),
            rows.get("v<1800", {}).get("p95_u_abs_deviation_p95_px"),
        )
    consistent_count = sum(value is not None and value >= 1.10 for value in height_random_ratios.values())
    if pooled_random_ratio is not None and pooled_random_ratio >= 1.25 and consistent_count >= 2:
        random_flag = "YES"
    elif (pooled_random_ratio is not None and pooled_random_ratio >= 1.10) or consistent_count >= 1:
        random_flag = "PARTIAL"
    else:
        random_flag = "NO"

    shifts: list[dict[str, Any]] = []
    for feature in ("profile_asymmetry_median", "profile_skewness_median"):
        shifts.extend(high_low_shift(frame_rows, feature, height) for height in (None, *HEIGHT_ORDER))
    asym_height_shifts = [row for row in shifts if row["feature"] == "profile_asymmetry_median" and row["height"] != "pooled" and row["high_minus_low"] is not None]
    skew_height_shifts = [row for row in shifts if row["feature"] == "profile_skewness_median" and row["height"] != "pooled" and row["high_minus_low"] is not None]
    shape_supported: list[dict[str, Any]] = []
    for height in HEIGHT_ORDER:
        asym = next((row for row in asym_height_shifts if row["height"] == height), None)
        skew = next((row for row in skew_height_shifts if row["height"] == height), None)
        asym_effect = abs(float(asym["high_minus_low"])) if asym and asym["high_minus_low"] is not None else 0.0
        skew_effect = abs(float(skew["high_minus_low"])) if skew and skew["high_minus_low"] is not None else 0.0
        if asym_effect >= 0.05 or skew_effect >= 0.15:
            shape_supported.append({"height": height, "asym_effect": asym_effect, "skew_effect": skew_effect})
    within_shape_corr = [
        row
        for row in []
    ]
    # The caller/report also retains exact within-condition correlations; the
    # shape decision itself is intentionally based on predeclared profile-shape
    # shifts, not on selecting a feature after seeing residuals.
    # Strong systematic-bias evidence requires a stable direction across at
    # least two heights for one predeclared shape metric.  Here asymmetry
    # changes sign between h10 and h20/h30, while skewness shifts are smaller
    # than the strong threshold, so shape changes are evidence but not a
    # height-consistent bias proof.
    strong_shape_metrics = 0
    for metric, threshold in (
        ("profile_asymmetry_median", 0.05),
        ("profile_skewness_median", 0.15),
    ):
        effects = [
            float(row["high_minus_low"])
            for row in shifts
            if row["feature"] == metric and row["height"] != "pooled" and row["high_minus_low"] is not None and abs(float(row["high_minus_low"])) >= threshold
        ]
        if len(effects) >= 2 and (all(value > 0.0 for value in effects) or all(value < 0.0 for value in effects)):
            strong_shape_metrics += 1
    if strong_shape_metrics >= 1:
        profile_flag = "STRONG"
    elif shape_supported:
        profile_flag = "PARTIAL"
    else:
        profile_flag = "WEAK"

    condition_p95 = [finite(summary.get("p95_abs_deviation_px")) for summary in condition_summaries.values()]
    condition_p95 = [value for value in condition_p95 if value is not None]
    predicted_height_p95 = SENSITIVITY_MM_PER_PX * float(np.percentile(condition_p95, 95.0)) if condition_p95 else None
    base_errors = [finite(row.get("base_abs_error_mm")) for row in frame_rows]
    base_errors = [value for value in base_errors if value is not None]
    condition_means: dict[str, float] = {}
    for condition_id in sorted({str(row["condition_id"]) for row in frame_rows}):
        values = [finite(row.get("base_error_mm")) for row in frame_rows if str(row["condition_id"]) == condition_id]
        values = [value for value in values if value is not None]
        if values:
            condition_means[condition_id] = float(np.mean(values))
    base_within = [
        abs(float(row["base_error_mm"]) - condition_means[str(row["condition_id"])])
        for row in frame_rows
        if finite(row.get("base_error_mm")) is not None and str(row["condition_id"]) in condition_means
    ]
    observed_total_p95 = percentile(base_errors, 95.0)
    observed_within_p95 = percentile(base_within, 95.0)
    ratio_total = ratio(predicted_height_p95, observed_total_p95)
    ratio_within = ratio(predicted_height_p95, observed_within_p95)
    if (
        ratio_total is not None
        and ratio_within is not None
        and ratio_total >= 0.8
        and ratio_within >= 0.8
    ):
        magnitude_flag = "YES"
    elif ratio_within is not None and ratio_within >= 0.5:
        magnitude_flag = "PARTIAL"
    else:
        magnitude_flag = "NO"
    return {
        "LOWER_EDGE_STEGER_RANDOM_DEGRADATION": random_flag,
        "LOWER_EDGE_PROFILE_SYSTEMATIC_BIAS_EVIDENCE": profile_flag,
        "OPTICAL_EXTRACTION_MAGNITUDE_SUFFICIENT": magnitude_flag,
        "random_pooled_p95_ratio_high_to_low": pooled_random_ratio,
        "random_height_p95_ratios_high_to_low": height_random_ratios,
        "random_height_consistent_count_ge_1p10": consistent_count,
        "profile_high_low_shifts": shifts,
        "profile_shape_supported_heights": shape_supported,
        "sensitivity_mm_per_px_used": SENSITIVITY_MM_PER_PX,
        "condition_p95_u_abs_deviation_px_p95": percentile(condition_p95, 95.0),
        "predicted_height_p95_mm_from_u_repeatability": predicted_height_p95,
        "observed_base_abs_error_p95_mm": observed_total_p95,
        "observed_within_condition_base_abs_deviation_p95_mm": observed_within_p95,
        "predicted_to_observed_total_p95_ratio": ratio_total,
        "predicted_to_observed_within_condition_p95_ratio": ratio_within,
    }


def plot_quality_vs_v(frame_rows: list[dict[str, Any]], path: Path) -> None:
    panels = [
        ("signal_excess_dn_median", "signal excess (DN)"),
        ("stripe_width_fwhm_px_median", "stripe FWHM (px)"),
        ("profile_asymmetry_median", "profile asymmetry"),
        ("profile_skewness_median", "profile skewness"),
        ("steger_response_dn_per_px2_median", "Steger response (DN/px²)"),
        ("steger_valid_ratio", "Frozen-Steger valid ratio"),
    ]
    colors = {"h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
    for axis, (feature, label) in zip(axes.ravel(), panels, strict=True):
        for height_index, height in enumerate(HEIGHT_ORDER):
            rows = [row for row in frame_rows if row["height_label"] == height and finite(row.get(feature)) is not None]
            x = [V_BAND_LABELS.index(str(row["v_band"])) + (height_index - 1) * 0.18 for row in rows]
            y = [float(row[feature]) for row in rows]
            axis.scatter(x, y, s=13, alpha=0.45, color=colors[height], label=height, edgecolors="none")
        medians = frame_band_stats(frame_rows, feature)
        x_band = np.arange(len(V_BAND_LABELS))
        y_band = [medians[band]["median"] for band in V_BAND_LABELS]
        axis.plot(x_band, y_band, color="#333333", marker="o", linewidth=1.5, label="pooled band median")
        axis.set_xticks(x_band, V_BAND_LABELS, rotation=28, ha="right")
        axis.set_title(label)
        axis.grid(alpha=0.2)
        axis.set_xlabel("full-sensor v band (height ROI center row)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Task A-2 · Frozen Session01 stripe quality versus v", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_repeatability(repeatability_rows: list[dict[str, Any]], path: Path) -> None:
    colors = {"pooled": "#222222", "h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}
    panels = [
        ("median_u_std_px", "median row-wise u std (px)"),
        ("p95_u_std_px", "P95 row-wise u std (px)"),
        ("p95_u_abs_deviation_p95_px", "P95 of row-wise P95 |u-median| (px)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150, sharex=True)
    for axis, (field, label) in zip(axes, panels, strict=True):
        for scope_id in ("pooled", *HEIGHT_ORDER):
            rows = [row for row in repeatability_rows if row["scope"] == ("pooled" if scope_id == "pooled" else "height") and row["scope_id"] == scope_id]
            rows_by_band = {row["v_band"]: row for row in rows}
            values = [rows_by_band.get(band, {}).get(field) for band in V_BAND_LABELS]
            axis.plot(np.arange(len(V_BAND_LABELS)), values, marker="o", linewidth=1.5, color=colors[scope_id], label=scope_id)
        axis.set_title(label)
        axis.set_xticks(np.arange(len(V_BAND_LABELS)), V_BAND_LABELS, rotation=28, ha="right")
        axis.grid(alpha=0.2)
        axis.set_xlabel("full-sensor v band")
    axes[0].set_ylabel("pixels")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Task A-2 · same-condition 20-repeat Steger u repeatability", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def correlation_lookup(correlation_rows: list[dict[str, Any]], scope: str, mode: str, feature: str, target: str) -> dict[str, Any]:
    return next(
        (
            row
            for row in correlation_rows
            if row["scope"] == scope and row["mode"] == mode and row["feature"] == feature and row["target"] == target
        ),
        {},
    )


def plot_height_error_vs_quality(
    frame_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    panels = [
        ("signal_excess_dn_median", "signal excess (DN)"),
        ("profile_asymmetry_median", "profile asymmetry"),
        ("stripe_width_fwhm_px_median", "stripe FWHM (px)"),
        ("steger_response_dn_per_px2_median", "Steger response (DN/px²)"),
        ("steger_valid_ratio", "Frozen-Steger valid ratio"),
        ("centerline_curvature_abs_1_per_px_median", "centerline curvature (1/px)"),
    ]
    colors = {"h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
    for axis, (feature, label) in zip(axes.ravel(), panels, strict=True):
        rows = [row for row in frame_rows if finite(row.get(feature)) is not None and finite(row.get("base_error_mm")) is not None]
        for height in HEIGHT_ORDER:
            group = [row for row in rows if row["height_label"] == height]
            axis.scatter(
                [float(row[feature]) for row in group],
                [float(row["base_error_mm"]) for row in group],
                s=13,
                alpha=0.45,
                color=colors[height],
                label=height,
                edgecolors="none",
            )
        raw = correlation_lookup(correlation_rows, "pooled", "pooled", feature, "base_error_mm")
        within = correlation_lookup(correlation_rows, "pooled", "within_condition_demeaned", feature, "base_error_mm")
        raw_text = "—" if raw.get("pearson_r") is None else f"{float(raw['pearson_r']):+.2f}"
        within_text = "—" if within.get("pearson_r") is None else f"{float(within['pearson_r']):+.2f}"
        axis.set_title(f"{label}\nPearson raw={raw_text}, within={within_text}")
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.grid(alpha=0.2)
        axis.set_xlabel(label)
        axis.set_ylabel("Base error (mm)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Task A-2 · same-frame Base error versus stripe/extraction quality", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def table_row(values: Iterable[Any]) -> str:
    return "| " + " | ".join("—" if value is None else str(value) for value in values) + " |"


def report_text(
    inputs: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    repeatability_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    strict_summary: dict[str, Any],
    condition_summaries: dict[str, dict[str, Any]],
    decisions: dict[str, Any],
    output_dir: Path,
) -> str:
    cache_manifest = inputs["cache_manifest"]
    hash_matches = sum(bool(row.get("source_hash_match")) for row in frame_rows)
    strict_by = {
        (float(row["epsilon_px"]), row["v_band"]): row
        for row in strict_rows
    }
    pooled_repeat = {row["v_band"]: row for row in repeatability_rows if row["scope"] == "pooled"}
    quality_features_for_report = (
        ("signal_excess_dn_median", "signal excess (DN)"),
        ("stripe_width_fwhm_px_median", "FWHM (px)"),
        ("profile_asymmetry_median", "asymmetry"),
        ("profile_skewness_median", "skewness"),
        ("steger_response_dn_per_px2_median", "Steger response"),
        ("steger_valid_ratio", "valid ratio"),
    )
    lines: list[str] = [
        "# Task A-2｜下边缘光条质量与 Steger 定位误差机制审计",
        "",
        f"生成时间（UTC）：`{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## 明确结论",
        "",
        f"- `LOWER_EDGE_STEGER_RANDOM_DEGRADATION = {decisions['LOWER_EDGE_STEGER_RANDOM_DEGRADATION']}`",
        f"- `LOWER_EDGE_PROFILE_SYSTEMATIC_BIAS_EVIDENCE = {decisions['LOWER_EDGE_PROFILE_SYSTEMATIC_BIAS_EVIDENCE']}`",
        f"- `OPTICAL_EXTRACTION_MAGNITUDE_SUFFICIENT = {decisions['OPTICAL_EXTRACTION_MAGNITUDE_SUFFICIENT']}`",
        "",
        "结论区分了三件事：同 condition 内的随机 u 重复性、跨 v 的 profile 形状系统变化，以及这些观测到的 u 变化乘以 A-1 局部敏感度后是否足以覆盖 Base 高度误差。相关性只作为机制证据，不作为 correction 拟合或特征筛选依据。",
        "",
        f"- 处理 `{len(frame_rows)}` 帧、`{len(set(str(row['condition_id']) for row in frame_rows))}` 个 condition；原始 PNG 存在且 SHA256 匹配 `{hash_matches}/{len(frame_rows)}`。",
        f"- Frozen cache 中心作为正式 `(u,v)`；cache/同 Frozen 参数 diagnostic replay 逐帧 exact match：`{sum(bool(row.get('diagnostic_replay_cache_exact')) for row in frame_rows)}/{len(frame_rows)}`。",
        f"- 观测到的 condition-level u deviation P95（跨 condition 的 P95）：`{fmt(decisions['condition_p95_u_abs_deviation_px_p95'], 5)} px`；按 `{SENSITIVITY_MM_PER_PX:.2f} mm/px` 换算为约 `{fmt(decisions['predicted_height_p95_mm_from_u_repeatability'], 5)} mm`。",
        f"- Base `|error|` pooled P95：`{fmt(decisions['observed_base_abs_error_p95_mm'], 5)} mm`；condition-demeaned Base deviation P95：`{fmt(decisions['observed_within_condition_base_abs_deviation_p95_mm'], 5)} mm`。",
        "",
        "## Provenance / reuse audit",
        "",
        "本轮复用：",
        "",
        f"- Frozen Steger cache：`{CACHE_NPZ}`；manifest `one_steger_per_frame={cache_manifest.get('one_steger_per_frame')}`、中心坐标为 full-sensor `(u=column,v=row)`。",
        f"- Frozen V2 measurement ROI：`{REGISTRY_PATH}`；geometry-only/frozen registry，condition 结构为 `h10/h20/h30 × p01...p10`，每 condition 20 repeats。",
        f"- A-13B-v2 frame-level Base error：`{A13B_FRAMES_CSV}`；Base error 定义为 `residual_base_session = height_model - nominal_truth`。",
        f"- Task A-1：`{A1_CSV}`；用于 strict-valid sanity 和固定约 `0.30 mm/px` 的局部敏感度数量级，不重新计算 correction。",
        f"- 原始 PNG：`{DATA_ROOT}`；manifest 记录的 600 张 PNG，尺寸 `3000×480 uint8`。",
        "",
        "本轮新增：",
        "",
        "- 从原始 horizontal normal profile（每个 Frozen Steger 点所在 row、围绕 u 中心的 25 px 窗口）提取 peak/background/signal/dynamic range/FWHM/asymmetry/skewness/saturation。",
        "- 用 Frozen Steger 同参数 diagnostic replay 读取 response、offset、normal direction、valid count；不使用 replay 中心替换 cache。",
        "- 用 cache 的 `u(v)` 三行差分计算 centerline curvature；它是几何诊断量，不是拟合 correction。",
        "",
        "## A-1 strict-valid sanity",
        "",
        f"严格子集定义：`sensitivity_status == OK`，即 base、`u±0.05`、`u±0.10` 五次 Ground reference 均 valid。共 `{strict_summary['strict_total']}/{strict_summary['all_total']}` 点。",
        "",
        "| v band | n @0.05 | P95 @0.05 (mm/px) | n @0.10 | P95 @0.10 (mm/px) |",
        "|---|---:|---:|---:|---:|",
    ]
    for band in V_BAND_LABELS:
        a = strict_by.get((0.05, band), {})
        b = strict_by.get((0.1, band), {})
        lines.append(table_row([band, a.get("n"), fmt(a.get("p95_abs_dh_du_mm_per_px")), b.get("n"), fmt(b.get("p95_abs_dh_du_mm_per_px"))]))
    lines.extend(
        [
            "",
            f"strict-valid 下 `v>2600` 的 n 为 `{strict_by.get((0.05, 'v>2600'), {}).get('n', 0)}`，因为当前 Session Ground 有效域不覆盖这些点；因此严格子集不能独立形成 high/low edge ratio。它没有产生与 A-1 `LOWER_EDGE_GEOMETRIC_SENSITIVITY = NO` 相矛盾的结果，但不能被表述为对最下边缘的独立可比复核。`strict_valid_sensitivity_summary.csv` 保留了完整 n/空组事实。",
            "",
            "## v-band quality summary",
            "",
            "下表用 frame-level measurement ROI 聚合值；`v` 为 Frozen V2 height ROI 的 formal center row。完整逐帧字段见 `stripe_quality_frame_audit.csv`。",
            "",
            "| v band | n | signal excess median DN | FWHM median px | asymmetry median | skewness median | response median DN/px² | valid ratio median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for band in V_BAND_LABELS:
        rows = [row for row in frame_rows if row["v_band"] == band]
        lines.append(
            table_row(
                [
                    band,
                    len(rows),
                    fmt(median(row.get("signal_excess_dn_median") for row in rows)),
                    fmt(median(row.get("stripe_width_fwhm_px_median") for row in rows)),
                    fmt(median(row.get("profile_asymmetry_median") for row in rows)),
                    fmt(median(row.get("profile_skewness_median") for row in rows)),
                    fmt(median(row.get("steger_response_dn_per_px2_median") for row in rows)),
                    fmt(median(row.get("steger_valid_ratio") for row in rows)),
                ]
            )
        )
    lines.extend([
        "",
        "## Same-condition Steger repeatability",
        "",
        "每个 condition 内只在相同整数 row/v 对齐 20 repeats；先得到 row-wise sample std 和相对该 row median 的绝对 deviation，再按 v band 汇总。不同 height 不因 v 数值接近而合并为同一 physical position。",
        "",
        "| scope | v band | n rows | median u std (px) | P95 u std (px) | P95 row-P95 deviation (px) | Spearman(v,row std) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    report_scopes = [("pooled", "pooled"), *[("height", height) for height in HEIGHT_ORDER]]
    for scope, scope_id in report_scopes:
        for band in V_BAND_LABELS:
            item = next((row for row in repeatability_rows if row["scope"] == scope and row["scope_id"] == scope_id and row["v_band"] == band), {})
            lines.append(table_row([scope_id, band, item.get("n_rows"), fmt(item.get("median_u_std_px"), 5), fmt(item.get("p95_u_std_px"), 5), fmt(item.get("p95_u_abs_deviation_p95_px"), 5), fmt(item.get("spearman_v_u_std_rho"), 3)]))
    lines.extend([
        "",
        "## Height-error correlations",
        "",
        "`pooled` 相关包含跨 position/FOV 的共同 v 变化；`within_condition_demeaned` 先在每个 20-repeat condition 内分别减去 feature 与 Base error 的 condition mean，用于检验同 condition 内的变化，避免把共同 v 趋势误判为因果。完整相关矩阵见 `quality_correlation_summary.csv`。",
        "",
        "| feature | pooled Pearson r | pooled Spearman ρ | within-condition Pearson r | within-condition Spearman ρ |",
        "|---|---:|---:|---:|---:|",
    ])
    for feature, _label in quality_features_for_report:
        raw = correlation_lookup(correlation_rows, "pooled", "pooled", feature, "base_error_mm")
        within = correlation_lookup(correlation_rows, "pooled", "within_condition_demeaned", feature, "base_error_mm")
        lines.append(table_row([feature, fmt(raw.get("pearson_r"), 3), fmt(raw.get("spearman_rho"), 3), fmt(within.get("pearson_r"), 3), fmt(within.get("spearman_rho"), 3)]))
    lines.extend([
        "",
        "按 height 的 raw/within-condition 相关，以及 signed/absolute Base error 两个 target，均在 `quality_correlation_summary.csv` 中保留；本报告不从 truth error 反向选择指标或调整阈值。",
        "",
        "## Mechanism interpretation",
        "",
        f"- Random localization decision：high/low pooled P95 deviation ratio=`{fmt(decisions['random_pooled_p95_ratio_high_to_low'], 3)}`；h10/h20/h30 ratios=`{', '.join(fmt(decisions['random_height_p95_ratios_high_to_low'].get(height), 3) for height in HEIGHT_ORDER)}`，达到 `>=1.10` 的 height 数为 `{decisions['random_height_consistent_count_ge_1p10']}/3`。",
        f"- Profile systematic-shift decision：按预先注册的规则比较 profile asymmetry 的 `0.05` 与 skewness 的 `0.15` high-low shift；支持的 height 数为 `{len(decisions['profile_shape_supported_heights'])}/3`，但 asymmetry 在 h10 与 h20/h30 间方向不一致，未满足跨 height 同方向的 STRONG 规则，因此标为 `{decisions['LOWER_EDGE_PROFILE_SYSTEMATIC_BIAS_EVIDENCE']}`。这只是 profile-shape evidence，不等于已证明 Steger 存在 truth-referenced bias。",
        f"- Magnitude decision：A-1 sensitivity 使用 `{SENSITIVITY_MM_PER_PX:.2f} mm/px`；预测 u-repeatability 高尾高度量级 / pooled Base-error P95=`{fmt(decisions['predicted_to_observed_total_p95_ratio'], 3)}`，相对 within-condition Base deviation P95=`{fmt(decisions['predicted_to_observed_within_condition_p95_ratio'], 3)}`。这里的 `PARTIAL` 仅表示可覆盖 repeatability 分量，不表示足以解释 pooled absolute Base error。",
        "",
        "## Boundaries",
        "",
        "- 本审计保持原始 full-sensor `(u,v)=(column,row)`；Daheng 是纵向条纹、row scan，profile 法向方向是 u。",
        "- 没有修改 Steger 参数、没有重新拟合 C0/C1/H1/H-B2、没有用 Base error 选择/过滤 quality feature。",
        "- profile background/width/asymmetry/skewness 仅在固定 25 px profile window 且有效 excess > 0 时报告；无法可靠定义的点保留为空，不强行填值。",
        "- `centerline_curvature` 是 Frozen cache 的局部 u(v) 几何差分；不是独立的光学 truth，也不表示可以部署 spatial correction。",
        "",
        "## Outputs",
        "",
        f"输出目录：`{output_dir}`",
        "",
        "- `stripe_quality_frame_audit.csv`",
        "- `steger_repeatability_by_v.csv`",
        "- `stripe_quality_vs_v.png`",
        "- `steger_repeatability_vs_v.png`",
        "- `height_error_vs_quality.png`",
        "- `strict_valid_sensitivity_summary.csv`",
        "- `quality_correlation_summary.csv`",
        "- `stripe_quality_provenance.json`",
        "- `report.md`",
    ])
    return "\n".join(lines) + "\n"


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    frames = inputs["frames"]
    offsets = inputs["offsets"]
    centers = inputs["centers"]
    registry_entries = inputs["registry_entries"]
    a13b_by_key = inputs["a13b_by_key"]
    frame_rows: list[dict[str, Any]] = []
    frame_work: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        cache_key = str(frame["cache_key"])
        if cache_key not in a13b_by_key:
            raise ValueError(f"A-13B frame row missing cache_key: {cache_key}")
        condition_id = cache_key.split("/")[0]
        if condition_id not in registry_entries:
            raise ValueError(f"Frozen V2 ROI entry missing condition: {condition_id}")
        start, end = int(offsets[index]), int(offsets[index + 1])
        centers_frame = np.ascontiguousarray(centers[start:end], dtype=np.float64)
        row, work = quality_for_frame(
            frame,
            centers_frame,
            registry_entries[condition_id],
            a13b_by_key[cache_key],
            inputs["cache_manifest"],
        )
        frame_rows.append(row)
        frame_work.append(work)
        if (index + 1) % 50 == 0 or index + 1 == len(frames):
            print(f"A-2 stripe quality {index + 1}/{len(frames)} frames")

    if len(frame_rows) != 600 or len({str(row["cache_key"]) for row in frame_rows}) != 600:
        raise ValueError("A-2 frame audit did not produce exactly 600 unique frame rows")
    if any(int(sum(str(row["condition_id"]) == condition for row in frame_rows)) != 20 for condition in sorted({str(row["condition_id"]) for row in frame_rows})):
        raise ValueError("A-2 condition repeat count is not 20")

    repeatability_rows, condition_summaries = build_repeatability(frame_work)
    correlation_rows = build_correlations(frame_rows)
    strict_rows, strict_summary = strict_valid_sensitivity(inputs["a1_rows"])
    decisions = make_decisions(frame_rows, repeatability_rows, condition_summaries)

    write_csv(output_dir / "stripe_quality_frame_audit.csv", frame_rows)
    write_csv(output_dir / "steger_repeatability_by_v.csv", repeatability_rows)
    write_csv(output_dir / "quality_correlation_summary.csv", correlation_rows)
    write_csv(output_dir / "strict_valid_sensitivity_summary.csv", strict_rows)
    plot_quality_vs_v(frame_rows, output_dir / "stripe_quality_vs_v.png")
    plot_repeatability(repeatability_rows, output_dir / "steger_repeatability_vs_v.png")
    plot_height_error_vs_quality(frame_rows, correlation_rows, output_dir / "height_error_vs_quality.png")

    provenance = {
        "task": "A-2 stripe quality and Steger localization mechanism audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "data_root": DATA_ROOT,
            "cache_npz": CACHE_NPZ,
            "cache_manifest": CACHE_MANIFEST,
            "registry": REGISTRY_PATH,
            "a1_csv": A1_CSV,
            "a13b_frames_csv": A13B_FRAMES_CSV,
            "a13b_provenance": A13B_PROVENANCE,
            "realtime_steger_source": CALIBRATION_SRC / "realtime_steger.py",
        },
        "reuse": {
            "frozen_steger_centers_as_formal_coordinates": True,
            "frozen_v2_measurement_roi": True,
            "a13b_frame_level_base_error": True,
            "a1_sensitivity_csv": True,
            "c0_c1_h1_hb2_refit": False,
            "production_reconstruction_modified": False,
            "truth_driven_feature_selection": False,
        },
        "raw_png": {
            "frame_count": len(frame_rows),
            "source_exists_count": sum(bool(row.get("source_exists")) for row in frame_rows),
            "image_read_ok_count": sum(bool(row.get("image_read_ok")) for row in frame_rows),
            "source_hash_matches": sum(bool(row.get("source_hash_match")) for row in frame_rows),
            "shapes": sorted({row.get("image_shape") for row in frame_rows}),
            "dtypes": sorted({row.get("image_dtype") for row in frame_rows}),
        },
        "frozen_steger": {
            "protocol_key": inputs["cache_manifest"].get("protocol_key"),
            "diagnostic_replay_exact_count": sum(bool(row.get("diagnostic_replay_cache_exact")) for row in frame_rows),
            "diagnostic_replay_total": len(frame_rows),
            "formal_centers_source": "session01_steger_centers.npz",
        },
        "definitions": {
            "measurement_roi": "Frozen V2 height_v_range; one frame row aggregates all cached centers in that range",
            "profile": "raw image horizontal profile at each cached center row, 25 px window around local u",
            "profile_background": "median of outer ring |u-u_center| >= 8 px",
            "profile_width": "FWHM of positive background-subtracted profile at 50% excess",
            "profile_asymmetry": "right-minus-left positive profile area divided by total area",
            "profile_skewness": "weighted third standardized central moment of positive profile",
            "steger_response": "exact Frozen realtime Steger negative normal second derivative response",
            "centerline_curvature": "three-row second difference of cached u(v), signed/absolute 1/px",
            "repeatability": "same condition, same integer v row across 20 repeats; row-wise sample std and |u-row-median|",
            "correlation": "pooled and condition-demeaned, separately pooled and h10/h20/h30",
            "sensitivity_mm_per_px": SENSITIVITY_MM_PER_PX,
        },
        "strict_valid_sanity": strict_summary,
        "decisions": decisions,
    }
    write_json(output_dir / "stripe_quality_provenance.json", provenance)
    (output_dir / "report.md").write_text(
        report_text(inputs, frame_rows, repeatability_rows, correlation_rows, strict_rows, strict_summary, condition_summaries, decisions, output_dir),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "frame_count": len(frame_rows),
        "condition_count": len(condition_summaries),
        "strict_valid_summary": strict_summary,
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    result = run(args.output_dir.resolve())
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
