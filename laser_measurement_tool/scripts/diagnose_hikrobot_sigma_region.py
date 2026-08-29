#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""海康红光区域化 Steger 尺度适配与条纹形态诊断。

本脚本只读取 400/500 us recording 下的原始 Mono8 PNG、frames.csv、当前配置
以及 A-2 的小型诊断产物。它不读取三维结果，不重新进行三维恢复，也不修改
生产配置、标定文件或生产代码。

固定当前 Steger 的所有参数，只覆盖 sigma:
1.0, 1.2, 1.5, 1.8, 2.0 px。

运行示例（仓库根目录）：

    .venv\\Scripts\\python.exe \\
      laser_measurement_tool\\scripts\\diagnose_hikrobot_sigma_region.py

默认输出目录：
laser_measurement_tool/output_haikang_0828/online_recordings/sigma_region_diagnostic
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402


TOOL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOL_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_RECORDINGS_ROOT = (
    TOOL_ROOT / "output_haikang_0828" / "online_recordings"
)
DEFAULT_CONFIG_PATH = TOOL_ROOT / "configs" / "measure_tool.yaml"
DEFAULT_PROFILE_PATH = WORKSPACE_ROOT / "calibration" / "config" / "realtime_steger.yaml"
DEFAULT_A2_DIR = DEFAULT_RECORDINGS_ROOT / "exposure_diagnostic"
DEFAULT_OUTPUT_DIR = DEFAULT_RECORDINGS_ROOT / "sigma_region_diagnostic"

TARGET_EXPOSURES_US = (400.0, 500.0)
SIGMAS_PX = (1.0, 1.2, 1.5, 1.8, 2.0)
CURRENT_SIGMA_PX = 1.5
EXPECTED_FRAME_COUNT = 20
MIN_FRAME_COUNT = 18
MAX_FRAME_COUNT = 22
PROFILE_BACKGROUND_PERCENTILE = 20.0
SATURATION_DN = 255.0
LOW_FREQUENCY_SIGMA_PX = 32.0
MIN_TEMPORAL_FRAME_RATIO = 0.90
PROFILE_SMOOTHING_SIGMA_PX = 0.5
PROFILE_SAMPLE_SHAPE_WARN_CV = 0.10
SCALE_MISMATCH_GAIN_WARN = 0.10
MORPHOLOGY_ASYMMETRY_WARN = 0.20
MORPHOLOGY_DOUBLE_PEAK_WARN = 0.20
MORPHOLOGY_SHOULDER_WARN = 0.30
MORPHOLOGY_PLATEAU_WARN = 0.20
CENTROID_EXPOSURE_RISE_WARN = 0.15
CENTROID_EXPOSURE_RISE_MIN_PX = 0.10

CAPTURE_KEYS = (
    "exposure_us",
    "gain_db",
    "pixel_format",
    "offset_x",
    "offset_y",
    "width",
    "height",
    "frame_gap",
)
COMMON_CAPTURE_KEYS = tuple(key for key in CAPTURE_KEYS if key != "exposure_us")
REGIONS = ("left", "center", "right")
ALGORITHMS = ("steger", "centroid")
FRAME_PATTERN = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)

if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import AppConfig, load_app_config  # noqa: E402
from laser.backends import centroid_backend, steger_backend  # noqa: E402


class DiagnosticError(RuntimeError):
    """输入、配置或统计协议不满足诊断要求。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return "" if not math.isfinite(number) else f"{number:.10g}"
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DiagnosticError(f"missing CSV: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DiagnosticError(f"CSV has no header: {path}")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise DiagnosticError(f"cannot read CSV: {path}: {error}") from error


def read_image(path: Path) -> np.ndarray:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as error:
        raise DiagnosticError(f"cannot read image: {path}") from error
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DiagnosticError(f"cannot decode image: {path}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise DiagnosticError(
            f"image must be single-channel Mono8: {path}, "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    return np.ascontiguousarray(image)


def canonical_capture_value(key: str, value: Any) -> str:
    if key in {"exposure_us", "gain_db"}:
        parsed = finite_float(value)
        if parsed is None:
            return str(value).strip()
        return f"{parsed:.9g}"
    if key in {"offset_x", "offset_y", "width", "height", "frame_gap"}:
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value).strip()
    return str(value).strip()


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = FRAME_PATTERN.match(path.name)
    return (int(match.group(1)), path.name) if match else (10**12, path.name)


def unique_capture_values(
    rows: Sequence[Mapping[str, str]], key: str
) -> list[str]:
    return sorted(
        {canonical_capture_value(key, row.get(key, "")) for row in rows}
    )


def validate_recording(
    recording_dir: Path,
) -> dict[str, Any]:
    metadata = read_csv(recording_dir / "frames.csv")
    frame_paths = sorted(recording_dir.glob("frame_*.png"), key=frame_sort_key)
    if not (MIN_FRAME_COUNT <= len(frame_paths) <= MAX_FRAME_COUNT):
        raise DiagnosticError(
            f"frame count {len(frame_paths)} is outside "
            f"[{MIN_FRAME_COUNT}, {MAX_FRAME_COUNT}]"
        )
    if len(metadata) != len(frame_paths):
        raise DiagnosticError(
            f"frames.csv rows {len(metadata)} != PNG count {len(frame_paths)}"
        )
    names = [row.get("filename", "") for row in metadata]
    actual_names = [path.name for path in frame_paths]
    if names != actual_names or len(set(names)) != len(names):
        raise DiagnosticError("frames.csv filename order does not match frame PNGs")
    first = metadata[0] if metadata else {}
    for key in CAPTURE_KEYS:
        if key not in first:
            raise DiagnosticError(f"frames.csv missing capture field: {key}")
    unique_values = {
        key: unique_capture_values(metadata, key) for key in CAPTURE_KEYS
    }
    inconsistent = {
        key: values for key, values in unique_values.items() if len(values) != 1
    }
    if inconsistent:
        raise DiagnosticError(
            "capture fields vary within recording: "
            + json.dumps(inconsistent, ensure_ascii=False)
        )
    pixel_format = unique_values["pixel_format"][0]
    if pixel_format.lower() != "mono8":
        raise DiagnosticError(f"pixel_format is not Mono8: {pixel_format}")
    exposure = finite_float(first.get("exposure_us"))
    if exposure is None or exposure <= 0.0:
        raise DiagnosticError("frames.csv exposure_us is invalid")
    return {
        "recording": recording_dir.name,
        "path": recording_dir,
        "frame_paths": frame_paths,
        "metadata": metadata,
        "frame_count": len(frame_paths),
        "exposure_us": exposure,
        "unique_values": unique_values,
        "common_signature": tuple(
            (key, unique_values[key][0]) for key in COMMON_CAPTURE_KEYS
        ),
    }


def discover_target_recordings(
    recordings_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not recordings_root.is_dir():
        raise DiagnosticError(f"recordings root does not exist: {recordings_root}")
    candidates = sorted(
        [
            path
            for path in recordings_root.iterdir()
            if path.is_dir() and path.name.startswith("recording_")
        ],
        key=lambda path: path.name,
    )
    if not candidates:
        raise DiagnosticError(f"no recording_* directory under {recordings_root}")

    selected: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    target_set = set(TARGET_EXPOSURES_US)
    for recording_dir in candidates:
        try:
            item = validate_recording(recording_dir)
            if not any(
                math.isclose(item["exposure_us"], exposure, abs_tol=1.0e-9)
                for exposure in target_set
            ):
                ignored.append(
                    {
                        "recording": recording_dir.name,
                        "reason": "exposure is outside the A-3 target set",
                        "exposure_us": item["exposure_us"],
                    }
                )
                continue
            selected.append(item)
        except (DiagnosticError, OSError, UnicodeError, csv.Error) as error:
            ignored.append({"recording": recording_dir.name, "reason": str(error)})

    for exposure in TARGET_EXPOSURES_US:
        matches = [
            item
            for item in selected
            if math.isclose(item["exposure_us"], exposure, abs_tol=1.0e-9)
        ]
        if len(matches) != 1:
            raise DiagnosticError(
                f"expected exactly one valid recording at {exposure:g} us, "
                f"found {len(matches)}"
            )
    selected.sort(key=lambda item: float(item["exposure_us"]))
    signatures = {item["common_signature"] for item in selected}
    if len(signatures) != 1:
        raise DiagnosticError(
            "400/500 us recordings do not share the same non-exposure "
            f"capture signature: {signatures}"
        )
    return selected, ignored


def load_profile_options(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticError(f"Steger profile does not exist: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise DiagnosticError(f"cannot read Steger profile: {path}") from error
    if not isinstance(document, Mapping):
        raise DiagnosticError("Steger profile root must be a mapping")
    options = document.get("steger", document.get("options", {}))
    if not isinstance(options, Mapping):
        raise DiagnosticError("Steger profile steger/options must be a mapping")
    return dict(options)


def profile_path_from_config(config_path: Path) -> Path | None:
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise DiagnosticError(f"cannot parse config: {config_path}") from error
    extraction = document.get("extraction", {})
    if not isinstance(extraction, Mapping):
        return None
    value = extraction.get("profile")
    if value in (None, ""):
        return None
    raw = Path(str(value))
    return (config_path.parent / raw).resolve() if not raw.is_absolute() else raw.resolve()


def load_fixed_options(
    config_path: Path, external_profile_path: Path
) -> tuple[AppConfig, dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        config: AppConfig = load_app_config(config_path)
    except Exception as error:  # noqa: BLE001 - provide diagnostic context
        raise DiagnosticError(f"cannot load current app config: {config_path}: {error}") from error
    steger_options = dict(
        config.extraction_options_by_method.get("steger", config.extraction_options)
    )
    centroid_options = dict(config.extraction_options_by_method.get("centroid", {}))
    if not steger_options:
        raise DiagnosticError("current config has no steger options")
    configured_sigma = finite_float(steger_options.get("sigma"))
    if configured_sigma is None or not math.isclose(
        configured_sigma, CURRENT_SIGMA_PX, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise DiagnosticError(
            f"current configured sigma is {configured_sigma}, "
            f"expected {CURRENT_SIGMA_PX}"
        )
    external_profile = load_profile_options(external_profile_path)
    compare_keys = (
        "sigma",
        "threshold",
        "deriv_thresh",
        "roi_margin",
        "roi_max_height",
        "scan_axis",
    )
    profile_mismatches = {
        key: {
            "config": steger_options.get(key),
            "external_profile": external_profile.get(key),
        }
        for key in compare_keys
        if key in external_profile
        and key in steger_options
        and external_profile.get(key) != steger_options.get(key)
    }
    if profile_mismatches:
        raise DiagnosticError(
            "external Steger profile differs from current config: "
            + json.dumps(jsonable(profile_mismatches), ensure_ascii=False)
        )
    referenced_profile = profile_path_from_config(config_path)
    return config, steger_options, centroid_options, {
        "external_profile_options": external_profile,
        "config_referenced_profile": referenced_profile,
        "profile_mismatches": profile_mismatches,
    }


def parse_search_roi(
    options: Mapping[str, Any], image_shape: tuple[int, int]
) -> dict[str, int]:
    height, width = image_shape
    value = options.get("search_roi")
    if value is None:
        return {
            "configured_left": 0,
            "configured_top": 0,
            "configured_width": width,
            "configured_height": height,
            "left": 0,
            "top": 0,
            "right": width,
            "bottom": height,
        }
    if not isinstance(value, Mapping):
        raise DiagnosticError("steger.search_roi must be a mapping")
    required = ("offset_x", "offset_y", "width", "height")
    if any(key not in value for key in required):
        raise DiagnosticError("steger.search_roi is missing a required field")
    raw: dict[str, int] = {}
    for key in required:
        try:
            raw[key] = int(value[key])
        except (TypeError, ValueError) as error:
            raise DiagnosticError(f"invalid search_roi.{key}") from error
    if min(raw["offset_x"], raw["offset_y"]) < 0:
        raise DiagnosticError("search_roi offsets must be non-negative")
    if min(raw["width"], raw["height"]) <= 0:
        raise DiagnosticError("search_roi width/height must be positive")
    left = max(0, raw["offset_x"])
    top = max(0, raw["offset_y"])
    right = min(width, raw["offset_x"] + raw["width"])
    bottom = min(height, raw["offset_y"] + raw["height"])
    if right <= left or bottom <= top:
        raise DiagnosticError("search_roi does not intersect the image")
    return {
        "configured_left": raw["offset_x"],
        "configured_top": raw["offset_y"],
        "configured_width": raw["width"],
        "configured_height": raw["height"],
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def region_bounds(region: str, roi_width: int) -> tuple[int, int]:
    first = int(roi_width / 3.0)
    second = int(roi_width * 2.0 / 3.0)
    if region == "left":
        return 0, first
    if region == "center":
        return first, second
    if region == "right":
        return second, roi_width
    raise DiagnosticError(f"unknown region: {region}")


def region_mask(region: str, roi_width: int) -> np.ndarray:
    left, right = region_bounds(region, roi_width)
    mask = np.zeros(roi_width, dtype=bool)
    mask[left:right] = True
    return mask


def representative_columns(roi: Mapping[str, int]) -> dict[str, int]:
    width = int(roi["right"] - roi["left"])
    result: dict[str, int] = {}
    for region in REGIONS:
        left, right = region_bounds(region, width)
        result[region] = int(roi["left"] + left + (right - left - 1) // 2)
    return result


def finite_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def stats_at(stats: Mapping[str, Any], key: str = "median") -> float | None:
    return finite_float(stats.get(key))


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def crossing_position(values: np.ndarray, index_a: int, index_b: int, level: float) -> float:
    value_a = float(values[index_a])
    value_b = float(values[index_b])
    if abs(value_b - value_a) <= np.finfo(float).eps:
        return float(index_a)
    return float(index_a + (level - value_a) / (value_b - value_a))


def level_edges(
    values: np.ndarray, peak_index: int, level: float
) -> tuple[float, float] | None:
    if peak_index <= 0 or peak_index >= len(values) - 1:
        return None
    left = peak_index
    while left > 0 and values[left - 1] >= level:
        left -= 1
    right = peak_index
    while right < len(values) - 1 and values[right + 1] >= level:
        right += 1
    if left == 0 or right == len(values) - 1:
        return None
    return (
        crossing_position(values, left - 1, left, level),
        crossing_position(values, right, right + 1, level),
    )


def profile_shape_metrics(profile: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(profile, dtype=np.float64)
    smooth = gaussian_filter1d(raw, sigma=PROFILE_SMOOTHING_SIGMA_PX, mode="nearest")
    peak_index = int(np.argmax(smooth))
    peak_dn = float(np.max(raw))
    peak_v_offset = int(np.argmax(raw))
    background_dn = float(np.percentile(raw, PROFILE_BACKGROUND_PERCENTILE))
    contrast_dn = float(max(0.0, peak_dn - background_dn))
    signal = np.maximum(smooth - background_dn, 0.0)
    peak_signal = float(np.max(signal))
    normalized = signal / peak_signal if peak_signal > 0.0 else np.zeros_like(signal)

    half_level = background_dn + 0.5 * peak_signal
    half_edges = level_edges(smooth, peak_index, half_level)
    fwhm = (
        float(half_edges[1] - half_edges[0])
        if half_edges is not None and half_edges[1] > half_edges[0]
        else float("nan")
    )
    tenth_edges = level_edges(
        smooth, peak_index, background_dn + 0.1 * peak_signal
    )
    width_10 = (
        float(tenth_edges[1] - tenth_edges[0])
        if tenth_edges is not None and tenth_edges[1] > tenth_edges[0]
        else float("nan")
    )
    if half_edges is None:
        symmetry = float("nan")
        area_symmetry = float("nan")
    else:
        left_half = float(peak_index - half_edges[0])
        right_half = float(half_edges[1] - peak_index)
        symmetry = (
            abs(left_half - right_half) / (left_half + right_half)
            if left_half + right_half > 0.0
            else float("nan")
        )
        left_i = max(0, int(math.floor(half_edges[0])))
        right_i = min(len(signal) - 1, int(math.ceil(half_edges[1])))
        left_area = float(np.sum(signal[left_i : peak_index + 1]))
        right_area = float(np.sum(signal[peak_index : right_i + 1]))
        area_symmetry = (
            abs(left_area - right_area) / (left_area + right_area)
            if left_area + right_area > 0.0
            else float("nan")
        )

    peak_ids, _ = find_peaks(
        normalized,
        distance=2,
        prominence=0.05,
    )
    candidates = [
        int(index)
        for index in peak_ids
        if normalized[index] >= 0.10
    ]
    if peak_index not in candidates and peak_signal > 0.0:
        candidates.append(peak_index)
    candidates.sort(key=lambda index: float(normalized[index]), reverse=True)
    secondary_ratio = float("nan")
    peak_separation = float("nan")
    valley_ratio = float("nan")
    double_peak = False
    shoulder = False
    if len(candidates) >= 2 and peak_signal > 0.0:
        secondary = candidates[1]
        secondary_ratio = float(normalized[secondary])
        peak_separation = float(abs(secondary - peak_index))
        low, high = sorted((secondary, peak_index))
        valley_ratio = float(np.min(normalized[low : high + 1]))
        double_peak = bool(
            secondary_ratio >= 0.25
            and peak_separation >= 2.0
            and valley_ratio <= 0.85 * min(1.0, secondary_ratio)
        )
        shoulder = bool(
            not double_peak
            and secondary_ratio >= 0.10
            and peak_separation <= 12.0
        )

    top_mask = normalized >= 0.90 if peak_signal > 0.0 else np.zeros_like(normalized, dtype=bool)
    plateau_width = 0
    plateau_left = peak_index
    plateau_right = peak_index
    if top_mask[peak_index]:
        left = peak_index
        right = peak_index
        while left > 0 and top_mask[left - 1]:
            left -= 1
        while right < len(top_mask) - 1 and top_mask[right + 1]:
            right += 1
        plateau_left = left
        plateau_right = right
        plateau_width = right - left + 1
    plateau_flatness = float("nan")
    if plateau_width:
        plateau_values = smooth[plateau_left : plateau_right + 1]
        plateau_flatness = float(
            (np.max(plateau_values) - np.min(plateau_values)) / max(peak_signal, 1.0e-9)
        )
    plateau = bool(
        plateau_width >= 3
        and math.isfinite(plateau_flatness)
        and plateau_flatness <= 0.05
    )
    saturation_count = int(np.count_nonzero(raw >= SATURATION_DN))
    profile_saturation_rate = ratio(saturation_count, len(raw))
    shape_class = "single"
    if double_peak:
        shape_class = "double_peak"
    elif shoulder:
        shape_class = "shoulder"
    elif plateau and saturation_count:
        shape_class = "saturated_plateau"
    elif plateau:
        shape_class = "plateau"
    return {
        "peak_dn": peak_dn,
        "peak_v_offset_px": peak_v_offset,
        "background_dn": background_dn,
        "contrast_dn": contrast_dn,
        "fwhm_px": fwhm,
        "width_10pct_px": width_10,
        "symmetry_index": symmetry,
        "area_symmetry_index": area_symmetry,
        "profile_saturation_rate": profile_saturation_rate,
        "peak_saturated": bool(peak_dn >= SATURATION_DN),
        "double_peak": double_peak,
        "shoulder": shoulder,
        "plateau": plateau,
        "plateau_width_px": float(plateau_width),
        "plateau_flatness": plateau_flatness,
        "secondary_peak_ratio": secondary_ratio,
        "peak_separation_px": peak_separation,
        "valley_ratio": valley_ratio,
        "shape_class": shape_class,
    }


def aggregate_profile_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    boolean_fields = ("peak_saturated", "double_peak", "shoulder", "plateau")
    numeric_fields = (
        "peak_dn",
        "background_dn",
        "contrast_dn",
        "fwhm_px",
        "width_10pct_px",
        "symmetry_index",
        "area_symmetry_index",
        "profile_saturation_rate",
        "plateau_width_px",
        "plateau_flatness",
        "secondary_peak_ratio",
        "peak_separation_px",
        "valley_ratio",
    )
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "shape_class_counts": dict(
            sorted(
                {
                    str(name): sum(1 for row in rows if row.get("shape_class") == name)
                    for name in {str(row.get("shape_class", "")) for row in rows}
                }.items()
            )
        ),
    }
    for field in numeric_fields:
        result[field] = finite_stats(
            [float(row[field]) for row in rows if finite_float(row.get(field)) is not None]
        )
    for field in boolean_fields:
        result[f"{field}_fraction"] = (
            float(sum(bool(row.get(field)) for row in rows) / len(rows))
            if rows
            else float("nan")
        )
    fwhm_stats = result["fwhm_px"]
    median_fwhm = stats_at(fwhm_stats)
    fwhm_std = stats_at(fwhm_stats, "std")
    result["fwhm_cv"] = (
        fwhm_std / median_fwhm
        if fwhm_std is not None and median_fwhm is not None and median_fwhm > 0.0
        else float("nan")
    )
    result["morphology_anomaly"] = bool(
        (stats_at(result["symmetry_index"]) or 0.0) > MORPHOLOGY_ASYMMETRY_WARN
        or (stats_at(result["area_symmetry_index"]) or 0.0) > MORPHOLOGY_ASYMMETRY_WARN
        or result["double_peak_fraction"] > MORPHOLOGY_DOUBLE_PEAK_WARN
        or result["shoulder_fraction"] > MORPHOLOGY_SHOULDER_WARN
        or result["plateau_fraction"] > MORPHOLOGY_PLATEAU_WARN
        or (finite_float(result["fwhm_cv"]) or 0.0) > PROFILE_SAMPLE_SHAPE_WARN_CV
    )
    return result


def points_to_vector(points: np.ndarray, image_width: int) -> np.ndarray:
    vector = np.full(image_width, np.nan, dtype=np.float64)
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return vector
    array = array.reshape((-1, 2))
    finite = np.isfinite(array).all(axis=1)
    columns = np.rint(array[:, 0]).astype(np.int64)
    valid = finite & (columns >= 0) & (columns < image_width)
    grouped: dict[int, list[float]] = defaultdict(list)
    for column, value in zip(columns[valid], array[valid, 1], strict=True):
        grouped[int(column)].append(float(value))
    for column, values in grouped.items():
        vector[column] = float(np.median(values))
    return vector


def fill_for_filter(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return np.full(array.shape, np.nan, dtype=np.float64), valid
    support = np.arange(len(array), dtype=np.float64)
    filled = np.interp(support, indices, array[indices])
    return filled, valid


def build_common_geometry_trend(
    matrices: Sequence[np.ndarray], low_frequency_sigma_px: float
) -> tuple[np.ndarray, np.ndarray]:
    if not matrices:
        raise DiagnosticError("no matrices for common geometry trend")
    stacked = np.concatenate(matrices, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pooled = np.nanmedian(stacked, axis=0)
    support_count = np.count_nonzero(np.isfinite(stacked), axis=0)
    filled, valid = fill_for_filter(pooled)
    if not np.any(valid):
        raise DiagnosticError("cannot estimate common geometry trend")
    if np.count_nonzero(valid) == 1:
        filled[:] = pooled[valid][0]
    trend = gaussian_filter1d(
        filled,
        sigma=low_frequency_sigma_px,
        mode="nearest",
        truncate=4.0,
    )
    return trend, support_count


def high_frequency_matrix(
    matrix: np.ndarray,
    common_trend: np.ndarray,
    low_frequency_sigma_px: float,
) -> np.ndarray:
    high_matrix = np.full(matrix.shape, np.nan, dtype=np.float64)
    for frame_index, values in enumerate(matrix):
        valid = np.isfinite(values) & np.isfinite(common_trend)
        indices = np.flatnonzero(valid)
        if len(indices) < 3:
            continue
        residual = values[indices] - common_trend[indices]
        support = np.arange(len(values), dtype=np.float64)
        filled = np.interp(support, indices, residual)
        local_low = gaussian_filter1d(
            filled,
            sigma=low_frequency_sigma_px,
            mode="nearest",
            truncate=4.0,
        )
        high_matrix[frame_index, indices] = residual - local_low[indices]
    return high_matrix


def temporal_metrics(
    matrix: np.ndarray, minimum_frame_ratio: float
) -> dict[str, np.ndarray]:
    frame_count, column_count = matrix.shape
    valid_ratio = np.mean(np.isfinite(matrix), axis=0)
    std = np.full(column_count, np.nan, dtype=np.float64)
    for column in range(column_count):
        values = matrix[:, column]
        values = values[np.isfinite(values)]
        if len(values) >= 2:
            std[column] = float(np.std(values, ddof=0))
    support = np.isfinite(std) & (
        valid_ratio >= minimum_frame_ratio
    )
    return {
        "valid_ratio": valid_ratio,
        "std": std,
        "support": support,
    }


def frame_region_metrics(
    matrix: np.ndarray,
    high_matrix: np.ndarray,
    region: str,
    roi_width: int,
) -> dict[str, Any]:
    mask = region_mask(region, roi_width)
    valid_ratios: list[float] = []
    high_rms: list[float] = []
    high_p95: list[float] = []
    high_max: list[float] = []
    for values, high in zip(matrix, high_matrix, strict=True):
        local_values = values[mask]
        local_high = high[mask]
        finite_values = local_values[np.isfinite(local_values)]
        finite_high = local_high[np.isfinite(local_high)]
        valid_ratios.append(ratio(len(finite_values), int(np.count_nonzero(mask))))
        high_rms.append(
            float(np.sqrt(np.mean(finite_high**2)))
            if len(finite_high)
            else float("nan")
        )
        high_p95.append(
            float(np.percentile(np.abs(finite_high), 95))
            if len(finite_high)
            else float("nan")
        )
        high_max.append(
            float(np.max(np.abs(finite_high)))
            if len(finite_high)
            else float("nan")
        )
    temporal = temporal_metrics(matrix, MIN_TEMPORAL_FRAME_RATIO)
    temporal_std = temporal["std"][mask & temporal["support"]]
    support_ratios = temporal["valid_ratio"][mask]
    return {
        "valid_ratio": finite_stats(valid_ratios),
        "high_frequency_rms_px": finite_stats(high_rms),
        "high_frequency_abs_p95_px": finite_stats(high_p95),
        "high_frequency_abs_max_px": finite_stats(high_max),
        "temporal_std_px": finite_stats(temporal_std),
        "temporal_valid_ratio": finite_stats(support_ratios),
        "temporal_supported_column_count": int(
            np.count_nonzero(mask & temporal["support"])
        ),
        "temporal_supported_column_ratio": ratio(
            int(np.count_nonzero(mask & temporal["support"])),
            int(np.count_nonzero(mask)),
        ),
    }


def delta_region_metrics(
    delta_matrix: np.ndarray, region: str, roi_width: int
) -> dict[str, Any]:
    mask = region_mask(region, roi_width)
    values = delta_matrix[:, mask]
    paired = values[np.isfinite(values)]
    temporal = temporal_metrics(delta_matrix, MIN_TEMPORAL_FRAME_RATIO)
    temporal_std = temporal["std"][mask & temporal["support"]]
    return {
        "paired_ratio": ratio(
            int(len(paired)),
            int(delta_matrix.shape[0] * np.count_nonzero(mask)),
        ),
        "delta_signed_px": finite_stats(paired),
        "delta_abs_px": finite_stats(np.abs(paired)),
        "delta_temporal_std_px": finite_stats(temporal_std),
        "delta_temporal_supported_column_count": int(
            np.count_nonzero(mask & temporal["support"])
        ),
        "delta_temporal_supported_column_ratio": ratio(
            int(np.count_nonzero(mask & temporal["support"])),
            int(np.count_nonzero(mask)),
        ),
    }


def profile_rows_for_result(
    result: Mapping[str, Any], region: str
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in result["profile_rows"]
        if row["region"] == region
    ]


def build_sigma_region_rows(
    results: Sequence[dict[str, Any]],
    common_trend: np.ndarray,
    common_support: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        roi = result["roi"]
        roi_width = int(roi["right"] - roi["left"])
        exposure = float(result["exposure_us"])
        for sigma in SIGMAS_PX:
            sigma_key = f"{sigma:.6g}"
            steger_matrix = result["vectors"]["steger"][sigma_key]
            centroid_matrix = result["vectors"]["centroid"]
            steger_high = high_frequency_matrix(
                steger_matrix, common_trend, LOW_FREQUENCY_SIGMA_PX
            )
            centroid_high = high_frequency_matrix(
                centroid_matrix, common_trend, LOW_FREQUENCY_SIGMA_PX
            )
            delta_matrix = steger_matrix - centroid_matrix
            result.setdefault("derived", {})[sigma_key] = {
                "steger": {},
                "centroid": {},
                "delta": {},
            }
            for region in REGIONS:
                profile = result["profile_summary"][region]
                steger = frame_region_metrics(
                    steger_matrix, steger_high, region, roi_width
                )
                centroid = frame_region_metrics(
                    centroid_matrix, centroid_high, region, roi_width
                )
                delta = delta_region_metrics(delta_matrix, region, roi_width)
                local_left, local_right = region_bounds(region, roi_width)
                support_values = common_support[local_left:local_right]
                support_ratio = finite_stats(
                    support_values
                    / max(
                        sum(
                            int(item["frame_count"])
                            for item in results
                        )
                        * 2,
                        1,
                    )
                )
                row = {
                    "recording": result["recording"],
                    "exposure_us": exposure,
                    "frame_count": result["frame_count"],
                    "sigma_px": float(sigma),
                    "region": region,
                    "representative_u_px": result["representative_columns"][region],
                    "roi_left_px": roi["left"],
                    "roi_top_px": roi["top"],
                    "roi_width_px": roi_width,
                    "roi_height_px": roi["bottom"] - roi["top"],
                    "peak_dn_p50": stats_at(profile["peak_dn"]),
                    "contrast_dn_p50": stats_at(profile["contrast_dn"]),
                    "fwhm_px_p50": stats_at(profile["fwhm_px"]),
                    "profile_fwhm_cv": profile["fwhm_cv"],
                    "profile_symmetry_p50": stats_at(profile["symmetry_index"]),
                    "profile_double_peak_fraction": profile["double_peak_fraction"],
                    "profile_shoulder_fraction": profile["shoulder_fraction"],
                    "profile_plateau_fraction": profile["plateau_fraction"],
                    "profile_peak_saturated_fraction": profile[
                        "peak_saturated_fraction"
                    ],
                    "steger_valid_ratio_p50": stats_at(steger["valid_ratio"]),
                    "steger_valid_ratio_p05": stats_at(steger["valid_ratio"], "p05"),
                    "steger_valid_ratio_p95": stats_at(steger["valid_ratio"], "p95"),
                    "centroid_valid_ratio_p50": stats_at(centroid["valid_ratio"]),
                    "centroid_valid_ratio_p05": stats_at(
                        centroid["valid_ratio"], "p05"
                    ),
                    "centroid_valid_ratio_p95": stats_at(
                        centroid["valid_ratio"], "p95"
                    ),
                    "steger_high_frequency_rms_p50_px": stats_at(
                        steger["high_frequency_rms_px"]
                    ),
                    "steger_high_frequency_rms_p95_px": stats_at(
                        steger["high_frequency_rms_px"], "p95"
                    ),
                    "steger_high_frequency_abs_p95_p50_px": stats_at(
                        steger["high_frequency_abs_p95_px"]
                    ),
                    "steger_high_frequency_abs_p95_p95_px": stats_at(
                        steger["high_frequency_abs_p95_px"], "p95"
                    ),
                    "centroid_high_frequency_rms_p50_px": stats_at(
                        centroid["high_frequency_rms_px"]
                    ),
                    "centroid_high_frequency_rms_p95_px": stats_at(
                        centroid["high_frequency_rms_px"], "p95"
                    ),
                    "centroid_high_frequency_abs_p95_p50_px": stats_at(
                        centroid["high_frequency_abs_p95_px"]
                    ),
                    "centroid_high_frequency_abs_p95_p95_px": stats_at(
                        centroid["high_frequency_abs_p95_px"], "p95"
                    ),
                    "steger_temporal_std_p50_px": stats_at(
                        steger["temporal_std_px"]
                    ),
                    "steger_temporal_std_p95_px": stats_at(
                        steger["temporal_std_px"], "p95"
                    ),
                    "steger_temporal_supported_column_ratio": steger[
                        "temporal_supported_column_ratio"
                    ],
                    "centroid_temporal_std_p50_px": stats_at(
                        centroid["temporal_std_px"]
                    ),
                    "centroid_temporal_std_p95_px": stats_at(
                        centroid["temporal_std_px"], "p95"
                    ),
                    "centroid_temporal_supported_column_ratio": centroid[
                        "temporal_supported_column_ratio"
                    ],
                    "delta_signed_p50_px": stats_at(delta["delta_signed_px"]),
                    "delta_abs_p50_px": stats_at(delta["delta_abs_px"]),
                    "delta_abs_p95_px": stats_at(delta["delta_abs_px"], "p95"),
                    "delta_pair_valid_ratio": delta["paired_ratio"],
                    "delta_temporal_std_p50_px": stats_at(
                        delta["delta_temporal_std_px"]
                    ),
                    "delta_temporal_std_p95_px": stats_at(
                        delta["delta_temporal_std_px"], "p95"
                    ),
                    "delta_temporal_supported_column_ratio": delta[
                        "delta_temporal_supported_column_ratio"
                    ],
                    "common_geometry_support_ratio_p50": stats_at(support_ratio),
                }
                rows.append(row)
                result["derived"][sigma_key]["steger"][region] = steger
                result["derived"][sigma_key]["centroid"][region] = centroid
                result["derived"][sigma_key]["delta"][region] = delta
    return rows


def rows_for(
    rows: Sequence[Mapping[str, Any]],
    exposure: float | None = None,
    sigma: float | None = None,
    region: str | None = None,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        if exposure is not None and not math.isclose(
            float(row["exposure_us"]), exposure, abs_tol=1.0e-9
        ):
            continue
        if sigma is not None and not math.isclose(
            float(row["sigma_px"]), sigma, abs_tol=1.0e-9
        ):
            continue
        if region is not None and row["region"] != region:
            continue
        selected.append(row)
    return selected


def choose_sigma_recommendation(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_region: dict[str, Any] = {}
    for region in REGIONS:
        per_exposure: dict[str, Any] = {}
        for exposure in TARGET_EXPOSURES_US:
            candidates = rows_for(rows, exposure, region=region)
            best = min(
                candidates,
                key=lambda row: float(row["steger_high_frequency_rms_p50_px"])
                if finite_float(row["steger_high_frequency_rms_p50_px"]) is not None
                else float("inf"),
            )
            max_valid = max(
                float(row["steger_valid_ratio_p50"])
                for row in candidates
                if finite_float(row["steger_valid_ratio_p50"]) is not None
            )
            per_exposure[f"{exposure:g}"] = {
                "best_sigma_px": best["sigma_px"],
                "best_high_frequency_rms_p50_px": best[
                    "steger_high_frequency_rms_p50_px"
                ],
                "current_sigma_high_frequency_rms_p50_px": next(
                    row["steger_high_frequency_rms_p50_px"]
                    for row in candidates
                    if math.isclose(
                        float(row["sigma_px"]),
                        CURRENT_SIGMA_PX,
                        abs_tol=1.0e-9,
                    )
                ),
                "max_valid_ratio_p50": max_valid,
            }
        acceptable: list[float] = []
        for sigma in SIGMAS_PX:
            valid = True
            for exposure in TARGET_EXPOSURES_US:
                row = rows_for(rows, exposure, sigma, region=region)[0]
                best_info = per_exposure[f"{exposure:g}"]
                best_hf = float(best_info["best_high_frequency_rms_p50_px"])
                hf = finite_float(row["steger_high_frequency_rms_p50_px"])
                valid_ratio = finite_float(row["steger_valid_ratio_p50"])
                if (
                    hf is None
                    or hf > best_hf * 1.10
                    or valid_ratio is None
                    or valid_ratio < float(best_info["max_valid_ratio_p50"]) - 0.05
                ):
                    valid = False
            if valid:
                acceptable.append(float(sigma))
        score_by_sigma: dict[str, float] = {}
        for sigma in SIGMAS_PX:
            sigma_rows = rows_for(rows, sigma=sigma, region=region)
            normalized: list[float] = []
            for exposure in TARGET_EXPOSURES_US:
                candidates = rows_for(rows, exposure, region=region)
                best_hf = min(
                    float(row["steger_high_frequency_rms_p50_px"])
                    for row in candidates
                    if finite_float(row["steger_high_frequency_rms_p50_px"]) is not None
                )
                current = next(
                    row
                    for row in sigma_rows
                    if math.isclose(
                        float(row["exposure_us"]), exposure, abs_tol=1.0e-9
                    )
                )
                normalized.append(
                    float(current["steger_high_frequency_rms_p50_px"]) / max(best_hf, 1.0e-9)
                )
            score_by_sigma[f"{sigma:g}"] = float(np.mean(normalized))
        best_sigma = min(score_by_sigma, key=score_by_sigma.get)
        by_region[region] = {
            "per_exposure": per_exposure,
            "acceptable_sigma_values_px": acceptable,
            "acceptable_sigma_range_px": (
                [min(acceptable), max(acceptable)] if acceptable else None
            ),
            "best_robust_sigma_px": float(best_sigma),
            "current_sigma_in_acceptable_range": any(
                math.isclose(value, CURRENT_SIGMA_PX, abs_tol=1.0e-9)
                for value in acceptable
            ),
            "normalized_score_by_sigma": score_by_sigma,
            "candidate_only": True,
            "used_for_production": False,
        }

    edge_scores: dict[str, float] = {}
    for sigma in SIGMAS_PX:
        normalized: list[float] = []
        for region in ("left", "right"):
            for exposure in TARGET_EXPOSURES_US:
                candidates = rows_for(rows, exposure, region=region)
                best_hf = min(
                    float(row["steger_high_frequency_rms_p50_px"])
                    for row in candidates
                    if finite_float(row["steger_high_frequency_rms_p50_px"]) is not None
                )
                current = rows_for(rows, exposure, sigma, region)[0]
                normalized.append(
                    float(current["steger_high_frequency_rms_p50_px"]) / max(best_hf, 1.0e-9)
                )
        edge_scores[f"{sigma:g}"] = float(np.mean(normalized))
    edge_best = min(edge_scores, key=edge_scores.get)
    return {
        "by_region": by_region,
        "edge_robust_candidate": {
            "best_sigma_px": float(edge_best),
            "normalized_score_by_sigma": edge_scores,
            "candidate_only": True,
            "used_for_production": False,
        },
    }


def build_right_exposure_comparison(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for sigma in SIGMAS_PX:
        at_400 = rows_for(rows, TARGET_EXPOSURES_US[0], sigma, "right")[0]
        at_500 = rows_for(rows, TARGET_EXPOSURES_US[1], sigma, "right")[0]
        c400 = finite_float(at_400["centroid_high_frequency_rms_p50_px"])
        c500 = finite_float(at_500["centroid_high_frequency_rms_p50_px"])
        comparison[f"{sigma:g}"] = {
            "sigma_px": sigma,
            "steger_400_high_frequency_rms_p50_px": at_400[
                "steger_high_frequency_rms_p50_px"
            ],
            "steger_500_high_frequency_rms_p50_px": at_500[
                "steger_high_frequency_rms_p50_px"
            ],
            "centroid_400_high_frequency_rms_p50_px": c400,
            "centroid_500_high_frequency_rms_p50_px": c500,
            "centroid_difference_500_minus_400_px": (
                c500 - c400 if c400 is not None and c500 is not None else None
            ),
            "centroid_ratio_500_over_400": (
                c500 / c400 if c400 and c500 is not None else None
            ),
            "steger_400_valid_ratio_p50": at_400["steger_valid_ratio_p50"],
            "steger_500_valid_ratio_p50": at_500["steger_valid_ratio_p50"],
        }
    current = comparison[f"{CURRENT_SIGMA_PX:g}"]
    persistent = all(
        item["centroid_difference_500_minus_400_px"] is not None
        and item["centroid_difference_500_minus_400_px"] >= CENTROID_EXPOSURE_RISE_MIN_PX
        and item["centroid_ratio_500_over_400"] is not None
        and item["centroid_ratio_500_over_400"] >= 1.0 + CENTROID_EXPOSURE_RISE_WARN
        for item in comparison.values()
    )
    return {
        "by_sigma": comparison,
        "current_sigma": current,
        "centroid_rise_persists_across_sigma": persistent,
    }


def build_regional_diagnosis(
    rows: Sequence[Mapping[str, Any]],
    profile_summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    diagnosis: dict[str, Any] = {}
    for region in REGIONS:
        gains: list[dict[str, Any]] = []
        for exposure in TARGET_EXPOSURES_US:
            current = rows_for(rows, exposure, CURRENT_SIGMA_PX, region=region)[0]
            current_hf = finite_float(current["steger_high_frequency_rms_p50_px"])
            candidates = rows_for(rows, exposure, region=region)
            best = min(
                candidates,
                key=lambda row: float(row["steger_high_frequency_rms_p50_px"])
                if finite_float(row["steger_high_frequency_rms_p50_px"]) is not None
                else float("inf"),
            )
            best_hf = finite_float(best["steger_high_frequency_rms_p50_px"])
            gain = (
                (current_hf - best_hf) / current_hf
                if current_hf is not None and best_hf is not None and current_hf > 0.0
                else None
            )
            gains.append(
                {
                    "exposure_us": exposure,
                    "current_sigma_hf_rms_p50_px": current_hf,
                    "best_sigma_px": best["sigma_px"],
                    "best_hf_rms_p50_px": best_hf,
                    "relative_gain_current_to_best": gain,
                    "best_valid_ratio_p50": best["steger_valid_ratio_p50"],
                    "current_valid_ratio_p50": current["steger_valid_ratio_p50"],
                    "valid_ratio_change_best_minus_current": (
                        float(best["steger_valid_ratio_p50"])
                        - float(current["steger_valid_ratio_p50"])
                    ),
                }
            )
        edge_scale_evidence = bool(
            len(gains) == 2
            and np.mean(
                [
                    float(item["relative_gain_current_to_best"])
                    for item in gains
                    if item["relative_gain_current_to_best"] is not None
                ]
            )
            >= SCALE_MISMATCH_GAIN_WARN
            and all(
                float(item["valid_ratio_change_best_minus_current"]) >= -0.05
                for item in gains
            )
            and any(
                not math.isclose(
                    float(item["best_sigma_px"]),
                    CURRENT_SIGMA_PX,
                    abs_tol=1.0e-9,
                )
                for item in gains
            )
        )
        centroid_400 = rows_for(rows, TARGET_EXPOSURES_US[0], CURRENT_SIGMA_PX, region)[0]
        centroid_500 = rows_for(rows, TARGET_EXPOSURES_US[1], CURRENT_SIGMA_PX, region)[0]
        c400 = finite_float(centroid_400["centroid_high_frequency_rms_p50_px"])
        c500 = finite_float(centroid_500["centroid_high_frequency_rms_p50_px"])
        c_difference = (
            c500 - c400 if c400 is not None and c500 is not None else None
        )
        c_ratio = c500 / c400 if c400 and c500 is not None else None
        centroid_exposure_persistence = bool(
            c_difference is not None
            and c_difference >= CENTROID_EXPOSURE_RISE_MIN_PX
            and c_ratio is not None
            and c_ratio >= 1.0 + CENTROID_EXPOSURE_RISE_WARN
        )
        profile_items = [
            profile_summary[f"{exposure:g}"][region]
            for exposure in TARGET_EXPOSURES_US
        ]
        raw_morphology_evidence = any(
            bool(item["morphology_anomaly"]) for item in profile_items
        )
        diagnosis[region] = {
            "sigma_scale_evidence": edge_scale_evidence,
            "scale_gain_by_exposure": gains,
            "raw_profile_morphology_evidence": raw_morphology_evidence,
            "centroid_400_high_frequency_rms_p50_px": c400,
            "centroid_500_high_frequency_rms_p50_px": c500,
            "centroid_500_minus_400_px": c_difference,
            "centroid_500_over_400": c_ratio,
            "centroid_exposure_persistence": centroid_exposure_persistence,
            "morphology_evidence": bool(
                raw_morphology_evidence or centroid_exposure_persistence
            ),
            "dominant_evidence": (
                "both"
                if edge_scale_evidence
                and (raw_morphology_evidence or centroid_exposure_persistence)
                else "sigma_scale"
                if edge_scale_evidence
                else "stripe_morphology"
                if (raw_morphology_evidence or centroid_exposure_persistence)
                else "none"
            ),
        }
    edge_scale = any(
        diagnosis[region]["sigma_scale_evidence"] for region in ("left", "right")
    )
    edge_morphology = any(
        diagnosis[region]["morphology_evidence"] for region in ("left", "right")
    )
    if edge_scale and edge_morphology:
        classification = "MIXED"
    elif edge_scale:
        classification = "STEGER_SCALE_MISMATCH"
    elif edge_morphology:
        classification = "STRIPE_MORPHOLOGY_LIMITED"
    else:
        classification = "INCONCLUSIVE"
    return {
        "classification": classification,
        "by_region": diagnosis,
        "edge_scale_evidence": edge_scale,
        "edge_morphology_evidence": edge_morphology,
        "thresholds": {
            "scale_relative_gain_warn": SCALE_MISMATCH_GAIN_WARN,
            "raw_fwhm_cv_warn": PROFILE_SAMPLE_SHAPE_WARN_CV,
            "raw_asymmetry_warn": MORPHOLOGY_ASYMMETRY_WARN,
            "double_peak_fraction_warn": MORPHOLOGY_DOUBLE_PEAK_WARN,
            "shoulder_fraction_warn": MORPHOLOGY_SHOULDER_WARN,
            "plateau_fraction_warn": MORPHOLOGY_PLATEAU_WARN,
            "centroid_exposure_rise_warn": CENTROID_EXPOSURE_RISE_WARN,
            "centroid_exposure_rise_min_px": CENTROID_EXPOSURE_RISE_MIN_PX,
        },
    }


def profile_change_summary(
    profile_summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for region in REGIONS:
        first = profile_summary[f"{TARGET_EXPOSURES_US[0]:g}"][region]
        second = profile_summary[f"{TARGET_EXPOSURES_US[1]:g}"][region]
        fields = (
            "peak_dn",
            "contrast_dn",
            "fwhm_px",
            "symmetry_index",
            "profile_saturation_rate",
        )
        result[region] = {}
        for field in fields:
            a = stats_at(first[field])
            b = stats_at(second[field])
            result[region][field] = {
                "400_us": a,
                "500_us": b,
                "difference_500_minus_400": b - a if a is not None and b is not None else None,
                "relative_change": (
                    (b - a) / a if a is not None and b is not None and a != 0.0 else None
                ),
            }
    return result


def audit_a2_artifacts(
    a2_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    config_path: Path,
    profile_path: Path,
) -> dict[str, Any]:
    summary_path = a2_dir / "exposure_diagnostic_summary.json"
    region_path = a2_dir / "exposure_region_summary.csv"
    spatial_path = a2_dir / "exposure_spatial_structure.csv"
    required = (summary_path, region_path, spatial_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise DiagnosticError(
            "A-2 artifacts required for provenance audit are missing: "
            + ", ".join(missing)
        )
    try:
        a2_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"cannot parse A-2 summary: {summary_path}") from error
    if not isinstance(a2_summary, Mapping):
        raise DiagnosticError("A-2 summary root is not a mapping")
    region_rows = read_csv(region_path)
    spatial_rows = read_csv(spatial_path)
    target_names = {str(item["recording"]) for item in selected}
    a2_included = a2_summary.get("input", {}).get("included_recordings", [])
    a2_names = {
        str(item.get("recording"))
        for item in a2_included
        if isinstance(item, Mapping)
        and finite_float(item.get("exposure_us")) in TARGET_EXPOSURES_US
    }
    selected_exposures = {float(item["exposure_us"]) for item in selected}
    a2_region_target = [
        row
        for row in region_rows
        if finite_float(row.get("exposure_us")) in selected_exposures
    ]
    a2_spatial_target = [
        row
        for row in spatial_rows
        if finite_float(row.get("exposure_us")) in selected_exposures
    ]
    a2_sigma = finite_float(
        a2_summary.get("parameters", {})
        .get("steger_options_fixed", {})
        .get("sigma")
    )
    issues: list[str] = []
    if target_names != a2_names:
        issues.append(
            f"A-2 target recordings differ: current={sorted(target_names)}, "
            f"A-2={sorted(a2_names)}"
        )
    if len(a2_region_target) != 6:
        issues.append(
            f"A-2 target region rows expected 6, found {len(a2_region_target)}"
        )
    if not a2_spatial_target:
        issues.append("A-2 target spatial_structure rows are empty")
    if a2_sigma is None or not math.isclose(
        a2_sigma, CURRENT_SIGMA_PX, abs_tol=1.0e-9
    ):
        issues.append(f"A-2 current sigma is {a2_sigma}, expected {CURRENT_SIGMA_PX}")
    if a2_summary.get("provenance", {}).get("three_dimensional_results_used") is not False:
        issues.append("A-2 provenance does not explicitly confirm no 3D result use")
    current_config_sha = sha256_file(config_path)
    a2_config_sha = a2_summary.get("provenance", {}).get("config_sha256")
    current_profile_sha = sha256_file(profile_path)
    a2_profile_sha = a2_summary.get("provenance", {}).get("profile_sha256")
    return {
        "directory": a2_dir,
        "artifacts": {
            "summary_json": summary_path,
            "region_summary_csv": region_path,
            "spatial_structure_csv": spatial_path,
        },
        "artifact_sha256": {
            "summary_json": sha256_file(summary_path),
            "region_summary_csv": sha256_file(region_path),
            "spatial_structure_csv": sha256_file(spatial_path),
        },
        "target_recordings_current": sorted(target_names),
        "target_recordings_in_a2": sorted(a2_names),
        "target_region_row_count": len(a2_region_target),
        "target_spatial_row_count": len(a2_spatial_target),
        "a2_current_sigma_px": a2_sigma,
        "a2_config_sha256": a2_config_sha,
        "current_config_sha256_at_audit": current_config_sha,
        "config_hash_match": a2_config_sha == current_config_sha,
        "a2_profile_sha256": a2_profile_sha,
        "current_profile_sha256_at_audit": current_profile_sha,
        "profile_hash_match": a2_profile_sha == current_profile_sha,
        "compatible_for_protocol_cross_check": not issues,
        "issues": issues,
        "numeric_statistics_reused": False,
        "reused_for": [
            "验证 400/500 recording 已纳入 A-2",
            "交叉核对帧数、ROI、采集签名和固定 sigma 协议",
            "记录 A-2 结果文件 provenance 与 hash",
        ],
        "new_calculations": [
            "固定代表列原始纵向 profile 与形态指标",
            "五组 sigma 的 Steger 中心提取",
            "A-3 共同低频趋势后的空间高频 RMS/P95",
            "A-3 temporal std、Steger-centroid delta 和分类",
        ],
    }


def fmt(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(item) for item in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def save_profiles_plot(
    path: Path,
    results: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 9.0), sharex=False)
    axes = np.asarray(axes)
    for column, region in enumerate(REGIONS):
        top_axis = axes[0, column]
        zoom_axis = axes[1, column]
        all_peaks: list[float] = []
        for result in results:
            profiles = np.asarray(
                result["representative_profiles"][region], dtype=np.float64
            )
            if profiles.ndim != 2 or len(profiles) == 0:
                continue
            vertical = np.arange(
                int(result["roi"]["top"]), int(result["roi"]["bottom"])
            )
            median = np.median(profiles, axis=0)
            low = np.percentile(profiles, 10, axis=0)
            high = np.percentile(profiles, 90, axis=0)
            peak_v = float(vertical[int(np.argmax(median))])
            all_peaks.append(peak_v)
            label = f"{float(result['exposure_us']):g} us"
            (line,) = top_axis.plot(median, vertical, linewidth=1.5, label=label)
            top_axis.fill_betweenx(
                vertical, low, high, color=line.get_color(), alpha=0.10
            )
            zoom_axis.plot(median, vertical, linewidth=1.5, label=label)
            zoom_axis.fill_betweenx(
                vertical, low, high, color=line.get_color(), alpha=0.10
            )
        top_axis.set_title(
            f"{region}: representative u={results[0]['representative_columns'][region]} px"
        )
        top_axis.set_ylabel("v (px)")
        top_axis.set_xlabel("raw DN")
        top_axis.set_xlim(0, 260)
        top_axis.set_ylim(
            float(results[0]["roi"]["bottom"]),
            float(results[0]["roi"]["top"]),
        )
        top_axis.grid(True, alpha=0.25)
        top_axis.legend(fontsize=8)
        zoom_axis.set_title("ridge shape zoom")
        zoom_axis.set_xlabel("raw DN")
        zoom_axis.set_ylabel("v (px)")
        zoom_axis.set_xlim(0, 260)
        if all_peaks:
            center = float(np.median(all_peaks))
            zoom_axis.set_ylim(center + 15.0, center - 15.0)
        zoom_axis.grid(True, alpha=0.25)
        zoom_axis.legend(fontsize=8)
    figure.suptitle(
        "Hikrobot red raw vertical profiles: 400 vs 500 us; "
        "shading=P10-P90 across 20 frames"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_sigma_metrics_plot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(18.0, 12.0), sharex=True)
    metric_titles = (
        "local high-frequency RMS (px)",
        "valid ratio",
        "temporal std (px)",
        "|Steger-centroid| P95 (px)",
    )
    for row_index, region in enumerate(REGIONS):
        for column_index, metric_title in enumerate(metric_titles):
            axis = axes[row_index, column_index]
            for exposure, color in zip(
                TARGET_EXPOSURES_US, ("tab:blue", "tab:orange"), strict=True
            ):
                selected = sorted(
                    rows_for(rows, exposure, region=region),
                    key=lambda row: float(row["sigma_px"]),
                )
                x = [float(item["sigma_px"]) for item in selected]
                if metric_title.startswith("local"):
                    steger_y = [
                        item["steger_high_frequency_rms_p50_px"]
                        for item in selected
                    ]
                    centroid_y = [
                        item["centroid_high_frequency_rms_p50_px"]
                        for item in selected
                    ]
                elif metric_title == "valid ratio":
                    steger_y = [item["steger_valid_ratio_p50"] for item in selected]
                    centroid_y = [
                        item["centroid_valid_ratio_p50"] for item in selected
                    ]
                elif metric_title.startswith("temporal"):
                    steger_y = [
                        item["steger_temporal_std_p50_px"] for item in selected
                    ]
                    centroid_y = [
                        item["centroid_temporal_std_p50_px"] for item in selected
                    ]
                else:
                    steger_y = [item["delta_abs_p95_px"] for item in selected]
                    centroid_y = None
                axis.plot(
                    x,
                    steger_y,
                    color=color,
                    marker="o",
                    linewidth=1.5,
                    label=f"Steger {exposure:g} us",
                )
                if centroid_y is not None:
                    axis.plot(
                        x,
                        centroid_y,
                        color=color,
                        marker="x",
                        linestyle="--",
                        linewidth=1.0,
                        label=f"centroid {exposure:g} us",
                    )
            axis.set_title(f"{region}: {metric_title}")
            axis.set_xlabel("sigma (px)")
            axis.set_ylabel(metric_title)
            axis.grid(True, alpha=0.25)
            if metric_title == "valid ratio":
                axis.set_ylim(0.0, 1.05)
            axis.legend(fontsize=7, loc="best")
    figure.suptitle(
        "A-3 sigma × region metrics; solid/circle=Steger, "
        "dashed/x=centroid where applicable"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def profile_report_rows(
    profile_summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for exposure in TARGET_EXPOSURES_US:
        for region in REGIONS:
            item = profile_summary[f"{exposure:g}"][region]
            rows.append(
                [
                    f"{exposure:g}",
                    region,
                    item["representative_u_px"],
                    fmt(stats_at(item["peak_dn"])),
                    fmt(stats_at(item["contrast_dn"])),
                    fmt(stats_at(item["fwhm_px"])),
                    fmt(stats_at(item["symmetry_index"])),
                    f"{100 * item['double_peak_fraction']:.1f}%",
                    f"{100 * item['shoulder_fraction']:.1f}%",
                    f"{100 * item['plateau_fraction']:.1f}%",
                    f"{100 * item['peak_saturated_fraction']:.1f}%",
                    fmt(item["fwhm_cv"], 3),
                ]
            )
    return rows


def current_sigma_report_rows(
    rows: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any],
) -> list[list[Any]]:
    output: list[list[Any]] = []
    for region in REGIONS:
        region_rec = recommendation["by_region"][region]
        for exposure in TARGET_EXPOSURES_US:
            item = rows_for(rows, exposure, CURRENT_SIGMA_PX, region)[0]
            output.append(
                [
                    f"{exposure:g}",
                    region,
                    fmt(item["steger_valid_ratio_p50"] * 100, 1),
                    fmt(item["steger_high_frequency_rms_p50_px"]),
                    fmt(item["steger_high_frequency_abs_p95_p50_px"]),
                    fmt(item["centroid_high_frequency_rms_p50_px"]),
                    fmt(item["centroid_temporal_std_p50_px"]),
                    fmt(item["delta_abs_p50_px"]),
                    fmt(item["delta_abs_p95_px"]),
                    fmt(
                        region_rec["per_exposure"][f"{exposure:g}"][
                            "best_sigma_px"
                        ],
                        1,
                    ),
                ]
            )
    return output


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    classification = summary["classification"]
    input_info = summary["input"]
    profile_summary = summary["profiles"]["by_exposure_region"]
    recommendation = summary["recommendation"]
    diagnosis = summary["diagnosis"]
    right = summary["right_exposure_comparison"]
    current_sigma_rows = current_sigma_report_rows(rows, recommendation)
    profile_rows = profile_report_rows(profile_summary)
    selected_names = ", ".join(
        f"{item['recording']} ({float(item['exposure_us']):g} us)"
        for item in input_info["selected_recordings"]
    )
    right_current = right["current_sigma"]
    right_ratio = right_current["centroid_ratio_500_over_400"]
    right_difference = right_current["centroid_difference_500_minus_400_px"]
    right_400_profile = profile_summary[f"{TARGET_EXPOSURES_US[0]:g}"]["right"]
    right_500_profile = profile_summary[f"{TARGET_EXPOSURES_US[1]:g}"]["right"]
    right_shape_note = (
        f"right 代表列 profile 在 400/500 us 均未出现 double peak、shoulder 或 plateau；"
        f"symmetry index 为 {fmt(stats_at(right_400_profile['symmetry_index']))}/"
        f"{fmt(stats_at(right_500_profile['symmetry_index']))}，"
        f"FWHM 为 {fmt(stats_at(right_400_profile['fwhm_px']))}/"
        f"{fmt(stats_at(right_500_profile['fwhm_px']))} px。"
        "因此本轮的 morphology evidence 主要是区域化非对称/翼部形态和 "
        "sigma-independent 的 centroid 空间结构，不是明显双峰或饱和平台。"
    )
    if classification == "STEGER_SCALE_MISMATCH":
        next_step = (
            "左右边缘存在可量化的 sigma 敏感性；下一步仅在离线验证中按区域 "
            "评估候选 sigma，先确认有效率和高频 P95 不恶化，再决定是否进入独立的生产变更评审。"
        )
    elif classification == "STRIPE_MORPHOLOGY_LIMITED":
        next_step = (
            "条纹形态/曝光相关空间结构是主因；下一步应优先检查光学焦面、激光入射/反射几何、"
            "ROI 和原始 profile 形态，必要时改进形态感知的中心提取。仅继续 sweep sigma "
            "预计收益有限。"
        )
    elif classification == "MIXED":
        next_step = (
            "sigma 与条纹形态均有证据；下一步分离为两个离线实验：先固定曝光检查区域化 sigma，"
            "再固定候选 sigma 检查 profile 的肩峰/饱和/非对称性，生产参数仍保持不变。"
        )
    else:
        next_step = (
            "本批数据不能把边缘起伏明确归因于 sigma 或 profile 形态；下一步需要扩大固定列/帧的"
            "形态样本或补充同协议数据，但不应据此修改生产参数。"
        )
    lines = [
        "# 海康红光区域化 Steger 尺度适配与条纹形态诊断",
        "",
        f"- 诊断分类：**{classification}**",
        f"- 输入：{selected_names}；每组 {input_info['frame_count']} 帧。",
        "- 本轮仅 sweep sigma=1.0、1.2、1.5、1.8、2.0 px；threshold、deriv_thresh、roi_margin、"
        "roi_max_height、search ROI、centroid 参数和原始图像均固定。",
        "- 生产配置、外部 Steger profile、标定文件和生产代码参数均未修改。",
        "",
        "## 1. Provenance / reuse audit",
        "",
        "A-2 的 exposure_region_summary.csv、exposure_spatial_structure.csv 和 "
        "exposure_diagnostic_summary.json 已完成兼容性审计。本轮只复用其录制选择、"
        "采集协议、ROI、固定参数和文件 hash；A-2 的数值统计不作为 A-3 数值计算输入。",
        f"- A-2 target region 行数：{summary['provenance']['a2_audit']['target_region_row_count']}；"
        f"target spatial 行数：{summary['provenance']['a2_audit']['target_spatial_row_count']}。",
        f"- A-2 与当前 config hash 一致：{summary['provenance']['a2_audit']['config_hash_match']}；"
        f"profile hash 一致：{summary['provenance']['a2_audit']['profile_hash_match']}。",
        f"- A-2 协议可用于交叉核对：{summary['provenance']['a2_audit']['compatible_for_protocol_cross_check']}；"
        f"问题：{summary['provenance']['a2_audit']['issues'] or '无'}。",
        "",
        "## 2. 固定代表列与原始 profile",
        "",
        "代表列按当前有效 ROI 的左/中/右三等分固定选择，不根据 sigma、曝光或提取结果动态挑列。"
        "profile 在完整 ROI 纵向方向上计算 background=P20、contrast=peak-background、"
        "FWHM、半高宽对称性、积分对称性、双峰、肩峰、平台和饱和率。",
        "",
        markdown_table(
            [
                "曝光(us)",
                "区域",
                "u(px)",
                "peak P50",
                "contrast P50",
                "FWHM(px)",
                "symmetry",
                "double",
                "shoulder",
                "plateau",
                "peak=255",
                "FWHM CV",
            ],
            profile_rows,
        ),
        "",
        "形态判据：double peak 为次峰至少 0.25 主峰且峰间存在明显谷；shoulder 为较弱近邻次峰；"
        "plateau 为峰顶连续至少 3 px 且相对起伏不超过 5%。饱和平台单独保留 peak=255 比例，"
        "不把饱和削顶直接当成真实光学平台。",
        f"- {right_shape_note}",
        "",
        "## 3. sigma × region 结果",
        "",
        "空间高频指标不是全局直线拟合残差。先用当前 sigma=1.5 的 Steger 与 centroid "
        "中心估计共同低频几何趋势 g(u)，再对每条 trace 计算 "
        "(v-g)-G((v-g))，其中 G 的 Gaussian sigma=32 px；RMS/P95 只在对应区域的有效点上统计。",
        "这个共同趋势固定用于所有 sigma，因此 sigma 曲线适合做相对比较，不能当作每个 sigma "
        "独立基准下的绝对残差。每帧先计算区域 RMS/绝对残差 P95，再在 20 帧之间取 P50/P95；"
        "因此 CSV 中的 *_high_frequency_abs_p95_p50_px 不是所有有效像素池化后的 P95。",
        "temporal std 是同一曝光、同一列跨 20 帧的中心坐标标准差，要求至少 18 帧有效；"
        "supported-column ratio 同时保留在 CSV 中。",
        "",
        markdown_table(
            [
                "曝光(us)",
                "区域",
                "Steger valid(%)",
                "Steger HF RMS",
                "Steger |HF| P95 (frame P50)",
                "centroid HF RMS",
                "centroid temporal",
                "|delta| P50",
                "|delta| P95",
                "per-exposure best sigma",
            ],
            current_sigma_rows,
        ),
        "",
        "完整五组 sigma、三个区域、两种曝光以及 P50/P95 字段见 sigma_region_summary.csv；"
        "sigma_vs_region_metrics.png 展示所有 sigma 曲线。",
        "",
        "## 4. 400 -> 500 us，重点 right 区域",
        "",
        f"right 区域在当前 sigma=1.5 时 centroid 高频 RMS：400 us="
        f"{fmt(right_current['centroid_400_high_frequency_rms_p50_px'])} px，"
        f"500 us={fmt(right_current['centroid_500_high_frequency_rms_p50_px'])} px，"
        f"差值={fmt(right_difference)} px，500/400={fmt(right_ratio, 2)}。",
        f"该 centroid 指标与 sigma 无关；五个 sigma 上升是否持续："
        f"**{right['centroid_rise_persists_across_sigma']}**。",
        "因此，如果该上升持续，而 Steger 的最优 sigma 只改变 Steger 曲线，"
        "则不能把 400->500 的 centroid 空间起伏归咎于 Steger sigma。"
        "这是基于同一批原始帧、同一 centroid backend 的对照推断，不是三维结果反推。",
        "",
        markdown_table(
            [
                "sigma(px)",
                "Steger 400 HF RMS",
                "Steger 500 HF RMS",
                "centroid 400 HF RMS",
                "centroid 500 HF RMS",
                "centroid 500/400",
            ],
            [
                [
                    fmt(item["sigma_px"], 1),
                    fmt(item["steger_400_high_frequency_rms_p50_px"]),
                    fmt(item["steger_500_high_frequency_rms_p50_px"]),
                    fmt(item["centroid_400_high_frequency_rms_p50_px"]),
                    fmt(item["centroid_500_high_frequency_rms_p50_px"]),
                    fmt(item["centroid_ratio_500_over_400"], 2),
                ]
                for item in right["by_sigma"].values()
            ],
        ),
        "",
        "## 5. 根因判定",
        "",
        markdown_table(
            ["区域", "sigma evidence", "raw morphology", "centroid 500/400", "dominant"],
            [
                [
                    region,
                    diagnosis["by_region"][region]["sigma_scale_evidence"],
                    diagnosis["by_region"][region]["raw_profile_morphology_evidence"],
                    fmt(diagnosis["by_region"][region]["centroid_500_over_400"], 2),
                    diagnosis["by_region"][region]["dominant_evidence"],
                ]
                for region in REGIONS
            ],
        ),
        "",
        f"分类依据：{classification}。sigma evidence 使用当前 sigma 到同曝光最优 sigma 的高频 RMS "
        f"相对改善（阈值 {SCALE_MISMATCH_GAIN_WARN:.0%}），并要求有效率不下降超过 5 个百分点；"
        "raw morphology 使用 profile 的 FWHM CV、非对称、双峰、肩峰和平台判据；"
        "centroid 500/400 持续升高则作为 sigma-independent 的空间形态证据。",
        "",
        "## 6. 推荐下一步",
        "",
        f"- 边缘联合的离线候选 sigma：{recommendation['edge_robust_candidate']['best_sigma_px']:g} px；"
        "它只是本批 400/500 us 的候选，不是生产配置建议。",
        *[
            f"- {region} 可接受候选范围："
            f"{recommendation['by_region'][region]['acceptable_sigma_range_px'] or '无稳定范围'}；"
            f"当前 1.5 px 是否在范围内："
            f"{recommendation['by_region'][region]['current_sigma_in_acceptable_range']}。"
            for region in REGIONS
        ],
        f"- {next_step}",
        "",
        "## 7. 输出与边界",
        "",
        "本轮新增计算：原始 profile、五组 sigma 的 Steger 提取、区域高频空间指标、"
        "temporal std、Steger-centroid delta、形态诊断和图表。没有读取三维结果，没有重新采集，"
        "没有 sweep threshold/deriv_thresh，没有修改生产配置。",
        "输出：sigma_region_summary.csv、representative_profiles.csv、"
        "sigma_region_diagnostic_summary.json、profiles_by_region.png、"
        "sigma_vs_region_metrics.png、sigma_region_diagnostic_report.md。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steger-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--a2-diagnostic-dir", type=Path, default=DEFAULT_A2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def build_profile_csv_rows(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        top = int(result["roi"]["top"])
        for region in REGIONS:
            profiles = np.asarray(
                result["representative_profiles"][region], dtype=np.float64
            )
            if profiles.ndim != 2 or len(profiles) == 0:
                continue
            median = np.median(profiles, axis=0)
            low = np.percentile(profiles, 10, axis=0)
            high = np.percentile(profiles, 90, axis=0)
            for index, (p50, p10, p90) in enumerate(
                zip(median, low, high, strict=True)
            ):
                rows.append(
                    {
                        "recording": result["recording"],
                        "exposure_us": result["exposure_us"],
                        "region": region,
                        "u_px": result["representative_columns"][region],
                        "v_px": top + index,
                        "median_dn": p50,
                        "p10_dn": p10,
                        "p90_dn": p90,
                    }
                )
    return rows


def extract_results(
    selected: Sequence[Mapping[str, Any]],
    steger_options: Mapping[str, Any],
    centroid_options: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], tuple[int, int]]:
    results: list[dict[str, Any]] = []
    first_shape: tuple[int, int] | None = None
    first_roi: dict[str, int] | None = None
    for recording_number, item in enumerate(selected, start=1):
        print(
            f"[{recording_number}/{len(selected)}] processing {item['recording']} "
            f"({float(item['exposure_us']):g} us, {item['frame_count']} frames)",
            flush=True,
        )
        vectors: dict[str, dict[str, list[np.ndarray]] | list[np.ndarray]] = {
            "steger": {f"{sigma:.6g}": [] for sigma in SIGMAS_PX},
            "centroid": [],
        }
        profile_rows: list[dict[str, Any]] = []
        representative_profiles: dict[str, list[np.ndarray]] = {
            region: [] for region in REGIONS
        }
        image_shape: tuple[int, int] | None = None
        roi: dict[str, int] | None = None
        representatives: dict[str, int] | None = None
        for frame_index, (frame_path, metadata) in enumerate(
            zip(item["frame_paths"], item["metadata"], strict=True),
            start=1,
        ):
            image = read_image(frame_path)
            shape = tuple(int(value) for value in image.shape)
            if image_shape is None:
                image_shape = shape
                roi = parse_search_roi(steger_options, image_shape)
                representatives = representative_columns(roi)
                first_shape = first_shape or image_shape
                first_roi = first_roi or dict(roi)
            elif shape != image_shape:
                raise DiagnosticError(
                    f"image shape changed in {item['recording']}: {shape} != {image_shape}"
                )
            assert image_shape is not None and roi is not None and representatives is not None
            for key, image_value in (("width", image_shape[1]), ("height", image_shape[0])):
                csv_value_raw = metadata.get(key)
                if csv_value_raw and int(float(csv_value_raw)) != image_value:
                    raise DiagnosticError(
                        f"frames.csv {key} mismatch for {frame_path}: "
                        f"{csv_value_raw} != {image_value}"
                    )
            roi_top = int(roi["top"])
            roi_bottom = int(roi["bottom"])
            for region, u_px in representatives.items():
                profile = image[roi_top:roi_bottom, u_px].astype(
                    np.float64, copy=True
                )
                shape_metrics = profile_shape_metrics(profile)
                profile_rows.append(
                    {
                        "recording": item["recording"],
                        "exposure_us": item["exposure_us"],
                        "frame_index": frame_index,
                        "frame_filename": frame_path.name,
                        "region": region,
                        "u_px": u_px,
                        **shape_metrics,
                    }
                )
                representative_profiles[region].append(profile)
            try:
                centroid_points = np.asarray(
                    centroid_backend(image, centroid_options), dtype=np.float64
                )
                centroid_full = points_to_vector(centroid_points, image_shape[1])
                centroid_roi = centroid_full[roi["left"] : roi["right"]]
                vectors["centroid"].append(centroid_roi)
                for sigma in SIGMAS_PX:
                    trial_options = dict(steger_options)
                    trial_options["sigma"] = float(sigma)
                    steger_points = np.asarray(
                        steger_backend(image, trial_options), dtype=np.float64
                    )
                    steger_full = points_to_vector(steger_points, image_shape[1])
                    vectors["steger"][f"{sigma:.6g}"].append(
                        steger_full[roi["left"] : roi["right"]]
                    )
            except Exception as error:  # noqa: BLE001 - add frame provenance
                raise DiagnosticError(
                    f"backend extraction failed for {item['recording']} "
                    f"{frame_path.name}: {error}"
                ) from error
            if frame_index == 1 or frame_index == item["frame_count"]:
                print(
                    f"  frame {frame_index}/{item['frame_count']} extracted",
                    flush=True,
                )
        assert image_shape is not None and roi is not None and representatives is not None
        result = {
            "recording": item["recording"],
            "recording_dir": item["path"],
            "exposure_us": float(item["exposure_us"]),
            "frame_count": int(item["frame_count"]),
            "image_shape": image_shape,
            "roi": roi,
            "representative_columns": representatives,
            "profile_rows": profile_rows,
            "representative_profiles": {
                region: np.asarray(values, dtype=np.float64)
                for region, values in representative_profiles.items()
            },
            "profile_summary": {
                region: {
                    "representative_u_px": representatives[region],
                    **aggregate_profile_metrics(
                        profile_rows_for_result(
                            {"profile_rows": profile_rows}, region
                        )
                    ),
                }
                for region in REGIONS
            },
            "vectors": {
                "steger": {
                    sigma: np.asarray(values, dtype=np.float64)
                    for sigma, values in vectors["steger"].items()
                },
                "centroid": np.asarray(vectors["centroid"], dtype=np.float64),
            },
            "capture_values": item["unique_values"],
        }
        results.append(result)
        print(
            f"  finite centers: centroid={np.count_nonzero(np.isfinite(result['vectors']['centroid']))}, "
            f"Steger@1.5={np.count_nonzero(np.isfinite(result['vectors']['steger']['1.5']))}",
            flush=True,
        )
    if first_shape is None or first_roi is None:
        raise DiagnosticError("no images were processed")
    return results, first_roi, first_shape


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recordings_root = args.recordings_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    profile_path = args.steger_profile.expanduser().resolve()
    a2_dir = args.a2_diagnostic_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    selected, ignored = discover_target_recordings(recordings_root)
    config, steger_options, centroid_options, profile_info = load_fixed_options(
        config_path, profile_path
    )
    a2_audit = audit_a2_artifacts(a2_dir, selected, config_path, profile_path)
    results, first_roi, first_shape = extract_results(
        selected, steger_options, centroid_options
    )
    if any(result["roi"] != first_roi for result in results):
        raise DiagnosticError("selected recordings do not share the same effective ROI")
    if any(result["image_shape"] != first_shape for result in results):
        raise DiagnosticError("selected recordings do not share the same image shape")

    baseline_matrices = [
        result["vectors"]["steger"][f"{CURRENT_SIGMA_PX:.6g}"]
        for result in results
    ] + [result["vectors"]["centroid"] for result in results]
    common_trend, common_support = build_common_geometry_trend(
        baseline_matrices, LOW_FREQUENCY_SIGMA_PX
    )
    sigma_rows = build_sigma_region_rows(results, common_trend, common_support)
    profile_summary = {
        f"{float(result['exposure_us']):g}": result["profile_summary"]
        for result in results
    }
    recommendation = choose_sigma_recommendation(sigma_rows)
    right_comparison = build_right_exposure_comparison(sigma_rows)
    diagnosis = build_regional_diagnosis(sigma_rows, profile_summary)
    profile_changes = profile_change_summary(profile_summary)
    generated_at = datetime.now(timezone.utc).isoformat()

    config_referenced_profile = profile_info["config_referenced_profile"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "task": "hikrobot_red_region_sigma_and_stripe_morphology",
        "classification": diagnosis["classification"],
        "generated_at_utc": generated_at,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input": {
            "recordings_root": recordings_root,
            "selected_recordings": [
                {
                    "recording": item["recording"],
                    "directory": item["path"],
                    "exposure_us": item["exposure_us"],
                    "frame_count": item["frame_count"],
                    "capture_values": item["unique_values"],
                }
                for item in selected
            ],
            "ignored_recordings": ignored,
            "target_exposures_us": TARGET_EXPOSURES_US,
            "frame_count": EXPECTED_FRAME_COUNT,
            "image_shape": first_shape,
            "dtype": "uint8",
            "roi": first_roi,
            "representative_columns_px": results[0]["representative_columns"],
            "same_non_exposure_capture_signature": selected[0]["common_signature"],
        },
        "provenance": {
            "calculation_source": "raw Mono8 PNG and frames.csv only",
            "three_dimensional_results_used": False,
            "a2_audit": a2_audit,
            "config_path": config_path,
            "config_sha256": sha256_file(config_path),
            "profile_path": profile_path,
            "profile_sha256": sha256_file(profile_path),
            "config_referenced_profile_path": config_referenced_profile,
            "config_referenced_profile_sha256": (
                sha256_file(config_referenced_profile)
                if config_referenced_profile is not None
                and config_referenced_profile.is_file()
                else None
            ),
            "implementation_sha256": {
                "laser_backends.py": sha256_file(TOOL_ROOT / "laser" / "backends.py"),
                "laser_realtime_steger.py": sha256_file(
                    TOOL_ROOT / "laser" / "realtime_steger.py"
                ),
            },
            "production_configuration_modified": False,
            "production_code_parameters_modified": False,
        },
        "parameters": {
            "steger_options_fixed_except_sigma": steger_options,
            "centroid_options_fixed": centroid_options,
            "sigma_sweep_px": SIGMAS_PX,
            "current_sigma_px": CURRENT_SIGMA_PX,
            "low_frequency_geometry_sigma_px": LOW_FREQUENCY_SIGMA_PX,
            "profile_background_percentile": PROFILE_BACKGROUND_PERCENTILE,
            "profile_smoothing_sigma_px": PROFILE_SMOOTHING_SIGMA_PX,
            "saturation_dn": SATURATION_DN,
            "minimum_temporal_frame_ratio": MIN_TEMPORAL_FRAME_RATIO,
            "only_sigma_changed": True,
            "threshold_swept": False,
            "deriv_thresh_swept": False,
            "roi_swept": False,
        },
        "common_geometry_trend": {
            "source": "current sigma=1.5 Steger and fixed centroid, 400/500 us, all frames",
            "support_count_min": int(np.min(common_support)),
            "support_count_median": float(np.median(common_support)),
            "support_count_max": int(np.max(common_support)),
            "trace_count": len(baseline_matrices),
        },
        "profiles": {
            "by_exposure_region": profile_summary,
            "400_to_500_change": profile_changes,
        },
        "sigma_region_summary": sigma_rows,
        "recommendation": recommendation,
        "right_exposure_comparison": right_comparison,
        "diagnosis": diagnosis,
        "outputs": {
            "sigma_region_summary_csv": output_dir / "sigma_region_summary.csv",
            "representative_profiles_csv": output_dir / "representative_profiles.csv",
            "summary_json": output_dir / "sigma_region_diagnostic_summary.json",
            "profiles_png": output_dir / "profiles_by_region.png",
            "sigma_metrics_png": output_dir / "sigma_vs_region_metrics.png",
            "report_md": output_dir / "sigma_region_diagnostic_report.md",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_fields = [
        "recording",
        "exposure_us",
        "frame_count",
        "sigma_px",
        "region",
        "representative_u_px",
        "roi_left_px",
        "roi_top_px",
        "roi_width_px",
        "roi_height_px",
        "peak_dn_p50",
        "contrast_dn_p50",
        "fwhm_px_p50",
        "profile_fwhm_cv",
        "profile_symmetry_p50",
        "profile_double_peak_fraction",
        "profile_shoulder_fraction",
        "profile_plateau_fraction",
        "profile_peak_saturated_fraction",
        "steger_valid_ratio_p50",
        "steger_valid_ratio_p05",
        "steger_valid_ratio_p95",
        "centroid_valid_ratio_p50",
        "centroid_valid_ratio_p05",
        "centroid_valid_ratio_p95",
        "steger_high_frequency_rms_p50_px",
        "steger_high_frequency_rms_p95_px",
        "steger_high_frequency_abs_p95_p50_px",
        "steger_high_frequency_abs_p95_p95_px",
        "centroid_high_frequency_rms_p50_px",
        "centroid_high_frequency_rms_p95_px",
        "centroid_high_frequency_abs_p95_p50_px",
        "centroid_high_frequency_abs_p95_p95_px",
        "steger_temporal_std_p50_px",
        "steger_temporal_std_p95_px",
        "steger_temporal_supported_column_ratio",
        "centroid_temporal_std_p50_px",
        "centroid_temporal_std_p95_px",
        "centroid_temporal_supported_column_ratio",
        "delta_signed_p50_px",
        "delta_abs_p50_px",
        "delta_abs_p95_px",
        "delta_pair_valid_ratio",
        "delta_temporal_std_p50_px",
        "delta_temporal_std_p95_px",
        "delta_temporal_supported_column_ratio",
        "common_geometry_support_ratio_p50",
    ]
    write_csv(output_dir / "sigma_region_summary.csv", sigma_rows, sigma_fields)
    profile_csv_rows = build_profile_csv_rows(results)
    write_csv(
        output_dir / "representative_profiles.csv",
        profile_csv_rows,
        [
            "recording",
            "exposure_us",
            "region",
            "u_px",
            "v_px",
            "median_dn",
            "p10_dn",
            "p90_dn",
        ],
    )
    (output_dir / "sigma_region_diagnostic_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_profiles_plot(output_dir / "profiles_by_region.png", results)
    save_sigma_metrics_plot(output_dir / "sigma_vs_region_metrics.png", sigma_rows)
    write_report(
        output_dir / "sigma_region_diagnostic_report.md",
        summary,
        sigma_rows,
    )
    print(f"classification: {diagnosis['classification']}")
    print(
        "right centroid HF RMS 400->500 at sigma=1.5: "
        f"{fmt(right_comparison['current_sigma']['centroid_400_high_frequency_rms_p50_px'])}"
        " -> "
        f"{fmt(right_comparison['current_sigma']['centroid_500_high_frequency_rms_p50_px'])} px"
    )
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"diagnostic error: {error}", file=sys.stderr)
        raise SystemExit(2)
