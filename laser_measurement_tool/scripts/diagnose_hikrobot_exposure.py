#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""海康 Mono8 红光曝光敏感性与空间结构诊断。

本脚本只读取 ``recording_*`` 下的原始二维 Mono8 PNG 和 ``frames.csv``，
不会读取或反推任何三维结果，也不会修改生产配置、标定文件或生产代码。

分析固定当前配置中的 Steger/centroid 参数，只让输入 recording 的曝光变化：

* 自动发现参数一致且约 20 帧的 recording；曝光值始终从 frames.csv 读取；
* 沿当前 Steger search ROI 的固定列测量 peak/background/contrast/FWHM/饱和；
* 用当前 sigma=1.5 的 Steger 与现有 centroid backend 提取中心；
* 用所有曝光和两种算法的共同低频几何趋势建立空间高通残差，统计局部高频
  起伏，而不是把全局直线拟合残差当作提取误差；
* 输出 CSV、JSON、Markdown 和无 GUI PNG 图。

运行示例（仓库根目录）：

    .venv\\Scripts\\python.exe \\
      laser_measurement_tool\\scripts\\diagnose_hikrobot_exposure.py

默认输出到：
``laser_measurement_tool/output_haikang_0828/online_recordings/exposure_diagnostic``。
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
from scipy.stats import rankdata  # noqa: E402


TOOL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOL_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_RECORDINGS_ROOT = (
    TOOL_ROOT / "output_haikang_0828" / "online_recordings"
)
DEFAULT_CONFIG_PATH = TOOL_ROOT / "configs" / "measure_tool.yaml"
DEFAULT_PROFILE_PATH = WORKSPACE_ROOT / "calibration" / "config" / "realtime_steger.yaml"
DEFAULT_OUTPUT_DIR = DEFAULT_RECORDINGS_ROOT / "exposure_diagnostic"

EXPECTED_FRAME_COUNT = 20
MIN_FRAME_COUNT = 18
MAX_FRAME_COUNT = 22
CURRENT_SIGMA_PX = 1.5
PROFILE_BACKGROUND_PERCENTILE = 20.0
SATURATION_DN = 255.0
PROFILE_SAMPLE_STEP_PX = 16
LOW_FREQUENCY_SIGMA_PX = 32.0
MIN_TEMPORAL_FRAME_RATIO = 0.90
MIN_PROFILE_CONTRAST_DN = 20.0
MIN_EDGE_VALID_RATIO = 0.90
CENTER_PEAK_SATURATION_WARN = 0.10
CENTER_PEAK_SATURATION_SEVERE = 0.25
HF_RELATIVE_TRADEOFF = 0.15
HF_ACCEPTABLE_RELATIVE_TO_BEST = 1.25
MIN_COLUMNS_FOR_CORRELATION = 5
RECOMMEND_CENTER_PEAK_SATURATION_MAX = 0.25
RECOMMEND_EDGE_CONTRAST_MIN_DN = 50.0
RECOMMEND_EDGE_VALID_PLATFORM_FRACTION = 0.95

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
REGIONS = ("left", "center", "right", "all")
ALGORITHMS = ("steger", "centroid")
FRAME_PATTERN = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)

if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import AppConfig, load_app_config  # noqa: E402
from laser.backends import centroid_backend, steger_backend  # noqa: E402


class DiagnosticError(RuntimeError):
    """输入、配置或统计协议不满足诊断要求。"""


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive float") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive float")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DiagnosticError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def read_image(path: Path) -> np.ndarray:
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DiagnosticError(f"cannot decode image: {path}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise DiagnosticError(
            f"image must be single-channel Mono8: {path}, "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    return np.ascontiguousarray(image)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def region_for_local_u(local_u: int, roi_width: int) -> str:
    if local_u < roi_width / 3.0:
        return "left"
    if local_u < roi_width * 2.0 / 3.0:
        return "center"
    return "right"


def region_mask(region: str, roi_width: int) -> np.ndarray:
    columns = np.arange(roi_width)
    if region == "left":
        return columns < roi_width / 3.0
    if region == "center":
        return (columns >= roi_width / 3.0) & (columns < roi_width * 2.0 / 3.0)
    if region == "right":
        return columns >= roi_width * 2.0 / 3.0
    return np.ones(roi_width, dtype=bool)


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


def profile_fwhm(profile: np.ndarray, peak_index: int, level: float) -> float:
    values = np.asarray(profile, dtype=np.float64)
    if len(values) < 3 or peak_index <= 0 or peak_index >= len(values) - 1:
        return float("nan")
    left_index = peak_index
    while left_index > 0 and values[left_index - 1] >= level:
        left_index -= 1
    right_index = peak_index
    while right_index < len(values) - 1 and values[right_index + 1] >= level:
        right_index += 1
    if left_index == 0 or right_index == len(values) - 1:
        return float("nan")

    def crossing(index_a: int, index_b: int) -> float:
        value_a = float(values[index_a])
        value_b = float(values[index_b])
        if abs(value_b - value_a) <= np.finfo(float).eps:
            return float(index_a)
        return index_a + (level - value_a) / (value_b - value_a)

    left_crossing = crossing(left_index - 1, left_index)
    right_crossing = crossing(right_index, right_index + 1)
    width = right_crossing - left_crossing
    return float(width) if width > 0.0 and math.isfinite(width) else float("nan")


def measure_profile(
    image: np.ndarray, u_px: int, roi: Mapping[str, int]
) -> dict[str, Any]:
    top = int(roi["top"])
    bottom = int(roi["bottom"])
    profile = image[top:bottom, int(u_px)].astype(np.float64, copy=False)
    peak_index = int(np.argmax(profile))
    peak_dn = float(profile[peak_index])
    background_dn = float(np.percentile(profile, PROFILE_BACKGROUND_PERCENTILE))
    contrast_dn = peak_dn - background_dn
    half_level = background_dn + 0.5 * contrast_dn
    fwhm = profile_fwhm(profile, peak_index, half_level) if contrast_dn > 0 else float("nan")
    saturated_count = int(np.count_nonzero(profile >= SATURATION_DN))
    return {
        "peak_dn": peak_dn,
        "peak_v_px": float(top + peak_index),
        "background_dn": background_dn,
        "contrast_dn": contrast_dn,
        "half_level_dn": half_level,
        "fwhm_px": fwhm,
        "profile_saturation_rate": saturated_count / max(len(profile), 1),
        "profile_saturated_pixel_count": saturated_count,
        "peak_saturated": bool(peak_dn >= SATURATION_DN),
    }


def finite_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
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


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


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


def parse_frame_metadata(recording_dir: Path) -> tuple[list[Path], list[dict[str, str]], dict[str, Any]]:
    metadata = read_csv(recording_dir / "frames.csv")
    frame_paths = sorted(recording_dir.glob("frame_*.png"), key=frame_sort_key)
    if not (MIN_FRAME_COUNT <= len(frame_paths) <= MAX_FRAME_COUNT):
        raise DiagnosticError(
            f"frame count {len(frame_paths)} is outside [{MIN_FRAME_COUNT}, {MAX_FRAME_COUNT}]"
        )
    if len(metadata) != len(frame_paths):
        raise DiagnosticError(
            f"frames.csv rows {len(metadata)} != PNG count {len(frame_paths)}"
        )
    names = [row.get("filename", "") for row in metadata]
    actual_names = [path.name for path in frame_paths]
    if names != actual_names or len(set(names)) != len(names):
        raise DiagnosticError("frames.csv filename order does not match frame_*.png")
    for key in CAPTURE_KEYS:
        if key not in (metadata[0] if metadata else {}):
            raise DiagnosticError(f"frames.csv missing capture field: {key}")
    unique_values_by_key = {
        key: unique_capture_values(metadata, key) for key in CAPTURE_KEYS
    }
    inconsistent = {
        key: values for key, values in unique_values_by_key.items() if len(values) != 1
    }
    if inconsistent:
        raise DiagnosticError(
            "capture fields vary within recording: "
            + json.dumps(inconsistent, ensure_ascii=False)
        )
    pixel_format = unique_values_by_key["pixel_format"][0]
    if pixel_format.lower() != "mono8":
        raise DiagnosticError(f"pixel_format is not Mono8: {pixel_format}")
    exposure = finite_float(metadata[0].get("exposure_us"))
    if exposure is None or exposure <= 0.0:
        raise DiagnosticError("frames.csv exposure_us is invalid")
    return frame_paths, metadata, {
        "unique_values": unique_values_by_key,
        "exposure_us": exposure,
        "common_signature": tuple(
            (key, unique_values_by_key[key][0]) for key in COMMON_CAPTURE_KEYS
        ),
    }


def discover_recordings(
    recordings_root: Path,
    min_frames: int,
    max_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not recordings_root.is_dir():
        raise DiagnosticError(f"recordings root does not exist: {recordings_root}")
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    candidates = sorted(
        [path for path in recordings_root.iterdir() if path.is_dir() and path.name.startswith("recording_")],
        key=lambda path: path.name,
    )
    if not candidates:
        raise DiagnosticError(f"no recording_* directory under {recordings_root}")
    reference_signature: tuple[tuple[str, str], ...] | None = None
    for recording_dir in candidates:
        item: dict[str, Any] = {"recording": recording_dir.name, "path": recording_dir}
        try:
            # Use the caller's approximate-20-frame policy here so the same policy
            # is recorded in the JSON rather than silently accepting a different size.
            metadata = read_csv(recording_dir / "frames.csv")
            frame_paths = sorted(recording_dir.glob("frame_*.png"), key=frame_sort_key)
            if not (min_frames <= len(frame_paths) <= max_frames):
                raise DiagnosticError(
                    f"frame count {len(frame_paths)} is outside [{min_frames}, {max_frames}]"
                )
            if len(metadata) != len(frame_paths):
                raise DiagnosticError("frames.csv row count does not match PNG count")
            names = [row.get("filename", "") for row in metadata]
            if names != [path.name for path in frame_paths] or len(set(names)) != len(names):
                raise DiagnosticError("frames.csv filename order does not match PNG files")
            unique_values_by_key = {
                key: unique_capture_values(metadata, key) for key in CAPTURE_KEYS
            }
            inconsistent = {
                key: values for key, values in unique_values_by_key.items() if len(values) != 1
            }
            if inconsistent:
                raise DiagnosticError(
                    "capture fields vary within recording: "
                    + json.dumps(inconsistent, ensure_ascii=False)
                )
            if unique_values_by_key["pixel_format"][0].lower() != "mono8":
                raise DiagnosticError("pixel_format is not Mono8")
            exposure = finite_float(metadata[0].get("exposure_us"))
            if exposure is None or exposure <= 0.0:
                raise DiagnosticError("invalid exposure_us")
            signature = tuple(
                (key, unique_values_by_key[key][0]) for key in COMMON_CAPTURE_KEYS
            )
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise DiagnosticError(
                    "capture parameters differ from the first valid recording"
                )
            item.update(
                {
                    "frame_paths": frame_paths,
                    "metadata": metadata,
                    "frame_count": len(frame_paths),
                    "exposure_us": exposure,
                    "unique_values": unique_values_by_key,
                    "common_signature": signature,
                }
            )
            included.append(item)
        except (DiagnosticError, OSError, UnicodeError, csv.Error) as error:
            item["reason"] = str(error)
            excluded.append(item)
    if not included:
        raise DiagnosticError("no recording passed the inclusion policy")
    return included, excluded


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
        raise DiagnosticError(f"cannot parse config for extraction.profile: {config_path}") from error
    extraction = document.get("extraction", {})
    if not isinstance(extraction, Mapping):
        return None
    value = extraction.get("profile")
    if value in (None, ""):
        return None
    raw = Path(str(value))
    return (config_path.parent / raw).resolve() if not raw.is_absolute() else raw.resolve()


def load_fixed_options(
    config_path: Path, profile_path: Path
) -> tuple[AppConfig, dict[str, Any], dict[str, Any], dict[str, Any]]:
    config: AppConfig = load_app_config(config_path)
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
            f"current configured sigma is {configured_sigma}, expected {CURRENT_SIGMA_PX}"
        )
    external_profile = load_profile_options(profile_path)
    compare_keys = ("sigma", "threshold", "deriv_thresh", "roi_margin", "roi_max_height", "scan_axis")
    mismatches = {
        key: {"config": steger_options.get(key), "profile": external_profile.get(key)}
        for key in compare_keys
        if key in external_profile
        and key in steger_options
        and external_profile.get(key) != steger_options.get(key)
    }
    if mismatches:
        raise DiagnosticError(
            "external Steger profile differs from current config: "
            + json.dumps(jsonable(mismatches), ensure_ascii=False)
        )
    return config, steger_options, centroid_options, {
        "profile_options": external_profile,
        "config_referenced_profile": profile_path_from_config(config_path),
        "profile_mismatches": mismatches,
    }


def aggregate_profile_rows(
    rows: Sequence[Mapping[str, Any]], region: str, roi_width: int
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if region == "all" or row["region"] == region
    ]
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in selected if row.get(name) is not None]

    peak = finite_stats(values("peak_dn"))
    background = finite_stats(values("background_dn"))
    contrast = finite_stats(values("contrast_dn"))
    saturation = finite_stats(values("profile_saturation_rate"))
    fwhm = finite_stats(values("fwhm_px"))
    saturated_peaks = [bool(row["peak_saturated"]) for row in selected]
    return {
        "region": region,
        "sample_count": len(selected),
        "region_column_count": int(np.count_nonzero(region_mask(region, roi_width))),
        "peak_dn": peak,
        "background_dn": background,
        "contrast_dn": contrast,
        "profile_saturation_rate": saturation,
        "fwhm_px": fwhm,
        "fwhm_valid_ratio": (
            fwhm["count"] / len(selected) if selected else float("nan")
        ),
        "peak_saturated_fraction": (
            float(np.count_nonzero(saturated_peaks) / len(saturated_peaks))
            if saturated_peaks
            else float("nan")
        ),
    }


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
    vectors_by_algorithm: Sequence[np.ndarray], low_frequency_sigma_px: float
) -> tuple[np.ndarray, np.ndarray]:
    if not vectors_by_algorithm:
        raise DiagnosticError("no extraction vectors for common geometry trend")
    stacked = np.concatenate(vectors_by_algorithm, axis=0)
    with np.errstate(all="ignore"):
        # nanmedian emits a warning for columns unsupported by every extractor;
        # those columns are retained as unsupported and excluded downstream.
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


def high_frequency_residual(
    values: np.ndarray,
    common_trend: np.ndarray,
    low_frequency_sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    common_trend = np.asarray(common_trend, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(common_trend)
    residual = np.full(values.shape, np.nan, dtype=np.float64)
    high = np.full(values.shape, np.nan, dtype=np.float64)
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return residual, high
    residual[indices] = values[indices] - common_trend[indices]
    support = np.arange(len(values), dtype=np.float64)
    filled = np.interp(support, indices, residual[indices])
    local_low = gaussian_filter1d(
        filled,
        sigma=low_frequency_sigma_px,
        mode="nearest",
        truncate=4.0,
    )
    high[indices] = residual[indices] - local_low[indices]
    return residual, high


def per_frame_region_metrics(
    matrix: np.ndarray,
    high_matrix: np.ndarray,
    region: str,
    roi_width: int,
) -> list[dict[str, Any]]:
    mask = region_mask(region, roi_width)
    rows: list[dict[str, Any]] = []
    for frame_index, (values, high) in enumerate(zip(matrix, high_matrix, strict=True), start=1):
        local_values = values[mask]
        local_high = high[mask]
        finite_values = local_values[np.isfinite(local_values)]
        finite_high = local_high[np.isfinite(local_high)]
        rows.append(
            {
                "frame_index": frame_index,
                "valid_ratio": ratio(len(finite_values), int(np.count_nonzero(mask))),
                "high_frequency_rms_px": (
                    float(np.sqrt(np.mean(finite_high**2))) if len(finite_high) else float("nan")
                ),
                "high_frequency_abs_p95_px": (
                    float(np.percentile(np.abs(finite_high), 95)) if len(finite_high) else float("nan")
                ),
                "high_frequency_abs_max_px": (
                    float(np.max(np.abs(finite_high))) if len(finite_high) else float("nan")
                ),
            }
        )
    return rows


def temporal_column_metrics(
    matrix: np.ndarray, minimum_frame_count: int
) -> dict[str, np.ndarray]:
    columns = matrix.shape[1]
    std = np.full(columns, np.nan, dtype=np.float64)
    valid_ratio = np.zeros(columns, dtype=np.float64)
    value_p50 = np.full(columns, np.nan, dtype=np.float64)
    value_abs_p95 = np.full(columns, np.nan, dtype=np.float64)
    value_range = np.full(columns, np.nan, dtype=np.float64)
    for column in range(columns):
        values = matrix[:, column]
        finite = values[np.isfinite(values)]
        valid_ratio[column] = len(finite) / max(matrix.shape[0], 1)
        if len(finite):
            value_p50[column] = float(np.median(finite))
            value_abs_p95[column] = float(np.percentile(np.abs(finite), 95))
            value_range[column] = float(np.max(finite) - np.min(finite))
        if len(finite) >= 2:
            std[column] = float(np.std(finite, ddof=0))
    supported = np.count_nonzero(np.isfinite(std) & (valid_ratio >= minimum_frame_count / matrix.shape[0]))
    return {
        "temporal_std_px": std,
        "valid_frame_ratio": valid_ratio,
        "value_p50_px": value_p50,
        "value_abs_p95_px": value_abs_p95,
        "value_range_px": value_range,
        "supported_column_count": np.asarray([supported], dtype=np.int64),
    }


def column_high_frequency_metrics(high_matrix: np.ndarray) -> dict[str, np.ndarray]:
    columns = high_matrix.shape[1]
    rms = np.full(columns, np.nan, dtype=np.float64)
    abs_p95 = np.full(columns, np.nan, dtype=np.float64)
    for column in range(columns):
        values = high_matrix[:, column]
        finite = values[np.isfinite(values)]
        if len(finite):
            rms[column] = float(np.sqrt(np.mean(finite**2)))
            abs_p95[column] = float(np.percentile(np.abs(finite), 95))
    return {"high_frequency_rms_px": rms, "high_frequency_abs_p95_px": abs_p95}


def region_temporal_summary(
    temporal: Mapping[str, np.ndarray], region: str, roi_width: int, frame_count: int
) -> dict[str, Any]:
    mask = region_mask(region, roi_width)
    support_mask = (
        np.isfinite(temporal["temporal_std_px"])
        & (temporal["valid_frame_ratio"] >= MIN_TEMPORAL_FRAME_RATIO)
        & mask
    )
    std_values = temporal["temporal_std_px"][support_mask]
    valid_values = temporal["valid_frame_ratio"][mask]
    return {
        "region": region,
        "minimum_supported_frames": max(2, int(math.ceil(frame_count * MIN_TEMPORAL_FRAME_RATIO))),
        "supported_column_count": int(len(std_values)),
        "temporal_std_px": finite_stats(std_values),
        "valid_frame_ratio": finite_stats(valid_values),
    }


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if len(first) < MIN_COLUMNS_FOR_CORRELATION:
        return {"count": int(len(first)), "pearson": None, "spearman": None}
    if np.std(first) <= np.finfo(float).eps or np.std(second) <= np.finfo(float).eps:
        return {"count": int(len(first)), "pearson": 0.0, "spearman": 0.0}
    first_rank = rankdata(first)
    second_rank = rankdata(second)
    return {
        "count": int(len(first)),
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(np.corrcoef(first_rank, second_rank)[0, 1]),
    }


def overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, Any]:
    first = np.asarray(mask_a, dtype=bool)
    second = np.asarray(mask_b, dtype=bool)
    count_a = int(np.count_nonzero(first))
    count_b = int(np.count_nonzero(second))
    intersection = int(np.count_nonzero(first & second))
    return {
        "a_count": count_a,
        "b_count": count_b,
        "intersection_count": intersection,
        "intersection_over_a": intersection / count_a if count_a else None,
        "intersection_over_b": intersection / count_b if count_b else None,
        "union_ratio": intersection / int(np.count_nonzero(first | second))
        if np.count_nonzero(first | second)
        else None,
    }


def stats_value(item: Mapping[str, Any], key: str = "median") -> float | None:
    value = item.get(key)
    return finite_float(value)


def mean_or_none(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def aggregate_frame_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "frame_count": len(rows),
        "valid_ratio": finite_stats([float(row["valid_ratio"]) for row in rows]),
        "high_frequency_rms_px": finite_stats(
            [float(row["high_frequency_rms_px"]) for row in rows]
        ),
        "high_frequency_abs_p95_px": finite_stats(
            [float(row["high_frequency_abs_p95_px"]) for row in rows]
        ),
        "high_frequency_abs_max_px": finite_stats(
            [float(row["high_frequency_abs_max_px"]) for row in rows]
        ),
    }


def summarize_delta(
    delta_matrix: np.ndarray,
    region: str,
    roi_width: int,
    frame_count: int,
) -> dict[str, Any]:
    mask = region_mask(region, roi_width)
    values = delta_matrix[:, mask]
    paired = values[np.isfinite(values)]
    temporal = temporal_column_metrics(delta_matrix, max(2, int(math.ceil(frame_count * MIN_TEMPORAL_FRAME_RATIO))))
    temporal_summary = region_temporal_summary(temporal, region, roi_width, frame_count)
    return {
        "paired_count": int(len(paired)),
        "paired_ratio": len(paired) / max(frame_count * int(np.count_nonzero(mask)), 1),
        "delta_signed_px": finite_stats(paired),
        "delta_abs_px": finite_stats(np.abs(paired)),
        "delta_temporal_std_px": temporal_summary["temporal_std_px"],
        "delta_valid_frame_ratio": temporal_summary["valid_frame_ratio"],
    }


def sample_profile_aggregates(
    profile_rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    by_u: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        by_u[int(row["u_px"])].append(row)
    result: dict[int, dict[str, Any]] = {}
    for u_px, rows in by_u.items():
        result[u_px] = {
            "peak_dn_median": stats_value(finite_stats([float(row["peak_dn"]) for row in rows])),
            "background_dn_median": stats_value(finite_stats([float(row["background_dn"]) for row in rows])),
            "contrast_dn_median": stats_value(finite_stats([float(row["contrast_dn"]) for row in rows])),
            "fwhm_px_median": stats_value(finite_stats([float(row["fwhm_px"]) for row in rows])),
            "profile_saturation_rate_median": stats_value(
                finite_stats([float(row["profile_saturation_rate"]) for row in rows])
            ),
            "peak_saturated_fraction": float(
                np.count_nonzero([bool(row["peak_saturated"]) for row in rows]) / len(rows)
            ),
            "profile_count": len(rows),
        }
    return result


def build_column_rows(
    result: Mapping[str, Any],
    common_trend: np.ndarray,
    support_count: np.ndarray,
    low_frequency_sigma_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    roi = result["roi"]
    roi_width = int(roi["right"] - roi["left"])
    frame_count = int(result["frame_count"])
    minimum_frames = max(2, int(math.ceil(frame_count * MIN_TEMPORAL_FRAME_RATIO)))
    profile_by_u = sample_profile_aggregates(result["profile_rows"])
    vectors = result["vectors"]
    high_vectors: dict[str, np.ndarray] = {}
    residual_vectors: dict[str, np.ndarray] = {}
    all_frame_region: dict[str, dict[str, dict[str, Any]]] = {}
    for algorithm in ALGORITHMS:
        high_matrix = np.full(vectors[algorithm].shape, np.nan, dtype=np.float64)
        residual_matrix = np.full(vectors[algorithm].shape, np.nan, dtype=np.float64)
        for frame_index, values in enumerate(vectors[algorithm], start=1):
            residual, high = high_frequency_residual(
                values, common_trend, low_frequency_sigma_px
            )
            residual_matrix[frame_index - 1] = residual
            high_matrix[frame_index - 1] = high
        high_vectors[algorithm] = high_matrix
        residual_vectors[algorithm] = residual_matrix
        all_frame_region[algorithm] = {}
        for region in REGIONS:
            frame_metrics = per_frame_region_metrics(
                vectors[algorithm], high_matrix, region, roi_width
            )
            all_frame_region[algorithm][region] = {
                "frame_metrics": frame_metrics,
                "aggregate": aggregate_frame_metric_rows(frame_metrics),
                "temporal": region_temporal_summary(
                    temporal_column_metrics(vectors[algorithm], minimum_frames),
                    region,
                    roi_width,
                    frame_count,
                ),
            }

    delta_matrix = vectors["steger"] - vectors["centroid"]
    delta_temporal = temporal_column_metrics(delta_matrix, minimum_frames)
    temporal_by_algorithm = {
        algorithm: temporal_column_metrics(vectors[algorithm], minimum_frames)
        for algorithm in ALGORITHMS
    }
    high_by_algorithm = {
        algorithm: column_high_frequency_metrics(high_vectors[algorithm])
        for algorithm in ALGORITHMS
    }
    delta_by_region = {
        region: summarize_delta(delta_matrix, region, roi_width, frame_count)
        for region in REGIONS
    }

    rows: list[dict[str, Any]] = []
    exposure = float(result["exposure_us"])
    recording = str(result["recording"])
    for local_u in range(roi_width):
        global_u = int(roi["left"] + local_u)
        profile_item = profile_by_u.get(global_u, {})
        base = {
            "scope": "column",
            "recording": recording,
            "exposure_us": exposure,
            "frame_index": None,
            "u_px": global_u,
            "region": region_for_local_u(local_u, roi_width),
            "common_geometry_support_count": int(support_count[local_u]),
            "common_geometry_support_ratio": float(
                support_count[local_u] / max(sum(result["algorithm_frame_counts"]), 1)
            ),
            "peak_dn_median": profile_item.get("peak_dn_median"),
            "background_dn_median": profile_item.get("background_dn_median"),
            "contrast_dn_median": profile_item.get("contrast_dn_median"),
            "fwhm_px_median": profile_item.get("fwhm_px_median"),
            "profile_saturation_rate_median": profile_item.get(
                "profile_saturation_rate_median"
            ),
            "peak_saturated_fraction": profile_item.get("peak_saturated_fraction"),
        }
        for algorithm in ALGORITHMS:
            temporal = temporal_by_algorithm[algorithm]
            high = high_by_algorithm[algorithm]
            rows.append(
                {
                    **base,
                    "algorithm": algorithm,
                    "valid_frame_ratio": temporal["valid_frame_ratio"][local_u],
                    "temporal_std_px": temporal["temporal_std_px"][local_u],
                    "temporal_range_px": temporal["value_range_px"][local_u],
                    "high_frequency_rms_px": high["high_frequency_rms_px"][local_u],
                    "high_frequency_abs_p95_px": high["high_frequency_abs_p95_px"][local_u],
                    "delta_median_px": None,
                    "delta_abs_p50_px": None,
                    "delta_abs_p95_px": None,
                    "delta_temporal_std_px": None,
                }
            )
        delta_values = delta_matrix[:, local_u]
        finite_delta = delta_values[np.isfinite(delta_values)]
        rows.append(
            {
                **base,
                "algorithm": "delta_v",
                "valid_frame_ratio": float(
                    len(finite_delta) / max(frame_count, 1)
                ),
                "temporal_std_px": delta_temporal["temporal_std_px"][local_u],
                "temporal_range_px": delta_temporal["value_range_px"][local_u],
                "high_frequency_rms_px": None,
                "high_frequency_abs_p95_px": None,
                "delta_median_px": (
                    float(np.median(finite_delta)) if len(finite_delta) else float("nan")
                ),
                "delta_abs_p50_px": (
                    float(np.percentile(np.abs(finite_delta), 50)) if len(finite_delta) else float("nan")
                ),
                "delta_abs_p95_px": (
                    float(np.percentile(np.abs(finite_delta), 95)) if len(finite_delta) else float("nan")
                ),
                "delta_temporal_std_px": delta_temporal["temporal_std_px"][local_u],
            }
        )

    region_rows: list[dict[str, Any]] = []
    for region in REGIONS:
        profile_summary = aggregate_profile_rows(result["profile_rows"], region, roi_width)
        for algorithm in ALGORITHMS:
            item = all_frame_region[algorithm][region]
            region_rows.append(
                {
                    "scope": "region",
                    "recording": recording,
                    "exposure_us": exposure,
                    "frame_index": None,
                    "u_px": None,
                    "region": region,
                    "algorithm": algorithm,
                    "valid_frame_ratio": item["aggregate"]["valid_ratio"]["median"],
                    "temporal_std_px": item["temporal"]["temporal_std_px"]["median"],
                    "temporal_range_px": None,
                    "high_frequency_rms_px": item["aggregate"]["high_frequency_rms_px"]["median"],
                    "high_frequency_abs_p95_px": item["aggregate"]["high_frequency_abs_p95_px"]["median"],
                    "delta_median_px": None,
                    "delta_abs_p50_px": None,
                    "delta_abs_p95_px": None,
                    "delta_temporal_std_px": None,
                    "peak_dn_median": profile_summary["peak_dn"]["median"],
                    "background_dn_median": profile_summary["background_dn"]["median"],
                    "contrast_dn_median": profile_summary["contrast_dn"]["median"],
                    "fwhm_px_median": profile_summary["fwhm_px"]["median"],
                    "profile_saturation_rate_median": profile_summary[
                        "profile_saturation_rate"
                    ]["median"],
                    "peak_saturated_fraction": profile_summary[
                        "peak_saturated_fraction"
                    ],
                    "common_geometry_support_count": None,
                    "common_geometry_support_ratio": None,
                }
            )
        delta_item = delta_by_region[region]
        region_rows.append(
            {
                "scope": "region",
                "recording": recording,
                "exposure_us": exposure,
                "frame_index": None,
                "u_px": None,
                "region": region,
                "algorithm": "delta_v",
                "valid_frame_ratio": delta_item["paired_ratio"],
                "temporal_std_px": delta_item["delta_temporal_std_px"]["median"],
                "temporal_range_px": None,
                "high_frequency_rms_px": None,
                "high_frequency_abs_p95_px": None,
                "delta_median_px": delta_item["delta_signed_px"]["median"],
                "delta_abs_p50_px": delta_item["delta_abs_px"]["median"],
                "delta_abs_p95_px": delta_item["delta_abs_px"]["p95"],
                "delta_temporal_std_px": delta_item["delta_temporal_std_px"]["median"],
                "peak_dn_median": profile_summary["peak_dn"]["median"],
                "background_dn_median": profile_summary["background_dn"]["median"],
                "contrast_dn_median": profile_summary["contrast_dn"]["median"],
                "fwhm_px_median": profile_summary["fwhm_px"]["median"],
                "profile_saturation_rate_median": profile_summary[
                    "profile_saturation_rate"
                ]["median"],
                "peak_saturated_fraction": profile_summary["peak_saturated_fraction"],
                "common_geometry_support_count": None,
                "common_geometry_support_ratio": None,
            }
        )

    result["high_vectors"] = high_vectors
    result["residual_vectors"] = residual_vectors
    result["delta_matrix"] = delta_matrix
    result["column_rows"] = rows
    result["region_rows"] = region_rows
    result["region_profile_summary"] = {
        region: aggregate_profile_rows(result["profile_rows"], region, roi_width)
        for region in REGIONS
    }
    result["region_algorithm_summary"] = all_frame_region
    result["region_delta_summary"] = delta_by_region
    result["temporal_by_algorithm"] = temporal_by_algorithm
    result["high_by_algorithm"] = high_by_algorithm
    result["temporal_delta"] = delta_temporal
    return rows + region_rows, {
        "region_profile_summary": result["region_profile_summary"],
        "region_algorithm_summary": all_frame_region,
        "region_delta_summary": delta_by_region,
    }


def build_correlation_summary(
    result: Mapping[str, Any], sample_step_px: int
) -> dict[str, Any]:
    roi = result["roi"]
    roi_left = int(roi["left"])
    roi_width = int(roi["right"] - roi["left"])
    sample_columns = np.asarray(result["sample_columns"], dtype=np.int64)
    local_sample_columns = sample_columns - roi_left
    sample_lookup = sample_profile_aggregates(result["profile_rows"])
    clip = np.asarray(
        [sample_lookup[int(u)].get("peak_saturated_fraction", np.nan) for u in sample_columns],
        dtype=np.float64,
    )
    result_by_algorithm: dict[str, Any] = {}
    center_mask = np.asarray(
        [region_for_local_u(int(u), roi_width) == "center" for u in local_sample_columns],
        dtype=bool,
    )
    for algorithm in ALGORITHMS:
        high = result["high_vectors"][algorithm]
        high_rms = np.full(roi_width, np.nan, dtype=np.float64)
        high_p95 = np.full(roi_width, np.nan, dtype=np.float64)
        for local_u in range(roi_width):
            values = high[:, local_u]
            values = values[np.isfinite(values)]
            if len(values):
                high_rms[local_u] = float(np.sqrt(np.mean(values**2)))
                high_p95[local_u] = float(np.percentile(np.abs(values), 95))
        sampled_rms = high_rms[local_sample_columns]
        sampled_p95 = high_p95[local_sample_columns]
        center_clip = clip[center_mask]
        center_rms = sampled_rms[center_mask]
        center_p95 = sampled_p95[center_mask]
        clip_nonzero = center_clip > 0.0
        rms_top = center_rms >= np.nanpercentile(center_rms, 90) if np.any(np.isfinite(center_rms)) else np.zeros(center_rms.shape, dtype=bool)
        p95_top = center_p95 >= np.nanpercentile(center_p95, 90) if np.any(np.isfinite(center_p95)) else np.zeros(center_p95.shape, dtype=bool)
        result_by_algorithm[algorithm] = {
            "peak_saturated_fraction_vs_high_frequency_rms": correlation(center_clip, center_rms),
            "peak_saturated_fraction_vs_high_frequency_abs_p95": correlation(center_clip, center_p95),
            "peak_saturated_positions_vs_top10_high_frequency_rms": overlap_ratio(clip_nonzero, rms_top),
            "peak_saturated_positions_vs_top10_high_frequency_abs_p95": overlap_ratio(clip_nonzero, p95_top),
            "center_sample_count": int(np.count_nonzero(center_mask)),
            "center_peak_saturated_position_count": int(np.count_nonzero(clip_nonzero)),
        }

    delta_matrix = result["delta_matrix"]
    delta_abs_p95 = np.full(roi_width, np.nan, dtype=np.float64)
    for local_u in range(roi_width):
        values = delta_matrix[:, local_u]
        values = values[np.isfinite(values)]
        if len(values):
            delta_abs_p95[local_u] = float(np.percentile(np.abs(values), 95))
    sampled_delta = delta_abs_p95[local_sample_columns]
    center_delta = sampled_delta[center_mask]
    center_clip = clip[center_mask]
    delta_top = center_delta >= np.nanpercentile(center_delta, 90) if np.any(np.isfinite(center_delta)) else np.zeros(center_delta.shape, dtype=bool)
    return {
        "sample_step_px": sample_step_px,
        "sample_columns_px": [int(value) for value in sample_columns],
        "center_sample_count": int(np.count_nonzero(center_mask)),
        "center_peak_saturated_fraction_vs_delta_abs_p95": correlation(center_clip, center_delta),
        "peak_saturated_positions_vs_top10_delta_abs_p95": overlap_ratio(
            center_clip > 0.0, delta_top
        ),
        "by_algorithm": result_by_algorithm,
    }


def read_a1_reference(recording_dir: Path) -> dict[str, Any]:
    a1_dir = recording_dir / "steger_diagnostic"
    summary_path = a1_dir / "diagnostic_summary.json"
    if not summary_path.is_file():
        return {"available": False, "directory": a1_dir}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {"available": False, "directory": a1_dir, "error": str(error)}
    if not isinstance(summary, Mapping):
        return {"available": False, "directory": a1_dir, "error": "summary is not a mapping"}
    return {
        "available": True,
        "directory": a1_dir,
        "summary_path": summary_path,
        "classification": summary.get("classification"),
        "fwhm_median_px": summary.get("stripe_quality", {}).get("fwhm_stats", {}).get("median"),
        "current_sigma_match": summary.get("sigma_sweep", {}).get("recommendation", {}).get("current_sigma_match")
        if isinstance(summary.get("sigma_sweep"), Mapping)
        else None,
        "reused_for_calculation": False,
    }


def get_region_value(
    results: Sequence[Mapping[str, Any]], exposure: float, region: str, key_path: Sequence[str]
) -> float | None:
    candidates = [item for item in results if math.isclose(float(item["exposure_us"]), exposure, abs_tol=1.0e-9)]
    if not candidates:
        return None
    item: Any = candidates[0]
    for key in key_path:
        item = item.get(key) if isinstance(item, Mapping) else None
    if isinstance(item, Mapping):
        return stats_value(item)
    return finite_float(item)


def result_region_metric(
    result: Mapping[str, Any], region: str, algorithm: str, metric: str
) -> float | None:
    if algorithm == "profile":
        item = result["region_profile_summary"][region]
        if metric == "peak_dn":
            return stats_value(item["peak_dn"])
        if metric == "contrast_dn":
            return stats_value(item["contrast_dn"])
        if metric == "fwhm_px":
            return stats_value(item["fwhm_px"])
        if metric == "profile_saturation_rate":
            return stats_value(item["profile_saturation_rate"])
        if metric == "peak_saturated_fraction":
            return finite_float(item["peak_saturated_fraction"])
        if metric == "fwhm_valid_ratio":
            return finite_float(item["fwhm_valid_ratio"])
    elif algorithm in ALGORITHMS:
        item = result["region_algorithm_summary"][algorithm][region]
        if metric == "valid_ratio":
            return stats_value(item["aggregate"]["valid_ratio"])
        if metric == "high_frequency_rms_px":
            return stats_value(item["aggregate"]["high_frequency_rms_px"])
        if metric == "high_frequency_abs_p95_px":
            return stats_value(item["aggregate"]["high_frequency_abs_p95_px"])
        if metric == "temporal_std_px":
            return stats_value(item["temporal"]["temporal_std_px"])
        if metric == "temporal_valid_ratio":
            return stats_value(item["temporal"]["valid_frame_ratio"])
    elif algorithm == "delta_v":
        item = result["region_delta_summary"][region]
        if metric == "delta_abs_p50_px":
            return stats_value(item["delta_abs_px"])
        if metric == "delta_abs_p95_px":
            return stats_value(item["delta_abs_px"], "p95")
        if metric == "delta_temporal_std_px":
            return stats_value(item["delta_temporal_std_px"])
    return None


def extract_current_exposure(config: AppConfig) -> float | None:
    camera = getattr(config, "camera", None)
    if camera is None:
        return None
    return finite_float(getattr(camera, "exposure_us", None))


def choose_recommended_exposure(
    results: Sequence[Mapping[str, Any]], current_exposure: float | None
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: float(item["exposure_us"]))
    exposures = [float(item["exposure_us"]) for item in ordered]
    center_hf = [
        result_region_metric(item, "center", "steger", "high_frequency_rms_px")
        for item in ordered
    ]
    finite_hf = [value for value in center_hf if value is not None]
    best_hf = min(finite_hf) if finite_hf else None
    edge_valid_values = [
        min(
            result_region_metric(item, "left", "steger", "valid_ratio")
            if result_region_metric(item, "left", "steger", "valid_ratio") is not None
            else float("nan"),
            result_region_metric(item, "right", "steger", "valid_ratio")
            if result_region_metric(item, "right", "steger", "valid_ratio") is not None
            else float("nan"),
        )
        for item in ordered
    ]
    finite_edge_valid = np.asarray(
        [value for value in edge_valid_values if math.isfinite(value)], dtype=np.float64
    )
    plateau_values = finite_edge_valid[-min(3, len(finite_edge_valid)) :] if len(finite_edge_valid) else np.asarray([], dtype=np.float64)
    edge_valid_platform = float(np.median(plateau_values)) if len(plateau_values) else float("nan")
    edge_valid_floor = max(
        0.75,
        RECOMMEND_EDGE_VALID_PLATFORM_FRACTION * edge_valid_platform
        if math.isfinite(edge_valid_platform)
        else 0.75,
    )
    edge_contrast_values = [
        min(
            result_region_metric(item, "left", "profile", "contrast_dn")
            if result_region_metric(item, "left", "profile", "contrast_dn") is not None
            else float("nan"),
            result_region_metric(item, "right", "profile", "contrast_dn")
            if result_region_metric(item, "right", "profile", "contrast_dn") is not None
            else float("nan"),
        )
        for item in ordered
    ]
    finite_edge_contrast = [value for value in edge_contrast_values if math.isfinite(value)]
    edge_contrast_floor = max(
        RECOMMEND_EDGE_CONTRAST_MIN_DN,
        0.45 * max(finite_edge_contrast) if finite_edge_contrast else RECOMMEND_EDGE_CONTRAST_MIN_DN,
    )
    acceptable: list[dict[str, Any]] = []
    for item, hf in zip(ordered, center_hf, strict=True):
        center_clip = result_region_metric(item, "center", "profile", "peak_saturated_fraction")
        left_contrast = result_region_metric(item, "left", "profile", "contrast_dn")
        right_contrast = result_region_metric(item, "right", "profile", "contrast_dn")
        left_valid = result_region_metric(item, "left", "steger", "valid_ratio")
        right_valid = result_region_metric(item, "right", "steger", "valid_ratio")
        conditions = {
            "center_peak_saturation_le_25_percent": center_clip is not None and center_clip <= RECOMMEND_CENTER_PEAK_SATURATION_MAX,
            "both_edge_contrast_ge_recommendation_floor": left_contrast is not None and right_contrast is not None and min(left_contrast, right_contrast) >= edge_contrast_floor,
            "both_edge_steger_valid_ratio_ge_recommendation_floor": left_valid is not None and right_valid is not None and min(left_valid, right_valid) >= edge_valid_floor,
            "center_high_frequency_within_25_percent_of_best": hf is not None and best_hf is not None and hf <= best_hf * HF_ACCEPTABLE_RELATIVE_TO_BEST,
        }
        if all(conditions.values()):
            acceptable.append({"exposure_us": float(item["exposure_us"]), "conditions": conditions})

    if acceptable:
        recommended_values = [float(item["exposure_us"]) for item in acceptable]
        selection_method = "all exposures passing clipping, edge SNR, valid-ratio and high-frequency gates"
    else:
        # A fallback is needed when the measured data do not contain a value that
        # satisfies every conservative gate. It ranks a middle operating point by
        # normalized clipping, edge SNR and high-frequency terms, still reporting
        # the actual measured exposure values rather than inventing one.
        score_rows: list[tuple[float, float]] = []
        for item, hf in zip(ordered, center_hf, strict=True):
            clip = result_region_metric(item, "center", "profile", "peak_saturated_fraction") or 0.0
            edge_contrast = min(
                result_region_metric(item, "left", "profile", "contrast_dn") or 0.0,
                result_region_metric(item, "right", "profile", "contrast_dn") or 0.0,
            )
            edge_valid = min(
                result_region_metric(item, "left", "steger", "valid_ratio") or 0.0,
                result_region_metric(item, "right", "steger", "valid_ratio") or 0.0,
            )
            hf_term = (hf / best_hf) if hf is not None and best_hf and best_hf > 0 else 2.0
            score = 3.0 * clip + max(0.0, (MIN_PROFILE_CONTRAST_DN - edge_contrast) / 100.0) + max(0.0, 0.90 - edge_valid) + 0.25 * hf_term
            score_rows.append((score, float(item["exposure_us"])))
        score_rows.sort()
        best_value = score_rows[0][1]
        best_index = exposures.index(best_value)
        selected_indices = sorted({max(0, best_index - 1), best_index, min(len(exposures) - 1, best_index + 1)})
        recommended_values = [exposures[index] for index in selected_indices]
        selection_method = "fallback three-point neighborhood around the best measured balance score"

    current_in_recommendation = (
        current_exposure is not None
        and any(math.isclose(current_exposure, value, abs_tol=1.0e-9) for value in recommended_values)
    )
    selected_indices = sorted(
        exposures.index(value)
        for value in recommended_values
        if value in exposures
    )
    contiguous = (
        selected_indices == list(range(selected_indices[0], selected_indices[-1] + 1))
        if selected_indices
        else False
    )
    return {
        "recommended_exposure_values_us": recommended_values,
        "recommended_exposure_range_us": [min(recommended_values), max(recommended_values)] if recommended_values else None,
        "recommended_exposure_is_contiguous_in_measured_set": contiguous,
        "selection_method": selection_method,
        "current_exposure_us": current_exposure,
        "current_exposure_in_recommendation": current_in_recommendation,
        "best_center_high_frequency_rms_px": best_hf,
        "gates": {
            "center_peak_saturation_max_fraction": RECOMMEND_CENTER_PEAK_SATURATION_MAX,
            "edge_min_contrast_dn": edge_contrast_floor,
            "edge_valid_platform_median": edge_valid_platform,
            "edge_min_steger_valid_ratio": edge_valid_floor,
            "high_frequency_max_relative_to_best": HF_ACCEPTABLE_RELATIVE_TO_BEST,
        },
    }


def classify_exposure_effects(
    results: Sequence[Mapping[str, Any]],
    current_exposure: float | None,
    recommendation: Mapping[str, Any],
    correlation_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: float(item["exposure_us"]))
    exposures = np.asarray([float(item["exposure_us"]) for item in ordered], dtype=np.float64)

    def metric_or_nan(item: Mapping[str, Any], region: str, algorithm: str, metric: str) -> float:
        value = result_region_metric(item, region, algorithm, metric)
        return float(value) if value is not None and math.isfinite(value) else float("nan")

    center_clip = np.asarray(
        [metric_or_nan(item, "center", "profile", "peak_saturated_fraction") for item in ordered], dtype=np.float64
    )
    center_hf_s = np.asarray(
        [metric_or_nan(item, "center", "steger", "high_frequency_rms_px") for item in ordered], dtype=np.float64
    )
    center_hf_c = np.asarray(
        [metric_or_nan(item, "center", "centroid", "high_frequency_rms_px") for item in ordered], dtype=np.float64
    )
    edge_valid_s = np.asarray(
        [min(
            metric_or_nan(item, "left", "steger", "valid_ratio"),
            metric_or_nan(item, "right", "steger", "valid_ratio"),
        ) for item in ordered], dtype=np.float64
    )
    edge_contrast = np.asarray(
        [min(
            metric_or_nan(item, "left", "profile", "contrast_dn"),
            metric_or_nan(item, "right", "profile", "contrast_dn"),
        ) for item in ordered], dtype=np.float64
    )
    def trend(values: np.ndarray) -> dict[str, Any]:
        valid = np.isfinite(values)
        return correlation(exposures[valid], values[valid])

    current_item = None
    if current_exposure is not None:
        current_item = next(
            (item for item in ordered if math.isclose(float(item["exposure_us"]), current_exposure, abs_tol=1.0e-9)),
            None,
        )
    current_clip = result_region_metric(current_item, "center", "profile", "peak_saturated_fraction") if current_item else None
    current_hf = result_region_metric(current_item, "center", "steger", "high_frequency_rms_px") if current_item else None
    nonclipped_hf = [
        result_region_metric(item, "center", "steger", "high_frequency_rms_px")
        for item in ordered
        if (result_region_metric(item, "center", "profile", "peak_saturated_fraction") or 0.0) <= CENTER_PEAK_SATURATION_WARN
    ]
    nonclipped_hf = [value for value in nonclipped_hf if value is not None]
    best_nonclipped_hf = min(nonclipped_hf) if nonclipped_hf else None
    hf_reduction_from_current = (
        (current_hf - best_nonclipped_hf) / current_hf
        if current_hf is not None and best_nonclipped_hf is not None and current_hf > 0
        else None
    )
    low_edge_valid = float(np.nanmin(edge_valid_s)) if np.any(np.isfinite(edge_valid_s)) else None
    central_edge_valid = float(np.nanmedian(edge_valid_s)) if np.any(np.isfinite(edge_valid_s)) else None
    edge_valid_drop = (
        central_edge_valid - low_edge_valid
        if central_edge_valid is not None and low_edge_valid is not None
        else None
    )
    central_clipping_present = bool(
        np.any(np.isfinite(center_clip))
        and np.nanmax(center_clip) >= CENTER_PEAK_SATURATION_WARN
    )
    clipping_drives_center_fluctuation = bool(
        hf_reduction_from_current is not None
        and hf_reduction_from_current >= HF_RELATIVE_TRADEOFF
    )
    clipping_effect = bool(
        central_clipping_present
        and (
            clipping_drives_center_fluctuation
            or (current_clip is not None and current_clip >= CENTER_PEAK_SATURATION_SEVERE and recommendation.get("current_exposure_in_recommendation") is False)
        )
    )
    low_snr_effect = bool(
        (low_edge_valid is not None and low_edge_valid < MIN_EDGE_VALID_RATIO)
        or (edge_valid_drop is not None and edge_valid_drop >= 0.05)
        or (np.nanmin(edge_contrast) < MIN_PROFILE_CONTRAST_DN if np.any(np.isfinite(edge_contrast)) else False)
    )
    if clipping_effect and low_snr_effect:
        classification = "EXPOSURE_TRADEOFF"
    elif clipping_effect:
        classification = "EXPOSURE_CLIPPING_DOMINANT"
    elif low_snr_effect:
        classification = "LOW_EXPOSURE_SNR_LIMITED"
    else:
        classification = "EXPOSURE_NOT_PRIMARY"

    correlation_by_recording = {
        str(item["recording"]): item.get("center_peak_saturated_fraction_vs_delta_abs_p95")
        for item in correlation_summaries
    }
    return {
        "classification": classification,
        "evidence": {
            "center_peak_saturation_fraction_by_exposure": {
                str(float(exposure)): (float(value) if math.isfinite(float(value)) else None)
                for exposure, value in zip(exposures, center_clip, strict=True)
            },
            "center_steger_high_frequency_rms_by_exposure": {
                str(float(exposure)): (float(value) if math.isfinite(float(value)) else None)
                for exposure, value in zip(exposures, center_hf_s, strict=True)
            },
            "center_centroid_high_frequency_rms_by_exposure": {
                str(float(exposure)): (float(value) if math.isfinite(float(value)) else None)
                for exposure, value in zip(exposures, center_hf_c, strict=True)
            },
            "edge_steger_valid_ratio_min_by_exposure": {
                str(float(exposure)): (float(value) if math.isfinite(float(value)) else None)
                for exposure, value in zip(exposures, edge_valid_s, strict=True)
            },
            "edge_contrast_min_by_exposure": {
                str(float(exposure)): (float(value) if math.isfinite(float(value)) else None)
                for exposure, value in zip(exposures, edge_contrast, strict=True)
            },
            "current_exposure_us": current_exposure,
            "current_center_peak_saturation_fraction": current_clip,
            "current_center_steger_high_frequency_rms_px": current_hf,
            "best_nonclipped_center_steger_high_frequency_rms_px": best_nonclipped_hf,
            "center_high_frequency_reduction_current_to_best_nonclipped": hf_reduction_from_current,
            "edge_valid_ratio_median_across_exposures": central_edge_valid,
            "minimum_edge_valid_ratio": low_edge_valid,
            "minimum_edge_valid_ratio_drop": edge_valid_drop,
            "clipping_effect_detected": clipping_effect,
            "central_clipping_present": central_clipping_present,
            "clipping_drives_center_fluctuation": clipping_drives_center_fluctuation,
            "low_snr_effect_detected": low_snr_effect,
            "peak_saturation_delta_correlation_by_recording": correlation_by_recording,
            "trend_exposure_vs_center_steger_high_frequency": trend(center_hf_s),
            "trend_exposure_vs_center_centroid_high_frequency": trend(center_hf_c),
            "trend_exposure_vs_center_peak_saturation": trend(center_clip),
            "trend_exposure_vs_edge_valid_ratio": trend(edge_valid_s),
            "trend_exposure_vs_edge_contrast": trend(edge_contrast),
        },
        "thresholds": {
            "center_peak_saturation_warn_fraction": CENTER_PEAK_SATURATION_WARN,
            "center_peak_saturation_severe_fraction": CENTER_PEAK_SATURATION_SEVERE,
            "high_frequency_tradeoff_relative_change": HF_RELATIVE_TRADEOFF,
            "edge_valid_ratio_min": MIN_EDGE_VALID_RATIO,
            "edge_valid_ratio_drop_warn": 0.05,
            "edge_contrast_min_dn": MIN_PROFILE_CONTRAST_DN,
        },
    }


def make_region_metric_rows(
    results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: float(item["exposure_us"])):
        base = {
            "recording": result["recording"],
            "exposure_us": result["exposure_us"],
            "frame_count": result["frame_count"],
            "image_width_px": result["image_shape"][1],
            "image_height_px": result["image_shape"][0],
            "roi_left_px": result["roi"]["left"],
            "roi_top_px": result["roi"]["top"],
            "roi_width_px": result["roi"]["right"] - result["roi"]["left"],
            "roi_height_px": result["roi"]["bottom"] - result["roi"]["top"],
        }
        all_profile = result["region_profile_summary"]["all"]
        all_s = result["region_algorithm_summary"]["steger"]["all"]
        all_c = result["region_algorithm_summary"]["centroid"]["all"]
        all_delta = result["region_delta_summary"]["all"]
        summary_rows.append(
            {
                **base,
                "peak_dn_p50": all_profile["peak_dn"]["median"],
                "peak_dn_p05": all_profile["peak_dn"]["p05"],
                "peak_dn_p95": all_profile["peak_dn"]["p95"],
                "background_dn_p50": all_profile["background_dn"]["median"],
                "contrast_dn_p50": all_profile["contrast_dn"]["median"],
                "contrast_dn_p05": all_profile["contrast_dn"]["p05"],
                "fwhm_px_p50": all_profile["fwhm_px"]["median"],
                "fwhm_px_p05": all_profile["fwhm_px"]["p05"],
                "fwhm_px_p95": all_profile["fwhm_px"]["p95"],
                "fwhm_valid_ratio": all_profile["fwhm_valid_ratio"],
                "profile_saturation_rate_p50": all_profile["profile_saturation_rate"]["median"],
                "peak_saturated_fraction": all_profile["peak_saturated_fraction"],
                "steger_valid_ratio_p50": all_s["aggregate"]["valid_ratio"]["median"],
                "centroid_valid_ratio_p50": all_c["aggregate"]["valid_ratio"]["median"],
                "steger_high_frequency_rms_p50_px": all_s["aggregate"]["high_frequency_rms_px"]["median"],
                "centroid_high_frequency_rms_p50_px": all_c["aggregate"]["high_frequency_rms_px"]["median"],
                "steger_high_frequency_abs_p95_p50_px": all_s["aggregate"]["high_frequency_abs_p95_px"]["median"],
                "centroid_high_frequency_abs_p95_p50_px": all_c["aggregate"]["high_frequency_abs_p95_px"]["median"],
                "steger_temporal_std_p50_px": all_s["temporal"]["temporal_std_px"]["median"],
                "steger_temporal_std_p95_px": all_s["temporal"]["temporal_std_px"]["p95"],
                "centroid_temporal_std_p50_px": all_c["temporal"]["temporal_std_px"]["median"],
                "centroid_temporal_std_p95_px": all_c["temporal"]["temporal_std_px"]["p95"],
                "delta_abs_p50_px": all_delta["delta_abs_px"]["median"],
                "delta_abs_p95_px": all_delta["delta_abs_px"]["p95"],
                "delta_temporal_std_p50_px": all_delta["delta_temporal_std_px"]["median"],
                "delta_temporal_std_p95_px": all_delta["delta_temporal_std_px"]["p95"],
            }
        )
        for region in ("left", "center", "right"):
            profile = result["region_profile_summary"][region]
            s = result["region_algorithm_summary"]["steger"][region]
            c = result["region_algorithm_summary"]["centroid"][region]
            delta = result["region_delta_summary"][region]
            region_rows.append(
                {
                    **base,
                    "region": region,
                    "sample_count": profile["sample_count"],
                    "peak_dn_p50": profile["peak_dn"]["median"],
                    "peak_dn_p05": profile["peak_dn"]["p05"],
                    "peak_dn_p95": profile["peak_dn"]["p95"],
                    "background_dn_p50": profile["background_dn"]["median"],
                    "contrast_dn_p50": profile["contrast_dn"]["median"],
                    "contrast_dn_p05": profile["contrast_dn"]["p05"],
                    "fwhm_px_p50": profile["fwhm_px"]["median"],
                    "fwhm_px_p05": profile["fwhm_px"]["p05"],
                    "fwhm_px_p95": profile["fwhm_px"]["p95"],
                    "fwhm_valid_ratio": profile["fwhm_valid_ratio"],
                    "profile_saturation_rate_p50": profile["profile_saturation_rate"]["median"],
                    "peak_saturated_fraction": profile["peak_saturated_fraction"],
                    "steger_valid_ratio_p50": s["aggregate"]["valid_ratio"]["median"],
                    "centroid_valid_ratio_p50": c["aggregate"]["valid_ratio"]["median"],
                    "steger_high_frequency_rms_p50_px": s["aggregate"]["high_frequency_rms_px"]["median"],
                    "centroid_high_frequency_rms_p50_px": c["aggregate"]["high_frequency_rms_px"]["median"],
                    "steger_high_frequency_abs_p95_p50_px": s["aggregate"]["high_frequency_abs_p95_px"]["median"],
                    "centroid_high_frequency_abs_p95_p50_px": c["aggregate"]["high_frequency_abs_p95_px"]["median"],
                    "steger_temporal_std_p50_px": s["temporal"]["temporal_std_px"]["median"],
                    "steger_temporal_std_p95_px": s["temporal"]["temporal_std_px"]["p95"],
                    "centroid_temporal_std_p50_px": c["temporal"]["temporal_std_px"]["median"],
                    "centroid_temporal_std_p95_px": c["temporal"]["temporal_std_px"]["p95"],
                    "delta_abs_p50_px": delta["delta_abs_px"]["median"],
                    "delta_abs_p95_px": delta["delta_abs_px"]["p95"],
                    "delta_temporal_std_p50_px": delta["delta_temporal_std_px"]["median"],
                    "delta_temporal_std_p95_px": delta["delta_temporal_std_px"]["p95"],
                }
            )
    return summary_rows, region_rows


def series_for_plot(
    results: Sequence[Mapping[str, Any]], region: str, algorithm: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(results, key=lambda item: float(item["exposure_us"]))
    x: list[float] = []
    y: list[float] = []
    for item in ordered:
        value = result_region_metric(item, region, algorithm, metric)
        if value is not None and math.isfinite(value):
            x.append(float(item["exposure_us"]))
            y.append(value)
    return np.asarray(x), np.asarray(y)


def mark_regions(axis: Any) -> None:
    axis.grid(True, alpha=0.25)


def save_exposure_metric_plot(
    path: Path,
    results: Sequence[Mapping[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    algorithms: Sequence[str] = ("profile",),
    region_panels: bool = False,
    horizontal_lines: Sequence[tuple[float, str]] = (),
) -> None:
    if region_panels:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
        axes = np.asarray(axes).ravel()
        for axis, region in zip(axes, ("left", "center", "right"), strict=True):
            for algorithm in algorithms:
                x, y = series_for_plot(results, region, algorithm, metric)
                if len(x):
                    label = {"profile": "profile", "steger": "Steger", "centroid": "centroid"}.get(algorithm, algorithm)
                    axis.plot(x, y, marker="o", linewidth=1.5, label=label)
            for value, label in horizontal_lines:
                axis.axhline(value, color="0.45", linestyle=":", linewidth=0.9, label=label)
            axis.set_title(region)
            axis.set_xlabel("exposure (µs)")
            axis.set_ylabel(ylabel)
            mark_regions(axis)
            axis.legend(fontsize=8, loc="best")
        figure.suptitle(title)
        figure.tight_layout()
    else:
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for region in ("left", "center", "right"):
            x, y = series_for_plot(results, region, algorithms[0], metric)
            if len(x):
                axis.plot(x, y, marker="o", linewidth=1.5, label=region)
        for value, label in horizontal_lines:
            axis.axhline(value, color="0.45", linestyle=":", linewidth=0.9, label=label)
        axis.set_xlabel("exposure (µs)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        mark_regions(axis)
        axis.legend(fontsize=8, loc="best")
        figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_delta_plot(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    axes = np.asarray(axes).ravel()
    ordered = sorted(results, key=lambda item: float(item["exposure_us"]))
    for axis, region in zip(axes, ("left", "center", "right"), strict=True):
        exposures: list[float] = []
        p50: list[float] = []
        p95: list[float] = []
        for item in ordered:
            median = result_region_metric(item, region, "delta_v", "delta_abs_p50_px")
            high = result_region_metric(item, region, "delta_v", "delta_abs_p95_px")
            if median is not None:
                exposures.append(float(item["exposure_us"]))
                p50.append(median)
                p95.append(high if high is not None else np.nan)
        axis.plot(exposures, p50, marker="o", linewidth=1.5, label="|delta_v| P50")
        axis.plot(exposures, p95, marker="s", linewidth=1.2, label="|delta_v| P95")
        axis.set_title(region)
        axis.set_xlabel("exposure (µs)")
        axis.set_ylabel("Steger − centroid (px)")
        mark_regions(axis)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Steger–centroid center difference vs exposure")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_center_profiles_plot(
    path: Path, results: Sequence[Mapping[str, Any]], center_u: int, roi: Mapping[str, int]
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 9.0), sharex=True)
    full_axis, zoom_axis = np.asarray(axes).ravel()
    vertical = np.arange(int(roi["top"]), int(roi["bottom"]))
    peak_positions: list[float] = []
    for result in sorted(results, key=lambda item: float(item["exposure_us"])):
        profiles = np.asarray(result["center_profiles"], dtype=np.float64)
        if profiles.ndim != 2 or profiles.shape[0] == 0:
            continue
        median = np.median(profiles, axis=0)
        low = np.percentile(profiles, 10, axis=0)
        high = np.percentile(profiles, 90, axis=0)
        label = f"{float(result['exposure_us']):g} µs"
        peak_positions.append(float(vertical[int(np.argmax(median))]))
        (line,) = full_axis.plot(median, vertical, linewidth=1.5, label=label)
        full_axis.fill_betweenx(vertical, low, high, color=line.get_color(), alpha=0.08)
        zoom_axis.plot(median, vertical, linewidth=1.5, label=label)
        zoom_axis.fill_betweenx(vertical, low, high, color=line.get_color(), alpha=0.08)
    full_axis.set_ylabel("v (px)")
    full_axis.set_title(f"Representative raw vertical profiles at center u={center_u}px (full ROI)")
    full_axis.set_xlim(0, 260)
    full_axis.invert_yaxis()
    full_axis.grid(True, alpha=0.25)
    full_axis.legend(fontsize=8, loc="best", ncol=2)
    zoom_axis.set_xlabel("raw DN")
    zoom_axis.set_ylabel("v (px)")
    zoom_axis.set_title("Zoom around the representative laser ridge")
    zoom_axis.set_xlim(0, 260)
    if peak_positions:
        peak_v = float(np.median(peak_positions))
        zoom_axis.set_ylim(peak_v + 15.0, peak_v - 15.0)
    zoom_axis.grid(True, alpha=0.25)
    zoom_axis.legend(fontsize=8, loc="best", ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


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


def report_region_rows(results: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in sorted(results, key=lambda value: float(value["exposure_us"])):
        exposure = f"{float(item['exposure_us']):g}"
        for region in ("left", "center", "right"):
            profile = item["region_profile_summary"][region]
            steger = item["region_algorithm_summary"]["steger"][region]
            centroid = item["region_algorithm_summary"]["centroid"][region]
            delta = item["region_delta_summary"][region]
            rows.append(
                [
                    exposure,
                    region,
                    fmt(profile["peak_dn"]["median"]),
                    fmt(profile["contrast_dn"]["median"]),
                    fmt(profile["fwhm_px"]["median"]),
                    f"{100 * profile['peak_saturated_fraction']:.1f}%" if finite_float(profile["peak_saturated_fraction"]) is not None else "—",
                    fmt(steger["aggregate"]["valid_ratio"]["median"] * 100, 1),
                    fmt(steger["aggregate"]["high_frequency_rms_px"]["median"]),
                    fmt(centroid["aggregate"]["high_frequency_rms_px"]["median"]),
                    fmt(steger["temporal"]["temporal_std_px"]["median"]),
                    fmt(centroid["temporal"]["temporal_std_px"]["median"]),
                    fmt(delta["delta_abs_px"]["median"]),
                    fmt(delta["delta_abs_px"]["p95"]),
                ]
            )
    return rows


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    profile_roi: Mapping[str, int],
    current_exposure: float | None,
) -> None:
    classification = summary["classification"]
    recommendation = summary["recommendation"]
    effects = summary["exposure_effects"]
    a1 = summary["a1_reference"]
    included = summary["input"]["included_recordings"]
    region_table = report_region_rows(results)
    center_current = effects["evidence"].get("current_center_peak_saturation_fraction")
    current_hf = effects["evidence"].get("current_center_steger_high_frequency_rms_px")
    best_nonclipped_hf = effects["evidence"].get("best_nonclipped_center_steger_high_frequency_rms_px")
    reduction = effects["evidence"].get("center_high_frequency_reduction_current_to_best_nonclipped")
    edge_drop = effects["evidence"].get("minimum_edge_valid_ratio_drop")
    current_overexposed = summary["answers"]["current_config_exposure_overexposed"]
    correlation_table_rows: list[list[Any]] = []
    for item in sorted(
        summary["spatial_structure"]["correlation_by_recording"],
        key=lambda value: float(value["exposure_us"]),
    ):
        delta_correlation = item.get("center_peak_saturated_fraction_vs_delta_abs_p95", {})
        steger_correlation = item.get("by_algorithm", {}).get("steger", {}).get(
            "peak_saturated_fraction_vs_high_frequency_abs_p95", {}
        )
        overlap = item.get("peak_saturated_positions_vs_top10_delta_abs_p95", {})
        overlap_value = overlap.get("intersection_over_b")
        correlation_table_rows.append(
            [
                f"{float(item['exposure_us']):g}",
                item.get("center_sample_count"),
                fmt(delta_correlation.get("spearman")),
                fmt(steger_correlation.get("spearman")),
                f"{100 * overlap_value:.1f}%" if finite_float(overlap_value) is not None else "—",
            ]
        )
    lines = [
        "# 海康红光曝光敏感性与空间结构诊断报告",
        "",
        f"- 诊断分类：**`{classification}`**",
        f"- 生成时间（UTC）：`{summary['generated_at_utc']}`",
        f"- 纳入 recording：`{len(included)}` 组；每组约 20 帧；所有统计仅基于原始二维 Mono8 PNG。",
        f"- 固定 Steger：`sigma={summary['parameters']['steger_options_fixed']['sigma']}`，其余参数全部来自当前配置；没有 sweep sigma、threshold、deriv_thresh、曝光或 ROI。",
        f"- 当前 Steger profile ROI：`x={profile_roi['left']}:{profile_roi['right']}, y={profile_roi['top']}:{profile_roi['bottom']}`。",
        "",
        "## 1. 数据纳入与方法",
        "",
        "曝光由每个 recording 的 `frames.csv.exposure_us` 读取，没有按目录名或代码硬编码曝光值。"
        "只有 frames.csv 与 PNG 数量/顺序一致、同一 recording 内采集参数唯一、Mono8、帧数在约 20 帧范围内，"
        "并且除曝光外的采集参数与其它 recording 一致的数据被纳入。",
        "",
        "原始 profile 在当前 Steger search ROI 内按固定列步长采样；background 为纵向 profile 的 P20，"
        "contrast = peak − background，FWHM 为 background + 0.5×contrast 水平处的线性插值宽度，"
        "profile saturation rate 为纵向 profile 中 DN=255 的像素比例，peak saturated fraction 为 peak=255 的 profile 比例。",
        "",
        "空间结构不使用全局直线 RMS。先把所有曝光、两种算法、所有帧的中心按列取稳健中位数并用"
        f"{LOW_FREQUENCY_SIGMA_PX:g} px Gaussian 得到共同低频几何趋势 g(u)，再对每条 trace 计算"
        " `(v−g)−G((v−g))` 的局部高频残差；RMS/P95 均只在对应左/中/右区域和有效列上统计。",
        "",
        "## 2. 曝光 × 区域结果",
        "",
        markdown_table(
            [
                "曝光(µs)",
                "区域",
                "peak P50",
                "contrast P50",
                "FWHM P50(px)",
                "peak=255",
                "Steger有效率(%)",
                "Steger高频RMS",
                "centroid高频RMS",
                "Steger temporal std",
                "centroid temporal std",
                "|delta| P50",
                "|delta| P95",
            ],
            region_table,
        ),
        "",
        "说明：高频 RMS/P95 是去除共同低频几何趋势后的局部空间起伏；temporal std 是同一曝光同一列跨帧中心坐标的标准差（ddof=0），不是全局直线残差。",
        "",
        "## 3. 曝光趋势与饱和相关性",
        "",
        f"当前配置曝光：`{fmt(current_exposure, 1)} µs`。当前曝光中央 peak=255 比例：**{fmt(center_current * 100 if center_current is not None else None, 1)}%**；"
        f"中央 Steger 高频 RMS：**{fmt(current_hf)} px**。在中央 peak 饱和不超过 {CENTER_PEAK_SATURATION_WARN:.0%} 的实测曝光中，"
        f"最低中央 Steger 高频 RMS 为 **{fmt(best_nonclipped_hf)} px**，从当前曝光降低的相对变化为 **{fmt(reduction * 100 if reduction is not None else None, 1)}%**。",
        "",
        "每个曝光的 `peak=255` 空间位置与中央高频起伏、异常 `delta_v` 的 Pearson/Spearman 相关和 top-10% 重叠率已写入 `exposure_diagnostic_summary.json`；"
        "其含义是位置相关性证据，不把相关性自动解释为因果。",
        "",
        markdown_table(
            ["曝光(µs)", "中心采样列", "peak 饱和 vs |delta_v| P95 Spearman", "peak 饱和 vs Steger 高频 P95 Spearman", "饱和位置∩delta top10 / delta top10"],
            correlation_table_rows,
        ),
        "",
        "## 4. 判定与推荐工作区间",
        "",
        f"- 中央 clipping 存在：`{effects['evidence']['central_clipping_present']}`；"
        f"是否驱动中央高频起伏：`{effects['evidence']['clipping_drives_center_fluctuation']}`；"
        f"low-exposure SNR effect：`{effects['evidence']['low_snr_effect_detected']}`。",
        f"- 边缘最小 Steger 有效率的跨曝光下降量（相对跨曝光中位数）：**{fmt(edge_drop * 100 if edge_drop is not None else None, 1)} percentage points**。",
        f"- 推荐实测曝光点：**{', '.join(f'{value:g}' for value in recommendation['recommended_exposure_values_us'])} µs**；建议工作区间：**{recommendation['recommended_exposure_range_us'][0]:g}–{recommendation['recommended_exposure_range_us'][1]:g} µs**。",
        f"- 推荐方式：`{recommendation['selection_method']}`。当前曝光是否落在推荐集合：`{recommendation['current_exposure_in_recommendation']}`。",
        "",
        "## 5. 最后回答五个问题",
        "",
        f"1. 中央空间起伏是否随降低曝光明显减小？**{summary['answers']['center_fluctuation_reduces_when_lowering_exposure']}**。"
        f" 当前值 {fmt(current_hf)} px，对比低饱和区最佳值 {fmt(best_nonclipped_hf)} px；相对变化 {fmt(reduction * 100 if reduction is not None else None, 1)}%。",
        f"2. peak=255 与异常中心位置是否相关？**{summary['answers']['peak_255_correlated_with_center_anomaly']}**。"
        " 详细相关系数、样本数和重叠率见 JSON；相关分析使用中心固定采样列的 peak 饱和比例与逐列高频/|delta_v| P95。",
        f"3. 降低曝光是否会导致边缘有效率明显下降？**{summary['answers']['lower_exposure_reduces_edge_valid_ratio']}**。"
        f" 最小边缘 Steger 有效率 {fmt(effects['evidence']['minimum_edge_valid_ratio'] * 100 if effects['evidence']['minimum_edge_valid_ratio'] is not None else None, 1)}%，下降量 {fmt(edge_drop * 100 if edge_drop is not None else None, 1)} 个百分点。",
        f"4. 当前 {fmt(current_exposure, 1)} µs 是否过曝？**{current_overexposed}**。判据是中央 peak=255 比例达到 {fmt(center_current * 100 if center_current is not None else None, 1)}%，并结合高频起伏与推荐区间；这是二维 sensor clipping 判断，不是三维结果反推。",
        f"5. 推荐曝光工作区间：**{recommendation['recommended_exposure_range_us'][0]:g}–{recommendation['recommended_exposure_range_us'][1]:g} µs**，实测推荐点为 `{', '.join(f'{value:g}' for value in recommendation['recommended_exposure_values_us'])}` µs。",
        "",
        "## 6. A-1 参考与输出物",
        "",
        f"A-1 当前曝光诊断产物：`{a1.get('directory')}`；本轮仅读取其 summary 做 provenance 交叉核对，A-2 的 profile、中心、temporal 和曝光比较均重新从原始 PNG 计算。",
        "",
        "输出：`exposure_summary.csv`、`exposure_region_summary.csv`、`exposure_spatial_structure.csv`、"
        "`exposure_diagnostic_summary.json`、以及 `exposure_vs_peak.png`、`exposure_vs_saturation.png`、"
        "`exposure_vs_contrast.png`、`exposure_vs_fwhm.png`、`exposure_vs_high_frequency_rms.png`、"
        "`exposure_vs_valid_ratio.png`、`exposure_vs_steger_centroid_delta.png`、`center_profiles_by_exposure.png`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steger-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-step", type=positive_int, default=PROFILE_SAMPLE_STEP_PX)
    parser.add_argument("--min-frames", type=positive_int, default=MIN_FRAME_COUNT)
    parser.add_argument("--max-frames", type=positive_int, default=MAX_FRAME_COUNT)
    parser.add_argument("--low-frequency-sigma", type=positive_float, default=LOW_FREQUENCY_SIGMA_PX)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recordings_root = args.recordings_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    profile_path = args.steger_profile.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.min_frames > args.max_frames:
        raise DiagnosticError("--min-frames cannot exceed --max-frames")

    included, excluded = discover_recordings(
        recordings_root, int(args.min_frames), int(args.max_frames)
    )
    config, steger_options, centroid_options, profile_info = load_fixed_options(
        config_path, profile_path
    )

    first_shape: tuple[int, int] | None = None
    first_roi: dict[str, int] | None = None
    results: list[dict[str, Any]] = []
    all_column_rows: list[dict[str, Any]] = []

    for recording_number, recording_item in enumerate(included, start=1):
        print(
            f"[{recording_number}/{len(included)}] processing {recording_item['recording']} "
            f"({float(recording_item['exposure_us']):g} us, {recording_item['frame_count']} frames)",
            flush=True,
        )
        frame_paths = recording_item["frame_paths"]
        metadata = recording_item["metadata"]
        vectors: dict[str, list[np.ndarray]] = {algorithm: [] for algorithm in ALGORITHMS}
        profile_rows: list[dict[str, Any]] = []
        center_profiles: list[np.ndarray] = []
        image_shape: tuple[int, int] | None = None
        roi: dict[str, int] | None = None
        sample_columns: np.ndarray | None = None
        for frame_index, (frame_path, frame_metadata) in enumerate(
            zip(frame_paths, metadata, strict=True), start=1
        ):
            image = read_image(frame_path)
            if image_shape is None:
                image_shape = tuple(int(value) for value in image.shape)
                roi = parse_search_roi(steger_options, image_shape)
                first_shape = first_shape or image_shape
                first_roi = first_roi or dict(roi)
                roi_width = roi["right"] - roi["left"]
                sample_columns = np.arange(
                    roi["left"], roi["right"], int(args.sample_step), dtype=np.int64
                )
                if len(sample_columns) == 0:
                    raise DiagnosticError("no profile sample columns in search ROI")
            elif tuple(int(value) for value in image.shape) != image_shape:
                raise DiagnosticError(
                    f"image shape changed in {recording_item['recording']}: {image.shape} != {image_shape}"
                )
            assert roi is not None and sample_columns is not None and image_shape is not None
            csv_width = frame_metadata.get("width")
            csv_height = frame_metadata.get("height")
            if csv_width and int(float(csv_width)) != image_shape[1]:
                raise DiagnosticError(f"frames.csv width mismatch: {frame_path}")
            if csv_height and int(float(csv_height)) != image_shape[0]:
                raise DiagnosticError(f"frames.csv height mismatch: {frame_path}")

            for u_px in sample_columns:
                measured = measure_profile(image, int(u_px), roi)
                profile_rows.append(
                    {
                        "recording": recording_item["recording"],
                        "exposure_us": recording_item["exposure_us"],
                        "frame_index": frame_index,
                        "frame_filename": frame_path.name,
                        "u_px": int(u_px),
                        "region": region_for_local_u(int(u_px - roi["left"]), roi_width),
                        **measured,
                    }
                )
            center_u = int(roi["left"] + (roi["right"] - roi["left"]) // 2)
            center_profiles.append(
                image[roi["top"] : roi["bottom"], center_u].astype(np.float64, copy=True)
            )

            try:
                steger_points = np.asarray(
                    steger_backend(image, steger_options), dtype=np.float64
                )
                centroid_points = np.asarray(
                    centroid_backend(image, centroid_options), dtype=np.float64
                )
            except Exception as error:  # noqa: BLE001 - include frame provenance in error
                raise DiagnosticError(
                    f"backend extraction failed for {recording_item['recording']} {frame_path.name}: {error}"
                ) from error
            full_steger = points_to_vector(steger_points, image_shape[1])
            full_centroid = points_to_vector(centroid_points, image_shape[1])
            vectors["steger"].append(full_steger[roi["left"] : roi["right"]])
            vectors["centroid"].append(full_centroid[roi["left"] : roi["right"]])

        assert image_shape is not None and roi is not None and sample_columns is not None
        result = {
            "recording": recording_item["recording"],
            "recording_dir": recording_item["path"],
            "exposure_us": float(recording_item["exposure_us"]),
            "frame_count": len(frame_paths),
            "algorithm_frame_counts": [len(frame_paths), len(frame_paths)],
            "image_shape": image_shape,
            "roi": roi,
            "sample_columns": sample_columns,
            "profile_rows": profile_rows,
            "vectors": {
                algorithm: np.asarray(values, dtype=np.float64)
                for algorithm, values in vectors.items()
            },
            "center_profiles": np.asarray(center_profiles, dtype=np.float64),
            "capture_values": recording_item["unique_values"],
        }
        results.append(result)
        print(
            f"[{recording_number}/{len(included)}] extracted "
            f"Steger={np.count_nonzero(np.isfinite(result['vectors']['steger']))} "
            f"centroid={np.count_nonzero(np.isfinite(result['vectors']['centroid']))} columns",
            flush=True,
        )

    if first_shape is None or first_roi is None:
        raise DiagnosticError("no images were processed")
    roi_width = first_roi["right"] - first_roi["left"]
    common_trend, common_support = build_common_geometry_trend(
        [
            result["vectors"][algorithm]
            for result in results
            for algorithm in ALGORITHMS
        ],
        float(args.low_frequency_sigma),
    )
    correlation_summaries: list[dict[str, Any]] = []
    for result in results:
        rows, _details = build_column_rows(
            result,
            common_trend,
            common_support,
            float(args.low_frequency_sigma),
        )
        all_column_rows.extend(rows)
        correlation_summary = build_correlation_summary(result, int(args.sample_step))
        correlation_summary["recording"] = result["recording"]
        correlation_summary["exposure_us"] = result["exposure_us"]
        correlation_summaries.append(correlation_summary)

    summary_rows, region_summary_rows = make_region_metric_rows(results)
    current_exposure = extract_current_exposure(config)
    recommendation = choose_recommended_exposure(results, current_exposure)
    exposure_effects = classify_exposure_effects(
        results, current_exposure, recommendation, correlation_summaries
    )

    current_item = next(
        (
            item
            for item in results
            if current_exposure is not None
            and math.isclose(float(item["exposure_us"]), current_exposure, abs_tol=1.0e-9)
        ),
        None,
    )
    current_clip = (
        result_region_metric(current_item, "center", "profile", "peak_saturated_fraction")
        if current_item
        else None
    )
    current_hf = (
        result_region_metric(current_item, "center", "steger", "high_frequency_rms_px")
        if current_item
        else None
    )
    best_nonclipped = exposure_effects["evidence"].get(
        "best_nonclipped_center_steger_high_frequency_rms_px"
    )
    reduction = exposure_effects["evidence"].get(
        "center_high_frequency_reduction_current_to_best_nonclipped"
    )
    edge_drop = exposure_effects["evidence"].get("minimum_edge_valid_ratio_drop")
    corr_has_positive_anomaly = False
    for item in correlation_summaries:
        delta_corr = item.get("center_peak_saturated_fraction_vs_delta_abs_p95", {})
        if (delta_corr.get("spearman") or 0.0) >= 0.3:
            corr_has_positive_anomaly = True
    answers = {
        "center_fluctuation_reduces_when_lowering_exposure": (
            "YES" if reduction is not None and reduction >= HF_RELATIVE_TRADEOFF else "NO_OR_WEAK"
        ),
        "peak_255_correlated_with_center_anomaly": (
            "YES" if corr_has_positive_anomaly else "NO_OR_WEAK"
        ),
        "lower_exposure_reduces_edge_valid_ratio": (
            "YES" if edge_drop is not None and edge_drop >= 0.05 else "NO_OR_WEAK"
        ),
        "current_config_exposure_overexposed": (
            "YES" if current_clip is not None and current_clip >= CENTER_PEAK_SATURATION_WARN else "NO"
        ),
        "recommended_exposure_range_us": recommendation["recommended_exposure_range_us"],
    }
    a1_reference = read_a1_reference(
        next(
            (
                item["recording_dir"]
                for item in results
                if current_exposure is not None
                and math.isclose(
                    float(item["exposure_us"]), current_exposure, abs_tol=1.0e-9
                )
            ),
            results[0]["recording_dir"],
        )
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "task": "hikrobot_red_exposure_sensitivity_and_spatial_structure",
        "classification": exposure_effects["classification"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input": {
            "recordings_root": recordings_root,
            "included_recordings": [
                {
                    "recording": item["recording"],
                    "directory": item["path"],
                    "exposure_us": item["exposure_us"],
                    "frame_count": item["frame_count"],
                    "capture_values": item["unique_values"],
                }
                for item in included
            ],
            "excluded_recordings": excluded,
            "same_non_exposure_capture_signature": included[0]["common_signature"],
            "image_shape": first_shape,
            "dtype": "uint8",
            "profile_sample_step_px": int(args.sample_step),
            "profile_sample_count": len(results[0]["sample_columns"]),
        },
        "provenance": {
            "calculation_source": "raw Mono8 PNG and frames.csv only",
            "three_dimensional_results_used": False,
            "config_path": config_path,
            "config_sha256": sha256_file(config_path),
            "profile_path": profile_path,
            "profile_sha256": sha256_file(profile_path),
            "config_referenced_profile_path": profile_info["config_referenced_profile"],
            "config_referenced_profile_sha256": (
                sha256_file(profile_info["config_referenced_profile"])
                if profile_info["config_referenced_profile"] is not None
                and profile_info["config_referenced_profile"].is_file()
                else None
            ),
            "implementation_sha256": {
                "laser_backends.py": sha256_file(TOOL_ROOT / "laser" / "backends.py"),
                "laser_realtime_steger.py": sha256_file(TOOL_ROOT / "laser" / "realtime_steger.py"),
            },
            "production_configuration_modified": False,
            "production_code_parameters_modified": False,
            "a1_reference": a1_reference,
        },
        "parameters": {
            "steger_options_fixed": steger_options,
            "centroid_options_fixed": centroid_options,
            "current_sigma_px": CURRENT_SIGMA_PX,
            "profile_background_percentile": PROFILE_BACKGROUND_PERCENTILE,
            "saturation_dn": SATURATION_DN,
            "low_frequency_geometry_sigma_px": float(args.low_frequency_sigma),
            "minimum_temporal_frame_ratio": MIN_TEMPORAL_FRAME_RATIO,
            "approximate_frame_count_policy": {
                "expected": EXPECTED_FRAME_COUNT,
                "min": int(args.min_frames),
                "max": int(args.max_frames),
            },
        },
        "common_geometry_trend": {
            "roi_left_px": first_roi["left"],
            "roi_right_px": first_roi["right"],
            "support_count_min": int(np.min(common_support)),
            "support_count_median": int(np.median(common_support)),
            "support_count_max": int(np.max(common_support)),
            "support_ratio_min": float(np.min(common_support) / max(2 * sum(item["frame_count"] for item in results), 1)),
            "support_ratio_median": float(np.median(common_support) / max(2 * sum(item["frame_count"] for item in results), 1)),
        },
        "exposure_summary": summary_rows,
        "exposure_region_summary": region_summary_rows,
        "spatial_structure": {
            "correlation_by_recording": correlation_summaries,
            "column_row_count": sum(len(item["column_rows"]) for item in results),
            "region_row_count": sum(len(item["region_rows"]) for item in results),
        },
        "recommendation": recommendation,
        "exposure_effects": exposure_effects,
        "answers": answers,
        "a1_reference": a1_reference,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    region_fields = list(region_summary_rows[0].keys()) if region_summary_rows else []
    spatial_fields = [
        "scope",
        "recording",
        "exposure_us",
        "frame_index",
        "u_px",
        "region",
        "algorithm",
        "common_geometry_support_count",
        "common_geometry_support_ratio",
        "peak_dn_median",
        "background_dn_median",
        "contrast_dn_median",
        "fwhm_px_median",
        "profile_saturation_rate_median",
        "peak_saturated_fraction",
        "valid_frame_ratio",
        "temporal_std_px",
        "temporal_range_px",
        "high_frequency_rms_px",
        "high_frequency_abs_p95_px",
        "delta_median_px",
        "delta_abs_p50_px",
        "delta_abs_p95_px",
        "delta_temporal_std_px",
    ]
    write_csv(output_dir / "exposure_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "exposure_region_summary.csv", region_summary_rows, region_fields)
    # build_column_rows returns both column and region rows; do not append the
    # region rows a second time here.
    write_csv(output_dir / "exposure_spatial_structure.csv", all_column_rows, spatial_fields)
    (output_dir / "exposure_diagnostic_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    save_exposure_metric_plot(
        output_dir / "exposure_vs_peak.png",
        results,
        "peak_dn",
        "peak DN (P50)",
        "Raw profile peak vs exposure",
    )
    save_exposure_metric_plot(
        output_dir / "exposure_vs_saturation.png",
        results,
        "peak_saturated_fraction",
        "peak=255 fraction",
        "Peak clipping vs exposure",
        horizontal_lines=((CENTER_PEAK_SATURATION_WARN, "10% warning"),),
    )
    save_exposure_metric_plot(
        output_dir / "exposure_vs_contrast.png",
        results,
        "contrast_dn",
        "local contrast DN (P50)",
        "Raw profile contrast vs exposure",
        horizontal_lines=((MIN_PROFILE_CONTRAST_DN, "20 DN minimum"),),
    )
    save_exposure_metric_plot(
        output_dir / "exposure_vs_fwhm.png",
        results,
        "fwhm_px",
        "FWHM (px, P50)",
        "Raw profile FWHM vs exposure",
    )
    save_exposure_metric_plot(
        output_dir / "exposure_vs_high_frequency_rms.png",
        results,
        "high_frequency_rms_px",
        "local high-frequency RMS (px, P50)",
        "Local high-frequency spatial structure vs exposure",
        algorithms=("steger", "centroid"),
        region_panels=True,
    )
    save_exposure_metric_plot(
        output_dir / "exposure_vs_valid_ratio.png",
        results,
        "valid_ratio",
        "valid ratio (P50)",
        "Valid ratio vs exposure",
        algorithms=("steger", "centroid"),
        region_panels=True,
    )
    save_delta_plot(output_dir / "exposure_vs_steger_centroid_delta.png", results)
    save_center_profiles_plot(
        output_dir / "center_profiles_by_exposure.png",
        results,
        int(first_roi["left"] + roi_width // 2),
        first_roi,
    )
    write_report(
        output_dir / "exposure_diagnostic_report.md",
        summary,
        results,
        first_roi,
        current_exposure,
    )
    print(f"included recordings: {len(results)}")
    print(
        "exposures (us): "
        + ", ".join(f"{float(item['exposure_us']):g}" for item in sorted(results, key=lambda value: float(value["exposure_us"])))
    )
    print(f"classification: {exposure_effects['classification']}")
    print(
        "recommended exposure range (us): "
        + ("—" if recommendation["recommended_exposure_range_us"] is None else f"{recommendation['recommended_exposure_range_us'][0]:g}-{recommendation['recommended_exposure_range_us'][1]:g}")
    )
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"diagnostic error: {error}", file=sys.stderr)
        raise SystemExit(2)
