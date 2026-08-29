#!/usr/bin/env python3
"""海康红光中心线空间结构拆解。

本脚本只读取同一批 20 帧原始 Mono8 图像，并在当前 sigma=1.5 下：

* 重新计算当前 Steger 与 centroid 的逐列中心；
* 统计逐列 delta_v、局部跳变和跨帧固定性；
* 用低频 Gaussian trend / 高频 residual 拆解空间结构；
* 自动选择 delta、跳变、中心高频和边缘低对比度异常列；
* 为每个异常列保存完整原始 DN profile，并叠加两种中心位置；
* 输出 CSV、JSON、Markdown 与无 GUI PNG 图像。

A-1 的六个产物会先被校验并作为 provenance / 候选选择的输入，但中心线和
profile 数值均从本脚本读取的原始图像重新计算。脚本不会修改生产配置、标定
文件或生产算法参数，也不会进行 sigma sweep。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402


TOOL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOL_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_RECORDING_DIR = (
    TOOL_ROOT
    / "output_haikang_0828"
    / "online_recordings"
    / "recording_20260828_164418"
)
DEFAULT_CONFIG_PATH = TOOL_ROOT / "configs" / "measure_tool.yaml"
DEFAULT_PROFILE_PATH = WORKSPACE_ROOT / "calibration" / "config" / "realtime_steger.yaml"
DEFAULT_A1_DIRNAME = "steger_diagnostic"
DEFAULT_OUTPUT_DIRNAME = "steger_spatial_diagnostic"

CURRENT_SIGMA = 1.5
EXPECTED_FRAMES = 20
PROFILE_BACKGROUND_PERCENTILE = 20.0
PROFILE_SATURATION_DN = 255.0
DECOMPOSITION_GAUSSIAN_SIGMA_PX = 32.0
HIGH_RESIDUAL_SIGNIFICANCE_PX = 0.15
HIGH_SYNC_MIN_ABS_PX = 0.10
STEGER_JUMP_EVIDENCE_PX = 1.0
EDGE_RELATIVE_CONTRAST_FRACTION = 0.60
DELTA_THRESHOLDS_PX = (0.1, 0.3, 0.5)
MIN_FIXED_FRAME_RATIO = 0.50
MAX_PROFILE_CANDIDATES = 28
POSITION_SELECTION_MIN_SEPARATION_PX = 24

if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import AppConfig, load_app_config  # noqa: E402
from laser import realtime_steger  # noqa: E402
from laser.backends import centroid_backend, steger_backend  # noqa: E402


class DiagnosticError(RuntimeError):
    """输入、配置或诊断协议不满足要求。"""


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是整数") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
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
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.10g}"
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
        raise DiagnosticError(f"缺少 CSV 产物: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DiagnosticError(f"CSV 没有表头: {path}")
        return [dict(row) for row in reader]


def read_image_unicode(path: Path) -> np.ndarray:
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DiagnosticError(f"无法解码图像: {path}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise DiagnosticError(
            f"图像必须是单通道 Mono8: {path}, shape={image.shape}, dtype={image.dtype}"
        )
    return np.ascontiguousarray(image)


def read_frame_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DiagnosticError(f"缺少 frames.csv: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DiagnosticError(f"frames.csv 没有表头: {path}")
        return [dict(row) for row in reader]


def discover_frames(
    recording_dir: Path, expected_count: int
) -> tuple[list[Path], list[dict[str, str]]]:
    frame_paths = sorted(recording_dir.glob("frame_*.png"))
    if len(frame_paths) != expected_count:
        raise DiagnosticError(
            f"期望 {expected_count} 个 frame_*.png，实际 {len(frame_paths)} 个"
        )
    metadata = read_frame_metadata(recording_dir / "frames.csv")
    if len(metadata) != expected_count:
        raise DiagnosticError(
            f"期望 frames.csv 有 {expected_count} 条数据，实际 {len(metadata)} 条"
        )
    names = [row.get("filename", "") for row in metadata]
    actual_names = [path.name for path in frame_paths]
    if names != actual_names or len(set(names)) != len(names):
        raise DiagnosticError("frames.csv 的 filename 顺序、数量或唯一性不一致")
    return frame_paths, metadata


def load_frames(
    frame_paths: Sequence[Path], metadata: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    shape: tuple[int, int] | None = None
    for index, (path, row) in enumerate(zip(frame_paths, metadata, strict=True), start=1):
        image = read_image_unicode(path)
        if shape is None:
            shape = image.shape
        if image.shape != shape:
            raise DiagnosticError(
                f"图像尺寸不一致: {path.name}={image.shape}, expected={shape}"
            )
        if row.get("width") and int(row["width"]) != image.shape[1]:
            raise DiagnosticError(f"frames.csv width 与图像不一致: {path.name}")
        if row.get("height") and int(row["height"]) != image.shape[0]:
            raise DiagnosticError(f"frames.csv height 与图像不一致: {path.name}")
        frames.append(
            {
                "index": index,
                "path": path,
                "filename": path.name,
                "metadata": dict(row),
                "image": image,
            }
        )
    return frames


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticError(f"YAML 文件不存在: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise DiagnosticError(f"无法读取 YAML: {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise DiagnosticError(f"YAML 根节点必须是 mapping: {path}")
    return dict(document)


def profile_path_declared_by_config(config_path: Path) -> Path | None:
    document = load_yaml_mapping(config_path)
    extraction = document.get("extraction", {})
    if not isinstance(extraction, Mapping):
        raise DiagnosticError("配置 extraction 段必须是 mapping")
    value = extraction.get("profile")
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise DiagnosticError("extraction.profile 必须是路径字符串")
    return (config_path.resolve().parent / value).resolve()


def load_profile_options(path: Path) -> dict[str, Any]:
    document = load_yaml_mapping(path)
    options = document.get("steger", document.get("options", {}))
    if not isinstance(options, Mapping):
        raise DiagnosticError(f"Steger profile 缺少 steger mapping: {path}")
    return dict(options)


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values: Iterable[Any]) -> dict[str, Any]:
    array = finite_values(values)
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


def column_stat_array(matrix: np.ndarray, percentile: float) -> np.ndarray:
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        values = values[np.isfinite(values)]
        if values.size:
            result[column] = float(np.percentile(values, percentile))
    return result


def column_count_array(matrix: np.ndarray) -> np.ndarray:
    return np.count_nonzero(np.isfinite(matrix), axis=0).astype(np.int64)


def _parse_roi(value: Any, image_shape: tuple[int, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise DiagnosticError("steger.search_roi 必须是 mapping")
    required = ("offset_x", "offset_y", "width", "height")
    if set(value) != set(required):
        raise DiagnosticError(
            f"steger.search_roi 字段必须严格为 {sorted(required)}，实际 {sorted(value)}"
        )
    parsed: dict[str, int] = {}
    for key in required:
        raw = value[key]
        if isinstance(raw, bool):
            raise DiagnosticError(f"steger.search_roi.{key} 不能是 bool")
        try:
            parsed[key] = int(raw)
        except (TypeError, ValueError) as error:
            raise DiagnosticError(f"steger.search_roi.{key} 必须是整数") from error
    if min(parsed.values()) < 0 or min(parsed["width"], parsed["height"]) <= 0:
        raise DiagnosticError("steger.search_roi 偏移必须非负且宽高必须为正")
    height, width = image_shape
    left = max(0, parsed["offset_x"])
    top = max(0, parsed["offset_y"])
    right = min(width, parsed["offset_x"] + parsed["width"])
    bottom = min(height, parsed["offset_y"] + parsed["height"])
    if right <= left or bottom <= top:
        raise DiagnosticError("steger.search_roi 与输入图像没有有效交集")
    return {
        "configured_left": parsed["offset_x"],
        "configured_top": parsed["offset_y"],
        "configured_width": parsed["width"],
        "configured_height": parsed["height"],
        "effective_left": left,
        "effective_top": top,
        "effective_right": right,
        "effective_bottom": bottom,
    }


def points_to_vector(points: np.ndarray, image_width: int) -> np.ndarray:
    vector = np.full(image_width, np.nan, dtype=np.float64)
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return vector
    array = array.reshape((-1, 2))
    columns = np.rint(array[:, 0]).astype(np.int64)
    valid = (columns >= 0) & (columns < image_width) & np.isfinite(array[:, 1])
    grouped: dict[int, list[float]] = defaultdict(list)
    for column, value in zip(columns[valid], array[valid, 1], strict=True):
        grouped[int(column)].append(float(value))
    for column, values in grouped.items():
        vector[column] = float(np.median(values))
    return vector


def max_point_delta(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape((-1, 2))
    second = np.asarray(second, dtype=np.float64).reshape((-1, 2))
    if len(first) != len(second):
        return None
    if len(first) == 0:
        return 0.0
    first_order = np.lexsort((first[:, 1], first[:, 0]))
    second_order = np.lexsort((second[:, 1], second[:, 0]))
    delta = np.abs(first[first_order] - second[second_order])
    return float(np.max(delta))


def effective_steger_call(
    image: np.ndarray, options: Mapping[str, Any]
) -> dict[str, Any]:
    """使用与 backends.steger_backend 相同的 ROI 语义，同时保留 diagnostic。"""

    resolved = dict(options)
    search_roi_value = resolved.pop("search_roi", None)
    resolved.pop("_image_offset", None)
    if str(resolved.get("scan_axis", "column")) != "column":
        raise DiagnosticError("A-2 当前仅支持 scan_axis=column")
    image_height, image_width = image.shape
    if search_roi_value is None:
        left = top = 0
        cropped = image
        search_region = None
        expected_scanline_count = image_width
        extracted = realtime_steger.extract_steger(
            cropped,
            resolved,
            diagnostic=True,
            use_auto_band=True,
        )
    else:
        roi = _parse_roi(search_roi_value, image.shape)
        left = roi["effective_left"]
        top = roi["effective_top"]
        right = roi["effective_right"]
        bottom = roi["effective_bottom"]
        cropped = np.ascontiguousarray(image[top:bottom, left:right])
        search_region = realtime_steger.LaserSearchRegion(
            0, cropped.shape[0], "configured_search_roi"
        )
        expected_scanline_count = cropped.shape[1]
        extracted = realtime_steger.extract_steger(
            cropped,
            resolved,
            search_region=search_region,
            diagnostic=True,
            use_auto_band=False,
        )

    local_indices = np.flatnonzero(extracted.valid & np.isfinite(extracted.v_px))
    local_u = extracted.u_px[local_indices]
    local_v = extracted.v_px[local_indices]
    direct_points = np.column_stack(
        [local_u + left, local_v + top]
    ).astype(np.float64)
    direct_v = np.full(image_width, np.nan, dtype=np.float64)
    direct_columns = np.rint(local_u).astype(np.int64) + left
    in_bounds = (direct_columns >= 0) & (direct_columns < image_width)
    direct_v[direct_columns[in_bounds]] = local_v[in_bounds] + top

    wrapper_points = np.asarray(
        steger_backend(image, options), dtype=np.float64
    ).reshape((-1, 2))
    wrapper_v = points_to_vector(wrapper_points, image_width)

    diagnostic_arrays: dict[str, np.ndarray] = {}
    diagnostic_object = getattr(extracted, "diagnostics", None)
    diagnostic_fields = (
        "full_image_max_intensity_dn",
        "max_intensity_dn",
        "max_ridge_response",
        "min_subpixel_offset_px",
        "intensity_peak_present",
        "intensity_peak_outside_detected_band",
        "derivative_condition_passed",
        "ridge_response_passed",
        "subpixel_offset_passed",
        "accepted",
        "rejection_reason",
    )
    if diagnostic_object is not None:
        for field_name in diagnostic_fields:
            values = np.asarray(getattr(diagnostic_object, field_name))
            if values.dtype.kind in "b":
                target = np.zeros(image_width, dtype=bool)
            elif values.dtype.kind in "SU":
                target = np.full(image_width, "", dtype=values.dtype)
            else:
                target = np.full(image_width, np.nan, dtype=np.float64)
            local_columns = np.arange(values.shape[0], dtype=np.int64) + left
            valid_columns = (local_columns >= 0) & (local_columns < image_width)
            target[local_columns[valid_columns]] = values[valid_columns]
            diagnostic_arrays[field_name] = target

    response = np.full(image_width, np.nan, dtype=np.float64)
    offset = np.full(image_width, np.nan, dtype=np.float64)
    response[direct_columns[in_bounds]] = extracted.response[local_indices][in_bounds]
    offset[direct_columns[in_bounds]] = extracted.offset_px[local_indices][in_bounds]
    return {
        "points": wrapper_points,
        "direct_points": direct_points,
        "v_by_u": wrapper_v,
        "direct_v_by_u": direct_v,
        "response_by_u": response,
        "offset_by_u": offset,
        "valid_column_count": int(len(wrapper_points)),
        "direct_valid_column_count": int(len(direct_points)),
        "expected_scanline_count": int(expected_scanline_count),
        "diagnostic_arrays": diagnostic_arrays,
        "crop": {
            "left": int(left),
            "top": int(top),
            "width": int(cropped.shape[1]),
            "height": int(cropped.shape[0]),
        },
        "backend_max_abs_delta_px": max_point_delta(direct_points, wrapper_points),
        "backend_point_count_equal": bool(len(direct_points) == len(wrapper_points)),
    }


def local_jump_vector(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid_pair = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    differences = np.abs(np.diff(values))
    pair_indices = np.flatnonzero(valid_pair)
    for index in pair_indices:
        value = float(differences[index])
        if not math.isfinite(result[index]) or value > result[index]:
            result[index] = value
        if not math.isfinite(result[index + 1]) or value > result[index + 1]:
            result[index + 1] = value
    return result


def decompose_vector(
    values: np.ndarray, gaussian_sigma_px: float
) -> tuple[np.ndarray, np.ndarray, int]:
    valid_indices = np.flatnonzero(np.isfinite(values))
    low = np.full(values.shape[0], np.nan, dtype=np.float64)
    high = np.full(values.shape[0], np.nan, dtype=np.float64)
    if len(valid_indices) < 3:
        return low, high, 0
    first = int(valid_indices[0])
    last = int(valid_indices[-1])
    support = np.arange(first, last + 1, dtype=np.int64)
    filled = np.interp(support, valid_indices, values[valid_indices])
    trend = gaussian_filter1d(
        filled,
        sigma=gaussian_sigma_px,
        mode="nearest",
        truncate=4.0,
    )
    low[valid_indices] = trend[valid_indices - first]
    high[valid_indices] = values[valid_indices] - trend[valid_indices - first]
    internal_gap_count = int(np.count_nonzero(np.diff(valid_indices) > 1))
    return low, high, internal_gap_count


def region_for_u(u_px: int, image_width: int) -> str:
    if u_px < image_width / 3.0:
        return "left"
    if u_px < image_width * 2.0 / 3.0:
        return "center"
    return "right"


def region_mask(region: str, image_width: int) -> np.ndarray:
    columns = np.arange(image_width)
    if region == "left":
        return columns < image_width / 3.0
    if region == "center":
        return (columns >= image_width / 3.0) & (columns < image_width * 2.0 / 3.0)
    if region == "right":
        return columns >= image_width * 2.0 / 3.0
    return np.ones(image_width, dtype=bool)


def paired_correlation(
    first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None
) -> float | None:
    if mask is None:
        mask = np.ones(first.shape, dtype=bool)
    valid = np.isfinite(first) & np.isfinite(second) & mask
    first_values = first[valid]
    second_values = second[valid]
    if len(first_values) < 3:
        return None
    if np.std(first_values) <= np.finfo(float).eps or np.std(second_values) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(first_values, second_values)[0, 1])


def same_direction_ratio(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray | None = None,
    minimum_abs: float = HIGH_SYNC_MIN_ABS_PX,
) -> tuple[float | None, int]:
    if mask is None:
        mask = np.ones(first.shape, dtype=bool)
    valid = (
        np.isfinite(first)
        & np.isfinite(second)
        & mask
        & (np.abs(first) >= minimum_abs)
        & (np.abs(second) >= minimum_abs)
    )
    if not np.any(valid):
        return None, 0
    product = first[valid] * second[valid]
    return float(np.count_nonzero(product >= 0.0) / len(product)), int(len(product))


def low_span_by_frame(matrix: np.ndarray, region: str, image_width: int) -> np.ndarray:
    mask = region_mask(region, image_width)
    spans: list[float] = []
    for row in matrix:
        values = row[mask]
        values = values[np.isfinite(values)]
        if len(values) >= 3:
            spans.append(float(np.percentile(values, 95) - np.percentile(values, 5)))
    return np.asarray(spans, dtype=np.float64)


def decomposition_region_summary(
    name: str,
    low_matrix: np.ndarray,
    high_matrix: np.ndarray,
    image_width: int,
) -> dict[str, Any]:
    mask = region_mask(name, image_width)
    high_values = high_matrix[:, mask].ravel()
    high_values = high_values[np.isfinite(high_values)]
    low_values = low_matrix[:, mask].ravel()
    low_values = low_values[np.isfinite(low_values)]
    spans = low_span_by_frame(low_matrix, name, image_width)
    low_trend_stats = stats(low_values)
    low_trend_amplitude = (
        float(np.percentile(low_values, 95) - np.percentile(low_values, 5))
        if len(low_values)
        else None
    )
    return {
        "region": name,
        "low_trend_pooled_px": low_trend_stats,
        "low_trend_amplitude_pooled_px": low_trend_amplitude,
        "low_trend_span_by_frame_px": stats(spans),
        "high_residual_abs_px": stats(np.abs(high_values)),
        "high_residual_signed_px": stats(high_values),
        "high_residual_rms_px": (
            float(np.sqrt(np.mean(high_values**2))) if len(high_values) else None
        ),
    }


def decomposition_summary(
    steger_low: np.ndarray,
    steger_high: np.ndarray,
    centroid_low: np.ndarray,
    centroid_high: np.ndarray,
    image_width: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("left", "center", "right", "all"):
        s_summary = decomposition_region_summary(
            name, steger_low, steger_high, image_width
        )
        c_summary = decomposition_region_summary(
            name, centroid_low, centroid_high, image_width
        )
        mask = region_mask(name, image_width)
        s_high = steger_high[:, mask]
        c_high = centroid_high[:, mask]
        correlation = paired_correlation(s_high.ravel(), c_high.ravel())
        direction, direction_count = same_direction_ratio(
            s_high.ravel(), c_high.ravel()
        )
        result[name] = {
            "steger": s_summary,
            "centroid": c_summary,
            "high_frequency_paired_correlation": correlation,
            "high_frequency_same_direction_ratio": direction,
            "high_frequency_same_direction_count": direction_count,
        }
    return result


def load_a1_artifacts(
    recording_dir: Path, image_shape: tuple[int, int], expected_count: int
) -> dict[str, Any]:
    a1_dir = recording_dir / DEFAULT_A1_DIRNAME
    names = (
        "stripe_quality_by_u.csv",
        "sigma_sweep.csv",
        "temporal_repeatability.csv",
        "centroid_vs_steger.csv",
        "diagnostic_summary.json",
        "diagnostic_report.md",
    )
    paths = {name: a1_dir / name for name in names}
    for path in paths.values():
        if not path.is_file():
            raise DiagnosticError(f"缺少 A-1 产物: {path}")

    try:
        a1_summary = json.loads(
            paths["diagnostic_summary.json"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosticError("无法读取 A-1 diagnostic_summary.json") from error
    if not isinstance(a1_summary, Mapping):
        raise DiagnosticError("A-1 diagnostic_summary.json 根节点必须是 mapping")
    input_summary = a1_summary.get("input", {})
    if input_summary.get("frame_count") != expected_count:
        raise DiagnosticError("A-1 frame_count 与当前 20 帧协议不一致")
    if tuple(input_summary.get("image_size", ())) != (image_shape[1], image_shape[0]):
        raise DiagnosticError("A-1 image_size 与当前原始帧不一致")
    current_sigma = parse_float(
        a1_summary.get("parameters", {}).get("current_sigma_px")
    )
    if not math.isclose(current_sigma, CURRENT_SIGMA, rel_tol=0.0, abs_tol=1.0e-9):
        raise DiagnosticError("A-1 current sigma 与 A-2 sigma=1.5 不一致")

    quality_rows = read_csv(paths["stripe_quality_by_u.csv"])
    sigma_rows = read_csv(paths["sigma_sweep.csv"])
    temporal_rows = read_csv(paths["temporal_repeatability.csv"])
    comparison_rows = read_csv(paths["centroid_vs_steger.csv"])
    if len(quality_rows) != expected_count * 153:
        raise DiagnosticError("A-1 stripe_quality_by_u.csv 行数不是 20×153")
    sigma_values = [parse_float(row.get("sigma_px")) for row in sigma_rows]
    if sigma_values != [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        raise DiagnosticError("A-1 sigma_sweep.csv 不是预期的 7 组 sigma")
    if len(temporal_rows) != image_shape[1] * 2:
        raise DiagnosticError("A-1 temporal_repeatability.csv 行数与全宽双算法不一致")
    if len(comparison_rows) < expected_count * 2:
        raise DiagnosticError("A-1 centroid_vs_steger.csv 缺少逐帧 A/B 数据")
    report_text = paths["diagnostic_report.md"].read_text(encoding="utf-8")
    if not report_text.strip():
        raise DiagnosticError("A-1 diagnostic_report.md 为空")

    return {
        "directory": a1_dir,
        "paths": paths,
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "summary": dict(a1_summary),
        "quality_rows": quality_rows,
        "sigma_rows": sigma_rows,
        "temporal_rows": temporal_rows,
        "comparison_rows": comparison_rows,
    }


def aggregate_a1_quality(
    quality_rows: Sequence[Mapping[str, str]], image_width: int
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    by_u: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in quality_rows:
        by_u[int(float(row["u_px"]))].append(row)
    sample_columns = sorted(by_u)
    quality: dict[int, dict[str, Any]] = {}
    for u_px, rows in by_u.items():
        contrasts = [parse_float(row.get("local_contrast_dn")) for row in rows]
        peaks = [parse_float(row.get("peak_dn")) for row in rows]
        fwhm = [parse_float(row.get("fwhm_px")) for row in rows]
        saturation = [
            str(row.get("peak_saturated", "")).strip().lower() == "true"
            for row in rows
        ]
        quality[u_px] = {
            "a1_sample_count": len(rows),
            "a1_peak_dn_median": float(np.nanmedian(peaks)),
            "a1_contrast_median": float(np.nanmedian(contrasts)),
            "a1_low_contrast_ratio": float(
                np.count_nonzero(np.asarray(contrasts) < 20.0) / len(rows)
            ),
            "a1_fwhm_median": (
                float(np.nanmedian(fwhm)) if np.any(np.isfinite(fwhm)) else None
            ),
            "a1_fwhm_valid_ratio": float(
                np.count_nonzero(np.isfinite(fwhm)) / len(rows)
            ),
            "a1_peak_saturation_ratio": float(np.count_nonzero(saturation) / len(rows)),
        }
    return np.asarray(sample_columns, dtype=np.int64), quality


def nearest_quality_arrays(
    sample_columns: np.ndarray,
    quality: Mapping[int, Mapping[str, Any]],
    image_width: int,
) -> dict[str, np.ndarray]:
    arrays = {
        "sample_u": np.full(image_width, np.nan, dtype=np.float64),
        "peak_dn_median": np.full(image_width, np.nan, dtype=np.float64),
        "contrast_median": np.full(image_width, np.nan, dtype=np.float64),
        "low_contrast_ratio": np.full(image_width, np.nan, dtype=np.float64),
        "fwhm_median": np.full(image_width, np.nan, dtype=np.float64),
        "fwhm_valid_ratio": np.full(image_width, np.nan, dtype=np.float64),
        "peak_saturation_ratio": np.full(image_width, np.nan, dtype=np.float64),
    }
    if len(sample_columns) == 0:
        return arrays
    for u_px in range(image_width):
        insertion = int(np.searchsorted(sample_columns, u_px))
        candidates = []
        if insertion < len(sample_columns):
            candidates.append(int(sample_columns[insertion]))
        if insertion > 0:
            candidates.append(int(sample_columns[insertion - 1]))
        sample_u = min(candidates, key=lambda value: abs(value - u_px))
        item = quality[sample_u]
        arrays["sample_u"][u_px] = sample_u
        arrays["peak_dn_median"][u_px] = parse_float(item["a1_peak_dn_median"])
        arrays["contrast_median"][u_px] = parse_float(item["a1_contrast_median"])
        arrays["low_contrast_ratio"][u_px] = parse_float(item["a1_low_contrast_ratio"])
        arrays["fwhm_median"][u_px] = parse_float(item["a1_fwhm_median"])
        arrays["fwhm_valid_ratio"][u_px] = parse_float(item["a1_fwhm_valid_ratio"])
        arrays["peak_saturation_ratio"][u_px] = parse_float(
            item["a1_peak_saturation_ratio"]
        )
    return arrays


def profile_fwhm(
    profile: np.ndarray, peak_index: int, level: float
) -> tuple[float | None, float | None, float | None]:
    if peak_index <= 0 or peak_index >= len(profile) - 1:
        return None, None, None
    left = peak_index
    while left > 0 and profile[left] >= level:
        left -= 1
    if left == 0 and profile[left] >= level:
        return None, None, None
    left_fraction = (
        (level - profile[left]) / (profile[left + 1] - profile[left])
        if profile[left + 1] != profile[left]
        else 0.5
    )
    left_cross = left + float(np.clip(left_fraction, 0.0, 1.0))

    right = peak_index
    while right < len(profile) - 1 and profile[right] >= level:
        right += 1
    if right == len(profile) - 1 and profile[right] >= level:
        return None, None, None
    right_fraction = (
        (level - profile[right - 1]) / (profile[right] - profile[right - 1])
        if profile[right] != profile[right - 1]
        else 0.5
    )
    right_cross = right - 1 + float(np.clip(right_fraction, 0.0, 1.0))
    return right_cross - left_cross, peak_index - left_cross, right_cross - peak_index


def profile_metrics(
    image: np.ndarray, u_px: int, profile_top: int, profile_bottom: int
) -> dict[str, Any]:
    profile = image[profile_top:profile_bottom, u_px].astype(np.float64)
    v_axis = np.arange(profile_top, profile_bottom, dtype=np.float64)
    peak_index = int(np.argmax(profile))
    peak_dn = float(profile[peak_index])
    background_dn = float(np.percentile(profile, PROFILE_BACKGROUND_PERCENTILE))
    contrast_dn = peak_dn - background_dn
    half_level = background_dn + 0.5 * contrast_dn
    fwhm, left_half_width, right_half_width = profile_fwhm(
        profile, peak_index, half_level
    )
    positive_signal = np.clip(profile - background_dn, 0.0, None)
    signal_sum = float(np.sum(positive_signal))
    if signal_sum > 0.0:
        weighted_mean = float(np.sum(v_axis * positive_signal) / signal_sum)
        weighted_std = float(
            np.sqrt(np.sum(((v_axis - weighted_mean) ** 2) * positive_signal) / signal_sum)
        )
        weighted_skew = float(
            np.sum(((v_axis - weighted_mean) / max(weighted_std, 1.0e-9)) ** 3 * positive_signal)
            / signal_sum
        )
    else:
        weighted_skew = 0.0

    prominence = max(2.0, 0.08 * contrast_dn)
    height = background_dn + max(1.0, 0.20 * contrast_dn)
    if contrast_dn > 0.0:
        peaks, properties = find_peaks(
            profile,
            height=height,
            prominence=prominence,
            distance=2,
            plateau_size=(1, None),
        )
        peaks = np.asarray(peaks, dtype=np.int64)
    else:
        peaks = np.empty(0, dtype=np.int64)
        properties = {}
    if contrast_dn > 0.0 and peak_index not in set(peaks.tolist()):
        peaks = np.sort(np.append(peaks, peak_index))
    peak_positions = (peaks + profile_top).astype(np.float64)
    peak_spacings = np.diff(peak_positions)
    asymmetry_index = (
        abs(float(left_half_width) - float(right_half_width))
        / max(float(left_half_width) + float(right_half_width), 1.0e-9)
        if left_half_width is not None and right_half_width is not None
        else None
    )
    profile_saturated_pixel_count = int(
        np.count_nonzero(profile >= PROFILE_SATURATION_DN)
    )
    peak_saturated = bool(peak_dn >= PROFILE_SATURATION_DN)
    low_contrast = bool(contrast_dn < 20.0)
    multi_peak = bool(len(peaks) >= 2)
    asymmetric = bool(
        (asymmetry_index is not None and asymmetry_index > 0.25)
        or abs(weighted_skew) > 0.50
    )
    labels: list[str] = []
    if len(peaks) == 1:
        labels.append("single_peak")
    if multi_peak:
        labels.append("multi_peak")
    if peak_saturated:
        labels.append("saturated_peak")
    if low_contrast:
        labels.append("low_contrast")
    if asymmetric and not low_contrast:
        labels.append("asymmetric_peak")
    normal = bool(
        len(peaks) == 1
        and not peak_saturated
        and not low_contrast
        and not asymmetric
    )
    if normal:
        labels.append("normal")
    if low_contrast:
        primary = "low_contrast"
    elif multi_peak:
        primary = "multi_peak"
    elif peak_saturated:
        primary = "saturated_peak"
    elif asymmetric:
        primary = "asymmetric_peak"
    elif len(peaks) == 1:
        primary = "single_peak"
    else:
        primary = "normal"
    return {
        "profile": profile,
        "v_axis": v_axis,
        "peak_v_px": float(profile_top + peak_index),
        "peak_dn": peak_dn,
        "background_dn": background_dn,
        "contrast_dn": contrast_dn,
        "half_max_level_dn": half_level,
        "fwhm_px": fwhm,
        "left_half_width_px": left_half_width,
        "right_half_width_px": right_half_width,
        "peak_count": int(len(peaks)),
        "peak_positions_v_px": peak_positions,
        "peak_spacing_px": peak_spacings,
        "minimum_peak_spacing_px": (
            float(np.min(peak_spacings)) if len(peak_spacings) else None
        ),
        "profile_saturated_pixel_count": profile_saturated_pixel_count,
        "profile_saturation_rate": float(
            profile_saturated_pixel_count / max(len(profile), 1)
        ),
        "peak_saturated": peak_saturated,
        "weighted_skew": weighted_skew,
        "asymmetry_index": asymmetry_index,
        "morphology_labels": labels,
        "primary_morphology": primary,
    }


def select_positions(
    values: np.ndarray,
    eligible: np.ndarray,
    *,
    descending: bool,
    count: int,
    min_separation_px: int = POSITION_SELECTION_MIN_SEPARATION_PX,
) -> list[int]:
    indexes = np.flatnonzero(eligible & np.isfinite(values))
    if descending:
        order = indexes[np.argsort(values[indexes])[::-1]]
    else:
        order = indexes[np.argsort(values[indexes])]
    selected: list[int] = []
    for index in order:
        index_int = int(index)
        if all(abs(index_int - previous) >= min_separation_px for previous in selected):
            selected.append(index_int)
            if len(selected) >= count:
                break
    return selected


def choose_frame_for_candidate(
    u_px: int,
    reasons: Sequence[str],
    delta_matrix: np.ndarray,
    jump_matrix: np.ndarray,
    steger_high_matrix: np.ndarray,
    profile_contrasts: np.ndarray,
) -> tuple[int, str]:
    priority = (
        ("delta_peak", delta_matrix[:, u_px], True),
        ("steger_local_jump", jump_matrix[:, u_px], True),
        ("center_high_frequency", steger_high_matrix[:, u_px], True),
        ("edge_low_contrast", profile_contrasts, False),
    )
    for reason, values, descending in priority:
        if reason not in reasons:
            continue
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        ranking_values = np.abs(values) if descending else values
        ranking_values = np.where(finite, ranking_values, -np.inf if descending else np.inf)
        frame_index = int(np.argmax(ranking_values) if descending else np.argmin(ranking_values))
        if finite[frame_index]:
            return frame_index, reason
    for frame_index in range(delta_matrix.shape[0]):
        if math.isfinite(profile_contrasts[frame_index]):
            return frame_index, "fallback"
    return 0, "fallback"


def merge_intervals(mask: np.ndarray, max_gap_px: int = 8) -> list[dict[str, Any]]:
    indexes = np.flatnonzero(mask)
    if len(indexes) == 0:
        return []
    intervals: list[tuple[int, int]] = []
    start = previous = int(indexes[0])
    for index in indexes[1:]:
        current = int(index)
        if current - previous > max_gap_px + 1:
            intervals.append((start, previous))
            start = current
        previous = current
    intervals.append((start, previous))
    result = []
    for start, end in intervals:
        count = int(np.count_nonzero(mask[start : end + 1]))
        result.append(
            {
                "u_start_px": start,
                "u_end_px": end,
                "span_px": int(end - start + 1),
                "true_column_count": count,
                "image_width_fraction": float(count / len(mask)),
            }
        )
    return result


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def mark_regions(axis: Any, image_width: int) -> None:
    one_third = image_width / 3.0
    two_third = image_width * 2.0 / 3.0
    axis.axvline(one_third, color="0.45", linestyle="--", linewidth=0.8)
    axis.axvline(two_third, color="0.45", linestyle="--", linewidth=0.8)


def plot_delta(
    path: Path,
    image_width: int,
    delta_matrix: np.ndarray,
    candidate_u: Sequence[int],
) -> None:
    x = np.arange(image_width)
    delta_median = column_stat_array(delta_matrix, 50)
    delta_p05 = column_stat_array(delta_matrix, 5)
    delta_p95 = column_stat_array(delta_matrix, 95)
    abs_delta = np.abs(delta_matrix)
    ratios = [
        np.divide(
            np.count_nonzero(abs_delta > threshold, axis=0),
            np.maximum(np.count_nonzero(np.isfinite(delta_matrix), axis=0), 1),
        )
        for threshold in DELTA_THRESHOLDS_PX
    ]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x, delta_median, color="tab:red", linewidth=1.0, label="median delta_v")
    axes[0].fill_between(
        x, delta_p05, delta_p95, color="tab:red", alpha=0.18, label="P05–P95"
    )
    for threshold in DELTA_THRESHOLDS_PX:
        axes[0].axhline(threshold, color="0.55", linestyle=":", linewidth=0.7)
        axes[0].axhline(-threshold, color="0.55", linestyle=":", linewidth=0.7)
    for u_px in candidate_u:
        axes[0].axvline(u_px, color="black", alpha=0.22, linewidth=0.7)
    axes[0].set_ylabel("delta_v (px)")
    axes[0].set_title("Steger - centroid delta_v by u · 20 frames")
    axes[0].legend(loc="upper right")
    for threshold, ratio in zip(DELTA_THRESHOLDS_PX, ratios, strict=True):
        axes[1].plot(x, ratio, linewidth=1.0, label=f"|delta_v|>{threshold:g} px")
    for u_px in candidate_u:
        axes[1].axvline(u_px, color="black", alpha=0.22, linewidth=0.7)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("frame ratio")
    axes[1].set_xlabel("u (px)")
    axes[1].legend(loc="upper right")
    mark_regions(axes[0], image_width)
    mark_regions(axes[1], image_width)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_decomposition(
    path: Path,
    image_width: int,
    steger_vectors: np.ndarray,
    centroid_vectors: np.ndarray,
    steger_low: np.ndarray,
    centroid_low: np.ndarray,
    steger_high: np.ndarray,
    centroid_high: np.ndarray,
) -> None:
    x = np.arange(image_width)
    median_s = column_stat_array(steger_vectors, 50)
    median_c = column_stat_array(centroid_vectors, 50)
    median_s_low = column_stat_array(steger_low, 50)
    median_c_low = column_stat_array(centroid_low, 50)
    median_s_high = column_stat_array(steger_high, 50)
    median_c_high = column_stat_array(centroid_high, 50)
    s_high_p95 = column_stat_array(np.abs(steger_high), 95)
    c_high_p95 = column_stat_array(np.abs(centroid_high), 95)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(x, median_s, color="tab:red", linewidth=0.9, label="Steger median v")
    axes[0].plot(
        x, median_s_low, color="tab:red", linestyle="--", linewidth=1.2, label="Steger low trend"
    )
    axes[0].plot(
        x, median_c, color="tab:blue", linewidth=0.8, alpha=0.8, label="centroid median v"
    )
    axes[0].plot(
        x, median_c_low, color="tab:blue", linestyle="--", linewidth=1.2, label="centroid low trend"
    )
    axes[0].set_ylabel("v (px)")
    axes[0].set_title(
        f"Low/high spatial decomposition · Gaussian trend sigma={DECOMPOSITION_GAUSSIAN_SIGMA_PX:g} px"
    )
    axes[0].legend(loc="best", ncol=2)
    axes[1].plot(x, median_s_high, color="tab:red", linewidth=0.9, label="Steger high residual")
    axes[1].plot(
        x, median_c_high, color="tab:blue", linewidth=0.9, label="centroid high residual"
    )
    axes[1].axhline(0.0, color="0.5", linewidth=0.7)
    axes[1].set_ylabel("high residual (px)")
    axes[1].legend(loc="best")
    axes[2].plot(x, s_high_p95, color="tab:red", linewidth=0.9, label="Steger |high| P95")
    axes[2].plot(x, c_high_p95, color="tab:blue", linewidth=0.9, label="centroid |high| P95")
    axes[2].axhline(
        HIGH_RESIDUAL_SIGNIFICANCE_PX,
        color="0.45",
        linestyle=":",
        linewidth=0.8,
        label=f"evidence {HIGH_RESIDUAL_SIGNIFICANCE_PX:g} px",
    )
    axes[2].set_ylabel("|high| P95 (px)")
    axes[2].set_xlabel("u (px)")
    axes[2].legend(loc="best")
    for axis in axes:
        mark_regions(axis, image_width)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_abnormal_locations(
    path: Path,
    image_width: int,
    delta_abs_p95: np.ndarray,
    jump_p95: np.ndarray,
    high_abs_p50: np.ndarray,
    edge_quality: np.ndarray,
    masks: Mapping[str, np.ndarray],
    candidate_records: Sequence[Mapping[str, Any]],
) -> None:
    x = np.arange(image_width)
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(x, delta_abs_p95, color="tab:red", linewidth=0.9)
    axes[0].axhline(0.3, color="0.45", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("|delta| P95")
    axes[1].plot(x, jump_p95, color="tab:purple", linewidth=0.9)
    axes[1].axhline(STEGER_JUMP_EVIDENCE_PX, color="0.45", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("jump P95 (px)")
    axes[2].plot(x, high_abs_p50, color="tab:green", linewidth=0.9)
    axes[2].axhline(
        HIGH_RESIDUAL_SIGNIFICANCE_PX, color="0.45", linestyle=":", linewidth=0.8
    )
    axes[2].set_ylabel("high |res| P50")
    axes[3].plot(x, edge_quality, color="tab:orange", linewidth=0.9)
    axes[3].set_ylabel("A-1 contrast DN")
    axes[3].set_xlabel("u (px)")
    colors = {
        "edge_optical": "tab:orange",
        "raw_structure": "tab:green",
        "ridge_selection": "tab:red",
    }
    for name, mask in masks.items():
        axes[3].fill_between(
            x,
            0,
            1,
            where=mask,
            transform=axes[3].get_xaxis_transform(),
            color=colors.get(name, "0.5"),
            alpha=0.10,
            label=name,
        )
    for candidate in candidate_records:
        u_px = int(candidate["u_px"])
        for axis in axes:
            axis.axvline(u_px, color="black", alpha=0.25, linewidth=0.6)
    axes[3].legend(loc="upper right", ncol=3)
    for axis in axes:
        mark_regions(axis, image_width)
    fig.suptitle("Automatically selected abnormal u locations")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_profile(
    axis: Any, candidate: Mapping[str, Any], show_legend: bool = False
) -> None:
    profile = np.asarray(candidate["profile"], dtype=np.float64)
    v_axis = np.asarray(candidate["v_axis"], dtype=np.float64)
    axis.plot(profile, v_axis, color="black", linewidth=1.0, label="raw DN")
    axis.axvline(
        float(candidate["background_dn"]),
        color="0.45",
        linestyle=":",
        linewidth=0.7,
    )
    peak_v = parse_float(candidate.get("peak_v_px"))
    steger_v = parse_float(candidate.get("steger_v_px"))
    centroid_v = parse_float(candidate.get("centroid_v_px"))
    if math.isfinite(peak_v):
        axis.axhline(peak_v, color="black", linestyle=":", linewidth=0.7)
    if math.isfinite(steger_v):
        axis.axhline(steger_v, color="tab:red", linewidth=1.0, label="Steger")
    if math.isfinite(centroid_v):
        axis.axhline(centroid_v, color="tab:blue", linewidth=1.0, label="centroid")
    title = (
        f"{candidate['candidate_id']} u={candidate['u_px']} "
        f"f={candidate['frame_index']} {candidate['primary_morphology']}"
    )
    axis.set_title(title, fontsize=8)
    axis.set_xlabel("DN")
    axis.set_ylabel("v (px)")
    axis.set_xlim(left=0)
    axis.invert_yaxis()
    if show_legend:
        axis.legend(fontsize=7, loc="best")


def plot_profile_grid(
    path: Path,
    candidates: Sequence[Mapping[str, Any]],
    title: str,
    columns: int = 4,
) -> None:
    if not candidates:
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.text(0.5, 0.5, "No selected profiles", ha="center", va="center")
        axis.set_axis_off()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    rows = int(math.ceil(len(candidates) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.0, rows * 3.2))
    axes_array = np.atleast_1d(axes).ravel()
    for index, candidate in enumerate(candidates):
        plot_profile(axes_array[index], candidate, show_legend=index == 0)
    for axis in axes_array[len(candidates) :]:
        axis.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def profile_aggregate_for_u(
    images: Sequence[np.ndarray],
    u_px: int,
    profile_top: int,
    profile_bottom: int,
) -> list[dict[str, Any]]:
    return [
        profile_metrics(image, u_px, profile_top, profile_bottom)
        for image in images
    ]


def candidate_row(
    candidate: Mapping[str, Any],
    per_u: Mapping[str, Any],
    profile: Mapping[str, Any],
    profile_metrics_all_frames: Sequence[Mapping[str, Any]],
    frame_index: int,
    frame_filename: str,
    steger_v: float,
    centroid_v: float,
    steger_jump: float,
    steger_high: float,
    centroid_high: float,
    rejection_reason: str,
) -> dict[str, Any]:
    abnormal_frame_flags = []
    for item in profile_metrics_all_frames:
        abnormal_frame_flags.append(
            item["primary_morphology"]
            in {"low_contrast", "multi_peak", "saturated_peak", "asymmetric_peak"}
        )
    delta_ratio = parse_float(per_u.get("delta_gt_0_3_ratio"))
    jump_ratio = parse_float(per_u.get("jump_gt_1_0_ratio"))
    high_ratio = parse_float(per_u.get("steger_high_significant_ratio"))
    fixed_evidence: list[str] = []
    if math.isfinite(delta_ratio) and delta_ratio >= MIN_FIXED_FRAME_RATIO:
        fixed_evidence.append(f"|delta_v|>0.3 in {delta_ratio:.0%} frames")
    if math.isfinite(jump_ratio) and jump_ratio >= MIN_FIXED_FRAME_RATIO:
        fixed_evidence.append(f"Steger jump>1.0 in {jump_ratio:.0%} frames")
    if math.isfinite(high_ratio) and high_ratio >= MIN_FIXED_FRAME_RATIO:
        fixed_evidence.append(f"Steger high residual>0.15 in {high_ratio:.0%} frames")
    profile_abnormal_ratio = float(
        np.count_nonzero(abnormal_frame_flags) / max(len(abnormal_frame_flags), 1)
    )
    if profile_abnormal_ratio >= MIN_FIXED_FRAME_RATIO:
        fixed_evidence.append(f"profile abnormal in {profile_abnormal_ratio:.0%} frames")
    return {
        "candidate_id": candidate["candidate_id"],
        "u_px": candidate["u_px"],
        "region": candidate["region"],
        "selection_reasons": json.dumps(
            candidate["selection_reasons"], ensure_ascii=False, separators=(",", ":")
        ),
        "selection_reason_for_frame": candidate["selection_reason_for_frame"],
        "selection_rank_score": candidate["selection_rank_score"],
        "frame_index": frame_index + 1,
        "frame_filename": frame_filename,
        "profile_top_v_px": candidate["profile_top_v_px"],
        "profile_bottom_exclusive_v_px": candidate["profile_bottom_exclusive_v_px"],
        "peak_v_px": profile["peak_v_px"],
        "peak_dn": profile["peak_dn"],
        "background_dn": profile["background_dn"],
        "contrast_dn": profile["contrast_dn"],
        "half_max_level_dn": profile["half_max_level_dn"],
        "fwhm_px": profile["fwhm_px"],
        "left_half_width_px": profile["left_half_width_px"],
        "right_half_width_px": profile["right_half_width_px"],
        "peak_count": profile["peak_count"],
        "peak_positions_v_px_json": json.dumps(
            jsonable(profile["peak_positions_v_px"]),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "peak_spacing_px_json": json.dumps(
            jsonable(profile["peak_spacing_px"]),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "minimum_peak_spacing_px": profile["minimum_peak_spacing_px"],
        "profile_saturated_pixel_count": profile["profile_saturated_pixel_count"],
        "profile_saturation_rate": profile["profile_saturation_rate"],
        "peak_saturated": profile["peak_saturated"],
        "weighted_skew": profile["weighted_skew"],
        "asymmetry_index": profile["asymmetry_index"],
        "morphology_labels": json.dumps(
            profile["morphology_labels"], ensure_ascii=False, separators=(",", ":")
        ),
        "primary_morphology": profile["primary_morphology"],
        "steger_v_px": steger_v,
        "centroid_v_px": centroid_v,
        "delta_v_px": steger_v - centroid_v
        if math.isfinite(steger_v) and math.isfinite(centroid_v)
        else None,
        "steger_local_jump_px": steger_jump,
        "steger_high_residual_px": steger_high,
        "centroid_high_residual_px": centroid_high,
        "steger_rejection_reason": rejection_reason,
        "profile_abnormal_frame_ratio": profile_abnormal_ratio,
        "profile_low_contrast_frame_ratio": float(
            np.count_nonzero(
                [item["primary_morphology"] == "low_contrast" for item in profile_metrics_all_frames]
            )
            / max(len(profile_metrics_all_frames), 1)
        ),
        "profile_multi_peak_frame_ratio": float(
            np.count_nonzero(
                ["multi_peak" in item["morphology_labels"] for item in profile_metrics_all_frames]
            )
            / max(len(profile_metrics_all_frames), 1)
        ),
        "profile_saturated_frame_ratio": float(
            np.count_nonzero([item["peak_saturated"] for item in profile_metrics_all_frames])
            / max(len(profile_metrics_all_frames), 1)
        ),
        "profile_asymmetric_frame_ratio": float(
            np.count_nonzero(
                ["asymmetric_peak" in item["morphology_labels"] for item in profile_metrics_all_frames]
            )
            / max(len(profile_metrics_all_frames), 1)
        ),
        "anomaly_fixed_in_20_frames": bool(
            len(fixed_evidence) > 0
        ),
        "fixed_evidence": json.dumps(
            fixed_evidence, ensure_ascii=False, separators=(",", ":")
        ),
        "profile_v_px_json": json.dumps(
            jsonable(profile["v_axis"]), ensure_ascii=False, separators=(",", ":")
        ),
        "raw_profile_dn_json": json.dumps(
            jsonable(profile["profile"]), ensure_ascii=False, separators=(",", ":")
        ),
    }


def interval_text(intervals: Sequence[Mapping[str, Any]], limit: int = 20) -> str:
    if not intervals:
        return "无"
    parts = [
        f"{item['u_start_px']}–{item['u_end_px']} ({item['true_column_count']} cols)"
        for item in intervals[:limit]
    ]
    if len(intervals) > limit:
        parts.append(f"... 共 {len(intervals)} 段")
    return ", ".join(parts)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    classification = summary["classification"]
    delta = summary["delta_analysis"]
    decomposition = summary["decomposition"]
    categories = summary["category_analysis"]
    candidates = summary["abnormal_profiles"]["candidates"]
    region_delta_rows = []
    for region in ("left", "center", "right", "all"):
        item = delta["by_region"][region]
        region_delta_rows.append(
            [
                region,
                item["paired_count"],
                item["abs_delta_px"]["median"],
                item["abs_delta_px"]["p95"],
                item["abs_delta_px"]["max"],
                item["within_0_1_ratio"],
                item["within_0_3_ratio"],
                item["within_0_5_ratio"],
            ]
        )
    decomposition_rows = []
    for region in ("left", "center", "right", "all"):
        item = decomposition["regions"][region]
        decomposition_rows.append(
            [
                region,
                item["steger"]["low_trend_span_by_frame_px"]["median"],
                item["steger"]["high_residual_abs_px"]["median"],
                item["steger"]["high_residual_abs_px"]["p95"],
                item["centroid"]["high_residual_abs_px"]["median"],
                item["high_frequency_paired_correlation"],
                item["high_frequency_same_direction_ratio"],
            ]
        )
    category_rows = []
    for name in (
        "RAW_STRIPE_SPATIAL_STRUCTURE",
        "STEGER_LOCAL_RIDGE_SELECTION",
        "EDGE_OPTICAL_DEGRADATION",
    ):
        item = categories["categories"][name]
        category_rows.append(
            [
                name,
                item["column_count"],
                item["image_width_fraction"],
                interval_text(item["intervals"], limit=100),
            ]
        )
    edge_by_region = categories["categories"]["EDGE_OPTICAL_DEGRADATION"].get(
        "by_region", {}
    )
    edge_region_text = "；".join(
        f"{region}: {edge_by_region[region]['column_count']}/{edge_by_region[region]['region_column_count']} cols ({fmt(edge_by_region[region]['region_fraction'] * 100, 1)}% of region), u={interval_text(edge_by_region[region]['intervals'], limit=100)}"
        for region in ("left", "right")
        if region in edge_by_region
    ) or "无"
    candidate_rows = []
    for candidate in candidates:
        candidate_rows.append(
            [
                candidate["candidate_id"],
                candidate["u_px"],
                candidate["region"],
                candidate["selection_reasons"],
                candidate["frame_index"],
                candidate["primary_morphology"],
                candidate["peak_dn"],
                candidate["contrast_dn"],
                candidate["fwhm_px"],
                candidate["peak_count"],
                candidate["delta_v_px"],
                candidate["steger_local_jump_px"],
                candidate["anomaly_fixed_in_20_frames"],
            ]
        )
    lines = [
        "# 海康红光空间结构拆解报告",
        "",
        f"- 最终分类：**{classification['classification']}**",
        f"- 输入：{summary['input']['frame_count']} 帧，{summary['input']['image_size'][0]}×{summary['input']['image_size'][1]} Mono8",
        f"- 当前 Steger sigma：{summary['parameters']['current_sigma_px']}",
        "- 本报告的中心线、delta 和 profile 均由原始二维图像重新计算；没有使用三维结果。",
        "",
        "## 1. 数据与复用审计",
        "",
        f"输入目录：{summary['input']['recording_dir']}",
        f"A-1 产物目录：{summary['provenance']['a1_artifact_dir']}",
        f"本轮复用的 A-1 信息：{', '.join(summary['provenance']['reused_artifacts'])}。",
        f"被检查但排除的文件：{', '.join(summary['provenance']['excluded_artifacts']) or '无'}。",
        "A-1 产物用于协议校验、边缘候选选择和 provenance；没有把 A-1 的中心线结果当作本轮算法输出。",
        "",
        "## 2. Steger / centroid 逐列差异",
        "",
        "delta_v = v_steger - v_centroid。统计只在同一帧同一列两种算法都有效时计算；CSV 仍保留全部帧列及阈值标记。",
        "",
        markdown_table(
            [
                "区域",
                "paired count",
                "|delta| P50(px)",
                "|delta| P95(px)",
                "|delta| max(px)",
                "≤0.1",
                "≤0.3",
                "≤0.5",
            ],
            region_delta_rows,
        ),
        "",
        f"全局 |delta_v| P50/P95/max = **{fmt(delta['global']['abs_delta_px']['median'])}/{fmt(delta['global']['abs_delta_px']['p95'])}/{fmt(delta['global']['abs_delta_px']['max'])} px**；落在 0.1/0.3/0.5 px 内的比例分别为 **{fmt(delta['global']['within_0_1_ratio'] * 100, 1)}% / {fmt(delta['global']['within_0_3_ratio'] * 100, 1)}% / {fmt(delta['global']['within_0_5_ratio'] * 100, 1)}%**。",
        "",
        f"按 20 帧中至少一次超过阈值的 u 区间：0.1 px：{interval_text(delta['threshold_intervals']['any_frame_gt_0_1'])}；0.3 px：{interval_text(delta['threshold_intervals']['any_frame_gt_0_3'])}；0.5 px：{interval_text(delta['threshold_intervals']['any_frame_gt_0_5'])}。",
        "",
        "## 3. 低频趋势与局部高频结构",
        "",
        f"低频趋势使用 Gaussian sigma={DECOMPOSITION_GAUSSIAN_SIGMA_PX:g} px；高频 residual = 有效中心线 - 低频趋势。这里不使用全局直线 RMS 作为主判断指标。",
        "",
        markdown_table(
            [
                "区域",
                "Steger low span P50(px)",
                "Steger high |res| P50(px)",
                "Steger high |res| P95(px)",
                "centroid high |res| P50(px)",
                "high corr",
                "same direction",
            ],
            decomposition_rows,
        ),
        "",
        f"中央区域高频同步相关系数 = **{fmt(decomposition['regions']['center']['high_frequency_paired_correlation'])}**，同方向比例 = **{fmt(decomposition['regions']['center']['high_frequency_same_direction_ratio'] * 100, 1)}%**（有效配对样本 {decomposition['regions']['center']['high_frequency_same_direction_count']}）。",
        "",
        "## 4. 异常列与原始 profile",
        "",
        markdown_table(
            [
                "ID",
                "u",
                "区域",
                "选择原因",
                "帧",
                "形态",
                "peak DN",
                "contrast DN",
                "FWHM",
                "峰数",
                "delta",
                "jump",
                "20 帧固定",
            ],
            candidate_rows,
        ),
        "",
        "abnormal_profile_summary.csv 为每个候选列一行；其中 profile_v_px_json 和 raw_profile_dn_json 保存完整的原始纵向 DN 曲线，不是分类摘要的替代品。abnormal_profiles_grid.png、center_region_profiles.png、edge_region_profiles.png 同时绘制 raw DN 曲线及 Steger/centroid 中心位置。",
        f"形态计数中 morphology_counts 采用每个候选的互斥 primary_morphology；morphology_label_counts 统计 morphology_labels 的标签出现次数，标签可以重叠。候选中有 {summary['abnormal_profiles']['fixed_candidate_count']}/{summary['abnormal_profiles']['candidate_count']} 个满足 20 帧固定判据；其余候选仍保留为局部检查对象。",
        "",
        "## 5. 根因分类与 u 区间",
        "",
        markdown_table(
            ["类别", "列数", "占图像宽度", "u 区间"],
            category_rows,
        ),
        "",
        f"边缘光学退化判据包含：边缘 A-1 contrast median < max(20 DN, {EDGE_RELATIVE_CONTRAST_FRACTION:.2f} × 中央参考 contrast median)；本批中央参考为 {fmt(summary['classification']['evidence_thresholds']['edge_center_reference_contrast_dn'])} DN，对应阈值为 {fmt(summary['classification']['evidence_thresholds']['edge_relative_contrast_threshold_dn'])} DN。类别区间可与 ridge 证据区间重叠。",
        f"边缘按左右区域拆分：{edge_region_text}。",
        "",
        f"最终分类规则检测到：{', '.join(classification['detected_categories']) or '无明确类别'}。分类为 **{classification['classification']}**。",
        "",
        "## 6. 最后回答四个问题",
        "",
        f"1. Steger 与 centroid 在绝大多数位置是否重合？**{classification['answers']['q1_overlap']}**。全局配对位置中 |delta|≤0.1/0.3/0.5 px 的比例为 {fmt(delta['global']['within_0_1_ratio'] * 100, 1)}%/{fmt(delta['global']['within_0_3_ratio'] * 100, 1)}%/{fmt(delta['global']['within_0_5_ratio'] * 100, 1)}%。",
        f"2. 中央区域的空间起伏是否来自原始条纹自身？**{classification['answers']['q2_center_raw_structure']}**。依据是中央高频分量的 Steger/centroid 同步相关和同方向比例，而不是全局直线拟合残差。",
        f"3. 是否存在明确的局部多峰或 ridge 选错？**{classification['answers']['q3_ridge_selection']}**。候选 profile 中带 multi_peak 标签的数量为 {summary['abnormal_profiles']['morphology_label_counts'].get('multi_peak', 0)}；对应 ridge 证据区间为 {interval_text(categories['categories']['STEGER_LOCAL_RIDGE_SELECTION']['intervals'], limit=100)}。",
        f"4. 左右边缘问题与中央问题是否属于同一种根因？**{classification['answers']['q4_edge_vs_center']}**。左右边缘分别为 {edge_region_text}；中央结构区间为 {interval_text(categories['categories']['RAW_STRIPE_SPATIAL_STRUCTURE']['intervals'], limit=100)}。",
        "",
        "## 7. 产物",
        "",
        "- spatial_structure_by_u.csv：逐列汇总、低/高频分量、同步性、A-1 质量映射和类别标记。",
        "- steger_centroid_delta.csv：20 帧逐列 Steger/centroid/delta 及 0.1/0.3/0.5 px 标记。",
        "- abnormal_profile_summary.csv：异常列的形态统计与完整原始 profile JSON。",
        "- spatial_structure_summary.json：参数、provenance、统计、分类和区间。",
        "- steger_vs_centroid_delta_v.png、low_high_frequency_decomposition.png、abnormal_u_locations.png、abnormal_profiles_grid.png、center_region_profiles.png、edge_region_profiles.png。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="海康红光 Steger / centroid 空间结构拆解（固定 sigma=1.5）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "recording_dir", nargs="?", type=Path, default=DEFAULT_RECORDING_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steger-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--expected-frames", type=positive_int, default=EXPECTED_FRAMES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recording_dir = args.recording_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    profile_path = args.steger_profile.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else recording_dir / DEFAULT_OUTPUT_DIRNAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, frame_metadata = discover_frames(recording_dir, args.expected_frames)
    frames = load_frames(frame_paths, frame_metadata)
    image_height, image_width = frames[0]["image"].shape
    a1 = load_a1_artifacts(recording_dir, (image_height, image_width), args.expected_frames)
    config: AppConfig = load_app_config(config_path)
    profile_options = load_profile_options(profile_path)
    declared_profile = profile_path_declared_by_config(config_path)
    if declared_profile is not None and sha256_file(declared_profile) != sha256_file(profile_path):
        raise DiagnosticError("外部 Steger profile 与配置声明的 profile 内容不一致")
    configured_steger_options = dict(config.extraction_options)
    if config.extraction_method != "steger":
        raise DiagnosticError(f"当前配置 extraction.method 不是 steger: {config.extraction_method}")
    if str(configured_steger_options.get("scan_axis", "column")) != "column":
        raise DiagnosticError("当前 Steger scan_axis 不是 column")
    configured_sigma = parse_float(configured_steger_options.get("sigma"))
    if not math.isclose(configured_sigma, CURRENT_SIGMA, rel_tol=0.0, abs_tol=1.0e-9):
        raise DiagnosticError(f"当前配置 sigma={configured_sigma}，A-2 要求 sigma=1.5")
    centroid_options = dict(
        config.extraction_options_by_method.get("centroid", {})
    )
    if not centroid_options:
        raise DiagnosticError("当前配置缺少 centroid options，无法进行 A/B")

    profile_roi_value = configured_steger_options.get("search_roi")
    if profile_roi_value is None:
        profile_top = 0
        profile_bottom = image_height
        effective_roi = {
            "effective_left": 0,
            "effective_top": 0,
            "effective_right": image_width,
            "effective_bottom": image_height,
        }
    else:
        effective_roi = _parse_roi(profile_roi_value, (image_height, image_width))
        profile_top = effective_roi["effective_top"]
        profile_bottom = effective_roi["effective_bottom"]

    steger_runs: list[dict[str, Any]] = []
    centroid_vectors: list[np.ndarray] = []
    steger_vectors: list[np.ndarray] = []
    steger_jumps: list[np.ndarray] = []
    backend_checks: list[dict[str, Any]] = []
    for frame in frames:
        steger_run = effective_steger_call(frame["image"], configured_steger_options)
        centroid_points = np.asarray(
            centroid_backend(frame["image"], centroid_options), dtype=np.float64
        ).reshape((-1, 2))
        steger_runs.append(steger_run)
        steger_vectors.append(steger_run["v_by_u"])
        centroid_vectors.append(points_to_vector(centroid_points, image_width))
        steger_jumps.append(local_jump_vector(steger_run["v_by_u"]))
        backend_checks.append(
            {
                "frame_index": frame["index"],
                "frame_filename": frame["filename"],
                "point_count_equal": steger_run["backend_point_count_equal"],
                "direct_valid_column_count": steger_run["direct_valid_column_count"],
                "wrapper_valid_point_count": steger_run["valid_column_count"],
                "max_abs_delta_px": steger_run["backend_max_abs_delta_px"],
            }
        )

    steger_matrix = np.asarray(steger_vectors, dtype=np.float64)
    centroid_matrix = np.asarray(centroid_vectors, dtype=np.float64)
    jump_matrix = np.asarray(steger_jumps, dtype=np.float64)
    pair_matrix = np.where(
        np.isfinite(steger_matrix) & np.isfinite(centroid_matrix),
        steger_matrix - centroid_matrix,
        np.nan,
    )
    steger_low_rows: list[np.ndarray] = []
    steger_high_rows: list[np.ndarray] = []
    centroid_low_rows: list[np.ndarray] = []
    centroid_high_rows: list[np.ndarray] = []
    internal_gap_counts = {"steger": [], "centroid": []}
    for steger_values, centroid_values in zip(
        steger_matrix, centroid_matrix, strict=True
    ):
        s_low, s_high, s_gap = decompose_vector(
            steger_values, DECOMPOSITION_GAUSSIAN_SIGMA_PX
        )
        c_low, c_high, c_gap = decompose_vector(
            centroid_values, DECOMPOSITION_GAUSSIAN_SIGMA_PX
        )
        steger_low_rows.append(s_low)
        steger_high_rows.append(s_high)
        centroid_low_rows.append(c_low)
        centroid_high_rows.append(c_high)
        internal_gap_counts["steger"].append(s_gap)
        internal_gap_counts["centroid"].append(c_gap)
    steger_low = np.asarray(steger_low_rows)
    steger_high = np.asarray(steger_high_rows)
    centroid_low = np.asarray(centroid_low_rows)
    centroid_high = np.asarray(centroid_high_rows)

    sample_columns, a1_quality = aggregate_a1_quality(
        a1["quality_rows"], image_width
    )
    a1_quality_arrays = nearest_quality_arrays(sample_columns, a1_quality, image_width)

    delta_abs = np.abs(pair_matrix)
    delta_abs_p50 = column_stat_array(delta_abs, 50)
    delta_abs_p95 = column_stat_array(delta_abs, 95)
    delta_abs_max = column_stat_array(delta_abs, 100)
    jump_p50 = column_stat_array(jump_matrix, 50)
    jump_p95 = column_stat_array(jump_matrix, 95)
    jump_max = column_stat_array(jump_matrix, 100)
    steger_high_abs_p50 = column_stat_array(np.abs(steger_high), 50)
    steger_high_abs_p95 = column_stat_array(np.abs(steger_high), 95)
    centroid_high_abs_p50 = column_stat_array(np.abs(centroid_high), 50)
    centroid_high_abs_p95 = column_stat_array(np.abs(centroid_high), 95)
    pair_count = column_count_array(pair_matrix)
    steger_count = column_count_array(steger_matrix)
    centroid_count = column_count_array(centroid_matrix)
    high_sync_count = np.zeros(image_width, dtype=np.int64)
    high_sync_same_direction_count = np.zeros(image_width, dtype=np.int64)
    high_significant_count = np.zeros(image_width, dtype=np.int64)
    for u_px in range(image_width):
        s_values = steger_high[:, u_px]
        c_values = centroid_high[:, u_px]
        valid = (
            np.isfinite(s_values)
            & np.isfinite(c_values)
            & (np.abs(s_values) >= HIGH_SYNC_MIN_ABS_PX)
            & (np.abs(c_values) >= HIGH_SYNC_MIN_ABS_PX)
        )
        high_sync_count[u_px] = int(np.count_nonzero(valid))
        high_significant_count[u_px] = int(
            np.count_nonzero(np.isfinite(s_values) & (np.abs(s_values) >= HIGH_RESIDUAL_SIGNIFICANCE_PX))
        )
        if np.any(valid):
            high_sync_same_direction_count[u_px] = int(
                np.count_nonzero(s_values[valid] * c_values[valid] >= 0.0)
            )
    high_sync_ratio = np.divide(
        high_sync_same_direction_count,
        np.maximum(high_sync_count, 1),
        where=np.ones(image_width, dtype=bool),
    ).astype(np.float64)
    high_significant_ratio = np.divide(
        high_significant_count,
        np.maximum(steger_count, 1),
        where=np.ones(image_width, dtype=bool),
    ).astype(np.float64)

    center_quality_values = a1_quality_arrays["contrast_median"][
        region_mask("center", image_width)
    ]
    center_quality_values = center_quality_values[np.isfinite(center_quality_values)]
    center_reference_contrast_dn = (
        float(np.median(center_quality_values))
        if len(center_quality_values)
        else None
    )
    edge_relative_contrast_threshold_dn = (
        max(20.0, EDGE_RELATIVE_CONTRAST_FRACTION * center_reference_contrast_dn)
        if center_reference_contrast_dn is not None
        else 20.0
    )
    edge_relative_contrast_flag = (
        a1_quality_arrays["contrast_median"] < edge_relative_contrast_threshold_dn
    )
    preliminary_edge_mask = np.zeros(image_width, dtype=bool)
    for region in ("left", "right"):
        region_columns = region_mask(region, image_width)
        preliminary_edge_mask |= region_columns & (
            edge_relative_contrast_flag
            | (a1_quality_arrays["low_contrast_ratio"] >= 0.50)
            | (a1_quality_arrays["fwhm_valid_ratio"] < 0.90)
        )
    preliminary_raw_mask = (
        region_mask("center", image_width)
        & (steger_high_abs_p50 >= HIGH_RESIDUAL_SIGNIFICANCE_PX)
        & (centroid_high_abs_p50 >= HIGH_RESIDUAL_SIGNIFICANCE_PX)
        & (high_sync_ratio >= 0.65)
    )
    preliminary_ridge_mask = (
        ((delta_abs_p95 >= 0.30) & (jump_p95 >= STEGER_JUMP_EVIDENCE_PX))
        | ((delta_abs_max >= 0.50) & (jump_max >= STEGER_JUMP_EVIDENCE_PX * 1.5))
    )

    candidate_reasons: dict[int, list[str]] = defaultdict(list)
    delta_candidates = select_positions(
        delta_abs_max,
        pair_count > 0,
        descending=True,
        count=6,
    )
    jump_candidates = select_positions(
        jump_max,
        np.isfinite(jump_max),
        descending=True,
        count=6,
    )
    center_candidates = select_positions(
        np.maximum(steger_high_abs_p50, centroid_high_abs_p50),
        region_mask("center", image_width),
        descending=True,
        count=6,
    )
    edge_candidates: list[int] = []
    for region in ("left", "right"):
        edge_candidates.extend(
            select_positions(
                a1_quality_arrays["contrast_median"],
                region_mask(region, image_width)
                & np.isfinite(a1_quality_arrays["contrast_median"]),
                descending=False,
                count=5,
            )
        )
    for u_px in delta_candidates:
        candidate_reasons[u_px].append("delta_peak")
    for u_px in jump_candidates:
        candidate_reasons[u_px].append("steger_local_jump")
    for u_px in center_candidates:
        candidate_reasons[u_px].append("center_high_frequency")
    for u_px in edge_candidates:
        candidate_reasons[u_px].append("edge_low_contrast")

    candidate_u_sorted = sorted(candidate_reasons)
    if len(candidate_u_sorted) > MAX_PROFILE_CANDIDATES:
        candidate_u_sorted = candidate_u_sorted[:MAX_PROFILE_CANDIDATES]
    candidate_records: list[dict[str, Any]] = []
    image_list = [frame["image"] for frame in frames]
    all_profile_metrics: dict[int, list[dict[str, Any]]] = {}
    for candidate_number, u_px in enumerate(candidate_u_sorted, start=1):
        reasons = candidate_reasons[u_px]
        profile_all = profile_aggregate_for_u(
            image_list, u_px, profile_top, profile_bottom
        )
        all_profile_metrics[u_px] = profile_all
        contrast_values = np.asarray(
            [item["contrast_dn"] for item in profile_all], dtype=np.float64
        )
        frame_zero_index, selected_reason = choose_frame_for_candidate(
            u_px,
            reasons,
            delta_abs,
            jump_matrix,
            np.abs(steger_high),
            contrast_values,
        )
        profile = profile_all[frame_zero_index]
        score_values = []
        if "delta_peak" in reasons and math.isfinite(delta_abs_max[u_px]):
            score_values.append(float(delta_abs_max[u_px]))
        if "steger_local_jump" in reasons and math.isfinite(jump_max[u_px]):
            score_values.append(float(jump_max[u_px]))
        if "center_high_frequency" in reasons and math.isfinite(steger_high_abs_p50[u_px]):
            score_values.append(float(steger_high_abs_p50[u_px]))
        if "edge_low_contrast" in reasons and math.isfinite(
            a1_quality_arrays["contrast_median"][u_px]
        ):
            score_values.append(float(a1_quality_arrays["contrast_median"][u_px]))
        candidate_records.append(
            {
                "candidate_id": f"A{candidate_number:02d}",
                "u_px": u_px,
                "region": region_for_u(u_px, image_width),
                "selection_reasons": list(reasons),
                "selection_reason_for_frame": selected_reason,
                "selection_rank_score": max(score_values) if score_values else None,
                "profile_top_v_px": profile_top,
                "profile_bottom_exclusive_v_px": profile_bottom,
                "profile": profile["profile"],
                "v_axis": profile["v_axis"],
                "peak_v_px": profile["peak_v_px"],
            }
        )

    profile_ridge_mask = np.zeros(image_width, dtype=bool)
    morphology_counts: Counter[str] = Counter()
    profile_rows: list[dict[str, Any]] = []
    for candidate in candidate_records:
        u_px = int(candidate["u_px"])
        profile_all = all_profile_metrics[u_px]
        selected_frame_index = choose_frame_for_candidate(
            u_px,
            candidate["selection_reasons"],
            delta_abs,
            jump_matrix,
            np.abs(steger_high),
            np.asarray([item["contrast_dn"] for item in profile_all], dtype=np.float64),
        )[0]
        selected_profile = profile_all[selected_frame_index]
        morphology_counts[selected_profile["primary_morphology"]] += 1
        if (
            "multi_peak" in selected_profile["morphology_labels"]
            and (
                "delta_peak" in candidate["selection_reasons"]
                or "steger_local_jump" in candidate["selection_reasons"]
            )
            and (
                (
                    math.isfinite(delta_abs_max[u_px])
                    and delta_abs_max[u_px] >= 0.30
                )
                or (
                    math.isfinite(jump_max[u_px])
                    and jump_max[u_px] >= STEGER_JUMP_EVIDENCE_PX
                )
            )
        ):
            profile_ridge_mask[u_px] = True
        diagnostics = steger_runs[selected_frame_index]["diagnostic_arrays"]
        rejection = diagnostics.get("rejection_reason", np.full(image_width, ""))
        rejection_reason = str(rejection[u_px])
        per_u_placeholder = {
            "delta_gt_0_3_ratio": (
                float(np.count_nonzero(delta_abs[:, u_px] > 0.3) / max(pair_count[u_px], 1))
                if pair_count[u_px]
                else None
            ),
            "jump_gt_1_0_ratio": (
                float(np.count_nonzero(jump_matrix[:, u_px] > 1.0) / max(np.count_nonzero(np.isfinite(jump_matrix[:, u_px])), 1))
                if np.count_nonzero(np.isfinite(jump_matrix[:, u_px]))
                else None
            ),
            "steger_high_significant_ratio": (
                float(np.count_nonzero(np.isfinite(steger_high[:, u_px]) & (np.abs(steger_high[:, u_px]) >= HIGH_RESIDUAL_SIGNIFICANCE_PX)) / max(steger_count[u_px], 1))
                if steger_count[u_px]
                else None
            ),
        }
        profile_rows.append(
            candidate_row(
                candidate,
                per_u_placeholder,
                selected_profile,
                profile_all,
                selected_frame_index,
                frames[selected_frame_index]["filename"],
                steger_matrix[selected_frame_index, u_px],
                centroid_matrix[selected_frame_index, u_px],
                jump_matrix[selected_frame_index, u_px],
                steger_high[selected_frame_index, u_px],
                centroid_high[selected_frame_index, u_px],
                rejection_reason,
            )
        )

    ridge_mask = preliminary_ridge_mask | profile_ridge_mask
    edge_mask = preliminary_edge_mask
    raw_mask = preliminary_raw_mask & ~ridge_mask
    issue_category = np.full(image_width, "NORMAL", dtype=object)
    issue_category[raw_mask] = "RAW_STRIPE_SPATIAL_STRUCTURE"
    issue_category[ridge_mask] = "STEGER_LOCAL_RIDGE_SELECTION"
    issue_category[edge_mask & ~ridge_mask] = "EDGE_OPTICAL_DEGRADATION"

    delta_rows: list[dict[str, Any]] = []
    for frame_zero_index, frame in enumerate(frames):
        for u_px in range(image_width):
            delta_value = pair_matrix[frame_zero_index, u_px]
            delta_rows.append(
                {
                    "frame_index": frame["index"],
                    "frame_filename": frame["filename"],
                    "u_px": u_px,
                    "region": region_for_u(u_px, image_width),
                    "steger_v_px": steger_matrix[frame_zero_index, u_px],
                    "centroid_v_px": centroid_matrix[frame_zero_index, u_px],
                    "both_valid": bool(math.isfinite(delta_value)),
                    "delta_v_px": delta_value,
                    "abs_delta_v_px": abs(delta_value)
                    if math.isfinite(delta_value)
                    else None,
                    "steger_local_jump_px": jump_matrix[frame_zero_index, u_px],
                    "steger_high_residual_px": steger_high[frame_zero_index, u_px],
                    "centroid_high_residual_px": centroid_high[frame_zero_index, u_px],
                    "high_same_direction": bool(
                        math.isfinite(steger_high[frame_zero_index, u_px])
                        and math.isfinite(centroid_high[frame_zero_index, u_px])
                        and abs(steger_high[frame_zero_index, u_px]) >= HIGH_SYNC_MIN_ABS_PX
                        and abs(centroid_high[frame_zero_index, u_px]) >= HIGH_SYNC_MIN_ABS_PX
                        and steger_high[frame_zero_index, u_px]
                        * centroid_high[frame_zero_index, u_px]
                        >= 0.0
                    ),
                    "abs_delta_gt_0_1_px": bool(
                        math.isfinite(delta_value) and abs(delta_value) > 0.1
                    ),
                    "abs_delta_gt_0_3_px": bool(
                        math.isfinite(delta_value) and abs(delta_value) > 0.3
                    ),
                    "abs_delta_gt_0_5_px": bool(
                        math.isfinite(delta_value) and abs(delta_value) > 0.5
                    ),
                    "issue_category": issue_category[u_px],
                }
            )

    def region_delta_summary(region: str) -> dict[str, Any]:
        mask = region_mask(region, image_width)
        values = delta_abs[:, mask].ravel()
        values = values[np.isfinite(values)]
        pair_values = pair_matrix[:, mask]
        return {
            "region": region,
            "paired_count": int(len(values)),
            "abs_delta_px": stats(values),
            "within_0_1_ratio": float(np.count_nonzero(values <= 0.1) / len(values))
            if len(values)
            else None,
            "within_0_3_ratio": float(np.count_nonzero(values <= 0.3) / len(values))
            if len(values)
            else None,
            "within_0_5_ratio": float(np.count_nonzero(values <= 0.5) / len(values))
            if len(values)
            else None,
            "signed_delta_px": stats(pair_values.ravel()),
        }

    global_delta = region_delta_summary("all")
    by_region_delta = {"all": global_delta}
    by_region_delta.update(
        {
            region: region_delta_summary(region)
            for region in ("left", "center", "right")
        }
    )
    threshold_intervals = {
        f"any_frame_gt_{threshold:.1f}".replace(".", "_"): merge_intervals(
            np.any(delta_abs > threshold, axis=0)
        )
        for threshold in DELTA_THRESHOLDS_PX
    }

    category_masks = {
        "raw_structure": raw_mask,
        "ridge_selection": ridge_mask,
        "edge_optical": edge_mask,
    }
    category_names = {
        "raw_structure": "RAW_STRIPE_SPATIAL_STRUCTURE",
        "ridge_selection": "STEGER_LOCAL_RIDGE_SELECTION",
        "edge_optical": "EDGE_OPTICAL_DEGRADATION",
    }
    category_analysis: dict[str, Any] = {
        "categories": {},
        "issue_category_counts": dict(Counter(issue_category.tolist())),
    }
    for key, mask in category_masks.items():
        category_item = {
            "column_count": int(np.count_nonzero(mask)),
            "image_width_fraction": float(np.count_nonzero(mask) / image_width),
            "intervals": merge_intervals(mask),
            "u_values": np.flatnonzero(mask).astype(int).tolist(),
        }
        category_item["by_region"] = {
            region: {
                "column_count": int(
                    np.count_nonzero(mask & region_mask(region, image_width))
                ),
                "region_column_count": int(
                    np.count_nonzero(region_mask(region, image_width))
                ),
                "region_fraction": float(
                    np.count_nonzero(mask & region_mask(region, image_width))
                    / max(np.count_nonzero(region_mask(region, image_width)), 1)
                ),
                "intervals": merge_intervals(mask & region_mask(region, image_width)),
            }
            for region in ("left", "center", "right")
        }
        category_analysis["categories"][category_names[key]] = category_item

    detected_categories: list[str] = []
    center = decomposition_summary(
        steger_low, steger_high, centroid_low, centroid_high, image_width
    )["center"]
    raw_detected = bool(
        np.any(raw_mask)
        or (
            center["high_frequency_paired_correlation"] is not None
            and center["high_frequency_paired_correlation"] >= 0.70
            and center["high_frequency_same_direction_ratio"] is not None
            and center["high_frequency_same_direction_ratio"] >= 0.65
            and center["steger"]["high_residual_abs_px"]["p50"] is not None
            and center["steger"]["high_residual_abs_px"]["p50"]
            >= HIGH_RESIDUAL_SIGNIFICANCE_PX
        )
    )
    ridge_detected = bool(np.any(ridge_mask))
    edge_detected = bool(np.any(edge_mask))
    if raw_detected:
        detected_categories.append("RAW_STRIPE_SPATIAL_STRUCTURE")
    if ridge_detected:
        detected_categories.append("STEGER_LOCAL_RIDGE_SELECTION")
    if edge_detected:
        detected_categories.append("EDGE_OPTICAL_DEGRADATION")
    if len(detected_categories) == 1:
        final_classification = detected_categories[0]
    else:
        final_classification = "MIXED"

    profile_multi_count = int(
        sum(
            "multi_peak" in row["morphology_labels"]
            for row in profile_rows
        )
    )
    profile_morphology_label_counts = {
        label: int(
            sum(label in row["morphology_labels"] for row in profile_rows)
        )
        for label in (
            "single_peak",
            "multi_peak",
            "saturated_peak",
            "low_contrast",
            "asymmetric_peak",
            "normal",
        )
    }
    profile_ridge_evidence_count = int(np.count_nonzero(profile_ridge_mask))
    fixed_profile_count = int(
        sum(bool(row["anomaly_fixed_in_20_frames"]) for row in profile_rows)
    )
    overlap_ratio = global_delta["within_0_3_ratio"]
    q1 = (
        "是（绝大多数位置 |delta_v|≤0.3 px）"
        if overlap_ratio is not None and overlap_ratio >= 0.90
        else "否（|delta_v|≤0.3 px 未达到 90%）"
    )
    q2 = (
        "是，Steger 与 centroid 的中央高频分量同步"
        if raw_detected
        else "证据不足，未达到中央高频同步判据"
    )
    q3 = (
        f"存在 {profile_multi_count} 个候选 profile 为 multi_peak，且 ridge 证据列数为 {profile_ridge_evidence_count}"
        if profile_multi_count or ridge_detected
        else "未发现明确的多峰或 ridge 选错证据"
    )
    q4 = (
        "不是同一种：边缘是低对比度/光学退化，中央是可重复空间结构"
        if edge_detected and raw_detected
        else "当前数据未显示边缘与中央同时存在可区分的两类问题"
    )
    classification = {
        "classification": final_classification,
        "detected_categories": detected_categories,
        "answers": {
            "q1_overlap": q1,
            "q2_center_raw_structure": q2,
            "q3_ridge_selection": q3,
            "q4_edge_vs_center": q4,
        },
        "evidence_thresholds": {
            "delta_thresholds_px": list(DELTA_THRESHOLDS_PX),
            "high_residual_significance_px": HIGH_RESIDUAL_SIGNIFICANCE_PX,
            "high_sync_min_abs_px": HIGH_SYNC_MIN_ABS_PX,
            "high_sync_correlation": 0.70,
            "high_sync_same_direction_ratio": 0.65,
            "steger_jump_evidence_px": STEGER_JUMP_EVIDENCE_PX,
            "edge_relative_contrast_fraction_of_center": EDGE_RELATIVE_CONTRAST_FRACTION,
            "edge_center_reference_contrast_dn": center_reference_contrast_dn,
            "edge_relative_contrast_threshold_dn": edge_relative_contrast_threshold_dn,
            "fixed_frame_ratio": MIN_FIXED_FRAME_RATIO,
        },
    }

    spatial_rows: list[dict[str, Any]] = []
    for u_px in range(image_width):
        s_high_values = steger_high[:, u_px]
        c_high_values = centroid_high[:, u_px]
        signed_high_stats_s = stats(s_high_values)
        signed_high_stats_c = stats(c_high_values)
        s_v_stats = stats(steger_matrix[:, u_px])
        c_v_stats = stats(centroid_matrix[:, u_px])
        pair_values = pair_matrix[:, u_px]
        pair_count_u = int(pair_count[u_px])
        jump_count_u = int(np.count_nonzero(np.isfinite(jump_matrix[:, u_px])))
        spatial_rows.append(
            {
                "u_px": u_px,
                "region": region_for_u(u_px, image_width),
                "steger_valid_frame_count": int(steger_count[u_px]),
                "centroid_valid_frame_count": int(centroid_count[u_px]),
                "paired_frame_count": pair_count_u,
                "steger_v_median_px": s_v_stats["median"],
                "centroid_v_median_px": c_v_stats["median"],
                "delta_v_median_px": stats(pair_values)["median"],
                "delta_abs_p50_px": delta_abs_p50[u_px],
                "delta_abs_p95_px": delta_abs_p95[u_px],
                "delta_abs_max_px": delta_abs_max[u_px],
                "delta_gt_0_1_frame_count": int(np.count_nonzero(delta_abs[:, u_px] > 0.1)),
                "delta_gt_0_1_ratio": float(
                    np.count_nonzero(delta_abs[:, u_px] > 0.1) / pair_count_u
                )
                if pair_count_u
                else None,
                "delta_gt_0_3_frame_count": int(np.count_nonzero(delta_abs[:, u_px] > 0.3)),
                "delta_gt_0_3_ratio": float(
                    np.count_nonzero(delta_abs[:, u_px] > 0.3) / pair_count_u
                )
                if pair_count_u
                else None,
                "delta_gt_0_5_frame_count": int(np.count_nonzero(delta_abs[:, u_px] > 0.5)),
                "delta_gt_0_5_ratio": float(
                    np.count_nonzero(delta_abs[:, u_px] > 0.5) / pair_count_u
                )
                if pair_count_u
                else None,
                "steger_local_jump_valid_frame_count": jump_count_u,
                "steger_local_jump_p50_px": jump_p50[u_px],
                "steger_local_jump_p95_px": jump_p95[u_px],
                "steger_local_jump_max_px": jump_max[u_px],
                "jump_gt_1_0_frame_count": int(
                    np.count_nonzero(jump_matrix[:, u_px] > 1.0)
                ),
                "jump_gt_1_0_ratio": float(
                    np.count_nonzero(jump_matrix[:, u_px] > 1.0) / jump_count_u
                )
                if jump_count_u
                else None,
                "steger_low_trend_median_px": column_stat_array(
                    steger_low[:, u_px : u_px + 1], 50
                )[0],
                "centroid_low_trend_median_px": column_stat_array(
                    centroid_low[:, u_px : u_px + 1], 50
                )[0],
                "steger_high_residual_median_px": signed_high_stats_s["median"],
                "steger_high_abs_p50_px": steger_high_abs_p50[u_px],
                "steger_high_abs_p95_px": steger_high_abs_p95[u_px],
                "centroid_high_residual_median_px": signed_high_stats_c["median"],
                "centroid_high_abs_p50_px": centroid_high_abs_p50[u_px],
                "centroid_high_abs_p95_px": centroid_high_abs_p95[u_px],
                "high_frequency_sync_frame_count": int(high_sync_count[u_px]),
                "high_frequency_same_direction_ratio": (
                    high_sync_ratio[u_px] if high_sync_count[u_px] else None
                ),
                "steger_high_significant_frame_ratio": (
                    high_significant_ratio[u_px] if steger_count[u_px] else None
                ),
                "a1_quality_sample_u_px": a1_quality_arrays["sample_u"][u_px],
                "a1_peak_dn_median": a1_quality_arrays["peak_dn_median"][u_px],
                "a1_contrast_median_dn": a1_quality_arrays["contrast_median"][u_px],
                "a1_low_contrast_ratio": a1_quality_arrays["low_contrast_ratio"][u_px],
                "a1_fwhm_median_px": a1_quality_arrays["fwhm_median"][u_px],
                "a1_fwhm_valid_ratio": a1_quality_arrays["fwhm_valid_ratio"][u_px],
                "a1_peak_saturation_ratio": a1_quality_arrays["peak_saturation_ratio"][u_px],
                "edge_relative_contrast_flag": bool(edge_relative_contrast_flag[u_px]),
                "raw_structure_flag": bool(raw_mask[u_px]),
                "ridge_selection_flag": bool(ridge_mask[u_px]),
                "edge_optical_flag": bool(edge_mask[u_px]),
                "issue_category": issue_category[u_px],
            }
        )

    decomposition = {
        "gaussian_sigma_px": DECOMPOSITION_GAUSSIAN_SIGMA_PX,
        "internal_interpolation_gap_count_by_frame": internal_gap_counts,
        "regions": decomposition_summary(
            steger_low, steger_high, centroid_low, centroid_high, image_width
        ),
    }
    summary = {
        "schema_version": 1,
        "task": "hikrobot_red_spatial_structure_decomposition",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input": {
            "recording_dir": recording_dir,
            "frames_csv": recording_dir / "frames.csv",
            "frame_count": len(frames),
            "frame_filenames": [frame["filename"] for frame in frames],
            "image_size": [image_width, image_height],
            "dtype": str(frames[0]["image"].dtype),
            "pixel_min": int(min(np.min(frame["image"]) for frame in frames)),
            "pixel_max": int(max(np.max(frame["image"]) for frame in frames)),
            "capture_settings": {
                key: sorted({frame["metadata"].get(key, "") for frame in frames})
                for key in ("exposure_us", "gain_db", "pixel_format", "width", "height")
            },
        },
        "provenance": {
            "reused_artifacts": [
                "A-1 stripe_quality_by_u.csv for edge candidate quality",
                "A-1 sigma_sweep.csv for protocol validation",
                "A-1 temporal_repeatability.csv for protocol validation",
                "A-1 centroid_vs_steger.csv for protocol validation",
                "A-1 diagnostic_summary.json for frame/ROI/sigma validation",
                "A-1 diagnostic_report.md for artifact provenance validation",
            ],
            "a1_artifact_dir": a1["directory"],
            "a1_artifact_sha256": a1["hashes"],
            "excluded_artifacts": (
                ["height_shadow.csv"]
                if (recording_dir / "height_shadow.csv").is_file()
                else []
            ),
            "config_path": config_path,
            "config_sha256": sha256_file(config_path),
            "profile_path": profile_path,
            "profile_sha256": sha256_file(profile_path),
            "config_referenced_profile_path": declared_profile,
            "config_referenced_profile_sha256": (
                sha256_file(declared_profile) if declared_profile is not None else None
            ),
            "implementation_sha256": {
                "laser_backends.py": sha256_file(TOOL_ROOT / "laser" / "backends.py"),
                "laser_realtime_steger.py": sha256_file(
                    TOOL_ROOT / "laser" / "realtime_steger.py"
                ),
                "laser_steger_laser_center.py": sha256_file(
                    TOOL_ROOT / "laser" / "steger_laser_center.py"
                ),
            },
            "production_configuration_modified": False,
            "production_code_parameters_modified": False,
        },
        "parameters": {
            "current_sigma_px": CURRENT_SIGMA,
            "steger_options_fixed": configured_steger_options,
            "centroid_options_fixed": centroid_options,
            "search_roi_effective": effective_roi,
            "profile_top_px": profile_top,
            "profile_bottom_exclusive_px": profile_bottom,
            "decomposition_gaussian_sigma_px": DECOMPOSITION_GAUSSIAN_SIGMA_PX,
            "delta_thresholds_px": list(DELTA_THRESHOLDS_PX),
            "edge_relative_contrast_fraction_of_center": EDGE_RELATIVE_CONTRAST_FRACTION,
            "edge_center_reference_contrast_dn": center_reference_contrast_dn,
            "edge_relative_contrast_threshold_dn": edge_relative_contrast_threshold_dn,
            "max_profile_candidates": MAX_PROFILE_CANDIDATES,
        },
        "delta_analysis": {
            "definition": "delta_v = v_steger - v_centroid; statistics use paired valid columns",
            "global": global_delta,
            "by_region": by_region_delta,
            "threshold_intervals": threshold_intervals,
        },
        "decomposition": decomposition,
        "abnormal_profiles": {
            "profile_definition": {
                "background_percentile": PROFILE_BACKGROUND_PERCENTILE,
                "saturation_dn": PROFILE_SATURATION_DN,
                "fwhm_level": "background + 0.5 * contrast",
                "profile_curve_storage": "abnormal_profile_summary.csv JSON columns raw_profile_dn_json/profile_v_px_json",
            },
            "candidate_count": len(candidate_records),
            "morphology_counts": dict(morphology_counts),
            "morphology_counts_definition": "primary_morphology of the selected frame; mutually exclusive per candidate",
            "morphology_label_counts": profile_morphology_label_counts,
            "morphology_label_counts_definition": "presence of a label in morphology_labels; labels may overlap",
            "fixed_candidate_count": fixed_profile_count,
            "fixed_candidate_fraction": float(
                fixed_profile_count / max(len(profile_rows), 1)
            ),
            "candidates": profile_rows,
        },
        "category_analysis": category_analysis,
        "classification": classification,
        "backend_equivalence": {
            "all_point_counts_equal": all(
                check["point_count_equal"] for check in backend_checks
            ),
            "max_abs_delta_px": max(
                (check["max_abs_delta_px"] or 0.0 for check in backend_checks),
                default=0.0,
            ),
            "per_frame": backend_checks,
        },
    }

    write_csv(
        output_dir / "steger_centroid_delta.csv",
        delta_rows,
        (
            "frame_index",
            "frame_filename",
            "u_px",
            "region",
            "steger_v_px",
            "centroid_v_px",
            "both_valid",
            "delta_v_px",
            "abs_delta_v_px",
            "steger_local_jump_px",
            "steger_high_residual_px",
            "centroid_high_residual_px",
            "high_same_direction",
            "abs_delta_gt_0_1_px",
            "abs_delta_gt_0_3_px",
            "abs_delta_gt_0_5_px",
            "issue_category",
        ),
    )
    write_csv(
        output_dir / "spatial_structure_by_u.csv",
        spatial_rows,
        tuple(spatial_rows[0].keys()) if spatial_rows else ("u_px",),
    )
    write_csv(
        output_dir / "abnormal_profile_summary.csv",
        profile_rows,
        (
            "candidate_id",
            "u_px",
            "region",
            "selection_reasons",
            "selection_reason_for_frame",
            "selection_rank_score",
            "frame_index",
            "frame_filename",
            "profile_top_v_px",
            "profile_bottom_exclusive_v_px",
            "peak_v_px",
            "peak_dn",
            "background_dn",
            "contrast_dn",
            "half_max_level_dn",
            "fwhm_px",
            "left_half_width_px",
            "right_half_width_px",
            "peak_count",
            "peak_positions_v_px_json",
            "peak_spacing_px_json",
            "minimum_peak_spacing_px",
            "profile_saturated_pixel_count",
            "profile_saturation_rate",
            "peak_saturated",
            "weighted_skew",
            "asymmetry_index",
            "morphology_labels",
            "primary_morphology",
            "steger_v_px",
            "centroid_v_px",
            "delta_v_px",
            "steger_local_jump_px",
            "steger_high_residual_px",
            "centroid_high_residual_px",
            "steger_rejection_reason",
            "profile_abnormal_frame_ratio",
            "profile_low_contrast_frame_ratio",
            "profile_multi_peak_frame_ratio",
            "profile_saturated_frame_ratio",
            "profile_asymmetric_frame_ratio",
            "anomaly_fixed_in_20_frames",
            "fixed_evidence",
            "profile_v_px_json",
            "raw_profile_dn_json",
        ),
    )
    summary_path = output_dir / "spatial_structure_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "spatial_structure_report.md", jsonable(summary))

    candidate_us = [int(candidate["u_px"]) for candidate in candidate_records]
    # profile_rows contain JSON curves for CSV but plotting needs the in-memory arrays.
    profile_by_id = {
        row["candidate_id"]: {
            **row,
            "profile": all_profile_metrics[int(row["u_px"])][int(row["frame_index"]) - 1]["profile"],
            "v_axis": all_profile_metrics[int(row["u_px"])][int(row["frame_index"]) - 1]["v_axis"],
        }
        for row in profile_rows
    }
    plot_delta(
        output_dir / "steger_vs_centroid_delta_v.png",
        image_width,
        pair_matrix,
        candidate_us,
    )
    plot_decomposition(
        output_dir / "low_high_frequency_decomposition.png",
        image_width,
        steger_matrix,
        centroid_matrix,
        steger_low,
        centroid_low,
        steger_high,
        centroid_high,
    )
    plot_abnormal_locations(
        output_dir / "abnormal_u_locations.png",
        image_width,
        delta_abs_p95,
        jump_p95,
        np.maximum(steger_high_abs_p50, centroid_high_abs_p50),
        a1_quality_arrays["contrast_median"],
        category_masks,
        candidate_records,
    )
    plot_profile_grid(
        output_dir / "abnormal_profiles_grid.png",
        [profile_by_id[row["candidate_id"]] for row in profile_rows],
        "Selected abnormal raw DN profiles with Steger / centroid centers",
    )
    center_profile_rows = [
        row for row in profile_rows if row["region"] == "center"
    ][:8]
    edge_profile_rows = [
        row for row in profile_rows if row["region"] in {"left", "right"}
    ][:10]
    plot_profile_grid(
        output_dir / "center_region_profiles.png",
        [profile_by_id[row["candidate_id"]] for row in center_profile_rows],
        "Center-region raw DN profiles",
        columns=4,
    )
    plot_profile_grid(
        output_dir / "edge_region_profiles.png",
        [profile_by_id[row["candidate_id"]] for row in edge_profile_rows],
        "Edge-region raw DN profiles",
        columns=4,
    )

    print(f"空间结构诊断完成: {output_dir}")
    print(f"分类: {final_classification}")
    print(
        "全局 |delta_v| P50/P95/max: "
        f"{fmt(global_delta['abs_delta_px']['median'])}/"
        f"{fmt(global_delta['abs_delta_px']['p95'])}/"
        f"{fmt(global_delta['abs_delta_px']['max'])} px"
    )
    print(
        "中央高频 corr/same-direction: "
        f"{fmt(center['high_frequency_paired_correlation'])}/"
        f"{fmt(center['high_frequency_same_direction_ratio'] * 100, 1)}%"
    )
    print(
        "候选 profile: "
        f"{len(profile_rows)}; multi_peak={profile_multi_count}; "
        f"ridge_columns={profile_ridge_evidence_count}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"诊断失败: {error}", file=sys.stderr)
        raise SystemExit(2)
