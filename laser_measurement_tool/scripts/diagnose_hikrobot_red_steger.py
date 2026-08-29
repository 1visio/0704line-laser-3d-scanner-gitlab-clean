#!/usr/bin/env python3
"""海康 Mono8 红光条纹 / 实时 Steger 根因诊断。

本脚本只读取二维原始图像，不读取或反推任何三维结果。默认使用当前
``measure_tool.yaml`` 解析出的 Steger/centroid 参数，在同一批帧上完成：

* 固定列的原始纵向 profile 质量统计；
* sigma = 1.0 ... 4.0 的 Steger sweep；
* 当前 sigma 的跨帧逐列 temporal std；
* 当前 Steger 与 centroid 的 A/B 对照；
* CSV、JSON、Markdown 和无 GUI 绘图输出。

运行示例（仓库根目录）：

    .venv\\Scripts\\python.exe \\
      laser_measurement_tool\\scripts\\diagnose_hikrobot_red_steger.py

也可以显式传入录制目录。脚本本身会把 ``laser_measurement_tool`` 加入
``sys.path``，不依赖当前工作目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
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
SIGMA_SWEEP = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
CURRENT_SIGMA = 1.5
QUALITY_BACKGROUND_PERCENTILE = 20.0
QUALITY_SATURATION_DN = 255.0
TEMPORAL_STD_WARN_P50_PX = 0.25
TEMPORAL_STD_WARN_P95_PX = 0.50
TEMPORAL_MIN_VALID_FRAME_RATIO = 0.90

if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import AppConfig, load_app_config  # noqa: E402
from laser.backends import centroid_backend, steger_backend  # noqa: E402
from laser import realtime_steger  # noqa: E402


class DiagnosticError(RuntimeError):
    """输入或诊断协议不满足要求。"""


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
    """把 numpy 标量、Path 和非有限浮点转换成稳定 JSON 值。"""

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
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def read_image_unicode(path: Path) -> np.ndarray:
    """使用 fromfile+imdecode 读取 Windows/中文路径，并保留原始灰度 dtype。"""

    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DiagnosticError(f"无法解码图像: {path}")
    if image.ndim != 2:
        raise DiagnosticError(f"图像不是单通道灰度图: {path}，shape={image.shape}")
    if image.dtype != np.uint8:
        raise DiagnosticError(f"图像不是 Mono8 uint8: {path}，dtype={image.dtype}")
    return np.ascontiguousarray(image)


def read_frame_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DiagnosticError(f"缺少 frames.csv: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DiagnosticError(f"frames.csv 没有表头: {path}")
        rows = [dict(row) for row in reader]
    return rows


def discover_frames(recording_dir: Path, expected_count: int) -> tuple[list[Path], list[dict[str, str]]]:
    frame_paths = sorted(recording_dir.glob("frame_*.png"))
    if len(frame_paths) != expected_count:
        raise DiagnosticError(
            f"期望 {expected_count} 个 frame_*.png，实际 {len(frame_paths)} 个"
            f"（目录: {recording_dir}）"
        )
    metadata = read_frame_metadata(recording_dir / "frames.csv")
    if len(metadata) != expected_count:
        raise DiagnosticError(
            f"期望 frames.csv 有 {expected_count} 条数据，实际 {len(metadata)} 条"
        )
    names = [row.get("filename", "") for row in metadata]
    actual_names = [path.name for path in frame_paths]
    if names != actual_names:
        raise DiagnosticError("frames.csv 的 filename 顺序与 frame_*.png 不一致")
    if len(set(names)) != len(names):
        raise DiagnosticError("frames.csv 存在重复 filename")
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
                f"图像尺寸不一致: {path.name} 为 {image.shape}，期望 {shape}"
            )
        csv_width = row.get("width")
        csv_height = row.get("height")
        if csv_width and int(csv_width) != image.shape[1]:
            raise DiagnosticError(f"frames.csv width 与图像不一致: {path.name}")
        if csv_height and int(csv_height) != image.shape[0]:
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


def load_profile_options(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticError(f"Steger profile 不存在: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise DiagnosticError(f"无法读取 Steger profile {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise DiagnosticError(f"Steger profile 根节点必须是 mapping: {path}")
    options = document.get("steger", document.get("options", {}))
    if not isinstance(options, Mapping):
        raise DiagnosticError(f"Steger profile 缺少 steger mapping: {path}")
    return dict(options)


def profile_path_declared_by_config(config_path: Path) -> Path | None:
    """解析 extraction.profile，而不是假定 profile 文件名。"""

    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise DiagnosticError(
            f"无法读取配置以解析 extraction.profile: {config_path}: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise DiagnosticError(f"配置根节点必须是 mapping: {config_path}")
    extraction = document.get("extraction", {})
    if not isinstance(extraction, Mapping):
        raise DiagnosticError(f"配置 extraction 段必须是 mapping: {config_path}")
    value = extraction.get("profile")
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise DiagnosticError("extraction.profile 必须是路径字符串")
    raw = Path(value)
    return (config_path.parent / raw).resolve() if not raw.is_absolute() else raw.resolve()


def effective_search_roi(
    options: Mapping[str, Any], image_shape: tuple[int, int]
) -> dict[str, int]:
    height, width = image_shape
    value = options.get("search_roi")
    if value is None:
        return {"offset_x": 0, "offset_y": 0, "width": width, "height": height}
    if not isinstance(value, Mapping):
        raise DiagnosticError("steger.search_roi 必须是 mapping")
    required = ("offset_x", "offset_y", "width", "height")
    if any(key not in value for key in required):
        raise DiagnosticError("steger.search_roi 缺少 offset_x/offset_y/width/height")
    raw = {key: int(value[key]) for key in required}
    if raw["width"] <= 0 or raw["height"] <= 0:
        raise DiagnosticError("steger.search_roi 宽高必须为正数")
    left = max(0, raw["offset_x"])
    top = max(0, raw["offset_y"])
    right = min(width, raw["offset_x"] + raw["width"])
    bottom = min(height, raw["offset_y"] + raw["height"])
    if right <= left or bottom <= top:
        raise DiagnosticError(
            f"steger.search_roi 与图像无交集: {raw} vs shape={image_shape}"
        )
    return {
        **raw,
        "effective_left": left,
        "effective_top": top,
        "effective_right": right,
        "effective_bottom": bottom,
    }


def _linear_crossing(left_value: float, right_value: float, level: float) -> float:
    delta = right_value - left_value
    if abs(delta) <= np.finfo(float).eps:
        return 0.5
    return float(np.clip((level - left_value) / delta, 0.0, 1.0))


def profile_fwhm(profile: np.ndarray, peak_index: int, level: float) -> float:
    """在半高水平上做线性插值，返回像素单位 FWHM；边界截断返回 NaN。"""

    values = np.asarray(profile, dtype=np.float64)
    if peak_index <= 0 or peak_index >= len(values) - 1:
        return float("nan")
    left = peak_index
    while left > 0 and values[left - 1] >= level:
        left -= 1
    if left == 0 and values[left] >= level:
        return float("nan")
    right = peak_index
    while right < len(values) - 1 and values[right + 1] >= level:
        right += 1
    if right == len(values) - 1 and values[right] >= level:
        return float("nan")
    left_boundary = (left - 1) + _linear_crossing(
        float(values[left - 1]), float(values[left]), level
    )
    right_boundary = right + _linear_crossing(
        float(values[right]), float(values[right + 1]), level
    )
    width = right_boundary - left_boundary
    return float(width) if width > 0.0 else float("nan")


def region_for_u(u_px: float, image_width: int) -> str:
    one_third = image_width / 3.0
    if u_px < one_third:
        return "left"
    if u_px < 2.0 * one_third:
        return "center"
    return "right"


def measure_profile_quality(
    image: np.ndarray,
    u_px: int,
    roi: Mapping[str, int],
    frame_index: int,
    frame_name: str,
) -> dict[str, Any]:
    top = int(roi["effective_top"])
    bottom = int(roi["effective_bottom"])
    profile = image[top:bottom, int(u_px)].astype(np.float64, copy=False)
    peak_local = int(np.argmax(profile))
    peak_dn = float(profile[peak_local])
    background_dn = float(np.percentile(profile, QUALITY_BACKGROUND_PERCENTILE))
    contrast_dn = peak_dn - background_dn
    half_level = background_dn + 0.5 * contrast_dn
    fwhm_px = profile_fwhm(profile, peak_local, half_level) if contrast_dn > 0.0 else float("nan")
    saturation_count = int(np.count_nonzero(profile >= QUALITY_SATURATION_DN))
    return {
        "frame_index": frame_index,
        "frame_filename": frame_name,
        "u_px": int(u_px),
        "region": region_for_u(u_px, image.shape[1]),
        "profile_top_px": top,
        "profile_bottom_exclusive_px": bottom,
        "peak_v_px": top + peak_local,
        "peak_dn": peak_dn,
        "background_dn": background_dn,
        "local_contrast_dn": contrast_dn,
        "half_max_level_dn": half_level,
        "saturation_rate": saturation_count / max(len(profile), 1),
        "saturated_pixel_count": saturation_count,
        "peak_saturated": peak_dn >= QUALITY_SATURATION_DN,
        "fwhm_px": fwhm_px,
    }


def finite_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
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
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def bool_fraction(values: Iterable[bool]) -> float | None:
    array = np.asarray(list(values), dtype=bool)
    return float(np.mean(array)) if array.size else None


def fit_line_metrics(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return {
            "spatial_rms_px": None,
            "spatial_p95_px": None,
            "spatial_max_px": None,
            "line_slope_v_per_u": None,
            "line_angle_deg": None,
        }
    centre = np.mean(points, axis=0)
    _, _, right = np.linalg.svd(points - centre, full_matrices=False)
    direction = np.asarray(right[0], dtype=np.float64)
    if direction[0] < 0.0 or (abs(direction[0]) <= np.finfo(float).eps and direction[1] < 0.0):
        direction = -direction
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    residual = (points - centre) @ normal
    absolute = np.abs(residual)
    slope = (
        float(direction[1] / direction[0])
        if abs(direction[0]) > np.finfo(float).eps
        else None
    )
    return {
        "spatial_rms_px": float(np.sqrt(np.mean(residual**2))),
        "spatial_p95_px": float(np.percentile(absolute, 95)),
        "spatial_max_px": float(np.max(absolute)),
        "line_slope_v_per_u": slope,
        "line_angle_deg": float(np.degrees(np.arctan2(direction[1], direction[0]))),
    }


def vector_from_points(points: np.ndarray, image_width: int) -> np.ndarray:
    vector = np.full(image_width, np.nan, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not points.size:
        return vector
    columns = np.rint(points[:, 0]).astype(np.int64)
    valid = (columns >= 0) & (columns < image_width) & np.isfinite(points[:, 1])
    vector[columns[valid]] = points[valid, 1]
    return vector


def frame_metric(
    points: np.ndarray,
    image_width: int,
    expected_scanline_count: int,
    frame_index: int,
    frame_name: str,
    valid_count: int | None = None,
) -> dict[str, Any]:
    vector = vector_from_points(points, image_width)
    finite_count = (
        int(valid_count)
        if valid_count is not None
        else int(np.count_nonzero(np.isfinite(vector)))
    )
    return {
        "frame_index": frame_index,
        "frame_filename": frame_name,
        "expected_scanline_count": expected_scanline_count,
        "valid_points": finite_count,
        "valid_ratio": finite_count / max(expected_scanline_count, 1),
        **fit_line_metrics(points),
    }


def aggregate_frame_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in metrics if row.get(name) is not None]

    valid = finite_stats(values("valid_ratio"))
    rms = finite_stats(values("spatial_rms_px"))
    p95 = finite_stats(values("spatial_p95_px"))
    maximum = finite_stats(values("spatial_max_px"))
    return {
        "frame_count": len(metrics),
        "frames_with_line_fit": rms["count"],
        "valid_ratio_mean": valid["mean"],
        "valid_ratio_p05": valid["p05"],
        "valid_ratio_p50": valid["median"],
        "valid_ratio_p95": valid["p95"],
        "valid_ratio_min": valid["min"],
        "valid_ratio_max": valid["max"],
        "spatial_rms_mean_px": rms["mean"],
        "spatial_rms_p05_px": rms["p05"],
        "spatial_rms_p50_px": rms["median"],
        "spatial_rms_p95_px": rms["p95"],
        "spatial_rms_max_px": rms["max"],
        "spatial_p95_mean_px": p95["mean"],
        "spatial_p95_p50_px": p95["median"],
        "spatial_p95_p95_px": p95["p95"],
        "spatial_p95_max_px": p95["max"],
        "spatial_max_mean_px": maximum["mean"],
        "spatial_max_p50_px": maximum["median"],
        "spatial_max_p95_px": maximum["p95"],
        "spatial_max_max_px": maximum["max"],
        "valid_points_mean": finite_stats(
            [float(row["valid_points"]) for row in metrics]
        )["mean"],
    }


def effective_steger_call(
    image: np.ndarray,
    options: Mapping[str, Any],
    *,
    diagnostic: bool = False,
) -> dict[str, Any]:
    """走当前 wrapper 的同一 crop/search-region 语义，并保留逐列诊断字段。

    ``laser.backends.steger_backend`` 在配置了 ``search_roi`` 时正是按下面的
    crop 规则调用 ``realtime_steger``。这里直接调用其 ``extract_steger`` 以
    取得 wrapper 返回点数组之外的 offset 和 valid mask；公式和 search region
    没有复制，仍由当前实时模块执行。
    """

    resolved = dict(options)
    search_roi_value = resolved.pop("search_roi", None)
    resolved.pop("_image_offset", None)
    scan_axis = str(resolved.get("scan_axis", "column"))
    if scan_axis != "column":
        raise DiagnosticError("本诊断当前按横向列采样，要求 scan_axis=column")
    _height, width = image.shape
    if search_roi_value is None:
        left = top = 0
        cropped = image
        search_region = None
        expected_scanline_count = width
        extracted = realtime_steger.extract_steger(
            cropped,
            resolved,
            diagnostic=diagnostic,
            use_auto_band=True,
        )
    else:
        roi = effective_search_roi({"search_roi": search_roi_value}, image.shape)
        left = int(roi["effective_left"])
        top = int(roi["effective_top"])
        right = int(roi["effective_right"])
        bottom = int(roi["effective_bottom"])
        cropped = np.ascontiguousarray(image[top:bottom, left:right])
        search_region = realtime_steger.LaserSearchRegion(
            0,
            cropped.shape[0],
            "configured_search_roi",
        )
        expected_scanline_count = cropped.shape[1]
        extracted = realtime_steger.extract_steger(
            cropped,
            resolved,
            search_region=search_region,
            diagnostic=diagnostic,
            use_auto_band=False,
        )

    local_indices = np.flatnonzero(extracted.valid & np.isfinite(extracted.v_px))
    local_u = extracted.u_px[local_indices]
    local_v = extracted.v_px[local_indices]
    points = np.column_stack([local_u + left, local_v + top]).astype(np.float64)
    v_by_u = np.full(width, np.nan, dtype=np.float64)
    offset_by_u = np.full(width, np.nan, dtype=np.float64)
    response_by_u = np.full(width, np.nan, dtype=np.float64)
    global_columns = np.rint(local_u).astype(np.int64) + left
    in_bounds = (global_columns >= 0) & (global_columns < width)
    v_by_u[global_columns[in_bounds]] = local_v[in_bounds] + top
    offset_by_u[global_columns[in_bounds]] = extracted.offset_px[local_indices][in_bounds]
    response_by_u[global_columns[in_bounds]] = extracted.response[local_indices][in_bounds]
    return {
        "points": points,
        "v_by_u": v_by_u,
        "offset_by_u": offset_by_u,
        "response_by_u": response_by_u,
        "valid_column_count": int(local_indices.size),
        "expected_scanline_count": expected_scanline_count,
        "extraction": extracted,
        "crop": {"left": left, "top": top, "width": cropped.shape[1], "height": cropped.shape[0]},
    }


def rejection_counts(extraction: Any) -> dict[str, int]:
    diagnostics = getattr(extraction, "diagnostics", None)
    if diagnostics is None:
        return {}
    reasons = np.asarray(diagnostics.rejection_reason)
    return {str(key): int(value) for key, value in sorted(Counter(reasons).items())}


def sigma_row(
    sigma: float,
    frame_metrics: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_frame_metrics(frame_metrics)
    offsets = np.concatenate(
        [run["offset_by_u"][np.isfinite(run["offset_by_u"])] for run in runs]
    )
    offset = finite_stats(offsets)
    return {
        "sigma_px": sigma,
        **aggregate,
        "offset_count": offset["count"],
        "offset_mean_px": offset["mean"],
        "offset_p05_px": offset["p05"],
        "offset_p50_px": offset["median"],
        "offset_p95_px": offset["p95"],
        "offset_max_px": offset["max"],
    }


def temporal_rows(
    algorithm: str,
    vectors: np.ndarray,
    image_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_frames = vectors.shape[0]
    rows: list[dict[str, Any]] = []
    all_std_values: list[float] = []
    supported_std_values: list[float] = []
    valid_ratios: list[float] = []
    minimum_supported_frames = max(
        2, int(math.ceil(total_frames * TEMPORAL_MIN_VALID_FRAME_RATIO))
    )
    for u_px in range(image_width):
        values = vectors[:, u_px]
        finite = values[np.isfinite(values)]
        count = int(finite.size)
        ratio = count / max(total_frames, 1)
        std = float(np.std(finite, ddof=0)) if count >= 2 else float("nan")
        if math.isfinite(std):
            all_std_values.append(std)
            if count >= minimum_supported_frames:
                supported_std_values.append(std)
        valid_ratios.append(ratio)
        rows.append(
            {
                "algorithm": algorithm,
                "u_px": u_px,
                "region": region_for_u(u_px, image_width),
                "valid_frame_count": count,
                "total_frame_count": total_frames,
                "valid_frame_ratio": ratio,
                "mean_v_px": float(np.mean(finite)) if count else float("nan"),
                "temporal_std_px": std,
                "temporal_range_px": (
                    float(np.max(finite) - np.min(finite)) if count else float("nan")
                ),
                "min_v_px": float(np.min(finite)) if count else float("nan"),
                "max_v_px": float(np.max(finite)) if count else float("nan"),
            }
        )
    all_std_stats = finite_stats(all_std_values)
    std_stats = finite_stats(supported_std_values or all_std_values)
    summary = {
        "algorithm": algorithm,
        "total_frames": total_frames,
        "columns_total": image_width,
        "minimum_supported_frames": minimum_supported_frames,
        "minimum_supported_frame_ratio": TEMPORAL_MIN_VALID_FRAME_RATIO,
        "columns_with_at_least_two_frames": all_std_stats["count"],
        "columns_with_required_frame_support": std_stats["count"],
        "valid_frame_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else None,
        "temporal_std_mean_px": std_stats["mean"],
        "temporal_std_p05_px": std_stats["p05"],
        "temporal_std_p50_px": std_stats["median"],
        "temporal_std_p95_px": std_stats["p95"],
        "temporal_std_max_px": std_stats["max"],
        "temporal_std_all_columns_p50_px": all_std_stats["median"],
        "temporal_std_all_columns_p95_px": all_std_stats["p95"],
        "temporal_std_all_columns_max_px": all_std_stats["max"],
        "columns_over_p50_warning": int(np.count_nonzero(np.asarray(supported_std_values or all_std_values) > TEMPORAL_STD_WARN_P50_PX)),
        "columns_over_p95_warning": int(np.count_nonzero(np.asarray(supported_std_values or all_std_values) > TEMPORAL_STD_WARN_P95_PX)),
    }
    return rows, summary


def algorithm_spatial_summary(
    algorithm: str,
    frame_metrics: Sequence[Mapping[str, Any]],
    sigma_px: float | None = None,
) -> dict[str, Any]:
    aggregate = aggregate_frame_metrics(frame_metrics)
    return {"algorithm": algorithm, "sigma_px": sigma_px, **aggregate}


def build_comparison_rows(
    algorithms: Mapping[str, Mapping[str, Any]],
    frame_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for algorithm, payload in algorithms.items():
        sigma_px = payload.get("sigma_px")
        frame_metrics = payload["frame_metrics"]
        for metric in frame_metrics:
            rows.append(
                {
                    "scope": "frame",
                    "algorithm": algorithm,
                    "sigma_px": sigma_px,
                    "frame_index": metric["frame_index"],
                    "frame_filename": metric["frame_filename"],
                    "expected_scanline_count": metric["expected_scanline_count"],
                    "valid_points": metric["valid_points"],
                    "valid_ratio": metric["valid_ratio"],
                    "spatial_rms_px": metric["spatial_rms_px"],
                    "spatial_p95_px": metric["spatial_p95_px"],
                    "spatial_max_px": metric["spatial_max_px"],
                    "temporal_columns": None,
                    "temporal_std_mean_px": None,
                    "temporal_std_p50_px": None,
                    "temporal_std_p95_px": None,
                    "temporal_std_max_px": None,
                }
            )
        spatial = algorithm_spatial_summary(algorithm, frame_metrics, sigma_px)
        summaries[algorithm] = {
            "spatial": spatial,
            "temporal": payload["temporal_summary"],
        }
        rows.append(
            {
                "scope": "spatial_summary",
                "algorithm": algorithm,
                "sigma_px": sigma_px,
                "frame_index": None,
                "frame_filename": None,
                "expected_scanline_count": None,
                "valid_points": spatial["valid_points_mean"],
                "valid_ratio": spatial["valid_ratio_p50"],
                "spatial_rms_px": spatial["spatial_rms_p50_px"],
                "spatial_p95_px": spatial["spatial_p95_p50_px"],
                "spatial_max_px": spatial["spatial_max_p50_px"],
                "temporal_columns": None,
                "temporal_std_mean_px": None,
                "temporal_std_p50_px": None,
                "temporal_std_p95_px": None,
                "temporal_std_max_px": None,
            }
        )
        temporal = payload["temporal_summary"]
        rows.append(
            {
                "scope": "temporal_summary",
                "algorithm": algorithm,
                "sigma_px": sigma_px,
                "frame_index": None,
                "frame_filename": None,
                "expected_scanline_count": None,
                "valid_points": None,
                "valid_ratio": None,
                "spatial_rms_px": None,
                "spatial_p95_px": None,
                "spatial_max_px": None,
                "temporal_columns": temporal["columns_with_at_least_two_frames"],
                "temporal_std_mean_px": temporal["temporal_std_mean_px"],
                "temporal_std_p50_px": temporal["temporal_std_p50_px"],
                "temporal_std_p95_px": temporal["temporal_std_p95_px"],
                "temporal_std_max_px": temporal["temporal_std_max_px"],
            }
        )
    return rows, summaries


def recommend_sigma(sweep_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finite_rows = [
        row
        for row in sweep_rows
        if row.get("spatial_rms_p50_px") is not None
        and row.get("valid_ratio_p50") is not None
    ]
    if not finite_rows:
        return {
            "best_sigma_by_spatial_rms_px": None,
            "best_sigma_by_valid_ratio": None,
            "recommended_sigma_range_px": None,
            "current_sigma_match": "UNDETERMINED",
            "current_sigma_rms_improvement_vs_best": None,
            "current_sigma_valid_ratio_delta_vs_best": None,
        }
    best_spatial_row = min(
        finite_rows,
        key=lambda row: (float(row["spatial_rms_p50_px"]), -float(row["valid_ratio_p50"])),
    )
    best_valid_row = max(
        finite_rows,
        key=lambda row: (float(row["valid_ratio_p50"]), -float(row["spatial_rms_p50_px"])),
    )
    best_rms = float(best_spatial_row["spatial_rms_p50_px"])
    best_valid = float(best_valid_row["valid_ratio_p50"])
    candidates = [
        row
        for row in finite_rows
        if float(row["spatial_rms_p50_px"]) <= max(best_rms * 1.10, best_rms + 0.02)
        and float(row["valid_ratio_p50"]) >= best_valid - 0.02
    ]
    if not candidates:
        candidates = [best_spatial_row]
    recommended = [float(row["sigma_px"]) for row in candidates]
    current = next((row for row in finite_rows if float(row["sigma_px"]) == CURRENT_SIGMA), None)
    if current is None:
        match = "UNDETERMINED"
        improvement = None
        valid_delta = None
    else:
        current_rms = float(current["spatial_rms_p50_px"])
        current_valid = float(current["valid_ratio_p50"])
        improvement = (
            (current_rms - best_rms) / current_rms if abs(current_rms) > np.finfo(float).eps else 0.0
        )
        valid_delta = best_valid - current_valid
        match = (
            "MISMATCHED"
            if float(current["sigma_px"]) not in recommended
            and (improvement >= 0.10 or valid_delta >= 0.05)
            else "MATCHED"
        )
    return {
        "best_sigma_by_spatial_rms_px": float(best_spatial_row["sigma_px"]),
        "best_sigma_by_valid_ratio": float(best_valid_row["sigma_px"]),
        "recommended_sigma_range_px": [min(recommended), max(recommended)],
        "recommended_sigma_values_px": sorted(recommended),
        "current_sigma_match": match,
        "current_sigma_rms_improvement_vs_best": improvement,
        "current_sigma_valid_ratio_delta_vs_best": valid_delta,
    }


def classify_root_cause(
    quality_summary: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
    temporal_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    quality_flags: list[str] = []
    fwhm = quality_summary["fwhm_stats"]
    contrast = quality_summary["contrast_stats"]
    saturation = quality_summary["saturation_rate_stats"]
    peak_saturated_fraction = quality_summary["peak_saturated_fraction"]
    if quality_summary["fwhm_valid_ratio"] is not None and quality_summary["fwhm_valid_ratio"] < 0.90:
        quality_flags.append("FWHM_VALID_RATIO_LT_90_PERCENT")
    if contrast["p05"] is not None and contrast["p05"] < 20.0:
        quality_flags.append("LOCAL_CONTRAST_P05_LT_20_DN")
    if peak_saturated_fraction is not None and peak_saturated_fraction > 0.25:
        quality_flags.append("PEAK_SATURATION_GT_25_PERCENT")
    if saturation["p95"] is not None and saturation["p95"] > 0.05:
        quality_flags.append("PROFILE_SATURATION_P95_GT_5_PERCENT")

    scale_flag = scale_summary.get("current_sigma_match") == "MISMATCHED"
    scale_flags = ["CURRENT_SIGMA_NOT_IN_RECOMMENDED_RANGE"] if scale_flag else []

    temporal_flags: list[str] = []
    steger_temporal = temporal_summary.get("steger", {})
    if (
        steger_temporal.get("temporal_std_p50_px") is not None
        and steger_temporal["temporal_std_p50_px"] > TEMPORAL_STD_WARN_P50_PX
    ):
        temporal_flags.append("STEGER_TEMPORAL_STD_P50_GT_0_25_PX")
    if (
        steger_temporal.get("temporal_std_p95_px") is not None
        and steger_temporal["temporal_std_p95_px"] > TEMPORAL_STD_WARN_P95_PX
    ):
        temporal_flags.append("STEGER_TEMPORAL_STD_P95_GT_0_50_PX")

    causes = []
    if quality_flags:
        causes.append("STRIPE_QUALITY_LIMITED")
    if scale_flag:
        causes.append("STEGER_SCALE_MISMATCH")
    if temporal_flags:
        causes.append("TEMPORAL_INSTABILITY")
    if len(causes) == 1:
        classification = causes[0]
    else:
        classification = "MIXED"
    return {
        "classification": classification,
        "quality_flags": quality_flags,
        "scale_flags": scale_flags,
        "temporal_flags": temporal_flags,
        "causes_detected": causes,
        "decision_thresholds": {
            "contrast_p05_dn": 20.0,
            "fwhm_valid_ratio": 0.90,
            "peak_saturation_fraction": 0.25,
            "profile_saturation_rate_p95": 0.05,
            "temporal_std_p50_px": TEMPORAL_STD_WARN_P50_PX,
            "temporal_std_p95_px": TEMPORAL_STD_WARN_P95_PX,
            "sigma_rms_relative_improvement": 0.10,
            "sigma_valid_ratio_gain": 0.05,
        },
    }


def make_quality_plot(
    path: Path,
    sample_columns: np.ndarray,
    values: np.ndarray,
    image_width: int,
    ylabel: str,
    title: str,
) -> None:
    median = np.full(values.shape[1], np.nan)
    p05 = np.full(values.shape[1], np.nan)
    p95 = np.full(values.shape[1], np.nan)
    for index in range(values.shape[1]):
        stats = finite_stats(values[:, index])
        median[index] = np.nan if stats["median"] is None else stats["median"]
        p05[index] = np.nan if stats["p05"] is None else stats["p05"]
        p95[index] = np.nan if stats["p95"] is None else stats["p95"]
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(sample_columns, median, color="#b2182b", linewidth=1.4, label="median over 20 frames")
    axis.fill_between(sample_columns, p05, p95, color="#ef8a62", alpha=0.28, label="P05–P95")
    width = float(image_width)
    axis.axvline(width / 3.0, color="0.5", linestyle="--", linewidth=0.8)
    axis.axvline(2.0 * width / 3.0, color="0.5", linestyle="--", linewidth=0.8)
    axis.set_xlabel("u (px)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_sigma_plot(
    path: Path,
    sweep_rows: Sequence[Mapping[str, Any]],
    value_name: str,
    ylabel: str,
    title: str,
) -> None:
    x = np.asarray([float(row["sigma_px"]) for row in sweep_rows])
    y = np.asarray(
        [float(row[value_name]) if row.get(value_name) is not None else np.nan for row in sweep_rows]
    )
    if value_name == "spatial_rms_p50_px":
        lower_name, upper_name = "spatial_rms_p05_px", "spatial_rms_p95_px"
    else:
        lower_name, upper_name = "valid_ratio_p05", "valid_ratio_p95"
    lower = np.asarray(
        [float(row[lower_name]) if row.get(lower_name) is not None else np.nan for row in sweep_rows]
    )
    upper = np.asarray(
        [float(row[upper_name]) if row.get(upper_name) is not None else np.nan for row in sweep_rows]
    )
    fig, axis = plt.subplots(figsize=(8, 5.2))
    axis.plot(x, y, "o-", color="#2166ac", linewidth=1.6, label="median over frames")
    axis.fill_between(x, lower, upper, color="#67a9cf", alpha=0.25, label="P05–P95 over frames")
    axis.axvline(CURRENT_SIGMA, color="#b2182b", linestyle="--", linewidth=1.0, label="current sigma=1.5")
    axis.set_xlabel("Steger sigma (px)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_temporal_plot(
    path: Path,
    temporal_rows_by_algorithm: Mapping[str, Sequence[Mapping[str, Any]]],
    image_width: int,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.5))
    colors = {"steger": "#b2182b", "centroid": "#2166ac"}
    for algorithm, rows in temporal_rows_by_algorithm.items():
        x = np.asarray([int(row["u_px"]) for row in rows])
        y = np.asarray(
            [float(row["temporal_std_px"]) if row.get("temporal_std_px") is not None else np.nan for row in rows]
        )
        axis.plot(x, y, linewidth=1.0, color=colors.get(algorithm, None), label=algorithm)
    axis.axhline(TEMPORAL_STD_WARN_P50_PX, color="0.45", linestyle="--", linewidth=0.8, label="P50 warning 0.25 px")
    axis.axhline(TEMPORAL_STD_WARN_P95_PX, color="0.45", linestyle=":", linewidth=0.8, label="P95 warning 0.50 px")
    axis.axvline(image_width / 3.0, color="0.65", linestyle="--", linewidth=0.7)
    axis.axvline(2.0 * image_width / 3.0, color="0.65", linestyle="--", linewidth=0.7)
    axis.set_xlabel("u (px)")
    axis.set_ylabel("temporal std of v (px)")
    axis.set_title("20-frame temporal repeatability")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_overlay_plot(
    path: Path,
    image: np.ndarray,
    steger_points: np.ndarray,
    centroid_points: np.ndarray,
    roi: Mapping[str, int],
    frame_name: str,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper", interpolation="nearest")
    if len(centroid_points):
        axis.scatter(
            centroid_points[:, 0],
            centroid_points[:, 1],
            s=1.0,
            c="#2c7bb6",
            alpha=0.75,
            linewidths=0,
            label=f"centroid ({len(centroid_points)})",
        )
    if len(steger_points):
        axis.scatter(
            steger_points[:, 0],
            steger_points[:, 1],
            s=1.2,
            c="#d7191c",
            alpha=0.75,
            linewidths=0,
            label=f"steger sigma=1.5 ({len(steger_points)})",
        )
    axis.add_patch(
        Rectangle(
            (roi["effective_left"], roi["effective_top"]),
            roi["effective_right"] - roi["effective_left"],
            roi["effective_bottom"] - roi["effective_top"],
            fill=False,
            edgecolor="#fdae61",
            linewidth=1.0,
            linestyle="--",
            label="configured Steger search ROI",
        )
    )
    axis.set_xlabel("u (px)")
    axis.set_ylabel("v (px)")
    axis.set_title(f"Centroid vs Steger overlay · {frame_name}")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def unique_capture_values(metadata: Sequence[Mapping[str, str]], key: str) -> list[str]:
    return sorted({str(row.get(key, "")) for row in metadata})


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    quality = summary["stripe_quality"]
    scale = summary["scale_assessment"]
    classification = summary["classification"]
    temporal = summary["temporal_repeatability"]["algorithms"]
    comparison = summary["centroid_vs_steger"]["algorithms"]
    sweep = summary["sigma_sweep"]["rows"]
    region_rows = []
    for region in ("left", "center", "right", "all"):
        item = quality["regions"][region]
        region_rows.append(
            [
                region,
                item["sample_count"],
                item["peak_dn"]["median"],
                item["background_dn"]["median"],
                item["contrast_dn"]["median"],
                item["fwhm_px"]["median"],
                item["fwhm_px"]["p05"],
                item["fwhm_px"]["p95"],
                item["peak_saturated_fraction"] * 100
                if item["peak_saturated_fraction"] is not None
                else None,
                item["fwhm_valid_ratio"] * 100
                if item["fwhm_valid_ratio"] is not None
                else None,
            ]
        )
    sigma_rows = [
        [
            row["sigma_px"],
            row["valid_ratio_p50"],
            row["valid_ratio_p05"],
            row["spatial_rms_p50_px"],
            row["spatial_rms_p95_px"],
            row["spatial_p95_p50_px"],
            row["spatial_max_p50_px"],
            row["offset_p50_px"],
            row["offset_p95_px"],
        ]
        for row in sweep
    ]
    algorithm_rows = []
    for algorithm in ("steger", "centroid"):
        spatial = comparison[algorithm]["spatial"]
        repeat = temporal[algorithm]
        algorithm_rows.append(
            [
                algorithm,
                spatial["valid_ratio_p50"],
                spatial["spatial_rms_p50_px"],
                spatial["spatial_p95_p50_px"],
                spatial["spatial_max_p50_px"],
                repeat["temporal_std_p50_px"],
                repeat["temporal_std_p95_px"],
                repeat["temporal_std_max_px"],
            ]
        )

    lines = [
        "# 海康红光 Steger 根因诊断报告",
        "",
        f"- 诊断分类：**`{classification['classification']}`**",
        f"- 生成时间（UTC）：`{summary['generated_at_utc']}`",
        f"- 输入帧：`{summary['input']['frame_count']}` 帧，`{summary['input']['image_size'][0]}×{summary['input']['image_size'][1]}`，`{summary['input']['dtype']}`",
        "- 本报告只使用原始二维 Mono8 图像；没有使用 `height_shadow.csv` 或任何三维结果。",
        "",
        "## 1. 数据与协议",
        "",
        f"输入目录：`{summary['input']['recording_dir']}`。`frames.csv` 与 PNG 文件名、尺寸和顺序一致；采集参数唯一值为：曝光 `{', '.join(summary['input']['capture_settings']['exposure_us'])}` µs，增益 `{', '.join(summary['input']['capture_settings']['gain_db'])}` dB，像素格式 `{', '.join(summary['input']['capture_settings']['pixel_format'])}`。",
        "",
        "本轮没有同协议既有诊断结果可复用；`height_shadow.csv` 仅被检查并明确排除。所有质量、sigma、时间和 A/B 数值均为本轮新增计算。",
        "",
        "Steger sweep 只改变 `sigma`；`threshold`、`deriv_thresh`、`roi_margin`、`roi_max_height`、`scan_axis` 和 `search_roi` 全部固定为当前配置。",
        "",
        "## 2. 原始条纹质量",
        "",
        f"profile 在当前 Steger search ROI 的纵向截面上测量；background 为该 profile 的 P{QUALITY_BACKGROUND_PERCENTILE:g}，local contrast = peak − background，FWHM 在 raw DN 的 background + 0.5×contrast 水平上做线性插值，saturation rate 为 profile 内 DN≥255 的像素比例。",
        "",
        markdown_table(
            ["区域", "样本数", "peak DN 中位", "background DN 中位", "contrast DN 中位", "FWHM 中位(px)", "FWHM P05", "FWHM P95", "peak 饱和比例(%)", "FWHM 有效率(%)"],
            region_rows,
        ),
        "",
        f"全体典型 background-relative FWHM：中位 **{fmt(quality['fwhm_stats']['median'])} px**，P05–P95 = **{fmt(quality['fwhm_stats']['p05'])}–{fmt(quality['fwhm_stats']['p95'])} px**，FWHM 有效率 **{fmt(quality['fwhm_valid_ratio'] * 100 if quality['fwhm_valid_ratio'] is not None else None, 1)}%**。",
        "",
        "## 3. Sigma sweep",
        "",
        markdown_table(
            ["sigma(px)", "valid ratio P50", "valid ratio P05", "spatial RMS P50(px)", "spatial RMS P95(px)", "spatial P95 P50(px)", "spatial max P50(px)", "offset P50(px)", "offset P95(px)"],
            sigma_rows,
        ),
        "",
        f"按跨帧 spatial RMS P50，最优 sigma = **{fmt(scale['best_sigma_by_spatial_rms_px'], 1)}**；按 valid ratio P50，最优 sigma = **{fmt(scale['best_sigma_by_valid_ratio'], 1)}**。综合保留空间残差不超过最优值约 10%、且 valid ratio 不低于最高值 2 个百分点的范围：**{fmt(scale['recommended_sigma_range_px'][0] if scale['recommended_sigma_range_px'] else None, 1)}–{fmt(scale['recommended_sigma_range_px'][1] if scale['recommended_sigma_range_px'] else None, 1)} px**。",
        "",
        "## 4. 跨帧重复性与 A/B",
        "",
        markdown_table(
            ["算法", "valid ratio P50", "spatial RMS P50(px)", "spatial P95 P50(px)", "spatial max P50(px)", "temporal std P50(px)", "temporal std P95(px)", "temporal std max(px)"],
            algorithm_rows,
        ),
        "",
        f"`spatial residual` 是每帧对 `(u,v)` 点做 total-least-squares 直线拟合后的正交距离；`temporal std` 是 20 帧同一整数列 `u` 的 `v` 坐标标准差，使用 `ddof=0`。temporal 汇总优先只纳入至少 {temporal['steger']['minimum_supported_frames']} 帧有效的列，CSV 仍保留每列的实际有效帧数。",
        "",
        f"当前 Steger 的 temporal std P50/P95 = **{fmt(temporal['steger']['temporal_std_p50_px'])}/{fmt(temporal['steger']['temporal_std_p95_px'])} px**；centroid 为 **{fmt(temporal['centroid']['temporal_std_p50_px'])}/{fmt(temporal['centroid']['temporal_std_p95_px'])} px**。",
        "",
        "## 5. 根因判断",
        "",
        f"质量触发项：`{', '.join(classification['quality_flags']) if classification['quality_flags'] else '无'}`。",
        f"尺度触发项：`{', '.join(classification['scale_flags']) if classification['scale_flags'] else '无'}`。",
        f"时间触发项：`{', '.join(classification['temporal_flags']) if classification['temporal_flags'] else '无'}`。",
        "",
        "判定规则已写入 `diagnostic_summary.json`；它们是诊断分层阈值，不会写回生产配置。",
        "",
        "## 6. 最后回答三个问题",
        "",
        f"1. 海康红光实际典型 background-relative FWHM：**{fmt(quality['fwhm_stats']['median'])} px**（本批有效 profile 的 P05–P95 为 {fmt(quality['fwhm_stats']['p05'])}–{fmt(quality['fwhm_stats']['p95'])} px）。",
        f"2. 当前 `sigma=1.5` 是否匹配：**{scale['current_sigma_match']}**。当前值相对 sweep 最优空间 RMS 的改善空间为 {fmt(scale['current_sigma_rms_improvement_vs_best'] * 100 if scale['current_sigma_rms_improvement_vs_best'] is not None else None, 1)}%，valid ratio 差值为 {fmt(scale['current_sigma_valid_ratio_delta_vs_best'] * 100 if scale['current_sigma_valid_ratio_delta_vs_best'] is not None else None, 1)} 个百分点。推荐范围为 **{fmt(scale['recommended_sigma_range_px'][0] if scale['recommended_sigma_range_px'] else None, 1)}–{fmt(scale['recommended_sigma_range_px'][1] if scale['recommended_sigma_range_px'] else None, 1)} px**。",
        f"3. 中心线波动主要来源：**`{classification['classification']}`**。应结合上面的 raw stripe quality、sigma sweep 和 temporal std 证据理解；本轮没有修改 `measure_tool.yaml`、`realtime_steger.yaml`、标定文件或生产参数。",
        "",
        "## 7. 产物",
        "",
        "- `stripe_quality_by_u.csv`：每帧、每个固定采样列的 peak/background/contrast/saturation/FWHM。",
        "- `sigma_sweep.csv`：7 组 sigma 的跨帧汇总，以及 Steger offset 分布。",
        "- `temporal_repeatability.csv`：Steger 与 centroid 在每个 u 上的 20 帧 temporal std。",
        "- `centroid_vs_steger.csv`：逐帧 spatial 指标与算法级 temporal 汇总。",
        "- `fwhm_vs_u.png`、`contrast_vs_u.png`、`sigma_vs_spatial_rms.png`、`sigma_vs_valid_ratio.png`、`temporal_std_vs_u.png`、`centroid_vs_steger_overlay.png`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 20 帧海康 Mono8 原始图像的红光 Steger 根因诊断",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("recording_dir", nargs="?", type=Path, default=DEFAULT_RECORDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steger-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--column-step", type=positive_int, default=16)
    parser.add_argument("--expected-frames", type=positive_int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recording_dir = args.recording_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    profile_path = args.steger_profile.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else recording_dir / "steger_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, frame_metadata = discover_frames(recording_dir, args.expected_frames)
    frames = load_frames(frame_paths, frame_metadata)
    image_height, image_width = frames[0]["image"].shape
    config: AppConfig = load_app_config(config_path)
    profile_options = load_profile_options(profile_path)
    configured_profile_path = profile_path_declared_by_config(config_path)
    configured_steger_options = dict(config.extraction_options)
    centroid_options = dict(config.extraction_options_by_method.get("centroid", {}))
    compare_keys = ("sigma", "threshold", "deriv_thresh", "roi_margin", "roi_max_height", "scan_axis")
    profile_mismatches = {
        key: {"config": configured_steger_options.get(key), "profile": profile_options.get(key)}
        for key in compare_keys
        if key in profile_options
        and key in configured_steger_options
        and profile_options[key] != configured_steger_options[key]
    }
    if profile_mismatches:
        raise DiagnosticError(
            "外部 profile 与当前配置解析出的 profile 不一致，拒绝混用: "
            + json.dumps(jsonable(profile_mismatches), ensure_ascii=False)
        )
    if configured_steger_options.get("scan_axis", "column") != "column":
        raise DiagnosticError("当前诊断要求当前配置 scan_axis=column")

    roi = effective_search_roi(configured_steger_options, (image_height, image_width))
    sample_columns = np.arange(
        roi["effective_left"], roi["effective_right"], args.column_step, dtype=np.int64
    )
    if sample_columns.size == 0:
        raise DiagnosticError("固定列采样为空")

    quality_rows: list[dict[str, Any]] = []
    quality_grid: dict[str, list[list[float]]] = {
        "fwhm_px": [],
        "local_contrast_dn": [],
    }
    quality_metric_names = (
        "peak_dn",
        "background_dn",
        "local_contrast_dn",
        "saturation_rate",
        "fwhm_px",
    )
    for frame in frames:
        frame_values: dict[str, list[float]] = {name: [] for name in quality_metric_names}
        for u_px in sample_columns:
            row = measure_profile_quality(
                frame["image"],
                int(u_px),
                roi,
                int(frame["index"]),
                str(frame["filename"]),
            )
            quality_rows.append(row)
            for name in quality_metric_names:
                frame_values[name].append(float(row[name]))
        quality_grid["fwhm_px"].append(frame_values["fwhm_px"])
        quality_grid["local_contrast_dn"].append(frame_values["local_contrast_dn"])

    all_quality = {name: [row[name] for row in quality_rows] for name in quality_metric_names}
    quality_regions: dict[str, Any] = {}
    for region in ("left", "center", "right", "all"):
        region_rows = (
            quality_rows
            if region == "all"
            else [row for row in quality_rows if row["region"] == region]
        )
        quality_regions[region] = {
            "sample_count": len(region_rows),
            "peak_dn": finite_stats(row["peak_dn"] for row in region_rows),
            "background_dn": finite_stats(row["background_dn"] for row in region_rows),
            "contrast_dn": finite_stats(row["local_contrast_dn"] for row in region_rows),
            "saturation_rate": finite_stats(row["saturation_rate"] for row in region_rows),
            "fwhm_px": finite_stats(row["fwhm_px"] for row in region_rows),
            "peak_saturated_fraction": bool_fraction(row["peak_saturated"] for row in region_rows),
            "fwhm_valid_ratio": (
                finite_stats(row["fwhm_px"] for row in region_rows)["count"] / len(region_rows)
                if region_rows
                else None
            ),
        }
    quality_summary = {
        "sample_step_px": args.column_step,
        "sample_columns": [int(value) for value in sample_columns],
        "sample_count_per_frame": int(sample_columns.size),
        "profile_roi": roi,
        "background_percentile": QUALITY_BACKGROUND_PERCENTILE,
        "saturation_dn": QUALITY_SATURATION_DN,
        "fwhm_definition": "raw profile width at background_dn + 0.5 * (peak_dn - background_dn), with background_dn=P20",
        "peak_dn_stats": finite_stats(all_quality["peak_dn"]),
        "background_dn_stats": finite_stats(all_quality["background_dn"]),
        "contrast_stats": finite_stats(all_quality["local_contrast_dn"]),
        "saturation_rate_stats": finite_stats(all_quality["saturation_rate"]),
        "fwhm_stats": finite_stats(all_quality["fwhm_px"]),
        "fwhm_valid_ratio": (
            finite_stats(all_quality["fwhm_px"])["count"] / len(all_quality["fwhm_px"])
            if all_quality["fwhm_px"]
            else None
        ),
        "peak_saturated_fraction": bool_fraction(row["peak_saturated"] for row in quality_rows),
        "regions": quality_regions,
    }
    write_csv(
        output_dir / "stripe_quality_by_u.csv",
        quality_rows,
        (
            "frame_index",
            "frame_filename",
            "u_px",
            "region",
            "profile_top_px",
            "profile_bottom_exclusive_px",
            "peak_v_px",
            "peak_dn",
            "background_dn",
            "local_contrast_dn",
            "half_max_level_dn",
            "saturation_rate",
            "saturated_pixel_count",
            "peak_saturated",
            "fwhm_px",
        ),
    )

    sigma_rows: list[dict[str, Any]] = []
    sigma_frame_metrics: dict[str, list[dict[str, Any]]] = {}
    sigma_runs: dict[str, list[dict[str, Any]]] = {}
    current_steger_runs: list[dict[str, Any]] | None = None
    current_rejection_counts: list[dict[str, int]] = []
    for sigma in SIGMA_SWEEP:
        options = dict(configured_steger_options)
        options["sigma"] = sigma
        runs: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for frame in frames:
            run = effective_steger_call(
                frame["image"],
                options,
                diagnostic=(sigma == CURRENT_SIGMA),
            )
            runs.append(run)
            metrics.append(
                frame_metric(
                    run["points"],
                    image_width,
                    int(run["expected_scanline_count"]),
                    int(frame["index"]),
                    str(frame["filename"]),
                    valid_count=int(run["valid_column_count"]),
                )
            )
            if sigma == CURRENT_SIGMA:
                current_rejection_counts.append(rejection_counts(run["extraction"]))
        key = f"{sigma:.1f}"
        sigma_frame_metrics[key] = metrics
        sigma_runs[key] = runs
        row = sigma_row(sigma, metrics, runs)
        sigma_rows.append(row)
        if sigma == CURRENT_SIGMA:
            current_steger_runs = runs
    if current_steger_runs is None:
        raise DiagnosticError("sigma sweep 未产生当前 sigma=1.5 结果")
    write_csv(
        output_dir / "sigma_sweep.csv",
        sigma_rows,
        (
            "sigma_px",
            "frame_count",
            "frames_with_line_fit",
            "valid_points_mean",
            "valid_ratio_mean",
            "valid_ratio_p05",
            "valid_ratio_p50",
            "valid_ratio_p95",
            "valid_ratio_min",
            "valid_ratio_max",
            "spatial_rms_mean_px",
            "spatial_rms_p05_px",
            "spatial_rms_p50_px",
            "spatial_rms_p95_px",
            "spatial_rms_max_px",
            "spatial_p95_mean_px",
            "spatial_p95_p50_px",
            "spatial_p95_p95_px",
            "spatial_p95_max_px",
            "spatial_max_mean_px",
            "spatial_max_p50_px",
            "spatial_max_p95_px",
            "spatial_max_max_px",
            "offset_count",
            "offset_mean_px",
            "offset_p05_px",
            "offset_p50_px",
            "offset_p95_px",
            "offset_max_px",
        ),
    )

    centroid_points_by_frame: list[np.ndarray] = []
    centroid_vectors: list[np.ndarray] = []
    centroid_frame_metrics: list[dict[str, Any]] = []
    for frame in frames:
        points = np.asarray(centroid_backend(frame["image"], centroid_options), dtype=np.float64)
        centroid_points_by_frame.append(points)
        centroid_vectors.append(vector_from_points(points, image_width))
        centroid_frame_metrics.append(
            frame_metric(
                points,
                image_width,
                image_width,
                int(frame["index"]),
                str(frame["filename"]),
                valid_count=len(points),
            )
        )
    steger_vectors = np.asarray([run["v_by_u"] for run in current_steger_runs])
    centroid_vectors_array = np.asarray(centroid_vectors)
    temporal_rows_by_algorithm: dict[str, list[dict[str, Any]]] = {}
    temporal_summaries: dict[str, dict[str, Any]] = {}
    for algorithm, vectors in (
        ("steger", steger_vectors),
        ("centroid", centroid_vectors_array),
    ):
        rows, temporal_summary = temporal_rows(algorithm, vectors, image_width)
        temporal_rows_by_algorithm[algorithm] = rows
        temporal_summaries[algorithm] = temporal_summary
    write_csv(
        output_dir / "temporal_repeatability.csv",
        [row for algorithm in ("steger", "centroid") for row in temporal_rows_by_algorithm[algorithm]],
        (
            "algorithm",
            "u_px",
            "region",
            "valid_frame_count",
            "total_frame_count",
            "valid_frame_ratio",
            "mean_v_px",
            "temporal_std_px",
            "temporal_range_px",
            "min_v_px",
            "max_v_px",
        ),
    )

    algorithms = {
        "steger": {
            "sigma_px": CURRENT_SIGMA,
            "frame_metrics": sigma_frame_metrics[f"{CURRENT_SIGMA:.1f}"],
            "temporal_summary": temporal_summaries["steger"],
        },
        "centroid": {
            "sigma_px": None,
            "frame_metrics": centroid_frame_metrics,
            "temporal_summary": temporal_summaries["centroid"],
        },
    }
    comparison_rows, comparison_summaries = build_comparison_rows(algorithms, len(frames))
    write_csv(
        output_dir / "centroid_vs_steger.csv",
        comparison_rows,
        (
            "scope",
            "algorithm",
            "sigma_px",
            "frame_index",
            "frame_filename",
            "expected_scanline_count",
            "valid_points",
            "valid_ratio",
            "spatial_rms_px",
            "spatial_p95_px",
            "spatial_max_px",
            "temporal_columns",
            "temporal_std_mean_px",
            "temporal_std_p50_px",
            "temporal_std_p95_px",
            "temporal_std_max_px",
        ),
    )

    equivalence = effective_steger_call(frames[0]["image"], configured_steger_options)["points"]
    wrapper_points = np.asarray(steger_backend(frames[0]["image"], configured_steger_options), dtype=np.float64)
    if len(equivalence) != len(wrapper_points):
        backend_equivalence = {
            "passed": False,
            "point_count_direct": len(equivalence),
            "point_count_wrapper": len(wrapper_points),
            "max_abs_delta_px": None,
        }
    elif len(equivalence):
        direct_order = np.lexsort((equivalence[:, 1], equivalence[:, 0]))
        wrapper_order = np.lexsort((wrapper_points[:, 1], wrapper_points[:, 0]))
        delta = np.abs(equivalence[direct_order] - wrapper_points[wrapper_order])
        backend_equivalence = {
            "passed": bool(np.allclose(delta, 0.0, rtol=0.0, atol=1.0e-12)),
            "point_count_direct": len(equivalence),
            "point_count_wrapper": len(wrapper_points),
            "max_abs_delta_px": float(np.max(delta)),
        }
    else:
        backend_equivalence = {
            "passed": True,
            "point_count_direct": 0,
            "point_count_wrapper": 0,
            "max_abs_delta_px": 0.0,
        }

    scale_assessment = recommend_sigma(sigma_rows)
    quality_classification = classify_root_cause(
        quality_summary,
        scale_assessment,
        temporal_summaries,
    )

    quality_grid_array = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in quality_grid.items()
    }
    make_quality_plot(
        output_dir / "fwhm_vs_u.png",
        sample_columns,
        quality_grid_array["fwhm_px"],
        image_width,
        "FWHM (px)",
        "Raw stripe FWHM vs u · 20-frame P05–P95",
    )
    make_quality_plot(
        output_dir / "contrast_vs_u.png",
        sample_columns,
        quality_grid_array["local_contrast_dn"],
        image_width,
        "local contrast (DN)",
        "Raw stripe local contrast vs u · 20-frame P05–P95",
    )
    make_sigma_plot(
        output_dir / "sigma_vs_spatial_rms.png",
        sigma_rows,
        "spatial_rms_p50_px",
        "spatial RMS after line fit (px)",
        "Steger sigma vs spatial residual",
    )
    make_sigma_plot(
        output_dir / "sigma_vs_valid_ratio.png",
        sigma_rows,
        "valid_ratio_p50",
        "valid point ratio",
        "Steger sigma vs valid point ratio",
    )
    make_temporal_plot(
        output_dir / "temporal_std_vs_u.png",
        temporal_rows_by_algorithm,
        image_width,
    )
    make_overlay_plot(
        output_dir / "centroid_vs_steger_overlay.png",
        frames[len(frames) // 2]["image"],
        current_steger_runs[len(frames) // 2]["points"],
        centroid_points_by_frame[len(frames) // 2],
        roi,
        str(frames[len(frames) // 2]["filename"]),
    )

    current_rejections = Counter()
    for counts in current_rejection_counts:
        current_rejections.update(counts)
    input_summary = {
        "recording_dir": recording_dir,
        "frames_csv": recording_dir / "frames.csv",
        "frame_count": len(frames),
        "frame_filenames": [frame["filename"] for frame in frames],
        "image_size": [image_width, image_height],
        "dtype": str(frames[0]["image"].dtype),
        "pixel_min": int(min(np.min(frame["image"]) for frame in frames)),
        "pixel_max": int(max(np.max(frame["image"]) for frame in frames)),
        "frames_csv_filename_order_matches": True,
        "capture_settings": {
            key: unique_capture_values(frame_metadata, key)
            for key in ("exposure_us", "gain_db", "pixel_format", "offset_x", "offset_y", "width", "height", "frame_gap")
        },
    }
    provenance = {
        "reused_artifacts": [],
        "inspected_but_excluded_artifacts": [recording_dir / "height_shadow.csv"],
        "new_calculations": [
            "raw Mono8 vertical profiles",
            "Steger sigma sweep",
            "20-frame temporal repeatability",
            "centroid vs Steger A/B",
        ],
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "profile_path": profile_path,
        "profile_sha256": sha256_file(profile_path),
        "config_referenced_profile_path": configured_profile_path,
        "config_referenced_profile_sha256": (
            sha256_file(configured_profile_path)
            if configured_profile_path is not None and configured_profile_path.is_file()
            else None
        ),
        "implementation_sha256": {
            "laser_backends.py": sha256_file(TOOL_ROOT / "laser" / "backends.py"),
            "laser_realtime_steger.py": sha256_file(TOOL_ROOT / "laser" / "realtime_steger.py"),
            "laser_centroid_and_shared_adapter.py": sha256_file(TOOL_ROOT / "laser" / "steger_laser_center.py"),
        },
        "external_profile_matches_config": not profile_mismatches,
        "production_configuration_modified": False,
        "production_code_parameters_modified": False,
    }
    summary = {
        "schema_version": 1,
        "task": "hikrobot_red_steger_root_cause_diagnosis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input": input_summary,
        "provenance": provenance,
        "parameters": {
            "steger_options_fixed": configured_steger_options,
            "centroid_options_fixed": centroid_options,
            "sigma_sweep_px": list(SIGMA_SWEEP),
            "current_sigma_px": CURRENT_SIGMA,
            "sample_column_step_px": args.column_step,
            "profile_options": profile_options,
        },
        "stripe_quality": quality_summary,
        "sigma_sweep": {
            "rows": sigma_rows,
            "per_frame": sigma_frame_metrics,
            "current_sigma_rejection_reason_counts": dict(sorted(current_rejections.items())),
        },
        "scale_assessment": scale_assessment,
        "temporal_repeatability": {
            "algorithms": temporal_summaries,
            "warning_thresholds_px": {
                "p50": TEMPORAL_STD_WARN_P50_PX,
                "p95": TEMPORAL_STD_WARN_P95_PX,
            },
        },
        "centroid_vs_steger": {
            "algorithms": comparison_summaries,
            "backend_equivalence_check_first_frame_current_sigma": backend_equivalence,
        },
        "classification": quality_classification,
    }
    summary_path = output_dir / "diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "diagnostic_report.md", jsonable(summary))

    print(f"诊断完成: {output_dir}")
    print(f"分类: {quality_classification['classification']}")
    print(
        "典型 FWHM: "
        f"{fmt(quality_summary['fwhm_stats']['median'])} px "
        f"(P05–P95 {fmt(quality_summary['fwhm_stats']['p05'])}–{fmt(quality_summary['fwhm_stats']['p95'])} px)"
    )
    print(
        "推荐 sigma: "
        f"{fmt(scale_assessment['recommended_sigma_range_px'][0] if scale_assessment['recommended_sigma_range_px'] else None, 1)}–"
        f"{fmt(scale_assessment['recommended_sigma_range_px'][1] if scale_assessment['recommended_sigma_range_px'] else None, 1)} px; "
        f"current 1.5 = {scale_assessment['current_sigma_match']}"
    )
    print(
        "Steger temporal std P50/P95: "
        f"{fmt(temporal_summaries['steger']['temporal_std_p50_px'])}/"
        f"{fmt(temporal_summaries['steger']['temporal_std_p95_px'])} px"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"诊断失败: {error}", file=sys.stderr)
        raise SystemExit(2) from error
