"""可注入 ``extract_laser_center`` 的具体提取算法实现。

当前提供：
- ``centroid``：移植自 ``reconstruct_ground_pointcloud_v3.py`` 的
  背景抑制 + 逐列峰值邻域灰度重心 + 连续性分段 + 段内加权校正，
  已在真实 Mono8 饱和亮条纹上验证。
- ``steger``：移植并修正自 ``stripe_center_experiment`` 的高斯导数 +
  Hessian 主曲率 + 二阶泰勒亚像素定位。

新增算法只需实现 ``backend(image, options) -> (N, 2) ndarray`` 并注册到
``AVAILABLE_METHODS``，界面与配置即可直接选用。
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import cv2
import numpy as np

from .laser_extractor import LaserExtractionParams


_SCAN_AXES = ("column", "row")


@dataclass(frozen=True, slots=True)
class CentroidParams:
    """逐列重心法的全部可调参数（与 v3 配置字段一一对应）。"""

    background_kernel: int = 51
    min_local_contrast_dn: float = 20.0
    centroid_window_radius: int = 5
    segment_min_columns: int = 42
    continuity_max_column_gap: int = 2
    continuity_max_vertical_jump: float = 14.0
    correction_window: int = 7
    correction_max_shift: float = 3.5
    scan_axis: str = "column"

    def __post_init__(self) -> None:
        if self.background_kernel < 3 or self.background_kernel % 2 == 0:
            raise ValueError("background_kernel 必须是 >=3 的奇数")
        if self.centroid_window_radius < 1:
            raise ValueError("centroid_window_radius 必须 >= 1")
        if self.segment_min_columns < 1:
            raise ValueError("segment_min_columns 必须 >= 1")
        if self.continuity_max_column_gap < 1:
            raise ValueError("continuity_max_column_gap 必须 >= 1")
        if self.continuity_max_vertical_jump <= 0.0:
            raise ValueError("continuity_max_vertical_jump 必须为正数")
        if self.correction_window < 1 or self.correction_window % 2 == 0:
            raise ValueError("correction_window 必须是正奇数")
        if self.correction_max_shift < 0.0:
            raise ValueError("correction_max_shift 不能为负数")
        if self.scan_axis not in _SCAN_AXES:
            raise ValueError(f"scan_axis 必须是 {_SCAN_AXES} 之一")


@dataclass(frozen=True, slots=True)
class StegerParams:
    """Steger 条纹中心提取参数。"""

    sigma: float = 1.5
    threshold: float = 30.0
    deriv_thresh: float = 0.5
    roi_margin: int = 120
    roi_max_height: int = 512
    scan_axis: str = "column"

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma 必须为正数")
        if self.threshold < 0.0:
            raise ValueError("threshold 不能为负数")
        if self.deriv_thresh <= 0.0:
            raise ValueError("deriv_thresh 必须为正数")
        if self.roi_margin < 0:
            raise ValueError("roi_margin 不能为负数")
        if self.roi_max_height < 3:
            raise ValueError("roi_max_height 必须 >= 3")
        if self.scan_axis not in _SCAN_AXES:
            raise ValueError(f"scan_axis 必须是 {_SCAN_AXES} 之一")


@dataclass(frozen=True, slots=True)
class SharedStegerParams:
    """Shared Steger extractor parameters aligned with calibration V4."""

    sigma_px: float = 1.5
    max_offset_px: float = 0.75
    min_normal_y: float = 0.5
    min_response_ratio: float = 0.0005
    background_window_px: int = 31
    background_percentile: float = 20.0
    min_prominence_ratio: float = 0.010
    profile_smoothing_sigma_px: float = 0.8
    sensor_max_value: float | None = None
    segment_min_columns: int = 42
    continuity_max_column_gap: int = 2
    continuity_max_vertical_jump: float = 14.0
    post_filter: str = "reconstruction"
    scan_axis: str = "column"

    def __post_init__(self) -> None:
        if self.sigma_px <= 0.0:
            raise ValueError("sigma_px must be positive")
        if self.max_offset_px <= 0.0:
            raise ValueError("max_offset_px must be positive")
        if not 0.0 <= self.min_normal_y <= 1.0:
            raise ValueError("min_normal_y must be in [0, 1]")
        if self.min_response_ratio < 0.0:
            raise ValueError("min_response_ratio must be non-negative")
        if self.background_window_px <= 0 or self.background_window_px % 2 == 0:
            raise ValueError("background_window_px must be a positive odd integer")
        if not 0.0 <= self.background_percentile <= 100.0:
            raise ValueError("background_percentile must be in [0, 100]")
        if self.min_prominence_ratio < 0.0:
            raise ValueError("min_prominence_ratio must be non-negative")
        if self.profile_smoothing_sigma_px < 0.0:
            raise ValueError("profile_smoothing_sigma_px must be non-negative")
        if self.sensor_max_value is not None and self.sensor_max_value <= 0.0:
            raise ValueError("sensor_max_value must be positive or null")
        if self.segment_min_columns < 1:
            raise ValueError("segment_min_columns must be >= 1")
        if self.continuity_max_column_gap < 1:
            raise ValueError("continuity_max_column_gap must be >= 1")
        if self.continuity_max_vertical_jump <= 0.0:
            raise ValueError("continuity_max_vertical_jump must be positive")
        if self.post_filter not in ("reconstruction", "none"):
            raise ValueError("post_filter must be reconstruction or none")
        if self.scan_axis not in _SCAN_AXES:
            raise ValueError(f"scan_axis must be one of {_SCAN_AXES}")


def centroid_params_from_options(options: Mapping[str, Any]) -> CentroidParams:
    """把配置映射转换为 ``CentroidParams``，未知键报错。"""
    known = {field.name for field in fields(CentroidParams)}
    unknown = set(options) - known
    if unknown:
        raise ValueError(f"centroid 提取不认识的参数: {sorted(unknown)}")
    return CentroidParams(**dict(options))


def steger_params_from_options(options: Mapping[str, Any]) -> StegerParams:
    """把配置映射转换为 ``StegerParams``，未知键报错。"""
    known = {field.name for field in fields(StegerParams)}
    unknown = set(options) - known - {"search_roi", "_image_offset"}
    if unknown:
        raise ValueError(f"steger 提取不认识的参数: {sorted(unknown)}")
    if "search_roi" in options:
        _parse_search_roi(options["search_roi"])
    resolved = {key: value for key, value in options.items() if key in known}
    return StegerParams(**resolved)


def _parse_search_roi(value: Any) -> tuple[int, int, int, int]:
    """解析全幅传感器坐标中的 Steger 搜索矩形。"""
    if not isinstance(value, Mapping):
        raise ValueError("steger.search_roi 必须是包含 offset_x/offset_y/width/height 的映射")
    required = {"offset_x", "offset_y", "width", "height"}
    unknown = set(value) - required
    missing = required - set(value)
    if missing or unknown:
        raise ValueError(
            "steger.search_roi 字段错误："
            f"缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    parsed: list[int] = []
    for name in ("offset_x", "offset_y", "width", "height"):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError(f"steger.search_roi.{name} 必须是整数")
        parsed.append(int(raw))
    offset_x, offset_y, width, height = parsed
    if min(offset_x, offset_y) < 0:
        raise ValueError("steger.search_roi 偏移不能为负数")
    if min(width, height) <= 0:
        raise ValueError("steger.search_roi 宽高必须为正数")
    return offset_x, offset_y, width, height


def _parse_image_offset(value: Any) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("内部 image_offset 必须是两个整数")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in value):
        raise ValueError("内部 image_offset 必须是两个整数")
    offset = int(value[0]), int(value[1])
    if min(offset) < 0:
        raise ValueError("内部 image_offset 不能为负数")
    return offset


def shared_steger_params_from_options(
    options: Mapping[str, Any]
) -> SharedStegerParams:
    """Convert config mapping to ``SharedStegerParams``."""
    known = {field.name for field in fields(SharedStegerParams)}
    unknown = set(options) - known
    if unknown:
        raise ValueError(f"shared_steger 提取不认识的参数: {sorted(unknown)}")
    return SharedStegerParams(**dict(options))


def _correct_segment_v(
    values: np.ndarray, contrast: np.ndarray, params: CentroidParams
) -> np.ndarray:
    if params.correction_window == 1 or len(values) < 3:
        return values
    weights = contrast / max(float(np.max(contrast)), 1.0e-6) + 1.0e-4
    radius = params.correction_window // 2
    indexes = np.arange(len(values))
    left = np.maximum(0, indexes - radius)
    right = np.minimum(len(values), indexes + radius + 1)
    weight_prefix = np.concatenate(([0.0], np.cumsum(weights)))
    value_prefix = np.concatenate(([0.0], np.cumsum(weights * values)))
    estimates = (
        value_prefix[right] - value_prefix[left]
    ) / (weight_prefix[right] - weight_prefix[left])
    shifts = np.clip(
        estimates - values,
        -params.correction_max_shift,
        params.correction_max_shift,
    )
    return values + shifts


def _extract_columnwise(
    gray: np.ndarray, params: CentroidParams
) -> np.ndarray:
    """逐列提取亚像素灰度重心；假设条纹接近水平（每列一个中心）。"""
    background = cv2.GaussianBlur(
        gray, (params.background_kernel, params.background_kernel), 0
    )
    signal = cv2.subtract(gray, background).astype(np.float32)
    image_height, image_width = gray.shape
    columns = np.arange(image_width)
    peak_rows = np.argmax(signal, axis=0)
    peak_contrast = signal[peak_rows, columns]

    radius = params.centroid_window_radius
    offsets = np.arange(-radius, radius + 1)[:, None]
    raw_rows = peak_rows[None, :] + offsets
    inside = (raw_rows >= 0) & (raw_rows < image_height)
    sample_rows = np.clip(raw_rows, 0, image_height - 1)
    weights = np.take_along_axis(signal, sample_rows, axis=0) * inside
    weight_sum = weights.sum(axis=0)
    centre_v = np.full(image_width, np.nan, dtype=np.float64)
    nonzero = weight_sum > 0.0
    centre_v[nonzero] = (
        (weights[:, nonzero] * raw_rows[:, nonzero]).sum(axis=0)
        / weight_sum[nonzero]
    )
    valid = (
        nonzero
        & np.isfinite(centre_v)
        & (peak_contrast >= params.min_local_contrast_dn)
    )
    candidate_u = columns[valid].astype(np.float64)
    candidate_v = centre_v[valid]
    candidate_contrast = peak_contrast[valid].astype(np.float64)

    if len(candidate_u):
        breaks = np.where(
            (np.diff(candidate_u) > params.continuity_max_column_gap)
            | (
                np.abs(np.diff(candidate_v))
                > params.continuity_max_vertical_jump
            )
        )[0] + 1
        raw_segments = np.split(np.arange(len(candidate_u)), breaks)
    else:
        raw_segments = []
    accepted = [
        indexes
        for indexes in raw_segments
        if len(indexes) >= params.segment_min_columns
    ]
    points_by_segment: list[np.ndarray] = []
    for indexes in accepted:
        corrected_v = _correct_segment_v(
            candidate_v[indexes], candidate_contrast[indexes], params
        )
        points_by_segment.append(
            np.column_stack([candidate_u[indexes], corrected_v])
        )
    if not points_by_segment:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(points_by_segment, axis=0)


def centroid_backend(
    image: np.ndarray, options: Mapping[str, Any]
) -> np.ndarray:
    """逐列/逐行灰度重心提取，返回 ``(N, 2)`` 亚像素 ``(u, v)``。

    ``scan_axis="column"``：沿每一列找峰（条纹接近水平时使用）。
    ``scan_axis="row"``：沿每一行找峰（条纹接近竖直时使用），
    通过转置图像复用同一实现。
    """
    params = centroid_params_from_options(options)
    gray = np.ascontiguousarray(image)

    if params.scan_axis == "column":
        return _extract_columnwise(gray, params)

    transposed = _extract_columnwise(np.ascontiguousarray(gray.T), params)
    if transposed.size == 0:
        return transposed
    # 转置域的 (u', v') = (原 v, 原 u)，交换回 (u, v)。
    return np.ascontiguousarray(transposed[:, ::-1])


def _detect_steger_band(
    gray: np.ndarray, params: StegerParams
) -> tuple[int, int] | None:
    """沿扫描列方向定位亮条纹带，限制 Hessian 计算范围。"""
    row_peak = np.max(gray, axis=1)
    if float(np.max(row_peak)) < params.threshold:
        return None

    seed = int(np.sum(gray, axis=1, dtype=np.float64).argmax())
    adaptive_threshold = max(
        params.threshold, 0.3 * float(row_peak[seed])
    )
    active = row_peak >= adaptive_threshold
    top = seed
    while top > 0 and active[top - 1]:
        top -= 1
    bottom = seed
    while bottom < len(active) - 1 and active[bottom + 1]:
        bottom += 1

    top = max(0, top - params.roi_margin)
    bottom = min(gray.shape[0], bottom + params.roi_margin + 1)
    if bottom - top > params.roi_max_height:
        top = max(0, seed - params.roi_max_height // 2)
        bottom = min(gray.shape[0], top + params.roi_max_height)
        top = max(0, bottom - params.roi_max_height)
    return top, bottom


# 历史实现保留用于旧 import；公开 backend 已统一委托 realtime_steger。
def _extract_steger_columnwise(
    gray: np.ndarray, params: StegerParams
) -> np.ndarray:
    """对接近水平的亮条纹逐列执行 Steger 亚像素定位。"""
    band_bounds = _detect_steger_band(gray, params)
    if band_bounds is None:
        return np.empty((0, 2), dtype=np.float64)

    try:
        from scipy import ndimage
    except ImportError as error:
        raise RuntimeError(
            "Steger 算法需要 scipy；请执行 pip install -r requirements.txt"
        ) from error

    top, bottom = band_bounds
    band = np.asarray(gray[top:bottom], dtype=np.float64)
    ry = ndimage.gaussian_filter(band, params.sigma, order=(1, 0))
    rx = ndimage.gaussian_filter(band, params.sigma, order=(0, 1))
    ryy = ndimage.gaussian_filter(band, params.sigma, order=(2, 0))
    rxx = ndimage.gaussian_filter(band, params.sigma, order=(0, 2))
    rxy = ndimage.gaussian_filter(band, params.sigma, order=(1, 1))

    root = np.sqrt(((rxx - ryy) * 0.5) ** 2 + rxy**2)
    midpoint = (rxx + ryy) * 0.5
    eigenvalue_1 = midpoint + root
    eigenvalue_2 = midpoint - root
    main_eigenvalue = np.where(
        np.abs(eigenvalue_2) >= np.abs(eigenvalue_1),
        eigenvalue_2,
        eigenvalue_1,
    )

    # 两个等价特征向量候选中选范数较大的一个；这避免水平或竖直
    # 条纹的 rxy=0 情形退化为零向量。
    candidate_1_x = rxy
    candidate_1_y = main_eigenvalue - rxx
    candidate_2_x = main_eigenvalue - ryy
    candidate_2_y = rxy
    norm_1 = np.hypot(candidate_1_x, candidate_1_y)
    norm_2 = np.hypot(candidate_2_x, candidate_2_y)
    use_first = norm_1 >= norm_2
    normal_x = np.where(use_first, candidate_1_x, candidate_2_x)
    normal_y = np.where(use_first, candidate_1_y, candidate_2_y)
    normal_norm = np.hypot(normal_x, normal_y)
    safe_normal = normal_norm > np.finfo(np.float64).eps
    normal_x = np.divide(
        normal_x,
        normal_norm,
        out=np.zeros_like(normal_x),
        where=safe_normal,
    )
    normal_y = np.divide(
        normal_y,
        normal_norm,
        out=np.zeros_like(normal_y),
        where=safe_normal,
    )

    first_derivative = rx * normal_x + ry * normal_y
    second_derivative = (
        rxx * normal_x**2
        + 2.0 * rxy * normal_x * normal_y
        + ryy * normal_y**2
    )
    offset = np.divide(
        -first_derivative,
        second_derivative,
        out=np.full_like(first_derivative, np.nan),
        where=np.abs(second_derivative) > np.finfo(np.float64).eps,
    )
    offset_x = offset * normal_x
    offset_y = offset * normal_y
    valid = (
        safe_normal
        & (second_derivative < -params.deriv_thresh)
        & (np.abs(offset_x) <= 0.6)
        & (np.abs(offset_y) <= 0.6)
        & (band >= params.threshold)
    )

    strength = np.where(valid, -second_derivative, -np.inf)
    best_row = np.argmax(strength, axis=0)
    columns = np.arange(band.shape[1])
    best_strength = strength[best_row, columns]
    has_candidate = np.isfinite(best_strength) & (best_strength > 0.0)
    centre_v = best_row.astype(np.float64) + offset_y[best_row, columns] + top
    return np.column_stack(
        [columns[has_candidate].astype(np.float64), centre_v[has_candidate]]
    )


def steger_backend(
    image: np.ndarray, options: Mapping[str, Any]
) -> np.ndarray:
    """Steger/Hessian 亚像素中心提取，返回 ``(N, 2)`` 的 ``(u, v)``。"""
    resolved = dict(options)
    search_roi_value = resolved.pop("search_roi", None)
    image_offset = _parse_image_offset(resolved.pop("_image_offset", (0, 0)))
    params = steger_params_from_options(resolved)
    realtime = _load_realtime_steger_module()
    if search_roi_value is None:
        return realtime.steger_backend(image, resolved)

    roi_x, roi_y, roi_width, roi_height = _parse_search_roi(search_roi_value)
    frame_x, frame_y = image_offset
    image_height, image_width = image.shape
    left = max(frame_x, roi_x)
    top = max(frame_y, roi_y)
    right = min(frame_x + image_width, roi_x + roi_width)
    bottom = min(frame_y + image_height, roi_y + roi_height)
    if right <= left or bottom <= top:
        return np.empty((0, 2), dtype=np.float64)

    local_left = left - frame_x
    local_top = top - frame_y
    local_right = right - frame_x
    local_bottom = bottom - frame_y
    cropped = np.ascontiguousarray(
        image[local_top:local_bottom, local_left:local_right]
    )
    normal_extent = cropped.shape[1] if params.scan_axis == "row" else cropped.shape[0]
    search_region = realtime.LaserSearchRegion(
        0, normal_extent, "configured_search_roi"
    )
    points = realtime.steger_backend(
        cropped,
        resolved,
        search_region=search_region,
        use_auto_band=False,
    )
    if points.size:
        points = np.ascontiguousarray(points, dtype=np.float64)
        points[:, 0] += local_left
        points[:, 1] += local_top
    return points


def _load_realtime_steger_module() -> Any:
    """加载随测量工具发布的实时 Steger 实现。"""
    from . import realtime_steger

    return realtime_steger


def _load_shared_steger_module() -> Any:
    from . import steger_laser_center

    return steger_laser_center


def _extract_shared_steger_columnwise(
    gray: np.ndarray, params: SharedStegerParams
) -> np.ndarray:
    shared = _load_shared_steger_module()
    settings = shared.StegerSettings(
        sigma_px=params.sigma_px,
        max_offset_px=params.max_offset_px,
        min_normal_y=params.min_normal_y,
        min_response_ratio=params.min_response_ratio,
        background_window_px=params.background_window_px,
        background_percentile=params.background_percentile,
        min_prominence_ratio=params.min_prominence_ratio,
        profile_smoothing_sigma_px=params.profile_smoothing_sigma_px,
        sensor_max_value=params.sensor_max_value,
    )
    extracted = shared.extract_steger_columns(gray, settings)
    if params.post_filter == "none":
        return extracted.pixels
    points, _metadata = shared.points_from_valid_columns(
        extracted.u_px,
        extracted.v_px,
        extracted.valid,
        params.continuity_max_column_gap,
        params.continuity_max_vertical_jump,
        params.segment_min_columns,
    )
    return points


def shared_steger_backend(
    image: np.ndarray, options: Mapping[str, Any]
) -> np.ndarray:
    """兼容旧名称；实际调用统一的实时 Steger extractor。"""
    realtime = dict(options)
    if "sigma" not in realtime:
        realtime = {
            "sigma": realtime.get("sigma_px", 1.5),
            "threshold": realtime.get("threshold", 30.0),
            "deriv_thresh": realtime.get("deriv_thresh", 0.5),
            "roi_margin": realtime.get("roi_margin", 120),
            "roi_max_height": realtime.get("roi_max_height", 512),
            "scan_axis": realtime.get("scan_axis", "column"),
        }
    return _load_realtime_steger_module().steger_backend(image, realtime)


#: 界面与配置可选的提取算法。
AVAILABLE_METHODS: dict[str, Any] = {
    "centroid": centroid_backend,
    "steger": steger_backend,
    "shared_steger": shared_steger_backend,
}


def create_extraction_params(
    method: str, options: Mapping[str, Any] | None = None
) -> LaserExtractionParams:
    """按算法名构造 ``LaserExtractionParams``；未接入的算法 backend 为 None。"""
    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"未知提取算法 {method!r}；可选: {sorted(AVAILABLE_METHODS)}"
        )
    resolved_options = dict(options or {})
    if method == "centroid":
        centroid_params_from_options(resolved_options)
    elif method == "steger":
        steger_params_from_options(resolved_options)
    elif method == "shared_steger":
        if "sigma" in resolved_options:
            steger_params_from_options(resolved_options)
        else:
            shared_steger_params_from_options(resolved_options)
    return LaserExtractionParams(
        method=method,
        backend=AVAILABLE_METHODS[method],
        options=resolved_options,
    )
