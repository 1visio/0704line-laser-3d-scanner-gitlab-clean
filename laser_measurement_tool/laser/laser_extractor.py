"""激光中心提取的稳定接口与算法适配协议。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


LaserCenterArray: TypeAlias = NDArray[np.float64]


class LaserCenterBackend(Protocol):
    """Steger、Gaussian 或 RANSAC 实现需要满足的调用协议。"""

    def __call__(
        self,
        image: np.ndarray,
        options: Mapping[str, Any],
    ) -> ArrayLike:
        """返回按 ``(u, v)`` 排列的中心点。"""
        ...


@dataclass(frozen=True, slots=True)
class LaserExtractionParams:
    """激光中心提取算法及其参数。"""

    method: str = "steger"
    backend: LaserCenterBackend | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


LaserExtractionParamsInput: TypeAlias = (
    LaserExtractionParams | Mapping[str, Any]
)


class LaserExtractionError(RuntimeError):
    """激光中心提取失败。"""


class LaserAlgorithmNotConfiguredError(LaserExtractionError):
    """尚未为所选方法接入实际算法。"""


def extract_laser_center(
    image: np.ndarray,
    params: LaserExtractionParamsInput,
    *,
    image_offset: tuple[int, int] = (0, 0),
) -> LaserCenterArray:
    """调用已注入算法，返回形状为 ``(N, 2)`` 的亚像素 ``(u, v)`` 数组。"""
    grayscale = np.asarray(image)
    if grayscale.ndim != 2:
        raise ValueError("激光中心提取只接受二维灰度图像")
    if not np.issubdtype(grayscale.dtype, np.number):
        raise TypeError("灰度图像必须是数值数组")

    configuration = _coerce_params(params)
    if configuration.backend is None:
        raise LaserAlgorithmNotConfiguredError(
            f"{configuration.method} 算法尚未接入；请通过 params.backend 注入现有实现"
        )
    if not callable(configuration.backend):
        raise TypeError("params.backend 必须是可调用对象")

    options = configuration.options
    if "search_roi" in options:
        options = dict(options)
        options["_image_offset"] = image_offset

    try:
        raw_centers = configuration.backend(grayscale, options)
    except Exception as error:
        raise LaserExtractionError(
            f"{configuration.method} 激光中心提取失败: {error}"
        ) from error

    return _validate_centers(raw_centers)


def _coerce_params(params: LaserExtractionParamsInput) -> LaserExtractionParams:
    if isinstance(params, LaserExtractionParams):
        configuration = params
    elif isinstance(params, Mapping):
        configuration = LaserExtractionParams(
            method=str(params.get("method", "steger")),
            backend=params.get("backend"),
            options=params.get("options", {}),
        )
    else:
        raise TypeError("params 必须是 LaserExtractionParams 或映射")

    method = configuration.method.strip()
    if not method:
        raise ValueError("params.method 不能为空")
    if not isinstance(configuration.options, Mapping):
        raise TypeError("params.options 必须是映射")
    return configuration


def _validate_centers(raw_centers: ArrayLike) -> LaserCenterArray:
    centers = np.asarray(raw_centers, dtype=np.float64)
    if centers.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise LaserExtractionError("算法输出必须是形状为 (N, 2) 的 (u, v) 数组")
    if not np.isfinite(centers).all():
        raise LaserExtractionError("算法输出包含 NaN 或无穷值")
    return np.ascontiguousarray(centers)
