"""亚像素激光中心点到地面坐标系三维点的多模型重建。

支持 ``global_plane``、``quadratic_graph`` 和 ``circular_cone`` 三种
相机坐标系激光表面。统一流程是去畸变得到相机射线、与激光表面求交，
再用 ``T_ground_from_camera`` 变换到地面系；所有长度单位为 mm。
没有 ``laser_model`` 时仍兼容旧格式 ``plane_abcd``。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import warnings

import cv2
import numpy as np

from .laser_ray_correction import (
    FrozenLaserRayCorrection,
    LaserRayCorrectionError,
    evaluate_frozen_laser_ray_correction,
)


class ReconstructionInputError(ValueError):
    """重建输入（点、标定或参数）不满足约束。"""


@dataclass(frozen=True, slots=True)
class ReconstructionParams:
    """射线-激光表面求交的数值、工作距离与可选图像 ROI 约束。"""

    parallel_epsilon: float = 1.0e-9
    quadratic_epsilon: float = 1.0e-12
    min_camera_depth_mm: float = 100.0
    max_camera_depth_mm: float = 1500.0
    # 模型自身 z_valid_range_mm 的边界外扩，避免边界噪声误删。
    model_range_margin_mm: float = 50.0
    # 固定姿态下的棋盘格内部多边形，坐标为原始图像像素 (u, v)。
    # None 表示不启用图像 ROI，保持历史全幅重建行为。
    image_roi_polygon: tuple[tuple[float, float], ...] | None = None
    # Frozen C1 is opt-in; the historical C0-only path remains the default.
    enable_laser_ray_correction: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enable_laser_ray_correction, bool):
            raise ReconstructionInputError(
                "enable_laser_ray_correction 必须是布尔值"
            )
        if self.parallel_epsilon <= 0.0:
            raise ReconstructionInputError("parallel_epsilon 必须为正数")
        if self.quadratic_epsilon <= 0.0:
            raise ReconstructionInputError("quadratic_epsilon 必须为正数")
        if self.model_range_margin_mm < 0.0:
            raise ReconstructionInputError("model_range_margin_mm 不能为负数")
        if not 0.0 <= self.min_camera_depth_mm < self.max_camera_depth_mm:
            raise ReconstructionInputError(
                "工作距离必须满足 0 <= min_camera_depth_mm < max_camera_depth_mm"
            )
        if self.image_roi_polygon is None:
            return
        try:
            polygon = np.asarray(self.image_roi_polygon, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ReconstructionInputError(
                "image_roi_polygon 必须是至少 3 个 (u, v) 像素坐标"
            ) from error
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or polygon.shape[0] < 3
            or not np.isfinite(polygon).all()
        ):
            raise ReconstructionInputError(
                "image_roi_polygon 必须是至少 3 个有限的 (u, v) 像素坐标"
            )
        # 拒绝退化多边形，避免 ROI 开关打开后静默保留/丢弃全部点。
        area_twice = float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
        if abs(area_twice) <= np.finfo(np.float64).eps:
            raise ReconstructionInputError("image_roi_polygon 不能是退化多边形")
        object.__setattr__(
            self,
            "image_roi_polygon",
            tuple((float(u), float(v)) for u, v in polygon),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """有效点的像素坐标、相机系坐标与地面系坐标（逐行对齐）。"""

    pixels_uv: np.ndarray
    points_camera: np.ndarray
    points_ground: np.ndarray
    filtered: dict[str, int]
    # Frozen-C0 camera points are retained only as aligned diagnostics.  They
    # are not used to replace the C0+C1 output coordinates.
    points_camera_c0: np.ndarray | None = None
    q1_c0: np.ndarray | None = None
    q2_c0: np.ndarray | None = None
    c1_clamped: np.ndarray | None = None

    @property
    def point_count(self) -> int:
        return len(self.pixels_uv)


def build_ground_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """由 R、t 组装 4x4 的 ``T_ground_from_camera``。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(R, dtype=np.float64)
    transform[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return transform


def apply_ground_u_compensation(
    points_ground: np.ndarray,
    pixels_uv: np.ndarray,
    compensation: Mapping[str, Any] | None,
) -> np.ndarray:
    """按配置的图像轴逐点插值，并执行 ``Zg_corrected = Zg_raw - bias``。

    函数名和 ``ground_u_compensation`` 配置键保留用于旧标定包兼容；没有
    ``compensation_axis`` 的历史表始终按 ``u`` 解释。
    """
    points = np.asarray(points_ground, dtype=np.float64)
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    if compensation is None or len(points) == 0:
        return np.ascontiguousarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ReconstructionInputError("points_ground 必须是形状为 (N, 3) 的数组")
    if pixels.ndim != 2 or pixels.shape != (len(points), 2):
        raise ReconstructionInputError("pixels_uv 必须与 points_ground 逐行对齐")

    raw_axis = compensation.get("compensation_axis", "u")
    axis = str(raw_axis).strip().lower()
    if axis not in {"u", "v"}:
        raise ReconstructionInputError(
            f"ground_u_compensation.compensation_axis 必须是 u 或 v，实际为 {raw_axis!r}"
        )
    coordinate_key = "column_u_px" if axis == "u" else "row_v_px"
    try:
        coordinate_values = (
            compensation["coordinate_px"]
            if "coordinate_px" in compensation
            else compensation[coordinate_key]
        )
        columns = np.asarray(coordinate_values, dtype=np.float64).reshape(-1)
        bias = np.asarray(compensation["bias_mm"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionInputError(
            f"ground_u_compensation 必须包含数值数组 {coordinate_key} 和 bias_mm"
        ) from error
    if len(columns) == 0 or len(columns) != len(bias):
        raise ReconstructionInputError("ground_u_compensation 两列必须非空且等长")
    if not np.isfinite(columns).all() or not np.isfinite(bias).all():
        raise ReconstructionInputError("ground_u_compensation 包含 NaN 或无穷值")
    if np.any(np.diff(columns) <= 0.0):
        raise ReconstructionInputError(
            f"ground_u_compensation 的 {coordinate_key} 必须严格递增"
        )

    z_offset = _compensation_z_offset(compensation)
    corrected = points.copy()
    coordinate = pixels[:, 0 if axis == "u" else 1]
    outside = (coordinate < columns[0]) | (coordinate > columns[-1])
    if np.any(outside):
        warnings.warn(
            f"{np.count_nonzero(outside)} point(s) are outside the ground-bias "
            f"{axis} range [{columns[0]:g}, {columns[-1]:g}] px; "
            "using the nearest endpoint bias",
            RuntimeWarning,
            stacklevel=2,
        )
    corrected[:, 2] -= np.interp(coordinate, columns, bias)
    corrected[:, 2] -= z_offset
    return np.ascontiguousarray(corrected)


def _compensation_z_offset(compensation: Mapping[str, Any]) -> float:
    if "z_offset_mm" not in compensation:
        return 0.0
    try:
        value = np.asarray(compensation["z_offset_mm"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ReconstructionInputError(
            "ground_u_compensation 的 z_offset_mm 必须是数值"
        ) from error
    if value.size != 1 or not np.isfinite(value).all():
        raise ReconstructionInputError(
            "ground_u_compensation 的 z_offset_mm 必须是单个有限数值"
        )
    return float(value.reshape(-1)[0])


def _axis_index(name: str) -> int:
    lookup = {"X": 0, "Y": 1, "Z": 2}
    try:
        return lookup[str(name).upper()]
    except KeyError as error:
        raise ReconstructionInputError(f"不支持的坐标轴名称：{name!r}") from error


def _model_z_range(model: Mapping[str, Any]) -> tuple[float, float] | None:
    value = model.get("z_valid_range_mm")
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ReconstructionInputError(
            "z_valid_range_mm 必须是递增的两个有限数值"
        ) from error
    if arr.size != 2 or not np.isfinite(arr).all() or arr[0] >= arr[1]:
        raise ReconstructionInputError("z_valid_range_mm 必须是递增的两个有限数值")
    return float(arr[0]), float(arr[1])


def _solve_quadratic_all(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """逐行求 a*x^2+b*x+c=0 的实根；无实根位置为 NaN。"""
    aa, bb, cc = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
    )
    roots = np.full((aa.size, 2), np.nan, dtype=np.float64)
    af, bf, cf = aa.reshape(-1), bb.reshape(-1), cc.reshape(-1)

    linear = np.abs(af) < epsilon
    valid_linear = linear & (np.abs(bf) >= epsilon)
    roots[valid_linear, 0] = -cf[valid_linear] / bf[valid_linear]

    quadratic = ~linear
    discriminant = bf * bf - 4.0 * af * cf
    # 浮点舍入可能把理论上的切线根算成极小负数，按相对尺度容忍。
    discriminant_scale = np.maximum(
        1.0, np.maximum(np.abs(bf * bf), np.abs(4.0 * af * cf))
    )
    valid_quadratic = quadratic & (
        discriminant >= -epsilon * discriminant_scale
    )
    if np.any(valid_quadratic):
        sqrt_disc = np.sqrt(np.maximum(discriminant[valid_quadratic], 0.0))
        bq = bf[valid_quadratic]
        aq = af[valid_quadratic]
        cq = cf[valid_quadratic]
        # 比直接 (-b +- sqrt(D))/(2a) 更稳定的形式。
        q = -0.5 * (bq + np.copysign(sqrt_disc, bq))
        r1 = q / aq
        r2 = np.where(
            np.abs(q) >= epsilon,
            cq / q,
            (-bq - sqrt_disc) / (2.0 * aq),
        )
        roots[valid_quadratic, 0] = r1
        roots[valid_quadratic, 1] = r2
    return roots


def _choose_roots(
    roots: np.ndarray,
    rays: np.ndarray,
    params: ReconstructionParams,
    z_range: tuple[float, float] | None,
    apex: np.ndarray | None = None,
    axis: np.ndarray | None = None,
) -> np.ndarray:
    """从两个实根中选择物理有效根。

    归一化射线第三分量恒为 1，因此 lambda 就是相机深度 Zc。
    """
    # ``roots`` is normally an ``(N, 2)`` array from ``_solve_quadratic_all``.
    # Keep the selection fully vectorized: the previous implementation walked
    # every ray in Python and performed the same masking/argmin work N times.
    chosen = np.full(len(rays), np.nan, dtype=np.float64)
    lo = params.min_camera_depth_mm
    hi = params.max_camera_depth_mm
    if z_range is not None:
        model_lo = z_range[0] - params.model_range_margin_mm
        model_hi = z_range[1] + params.model_range_margin_mm
        lo = max(lo, model_lo)
        hi = min(hi, model_hi)
        hint = 0.5 * (z_range[0] + z_range[1])
    else:
        hint = 0.5 * (lo + hi)
    if lo > hi:
        return chosen

    root_values = np.asarray(roots, dtype=np.float64)
    valid = (
        np.isfinite(root_values)
        & (root_values > 0.0)
        & (root_values >= lo)
        & (root_values <= hi)
    )
    if apex is not None and axis is not None:
        # For a candidate point lambda*r, the forward test is
        # ((lambda*r - apex) · axis) >= 0.  Computing the two dot products
        # once avoids allocating an (N, 2, 3) temporary for every frame.
        ray_axis = rays @ axis
        apex_axis = float(np.asarray(apex, dtype=np.float64) @ axis)
        valid &= (root_values * ray_axis[:, None] - apex_axis) >= 0.0

    distance = np.where(valid, np.abs(root_values - hint), np.inf)
    best_index = np.argmin(distance, axis=1)
    chosen = np.take_along_axis(
        root_values, best_index[:, None], axis=1
    )[:, 0]
    chosen[~np.any(valid, axis=1)] = np.nan
    return chosen


def _intersect_global_plane(
    rays: np.ndarray,
    model: Mapping[str, Any],
    params: ReconstructionParams,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        normal = np.asarray(model["normal"], dtype=np.float64).reshape(3)
        d = float(model["d_mm"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionInputError(
            "global_plane 需要 normal(3) 和 d_mm"
        ) from error
    norm = float(np.linalg.norm(normal))
    if norm <= np.finfo(np.float64).eps:
        raise ReconstructionInputError("global_plane.normal 不能为零向量")
    normal = normal / norm
    d /= norm
    denominator = rays @ normal
    stable = np.abs(denominator) > params.parallel_epsilon
    lam = np.full(len(rays), np.nan, dtype=np.float64)
    lam[stable] = -d / denominator[stable]
    lam[~np.isfinite(lam)] = np.nan
    return lam, stable


def _intersect_quadratic_graph(
    rays: np.ndarray,
    model: Mapping[str, Any],
    params: ReconstructionParams,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        dep_axis = _axis_index(str(model["dependent_axis"]))
        ind_names = list(model["independent_axes"])
        if len(ind_names) != 2:
            raise ValueError("independent_axes 长度必须为 2")
        ind_axes = (_axis_index(ind_names[0]), _axis_index(ind_names[1]))
        center = np.asarray(
            model["normalization"]["independent_center_mm"],
            dtype=np.float64,
        ).reshape(2)
        scale = np.asarray(
            model["normalization"]["independent_scale_mm"],
            dtype=np.float64,
        ).reshape(2)
        beta = np.asarray(model["coefficients"], dtype=np.float64).reshape(6)
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionInputError("quadratic_graph 模型参数不完整") from error

    if dep_axis in ind_axes or len({dep_axis, *ind_axes}) != 3:
        raise ReconstructionInputError(
            "dependent_axis 与 independent_axes 必须覆盖 X/Y/Z"
        )
    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(beta).all()
    ):
        raise ReconstructionInputError("quadratic_graph 参数包含 NaN 或无穷值")
    if np.any(scale <= 0.0):
        raise ReconstructionInputError("independent_scale_mm 必须为正数")

    rp = rays[:, ind_axes[0]]
    rq = rays[:, ind_axes[1]]
    rd = rays[:, dep_axis]
    ap = rp / scale[0]
    aq = rq / scale[1]
    bp = -center[0] / scale[0]
    bq = -center[1] / scale[1]
    b0, b1, b2, b3, b4, b5 = beta

    quad_rhs = b3 * ap * ap + b4 * ap * aq + b5 * aq * aq
    linear_rhs = (
        b1 * ap
        + b2 * aq
        + 2.0 * b3 * ap * bp
        + b4 * (ap * bq + aq * bp)
        + 2.0 * b5 * aq * bq
    )
    const_rhs = (
        b0
        + b1 * bp
        + b2 * bq
        + b3 * bp * bp
        + b4 * bp * bq
        + b5 * bq * bq
    )

    aa = -quad_rhs
    bb = rd - linear_rhs
    cc = np.full(len(rays), -const_rhs, dtype=np.float64)
    roots = _solve_quadratic_all(aa, bb, cc, params.quadratic_epsilon)
    lam = _choose_roots(roots, rays, params, _model_z_range(model))
    return lam, np.isfinite(lam)


def _intersect_circular_cone(
    rays: np.ndarray,
    model: Mapping[str, Any],
    params: ReconstructionParams,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        axis = np.asarray(model["axis_unit_camera"], dtype=np.float64).reshape(3)
        apex = np.asarray(model["apex_camera_mm"], dtype=np.float64).reshape(3)
        alpha_deg = float(model["half_apex_angle_deg"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionInputError("circular_cone 模型参数不完整") from error

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= np.finfo(np.float64).eps:
        raise ReconstructionInputError("axis_unit_camera 不能为零向量")
    axis = axis / axis_norm
    if not np.isfinite(apex).all() or not np.isfinite(alpha_deg):
        raise ReconstructionInputError("circular_cone 参数包含 NaN 或无穷值")
    if not 0.0 < alpha_deg < 90.0:
        raise ReconstructionInputError("half_apex_angle_deg 必须位于 (0, 90) 度")

    cos2 = float(np.cos(np.deg2rad(alpha_deg)) ** 2)
    ray_axis = rays @ axis
    apex_axis = float(apex @ axis)
    # ((lambda*r-C)·a)^2 - cos(alpha)^2 ||lambda*r-C||^2 = 0
    aa = ray_axis * ray_axis - cos2 * np.sum(rays * rays, axis=1)
    bb = -2.0 * ray_axis * apex_axis + 2.0 * cos2 * (rays @ apex)
    cc_value = apex_axis * apex_axis - cos2 * float(apex @ apex)
    cc = np.full(len(rays), cc_value, dtype=np.float64)

    roots = _solve_quadratic_all(aa, bb, cc, params.quadratic_epsilon)
    lam = _choose_roots(
        roots,
        rays,
        params,
        _model_z_range(model),
        apex=apex,
        axis=axis,
    )
    return lam, np.isfinite(lam)


def _legacy_plane_model(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    """把旧 plane_abcd 转成新的 global_plane 模型映射。"""
    try:
        plane = np.asarray(calibration["plane_abcd"], dtype=np.float64).reshape(4)
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionInputError(
            "calibration 需要 laser_model，或旧格式 plane_abcd"
        ) from error
    norm = float(np.linalg.norm(plane[:3]))
    if norm <= np.finfo(np.float64).eps:
        raise ReconstructionInputError("激光平面法向量长度不能为零")
    plane = plane / norm
    return {
        "model_type": "global_plane",
        "normal": plane[:3],
        "d_mm": float(plane[3]),
    }


def _intersect_laser_surface(
    rays: np.ndarray,
    calibration: Mapping[str, Any],
    params: ReconstructionParams,
) -> tuple[np.ndarray, np.ndarray, str]:
    raw_model = calibration.get("laser_model")
    if raw_model is None:
        model = _legacy_plane_model(calibration)
    elif not isinstance(raw_model, Mapping):
        raise ReconstructionInputError("calibration['laser_model'] 必须是 Mapping")
    else:
        model = raw_model

    model_type = str(model.get("model_type", "global_plane")).lower()
    if model_type == "global_plane":
        lam, stable = _intersect_global_plane(rays, model, params)
    elif model_type == "quadratic_graph":
        lam, stable = _intersect_quadratic_graph(rays, model, params)
    elif model_type == "circular_cone":
        lam, stable = _intersect_circular_cone(rays, model, params)
    else:
        raise ReconstructionInputError(f"不支持的激光表面模型：{model_type}")
    return lam, stable, model_type


def _points_inside_polygon(
    points_uv: np.ndarray,
    polygon: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """返回像素点是否在简单多边形内；边界点也算 ROI 内。"""
    points = np.asarray(points_uv, dtype=np.float64)
    vertices = np.asarray(polygon, dtype=np.float64)
    x = points[:, 0, None]
    y = points[:, 1, None]
    x0 = vertices[:, 0][None, :]
    y0 = vertices[:, 1][None, :]
    x1 = np.roll(vertices[:, 0], -1)[None, :]
    y1 = np.roll(vertices[:, 1], -1)[None, :]

    # Ray crossing；水平边不产生 crossing，除以零只在屏蔽位置发生。
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    inside = np.count_nonzero(crosses & (x < x_at_y), axis=1) % 2 == 1

    # 对边界做显式包含，避免棋盘格边缘的亚像素中心被误删。
    cross_product = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
    scale = np.maximum(
        1.0,
        np.maximum(
            np.maximum(np.abs(x0), np.abs(y0)),
            np.maximum(np.abs(x1), np.abs(y1)),
        ),
    )
    on_line = np.abs(cross_product) <= 1.0e-9 * scale
    on_segment = (
        on_line
        & (x >= np.minimum(x0, x1) - 1.0e-9)
        & (x <= np.maximum(x0, x1) + 1.0e-9)
        & (y >= np.minimum(y0, y1) - 1.0e-9)
        & (y <= np.maximum(y0, y1) + 1.0e-9)
    )
    return inside | np.any(on_segment, axis=1)


def frozen_c0_q_coordinates(
    points_camera_c0: np.ndarray,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return Frozen-C0 ``(q1, q2)`` from ``P_c0`` camera coordinates.

    This deliberately uses the C0 intersection points and the quadratic
    model's independent-axis normalization.  It never reads C1-corrected,
    ground-transformed, or height-corrected coordinates.
    """
    points = np.asarray(points_camera_c0, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ReconstructionInputError(
            "points_camera_c0 必须是形状为 (N, 3) 的数组"
        )
    model = calibration.get("laser_model")
    if not isinstance(model, Mapping) or model.get("model_type") != "quadratic_graph":
        raise ReconstructionInputError(
            "Frozen-C0 q1/q2 只适用于 quadratic_graph 激光模型"
        )
    try:
        independent_axes = tuple(
            _axis_index(str(axis)) for axis in model["independent_axes"]
        )
        normalization = model["normalization"]
        center = np.asarray(
            normalization["independent_center_mm"], dtype=np.float64
        ).reshape(2)
        scale = np.asarray(
            normalization["independent_scale_mm"], dtype=np.float64
        ).reshape(2)
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionInputError(
            "quadratic_graph 模型缺少 Frozen-C0 q1/q2 normalization"
        ) from error
    if len(independent_axes) != 2 or not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ReconstructionInputError("Frozen-C0 q1/q2 normalization 非法")
    if np.any(scale <= 0.0):
        raise ReconstructionInputError("Frozen-C0 independent_scale_mm 必须为正数")
    normalized = (
        points[:, independent_axes] - center[None, :]
    ) / scale[None, :]
    return (
        np.ascontiguousarray(normalized[:, 0]),
        np.ascontiguousarray(normalized[:, 1]),
    )


def reconstruct_uv_to_ground(
    pixels_uv: np.ndarray,
    calibration: Mapping[str, Any],
    params: ReconstructionParams | None = None,
) -> ReconstructionResult:
    """把 ``(N, 2)`` 亚像素 ``(u, v)`` 重建为地面系三维点。

    ``calibration`` 使用 ``calibration.config_loader`` 返回的字典，
    至少包含 ``K``、``D``、``laser_model``、``R``、``t``；也兼容旧
    ``plane_abcd``。无效点（近平行、无交点、负深度、超出工作距离、
    非有限值）被剔除并计数。
    """
    if params is None:
        params = ReconstructionParams()

    points = np.asarray(pixels_uv, dtype=np.float64)
    empty_filtered = {
        "near_parallel": 0,
        "negative_depth": 0,
        "outside_working_distance": 0,
        "non_finite": 0,
        "no_valid_intersection": 0,
        "outside_image_roi": 0,
    }
    if points.size == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return ReconstructionResult(
            pixels_uv=empty,
            points_camera=np.empty((0, 3), dtype=np.float64),
            points_ground=np.empty((0, 3), dtype=np.float64),
            filtered=empty_filtered,
        )
    if points.ndim != 2 or points.shape[1] != 2:
        raise ReconstructionInputError("pixels_uv 必须是形状为 (N, 2) 的数组")
    if not np.isfinite(points).all():
        raise ReconstructionInputError("pixels_uv 包含 NaN 或无穷值")

    if params.image_roi_polygon is not None:
        inside_roi = _points_inside_polygon(points, params.image_roi_polygon)
        empty_filtered["outside_image_roi"] = int(np.count_nonzero(~inside_roi))
        points = points[inside_roi]
        if points.size == 0:
            return ReconstructionResult(
                pixels_uv=np.empty((0, 2), dtype=np.float64),
                points_camera=np.empty((0, 3), dtype=np.float64),
                points_ground=np.empty((0, 3), dtype=np.float64),
                filtered=empty_filtered,
            )

    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)
    transform = build_ground_transform(calibration["R"], calibration["t"])

    normalized = cv2.undistortPoints(
        points.reshape(-1, 1, 2), K, D
    ).reshape(-1, 2)
    rays = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    lambda_c0, stable, model_type = _intersect_laser_surface(
        rays, calibration, params
    )
    points_camera_c0_all = rays * lambda_c0[:, None]
    lambda_final = lambda_c0
    c1_clamped: np.ndarray | None = None
    if params.enable_laser_ray_correction:
        if model_type != "quadratic_graph":
            raise ReconstructionInputError(
                "laser_ray_correction 只能用于 quadratic_graph 基础模型"
            )
        correction = calibration.get("laser_ray_correction")
        if not isinstance(correction, FrozenLaserRayCorrection):
            raise ReconstructionInputError(
                "已开启 laser_ray_correction，但 calibration 缺少有效 frozen C1 参数"
            )
        try:
            evaluation = evaluate_frozen_laser_ray_correction(rays, correction)
        except LaserRayCorrectionError as error:
            raise ReconstructionInputError(
                f"laser_ray_correction 参数或运行时输入非法: {error}"
            ) from error
        lambda_final = np.ascontiguousarray(
            lambda_c0 + evaluation.correction_mm
        )
        c1_clamped = evaluation.clamped
    points_camera = rays * lambda_final[:, None]

    finite = np.isfinite(points_camera).all(axis=1) & np.isfinite(lambda_final)
    positive = lambda_final > 0.0
    within_distance = (
        (points_camera[:, 2] >= params.min_camera_depth_mm)
        & (points_camera[:, 2] <= params.max_camera_depth_mm)
    )
    valid = stable & finite & positive & within_distance
    no_intersection = ~np.isfinite(lambda_final)
    filtered = {
        "near_parallel": (
            int(np.count_nonzero(~stable)) if model_type == "global_plane" else 0
        ),
        "negative_depth": int(np.count_nonzero(finite & ~positive)),
        "outside_working_distance": int(
            np.count_nonzero(finite & positive & ~within_distance)
        ),
        "non_finite": int(np.count_nonzero(~finite & ~no_intersection)),
        "no_valid_intersection": int(np.count_nonzero(no_intersection)),
        "outside_image_roi": empty_filtered["outside_image_roi"],
    }

    points_camera = points_camera[valid]
    points_camera_c0 = points_camera_c0_all[valid]
    if c1_clamped is not None:
        c1_clamped = c1_clamped[valid]
    valid_pixels = points[valid]
    homogeneous = np.column_stack(
        [points_camera, np.ones(len(points_camera), dtype=np.float64)]
    )
    # 方向严格为 ground <- camera；不得改用逆矩阵，也不对 Zg 取绝对值。
    points_ground = (transform @ homogeneous.T).T[:, :3]
    points_ground = apply_ground_u_compensation(
        points_ground,
        valid_pixels,
        calibration.get("ground_u_compensation"),
    )

    final_finite = np.isfinite(points_ground).all(axis=1)
    filtered["non_finite"] += int(np.count_nonzero(~final_finite))
    points_camera_c0 = points_camera_c0[final_finite]
    if c1_clamped is not None:
        c1_clamped = c1_clamped[final_finite]
    q1_c0: np.ndarray | None = None
    q2_c0: np.ndarray | None = None
    if model_type == "quadratic_graph" and len(points_camera_c0):
        q1_c0, q2_c0 = frozen_c0_q_coordinates(
            points_camera_c0,
            calibration,
        )
    return ReconstructionResult(
        pixels_uv=np.ascontiguousarray(valid_pixels[final_finite]),
        points_camera=np.ascontiguousarray(points_camera[final_finite]),
        points_ground=np.ascontiguousarray(points_ground[final_finite]),
        filtered=filtered,
        points_camera_c0=np.ascontiguousarray(points_camera_c0),
        q1_c0=(None if q1_c0 is None else np.ascontiguousarray(q1_c0)),
        q2_c0=(None if q2_c0 is None else np.ascontiguousarray(q2_c0)),
        c1_clamped=(
            None if c1_clamped is None else np.ascontiguousarray(c1_clamped)
        ),
    )


def project_ground_points_to_pixels(
    points_ground: np.ndarray,
    calibration: Mapping[str, Any],
) -> np.ndarray:
    """把地面系三维点投影回图像像素坐标（用于叠加显示）。"""
    points = np.asarray(points_ground, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    R = np.asarray(calibration["R"], dtype=np.float64)
    t = np.asarray(calibration["t"], dtype=np.float64).reshape(3)
    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)

    # ground -> camera：p_c = R^T (p_g - t)
    points_camera = (points - t) @ R
    pixels, _ = cv2.projectPoints(
        points_camera.reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        D,
    )
    return np.ascontiguousarray(pixels.reshape(-1, 2))
