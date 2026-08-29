"""Frozen Daheng C1 ray-depth correction.

This module is deliberately a runtime evaluator only.  It consumes the
machine-readable frozen JSON parameters and never recomputes the Full-36 PCA
or refits the spline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import BSpline


class LaserRayCorrectionError(ValueError):
    """Frozen C1 parameters or runtime rays are invalid."""


@dataclass(frozen=True, slots=True)
class FrozenLaserRayCorrection:
    """Validated, immutable-in-contract view of a frozen C1 JSON."""

    model_id: str
    center_xn: float
    center_yn: float
    axis_s_xn: float
    axis_s_yn: float
    domain_min: float
    domain_max: float
    degree: int
    knots: np.ndarray
    coefficients_mm: np.ndarray
    source_path: str


@dataclass(frozen=True, slots=True)
class LaserRayCorrectionEvaluation:
    """Intermediate values retained for exact math and clamp tests."""

    s_raw: np.ndarray
    s_eval: np.ndarray
    correction_mm: np.ndarray
    clamped: np.ndarray


def load_frozen_laser_ray_correction(
    path: str | Path,
) -> FrozenLaserRayCorrection:
    """Read and strictly validate a frozen C1 JSON parameter file."""
    file_path = Path(path)
    try:
        document = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LaserRayCorrectionError(
            f"无法读取 frozen laser ray correction {file_path}: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise LaserRayCorrectionError("frozen C1 根节点必须是映射")

    if document.get("schema_version") != 1:
        raise LaserRayCorrectionError("frozen C1 schema_version 必须为 1")
    if document.get("model_id") != "C1_4k" or document.get("operational_model") != "C1_4k":
        raise LaserRayCorrectionError("frozen C1 必须是 C1_4k 模型")
    if document.get("frozen") is not True:
        raise LaserRayCorrectionError("frozen C1 必须标记 frozen=true")

    pca = _mapping(document, "pca_s")
    spline = _mapping(document, "spline")
    protocol = _mapping(document, "protocol")
    fit = _mapping(document, "fit")

    required_pca = ("center_xn", "center_yn", "axis_s_xn", "axis_s_yn")
    pca_values = {
        key: _finite_scalar(pca.get(key), f"pca_s.{key}")
        for key in required_pca
    }
    axis_norm = float(
        np.hypot(pca_values["axis_s_xn"], pca_values["axis_s_yn"])
    )
    if not np.isclose(axis_norm, 1.0, rtol=0.0, atol=1.0e-12):
        raise LaserRayCorrectionError("pca_s.axis_s 必须是单位向量")

    domain_min = _finite_scalar(pca.get("domain_min"), "pca_s.domain_min")
    domain_max = _finite_scalar(pca.get("domain_max"), "pca_s.domain_max")
    if not domain_min < domain_max:
        raise LaserRayCorrectionError("pca_s domain 必须严格递增")

    degree = _integer(spline.get("degree"), "spline.degree")
    interior_knot_count = _integer(
        spline.get("interior_knot_count"), "spline.interior_knot_count"
    )
    basis_count = _integer(spline.get("basis_count"), "spline.basis_count")
    if degree != 3 or interior_knot_count != 4 or basis_count != 8:
        raise LaserRayCorrectionError(
            "frozen C1 必须是 Full-36 C1_4k cubic B-spline"
        )

    knots = _finite_vector(spline.get("knots"), "spline.knots")
    coefficients = _finite_vector(
        spline.get("coefficients_mm"), "spline.coefficients_mm"
    )
    expected_knot_count = basis_count + degree + 1
    if len(knots) != expected_knot_count or len(coefficients) != basis_count:
        raise LaserRayCorrectionError(
            "frozen C1 knots/coefficients 长度与 cubic B-spline 不一致"
        )
    if np.any(np.diff(knots) < 0.0):
        raise LaserRayCorrectionError("frozen C1 knots 必须非递减")
    if not np.all(knots[: degree + 1] == domain_min) or not np.all(
        knots[-degree - 1 :] == domain_max
    ):
        raise LaserRayCorrectionError(
            "frozen C1 knots 端点必须与 PCA-s domain 完全一致"
        )
    interior = knots[degree + 1 : -degree - 1]
    if len(interior) != interior_knot_count or np.any(
        np.diff(interior) <= 0.0
    ) or np.any((interior <= domain_min) | (interior >= domain_max)):
        raise LaserRayCorrectionError("frozen C1 interior knots 非法")

    if protocol.get("spline_basis") != "cubic_B_spline":
        raise LaserRayCorrectionError("frozen C1 spline_basis 不是 cubic_B_spline")
    if protocol.get("spline_degree") != 3:
        raise LaserRayCorrectionError("frozen C1 protocol spline_degree 必须为 3")
    if protocol.get("basis_count") != basis_count:
        raise LaserRayCorrectionError("frozen C1 protocol basis_count 不一致")
    if protocol.get("interior_knot_count") != interior_knot_count:
        raise LaserRayCorrectionError(
            "frozen C1 protocol interior_knot_count 不一致"
        )
    if protocol.get("extrapolation") != "clip_to_pca_s_domain":
        raise LaserRayCorrectionError(
            "frozen C1 extrapolation policy 必须为 clip_to_pca_s_domain"
        )
    if protocol.get("lambda_formula") != "lambda_final = lambda_quadratic + F(pca_s)":
        raise LaserRayCorrectionError("frozen C1 lambda_formula 不一致")
    if protocol.get("pca_definition") != (
        "Full-36 PCA from xn/yn; PCA includes frame027 and is not recomputed on Operational-35"
    ):
        raise LaserRayCorrectionError("frozen C1 PCA definition 不一致")

    if fit.get("training_excludes_027") is not True:
        raise LaserRayCorrectionError("frozen C1 fit 必须排除 027 训练")
    training_pose_ids = fit.get("training_pose_ids")
    if not isinstance(training_pose_ids, list) or len(training_pose_ids) != 35:
        raise LaserRayCorrectionError("frozen C1 fit 必须包含 35 个训练姿态")

    knots.setflags(write=False)
    coefficients.setflags(write=False)
    return FrozenLaserRayCorrection(
        model_id="C1_4k",
        center_xn=pca_values["center_xn"],
        center_yn=pca_values["center_yn"],
        axis_s_xn=pca_values["axis_s_xn"],
        axis_s_yn=pca_values["axis_s_yn"],
        domain_min=domain_min,
        domain_max=domain_max,
        degree=degree,
        knots=knots,
        coefficients_mm=coefficients,
        source_path=str(file_path.resolve()),
    )


def evaluate_frozen_laser_ray_correction(
    rays: np.ndarray,
    correction: FrozenLaserRayCorrection,
) -> LaserRayCorrectionEvaluation:
    """Evaluate ``F(s)`` from normalized camera rays without extrapolation."""
    values = np.asarray(rays, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise LaserRayCorrectionError("rays 必须是形状为 (N, 3) 的数组")
    if not np.isfinite(values).all():
        raise LaserRayCorrectionError("rays 包含 NaN 或无穷值")
    if len(values) == 0:
        empty = np.empty(0, dtype=np.float64)
        return LaserRayCorrectionEvaluation(empty, empty, empty, np.empty(0, dtype=bool))

    ray_z = values[:, 2]
    if np.any(np.abs(ray_z) <= 1.0e-15):
        raise LaserRayCorrectionError("rays 的 ray_z 不能为零")
    normalized_xy = values[:, :2] / ray_z[:, None]
    center = np.array([correction.center_xn, correction.center_yn], dtype=np.float64)
    axis_s = np.array([correction.axis_s_xn, correction.axis_s_yn], dtype=np.float64)
    s_raw = (normalized_xy - center) @ axis_s
    s_eval = np.clip(s_raw, correction.domain_min, correction.domain_max)
    clamped = (s_raw < correction.domain_min) | (s_raw > correction.domain_max)

    # The input is clipped before this call, and extrapolate=False makes any
    # future accidental domain violation fail instead of silently extrapolating.
    spline = BSpline(
        correction.knots,
        correction.coefficients_mm,
        correction.degree,
        extrapolate=False,
    )
    correction_mm = np.asarray(spline(s_eval), dtype=np.float64)
    if not np.isfinite(correction_mm).all():
        raise LaserRayCorrectionError("frozen C1 spline 输出包含 NaN 或无穷值")
    return LaserRayCorrectionEvaluation(
        s_raw=np.ascontiguousarray(s_raw),
        s_eval=np.ascontiguousarray(s_eval),
        correction_mm=np.ascontiguousarray(correction_mm),
        clamped=np.ascontiguousarray(clamped),
    )


def apply_frozen_laser_ray_correction(
    lambda_c0: np.ndarray,
    rays: np.ndarray,
    correction: FrozenLaserRayCorrection,
) -> np.ndarray:
    """Return ``lambda_final = lambda_c0 + F(s)`` element by element."""
    values = np.asarray(lambda_c0, dtype=np.float64)
    if values.ndim != 1:
        raise LaserRayCorrectionError("lambda_c0 必须是一维数组")
    if values.shape[0] != len(rays):
        raise LaserRayCorrectionError("lambda_c0 必须与 rays 逐行对齐")
    evaluation = evaluate_frozen_laser_ray_correction(rays, correction)
    return np.ascontiguousarray(values + evaluation.correction_mm)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping) and name in value:
        value = value[name]
    if not isinstance(value, Mapping):
        raise LaserRayCorrectionError(f"frozen C1 缺少映射 {name}")
    return value


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LaserRayCorrectionError(f"{name} 必须是有限数值")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise LaserRayCorrectionError(f"{name} 必须是有限数值") from error
    if not np.isfinite(result):
        raise LaserRayCorrectionError(f"{name} 必须是有限数值")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LaserRayCorrectionError(f"{name} 必须是整数")
    return value


def _finite_vector(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise LaserRayCorrectionError(f"{name} 必须是数值数组") from error
    if not len(result) or not np.isfinite(result).all():
        raise LaserRayCorrectionError(f"{name} 必须是非空有限数值数组")
    return np.ascontiguousarray(result)
