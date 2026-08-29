"""Reusable robust linear ground reference fitting.

The fitting kernel in this module is the original baseline-ROI kernel from
``height_measure``.  It models the ground only as ``Zg = a*S + b``; no
additional surface model or calibration parameter is introduced here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


GROUND_SUPPORT_PNP_BOARD_MASK = "pnp_board_mask"
GROUND_SUPPORT_MANUAL_ROI = "manual_ground_roi"
GROUND_SUPPORT_FROZEN_SESSION = "ground5c_frozen_session_linear"
SUPPORTED_GROUND_SUPPORT_SOURCES = frozenset(
    {
        GROUND_SUPPORT_PNP_BOARD_MASK,
        GROUND_SUPPORT_MANUAL_ROI,
        GROUND_SUPPORT_FROZEN_SESSION,
    }
)
SUPPORTED_GROUND_FIT_SUPPORT_SOURCES = frozenset(
    {GROUND_SUPPORT_PNP_BOARD_MASK, GROUND_SUPPORT_MANUAL_ROI}
)

FROZEN_SESSION_COORDINATE = "physical_S"
FROZEN_SESSION_FORMULA = "S=(XY-origin_xy) dot direction_xy"
FROZEN_SESSION_FIT_POSE_IDS = ("001", "002", "003", "004", "005")


class MeasurementError(RuntimeError):
    """Raised when input points are insufficient or geometrically degenerate."""


@dataclass(frozen=True, slots=True)
class LineFitXY:
    """Orthogonal least-squares line fit in the ground XY plane."""

    centre_xy: np.ndarray
    direction_xy: np.ndarray
    endpoints_xy: np.ndarray
    rmse_mm: float
    inlier_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class GroundProfileFit:
    """Linear ground height model along a ground-line direction."""

    origin_xy: np.ndarray
    direction_xy: np.ndarray
    slope_z_per_mm: float
    intercept_z_mm: float
    rmse_mm: float
    inlier_mask: np.ndarray

    def project_s(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64)
        return (points - self.origin_xy) @ self.direction_xy

    def predict_z(self, points_xy: np.ndarray) -> np.ndarray:
        s = self.project_s(points_xy)
        return self.slope_z_per_mm * s + self.intercept_z_mm


@dataclass(frozen=True, slots=True)
class SessionGroundReference:
    """Frozen runtime ``Zg = a*S+b`` reference for one online session.

    ``valid_s_range_mm`` is the S span of the fitted inliers.  Runtime point
    application never extrapolates outside that span: out-of-range points are
    returned unchanged and reported by the boolean mask from
    :meth:`apply_to_points`.
    """

    origin_xy: np.ndarray
    direction_xy: np.ndarray
    slope_z_per_mm: float
    intercept_z_mm: float
    rmse_mm: float
    valid_s_range_mm: tuple[float, float]
    status: str = "VALID"
    source: str = "session_laser_ground"
    inlier_mask: np.ndarray | None = None
    point_count: int = 0
    inlier_count: int = 0
    # ``source`` remains the fitter/runtime identifier for compatibility.
    # ``support_source`` is mandatory for runtime activation and records how
    # the points were explicitly known to belong to a real ground plane.
    support_source: str | None = None
    active_ground_extrinsic_source: str | None = None
    ground_extrinsic_generation: int | None = None
    frame_host_monotonic_ns: int | None = None
    mask_inset_mm: float | None = None
    support_metadata: dict[str, Any] = field(default_factory=dict)
    coordinate: str | None = None
    coordinate_units: str | None = None
    coordinate_formula: str | None = None
    frozen_json_path: str | None = None
    frozen_json_sha256: str | None = None
    frozen_schema_version: int | None = None
    fit_pose_ids: tuple[str, ...] = ()

    @property
    def slope(self) -> float:
        return self.slope_z_per_mm

    @property
    def intercept(self) -> float:
        return self.intercept_z_mm

    @property
    def rmse(self) -> float:
        return self.rmse_mm

    @property
    def valid_s_range(self) -> tuple[float, float]:
        return self.valid_s_range_mm

    @property
    def provenance_source(self) -> str:
        """Return the explicit support source used for this reference."""
        return self.support_source or self.source

    def project_s(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise MeasurementError("ground reference XY points must have shape (N, 2)")
        if not np.isfinite(points).all():
            raise MeasurementError("ground reference XY points contain NaN or infinite values")
        return (points - self.origin_xy) @ self.direction_xy

    def predict_z(self, points_xy: np.ndarray) -> np.ndarray:
        s = self.project_s(points_xy)
        return self.slope_z_per_mm * s + self.intercept_z_mm

    def valid_s_mask(self, points_xy: np.ndarray) -> np.ndarray:
        s = self.project_s(points_xy)
        lower, upper = self.valid_s_range_mm
        return np.isfinite(s) & (s >= lower) & (s <= upper)

    def apply_to_points(
        self, points_ground: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Level valid points and return ``(points, valid_mask)``.

        The returned array is always a copy.  Points outside the fitted S
        domain remain raw so callers cannot silently extrapolate the session
        reference.
        """
        points = np.asarray(points_ground, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise MeasurementError("ground points must have shape (N, 3)")
        if not np.isfinite(points).all():
            raise MeasurementError("ground points contain NaN or infinite values")
        corrected = np.ascontiguousarray(points.copy())
        if not len(points):
            return corrected, np.empty(0, dtype=bool)
        valid = self.valid_s_mask(points[:, :2])
        if np.any(valid):
            corrected[valid, 2] -= self.predict_z(points[valid, :2])
        return corrected, valid

    # Descriptive aliases keep the runtime object convenient for callers that
    # think in terms of levelling a point cloud rather than applying a model.
    level_points = apply_to_points
    apply = apply_to_points

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe runtime snapshot."""
        return {
            "status": self.status,
            "source": self.provenance_source,
            "fit_source": self.source,
            "support_source": self.support_source,
            "active_ground_extrinsic_source": self.active_ground_extrinsic_source,
            "ground_extrinsic_generation": self.ground_extrinsic_generation,
            "frame_host_monotonic_ns": self.frame_host_monotonic_ns,
            "mask_inset_mm": self.mask_inset_mm,
            "support": dict(self.support_metadata),
            "origin_xy": np.asarray(self.origin_xy, dtype=np.float64).tolist(),
            "direction_xy": np.asarray(self.direction_xy, dtype=np.float64).tolist(),
            "slope": float(self.slope_z_per_mm),
            "intercept": float(self.intercept_z_mm),
            "slope_z_per_mm": float(self.slope_z_per_mm),
            "intercept_z_mm": float(self.intercept_z_mm),
            "rmse_mm": float(self.rmse_mm),
            "valid_s_range_mm": [
                float(self.valid_s_range_mm[0]),
                float(self.valid_s_range_mm[1]),
            ],
            "point_count": int(self.point_count),
            "inlier_count": int(self.inlier_count),
            "coordinate": self.coordinate,
            "coordinate_units": self.coordinate_units,
            "coordinate_formula": self.coordinate_formula,
            "frozen_json_path": self.frozen_json_path,
            "frozen_json_sha256": self.frozen_json_sha256,
            "frozen_schema_version": self.frozen_schema_version,
            "fit_pose_ids": list(self.fit_pose_ids),
        }


def load_frozen_session_ground_reference(
    path: str | Path,
    *,
    active_ground_extrinsic_source: str,
    ground_extrinsic_generation: int,
) -> SessionGroundReference:
    """Load the validated Ground-5C A-2 physical-S fit without refitting.

    The loader intentionally accepts only the A-2 fit-only schema and binds the
    loaded model to the *currently active* Session PnP generation.  The JSON
    file is hashed from its exact bytes; no value is silently normalized or
    recomputed apart from validating the stored support metadata.
    """
    frozen_path = Path(path).expanduser()
    raw = frozen_path.read_bytes()
    json_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Frozen Session Ground JSON 无法解析: {frozen_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Frozen Session Ground JSON 顶层必须是 object")

    _require_equal(payload.get("schema_version"), 1, "schema_version")
    _require_equal(payload.get("status"), "frozen_fit_only", "status")
    _require_equal(payload.get("coordinate"), FROZEN_SESSION_COORDINATE, "coordinate")
    _require_equal(payload.get("coordinate_units"), "mm", "coordinate_units")

    active_source = str(active_ground_extrinsic_source).strip().lower()
    if active_source != "session":
        raise ValueError(
            "Ground-5C Frozen Session Ground 只能绑定有效的 Session PnP 外参"
        )
    if isinstance(ground_extrinsic_generation, bool):
        raise ValueError("ground_extrinsic_generation 必须是非负整数")
    try:
        generation_value = float(ground_extrinsic_generation)
    except (TypeError, ValueError) as error:
        raise ValueError("ground_extrinsic_generation 必须是非负整数") from error
    if (
        not np.isfinite(generation_value)
        or not generation_value.is_integer()
        or generation_value < 0.0
    ):
        raise ValueError("ground_extrinsic_generation 必须是非负整数")

    slope = _finite_json_float(payload.get("a_session"), "a_session")
    intercept = _finite_json_float(payload.get("b_session"), "b_session")
    valid_domain = _finite_json_pair(payload.get("valid_domain"), "valid_domain")
    bin_edges = _finite_json_array(payload.get("bin_edges"), "bin_edges")
    if len(bin_edges) < 2 or np.any(np.diff(bin_edges) <= 0.0):
        raise ValueError("Frozen Session Ground bin_edges 必须严格递增")

    formal_indices = _json_int_list(
        payload.get("formal_bin_indices"), "formal_bin_indices"
    )
    if not formal_indices:
        raise ValueError("Frozen Session Ground 必须包含 formal bin support")
    if formal_indices != list(
        range(formal_indices[0], formal_indices[-1] + 1)
    ):
        raise ValueError("Frozen Session Ground formal bin support 必须连续")
    if formal_indices[0] < 0 or formal_indices[-1] >= len(bin_edges) - 1:
        raise ValueError("Frozen Session Ground formal bin index 超出 bin_edges")
    expected_domain = (
        float(bin_edges[formal_indices[0]]),
        float(bin_edges[formal_indices[-1] + 1]),
    )
    if not np.allclose(valid_domain, expected_domain, rtol=0.0, atol=1.0e-7):
        raise ValueError("Frozen Session Ground valid_domain 与 formal support 不一致")

    formal_support = _require_mapping(payload.get("formal_support"), "formal_support")
    _require_equal(
        formal_support.get("coordinate"),
        FROZEN_SESSION_COORDINATE,
        "formal_support.coordinate",
    )
    _require_equal(
        formal_support.get("formal_bin_count"),
        len(formal_indices),
        "formal_support.formal_bin_count",
    )
    if str(formal_support.get("classification", "")).upper() != "PASS":
        raise ValueError("Frozen Session Ground formal support 不是 PASS")

    fit_pose_ids = tuple(
        str(value) for value in _json_string_list(payload.get("fit_pose_ids"), "fit_pose_ids")
    )
    if fit_pose_ids != FROZEN_SESSION_FIT_POSE_IDS:
        raise ValueError(
            "Frozen Session Ground 必须只来自 fit pose001–005，不能混入 held-out pose"
        )

    physical_s = _require_mapping(payload.get("physical_S"), "physical_S")
    _require_equal(
        physical_s.get("formula"), FROZEN_SESSION_FORMULA, "physical_S.formula"
    )
    _require_equal(
        physical_s.get("per_frame_redefinition"),
        False,
        "physical_S.per_frame_redefinition",
    )
    origin_xy = _finite_json_array(
        physical_s.get("origin_xy_mm"), "physical_S.origin_xy_mm"
    )
    direction_xy = _finite_json_array(
        physical_s.get("direction_xy"), "physical_S.direction_xy"
    )
    if origin_xy.shape != (2,) or direction_xy.shape != (2,):
        raise ValueError("Frozen Session Ground physical_S origin/direction 必须是 (2,)")
    direction_norm = float(np.linalg.norm(direction_xy))
    if not np.isclose(direction_norm, 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("Frozen Session Ground physical_S direction 必须是单位向量")

    candidates = _require_mapping(
        payload.get("coordinate_candidates"), "coordinate_candidates"
    )
    physical_candidate = _require_mapping(
        candidates.get(FROZEN_SESSION_COORDINATE),
        "coordinate_candidates.physical_S",
    )
    candidate_slope = _finite_json_float(
        physical_candidate.get("a_session"),
        "coordinate_candidates.physical_S.a_session",
    )
    candidate_intercept = _finite_json_float(
        physical_candidate.get("b_session"),
        "coordinate_candidates.physical_S.b_session",
    )
    if not np.isclose(slope, candidate_slope, rtol=0.0, atol=1.0e-12):
        raise ValueError("Frozen Session Ground a_session 与 physical_S candidate 不一致")
    if not np.isclose(intercept, candidate_intercept, rtol=0.0, atol=1.0e-12):
        raise ValueError("Frozen Session Ground b_session 与 physical_S candidate 不一致")
    candidate_domain = _finite_json_pair(
        physical_candidate.get("valid_domain"),
        "coordinate_candidates.physical_S.valid_domain",
    )
    if not np.allclose(valid_domain, candidate_domain, rtol=0.0, atol=1.0e-7):
        raise ValueError("Frozen Session Ground physical_S candidate domain 不一致")
    fit_metrics = _require_mapping(
        physical_candidate.get("fit_metrics"),
        "coordinate_candidates.physical_S.fit_metrics",
    )
    fit_rmse = _finite_json_float(
        fit_metrics.get("rmse_mm"),
        "coordinate_candidates.physical_S.fit_metrics.rmse_mm",
    )

    fit_protocol = _require_mapping(payload.get("fit_protocol"), "fit_protocol")
    for key in (
        "raw_point_pooling",
        "z_residual_filtering",
        "extrapolation",
        "clamp",
        "factory_profile_used",
        "ground3_numeric_parameters_used",
        "h1_used",
    ):
        _require_equal(fit_protocol.get(key), False, f"fit_protocol.{key}")

    provenance = _require_mapping(payload.get("provenance"), "provenance")
    for key in ("fit_dir", "config_path", "manifest_path", "ground5a_output"):
        if not str(provenance.get(key, "")).strip():
            raise ValueError(f"Frozen Session Ground provenance.{key} 缺失")
    physical_source_sha = str(physical_s.get("source_sha256", "")).strip().lower()
    if not _is_sha256(physical_source_sha):
        raise ValueError("physical_S.source_sha256 不是有效 SHA256")
    config_sha = str(provenance.get("config_sha256", "")).strip().lower()
    if not _is_sha256(config_sha):
        raise ValueError("provenance.config_sha256 不是有效 SHA256")
    cache = _require_mapping(provenance.get("cache"), "provenance.cache")
    cache_sha = str(cache.get("cache_sha256", "")).strip().lower()
    if not _is_sha256(cache_sha):
        raise ValueError("provenance.cache.cache_sha256 不是有效 SHA256")

    support_metadata = {
        "kind": GROUND_SUPPORT_FROZEN_SESSION,
        "coordinate": FROZEN_SESSION_COORDINATE,
        "coordinate_units": "mm",
        "formula": FROZEN_SESSION_FORMULA,
        "frozen_json_sha256": json_sha256,
        "frozen_schema_version": 1,
        "fit_pose_ids": list(fit_pose_ids),
        "formal_bin_indices": formal_indices,
        "strong_bin_indices": _json_int_list(
            payload.get("strong_bin_indices"), "strong_bin_indices"
        ),
        "weak_bin_indices": _json_int_list(
            payload.get("weak_bin_indices"), "weak_bin_indices"
        ),
        "bin_edges": bin_edges.tolist(),
        "valid_domain": list(valid_domain),
        "fit_protocol": dict(fit_protocol),
        "physical_s_source_path": str(physical_s.get("source_path", "")),
        "physical_s_source_sha256": physical_source_sha,
        "provenance": {
            "fit_dir": str(provenance["fit_dir"]),
            "config_path": str(provenance["config_path"]),
            "config_sha256": config_sha,
            "manifest_path": str(provenance["manifest_path"]),
            "ground5a_output": str(provenance["ground5a_output"]),
            "cache_path": str(cache.get("cache_path", "")),
            "cache_sha256": cache_sha,
        },
        "support_classification": str(formal_support.get("classification")),
        "formal_bin_count": len(formal_indices),
        "fit_rmse_mm": fit_rmse,
    }
    return SessionGroundReference(
        origin_xy=np.ascontiguousarray(origin_xy.copy()),
        direction_xy=np.ascontiguousarray(direction_xy.copy()),
        slope_z_per_mm=slope,
        intercept_z_mm=intercept,
        rmse_mm=fit_rmse,
        valid_s_range_mm=valid_domain,
        status="VALID",
        source=GROUND_SUPPORT_FROZEN_SESSION,
        point_count=len(formal_indices),
        inlier_count=len(formal_indices),
        support_source=GROUND_SUPPORT_FROZEN_SESSION,
        active_ground_extrinsic_source=active_source,
        ground_extrinsic_generation=int(generation_value),
        support_metadata=support_metadata,
        coordinate=FROZEN_SESSION_COORDINATE,
        coordinate_units="mm",
        coordinate_formula=FROZEN_SESSION_FORMULA,
        frozen_json_path=str(frozen_path.resolve()),
        frozen_json_sha256=json_sha256,
        frozen_schema_version=1,
        fit_pose_ids=fit_pose_ids,
    )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 object")
    return value


def _require_equal(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(f"{name} 不符合 Frozen Session Ground schema")


def _finite_json_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限数") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    return number


def _finite_json_pair(value: Any, name: str) -> tuple[float, float]:
    array = _finite_json_array(value, name)
    if array.shape != (2,) or float(array[0]) > float(array[1]):
        raise ValueError(f"{name} 必须是递增的两个有限数")
    return float(array[0]), float(array[1])


def _finite_json_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限数数组") from error
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是一维有限数数组")
    return array


def _json_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是整数数组")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} 必须是整数数组")
        if int(item) != item:
            raise ValueError(f"{name} 必须是整数数组")
        result.append(int(item))
    return result


def _json_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{name} 必须是非空字符串数组")
    return [str(item).strip() for item in value]


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_points(points: np.ndarray, name: str, minimum: int) -> np.ndarray:
    """Validate and normalize a 3D ground-point array."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise MeasurementError(f"{name} must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise MeasurementError(f"{name} contains NaN or infinite values")
    if len(array) < minimum:
        raise MeasurementError(
            f"{name} has too few points: {len(array)} < {minimum}"
        )
    return array


def robust_sigma(residuals: np.ndarray) -> float:
    """MAD-based robust standard deviation, with std fallback."""
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    sigma = 1.4826 * mad
    if sigma <= np.finfo(np.float64).eps:
        sigma = float(np.std(residuals))
    return sigma


def fit_line_xy(
    points_xy: np.ndarray, params: Any, name: str
) -> LineFitXY:
    """Fit a robust 2D line in XY and reject orthogonal outliers."""
    # This body intentionally mirrors the former height_measure kernel.
    mask = np.ones(len(points_xy), dtype=bool)
    centre = np.zeros(2)
    direction = np.array([1.0, 0.0])
    for _ in range(params.outlier_max_iterations):
        selected = points_xy[mask]
        if len(selected) < 2:
            raise MeasurementError(f"{name} has too few inliers")
        centre = np.mean(selected, axis=0)
        centred = selected - centre
        _, singular_values, right_vectors = np.linalg.svd(
            centred, full_matrices=False
        )
        if singular_values[0] <= np.finfo(np.float64).eps:
            raise MeasurementError(f"{name} points are degenerate")
        direction = right_vectors[0]
        if direction[0] < 0.0 or (direction[0] == 0.0 and direction[1] < 0.0):
            direction = -direction

        normal = np.array([-direction[1], direction[0]])
        signed_residuals = (points_xy - centre) @ normal
        sigma = robust_sigma(signed_residuals[mask])
        if sigma <= np.finfo(np.float64).eps:
            break
        new_mask = (
            np.abs(signed_residuals)
            <= params.outlier_sigma_multiplier * sigma
        )
        if new_mask.sum() < 2 or bool(np.all(new_mask == mask)):
            break
        mask = new_mask

    selected = points_xy[mask]
    projections = (selected - centre) @ direction
    endpoints = np.vstack(
        [
            centre + float(np.min(projections)) * direction,
            centre + float(np.max(projections)) * direction,
        ]
    )
    normal = np.array([-direction[1], direction[0]])
    orthogonal = np.abs((selected - centre) @ normal)
    return LineFitXY(
        centre_xy=np.ascontiguousarray(centre),
        direction_xy=np.ascontiguousarray(direction),
        endpoints_xy=np.ascontiguousarray(endpoints),
        rmse_mm=float(np.sqrt(np.mean(orthogonal**2))),
        inlier_mask=mask,
    )


def fit_ground_profile(
    baseline_points: np.ndarray,
    params: Any,
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> tuple[GroundProfileFit, float]:
    """Fit the existing robust linear ground profile ``Zg = a*S+b``."""
    # This body intentionally mirrors the former height_measure kernel.
    s = (baseline_points[:, :2] - origin_xy) @ direction_xy
    z = baseline_points[:, 2]
    mask = np.ones(len(z), dtype=bool)

    def fit_selected(
        selected_s: np.ndarray, selected_z: np.ndarray
    ) -> tuple[float, float]:
        if float(np.ptp(selected_s)) < 1.0:
            return 0.0, float(np.median(selected_z))
        design = np.column_stack([selected_s, np.ones_like(selected_s)])
        slope, intercept = np.linalg.lstsq(design, selected_z, rcond=None)[0]
        return float(slope), float(intercept)

    slope = 0.0
    intercept = float(np.median(z))
    sigma = robust_sigma(z - intercept)
    for _ in range(params.outlier_max_iterations):
        selected_s = s[mask]
        selected_z = z[mask]
        if len(selected_z) < 2:
            raise MeasurementError("baseline has too few ground-profile inliers")
        slope, intercept = fit_selected(selected_s, selected_z)
        residuals = z - (slope * s + intercept)
        sigma = robust_sigma(residuals[mask])
        if sigma <= np.finfo(np.float64).eps:
            break
        new_mask = (
            np.abs(residuals) <= params.outlier_sigma_multiplier * sigma
        )
        if new_mask.sum() < 2 or bool(np.all(new_mask == mask)):
            break
        mask = new_mask

    selected_residuals = z[mask] - (slope * s[mask] + intercept)
    profile = GroundProfileFit(
        origin_xy=np.ascontiguousarray(origin_xy),
        direction_xy=np.ascontiguousarray(direction_xy),
        slope_z_per_mm=slope,
        intercept_z_mm=intercept,
        rmse_mm=float(np.sqrt(np.mean(selected_residuals**2))),
        inlier_mask=mask,
    )
    return profile, sigma


def fit_session_ground_reference(
    points_ground: np.ndarray,
    params: Any | None = None,
    *,
    source: str = "session_laser_ground",
) -> SessionGroundReference:
    """Fit one frozen session ground reference from an empty-ground point cloud."""
    if params is None:
        # Import lazily to avoid a module cycle: height_measure re-exports the
        # same kernel and imports this module at import time.
        from .height_measure import MeasurementParams

        params = MeasurementParams()
    points = validate_points(points_ground, "session ground reference", params.min_baseline_points)
    line_fit = fit_line_xy(points[:, :2], params, "session ground reference")
    profile, _ = fit_ground_profile(
        points,
        params,
        line_fit.centre_xy,
        line_fit.direction_xy,
    )
    s = profile.project_s(points[:, :2])
    inlier_s = s[profile.inlier_mask]
    if len(inlier_s) < 2 or not np.isfinite(inlier_s).all():
        raise MeasurementError("session ground reference has too few valid S points")
    return SessionGroundReference(
        origin_xy=np.ascontiguousarray(profile.origin_xy.copy()),
        direction_xy=np.ascontiguousarray(profile.direction_xy.copy()),
        slope_z_per_mm=profile.slope_z_per_mm,
        intercept_z_mm=profile.intercept_z_mm,
        rmse_mm=profile.rmse_mm,
        valid_s_range_mm=(float(np.min(inlier_s)), float(np.max(inlier_s))),
        status="VALID",
        source=source,
        inlier_mask=np.ascontiguousarray(profile.inlier_mask.copy()),
        point_count=len(points),
        inlier_count=int(profile.inlier_mask.sum()),
    )


def fit_session_ground_reference_from_support(
    points_ground: np.ndarray,
    params: Any | None = None,
    *,
    support_source: str,
    active_ground_extrinsic_source: str,
    ground_extrinsic_generation: int,
    frame_host_monotonic_ns: int,
    mask_inset_mm: float | None = None,
    support_metadata: dict[str, Any] | None = None,
) -> SessionGroundReference:
    """Fit the existing kernel and bind explicit runtime provenance.

    The fitter itself intentionally keeps the historical
    ``fit_session_ground_reference(points_ground, params)`` signature.  This
    wrapper is the only supported path for creating a runtime reference: it
    records the ground-support source and binds the result to the active
    ground-extrinsic generation.
    """
    normalized_source = str(support_source).strip().lower()
    if normalized_source not in SUPPORTED_GROUND_FIT_SUPPORT_SOURCES:
        allowed = ", ".join(sorted(SUPPORTED_GROUND_FIT_SUPPORT_SOURCES))
        raise ValueError(f"ground support source 必须是: {allowed}")
    normalized_extrinsic_source = str(active_ground_extrinsic_source).strip().lower()
    if normalized_extrinsic_source not in {"reference", "session"}:
        raise ValueError("active ground extrinsic source 必须是 reference 或 session")
    if isinstance(ground_extrinsic_generation, bool) or int(
        ground_extrinsic_generation
    ) < 0:
        raise ValueError("ground_extrinsic_generation 必须是非负整数")
    if isinstance(frame_host_monotonic_ns, bool) or int(frame_host_monotonic_ns) < 0:
        raise ValueError("frame_host_monotonic_ns 必须是非负整数")
    if mask_inset_mm is not None and (
        not np.isfinite(float(mask_inset_mm)) or float(mask_inset_mm) < 0.0
    ):
        raise ValueError("mask_inset_mm 必须是有限非负数")
    reference = fit_session_ground_reference(points_ground, params)
    return replace(
        reference,
        support_source=normalized_source,
        active_ground_extrinsic_source=normalized_extrinsic_source,
        ground_extrinsic_generation=int(ground_extrinsic_generation),
        frame_host_monotonic_ns=int(frame_host_monotonic_ns),
        mask_inset_mm=(None if mask_inset_mm is None else float(mask_inset_mm)),
        support_metadata=dict(support_metadata or {}),
    )


__all__ = [
    "FROZEN_SESSION_COORDINATE",
    "FROZEN_SESSION_FIT_POSE_IDS",
    "FROZEN_SESSION_FORMULA",
    "GroundProfileFit",
    "GROUND_SUPPORT_FROZEN_SESSION",
    "GROUND_SUPPORT_MANUAL_ROI",
    "GROUND_SUPPORT_PNP_BOARD_MASK",
    "LineFitXY",
    "MeasurementError",
    "SessionGroundReference",
    "SUPPORTED_GROUND_SUPPORT_SOURCES",
    "SUPPORTED_GROUND_FIT_SUPPORT_SOURCES",
    "fit_ground_profile",
    "fit_line_xy",
    "fit_session_ground_reference",
    "fit_session_ground_reference_from_support",
    "load_frozen_session_ground_reference",
    "robust_sigma",
    "validate_points",
]
