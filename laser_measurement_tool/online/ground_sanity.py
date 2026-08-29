"""Diagnostic laser-on ground-plane check for a Session calibration.

This module deliberately only measures the points produced by the existing
online reconstruction pipeline.  It never subtracts the measured bias, fits
an ``a*S+b`` correction, or changes any calibration object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping

import numpy as np

from measurement.board_mask import (
    full_board_physical_polygon,
    select_board_ground_points,
)

# Compatibility name retained for existing callers and regression tests.  The
# implementation lives in the shared board-mask module so Ground Sanity and
# Session Ground Reference cannot drift apart.
select_points_inside_board_mask = select_board_ground_points


_DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "min_valid_points": 20,
    "max_abs_bias_mm": 2.0,
    "max_rmse_mm": 2.0,
    "max_p95_abs_mm": 3.0,
    "max_abs_mm": 5.0,
    "max_abs_slope_mm_per_mm": 0.02,
}


@dataclass(frozen=True, slots=True)
class GroundSanityResult:
    """JSON-safe summary of one laser-on ground sanity check."""

    status: str
    message: str
    ground_extrinsic_source: str
    frame_number: int | None
    session_calibration_frame_number: int | None
    input_point_count: int
    valid_point_count: int
    bias_zg_mm: float | None
    rmse_zg_mm: float | None
    p95_abs_zg_mm: float | None
    max_abs_zg_mm: float | None
    ground_slope_mm_per_mm: float | None
    warnings: tuple[str, ...] = ()
    threshold_violations: tuple[str, ...] = ()
    thresholds: dict[str, float | int] = field(default_factory=dict)
    evaluated_at_utc: str = ""
    mask: dict[str, Any] = field(default_factory=dict)
    frame_host_monotonic_ns: int | None = None
    session_calibration_host_monotonic_ns: int | None = None
    session_generation: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a stable record suitable for ``session_ground_calibration.json``."""
        return {
            "schema_version": 1,
            "saved_at_utc": self.evaluated_at_utc,
            "status": self.status,
            "valid": self.status == "VALID",
            "message": self.message,
            "ground_extrinsic_source": self.ground_extrinsic_source,
            "frame": {
                "camera_frame_number": self.frame_number,
                "session_calibration_frame_number": self.session_calibration_frame_number,
                "host_monotonic_ns": self.frame_host_monotonic_ns,
                "session_calibration_host_monotonic_ns": self.session_calibration_host_monotonic_ns,
                "session_generation": self.session_generation,
            },
            "formal_chain": [
                "Steger",
                "Frozen C0",
                "Frozen C1",
                "Session ground extrinsic",
            ],
            "metrics": {
                "input_point_count": self.input_point_count,
                "valid_point_count": self.valid_point_count,
                "bias_zg_mm": self.bias_zg_mm,
                "rmse_zg_mm": self.rmse_zg_mm,
                "p95_abs_zg_mm": self.p95_abs_zg_mm,
                "max_abs_zg_mm": self.max_abs_zg_mm,
                "ground_slope_mm_per_mm": self.ground_slope_mm_per_mm,
            },
            "thresholds": dict(self.thresholds),
            "mask": dict(self.mask),
            "warnings": list(self.warnings),
            "threshold_violations": list(self.threshold_violations),
            "correction_applied": False,
            "surface_correction_applied": False,
            "stage_a_applied": False,
        }


def evaluate_ground_sanity(
    points_ground: np.ndarray,
    *,
    ground_extrinsic_source: str,
    frame_number: int | None,
    session_calibration_frame_number: int | None,
    frame_host_monotonic_ns: int | None = None,
    session_calibration_host_monotonic_ns: int | None = None,
    session_generation: int | None = None,
    thresholds: Any | None = None,
    mask: Mapping[str, Any] | None = None,
) -> GroundSanityResult:
    """Evaluate raw ground ``Zg`` values without applying a correction.

    ``ground_slope_mm_per_mm`` is the diagnostic linear trend of ``Zg`` along
    the dominant XY direction of the laser stripe.  The intercept and fitted
    values are intentionally not returned or applied.
    """
    points = np.asarray(points_ground, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ground 必须是形状为 (N, 3) 的数组")

    threshold_values = _normalise_thresholds(thresholds)
    if mask is None:
        mask_metadata: dict[str, Any] = {}
    elif isinstance(mask, Mapping):
        mask_metadata = dict(mask)
    else:
        raise ValueError("mask 必须是映射")
    input_count = int(len(points))
    warnings: list[str] = []
    violations: list[str] = []

    source = str(ground_extrinsic_source).strip().lower()
    if source != "session":
        warnings.append("ground_extrinsic_source_not_session")
    if session_calibration_host_monotonic_ns is not None:
        if (
            frame_host_monotonic_ns is None
            or frame_host_monotonic_ns <= session_calibration_host_monotonic_ns
        ):
            warnings.append("laser_on_frame_not_after_session_calibration")
    elif session_calibration_frame_number is None:
        warnings.append("session_calibration_frame_number_missing")
    elif frame_number is None or frame_number <= session_calibration_frame_number:
        # Compatibility path for old callers/records.  The online GUI always
        # supplies host monotonic timestamps, which are immune to SDK counter
        # resets after stop/start.
        warnings.append("laser_on_frame_not_after_session_calibration")
    if mask_metadata.get("enabled") and mask_metadata.get("status") != "applied":
        warnings.append("board_mask_unavailable")

    finite_mask = np.isfinite(points).all(axis=1)
    if not bool(np.all(finite_mask)):
        warnings.append("nonfinite_ground_points")
    valid_points = points[finite_mask]
    valid_count = int(len(valid_points))
    if valid_count < int(threshold_values["min_valid_points"]):
        violations.append("valid_point_count_below_minimum")

    bias: float | None = None
    rmse: float | None = None
    p95_abs: float | None = None
    max_abs: float | None = None
    slope: float | None = None
    if valid_count:
        zg = valid_points[:, 2]
        absolute_zg = np.abs(zg)
        bias = float(np.mean(zg))
        rmse = float(np.sqrt(np.mean(np.square(zg))))
        p95_abs = float(np.percentile(absolute_zg, 95.0))
        max_abs = float(np.max(absolute_zg))
        slope = _ground_slope(valid_points[:, :2], zg)
        if slope is None:
            warnings.append("ground_slope_undefined")
    else:
        warnings.append("no_finite_ground_points")

    _append_threshold_violation(
        violations,
        "abs_bias_exceeds_limit",
        bias,
        threshold_values["max_abs_bias_mm"],
        absolute=True,
    )
    _append_threshold_violation(
        violations,
        "rmse_exceeds_limit",
        rmse,
        threshold_values["max_rmse_mm"],
    )
    _append_threshold_violation(
        violations,
        "p95_abs_exceeds_limit",
        p95_abs,
        threshold_values["max_p95_abs_mm"],
    )
    _append_threshold_violation(
        violations,
        "max_abs_exceeds_limit",
        max_abs,
        threshold_values["max_abs_mm"],
    )
    _append_threshold_violation(
        violations,
        "abs_slope_exceeds_limit",
        slope,
        threshold_values["max_abs_slope_mm_per_mm"],
        absolute=True,
    )

    all_reasons = list(dict.fromkeys(warnings + violations))
    status = "VALID" if not all_reasons else "INVALID"
    message = (
        "laser ground sanity passed"
        if status == "VALID"
        else "laser ground sanity failed: " + ", ".join(all_reasons)
    )
    return GroundSanityResult(
        status=status,
        message=message,
        ground_extrinsic_source=source,
        frame_number=None if frame_number is None else int(frame_number),
        session_calibration_frame_number=(
            None
            if session_calibration_frame_number is None
            else int(session_calibration_frame_number)
        ),
        input_point_count=input_count,
        valid_point_count=valid_count,
        bias_zg_mm=bias,
        rmse_zg_mm=rmse,
        p95_abs_zg_mm=p95_abs,
        max_abs_zg_mm=max_abs,
        ground_slope_mm_per_mm=slope,
        warnings=tuple(warnings),
        threshold_violations=tuple(violations),
        thresholds=threshold_values,
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        mask=mask_metadata,
        frame_host_monotonic_ns=(
            None if frame_host_monotonic_ns is None else int(frame_host_monotonic_ns)
        ),
        session_calibration_host_monotonic_ns=(
            None
            if session_calibration_host_monotonic_ns is None
            else int(session_calibration_host_monotonic_ns)
        ),
        session_generation=(
            None if session_generation is None else int(session_generation)
        ),
    )


def _normalise_thresholds(thresholds: Any | None) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for name, default in _DEFAULT_THRESHOLDS.items():
        value = (
            thresholds.get(name, default)
            if isinstance(thresholds, Mapping)
            else getattr(thresholds, name, default)
        )
        if name == "min_valid_points":
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError("min_valid_points 必须是正整数")
            values[name] = int(value)
        else:
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")
            values[name] = numeric
    return values


def _append_threshold_violation(
    violations: list[str],
    name: str,
    value: float | None,
    limit: float | int,
    *,
    absolute: bool = False,
) -> None:
    if value is None:
        return
    measured = abs(value) if absolute else value
    if measured > float(limit):
        violations.append(name)


def _ground_slope(points_xy: np.ndarray, zg: np.ndarray) -> float | None:
    """Return the signed ``dZg/dS`` trend along dominant XY direction."""
    centred = np.asarray(points_xy, dtype=np.float64) - np.mean(points_xy, axis=0)
    if len(centred) < 2:
        return None
    _, singular_values, right_vectors = np.linalg.svd(
        centred, full_matrices=False
    )
    if not len(singular_values) or singular_values[0] <= np.finfo(float).eps:
        return None
    direction = right_vectors[0]
    if direction[0] < 0.0 or (direction[0] == 0.0 and direction[1] < 0.0):
        direction = -direction
    distance = centred @ direction
    denominator = float(np.dot(distance, distance))
    if denominator <= np.finfo(float).eps:
        return None
    centred_zg = np.asarray(zg, dtype=np.float64) - float(np.mean(zg))
    return float(np.dot(distance, centred_zg) / denominator)


__all__ = [
    "GroundSanityResult",
    "evaluate_ground_sanity",
    "full_board_physical_polygon",
    "select_board_ground_points",
    "select_points_inside_board_mask",
]
