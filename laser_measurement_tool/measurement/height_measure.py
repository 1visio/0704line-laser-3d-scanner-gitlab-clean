"""Height measurement from reconstructed laser-line points.

Inputs are 3D points in the ground coordinate system: ``(Xg, Yg, Zg)`` in mm.
When baseline points are available, the module fits a local ground profile
``Zg = a*s + b`` along the measured obstacle direction and subtracts that
local ground height from each obstacle point. Without baseline points it falls
back to the fixed ``Zg = 0`` reference.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .ground_reference import (
    GroundProfileFit,
    LineFitXY,
    MeasurementError,
    fit_ground_profile,
    fit_line_xy,
    robust_sigma,
    validate_points,
)


@dataclass(frozen=True, slots=True)
class MeasurementParams:
    """Robust measurement parameters."""

    outlier_sigma_multiplier: float = 2.0
    outlier_max_iterations: int = 5
    min_baseline_points: int = 20
    min_height_points: int = 20

    def __post_init__(self) -> None:
        if self.outlier_sigma_multiplier <= 0.0:
            raise ValueError("outlier_sigma_multiplier must be positive")
        if self.outlier_max_iterations < 1:
            raise ValueError("outlier_max_iterations must be >= 1")
        if self.min_baseline_points < 2:
            raise ValueError("min_baseline_points must be >= 2")
        if self.min_height_points < 2:
            raise ValueError("min_height_points must be >= 2")


@dataclass(frozen=True, slots=True)
class HeightLineMeasurement:
    """Measured obstacle height-line result."""

    ground_baseline_zg_mm: float
    ground_noise_sigma_mm: float | None
    ground_reference_mode: str
    baseline_fit: LineFitXY | None
    ground_profile_fit: GroundProfileFit | None
    height_fit: LineFitXY
    height_mean_mm: float
    height_median_mm: float
    height_std_mm: float
    length_mm: float
    endpoints_ground: np.ndarray
    angle_with_baseline_deg: float | None
    baseline_point_count: int
    baseline_inlier_count: int
    height_point_count: int
    height_inlier_count: int

# Preserve the old private names for callers/tests while keeping the actual
# fitter in the reusable module above.
_validate_points = validate_points
_robust_sigma = robust_sigma
_fit_line_xy = fit_line_xy
_fit_ground_profile = fit_ground_profile


def measure_height_line(
    baseline_ground: np.ndarray | None,
    height_ground: np.ndarray,
    params: MeasurementParams | None = None,
    *,
    ground_correction_mode: str = "auto",
) -> HeightLineMeasurement:
    """Measure a laser-line obstacle height."""
    if params is None:
        params = MeasurementParams()
    if ground_correction_mode not in {"auto", "session_reference", "zg_zero"}:
        raise ValueError(
            "ground_correction_mode 必须是 auto、session_reference 或 zg_zero"
        )

    height = _validate_points(height_ground, "height line", params.min_height_points)
    if baseline_ground is None:
        baseline = np.empty((0, 3), dtype=np.float64)
    else:
        baseline = _validate_points(
            baseline_ground, "baseline line", params.min_baseline_points
        )

    height_fit = _fit_line_xy(height[:, :2], params, "height line")
    height_inliers = height[height_fit.inlier_mask]

    if ground_correction_mode == "session_reference":
        # The incoming points have already been leveled by the frozen
        # SessionGroundReference.  Keep the baseline ROI available for point
        # counts/diagnostics, but never fit or subtract a second local profile.
        reference_zg = 0.0
        ground_sigma = None
        baseline_z_mask = np.zeros(len(baseline), dtype=bool)
        baseline_fit = None
        ground_profile_fit = None
        ground_reference_mode = "session_reference"
        local_ground_z = np.zeros(len(height_inliers), dtype=np.float64)
    elif len(baseline) == 0 or ground_correction_mode == "zg_zero":
        reference_zg = 0.0
        ground_sigma = None
        baseline_z_mask = np.empty(0, dtype=bool)
        baseline_fit = None
        ground_profile_fit = None
        ground_reference_mode = "zg_zero"
        local_ground_z = np.zeros(len(height_inliers), dtype=np.float64)
    else:
        ground_profile_fit, ground_sigma = _fit_ground_profile(
            baseline,
            params,
            height_fit.centre_xy,
            height_fit.direction_xy,
        )
        baseline_z_mask = ground_profile_fit.inlier_mask
        baseline_fit = _fit_line_xy(
            baseline[baseline_z_mask][:, :2], params, "baseline line"
        )
        local_ground_z = ground_profile_fit.predict_z(height_inliers[:, :2])
        reference_zg = float(np.mean(local_ground_z))
        ground_reference_mode = "baseline_roi_profile"

    relative_heights = height_inliers[:, 2] - local_ground_z
    projections = (
        height_inliers[:, :2] - height_fit.centre_xy
    ) @ height_fit.direction_xy
    length = float(np.max(projections) - np.min(projections))
    mean_height = float(np.mean(relative_heights))

    if ground_profile_fit is None:
        endpoint_z = np.full(2, mean_height)
    else:
        endpoint_z = (
            ground_profile_fit.predict_z(height_fit.endpoints_xy) + mean_height
        )
    endpoints_ground = np.column_stack([height_fit.endpoints_xy, endpoint_z])

    angle: float | None = None
    if baseline_fit is not None:
        cosine = float(
            np.clip(
                np.abs(height_fit.direction_xy @ baseline_fit.direction_xy),
                0.0,
                1.0,
            )
        )
        angle = float(np.degrees(np.arccos(cosine)))

    return HeightLineMeasurement(
        ground_baseline_zg_mm=reference_zg,
        ground_noise_sigma_mm=ground_sigma,
        ground_reference_mode=ground_reference_mode,
        baseline_fit=baseline_fit,
        ground_profile_fit=ground_profile_fit,
        height_fit=height_fit,
        height_mean_mm=mean_height,
        height_median_mm=float(np.median(relative_heights)),
        height_std_mm=float(np.std(relative_heights)),
        length_mm=length,
        endpoints_ground=np.ascontiguousarray(endpoints_ground),
        angle_with_baseline_deg=angle,
        baseline_point_count=len(baseline),
        baseline_inlier_count=int(baseline_z_mask.sum()),
        height_point_count=len(height),
        height_inlier_count=int(height_fit.inlier_mask.sum()),
    )


def measure_height_lines(
    baseline_ground: np.ndarray | None,
    height_groups_ground: Sequence[np.ndarray],
    params: MeasurementParams | None = None,
    *,
    ground_correction_mode: str = "auto",
) -> list[HeightLineMeasurement]:
    """Measure each obstacle group with the same baseline point set."""
    if not height_groups_ground:
        raise MeasurementError("at least one obstacle group is required")

    measurements: list[HeightLineMeasurement] = []
    for index, height_ground in enumerate(height_groups_ground, start=1):
        try:
            measurement = measure_height_line(
                baseline_ground,
                height_ground,
                params,
                ground_correction_mode=ground_correction_mode,
            )
        except MeasurementError as error:
            raise MeasurementError(f"obstacle {index}: {error}") from error
        measurements.append(measurement)
    return measurements
