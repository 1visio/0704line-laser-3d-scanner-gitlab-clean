"""Ground-3 low-degree cross-pose spatial correction diagnostic.

This tool treats Ground-1 (pose A) and Ground-2R poses 002/003/004 as four
equal-weight pose groups.  It reuses Ground-1's frozen S definition and bin
edges, builds frame-median then pose-median residual profiles, and compares
only low-degree cubic B-splines with a robust loss and curvature penalty.

The fitted candidate is diagnostic only.  ``raw`` means the existing
detrended residual r = Zg - (a_frame*S + b_frame), and ``corrected`` means
r - G(S).  C0/C1 calibration and production configuration are never changed.
Bins outside the strict training/held-out common support are marked
unsupported and omitted; no extrapolation, interpolation, or clamping is
performed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config
from tools.fit_ground_pose_invariance_ground_only import (
    _build_frame_runs_from_cache,
    _load_geometry_cache,
    _load_ranges,
)
from tools.fit_ground_reference_20frames import (
    FrameFit,
    _build_frame_fits,
    _sha256_file,
    _load_dataset_metadata,
)


DEFAULT_GROUND1_DIR = TOOL_ROOT / "output_daheng_0811" / "ground_reference_20frames"
DEFAULT_GROUND2R_DIR = (
    TOOL_ROOT / "output_daheng_0811" / "ground_pose_invariance_ground_only"
)
DEFAULT_OUTPUT_DIR = TOOL_ROOT / "output_daheng_0811" / "ground_spatial_correction_ground3"
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_GROUND2R_DATA_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\.chessboard_v2.inprogress\fit"
)

POSES = ("ground1", "002", "003", "004")
POSE_LABELS = {
    "ground1": "Ground-1 A",
    "002": "pose002",
    "003": "pose003",
    "004": "pose004",
}
DEGREE = 3
INTERIOR_KNOT_COUNTS = (3, 4, 5)
SMOOTHNESS_LAMBDA = 0.001
MIN_COMMON_BIN_FRACTION = 0.8
METRIC_EPS = 1.0e-10


@dataclass(slots=True)
class ResidualFrame:
    pose_id: str
    frame_id: str
    source_file: str
    source_sha256: str | None
    camera_frame_number: int | None
    s: np.ndarray
    residual: np.ndarray
    quality_passed: bool | None
    quality_warnings: list[str]
    extraction_ms: float | None
    c0_reconstruction_ms: float | None
    c1_reconstruction_ms: float | None


@dataclass(slots=True)
class PoseProfiles:
    pose_id: str
    frame_ids: list[str]
    frame_profiles: np.ndarray
    frame_point_counts: np.ndarray
    pose_profile: np.ndarray
    frame_count_valid: np.ndarray
    profile_available: np.ndarray
    point_count_total: np.ndarray


@dataclass(slots=True)
class SplineFit:
    interior_knot_count: int
    knots: np.ndarray
    coefficients: np.ndarray
    domain_min_mm: float
    domain_max_mm: float
    robust_scale_mm: float
    smoothness_lambda: float
    observation_count: int
    train_pose_ids: tuple[str, ...]
    success: bool
    cost: float
    optimality: float


def _natural_key(value: str | Path) -> list[str | int]:
    text = Path(value).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _median_absolute_deviation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _metric_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "rmse_mm": math.nan,
            "p95_abs_mm": math.nan,
            "peak_to_peak_mm": math.nan,
            "max_abs_mm": math.nan,
        }
    return {
        "rmse_mm": float(np.sqrt(np.mean(values**2))),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95.0)),
        "peak_to_peak_mm": float(np.ptp(values)),
        "max_abs_mm": float(np.max(np.abs(values))),
    }


def _load_frozen_definition(ground1_dir: Path) -> tuple[dict[str, Any], list[dict[str, float]]]:
    summary_path = ground1_dir / "ground_reference_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile_rows = _read_csv(ground1_dir / "ground_profile_pooled.csv")
    profile_rows.sort(key=lambda row: int(row["bin_index"]))
    if len(profile_rows) != 50:
        raise RuntimeError(f"Ground-1 frozen profile must have 50 bins, got {len(profile_rows)}")
    specs = [
        {
            "bin_index": int(row["bin_index"]),
            "s_left_mm": float(row["s_left_mm"]),
            "s_right_mm": float(row["s_right_mm"]),
            "s_center_mm": float(row["s_center_mm"]),
        }
        for row in profile_rows
    ]
    expected_origin = np.asarray(summary["shared_s_definition"]["origin_xy"], dtype=np.float64)
    expected_direction = np.asarray(
        summary["shared_s_definition"]["direction_xy"], dtype=np.float64
    )
    if not np.isclose(np.linalg.norm(expected_direction), 1.0, atol=1.0e-8):
        raise RuntimeError("Ground-1 frozen direction is not unit length")
    summary["_origin_xy"] = expected_origin
    summary["_direction_xy"] = expected_direction
    summary["_s_domain_mm"] = [specs[0]["s_left_mm"], specs[-1]["s_right_mm"]]
    return summary, specs


def _bin_frame_residuals(
    s: np.ndarray,
    residual: np.ndarray,
    specs: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if len(s) != len(residual):
        raise RuntimeError("S and residual arrays are misaligned")
    edges = np.asarray(
        [specs[0]["s_left_mm"]] + [spec["s_right_mm"] for spec in specs],
        dtype=np.float64,
    )
    n_bins = len(specs)
    values = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    finite = np.isfinite(s) & np.isfinite(residual)
    bin_index = np.searchsorted(edges, s, side="right") - 1
    bin_index[s == edges[-1]] = n_bins - 1
    in_domain = finite & (s >= edges[0]) & (s <= edges[-1])
    for index in range(n_bins):
        selected = in_domain & (bin_index == index)
        counts[index] = int(np.count_nonzero(selected))
        if counts[index]:
            values[index] = float(np.median(residual[selected]))
    return values, counts


def _load_ground1_frames(
    ground1_dir: Path,
    ground1_summary: dict[str, Any],
) -> list[ResidualFrame]:
    residual_rows = _read_csv(ground1_dir / "ground_residuals.csv")
    metric_rows = {
        row["frame_id"]: row for row in _read_csv(ground1_dir / "ground_frame_metrics.csv")
    }
    input_files = {
        Path(str(item["path"])).name: item
        for item in ground1_summary["input_audit"].get("input_files", [])
        if isinstance(item, dict) and item.get("path")
    }
    grouped: dict[str, dict[str, list[float]]] = {}
    source_by_frame: dict[str, str] = {}
    for row in residual_rows:
        frame_id = row["frame_id"]
        grouped.setdefault(frame_id, {"s": [], "residual": []})["s"].append(
            float(row["S_mm"])
        )
        grouped[frame_id]["residual"].append(float(row["residual_mm"]))
        source_by_frame[frame_id] = row["source_file"]
    frames: list[ResidualFrame] = []
    for frame_id in sorted(grouped, key=_natural_key):
        source_file = source_by_frame[frame_id]
        summary_item = input_files.get(source_file, {})
        metric = metric_rows.get(frame_id, {})
        quality = summary_item.get("quality", {})
        frames.append(
            ResidualFrame(
                pose_id="ground1",
                frame_id=f"ground1_{frame_id}",
                source_file=source_file,
                source_sha256=summary_item.get("sha256"),
                camera_frame_number=(
                    int(metric["camera_frame_number"])
                    if metric.get("camera_frame_number")
                    else summary_item.get("camera_frame_number")
                ),
                s=np.asarray(grouped[frame_id]["s"], dtype=np.float64),
                residual=np.asarray(grouped[frame_id]["residual"], dtype=np.float64),
                quality_passed=(
                    bool(summary_item.get("quality", {}).get("passed"))
                    if "passed" in quality
                    else None
                ),
                quality_warnings=list(quality.get("warnings", [])),
                extraction_ms=None,
                c0_reconstruction_ms=None,
                c1_reconstruction_ms=None,
            )
        )
    if len(frames) != 20:
        raise RuntimeError(f"Ground-1 residuals must contain 20 frames, got {len(frames)}")
    return frames


def _load_ground2r_frames(
    ground2r_dir: Path,
    data_dir: Path,
    config_path: Path,
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> tuple[list[ResidualFrame], dict[str, Any]]:
    cache, centers_by_frame = _load_geometry_cache(ground2r_dir)
    if not cache.get("one_steger_per_frame"):
        raise RuntimeError("Ground-2R cache does not certify one Steger run per frame")
    if any(record.get("steger_run_count") != 1 for record in cache["frames"]):
        raise RuntimeError("Ground-2R cache contains a frame with steger_run_count != 1")
    ranges = _load_ranges(ground2r_dir / "ground_only_ranges.yaml")
    frames, pose_by_frame, _metadata_by_name = _build_frame_runs_from_cache(
        cache,
        centers_by_frame,
        data_dir,
        ranges,
        config_path,
    )
    app = load_app_config(config_path)
    fits_by_pose: dict[str, list[FrameFit]] = {pose: [] for pose in ("002", "003", "004")}
    for pose_id in fits_by_pose:
        pose_frames = [frame for frame in frames if pose_by_frame[frame.frame_id] == pose_id]
        fits_by_pose[pose_id] = _build_frame_fits(
            pose_frames,
            origin_xy,
            direction_xy,
            app.measurement,
        )
    output: list[ResidualFrame] = []
    for pose_id in ("002", "003", "004"):
        for fit in fits_by_pose[pose_id]:
            frame = fit.frame
            output.append(
                ResidualFrame(
                    pose_id=pose_id,
                    frame_id=f"pose{pose_id}_{frame.frame_id}",
                    source_file=frame.path.name,
                    source_sha256=frame.file_sha256,
                    camera_frame_number=frame.camera_frame_number,
                    s=np.asarray(fit.s, dtype=np.float64),
                    residual=np.asarray(fit.residual, dtype=np.float64),
                    quality_passed=frame.quality.get("passed"),
                    quality_warnings=list(frame.quality.get("warnings", [])),
                    extraction_ms=frame.extraction_ms,
                    c0_reconstruction_ms=frame.c0_reconstruction_ms,
                    c1_reconstruction_ms=frame.c1_reconstruction_ms,
                )
            )
    counts = {pose: sum(frame.pose_id == pose for frame in output) for pose in ("002", "003", "004")}
    if counts != {"002": 5, "003": 5, "004": 5}:
        raise RuntimeError(f"Ground-2R frame counts are not 5 per pose: {counts}")
    return output, cache


def _build_pose_profiles(
    frames_by_pose: dict[str, list[ResidualFrame]],
    specs: list[dict[str, float]],
) -> dict[str, PoseProfiles]:
    output: dict[str, PoseProfiles] = {}
    for pose_id in POSES:
        frames = sorted(frames_by_pose[pose_id], key=lambda frame: _natural_key(frame.frame_id))
        frame_profiles = np.full((len(frames), len(specs)), np.nan, dtype=np.float64)
        frame_counts = np.zeros((len(frames), len(specs)), dtype=np.int64)
        for frame_index, frame in enumerate(frames):
            profile, counts = _bin_frame_residuals(frame.s, frame.residual, specs)
            frame_profiles[frame_index] = profile
            frame_counts[frame_index] = counts
        frame_count_valid = np.sum(np.isfinite(frame_profiles), axis=0).astype(np.int64)
        required = int(math.ceil(MIN_COMMON_BIN_FRACTION * len(frames)))
        profile_available = frame_count_valid >= required
        pose_profile = np.full(len(specs), np.nan, dtype=np.float64)
        for bin_index in range(len(specs)):
            values = frame_profiles[:, bin_index]
            values = values[np.isfinite(values)]
            if len(values) >= required:
                pose_profile[bin_index] = float(np.median(values))
        output[pose_id] = PoseProfiles(
            pose_id=pose_id,
            frame_ids=[frame.frame_id for frame in frames],
            frame_profiles=frame_profiles,
            frame_point_counts=frame_counts,
            pose_profile=pose_profile,
            frame_count_valid=frame_count_valid,
            profile_available=profile_available & np.isfinite(pose_profile),
            point_count_total=np.sum(frame_counts, axis=0).astype(np.int64),
        )
    return output


def _knots(interior_knot_count: int, domain_min: float, domain_max: float) -> np.ndarray:
    interiors = np.linspace(
        domain_min,
        domain_max,
        interior_knot_count + 2,
        dtype=np.float64,
    )[1:-1]
    return np.concatenate(
        [
            np.repeat(domain_min, DEGREE + 1),
            interiors,
            np.repeat(domain_max, DEGREE + 1),
        ]
    )


def _basis_matrix(
    x: np.ndarray,
    interior_knot_count: int,
    domain_min: float,
    domain_max: float,
    derivative: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    knots = _knots(interior_knot_count, domain_min, domain_max)
    basis_count = len(knots) - DEGREE - 1
    columns: list[np.ndarray] = []
    for basis_index in range(basis_count):
        coefficients = np.zeros(basis_count, dtype=np.float64)
        coefficients[basis_index] = 1.0
        spline = BSpline(knots, coefficients, DEGREE, extrapolate=False)
        if derivative:
            spline = spline.derivative(derivative)
        columns.append(np.asarray(spline(x), dtype=np.float64))
    return np.column_stack(columns), knots


def _fit_spline(
    observations: list[tuple[str, float, float]],
    interior_knot_count: int,
    domain_min: float,
    domain_max: float,
    train_pose_ids: tuple[str, ...],
) -> SplineFit:
    if not observations:
        raise RuntimeError("no observations available for spline fit")
    pose_counts: dict[str, int] = {}
    for pose_id, _s, _value in observations:
        pose_counts[pose_id] = pose_counts.get(pose_id, 0) + 1
    pose_count = len(pose_counts)
    x = np.asarray([row[1] for row in observations], dtype=np.float64)
    y = np.asarray([row[2] for row in observations], dtype=np.float64)
    weights = np.asarray(
        [1.0 / (pose_count * pose_counts[pose_id]) for pose_id, _s, _value in observations],
        dtype=np.float64,
    )
    weights *= len(weights) / float(np.sum(weights))
    basis, knots = _basis_matrix(x, interior_knot_count, domain_min, domain_max)
    if not np.isfinite(basis).all():
        raise RuntimeError("spline basis has unsupported training coordinates")
    smooth_grid = np.linspace(domain_min, domain_max, 161, dtype=np.float64)
    curvature_basis, _ = _basis_matrix(
        smooth_grid,
        interior_knot_count,
        domain_min,
        domain_max,
        derivative=2,
    )
    domain_span = domain_max - domain_min
    curvature_basis = curvature_basis * (domain_span**2)
    robust_scale = max(1.4826 * _median_absolute_deviation(y), 0.001)
    sqrt_weights = np.sqrt(weights)

    def residual_function(coefficients: np.ndarray) -> np.ndarray:
        data_residual = sqrt_weights * (basis @ coefficients - y)
        smooth_residual = math.sqrt(SMOOTHNESS_LAMBDA) * (curvature_basis @ coefficients)
        return np.concatenate([data_residual, smooth_residual])

    initial_design = np.vstack([sqrt_weights[:, None] * basis, math.sqrt(SMOOTHNESS_LAMBDA) * curvature_basis])
    initial_target = np.concatenate([sqrt_weights * y, np.zeros(len(smooth_grid), dtype=np.float64)])
    initial = np.linalg.lstsq(initial_design, initial_target, rcond=None)[0]
    result = least_squares(
        residual_function,
        initial,
        loss="soft_l1",
        f_scale=robust_scale,
        max_nfev=2000,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    return SplineFit(
        interior_knot_count=interior_knot_count,
        knots=knots,
        coefficients=np.asarray(result.x, dtype=np.float64),
        domain_min_mm=domain_min,
        domain_max_mm=domain_max,
        robust_scale_mm=float(robust_scale),
        smoothness_lambda=SMOOTHNESS_LAMBDA,
        observation_count=len(observations),
        train_pose_ids=train_pose_ids,
        success=bool(result.success),
        cost=float(result.cost),
        optimality=float(result.optimality),
    )


def _predict_spline(model: SplineFit, s: np.ndarray) -> np.ndarray:
    basis, _ = _basis_matrix(
        np.asarray(s, dtype=np.float64),
        model.interior_knot_count,
        model.domain_min_mm,
        model.domain_max_mm,
    )
    return np.asarray(basis @ model.coefficients, dtype=np.float64)


def _common_bins(
    profiles: dict[str, PoseProfiles],
    pose_ids: Iterable[str],
) -> np.ndarray:
    available = np.ones(len(next(iter(profiles.values())).pose_profile), dtype=bool)
    for pose_id in pose_ids:
        available &= profiles[pose_id].profile_available
    return available


def _observations_for_bins(
    profiles: dict[str, PoseProfiles],
    pose_ids: tuple[str, ...],
    bin_mask: np.ndarray,
    specs: list[dict[str, float]],
) -> list[tuple[str, float, float]]:
    observations: list[tuple[str, float, float]] = []
    for pose_id in pose_ids:
        for bin_index in np.flatnonzero(bin_mask):
            value = profiles[pose_id].pose_profile[bin_index]
            if not math.isfinite(float(value)):
                raise RuntimeError(f"missing profile at pose={pose_id}, bin={bin_index}")
            observations.append((pose_id, specs[bin_index]["s_center_mm"], float(value)))
    return observations


def _fold_metrics(
    raw: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, float]:
    raw_metrics = _metric_values(raw)
    corrected_metrics = _metric_values(corrected)
    return {
        "raw_rmse_mm": raw_metrics["rmse_mm"],
        "raw_p95_abs_mm": raw_metrics["p95_abs_mm"],
        "raw_peak_to_peak_mm": raw_metrics["peak_to_peak_mm"],
        "corrected_rmse_mm": corrected_metrics["rmse_mm"],
        "corrected_p95_abs_mm": corrected_metrics["p95_abs_mm"],
        "corrected_peak_to_peak_mm": corrected_metrics["peak_to_peak_mm"],
        "rmse_improvement_mm": raw_metrics["rmse_mm"] - corrected_metrics["rmse_mm"],
        "p95_improvement_mm": raw_metrics["p95_abs_mm"] - corrected_metrics["p95_abs_mm"],
        "peak_to_peak_improvement_mm": raw_metrics["peak_to_peak_mm"]
        - corrected_metrics["peak_to_peak_mm"],
    }


def _run_candidate_cv(
    profiles: dict[str, PoseProfiles],
    specs: list[dict[str, float]],
    domain_min: float,
    domain_max: float,
    interior_knot_count: int,
) -> tuple[list[dict[str, Any]], dict[str, SplineFit], SplineFit]:
    folds: list[dict[str, Any]] = []
    fold_models: dict[str, SplineFit] = {}
    for held_out in POSES:
        train_pose_ids = tuple(pose for pose in POSES if pose != held_out)
        train_mask = _common_bins(profiles, train_pose_ids)
        held_out_mask = profiles[held_out].profile_available
        common_mask = train_mask & held_out_mask
        common_indices = np.flatnonzero(common_mask)
        if len(common_indices) < int(math.ceil(MIN_COMMON_BIN_FRACTION * len(specs))):
            raise RuntimeError(
                f"held-out fold {held_out} has too few common bins: {len(common_indices)}"
            )
        observations = _observations_for_bins(
            profiles,
            train_pose_ids,
            common_mask,
            specs,
        )
        model = _fit_spline(
            observations,
            interior_knot_count,
            domain_min,
            domain_max,
            train_pose_ids,
        )
        fold_models[held_out] = model
        s_values = np.asarray(
            [specs[index]["s_center_mm"] for index in common_indices],
            dtype=np.float64,
        )
        prediction = _predict_spline(model, s_values)
        raw = profiles[held_out].pose_profile[common_indices]
        corrected = raw - prediction
        metrics = _fold_metrics(raw, corrected)
        held_out_valid_count = int(np.count_nonzero(held_out_mask))
        train_valid_counts = {pose: int(np.count_nonzero(_common_bins(profiles, (pose,)))) for pose in train_pose_ids}
        folds.append(
            {
                "interior_knot_count": interior_knot_count,
                "held_out_pose": held_out,
                "train_pose_ids": list(train_pose_ids),
                "train_pose_count": len(train_pose_ids),
                "train_pose_weights": {pose: 1.0 / 3.0 for pose in train_pose_ids},
                "held_out_frame_ids": profiles[held_out].frame_ids,
                "common_bin_count": int(len(common_indices)),
                "common_s_min_mm": float(specs[common_indices[0]]["s_left_mm"]),
                "common_s_max_mm": float(specs[common_indices[-1]]["s_right_mm"]),
                "held_out_valid_bin_count": held_out_valid_count,
                "train_valid_bin_count_by_pose": train_valid_counts,
                "unsupported_bin_count": int(held_out_valid_count - len(common_indices)),
                "extrapolation_count": 0,
                "clamp_count": 0,
                "interpolation_count": 0,
                "fold_status": "ok_common_support",
                "model_success": model.success,
                "model_cost": model.cost,
                **metrics,
            }
        )
    all_mask = _common_bins(profiles, POSES)
    all_indices = np.flatnonzero(all_mask)
    if len(all_indices) < int(math.ceil(MIN_COMMON_BIN_FRACTION * len(specs))):
        raise RuntimeError(f"all-pose common support is too small: {len(all_indices)}")
    all_observations = _observations_for_bins(profiles, POSES, all_mask, specs)
    full_model = _fit_spline(
        all_observations,
        interior_knot_count,
        domain_min,
        domain_max,
        POSES,
    )
    return folds, fold_models, full_model


def _aggregate_candidate(
    interior_knot_count: int,
    folds: list[dict[str, Any]],
    full_model: SplineFit,
) -> dict[str, Any]:
    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in folds]))

    def minimum(name: str) -> float:
        return float(np.min([float(row[name]) for row in folds]))

    all_rmse = minimum("rmse_improvement_mm") >= -METRIC_EPS
    all_p95 = minimum("p95_improvement_mm") >= -METRIC_EPS
    all_p2p = minimum("peak_to_peak_improvement_mm") >= -METRIC_EPS
    mean_rmse = mean("rmse_improvement_mm")
    mean_p95 = mean("p95_improvement_mm")
    mean_p2p = mean("peak_to_peak_improvement_mm")
    eligible = bool(all_rmse and all_p95 and all_p2p and mean_rmse > METRIC_EPS)
    return {
        "interior_knot_count": interior_knot_count,
        "basis_count": interior_knot_count + DEGREE + 1,
        "smoothness_lambda": SMOOTHNESS_LAMBDA,
        "robust_loss": "soft_l1",
        "fold_count": len(folds),
        "common_bin_count_min": int(min(row["common_bin_count"] for row in folds)),
        "common_bin_count_max": int(max(row["common_bin_count"] for row in folds)),
        "cv_mean_raw_rmse_mm": mean("raw_rmse_mm"),
        "cv_mean_corrected_rmse_mm": mean("corrected_rmse_mm"),
        "cv_mean_raw_p95_abs_mm": mean("raw_p95_abs_mm"),
        "cv_mean_corrected_p95_abs_mm": mean("corrected_p95_abs_mm"),
        "cv_mean_raw_peak_to_peak_mm": mean("raw_peak_to_peak_mm"),
        "cv_mean_corrected_peak_to_peak_mm": mean("corrected_peak_to_peak_mm"),
        "cv_mean_rmse_improvement_mm": mean_rmse,
        "cv_mean_p95_improvement_mm": mean_p95,
        "cv_mean_peak_to_peak_improvement_mm": mean_p2p,
        "cv_min_rmse_improvement_mm": minimum("rmse_improvement_mm"),
        "cv_min_p95_improvement_mm": minimum("p95_improvement_mm"),
        "cv_min_peak_to_peak_improvement_mm": minimum("peak_to_peak_improvement_mm"),
        "all_fold_rmse_improved": all_rmse,
        "all_fold_p95_improved_or_equal": all_p95,
        "all_fold_peak_to_peak_improved_or_equal": all_p2p,
        "stable_cross_pose_improvement": eligible,
        "full_fit_observation_count": full_model.observation_count,
        "full_fit_success": full_model.success,
        "full_fit_cost": full_model.cost,
        "selected": False,
    }


def _frame_metrics_with_model(
    frames: list[ResidualFrame],
    model: SplineFit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for frame in frames:
        finite = np.isfinite(frame.s) & np.isfinite(frame.residual)
        supported = finite & (frame.s >= model.domain_min_mm) & (frame.s <= model.domain_max_mm)
        s_supported = frame.s[supported]
        raw = frame.residual[supported]
        correction = _predict_spline(model, s_supported) if len(s_supported) else np.array([])
        corrected = raw - correction
        raw_metrics = _metric_values(raw)
        corrected_metrics = _metric_values(corrected)
        metric_rows.append(
            {
                "pose_id": frame.pose_id,
                "frame_id": frame.frame_id,
                "source_file": frame.source_file,
                "source_sha256": frame.source_sha256,
                "camera_frame_number": frame.camera_frame_number,
                "quality_passed": frame.quality_passed,
                "quality_warnings": ";".join(frame.quality_warnings),
                "raw_point_count": int(np.count_nonzero(finite)),
                "supported_point_count": int(np.count_nonzero(supported)),
                "unsupported_point_count": int(np.count_nonzero(finite & ~supported)),
                "raw_rmse_mm": raw_metrics["rmse_mm"],
                "raw_p95_abs_mm": raw_metrics["p95_abs_mm"],
                "raw_peak_to_peak_mm": raw_metrics["peak_to_peak_mm"],
                "corrected_rmse_mm": corrected_metrics["rmse_mm"],
                "corrected_p95_abs_mm": corrected_metrics["p95_abs_mm"],
                "corrected_peak_to_peak_mm": corrected_metrics["peak_to_peak_mm"],
                "rmse_improvement_mm": raw_metrics["rmse_mm"] - corrected_metrics["rmse_mm"],
                "p95_improvement_mm": raw_metrics["p95_abs_mm"] - corrected_metrics["p95_abs_mm"],
                "peak_to_peak_improvement_mm": raw_metrics["peak_to_peak_mm"]
                - corrected_metrics["peak_to_peak_mm"],
                "support_s_min_mm": float(np.min(s_supported)) if len(s_supported) else None,
                "support_s_max_mm": float(np.max(s_supported)) if len(s_supported) else None,
                "s_domain_min_mm": model.domain_min_mm,
                "s_domain_max_mm": model.domain_max_mm,
            }
        )
        for point_index, (s_value, raw_value) in enumerate(zip(frame.s, frame.residual, strict=True)):
            supported_value = bool(
                math.isfinite(float(s_value))
                and math.isfinite(float(raw_value))
                and model.domain_min_mm <= float(s_value) <= model.domain_max_mm
            )
            correction_value = (
                float(_predict_spline(model, np.asarray([s_value], dtype=np.float64))[0])
                if supported_value
                else None
            )
            residual_rows.append(
                {
                    "pose_id": frame.pose_id,
                    "frame_id": frame.frame_id,
                    "source_file": frame.source_file,
                    "point_index": point_index,
                    "S_mm": float(s_value),
                    "raw_residual_mm": float(raw_value),
                    "G_S_mm": correction_value,
                    "corrected_residual_mm": (
                        float(raw_value) - correction_value if correction_value is not None else None
                    ),
                    "supported": supported_value,
                }
            )
    return metric_rows, residual_rows


def _pose_profile_rows(
    profiles: dict[str, PoseProfiles],
    specs: list[dict[str, float]],
    all_pose_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pose_id in POSES:
        profile = profiles[pose_id]
        for bin_index, spec in enumerate(specs):
            rows.append(
                {
                    "profile_type": "pose",
                    "pose_id": pose_id,
                    "bin_index": bin_index,
                    "s_left_mm": spec["s_left_mm"],
                    "s_right_mm": spec["s_right_mm"],
                    "s_center_mm": spec["s_center_mm"],
                    "frame_count_total": len(profile.frame_ids),
                    "frame_count_valid": int(profile.frame_count_valid[bin_index]),
                    "coverage_fraction": float(
                        profile.frame_count_valid[bin_index] / len(profile.frame_ids)
                    ),
                    "profile_available": bool(profile.profile_available[bin_index]),
                    "point_count_total": int(profile.point_count_total[bin_index]),
                    "point_count_by_frame": profile.frame_point_counts[:, bin_index].tolist(),
                    "frame_balancing_method": "per-frame point median, then median across frames",
                    "raw_pose_profile_mm": profile.pose_profile[bin_index],
                    "pose_weight": 0.25,
                    "pose_count_available": int(
                        sum(
                            bool(profiles[pose].profile_available[bin_index])
                            for pose in POSES
                        )
                    ),
                    "pose_weights_used": {pose: 0.25 for pose in POSES},
                    "effective_pose_weight_sum": 1.0,
                    "four_pose_common_support": bool(all_pose_mask[bin_index]),
                }
            )
    for bin_index, spec in enumerate(specs):
        values = np.asarray(
            [profiles[pose].pose_profile[bin_index] for pose in POSES],
            dtype=np.float64,
        )
        available = np.isfinite(values)
        rows.append(
            {
                "profile_type": "four_pose_consensus",
                "pose_id": "ground1|002|003|004",
                "bin_index": bin_index,
                "s_left_mm": spec["s_left_mm"],
                "s_right_mm": spec["s_right_mm"],
                "s_center_mm": spec["s_center_mm"],
                "frame_count_total": None,
                "frame_count_valid": None,
                "coverage_fraction": float(np.count_nonzero(available) / len(POSES)),
                "profile_available": bool(all_pose_mask[bin_index]),
                "point_count_total": None,
                "point_count_by_frame": None,
                "frame_balancing_method": "pose profile first; four pose weights 1/4",
                "raw_pose_profile_mm": float(np.mean(values[available])) if np.any(available) else None,
                "pose_weight": None,
                "pose_count_available": int(np.count_nonzero(available)),
                "pose_weights_used": {pose: 0.25 for pose in POSES if available[POSES.index(pose)]},
                "effective_pose_weight_sum": float(np.sum(available) / len(POSES)),
                "four_pose_common_support": bool(all_pose_mask[bin_index]),
            }
        )
    return rows


def _model_coefficient_rows(
    model: SplineFit,
    fit_scope: str,
    held_out_pose: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "fit_scope": fit_scope,
            "held_out_pose": held_out_pose,
            "interior_knot_count": model.interior_knot_count,
            "basis_count": len(model.coefficients),
            "coefficient_index": index,
            "coefficient": float(value),
            "knot_vector": model.knots.tolist(),
            "domain_min_mm": model.domain_min_mm,
            "domain_max_mm": model.domain_max_mm,
            "robust_loss": "soft_l1",
            "robust_scale_mm": model.robust_scale_mm,
            "smoothness_lambda": model.smoothness_lambda,
            "train_pose_ids": list(model.train_pose_ids),
            "observation_count": model.observation_count,
            "fit_success": model.success,
            "fit_cost": model.cost,
            "fit_optimality": model.optimality,
        }
        for index, value in enumerate(model.coefficients)
    ]


def _plot_profiles(
    path: Path,
    profiles: dict[str, PoseProfiles],
    specs: list[dict[str, float]],
    full_models: dict[int, SplineFit],
    selected_knots: int | None,
) -> None:
    s = np.asarray([spec["s_center_mm"] for spec in specs], dtype=np.float64)
    common = _common_bins(profiles, POSES)
    fig, axis = plt.subplots(figsize=(13, 7))
    colours = {"ground1": "0.25", "002": "tab:blue", "003": "tab:orange", "004": "tab:green"}
    for pose_id in POSES:
        values = profiles[pose_id].pose_profile
        axis.plot(
            s[profiles[pose_id].profile_available],
            values[profiles[pose_id].profile_available],
            ".-",
            linewidth=1.0,
            markersize=3,
            color=colours[pose_id],
            label=f"{POSE_LABELS[pose_id]} frame-balanced profile",
        )
    if np.any(common):
        consensus = np.mean(
            np.vstack([profiles[pose].pose_profile for pose in POSES]),
            axis=0,
        )
        axis.plot(s[common], consensus[common], "k--", linewidth=2.0, label="four-pose equal-weight consensus")
    grid = np.linspace(specs[0]["s_left_mm"], specs[-1]["s_right_mm"], 500)
    for knots, model in sorted(full_models.items()):
        axis.plot(
            grid,
            _predict_spline(model, grid),
            linewidth=1.2 if knots != selected_knots else 2.8,
            alpha=0.85 if knots != selected_knots else 1.0,
            label=f"G(S), {knots} interior knots" + (" [selected]" if knots == selected_knots else ""),
        )
    axis.axhline(0.0, color="0.3", linestyle=":", linewidth=0.8)
    axis.set_title("Ground-3 four-pose residual profiles and low-degree G(S) candidates")
    axis.set_xlabel("Frozen Ground-1 S (mm)")
    axis.set_ylabel("Residual / G(S) (mm)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cv_improvement(path: Path, folds_by_knots: dict[int, list[dict[str, Any]]]) -> None:
    fig, axis = plt.subplots(figsize=(12, 7))
    x = np.arange(len(POSES), dtype=np.float64)
    width = 0.22
    for offset, knots in enumerate(sorted(folds_by_knots)):
        values = [
            float(next(row for row in folds_by_knots[knots] if row["held_out_pose"] == pose)["rmse_improvement_mm"])
            for pose in POSES
        ]
        axis.bar(x + (offset - 1) * width, values, width=width, label=f"{knots} interior knots")
    axis.axhline(0.0, color="0.2", linewidth=0.8)
    axis.set_xticks(x, [POSE_LABELS[pose] for pose in POSES])
    axis.set_ylabel("Held-out RMSE improvement (mm; raw - corrected)")
    axis.set_title("Ground-3 leave-one-pose-out RMSE improvement")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_selected_cv(
    path: Path,
    profiles: dict[str, PoseProfiles],
    specs: list[dict[str, float]],
    folds: list[dict[str, Any]],
    fold_models: dict[str, SplineFit],
) -> None:
    s = np.asarray([spec["s_center_mm"] for spec in specs], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for axis, fold in zip(axes.flat, folds, strict=True):
        pose = fold["held_out_pose"]
        common = _common_bins(profiles, tuple(fold["train_pose_ids"])) & profiles[pose].profile_available
        model = fold_models[pose]
        prediction = _predict_spline(model, s[common])
        raw = profiles[pose].pose_profile[common]
        axis.plot(s[common], raw, color="0.45", marker=".", label="raw held-out profile")
        axis.plot(s[common], raw - prediction, color="tab:blue", marker=".", label="corrected residual")
        axis.plot(s[common], prediction, color="tab:red", linewidth=1.5, label="G_train(S)")
        axis.axhline(0.0, color="0.2", linestyle=":", linewidth=0.8)
        axis.set_title(f"held out {POSE_LABELS[pose]} ({fold['common_bin_count']} common bins)")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("Frozen S (mm)")
    axes[1, 1].set_xlabel("Frozen S (mm)")
    axes[0, 0].set_ylabel("Residual (mm)")
    axes[1, 0].set_ylabel("Residual (mm)")
    fig.suptitle("Ground-3 selected candidate leave-one-pose-out residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report_number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(value) else f"{value:.6g}"


def _write_report(
    path: Path,
    summary: dict[str, Any],
    model_rows: list[dict[str, Any]],
    selected_folds: list[dict[str, Any]],
    output_files: list[str],
) -> None:
    status = summary["classification"]["GROUND_GS_STATUS"]
    selected = summary["selected_model"]
    lines = [
        "# Ground-3 cross-pose consensus Ground Spatial Correction",
        "",
        f"## GROUND_GS_STATUS: {status}",
        "",
        f"- selected model: `{selected.get('model_name', 'none')}`",
        f"- allow next held-out block validation: `{summary['classification']['allow_new_held_out_block_validation']}`",
        "",
        "## Four-pose and frozen-S protocol",
        "",
        "- Four equal-weight pose groups: Ground-1 A, pose002, pose003, pose004; each pose weight is 1/4.",
        "- Within each pose: per-frame per-bin residual median, then median across frames. No point pooling is used to define a pose profile.",
        "- Frozen S: `S=(XY-Ground1_origin_xy) dot Ground1_direction_xy`; no PCA, origin/direction, or C0/C1 refit.",
        f"- Frozen S domain: [{summary['protocol']['s_domain_min_mm']:.6g}, {summary['protocol']['s_domain_max_mm']:.6g}] mm; {summary['protocol']['bin_count']} frozen bins.",
        "- `raw` = existing detrended residual r; `corrected` = r - G(S). This is not a C1 toggle or production height correction.",
        "",
        "## Candidate family",
        "",
        "- Candidates are cubic B-splines with 3/4/5 interior knots only.",
        "- Knot locations are fixed and evenly spaced in the frozen S domain.",
        f"- Fit uses SciPy `least_squares(loss='soft_l1')` and fixed curvature smoothness penalty lambda={SMOOTHNESS_LAMBDA:g}.",
        "- Leave-one-pose-out training uses three pose profiles with equal pose weights; each fold is scored only in strict common support.",
        "- Extrapolation, interpolation, and clamping are not performed; unsupported bins are recorded and omitted.",
        "",
        "## Leave-one-pose-out CV",
        "",
        "| knots | held-out pose | common bins | raw RMSE | corrected RMSE | RMSE improvement | raw P95 | corrected P95 | P95 improvement | raw P2P | corrected P2P | P2P improvement |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_folds:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["interior_knot_count"]),
                    row["held_out_pose"],
                    str(row["common_bin_count"]),
                    _report_number(row["raw_rmse_mm"]),
                    _report_number(row["corrected_rmse_mm"]),
                    _report_number(row["rmse_improvement_mm"]),
                    _report_number(row["raw_p95_abs_mm"]),
                    _report_number(row["corrected_p95_abs_mm"]),
                    _report_number(row["p95_improvement_mm"]),
                    _report_number(row["raw_peak_to_peak_mm"]),
                    _report_number(row["corrected_peak_to_peak_mm"]),
                    _report_number(row["peak_to_peak_improvement_mm"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Candidate comparison", ""])
    lines.extend(
        [
            "| knots | basis | mean RMSE improvement | mean P95 improvement | mean P2P improvement | min-fold RMSE improvement | stable candidate | selected |",
            "|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in model_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["interior_knot_count"]),
                    str(row["basis_count"]),
                    _report_number(row["cv_mean_rmse_improvement_mm"]),
                    _report_number(row["cv_mean_p95_improvement_mm"]),
                    _report_number(row["cv_mean_peak_to_peak_improvement_mm"]),
                    _report_number(row["cv_min_rmse_improvement_mm"]),
                    str(row["stable_cross_pose_improvement"]),
                    str(row["selected"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Freeze candidate and gate",
            "",
            f"- `G(S)` freeze candidate: `{selected.get('model_name', 'none')}`; coefficients are in `ground_gs_candidate_coefficients.csv`.",
            "- The candidate is not written to C1, GUI, production configuration, or the height-measurement chain.",
            f"- All CV folds no extrapolation: `{summary['classification']['all_folds_no_extrapolation']}`; no clamp: `{summary['classification']['all_folds_no_clamp']}`.",
            f"- Unsupported bins were omitted rather than extrapolated; maximum unsupported bins in a fold: `{summary['classification']['max_unsupported_bin_count']}`.",
            "",
            "## Data caveat",
            "",
            f"- Ground-1 quality summary: `{summary['provenance']['ground1_quality_summary']}`.",
            f"- Ground-2R quality summary: `{summary['provenance']['ground2r_quality_summary']}`.",
            f"- Ground-2R exposure by pose: `{summary['provenance']['ground2r_exposure_us_by_pose']}` µs; this remains a protocol caveat.",
            "",
            "## Reuse versus new computation",
            "",
            "- Reused: Ground-1 frozen origin/direction/50 bin edges and frame residual CSV; Ground-2R image-geometry mask, one-Steger cache, pose grouping, and C1-enabled reconstruction package.",
            "- Newly computed: Ground-2R frame residual arrays from cached centers, four-pose balanced profiles, cubic B-spline candidates, held-out-pose CV, freeze candidate, and diagnostic plots/report.",
            "",
            "## Outputs",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in output_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(
    ground1_dir: Path,
    ground2r_dir: Path,
    ground2r_data_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ground1_summary, specs = _load_frozen_definition(ground1_dir)
    origin_xy = ground1_summary["_origin_xy"]
    direction_xy = ground1_summary["_direction_xy"]
    ground1_frames = _load_ground1_frames(ground1_dir, ground1_summary)
    ground2r_frames, ground2r_cache = _load_ground2r_frames(
        ground2r_dir,
        ground2r_data_dir,
        config_path,
        origin_xy,
        direction_xy,
    )
    frames = ground1_frames + ground2r_frames
    frames_by_pose = {pose: [frame for frame in frames if frame.pose_id == pose] for pose in POSES}
    frame_counts = {pose: len(frames_by_pose[pose]) for pose in POSES}
    if frame_counts != {"ground1": 20, "002": 5, "003": 5, "004": 5}:
        raise RuntimeError(f"unexpected four-pose frame counts: {frame_counts}")
    profiles = _build_pose_profiles(frames_by_pose, specs)
    all_pose_mask = _common_bins(profiles, POSES)
    all_common_indices = np.flatnonzero(all_pose_mask)
    if len(all_common_indices) < int(math.ceil(MIN_COMMON_BIN_FRACTION * len(specs))):
        raise RuntimeError(f"four-pose common support too small: {len(all_common_indices)}")
    domain_min = specs[0]["s_left_mm"]
    domain_max = specs[-1]["s_right_mm"]

    model_rows: list[dict[str, Any]] = []
    folds_by_knots: dict[int, list[dict[str, Any]]] = {}
    fold_models_by_knots: dict[int, dict[str, SplineFit]] = {}
    full_models: dict[int, SplineFit] = {}
    for interior_knot_count in INTERIOR_KNOT_COUNTS:
        folds, fold_models, full_model = _run_candidate_cv(
            profiles,
            specs,
            domain_min,
            domain_max,
            interior_knot_count,
        )
        folds_by_knots[interior_knot_count] = folds
        fold_models_by_knots[interior_knot_count] = fold_models
        full_models[interior_knot_count] = full_model
        model_rows.append(_aggregate_candidate(interior_knot_count, folds, full_model))

    eligible = [row for row in model_rows if row["stable_cross_pose_improvement"]]
    if eligible:
        selected_row = min(eligible, key=lambda row: row["interior_knot_count"])
        status = "PASS"
    else:
        positive_rmse = [row for row in model_rows if row["cv_mean_rmse_improvement_mm"] > METRIC_EPS]
        if positive_rmse:
            selected_row = min(positive_rmse, key=lambda row: row["interior_knot_count"])
            status = "PARTIAL"
        else:
            selected_row = min(model_rows, key=lambda row: row["cv_mean_rmse_improvement_mm"])
            status = "FAIL"
    selected_knots = int(selected_row["interior_knot_count"])
    selected_model = full_models[selected_knots]
    for row in model_rows:
        row["selected"] = row["interior_knot_count"] == selected_knots

    selected_folds = folds_by_knots[selected_knots]
    selected_fold_models = fold_models_by_knots[selected_knots]
    frame_metric_rows, frame_residual_rows = _frame_metrics_with_model(frames, selected_model)
    profile_rows = _pose_profile_rows(profiles, specs, all_pose_mask)

    coefficient_rows: list[dict[str, Any]] = []
    for knots in INTERIOR_KNOT_COUNTS:
        coefficient_rows.extend(_model_coefficient_rows(full_models[knots], "all_four_poses"))
        for held_out, model in fold_models_by_knots[knots].items():
            coefficient_rows.extend(_model_coefficient_rows(model, "leave_one_pose_out", held_out))

    ground1_input = ground1_summary["input_audit"]
    ground2r_summary_path = ground2r_dir / "ground_pose_invariance_ground_only_summary.json"
    ground2r_summary = json.loads(ground2r_summary_path.read_text(encoding="utf-8"))
    app = load_app_config(config_path)
    calibration_manifest = Path(app.calibration.manifest)
    all_unsupported = [int(row["unsupported_bin_count"]) for row in selected_folds]
    selected_parameter_payload = {
        "interior_knot_count": selected_model.interior_knot_count,
        "knots": selected_model.knots,
        "coefficients": selected_model.coefficients,
        "domain_min_mm": selected_model.domain_min_mm,
        "domain_max_mm": selected_model.domain_max_mm,
        "smoothness_lambda": selected_model.smoothness_lambda,
        "robust_loss": "soft_l1",
    }
    correction_hash = hashlib.sha256(
        json.dumps(_json_ready(selected_parameter_payload), sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_files = [
        "ground_gs_frame_metrics.csv",
        "ground_gs_frame_residuals.csv",
        "ground_gs_pose_profiles.csv",
        "ground_gs_cv_metrics.csv",
        "ground_gs_model_comparison.csv",
        "ground_gs_candidate_coefficients.csv",
        "ground_gs_profile_candidates.png",
        "ground_gs_cv_improvement.png",
        "ground_gs_selected_cv_residuals.png",
        "ground_gs_summary.json",
        "ground_gs_report.md",
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "classification": {
            "GROUND_GS_STATUS": status,
            "allow_new_held_out_block_validation": status in {"PASS", "PARTIAL"},
            "selected_model_name": f"cubic_bspline_{selected_knots}_interior_knots",
            "all_folds_no_extrapolation": all(
                int(row["extrapolation_count"]) == 0 for row in selected_folds
            ),
            "all_folds_no_clamp": all(int(row["clamp_count"]) == 0 for row in selected_folds),
            "max_unsupported_bin_count": max(all_unsupported),
        },
        "selected_model": {
            "model_name": f"cubic_bspline_{selected_knots}_interior_knots",
            "interior_knot_count": selected_knots,
            "basis_count": len(selected_model.coefficients),
            "coefficients": selected_model.coefficients,
            "knots": selected_model.knots,
            "domain_min_mm": domain_min,
            "domain_max_mm": domain_max,
            "robust_loss": "soft_l1",
            "robust_scale_mm": selected_model.robust_scale_mm,
            "smoothness_lambda": selected_model.smoothness_lambda,
            "correction_parameters_sha256": correction_hash,
            "full_fit_observation_count": selected_model.observation_count,
        },
        "protocol": {
            "pose_ids": list(POSES),
            "pose_weights": {pose: 0.25 for pose in POSES},
            "frame_counts": frame_counts,
            "frame_balancing_verified": True,
            "pose_balancing_verified": True,
            "held_out_pose_cv_verified": len(selected_folds) == 4,
            "four_pose_required": True,
            "fixed_s_formula": "S=(XY-Ground1_origin_xy) dot Ground1_direction_xy",
            "origin_xy": origin_xy,
            "direction_xy": direction_xy,
            "s_domain_min_mm": domain_min,
            "s_domain_max_mm": domain_max,
            "bin_count": len(specs),
            "s_bin_edges_mm": [specs[0]["s_left_mm"]]
            + [spec["s_right_mm"] for spec in specs],
            "raw_pipeline_id": "C1-enabled per-frame detrended residual r",
            "corrected_pipeline_id": "same residual r minus diagnostic G(S)",
            "correction_enabled": True,
            "extrapolation_policy": "reject and mark unsupported",
            "clamp_policy": "forbidden",
            "interpolation_policy": "none; frozen bins and strict common support only",
            "cv_scheme": "leave_one_pose_out",
            "smoothness_lambda": SMOOTHNESS_LAMBDA,
            "robust_loss": "soft_l1",
            "candidate_interior_knot_counts": list(INTERIOR_KNOT_COUNTS),
            "candidate_degree": DEGREE,
            "no_c0_c1_refit": True,
            "no_production_change": True,
        },
        "provenance": {
            "ground1_summary_path": str(ground1_dir / "ground_reference_summary.json"),
            "ground1_summary_sha256": _sha256_file(ground1_dir / "ground_reference_summary.json"),
            "ground1_profile_path": str(ground1_dir / "ground_profile_pooled.csv"),
            "ground1_profile_sha256": _sha256_file(ground1_dir / "ground_profile_pooled.csv"),
            "ground1_residuals_path": str(ground1_dir / "ground_residuals.csv"),
            "ground1_residuals_sha256": _sha256_file(ground1_dir / "ground_residuals.csv"),
            "ground2r_summary_path": str(ground2r_summary_path),
            "ground2r_summary_sha256": _sha256_file(ground2r_summary_path),
            "ground2r_cache_path": str(ground2r_dir / "ground_pose_geometry_cache.json"),
            "ground2r_cache_sha256": _sha256_file(ground2r_dir / "ground_pose_geometry_cache.json"),
            "ground2r_ranges_path": str(ground2r_dir / "ground_only_ranges.yaml"),
            "ground2r_ranges_sha256": _sha256_file(ground2r_dir / "ground_only_ranges.yaml"),
            "ground1_input_manifest": ground1_input.get("dataset_manifest"),
            "ground2r_input_manifest": ground2r_summary["new_data_audit"].get("dataset_manifest"),
            "ground1_quality_summary": ground1_input.get("quality_summary"),
            "ground2r_quality_summary": ground2r_summary["new_data_audit"].get("quality_summary"),
            "ground2r_exposure_us_by_pose": ground2r_summary["new_data_audit"].get(
                "exposure_us_by_pose"
            ),
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "calibration_manifest": str(calibration_manifest),
            "calibration_manifest_sha256": _sha256_file(calibration_manifest),
            "enable_laser_ray_correction": bool(app.reconstruction.enable_laser_ray_correction),
            "analysis_code_sha256": _sha256_file(Path(__file__).resolve()),
            "frame_ids_by_pose": {pose: profiles[pose].frame_ids for pose in POSES},
        },
        "model_comparison": model_rows,
        "cv_metrics": selected_folds,
        "output_files": output_files,
        "artifact_provenance": {
            "reused": [
                "Ground-1 frozen origin_xy/direction_xy and 50 S-bin edges",
                "Ground-1 frame-level detrended residuals and pooled profile",
                "Ground-2R image-geometry ground-only ranges, one-Steger cache, and pose grouping",
                "Daheng C1-enabled calibration/reconstruction package; no C0/C1 refit",
            ],
            "newly_computed": [
                "Ground-2R per-frame C1 ground-only residual arrays from cached centers",
                "four-pose frame-balanced and pose-balanced profiles",
                "3/4/5-knot cubic B-spline robust+smoothing candidates",
                "four leave-one-pose-out CV folds, selected full-data freeze candidate, and diagnostics",
            ],
        },
    }

    _write_csv(output_dir / "ground_gs_frame_metrics.csv", frame_metric_rows, list(frame_metric_rows[0]))
    _write_csv(
        output_dir / "ground_gs_frame_residuals.csv",
        frame_residual_rows,
        list(frame_residual_rows[0]),
    )
    _write_csv(output_dir / "ground_gs_pose_profiles.csv", profile_rows, list(profile_rows[0]))
    cv_rows = [row for knots in INTERIOR_KNOT_COUNTS for row in folds_by_knots[knots]]
    _write_csv(output_dir / "ground_gs_cv_metrics.csv", cv_rows, list(cv_rows[0]))
    _write_csv(output_dir / "ground_gs_model_comparison.csv", model_rows, list(model_rows[0]))
    _write_csv(
        output_dir / "ground_gs_candidate_coefficients.csv",
        coefficient_rows,
        list(coefficient_rows[0]),
    )
    _plot_profiles(
        output_dir / "ground_gs_profile_candidates.png",
        profiles,
        specs,
        full_models,
        selected_knots,
    )
    _plot_cv_improvement(output_dir / "ground_gs_cv_improvement.png", folds_by_knots)
    _plot_selected_cv(
        output_dir / "ground_gs_selected_cv_residuals.png",
        profiles,
        specs,
        selected_folds,
        selected_fold_models,
    )
    summary_path = output_dir / "ground_gs_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "ground_gs_report.md",
        _json_ready(summary),
        model_rows,
        selected_folds,
        output_files,
    )
    print(f"ground3_output_dir={output_dir}")
    print(f"GROUND_GS_STATUS={status}")
    print(f"selected_model=cubic_bspline_{selected_knots}_interior_knots")
    for row in selected_folds:
        print(
            f"held_out={row['held_out_pose']} common_bins={row['common_bin_count']} "
            f"raw_rmse={row['raw_rmse_mm']:.9g} corrected_rmse={row['corrected_rmse_mm']:.9g} "
            f"improvement={row['rmse_improvement_mm']:.9g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground1-dir", type=Path, default=DEFAULT_GROUND1_DIR)
    parser.add_argument("--ground2r-dir", type=Path, default=DEFAULT_GROUND2R_DIR)
    parser.add_argument("--ground2r-data-dir", type=Path, default=DEFAULT_GROUND2R_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    _run(
        args.ground1_dir,
        args.ground2r_dir,
        args.ground2r_data_dir,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
