"""Fit a shared session ground profile from 20 Daheng laser frames.

This is a diagnostic tool.  It deliberately reuses the online Steger and
reconstruction entry points, but does not modify the GUI or production
measurement path.

Protocol:

* one Steger extraction per source image;
* the same center array is passed to the C0 base reconstruction and to the
  C1-enabled reconstruction;
* every C1-valid reconstructed point is retained (no analytical ROI and no
  chessboard-gap interpolation);
* one robust XY origin/direction is estimated from all frames and then used
  unchanged for every S coordinate;
* per-frame Zg = a*S + b fits use the existing robust ground-profile kernel.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Allow the tool to be run as ``python tools/<script>.py`` from the tool root,
# matching the existing diagnostic scripts in this repository.
TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config
from calibration.manifest import load_calibration_package
from laser.backends import create_extraction_params
from laser.laser_extractor import extract_laser_center
from measurement.height_measure import (
    MeasurementParams,
    _fit_ground_profile,
    _fit_line_xy,
    _robust_sigma,
)
from reconstruction.reconstructor import reconstruct_uv_to_ground


DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_DATA_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\chessboard\fit"
)
DEFAULT_OUTPUT_DIR = TOOL_ROOT / "output_daheng_0811" / "ground_reference_20frames"


@dataclass(slots=True)
class FrameRun:
    index: int
    frame_id: str
    path: Path
    camera_frame_number: int | None
    centers_uv: np.ndarray
    valid_pixels_uv: np.ndarray
    points_ground: np.ndarray
    c0_point_count: int
    c1_point_count: int
    c0_filtered: dict[str, int]
    c1_filtered: dict[str, int]
    image_shape: tuple[int, int]
    image_dtype: str
    file_sha256: str
    quality: dict[str, Any]
    extraction_ms: float
    c0_reconstruction_ms: float
    c1_reconstruction_ms: float


@dataclass(slots=True)
class FrameFit:
    frame: FrameRun
    s: np.ndarray
    predicted_z: np.ndarray
    residual: np.ndarray
    inlier_mask: np.ndarray
    slope: float
    intercept: float
    fit_rmse: float
    sigma: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(path: Path) -> list[str | int]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _finite_or_none(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


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


def _filtered_total(filtered: dict[str, int]) -> int:
    return int(sum(int(value) for value in filtered.values()))


def _load_dataset_metadata(data_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path, Path]:
    dataset_dir = data_dir.resolve().parent
    manifest_path = dataset_dir / "dataset_manifest.yaml"
    frames_csv_path = dataset_dir / "frames.csv"
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError(f"dataset manifest root is not a mapping: {manifest_path}")

    by_name: dict[str, dict[str, Any]] = {}
    for entry in document.get("frames", []):
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        if filename:
            by_name[Path(str(filename)).name] = entry

    return document, by_name, manifest_path, frames_csv_path


def _read_frame(
    index: int,
    path: Path,
    metadata: dict[str, Any],
    extraction_params: Any,
    calibration: dict[str, Any],
    params_c0: Any,
    params_c1: Any,
) -> FrameRun:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read TIFF: {path}")
    if image.ndim != 2 or image.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise RuntimeError(
            f"expected 2-D uint8/uint16 image, got {image.shape} {image.dtype}: {path}"
        )

    # The source TIFFs are full-frame images.  The configured Steger search
    # rectangle is handled by the detector adapter; (0, 0) is the image origin
    # and no measurement/output ROI is applied here.
    extraction_start = time.perf_counter()
    centers_uv = extract_laser_center(
        image,
        extraction_params,
        image_offset=(0, 0),
    )
    extraction_ms = (time.perf_counter() - extraction_start) * 1000.0

    centers_full = np.ascontiguousarray(centers_uv, dtype=np.float64)
    c0_start = time.perf_counter()
    c0_result = reconstruct_uv_to_ground(centers_full, calibration, params_c0)
    c0_ms = (time.perf_counter() - c0_start) * 1000.0

    c1_start = time.perf_counter()
    c1_result = reconstruct_uv_to_ground(centers_full, calibration, params_c1)
    c1_ms = (time.perf_counter() - c1_start) * 1000.0

    points_ground = np.ascontiguousarray(c1_result.points_ground, dtype=np.float64)
    if len(points_ground) != len(c1_result.pixels_uv):
        raise RuntimeError(f"C1 result arrays are misaligned: {path}")
    if len(points_ground) == 0:
        raise RuntimeError(f"C1 reconstructed zero valid points: {path}")
    if not np.isfinite(points_ground).all():
        raise RuntimeError(f"C1 reconstructed non-finite points: {path}")

    quality = metadata.get("quality", {})
    if not isinstance(quality, dict):
        quality = {}
    return FrameRun(
        index=index,
        frame_id=f"frame_{index:02d}",
        path=path,
        camera_frame_number=(
            int(metadata["camera_frame_number"])
            if metadata.get("camera_frame_number") is not None
            else None
        ),
        centers_uv=centers_full,
        valid_pixels_uv=np.ascontiguousarray(c1_result.pixels_uv, dtype=np.float64),
        points_ground=points_ground,
        c0_point_count=int(c0_result.point_count),
        c1_point_count=int(c1_result.point_count),
        c0_filtered={str(key): int(value) for key, value in c0_result.filtered.items()},
        c1_filtered={str(key): int(value) for key, value in c1_result.filtered.items()},
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        image_dtype=str(image.dtype),
        file_sha256=_sha256_file(path),
        quality={
            "passed": quality.get("passed"),
            "warnings": quality.get("warnings", []),
            "dynamic_range_u8": quality.get("dynamic_range_u8"),
            "dark_fraction": quality.get("dark_fraction"),
            "laser_coverage": quality.get("laser_coverage"),
        },
        extraction_ms=float(extraction_ms),
        c0_reconstruction_ms=float(c0_ms),
        c1_reconstruction_ms=float(c1_ms),
    )


def _fit_weighted_linear(
    s: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    selected_s = s[mask]
    selected_z = z[mask]
    selected_weights = weights[mask]
    if float(np.ptp(selected_s)) < 1.0:
        return 0.0, float(np.average(selected_z, weights=selected_weights))
    design = np.column_stack([selected_s, np.ones_like(selected_s)])
    sqrt_weights = np.sqrt(selected_weights)
    slope, intercept = np.linalg.lstsq(
        design * sqrt_weights[:, None],
        selected_z * sqrt_weights,
        rcond=None,
    )[0]
    return float(slope), float(intercept)


def _fit_frame_balanced_pooled(
    fits: list[FrameFit],
    params: MeasurementParams,
) -> dict[str, Any]:
    s = np.concatenate([fit.s for fit in fits])
    z = np.concatenate([fit.frame.points_ground[:, 2] for fit in fits])
    frame_index = np.concatenate(
        [np.full(len(fit.s), fit.frame.index, dtype=np.int16) for fit in fits]
    )
    weights = np.concatenate(
        [np.full(len(fit.s), 1.0 / len(fit.s), dtype=np.float64) for fit in fits]
    )
    mask = np.ones(len(s), dtype=bool)
    slope = 0.0
    intercept = float(np.median(z))
    sigma = _robust_sigma(z - intercept)
    for _ in range(params.outlier_max_iterations):
        if int(mask.sum()) < 2:
            raise RuntimeError("pooled ground profile has too few robust inliers")
        slope, intercept = _fit_weighted_linear(s, z, weights, mask)
        residual = z - (slope * s + intercept)
        sigma = _robust_sigma(residual[mask])
        if sigma <= np.finfo(np.float64).eps:
            break
        new_mask = np.abs(residual) <= params.outlier_sigma_multiplier * sigma
        if int(new_mask.sum()) < 2 or bool(np.all(new_mask == mask)):
            break
        mask = new_mask

    residual = z - (slope * s + intercept)
    return {
        "s": s,
        "z": z,
        "frame_index": frame_index,
        "weights": weights,
        "inlier_mask": mask,
        "slope": float(slope),
        "intercept": float(intercept),
        "sigma": float(sigma),
        "residual": residual,
    }


def _metric_values(residual: np.ndarray) -> tuple[float, float, float]:
    absolute = np.abs(residual)
    return (
        float(np.sqrt(np.mean(residual**2))),
        float(np.percentile(absolute, 95.0)),
        float(np.max(absolute)),
    )


def _build_frame_fits(
    frames: list[FrameRun],
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    params: MeasurementParams,
) -> list[FrameFit]:
    fits: list[FrameFit] = []
    for frame in frames:
        xy = frame.points_ground[:, :2]
        s = (xy - origin_xy) @ direction_xy
        profile, sigma = _fit_ground_profile(
            frame.points_ground,
            params,
            origin_xy,
            direction_xy,
        )
        predicted_z = profile.slope_z_per_mm * s + profile.intercept_z_mm
        residual = frame.points_ground[:, 2] - predicted_z
        fits.append(
            FrameFit(
                frame=frame,
                s=np.ascontiguousarray(s),
                predicted_z=np.ascontiguousarray(predicted_z),
                residual=np.ascontiguousarray(residual),
                inlier_mask=np.ascontiguousarray(profile.inlier_mask),
                slope=float(profile.slope_z_per_mm),
                intercept=float(profile.intercept_z_mm),
                fit_rmse=float(profile.rmse_mm),
                sigma=float(sigma),
            )
        )
    return fits


def _build_binned_profile(
    fits: list[FrameFit],
    pooled: dict[str, Any],
    bin_count: int,
) -> list[dict[str, Any]]:
    all_s = np.concatenate([fit.s for fit in fits])
    low = float(np.min(all_s))
    high = float(np.max(all_s))
    if high <= low:
        high = low + 1.0
    edges = np.linspace(low, high, bin_count + 1)
    rows: list[dict[str, Any]] = []

    for bin_index in range(bin_count):
        left = float(edges[bin_index])
        right = float(edges[bin_index + 1])
        frame_medians_z: list[float] = []
        frame_medians_residual: list[float] = []
        frame_s_centres: list[float] = []
        point_count = 0
        for fit in fits:
            in_bin = (fit.s >= left) & (
                fit.s <= right if bin_index == bin_count - 1 else fit.s < right
            )
            count = int(np.count_nonzero(in_bin))
            if count == 0:
                continue
            frame_s_centres.append(float(np.median(fit.s[in_bin])))
            frame_medians_z.append(
                float(np.median(fit.frame.points_ground[in_bin, 2]))
            )
            frame_medians_residual.append(float(np.median(fit.residual[in_bin])))
            point_count += count

        # The pooled point arrays use the same frame ordering as fits.  Keep
        # this explicit so the CSV can expose both frame-balanced and raw
        # point-weighted views of the same common S bins.
        pooled_bin = (pooled["s"] >= left) & (
            pooled["s"] <= right if bin_index == bin_count - 1 else pooled["s"] < right
        )
        pooled_point_residual = pooled["residual"][pooled_bin]
        pooled_point_z = pooled["z"][pooled_bin]
        s_center = (left + right) / 2.0
        rows.append(
            {
                "bin_index": bin_index,
                "s_left_mm": left,
                "s_right_mm": right,
                "s_center_mm": s_center,
                "frame_count": len(frame_medians_residual),
                "coverage_fraction": len(frame_medians_residual) / len(fits),
                "point_count": point_count,
                "frame_balanced_s_median_mm": (
                    float(np.mean(frame_s_centres)) if frame_s_centres else None
                ),
                "frame_balanced_z_mean_mm": (
                    float(np.mean(frame_medians_z)) if frame_medians_z else None
                ),
                "frame_balanced_z_median_mm": (
                    float(np.median(frame_medians_z)) if frame_medians_z else None
                ),
                "frame_balanced_z_std_mm": (
                    float(np.std(frame_medians_z)) if frame_medians_z else None
                ),
                "pooled_model_z_mm": float(
                    pooled["slope"] * s_center + pooled["intercept"]
                ),
                "detrended_residual_mean_mm": (
                    float(np.mean(frame_medians_residual))
                    if frame_medians_residual
                    else None
                ),
                "detrended_residual_median_mm": (
                    float(np.median(frame_medians_residual))
                    if frame_medians_residual
                    else None
                ),
                "detrended_residual_std_mm": (
                    float(np.std(frame_medians_residual))
                    if frame_medians_residual
                    else None
                ),
                "pooled_point_z_mean_mm": (
                    float(np.mean(pooled_point_z)) if len(pooled_point_z) else None
                ),
                "pooled_point_residual_mean_mm": (
                    float(np.mean(pooled_point_residual))
                    if len(pooled_point_residual)
                    else None
                ),
            }
        )
    return rows


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "range": {
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "span": float(np.ptp(array)),
        },
    }


def _spatial_structure_check(
    binned: list[dict[str, Any]],
    residual_sigma: float,
    frame_count: int,
) -> dict[str, Any]:
    covered = [
        row
        for row in binned
        if row["frame_count"] >= max(2, int(math.ceil(0.8 * frame_count)))
        and row["detrended_residual_median_mm"] is not None
    ]
    if not covered:
        return {
            "status": "insufficient_coverage",
            "covered_bin_count": 0,
            "coverage_requirement_frames": max(2, int(math.ceil(0.8 * frame_count))),
            "median_profile_rms_mm": None,
            "median_profile_peak_to_peak_mm": None,
            "noise_sigma_mm": float(residual_sigma),
            "structure_to_noise_ratio": None,
            "profile_slope_mm_per_mm": None,
        }

    s = np.asarray([row["s_center_mm"] for row in covered], dtype=np.float64)
    profile = np.asarray(
        [row["detrended_residual_median_mm"] for row in covered], dtype=np.float64
    )
    repeatability = np.asarray(
        [row["detrended_residual_std_mm"] for row in covered], dtype=np.float64
    )
    profile_rms = float(np.sqrt(np.mean(profile**2)))
    peak_to_peak = float(np.ptp(profile))
    repeatability_sigma = float(np.median(repeatability))
    correlation = float(np.corrcoef(s, profile)[0, 1]) if len(s) > 1 else 0.0
    if not math.isfinite(correlation):
        correlation = 0.0
    # The question here is whether the residual shape repeats across frames,
    # not whether its amplitude exceeds individual point noise.  A repeated
    # bin-median waveform can be much larger than the across-frame spread
    # while remaining smaller than the raw point-wise residual sigma.
    amplitude_threshold = max(3.0 * repeatability_sigma, 1.0e-6)
    stable_candidate = (
        profile_rms > amplitude_threshold and peak_to_peak > amplitude_threshold
    )
    return {
        "status": "stable_spatial_structure_candidate"
        if stable_candidate
        else "no_clear_stable_spatial_structure",
        "covered_bin_count": len(covered),
        "coverage_requirement_frames": max(2, int(math.ceil(0.8 * frame_count))),
        "median_profile_rms_mm": profile_rms,
        "median_profile_peak_to_peak_mm": peak_to_peak,
        "noise_sigma_mm": float(residual_sigma),
        "frame_median_repeatability_sigma_mm": repeatability_sigma,
        "structure_to_noise_ratio": float(profile_rms / max(residual_sigma, 1.0e-12)),
        "structure_to_repeatability_ratio": float(
            profile_rms / max(repeatability_sigma, 1.0e-12)
        ),
        "profile_s_vs_residual_correlation": correlation,
        "heuristic": "candidate when both profile RMS and peak-to-peak exceed 3x across-frame bin-median repeatability",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else _finite_or_none(row.get(key))
                    for key in fieldnames
                }
            )


def _plot_all_profiles(
    path: Path,
    fits: list[FrameFit],
    pooled: dict[str, Any],
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 6.5))
    colours = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(fits)))
    for colour, fit in zip(colours, fits, strict=True):
        axis.scatter(
            fit.s,
            fit.frame.points_ground[:, 2],
            s=2.0,
            alpha=0.22,
            color=colour,
            rasterized=True,
            label=fit.frame.frame_id,
        )
    s_min = min(float(np.min(fit.s)) for fit in fits)
    s_max = max(float(np.max(fit.s)) for fit in fits)
    grid = np.linspace(s_min, s_max, 300)
    axis.plot(
        grid,
        pooled["slope"] * grid + pooled["intercept"],
        color="black",
        linewidth=2.0,
        label="frame-balanced pooled fit",
    )
    axis.set_title("C1 reconstructed ground points with one shared S definition")
    axis.set_xlabel("S = (XY - origin) dot direction (mm)")
    axis.set_ylabel("Zg (mm)")
    axis.grid(True, alpha=0.25)
    axis.text(
        0.01,
        0.99,
        f"origin=({origin_xy[0]:.3f}, {origin_xy[1]:.3f}) mm\n"
        f"direction=({direction_xy[0]:.6f}, {direction_xy[1]:.6f})",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
    )
    axis.legend(loc="best", ncol=2, fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_residuals(path: Path, fits: list[FrameFit], binned: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(11, 6.5))
    colours = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(fits)))
    for colour, fit in zip(colours, fits, strict=True):
        axis.scatter(
            fit.s,
            fit.residual,
            s=2.0,
            alpha=0.20,
            color=colour,
            rasterized=True,
        )
    covered = [
        row
        for row in binned
        if row["detrended_residual_median_mm"] is not None
    ]
    if covered:
        s = np.asarray([row["s_center_mm"] for row in covered])
        median = np.asarray(
            [row["detrended_residual_median_mm"] for row in covered]
        )
        spread = np.asarray(
            [row["detrended_residual_std_mm"] or 0.0 for row in covered]
        )
        axis.plot(s, median, color="black", linewidth=2.0, label="frame-balanced bin median")
        axis.fill_between(s, median - spread, median + spread, color="black", alpha=0.10)
    axis.axhline(0.0, color="0.25", linewidth=1.0, linestyle="--")
    axis.set_title("Per-frame detrended ground residual versus shared S")
    axis.set_xlabel("Shared S (mm)")
    axis.set_ylabel("r = Zg - (a*S + b) (mm)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ab_stability(path: Path, fits: list[FrameFit]) -> None:
    frame_labels = [fit.frame.frame_id for fit in fits]
    slopes = np.asarray([fit.slope for fit in fits])
    intercepts = np.asarray([fit.intercept for fit in fits])
    rmses = np.asarray([_metric_values(fit.residual)[0] for fit in fits])
    p95s = np.asarray([_metric_values(fit.residual)[1] for fit in fits])
    points = np.asarray([fit.frame.c1_point_count for fit in fits])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(frame_labels, slopes, "o-")
    axes[0, 0].axhline(float(np.mean(slopes)), color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("a stability")
    axes[0, 0].set_ylabel("a (mm/mm)")
    axes[0, 1].plot(frame_labels, intercepts, "o-")
    axes[0, 1].axhline(float(np.mean(intercepts)), color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("b stability")
    axes[0, 1].set_ylabel("b (mm)")
    axes[1, 0].plot(frame_labels, rmses, "o-", label="RMSE")
    axes[1, 0].plot(frame_labels, p95s, "s-", label="P95 abs")
    axes[1, 0].set_title("detrended residual metrics")
    axes[1, 0].set_ylabel("error (mm)")
    axes[1, 0].legend()
    axes[1, 1].bar(frame_labels, points)
    axes[1, 1].set_title("C1-valid point count")
    axes[1, 1].set_ylabel("points")
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=45)
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_report(
    path: Path,
    summary: dict[str, Any],
    output_names: list[str],
) -> None:
    shared = summary["shared_s_definition"]
    pooled = summary["pooled_ground_profile"]
    conclusions = summary["conclusions"]
    quality = summary["input_audit"]["quality_summary"]
    lines = [
        "# Ground reference report",
        "",
        f"## GROUND_LINEAR_MODEL: {conclusions['GROUND_LINEAR_MODEL']}",
        "",
        "Model: `Zg = a*S + b`; the values below are diagnostic freeze candidates only.",
        "",
        f"- frame-balanced pooled `a`: {pooled['a_mm_per_mm']:.12g} mm/mm",
        f"- frame-balanced pooled `b`: {pooled['b_mm']:.12g} mm",
        f"- pooled RMSE / P95 / Max: {pooled['rmse_mm']:.6g} / "
        f"{pooled['p95_abs_mm']:.6g} / {pooled['max_abs_mm']:.6g} mm",
        f"- frame count / pooled point count: {summary['frame_count']} / "
        f"{pooled['point_count']}",
        "",
        "## GROUND_PROFILE_STABILITY: "
        f"{conclusions['GROUND_PROFILE_STABILITY']}",
        "",
        f"- a mean / median / std / range: {summary['a_stats']['mean']:.6g} / "
        f"{summary['a_stats']['median']:.6g} / {summary['a_stats']['std']:.6g} / "
        f"{summary['a_stats']['range']['span']:.6g} mm/mm",
        f"- b mean / median / std / range: {summary['b_stats']['mean']:.6g} / "
        f"{summary['b_stats']['median']:.6g} / {summary['b_stats']['std']:.6g} / "
        f"{summary['b_stats']['range']['span']:.6g} mm",
        f"- residual-vs-S check: {summary['spatial_structure']['status']}",
        f"- residual profile RMS / peak-to-peak: "
        f"{summary['spatial_structure']['median_profile_rms_mm']} / "
        f"{summary['spatial_structure']['median_profile_peak_to_peak_mm']} mm",
        f"- cross-frame bin-median repeatability sigma / structure ratio: "
        f"{summary['spatial_structure'].get('frame_median_repeatability_sigma_mm')} / "
        f"{summary['spatial_structure'].get('structure_to_repeatability_ratio')}.",
        "",
        "## Shared coordinate definition",
        "",
        f"- origin_xy (mm): `{shared['origin_xy']}`",
        f"- direction_xy: `{shared['direction_xy']}`",
        "- Every frame uses `S=(XY-origin_xy) dot direction_xy`; no frame-local re-centering or re-orientation.",
        "",
        "## Protocol and provenance",
        "",
        "- 20 source TIFFs were processed; each frame ran Steger once.",
        "- The same centers were passed to the C0 base and C1-enabled reconstruction calls; C1-valid points are the ground points used below.",
        "- No analytical ROI, reconstruction image ROI, black-cell interpolation, spline/LUT, or height compensation was used.",
        "- The configured Steger search rectangle is recorded as a detector search window only; it is not a post-extraction point-selection ROI.",
        f"- `enable_laser_ray_correction`: `{summary['protocol']['enable_laser_ray_correction']}`.",
        f"- Dataset quality summary: `{quality}`. The low dynamic-range warning was retained and not used to drop frames.",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- `{name}`" for name in output_names)
    lines.extend(
        [
            "",
            "The report is a diagnostic and parameter-freeze candidate; it does not modify GUI or production height-measurement behavior.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bins < 5:
        raise SystemExit("--bins must be >= 5")
    data_dir = args.data_dir.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("*.tif"), key=_natural_key)
    if len(paths) != 20:
        raise SystemExit(f"expected exactly 20 TIFF frames, found {len(paths)} in {data_dir}")
    dataset_document, metadata_by_name, dataset_manifest_path, frames_csv_path = _load_dataset_metadata(data_dir)

    app = load_app_config(config_path)
    if app.extraction_method != "steger":
        raise SystemExit(f"configured extraction method must be steger, got {app.extraction_method!r}")
    if not app.reconstruction.enable_laser_ray_correction:
        raise SystemExit("measure_tool_daheng_0811.yaml must have enable_laser_ray_correction=true")
    if app.reconstruction.image_roi_polygon is not None:
        raise SystemExit("analytical reconstruction image_roi_polygon must be null for this run")
    if app.calibration.manifest is None:
        raise SystemExit("Daheng config must declare calibration.manifest")

    package = load_calibration_package(app.calibration.manifest)
    extraction_params = create_extraction_params(
        app.extraction_method,
        app.extraction_options,
    )
    params_c0 = replace(app.reconstruction, enable_laser_ray_correction=False)
    params_c1 = app.reconstruction

    frames: list[FrameRun] = []
    for index, path in enumerate(paths, start=1):
        metadata = metadata_by_name.get(path.name, {})
        if not metadata:
            raise SystemExit(f"missing dataset manifest metadata for {path.name}")
        frame = _read_frame(
            index,
            path,
            metadata,
            extraction_params,
            package.calibration,
            params_c0,
            params_c1,
        )
        frames.append(frame)
        print(
            f"{frame.frame_id} {path.name}: centers={len(frame.centers_uv)} "
            f"C0={frame.c0_point_count} C1={frame.c1_point_count} "
            f"extract={frame.extraction_ms:.1f}ms"
        )

    all_xy = np.concatenate([frame.points_ground[:, :2] for frame in frames])
    line_fit = _fit_line_xy(all_xy, app.measurement, "session ground XY")
    origin_xy = np.ascontiguousarray(line_fit.centre_xy, dtype=np.float64)
    direction_xy = np.ascontiguousarray(line_fit.direction_xy, dtype=np.float64)
    frame_fits = _build_frame_fits(frames, origin_xy, direction_xy, app.measurement)
    pooled = _fit_frame_balanced_pooled(frame_fits, app.measurement)
    binned = _build_binned_profile(frame_fits, pooled, args.bins)

    all_residual = np.concatenate([fit.residual for fit in frame_fits])
    pooled_rmse, pooled_p95, pooled_max = _metric_values(pooled["residual"])
    pooled["rmse_mm"] = pooled_rmse
    pooled["p95_abs_mm"] = pooled_p95
    pooled["max_abs_mm"] = pooled_max
    pooled["point_count"] = int(len(pooled["residual"]))
    pooled["inlier_count"] = int(np.count_nonzero(pooled["inlier_mask"]))
    pooled["s_span_mm"] = float(np.ptp(pooled["s"]))

    robust_residual_sigma = float(_robust_sigma(all_residual))
    spatial_structure = _spatial_structure_check(
        binned,
        robust_residual_sigma,
        len(frames),
    )
    s_min = float(np.min(pooled["s"]))
    s_max = float(np.max(pooled["s"]))
    endpoint_deviations = []
    for fit in frame_fits:
        frame_prediction = np.array(
            [fit.slope * s_min + fit.intercept, fit.slope * s_max + fit.intercept]
        )
        pooled_prediction = np.array(
            [pooled["slope"] * s_min + pooled["intercept"], pooled["slope"] * s_max + pooled["intercept"]]
        )
        endpoint_deviations.append(float(np.max(np.abs(frame_prediction - pooled_prediction))))
    max_endpoint_deviation = float(max(endpoint_deviations))
    model_ab_stable = max_endpoint_deviation <= 3.0 * max(robust_residual_sigma, 1.0e-9)
    no_clear_structure = spatial_structure["status"] == "no_clear_stable_spatial_structure"
    ground_linear_model = (
        "FREEZE_CANDIDATE"
        if model_ab_stable and no_clear_structure
        else "REVIEW_REQUIRED_SPATIAL_STRUCTURE"
        if spatial_structure["status"] == "stable_spatial_structure_candidate"
        else "REVIEW_REQUIRED"
    )
    ground_profile_stability = (
        "STABLE_CANDIDATE"
        if no_clear_structure and spatial_structure["covered_bin_count"] > 0
        else "STABLE_SPATIAL_STRUCTURE"
        if spatial_structure["status"] == "stable_spatial_structure_candidate"
        else spatial_structure["status"].upper()
    )

    frame_metric_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for fit in frame_fits:
        rmse, p95, max_abs = _metric_values(fit.residual)
        frame = fit.frame
        frame_metric_rows.append(
            {
                "frame_index": frame.index,
                "frame_id": frame.frame_id,
                "source_file": frame.path.name,
                "camera_frame_number": frame.camera_frame_number,
                "center_count": len(frame.centers_uv),
                "c0_point_count": frame.c0_point_count,
                "c1_point_count": frame.c1_point_count,
                "point_count": len(frame.points_ground),
                "fit_inlier_count": int(np.count_nonzero(fit.inlier_mask)),
                "filtered_c0_total": _filtered_total(frame.c0_filtered),
                "filtered_c1_total": _filtered_total(frame.c1_filtered),
                "a": fit.slope,
                "b": fit.intercept,
                "fit_rmse_mm": fit.fit_rmse,
                "rmse_mm": rmse,
                "p95_abs_mm": p95,
                "max_abs_mm": max_abs,
                "s_min_mm": float(np.min(fit.s)),
                "s_max_mm": float(np.max(fit.s)),
                "s_span_mm": float(np.ptp(fit.s)),
                "quality_passed": frame.quality.get("passed"),
                "quality_dynamic_range_u8": frame.quality.get("dynamic_range_u8"),
                "quality_warnings": ";".join(map(str, frame.quality.get("warnings", []))),
                "extraction_ms": frame.extraction_ms,
                "c0_reconstruction_ms": frame.c0_reconstruction_ms,
                "c1_reconstruction_ms": frame.c1_reconstruction_ms,
            }
        )
        pooled_residual = pooled["residual"][pooled["frame_index"] == frame.index]
        if len(pooled_residual) != len(fit.s):
            raise RuntimeError("pooled residual ordering does not match frame fits")
        for point_index, (pixel, point, s_value, pred, residual, inlier, pooled_r) in enumerate(
            zip(
                frame.valid_pixels_uv,
                frame.points_ground,
                fit.s,
                fit.predicted_z,
                fit.residual,
                fit.inlier_mask,
                pooled_residual,
                strict=True,
            )
        ):
            residual_rows.append(
                {
                    "frame_index": frame.index,
                    "frame_id": frame.frame_id,
                    "source_file": frame.path.name,
                    "point_index": point_index,
                    "u_px": float(pixel[0]),
                    "v_px": float(pixel[1]),
                    "Xg_mm": float(point[0]),
                    "Yg_mm": float(point[1]),
                    "Zg_mm": float(point[2]),
                    "S_mm": float(s_value),
                    "predicted_Zg_mm": float(pred),
                    "residual_mm": float(residual),
                    "fit_inlier": bool(inlier),
                    "pooled_predicted_Zg_mm": float(pooled["slope"] * s_value + pooled["intercept"]),
                    "pooled_residual_mm": float(pooled_r),
                }
            )

    input_files = []
    for path, frame in zip(paths, frames, strict=True):
        metadata = metadata_by_name[path.name]
        input_files.append(
            {
                "path": str(path),
                "sha256": frame.file_sha256,
                "size_bytes": path.stat().st_size,
                "shape": list(frame.image_shape),
                "dtype": frame.image_dtype,
                "camera_frame_number": frame.camera_frame_number,
                "quality": frame.quality,
                "manifest_sha256": metadata.get("sha256"),
            }
        )

    frame_metric_fields = list(frame_metric_rows[0])
    residual_fields = list(residual_rows[0])
    _write_csv(output_dir / "ground_frame_metrics.csv", frame_metric_rows, frame_metric_fields)
    _write_csv(output_dir / "ground_residuals.csv", residual_rows, residual_fields)
    pooled_fields = list(binned[0])
    _write_csv(output_dir / "ground_profile_pooled.csv", binned, pooled_fields)

    _plot_all_profiles(
        output_dir / "ground_profile_all_frames.png",
        frame_fits,
        pooled,
        origin_xy,
        direction_xy,
    )
    _plot_residuals(
        output_dir / "ground_detrended_residual_vs_s.png",
        frame_fits,
        binned,
    )
    _plot_ab_stability(output_dir / "ground_ab_stability.png", frame_fits)

    a_stats = _stats([fit.slope for fit in frame_fits])
    b_stats = _stats([fit.intercept for fit in frame_fits])
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frame_count": len(frames),
        "input_audit": {
            "data_dir": str(data_dir),
            "dataset_manifest": str(dataset_manifest_path),
            "frames_csv": str(frames_csv_path),
            "dataset_status": dataset_document.get("status"),
            "frames_expected": dataset_document.get("tasks", {}).get(
                "fit_001_02_laser", {}
            ).get("frames_expected"),
            "frames_captured": dataset_document.get("tasks", {}).get(
                "fit_001_02_laser", {}
            ).get("frames_captured"),
            "quality_summary": dataset_document.get("quality_summary"),
            "input_files": input_files,
            "all_images_same_shape": len({tuple(frame.image_shape) for frame in frames}) == 1,
            "all_images_same_dtype": len({frame.image_dtype for frame in frames}) == 1,
        },
        "configuration": {
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "calibration_manifest": str(app.calibration.manifest),
            "calibration_manifest_sha256": _sha256_file(app.calibration.manifest),
            "package_id": package.package_id,
            "package_manifest_sha256": package.manifest_sha256,
            "laser_model": str(app.calibration.laser_model),
            "extrinsics": str(app.calibration.extrinsics),
            "intrinsics": str(app.calibration.intrinsics),
            "laser_ray_correction": str(app.calibration.laser_ray_correction),
            "code_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "protocol": {
            "source_frames": 20,
            "steger_once_per_frame": True,
            "extraction_method": app.extraction_method,
            "extraction_options": app.extraction_options,
            "image_offset_xy": [0, 0],
            "analytical_roi_used": False,
            "reconstruction_image_roi_polygon": None,
            "configured_detector_search_roi_is_not_point_selection": True,
            "all_c1_valid_reconstructed_points_retained": True,
            "reconstruction_invalid_points_only_removed": True,
            "chessboard_black_cell_interpolation": False,
            "spline_or_lut": False,
            "height_linear_compensation": False,
            "enable_laser_ray_correction": bool(
                app.reconstruction.enable_laser_ray_correction
            ),
            "c0_base_and_c1_enabled_calls_share_centers": True,
        },
        "artifact_provenance": {
            "reused": [
                "existing Daheng application configuration",
                "existing Daheng calibration manifest and frozen C1_4k parameters",
                "existing Steger extractor, C0 quadratic reconstruction, C1 correction, and robust ground-profile kernel",
            ],
            "reused_as_target_results": [],
            "newly_computed": [
                "one Steger extraction per each of 20 TIFF frames",
                "C0 base and C1-enabled reconstruction for each shared center array",
                "one joint robust XY origin/direction and shared S coordinate",
                "20 per-frame robust Zg=a*S+b fits and residuals",
                "frame-balanced pooled linear profile, residual structure check, plots, CSVs, and report",
            ],
        },
        "shared_s_definition": {
            "origin_xy": origin_xy,
            "direction_xy": direction_xy,
            "formula": "S=(XY-origin_xy) dot direction_xy",
            "origin_source": "joint robust XY line fit over all C1-valid ground points",
            "direction_source": "joint robust XY line fit over all C1-valid ground points",
            "joint_line_fit_inlier_count_for_axis_only": int(np.count_nonzero(line_fit.inlier_mask)),
            "joint_line_fit_point_count": len(all_xy),
            "joint_line_fit_rmse_mm": float(line_fit.rmse_mm),
            "per_frame_redefinition": False,
        },
        "a_stats": a_stats,
        "b_stats": b_stats,
        "robust_residual_sigma_mm": robust_residual_sigma,
        "max_frame_model_deviation_at_global_s_endpoints_mm": max_endpoint_deviation,
        "pooled_ground_profile": {
            "method": "weighted robust linear least squares with each frame total weight=1",
            "a_mm_per_mm": pooled["slope"],
            "b_mm": pooled["intercept"],
            "rmse_mm": pooled["rmse_mm"],
            "p95_abs_mm": pooled["p95_abs_mm"],
            "max_abs_mm": pooled["max_abs_mm"],
            "point_count": pooled["point_count"],
            "inlier_count": pooled["inlier_count"],
            "s_span_mm": pooled["s_span_mm"],
        },
        "spatial_structure": spatial_structure,
        "conclusions": {
            "GROUND_LINEAR_MODEL": ground_linear_model,
            "GROUND_PROFILE_STABILITY": ground_profile_stability,
            "diagnostic_only": True,
            "parameter_freeze_candidate_only": True,
        },
        "outputs": [
            "ground_frame_metrics.csv",
            "ground_residuals.csv",
            "ground_profile_pooled.csv",
            "ground_reference_summary.json",
            "ground_profile_all_frames.png",
            "ground_detrended_residual_vs_s.png",
            "ground_ab_stability.png",
            "ground_reference_report.md",
        ],
    }
    summary_path = output_dir / "ground_reference_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _build_report(
        output_dir / "ground_reference_report.md",
        _json_ready(summary),
        summary["outputs"],
    )

    print(f"output_dir={output_dir}")
    print(f"GROUND_LINEAR_MODEL={ground_linear_model}")
    print(f"GROUND_PROFILE_STABILITY={ground_profile_stability}")
    print(
        f"shared origin={origin_xy.tolist()} direction={direction_xy.tolist()} "
        f"pooled a={pooled['slope']:.12g} b={pooled['intercept']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
