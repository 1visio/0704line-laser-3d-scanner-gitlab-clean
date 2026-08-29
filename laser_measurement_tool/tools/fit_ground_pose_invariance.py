"""Audit Ground-1 residual-profile invariance across board poses.

Ground-1 owns the coordinate definition and the 50-bin S grid.  This tool
loads those frozen values, processes the 15 new images with the existing
one-Steger/C0+C1 path, fits each frame independently, and only then builds
one frame-balanced residual profile per pose.

No new XY PCA, spline/LUT, height compensation, ROI point selection, or
cross-pose point pooling is performed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config
from calibration.manifest import load_calibration_package
from laser.backends import create_extraction_params
from measurement.height_measure import MeasurementParams
from tools.fit_ground_reference_20frames import (
    FrameFit,
    FrameRun,
    _build_frame_fits,
    _finite_or_none,
    _json_ready,
    _load_dataset_metadata,
    _metric_values,
    _natural_key,
    _read_frame,
    _sha256_file,
    _write_csv,
)


DEFAULT_DATA_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\.chessboard_v2.inprogress\fit"
)
DEFAULT_GROUND1_DIR = (
    TOOL_ROOT / "output_daheng_0811" / "ground_reference_20frames"
)
DEFAULT_OUTPUT_DIR = (
    TOOL_ROOT / "output_daheng_0811" / "ground_pose_invariance_20frames"
)
EXPECTED_POSES = ("002", "003", "004")
MIN_COVERAGE_FRACTION = 0.8
FULL_CORRELATION_THRESHOLD = 0.8
FULL_NORMALIZED_DIFFERENCE_THRESHOLD = 0.5
LOW_FREQUENCY_CORRELATION_THRESHOLD = 0.8
MOVING_WINDOW_BINS = 5


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _pose_from_metadata(metadata: dict[str, Any], path: Path) -> str:
    value = metadata.get("pose_id")
    if value is not None:
        return f"{int(value):03d}"
    match = re.search(r"laser\s+(\d+)", path.name)
    if match is None:
        raise RuntimeError(f"cannot infer pose from {path.name}")
    return f"{int(match.group(1)):03d}"


def _load_ground1(
    ground1_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    summary_path = ground1_dir / "ground_reference_summary.json"
    profile_path = ground1_dir / "ground_profile_pooled.csv"
    metrics_path = ground1_dir / "ground_frame_metrics.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile_rows = _read_csv(profile_path)
    metric_rows = _read_csv(metrics_path)
    if len(profile_rows) != 50:
        raise RuntimeError(f"Ground-1 profile must contain 50 bins, got {len(profile_rows)}")
    return summary, profile_rows, metric_rows


def _bin_specs(ground1_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ordered = sorted(ground1_rows, key=lambda row: int(row["bin_index"]))
    specs: list[dict[str, Any]] = []
    for row in ordered:
        specs.append(
            {
                "bin_index": int(row["bin_index"]),
                "s_left_mm": float(row["s_left_mm"]),
                "s_right_mm": float(row["s_right_mm"]),
                "s_center_mm": float(row["s_center_mm"]),
            }
        )
    for previous, current in zip(specs[:-1], specs[1:], strict=True):
        if not math.isclose(previous["s_right_mm"], current["s_left_mm"], abs_tol=1.0e-9):
            raise RuntimeError("Ground-1 S bins are not contiguous")
    return specs


def _ground1_profile_rows(
    ground1_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(ground1_rows, key=lambda row: int(row["bin_index"])):
        rows.append(
            {
                "profile_group": "ground1",
                "pose_id": "A",
                "profile_source": "Ground-1 pooled 20-frame profile",
                "bin_index": int(source["bin_index"]),
                "s_left_mm": float(source["s_left_mm"]),
                "s_right_mm": float(source["s_right_mm"]),
                "s_center_mm": float(source["s_center_mm"]),
                "frame_count": int(source["frame_count"]),
                "coverage_fraction": float(source["coverage_fraction"]),
                "point_count": int(source["point_count"]),
                "frame_balanced_s_median_mm": _float(
                    source["frame_balanced_s_median_mm"]
                ),
                "residual_mean_mm": _float(source["detrended_residual_mean_mm"]),
                "residual_median_mm": _float(
                    source["detrended_residual_median_mm"]
                ),
                "residual_std_mm": _float(source["detrended_residual_std_mm"]),
                "profile_available": True,
            }
        )
    return rows


def _pose_profile_rows(
    pose_id: str,
    fits: list[FrameFit],
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        left = spec["s_left_mm"]
        right = spec["s_right_mm"]
        per_frame_s: list[float] = []
        per_frame_residual: list[float] = []
        point_count = 0
        for fit in fits:
            in_bin = (fit.s >= left) & (
                fit.s <= right
                if spec["bin_index"] == specs[-1]["bin_index"]
                else fit.s < right
            )
            count = int(np.count_nonzero(in_bin))
            if count == 0:
                continue
            per_frame_s.append(float(np.median(fit.s[in_bin])))
            per_frame_residual.append(float(np.median(fit.residual[in_bin])))
            point_count += count
        rows.append(
            {
                "profile_group": f"pose{pose_id}",
                "pose_id": pose_id,
                "profile_source": f"5-frame-balanced residual profile for pose {pose_id}",
                "bin_index": spec["bin_index"],
                "s_left_mm": left,
                "s_right_mm": right,
                "s_center_mm": spec["s_center_mm"],
                "frame_count": len(per_frame_residual),
                "coverage_fraction": len(per_frame_residual) / len(fits),
                "point_count": point_count,
                "frame_balanced_s_median_mm": (
                    float(np.mean(per_frame_s)) if per_frame_s else None
                ),
                "residual_mean_mm": (
                    float(np.mean(per_frame_residual)) if per_frame_residual else None
                ),
                "residual_median_mm": (
                    float(np.median(per_frame_residual))
                    if per_frame_residual
                    else None
                ),
                "residual_std_mm": (
                    float(np.std(per_frame_residual)) if per_frame_residual else None
                ),
                "profile_available": bool(per_frame_residual),
            }
        )
    return rows


def _consensus_profile_rows(
    pose_profile_rows: dict[str, list[dict[str, Any]]],
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        values: list[float] = []
        s_values: list[float] = []
        point_count = 0
        for pose_id in EXPECTED_POSES:
            source = pose_profile_rows[pose_id][spec["bin_index"]]
            if (
                source["profile_available"]
                and source["coverage_fraction"] >= MIN_COVERAGE_FRACTION
            ):
                values.append(float(source["residual_median_mm"]))
                if source["frame_balanced_s_median_mm"] is not None:
                    s_values.append(float(source["frame_balanced_s_median_mm"]))
                point_count += int(source["point_count"])
        rows.append(
            {
                "profile_group": "pose_consensus",
                "pose_id": "002|003|004",
                "profile_source": "median across pose residual profiles",
                "bin_index": spec["bin_index"],
                "s_left_mm": spec["s_left_mm"],
                "s_right_mm": spec["s_right_mm"],
                "s_center_mm": spec["s_center_mm"],
                "frame_count": len(values),
                "coverage_fraction": len(values) / len(EXPECTED_POSES),
                "point_count": point_count,
                "frame_balanced_s_median_mm": (
                    float(np.mean(s_values)) if s_values else None
                ),
                "residual_mean_mm": float(np.mean(values)) if values else None,
                "residual_median_mm": float(np.median(values)) if values else None,
                "residual_std_mm": float(np.std(values)) if values else None,
                "profile_available": bool(values),
            }
        )
    return rows


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    width = min(window, len(values))
    if width % 2 == 0:
        width -= 1
    if width < 3:
        return values.copy()
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _compare_profiles(
    reference_group: str,
    target_group: str,
    reference_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    reference_metric_rows: list[dict[str, Any]],
    target_metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ref_by_bin = {int(row["bin_index"]): row for row in reference_rows}
    target_by_bin = {int(row["bin_index"]): row for row in target_rows}
    common = []
    for bin_index in sorted(set(ref_by_bin) & set(target_by_bin)):
        ref = ref_by_bin[bin_index]
        target = target_by_bin[bin_index]
        if (
            ref["profile_available"]
            and target["profile_available"]
            and float(ref["coverage_fraction"]) >= MIN_COVERAGE_FRACTION
            and float(target["coverage_fraction"]) >= MIN_COVERAGE_FRACTION
            and ref["residual_median_mm"] is not None
            and target["residual_median_mm"] is not None
        ):
            common.append((bin_index, ref, target))

    result: dict[str, Any] = {
        "reference_group": reference_group,
        "target_group": target_group,
        "comparison_type": "ground1_vs_pose"
        if reference_group == "ground1"
        else "pose_pair",
        "coverage_requirement_fraction": MIN_COVERAGE_FRACTION,
        "common_bin_count": len(common),
        "common_s_min_mm": None,
        "common_s_max_mm": None,
        "common_s_span_mm": None,
        "profile_correlation": None,
        "low_frequency_correlation": None,
        "high_frequency_correlation": None,
        "profile_rmse_difference_mm": None,
        "profile_median_abs_difference_mm": None,
        "profile_rms_reference_mm": None,
        "profile_rms_target_mm": None,
        "normalized_profile_difference": None,
        "reference_peak_s_mm": None,
        "target_peak_s_mm": None,
        "peak_s_offset_mm": None,
        "reference_valley_s_mm": None,
        "target_valley_s_mm": None,
        "valley_s_offset_mm": None,
        "reference_frame_rmse_median_mm": None,
        "target_frame_rmse_median_mm": None,
        "detrended_rmse_difference_mm": None,
        "status": "insufficient_common_coverage",
    }
    if len(common) < 3:
        return result

    s = np.asarray([item[1]["s_center_mm"] for item in common], dtype=np.float64)
    ref_values = np.asarray(
        [item[1]["residual_median_mm"] for item in common], dtype=np.float64
    )
    target_values = np.asarray(
        [item[2]["residual_median_mm"] for item in common], dtype=np.float64
    )
    difference = target_values - ref_values
    reference_low = _moving_average(ref_values, MOVING_WINDOW_BINS)
    target_low = _moving_average(target_values, MOVING_WINDOW_BINS)
    reference_high = ref_values - reference_low
    target_high = target_values - target_low
    ref_peak = int(np.argmax(ref_values))
    target_peak = int(np.argmax(target_values))
    ref_valley = int(np.argmin(ref_values))
    target_valley = int(np.argmin(target_values))
    reference_rmse = [float(row["rmse_mm"]) for row in reference_metric_rows]
    target_rmse = [float(row["rmse_mm"]) for row in target_metric_rows]
    profile_rmse = float(np.sqrt(np.mean(difference**2)))
    ref_rms = float(np.sqrt(np.mean(ref_values**2)))
    target_rms = float(np.sqrt(np.mean(target_values**2)))
    result.update(
        {
            "common_s_min_mm": float(np.min(s)),
            "common_s_max_mm": float(np.max(s)),
            "common_s_span_mm": float(np.ptp(s)),
            "profile_correlation": _corr(ref_values, target_values),
            "low_frequency_correlation": _corr(reference_low, target_low),
            "high_frequency_correlation": _corr(reference_high, target_high),
            "profile_rmse_difference_mm": profile_rmse,
            "profile_median_abs_difference_mm": float(np.median(np.abs(difference))),
            "profile_rms_reference_mm": ref_rms,
            "profile_rms_target_mm": target_rms,
            "normalized_profile_difference": profile_rmse / max(ref_rms, target_rms, 1.0e-12),
            "reference_peak_s_mm": float(s[ref_peak]),
            "target_peak_s_mm": float(s[target_peak]),
            "peak_s_offset_mm": float(s[target_peak] - s[ref_peak]),
            "reference_valley_s_mm": float(s[ref_valley]),
            "target_valley_s_mm": float(s[target_valley]),
            "valley_s_offset_mm": float(s[target_valley] - s[ref_valley]),
            "reference_frame_rmse_median_mm": float(np.median(reference_rmse)),
            "target_frame_rmse_median_mm": float(np.median(target_rmse)),
            "detrended_rmse_difference_mm": float(
                np.median(target_rmse) - np.median(reference_rmse)
            ),
            "status": "ok",
        }
    )
    return result


def _classify(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    pose_comparisons = [
        row for row in comparisons if row["comparison_type"] == "ground1_vs_pose"
    ]
    valid = [row for row in pose_comparisons if row["status"] == "ok"]
    if len(valid) != len(EXPECTED_POSES):
        return {
            "GROUND_POSE_INVARIANCE": "INSUFFICIENT_COMMON_COVERAGE",
            "CAMERA_SPACE_STRUCTURE": "UNDETERMINED",
            "BOARD_DEPENDENT_STRUCTURE": "UNDETERMINED",
            "MIXED_STRUCTURE": "UNDETERMINED",
            "ground3_recommendation": "BLOCK_UNTIL_COVERAGE_IS_SUFFICIENT",
            "valid_pose_comparison_count": len(valid),
            "thresholds": {
                "full_correlation_min": FULL_CORRELATION_THRESHOLD,
                "full_normalized_difference_max": FULL_NORMALIZED_DIFFERENCE_THRESHOLD,
                "low_frequency_correlation_min": LOW_FREQUENCY_CORRELATION_THRESHOLD,
            },
        }

    full_flags = [
        row["profile_correlation"] is not None
        and row["profile_correlation"] >= FULL_CORRELATION_THRESHOLD
        and row["normalized_profile_difference"] is not None
        and row["normalized_profile_difference"] <= FULL_NORMALIZED_DIFFERENCE_THRESHOLD
        for row in valid
    ]
    low_flags = [
        row["low_frequency_correlation"] is not None
        and row["low_frequency_correlation"] >= LOW_FREQUENCY_CORRELATION_THRESHOLD
        for row in valid
    ]
    if all(full_flags):
        invariance = "SUPPORTED"
        camera = "SUPPORTED"
        board = "NOT_SUPPORTED"
        mixed = "NOT_SELECTED"
        recommendation = "ALLOW"
    elif all(low_flags):
        invariance = "MIXED_STRUCTURE"
        camera = "LOW_FREQUENCY_ONLY"
        board = "POSSIBLE_HIGH_FREQUENCY_POSE_DEPENDENCE"
        mixed = "SUPPORTED"
        recommendation = "CONDITIONAL_ALLOW_LOW_FREQUENCY_ONLY"
    else:
        invariance = "NOT_SUPPORTED_BOARD_DEPENDENT"
        camera = "NOT_SUPPORTED"
        board = "SUPPORTED"
        mixed = "NOT_SELECTED"
        recommendation = "BLOCK"
    return {
        "GROUND_POSE_INVARIANCE": invariance,
        "CAMERA_SPACE_STRUCTURE": camera,
        "BOARD_DEPENDENT_STRUCTURE": board,
        "MIXED_STRUCTURE": mixed,
        "ground3_recommendation": recommendation,
        "valid_pose_comparison_count": len(valid),
        "full_profile_alignment_flags": dict(
            zip([row["target_group"] for row in valid], full_flags, strict=True)
        ),
        "low_frequency_alignment_flags": dict(
            zip([row["target_group"] for row in valid], low_flags, strict=True)
        ),
        "thresholds": {
            "full_correlation_min": FULL_CORRELATION_THRESHOLD,
            "full_normalized_difference_max": FULL_NORMALIZED_DIFFERENCE_THRESHOLD,
            "low_frequency_correlation_min": LOW_FREQUENCY_CORRELATION_THRESHOLD,
            "moving_average_window_bins": MOVING_WINDOW_BINS,
        },
    }


def _profile_lookup(rows: list[dict[str, Any]], group: str) -> dict[int, dict[str, Any]]:
    return {
        int(row["bin_index"]): row
        for row in rows
        if row["profile_group"] == group
    }


def _plot_overlay(path: Path, profile_rows: list[dict[str, Any]]) -> None:
    groups = ["ground1", "pose002", "pose003", "pose004"]
    labels = {
        "ground1": "Ground-1 R_A(S)",
        "pose002": "pose 002",
        "pose003": "pose 003",
        "pose004": "pose 004",
    }
    colours = {"ground1": "black", "pose002": "tab:blue", "pose003": "tab:orange", "pose004": "tab:green"}
    fig, axis = plt.subplots(figsize=(11, 6.5))
    for group in groups:
        rows = [row for row in profile_rows if row["profile_group"] == group and row["profile_available"]]
        rows.sort(key=lambda row: row["bin_index"])
        if not rows:
            continue
        axis.plot(
            [row["s_center_mm"] for row in rows],
            [row["residual_median_mm"] for row in rows],
            marker="o" if group != "ground1" else None,
            markersize=2.5,
            linewidth=1.8 if group == "ground1" else 1.2,
            color=colours[group],
            label=labels[group],
        )
    axis.axhline(0.0, color="0.3", linestyle="--", linewidth=0.8)
    axis.set_title("Ground residual profiles on the frozen Ground-1 camera-S axis")
    axis.set_xlabel("Frozen S (mm)")
    axis.set_ylabel("Frame-balanced residual median (mm)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pairwise(
    path: Path,
    comparisons: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> None:
    lookups = {
        group: _profile_lookup(profile_rows, group)
        for group in ("ground1", "pose002", "pose003", "pose004")
    }
    fig, axis = plt.subplots(figsize=(11, 6.5))
    for comparison in comparisons:
        if comparison["status"] != "ok":
            continue
        ref = lookups[comparison["reference_group"]]
        target = lookups[comparison["target_group"]]
        points = []
        for bin_index in sorted(set(ref) & set(target)):
            if (
                ref[bin_index]["profile_available"]
                and target[bin_index]["profile_available"]
                and ref[bin_index]["coverage_fraction"] >= MIN_COVERAGE_FRACTION
                and target[bin_index]["coverage_fraction"] >= MIN_COVERAGE_FRACTION
            ):
                points.append(
                    (
                        ref[bin_index]["s_center_mm"],
                        target[bin_index]["residual_median_mm"]
                        - ref[bin_index]["residual_median_mm"],
                    )
                )
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                linewidth=1.4,
                label=f"{comparison['target_group']} - {comparison['reference_group']}",
            )
    axis.axhline(0.0, color="0.3", linestyle="--", linewidth=0.8)
    axis.set_title("Pairwise residual-profile difference on common frozen S bins")
    axis.set_xlabel("Frozen S (mm)")
    axis.set_ylabel("Target residual - reference residual (mm)")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_consensus(path: Path, profile_rows: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(11, 6.5))
    for group, colour in (
        ("pose002", "tab:blue"),
        ("pose003", "tab:orange"),
        ("pose004", "tab:green"),
    ):
        rows = [row for row in profile_rows if row["profile_group"] == group and row["profile_available"]]
        rows.sort(key=lambda row: row["bin_index"])
        axis.plot(
            [row["s_center_mm"] for row in rows],
            [row["residual_median_mm"] for row in rows],
            color=colour,
            alpha=0.55,
            linewidth=1.0,
            label=group,
        )
    consensus = [
        row for row in profile_rows
        if row["profile_group"] == "pose_consensus" and row["profile_available"]
    ]
    consensus.sort(key=lambda row: row["bin_index"])
    if consensus:
        axis.plot(
            [row["s_center_mm"] for row in consensus],
            [row["residual_median_mm"] for row in consensus],
            color="black",
            linewidth=2.2,
            label="pose consensus median",
        )
    ground1 = [row for row in profile_rows if row["profile_group"] == "ground1"]
    ground1.sort(key=lambda row: row["bin_index"])
    axis.plot(
        [row["s_center_mm"] for row in ground1],
        [row["residual_median_mm"] for row in ground1],
        color="0.4",
        linestyle="--",
        linewidth=1.5,
        label="Ground-1 R_A(S)",
    )
    axis.axhline(0.0, color="0.3", linestyle=":", linewidth=0.8)
    axis.set_title("Cross-pose consensus residual profile")
    axis.set_xlabel("Frozen S (mm)")
    axis.set_ylabel("Residual median (mm)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_report(
    path: Path,
    summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    output_names: list[str],
) -> None:
    classification = summary["classification"]
    lines = [
        "# Ground-2 pose invariance report",
        "",
        f"## GROUND_POSE_INVARIANCE: {classification['GROUND_POSE_INVARIANCE']}",
        "",
        f"- CAMERA_SPACE_STRUCTURE: `{classification['CAMERA_SPACE_STRUCTURE']}`",
        f"- BOARD_DEPENDENT_STRUCTURE: `{classification['BOARD_DEPENDENT_STRUCTURE']}`",
        f"- MIXED_STRUCTURE: `{classification['MIXED_STRUCTURE']}`",
        f"- Ground-3 recommendation: `{classification['ground3_recommendation']}`",
        "",
        "## Frozen Ground-1 coordinate definition",
        "",
        f"- origin_xy (mm): `{summary['ground1_reuse']['frozen_origin_xy']}`",
        f"- direction_xy: `{summary['ground1_reuse']['frozen_direction_xy']}`",
        "- S formula: `S=(XY-origin_xy) dot direction_xy`.",
        f"- Ground-1 S coverage: [{summary['ground1_reuse']['s_min_mm']:.6g}, {summary['ground1_reuse']['s_max_mm']:.6g}] mm; 50 bins.",
        "- No pose-specific PCA, re-centering, S redefinition, cross-pose point pooling, interpolation, spline/LUT, or height compensation was used.",
        "",
        "## Ground-1 versus each new pose",
        "",
        "| pose | common S bins | correlation | profile RMSE difference (mm) | median abs difference (mm) | peak offset (mm) | valley offset (mm) | frame RMSE difference (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["comparison_type"] != "ground1_vs_pose":
            continue
        values = [
            row["target_group"],
            row["common_bin_count"],
            _report_number(row["profile_correlation"]),
            _report_number(row["profile_rmse_difference_mm"]),
            _report_number(row["profile_median_abs_difference_mm"]),
            _report_number(row["peak_s_offset_mm"]),
            _report_number(row["valley_s_offset_mm"]),
            _report_number(row["detrended_rmse_difference_mm"]),
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines.extend(
        [
            "",
            "## New-data audit",
            "",
            f"- 15 TIFFs: 002/003/004 each 5 frames; all retained in the report.",
            f"- quality summary: `{summary['new_data_audit']['quality_summary']}`; no frame was dropped.",
            f"- exposure by pose: `{summary['new_data_audit']['exposure_us_by_pose']}` µs; this differs from the online config exposure of 2000 µs and is recorded as a protocol risk.",
            "- `enable_laser_ray_correction=true`; existing frozen C1 parameters were reused and C0/C1 were not refit.",
            "- The configured Steger search rectangle is detector search configuration only; no analytical/output ROI was applied.",
            "",
            "## Ground-3 gate",
            "",
            "The gate is based on raw binned residual profiles on the frozen S axis. It does not authorize fitting or deploying a new correction.",
            f"- thresholds: full correlation >= {FULL_CORRELATION_THRESHOLD}, normalized profile difference <= {FULL_NORMALIZED_DIFFERENCE_THRESHOLD}; low-frequency correlation >= {LOW_FREQUENCY_CORRELATION_THRESHOLD} using a {MOVING_WINDOW_BINS}-bin diagnostic moving average.",
            "",
            "## Outputs",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in output_names)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _report_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ground1-dir", type=Path, default=DEFAULT_GROUND1_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    ground1_dir = args.ground1_dir.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ground1_summary, ground1_source_rows, ground1_metric_rows = _load_ground1(ground1_dir)
    frozen_origin = np.asarray(
        ground1_summary["shared_s_definition"]["origin_xy"], dtype=np.float64
    )
    frozen_direction = np.asarray(
        ground1_summary["shared_s_definition"]["direction_xy"], dtype=np.float64
    )
    if frozen_origin.shape != (2,) or frozen_direction.shape != (2,):
        raise SystemExit("Ground-1 frozen origin/direction must both have shape (2,)")
    if not math.isclose(float(np.linalg.norm(frozen_direction)), 1.0, rel_tol=1.0e-9):
        raise SystemExit("Ground-1 frozen direction is not normalized")
    bin_specs = _bin_specs(ground1_source_rows)

    dataset_document, metadata_by_name, dataset_manifest_path, frames_csv_path = _load_dataset_metadata(data_dir)
    paths = sorted(data_dir.glob("*.tif"), key=_natural_key)
    if len(paths) != 15:
        raise SystemExit(f"expected exactly 15 TIFF frames, found {len(paths)} in {data_dir}")
    pose_by_name: dict[str, str] = {}
    for path in paths:
        metadata = metadata_by_name.get(path.name)
        if metadata is None:
            raise SystemExit(f"missing manifest metadata for {path.name}")
        pose_by_name[path.name] = _pose_from_metadata(metadata, path)
    if set(pose_by_name.values()) != set(EXPECTED_POSES):
        raise SystemExit(f"expected poses {EXPECTED_POSES}, got {sorted(set(pose_by_name.values()))}")
    counts = {pose_id: sum(pose == pose_id for pose in pose_by_name.values()) for pose_id in EXPECTED_POSES}
    if counts != {pose_id: 5 for pose_id in EXPECTED_POSES}:
        raise SystemExit(f"each pose must have 5 frames, got {counts}")

    app = load_app_config(config_path)
    if app.extraction_method != "steger":
        raise SystemExit(f"expected steger extraction, got {app.extraction_method!r}")
    if not app.reconstruction.enable_laser_ray_correction:
        raise SystemExit("enable_laser_ray_correction must be true")
    if app.reconstruction.image_roi_polygon is not None:
        raise SystemExit("analytical image_roi_polygon must be null")
    if app.calibration.manifest is None:
        raise SystemExit("Daheng config must declare calibration.manifest")
    package = load_calibration_package(app.calibration.manifest)
    extraction_params = create_extraction_params(app.extraction_method, app.extraction_options)
    params_c0 = replace(app.reconstruction, enable_laser_ray_correction=False)
    params_c1 = app.reconstruction

    frames: list[FrameRun] = []
    for index, path in enumerate(paths, start=1):
        frame = _read_frame(
            index,
            path,
            metadata_by_name[path.name],
            extraction_params,
            package.calibration,
            params_c0,
            params_c1,
        )
        frames.append(frame)
        print(
            f"{frame.frame_id} pose{pose_by_name[path.name]} {path.name}: "
            f"centers={len(frame.centers_uv)} C0={frame.c0_point_count} "
            f"C1={frame.c1_point_count}"
        )

    frames_by_pose = {
        pose_id: [frame for frame in frames if pose_by_name[frame.path.name] == pose_id]
        for pose_id in EXPECTED_POSES
    }
    fits_by_pose: dict[str, list[FrameFit]] = {}
    for pose_id in EXPECTED_POSES:
        # This is the only fit call for each pose's frames.  It receives the
        # Ground-1 arrays directly and never invokes a new XY line fit/PCA.
        fits_by_pose[pose_id] = _build_frame_fits(
            frames_by_pose[pose_id],
            frozen_origin,
            frozen_direction,
            app.measurement,
        )

    profile_rows = _ground1_profile_rows(ground1_source_rows)
    pose_profile_rows: dict[str, list[dict[str, Any]]] = {}
    pose_metric_rows: list[dict[str, Any]] = []
    for pose_id in EXPECTED_POSES:
        pose_fits = fits_by_pose[pose_id]
        pose_profile_rows[pose_id] = _pose_profile_rows(pose_id, pose_fits, bin_specs)
        for fit in pose_fits:
            frame = fit.frame
            rmse, p95, max_abs = _metric_values(fit.residual)
            pose_metric_rows.append(
                {
                    "frame_index": frame.index,
                    "frame_id": frame.frame_id,
                    "pose_id": pose_id,
                    "source_file": frame.path.name,
                    "camera_frame_number": frame.camera_frame_number,
                    "center_count": len(frame.centers_uv),
                    "c0_point_count": frame.c0_point_count,
                    "c1_point_count": frame.c1_point_count,
                    "point_count": len(frame.points_ground),
                    "filtered_c0_total": sum(frame.c0_filtered.values()),
                    "filtered_c1_total": sum(frame.c1_filtered.values()),
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

    consensus_rows = _consensus_profile_rows(pose_profile_rows, bin_specs)
    profile_rows.extend(
        row for pose_id in EXPECTED_POSES for row in pose_profile_rows[pose_id]
    )
    profile_rows.extend(consensus_rows)
    profile_rows_by_group = {
        group: [row for row in profile_rows if row["profile_group"] == group]
        for group in ("ground1", "pose002", "pose003", "pose004")
    }
    comparisons: list[dict[str, Any]] = []
    for pose_id in EXPECTED_POSES:
        comparisons.append(
            _compare_profiles(
                "ground1",
                f"pose{pose_id}",
                profile_rows_by_group["ground1"],
                profile_rows_by_group[f"pose{pose_id}"],
                ground1_metric_rows,
                [row for row in pose_metric_rows if row["pose_id"] == pose_id],
            )
        )
    for index, reference_pose in enumerate(EXPECTED_POSES):
        for target_pose in EXPECTED_POSES[index + 1 :]:
            comparisons.append(
                _compare_profiles(
                    f"pose{reference_pose}",
                    f"pose{target_pose}",
                    profile_rows_by_group[f"pose{reference_pose}"],
                    profile_rows_by_group[f"pose{target_pose}"],
                    [row for row in pose_metric_rows if row["pose_id"] == reference_pose],
                    [row for row in pose_metric_rows if row["pose_id"] == target_pose],
                )
            )

    classification = _classify(comparisons)
    new_input_files = []
    for frame in frames:
        metadata = metadata_by_name[frame.path.name]
        new_input_files.append(
            {
                "path": str(frame.path),
                "pose_id": pose_by_name[frame.path.name],
                "sha256": frame.file_sha256,
                "manifest_sha256": metadata.get("sha256"),
                "camera_frame_number": frame.camera_frame_number,
                "shape": list(frame.image_shape),
                "dtype": frame.image_dtype,
                "quality": frame.quality,
            }
        )

    exposure_by_pose: dict[str, list[float]] = {pose_id: [] for pose_id in EXPECTED_POSES}
    for path in paths:
        metadata = metadata_by_name[path.name]
        task_id = str(metadata.get("task_id", ""))
        for task in dataset_document.get("plan", {}).get("tasks", []):
            if isinstance(task, dict) and task.get("task_id") == task_id:
                camera = task.get("camera", {})
                if isinstance(camera, dict) and camera.get("exposure_us") is not None:
                    exposure_by_pose[pose_by_name[path.name]].append(float(camera["exposure_us"]))
    exposure_summary = {
        pose_id: sorted(set(values)) for pose_id, values in exposure_by_pose.items()
    }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ground1_reuse": {
            "summary_path": str(ground1_dir / "ground_reference_summary.json"),
            "summary_sha256": _sha256_file(ground1_dir / "ground_reference_summary.json"),
            "profile_path": str(ground1_dir / "ground_profile_pooled.csv"),
            "profile_sha256": _sha256_file(ground1_dir / "ground_profile_pooled.csv"),
            "frozen_origin_xy": frozen_origin,
            "frozen_direction_xy": frozen_direction,
            "bin_count": len(bin_specs),
            "s_min_mm": bin_specs[0]["s_left_mm"],
            "s_max_mm": bin_specs[-1]["s_right_mm"],
            "ground1_profile_status": ground1_summary["conclusions"],
        },
        "new_data_audit": {
            "data_dir": str(data_dir),
            "dataset_manifest": str(dataset_manifest_path),
            "frames_csv": str(frames_csv_path),
            "dataset_status": dataset_document.get("status"),
            "quality_summary": dataset_document.get("quality_summary"),
            "pose_counts": counts,
            "exposure_us_by_pose": exposure_summary,
            "input_files": new_input_files,
        },
        "configuration": {
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "enable_laser_ray_correction": bool(
                app.reconstruction.enable_laser_ray_correction
            ),
            "calibration_manifest": str(app.calibration.manifest),
            "calibration_package_id": package.package_id,
            "calibration_package_manifest_sha256": package.manifest_sha256,
            "code_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "protocol": {
            "one_steger_per_frame": True,
            "same_centers_to_c0_and_c1": True,
            "all_c1_valid_points_retained": True,
            "analytical_roi_used": False,
            "reconstruction_image_roi_polygon": None,
            "black_cell_interpolation": False,
            "new_xy_pca": False,
            "new_origin_or_direction": False,
            "cross_pose_point_pooling_before_fit": False,
            "spline_or_lut": False,
            "height_linear_compensation": False,
            "fixed_s_formula": "S=(XY-Ground1_origin_xy) dot Ground1_direction_xy",
            "profile_binning": "Ground-1 frozen 50-bin edges; no extrapolation outside common coverage",
        },
        "classification": classification,
        "comparison_rows": comparisons,
        "output_files": [
            "ground_pose_frame_metrics.csv",
            "ground_pose_profiles.csv",
            "ground_pose_comparison.csv",
            "ground_pose_residual_overlay.png",
            "ground_pose_pairwise_difference.png",
            "ground_pose_consensus_profile.png",
            "ground_pose_invariance_report.md",
            "ground_pose_invariance_summary.json",
        ],
        "artifact_provenance": {
            "reused": [
                "Ground-1 frozen origin_xy/direction_xy and 50 S-bin edges",
                "Ground-1 pooled residual profile R_A(S)",
                "existing Daheng configuration, calibration package, Steger, C0/C1 reconstruction, and robust Zg=a*S+b kernel",
            ],
            "reused_as_new_target_results": [],
            "newly_computed": [
                "15 new frames: one Steger plus C0/C1 reconstruction each",
                "15 fixed-axis frame fits grouped as pose 002/003/004",
                "5-frame-balanced residual profile for each pose",
                "common-S profile comparisons, pairwise differences, consensus curve, and attribution report",
            ],
        },
    }

    frame_fields = list(pose_metric_rows[0])
    profile_fields = list(profile_rows[0])
    comparison_fields = list(comparisons[0])
    _write_csv(output_dir / "ground_pose_frame_metrics.csv", pose_metric_rows, frame_fields)
    _write_csv(output_dir / "ground_pose_profiles.csv", profile_rows, profile_fields)
    _write_csv(output_dir / "ground_pose_comparison.csv", comparisons, comparison_fields)
    _plot_overlay(output_dir / "ground_pose_residual_overlay.png", profile_rows)
    _plot_pairwise(
        output_dir / "ground_pose_pairwise_difference.png",
        comparisons,
        profile_rows,
    )
    _plot_consensus(output_dir / "ground_pose_consensus_profile.png", profile_rows)
    summary_path = output_dir / "ground_pose_invariance_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _build_report(
        output_dir / "ground_pose_invariance_report.md",
        _json_ready(summary),
        comparisons,
        summary["output_files"],
    )

    print(f"output_dir={output_dir}")
    print(f"GROUND_POSE_INVARIANCE={classification['GROUND_POSE_INVARIANCE']}")
    print(f"CAMERA_SPACE_STRUCTURE={classification['CAMERA_SPACE_STRUCTURE']}")
    print(f"BOARD_DEPENDENT_STRUCTURE={classification['BOARD_DEPENDENT_STRUCTURE']}")
    print(f"GROUND3={classification['ground3_recommendation']}")
    for row in comparisons:
        if row["comparison_type"] == "ground1_vs_pose":
            print(
                f"{row['target_group']}: n={row['common_bin_count']} "
                f"corr={row['profile_correlation']} "
                f"profile_rmse_diff={row['profile_rmse_difference_mm']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
