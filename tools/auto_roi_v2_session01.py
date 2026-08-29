#!/usr/bin/env python3
"""Generate geometry-only Auto ROI V2 candidates for Session01.

This tool deliberately stops at automatic review candidates.  It reuses the
immutable A-13A Frozen Steger and median-centerline artifacts, does not run
Steger, does not reconstruct 3-D quantities, and never writes a frozen/manual
registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze"
DEFAULT_DATA_ROOT = Path(
    r"D:\Docs\linelascan\calibration_tool\projects\daheng\outputs\0822\session01"
)
FULL_SENSOR_HEIGHT = 3000
PNG_WIDTH = 480
HEIGHT_LABELS = ("h10", "h20", "h30")
POSITION_IDS = tuple(f"p{index:02d}" for index in range(1, 11))
REPEAT_COUNT = 20


PARAMETERS: dict[str, Any] = {
    "schema_version": 2,
    "algorithm": "ground_object_top_ground_interval_v2",
    "geometry_only": True,
    "forbidden_inputs": [
        "nominal_or_certified_height",
        "height_shadow.csv",
        "C0_C1_XYZ",
        "SessionGround_XYZ",
        "Base_H1_HB2_result",
        "residual_or_error",
        "q1_q2",
        "A13B_outcome",
    ],
    "profile": {
        "smoothing_sigma_px": 5.0,
        "derivative_smoothing_sigma_px": 2.0,
        "edge_peak_distance_px": 15,
        "edge_prominence_min_px": 0.25,
        "edge_prominence_noise_multiplier": 8.0,
        "edge_local_support_radius_px": 10,
        "edge_local_support_min_fraction": 0.35,
        "max_interp_gap_px": 35,
    },
    "object_interval": {
        "edge_pair_min_width_px": 50.0,
        "edge_pair_max_width_px": 180.0,
        "transition_exclusion_margin_px": 15,
        "height_interior_min_width_px": 30.0,
        "height_interior_max_width_px": 160.0,
        "minimum_step_amplitude_px": 1.0,
    },
    "stable_segment": {
        "maximum_abs_slope_px_per_v": 0.08,
        "maximum_roughness_px": 0.75,
        "minimum_valid_fraction": 0.70,
        "maximum_missing_run_px": 35,
    },
    "baseline": {
        "safety_gap_px": 20,
        "requested_width_px": 180,
        "minimum_available_width_px": 20,
        "minimum_points_per_repeat": 20,
        "minimum_support_fraction": 0.25,
    },
    "height_support": {
        "minimum_points_per_repeat": 20,
        "minimum_support_fraction": 0.50,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_v1_snapshot(output_dir: Path) -> dict[str, dict[str, Any]]:
    candidates_path = output_dir / "session01_roi_candidates.json"
    registry_path = output_dir / "session01_roi_registry_manual.json"
    candidates_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    candidate_rows = {
        row["condition_id"]: row for row in candidates_payload.get("candidates", [])
    }
    registry_rows = {
        row["condition_id"]: row for row in registry_payload.get("entries", [])
    }
    result: dict[str, dict[str, Any]] = {}
    for height_label in HEIGHT_LABELS:
        for position_id in POSITION_IDS:
            condition_id = f"{height_label}_{position_id}"
            row = candidate_rows.get(condition_id, {})
            entry = registry_rows.get(condition_id, {})
            selected = row.get("selected_candidate", {})
            height_range = row.get("roi_ranges", {}).get("height")
            if height_range is None:
                height_range = entry.get("height_v_range")
            baseline_ranges = row.get("roi_ranges", {})
            if baseline_ranges:
                baseline_ranges = [
                    baseline_ranges.get("baseline_before"),
                    baseline_ranges.get("baseline_after"),
                ]
            else:
                baseline_ranges = entry.get("baseline_v_ranges")
            center = selected.get("v_center_px", entry.get("height_roi_center_v"))
            result[condition_id] = {
                "v_center_px": center,
                "height_v_range": height_range,
                "baseline_v_ranges": baseline_ranges,
                "height_width_px": (
                    int(height_range[1] - height_range[0] + 1)
                    if height_range
                    else None
                ),
                "candidate_rank": row.get("selected_candidate_rank"),
                "candidate_prominence_px": selected.get("prominence_px"),
                "candidate_support_width_px": selected.get("support_width_px"),
                "registry_review_status": entry.get("review_status"),
            }
    return result


def load_cache(
    output_dir: Path,
    data_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = output_dir / "session01_steger_centers.npz"
    manifest_path = output_dir / "session01_steger_centers_manifest.json"
    with np.load(npz_path, allow_pickle=False) as cached:
        centers_full = np.asarray(cached["centers_full"])
        frame_offsets = np.asarray(cached["frame_offsets"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    errors: list[str] = []
    if not manifest.get("one_steger_per_frame"):
        errors.append("manifest.one_steger_per_frame is not true")
    if centers_full.ndim != 2 or centers_full.shape[1] != 2:
        errors.append(f"centers_full shape is {centers_full.shape}, expected (N,2)")
    if frame_offsets.ndim != 1 or len(frame_offsets) != len(frames) + 1:
        errors.append("frame_offsets is not aligned with manifest frames")
    if len(frames) != 600:
        errors.append(f"manifest frame count is {len(frames)}, expected 600")
    if len(frame_offsets) and int(frame_offsets[-1]) != len(centers_full):
        errors.append("frame_offsets[-1] does not equal centers_full length")
    if errors:
        raise RuntimeError("Frozen cache audit failed: " + "; ".join(errors))

    centers_by_key: dict[str, np.ndarray] = {}
    duplicate_keys: list[str] = []
    source_matches = 0
    source_missing = 0
    source_mismatches = 0
    source_checked = 0
    run_count_bad = 0
    for index, frame in enumerate(frames):
        key = str(frame["cache_key"])
        if key in centers_by_key:
            duplicate_keys.append(key)
        start = int(frame_offsets[index])
        end = int(frame_offsets[index + 1])
        centers_by_key[key] = centers_full[start:end]
        if int(frame.get("steger_run_count", 0)) != 1:
            run_count_bad += 1
        source_path = Path(str(frame.get("source_path", "")))
        if not source_path.is_absolute():
            source_path = data_root / source_path
        source_checked += 1
        if not source_path.exists():
            source_missing += 1
        else:
            actual_hash = sha256_file(source_path)
            if actual_hash == frame.get("source_sha256"):
                source_matches += 1
            else:
                source_mismatches += 1
    if duplicate_keys or run_count_bad or source_mismatches or source_missing:
        raise RuntimeError(
            "Frozen cache/source audit failed: "
            f"duplicate_keys={len(duplicate_keys)}, run_count_bad={run_count_bad}, "
            f"source_missing={source_missing}, source_mismatches={source_mismatches}"
        )
    audit = {
        "cache_npz": str(npz_path.resolve()),
        "cache_manifest": str(manifest_path.resolve()),
        "one_steger_per_frame": bool(manifest["one_steger_per_frame"]),
        "manifest_reused_existing_cache_field": bool(manifest.get("reused_existing_cache", False)),
        "frame_count": len(frames),
        "center_point_count": int(len(centers_full)),
        "frame_offsets_shape": list(frame_offsets.shape),
        "source_checked": source_checked,
        "source_hash_matches": source_matches,
        "source_hash_missing": source_missing,
        "source_hash_mismatches": source_mismatches,
        "source_identity_ok": source_matches == source_checked == 600,
        "steger_run_count_bad": run_count_bad,
        "selection_basis": manifest.get("selection_basis"),
        "height_shadow_used": bool(manifest.get("height_shadow_used", True)),
        "residual_used": bool(manifest.get("residual_used", True)),
    }
    if not audit["source_identity_ok"]:
        raise RuntimeError("Frozen cache source identity is not complete")
    return centers_by_key, audit


def load_median_centerlines(output_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = output_dir / "session01_median_centerlines.npz"
    with np.load(path, allow_pickle=False) as cached:
        arrays = {key: np.asarray(cached[key]) for key in cached.files}
    expected = {f"{height}_{position}" for height in HEIGHT_LABELS for position in POSITION_IDS}
    missing = sorted(expected - set(arrays))
    extra = sorted(set(arrays) - expected)
    bad = {
        key: list(value.shape)
        for key, value in arrays.items()
        if value.ndim != 2 or value.shape[1] != 2
    }
    if missing or extra or bad:
        raise RuntimeError(
            f"median centerline audit failed: missing={missing}, extra={extra}, bad={bad}"
        )
    return arrays, {
        "path": str(path.resolve()),
        "condition_count": len(arrays),
        "point_count_by_condition": {key: int(len(value)) for key, value in arrays.items()},
    }


def integer_profile(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-v raw u profile, interpolation, and raw-valid mask."""
    values = np.asarray(centerline, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 50:
        raise ValueError("median centerline has fewer than 50 finite points")
    v = np.rint(values[:, 1]).astype(np.int64)
    u = values[:, 0]
    order = np.argsort(v)
    v = v[order]
    u = u[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    unique_u = np.zeros(len(unique_v), dtype=np.float64)
    for index in range(len(unique_v)):
        unique_u[index] = float(np.median(u[inverse == index]))
    grid = np.arange(FULL_SENSOR_HEIGHT, dtype=np.int64)
    raw = np.full(FULL_SENSOR_HEIGHT, np.nan, dtype=np.float64)
    in_bounds = (unique_v >= 0) & (unique_v < FULL_SENSOR_HEIGHT)
    raw[unique_v[in_bounds]] = unique_u[in_bounds]
    valid = np.isfinite(raw)
    if int(valid.sum()) < 50:
        raise ValueError("median centerline has too few valid v rows")
    interpolated = np.interp(grid.astype(np.float64), grid[valid], raw[valid])
    return grid.astype(np.float64), raw, interpolated


def interval_stats(
    raw: np.ndarray,
    start: int | None,
    end: int | None,
) -> dict[str, Any]:
    if start is None or end is None or end < start:
        return {
            "range": [],
            "width_px": 0,
            "valid_points": 0,
            "valid_fraction": 0.0,
            "missing_run_max_px": None,
            "slope_px_per_v": None,
            "roughness_px": None,
            "median_u": None,
            "stable": False,
            "available": False,
        }
    start = max(0, int(start))
    end = min(FULL_SENSOR_HEIGHT - 1, int(end))
    if end < start:
        return interval_stats(raw, None, None)
    segment = raw[start : end + 1]
    valid = np.isfinite(segment)
    valid_points = int(valid.sum())
    width = int(len(segment))
    valid_fraction = float(valid_points / width) if width else 0.0
    missing_run = 0
    current = 0
    for is_valid in valid:
        if is_valid:
            current = 0
        else:
            current += 1
            missing_run = max(missing_run, current)
    if valid_points < 3:
        return {
            "range": [start, end],
            "width_px": width,
            "valid_points": valid_points,
            "valid_fraction": valid_fraction,
            "missing_run_max_px": missing_run,
            "slope_px_per_v": None,
            "roughness_px": None,
            "median_u": None,
            "stable": False,
            "available": True,
        }
    x = np.arange(start, end + 1, dtype=np.float64)[valid]
    y = segment[valid]
    coefficient = np.polyfit(x, y, 1)
    residual = y - np.polyval(coefficient, x)
    slope = float(coefficient[0])
    roughness = float(np.std(residual))
    stable_params = PARAMETERS["stable_segment"]
    stable = bool(
        valid_fraction >= float(stable_params["minimum_valid_fraction"])
        and missing_run <= int(stable_params["maximum_missing_run_px"])
        and abs(slope) <= float(stable_params["maximum_abs_slope_px_per_v"])
        and roughness <= float(stable_params["maximum_roughness_px"])
    )
    return {
        "range": [start, end],
        "width_px": width,
        "valid_points": valid_points,
        "valid_fraction": valid_fraction,
        "missing_run_max_px": missing_run,
        "slope_px_per_v": slope,
        "roughness_px": roughness,
        "median_u": float(np.median(y)),
        "stable": stable,
        "available": True,
    }


def support_stats(
    frame_arrays: list[np.ndarray],
    interval: list[int],
    minimum_points: int,
    minimum_fraction: float,
) -> dict[str, Any]:
    if not interval or len(interval) != 2 or interval[1] < interval[0]:
        return {
            "range": [],
            "width_px": 0,
            "counts_by_repeat": {},
            "min_points": 0,
            "median_points": 0.0,
            "median_support_fraction": 0.0,
            "all_repeats_nonempty": False,
            "all_repeats_minimum_points": False,
            "support_ok": False,
        }
    start, end = max(0, int(interval[0])), min(FULL_SENSOR_HEIGHT - 1, int(interval[1]))
    width = max(0, end - start + 1)
    counts: dict[str, int] = {}
    for index, points in enumerate(frame_arrays, start=1):
        if len(points):
            count = int(np.count_nonzero((points[:, 1] >= start) & (points[:, 1] <= end)))
        else:
            count = 0
        counts[str(index)] = count
    values = np.asarray(list(counts.values()), dtype=np.float64)
    fractions = values / max(1, width)
    return {
        "range": [start, end],
        "width_px": width,
        "counts_by_repeat": counts,
        "min_points": int(values.min()) if len(values) else 0,
        "median_points": float(np.median(values)) if len(values) else 0.0,
        "median_support_fraction": float(np.median(fractions)) if len(fractions) else 0.0,
        "all_repeats_nonempty": bool(np.all(values > 0)) if len(values) else False,
        "all_repeats_minimum_points": bool(np.all(values >= minimum_points)) if len(values) else False,
        "support_ok": bool(
            len(values) == REPEAT_COUNT
            and np.all(values >= minimum_points)
            and np.median(fractions) >= minimum_fraction
        ),
    }


def local_support(raw: np.ndarray, center: int, radius: int) -> float:
    start = max(0, int(center) - radius)
    end = min(FULL_SENSOR_HEIGHT - 1, int(center) + radius)
    return float(np.isfinite(raw[start : end + 1]).mean())


def line_fit_from_stats(before: dict[str, Any], after: dict[str, Any], raw: np.ndarray) -> tuple[float, float] | None:
    arrays: list[np.ndarray] = []
    for stats in (before, after):
        if not stats.get("available"):
            continue
        start, end = stats["range"]
        x = np.arange(start, end + 1, dtype=np.float64)
        y = raw[start : end + 1]
        good = np.isfinite(y)
        if int(good.sum()) >= 3:
            arrays.append(np.column_stack([x[good], y[good]]))
    if not arrays:
        return None
    points = np.concatenate(arrays, axis=0)
    if len(points) < 6:
        return None
    coefficient = np.polyfit(points[:, 0], points[:, 1], 1)
    return float(coefficient[0]), float(coefficient[1])


def transition_range(
    derivative: np.ndarray,
    peak: int,
    expected_sign: float,
) -> list[int]:
    signal = expected_sign * derivative
    peak_value = max(0.0, float(signal[peak]))
    threshold = max(0.20, 0.20 * peak_value)
    left = int(peak)
    right = int(peak)
    while left > 0 and signal[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(signal) and signal[right + 1] >= threshold:
        right += 1
    return [left, right]


def build_edge_pairs(
    raw: np.ndarray,
    interpolated: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sigma = float(PARAMETERS["profile"]["smoothing_sigma_px"])
    derivative_sigma = float(PARAMETERS["profile"]["derivative_smoothing_sigma_px"])
    derivative = gaussian_filter1d(np.gradient(gaussian_filter1d(interpolated, sigma)), derivative_sigma)
    noise = 1.4826 * float(np.median(np.abs(derivative - np.median(derivative))))
    prominence_threshold = max(
        float(PARAMETERS["profile"]["edge_prominence_min_px"]),
        float(PARAMETERS["profile"]["edge_prominence_noise_multiplier"]) * noise,
    )
    distance = int(PARAMETERS["profile"]["edge_peak_distance_px"])
    negative, negative_props = find_peaks(
        -derivative,
        distance=distance,
        prominence=prominence_threshold,
    )
    positive, positive_props = find_peaks(
        derivative,
        distance=distance,
        prominence=prominence_threshold,
    )
    peak_sets = {
        "negative_to_positive": (
            negative,
            negative_props.get("prominences", np.empty(0)),
            positive,
            positive_props.get("prominences", np.empty(0)),
            -1.0,
            1.0,
        ),
        "positive_to_negative": (
            positive,
            positive_props.get("prominences", np.empty(0)),
            negative,
            negative_props.get("prominences", np.empty(0)),
            1.0,
            -1.0,
        ),
    }
    pairs: list[dict[str, Any]] = []
    min_width = float(PARAMETERS["object_interval"]["edge_pair_min_width_px"])
    max_width = float(PARAMETERS["object_interval"]["edge_pair_max_width_px"])
    support_radius = int(PARAMETERS["profile"]["edge_local_support_radius_px"])
    support_min_fraction = float(PARAMETERS["profile"]["edge_local_support_min_fraction"])
    for orientation, (first_peaks, first_proms, second_peaks, second_proms, first_sign, second_sign) in peak_sets.items():
        for first_index, first_peak in enumerate(first_peaks):
            for second_index, second_peak in enumerate(second_peaks):
                if second_peak <= first_peak:
                    continue
                width = float(second_peak - first_peak)
                if width < min_width or width > max_width:
                    continue
                edge1_local_support = local_support(raw, int(first_peak), support_radius)
                edge2_local_support = local_support(raw, int(second_peak), support_radius)
                margin = int(PARAMETERS["object_interval"]["transition_exclusion_margin_px"])
                height_start = int(first_peak + margin)
                height_end = int(second_peak - margin)
                if height_end < height_start:
                    continue
                safety_gap = int(PARAMETERS["baseline"]["safety_gap_px"])
                baseline_width = int(PARAMETERS["baseline"]["requested_width_px"])
                before_end = int(first_peak - safety_gap)
                before_start = max(0, before_end - baseline_width + 1) if before_end >= 0 else None
                after_start = int(second_peak + safety_gap)
                after_end = min(FULL_SENSOR_HEIGHT - 1, after_start + baseline_width - 1)
                before_range = (
                    [before_start, before_end]
                    if before_start is not None and before_end >= before_start
                    else []
                )
                after_range = (
                    [after_start, after_end]
                    if after_start <= after_end
                    else []
                )
                before_stats = interval_stats(raw, *(before_range or [None, None]))
                plateau_stats = interval_stats(raw, height_start, height_end)
                after_stats = interval_stats(raw, *(after_range or [None, None]))
                ground_fit = line_fit_from_stats(before_stats, after_stats, raw)
                plateau_mid = (height_start + height_end) / 2.0
                predicted_ground = (
                    float(np.polyval(ground_fit, plateau_mid)) if ground_fit is not None else None
                )
                plateau_delta = (
                    float(plateau_stats["median_u"] - predicted_ground)
                    if predicted_ground is not None and plateau_stats["median_u"] is not None
                    else None
                )
                expected_delta_ok = bool(
                    plateau_delta is not None
                    and ((first_sign < 0 and plateau_delta < 0) or (first_sign > 0 and plateau_delta > 0))
                )
                step_amplitude = abs(plateau_delta) if plateau_delta is not None else None
                edge1_prom = float(first_proms[first_index])
                edge2_prom = float(second_proms[second_index])
                edge1_transition = transition_range(derivative, int(first_peak), first_sign)
                edge2_transition = transition_range(derivative, int(second_peak), second_sign)
                height_width = int(height_end - height_start + 1)
                pair_score = float(
                    min(edge1_prom, edge2_prom)
                    + 0.10 * (step_amplitude or 0.0)
                    + 0.01 * min(width, 120.0)
                )
                reasons: list[str] = []
                if step_amplitude is None or step_amplitude < float(
                    PARAMETERS["object_interval"]["minimum_step_amplitude_px"]
                ):
                    reasons.append("step_amplitude_below_minimum")
                if edge1_local_support < support_min_fraction:
                    reasons.append("edge1_local_centerline_support_low")
                if edge2_local_support < support_min_fraction:
                    reasons.append("edge2_local_centerline_support_low")
                if not expected_delta_ok:
                    reasons.append("plateau_displacement_polarity_inconsistent")
                if not plateau_stats["stable"]:
                    reasons.append("height_plateau_not_stable")
                if not before_stats["stable"]:
                    reasons.append("baseline_before_not_stable_or_unavailable")
                if not after_stats["stable"]:
                    reasons.append("baseline_after_not_stable_or_unavailable")
                if height_width < float(PARAMETERS["object_interval"]["height_interior_min_width_px"]):
                    reasons.append("height_interior_width_below_minimum")
                if height_width > float(PARAMETERS["object_interval"]["height_interior_max_width_px"]):
                    reasons.append("height_interior_width_above_maximum")
                if before_stats["width_px"] < int(PARAMETERS["baseline"]["minimum_available_width_px"]):
                    reasons.append("baseline_before_available_width_below_minimum")
                if after_stats["width_px"] < int(PARAMETERS["baseline"]["minimum_available_width_px"]):
                    reasons.append("baseline_after_available_width_below_minimum")
                pairs.append(
                    {
                        "orientation": orientation,
                        "edge1_peak_v": int(first_peak),
                        "edge2_peak_v": int(second_peak),
                        "edge1_v": int(first_peak),
                        "edge2_v": int(second_peak),
                        "object_width_px": width,
                        "height_v_range": [height_start, height_end],
                        "height_interior_width_px": height_width,
                        "transition_exclusion_margin_px": margin,
                        "transition_v_ranges": {
                            "edge1": edge1_transition,
                            "edge2": edge2_transition,
                        },
                        "baseline_v_ranges": [before_range, after_range],
                        "baseline_clipped": {
                            "before": bool(not before_range or before_range[0] == 0),
                            "after": bool(not after_range or after_range[1] == FULL_SENSOR_HEIGHT - 1),
                            "before_unavailable": not bool(before_range),
                            "after_unavailable": not bool(after_range),
                        },
                        "edge1_prominence_px": edge1_prom,
                        "edge2_prominence_px": edge2_prom,
                        "edge1_local_support_fraction": edge1_local_support,
                        "edge2_local_support_fraction": edge2_local_support,
                        "edge_min_prominence_px": min(edge1_prom, edge2_prom),
                        "profile_noise_px_per_v": noise,
                        "before_stats": before_stats,
                        "height_stats": plateau_stats,
                        "after_stats": after_stats,
                        "ground_fit_slope_px_per_v": ground_fit[0] if ground_fit else None,
                        "ground_fit_intercept_px": ground_fit[1] if ground_fit else None,
                        "predicted_ground_u_at_height_mid": predicted_ground,
                        "plateau_delta_u_px": plateau_delta,
                        "step_amplitude_px": step_amplitude,
                        "pair_score": pair_score,
                        "pair_gate_reasons": reasons,
                        "edge_pair_geometry_ok": not reasons,
                    }
                )
    pairs.sort(
        key=lambda item: (
            bool(item["edge_pair_geometry_ok"]),
            -len(item["pair_gate_reasons"]),
            item["pair_score"],
        ),
        reverse=True,
    )
    return pairs, {
        "profile_noise_px_per_v": noise,
        "edge_prominence_threshold_px": prominence_threshold,
        "negative_peak_count": int(len(negative)),
        "positive_peak_count": int(len(positive)),
        "candidate_pair_count": len(pairs),
    }


def assess_condition(
    condition_id: str,
    median_centerline: np.ndarray,
    frame_arrays: list[np.ndarray],
    v1: dict[str, Any],
) -> dict[str, Any]:
    _, raw, interpolated = integer_profile(median_centerline)
    pairs, detector = build_edge_pairs(raw, interpolated)
    if not pairs:
        return {
            "condition_id": condition_id,
            "height_label": condition_id.split("_")[0],
            "position_id": condition_id.split("_")[1],
            "auto_qc_status": "FAIL",
            "auto_qc_reasons": ["no_two_edge_object_interval_candidate"],
            "detector_summary": detector,
            "all_edge_pairs": [],
            "transition_exclusion_margin_px": int(
                PARAMETERS["object_interval"]["transition_exclusion_margin_px"]
            ),
            "v1_comparison": v1,
            "auto_candidate_generated": True,
            "human_reviewed": False,
            "human_decision": "PENDING",
            "manual_confirmed": False,
            "frozen": False,
        }
    selected = pairs[0]
    reasons = list(selected["pair_gate_reasons"])
    height_support = support_stats(
        frame_arrays,
        selected["height_v_range"],
        int(PARAMETERS["height_support"]["minimum_points_per_repeat"]),
        float(PARAMETERS["height_support"]["minimum_support_fraction"]),
    )
    before_support = support_stats(
        frame_arrays,
        selected["baseline_v_ranges"][0],
        int(PARAMETERS["baseline"]["minimum_points_per_repeat"]),
        float(PARAMETERS["baseline"]["minimum_support_fraction"]),
    )
    after_support = support_stats(
        frame_arrays,
        selected["baseline_v_ranges"][1],
        int(PARAMETERS["baseline"]["minimum_points_per_repeat"]),
        float(PARAMETERS["baseline"]["minimum_support_fraction"]),
    )
    if not height_support["support_ok"]:
        reasons.append("height_20_repeat_formal_support_insufficient")
    if not before_support["support_ok"]:
        reasons.append("baseline_before_20_repeat_support_insufficient")
    if not after_support["support_ok"]:
        reasons.append("baseline_after_20_repeat_support_insufficient")

    if selected["height_interior_width_px"] < float(PARAMETERS["object_interval"]["height_interior_min_width_px"]):
        reasons.append("height_interior_width_below_minimum")
    if selected["height_interior_width_px"] > float(PARAMETERS["object_interval"]["height_interior_max_width_px"]):
        reasons.append("height_interior_width_above_maximum")
    if selected["object_width_px"] < float(PARAMETERS["object_interval"]["edge_pair_min_width_px"]):
        reasons.append("object_width_below_minimum")
    if selected["object_width_px"] > float(PARAMETERS["object_interval"]["edge_pair_max_width_px"]):
        reasons.append("object_width_above_maximum")
    if selected["baseline_v_ranges"][0] and selected["baseline_v_ranges"][1]:
        if selected["baseline_v_ranges"][0][1] >= selected["height_v_range"][0]:
            reasons.append("baseline_before_overlaps_height")
        if selected["height_v_range"][1] >= selected["baseline_v_ranges"][1][0]:
            reasons.append("height_overlaps_baseline_after")
    else:
        reasons.append("baseline_range_unavailable")

    # A close alternative edge pair is a geometry ambiguity, not a reason to
    # silently select a different condition-specific parameter.
    if len(pairs) > 1 and pairs[1]["pair_score"] > 0:
        gap = (selected["pair_score"] - pairs[1]["pair_score"]) / selected["pair_score"]
        selected["runner_up_pair_score"] = pairs[1]["pair_score"]
        selected["pair_score_relative_gap"] = float(gap)
        if gap < 0.10:
            reasons.append("edge_pair_score_ambiguous")
    else:
        selected["runner_up_pair_score"] = None
        selected["pair_score_relative_gap"] = None

    # Keep duplicate reasons out of the review text while retaining a stable
    # reason order for CSV/report comparison.
    reasons = list(dict.fromkeys(reasons))
    status = "PASS" if not reasons else "UNCERTAIN"
    height_range = selected["height_v_range"]
    v2_center = float((height_range[0] + height_range[1]) / 2.0)
    v1_range = v1.get("height_v_range") or []
    v1_center = v1.get("v_center_px")
    return {
        "condition_id": condition_id,
        "height_label": condition_id.split("_")[0],
        "position_id": condition_id.split("_")[1],
        "auto_qc_status": status,
        "auto_qc_reasons": reasons,
        "auto_candidate_generated": True,
        "human_reviewed": False,
        "human_decision": "PENDING",
        "manual_confirmed": False,
        "frozen": False,
        "geometry_only": True,
        "truth_height_used_for_roi": False,
        "height_shadow_used_for_roi": False,
        "c0_c1_ground_used_for_roi": False,
        "base_h1_hb2_used_for_roi": False,
        "residual_used_for_roi": False,
        "q1_q2_used_for_roi": False,
        "whole_frame_v_median_used_as_position": False,
        "detected_edges": {
            "v_edge_1": selected["edge1_v"],
            "v_edge_2": selected["edge2_v"],
            "edge1_peak_v": selected["edge1_peak_v"],
            "edge2_peak_v": selected["edge2_peak_v"],
            "transition_v_ranges": selected["transition_v_ranges"],
            "orientation": selected["orientation"],
        },
        "object_width_px": selected["object_width_px"],
        "height_interior_width_px": selected["height_interior_width_px"],
        "transition_exclusion_margin_px": selected["transition_exclusion_margin_px"],
        "height_roi_center_v": v2_center,
        "height_v_range": selected["height_v_range"],
        "baseline_v_ranges": selected["baseline_v_ranges"],
        "baseline_clipped": selected["baseline_clipped"],
        "edge_min_prominence_px": selected["edge_min_prominence_px"],
        "step_amplitude_px": selected["step_amplitude_px"],
        "plateau_delta_u_px": selected["plateau_delta_u_px"],
        "profile_noise_px_per_v": selected["profile_noise_px_per_v"],
        "profile_quality": {
            "baseline_before": selected["before_stats"],
            "height_plateau": selected["height_stats"],
            "baseline_after": selected["after_stats"],
        },
        "repeat_support": {
            "height": height_support,
            "baseline_before": before_support,
            "baseline_after": after_support,
        },
        "detector_summary": detector,
        "selected_pair_score": selected["pair_score"],
        "selected_pair_score_relative_gap": selected["pair_score_relative_gap"],
        "all_edge_pairs": pairs,
        "v1_comparison": {
            **v1,
            "v2_center_delta_px": (
                float(v2_center - v1_center) if v1_center is not None else None
            ),
            "v1_v2_width_delta_px": (
                int(selected["height_interior_width_px"] - v1["height_width_px"])
                if v1.get("height_width_px") is not None
                else None
            ),
            "v2_center": v2_center,
        },
    }


def draw_ranges(axis: Any, entry: dict[str, Any], v1: bool = False) -> None:
    if v1:
        ranges = {
            "baseline_before": (entry.get("v1_comparison", {}).get("baseline_v_ranges") or [[], []])[0],
            "height": entry.get("v1_comparison", {}).get("height_v_range") or [],
            "baseline_after": (entry.get("v1_comparison", {}).get("baseline_v_ranges") or [[], []])[1],
        }
        colors = {"baseline_before": "#4fc3f7", "height": "#ef5350", "baseline_after": "#4fc3f7"}
        alpha = 0.10
        linestyle = "--"
        label_prefix = "V1 "
    else:
        ranges = {
            "baseline_before": (entry.get("baseline_v_ranges") or [[], []])[0],
            "height": entry.get("height_v_range") or [],
            "baseline_after": (entry.get("baseline_v_ranges") or [[], []])[1],
        }
        colors = {"baseline_before": "#42a5f5", "height": "#66bb6a", "baseline_after": "#42a5f5"}
        alpha = 0.20
        linestyle = "-"
        label_prefix = "V2 "
    for name, value in ranges.items():
        if not value or len(value) != 2:
            continue
        axis.axhspan(
            value[0],
            value[1],
            color=colors[name],
            alpha=alpha,
            linestyle=linestyle,
            label=f"{label_prefix}{name}",
        )


def render_overlay(
    path: Path,
    image: np.ndarray,
    median_centerline: np.ndarray,
    entry: dict[str, Any],
    offset_x: int,
    include_v1: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_float = image.astype(np.float32)
    low, high = np.percentile(image_float, [1.0, 99.8])
    display = np.clip((image_float - low) * 255.0 / max(1.0, high - low), 0, 255)
    center = np.asarray(median_centerline, dtype=np.float64)
    valid = np.isfinite(center).all(axis=1)
    x_local = center[valid, 0] - float(offset_x)
    y_full = center[valid, 1]
    edge_data = entry.get("detected_edges", {})
    edge1 = edge_data.get("v_edge_1")
    edge2 = edge_data.get("v_edge_2")
    height_range = entry.get("height_v_range") or []
    zoom_center = (
        float((height_range[0] + height_range[1]) / 2.0)
        if len(height_range) == 2
        else float(np.nanmedian(y_full))
    )
    zoom_half = max(130.0, float((height_range[1] - height_range[0]) * 2.2)) if len(height_range) == 2 else 180.0
    y0 = max(0.0, zoom_center - zoom_half)
    y1 = min(float(FULL_SENSOR_HEIGHT - 1), zoom_center + zoom_half)
    x_center = float(np.nanmedian(x_local)) if len(x_local) else PNG_WIDTH / 2.0
    x0 = max(0.0, x_center - 130.0)
    x1 = min(float(PNG_WIDTH - 1), x_center + 130.0)
    status = entry.get("auto_qc_status", "FAIL")
    reasons = ", ".join(entry.get("auto_qc_reasons", [])) or "none"
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 10),
        gridspec_kw={"width_ratios": (1, 1.2)},
        constrained_layout=True,
    )
    for axis, view in zip(axes, ("full PNG", "object interval zoom")):
        axis.imshow(
            display,
            cmap="gray",
            vmin=0,
            vmax=255,
            origin="upper",
            extent=(0, PNG_WIDTH - 1, FULL_SENSOR_HEIGHT - 1, 0),
            aspect="auto" if view == "full PNG" else "equal",
        )
        axis.scatter(
            x_local,
            y_full,
            s=2.0,
            color="#e040fb",
            alpha=0.80,
            linewidths=0,
            label="median Frozen Steger",
        )
        if include_v1:
            draw_ranges(axis, entry, v1=True)
        draw_ranges(axis, entry, v1=False)
        if edge1 is not None:
            axis.axhline(edge1, color="#ff9800", linestyle="--", linewidth=1.2, label="V2 edge1")
            if len(height_range) == 2:
                axis.axhspan(edge1, height_range[0], color="#ff9800", alpha=0.18, label="transition exclusion")
        if edge2 is not None:
            axis.axhline(edge2, color="#ff9800", linestyle="--", linewidth=1.2, label="V2 edge2")
            if len(height_range) == 2:
                axis.axhspan(height_range[1], edge2, color="#ff9800", alpha=0.18)
        axis.set_title(view)
        axis.set_xlabel("u in PNG [px] (full u = PNG u + offset_x)")
        axis.set_ylabel("full-sensor v [px]")
        axis.grid(alpha=0.15)
        if view == "object interval zoom":
            axis.set_xlim(x0, x1)
            axis.set_ylim(y1, y0)
        else:
            axis.set_xlim(0, PNG_WIDTH - 1)
            axis.set_ylim(FULL_SENSOR_HEIGHT - 1, 0)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), loc="upper right", fontsize=7)
    figure.suptitle(
        f"{entry.get('condition_id', '')} | AUTO_ROI_STATUS={status} | "
        f"edge1={edge1} edge2={edge2} | object_width={entry.get('object_width_px')} px"
    )
    figure.text(0.01, 0.01, f"QC reasons: {reasons}", ha="left", va="bottom", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def make_csv_rows(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        before, after = (entry.get("baseline_v_ranges") or [[], []])[:2]
        qc = entry.get("profile_quality", {})
        support = entry.get("repeat_support", {})
        v1 = entry.get("v1_comparison", {})
        rows.append(
            {
                "dataset": "session01",
                "height_label": entry.get("height_label"),
                "position_id": entry.get("position_id"),
                "condition_id": entry.get("condition_id"),
                "auto_qc_status": entry.get("auto_qc_status"),
                "auto_qc_reasons": ";".join(entry.get("auto_qc_reasons", [])),
                "v_edge_1": entry.get("detected_edges", {}).get("v_edge_1"),
                "v_edge_2": entry.get("detected_edges", {}).get("v_edge_2"),
                "object_width_px": entry.get("object_width_px"),
                "transition_exclusion_margin_px": entry.get("transition_exclusion_margin_px"),
                "height_v_start": (entry.get("height_v_range") or [None, None])[0],
                "height_v_end": (entry.get("height_v_range") or [None, None])[1],
                "height_roi_center_v": entry.get("height_roi_center_v"),
                "height_interior_width_px": entry.get("height_interior_width_px"),
                "baseline_before_start": before[0] if before else None,
                "baseline_before_end": before[1] if before else None,
                "baseline_after_start": after[0] if after else None,
                "baseline_after_end": after[1] if after else None,
                "baseline_before_clipped": entry.get("baseline_clipped", {}).get("before"),
                "baseline_after_clipped": entry.get("baseline_clipped", {}).get("after"),
                "baseline_before_unavailable": entry.get("baseline_clipped", {}).get("before_unavailable"),
                "baseline_after_unavailable": entry.get("baseline_clipped", {}).get("after_unavailable"),
                "edge_min_prominence_px": entry.get("edge_min_prominence_px"),
                "step_amplitude_px": entry.get("step_amplitude_px"),
                "plateau_slope_px_per_v": qc.get("height_plateau", {}).get("slope_px_per_v"),
                "plateau_roughness_px": qc.get("height_plateau", {}).get("roughness_px"),
                "plateau_valid_fraction": qc.get("height_plateau", {}).get("valid_fraction"),
                "ground_before_slope_px_per_v": qc.get("baseline_before", {}).get("slope_px_per_v"),
                "ground_before_roughness_px": qc.get("baseline_before", {}).get("roughness_px"),
                "ground_after_slope_px_per_v": qc.get("baseline_after", {}).get("slope_px_per_v"),
                "ground_after_roughness_px": qc.get("baseline_after", {}).get("roughness_px"),
                "height_min_points_per_repeat": support.get("height", {}).get("min_points"),
                "height_median_support_fraction": support.get("height", {}).get("median_support_fraction"),
                "height_support_ok": support.get("height", {}).get("support_ok"),
                "baseline_before_support_ok": support.get("baseline_before", {}).get("support_ok"),
                "baseline_after_support_ok": support.get("baseline_after", {}).get("support_ok"),
                "v1_center_px": v1.get("v_center_px"),
                "v2_minus_v1_center_px": v1.get("v2_center_delta_px"),
                "v1_height_width_px": v1.get("height_width_px"),
                "v2_minus_v1_width_px": v1.get("v1_v2_width_delta_px"),
                "auto_candidate_generated": entry.get("auto_candidate_generated"),
                "human_reviewed": entry.get("human_reviewed"),
                "human_decision": entry.get("human_decision"),
                "manual_confirmed": entry.get("manual_confirmed"),
                "frozen": entry.get("frozen"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(json_safe(rows))


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(entry.get("auto_qc_status") for entry in entries)
    object_widths = [entry["object_width_px"] for entry in entries if entry.get("object_width_px") is not None]
    interior_widths = [
        entry["height_interior_width_px"]
        for entry in entries
        if entry.get("height_interior_width_px") is not None
    ]
    edge_margins = [
        entry.get("transition_exclusion_margin_px")
        for entry in entries
        if entry.get("transition_exclusion_margin_px") is not None
    ]
    clipped_count = sum(
        bool(entry.get("baseline_clipped", {}).get("before"))
        or bool(entry.get("baseline_clipped", {}).get("after"))
        for entry in entries
    )
    focused = [
        entry["condition_id"]
        for entry in entries
        if entry.get("auto_qc_status") != "PASS"
        or entry.get("baseline_clipped", {}).get("before")
        or entry.get("baseline_clipped", {}).get("after")
    ]
    center_deltas = [
        entry.get("v1_comparison", {}).get("v2_center_delta_px")
        for entry in entries
        if entry.get("v1_comparison", {}).get("v2_center_delta_px") is not None
    ]
    width_deltas = [
        entry.get("v1_comparison", {}).get("v1_v2_width_delta_px")
        for entry in entries
        if entry.get("v1_comparison", {}).get("v1_v2_width_delta_px") is not None
    ]
    return {
        "condition_count": len(entries),
        "status_counts": dict(status_counts),
        "object_width_px": distribution(object_widths),
        "height_interior_width_px": distribution(interior_widths),
        "edge_margin_px": distribution(edge_margins),
        "v2_minus_v1_center_delta_px": distribution(center_deltas),
        "v2_minus_v1_width_delta_px": distribution(width_deltas),
        "baseline_clipping_condition_count": clipped_count,
        "focused_review_conditions": focused,
    }


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None, "p05": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def build_report(
    output_dir: Path,
    entries: list[dict[str, Any]],
    summary: dict[str, Any],
    cache_audit: dict[str, Any],
    median_audit: dict[str, Any],
    development_cases: list[str],
    all_conditions_run: bool,
) -> str:
    status = {key: int(value) for key, value in summary["status_counts"].items()}
    dev_lines = []
    for case in development_cases:
        entry = next(item for item in entries if item["condition_id"] == case)
        dev_lines.append(
            f"| `{case}` | `{entry['auto_qc_status']}` | "
            f"{entry.get('detected_edges', {}).get('v_edge_1')} / "
            f"{entry.get('detected_edges', {}).get('v_edge_2')} | "
            f"{entry.get('object_width_px')} | {entry.get('height_interior_width_px')} | "
            f"{'; '.join(entry.get('auto_qc_reasons', [])) or 'none'} |"
        )
    focused = ", ".join(f"`{item}`" for item in summary["focused_review_conditions"]) or "none"
    lines = [
        "# Auto ROI V2 report｜Session01",
        "",
        "本轮只生成 geometry-only Auto ROI V2 candidates、QC、overlay 和 draft registry；没有重跑 Steger、没有重跑 A-13B、没有拟合/修改 C0/C1/Session PnP/Ground/H1/H-B2，也没有写入人工 freeze。",
        "",
        "## V1 root cause",
        "",
        "V1 failure mechanism 已单独记录在 `auto_roi_v1_failure_audit.md`：profile notch/step peak 被当作 object center，再套固定 `candidate_v ±45 px`；范围/点支持检查没有验证完整的 ground→object-top→ground 结构。",
        "",
        "## V2 geometry rule",
        "",
        "V2 在 median Frozen Steger `u(v)` 上检测两个相反方向的 transition edges，形成 `stable ground → edge1 → object-top plateau → edge2 → stable ground`。height ROI 使用 `edge1/edge2` 内部并扣除统一 transition safety margin；baseline 由两侧稳定 ground segment 生成，边界裁剪显式记录。所有参数在 `auto_roi_v2_parameters.json` 中统一冻结，本轮没有按 condition 改参数。",
        "",
        "Quality gates：两个 edge、object width、interior width、edge margin、plateau slope/roughness/continuity、两侧 ground 稳定性、baseline/height 不重叠、20 repeats formal support。候选未通过时标为 `UNCERTAIN`，不能自动 freeze。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- Frozen cache source hashes: `{cache_audit['source_hash_matches']}/{cache_audit['source_checked']}` match; center points `{cache_audit['center_point_count']}`; `one_steger_per_frame={cache_audit['one_steger_per_frame']}`.",
        f"- Median centerline NPZ: `{median_audit['condition_count']}` conditions; loaded from existing A-13A artifact, not regenerated.",
        "- ROI selection inputs were image/centerline geometry only. V1 ranges were read only for the requested V1-vs-V2 geometry diff and were not used to select V2 edges.",
        "",
        "## Development cases",
        "",
        "| case | V2 QC | edge1 / edge2 | object width | interior width | reasons |",
        "|---|---|---:|---:|---:|---|",
        *dev_lines,
        "",
        "四个 development case 均生成了 V1-vs-V2 overlay；本表没有查看或计算任何高度误差。",
        "",
        "## All-condition QC",
        "",
        f"- Conditions run: `{summary['condition_count']}`; PASS `{status.get('PASS', 0)}`; UNCERTAIN `{status.get('UNCERTAIN', 0)}`; FAIL `{status.get('FAIL', 0)}`.",
        f"- Object width distribution (px): `{json.dumps(summary['object_width_px'], ensure_ascii=False)}`.",
        f"- Height interior width distribution (px): `{json.dumps(summary['height_interior_width_px'], ensure_ascii=False)}`.",
        f"- Transition exclusion margin distribution (px): `{json.dumps(summary['edge_margin_px'], ensure_ascii=False)}`.",
        f"- V2 center minus V1 center distribution (px): `{json.dumps(summary['v2_minus_v1_center_delta_px'], ensure_ascii=False)}`.",
        f"- V2 height width minus V1 height width distribution (px): `{json.dumps(summary['v2_minus_v1_width_delta_px'], ensure_ascii=False)}`.",
        f"- Baseline-clipped conditions: `{summary['baseline_clipping_condition_count']}`.",
        f"- Conditions requiring focused human review: {focused}.",
        "",
        "## V1 versus V2 geometry",
        "",
        "`session01_auto_roi_v2_qc.csv` records V1 center/width, V2 center/width, center delta and width delta. These are geometry changes only; no residual/error column is generated or consulted.",
        "",
        "## Draft-only boundary",
        "",
        "The automatic output is `session01_roi_registry_v2_draft.json`. Every entry has `auto_candidate_generated=true`, `human_reviewed=false`, `human_decision=PENDING`, `manual_confirmed=false`, and `frozen=false`. A manual registry is intentionally not produced in this task. Stop here for genuine human review of all 30 overlays.",
        "",
        "## Final flags",
        "",
        "```text",
        "ROI_V1_ROOT_CAUSE_IDENTIFIED=YES",
        "AUTO_ROI_V2_IMPLEMENTED=YES",
    ]
    dev_flag_names = {
        "h10_p05": "DEV_H10_P05_GEOMETRY_OK",
        "h20_p03": "DEV_H20_P03_GEOMETRY_OK",
        "h30_p04": "DEV_H30_P04_GEOMETRY_OK",
        "h30_p07": "DEV_H30_P07_GEOMETRY_OK",
    }
    by_condition = {entry["condition_id"]: entry for entry in entries}
    for condition, flag in dev_flag_names.items():
        lines.append(f"{flag}={'YES' if by_condition.get(condition, {}).get('auto_qc_status') == 'PASS' else 'NO'}")
    lines.extend(
        [
            f"AUTO_ROI_V2_ALL_CONDITIONS_RUN={'YES' if all_conditions_run else 'NO'}",
            f"AUTO_ROI_V2_PASS_COUNT={status.get('PASS', 0)}",
            f"AUTO_ROI_V2_UNCERTAIN_COUNT={status.get('UNCERTAIN', 0)}",
            f"AUTO_ROI_V2_FAIL_COUNT={status.get('FAIL', 0)}",
            "AUTO_ROI_CAN_FREEZE_WITHOUT_HUMAN=NO",
            "HUMAN_REVIEW_REQUIRED=YES",
            "A13B_V1_INVALIDATED_BY_ROI_SELECTION=YES",
            "A13B_V2_ALLOWED=NO",
            "```",
            "",
            "人工 review 完成并生成新的 frozen registry 之前，禁止重跑 A-13B。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(
    output_dir: Path,
    data_root: Path,
    condition_filter: set[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "auto_roi_v2_parameters.json", PARAMETERS)
    v1 = load_v1_snapshot(output_dir)
    centers_by_key, cache_audit = load_cache(output_dir, data_root)
    median_by_key, median_audit = load_median_centerlines(output_dir)
    frame_arrays_by_condition: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, centers in centers_by_key.items():
        condition = key.split("/", 1)[0]
        frame_arrays_by_condition[condition].append(centers)
    for condition in frame_arrays_by_condition:
        frame_arrays_by_condition[condition].sort(key=lambda points: len(points))
    all_condition_ids = [
        f"{height_label}_{position_id}"
        for height_label in HEIGHT_LABELS
        for position_id in POSITION_IDS
    ]
    condition_ids = [
        condition_id
        for condition_id in all_condition_ids
        if condition_filter is None or condition_id in condition_filter
    ]
    entries: list[dict[str, Any]] = []
    for condition_id in condition_ids:
        frames = frame_arrays_by_condition.get(condition_id, [])
        if len(frames) != REPEAT_COUNT:
            raise RuntimeError(f"{condition_id}: expected 20 cached repeats, got {len(frames)}")
        entry = assess_condition(condition_id, median_by_key[condition_id], frames, v1[condition_id])
        entries.append(entry)

    development_cases = ["h10_p05", "h20_p03", "h30_p04", "h30_p07"]
    overlay_dir = output_dir / "roi_v2_review_overlays"
    development_dir = output_dir / "roi_v1_vs_v2_development_cases"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    development_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        condition_id = entry["condition_id"]
        image_path = output_dir / "median_images" / f"{condition_id}_median.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"missing median PNG: {image_path}")
        render_overlay(
            overlay_dir / f"{condition_id}_roi_v2_review_overlay.png",
            image,
            median_by_key[condition_id],
            entry,
            offset_x=1760,
            include_v1=False,
        )
        if condition_id in development_cases:
            render_overlay(
                development_dir / f"{condition_id}_v1_vs_v2_overlay.png",
                image,
                median_by_key[condition_id],
                entry,
                offset_x=1760,
                include_v1=True,
            )

    summary = summarize(entries)
    candidates_payload = {
        "schema_version": 2,
        "dataset": "session01",
        "created_at_utc": utc_now(),
        "geometry_only": True,
        "auto_candidate_generated": True,
        "human_reviewed": False,
        "human_decision": "PENDING",
        "manual_confirmed": False,
        "frozen": False,
        "candidate_rule": "stable ground -> edge1 -> object-top plateau -> edge2 -> stable ground",
        "parameters": PARAMETERS,
        "cache_audit": cache_audit,
        "median_centerline_audit": median_audit,
        "candidates": entries,
        "summary": summary,
    }
    write_json(output_dir / "session01_roi_candidates_v2.json", candidates_payload)
    draft_entries = []
    for entry in entries:
        draft_entries.append(
            {
                "dataset": "session01",
                "height_label": entry["height_label"],
                "position_id": entry["position_id"],
                "condition_id": entry["condition_id"],
                "height_v_range": entry.get("height_v_range", []),
                "baseline_v_ranges": entry.get("baseline_v_ranges", [[], []]),
                "height_roi_center_v": entry.get("height_roi_center_v"),
                "v_edge_1": entry.get("detected_edges", {}).get("v_edge_1"),
                "v_edge_2": entry.get("detected_edges", {}).get("v_edge_2"),
                "object_width_px": entry.get("object_width_px"),
                "height_interior_width_px": entry.get("height_interior_width_px"),
                "baseline_clipped": entry.get("baseline_clipped", {}),
                "auto_qc_status": entry.get("auto_qc_status"),
                "auto_qc_reasons": entry.get("auto_qc_reasons", []),
                "auto_candidate_generated": True,
                "human_reviewed": False,
                "human_decision": "PENDING",
                "manual_confirmed": False,
                "frozen": False,
                "geometry_only": True,
                "review_overlay": f"roi_v2_review_overlays/{entry['condition_id']}_roi_v2_review_overlay.png",
                "candidate_snapshot": entry,
            }
        )
    draft_payload = {
        "schema_version": 2,
        "dataset": "session01",
        "roi_stage": "auto_draft",
        "created_at_utc": utc_now(),
        "geometry_only": True,
        "auto_candidate_generated": True,
        "auto_qc_summary": summary,
        "human_reviewed": False,
        "human_decision": "PENDING",
        "manual_confirmed": False,
        "frozen": False,
        "frozen_at": None,
        "entries": draft_entries,
    }
    write_json(output_dir / "session01_roi_registry_v2_draft.json", draft_payload)
    write_csv(output_dir / "session01_auto_roi_v2_qc.csv", make_csv_rows(entries))
    write_json(
        output_dir / "auto_roi_v2_provenance_audit.json",
        {
            "created_at_utc": utc_now(),
            "data_root": str(data_root),
            "cache_audit": cache_audit,
            "median_centerline_audit": median_audit,
            "v1_candidates_read_only": True,
            "a13b_error_or_result_used": False,
            "steger_rerun": False,
            "new_correction_fit": False,
            "manual_registry_written": False,
            "draft_registry_written": True,
        },
    )
    report = build_report(
        output_dir,
        entries,
        summary,
        cache_audit,
        median_audit,
        development_cases,
        all_conditions_run=len(entries) == len(all_condition_ids),
    )
    (output_dir / "auto_roi_v2_report.md").write_text(report, encoding="utf-8")
    flags = {
        "ROI_V1_ROOT_CAUSE_IDENTIFIED": "YES",
        "AUTO_ROI_V2_IMPLEMENTED": "YES",
        **{
            name: "YES" if next(item for item in entries if item["condition_id"] == condition)["auto_qc_status"] == "PASS" else "NO"
            for condition, name in {
                "h10_p05": "DEV_H10_P05_GEOMETRY_OK",
                "h20_p03": "DEV_H20_P03_GEOMETRY_OK",
                "h30_p04": "DEV_H30_P04_GEOMETRY_OK",
                "h30_p07": "DEV_H30_P07_GEOMETRY_OK",
            }.items()
        },
        "AUTO_ROI_V2_ALL_CONDITIONS_RUN": "YES" if len(entries) == len(all_condition_ids) else "NO",
        "AUTO_ROI_V2_PASS_COUNT": int(summary["status_counts"].get("PASS", 0)),
        "AUTO_ROI_V2_UNCERTAIN_COUNT": int(summary["status_counts"].get("UNCERTAIN", 0)),
        "AUTO_ROI_V2_FAIL_COUNT": int(summary["status_counts"].get("FAIL", 0)),
        "AUTO_ROI_CAN_FREEZE_WITHOUT_HUMAN": "NO",
        "HUMAN_REVIEW_REQUIRED": "YES",
        "A13B_V1_INVALIDATED_BY_ROI_SELECTION": "YES",
        "A13B_V2_ALLOWED": "NO",
        "manual_registry_v2_written": False,
        "steger_rerun": False,
        "a13b_rerun": False,
    }
    write_json(output_dir / "auto_roi_v2_flags.json", flags)
    return {
        "summary": summary,
        "flags": flags,
        "cache_audit": cache_audit,
        "median_audit": median_audit,
        "output_dir": str(output_dir.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Run only the four named development cases before the 30-condition pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development_cases = {"h10_p05", "h20_p03", "h30_p04", "h30_p07"}
    result = run(
        args.output_dir.resolve(),
        args.data_root.resolve(),
        condition_filter=development_cases if args.development_only else None,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
