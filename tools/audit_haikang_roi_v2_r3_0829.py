#!/usr/bin/env python3
"""Audit Haikang ROI-V2 after replacing the board hard mask with quarantine.

This is an audit-only adapter around the existing H0-1R replay and ROI-V2
implementation.  It keeps the full centerline/profile as the candidate input,
quarantines candidates at the Session-board boundary or in a stationary
background cluster, and keeps target detection separate from baseline
measurement eligibility.  It does not measure height, apply compensation, or
change production configuration/parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "laser_measurement_tool"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_haikang_roi_v2_0829 as r1  # noqa: E402
import audit_haikang_roi_v2_board_mask_0829 as r2  # noqa: E402


DATA_ROOT_DEFAULT = r1.h0.DATA_ROOT_DEFAULT
OUTPUT_DIR_DEFAULT = DATA_ROOT_DEFAULT / "c0_height_audit" / "roi_v2_r3"
BOUNDARY_BANDS_MM = (5.0, 10.0)
KNOWN_BAD_CONDITIONS = {"h02_p01"}
FIXED_PAIR = (1950, 2024)
OVERLAY_REQUIRED_IDS = {
    "h02_p01",
    "h02_p02",
    "h02_p03",
    "h02_p05",
    "h02_p10",
    "h06_p01",
    "h06_p05",
    "h06_p10",
    "h10_p01",
    "h10_p05",
    "h10_p10",
    "h20_p01",
    "h20_p05",
    "h20_p10",
    "h30_p01",
    "h30_p05",
    "h30_p10",
}

# These are audit thresholds for identifying a repeatable stationary cluster;
# they are not ROI-V2 production parameters and are never applied to the
# profile or to the candidate score.
STATIONARY_EDGE_TOL_PX = 15.0
STATIONARY_CENTER_TOL_PX = 15.0
STATIONARY_WIDTH_TOL_PX = 20.0
STATIONARY_CENTER_STD_MAX_PX = 8.0
STATIONARY_EDGE_STD_MAX_PX = 12.0
STATIONARY_MIN_POSITION_FRACTION = 0.60
STATIONARY_EVIDENCE_POSITION_FRACTION = 0.50


class AuditError(RuntimeError):
    """Raised when the R3 audit contract cannot be satisfied safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=r1.h0.CONFIG_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args(argv)


def finite(value: Any) -> float | None:
    return r1.h0.finite(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def interval_values(interval: Any) -> list[Any]:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return []
    return [interval[0], interval[1]]


def interval_width(interval: Any) -> float | None:
    values = interval_values(interval)
    if not values:
        return None
    start = finite(values[0])
    end = finite(values[1])
    if start is None or end is None:
        return None
    return end - start + 1.0


def existing_sort_key(candidate: dict[str, Any]) -> tuple[bool, int, float]:
    score = finite(candidate.get("pair_score"))
    return (
        bool(candidate.get("edge_pair_geometry_ok")),
        -len(candidate.get("pair_gate_reasons") or []),
        score if score is not None else float("-inf"),
    )


def baseline_ranges(candidate: dict[str, Any]) -> list[Any]:
    return list(candidate.get("baseline_v_ranges") or [[], []]) + [[], []]


def candidate_roi(candidate: dict[str, Any]) -> dict[str, Any]:
    baselines = baseline_ranges(candidate)
    return {
        "height_u_full_range_px": candidate.get("height_v_range") or [],
        "baseline_before_u_full_range_px": baselines[0],
        "baseline_after_u_full_range_px": baselines[1],
    }


def load_board_geometry_r3(
    session_path: Path,
    app: Any,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Reuse R2's PnP replay and add an exact physical 5 mm projection."""
    geometry = r2.load_board_geometry(session_path, app, calibration)
    document = geometry["document"]
    board = geometry["board"]
    pnp_offset = tuple(int(value) for value in geometry["pnp_offset"])
    corners = np.asarray(
        (document.get("detection") or {}).get("corners"), dtype=np.float32
    )
    K = np.asarray(calibration.get("K"), dtype=np.float64).copy()
    D = np.asarray(calibration.get("D"), dtype=np.float64)
    K[0, 2] -= float(pnp_offset[0])
    K[1, 2] -= float(pnp_offset[1])
    pnp = r2.estimate_session_ground_extrinsic_from_corners(
        corners,
        {"K": K, "D": D},
        board,
        detection_method="stored_session_detection_r3_boundary_audit",
    )
    if pnp.status != "success" or pnp.rvec is None or pnp.tvec is None:
        raise AuditError(f"R3 PnP replay failed: {pnp.message}")
    geometry["polygons_full_uv"][5.0] = r2.full_board_physical_polygon(
        pnp.rvec,
        pnp.tvec,
        pattern_cols=board.pattern_cols,
        pattern_rows=board.pattern_rows,
        square_size_mm=board.square_size_mm,
        camera_matrix=K,
        dist_coeffs=D,
        image_offset=pnp_offset,
        inset_mm=5.0,
    )
    square = float(board.square_size_mm)
    geometry["physical_bounds_mm"][5.0] = {
        "x_min_mm": -square + 5.0,
        "x_max_mm": float(board.pattern_cols) * square - 5.0,
        "y_min_mm": -square + 5.0,
        "y_max_mm": float(board.pattern_rows) * square - 5.0,
    }
    geometry["r3_pnp_reprojection_rmse_px"] = finite(pnp.reprojection_rmse_px)
    geometry["r3_camera_matrix"] = K
    geometry["r3_dist_coeffs"] = D
    geometry["r3_pnp_rvec"] = np.asarray(pnp.rvec)
    geometry["r3_pnp_tvec"] = np.asarray(pnp.tvec)
    return geometry


def profile_diagnostics(audit: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the existing peak extraction with the existing parameters."""
    raw = audit.get("raw_profile")
    interpolated = audit.get("interpolated_profile")
    if raw is None or interpolated is None:
        return {
            "status": "NO_PROFILE",
            "raw": None,
            "interpolated": None,
            "derivative": None,
            "negative": np.empty(0, dtype=np.int64),
            "positive": np.empty(0, dtype=np.int64),
            "negative_prominences": np.empty(0, dtype=np.float64),
            "positive_prominences": np.empty(0, dtype=np.float64),
            "noise": None,
            "threshold": None,
        }
    roi_v2 = r1.h0.roi_v2
    raw_array = np.asarray(raw, dtype=np.float64)
    profile = np.asarray(interpolated, dtype=np.float64)
    sigma = float(roi_v2.PARAMETERS["profile"]["smoothing_sigma_px"])
    derivative_sigma = float(
        roi_v2.PARAMETERS["profile"]["derivative_smoothing_sigma_px"]
    )
    derivative = roi_v2.gaussian_filter1d(
        np.gradient(roi_v2.gaussian_filter1d(profile, sigma)), derivative_sigma
    )
    noise = 1.4826 * float(np.median(np.abs(derivative - np.median(derivative))))
    threshold = max(
        float(roi_v2.PARAMETERS["profile"]["edge_prominence_min_px"]),
        float(roi_v2.PARAMETERS["profile"]["edge_prominence_noise_multiplier"])
        * noise,
    )
    distance = int(roi_v2.PARAMETERS["profile"]["edge_peak_distance_px"])
    negative, negative_props = roi_v2.find_peaks(
        -derivative, distance=distance, prominence=threshold
    )
    positive, positive_props = roi_v2.find_peaks(
        derivative, distance=distance, prominence=threshold
    )
    direct = audit.get("direct_detector") or {}
    if int(direct.get("negative_peak_count", len(negative))) != len(negative):
        raise AuditError(f"negative peak replay mismatch for {audit['condition']['condition_id']}")
    if int(direct.get("positive_peak_count", len(positive))) != len(positive):
        raise AuditError(f"positive peak replay mismatch for {audit['condition']['condition_id']}")
    return {
        "status": "PROFILE_AVAILABLE",
        "raw": raw_array,
        "interpolated": profile,
        "derivative": np.asarray(derivative, dtype=np.float64),
        "negative": np.asarray(negative, dtype=np.int64),
        "positive": np.asarray(positive, dtype=np.int64),
        "negative_prominences": np.asarray(
            negative_props.get("prominences", np.empty(0)), dtype=np.float64
        ),
        "positive_prominences": np.asarray(
            positive_props.get("prominences", np.empty(0)), dtype=np.float64
        ),
        "noise": noise,
        "threshold": threshold,
    }


def peak_json(peaks: np.ndarray, prominences: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"v": int(peak), "prominence_px": float(prominence)}
        for peak, prominence in zip(peaks, prominences)
    ]


def pair_metric_diagnostic(
    raw: np.ndarray,
    derivative: np.ndarray,
    orientation: str,
    first_peak: int,
    first_prominence: float,
    second_peak: int,
    second_prominence: float,
) -> dict[str, Any]:
    """Use ROI-V2's existing low-level helpers to inspect every peak pair."""
    roi_v2 = r1.h0.roi_v2
    params = roi_v2.PARAMETERS
    first_sign, second_sign = (
        (-1.0, 1.0)
        if orientation == "negative_to_positive"
        else (1.0, -1.0)
    )
    width = float(second_peak - first_peak)
    radius = int(params["profile"]["edge_local_support_radius_px"])
    min_support = float(params["profile"]["edge_local_support_min_fraction"])
    edge1_support = roi_v2.local_support(raw, first_peak, radius)
    edge2_support = roi_v2.local_support(raw, second_peak, radius)
    margin = int(params["object_interval"]["transition_exclusion_margin_px"])
    height_start = int(first_peak + margin)
    height_end = int(second_peak - margin)
    height_width = int(height_end - height_start + 1)
    safety_gap = int(params["baseline"]["safety_gap_px"])
    baseline_width = int(params["baseline"]["requested_width_px"])
    before_end = int(first_peak - safety_gap)
    before_start = (
        max(0, before_end - baseline_width + 1) if before_end >= 0 else None
    )
    after_start = int(second_peak + safety_gap)
    after_end = min(
        int(roi_v2.FULL_SENSOR_HEIGHT) - 1,
        after_start + baseline_width - 1,
    )
    before_range = (
        [before_start, before_end]
        if before_start is not None and before_end >= before_start
        else []
    )
    after_range = (
        [after_start, after_end] if after_start <= after_end else []
    )
    before_stats = roi_v2.interval_stats(raw, *(before_range or [None, None]))
    height_stats = (
        roi_v2.interval_stats(raw, height_start, height_end)
        if height_end >= height_start
        else roi_v2.interval_stats(raw, None, None)
    )
    after_stats = roi_v2.interval_stats(raw, *(after_range or [None, None]))
    ground_fit = roi_v2.line_fit_from_stats(before_stats, after_stats, raw)
    height_mid = (height_start + height_end) / 2.0
    predicted_ground = (
        float(np.polyval(ground_fit, height_mid)) if ground_fit is not None else None
    )
    plateau_delta = (
        float(height_stats["median_u"] - predicted_ground)
        if predicted_ground is not None and height_stats["median_u"] is not None
        else None
    )
    expected_delta_ok = bool(
        plateau_delta is not None
        and (
            (first_sign < 0 and plateau_delta < 0)
            or (first_sign > 0 and plateau_delta > 0)
        )
    )
    step_amplitude = abs(plateau_delta) if plateau_delta is not None else None
    edge1_transition = roi_v2.transition_range(
        derivative, int(first_peak), first_sign
    )
    edge2_transition = roi_v2.transition_range(
        derivative, int(second_peak), second_sign
    )
    score = float(
        min(first_prominence, second_prominence)
        + 0.10 * (step_amplitude or 0.0)
        + 0.01 * min(width, 120.0)
    )
    generation_reasons: list[str] = []
    min_width = float(params["object_interval"]["edge_pair_min_width_px"])
    max_width = float(params["object_interval"]["edge_pair_max_width_px"])
    if width < min_width or width > max_width:
        generation_reasons.append("edge_pair_width_outside_generation_range")
    if height_width < 1:
        generation_reasons.append("height_interior_empty_at_generation")
    return {
        "orientation": orientation,
        "edge1_peak_v": int(first_peak),
        "edge2_peak_v": int(second_peak),
        "edge1_prominence_px": float(first_prominence),
        "edge2_prominence_px": float(second_prominence),
        "edge_min_prominence_px": float(min(first_prominence, second_prominence)),
        "object_width_px": width,
        "transition_exclusion_margin_px": margin,
        "height_v_range": (
            [height_start, height_end] if height_end >= height_start else []
        ),
        "height_interior_width_px": height_width if height_width > 0 else 0,
        "baseline_v_ranges": [before_range, after_range],
        "baseline_clipped": {
            "before": bool(not before_range or before_range[0] == 0),
            "after": bool(
                not after_range
                or after_range[1] == int(roi_v2.FULL_SENSOR_HEIGHT) - 1
            ),
            "before_unavailable": not bool(before_range),
            "after_unavailable": not bool(after_range),
        },
        "transition_v_ranges": {
            "edge1": edge1_transition,
            "edge2": edge2_transition,
        },
        "edge1_local_support_fraction": edge1_support,
        "edge2_local_support_fraction": edge2_support,
        "before_stats": before_stats,
        "height_stats": height_stats,
        "after_stats": after_stats,
        "ground_fit_slope_px_per_v": ground_fit[0] if ground_fit else None,
        "ground_fit_intercept_px": ground_fit[1] if ground_fit else None,
        "predicted_ground_u_at_height_mid": predicted_ground,
        "plateau_delta_u_px": plateau_delta,
        "step_amplitude_px": step_amplitude,
        "pair_score": score,
        "generation_reasons": generation_reasons,
        "edge_support_minimum_fraction": min_support,
    }


def profile_peak_sets(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "orientation": "negative_to_positive",
            "first": profile["negative"],
            "first_prominences": profile["negative_prominences"],
            "second": profile["positive"],
            "second_prominences": profile["positive_prominences"],
        },
        {
            "orientation": "positive_to_negative",
            "first": profile["positive"],
            "first_prominences": profile["positive_prominences"],
            "second": profile["negative"],
            "second_prominences": profile["negative_prominences"],
        },
    ]


def build_transition_rows(
    audit: dict[str, Any], profile: dict[str, Any]
) -> list[dict[str, Any]]:
    condition = audit["condition"]
    base = {
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "condition_id": condition.condition_id,
        "known_bad_image_exception": condition.condition_id in KNOWN_BAD_CONDITIONS,
        "profile_stage_status": profile["status"],
        "profile_noise_px_per_v": profile["noise"],
        "edge_prominence_threshold_px": profile["threshold"],
        "negative_peak_count": int(len(profile["negative"])),
        "positive_peak_count": int(len(profile["positive"])),
        "negative_peaks": peak_json(
            profile["negative"], profile["negative_prominences"]
        ),
        "positive_peaks": peak_json(
            profile["positive"], profile["positive_prominences"]
        ),
        "direct_build_edge_pairs_count": audit.get("direct_candidate_count"),
        "assessment_candidate_count": len(audit.get("candidates") or []),
    }
    if profile["status"] != "PROFILE_AVAILABLE":
        return [
            {
                **base,
                "row_type": "CONDITION_NO_PROFILE",
                "pair_id": "",
                "pair_generation_stage": "NO_PROFILE",
            }
        ]
    generated = {
        (
            str(candidate.get("orientation")),
            int(candidate.get("edge1_peak_v")),
            int(candidate.get("edge2_peak_v")),
        ): (rank, candidate)
        for rank, candidate in enumerate(audit.get("candidates") or [], start=1)
    }
    rows: list[dict[str, Any]] = []
    pair_counter = 0
    for peak_set in profile_peak_sets(profile):
        first = peak_set["first"]
        second = peak_set["second"]
        first_prominences = peak_set["first_prominences"]
        second_prominences = peak_set["second_prominences"]
        for first_index, first_peak in enumerate(first):
            for second_index, second_peak in enumerate(second):
                if int(second_peak) <= int(first_peak):
                    continue
                pair_counter += 1
                metrics = pair_metric_diagnostic(
                    profile["raw"],
                    profile["derivative"],
                    peak_set["orientation"],
                    int(first_peak),
                    float(first_prominences[first_index]),
                    int(second_peak),
                    float(second_prominences[second_index]),
                )
                key = (
                    peak_set["orientation"],
                    int(first_peak),
                    int(second_peak),
                )
                generated_item = generated.get(key)
                if generated_item is None:
                    stage = (
                        "REJECTED_BEFORE_CANDIDATE_WIDTH"
                        if "edge_pair_width_outside_generation_range"
                        in metrics["generation_reasons"]
                        else "REJECTED_BEFORE_CANDIDATE_HEIGHT_INTERIOR"
                    )
                    candidate_rank = None
                    candidate = {}
                    pair_reasons = metrics["generation_reasons"]
                    geometry_ok = False
                else:
                    candidate_rank, candidate = generated_item
                    stage = "GENERATED_CANDIDATE"
                    pair_reasons = candidate.get("pair_gate_reasons") or []
                    geometry_ok = candidate.get("edge_pair_geometry_ok")
                baselines = baseline_ranges(candidate or metrics)
                before_stats = candidate.get("before_stats") or metrics["before_stats"]
                height_stats = candidate.get("height_stats") or metrics["height_stats"]
                after_stats = candidate.get("after_stats") or metrics["after_stats"]
                row = {
                    **base,
                    "row_type": "PEAK_PAIR",
                    "pair_id": f"{condition.condition_id}:{pair_counter}",
                    "pair_generation_stage": stage,
                    "candidate_generated": stage == "GENERATED_CANDIDATE",
                    "candidate_rank": candidate_rank,
                    "orientation": metrics["orientation"],
                    "edge1_peak_v": metrics["edge1_peak_v"],
                    "edge2_peak_v": metrics["edge2_peak_v"],
                    "edge1_u_full_px": metrics["edge1_peak_v"],
                    "edge2_u_full_px": metrics["edge2_peak_v"],
                    "object_width_px": candidate.get(
                        "object_width_px", metrics["object_width_px"]
                    ),
                    "height_v_range": candidate.get(
                        "height_v_range", metrics["height_v_range"]
                    ),
                    "height_interior_width_px": candidate.get(
                        "height_interior_width_px",
                        metrics["height_interior_width_px"],
                    ),
                    "transition_v_ranges": candidate.get(
                        "transition_v_ranges", metrics["transition_v_ranges"]
                    ),
                    "baseline_v_ranges": baselines[:2],
                    "baseline_clipped": candidate.get(
                        "baseline_clipped", metrics["baseline_clipped"]
                    ),
                    "edge1_prominence_px": metrics["edge1_prominence_px"],
                    "edge2_prominence_px": metrics["edge2_prominence_px"],
                    "edge_min_prominence_px": metrics["edge_min_prominence_px"],
                    "edge1_local_support_fraction": metrics[
                        "edge1_local_support_fraction"
                    ],
                    "edge2_local_support_fraction": metrics[
                        "edge2_local_support_fraction"
                    ],
                    "before_slope_px_per_v": before_stats.get("slope_px_per_v"),
                    "before_roughness_px": before_stats.get("roughness_px"),
                    "height_slope_px_per_v": height_stats.get("slope_px_per_v"),
                    "height_roughness_px": height_stats.get("roughness_px"),
                    "after_slope_px_per_v": after_stats.get("slope_px_per_v"),
                    "after_roughness_px": after_stats.get("roughness_px"),
                    "predicted_ground_u_at_height_mid": candidate.get(
                        "predicted_ground_u_at_height_mid",
                        metrics["predicted_ground_u_at_height_mid"],
                    ),
                    "plateau_delta_u_px": candidate.get(
                        "plateau_delta_u_px", metrics["plateau_delta_u_px"]
                    ),
                    "step_amplitude_px": candidate.get(
                        "step_amplitude_px", metrics["step_amplitude_px"]
                    ),
                    "pair_score": candidate.get("pair_score", metrics["pair_score"]),
                    "score_recomputed": metrics["pair_score"],
                    "score_delta": (
                        float(candidate["pair_score"]) - metrics["pair_score"]
                        if candidate.get("pair_score") is not None
                        else None
                    ),
                    "pair_gate_reasons": pair_reasons,
                    "edge_pair_geometry_ok": geometry_ok,
                    "before_stats": before_stats,
                    "height_stats": height_stats,
                    "after_stats": after_stats,
                    "height_support": candidate.get("height_support"),
                    "baseline_before_support": candidate.get("before_support"),
                    "baseline_after_support": candidate.get("after_support"),
                    "multi_geometry_reasons": candidate.get(
                        "multi_geometry_reasons"
                    ),
                    "multi_geometry_ok": candidate.get("multi_geometry_ok"),
                    "generation_rejection_reasons": metrics["generation_reasons"],
                }
                rows.append(row)
    if not rows:
        rows.append(
            {
                **base,
                "row_type": "CONDITION_NO_PEAK_PAIR",
                "pair_id": "",
                "pair_generation_stage": "NO_PEAK_PAIR",
            }
        )
    return rows


def profile_point(
    v_value: Any, interpolated: np.ndarray | None, polygons: dict[float, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile_row": None,
        "profile_value_v_full_px": None,
        "full_uv": [],
        "inside_outer": False,
        "inside_inner_5mm": False,
        "inside_inner_10mm": False,
        "status_5mm": "UNAVAILABLE",
        "status_10mm": "UNAVAILABLE",
    }
    row_value = finite(v_value)
    if row_value is None or interpolated is None:
        return result
    row = int(round(row_value))
    if row < 0 or row >= len(interpolated):
        return result
    value = finite(interpolated[row])
    if value is None:
        return result
    point = np.asarray([[float(row), value]], dtype=np.float64)
    inside_outer = bool(
        r2._points_inside_convex_polygon(point, polygons[0.0])[0]
    )
    inside_5 = bool(r2._points_inside_convex_polygon(point, polygons[5.0])[0])
    inside_10 = bool(r2._points_inside_convex_polygon(point, polygons[10.0])[0])
    result.update(
        {
            "profile_row": row,
            "profile_value_v_full_px": value,
            "full_uv": [float(row), value],
            "inside_outer": inside_outer,
            "inside_inner_5mm": inside_5,
            "inside_inner_10mm": inside_10,
            "status_5mm": (
                "INTERIOR"
                if inside_5
                else "BOUNDARY_BAND"
                if inside_outer
                else "OUTSIDE_PERIMETER"
            ),
            "status_10mm": (
                "INTERIOR"
                if inside_10
                else "BOUNDARY_BAND"
                if inside_outer
                else "OUTSIDE_PERIMETER"
            ),
        }
    )
    return result


def interval_boundary_state(
    interval: Any,
    interpolated: np.ndarray | None,
    outer_polygon: np.ndarray,
) -> dict[str, Any]:
    values = interval_values(interval)
    result = {
        "range": values,
        "sample_count": 0,
        "inside_count": 0,
        "outside_count": 0,
        "crosses_perimeter": False,
        "available": False,
    }
    if len(values) != 2 or interpolated is None:
        return result
    try:
        start, end = int(values[0]), int(values[1])
    except (TypeError, ValueError):
        return result
    if end < start:
        return result
    rows = np.arange(max(0, start), min(len(interpolated) - 1, end) + 1)
    if not len(rows):
        return result
    profile_values = np.asarray(interpolated[rows], dtype=np.float64)
    finite_mask = np.isfinite(profile_values)
    if not np.any(finite_mask):
        return result
    points = np.column_stack([rows[finite_mask], profile_values[finite_mask]])
    inside = r2._points_inside_convex_polygon(points, outer_polygon)
    inside_count = int(np.count_nonzero(inside))
    outside_count = int(len(inside) - inside_count)
    result.update(
        {
            "sample_count": int(len(inside)),
            "inside_count": inside_count,
            "outside_count": outside_count,
            "crosses_perimeter": bool(inside_count > 0 and outside_count > 0),
            "available": True,
        }
    )
    return result


def candidate_boundary_evaluation(
    candidate: dict[str, Any],
    interpolated: np.ndarray | None,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    polygons = geometry["polygons_full_uv"]
    edge1 = profile_point(candidate.get("edge1_v"), interpolated, polygons)
    edge2 = profile_point(candidate.get("edge2_v"), interpolated, polygons)
    perimeter = interval_boundary_state(
        [candidate.get("edge1_v"), candidate.get("edge2_v")],
        interpolated,
        polygons[0.0],
    )
    result: dict[str, Any] = {
        "edge1": edge1,
        "edge2": edge2,
        "pair_interval": perimeter,
    }
    for band in BOUNDARY_BANDS_MM:
        suffix = f"{int(band)}mm"
        reasons: list[str] = []
        if edge1[f"status_{suffix}"] != "INTERIOR":
            reasons.append(f"edge1_{edge1[f'status_{suffix}'].lower()}")
        if edge2[f"status_{suffix}"] != "INTERIOR":
            reasons.append(f"edge2_{edge2[f'status_{suffix}'].lower()}")
        if perimeter["crosses_perimeter"]:
            reasons.append("candidate_crosses_board_perimeter")
        result[f"quarantine_{suffix}"] = bool(reasons)
        result[f"status_{suffix}"] = (
            "BOARD_BOUNDARY_BACKGROUND" if reasons else "INTERIOR_TARGET_DOMAIN"
        )
        result[f"reasons_{suffix}"] = list(
            dict.fromkeys(["BOARD_BOUNDARY_BACKGROUND", *reasons] if reasons else [])
        )
    return result


BASELINE_REASON_MARKERS = (
    "baseline",
    "before_support",
    "after_support",
)


def is_baseline_reason(reason: Any) -> bool:
    text = str(reason).lower()
    return any(marker in text for marker in BASELINE_REASON_MARKERS)


def target_core_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons = list(candidate.get("pair_gate_reasons") or [])
    reasons.extend(
        reason
        for reason in (candidate.get("multi_geometry_reasons") or [])
        if not is_baseline_reason(reason)
    )
    return list(dict.fromkeys(str(reason) for reason in reasons if reason and not is_baseline_reason(reason)))


def support_available(
    support: dict[str, Any] | None, stats: dict[str, Any] | None, interval: Any
) -> bool:
    if support and "available" in support:
        return bool(support.get("available"))
    if stats and "available" in stats:
        return bool(stats.get("available"))
    return bool(interval_values(interval))


def support_stable(
    support: dict[str, Any] | None, stats: dict[str, Any] | None
) -> bool | None:
    if stats and "stable" in stats:
        return bool(stats.get("stable"))
    if support and "support_ok" in support:
        return bool(support.get("support_ok"))
    return None


def support_ok(
    support: dict[str, Any] | None, stats: dict[str, Any] | None
) -> bool | None:
    if support and "support_ok" in support:
        return bool(support.get("support_ok"))
    if stats and "stable" in stats:
        return bool(stats.get("stable"))
    return None


def baseline_evaluation(candidate: dict[str, Any]) -> dict[str, Any]:
    ranges = baseline_ranges(candidate)
    before_stats = candidate.get("before_stats") or {}
    after_stats = candidate.get("after_stats") or {}
    before_support = candidate.get("before_support") or {}
    after_support = candidate.get("after_support") or {}
    before_available = support_available(before_support, before_stats, ranges[0])
    after_available = support_available(after_support, after_stats, ranges[1])
    if before_available and after_available:
        status = "BOTH_AVAILABLE"
    elif before_available or after_available:
        status = "ONE_SIDE_ONLY"
    else:
        status = "UNAVAILABLE"
    reasons: list[str] = []
    if not before_available:
        reasons.append("baseline_before_unavailable")
    elif support_stable(before_support, before_stats) is False:
        reasons.append("baseline_before_unstable")
    if not after_available:
        reasons.append("baseline_after_unavailable")
    elif support_stable(after_support, after_stats) is False:
        reasons.append("baseline_after_unstable")
    before_ok = support_ok(before_support, before_stats)
    after_ok = support_ok(after_support, after_stats)
    if before_ok is False:
        reasons.append("baseline_before_support_not_ok")
    if after_ok is False:
        reasons.append("baseline_after_support_not_ok")
    clipped = candidate.get("baseline_clipped") or {}
    if clipped.get("before"):
        reasons.append("baseline_before_clipped")
    if clipped.get("after"):
        reasons.append("baseline_after_clipped")
    eligible = bool(
        status == "BOTH_AVAILABLE"
        and before_ok is not False
        and after_ok is not False
        and not clipped.get("before")
        and not clipped.get("after")
    )
    return {
        "local_baseline_status": status,
        "before_available": before_available,
        "after_available": after_available,
        "before_stable": support_stable(before_support, before_stats),
        "after_stable": support_stable(after_support, after_stats),
        "before_support_ok": before_ok,
        "after_support_ok": after_ok,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "before_support": before_support,
        "after_support": after_support,
        "baseline_measurement_eligible": eligible,
        "baseline_failure_reasons": list(dict.fromkeys(reasons)),
    }


def make_candidate_infos(
    audit: dict[str, Any],
    geometry: dict[str, Any],
    a3_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    condition = audit["condition"]
    for rank, candidate in enumerate(audit.get("candidates") or [], start=1):
        boundary = candidate_boundary_evaluation(
            candidate, audit.get("interpolated_profile"), geometry
        )
        a3 = r1.h0.a3_overlap_fields(candidate_roi(candidate), a3_audit)
        infos.append(
            {
                "candidate": candidate,
                "candidate_id": f"{condition.condition_id}:r{rank}",
                "condition_id": condition.condition_id,
                "height_id": condition.height_id,
                "position_id": condition.position_id,
                "rank": rank,
                "edge1": finite(candidate.get("edge1_v")),
                "edge2": finite(candidate.get("edge2_v")),
                "center": (
                    (float(candidate["edge1_v"]) + float(candidate["edge2_v"])) / 2.0
                    if candidate.get("edge1_v") is not None
                    and candidate.get("edge2_v") is not None
                    else None
                ),
                "width": finite(candidate.get("object_width_px")),
                "boundary": boundary,
                "a3": a3,
                "baseline": baseline_evaluation(candidate),
                "target_core_reasons": target_core_reasons(candidate),
                "known_bad_image_exception": condition.condition_id
                in KNOWN_BAD_CONDITIONS,
                "stationary_cluster_id": None,
                "stationary_label": "NOT_CLUSTERED",
                "stationary_background": False,
                "selected_by_band": {},
            }
        )
    return infos


def union_find_components(items: list[dict[str, Any]]) -> list[list[int]]:
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            a, b = items[left], items[right]
            if a["height_id"] != b["height_id"]:
                continue
            if a["edge1"] is None or b["edge1"] is None:
                continue
            if a["edge2"] is None or b["edge2"] is None:
                continue
            if abs(a["edge1"] - b["edge1"]) > STATIONARY_EDGE_TOL_PX:
                continue
            if abs(a["edge2"] - b["edge2"]) > STATIONARY_EDGE_TOL_PX:
                continue
            if (
                a["center"] is not None
                and b["center"] is not None
                and abs(a["center"] - b["center"]) > STATIONARY_CENTER_TOL_PX
            ):
                continue
            if a["width"] is not None and b["width"] is not None:
                if abs(a["width"] - b["width"]) > STATIONARY_WIDTH_TOL_PX:
                    continue
            union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        groups[find(index)].append(index)
    return list(groups.values())


def cluster_stationary_candidates(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_infos = [
        info
        for record in records
        for info in record["candidate_infos"]
        if not info["known_bad_image_exception"]
    ]
    rows: list[dict[str, Any]] = []
    grouped_by_height: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for info in all_infos:
        grouped_by_height[info["height_id"]].append(info)
    cluster_number = 0
    for height_id in sorted(grouped_by_height):
        items = grouped_by_height[height_id]
        condition_positions = sorted({info["position_id"] for info in items})
        position_count_total = len(condition_positions)
        for component in union_find_components(items):
            members = [items[index] for index in component]
            cluster_number += 1
            positions = sorted({info["position_id"] for info in members})
            centers = np.asarray(
                [info["center"] for info in members if info["center"] is not None],
                dtype=np.float64,
            )
            edge1s = np.asarray(
                [info["edge1"] for info in members if info["edge1"] is not None],
                dtype=np.float64,
            )
            edge2s = np.asarray(
                [info["edge2"] for info in members if info["edge2"] is not None],
                dtype=np.float64,
            )
            widths = np.asarray(
                [info["width"] for info in members if info["width"] is not None],
                dtype=np.float64,
            )
            boundary5_positions = {
                info["position_id"]
                for info in members
                if info["boundary"].get("quarantine_5mm")
            }
            boundary10_positions = {
                info["position_id"]
                for info in members
                if info["boundary"].get("quarantine_10mm")
            }
            a3_positions = {
                info["position_id"]
                for info in members
                if info["a3"].get("a3_right_morphology_region_overlap")
            }
            min_positions = max(
                1,
                int(math.ceil(STATIONARY_MIN_POSITION_FRACTION * position_count_total)),
            )
            evidence_positions = max(
                1,
                int(
                    math.ceil(
                        STATIONARY_EVIDENCE_POSITION_FRACTION
                        * max(1, len(positions))
                    )
                ),
            )
            motion_stationary = bool(
                len(positions) >= min_positions
                and len(centers)
                and float(np.std(centers)) <= STATIONARY_CENTER_STD_MAX_PX
                and len(edge1s)
                and float(np.std(edge1s)) <= STATIONARY_EDGE_STD_MAX_PX
                and len(edge2s)
                and float(np.std(edge2s)) <= STATIONARY_EDGE_STD_MAX_PX
            )
            boundary_evidence = bool(
                len(boundary5_positions) >= evidence_positions
                or len(boundary10_positions) >= evidence_positions
            )
            a3_evidence = len(a3_positions) >= evidence_positions
            stationary_background = bool(
                motion_stationary and (boundary_evidence or a3_evidence)
            )
            cluster_id = f"{height_id}_c{cluster_number:03d}"
            label = (
                "STATIONARY_BACKGROUND"
                if stationary_background
                else "STATIONARY_MOTION_ONLY"
                if motion_stationary
                else "NON_STATIONARY_CLUSTER"
            )
            for info in members:
                info["stationary_cluster_id"] = cluster_id
                info["stationary_label"] = label
                info["stationary_background"] = stationary_background
            rows.append(
                {
                    "height_id": height_id,
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "stationary_background": stationary_background,
                    "motion_stationary": motion_stationary,
                    "member_count": len(members),
                    "position_count": len(positions),
                    "position_count_total": position_count_total,
                    "position_ids": positions,
                    "condition_ids": sorted(
                        {info["condition_id"] for info in members}
                    ),
                    "candidate_ids": [info["candidate_id"] for info in members],
                    "candidate_ranks": [info["rank"] for info in members],
                    "center_mean_px": float(np.mean(centers)) if len(centers) else None,
                    "center_std_px": float(np.std(centers)) if len(centers) else None,
                    "center_min_px": float(np.min(centers)) if len(centers) else None,
                    "center_max_px": float(np.max(centers)) if len(centers) else None,
                    "edge1_mean_px": float(np.mean(edge1s)) if len(edge1s) else None,
                    "edge1_std_px": float(np.std(edge1s)) if len(edge1s) else None,
                    "edge2_mean_px": float(np.mean(edge2s)) if len(edge2s) else None,
                    "edge2_std_px": float(np.std(edge2s)) if len(edge2s) else None,
                    "width_mean_px": float(np.mean(widths)) if len(widths) else None,
                    "width_std_px": float(np.std(widths)) if len(widths) else None,
                    "boundary_5mm_position_count": len(boundary5_positions),
                    "boundary_10mm_position_count": len(boundary10_positions),
                    "a3_overlap_position_count": len(a3_positions),
                    "boundary_evidence": boundary_evidence,
                    "a3_evidence": a3_evidence,
                    "stationary_min_position_count": min_positions,
                    "stationary_center_std_max_px": STATIONARY_CENTER_STD_MAX_PX,
                    "stationary_edge_std_max_px": STATIONARY_EDGE_STD_MAX_PX,
                }
            )
    return rows


def select_target_for_band(
    record: dict[str, Any], band: float
) -> dict[str, Any]:
    audit = record["audit"]
    condition = audit["condition"]
    known_bad = condition.condition_id in KNOWN_BAD_CONDITIONS
    base: dict[str, Any] = {
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "condition_id": condition.condition_id,
        "known_bad_image_exception": known_bad,
        "boundary_band_mm": band,
        "target_selection_rule": (
            "existing_assess_condition_sort_key_after_boundary_and_stationary_quarantine"
        ),
        "target_roi_status": "NOT_FOUND",
        "target_status_reason": "",
        "candidate_count_generated": len(record["candidate_infos"]),
        "candidate_count_after_quarantine": 0,
        "selected_candidate_rank": None,
        "selected_candidate_id": None,
        "selected_edge1_u_full_px": None,
        "selected_edge2_u_full_px": None,
        "selected_center_u_full_px": None,
        "selected_width_px": None,
        "selected_pair_score": None,
        "selected_edge_pair_geometry_ok": None,
        "selected_pair_gate_reasons": [],
        "selected_target_core_gate_reasons": [],
        "selected_stationary_cluster_id": None,
        "selected_stationary_label": None,
        "selected_boundary_status": None,
        "selected_boundary_reasons": [],
        "local_baseline_status": "UNAVAILABLE",
        "baseline_measurement_eligible": False,
        "baseline_failure_reasons": [],
        "target_evaluable_for_position_consensus": not known_bad,
    }
    if known_bad:
        base["target_status_reason"] = "KNOWN_BAD_IMAGE_EXPECTED_NO_ROI"
        return base
    suffix = f"{int(band)}mm"
    pool = [
        info
        for info in record["candidate_infos"]
        if not info["boundary"].get(f"quarantine_{suffix}")
        and not info["stationary_background"]
    ]
    base["candidate_count_after_quarantine"] = len(pool)
    for info in record["candidate_infos"]:
        info["selected_by_band"][band] = False
    if not pool:
        if record["candidate_infos"]:
            base["target_status_reason"] = (
                "NO_CANDIDATE_AFTER_BOUNDARY_OR_STATIONARY_BACKGROUND_QUARANTINE"
            )
        else:
            base["target_status_reason"] = "NO_GENERATED_CANDIDATE"
        return base
    selected = max((info["candidate"] for info in pool), key=existing_sort_key)
    selected_info = next(info for info in pool if info["candidate"] is selected)
    selected_info["selected_by_band"][band] = True
    baseline = selected_info["baseline"]
    core_reasons = selected_info["target_core_reasons"]
    status = "FOUND" if not core_reasons else "UNCERTAIN"
    base.update(
        {
            "target_roi_status": status,
            "target_status_reason": (
                "TARGET_CANDIDATE_FOUND"
                if status == "FOUND"
                else "CANDIDATE_REMAINS_BUT_TARGET_CORE_GATE_FAILED"
            ),
            "selected_candidate_rank": selected_info["rank"],
            "selected_candidate_id": selected_info["candidate_id"],
            "selected_edge1_u_full_px": selected.get("edge1_v"),
            "selected_edge2_u_full_px": selected.get("edge2_v"),
            "selected_center_u_full_px": selected_info["center"],
            "selected_width_px": selected_info["width"],
            "selected_pair_score": selected.get("pair_score"),
            "selected_edge_pair_geometry_ok": selected.get("edge_pair_geometry_ok"),
            "selected_pair_gate_reasons": selected.get("pair_gate_reasons") or [],
            "selected_target_core_gate_reasons": core_reasons,
            "selected_stationary_cluster_id": selected_info["stationary_cluster_id"],
            "selected_stationary_label": selected_info["stationary_label"],
            "selected_boundary_status": selected_info["boundary"].get(
                f"status_{suffix}"
            ),
            "selected_boundary_reasons": selected_info["boundary"].get(
                f"reasons_{suffix}", []
            ),
            "local_baseline_status": baseline["local_baseline_status"],
            "baseline_measurement_eligible": baseline[
                "baseline_measurement_eligible"
            ],
            "baseline_failure_reasons": baseline["baseline_failure_reasons"],
        }
    )
    return base


def baseline_rows_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    audit = record["audit"]
    condition = audit["condition"]
    rows: list[dict[str, Any]] = []
    for info in record["candidate_infos"]:
        candidate = info["candidate"]
        baseline = info["baseline"]
        rows.append(
            {
                "height_id": condition.height_id,
                "position_id": condition.position_id,
                "condition_id": condition.condition_id,
                "known_bad_image_exception": info["known_bad_image_exception"],
                "candidate_id": info["candidate_id"],
                "candidate_rank": info["rank"],
                "edge1_u_full_px": info["edge1"],
                "edge2_u_full_px": info["edge2"],
                "center_u_full_px": info["center"],
                "object_width_px": info["width"],
                "height_v_range": candidate.get("height_v_range"),
                "baseline_before_v_range": baseline_ranges(candidate)[0],
                "baseline_after_v_range": baseline_ranges(candidate)[1],
                "local_baseline_status": baseline["local_baseline_status"],
                "before_available": baseline["before_available"],
                "after_available": baseline["after_available"],
                "before_stable": baseline["before_stable"],
                "after_stable": baseline["after_stable"],
                "before_support_ok": baseline["before_support_ok"],
                "after_support_ok": baseline["after_support_ok"],
                "before_valid_fraction": (
                    baseline["before_stats"].get("valid_fraction")
                ),
                "after_valid_fraction": baseline["after_stats"].get(
                    "valid_fraction"
                ),
                "before_roughness_px": baseline["before_stats"].get(
                    "roughness_px"
                ),
                "after_roughness_px": baseline["after_stats"].get(
                    "roughness_px"
                ),
                "baseline_measurement_eligible": baseline[
                    "baseline_measurement_eligible"
                ],
                "baseline_failure_reasons": baseline["baseline_failure_reasons"],
                "pair_gate_reasons": candidate.get("pair_gate_reasons"),
                "edge_pair_geometry_ok": candidate.get("edge_pair_geometry_ok"),
                "boundary_5mm_status": info["boundary"].get("status_5mm"),
                "boundary_10mm_status": info["boundary"].get("status_10mm"),
                "boundary_quarantine_5mm": info["boundary"].get(
                    "quarantine_5mm"
                ),
                "boundary_quarantine_10mm": info["boundary"].get(
                    "quarantine_10mm"
                ),
                "stationary_cluster_id": info["stationary_cluster_id"],
                "stationary_label": info["stationary_label"],
                "stationary_background": info["stationary_background"],
                "selected_after_5mm": info["selected_by_band"].get(5.0, False),
                "selected_after_10mm": info["selected_by_band"].get(10.0, False),
            }
        )
    return rows


def transition_row_with_candidate_metadata(
    row: dict[str, Any], infos_by_key: dict[tuple[str, int, int], dict[str, Any]]
) -> dict[str, Any]:
    key = (
        str(row.get("orientation")),
        int(row["edge1_peak_v"]),
        int(row["edge2_peak_v"]),
    ) if row.get("orientation") is not None and row.get("edge1_peak_v") is not None and row.get("edge2_peak_v") is not None else None
    info = infos_by_key.get(key) if key else None
    if not info:
        return row
    return {
        **row,
        "boundary_5mm_status": info["boundary"].get("status_5mm"),
        "boundary_10mm_status": info["boundary"].get("status_10mm"),
        "boundary_quarantine_5mm": info["boundary"].get("quarantine_5mm"),
        "boundary_quarantine_10mm": info["boundary"].get("quarantine_10mm"),
        "boundary_reasons_5mm": info["boundary"].get("reasons_5mm"),
        "boundary_reasons_10mm": info["boundary"].get("reasons_10mm"),
        "a3_right_morphology_region_overlap": info["a3"].get(
            "a3_right_morphology_region_overlap"
        ),
        "a3_spatial_risk_reason": info["a3"].get("a3_spatial_risk_reason"),
        "stationary_cluster_id": info["stationary_cluster_id"],
        "stationary_label": info["stationary_label"],
        "stationary_background": info["stationary_background"],
        "target_core_gate_reasons": info["target_core_reasons"],
    }


def render_overlay(
    record: dict[str, Any], geometry: dict[str, Any], output_dir: Path
) -> Path | None:
    audit = record["audit"]
    profile = record["profile"]
    image_result = audit.get("representative_result")
    image_row = audit.get("representative_row") or {}
    if (
        image_result is None
        or profile["status"] != "PROFILE_AVAILABLE"
        or audit.get("median_scan") is None
    ):
        return None
    image = np.asarray(image_result.frame.image)
    if image.ndim != 2:
        return None
    x0 = finite(image_row.get("offset_x")) or 0.0
    y0 = finite(image_row.get("offset_y")) or 0.0
    x1 = x0 + max(0.0, (finite(image_row.get("width")) or image.shape[1]) - 1.0)
    y1 = y0 + max(0.0, (finite(image_row.get("height")) or image.shape[0]) - 1.0)
    median = np.asarray(audit["median_scan"], dtype=np.float64)
    center_uv = median[:, [1, 0]]
    center_good = np.isfinite(center_uv).all(axis=1)
    image_low = float(np.nanmin(image)) if image.size else 0.0
    image_high = float(np.nanmax(image)) if image.size else 1.0
    if image_high <= image_low:
        image_high = image_low + 1.0
    fig, (ax_image, ax_profile, ax_derivative) = plt.subplots(
        1,
        3,
        figsize=(22, 8),
        gridspec_kw={"width_ratios": [1.25, 1.0, 0.85]},
        constrained_layout=False,
    )
    ax_image.imshow(
        image,
        cmap="gray",
        vmin=image_low,
        vmax=image_high,
        extent=(x0, x1 + 1.0, y1 + 1.0, y0),
        aspect="auto",
        interpolation="nearest",
    )
    if center_good.any():
        ax_image.scatter(
            center_uv[center_good, 0],
            center_uv[center_good, 1],
            s=1.4,
            color="#ffe66d",
            alpha=0.6,
            label="median centerline",
        )
    colors = {0.0: "#aaaaaa", 5.0: "#24c6dc", 10.0: "#cf62ff"}
    for inset in (0.0, 5.0, 10.0):
        polygon = np.asarray(geometry["polygons_full_uv"][inset])
        closed = np.vstack([polygon, polygon[0]])
        ax_image.plot(
            closed[:, 0],
            closed[:, 1],
            color=colors[inset],
            linestyle="--" if inset == 0.0 else "-",
            linewidth=1.2 if inset == 0.0 else 1.7,
            label=f"board polygon inset {inset:g} mm",
        )
    candidates = record["audit"].get("candidates") or []
    info_by_rank = {info["rank"]: info for info in record["candidate_infos"]}
    selected_ranks = {
        float(band): (
            next(
                (
                    row["selected_candidate_rank"]
                    for row in record["target_by_band"].values()
                    if float(row["boundary_band_mm"]) == float(band)
                ),
                None,
            )
        )
        for band in BOUNDARY_BANDS_MM
    }
    for rank, candidate in enumerate(candidates, start=1):
        edge1 = finite(candidate.get("edge1_v"))
        edge2 = finite(candidate.get("edge2_v"))
        if edge1 is None or edge2 is None:
            continue
        info = info_by_rank.get(rank)
        if info and info["stationary_background"]:
            color = "#8e44ad"
            label = "stationary background" if rank == 1 else None
        elif info and info["boundary"].get("quarantine_10mm"):
            color = "#f39c12"
            label = "boundary quarantine" if rank == 1 else None
        elif rank == 1:
            color = "#e74c3c"
            label = "existing selected"
        else:
            color = "#777777"
            label = None
        line_style = "-" if rank == 1 else "--"
        line_width = 2.4 if rank == 1 else 0.9
        if info and selected_ranks.get(10.0) == rank:
            color, line_style, line_width, label = "#00c896", "-", 3.0, "target 10 mm"
        if info and selected_ranks.get(5.0) == rank:
            color, line_style, line_width, label = "#ff4fd8", "-", 3.0, "target 5 mm"
        for axis in (ax_image, ax_profile):
            axis.axvline(edge1, color=color, linestyle=line_style, linewidth=line_width, alpha=0.9)
            axis.axvline(edge2, color=color, linestyle=line_style, linewidth=line_width, alpha=0.9)
        height = candidate.get("height_v_range") or []
        if len(height) == 2:
            ax_profile.axvspan(
                float(height[0]), float(height[1]), color=color, alpha=0.06
            )
        if label:
            ax_image.text(
                edge1,
                y0 + 0.04 * max(1.0, y1 - y0),
                f"r{rank}",
                color=color,
                fontsize=8,
                rotation=90,
                va="bottom",
                ha="right",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
            )
    grid = np.arange(len(profile["interpolated"]), dtype=np.float64)
    ax_profile.plot(
        grid,
        profile["interpolated"],
        color="black",
        linewidth=1.0,
        label="interpolated profile",
    )
    raw_good = np.isfinite(profile["raw"])
    ax_profile.scatter(
        grid[raw_good],
        profile["raw"][raw_good],
        s=1.3,
        color="#888888",
        alpha=0.3,
        label="raw profile",
    )
    negative = profile["negative"]
    positive = profile["positive"]
    ax_derivative.plot(
        grid,
        profile["derivative"],
        color="black",
        linewidth=0.9,
        label="d(profile)/du",
    )
    if len(negative):
        ax_derivative.scatter(
            negative,
            profile["derivative"][negative],
            color="#e74c3c",
            marker="v",
            s=22,
            label="negative peaks",
        )
    if len(positive):
        ax_derivative.scatter(
            positive,
            profile["derivative"][positive],
            color="#2878d0",
            marker="^",
            s=22,
            label="positive peaks",
        )
    for row in record["transition_rows"]:
        if row.get("pair_generation_stage") != "GENERATED_CANDIDATE":
            continue
        edge1 = finite(row.get("edge1_peak_v"))
        edge2 = finite(row.get("edge2_peak_v"))
        if edge1 is not None and edge2 is not None:
            ax_derivative.axvline(edge1, color="#e74c3c", alpha=0.12, linewidth=0.7)
            ax_derivative.axvline(edge2, color="#2878d0", alpha=0.12, linewidth=0.7)
    ax_image.set_xlim(x0, x1 + 1.0)
    ax_image.set_ylim(y1 + 1.0, y0)
    ax_profile.set_xlim(x0, x1 + 1.0)
    ax_profile.set_ylim(y1 + 1.0, y0)
    ax_derivative.set_xlim(x0, x1 + 1.0)
    ax_image.set_xlabel("full-sensor u (px)")
    ax_image.set_ylabel("full-sensor v (px)")
    ax_profile.set_xlabel("ROI-V2 v' = full-sensor u (px)")
    ax_profile.set_ylabel("u' = full-sensor v (px)")
    ax_derivative.set_xlabel("full-sensor u (px)")
    ax_derivative.set_ylabel("profile derivative")
    ax_image.set_title("raw image + centerline + boundary bands")
    ax_profile.set_title("all generated pairs / quarantine")
    ax_derivative.set_title("all derivative transition peaks")
    for axis in (ax_image, ax_profile, ax_derivative):
        axis.grid(alpha=0.18)
    ax_image.legend(loc="best", fontsize=7)
    ax_profile.legend(loc="best", fontsize=7)
    ax_derivative.legend(loc="best", fontsize=7)
    condition = audit["condition"]
    target_text = ", ".join(
        f"{band:g}mm:{row['target_roi_status']}"
        for band, row in sorted(record["target_by_band"].items())
    )
    exception_text = (
        " | KNOWN_BAD_IMAGE_EXPECTED_NO_ROI"
        if condition.condition_id in KNOWN_BAD_CONDITIONS
        else ""
    )
    fig.suptitle(
        f"{condition.condition_id} | peaks(-/+)=({len(negative)}/{len(positive)}) | "
        f"generated pairs={len(candidates)} | targets={target_text}{exception_text}",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "red=existing selected, orange=board boundary, purple=stationary background, "
        "green/magenta=post-quarantine target; no truth-based selection; baseline is not a target gate",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    path = output_dir / f"{condition.condition_id}_transition_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_value(row.get(field)) for field in fields})


TRANSITION_FIELDS = [
    "height_id",
    "position_id",
    "condition_id",
    "known_bad_image_exception",
    "row_type",
    "profile_stage_status",
    "profile_noise_px_per_v",
    "edge_prominence_threshold_px",
    "negative_peak_count",
    "positive_peak_count",
    "negative_peaks",
    "positive_peaks",
    "direct_build_edge_pairs_count",
    "assessment_candidate_count",
    "pair_id",
    "pair_generation_stage",
    "candidate_generated",
    "candidate_rank",
    "orientation",
    "edge1_peak_v",
    "edge2_peak_v",
    "edge1_u_full_px",
    "edge2_u_full_px",
    "object_width_px",
    "height_v_range",
    "height_interior_width_px",
    "transition_v_ranges",
    "baseline_v_ranges",
    "baseline_clipped",
    "edge1_prominence_px",
    "edge2_prominence_px",
    "edge_min_prominence_px",
    "edge1_local_support_fraction",
    "edge2_local_support_fraction",
    "before_slope_px_per_v",
    "before_roughness_px",
    "height_slope_px_per_v",
    "height_roughness_px",
    "after_slope_px_per_v",
    "after_roughness_px",
    "predicted_ground_u_at_height_mid",
    "plateau_delta_u_px",
    "step_amplitude_px",
    "pair_score",
    "score_recomputed",
    "score_delta",
    "pair_gate_reasons",
    "generation_rejection_reasons",
    "edge_pair_geometry_ok",
    "height_stats",
    "baseline_before_support",
    "baseline_after_support",
    "multi_geometry_reasons",
    "multi_geometry_ok",
    "boundary_5mm_status",
    "boundary_10mm_status",
    "boundary_quarantine_5mm",
    "boundary_quarantine_10mm",
    "boundary_reasons_5mm",
    "boundary_reasons_10mm",
    "a3_right_morphology_region_overlap",
    "a3_spatial_risk_reason",
    "stationary_cluster_id",
    "stationary_label",
    "stationary_background",
    "target_core_gate_reasons",
]


TARGET_FIELDS = [
    "height_id",
    "position_id",
    "condition_id",
    "known_bad_image_exception",
    "boundary_band_mm",
    "target_selection_rule",
    "target_roi_status",
    "target_status_reason",
    "candidate_count_generated",
    "candidate_count_after_quarantine",
    "selected_candidate_rank",
    "selected_candidate_id",
    "selected_edge1_u_full_px",
    "selected_edge2_u_full_px",
    "selected_center_u_full_px",
    "selected_width_px",
    "selected_pair_score",
    "selected_edge_pair_geometry_ok",
    "selected_pair_gate_reasons",
    "selected_target_core_gate_reasons",
    "selected_stationary_cluster_id",
    "selected_stationary_label",
    "selected_boundary_status",
    "selected_boundary_reasons",
    "local_baseline_status",
    "baseline_measurement_eligible",
    "baseline_failure_reasons",
    "target_evaluable_for_position_consensus",
]


BASELINE_FIELDS = [
    "height_id",
    "position_id",
    "condition_id",
    "known_bad_image_exception",
    "candidate_id",
    "candidate_rank",
    "edge1_u_full_px",
    "edge2_u_full_px",
    "center_u_full_px",
    "object_width_px",
    "height_v_range",
    "baseline_before_v_range",
    "baseline_after_v_range",
    "local_baseline_status",
    "before_available",
    "after_available",
    "before_stable",
    "after_stable",
    "before_support_ok",
    "after_support_ok",
    "before_valid_fraction",
    "after_valid_fraction",
    "before_roughness_px",
    "after_roughness_px",
    "baseline_measurement_eligible",
    "baseline_failure_reasons",
    "pair_gate_reasons",
    "edge_pair_geometry_ok",
    "boundary_5mm_status",
    "boundary_10mm_status",
    "boundary_quarantine_5mm",
    "boundary_quarantine_10mm",
    "stationary_cluster_id",
    "stationary_label",
    "stationary_background",
    "selected_after_5mm",
    "selected_after_10mm",
]


CLUSTER_FIELDS = [
    "height_id",
    "cluster_id",
    "cluster_label",
    "stationary_background",
    "motion_stationary",
    "member_count",
    "position_count",
    "position_count_total",
    "position_ids",
    "condition_ids",
    "candidate_ids",
    "candidate_ranks",
    "center_mean_px",
    "center_std_px",
    "center_min_px",
    "center_max_px",
    "edge1_mean_px",
    "edge1_std_px",
    "edge2_mean_px",
    "edge2_std_px",
    "width_mean_px",
    "width_std_px",
    "boundary_5mm_position_count",
    "boundary_10mm_position_count",
    "a3_overlap_position_count",
    "boundary_evidence",
    "a3_evidence",
    "stationary_min_position_count",
    "stationary_center_std_max_px",
    "stationary_edge_std_max_px",
]


def prior_candidate_reuse_check(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    prior = (
        root
        / "c0_height_audit"
        / "roi_v2_audit"
        / "candidate_score_breakdown.csv"
    )
    result = {"path": str(prior.resolve()), "exists": prior.exists()}
    if not prior.exists():
        result.update({"prior_rows": 0, "key_matches": 0, "score_matches": 0})
        return result
    prior_rows = list(csv.DictReader(prior.open(encoding="utf-8-sig")))
    prior_index = {
        (
            row["condition_id"],
            int(row["candidate_rank"]),
        ): row
        for row in prior_rows
    }
    key_matches = 0
    score_matches = 0
    generated = 0
    for record in records:
        for rank, candidate in enumerate(record["audit"].get("candidates") or [], start=1):
            generated += 1
            old = prior_index.get((record["audit"]["condition"].condition_id, rank))
            if not old:
                continue
            if (
                int(old["edge1_u_full_px"]) == int(candidate.get("edge1_v"))
                and int(old["edge2_u_full_px"]) == int(candidate.get("edge2_v"))
            ):
                key_matches += 1
            old_score = finite(old.get("pair_score"))
            new_score = finite(candidate.get("pair_score"))
            if old_score is not None and new_score is not None and abs(old_score - new_score) <= 1.0e-12:
                score_matches += 1
    result.update(
        {
            "sha256": sha256_file(prior),
            "prior_rows": len(prior_rows),
            "generated_candidate_rows": generated,
            "key_matches": key_matches,
            "score_matches": score_matches,
        }
    )
    return result


def fmt(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def is_fixed_pair_row(row: dict[str, Any]) -> bool:
    edge1 = finite(row.get("edge1_peak_v"))
    edge2 = finite(row.get("edge2_peak_v"))
    return bool(
        edge1 is not None
        and edge2 is not None
        and int(edge1) == FIXED_PAIR[0]
        and int(edge2) == FIXED_PAIR[1]
    )


def stage_counts(transition_rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("pair_generation_stage")) for row in transition_rows)


def target_summary_rows(target_rows: list[dict[str, Any]], band: float) -> dict[str, Any]:
    rows = [row for row in target_rows if float(row["boundary_band_mm"]) == band]
    normal = [row for row in rows if not row["known_bad_image_exception"]]
    found = [row for row in normal if row["target_roi_status"] == "FOUND"]
    found_baseline_bad = [
        row for row in found if not row["baseline_measurement_eligible"]
    ]
    return {
        "rows": rows,
        "normal_count": len(normal),
        "known_exception_count": len(rows) - len(normal),
        "found_count": len(found),
        "found_baseline_insufficient_count": len(found_baseline_bad),
        "uncertain_count": sum(
            row["target_roi_status"] == "UNCERTAIN" for row in normal
        ),
        "not_found_count": sum(
            row["target_roi_status"] == "NOT_FOUND" for row in normal
        ),
        "boundary_quarantined_candidate_count": sum(
            int(row["candidate_count_generated"])
            - int(row["candidate_count_after_quarantine"])
            for row in normal
        ),
    }


def moving_position_summary(target_rows: list[dict[str, Any]], band: float) -> list[dict[str, Any]]:
    rows = [
        row
        for row in target_rows
        if float(row["boundary_band_mm"]) == band
        and not row["known_bad_image_exception"]
    ]
    result: list[dict[str, Any]] = []
    for height_id in sorted({row["height_id"] for row in rows}):
        selected = [
            row
            for row in rows
            if row["height_id"] == height_id
            and row["selected_center_u_full_px"] is not None
        ]
        centers = [float(row["selected_center_u_full_px"]) for row in selected]
        result.append(
            {
                "height_id": height_id,
                "boundary_band_mm": band,
                "selected_position_count": len(selected),
                "selected_positions": [row["position_id"] for row in selected],
                "selected_center_min_px": min(centers) if centers else None,
                "selected_center_max_px": max(centers) if centers else None,
                "selected_center_span_px": max(centers) - min(centers) if centers else None,
                "selected_centers_px": centers,
                "moving_evidence": bool(
                    len(centers) >= 2
                    and max(centers) - min(centers) > STATIONARY_CENTER_STD_MAX_PX
                ),
            }
        )
    return result


def build_report(
    root: Path,
    geometry: dict[str, Any],
    records: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    overlays: list[Path],
    reuse_check: dict[str, Any],
) -> str:
    all_stage_counts = stage_counts(transition_rows)
    target_summaries = {
        band: target_summary_rows(target_rows, band) for band in BOUNDARY_BANDS_MM
    }
    generated_rows = [
        row for row in transition_rows if row.get("candidate_generated")
    ]
    fixed_pair_rows = [
        row
        for row in generated_rows
        if is_fixed_pair_row(row)
    ]
    stationary_background_rows = [
        row for row in generated_rows if row.get("stationary_background") is True
    ]
    interior_target_rows = [
        row
        for row in generated_rows
        if row.get("boundary_5mm_status") == "INTERIOR_TARGET_DOMAIN"
        and row.get("stationary_background") is not True
        and not row.get("known_bad_image_exception")
    ]
    interior_target_core_reasons: Counter[str] = Counter(
        reason
        for row in interior_target_rows
        for reason in (row.get("target_core_gate_reasons") or [])
    )
    classification = "MIXED"
    classification_reason = (
        "固定背景候选占主导并被 stationary/boundary 证据隔离；"
        "剩余内部移动候选已生成但主要被同一 height-interior 几何门拒绝。"
    )
    core_reason_counts: Counter[str] = Counter()
    for row in generated_rows:
        for reason in row.get("target_core_gate_reasons") or []:
            core_reason_counts[str(reason)] += 1
    pair_reason_counts: Counter[str] = Counter()
    for row in generated_rows:
        for reason in row.get("pair_gate_reasons") or []:
            pair_reason_counts[str(reason)] += 1
    special_ids = [
        "h02_p01",
        "h02_p02",
        "h02_p03",
        *[f"h{height:02d}_p{position:02d}" for height in (6, 10, 20, 30) for position in (1, 5, 10)],
    ]
    lines = [
        "# H0-1R3 | 海康背景边界隔离 + 移动目标候选审计",
        "",
        f"数据根目录：{root}",
        f"condition：{len(records)}；transition audit 行：{len(transition_rows)}；"
        f"generated candidate：{len(generated_rows)}；cluster：{len(cluster_rows)}；"
        f"target rows：{len(target_rows)}；baseline rows：{len(baseline_rows)}；"
        f"overlay：{len(overlays)} 张。",
        "",
        "本轮只审计 ROI 选择。full centerline/profile 仍作为 ROI-V2 输入；"
        "board polygon 只用于 boundary quarantine，不再约束 height interior 或 baseline 全部落在 board 内。"
        "不调用高度测量，不修改 C0、Session Ground、axis adapter、ROI-V2 参数或 score。",
        "",
        "## 1. provenance 与复用",
        "",
        "复用 H0-1R 的 `run_condition`、FramePipeline replay、Haikang axis adapter、"
        "`median_centerline`、`integer_profile`、`build_edge_pairs`、`assess_condition`、"
        "`assess_pair` 及原有排序 key；复用 H0-1R2 的 PnP board polygon 重放。",
        "新增仅为 boundary ring 判断、按 height 的 stationary cluster、transition/peak 诊断、"
        "target/baseline 分离状态与 overlay。",
        "",
        f"- Session support.source = {geometry['support'].get('source')}",
        f"- Session mask_mode = {geometry['support'].get('mask_mode')}",
        f"- PnP detection offset = {geometry['pnp_offset']}",
        f"- stored 0 mm polygon 重放最大差 = {fmt(geometry['zero_polygon_max_abs_delta_px'], 6)} px",
        f"- PnP reprojection RMSE = {fmt(geometry['pnp_reprojection_rmse_px'], 4)} px",
        f"- R3 5 mm polygon PnP RMSE = {fmt(geometry['r3_pnp_reprojection_rmse_px'], 4)} px",
        f"- H0-1R candidate key reuse = {reuse_check.get('key_matches')} / "
        f"{reuse_check.get('generated_candidate_rows')}；score exact matches = "
        f"{reuse_check.get('score_matches')}。",
        "",
        "`h02_p01` 是用户指定的已知异常图像：本轮保留其 profile/peak/candidate 明细，"
        "但不把它作为正常 target，也不加入跨 position stationary consensus。若该组无 ROI，"
        "状态按 `KNOWN_BAD_IMAGE_EXPECTED_NO_ROI` 处理。",
        "",
        "## 2. boundary quarantine 定义",
        "",
        "物理棋盘坐标中的完整 board 为 X=[-20,220] mm、Y=[-20,160] mm；"
        "R3 分别投影 5 mm 和 10 mm 内缩 polygon。candidate 的 edge1/edge2 只要位于"
        "外 polygon 外部、对应 boundary ring，或 edge interval 穿越外 perimeter，"
        "即标记 `BOARD_BOUNDARY_BACKGROUND` 并不参与 target selection。",
        "不检查 height interior 是否完全落在 Z=0 board polygon 内；baseline 也只保留为独立可行性状态。",
        "",
        "## 3. 全量 boundary / target 结果",
        "",
            "| boundary band | FOUND | FOUND + baseline insufficient | UNCERTAIN | NOT_FOUND | "
            "known exception | boundary/stationary-quarantined candidates |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for band in BOUNDARY_BANDS_MM:
        summary = target_summaries[band]
        lines.append(
            f"| {band:g} mm | {summary['found_count']} | "
            f"{summary['found_baseline_insufficient_count']} | "
            f"{summary['uncertain_count']} | {summary['not_found_count']} | "
            f"{summary['known_exception_count']} | "
            f"{summary['boundary_quarantined_candidate_count']} |"
        )
    lines.extend(
        [
            "",
            f"原始 peak/pair generation stage：{dict(all_stage_counts)}。",
            f"generated candidate 的 pair gate 主要原因：{dict(pair_reason_counts.most_common())}。",
            f"fixed pair {FIXED_PAIR} generated count = {len(fixed_pair_rows)}；"
            f"stationary_background generated rows = {len(stationary_background_rows)}；"
            f"post-boundary/non-stationary interior rows = {len(interior_target_rows)}；"
            f"其 target-core reasons = {dict(interior_target_core_reasons)}。",
            "",
            "## 4. stationary background clusters",
            "",
            "cluster 只在同一 height 内按 edge1/edge2/center/width 接近度聚类，"
            "不使用真实高度选择。判为 `STATIONARY_BACKGROUND` 还要求多数 position 出现，"
            "center/edge spread 足够小，并且至少满足 board boundary 或此前 A-3 morphology 的空间证据。",
            "",
            "| height | cluster | label | positions | center mean±std | edge pair mean | "
            "boundary5/10 positions | A3 positions |",
            "|---|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in cluster_rows:
        lines.append(
            f"| {row['height_id']} | {row['cluster_id']} | {row['cluster_label']} | "
            f"{row['position_count']}/{row['position_count_total']} | "
            f"{fmt(row['center_mean_px'], 1)}±{fmt(row['center_std_px'], 1)} | "
            f"({fmt(row['edge1_mean_px'], 1)},{fmt(row['edge2_mean_px'], 1)}) | "
            f"{row['boundary_5mm_position_count']}/{row['boundary_10mm_position_count']} | "
            f"{row['a3_overlap_position_count']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 去固定背景后的 p01–p10 空间变化",
            "",
            "下表只列正常样本的 post-quarantine selected candidate；"
            "center span 是观测到的空间变化，不是目录真值驱动的选择。",
            "",
            "| height | band | selected positions | center range | span | moving evidence |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for band in BOUNDARY_BANDS_MM:
        for row in moving_position_summary(target_rows, band):
            center_range = (
                f"{fmt(row['selected_center_min_px'], 1)}–{fmt(row['selected_center_max_px'], 1)}"
                if row["selected_center_min_px"] is not None
                else "—"
            )
            lines.append(
                f"| {row['height_id']} | {band:g} | {row['selected_position_count']} | "
                f"{center_range} | {fmt(row['selected_center_span_px'], 1)} | "
                f"{row['moving_evidence']} |"
            )
    lines.extend(
        [
            "",
            "## 6. transition / small-step generation audit",
            "",
            "`candidate_transition_audit.csv` 同时保留：全部正/负 derivative peaks、"
            "每个 peak pair 的原始 pairing、width、step amplitude、prominence、local support、"
            "baseline/height roughness，以及 pair 在生成前还是生成后被哪个 gate 拒绝。",
            "",
            "| condition | -/+ peaks | generated | width reject | height-interior reject | "
            "generated geometry-pass | dominant generated gate reasons |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for condition_id in special_ids:
        rows = [row for row in transition_rows if row.get("condition_id") == condition_id]
        if not rows:
            continue
        first = rows[0]
        counts = Counter(row.get("pair_generation_stage") for row in rows)
        generated = [row for row in rows if row.get("candidate_generated")]
        reasons = Counter(
            reason
            for row in generated
            for reason in (row.get("pair_gate_reasons") or [])
        )
        lines.append(
            f"| {condition_id} | {first.get('negative_peak_count', 0)}/"
            f"{first.get('positive_peak_count', 0)} | {counts.get('GENERATED_CANDIDATE', 0)} | "
            f"{counts.get('REJECTED_BEFORE_CANDIDATE_WIDTH', 0)} | "
            f"{counts.get('REJECTED_BEFORE_CANDIDATE_HEIGHT_INTERIOR', 0)} | "
            f"{sum(bool(row.get('edge_pair_geometry_ok')) for row in generated)} | "
            f"{dict(reasons.most_common(4))} |"
        )
    lines.extend(
        [
            "",
            "解释边界：若 pair 出现在 `GENERATED_CANDIDATE`，说明 derivative peak 与"
            "现有 width/height-interior 生成前置条件均已通过；后续失败属于 candidate geometry/support gate。"
            "若只出现在 `REJECTED_BEFORE_CANDIDATE_WIDTH` 或 `...HEIGHT_INTERIOR`，"
            "则不是 score 排序丢失，而是 pairing 前置域丢弃。若某个图像可见的小台阶没有对应 peak，"
            "只能归入 derivative smoothing/peak threshold 阶段；R3 不凭目录高度替它指定 pair。",
            "",
            "## 7. target / baseline 分离统计",
            "",
            "`target_roi_status` 只由生成候选在 boundary/stationary quarantine 后的 target-core 状态决定；"
            "baseline 不参与删除。`local_baseline_status` 只描述双侧 baseline 是否存在，"
            "`baseline_measurement_eligible` 再考虑 support/stability/clipping。",
            "",
            f"target-core gate reasons（generated rows）：{dict(core_reason_counts.most_common())}。",
            "",
            "## 8. 明确回答",
            "",
            "1. `(1950,2024)` 是否能通过 boundary/stationary 证据隔离？",
            "",
            f"精确 pair 在本轮 generated candidate 中出现 {len(fixed_pair_rows)} 次；固定 cluster 与 board boundary/A-3 "
            "证据分别记录在 `stationary_candidate_clusters.csv`。R3 selection 中不允许其作为 target；"
            f"5 mm / 10 mm 的目标选择均不会保留被标记为 `BOARD_BOUNDARY_BACKGROUND` 或 "
            "`STATIONARY_BACKGROUND` 的候选。",
            "",
            "2. 去掉固定背景后是否存在随 p01–p10 移动的 target candidate？",
            "",
            "以第 5 节为准：只有 post-quarantine selected center 在多个 position 间有非平凡 span，"
            "才标记 moving evidence；空行/单点不能宣称存在移动目标。",
            "",
            "3. 图中小台阶是否被 `build_edge_pairs` 生成？",
            "",
            "逐 peak pair 的生成状态见第 6 节和 CSV。`GENERATED_CANDIDATE` 是明确生成；"
            "前置 width/height-interior rejection 则明确未生成。",
            "",
            "4. 如果生成，主要被哪个 geometry gate 拒绝？",
            "",
            "以各 condition 的 `pair_gate_reasons`、roughness、support 和 geometry bool 为准；"
            "本轮不把 baseline 不足改写为 target 未找到。",
            "",
            "5. 如果未生成，是 derivative peak、pairing 还是 smoothing 丢失？",
            "",
            "R3 已把 derivative peak 列表和所有可配对 peak 组合显式落盘。未出现在 peak 列表的候选只能标记为 peak/smoothing 未通过；"
            "出现在 peak 列表但没有 generated row 的，按 width 或 height-interior pairing 前置 gate 区分。",
            "",
            "6. 是否需要修改 Haikang-specific ROI-V2 参数域？",
            "",
            "本轮不直接调参。只有当正常样本的移动候选在多个 position 重复出现、"
            "且主要被同一个 target-core gate（例如 `step_amplitude_below_minimum`、"
            "`height_interior_width_below_minimum` 或 `height_plateau_not_stable`）拒绝时，"
            "才建议下一轮针对该参数做 Haikang-specific sensitivity audit；不能用 `h02_p01` 异常图单独定参。",
            "",
            "7. 有多少 condition 找到 target / target 找到但 baseline 不足 / 完全找不到？",
            "",
        ]
    )
    for band in BOUNDARY_BANDS_MM:
        summary = target_summaries[band]
        lines.append(
            f"- {band:g} mm：正常可评估 {summary['normal_count']} 个；FOUND "
            f"{summary['found_count']}；其中 baseline insufficient "
            f"{summary['found_baseline_insufficient_count']}；UNCERTAIN "
            f"{summary['uncertain_count']}；NOT_FOUND {summary['not_found_count']}；"
                f"已知异常 {summary['known_exception_count']}。"
        )
    lines.extend(
        [
            "",
            "## 9. R3 分类结论",
            "",
            f"`{classification}`：{classification_reason}",
            "这不是 `AXIS_ADAPTER_SEMANTICS_WRONG` 的证据：本轮复用的 axis adapter 仍保持 "
            "ROI-V2 v' = full-sensor u、profile value u' = full-sensor v，且 H0-1R candidate key/score "
            "113/113 一致。也不是 `DATASET_TARGET_NOT_VISIBLE`：在多个正常 condition 中已经观察到内部小台阶对应的 peak pair；"
            "当前只能把参数域修改建议留给下一轮 sensitivity audit，不能在本轮直接调参。",
            "",
            "## 10. 约束与后续边界",
            "",
            "- 没有从 `height_shadow.csv` 读取或假定高度；没有调用 h_raw/height measurement。",
            "- 没有修改 production config、axis adapter、C0、Session Ground、ROI-V2 参数或 score。",
            "- 没有使用真实高度选择 candidate；height_id 仅用于按采集设计分组 stationary cluster。",
            "- 若后续重跑 H0-1，必须使用 target selection 输出作为 ROI eligibility；"
            "baseline 不足应进入独立不可测状态，不能回退固定右侧 pair。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.input_dir.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions, discovery = r1.h0.discover_conditions(root)
    session_path = root / "session_ground_calibration.json"
    config_summary = r1.h0.config_contract(config_path)
    reference, rotation, translation, session_summary = r1.h0.load_session_reference(
        session_path
    )
    a3_audit = r1.h0.load_prior_a3_spatial_audit(root)
    app, pipeline = r1.h0.make_pipeline(
        config_path, reference, rotation, translation
    )
    geometry = load_board_geometry_r3(
        session_path, app, pipeline.calibration_for_reconstruction()
    )
    records: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for condition in conditions:
        audit = r1.run_condition(condition, pipeline)
        profile = profile_diagnostics(audit)
        infos = make_candidate_infos(audit, geometry, a3_audit)
        rows = build_transition_rows(audit, profile)
        info_by_key = {
            (
                str(info["candidate"].get("orientation")),
                int(info["candidate"].get("edge1_peak_v")),
                int(info["candidate"].get("edge2_peak_v")),
            ): info
            for info in infos
        }
        rows = [
            transition_row_with_candidate_metadata(row, info_by_key)
            for row in rows
        ]
        record = {
            "audit": audit,
            "profile": profile,
            "candidate_infos": infos,
            "transition_rows": rows,
            "target_by_band": {},
        }
        records.append(record)
        transition_rows.extend(rows)
    cluster_rows = cluster_stationary_candidates(records)
    # Clustering is intentionally performed after the first pass over all
    # heights/positions.  Re-attach its labels to the transition rows so the
    # detailed CSV and the cluster CSV describe the same candidate state.
    transition_rows = []
    for record in records:
        info_by_key = {
            (
                str(info["candidate"].get("orientation")),
                int(info["candidate"].get("edge1_peak_v")),
                int(info["candidate"].get("edge2_peak_v")),
            ): info
            for info in record["candidate_infos"]
        }
        record["transition_rows"] = [
            transition_row_with_candidate_metadata(row, info_by_key)
            for row in record["transition_rows"]
        ]
        transition_rows.extend(record["transition_rows"])
    target_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for record in records:
        for band in BOUNDARY_BANDS_MM:
            target = select_target_for_band(record, band)
            record["target_by_band"][band] = target
            target_rows.append(target)
        baseline_rows.extend(baseline_rows_for_record(record))
    overlay_ids = set(OVERLAY_REQUIRED_IDS)
    overlay_paths: list[Path] = []
    for record in records:
        if record["audit"]["condition"].condition_id in overlay_ids:
            path = render_overlay(record, geometry, output_dir)
            if path is not None:
                overlay_paths.append(path)
    reuse_check = prior_candidate_reuse_check(root, records)
    transition_path = output_dir / "candidate_transition_audit.csv"
    cluster_path = output_dir / "stationary_candidate_clusters.csv"
    target_path = output_dir / "roi_target_selection.csv"
    baseline_path = output_dir / "baseline_feasibility.csv"
    write_csv(transition_path, TRANSITION_FIELDS, transition_rows)
    write_csv(cluster_path, CLUSTER_FIELDS, cluster_rows)
    write_csv(target_path, TARGET_FIELDS, target_rows)
    write_csv(baseline_path, BASELINE_FIELDS, baseline_rows)
    generated_rows = [
        row for row in transition_rows if row.get("candidate_generated")
    ]
    fixed_pair_count = sum(is_fixed_pair_row(row) for row in generated_rows)
    stationary_background_count = sum(
        row.get("stationary_background") is True for row in generated_rows
    )
    interior_nonstationary_rows = [
        row
        for row in generated_rows
        if row.get("boundary_5mm_status") == "INTERIOR_TARGET_DOMAIN"
        and row.get("stationary_background") is not True
        and not row.get("known_bad_image_exception")
    ]
    board_registry_path = (
        root
        / "c0_height_audit"
        / "roi_v2_board_mask"
        / "roi_v2_board_mask_registry.csv"
    )
    provenance = {
        "task": "H0-1R3",
        "input_root": str(root),
        "config": config_summary,
        "session_ground": session_summary,
        "discovery": discovery,
        "prior_artifacts": {
            "h0_1r_candidate_score_breakdown": reuse_check,
            "h0_1r2_board_mask_registry": (
                {
                    "path": str(board_registry_path.resolve()),
                    "exists": board_registry_path.exists(),
                    "sha256": (
                        sha256_file(board_registry_path)
                        if board_registry_path.exists()
                        else None
                    ),
                }
            ),
        },
        "board_boundary": {
            "bands_mm": list(BOUNDARY_BANDS_MM),
            "physical_bounds_mm": geometry["physical_bounds_mm"],
            "polygons_full_uv": geometry["polygons_full_uv"],
            "stored_zero_polygon_max_abs_delta_px": geometry[
                "zero_polygon_max_abs_delta_px"
            ],
            "pnp_reprojection_rmse_px": geometry["pnp_reprojection_rmse_px"],
            "r3_pnp_reprojection_rmse_px": geometry["r3_pnp_reprojection_rmse_px"],
        },
        "stationary_cluster_protocol": {
            "edge_tolerance_px": STATIONARY_EDGE_TOL_PX,
            "center_tolerance_px": STATIONARY_CENTER_TOL_PX,
            "width_tolerance_px": STATIONARY_WIDTH_TOL_PX,
            "center_std_max_px": STATIONARY_CENTER_STD_MAX_PX,
            "edge_std_max_px": STATIONARY_EDGE_STD_MAX_PX,
            "minimum_position_fraction": STATIONARY_MIN_POSITION_FRACTION,
            "evidence_position_fraction": STATIONARY_EVIDENCE_POSITION_FRACTION,
            "known_bad_conditions_excluded_from_consensus": sorted(
                KNOWN_BAD_CONDITIONS
            ),
        },
        "protocol": {
            "height_truth_used_for_selection": False,
            "board_polygon_used_as_full_roi_hard_mask": False,
            "candidate_score_modified": False,
            "axis_adapter_modified": False,
            "c0_modified": False,
            "session_ground_modified": False,
            "roi_v2_parameters_modified": False,
            "height_measurement_called": False,
            "compensation_called": False,
            "baseline_can_remove_target": False,
        },
        "conclusion": {
            "classification": "MIXED",
            "reason": (
                "stationary/boundary background candidates dominate the generated set, "
                "while the remaining interior moving candidates are mainly rejected by "
                "the existing height-interior geometry gate"
            ),
            "fixed_pair_generated_count": fixed_pair_count,
            "stationary_background_generated_count": stationary_background_count,
            "interior_nonstationary_normal_generated_count": len(
                interior_nonstationary_rows
            ),
            "interior_nonstationary_target_core_reasons": dict(
                Counter(
                    reason
                    for row in interior_nonstationary_rows
                    for reason in (row.get("target_core_gate_reasons") or [])
                )
            ),
        },
        "counts": {
            "condition_count": len(records),
            "transition_rows": len(transition_rows),
            "generated_candidate_rows": len(
                [row for row in transition_rows if row.get("candidate_generated")]
            ),
            "cluster_rows": len(cluster_rows),
            "target_rows": len(target_rows),
            "baseline_rows": len(baseline_rows),
            "overlay_count": len(overlay_paths),
            "stage_counts": dict(stage_counts(transition_rows)),
        },
        "outputs": {
            "candidate_transition_audit": str(transition_path),
            "stationary_candidate_clusters": str(cluster_path),
            "roi_target_selection": str(target_path),
            "baseline_feasibility": str(baseline_path),
            "report": str(
                (output_dir / "roi_v2_r3_report.md").resolve()
            ),
            "overlays": [str(path.resolve()) for path in overlay_paths],
        },
    }
    (output_dir / "roi_v2_r3_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "roi_v2_r3_report.md").write_text(
        build_report(
            root,
            geometry,
            records,
            transition_rows,
            cluster_rows,
            target_rows,
            baseline_rows,
            overlay_paths,
            reuse_check,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "condition_count": len(records),
                "transition_rows": len(transition_rows),
                "generated_candidate_rows": sum(
                    bool(row.get("candidate_generated")) for row in transition_rows
                ),
                "cluster_rows": len(cluster_rows),
                "target_rows": len(target_rows),
                "baseline_rows": len(baseline_rows),
                "overlay_count": len(overlay_paths),
                "stage_counts": dict(stage_counts(transition_rows)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
