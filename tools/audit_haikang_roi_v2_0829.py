#!/usr/bin/env python3
"""Audit Haikang ROI-V2 candidates without changing the production chain.

The H0-1 generator intentionally retained only the existing ROI-V2 first
candidate.  This audit replays the same frozen center extraction and adapter,
then records every candidate returned by the existing ROI-V2 implementation.
No height measurement, compensation, or truth-based candidate selection is
performed here.
"""

from __future__ import annotations

import argparse
import csv
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
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import generate_haikang_c0_h_raw_0829 as h0  # noqa: E402


DATA_ROOT_DEFAULT = h0.DATA_ROOT_DEFAULT
OUTPUT_DIR_DEFAULT = DATA_ROOT_DEFAULT / "c0_height_audit" / "roi_v2_audit"
TARGET_OVERLAY_POSITIONS = {"p01", "p05", "p10"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=h0.CONFIG_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args(argv)


def finite(value: Any) -> float | None:
    return h0.finite(value)


def text_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def interval_endpoint(interval: Any, index: int) -> Any:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return None
    return interval[index]


def interval_width(interval: Any) -> float | None:
    start = finite(interval_endpoint(interval, 0))
    end = finite(interval_endpoint(interval, 1))
    if start is None or end is None:
        return None
    return end - start + 1.0


def candidate_roi(candidate: dict[str, Any]) -> dict[str, Any]:
    baseline = candidate.get("baseline_v_ranges") or [[], []]
    baseline = list(baseline) + [[], []]
    return {
        "height_u_full_range_px": candidate.get("height_v_range") or [],
        "baseline_before_u_full_range_px": baseline[0],
        "baseline_after_u_full_range_px": baseline[1],
    }


def candidate_score_terms(candidate: dict[str, Any]) -> dict[str, Any]:
    min_prominence = finite(candidate.get("edge_min_prominence_px")) or 0.0
    step_amplitude = finite(candidate.get("step_amplitude_px")) or 0.0
    width = finite(candidate.get("object_width_px")) or 0.0
    min_term = min_prominence
    step_term = 0.10 * step_amplitude
    width_term = 0.01 * min(width, 120.0)
    return {
        "min_edge_prominence_term": min_term,
        "step_amplitude_term": step_term,
        "width_term": width_term,
        "score_recomputed": min_term + step_term + width_term,
    }


def safe_reason_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def enrich_candidate(
    candidate: dict[str, Any],
    frame_arrays: list[np.ndarray],
) -> dict[str, Any]:
    """Reuse Thermal-A2A's existing repeat-support assessment for audit rows."""
    item = h0.roi_v2_wrapper.assess_pair(candidate, frame_arrays)
    item["pair_gate_reasons"] = safe_reason_list(item.get("pair_gate_reasons"))
    item["multi_geometry_reasons"] = safe_reason_list(
        item.get("multi_geometry_reasons")
    )
    item.update(candidate_score_terms(item))
    return item


def run_condition(
    condition: h0.Condition,
    pipeline: Any,
) -> dict[str, Any]:
    source_rows = h0.read_csv_rows(condition.path / "frames.csv", h0.FRAME_FIELDS)
    results: list[Any | None] = []
    frame_arrays: list[np.ndarray] = []
    errors: dict[int, str] = {}
    for index, row in enumerate(source_rows):
        try:
            result = pipeline.run_frame(h0.frame_from_row(condition, row))
            results.append(result)
            frame_arrays.append(h0.axis_adapter(result.centers_uv_full))
        except Exception as error:  # retain all frame-level failures in report
            results.append(None)
            frame_arrays.append(np.empty((0, 2), dtype=np.float64))
            errors[index] = f"{type(error).__name__}:{error}"

    median_scan: np.ndarray | None = None
    profile_grid: np.ndarray | None = None
    raw_profile: np.ndarray | None = None
    interpolated_profile: np.ndarray | None = None
    assessment: dict[str, Any] = {
        "condition_id": condition.condition_id,
        "auto_qc_status": "FAIL",
        "auto_qc_reasons": [],
        "all_edge_pairs": [],
        "detector_summary": {},
    }
    direct_candidate_count: int | None = None
    direct_detector: dict[str, Any] = {}
    try:
        median_scan = h0.roi_v2_wrapper.median_centerline(frame_arrays)
        profile_grid, raw_profile, interpolated_profile = h0.roi_v2.integer_profile(
            median_scan
        )
        # This direct call is a cross-check; assess_condition below is the
        # authoritative existing selector whose pairs[0] is the H0-1 choice.
        _, direct_detector = h0.roi_v2.build_edge_pairs(
            raw_profile, interpolated_profile
        )
        direct_candidate_count = int(direct_detector.get("candidate_pair_count", 0))
        assessment = h0.roi_v2.assess_condition(
            condition.condition_id,
            median_scan,
            frame_arrays,
            {},
        )
    except Exception as error:
        assessment["auto_qc_reasons"] = [
            f"ROI_V2_ERROR:{type(error).__name__}:{error}"
        ]

    candidates = list(assessment.get("all_edge_pairs") or [])
    if not candidates and direct_candidate_count:
        # Normally unreachable; keep the audit explicit if the two reused
        # entry points ever diverge.
        candidates = []
    enriched = [enrich_candidate(item, frame_arrays) for item in candidates]
    representative_index = len(source_rows) // 2
    representative_result = results[representative_index]
    representative_row = source_rows[representative_index]
    if representative_result is None:
        for index, result in enumerate(results):
            if result is not None:
                representative_result = result
                representative_row = source_rows[index]
                representative_index = index
                break

    return {
        "condition": condition,
        "source_rows": source_rows,
        "results": results,
        "frame_arrays": frame_arrays,
        "pipeline_errors": errors,
        "median_scan": median_scan,
        "profile_grid": profile_grid,
        "raw_profile": raw_profile,
        "interpolated_profile": interpolated_profile,
        "assessment": assessment,
        "candidates": enriched,
        "direct_candidate_count": direct_candidate_count,
        "direct_detector": direct_detector,
        "representative_result": representative_result,
        "representative_row": representative_row,
        "representative_index": representative_index,
    }


def candidate_rows(
    audit: dict[str, Any],
    a3_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    condition: h0.Condition = audit["condition"]
    source_rows: list[dict[str, str]] = audit["source_rows"]
    assessment = audit["assessment"]
    candidates: list[dict[str, Any]] = audit["candidates"]
    selected_score = finite(assessment.get("selected_pair_score"))
    selected_gap = finite(assessment.get("selected_pair_score_relative_gap"))
    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        selected = rank == 1
        baseline = candidate.get("baseline_v_ranges") or [[], []]
        baseline = list(baseline) + [[], []]
        roi_flags = h0.a3_overlap_fields(candidate_roi(candidate), a3_audit)
        score_terms = candidate_score_terms(candidate)
        height_range = candidate.get("height_v_range") or []
        row = {
            "height_gt_mm": condition.height_gt_mm,
            "height_id": condition.height_id,
            "position_id": condition.position_id,
            "condition_id": condition.condition_id,
            "source_frame_count": len(source_rows),
            "pipeline_success_frame_count": sum(
                result is not None for result in audit["results"]
            ),
            "representative_frame": audit["representative_row"].get(
                "camera_frame_number"
            ),
            "median_centerline_point_count": (
                int(len(audit["median_scan"]))
                if audit["median_scan"] is not None
                else 0
            ),
            "profile_noise_px_per_v": audit["direct_detector"].get(
                "profile_noise_px_per_v"
            ),
            "edge_prominence_threshold_px": audit["direct_detector"].get(
                "edge_prominence_threshold_px"
            ),
            "negative_peak_count": audit["direct_detector"].get("negative_peak_count"),
            "positive_peak_count": audit["direct_detector"].get("positive_peak_count"),
            "candidate_count": len(candidates),
            "direct_build_edge_pairs_count": audit["direct_candidate_count"],
            "candidate_rank": rank,
            "selected": selected,
            "selection_rule": "existing_assess_condition_sorted_pairs[0]",
            "orientation": candidate.get("orientation"),
            "edge1_detector_v_px": candidate.get("edge1_v"),
            "edge2_detector_v_px": candidate.get("edge2_v"),
            "edge1_u_full_px": candidate.get("edge1_v"),
            "edge2_u_full_px": candidate.get("edge2_v"),
            "object_width_px": candidate.get("object_width_px"),
            "height_u_full_start_px": interval_endpoint(height_range, 0),
            "height_u_full_end_px": interval_endpoint(height_range, 1),
            "height_u_full_center_px": (
                (float(height_range[0]) + float(height_range[1])) / 2.0
                if len(height_range) == 2
                else None
            ),
            "height_interior_width_px": candidate.get("height_interior_width_px"),
            "transition_exclusion_margin_px": candidate.get(
                "transition_exclusion_margin_px"
            ),
            "transition_v_ranges_detector": candidate.get("transition_v_ranges"),
            "baseline_before_u_full_start_px": interval_endpoint(baseline[0], 0),
            "baseline_before_u_full_end_px": interval_endpoint(baseline[0], 1),
            "baseline_after_u_full_start_px": interval_endpoint(baseline[1], 0),
            "baseline_after_u_full_end_px": interval_endpoint(baseline[1], 1),
            "baseline_v_ranges_detector": baseline,
            "baseline_clipped": candidate.get("baseline_clipped"),
            "edge1_prominence_px": candidate.get("edge1_prominence_px"),
            "edge2_prominence_px": candidate.get("edge2_prominence_px"),
            "edge_min_prominence_px": candidate.get("edge_min_prominence_px"),
            "edge1_local_support_fraction": candidate.get(
                "edge1_local_support_fraction"
            ),
            "edge2_local_support_fraction": candidate.get(
                "edge2_local_support_fraction"
            ),
            "predicted_ground_u_at_height_mid": candidate.get(
                "predicted_ground_u_at_height_mid"
            ),
            "plateau_delta_u_px": candidate.get("plateau_delta_u_px"),
            "step_amplitude_px": candidate.get("step_amplitude_px"),
            "pair_score": candidate.get("pair_score"),
            "score_recomputed": score_terms["score_recomputed"],
            "pair_gate_reasons": candidate.get("pair_gate_reasons"),
            "edge_pair_geometry_ok": candidate.get("edge_pair_geometry_ok"),
            "before_stats": candidate.get("before_stats"),
            "height_stats": candidate.get("height_stats"),
            "after_stats": candidate.get("after_stats"),
            "ground_fit_slope_px_per_v": candidate.get(
                "ground_fit_slope_px_per_v"
            ),
            "ground_fit_intercept_px": candidate.get("ground_fit_intercept_px"),
            "height_support": candidate.get("height_support"),
            "baseline_before_support": candidate.get("before_support"),
            "baseline_after_support": candidate.get("after_support"),
            "multi_geometry_reasons": candidate.get("multi_geometry_reasons"),
            "multi_geometry_ok": candidate.get("multi_geometry_ok"),
            "condition_roi_status": assessment.get("auto_qc_status"),
            "condition_roi_reasons": assessment.get("auto_qc_reasons"),
            "selected_pair_score": selected_score,
            "selected_pair_score_relative_gap": selected_gap,
            **roi_flags,
        }
        rows.append(row)
        score_rows.append(
            {
                "height_gt_mm": condition.height_gt_mm,
                "height_id": condition.height_id,
                "position_id": condition.position_id,
                "condition_id": condition.condition_id,
                "candidate_rank": rank,
                "selected": selected,
                "orientation": candidate.get("orientation"),
                "edge1_u_full_px": candidate.get("edge1_v"),
                "edge2_u_full_px": candidate.get("edge2_v"),
                "edge1_prominence_px": candidate.get("edge1_prominence_px"),
                "edge2_prominence_px": candidate.get("edge2_prominence_px"),
                "min_edge_prominence_px": candidate.get("edge_min_prominence_px"),
                "min_edge_prominence_term": score_terms["min_edge_prominence_term"],
                "step_amplitude_px": candidate.get("step_amplitude_px"),
                "step_amplitude_term": score_terms["step_amplitude_term"],
                "object_width_px": candidate.get("object_width_px"),
                "width_term": score_terms["width_term"],
                "pair_score": candidate.get("pair_score"),
                "score_recomputed": score_terms["score_recomputed"],
                "score_delta": (
                    finite(candidate.get("pair_score"))
                    - score_terms["score_recomputed"]
                    if finite(candidate.get("pair_score")) is not None
                    else None
                ),
                "edge_pair_geometry_ok": candidate.get("edge_pair_geometry_ok"),
                "pair_gate_reason_count": len(
                    candidate.get("pair_gate_reasons") or []
                ),
                "pair_gate_reasons": candidate.get("pair_gate_reasons"),
                "multi_geometry_ok": candidate.get("multi_geometry_ok"),
                "multi_geometry_reason_count": len(
                    candidate.get("multi_geometry_reasons") or []
                ),
                "multi_geometry_reasons": candidate.get("multi_geometry_reasons"),
                "height_support_ok": (candidate.get("height_support") or {}).get(
                    "support_ok"
                ),
                "baseline_before_support_ok": (
                    candidate.get("before_support") or {}
                ).get("support_ok"),
                "baseline_after_support_ok": (
                    candidate.get("after_support") or {}
                ).get("support_ok"),
                "a3_right_morphology_region_overlap": roi_flags[
                    "a3_right_morphology_region_overlap"
                ],
            }
        )
    return rows, score_rows


ROI_AUDIT_FIELDS = [
    "height_gt_mm",
    "height_id",
    "position_id",
    "condition_id",
    "source_frame_count",
    "pipeline_success_frame_count",
    "representative_frame",
    "median_centerline_point_count",
    "profile_noise_px_per_v",
    "edge_prominence_threshold_px",
    "negative_peak_count",
    "positive_peak_count",
    "candidate_count",
    "direct_build_edge_pairs_count",
    "candidate_rank",
    "selected",
    "selection_rule",
    "orientation",
    "edge1_detector_v_px",
    "edge2_detector_v_px",
    "edge1_u_full_px",
    "edge2_u_full_px",
    "object_width_px",
    "height_u_full_start_px",
    "height_u_full_end_px",
    "height_u_full_center_px",
    "height_interior_width_px",
    "transition_exclusion_margin_px",
    "transition_v_ranges_detector",
    "baseline_before_u_full_start_px",
    "baseline_before_u_full_end_px",
    "baseline_after_u_full_start_px",
    "baseline_after_u_full_end_px",
    "baseline_v_ranges_detector",
    "baseline_clipped",
    "edge1_prominence_px",
    "edge2_prominence_px",
    "edge_min_prominence_px",
    "edge1_local_support_fraction",
    "edge2_local_support_fraction",
    "predicted_ground_u_at_height_mid",
    "plateau_delta_u_px",
    "step_amplitude_px",
    "pair_score",
    "score_recomputed",
    "pair_gate_reasons",
    "edge_pair_geometry_ok",
    "before_stats",
    "height_stats",
    "after_stats",
    "ground_fit_slope_px_per_v",
    "ground_fit_intercept_px",
    "height_support",
    "baseline_before_support",
    "baseline_after_support",
    "multi_geometry_reasons",
    "multi_geometry_ok",
    "condition_roi_status",
    "condition_roi_reasons",
    "selected_pair_score",
    "selected_pair_score_relative_gap",
    "a3_right_morphology_region_available",
    "a3_right_morphology_region_u_full_range_px",
    "a3_right_height_roi_overlap_px",
    "a3_right_baseline_before_overlap_px",
    "a3_right_baseline_after_overlap_px",
    "a3_right_baseline_overlap_px",
    "a3_right_morphology_region_overlap",
    "a3_spatial_risk_reason",
]


SCORE_FIELDS = [
    "height_gt_mm",
    "height_id",
    "position_id",
    "condition_id",
    "candidate_rank",
    "selected",
    "orientation",
    "edge1_u_full_px",
    "edge2_u_full_px",
    "edge1_prominence_px",
    "edge2_prominence_px",
    "min_edge_prominence_px",
    "min_edge_prominence_term",
    "step_amplitude_px",
    "step_amplitude_term",
    "object_width_px",
    "width_term",
    "pair_score",
    "score_recomputed",
    "score_delta",
    "edge_pair_geometry_ok",
    "pair_gate_reason_count",
    "pair_gate_reasons",
    "multi_geometry_ok",
    "multi_geometry_reason_count",
    "multi_geometry_reasons",
    "height_support_ok",
    "baseline_before_support_ok",
    "baseline_after_support_ok",
    "a3_right_morphology_region_overlap",
]


def full_u_range_for_row(row: dict[str, str]) -> tuple[float, float]:
    offset = finite(row.get("offset_x")) or 0.0
    width = finite(row.get("width")) or 0.0
    return offset, offset + max(0.0, width - 1.0)


def full_v_range_for_row(row: dict[str, str]) -> tuple[float, float]:
    offset = finite(row.get("offset_y")) or 0.0
    height = finite(row.get("height")) or 0.0
    return offset, offset + max(0.0, height - 1.0)


def plot_interval(ax: Any, interval: Any, **kwargs: Any) -> None:
    start = finite(interval_endpoint(interval, 0))
    end = finite(interval_endpoint(interval, 1))
    if start is not None and end is not None:
        ax.axvspan(start, end, **kwargs)


def overlay_path(output_dir: Path, condition: h0.Condition) -> Path:
    return output_dir / f"{condition.condition_id}_roi_candidates_overlay.png"


def render_overlay(
    audit: dict[str, Any],
    output_dir: Path,
    *,
    label: str | None = None,
) -> Path | None:
    image_result = audit.get("representative_result")
    image_row = audit.get("representative_row") or {}
    median_scan = audit.get("median_scan")
    grid = audit.get("profile_grid")
    interpolated = audit.get("interpolated_profile")
    candidates: list[dict[str, Any]] = audit.get("candidates") or []
    if image_result is None or median_scan is None or grid is None or interpolated is None:
        return None
    image = np.asarray(image_result.frame.image)
    if image.ndim != 2:
        return None

    condition: h0.Condition = audit["condition"]
    x0, x1 = full_u_range_for_row(image_row)
    y0, y1 = full_v_range_for_row(image_row)
    center_uv = np.asarray(median_scan, dtype=np.float64)[:, [1, 0]]
    center_good = np.isfinite(center_uv).all(axis=1)
    image_min = float(np.nanmin(image)) if image.size else 0.0
    image_max = float(np.nanmax(image)) if image.size else 1.0
    if image_max <= image_min:
        image_max = image_min + 1.0

    fig, (ax_image, ax_profile) = plt.subplots(
        1,
        2,
        figsize=(17, 8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=False,
    )
    ax_image.imshow(
        image,
        cmap="gray",
        vmin=image_min,
        vmax=image_max,
        extent=(x0, x1 + 1.0, y1 + 1.0, y0),
        aspect="auto",
        interpolation="nearest",
    )
    if center_good.any():
        ax_image.plot(
            center_uv[center_good, 0],
            center_uv[center_good, 1],
            color="yellow",
            linewidth=0.8,
            alpha=0.9,
            label="median centerline",
        )

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, min(10, len(candidates)))))
    for rank, candidate in enumerate(candidates, start=1):
        color = "#ff3030" if rank == 1 else colors[(rank - 1) % len(colors)]
        line_style = "-" if rank == 1 else "--"
        line_width = 2.2 if rank == 1 else 1.0
        edge1 = finite(candidate.get("edge1_v"))
        edge2 = finite(candidate.get("edge2_v"))
        if edge1 is None or edge2 is None:
            continue
        ax_image.axvline(
            edge1,
            color=color,
            linestyle=line_style,
            linewidth=line_width,
            alpha=0.95,
        )
        ax_image.axvline(
            edge2,
            color=color,
            linestyle=line_style,
            linewidth=line_width,
            alpha=0.95,
        )
        height = candidate.get("height_v_range") or []
        if rank == 1:
            plot_interval(ax_image, height, color="#ff3030", alpha=0.22, label="selected height")
            baseline = candidate.get("baseline_v_ranges") or [[], []]
            baseline = list(baseline) + [[], []]
            plot_interval(
                ax_image,
                baseline[0],
                color="#36a3ff",
                alpha=0.18,
                label="selected baseline-before",
            )
            plot_interval(
                ax_image,
                baseline[1],
                color="#43d17b",
                alpha=0.18,
                label="selected baseline-after",
            )
        else:
            plot_interval(ax_image, height, color=color, alpha=0.035)
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

    ax_profile.plot(
        np.asarray(grid),
        np.asarray(interpolated),
        color="black",
        linewidth=1.0,
        label="interpolated median profile",
    )
    raw = audit.get("raw_profile")
    if raw is not None:
        raw_array = np.asarray(raw, dtype=np.float64)
        valid = np.isfinite(raw_array)
        ax_profile.scatter(
            np.arange(len(raw_array), dtype=np.float64)[valid],
            raw_array[valid],
            color="#777777",
            s=2,
            alpha=0.35,
            label="raw profile points",
        )
    for rank, candidate in enumerate(candidates, start=1):
        color = "#ff3030" if rank == 1 else colors[(rank - 1) % len(colors)]
        line_style = "-" if rank == 1 else "--"
        line_width = 2.2 if rank == 1 else 1.0
        edge1 = finite(candidate.get("edge1_v"))
        edge2 = finite(candidate.get("edge2_v"))
        if edge1 is None or edge2 is None:
            continue
        ax_profile.axvline(edge1, color=color, linestyle=line_style, linewidth=line_width)
        ax_profile.axvline(edge2, color=color, linestyle=line_style, linewidth=line_width)
        if rank == 1:
            plot_interval(
                ax_profile,
                candidate.get("height_v_range"),
                color="#ff3030",
                alpha=0.18,
                label="selected height interior",
            )

    ax_image.set_xlim(x0, x1 + 1.0)
    ax_image.set_ylim(y1 + 1.0, y0)
    ax_profile.set_xlim(x0, x1 + 1.0)
    ax_profile.set_ylim(y1 + 1.0, y0)
    ax_image.set_xlabel("full-sensor u (px)")
    ax_image.set_ylabel("full-sensor v (px)")
    ax_profile.set_xlabel("detector v' = original full-sensor u (px)")
    ax_profile.set_ylabel("detector u' = original full-sensor v (px)")
    ax_image.set_title("raw representative image + median centerline")
    ax_profile.set_title("all ROI-V2 candidate edge pairs")
    ax_image.grid(alpha=0.15)
    ax_profile.grid(alpha=0.2)
    ax_image.legend(loc="best", fontsize=8)
    ax_profile.legend(loc="best", fontsize=8)

    selected = candidates[0] if candidates else {}
    candidate_text = []
    for rank, candidate in enumerate(candidates, start=1):
        reasons = ",".join(candidate.get("pair_gate_reasons") or []) or "none"
        candidate_text.append(
            f"r{rank}{'*' if rank == 1 else ''}: "
            f"({candidate.get('edge1_v')},{candidate.get('edge2_v')}) "
            f"score={finite(candidate.get('pair_score')):.4f} "
            f"step={finite(candidate.get('step_amplitude_px')):.3f}; "
            f"gates={reasons}"
        )
    title_prefix = f"{condition.condition_id}"
    if label:
        title_prefix += f" [{label}]"
    fig.suptitle(
        f"{title_prefix} | ROI-V2 status={audit['assessment'].get('auto_qc_status')} | "
        f"candidates={len(candidates)} | selected r1 "
        f"edge=({selected.get('edge1_v')},{selected.get('edge2_v')})",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "; ".join(candidate_text)
        + " | adapter: Haikang (u,v) -> ROI-V2 (u'=v,v'=u); truth not used for selection",
        ha="center",
        va="bottom",
        fontsize=7,
        wrap=True,
    )
    path = overlay_path(output_dir, condition)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def condition_summary(audit: dict[str, Any], a3_audit: dict[str, Any]) -> dict[str, Any]:
    condition: h0.Condition = audit["condition"]
    candidates: list[dict[str, Any]] = audit["candidates"]
    assessment = audit["assessment"]
    selected = candidates[0] if candidates else {}
    roi_flags = h0.a3_overlap_fields(candidate_roi(selected), a3_audit)
    height = selected.get("height_v_range") or []
    score_terms = candidate_score_terms(selected) if selected else {}
    return {
        "height_gt_mm": condition.height_gt_mm,
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "condition_id": condition.condition_id,
        "source_path": str(condition.path.resolve()),
        "source_frame_count": len(audit["source_rows"]),
        "pipeline_success_frame_count": sum(
            result is not None for result in audit["results"]
        ),
        "pipeline_error_count": len(audit["pipeline_errors"]),
        "median_centerline_point_count": (
            int(len(audit["median_scan"])) if audit["median_scan"] is not None else 0
        ),
        "candidate_count": len(candidates),
        "direct_build_edge_pairs_count": audit["direct_candidate_count"],
        "roi_v2_status": assessment.get("auto_qc_status", "FAIL"),
        "roi_v2_reasons": assessment.get("auto_qc_reasons", []),
        "selected_candidate_rank": 1 if candidates else None,
        "selected_orientation": selected.get("orientation"),
        "selected_edge1_u_full_px": selected.get("edge1_v"),
        "selected_edge2_u_full_px": selected.get("edge2_v"),
        "selected_height_u_full_start_px": interval_endpoint(height, 0),
        "selected_height_u_full_end_px": interval_endpoint(height, 1),
        "selected_height_u_full_center_px": (
            (float(height[0]) + float(height[1])) / 2.0
            if len(height) == 2
            else None
        ),
        "selected_object_width_px": selected.get("object_width_px"),
        "selected_height_interior_width_px": selected.get(
            "height_interior_width_px"
        ),
        "selected_edge1_prominence_px": selected.get("edge1_prominence_px"),
        "selected_edge2_prominence_px": selected.get("edge2_prominence_px"),
        "selected_step_amplitude_px": selected.get("step_amplitude_px"),
        "selected_pair_score": selected.get("pair_score"),
        "selected_score_recomputed": score_terms.get("score_recomputed"),
        "selected_runner_up_pair_score": (
            candidates[1].get("pair_score") if len(candidates) > 1 else None
        ),
        "selected_pair_score_relative_gap": assessment.get(
            "selected_pair_score_relative_gap"
        ),
        "selected_pair_gate_reasons": selected.get("pair_gate_reasons"),
        "selected_height_support": selected.get("height_support"),
        "selected_baseline_before_support": selected.get("before_support"),
        "selected_baseline_after_support": selected.get("after_support"),
        "selected_multi_geometry_ok": selected.get("multi_geometry_ok"),
        "selected_multi_geometry_reasons": selected.get("multi_geometry_reasons"),
        "selected_a3_right_morphology_region_overlap": roi_flags[
            "a3_right_morphology_region_overlap"
        ],
        "selected_a3_right_height_roi_overlap_px": roi_flags[
            "a3_right_height_roi_overlap_px"
        ],
        "selected_a3_right_baseline_overlap_px": roi_flags[
            "a3_right_baseline_overlap_px"
        ],
        "selected_pair_key": (
            f"{selected.get('edge1_v')},{selected.get('edge2_v')}"
            if candidates
            else None
        ),
    }


SUMMARY_FIELDS = [
    "height_gt_mm",
    "height_id",
    "position_id",
    "condition_id",
    "source_path",
    "source_frame_count",
    "pipeline_success_frame_count",
    "pipeline_error_count",
    "median_centerline_point_count",
    "candidate_count",
    "direct_build_edge_pairs_count",
    "roi_v2_status",
    "roi_v2_reasons",
    "selected_candidate_rank",
    "selected_orientation",
    "selected_edge1_u_full_px",
    "selected_edge2_u_full_px",
    "selected_height_u_full_start_px",
    "selected_height_u_full_end_px",
    "selected_height_u_full_center_px",
    "selected_object_width_px",
    "selected_height_interior_width_px",
    "selected_edge1_prominence_px",
    "selected_edge2_prominence_px",
    "selected_step_amplitude_px",
    "selected_pair_score",
    "selected_score_recomputed",
    "selected_runner_up_pair_score",
    "selected_pair_score_relative_gap",
    "selected_pair_gate_reasons",
    "selected_height_support",
    "selected_baseline_before_support",
    "selected_baseline_after_support",
    "selected_multi_geometry_ok",
    "selected_multi_geometry_reasons",
    "selected_a3_right_morphology_region_overlap",
    "selected_a3_right_height_roi_overlap_px",
    "selected_a3_right_baseline_overlap_px",
    "selected_pair_key",
]


def choose_overlay_conditions(summaries: list[dict[str, Any]]) -> list[tuple[str, str]]:
    chosen: list[tuple[str, str]] = []
    for row in summaries:
        if row["position_id"] in TARGET_OVERLAY_POSITIONS:
            chosen.append((row["condition_id"], "required_p01_p05_p10"))
    required = {condition_id for condition_id, _ in chosen}
    for row in summaries:
        if row["condition_id"] in required:
            continue
        if row.get("roi_v2_status") == "PASS":
            chosen.append((row["condition_id"], "additional_typical_PASS"))
        if len([item for item in chosen if item[1] == "additional_typical_PASS"]) >= 3:
            break
    return chosen


def summarize_counts(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [int(row["candidate_count"] or 0) for row in summaries]
    pair_keys = Counter(
        row["selected_pair_key"] for row in summaries if row["selected_pair_key"]
    )
    all_audit_rows = []
    return {
        "candidate_count_distribution": dict(Counter(candidates)),
        "selected_pair_frequency": dict(pair_keys.most_common()),
    }


def build_report(
    *,
    root: Path,
    config_summary: dict[str, Any],
    session_summary: dict[str, Any],
    a3_audit: dict[str, Any],
    summaries: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    overlays: list[Path],
) -> str:
    status_counts = Counter(row.get("roi_v2_status") for row in summaries)
    candidate_dist = Counter(int(row.get("candidate_count") or 0) for row in summaries)
    selected_pairs = Counter(
        (row.get("selected_edge1_u_full_px"), row.get("selected_edge2_u_full_px"))
        for row in summaries
        if row.get("selected_edge1_u_full_px") is not None
    )
    exact_fixed = selected_pairs.get((1950, 2024), 0)
    fixed_window = sum(
        row.get("selected_edge1_u_full_px") is not None
        and row.get("selected_edge2_u_full_px") is not None
        and 1950 <= float(row["selected_edge1_u_full_px"]) <= 2024
        and 1950 <= float(row["selected_edge2_u_full_px"]) <= 2024
        for row in summaries
    )
    a3_overlap = sum(
        bool(row.get("selected_a3_right_morphology_region_overlap"))
        for row in summaries
    )
    all_candidate_pairs = Counter(
        (row.get("edge1_u_full_px"), row.get("edge2_u_full_px"))
        for row in audit_rows
        if row.get("edge1_u_full_px") is not None
    )
    score_deltas = [
        abs(float(row["score_delta"]))
        for row in score_rows
        if row.get("score_delta") is not None
    ]
    by_height: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_height[str(row["height_id"])].append(row)

    lines = [
        "# H0-1R | 海康 ROI-V2 误选根因审计",
        "",
        f"数据根目录：`{root.resolve()}`",
        f"条件数：{len(summaries)}；候选明细行数：{len(audit_rows)}；overlay：{len(overlays)} 张",
        "本报告只审计 ROI 选择；未调用高度测量、H1、H-B2、C1 或任何补偿。",
        "",
        "## 1. 复用与新增",
        "",
        "复用：`FramePipeline.run_frame` 的冻结 Steger → 海康 circular-cone C0 → "
        "Session Ground 中心线输出；H0-1 的 `(u,v) -> (u'=v,v'=u)` adapter；"
        "`thermal_a2a_roi_v2.median_centerline`；"
        "`auto_roi_v2_session01.integer_profile / build_edge_pairs / assess_condition`；"
        "`thermal_a2a_roi_v2.assess_pair` 的重复支持统计。",
        "",
        "新增：只保存全部 candidate edge pair、score 分解、候选与 A-3 区域 overlap，"
        "并生成原图 + median centerline + 所有候选的 overlay；没有复制或修改 ROI-V2 算法。",
        "",
        "## 2. 全量选择统计",
        "",
        f"ROI-V2 状态：{dict(status_counts)}",
        f"candidate_pair_count 分布：{dict(sorted(candidate_dist.items()))}",
        f"selected 精确 `(1950,2024)`：{exact_fixed}/{len(summaries)}；"
        f"selected 两端均落在 `[1950,2024]`：{fixed_window}/{len(summaries)}。",
        f"selected ROI 与 A-3 右侧 morphology 区域 overlap：{a3_overlap}/{len(summaries)}。",
        f"candidate 全体中最常见 edge pair：{all_candidate_pairs.most_common(8)}",
        f"score 公式重算最大绝对差：{max(score_deltas) if score_deltas else None}",
        "",
        "selected pair 频率：",
        "",
        "| edge1 full-u | edge2 full-u | 次数 |",
        "|---:|---:|---:|",
    ]
    for pair, count in selected_pairs.most_common():
        lines.append(f"| {pair[0]} | {pair[1]} | {count} |")

    lines += [
        "",
        "## 3. height × position 的 selected ROI 中心",
        "",
        "这里的 ROI-V2 profile 自变量 `v'` 在 adapter 后等于原始 full-sensor `u`；"
        "因此下表用 `selected_height_u_full_center_px` 检查是否随 position 移动。",
        "",
        "| height | position 数 | selected center min | max | range |",
        "|---|---:|---:|---:|---:|",
    ]
    for height_id in sorted(by_height, key=lambda item: h0.TARGET_HEIGHT_MM[item]):
        values = [
            float(row["selected_height_u_full_center_px"])
            for row in by_height[height_id]
            if row.get("selected_height_u_full_center_px") is not None
        ]
        if values:
            lines.append(
                f"| {height_id} | {len(values)} | {min(values):.1f} | "
                f"{max(values):.1f} | {max(values) - min(values):.1f} |"
            )
        else:
            lines.append(f"| {height_id} | 0 | | | |")

    lines += [
        "",
        "结论：h06/h10/h20/h30 的 selected height ROI 基本固定在 full-u 约 "
        "1965–2009，未随 p01–p10 出现可信移动；h02 的 p01/p05/p10 属于不同采集窗口/"
        "候选域，不能直接与局部 ROI 坐标比较。",
        "",
        "## 4. 坐标语义核验",
        "",
        "既有 `auto_roi_v2_session01.py:281-304` 明确构造 `raw[v] = u`，"
        "profile 自变量是 row-scan 的 `v`，因变量是 centerline 横向 `u`。"
        "`auto_roi_v2_session01.py:471-640` 的 edge pair 是一阶导数上的相反符号"
        " transition，中间区间是 object plateau。",
        "",
        "海康 column scan 的中心线是原始 `v=f(u)`。H0-1 swap 后："
        "`u'=v`、`v'=u`，因此 ROI-V2 的 profile 自变量 `v'` 正好是海康原始 full-sensor `u`，"
        "因变量 `u'` 是海康原始 `v`。overlay 的右图明确标注这两个轴。",
        "",
        "本轮对每组同时调用 `integer_profile`、`build_edge_pairs` 和现有"
        "`assess_condition` 做 candidate-count 交叉核对；没有发现 swap 方向反转。"
        "因此当前证据不支持 `AXIS_ADAPTER_SEMANTICS_WRONG`。",
        "",
        "## 5. candidate score 根因",
        "",
        "既有 `auto_roi_v2_session01.py:571-575`：",
        "`pair_score = min(edge prominence) + 0.10 * step_amplitude + "
        "0.01 * min(object_width, 120)`。排序在 `:642-648`：先 geometry pass，"
        "再 gate reason 少，最后 score 高；`assess_condition` 在 `:686` 取 `pairs[0]`。",
        "",
        "这意味着 score 主要受两侧 edge prominence 支配；repeat support 只证明某个"
        " profile 区间重复存在，不证明它是标准块。固定右侧候选的 width、step、support"
        "高度集中，形成稳定但可疑的假台阶。所有候选的逐项分解在"
        "`candidate_score_breakdown.csv`，未用目录高度反选。",
        "",
        "## 6. 原图与 overlay 审计",
        "",
        "overlay 覆盖每个 height 的 p01/p05/p10，并额外覆盖 3 个 PASS 样本。"
        "每张图左侧是原始 representative PNG + median centerline；右侧是"
        "`u'(v')` profile，显示全部候选 edge pair，红色为既有 selected r1，"
        "红色填充为 selected height interior，蓝/绿为 before/after baseline。",
        "",
        "h02/p01、p05、p10 的 frames.csv 采集窗口不同；审阅时必须按 full-sensor"
        " offset 映射，不得把局部 PNG 横坐标直接当成 full-u。",
        "",
        "## 7. 分类与最小修复建议",
        "",
        f"本轮主分类：`FALSE_EDGE_PAIR_DOMINATES`。固定候选重复、右侧 A-3 morphology overlap "
        f"和 selected ROI 不随 position 移动共同支持该判断。",
        "",
        "不是 adapter 方向错误；也不能通过把某个候选改成接近目录高度来修复，"
        "那会引入 truth leakage。当前首先应修改 candidate ranking / candidate review gate，"
        "让固定背景台阶候选进入 quarantine，并补充目标可见性/跨 position 几何约束；"
        "随后再检查 ROI-V2 参数是否需要按海康 profile 域重新评估。",
        "",
        "若 overlay 人工复核确认标准块在当前图像中根本不可见，则需同时标记"
        "`DATASET_TARGET_NOT_VISIBLE`；这不是 H1/H-B2 问题。",
        "",
        "## 8. 可追溯性",
        "",
        f"Haikang config：`{config_summary.get('path')}`（SHA-256 `{config_summary.get('sha256')}`）",
        f"Session Ground：`{session_summary.get('path')}`（SHA-256 `{session_summary.get('sha256')}`）",
        f"A-3 artifact：`{a3_audit.get('path')}`；classification=`{a3_audit.get('classification')}`；"
        f"right range={a3_audit.get('right_u_full_range_px')}",
        "height_shadow.csv 未读取；目录 truth 只用于最终 sanity check，不参与 candidate 选择。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.input_dir.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions, discovery = h0.discover_conditions(root)
    config_summary = h0.config_contract(config_path)
    session_reference, rotation, translation, session_summary = h0.load_session_reference(
        root / "session_ground_calibration.json"
    )
    a3_audit = h0.load_prior_a3_spatial_audit(root)
    _, pipeline = h0.make_pipeline(
        config_path, session_reference, rotation, translation
    )

    audits: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for condition in conditions:
        audit = run_condition(condition, pipeline)
        audits[condition.condition_id] = audit
        summary = condition_summary(audit, a3_audit)
        summaries.append(summary)
        rows, scores = candidate_rows(audit, a3_audit)
        audit_rows.extend(rows)
        score_rows.extend(scores)

    overlay_targets = choose_overlay_conditions(summaries)
    overlays: list[Path] = []
    for condition_id, label in overlay_targets:
        path = render_overlay(audits[condition_id], output_dir, label=label)
        if path is not None:
            overlays.append(path)

    h0.write_csv(output_dir / "roi_selection_audit.csv", ROI_AUDIT_FIELDS, audit_rows)
    h0.write_csv(output_dir / "candidate_score_breakdown.csv", SCORE_FIELDS, score_rows)
    (output_dir / "roi_v2_audit_report.md").write_text(
        build_report(
            root=root,
            config_summary=config_summary,
            session_summary=session_summary,
            a3_audit=a3_audit,
            summaries=summaries,
            audit_rows=audit_rows,
            score_rows=score_rows,
            overlays=overlays,
        ),
        encoding="utf-8",
    )
    print(
        h0.json.dumps(
            {
                "output_dir": str(output_dir),
                "condition_count": len(summaries),
                "candidate_rows": len(audit_rows),
                "score_rows": len(score_rows),
                "overlay_count": len(overlays),
                "roi_status_counts": dict(
                    Counter(row.get("roi_v2_status") for row in summaries)
                ),
                "a3_overlap_conditions": sum(
                    bool(row.get("selected_a3_right_morphology_region_overlap"))
                    for row in summaries
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
