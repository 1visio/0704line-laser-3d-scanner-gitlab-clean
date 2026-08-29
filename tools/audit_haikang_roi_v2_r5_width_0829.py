#!/usr/bin/env python3
"""Audit Haikang ROI-V2 with the R5 width-gate test values.

This script is an offline, Haikang-specific ROI audit.  It reuses the R4
Session search ROI and replay, runs the existing ROI-V2 implementation once
with the current values and once with only the three requested width-related
values changed, and writes a before/after comparison.  The parameter change
is process-local and is restored after every phase; no production file is
modified.  No height measurement, compensation, H1, H-B2, C1, or accuracy
claim is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import audit_haikang_roi_v2_manual_search_0829 as r4  # noqa: E402


DATA_ROOT_DEFAULT = r4.DATA_ROOT_DEFAULT
OUTPUT_DIR_DEFAULT = DATA_ROOT_DEFAULT / "c0_height_audit" / "roi_v2_r5_width"
REUSED_ROI_DEFAULT = DATA_ROOT_DEFAULT / "c0_height_audit" / "roi_v2_manual_search" / r4.ROI_JSON_NAME

BEFORE_GATES: dict[str, float | int] = {
    "edge_pair_min_width_px": 50,
    "transition_exclusion_margin_px": 15,
    "height_interior_min_width_px": 30,
}
R5_GATES: dict[str, float | int] = {
    "edge_pair_min_width_px": 44,
    "transition_exclusion_margin_px": 10,
    "height_interior_min_width_px": 20,
}
WIDTH_GATE_KEYS = tuple(BEFORE_GATES)
OVERLAY_IDS = {
    "h02_p01",
    *(f"{height}_p{position:02d}" for height in ("h06", "h10", "h20", "h30") for position in (1, 5, 10)),
}


class AuditError(RuntimeError):
    """Raised when the R5 audit contract is not satisfied."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=r4.r1.h0.CONFIG_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--roi-json", type=Path, default=REUSED_ROI_DEFAULT)
    return parser.parse_args(argv)


def finite(value: Any) -> float | None:
    return r4.finite(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate_snapshot() -> dict[str, float | int]:
    params = r4.r1.h0.roi_v2.PARAMETERS["object_interval"]
    return {key: params[key] for key in WIDTH_GATE_KEYS}


@contextmanager
def temporary_gates(values: dict[str, float | int]) -> Iterator[None]:
    """Apply only the requested process-local gates and always restore them."""
    params = r4.r1.h0.roi_v2.PARAMETERS["object_interval"]
    original = {key: params[key] for key in WIDTH_GATE_KEYS}
    params.update(values)
    try:
        yield
    finally:
        params.update(original)


def phase_record(
    base: dict[str, Any],
    audit: dict[str, Any],
    profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    record = dict(base)
    record.update(
        {
            "manual": audit,
            "manual_profile": profile,
            "manual_candidates": candidates,
            "manual_candidate_count": len(candidates),
            "manual_error": error,
        }
    )
    return record


def filter_median(median: np.ndarray | None, polygon: np.ndarray) -> tuple[np.ndarray | None, int, int, str | None]:
    if median is None:
        return None, 0, 0, "full_median_centerline_unavailable"
    values = np.asarray(median, dtype=np.float64)
    total = len(values)
    full_uv = values[:, [1, 0]]
    finite_mask = np.isfinite(full_uv).all(axis=1)
    inside = np.zeros(total, dtype=bool)
    if np.any(finite_mask):
        inside[finite_mask] = r4.points_inside_polygon(full_uv[finite_mask], polygon)
    return values[inside], total, int(inside.sum()), None


def phase_transition_rows(record: dict[str, Any], gates: dict[str, float | int]) -> list[dict[str, Any]]:
    with temporary_gates(gates):
        rows = r4.make_transition_rows(record)
    return rows


def phase_selection(record: dict[str, Any], gates: dict[str, float | int]) -> dict[str, Any]:
    with temporary_gates(gates):
        return r4.select_target(record, float(gates["height_interior_min_width_px"]))


def run_condition(
    condition: r4.r1.h0.Condition,
    pipeline: Any,
    polygon: np.ndarray,
) -> dict[str, Any]:
    """Replay one condition and evaluate current/R5 gates on the same profile."""
    with temporary_gates(BEFORE_GATES):
        full = r4.r1.run_condition(condition, pipeline)
        full_profile = r4.r3.profile_diagnostics(full)
        filtered, total_points, retained_points, filter_error = filter_median(
            full.get("median_scan"), polygon
        )
        before_audit, before_profile, before_candidates, before_error = r4.make_profile_audit(
            full,
            filtered,
            full.get("frame_arrays") or [],
            condition.condition_id,
        )
    base = {
        "condition": condition,
        "full": full,
        "full_profile": full_profile,
        "full_candidate_count": len(full.get("candidates") or []),
        "full_median_points": total_points,
        "manual_median_points": retained_points,
        "manual_filter_error": filter_error,
    }
    before_record = phase_record(
        base, before_audit, before_profile, before_candidates, before_error
    )
    with temporary_gates(R5_GATES):
        after_audit, after_profile, after_candidates, after_error = r4.make_profile_audit(
            full,
            filtered,
            full.get("frame_arrays") or [],
            condition.condition_id,
        )
    after_record = phase_record(
        base, after_audit, after_profile, after_candidates, after_error
    )
    before_rows = phase_transition_rows(before_record, BEFORE_GATES)
    after_rows = phase_transition_rows(after_record, R5_GATES)
    for row in before_rows:
        row["phase"] = "BEFORE"
    for row in after_rows:
        row["phase"] = "AFTER_R5"
    return {
        **base,
        "before": before_record,
        "after": after_record,
        "before_profile": before_profile,
        "after_profile": after_profile,
        "before_candidates": before_candidates,
        "after_candidates": after_candidates,
        "before_rows": before_rows,
        "after_rows": after_rows,
        "before_selection": phase_selection(before_record, BEFORE_GATES),
        "after_selection": phase_selection(after_record, R5_GATES),
    }


def stage_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("pair_generation_stage") or "") for row in rows)


def stage_condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[str(row.get("pair_generation_stage") or "")].add(str(row.get("condition_id")))
    return {stage: len(condition_ids) for stage, condition_ids in grouped.items()}


def selected_candidate(record: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    rank = selection.get("selected_candidate_rank")
    candidates = record.get("manual_candidates") or []
    if rank is None or not isinstance(rank, int) or rank < 1 or rank > len(candidates):
        return {}
    return candidates[rank - 1]


def make_before_after_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        condition = record["condition"]
        before = record["before_selection"]
        after = record["after_selection"]
        before_stage = stage_counts(record["before_rows"])
        after_stage = stage_counts(record["after_rows"])
        rows.append(
            {
                "condition_id": condition.condition_id,
                "height_id": condition.height_id,
                "height_gt_mm": condition.height_gt_mm,
                "position_id": condition.position_id,
                "before_target_roi_status": before.get("target_roi_status"),
                "before_target_status_reason": before.get("target_status_reason"),
                "before_candidate_count_generated": before.get("candidate_count_generated"),
                "before_selected_candidate_rank": before.get("selected_candidate_rank"),
                "before_selected_edge1_u_full_px": before.get("selected_edge1_u_full_px"),
                "before_selected_edge2_u_full_px": before.get("selected_edge2_u_full_px"),
                "before_selected_center_u_full_px": before.get("selected_center_u_full_px"),
                "before_selected_width_px": before.get("selected_width_px"),
                "before_selected_height_interior_width_px": before.get("selected_height_interior_width_px"),
                "before_local_baseline_status": before.get("local_baseline_status"),
                "before_baseline_measurement_eligible": before.get("baseline_measurement_eligible"),
                "before_no_peak_pair_count": before_stage.get("NO_PEAK_PAIR", 0),
                "before_rejected_before_candidate_width_count": before_stage.get("REJECTED_BEFORE_CANDIDATE_WIDTH", 0),
                "before_generated_candidate_count": before_stage.get("GENERATED_CANDIDATE", 0),
                "after_target_roi_status": after.get("target_roi_status"),
                "after_target_status_reason": after.get("target_status_reason"),
                "after_candidate_count_generated": after.get("candidate_count_generated"),
                "after_selected_candidate_rank": after.get("selected_candidate_rank"),
                "after_selected_edge1_u_full_px": after.get("selected_edge1_u_full_px"),
                "after_selected_edge2_u_full_px": after.get("selected_edge2_u_full_px"),
                "after_selected_center_u_full_px": after.get("selected_center_u_full_px"),
                "after_selected_width_px": after.get("selected_width_px"),
                "after_selected_height_interior_width_px": after.get("selected_height_interior_width_px"),
                "after_local_baseline_status": after.get("local_baseline_status"),
                "after_baseline_measurement_eligible": after.get("baseline_measurement_eligible"),
                "after_no_peak_pair_count": after_stage.get("NO_PEAK_PAIR", 0),
                "after_rejected_before_candidate_width_count": after_stage.get("REJECTED_BEFORE_CANDIDATE_WIDTH", 0),
                "after_generated_candidate_count": after_stage.get("GENERATED_CANDIDATE", 0),
                "manual_median_points_inside_search_roi": record["manual_median_points"],
            }
        )
    return rows


def make_target_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        after = dict(record["after_selection"])
        before = record["before_selection"]
        after.update(
            {
                "height_gt_mm": record["condition"].height_gt_mm,
                "full_candidate_count_before_search": record["full_candidate_count"],
                "manual_candidate_count_after_search": len(record["after"].get("manual_candidates") or []),
                "full_median_points": record["full_median_points"],
                "manual_median_points_inside_search_roi": record["manual_median_points"],
                "before_target_roi_status": before.get("target_roi_status"),
                "before_target_status_reason": before.get("target_status_reason"),
                "before_local_baseline_status": before.get("local_baseline_status"),
                "before_selected_center_u_full_px": before.get("selected_center_u_full_px"),
                "before_selected_width_px": before.get("selected_width_px"),
                "r5_edge_pair_min_width_px": R5_GATES["edge_pair_min_width_px"],
                "r5_transition_exclusion_margin_px": R5_GATES["transition_exclusion_margin_px"],
                "r5_height_interior_min_width_px": R5_GATES["height_interior_min_width_px"],
                "known_bad_image_exception": False,
                "corrected_h02_p01_included_normally": record["condition"].condition_id == "h02_p01",
            }
        )
        rows.append(after)
    return rows


def make_width_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, selection_key, row_key, gates in (
        ("BEFORE", "before_selection", "before_rows", BEFORE_GATES),
        ("AFTER_R5", "after_selection", "after_rows", R5_GATES),
    ):
        status_counter = Counter(record[selection_key]["target_roi_status"] for record in records)
        for status in ("FOUND", "UNCERTAIN", "NOT_FOUND"):
            rows.append(
                {
                    "phase": phase,
                    "metric": "target_roi_status",
                    "status": status,
                    "count": status_counter.get(status, 0),
                    "condition_count": status_counter.get(status, 0),
                    "parameter_values": gates,
                    "notes": "condition-level target selection status",
                }
            )
        transition = [row for record in records for row in record[row_key]]
        counts = stage_counts(transition)
        condition_counts = stage_condition_counts(transition)
        for status in ("NO_PEAK_PAIR", "REJECTED_BEFORE_CANDIDATE_WIDTH", "GENERATED_CANDIDATE"):
            rows.append(
                {
                    "phase": phase,
                    "metric": "pair_generation_stage",
                    "status": status,
                    "count": counts.get(status, 0),
                    "condition_count": condition_counts.get(status, 0),
                    "parameter_values": gates,
                    "notes": "transition/pair rows; repeated frames are not independent conditions",
                }
            )
        baseline_counter = Counter(record[selection_key]["local_baseline_status"] for record in records)
        for status in ("BOTH_AVAILABLE", "ONE_SIDE_ONLY", "UNAVAILABLE"):
            rows.append(
                {
                    "phase": phase,
                    "metric": "selected_local_baseline_status",
                    "status": status,
                    "count": baseline_counter.get(status, 0),
                    "condition_count": baseline_counter.get(status, 0),
                    "parameter_values": gates,
                    "notes": "selected candidate baseline status, recorded independently",
                }
            )
        candidates = [
            row for row in transition if row.get("pair_generation_stage") == "GENERATED_CANDIDATE"
        ]
        widths = [finite(row.get("object_width_px")) for row in candidates]
        widths = [value for value in widths if value is not None]
        in_range = sum(40 <= value <= 60 for value in widths)
        rows.append(
            {
                "phase": phase,
                "metric": "generated_object_width_px",
                "status": "40_TO_60_PX",
                "count": in_range,
                "condition_count": len(widths),
                "parameter_values": gates,
                "notes": "count is candidate-pair rows; condition_count is all generated candidate-pair rows",
            }
        )
        rows.append(
            {
                "phase": phase,
                "metric": "parameter_values",
                "status": "WIDTH_GATES",
                "count": len(gates),
                "condition_count": len(records),
                "parameter_values": gates,
                "notes": "only these three process-local values differ between phases",
            }
        )
    return rows


def draw_interval(ax: Any, interval: Any, color: str, label: str, alpha: float = 0.14) -> None:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return
    start, end = finite(interval[0]), finite(interval[1])
    if start is None or end is None or end < start:
        return
    ax.axvspan(start, end, color=color, alpha=alpha, label=label)


def selected_phase_candidate(record: dict[str, Any], phase: str) -> dict[str, Any]:
    if phase == "BEFORE":
        return selected_candidate(record["before"], record["before_selection"])
    return selected_candidate(record["after"], record["after_selection"])


def render_r5_overlay(record: dict[str, Any], polygon: np.ndarray, output_path: Path) -> bool:
    full = record["full"]
    representative = full.get("representative_result")
    image_row = full.get("representative_row") or {}
    if representative is None:
        return False
    image = np.asarray(representative.frame.image)
    if image.ndim != 2:
        return False
    x0, x1, y1, y0 = r4.image_extent(image_row, image)
    low = float(np.nanmin(image)) if image.size else 0.0
    high = float(np.nanmax(image)) if image.size else 1.0
    if high <= low:
        high = low + 1.0
    before_profile = record["before_profile"]
    after_profile = record["after_profile"]
    before_selection = record["before_selection"]
    after_selection = record["after_selection"]
    before_candidate = selected_phase_candidate(record, "BEFORE")
    after_candidate = selected_phase_candidate(record, "AFTER_R5")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    ax_image, ax_before, ax_after, ax_derivative = axes.ravel()

    ax_image.imshow(
        image,
        cmap="gray",
        vmin=low,
        vmax=high,
        extent=(x0, x1, y1, y0),
        aspect="auto",
        interpolation="nearest",
    )
    median = full.get("median_scan")
    if median is not None and len(median):
        center_uv = np.asarray(median)[:, [1, 0]]
        good = np.isfinite(center_uv).all(axis=1)
        ax_image.scatter(
            center_uv[good, 0],
            center_uv[good, 1],
            s=1.1,
            color="#ffe66d",
            alpha=0.55,
            label="full median centerline",
        )
    closed = np.vstack([polygon, polygon[0]])
    ax_image.plot(closed[:, 0], closed[:, 1], color="#00e5a0", linewidth=2.0, label="manual search ROI")
    ax_image.fill(closed[:, 0], closed[:, 1], color="#00e5a0", alpha=0.10)
    if after_candidate:
        for edge in (after_candidate.get("edge1_v"), after_candidate.get("edge2_v")):
            value = finite(edge)
            if value is not None:
                ax_image.axvline(value, color="#ff4d6d", linestyle="--", linewidth=1.4)
        draw_interval(ax_image, after_candidate.get("baseline_v_ranges", [[], []])[0], "#2878d0", "baseline-before")
        draw_interval(ax_image, after_candidate.get("height_v_range"), "#f2c94c", "height interior")
        draw_interval(ax_image, after_candidate.get("baseline_v_ranges", [[], []])[1], "#9b51e0", "baseline-after")
    ax_image.set_xlim(x0, x1)
    ax_image.set_ylim(y1, y0)
    ax_image.set_xlabel("full-sensor u (px)")
    ax_image.set_ylabel("full-sensor v (px)")
    ax_image.set_title("raw image + full median + Session search ROI")
    ax_image.legend(loc="best", fontsize=7)

    before_grid = (
        np.arange(len(before_profile["interpolated"]), dtype=np.float64)
        if before_profile.get("interpolated") is not None
        else np.empty(0)
    )
    if len(before_grid):
        ax_before.plot(before_grid, before_profile["interpolated"], color="black", linewidth=0.9, label="before profile")
    r4.draw_candidate_edges(
        ax_before,
        record["before"].get("manual_candidates") or [],
        "#e74c3c",
        before_selection.get("selected_candidate_rank"),
    )
    ax_before.set_xlim(x0, x1)
    ax_before.set_xlabel("ROI-V2 v' = full u (px)")
    ax_before.set_ylabel("ROI-V2 u' = full v (px)")
    ax_before.set_title(
        f"BEFORE: gates {int(BEFORE_GATES['edge_pair_min_width_px'])}/{BEFORE_GATES['transition_exclusion_margin_px']}/{int(BEFORE_GATES['height_interior_min_width_px'])} "
        f"| target={before_selection['target_roi_status']}"
    )
    ax_before.grid(alpha=0.18)

    after_grid = (
        np.arange(len(after_profile["interpolated"]), dtype=np.float64)
        if after_profile.get("interpolated") is not None
        else np.empty(0)
    )
    if len(after_grid):
        ax_after.plot(after_grid, after_profile["interpolated"], color="black", linewidth=0.9, label="R5 profile")
    r4.draw_candidate_edges(
        ax_after,
        record["after"].get("manual_candidates") or [],
        "#00a884",
        after_selection.get("selected_candidate_rank"),
    )
    if after_candidate:
        baselines = after_candidate.get("baseline_v_ranges") or [[], []]
        draw_interval(ax_after, baselines[0] if len(baselines) > 0 else [], "#2878d0", "baseline-before")
        draw_interval(ax_after, after_candidate.get("height_v_range"), "#f2c94c", "height interior")
        draw_interval(ax_after, baselines[1] if len(baselines) > 1 else [], "#9b51e0", "baseline-after")
    if len(after_grid):
        ax_after.set_xlim(max(x0, float(np.nanmin(after_grid))), min(x1, float(np.nanmax(after_grid))))
    else:
        ax_after.set_xlim(x0, x1)
    ax_after.set_xlabel("ROI-V2 v' = full u (px)")
    ax_after.set_ylabel("ROI-V2 u' = full v (px)")
    ax_after.set_title(
        f"AFTER R5: gates {int(R5_GATES['edge_pair_min_width_px'])}/{R5_GATES['transition_exclusion_margin_px']}/{int(R5_GATES['height_interior_min_width_px'])} "
        f"| target={after_selection['target_roi_status']} | edges/interior/baselines"
    )
    ax_after.grid(alpha=0.18)
    ax_after.legend(loc="best", fontsize=7)

    derivative = after_profile.get("derivative")
    if derivative is not None and len(after_grid):
        ax_derivative.plot(after_grid, derivative, color="black", linewidth=0.8, label="d(profile)/du")
        negative_values = after_profile.get("negative")
        positive_values = after_profile.get("positive")
        negative = np.asarray(
            negative_values if negative_values is not None else np.empty(0),
            dtype=np.int64,
        )
        positive = np.asarray(
            positive_values if positive_values is not None else np.empty(0),
            dtype=np.int64,
        )
        if len(negative):
            ax_derivative.scatter(negative, derivative[negative], color="#e74c3c", marker="v", s=18, label="negative peaks")
        if len(positive):
            ax_derivative.scatter(positive, derivative[positive], color="#2878d0", marker="^", s=18, label="positive peaks")
        r4.draw_candidate_edges(
            ax_derivative,
            record["after"].get("manual_candidates") or [],
            "#00a884",
            after_selection.get("selected_candidate_rank"),
        )
        if after_candidate:
            baselines = after_candidate.get("baseline_v_ranges") or [[], []]
            draw_interval(ax_derivative, baselines[0] if len(baselines) > 0 else [], "#2878d0", "baseline-before", 0.10)
            draw_interval(ax_derivative, after_candidate.get("height_v_range"), "#f2c94c", "height interior", 0.10)
            draw_interval(ax_derivative, baselines[1] if len(baselines) > 1 else [], "#9b51e0", "baseline-after", 0.10)
    ax_derivative.set_xlim(ax_after.get_xlim())
    ax_derivative.set_xlabel("ROI-V2 v' = full u (px)")
    ax_derivative.set_ylabel("profile derivative")
    ax_derivative.set_title("AFTER R5: derivative peaks and selected geometry")
    ax_derivative.grid(alpha=0.18)
    ax_derivative.legend(loc="best", fontsize=7)
    condition = record["condition"]
    fig.suptitle(
        f"{condition.condition_id} | full median={record['full_median_points']} | "
        f"inside search ROI={record['manual_median_points']} | no truth-based ROI selection",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def height_movement_rows(target_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[str(row["height_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["position_id"]))
    return dict(grouped)


def selected_centers(target_rows: list[dict[str, Any]], height_id: str) -> list[float]:
    values = []
    for row in height_movement_rows(target_rows).get(height_id, []):
        value = finite(row.get("selected_center_u_full_px"))
        if value is not None:
            values.append(value)
    return values


def candidate_width_stats(records: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    rows = [
        row
        for record in records
        for row in record["after_rows" if phase == "AFTER_R5" else "before_rows"]
        if row.get("pair_generation_stage") == "GENERATED_CANDIDATE"
    ]
    widths = [finite(row.get("object_width_px")) for row in rows]
    widths = [value for value in widths if value is not None]
    in_range = [value for value in widths if 40 <= value <= 60]
    return {
        "count": len(widths),
        "in_40_60_count": len(in_range),
        "in_40_60_fraction": len(in_range) / len(widths) if widths else None,
        "min": min(widths) if widths else None,
        "max": max(widths) if widths else None,
        "median": float(np.median(widths)) if widths else None,
    }


def build_report(
    output_dir: Path,
    roi_document: dict[str, Any],
    discovery: dict[str, Any],
    session_summary: dict[str, Any],
    config_contract: dict[str, Any],
    records: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    overlay_paths: list[Path],
) -> str:
    before_status = Counter(row["before_target_roi_status"] for row in target_rows)
    after_status = Counter(row["target_roi_status"] for row in target_rows)
    before_rows = [row for record in records for row in record["before_rows"]]
    after_rows = [row for record in records for row in record["after_rows"]]
    before_stages = stage_counts(before_rows)
    after_stages = stage_counts(after_rows)
    before_stage_conditions = stage_condition_counts(before_rows)
    after_stage_conditions = stage_condition_counts(after_rows)
    before_width = candidate_width_stats(records, "BEFORE")
    after_width = candidate_width_stats(records, "AFTER_R5")
    after_baseline = Counter(record["after_selection"]["local_baseline_status"] for record in records)
    after_candidates = {
        (str(row.get("orientation")), int(row["edge1_peak_v"]), int(row["edge2_peak_v"]))
        for row in after_rows
        if row.get("pair_generation_stage") == "GENERATED_CANDIDATE"
        and row.get("edge1_peak_v") not in (None, "")
        and row.get("edge2_peak_v") not in (None, "")
    }
    before_candidates = {
        (str(row.get("orientation")), int(row["edge1_peak_v"]), int(row["edge2_peak_v"]))
        for row in before_rows
        if row.get("pair_generation_stage") == "GENERATED_CANDIDATE"
        and row.get("edge1_peak_v") not in (None, "")
        and row.get("edge2_peak_v") not in (None, "")
    }
    newly_admitted = len(after_candidates - before_candidates)
    fixed_background = sum(
        1
        for row in after_rows
        if row.get("pair_generation_stage") == "GENERATED_CANDIDATE"
        and int(row.get("edge1_peak_v") or -1) == 1950
        and int(row.get("edge2_peak_v") or -1) == 2024
    )
    lines = [
        "# H0-1R5-A 海康 ROI-V2 width gate 审计",
        "",
        "- 本轮是 Haikang-specific 离线 gate 验证，不计算 h_raw、高度精度或补偿效果。",
        "- 当前数据修正：`h02_p01` 使用当前正确 recording，未复用旧 R3 的 bad-image 例外；若仍无 pair，保留给 R5-B，不调整 peak prominence。",
        "",
        "## 1. Provenance / 复用审计",
        "",
        "本轮复用：",
        "",
        "- R4 的原始 PNG/frames.csv replay、FramePipeline、Session Ground replay、Haikang axis adapter 和 Session search ROI 过滤流程。",
        "- 既有 `median_centerline`、`integer_profile`、`build_edge_pairs`、`assess_condition`、`assess_pair` 及 R3 transition/profile 诊断。",
        f"- Session search ROI：`{roi_document.get('polygon_full_uv')}`，坐标系 `{roi_document.get('coordinate_system')}`，ROI JSON SHA-256=`{roi_document.get('path_sha256')}`。",
        "",
        "本轮新增：",
        "",
        f"- 在同一 manual search profile 上分别运行 current gates `{BEFORE_GATES}` 和 R5 gates `{R5_GATES}`；只覆盖三个 width-related gate。",
        "- 生成 before/after candidate stage、target selection、baseline、width 分布和典型 overlay；没有修改模块文件中的生产参数。",
        "",
        f"目标集：{discovery.get('condition_count')} 组（{discovery.get('discovered_height_ids')} × {discovery.get('discovered_position_ids')}）；排除目录：{discovery.get('excluded_height_directories') or '无'}。",
        f"Session Ground：`{session_summary.get('status')}`，support source=`{session_summary.get('support_source')}`，SHA-256=`{session_summary.get('sha256')}`。",
        f"配置 contract：`{config_contract}`；本轮未写入生产配置。",
        "",
        "## 2. Gate before / after",
        "",
        "| phase | edge_pair_min_width_px | transition_exclusion_margin_px | height_interior_min_width_px |",
        "|---|---:|---:|---:|",
        f"| BEFORE | {BEFORE_GATES['edge_pair_min_width_px']} | {BEFORE_GATES['transition_exclusion_margin_px']} | {BEFORE_GATES['height_interior_min_width_px']} |",
        f"| AFTER_R5 | {R5_GATES['edge_pair_min_width_px']} | {R5_GATES['transition_exclusion_margin_px']} | {R5_GATES['height_interior_min_width_px']} |",
        "",
        "其它 profile、peak、support、stable-segment、baseline、score、axis adapter 和 C0/Ground 均保持不变；R5 值只在本进程临时覆盖并在每个 phase 后恢复。",
        "",
        "## 3. 全量状态比较",
        "",
        f"- condition 数：{len(records)}；full median 可用：{sum(record['full_median_points'] >= 50 for record in records)}；manual search profile 可用：{sum(record['after']['manual_profile'].get('status') == 'PROFILE_AVAILABLE' for record in records)}。",
        f"- target status BEFORE：`{dict(before_status)}`。",
        f"- target status AFTER_R5：`{dict(after_status)}`。",
        f"- pair stage BEFORE：`{dict(before_stages)}`（condition 数：`{before_stage_conditions}`）。",
        f"- pair stage AFTER_R5：`{dict(after_stages)}`（condition 数：`{after_stage_conditions}`）。",
        f"- 新增进入 generated candidate 的 pair 数：{newly_admitted}；固定 `(1950,2024)` pair 重新出现数：{fixed_background}。",
        f"- selected AFTER_R5 local baseline：`{dict(after_baseline)}`。",
        "",
        "### object width 分布",
        "",
        f"- BEFORE generated pair：{before_width}。",
        f"- AFTER_R5 generated pair：{after_width}。",
        "- 40–60 px 仅是尺寸合理性统计，不作为 candidate 选择条件，也不使用目录真实高度。",
        "",
        "## 4. 各高度 selected center（full u）",
        "",
        "| height | p01…p10 AFTER_R5 selected center | valid center count | center span |",
        "|---|---|---:|---:|",
    ]
    grouped = height_movement_rows(target_rows)
    for height in sorted(grouped):
        rows = grouped[height]
        values = [
            f"{float(row['selected_center_u_full_px']):.1f}"
            if finite(row.get("selected_center_u_full_px")) is not None
            else str(row.get("target_roi_status") or "—")
            for row in rows
        ]
        centers = [finite(row.get("selected_center_u_full_px")) for row in rows]
        centers = [value for value in centers if value is not None]
        span = f"{min(centers):.1f}…{max(centers):.1f} ({max(centers) - min(centers):.1f})" if centers else "—"
        lines.append(f"| {height} | {', '.join(values)} | {len(centers)} | {span} |")
    lines += [
        "",
        "中心仅用于 position 跟随性审计；没有参与 candidate ranking 或参数选择。",
        "",
        "## 5. 对任务问题的回答",
        "",
        f"1. 47–49 px pair 是否能生成 candidate：BEFORE 有 `{before_stages.get('REJECTED_BEFORE_CANDIDATE_WIDTH', 0)}` 个 pair 在生成阶段被拒；AFTER_R5 的最小生成宽度为 44 px，因此该类 pair 已不再因 50 px 下限直接拒绝，具体见 CSV。",
        f"2. 21–24 px interior 是否不再错误拒绝：R5 interior gate=20 px；当前 AFTER_R5 的 `height_interior_width_below_minimum` 只会保留真实低于 20 px 的失败，不再拒绝 20–24 px；本轮没有修改其它 geometry gate。",
        f"3. h06/h10/h20/h30 是否跟随 p01–p10：见上表和 `r5_target_selection.csv`；selected center 的有效覆盖为各高度实际生成 candidate 的数量，未生成者仍明确标为 NOT_FOUND。",
        f"4. 是否新增明显假 candidate：manual search 域内固定 `(1950,2024)` pair 出现 `{fixed_background}` 次；AFTER_R5 新增 pair `{newly_admitted}` 个，宽度范围/40–60 比例为 `{after_width}`，因此不能仅凭 gate PASS 认定为真实目标，必须人工抽查 overlay。",
        f"5. 是否可进入人工抽查后重跑 H0-1：可以进入“ROI 人工抽查”阶段；但 h02 仍保留给 R5-B，且本报告没有授权或执行 H0-1 h_raw 重跑，也不构成精度声明。",
        "",
        "## 6. h02 及限制",
        "",
        "- `h02_p01` 当前正确数据按普通 condition 处理；本轮不提高 peak prominence、不为它单独调 ROI、不使用高度真值。",
        "- 20 帧仍是重复帧；CSV 的 condition-level status 没有把重复帧当独立 position。",
        "- baseline-before/after 与 target status 分开记录；baseline 不会把 target pair 从 candidate generation 中删除。",
        "- 不修改 C0、Session Ground、axis adapter、candidate score 或生产配置；不执行 H1/H-B2/C1。",
        "",
        "## 7. 输出",
        "",
        f"- `r5_width_gate_summary.csv`：{output_dir / 'r5_width_gate_summary.csv'}",
        f"- `r5_target_selection.csv`：{output_dir / 'r5_target_selection.csv'}",
        f"- `r5_before_after.csv`：{output_dir / 'r5_before_after.csv'}",
        f"- overlay 数量：{len(overlay_paths)}。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    roi_path = args.roi_json.resolve()
    try:
        initial_gates = gate_snapshot()
        if initial_gates != BEFORE_GATES:
            raise AuditError(f"unexpected current ROI-V2 width gates: {initial_gates}; expected {BEFORE_GATES}")
        roi_document = r4.read_manual_roi(roi_path)
        polygon = np.asarray(roi_document["polygon_full_uv"], dtype=np.float64)
        conditions, discovery = r4.r1.h0.discover_conditions(input_dir)
        session_path = input_dir / "session_ground_calibration.json"
        reference, rotation, translation, session_summary = r4.r1.h0.load_session_reference(session_path)
        config_contract = r4.r1.h0.config_contract(args.config)
        _app, pipeline = r4.r1.h0.make_pipeline(args.config, reference, rotation, translation)
        records: list[dict[str, Any]] = []
        for index, condition in enumerate(conditions, start=1):
            print(f"[{index}/{len(conditions)}] {condition.condition_id}", flush=True)
            records.append(run_condition(condition, pipeline, polygon))
        output_dir.mkdir(parents=True, exist_ok=True)
        before_after_rows = make_before_after_rows(records)
        target_rows = make_target_rows(records)
        transition_rows = [row for record in records for row in (record["before_rows"] + record["after_rows"])]
        write_csv(output_dir / "r5_before_after.csv", before_after_rows)
        write_csv(output_dir / "r5_target_selection.csv", target_rows)
        write_csv(output_dir / "r5_width_gate_summary.csv", make_width_summary(records))
        write_csv(output_dir / "r5_candidate_audit.csv", transition_rows)
        overlay_paths: list[Path] = []
        for record in records:
            if record["condition"].condition_id not in OVERLAY_IDS:
                continue
            path = output_dir / f"{record['condition'].condition_id}_before_after_overlay.png"
            if render_r5_overlay(record, polygon, path):
                overlay_paths.append(path)
        provenance = {
            "task": "H0-1R5-A",
            "input_root": str(input_dir),
            "output_dir": str(output_dir),
            "manual_roi_json": str(roi_path),
            "manual_roi_sha256": sha256_file(roi_path),
            "manual_roi_contract": {
                "coordinate_system": roi_document.get("coordinate_system"),
                "polygon_full_uv": roi_document.get("polygon_full_uv"),
                "created_mode": roi_document.get("created_mode"),
                "purpose": roi_document.get("purpose"),
                "shared_session_roi": True,
                "directory_height_truth_used_for_selection": False,
                "board_polygon_used_as_full_search_hard_mask": False,
            },
            "data_revision_note": {
                "current_h02_p01_is_corrected_recording": True,
                "prior_r3_bad_image_label_reused": False,
                "h02_p01_frames_csv_sha256": sha256_file(input_dir / "h02" / "h02_p01" / "frames.csv"),
                "h02_p01_representative_png_sha256": sha256_file(input_dir / "h02" / "h02_p01" / "frame_000010.png"),
            },
            "reused_artifacts": {
                "r4_script": str((TOOLS_ROOT / "audit_haikang_roi_v2_manual_search_0829.py").resolve()),
                "r3_script": str((TOOLS_ROOT / "audit_haikang_roi_v2_r3_0829.py").resolve()),
                "roi_v2_profile_module": str((TOOLS_ROOT / "auto_roi_v2_session01.py").resolve()),
                "roi_v2_support_module": str((TOOLS_ROOT / "thermal_a2a_roi_v2.py").resolve()),
                "prior_r4_report": str((input_dir / "c0_height_audit" / "roi_v2_manual_search" / "roi_v2_manual_search_report.md").resolve()),
                "prior_r4_report_sha256": sha256_file(input_dir / "c0_height_audit" / "roi_v2_manual_search" / "roi_v2_manual_search_report.md"),
            },
            "new_audit_operations": [
                "replay same 50 conditions with current gates",
                "replay same manual Session search profile with only three process-local R5 gate overrides",
                "before_after target/stage/baseline/width comparison",
                "typical before_after overlays with detected edges, height interior, baseline-before, baseline-after",
            ],
            "discovery": discovery,
            "session_ground": json_safe(session_summary),
            "config_contract": json_safe(config_contract),
            "gate_comparison": {
                "before": BEFORE_GATES,
                "after_r5": R5_GATES,
                "initial_snapshot": initial_gates,
                "only_three_width_related_gates_changed": True,
                "profile_peak_support_stable_baseline_score_unchanged": True,
                "production_config_written": False,
            },
            "outputs": {
                "width_gate_summary": str((output_dir / "r5_width_gate_summary.csv").resolve()),
                "target_selection": str((output_dir / "r5_target_selection.csv").resolve()),
                "before_after": str((output_dir / "r5_before_after.csv").resolve()),
                "candidate_audit": str((output_dir / "r5_candidate_audit.csv").resolve()),
                "overlays": [str(path.resolve()) for path in overlay_paths],
            },
        }
        (output_dir / "r5_width_gate_provenance.json").write_text(
            json.dumps(json_safe(provenance), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = build_report(
            output_dir,
            roi_document,
            discovery,
            session_summary,
            config_contract,
            records,
            target_rows,
            overlay_paths,
        )
        (output_dir / "r5_width_gate_report.md").write_text(report, encoding="utf-8")
        after_status = Counter(row["target_roi_status"] for row in target_rows)
        after_stages = stage_counts([row for record in records for row in record["after_rows"]])
        print(
            json.dumps(
                {
                    "conditions": len(records),
                    "before_status": Counter(row["before_target_roi_status"] for row in before_after_rows),
                    "after_status": after_status,
                    "before_stages": stage_counts([row for record in records for row in record["before_rows"]]),
                    "after_stages": after_stages,
                    "overlays": len(overlay_paths),
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                default=dict,
            )
        )
        return 0
    except (AuditError, r4.AuditError, r4.r1.h0.AuditError, OSError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
