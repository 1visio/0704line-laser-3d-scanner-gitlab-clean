#!/usr/bin/env python3
"""Audit Haikang ROI-V2 inside a manual Session search polygon.

This is an offline ROI audit only.  It reuses the H0-1R replay and the
existing ROI-V2 implementation, filters only the median centerline by a
Session-level full-sensor polygon, and then calls the unchanged
``integer_profile`` / ``build_edge_pairs`` / ``assess_condition`` chain.
The complete frame arrays remain available for the existing repeat-support
check.  No height measurement, compensation, or production configuration is
modified.
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
from matplotlib.path import Path as MplPath  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import audit_haikang_roi_v2_0829 as r1  # noqa: E402
import audit_haikang_roi_v2_r3_0829 as r3  # noqa: E402


DATA_ROOT_DEFAULT = r1.h0.DATA_ROOT_DEFAULT
OUTPUT_DIR_DEFAULT = DATA_ROOT_DEFAULT / "c0_height_audit" / "roi_v2_manual_search"
ROI_JSON_NAME = "session_measurement_search_roi.json"
# R3's old h02_p01 exception belonged to the misplaced recording.  The user
# has replaced that source data, so R4 evaluates h02_p01 normally.
PRIOR_BAD_CONDITIONS = {"h02_p01"}
KNOWN_BAD_CONDITIONS: set[str] = set()
WIDTH_REASON = "height_interior_width_below_minimum"
REQUIRED_OVERLAY_IDS = {
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


class AuditError(RuntimeError):
    """Raised when the manual search audit contract is not satisfied."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=r1.h0.CONFIG_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument(
        "--roi-json",
        type=Path,
        default=None,
        help="manual Session ROI JSON; defaults to output-dir/session_measurement_search_roi.json",
    )
    return parser.parse_args(argv)


def finite(value: Any) -> float | None:
    return r1.h0.finite(value)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_manual_roi(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing manual Session search ROI: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read manual Session search ROI: {error}") from error
    if document.get("coordinate_system") != "full_sensor_uv":
        raise AuditError("manual ROI coordinate_system must be full_sensor_uv")
    if document.get("created_mode") != "manual":
        raise AuditError("manual ROI created_mode must be manual")
    if document.get("purpose") != "target_search_only":
        raise AuditError("manual ROI purpose must be target_search_only")
    polygon = np.asarray(document.get("polygon_full_uv"), dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise AuditError("manual ROI polygon_full_uv must contain at least 3 (u,v) points")
    if not np.isfinite(polygon).all():
        raise AuditError("manual ROI polygon_full_uv contains non-finite values")
    closed = np.vstack([polygon, polygon[0]])
    area = 0.5 * float(
        np.sum(closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1])
    )
    if abs(area) < 1.0:
        raise AuditError("manual ROI polygon has near-zero area")
    document["polygon_full_uv"] = polygon.tolist()
    document["path_sha256"] = sha256_file(path)
    return document


def points_inside_polygon(points_uv: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    values = np.asarray(points_uv, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 2:
        raise AuditError(f"unexpected full-sensor centerline shape: {values.shape}")
    # A small positive radius makes points exactly on a manually drawn edge
    # count as inside without changing the Session-level search domain.
    # Do not pass ``closed=True`` here: Matplotlib treats the last supplied
    # vertex as CLOSEPOLY and can effectively drop it.  The default Path
    # containment closes the polygon for this query and preserves all manual
    # vertices (including the left edge of a quadrilateral).
    return MplPath(np.asarray(polygon, dtype=np.float64)).contains_points(
        values, radius=1e-7
    )


def empty_profile() -> dict[str, Any]:
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


def make_profile_audit(
    full_audit: dict[str, Any],
    median_scan: np.ndarray | None,
    frame_arrays: list[np.ndarray],
    condition_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str | None]:
    """Run the unchanged ROI-V2 profile/candidate chain on one median scan."""
    result = dict(full_audit)
    result["median_scan"] = median_scan
    result["candidates"] = []
    result["direct_candidate_count"] = 0
    result["direct_detector"] = {}
    result["assessment"] = {
        "condition_id": condition_id,
        "auto_qc_status": "FAIL",
        "auto_qc_reasons": [],
        "all_edge_pairs": [],
        "detector_summary": {},
    }
    if median_scan is None or len(median_scan) < 50:
        result["assessment"]["auto_qc_reasons"] = [
            "manual_search_profile_has_fewer_than_50_points"
        ]
        return result, empty_profile(), [], result["assessment"]["auto_qc_reasons"][0]
    try:
        grid, raw, interpolated = r1.h0.roi_v2.integer_profile(median_scan)
        pairs, detector = r1.h0.roi_v2.build_edge_pairs(raw, interpolated)
        assessment = r1.h0.roi_v2.assess_condition(
            condition_id, median_scan, frame_arrays, {}
        )
        candidates = [r1.enrich_candidate(item, frame_arrays) for item in assessment.get("all_edge_pairs", [])]
        if len(candidates) != len(pairs):
            raise AuditError(
                f"manual ROI-V2 replay mismatch for {condition_id}: "
                f"build_edge_pairs={len(pairs)}, assess_condition={len(candidates)}"
            )
        result.update(
            {
                "profile_grid": grid,
                "raw_profile": raw,
                "interpolated_profile": interpolated,
                "assessment": assessment,
                "candidates": candidates,
                "direct_candidate_count": int(detector.get("candidate_pair_count", len(pairs))),
                "direct_detector": detector,
            }
        )
        profile = r3.profile_diagnostics(result)
        return result, profile, candidates, None
    except Exception as error:  # retain a controlled per-condition failure
        result["assessment"]["auto_qc_reasons"] = [
            f"MANUAL_ROI_V2_ERROR:{type(error).__name__}:{error}"
        ]
        return result, empty_profile(), [], result["assessment"]["auto_qc_reasons"][0]


def run_manual_condition(
    condition: r1.h0.Condition,
    pipeline: Any,
    polygon: np.ndarray,
) -> dict[str, Any]:
    full = r1.run_condition(condition, pipeline)
    full_profile = r3.profile_diagnostics(full)
    median = full.get("median_scan")
    retained_count = 0
    total_count = 0
    filtered: np.ndarray | None = None
    filter_error: str | None = None
    if median is not None:
        values = np.asarray(median, dtype=np.float64)
        total_count = len(values)
        full_uv = values[:, [1, 0]]
        mask = points_inside_polygon(full_uv, polygon)
        retained_count = int(mask.sum())
        filtered = values[mask]
    else:
        filter_error = "full_median_centerline_unavailable"
    manual, manual_profile, candidates, manual_error = make_profile_audit(
        full, filtered, full.get("frame_arrays") or [], condition.condition_id
    )
    return {
        "condition": condition,
        "full": full,
        "manual": manual,
        "full_profile": full_profile,
        "manual_profile": manual_profile,
        "manual_candidates": candidates,
        "full_candidate_count": len(full.get("candidates") or []),
        "manual_candidate_count": len(candidates),
        "full_median_points": total_count,
        "manual_median_points": retained_count,
        "manual_filter_error": filter_error,
        "manual_error": manual_error,
        "transition_rows": [],
    }


def candidate_key(candidate: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(candidate.get("orientation")),
        int(candidate.get("edge1_peak_v")),
        int(candidate.get("edge2_peak_v")),
    )


def current_width_threshold() -> float:
    return float(r1.h0.roi_v2.PARAMETERS["object_interval"]["height_interior_min_width_px"])


def adjusted_candidate(candidate: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Change only the audited width gate for a ranking what-if row."""
    item = dict(candidate)
    pair_reasons = [
        str(reason)
        for reason in (candidate.get("pair_gate_reasons") or [])
        if str(reason) != WIDTH_REASON
    ]
    height_width = finite(candidate.get("height_interior_width_px"))
    width_fails = height_width is None or height_width < float(threshold)
    if width_fails:
        pair_reasons.append(WIDTH_REASON)
    item["pair_gate_reasons"] = list(dict.fromkeys(pair_reasons))
    item["edge_pair_geometry_ok"] = not item["pair_gate_reasons"]
    multi_reasons = [
        str(reason)
        for reason in (candidate.get("multi_geometry_reasons") or [])
        if str(reason) != WIDTH_REASON
    ]
    if width_fails:
        multi_reasons.append(WIDTH_REASON)
    item["multi_geometry_reasons"] = list(dict.fromkeys(multi_reasons))
    item["multi_geometry_ok"] = not item["multi_geometry_reasons"]
    return item


def baseline_for(candidate: dict[str, Any]) -> dict[str, Any]:
    return r3.baseline_evaluation(candidate)


def select_target(
    record: dict[str, Any], threshold: float, *, diagnostic_bad_image: bool = True
) -> dict[str, Any]:
    condition = record["condition"]
    candidates = list(record.get("manual_candidates") or [])
    base: dict[str, Any] = {
        "condition_id": condition.condition_id,
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "known_bad_image_exception": condition.condition_id in KNOWN_BAD_CONDITIONS,
        "target_evaluable_for_position_consensus": condition.condition_id not in KNOWN_BAD_CONDITIONS,
        "threshold_px": float(threshold),
        "target_roi_status": "NOT_FOUND",
        "target_status_reason": "NO_GENERATED_CANDIDATE",
        "candidate_count_generated": len(candidates),
        "candidate_count_width_gate_pass": 0,
        "selected_candidate_rank": None,
        "selected_candidate_id": None,
        "selected_edge1_u_full_px": None,
        "selected_edge2_u_full_px": None,
        "selected_center_u_full_px": None,
        "selected_width_px": None,
        "selected_height_interior_width_px": None,
        "selected_step_amplitude_px": None,
        "selected_pair_score": None,
        "selected_pair_gate_reasons": [],
        "selected_target_core_gate_reasons": [],
        "selected_other_gate_failures": [],
        "local_baseline_status": "UNAVAILABLE",
        "baseline_measurement_eligible": False,
        "baseline_failure_reasons": [],
        "manual_search_profile_status": record["manual_profile"].get("status"),
    }
    adjusted: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for rank, original in enumerate(candidates, start=1):
        item = adjusted_candidate(original, threshold)
        adjusted.append((rank, original, item))
    base["candidate_count_width_gate_pass"] = sum(
        1
        for _rank, _original, item in adjusted
        if WIDTH_REASON not in (item.get("pair_gate_reasons") or [])
    )
    if not adjusted:
        if record.get("manual_error"):
            base["target_status_reason"] = record["manual_error"]
        if condition.condition_id in KNOWN_BAD_CONDITIONS and diagnostic_bad_image:
            base["target_status_reason"] = "KNOWN_BAD_IMAGE_EXPECTED_NO_ROI"
        return base
    rank, original, item = max(
        adjusted, key=lambda triple: r3.existing_sort_key(triple[2])
    )
    baseline = baseline_for(original)
    core_reasons = r3.target_core_reasons(item)
    status = "FOUND" if not core_reasons else "UNCERTAIN"
    if condition.condition_id in KNOWN_BAD_CONDITIONS and diagnostic_bad_image:
        status = "NOT_FOUND"
        reason = "KNOWN_BAD_IMAGE_EXPECTED_NO_ROI"
    else:
        reason = "TARGET_CANDIDATE_FOUND" if status == "FOUND" else "CANDIDATE_REMAINS_BUT_TARGET_CORE_GATE_FAILED"
    edge1 = finite(original.get("edge1_v"))
    edge2 = finite(original.get("edge2_v"))
    center = (edge1 + edge2) / 2.0 if edge1 is not None and edge2 is not None else None
    base.update(
        {
            "target_roi_status": status,
            "target_status_reason": reason,
            "selected_candidate_rank": rank,
            "selected_candidate_id": f"{condition.condition_id}:manual:r{rank}",
            "selected_edge1_u_full_px": edge1,
            "selected_edge2_u_full_px": edge2,
            "selected_center_u_full_px": center,
            "selected_width_px": finite(original.get("object_width_px")),
            "selected_height_interior_width_px": finite(original.get("height_interior_width_px")),
            "selected_step_amplitude_px": finite(original.get("step_amplitude_px")),
            "selected_pair_score": finite(original.get("pair_score")),
            "selected_pair_gate_reasons": item.get("pair_gate_reasons") or [],
            "selected_target_core_gate_reasons": core_reasons,
            "selected_other_gate_failures": [
                reason_item for reason_item in core_reasons if reason_item != WIDTH_REASON
            ],
            "local_baseline_status": baseline["local_baseline_status"],
            "baseline_measurement_eligible": baseline["baseline_measurement_eligible"],
            "baseline_failure_reasons": baseline["baseline_failure_reasons"],
        }
    )
    return base


def make_transition_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    manual = record["manual"]
    profile = record["manual_profile"]
    rows = r3.build_transition_rows(manual, profile)
    by_key = {candidate_key(candidate): (rank, candidate) for rank, candidate in enumerate(record["manual_candidates"], start=1)}
    base_selection = select_target(record, current_width_threshold())
    selected_rank = base_selection.get("selected_candidate_rank")
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        key = (
            str(row.get("orientation")),
            int(row["edge1_peak_v"]) if row.get("edge1_peak_v") not in (None, "") else None,
            int(row["edge2_peak_v"]) if row.get("edge2_peak_v") not in (None, "") else None,
        )
        pair = by_key.get(key) if None not in key else None
        rank = pair[0] if pair else None
        candidate = pair[1] if pair else None
        baseline = baseline_for(candidate) if candidate else {}
        generated = bool(row.get("candidate_generated"))
        if generated and candidate:
            item.update(
                {
                    "edge1_u_full_px": finite(candidate.get("edge1_v")),
                    "edge2_u_full_px": finite(candidate.get("edge2_v")),
                    "candidate_rank": rank,
                    "candidate_id": f"{record['condition'].condition_id}:manual:r{rank}",
                    "local_baseline_status": baseline.get("local_baseline_status"),
                    "baseline_measurement_eligible": baseline.get("baseline_measurement_eligible"),
                    "baseline_failure_reasons": baseline.get("baseline_failure_reasons"),
                    "target_core_gate_reasons": r3.target_core_reasons(candidate),
                    "candidate_width_gate_pass_current": finite(candidate.get("height_interior_width_px")) is not None
                    and finite(candidate.get("height_interior_width_px")) >= current_width_threshold(),
                    "target_candidate_state": (
                        "SELECTED_FOUND"
                        if rank == selected_rank and base_selection["target_roi_status"] == "FOUND"
                        else "SELECTED_UNCERTAIN"
                        if rank == selected_rank
                        else "GENERATED_NOT_SELECTED"
                    ),
                }
            )
        else:
            item.update(
                {
                    "candidate_id": "",
                    "local_baseline_status": "",
                    "baseline_measurement_eligible": "",
                    "baseline_failure_reasons": "",
                    "target_core_gate_reasons": "",
                    "candidate_width_gate_pass_current": "",
                    "target_candidate_state": "REJECTED_BEFORE_CANDIDATE",
                }
            )
        item.update(
            {
                "height_gt_mm": record["condition"].height_gt_mm,
                "known_bad_image_exception": record["condition"].condition_id in KNOWN_BAD_CONDITIONS,
                "full_candidate_count_before_search": record["full_candidate_count"],
                "manual_candidate_count_after_search": record["manual_candidate_count"],
                "full_median_points": record["full_median_points"],
                "manual_median_points_inside_search_roi": record["manual_median_points"],
                "manual_search_profile_status": profile.get("status"),
                "manual_search_error": record.get("manual_error") or record.get("manual_filter_error"),
                "base_target_roi_status": base_selection["target_roi_status"],
                "base_local_baseline_status": base_selection["local_baseline_status"],
                "base_selected_center_u_full_px": base_selection["selected_center_u_full_px"],
                "base_selected_candidate_rank": base_selection["selected_candidate_rank"],
            }
        )
        output.append(item)
    return output


def make_condition_summary(record: dict[str, Any]) -> dict[str, Any]:
    selected = select_target(record, current_width_threshold())
    return {
        **selected,
        "height_gt_mm": record["condition"].height_gt_mm,
        "full_candidate_count_before_search": record["full_candidate_count"],
        "manual_candidate_count_after_search": record["manual_candidate_count"],
        "full_median_points": record["full_median_points"],
        "manual_median_points_inside_search_roi": record["manual_median_points"],
        "manual_search_error": record.get("manual_error") or record.get("manual_filter_error"),
    }


def derive_width_thresholds(records: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    current = int(round(current_width_threshold()))
    failed = sorted(
        {
            int(round(float(candidate["height_interior_width_px"])))
            for record in records
            for candidate in record.get("manual_candidates") or []
            if finite(candidate.get("height_interior_width_px")) is not None
            and float(candidate["height_interior_width_px"]) < current
        }
    )
    thresholds = {current}
    if failed:
        thresholds.add(max(1, failed[0] - 1))
        thresholds.add(failed[-1] + 1)
        thresholds.update(value for value in failed if value > 0)
    return sorted(thresholds), failed


def make_width_rows(
    records: list[dict[str, Any]], thresholds: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        condition = record["condition"]
        for threshold in thresholds:
            selected = select_target(record, threshold)
            candidate_items = [
                (rank, candidate, adjusted_candidate(candidate, threshold))
                for rank, candidate in enumerate(record.get("manual_candidates") or [], start=1)
            ]
            selected_rank = selected.get("selected_candidate_rank")
            rows.append(
                {
                    "row_type": "CONDITION_SUMMARY",
                    "condition_id": condition.condition_id,
                    "height_id": condition.height_id,
                    "height_gt_mm": condition.height_gt_mm,
                    "position_id": condition.position_id,
                    "known_bad_image_exception": condition.condition_id in KNOWN_BAD_CONDITIONS,
                    "threshold_px": threshold,
                    "manual_search_profile_status": record["manual_profile"].get("status"),
                    "manual_candidate_count": len(candidate_items),
                    "candidate_width_gate_pass_count": selected["candidate_count_width_gate_pass"],
                    "target_roi_status": selected["target_roi_status"],
                    "target_status_reason": selected["target_status_reason"],
                    "selected_candidate_rank": selected_rank,
                    "selected_candidate_id": selected.get("selected_candidate_id"),
                    "selected_edge1_u_full_px": selected.get("selected_edge1_u_full_px"),
                    "selected_edge2_u_full_px": selected.get("selected_edge2_u_full_px"),
                    "selected_center_u_full_px": selected.get("selected_center_u_full_px"),
                    "selected_width_px": selected.get("selected_width_px"),
                    "selected_height_interior_width_px": selected.get("selected_height_interior_width_px"),
                    "selected_step_amplitude_px": selected.get("selected_step_amplitude_px"),
                    "selected_pair_score": selected.get("selected_pair_score"),
                    "selected_pair_gate_reasons": selected.get("selected_pair_gate_reasons"),
                    "selected_target_core_gate_reasons": selected.get("selected_target_core_gate_reasons"),
                    "selected_other_gate_failures": selected.get("selected_other_gate_failures"),
                    "local_baseline_status": selected.get("local_baseline_status"),
                    "baseline_measurement_eligible": selected.get("baseline_measurement_eligible"),
                    "baseline_failure_reasons": selected.get("baseline_failure_reasons"),
                    "target_evaluable_for_position_consensus": selected.get("target_evaluable_for_position_consensus"),
                    "height_interior_width_px": "",
                    "candidate_rank": "",
                    "candidate_id": "",
                    "candidate_width_gate_pass": "",
                    "pair_geometry_pass_if_width_only_changed": "",
                    "candidate_pair_gate_reasons_current": "",
                    "candidate_other_gate_failures": "",
                    "candidate_target_core_gate_reasons_at_threshold": "",
                    "candidate_local_baseline_status": "",
                    "candidate_baseline_measurement_eligible": "",
                    "candidate_selected_for_threshold": "",
                }
            )
            for rank, candidate, adjusted in candidate_items:
                baseline = baseline_for(candidate)
                core = r3.target_core_reasons(adjusted)
                rows.append(
                    {
                        "row_type": "CANDIDATE",
                        "condition_id": condition.condition_id,
                        "height_id": condition.height_id,
                        "height_gt_mm": condition.height_gt_mm,
                        "position_id": condition.position_id,
                        "known_bad_image_exception": condition.condition_id in KNOWN_BAD_CONDITIONS,
                        "threshold_px": threshold,
                        "manual_search_profile_status": record["manual_profile"].get("status"),
                        "manual_candidate_count": len(candidate_items),
                        "candidate_width_gate_pass_count": "",
                        "target_roi_status": "",
                        "target_status_reason": "",
                        "selected_candidate_rank": "",
                        "selected_candidate_id": "",
                        "selected_edge1_u_full_px": "",
                        "selected_edge2_u_full_px": "",
                        "selected_center_u_full_px": "",
                        "selected_width_px": "",
                        "selected_height_interior_width_px": "",
                        "selected_step_amplitude_px": "",
                        "selected_pair_score": "",
                        "selected_pair_gate_reasons": "",
                        "selected_target_core_gate_reasons": "",
                        "selected_other_gate_failures": "",
                        "local_baseline_status": "",
                        "baseline_measurement_eligible": "",
                        "baseline_failure_reasons": "",
                        "target_evaluable_for_position_consensus": condition.condition_id not in KNOWN_BAD_CONDITIONS,
                        "height_interior_width_px": finite(candidate.get("height_interior_width_px")),
                        "candidate_rank": rank,
                        "candidate_id": f"{condition.condition_id}:manual:r{rank}",
                        "candidate_width_gate_pass": WIDTH_REASON not in (adjusted.get("pair_gate_reasons") or []),
                        "pair_geometry_pass_if_width_only_changed": bool(adjusted.get("edge_pair_geometry_ok")),
                        "candidate_pair_gate_reasons_current": candidate.get("pair_gate_reasons"),
                        "candidate_other_gate_failures": [reason for reason in core if reason != WIDTH_REASON],
                        "candidate_target_core_gate_reasons_at_threshold": core,
                        "candidate_local_baseline_status": baseline.get("local_baseline_status"),
                        "candidate_baseline_measurement_eligible": baseline.get("baseline_measurement_eligible"),
                        "candidate_selected_for_threshold": rank == selected_rank,
                    }
                )
    return rows


def image_extent(row: dict[str, str], image: np.ndarray) -> tuple[float, float, float, float]:
    x0 = float(r1.h0.parse_int(row["offset_x"], "offset_x"))
    y0 = float(r1.h0.parse_int(row["offset_y"], "offset_y"))
    width = float(r1.h0.parse_int(row["width"], "width"))
    height = float(r1.h0.parse_int(row["height"], "height"))
    return x0, x0 + width, y0 + height, y0


def draw_candidate_edges(ax: Any, candidates: list[dict[str, Any]], color: str, selected_rank: int | None) -> None:
    for rank, candidate in enumerate(candidates, start=1):
        edge1 = finite(candidate.get("edge1_v"))
        edge2 = finite(candidate.get("edge2_v"))
        if edge1 is None or edge2 is None:
            continue
        selected = rank == selected_rank
        ax.axvline(edge1, color=color if selected else "#888888", linewidth=2.2 if selected else 0.65, alpha=0.95 if selected else 0.5)
        ax.axvline(edge2, color=color if selected else "#888888", linewidth=2.2 if selected else 0.65, alpha=0.95 if selected else 0.5)
        if selected:
            ax.axvspan(edge1, edge2, color=color, alpha=0.08, label=f"selected r{rank}")


def render_before_after(record: dict[str, Any], polygon: np.ndarray, output_path: Path) -> bool:
    condition = record["condition"]
    full = record["full"]
    image_result = full.get("representative_result")
    image_row = full.get("representative_row") or {}
    if image_result is None:
        return False
    image = np.asarray(image_result.frame.image)
    if image.ndim != 2:
        return False
    x0, x1, y1, y0 = image_extent(image_row, image)
    low = float(np.nanmin(image)) if image.size else 0.0
    high = float(np.nanmax(image)) if image.size else 1.0
    if high <= low:
        high = low + 1.0
    full_profile = record["full_profile"]
    manual_profile = record["manual_profile"]
    full_selection = select_target(record, current_width_threshold())
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    ax_image, ax_full, ax_manual, ax_derivative = axes.ravel()
    ax_image.imshow(image, cmap="gray", vmin=low, vmax=high, extent=(x0, x1, y1, y0), aspect="auto", interpolation="nearest")
    median = full.get("median_scan")
    if median is not None and len(median):
        center_uv = np.asarray(median)[:, [1, 0]]
        good = np.isfinite(center_uv).all(axis=1)
        ax_image.scatter(center_uv[good, 0], center_uv[good, 1], s=1.1, color="#ffe66d", alpha=0.55, label="full median centerline")
    closed = np.vstack([polygon, polygon[0]])
    ax_image.plot(closed[:, 0], closed[:, 1], color="#00e5a0", linewidth=2.0, label="manual search ROI")
    ax_image.fill(closed[:, 0], closed[:, 1], color="#00e5a0", alpha=0.10)
    if full_selection.get("selected_edge1_u_full_px") is not None:
        for edge in (full_selection["selected_edge1_u_full_px"], full_selection["selected_edge2_u_full_px"]):
            ax_image.axvline(edge, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_image.set_xlim(x0, x1)
    ax_image.set_ylim(y1, y0)
    ax_image.set_xlabel("full-sensor u (px)")
    ax_image.set_ylabel("full-sensor v (px)")
    ax_image.set_title("raw image + full median + Session search polygon")
    ax_image.legend(loc="best", fontsize=7)

    full_grid = np.arange(len(full_profile["interpolated"]), dtype=np.float64) if full_profile.get("interpolated") is not None else np.empty(0)
    if len(full_grid):
        ax_full.plot(full_grid, full_profile["interpolated"], color="black", linewidth=0.9, label="full profile")
        good = np.isfinite(full_profile["raw"])
        ax_full.scatter(full_grid[good], full_profile["raw"][good], s=1.0, color="#888888", alpha=0.3)
    draw_candidate_edges(ax_full, full.get("candidates") or [], "#e74c3c", None)
    ax_full.set_xlim(x0, x1)
    ax_full.set_xlabel("ROI-V2 v' = full u (px)")
    ax_full.set_ylabel("ROI-V2 u' = full v (px)")
    ax_full.set_title(f"BEFORE: full search, candidates={len(full.get('candidates') or [])}")
    ax_full.grid(alpha=0.18)

    manual_grid = np.arange(len(manual_profile["interpolated"]), dtype=np.float64) if manual_profile.get("interpolated") is not None else np.empty(0)
    if len(manual_grid):
        ax_manual.plot(manual_grid, manual_profile["interpolated"], color="black", linewidth=0.9, label="manual-filtered profile")
        good = np.isfinite(manual_profile["raw"])
        ax_manual.scatter(manual_grid[good], manual_profile["raw"][good], s=1.0, color="#888888", alpha=0.3)
    draw_candidate_edges(ax_manual, record.get("manual_candidates") or [], "#00a884", full_selection.get("selected_candidate_rank"))
    if len(manual_grid):
        ax_manual.set_xlim(max(x0, float(np.nanmin(manual_grid))), min(x1, float(np.nanmax(manual_grid))))
    else:
        ax_manual.set_xlim(x0, x1)
    ax_manual.set_xlabel("ROI-V2 v' = full u (px)")
    ax_manual.set_ylabel("ROI-V2 u' = full v (px)")
    ax_manual.set_title(
        f"AFTER: inside manual search ROI, candidates={record['manual_candidate_count']} | "
        f"target={full_selection['target_roi_status']}"
    )
    ax_manual.grid(alpha=0.18)

    derivative = manual_profile.get("derivative")
    if derivative is not None and len(manual_grid):
        ax_derivative.plot(manual_grid, derivative, color="black", linewidth=0.8, label="d(profile)/du")
        negative = manual_profile.get("negative")
        positive = manual_profile.get("positive")
        negative = np.asarray(negative if negative is not None else np.empty(0), dtype=np.int64)
        positive = np.asarray(positive if positive is not None else np.empty(0), dtype=np.int64)
        if len(negative):
            ax_derivative.scatter(negative, derivative[negative], color="#e74c3c", marker="v", s=18, label="negative peaks")
        if len(positive):
            ax_derivative.scatter(positive, derivative[positive], color="#2878d0", marker="^", s=18, label="positive peaks")
        draw_candidate_edges(ax_derivative, record.get("manual_candidates") or [], "#00a884", full_selection.get("selected_candidate_rank"))
    ax_derivative.set_xlim(ax_manual.get_xlim())
    ax_derivative.set_xlabel("ROI-V2 v' = full u (px)")
    ax_derivative.set_ylabel("profile derivative")
    ax_derivative.set_title("AFTER: derivative peaks and generated pairs")
    ax_derivative.grid(alpha=0.18)
    ax_derivative.legend(loc="best", fontsize=7)
    fig.suptitle(
        f"{condition.condition_id} | full median points={record['full_median_points']} | "
        f"inside manual search={record['manual_median_points']} | "
        "search-domain audit only; no truth-based ROI selection",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def summarize_height_movement(summaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        if row.get("known_bad_image_exception"):
            continue
        grouped[str(row["height_id"])].append(row)
    return {height: sorted(rows, key=lambda row: str(row["position_id"])) for height, rows in grouped.items()}


def classify(
    records: list[dict[str, Any]],
    width_rows: list[dict[str, Any]],
    thresholds: list[int],
) -> tuple[str, dict[str, Any]]:
    normal = [record for record in records if record["condition"].condition_id not in KNOWN_BAD_CONDITIONS]
    summaries = [make_condition_summary(record) for record in normal]
    base_found = sum(row["target_roi_status"] == "FOUND" for row in summaries)
    generated = sum(row["manual_candidate_count_after_search"] > 0 for row in summaries)
    lower_thresholds = [value for value in thresholds if value < int(round(current_width_threshold()))]
    lower_summary_rows = [
        row
        for row in width_rows
        if row.get("row_type") == "CONDITION_SUMMARY"
        and row.get("threshold_px") in lower_thresholds
        and not row.get("known_bad_image_exception")
    ]
    recoverable: set[str] = {
        str(row["condition_id"])
        for row in lower_summary_rows
        if row.get("target_roi_status") == "FOUND"
    }
    all_lower_found = {str(row["condition_id"]) for row in lower_summary_rows if row.get("target_roi_status") == "FOUND"}
    no_nonwidth = 0
    for condition_id in {record["condition"].condition_id for record in normal}:
        rows = [row for row in lower_summary_rows if row.get("condition_id") == condition_id]
        if rows and any(not row.get("selected_other_gate_failures") for row in rows if row.get("target_roi_status") == "FOUND"):
            no_nonwidth += 1
    if normal and base_found == len(normal):
        classification = "MANUAL_SEARCH_ROI_SUFFICIENT"
    elif not generated:
        classification = "TARGET_SIGNAL_TOO_WEAK"
    elif recoverable and len(recoverable) >= max(1, math.ceil(0.70 * len(normal))) and no_nonwidth >= len(recoverable):
        classification = "HAIKANG_WIDTH_GATE_ADAPTATION_REQUIRED"
    elif recoverable:
        classification = "MIXED"
    else:
        classification = "TARGET_SIGNAL_TOO_WEAK" if generated == 0 else "MIXED"
    return classification, {
        "normal_condition_count": len(normal),
        "base_found_count": base_found,
        "generated_candidate_condition_count": generated,
        "lower_thresholds": lower_thresholds,
        "width_recoverable_condition_ids": sorted(recoverable),
        "lower_found_count": len(all_lower_found),
        "lower_found_without_nonwidth_gate_count": no_nonwidth,
    }


def fmt(value: Any, digits: int = 2) -> str:
    number = finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def build_report(
    output_dir: Path,
    roi_document: dict[str, Any],
    session_summary: dict[str, Any],
    config_summary: dict[str, Any],
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    width_rows: list[dict[str, Any]],
    thresholds: list[int],
    failed_widths: list[int],
    classification: str,
    classification_detail: dict[str, Any],
    overlay_paths: list[Path],
) -> str:
    normal_summaries = [row for row in summaries if not row["known_bad_image_exception"]]
    by_height = summarize_height_movement(summaries)
    status_counts = Counter(row["target_roi_status"] for row in summaries)
    baseline_counts = Counter(row["local_baseline_status"] for row in summaries)
    transition_rows = [row for record in records for row in record["transition_rows"]]
    stage_counts = Counter(row.get("pair_generation_stage") for row in transition_rows)
    pre_candidate_widths = Counter(
        int(round(float(row["object_width_px"])))
        for row in transition_rows
        if row.get("pair_generation_stage") == "REJECTED_BEFORE_CANDIDATE_WIDTH"
        and finite(row.get("object_width_px")) is not None
    )
    generated_candidates = [
        row for row in transition_rows
        if row.get("pair_generation_stage") == "GENERATED_CANDIDATE"
    ]
    generated_widths = Counter(
        int(round(float(row["height_interior_width_px"])))
        for row in generated_candidates
        if finite(row.get("height_interior_width_px")) is not None
    )
    lower_rows = [
        row for row in width_rows
        if row.get("row_type") == "CONDITION_SUMMARY"
        and int(row.get("threshold_px") or 0) < int(round(current_width_threshold()))
        and not row.get("known_bad_image_exception")
    ]
    lines = [
        "# H0-1R4 海康 ROI-V2 Session Search ROI + width gate 审计",
        "",
        f"- 结论分类：`{classification}`",
        "- 本轮范围：仅 ROI 搜索域和 `height_interior_min_width_px` what-if（假设）审计；不计算高度精度。",
        "- 数据版本修正：此前 R3 的 `h02_p01` 例外对应错放数据；当前已替换为正确数据，本轮按普通 condition 纳入统计。",
        "",
        "## 1. Provenance / 复用审计",
        "",
        "本轮复用：",
        "",
        "- H0-1R 的 `run_condition`、原始 PNG/frames.csv replay、FramePipeline（帧处理管线）、Session Ground 应用和 Haikang axis adapter（坐标适配）。",
        "- 既有 `thermal_a2a_roi_v2.median_centerline`（中位中心线）与 `auto_roi_v2_session01.integer_profile`、`build_edge_pairs`、`assess_condition`。",
        "- 既有 `thermal_a2a_roi_v2.assess_pair` 的重复帧 support（支撑）检查，以及 H0-1R3 的 transition/profile（跃迁/剖面）诊断和绘图结构。",
        "",
        "本轮新增：",
        "",
        "- 一次性的 Session 级人工 `polygon_full_uv` search ROI；只在 full median centerline 上做 profile 搜索域过滤，不指定任何 condition 的最终高度 ROI。",
        "- 全 50 组的 before/after 候选重放、候选/transition CSV，以及只改变 width gate 的敏感性表。",
        "",
        f"目标集：{session_summary.get('condition_count')} 组（{session_summary.get('discovered_height_ids')} × {session_summary.get('discovered_position_ids')}）；额外目录排除：{session_summary.get('excluded_height_directories') or '无'}。",
        f"Session Ground：`{config_summary.get('session_ground', {}).get('status')}`，support source=`{config_summary.get('session_ground', {}).get('support_source')}`，文件 SHA-256=`{config_summary.get('session_ground', {}).get('sha256')}`。",
        f"生产配置审计：correction.mode=none，stage-A height scale disabled，H-B2 config absent；本轮未写入配置。",
        "",
        "## 2. Manual search ROI",
        "",
        f"- 坐标系：`{roi_document.get('coordinate_system')}`。",
        f"- `created_mode={roi_document.get('created_mode')}`，`purpose={roi_document.get('purpose')}`，selection method=`{roi_document.get('selection_method')}`。",
        f"- source frame：`{(roi_document.get('source_frame') or {}).get('condition_id')}` / `{(roi_document.get('source_frame') or {}).get('filename')}`；仅作代表性图像上下文，不使用目录高度真值。",
        f"- polygon：`{json.dumps(roi_document.get('polygon_full_uv'), ensure_ascii=False)}`。",
        "- board polygon 没有作为整个搜索域 hard mask；本轮的主要 prior（先验）是这个人工 Session search polygon。",
        "",
        "## 3. 全量结果",
        "",
        f"- 条件数：{len(records)}；full median centerline 可用：{sum(record['full_median_points'] >= 50 for record in records)}；manual search profile 可用：{sum(record['manual_profile'].get('status') == 'PROFILE_AVAILABLE' for record in records)}。",
        f"- full 候选总数：{sum(record['full_candidate_count'] for record in records)}；manual search 后候选总数：{sum(record['manual_candidate_count'] for record in records)}。",
        f"- manual profile 的 pair-generation stages：{dict(stage_counts)}。",
        f"- `REJECTED_BEFORE_CANDIDATE_WIDTH` 的 object_width_px：{dict(sorted(pre_candidate_widths.items())) or '无'}；这些 pair 没有进入后续 `height_interior_min_width_px` sensitivity，不能被事后宽度表当作已生成 candidate。",
        f"- 已生成 candidate 的 `height_interior_width_px`：{dict(sorted(generated_widths.items())) or '无'}；当前 30 px gate 下它们仍需通过后续 height-interior 几何检查。",
        f"- 当前 width gate={int(round(current_width_threshold()))} px：target status={dict(status_counts)}；local baseline status={dict(baseline_counts)}。",
        f"- `h02_p01`：`{next((row['target_status_reason'] for row in summaries if row['condition_id'] == 'h02_p01'), '未找到')}`；已按当前正确数据纳入 50 组。",
        "",
        "### 当前门限下按高度的位置中心",
        "",
        "| height | p01…p10 selected center (full u, px) | normal position center span |",
        "|---|---|---|",
    ]
    for height in sorted(by_height):
        rows = by_height[height]
        values = [fmt(row.get("selected_center_u_full_px")) if row.get("selected_center_u_full_px") is not None else row.get("target_roi_status", "—") for row in rows]
        centers = [finite(row.get("selected_center_u_full_px")) for row in rows]
        centers = [value for value in centers if value is not None]
        span = f"{min(centers):.1f}…{max(centers):.1f} ({max(centers)-min(centers):.1f} px)" if centers else "—"
        lines.append(f"| {height} | {', '.join(values)} | {span} |")
    lines += [
        "",
        "中心位置仅用于 ROI 跟随性诊断；没有用来选择 polygon、candidate 或 width threshold。",
        "",
        "## 4. Width sensitivity / 宽度敏感性",
        "",
        f"当前 `height_interior_min_width_px={int(round(current_width_threshold()))}`；本轮从 manual search 域内实际失败 candidate 的 width 分布导出测试门限。",
        f"观察到的失败 `height_interior_width_px`：`{failed_widths or '无'}`；测试门限：`{thresholds}`。",
        "这些是 post-hoc（事后）候选恢复 what-if，不是已应用参数，也不构成精度改善声明；所有其它 geometry gate、score、axis adapter、C0 和 Ground 均保持不变。",
        "",
        "| threshold px | normal FOUND | normal UNCERTAIN | normal NOT_FOUND |",
        "|---:|---:|---:|---:|",
    ]
    for threshold in thresholds:
        rows = [row for row in width_rows if row.get("row_type") == "CONDITION_SUMMARY" and int(row.get("threshold_px") or 0) == threshold and not row.get("known_bad_image_exception")]
        counts = Counter(row.get("target_roi_status") for row in rows)
        lines.append(f"| {threshold} | {counts.get('FOUND', 0)} | {counts.get('UNCERTAIN', 0)} | {counts.get('NOT_FOUND', 0)} |")
    if lower_rows:
        lines += [
            "",
            "降低门限后得到 `FOUND` 只表示候选通过当前 geometry core（几何核心）门；仍需独立 held-out（留出）数据确认，不能用本批数据宣称补偿效果。",
        ]
    else:
        lines += ["", "manual search 域内没有可用于降低 width gate 的失败 candidate。"]
    lines += [
        "",
        "## 5. 状态与限制",
        "",
        f"- 分类判据摘要：正常条件 {classification_detail.get('normal_condition_count')}；当前门限 FOUND {classification_detail.get('base_found_count')}；存在生成 candidate {classification_detail.get('generated_candidate_condition_count')}；宽度降低后可 FOUND 的条件数 {classification_detail.get('lower_found_count')}。",
        f"- width-only recovery 候选条件：`{classification_detail.get('width_recoverable_condition_ids') or '无'}`。",
        "- baseline status（BOTH_AVAILABLE / ONE_SIDE_ONLY / UNAVAILABLE）独立记录；baseline 不会把 target candidate 从搜索候选中删除。",
        "- `h02_p01` 的候选/transition 在 CSV 和 overlay 中，并按当前正确数据参与正常结论。",
        "- 本轮没有调用 local height measurement、H1、H-B2 或 C1，也没有写入任何生产配置；因此不能把本报告当作 h_raw 精度报告。",
        "",
        "## 6. 输出",
        "",
        f"- 候选审计：`{output_dir / 'manual_search_candidate_audit.csv'}`",
        f"- width 敏感性：`{output_dir / 'width_gate_sensitivity.csv'}`",
        f"- condition 选择汇总：`{output_dir / 'roi_target_selection.csv'}`",
        f"- before/after overlay 数量：{len(overlay_paths)}。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    roi_path = (args.roi_json or output_dir / ROI_JSON_NAME).resolve()
    try:
        roi_document = read_manual_roi(roi_path)
        polygon = np.asarray(roi_document["polygon_full_uv"], dtype=np.float64)
        conditions, discovery = r1.h0.discover_conditions(args.input_dir)
        session_path = args.input_dir / "session_ground_calibration.json"
        reference, rotation, translation, session_summary = r1.h0.load_session_reference(session_path)
        config_summary = {
            "session_ground": session_summary,
            "config": r1.h0.config_contract(args.config),
        }
        app, pipeline = r1.h0.make_pipeline(args.config, reference, rotation, translation)
        records: list[dict[str, Any]] = []
        for index, condition in enumerate(conditions, start=1):
            print(f"[{index}/{len(conditions)}] {condition.condition_id}", flush=True)
            record = run_manual_condition(condition, pipeline, polygon)
            record["transition_rows"] = make_transition_rows(record)
            records.append(record)
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows = [row for record in records for row in record["transition_rows"]]
        candidate_fields = sorted({key for row in candidate_rows for key in row})
        write_csv(output_dir / "manual_search_candidate_audit.csv", candidate_rows, candidate_fields)
        summaries = [make_condition_summary(record) for record in records]
        summary_fields = sorted({key for row in summaries for key in row})
        write_csv(output_dir / "roi_target_selection.csv", summaries, summary_fields)
        thresholds, failed_widths = derive_width_thresholds(records)
        width_rows = make_width_rows(records, thresholds)
        width_fields = sorted({key for row in width_rows for key in row})
        write_csv(output_dir / "width_gate_sensitivity.csv", width_rows, width_fields)
        overlay_paths: list[Path] = []
        for record in records:
            if record["condition"].condition_id not in REQUIRED_OVERLAY_IDS:
                continue
            path = output_dir / f"{record['condition'].condition_id}_before_after_overlay.png"
            if render_before_after(record, polygon, path):
                overlay_paths.append(path)
        classification, detail = classify(records, width_rows, thresholds)
        provenance = {
            "task": "H0-1R4",
            "classification": classification,
            "input_root": str(args.input_dir.resolve()),
            "output_dir": str(output_dir),
            "manual_roi_json": str(roi_path),
            "manual_roi_sha256": sha256_file(roi_path),
            "manual_roi_contract": {
                "coordinate_system": roi_document.get("coordinate_system"),
                "created_mode": roi_document.get("created_mode"),
                "purpose": roi_document.get("purpose"),
                "directory_height_truth_used_for_selection": False,
                "per_condition_final_roi_selected_manually": False,
                "board_polygon_used_as_full_search_hard_mask": False,
            },
            "data_revision_note": {
                "current_h02_p01_is_corrected_recording": True,
                "prior_r3_bad_image_label_reused": False,
                "prior_bad_conditions": sorted(PRIOR_BAD_CONDITIONS),
            },
            "reused_artifacts": {
                "h0_1r_script": str((TOOLS_ROOT / "audit_haikang_roi_v2_0829.py").resolve()),
                "h0_1r3_script": str((TOOLS_ROOT / "audit_haikang_roi_v2_r3_0829.py").resolve()),
                "roi_v2_profile_module": str((TOOLS_ROOT / "auto_roi_v2_session01.py").resolve()),
                "roi_v2_support_module": str((TOOLS_ROOT / "thermal_a2a_roi_v2.py").resolve()),
                "prior_r3_transition_audit": str((args.input_dir / "c0_height_audit" / "roi_v2_r3" / "candidate_transition_audit.csv").resolve()),
                "prior_r3_transition_audit_sha256": sha256_file(args.input_dir / "c0_height_audit" / "roi_v2_r3" / "candidate_transition_audit.csv"),
            },
            "new_audit_operations": [
                "filter full median centerline by one Session-level full_sensor_uv polygon",
                "replay unchanged ROI-V2 profile/derivative/peak/pair/geometry gates",
                "post-hoc width threshold what-if with all other gates and score held fixed",
                "before_after overlays and CSV report",
            ],
            "discovery": discovery,
            "session_ground": json_safe(session_summary),
            "config_contract": json_safe(r1.h0.config_contract(args.config)),
            "width_gate": {
                "current_min_px": current_width_threshold(),
                "failed_widths_observed_in_manual_domain_px": failed_widths,
                "tested_thresholds_px": thresholds,
                "parameters_other_than_width_changed": False,
                "same_data_accuracy_claim": False,
            },
            "classification_detail": detail,
            "outputs": {
                "candidate_csv": str((output_dir / "manual_search_candidate_audit.csv").resolve()),
                "summary_csv": str((output_dir / "roi_target_selection.csv").resolve()),
                "width_csv": str((output_dir / "width_gate_sensitivity.csv").resolve()),
                "overlays": [str(path.resolve()) for path in overlay_paths],
            },
        }
        (output_dir / "manual_search_provenance.json").write_text(
            json.dumps(json_safe(provenance), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = build_report(
            output_dir,
            roi_document,
            discovery,
            config_summary,
            records,
            summaries,
            width_rows,
            thresholds,
            failed_widths,
            classification,
            detail,
            overlay_paths,
        )
        (output_dir / "roi_v2_manual_search_report.md").write_text(report, encoding="utf-8")
        print(json.dumps({"classification": classification, "conditions": len(records), "manual_candidates": sum(record["manual_candidate_count"] for record in records), "thresholds": thresholds, "failed_widths": failed_widths, "output_dir": str(output_dir)}, ensure_ascii=False))
        return 0
    except (AuditError, r1.h0.AuditError, OSError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
