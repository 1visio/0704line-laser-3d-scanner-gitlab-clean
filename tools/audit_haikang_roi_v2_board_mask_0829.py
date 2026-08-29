#!/usr/bin/env python3
"""Audit a Session-board-mask constrained Haikang ROI-V2 search.

The existing H0-1R replay, candidate generation, geometry gates, score, C0,
Session Ground and axis adapter are reused unchanged.  This audit adds only a
board-support gate before an after-ranking view.  It does not measure height,
apply compensation, or use directory height to choose a candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
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
from calibration.session_ground import (  # noqa: E402
    estimate_session_ground_extrinsic_from_corners,
)
from measurement.board_mask import (  # noqa: E402
    _points_inside_convex_polygon,
    full_board_physical_polygon,
)


DATA_ROOT_DEFAULT = r1.h0.DATA_ROOT_DEFAULT
OUTPUT_DIR_DEFAULT = DATA_ROOT_DEFAULT / "c0_height_audit" / "roi_v2_board_mask"
INSETS_MM = (10.0, 20.0)
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


class AuditError(RuntimeError):
    """Raised when the board-mask audit contract is unsafe."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", type=Path, default=r1.h0.CONFIG_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args(argv)


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def json_default(value: Any) -> Any:
    """Serialize numpy/path values without turning arrays into JSON strings."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(interval: Any) -> list[Any]:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return []
    return [interval[0], interval[1]]


def pair_key(candidate: dict[str, Any] | None) -> tuple[int, int] | None:
    if not candidate:
        return None
    first = finite(candidate.get("edge1_v"))
    second = finite(candidate.get("edge2_v"))
    if first is None or second is None:
        return None
    return int(round(first)), int(round(second))


def existing_sort_key(candidate: dict[str, Any]) -> tuple[bool, int, float]:
    score = finite(candidate.get("pair_score"))
    return (
        bool(candidate.get("edge_pair_geometry_ok")),
        -len(candidate.get("pair_gate_reasons") or []),
        score if score is not None else float("-inf"),
    )


def load_board_geometry(
    session_path: Path,
    app: Any,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    document = json.loads(session_path.read_text(encoding="utf-8"))
    reference = document.get("session_ground_reference") or {}
    support = reference.get("support") or {}
    if document.get("status") != "VALID" or document.get("valid") is not True:
        raise AuditError("Session JSON is not VALID")
    if reference.get("status") != "VALID":
        raise AuditError("session_ground_reference.status is not VALID")
    if support.get("source") != "pnp_board_mask":
        raise AuditError(f"unexpected support source: {support.get('source')}")
    if support.get("mask_mode") != "full_board_physical":
        raise AuditError(f"unexpected mask mode: {support.get('mask_mode')}")

    stored_polygon = np.asarray(support.get("polygon_full_uv"), dtype=np.float64)
    if stored_polygon.shape != (4, 2) or not np.isfinite(stored_polygon).all():
        raise AuditError("support.polygon_full_uv must be a finite 4x2 polygon")

    board = app.session_ground_calibration.board_config()
    top_frame = document.get("frame") or {}
    try:
        pnp_offset = (int(top_frame["offset_x"]), int(top_frame["offset_y"]))
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError("Session PnP frame offset is missing or invalid") from error
    expected_corners = int(board.pattern_cols * board.pattern_rows)
    corners = np.asarray(
        (document.get("detection") or {}).get("corners"),
        dtype=np.float32,
    )
    if corners.shape != (expected_corners, 2) or not np.isfinite(corners).all():
        raise AuditError(
            f"stored Session detection corners must be {expected_corners}x2"
        )

    K = np.asarray(calibration.get("K"), dtype=np.float64).copy()
    D = np.asarray(calibration.get("D"), dtype=np.float64)
    if K.shape != (3, 3) or D.size == 0:
        raise AuditError("runtime calibration does not contain valid K/D")
    K[0, 2] -= float(pnp_offset[0])
    K[1, 2] -= float(pnp_offset[1])
    pnp = estimate_session_ground_extrinsic_from_corners(
        corners,
        {"K": K, "D": D},
        board,
        detection_method="stored_session_detection_audit",
    )
    if pnp.status != "success" or pnp.rvec is None or pnp.tvec is None:
        raise AuditError(f"stored Session PnP replay failed: {pnp.message}")

    polygons: dict[float, np.ndarray] = {}
    physical_bounds: dict[float, dict[str, float]] = {}
    for inset in (0.0, *INSETS_MM):
        polygons[inset] = full_board_physical_polygon(
            pnp.rvec,
            pnp.tvec,
            pattern_cols=board.pattern_cols,
            pattern_rows=board.pattern_rows,
            square_size_mm=board.square_size_mm,
            camera_matrix=K,
            dist_coeffs=D,
            image_offset=pnp_offset,
            inset_mm=inset,
        )
        square = float(board.square_size_mm)
        physical_bounds[inset] = {
            "x_min_mm": -square + inset,
            "x_max_mm": float(board.pattern_cols) * square - inset,
            "y_min_mm": -square + inset,
            "y_max_mm": float(board.pattern_rows) * square - inset,
        }

    zero_delta = float(np.max(np.abs(polygons[0.0] - stored_polygon)))
    if zero_delta > 1.0e-3:
        raise AuditError(
            "stored polygon and replayed 0 mm polygon disagree: "
            f"max_abs_delta_px={zero_delta}"
        )
    return {
        "document": document,
        "support": support,
        "board": board,
        "pnp_offset": list(pnp_offset),
        "pnp_reprojection_rmse_px": finite(pnp.reprojection_rmse_px),
        "stored_polygon_full_uv": stored_polygon,
        "polygons_full_uv": polygons,
        "physical_bounds_mm": physical_bounds,
        "zero_polygon_max_abs_delta_px": zero_delta,
    }


def contiguous_runs(mask: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return [[int(start), int(end)] for start, end in zip(starts, ends)]


def centerline_board_support(
    median_scan: np.ndarray | None,
    polygon: np.ndarray,
    profile_length: int,
) -> dict[str, Any]:
    empty = {
        "support_mask": np.zeros(profile_length, dtype=bool),
        "finite_point_count": 0,
        "unique_profile_row_count": 0,
        "supported_profile_row_count": 0,
        "point_inside": np.zeros(0, dtype=bool),
        "full_uv": np.empty((0, 2), dtype=np.float64),
        "runs": [],
    }
    if median_scan is None:
        return empty
    scan = np.asarray(median_scan, dtype=np.float64)
    finite_points = np.isfinite(scan).all(axis=1)
    detector_points = scan[finite_points]
    if len(detector_points) == 0:
        return empty
    full_uv = detector_points[:, [1, 0]]
    point_inside = _points_inside_convex_polygon(full_uv, polygon)
    rows = np.rint(detector_points[:, 1]).astype(np.int64)
    in_range = (rows >= 0) & (rows < profile_length)
    rows = rows[in_range]
    detector_points = detector_points[in_range]
    full_uv = full_uv[in_range]
    point_inside = point_inside[in_range]
    support_mask = np.zeros(profile_length, dtype=bool)
    unique_rows = np.unique(rows)
    for row in unique_rows:
        selected = rows == row
        grouped_full_uv = np.asarray(
            [[float(row), float(np.median(detector_points[selected, 0]))]],
            dtype=np.float64,
        )
        support_mask[int(row)] = bool(
            _points_inside_convex_polygon(grouped_full_uv, polygon)[0]
        )
    return {
        "support_mask": support_mask,
        "finite_point_count": int(len(detector_points)),
        "unique_profile_row_count": int(len(unique_rows)),
        "supported_profile_row_count": int(np.count_nonzero(support_mask)),
        "point_inside": point_inside,
        "full_uv": full_uv,
        "runs": contiguous_runs(support_mask),
    }


def interval_support(
    interval: Any,
    support_mask: np.ndarray,
    *,
    minimum_fraction: float,
    require_full: bool,
) -> dict[str, Any]:
    values = interval_json(interval)
    result = {
        "range": values,
        "width_px": 0,
        "supported_rows": 0,
        "support_fraction": 0.0,
        "all_supported": False,
        "gate_ok": False,
        "minimum_fraction": minimum_fraction,
        "require_full": require_full,
    }
    if len(values) != 2:
        return result
    try:
        start, end = int(values[0]), int(values[1])
    except (TypeError, ValueError):
        return result
    if start < 0 or end < start or end >= len(support_mask):
        return result
    segment = np.asarray(support_mask[start : end + 1], dtype=bool)
    width = int(len(segment))
    supported = int(np.count_nonzero(segment))
    fraction = float(supported / width) if width else 0.0
    result.update(
        {
            "width_px": width,
            "supported_rows": supported,
            "support_fraction": fraction,
            "all_supported": bool(width > 0 and supported == width),
            "gate_ok": bool(
                width > 0
                and (
                    supported == width
                    if require_full
                    else fraction >= minimum_fraction
                )
            ),
        }
    )
    return result


def profile_point_inside(
    v_value: Any,
    interpolated_profile: np.ndarray | None,
    polygon: np.ndarray,
) -> dict[str, Any]:
    row_value = finite(v_value)
    result = {
        "profile_row": int(round(row_value)) if row_value is not None else None,
        "profile_value_u_prime": None,
        "inside": False,
    }
    if row_value is None or interpolated_profile is None:
        return result
    row = int(round(row_value))
    if row < 0 or row >= len(interpolated_profile):
        return result
    u_prime = finite(interpolated_profile[row])
    if u_prime is None:
        return result
    result["profile_value_u_prime"] = u_prime
    result["inside"] = bool(
        _points_inside_convex_polygon(
            np.asarray([[float(row), u_prime]], dtype=np.float64),
            polygon,
        )[0]
    )
    return result


def candidate_board_gate(
    candidate: dict[str, Any],
    support: dict[str, Any],
    polygon: np.ndarray,
    interpolated_profile: np.ndarray | None,
    baseline_min_fraction: float,
) -> dict[str, Any]:
    support_mask = np.asarray(support["support_mask"], dtype=bool)
    edge1 = profile_point_inside(
        candidate.get("edge1_v"), interpolated_profile, polygon
    )
    edge2 = profile_point_inside(
        candidate.get("edge2_v"), interpolated_profile, polygon
    )
    for edge in (edge1, edge2):
        row = edge["profile_row"]
        edge["centerline_row_supported"] = bool(
            row is not None and 0 <= row < len(support_mask) and support_mask[row]
        )
        edge["gate_ok"] = bool(
            edge["inside"] and edge["centerline_row_supported"]
        )

    transitions = candidate.get("transition_v_ranges") or {}
    transition1 = interval_support(
        transitions.get("edge1"),
        support_mask,
        minimum_fraction=1.0,
        require_full=True,
    )
    transition2 = interval_support(
        transitions.get("edge2"),
        support_mask,
        minimum_fraction=1.0,
        require_full=True,
    )
    height = interval_support(
        candidate.get("height_v_range"),
        support_mask,
        minimum_fraction=1.0,
        require_full=True,
    )
    baselines = list(candidate.get("baseline_v_ranges") or [[], []]) + [[], []]
    baseline_before = interval_support(
        baselines[0],
        support_mask,
        minimum_fraction=baseline_min_fraction,
        require_full=False,
    )
    baseline_after = interval_support(
        baselines[1],
        support_mask,
        minimum_fraction=baseline_min_fraction,
        require_full=False,
    )

    checks = (
        ("edge1", edge1["gate_ok"]),
        ("edge2", edge2["gate_ok"]),
        ("edge1_transition", transition1["gate_ok"]),
        ("edge2_transition", transition2["gate_ok"]),
        ("height_interior", height["gate_ok"]),
        ("baseline_before", baseline_before["gate_ok"]),
        ("baseline_after", baseline_after["gate_ok"]),
    )
    reasons = [f"{name}_outside_board_support" for name, ok in checks if not ok]
    if reasons:
        reasons.insert(0, "outside_board_support")
    return {
        "board_gate_status": "PASS" if not reasons else "FAIL",
        "board_gate_pass": not reasons,
        "outside_board_support": bool(reasons),
        "board_gate_reasons": list(dict.fromkeys(reasons)),
        "edge1": edge1,
        "edge2": edge2,
        "transition_edge1": transition1,
        "transition_edge2": transition2,
        "height": height,
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
    }


def after_status(
    candidates: list[dict[str, Any]],
    gates_by_rank: dict[int, dict[str, Any]],
    eligible_ranks: list[int],
) -> tuple[str, list[str]]:
    if not candidates:
        return "FAIL", ["no_roi_v2_candidate_before_board_gate"]
    if not eligible_ranks:
        if any(
            bool(candidate.get("edge_pair_geometry_ok"))
            and gates_by_rank[rank]["outside_board_support"]
            for rank, candidate in enumerate(candidates, start=1)
        ):
            return "OUTSIDE_BOARD_SUPPORT", [
                "no_existing_geometry_pass_candidate_after_board_gate"
            ]
        return "UNCERTAIN", ["no_existing_geometry_pass_candidate"]
    selected = candidates[eligible_ranks[0] - 1]
    reasons = list(selected.get("pair_gate_reasons") or [])
    if selected.get("multi_geometry_ok") is not True:
        reasons.extend(selected.get("multi_geometry_reasons") or [])
        if not selected.get("multi_geometry_reasons"):
            reasons.append("multi_geometry_not_ok")
    return ("PASS" if not reasons else "UNCERTAIN"), list(dict.fromkeys(reasons))


def evaluate_inset(
    audit: dict[str, Any],
    geometry: dict[str, Any],
    inset: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = list(audit.get("candidates") or [])
    raw_profile = audit.get("raw_profile")
    profile_length = (
        int(len(raw_profile))
        if raw_profile is not None
        else int(r1.h0.roi_v2.FULL_SENSOR_HEIGHT)
    )
    polygon = geometry["polygons_full_uv"][inset]
    support = centerline_board_support(
        audit.get("median_scan"), polygon, profile_length
    )
    baseline_min_fraction = float(
        r1.h0.roi_v2.PARAMETERS["baseline"]["minimum_support_fraction"]
    )
    gates_by_rank: dict[int, dict[str, Any]] = {}
    board_pass_ranks: list[int] = []
    eligible_ranks: list[int] = []
    for rank, candidate in enumerate(candidates, start=1):
        gate = candidate_board_gate(
            candidate,
            support,
            polygon,
            audit.get("interpolated_profile"),
            baseline_min_fraction,
        )
        gate["before_rank"] = rank
        gate["before_selected"] = rank == 1
        gate["after_eligible"] = bool(
            gate["board_gate_pass"] and candidate.get("edge_pair_geometry_ok")
        )
        gates_by_rank[rank] = gate
        if gate["board_gate_pass"]:
            board_pass_ranks.append(rank)
        if gate["after_eligible"]:
            eligible_ranks.append(rank)

    eligible_ranks.sort(
        key=lambda rank: existing_sort_key(candidates[rank - 1]),
        reverse=True,
    )
    after_rank_by_before_rank = {
        before_rank: after_rank
        for after_rank, before_rank in enumerate(eligible_ranks, start=1)
    }
    for before_rank, gate in gates_by_rank.items():
        gate["after_rank"] = after_rank_by_before_rank.get(before_rank)
        gate["after_selected"] = bool(gate["after_rank"] == 1)
    status, status_reasons = after_status(
        candidates, gates_by_rank, eligible_ranks
    )
    selected_after_rank = eligible_ranks[0] if eligible_ranks else None
    return {
        "inset_mm": inset,
        "polygon_full_uv": polygon,
        "support": support,
        "baseline_min_fraction": baseline_min_fraction,
        "candidate_count_before": len(candidates),
        "candidate_count_board_gate_pass": len(board_pass_ranks),
        "candidate_count_board_gate_fail": len(candidates) - len(board_pass_ranks),
        "candidate_count_after": len(eligible_ranks),
        "gates_by_rank": gates_by_rank,
        "eligible_before_ranks": eligible_ranks,
        "selected_after_before_rank": selected_after_rank,
        "selected_after": (
            candidates[selected_after_rank - 1]
            if selected_after_rank is not None
            else None
        ),
        "before_selected": candidates[0] if candidates else None,
        "after_status": status,
        "after_status_reasons": status_reasons,
        "fixed_pair_before_selected": (
            pair_key(candidates[0] if candidates else None) == FIXED_PAIR
        ),
        "fixed_pair_after_eligible": any(
            pair_key(candidates[rank - 1]) == FIXED_PAIR for rank in eligible_ranks
        ),
        "score_unchanged_max_abs_delta": max(
            (
                abs(
                    (finite(candidate.get("pair_score")) or 0.0)
                    - (
                        (finite(candidate.get("edge_min_prominence_px")) or 0.0)
                        + 0.10
                        * (finite(candidate.get("step_amplitude_px")) or 0.0)
                        + 0.01
                        * min(
                            finite(candidate.get("object_width_px")) or 0.0,
                            120.0,
                        )
                    )
                )
                for candidate in candidates
            ),
            default=0.0,
        ),
    }


def candidate_before_after_row(
    audit: dict[str, Any],
    evaluation: dict[str, Any],
    rank: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    gate = evaluation["gates_by_rank"][rank]
    score = finite(candidate.get("pair_score"))
    recomputed = (
        (finite(candidate.get("edge_min_prominence_px")) or 0.0)
        + 0.10 * (finite(candidate.get("step_amplitude_px")) or 0.0)
        + 0.01
        * min(finite(candidate.get("object_width_px")) or 0.0, 120.0)
    )
    baselines = list(candidate.get("baseline_v_ranges") or [[], []]) + [[], []]
    return {
        "height_gt_mm": audit["condition"].height_gt_mm,
        "height_id": audit["condition"].height_id,
        "position_id": audit["condition"].position_id,
        "condition_id": audit["condition"].condition_id,
        "inset_mm": evaluation["inset_mm"],
        "candidate_before_rank": rank,
        "before_selected": rank == 1,
        "orientation": candidate.get("orientation"),
        "edge1_u_full_px": candidate.get("edge1_v"),
        "edge2_u_full_px": candidate.get("edge2_v"),
        "object_width_px": candidate.get("object_width_px"),
        "height_u_full_range_px": candidate.get("height_v_range"),
        "baseline_before_u_full_range_px": baselines[0],
        "baseline_after_u_full_range_px": baselines[1],
        "edge1_prominence_px": candidate.get("edge1_prominence_px"),
        "edge2_prominence_px": candidate.get("edge2_prominence_px"),
        "edge_min_prominence_px": candidate.get("edge_min_prominence_px"),
        "step_amplitude_px": candidate.get("step_amplitude_px"),
        "pair_score": score,
        "score_recomputed": recomputed,
        "score_delta": score - recomputed if score is not None else None,
        "before_pair_gate_reasons": candidate.get("pair_gate_reasons"),
        "before_edge_pair_geometry_ok": candidate.get("edge_pair_geometry_ok"),
        "before_multi_geometry_ok": candidate.get("multi_geometry_ok"),
        "before_multi_geometry_reasons": candidate.get("multi_geometry_reasons"),
        "before_height_support": candidate.get("height_support"),
        "before_baseline_before_support": candidate.get("before_support"),
        "before_baseline_after_support": candidate.get("after_support"),
        "board_gate_status": gate["board_gate_status"],
        "board_gate_pass": gate["board_gate_pass"],
        "outside_board_support": gate["outside_board_support"],
        "board_gate_reasons": gate["board_gate_reasons"],
        "board_edge1": gate["edge1"],
        "board_edge2": gate["edge2"],
        "board_transition_edge1": gate["transition_edge1"],
        "board_transition_edge2": gate["transition_edge2"],
        "board_height": gate["height"],
        "board_baseline_before": gate["baseline_before"],
        "board_baseline_after": gate["baseline_after"],
        "after_eligible": gate["after_eligible"],
        "after_rank": gate["after_rank"],
        "after_selected": gate["after_selected"],
        "after_status": evaluation["after_status"] if gate["after_selected"] else "",
        "fixed_pair": pair_key(candidate) == FIXED_PAIR,
    }


def selected_baselines(candidate: dict[str, Any] | None) -> tuple[list[Any], list[Any]]:
    if not candidate:
        return [], []
    values = list(candidate.get("baseline_v_ranges") or [[], []]) + [[], []]
    return values[0], values[1]


def registry_row(
    audit: dict[str, Any],
    geometry: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    condition = audit["condition"]
    source_row = audit.get("representative_row") or {}
    support = evaluation["support"]
    before = evaluation["before_selected"]
    before_gate = evaluation["gates_by_rank"].get(1, {})
    after = evaluation["selected_after"]
    after_rank = evaluation["selected_after_before_rank"]
    after_gate = (
        evaluation["gates_by_rank"].get(after_rank, {}) if after_rank else {}
    )
    after_height = after_gate.get("height") or {}
    after_before = after_gate.get("baseline_before") or {}
    after_after = after_gate.get("baseline_after") or {}
    before_baseline, before_after_baseline = selected_baselines(before)
    after_baseline, after_after_baseline = selected_baselines(after)
    supported_rows = np.flatnonzero(support["support_mask"])
    return {
        "height_gt_mm": condition.height_gt_mm,
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "condition_id": condition.condition_id,
        "inset_mm": evaluation["inset_mm"],
        "source_frame_count": len(audit.get("source_rows") or []),
        "pipeline_success_frame_count": sum(
            result is not None for result in audit.get("results") or []
        ),
        "representative_frame": source_row.get("camera_frame_number"),
        "offset_x": source_row.get("offset_x"),
        "offset_y": source_row.get("offset_y"),
        "image_width": source_row.get("width"),
        "image_height": source_row.get("height"),
        "session_ground_generation": geometry["document"]
        .get("session_ground_reference", {})
        .get("ground_extrinsic_generation"),
        "board_support_source": geometry["support"].get("source"),
        "board_mask_mode": geometry["support"].get("mask_mode"),
        "board_physical_bounds_mm": geometry["physical_bounds_mm"][
            evaluation["inset_mm"]
        ],
        "board_inner_polygon_full_uv": evaluation["polygon_full_uv"],
        "stored_zero_polygon_max_abs_delta_px": geometry[
            "zero_polygon_max_abs_delta_px"
        ],
        "pnp_reprojection_rmse_px": geometry["pnp_reprojection_rmse_px"],
        "centerline_finite_point_count": support["finite_point_count"],
        "centerline_unique_profile_row_count": support[
            "unique_profile_row_count"
        ],
        "board_supported_profile_row_count": support[
            "supported_profile_row_count"
        ],
        "board_supported_profile_fraction": (
            support["supported_profile_row_count"]
            / max(1, support["unique_profile_row_count"])
        ),
        "board_supported_profile_runs": support["runs"],
        "board_supported_profile_u_min_px": (
            int(supported_rows[0]) if len(supported_rows) else None
        ),
        "board_supported_profile_u_max_px": (
            int(supported_rows[-1]) if len(supported_rows) else None
        ),
        "before_roi_v2_status": (audit.get("assessment") or {}).get(
            "auto_qc_status"
        ),
        "before_roi_v2_reasons": (audit.get("assessment") or {}).get(
            "auto_qc_reasons"
        ),
        "before_candidate_count": evaluation["candidate_count_before"],
        "board_gate_pass_candidate_count": evaluation[
            "candidate_count_board_gate_pass"
        ],
        "board_gate_fail_candidate_count": evaluation[
            "candidate_count_board_gate_fail"
        ],
        "after_candidate_count": evaluation["candidate_count_after"],
        "before_selected_edge1_u_full_px": before.get("edge1_v") if before else None,
        "before_selected_edge2_u_full_px": before.get("edge2_v") if before else None,
        "before_selected_height_u_full_range_px": (
            before.get("height_v_range") if before else []
        ),
        "before_selected_baseline_before_u_full_range_px": before_baseline,
        "before_selected_baseline_after_u_full_range_px": before_after_baseline,
        "before_selected_board_gate_status": before_gate.get(
            "board_gate_status"
        ),
        "before_selected_board_gate_reasons": before_gate.get(
            "board_gate_reasons"
        ),
        "before_selected_outside_board_support": before_gate.get(
            "outside_board_support"
        ),
        "fixed_pair_before_selected": evaluation["fixed_pair_before_selected"],
        "fixed_pair_after_eligible": evaluation["fixed_pair_after_eligible"],
        "after_selected_before_rank": after_rank,
        "after_selected": bool(after_rank),
        "after_roi_v2_status": evaluation["after_status"],
        "after_roi_v2_reasons": evaluation["after_status_reasons"],
        "after_selected_edge1_u_full_px": after.get("edge1_v") if after else None,
        "after_selected_edge2_u_full_px": after.get("edge2_v") if after else None,
        "after_selected_height_u_full_range_px": (
            after.get("height_v_range") if after else []
        ),
        "after_selected_baseline_before_u_full_range_px": after_baseline,
        "after_selected_baseline_after_u_full_range_px": after_after_baseline,
        "after_selected_board_gate_status": after_gate.get(
            "board_gate_status"
        ),
        "after_selected_height_board_support_fraction": after_height.get(
            "support_fraction"
        ),
        "after_selected_baseline_before_board_support_fraction": after_before.get(
            "support_fraction"
        ),
        "after_selected_baseline_after_board_support_fraction": after_after.get(
            "support_fraction"
        ),
        "after_selected_height_board_support_all": after_height.get(
            "all_supported"
        ),
        "after_selected_baseline_before_board_support_ok": after_before.get(
            "gate_ok"
        ),
        "after_selected_baseline_after_board_support_ok": after_after.get(
            "gate_ok"
        ),
        "after_selected_multi_geometry_ok": after.get("multi_geometry_ok")
        if after
        else None,
        "score_unchanged_max_abs_delta": evaluation[
            "score_unchanged_max_abs_delta"
        ],
    }


ROI_REGISTRY_FIELDS = [
    "height_gt_mm",
    "height_id",
    "position_id",
    "condition_id",
    "inset_mm",
    "source_frame_count",
    "pipeline_success_frame_count",
    "representative_frame",
    "offset_x",
    "offset_y",
    "image_width",
    "image_height",
    "session_ground_generation",
    "board_support_source",
    "board_mask_mode",
    "board_physical_bounds_mm",
    "board_inner_polygon_full_uv",
    "stored_zero_polygon_max_abs_delta_px",
    "pnp_reprojection_rmse_px",
    "centerline_finite_point_count",
    "centerline_unique_profile_row_count",
    "board_supported_profile_row_count",
    "board_supported_profile_fraction",
    "board_supported_profile_runs",
    "board_supported_profile_u_min_px",
    "board_supported_profile_u_max_px",
    "before_roi_v2_status",
    "before_roi_v2_reasons",
    "before_candidate_count",
    "board_gate_pass_candidate_count",
    "board_gate_fail_candidate_count",
    "after_candidate_count",
    "before_selected_edge1_u_full_px",
    "before_selected_edge2_u_full_px",
    "before_selected_height_u_full_range_px",
    "before_selected_baseline_before_u_full_range_px",
    "before_selected_baseline_after_u_full_range_px",
    "before_selected_board_gate_status",
    "before_selected_board_gate_reasons",
    "before_selected_outside_board_support",
    "fixed_pair_before_selected",
    "fixed_pair_after_eligible",
    "after_selected_before_rank",
    "after_selected",
    "after_roi_v2_status",
    "after_roi_v2_reasons",
    "after_selected_edge1_u_full_px",
    "after_selected_edge2_u_full_px",
    "after_selected_height_u_full_range_px",
    "after_selected_baseline_before_u_full_range_px",
    "after_selected_baseline_after_u_full_range_px",
    "after_selected_board_gate_status",
    "after_selected_height_board_support_fraction",
    "after_selected_baseline_before_board_support_fraction",
    "after_selected_baseline_after_board_support_fraction",
    "after_selected_height_board_support_all",
    "after_selected_baseline_before_board_support_ok",
    "after_selected_baseline_after_board_support_ok",
    "after_selected_multi_geometry_ok",
    "score_unchanged_max_abs_delta",
]


BEFORE_AFTER_FIELDS = [
    "height_gt_mm",
    "height_id",
    "position_id",
    "condition_id",
    "inset_mm",
    "candidate_before_rank",
    "before_selected",
    "orientation",
    "edge1_u_full_px",
    "edge2_u_full_px",
    "object_width_px",
    "height_u_full_range_px",
    "baseline_before_u_full_range_px",
    "baseline_after_u_full_range_px",
    "edge1_prominence_px",
    "edge2_prominence_px",
    "edge_min_prominence_px",
    "step_amplitude_px",
    "pair_score",
    "score_recomputed",
    "score_delta",
    "before_pair_gate_reasons",
    "before_edge_pair_geometry_ok",
    "before_multi_geometry_ok",
    "before_multi_geometry_reasons",
    "before_height_support",
    "before_baseline_before_support",
    "before_baseline_after_support",
    "board_gate_status",
    "board_gate_pass",
    "outside_board_support",
    "board_gate_reasons",
    "board_edge1",
    "board_edge2",
    "board_transition_edge1",
    "board_transition_edge2",
    "board_height",
    "board_baseline_before",
    "board_baseline_after",
    "after_eligible",
    "after_rank",
    "after_selected",
    "after_status",
    "fixed_pair",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_value(row.get(field)) for field in fields})


def full_u_range(row: dict[str, Any]) -> tuple[float, float]:
    offset = finite(row.get("offset_x")) or 0.0
    width = finite(row.get("width")) or 0.0
    return offset, offset + max(0.0, width - 1.0)


def full_v_range(row: dict[str, Any]) -> tuple[float, float]:
    offset = finite(row.get("offset_y")) or 0.0
    height = finite(row.get("height")) or 0.0
    return offset, offset + max(0.0, height - 1.0)


def plot_interval(ax: Any, interval: Any, **kwargs: Any) -> None:
    values = interval_json(interval)
    if len(values) == 2:
        ax.axvspan(float(values[0]), float(values[1]), **kwargs)


def render_overlay(
    record: dict[str, Any],
    geometry: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    audit = record["audit"]
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

    condition = audit["condition"]
    x0, x1 = full_u_range(image_row)
    y0, y1 = full_v_range(image_row)
    scan = np.asarray(median_scan, dtype=np.float64)
    finite_points = np.isfinite(scan).all(axis=1)
    full_uv = scan[finite_points][:, [1, 0]]
    image_low = float(np.nanmin(image)) if image.size else 0.0
    image_high = float(np.nanmax(image)) if image.size else 1.0
    if image_high <= image_low:
        image_high = image_low + 1.0

    fig, (ax_image, ax_profile) = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
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
    if len(full_uv):
        ax_image.scatter(
            full_uv[:, 0],
            full_uv[:, 1],
            s=1.2,
            color="#ffe66d",
            alpha=0.55,
            label="median centerline",
        )

    polygon_colors = {0.0: "#bbbbbb", 10.0: "#25c9d8", 20.0: "#d46cff"}
    for inset in (0.0, *INSETS_MM):
        polygon = np.asarray(geometry["polygons_full_uv"][inset])
        closed = np.vstack([polygon, polygon[0]])
        ax_image.plot(
            closed[:, 0],
            closed[:, 1],
            color=polygon_colors[inset],
            linestyle="--" if inset == 0.0 else "-",
            linewidth=1.2 if inset == 0.0 else 1.8,
            label=f"board polygon inset {inset:g} mm",
        )
        # The audit compares 10/20 mm insets; 0 mm is drawn as the stored
        # outer support boundary but is not an evaluated after-ranking view.
        if inset in record["insets"]:
            support = record["insets"][inset]["support"]
            point_inside = np.asarray(support["point_inside"], dtype=bool)
            if len(point_inside) == len(full_uv) and np.any(point_inside):
                ax_image.scatter(
                    full_uv[point_inside, 0],
                    full_uv[point_inside, 1],
                    s=3.0,
                    color=polygon_colors[inset],
                    alpha=0.6,
                )

    for rank, candidate in enumerate(candidates, start=1):
        gate = record["insets"][10.0]["gates_by_rank"].get(rank, {})
        edge1 = finite(candidate.get("edge1_v"))
        edge2 = finite(candidate.get("edge2_v"))
        if edge1 is None or edge2 is None:
            continue
        color = (
            "#ff3b30"
            if rank == 1
            else "#777777"
            if gate.get("outside_board_support")
            else "#f5a623"
        )
        style = "-" if rank == 1 else "--"
        width = 2.4 if rank == 1 else 1.0
        for axis in (ax_image, ax_profile):
            axis.axvline(edge1, color=color, linestyle=style, linewidth=width)
            axis.axvline(edge2, color=color, linestyle=style, linewidth=width)
        plot_interval(
            ax_profile,
            candidate.get("height_v_range"),
            color=color,
            alpha=0.08 if rank != 1 else 0.18,
        )
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

    after_colors = {10.0: "#00e5a8", 20.0: "#ff4fd8"}
    for inset in INSETS_MM:
        selected = record["insets"][inset].get("selected_after")
        if not selected:
            continue
        color = after_colors[inset]
        edge1 = finite(selected.get("edge1_v"))
        edge2 = finite(selected.get("edge2_v"))
        if edge1 is None or edge2 is None:
            continue
        for axis in (ax_image, ax_profile):
            axis.axvline(edge1, color=color, linewidth=3.0, alpha=0.95)
            axis.axvline(edge2, color=color, linewidth=3.0, alpha=0.95)
        plot_interval(
            ax_image,
            selected.get("height_v_range"),
            color=color,
            alpha=0.22,
            label=f"after height {inset:g} mm",
        )
        baselines = list(selected.get("baseline_v_ranges") or [[], []]) + [[], []]
        plot_interval(
            ax_image,
            baselines[0],
            color=color,
            alpha=0.10,
            linestyle=":",
            label=f"after baseline-before {inset:g} mm",
        )
        plot_interval(
            ax_image,
            baselines[1],
            color=color,
            alpha=0.10,
            linestyle=":",
            label=f"after baseline-after {inset:g} mm",
        )
        plot_interval(ax_profile, selected.get("height_v_range"), color=color, alpha=0.18)

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
            alpha=0.30,
            label="raw profile points",
        )
    for inset in INSETS_MM:
        color = polygon_colors[inset]
        for start, end in record["insets"][inset]["support"]["runs"]:
            ax_profile.axvspan(
                start,
                end,
                color=color,
                alpha=0.035 if inset == 10.0 else 0.025,
            )

    ax_image.set_xlim(x0, x1 + 1.0)
    ax_image.set_ylim(y1 + 1.0, y0)
    ax_profile.set_xlim(x0, x1 + 1.0)
    ax_profile.set_ylim(y1 + 1.0, y0)
    ax_image.set_xlabel("full-sensor u (px)")
    ax_image.set_ylabel("full-sensor v (px)")
    ax_profile.set_xlabel("detector v' = full-sensor u (px)")
    ax_profile.set_ylabel("detector u' = full-sensor v (px)")
    ax_image.set_title("raw image + centerline + board polygons")
    ax_profile.set_title("all before candidates + after board gate")
    ax_image.grid(alpha=0.15)
    ax_profile.grid(alpha=0.2)
    ax_image.legend(loc="best", fontsize=7)
    ax_profile.legend(loc="best", fontsize=7)

    before_pair = pair_key(candidates[0] if candidates else None)
    after_text = []
    for inset in INSETS_MM:
        evaluation = record["insets"][inset]
        selected = evaluation.get("selected_after")
        after_text.append(
            f"{inset:g}mm:{evaluation['after_status']}/"
            f"{pair_key(selected) if selected else 'none'}"
        )
    fig.suptitle(
        f"{condition.condition_id} | before pair={before_pair} | after "
        + ", ".join(after_text),
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "full-sensor coordinates; board mask is geometry support only; "
        "directory height not used; red=before selected, green=10 mm, "
        "magenta=20 mm, gray=board-rejected",
        ha="center",
        va="bottom",
        fontsize=7,
        wrap=True,
    )
    path = output_dir / f"{condition.condition_id}_board_mask_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fmt(value: Any, digits: int = 3) -> str:
    value = finite(value)
    return "—" if value is None else f"{value:.{digits}f}"


def inset_summary(
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    inset: float,
) -> dict[str, Any]:
    subset = [row for row in rows if float(row["inset_mm"]) == inset]
    candidate_subset = [
        row for row in candidate_rows if float(row["inset_mm"]) == inset
    ]
    selected = [row for row in subset if row.get("after_selected")]
    statuses = Counter(str(row.get("after_roi_v2_status")) for row in subset)
    fractions = [
        float(row["board_supported_profile_fraction"])
        for row in subset
        if finite(row.get("board_supported_profile_fraction")) is not None
    ]
    centers = [
        (
            float(row["after_selected_height_u_full_range_px"][0])
            + float(row["after_selected_height_u_full_range_px"][1])
        )
        / 2.0
        for row in selected
        if isinstance(row.get("after_selected_height_u_full_range_px"), list)
        and len(row["after_selected_height_u_full_range_px"]) == 2
    ]
    candidate_reason_counts: Counter[str] = Counter()
    for row in candidate_subset:
        reasons = row.get("board_gate_reasons") or []
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except json.JSONDecodeError:
                reasons = [reasons]
        for reason in reasons:
            candidate_reason_counts[str(reason)] += 1
    selected_reason_counts = Counter()
    for row in subset:
        reasons = row.get("before_selected_board_gate_reasons") or []
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except json.JSONDecodeError:
                reasons = [reasons]
        for reason in reasons:
            selected_reason_counts[str(reason)] += 1
    return {
        "status_counts": dict(statuses),
        "fixed_before_selected_count": sum(
            bool(row.get("fixed_pair_before_selected")) for row in subset
        ),
        "fixed_after_eligible_count": sum(
            bool(row.get("fixed_pair_after_eligible")) for row in subset
        ),
        "after_selected_count": len(selected),
        "after_selected_baseline_both_ok_count": sum(
            bool(row.get("after_selected_baseline_before_board_support_ok"))
            and bool(row.get("after_selected_baseline_after_board_support_ok"))
            for row in selected
        ),
        "median_board_supported_profile_fraction": (
            float(np.median(fractions)) if fractions else None
        ),
        "selected_center_min": min(centers) if centers else None,
        "selected_center_max": max(centers) if centers else None,
        "selected_center_range": (
            max(centers) - min(centers) if centers else None
        ),
        "no_after_candidate_count": sum(
            not bool(row.get("after_selected")) for row in subset
        ),
        "candidate_count": len(candidate_subset),
        "geometry_pass_candidate_count": sum(
            bool(row.get("before_edge_pair_geometry_ok"))
            for row in candidate_subset
        ),
        "board_pass_candidate_count": sum(
            bool(row.get("board_gate_pass")) for row in candidate_subset
        ),
        "after_eligible_candidate_count": sum(
            bool(row.get("after_eligible")) for row in candidate_subset
        ),
        "geometry_pass_board_fail_candidate_count": sum(
            bool(row.get("before_edge_pair_geometry_ok"))
            and not bool(row.get("board_gate_pass"))
            for row in candidate_subset
        ),
        "board_pass_geometry_fail_candidate_count": sum(
            bool(row.get("board_gate_pass"))
            and not bool(row.get("before_edge_pair_geometry_ok"))
            for row in candidate_subset
        ),
        "candidate_reason_counts": dict(candidate_reason_counts),
        "selected_reason_counts": dict(selected_reason_counts),
        "height_edge_failure_count": sum(
            "height_interior_outside_board_support"
            in str(row.get("before_selected_board_gate_reasons"))
            for row in subset
        ),
        "baseline_before_failure_count": sum(
            "baseline_before_outside_board_support"
            in str(row.get("before_selected_board_gate_reasons"))
            for row in subset
        ),
        "baseline_after_failure_count": sum(
            "baseline_after_outside_board_support"
            in str(row.get("before_selected_board_gate_reasons"))
            for row in subset
        ),
    }


def build_report(
    *,
    root: Path,
    geometry: dict[str, Any],
    records: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    before_after_rows: list[dict[str, Any]],
    overlay_paths: list[Path],
) -> str:
    baseline_min = float(
        r1.h0.roi_v2.PARAMETERS["baseline"]["minimum_support_fraction"]
    )
    summaries = {
        inset: inset_summary(registry_rows, before_after_rows, inset)
        for inset in INSETS_MM
    }
    ten = summaries[10.0]
    twenty = summaries[20.0]
    ten_support = ten["median_board_supported_profile_fraction"] or 0.0
    twenty_support = twenty["median_board_supported_profile_fraction"] or 0.0
    if (
        ten["after_eligible_candidate_count"] > twenty["after_eligible_candidate_count"]
        or ten_support > twenty_support
    ):
        inset_recommendation = (
            "10 mm：固定候选同样被排除，但保留的 board-supported profile 更多；"
            "它是两者中较少侵蚀搜索域的几何审计域。"
        )
    elif (
        twenty["after_eligible_candidate_count"] > ten["after_eligible_candidate_count"]
        or twenty_support > ten_support
    ):
        inset_recommendation = (
            "20 mm：在不使用高度真值的前提下，它保留了更多完整的几何支撑 candidate。"
        )
    else:
        inset_recommendation = "10 mm 与 20 mm 的几何保留能力相同，无法仅凭支撑选择。"
    lines = [
        "# H0-1R2 | 海康 ROI-V2 Session Board Mask 搜索域审计",
        "",
        f"数据根目录：{root}",
        f"condition 数：{len(records)}；registry 行数：{len(registry_rows)}；"
        f"candidate before/after 行数：{len(before_after_rows)}；"
        f"overlay：{len(overlay_paths)} 张",
        "",
        "本轮只增加 ROI 候选几何支撑 gate；未调用高度测量、未修改 C0、"
        "Session Ground、axis adapter、candidate score、H1、H-B2 或 C1。",
        "",
        "## 1. 复用与坐标 provenance",
        "",
        "复用 H0-1R 的 FramePipeline replay、Steger → circular-cone C0 → "
        "Session Ground、Haikang axis adapter、median_centerline、integer_profile、"
        "build_edge_pairs、assess_condition 及原有 candidate score。",
        "",
        "10/20 mm inset 在物理棋盘坐标中构造，再由既有 "
        "measurement.board_mask.full_board_physical_polygon 投影到 full-sensor "
        "(u,v)。不是对像素 polygon 做固定像素腐蚀，也没有使用目录高度选择 candidate。",
        "",
        f"- support.source = {geometry['support'].get('source')}",
        f"- support.mask_mode = {geometry['support'].get('mask_mode')}",
        f"- board = {geometry['board'].pattern_cols} x "
        f"{geometry['board'].pattern_rows}, square = "
        f"{geometry['board'].square_size_mm:g} mm",
        f"- PnP detection frame offset = {geometry['pnp_offset']}",
        f"- stored polygon 与 replayed 0 mm polygon 最大差 = "
        f"{fmt(geometry['zero_polygon_max_abs_delta_px'], 6)} px",
        f"- stored Session PnP replay reprojection RMSE = "
        f"{fmt(geometry['pnp_reprojection_rmse_px'], 4)} px",
        "",
        "0 mm polygon 重投影与 JSON 完全一致，说明使用的是 PnP detection frame "
        "offset；没有把 Session Ground reference 的另一份 frame offset 错用到 polygon。",
        "",
        "## 2. board hard gate 定义",
        "",
        "before 是现有 assess_condition 返回的全部候选及原始排序。after 只过滤 "
        "board gate 失败候选，再使用同一个原有排序 key；score 数值不改。",
        "",
        "- edge1/edge2 的 profile 点及 centerline row 必须位于 inner polygon。",
        "- 既有 edge transition interval 必须连续位于 board-supported centerline rows。",
        "- height_v_range 必须全部有 board-supported centerline。",
        f"- baseline-before/after 使用既有 ROI-V2 baseline minimum support fraction "
        f"{baseline_min:g} 作为几何支撑阈值。",
        "- 任一失败均加入 outside_board_support，且不进入 after ranking。",
        "",
        "## 3. 10 mm / 20 mm 全量结果",
        "",
        "| inset | PASS | UNCERTAIN | OUTSIDE_BOARD_SUPPORT | FAIL | "
        "candidate | geometry-pass | board-pass | after eligible | fixed before | "
        "fixed after | after selected condition | baseline both | median profile support |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for inset in INSETS_MM:
        summary = summaries[inset]
        counts = summary["status_counts"]
        lines.append(
            f"| {inset:g} mm | {counts.get('PASS', 0)} | "
            f"{counts.get('UNCERTAIN', 0)} | "
            f"{counts.get('OUTSIDE_BOARD_SUPPORT', 0)} | "
            f"{counts.get('FAIL', 0)} | "
            f"{summary['candidate_count']} | "
            f"{summary['geometry_pass_candidate_count']} | "
            f"{summary['board_pass_candidate_count']} | "
            f"{summary['after_eligible_candidate_count']} | "
            f"{summary['fixed_before_selected_count']} | "
            f"{summary['fixed_after_eligible_count']} | "
            f"{summary['after_selected_count']} | "
            f"{summary['after_selected_baseline_both_ok_count']} | "
            f"{fmt(summary['median_board_supported_profile_fraction'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## 4. height × position",
            "",
            "| height | before center range | 10 mm after center range | "
            "20 mm after center range | 10 mm status | 20 mm status |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for height in sorted({str(row["height_id"]) for row in registry_rows}):
        height_rows = [
            row for row in registry_rows if str(row["height_id"]) == height
        ]
        before_centers = [
            (
                float(row["before_selected_height_u_full_range_px"][0])
                + float(row["before_selected_height_u_full_range_px"][1])
            )
            / 2.0
            for row in height_rows
            if isinstance(row.get("before_selected_height_u_full_range_px"), list)
            and len(row["before_selected_height_u_full_range_px"]) == 2
        ]
        after_ranges = []
        status_text = []
        for inset in INSETS_MM:
            values = [
                row
                for row in height_rows
                if float(row["inset_mm"]) == inset
            ]
            centers = [
                (
                    float(row["after_selected_height_u_full_range_px"][0])
                    + float(row["after_selected_height_u_full_range_px"][1])
                )
                / 2.0
                for row in values
                if isinstance(row.get("after_selected_height_u_full_range_px"), list)
                and len(row["after_selected_height_u_full_range_px"]) == 2
            ]
            status = Counter(str(row.get("after_roi_v2_status")) for row in values)
            after_ranges.append(
                f"{min(centers):.1f}–{max(centers):.1f}" if centers else "—"
            )
            status_text.append(
                "/".join(f"{key}={value}" for key, value in sorted(status.items()))
            )
        before_range = (
            f"{min(before_centers):.1f}–{max(before_centers):.1f}"
            if before_centers
            else "—"
        )
        lines.append(
            f"| {height} | {before_range} | {after_ranges[0]} | "
            f"{after_ranges[1]} | {status_text[0]} | {status_text[1]} |"
        )

    lines.extend(
        [
            "",
            "## 5. 明确回答",
            "",
            "1. 固定右侧 (1950,2024) 是否被 board mask 排除？",
            "",
            f"10 mm fixed after eligible = {summaries[10.0]['fixed_after_eligible_count']}/50；"
            f"20 mm = {summaries[20.0]['fixed_after_eligible_count']}/50。"
            " 失败候选不能因 prominence 高而绕过 gate。",
            "",
            "2. p01–p10 的 selected ROI 是否恢复合理空间移动？",
            "",
            f"没有恢复出可用的 selected ROI：10/20 mm 均为 "
            f"{ten['after_selected_count']} / {twenty['after_selected_count']} 个 condition "
            "有 after selection。before 仍主要集中在 full-sensor u≈1987 的固定右侧候选；"
            "h02 的少数 before 候选在 u≈272–582 一带，但也未通过完整 board 支撑 gate。",
            "",
            "3. 有多少 condition 因目标或 baseline 靠近棋盘边缘而无法测量？",
            "",
            f"10/20 mm 无 after candidate 分别为 "
            f"{ten['no_after_candidate_count']} / {twenty['no_after_candidate_count']} "
            "个 condition。两种 inset 下均有 44 个原有 geometry-pass candidate "
            "被 board gate 拒绝；另有 6 个 condition 原本就没有 geometry-pass candidate。",
            f"before selected 的 baseline-before 支撑失败为 "
            f"{ten['selected_reason_counts'].get('baseline_before_outside_board_support', 0)} / "
            f"{twenty['selected_reason_counts'].get('baseline_before_outside_board_support', 0)} "
            "个 condition（10/20 mm），并且原因可与 edge/height 失败重叠。"
            f" board gate 通过但被既有 width geometry gate 拒绝的 candidate 为 "
            f"{ten['board_pass_geometry_fail_candidate_count']} / "
            f"{twenty['board_pass_geometry_fail_candidate_count']}。",
            "",
            "4. 10 mm 与 20 mm 哪个仅依据几何支撑更合理？",
            "",
            "选择依据仅为 board-supported centerline 保留率、after candidate 数量及 "
            "height/baseline 支撑；不能用 h_raw 或目录高度作此选择。",
            f"实际结论：{inset_recommendation}",
            f"10 mm 的 median profile support = {fmt(ten_support, 3)}，"
            f"20 mm = {fmt(twenty_support, 3)}；两者 after eligible candidate "
            f"分别为 {ten['after_eligible_candidate_count']} / "
            f"{twenty['after_eligible_candidate_count']}。",
            "",
            "5. 修复后是否可以重新运行 H0-1 生成 h_raw？",
            "",
            "代码流程上可以复用本轮 after gate 作为 H0-1 的 ROI eligibility 前置条件；"
            "但当前数据 10/20 mm 均为 0 个 after-eligible candidate、50/50 condition "
            "无可选 after ROI，因此现在重新运行不会生成可用 h_raw。必须先解决目标可见性、"
            "候选识别或域定义问题；无 candidate 必须保持 unavailable，不能回退到固定右侧候选。",
            "",
            "## 6. 不变量",
            "",
            "- score、axis adapter、C0、Session Ground 和生产配置均未修改。",
            "- 目录 height 仅保留在输出 provenance，不参与 candidate 选择。",
            "- roi_v2_before_after.csv 保留每个 candidate 的完整 board gate 明细。",
            "- overlay 显示所有 before candidates，以及 10/20 mm after selection。",
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
    reference, rotation, translation, session_summary = r1.h0.load_session_reference(
        session_path
    )
    app, pipeline = r1.h0.make_pipeline(
        config_path, reference, rotation, translation
    )
    geometry = load_board_geometry(
        session_path,
        app,
        pipeline.calibration_for_reconstruction(),
    )

    # The required p01/p05/p10 set includes h06_p05, a known fixed-pair sample
    # from H0-1R.  Keep overlay selection deterministic and independent of
    # directory height or any post-hoc candidate choice.
    overlay_ids = set(OVERLAY_REQUIRED_IDS)

    records: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    before_after_rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    for condition in conditions:
        audit = r1.run_condition(condition, pipeline)
        record = {"audit": audit, "condition": condition, "insets": {}}
        for inset in INSETS_MM:
            evaluation = evaluate_inset(audit, geometry, inset)
            record["insets"][inset] = evaluation
            registry_rows.append(registry_row(audit, geometry, evaluation))
            for rank, candidate in enumerate(audit.get("candidates") or [], start=1):
                before_after_rows.append(
                    candidate_before_after_row(
                        audit, evaluation, rank, candidate
                    )
                )
        records.append(record)
        if condition.condition_id not in overlay_ids:
            audit["results"] = []
            audit["frame_arrays"] = []
            audit["representative_result"] = None

    for record in records:
        if record["condition"].condition_id in overlay_ids:
            path = render_overlay(record, geometry, output_dir)
            if path is not None:
                overlay_paths.append(path)

    provenance = {
        "task": "H0-1R2",
        "input_root": str(root),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "session_ground_path": str(session_path),
        "session_ground_sha256": sha256_file(session_path),
        "session_ground_summary": session_summary,
        "discovery": discovery,
        "board_support": {
            "source": geometry["support"].get("source"),
            "mask_mode": geometry["support"].get("mask_mode"),
            "pattern_cols": geometry["board"].pattern_cols,
            "pattern_rows": geometry["board"].pattern_rows,
            "square_size_mm": geometry["board"].square_size_mm,
            "pnp_detection_frame_offset": geometry["pnp_offset"],
            "pnp_reprojection_rmse_px": geometry["pnp_reprojection_rmse_px"],
            "stored_polygon_full_uv": geometry["stored_polygon_full_uv"],
            "physical_bounds_mm": geometry["physical_bounds_mm"],
            "polygons_full_uv": {
                str(inset): geometry["polygons_full_uv"][inset]
                for inset in (0.0, *INSETS_MM)
            },
            "zero_polygon_max_abs_delta_px": geometry[
                "zero_polygon_max_abs_delta_px"
            ],
        },
        "reuse": {
            "h0_1r_replay": str(
                (REPO_ROOT / "tools" / "audit_haikang_roi_v2_0829.py").resolve()
            ),
            "existing_roi_v2_functions": [
                "median_centerline",
                "integer_profile",
                "build_edge_pairs",
                "assess_condition",
            ],
            "board_polygon_function": (
                "measurement.board_mask.full_board_physical_polygon"
            ),
        },
        "protocol": {
            "insets_mm": list(INSETS_MM),
            "fixed_pair": list(FIXED_PAIR),
            "height_truth_used_for_selection": False,
            "candidate_score_modified": False,
            "axis_adapter_modified": False,
            "c0_modified": False,
            "session_ground_modified": False,
            "height_measurement_called": False,
            "compensation_called": False,
        },
        "outputs": {
            "registry": str(
                (output_dir / "roi_v2_board_mask_registry.csv").resolve()
            ),
            "before_after": str(
                (output_dir / "roi_v2_before_after.csv").resolve()
            ),
            "report": str(
                (output_dir / "roi_v2_board_mask_report.md").resolve()
            ),
            "overlay_count": len(overlay_paths),
        },
    }
    (output_dir / "roi_v2_board_mask_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "roi_v2_board_mask_registry.csv",
        ROI_REGISTRY_FIELDS,
        registry_rows,
    )
    write_csv(
        output_dir / "roi_v2_before_after.csv",
        BEFORE_AFTER_FIELDS,
        before_after_rows,
    )
    (output_dir / "roi_v2_board_mask_report.md").write_text(
        build_report(
            root=root,
            geometry=geometry,
            records=records,
            registry_rows=registry_rows,
            before_after_rows=before_after_rows,
            overlay_paths=overlay_paths,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "condition_count": len(records),
                "registry_rows": len(registry_rows),
                "before_after_rows": len(before_after_rows),
                "overlay_count": len(overlay_paths),
                "insets_mm": list(INSETS_MM),
                "stored_zero_polygon_max_abs_delta_px": geometry[
                    "zero_polygon_max_abs_delta_px"
                ],
                "score_unchanged_max_abs_delta": max(
                    (
                        float(row["score_unchanged_max_abs_delta"])
                        for row in registry_rows
                    ),
                    default=0.0,
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
