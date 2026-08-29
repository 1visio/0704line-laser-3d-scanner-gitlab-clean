"""Ground-5B Session Linear minimality audit.

This diagnostic consumes the Ground-5A frozen Factory candidate and the
Ground-5A center cache.  It deliberately refuses to run Steger when the
cache is missing or incompatible.  PnP, C0/C1 reconstruction and the
physical-board selector are the existing Ground-5A entry points; no new
point-selection or Factory fitting logic is introduced here.

Chains on the strict common Factory support are:

``A: Zg``
``B: Zg - F(full_v)``
``C: Zg - F(full_v) - session_linear(full_v)``
``D-v: Zg - session_linear(full_v)``
``D-S: Zg - session_linear(physical_S)``

For D, the session line is fitted on all board-mask-selected points.  The
same fitted D line is also evaluated on the common Factory support so that
C-versus-D measures the Factory increment after Session Linear.  All linear
fits use the Ground-5A frame-balanced least-squares convention: equal total
weight per frame and equal point weight within a frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from tools import fit_ground5a_factory_profile as g5a


FIT_POSES = g5a.FIT_POSES
HELDOUT_POSES = g5a.HELDOUT_POSES
ALL_POSES = g5a.ALL_POSES
CHAINS = ("A", "B", "C", "D-v", "D-S")
ALL_BOARD_SCOPE = "all_board_points"
COMMON_SCOPE = "common_factory_support"

# These are engineering decision thresholds declared before reading the
# held-out metrics.  They do not change any fitted parameter.
SESSION_STRONG_RMSE_GAIN = 0.10
SESSION_STRONG_P95_GAIN = 0.05
FACTORY_INCREMENT_RMSE_THRESHOLD_MM = 0.005
FACTORY_INCREMENT_P95_THRESHOLD_MM = 0.01

DEFAULT_FIT_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\chessboard_0821\fit"
)
DEFAULT_VALIDATION_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\chessboard_0821\validation"
)
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_GROUND5A_OUTPUT = TOOL_ROOT.parent / "outputs" / "ground5a_factory_profile_0821"
DEFAULT_OUTPUT = TOOL_ROOT.parent / "outputs" / "ground5b_session_linear_minimality_0821"
DEFAULT_GROUND1_SUMMARY = (
    TOOL_ROOT.parent
    / "reports"
    / "experiments"
    / "daheng_0811"
    / "ground_reference_20frames"
    / "ground_reference_summary.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "bias_mm": math.nan,
            "rmse_mm": math.nan,
            "p95_abs_mm": math.nan,
            "max_abs_mm": math.nan,
            "peak_to_peak_mm": math.nan,
        }
    return {
        "bias_mm": float(np.mean(values)),
        "rmse_mm": float(np.sqrt(np.mean(values**2))),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95.0)),
        "max_abs_mm": float(np.max(np.abs(values))),
        "peak_to_peak_mm": float(np.ptp(values)),
    }


def _report_number(value: Any, digits: int = 7) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(number) else f"{number:.{digits}g}"


def _load_ground1_s(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    definition = document.get("shared_s_definition")
    if not isinstance(definition, dict):
        raise RuntimeError(f"Ground-1 summary has no shared_s_definition: {path}")
    if definition.get("formula") != "S=(XY-origin_xy) dot direction_xy":
        raise RuntimeError(f"unexpected Ground-1 S formula in {path}")
    origin = np.asarray(definition.get("origin_xy"), dtype=np.float64)
    direction = np.asarray(definition.get("direction_xy"), dtype=np.float64)
    if origin.shape != (2,) or direction.shape != (2,):
        raise RuntimeError(f"invalid Ground-1 S axis in {path}")
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-6:
        raise RuntimeError(f"Ground-1 direction is not unit length in {path}: {norm}")
    return {
        "source_path": path,
        "source_sha256": g5a._sha256_file(path),
        "origin_xy": origin,
        "direction_xy": direction,
        "formula": definition["formula"],
        "origin_source": definition.get("origin_source"),
        "direction_source": definition.get("direction_source"),
        "per_frame_redefinition": definition.get("per_frame_redefinition"),
    }


def _load_factory_candidate(path: Path) -> tuple[g5a.FactorySpline, dict[str, Any]]:
    candidate = _read_json(path)
    if candidate.get("coordinate") != "full_v":
        raise RuntimeError(f"Ground-5B requires the frozen Ground-5A full_v candidate: {path}")
    if candidate.get("ground3_numeric_parameters_reused") is not False:
        raise RuntimeError("Ground-5A candidate provenance says Ground-3 numeric parameters were reused")
    knots = np.asarray(candidate["knots"], dtype=np.float64)
    coefficients = np.asarray(candidate["coefficients_mm"], dtype=np.float64)
    if len(knots) != len(coefficients) + int(candidate["degree"]) + 1:
        raise RuntimeError(f"invalid Frozen Factory spline dimensions: {path}")
    factory = g5a.FactorySpline(
        interior_knot_count=int(candidate["interior_knot_count"]),
        knots=knots,
        coefficients=coefficients,
        domain_min=float(candidate["support_domain"][0]),
        domain_max=float(candidate["support_domain"][1]),
        degree=int(candidate["degree"]),
        smoothness_lambda=float(candidate["smoothness_lambda"]),
        train_pose_ids=tuple(candidate["fit_pose_ids"]),
        observation_count=int(candidate.get("observation_count", 0)),
        fit_rmse_mm=float(candidate["fit_rmse_mm"]),
        cv_rmse_mm=float(candidate["leave_one_fit_pose_out_cv_rmse_mm"]),
    )
    return factory, candidate


def _load_cached_centers_only(
    records: list[g5a.PoseRecord],
    cache_output: Path,
    app: Any,
    config_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read the existing Ground-5A cache; never fall back to Steger."""
    cache_path = cache_output / "steger_geometry_cache.json"
    if not cache_path.is_file():
        raise RuntimeError(f"Ground-5A Steger cache is missing: {cache_path}")
    cache = _read_json(cache_path)
    if cache.get("one_steger_per_frame") is not True:
        raise RuntimeError("Ground-5A cache does not declare one Steger run per frame")
    if cache.get("protocol_key") != g5a._cache_key(app, config_path):
        raise RuntimeError("Ground-5A cache protocol key does not match current config/extraction settings")
    desired_paths = [
        str(path.resolve())
        for record in records
        for path in record.laser_paths
    ]
    cached_by_path = {str(item["source_path"]): item for item in cache.get("frames", [])}
    if set(desired_paths) != set(cached_by_path):
        raise RuntimeError("Ground-5A cache frame set does not exactly match poses001-007")
    centers_by_path: dict[str, np.ndarray] = {}
    for source_path in desired_paths:
        source = Path(source_path)
        item = cached_by_path[source_path]
        source_sha = g5a._sha256_file(source)
        if item.get("source_sha256") != source_sha:
            raise RuntimeError(f"source SHA mismatch in Ground-5A cache: {source}")
        if int(item.get("steger_run_count", 0)) != 1:
            raise RuntimeError(f"cache entry is not exactly one Steger run: {source}")
        center_path = Path(item["centers_path"])
        centers = np.asarray(np.load(center_path), dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
            raise RuntimeError(f"invalid cached centers: {center_path}")
        centers_by_path[source_path] = np.ascontiguousarray(centers)
    cache["read_only_reuse_in_ground5b"] = True
    cache["cache_source_path"] = cache_path
    cache["cache_source_sha256"] = g5a._sha256_file(cache_path)
    return centers_by_path, cache


def _physical_s(frame: g5a.GroundFrame, origin_xy: np.ndarray, direction_xy: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        (np.asarray(frame.points_ground[:, :2], dtype=np.float64) - origin_xy[None, :]) @ direction_xy
    )


def _verify_ground5a_frame_counts(
    ground5a_summary: dict[str, Any],
    frames: list[g5a.GroundFrame],
) -> dict[str, Any]:
    expected = {
        str(row["frame_id"]): (
            int(row["point_count"]),
            int(row["c0_point_count"]),
            int(row["c1_point_count"]),
            int(row["board_mask_point_count"]),
        )
        for row in ground5a_summary.get("frame_metrics", [])
        if row.get("coordinate") == "full_v"
    }
    actual = {
        frame.frame_id: (
            int(len(frame.coordinates["full_v"].x)),
            int(frame.c0_point_count),
            int(frame.c1_point_count),
            int(frame.c1_selected_count),
        )
        for frame in frames
    }
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key])
        raise RuntimeError(
            f"Ground-5B deterministic frame provenance mismatch: missing={missing}, extra={extra}, changed={changed}"
        )
    return {
        "ground5a_frame_metric_rows_compared": len(expected),
        "ground5a_point_and_reconstruction_counts_match": True,
    }


def _fit_frame_balanced_line(
    frame_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float, int, int]:
    """Fit a line with Ground-5A's equal-frame/equal-point weighting."""
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for x, y, mask in frame_data:
        finite = np.isfinite(x) & np.isfinite(y) & np.asarray(mask, dtype=bool)
        if np.count_nonzero(finite) < 2:
            continue
        x_values.append(np.asarray(x[finite], dtype=np.float64))
        y_values.append(np.asarray(y[finite], dtype=np.float64))
        weights.append(np.full(np.count_nonzero(finite), 1.0 / np.count_nonzero(finite), dtype=np.float64))
    if not x_values:
        raise RuntimeError("session linear fit has no supported frame")
    x = np.concatenate(x_values)
    y = np.concatenate(y_values)
    weight = np.concatenate(weights)
    weight /= float(len(x_values))
    slope, intercept = np.linalg.lstsq(
        np.column_stack([x, np.ones_like(x)]) * np.sqrt(weight)[:, None],
        y * np.sqrt(weight),
        rcond=None,
    )[0]
    return float(slope), float(intercept), int(len(x)), int(len(x_values))


def _frame_data(
    frames: list[g5a.GroundFrame],
    factory: g5a.FactorySpline,
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    data: dict[str, dict[str, np.ndarray]] = {}
    for frame in frames:
        z = np.asarray(frame.points_ground[:, 2], dtype=np.float64)
        v = np.asarray(frame.coordinates["full_v"].x, dtype=np.float64)
        s = _physical_s(frame, origin_xy, direction_xy)
        factory_value, factory_supported = g5a._predict_factory(factory, v)
        factory_supported &= np.isfinite(z) & np.isfinite(factory_value)
        data[frame.frame_id] = {
            "v": v,
            "S": s,
            "z": z,
            "factory": factory_value,
            "factory_supported": factory_supported,
        }
    return data


def _fit_session_parameters(
    validation_frames: list[g5a.GroundFrame],
    data: dict[str, dict[str, np.ndarray]],
    factory: g5a.FactorySpline,
) -> dict[tuple[str, str], dict[str, Any]]:
    parameters: dict[tuple[str, str], dict[str, Any]] = {}
    for pose in HELDOUT_POSES:
        pose_frames = [frame for frame in validation_frames if frame.pose_id == pose]
        c_data = [
            (
                data[frame.frame_id]["v"],
                data[frame.frame_id]["z"] - data[frame.frame_id]["factory"],
                data[frame.frame_id]["factory_supported"],
            )
            for frame in pose_frames
        ]
        c_a, c_b, c_count, c_frame_count = _fit_frame_balanced_line(c_data)
        parameters[(pose, "C")] = {
            "chain": "C",
            "coordinate": "full_v",
            "a": c_a,
            "b": c_b,
            "fit_point_count": c_count,
            "fit_frame_count": c_frame_count,
            "fit_scope": COMMON_SCOPE,
            "factory_applied": True,
            "fit_source": "heldout_pose_laser_ground_only",
        }
        for chain, coordinate in (("D-v", "v"), ("D-S", "S")):
            d_data = [
                (
                    data[frame.frame_id][coordinate],
                    data[frame.frame_id]["z"],
                    np.ones(len(data[frame.frame_id]["z"]), dtype=bool),
                )
                for frame in pose_frames
            ]
            a, b, count, frame_count = _fit_frame_balanced_line(d_data)
            parameters[(pose, chain)] = {
                "chain": chain,
                "coordinate": "full_v" if chain == "D-v" else "physical_S",
                "a": a,
                "b": b,
                "fit_point_count": count,
                "fit_frame_count": frame_count,
                "fit_scope": ALL_BOARD_SCOPE,
                "factory_applied": False,
                "fit_source": "heldout_pose_laser_ground_only",
            }
    return parameters


def _chain_values(
    chain: str,
    scope: str,
    frame_id: str,
    data: dict[str, dict[str, np.ndarray]],
    parameters: dict[tuple[str, str], dict[str, Any]],
    pose: str,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    item = data[frame_id]
    z = item["z"]
    if scope == ALL_BOARD_SCOPE:
        if chain not in ("A", "D-v", "D-S"):
            return None
        mask = np.ones(len(z), dtype=bool)
    else:
        mask = item["factory_supported"]
    if chain == "A":
        residual = z
        coordinate = item["v"]
    elif chain == "B":
        residual = z - item["factory"]
        coordinate = item["v"]
    elif chain == "C":
        p = parameters[(pose, "C")]
        residual = z - item["factory"] - (p["a"] * item["v"] + p["b"])
        coordinate = item["v"]
    elif chain == "D-v":
        p = parameters[(pose, "D-v")]
        residual = z - (p["a"] * item["v"] + p["b"])
        coordinate = item["v"]
    elif chain == "D-S":
        p = parameters[(pose, "D-S")]
        residual = z - (p["a"] * item["S"] + p["b"])
        coordinate = item["S"]
    else:
        raise ValueError(chain)
    return np.asarray(coordinate[mask], dtype=np.float64), np.asarray(residual[mask], dtype=np.float64), True


def _build_metric_rows(
    validation_frames: list[g5a.GroundFrame],
    data: dict[str, dict[str, np.ndarray]],
    parameters: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scope_chains = {
        ALL_BOARD_SCOPE: ("A", "D-v", "D-S"),
        COMMON_SCOPE: CHAINS,
    }
    for scope, chains in scope_chains.items():
        for frame in validation_frames:
            item = data[frame.frame_id]
            for chain in chains:
                result = _chain_values(chain, scope, frame.frame_id, data, parameters, frame.pose_id)
                if result is None:
                    continue
                x, residual, _ = result
                metrics = _metric_values(residual)
                parameter = parameters.get((frame.pose_id, chain))
                rows.append(
                    {
                        "aggregation_level": "frame",
                        "evaluation_scope": scope,
                        "split": "validation",
                        "pose_id": frame.pose_id,
                        "frame_id": frame.frame_id,
                        "chain": chain,
                        "coordinate": "physical_S" if chain == "D-S" else "full_v",
                        "factory_applied": chain in ("B", "C"),
                        "session_linear_applied": chain in ("C", "D-v", "D-S"),
                        "point_count_board_mask": len(item["z"]),
                        "evaluation_point_count": len(residual),
                        "factory_support_fraction": float(np.mean(item["factory_supported"])),
                        "bias_mm": metrics["bias_mm"],
                        "rmse_mm": metrics["rmse_mm"],
                        "p95_abs_mm": metrics["p95_abs_mm"],
                        "max_abs_mm": metrics["max_abs_mm"],
                        "peak_to_peak_mm": metrics["peak_to_peak_mm"],
                        "session_a_mm_per_coordinate": parameter["a"] if parameter else "",
                        "session_b_mm": parameter["b"] if parameter else "",
                        "session_fit_point_count": parameter["fit_point_count"] if parameter else "",
                        "session_fit_frame_count": parameter["fit_frame_count"] if parameter else "",
                        "session_fit_scope": parameter["fit_scope"] if parameter else "",
                        "fit_source": parameter["fit_source"] if parameter else "",
                        "residual_or_truth_used_for_mask": False,
                        "frame_repeatability_rmse_mm": "",
                    }
                )
    for scope, chains in scope_chains.items():
        for pose in HELDOUT_POSES:
            for chain in chains:
                selected = [
                    row
                    for row in rows
                    if row["aggregation_level"] == "frame"
                    and row["evaluation_scope"] == scope
                    and row["pose_id"] == pose
                    and row["chain"] == chain
                ]
                if not selected:
                    continue
                mean_bias = float(np.mean([row["bias_mm"] for row in selected]))
                repeatability = float(
                    np.sqrt(np.mean((np.asarray([row["bias_mm"] for row in selected]) - mean_bias) ** 2))
                )
                first = selected[0]
                pose_row = dict(first)
                pose_row.update(
                    {
                        "aggregation_level": "pose",
                        "frame_id": "pose_mean_equal_frame_weight",
                        "evaluation_point_count": int(round(np.mean([row["evaluation_point_count"] for row in selected]))),
                        "point_count_board_mask": int(round(np.mean([row["point_count_board_mask"] for row in selected]))),
                        "factory_support_fraction": float(np.mean([row["factory_support_fraction"] for row in selected])),
                        "bias_mm": mean_bias,
                        "rmse_mm": float(np.mean([row["rmse_mm"] for row in selected])),
                        "p95_abs_mm": float(np.mean([row["p95_abs_mm"] for row in selected])),
                        "max_abs_mm": float(np.mean([row["max_abs_mm"] for row in selected])),
                        "peak_to_peak_mm": float(np.mean([row["peak_to_peak_mm"] for row in selected])),
                        "frame_repeatability_rmse_mm": repeatability,
                    }
                )
                rows.append(pose_row)
    return rows


def _pose_row(rows: list[dict[str, Any]], pose: str, scope: str, chain: str) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["aggregation_level"] == "pose"
        and row["pose_id"] == pose
        and row["evaluation_scope"] == scope
        and row["chain"] == chain
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one pose row for {pose}/{scope}/{chain}, got {len(matches)}")
    return matches[0]


def _session_coordinate_comparison(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for scope in (ALL_BOARD_SCOPE, COMMON_SCOPE):
        for coordinate, chain in (("full_v", "D-v"), ("physical_S", "D-S")):
            pose_rows = [_pose_row(rows, pose, scope, chain) for pose in HELDOUT_POSES]
            a_rows = [_pose_row(rows, pose, scope, "A") for pose in HELDOUT_POSES]
            c_rows = [_pose_row(rows, pose, scope, "C") for pose in HELDOUT_POSES] if scope == COMMON_SCOPE else []
            mean_rmse = float(np.mean([row["rmse_mm"] for row in pose_rows]))
            mean_p95 = float(np.mean([row["p95_abs_mm"] for row in pose_rows]))
            mean_repeatability = float(np.mean([row["frame_repeatability_rmse_mm"] for row in pose_rows]))
            for pose, d_row, a_row in zip(HELDOUT_POSES, pose_rows, a_rows, strict=True):
                c_row = _pose_row(rows, pose, scope, "C") if scope == COMMON_SCOPE else None
                output.append(
                    {
                        "evaluation_scope": scope,
                        "pose_id": pose,
                        "coordinate": coordinate,
                        "chain": chain,
                        "rmse_mm": d_row["rmse_mm"],
                        "p95_abs_mm": d_row["p95_abs_mm"],
                        "max_abs_mm": d_row["max_abs_mm"],
                        "peak_to_peak_mm": d_row["peak_to_peak_mm"],
                        "bias_mm": d_row["bias_mm"],
                        "frame_repeatability_rmse_mm": d_row["frame_repeatability_rmse_mm"],
                        "rmse_gain_vs_A": 1.0 - d_row["rmse_mm"] / a_row["rmse_mm"] if a_row["rmse_mm"] else math.nan,
                        "p95_gain_vs_A": 1.0 - d_row["p95_abs_mm"] / a_row["p95_abs_mm"] if a_row["p95_abs_mm"] else math.nan,
                        "factory_increment_rmse_D_minus_C_mm": d_row["rmse_mm"] - c_row["rmse_mm"] if c_row else "",
                        "factory_increment_p95_D_minus_C_mm": d_row["p95_abs_mm"] - c_row["p95_abs_mm"] if c_row else "",
                        "mean_coordinate_rmse_mm": mean_rmse,
                        "mean_coordinate_p95_abs_mm": mean_p95,
                        "mean_coordinate_frame_repeatability_rmse_mm": mean_repeatability,
                        "selected": False,
                    }
                )
            scores.append(
                {
                    "evaluation_scope": scope,
                    "coordinate": coordinate,
                    "chain": chain,
                    "mean_rmse_mm": mean_rmse,
                    "mean_p95_abs_mm": mean_p95,
                    "mean_frame_repeatability_rmse_mm": mean_repeatability,
                }
            )
    all_scores = [row for row in scores if row["evaluation_scope"] == ALL_BOARD_SCOPE]
    # Lower residual metrics win; physical_S wins an exact tie because it is
    # the established physical coordinate rather than a sensor coordinate.
    chosen_score = sorted(
        all_scores,
        key=lambda row: (
            row["mean_rmse_mm"],
            row["mean_p95_abs_mm"],
            row["mean_frame_repeatability_rmse_mm"],
            0 if row["coordinate"] == "physical_S" else 1,
        ),
    )[0]
    for row in output:
        if row["evaluation_scope"] == ALL_BOARD_SCOPE and row["coordinate"] == chosen_score["coordinate"]:
            row["selected"] = True
    return output, {
        "rule": [
            "compare D-v and D-S on all board-mask-selected points",
            "minimize mean pose RMSE, then mean pose P95, then mean frame repeatability",
            "exact ties prefer the frozen Ground-1 physical_S coordinate",
        ],
        "chosen_coordinate": chosen_score["coordinate"],
        "chosen_chain": chosen_score["chain"],
        "scores": scores,
    }


def _classification(
    rows: list[dict[str, Any]],
    coordinate_selection: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any], str]:
    chosen_coordinate = coordinate_selection["chosen_coordinate"]
    chosen_chain = coordinate_selection["chosen_chain"]
    session_effect_rows: list[dict[str, Any]] = []
    for pose in HELDOUT_POSES:
        a = _pose_row(rows, pose, ALL_BOARD_SCOPE, "A")
        d = _pose_row(rows, pose, ALL_BOARD_SCOPE, chosen_chain)
        session_effect_rows.append(
            {
                "pose_id": pose,
                "rmse_gain": 1.0 - d["rmse_mm"] / a["rmse_mm"],
                "p95_gain": 1.0 - d["p95_abs_mm"] / a["p95_abs_mm"],
                "A_rmse_mm": a["rmse_mm"],
                "D_rmse_mm": d["rmse_mm"],
                "A_p95_abs_mm": a["p95_abs_mm"],
                "D_p95_abs_mm": d["p95_abs_mm"],
            }
        )
    session_strong = all(
        row["rmse_gain"] >= SESSION_STRONG_RMSE_GAIN and row["p95_gain"] >= SESSION_STRONG_P95_GAIN
        for row in session_effect_rows
    )
    factory_rows: list[dict[str, Any]] = []
    for pose in HELDOUT_POSES:
        c = _pose_row(rows, pose, COMMON_SCOPE, "C")
        d = _pose_row(rows, pose, COMMON_SCOPE, chosen_chain)
        factory_rows.append(
            {
                "pose_id": pose,
                "D_minus_C_rmse_mm": d["rmse_mm"] - c["rmse_mm"],
                "D_minus_C_p95_abs_mm": d["p95_abs_mm"] - c["p95_abs_mm"],
                "C_minus_D_rmse_mm": c["rmse_mm"] - d["rmse_mm"],
                "C_minus_D_p95_abs_mm": c["p95_abs_mm"] - d["p95_abs_mm"],
                "C_rmse_mm": c["rmse_mm"],
                "D_rmse_mm": d["rmse_mm"],
                "C_p95_abs_mm": c["p95_abs_mm"],
                "D_p95_abs_mm": d["p95_abs_mm"],
            }
        )
    factory_useful = all(
        row["D_minus_C_rmse_mm"] >= FACTORY_INCREMENT_RMSE_THRESHOLD_MM
        and row["D_minus_C_p95_abs_mm"] >= FACTORY_INCREMENT_P95_THRESHOLD_MM
        for row in factory_rows
    )
    classification = {
        "SESSION_LINEAR_EFFECT": "STRONG" if session_strong else "WEAK",
        "FACTORY_PROFILE_INCREMENT_AFTER_SESSION_LINEAR": "USEFUL" if factory_useful else "NEUTRAL",
        "RECOMMENDED_SESSION_COORDINATE": chosen_coordinate,
    }
    if factory_useful:
        production_chain = (
            f"PnP -> Frozen C0+C1 -> Frozen Factory F(full_v) -> "
            f"held-out Session Linear in {chosen_coordinate}"
        )
    else:
        production_chain = f"PnP -> Frozen C0+C1 -> Session Linear in {chosen_coordinate}; omit Factory Profile"
    diagnostics = {
        "session_effect_by_pose": session_effect_rows,
        "factory_increment_by_pose": factory_rows,
        "thresholds": {
            "session_strong_rmse_gain": SESSION_STRONG_RMSE_GAIN,
            "session_strong_p95_gain": SESSION_STRONG_P95_GAIN,
            "factory_increment_rmse_mm": FACTORY_INCREMENT_RMSE_THRESHOLD_MM,
            "factory_increment_p95_mm": FACTORY_INCREMENT_P95_THRESHOLD_MM,
        },
    }
    return classification, diagnostics, production_chain


def _plot_abcd(path: Path, rows: list[dict[str, Any]], coordinate_selection: dict[str, Any]) -> None:
    pose_rows = [row for row in rows if row["aggregation_level"] == "pose" and row["evaluation_scope"] == COMMON_SCOPE]
    chains = CHAINS
    colours = {"A": "tab:gray", "B": "tab:blue", "C": "tab:orange", "D-v": "tab:green", "D-S": "tab:red"}
    x = np.arange(len(HELDOUT_POSES))
    width = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for offset, chain in enumerate(chains):
        values = [next(row["rmse_mm"] for row in pose_rows if row["pose_id"] == pose and row["chain"] == chain) for pose in HELDOUT_POSES]
        p95 = [next(row["p95_abs_mm"] for row in pose_rows if row["pose_id"] == pose and row["chain"] == chain) for pose in HELDOUT_POSES]
        axes[0].bar(x + (offset - 2) * width, values, width, label=chain, color=colours[chain])
        axes[1].bar(x + (offset - 2) * width, p95, width, label=chain, color=colours[chain])
    for axis, ylabel in zip(axes, ("RMSE (mm)", "P95 abs (mm)"), strict=True):
        axis.set_xticks(x, [f"pose{pose}" for pose in HELDOUT_POSES])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=2)
    fig.suptitle(f"Ground-5B A/B/C/D on common Frozen Factory support; selected Session coordinate={coordinate_selection['chosen_coordinate']}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_coordinate_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    pose_rows = [row for row in rows if row["aggregation_level"] == "pose" and row["evaluation_scope"] == ALL_BOARD_SCOPE]
    x = np.arange(len(HELDOUT_POSES))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for offset, chain in enumerate(("D-v", "D-S")):
        d_rows = [next(row for row in pose_rows if row["pose_id"] == pose and row["chain"] == chain) for pose in HELDOUT_POSES]
        axes[0].bar(x + (offset - 0.5) * width, [row["rmse_mm"] for row in d_rows], width, label=chain)
        axes[1].bar(x + (offset - 0.5) * width, [row["p95_abs_mm"] for row in d_rows], width, label=chain)
    for axis, ylabel in zip(axes, ("RMSE (mm)", "P95 abs (mm)"), strict=True):
        axis.set_xticks(x, [f"pose{pose}" for pose in HELDOUT_POSES])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("D-v vs D-S on all board-mask-selected points")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_c_minus_d(
    path: Path,
    validation_frames: list[g5a.GroundFrame],
    data: dict[str, dict[str, np.ndarray]],
    parameters: dict[tuple[str, str], dict[str, Any]],
    chosen_chain: str,
) -> None:
    fig, axes = plt.subplots(len(HELDOUT_POSES), 1, figsize=(13, 8), sharex=False)
    if len(HELDOUT_POSES) == 1:
        axes = [axes]
    for axis, pose in zip(axes, HELDOUT_POSES, strict=True):
        pose_frames = [frame for frame in validation_frames if frame.pose_id == pose]
        for frame in pose_frames:
            item = data[frame.frame_id]
            supported = item["factory_supported"]
            c = item["z"] - item["factory"] - (
                parameters[(pose, "C")]["a"] * item["v"] + parameters[(pose, "C")]["b"]
            )
            if chosen_chain == "D-v":
                d = item["z"] - (parameters[(pose, "D-v")]["a"] * item["v"] + parameters[(pose, "D-v")]["b"])
            else:
                d = item["z"] - (parameters[(pose, "D-S")]["a"] * item["S"] + parameters[(pose, "D-S")]["b"])
            axis.plot(item["v"][supported], (c - d)[supported], linewidth=0.8, alpha=0.65)
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set_title(f"pose{pose}: C - {chosen_chain}; negative means Factory improves over Session Linear")
        axis.set_ylabel("residual delta (mm)")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("full-sensor v (px); common Factory support only")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_report(
    path: Path,
    summary: dict[str, Any],
    output_files: list[str],
) -> None:
    classification = summary["classification"]
    lines = [
        "# Ground-5B｜Session Linear Minimality Audit",
        "",
        f"数据集：`chessboard_0821`；生成时间：`{summary['created_at_local']}`。",
        "",
        "## 最终结论",
        "",
        f"- `SESSION_LINEAR_EFFECT = {classification['SESSION_LINEAR_EFFECT']}`",
        f"- `FACTORY_PROFILE_INCREMENT_AFTER_SESSION_LINEAR = {classification['FACTORY_PROFILE_INCREMENT_AFTER_SESSION_LINEAR']}`",
        f"- `RECOMMENDED_SESSION_COORDINATE = {classification['RECOMMENDED_SESSION_COORDINATE']}`",
        f"- `PRODUCTION_GROUND_CHAIN = {summary['production_ground_chain']}`",
        "",
        "## Protocol lock / provenance",
        "",
        "- Fit poses remain `001–005`; held-out poses are `006–007`.",
        "- Ground-5A PnP, physical-board mask, cached centers, Frozen C0/C1 and Factory candidate are reused.",
        "- Ground-5B reads the existing Steger cache in read-only mode and refuses cache miss; no Steger extraction is performed.",
        "- No C0/C1 refit, H1, Session Ground Reference or Ground-3 numeric parameter reuse.",
        "- The Factory candidate is loaded from Ground-5A JSON; it is not refit or retuned.",
        f"- Ground-5A Factory coordinate/domain: `full_v`, `[{_report_number(summary['factory_profile']['domain_min'])}, {_report_number(summary['factory_profile']['domain_max'])}]`.",
        "",
        "### Frozen physical S",
        "",
        f"`S = (XY - origin_xy) dot direction_xy`, with `origin_xy={summary['ground1_s']['origin_xy']}` mm and `direction_xy={summary['ground1_s']['direction_xy']}`.",
        "This is the Ground-1/4A global physical along-stripe coordinate, not a height-measurement obstacle-local axis.",
        "",
        "## Session Linear parameters",
        "",
        "D-v and D-S parameters are fitted independently for each held-out pose using all board-mask-selected points, with equal total frame weight. C uses only the common Factory support, matching Ground-5A.",
        "",
        "| pose | chain | coordinate | fit scope | a | b | fit points |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in summary["session_parameters"]:
        lines.append(
            f"| {row['pose_id']} | {row['chain']} | {row['coordinate']} | {row['fit_scope']} | "
            f"{_report_number(row['a'])} | {_report_number(row['b'])} | {row['fit_point_count']} |"
        )
    lines.extend([
        "",
        "## Held-out metrics: common Factory support",
        "",
        "Metrics are equal-frame means at pose level. `frame_repeatability_rmse_mm` is the RMS spread of per-frame bias around the pose mean; mask selection is unchanged.",
        "",
        "| pose | chain | coordinate | Bias | RMSE | P95 | Max | P2P | frame repeatability |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for pose in HELDOUT_POSES:
        for chain in CHAINS:
            row = _pose_row(summary["abcd_comparison"], pose, COMMON_SCOPE, chain)
            lines.append(
                f"| {pose} | {chain} | {row['coordinate']} | {_report_number(row['bias_mm'])} | {_report_number(row['rmse_mm'])} | "
                f"{_report_number(row['p95_abs_mm'])} | {_report_number(row['max_abs_mm'])} | {_report_number(row['peak_to_peak_mm'])} | "
                f"{_report_number(row['frame_repeatability_rmse_mm'])} |"
            )
    lines.extend([
        "",
        "## Held-out metrics: all board-mask points",
        "",
        "A/D comparison uses every point retained by the unchanged physical-board mask; `Max` is maximum absolute residual.",
        "",
        "| pose | chain | coordinate | Bias | RMSE | P95 | Max | P2P | frame repeatability |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for pose in HELDOUT_POSES:
        for chain in ("A", "D-v", "D-S"):
            row = _pose_row(summary["abcd_comparison"], pose, ALL_BOARD_SCOPE, chain)
            lines.append(
                f"| {pose} | {chain} | {row['coordinate']} | {_report_number(row['bias_mm'])} | {_report_number(row['rmse_mm'])} | "
                f"{_report_number(row['p95_abs_mm'])} | {_report_number(row['max_abs_mm'])} | {_report_number(row['peak_to_peak_mm'])} | "
                f"{_report_number(row['frame_repeatability_rmse_mm'])} |"
            )
    lines.extend([
        "",
        "## D vs A and C vs D",
        "",
        "D-v/D-S vs A uses all board-mask-selected points. C vs D uses the same strict Factory support for all chains; positive `D-C` means C is better.",
        "",
        "| pose | selected D | D RMSE gain vs A | D P95 gain vs A | D-C RMSE (mm) | D-C P95 (mm) |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for session_row, factory_row in zip(summary["diagnostics"]["session_effect_by_pose"], summary["diagnostics"]["factory_increment_by_pose"], strict=True):
        lines.append(
            f"| {session_row['pose_id']} | {summary['coordinate_selection']['chosen_chain']} | {_report_number(session_row['rmse_gain'])} | "
            f"{_report_number(session_row['p95_gain'])} | {_report_number(factory_row['D_minus_C_rmse_mm'])} | {_report_number(factory_row['D_minus_C_p95_abs_mm'])} |"
        )
    lines.extend([
        "",
        "Thresholds: Session Linear STRONG requires both held-out poses to reach 10% RMSE and 5% P95 gain. Factory increment USEFUL requires C to improve over selected D by at least 0.005 mm RMSE and 0.01 mm P95 for both poses.",
        "",
        "## Coordinate decision",
        "",
        "D-v and D-S are ranked on all board-mask-selected points by mean pose RMSE, then P95, then frame repeatability; exact ties prefer frozen physical_S.",
        "",
        "| scope | coordinate | mean RMSE | mean P95 | mean frame repeatability | selected |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in summary["coordinate_selection"]["scores"]:
        lines.append(
            f"| {row['evaluation_scope']} | {row['coordinate']} | {_report_number(row['mean_rmse_mm'])} | "
            f"{_report_number(row['mean_p95_abs_mm'])} | {_report_number(row['mean_frame_repeatability_rmse_mm'])} | "
            f"{'YES' if row['evaluation_scope'] == ALL_BOARD_SCOPE and row['coordinate'] == summary['coordinate_selection']['chosen_coordinate'] else 'NO'} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
    ])
    lines.extend(f"- `{name}`" for name in output_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    fit_dir = args.fit_dir.resolve()
    validation_dir = args.validation_dir.resolve()
    config_path = args.config.resolve()
    ground5a_output = args.ground5a_output.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_records = g5a._discover_pose_records(fit_dir, "fit", FIT_POSES)
    validation_records = g5a._discover_pose_records(validation_dir, "validation", HELDOUT_POSES)
    all_records = fit_records + validation_records
    dataset_document, metadata_by_name, manifest_path, frames_csv_path = g5a._load_manifest_context(fit_dir)
    ground1_s = _load_ground1_s(args.ground1_summary.resolve())
    factory, factory_candidate = _load_factory_candidate(ground5a_output / "factory_profile_candidate.json")
    ground5a_summary = _read_json(ground5a_output / "ground5a_summary.json")
    if ground5a_summary["dataset"]["fit_pose_ids"] != list(FIT_POSES) or ground5a_summary["dataset"]["heldout_pose_ids"] != list(HELDOUT_POSES):
        raise RuntimeError("Ground-5A split does not match the locked 001-005/006-007 protocol")
    if ground5a_summary["configuration"]["c0_refit"] or ground5a_summary["configuration"]["c1_refit"]:
        raise RuntimeError("Ground-5A provenance reports a C0/C1 refit")

    app = g5a.load_app_config(config_path)
    if app.reconstruction.image_roi_polygon is not None or not app.reconstruction.enable_laser_ray_correction:
        raise RuntimeError("Ground-5B requires full-sensor Frozen C1 reconstruction")
    if app.calibration.manifest is None:
        raise RuntimeError("Ground-5B requires the existing calibration manifest")
    package = g5a.load_calibration_package(app.calibration.manifest)
    base_calibration = dict(package.calibration)
    params_c0 = g5a.replace(app.reconstruction, enable_laser_ray_correction=False)
    params_c1 = app.reconstruction
    board_config = app.session_ground_calibration.board_config()
    mask_inset_mm = float(app.session_ground_calibration.sanity.mask_inset_mm)
    if mask_inset_mm != float(app.session_ground_calibration.ground_reference.mask_inset_mm):
        raise RuntimeError("Ground-5B requires the existing physical-board mask inset agreement")

    centers_by_path, cache = _load_cached_centers_only(all_records, ground5a_output, app, config_path)
    pnp_by_pose = g5a._run_pnp(all_records, base_calibration, board_config)
    all_frames = g5a._process_frames(
        all_records,
        pnp_by_pose,
        centers_by_path,
        metadata_by_name,
        base_calibration,
        params_c0,
        params_c1,
        board_config,
        mask_inset_mm,
        app.measurement,
    )
    frame_provenance_check = _verify_ground5a_frame_counts(ground5a_summary, all_frames)
    validation_frames = [frame for frame in all_frames if frame.split == "validation"]
    data = _frame_data(validation_frames, factory, ground1_s["origin_xy"], ground1_s["direction_xy"])
    parameters = _fit_session_parameters(validation_frames, data, factory)
    abcd_rows = _build_metric_rows(validation_frames, data, parameters)
    coordinate_rows, coordinate_selection = _session_coordinate_comparison(abcd_rows)
    classification, diagnostics, production_chain = _classification(abcd_rows, coordinate_selection)

    parameter_rows = []
    for pose in HELDOUT_POSES:
        for chain in ("C", "D-v", "D-S"):
            parameter = parameters[(pose, chain)]
            parameter_rows.append(
                {
                    "pose_id": pose,
                    "chain": chain,
                    "coordinate": parameter["coordinate"],
                    "a": parameter["a"],
                    "b": parameter["b"],
                    "fit_point_count": parameter["fit_point_count"],
                    "fit_frame_count": parameter["fit_frame_count"],
                    "fit_scope": parameter["fit_scope"],
                    "factory_applied": parameter["factory_applied"],
                    "fit_source": parameter["fit_source"],
                }
            )

    output_files = [
        "ground5b_report.md",
        "abcd_comparison.csv",
        "session_linear_parameters.csv",
        "session_coordinate_comparison.csv",
        "abcd_rmse_p95.png",
        "session_coordinate_comparison.png",
        "c_minus_d_residual_delta.png",
        "ground5b_summary.json",
        "cache_provenance.json",
    ]
    g5a._write_csv(output_dir / "abcd_comparison.csv", abcd_rows, list(abcd_rows[0]))
    g5a._write_csv(output_dir / "session_linear_parameters.csv", parameter_rows, list(parameter_rows[0]))
    g5a._write_csv(output_dir / "session_coordinate_comparison.csv", coordinate_rows, list(coordinate_rows[0]))
    _plot_abcd(output_dir / "abcd_rmse_p95.png", abcd_rows, coordinate_selection)
    _plot_coordinate_comparison(output_dir / "session_coordinate_comparison.png", abcd_rows)
    _plot_c_minus_d(
        output_dir / "c_minus_d_residual_delta.png",
        validation_frames,
        data,
        parameters,
        coordinate_selection["chosen_chain"],
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_local": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "dataset_id": dataset_document.get("dataset_id"),
            "manifest_path": manifest_path,
            "frames_csv_path": frames_csv_path,
            "fit_dir": fit_dir,
            "validation_dir": validation_dir,
            "fit_pose_ids": list(FIT_POSES),
            "heldout_pose_ids": list(HELDOUT_POSES),
        },
        "protocol": {
            "ground5a_output": ground5a_output,
            "cache_source_path": cache.get("cache_source_path"),
            "cache_source_sha256": cache.get("cache_source_sha256"),
            "cache_read_only_reuse": True,
            "one_steger_per_laser_tiff": True,
            "steger_rerun": False,
            "same_centers_to_frozen_c0_and_c1": True,
            "pnp_reused": True,
            "physical_board_mask_reused": True,
            "mask_residual_used": False,
            "c0_refit": False,
            "c1_refit": False,
            "h1_applied": False,
            "session_ground_reference_applied": False,
            "factory_refit": False,
            "factory_selection_data_split": "fit_001_005_only",
            "formal_validation_after_factory_freeze": True,
            **frame_provenance_check,
        },
        "ground1_s": ground1_s,
        "factory_profile": {
            "source_path": ground5a_output / "factory_profile_candidate.json",
            "source_sha256": g5a._sha256_file(ground5a_output / "factory_profile_candidate.json"),
            "coordinate": factory_candidate["coordinate"],
            "domain_min": factory.domain_min,
            "domain_max": factory.domain_max,
            "degree": factory.degree,
            "interior_knot_count": factory.interior_knot_count,
            "fit_rmse_mm": factory.fit_rmse_mm,
            "cv_rmse_mm": factory.cv_rmse_mm,
            "refit": False,
        },
        "classification": classification,
        "production_ground_chain": production_chain,
        "coordinate_selection": coordinate_selection,
        "diagnostics": diagnostics,
        "session_parameters": parameter_rows,
        "abcd_comparison": abcd_rows,
        "session_coordinate_comparison": coordinate_rows,
        "output_files": output_files,
        "frame_counts": {
            "all_processed": len(all_frames),
            "validation_processed": len(validation_frames),
            "board_mask_points_min": int(min(frame.c1_selected_count for frame in all_frames)),
            "board_mask_source_values": sorted({frame.mask_metadata.get("source") for frame in all_frames}),
        },
        "artifact_provenance": {
            "reused_artifacts": [
                "Ground-5A protocol-compatible Steger centers for 35 TIFFs",
                "Ground-5A PnP/mask/C0/C1 computation path",
                "Ground-1 frozen physical_S origin/direction",
                "Ground-5A frozen Factory full_v candidate",
            ],
            "newly_computed": [
                "held-out D-v and D-S session-linear fits",
                "held-out A/B/C/D metrics and minimality deltas",
                "Ground-5B plots and report",
            ],
            "not_recomputed": [
                "Steger extraction",
                "Factory Profile fit or model selection",
                "C0/C1 parameters",
                "ground points/mask policy",
            ],
        },
    }
    (output_dir / "cache_provenance.json").write_text(
        json.dumps(g5a._json_ready({
            "cache_source_path": cache.get("cache_source_path"),
            "cache_source_sha256": cache.get("cache_source_sha256"),
            "one_steger_per_frame": cache.get("one_steger_per_frame"),
            "protocol_key": cache.get("protocol_key"),
            "frame_count": len(cache.get("frames", [])),
            "read_only_reuse": True,
            "steger_rerun": False,
        }), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ground5b_summary.json").write_text(
        json.dumps(g5a._json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "ground5b_report.md", g5a._json_ready(summary), output_files)
    print(f"output_dir={output_dir}")
    for key, value in classification.items():
        print(f"{key}={value}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ground5a-output", type=Path, default=DEFAULT_GROUND5A_OUTPUT)
    parser.add_argument("--ground1-summary", type=Path, default=DEFAULT_GROUND1_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    _run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
