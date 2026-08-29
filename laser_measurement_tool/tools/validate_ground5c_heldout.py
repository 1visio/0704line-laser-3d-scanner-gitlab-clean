"""Strict held-out validation for the Ground-5C A-2 frozen Session Linear.

The B chain in this script is deliberately a pure read of the A-2 frozen
``physical_S`` coordinate, parameters, valid domain and bin edges.  It never
fits or changes them.  The only fitted line is C_oracle, which is a diagnostic
upper bound fitted independently inside each held-out pose from equal-weight
pose/bin medians.

Validation support is fixed before any metric is evaluated:

* the A-2 frozen valid-domain bins;
* board-mask-selected points reconstructed by Frozen C0+C1;
* a bin is supported for a pose when at least ceil(0.8 * 5) frames have a
  finite raw-Z bin median.

A, B and C_oracle then use exactly the same supported bins and exactly the
same raw points inside those bins.  No extrapolation, clamping, Factory
Profile, H1 or residual-based point/bin deletion is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from tools import fit_ground5a_factory_profile as g5a
from tools import fit_ground5b_session_linear_minimality as g5b
from tools import fit_ground5c_frozen_session_linear as g5c


HELDOUT_POSES = ("006", "007")
COORDINATE = "physical_S"
MIN_FRAME_FRACTION = 0.8
REQUIRED_FRAME_COUNT = int(math.ceil(MIN_FRAME_FRACTION * 5))

# These criteria are declared before the held-out result is read and never
# adjusted after metric generation.
PASS_B_RMSE_MM = 0.08
PASS_RMSE_IMPROVEMENT_FRACTION = 0.50
PASS_ORACLE_RMSE_GAP_MM = 0.03
PASS_SUPPORT_COVERAGE = 0.80
CLEAR_RMSE_IMPROVEMENT_FRACTION = 0.20
CLEAR_P95_IMPROVEMENT_FRACTION = 0.10

DEFAULT_VALIDATION_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\chessboard_0821\validation"
)
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_GROUND5A_OUTPUT = TOOL_ROOT.parent / "outputs" / "ground5a_factory_profile_0821"
DEFAULT_FROZEN_JSON = (
    TOOL_ROOT.parent
    / "outputs"
    / "ground5c_frozen_session_linear_0821"
    / "frozen_session_linear.json"
)
DEFAULT_GROUND1_SUMMARY = (
    TOOL_ROOT.parent
    / "reports"
    / "experiments"
    / "daheng_0811"
    / "ground_reference_20frames"
    / "ground_reference_summary.json"
)
DEFAULT_OUTPUT = TOOL_ROOT.parent / "outputs" / "ground5c_heldout_validation_0821"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metric_values(values: np.ndarray) -> dict[str, float]:
    return g5c._metric_values(np.asarray(values, dtype=np.float64))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    g5a._write_csv(path, rows, fields)


def _load_frozen_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    sha256 = _sha256_bytes(raw)
    document = json.loads(raw.decode("utf-8"))
    if document.get("coordinate") != COORDINATE:
        raise RuntimeError(f"A-2 frozen coordinate is not {COORDINATE}: {path}")
    if document.get("coordinate_units") != "mm":
        raise RuntimeError("A-2 physical_S coordinate is not recorded in mm")
    if list(document.get("fit_pose_ids", [])) != ["001", "002", "003", "004", "005"]:
        raise RuntimeError("A-2 frozen fit pose IDs are not exactly 001-005")
    physical_s = document.get("physical_S")
    if not isinstance(physical_s, dict):
        raise RuntimeError("A-2 frozen JSON has no physical_S definition")
    if physical_s.get("formula") != "S=(XY-origin_xy) dot direction_xy":
        raise RuntimeError("A-2 physical_S formula does not match Ground-1")
    if physical_s.get("per_frame_redefinition") is not False:
        raise RuntimeError("A-2 physical_S permits frame-local redefinition")
    edges = np.asarray(document.get("bin_edges"), dtype=np.float64)
    domain = np.asarray(document.get("valid_domain"), dtype=np.float64)
    origin = np.asarray(physical_s.get("origin_xy_mm"), dtype=np.float64)
    direction = np.asarray(physical_s.get("direction_xy"), dtype=np.float64)
    if edges.shape != (41,) or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise RuntimeError("A-2 frozen bin_edges are not 40 strictly increasing bins")
    if domain.shape != (2,) or not np.all(np.isfinite(domain)) or not domain[0] < domain[1]:
        raise RuntimeError("A-2 frozen valid_domain is invalid")
    if origin.shape != (2,) or direction.shape != (2,) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
        raise RuntimeError("A-2 frozen physical_S axis is invalid")
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-6):
        raise RuntimeError("A-2 frozen physical_S direction is not unit length")
    formal_indices = np.asarray(document.get("formal_bin_indices"), dtype=np.int64)
    if len(formal_indices) == 0 or np.any(formal_indices < 0) or np.any(formal_indices >= 40):
        raise RuntimeError("A-2 formal bin indices are invalid")
    formal_mask = np.zeros(40, dtype=bool)
    formal_mask[formal_indices] = True
    domain_mask = (edges[:-1] >= domain[0] - 1.0e-9) & (edges[1:] <= domain[1] + 1.0e-9)
    evaluation_domain_mask = formal_mask & domain_mask
    if not np.any(evaluation_domain_mask):
        raise RuntimeError("A-2 frozen valid domain has no formal evaluation bin")
    frozen = {
        "coordinate": COORDINATE,
        "coordinate_units": "mm",
        "a_session": float(document["a_session"]),
        "b_session": float(document["b_session"]),
        "valid_domain": domain,
        "bin_edges": edges,
        "bin_centers": (edges[:-1] + edges[1:]) / 2.0,
        "origin_xy": origin,
        "direction_xy": direction,
        "formal_indices": formal_indices,
        "evaluation_domain_mask": evaluation_domain_mask,
    }
    if not math.isfinite(frozen["a_session"]) or not math.isfinite(frozen["b_session"]):
        raise RuntimeError("A-2 frozen Session Linear parameters are not finite")
    return frozen, {
        "path": path,
        "sha256_initial": sha256,
        "fit_pose_ids": list(document["fit_pose_ids"]),
        "coordinate": document["coordinate"],
        "a_session": document["a_session"],
        "b_session": document["b_session"],
        "valid_domain": document["valid_domain"],
        "bin_edges": document["bin_edges"],
        "physical_S": physical_s,
        "formal_bin_indices": document["formal_bin_indices"],
    }


def _load_validation_cached_centers_only(
    records: list[g5a.PoseRecord],
    cache_output: Path,
    app: Any,
    config_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only validation cache entries; never fall back to Steger."""

    cache_path = cache_output / "steger_geometry_cache.json"
    if not cache_path.is_file():
        raise RuntimeError(f"Ground-5A Steger cache is missing: {cache_path}")
    cache = _read_json(cache_path)
    if cache.get("one_steger_per_frame") is not True:
        raise RuntimeError("Ground-5A cache does not certify one Steger run per frame")
    if cache.get("protocol_key") != g5a._cache_key(app, config_path):
        raise RuntimeError("Ground-5A cache protocol key does not match current config")

    requested: list[str] = []
    expected_pose: dict[str, str] = {}
    for record in records:
        if record.split != "validation" or record.pose_id not in HELDOUT_POSES:
            raise RuntimeError(f"non-validation record passed to held-out cache loader: {record}")
        for path in record.laser_paths:
            source = str(path.resolve())
            requested.append(source)
            expected_pose[source] = record.pose_id

    cached_by_path: dict[str, dict[str, Any]] = {}
    for item in cache.get("frames", []):
        if item.get("split") != "validation" or str(item.get("pose_id")) not in HELDOUT_POSES:
            continue
        source = str(Path(str(item.get("source_path"))).resolve())
        cached_by_path[source] = item
    if set(requested) != set(cached_by_path):
        raise RuntimeError(
            "validation cache entry set does not exactly match pose006/007 laser records"
        )
    if len(requested) != 10:
        raise RuntimeError(f"expected 10 validation laser records, got {len(requested)}")

    centers_by_path: dict[str, np.ndarray] = {}
    selected_entries: list[dict[str, Any]] = []
    for source_path in requested:
        source = Path(source_path)
        item = cached_by_path[source_path]
        if str(item.get("pose_id")) != expected_pose[source_path]:
            raise RuntimeError(f"validation cache pose mismatch: {source}")
        source_sha = g5a._sha256_file(source)
        if item.get("source_sha256") != source_sha:
            raise RuntimeError(f"validation source SHA mismatch: {source}")
        if int(item.get("steger_run_count", 0)) != 1:
            raise RuntimeError(f"validation cache entry is not one Steger run: {source}")
        center_path = Path(str(item["centers_path"]))
        centers = np.asarray(np.load(center_path), dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
            raise RuntimeError(f"invalid validation centers: {center_path}")
        centers_by_path[source_path] = np.ascontiguousarray(centers)
        selected_entries.append(
            {
                "pose_id": expected_pose[source_path],
                "source_path": source_path,
                "source_sha256": source_sha,
                "centers_path": str(center_path.resolve()),
                "center_count": int(len(centers)),
                "steger_run_count": 1,
            }
        )
    return centers_by_path, {
        "cache_path": cache_path,
        "cache_sha256": g5a._sha256_file(cache_path),
        "protocol_key": cache["protocol_key"],
        "selected_validation_entry_count": len(selected_entries),
        "selected_validation_entries": selected_entries,
        "read_only": True,
        "steger_rerun": False,
        "fit_entries_used": False,
    }


def _build_validation_support(
    data: dict[str, dict[str, Any]],
    pose: str,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    edges = frozen["bin_edges"]
    domain_mask = frozen["evaluation_domain_mask"]
    pose_items = sorted(
        [item for item in data.values() if item["pose_id"] == pose],
        key=lambda item: str(item["frame_id"]),
    )
    if len(pose_items) != 5:
        raise RuntimeError(f"pose{pose} does not have exactly five processed frames")
    frame_values: dict[str, np.ndarray] = {}
    frame_counts: dict[str, np.ndarray] = {}
    frame_bin_indices: dict[str, np.ndarray] = {}
    frame_domain_point_count: dict[str, int] = {}
    frame_eval_point_count: dict[str, int] = {}
    values = np.full((len(pose_items), len(edges) - 1), np.nan, dtype=np.float64)
    counts = np.zeros_like(values, dtype=np.int64)
    for frame_index, item in enumerate(pose_items):
        profile, point_counts = g5c._bin_frame_medians(
            item[COORDINATE], item["z"], edges
        )
        profile[~domain_mask] = np.nan
        point_counts[~domain_mask] = 0
        values[frame_index] = profile
        counts[frame_index] = point_counts
        frame_id = str(item["frame_id"])
        frame_values[frame_id] = profile
        frame_counts[frame_id] = point_counts
        frame_bin_indices[frame_id] = _bin_indices(item[COORDINATE], edges)
        frame_domain_point_count[frame_id] = _count_points_in_bins(
            item[COORDINATE], edges, domain_mask
        )
        frame_eval_point_count[frame_id] = 0
    frame_count_valid = np.sum(np.isfinite(values), axis=0).astype(np.int64)
    supported = frame_count_valid >= REQUIRED_FRAME_COUNT
    pose_profile = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    for bin_index in np.flatnonzero(supported):
        pose_profile[bin_index] = float(np.median(values[:, bin_index][np.isfinite(values[:, bin_index])]))
    supported &= np.isfinite(pose_profile)
    return {
        "pose_id": pose,
        "frame_values": frame_values,
        "frame_counts": frame_counts,
        "frame_bin_indices": frame_bin_indices,
        "frame_domain_point_count": frame_domain_point_count,
        "frame_eval_point_count": frame_eval_point_count,
        "frame_count_valid": frame_count_valid,
        "pose_profile": pose_profile,
        "supported_mask": supported,
        "domain_mask": domain_mask,
        "domain_bin_count": int(np.count_nonzero(domain_mask)),
        "supported_bin_count": int(np.count_nonzero(supported)),
        "valid_support_coverage": float(np.count_nonzero(supported) / np.count_nonzero(domain_mask)),
        "required_frame_count": REQUIRED_FRAME_COUNT,
        "frame_ids": [str(item["frame_id"]) for item in pose_items],
    }


def _bin_indices(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, np.asarray(x, dtype=np.float64), side="right") - 1
    indices[np.asarray(x) == edges[-1]] = len(edges) - 2
    return indices.astype(np.int64, copy=False)


def _count_points_in_bins(x: np.ndarray, edges: np.ndarray, bin_mask: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float64)
    indices = _bin_indices(x, edges)
    in_range = np.isfinite(x) & (x >= edges[0]) & (x <= edges[-1])
    safe = np.clip(indices, 0, len(bin_mask) - 1)
    return int(np.count_nonzero(in_range & bin_mask[safe]))


def _fit_oracle_line(support: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    mask = support["supported_mask"]
    x = frozen["bin_centers"][mask]
    z = support["pose_profile"][mask]
    if len(x) < 2 or float(np.ptp(x)) <= np.finfo(np.float64).eps:
        raise RuntimeError(f"pose{support['pose_id']} has insufficient oracle support")
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), z, rcond=None)[0]
    residual = z - (a * x + b)
    return {
        "pose_id": support["pose_id"],
        "a_oracle": float(a),
        "b_oracle": float(b),
        "fit_bin_count": int(len(x)),
        "fit_support_coverage": support["valid_support_coverage"],
        "fit_bias_mm": _metric_values(residual)["bias_mm"],
        "fit_rmse_mm": _metric_values(residual)["rmse_mm"],
        "fit_p95_abs_mm": _metric_values(residual)["p95_abs_mm"],
        "fit_max_abs_mm": _metric_values(residual)["max_abs_mm"],
        "fit_peak_to_peak_mm": _metric_values(residual)["peak_to_peak_mm"],
        "fit_source": "heldout_pose_supported_bin_medians_equal_weight",
    }


def _equal_bin_rows(
    support: dict[str, Any],
    oracle: dict[str, Any],
    frozen: dict[str, Any],
) -> list[dict[str, Any]]:
    mask = support["supported_mask"]
    x = frozen["bin_centers"]
    z = support["pose_profile"]
    a = frozen["a_session"]
    b = frozen["b_session"]
    oracle_a = oracle["a_oracle"]
    oracle_b = oracle["b_oracle"]
    residuals = {
        "A_PnP_only": z,
        "B_frozen_session_linear": z - (a * x + b),
        "C_oracle": z - (oracle_a * x + oracle_b),
    }
    rows: list[dict[str, Any]] = []
    for chain, residual in residuals.items():
        metrics = _metric_values(residual[mask])
        rows.append(
            {
                "pose_id": support["pose_id"],
                "chain": chain,
                "coordinate": COORDINATE,
                "evaluation_level": "equal_bin",
                "domain_bin_count": support["domain_bin_count"],
                "supported_bin_count": support["supported_bin_count"],
                "valid_support_coverage": support["valid_support_coverage"],
                "required_frame_count": support["required_frame_count"],
                "frame_count": len(support["frame_ids"]),
                "bias_mm": metrics["bias_mm"],
                "rmse_mm": metrics["rmse_mm"],
                "p95_abs_mm": metrics["p95_abs_mm"],
                "max_abs_mm": metrics["max_abs_mm"],
                "peak_to_peak_mm": metrics["peak_to_peak_mm"],
                "same_evaluation_support": True,
                "raw_point_pooling": False,
                "fit_source": "raw_Z_frame_median_then_pose_median",
            }
        )
    return rows


def _raw_point_rows(
    pose: str,
    data: dict[str, dict[str, Any]],
    support: dict[str, Any],
    oracle: dict[str, Any],
    frozen: dict[str, Any],
) -> list[dict[str, Any]]:
    edges = frozen["bin_edges"]
    domain_mask = support["domain_mask"]
    eval_mask = support["supported_mask"]
    a = frozen["a_session"]
    b = frozen["b_session"]
    oracle_a = oracle["a_oracle"]
    oracle_b = oracle["b_oracle"]
    chain_values: dict[str, list[np.ndarray]] = {
        "A_PnP_only": [],
        "B_frozen_session_linear": [],
        "C_oracle": [],
    }
    total_board_points = 0
    domain_points = 0
    evaluation_points = 0
    frames = [item for item in data.values() if item["pose_id"] == pose]
    for item in frames:
        s = np.asarray(item[COORDINATE], dtype=np.float64)
        z = np.asarray(item["z"], dtype=np.float64)
        indices = _bin_indices(s, edges)
        in_range = np.isfinite(s) & np.isfinite(z) & (s >= edges[0]) & (s <= edges[-1])
        safe = np.clip(indices, 0, len(edges) - 2)
        domain_points += int(np.count_nonzero(in_range & domain_mask[safe]))
        selected = in_range & eval_mask[safe]
        total_board_points += int(len(z))
        evaluation_points += int(np.count_nonzero(selected))
        for chain, residual in (
            ("A_PnP_only", z),
            ("B_frozen_session_linear", z - (a * s + b)),
            ("C_oracle", z - (oracle_a * s + oracle_b)),
        ):
            chain_values[chain].append(np.asarray(residual[selected], dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for chain, values in chain_values.items():
        residual = np.concatenate(values) if values else np.array([], dtype=np.float64)
        metrics = _metric_values(residual)
        rows.append(
            {
                "pose_id": pose,
                "chain": chain,
                "coordinate": COORDINATE,
                "evaluation_level": "same_support_raw_points",
                "board_mask_point_count": total_board_points,
                "frozen_domain_point_count": domain_points,
                "evaluation_point_count": evaluation_points,
                "evaluation_point_coverage_of_domain": float(evaluation_points / domain_points) if domain_points else math.nan,
                "evaluation_point_coverage_of_board_mask": float(evaluation_points / total_board_points) if total_board_points else math.nan,
                "supported_bin_count": support["supported_bin_count"],
                "valid_support_coverage": support["valid_support_coverage"],
                "bias_mm": metrics["bias_mm"],
                "rmse_mm": metrics["rmse_mm"],
                "p95_abs_mm": metrics["p95_abs_mm"],
                "max_abs_mm": metrics["max_abs_mm"],
                "peak_to_peak_mm": metrics["peak_to_peak_mm"],
                "same_evaluation_support": True,
                "raw_point_pooling": False,
                "fit_source": "evaluation_only; no raw_point_fit",
            }
        )
    return rows


def _comparison_rows(
    equal_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    supports: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> list[dict[str, Any]]:
    by_pose_chain = {(row["pose_id"], row["chain"]): row for row in equal_rows}
    raw_by_pose_chain = {(row["pose_id"], row["chain"]): row for row in raw_rows}
    oracle_by_pose = {row["pose_id"]: row for row in oracle_rows}
    rows: list[dict[str, Any]] = []
    for pose in HELDOUT_POSES:
        a = by_pose_chain[(pose, "A_PnP_only")]
        b = by_pose_chain[(pose, "B_frozen_session_linear")]
        c = by_pose_chain[(pose, "C_oracle")]
        a_raw = raw_by_pose_chain[(pose, "A_PnP_only")]
        b_raw = raw_by_pose_chain[(pose, "B_frozen_session_linear")]
        c_raw = raw_by_pose_chain[(pose, "C_oracle")]
        row = {
            "aggregation_level": "pose",
            "pose_id": pose,
            "coordinate": COORDINATE,
            "valid_domain_left": frozen["valid_domain"][0],
            "valid_domain_right": frozen["valid_domain"][1],
            "domain_bin_count": a["domain_bin_count"],
            "supported_bin_count": b["supported_bin_count"],
            "valid_support_coverage": b["valid_support_coverage"],
            "oracle_a": oracle_by_pose[pose]["a_oracle"],
            "oracle_b": oracle_by_pose[pose]["b_oracle"],
        }
        for prefix, item in (("A_equal", a), ("B_equal", b), ("C_equal", c), ("A_raw", a_raw), ("B_raw", b_raw), ("C_raw", c_raw)):
            row[f"{prefix}_bias_mm"] = item["bias_mm"]
            row[f"{prefix}_rmse_mm"] = item["rmse_mm"]
            row[f"{prefix}_p95_abs_mm"] = item["p95_abs_mm"]
            row[f"{prefix}_max_abs_mm"] = item["max_abs_mm"]
            row[f"{prefix}_peak_to_peak_mm"] = item["peak_to_peak_mm"]
        row.update(
            {
                "B_vs_A_rmse_gain_mm": a["rmse_mm"] - b["rmse_mm"],
                "B_vs_A_rmse_improvement_fraction": (a["rmse_mm"] - b["rmse_mm"]) / a["rmse_mm"] if a["rmse_mm"] else math.nan,
                "B_vs_A_p95_gain_mm": a["p95_abs_mm"] - b["p95_abs_mm"],
                "B_vs_A_p95_improvement_fraction": (a["p95_abs_mm"] - b["p95_abs_mm"]) / a["p95_abs_mm"] if a["p95_abs_mm"] else math.nan,
                "B_minus_C_oracle_rmse_gap_mm": b["rmse_mm"] - c["rmse_mm"],
                "B_minus_C_oracle_p95_gap_mm": b["p95_abs_mm"] - c["p95_abs_mm"],
                "raw_B_vs_A_rmse_improvement_fraction": (a_raw["rmse_mm"] - b_raw["rmse_mm"]) / a_raw["rmse_mm"] if a_raw["rmse_mm"] else math.nan,
            }
        )
        rows.append(row)
    return rows


def _classify(rows: list[dict[str, Any]]) -> dict[str, str]:
    strict = all(
        row["B_equal_rmse_mm"] <= PASS_B_RMSE_MM
        and row["B_vs_A_rmse_improvement_fraction"] >= PASS_RMSE_IMPROVEMENT_FRACTION
        and row["B_minus_C_oracle_rmse_gap_mm"] <= PASS_ORACLE_RMSE_GAP_MM
        and row["valid_support_coverage"] >= PASS_SUPPORT_COVERAGE
        for row in rows
    )
    clear = all(
        row["B_vs_A_rmse_improvement_fraction"] >= CLEAR_RMSE_IMPROVEMENT_FRACTION
        and row["B_vs_A_p95_improvement_fraction"] >= CLEAR_P95_IMPROVEMENT_FRACTION
        for row in rows
    )
    if strict:
        status = "PASS"
        refit = "NO"
    elif clear:
        status = "PARTIAL"
        refit = "UNCERTAIN"
    else:
        status = "FAIL"
        refit = "YES"
    coverage = "YES" if any(row["valid_support_coverage"] < PASS_SUPPORT_COVERAGE for row in rows) else "NO"
    core = "YES" if status == "PASS" else "PARTIAL" if status == "PARTIAL" else "NO"
    return {
        "FROZEN_SESSION_LINEAR": status,
        "PER_POSE_REFIT_NEEDED": refit,
        "RECOMMENDED_COORDINATE": COORDINATE,
        "MORE_COVERAGE_REQUIRED": coverage,
        "CORE_ANSWER": core,
    }


def _plot_residuals(
    equal_rows: list[dict[str, Any]],
    supports: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
    output: Path,
) -> None:
    by_key = {(row["pose_id"], row["chain"]): row for row in equal_rows}
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    colors = {"A_PnP_only": "#b45309", "B_frozen_session_linear": "#1d4ed8", "C_oracle": "#15803d"}
    labels = {"A_PnP_only": "A: PnP only", "B_frozen_session_linear": "B: frozen physical_S line", "C_oracle": "C_oracle: held-out refit"}
    for axis, pose in zip(axes, HELDOUT_POSES, strict=True):
        support = supports[pose]
        x = frozen["bin_centers"][support["supported_mask"]]
        for chain in colors:
            row = by_key[(pose, chain)]
            # Reconstruct the equal-bin residual from the same frozen support.
            z = support["pose_profile"][support["supported_mask"]]
            if chain == "A_PnP_only":
                residual = z
            elif chain == "B_frozen_session_linear":
                residual = z - (frozen["a_session"] * x + frozen["b_session"])
            else:
                oracle_a = float(row.get("oracle_a", math.nan))
                # oracle parameters are attached below by the caller through the row only when present;
                # use the per-pose values stored on the support object otherwise.
                oracle_a = support["oracle_a"]
                oracle_b = support["oracle_b"]
                residual = z - (oracle_a * x + oracle_b)
            axis.plot(x, residual, marker="o", markersize=3.5, linewidth=1.0, color=colors[chain], label=labels[chain])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(frozen["valid_domain"][0], color="#6b7280", linestyle="--", linewidth=0.8)
        axis.axvline(frozen["valid_domain"][1], color="#6b7280", linestyle="--", linewidth=0.8)
        axis.set_title(f"pose{pose}: same supported bins, A/B/C_oracle residual vs physical_S")
        axis.set_ylabel("Zg residual (mm)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("physical_S (mm)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _build_report(
    output: Path,
    frozen_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    comparisons: list[dict[str, Any]],
    equal_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    classifications: dict[str, str],
) -> None:
    lines = [
        "# Ground-5C A-3｜Strict Held-out Validation",
        "",
        "## 最终判定",
        "",
        f"- `FROZEN_SESSION_LINEAR = {classifications['FROZEN_SESSION_LINEAR']}`",
        f"- `PER_POSE_REFIT_NEEDED = {classifications['PER_POSE_REFIT_NEEDED']}`",
        f"- `RECOMMENDED_COORDINATE = {classifications['RECOMMENDED_COORDINATE']}`",
        f"- `MORE_COVERAGE_REQUIRED = {classifications['MORE_COVERAGE_REQUIRED']}`",
        "",
        f"核心问题：`Can the single physical_S Session Linear frozen from poses001–005 predict poses006–007 without re-grounding? {classifications['CORE_ANSWER']}`",
        "",
        "B 链没有重新拟合：直接读取 A-2 frozen JSON 的 physical_S、origin/direction、a_session、b_session、valid_domain 和 bin_edges。C_oracle 只作为 held-out 上限诊断，不写回任何冻结参数。",
        "",
        "## Frozen input and provenance",
        "",
        f"- frozen JSON: `{frozen_meta['path']}`",
        f"- frozen JSON SHA-256 at startup: `{frozen_meta['sha256_initial']}`",
        f"- coordinate: `{frozen_meta['coordinate']}`",
        f"- `a_session={frozen_meta['a_session']}`, `b_session={frozen_meta['b_session']}`",
        f"- valid domain: `{frozen_meta['valid_domain']}`",
        f"- bin edge count: `{len(frozen_meta['bin_edges'])}` (40 bins)",
        f"- validation cache entries used: `{cache_meta['selected_validation_entry_count']}`; `steger_rerun=false`",
        "",
        "validation 只加载 pose006/007；没有加载 fit records，没有读取 Ground5A/5B held-out 指标。C0/C1、PnP、physical-board mask 均复用既有链。",
        "",
        "## Predeclared engineering criteria",
        "",
        "判据在读取 held-out 结果前固定，运行后未调整：",
        f"- strict PASS per pose: B RMSE <= `{PASS_B_RMSE_MM:.3f} mm`, B-vs-A RMSE improvement >= `{PASS_RMSE_IMPROVEMENT_FRACTION:.0%}`, B-C_oracle RMSE gap <= `{PASS_ORACLE_RMSE_GAP_MM:.3f} mm`, coverage >= `{PASS_SUPPORT_COVERAGE:.0%}`。",
        f"- PARTIAL 的“明显优于 A”固定为 RMSE improvement >= `{CLEAR_RMSE_IMPROVEMENT_FRACTION:.0%}` 且 P95 improvement >= `{CLEAR_P95_IMPROVEMENT_FRACTION:.0%}`，两个 pose 都满足但至少一个未达 strict PASS。",
        "",
        "## Equal-bin metrics",
        "",
        "主指标是在相同 frozen-domain / board-mask / supported-bin 上计算；每个 frame/bin 取 raw Zg median，再每个 pose/bin 取 frame median 的 median。C_oracle 的拟合也是 supported pose/bin 等权，不使用 pooled raw points。",
        "",
        "| pose | A RMSE | B RMSE | C_oracle RMSE | B-A RMSE improvement | B-A P95 improvement | B-C RMSE gap | coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['pose_id']} | {row['A_equal_rmse_mm']:.6g} | {row['B_equal_rmse_mm']:.6g} | {row['C_equal_rmse_mm']:.6g} | {row['B_vs_A_rmse_improvement_fraction']:.2%} | {row['B_vs_A_p95_improvement_fraction']:.2%} | {row['B_minus_C_oracle_rmse_gap_mm']:.6g} | {row['valid_support_coverage']:.2%} |"
        )
    lines.extend(
        [
            "",
            "完整 Bias/RMSE/P95/Max/P2P 见 `heldout_equal_bin_metrics.csv`；同一 support 的 raw-point 工程诊断见 `heldout_raw_point_metrics.csv`。",
            "",
            "## Oracle parameters",
            "",
            "| pose | a_oracle (mm/mm) | b_oracle (mm) | fit bins | oracle fit RMSE (mm) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['pose_id']} | {row['oracle_a']:.8g} | {row['oracle_b']:.8g} | {row['supported_bin_count']} | {next(item['rmse_mm'] for item in equal_rows if item['pose_id']==row['pose_id'] and item['chain']=='C_oracle'):.8g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A 是 raw PnP-ground Zg；B 是 A-2 冻结的单一 physical_S Session Linear；C_oracle 允许每个 held-out pose 自己拟合，仅用于判断 session-to-session 参数迁移损失。B/C 差距不能反向修改 A-2。",
            "",
            "输出：`heldout_comparison.csv`、`heldout_equal_bin_metrics.csv`、`heldout_raw_point_metrics.csv`、`heldout_oracle_parameters.csv`、`heldout_residual_ABC.png`、`validation_provenance.json`。",
        ]
    )
    (output / "ground5c_heldout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.frozen_json.resolve()
    frozen, frozen_meta = _load_frozen_snapshot(frozen_path)

    validation_dir = args.validation_dir.resolve()
    config_path = args.config.resolve()
    ground5a_output = args.ground5a_output.resolve()
    ground1_path = args.ground1_summary.resolve()
    records = g5a._discover_pose_records(validation_dir, "validation", HELDOUT_POSES)

    app = g5a.load_app_config(config_path)
    if app.reconstruction.image_roi_polygon is not None:
        raise RuntimeError("Ground-5C A-3 requires reconstruction.image_roi_polygon=null")
    if not app.reconstruction.enable_laser_ray_correction:
        raise RuntimeError("Ground-5C A-3 requires Frozen C1 to be enabled")
    if app.calibration.manifest is None:
        raise RuntimeError("Ground-5C A-3 requires calibration.manifest")
    package = g5a.load_calibration_package(app.calibration.manifest)
    base_calibration = dict(package.calibration)
    params_c0 = g5a.replace(app.reconstruction, enable_laser_ray_correction=False)
    params_c1 = app.reconstruction
    board_config = app.session_ground_calibration.board_config()
    mask_inset_mm = float(app.session_ground_calibration.sanity.mask_inset_mm)
    if mask_inset_mm != float(app.session_ground_calibration.ground_reference.mask_inset_mm):
        raise RuntimeError("Session sanity and ground-reference mask inset differ")

    centers_by_path, cache_meta = _load_validation_cached_centers_only(
        records, ground5a_output, app, config_path
    )
    pnp_by_pose = g5a._run_pnp(records, base_calibration, board_config)
    frames = g5a._process_frames(
        records,
        pnp_by_pose,
        centers_by_path,
        {},
        base_calibration,
        params_c0,
        params_c1,
        board_config,
        mask_inset_mm,
        app.measurement,
    )
    if len(frames) != 10 or {frame.pose_id for frame in frames} != set(HELDOUT_POSES):
        raise RuntimeError("A-3 did not produce exactly five frames for pose006 and pose007")

    ground1 = g5b._load_ground1_s(ground1_path)
    if ground1["source_sha256"] != frozen_meta["physical_S"].get("source_sha256"):
        raise RuntimeError("Ground-1 physical_S source SHA differs from A-2 frozen JSON")
    if not np.allclose(ground1["origin_xy"], frozen["origin_xy"], atol=1.0e-9):
        raise RuntimeError("Ground-1 origin differs from A-2 frozen physical_S origin")
    if not np.allclose(ground1["direction_xy"], frozen["direction_xy"], atol=1.0e-9):
        raise RuntimeError("Ground-1 direction differs from A-2 frozen physical_S direction")
    data = g5c._frame_data(frames, frozen["origin_xy"], frozen["direction_xy"])

    supports: dict[str, dict[str, Any]] = {}
    oracle_rows: list[dict[str, Any]] = []
    equal_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for pose in HELDOUT_POSES:
        support = _build_validation_support(data, pose, frozen)
        oracle = _fit_oracle_line(support, frozen)
        support["oracle_a"] = oracle["a_oracle"]
        support["oracle_b"] = oracle["b_oracle"]
        supports[pose] = support
        oracle_rows.append(oracle)
        equal_rows.extend(_equal_bin_rows(support, oracle, frozen))
        raw_rows.extend(_raw_point_rows(pose, data, support, oracle, frozen))

    comparisons = _comparison_rows(equal_rows, raw_rows, oracle_rows, supports, frozen)
    classifications = _classify(comparisons)

    comparison_fields = [
        "aggregation_level", "pose_id", "coordinate", "valid_domain_left", "valid_domain_right", "domain_bin_count", "supported_bin_count",
        "valid_support_coverage", "oracle_a", "oracle_b",
    ]
    for prefix in ("A_equal", "B_equal", "C_equal", "A_raw", "B_raw", "C_raw"):
        comparison_fields.extend([f"{prefix}_{name}" for name in ("bias_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm", "peak_to_peak_mm")])
    comparison_fields.extend(
        [
            "B_vs_A_rmse_gain_mm", "B_vs_A_rmse_improvement_fraction", "B_vs_A_p95_gain_mm",
            "B_vs_A_p95_improvement_fraction", "B_minus_C_oracle_rmse_gap_mm", "B_minus_C_oracle_p95_gap_mm",
            "raw_B_vs_A_rmse_improvement_fraction",
        ]
    )
    _write_csv(output / "heldout_comparison.csv", comparisons, comparison_fields)

    _write_csv(
        output / "heldout_equal_bin_metrics.csv",
        equal_rows,
        [
            "pose_id", "chain", "coordinate", "evaluation_level", "domain_bin_count", "supported_bin_count",
            "valid_support_coverage", "required_frame_count", "frame_count", "bias_mm", "rmse_mm", "p95_abs_mm",
            "max_abs_mm", "peak_to_peak_mm", "same_evaluation_support", "raw_point_pooling", "fit_source",
        ],
    )
    _write_csv(
        output / "heldout_raw_point_metrics.csv",
        raw_rows,
        [
            "pose_id", "chain", "coordinate", "evaluation_level", "board_mask_point_count", "frozen_domain_point_count",
            "evaluation_point_count", "evaluation_point_coverage_of_domain", "evaluation_point_coverage_of_board_mask",
            "supported_bin_count", "valid_support_coverage", "bias_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm",
            "peak_to_peak_mm", "same_evaluation_support", "raw_point_pooling", "fit_source",
        ],
    )
    _write_csv(
        output / "heldout_oracle_parameters.csv",
        oracle_rows,
        [
            "pose_id", "a_oracle", "b_oracle", "fit_bin_count", "fit_support_coverage", "fit_bias_mm",
            "fit_rmse_mm", "fit_p95_abs_mm", "fit_max_abs_mm", "fit_peak_to_peak_mm", "fit_source",
        ],
    )

    _plot_residuals(equal_rows, supports, frozen, output / "heldout_residual_ABC.png")

    frozen_sha_final = _sha256_bytes(frozen_path.read_bytes())
    if frozen_sha_final != frozen_meta["sha256_initial"]:
        raise RuntimeError("A-2 frozen_session_linear.json changed during held-out validation")
    pnp_meta = {
        pose: {
            "split": pnp_by_pose[pose].split,
            "chess_path": pnp_by_pose[pose].chess_path,
            "reprojection_rmse_px": pnp_by_pose[pose].reprojection_rmse_px,
            "detection_method": pnp_by_pose[pose].detection_method,
            "corner_count": len(pnp_by_pose[pose].result.detected_corners),
        }
        for pose in HELDOUT_POSES
    }
    validation_sources = []
    for record in records:
        validation_sources.append(
            {
                "pose_id": record.pose_id,
                "chess_path": record.chess_path,
                "chess_sha256": g5a._sha256_file(record.chess_path),
                "laser_files": [
                    {"path": path, "sha256": g5a._sha256_file(path)}
                    for path in record.laser_paths
                ],
            }
        )
    provenance = {
        "schema_version": 1,
        "status": "strict_heldout_validation",
        "validation_pose_ids": list(HELDOUT_POSES),
        "coordinate": COORDINATE,
        "frozen_json": frozen_meta,
        "frozen_json_sha256_final": frozen_sha_final,
        "frozen_input_unchanged": frozen_sha_final == frozen_meta["sha256_initial"],
        "frozen_parameters_used": {
            "a_session": frozen_meta["a_session"],
            "b_session": frozen_meta["b_session"],
            "valid_domain": frozen_meta["valid_domain"],
            "bin_edges": frozen_meta["bin_edges"],
            "physical_S": frozen_meta["physical_S"],
        },
        "config_path": config_path,
        "config_sha256": g5a._sha256_file(config_path),
        "ground5a_output": ground5a_output,
        "ground1_summary": {
            "path": ground1_path,
            "sha256": ground1["source_sha256"],
            "formula": ground1["formula"],
            "origin_xy": ground1["origin_xy"],
            "direction_xy": ground1["direction_xy"],
        },
        "validation_sources": validation_sources,
        "cache": cache_meta,
        "pnp": pnp_meta,
        "frame_count": len(frames),
        "point_selection": "Ground5A physical-board mask; no new mask selection",
        "frozen_c0_c1": True,
        "h1_used": False,
        "factory_profile_used": False,
        "steger_rerun": False,
        "B_parameter_fit": False,
        "B_parameter_source": "A-2 frozen_session_linear.json",
        "C_oracle_production_parameter": False,
        "extrapolation": False,
        "clamp": False,
        "raw_point_fit": False,
        "evaluation_support": {
            "frozen_valid_domain_only": True,
            "frozen_formal_bins_only": True,
            "minimum_frame_fraction": MIN_FRAME_FRACTION,
            "required_frame_count": REQUIRED_FRAME_COUNT,
            "same_bins_for_A_B_C": True,
            "same_raw_points_for_A_B_C": True,
        },
        "fixed_criteria": {
            "pass_B_rmse_mm": PASS_B_RMSE_MM,
            "pass_rmse_improvement_fraction": PASS_RMSE_IMPROVEMENT_FRACTION,
            "pass_oracle_rmse_gap_mm": PASS_ORACLE_RMSE_GAP_MM,
            "pass_support_coverage": PASS_SUPPORT_COVERAGE,
            "clear_rmse_improvement_fraction": CLEAR_RMSE_IMPROVEMENT_FRACTION,
            "clear_p95_improvement_fraction": CLEAR_P95_IMPROVEMENT_FRACTION,
        },
        "classification": classifications,
    }
    (output / "validation_provenance.json").write_text(
        json.dumps(_json_ready(provenance), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _build_report(output, frozen_meta, cache_meta, comparisons, equal_rows, raw_rows, classifications)
    return classifications


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ground5a-output", type=Path, default=DEFAULT_GROUND5A_OUTPUT)
    parser.add_argument("--frozen-json", type=Path, default=DEFAULT_FROZEN_JSON)
    parser.add_argument("--ground1-summary", type=Path, default=DEFAULT_GROUND1_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    result = _run(_parse_args())
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))
