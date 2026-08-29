"""Ground-5C fit-only Frozen Session Linear audit.

This diagnostic uses only chessboard_0821 fit poses 001--005.  It reuses the
Ground-5A PnP, frozen C0/C1 reconstruction and physical-board selector, but
loads only the 25 fit-frame entries from the existing Steger cache.  It never
discovers the validation directory and never runs Steger as a fallback.

For each coordinate (``full_v`` and Ground-1's frozen ``physical_S``), the
protocol is:

* one fixed 40-bin union support over fit poses 001--005;
* per-frame per-bin Zg median;
* per-pose median over supported frames;
* pose_count >= 2 for formal support and >= 3 for strong support;
* one equal-weight pose-balanced observation per formal bin;
* an ordinary least-squares session line on those bin observations;
* fit-only 4-pose-to-1-pose LOPO prediction.

No Z residual is used to select a point, bin, coordinate, or model.  The
``pooled`` plot and all fitting inputs are bin-balanced observations, never
pooled raw laser points.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from tools import fit_ground5a_factory_profile as g5a
from tools import fit_ground5b_session_linear_minimality as g5b


FIT_POSES = tuple(g5a.FIT_POSES)
COORDINATES = ("full_v", "physical_S")
PROFILE_BIN_COUNT = 40
MIN_FRAME_FRACTION = 0.8
FORMAL_MIN_POSES = 2
STRONG_MIN_POSES = 3
COORDINATE_TIE_TOLERANCE_MM = 1.0e-6
COORDINATE_TIE_TOLERANCE_COVERAGE = 1.0e-6

DEFAULT_FIT_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\chessboard_0821\fit"
)
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_GROUND5A_OUTPUT = TOOL_ROOT.parent / "outputs" / "ground5a_factory_profile_0821"
DEFAULT_GROUND1_SUMMARY = (
    TOOL_ROOT.parent
    / "reports"
    / "experiments"
    / "daheng_0811"
    / "ground_reference_20frames"
    / "ground_reference_summary.json"
)
DEFAULT_OUTPUT = TOOL_ROOT.parent / "outputs" / "ground5c_frozen_session_linear_0821"


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


def _metric_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    g5a._write_csv(path, rows, fields)


def _load_fit_cached_centers_only(
    records: list[g5a.PoseRecord],
    cache_output: Path,
    app: Any,
    config_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read exactly the requested fit cache entries; never run Steger.

    Ground-5A's combined cache also contains validation entries.  They are
    intentionally ignored here.  Only the exact source paths from the fit
    records are hashed and dereferenced.
    """

    cache_path = cache_output / "steger_geometry_cache.json"
    if not cache_path.is_file():
        raise RuntimeError(f"Ground-5A Steger cache is missing: {cache_path}")
    cache = _read_json(cache_path)
    if cache.get("one_steger_per_frame") is not True:
        raise RuntimeError("Ground-5A cache does not certify one Steger run per frame")
    if cache.get("protocol_key") != g5a._cache_key(app, config_path):
        raise RuntimeError("Ground-5A cache protocol key does not match current config")

    requested: list[str] = []
    requested_pose_by_path: dict[str, str] = {}
    for record in records:
        if record.split != "fit" or record.pose_id not in FIT_POSES:
            raise RuntimeError(f"Ground-5C received a non-fit record: {record}")
        for path in record.laser_paths:
            source_path = str(path.resolve())
            requested.append(source_path)
            requested_pose_by_path[source_path] = record.pose_id

    fit_cache_by_path: dict[str, dict[str, Any]] = {}
    for item in cache.get("frames", []):
        if item.get("split") != "fit" or str(item.get("pose_id")) not in FIT_POSES:
            continue
        source = Path(str(item.get("source_path"))).resolve()
        fit_cache_by_path[str(source)] = item

    if set(requested) != set(fit_cache_by_path).intersection(requested):
        missing = sorted(set(requested) - set(fit_cache_by_path))
        raise RuntimeError(f"fit-only cache is missing requested entries: {missing}")
    if len(set(requested)) != len(FIT_POSES) * 5:
        raise RuntimeError(f"Ground-5C expected 25 unique fit laser paths, got {len(set(requested))}")

    centers_by_path: dict[str, np.ndarray] = {}
    selected_cache_rows: list[dict[str, Any]] = []
    for source_path in requested:
        source = Path(source_path)
        item = fit_cache_by_path[source_path]
        if str(item.get("pose_id")) != requested_pose_by_path[source_path]:
            raise RuntimeError(f"cache pose mismatch for {source}")
        source_sha = g5a._sha256_file(source)
        if item.get("source_sha256") != source_sha:
            raise RuntimeError(f"source SHA mismatch in Ground-5A cache: {source}")
        if int(item.get("steger_run_count", 0)) != 1:
            raise RuntimeError(f"cache entry is not exactly one Steger run: {source}")
        center_path = Path(str(item["centers_path"]))
        centers = np.asarray(np.load(center_path), dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
            raise RuntimeError(f"invalid cached centers: {center_path}")
        centers_by_path[source_path] = np.ascontiguousarray(centers)
        selected_cache_rows.append(
            {
                "pose_id": requested_pose_by_path[source_path],
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
        "selected_fit_entry_count": len(selected_cache_rows),
        "selected_fit_entries": selected_cache_rows,
        "read_only": True,
        "steger_rerun": False,
        "validation_entries_used": False,
    }


def _frame_data(
    frames: list[g5a.GroundFrame],
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for frame in frames:
        z = np.asarray(frame.points_ground[:, 2], dtype=np.float64)
        full_v = np.asarray(frame.coordinates["full_v"].x, dtype=np.float64)
        physical_s = g5b._physical_s(frame, origin_xy, direction_xy)
        if len(z) != len(full_v) or len(z) != len(physical_s):
            raise RuntimeError(f"coordinate length mismatch in {frame.frame_id}")
        data[frame.frame_id] = {
            "pose_id": frame.pose_id,
            "frame_id": frame.frame_id,
            "z": z,
            "full_v": full_v,
            "physical_S": physical_s,
            "point_count": int(len(z)),
            "source_path": frame.path,
        }
    return data


def _bin_frame_medians(
    x: np.ndarray,
    z: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw-Z median and point count for every fixed bin."""

    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if len(x) != len(z):
        raise RuntimeError("coordinate and Z arrays are misaligned")
    count = len(edges) - 1
    values = np.full(count, np.nan, dtype=np.float64)
    point_counts = np.zeros(count, dtype=np.int64)
    finite = np.isfinite(x) & np.isfinite(z)
    indices = np.searchsorted(edges, x, side="right") - 1
    indices[x == edges[-1]] = count - 1
    in_domain = finite & (x >= edges[0]) & (x <= edges[-1])
    for bin_index in range(count):
        selected = in_domain & (indices == bin_index)
        point_counts[bin_index] = int(np.count_nonzero(selected))
        if point_counts[bin_index]:
            values[bin_index] = float(np.median(z[selected]))
    return values, point_counts


def _build_support(
    data: dict[str, dict[str, Any]],
    pose_ids: Iterable[str],
    coordinate: str,
    edges: np.ndarray,
) -> dict[str, Any]:
    frames_by_pose: dict[str, list[dict[str, Any]]] = {pose: [] for pose in pose_ids}
    for item in data.values():
        frames_by_pose.setdefault(str(item["pose_id"]), []).append(item)
    frame_values: dict[str, np.ndarray] = {}
    frame_counts: dict[str, np.ndarray] = {}
    pose_profiles: dict[str, np.ndarray] = {}
    pose_frame_valid: dict[str, np.ndarray] = {}
    pose_point_counts: dict[str, np.ndarray] = {}
    pose_dispersion: dict[str, np.ndarray] = {}
    required_frames: dict[str, int] = {}

    for pose in pose_ids:
        pose_frames = sorted(frames_by_pose.get(pose, []), key=lambda item: str(item["frame_id"]))
        if not pose_frames:
            raise RuntimeError(f"no frames for fit pose{pose}")
        required = int(math.ceil(MIN_FRAME_FRACTION * len(pose_frames)))
        required_frames[pose] = required
        values = np.full((len(pose_frames), len(edges) - 1), np.nan, dtype=np.float64)
        counts = np.zeros_like(values, dtype=np.int64)
        for frame_index, item in enumerate(pose_frames):
            frame_value, frame_count = _bin_frame_medians(item[coordinate], item["z"], edges)
            values[frame_index] = frame_value
            counts[frame_index] = frame_count
            frame_values[str(item["frame_id"])] = frame_value
            frame_counts[str(item["frame_id"])] = frame_count
        valid_count = np.sum(np.isfinite(values), axis=0).astype(np.int64)
        profile = np.full(len(edges) - 1, np.nan, dtype=np.float64)
        dispersion = np.full(len(edges) - 1, np.nan, dtype=np.float64)
        available = valid_count >= required
        for bin_index in range(len(edges) - 1):
            supported = values[:, bin_index]
            supported = supported[np.isfinite(supported)]
            if len(supported) >= required:
                median = float(np.median(supported))
                profile[bin_index] = median
                dispersion[bin_index] = float(np.sqrt(np.mean((supported - median) ** 2)))
        pose_profiles[pose] = profile
        pose_frame_valid[pose] = valid_count
        pose_point_counts[pose] = np.sum(counts, axis=0).astype(np.int64)
        pose_dispersion[pose] = dispersion

    pose_count = np.sum(
        np.stack([np.isfinite(pose_profiles[pose]) for pose in pose_ids], axis=0), axis=0
    ).astype(np.int64)
    union_point_count = np.sum(
        np.stack([pose_point_counts[pose] for pose in pose_ids], axis=0), axis=0
    ).astype(np.int64)
    return {
        "coordinate": coordinate,
        "edges": np.asarray(edges, dtype=np.float64),
        "centers": (edges[:-1] + edges[1:]) / 2.0,
        "frame_values": frame_values,
        "frame_counts": frame_counts,
        "pose_profiles": pose_profiles,
        "pose_frame_valid": pose_frame_valid,
        "pose_point_counts": pose_point_counts,
        "pose_dispersion": pose_dispersion,
        "required_frames": required_frames,
        "pose_count": pose_count,
        "union_point_count": union_point_count,
        "formal_mask": pose_count >= FORMAL_MIN_POSES,
        "strong_mask": pose_count >= STRONG_MIN_POSES,
        "weak_mask": pose_count == 1,
        "union_mask": pose_count >= 1,
    }


def _fit_equal_bin_line(
    x: np.ndarray,
    z: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(x) & np.isfinite(z)
    x = np.asarray(x[finite], dtype=np.float64)
    z = np.asarray(z[finite], dtype=np.float64)
    if len(x) < 2 or float(np.ptp(x)) <= np.finfo(np.float64).eps:
        raise RuntimeError("formal bin support is insufficient for a linear fit")
    slope, intercept = np.linalg.lstsq(
        np.column_stack([x, np.ones_like(x)]), z, rcond=None
    )[0]
    residual = z - (slope * x + intercept)
    return {
        "a_session": float(slope),
        "b_session": float(intercept),
        "fit_bin_count": int(len(x)),
        "fit_x": x,
        "fit_z": z,
        "fit_residual": residual,
        "metrics": _metric_values(residual),
    }


def _formal_observations(support: dict[str, Any], pose_ids: Iterable[str]) -> dict[str, Any]:
    pose_ids = tuple(pose_ids)
    formal = np.asarray(support["formal_mask"], dtype=bool)
    pose_profiles = support["pose_profiles"]
    z_observation = np.full(len(formal), np.nan, dtype=np.float64)
    pose_spread = np.full(len(formal), np.nan, dtype=np.float64)
    for bin_index in np.flatnonzero(formal):
        values = np.asarray(
            [pose_profiles[pose][bin_index] for pose in pose_ids if np.isfinite(pose_profiles[pose][bin_index])],
            dtype=np.float64,
        )
        z_observation[bin_index] = float(np.mean(values))
        pose_spread[bin_index] = float(np.sqrt(np.mean((values - z_observation[bin_index]) ** 2)))
    return {
        "x": np.asarray(support["centers"], dtype=np.float64),
        "z": z_observation,
        "pose_spread": pose_spread,
        "formal_mask": formal,
        "strong_mask": np.asarray(support["strong_mask"], dtype=bool),
    }


def _fit_final_coordinate(support: dict[str, Any], pose_ids: Iterable[str]) -> dict[str, Any]:
    observations = _formal_observations(support, pose_ids)
    fit = _fit_equal_bin_line(observations["x"][observations["formal_mask"]], observations["z"][observations["formal_mask"]])
    formal_indices = np.flatnonzero(observations["formal_mask"])
    fit.update(
        {
            "formal_bin_indices": formal_indices.astype(int).tolist(),
            "strong_bin_indices": np.flatnonzero(observations["strong_mask"]).astype(int).tolist(),
            "weak_bin_indices": np.flatnonzero(np.asarray(support["weak_mask"], dtype=bool)).astype(int).tolist(),
            "observations": observations,
            "valid_domain": [
                float(support["edges"][formal_indices[0]]),
                float(support["edges"][formal_indices[-1] + 1]),
            ],
        }
    )
    return fit


def _frame_repeatability(
    support: dict[str, Any],
    pose: str,
    evaluated_mask: np.ndarray,
) -> float:
    pose_profile = support["pose_profiles"][pose]
    values: list[float] = []
    for frame_id, frame_profile in support["frame_values"].items():
        if not frame_id.startswith(f"fit_pose{pose}_"):
            continue
        valid = evaluated_mask & np.isfinite(frame_profile) & np.isfinite(pose_profile)
        if np.count_nonzero(valid):
            values.append(float(np.sqrt(np.mean((frame_profile[valid] - pose_profile[valid]) ** 2))))
    return float(np.mean(values)) if values else math.nan


def _lopo_rows(
    support: dict[str, Any],
    pose_ids: Iterable[str],
) -> list[dict[str, Any]]:
    pose_ids = tuple(pose_ids)
    rows: list[dict[str, Any]] = []
    for heldout in pose_ids:
        train = tuple(pose for pose in pose_ids if pose != heldout)
        train_available = np.stack(
            [np.isfinite(support["pose_profiles"][pose]) for pose in train], axis=0
        )
        train_pose_count = np.sum(train_available, axis=0)
        train_formal = train_pose_count >= FORMAL_MIN_POSES
        train_strong = train_pose_count >= STRONG_MIN_POSES
        train_z = np.full(len(support["centers"]), np.nan, dtype=np.float64)
        for bin_index in np.flatnonzero(train_formal):
            values = np.asarray(
                [support["pose_profiles"][pose][bin_index] for pose in train if np.isfinite(support["pose_profiles"][pose][bin_index])],
                dtype=np.float64,
            )
            train_z[bin_index] = float(np.mean(values))
        fit = _fit_equal_bin_line(support["centers"][train_formal], train_z[train_formal])
        heldout_profile = support["pose_profiles"][heldout]
        heldout_supported = np.isfinite(heldout_profile)
        evaluated = train_formal & heldout_supported
        residual = heldout_profile[evaluated] - (
            fit["a_session"] * support["centers"][evaluated] + fit["b_session"]
        )
        metrics = _metric_values(residual)
        rows.append(
            {
                "aggregation_level": "fold",
                "coordinate": support["coordinate"],
                "heldout_pose_id": heldout,
                "train_pose_ids": ",".join(train),
                "train_formal_bin_count": int(np.count_nonzero(train_formal)),
                "train_strong_bin_count": int(np.count_nonzero(train_strong)),
                "heldout_supported_bin_count": int(np.count_nonzero(heldout_supported)),
                "evaluated_bin_count": int(np.count_nonzero(evaluated)),
                "support_coverage": float(np.mean(evaluated[heldout_supported])) if np.count_nonzero(heldout_supported) else math.nan,
                "union_coverage": float(np.mean(evaluated)),
                "a_session": fit["a_session"],
                "b_session": fit["b_session"],
                "fit_bin_count": fit["fit_bin_count"],
                "bias_mm": metrics["bias_mm"],
                "rmse_mm": metrics["rmse_mm"],
                "p95_abs_mm": metrics["p95_abs_mm"],
                "max_abs_mm": metrics["max_abs_mm"],
                "peak_to_peak_mm": metrics["peak_to_peak_mm"],
                "frame_repeatability_rmse_mm": _frame_repeatability(support, heldout, evaluated),
                "no_extrapolation": True,
                "fit_source": "four_fit_poses_equal_pose_bin_observations",
            }
        )
    numeric_fields = (
        "support_coverage",
        "union_coverage",
        "a_session",
        "b_session",
        "bias_mm",
        "rmse_mm",
        "p95_abs_mm",
        "max_abs_mm",
        "peak_to_peak_mm",
        "frame_repeatability_rmse_mm",
    )
    mean_row: dict[str, Any] = {
        "aggregation_level": "mean_lopo",
        "coordinate": support["coordinate"],
        "heldout_pose_id": "ALL_FIT_POSES_EQUAL_WEIGHT",
        "train_pose_ids": "",
        "train_formal_bin_count": float(np.mean([row["train_formal_bin_count"] for row in rows])),
        "train_strong_bin_count": float(np.mean([row["train_strong_bin_count"] for row in rows])),
        "heldout_supported_bin_count": float(np.mean([row["heldout_supported_bin_count"] for row in rows])),
        "evaluated_bin_count": float(np.mean([row["evaluated_bin_count"] for row in rows])),
        "fit_bin_count": float(np.mean([row["fit_bin_count"] for row in rows])),
        "no_extrapolation": True,
        "fit_source": "equal_weight_mean_of_five_lopo_folds",
    }
    for field in numeric_fields:
        mean_row[field] = float(np.mean([row[field] for row in rows]))
    rows.append(mean_row)
    return rows


def _per_pose_diagnostics(
    support: dict[str, Any],
    data: dict[str, dict[str, Any]],
    pose_ids: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pose in pose_ids:
        available = np.isfinite(support["pose_profiles"][pose])
        x = support["centers"][available]
        z = support["pose_profiles"][pose][available]
        if len(x) >= 2 and float(np.ptp(x)) > np.finfo(np.float64).eps:
            slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), z, rcond=None)[0]
            residual = z - (slope * x + intercept)
            metrics = _metric_values(residual)
        else:
            slope = intercept = math.nan
            metrics = _metric_values(np.array([], dtype=np.float64))
        frame_ids = [frame_id for frame_id, item in data.items() if item["pose_id"] == pose]
        rows.append(
            {
                "coordinate": support["coordinate"],
                "pose_id": pose,
                "fit_source": "pose_bin_medians_not_raw_points",
                "support_bin_count": int(np.count_nonzero(available)),
                "formal_bin_count": int(np.count_nonzero(available & support["formal_mask"])),
                "strong_bin_count": int(np.count_nonzero(available & support["strong_mask"])),
                "frame_count": len(frame_ids),
                "frame_support_count_total": int(sum(np.count_nonzero(support["frame_values"][frame_id]) for frame_id in frame_ids)),
                "point_count_total": int(np.sum(support["pose_point_counts"][pose])),
                "support_min": float(np.min(x)) if len(x) else math.nan,
                "support_max": float(np.max(x)) if len(x) else math.nan,
                "a_pose": float(slope),
                "b_pose": float(intercept),
                "bias_mm": metrics["bias_mm"],
                "rmse_mm": metrics["rmse_mm"],
                "p95_abs_mm": metrics["p95_abs_mm"],
                "max_abs_mm": metrics["max_abs_mm"],
                "peak_to_peak_mm": metrics["peak_to_peak_mm"],
                "mean_frame_dispersion_mm": float(np.nanmean(support["pose_dispersion"][pose])) if np.any(np.isfinite(support["pose_dispersion"][pose])) else math.nan,
            }
        )
    return rows


def _support_summary(support: dict[str, Any], pose_ids: Iterable[str]) -> dict[str, Any]:
    pose_ids = tuple(pose_ids)
    formal = np.asarray(support["formal_mask"], dtype=bool)
    strong = np.asarray(support["strong_mask"], dtype=bool)
    union = np.asarray(support["union_mask"], dtype=bool)
    per_pose_formal = {
        pose: int(np.count_nonzero(np.isfinite(support["pose_profiles"][pose]) & formal))
        for pose in pose_ids
    }
    formal_count = int(np.count_nonzero(formal))
    strong_count = int(np.count_nonzero(strong))
    union_count = int(np.count_nonzero(union))
    min_pose_fraction = (
        min(per_pose_formal.values()) / formal_count if formal_count else 0.0
    )
    if (
        formal_count >= int(0.8 * len(formal))
        and strong_count >= int(0.5 * len(strong))
        and min_pose_fraction >= 0.8
    ):
        classification = "PASS"
    elif formal_count >= 2 and min_pose_fraction >= 0.5:
        classification = "PARTIAL"
    else:
        classification = "FAIL"
    return {
        "coordinate": support["coordinate"],
        "bin_count": len(formal),
        "union_bin_count": union_count,
        "formal_bin_count": formal_count,
        "strong_bin_count": strong_count,
        "weak_bin_count": int(np.count_nonzero(support["weak_mask"])),
        "unsupported_bin_count": int(len(formal) - union_count),
        "formal_fraction": formal_count / len(formal),
        "strong_fraction": strong_count / len(strong),
        "min_pose_formal_fraction": min_pose_fraction,
        "per_pose_formal_bin_count": per_pose_formal,
        "classification": classification,
    }


def _mean_lopo(rows: list[dict[str, Any]]) -> dict[str, float]:
    folds = [row for row in rows if row["aggregation_level"] == "fold"]
    return {
        "rmse_mm": float(np.mean([row["rmse_mm"] for row in folds])),
        "p95_abs_mm": float(np.mean([row["p95_abs_mm"] for row in folds])),
        "support_coverage": float(np.mean([row["support_coverage"] for row in folds])),
        "frame_repeatability_rmse_mm": float(np.mean([row["frame_repeatability_rmse_mm"] for row in folds])),
    }


def _select_coordinate(lopo_by_coordinate: dict[str, list[dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    scores = {coordinate: _mean_lopo(rows) for coordinate, rows in lopo_by_coordinate.items()}
    v = scores["full_v"]
    s = scores["physical_S"]
    decision = "physical_S"
    reason = "tie_break_priority_physical_S"
    if abs(v["rmse_mm"] - s["rmse_mm"]) > COORDINATE_TIE_TOLERANCE_MM:
        decision = "full_v" if v["rmse_mm"] < s["rmse_mm"] else "physical_S"
        reason = "mean_lopo_rmse"
    elif abs(v["p95_abs_mm"] - s["p95_abs_mm"]) > COORDINATE_TIE_TOLERANCE_MM:
        decision = "full_v" if v["p95_abs_mm"] < s["p95_abs_mm"] else "physical_S"
        reason = "mean_lopo_p95"
    elif abs(v["support_coverage"] - s["support_coverage"]) > COORDINATE_TIE_TOLERANCE_COVERAGE:
        decision = "full_v" if v["support_coverage"] > s["support_coverage"] else "physical_S"
        reason = "mean_lopo_support_coverage"
    elif abs(v["frame_repeatability_rmse_mm"] - s["frame_repeatability_rmse_mm"]) > COORDINATE_TIE_TOLERANCE_MM:
        decision = "full_v" if v["frame_repeatability_rmse_mm"] < s["frame_repeatability_rmse_mm"] else "physical_S"
        reason = "mean_lopo_repeatability"
    return decision, {
        "selected_coordinate": decision,
        "reason": reason,
        "ranking_order": ["mean_lopo_rmse", "mean_lopo_p95", "mean_lopo_support_coverage", "mean_lopo_repeatability"],
        "tie_tolerance_mm": COORDINATE_TIE_TOLERANCE_MM,
        "tie_tolerance_coverage": COORDINATE_TIE_TOLERANCE_COVERAGE,
        "scores": scores,
    }


def _plot_union_coverage(
    supports: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, coordinate in zip(axes, COORDINATES, strict=True):
        support = supports[coordinate]
        centers = support["centers"]
        axis.step(centers, support["pose_count"], where="mid", color="#1f4e79", linewidth=1.8, label="pose_count")
        for pose in FIT_POSES:
            axis.plot(
                centers,
                np.isfinite(support["pose_profiles"][pose]).astype(float),
                linewidth=0.9,
                alpha=0.55,
                label=f"pose{pose}",
            )
        axis.axhline(FORMAL_MIN_POSES, color="#c27c0e", linestyle="--", linewidth=1.0, label="formal >=2")
        axis.axhline(STRONG_MIN_POSES, color="#2f855a", linestyle=":", linewidth=1.2, label="strong >=3")
        axis.set_ylim(-0.1, len(FIT_POSES) + 0.3)
        axis.set_ylabel("pose support count")
        axis.set_title(f"{coordinate}: 40-bin union support")
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("coordinate (full_v px or physical_S mm)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_pooled_fit_residual(
    supports: dict[str, dict[str, Any]],
    final_fits: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, coordinate in zip(axes, COORDINATES, strict=True):
        support = supports[coordinate]
        fit = final_fits[coordinate]
        obs = fit["observations"]
        formal = np.asarray(obs["formal_mask"], dtype=bool)
        x = obs["x"][formal]
        residual = obs["z"][formal] - (fit["a_session"] * x + fit["b_session"])
        strong = np.asarray(obs["strong_mask"], dtype=bool)[formal]
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.scatter(x[~strong], residual[~strong], s=28, color="#c27c0e", label="formal (2 poses)")
        axis.scatter(x[strong], residual[strong], s=30, color="#2f855a", label="strong (>=3 poses)")
        axis.set_ylabel("bin-balanced fit residual Zg (mm)")
        axis.set_title(
            f"{coordinate}: equal-pose formal-bin residual; RMSE={fit['metrics']['rmse_mm']:.6g} mm"
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("coordinate (full_v px or physical_S mm)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _build_union_rows(
    supports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coordinate in COORDINATES:
        support = supports[coordinate]
        for index, center in enumerate(support["centers"]):
            count = int(support["pose_count"][index])
            row: dict[str, Any] = {
                "coordinate": coordinate,
                "bin_index": index,
                "x_left": support["edges"][index],
                "x_right": support["edges"][index + 1],
                "x_center": center,
                "union_point_count": support["union_point_count"][index],
                "pose_count": count,
                "support_class": "strong" if count >= STRONG_MIN_POSES else "formal" if count >= FORMAL_MIN_POSES else "weak" if count == 1 else "unsupported",
                "formal_support": count >= FORMAL_MIN_POSES,
                "strong_support": count >= STRONG_MIN_POSES,
                "weak_support": count == 1,
            }
            for pose in FIT_POSES:
                row[f"pose{pose}_frame_support_count"] = support["pose_frame_valid"][pose][index]
                row[f"pose{pose}_point_count"] = support["pose_point_counts"][pose][index]
                row[f"pose{pose}_zg_median_mm"] = support["pose_profiles"][pose][index]
                row[f"pose{pose}_dispersion_mm"] = support["pose_dispersion"][pose][index]
                row[f"pose{pose}_profile_available"] = bool(np.isfinite(support["pose_profiles"][pose][index]))
            rows.append(row)
    return rows


def _build_report(
    output: Path,
    supports: dict[str, dict[str, Any]],
    support_summaries: dict[str, dict[str, Any]],
    final_fits: dict[str, dict[str, Any]],
    lopo_by_coordinate: dict[str, list[dict[str, Any]]],
    selection: dict[str, Any],
    frozen: dict[str, Any],
) -> None:
    selected = frozen["coordinate"]
    selected_summary = support_summaries[selected]
    selected_lopo = _mean_lopo(lopo_by_coordinate[selected])
    if selected_summary["classification"] == "PASS" and selected_lopo["support_coverage"] >= 0.8:
        frozen_fit_status = "PASS"
    elif np.isfinite(selected_lopo["rmse_mm"]):
        frozen_fit_status = "PARTIAL"
    else:
        frozen_fit_status = "FAIL"
    more_coverage = "NO" if selected_summary["classification"] == "PASS" and frozen_fit_status == "PASS" else "YES"
    lines = [
        "# Ground-5C｜Frozen Session Linear Fit-only Audit",
        "",
        "## 结论",
        "",
        f"- `UNION_SUPPORT = {selected_summary['classification']}`",
        f"- `FROZEN_SESSION_LINEAR_FIT = {frozen_fit_status}`",
        f"- `RECOMMENDED_COORDINATE = {selected}`",
        f"- `MORE_COVERAGE_REQUIRED = {more_coverage}`",
        f"- frozen `a_session = {frozen['a_session']:.12g}`, `b_session = {frozen['b_session']:.12g}`",
        f"- frozen valid domain = `{frozen['valid_domain']}`",
        "",
        "本轮只读取 fit 目录的 pose001–005；没有发现 validation 目录，没有读取 pose006/007 评价结果，也没有运行正式 held-out 验证。",
        "",
        "## Provenance / reuse audit",
        "",
        "- 复用：Ground5A 的 PnP、physical-board mask、Frozen C0/C1 reconstruction、图像/标定配置解析。",
        "- 复用：Ground5A 的 Steger cache；本轮只解引用 25 个 fit cache entry，`steger_rerun=false`。",
        "- 复用：Ground-1 的全局 physical-S origin/direction；没有重新定义 S。",
        "- 本轮新增：40-bin union-support、frame median → pose median、pose_count 门槛、equal-bin linear fit 和 fit-only LOPO。",
        "- 未使用：Factory Profile、Ground-3 数值参数、C0/C1/H1 修改、residual/truth mask、raw-point pooled fit。",
        "",
        "## Support",
        "",
        "固定 bins 的范围是该坐标在 001–005 所有 board-mask-selected laser ground points 的 union min/max；LOPO 使用这组预先固定的 fit-only bins。每个 frame/bin 只取 raw PnP-ground `Zg` median；每个 pose/bin 再取 frame median 的 median，空 bin 不补值。",
        "",
        "| coordinate | union bins | formal (>=2 poses) | strong (>=3 poses) | weak (1 pose) | min pose formal fraction | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for coordinate in COORDINATES:
        summary = support_summaries[coordinate]
        lines.append(
            f"| {coordinate} | {summary['union_bin_count']} | {summary['formal_bin_count']} | {summary['strong_bin_count']} | {summary['weak_bin_count']} | {summary['min_pose_formal_fraction']:.4g} | {summary['classification']} |"
        )
    lines.extend(
        [
            "",
            "正式 fit 的每个 bin 先对其有效 pose 的 pose/bin median 做算术平均，因此每个 formal bin 只贡献一个等权观测；不按原始点数或 Z residual 加权/删 bin。valid domain 只表示 formal bin 的外边界，内部仍以 formal bin index 为准，不外推、不 clamp。",
            "",
            "## Coordinate selection by fit-only LOPO",
            "",
            "选择顺序冻结为 mean LOPO RMSE → mean LOPO P95 → mean support coverage → mean frame repeatability；前三级近似持平时优先 physical_S。",
            "",
            "| coordinate | mean LOPO RMSE (mm) | mean LOPO P95 (mm) | mean coverage | mean repeatability (mm) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for coordinate in COORDINATES:
        score = selection["scores"][coordinate]
        lines.append(
            f"| {coordinate} | {score['rmse_mm']:.8g} | {score['p95_abs_mm']:.8g} | {score['support_coverage']:.6f} | {score['frame_repeatability_rmse_mm']:.8g} |"
        )
    lines.extend(
        [
            "",
            f"冻结结果：`{selected}`，原因：`{selection['reason']}`。",
            "",
            "## Frozen parameters",
            "",
            f"- coordinate: `{frozen['coordinate']}`",
            f"- `a_session`: `{frozen['a_session']:.12g}` mm / coordinate-unit",
            f"- `b_session`: `{frozen['b_session']:.12g}` mm",
            f"- valid domain: `{frozen['valid_domain']}`",
            f"- formal bin count: `{len(frozen['formal_bin_indices'])}`; strong: `{len(frozen['strong_bin_indices'])}`; weak diagnostic: `{len(frozen['weak_bin_indices'])}`",
            "",
            "## Outputs",
            "",
            "- `union_coverage.csv` / `union_coverage.png`",
            "- `per_pose_linear_diagnostics.csv`",
            "- `fit_lopo_coordinate_comparison.csv`",
            "- `pooled_fit_residual.png`",
            "- `frozen_session_linear.json`",
        ]
    )
    (output / "ground5c_fit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frozen["classification"] = {
        "UNION_SUPPORT": selected_summary["classification"],
        "FROZEN_SESSION_LINEAR_FIT": frozen_fit_status,
        "MORE_COVERAGE_REQUIRED": more_coverage,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    fit_dir = args.fit_dir.resolve()
    config_path = args.config.resolve()
    ground5a_output = args.ground5a_output.resolve()
    ground1_summary_path = args.ground1_summary.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Deliberately no validation_dir argument or discovery call exists here.
    records = g5a._discover_pose_records(fit_dir, "fit", FIT_POSES)
    dataset_document, metadata_by_name, manifest_path, frames_csv_path = g5a._load_manifest_context(fit_dir)
    app = g5a.load_app_config(config_path)
    if app.reconstruction.image_roi_polygon is not None:
        raise RuntimeError("Ground-5C requires reconstruction.image_roi_polygon=null")
    if not app.reconstruction.enable_laser_ray_correction:
        raise RuntimeError("Ground-5C requires Frozen C1 to be enabled")
    if app.calibration.manifest is None:
        raise RuntimeError("Ground-5C requires calibration.manifest")
    package = g5a.load_calibration_package(app.calibration.manifest)
    base_calibration = dict(package.calibration)
    params_c0 = g5a.replace(app.reconstruction, enable_laser_ray_correction=False)
    params_c1 = app.reconstruction
    board_config = app.session_ground_calibration.board_config()
    mask_inset_mm = float(app.session_ground_calibration.sanity.mask_inset_mm)
    if mask_inset_mm != float(app.session_ground_calibration.ground_reference.mask_inset_mm):
        raise RuntimeError("Session sanity and ground-reference mask inset differ")

    centers_by_path, cache_provenance = _load_fit_cached_centers_only(
        records, ground5a_output, app, config_path
    )
    pnp_by_pose = g5a._run_pnp(records, base_calibration, board_config)
    frames = g5a._process_frames(
        records,
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
    if len(frames) != 25 or {frame.pose_id for frame in frames} != set(FIT_POSES):
        raise RuntimeError("Ground-5C did not produce exactly 5 frames for each fit pose")

    ground1 = g5b._load_ground1_s(ground1_summary_path)
    origin_xy = np.asarray(ground1["origin_xy"], dtype=np.float64)
    direction_xy = np.asarray(ground1["direction_xy"], dtype=np.float64)
    data = _frame_data(frames, origin_xy, direction_xy)

    supports: dict[str, dict[str, Any]] = {}
    for coordinate in COORDINATES:
        all_x = np.concatenate([np.asarray(item[coordinate], dtype=np.float64) for item in data.values()])
        finite_x = all_x[np.isfinite(all_x)]
        if len(finite_x) == 0 or not float(np.min(finite_x)) < float(np.max(finite_x)):
            raise RuntimeError(f"no union support for {coordinate}")
        edges = np.linspace(float(np.min(finite_x)), float(np.max(finite_x)), PROFILE_BIN_COUNT + 1)
        supports[coordinate] = _build_support(data, FIT_POSES, coordinate, edges)

    support_summaries = {coordinate: _support_summary(supports[coordinate], FIT_POSES) for coordinate in COORDINATES}
    final_fits = {coordinate: _fit_final_coordinate(supports[coordinate], FIT_POSES) for coordinate in COORDINATES}
    lopo_by_coordinate = {coordinate: _lopo_rows(supports[coordinate], FIT_POSES) for coordinate in COORDINATES}
    selected_coordinate, selection = _select_coordinate(lopo_by_coordinate)
    frozen_fit = final_fits[selected_coordinate]
    frozen = {
        "coordinate": selected_coordinate,
        "a_session": frozen_fit["a_session"],
        "b_session": frozen_fit["b_session"],
        "valid_domain": frozen_fit["valid_domain"],
        "formal_bin_indices": frozen_fit["formal_bin_indices"],
        "strong_bin_indices": frozen_fit["strong_bin_indices"],
        "weak_bin_indices": frozen_fit["weak_bin_indices"],
    }

    union_rows = _build_union_rows(supports)
    union_fields = [
        "coordinate", "bin_index", "x_left", "x_right", "x_center", "union_point_count",
        "pose_count", "support_class", "formal_support", "strong_support", "weak_support",
    ]
    for pose in FIT_POSES:
        union_fields.extend(
            [
                f"pose{pose}_frame_support_count",
                f"pose{pose}_point_count",
                f"pose{pose}_zg_median_mm",
                f"pose{pose}_dispersion_mm",
                f"pose{pose}_profile_available",
            ]
        )
    _write_csv(output / "union_coverage.csv", union_rows, union_fields)

    diagnostic_rows: list[dict[str, Any]] = []
    for coordinate in COORDINATES:
        diagnostic_rows.extend(_per_pose_diagnostics(supports[coordinate], data, FIT_POSES))
    _write_csv(
        output / "per_pose_linear_diagnostics.csv",
        diagnostic_rows,
        [
            "coordinate", "pose_id", "fit_source", "support_bin_count", "formal_bin_count", "strong_bin_count",
            "frame_count", "frame_support_count_total", "point_count_total", "support_min", "support_max",
            "a_pose", "b_pose", "bias_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm", "peak_to_peak_mm",
            "mean_frame_dispersion_mm",
        ],
    )

    lopo_rows = [row for coordinate in COORDINATES for row in lopo_by_coordinate[coordinate]]
    _write_csv(
        output / "fit_lopo_coordinate_comparison.csv",
        lopo_rows,
        [
            "aggregation_level", "coordinate", "heldout_pose_id", "train_pose_ids", "train_formal_bin_count",
            "train_strong_bin_count", "heldout_supported_bin_count", "evaluated_bin_count", "support_coverage",
            "union_coverage", "a_session", "b_session", "fit_bin_count", "bias_mm", "rmse_mm", "p95_abs_mm",
            "max_abs_mm", "peak_to_peak_mm", "frame_repeatability_rmse_mm", "no_extrapolation", "fit_source",
        ],
    )

    _plot_union_coverage(supports, output / "union_coverage.png")
    _plot_pooled_fit_residual(supports, final_fits, output / "pooled_fit_residual.png")

    source_files = []
    for record in records:
        for path in record.laser_paths:
            source_files.append(
                {
                    "pose_id": record.pose_id,
                    "source_file": path.name,
                    "source_path": path.resolve(),
                    "source_sha256": g5a._sha256_file(path),
                }
            )
    pnp_provenance = {
        pose: {
            "split": pnp_by_pose[pose].split,
            "chess_path": pnp_by_pose[pose].chess_path,
            "reprojection_rmse_px": pnp_by_pose[pose].reprojection_rmse_px,
            "detection_method": pnp_by_pose[pose].detection_method,
            "corner_count": len(pnp_by_pose[pose].result.detected_corners),
        }
        for pose in FIT_POSES
    }
    frozen_json = {
        "schema_version": 1,
        "status": "frozen_fit_only",
        "coordinate": selected_coordinate,
        "coordinate_units": "px" if selected_coordinate == "full_v" else "mm",
        "a_session": frozen_fit["a_session"],
        "b_session": frozen_fit["b_session"],
        "valid_domain": frozen_fit["valid_domain"],
        "bin_edges": supports[selected_coordinate]["edges"],
        "bin_centers": supports[selected_coordinate]["centers"],
        "formal_bin_indices": frozen_fit["formal_bin_indices"],
        "strong_bin_indices": frozen_fit["strong_bin_indices"],
        "weak_bin_indices": frozen_fit["weak_bin_indices"],
        "formal_support": support_summaries[selected_coordinate],
        "support_by_coordinate": support_summaries,
        "coordinate_candidates": {
            coordinate: {
                "a_session": final_fits[coordinate]["a_session"],
                "b_session": final_fits[coordinate]["b_session"],
                "valid_domain": final_fits[coordinate]["valid_domain"],
                "formal_bin_indices": final_fits[coordinate]["formal_bin_indices"],
                "strong_bin_indices": final_fits[coordinate]["strong_bin_indices"],
                "weak_bin_indices": final_fits[coordinate]["weak_bin_indices"],
                "fit_metrics": final_fits[coordinate]["metrics"],
            }
            for coordinate in COORDINATES
        },
        "fit_pose_ids": list(FIT_POSES),
        "lopo_metrics": lopo_by_coordinate,
        "coordinate_selection": selection,
        "physical_S": {
            "formula": ground1["formula"],
            "origin_xy_mm": origin_xy,
            "direction_xy": direction_xy,
            "per_frame_redefinition": ground1["per_frame_redefinition"],
            "source_path": ground1["source_path"],
            "source_sha256": ground1["source_sha256"],
        },
        "fit_protocol": {
            "bin_count": PROFILE_BIN_COUNT,
            "bin_edges_are_fit_only_union_min_max": True,
            "frame_aggregation": "median_Zg_per_frame_bin",
            "pose_aggregation": "median_over_supported_frame_bin_medians",
            "minimum_frame_fraction": MIN_FRAME_FRACTION,
            "formal_rule": "pose_count>=2",
            "strong_rule": "pose_count>=3",
            "weak_rule": "pose_count==1_diagnostic_only",
            "formal_observation": "arithmetic_mean_of_available_pose_bin_medians_equal_pose_weight",
            "fit_weighting": "one_equal_weight_observation_per_formal_bin",
            "raw_point_pooling": False,
            "z_residual_filtering": False,
            "extrapolation": False,
            "clamp": False,
            "factory_profile_used": False,
            "ground3_numeric_parameters_used": False,
            "h1_used": False,
        },
        "provenance": {
            "fit_dir": fit_dir,
            "config_path": config_path,
            "config_sha256": g5a._sha256_file(config_path),
            "manifest_path": manifest_path,
            "frames_csv_path": frames_csv_path,
            "ground5a_output": ground5a_output,
            "cache": cache_provenance,
            "source_files": source_files,
            "pnp": pnp_provenance,
            "frame_count": len(frames),
            "point_selection": "Ground5A physical-board mask; no new mask selection",
            "validation_data_read": False,
            "validation_metrics_read": False,
            "steger_rerun": False,
        },
    }
    (output / "frozen_session_linear.json").write_text(
        json.dumps(_json_ready(frozen_json), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _build_report(
        output,
        supports,
        support_summaries,
        final_fits,
        lopo_by_coordinate,
        selection,
        frozen,
    )
    return frozen_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ground5a-output", type=Path, default=DEFAULT_GROUND5A_OUTPUT)
    parser.add_argument("--ground1-summary", type=Path, default=DEFAULT_GROUND1_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    result = _run(_parse_args())
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))
