"""Retrospective Ground-4A replay on the frozen Daheng gauge-block audit.

The replay consumes the already audited C1 pointwise output.  It therefore
uses the original manual-frozen ROI, one Steger result per image, and the same
C1 reconstruction without touching the production chain.  Only the session
linear proxy and its diagnostic G(S) variant are newly computed here.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
if str(MEASUREMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_ROOT))

from app_config import load_app_config
from measurement.height_measure import MeasurementParams, _fit_line_xy


DEFAULT_GAUGE_DIR = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_manual_frozen"
DEFAULT_GROUND3_SUMMARY = (
    MEASUREMENT_ROOT
    / "output_daheng_0811"
    / "ground_spatial_correction_ground3"
    / "ground_gs_summary.json"
)
DEFAULT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"

DATASET_ORDER = {
    "obs_1mm": 0,
    "obs_2mm": 1,
    "obs_6mm": 2,
    "obs_10mm": 3,
    "obs_20mm": 4,
    "obs_30mm": 5,
}
CHAIN_NAMES = ("fixed_zg_zero", "session_linear", "session_linear_G", "local_adjacent")
CHAIN_LABELS = {
    "fixed_zg_zero": "A fixed_zg_zero",
    "session_linear": "B session_linear",
    "session_linear_G": "C session_linear_G",
    "local_adjacent": "D local_adjacent",
}
TRUTH_DEFAULT = {
    "obs_1mm": 1.001,
    "obs_2mm": 2.0,
    "obs_6mm": 6.0,
    "obs_10mm": 10.0,
    "obs_20mm": 20.0,
    "obs_30mm": 30.0,
}


@dataclass(slots=True)
class GroundModel:
    slope: float
    intercept: float
    rmse: float
    point_count: int
    inlier_count: int
    s_min: float
    s_max: float
    unsupported_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _finite_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if sigma <= np.finfo(np.float64).eps:
        sigma = float(np.std(values))
    return sigma


def _fit_fixed_s_profile(s: np.ndarray, z: np.ndarray, params: MeasurementParams) -> GroundModel:
    """Reuse height_measure.py's MAD/iterative robust profile convention."""
    s = np.asarray(s, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(z)
    s = s[finite]
    z = z[finite]
    if len(z) < 2:
        raise ValueError("ground proxy has too few points")
    mask = np.ones(len(z), dtype=bool)
    slope = 0.0
    intercept = float(np.median(z))
    for _ in range(params.outlier_max_iterations):
        selected_s = s[mask]
        selected_z = z[mask]
        if len(selected_z) < 2:
            raise ValueError("ground proxy has too few robust inliers")
        if float(np.ptp(selected_s)) < 1.0:
            slope, intercept = 0.0, float(np.median(selected_z))
        else:
            slope, intercept = np.linalg.lstsq(
                np.column_stack([selected_s, np.ones_like(selected_s)]),
                selected_z,
                rcond=None,
            )[0]
            slope = float(slope)
            intercept = float(intercept)
        residual = z - (slope * s + intercept)
        sigma = _robust_sigma(residual[mask])
        if sigma <= np.finfo(np.float64).eps:
            break
        new_mask = np.abs(residual) <= params.outlier_sigma_multiplier * sigma
        if int(new_mask.sum()) < 2 or bool(np.all(new_mask == mask)):
            break
        mask = new_mask
    selected_residual = z[mask] - (slope * s[mask] + intercept)
    return GroundModel(
        slope=float(slope),
        intercept=float(intercept),
        rmse=float(np.sqrt(np.mean(selected_residual**2))),
        point_count=int(len(z)),
        inlier_count=int(mask.sum()),
        s_min=float(np.min(s)),
        s_max=float(np.max(s)),
        unsupported_count=0,
    )


def _load_ground3(summary_path: Path) -> tuple[dict[str, Any], BSpline, np.ndarray, np.ndarray]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary["selected_model"]
    if selected["model_name"] != "cubic_bspline_3_interior_knots":
        raise RuntimeError("Ground-4A requires the frozen Ground-3 3-knot candidate")
    knots = np.asarray(selected["knots"], dtype=np.float64)
    coefficients = np.asarray(selected["coefficients"], dtype=np.float64)
    spline = BSpline(knots, coefficients, 3, extrapolate=False)
    origin = np.asarray(summary["protocol"]["origin_xy"], dtype=np.float64)
    direction = np.asarray(summary["protocol"]["direction_xy"], dtype=np.float64)
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-8):
        raise RuntimeError("Ground-1 frozen direction is not unit length")
    return summary, spline, origin, direction


def _load_and_validate_provenance(gauge_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    audit = json.loads((gauge_dir / "audit_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((gauge_dir / "provenance.json").read_text(encoding="utf-8"))
    registry = json.loads((gauge_dir / "roi_registry.json").read_text(encoding="utf-8"))
    if audit.get("image_count") != 150 or audit.get("steger_call_count") != 150:
        raise RuntimeError("frozen gauge audit is not the expected 150-image/150-Steger artifact")
    if provenance["reconstruction_params_c1"].get("enable_laser_ray_correction") is not True:
        raise RuntimeError("frozen gauge artifact does not certify C1 enabled")
    summary = registry.get("summary", {})
    if not (
        summary.get("manual_confirmed") is True
        and summary.get("manual_confirmed_count") == 30
        and summary.get("manual_review_required") is False
    ):
        raise RuntimeError("manual-frozen ROI registry is not fully confirmed")
    frame_rows = _read_csv(gauge_dir / "frame_metrics.csv")
    if len(frame_rows) != 150 or not all(row.get("steger_called_once") == "True" for row in frame_rows):
        raise RuntimeError("frame audit does not certify one Steger per all 150 frames")
    return frame_rows, provenance


def _point_dict(row: dict[str, str], origin: np.ndarray, direction: np.ndarray) -> dict[str, Any] | None:
    if row.get("c1_status") != "valid":
        return None
    x = _finite_float(row.get("c1_Xg_mm"))
    y = _finite_float(row.get("c1_Yg_mm"))
    z = _finite_float(row.get("c1_Zg_mm"))
    if x is None or y is None or z is None:
        return None
    xy = np.asarray([x, y], dtype=np.float64)
    return {
        "xy": xy,
        "z": z,
        "s": float((xy - origin) @ direction),
        "region": row["image_region"],
    }


def _load_frames(gauge_dir: Path, origin: np.ndarray, direction: np.ndarray) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = _read_csv(gauge_dir / "pointwise_diagnostics.csv")
    frames: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], int(row["position_rank"]), int(row["repeat_index"]))
        frame = frames.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "position_rank": int(row["position_rank"]),
                "repeat_index": int(row["repeat_index"]),
                "pose_id": row["pose_id"],
                "filename": row["filename"],
                "truth_mm": None,
                "baseline": [],
                "height": [],
            },
        )
        point = _point_dict(row, origin, direction)
        if point is None:
            continue
        frame["truth_mm"] = float(row["truth_mm"])
        if row["image_region"] in {"baseline_before", "baseline_after"}:
            frame["baseline"].append(point)
        elif row["image_region"] == "height":
            frame["height"].append(point)
    if len(frames) != 150:
        raise RuntimeError(f"expected 150 gauge frames in pointwise output, got {len(frames)}")
    return frames


def _g_values(spline: BSpline, s: np.ndarray, domain_min: float, domain_max: float) -> tuple[np.ndarray, int]:
    s = np.asarray(s, dtype=np.float64)
    supported = np.isfinite(s) & (s >= domain_min) & (s <= domain_max)
    values = np.full(len(s), np.nan, dtype=np.float64)
    if np.any(supported):
        values[supported] = np.asarray(spline(s[supported]), dtype=np.float64)
    return values, int(np.count_nonzero(~supported))


def _fit_session_models(
    frames: dict[tuple[str, int, int], dict[str, Any]],
    spline: BSpline,
    domain_min: float,
    domain_max: float,
    params: MeasurementParams,
) -> dict[tuple[str, int], dict[str, Any]]:
    models: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in sorted({key[0] for key in frames}, key=lambda name: DATASET_ORDER[name]):
        for position in range(1, 6):
            key = (dataset, position, 1)
            frame = frames[key]
            ground = frame["baseline"]
            s = np.asarray([point["s"] for point in ground], dtype=np.float64)
            z = np.asarray([point["z"] for point in ground], dtype=np.float64)
            g, unsupported = _g_values(spline, s, domain_min, domain_max)
            if unsupported:
                raise RuntimeError(f"Ground-3 G domain excludes calibration points: {dataset}/{position}")
            linear = _fit_fixed_s_profile(s, z, params)
            linear_g = _fit_fixed_s_profile(s, z - g, params)
            linear.unsupported_count = unsupported
            linear_g.unsupported_count = unsupported
            models[(dataset, position)] = {
                "linear": linear,
                "linear_g": linear_g,
                "calibration_repeat": 1,
            }
    return models


def _failed_result(status: str, point_count: int = 0, unsupported_count: int = 0) -> dict[str, Any]:
    return {
        "value": None,
        "median": None,
        "std": None,
        "status": status,
        "point_count": int(point_count),
        "inlier_count": 0,
        "unsupported_count": int(unsupported_count),
    }


def _measure_fixed_profile(
    height_points: list[dict[str, Any]],
    slope: float,
    intercept: float,
    spline: BSpline | None,
    domain_min: float,
    domain_max: float,
    params: MeasurementParams,
) -> dict[str, Any]:
    count = len(height_points)
    if count < params.min_height_points:
        return _failed_result(
            f"MeasurementError: height line has too few points: {count} < {params.min_height_points}",
            count,
        )
    xyz = np.asarray(
        [[point["xy"][0], point["xy"][1], point["z"]] for point in height_points],
        dtype=np.float64,
    )
    try:
        height_fit = _fit_line_xy(xyz[:, :2], params, "height line")
    except Exception as error:
        return _failed_result(f"{type(error).__name__}: {error}", count)
    inliers = xyz[height_fit.inlier_mask]
    s = (inliers[:, :2] - np.asarray([0.0, 0.0]))  # replaced below by point S values
    s = np.asarray([height_points[index]["s"] for index in np.flatnonzero(height_fit.inlier_mask)], dtype=np.float64)
    base = slope * s + intercept
    unsupported = 0
    if spline is not None:
        correction, unsupported = _g_values(spline, s, domain_min, domain_max)
        if unsupported:
            return _failed_result("unsupported_G_domain", count, unsupported)
        base = base + correction
    relative = inliers[:, 2] - base
    return {
        "value": float(np.mean(relative)),
        "median": float(np.median(relative)),
        "std": float(np.std(relative)),
        "status": "success",
        "point_count": count,
        "inlier_count": int(height_fit.inlier_mask.sum()),
        "unsupported_count": unsupported,
    }


def _measure_fixed_zero(height_points: list[dict[str, Any]], params: MeasurementParams) -> dict[str, Any]:
    count = len(height_points)
    if count < params.min_height_points:
        return _failed_result(
            f"MeasurementError: height line has too few points: {count} < {params.min_height_points}",
            count,
        )
    xyz = np.asarray(
        [[point["xy"][0], point["xy"][1], point["z"]] for point in height_points],
        dtype=np.float64,
    )
    try:
        fit = _fit_line_xy(xyz[:, :2], params, "height line")
    except Exception as error:
        return _failed_result(f"{type(error).__name__}: {error}", count)
    values = xyz[fit.inlier_mask, 2]
    return {
        "value": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "status": "success",
        "point_count": count,
        "inlier_count": int(fit.inlier_mask.sum()),
        "unsupported_count": 0,
    }


def _existing_measurements(gauge_dir: Path) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    output: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in _read_csv(gauge_dir / "height_measurements.csv"):
        if row["model"] != "C1" or row["mode"] not in {"fixed_zg_zero", "local_adjacent"}:
            continue
        key = (row["dataset"], int(row["position_rank"]), int(row["repeat_index"]), row["mode"])
        output[key] = {
            "value": _finite_float(row.get("height_mean_mm")),
            "median": _finite_float(row.get("height_median_mm")),
            "std": _finite_float(row.get("height_std_mm")),
            "status": row["status"],
            "error": row.get("error", ""),
            "point_count": int(float(row.get("height_point_count") or 0)),
            "inlier_count": None,
            "unsupported_count": 0,
        }
    return output


def _build_frame_rows(
    frames: dict[tuple[str, int, int], dict[str, Any]],
    models: dict[tuple[str, int], dict[str, Any]],
    existing: dict[tuple[str, int, int, str], dict[str, Any]],
    spline: BSpline,
    domain_min: float,
    domain_max: float,
    params: MeasurementParams,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key in sorted(frames, key=lambda item: (DATASET_ORDER[item[0]], item[1], item[2])):
        dataset, position, repeat = key
        frame = frames[key]
        truth = float(frame["truth_mm"])
        session = models[(dataset, position)]
        linear = session["linear"]
        linear_g = session["linear_g"]
        height = frame["height"]
        calculated_a = _measure_fixed_zero(height, params)
        existing_a = existing[(dataset, position, repeat, "fixed_zg_zero")]
        existing_d = existing[(dataset, position, repeat, "local_adjacent")]
        # A and D are deliberately taken from the previously audited C1 replay.
        # The A recalculation is only a consistency check on the same point set.
        if calculated_a["status"] == "success" and existing_a["status"] == "success":
            if abs(float(calculated_a["value"]) - float(existing_a["value"])) > 1.0e-9:
                raise RuntimeError(f"fixed_zg_zero mismatch at {dataset}/{position}/repeat{repeat}")
        result_b = _measure_fixed_profile(
            height, linear.slope, linear.intercept, None, domain_min, domain_max, params
        )
        result_c = _measure_fixed_profile(
            height, linear_g.slope, linear_g.intercept, spline, domain_min, domain_max, params
        )
        results = {
            "fixed_zg_zero": existing_a,
            "session_linear": result_b,
            "session_linear_G": result_c,
            "local_adjacent": existing_d,
        }
        row: dict[str, Any] = {
            "dataset": dataset,
            "truth_mm": truth,
            "pose_id": frame["pose_id"],
            "position_rank": position,
            "repeat_index": repeat,
            "filename": frame["filename"],
            "scope": "calibration_repeat1_in_sample" if repeat == 1 else "evaluation_repeat2_5",
            "session_linear_a_mm_per_mm": linear.slope,
            "session_linear_b_mm": linear.intercept,
            "session_linear_G_a_mm_per_mm": linear_g.slope,
            "session_linear_G_b_mm": linear_g.intercept,
            "ground_calibration_point_count": linear.point_count,
            "ground_calibration_inlier_count": linear.inlier_count,
            "ground_calibration_s_span_mm": linear.s_max - linear.s_min,
            "ground_calibration_rmse_mm": linear.rmse,
            "ground_calibration_G_point_count": linear_g.point_count,
            "ground_calibration_G_inlier_count": linear_g.inlier_count,
            "ground_calibration_G_s_span_mm": linear_g.s_max - linear_g.s_min,
            "ground_calibration_G_rmse_mm": linear_g.rmse,
        }
        for chain in CHAIN_NAMES:
            result = results[chain]
            value = result.get("value")
            error = None if value is None else float(value) - truth
            row[f"{chain}_height_mm"] = value
            row[f"{chain}_error_mm"] = error
            row[f"{chain}_abs_error_mm"] = None if error is None else abs(error)
            row[f"{chain}_median_mm"] = result.get("median")
            row[f"{chain}_std_mm"] = result.get("std")
            row[f"{chain}_status"] = result.get("status", "")
            row[f"{chain}_point_count"] = result.get("point_count")
            row[f"{chain}_inlier_count"] = result.get("inlier_count")
            row[f"{chain}_unsupported_count"] = result.get("unsupported_count", 0)
        row["C_minus_B_height_delta_mm"] = (
            row["session_linear_G_height_mm"] - row["session_linear_height_mm"]
            if row["session_linear_G_height_mm"] is not None and row["session_linear_height_mm"] is not None
            else None
        )
        row["C_minus_B_abs_error_improvement_mm"] = (
            row["session_linear_abs_error_mm"] - row["session_linear_G_abs_error_mm"]
            if row["session_linear_abs_error_mm"] is not None and row["session_linear_G_abs_error_mm"] is not None
            else None
        )
        row["C_minus_D_height_delta_mm"] = (
            row["session_linear_G_height_mm"] - row["local_adjacent_height_mm"]
            if row["session_linear_G_height_mm"] is not None and row["local_adjacent_height_mm"] is not None
            else None
        )
        row["C_minus_D_abs_error_gap_mm"] = (
            row["session_linear_G_abs_error_mm"] - row["local_adjacent_abs_error_mm"]
            if row["session_linear_G_abs_error_mm"] is not None and row["local_adjacent_abs_error_mm"] is not None
            else None
        )
        output.append(row)
    return output


def _metric(errors: Iterable[float], expected_count: int, limit: float) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=np.float64)
    if len(values):
        absolute = np.abs(values)
        return {
            "expected_count": expected_count,
            "count": int(len(values)),
            "failed_count": int(expected_count - len(values)),
            "bias_mm": float(np.mean(values)),
            "mae_mm": float(np.mean(absolute)),
            "rmse_mm": float(np.sqrt(np.mean(values**2))),
            "p95_mm": float(np.percentile(absolute, 95.0)),
            "max_mm": float(np.max(absolute)),
            "pass_count": int(np.count_nonzero(absolute <= limit)),
            "pass_rate": float(np.mean(absolute <= limit)),
            "limit_mm": limit,
        }
    return {
        "expected_count": expected_count,
        "count": 0,
        "failed_count": expected_count,
        "bias_mm": None,
        "mae_mm": None,
        "rmse_mm": None,
        "p95_mm": None,
        "max_mm": None,
        "pass_count": 0,
        "pass_rate": None,
        "limit_mm": limit,
    }


def _chain_metrics(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if scope == "all150" or row["repeat_index"] > 1]
    expected = len(selected)
    output = []
    for chain in CHAIN_NAMES:
        errors = [
            float(row[f"{chain}_error_mm"])
            for row in selected
            if row[f"{chain}_status"] == "success" and row[f"{chain}_error_mm"] is not None
        ]
        item = {"scope": scope, "chain": chain, **_metric(errors, expected, 0.2)}
        item["pass_0p1_count"] = int(
            sum(abs(value) <= 0.1 for value in errors)
        )
        item["pass_0p1_rate"] = float(item["pass_0p1_count"] / len(errors)) if errors else None
        output.append(item)
    return output


def _condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["repeat_index"] > 1]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[(row["dataset"], row["position_rank"])].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, position), group in sorted(groups.items(), key=lambda item: (DATASET_ORDER[item[0][0]], item[0][1])):
        truth = float(group[0]["truth_mm"])
        for chain in CHAIN_NAMES:
            values = np.asarray(
                [float(row[f"{chain}_height_mm"]) for row in group if row[f"{chain}_status"] == "success" and row[f"{chain}_height_mm"] is not None],
                dtype=np.float64,
            )
            errors = values - truth if len(values) else np.empty(0, dtype=np.float64)
            output.append(
                {
                    "dataset": dataset,
                    "truth_mm": truth,
                    "position_rank": position,
                    "chain": chain,
                    "expected_repeat2_5": len(group),
                    "successful_repeat2_5": int(len(values)),
                    "failed_repeat2_5": int(len(group) - len(values)),
                    "measured_mean_mm": float(np.mean(values)) if len(values) else None,
                    "measured_median_mm": float(np.median(values)) if len(values) else None,
                    "signed_bias_mm": float(np.mean(errors)) if len(errors) else None,
                    "mae_mm": float(np.mean(np.abs(errors))) if len(errors) else None,
                    "rmse_mm": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                    "repeatability_sigma_mm": float(np.std(values)) if len(values) else None,
                    "p95_mm": float(np.percentile(np.abs(errors), 95.0)) if len(errors) else None,
                    "max_mm": float(np.max(np.abs(errors))) if len(errors) else None,
                    "pass_0p1_rate": float(np.mean(np.abs(errors) <= 0.1)) if len(errors) else None,
                    "pass_0p2_rate": float(np.mean(np.abs(errors) <= 0.2)) if len(errors) else None,
                }
            )
    return output


def _position_ranges(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        if row["signed_bias_mm"] is not None:
            groups[(row["dataset"], row["chain"])].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, chain), group in sorted(groups.items(), key=lambda item: (DATASET_ORDER[item[0][0]], item[0][1])):
        biases = np.asarray([float(row["signed_bias_mm"]) for row in group], dtype=np.float64)
        output.append(
            {
                "dataset": dataset,
                "chain": chain,
                "successful_position_count": len(biases),
                "position_bias_min_mm": float(np.min(biases)),
                "position_bias_max_mm": float(np.max(biases)),
                "position_bias_range_mm": float(np.ptp(biases)),
                "position_bias_mae_mm": float(np.mean(np.abs(biases))),
                "pass_positions_0p1": int(sum(abs(value) <= 0.1 for value in biases)),
                "pass_positions_0p2": int(sum(abs(value) <= 0.2 for value in biases)),
            }
        )
    return output


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, selected in (
        ("repeat2_5", [row for row in rows if row["repeat_index"] > 1]),
        ("all150", rows),
    ):
        for comparison, left, right in (
            ("C_minus_B", "session_linear_G", "session_linear"),
            ("C_minus_D", "session_linear_G", "local_adjacent"),
        ):
            value_delta = []
            abs_error_delta = []
            for row in selected:
                if row[f"{left}_status"] == "success" and row[f"{right}_status"] == "success":
                    value_delta.append(float(row[f"{left}_height_mm"]) - float(row[f"{right}_height_mm"]))
                    abs_error_delta.append(float(row[f"{left}_abs_error_mm"]) - float(row[f"{right}_abs_error_mm"]))
            values = np.asarray(value_delta, dtype=np.float64)
            errors = np.asarray(abs_error_delta, dtype=np.float64)
            output.append(
                {
                    "scope": scope,
                    "comparison": comparison,
                    "count": int(len(values)),
                    "value_delta_mean_mm": float(np.mean(values)) if len(values) else None,
                    "value_delta_median_mm": float(np.median(values)) if len(values) else None,
                    "value_delta_rmse_mm": float(np.sqrt(np.mean(values**2))) if len(values) else None,
                    "abs_error_delta_mean_mm": float(np.mean(errors)) if len(errors) else None,
                    "abs_error_delta_median_mm": float(np.median(errors)) if len(errors) else None,
                    "abs_error_delta_rmse_mm": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                    "positive_abs_error_improvement_rate": float(np.mean(errors < 0.0)) if len(errors) else None,
                }
            )
    return output


def _repeatability_summary(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for chain in CHAIN_NAMES:
        values = np.asarray(
            [float(row["repeatability_sigma_mm"]) for row in condition_rows if row["chain"] == chain and row["repeatability_sigma_mm"] is not None],
            dtype=np.float64,
        )
        output.append(
            {
                "chain": chain,
                "condition_count": int(len(values)),
                "repeatability_median_sigma_mm": float(np.median(values)) if len(values) else None,
                "repeatability_p95_sigma_mm": float(np.percentile(values, 95.0)) if len(values) else None,
                "repeatability_max_sigma_mm": float(np.max(values)) if len(values) else None,
            }
        )
    return output


def _plot_height_error(output_dir: Path, condition_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        for chain in CHAIN_NAMES:
            rows = [row for row in condition_rows if row["dataset"] == dataset and row["chain"] == chain]
            rows.sort(key=lambda row: int(row["position_rank"]))
            x = np.asarray([int(row["position_rank"]) for row in rows], dtype=np.float64)
            y = np.asarray([
                float(row["signed_bias_mm"]) if row["signed_bias_mm"] is not None else np.nan
                for row in rows
            ])
            axis.plot(x, y, marker="o", linewidth=1.4, label=CHAIN_LABELS[chain])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhline(0.1, color="gray", linewidth=0.6, linestyle="--")
        axis.axhline(-0.1, color="gray", linewidth=0.6, linestyle="--")
        axis.set_title(dataset)
        axis.set_xlabel("position rank")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("mean signed height error (mm)")
    axes[1, 0].set_ylabel("mean signed height error (mm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False)
    fig.suptitle("Ground-4A height error vs position (formal metrics: repeat2–5)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "ground4a_height_error_vs_position.png", dpi=180)
    plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "MISSING"
    return f"{float(value):.{digits}f}"


def _write_report(
    output_dir: Path,
    gauge_dir: Path,
    ground3_summary_path: Path,
    ground3_summary: dict[str, Any],
    provenance: dict[str, Any],
    chain_summary: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    repeatability_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    b_d_gap_mean: float | None,
    c_d_gap_mean: float | None,
    gs_value: str,
    obstacle_baseline: str,
) -> None:
    formal = {row["chain"]: row for row in chain_summary if row["scope"] == "repeat2_5"}
    deltas = {row["comparison"]: row for row in delta_rows if row["scope"] == "repeat2_5"}
    lines = [
        "# Ground-4A 旧量块数据四链回放（retrospective diagnostic）",
        "",
        "## 结论",
        "",
        f"- `GS_HEIGHT_VALUE = {gs_value}`",
        f"- `OBSTACLE_ONLY_APPROACHES_LOCAL_BASELINE = {obstacle_baseline}`",
        "- `GROUND4A_STATUS = RETROSPECTIVE_DIAGNOSTIC_ONLY`",
        "- 本轮不修改 C0/C1、G(S)、ROI、GUI、生产测高链路，也不构成新的 held-out engineering acceptance。",
        "",
        "## 判定口径",
        "",
        "- 正式指标只使用每个 height × position condition 的 repeat2–5；repeat1 仅用于冻结该 condition 的 session calibration proxy。",
        "- `A fixed_zg_zero` 与 `D local_adjacent` 复用 frozen C1 audit 的同一帧结果；B/C 使用同一批 C1 height ROI 点和相同 XY robust height-line inlier 规则。",
        "- B：repeat1 基准面 ROI 点 robust 拟合 `Zg=a*S+b`，随后 `Zobj-(aS+b)`；C：先拟合 `Zg-G(S)=a*S+b`，随后 `Zobj-(aS+b+G(S))`。",
        "- `S=(XY-origin_xy)·direction_xy` 严格复用 Ground-1；G(S) 域外不外推、不 clamp，当前量块有效 ROI 点均在 frozen domain 内。",
        "",
        "## Artifact provenance / reuse audit",
        "",
        f"- 复用：`{gauge_dir}` 的 150 图像审计、150 次 Steger、C1 重建、30/30 manual-frozen geometry-only ROI、逐点 C1 XYZ 与既有 A/D 测量。",
        f"- 复用：Ground-3 freeze candidate `{ground3_summary_path}`；模型 `{ground3_summary['selected_model']['model_name']}`，参数 hash `{ground3_summary['selected_model']['correction_parameters_sha256']}`。",
        "- 本轮新增：30 个 condition 的 repeat1 ground proxy、B/C 四链回放、repeat2–5 正式统计、全 150 帧参考行、position range、repeatability 与图表。",
        "- 未做：Steger 重跑、C0/C1 重建或重拟合、ROI 重选、G(S) 重拟合、height linear correction、生产配置写入。",
        f"- frozen C1 config hash：`{provenance['config']['sha256']}`；`enable_laser_ray_correction=true`。",
        "",
        "## 四链正式指标（repeat2–5）",
        "",
        "| chain | n/expected | failed | Bias | MAE | RMSE | P95 | Max | ±0.1 pass | ±0.2 pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for chain in CHAIN_NAMES:
        row = formal[chain]
        lines.append(
            f"| {CHAIN_LABELS[chain]} | {row['count']}/{row['expected_count']} | {row['failed_count']} | "
            f"{_fmt(row['bias_mm'])} | {_fmt(row['mae_mm'])} | {_fmt(row['rmse_mm'])} | "
            f"{_fmt(row['p95_mm'])} | {_fmt(row['max_mm'])} | "
            f"{row['pass_0p1_count']}/{row['count']} ({_fmt(100*row['pass_0p1_rate'], 1)}%) | "
            f"{row['pass_count']}/{row['count']} ({_fmt(100*row['pass_rate'], 1)}%) |"
        )
    lines += [
        "",
        "## C-B 增量与 C-D 差距（repeat2–5）",
        "",
        "负的 `abs_error_delta` 表示左侧链路绝对误差更小。",
        "",
        "| comparison | n | value delta mean | value delta RMSE | abs-error delta mean | abs-error delta median | positive improvement rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("C_minus_B", "C_minus_D"):
        row = deltas[name]
        lines.append(
            f"| {name} | {row['count']} | {_fmt(row['value_delta_mean_mm'])} | {_fmt(row['value_delta_rmse_mm'])} | "
            f"{_fmt(row['abs_error_delta_mean_mm'])} | {_fmt(row['abs_error_delta_median_mm'])} | "
            f"{_fmt(100*row['positive_abs_error_improvement_rate'], 1)}% |"
        )
    lines += [
        "",
        f"- C 相对 B 的 MAE：`{_fmt(formal['session_linear']['mae_mm'])} -> {_fmt(formal['session_linear_G']['mae_mm'])}` mm；RMSE：`{_fmt(formal['session_linear']['rmse_mm'])} -> {_fmt(formal['session_linear_G']['rmse_mm'])}` mm。",
        f"- C 相对 D 的 MAE 差距（C abs error - D abs error）：`{_fmt(formal['session_linear_G']['mae_mm'] - formal['local_adjacent']['mae_mm'])}` mm。",
        f"- obstacle-only 与 D 的平均高度值差距：B=`{_fmt(b_d_gap_mean)}` mm，C=`{_fmt(c_d_gap_mean)}` mm；C 未缩小该差距，因此结论为 `{obstacle_baseline}`。",
        "",
        "## Position bias range（repeat2–5 condition mean）",
        "",
        "详表见 `ground4a_position_bias_ranges.csv`；每行是同一高度五个 position 的 condition bias 范围。",
        "",
        "| dataset | chain | positions | bias min | bias max | range | position MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in position_rows:
        lines.append(
            f"| {row['dataset']} | {row['chain']} | {row['successful_position_count']} | "
            f"{_fmt(row['position_bias_min_mm'])} | {_fmt(row['position_bias_max_mm'])} | "
            f"{_fmt(row['position_bias_range_mm'])} | {_fmt(row['position_bias_mae_mm'])} |"
        )
    lines += [
        "",
        "## Repeatability",
        "",
        "| chain | conditions | median sigma | P95 sigma | Max sigma |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in repeatability_rows:
        lines.append(
            f"| {CHAIN_LABELS[row['chain']]} | {row['condition_count']} | "
            f"{_fmt(row['repeatability_median_sigma_mm'])} | {_fmt(row['repeatability_p95_sigma_mm'])} | "
            f"{_fmt(row['repeatability_max_sigma_mm'])} |"
        )
    lines += [
        "",
        "## 文件",
        "",
        "- `ground4a_frame_comparison.csv`：全 150 帧；repeat1 标为 in-sample calibration，repeat2–5 标为 formal evaluation。",
        "- `ground4a_condition_comparison.csv`：30 个 condition 的 repeat2–5 汇总。",
        "- `ground4a_four_chain_summary.csv`：四链的 repeat2–5 与 all150 统计。",
        "- `ground4a_position_bias_ranges.csv`、`ground4a_repeatability_summary.csv`、`ground4a_chain_deltas.csv`。",
        "- `ground4a_height_error_vs_position.png`。",
        "",
        "## 备注",
        "",
        "- `obs_2mm / position5` 沿用原 frozen C1 audit 的 height ROI 点数不足失败状态；没有因为本轮结果删除该 condition。",
        "- 全 150 帧输出仅供回放追踪；任何正式结论均不把 repeat1 的 in-sample 结果混入 repeat2–5 指标。",
        "",
    ]
    (output_dir / "ground4a_retrospective_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge-dir", type=Path, default=DEFAULT_GAUGE_DIR)
    parser.add_argument("--ground3-summary", type=Path, default=DEFAULT_GROUND3_SUMMARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gauge_dir = args.gauge_dir.resolve()
    ground3_summary_path = args.ground3_summary.resolve()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _frame_audit_rows, provenance = _load_and_validate_provenance(gauge_dir)
    ground3_summary, spline, origin, direction = _load_ground3(ground3_summary_path)
    frames = _load_frames(gauge_dir, origin, direction)
    truth_config = json.loads((gauge_dir / "truth_config.json").read_text(encoding="utf-8"))
    truth_map = truth_config.get("truth_mm", TRUTH_DEFAULT)
    for frame in frames.values():
        frame["truth_mm"] = float(truth_map[frame["dataset"]])

    app = load_app_config(config_path)
    params = app.measurement if hasattr(app, "measurement") else MeasurementParams()
    domain_min = float(ground3_summary["selected_model"]["domain_min_mm"])
    domain_max = float(ground3_summary["selected_model"]["domain_max_mm"])
    models = _fit_session_models(frames, spline, domain_min, domain_max, params)
    existing = _existing_measurements(gauge_dir)
    rows = _build_frame_rows(frames, models, existing, spline, domain_min, domain_max, params)

    session_rows = []
    for (dataset, position), model_pair in sorted(models.items(), key=lambda item: (DATASET_ORDER[item[0][0]], item[0][1])):
        for name, model in (("session_linear", model_pair["linear"]), ("session_linear_G", model_pair["linear_g"])):
            session_rows.append(
                {
                    "dataset": dataset,
                    "position_rank": position,
                    "calibration_repeat": 1,
                    "proxy": name,
                    "a_mm_per_mm": model.slope,
                    "b_mm": model.intercept,
                    "rmse_mm": model.rmse,
                    "point_count": model.point_count,
                    "inlier_count": model.inlier_count,
                    "S_min_mm": model.s_min,
                    "S_max_mm": model.s_max,
                    "S_span_mm": model.s_max - model.s_min,
                    "unsupported_G_point_count": model.unsupported_count,
                }
            )

    condition_rows = _condition_rows(rows)
    position_rows = _position_ranges(condition_rows)
    chain_summary = _chain_metrics(rows, "repeat2_5") + _chain_metrics(rows, "all150")
    repeatability_rows = _repeatability_summary(condition_rows)
    delta_rows = _delta_rows(rows)

    formal = {row["chain"]: row for row in chain_summary if row["scope"] == "repeat2_5"}
    c_vs_b_mae = float(formal["session_linear_G"]["mae_mm"]) - float(formal["session_linear"]["mae_mm"])
    c_vs_b_rmse = float(formal["session_linear_G"]["rmse_mm"]) - float(formal["session_linear"]["rmse_mm"])
    c_vs_b_p95 = float(formal["session_linear_G"]["p95_mm"]) - float(formal["session_linear"]["p95_mm"])
    if c_vs_b_mae < -1.0e-9 and c_vs_b_rmse < -1.0e-9 and c_vs_b_p95 <= 1.0e-9:
        gs_value = "POSITIVE"
    elif c_vs_b_mae > 1.0e-9 or c_vs_b_rmse > 1.0e-9:
        gs_value = "NEGATIVE"
    else:
        gs_value = "NEUTRAL"
    delta_formal = {row["comparison"]: row for row in delta_rows if row["scope"] == "repeat2_5"}
    b_d_gap = np.asarray([
        abs(float(row["session_linear_height_mm"]) - float(row["local_adjacent_height_mm"]))
        for row in rows
        if row["repeat_index"] > 1 and row["session_linear_status"] == "success" and row["local_adjacent_status"] == "success"
    ])
    c_d_gap = np.asarray([
        abs(float(row["session_linear_G_height_mm"]) - float(row["local_adjacent_height_mm"]))
        for row in rows
        if row["repeat_index"] > 1 and row["session_linear_G_status"] == "success" and row["local_adjacent_status"] == "success"
    ])
    obstacle_baseline = "YES" if len(b_d_gap) and len(c_d_gap) and float(np.mean(c_d_gap)) < float(np.mean(b_d_gap)) else "NO"

    frame_fields = [
        "dataset", "truth_mm", "pose_id", "position_rank", "repeat_index", "filename", "scope",
        "session_linear_a_mm_per_mm", "session_linear_b_mm", "session_linear_G_a_mm_per_mm", "session_linear_G_b_mm",
        "ground_calibration_point_count", "ground_calibration_inlier_count", "ground_calibration_s_span_mm", "ground_calibration_rmse_mm",
        "ground_calibration_G_point_count", "ground_calibration_G_inlier_count", "ground_calibration_G_s_span_mm", "ground_calibration_G_rmse_mm",
    ]
    for chain in CHAIN_NAMES:
        frame_fields += [
            f"{chain}_height_mm", f"{chain}_error_mm", f"{chain}_abs_error_mm", f"{chain}_median_mm", f"{chain}_std_mm",
            f"{chain}_status", f"{chain}_point_count", f"{chain}_inlier_count", f"{chain}_unsupported_count",
        ]
    frame_fields += ["C_minus_B_height_delta_mm", "C_minus_B_abs_error_improvement_mm", "C_minus_D_height_delta_mm", "C_minus_D_abs_error_gap_mm"]
    _write_csv(output_dir / "ground4a_frame_comparison.csv", rows, frame_fields)
    _write_csv(output_dir / "ground4a_session_calibration.csv", session_rows, list(session_rows[0].keys()))
    _write_csv(output_dir / "ground4a_condition_comparison.csv", condition_rows, list(condition_rows[0].keys()))
    _write_csv(output_dir / "ground4a_four_chain_summary.csv", chain_summary, list(chain_summary[0].keys()))
    _write_csv(output_dir / "ground4a_position_bias_ranges.csv", position_rows, list(position_rows[0].keys()))
    _write_csv(output_dir / "ground4a_repeatability_summary.csv", repeatability_rows, list(repeatability_rows[0].keys()))
    _write_csv(output_dir / "ground4a_chain_deltas.csv", delta_rows, list(delta_rows[0].keys()))
    _plot_height_error(output_dir, condition_rows)

    summary = {
        "schema_version": 1,
        "classification": {
            "GS_HEIGHT_VALUE": gs_value,
            "OBSTACLE_ONLY_APPROACHES_LOCAL_BASELINE": obstacle_baseline,
            "GROUND4A_STATUS": "RETROSPECTIVE_DIAGNOSTIC_ONLY",
            "formal_scope": "repeat2_5",
        },
        "formal_chain_summary": formal,
        "all150_chain_summary": {row["chain"]: row for row in chain_summary if row["scope"] == "all150"},
        "delta_summary": {
            "repeat2_5": delta_formal,
            "all150": {row["comparison"]: row for row in delta_rows if row["scope"] == "all150"},
        },
        "obstacle_local_gap": {
            "B_to_D_mean_abs_height_gap_mm": float(np.mean(b_d_gap)) if len(b_d_gap) else None,
            "C_to_D_mean_abs_height_gap_mm": float(np.mean(c_d_gap)) if len(c_d_gap) else None,
        },
        "provenance": {
            "gauge_audit_dir": gauge_dir,
            "gauge_audit_dir_sha256_not_applicable": True,
            "gauge_pointwise_sha256": _sha256(gauge_dir / "pointwise_diagnostics.csv"),
            "gauge_height_measurements_sha256": _sha256(gauge_dir / "height_measurements.csv"),
            "gauge_roi_registry_sha256": _sha256(gauge_dir / "roi_registry.json"),
            "gauge_provenance_sha256": _sha256(gauge_dir / "provenance.json"),
            "ground3_summary_path": ground3_summary_path,
            "ground3_summary_sha256": _sha256(ground3_summary_path),
            "ground3_model_name": ground3_summary["selected_model"]["model_name"],
            "ground3_parameter_sha256": ground3_summary["selected_model"]["correction_parameters_sha256"],
            "origin_xy": origin,
            "direction_xy": direction,
            "s_domain_mm": [domain_min, domain_max],
            "reuse": [
                "150-image input audit and one-Steger-per-frame result",
                "manual-frozen geometry-only ROI registry (30/30)",
                "C1 pointwise reconstructed XYZ and existing C1 fixed_zg_zero/local_adjacent measurements",
                "Ground-3 frozen 3-interior-knot G(S) candidate",
            ],
            "newly_computed": [
                "repeat1-only per-condition robust session linear proxies",
                "session_linear and session_linear_G replay",
                "repeat2-5 formal four-chain metrics and plots",
            ],
            "no_production_change": True,
        },
    }
    _write_json(output_dir / "ground4a_summary.json", summary)
    _write_report(
        output_dir,
        gauge_dir,
        ground3_summary_path,
        ground3_summary,
        provenance,
        chain_summary,
        condition_rows,
        position_rows,
        repeatability_rows,
        delta_rows,
        float(np.mean(b_d_gap)) if len(b_d_gap) else None,
        float(np.mean(c_d_gap)) if len(c_d_gap) else None,
        gs_value,
        obstacle_baseline,
    )
    print(json.dumps({"output": str(output_dir), "GS_HEIGHT_VALUE": gs_value, "OBSTACLE_ONLY_APPROACHES_LOCAL_BASELINE": obstacle_baseline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
