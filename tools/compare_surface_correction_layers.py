"""Compare q2-only height-layer and lambda-layer surface corrections.

This is a diagnostic replay only.  It reuses the frozen C0/C1, one-pass
Steger outputs, geometry-only/manual ROI decisions, q2 definition, and the
session-linear ground-proxy protocol.  It never writes calibration or
production configuration.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, QhullError


ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = ROOT / "laser_measurement_tool"
TOOLS_ROOT = ROOT / "tools"
for item in (MEASUREMENT_ROOT, TOOLS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from measurement.height_measure import MeasurementParams
from replay_daheng_ground4a import _fit_fixed_s_profile
import analyze_surface1a as surface1a


BASE = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
SURFACE1A = BASE / "surface1a"
SURFACE2 = BASE / "surface2"
SURFACE2B = SURFACE2 / "surface2b"
BR2 = SURFACE2 / "surface2br2"
MANUAL_FROZEN = BASE.parent / "daheng_c1_gauge_blocks_20260819_manual_frozen"
HEIGHT50 = BASE / "height50_heldout"
DEFAULT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_OUTPUT = SURFACE2 / "correction_layer"

OLD_DATASETS = ("obs_1mm", "obs_2mm", "obs_6mm", "obs_10mm", "obs_20mm")
SURFACE2_DATASETS = ("obs_30mm", "obs_36mm", "obs_40mm", "obs_46mm")
ALL_DEV_DATASETS = OLD_DATASETS + SURFACE2_DATASETS
ALL_DATASETS = ALL_DEV_DATASETS + ("obs_50mm",)
NOMINAL_HEIGHT = {
    "obs_1mm": 1.0,
    "obs_2mm": 2.0,
    "obs_6mm": 6.0,
    "obs_10mm": 10.0,
    "obs_20mm": 20.0,
    "obs_30mm": 30.0,
    "obs_36mm": 36.0,
    "obs_40mm": 40.0,
    "obs_46mm": 46.0,
    "obs_50mm": 50.0,
}
HEIGHT_ORDER = (1.0, 2.0, 6.0, 10.0, 20.0, 30.0, 36.0, 40.0, 46.0)
HEIGHT_BANDS = {
    "low_1_2_6_10": (1.0, 2.0, 6.0, 10.0),
    "mid_20_30": (20.0, 30.0),
    "high_36_40_46": (36.0, 40.0, 46.0),
}
DEV_SCHEMES = ("LOHO_height", "LOPO_position_rank", "LOBO_height_band")
LAYERS = ("RAW", "H-B2", "L-B2")
METRIC_NAMES = ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm")
EPS = 1.0e-12


@dataclass
class PointSet:
    q1: np.ndarray
    q2: np.ndarray
    rays: np.ndarray
    lambda_c1: np.ndarray
    ground: np.ndarray
    dground_dlambda: np.ndarray
    c1_clamped: np.ndarray
    raw_residual: np.ndarray | None
    source_count: int


@dataclass
class Condition:
    condition_id: str
    dataset: str
    nominal_height_mm: float
    true_height_mm: float
    position_rank: int
    baseline: PointSet
    formal: PointSet
    raw_proxy: Any


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        raise RuntimeError(f"no rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def f(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite numeric value: {value}")
    return result


def condition_id(dataset: str, rank: int) -> str:
    return f"{dataset}/rank{int(rank)}"


def height_group(value: float) -> str:
    return f"height_{value:g}mm"


def metric_values(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if not len(data):
        return {name: float("nan") for name in METRIC_NAMES}
    absolute = np.abs(data)
    return {
        "bias_mm": float(np.mean(data)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(data * data))),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "max_abs_mm": float(np.max(absolute)),
    }


def distribution(values: Iterable[float], prefix: str) -> dict[str, Any]:
    data = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if not len(data):
        return {f"{prefix}_{key}": None for key in ("min", "p05", "median", "p95", "max", "count")}
    return {
        f"{prefix}_min": float(np.min(data)),
        f"{prefix}_p05": float(np.percentile(data, 5.0)),
        f"{prefix}_median": float(np.median(data)),
        f"{prefix}_p95": float(np.percentile(data, 95.0)),
        f"{prefix}_max": float(np.max(data)),
        f"{prefix}_count": int(len(data)),
    }


def validate_roi_registry(path: Path, expected_count: int, expected_datasets: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    registry = read_json(path)
    entries = registry.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise RuntimeError(f"ROI registry {path} is not {expected_count} entries")
    summary = registry.get("summary", {})
    manually_confirmed = (
        registry.get("manual_confirmed") is True
        or summary.get("manual_confirmed") is True
    )
    if not manually_confirmed:
        raise RuntimeError(f"ROI registry is not manually confirmed: {path}")
    confirmed_count = registry.get("manual_confirmed_count", summary.get("manual_confirmed_count", 0))
    if int(confirmed_count) != expected_count:
        raise RuntimeError(f"ROI registry confirmation count mismatch: {path}")
    if any("manual_confirmed" in entry and entry.get("manual_confirmed") is not True for entry in entries):
        raise RuntimeError(f"ROI registry contains unconfirmed entry: {path}")
    result = {}
    for entry in entries:
        dataset = str(entry.get("dataset", "obs_50mm" if expected_datasets == {"obs_50mm"} else ""))
        pose = str(entry["pose_id"])
        if dataset not in expected_datasets:
            raise RuntimeError(f"unexpected dataset in ROI registry: {dataset}")
        result[(dataset, pose)] = entry
    if len(result) != expected_count:
        raise RuntimeError(f"duplicate ROI keys: {path}")
    return result


def load_inputs(config_path: Path) -> dict[str, Any]:
    paths = {
        "config": config_path,
        "surface1a_points": SURFACE1A / "surface1a_points.csv",
        "surface1a_summary": SURFACE1A / "surface1a_summary.json",
        "surface_coordinate": SURFACE1A / "surface_coordinate_definition.json",
        "surface2b_samples": SURFACE2B / "surface2b_samples.csv",
        "surface2br2_condition_table": BR2 / "surface2br2_condition_table.csv",
        "surface2br2_predictions": BR2 / "surface2br2_condition_predictions.csv",
        "surface2br2_cv_metrics": BR2 / "surface2br2_cv_metrics.csv",
        "surface2br2_coefficients": BR2 / "surface2br2_coefficients.csv",
        "surface2br2_stability": BR2 / "surface2br2_coefficient_stability.csv",
        "surface2br2_summary": BR2 / "surface2br2_summary.json",
        "model_selection_summary": SURFACE2 / "surface2_model_selection" / "surface2_model_selection_summary.json",
        "pointwise_diagnostics": MANUAL_FROZEN / "pointwise_diagnostics.csv",
        "surface2_center_cache": SURFACE2 / "surface2_center_cache.csv",
        "surface2_roi": SURFACE2 / "manual_roi" / "roi_registry_manual.json",
        "height50_roi": HEIGHT50 / "height50_manual_roi_registry.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required provenance/input artifact: " + ", ".join(missing))

    app = load_app_config(config_path)
    if app.reconstruction.enable_laser_ray_correction is not True:
        raise RuntimeError("enable_laser_ray_correction must be true")
    if app.calibration.laser_ray_correction is None:
        raise RuntimeError("Frozen C1 path is missing")
    if app.calibration.ground_u_compensation is not None:
        raise RuntimeError("This replay requires the frozen null ground_u_compensation path")
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_plane,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        laser_ray_correction=app.calibration.laser_ray_correction,
    )
    correction = calibration["laser_ray_correction"]
    laser_model = calibration["laser_model"]
    if laser_model.get("model_type") != "quadratic_graph":
        raise RuntimeError("Frozen C0 is not quadratic_graph")

    s1_summary = read_json(paths["surface1a_summary"])
    coordinate = read_json(paths["surface_coordinate"])
    model_norm = laser_model["normalization"]
    if list(laser_model["independent_axes"]) != list(coordinate["independent_axes"]):
        raise RuntimeError("q independent axes differ from frozen Surface-1A definition")
    for key in ("independent_center_mm", "independent_scale_mm"):
        if not np.allclose(model_norm[key], coordinate[key], atol=1e-12):
            raise RuntimeError(f"q coordinate {key} differs from frozen definition")
    selection = read_json(paths["model_selection_summary"])
    if selection.get("SELECTED_SURFACE_MODEL") != "B2" or selection.get("Q1_RETAINED") != "NO":
        raise RuntimeError("historical Surface model selection is not frozen B2/q1=no")
    br2_summary = read_json(paths["surface2br2_summary"])
    historical = br2_summary.get("original_surface2b_conclusion", {})
    if historical.get("Q2_GAP_FILLED") != "NO" or historical.get("SURFACE2C_ALLOWED") != "NO":
        raise RuntimeError("historical Surface-2B conclusion must remain Q2_GAP_FILLED=NO/SURFACE2C_ALLOWED=NO")

    roi2 = validate_roi_registry(paths["surface2_roi"], 15, set(SURFACE2_DATASETS))
    roi50 = validate_roi_registry(paths["height50_roi"], 5, {"obs_50mm"})
    surface_definition = s1_summary["surface_definition"]
    origin = np.asarray(surface_definition["ground_origin_xy"], dtype=np.float64)
    direction = np.asarray(surface_definition["ground_direction_xy"], dtype=np.float64)
    if origin.shape != (2,) or direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-8):
        raise RuntimeError("Frozen Ground-1 origin/direction is invalid")
    transform = surface1a.build_ground_transform(calibration["R"], calibration["t"])
    if not np.all(np.isfinite(transform)) or transform.shape != (4, 4):
        raise RuntimeError("invalid camera-to-ground transform")

    current_hashes = {
        "config_sha256": sha256(config_path),
        "quadratic_c0_sha256": sha256(app.calibration.laser_plane),
        "frozen_c1_sha256": sha256(app.calibration.laser_ray_correction),
        "intrinsics_sha256": sha256(app.calibration.intrinsics),
        "extrinsics_sha256": sha256(app.calibration.extrinsics),
    }
    return {
        "paths": paths,
        "app": app,
        "calibration": calibration,
        "correction": correction,
        "laser_model": laser_model,
        "origin": origin,
        "direction": direction,
        "transform": transform,
        "roi2": roi2,
        "roi50": roi50,
        "surface1a_summary": s1_summary,
        "surface2br2_summary": br2_summary,
        "model_selection_summary": selection,
        "hashes": current_hashes,
        "surface2br2_input_hashes": {
            name: sha256(paths[name])
            for name in (
                "surface2br2_condition_table",
                "surface2br2_predictions",
                "surface2br2_cv_metrics",
                "surface2br2_coefficients",
                "surface2br2_stability",
                "surface2br2_summary",
            )
        },
    }


def pointset_from_geometry(
    geometry: Mapping[str, np.ndarray],
    transform: np.ndarray,
    laser_model: Mapping[str, Any],
    raw_residual: np.ndarray | None = None,
) -> PointSet:
    valid = np.asarray(geometry["valid"], dtype=bool)
    if not np.any(valid):
        raise RuntimeError("geometry evaluation returned no valid points")
    p0 = np.asarray(geometry["P_c0"], dtype=np.float64)
    coords = surface1a.surface_coordinates(p0, laser_model)
    rays = np.asarray(geometry["rays"], dtype=np.float64)
    dground = (transform[:3, :3] @ rays.T).T
    ground = np.asarray(geometry["ground"], dtype=np.float64)
    q1 = np.asarray(coords["q1"], dtype=np.float64)
    q2 = np.asarray(coords["q2"], dtype=np.float64)
    residual = None if raw_residual is None else np.asarray(raw_residual, dtype=np.float64)
    if residual is not None and len(residual) != len(valid):
        raise RuntimeError("raw residual length does not match geometry point count")
    return PointSet(
        q1=q1[valid],
        q2=q2[valid],
        rays=rays[valid],
        lambda_c1=np.asarray(geometry["lambda_c1"], dtype=np.float64)[valid],
        ground=ground[valid],
        dground_dlambda=dground[valid],
        c1_clamped=np.asarray(geometry["c1_clamped"], dtype=bool)[valid],
        raw_residual=None if residual is None else residual[valid],
        source_count=int(np.count_nonzero(valid)),
    )


def pointset_from_formal_rows(
    rows: list[dict[str, str]],
    transform: np.ndarray,
    laser_model: Mapping[str, Any],
) -> PointSet:
    if not rows:
        raise RuntimeError("formal condition has no analysis rows")
    xn = np.asarray([f(row["xn"]) for row in rows], dtype=np.float64)
    yn = np.asarray([f(row["yn"]) for row in rows], dtype=np.float64)
    rays = np.column_stack((xn, yn, np.ones(len(rows), dtype=np.float64)))
    lambda_c0 = np.asarray([f(row["lambda_c0"]) for row in rows], dtype=np.float64)
    lambda_c1 = np.asarray([f(row["lambda_c1"]) for row in rows], dtype=np.float64)
    p0 = rays * lambda_c0[:, None]
    coords = surface1a.surface_coordinates(p0, laser_model)
    stored_q2 = np.asarray([f(row["q2"]) for row in rows], dtype=np.float64)
    if not np.allclose(stored_q2, coords["q2"], atol=2.0e-8, rtol=0.0):
        raise RuntimeError("formal row q2 does not match Frozen-C0 intrinsic q2")
    ground = np.asarray(
        [[f(row["Xg"]), f(row["Yg"]), f(row["Zg"])] for row in rows],
        dtype=np.float64,
    )
    dground = (transform[:3, :3] @ rays.T).T
    return PointSet(
        q1=np.asarray([f(row["q1"]) for row in rows], dtype=np.float64),
        q2=stored_q2,
        rays=rays,
        lambda_c1=lambda_c1,
        ground=ground,
        dground_dlambda=dground,
        c1_clamped=np.asarray([as_bool(row.get("C1_s_clamped")) for row in rows], dtype=bool),
        raw_residual=np.asarray([f(row["height_residual_mm"]) for row in rows], dtype=np.float64),
        source_count=len(rows),
    )


def fit_raw_proxy(
    pointset: PointSet,
    origin: np.ndarray,
    direction: np.ndarray,
    measurement_params: MeasurementParams,
) -> Any:
    S = (pointset.ground[:, :2] - origin[None, :]) @ direction
    return _fit_fixed_s_profile(S, pointset.ground[:, 2], measurement_params)


def roi_height_mask(pixels: np.ndarray, roi: Mapping[str, Any], region: str) -> np.ndarray:
    v = np.asarray(pixels[:, 1], dtype=np.float64)
    if region == "height":
        lo, hi = roi["height_v_range"]
        return (v >= float(lo)) & (v <= float(hi))
    if region == "baseline":
        ranges = roi["baseline_v_ranges"]
        return (
            ((v >= float(ranges[0][0])) & (v <= float(ranges[0][1])))
            | ((v >= float(ranges[1][0])) & (v <= float(ranges[1][1])))
        )
    raise ValueError(region)


def load_old_baselines(
    path: Path,
    calibration: Mapping[str, Any],
    params: Any,
    correction: Any,
    transform: np.ndarray,
    laser_model: Mapping[str, Any],
) -> dict[tuple[str, int], PointSet]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if row.get("dataset") not in set(OLD_DATASETS) | {"obs_30mm"}:
            continue
        if int(row.get("repeat_index", "0")) != 1:
            continue
        if row.get("image_region") not in {"baseline_before", "baseline_after"}:
            continue
        if row.get("c1_status") != "valid":
            continue
        groups[(row["dataset"], int(row["position_rank"]))].append(row)
    result: dict[tuple[str, int], PointSet] = {}
    for key, rows in sorted(groups.items()):
        pixels = np.asarray([[f(row["u_px"]), f(row["v_px"])] for row in rows], dtype=np.float64)
        geometry = surface1a.evaluate_geometry(pixels, calibration, params, correction)
        pointset = pointset_from_geometry(geometry, transform, laser_model)
        if pointset.source_count < 10:
            raise RuntimeError(f"too few frozen repeat1 baseline points: {key}")
        result[key] = pointset
    return result


def load_surface2_cache(path: Path) -> dict[tuple[str, str, int], np.ndarray]:
    groups: dict[tuple[str, str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for row in read_csv(path):
        key = (row["dataset"], str(row["pose_id"]), int(row["repeat_index"]))
        groups[key].append((int(row["point_index"]), f(row["u_px"]), f(row["v_px"])))
    expected = {
        (dataset, pose, repeat)
        for dataset in ("obs_36mm", "obs_40mm", "obs_46mm")
        for pose in (f"{index:03d}" for index in range(1, 6))
        for repeat in range(1, 6)
    }
    if set(groups) != expected:
        missing = sorted(expected - set(groups))
        raise RuntimeError(f"Surface-2 center cache key mismatch; missing={missing[:5]}")
    result = {}
    for key, values in groups.items():
        values.sort(key=lambda item: item[0])
        result[key] = np.asarray([(value[1], value[2]) for value in values], dtype=np.float64)
    return result


def load_new_baselines(
    inputs: dict[str, Any],
) -> dict[tuple[str, int], PointSet]:
    calibration = inputs["calibration"]
    params = inputs["app"].reconstruction
    correction = inputs["correction"]
    transform = inputs["transform"]
    result: dict[tuple[str, int], PointSet] = {}
    cache = load_surface2_cache(inputs["paths"]["surface2_center_cache"])
    for (dataset, pose), roi in inputs["roi2"].items():
        pixels_all = cache[(dataset, pose, 1)]
        selected = pixels_all[roi_height_mask(pixels_all, roi, "baseline")]
        geometry = surface1a.evaluate_geometry(selected, calibration, params, correction)
        pointset = pointset_from_geometry(geometry, transform, inputs["laser_model"])
        result[(dataset, int(roi["position_rank"]))] = pointset

    pose_to_rank = {}
    for row in read_csv(inputs["paths"]["surface2b_samples"]):
        if row.get("dataset") == "obs_50mm" and row.get("pose_id"):
            pose_to_rank[str(row["pose_id"])] = int(row["position_rank"])
    if set(pose_to_rank) != {f"{index:03d}" for index in range(1, 6)}:
        raise RuntimeError("50mm formal rows do not provide a complete pose-to-q1-rank mapping")
    for (_, pose), roi in inputs["roi50"].items():
        cache_path = HEIGHT50 / "center_cache" / f"laser{pose}_{1:02d}.npy"
        if not cache_path.is_file():
            raise RuntimeError(f"missing 50mm repeat1 center cache: {cache_path}")
        pixels_all = np.load(cache_path, allow_pickle=False).astype(np.float64, copy=False).reshape(-1, 2)
        selected = pixels_all[roi_height_mask(pixels_all, roi, "baseline")]
        geometry = surface1a.evaluate_geometry(selected, calibration, params, correction)
        pointset = pointset_from_geometry(geometry, transform, inputs["laser_model"])
        result[("obs_50mm", pose_to_rank[pose])] = pointset
    if len(result) != 20:
        raise RuntimeError(f"expected 20 new/heldout baseline conditions, got {len(result)}")
    return result


def load_formal_rows(inputs: dict[str, Any]) -> dict[tuple[str, int], PointSet]:
    result_rows: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(inputs["paths"]["surface1a_points"]):
        if row.get("dataset") not in OLD_DATASETS:
            continue
        if row.get("split_role") != "development_formal_repeat2_5":
            continue
        if not (as_bool(row.get("height_measurement_inlier")) and as_bool(row.get("jacobian_valid"))):
            continue
        result_rows[(row["dataset"], int(row["position_rank"]))].append(row)
    for row in read_csv(inputs["paths"]["surface2b_samples"]):
        if row.get("dataset") not in set(SURFACE2_DATASETS) | {"obs_50mm"}:
            continue
        if not as_bool(row.get("analysis_included")):
            continue
        result_rows[(row["dataset"], int(row["position_rank"]))].append(row)
    result = {}
    for key, rows in sorted(result_rows.items()):
        result[key] = pointset_from_formal_rows(rows, inputs["transform"], inputs["laser_model"])
    expected_dev = {
        (dataset, rank)
        for dataset in ALL_DEV_DATASETS
        for rank in range(1, 6)
    } - {("obs_2mm", 5)}
    expected_50 = {("obs_50mm", rank) for rank in range(1, 6)}
    if set(result) != expected_dev | expected_50:
        missing = sorted((expected_dev | expected_50) - set(result))
        extra = sorted(set(result) - (expected_dev | expected_50))
        raise RuntimeError(f"formal condition keys mismatch; missing={missing}, extra={extra}")
    return result


def build_conditions(inputs: dict[str, Any]) -> tuple[dict[str, Condition], list[dict[str, Any]]]:
    measurement_params = inputs["app"].measurement
    if not isinstance(measurement_params, MeasurementParams):
        measurement_params = MeasurementParams()
    baselines = load_old_baselines(
        inputs["paths"]["pointwise_diagnostics"],
        inputs["calibration"],
        inputs["app"].reconstruction,
        inputs["correction"],
        inputs["transform"],
        inputs["laser_model"],
    )
    baselines.update(load_new_baselines(inputs))
    formal = load_formal_rows(inputs)
    conditions: dict[str, Condition] = {}
    consistency_rows: list[dict[str, Any]] = []
    for key, formal_ps in sorted(formal.items()):
        dataset, rank = key
        if key not in baselines:
            raise RuntimeError(f"missing repeat1 baseline for {key}")
        baseline_ps = baselines[key]
        proxy = fit_raw_proxy(baseline_ps, inputs["origin"], inputs["direction"], measurement_params)
        condition_key = condition_id(dataset, rank)
        true_height = float(NOMINAL_HEIGHT[dataset])
        if dataset == "obs_1mm":
            true_height = 1.001
        conditions[condition_key] = Condition(
            condition_id=condition_key,
            dataset=dataset,
            nominal_height_mm=NOMINAL_HEIGHT[dataset],
            true_height_mm=true_height,
            position_rank=rank,
            baseline=baseline_ps,
            formal=formal_ps,
            raw_proxy=proxy,
        )
        S = (formal_ps.ground[:, :2] - inputs["origin"][None, :]) @ inputs["direction"]
        recomputed = formal_ps.ground[:, 2] - (float(proxy.slope) * S + float(proxy.intercept)) - true_height
        artifact = np.asarray(formal_ps.raw_residual, dtype=np.float64)
        consistency_rows.append({
            "condition_id": condition_key,
            "dataset": dataset,
            "true_height_mm": true_height,
            "position_rank": rank,
            "baseline_point_count": baseline_ps.source_count,
            "formal_point_count": formal_ps.source_count,
            "raw_proxy_a_mm_per_mm": float(proxy.slope),
            "raw_proxy_b_mm": float(proxy.intercept),
            "raw_proxy_rmse_mm": float(proxy.rmse),
            "raw_proxy_S_span_mm": float(proxy.s_max - proxy.s_min),
            "max_abs_raw_residual_replay_difference_mm": float(np.max(np.abs(recomputed - artifact))),
            "mean_raw_residual_replay_difference_mm": float(np.mean(recomputed - artifact)),
            "formal_c1_clamp_count": int(np.count_nonzero(formal_ps.c1_clamped)),
            "formal_c1_clamp_rate": float(np.mean(formal_ps.c1_clamped)),
            "baseline_c1_clamp_count": int(np.count_nonzero(baseline_ps.c1_clamped)),
            "baseline_c1_clamp_rate": float(np.mean(baseline_ps.c1_clamped)),
        })
    max_difference = max(row["max_abs_raw_residual_replay_difference_mm"] for row in consistency_rows)
    if max_difference > 2.0e-3:
        raise RuntimeError(f"raw ground-proxy replay differs from frozen residuals by {max_difference:.6g} mm")
    return conditions, consistency_rows


def transform_lambda(
    pointset: PointSet,
    beta: np.ndarray,
    params: Any,
    transform: np.ndarray,
) -> dict[str, Any]:
    beta = np.asarray(beta, dtype=np.float64).reshape(2)
    delta = beta[0] + beta[1] * pointset.q2
    lambda_new = pointset.lambda_c1 + delta
    camera_points_new = pointset.rays * lambda_new[:, None]
    homogeneous = np.column_stack((camera_points_new, np.ones(len(camera_points_new), dtype=np.float64)))
    ground_new = (transform @ homogeneous.T).T[:, :3]
    finite_mask = np.isfinite(lambda_new) & np.isfinite(ground_new).all(axis=1)
    valid = (
        finite_mask
        & (lambda_new > 0.0)
        & (lambda_new >= float(params.min_camera_depth_mm))
        & (lambda_new <= float(params.max_camera_depth_mm))
    )
    return {
        "delta_lambda": delta,
        "lambda_new": lambda_new,
        "ground_new": ground_new,
        "valid": valid,
        "invalid_count": int(np.count_nonzero(~valid)),
        "delta_Zg": pointset.dground_dlambda[:, 2] * delta,
    }


def fit_proxy_for_beta(
    condition: Condition,
    beta: np.ndarray,
    inputs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    params = inputs["app"].reconstruction
    transformed = transform_lambda(condition.baseline, beta, params, inputs["transform"])
    valid = transformed["valid"]
    if int(np.count_nonzero(valid)) < 10:
        raise RuntimeError(f"lambda-layer baseline became invalid: {condition.condition_id}")
    ground = transformed["ground_new"][valid]
    S = (ground[:, :2] - inputs["origin"][None, :]) @ inputs["direction"]
    proxy = _fit_fixed_s_profile(S, ground[:, 2], inputs["measurement_params"])
    return proxy, transformed


def evaluate_lambda_condition(
    condition: Condition,
    beta: np.ndarray,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    params = inputs["app"].reconstruction
    proxy, baseline_transformed = fit_proxy_for_beta(condition, beta, inputs)
    formal_transformed = transform_lambda(condition.formal, beta, params, inputs["transform"])
    valid = formal_transformed["valid"]
    if not np.any(valid):
        return {
            "condition_id": condition.condition_id,
            "proxy": proxy,
            "baseline": baseline_transformed,
            "formal": formal_transformed,
            "valid": False,
            "residual": np.asarray([], dtype=np.float64),
            "invalid_count": int(formal_transformed["invalid_count"]),
        }
    ground = formal_transformed["ground_new"][valid]
    S = (ground[:, :2] - inputs["origin"][None, :]) @ inputs["direction"]
    residual = ground[:, 2] - (float(proxy.slope) * S + float(proxy.intercept)) - condition.true_height_mm
    return {
        "condition_id": condition.condition_id,
        "proxy": proxy,
        "baseline": baseline_transformed,
        "formal": formal_transformed,
        "valid": bool(np.all(valid)),
        "residual": residual,
        "invalid_count": int(formal_transformed["invalid_count"]),
        "valid_mask": valid,
    }


def lambda_residual_vector(
    conditions: list[Condition],
    beta: np.ndarray,
    inputs: dict[str, Any],
    require_valid: bool = True,
) -> tuple[np.ndarray, dict[str, dict[str, Any]], bool]:
    vectors: list[np.ndarray] = []
    evaluations: dict[str, dict[str, Any]] = {}
    all_valid = True
    for condition in conditions:
        try:
            evaluation = evaluate_lambda_condition(condition, beta, inputs)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            evaluation = {
                "condition_id": condition.condition_id,
                "valid": False,
                "invalid_count": condition.baseline.source_count + condition.formal.source_count,
                "residual": np.asarray([], dtype=np.float64),
                "error": f"{type(error).__name__}: {error}",
            }
        evaluations[condition.condition_id] = evaluation
        if not evaluation["valid"] or evaluation["invalid_count"]:
            all_valid = False
        residual = np.asarray(evaluation["residual"], dtype=np.float64)
        if not len(residual):
            all_valid = False
            continue
        if require_valid and not evaluation["valid"]:
            continue
        vectors.append(residual / math.sqrt(len(residual)))
    if not vectors:
        return np.asarray([], dtype=np.float64), evaluations, all_valid
    return np.concatenate(vectors), evaluations, all_valid


def fit_height_model(conditions: list[Condition]) -> dict[str, Any]:
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for condition in conditions:
        q2 = condition.formal.q2
        residual = np.asarray(condition.formal.raw_residual, dtype=np.float64)
        matrices.append(np.column_stack((np.ones(len(q2)), q2)) / math.sqrt(len(q2)))
        targets.append(residual / math.sqrt(len(residual)))
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    beta, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    return {
        "layer": "H-B2",
        "beta": np.asarray(beta, dtype=np.float64),
        "design_rank": int(rank),
        "design_condition_number": float(np.linalg.cond(matrix)),
        "singular_values": np.asarray(singular, dtype=np.float64),
        "fit_status": "success",
        "iterations": 1,
        "objective": float(np.mean((matrix @ beta - target) ** 2)),
    }


def fit_lambda_model(
    conditions: list[Condition],
    inputs: dict[str, Any],
    max_iterations: int = 10,
) -> dict[str, Any]:
    beta = np.zeros(2, dtype=np.float64)
    residual, evaluations, valid = lambda_residual_vector(conditions, beta, inputs)
    if not valid or not len(residual):
        return {
            "layer": "L-B2",
            "beta": beta,
            "fit_status": "initial_invalid",
            "design_rank": 0,
            "design_condition_number": float("inf"),
            "singular_values": np.asarray([], dtype=np.float64),
            "iterations": 0,
            "objective": float("inf"),
            "evaluations": evaluations,
        }
    objective = float(np.mean(residual * residual))
    status = "success"
    rank = 0
    condition_number = float("inf")
    singular = np.asarray([], dtype=np.float64)
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        jacobian_columns: list[np.ndarray] = []
        for column in range(2):
            trial_beta = beta.copy()
            trial_beta[column] += 1.0e-4
            trial_residual, _, trial_valid = lambda_residual_vector(
                conditions, trial_beta, inputs
            )
            if not trial_valid or len(trial_residual) != len(residual):
                status = "finite_difference_invalid"
                break
            jacobian_columns.append((trial_residual - residual) / 1.0e-4)
        if len(jacobian_columns) != 2:
            break
        jacobian = np.column_stack(jacobian_columns)
        step, _, rank, singular = np.linalg.lstsq(jacobian, -residual, rcond=None)
        rank = int(rank)
        condition_number = float(np.linalg.cond(jacobian))
        if not np.all(np.isfinite(step)):
            status = "nonfinite_step"
            break
        if float(np.linalg.norm(step)) < 1.0e-8:
            break
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = beta + scale * step
            candidate_residual, candidate_evaluations, candidate_valid = lambda_residual_vector(
                conditions, candidate, inputs
            )
            if candidate_valid and len(candidate_residual) == len(residual):
                candidate_objective = float(np.mean(candidate_residual * candidate_residual))
                if candidate_objective <= objective + 1.0e-14:
                    beta = candidate
                    residual = candidate_residual
                    evaluations = candidate_evaluations
                    objective = candidate_objective
                    accepted = True
                    break
        if not accepted:
            break
    if rank < 2:
        status = "rank_deficient" if status == "success" else status
    return {
        "layer": "L-B2",
        "beta": beta,
        "fit_status": status,
        "design_rank": rank,
        "design_condition_number": condition_number,
        "singular_values": singular,
        "iterations": iterations,
        "objective": objective,
        "evaluations": evaluations,
    }


def apply_height_residual(condition: Condition, beta: np.ndarray) -> np.ndarray:
    return np.asarray(condition.formal.raw_residual, dtype=np.float64) - (
        float(beta[0]) + float(beta[1]) * condition.formal.q2
    )


def support_from_q(train: list[Condition], test: Condition) -> dict[str, Any]:
    train_q = np.vstack([
        np.column_stack((condition.formal.q1, condition.formal.q2))
        for condition in train
    ])
    test_q = np.column_stack((test.formal.q1, test.formal.q2))
    lower = np.min(train_q, axis=0)
    upper = np.max(train_q, axis=0)
    bbox_inside = np.all((test_q >= lower - EPS) & (test_q <= upper + EPS), axis=1)
    unique = np.unique(train_q, axis=0)
    if len(unique) >= 3:
        try:
            hull_inside = Delaunay(unique).find_simplex(test_q) >= 0
        except (QhullError, ValueError, np.linalg.LinAlgError):
            hull_inside = bbox_inside.copy()
    else:
        hull_inside = bbox_inside.copy()
    state = (
        "IN_DOMAIN"
        if np.all(bbox_inside) and np.all(hull_inside)
        else "BBOX_EXTRAPOLATION"
        if np.any(~bbox_inside)
        else "HULL_EXTRAPOLATION"
    )
    return {
        "state": state,
        "bbox_oob_count": int(np.count_nonzero(~bbox_inside)),
        "hull_oob_count": int(np.count_nonzero(~hull_inside)),
        "test_count": int(len(test_q)),
        "train_q1_min": float(lower[0]),
        "train_q1_max": float(upper[0]),
        "train_q2_min": float(lower[1]),
        "train_q2_max": float(upper[1]),
    }


def existing_support_map(path: Path) -> dict[tuple[str, str, str], str]:
    result: dict[tuple[str, str, str], str] = {}
    for row in read_csv(path):
        if row.get("model") != "B2":
            continue
        key = (
            row["cv_scheme"],
            row["heldout_group"],
            condition_id(row["dataset"], int(row["position_rank"])),
        )
        result[key] = row["support_state"]
    return result


def fold_definitions(conditions: dict[str, Condition]) -> list[tuple[str, str, list[Condition], list[Condition]]]:
    values = list(conditions.values())
    folds: list[tuple[str, str, list[Condition], list[Condition]]] = []
    for height in HEIGHT_ORDER:
        train = [c for c in values if c.nominal_height_mm != height]
        test = [c for c in values if c.nominal_height_mm == height]
        folds.append(("LOHO_height", height_group(height), train, test))
    for rank in range(1, 6):
        train = [c for c in values if c.position_rank != rank]
        test = [c for c in values if c.position_rank == rank]
        if test:
            folds.append(("LOPO_position_rank", f"rank_{rank}", train, test))
    for band, heights in HEIGHT_BANDS.items():
        train = [c for c in values if c.nominal_height_mm not in heights]
        test = [c for c in values if c.nominal_height_mm in heights]
        folds.append(("LOBO_height_band", band, train, test))
    heldout = [c for c in values if c.nominal_height_mm == 50.0]
    dev = [c for c in values if c.nominal_height_mm != 50.0]
    folds.append(("strict_50mm_validation", "height_50mm_strict_heldout", dev, heldout))
    return folds


def layer_metric_row(
    scheme: str,
    heldout_group: str,
    aggregation: str,
    layer: str,
    condition_rows: list[dict[str, Any]],
    invalid_count: int = 0,
) -> dict[str, Any]:
    biases = [row["condition_bias_mm"] for row in condition_rows if finite(row.get("condition_bias_mm"))]
    summary = metric_values(biases)
    worst = max(
        (row for row in condition_rows if finite(row.get("condition_bias_mm"))),
        key=lambda row: abs(float(row["condition_bias_mm"])),
        default=None,
    )
    row = {
        "cv_scheme": scheme,
        "heldout_group": heldout_group,
        "aggregation": aggregation,
        "layer": layer,
        "condition_count": len(condition_rows),
        "formal_point_count": sum(int(row.get("point_count", 0)) for row in condition_rows),
        "invalid_point_count": int(invalid_count),
        "worst_condition_id": "" if worst is None else worst["condition_id"],
        "worst_condition_abs_mm": None if worst is None else abs(float(worst["condition_bias_mm"])),
        "support_state_counts": json.dumps(
            {
                state: sum(row.get("support_state") == state for row in condition_rows)
                for state in ("IN_DOMAIN", "HULL_EXTRAPOLATION", "BBOX_EXTRAPOLATION")
            },
            sort_keys=True,
        ),
    }
    row.update(summary)
    return row


def support_row(
    scheme: str,
    heldout_group: str,
    aggregation: str,
    layer: str,
    support_state: str,
    condition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [row for row in condition_rows if row["support_state"] == support_state]
    row = layer_metric_row(
        scheme,
        heldout_group,
        aggregation,
        layer,
        selected,
    )
    row["support_state"] = support_state
    return row


def append_magnitude_rows(
    output_rows: list[dict[str, Any]],
    scheme: str,
    heldout_group: str,
    layer: str,
    beta: np.ndarray,
    conditions: list[Condition],
    inputs: dict[str, Any],
) -> None:
    params = inputs["app"].reconstruction
    if layer == "H-B2":
        scopes = [
            (
                "formal",
                np.concatenate([
                    beta[0] + beta[1] * condition.formal.q2
                    for condition in conditions
                ]),
                np.asarray([], dtype=np.float64),
                np.asarray([], dtype=np.float64),
            )
        ]
    else:
        formal_delta: list[np.ndarray] = []
        formal_dz: list[np.ndarray] = []
        baseline_delta: list[np.ndarray] = []
        baseline_dz: list[np.ndarray] = []
        for condition in conditions:
            formal = transform_lambda(condition.formal, beta, params, inputs["transform"])
            baseline = transform_lambda(condition.baseline, beta, params, inputs["transform"])
            formal_delta.append(formal["delta_lambda"])
            formal_dz.append(formal["delta_Zg"])
            baseline_delta.append(baseline["delta_lambda"])
            baseline_dz.append(baseline["delta_Zg"])
        scopes = [
            ("formal", np.concatenate(formal_delta), np.concatenate(formal_dz), np.asarray([], dtype=np.float64)),
            ("baseline", np.concatenate(baseline_delta), np.concatenate(baseline_dz), np.asarray([], dtype=np.float64)),
        ]
    for scope, delta_lambda, delta_zg, _ in scopes:
        row = {
            "cv_scheme": scheme,
            "heldout_group": heldout_group,
            "layer": layer,
            "scope": scope,
            "beta_0": float(beta[0]),
            "beta_1": float(beta[1]),
        }
        if layer == "H-B2":
            row.update(distribution(delta_lambda, "delta_h_mm"))
            row.update(distribution(np.abs(delta_lambda), "abs_delta_h_mm"))
        else:
            row.update(distribution(delta_lambda, "delta_lambda_mm"))
            row.update(distribution(np.abs(delta_lambda), "abs_delta_lambda_mm"))
            row.update(distribution(delta_zg, "delta_Zg_mm"))
            row.update(distribution(np.abs(delta_zg), "abs_delta_Zg_mm"))
        output_rows.append(row)


def run_fold(
    scheme: str,
    heldout_group: str,
    train: list[Condition],
    test: list[Condition],
    inputs: dict[str, Any],
    support_map: dict[tuple[str, str, str], str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    hfit = fit_height_model(train)
    lfit = fit_lambda_model(train, inputs)
    hbeta = np.asarray(hfit["beta"], dtype=np.float64)
    lbeta = np.asarray(lfit["beta"], dtype=np.float64)
    condition_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    magnitude_rows: list[dict[str, Any]] = []
    invalid_total = 0
    for condition in test:
        support = support_map.get(
            (scheme, heldout_group, condition.condition_id),
            support_from_q(train, condition)["state"],
        )
        raw = np.asarray(condition.formal.raw_residual, dtype=np.float64)
        hresidual = apply_height_residual(condition, hbeta)
        leval = evaluate_lambda_condition(condition, lbeta, inputs)
        invalid_total += int(leval["invalid_count"])
        lresidual = np.full(condition.formal.source_count, np.nan, dtype=np.float64)
        if len(leval["residual"]) == condition.formal.source_count:
            lresidual[:] = leval["residual"]
        for layer, values in (("RAW", raw), ("H-B2", hresidual), ("L-B2", lresidual)):
            valid_values = values[np.isfinite(values)]
            stats = metric_values([float(np.mean(valid_values))] if len(valid_values) else [])
            condition_rows.append({
                "cv_scheme": scheme,
                "heldout_group": heldout_group,
                "aggregation": "fold",
                "layer": layer,
                "condition_id": condition.condition_id,
                "dataset": condition.dataset,
                "nominal_height_mm": condition.nominal_height_mm,
                "true_height_mm": condition.true_height_mm,
                "position_rank": condition.position_rank,
                "point_count": int(len(valid_values)),
                "invalid_point_count": int(len(values) - len(valid_values)),
                "support_state": support,
                "condition_bias_mm": float(np.mean(valid_values)) if len(valid_values) else float("nan"),
                "condition_mae_mm": float(np.mean(np.abs(valid_values))) if len(valid_values) else float("nan"),
                "condition_rmse_mm": float(np.sqrt(np.mean(valid_values * valid_values))) if len(valid_values) else float("nan"),
                "condition_p95_abs_mm": float(np.percentile(np.abs(valid_values), 95.0)) if len(valid_values) else float("nan"),
                "condition_max_abs_mm": float(np.max(np.abs(valid_values))) if len(valid_values) else float("nan"),
                "condition_raw_bias_mm": float(np.mean(raw)),
            })
        proxy_after = leval.get("proxy")
        baseline_transformed = leval.get("baseline", {})
        proxy_rows.append({
            "cv_scheme": scheme,
            "heldout_group": heldout_group,
            "condition_id": condition.condition_id,
            "dataset": condition.dataset,
            "nominal_height_mm": condition.nominal_height_mm,
            "position_rank": condition.position_rank,
            "layer": "L-B2",
            "raw_proxy_a_mm_per_mm": float(condition.raw_proxy.slope),
            "raw_proxy_b_mm": float(condition.raw_proxy.intercept),
            "corrected_proxy_a_mm_per_mm": None if proxy_after is None else float(proxy_after.slope),
            "corrected_proxy_b_mm": None if proxy_after is None else float(proxy_after.intercept),
            "delta_proxy_a_mm_per_mm": None if proxy_after is None else float(proxy_after.slope - condition.raw_proxy.slope),
            "delta_proxy_b_mm": None if proxy_after is None else float(proxy_after.intercept - condition.raw_proxy.intercept),
            "raw_proxy_rmse_mm": float(condition.raw_proxy.rmse),
            "corrected_proxy_rmse_mm": None if proxy_after is None else float(proxy_after.rmse),
            "baseline_point_count": int(condition.baseline.source_count),
            "baseline_invalid_point_count": int(baseline_transformed.get("invalid_count", 0)),
            "formal_point_count": int(condition.formal.source_count),
            "formal_invalid_point_count": int(leval["invalid_count"]),
            "raw_baseline_c1_clamp_count": int(np.count_nonzero(condition.baseline.c1_clamped)),
            "corrected_baseline_c1_clamp_count": int(np.count_nonzero(condition.baseline.c1_clamped)),
            "formal_c1_clamp_count": int(np.count_nonzero(condition.formal.c1_clamped)),
            "lambda_delta_baseline_p95_abs_mm": float(np.percentile(np.abs(baseline_transformed["delta_lambda"]), 95.0)),
            "lambda_delta_formal_p95_abs_mm": float(np.percentile(np.abs(transform_lambda(condition.formal, lbeta, inputs["app"].reconstruction, inputs["transform"])["delta_lambda"]), 95.0)),
            "delta_Zg_baseline_p95_abs_mm": float(np.percentile(np.abs(baseline_transformed["delta_Zg"]), 95.0)),
            "fit_status": lfit["fit_status"],
        })
    metric_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        selected = [row for row in condition_rows if row["layer"] == layer]
        metric_rows.append(layer_metric_row(scheme, heldout_group, "fold", layer, selected, invalid_total if layer == "L-B2" else 0))
    support_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        selected = [row for row in condition_rows if row["layer"] == layer]
        for state in ("IN_DOMAIN", "HULL_EXTRAPOLATION", "BBOX_EXTRAPOLATION"):
            if any(row["support_state"] == state for row in selected):
                support_rows.append(support_row(scheme, heldout_group, "fold", layer, state, selected))
    coefficient_rows = [
        {
            "cv_scheme": scheme,
            "heldout_group": heldout_group,
            "layer": "H-B2",
            "parameter": "a0",
            "value": float(hbeta[0]),
            "fit_status": hfit["fit_status"],
            "design_rank": hfit["design_rank"],
            "design_condition_number": hfit["design_condition_number"],
            "iterations": hfit["iterations"],
            "objective": hfit["objective"],
        },
        {
            "cv_scheme": scheme,
            "heldout_group": heldout_group,
            "layer": "H-B2",
            "parameter": "a2",
            "value": float(hbeta[1]),
            "fit_status": hfit["fit_status"],
            "design_rank": hfit["design_rank"],
            "design_condition_number": hfit["design_condition_number"],
            "iterations": hfit["iterations"],
            "objective": hfit["objective"],
        },
        {
            "cv_scheme": scheme,
            "heldout_group": heldout_group,
            "layer": "L-B2",
            "parameter": "b0",
            "value": float(lbeta[0]),
            "fit_status": lfit["fit_status"],
            "design_rank": lfit["design_rank"],
            "design_condition_number": lfit["design_condition_number"],
            "iterations": lfit["iterations"],
            "objective": lfit["objective"],
        },
        {
            "cv_scheme": scheme,
            "heldout_group": heldout_group,
            "layer": "L-B2",
            "parameter": "b2",
            "value": float(lbeta[1]),
            "fit_status": lfit["fit_status"],
            "design_rank": lfit["design_rank"],
            "design_condition_number": lfit["design_condition_number"],
            "iterations": lfit["iterations"],
            "objective": lfit["objective"],
        },
    ]
    append_magnitude_rows(magnitude_rows, scheme, heldout_group, "H-B2", hbeta, test, inputs)
    append_magnitude_rows(magnitude_rows, scheme, heldout_group, "L-B2", lbeta, test, inputs)
    return metric_rows, condition_rows, support_rows, coefficient_rows, magnitude_rows, proxy_rows, {
        "hfit": hfit,
        "lfit": lfit,
        "invalid_total": invalid_total,
    }


def pooled_rows(
    metric_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pooled_metrics: list[dict[str, Any]] = []
    pooled_support: list[dict[str, Any]] = []
    schemes = sorted({row["cv_scheme"] for row in metric_rows})
    for scheme in schemes:
        if scheme == "strict_50mm_validation":
            groups = ["height_50mm_strict_heldout"]
        else:
            groups = sorted({row["heldout_group"] for row in metric_rows if row["cv_scheme"] == scheme})
        selected_conditions = [
            row for row in condition_rows
            if row["cv_scheme"] == scheme and row["heldout_group"] in groups
        ]
        for layer in LAYERS:
            layer_conditions = [row for row in selected_conditions if row["layer"] == layer]
            invalid = sum(int(row.get("invalid_point_count", 0)) for row in layer_conditions)
            pooled_metrics.append(layer_metric_row(scheme, "ALL_FOLDS", "pooled_condition_means", layer, layer_conditions, invalid))
            for state in ("IN_DOMAIN", "HULL_EXTRAPOLATION", "BBOX_EXTRAPOLATION"):
                if any(row["support_state"] == state for row in layer_conditions):
                    pooled_support.append(support_row(scheme, "ALL_FOLDS", "pooled_condition_means", layer, state, layer_conditions))
    return pooled_metrics, pooled_support


def incremental_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = sorted({(row["cv_scheme"], row["heldout_group"], row["aggregation"]) for row in metric_rows})
    for scheme, group, aggregation in keys:
        selected = {
            row["layer"]: row
            for row in metric_rows
            if row["cv_scheme"] == scheme
            and row["heldout_group"] == group
            and row["aggregation"] == aggregation
        }
        if "H-B2" not in selected or "L-B2" not in selected:
            continue
        h = selected["H-B2"]
        l = selected["L-B2"]
        h_worst = h["worst_condition_abs_mm"]
        l_worst = l["worst_condition_abs_mm"]
        h_worst_value = float(h_worst) if finite(h_worst) else float("nan")
        l_worst_value = float(l_worst) if finite(l_worst) else float("nan")
        result.append({
            "cv_scheme": scheme,
            "heldout_group": group,
            "aggregation": aggregation,
            "h_rmse_mm": h["rmse_mm"],
            "l_rmse_mm": l["rmse_mm"],
            "delta_rmse_L_minus_H_mm": l["rmse_mm"] - h["rmse_mm"],
            "h_p95_abs_mm": h["p95_abs_mm"],
            "l_p95_abs_mm": l["p95_abs_mm"],
            "delta_p95_L_minus_H_mm": l["p95_abs_mm"] - h["p95_abs_mm"],
            "h_worst_condition_abs_mm": h["worst_condition_abs_mm"],
            "l_worst_condition_abs_mm": l["worst_condition_abs_mm"],
            "delta_worst_L_minus_H_mm": l_worst_value - h_worst_value,
            "worst_condition_improvement_ratio_L_vs_H": (
                (h_worst_value - l_worst_value) / h_worst_value
                if finite(h_worst_value) and finite(l_worst_value) and abs(h_worst_value) > EPS
                else None
            ),
            "l_not_worse_rmse_p95_worst": bool(
                finite(l["rmse_mm"])
                and finite(l["p95_abs_mm"])
                and finite(l_worst_value)
                and l["rmse_mm"] <= h["rmse_mm"] + 1.0e-12
                and l["p95_abs_mm"] <= h["p95_abs_mm"] + 1.0e-12
                and l_worst_value <= h_worst_value + 1.0e-12
            ),
        })
    return result


def coefficient_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for layer in ("H-B2", "L-B2"):
        for parameter in (("a0", "a2") if layer == "H-B2" else ("b0", "b2")):
            selected = [
                float(row["value"]) for row in rows
                if row["layer"] == layer
                and row["parameter"] == parameter
                and row["cv_scheme"] in DEV_SCHEMES
            ]
            if not selected:
                continue
            values = np.asarray(selected, dtype=np.float64)
            mean = float(np.mean(values))
            value_range = float(np.max(values) - np.min(values))
            result.append({
                "layer": layer,
                "parameter": parameter,
                "fold_count": int(len(values)),
                "mean": mean,
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": value_range,
                "relative_range_to_abs_mean": value_range / max(abs(mean), EPS),
                "sign_consistency": bool(np.all(values >= 0.0) or np.all(values <= 0.0)),
            })
    return result


def physical_audit(
    metric_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    magnitude_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_fold_metrics = [
        row for row in metric_rows
        if row["aggregation"] == "fold"
        and row["layer"] == "L-B2"
    ]
    dev_fold_metrics = [row for row in all_fold_metrics if row["cv_scheme"] in DEV_SCHEMES]
    invalid_total = sum(int(row.get("invalid_point_count", 0)) for row in all_fold_metrics)
    invalid_dev_total = sum(int(row.get("invalid_point_count", 0)) for row in dev_fold_metrics)
    lambda_magnitudes = [
        row for row in magnitude_rows
        if row["layer"] == "L-B2" and row["scope"] in {"formal", "baseline"}
    ]
    dev_lambda_magnitudes = [
        row for row in lambda_magnitudes if row["cv_scheme"] in DEV_SCHEMES
    ]
    def max_field(rows: list[dict[str, Any]], field: str) -> float:
        return max(
            (float(row[field]) for row in rows if finite(row.get(field))),
            default=float("nan"),
        )

    max_abs_lambda = max(
        (float(row["abs_delta_lambda_mm_max"]) for row in lambda_magnitudes if finite(row.get("abs_delta_lambda_mm_max"))),
        default=float("nan"),
    )
    p95_abs_lambda = max(
        (float(row["abs_delta_lambda_mm_p95"]) for row in lambda_magnitudes if finite(row.get("abs_delta_lambda_mm_p95"))),
        default=float("nan"),
    )
    max_abs_zg = max(
        (float(row["abs_delta_Zg_mm_max"]) for row in lambda_magnitudes if finite(row.get("abs_delta_Zg_mm_max"))),
        default=float("nan"),
    )
    clamp_changes = sum(
        abs(int(row["raw_baseline_c1_clamp_count"]) - int(row["corrected_baseline_c1_clamp_count"]))
        + abs(int(row["formal_c1_clamp_count"]) - int(row["formal_c1_clamp_count"]))
        for row in proxy_rows
    )
    dev_clamp_changes = sum(
        abs(int(row["raw_baseline_c1_clamp_count"]) - int(row["corrected_baseline_c1_clamp_count"]))
        for row in proxy_rows
        if row["cv_scheme"] in DEV_SCHEMES
    )
    l_coefficients = [
        row for row in coefficient_rows
        if row["layer"] == "L-B2" and row["cv_scheme"] in DEV_SCHEMES
    ]
    failed_fits = [
        row for row in l_coefficients
        if row.get("fit_status") not in {"success", "rank_deficient"}
    ]
    stability_rel = max(
        (float(row["relative_range_to_abs_mean"]) for row in stability_rows if row["layer"] == "L-B2"),
        default=float("inf"),
    )
    conditions = {
        "fit_failure": bool(failed_fits),
        "new_invalid_point_count": int(invalid_total),
        "development_new_invalid_point_count": int(invalid_dev_total),
        "c1_clamp_state_change_count": int(clamp_changes),
        "max_abs_delta_lambda_mm": max_abs_lambda,
        "max_p95_abs_delta_lambda_mm": p95_abs_lambda,
        "max_abs_delta_Zg_mm": max_abs_zg,
        "development_max_abs_delta_lambda_mm": max_field(dev_lambda_magnitudes, "abs_delta_lambda_mm_max"),
        "development_max_p95_abs_delta_lambda_mm": max_field(dev_lambda_magnitudes, "abs_delta_lambda_mm_p95"),
        "development_max_abs_delta_Zg_mm": max_field(dev_lambda_magnitudes, "abs_delta_Zg_mm_max"),
        "development_c1_clamp_state_change_count": int(dev_clamp_changes),
        "lambda_coefficient_max_relative_range": stability_rel,
        "thresholds": {
            "max_abs_delta_lambda_mm": 2.0,
            "max_p95_abs_delta_lambda_mm": 1.0,
            "max_abs_delta_Zg_mm": 2.0,
            "max_coefficient_relative_range": 1.0,
        },
    }
    if (
        conditions["fit_failure"]
        or conditions["new_invalid_point_count"] > 0
        or conditions["c1_clamp_state_change_count"] > 0
    ):
        status = "NOT_SUPPORTED"
    elif (
        finite(max_abs_lambda)
        and finite(p95_abs_lambda)
        and finite(max_abs_zg)
        and max_abs_lambda <= 2.0
        and p95_abs_lambda <= 1.0
        and max_abs_zg <= 2.0
        and stability_rel <= 1.0
    ):
        status = "SUPPORTED"
    else:
        status = "PARTIAL"
    if (
        conditions["fit_failure"]
        or conditions["development_new_invalid_point_count"] > 0
        or conditions["development_c1_clamp_state_change_count"] > 0
    ):
        development_status = "NOT_SUPPORTED"
    elif (
        finite(conditions["development_max_abs_delta_lambda_mm"])
        and finite(conditions["development_max_p95_abs_delta_lambda_mm"])
        and finite(conditions["development_max_abs_delta_Zg_mm"])
        and conditions["development_max_abs_delta_lambda_mm"] <= 2.0
        and conditions["development_max_p95_abs_delta_lambda_mm"] <= 1.0
        and conditions["development_max_abs_delta_Zg_mm"] <= 2.0
        and stability_rel <= 1.0
    ):
        development_status = "SUPPORTED"
    else:
        development_status = "PARTIAL"
    conditions["LAMBDA_LAYER_PHYSICAL_VALIDITY"] = status
    conditions["development_LAMBDA_LAYER_PHYSICAL_VALIDITY"] = development_status
    return conditions


def decision_from_results(
    metric_rows: list[dict[str, Any]],
    incremental: list[dict[str, Any]],
    physical: dict[str, Any],
    historical: Mapping[str, Any],
) -> dict[str, Any]:
    pooled_dev = [
        row for row in metric_rows
        if row["aggregation"] == "pooled_condition_means"
        and row["cv_scheme"] in DEV_SCHEMES
    ]
    raw_by_scheme = {
        row["cv_scheme"]: row
        for row in pooled_dev
        if row["layer"] == "RAW"
    }
    h_by_scheme = {
        row["cv_scheme"]: row
        for row in pooled_dev
        if row["layer"] == "H-B2"
    }
    l_by_scheme = {
        row["cv_scheme"]: row
        for row in pooled_dev
        if row["layer"] == "L-B2"
    }
    h_improves_raw = all(
        h_by_scheme[scheme]["rmse_mm"] < raw_by_scheme[scheme]["rmse_mm"] - 1.0e-12
        and h_by_scheme[scheme]["p95_abs_mm"] < raw_by_scheme[scheme]["p95_abs_mm"] - 1.0e-12
        for scheme in DEV_SCHEMES
        if scheme in h_by_scheme and scheme in raw_by_scheme
    )
    l_improves_raw = all(
        l_by_scheme[scheme]["rmse_mm"] < raw_by_scheme[scheme]["rmse_mm"] - 1.0e-12
        and l_by_scheme[scheme]["p95_abs_mm"] < raw_by_scheme[scheme]["p95_abs_mm"] - 1.0e-12
        for scheme in DEV_SCHEMES
        if scheme in l_by_scheme and scheme in raw_by_scheme
    )
    l_noninferior = all(
        row["l_not_worse_rmse_p95_worst"]
        for row in incremental
        if row["aggregation"] == "pooled_condition_means"
        and row["cv_scheme"] in DEV_SCHEMES
    )
    if l_noninferior and physical["development_LAMBDA_LAYER_PHYSICAL_VALIDITY"] == "SUPPORTED":
        selected_layer = "LAMBDA"
    elif h_improves_raw:
        selected_layer = "HEIGHT"
    else:
        selected_layer = "UNDECIDED"
    q2_candidate = "YES" if h_improves_raw or l_improves_raw else "NO"
    more_data = "YES" if (
        historical.get("MORE_HEIGHT_ACQUISITION_REQUIRED") == "YES"
        or historical.get("HEIGHT_GAP_ACQUISITION_STILL_JUSTIFIED") == "YES"
    ) else "NO"
    return {
        "SELECTED_CORRECTION_LAYER": selected_layer,
        "LAMBDA_LAYER_PHYSICAL_VALIDITY": physical["LAMBDA_LAYER_PHYSICAL_VALIDITY"],
        "DEVELOPMENT_LAMBDA_LAYER_PHYSICAL_VALIDITY": physical["development_LAMBDA_LAYER_PHYSICAL_VALIDITY"],
        "Q2_ONLY_CORRECTION_CANDIDATE": q2_candidate,
        "MORE_DATA_REQUIRED_BEFORE_NEXT_ANALYSIS": more_data,
        "H_B2_IMPROVES_RAW_ALL_DEVELOPMENT_SCHEMES": h_improves_raw,
        "L_B2_IMPROVES_RAW_ALL_DEVELOPMENT_SCHEMES": l_improves_raw,
        "L_B2_NONINFERIOR_TO_H_B2_ALL_DEVELOPMENT_SCHEMES": l_noninferior,
        "historical_surface_model": "B2",
        "historical_q1_retained": "NO",
        "historical_Q2_GAP_FILLED": historical.get("Q2_GAP_FILLED"),
        "historical_SURFACE2C_ALLOWED": historical.get("SURFACE2C_ALLOWED"),
    }


def plot_comparison(output: Path, metric_rows: list[dict[str, Any]]) -> None:
    selected = [
        row for row in metric_rows
        if row["aggregation"] == "pooled_condition_means"
        and row["cv_scheme"] in DEV_SCHEMES + ("strict_50mm_validation",)
    ]
    labels = []
    values = {name: [] for name in ("RAW", "H-B2", "L-B2")}
    for scheme in DEV_SCHEMES + ("strict_50mm_validation",):
        labels.append({
            "LOHO_height": "LOHO",
            "LOPO_position_rank": "LOPO",
            "LOBO_height_band": "LOBO",
            "strict_50mm_validation": "50 strict",
        }[scheme])
        for layer in values:
            match = next(
                row for row in selected
                if row["cv_scheme"] == scheme and row["layer"] == layer
            )
            values[layer].append(match)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.24
    for index, layer in enumerate(("RAW", "H-B2", "L-B2")):
        axes[0].bar(x + (index - 1) * width, [row["rmse_mm"] for row in values[layer]], width, label=layer)
        axes[1].bar(x + (index - 1) * width, [row["p95_abs_mm"] for row in values[layer]], width, label=layer)
        axes[2].bar(
            x + (index - 1) * width,
            [float(row["worst_condition_abs_mm"]) if finite(row["worst_condition_abs_mm"]) else np.nan for row in values[layer]],
            width,
            label=layer,
        )
    for axis, title in zip(axes, ("Condition-mean RMSE (mm)", "Condition-mean P95 (mm)", "Worst condition abs bias (mm)")):
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.savefig(output / "surface2_correction_layer_raw_h_l.png", dpi=180)
    plt.close(figure)


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def report_text(
    decision: dict[str, Any],
    physical: dict[str, Any],
    consistency: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    incremental: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> str:
    pooled = [
        row for row in metric_rows
        if row["aggregation"] == "pooled_condition_means"
        and row["cv_scheme"] in DEV_SCHEMES + ("strict_50mm_validation",)
    ]
    pooled_lines = []
    for scheme in DEV_SCHEMES + ("strict_50mm_validation",):
        for layer in LAYERS:
            row = next(row for row in pooled if row["cv_scheme"] == scheme and row["layer"] == layer)
            pooled_lines.append(
                f"| {scheme} | {layer} | {row['condition_count']} | {fmt(row['bias_mm'])} | {fmt(row['mae_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['max_abs_mm'])} | {fmt(row['worst_condition_abs_mm'])} |"
            )
    inc_lines = []
    for row in incremental:
        if row["aggregation"] != "pooled_condition_means":
            continue
        inc_lines.append(
            f"| {row['cv_scheme']} | {fmt(row['delta_rmse_L_minus_H_mm'])} | {fmt(row['delta_p95_L_minus_H_mm'])} | {fmt(row['delta_worst_L_minus_H_mm'])} | {row['l_not_worse_rmse_p95_worst']} |"
        )
    support_lines = []
    for row in support_rows:
        if row["aggregation"] != "pooled_condition_means" or row["cv_scheme"] not in DEV_SCHEMES:
            continue
        if row["layer"] not in {"H-B2", "L-B2"}:
            continue
        support_lines.append(
            f"| {row['cv_scheme']} | {row['support_state']} | {row['layer']} | {row['condition_count']} | {fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['worst_condition_abs_mm'])} |"
        )
    stability_lines = [
        f"| {row['layer']} | {row['parameter']} | {fmt(row['mean'], 6)} | {fmt(row['std'], 6)} | {fmt(row['range'], 6)} | {fmt(row['relative_range_to_abs_mean'], 3)} | {row['sign_consistency']} |"
        for row in stability
    ]
    consistency_max = max(row["max_abs_raw_residual_replay_difference_mm"] for row in consistency)
    return f"""# Surface correction layer replay

## Decision

SELECTED_CORRECTION_LAYER={decision['SELECTED_CORRECTION_LAYER']}

LAMBDA_LAYER_PHYSICAL_VALIDITY={decision['LAMBDA_LAYER_PHYSICAL_VALIDITY']}

DEVELOPMENT_LAMBDA_LAYER_PHYSICAL_VALIDITY={decision['DEVELOPMENT_LAMBDA_LAYER_PHYSICAL_VALIDITY']}

Q2_ONLY_CORRECTION_CANDIDATE={decision['Q2_ONLY_CORRECTION_CANDIDATE']}

MORE_DATA_REQUIRED_BEFORE_NEXT_ANALYSIS={decision['MORE_DATA_REQUIRED_BEFORE_NEXT_ANALYSIS']}

The historical conclusion is preserved: SELECTED_SURFACE_MODEL=B2,
Q1_RETAINED=NO, Q2_GAP_FILLED={decision['historical_Q2_GAP_FILLED']},
SURFACE2C_ALLOWED={decision['historical_SURFACE2C_ALLOWED']}.  This report is
diagnostic and is not production validation.

## Protocol and provenance

- H-B2 is a0+a2*q2 applied to the final height residual.
- L-B2 is b0+b2*q2 added to lambda_C1 before camera-point construction.
- L-B2 applies the same lambda change to repeat1 ground points and formal points.
  Each fold refits the session-linear ground proxy from the corrected repeat1
  baseline before evaluating formal residuals.
- Development conditions are condition-balanced; there is no random point split.
- LOHO, LOPO and LOBO are selection folds.  50 mm is a strict diagnostic only
  and is not used for fitting, selection, or threshold adjustment.
- Frozen C0 SHA256: {inputs['hashes']['quadratic_c0_sha256']}
- Frozen C1 SHA256: {inputs['hashes']['frozen_c1_sha256']}
- Config SHA256: {inputs['hashes']['config_sha256']}
- q2 is the existing Frozen-C0 intrinsic coordinate; no q redefinition.
- Maximum raw replay difference against the existing formal residual column:
  {fmt(consistency_max, 8)} mm.

## Grouped metrics

These are metrics over condition means, so each height x position condition has
equal weight.

| scheme | layer | conditions | Bias | MAE | RMSE | P95 | Max | worst condition |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pooled_lines)}

## Support stratification

Support is inherited from the frozen Surface-2BR2 B2 q-space protocol.  It is
reported separately so an extrapolation improvement is not treated as
in-domain generalization.

| scheme | support state | layer | conditions | RMSE | P95 | worst condition |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(support_lines)}

## L-B2 relative to H-B2

Negative values mean L-B2 is smaller.

| scheme | delta RMSE L-H | delta P95 L-H | delta worst L-H | L not worse |
|---|---:|---:|---:|---|
{chr(10).join(inc_lines)}

## Lambda physical and numerical audit

- New invalid final points across development plus strict-50 diagnostic folds: {physical['new_invalid_point_count']}
- Development-only new invalid final points: {physical['development_new_invalid_point_count']}
- Development-only physical status: {physical['development_LAMBDA_LAYER_PHYSICAL_VALIDITY']}
- C1 clamp-state changes: {physical['c1_clamp_state_change_count']}
- Maximum absolute delta lambda: {fmt(physical['max_abs_delta_lambda_mm'])} mm
- Maximum P95 absolute delta lambda: {fmt(physical['max_p95_abs_delta_lambda_mm'])} mm
- Maximum absolute induced delta Zg: {fmt(physical['max_abs_delta_Zg_mm'])} mm
- Maximum lambda coefficient relative fold range: {fmt(physical['lambda_coefficient_max_relative_range'], 3)}
- Physical thresholds used: abs delta lambda <= 2 mm, P95 abs delta lambda <= 1 mm,
  abs induced delta Zg <= 2 mm, coefficient relative range <= 1.
- C1 clamp is evaluated before the layer correction and is therefore expected
  to remain unchanged.  Any new final-depth invalid point would fail the
  physical validity check.
- The strict-50 invalid count contributes only to the diagnostic
  LAMBDA_LAYER_PHYSICAL_VALIDITY flag; it is not used to fit either layer,
  choose a fold, or adjust a threshold.  Development already fails the
  lambda physical-magnitude limits independently.

## Coefficient stability

| layer | parameter | mean | std | range | relative range | sign consistent |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(stability_lines)}

## Interpretation

The layer decision is based on development grouped CV only.  L-B2 must be no
worse than H-B2 simultaneously in RMSE, P95 and worst-condition error for
LOHO, LOPO and LOBO, and must pass the independent physical audit.  Lambda is
not preferred merely because it is more physical.  The strict 50 mm rows are
shown in the table only as an unselected diagnostic.

No C0/C1, ROI, q2, Steger, GUI, Ground G(S), H1, or online production
configuration was modified.  Existing Q2_GAP_FILLED=NO and
SURFACE2C_ALLOWED=NO remain historical conclusions.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(args.config.resolve())
    measurement_params = inputs["app"].measurement
    if not isinstance(measurement_params, MeasurementParams):
        measurement_params = MeasurementParams()
    inputs["measurement_params"] = measurement_params
    conditions, consistency = build_conditions(inputs)
    development = {
        key: value for key, value in conditions.items()
        if value.nominal_height_mm != 50.0
    }
    if len(development) != 44:
        raise RuntimeError(f"expected 44 development conditions, got {len(development)}")
    support_map = existing_support_map(inputs["paths"]["surface2br2_predictions"])
    metric_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    magnitude_rows: list[dict[str, Any]] = []
    proxy_audit_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    all_for_folds = conditions
    for scheme, group, train, test in fold_definitions(all_for_folds):
        if not train or not test:
            continue
        result = run_fold(scheme, group, train, test, inputs, support_map)
        fold_metrics, fold_condition, fold_support, fold_coefficients, fold_magnitudes, fold_proxies, fits = result
        metric_rows.extend(fold_metrics)
        condition_rows.extend(fold_condition)
        support_rows.extend(fold_support)
        coefficient_rows.extend(fold_coefficients)
        magnitude_rows.extend(fold_magnitudes)
        proxy_audit_rows.extend(fold_proxies)
        fold_records.append({
            "cv_scheme": scheme,
            "heldout_group": group,
            "train_condition_count": len(train),
            "test_condition_count": len(test),
            "H_fit_status": fits["hfit"]["fit_status"],
            "L_fit_status": fits["lfit"]["fit_status"],
            "L_invalid_train_points": int(fits["lfit"].get("invalid_count", 0)),
        })
    pooled_metric_rows, pooled_support_rows = pooled_rows(metric_rows, condition_rows, support_rows)
    metric_rows.extend(pooled_metric_rows)
    support_rows.extend(pooled_support_rows)
    incremental = incremental_rows(metric_rows)
    stability = coefficient_stability(coefficient_rows)
    physical = physical_audit(metric_rows, coefficient_rows, stability, magnitude_rows, proxy_audit_rows)
    historical = {
        "Q2_GAP_FILLED": "NO",
        "SURFACE2C_ALLOWED": "NO",
        "MORE_HEIGHT_ACQUISITION_REQUIRED": inputs["model_selection_summary"].get("MORE_HEIGHT_ACQUISITION_REQUIRED"),
    }
    decision = decision_from_results(metric_rows, incremental, physical, historical)
    write_csv(output / "surface2_correction_layer_cv_metrics.csv", metric_rows)
    write_csv(output / "surface2_correction_layer_condition_metrics.csv", condition_rows)
    write_csv(output / "surface2_correction_layer_support_metrics.csv", support_rows)
    write_csv(output / "surface2_correction_layer_incremental.csv", incremental)
    write_csv(output / "surface2_correction_layer_coefficients.csv", coefficient_rows)
    write_csv(output / "surface2_correction_layer_coefficient_stability.csv", stability)
    write_csv(output / "surface2_correction_layer_magnitude.csv", magnitude_rows)
    write_csv(output / "surface2_correction_layer_ground_proxy_audit.csv", proxy_audit_rows)
    write_csv(output / "surface2_correction_layer_raw_replay_audit.csv", consistency)
    write_csv(output / "surface2_correction_layer_fold_audit.csv", fold_records)
    plot_comparison(output, metric_rows)
    summary = {
        "decision": decision,
        "physical_audit": physical,
        "protocol": {
            "condition_equal_weight": True,
            "random_point_split": False,
            "heldout_50_excluded_from_fit_and_selection": True,
            "C0_refit": False,
            "C1_refit": False,
            "q2_redefined": False,
            "q1_used": False,
            "quadratic_terms": False,
            "spline_or_ml": False,
            "production_validation": False,
            "lambda_layer_recomputed_repeat1_proxy": True,
            "ground_u_compensation": None,
        },
        "provenance": {
            "input_paths": {name: str(path) for name, path in inputs["paths"].items()},
            "input_sha256": {
                name: sha256(path) for name, path in inputs["paths"].items()
            },
            "frozen_hashes": inputs["hashes"],
            "reused_surface2br2_hashes": inputs["surface2br2_input_hashes"],
            "development_condition_count": len(development),
            "strict_50_condition_count": len(conditions) - len(development),
            "development_formal_point_count": sum(c.formal.source_count for c in development.values()),
            "strict_50_formal_point_count": sum(c.formal.source_count for c in conditions.values() if c.nominal_height_mm == 50.0),
            "raw_replay_max_difference_mm": max(row["max_abs_raw_residual_replay_difference_mm"] for row in consistency),
        },
        "historical_conclusion_preserved": {
            "SELECTED_SURFACE_MODEL": "B2",
            "Q1_RETAINED": "NO",
            "Q2_GAP_FILLED": "NO",
            "SURFACE2C_ALLOWED": "NO",
        },
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface2_correction_layer_summary.json", summary)
    (output / "surface2_correction_layer_report.md").write_text(
        report_text(decision, physical, consistency, metric_rows, support_rows, incremental, stability, inputs),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        **decision,
        "raw_replay_max_difference_mm": summary["provenance"]["raw_replay_max_difference_mm"],
        "development_condition_count": len(development),
        "strict_50_condition_count": len(conditions) - len(development),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
