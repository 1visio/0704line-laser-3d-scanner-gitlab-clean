"""Surface-2B: frozen-q domain continuity and residual consistency audit.

The script reuses the existing Surface-1A point table for 30/50 mm and the
one-pass Steger cache for the manually frozen 36/40/46 mm acquisitions.  New
points are evaluated with the same Frozen Quadratic C0 and Frozen C1.  It does
not fit C0/C1, redefine q1/q2, adjust ROIs from residuals, or fit a correction.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
TOOLS_ROOT = REPO_ROOT / "tools"
for item in (MEASUREMENT_ROOT, TOOLS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from replay_daheng_ground4a import _fit_fixed_s_profile
import analyze_surface1a as surface1a


BASE_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "daheng_c1_gauge_blocks_20260819_ground4a"
)
DEFAULT_SURFACE1A = BASE_OUTPUT / "surface1a"
DEFAULT_SURFACE2 = BASE_OUTPUT / "surface2"
DEFAULT_OUTPUT = DEFAULT_SURFACE2 / "surface2b"
DEFAULT_CONFIG = (
    MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
)
DEFAULT_REGISTRY = DEFAULT_SURFACE2 / "manual_roi" / "roi_registry_manual.json"
DEFAULT_DRAFT = DEFAULT_SURFACE2 / "manual_roi" / "roi_registry_manual_draft.json"
DEFAULT_CENTER_CACHE = DEFAULT_SURFACE2 / "surface2_center_cache.csv"

NEW_DATASETS = ("obs_36mm", "obs_40mm", "obs_46mm")
TRUTH_MM = {
    "obs_30mm": 30.0,
    "obs_36mm": 36.0,
    "obs_40mm": 40.0,
    "obs_46mm": 46.0,
    "obs_50mm": 50.0,
}
HEIGHTS = (30.0, 36.0, 40.0, 46.0, 50.0)
POSE_IDS = tuple(f"{index:03d}" for index in range(1, 6))
Q_TOLERANCE = 0.05
RESIDUAL_MEDIAN_SUPPORTED_MM = 0.05
RESIDUAL_P95_SUPPORTED_MM = 0.10
RESIDUAL_MEDIAN_PARTIAL_MM = 0.10
RESIDUAL_P95_PARTIAL_MM = 0.20


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
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.15g}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", restval=""
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def distribution(values: Iterable[float], prefix: str = "") -> dict[str, Any]:
    data = np.asarray([float(value) for value in values if finite(value)], dtype=np.float64)
    names = ("min", "p05", "median", "p95", "max")
    if not len(data):
        result = {name: None for name in names}
        result["count"] = 0
    else:
        result = {
            "count": int(len(data)),
            "min": float(np.min(data)),
            "p05": float(np.percentile(data, 5.0)),
            "median": float(np.median(data)),
            "p95": float(np.percentile(data, 95.0)),
            "max": float(np.max(data)),
        }
    if not prefix:
        return result
    return {f"{prefix}_{key}": value for key, value in result.items()}


def residual_metrics(values: Iterable[float]) -> dict[str, Any]:
    data = np.asarray([float(value) for value in values if finite(value)], dtype=np.float64)
    if not len(data):
        return {
            "residual_count": 0,
            "residual_bias_mm": None,
            "residual_mae_mm": None,
            "residual_rmse_mm": None,
            "residual_p95_abs_mm": None,
            "residual_max_abs_mm": None,
        }
    absolute = np.abs(data)
    return {
        "residual_count": int(len(data)),
        "residual_bias_mm": float(np.mean(data)),
        "residual_mae_mm": float(np.mean(absolute)),
        "residual_rmse_mm": float(np.sqrt(np.mean(data**2))),
        "residual_p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "residual_max_abs_mm": float(np.max(absolute)),
    }


def validate_registry(
    final_path: Path, draft_path: Path
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    final = read_json(final_path)
    draft = read_json(draft_path)
    entries = final.get("entries")
    draft_entries = draft.get("entries")
    if not isinstance(entries, list) or len(entries) != 15:
        raise RuntimeError("Surface-2B requires exactly 15 frozen ROI entries")
    if not isinstance(draft_entries, list) or len(draft_entries) != 15:
        raise RuntimeError("manual draft must contain exactly 15 entries")
    if entries != draft_entries:
        raise RuntimeError("manual final and draft ROI entries differ")
    if final.get("manual_confirmed") is not True:
        raise RuntimeError("final ROI registry top-level manual_confirmed is not true")
    if int(final.get("manual_confirmed_count", 0)) != 15:
        raise RuntimeError("final ROI registry is not 15/15 confirmed")
    if not all(entry.get("manual_confirmed") is True for entry in entries):
        raise RuntimeError("one or more ROI entries are not manually confirmed")
    keys = {(str(entry["dataset"]), str(entry["pose_id"])) for entry in entries}
    expected = {(dataset, pose) for dataset in NEW_DATASETS for pose in POSE_IDS}
    if keys != expected:
        raise RuntimeError(f"ROI registry keys mismatch: {sorted(expected - keys)}")
    for dataset in NEW_DATASETS:
        ranks = sorted(
            int(entry["position_rank"])
            for entry in entries
            if entry["dataset"] == dataset
        )
        if ranks != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"{dataset} position ranks are not 1..5: {ranks}")
    return (
        {(str(entry["dataset"]), str(entry["pose_id"])): entry for entry in entries},
        {
            "final_path": final_path,
            "final_sha256": sha256(final_path),
            "draft_path": draft_path,
            "draft_sha256": sha256(draft_path),
            "entry_count": 15,
            "entry_manual_confirmed_count": 15,
            "entries_identical": True,
            "frozen_at": final.get("frozen_at"),
        },
    )


def load_center_cache(path: Path) -> dict[tuple[str, str, int], np.ndarray]:
    groups: dict[tuple[str, str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for row in read_csv(path):
        key = (row["dataset"], row["pose_id"], int(row["repeat_index"]))
        groups[key].append(
            (int(row["point_index"]), float(row["u_px"]), float(row["v_px"]))
        )
    expected = {
        (dataset, pose, repeat)
        for dataset in NEW_DATASETS
        for pose in POSE_IDS
        for repeat in range(1, 6)
    }
    if set(groups) != expected:
        raise RuntimeError(
            f"Surface-2 center cache key mismatch; missing={sorted(expected-set(groups))[:5]}"
        )
    output: dict[tuple[str, str, int], np.ndarray] = {}
    for key, values in groups.items():
        values.sort(key=lambda item: item[0])
        output[key] = np.asarray(
            [(item[1], item[2]) for item in values], dtype=np.float64
        )
    return output


def validate_frozen_provenance(
    config_path: Path,
    surface1a_summary_path: Path,
    coordinate_path: Path,
) -> tuple[Any, dict[str, Any], Any, Mapping[str, Any], dict[str, Any]]:
    app = load_app_config(config_path)
    if app.reconstruction.enable_laser_ray_correction is not True:
        raise RuntimeError("enable_laser_ray_correction must remain true")
    if app.calibration.laser_ray_correction is None:
        raise RuntimeError("Frozen C1 path is missing")
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_plane,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        laser_ray_correction=app.calibration.laser_ray_correction,
    )
    correction = calibration.get("laser_ray_correction")
    laser_model = calibration.get("laser_model")
    if correction is None or not isinstance(laser_model, Mapping):
        raise RuntimeError("Frozen C0/C1 calibration did not load")
    if laser_model.get("model_type") != "quadratic_graph":
        raise RuntimeError("Surface-2B requires Frozen Quadratic C0")
    summary = read_json(surface1a_summary_path)
    previous = summary.get("provenance", {})
    current = {
        "config_sha256": sha256(config_path),
        "quadratic_c0_sha256": sha256(app.calibration.laser_plane),
        "frozen_c1_sha256": sha256(app.calibration.laser_ray_correction),
        "intrinsics_sha256": sha256(app.calibration.intrinsics),
        "extrinsics_sha256": sha256(app.calibration.extrinsics),
    }
    for key, value in current.items():
        if previous.get(key) != value:
            raise RuntimeError(
                f"Frozen provenance mismatch for {key}: {value} != {previous.get(key)}"
            )
    coordinate = read_json(coordinate_path)
    center = np.asarray(
        laser_model["normalization"]["independent_center_mm"], dtype=np.float64
    )
    scale = np.asarray(
        laser_model["normalization"]["independent_scale_mm"], dtype=np.float64
    )
    if not np.allclose(center, coordinate["independent_center_mm"], atol=1e-12):
        raise RuntimeError("q coordinate center differs from Surface-1A")
    if not np.allclose(scale, coordinate["independent_scale_mm"], atol=1e-12):
        raise RuntimeError("q coordinate scale differs from Surface-1A")
    if list(laser_model["independent_axes"]) != list(coordinate["independent_axes"]):
        raise RuntimeError("q independent axes differ from Surface-1A")
    current.update(
        {
            "config_path": config_path,
            "quadratic_c0_path": app.calibration.laser_plane,
            "frozen_c1_path": app.calibration.laser_ray_correction,
            "surface1a_summary_path": surface1a_summary_path,
            "surface_coordinate_definition_path": coordinate_path,
            "q_definition_match": True,
        }
    )
    return app, calibration, correction, laser_model, {
        "current": current,
        "surface1a_reference": previous,
    }


def roi_masks(pixels: np.ndarray, roi: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    v = pixels[:, 1]
    height = (v >= int(roi["height_v_range"][0])) & (
        v <= int(roi["height_v_range"][1])
    )
    baseline = (
        (v >= int(roi["baseline_v_ranges"][0][0]))
        & (v <= int(roi["baseline_v_ranges"][0][1]))
    ) | (
        (v >= int(roi["baseline_v_ranges"][1][0]))
        & (v <= int(roi["baseline_v_ranges"][1][1]))
    )
    return height, baseline


def fit_new_proxies(
    cache: dict[tuple[str, str, int], np.ndarray],
    registry: dict[tuple[str, str], dict[str, Any]],
    calibration: Mapping[str, Any],
    params: Any,
    correction: Any,
    origin: np.ndarray,
    direction: np.ndarray,
    measurement_params: Any,
) -> tuple[dict[tuple[str, str], Any], list[dict[str, Any]]]:
    models: dict[tuple[str, str], Any] = {}
    rows: list[dict[str, Any]] = []
    for dataset in NEW_DATASETS:
        for pose in POSE_IDS:
            pixels = cache[(dataset, pose, 1)]
            _, baseline_mask = roi_masks(pixels, registry[(dataset, pose)])
            selected = pixels[baseline_mask]
            geometry = surface1a.evaluate_geometry(
                selected, calibration, params, correction
            )
            valid = geometry["valid"]
            ground = geometry["ground"][valid]
            S = (ground[:, :2] - origin[None, :]) @ direction
            Z = ground[:, 2]
            model = _fit_fixed_s_profile(S, Z, measurement_params)
            models[(dataset, pose)] = model
            clamped = np.asarray(geometry["c1_clamped"], dtype=bool)
            rows.append(
                {
                    "dataset": dataset,
                    "true_height_mm": TRUTH_MM[dataset],
                    "pose_id": pose,
                    "position_rank": int(registry[(dataset, pose)]["position_rank"]),
                    "selected_baseline_count": int(len(selected)),
                    "valid_baseline_count": int(np.count_nonzero(valid)),
                    "clamp_count": int(np.count_nonzero(clamped[valid])),
                    "clamp_rate": float(np.mean(clamped[valid])) if np.any(valid) else None,
                    "a_mm_per_mm": float(model.slope),
                    "b_mm": float(model.intercept),
                    "point_count": int(model.point_count),
                    "inlier_count": int(model.inlier_count),
                    "rmse_mm": float(model.rmse),
                    "S_span_mm": float(model.s_max - model.s_min),
                    "status": "success",
                }
            )
    return models, rows


def evaluate_new_frame(
    dataset: str,
    pose: str,
    repeat: int,
    pixels_all: np.ndarray,
    roi: Mapping[str, Any],
    model: Any,
    calibration: Mapping[str, Any],
    params: Any,
    correction: Any,
    laser_model: Mapping[str, Any],
    origin: np.ndarray,
    direction: np.ndarray,
    measurement_params: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    height_mask, _ = roi_masks(pixels_all, roi)
    indices = np.flatnonzero(height_mask)
    pixels = pixels_all[indices]
    if not len(pixels):
        return [], {
            "dataset": dataset,
            "true_height_mm": TRUTH_MM[dataset],
            "pose_id": pose,
            "position_rank": int(roi["position_rank"]),
            "repeat_index": repeat,
            "split_role": "surface2_calibration_repeat1" if repeat == 1 else "surface2_formal_repeat2_5",
            "source": "surface2_new_reconstructed",
            "selected_height_count": 0,
            "valid_height_count": 0,
            "analysis_point_count": 0,
            "status": "empty_height_roi",
        }
    base, derivatives = surface1a.evaluate_jacobian(
        pixels, calibration, params, correction, surface1a.JACOBIAN_EPS_PX
    )
    valid = np.asarray(base["valid"], dtype=bool)
    ground = base["ground"]
    inlier = np.zeros(len(pixels), dtype=bool)
    fit_status = "success"
    if int(np.count_nonzero(valid)) < int(measurement_params.min_height_points):
        fit_status = (
            f"too_few_points:{int(np.count_nonzero(valid))}<"
            f"{measurement_params.min_height_points}"
        )
    else:
        try:
            line = surface1a._fit_line_xy(
                ground[valid, :2], measurement_params, "height line"
            )
            valid_indices = np.flatnonzero(valid)
            inlier[valid_indices[line.inlier_mask]] = True
        except Exception as error:
            fit_status = f"{type(error).__name__}: {error}"
    coords = surface1a.surface_coordinates(base["P_c0"], laser_model)
    S = (ground[:, :2] - origin[None, :]) @ direction
    height_value = ground[:, 2] - (float(model.slope) * S + float(model.intercept))
    residual = height_value - TRUTH_MM[dataset]
    jacobian_valid = np.asarray(derivatives["jacobian_valid"], dtype=bool)
    analysis = valid & inlier & jacobian_valid & np.isfinite(residual)
    split_role = (
        "surface2_calibration_repeat1"
        if repeat == 1
        else "surface2_formal_repeat2_5"
    )
    rows: list[dict[str, Any]] = []
    for local_index in np.flatnonzero(valid):
        p0 = base["P_c0"][local_index]
        g = ground[local_index]
        rows.append(
            {
                "schema_version": 1,
                "source": "surface2_new_reconstructed",
                "dataset": dataset,
                "true_height_mm": TRUTH_MM[dataset],
                "pose_id": pose,
                "position_rank": int(roi["position_rank"]),
                "spatial_position_key": f"q1_rank_{int(roi['position_rank'])}",
                "repeat_index": repeat,
                "frame_id": f"{dataset}/pose{pose}/repeat{repeat}",
                "point_index": int(indices[local_index]),
                "split_role": split_role,
                "u": float(base["pixels"][local_index, 0]),
                "v": float(base["pixels"][local_index, 1]),
                "xn": float(base["xn"][local_index]),
                "yn": float(base["yn"][local_index]),
                "C1_s": float(base["c1_s"][local_index]),
                "C1_s_raw": float(base["c1_s_raw"][local_index]),
                "C1_s_clamped": bool(base["c1_clamped"][local_index]),
                "lambda_c0": float(base["lambda_c0"][local_index]),
                "lambda_c1": float(base["lambda_c1"][local_index]),
                "P_c0_x_mm": float(p0[0]),
                "P_c0_y_mm": float(p0[1]),
                "P_c0_z_mm": float(p0[2]),
                "q1": float(coords["q1"][local_index]),
                "q2": float(coords["q2"][local_index]),
                "q1_mm": float(coords["q1_mm"][local_index]),
                "q2_mm": float(coords["q2_mm"][local_index]),
                "Xg": float(g[0]),
                "Yg": float(g[1]),
                "Zg": float(g[2]),
                "S_mm": float(S[local_index]),
                "ground_proxy_a_mm_per_mm": float(model.slope),
                "ground_proxy_b_mm": float(model.intercept),
                "height_value_mm": float(height_value[local_index]),
                "height_residual_mm": float(residual[local_index]),
                "height_measurement_inlier": bool(inlier[local_index]),
                "jacobian_valid": bool(jacobian_valid[local_index]),
                "analysis_included": bool(analysis[local_index]),
                "height_fit_status": fit_status,
            }
        )
    valid_clamped = np.asarray(base["c1_clamped"], dtype=bool)[valid]
    return rows, {
        "dataset": dataset,
        "true_height_mm": TRUTH_MM[dataset],
        "pose_id": pose,
        "position_rank": int(roi["position_rank"]),
        "repeat_index": repeat,
        "split_role": split_role,
        "source": "surface2_new_reconstructed",
        "selected_height_count": int(len(pixels)),
        "valid_height_count": int(np.count_nonzero(valid)),
        "height_inlier_count": int(np.count_nonzero(inlier)),
        "jacobian_valid_count": int(np.count_nonzero(valid & jacobian_valid)),
        "analysis_point_count": int(np.count_nonzero(analysis)),
        "c1_clamp_count": int(np.count_nonzero(valid_clamped)),
        "c1_clamp_rate": float(np.mean(valid_clamped)) if len(valid_clamped) else None,
        "C1_s_raw_min": float(np.min(base["c1_s_raw"][valid])) if np.any(valid) else None,
        "C1_s_raw_max": float(np.max(base["c1_s_raw"][valid])) if np.any(valid) else None,
        "proxy_a_mm_per_mm": float(model.slope),
        "proxy_b_mm": float(model.intercept),
        "proxy_rmse_mm": float(model.rmse),
        "proxy_S_span_mm": float(model.s_max - model.s_min),
        "height_fit_status": fit_status,
        "status": "success" if fit_status == "success" else fit_status,
    }


def old_formal_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    accepted = {
        "development_formal_repeat2_5",
        "heldout_formal_repeat2_5",
    }
    for raw in read_csv(path):
        dataset = raw.get("dataset", "")
        if dataset not in {"obs_30mm", "obs_50mm"}:
            continue
        if raw.get("split_role") not in accepted:
            continue
        required = (
            "q1",
            "q2",
            "height_residual_mm",
            "u",
            "v",
            "lambda_c0",
            "lambda_c1",
            "Zg",
        )
        if not all(finite(raw.get(field)) for field in required):
            continue
        position_rank = int(raw["position_rank"])
        analysis = (
            as_bool(raw.get("height_measurement_inlier"))
            and as_bool(raw.get("jacobian_valid"))
        )
        output.append(
            {
                "schema_version": 1,
                "source": "surface1a_reused",
                "dataset": dataset,
                "true_height_mm": float(raw["true_height_mm"]),
                "pose_id": raw["pose_id"],
                "position_rank": position_rank,
                "spatial_position_key": f"q1_rank_{position_rank}",
                "repeat_index": int(raw["repeat_index"]),
                "frame_id": raw["frame_id"],
                "point_index": int(raw["point_index"]),
                "split_role": raw["split_role"],
                "u": float(raw["u"]),
                "v": float(raw["v"]),
                "xn": float(raw["xn"]),
                "yn": float(raw["yn"]),
                "C1_s": float(raw["C1_s"]),
                "C1_s_raw": float(raw["C1_s_raw"]),
                "C1_s_clamped": as_bool(raw["C1_s_clamped"]),
                "lambda_c0": float(raw["lambda_c0"]),
                "lambda_c1": float(raw["lambda_c1"]),
                "P_c0_x_mm": float(raw["P_c0_x_mm"]),
                "P_c0_y_mm": float(raw["P_c0_y_mm"]),
                "P_c0_z_mm": float(raw["P_c0_z_mm"]),
                "q1": float(raw["q1"]),
                "q2": float(raw["q2"]),
                "q1_mm": float(raw["q1_mm"]),
                "q2_mm": float(raw["q2_mm"]),
                "Xg": float(raw["Xg"]),
                "Yg": float(raw["Yg"]),
                "Zg": float(raw["Zg"]),
                "S_mm": float(raw["S_mm"]),
                "ground_proxy_a_mm_per_mm": float(raw["ground_proxy_a_mm_per_mm"]),
                "ground_proxy_b_mm": float(raw["ground_proxy_b_mm"]),
                "height_value_mm": float(raw["height_value_mm"]),
                "height_residual_mm": float(raw["height_residual_mm"]),
                "height_measurement_inlier": as_bool(raw["height_measurement_inlier"]),
                "jacobian_valid": as_bool(raw["jacobian_valid"]),
                "analysis_included": bool(analysis),
                "height_fit_status": raw.get("height_fit_status", ""),
            }
        )
    return output


def reused_frame_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["frame_id"]].append(row)
    result: list[dict[str, Any]] = []
    for frame_id, values in sorted(groups.items()):
        first = values[0]
        clamp = np.asarray([bool(row["C1_s_clamped"]) for row in values], dtype=bool)
        result.append(
            {
                "dataset": first["dataset"],
                "true_height_mm": first["true_height_mm"],
                "pose_id": first["pose_id"],
                "position_rank": first["position_rank"],
                "repeat_index": first["repeat_index"],
                "split_role": first["split_role"],
                "source": "surface1a_reused",
                "selected_height_count": None,
                "valid_height_count": len(values),
                "height_inlier_count": sum(bool(row["height_measurement_inlier"]) for row in values),
                "jacobian_valid_count": sum(bool(row["jacobian_valid"]) for row in values),
                "analysis_point_count": sum(bool(row["analysis_included"]) for row in values),
                "c1_clamp_count": int(np.count_nonzero(clamp)),
                "c1_clamp_rate": float(np.mean(clamp)),
                "C1_s_raw_min": min(float(row["C1_s_raw"]) for row in values),
                "C1_s_raw_max": max(float(row["C1_s_raw"]) for row in values),
                "status": "reused_formal_valid_rows",
            }
        )
    return result


def assign_q1_position_ranks(
    formal_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Define the cross-dataset spatial rank from q1, never from pose_id."""
    analysis = [row for row in formal_rows if row["analysis_included"]]
    mappings: dict[tuple[str, str], int] = {}
    audit_rows: list[dict[str, Any]] = []
    for height in HEIGHTS:
        selected = [row for row in analysis if row["true_height_mm"] == height]
        dataset = next((str(row["dataset"]) for row in selected), "")
        pose_medians = []
        for pose in sorted({str(row["pose_id"]) for row in selected}):
            values = [
                float(row["q1"]) for row in selected if str(row["pose_id"]) == pose
            ]
            if values:
                pose_medians.append((float(np.median(values)), pose))
        if len(pose_medians) != 5:
            raise RuntimeError(
                f"Expected five q1 position conditions at {height:g} mm, "
                f"got {len(pose_medians)}"
            )
        for rank, (q1_median, pose) in enumerate(sorted(pose_medians), start=1):
            mappings[(dataset, pose)] = rank
            audit_rows.append(
                {
                    "true_height_mm": height,
                    "dataset": dataset,
                    "pose_id": pose,
                    "position_rank": rank,
                    "q1_median": q1_median,
                    "definition": "ascending formal-analysis q1 median",
                }
            )
    for rows in (formal_rows, frame_rows, proxy_rows):
        for row in rows:
            key = (str(row["dataset"]), str(row["pose_id"]))
            if key not in mappings:
                raise RuntimeError(f"No q1 position-rank mapping for {key}")
            rank = mappings[key]
            row["position_rank"] = rank
            if "spatial_position_key" in row:
                row["spatial_position_key"] = f"q1_rank_{rank}"
    return audit_rows


def domain_and_condition_stats(
    formal_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    analysis = [row for row in formal_rows if row["analysis_included"]]
    domain_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    clamp_rows: list[dict[str, Any]] = []
    for height in HEIGHTS:
        selected = [row for row in analysis if row["true_height_mm"] == height]
        all_valid = [row for row in formal_rows if row["true_height_mm"] == height]
        domain_rows.append(
            {
                "true_height_mm": height,
                "dataset": next((row["dataset"] for row in selected), ""),
                "condition_count": len({row["position_rank"] for row in selected}),
                **distribution((row["q1"] for row in selected), "q1"),
                **distribution((row["q2"] for row in selected), "q2"),
                **residual_metrics(row["height_residual_mm"] for row in selected),
                "formal_valid_point_count": len(all_valid),
                "c1_clamp_count": sum(bool(row["C1_s_clamped"]) for row in all_valid),
                "c1_clamp_rate": (
                    sum(bool(row["C1_s_clamped"]) for row in all_valid) / len(all_valid)
                    if all_valid
                    else None
                ),
            }
        )
        for rank in range(1, 6):
            points = [
                row for row in selected if int(row["position_rank"]) == rank
            ]
            all_condition = [
                row
                for row in all_valid
                if int(row["position_rank"]) == rank
            ]
            pose_ids = sorted({row["pose_id"] for row in all_condition})
            condition_rows.append(
                {
                    "true_height_mm": height,
                    "dataset": next((row["dataset"] for row in points), ""),
                    "position_rank": rank,
                    "spatial_position_key": f"q1_rank_{rank}",
                    "source_pose_ids": ",".join(pose_ids),
                    "formal_repeat_count": len({row["repeat_index"] for row in all_condition}),
                    **distribution((row["q1"] for row in points), "q1"),
                    **distribution((row["q2"] for row in points), "q2"),
                    **residual_metrics(row["height_residual_mm"] for row in points),
                    "formal_valid_point_count": len(all_condition),
                    "c1_clamp_count": sum(bool(row["C1_s_clamped"]) for row in all_condition),
                    "c1_clamp_rate": (
                        sum(bool(row["C1_s_clamped"]) for row in all_condition)
                        / len(all_condition)
                        if all_condition
                        else None
                    ),
                }
            )
            clamp_rows.append(
                {
                    "true_height_mm": height,
                    "position_rank": rank,
                    "formal_valid_point_count": len(all_condition),
                    "c1_clamp_count": sum(bool(row["C1_s_clamped"]) for row in all_condition),
                    "c1_clamp_rate": (
                        sum(bool(row["C1_s_clamped"]) for row in all_condition)
                        / len(all_condition)
                        if all_condition
                        else None
                    ),
                    "C1_s_raw_min": min((row["C1_s_raw"] for row in all_condition), default=None),
                    "C1_s_raw_max": max((row["C1_s_raw"] for row in all_condition), default=None),
                }
            )
    return domain_rows, condition_rows, clamp_rows


def interval_relation(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float]:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    gap = max(0.0, max(a0, b0) - min(a1, b1))
    return gap, overlap


def q2_gap_payload(domain_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_height = {float(row["true_height_mm"]): row for row in domain_rows}
    pairs: list[dict[str, Any]] = []
    for lower, upper in zip(HEIGHTS[:-1], HEIGHTS[1:]):
        a = by_height[lower]
        b = by_height[upper]
        full_gap, full_overlap = interval_relation(
            float(a["q2_min"]), float(a["q2_max"]),
            float(b["q2_min"]), float(b["q2_max"]),
        )
        robust_gap, robust_overlap = interval_relation(
            float(a["q2_p05"]), float(a["q2_p95"]),
            float(b["q2_p05"]), float(b["q2_p95"]),
        )
        pairs.append(
            {
                "height_low_mm": lower,
                "height_high_mm": upper,
                "median_delta_q2_high_minus_low": float(b["q2_median"] - a["q2_median"]),
                "full_gap_q2": full_gap,
                "full_overlap_q2": full_overlap,
                "robust_p05_p95_gap_q2": robust_gap,
                "robust_p05_p95_overlap_q2": robust_overlap,
                "full_gap_within_q_tolerance": full_gap <= Q_TOLERANCE,
                "robust_gap_within_q_tolerance": robust_gap <= Q_TOLERANCE,
            }
        )
    medians = [float(by_height[height]["q2_median"]) for height in HEIGHTS]
    deltas = np.diff(medians)
    direction = "decreasing" if medians[-1] < medians[0] else "increasing"
    ordered = bool(np.all(deltas < 0)) if direction == "decreasing" else bool(np.all(deltas > 0))
    robust_pass = sum(bool(row["robust_gap_within_q_tolerance"]) for row in pairs)
    full_pass = sum(bool(row["full_gap_within_q_tolerance"]) for row in pairs)
    if ordered and robust_pass == len(pairs):
        status = "YES"
    elif ordered and (full_pass == len(pairs) or robust_pass >= len(pairs) - 1):
        status = "PARTIAL"
    else:
        status = "NO"
    return {
        "q_pair_tolerance_normalized": Q_TOLERANCE,
        "band_definitions": {
            "full": "point min..max",
            "robust": "point P05..P95",
        },
        "expected_q2_direction_from_30_to_50": direction,
        "q2_height_medians": dict(zip((str(item) for item in HEIGHTS), medians)),
        "strictly_ordered": ordered,
        "adjacent_pairs": pairs,
        "robust_pair_pass_count": robust_pass,
        "full_pair_pass_count": full_pass,
        "Q2_GAP_FILLED": status,
    }


def rank_residual_continuity(
    condition_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {
        (float(row["true_height_mm"]), int(row["position_rank"])): row
        for row in condition_rows
    }
    rows: list[dict[str, Any]] = []
    for rank in range(1, 6):
        for lower, upper in zip(HEIGHTS[:-1], HEIGHTS[1:]):
            a = by_key[(lower, rank)]
            b = by_key[(upper, rank)]
            delta = float(b["residual_bias_mm"] - a["residual_bias_mm"])
            rows.append(
                {
                    "position_rank": rank,
                    "height_low_mm": lower,
                    "height_high_mm": upper,
                    "q1_median_low": a["q1_median"],
                    "q1_median_high": b["q1_median"],
                    "q1_median_delta": float(b["q1_median"] - a["q1_median"]),
                    "q2_median_low": a["q2_median"],
                    "q2_median_high": b["q2_median"],
                    "q2_median_delta": float(b["q2_median"] - a["q2_median"]),
                    "residual_bias_low_mm": a["residual_bias_mm"],
                    "residual_bias_high_mm": b["residual_bias_mm"],
                    "residual_bias_delta_mm": delta,
                    "residual_bias_abs_delta_mm": abs(delta),
                    "same_q_state": False,
                }
            )
    values = np.asarray(
        [row["residual_bias_abs_delta_mm"] for row in rows], dtype=np.float64
    )
    return rows, {
        "definition": (
            "descriptive adjacent-height change at the same ascending-q1 rank; "
            "not a same-(q1,q2) comparison"
        ),
        "pair_count": len(rows),
        "median_abs_residual_bias_delta_mm": float(np.median(values)),
        "p95_abs_residual_bias_delta_mm": float(np.percentile(values, 95.0)),
        "max_abs_residual_bias_delta_mm": float(np.max(values)),
    }


def near_q_pair_metrics(
    formal_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analysis = [row for row in formal_rows if row["analysis_included"]]
    groups: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in analysis:
        groups[(float(row["true_height_mm"]), int(row["position_rank"]))].append(row)
    keys = sorted(groups)
    pair_rows: list[dict[str, Any]] = []
    for index, key_a in enumerate(keys):
        for key_b in keys[index + 1 :]:
            if key_a == key_b:
                continue
            a = groups[key_a]
            b = groups[key_b]
            qa = np.asarray([[row["q1"], row["q2"]] for row in a], dtype=np.float64)
            qb = np.asarray([[row["q1"], row["q2"]] for row in b], dtype=np.float64)
            ra = np.asarray([row["height_residual_mm"] for row in a], dtype=np.float64)
            rb = np.asarray([row["height_residual_mm"] for row in b], dtype=np.float64)
            tree_b = cKDTree(qb)
            dist_ab, near_ab = tree_b.query(
                qa, k=1, p=np.inf, distance_upper_bound=Q_TOLERANCE
            )
            mask_ab = np.isfinite(dist_ab) & (near_ab < len(b))
            tree_a = cKDTree(qa)
            dist_ba, near_ba = tree_a.query(
                qb, k=1, p=np.inf, distance_upper_bound=Q_TOLERANCE
            )
            mask_ba = np.isfinite(dist_ba) & (near_ba < len(a))
            abs_diff = []
            distances = []
            if np.any(mask_ab):
                abs_diff.extend(np.abs(ra[mask_ab] - rb[near_ab[mask_ab]]).tolist())
                distances.extend(dist_ab[mask_ab].tolist())
            if np.any(mask_ba):
                abs_diff.extend(np.abs(rb[mask_ba] - ra[near_ba[mask_ba]]).tolist())
                distances.extend(dist_ba[mask_ba].tolist())
            if not abs_diff:
                continue
            values = np.asarray(abs_diff, dtype=np.float64)
            distance_values = np.asarray(distances, dtype=np.float64)
            height_a, rank_a = key_a
            height_b, rank_b = key_b
            pair_rows.append(
                {
                    "height_a_mm": height_a,
                    "position_rank_a": rank_a,
                    "height_b_mm": height_b,
                    "position_rank_b": rank_b,
                    "same_height": height_a == height_b,
                    "same_position_rank": rank_a == rank_b,
                    "adjacent_height_pair": (
                        (height_a, height_b) in set(zip(HEIGHTS[:-1], HEIGHTS[1:]))
                    ),
                    "point_count_a": len(a),
                    "point_count_b": len(b),
                    "matched_from_a": int(np.count_nonzero(mask_ab)),
                    "matched_from_b": int(np.count_nonzero(mask_ba)),
                    "coverage_a": float(np.mean(mask_ab)),
                    "coverage_b": float(np.mean(mask_ba)),
                    "symmetric_match_count": int(len(values)),
                    "q_distance_median": float(np.median(distance_values)),
                    "q_distance_max": float(np.max(distance_values)),
                    "residual_abs_diff_median_mm": float(np.median(values)),
                    "residual_abs_diff_p95_mm": float(np.percentile(values, 95.0)),
                    "residual_abs_diff_max_mm": float(np.max(values)),
                }
            )
    eligible = [row for row in pair_rows if not row["same_height"]]
    pair_medians = np.asarray(
        [row["residual_abs_diff_median_mm"] for row in eligible], dtype=np.float64
    )
    pair_p95 = np.asarray(
        [row["residual_abs_diff_p95_mm"] for row in eligible], dtype=np.float64
    )
    adjacent_covered = {
        (row["height_a_mm"], row["height_b_mm"])
        for row in eligible
        if row["adjacent_height_pair"]
    }
    by_rank: dict[str, Any] = {}
    for rank in range(1, 6):
        selected = [
            row
            for row in eligible
            if row["position_rank_a"] == rank and row["position_rank_b"] == rank
        ]
        by_rank[str(rank)] = {
            "condition_pair_count": len(selected),
            "median_of_pair_medians_mm": (
                float(np.median([row["residual_abs_diff_median_mm"] for row in selected]))
                if selected
                else None
            ),
            "max_pair_p95_mm": (
                float(max(row["residual_abs_diff_p95_mm"] for row in selected))
                if selected
                else None
            ),
        }
    summary = {
        "q_tolerance_chebyshev": Q_TOLERANCE,
        "cross_height_condition_pair_count": len(eligible),
        "adjacent_height_pairs_with_matches": [list(item) for item in sorted(adjacent_covered)],
        "adjacent_height_pair_match_count": len(adjacent_covered),
        "median_of_condition_pair_median_abs_diff_mm": (
            float(np.median(pair_medians)) if len(pair_medians) else None
        ),
        "p95_of_condition_pair_median_abs_diff_mm": (
            float(np.percentile(pair_medians, 95.0)) if len(pair_medians) else None
        ),
        "median_of_condition_pair_p95_abs_diff_mm": (
            float(np.median(pair_p95)) if len(pair_p95) else None
        ),
        "by_same_position_rank": by_rank,
    }
    return pair_rows, summary


def classify_consistency(
    q2_status: str,
    near_summary: dict[str, Any],
    rank_trend: dict[str, Any],
) -> tuple[str, str]:
    adjacent = int(near_summary["adjacent_height_pair_match_count"])
    median_value = near_summary["median_of_condition_pair_median_abs_diff_mm"]
    p95_value = near_summary["p95_of_condition_pair_median_abs_diff_mm"]
    rank_values = [
        item["median_of_pair_medians_mm"]
        for item in near_summary["by_same_position_rank"].values()
        if item["median_of_pair_medians_mm"] is not None
    ]
    ranks_ok = len(rank_values) == 5 and max(rank_values) <= RESIDUAL_P95_SUPPORTED_MM
    supported = (
        q2_status == "YES"
        and adjacent == 4
        and median_value is not None
        and p95_value is not None
        and median_value <= RESIDUAL_MEDIAN_SUPPORTED_MM
        and p95_value <= RESIDUAL_P95_SUPPORTED_MM
        and ranks_ok
    )
    partial = (
        adjacent >= 2
        and median_value is not None
        and p95_value is not None
        and median_value <= RESIDUAL_MEDIAN_PARTIAL_MM
        and p95_value <= RESIDUAL_P95_PARTIAL_MM
    )
    trend_only_partial = (
        adjacent == 0
        and rank_trend["pair_count"] == 20
        and rank_trend["median_abs_residual_bias_delta_mm"]
        <= RESIDUAL_MEDIAN_SUPPORTED_MM
        and rank_trend["p95_abs_residual_bias_delta_mm"]
        <= RESIDUAL_P95_SUPPORTED_MM
    )
    if supported:
        consistency = "SUPPORTED"
    elif partial or trend_only_partial:
        consistency = "PARTIAL"
    else:
        consistency = "NOT_SUPPORTED"
    allowed = "YES" if q2_status == "YES" and consistency == "SUPPORTED" else "NO"
    return consistency, allowed


def make_plots(
    output: Path,
    formal_rows: list[dict[str, Any]],
    domain_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
) -> None:
    analysis = [row for row in formal_rows if row["analysis_included"]]
    colors = {
        30.0: "#1565c0",
        36.0: "#00897b",
        40.0: "#7cb342",
        46.0: "#fb8c00",
        50.0: "#c62828",
    }
    rng = np.random.default_rng(20260820)

    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for height in HEIGHTS:
        rows = [row for row in analysis if row["true_height_mm"] == height]
        if len(rows) > 1800:
            indices = rng.choice(len(rows), 1800, replace=False)
            rows = [rows[index] for index in indices]
        axis.scatter(
            [row["q1"] for row in rows],
            [row["q2"] for row in rows],
            s=5,
            alpha=0.22,
            linewidths=0,
            color=colors[height],
            label=f"{height:g} mm",
        )
    for row in condition_rows:
        axis.scatter(
            row["q1_median"], row["q2_median"],
            s=34, facecolors="none", edgecolors=colors[row["true_height_mm"]],
            linewidths=1.2,
        )
    axis.set_xlabel("q1 (Frozen C0 intrinsic coordinate)")
    axis.set_ylabel("q2 (Frozen C0 intrinsic coordinate)")
    axis.set_title("Surface-2B q1-q2 formal coverage")
    axis.grid(alpha=0.25)
    axis.legend(ncol=5)
    fig.savefig(output / "surface2b_q1_q2_coverage.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    rank_colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, 5))
    for rank, color in zip(range(1, 6), rank_colors):
        rows = sorted(
            [row for row in condition_rows if row["position_rank"] == rank],
            key=lambda row: row["true_height_mm"],
        )
        x = np.asarray([row["true_height_mm"] for row in rows])
        y = np.asarray([row["q2_median"] for row in rows])
        low = y - np.asarray([row["q2_p05"] for row in rows])
        high = np.asarray([row["q2_p95"] for row in rows]) - y
        axis.errorbar(
            x, y, yerr=np.vstack([low, high]), marker="o", linewidth=1.4,
            capsize=2, color=color, label=f"position_rank {rank}",
        )
    axis.plot(
        [row["true_height_mm"] for row in domain_rows],
        [row["q2_median"] for row in domain_rows],
        color="#212121", marker="s", linewidth=2.2, label="height pooled median",
    )
    axis.set_xlabel("true height [mm]")
    axis.set_ylabel("q2")
    axis.set_title("q2 continuity versus height (P05-P95 by spatial rank)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.savefig(output / "surface2b_q2_vs_height.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for height in HEIGHTS:
        rows = [row for row in analysis if row["true_height_mm"] == height]
        if len(rows) > 1600:
            indices = rng.choice(len(rows), 1600, replace=False)
            rows = [rows[index] for index in indices]
        q2 = np.asarray([row["q2"] for row in rows], dtype=np.float64)
        residual = np.asarray(
            [row["height_residual_mm"] for row in rows], dtype=np.float64
        )
        axis.scatter(
            q2, residual, s=5, alpha=0.18, linewidths=0,
            color=colors[height], label=f"{height:g} mm",
        )
        if len(q2) >= 20:
            edges = np.linspace(np.min(q2), np.max(q2), 13)
            centers = 0.5 * (edges[:-1] + edges[1:])
            medians = []
            valid_centers = []
            for left, right, center in zip(edges[:-1], edges[1:], centers):
                mask = (q2 >= left) & (q2 < right)
                if np.count_nonzero(mask) >= 5:
                    valid_centers.append(center)
                    medians.append(float(np.median(residual[mask])))
            axis.plot(valid_centers, medians, color=colors[height], linewidth=2.0)
    axis.axhline(0.0, color="#212121", linewidth=1.0)
    axis.set_xlabel("q2")
    axis.set_ylabel("raw session-linear height residual [mm]")
    axis.set_title("Raw residual versus Frozen-C0 q2 (descriptive bin medians)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=5)
    fig.savefig(output / "surface2b_raw_residual_vs_q2.png", dpi=180)
    plt.close(fig)


SAMPLE_FIELDS = [
    "schema_version", "source", "dataset", "true_height_mm", "pose_id",
    "position_rank", "spatial_position_key", "repeat_index", "frame_id",
    "point_index", "split_role", "u", "v", "xn", "yn", "C1_s",
    "C1_s_raw", "C1_s_clamped", "lambda_c0", "lambda_c1",
    "P_c0_x_mm", "P_c0_y_mm", "P_c0_z_mm", "q1", "q2", "q1_mm",
    "q2_mm", "Xg", "Yg", "Zg", "S_mm", "ground_proxy_a_mm_per_mm",
    "ground_proxy_b_mm", "height_value_mm", "height_residual_mm",
    "height_measurement_inlier", "jacobian_valid", "analysis_included",
    "height_fit_status",
]


def report_text(
    registry_info: dict[str, Any],
    provenance: dict[str, Any],
    domain_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    gap: dict[str, Any],
    near: dict[str, Any],
    rank_trend: dict[str, Any],
    consistency: str,
    allowed: str,
    new_proxy_rows: list[dict[str, Any]],
) -> str:
    domain_lines = []
    for row in domain_rows:
        domain_lines.append(
            "| {h:g} | {n} | {q1min:.4f} | {q1p05:.4f} | {q1med:.4f} | {q1p95:.4f} | {q1max:.4f} | {q2min:.4f} | {q2p05:.4f} | {q2med:.4f} | {q2p95:.4f} | {q2max:.4f} | {clamp}/{valid} ({rate:.2%}) |".format(
                h=row["true_height_mm"], n=row["q1_count"],
                q1min=row["q1_min"], q1p05=row["q1_p05"], q1med=row["q1_median"], q1p95=row["q1_p95"], q1max=row["q1_max"],
                q2min=row["q2_min"], q2p05=row["q2_p05"], q2med=row["q2_median"], q2p95=row["q2_p95"], q2max=row["q2_max"],
                clamp=row["c1_clamp_count"], valid=row["formal_valid_point_count"], rate=row["c1_clamp_rate"],
            )
        )
    gap_lines = []
    for row in gap["adjacent_pairs"]:
        gap_lines.append(
            "| {a:g}→{b:g} | {delta:.5f} | {fg:.5f} | {fo:.5f} | {rg:.5f} | {ro:.5f} | {ok} |".format(
                a=row["height_low_mm"], b=row["height_high_mm"], delta=row["median_delta_q2_high_minus_low"],
                fg=row["full_gap_q2"], fo=row["full_overlap_q2"], rg=row["robust_p05_p95_gap_q2"], ro=row["robust_p05_p95_overlap_q2"],
                ok=row["robust_gap_within_q_tolerance"],
            )
        )
    anomalies = []
    for row in condition_rows:
        if row["c1_clamp_count"] or row["formal_repeat_count"] != 4:
            anomalies.append(
                f"{row['true_height_mm']:g}mm/rank{row['position_rank']}: "
                f"clamp={row['c1_clamp_count']}/{row['formal_valid_point_count']}, "
                f"formal_repeats={row['formal_repeat_count']}"
            )
    if not anomalies:
        anomalies = ["无 condition-level clamp 或 formal repeat 缺失。"]
    condition_lines = []
    for row in condition_rows:
        condition_lines.append(
            "| {h:g} | {rank} | {pose} | {q1:.4f} | {q2:.4f} | {bias:.4f} | {rmse:.4f} | {p95:.4f} |".format(
                h=row["true_height_mm"], rank=row["position_rank"],
                pose=row["source_pose_ids"], q1=row["q1_median"],
                q2=row["q2_median"], bias=row["residual_bias_mm"],
                rmse=row["residual_rmse_mm"], p95=row["residual_p95_abs_mm"],
            )
        )
    worst_conditions = sorted(
        condition_rows, key=lambda row: abs(float(row["residual_bias_mm"])), reverse=True
    )[:3]
    return f"""# Surface-2B q1/q2 domain continuity 与 residual consistency 审计

## 结论

`Q2_GAP_FILLED={gap['Q2_GAP_FILLED']}`  
`Q1Q2_STATE_CONSISTENCY={consistency}`  
`SURFACE2C_ALLOWED={allowed}`

判定采用 Surface-1A 已冻结的 Chebyshev q tolerance `{Q_TOLERANCE}`。Residual consistency 的预先固定诊断阈值为：SUPPORTED 要求 condition-pair median ≤ `{RESIDUAL_MEDIAN_SUPPORTED_MM:.2f} mm` 且其 P95 ≤ `{RESIDUAL_P95_SUPPORTED_MM:.2f} mm`；PARTIAL 上限分别为 `{RESIDUAL_MEDIAN_PARTIAL_MM:.2f}/{RESIDUAL_P95_PARTIAL_MM:.2f} mm`。Surface-2C 仅在 q2=YES 且 consistency=SUPPORTED 时放行。

## Frozen provenance 与复用

- 人工 ROI：15/15 confirmed，final SHA `{registry_info['final_sha256']}`；draft entries 与 final 完全一致。
- Frozen C0 SHA：`{provenance['current']['quadratic_c0_sha256']}`。
- Frozen C1 SHA：`{provenance['current']['frozen_c1_sha256']}`。
- q1/q2 继续由 Frozen C0 的 `P_c0=lambda_c0*[xn,yn,1]` 及 Surface-1A center/scale 计算；C1 后坐标没有参与 q 定义。
- 30/50 mm 正式点直接复用 Surface-1A；36/40/46 mm 复用 75 帧一次-Steger cache，并用 frozen ROI 新增重建。
- repeat1 仅拟合当前 height×spatial-position 的 session-linear ground proxy；repeat2–5 为 formal。
- 未重拟 C0/C1，未按 residual 修改 ROI，未拟合 S0/S1/S2、Δh、Δlambda，也没有 random point split。

## Height-level q domain 与 C1 clamp

| height | analysis points | q1 min | q1 P05 | q1 median | q1 P95 | q1 max | q2 min | q2 P05 | q2 median | q2 P95 | q2 max | C1 clamp / formal valid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(domain_lines)}

Clamp rate 的分母是 frozen C1 重建成功的 formal height ROI 点；clamp 表示 Frozen C1 evaluator 在冻结 domain 边界取值，没有 extrapolation。

## q2 相邻 coverage

q2 median 的 30→50 方向为 `{gap['expected_q2_direction_from_30_to_50']}`；严格有序=`{gap['strictly_ordered']}`。

| adjacent height | median Δq2 | full gap | full overlap | P05–P95 gap | P05–P95 overlap | robust gap≤0.05 |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(gap_lines)}

## 相近 q1/q2 的 residual consistency

- 跨高度 condition-pair 数：`{near['cross_height_condition_pair_count']}`。
- 有近邻匹配的相邻 height pair：`{near['adjacent_height_pairs_with_matches']}`，共 `{near['adjacent_height_pair_match_count']}/4`。
- condition-pair median absolute residual difference 的中位数：`{near['median_of_condition_pair_median_abs_diff_mm']}` mm。
- 上述 condition-pair median 的 P95：`{near['p95_of_condition_pair_median_abs_diff_mm']}` mm。
- condition-pair P95 absolute difference 的中位数：`{near['median_of_condition_pair_p95_abs_diff_mm']}` mm。

按 q1 rank 对齐的 20 个相邻高度 condition（仅作趋势描述，不冒充同 q 状态）中，residual bias 的 |Δ| median/P95/max 为 `{rank_trend['median_abs_residual_bias_delta_mm']:.4f}/{rank_trend['p95_abs_residual_bias_delta_mm']:.4f}/{rank_trend['max_abs_residual_bias_delta_mm']:.4f} mm`。因此 residual-vs-height/q2 的离散轨迹没有突跳证据，但缺少 q-domain overlap，最多只能判为 `PARTIAL`。

由于相邻高度 q2 band 均没有进入 frozen q tolerance，本轮没有可辨识的跨高度“同一 q1/q2 状态”样本；因此不能升级为 `SUPPORTED`。这里的 `PARTIAL` 只来自 rank-level 趋势连续，不能解释为已经验证了同状态 residual 一致。

所有跨高度/空间比较使用实际 q1，并在每个高度内按 formal-analysis q1 median 从小到大定义 `position_rank=1..5`。原始 pose_id 只保留作 acquisition provenance，绝不直接作为跨高度统一位置。

### 按 q1 / position_rank 的描述性结果

| height | q1 rank | source pose | q1 median | q2 median | residual bias mm | RMSE mm | P95 abs mm |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(condition_lines)}

绝对 bias 最大的三个 condition（仅报告、不删点）：{'; '.join(f"{row['true_height_mm']:g}mm/rank{row['position_rank']}={row['residual_bias_mm']:.4f} mm" for row in worst_conditions)}。

## 异常 height/position

{chr(10).join('- ' + item for item in anomalies)}

新数据 15 个 repeat1 proxy 均成功：`{sum(row['status']=='success' for row in new_proxy_rows)}/15`。

## 输出

- `surface2b_samples.csv`：30/36/40/46/50 mm formal q/residual 点及 analysis flags。
- `surface2b_frame_metrics.csv`、`surface2b_ground_proxy_metrics.csv`、`surface2b_clamp_statistics.csv`。
- `surface2b_domain_statistics.csv`、`surface2b_condition_statistics.csv`。
- `surface2b_q2_gap_overlap.json`、`surface2b_q_near_pair_metrics.csv`、`surface2b_rank_residual_continuity.csv`、`surface2b_summary.json`。
- `surface2b_q1_q2_coverage.png`、`surface2b_q2_vs_height.png`、`surface2b_raw_residual_vs_q2.png`。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--center-cache", type=Path, default=DEFAULT_CENTER_CACHE)
    parser.add_argument("--surface1a", type=Path, default=DEFAULT_SURFACE1A)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry, registry_info = validate_registry(
        args.registry.resolve(), args.draft.resolve()
    )
    surface1a_dir = args.surface1a.resolve()
    coordinate_path = surface1a_dir / "surface_coordinate_definition.json"
    app, calibration, correction, laser_model, provenance = validate_frozen_provenance(
        args.config.resolve(), surface1a_dir / "surface1a_summary.json", coordinate_path
    )
    coordinate = read_json(coordinate_path)
    origin = np.asarray(coordinate["ground_origin_xy"], dtype=np.float64)
    direction = np.asarray(coordinate["ground_direction_xy"], dtype=np.float64)
    cache = load_center_cache(args.center_cache.resolve())
    models, proxy_rows = fit_new_proxies(
        cache, registry, calibration, app.reconstruction, correction,
        origin, direction, app.measurement,
    )
    new_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for dataset in NEW_DATASETS:
        for pose in POSE_IDS:
            for repeat in range(1, 6):
                rows, frame = evaluate_new_frame(
                    dataset, pose, repeat, cache[(dataset, pose, repeat)],
                    registry[(dataset, pose)], models[(dataset, pose)], calibration,
                    app.reconstruction, correction, laser_model, origin, direction,
                    app.measurement,
                )
                new_rows.extend(rows)
                frame_rows.append(frame)
    old_rows = old_formal_rows(surface1a_dir / "surface1a_points.csv")
    frame_rows.extend(reused_frame_metrics(old_rows))
    formal_rows = [
        row for row in new_rows if row["split_role"] == "surface2_formal_repeat2_5"
    ] + old_rows
    rank_mapping = assign_q1_position_ranks(formal_rows, frame_rows, proxy_rows)
    formal_rows.sort(
        key=lambda row: (
            float(row["true_height_mm"]), int(row["position_rank"]),
            int(row["repeat_index"]), int(row["point_index"]),
        )
    )
    frame_rows.sort(
        key=lambda row: (
            float(row["true_height_mm"]), int(row["position_rank"]),
            int(row["repeat_index"]),
        )
    )
    domain_rows, condition_rows, clamp_rows = domain_and_condition_stats(formal_rows)
    gap = q2_gap_payload(domain_rows)
    rank_trend_rows, rank_trend_summary = rank_residual_continuity(condition_rows)
    near_rows, near_summary = near_q_pair_metrics(formal_rows)
    consistency, allowed = classify_consistency(
        gap["Q2_GAP_FILLED"], near_summary, rank_trend_summary
    )

    write_csv(output / "surface2b_samples.csv", formal_rows, SAMPLE_FIELDS)
    write_csv(
        output / "surface2b_frame_metrics.csv", frame_rows,
        [
            "dataset", "true_height_mm", "pose_id", "position_rank", "repeat_index",
            "split_role", "source", "selected_height_count", "valid_height_count",
            "height_inlier_count", "jacobian_valid_count", "analysis_point_count",
            "c1_clamp_count", "c1_clamp_rate", "C1_s_raw_min", "C1_s_raw_max",
            "proxy_a_mm_per_mm", "proxy_b_mm", "proxy_rmse_mm", "proxy_S_span_mm",
            "height_fit_status", "status",
        ],
    )
    write_csv(
        output / "surface2b_ground_proxy_metrics.csv", proxy_rows,
        list(proxy_rows[0].keys()),
    )
    write_csv(
        output / "surface2b_domain_statistics.csv", domain_rows,
        list(domain_rows[0].keys()),
    )
    write_csv(
        output / "surface2b_condition_statistics.csv", condition_rows,
        list(condition_rows[0].keys()),
    )
    write_csv(
        output / "surface2b_clamp_statistics.csv", clamp_rows,
        list(clamp_rows[0].keys()),
    )
    write_csv(
        output / "surface2b_q_near_pair_metrics.csv", near_rows,
        list(near_rows[0].keys()) if near_rows else ["height_a_mm"],
    )
    write_csv(
        output / "surface2b_rank_residual_continuity.csv", rank_trend_rows,
        list(rank_trend_rows[0].keys()),
    )
    write_json(output / "surface2b_q2_gap_overlap.json", gap)
    make_plots(output, formal_rows, domain_rows, condition_rows)
    summary = {
        "Q2_GAP_FILLED": gap["Q2_GAP_FILLED"],
        "Q1Q2_STATE_CONSISTENCY": consistency,
        "SURFACE2C_ALLOWED": allowed,
        "registry": registry_info,
        "provenance": provenance,
        "q_definition": coordinate,
        "point_counts": {
            "formal_valid_total": len(formal_rows),
            "analysis_included_total": sum(row["analysis_included"] for row in formal_rows),
            "reused_30_50_formal_valid": len(old_rows),
            "new_36_40_46_formal_valid": len(formal_rows) - len(old_rows),
        },
        "domain_statistics": domain_rows,
        "condition_statistics": condition_rows,
        "position_rank_definition": {
            "rule": "within each height, ascending formal-analysis q1 median",
            "pose_id_cross_height_alignment": False,
            "mapping": rank_mapping,
        },
        "q2_gap_overlap": gap,
        "near_q_residual_consistency": near_summary,
        "rank_residual_continuity": rank_trend_summary,
        "classification_thresholds": {
            "q_tolerance_chebyshev": Q_TOLERANCE,
            "supported_median_abs_diff_mm": RESIDUAL_MEDIAN_SUPPORTED_MM,
            "supported_p95_abs_diff_mm": RESIDUAL_P95_SUPPORTED_MM,
            "partial_median_abs_diff_mm": RESIDUAL_MEDIAN_PARTIAL_MM,
            "partial_p95_abs_diff_mm": RESIDUAL_P95_PARTIAL_MM,
            "surface2c_gate": "Q2_GAP_FILLED=YES and Q1Q2_STATE_CONSISTENCY=SUPPORTED",
        },
        "constraints": {
            "c0_refit": False,
            "c1_refit": False,
            "q_redefined": False,
            "residual_driven_roi": False,
            "correction_fit": False,
            "random_point_split": False,
            "c1_extrapolation": False,
        },
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface2b_summary.json", summary)
    report = report_text(
        registry_info, provenance, domain_rows, condition_rows, gap,
        near_summary, rank_trend_summary, consistency, allowed, proxy_rows,
    )
    (output / "surface2b_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "Q2_GAP_FILLED": gap["Q2_GAP_FILLED"],
                "Q1Q2_STATE_CONSISTENCY": consistency,
                "SURFACE2C_ALLOWED": allowed,
                "formal_valid_points": len(formal_rows),
                "analysis_points": sum(row["analysis_included"] for row in formal_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
