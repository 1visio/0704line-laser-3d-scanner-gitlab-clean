"""Surface-1A diagnostic: C0 surface coordinates and local Jacobian attribution.

This module only consumes the already frozen Daheng C0/C1 artifacts and the
geometry-only ROI decisions.  It reconstructs the missing 50 mm point-level
rows from the cached one-pass Steger centers, then computes descriptive
surface/Jacobian relationships.  It never refits C0, C1, Ground G(S), or H1,
and it never writes production configuration.
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
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
TOOLS_ROOT = REPO_ROOT / "tools"
if str(MEASUREMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from measurement.height_measure import MeasurementParams, _fit_line_xy
from reconstruction.laser_ray_correction import evaluate_frozen_laser_ray_correction
from reconstruction.reconstructor import (
    _intersect_laser_surface,
    apply_ground_u_compensation,
    build_ground_transform,
)


DEFAULT_GROUND4A = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
DEFAULT_MANUAL_FROZEN = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_manual_frozen"
DEFAULT_HEIGHT50 = DEFAULT_GROUND4A / "height50_heldout"
DEFAULT_CONFIG = REPO_ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_GROUND3 = (
    MEASUREMENT_ROOT
    / "output_daheng_0811"
    / "ground_spatial_correction_ground3"
    / "ground_gs_summary.json"
)
DEFAULT_OUTPUT = DEFAULT_GROUND4A / "surface1a"

DEV_DATASETS = ("obs_1mm", "obs_2mm", "obs_6mm", "obs_10mm", "obs_20mm", "obs_30mm")
DEV_TRUE = {
    "obs_1mm": 1.001,
    "obs_2mm": 2.0,
    "obs_6mm": 6.0,
    "obs_10mm": 10.0,
    "obs_20mm": 20.0,
    "obs_30mm": 30.0,
}
DEV_ORDER = {name: index for index, name in enumerate(DEV_DATASETS)}
POSE_IDS = tuple(f"{index:03d}" for index in range(1, 6))
JACOBIAN_EPS_PX = 0.01
Q_PAIR_TOLERANCE = 0.05
EPS = 1.0e-12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON mapping: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if any(str(value or "").strip() for value in row.values())]


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(json_ready(value), ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def height_label(value: float) -> str:
    return "1mm" if abs(value - 1.001) < 1.0e-6 else f"{value:g}mm"


def axis_index(name: str) -> int:
    return {"X": 0, "Y": 1, "Z": 2}[str(name).upper()]


def array_json(value: np.ndarray) -> str:
    return json.dumps([float(item) for item in np.asarray(value).reshape(-1)], separators=(",", ":"))


def load_ground_reference(path: Path) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
    summary = read_json(path)
    protocol = summary.get("protocol", {})
    origin = np.asarray(protocol.get("origin_xy"), dtype=np.float64)
    direction = np.asarray(protocol.get("direction_xy"), dtype=np.float64)
    if origin.shape != (2,) or direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-8):
        raise RuntimeError("Ground-1 frozen origin/direction missing or not unit-normalized")
    domain_min = float(protocol["s_domain_min_mm"])
    domain_max = float(protocol["s_domain_max_mm"])
    if not domain_min < domain_max:
        raise RuntimeError("invalid frozen Ground-1 S domain")
    return origin, direction, domain_min, domain_max, summary


def load_proxies(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    rows = read_csv(path)
    proxies: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if row.get("proxy") != "session_linear":
            continue
        key = (row["dataset"], int(row["position_rank"]))
        proxies[key] = {
            "a": float(row["a_mm_per_mm"]),
            "b": float(row["b_mm"]),
            "s_span": float(row["S_span_mm"]),
            "point_count": float(row["point_count"]),
            "inlier_count": float(row["inlier_count"]),
            "rmse": float(row["rmse_mm"]),
        }
    expected = {(dataset, position) for dataset in DEV_DATASETS for position in range(1, 6)}
    if set(proxies) != expected:
        raise RuntimeError(f"Ground-4A session-linear proxy keys mismatch: {len(proxies)}")
    return proxies


def load_dev_frames(pointwise_path: Path, proxies: dict[tuple[str, int], dict[str, float]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    frames: dict[tuple[str, int, int], dict[str, Any]] = {}
    with pointwise_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("image_region") != "height" or row.get("c1_status") != "valid":
                continue
            dataset = row["dataset"]
            if dataset not in DEV_TRUE:
                continue
            position = int(row["position_rank"])
            repeat = int(row["repeat_index"])
            key = (dataset, position, repeat)
            frame = frames.setdefault(
                key,
                {
                    "dataset": dataset,
                    "height_group": height_label(DEV_TRUE[dataset]),
                    "true_height_mm": DEV_TRUE[dataset],
                    "pose_id": row["pose_id"],
                    "position_rank": position,
                    "position_id": f"laser{position:03d}",
                    "repeat_index": repeat,
                    "filename": row["filename"],
                    "proxy": proxies[(dataset, position)],
                    "held_out": False,
                    "split_role": "development_calibration_repeat1_in_sample" if repeat == 1 else "development_formal_repeat2_5",
                    "source": "manual_frozen/pointwise_diagnostics.csv",
                    "points": [],
                },
            )
            frame["points"].append(
                {
                    "point_index": int(row["point_index"]),
                    "u": float(row["u_px"]),
                    "v": float(row["v_px"]),
                    "expected_lambda_c0": float(row["lambda_c0_mm"]),
                    "expected_lambda_c1": float(row["lambda_c1_mm"]),
                    "expected_ground": np.asarray([float(row["c1_Xg_mm"]), float(row["c1_Yg_mm"]), float(row["c1_Zg_mm"])], dtype=np.float64),
                }
            )
    expected_frames = {(dataset, position, repeat) for dataset in DEV_DATASETS for position in range(1, 6) for repeat in range(1, 6)}
    if set(frames) != expected_frames:
        raise RuntimeError(f"Ground-4A height point frame keys mismatch: {len(frames)}/{len(expected_frames)}")
    return frames


def load_height50_frames(height50_dir: Path, height50_roi_path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    registry = read_json(height50_roi_path)
    if registry.get("protocol", {}).get("geometry_only") is not True or registry.get("protocol", {}).get("c0_c1_values_used") is not False:
        raise RuntimeError("50mm ROI registry is not geometry-only")
    if registry.get("summary", {}).get("manual_confirmed_count") != 5:
        raise RuntimeError("50mm ROI registry is not fully manually confirmed")
    entries = {str(item["pose_id"]): item for item in registry.get("entries", [])}
    frame_metrics = read_csv(height50_dir / "height50_frame_metrics.csv")
    if len(frame_metrics) != 25:
        raise RuntimeError(f"expected 25 height50 frame metrics, got {len(frame_metrics)}")
    frames: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in frame_metrics:
        pose_id = str(row["pose_id"])
        repeat = int(row["repeat_index"])
        cache_path = height50_dir / "center_cache" / f"laser{pose_id}_{repeat:02d}.npy"
        if not cache_path.is_file():
            raise RuntimeError(f"missing one-pass Steger cache: {cache_path}")
        centers = np.load(cache_path, allow_pickle=False).astype(np.float64, copy=False).reshape(-1, 2)
        roi = entries[pose_id]
        v0, v1 = (float(roi["height_v_range"][0]), float(roi["height_v_range"][1]))
        selected = np.isfinite(centers).all(axis=1) & (centers[:, 1] >= v0) & (centers[:, 1] <= v1)
        proxy = {
            "a": float(row["ground_proxy_a_mm_per_mm"]),
            "b": float(row["ground_proxy_b_mm"]),
            "s_span": float(row["ground_proxy_s_span_mm"]),
            "point_count": float(row["ground_proxy_point_count"]),
            "inlier_count": float(row["ground_proxy_inlier_count"]),
            "rmse": float(row["ground_proxy_rmse_mm"]),
        }
        if row.get("ground_proxy_status") != "success":
            raise RuntimeError(f"50mm ground proxy is not successful: {pose_id}/{repeat}")
        points = [
            {"point_index": int(index), "u": float(centers[index, 0]), "v": float(centers[index, 1])}
            for index in np.flatnonzero(selected)
        ]
        key = ("obs_50mm", int(pose_id), repeat)
        frames[key] = {
            "dataset": "obs_50mm",
            "height_group": "50mm",
            "true_height_mm": 50.0,
            "pose_id": pose_id,
            "position_rank": int(pose_id),
            "position_id": f"laser{pose_id}",
            "repeat_index": repeat,
            "filename": row["filename"],
            "proxy": proxy,
            "held_out": True,
            "split_role": "heldout_calibration_repeat1_in_sample" if repeat == 1 else "heldout_formal_repeat2_5",
            "source": "height50_heldout/center_cache + manual ROI",
            "points": points,
            "expected_height_point_count": int(row["height_point_count"]),
        }
    if len(frames) != 25:
        raise RuntimeError(f"expected 25 height50 frame keys, got {len(frames)}")
    return frames


def evaluate_geometry(
    pixels_uv: np.ndarray,
    calibration: Mapping[str, Any],
    params: Any,
    correction: Any,
) -> dict[str, np.ndarray]:
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        np.asarray(calibration["K"], dtype=np.float64),
        np.asarray(calibration["D"], dtype=np.float64),
    ).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
    lambda_c0, stable, model_type = _intersect_laser_surface(rays, calibration, params)
    if model_type != "quadratic_graph":
        raise RuntimeError(f"Surface-1A requires quadratic_graph C0, got {model_type}")
    c1_eval = evaluate_frozen_laser_ray_correction(rays, correction)
    lambda_c1 = np.asarray(lambda_c0 + c1_eval.correction_mm, dtype=np.float64)
    points_c0 = rays * lambda_c0[:, None]
    points_c1 = rays * lambda_c1[:, None]
    finite = np.isfinite(points_c1).all(axis=1) & np.isfinite(lambda_c1)
    positive = lambda_c1 > 0.0
    within_distance = (
        (points_c1[:, 2] >= float(params.min_camera_depth_mm))
        & (points_c1[:, 2] <= float(params.max_camera_depth_mm))
    )
    valid = stable & finite & positive & within_distance
    transform = build_ground_transform(calibration["R"], calibration["t"])
    homogeneous = np.column_stack([points_c1, np.ones(len(points_c1), dtype=np.float64)])
    points_ground = (transform @ homogeneous.T).T[:, :3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        points_ground = apply_ground_u_compensation(
            points_ground,
            pixels,
            calibration.get("ground_u_compensation"),
        )
    valid &= np.isfinite(points_ground).all(axis=1)
    return {
        "pixels": pixels,
        "xn": normalized[:, 0],
        "yn": normalized[:, 1],
        "rays": rays,
        "lambda_c0": np.asarray(lambda_c0, dtype=np.float64),
        "lambda_c1": lambda_c1,
        "c1_s_raw": c1_eval.s_raw,
        "c1_s": c1_eval.s_eval,
        "c1_clamped": c1_eval.clamped,
        "P_c0": points_c0,
        "P_c1": points_c1,
        "ground": points_ground,
        "valid": valid,
    }


def evaluate_jacobian(
    pixels: np.ndarray,
    calibration: Mapping[str, Any],
    params: Any,
    correction: Any,
    epsilon_px: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = evaluate_geometry(pixels, calibration, params, correction)
    plus_u = evaluate_geometry(pixels + np.array([epsilon_px, 0.0]), calibration, params, correction)
    minus_u = evaluate_geometry(pixels - np.array([epsilon_px, 0.0]), calibration, params, correction)
    plus_v = evaluate_geometry(pixels + np.array([0.0, epsilon_px]), calibration, params, correction)
    minus_v = evaluate_geometry(pixels - np.array([0.0, epsilon_px]), calibration, params, correction)
    scale = 2.0 * epsilon_px
    jac_valid_u = base["valid"] & plus_u["valid"] & minus_u["valid"]
    jac_valid_v = base["valid"] & plus_v["valid"] & minus_v["valid"]
    jac_valid = jac_valid_u & jac_valid_v
    derivatives = {
        "d_lambda_c0_du": np.divide(plus_u["lambda_c0"] - minus_u["lambda_c0"], scale, where=jac_valid_u, out=np.full(len(pixels), np.nan)),
        "d_lambda_c0_dv": np.divide(plus_v["lambda_c0"] - minus_v["lambda_c0"], scale, where=jac_valid_v, out=np.full(len(pixels), np.nan)),
        "d_lambda_du": np.divide(plus_u["lambda_c1"] - minus_u["lambda_c1"], scale, where=jac_valid_u, out=np.full(len(pixels), np.nan)),
        "d_lambda_dv": np.divide(plus_v["lambda_c1"] - minus_v["lambda_c1"], scale, where=jac_valid_v, out=np.full(len(pixels), np.nan)),
        "d_Zg_du": np.divide(plus_u["ground"][:, 2] - minus_u["ground"][:, 2], scale, where=jac_valid_u, out=np.full(len(pixels), np.nan)),
        "d_Zg_dv": np.divide(plus_v["ground"][:, 2] - minus_v["ground"][:, 2], scale, where=jac_valid_v, out=np.full(len(pixels), np.nan)),
        "jacobian_valid": jac_valid,
    }
    derivatives["jacobian_lambda_norm_mm_per_px"] = np.sqrt(derivatives["d_lambda_du"] ** 2 + derivatives["d_lambda_dv"] ** 2)
    derivatives["jacobian_Zg_norm_mm_per_px"] = np.sqrt(derivatives["d_Zg_du"] ** 2 + derivatives["d_Zg_dv"] ** 2)
    derivatives["jacobian_combined_norm"] = np.sqrt(
        derivatives["jacobian_lambda_norm_mm_per_px"] ** 2
        + derivatives["jacobian_Zg_norm_mm_per_px"] ** 2
    )
    return base, derivatives


def surface_coordinates(points_c0: np.ndarray, laser_model: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if laser_model.get("model_type") != "quadratic_graph":
        raise RuntimeError("surface coordinates require quadratic_graph")
    independent = [axis_index(value) for value in laser_model["independent_axes"]]
    normalization = laser_model["normalization"]
    center = np.asarray(normalization["independent_center_mm"], dtype=np.float64)
    scale = np.asarray(normalization["independent_scale_mm"], dtype=np.float64)
    raw = np.column_stack([points_c0[:, independent[0]], points_c0[:, independent[1]]])
    normalized = (raw - center[None, :]) / scale[None, :]
    return {
        "q1_mm": raw[:, 0],
        "q2_mm": raw[:, 1],
        "q1": normalized[:, 0],
        "q2": normalized[:, 1],
    }


def build_point_rows(
    frame: dict[str, Any],
    calibration: Mapping[str, Any],
    params: Any,
    correction: Any,
    laser_model: Mapping[str, Any],
    origin: np.ndarray,
    direction: np.ndarray,
    measurement_params: MeasurementParams,
    consistency: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points = frame["points"]
    pixels = np.asarray([[item["u"], item["v"]] for item in points], dtype=np.float64)
    if len(pixels) == 0:
        return [], {"point_count": 0, "valid_count": 0, "inlier_count": 0, "jacobian_valid_count": 0, "height_fit_status": "empty"}
    base, derivatives = evaluate_jacobian(pixels, calibration, params, correction, JACOBIAN_EPS_PX)
    valid = base["valid"]
    valid_count = int(np.count_nonzero(valid))
    if frame["held_out"] and "expected_height_point_count" in frame:
        if valid_count != int(frame["expected_height_point_count"]):
            raise RuntimeError(
                f"50mm point count mismatch {frame['pose_id']}/repeat{frame['repeat_index']}: "
                f"{valid_count} != {frame['expected_height_point_count']}"
            )
    if not frame["held_out"] and valid_count != len(points):
        raise RuntimeError(f"Ground-4A valid height point mismatch at {frame['dataset']}/{frame['position_rank']}/repeat{frame['repeat_index']}")

    ground = base["ground"]
    inlier_mask = np.zeros(len(points), dtype=bool)
    height_fit_status = "success"
    if valid_count < measurement_params.min_height_points:
        height_fit_status = f"too_few_points:{valid_count}<{measurement_params.min_height_points}"
    else:
        try:
            height_fit = _fit_line_xy(ground[valid, :2], measurement_params, "height line")
            valid_indices = np.flatnonzero(valid)
            inlier_mask[valid_indices[height_fit.inlier_mask]] = True
        except Exception as error:
            height_fit_status = f"{type(error).__name__}: {error}"

    coords = surface_coordinates(base["P_c0"], laser_model)
    S = (ground[:, :2] - origin[None, :]) @ direction
    proxy_a = float(frame["proxy"]["a"])
    proxy_b = float(frame["proxy"]["b"])
    height_values = ground[:, 2] - (proxy_a * S + proxy_b)
    residuals = height_values - float(frame["true_height_mm"])

    expected_lambda0 = np.asarray([item.get("expected_lambda_c0", np.nan) for item in points], dtype=np.float64)
    expected_lambda1 = np.asarray([item.get("expected_lambda_c1", np.nan) for item in points], dtype=np.float64)
    expected_ground = np.asarray([item.get("expected_ground", [np.nan, np.nan, np.nan]) for item in points], dtype=np.float64)
    if np.any(np.isfinite(expected_lambda0)):
        consistency["max_abs_lambda_c0_mm"] = max(consistency["max_abs_lambda_c0_mm"], float(np.nanmax(np.abs(base["lambda_c0"][valid] - expected_lambda0[valid]))))
    if np.any(np.isfinite(expected_lambda1)):
        consistency["max_abs_lambda_c1_mm"] = max(consistency["max_abs_lambda_c1_mm"], float(np.nanmax(np.abs(base["lambda_c1"][valid] - expected_lambda1[valid]))))
    if np.any(np.isfinite(expected_ground)):
        consistency["max_abs_Zg_mm"] = max(consistency["max_abs_Zg_mm"], float(np.nanmax(np.abs(ground[valid, 2] - expected_ground[valid, 2]))))

    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(valid):
        p0 = base["P_c0"][index]
        p1 = base["P_c1"][index]
        g = ground[index]
        jac_valid = bool(derivatives["jacobian_valid"][index])
        rows.append(
            {
                "schema_version": 1,
                "dataset": frame["dataset"],
                "height_group": frame["height_group"],
                "true_height_mm": float(frame["true_height_mm"]),
                "true_height": float(frame["true_height_mm"]),
                "pose_id": frame["pose_id"],
                "position_id": frame["position_id"],
                "position": frame["position_id"],
                "position_rank": int(frame["position_rank"]),
                "repeat_index": int(frame["repeat_index"]),
                "filename": frame["filename"],
                "frame_id": f"{frame['dataset']}/{frame['position_id']}/repeat{frame['repeat_index']}",
                "point_index": int(points[index]["point_index"]),
                "split_role": frame["split_role"],
                "held_out": bool(frame["held_out"]),
                "u": float(base["pixels"][index, 0]),
                "v": float(base["pixels"][index, 1]),
                "xn": float(base["xn"][index]),
                "yn": float(base["yn"][index]),
                "C1_s": float(base["c1_s"][index]),
                "C1_s_raw": float(base["c1_s_raw"][index]),
                "C1_s_clamped": bool(base["c1_clamped"][index]),
                "lambda_c0": float(base["lambda_c0"][index]),
                "lambda_c1": float(base["lambda_c1"][index]),
                "delta_lambda_c1_minus_c0": float(base["lambda_c1"][index] - base["lambda_c0"][index]),
                "P_c0": array_json(p0),
                "P_c1": array_json(p1),
                "P_c0_x_mm": float(p0[0]),
                "P_c0_y_mm": float(p0[1]),
                "P_c0_z_mm": float(p0[2]),
                "P_c1_x_mm": float(p1[0]),
                "P_c1_y_mm": float(p1[1]),
                "P_c1_z_mm": float(p1[2]),
                "q1": float(coords["q1"][index]),
                "q2": float(coords["q2"][index]),
                "q1_mm": float(coords["q1_mm"][index]),
                "q2_mm": float(coords["q2_mm"][index]),
                "Xg": float(g[0]),
                "Yg": float(g[1]),
                "Zg": float(g[2]),
                "S_mm": float(S[index]),
                "ground_proxy_a_mm_per_mm": proxy_a,
                "ground_proxy_b_mm": proxy_b,
                "height_value_mm": float(height_values[index]),
                "height_residual_mm": float(residuals[index]),
                "height_residual": float(residuals[index]),
                "height_measurement_inlier": bool(inlier_mask[index]),
                "height_fit_status": height_fit_status,
                "d_lambda_c0_du": float(derivatives["d_lambda_c0_du"][index]),
                "d_lambda_c0_dv": float(derivatives["d_lambda_c0_dv"][index]),
                "d_lambda_du": float(derivatives["d_lambda_du"][index]),
                "d_lambda_dv": float(derivatives["d_lambda_dv"][index]),
                "d_Zg_du": float(derivatives["d_Zg_du"][index]),
                "d_Zg_dv": float(derivatives["d_Zg_dv"][index]),
                "d_lambda/du": float(derivatives["d_lambda_du"][index]),
                "d_lambda/dv": float(derivatives["d_lambda_dv"][index]),
                "d_Zg/du": float(derivatives["d_Zg_du"][index]),
                "d_Zg/dv": float(derivatives["d_Zg_dv"][index]),
                "jacobian_valid": jac_valid,
                "jacobian_lambda_norm_mm_per_px": float(derivatives["jacobian_lambda_norm_mm_per_px"][index]),
                "jacobian_Zg_norm_mm_per_px": float(derivatives["jacobian_Zg_norm_mm_per_px"][index]),
                "jacobian_combined_norm": float(derivatives["jacobian_combined_norm"][index]),
            }
        )
    return rows, {
        "point_count": len(points),
        "valid_count": valid_count,
        "inlier_count": int(np.count_nonzero(inlier_mask)),
        "jacobian_valid_count": int(np.count_nonzero(derivatives["jacobian_valid"] & valid)),
        "height_fit_status": height_fit_status,
        "height_residual_mean_mm": float(np.mean(residuals[valid])) if valid_count else None,
        "height_residual_inlier_mean_mm": float(np.mean(residuals[inlier_mask])) if np.any(inlier_mask) else None,
    }


FEATURE_MODELS: dict[str, list[str]] = {
    "v": ["v"],
    "C1_s": ["C1_s"],
    "height": ["true_height_mm"],
    "q1": ["q1"],
    "(q1,q2)": ["q1", "q2"],
    "J_lambda_norm": ["jacobian_lambda_norm_mm_per_px"],
    "J_Zg_norm": ["jacobian_Zg_norm_mm_per_px"],
    "Jacobian": ["d_lambda_du", "d_lambda_dv", "d_Zg_du", "d_Zg_dv"],
    "v+C1_s": ["v", "C1_s"],
    "height+q1+q2": ["true_height_mm", "q1", "q2"],
    "q1+q2+Jacobian": ["q1", "q2", "d_lambda_du", "d_lambda_dv", "d_Zg_du", "d_Zg_dv"],
}

FORMAL_SPLIT_ROLES = {"development_formal_repeat2_5", "heldout_formal_repeat2_5"}


def finite_records(
    rows: list[dict[str, Any]],
    held_out: bool | None = None,
    formal_only: bool = False,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if held_out is not None and bool(row["held_out"]) != held_out:
            continue
        if formal_only and row["split_role"] not in FORMAL_SPLIT_ROLES:
            continue
        if not bool(row["height_measurement_inlier"]) or not bool(row["jacobian_valid"]):
            continue
        values = [row["height_residual_mm"]]
        for feature_list in FEATURE_MODELS.values():
            values.extend(row[name] for name in feature_list)
        if all(math.isfinite(float(value)) for value in values):
            output.append(row)
    return output


def condition_balanced(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['dataset']}/{row['position_id']}"].append(row)
    output = []
    for condition_id, group in sorted(groups.items()):
        item = {
            "condition_id": condition_id,
            "dataset": group[0]["dataset"],
            "position_id": group[0]["position_id"],
            "height_group": group[0]["height_group"],
            "true_height_mm": float(np.mean([float(row["true_height_mm"]) for row in group])),
            "held_out": bool(group[0]["held_out"]),
            "height_residual_mm": float(np.mean([float(row["height_residual_mm"]) for row in group])),
            "point_count": len(group),
        }
        for name in {feature for values in FEATURE_MODELS.values() for feature in values}:
            item[name] = float(np.mean([float(row[name]) for row in group]))
        output.append(item)
    return output


def rank_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.ptp(x) <= EPS or np.ptp(y) <= EPS:
        return None
    value = spearmanr(x, y).statistic
    return None if value is None or not math.isfinite(float(value)) else float(value)


def within_condition_stats(rows: list[dict[str, Any]], features: list[str]) -> tuple[float | None, float | None, float | None, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['dataset']}/{row['position_id']}"].append(row)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        x = np.asarray([[float(row[name]) for name in features] for row in group], dtype=np.float64)
        y = np.asarray([float(row["height_residual_mm"]) for row in group], dtype=np.float64)
        x = x - np.mean(x, axis=0, keepdims=True)
        y = y - np.mean(y)
        x_parts.append(x)
        y_parts.append(y)
    if not y_parts:
        return None, None, None, 0
    x_all = np.vstack(x_parts)
    y_all = np.concatenate(y_parts)
    nonzero = np.std(x_all, axis=0) > EPS
    if not np.any(nonzero) or np.sum(y_all**2) <= EPS:
        return None, None, None, len(y_all)
    x_use = x_all[:, nonzero]
    beta = np.linalg.lstsq(x_use, y_all, rcond=None)[0]
    predicted = x_use @ beta
    sst = float(np.sum(y_all**2))
    r2 = 1.0 - float(np.sum((y_all - predicted) ** 2)) / sst
    if len(features) == 1:
        pearson = float(np.corrcoef(x_use[:, 0], y_all)[0, 1]) if np.ptp(x_use[:, 0]) > EPS and np.ptp(y_all) > EPS else None
        spearman = rank_corr(x_use[:, 0], y_all)
    else:
        pearson = None
        spearman = None
    return float(r2), pearson, spearman, len(y_all)


def explanatory_metrics(rows: list[dict[str, Any]], scope: str, held_out: bool | None) -> list[dict[str, Any]]:
    selected = finite_records(rows, held_out=held_out, formal_only=True)
    aggregations = [("point_pooled", selected), ("condition_balanced", condition_balanced(selected))]
    output: list[dict[str, Any]] = []
    for aggregation, data in aggregations:
        for model, features in FEATURE_MODELS.items():
            if not data:
                output.append({"scope": scope, "aggregation": aggregation, "model": model, "feature_list": "+".join(features), "fit_status": "empty"})
                continue
            x = np.asarray([[float(row[name]) for name in features] for row in data], dtype=np.float64)
            y = np.asarray([float(row["height_residual_mm"]) for row in data], dtype=np.float64)
            feature_std = np.std(x, axis=0)
            rank = int(np.linalg.matrix_rank(x - np.mean(x, axis=0, keepdims=True)))
            fit_status = "success"
            if len(data) <= rank + 1 or np.any(feature_std <= EPS):
                fit_status = "degenerate_feature_or_sample_count"
                r2 = rmse = None
            else:
                x_scaled = (x - np.mean(x, axis=0, keepdims=True)) / feature_std[None, :]
                design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
                beta = np.linalg.lstsq(design, y, rcond=None)[0]
                predicted = design @ beta
                sse = float(np.sum((y - predicted) ** 2))
                sst = float(np.sum((y - np.mean(y)) ** 2))
                r2 = None if sst <= EPS else float(1.0 - sse / sst)
                rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))
            pearson = rank_value = None
            if len(features) == 1:
                pearson = float(np.corrcoef(x[:, 0], y)[0, 1]) if np.ptp(x[:, 0]) > EPS and np.ptp(y) > EPS else None
                rank_value = rank_corr(x[:, 0], y)
            within_r2, within_pearson, within_spearman, within_n = within_condition_stats(selected, features)
            output.append(
                {
                    "scope": scope,
                    "aggregation": aggregation,
                    "model": model,
                    "feature_list": "+".join(features),
                    "fit_status": fit_status,
                    "sample_count": len(data),
                    "point_count": len(selected),
                    "condition_count": len({f"{row['dataset']}/{row['position_id']}" for row in selected}),
                    "frame_count": len({row["frame_id"] for row in selected}),
                    "height_group_count": len({row["height_group"] for row in selected}),
                    "position_count": len({row["position_id"] for row in selected}),
                    "residual_bias_mm": float(np.mean(y)),
                    "residual_rmse_about_mean_mm": float(np.sqrt(np.mean((y - np.mean(y)) ** 2))),
                    "descriptive_ols_rmse_mm": rmse,
                    "descriptive_ols_r2": r2,
                    "pearson_r": pearson,
                    "spearman_r": rank_value,
                    "within_condition_r2": within_r2,
                    "within_condition_pearson_r": within_pearson,
                    "within_condition_spearman_r": within_spearman,
                    "within_condition_point_count": within_n,
                    "held_out": held_out is True,
                    "used_for_decision": held_out is not True,
                    "fit_protocol": "descriptive OLS only; no correction function fitted",
                }
            )
    return output


def q_domain(rows: list[dict[str, Any]], held_out: bool) -> dict[str, Any]:
    selected = [
        row for row in rows
        if bool(row["held_out"]) is held_out
        and row["split_role"] in FORMAL_SPLIT_ROLES
        and bool(row["jacobian_valid"])
    ]
    if not selected:
        return {"count": 0}
    q1 = np.asarray([float(row["q1"]) for row in selected])
    q2 = np.asarray([float(row["q2"]) for row in selected])
    return {
        "count": len(selected),
        "q1_min": float(np.min(q1)),
        "q1_max": float(np.max(q1)),
        "q2_min": float(np.min(q2)),
        "q2_max": float(np.max(q2)),
    }


def q_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = [
        row for row in rows
        if not bool(row["held_out"])
        and row["split_role"] in FORMAL_SPLIT_ROLES
        and bool(row["jacobian_valid"])
    ]
    held = [
        row for row in rows
        if bool(row["held_out"])
        and row["split_role"] in FORMAL_SPLIT_ROLES
        and bool(row["jacobian_valid"])
    ]
    if not dev or not held:
        return {"heldout_count": len(held), "heldout_inside_development_bbox_rate": None}
    d1 = np.asarray([row["q1"] for row in dev]); d2 = np.asarray([row["q2"] for row in dev])
    h1 = np.asarray([row["q1"] for row in held]); h2 = np.asarray([row["q2"] for row in held])
    inside = (h1 >= np.min(d1)) & (h1 <= np.max(d1)) & (h2 >= np.min(d2)) & (h2 <= np.max(d2))
    return {
        "heldout_count": len(held),
        "heldout_inside_development_bbox_count": int(np.count_nonzero(inside)),
        "heldout_inside_development_bbox_rate": float(np.mean(inside)),
        "development_q_bbox": {"q1": [float(np.min(d1)), float(np.max(d1))], "q2": [float(np.min(d2)), float(np.max(d2))]},
        "heldout_q_bbox": {"q1": [float(np.min(h1)), float(np.max(h1))], "q2": [float(np.min(h2)), float(np.max(h2))]},
    }


def q_similarity_pairs(rows: list[dict[str, Any]], tolerance: float = Q_PAIR_TOLERANCE) -> dict[str, Any]:
    selected = finite_records(rows, held_out=None, formal_only=True)
    conditions = condition_balanced(selected)
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(conditions):
        for right in conditions[i + 1 :]:
            if left["condition_id"] == right["condition_id"]:
                continue
            dq1 = abs(float(left["q1"]) - float(right["q1"]))
            dq2 = abs(float(left["q2"]) - float(right["q2"]))
            if max(dq1, dq2) <= tolerance:
                pairs.append(
                    {
                        "left": left["condition_id"],
                        "right": right["condition_id"],
                        "left_height_group": left["height_group"],
                        "right_height_group": right["height_group"],
                        "q_max_abs_difference": max(dq1, dq2),
                        "abs_residual_difference_mm": abs(float(left["height_residual_mm"]) - float(right["height_residual_mm"])),
                    }
                )
    differences = np.asarray([row["abs_residual_difference_mm"] for row in pairs], dtype=np.float64)
    condition_residuals = np.asarray([float(row["height_residual_mm"]) for row in conditions], dtype=np.float64)
    return {
        "tolerance_normalized": tolerance,
        "condition_count": len(conditions),
        "pair_count": len(pairs),
        "pair_median_abs_residual_difference_mm": float(np.median(differences)) if len(differences) else None,
        "pair_p95_abs_residual_difference_mm": float(np.percentile(differences, 95.0)) if len(differences) else None,
        "all_condition_residual_median_abs_deviation_mm": float(np.median(np.abs(condition_residuals - np.median(condition_residuals)))) if len(condition_residuals) else None,
        "pairs": pairs,
    }


def lookup_metric(metrics: list[dict[str, Any]], scope: str, aggregation: str, model: str) -> dict[str, Any]:
    for row in metrics:
        if row.get("scope") == scope and row.get("aggregation") == aggregation and row.get("model") == model:
            return row
    return {}


def classify(metrics: list[dict[str, Any]], overlap: dict[str, Any]) -> dict[str, str]:
    dev_point_q = lookup_metric(metrics, "development_formal", "point_pooled", "(q1,q2)")
    dev_point_h = lookup_metric(metrics, "development_formal", "point_pooled", "height")
    dev_bal_q = lookup_metric(metrics, "development_formal", "condition_balanced", "(q1,q2)")
    dev_bal_h = lookup_metric(metrics, "development_formal", "condition_balanced", "height")
    surface_values = [
        value for value in (dev_point_q.get("within_condition_r2"), dev_point_q.get("descriptive_ols_r2"), dev_bal_q.get("descriptive_ols_r2"))
        if value is not None and math.isfinite(float(value))
    ]
    q_within = max([float(dev_point_q.get("within_condition_r2") or 0.0)], default=0.0)
    q_global = max([float(value) for value in surface_values], default=0.0)
    h_global = max(float(dev_point_h.get("descriptive_ols_r2") or 0.0), float(dev_bal_h.get("descriptive_ols_r2") or 0.0))
    overlap_rate = float(overlap.get("heldout_inside_development_bbox_rate") or 0.0)
    if q_within >= 0.05 and overlap_rate >= 0.20:
        surface_status = "SUPPORTED"
    elif q_within >= 0.02 or (q_global >= h_global + 0.05 and overlap_rate >= 0.20):
        surface_status = "PARTIAL"
    else:
        surface_status = "NOT_SUPPORTED"

    j_candidates = []
    for model in ("J_lambda_norm", "J_Zg_norm", "Jacobian"):
        row = lookup_metric(metrics, "development_formal", "point_pooled", model)
        for key in ("within_condition_r2", "spearman_r", "within_condition_spearman_r"):
            value = row.get(key)
            if value is not None and math.isfinite(float(value)):
                j_candidates.append(abs(float(value)))
    j_signal = max(j_candidates, default=0.0)
    if j_signal >= 0.25:
        jacobian_status = "SUPPORTED"
    elif j_signal >= 0.15:
        jacobian_status = "PARTIAL"
    else:
        jacobian_status = "NOT_SUPPORTED"

    if h_global >= q_global + 0.05 and q_within < 0.02:
        height_only = "SUPPORTED"
    elif surface_status == "NOT_SUPPORTED" and jacobian_status == "NOT_SUPPORTED":
        height_only = "PARTIAL"
    else:
        height_only = "INSUFFICIENT"
    surface_aware = "YES_DIAGNOSTIC_ONLY" if surface_status != "NOT_SUPPORTED" or jacobian_status != "NOT_SUPPORTED" else "NO"
    return {
        "SURFACE_COORDINATE_EXPLANATORY_POWER": surface_status,
        "JACOBIAN_DEPENDENCE": jacobian_status,
        "HEIGHT_ONLY_MODEL": height_only,
        "SURFACE_AWARE_CORRECTION_RECOMMENDED": surface_aware,
    }


def downsample(rows: list[dict[str, Any]], limit: int = 16000) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    step = int(math.ceil(len(rows) / limit))
    return rows[::step]


def plot_surface(path: Path, rows: list[dict[str, Any]]) -> None:
    formal = [row for row in rows if row["split_role"] in {"development_formal_repeat2_5", "heldout_formal_repeat2_5"} and row["height_measurement_inlier"] and row["jacobian_valid"]]
    sample = downsample(formal)
    dev = [row for row in sample if not row["held_out"]]
    held = [row for row in sample if row["held_out"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, feature, label in zip(axes.flat, ("v", "C1_s", "q1", "q2"), ("v [px]", "C1_s [normalized]", "q1 [normalized C0 surface]", "q2 [normalized C0 surface]")):
        for group in sorted({row["height_group"] for row in dev}, key=lambda value: float(value[:-2])):
            group_rows = [row for row in dev if row["height_group"] == group]
            axis.scatter([row[feature] for row in group_rows], [row["height_residual_mm"] for row in group_rows], s=5, alpha=0.22, label=group)
        if held:
            axis.scatter([row[feature] for row in held], [row["height_residual_mm"] for row in held], s=18, facecolors="none", edgecolors="black", alpha=0.7, label="50mm held-out")
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_xlabel(label)
        axis.set_ylabel("height residual [mm]")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Surface-1A residual vs pixel/C1/surface coordinates")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_q_map(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["split_role"] in {"development_formal_repeat2_5", "heldout_formal_repeat2_5"} and row["height_measurement_inlier"] and row["jacobian_valid"]]
    sample = downsample(selected)
    dev = [row for row in sample if not row["held_out"]]
    held = [row for row in sample if row["held_out"]]
    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    if dev:
        scatter = axis.scatter([row["q1"] for row in dev], [row["q2"] for row in dev], c=[row["height_residual_mm"] for row in dev], cmap="coolwarm", vmin=-0.15, vmax=0.15, s=7, alpha=0.45, label="development formal")
        fig.colorbar(scatter, ax=axis, label="height residual [mm]")
    if held:
        axis.scatter([row["q1"] for row in held], [row["q2"] for row in held], s=28, marker="o", facecolors="none", edgecolors="black", label="50mm held-out")
    axis.set_xlabel("q1 = normalized C0 independent-axis 1")
    axis.set_ylabel("q2 = normalized C0 independent-axis 2")
    axis.set_title("Surface-1A residual map in unique C0 surface coordinates")
    axis.grid(alpha=0.2)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_jacobian(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["split_role"] in {"development_formal_repeat2_5", "heldout_formal_repeat2_5"} and row["height_measurement_inlier"] and row["jacobian_valid"]]
    sample = downsample(selected)
    dev = [row for row in sample if not row["held_out"]]
    held = [row for row in sample if row["held_out"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, feature, label in zip(axes, ("jacobian_Zg_norm_mm_per_px", "jacobian_lambda_norm_mm_per_px"), ("||dZg/d(u,v)|| [mm/px]", "||dλc1/d(u,v)|| [mm/px]")):
        if dev:
            axis.scatter([row[feature] for row in dev], [row["height_residual_mm"] for row in dev], s=6, alpha=0.25, label="development formal")
        if held:
            axis.scatter([row[feature] for row in held], [row["height_residual_mm"] for row in held], s=20, facecolors="none", edgecolors="black", label="50mm held-out")
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_xlabel(label)
        axis.set_ylabel("height residual [mm]")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Surface-1A residual vs local finite-difference Jacobian")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 5) -> str:
    if value is None or value == "":
        return "MISSING"
    return f"{float(value):.{digits}f}"


def condition_table(rows: list[dict[str, Any]], held_out: bool) -> list[dict[str, Any]]:
    selected = finite_records(rows, held_out=held_out, formal_only=True)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[(row["height_group"], row["position_id"])].append(row)
    output = []
    for (height_group, position_id), group in sorted(groups.items(), key=lambda item: (float(item[0][0][:-2]), item[0][1])):
        residual = np.asarray([row["height_residual_mm"] for row in group], dtype=np.float64)
        output.append({
            "height_group": height_group,
            "position_id": position_id,
            "point_count": len(group),
            "mean_residual_mm": float(np.mean(residual)),
            "median_residual_mm": float(np.median(residual)),
            "rmse_mm": float(np.sqrt(np.mean(residual**2))),
            "q1_mean": float(np.mean([row["q1"] for row in group])),
            "q2_mean": float(np.mean([row["q2"] for row in group])),
            "jacobian_Zg_norm_mean": float(np.mean([row["jacobian_Zg_norm_mm_per_px"] for row in group])),
        })
    return output


def write_report(
    path: Path,
    classifications: dict[str, str],
    diagnostics: dict[str, Any],
    metrics: list[dict[str, Any]],
    dev_conditions: list[dict[str, Any]],
    held_conditions: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    dev_rows = [row for row in metrics if row.get("scope") == "development_formal" and row.get("aggregation") == "point_pooled"]
    key_models = ["height", "v", "C1_s", "(q1,q2)", "J_Zg_norm", "J_lambda_norm", "Jacobian", "v+C1_s", "height+q1+q2", "q1+q2+Jacobian"]
    lines = [
        "# Surface-1A 激光曲面坐标 + 局部 Jacobian 残差归因",
        "",
        f"- `SURFACE_COORDINATE_EXPLANATORY_POWER = {classifications['SURFACE_COORDINATE_EXPLANATORY_POWER']}`",
        f"- `JACOBIAN_DEPENDENCE = {classifications['JACOBIAN_DEPENDENCE']}`",
        f"- `HEIGHT_ONLY_MODEL = {classifications['HEIGHT_ONLY_MODEL']}`",
        f"- `SURFACE_AWARE_CORRECTION_RECOMMENDED = {classifications['SURFACE_AWARE_CORRECTION_RECOMMENDED']}`",
        "",
        "本轮只做点级归因诊断；未重新拟合 C0/C1/G(S)/H1，也未生成或写入新的补偿函数。50 mm 全程保持 held-out 身份。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- 复用 Ground-4A manual-frozen 150 帧逐点 C1：`{provenance['dev_pointwise_path']}`，SHA-256 `{provenance['dev_pointwise_sha256']}`。",
        f"- 复用 Ground-4A repeat-1 `session_linear` proxy：`{provenance['dev_proxy_path']}`，不重新拟合永久参数。",
        f"- 复用 50 mm one-pass center cache、人工 geometry-only ROI 和既有 repeat-1 proxy；50 mm formal 为 20 帧、5 position。",
        f"- Frozen config `enable_laser_ray_correction=true`；config SHA-256 `{provenance['config_sha256']}`。",
        f"- 本轮新增：50 mm 点级 C0/C1 几何补全、统一 q1/q2、中心差分 Jacobian、height residual 字段、解释性统计和图表。",
        "",
        "## 统一 surface intrinsic coordinates",
        "",
        f"- Frozen C0 `dependent_axis={diagnostics['surface_definition']['dependent_axis']}`，`independent_axes={diagnostics['surface_definition']['independent_axes']}`。",
        "- 对每个 C0 交点 `P_c0`，定义 `q1=(P_c0[independent_axis_1]-center_1)/scale_1`、`q2=(P_c0[independent_axis_2]-center_2)/scale_2`；center/scale 直接来自 Frozen Quadratic C0。",
        "- q1/q2 是全数据共用的无量纲坐标；不按 height、position 或 held-out 数据重新 PCA、平移、缩放或定义坐标。CSV 同时保留 `q1_mm/q2_mm`。",
        f"- `height_residual = Zg - (a*S+b) - true_height`；a/b 使用同一 position 的 repeat-1 ground proxy，point-level metrics 仅使用与原 height line 相同的 XY robust inlier。",
        f"- Jacobian 为最终 C1 lambda/Zg 对原始像素 u/v 的中心差分，epsilon=`{JACOBIAN_EPS_PX:g}` px；导数单位分别为 mm/px。",
        "",
        "## Surface domain / held-out 检查",
        "",
        f"- development formal q bbox：q1 `{fmt(diagnostics['q_overlap']['development_q_bbox']['q1'][0])}`–`{fmt(diagnostics['q_overlap']['development_q_bbox']['q1'][1])}`，q2 `{fmt(diagnostics['q_overlap']['development_q_bbox']['q2'][0])}`–`{fmt(diagnostics['q_overlap']['development_q_bbox']['q2'][1])}`。",
        f"- 50 mm q bbox：q1 `{fmt(diagnostics['q_overlap']['heldout_q_bbox']['q1'][0])}`–`{fmt(diagnostics['q_overlap']['heldout_q_bbox']['q1'][1])}`，q2 `{fmt(diagnostics['q_overlap']['heldout_q_bbox']['q2'][0])}`–`{fmt(diagnostics['q_overlap']['heldout_q_bbox']['q2'][1])}`。",
        f"- 50 mm formal points 落在 development q bbox 内的比例：`{fmt(100*diagnostics['q_overlap']['heldout_inside_development_bbox_rate'], 1)}%`。50 mm 不参与 q/尺度/阈值或任何模型拟合；该比例仅作为 held-out 的事后 domain-coverage 诊断，并影响跨域支持级别的解释。",
        f"- 相近 q 条件对（固定 max(|Δq1|,|Δq2|)≤{Q_PAIR_TOLERANCE:g}）：`{diagnostics['q_similarity_pairs']['pair_count']}` 对；median residual difference `{fmt(diagnostics['q_similarity_pairs']['pair_median_abs_residual_difference_mm'])}` mm。",
        "",
        "## Development formal point-pooled explanatory metrics",
        "",
        "OLS 仅作为描述性解释量，不是补偿拟合；`within_condition_r2` 先去除每个完整 height×position condition 的均值，用于检查 surface/Jacobian 是否解释 condition 内空间结构。",
        "",
        "| model | features | R² | OLS RMSE | Pearson/Spearman | within-condition R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    metrics_by_model = {row["model"]: row for row in dev_rows}
    for model in key_models:
        row = metrics_by_model.get(model)
        if row is None:
            continue
        corr = f"{fmt(row.get('pearson_r'))}/{fmt(row.get('spearman_r'))}" if row.get("pearson_r") is not None else "—"
        lines.append(f"| {model} | {row.get('feature_list','')} | {fmt(row.get('descriptive_ols_r2'))} | {fmt(row.get('descriptive_ols_rmse_mm'))} | {corr} | {fmt(row.get('within_condition_r2'))} |")
    lines += [
        "",
        "## Height × position condition summaries",
        "",
        "development formal：",
        "",
        "| height | position | points | mean residual | median | RMSE | q1 mean | q2 mean | JZ norm mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dev_conditions:
        lines.append(f"| {row['height_group']} | {row['position_id']} | {row['point_count']} | {fmt(row['mean_residual_mm'])} | {fmt(row['median_residual_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['q1_mean'])} | {fmt(row['q2_mean'])} | {fmt(row['jacobian_Zg_norm_mean'])} |")
    lines += ["", "50 mm held-out formal：", "", "| height | position | points | mean residual | median | RMSE | q1 mean | q2 mean | JZ norm mean |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in held_conditions:
        lines.append(f"| {row['height_group']} | {row['position_id']} | {row['point_count']} | {fmt(row['mean_residual_mm'])} | {fmt(row['median_residual_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['q1_mean'])} | {fmt(row['q2_mean'])} | {fmt(row['jacobian_Zg_norm_mean'])} |")
    lines += [
        "",
        "## 结论解释",
        "",
        f"- `SURFACE_COORDINATE_EXPLANATORY_POWER = {classifications['SURFACE_COORDINATE_EXPLANATORY_POWER']}`：q1/q2 的统一定义和跨组比较已完成；该状态只表示诊断性解释能力，不代表可以直接把 q1/q2 变成 correction LUT。",
        f"- `JACOBIAN_DEPENDENCE = {classifications['JACOBIAN_DEPENDENCE']}`：使用最终 C1 的 `d_lambda/du,d_lambda/dv,d_Zg/du,d_Zg/dv` 及其 norm；50 mm 只做独立描述。",
        f"- `HEIGHT_ONLY_MODEL = {classifications['HEIGHT_ONLY_MODEL']}`：height-only 仅作为基线解释，不把高度相关性误认为 surface 充分性。",
        f"- `SURFACE_AWARE_CORRECTION_RECOMMENDED = {classifications['SURFACE_AWARE_CORRECTION_RECOMMENDED']}`：即使为 YES，也仅建议进入下一轮 held-out 诊断/候选设计；本轮不拟合、不冻结、不接生产链路。",
        "- 若 q 相近条件仍有明显 residual 差异，则 q1/q2 不是充分统计量；若 Jacobian 在相近 q 区域显著变化，则应优先考虑局部灵敏度/数值稳定性而非简单 height 或 v 补偿。",
        "",
        "## 输出",
        "",
        "- `surface1a_points.csv`：所有成功 C1 顶部点，含 u/v、C0/C1 lambda、P、Zg、q1/q2、height residual 与局部 Jacobian。",
        "- `surface_coordinate_definition.json`：唯一 q 定义、Frozen 参数、residual/Jacobian 口径与 provenance。",
        "- `surface1a_explanatory_metrics.csv`：point-pooled/condition-balanced 的描述性解释指标。",
        "- `residual_vs_surface_coordinate.png`、`surface_residual_map_q1_q2.png`、`residual_vs_jacobian.png`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground4a-dir", type=Path, default=DEFAULT_GROUND4A)
    parser.add_argument("--manual-frozen-dir", type=Path, default=DEFAULT_MANUAL_FROZEN)
    parser.add_argument("--height50-dir", type=Path, default=DEFAULT_HEIGHT50)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ground3-summary", type=Path, default=DEFAULT_GROUND3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ground4a_dir = args.ground4a_dir.resolve()
    manual_dir = args.manual_frozen_dir.resolve()
    height50_dir = args.height50_dir.resolve()
    config_path = args.config.resolve()
    ground3_path = args.ground3_summary.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    app = load_app_config(config_path)
    if app.reconstruction.enable_laser_ray_correction is not True:
        raise RuntimeError("Daheng config must have enable_laser_ray_correction=true")
    if app.calibration.laser_ray_correction is None:
        raise RuntimeError("Daheng config has no frozen C1 path")
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_plane,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        laser_ray_correction=app.calibration.laser_ray_correction,
    )
    params = app.reconstruction
    correction = calibration["laser_ray_correction"]
    origin, direction, s_domain_min, s_domain_max, ground3_summary = load_ground_reference(ground3_path)
    laser_model = calibration["laser_model"]
    if laser_model.get("model_type") != "quadratic_graph":
        raise RuntimeError("Surface-1A requires Frozen Quadratic C0")
    measurement_params = app.measurement if isinstance(app.measurement, MeasurementParams) else MeasurementParams()

    proxies = load_proxies(ground4a_dir / "ground4a_session_calibration.csv")
    dev_frames = load_dev_frames(manual_dir / "pointwise_diagnostics.csv", proxies)
    held_frames = load_height50_frames(height50_dir, height50_dir / "height50_manual_roi_registry.json")

    all_rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    consistency = {"max_abs_lambda_c0_mm": 0.0, "max_abs_lambda_c1_mm": 0.0, "max_abs_Zg_mm": 0.0}
    for frames in (dev_frames, held_frames):
        ordered = sorted(frames.values(), key=lambda frame: (bool(frame["held_out"]), DEV_ORDER.get(frame["dataset"], 99), frame["position_rank"], frame["repeat_index"]))
        for frame in ordered:
            rows, summary = build_point_rows(
                frame,
                calibration,
                params,
                correction,
                laser_model,
                origin,
                direction,
                measurement_params,
                consistency,
            )
            all_rows.extend(rows)
            frame_summaries.append({
                "dataset": frame["dataset"],
                "height_group": frame["height_group"],
                "position_id": frame["position_id"],
                "repeat_index": frame["repeat_index"],
                "split_role": frame["split_role"],
                "held_out": frame["held_out"],
                **summary,
            })

    if not all_rows:
        raise RuntimeError("Surface-1A produced no successful top points")
    point_fields = list(all_rows[0].keys())
    write_csv(output / "surface1a_points.csv", all_rows, point_fields)

    metrics = []
    metrics.extend(explanatory_metrics(all_rows, "development_formal", False))
    metrics.extend(explanatory_metrics(all_rows, "heldout_50_formal", True))
    metrics.extend(explanatory_metrics(all_rows, "combined_descriptive", None))
    metric_fields = list(metrics[0].keys())
    write_csv(output / "surface1a_explanatory_metrics.csv", metrics, metric_fields)

    overlap = q_overlap(all_rows)
    similarity = q_similarity_pairs(all_rows)
    classifications = classify(metrics, overlap)
    dev_conditions = condition_table(all_rows, False)
    held_conditions = condition_table(all_rows, True)
    q_domains = {"development_formal": q_domain(all_rows, False), "heldout_50_formal": q_domain(all_rows, True)}

    surface_definition = {
        "coordinate_name": "q1_q2",
        "coordinate_type": "unique_frozen_C0_quadratic_surface_intrinsic_coordinates",
        "model_type": laser_model["model_type"],
        "dependent_axis": laser_model["dependent_axis"],
        "independent_axes": list(laser_model["independent_axes"]),
        "independent_axis_indices_camera_xyz": [axis_index(value) for value in laser_model["independent_axes"]],
        "independent_center_mm": np.asarray(laser_model["normalization"]["independent_center_mm"], dtype=np.float64),
        "independent_scale_mm": np.asarray(laser_model["normalization"]["independent_scale_mm"], dtype=np.float64),
        "formula": "q_i=(P_c0[independent_axis_i]-independent_center_mm[i])/independent_scale_mm[i]",
        "P_c0_definition": "lambda_c0 times undistorted normalized camera ray; camera coordinates in mm",
        "S_definition": "S=(XY-Ground1_origin_xy) dot Ground1_direction_xy",
        "ground_origin_xy": origin,
        "ground_direction_xy": direction,
        "S_domain_mm": [s_domain_min, s_domain_max],
        "residual_definition": "height_residual=Zg-(a_position*S+b_position)-true_height; a/b are repeat-1 session_linear ground proxies",
        "Zg_definition": "final Frozen C1 reconstructed ground Z, including the current reconstruction path; no additional correction",
        "height_inlier_definition": "same XY robust height-line inlier protocol as existing C1 gauge replay",
        "jacobian_definition": "central finite difference of final C1 lambda and final ground Zg with respect to original pixel u/v",
        "jacobian_epsilon_px": JACOBIAN_EPS_PX,
        "q_pair_tolerance_normalized": Q_PAIR_TOLERANCE,
        "heldout_policy": "50mm is held-out validation only; not used to define q, scales, tolerance, or any fitted model; its q-domain overlap is used only for post-hoc cross-domain coverage classification",
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface_coordinate_definition.json", surface_definition)

    provenance = {
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "quadratic_c0_path": str(app.calibration.laser_plane),
        "quadratic_c0_sha256": sha256(app.calibration.laser_plane),
        "frozen_c1_path": str(app.calibration.laser_ray_correction),
        "frozen_c1_sha256": sha256(app.calibration.laser_ray_correction),
        "intrinsics_sha256": sha256(app.calibration.intrinsics),
        "extrinsics_sha256": sha256(app.calibration.extrinsics),
        "ground3_summary_path": str(ground3_path),
        "ground3_summary_sha256": sha256(ground3_path),
        "dev_pointwise_path": str(manual_dir / "pointwise_diagnostics.csv"),
        "dev_pointwise_sha256": sha256(manual_dir / "pointwise_diagnostics.csv"),
        "dev_proxy_path": str(ground4a_dir / "ground4a_session_calibration.csv"),
        "dev_proxy_sha256": sha256(ground4a_dir / "ground4a_session_calibration.csv"),
        "dev_roi_registry_path": str(manual_dir / "roi_registry.json"),
        "dev_roi_registry_sha256": sha256(manual_dir / "roi_registry.json"),
        "height50_frame_metrics_path": str(height50_dir / "height50_frame_metrics.csv"),
        "height50_frame_metrics_sha256": sha256(height50_dir / "height50_frame_metrics.csv"),
        "height50_roi_registry_path": str(height50_dir / "height50_manual_roi_registry.json"),
        "height50_roi_registry_sha256": sha256(height50_dir / "height50_manual_roi_registry.json"),
        "height50_input_audit_sha256": sha256(height50_dir / "height50_input_audit.csv"),
        "height50_center_cache_reused": True,
        "steger_rerun": False,
        "c0_c1_refit": False,
        "ground_gs_refit": False,
        "h1_refit": False,
        "production_change": False,
    }

    diagnostics = {
        "surface_definition": surface_definition,
        "q_domains": q_domains,
        "q_overlap": overlap,
        "q_similarity_pairs": similarity,
        "point_count": len(all_rows),
        "development_formal_point_count": sum(1 for row in all_rows if row["split_role"] == "development_formal_repeat2_5"),
        "heldout_50_formal_point_count": sum(1 for row in all_rows if row["split_role"] == "heldout_formal_repeat2_5"),
        "frame_count": len(frame_summaries),
        "frame_summaries": frame_summaries,
        "geometry_consistency": consistency,
        "classifications": classifications,
        "provenance": provenance,
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface1a_summary.json", diagnostics)
    plot_surface(output / "residual_vs_surface_coordinate.png", all_rows)
    plot_q_map(output / "surface_residual_map_q1_q2.png", all_rows)
    plot_jacobian(output / "residual_vs_jacobian.png", all_rows)
    write_report(output / "surface1a_report.md", classifications, diagnostics, metrics, dev_conditions, held_conditions, provenance)

    print(json.dumps({"output": str(output), **classifications, "point_count": len(all_rows), "dev_formal_points": diagnostics["development_formal_point_count"], "heldout_50_formal_points": diagnostics["heldout_50_formal_point_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
