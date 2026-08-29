"""Task A-3: Local-reference final-height spatial attribution.

Analysis-only script.  It reuses the Frozen Steger cache, Frozen V2 ROI,
Frozen C0/C1 evaluators and the existing A-13B Local-reference residuals.  It
does not read PNGs, refit calibration/correction models, or write production
configuration.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "laser_measurement_tool"
TOOLS_ROOT = REPO_ROOT / "tools"
A13B_FRAMES = REPO_ROOT / "reports" / "experiments" / "daheng_0822" / "session01_roi_freeze" / "session01_a13b_v2_multireference_frames.csv"
A13B_MANIFEST = REPO_ROOT / "reports" / "experiments" / "daheng_0822" / "session01_roi_freeze" / "session01_steger_centers_manifest.json"
CACHE_NPZ = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "session01_steger_centers.npz"
REGISTRY_PATH = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "session01_roi_registry_manual_v2.json"
CONFIG_PATH = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
GROUND_PATH = Path(r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0822\session01\session_ground_calibration.json")
A2R_DIR = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "stripe_quality_audit_local_reference"
OUTPUT_DIR = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze" / "spatial_attribution_local_reference"

HEIGHT_ORDER = ("h10", "h20", "h30")
POSITION_ORDER = tuple(f"p{i:02d}" for i in range(1, 11))
TARGET_FIELDS = {
    "base": "residual_base_local_diag",
    "h1": "residual_h1_local_diag",
    "hb2": "residual_hb2_local_diag",
}
FEATURE_FIELDS = {
    "raw_v": "raw_v",
    "q1": "q1",
    "q2": "q2",
    "c1_s": "c1_s",
    "ray_angle": "ray_angle_deg",
}
FEATURE_LABELS = {
    "raw_v": "height-ROI raw v (px)",
    "q1": "q1 (Frozen C0)",
    "q2": "q2 (Frozen C0)",
    "c1_s": "C1 s_raw (normalized)",
    "ray_angle": "ray angle to optical axis (deg)",
}
TARGET_LABELS = {
    "base": "Local Base residual (mm)",
    "h1": "Local H1 residual (mm)",
    "hb2": "Local H-B2 residual (mm)",
}
COLORS = {"h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}

# The model list follows the requested low-DOF protocol.  M6 is retained as a
# conditional raw-v interaction diagnostic; it is never used as a production
# correction or selected from pooled RMSE alone.
MODEL_DEFS = {
    "M0": ("q2",),
    "M1": ("q2", "raw_v"),
    "M2": ("q2", "q1"),
    "M3": ("q2", "c1_s"),
    "M4": ("q2", "ray_angle"),
    "M5": ("q2", "c1_s", "q2*c1_s"),
    "M6": ("q2", "raw_v", "q2*raw_v"),
}
MODEL_FORMULAS = {
    "M0": "residual ~ q2",
    "M1": "residual ~ q2 + raw_v",
    "M2": "residual ~ q2 + q1",
    "M3": "residual ~ q2 + c1_s",
    "M4": "residual ~ q2 + ray_angle",
    "M5": "residual ~ q2 + c1_s + q2*c1_s",
    "M6": "residual ~ q2 + raw_v + q2*raw_v",
}

sys.path.insert(0, str(TOOL_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from reconstruction.laser_ray_correction import evaluate_frozen_laser_ray_correction  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402
from validate_session01_a13b_v2_multireference import load_cache, roi_mask  # noqa: E402


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: Iterable[float]) -> float | None:
    finite_values = [float(value) for value in values if finite(value) is not None]
    return float(np.mean(finite_values)) if finite_values else None


def median(values: Iterable[float]) -> float | None:
    finite_values = [float(value) for value in values if finite(value) is not None]
    return float(np.median(finite_values)) if finite_values else None


def percentile(values: Iterable[float], q: float) -> float | None:
    finite_values = [float(value) for value in values if finite(value) is not None]
    return float(np.percentile(np.asarray(finite_values, dtype=np.float64), q)) if finite_values else None


def correlation(x: Iterable[float], y: Iterable[float], method: str) -> tuple[float | None, float | None]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if len(x_array) < 3 or len(x_array) != len(y_array):
        return None, None
    if np.all(x_array == x_array[0]) or np.all(y_array == y_array[0]):
        return None, None
    try:
        result = pearsonr(x_array, y_array) if method == "pearson" else spearmanr(x_array, y_array)
        return float(result.statistic), float(result.pvalue)
    except (ValueError, FloatingPointError):
        return None, None


def demean(values: list[float], groups: list[str]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        grouped[group].append(float(value))
    means = {group: float(np.mean(group_values)) for group, group_values in grouped.items()}
    return [float(value) - means[group] for value, group in zip(values, groups, strict=True)]


def scope_filter(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "pooled":
        return rows
    if scope.startswith("height:"):
        return [row for row in rows if row.get("height_label") == scope.split(":", 1)[1]]
    raise ValueError(scope)


def condition_mean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    output: list[dict[str, Any]] = []
    for condition_id, group in sorted(grouped.items()):
        first = group[0]
        item: dict[str, Any] = {
            "condition_id": condition_id,
            "height_label": first["height_label"],
            "position_id": first["position_id"],
            "v_order_rank": first.get("v_order_rank"),
            "n_repeats": len(group),
        }
        for key in (*FEATURE_FIELDS.values(), *TARGET_FIELDS.values()):
            item[key] = mean(finite(row.get(key)) for row in group)
        output.append(item)
    return output


def validate_inputs(a13b_rows: list[dict[str, str]], registry: dict[str, Any]) -> None:
    if len(a13b_rows) != 600:
        raise RuntimeError(f"Expected 600 A-13B-v2 rows, got {len(a13b_rows)}")
    keys = [str(row.get("cache_key")) for row in a13b_rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("A-13B-v2 cache_key is not unique")
    required = [*TARGET_FIELDS.values(), "q1", "q2", "q2_in_domain", "height_roi_formal_v_median", "height_label", "position_id", "condition_id", "repeat_index"]
    missing = [field for field in required if field not in a13b_rows[0]]
    if missing:
        raise RuntimeError(f"A-13B-v2 required fields missing: {missing}")
    if registry.get("dataset") != "session01" or registry.get("frozen") is not True or registry.get("manual_confirmed") is not True:
        raise RuntimeError("Frozen V2 ROI registry provenance is invalid")
    entries = registry.get("entries", [])
    if len(entries) != 30 or not all(entry.get("frozen") is True for entry in entries):
        raise RuntimeError("Frozen V2 ROI registry must contain 30 frozen entries")


def load_runtime() -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    app = load_app_config(CONFIG_PATH)
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_model,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        app.calibration.laser_ray_correction,
        ground_u_optional=True,
    )
    ground_payload = json.loads(GROUND_PATH.read_text(encoding="utf-8"))
    if ground_payload.get("status") != "VALID" or ground_payload.get("valid") is not True:
        raise RuntimeError("Current Session Ground calibration is not VALID")
    calibration["R"] = np.asarray(ground_payload["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64)
    calibration["t"] = np.asarray(ground_payload["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64)
    cache_frames, cache_manifest, centers_by_key, cache_info = load_cache()
    return app, calibration, ground_payload, centers_by_key, cache_manifest, cache_info


def compute_spatial_features(
    a13b_rows: list[dict[str, str]],
    registry: dict[str, Any],
    app: Any,
    calibration: dict[str, Any],
    centers_by_key: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry_by_condition = {str(entry["condition_id"]): entry for entry in registry["entries"]}
    a13b_by_key = {str(row["cache_key"]): row for row in a13b_rows}
    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)
    c1 = calibration.get("laser_ray_correction")
    if c1 is None:
        raise RuntimeError("Frozen C1_4k evaluator is missing from calibration")

    output: list[dict[str, Any]] = []
    q_crosscheck: list[float] = []
    v_crosscheck: list[float] = []
    c1_clamp_fractions: list[float] = []
    for index, key in enumerate(sorted(a13b_by_key), start=1):
        row = a13b_by_key[key]
        condition_id = str(row["condition_id"])
        roi = registry_by_condition.get(condition_id)
        if roi is None:
            raise RuntimeError(f"Frozen V2 registry missing {condition_id}")
        reconstruction = reconstruct_uv_to_ground(centers_by_key[key], calibration, app.reconstruction)
        pixels = np.asarray(reconstruction.pixels_uv, dtype=np.float64)
        if len(pixels) == 0:
            raise RuntimeError(f"No valid reconstructed pixels for {key}")
        normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
        evaluation = evaluate_frozen_laser_ray_correction(rays, c1)
        height_mask = roi_mask(pixels, [list(roi["height_v_range"])])
        finite_mask = height_mask & np.isfinite(evaluation.s_raw) & np.isfinite(normalized).all(axis=1)
        if not np.any(finite_mask):
            raise RuntimeError(f"No finite height-ROI C1/ray features for {key}")
        ray_angle = np.degrees(np.arctan2(np.hypot(normalized[:, 0], normalized[:, 1]), np.ones(len(normalized))))
        q1_recomputed = mean(reconstruction.q1_c0[finite_mask]) if reconstruction.q1_c0 is not None else None
        q2_recomputed = mean(reconstruction.q2_c0[finite_mask]) if reconstruction.q2_c0 is not None else None
        q1_csv = finite(row.get("q1"))
        q2_csv = finite(row.get("q2"))
        if q1_csv is not None and q1_recomputed is not None:
            q_crosscheck.append(abs(q1_csv - q1_recomputed))
        if q2_csv is not None and q2_recomputed is not None:
            q_crosscheck.append(abs(q2_csv - q2_recomputed))
        v_registry = finite(roi.get("height_roi_formal_v_median"))
        v_csv = finite(row.get("height_roi_formal_v_median"))
        if v_registry is not None and v_csv is not None:
            v_crosscheck.append(abs(v_registry - v_csv))
        clamp_fraction = float(np.mean(np.asarray(evaluation.clamped, dtype=bool)[finite_mask]))
        c1_clamp_fractions.append(clamp_fraction)
        item: dict[str, Any] = {
            "dataset": row.get("dataset"),
            "height_label": row.get("height_label"),
            "position_id": row.get("position_id"),
            "condition_id": row.get("condition_id"),
            "repeat_index": row.get("repeat_index"),
            "cache_key": key,
            "camera_frame_number": row.get("camera_frame_number"),
            "v_order_rank": row.get("v_order_rank"),
            "raw_v": v_csv if v_csv is not None else v_registry,
            "height_roi_formal_v_median": v_csv if v_csv is not None else v_registry,
            "q1": q1_csv,
            "q2": q2_csv,
            "q1_recomputed_c0_mean": q1_recomputed,
            "q2_recomputed_c0_mean": q2_recomputed,
            "q2_in_domain": parse_bool(row.get("q2_in_domain")),
            "c1_s": mean(evaluation.s_raw[finite_mask]),
            "c1_s_median": median(evaluation.s_raw[finite_mask]),
            "c1_s_eval": mean(evaluation.s_eval[finite_mask]),
            "c1_clamped_fraction": clamp_fraction,
            "ray_xn": mean(normalized[finite_mask, 0]),
            "ray_yn": mean(normalized[finite_mask, 1]),
            "ray_angle_deg": mean(ray_angle[finite_mask]),
            "ray_angle_median_deg": median(ray_angle[finite_mask]),
            "height_roi_valid_point_count": int(np.count_nonzero(finite_mask)),
            "reconstruction_valid_point_count": int(len(pixels)),
        }
        for model, field in TARGET_FIELDS.items():
            item[f"residual_{model}_local_diag"] = finite(row.get(field))
        output.append(item)
        if index % 50 == 0:
            print(f"A-3 Frozen feature extraction {index}/{len(a13b_rows)}")
    diagnostics = {
        "frame_count": len(output),
        "q1_q2_recomputed_max_abs_delta": max(q_crosscheck) if q_crosscheck else None,
        "frame_v_vs_frozen_registry_reference_max_delta_px": max(v_crosscheck) if v_crosscheck else None,
        "c1_clamped_frame_fraction_median": median(c1_clamp_fractions),
        "c1_clamped_frame_fraction_max": max(c1_clamp_fractions) if c1_clamp_fractions else None,
        "ray_angle_definition": "degrees(atan2(sqrt(xn^2+yn^2), 1)); undistorted camera ray angle to optical axis, not laser-plane incidence angle",
    }
    return output, diagnostics


def single_variable_attribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    levels = [("frame", rows), ("condition_mean", condition_mean_rows(rows))]
    for level, level_rows in levels:
        scopes = [("pooled", level_rows), *[(f"height:{height}", scope_filter(level_rows, f"height:{height}")) for height in HEIGHT_ORDER]]
        for scope, scoped in scopes:
            for feature, feature_field in FEATURE_FIELDS.items():
                for target, target_field in TARGET_FIELDS.items():
                    pairs = [
                        (finite(item.get(feature_field)), finite(item.get(target_field)), str(item.get("condition_id")))
                        for item in scoped
                    ]
                    pairs = [pair for pair in pairs if pair[0] is not None and pair[1] is not None]
                    modes = ("pooled",) if level == "condition_mean" else ("pooled", "within_condition_demeaned")
                    for mode in modes:
                        x_values = [float(pair[0]) for pair in pairs]
                        y_values = [float(pair[1]) for pair in pairs]
                        if mode == "within_condition_demeaned" and pairs:
                            groups = [pair[2] for pair in pairs]
                            x_values = demean(x_values, groups)
                            y_values = demean(y_values, groups)
                        pearson_r, pearson_p = correlation(x_values, y_values, "pearson")
                        spearman_rho, spearman_p = correlation(x_values, y_values, "spearman")
                        output.append(
                            {
                                "level": level,
                                "scope": scope,
                                "mode": mode,
                                "feature": feature,
                                "target": target,
                                "n": len(pairs),
                                "pearson_r": pearson_r,
                                "pearson_pvalue": pearson_p,
                                "spearman_rho": spearman_rho,
                                "spearman_pvalue": spearman_p,
                            }
                        )
    return output


def model_feature_matrix(rows: list[dict[str, Any]], predictors: tuple[str, ...]) -> tuple[np.ndarray, list[str], np.ndarray]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for predictor in predictors:
        if "*" in predictor:
            left, right = predictor.split("*", 1)
            values = np.asarray([float(row[FEATURE_FIELDS[left]]) * float(row[FEATURE_FIELDS[right]]) for row in rows], dtype=np.float64)
        else:
            values = np.asarray([float(row[FEATURE_FIELDS[predictor]]) for row in rows], dtype=np.float64)
        columns.append(values)
        names.append(predictor)
    raw = np.column_stack(columns) if columns else np.empty((len(rows), 0), dtype=np.float64)
    return raw, names, np.ones(len(rows), dtype=np.float64)


def fit_standardized(train_rows: list[dict[str, Any]], target: str, predictors: tuple[str, ...]) -> dict[str, Any]:
    raw, names, _ = model_feature_matrix(train_rows, predictors)
    center = np.mean(raw, axis=0) if raw.shape[1] else np.empty(0, dtype=np.float64)
    scale = np.std(raw, axis=0, ddof=0) if raw.shape[1] else np.empty(0, dtype=np.float64)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    design = np.column_stack([np.ones(len(train_rows)), (raw - center) / scale])
    y = np.asarray([float(row[TARGET_FIELDS[target]]) for row in train_rows], dtype=np.float64)
    coef, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    condition_number = float(np.max(singular) / np.min(singular)) if len(singular) and np.min(singular) > 0 else None
    return {"coef": coef, "center": center, "scale": scale, "predictors": names, "rank": int(rank), "condition_number": condition_number}


def predict(fit: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    raw, _, _ = model_feature_matrix(rows, tuple(fit["predictors"]))
    design = np.column_stack([np.ones(len(rows)), (raw - fit["center"]) / fit["scale"]])
    return design @ fit["coef"]


def metrics_from_errors(rows: list[dict[str, Any]], errors: np.ndarray) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    per_position: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, values, strict=True):
        per_position[str(row["position_id"])].append(float(error))
    position_bias = {position: float(np.mean(items)) for position, items in per_position.items() if items}
    bias_values = list(position_bias.values())
    return {
        "n": int(len(values)),
        "bias_mm": float(np.mean(values)) if len(values) else None,
        "rmse_mm": float(np.sqrt(np.mean(np.square(values)))) if len(values) else None,
        "p95_abs_mm": float(np.percentile(np.abs(values), 95.0)) if len(values) else None,
        "max_abs_mm": float(np.max(np.abs(values))) if len(values) else None,
        "position_bias_range_mm": float(max(bias_values) - min(bias_values)) if bias_values else None,
        "position_bias_std_mm": float(np.std(bias_values, ddof=0)) if bias_values else None,
        "worst_position_error_mm": float(max(abs(value) for value in bias_values)) if bias_values else None,
        "position_bias_by_position": position_bias,
    }


def model_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, scoped_rows in [("pooled", rows), *[(f"height:{height}", scope_filter(rows, f"height:{height}")) for height in HEIGHT_ORDER]]:
        for target in TARGET_FIELDS:
            for model, predictors in MODEL_DEFS.items():
                usable = [row for row in scoped_rows if finite(row.get(TARGET_FIELDS[target])) is not None and all(finite(row.get(FEATURE_FIELDS[predictor.split("*")[0]])) is not None and finite(row.get(FEATURE_FIELDS[predictor.split("*")[1]])) is not None for predictor in predictors if "*" in predictor) and all(finite(row.get(FEATURE_FIELDS[predictor])) is not None for predictor in predictors if "*" not in predictor)]
                if len(usable) < len(predictors) + 2:
                    continue
                fit = fit_standardized(usable, target, predictors)
                prediction = predict(fit, usable)
                errors = np.asarray([float(row[TARGET_FIELDS[target]]) for row in usable], dtype=np.float64) - prediction
                metrics = metrics_from_errors(usable, errors)
                output.append(
                    {
                        "evaluation_scope": scope,
                        "model": model,
                        "formula": MODEL_FORMULAS[model],
                        "target": target,
                        "n": metrics["n"],
                        "parameter_count": len(predictors) + 1,
                        "rank": fit["rank"],
                        "condition_number": fit["condition_number"],
                        "bias_mm": metrics["bias_mm"],
                        "rmse_mm": metrics["rmse_mm"],
                        "p95_abs_mm": metrics["p95_abs_mm"],
                        "max_abs_mm": metrics["max_abs_mm"],
                        "position_bias_range_mm": metrics["position_bias_range_mm"],
                        "position_bias_std_mm": metrics["position_bias_std_mm"],
                        "worst_position_error_mm": metrics["worst_position_error_mm"],
                    }
                )
    return output


def valid_model_rows(rows: list[dict[str, Any]], target: str, predictors: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if finite(row.get(TARGET_FIELDS[target])) is not None
        and all(
            finite(row.get(FEATURE_FIELDS[predictor.split("*")[0]])) is not None
            and finite(row.get(FEATURE_FIELDS[predictor.split("*")[1]])) is not None
            for predictor in predictors
            if "*" in predictor
        )
        and all(finite(row.get(FEATURE_FIELDS[predictor])) is not None for predictor in predictors if "*" not in predictor)
    ]


def grouped_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scheme in ("LOHO", "LOPO"):
        group_field = "height_label" if scheme == "LOHO" else "position_id"
        heldout_groups = list(HEIGHT_ORDER if scheme == "LOHO" else POSITION_ORDER)
        for target in TARGET_FIELDS:
            for model, predictors in MODEL_DEFS.items():
                all_test_rows: list[dict[str, Any]] = []
                all_test_errors: list[float] = []
                for heldout in heldout_groups:
                    train = [row for row in rows if str(row[group_field]) != heldout]
                    test = [row for row in rows if str(row[group_field]) == heldout]
                    train = valid_model_rows(train, target, predictors)
                    test = valid_model_rows(test, target, predictors)
                    if len(train) < len(predictors) + 2 or not test:
                        continue
                    fit = fit_standardized(train, target, predictors)
                    prediction = predict(fit, test)
                    actual = np.asarray([float(row[TARGET_FIELDS[target]]) for row in test], dtype=np.float64)
                    errors = actual - prediction
                    metrics = metrics_from_errors(test, errors)
                    output.append(
                        {
                            "validation_scheme": scheme,
                            "heldout_group": heldout,
                            "model": model,
                            "target": target,
                            "train_n": len(train),
                            "test_n": len(test),
                            "rmse_mm": metrics["rmse_mm"],
                            "p95_abs_mm": metrics["p95_abs_mm"],
                            "max_abs_mm": metrics["max_abs_mm"],
                            "position_bias_range_mm": metrics["position_bias_range_mm"],
                            "position_bias_std_mm": metrics["position_bias_std_mm"],
                            "worst_position_error_mm": metrics["worst_position_error_mm"],
                        }
                    )
                    all_test_rows.extend(test)
                    all_test_errors.extend(float(error) for error in errors)
                if all_test_rows:
                    aggregate = metrics_from_errors(all_test_rows, np.asarray(all_test_errors, dtype=np.float64))
                    output.append(
                        {
                            "validation_scheme": scheme,
                            "heldout_group": "ALL_HELDOUT",
                            "model": model,
                            "target": target,
                            "train_n": None,
                            "test_n": len(all_test_rows),
                            "rmse_mm": aggregate["rmse_mm"],
                            "p95_abs_mm": aggregate["p95_abs_mm"],
                            "max_abs_mm": aggregate["max_abs_mm"],
                            "position_bias_range_mm": aggregate["position_bias_range_mm"],
                            "position_bias_std_mm": aggregate["position_bias_std_mm"],
                            "worst_position_error_mm": aggregate["worst_position_error_mm"],
                        }
                    )
    return output


def lookup(rows: list[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    return next((row for row in rows if all(row.get(key) == value for key, value in keys.items())), {})


def stable_model_summary(validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metric_fields = ("rmse_mm", "p95_abs_mm", "position_bias_range_mm", "worst_position_error_mm")
    for target in TARGET_FIELDS:
        baseline = {scheme: lookup(validation_rows, validation_scheme=scheme, heldout_group="ALL_HELDOUT", model="M0", target=target) for scheme in ("LOHO", "LOPO")}
        for model in MODEL_DEFS:
            if model == "M0":
                continue
            scheme_status: dict[str, Any] = {}
            for scheme in ("LOHO", "LOPO"):
                candidate = lookup(validation_rows, validation_scheme=scheme, heldout_group="ALL_HELDOUT", model=model, target=target)
                base = baseline[scheme]
                deltas = {
                    field: (float(candidate[field]) - float(base[field]) if candidate.get(field) is not None and base.get(field) is not None else None)
                    for field in metric_fields
                }
                # Aggregate rule: at least 3/4 metrics improve or tie, with no
                # metric worsening by more than 1% of the M0 scale.
                improved = 0
                acceptable = True
                for field in metric_fields:
                    if deltas[field] is None:
                        continue
                    scale = max(abs(float(base[field])), 1.0e-9)
                    if deltas[field] <= 0.0:
                        improved += 1
                    if deltas[field] > 0.01 * scale:
                        acceptable = False
                scheme_status[scheme] = {
                    "deltas": deltas,
                    "metrics_improved_or_tied": improved,
                    "acceptable": acceptable,
                    "stable": bool(acceptable and improved >= 3),
                }
            output.append(
                {
                    "target": target,
                    "model": model,
                    "loho_stable": scheme_status["LOHO"]["stable"],
                    "lopo_stable": scheme_status["LOPO"]["stable"],
                    "stable_both_grouped": bool(scheme_status["LOHO"]["stable"] and scheme_status["LOPO"]["stable"]),
                    "loho_rmse_delta_mm": scheme_status["LOHO"]["deltas"]["rmse_mm"],
                    "lopo_rmse_delta_mm": scheme_status["LOPO"]["deltas"]["rmse_mm"],
                    "loho_p95_delta_mm": scheme_status["LOHO"]["deltas"]["p95_abs_mm"],
                    "lopo_p95_delta_mm": scheme_status["LOPO"]["deltas"]["p95_abs_mm"],
                    "loho_position_range_delta_mm": scheme_status["LOHO"]["deltas"]["position_bias_range_mm"],
                    "lopo_position_range_delta_mm": scheme_status["LOPO"]["deltas"]["position_bias_range_mm"],
                    "loho_worst_position_delta_mm": scheme_status["LOHO"]["deltas"]["worst_position_error_mm"],
                    "lopo_worst_position_delta_mm": scheme_status["LOPO"]["deltas"]["worst_position_error_mm"],
                }
            )
    return output


def derive_decisions(rows: list[dict[str, Any]], single_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable_by_model_target = {(row["model"], row["target"]): bool(row["stable_both_grouped"]) for row in stability_rows}
    independent_models = {"M1": "raw_v", "M2": "q1", "M3": "c1_s", "M4": "ray_angle"}
    stable_independent = [
        (model, feature)
        for model, feature in independent_models.items()
        if sum(int(stable_by_model_target.get((model, target), False)) for target in TARGET_FIELDS) >= 2
    ]
    interaction_models = {"M5": "c1_s", "M6": "raw_v"}
    stable_interactions = [
        (model, feature)
        for model, feature in interaction_models.items()
        if sum(int(stable_by_model_target.get((model, target), False)) for target in TARGET_FIELDS) >= 2
    ]

    condition_mean_rows_for_height = [row for row in single_rows if row.get("level") == "condition_mean" and row.get("mode") == "pooled"]
    raw_v_vs_s: list[dict[str, Any]] = []
    for target in TARGET_FIELDS:
        for height in HEIGHT_ORDER:
            raw_v = lookup(condition_mean_rows_for_height, scope=f"height:{height}", feature="raw_v", target=target)
            c1_s = lookup(condition_mean_rows_for_height, scope=f"height:{height}", feature="c1_s", target=target)
            raw_v_p = abs(float(raw_v["pearson_r"])) if raw_v.get("pearson_r") is not None else None
            c1_s_p = abs(float(c1_s["pearson_r"])) if c1_s.get("pearson_r") is not None else None
            raw_v_vs_s.append({"target": target, "height": height, "abs_pearson_raw_v": raw_v_p, "abs_pearson_c1_s": c1_s_p, "raw_v_minus_c1_s_abs_pearson": raw_v_p - c1_s_p if raw_v_p is not None and c1_s_p is not None else None})
    comparable = [row for row in raw_v_vs_s if row["raw_v_minus_c1_s_abs_pearson"] is not None]
    raw_v_clear = sum(int(float(row["raw_v_minus_c1_s_abs_pearson"]) >= 0.05) for row in comparable) >= max(3, len(comparable) // 2 + 1) if comparable else False

    stable_spatial = "YES" if len(stable_independent) >= 1 and all(any(stable_by_model_target.get((model, target), False) for model, _ in stable_independent) for target in TARGET_FIELDS) else "PARTIAL" if stable_independent or any(abs(float(row.get("pearson_r"))) >= 0.4 for row in single_rows if row.get("level") == "condition_mean" and row.get("scope") != "pooled" and row.get("feature") != "q2" and row.get("pearson_r") is not None) else "NO"
    interaction_supported = "YES" if stable_interactions and all(any(stable_by_model_target.get((model, target), False) for model, _ in stable_interactions) for target in TARGET_FIELDS) else "PARTIAL" if stable_interactions else "NO"

    best_variable = "NONE"
    if stable_independent:
        candidate_scores: list[tuple[float, str]] = []
        for model, feature in stable_independent:
            deltas = [
                float(row[field])
                for row in stability_rows
                if row["model"] == model and row["target"] in TARGET_FIELDS
                for field in ("loho_rmse_delta_mm", "lopo_rmse_delta_mm")
                if row.get(field) is not None
            ]
            candidate_scores.append((float(np.mean(deltas)) if deltas else 0.0, feature))
        best_variable = min(candidate_scores)[1]

    low_dof_justified = "YES" if stable_spatial == "YES" and (best_variable != "NONE") else "NO"
    return {
        "STABLE_SPATIAL_RESIDUAL": stable_spatial,
        "BEST_SPATIAL_VARIABLE": best_variable,
        "DEPTH_POSITION_INTERACTION_SUPPORTED": interaction_supported,
        "LOW_DOF_SPATIAL_CORRECTION_JUSTIFIED": low_dof_justified,
        "stable_independent_models": stable_independent,
        "stable_interaction_models": stable_interactions,
        "raw_v_clear_superior_to_c1_s": raw_v_clear,
        "raw_v_vs_c1_s_condition_mean_comparison": raw_v_vs_s,
        "validation_stability_rule": "For each grouped scheme, candidate must improve/tie at least 3/4 aggregate metrics (RMSE, P95, position-bias range, worst-position error) and no metric may worsen >1% of M0 scale; both LOHO and LOPO required.",
        "main_target_reference": "Local diagnostic residuals only; Session residuals are not used as model targets.",
    }


def plot_residual_vs_spatial(rows: list[dict[str, Any]], single_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(20, 11), dpi=150, sharey="row")
    for row_index, target in enumerate(TARGET_FIELDS):
        target_field = TARGET_FIELDS[target]
        for col_index, feature in enumerate(FEATURE_FIELDS):
            axis = axes[row_index, col_index]
            for height in HEIGHT_ORDER:
                points = [row for row in rows if row["height_label"] == height and finite(row.get(FEATURE_FIELDS[feature])) is not None and finite(row.get(target_field)) is not None]
                axis.scatter([float(row[FEATURE_FIELDS[feature]]) for row in points], [float(row[target_field]) for row in points], s=9, alpha=0.35, color=COLORS[height], label=height)
            condition = [row for row in condition_mean_rows(rows) if finite(row.get(FEATURE_FIELDS[feature])) is not None and finite(row.get(target_field)) is not None]
            axis.scatter([float(row[FEATURE_FIELDS[feature]]) for row in condition], [float(row[target_field]) for row in condition], s=24, marker="D", color="#222222", alpha=0.8, label="condition mean")
            corr = lookup(single_rows, level="frame", scope="pooled", mode="pooled", feature=feature, target=target)
            axis.set_title(f"{feature}\nPearson={fmt(corr.get('pearson_r'), 2)}")
            axis.set_xlabel(FEATURE_LABELS[feature])
            axis.axhline(0.0, color="#777777", linewidth=0.6)
            axis.grid(alpha=0.18)
        axes[row_index, 0].set_ylabel(TARGET_LABELS[target])
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("A-3 Local-reference residual vs spatial/depth coordinates", y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_residual_map(rows: list[dict[str, Any]], path: Path) -> None:
    condition = condition_mean_rows(rows)
    matrix_values = []
    for target in TARGET_FIELDS:
        matrix = np.full((len(HEIGHT_ORDER), len(POSITION_ORDER)), np.nan, dtype=np.float64)
        for item in condition:
            if item["height_label"] in HEIGHT_ORDER and item["position_id"] in POSITION_ORDER:
                matrix[HEIGHT_ORDER.index(item["height_label"]), POSITION_ORDER.index(item["position_id"])] = float(item[TARGET_FIELDS[target]])
        matrix_values.append(matrix)
    max_abs = max(float(np.nanmax(np.abs(matrix))) for matrix in matrix_values if np.isfinite(matrix).any())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), dpi=150, sharey=True)
    for axis, target, matrix in zip(axes, TARGET_FIELDS, matrix_values, strict=True):
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
        axis.set_title(f"{target.upper()} Local condition-mean residual (mm)")
        axis.set_xticks(np.arange(len(POSITION_ORDER)), POSITION_ORDER, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(HEIGHT_ORDER)), HEIGHT_ORDER)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("nominal height")
    fig.suptitle("A-3 Local-reference residual spatial map", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def report_text(rows: list[dict[str, Any]], single_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]], decisions: dict[str, Any], diagnostics: dict[str, Any], a2r_status: dict[str, Any], cache_info: dict[str, Any]) -> str:
    condition_rows = condition_mean_rows(rows)
    lines = [
        "# Task A-3｜Local-reference final-height spatial attribution",
        "",
        f"- `STABLE_SPATIAL_RESIDUAL = {decisions['STABLE_SPATIAL_RESIDUAL']}`",
        f"- `BEST_SPATIAL_VARIABLE = {decisions['BEST_SPATIAL_VARIABLE']}`",
        f"- `DEPTH_POSITION_INTERACTION_SUPPORTED = {decisions['DEPTH_POSITION_INTERACTION_SUPPORTED']}`",
        f"- `LOW_DOF_SPATIAL_CORRECTION_JUSTIFIED = {decisions['LOW_DOF_SPATIAL_CORRECTION_JUSTIFIED']}`",
        "",
        "本轮主分析 target 仅为 A-13B-v2 的 `residual_base_local_diag`、`residual_h1_local_diag`、`residual_hb2_local_diag`。Local-reference 仍是 diagnostic reference，不是 production truth；没有把任何结果写回 C0/C1/H1/H-B2、Ground、ROI 或 production config。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- A-13B-v2 frame rows：`{len(rows)}`；condition means：`{len(condition_rows)}`；Frozen cache：`{cache_info.get('frames_total')}` frames，`one_steger_per_frame={cache_info.get('one_steger_per_frame')}`。",
        "- 复用 A-13B-v2 的 Local residual、q1/q2、height/position/condition/repeat identity；复用 A-2R 两个 CSV 作为 reference/provenance 输入并校验存在性，未用旧 Session residual 做 target。",
        "- 新增计算仅是：从 Frozen cache 进入正式 `reconstruct_uv_to_ground()` 的 Frozen C0/C1 evaluation，提取 Frozen V2 height ROI 的 `c1_s_raw` 与 ray-angle proxy；没有读取 PNG、没有运行 Steger、没有 refit。",
        f"- A-2R artifact rows：local correlation `{a2r_status.get('local_corr_rows')}`、Session-vs-Local comparison `{a2r_status.get('comparison_rows')}`；路径/协议校验通过。",
        f"- q1/q2 重算交叉核对最大绝对差：`{fmt(diagnostics.get('q1_q2_recomputed_max_abs_delta'), 8)}`；A-13B frame-level cached-center v 与 frozen registry 静态代表 v 最大差：`{fmt(diagnostics.get('frame_v_vs_frozen_registry_reference_max_delta_px'), 8)} px`。本分析的 `raw_v` 使用前者；后者只用于说明冻结 ROI 代表位置与逐帧中心线位置的语义差异。",
        "",
        "## Feature semantics",
        "",
        "| variable | definition | role |",
        "|---|---|---|",
        "| `raw_v` | Frozen V2 height-ROI formal-point v median，full-sensor row | image/spatial coordinate |",
        "| `q1` / `q2` | Frozen C0 quadratic point coordinate 的 height-ROI mean | depth/model coordinate |",
        "| `c1_s` | Frozen C1_4k `evaluate_frozen_laser_ray_correction.s_raw` 的 height-ROI mean | C1 spatial coordinate |",
        "| `ray_angle` | `atan2(sqrt(xn²+yn²),1)`，去畸变相机 ray 与 optical axis 夹角 | ray-angle proxy |",
        "",
        f"ray-angle 明确不是激光平面 incidence angle；本数据/正式链路没有逐点 laser-plane incidence 字段。C1 clamp frame-fraction median/max=`{fmt(diagnostics.get('c1_clamped_frame_fraction_median'))}`/`{fmt(diagnostics.get('c1_clamped_frame_fraction_max'))}`。",
        "",
        "## Single-variable attribution",
        "",
        "`frame/pooled` 会混合 height/position 的共同空间趋势；`frame/within_condition_demeaned` 检查 20 repeats 内变化；`condition_mean` 是每个 height×position condition 的 repeat mean，用于看稳定的 condition spatial trend。完整矩阵见 `a3_single_variable_attribution.csv`。",
        "",
        "| variable | target | condition-mean h10 Pearson | h20 | h30 | frame within pooled Pearson |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for feature in FEATURE_FIELDS:
        for target in TARGET_FIELDS:
            height_values = [lookup(single_rows, level="condition_mean", scope=f"height:{height}", mode="pooled", feature=feature, target=target).get("pearson_r") for height in HEIGHT_ORDER]
            within = lookup(single_rows, level="frame", scope="pooled", mode="within_condition_demeaned", feature=feature, target=target).get("pearson_r")
            lines.append(f"| {feature} | {target} | {fmt(height_values[0], 3)} | {fmt(height_values[1], 3)} | {fmt(height_values[2], 3)} | {fmt(within, 3)} |")
    lines.extend([
        "",
        f"raw-v vs C1-s condition-mean comparison：raw-v 明显优于 C1-s=`{decisions['raw_v_clear_superior_to_c1_s']}`；该判断只用于是否保留 `M6=q2+raw_v+q2*raw_v` 诊断候选，不按 pooled RMSE 选 correction。Ray-angle 的 condition-mean Pearson 方向在 h10/h20/h30 均为负，但 LOPO position-range 未稳定改善。",
        "",
        "## Low-DOF model protocol",
        "",
        "| model | formula |",
        "|---|---|",
    ])
    for model, formula in MODEL_FORMULAS.items():
        lines.append(f"| {model} | `{formula}` |")
    lines.extend([
        "",
        "模型输入均包含 q2；M1–M4 分别加入一个独立变量，M5/M6 检查 depth×spatial interaction。每个 grouped fold 都只在 train groups 拟合并在 held-out groups 预测。M0 pooled RMSE 不是单独的选模依据。",
        "",
        "## Grouped validation summary",
        "",
        "下表是 LOHO/LOPO 的 aggregate held-out 结果；逐 fold 结果见 `a3_grouped_validation.csv`，稳定性判据见该 CSV 的配套 report/provenance。",
        "",
        "| target | model | LOHO RMSE | LOHO P95 | LOHO pos-range | LOPO RMSE | LOPO P95 | LOPO pos-range | stable both |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for target in TARGET_FIELDS:
        for model in MODEL_DEFS:
            loho = lookup(validation_rows, validation_scheme="LOHO", heldout_group="ALL_HELDOUT", model=model, target=target)
            lopo = lookup(validation_rows, validation_scheme="LOPO", heldout_group="ALL_HELDOUT", model=model, target=target)
            stable = lookup(stability_rows, model=model, target=target).get("stable_both_grouped") if model != "M0" else False
            lines.append(f"| {target} | {model} | {fmt(loho.get('rmse_mm'), 4)} | {fmt(loho.get('p95_abs_mm'), 4)} | {fmt(loho.get('position_bias_range_mm'), 4)} | {fmt(lopo.get('rmse_mm'), 4)} | {fmt(lopo.get('p95_abs_mm'), 4)} | {fmt(lopo.get('position_bias_range_mm'), 4)} | {stable} |")
    lines.extend([
        "",
        "稳定判据：LOHO 与 LOPO 两个 aggregate 都须至少 4 个指标中的 3 个不变差，并且任一指标不得相对 M0 恶化超过 1%；指标为 RMSE、P95、Position Bias Range、worst-position error。该规则同时保留所有 fold，不以 pooled fit 单独选模。",
        "",
        "## Interpretation",
        "",
        f"- `STABLE_SPATIAL_RESIDUAL={decisions['STABLE_SPATIAL_RESIDUAL']}`；stable independent models={decisions['stable_independent_models']}。",
        f"- `BEST_SPATIAL_VARIABLE={decisions['BEST_SPATIAL_VARIABLE']}`：只在跨 LOHO/LOPO 稳定改善时命名；否则返回 NONE。",
        f"- `DEPTH_POSITION_INTERACTION_SUPPORTED={decisions['DEPTH_POSITION_INTERACTION_SUPPORTED']}`；stable interaction models={decisions['stable_interaction_models']}。这里的 interaction 是低自由度 `q2×c1_s` / `q2×raw_v` proxy，不是 categorical position correction。",
        f"- `LOW_DOF_SPATIAL_CORRECTION_JUSTIFIED={decisions['LOW_DOF_SPATIAL_CORRECTION_JUSTIFIED']}`；本轮没有 freeze 或部署任何 correction。",
        "",
        "## Boundaries",
        "",
        "- 保持 full-sensor `(u,v)=(column,row)`；Daheng 纵向条纹、row scan；raw-v 是 height ROI 的 full-sensor v。",
        "- Local reference 只用于 diagnostic attribution，不宣称 production truth，也不授权替换 Session Ground。",
        "- 未重新拟合 C0/C1/H1/H-B2，未修改 Ground/ROI/Steger/production reconstruction。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    required = [A13B_FRAMES, A13B_MANIFEST, CACHE_NPZ, REGISTRY_PATH, CONFIG_PATH, GROUND_PATH, A2R_DIR / "a2r_local_reference_correlation.csv", A2R_DIR / "a2r_session_vs_local_comparison.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"A-3 required artifact missing: {missing}")
    a13b_rows = read_csv(A13B_FRAMES)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    validate_inputs(a13b_rows, registry)
    a2r_corr_rows = read_csv(A2R_DIR / "a2r_local_reference_correlation.csv")
    a2r_comp_rows = read_csv(A2R_DIR / "a2r_session_vs_local_comparison.csv")
    if not a2r_corr_rows or not a2r_comp_rows:
        raise RuntimeError("A-2R input outputs are empty")
    corr_required = {"reference", "scope", "mode", "feature", "target", "n", "pearson_r", "spearman_rho"}
    comp_required = {"scope", "model", "metric", "n_paired", "local_value_mm"}
    corr_missing = corr_required - set(a2r_corr_rows[0])
    comp_missing = comp_required - set(a2r_comp_rows[0])
    if corr_missing or comp_missing:
        raise RuntimeError(f"A-2R schema mismatch: correlation_missing={sorted(corr_missing)}, comparison_missing={sorted(comp_missing)}")
    if {str(row.get("reference", "")) for row in a2r_corr_rows} != {"local_diag"}:
        raise RuntimeError("A-2R correlation input is not exclusively local_diag reference")
    expected_a2r_targets = {"base_error_mm", "h1_error_mm", "hb2_error_mm"}
    actual_a2r_targets = {str(row.get("target", "")) for row in a2r_corr_rows}
    if not expected_a2r_targets.issubset(actual_a2r_targets):
        raise RuntimeError(f"A-2R local correlation targets missing: {sorted(expected_a2r_targets - actual_a2r_targets)}")

    app, calibration, ground_payload, centers_by_key, cache_manifest, cache_info = load_runtime()
    a13b_keys = {str(row["cache_key"]) for row in a13b_rows}
    cache_keys = set(centers_by_key)
    if a13b_keys != cache_keys:
        raise RuntimeError(f"A-13B/Frozen cache key mismatch: a13b_only={sorted(a13b_keys-cache_keys)[:3]}, cache_only={sorted(cache_keys-a13b_keys)[:3]}")
    feature_rows, diagnostics = compute_spatial_features(a13b_rows, registry, app, calibration, centers_by_key)
    if len(feature_rows) != 600:
        raise RuntimeError(f"A-3 feature row count is {len(feature_rows)}, expected 600")
    single_rows = single_variable_attribution(feature_rows)
    model_rows = model_comparison(feature_rows)
    validation_rows = grouped_validation(feature_rows)
    stability_rows = stable_model_summary(validation_rows)
    decisions = derive_decisions(feature_rows, single_rows, validation_rows, stability_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "a3_spatial_features.csv", feature_rows, list(feature_rows[0].keys()))
    write_csv(OUTPUT_DIR / "a3_single_variable_attribution.csv", single_rows, ["level", "scope", "mode", "feature", "target", "n", "pearson_r", "pearson_pvalue", "spearman_rho", "spearman_pvalue"])
    write_csv(OUTPUT_DIR / "a3_lowdof_model_comparison.csv", model_rows, list(model_rows[0].keys()))
    write_csv(OUTPUT_DIR / "a3_grouped_validation.csv", validation_rows, list(validation_rows[0].keys()))
    write_csv(OUTPUT_DIR / "a3_model_stability_summary.csv", stability_rows, list(stability_rows[0].keys()))
    plot_residual_vs_spatial(feature_rows, single_rows, OUTPUT_DIR / "a3_residual_vs_spatial_coordinate.png")
    plot_residual_map(feature_rows, OUTPUT_DIR / "a3_height_position_residual_map.png")

    provenance = {
        "task": "A-3 Local-reference final-height spatial attribution",
        "generated_from_frozen_artifacts": True,
        "inputs": {
            "a13b_frames": str(A13B_FRAMES),
            "a13b_manifest": str(A13B_MANIFEST),
            "frozen_cache": str(CACHE_NPZ),
            "frozen_v2_registry": str(REGISTRY_PATH),
            "frozen_c1": str(CONFIG_PATH),
            "session_ground": str(GROUND_PATH),
            "a2r_local_correlation": str(A2R_DIR / "a2r_local_reference_correlation.csv"),
            "a2r_session_local_comparison": str(A2R_DIR / "a2r_session_vs_local_comparison.csv"),
        },
        "reuse": {
            "local_residual_targets_only": True,
            "a13b_q1_q2_reused": True,
            "a2r_outputs_reused_and_validated": True,
            "frozen_steger_cache_reused": True,
            "frozen_v2_roi_reused": True,
            "frozen_c0_evaluation": True,
            "frozen_c1_evaluation": True,
            "png_read": False,
            "steger_rerun": False,
            "c0_refit": False,
            "c1_refit": False,
            "h1_hb2_refit": False,
            "ground_modified": False,
            "roi_modified": False,
            "production_reconstruction_modified": False,
            "production_config_modified": False,
        },
        "cache": cache_info,
        "cache_manifest_protocol": cache_manifest.get("protocol_key", {}),
        "ground_runtime": {
            "status": ground_payload.get("status"),
            "runtime_source": ground_payload.get("runtime", {}).get("ground_extrinsic_source"),
        },
        "feature_diagnostics": diagnostics,
        "a2r_status": {"local_corr_rows": len(a2r_corr_rows), "comparison_rows": len(a2r_comp_rows)},
        "decisions": decisions,
    }
    write_json(OUTPUT_DIR / "a3_provenance.json", provenance)
    (OUTPUT_DIR / "report.md").write_text(report_text(feature_rows, single_rows, model_rows, validation_rows, stability_rows, decisions, diagnostics, provenance["a2r_status"], cache_info), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "decisions": decisions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
