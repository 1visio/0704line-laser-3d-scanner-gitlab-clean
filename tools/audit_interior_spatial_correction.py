"""Task A-4: strict interior one-dimensional spatial correction feasibility.

This is a diagnostic-only audit.  It uses only A-3 Frozen features and the
A-3B strict-reference condition registry for p02-p09.  G(v) candidates are
fit independently inside each LOHO/LOPO fold; no result is written to the
production reconstruction configuration.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze"
A3_DIR = OUTPUT_ROOT / "spatial_attribution_local_reference"
A3B_DIR = OUTPUT_ROOT / "ray_surface_geometry_local_reference"
OUTPUT_DIR = OUTPUT_ROOT / "interior_spatial_correction_local_reference"
A3_FEATURES = A3_DIR / "a3_spatial_features.csv"
A3B_EDGE_AUDIT = A3B_DIR / "a3b_edge_reference_audit.csv"
A3B_REPORT = A3B_DIR / "report.md"

sys.path.insert(0, str(REPO_ROOT / "tools"))
from audit_local_spatial_attribution import finite, mean, read_csv, write_csv, write_json  # noqa: E402


HEIGHT_ORDER = ("h10", "h20", "h30")
POSITION_ORDER = tuple(f"p{i:02d}" for i in range(2, 10))
TARGET_FIELDS = {
    "h1": "residual_h1_local_diag",
    "hb2": "residual_hb2_local_diag",
}
TARGET_LABELS = {"h1": "Local H1 residual (mm)", "hb2": "Local H-B2 residual (mm)"}
STRICT_DEFINITION = "edge_baseline_clipped=False AND baseline_support_type=BOTH_SIDES AND local_baseline_support=BOTH_SIDES AND local_baseline_extrapolation=False"
MODEL_DEFS = {
    "PIECEWISE_LINEAR": "piecewise_linear_3_quantile_breakpoints",
    "SPLINE_3K": "cubic_bspline_3_interior_knots",
    "SPLINE_4K": "cubic_bspline_4_interior_knots",
}
MODEL_ORDER = tuple(MODEL_DEFS)
VALIDATION_MODES = ("FULL_WITH_EXTRAPOLATION_DIAGNOSTIC", "INTERPOLATION_ONLY")
METRICS = ("rmse_mm", "p95_abs_mm", "position_bias_range_mm", "worst_position_error_mm")
EPSILON = 1.0e-12
DEGREE = 3


def correlation(x: Iterable[float], y: Iterable[float], method: str) -> tuple[float | None, float | None]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if len(x_array) < 3 or len(x_array) != len(y_array):
        return None, None
    if np.all(x_array == x_array[0]) or np.all(y_array == y_array[0]):
        return None, None
    result = pearsonr(x_array, y_array) if method == "pearson" else spearmanr(x_array, y_array)
    return float(result.statistic), float(result.pvalue)


def abs_percentile(values: Iterable[float], q: float) -> float | None:
    items = [abs(float(value)) for value in values if finite(value) is not None]
    return float(np.percentile(np.asarray(items, dtype=np.float64), q)) if items else None


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("condition_id")), str(row.get("repeat_index"))


def validate_inputs(features: list[dict[str, str]], edge_audit: list[dict[str, str]]) -> None:
    if len(features) != 600:
        raise RuntimeError(f"Expected 600 A-3 feature rows, got {len(features)}")
    if len(edge_audit) != 30:
        raise RuntimeError(f"Expected 30 A-3B edge-audit rows, got {len(edge_audit)}")
    required_features = {"cache_key", "condition_id", "height_label", "position_id", "raw_v", *TARGET_FIELDS.values()}
    required_edge = {"condition_id", "height_label", "position_id", "strict_reference_available", "n_strict_frames"}
    if missing := required_features - set(features[0]):
        raise RuntimeError(f"A-3 fields missing: {sorted(missing)}")
    if missing := required_edge - set(edge_audit[0]):
        raise RuntimeError(f"A-3B edge audit fields missing: {sorted(missing)}")
    feature_keys = {str(row["cache_key"]) for row in features}
    if len(feature_keys) != 600:
        raise RuntimeError("A-3 cache_key is not unique")
    positions = {str(row["position_id"]) for row in features}
    if not set(POSITION_ORDER).issubset(positions):
        raise RuntimeError(f"A-3 does not contain all requested interior positions: {positions}")


def select_strict_interior(features: list[dict[str, str]], edge_audit: list[dict[str, str]]) -> list[dict[str, Any]]:
    audit_by_condition = {str(row["condition_id"]): row for row in edge_audit}
    selected: list[dict[str, Any]] = []
    for row in features:
        position = str(row["position_id"])
        condition = str(row["condition_id"])
        audit = audit_by_condition.get(condition)
        if position not in POSITION_ORDER or audit is None:
            continue
        if str(audit["strict_reference_available"]).strip().lower() not in {"true", "1", "yes"}:
            continue
        if int(float(audit["n_strict_frames"])) <= 0:
            continue
        item: dict[str, Any] = dict(row)
        item["raw_v"] = finite(row.get("raw_v"))
        item["height_label"] = str(row["height_label"])
        item["position_id"] = position
        item["condition_id"] = condition
        for target, field in TARGET_FIELDS.items():
            item[field] = finite(row.get(field))
        selected.append(item)
    if len(selected) != 480:
        raise RuntimeError(f"Strict p02-p09 selection expected 480 rows, got {len(selected)}")
    if {row["condition_id"] for row in selected} != {f"{height}_{position}" for height in HEIGHT_ORDER for position in POSITION_ORDER}:
        raise RuntimeError("Strict p02-p09 selection does not contain all 24 height-position conditions")
    return selected


def condition_profile_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["height_label"]), str(row["position_id"]))].append(row)
    height_means = {
        target: {
            height: float(np.mean([float(row[TARGET_FIELDS[target]]) for row in rows if row["height_label"] == height]))
            for height in HEIGHT_ORDER
        }
        for target in TARGET_FIELDS
    }
    output: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, dict[str, float]]] = {target: {} for target in TARGET_FIELDS}
    for height in HEIGHT_ORDER:
        for position in POSITION_ORDER:
            group = grouped[(height, position)]
            item: dict[str, Any] = {
                "height_label": height,
                "position_id": position,
                "condition_id": f"{height}_{position}",
                "n_frames": len(group),
                "v_mean_px": float(np.mean([float(row["raw_v"]) for row in group])),
                "v_min_px": float(np.min([float(row["raw_v"]) for row in group])),
                "v_max_px": float(np.max([float(row["raw_v"]) for row in group])),
            }
            for target, field in TARGET_FIELDS.items():
                residual_mean = float(np.mean([float(row[field]) for row in group]))
                spatial = residual_mean - height_means[target][height]
                item[f"{target}_residual_mean_mm"] = residual_mean
                item[f"{target}_height_mean_mm"] = height_means[target][height]
                item[f"{target}_spatial_residual_mm"] = spatial
                profiles[target].setdefault(height, {})[position] = spatial
            output.append(item)
    return output, profiles


def quantile_knots(values: np.ndarray, count: int) -> np.ndarray:
    low, high = float(np.min(values)), float(np.max(values))
    if high - low <= EPSILON or count <= 0:
        return np.empty(0, dtype=np.float64)
    knots = np.quantile(values, np.linspace(0.0, 1.0, count + 2)[1:-1])
    if len(np.unique(knots)) != count or np.min(np.diff(np.r_[low, knots, high])) <= EPSILON:
        knots = np.linspace(low, high, count + 2)[1:-1]
    return np.asarray(knots, dtype=np.float64)


def spline_design(values: np.ndarray, v_min: float, v_max: float, interior_knots: np.ndarray, extrapolate: bool) -> np.ndarray:
    knots = np.r_[np.repeat(v_min, DEGREE + 1), interior_knots, np.repeat(v_max, DEGREE + 1)]
    coefficient_count = len(knots) - DEGREE - 1
    basis = []
    for index in range(coefficient_count):
        coefficients = np.zeros(coefficient_count, dtype=np.float64)
        coefficients[index] = 1.0
        basis.append(BSpline(knots, coefficients, DEGREE, extrapolate=extrapolate)(values))
    return np.column_stack(basis)


def train_model(train_rows: list[dict[str, Any]], target: str, model_name: str, height_means: dict[str, float]) -> dict[str, Any]:
    target_field = TARGET_FIELDS[target]
    values = np.asarray([float(row["raw_v"]) for row in train_rows], dtype=np.float64)
    response = np.asarray([float(row[target_field]) - height_means[str(row["height_label"])] for row in train_rows], dtype=np.float64)
    v_min, v_max = float(np.min(values)), float(np.max(values))
    if model_name == "PIECEWISE_LINEAR":
        knots = quantile_knots(values, 3)
        scale = max(v_max - v_min, 1.0)
        normalized = (values - v_min) / scale
        normalized_knots = (knots - v_min) / scale
        design = np.column_stack([np.ones(len(values)), normalized, *[np.maximum(normalized - knot, 0.0) for knot in normalized_knots]])
        model = {"model": model_name, "v_min": v_min, "v_max": v_max, "knots": knots, "scale": scale, "normalized_knots": normalized_knots, "kind": "piecewise"}
    elif model_name in {"SPLINE_3K", "SPLINE_4K"}:
        count = 3 if model_name == "SPLINE_3K" else 4
        knots = quantile_knots(values, count)
        design = spline_design(values, v_min, v_max, knots, extrapolate=False)
        model = {"model": model_name, "v_min": v_min, "v_max": v_max, "knots": knots, "kind": "spline"}
    else:
        raise ValueError(model_name)
    coefficient, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    model["coefficient"] = coefficient
    model["rank"] = int(rank)
    model["parameter_count"] = int(len(coefficient))
    return model


def predict_model(model: dict[str, Any], values: np.ndarray, interpolation_only: bool) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    in_domain = (values >= float(model["v_min"]) - 1.0e-9) & (values <= float(model["v_max"]) + 1.0e-9)
    if model["kind"] == "piecewise":
        normalized = (values - float(model["v_min"])) / float(model["scale"])
        design = np.column_stack([np.ones(len(values)), normalized, *[np.maximum(normalized - knot, 0.0) for knot in model["normalized_knots"]]])
        prediction = design @ model["coefficient"]
    else:
        design = spline_design(values, float(model["v_min"]), float(model["v_max"]), np.asarray(model["knots"], dtype=np.float64), extrapolate=not interpolation_only)
        prediction = design @ model["coefficient"]
    valid = in_domain & np.isfinite(prediction)
    if interpolation_only:
        prediction = np.where(valid, prediction, np.nan)
    return prediction, in_domain


def metrics_from_errors(rows: list[dict[str, Any]], errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    by_position: dict[str, list[float]] = defaultdict(list)
    by_height_position: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row, error in zip(rows, errors, strict=True):
        position = str(row["position_id"])
        height = str(row["height_label"])
        by_position[position].append(float(error))
        by_height_position[(height, position)].append(float(error))
    position_bias = {position: float(np.mean(values)) for position, values in by_position.items()}
    height_ranges = []
    height_worsts = []
    for height in HEIGHT_ORDER:
        values = [float(np.mean(by_height_position[(height, position)])) for position in POSITION_ORDER if (height, position) in by_height_position]
        if values:
            height_ranges.append(max(values) - min(values))
            height_worsts.append(max(abs(value) for value in values))
    return {
        "n": int(len(errors)),
        "bias_mm": float(np.mean(errors)),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "p95_abs_mm": float(np.percentile(np.abs(errors), 95.0)),
        "max_abs_mm": float(np.max(np.abs(errors))),
        "position_bias_range_mm": float(max(position_bias.values()) - min(position_bias.values())) if position_bias else None,
        "mean_height_position_bias_range_mm": float(np.mean(height_ranges)) if height_ranges else None,
        "worst_position_error_mm": float(max(abs(value) for value in position_bias.values())) if position_bias else None,
        "mean_height_worst_position_error_mm": float(np.mean(height_worsts)) if height_worsts else None,
    }


def evaluate_before_after(rows: list[dict[str, Any]], target: str, prediction: np.ndarray) -> dict[str, Any]:
    target_field = TARGET_FIELDS[target]
    valid = np.isfinite(prediction)
    selected = [row for row, keep in zip(rows, valid, strict=True) if keep]
    if not selected:
        empty_metrics = {"n": 0, "bias_mm": None, **{metric: None for metric in METRICS}, "max_abs_mm": None, "mean_height_position_bias_range_mm": None, "mean_height_worst_position_error_mm": None}
        return {"prediction_n": 0, "before": empty_metrics, "after": dict(empty_metrics)}
    before = np.asarray([float(row[target_field]) for row in selected], dtype=np.float64)
    after = before - prediction[valid]
    before_metrics = metrics_from_errors(selected, before)
    after_metrics = metrics_from_errors(selected, after)
    return {"prediction_n": len(selected), "before": before_metrics, "after": after_metrics}


def run_validation(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    schemes = (("LOHO", "height_label", HEIGHT_ORDER), ("LOPO", "position_id", POSITION_ORDER))
    for scheme, group_field, heldout_groups in schemes:
        for target in TARGET_FIELDS:
            for model_name in MODEL_ORDER:
                fold_predictions: dict[str, list[dict[str, Any]]] = {mode: [] for mode in VALIDATION_MODES}
                for heldout in heldout_groups:
                    train = [row for row in rows if str(row[group_field]) != heldout]
                    test = [row for row in rows if str(row[group_field]) == heldout]
                    target_field = TARGET_FIELDS[target]
                    height_means = {
                        height: float(np.mean([float(row[target_field]) for row in train if str(row["height_label"]) == height]))
                        for height in HEIGHT_ORDER
                        if any(str(row["height_label"]) == height for row in train)
                    }
                    model = train_model(train, target, model_name, height_means)
                    test_v = np.asarray([float(row["raw_v"]) for row in test], dtype=np.float64)
                    predictions_full, in_domain = predict_model(model, test_v, interpolation_only=False)
                    interpolation_n = int(np.count_nonzero(in_domain & np.isfinite(predictions_full)))
                    for evaluation_mode in VALIDATION_MODES:
                        predictions_mode, _ = predict_model(model, test_v, interpolation_only=evaluation_mode == "INTERPOLATION_ONLY")
                        result = evaluate_before_after(test, target, predictions_mode)
                        extrapolated_n = int(np.count_nonzero(~in_domain))
                        row: dict[str, Any] = {
                            "validation_scheme": scheme,
                            "evaluation_mode": evaluation_mode,
                            "heldout_group": heldout,
                            "model": model_name,
                            "target": target,
                            "formula": MODEL_DEFS[model_name],
                            "train_n": len(train),
                            "test_n": len(test),
                            "prediction_n": result["prediction_n"],
                            "extrapolated_n": extrapolated_n,
                            "interpolation_coverage": float(interpolation_n / len(test)) if test else None,
                            "v_train_min_px": model["v_min"],
                            "v_train_max_px": model["v_max"],
                            "v_test_min_px": float(np.min(test_v)) if len(test_v) else None,
                            "v_test_max_px": float(np.max(test_v)) if len(test_v) else None,
                            "status": "FOLD" if result["prediction_n"] else "NO_INTERPOLATION_SUPPORT",
                        }
                        for prefix in ("before", "after"):
                            metrics = result[prefix]
                            for key, value in metrics.items():
                                row[f"{prefix}_{key}"] = value
                        validation.append(row)
                        if scheme == "LOPO" and evaluation_mode == "FULL_WITH_EXTRAPOLATION_DIAGNOSTIC":
                            for source_row, prediction, inside in zip(test, predictions_full, in_domain, strict=True):
                                predictions.append({
                                    "validation_scheme": scheme,
                                    "evaluation_mode": evaluation_mode,
                                    "heldout_group": heldout,
                                    "model": model_name,
                                    "target": target,
                                    "height_label": source_row["height_label"],
                                    "position_id": source_row["position_id"],
                                    "raw_v": source_row["raw_v"],
                                    "before": source_row[target_field],
                                    "prediction": prediction,
                                    "after": float(source_row[target_field]) - float(prediction),
                                    "in_domain": bool(inside),
                                })
                        fold_predictions[evaluation_mode].extend(
                            {"source": source_row, "prediction": float(prediction)}
                            for source_row, prediction in zip(test, predictions_mode, strict=True)
                            if np.isfinite(prediction)
                        )
                for evaluation_mode in VALIDATION_MODES:
                    all_pairs = fold_predictions[evaluation_mode]
                    aggregate_rows = [pair["source"] for pair in all_pairs]
                    predictions_array = np.asarray([pair["prediction"] for pair in all_pairs], dtype=np.float64)
                    result = evaluate_before_after(aggregate_rows, target, predictions_array) if all_pairs else None
                    aggregate: dict[str, Any] = {
                        "validation_scheme": scheme,
                        "evaluation_mode": evaluation_mode,
                        "heldout_group": "ALL_HELDOUT",
                        "model": model_name,
                        "target": target,
                        "formula": MODEL_DEFS[model_name],
                        "train_n": None,
                        "test_n": len(rows) if evaluation_mode == "FULL_WITH_EXTRAPOLATION_DIAGNOSTIC" else len(all_pairs),
                        "prediction_n": len(all_pairs),
                        "extrapolated_n": 0,
                        "interpolation_coverage": None,
                        "v_train_min_px": None,
                        "v_train_max_px": None,
                        "v_test_min_px": None,
                        "v_test_max_px": None,
                        "status": "AGGREGATE" if result else "NO_PREDICTIONS",
                    }
                    aggregate_interpolation_n = (
                        len(all_pairs)
                        if evaluation_mode == "INTERPOLATION_ONLY"
                        else sum(
                            int(row["prediction_n"])
                            for row in validation
                            if row["validation_scheme"] == scheme
                            and row["evaluation_mode"] == "INTERPOLATION_ONLY"
                            and row["heldout_group"] != "ALL_HELDOUT"
                            and row["model"] == model_name
                            and row["target"] == target
                        )
                    )
                    # Recompute extrapolation count from fold rows rather than
                    # carrying target data through the aggregate prediction list.
                    aggregate["extrapolated_n"] = int(sum(int(row["extrapolated_n"]) for row in validation if row["validation_scheme"] == scheme and row["evaluation_mode"] == evaluation_mode and row["heldout_group"] != "ALL_HELDOUT" and row["model"] == model_name and row["target"] == target))
                    aggregate["interpolation_coverage"] = float(aggregate_interpolation_n / len(rows)) if rows else None
                    if result:
                        for prefix in ("before", "after"):
                            for key, value in result[prefix].items():
                                aggregate[f"{prefix}_{key}"] = value
                    else:
                        for prefix in ("before", "after"):
                            for key in ("n", "bias_mm", *METRICS, "max_abs_mm", "mean_height_position_bias_range_mm", "mean_height_worst_position_error_mm"):
                                aggregate[f"{prefix}_{key}"] = None
                    validation.append(aggregate)
    return validation, predictions


def pooled_model_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target, target_field in TARGET_FIELDS.items():
        height_means = {height: float(np.mean([float(row[target_field]) for row in rows if row["height_label"] == height])) for height in HEIGHT_ORDER}
        for model_name in MODEL_ORDER:
            model = train_model(rows, target, model_name, height_means)
            values = np.asarray([float(row["raw_v"]) for row in rows], dtype=np.float64)
            prediction, _ = predict_model(model, values, interpolation_only=True)
            result = evaluate_before_after(rows, target, prediction)
            spatial_actual = np.asarray([float(row[target_field]) - height_means[str(row["height_label"])] for row in rows], dtype=np.float64)
            spatial_prediction = prediction
            valid = np.isfinite(spatial_prediction)
            output.append({
                "evaluation_scope": "POOLED_IN_SAMPLE",
                "evaluation_mode": "INTERPOLATION_ONLY",
                "model": model_name,
                "target": target,
                "formula": MODEL_DEFS[model_name],
                "n": result["prediction_n"],
                "parameter_count": model["parameter_count"],
                "v_min_px": model["v_min"],
                "v_max_px": model["v_max"],
                "spatial_fit_rmse_mm": float(np.sqrt(np.mean(np.square(spatial_actual[valid] - spatial_prediction[valid])))),
                "spatial_fit_p95_abs_mm": float(np.percentile(np.abs(spatial_actual[valid] - spatial_prediction[valid]), 95.0)),
                "before_rmse_mm": result["before"]["rmse_mm"],
                "after_rmse_mm": result["after"]["rmse_mm"],
                "before_p95_abs_mm": result["before"]["p95_abs_mm"],
                "after_p95_abs_mm": result["after"]["p95_abs_mm"],
                "before_position_bias_range_mm": result["before"]["position_bias_range_mm"],
                "after_position_bias_range_mm": result["after"]["position_bias_range_mm"],
                "before_mean_height_position_bias_range_mm": result["before"]["mean_height_position_bias_range_mm"],
                "after_mean_height_position_bias_range_mm": result["after"]["mean_height_position_bias_range_mm"],
                "before_worst_position_error_mm": result["before"]["worst_position_error_mm"],
                "after_worst_position_error_mm": result["after"]["worst_position_error_mm"],
                "extrapolated_n": 0,
                "interpolation_coverage": 1.0,
            })
    return output


def validation_model_rows(validation: list[dict[str, Any]], scheme: str, mode: str) -> list[dict[str, Any]]:
    return [row for row in validation if row["validation_scheme"] == scheme and row["evaluation_mode"] == mode and row["heldout_group"] == "ALL_HELDOUT"]


def candidate_metric_pass(row: dict[str, Any]) -> bool:
    if row.get("prediction_n") in (None, 0):
        return False
    improved = 0
    acceptable = True
    for metric in METRICS:
        before = finite(row.get(f"before_{metric}"))
        after = finite(row.get(f"after_{metric}"))
        if before is None or after is None:
            return False
        if after <= before:
            improved += 1
        if after > before + 0.01 * max(abs(before), 1.0e-9):
            acceptable = False
    return bool(acceptable and improved >= 3)


def derive_decisions(profile: dict[str, dict[str, dict[str, float]]], validation: list[dict[str, Any]], pooled: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile_correlations: list[dict[str, Any]] = []
    for target in TARGET_FIELDS:
        for left, right in (("h10", "h20"), ("h10", "h30"), ("h20", "h30")):
            left_values = [profile[target][left][position] for position in POSITION_ORDER]
            right_values = [profile[target][right][position] for position in POSITION_ORDER]
            pearson_r, pearson_p = correlation(left_values, right_values, "pearson")
            spearman_rho, spearman_p = correlation(left_values, right_values, "spearman")
            profile_correlations.append({"target": target, "height_pair": f"{left}_vs_{right}", "n_positions": len(POSITION_ORDER), "pearson_r": pearson_r, "pearson_pvalue": pearson_p, "spearman_rho": spearman_rho, "spearman_pvalue": spearman_p})
    repeatable = True
    for target in TARGET_FIELDS:
        values = [row["pearson_r"] for row in profile_correlations if row["target"] == target]
        repeatable = repeatable and all(value is not None and value >= 0.50 for value in values)

    candidate_status: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        loho_full = {target: next((row for row in validation_model_rows(validation, "LOHO", "FULL_WITH_EXTRAPOLATION_DIAGNOSTIC") if row["model"] == model_name and row["target"] == target), {}) for target in TARGET_FIELDS}
        lopo_full = {target: next((row for row in validation_model_rows(validation, "LOPO", "FULL_WITH_EXTRAPOLATION_DIAGNOSTIC") if row["model"] == model_name and row["target"] == target), {}) for target in TARGET_FIELDS}
        lopo_interp = {target: next((row for row in validation_model_rows(validation, "LOPO", "INTERPOLATION_ONLY") if row["model"] == model_name and row["target"] == target), {}) for target in TARGET_FIELDS}
        loho_pass = all(candidate_metric_pass(row) and finite(row.get("after_mean_height_position_bias_range_mm")) is not None and float(row["after_mean_height_position_bias_range_mm"]) < 0.05 for row in loho_full.values())
        lopo_full_metric = all(candidate_metric_pass(row) and finite(row.get("after_mean_height_position_bias_range_mm")) is not None and float(row["after_mean_height_position_bias_range_mm"]) < 0.05 for row in lopo_full.values())
        # Full diagnostic rows contain extrapolated predictions, so they cannot
        # establish strict LOPO interpolation coverage. Check the separate
        # INTERPOLATION_ONLY aggregate instead.
        lopo_interp_complete = all(float(row.get("interpolation_coverage") or 0.0) >= 1.0 for row in lopo_interp.values())
        lopo_pass = bool(lopo_full_metric and lopo_interp_complete)
        pooled_rows = [row for row in pooled if row["model"] == model_name]
        candidate_status.append({
            "model": model_name,
            "loho_full_metrics_pass_all_targets": loho_pass,
            "lopo_full_metrics_pass_all_targets": lopo_full_metric,
            "lopo_full_interpolation_complete": lopo_interp_complete,
            "lopo_interpolation_only_coverage": {target: lopo_interp[target].get("interpolation_coverage") for target in TARGET_FIELDS},
            "lopo_pass": lopo_pass,
            "pooled_after_mean_height_range_max_mm": max((float(row["after_mean_height_position_bias_range_mm"]) for row in pooled_rows), default=None),
        })
    loho_pass_any = any(row["loho_full_metrics_pass_all_targets"] for row in candidate_status)
    lopo_pass_any = any(row["lopo_pass"] for row in candidate_status)
    best_candidates = [row["model"] for row in candidate_status if row["loho_full_metrics_pass_all_targets"] and row["lopo_pass"]]
    range_below = bool(best_candidates)
    if best_candidates:
        best_model = best_candidates[0]
    else:
        best_model = "NONE"
    if best_model != "NONE":
        feasible = "YES"
    elif repeatable and (loho_pass_any or any(row["lopo_full_metrics_pass_all_targets"] for row in candidate_status)):
        feasible = "PARTIAL"
    else:
        feasible = "NO"
    decisions = {
        "INTERIOR_SPATIAL_PATTERN_REPEATABLE": "YES" if repeatable else "NO",
        "BEST_GV_MODEL": best_model,
        "LOHO_SPATIAL_CORRECTION_PASS": "YES" if loho_pass_any else "NO",
        "LOPO_SPATIAL_CORRECTION_PASS": "YES" if lopo_pass_any else "NO",
        "POSITION_BIAS_RANGE_BELOW_0P05MM": "YES" if range_below else "NO",
        "INTERIOR_SPATIAL_CORRECTION_FEASIBLE": feasible,
        "profile_correlations": profile_correlations,
        "candidate_status": candidate_status,
        "selection_rule": "A candidate must improve/tie at least 3/4 held-out metrics with no >1% worsening, keep mean height-wise Position Bias Range <0.05 mm for both H1/H-B2; LOPO additionally requires 100% interpolation coverage.",
    }
    return decisions, profile_correlations


def plot_profile(profile_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, target in zip(axes, TARGET_FIELDS, strict=True):
        for height in HEIGHT_ORDER:
            items = [row for row in profile_rows if row["height_label"] == height]
            items.sort(key=lambda row: float(row["v_mean_px"]))
            axis.plot([row["v_mean_px"] for row in items], [row[f"{target}_spatial_residual_mm"] for row in items], marker="o", color={"h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}[height], label=height)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(f"{target.upper()} strict p02–p09")
        axis.set_xlabel("height-ROI raw_v (px)")
        axis.set_ylabel("height-demeaned spatial residual (mm)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("A-4 strict interior spatial profiles", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_before_after(predictions: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    position_indices = {position: index for index, position in enumerate(POSITION_ORDER)}
    for row_index, target in enumerate(TARGET_FIELDS):
        base_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            base_grouped[(str(row["height_label"]), str(row["position_id"]))].append(float(row[TARGET_FIELDS[target]]))
        for col_index, model_name in enumerate(MODEL_ORDER):
            axis = axes[row_index, col_index]
            for height in HEIGHT_ORDER:
                before_y = [float(np.mean(base_grouped[(height, position)])) for position in POSITION_ORDER]
                after_grouped: dict[str, list[float]] = defaultdict(list)
                for item in predictions:
                    if item["target"] == target and item["model"] == model_name and item["height_label"] == height:
                        after_grouped[str(item["position_id"])].append(float(item["after"]))
                after_y = [float(np.mean(after_grouped[position])) if after_grouped[position] else np.nan for position in POSITION_ORDER]
                color = {"h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}[height]
                axis.plot(range(len(POSITION_ORDER)), before_y, color=color, marker="o", linestyle="--", alpha=0.55)
                axis.plot(range(len(POSITION_ORDER)), after_y, color=color, marker="s", linestyle="-", label=height if col_index == 0 and row_index == 0 else None)
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.axvspan(-0.5, 0.5, color="#999999", alpha=0.08)
            axis.axvspan(6.5, 7.5, color="#999999", alpha=0.08)
            axis.set_xticks(range(len(POSITION_ORDER)), POSITION_ORDER)
            axis.set_title(f"{target.upper()} · {model_name}", pad=8)
            axis.set_xlabel("position")
            axis.set_ylabel(TARGET_LABELS[target])
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "A-4 LOPO before/after position bias (full diagnostic prediction)\n"
        "solid = after; dashed = before; shaded p02/p09 = diagnostic extrapolation",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def report_text(
    rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    decisions: dict[str, Any],
    profile_correlations: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> str:
    lines = [
        "# Task A-4｜Interior spatial correction feasibility",
        "",
        *[f"- `{key} = {value}`" for key, value in decisions.items() if key.isupper()],
        "",
        "本轮只分析 strict-reference 的 p02–p09；target 为 A-3/A-3B Local-reference 的 H1 与 H-B2 residual。Local-reference 仍是 diagnostic reference，不是 production truth。",
        "",
        "## Provenance / filtering",
        "",
        f"- A-3 feature input=`{A3_FEATURES}`；A-3B strict registry=`{A3B_EDGE_AUDIT}`；严格筛选后 `{len(rows)}` 帧、24 个 height×position condition、每 condition 20 repeats。",
        f"- 过滤条件：`position_id in {{p02,...,p09}}` 且 A-3B `strict_reference_available=True`；其 strict 定义为 `{STRICT_DEFINITION}`；没有读取 PNG、没有运行 Steger、没有重新拟合 C0/C1/H1/H-B2。",
        "- 每个 height 的 descriptive profile 使用该 height 的全部 8 个 strict positions 计算 `e_spatial=e-mean(e_height)`。LOHO/LOPO 中的 height mean 只由该 fold 的 train rows 计算，test residual 不参与拟合或中心化。",
        "- 由于 raw_v 是 full-sensor v，LOHO 的 h30 p02/p09 可能超出两训练高度的 v 域；LOPO 的 p02/p09 位于 p02–p09 子集边界，无法被其余 positions 真正 bracket。两类情况均单独记录，外推结果不冒充 interpolation。",
        "",
        "## Spatial profile",
        "",
        "| height | position | raw_v mean | H1 spatial | H-B2 spatial |",
        "|---|---|---:|---:|---:|",
    ]
    for row in profile_rows:
        lines.append(f"| {row['height_label']} | {row['position_id']} | {fmt(row['v_mean_px'], 1)} | {fmt(row['h1_spatial_residual_mm'])} | {fmt(row['hb2_spatial_residual_mm'])} |")
    lines += [
        "",
        "### Cross-height profile repeatability",
        "",
        "| target | height pair | Pearson | Spearman |",
        "|---|---|---:|---:|",
    ]
    for row in profile_correlations:
        lines.append(f"| {row['target']} | {row['height_pair']} | {fmt(row['pearson_r'])} | {fmt(row['spearman_rho'])} |")
    lines += [
        "",
        "判定采用预注册的三组 height-pair Pearson 均 ≥ 0.50 且方向一致；该判定只说明 profile repeatability，不等于 correction 已通过 grouped validation。",
        "",
        "## G(v) candidates",
        "",
        "- `PIECEWISE_LINEAR`：线性 spline，3 个由 train v 的 25/50/75% 分位点确定的 interior breakpoints。",
        "- `SPLINE_3K` / `SPLINE_4K`：degree-3 clamped cubic B-spline，分别使用 3/4 个 train-derived interior knots；这是模型候选的低自由度定义，不是 dense LUT 或高阶 polynomial。",
        "- 每个 fold 的 knot、basis coefficient、height mean 均只从 train rows 得到。",
        "",
        "## Pooled fit（仅描述，不用于选模）",
        "",
        "| target | model | spatial fit RMSE | spatial fit P95 | before mean-range | after mean-range | after RMSE | after P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(f"| {row['target']} | {row['model']} | {fmt(row['spatial_fit_rmse_mm'])} | {fmt(row['spatial_fit_p95_abs_mm'])} | {fmt(row['before_mean_height_position_bias_range_mm'])} | {fmt(row['after_mean_height_position_bias_range_mm'])} | {fmt(row['after_rmse_mm'])} | {fmt(row['after_p95_abs_mm'])} |")
    lines += [
        "",
        "## LOHO / LOPO held-out validation",
        "",
        "选择规则同时要求四项 held-out 指标（RMSE、P95、Position Bias Range、worst-position error）至少 3 项改善/持平，且没有指标相对 before 恶化超过 1%；另外要求平均 height-wise Position Bias Range <0.05 mm。LOPO 还必须 100% interpolation coverage。",
        "",
        "| scheme | mode | target | model | test/pred n | coverage | before range | after range | before P95 | after P95 | before RMSE | after RMSE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scheme in ("LOHO", "LOPO"):
        for mode in VALIDATION_MODES:
            for target in TARGET_FIELDS:
                for model_name in MODEL_ORDER:
                    row = next((item for item in validation if item["validation_scheme"] == scheme and item["evaluation_mode"] == mode and item["heldout_group"] == "ALL_HELDOUT" and item["target"] == target and item["model"] == model_name), {})
                    lines.append(f"| {scheme} | {mode} | {target} | {model_name} | {row.get('test_n','NA')}/{row.get('prediction_n','NA')} | {fmt(row.get('interpolation_coverage'))} | {fmt(row.get('before_mean_height_position_bias_range_mm'))} | {fmt(row.get('after_mean_height_position_bias_range_mm'))} | {fmt(row.get('before_p95_abs_mm'))} | {fmt(row.get('after_p95_abs_mm'))} | {fmt(row.get('before_rmse_mm'))} | {fmt(row.get('after_rmse_mm'))} |")
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Profile repeatability=`{decisions['INTERIOR_SPATIAL_PATTERN_REPEATABLE']}`；该结论来自 p02–p09 的跨高度 demeaned profile。",
        f"- LOHO pass=`{decisions['LOHO_SPATIAL_CORRECTION_PASS']}`；LOPO pass=`{decisions['LOPO_SPATIAL_CORRECTION_PASS']}`。LOPO p02/p09 的严格插值 coverage 不完整时，不以外推后的 full diagnostic 数字升级为 PASS。",
        f"- `BEST_GV_MODEL={decisions['BEST_GV_MODEL']}`；`POSITION_BIAS_RANGE_BELOW_0P05MM={decisions['POSITION_BIAS_RANGE_BELOW_0P05MM']}`；`INTERIOR_SPATIAL_CORRECTION_FEASIBLE={decisions['INTERIOR_SPATIAL_CORRECTION_FEASIBLE']}`。",
        "- 本轮只判断 feasibility，没有 freeze G(v)，没有修改 production config、Ground、ROI、Steger 或 reconstruction。",
        "",
        "## Boundaries",
        "",
        "- p01/p10 从未进入本轮模型或指标。",
        "- Local-reference residual 仅用于 diagnostic attribution；任何 G(v) fit 都不是 production correction。",
        "- `a4_lopo_validation.csv` 同时保留 interpolation-only 与 full-with-extrapolation-diagnostic 两种结果，必须结合 coverage 解读。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    required = [A3_FEATURES, A3B_EDGE_AUDIT, A3B_REPORT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"A-4 required artifact missing: {missing}")
    features = read_csv(A3_FEATURES)
    edge_audit = read_csv(A3B_EDGE_AUDIT)
    validate_inputs(features, edge_audit)
    rows = select_strict_interior(features, edge_audit)
    profile_rows, profiles = condition_profile_rows(rows)
    pooled = pooled_model_comparison(rows)
    validation, predictions = run_validation(rows)
    decisions, profile_correlations = derive_decisions(profiles, validation, pooled)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "a4_spatial_profile_by_height.csv", profile_rows, list(profile_rows[0].keys()))
    model_comparison = list(pooled)
    for scheme in ("LOHO", "LOPO"):
        for mode in VALIDATION_MODES:
            for row in validation_model_rows(validation, scheme, mode):
                model_comparison.append({
                    "evaluation_scope": f"{scheme}_AGGREGATE",
                    "evaluation_mode": mode,
                    "model": row["model"],
                    "target": row["target"],
                    "formula": row["formula"],
                    "n": row["prediction_n"],
                    "parameter_count": None,
                    "v_min_px": None,
                    "v_max_px": None,
                    "spatial_fit_rmse_mm": None,
                    "spatial_fit_p95_abs_mm": None,
                    "before_rmse_mm": row.get("before_rmse_mm"),
                    "after_rmse_mm": row.get("after_rmse_mm"),
                    "before_p95_abs_mm": row.get("before_p95_abs_mm"),
                    "after_p95_abs_mm": row.get("after_p95_abs_mm"),
                    "before_position_bias_range_mm": row.get("before_position_bias_range_mm"),
                    "after_position_bias_range_mm": row.get("after_position_bias_range_mm"),
                    "before_mean_height_position_bias_range_mm": row.get("before_mean_height_position_bias_range_mm"),
                    "after_mean_height_position_bias_range_mm": row.get("after_mean_height_position_bias_range_mm"),
                    "before_worst_position_error_mm": row.get("before_worst_position_error_mm"),
                    "after_worst_position_error_mm": row.get("after_worst_position_error_mm"),
                    "extrapolated_n": row.get("extrapolated_n"),
                    "interpolation_coverage": row.get("interpolation_coverage"),
                })
    write_csv(OUTPUT_DIR / "a4_spatial_model_comparison.csv", model_comparison, list(model_comparison[0].keys()))
    validation_fields = list(validation[0].keys())
    write_csv(OUTPUT_DIR / "a4_loho_validation.csv", [row for row in validation if row["validation_scheme"] == "LOHO"], validation_fields)
    write_csv(OUTPUT_DIR / "a4_lopo_validation.csv", [row for row in validation if row["validation_scheme"] == "LOPO"], validation_fields)
    write_csv(OUTPUT_DIR / "a4_profile_correlations.csv", profile_correlations, list(profile_correlations[0].keys()))
    plot_profile(profile_rows, OUTPUT_DIR / "a4_spatial_profile.png")
    plot_before_after(predictions, rows, OUTPUT_DIR / "a4_before_after_position_bias.png")
    provenance = {
        "task": "A-4 Interior spatial correction feasibility",
        "inputs": {"a3_features": str(A3_FEATURES), "a3b_edge_audit": str(A3B_EDGE_AUDIT), "a3b_report": str(A3B_REPORT)},
        "reuse": {"png_read": False, "steger_rerun": False, "c0_c1_h1_hb2_refit": False, "production_modified": False, "p01_p10_used": False, "local_reference_diagnostic_only": True},
        "selection": {"frames": len(rows), "conditions": len({row["condition_id"] for row in rows}), "positions": list(POSITION_ORDER), "heights": list(HEIGHT_ORDER), "strict_source": "A-3B edge audit strict_reference_available=True", "strict_definition": STRICT_DEFINITION},
        "model_definitions": MODEL_DEFS,
        "validation_protocol": {"height_centering": "train-fold height mean for LOHO/LOPO; full strict height mean only for descriptive profile/pooled fit", "metric_pass": decisions["selection_rule"], "lopo_interpolation_only": True, "extrapolation_is_diagnostic_only": True},
        "decisions": decisions,
    }
    write_json(OUTPUT_DIR / "a4_provenance.json", provenance)
    (OUTPUT_DIR / "report.md").write_text(report_text(rows, profile_rows, decisions, profile_correlations, pooled, validation), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "decisions": decisions, "selected_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
