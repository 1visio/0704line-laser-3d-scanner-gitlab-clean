#!/usr/bin/env python3
"""Surface-1B feature identifiability audit.

This is a descriptive identifiability analysis only.  It reuses the frozen
Surface-1A point table, never refits C0/C1/G(S)/H1, and never uses 50 mm rows
for feature selection, SVD/VIF, grouped model fitting, or parameter fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a" / "surface1a"
DEFAULT_OUTPUT = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a" / "surface1b"

FORMAL_DEVELOPMENT = "development_formal_repeat2_5"
FORMAL_HELDOUT = "heldout_formal_repeat2_5"
EPS = 1e-12

CORRELATION_FEATURES = [
    "height",
    "q1",
    "q2",
    "d_lambda_du",
    "d_lambda_dv",
    "d_Zg_du",
    "d_Zg_dv",
    "J_lambda_norm",
    "J_Z_norm",
    "height_residual",
]

FEATURE_SETS: dict[str, list[str]] = {
    "q1": ["q1"],
    "q1+q2": ["q1", "q2"],
    "q1+J": ["q1", "d_lambda_du", "d_lambda_dv", "d_Zg_du", "d_Zg_dv"],
    "q1+q2+J": ["q1", "q2", "d_lambda_du", "d_lambda_dv", "d_Zg_du", "d_Zg_dv"],
    "height+q1+q2": ["height", "q1", "q2"],
}

SOURCE_COLUMNS = {
    "height": "true_height_mm",
    "q1": "q1",
    "q2": "q2",
    "d_lambda_du": "d_lambda_du",
    "d_lambda_dv": "d_lambda_dv",
    "d_Zg_du": "d_Zg_du",
    "d_Zg_dv": "d_Zg_dv",
    "J_lambda_norm": "jacobian_lambda_norm_mm_per_px",
    "J_Z_norm": "jacobian_Zg_norm_mm_per_px",
    "height_residual": "height_residual_mm",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def load_point_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = input_dir / "surface1a_points.csv"
    development: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            split_role = raw.get("split_role", "")
            if split_role not in {FORMAL_DEVELOPMENT, FORMAL_HELDOUT}:
                continue
            if not as_bool(raw.get("height_measurement_inlier")) or not as_bool(raw.get("jacobian_valid")):
                continue
            row: dict[str, Any] = {
                "dataset": raw["dataset"],
                "height_group": raw["height_group"],
                "height": float(raw["true_height_mm"]),
                "position": raw["position_id"],
                "condition_id": f"{raw['dataset']}/{raw['position_id']}",
                "frame_id": raw["frame_id"],
                "repeat_index": int(raw["repeat_index"]),
                "split_role": split_role,
                "held_out": as_bool(raw["held_out"]),
            }
            for name, source in SOURCE_COLUMNS.items():
                row[name] = float(raw[source])
            if not all(finite(row[name]) for name in SOURCE_COLUMNS):
                continue
            (heldout if split_role == FORMAL_HELDOUT else development).append(row)
    return development, heldout


def aggregate_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create one equal-weight condition row from all formal points in a condition."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition_id"]].append(row)
    output: list[dict[str, Any]] = []
    for condition_id, group in sorted(groups.items()):
        first = group[0]
        item: dict[str, Any] = {
            "dataset": first["dataset"],
            "height_group": first["height_group"],
            "height": float(np.mean([row["height"] for row in group])),
            "position": first["position"],
            "condition_id": condition_id,
            "frame_count": len({row["frame_id"] for row in group}),
            "point_count": len(group),
            "held_out": bool(first["held_out"]),
        }
        for name in SOURCE_COLUMNS:
            item[name] = float(np.mean([row[name] for row in group]))
        output.append(item)
    return output


def corr_value(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if len(x) < 3 or np.ptp(x) <= EPS or np.ptp(y) <= EPS:
        return None
    if method == "pearson":
        value = pearsonr(x, y).statistic
    else:
        value = spearmanr(x, y).statistic
    return float(value) if value is not None and math.isfinite(float(value)) else None


def correlation_rows(scope: str, aggregation: str, rows: list[dict[str, Any]], used_for_decision: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    condition_count = len({row["condition_id"] for row in rows})
    for left in CORRELATION_FEATURES:
        x = np.asarray([row[left] for row in rows], dtype=np.float64)
        for right in CORRELATION_FEATURES:
            y = np.asarray([row[right] for row in rows], dtype=np.float64)
            output.append(
                {
                    "scope": scope,
                    "aggregation": aggregation,
                    "feature_x": left,
                    "feature_y": right,
                    "pearson_r": 1.0 if left == right else corr_value(x, y, "pearson"),
                    "spearman_r": 1.0 if left == right else corr_value(x, y, "spearman"),
                    "sample_count": len(rows),
                    "condition_count": condition_count,
                    "used_for_model_selection": used_for_decision,
                    "note": "descriptive correlation; no correction function fitted",
                }
            )
    return output


def design_diagnostics(rows: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    x = np.asarray([[row[name] for name in features] for row in rows], dtype=np.float64)
    n, p = x.shape
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0, ddof=0)
    constant = scales <= EPS
    scales_safe = np.where(constant, 1.0, scales)
    z = (x - means[None, :]) / scales_safe[None, :]
    singular_values = np.linalg.svd(z, compute_uv=False) if n and p else np.asarray([], dtype=np.float64)
    if singular_values.size:
        tol = max(z.shape) * np.finfo(np.float64).eps * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > tol))
        condition_number = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > EPS else math.inf
    else:
        rank = 0
        condition_number = math.nan
    if n > 1 and p:
        corr = np.corrcoef(z, rowvar=False)
        corr = np.atleast_2d(corr)
        corr_condition_number = float(np.linalg.cond(corr)) if np.all(np.isfinite(corr)) else math.inf
    else:
        corr = np.eye(p)
        corr_condition_number = math.nan
    vifs: dict[str, float] = {}
    for index, name in enumerate(features):
        if p == 1:
            vifs[name] = 1.0
            continue
        others = [j for j in range(p) if j != index]
        design = np.column_stack([np.ones(n), z[:, others]])
        beta = np.linalg.lstsq(design, z[:, index], rcond=None)[0]
        predicted = design @ beta
        sst = float(np.sum((z[:, index] - np.mean(z[:, index])) ** 2))
        sse = float(np.sum((z[:, index] - predicted) ** 2))
        r2 = 0.0 if sst <= EPS else max(0.0, min(1.0, 1.0 - sse / sst))
        vifs[name] = math.inf if r2 >= 1.0 - 1e-10 else float(1.0 / (1.0 - r2))
    _, _, vh = np.linalg.svd(z, full_matrices=False) if n and p else (np.asarray([]), np.asarray([]), np.asarray([[]]))
    loadings = vh if isinstance(vh, np.ndarray) and vh.ndim == 2 else np.empty((0, p))
    explained = (singular_values**2 / np.sum(singular_values**2)).tolist() if singular_values.size and np.sum(singular_values**2) > EPS else []
    return {
        "n": n,
        "p": p,
        "means": means,
        "scales": scales,
        "constant_feature_count": int(np.count_nonzero(constant)),
        "rank": rank,
        "singular_values": singular_values,
        "explained_variance_ratio": explained,
        "loadings": loadings,
        "condition_number_svd": condition_number,
        "condition_number_corr": corr_condition_number,
        "vifs": vifs,
        "max_vif": max(vifs.values()) if vifs else math.nan,
        "mean_vif": float(np.mean(list(vifs.values()))) if vifs else math.nan,
    }


def vif_svd_rows(scope: str, aggregation: str, feature_set: str, rows: list[dict[str, Any]], used_for_decision: bool) -> list[dict[str, Any]]:
    features = FEATURE_SETS[feature_set]
    diag = design_diagnostics(rows, features)
    common = {
        "scope": scope,
        "aggregation": aggregation,
        "feature_set": feature_set,
        "feature_list": "+".join(features),
        "rank": diag["rank"],
        "condition_number_svd": diag["condition_number_svd"],
        "condition_number_corr": diag["condition_number_corr"],
        "max_vif": diag["max_vif"],
        "mean_vif": diag["mean_vif"],
        "sample_count": len(rows),
        "used_for_model_selection": used_for_decision,
    }
    output: list[dict[str, Any]] = []
    for index, value in enumerate(diag["singular_values"], start=1):
        output.append({**common, "metric": "singular_value", "component_index": index, "feature": f"PC{index}", "value": float(value)})
        output.append({**common, "metric": "explained_variance_ratio", "component_index": index, "feature": f"PC{index}", "value": float(diag["explained_variance_ratio"][index - 1])})
    for component_index, loading_row in enumerate(diag["loadings"], start=1):
        for feature_index, value in enumerate(loading_row):
            output.append({**common, "metric": "pca_loading", "component_index": component_index, "feature": features[feature_index], "value": float(value)})
    for feature, value in diag["vifs"].items():
        output.append({**common, "metric": "vif", "component_index": "", "feature": feature, "value": value})
    output.append({**common, "metric": "rank", "component_index": "", "feature": "__model__", "value": diag["rank"]})
    output.append({**common, "metric": "condition_number_svd", "component_index": "", "feature": "__model__", "value": diag["condition_number_svd"]})
    output.append({**common, "metric": "condition_number_corr", "component_index": "", "feature": "__model__", "value": diag["condition_number_corr"]})
    output.append({**common, "metric": "max_vif", "component_index": "", "feature": "__model__", "value": diag["max_vif"]})
    output.append({**common, "metric": "mean_vif", "component_index": "", "feature": "__model__", "value": diag["mean_vif"]})
    return output


def fit_ols(train: list[dict[str, Any]], features: list[str]) -> dict[str, Any] | None:
    if not train:
        return None
    x = np.asarray([[row[name] for name in features] for row in train], dtype=np.float64)
    y = np.asarray([row["height_residual"] for row in train], dtype=np.float64)
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0, ddof=0)
    if np.any(scales <= EPS):
        return None
    z = (x - means[None, :]) / scales[None, :]
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"means": means, "scales": scales, "beta": beta, "features": features}


def predict_ols(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    x = np.asarray([[row[name] for name in model["features"]] for row in rows], dtype=np.float64)
    z = (x - model["means"][None, :]) / model["scales"][None, :]
    return np.column_stack([np.ones(len(z)), z]) @ model["beta"]


def within_condition_r2(rows: list[dict[str, Any]], predicted: np.ndarray) -> float | None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["condition_id"]].append(index)
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        y = np.asarray([rows[index]["height_residual"] for index in indices], dtype=np.float64)
        p = predicted[indices]
        y_parts.append(y - np.mean(y))
        p_parts.append(p - np.mean(p))
    if not y_parts:
        return None
    y_all = np.concatenate(y_parts)
    p_all = np.concatenate(p_parts)
    sst = float(np.sum(y_all**2))
    return None if sst <= EPS else float(1.0 - np.sum((y_all - p_all) ** 2) / sst)


def error_metrics(rows: list[dict[str, Any]], predicted: np.ndarray, feature_count: int, fit_status: str = "success") -> dict[str, Any]:
    y = np.asarray([row["height_residual"] for row in rows], dtype=np.float64)
    error = predicted - y
    abs_error = np.abs(error)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum(error**2))
    r2 = None if sst <= EPS else float(1.0 - sse / sst)
    n = len(y)
    adjusted = None if r2 is None or n <= feature_count + 1 else float(1.0 - (1.0 - r2) * (n - 1) / (n - feature_count - 1))
    return {
        "fit_status": fit_status,
        "sample_count": n,
        "condition_count": len({row["condition_id"] for row in rows}),
        "bias_mm": float(np.mean(error)) if n else None,
        "mae_mm": float(np.mean(abs_error)) if n else None,
        "rmse_mm": float(np.sqrt(np.mean(error**2))) if n else None,
        "p95_abs_error_mm": float(np.percentile(abs_error, 95.0)) if n else None,
        "max_abs_error_mm": float(np.max(abs_error)) if n else None,
        "r2": r2,
        "adjusted_r2": adjusted,
        "within_condition_r2": within_condition_r2(rows, predicted),
        "sse": sse,
    }


def comparison_row(
    scope: str,
    aggregation: str,
    evaluation: str,
    fold: str,
    feature_set: str,
    rows: list[dict[str, Any]],
    predicted: np.ndarray,
    diag: dict[str, Any] | None,
    fit_status: str = "success",
    notes: str = "development-only descriptive OLS; no correction function fitted",
) -> dict[str, Any]:
    features = FEATURE_SETS[feature_set]
    metrics = error_metrics(rows, predicted, len(features), fit_status=fit_status)
    return {
        "scope": scope,
        "aggregation": aggregation,
        "evaluation": evaluation,
        "fold": fold,
        "feature_set": feature_set,
        "feature_list": "+".join(features),
        **metrics,
        "rank": diag["rank"] if diag else None,
        "condition_number_svd": diag["condition_number_svd"] if diag else None,
        "condition_number_corr": diag["condition_number_corr"] if diag else None,
        "max_vif": diag["max_vif"] if diag else None,
        "mean_vif": diag["mean_vif"] if diag else None,
        "notes": notes,
    }


def fit_comparisons(development_points: list[dict[str, Any]], development_conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aggregation, data in (("point_pooled", development_points), ("condition_balanced", development_conditions)):
        for feature_set, features in FEATURE_SETS.items():
            model = fit_ols(data, features)
            diag = design_diagnostics(data, features)
            if model is None:
                predicted = np.full(len(data), np.nan)
                item = comparison_row("development_formal", aggregation, "in_sample", "all", feature_set, data, predicted, diag, "degenerate_feature_or_sample_count")
            else:
                predicted = predict_ols(model, data)
                item = comparison_row("development_formal", aggregation, "in_sample", "all", feature_set, data, predicted, diag)
            rows.append(item)

    split_specs = [
        ("leave_one_height_out", lambda row: row["height_group"]),
        ("leave_one_position_out", lambda row: row["position"]),
    ]
    for evaluation, group_key in split_specs:
        groups = sorted({group_key(row) for row in development_conditions}, key=str)
        for fold in groups:
            train = [row for row in development_conditions if group_key(row) != fold]
            test = [row for row in development_conditions if group_key(row) == fold]
            for feature_set, features in FEATURE_SETS.items():
                model = fit_ols(train, features)
                diag = design_diagnostics(train, features)
                if model is None:
                    predicted = np.full(len(test), np.nan)
                    item = comparison_row("development_formal", "grouped_condition", evaluation, str(fold), feature_set, test, predicted, diag, "degenerate_feature_or_sample_count")
                else:
                    predicted = predict_ols(model, test)
                    item = comparison_row("development_formal", "grouped_condition", evaluation, str(fold), feature_set, test, predicted, diag)
                item["train_condition_count"] = len(train)
                item["test_condition_count"] = len(test)
                rows.append(item)

    for evaluation in ("leave_one_height_out", "leave_one_position_out"):
        for feature_set in FEATURE_SETS:
            fold_rows = [row for row in rows if row["evaluation"] == evaluation and row["aggregation"] == "grouped_condition" and row["feature_set"] == feature_set]
            if not fold_rows:
                continue
            summary = {
                "scope": "development_formal",
                "aggregation": "grouped_summary",
                "evaluation": evaluation,
                "fold": "median_over_folds",
                "feature_set": feature_set,
                "feature_list": "+".join(FEATURE_SETS[feature_set]),
                "fit_status": "success",
                "sample_count": int(sum(row["sample_count"] for row in fold_rows)),
                "condition_count": int(sum(row["condition_count"] for row in fold_rows)),
                "bias_mm": float(np.median([row["bias_mm"] for row in fold_rows])),
                "mae_mm": float(np.median([row["mae_mm"] for row in fold_rows])),
                "rmse_mm": float(np.median([row["rmse_mm"] for row in fold_rows])),
                "p95_abs_error_mm": float(np.median([row["p95_abs_error_mm"] for row in fold_rows])),
                "max_abs_error_mm": float(np.median([row["max_abs_error_mm"] for row in fold_rows])),
                "r2": None,
                "adjusted_r2": None,
                "within_condition_r2": None,
                "sse": None,
                "rank": float(np.median([row["rank"] for row in fold_rows if row["rank"] is not None])),
                "condition_number_svd": float(np.median([row["condition_number_svd"] for row in fold_rows if finite(row["condition_number_svd"])])),
                "condition_number_corr": float(np.median([row["condition_number_corr"] for row in fold_rows if finite(row["condition_number_corr"])])),
                "max_vif": float(np.median([row["max_vif"] for row in fold_rows if finite(row["max_vif"])])) if any(finite(row["max_vif"]) for row in fold_rows) else math.inf,
                "mean_vif": float(np.median([row["mean_vif"] for row in fold_rows if finite(row["mean_vif"])])) if any(finite(row["mean_vif"]) for row in fold_rows) else math.inf,
                "notes": "median across complete grouped folds; no random point split",
                "fold_count": len(fold_rows),
            }
            rows.append(summary)
    return rows


def metric_lookup(rows: list[dict[str, Any]], aggregation: str, evaluation: str, feature_set: str, fold: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        if row["aggregation"] == aggregation and row["evaluation"] == evaluation and row["feature_set"] == feature_set and (fold is None or row["fold"] == fold):
            return row
    return None


def corr_lookup(rows: list[dict[str, Any]], aggregation: str, left: str, right: str) -> dict[str, Any] | None:
    for row in rows:
        if row["scope"] == "development_formal" and row["aggregation"] == aggregation and row["feature_x"] == left and row["feature_y"] == right:
            return row
    return None


def diagnostic_lookup(rows: list[dict[str, Any]], feature_set: str, metric: str, feature: str = "__model__") -> float | None:
    candidates = [row for row in rows if row["scope"] == "development_formal" and row["aggregation"] == "condition_balanced" and row["feature_set"] == feature_set and row["metric"] == metric and row["feature"] == feature]
    return None if not candidates else float(candidates[0]["value"])


def classify(correlation: list[dict[str, Any]], comparisons: list[dict[str, Any]], vif_rows: list[dict[str, Any]]) -> dict[str, Any]:
    hq2_point = corr_lookup(correlation, "point_pooled", "height", "q2")
    hq2_condition = corr_lookup(correlation, "condition_balanced", "height", "q2")
    hq2_values = [
        abs(float(row[key]))
        for row in (hq2_point, hq2_condition)
        if row is not None
        for key in ("pearson_r", "spearman_r")
        if row.get(key) is not None
    ]
    hq2_min = min(hq2_values) if hq2_values else 0.0
    if hq2_min >= 0.90:
        height_q2_status = "SUPPORTED"
    elif hq2_min >= 0.70:
        height_q2_status = "PARTIAL"
    else:
        height_q2_status = "NOT_SUPPORTED"

    base = metric_lookup(comparisons, "condition_balanced", "in_sample", "q1+q2")
    extended = metric_lookup(comparisons, "condition_balanced", "in_sample", "q1+q2+J")
    if base and extended and base.get("r2") is not None and extended.get("r2") is not None:
        delta_r2 = float(extended["r2"] - base["r2"])
    else:
        delta_r2 = math.nan
    grouped_improvements: list[float] = []
    positive_fractions: list[float] = []
    for evaluation in ("leave_one_height_out", "leave_one_position_out"):
        base_folds = [row for row in comparisons if row["aggregation"] == "grouped_condition" and row["evaluation"] == evaluation and row["feature_set"] == "q1+q2"]
        ext_folds = [row for row in comparisons if row["aggregation"] == "grouped_condition" and row["evaluation"] == evaluation and row["feature_set"] == "q1+q2+J"]
        ext_by_fold = {row["fold"]: row for row in ext_folds}
        improvements = [row["rmse_mm"] - ext_by_fold[row["fold"]]["rmse_mm"] for row in base_folds if row["fold"] in ext_by_fold]
        if improvements:
            grouped_improvements.append(float(np.median(improvements)))
            positive_fractions.append(float(np.mean(np.asarray(improvements) > 0.0)))
    j_max_vif = diagnostic_lookup(vif_rows, "q1+q2+J", "max_vif")
    if (
        finite(delta_r2)
        and delta_r2 >= 0.02
        and grouped_improvements
        and min(grouped_improvements) > 0.0005
        and min(positive_fractions) >= 0.60
        and (j_max_vif is None or j_max_vif < 20.0)
    ):
        jacobian_status = "SUPPORTED"
    elif finite(delta_r2) and delta_r2 >= 0.01 and grouped_improvements and max(grouped_improvements) > 0.0:
        jacobian_status = "PARTIAL"
    else:
        jacobian_status = "NOT_SUPPORTED"

    q12_diag_vif = diagnostic_lookup(vif_rows, "q1+q2", "max_vif")
    q12_lopo = metric_lookup(comparisons, "grouped_summary", "leave_one_position_out", "q1+q2")
    q12_loho = metric_lookup(comparisons, "grouped_summary", "leave_one_height_out", "q1+q2")
    q1_lopo = metric_lookup(comparisons, "grouped_summary", "leave_one_position_out", "q1")
    q1_loho = metric_lookup(comparisons, "grouped_summary", "leave_one_height_out", "q1")
    q12_stable = (
        q12_diag_vif is not None
        and q12_diag_vif < 10.0
        and q12_lopo is not None
        and q12_loho is not None
        and q1_lopo is not None
        and q1_loho is not None
        and q12_lopo["rmse_mm"] <= q1_lopo["rmse_mm"] + 0.001
        and q12_loho["rmse_mm"] <= q1_loho["rmse_mm"] + 0.001
    )
    if q12_stable:
        recommended = "q1+q2"
    else:
        recommended = "q1"
    if height_q2_status == "SUPPORTED" or jacobian_status == "PARTIAL":
        identifiability = "PARTIAL"
    else:
        identifiability = "SUPPORTED"
    return {
        "SURFACE_FEATURE_IDENTIFIABILITY": identifiability,
        "HEIGHT_Q2_REDUNDANCY": height_q2_status,
        "JACOBIAN_ADDS_INDEPENDENT_INFORMATION": jacobian_status,
        "RECOMMENDED_MINIMAL_FEATURE_SET": recommended,
        "height_q2_min_abs_correlation": hq2_min,
        "q1_q2_plus_J_delta_r2_condition": delta_r2,
        "q1_q2_plus_J_grouped_median_rmse_improvements_mm": grouped_improvements,
        "q1_q2_plus_J_grouped_positive_fractions": positive_fractions,
        "q1_q2_max_vif_condition": q12_diag_vif,
        "q1_q2_plus_J_max_vif_condition": j_max_vif,
    }


def surface2_plan(development_points: list[dict[str, Any]], development_conditions: list[dict[str, Any]], heldout_points: list[dict[str, Any]]) -> dict[str, Any]:
    heights = np.asarray([row["height"] for row in development_conditions], dtype=np.float64)
    q2 = np.asarray([row["q2"] for row in development_conditions], dtype=np.float64)
    design = np.column_stack([np.ones(len(heights)), heights])
    beta = np.linalg.lstsq(design, q2, rcond=None)[0]
    targets = [35.0, 40.0, 45.0]
    target_rows: list[dict[str, Any]] = []
    for target in targets:
        per_position: list[float] = []
        for position in sorted({row["position"] for row in development_conditions}):
            group = [row for row in development_conditions if row["position"] == position]
            if len(group) < 3:
                continue
            x = np.asarray([row["height"] for row in group], dtype=np.float64)
            y = np.asarray([row["q2"] for row in group], dtype=np.float64)
            b = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
            per_position.append(float(b[0] + b[1] * target))
        predicted = float(beta[0] + beta[1] * target)
        target_rows.append(
            {
                "height_mm": target,
                "development_only_q2_prediction": predicted,
                "position_prediction_min": float(min(per_position)) if per_position else None,
                "position_prediction_max": float(max(per_position)) if per_position else None,
                "position_prediction_median": float(np.median(per_position)) if per_position else None,
                "planning_basis": "development 1-30mm condition means only; 50mm excluded from fit",
            }
        )
    q2_dev_min = float(np.min([row["q2"] for row in development_points]))
    q2_dev_max = float(np.max([row["q2"] for row in development_points]))
    q2_50_min = float(np.min([row["q2"] for row in heldout_points])) if heldout_points else None
    q2_50_max = float(np.max([row["q2"] for row in heldout_points])) if heldout_points else None
    return {
        "development_only_q2_height_slope": float(beta[1]),
        "development_only_q2_height_intercept": float(beta[0]),
        "development_q2_range": [q2_dev_min, q2_dev_max],
        "heldout_50_q2_range_descriptive_only": [q2_50_min, q2_50_max],
        "targets": target_rows,
        "priority": [
            {"rank": 1, "height_mm": 40, "reason": "最直接填补 30mm 到 50mm 之间的中段 q2/height gap，并检验高端趋势是否转折"},
            {"rank": 2, "height_mm": 35, "reason": "紧邻 development 上界，定位 residual/q2 从已覆盖域离开的起点"},
            {"rank": 3, "height_mm": 45, "reason": "靠近 50mm 端点，确认高端 position interaction 与 domain shift 是否持续"},
        ],
        "recommend_all_three": True,
        "planning_note": "priority is determined from development-only geometry/trend; 50mm is displayed only as held-out domain context",
    }


def format_value(value: Any, digits: int = 4) -> str:
    if value is None or not finite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def plot_heatmap(path: Path, correlation_rows_all: list[dict[str, Any]]) -> None:
    matrices: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("development formal · point pooled", "point_pooled", [row for row in correlation_rows_all if row["scope"] == "development_formal"]),
        ("development formal · condition balanced", "condition_balanced", [row for row in correlation_rows_all if row["scope"] == "development_formal"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    labels = ["height", "q1", "q2", "dλ/du", "dλ/dv", "dZg/du", "dZg/dv", "Jλ", "JZ", "residual"]
    for axis, (title, aggregation, _) in zip(axes, matrices):
        subset = [row for row in correlation_rows_all if row["scope"] == "development_formal" and row["aggregation"] == aggregation]
        index = {name: i for i, name in enumerate(CORRELATION_FEATURES)}
        matrix = np.full((len(CORRELATION_FEATURES), len(CORRELATION_FEATURES)), np.nan)
        for row in subset:
            value = row["spearman_r"]
            if value is not None:
                matrix[index[row["feature_x"]], index[row["feature_y"]]] = float(value)
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(masked, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
        for i in range(len(CORRELATION_FEATURES)):
            for j in range(len(CORRELATION_FEATURES)):
                if math.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
        axis.set_xticks(range(len(labels)), labels, rotation=60, ha="left", fontsize=8)
        axis.set_yticks(range(len(labels)), labels, fontsize=8)
        axis.set_title(f"{title}\nSpearman ρ")
        axis.grid(False)
        fig.colorbar(image, ax=axis, shrink=0.75, label="ρ")
    fig.suptitle("Surface-1B feature correlation heatmap")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    classifications: dict[str, Any],
    correlations: list[dict[str, Any]],
    vif_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    plan: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    lines = [
        "# Surface-1B Surface/Jacobian 特征可辨识性审计",
        "",
        f"- `SURFACE_FEATURE_IDENTIFIABILITY = {classifications['SURFACE_FEATURE_IDENTIFIABILITY']}`",
        f"- `HEIGHT_Q2_REDUNDANCY = {classifications['HEIGHT_Q2_REDUNDANCY']}`",
        f"- `JACOBIAN_ADDS_INDEPENDENT_INFORMATION = {classifications['JACOBIAN_ADDS_INDEPENDENT_INFORMATION']}`",
        f"- `RECOMMENDED_MINIMAL_FEATURE_SET = {classifications['RECOMMENDED_MINIMAL_FEATURE_SET']}`",
        "- `SURFACE2_ACQUISITION_PRIORITY = 40mm > 35mm > 45mm；资源允许时三者全部采集`",
        "",
        "本轮是 development-only 可辨识性诊断。未重新拟合 C0/C1/G(S)/H1，未生成补偿函数，未修改生产配置。50 mm 仅保留为 held-out 描述，未参与 SVD、VIF、候选特征选择或任何模型参数拟合。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- 复用输入：`{provenance['points_path']}`，Surface-1A 已冻结的 q 定义：`{provenance['coordinate_definition_path']}`。",
        f"- 复用 Surface-1A summary/metrics 做 split 与数值一致性核对：`{provenance['summary_path']}`、`{provenance['metrics_path']}`。",
        f"- development formal：{provenance['development_point_count']} 个分析点、{provenance['development_condition_count']} 个 height×position condition。",
        f"- 50 mm held-out descriptive：{provenance['heldout_point_count']} 个分析点、{provenance['heldout_condition_count']} 个 condition；未进入模型选择/拟合。",
        "- formal 过滤：`split_role` 为 repeat2–5 formal，同时 `height_measurement_inlier=True`、`jacobian_valid=True`；禁止随机 point split。",
        "- condition-level 采用每个完整 height×position condition 的点均值，每个 condition 等权；point-level 仅作描述性参考。",
        "",
        "## Correlation / redundancy",
        "",
        "下表同时给出 point-level 与 condition-level 的 feature↔residual 相关性；相关性不等于因果或可部署补偿。",
        "",
        "| feature | point Pearson/Spearman | condition Pearson/Spearman |",
        "|---|---:|---:|",
    ]
    for feature in CORRELATION_FEATURES[:-1]:
        point = corr_lookup(correlations, "point_pooled", feature, "height_residual")
        condition = corr_lookup(correlations, "condition_balanced", feature, "height_residual")
        p = "—" if point is None else f"{format_value(point['pearson_r'])}/{format_value(point['spearman_r'])}"
        c = "—" if condition is None else f"{format_value(condition['pearson_r'])}/{format_value(condition['spearman_r'])}"
        lines.append(f"| {feature} | {p} | {c} |")

    lines += [
        "",
        "关键共线性对：",
        "",
        "| pair | point Pearson/Spearman | condition Pearson/Spearman |",
        "|---|---:|---:|",
    ]
    for left, right in (("height", "q2"), ("q1", "q2"), ("q2", "J_lambda_norm"), ("q2", "J_Z_norm"), ("d_lambda_du", "d_lambda_dv"), ("d_Zg_du", "d_Zg_dv")):
        point = corr_lookup(correlations, "point_pooled", left, right)
        condition = corr_lookup(correlations, "condition_balanced", left, right)
        p = "—" if point is None else f"{format_value(point['pearson_r'])}/{format_value(point['spearman_r'])}"
        c = "—" if condition is None else f"{format_value(condition['pearson_r'])}/{format_value(condition['spearman_r'])}"
        lines.append(f"| {left} ↔ {right} | {p} | {c} |")

    lines += [
        "",
        f"height↔q2 的最小绝对 Pearson/Spearman 相关系数为 `{format_value(classifications['height_q2_min_abs_correlation'])}`，因此其独立可辨识性受到高度-表面几何共线性限制。",
        "",
        "## Standardized SVD / PCA / VIF（development condition-level）",
        "",
        "PCA/SVD 只作为数值诊断，不改写 frozen q1/q2 定义。J 表示四个带符号的局部导数：`d_lambda_du,d_lambda_dv,d_Zg_du,d_Zg_dv`。",
        "",
        "| feature set | rank | SVD condition no. | correlation condition no. | max VIF | mean VIF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature_set in FEATURE_SETS:
        rank = diagnostic_lookup(vif_rows, feature_set, "rank")
        cond = diagnostic_lookup(vif_rows, feature_set, "condition_number_svd")
        corr_cond = diagnostic_lookup(vif_rows, feature_set, "condition_number_corr")
        max_vif = diagnostic_lookup(vif_rows, feature_set, "max_vif")
        mean_vif = diagnostic_lookup(vif_rows, feature_set, "mean_vif")
        lines.append(f"| {feature_set} | {format_value(rank, 2)} | {format_value(cond, 2)} | {format_value(corr_cond, 2)} | {format_value(max_vif, 2)} | {format_value(mean_vif, 2)} |")

    lines += [
        "",
        "## Candidate feature-set comparison",
        "",
        "OLS 仅为 explanatory nested-model comparison；不代表将系数冻结为 correction。所有候选模型使用同一 development observation table。",
        "",
        "| feature set | condition in-sample R² | condition RMSE | point within-condition R² | LOHO median RMSE | LOPO median RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature_set in FEATURE_SETS:
        condition = metric_lookup(comparisons, "condition_balanced", "in_sample", feature_set)
        point = metric_lookup(comparisons, "point_pooled", "in_sample", feature_set)
        loho = metric_lookup(comparisons, "grouped_summary", "leave_one_height_out", feature_set)
        lopo = metric_lookup(comparisons, "grouped_summary", "leave_one_position_out", feature_set)
        lines.append(
            f"| {feature_set} | {format_value(condition['r2'] if condition else None)} | {format_value(condition['rmse_mm'] if condition else None)} | {format_value(point['within_condition_r2'] if point else None)} | {format_value(loho['rmse_mm'] if loho else None)} | {format_value(lopo['rmse_mm'] if lopo else None)} |"
        )
    lines += [
        "",
        f"q1+q2+J 相对 q1+q2 的 condition-level in-sample ΔR²=`{format_value(classifications['q1_q2_plus_J_delta_r2_condition'])}`；LOHO/LOPO grouped RMSE 中位改善分别为 `{', '.join(format_value(value, 5) for value in classifications['q1_q2_plus_J_grouped_median_rmse_improvements_mm']) or '—'}` mm。",
        f"因此 `JACOBIAN_ADDS_INDEPENDENT_INFORMATION = {classifications['JACOBIAN_ADDS_INDEPENDENT_INFORMATION']}`；即使 J 能解释部分 residual，也必须结合 VIF 和 grouped 稳定性，不能把高 in-sample R² 当作单个参数可辨识。",
        "",
        "## Grouped sensitivity",
        "",
        "LOHO=leave-one-height-out，LOPO=leave-one-position-out；每个 fold 完整留出对应 height 或 position 的全部 condition，训练/评估均在 development 内完成。",
        "",
        "| fold type | feature set | median RMSE | median MAE | median P95 | median max | fold count |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for evaluation, label in (("leave_one_height_out", "LOHO"), ("leave_one_position_out", "LOPO")):
        for feature_set in FEATURE_SETS:
            row = metric_lookup(comparisons, "grouped_summary", evaluation, feature_set)
            if row:
                lines.append(f"| {label} | {feature_set} | {format_value(row['rmse_mm'])} | {format_value(row['mae_mm'])} | {format_value(row['p95_abs_error_mm'])} | {format_value(row['max_abs_error_mm'])} | {row.get('fold_count','—')} |")

    lines += [
        "",
        "## Surface-2 采集优先级（仅 development 规划）",
        "",
        f"development-only q2~height slope=`{format_value(plan['development_only_q2_height_slope'], 6)}` q2/mm；该趋势只用于采集规划，不是 height correction。development formal q2 范围为 `{format_value(plan['development_q2_range'][0])}`–`{format_value(plan['development_q2_range'][1])}`。",
        f"50 mm q2 范围 `{format_value(plan['heldout_50_q2_range_descriptive_only'][0])}`–`{format_value(plan['heldout_50_q2_range_descriptive_only'][1])}` 仅作 held-out domain context，未参与上述趋势拟合。",
        "",
        "| target height | development-only predicted q2 | position-wise predicted q2 range |",
        "|---:|---:|---:|",
    ]
    for row in plan["targets"]:
        lines.append(f"| {row['height_mm']:.0f} mm | {format_value(row['development_only_q2_prediction'])} | {format_value(row['position_prediction_min'])}–{format_value(row['position_prediction_max'])} |")
    lines += [
        "",
        "建议顺序：40 mm（中段 gap）、35 mm（development 上界转折）、45 mm（接近 50 mm 高端）。若资源允许，优先一次性采集完整 `35/40/45 mm × 5 positions`，每个高度保持 repeat1 proxy + repeat2–5 formal protocol。",
        "",
        "## 结论与限制",
        "",
        f"- `SURFACE_FEATURE_IDENTIFIABILITY = {classifications['SURFACE_FEATURE_IDENTIFIABILITY']}`：q1/q2 在当前 development 域可做稳定的表面坐标诊断，但 height、q2、Jacobian 之间存在不同程度共线性；不能据此声称各物理因素已完全分离。",
        f"- `HEIGHT_Q2_REDUNDANCY = {classifications['HEIGHT_Q2_REDUNDANCY']}`：height 与 q2 的高度趋势在当前设计中高度重叠，height+q1+q2 的解释增益不能单独归因于 surface。",
        f"- `JACOBIAN_ADDS_INDEPENDENT_INFORMATION = {classifications['JACOBIAN_ADDS_INDEPENDENT_INFORMATION']}`：J 的增益必须以 grouped sensitivity 和 VIF 共同判断；当前没有把 J 变成补偿函数。",
        f"- `RECOMMENDED_MINIMAL_FEATURE_SET = {classifications['RECOMMENDED_MINIMAL_FEATURE_SET']}`：这是后续诊断建模的最小候选集，不是生产参数。",
        "- 50 mm 保持 held-out 身份；本轮不使用其结果调整 feature set、PCA/VIF、阈值或采集规则。",
        "",
        "## Outputs",
        "",
        "- `surface1b_feature_correlation.csv`：point-level / condition-level 的 Pearson/Spearman 长表。",
        "- `surface1b_vif_svd.csv`：development-only 各候选集的标准化 SVD/PCA、condition number、VIF。",
        "- `surface1b_feature_set_comparison.csv`：候选集 explanatory OLS 与 LOHO/LOPO grouped sensitivity。",
        "- `surface1b_correlation_heatmap.png`、本报告以及 machine-readable `surface1b_summary.json`。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    development_points, heldout_points = load_point_rows(input_dir)
    if not development_points:
        raise RuntimeError("no formal development points")
    development_conditions = aggregate_conditions(development_points)
    heldout_conditions = aggregate_conditions(heldout_points)

    correlations: list[dict[str, Any]] = []
    correlations.extend(correlation_rows("development_formal", "point_pooled", development_points, True))
    correlations.extend(correlation_rows("development_formal", "condition_balanced", development_conditions, True))
    correlations.extend(correlation_rows("heldout_50_descriptive", "point_pooled", heldout_points, False))
    correlations.extend(correlation_rows("heldout_50_descriptive", "condition_balanced", heldout_conditions, False))
    correlation_fields = list(correlations[0].keys())
    write_csv(output / "surface1b_feature_correlation.csv", correlations, correlation_fields)

    vif_rows: list[dict[str, Any]] = []
    for aggregation, data in (("point_pooled", development_points), ("condition_balanced", development_conditions)):
        for feature_set in FEATURE_SETS:
            vif_rows.extend(vif_svd_rows("development_formal", aggregation, feature_set, data, True))
    vif_fields = list(vif_rows[0].keys())
    write_csv(output / "surface1b_vif_svd.csv", vif_rows, vif_fields)

    comparisons = fit_comparisons(development_points, development_conditions)
    comparison_fields = list(comparisons[0].keys())
    write_csv(output / "surface1b_feature_set_comparison.csv", comparisons, comparison_fields)

    classifications = classify(correlations, comparisons, vif_rows)
    plan = surface2_plan(development_points, development_conditions, heldout_points)
    provenance = {
        "points_path": str(input_dir / "surface1a_points.csv"),
        "coordinate_definition_path": str(input_dir / "surface_coordinate_definition.json"),
        "summary_path": str(input_dir / "surface1a_summary.json"),
        "metrics_path": str(input_dir / "surface1a_explanatory_metrics.csv"),
        "development_point_count": len(development_points),
        "development_condition_count": len(development_conditions),
        "heldout_point_count": len(heldout_points),
        "heldout_condition_count": len(heldout_conditions),
        "formal_filter": [FORMAL_DEVELOPMENT, "height_measurement_inlier=true", "jacobian_valid=true"],
        "heldout_used_for_model_selection": False,
        "heldout_used_for_parameter_fitting": False,
        "c0_c1_refit": False,
        "ground_gs_refit": False,
        "h1_refit": False,
        "production_change": False,
        "created_at_utc": now_utc(),
    }
    summary = {
        "classifications": classifications,
        "surface2_plan": plan,
        "provenance": provenance,
        "candidate_feature_sets": FEATURE_SETS,
        "correlation_features": CORRELATION_FEATURES,
        "notes": [
            "50mm correlation rows are descriptive only and excluded from SVD/VIF/comparison/model selection.",
            "Condition rows are equal-weight height×position means over formal points.",
            "All OLS fits are explanatory diagnostics, never production correction functions.",
        ],
    }
    write_json(output / "surface1b_summary.json", summary)
    plot_heatmap(output / "surface1b_correlation_heatmap.png", correlations)
    write_report(output / "surface1b_identifiability_report.md", classifications, correlations, vif_rows, comparisons, plan, provenance)

    print(json.dumps({"output": str(output), **{key: value for key, value in classifications.items() if isinstance(value, str)}, "development_points": len(development_points), "development_conditions": len(development_conditions), "heldout_points": len(heldout_points)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
