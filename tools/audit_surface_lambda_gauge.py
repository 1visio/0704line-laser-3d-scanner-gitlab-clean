"""Audit the lambda intercept gauge with an anchored q2-only replay.

L0-B2 fixes delta_lambda(q2=0)=0 and reuses the exact reconstruction and
session-proxy path from compare_surface_correction_layers.py.  The previous
free-intercept L-B2 and H-B2 outputs are read as frozen comparison artifacts;
they are never overwritten.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compare_surface_correction_layers as base
from measurement.height_measure import MeasurementParams


BASE_OUTPUT = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
PREVIOUS_OUTPUT = BASE_OUTPUT / "surface2" / "correction_layer"
DEFAULT_OUTPUT = BASE_OUTPUT / "surface2" / "lambda_gauge_audit"
DEFAULT_CONFIG = ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
DEV_SCHEMES = ("LOHO_height", "LOPO_position_rank", "LOBO_height_band")
ALL_SCHEMES = DEV_SCHEMES + ("strict_50mm_validation",)
LAYERS = ("RAW", "H-B2", "L-B2", "L0-B2")
EPS = 1.0e-12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def f(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite value: {value}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def set_measurement_params(inputs: dict[str, Any]) -> None:
    params = inputs["app"].measurement
    inputs["measurement_params"] = params if isinstance(params, MeasurementParams) else MeasurementParams()


def fit_lambda0_model(
    conditions: list[base.Condition],
    inputs: dict[str, Any],
    max_iterations: int = 10,
) -> dict[str, Any]:
    """Fit only b2 while keeping b0 exactly zero."""
    beta = np.zeros(2, dtype=np.float64)
    residual, evaluations, valid = base.lambda_residual_vector(conditions, beta, inputs)
    if not valid or not len(residual):
        return {
            "layer": "L0-B2",
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
        trial = beta.copy()
        trial[1] += 1.0e-4
        trial_residual, _, trial_valid = base.lambda_residual_vector(
            conditions, trial, inputs
        )
        if not trial_valid or len(trial_residual) != len(residual):
            status = "finite_difference_invalid"
            break
        jacobian = ((trial_residual - residual) / 1.0e-4).reshape(-1, 1)
        step_array, _, rank_value, singular = np.linalg.lstsq(
            jacobian, -residual, rcond=None
        )
        rank = int(rank_value)
        condition_number = float(np.linalg.cond(jacobian))
        step = float(step_array[0])
        if not math.isfinite(step):
            status = "nonfinite_step"
            break
        if abs(step) < 1.0e-8:
            break
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = beta.copy()
            candidate[1] += scale * step
            candidate_residual, candidate_evaluations, candidate_valid = base.lambda_residual_vector(
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
    if rank < 1:
        status = "rank_deficient" if status == "success" else status
    return {
        "layer": "L0-B2",
        "beta": beta,
        "fit_status": status,
        "design_rank": rank,
        "design_condition_number": condition_number,
        "singular_values": singular,
        "iterations": iterations,
        "objective": objective,
        "evaluations": evaluations,
    }


def condition_row(
    scheme: str,
    group: str,
    condition: base.Condition,
    support_state: str,
    layer: str,
    values: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    valid_values = values[np.isfinite(values)]
    return {
        "cv_scheme": scheme,
        "heldout_group": group,
        "aggregation": "fold",
        "layer": layer,
        "condition_id": condition.condition_id,
        "dataset": condition.dataset,
        "nominal_height_mm": condition.nominal_height_mm,
        "true_height_mm": condition.true_height_mm,
        "position_rank": condition.position_rank,
        "point_count": int(len(valid_values)),
        "invalid_point_count": int(len(values) - len(valid_values)),
        "support_state": support_state,
        "condition_bias_mm": float(np.mean(valid_values)) if len(valid_values) else float("nan"),
        "condition_mae_mm": float(np.mean(np.abs(valid_values))) if len(valid_values) else float("nan"),
        "condition_rmse_mm": float(np.sqrt(np.mean(valid_values * valid_values))) if len(valid_values) else float("nan"),
        "condition_p95_abs_mm": float(np.percentile(np.abs(valid_values), 95.0)) if len(valid_values) else float("nan"),
        "condition_max_abs_mm": float(np.max(np.abs(valid_values))) if len(valid_values) else float("nan"),
        "condition_raw_bias_mm": float(np.mean(condition.formal.raw_residual)),
    }


def proxy_audit_row(
    scheme: str,
    group: str,
    condition: base.Condition,
    evaluation: dict[str, Any],
    fit: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    baseline = evaluation.get("baseline", {})
    proxy = evaluation.get("proxy")
    formal_delta = base.transform_lambda(
        condition.formal,
        fit["beta"],
        inputs["app"].reconstruction,
        inputs["transform"],
    )
    formal = evaluation.get("formal", {})
    return {
        "cv_scheme": scheme,
        "heldout_group": group,
        "condition_id": condition.condition_id,
        "dataset": condition.dataset,
        "nominal_height_mm": condition.nominal_height_mm,
        "position_rank": condition.position_rank,
        "layer": "L0-B2",
        "raw_proxy_a_mm_per_mm": float(condition.raw_proxy.slope),
        "raw_proxy_b_mm": float(condition.raw_proxy.intercept),
        "corrected_proxy_a_mm_per_mm": None if proxy is None else float(proxy.slope),
        "corrected_proxy_b_mm": None if proxy is None else float(proxy.intercept),
        "delta_proxy_a_mm_per_mm": None if proxy is None else float(proxy.slope - condition.raw_proxy.slope),
        "delta_proxy_b_mm": None if proxy is None else float(proxy.intercept - condition.raw_proxy.intercept),
        "raw_proxy_rmse_mm": float(condition.raw_proxy.rmse),
        "corrected_proxy_rmse_mm": None if proxy is None else float(proxy.rmse),
        "baseline_point_count": int(condition.baseline.source_count),
        "baseline_invalid_point_count": int(baseline.get("invalid_count", 0)),
        "formal_point_count": int(condition.formal.source_count),
        "formal_invalid_point_count": int(evaluation.get("invalid_count", 0)),
        "formal_valid_point_count": int(np.count_nonzero(formal.get("valid", np.zeros(condition.formal.source_count, dtype=bool)))),
        "raw_baseline_c1_clamp_count": int(np.count_nonzero(condition.baseline.c1_clamped)),
        "corrected_baseline_c1_clamp_count": int(np.count_nonzero(condition.baseline.c1_clamped)),
        "formal_c1_clamp_count": int(np.count_nonzero(condition.formal.c1_clamped)),
        "corrected_formal_c1_clamp_count": int(np.count_nonzero(condition.formal.c1_clamped)),
        "lambda_delta_baseline_p95_abs_mm": float(np.percentile(np.abs(baseline["delta_lambda"]), 95.0)),
        "lambda_delta_formal_p95_abs_mm": float(np.percentile(np.abs(formal_delta["delta_lambda"]), 95.0)),
        "delta_Zg_baseline_p95_abs_mm": float(np.percentile(np.abs(baseline["delta_Zg"]), 95.0)),
        "delta_Zg_formal_p95_abs_mm": float(np.percentile(np.abs(formal_delta["delta_Zg"]), 95.0)),
        "fit_status": fit["fit_status"],
        "evaluation_error": evaluation.get("error", ""),
    }


def safe_lambda_evaluation(
    condition: base.Condition,
    beta: np.ndarray,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    try:
        return base.evaluate_lambda_condition(condition, beta, inputs)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        params = inputs["app"].reconstruction
        baseline = base.transform_lambda(
            condition.baseline,
            beta,
            params,
            inputs["transform"],
        )
        formal = base.transform_lambda(
            condition.formal,
            beta,
            params,
            inputs["transform"],
        )
        return {
            "condition_id": condition.condition_id,
            "proxy": None,
            "baseline": baseline,
            "formal": formal,
            "valid": False,
            "residual": np.asarray([], dtype=np.float64),
            "invalid_count": int(formal["invalid_count"]),
            "error": f"{type(error).__name__}: {error}",
        }


def run_l0_fold(
    scheme: str,
    group: str,
    train: list[base.Condition],
    test: list[base.Condition],
    inputs: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    fit = fit_lambda0_model(train, inputs)
    condition_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    coefficient_rows = [
        {
            "cv_scheme": scheme,
            "heldout_group": group,
            "layer": "L0-B2",
            "parameter": "b0",
            "value": 0.0,
            "fixed": True,
            "fit_status": fit["fit_status"],
            "design_rank": fit["design_rank"],
            "design_condition_number": fit["design_condition_number"],
            "iterations": fit["iterations"],
            "objective": fit["objective"],
        },
        {
            "cv_scheme": scheme,
            "heldout_group": group,
            "layer": "L0-B2",
            "parameter": "b2",
            "value": float(fit["beta"][1]),
            "fixed": False,
            "fit_status": fit["fit_status"],
            "design_rank": fit["design_rank"],
            "design_condition_number": fit["design_condition_number"],
            "iterations": fit["iterations"],
            "objective": fit["objective"],
        },
    ]
    proxy_rows: list[dict[str, Any]] = []
    magnitude_rows: list[dict[str, Any]] = []
    invalid_formal_total = 0
    invalid_baseline_total = 0
    for condition in test:
        support_state = base.support_from_q(train, condition)["state"]
        evaluation = safe_lambda_evaluation(condition, fit["beta"], inputs)
        invalid_formal_total += int(evaluation["invalid_count"])
        invalid_baseline_total += int(evaluation.get("baseline", {}).get("invalid_count", 0))
        residual = np.full(condition.formal.source_count, np.nan, dtype=np.float64)
        if len(evaluation["residual"]) == condition.formal.source_count:
            residual[:] = evaluation["residual"]
        condition_rows.append(condition_row(scheme, group, condition, support_state, "L0-B2", residual))
        proxy_rows.append(proxy_audit_row(scheme, group, condition, evaluation, fit, inputs))
    metric_rows = [
        base.layer_metric_row(
            scheme,
            group,
            "fold",
            "L0-B2",
            condition_rows,
            invalid_formal_total,
        )
    ]
    for state in ("IN_DOMAIN", "HULL_EXTRAPOLATION", "BBOX_EXTRAPOLATION"):
        if any(row["support_state"] == state for row in condition_rows):
            support_rows.append(
                base.support_row(scheme, group, "fold", "L0-B2", state, condition_rows)
            )
    base.append_magnitude_rows(
        magnitude_rows,
        scheme,
        group,
        "L0-B2",
        fit["beta"],
        test,
        inputs,
    )
    return metric_rows, condition_rows, support_rows, coefficient_rows, magnitude_rows, proxy_rows, {
        "fit": fit,
        "invalid_formal_total": invalid_formal_total,
        "invalid_baseline_total": invalid_baseline_total,
    }


def pool_l0(
    metric_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pooled_metrics: list[dict[str, Any]] = []
    pooled_support: list[dict[str, Any]] = []
    for scheme in ALL_SCHEMES:
        selected = [
            row for row in condition_rows
            if row["cv_scheme"] == scheme and row["layer"] == "L0-B2"
        ]
        if not selected:
            continue
        pooled_metrics.append(
            base.layer_metric_row(
                scheme,
                "ALL_FOLDS",
                "pooled_condition_means",
                "L0-B2",
                selected,
                sum(int(row.get("invalid_point_count", 0)) for row in selected),
            )
        )
        for state in ("IN_DOMAIN", "HULL_EXTRAPOLATION", "BBOX_EXTRAPOLATION"):
            if any(row["support_state"] == state for row in selected):
                pooled_support.append(
                    base.support_row(
                        scheme,
                        "ALL_FOLDS",
                        "pooled_condition_means",
                        "L0-B2",
                        state,
                        selected,
                    )
                )
    return pooled_metrics, pooled_support


def strict_dev_fold_definitions(
    conditions: dict[str, base.Condition],
) -> list[tuple[str, str, list[base.Condition], list[base.Condition]]]:
    """Build the current protocol with 50 mm absent from every development fit."""
    values = list(conditions.values())
    development = [condition for condition in values if condition.nominal_height_mm != 50.0]
    heldout_50 = [condition for condition in values if condition.nominal_height_mm == 50.0]
    folds: list[tuple[str, str, list[base.Condition], list[base.Condition]]] = []
    for height in base.HEIGHT_ORDER:
        train = [condition for condition in development if condition.nominal_height_mm != height]
        test = [condition for condition in development if condition.nominal_height_mm == height]
        folds.append(("LOHO_height", base.height_group(height), train, test))
    for rank in range(1, 6):
        train = [condition for condition in development if condition.position_rank != rank]
        test = [condition for condition in development if condition.position_rank == rank]
        if test:
            folds.append(("LOPO_position_rank", f"rank_{rank}", train, test))
    for band, heights in base.HEIGHT_BANDS.items():
        train = [condition for condition in development if condition.nominal_height_mm not in heights]
        test = [condition for condition in development if condition.nominal_height_mm in heights]
        folds.append(("LOBO_height_band", band, train, test))
    folds.append((
        "strict_50mm_validation",
        "height_50mm_strict_heldout",
        development,
        heldout_50,
    ))
    return folds


def metric_number(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    return float(value) if finite(value) else float("nan")


def metric_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            row["cv_scheme"],
            row["heldout_group"],
            row["aggregation"],
            row["layer"],
        ): row
        for row in rows
    }


def prior_artifacts() -> dict[str, Any]:
    paths = {
        "summary": PREVIOUS_OUTPUT / "surface2_correction_layer_summary.json",
        "report": PREVIOUS_OUTPUT / "surface2_correction_layer_report.md",
        "cv_metrics": PREVIOUS_OUTPUT / "surface2_correction_layer_cv_metrics.csv",
        "condition_metrics": PREVIOUS_OUTPUT / "surface2_correction_layer_condition_metrics.csv",
        "support_metrics": PREVIOUS_OUTPUT / "surface2_correction_layer_support_metrics.csv",
        "coefficients": PREVIOUS_OUTPUT / "surface2_correction_layer_coefficients.csv",
        "coefficient_stability": PREVIOUS_OUTPUT / "surface2_correction_layer_coefficient_stability.csv",
        "magnitude": PREVIOUS_OUTPUT / "surface2_correction_layer_magnitude.csv",
        "fold_audit": PREVIOUS_OUTPUT / "surface2_correction_layer_fold_audit.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen previous result: " + ", ".join(missing))
    return {
        "paths": paths,
        "summary": read_json(paths["summary"]),
        "report_sha256": sha256(paths["report"]),
        "files_sha256": {name: sha256(path) for name, path in paths.items()},
        "cv_metrics": read_csv(paths["cv_metrics"]),
        "condition_metrics": read_csv(paths["condition_metrics"]),
        "support_metrics": read_csv(paths["support_metrics"]),
        "coefficients": read_csv(paths["coefficients"]),
        "coefficient_stability": read_csv(paths["coefficient_stability"]),
        "magnitude": read_csv(paths["magnitude"]),
        "fold_audit": read_csv(paths["fold_audit"]),
    }


def add_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [{**row, "source": source} for row in rows]


def previous_fold_protocol_audit(
    prior: dict[str, Any],
    current_config_sha256: str,
) -> dict[str, Any]:
    rows = prior["fold_audit"]
    development_rows = [
        row for row in rows
        if row.get("cv_scheme") in DEV_SCHEMES
    ]
    train_counts = [int(row["train_condition_count"]) for row in development_rows]
    prior_config_sha256 = (
        prior["summary"].get("provenance", {})
        .get("input_sha256", {})
        .get("config", "")
    )
    return {
        "previous_dev_fold_count": len(development_rows),
        "previous_dev_train_condition_counts": sorted(set(train_counts)),
        "previous_artifact_train_includes_50_for_dev_folds": bool(train_counts and max(train_counts) > 39),
        "previous_config_sha256": prior_config_sha256,
        "current_config_sha256": current_config_sha256,
        "config_sha256_matches_previous_artifact": bool(
            prior_config_sha256 and prior_config_sha256 == current_config_sha256
        ),
        "note": (
            "The frozen prior fold-audit records 44/45/39/40 training conditions "
            "for development folds, while strict 50-excluded folds should train "
            "on 39/40/35/36 conditions depending on the held-out group. "
            "Therefore this audit recomputes H-B2 and free L-B2 under the "
            "current strict-50-excluded protocol without modifying the prior files."
        ),
    }


def make_comparison_rows(
    current_metric_rows: list[dict[str, Any]],
    prior_metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current = metric_map(current_metric_rows)
    previous = metric_map(prior_metric_rows)
    keys = sorted({
        key[:3]
        for key in current
        if key[3] in {"H-B2", "L-B2", "L0-B2"}
    } | {
        key[:3]
        for key in previous
        if key[3] in {"H-B2", "L-B2"}
    })
    rows: list[dict[str, Any]] = []
    for scheme, group, aggregation in keys:
        c_h = current.get((scheme, group, aggregation, "H-B2"))
        c_l = current.get((scheme, group, aggregation, "L-B2"))
        c_l0 = current.get((scheme, group, aggregation, "L0-B2"))
        p_h = previous.get((scheme, group, aggregation, "H-B2"))
        p_l = previous.get((scheme, group, aggregation, "L-B2"))
        if c_h is None or c_l is None or c_l0 is None:
            continue
        row: dict[str, Any] = {
            "cv_scheme": scheme,
            "heldout_group": group,
            "aggregation": aggregation,
            "current_protocol": "strict_50mm_excluded_from_development_fits",
            "current_h_condition_count": c_h.get("condition_count"),
            "current_l_condition_count": c_l.get("condition_count"),
            "current_l0_condition_count": c_l0.get("condition_count"),
        }
        for prefix, source_row in (
            ("current_H_B2", c_h),
            ("current_L_B2", c_l),
            ("current_L0_B2", c_l0),
        ):
            for field in ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm", "worst_condition_abs_mm"):
                row[f"{prefix}_{field}"] = metric_number(source_row, field)
        for prefix, source_row in (("previous_H_B2", p_h), ("previous_L_B2", p_l)):
            for field in ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm", "worst_condition_abs_mm"):
                row[f"{prefix}_{field}"] = None if source_row is None else metric_number(source_row, field)
        row.update({
            "delta_L0_minus_H_rmse_mm": row["current_L0_B2_rmse_mm"] - row["current_H_B2_rmse_mm"],
            "delta_L0_minus_H_p95_mm": row["current_L0_B2_p95_abs_mm"] - row["current_H_B2_p95_abs_mm"],
            "delta_L0_minus_H_worst_mm": row["current_L0_B2_worst_condition_abs_mm"] - row["current_H_B2_worst_condition_abs_mm"],
            "delta_L0_minus_free_L_rmse_mm": row["current_L0_B2_rmse_mm"] - row["current_L_B2_rmse_mm"],
            "delta_L0_minus_free_L_p95_mm": row["current_L0_B2_p95_abs_mm"] - row["current_L_B2_p95_abs_mm"],
            "delta_L0_minus_free_L_worst_mm": row["current_L0_B2_worst_condition_abs_mm"] - row["current_L_B2_worst_condition_abs_mm"],
        })
        row["L0_not_worse_than_current_H_all_three"] = bool(
            row["current_L0_B2_rmse_mm"] <= row["current_H_B2_rmse_mm"] + EPS
            and row["current_L0_B2_p95_abs_mm"] <= row["current_H_B2_p95_abs_mm"] + EPS
            and row["current_L0_B2_worst_condition_abs_mm"] <= row["current_H_B2_worst_condition_abs_mm"] + EPS
        )
        rows.append(row)
    return rows


def make_support_comparison(
    current_support_rows: list[dict[str, Any]],
    prior_support_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current = {
        (
            row["cv_scheme"],
            row["heldout_group"],
            row["aggregation"],
            row["layer"],
            row["support_state"],
        ): row
        for row in current_support_rows
    }
    previous = {
        (
            row["cv_scheme"],
            row["heldout_group"],
            row["aggregation"],
            row["layer"],
            row["support_state"],
        ): row
        for row in prior_support_rows
    }
    rows: list[dict[str, Any]] = []
    for key, c_row in sorted(current.items()):
        scheme, group, aggregation, layer, state = key
        row = {
            "cv_scheme": scheme,
            "heldout_group": group,
            "aggregation": aggregation,
            "support_state": state,
            "layer": layer,
            "current_protocol": "strict_50mm_excluded_from_development_fits",
            "current_condition_count": c_row.get("condition_count"),
            "current_rmse_mm": metric_number(c_row, "rmse_mm"),
            "current_p95_abs_mm": metric_number(c_row, "p95_abs_mm"),
            "current_worst_condition_abs_mm": metric_number(c_row, "worst_condition_abs_mm"),
        }
        for prefix, p_layer in (("previous_H_B2", "H-B2"), ("previous_L_B2", "L-B2")):
            p_row = previous.get((scheme, group, aggregation, p_layer, state))
            row[f"{prefix}_condition_count"] = None if p_row is None else p_row.get("condition_count")
            row[f"{prefix}_rmse_mm"] = None if p_row is None else metric_number(p_row, "rmse_mm")
            row[f"{prefix}_p95_abs_mm"] = None if p_row is None else metric_number(p_row, "p95_abs_mm")
            row[f"{prefix}_worst_condition_abs_mm"] = None if p_row is None else metric_number(p_row, "worst_condition_abs_mm")
        rows.append(row)
    return rows


def summarize_stability(
    coefficient_rows: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    parameter_map = {
        "H-B2": ("a0", "a2"),
        "L-B2": ("b0", "b2"),
        "L0-B2": ("b0", "b2"),
    }
    for layer, parameters in parameter_map.items():
        for parameter in parameters:
            values = [
                float(row["value"])
                for row in coefficient_rows
                if row.get("layer") == layer
                and row.get("parameter") == parameter
                and row.get("cv_scheme") in DEV_SCHEMES
                and finite(row.get("value"))
            ]
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            mean = float(np.mean(array))
            value_range = float(np.max(array) - np.min(array))
            result.append({
                "source": source,
                "scope": "development_folds",
                "layer": layer,
                "parameter": parameter,
                "fold_count": len(values),
                "mean": mean,
                "median": float(np.median(array)),
                "std": float(np.std(array, ddof=0)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "range": value_range,
                "relative_range_to_abs_mean": value_range / max(abs(mean), EPS),
                "sign_consistency": bool(np.all(array >= 0.0) or np.all(array <= 0.0)),
            })
    return result


def prior_stability_rows(prior: dict[str, Any]) -> list[dict[str, Any]]:
    return add_source(prior["coefficient_stability"], "previous_frozen_artifact")


def physical_audit_l0(
    current_metric_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    magnitude_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fold_metrics = [
        row for row in current_metric_rows
        if row.get("aggregation") == "fold"
        and row.get("layer") == "L0-B2"
    ]
    dev_metrics = [row for row in fold_metrics if row.get("cv_scheme") in DEV_SCHEMES]
    invalid_formal = sum(int(row.get("invalid_point_count", 0)) for row in fold_metrics)
    invalid_formal_dev = sum(int(row.get("invalid_point_count", 0)) for row in dev_metrics)
    baseline_invalid = sum(
        int(row.get("baseline_invalid_point_count", 0))
        for row in proxy_rows
    )
    baseline_invalid_dev = sum(
        int(row.get("baseline_invalid_point_count", 0))
        for row in proxy_rows
        if row.get("cv_scheme") in DEV_SCHEMES
    )
    lambda_rows = [
        row for row in magnitude_rows
        if row.get("layer") == "L0-B2"
        and row.get("scope") in {"formal", "baseline"}
    ]
    dev_lambda_rows = [row for row in lambda_rows if row.get("cv_scheme") in DEV_SCHEMES]
    def maximum(rows: list[dict[str, Any]], field: str) -> float:
        values = [float(row[field]) for row in rows if finite(row.get(field))]
        return max(values, default=float("nan"))
    def maximum_abs(rows: list[dict[str, Any]], field: str) -> float:
        values = [abs(float(row[field])) for row in rows if finite(row.get(field))]
        return max(values, default=float("nan"))
    l0_stability = [
        row for row in stability_rows
        if row.get("source") == "anchored_l0_replay"
        and row.get("layer") == "L0-B2"
        and row.get("parameter") == "b2"
    ]
    relative_range = max(
        (float(row["relative_range_to_abs_mean"]) for row in l0_stability if finite(row.get("relative_range_to_abs_mean"))),
        default=float("inf"),
    )
    b2_sign_consistent = bool(l0_stability and all(bool(row.get("sign_consistency")) for row in l0_stability))
    c1_changes = sum(
        abs(int(row.get("raw_baseline_c1_clamp_count", 0)) - int(row.get("corrected_baseline_c1_clamp_count", 0)))
        + abs(int(row.get("formal_c1_clamp_count", 0)) - int(row.get("corrected_formal_c1_clamp_count", 0)))
        for row in proxy_rows
    )
    failed_fits = [
        row for row in coefficient_rows
        if row.get("layer") == "L0-B2"
        and row.get("parameter") == "b2"
        and row.get("fit_status") not in {"success", "rank_deficient"}
    ]
    max_abs_lambda = maximum(lambda_rows, "abs_delta_lambda_mm_max")
    max_p95_abs_lambda = maximum(lambda_rows, "abs_delta_lambda_mm_p95")
    max_abs_zg = maximum(lambda_rows, "abs_delta_Zg_mm_max")
    dev_max_abs_lambda = maximum(dev_lambda_rows, "abs_delta_lambda_mm_max")
    dev_max_p95_abs_lambda = maximum(dev_lambda_rows, "abs_delta_lambda_mm_p95")
    dev_max_abs_zg = maximum(dev_lambda_rows, "abs_delta_Zg_mm_max")
    max_abs_proxy_slope_change = maximum_abs(proxy_rows, "delta_proxy_a_mm_per_mm")
    max_abs_proxy_intercept_change = maximum_abs(proxy_rows, "delta_proxy_b_mm")
    max_abs_proxy_rmse = maximum(proxy_rows, "corrected_proxy_rmse_mm")
    thresholds = {
        "max_abs_delta_lambda_mm": 2.0,
        "max_p95_abs_delta_lambda_mm": 1.0,
        "max_abs_delta_Zg_mm": 2.0,
        "max_coefficient_relative_range": 1.0,
    }
    dev_supported = bool(
        not failed_fits
        and invalid_formal_dev == 0
        and baseline_invalid_dev == 0
        and c1_changes == 0
        and finite(dev_max_abs_lambda)
        and finite(dev_max_p95_abs_lambda)
        and finite(dev_max_abs_zg)
        and dev_max_abs_lambda <= thresholds["max_abs_delta_lambda_mm"]
        and dev_max_p95_abs_lambda <= thresholds["max_p95_abs_delta_lambda_mm"]
        and dev_max_abs_zg <= thresholds["max_abs_delta_Zg_mm"]
        and relative_range <= thresholds["max_coefficient_relative_range"]
        and b2_sign_consistent
    )
    overall_supported = bool(
        dev_supported
        and invalid_formal == 0
        and baseline_invalid == 0
    )
    return {
        "fit_failure": bool(failed_fits),
        "new_invalid_point_count": int(invalid_formal + baseline_invalid),
        "formal_new_invalid_point_count": int(invalid_formal),
        "baseline_new_invalid_point_count": int(baseline_invalid),
        "development_new_invalid_point_count": int(invalid_formal_dev + baseline_invalid_dev),
        "development_formal_new_invalid_point_count": int(invalid_formal_dev),
        "development_baseline_new_invalid_point_count": int(baseline_invalid_dev),
        "c1_clamp_state_change_count": int(c1_changes),
        "max_abs_delta_lambda_mm": max_abs_lambda,
        "max_p95_abs_delta_lambda_mm": max_p95_abs_lambda,
        "max_abs_delta_Zg_mm": max_abs_zg,
        "development_max_abs_delta_lambda_mm": dev_max_abs_lambda,
        "development_max_p95_abs_delta_lambda_mm": dev_max_p95_abs_lambda,
        "development_max_abs_delta_Zg_mm": dev_max_abs_zg,
        "max_abs_corrected_proxy_slope_change_mm_per_mm": max_abs_proxy_slope_change,
        "max_abs_corrected_proxy_intercept_change_mm": max_abs_proxy_intercept_change,
        "max_corrected_proxy_rmse_mm": max_abs_proxy_rmse,
        "lambda_coefficient_max_relative_range": relative_range,
        "lambda_coefficient_b2_sign_consistent": b2_sign_consistent,
        "thresholds": thresholds,
        "L0_B2_DEVELOPMENT_PHYSICAL_STATUS": "SUPPORTED" if dev_supported else "NOT_SUPPORTED",
        "L0_B2_PHYSICAL_STATUS": "SUPPORTED" if overall_supported else "NOT_SUPPORTED",
    }


def pooled_map(
    metric_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["cv_scheme"], row["layer"]): row
        for row in metric_rows
        if row.get("aggregation") == "pooled_condition_means"
    }


def decision_from_results(
    current_metric_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    physical: dict[str, Any],
    prior_summary: dict[str, Any],
    gauge_supported: bool,
) -> dict[str, Any]:
    pooled = pooled_map(current_metric_rows)
    h_improves_raw = all(
        (scheme, "H-B2") in pooled
        and (scheme, "RAW") in pooled
        and pooled[(scheme, "H-B2")]["rmse_mm"] < pooled[(scheme, "RAW")]["rmse_mm"] - EPS
        and pooled[(scheme, "H-B2")]["p95_abs_mm"] < pooled[(scheme, "RAW")]["p95_abs_mm"] - EPS
        for scheme in DEV_SCHEMES
    )
    l0_improves_raw = all(
        (scheme, "L0-B2") in pooled
        and (scheme, "RAW") in pooled
        and pooled[(scheme, "L0-B2")]["rmse_mm"] < pooled[(scheme, "RAW")]["rmse_mm"] - EPS
        and pooled[(scheme, "L0-B2")]["p95_abs_mm"] < pooled[(scheme, "RAW")]["p95_abs_mm"] - EPS
        for scheme in DEV_SCHEMES
    )
    current_dev_comparison = [
        row for row in comparison_rows
        if row["aggregation"] == "pooled_condition_means"
        and row["cv_scheme"] in DEV_SCHEMES
    ]
    l0_noninferior = bool(current_dev_comparison) and all(
        bool(row["L0_not_worse_than_current_H_all_three"])
        for row in current_dev_comparison
    )
    anchored_validity = (
        "SUPPORTED"
        if l0_noninferior and physical["L0_B2_DEVELOPMENT_PHYSICAL_STATUS"] == "SUPPORTED"
        else "PARTIAL"
        if l0_noninferior
        else "NOT_SUPPORTED"
    )
    if anchored_validity == "SUPPORTED":
        final_layer = "LAMBDA"
    elif h_improves_raw:
        final_layer = "HEIGHT"
    else:
        final_layer = "UNDECIDED"
    more_data = (
        "YES"
        if prior_summary.get("decision", {}).get("MORE_DATA_REQUIRED_BEFORE_NEXT_ANALYSIS") == "YES"
        else "NO"
    )
    return {
        "LAMBDA_INTERCEPT_IDENTIFIABILITY": "SUPPORTED" if gauge_supported else "WEAK",
        "ANCHORED_LAMBDA_LAYER_VALIDITY": anchored_validity,
        "FINAL_CORRECTION_LAYER": final_layer,
        "STOP_LAMBDA_ROUTE": "NO" if anchored_validity == "SUPPORTED" else "YES",
        "L0_B2_IMPROVES_RAW_ALL_DEVELOPMENT_SCHEMES": l0_improves_raw,
        "L0_B2_NONINFERIOR_TO_H_B2_ALL_DEVELOPMENT_SCHEMES": l0_noninferior,
        "H_B2_IMPROVES_RAW_ALL_DEVELOPMENT_SCHEMES": h_improves_raw,
        "Q2_ONLY_CORRECTION_CANDIDATE": "YES" if h_improves_raw or l0_improves_raw else "NO",
        "MORE_DATA_REQUIRED_BEFORE_NEXT_ANALYSIS": more_data,
        "historical_SELECTED_SURFACE_MODEL": "B2",
        "historical_Q1_RETAINED": "NO",
        "historical_Q2_GAP_FILLED": "NO",
        "historical_SURFACE2C_ALLOWED": "NO",
    }


def prior_free_stats(prior: dict[str, Any]) -> dict[str, Any]:
    stability = prior["coefficient_stability"]
    magnitude = prior["magnitude"]
    b0 = next((row for row in stability if row.get("layer") == "L-B2" and row.get("parameter") == "b0"), {})
    b2 = next((row for row in stability if row.get("layer") == "L-B2" and row.get("parameter") == "b2"), {})
    summary = prior["summary"].get("physical_audit", {})
    coefficients = [
        row for row in prior["coefficients"]
        if row.get("layer") == "L-B2"
        and row.get("cv_scheme") in DEV_SCHEMES
        and row.get("parameter") == "b0"
    ]
    condition_numbers = [
        float(row["design_condition_number"])
        for row in coefficients if finite(row.get("design_condition_number"))
    ]
    return {
        "b0_mean": float(b0.get("mean", "nan")),
        "b0_std": float(b0.get("std", "nan")),
        "b0_range": float(b0.get("range", "nan")),
        "b0_relative_range": float(b0.get("relative_range_to_abs_mean", "nan")),
        "b2_mean": float(b2.get("mean", "nan")),
        "b2_std": float(b2.get("std", "nan")),
        "b2_range": float(b2.get("range", "nan")),
        "b2_relative_range": float(b2.get("relative_range_to_abs_mean", "nan")),
        "b2_sign_consistency": str(b2.get("sign_consistency", "")),
        "design_condition_number_max": max(condition_numbers, default=float("nan")),
        "design_condition_number_median": float(np.median(condition_numbers)) if condition_numbers else float("nan"),
        "development_max_abs_delta_lambda_mm": summary.get("development_max_abs_delta_lambda_mm"),
        "development_max_abs_delta_Zg_mm": summary.get("development_max_abs_delta_Zg_mm"),
        "new_invalid_point_count": summary.get("new_invalid_point_count"),
        "magnitude_row_count": len(magnitude),
    }


def gauge_audit(
    prior_stats: dict[str, Any],
    anchored_stability: list[dict[str, Any]],
) -> dict[str, Any]:
    b2_rows = [
        row for row in anchored_stability
        if row.get("source") == "anchored_l0_replay"
        and row.get("layer") == "L0-B2"
        and row.get("parameter") == "b2"
    ]
    anchored_relative_range = max(
        (float(row["relative_range_to_abs_mean"]) for row in b2_rows if finite(row.get("relative_range_to_abs_mean"))),
        default=float("inf"),
    )
    anchored_sign_consistency = bool(b2_rows and all(bool(row.get("sign_consistency")) for row in b2_rows))
    common_mode_evidence = bool(
        finite(prior_stats["b0_mean"])
        and abs(prior_stats["b0_mean"]) > 5.0
        and finite(prior_stats["design_condition_number_max"])
        and prior_stats["design_condition_number_max"] > 1.0e4
        and str(prior_stats["b2_sign_consistency"]).lower() == "false"
    )
    anchored_b2_stable = anchored_relative_range <= 1.0 and anchored_sign_consistency
    return {
        "free_intercept_common_mode_evidence": common_mode_evidence,
        "free_intercept_b0_mean_mm": prior_stats["b0_mean"],
        "free_intercept_design_condition_number_max": prior_stats["design_condition_number_max"],
        "free_intercept_b2_sign_consistent": prior_stats["b2_sign_consistency"],
        "anchored_b2_relative_range": anchored_relative_range,
        "anchored_b2_sign_consistent": anchored_sign_consistency,
        "anchored_b2_stable": anchored_b2_stable,
        "interpretation": (
            "Free b0 is consistent with a common-mode/gauge direction: its magnitude "
            "is tens of millimetres, the design is ill-conditioned, and free b2 "
            "changes sign across development folds."
            if common_mode_evidence
            else "The frozen free-intercept diagnostics do not provide sufficient "
                 "evidence for a common-mode/gauge interpretation."
        ),
    }


def plot_comparison(output: Path, current_metric_rows: list[dict[str, Any]]) -> None:
    pooled = pooled_map(current_metric_rows)
    schemes = DEV_SCHEMES + ("strict_50mm_validation",)
    labels = ("LOHO", "LOPO", "LOBO", "50 strict")
    layers = ("RAW", "H-B2", "L-B2", "L0-B2")
    colors = {"RAW": "#7f8c8d", "H-B2": "#2c7fb8", "L-B2": "#d95f02", "L0-B2": "#1b9e77"}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    x = np.arange(len(schemes), dtype=np.float64)
    width = 0.19
    fields = ("rmse_mm", "p95_abs_mm", "worst_condition_abs_mm")
    titles = ("Condition-mean RMSE (mm)", "Condition-mean P95 (mm)", "Worst condition abs bias (mm)")
    for index, layer in enumerate(layers):
        values = [
            metric_number(pooled[(scheme, layer)], field)
            if (scheme, layer) in pooled else np.nan
            for scheme in schemes
            for field in []
        ]
        del values
        for axis, field, title in zip(axes, fields, titles):
            values = [
                metric_number(pooled[(scheme, layer)], field)
                if (scheme, layer) in pooled else np.nan
                for scheme in schemes
            ]
            axis.bar(
                x + (index - (len(layers) - 1) / 2) * width,
                values,
                width,
                label=layer,
                color=colors[layer],
            )
            axis.set_title(title)
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=20)
            axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.savefig(output / "surface2_lambda_gauge_audit_comparison.png", dpi=180)
    plt.close(figure)


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def report_text(
    decision: dict[str, Any],
    gauge: dict[str, Any],
    physical: dict[str, Any],
    prior_stats: dict[str, Any],
    prior_protocol: dict[str, Any],
    current_metric_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> str:
    pooled = pooled_map(current_metric_rows)
    metric_lines: list[str] = []
    for scheme in DEV_SCHEMES + ("strict_50mm_validation",):
        for layer in ("RAW", "H-B2", "L-B2", "L0-B2"):
            row = pooled.get((scheme, layer))
            if row is None:
                continue
            metric_lines.append(
                f"| {scheme} | {layer} | {row['condition_count']} | "
                f"{fmt(row['bias_mm'])} | {fmt(row['mae_mm'])} | "
                f"{fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | "
                f"{fmt(row['max_abs_mm'])} | {fmt(row['worst_condition_abs_mm'])} |"
            )
    comparison_lines: list[str] = []
    for row in comparison_rows:
        if row["aggregation"] != "pooled_condition_means" or row["cv_scheme"] not in DEV_SCHEMES + ("strict_50mm_validation",):
            continue
        comparison_lines.append(
            f"| {row['cv_scheme']} | {row['heldout_group']} | "
            f"{fmt(row['delta_L0_minus_H_rmse_mm'])} | "
            f"{fmt(row['delta_L0_minus_H_p95_mm'])} | "
            f"{fmt(row['delta_L0_minus_H_worst_mm'])} | "
            f"{fmt(row['delta_L0_minus_free_L_rmse_mm'])} | "
            f"{row['L0_not_worse_than_current_H_all_three']} |"
        )
    stability_lines = [
        f"| {row.get('source')} | {row.get('layer')} | {row.get('parameter')} | "
        f"{fmt(row.get('mean'), 7)} | {fmt(row.get('std'), 7)} | "
        f"{fmt(row.get('range'), 7)} | {fmt(row.get('relative_range_to_abs_mean'), 4)} | "
        f"{row.get('sign_consistency')} |"
        for row in stability_rows
        if row.get("layer") in {"L-B2", "L0-B2"}
    ]
    replay_difference = max(
        row["max_abs_raw_residual_replay_difference_mm"]
        for row in consistency
    )
    return f"""# Surface lambda gauge audit

## Decision

LAMBDA_INTERCEPT_IDENTIFIABILITY={decision['LAMBDA_INTERCEPT_IDENTIFIABILITY']}

ANCHORED_LAMBDA_LAYER_VALIDITY={decision['ANCHORED_LAMBDA_LAYER_VALIDITY']}

FINAL_CORRECTION_LAYER={decision['FINAL_CORRECTION_LAYER']}

STOP_LAMBDA_ROUTE={decision['STOP_LAMBDA_ROUTE']}

The historical conclusions remain unchanged:
SELECTED_SURFACE_MODEL=B2, Q1_RETAINED=NO, Q2_GAP_FILLED=NO,
SURFACE2C_ALLOWED=NO. This is a diagnostic audit, not production validation.

## Protocol and provenance

- L0-B2 is delta_lambda=b2*q2 with b0 fixed exactly to zero.
- The same correction is applied to repeat1 ground points and repeat2-5 formal
  points. Every fold refits the session-linear ground proxy after correcting
  repeat1, before evaluating formal residuals.
- Current H-B2 and free L-B2 comparison rows are a fresh strict-50-excluded
  replay under the same fold definitions as L0-B2. The previous H-B2/L-B2
  result files are preserved as frozen historical artifacts and are included
  in the comparison CSV without modification.
- LOHO, LOPO and LOBO are development grouped CV. No random point split is
  used. 50 mm is evaluated only in the separate strict diagnostic fold and is
  not used for development fitting, selection, or threshold adjustment.
- Frozen C0 SHA256: {inputs['hashes']['quadratic_c0_sha256']}
- Frozen C1 SHA256: {inputs['hashes']['frozen_c1_sha256']}
- Config SHA256: {inputs['hashes']['config_sha256']}
- q2 remains the Frozen-C0 intrinsic coordinate. C0, C1, ROI, Steger and the
  ground protocol were not changed.
- Maximum raw residual replay difference: {fmt(replay_difference, 10)} mm.

## Provenance audit of the previous artifact

The previous result files were not overwritten. Their hashes are recorded in
the summary JSON. The previous fold audit records development train counts
{prior_protocol['previous_dev_train_condition_counts']}; this indicates that
the old development folds included the 50 mm conditions. Because the present
request requires strict 50 mm held-out status, this report uses a new
50-excluded H/free replay for the direct comparison. The old H/free numbers
remain available under source=previous_frozen_artifact.

The previous artifact recorded config SHA256
{prior_protocol['previous_config_sha256']}, while this replay used
{prior_protocol['current_config_sha256']}; config_sha256_matches_previous_artifact=
{prior_protocol['config_sha256_matches_previous_artifact']}. C0 and C1 hashes
remain explicitly recorded above. The current strict-dev-only replay is therefore
the comparison authority for this audit; the previous H/free rows are retained
historical references only.

## Grouped metrics: current strict-50-excluded replay

| scheme | layer | conditions | Bias | MAE | RMSE | P95 | Max | worst condition |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

## Anchored L0 relative to H-B2

Negative deltas mean anchored L0-B2 is smaller.

| scheme | heldout group | delta RMSE L0-H | delta P95 L0-H | delta worst L0-H | delta RMSE L0-free-L | L0 not worse in all three |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(comparison_lines)}

## Gauge interpretation

- Previous free L-B2 b0 mean/std/range: {fmt(prior_stats['b0_mean'], 6)} /
  {fmt(prior_stats['b0_std'], 6)} / {fmt(prior_stats['b0_range'], 6)} mm.
- Previous free L-B2 b2 mean/std/range: {fmt(prior_stats['b2_mean'], 8)} /
  {fmt(prior_stats['b2_std'], 8)} / {fmt(prior_stats['b2_range'], 8)}.
- Previous free-L maximum development absolute delta lambda:
  {fmt(prior_stats['development_max_abs_delta_lambda_mm'], 6)} mm; induced
  absolute delta Zg: {fmt(prior_stats['development_max_abs_delta_Zg_mm'], 6)} mm.
- Previous free-L design condition number maximum:
  {fmt(prior_stats['design_condition_number_max'], 2)}.
- {gauge['interpretation']}
- Anchored b2 relative fold range:
  {fmt(gauge['anchored_b2_relative_range'], 6)}; sign consistent:
  {gauge['anchored_b2_sign_consistent']}.

The evidence therefore classifies the free intercept as a common-mode/gauge
direction only when the anchored coefficient and physical audit are considered
together; it does not make a physically large lambda correction acceptable.

## Anchored lambda physical and numerical audit

- Development status: {physical['L0_B2_DEVELOPMENT_PHYSICAL_STATUS']}
- All-fold status including strict diagnostic: {physical['L0_B2_PHYSICAL_STATUS']}
- Development new formal invalid points: {physical['development_formal_new_invalid_point_count']}
- Development new baseline invalid points: {physical['development_baseline_new_invalid_point_count']}
- All-fold new invalid points: {physical['new_invalid_point_count']}
- C1 clamp state changes: {physical['c1_clamp_state_change_count']}
- Development max absolute delta lambda: {fmt(physical['development_max_abs_delta_lambda_mm'], 6)} mm
- Development max P95 absolute delta lambda: {fmt(physical['development_max_p95_abs_delta_lambda_mm'], 6)} mm
- Development max absolute induced delta Zg: {fmt(physical['development_max_abs_delta_Zg_mm'], 6)} mm
- All-fold max absolute delta lambda: {fmt(physical['max_abs_delta_lambda_mm'], 6)} mm
- All-fold max absolute induced delta Zg: {fmt(physical['max_abs_delta_Zg_mm'], 6)} mm
- Maximum absolute corrected-proxy slope change:
  {fmt(physical['max_abs_corrected_proxy_slope_change_mm_per_mm'], 8)} mm/mm
- Maximum absolute corrected-proxy intercept change:
  {fmt(physical['max_abs_corrected_proxy_intercept_change_mm'], 6)} mm
- Maximum corrected-proxy RMSE:
  {fmt(physical['max_corrected_proxy_rmse_mm'], 6)} mm
- Thresholds: max absolute delta lambda <= 2 mm, P95 absolute delta lambda
  <= 1 mm, absolute induced delta Zg <= 2 mm, b2 relative range <= 1.

## Coefficient stability

| source | layer | parameter | mean | std | range | relative range | sign consistent |
|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(stability_lines)}

## Final interpretation

The anchored model is accepted as a lambda-layer candidate only if it is
non-inferior to H-B2 in RMSE, P95 and worst-condition error for every
development LOHO/LOPO/LOBO pooled fold and passes the independent physical
audit. A large free b0 by itself is not evidence to deploy lambda correction.
If the anchored model fails either criterion, lambda complexity is stopped and
the frozen height-layer route remains the final candidate. The 50 mm result
shown in the CSV and plot is strict diagnostic only.
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
    inputs = base.load_inputs(args.config.resolve())
    set_measurement_params(inputs)
    conditions, consistency = base.build_conditions(inputs)
    development = {
        key: value for key, value in conditions.items()
        if value.nominal_height_mm != 50.0
    }
    strict_50 = {
        key: value for key, value in conditions.items()
        if value.nominal_height_mm == 50.0
    }
    if len(development) != 44 or len(strict_50) != 5:
        raise RuntimeError(
            f"expected 44 development and 5 strict-50 conditions, got "
            f"{len(development)} and {len(strict_50)}"
        )
    prior = prior_artifacts()
    prior_protocol = previous_fold_protocol_audit(
        prior,
        inputs["hashes"]["config_sha256"],
    )
    folds = strict_dev_fold_definitions(conditions)
    current_metric_rows: list[dict[str, Any]] = []
    current_condition_rows: list[dict[str, Any]] = []
    current_support_rows: list[dict[str, Any]] = []
    current_coefficient_rows: list[dict[str, Any]] = []
    current_magnitude_rows: list[dict[str, Any]] = []
    current_proxy_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for scheme, group, train, test in folds:
        if not train or not test:
            continue
        fair = base.run_fold(scheme, group, train, test, inputs, {})
        fair_metrics, fair_conditions, fair_support, fair_coefficients, fair_magnitudes, fair_proxies, fair_fit = fair
        current_metric_rows.extend(add_source(fair_metrics, "strict_dev_only_replay"))
        current_condition_rows.extend(add_source(fair_conditions, "strict_dev_only_replay"))
        current_support_rows.extend(add_source(fair_support, "strict_dev_only_replay"))
        current_coefficient_rows.extend(add_source(fair_coefficients, "strict_dev_only_replay"))
        current_magnitude_rows.extend(add_source(fair_magnitudes, "strict_dev_only_replay"))
        current_proxy_rows.extend(add_source(fair_proxies, "strict_dev_only_replay"))
        l0 = run_l0_fold(scheme, group, train, test, inputs)
        l0_metrics, l0_conditions, l0_support, l0_coefficients, l0_magnitudes, l0_proxies, l0_fit = l0
        current_metric_rows.extend(add_source(l0_metrics, "anchored_l0_replay"))
        current_condition_rows.extend(add_source(l0_conditions, "anchored_l0_replay"))
        current_support_rows.extend(add_source(l0_support, "anchored_l0_replay"))
        current_coefficient_rows.extend(add_source(l0_coefficients, "anchored_l0_replay"))
        current_magnitude_rows.extend(add_source(l0_magnitudes, "anchored_l0_replay"))
        current_proxy_rows.extend(add_source(l0_proxies, "anchored_l0_replay"))
        fold_records.append({
            "cv_scheme": scheme,
            "heldout_group": group,
            "train_condition_count": len(train),
            "test_condition_count": len(test),
            "H_fit_status": fair_fit["hfit"]["fit_status"],
            "free_L_fit_status": fair_fit["lfit"]["fit_status"],
            "anchored_L0_fit_status": l0_fit["fit"]["fit_status"],
            "anchored_L0_invalid_formal_test_points": l0_fit["invalid_formal_total"],
            "anchored_L0_invalid_baseline_test_points": l0_fit["invalid_baseline_total"],
        })
    fair_fold_metric_rows = [
        row for row in current_metric_rows
        if row.get("source") == "strict_dev_only_replay"
    ]
    fair_fold_condition_rows = [
        row for row in current_condition_rows
        if row.get("source") == "strict_dev_only_replay"
    ]
    fair_fold_support_rows = [
        row for row in current_support_rows
        if row.get("source") == "strict_dev_only_replay"
    ]
    l0_fold_metric_rows = [
        row for row in current_metric_rows
        if row.get("source") == "anchored_l0_replay"
    ]
    l0_fold_condition_rows = [
        row for row in current_condition_rows
        if row.get("source") == "anchored_l0_replay"
    ]
    l0_fold_support_rows = [
        row for row in current_support_rows
        if row.get("source") == "anchored_l0_replay"
    ]
    fair_pooled, fair_pooled_support = base.pooled_rows(
        fair_fold_metric_rows,
        fair_fold_condition_rows,
        fair_fold_support_rows,
    )
    l0_pooled, l0_pooled_support = pool_l0(
        l0_fold_metric_rows,
        l0_fold_condition_rows,
    )
    current_metric_rows.extend(fair_pooled)
    current_metric_rows.extend(l0_pooled)
    current_support_rows.extend(fair_pooled_support)
    current_support_rows.extend(l0_pooled_support)
    prior_metric_rows = add_source(prior["cv_metrics"], "previous_frozen_artifact")
    comparison_rows = make_comparison_rows(current_metric_rows, prior_metric_rows)
    support_comparison_rows = make_support_comparison(
        current_support_rows,
        add_source(prior["support_metrics"], "previous_frozen_artifact"),
    )
    current_stability = summarize_stability(current_coefficient_rows, "strict_dev_only_replay")
    anchored_stability = summarize_stability(
        [
            row for row in current_coefficient_rows
            if row.get("source") == "anchored_l0_replay"
        ],
        "anchored_l0_replay",
    )
    stability_rows = prior_stability_rows(prior) + current_stability + anchored_stability
    physical = physical_audit_l0(
        current_metric_rows,
        current_coefficient_rows,
        stability_rows,
        current_magnitude_rows,
        [
            row for row in current_proxy_rows
            if row.get("source") == "anchored_l0_replay"
        ],
    )
    prior_stats = prior_free_stats(prior)
    gauge = gauge_audit(prior_stats, stability_rows)
    decision = decision_from_results(
        current_metric_rows,
        comparison_rows,
        physical,
        prior["summary"],
        gauge["free_intercept_common_mode_evidence"],
    )
    merged_cv_rows = prior_metric_rows + current_metric_rows
    merged_condition_rows = (
        add_source(prior["condition_metrics"], "previous_frozen_artifact")
        + current_condition_rows
    )
    merged_support_rows = (
        add_source(prior["support_metrics"], "previous_frozen_artifact")
        + current_support_rows
    )
    merged_coefficients = (
        add_source(prior["coefficients"], "previous_frozen_artifact")
        + current_coefficient_rows
    )
    merged_magnitude = (
        add_source(prior["magnitude"], "previous_frozen_artifact")
        + current_magnitude_rows
    )
    write_csv(output / "surface2_lambda_gauge_audit_cv_metrics.csv", merged_cv_rows)
    write_csv(output / "surface2_lambda_gauge_audit_condition_metrics.csv", merged_condition_rows)
    write_csv(output / "surface2_lambda_gauge_audit_support_metrics.csv", merged_support_rows)
    write_csv(output / "surface2_lambda_gauge_audit_comparison.csv", comparison_rows)
    write_csv(output / "surface2_lambda_gauge_audit_support_comparison.csv", support_comparison_rows)
    write_csv(output / "surface2_lambda_gauge_audit_coefficients.csv", merged_coefficients)
    write_csv(output / "surface2_lambda_gauge_audit_coefficient_stability.csv", stability_rows)
    write_csv(output / "surface2_lambda_gauge_audit_magnitude.csv", merged_magnitude)
    write_csv(output / "surface2_lambda_gauge_audit_ground_proxy.csv", current_proxy_rows)
    write_csv(output / "surface2_lambda_gauge_audit_fold_audit.csv", fold_records)
    write_csv(output / "surface2_lambda_gauge_audit_raw_replay.csv", consistency)
    plot_comparison(output, current_metric_rows)
    summary = {
        "decision": decision,
        "gauge_audit": gauge,
        "prior_free_intercept_stats": prior_stats,
        "physical_audit": physical,
        "protocol": {
            "condition_equal_weight": True,
            "random_point_split": False,
            "development_folds_exclude_50mm": True,
            "strict_50mm_diagnostic_only": True,
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
            "input_sha256": {name: sha256(path) for name, path in inputs["paths"].items()},
            "frozen_hashes": inputs["hashes"],
            "reused_previous_output_sha256": prior["files_sha256"],
            "development_condition_count": len(development),
            "strict_50_condition_count": len(strict_50),
            "development_formal_point_count": sum(c.formal.source_count for c in development.values()),
            "strict_50_formal_point_count": sum(c.formal.source_count for c in strict_50.values()),
            "raw_replay_max_difference_mm": max(
                row["max_abs_raw_residual_replay_difference_mm"] for row in consistency
            ),
        },
        "previous_artifact_protocol_audit": prior_protocol,
        "historical_conclusion_preserved": {
            "SELECTED_SURFACE_MODEL": "B2",
            "Q1_RETAINED": "NO",
            "Q2_GAP_FILLED": "NO",
            "SURFACE2C_ALLOWED": "NO",
        },
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface2_lambda_gauge_audit_summary.json", summary)
    (output / "surface2_lambda_gauge_audit_report.md").write_text(
        report_text(
            decision,
            gauge,
            physical,
            prior_stats,
            prior_protocol,
            current_metric_rows,
            comparison_rows,
            stability_rows,
            consistency,
            inputs,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        **decision,
        "anchored_development_physical_status": physical["L0_B2_DEVELOPMENT_PHYSICAL_STATUS"],
        "raw_replay_max_difference_mm": summary["provenance"]["raw_replay_max_difference_mm"],
        "development_condition_count": len(development),
        "strict_50_condition_count": len(strict_50),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
