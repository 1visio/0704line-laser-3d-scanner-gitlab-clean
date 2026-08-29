"""Freeze and replay the selected H-B2(q2-only) diagnostic candidate.

This script is deliberately separate from the production measurement path.
It reuses the canonical Frozen-C0/C1 and condition loader from
compare_surface_correction_layers.py, fits one H-B2 candidate on the 44
development conditions with 50 mm excluded, and writes an engineering
acceptance plan.  It never edits the historical Surface-2 artifacts or any
online configuration.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compare_surface_correction_layers as base
from measurement.height_measure import MeasurementParams


BASE_OUTPUT = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
SURFACE2 = BASE_OUTPUT / "surface2"
MODEL_SELECTION = SURFACE2 / "surface2_model_selection"
LAMBDA_AUDIT = SURFACE2 / "lambda_gauge_audit"
DEFAULT_OUTPUT = SURFACE2 / "surface3_hb2_candidate"
DEFAULT_CONFIG = ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
DEV_HEIGHTS = (1.0, 2.0, 6.0, 10.0, 20.0, 30.0, 36.0, 40.0, 46.0)
METRIC_FIELDS = ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm")
EPS = 1.0e-12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else f"{float(value):.15g}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def stats(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def error_metrics(values: Any) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {name: None for name in METRIC_FIELDS}
    absolute = np.abs(array)
    return {
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array * array))),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "max_abs_mm": float(np.max(absolute)),
    }


def set_measurement_params(inputs: dict[str, Any]) -> None:
    params = inputs["app"].measurement
    inputs["measurement_params"] = params if isinstance(params, MeasurementParams) else MeasurementParams()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def load_artifacts(config_path: Path) -> dict[str, Any]:
    inputs = base.load_inputs(config_path)
    set_measurement_params(inputs)
    conditions, consistency = base.build_conditions(inputs)
    development = {
        key: condition
        for key, condition in conditions.items()
        if condition.nominal_height_mm != 50.0
    }
    strict_50 = {
        key: condition
        for key, condition in conditions.items()
        if condition.nominal_height_mm == 50.0
    }
    if len(development) != 44 or len(strict_50) != 5:
        raise RuntimeError(
            f"expected 44 development and 5 strict-50 conditions, got "
            f"{len(development)} and {len(strict_50)}"
        )
    if sum(condition.formal.source_count for condition in development.values()) != 11160:
        raise RuntimeError("development formal point count is not the frozen 11160")
    if sum(condition.formal.source_count for condition in strict_50.values()) != 1100:
        raise RuntimeError("strict-50 formal point count is not the frozen 1100")

    model_summary_path = inputs["paths"]["model_selection_summary"]
    model_summary = read_json(model_summary_path)
    model_decision = model_summary.get("decision", {})
    if model_summary.get("SELECTED_SURFACE_MODEL") != "B2":
        raise RuntimeError("Surface-2 model selection is not frozen to B2")
    if model_summary.get("Q1_RETAINED") != "NO":
        raise RuntimeError("Surface-2 model selection does not freeze Q1_RETAINED=NO")
    if model_summary.get("Q2_ONLY_CORRECTION_RECOMMENDED") != "YES":
        raise RuntimeError("Surface-2 model selection does not recommend q2-only correction")
    if model_summary.get("historical_surface2b_conclusion", {}).get("Q2_GAP_FILLED") != "NO":
        raise RuntimeError("historical Q2_GAP_FILLED conclusion changed unexpectedly")
    if model_summary.get("historical_surface2b_conclusion", {}).get("SURFACE2C_ALLOWED") != "NO":
        raise RuntimeError("historical SURFACE2C_ALLOWED conclusion changed unexpectedly")
    if model_decision.get("SELECTED_SURFACE_MODEL") != "B2":
        raise RuntimeError("nested model selection decision is not B2")

    lambda_summary_path = LAMBDA_AUDIT / "surface2_lambda_gauge_audit_summary.json"
    if not lambda_summary_path.is_file():
        raise FileNotFoundError(f"missing lambda gauge audit summary: {lambda_summary_path}")
    lambda_summary = read_json(lambda_summary_path)
    lambda_decision = lambda_summary.get("decision", {})
    if lambda_decision.get("FINAL_CORRECTION_LAYER") != "HEIGHT":
        raise RuntimeError("lambda gauge audit does not freeze HEIGHT as final layer")
    if lambda_decision.get("STOP_LAMBDA_ROUTE") != "YES":
        raise RuntimeError("lambda gauge audit does not freeze STOP_LAMBDA_ROUTE=YES")

    if inputs["app"].reconstruction.enable_laser_ray_correction is not True:
        raise RuntimeError("enable_laser_ray_correction must remain true")

    model_paths = {
        "model_selection_summary": model_summary_path,
        "model_selection_report": MODEL_SELECTION / "surface2_model_selection_report.md",
        "model_selection_cv_metrics": MODEL_SELECTION / "surface2_model_selection_cv_metrics.csv",
        "model_selection_predictions": inputs["paths"]["surface2br2_predictions"],
        "model_selection_coefficients": MODEL_SELECTION / "surface2_model_selection_coefficients.csv",
        "model_selection_stability": MODEL_SELECTION / "surface2_model_selection_coefficient_stability.csv",
        "lambda_summary": lambda_summary_path,
        "lambda_report": LAMBDA_AUDIT / "surface2_lambda_gauge_audit_report.md",
        "surface1a_points": inputs["paths"]["surface1a_points"],
        "surface1a_summary": inputs["paths"]["surface1a_summary"],
        "surface_coordinate": inputs["paths"]["surface_coordinate"],
        "surface2b_samples": inputs["paths"]["surface2b_samples"],
        "surface2br2_condition_table": inputs["paths"]["surface2br2_condition_table"],
        "pointwise_diagnostics": inputs["paths"]["pointwise_diagnostics"],
        "surface2_roi": inputs["paths"]["surface2_roi"],
        "height50_roi": inputs["paths"]["height50_roi"],
        "config": inputs["paths"]["config"],
        "quadratic_c0": inputs["app"].calibration.laser_plane,
        "frozen_c1": inputs["app"].calibration.laser_ray_correction,
        "intrinsics": inputs["app"].calibration.intrinsics,
        "extrinsics": inputs["app"].calibration.extrinsics,
    }
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing candidate provenance artifact: " + ", ".join(missing))

    model_input_sha = model_summary.get("input_sha256", {})
    expected_model_sha = {
        "condition_table": sha256(inputs["paths"]["surface2br2_condition_table"]),
        "cv_metrics": sha256(inputs["paths"]["surface2br2_cv_metrics"]),
        "coefficients": sha256(inputs["paths"]["surface2br2_coefficients"]),
        "coefficient_stability": sha256(inputs["paths"]["surface2br2_stability"]),
        "condition_predictions": sha256(inputs["paths"]["surface2br2_predictions"]),
        "summary": sha256(inputs["paths"]["surface2br2_summary"]),
    }
    model_input_sha_match = {
        key: bool(model_input_sha.get(key) == value)
        for key, value in expected_model_sha.items()
    }
    if not all(model_input_sha_match.values()):
        raise RuntimeError(f"Surface-2 model-selection input SHA mismatch: {model_input_sha_match}")

    lambda_frozen = lambda_summary.get("provenance", {}).get("frozen_hashes", {})
    lambda_frozen_match = {
        "quadratic_c0_sha256": lambda_frozen.get("quadratic_c0_sha256") == inputs["hashes"]["quadratic_c0_sha256"],
        "frozen_c1_sha256": lambda_frozen.get("frozen_c1_sha256") == inputs["hashes"]["frozen_c1_sha256"],
    }
    if not all(lambda_frozen_match.values()):
        raise RuntimeError(f"lambda audit Frozen C0/C1 SHA mismatch: {lambda_frozen_match}")

    source_sha = {name: sha256(path) for name, path in model_paths.items()}
    return {
        "inputs": inputs,
        "conditions": conditions,
        "development": development,
        "strict_50": strict_50,
        "consistency": consistency,
        "model_summary": model_summary,
        "lambda_summary": lambda_summary,
        "model_paths": model_paths,
        "source_sha": source_sha,
        "model_input_sha_match": model_input_sha_match,
        "lambda_frozen_match": lambda_frozen_match,
    }


def fit_full_candidate(data: dict[str, Any]) -> dict[str, Any]:
    fit = base.fit_height_model(list(data["development"].values()))
    beta = np.asarray(fit["beta"], dtype=np.float64).reshape(2)
    if fit["fit_status"] != "success" or fit["design_rank"] != 2:
        raise RuntimeError(f"full H-B2 fit failed: {fit}")
    if not np.all(np.isfinite(beta)):
        raise RuntimeError("full H-B2 parameters are non-finite")
    return {
        "fit": fit,
        "a0": float(beta[0]),
        "a2": float(beta[1]),
        "train_condition_count": len(data["development"]),
        "train_point_count": sum(
            condition.formal.source_count
            for condition in data["development"].values()
        ),
    }


def condition_replay_row(
    condition: base.Condition,
    beta: np.ndarray,
    q2_lo: float,
    q2_hi: float,
    population: str,
    strategy: str = "raw_formula",
) -> dict[str, Any]:
    raw = np.asarray(condition.formal.raw_residual, dtype=np.float64)
    q2 = np.asarray(condition.formal.q2, dtype=np.float64)
    in_domain = (q2 >= q2_lo - EPS) & (q2 <= q2_hi + EPS)
    if strategy == "clamp":
        q2_used = np.clip(q2, q2_lo, q2_hi)
        accepted = np.ones(len(q2), dtype=bool)
    elif strategy == "reject":
        q2_used = q2
        accepted = in_domain
    else:
        q2_used = q2
        accepted = np.ones(len(q2), dtype=bool)
    delta_h = beta[0] + beta[1] * q2_used
    corrected = raw - delta_h
    selected_raw = raw[accepted]
    selected_corrected = corrected[accepted]
    selected_delta = delta_h[accepted]
    raw_metrics = error_metrics(selected_raw)
    corrected_metrics = error_metrics(selected_corrected)
    return {
        "condition_id": condition.condition_id,
        "dataset": condition.dataset,
        "nominal_height_mm": condition.nominal_height_mm,
        "true_height_mm": condition.true_height_mm,
        "position_rank": condition.position_rank,
        "population": population,
        "strategy": strategy,
        "point_count_total": int(len(q2)),
        "point_count_evaluated": int(np.count_nonzero(accepted)),
        "out_of_domain_point_count": int(np.count_nonzero(~in_domain)),
        "out_of_domain_rate": float(np.mean(~in_domain)),
        "clamped_point_count": int(
            np.count_nonzero((strategy == "clamp") & (~in_domain))
        ),
        "q2_min": float(np.min(q2)),
        "q2_p05": float(np.percentile(q2, 5.0)),
        "q2_median": float(np.median(q2)),
        "q2_p95": float(np.percentile(q2, 95.0)),
        "q2_max": float(np.max(q2)),
        "delta_h_min_mm": float(np.min(selected_delta)) if len(selected_delta) else None,
        "delta_h_p05_mm": float(np.percentile(selected_delta, 5.0)) if len(selected_delta) else None,
        "delta_h_median_mm": float(np.median(selected_delta)) if len(selected_delta) else None,
        "delta_h_p95_mm": float(np.percentile(selected_delta, 95.0)) if len(selected_delta) else None,
        "delta_h_max_mm": float(np.max(selected_delta)) if len(selected_delta) else None,
        "raw_condition_bias_mm": float(np.mean(selected_raw)) if len(selected_raw) else None,
        "corrected_condition_bias_mm": float(np.mean(selected_corrected)) if len(selected_corrected) else None,
        "raw_condition_abs_bias_mm": abs(float(np.mean(selected_raw))) if len(selected_raw) else None,
        "corrected_condition_abs_bias_mm": abs(float(np.mean(selected_corrected))) if len(selected_corrected) else None,
        "raw_point_bias_mm": raw_metrics["bias_mm"],
        "raw_point_mae_mm": raw_metrics["mae_mm"],
        "raw_point_rmse_mm": raw_metrics["rmse_mm"],
        "raw_point_p95_abs_mm": raw_metrics["p95_abs_mm"],
        "raw_point_max_abs_mm": raw_metrics["max_abs_mm"],
        "corrected_point_bias_mm": corrected_metrics["bias_mm"],
        "corrected_point_mae_mm": corrected_metrics["mae_mm"],
        "corrected_point_rmse_mm": corrected_metrics["rmse_mm"],
        "corrected_point_p95_abs_mm": corrected_metrics["p95_abs_mm"],
        "corrected_point_max_abs_mm": corrected_metrics["max_abs_mm"],
    }


def aggregate_condition_rows(
    rows: list[dict[str, Any]],
    group_type: str,
    group: str,
    population: str,
    strategy: str = "raw_formula",
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["population"] == population
        and row["strategy"] == strategy
    ]
    if group_type == "height":
        selected = [row for row in selected if row["nominal_height_mm"] == float(group)]
    elif group_type == "position_rank":
        selected = [row for row in selected if row["position_rank"] == int(group)]
    elif group_type == "strict_50mm":
        selected = [row for row in selected if row["population"] == "strict_50mm"]
    if not selected:
        return {
            "evaluation_scope": "full_candidate_replay",
            "group_type": group_type,
            "group": group,
            "population": population,
            "strategy": strategy,
            "condition_count": 0,
        }
    raw_condition = np.asarray(
        [row["raw_condition_bias_mm"] for row in selected if finite(row["raw_condition_bias_mm"])],
        dtype=np.float64,
    )
    corrected_condition = np.asarray(
        [row["corrected_condition_bias_mm"] for row in selected if finite(row["corrected_condition_bias_mm"])],
        dtype=np.float64,
    )
    q2 = np.concatenate([
        np.asarray([
            row["q2_min"], row["q2_p05"], row["q2_median"], row["q2_p95"], row["q2_max"]
        ], dtype=np.float64)
        for row in selected
    ])
    raw_metrics = error_metrics(raw_condition)
    corrected_metrics = error_metrics(corrected_condition)
    worst = max(
        (row for row in selected if finite(row["corrected_condition_bias_mm"])),
        key=lambda row: abs(float(row["corrected_condition_bias_mm"])),
        default=None,
    )
    return {
        "evaluation_scope": "full_candidate_replay",
        "group_type": group_type,
        "group": group,
        "population": population,
        "strategy": strategy,
        "condition_count": int(len(selected)),
        "point_count_total": int(sum(row["point_count_total"] for row in selected)),
        "point_count_evaluated": int(sum(row["point_count_evaluated"] for row in selected)),
        "out_of_domain_point_count": int(sum(row["out_of_domain_point_count"] for row in selected)),
        "out_of_domain_rate": float(
            sum(row["out_of_domain_point_count"] for row in selected)
            / max(sum(row["point_count_total"] for row in selected), 1)
        ),
        "clamped_point_count": int(sum(row["clamped_point_count"] for row in selected)),
        "q2_min": float(np.min(q2)),
        "q2_p05": float(np.percentile(q2, 5.0)),
        "q2_median": float(np.median(q2)),
        "q2_p95": float(np.percentile(q2, 95.0)),
        "q2_max": float(np.max(q2)),
        "raw_bias_mm": raw_metrics["bias_mm"],
        "raw_mae_mm": raw_metrics["mae_mm"],
        "raw_rmse_mm": raw_metrics["rmse_mm"],
        "raw_p95_abs_mm": raw_metrics["p95_abs_mm"],
        "raw_max_abs_mm": raw_metrics["max_abs_mm"],
        "corrected_bias_mm": corrected_metrics["bias_mm"],
        "corrected_mae_mm": corrected_metrics["mae_mm"],
        "corrected_rmse_mm": corrected_metrics["rmse_mm"],
        "corrected_p95_abs_mm": corrected_metrics["p95_abs_mm"],
        "corrected_max_abs_mm": corrected_metrics["max_abs_mm"],
        "delta_bias_corrected_minus_raw_mm": (
            corrected_metrics["bias_mm"] - raw_metrics["bias_mm"]
            if corrected_metrics["bias_mm"] is not None and raw_metrics["bias_mm"] is not None
            else None
        ),
        "worst_condition_id": "" if worst is None else worst["condition_id"],
        "worst_condition_abs_corrected_bias_mm": (
            None if worst is None else abs(float(worst["corrected_condition_bias_mm"]))
        ),
    }


def full_replay(data: dict[str, Any], beta: np.ndarray, q2_domain: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q2_lo = float(q2_domain["development_hard_min"])
    q2_hi = float(q2_domain["development_hard_max"])
    rows: list[dict[str, Any]] = []
    for condition in data["conditions"].values():
        population = "development" if condition.nominal_height_mm != 50.0 else "strict_50mm"
        rows.append(condition_replay_row(condition, beta, q2_lo, q2_hi, population, "raw_formula"))
    aggregate: list[dict[str, Any]] = []
    aggregate.append(aggregate_condition_rows(rows, "pooled", "development_pooled", "development"))
    for height in DEV_HEIGHTS:
        aggregate.append(aggregate_condition_rows(rows, "height", f"{height:g}", "development"))
    for rank in range(1, 6):
        aggregate.append(aggregate_condition_rows(rows, "position_rank", str(rank), "development"))
    aggregate.append(aggregate_condition_rows(rows, "strict_50mm", "50mm", "strict_50mm"))
    return rows, aggregate


def q2_domain_stats(data: dict[str, Any]) -> dict[str, Any]:
    development_q2 = np.concatenate([
        condition.formal.q2
        for condition in data["development"].values()
    ])
    strict_q2 = np.concatenate([
        condition.formal.q2
        for condition in data["strict_50"].values()
    ])
    dev_stats = stats(development_q2)
    strict_stats = stats(strict_q2)
    return {
        "coordinate_name": "q2",
        "coordinate_definition": "Frozen-C0 intrinsic coordinate from P_c0; no height/position-specific redefinition",
        "development_formal_repeat2_5": dev_stats,
        "strict_50mm_diagnostic_only": strict_stats,
        "development_hard_min": dev_stats["min"],
        "development_hard_max": dev_stats["max"],
        "development_robust_p05": dev_stats["p05"],
        "development_robust_p95": dev_stats["p95"],
        "strict_50_outside_hard_domain_rate": (
            float(np.mean((strict_q2 < dev_stats["min"] - EPS) | (strict_q2 > dev_stats["max"] + EPS)))
            if len(strict_q2) else None
        ),
        "strict_50_outside_hard_domain_count": int(
            np.count_nonzero((strict_q2 < dev_stats["min"] - EPS) | (strict_q2 > dev_stats["max"] + EPS))
        ),
        "domain_source": "development_only_1_2_6_10_20_30_36_40_46mm_formal_points",
        "strict_50_used_for_domain": False,
        "unbounded_extrapolation": False,
    }


def domain_strategy_rows(
    data: dict[str, Any],
    beta: np.ndarray,
    domain: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population, conditions in (
        ("development", data["development"]),
        ("strict_50mm", data["strict_50"]),
    ):
        for strategy in ("clamp", "reject"):
            condition_rows = [
                condition_replay_row(
                    condition,
                    beta,
                    float(domain["development_hard_min"]),
                    float(domain["development_hard_max"]),
                    population,
                    strategy,
                )
                for condition in conditions.values()
            ]
            aggregate = aggregate_condition_rows(
                condition_rows,
                "strategy",
                strategy,
                population,
                strategy,
            )
            rows.append({
                "evaluation_scope": "q2_domain_strategy_diagnostic",
                "population": population,
                "strategy": strategy,
                "domain_min": domain["development_hard_min"],
                "domain_max": domain["development_hard_max"],
                "condition_count_total": len(condition_rows),
                "condition_count_evaluated": aggregate.get("condition_count", 0),
                "point_count_total": aggregate.get("point_count_total", 0),
                "point_count_evaluated": aggregate.get("point_count_evaluated", 0),
                "out_of_domain_point_count": aggregate.get("out_of_domain_point_count", 0),
                "out_of_domain_rate": aggregate.get("out_of_domain_rate"),
                "clamped_point_count": aggregate.get("clamped_point_count", 0),
                "raw_bias_mm": aggregate.get("raw_bias_mm"),
                "raw_mae_mm": aggregate.get("raw_mae_mm"),
                "raw_rmse_mm": aggregate.get("raw_rmse_mm"),
                "raw_p95_abs_mm": aggregate.get("raw_p95_abs_mm"),
                "raw_max_abs_mm": aggregate.get("raw_max_abs_mm"),
                "corrected_bias_mm": aggregate.get("corrected_bias_mm"),
                "corrected_mae_mm": aggregate.get("corrected_mae_mm"),
                "corrected_rmse_mm": aggregate.get("corrected_rmse_mm"),
                "corrected_p95_abs_mm": aggregate.get("corrected_p95_abs_mm"),
                "corrected_max_abs_mm": aggregate.get("corrected_max_abs_mm"),
                "interpretation": (
                    "development protocol check; no out-of-domain points expected"
                    if population == "development"
                    else "strict-50 diagnostic only; not used to change domain or parameters"
                ),
            })
    return rows


def frozen_cv_replay(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = data["model_paths"]["model_selection_cv_metrics"]
    source_hash = sha256(path)
    source_rows = read_csv(path)
    b2_rows = [row for row in source_rows if row.get("model") == "B2"]
    expected_groups = {
        "LOHO_height": 9,
        "LOPO_position_rank": 5,
        "LOBO_height_band": 3,
        "strict_50mm_validation": 1,
    }
    if len(b2_rows) != 18:
        raise RuntimeError(f"expected 18 frozen H-B2 CV rows, got {len(b2_rows)}")
    output: list[dict[str, Any]] = []
    for row in b2_rows:
        output.append({
            "evaluation_scope": "frozen_cv_reuse",
            "source": "surface2_model_selection_cv_metrics.csv",
            "source_sha256": source_hash,
            "model": "H-B2",
            "cv_scheme": row["cv_scheme"],
            "heldout_group": row["heldout_group"],
            "population": "strict_50mm" if row["cv_scheme"] == "strict_50mm_validation" else "development",
            "support_state": row["support_state"],
            "condition_count": int(float(row["test_condition_count"])),
            "point_count": int(float(row["test_point_count"])),
            "raw_bias_mm": number(row["raw_bias_mm"]),
            "raw_mae_mm": number(row["raw_mae_mm"]),
            "raw_rmse_mm": number(row["raw_rmse_mm"]),
            "raw_p95_abs_mm": number(row["raw_p95_abs_mm"]),
            "raw_max_abs_mm": number(row["raw_max_abs_mm"]),
            "corrected_bias_mm": number(row["bias_mm"]),
            "corrected_mae_mm": number(row["mae_mm"]),
            "corrected_rmse_mm": number(row["rmse_mm"]),
            "corrected_p95_abs_mm": number(row["p95_abs_mm"]),
            "corrected_max_abs_mm": number(row["max_abs_mm"]),
            "worst_condition_id": None,
            "worst_condition_abs_corrected_bias_mm": None,
        })
    predictions = read_csv(data["model_paths"]["model_selection_predictions"])
    b2_predictions = [
        row for row in predictions
        if row.get("model") == "B2"
    ]
    pooled: list[dict[str, Any]] = []
    for scheme in expected_groups:
        selected = [
            row for row in b2_predictions
            if row.get("cv_scheme") == scheme
        ]
        if not selected:
            raise RuntimeError(f"missing frozen H-B2 predictions for {scheme}")
        raw = np.asarray([number(row["raw_bias_mm"]) for row in selected], dtype=np.float64)
        corrected = np.asarray([number(row["corrected_bias_mm"]) for row in selected], dtype=np.float64)
        pooled.append({
            "evaluation_scope": "frozen_cv_reuse_pooled",
            "source": "surface2_model_selection_condition_predictions.csv",
            "source_sha256": sha256(data["model_paths"]["model_selection_predictions"]),
            "model": "H-B2",
            "cv_scheme": scheme,
            "heldout_group": "ALL_FOLDS",
            "population": "strict_50mm" if scheme == "strict_50mm_validation" else "development",
            "support_state": "ALL",
            "condition_count": len(selected),
            "point_count": int(sum(int(float(row["point_count"])) for row in selected)),
            **{f"raw_{key}": value for key, value in error_metrics(raw).items()},
            **{f"corrected_{key}": value for key, value in error_metrics(corrected).items()},
            "worst_condition_id": max(
                selected,
                key=lambda row: abs(number(row["corrected_bias_mm"])),
            )["condition_id"],
            "worst_condition_abs_corrected_bias_mm": max(
                abs(number(row["corrected_bias_mm"])) for row in selected
            ),
        })
    consistency = {
        "frozen_cv_row_count": len(b2_rows),
        "frozen_cv_scheme_counts": {
            scheme: sum(row["cv_scheme"] == scheme for row in b2_rows)
            for scheme in expected_groups
        },
        "frozen_cv_required_scheme_counts_match": all(
            sum(row["cv_scheme"] == scheme for row in b2_rows) == count
            for scheme, count in expected_groups.items()
        ),
        "frozen_cv_all_metrics_finite": all(
            all(finite(row[field]) for field in (
                "raw_bias_mm", "raw_mae_mm", "raw_rmse_mm", "raw_p95_abs_mm", "raw_max_abs_mm",
                "bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm",
            ))
            for row in b2_rows
        ),
        "selection_summary_still_B2": data["model_summary"].get("SELECTED_SURFACE_MODEL") == "B2",
        "strict_50_is_diagnostic_only": True,
    }
    if not all(consistency.values()):
        raise RuntimeError(f"frozen H-B2 CV consistency check failed: {consistency}")
    return output + pooled, output, consistency


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NA"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(result) else f"{result:.{digits}f}"


def candidate_report(
    candidate: dict[str, Any],
    domain: dict[str, Any],
    cv_pooled: list[dict[str, Any]],
    full_aggregate: list[dict[str, Any]],
    strategy_rows: list[dict[str, Any]],
    statuses: dict[str, Any],
    reuse_audit: dict[str, Any],
) -> str:
    cv_lines = []
    for row in cv_pooled:
        if row["evaluation_scope"] != "frozen_cv_reuse_pooled":
            continue
        cv_lines.append(
            f"| {row['cv_scheme']} | {row['condition_count']} | "
            f"{fmt(row['raw_bias_mm'])} | {fmt(row['raw_mae_mm'])} | "
            f"{fmt(row['raw_rmse_mm'])} | {fmt(row['raw_p95_abs_mm'])} | "
            f"{fmt(row['corrected_bias_mm'])} | {fmt(row['corrected_mae_mm'])} | "
            f"{fmt(row['corrected_rmse_mm'])} | {fmt(row['corrected_p95_abs_mm'])} | "
            f"{fmt(row['corrected_max_abs_mm'])} |"
        )
    full_lines = []
    for row in full_aggregate:
        if row.get("group_type") not in {"pooled", "strict_50mm"}:
            continue
        full_lines.append(
            f"| {row['group_type']} | {row['group']} | {row['condition_count']} | "
            f"{fmt(row.get('raw_bias_mm'))} | {fmt(row.get('raw_mae_mm'))} | "
            f"{fmt(row.get('raw_rmse_mm'))} | {fmt(row.get('raw_p95_abs_mm'))} | "
            f"{fmt(row.get('corrected_bias_mm'))} | {fmt(row.get('corrected_mae_mm'))} | "
            f"{fmt(row.get('corrected_rmse_mm'))} | {fmt(row.get('corrected_p95_abs_mm'))} | "
            f"{fmt(row.get('corrected_max_abs_mm'))} | {row.get('worst_condition_id')} |"
        )
    strategy_lines = []
    for row in strategy_rows:
        if row["population"] != "strict_50mm":
            continue
        strategy_lines.append(
            f"| {row['strategy']} | {row['point_count_total']} | "
            f"{row['point_count_evaluated']} | {row['out_of_domain_point_count']} | "
            f"{fmt(row['out_of_domain_rate'], 4)} | {fmt(row['corrected_rmse_mm'])} | "
            f"{fmt(row['corrected_p95_abs_mm'])} |"
        )
    return f"""# Surface-3 H-B2 frozen candidate audit

## Status

HB2_CANDIDATE_FREEZE={statuses['HB2_CANDIDATE_FREEZE']}

HB2_PRODUCTION_ACCEPTED={statuses['HB2_PRODUCTION_ACCEPTED']}

IMMEDIATE_MORE_HEIGHT_ACQUISITION_REQUIRED={statuses['IMMEDIATE_MORE_HEIGHT_ACQUISITION_REQUIRED']}

MORE_DOMAIN_COVERAGE_REQUIRED_FOR_PRODUCTION={statuses['MORE_DOMAIN_COVERAGE_REQUIRED_FOR_PRODUCTION']}

UNTOUCHED_ENGINEERING_VALIDATION_REQUIRED={statuses['UNTOUCHED_ENGINEERING_VALIDATION_REQUIRED']}

Candidate freeze is a reproducibility decision only. It is not a production
acceptance decision. Historical Q2_GAP_FILLED=NO and SURFACE2C_ALLOWED=NO are
preserved; they do not invalidate this candidate freeze and do not authorize
online enablement.

IMMEDIATE_MORE_HEIGHT_ACQUISITION_REQUIRED=YES is a production/domain-coverage
recommendation inherited from the grouped evidence; it is not a precondition
for freezing this candidate, and no new data were acquired in this round.

## Frozen candidate

- Model: H-B2, height layer, q1 excluded.
- Formula: delta_h_mm = a0 + a2 * q2.
- a0 = {fmt(candidate['parameters']['a0'], 12)} mm.
- a2 = {fmt(candidate['parameters']['a2'], 12)} mm per Frozen-C0 q2.
- Runtime formula: h_corrected = h_raw - delta_h_mm.
- Parameter SHA256: {candidate['parameter_sha256']}.
- Candidate SHA256: {candidate['candidate_sha256']}.
- Training: 44 development conditions, 11160 formal points,
  heights 1/2/6/10/20/30/36/40/46 mm.
- 50 mm was excluded from fitting, parameter selection and domain definition.

## Runtime semantics

1. Run the existing one-pass Steger, Frozen C0 and Frozen C1 reconstruction.
2. Fit the session-linear ground proxy exactly as the existing protocol does.
3. Compute Frozen-C0 intrinsic q2 from P_c0; do not define q2 from C1 points
   or from height/position-specific coordinates.
4. Apply h_corrected = h_raw - (a0 + a2*q2).
5. Do not extrapolate outside the frozen q2 domain. The recommended engineering
   behavior is reject plus an explicit out-of-domain flag. Explicit clamping
   is retained only as a diagnostic fallback and must also emit a CLAMPED flag.

No C0/C1, ROI, q2, Steger, Ground G(S), H1 or online production configuration
was changed.

## q2 validity domain

The hard domain is frozen from development formal repeat2-5 points only:

- min = {fmt(domain['development_hard_min'], 9)}
- P05 = {fmt(domain['development_robust_p05'], 9)}
- P95 = {fmt(domain['development_robust_p95'], 9)}
- max = {fmt(domain['development_hard_max'], 9)}

The P05/P95 interval is a robust reference, not a 50 mm-tuned replacement for
the hard observed domain. The 50 mm statistics below are diagnostic only:

- 50 mm P05/P95 = {fmt(domain['strict_50mm_diagnostic_only']['p05'], 9)} /
  {fmt(domain['strict_50mm_diagnostic_only']['p95'], 9)}
- 50 mm outside hard-domain rate =
  {fmt(domain['strict_50_outside_hard_domain_rate'], 6)}
- 50 mm outside hard-domain points =
  {domain['strict_50_outside_hard_domain_count']}

| strategy | population | total points | evaluated | out-of-domain | rate | corrected RMSE | corrected P95 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(strategy_lines)}

Neither strategy was selected using 50 mm. Reject and clamp are runtime policy
designs to be evaluated in engineering validation; neither is an unbounded
extrapolation policy.

## Frozen H-B2 grouped CV replay

These rows are read-only reuse of the canonical Surface-2 model-selection
artifact, not a new parameter search. Each development fold had its own
training fit in the original audit; the candidate itself is fit once on all
development conditions for later engineering use.

| scheme | conditions | raw Bias | raw MAE | raw RMSE | raw P95 | corrected Bias | corrected MAE | corrected RMSE | corrected P95 | corrected Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cv_lines)}

The replay contains LOHO, LOPO and LOBO development folds plus a separate
strict 50 mm diagnostic. The source artifact remains B2-selected and all
required H-B2 metrics are finite.

## Full-candidate offline replay

This is a retrospective application of the one frozen full-development
candidate. Development rows are in-sample and therefore are not engineering
acceptance evidence. The 50 mm row is strict diagnostic only.
Its corrected value uses the historical raw-formula replay for comparability;
it is not an operational out-of-domain result and does not authorize
unbounded extrapolation.

| group type | group | conditions | raw Bias | raw MAE | raw RMSE | raw P95 | corrected Bias | corrected MAE | corrected RMSE | corrected P95 | corrected Max | worst condition |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(full_lines)}

Position-rank and per-height details are in the offline replay CSV. Position
rank is the existing q1-spatial rank; raw pose_id is not treated as a
cross-height coordinate.

## Provenance and reuse audit

- Current Frozen-C0 SHA256: {candidate['provenance']['frozen_c0_sha256']}
- Current Frozen-C1 SHA256: {candidate['provenance']['frozen_c1_sha256']}
- Current config SHA256: {candidate['provenance']['config_sha256']}
- Freeze script SHA256: {candidate['provenance']['freeze_script_sha256']}
- Base replay code SHA256: {candidate['provenance']['base_replay_script_sha256']}
- Git commit: {candidate['provenance'].get('git_commit') or 'unavailable'}
- Model-selection input SHA match: {reuse_audit['model_input_sha_match']}
- Lambda-audit Frozen C0/C1 SHA match: {reuse_audit['lambda_frozen_match']}
- Raw residual replay maximum difference: {fmt(reuse_audit['raw_replay_max_difference_mm'], 12)} mm.

The previous Surface-2 and lambda-gauge outputs were only read. No historical
file was overwritten.

## Decision boundary

The candidate can be frozen because the selected model, layer, q2 semantics,
full-development fit and provenance are internally consistent. Production
enablement remains NO because independent new-session/new-standard full-FOV
validation, zero unsupported-domain operation, repeatability and the
predefined error gates have not yet been demonstrated.
"""


def acceptance_plan(candidate: dict[str, Any], domain: dict[str, Any], statuses: dict[str, Any]) -> str:
    return f"""# Surface-3 H-B2 engineering acceptance plan

## Scope

This plan validates the frozen diagnostic candidate only. It does not change
C0/C1, q2, ROI, Steger, Ground G(S), H1 or the online production chain.

- Candidate: H-B2 height-layer correction
- Formula: h_corrected = h_raw - (a0 + a2*q2)
- a0 = {fmt(candidate['parameters']['a0'], 12)} mm
- a2 = {fmt(candidate['parameters']['a2'], 12)} mm/q2
- Candidate SHA256: {candidate['candidate_sha256']}
- Frozen q2 hard domain: [{fmt(domain['development_hard_min'], 9)},
  {fmt(domain['development_hard_max'], 9)}]

Current status:

- HB2_CANDIDATE_FREEZE={statuses['HB2_CANDIDATE_FREEZE']}
- HB2_PRODUCTION_ACCEPTED=NO
- IMMEDIATE_MORE_HEIGHT_ACQUISITION_REQUIRED={statuses['IMMEDIATE_MORE_HEIGHT_ACQUISITION_REQUIRED']}
- MORE_DOMAIN_COVERAGE_REQUIRED_FOR_PRODUCTION=YES
- UNTOUCHED_ENGINEERING_VALIDATION_REQUIRED=YES

## Required independent data

Use a new acquisition session and new standard artifacts that were not used in
Surface-2, Ground-4A, Height-1/2, or this candidate fit. Cover the complete
operating FOV with the existing spatial ranks 1-5, and include heights across
the candidate domain plus points near both q2 boundaries. The current historical
q2 gap remains a production-coverage risk; intermediate heights should be
chosen from measured q2 coverage, not from residual-driven tuning.

Use the same exposure, gain, laser power, mechanical state, C0/C1 files and
one-pass Steger protocol. If the engineering campaign needs intermediate
heights to fill the domain gap, acquire them as a separate, untouched
engineering dataset; do not alter this candidate from their results.

Minimum structure per height and spatial position:

- repeat1: ground-only session proxy input;
- repeat2-5: formal measurement repeats;
- all five spatial positions and all accepted points in the manual geometric
  ROI;
- no residual-based ROI editing and no random point split.

## Data roles and leakage control

Allowed calibration data:

- A designated repeat1 ground-only frame from the new session may be used to
  fit the existing session-linear ground proxy for that session/condition.
- It may be used to verify C1 clamp/invalid status and the mechanical/session
  setup.
- It may not refit a0/a2, redefine q2, change the hard q2 domain, edit C0/C1,
  or select a different model.

Untouched validation data:

- All formal repeat2-5 points from every new height and FOV position.
- The new standard artifacts and their true heights must remain hidden from
  any candidate/domain/model adjustment until the acceptance report is
  locked.
- If a new diagnostic height is used for domain coverage, it remains
  validation data for this candidate and cannot be used to expand the domain
  retrospectively.

If a failure causes model or domain changes, that is a new candidate version
and requires a new untouched validation campaign.

## Pre-registered metrics

For every formal point, define e = h_corrected - true_height_mm. Report pooled,
per height, per position-rank, per session, and worst-condition metrics:

- Bias = mean(e)
- MAE = mean(abs(e))
- RMSE = sqrt(mean(e^2))
- P95 = 95th percentile of abs(e)
- Max = max(abs(e))
- adjacent-height difference MAE: for each adjacent true-height pair at the
  same FOV position, MAE of
  (measured_height_next - measured_height_prev)
  - (true_height_next - true_height_prev)
- out-of-domain rate =
  count(q2 outside [{fmt(domain['development_hard_min'], 9)},
  {fmt(domain['development_hard_max'], 9)}]) / formal point count
- repeatability: within-condition repeat standard deviation/range over
  repeat2-5 after applying the frozen candidate.

Also report invalid-point count, C1 clamp count/rate, manual ROI completeness,
and the number of conditions/points included in every metric.

## Acceptance gates

Target gate, required for production consideration:

- absolute pooled Bias <= 0.2 mm;
- pooled MAE, RMSE, P95 and Max <= 0.2 mm;
- every height and every FOV position has absolute condition Bias <= 0.2 mm;
- adjacent-height difference MAE <= 0.2 mm;
- out-of-domain rate = 0;
- no unexplained new invalid points or C1 clamp-state changes;
- repeatability is reported for every condition and has no unreviewed
  outlier condition.

Expected engineering level, stronger than the target:

- absolute pooled Bias, MAE, RMSE and P95 approximately <= 0.1 mm;
- Max and adjacent-height difference MAE approximately <= 0.1 mm;
- the same approximately 0.1 mm behavior across all FOV positions and
  sessions, not only in pooled averages.

The approximately 0.1 mm values are an expectation, not a retroactively
relaxed pass/fail threshold. A result between 0.1 and 0.2 mm can meet the
target gate while still being below expectation and requiring engineering
review.

## Decision procedure

1. Freeze the candidate JSON, parameter SHA, candidate SHA and input/code SHA
   before opening any formal validation results.
2. Verify dataset completeness, true-height traceability, image dimensions,
   manual ROI registry and one-pass Steger provenance.
3. Run the unchanged reconstruction and session-linear ground proxy.
4. Apply only the frozen candidate. Use reject plus explicit
   OUT_OF_DOMAIN flag as the primary policy. A clamp replay may be shown as a
   diagnostic comparison, but must not silently turn unsupported q2 into a
   production result.
5. Compute all pre-registered metrics by session, height, position and
   pooled/worst-condition group.
6. Keep the formal validation dataset untouched until the report is signed.
7. Enable production only after the target gate, zero unsupported-domain
   operation, independent-session repeatability, and review of all
   worst/boundary conditions are satisfied.

Therefore this plan intentionally ends with HB2_PRODUCTION_ACCEPTED=NO for the
current evidence. It is an acceptance design, not a production PASS.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_artifacts(args.config.resolve())
    full_fit = fit_full_candidate(data)
    beta = np.asarray([full_fit["a0"], full_fit["a2"]], dtype=np.float64)
    domain = q2_domain_stats(data)
    if domain["development_hard_min"] is None or domain["development_hard_max"] is None:
        raise RuntimeError("development q2 domain is empty")

    parameter_core = {
        "candidate_id": "surface3_hb2",
        "model": "H-B2",
        "correction_layer": "HEIGHT",
        "q1_retained": "NO",
        "formula": "delta_h_mm = a0 + a2*q2",
        "runtime_formula": "h_corrected_mm = h_raw_mm - delta_h_mm",
        "a0_mm": full_fit["a0"],
        "a2_mm_per_q2": full_fit["a2"],
        "train_condition_count": full_fit["train_condition_count"],
        "train_point_count": full_fit["train_point_count"],
        "strict_50_excluded": True,
    }
    parameter_sha = sha256_text(canonical_json(parameter_core))
    code_sha = {
        "freeze_script_sha256": sha256(Path(__file__)),
        "base_replay_script_sha256": sha256(TOOLS / "compare_surface_correction_layers.py"),
        "surface_model_selection_script_sha256": sha256(TOOLS / "select_surface_model.py"),
    }
    training_data_names = {
        name: digest
        for name, digest in data["source_sha"].items()
        if name in {
            "surface1a_points",
            "surface2b_samples",
            "surface2br2_condition_table",
            "pointwise_diagnostics",
            "surface2_roi",
            "height50_roi",
        }
    }
    training_data_sha = sha256_text(canonical_json(training_data_names))
    provenance = {
        "git_commit": git_commit(),
        **code_sha,
        "config_sha256": data["source_sha"]["config"],
        "frozen_c0_sha256": data["source_sha"]["quadratic_c0"],
        "frozen_c1_sha256": data["source_sha"]["frozen_c1"],
        "intrinsics_sha256": data["source_sha"]["intrinsics"],
        "extrinsics_sha256": data["source_sha"]["extrinsics"],
        "training_data_sha256": training_data_sha,
        "training_data_file_sha256": training_data_names,
        "all_source_file_sha256": data["source_sha"],
        "model_selection_input_sha_match": data["model_input_sha_match"],
        "lambda_audit_frozen_c0_c1_sha_match": data["lambda_frozen_match"],
        "model_selection_summary_sha256": data["source_sha"]["model_selection_summary"],
        "lambda_audit_summary_sha256": data["source_sha"]["lambda_summary"],
    }
    candidate_core = {
        "schema_version": "surface3_hb2_candidate_v1",
        "candidate": parameter_core,
        "parameters": {
            "a0": full_fit["a0"],
            "a2": full_fit["a2"],
            "parameter_sha256": parameter_sha,
            "fit_status": full_fit["fit"]["fit_status"],
            "design_rank": full_fit["fit"]["design_rank"],
            "design_condition_number": full_fit["fit"]["design_condition_number"],
            "singular_values": [float(value) for value in full_fit["fit"]["singular_values"]],
            "objective": full_fit["fit"]["objective"],
        },
        "q2_validity_domain": {
            **domain,
            "default_policy": "REJECT_OUT_OF_DOMAIN_FLAG",
            "clamp_policy": "EXPLICIT_CLAMP_AND_FLAG_DIAGNOSTIC_ONLY",
            "reject_policy": "UNSUPPORTED_OUT_OF_DOMAIN_NO_CORRECTED_RESULT",
        },
        "runtime_semantics": {
            "q2_definition": "Frozen-C0 intrinsic coordinate derived from P_c0",
            "c0_c1_unchanged": True,
            "session_linear_ground_proxy_unchanged": True,
            "steger_unchanged": True,
            "roi_unchanged": True,
            "no_unbounded_extrapolation": True,
            "height_formula": "h_corrected = h_raw - (a0 + a2*q2)",
        },
        "provenance": provenance,
        "historical_conclusions_preserved": {
            "SELECTED_SURFACE_MODEL": "B2",
            "Q1_RETAINED": "NO",
            "FINAL_CORRECTION_LAYER": "HEIGHT",
            "STOP_LAMBDA_ROUTE": "YES",
            "Q2_GAP_FILLED": "NO",
            "SURFACE2C_ALLOWED": "NO",
        },
    }
    candidate_sha = sha256_text(canonical_json(candidate_core))
    candidate = {
        **candidate_core,
        "parameter_sha256": parameter_sha,
        "candidate_sha256": candidate_sha,
        "created_at_utc": now_utc(),
    }

    condition_rows, full_aggregate = full_replay(data, beta, domain)
    strategy_rows = domain_strategy_rows(data, beta, domain)
    cv_rows, cv_fold_rows, cv_consistency = frozen_cv_replay(data)
    cv_pooled = [
        row for row in cv_rows
        if row["evaluation_scope"] == "frozen_cv_reuse_pooled"
    ]
    statuses = {
        "HB2_CANDIDATE_FREEZE": "YES",
        "HB2_PRODUCTION_ACCEPTED": "NO",
        "IMMEDIATE_MORE_HEIGHT_ACQUISITION_REQUIRED": (
            "YES"
            if data["model_summary"].get("MORE_HEIGHT_ACQUISITION_REQUIRED") == "YES"
            else "NO"
        ),
        "MORE_DOMAIN_COVERAGE_REQUIRED_FOR_PRODUCTION": "YES",
        "UNTOUCHED_ENGINEERING_VALIDATION_REQUIRED": "YES",
    }
    reuse_audit = {
        "model_input_sha_match": data["model_input_sha_match"],
        "lambda_frozen_match": data["lambda_frozen_match"],
        "raw_replay_max_difference_mm": max(
            row["max_abs_raw_residual_replay_difference_mm"]
            for row in data["consistency"]
        ),
        "frozen_cv_consistency": cv_consistency,
    }
    summary = {
        "status": statuses,
        "candidate_sha256": candidate_sha,
        "parameter_sha256": parameter_sha,
        "fit": {
            "a0_mm": full_fit["a0"],
            "a2_mm_per_q2": full_fit["a2"],
            "train_condition_count": full_fit["train_condition_count"],
            "train_point_count": full_fit["train_point_count"],
            "design_rank": full_fit["fit"]["design_rank"],
            "design_condition_number": full_fit["fit"]["design_condition_number"],
            "objective": full_fit["fit"]["objective"],
        },
        "q2_domain": domain,
        "replay": {
            "frozen_cv_fold_count": len(cv_fold_rows),
            "frozen_cv_pooled_count": len(cv_pooled),
            "full_candidate_condition_row_count": len(condition_rows),
            "full_candidate_aggregate_row_count": len(full_aggregate),
            "q2_strategy_row_count": len(strategy_rows),
        },
        "provenance": provenance,
        "reuse_audit": reuse_audit,
        "historical_conclusions_preserved": candidate["historical_conclusions_preserved"],
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "surface3_hb2_candidate.json", candidate)
    write_json(output / "surface3_hb2_parameters.json", {
        "model": "H-B2",
        "layer": "HEIGHT",
        "a0_mm": full_fit["a0"],
        "a2_mm_per_q2": full_fit["a2"],
        "q2_domain_min": domain["development_hard_min"],
        "q2_domain_max": domain["development_hard_max"],
        "parameter_sha256": parameter_sha,
        "candidate_sha256": candidate_sha,
    })
    write_json(output / "surface3_hb2_q2_domain.json", domain)
    write_json(output / "surface3_hb2_provenance.json", provenance)
    write_json(output / "surface3_hb2_replay_summary.json", summary)
    write_csv(output / "surface3_hb2_offline_replay_metrics.csv", cv_rows + full_aggregate)
    write_csv(output / "surface3_hb2_offline_replay_conditions.csv", condition_rows)
    write_csv(output / "surface3_hb2_q2_domain_strategies.csv", strategy_rows)
    (output / "surface3_hb2_candidate_report.md").write_text(
        candidate_report(
            candidate,
            domain,
            cv_pooled,
            full_aggregate,
            strategy_rows,
            statuses,
            reuse_audit,
        ),
        encoding="utf-8",
    )
    (output / "surface3_engineering_acceptance_plan.md").write_text(
        acceptance_plan(candidate, domain, statuses),
        encoding="utf-8",
    )
    print(json.dumps({
        **statuses,
        "a0_mm": full_fit["a0"],
        "a2_mm_per_q2": full_fit["a2"],
        "q2_domain_min": domain["development_hard_min"],
        "q2_domain_max": domain["development_hard_max"],
        "candidate_sha256": candidate_sha,
        "parameter_sha256": parameter_sha,
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
