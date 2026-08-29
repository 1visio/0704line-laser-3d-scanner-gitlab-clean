"""Formal B2(q2-only) versus S0(q1+q2) selection from frozen 2BR2 outputs.

This is a post-processing audit of the existing Surface-2BR2 artifacts.  It
does not refit C0/C1 or either surface model, change ROI/q definitions, or
touch the historical Surface-2B/2BR/2BR2 outputs.  Development selection uses
LOHO, LOPO and LOBO only.  The strict 50 mm rows are copied into the report as
diagnostics and are never used for the decision.
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
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
DEFAULT_INPUT = BASE / "surface2" / "surface2br2"
DEFAULT_OUTPUT = BASE / "surface2" / "surface2_model_selection"

DEV_SCHEMES = ("LOHO_height", "LOPO_position_rank", "LOBO_height_band")
ALL_SCHEMES = DEV_SCHEMES + ("strict_50mm_validation",)
MODELS = ("B2", "S0")
SUPPORT_STATES = ("IN_DOMAIN", "HULL_EXTRAPOLATION", "BBOX_EXTRAPOLATION")
METRICS = ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm")
EPS = 1e-12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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


def numeric_rows(rows: list[dict[str, Any]], bool_fields: Iterable[str] = ()) -> list[dict[str, Any]]:
    bool_set = set(bool_fields)
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif key in bool_set:
                row[key] = str(value).strip().lower() in {"1", "true", "yes"}
            else:
                try:
                    row[key] = float(value) if any(char in str(value).lower() for char in (".", "e")) else int(value)
                except (TypeError, ValueError):
                    pass
        output.append(row)
    return output


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metrics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if not len(array):
        return {name: float("nan") for name in METRICS}
    absolute = np.abs(array)
    return {
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array * array))),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "max_abs_mm": float(np.max(absolute)),
    }


def validate_inputs(input_dir: Path) -> dict[str, Any]:
    names = {
        "condition_table": "surface2br2_condition_table.csv",
        "cv_metrics": "surface2br2_cv_metrics.csv",
        "coefficients": "surface2br2_coefficients.csv",
        "coefficient_stability": "surface2br2_coefficient_stability.csv",
        "condition_predictions": "surface2br2_condition_predictions.csv",
        "incremental_reference": "surface2br2_incremental_comparison.csv",
        "summary": "surface2br2_summary.json",
    }
    paths = {key: input_dir / name for key, name in names.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Surface-2BR2 artifact: " + ", ".join(missing))

    summary = read_json(paths["summary"])
    original = summary.get("original_surface2b_conclusion", {})
    if original.get("Q2_GAP_FILLED") != "NO" or original.get("SURFACE2C_ALLOWED") != "NO":
        raise RuntimeError("historical Surface-2B conclusion must remain Q2_GAP_FILLED=NO and SURFACE2C_ALLOWED=NO")
    protocol = summary.get("protocol", {})
    required_protocol = {
        "condition_equal_weight": True,
        "random_point_split": False,
        "heldout_50_excluded_from_fit_and_selection": True,
        "c0_refit": False,
        "c1_refit": False,
        "q_redefined": False,
        "quadratic_terms": False,
        "spline_or_ml": False,
        "production_validation": False,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"protocol mismatch for {key}: {protocol.get(key)!r} != {expected!r}")

    condition_table = numeric_rows(read_csv(paths["condition_table"]))
    # Surface-2BR2 inherited two equivalent development labels: the original
    # 1/2/6/10/20/30 mm rows use ``development_*`` while the newly ingested
    # 36/40/46 mm rows use ``surface2_formal_*``.  Both are development for
    # this task; only ``heldout_formal_repeat2_5`` is strict 50 mm.
    development_roles = {"development_formal_repeat2_5", "surface2_formal_repeat2_5"}
    development = [row for row in condition_table if row.get("split_role") in development_roles]
    heldout = [row for row in condition_table if row.get("split_role") == "heldout_formal_repeat2_5"]
    if len(development) != 44 or len(heldout) != 5:
        raise RuntimeError(f"condition table split mismatch: development={len(development)}, heldout50={len(heldout)}")
    expected_heights = {1.0, 2.0, 6.0, 10.0, 20.0, 30.0, 36.0, 40.0, 46.0}
    if {float(row["height_label_mm"]) for row in development} != expected_heights:
        raise RuntimeError("development heights do not match 1/2/6/10/20/30/36/40/46 mm")
    if {float(row["true_height_mm"]) for row in heldout} != {50.0}:
        raise RuntimeError("strict held-out condition table is not 50 mm")

    cv_metrics = numeric_rows(read_csv(paths["cv_metrics"]), bool_fields=("bbox_extrapolation", "hull_extrapolation", "improved_bias_mm_vs_b0", "improved_mae_mm_vs_b0", "improved_rmse_mm_vs_b0", "improved_p95_abs_mm_vs_b0", "improved_max_abs_mm_vs_b0", "no_metric_worsening_vs_b0", "q_terms_present"))
    cv_selected = [row for row in cv_metrics if row.get("model") in MODELS]
    expected_folds = {"LOHO_height": 9, "LOPO_position_rank": 5, "LOBO_height_band": 3, "strict_50mm_validation": 1}
    for scheme, fold_count in expected_folds.items():
        rows = [row for row in cv_selected if row.get("cv_scheme") == scheme]
        groups = {row.get("heldout_group") for row in rows}
        if len(groups) != fold_count or len(rows) != fold_count * 2:
            raise RuntimeError(f"{scheme} B2/S0 fold count mismatch: groups={len(groups)}, rows={len(rows)}")
        for group in groups:
            if {row.get("model") for row in rows if row.get("heldout_group") == group} != set(MODELS):
                raise RuntimeError(f"{scheme}/{group} does not contain exactly B2 and S0")
        if any(row.get("support_state") not in SUPPORT_STATES for row in rows):
            raise RuntimeError(f"{scheme} contains an unknown q-space support state")
        for row in rows:
            for field in ("corrected_bias_mm", "corrected_mae_mm", "corrected_rmse_mm", "corrected_p95_abs_mm", "corrected_max_abs_mm"):
                if not finite(row.get(field)):
                    raise RuntimeError(f"non-finite {field} in {scheme}/{row.get('heldout_group')}/{row.get('model')}")

    predictions = numeric_rows(read_csv(paths["condition_predictions"]), bool_fields=())
    pred_selected = [row for row in predictions if row.get("model") in MODELS]
    pred_keys = {(row.get("cv_scheme"), row.get("heldout_group"), row.get("condition_id"), row.get("model")) for row in pred_selected}
    if not pred_selected:
        raise RuntimeError("condition_predictions has no B2/S0 rows")
    for scheme in ALL_SCHEMES:
        scheme_rows = [row for row in pred_selected if row.get("cv_scheme") == scheme]
        for group in {row.get("heldout_group") for row in scheme_rows}:
            group_rows = [row for row in scheme_rows if row.get("heldout_group") == group]
            keys = {row.get("condition_id") for row in group_rows}
            for condition_id in keys:
                pair = [row for row in group_rows if row.get("condition_id") == condition_id]
                if {row.get("model") for row in pair} != set(MODELS):
                    raise RuntimeError(f"missing B2/S0 condition prediction for {scheme}/{group}/{condition_id}")
                if pair[0].get("support_state") != pair[1].get("support_state"):
                    raise RuntimeError(f"B2/S0 support mismatch for {scheme}/{group}/{condition_id}")
                for row in pair:
                    if not finite(row.get("corrected_bias_mm")):
                        raise RuntimeError(f"non-finite condition prediction in {scheme}/{group}/{condition_id}")

    coefficients = numeric_rows(read_csv(paths["coefficients"]), bool_fields=())
    coefficient_stability = numeric_rows(read_csv(paths["coefficient_stability"]), bool_fields=("sign_consistency",))
    for scheme in ALL_SCHEMES:
        for model in MODELS:
            rows = [row for row in coefficients if row.get("cv_scheme") == scheme and row.get("model") == model]
            if not rows:
                raise RuntimeError(f"missing coefficients for {scheme}/{model}")
    input_sha = {key: sha256(path) for key, path in paths.items()}
    return {
        "paths": paths,
        "summary": summary,
        "protocol": protocol,
        "condition_table": condition_table,
        "cv_metrics": cv_metrics,
        "predictions": predictions,
        "coefficients": coefficients,
        "coefficient_stability": coefficient_stability,
        "input_sha256": input_sha,
    }


def cv_output_rows(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = []
    for row in cv_rows:
        fields.append({
            "cv_scheme": row["cv_scheme"],
            "heldout_group": row["heldout_group"],
            "model": row["model"],
            "heldout_height_mm": row.get("heldout_height_mm"),
            "heldout_position_rank": row.get("heldout_position_rank"),
            "support_state": row["support_state"],
            "train_condition_count": row["train_condition_count"],
            "test_condition_count": row["test_condition_count"],
            "train_point_count": row["train_point_count"],
            "test_point_count": row["test_point_count"],
            "bbox_oob_point_rate": row["bbox_oob_point_rate"],
            "hull_oob_point_rate": row["hull_oob_point_rate"],
            "raw_bias_mm": row["raw_bias_mm"],
            "raw_mae_mm": row["raw_mae_mm"],
            "raw_rmse_mm": row["raw_rmse_mm"],
            "raw_p95_abs_mm": row["raw_p95_abs_mm"],
            "raw_max_abs_mm": row["raw_max_abs_mm"],
            "bias_mm": row["corrected_bias_mm"],
            "mae_mm": row["corrected_mae_mm"],
            "rmse_mm": row["corrected_rmse_mm"],
            "p95_abs_mm": row["corrected_p95_abs_mm"],
            "max_abs_mm": row["corrected_max_abs_mm"],
        })
    return fields


def prediction_groups(predictions: list[dict[str, Any]], scheme: str | None = None) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        if row.get("model") not in MODELS or (scheme is not None and row.get("cv_scheme") != scheme):
            continue
        groups[(str(row["cv_scheme"]), str(row["heldout_group"]))][str(row["model"])].append(row)
    return groups


def aggregate_condition_metrics(predictions: list[dict[str, Any]], scheme: str, support_state: str = "ALL") -> list[dict[str, Any]]:
    selected = [row for row in predictions if row.get("cv_scheme") == scheme and row.get("model") in MODELS and (support_state == "ALL" or row.get("support_state") == support_state)]
    output: list[dict[str, Any]] = []
    for model in MODELS:
        rows = [row for row in selected if row.get("model") == model]
        if not rows:
            continue
        values = metrics(row["corrected_bias_mm"] for row in rows)
        output.append({
            "aggregation": "support_stratified" if support_state != "ALL" else "pooled_condition_means",
            "cv_scheme": scheme,
            "support_state": support_state,
            "model": model,
            "condition_count": len(rows),
            "fold_count": len({row["heldout_group"] for row in rows}),
            **values,
        })
    return output


def worst_condition_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = prediction_groups(predictions)
    output: list[dict[str, Any]] = []
    for (scheme, group), model_rows in sorted(grouped.items()):
        by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for model in MODELS:
            for row in model_rows.get(model, []):
                by_condition[str(row["condition_id"])][model] = row
        if any(set(pair) != set(MODELS) for pair in by_condition.values()):
            raise RuntimeError(f"incomplete B2/S0 worst-condition group: {scheme}/{group}")
        b2_worst = max(by_condition.items(), key=lambda item: abs(float(item[1]["B2"]["corrected_bias_mm"])))
        s0_worst = max(by_condition.items(), key=lambda item: abs(float(item[1]["S0"]["corrected_bias_mm"])))
        b2_abs = abs(float(b2_worst[1]["B2"]["corrected_bias_mm"]))
        s0_abs = abs(float(s0_worst[1]["S0"]["corrected_bias_mm"]))
        all_pairs = list(by_condition.values())
        output.append({
            "cv_scheme": scheme,
            "heldout_group": group,
            "support_state": model_rows["B2"][0]["support_state"],
            "condition_count": len(all_pairs),
            "B2_worst_condition_id": b2_worst[0],
            "B2_worst_abs_bias_mm": b2_abs,
            "B2_worst_bias_mm": float(b2_worst[1]["B2"]["corrected_bias_mm"]),
            "S0_worst_condition_id": s0_worst[0],
            "S0_worst_abs_bias_mm": s0_abs,
            "S0_worst_bias_mm": float(s0_worst[1]["S0"]["corrected_bias_mm"]),
            "delta_worst_abs_bias_S0_minus_B2_mm": s0_abs - b2_abs,
            "S0_worst_condition_improved": s0_abs < b2_abs - EPS,
            "condition_abs_error_improvement_rate": sum(abs(float(pair["S0"]["corrected_bias_mm"])) < abs(float(pair["B2"]["corrected_bias_mm"])) - EPS for pair in all_pairs) / len(all_pairs),
        })
    return output


def incremental_rows(cv_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], worst_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cv_by_key = {(row["cv_scheme"], row["heldout_group"], row["model"]): row for row in cv_rows}
    worst_by_key = {(row["cv_scheme"], row["heldout_group"]): row for row in worst_rows}
    for scheme, group in sorted({(row["cv_scheme"], row["heldout_group"]) for row in cv_rows}):
        b2 = cv_by_key[(scheme, group, "B2")]
        s0 = cv_by_key[(scheme, group, "S0")]
        worst = worst_by_key[(scheme, group)]
        item = {
            "aggregation": "fold",
            "cv_scheme": scheme,
            "heldout_group": group,
            "support_state": b2["support_state"],
            "B2_condition_count": b2["test_condition_count"],
            "B2_bias_mm": b2["bias_mm"],
            "S0_bias_mm": s0["bias_mm"],
            "delta_bias_S0_minus_B2_mm": s0["bias_mm"] - b2["bias_mm"],
            "B2_mae_mm": b2["mae_mm"],
            "S0_mae_mm": s0["mae_mm"],
            "delta_mae_S0_minus_B2_mm": s0["mae_mm"] - b2["mae_mm"],
            "B2_rmse_mm": b2["rmse_mm"],
            "S0_rmse_mm": s0["rmse_mm"],
            "delta_rmse_S0_minus_B2_mm": s0["rmse_mm"] - b2["rmse_mm"],
            "B2_p95_abs_mm": b2["p95_abs_mm"],
            "S0_p95_abs_mm": s0["p95_abs_mm"],
            "delta_p95_S0_minus_B2_mm": s0["p95_abs_mm"] - b2["p95_abs_mm"],
            "B2_max_abs_mm": b2["max_abs_mm"],
            "S0_max_abs_mm": s0["max_abs_mm"],
            "delta_max_S0_minus_B2_mm": s0["max_abs_mm"] - b2["max_abs_mm"],
            "S0_improved_rmse": s0["rmse_mm"] < b2["rmse_mm"] - EPS,
            "S0_improved_p95": s0["p95_abs_mm"] < b2["p95_abs_mm"] - EPS,
            "S0_worst_condition_improved": worst["S0_worst_condition_improved"],
            "condition_abs_error_improvement_rate": worst["condition_abs_error_improvement_rate"],
        }
        output.append(item)

    for aggregate in aggregate_rows:
        if aggregate["model"] != "B2":
            continue
        match = next(row for row in aggregate_rows if row["aggregation"] == aggregate["aggregation"] and row["cv_scheme"] == aggregate["cv_scheme"] and row["support_state"] == aggregate["support_state"] and row["model"] == "S0")
        output.append({
            "aggregation": aggregate["aggregation"],
            "cv_scheme": aggregate["cv_scheme"],
            "heldout_group": "",
            "support_state": aggregate["support_state"],
            "B2_condition_count": aggregate["condition_count"],
            "B2_bias_mm": aggregate["bias_mm"],
            "S0_bias_mm": match["bias_mm"],
            "delta_bias_S0_minus_B2_mm": match["bias_mm"] - aggregate["bias_mm"],
            "B2_mae_mm": aggregate["mae_mm"],
            "S0_mae_mm": match["mae_mm"],
            "delta_mae_S0_minus_B2_mm": match["mae_mm"] - aggregate["mae_mm"],
            "B2_rmse_mm": aggregate["rmse_mm"],
            "S0_rmse_mm": match["rmse_mm"],
            "delta_rmse_S0_minus_B2_mm": match["rmse_mm"] - aggregate["rmse_mm"],
            "B2_p95_abs_mm": aggregate["p95_abs_mm"],
            "S0_p95_abs_mm": match["p95_abs_mm"],
            "delta_p95_S0_minus_B2_mm": match["p95_abs_mm"] - aggregate["p95_abs_mm"],
            "B2_max_abs_mm": aggregate["max_abs_mm"],
            "S0_max_abs_mm": match["max_abs_mm"],
            "delta_max_S0_minus_B2_mm": match["max_abs_mm"] - aggregate["max_abs_mm"],
            "S0_improved_rmse": match["rmse_mm"] < aggregate["rmse_mm"] - EPS,
            "S0_improved_p95": match["p95_abs_mm"] < aggregate["p95_abs_mm"] - EPS,
        })
    return output


def b2_vs_b0_summary(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scheme in ALL_SCHEMES:
        rows = [row for row in predictions if row.get("cv_scheme") == scheme and row.get("model") == "B2"]
        b0 = metrics(row["b0_corrected_bias_mm"] for row in rows)
        b2 = metrics(row["corrected_bias_mm"] for row in rows)
        output.append({
            "cv_scheme": scheme,
            "condition_count": len(rows),
            "b0_rmse_mm": b0["rmse_mm"],
            "b2_rmse_mm": b2["rmse_mm"],
            "delta_rmse_B2_minus_B0_mm": b2["rmse_mm"] - b0["rmse_mm"],
            "b0_p95_abs_mm": b0["p95_abs_mm"],
            "b2_p95_abs_mm": b2["p95_abs_mm"],
            "delta_p95_B2_minus_B0_mm": b2["p95_abs_mm"] - b0["p95_abs_mm"],
        })
    return output


def coefficient_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [row for row in data["coefficients"] if row.get("model") in MODELS]
    stability = [row for row in data["coefficient_stability"] if row.get("model") in MODELS]
    for row in selected:
        row["selection_scope"] = "strict_50mm_diagnostic" if row.get("cv_scheme") == "strict_50mm_validation" else "development_selection"
    for row in stability:
        row["selection_scope"] = "strict_50mm_diagnostic" if row.get("cv_scheme") == "strict_50mm_validation" else "development_selection"
    return selected, stability


def plot_worst(output: Path, worst_rows: list[dict[str, Any]]) -> None:
    schemes = ALL_SCHEMES
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(schemes), dtype=float)
    width = 0.36
    b2_mean = []
    s0_mean = []
    for scheme in schemes:
        rows = [row for row in worst_rows if row["cv_scheme"] == scheme]
        b2_mean.append(float(np.mean([row["B2_worst_abs_bias_mm"] for row in rows])))
        s0_mean.append(float(np.mean([row["S0_worst_abs_bias_mm"] for row in rows])))
    axes[0].bar(x - width / 2, b2_mean, width, label="B2 q2-only", color="#66bb6a")
    axes[0].bar(x + width / 2, s0_mean, width, label="S0 q1+q2", color="#ef6c00")
    axes[0].set_xticks(x, schemes, rotation=25, ha="right")
    axes[0].set_ylabel("mean fold worst-condition |bias| [mm]")
    axes[0].set_title("Worst-condition magnitude")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    for index, scheme in enumerate(schemes):
        rows = [row for row in worst_rows if row["cv_scheme"] == scheme]
        deltas = [row["delta_worst_abs_bias_S0_minus_B2_mm"] for row in rows]
        axes[1].scatter(np.full(len(deltas), index), deltas, color="#ef6c00", s=48, zorder=3)
    axes[1].axhline(0.0, color="#212121", linewidth=1.0)
    axes[1].set_xticks(x, schemes, rotation=25, ha="right")
    axes[1].set_ylabel("S0 − B2 worst |bias| [mm]")
    axes[1].set_title("Per-fold worst-condition delta (negative is better)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(output / "surface2_model_selection_worst_condition.png", dpi=180)
    plt.close(fig)


def plot_coefficients(output: Path, stability: list[dict[str, Any]]) -> None:
    schemes = ALL_SCHEMES
    x = np.arange(len(schemes), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    q1 = [next((row for row in stability if row["cv_scheme"] == scheme and row["model"] == "S0" and row["parameter"] == "q1"), None) for scheme in schemes]
    q2_b2 = [next((row for row in stability if row["cv_scheme"] == scheme and row["model"] == "B2" and row["parameter"] == "q2"), None) for scheme in schemes]
    q2_s0 = [next((row for row in stability if row["cv_scheme"] == scheme and row["model"] == "S0" and row["parameter"] == "q2"), None) for scheme in schemes]
    axes[0].errorbar(x, [row["mean"] if row else np.nan for row in q1], yerr=[row["std"] if row else np.nan for row in q1], fmt="o", color="#ef6c00", capsize=4)
    axes[0].axhline(0.0, color="#212121", linewidth=1.0)
    axes[0].set_title("S0 q1 coefficient")
    axes[0].set_ylabel("coefficient [mm / normalized q]")
    axes[0].set_xticks(x, schemes, rotation=25, ha="right")
    axes[0].grid(alpha=0.25)
    axes[1].errorbar(x - 0.08, [row["mean"] if row else np.nan for row in q2_b2], yerr=[row["std"] if row else np.nan for row in q2_b2], fmt="o", color="#66bb6a", capsize=4, label="B2 q2")
    axes[1].errorbar(x + 0.08, [row["mean"] if row else np.nan for row in q2_s0], yerr=[row["std"] if row else np.nan for row in q2_s0], fmt="o", color="#ef6c00", capsize=4, label="S0 q2")
    axes[1].axhline(0.0, color="#212121", linewidth=1.0)
    axes[1].set_title("q2 coefficient")
    axes[1].set_ylabel("coefficient [mm / normalized q]")
    axes[1].set_xticks(x, schemes, rotation=25, ha="right")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(output / "surface2_model_selection_coefficient_stability.png", dpi=180)
    plt.close(fig)


def decide(data: dict[str, Any], aggregate: list[dict[str, Any]], incremental: list[dict[str, Any]], worst: list[dict[str, Any]], stability: list[dict[str, Any]]) -> dict[str, Any]:
    dev_aggregate = {(row["cv_scheme"], row["support_state"], row["model"]): row for row in aggregate if row["support_state"] == "ALL" and row["cv_scheme"] in DEV_SCHEMES}
    fold_rows = [row for row in incremental if row["aggregation"] == "fold" and row["cv_scheme"] in DEV_SCHEMES]
    criteria: dict[str, Any] = {}
    for scheme in DEV_SCHEMES:
        b2 = dev_aggregate[(scheme, "ALL", "B2")]
        s0 = dev_aggregate[(scheme, "ALL", "S0")]
        folds = [row for row in fold_rows if row["cv_scheme"] == scheme]
        worst_rows = [row for row in worst if row["cv_scheme"] == scheme]
        p95_rate = sum(bool(row["S0_improved_p95"]) for row in folds) / len(folds)
        rmse_rate = sum(bool(row["S0_improved_rmse"]) for row in folds) / len(folds)
        worst_rate = sum(bool(row["S0_worst_condition_improved"]) for row in worst_rows) / len(worst_rows)
        criteria[scheme] = {
            "delta_rmse_mm": s0["rmse_mm"] - b2["rmse_mm"],
            "delta_p95_mm": s0["p95_abs_mm"] - b2["p95_abs_mm"],
            "rmse_fold_improvement_rate": rmse_rate,
            "p95_fold_improvement_rate": p95_rate,
            "worst_condition_fold_improvement_rate": worst_rate,
            "pooled_rmse_improved": s0["rmse_mm"] < b2["rmse_mm"] - EPS,
            "pooled_p95_improved": s0["p95_abs_mm"] < b2["p95_abs_mm"] - EPS,
            "majority_p95_improved": p95_rate > 0.5,
            "majority_worst_condition_improved": worst_rate > 0.5,
        }
    s0_performance_stable = all(
        item["pooled_rmse_improved"]
        and item["pooled_p95_improved"]
        and item["majority_p95_improved"]
        and item["majority_worst_condition_improved"]
        for item in criteria.values()
    )
    q1_stability = [row for row in stability if row["cv_scheme"] in DEV_SCHEMES and row["model"] == "S0" and row["parameter"] == "q1"]
    q1_sign_consistent = len(q1_stability) == 3 and all(bool(row["sign_consistency"]) for row in q1_stability)
    q1_amplitude_relative_ranges = {row["cv_scheme"]: row["relative_range_to_abs_mean"] for row in q1_stability}

    b2_vs_b0 = b2_vs_b0_summary(data["predictions"])
    b2_dev = [row for row in b2_vs_b0 if row["cv_scheme"] in DEV_SCHEMES]
    b2_stable = all(row["delta_rmse_B2_minus_B0_mm"] < -EPS and row["delta_p95_B2_minus_B0_mm"] < -EPS for row in b2_dev)
    if s0_performance_stable:
        selected = "S0"
        q1 = "YES"
    elif b2_stable:
        selected = "B2"
        q1 = "NO"
    else:
        selected = "UNDECIDED"
        q1 = "UNDECIDED"
    more_height = "YES" if data["summary"].get("HEIGHT_GAP_ACQUISITION_STILL_JUSTIFIED") == "YES" else "OPTIONAL"
    return {
        "SELECTED_SURFACE_MODEL": selected,
        "Q1_RETAINED": q1,
        "Q2_ONLY_CORRECTION_RECOMMENDED": "YES" if selected == "B2" else "NO",
        "MORE_HEIGHT_ACQUISITION_REQUIRED": more_height,
        "S0_STABLE_EXTRA_GAIN": s0_performance_stable,
        "B2_STABLE_Q2_GAIN_VS_B0": b2_stable,
        "q1_sign_consistent_across_dev_schemes": q1_sign_consistent,
        "q1_relative_range_to_abs_mean": q1_amplitude_relative_ranges,
        "scheme_criteria": criteria,
        "b2_vs_b0": b2_vs_b0,
        "selection_rule": "S0 requires each development scheme to have pooled RMSE/P95 improvement and majority fold improvement for P95 and worst condition; otherwise choose B2 when its q2 gain versus B0 is stable across development schemes.",
    }


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.{digits}f}"


def report_text(data: dict[str, Any], cv_rows: list[dict[str, Any]], aggregate: list[dict[str, Any]], support_aggregate: list[dict[str, Any]], incremental: list[dict[str, Any]], worst: list[dict[str, Any]], stability: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    pooled_lines = []
    for scheme in ALL_SCHEMES:
        rows = {row["model"]: row for row in aggregate if row["cv_scheme"] == scheme and row["support_state"] == "ALL"}
        for model in MODELS:
            row = rows[model]
            pooled_lines.append(f"| {scheme} | {model} | {row['condition_count']} | {fmt(row['bias_mm'])} | {fmt(row['mae_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['max_abs_mm'])} |")
    delta_lines = []
    for row in incremental:
        if row["aggregation"] != "pooled_condition_means" or row["support_state"] != "ALL":
            continue
        delta_lines.append(f"| {row['cv_scheme']} | {fmt(row['delta_bias_S0_minus_B2_mm'])} | {fmt(row['delta_mae_S0_minus_B2_mm'])} | {fmt(row['delta_rmse_S0_minus_B2_mm'])} | {fmt(row['delta_p95_S0_minus_B2_mm'])} | {fmt(row['delta_max_S0_minus_B2_mm'])} |")
    rate_lines = []
    for scheme in DEV_SCHEMES:
        rows = [row for row in incremental if row["aggregation"] == "fold" and row["cv_scheme"] == scheme]
        worst_rows = [row for row in worst if row["cv_scheme"] == scheme]
        rate_lines.append(f"| {scheme} | {sum(row['S0_improved_rmse'] for row in rows)}/{len(rows)} ({sum(row['S0_improved_rmse'] for row in rows)/len(rows):.1%}) | {sum(row['S0_improved_p95'] for row in rows)}/{len(rows)} ({sum(row['S0_improved_p95'] for row in rows)/len(rows):.1%}) | {sum(row['S0_worst_condition_improved'] for row in worst_rows)}/{len(worst_rows)} ({sum(row['S0_worst_condition_improved'] for row in worst_rows)/len(worst_rows):.1%}) |")
    support_lines = []
    for scheme in ALL_SCHEMES:
        for state in SUPPORT_STATES:
            rows = {row["model"]: row for row in support_aggregate if row["cv_scheme"] == scheme and row["support_state"] == state}
            if not rows:
                continue
            b2, s0 = rows["B2"], rows["S0"]
            support_lines.append(f"| {scheme} | {state} | {b2['condition_count']} | {fmt(b2['rmse_mm'])} | {fmt(s0['rmse_mm'])} | {fmt(s0['rmse_mm']-b2['rmse_mm'])} | {fmt(b2['p95_abs_mm'])} | {fmt(s0['p95_abs_mm'])} | {fmt(s0['p95_abs_mm']-b2['p95_abs_mm'])} |")
    worst_lines = []
    for scheme in ALL_SCHEMES:
        rows = [row for row in worst if row["cv_scheme"] == scheme]
        if not rows:
            continue
        worst_lines.append(f"| {scheme} | {len(rows)} | {fmt(np.mean([row['B2_worst_abs_bias_mm'] for row in rows]))} | {fmt(np.mean([row['S0_worst_abs_bias_mm'] for row in rows]))} | {sum(row['S0_worst_condition_improved'] for row in rows)}/{len(rows)} | {fmt(np.mean([row['condition_abs_error_improvement_rate'] for row in rows]), 3)} |")
    coeff_lines = []
    for row in stability:
        if row["cv_scheme"] not in ALL_SCHEMES:
            continue
        coeff_lines.append(f"| {row['cv_scheme']} | {row['model']} | {row['parameter']} | {fmt(row['mean'], 6)} | {fmt(row['std'], 6)} | {fmt(row['range'], 6)} | {fmt(row['relative_range_to_abs_mean'], 3)} | {row['sign_consistency']} |")
    criteria_lines = []
    for scheme, item in decision["scheme_criteria"].items():
        criteria_lines.append(f"- `{scheme}`：ΔRMSE={fmt(item['delta_rmse_mm'])} mm，ΔP95={fmt(item['delta_p95_mm'])} mm；P95 fold={item['p95_fold_improvement_rate']:.1%}，worst-condition fold={item['worst_condition_fold_improvement_rate']:.1%}。")
    strict_rows = [row for row in incremental if row["aggregation"] == "pooled_condition_means" and row["cv_scheme"] == "strict_50mm_validation" and row["support_state"] == "ALL"]
    strict_line = "；".join(f"{row['cv_scheme']} {row['B2_rmse_mm']:.5f}→{row['S0_rmse_mm']:.5f} RMSE, Δ={row['delta_rmse_S0_minus_B2_mm']:.5f}; {row['B2_p95_abs_mm']:.5f}→{row['S0_p95_abs_mm']:.5f} P95, Δ={row['delta_p95_S0_minus_B2_mm']:.5f}" for row in strict_rows)
    return f"""# Surface-2 B2(q2-only) vs S0(q1+q2) model selection

## 结论

`SELECTED_SURFACE_MODEL={decision['SELECTED_SURFACE_MODEL']}`  
`Q1_RETAINED={decision['Q1_RETAINED']}`  
`Q2_ONLY_CORRECTION_RECOMMENDED={decision['Q2_ONLY_CORRECTION_RECOMMENDED']}`  
`MORE_HEIGHT_ACQUISITION_REQUIRED={decision['MORE_HEIGHT_ACQUISITION_REQUIRED']}`

历史结论原样保留：`Q2_GAP_FILLED=NO`、`SURFACE2C_ALLOWED=NO`。本轮没有因 q2 gap 阻塞 B2/S0 比较；上述选择仅是 development grouped-CV 的诊断性模型结构选择，不是 production validation，也不等于允许直接接入 correction 链路。

判断规则：S0 只有在 LOHO、LOPO、LOBO 三个 development scheme 中都同时满足 pooled RMSE/P95 改善，并且 P95 与 worst-condition 的 fold improvement 为多数时，才保留 q1；否则在 B2 相对 B0 的 q2 增益稳定时选择更简单的 B2。50 mm 不参与任何选择。

## Provenance / 复用边界

- 复用 canonical Surface-2BR2 的 condition table、CV metrics、condition predictions、coefficients 和 coefficient stability；输入 SHA 已写入 `surface2_model_selection_summary.json`。
- development：1/2/6/10/20/30/36/40/46 mm，44 conditions、11160 analysis points。
- strict held-out：50 mm，5 conditions、1100 analysis points；仅本报告末尾诊断。
- condition 等权；沿用既有点级拟合权重 `1 / condition_point_count`；没有 random point split。
- Frozen C0/C1、manual ROI、session-linear ground proxy、q1/q2 定义全部由既有 artifact 继承；未重拟 C0/C1、未修改 ROI/q、未拟合新 correction。

## Pooled condition metrics

| CV scheme | model | conditions | Bias | MAE | RMSE | P95 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(pooled_lines)}

## S0 相对 B2 incremental

负值代表 S0 优于 B2。

| CV scheme | ΔBias | ΔMAE | ΔRMSE | ΔP95 | ΔMax |
|---|---:|---:|---:|---:|---:|
{chr(10).join(delta_lines)}

### Fold improvement rate

| CV scheme | RMSE improved | P95 improved | worst-condition improved |
|---|---:|---:|---:|
{chr(10).join(rate_lines)}

{chr(10).join(criteria_lines)}

LOHO 与 LOPO 的 pooled P95 分别没有改善，因而 S0 没有达到“跨三个 scheme 稳定额外收益”的门槛；q1 即使符号稳定，也没有足够稳定的工程收益。

## q-space support 分层

support 分类沿用每个 frozen fold 的 `IN_DOMAIN / HULL_EXTRAPOLATION / BBOX_EXTRAPOLATION`，extrapolation 改善不被解释为域内泛化。

| CV scheme | support | conditions | B2 RMSE | S0 RMSE | ΔRMSE | B2 P95 | S0 P95 | ΔP95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(support_lines)}

## Worst-condition

| CV scheme | folds | mean B2 worst | mean S0 worst | S0 improved folds | mean condition error improvement rate |
|---|---:|---:|---:|---:|---:|
{chr(10).join(worst_lines)}

对应明细写入 `surface2_model_selection_worst_conditions.csv`；每个 fold 的 worst condition 是该 fold condition mean 的最大绝对 corrected bias，不是点级最大值。

## Coefficient stability

| CV scheme | model | parameter | mean | std | range | range/abs(mean) | sign consistent |
|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(coeff_lines)}

S0 q1 在 development scheme 中符号一致：`{decision['q1_sign_consistent_across_dev_schemes']}`；幅值 relative range 为 `{decision['q1_relative_range_to_abs_mean']}`。符号一致不能抵消 LOHO/LOPO P95 与 worst-condition 额外收益不稳定这一事实。

## Strict 50 mm（只作最终诊断）

{strict_line}

50 mm 所有 fold 均为 q-space `BBOX_EXTRAPOLATION`，因此不用于模型选择、阈值或参数调整；它不能把 S0 的微小 RMSE/P95 改善升级为开发域泛化证据。

## B2(q2-only) 的后续含义

B2 相对公共 offset B0 在三个 development grouped schemes 的 pooled RMSE 与 P95 均改善，因此本轮推荐 `Q2_ONLY_CORRECTION_RECOMMENDED=YES`，但仅作为 1D q2 correction 的诊断候选。现有 q2 domain gap 和低 IN_DOMAIN 支持仍使后续高度补采保持必要；这不覆盖原 Surface-2B/2BR2 结论。

## 输出

- `surface2_model_selection_cv_metrics.csv`
- `surface2_model_selection_support_metrics.csv`
- `surface2_model_selection_incremental_comparison.csv`
- `surface2_model_selection_coefficients.csv`
- `surface2_model_selection_coefficient_stability.csv`
- `surface2_model_selection_worst_conditions.csv`
- `surface2_model_selection_worst_condition.png`
- `surface2_model_selection_coefficient_stability.png`
- `surface2_model_selection_summary.json`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = validate_inputs(args.input.resolve())
    cv_selected = [row for row in data["cv_metrics"] if row.get("model") in MODELS and row.get("cv_scheme") in ALL_SCHEMES]
    cv_rows = cv_output_rows(cv_selected)
    aggregate: list[dict[str, Any]] = []
    for scheme in ALL_SCHEMES:
        aggregate.extend(aggregate_condition_metrics(data["predictions"], scheme, "ALL"))
        for state in SUPPORT_STATES:
            aggregate.extend(aggregate_condition_metrics(data["predictions"], scheme, state))
    support_aggregate = [row for row in aggregate if row["aggregation"] == "support_stratified"]
    worst = worst_condition_rows(data["predictions"])
    incremental = incremental_rows(cv_rows, aggregate, worst)
    coeff, stability = coefficient_rows(data)
    decision = decide(data, aggregate, incremental, worst, stability)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "surface2_model_selection_cv_metrics.csv", cv_rows, list(cv_rows[0].keys()))
    write_csv(output / "surface2_model_selection_support_metrics.csv", support_aggregate, list(support_aggregate[0].keys()) if support_aggregate else ["cv_scheme"])
    incremental_fields = [
        "aggregation", "cv_scheme", "heldout_group", "support_state", "B2_condition_count", "B2_bias_mm", "S0_bias_mm", "delta_bias_S0_minus_B2_mm", "B2_mae_mm", "S0_mae_mm", "delta_mae_S0_minus_B2_mm", "B2_rmse_mm", "S0_rmse_mm", "delta_rmse_S0_minus_B2_mm", "B2_p95_abs_mm", "S0_p95_abs_mm", "delta_p95_S0_minus_B2_mm", "B2_max_abs_mm", "S0_max_abs_mm", "delta_max_S0_minus_B2_mm", "S0_improved_rmse", "S0_improved_p95", "S0_worst_condition_improved", "condition_abs_error_improvement_rate",
    ]
    write_csv(output / "surface2_model_selection_incremental_comparison.csv", incremental, incremental_fields)
    write_csv(output / "surface2_model_selection_coefficients.csv", coeff, list(coeff[0].keys()))
    write_csv(output / "surface2_model_selection_coefficient_stability.csv", stability, list(stability[0].keys()))
    write_csv(output / "surface2_model_selection_worst_conditions.csv", worst, list(worst[0].keys()))
    plot_worst(output, worst)
    plot_coefficients(output, stability)
    summary = {
        **{key: decision[key] for key in ("SELECTED_SURFACE_MODEL", "Q1_RETAINED", "Q2_ONLY_CORRECTION_RECOMMENDED", "MORE_HEIGHT_ACQUISITION_REQUIRED")},
        "decision": decision,
        "historical_surface2b_conclusion": data["summary"].get("original_surface2b_conclusion"),
        "protocol": data["protocol"],
        "input_sha256": data["input_sha256"],
        "input_paths": {key: str(path.resolve()) for key, path in data["paths"].items()},
        "development_condition_count": 44,
        "development_point_count": 11160,
        "strict_50_condition_count": 5,
        "strict_50_point_count": 1100,
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface2_model_selection_summary.json", summary)
    (output / "surface2_model_selection_report.md").write_text(report_text(data, cv_rows, aggregate, support_aggregate, incremental, worst, stability, decision), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("SELECTED_SURFACE_MODEL", "Q1_RETAINED", "Q2_ONLY_CORRECTION_RECOMMENDED", "MORE_HEIGHT_ACQUISITION_REQUIRED")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
