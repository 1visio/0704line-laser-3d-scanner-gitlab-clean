"""Surface-2B low-degree q correction feasibility audit.

This is a diagnostic-only follow-up to Surface-2B.  It never edits the
Surface-2B artifacts and never changes C0/C1, ROI, q coordinates, or a
production correction chain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, QhullError


ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
SURFACE2B = BASE_OUTPUT / "surface2" / "surface2b"
DEFAULT_OUTPUT = BASE_OUTPUT / "surface2" / "surface2br"

DEV_HEIGHTS = (30.0, 36.0, 40.0, 46.0)
HELDOUT_HEIGHT = 50.0
ALL_HEIGHTS = (*DEV_HEIGHTS, HELDOUT_HEIGHT)
MODELS = ("S0", "S1")
PARAMETERS = {
    "S0": ("intercept", "q1", "q2"),
    "S1": ("intercept", "q1", "q2", "q1_sq", "q1_q2", "q2_sq"),
}
METRICS = ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm")
Q_TOLERANCE = 0.05
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def metric_values(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {name: float("nan") for name in METRICS}
    absolute = np.abs(array)
    return {
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array * array))),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "max_abs_mm": float(np.max(absolute)),
    }


def condition_id(row: dict[str, Any]) -> str:
    return f"{row['dataset']}/rank{int(row['position_rank'])}"


def load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    samples_path = SURFACE2B / "surface2b_samples.csv"
    condition_path = SURFACE2B / "surface2b_condition_statistics.csv"
    domain_path = SURFACE2B / "surface2b_domain_statistics.csv"
    summary_path = SURFACE2B / "surface2b_summary.json"
    required = (samples_path, condition_path, domain_path, summary_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Surface-2B artifacts: " + ", ".join(missing))

    summary = read_json(summary_path)
    if summary.get("Q2_GAP_FILLED") != "NO" or summary.get("SURFACE2C_ALLOWED") != "NO":
        raise RuntimeError(
            "Surface-2BR expects the frozen original Surface-2B conclusion "
            "Q2_GAP_FILLED=NO and SURFACE2C_ALLOWED=NO."
        )
    q_definition = summary.get("q_definition", {})
    if q_definition.get("coordinate_name") != "q1_q2":
        raise RuntimeError("Unexpected Surface-2B q coordinate definition")
    if abs(float(q_definition.get("q_pair_tolerance_normalized", Q_TOLERANCE)) - Q_TOLERANCE) > EPS:
        raise RuntimeError("Surface-2B q tolerance is not the frozen 0.05")

    raw_samples = read_csv(samples_path)
    condition_stats = read_csv(condition_path)
    domain_stats = read_csv(domain_path)
    analysis_rows: list[dict[str, Any]] = []
    for raw in raw_samples:
        height = float(raw["true_height_mm"])
        if height not in ALL_HEIGHTS or not as_bool(raw.get("analysis_included")):
            continue
        required_fields = ("q1", "q2", "height_residual_mm", "position_rank")
        if not all(finite(raw.get(field)) for field in required_fields):
            continue
        row = dict(raw)
        row["true_height_mm"] = height
        row["q1"] = float(raw["q1"])
        row["q2"] = float(raw["q2"])
        row["height_residual_mm"] = float(raw["height_residual_mm"])
        row["position_rank"] = int(raw["position_rank"])
        row["condition_id"] = condition_id(row)
        row["point_index"] = int(raw.get("point_index", 0))
        analysis_rows.append(row)

    if not analysis_rows:
        raise RuntimeError("No analysis_included samples found")
    condition_counts = Counter(row["condition_id"] for row in analysis_rows)
    expected_conditions = {f"obs_{int(height)}mm/rank{rank}" for height in ALL_HEIGHTS for rank in range(1, 6)}
    if set(condition_counts) != expected_conditions:
        missing_conditions = sorted(expected_conditions - set(condition_counts))
        extra_conditions = sorted(set(condition_counts) - expected_conditions)
        raise RuntimeError(
            f"Condition set mismatch; missing={missing_conditions}, extra={extra_conditions}"
        )

    # Cross-check that the condition-level artifact covers the same 25 groups.
    condition_stat_keys = {
        f"{row.get('dataset')}/rank{int(row['position_rank'])}"
        for row in condition_stats
        if finite(row.get("true_height_mm")) and float(row["true_height_mm"]) in ALL_HEIGHTS
    }
    if condition_stat_keys != expected_conditions:
        raise RuntimeError("Surface-2B condition_statistics does not cover all 25 conditions")
    domain_heights = {float(row["true_height_mm"]) for row in domain_stats}
    if domain_heights != set(ALL_HEIGHTS):
        raise RuntimeError("Surface-2B domain_statistics does not cover 30/36/40/46/50 mm")
    return analysis_rows, condition_stats, domain_stats, summary, {
        "samples_path": samples_path,
        "condition_statistics_path": condition_path,
        "domain_statistics_path": domain_path,
        "summary_path": summary_path,
    }


def design_matrix(rows: list[dict[str, Any]], model: str) -> np.ndarray:
    q1 = np.asarray([row["q1"] for row in rows], dtype=np.float64)
    q2 = np.asarray([row["q2"] for row in rows], dtype=np.float64)
    if model == "S0":
        return np.column_stack((np.ones_like(q1), q1, q2))
    if model == "S1":
        return np.column_stack((np.ones_like(q1), q1, q2, q1 * q1, q1 * q2, q2 * q2))
    raise ValueError(f"Unknown model: {model}")


def fit_model(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"Cannot fit {model} with no rows")
    groups = Counter(row["condition_id"] for row in rows)
    weights = np.asarray([1.0 / groups[row["condition_id"]] for row in rows], dtype=np.float64)
    matrix = design_matrix(rows, model)
    target = np.asarray([row["height_residual_mm"] for row in rows], dtype=np.float64)
    sqrt_weight = np.sqrt(weights)
    weighted_matrix = matrix * sqrt_weight[:, None]
    weighted_target = target * sqrt_weight
    beta, _, rank, singular_values = np.linalg.lstsq(
        weighted_matrix, weighted_target, rcond=None
    )
    return {
        "model": model,
        "beta": np.asarray(beta, dtype=np.float64),
        "train_point_count": len(rows),
        "train_condition_count": len(groups),
        "design_rank": int(rank),
        "condition_number": float(np.linalg.cond(weighted_matrix)),
        "singular_values": np.asarray(singular_values, dtype=np.float64),
    }


def predict(rows: list[dict[str, Any]], fit: dict[str, Any]) -> np.ndarray:
    return design_matrix(rows, fit["model"]) @ fit["beta"]


def q_support(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_q = np.asarray([[row["q1"], row["q2"]] for row in train_rows], dtype=np.float64)
    test_q = np.asarray([[row["q1"], row["q2"]] for row in test_rows], dtype=np.float64)
    lower = np.min(train_q, axis=0)
    upper = np.max(train_q, axis=0)
    bbox_inside = np.all((test_q >= lower - EPS) & (test_q <= upper + EPS), axis=1)
    unique_train_q = np.unique(train_q, axis=0)
    hull_inside = np.zeros(len(test_q), dtype=bool)
    if len(unique_train_q) >= 3:
        try:
            hull_inside = Delaunay(unique_train_q).find_simplex(test_q) >= 0
        except (QhullError, ValueError, np.linalg.LinAlgError):
            hull_inside = bbox_inside.copy()
    bbox_oob = ~bbox_inside
    hull_oob = ~hull_inside
    return {
        "train_q1_min": float(lower[0]),
        "train_q1_max": float(upper[0]),
        "train_q2_min": float(lower[1]),
        "train_q2_max": float(upper[1]),
        "test_q1_min": float(np.min(test_q[:, 0])),
        "test_q1_max": float(np.max(test_q[:, 0])),
        "test_q2_min": float(np.min(test_q[:, 1])),
        "test_q2_max": float(np.max(test_q[:, 1])),
        "test_point_count": len(test_rows),
        "bbox_oob_point_count": int(np.count_nonzero(bbox_oob)),
        "bbox_oob_point_rate": float(np.mean(bbox_oob)),
        "hull_oob_point_count": int(np.count_nonzero(hull_oob)),
        "hull_oob_point_rate": float(np.mean(hull_oob)),
        "bbox_extrapolation": bool(np.any(bbox_oob)),
        "hull_extrapolation": bool(np.any(hull_oob)),
        "support_state": (
            "IN_DOMAIN"
            if not np.any(bbox_oob) and not np.any(hull_oob)
            else "BBOX_EXTRAPOLATION"
            if np.any(bbox_oob)
            else "HULL_EXTRAPOLATION"
        ),
    }


def grouped_condition_values(
    rows: list[dict[str, Any]],
    corrected: np.ndarray,
    prediction: np.ndarray,
    support_bbox: np.ndarray,
    support_hull: np.ndarray,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["condition_id"]].append(index)
    result = []
    for key in sorted(groups):
        indices = groups[key]
        first = rows[indices[0]]
        raw_values = np.asarray(
            [rows[index]["height_residual_mm"] for index in indices], dtype=np.float64
        )
        corrected_values = corrected[indices]
        prediction_values = prediction[indices]
        result.append(
            {
                "condition_id": key,
                "dataset": first["dataset"],
                "true_height_mm": first["true_height_mm"],
                "position_rank": first["position_rank"],
                "q1_median": float(np.median([rows[index]["q1"] for index in indices])),
                "q2_median": float(np.median([rows[index]["q2"] for index in indices])),
                "point_count": len(indices),
                "raw_bias_mm": float(np.mean(raw_values)),
                "predicted_bias_mm": float(np.mean(prediction_values)),
                "corrected_bias_mm": float(np.mean(corrected_values)),
                "bbox_oob_point_count": int(np.count_nonzero(~support_bbox[indices])),
                "hull_oob_point_count": int(np.count_nonzero(~support_hull[indices])),
                "support_state": (
                    "BBOX_EXTRAPOLATION"
                    if np.any(~support_bbox[indices])
                    else "HULL_EXTRAPOLATION"
                    if np.any(~support_hull[indices])
                    else "IN_DOMAIN"
                ),
            }
        )
    return result


def condition_metrics(values: list[dict[str, Any]], field: str) -> dict[str, float]:
    return metric_values([float(row[field]) for row in values])


def evaluate_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    model: str,
    scheme: str,
    heldout_group: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    fit = fit_model(train_rows, model)
    prediction = predict(test_rows, fit)
    raw = np.asarray([row["height_residual_mm"] for row in test_rows], dtype=np.float64)
    corrected = raw - prediction
    support = q_support(train_rows, test_rows)
    train_q = np.asarray([[row["q1"], row["q2"]] for row in train_rows], dtype=np.float64)
    test_q = np.asarray([[row["q1"], row["q2"]] for row in test_rows], dtype=np.float64)
    lower = np.min(train_q, axis=0)
    upper = np.max(train_q, axis=0)
    bbox_inside = np.all((test_q >= lower - EPS) & (test_q <= upper + EPS), axis=1)
    unique_train_q = np.unique(train_q, axis=0)
    if len(unique_train_q) >= 3:
        try:
            hull_inside = Delaunay(unique_train_q).find_simplex(test_q) >= 0
        except (QhullError, ValueError, np.linalg.LinAlgError):
            hull_inside = bbox_inside.copy()
    else:
        hull_inside = bbox_inside.copy()
    grouped = grouped_condition_values(test_rows, corrected, prediction, bbox_inside, hull_inside)
    raw_metrics = condition_metrics(grouped, "raw_bias_mm")
    corrected_metrics = condition_metrics(grouped, "corrected_bias_mm")
    row: dict[str, Any] = {
        "cv_scheme": scheme,
        "model": model,
        "heldout_group": heldout_group,
        "heldout_height_mm": (
            float(test_rows[0]["true_height_mm"])
            if len({row["true_height_mm"] for row in test_rows}) == 1
            else ""
        ),
        "heldout_position_rank": (
            int(test_rows[0]["position_rank"])
            if len({row["position_rank"] for row in test_rows}) == 1
            else ""
        ),
        "train_condition_count": fit["train_condition_count"],
        "train_point_count": fit["train_point_count"],
        "test_condition_count": len(grouped),
        "test_point_count": len(test_rows),
        **support,
    }
    for name in METRICS:
        raw_name = f"raw_{name}"
        corrected_name = f"corrected_{name}"
        row[raw_name] = raw_metrics[name]
        row[corrected_name] = corrected_metrics[name]
        row[f"delta_{name}"] = corrected_metrics[name] - raw_metrics[name]
        row[f"improved_{name}"] = corrected_metrics[name] < raw_metrics[name] - EPS
    row["raw_abs_bias_mm"] = abs(row["raw_bias_mm"])
    row["corrected_abs_bias_mm"] = abs(row["corrected_bias_mm"])
    row["delta_abs_bias_mm"] = row["corrected_abs_bias_mm"] - row["raw_abs_bias_mm"]
    row["improved_abs_bias"] = row["corrected_abs_bias_mm"] < row["raw_abs_bias_mm"] - EPS
    row["all_core_metrics_improved"] = bool(
        row["improved_mae_mm"] and row["improved_rmse_mm"]
    )
    row["no_metric_worsening"] = bool(
        row["delta_abs_bias_mm"] <= EPS
        and all(row[f"delta_{name}"] <= EPS for name in METRICS if name != "bias_mm")
    )
    coefficient_rows = []
    for parameter, coefficient in zip(PARAMETERS[model], fit["beta"]):
        coefficient_rows.append(
            {
                "cv_scheme": scheme,
                "model": model,
                "heldout_group": heldout_group,
                "fit_scope": "development_only" if scheme != "strict_50mm_validation" else "development_all_for_50mm",
                "parameter": parameter,
                "coefficient": float(coefficient),
                "train_condition_count": fit["train_condition_count"],
                "train_point_count": fit["train_point_count"],
                "design_rank": fit["design_rank"],
                "design_condition_number": fit["condition_number"],
            }
        )
    return row, grouped, {"fit": fit, "coefficient_rows": coefficient_rows}


def model_comparison(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    schemes = sorted({row["cv_scheme"] for row in predictions})
    for scheme in schemes:
        for model in MODELS:
            selected = [
                row for row in predictions
                if row["cv_scheme"] == scheme and row["model"] == model
            ]
            if not selected:
                continue
            raw = metric_values([row["raw_bias_mm"] for row in selected])
            corrected = metric_values([row["corrected_bias_mm"] for row in selected])
            result_row = {
                "cv_scheme": scheme,
                "model": model,
                "condition_count": len(selected),
                "raw_bias_mm": raw["bias_mm"],
                "raw_mae_mm": raw["mae_mm"],
                "raw_rmse_mm": raw["rmse_mm"],
                "raw_p95_abs_mm": raw["p95_abs_mm"],
                "raw_max_abs_mm": raw["max_abs_mm"],
                "corrected_bias_mm": corrected["bias_mm"],
                "corrected_mae_mm": corrected["mae_mm"],
                "corrected_rmse_mm": corrected["rmse_mm"],
                "corrected_p95_abs_mm": corrected["p95_abs_mm"],
                "corrected_max_abs_mm": corrected["max_abs_mm"],
                "delta_rmse_mm": corrected["rmse_mm"] - raw["rmse_mm"],
                "delta_mae_mm": corrected["mae_mm"] - raw["mae_mm"],
            }
            result.append(result_row)
    return result


def coefficient_stability(coefficient_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for scheme in sorted({row["cv_scheme"] for row in coefficient_rows}):
        for model in MODELS:
            for parameter in PARAMETERS[model]:
                selected = [
                    row["coefficient"] for row in coefficient_rows
                    if row["cv_scheme"] == scheme
                    and row["model"] == model
                    and row["parameter"] == parameter
                ]
                if not selected:
                    continue
                values = np.asarray(selected, dtype=np.float64)
                mean = float(np.mean(values))
                median = float(np.median(values))
                value_range = float(np.max(values) - np.min(values))
                result.append(
                    {
                        "cv_scheme": scheme,
                        "model": model,
                        "parameter": parameter,
                        "fold_count": len(values),
                        "mean": mean,
                        "median": median,
                        "std": float(np.std(values, ddof=0)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "range": value_range,
                        "relative_range_to_abs_mean": value_range / max(abs(mean), EPS),
                        "sign_consistency": bool(np.all(values >= 0) or np.all(values <= 0)),
                    }
                )
    return result


def make_plot(output: Path, metrics: list[dict[str, Any]]) -> None:
    schemes = ["LOHO_development", "LOPO_position_rank", "strict_50mm_validation"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for axis, metric_name, title in zip(
        axes,
        ("rmse_mm", "p95_abs_mm"),
        ("Condition-balanced RMSE", "Condition-balanced P95 absolute error"),
    ):
        x_labels = []
        raw_values = []
        s0_values = []
        s1_values = []
        for scheme in schemes:
            groups = sorted({row["heldout_group"] for row in metrics if row["cv_scheme"] == scheme})
            for group in groups:
                raw_row = next(
                    row for row in metrics
                    if row["cv_scheme"] == scheme and row["heldout_group"] == group and row["model"] == "S0"
                )
                s1_row = next(
                    row for row in metrics
                    if row["cv_scheme"] == scheme and row["heldout_group"] == group and row["model"] == "S1"
                )
                x_labels.append(f"{scheme.replace('_', ' ')}\n{group}")
                raw_values.append(raw_row[f"raw_{metric_name}"])
                s0_values.append(raw_row[f"corrected_{metric_name}"])
                s1_values.append(s1_row[f"corrected_{metric_name}"])
        positions = np.arange(len(x_labels), dtype=np.float64)
        width = 0.25
        axis.bar(positions - width, raw_values, width, label="raw", color="#9e9e9e")
        axis.bar(positions, s0_values, width, label="S0 corrected", color="#1976d2")
        axis.bar(positions + width, s1_values, width, label="S1 corrected", color="#ef6c00")
        axis.set_ylabel("mm")
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels(x_labels, rotation=70, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=3)
    fig.suptitle("Surface-2BR raw vs low-degree q correction (condition-balanced)")
    fig.savefig(output / "surface2br_raw_vs_corrected.png", dpi=180)
    plt.close(fig)


def report_text(
    input_paths: dict[str, Path],
    input_hashes: dict[str, str],
    summary: dict[str, Any],
    comparison: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    comparison_lines = []
    for row in comparison:
        comparison_lines.append(
            "| {scheme} | {model} | {n} | {raw:.5f} | {corr:.5f} | {delta:.5f} | {p95:.5f} |".format(
                scheme=row["cv_scheme"], model=row["model"], n=row["condition_count"],
                raw=row["raw_rmse_mm"], corr=row["corrected_rmse_mm"],
                delta=row["delta_rmse_mm"], p95=row["corrected_p95_abs_mm"],
            )
        )
    fold_lines = []
    for row in metrics:
        fold_lines.append(
            "| {scheme} | {model} | {group} | {raw:.5f} | {corr:.5f} | {delta:.5f} | {state} | {oob:.1%} |".format(
                scheme=row["cv_scheme"], model=row["model"], group=row["heldout_group"],
                raw=row["raw_rmse_mm"], corr=row["corrected_rmse_mm"],
                delta=row["delta_rmse_mm"], state=row["support_state"],
                oob=row["bbox_oob_point_rate"],
            )
        )
    s0_fold_lines = []
    for row in metrics:
        if row["model"] != "S0":
            continue
        s0_fold_lines.append(
            "| {scheme} | {group} | {rb:.5f} | {cb:.5f} | {ram:.5f} | {cam:.5f} | {rr:.5f} | {cr:.5f} | {rp:.5f} | {cp:.5f} | {rx:.5f} | {cx:.5f} |".format(
                scheme=row["cv_scheme"], group=row["heldout_group"],
                rb=row["raw_bias_mm"], cb=row["corrected_bias_mm"],
                ram=row["raw_abs_bias_mm"], cam=row["corrected_abs_bias_mm"],
                rr=row["raw_rmse_mm"], cr=row["corrected_rmse_mm"],
                rp=row["raw_p95_abs_mm"], cp=row["corrected_p95_abs_mm"],
                rx=row["raw_max_abs_mm"], cx=row["corrected_max_abs_mm"],
            )
        )
    stability_lines = []
    for row in stability:
        if row["cv_scheme"] == "strict_50mm_validation":
            continue
        stability_lines.append(
            "| {scheme} | {model} | {parameter} | {mean:.6g} | {std:.6g} | {range:.6g} | {relative:.3g} | {sign} |".format(
                scheme=row["cv_scheme"], model=row["model"], parameter=row["parameter"],
                mean=row["mean"], std=row["std"], range=row["range"],
                relative=row["relative_range_to_abs_mean"], sign=row["sign_consistency"],
            )
        )
    original = {
        "Q2_GAP_FILLED": summary.get("Q2_GAP_FILLED"),
        "SURFACE2C_ALLOWED": summary.get("SURFACE2C_ALLOWED"),
        "Q1Q2_STATE_CONSISTENCY": summary.get("Q1Q2_STATE_CONSISTENCY"),
    }
    return f"""# Surface-2BR 低自由度 q correction feasibility audit

## 独立结论

`SURFACE_CORRECTION_FEASIBILITY={decision['SURFACE_CORRECTION_FEASIBILITY']}`  
`MORE_HEIGHT_ACQUISITION_REQUIRED={decision['MORE_HEIGHT_ACQUISITION_REQUIRED']}`

本报告是 Surface-2B 的 feasibility audit，不是 production validation，也不覆盖原报告。原结论保持：`Q2_GAP_FILLED={original['Q2_GAP_FILLED']}`、`Q1Q2_STATE_CONSISTENCY={original['Q1Q2_STATE_CONSISTENCY']}`、`SURFACE2C_ALLOWED={original['SURFACE2C_ALLOWED']}`。

判定只在 30/36/40/46 mm development 上进行模型比较；50 mm 从未进入拟合、模型选择或阈值调整，仅用 development-all fit 做一次 strict held-out 描述。每个 height×position condition 总权重相同，拟合采用每个点的 `1 / condition_point_count` 权重；没有 random point split。

## 模型

- S0：`F(q1,q2)=a0+a1*q1+a2*q2`
- S1：`F(q1,q2)=a0+a1*q1+a2*q2+a3*q1²+a4*q1*q2+a5*q2²`
- corrected residual：`r_corrected = r - F(q1,q2)`。
- q1/q2 沿用 Frozen C0 的 Surface-2B 定义，未重新中心化、缩放或重定义。

## Provenance

| artifact | SHA256 |
|---|---|
| samples | `{input_hashes['samples']}` |
| condition statistics | `{input_hashes['condition_statistics']}` |
| domain statistics | `{input_hashes['domain_statistics']}` |
| Surface-2B summary | `{input_hashes['summary']}` |

## Pooled condition-balanced comparison

| CV scheme | model | conditions | raw RMSE | corrected RMSE | ΔRMSE | corrected P95 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

## Fold metrics and q-space support

`bbox_oob_rate` 是该 fold 测试点落在训练 fold q1/q2 轴对齐 bbox 外的比例；这不是自动删点，也不代表允许 extrapolation。所有预测即使越界也只作为诊断数值输出，并明确标记 unsupported。

| scheme | model | held-out group | raw RMSE | corrected RMSE | ΔRMSE | support | bbox OOB |
|---|---:|---|---:|---:|---:|---|---:|
{chr(10).join(fold_lines)}

### S0 各 held-out fold 完整指标

| scheme | held-out group | raw Bias | corrected Bias | raw abs(Bias) | corrected abs(Bias) | raw RMSE | corrected RMSE | raw P95 | corrected P95 | raw Max | corrected Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(s0_fold_lines)}

## Coefficient stability

| scheme | model | parameter | mean | std | range | range/abs(mean) | sign consistent |
|---|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(stability_lines)}

## 判定依据

- S0 development LOHO 所有折的无 q2 overlap 外推折数：`{decision['s0_loho_bbox_extrapolation_folds']}` / 4。
- S0 development LOPO 的 rank5 fold RMSE Δ：`{decision['s0_rank5_rmse_delta_mm']:.6f} mm`；rank5 bbox OOB rate：`{decision['s0_rank5_bbox_oob_rate']:.2%}`。
- S0 development folds 中 MAE 与 RMSE 同时改善的折数：`{decision['s0_core_improved_folds']}` / `{decision['s0_development_fold_count']}`。
- S0 development folds 中所有 abs(Bias)/MAE/RMSE/P95/Max 均未恶化的折数：`{decision['s0_no_worsening_folds']}` / `{decision['s0_development_fold_count']}`。
- S0 development pooled RMSE Δ：`{decision['s0_pooled_rmse_delta_mm']:.6f} mm`。
- S0 development 系数出现 sign flip 的项：`{', '.join(decision['s0_unstable_coefficient_terms']) or '无'}`；这些系数不应被视为已冻结参数。
- S1 仅作同协议二次项对照，不因为 50 mm 结果选择或调参。

当前 q2 band 仍没有提供跨高度共同 support；即使 S0 在部分外推折上改善，也不能把 q-space extrapolation 当成 Surface-2C 的泛化证据。因此本轮建议 `{decision['MORE_HEIGHT_ACQUISITION_REQUIRED']}`：若要继续，应优先补采 33/38/43/48 mm，并重新进行独立 grouped feasibility audit。

## 约束确认

- 未修改 C0/C1、ROI、Frozen q1/q2 定义。
- 未使用 50 mm 拟合、模型选择或参数调整。
- 未使用 spline、RF、MLP 或其他高自由度模型。
- 未把本轮结果包装为 production validation。

## 输出

- `surface2br_cv_metrics.csv`：逐 fold LOHO/LOPO/50 strict 指标。
- `surface2br_loho_metrics.csv`、`surface2br_lopo_metrics.csv`：独立 LOHO/LOPO 指标。
- `surface2br_model_comparison.csv`：按 CV scheme 的 condition-balanced 汇总。
- `surface2br_condition_predictions.csv`：逐 condition raw/predicted/corrected 结果及 support 标记。
- `surface2br_coefficients.csv`、`surface2br_coefficient_stability.csv`：逐 fold 系数和稳定性。
- `surface2br_raw_vs_corrected.png`：raw/S0/S1 对比图。
- `surface2br_summary.json`：机器可读结论、provenance 与判定字段。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis_rows, condition_stats, domain_stats, surface2b_summary, input_paths = load_inputs()
    input_hashes = {
        name: sha256(path)
        for name, path in (
            ("samples", input_paths["samples_path"]),
            ("condition_statistics", input_paths["condition_statistics_path"]),
            ("domain_statistics", input_paths["domain_statistics_path"]),
            ("summary", input_paths["summary_path"]),
        )
    }
    development_rows = [row for row in analysis_rows if row["true_height_mm"] in DEV_HEIGHTS]
    heldout_rows = [row for row in analysis_rows if row["true_height_mm"] == HELDOUT_HEIGHT]
    if not development_rows or not heldout_rows:
        raise RuntimeError("Both development 30/36/40/46 mm and held-out 50 mm are required")

    metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for height in DEV_HEIGHTS:
        train = [row for row in development_rows if row["true_height_mm"] != height]
        test = [row for row in development_rows if row["true_height_mm"] == height]
        for model in MODELS:
            metric, grouped, fit_info = evaluate_fold(
                train, test, model, "LOHO_development", f"height_{int(height)}mm"
            )
            metrics.append(metric)
            coefficient_rows.extend(fit_info["coefficient_rows"])
            prediction_rows.extend(
                {
                    "cv_scheme": "LOHO_development",
                    "model": model,
                    "heldout_group": f"height_{int(height)}mm",
                    **row,
                }
                for row in grouped
            )
    for rank in range(1, 6):
        train = [row for row in development_rows if row["position_rank"] != rank]
        test = [row for row in development_rows if row["position_rank"] == rank]
        for model in MODELS:
            metric, grouped, fit_info = evaluate_fold(
                train, test, model, "LOPO_position_rank", f"rank_{rank}"
            )
            metrics.append(metric)
            coefficient_rows.extend(fit_info["coefficient_rows"])
            prediction_rows.extend(
                {
                    "cv_scheme": "LOPO_position_rank",
                    "model": model,
                    "heldout_group": f"rank_{rank}",
                    **row,
                }
                for row in grouped
            )

    # Strict 50 mm check: fit on all development only, never use 50 mm in model selection.
    for model in MODELS:
        metric, grouped, fit_info = evaluate_fold(
            development_rows, heldout_rows, model,
            "strict_50mm_validation", "height_50mm_strict_heldout",
        )
        metrics.append(metric)
        coefficient_rows.extend(fit_info["coefficient_rows"])
        prediction_rows.extend(
            {
                "cv_scheme": "strict_50mm_validation",
                "model": model,
                "heldout_group": "height_50mm_strict_heldout",
                **row,
            }
            for row in grouped
        )

    comparison = model_comparison(prediction_rows)
    stability = coefficient_stability(coefficient_rows)
    development_metrics = [
        row for row in metrics
        if row["model"] == "S0" and row["cv_scheme"] in {"LOHO_development", "LOPO_position_rank"}
    ]
    s0_loho = [row for row in development_metrics if row["cv_scheme"] == "LOHO_development"]
    s0_rank5 = next(row for row in development_metrics if row["cv_scheme"] == "LOPO_position_rank" and row["heldout_group"] == "rank_5")
    s0_dev_comparison = next(
        row for row in comparison
        if row["cv_scheme"] == "LOHO_development" and row["model"] == "S0"
    )
    s0_core_improved = sum(bool(row["all_core_metrics_improved"]) for row in development_metrics)
    s0_no_worsening = sum(bool(row["no_metric_worsening"]) for row in development_metrics)
    s0_loho_oob = sum(bool(row["bbox_extrapolation"]) for row in s0_loho)
    s0_unstable_terms = [
        f"{row['cv_scheme']}:{row['model']}:{row['parameter']}"
        for row in stability
        if row["model"] == "S0"
        and row["cv_scheme"] in {"LOHO_development", "LOPO_position_rank"}
        and not row["sign_consistency"]
    ]
    s0_overall_improved = s0_dev_comparison["delta_rmse_mm"] < -EPS
    s0_all_improvement = s0_core_improved == len(development_metrics)
    s0_no_extrapolation = not any(row["bbox_extrapolation"] for row in development_metrics)
    if s0_all_improvement and s0_no_worsening == len(development_metrics) and s0_no_extrapolation:
        feasibility = "SUPPORTED"
    elif s0_overall_improved and s0_core_improved >= len(development_metrics) // 2:
        feasibility = "PARTIAL"
    else:
        feasibility = "NOT_SUPPORTED"
    decision = {
        "SURFACE_CORRECTION_FEASIBILITY": feasibility,
        "MORE_HEIGHT_ACQUISITION_REQUIRED": "NO" if feasibility == "SUPPORTED" else "YES",
        "s0_development_fold_count": len(development_metrics),
        "s0_core_improved_folds": s0_core_improved,
        "s0_no_worsening_folds": s0_no_worsening,
        "s0_unstable_coefficient_terms": s0_unstable_terms,
        "s0_loho_bbox_extrapolation_folds": s0_loho_oob,
        "s0_rank5_rmse_delta_mm": float(s0_rank5["delta_rmse_rmse_mm"] if "delta_rmse_rmse_mm" in s0_rank5 else s0_rank5["delta_rmse_mm"]),
        "s0_rank5_bbox_oob_rate": float(s0_rank5["bbox_oob_point_rate"]),
        "s0_pooled_rmse_delta_mm": float(s0_dev_comparison["delta_rmse_mm"]),
        "s0_all_core_metrics_improved": bool(s0_all_improvement),
        "s0_no_extrapolation": bool(s0_no_extrapolation),
        "s0_overall_improved": bool(s0_overall_improved),
    }

    metric_fields = list(metrics[0].keys())
    prediction_fields = list(prediction_rows[0].keys())
    coefficient_fields = list(coefficient_rows[0].keys())
    stability_fields = list(stability[0].keys())
    write_csv(output / "surface2br_cv_metrics.csv", metrics, metric_fields)
    write_csv(
        output / "surface2br_loho_metrics.csv",
        [row for row in metrics if row["cv_scheme"] == "LOHO_development"],
        metric_fields,
    )
    write_csv(
        output / "surface2br_lopo_metrics.csv",
        [row for row in metrics if row["cv_scheme"] == "LOPO_position_rank"],
        metric_fields,
    )
    write_csv(
        output / "surface2br_50mm_strict_metrics.csv",
        [row for row in metrics if row["cv_scheme"] == "strict_50mm_validation"],
        metric_fields,
    )
    write_csv(output / "surface2br_condition_predictions.csv", prediction_rows, prediction_fields)
    write_csv(output / "surface2br_coefficients.csv", coefficient_rows, coefficient_fields)
    write_csv(output / "surface2br_coefficient_stability.csv", stability, stability_fields)
    write_csv(output / "surface2br_model_comparison.csv", comparison, list(comparison[0].keys()))
    make_plot(output, metrics)
    summary = {
        "SURFACE_CORRECTION_FEASIBILITY": feasibility,
        "MORE_HEIGHT_ACQUISITION_REQUIRED": decision["MORE_HEIGHT_ACQUISITION_REQUIRED"],
        "original_surface2b_conclusion": {
            "Q2_GAP_FILLED": surface2b_summary["Q2_GAP_FILLED"],
            "Q1Q2_STATE_CONSISTENCY": surface2b_summary["Q1Q2_STATE_CONSISTENCY"],
            "SURFACE2C_ALLOWED": surface2b_summary["SURFACE2C_ALLOWED"],
        },
        "protocol": {
            "model_selection_heights_mm": list(DEV_HEIGHTS),
            "heldout_50mm_excluded_from_fit_and_selection": True,
            "condition_equal_weight": True,
            "random_point_split": False,
            "q_redefined": False,
            "c0_refit": False,
            "c1_refit": False,
            "production_validation": False,
        },
        "provenance": {
            "input_paths": {name: str(path) for name, path in input_paths.items()},
            "input_sha256": input_hashes,
            "surface2b_summary_q_definition": surface2b_summary["q_definition"],
        },
        "decision_details": decision,
        "model_formulas": {
            "S0": "a0+a1*q1+a2*q2",
            "S1": "a0+a1*q1+a2*q2+a3*q1^2+a4*q1*q2+a5*q2^2",
            "corrected_residual": "r-F(q1,q2)",
        },
        "model_comparison": comparison,
        "coefficient_stability": stability,
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface2br_summary.json", summary)
    report = report_text(
        input_paths, input_hashes, surface2b_summary, comparison, metrics, stability, decision
    )
    (output / "surface2br_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "SURFACE_CORRECTION_FEASIBILITY": feasibility,
        "MORE_HEIGHT_ACQUISITION_REQUIRED": decision["MORE_HEIGHT_ACQUISITION_REQUIRED"],
        "s0_pooled_rmse_delta_mm": decision["s0_pooled_rmse_delta_mm"],
        "s0_core_improved_folds": f"{s0_core_improved}/{len(development_metrics)}",
        "s0_no_worsening_folds": f"{s0_no_worsening}/{len(development_metrics)}",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
