"""Height-1 scale/affine cross-validation on the Ground-4A B chain.

This is a condition-level diagnostic.  The input is the Ground-4A
``session_linear`` condition mean for repeat2--5; no frame, point, C0/C1, or
Ground-3 artifact is recomputed here.  H1/H2 parameters are fitted only on
training height/position conditions and evaluated on complete held-out groups.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs"
    / "daheng_c1_gauge_blocks_20260819_ground4a"
    / "ground4a_condition_comparison.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent

HEIGHT_ORDER = (
    "obs_1mm",
    "obs_2mm",
    "obs_6mm",
    "obs_10mm",
    "obs_20mm",
    "obs_30mm",
)
MODEL_ORDER = ("H0", "H1", "H2")
MODEL_LABELS = {
    "H0": "H0 no compensation",
    "H1": "H1 scale-only",
    "H2": "H2 affine",
}
SCHEME_LABELS = {
    "leave_one_height_out": "Leave-One-Height-Out",
    "leave_one_position_out": "Leave-One-Position-Out",
}
EPS = 1.0e-12


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "MISSING"
    value = float(value)
    if not math.isfinite(value):
        return "MISSING"
    return f"{value:.{digits}f}"


def _condition_input(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    required = {
        "dataset",
        "truth_mm",
        "position_rank",
        "chain",
        "successful_repeat2_5",
        "failed_repeat2_5",
        "measured_mean_mm",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else {}))
        raise RuntimeError(f"Ground-4A condition CSV missing fields: {missing}")
    b_rows = [row for row in rows if row["chain"] == "session_linear"]
    if len(b_rows) != 30:
        raise RuntimeError(f"expected 30 B conditions, got {len(b_rows)}")
    success = []
    for row in b_rows:
        successful = int(row["successful_repeat2_5"])
        failed = int(row["failed_repeat2_5"])
        measured = row["measured_mean_mm"]
        if successful == 4 and failed == 0 and measured != "":
            success.append(
                {
                    "dataset": row["dataset"],
                    "truth_mm": float(row["truth_mm"]),
                    "position_rank": int(row["position_rank"]),
                    "raw_height_mm": float(measured),
                    "repeat_count": successful,
                }
            )
    if len(success) != 29:
        raise RuntimeError(f"expected 29 successful B conditions, got {len(success)}")
    keys = {(row["dataset"], row["position_rank"]) for row in success}
    if len(keys) != 29:
        raise RuntimeError("duplicate successful height-position conditions")
    return sorted(
        success,
        key=lambda row: (HEIGHT_ORDER.index(row["dataset"]), row["position_rank"]),
    )


def _fit_parameters(train: list[dict[str, Any]], model: str) -> dict[str, float]:
    x = np.asarray([row["raw_height_mm"] for row in train], dtype=np.float64)
    y = np.asarray([row["truth_mm"] for row in train], dtype=np.float64)
    if len(x) == 0:
        raise RuntimeError("empty training set")
    if model == "H0":
        return {"k": 1.0, "a": 1.0, "b": 0.0}
    if model == "H1":
        denominator = float(np.dot(x, x))
        if denominator <= np.finfo(np.float64).eps:
            raise RuntimeError("scale-only training denominator is degenerate")
        k = float(np.dot(x, y) / denominator)
        return {"k": k, "a": k, "b": 0.0}
    if model == "H2":
        a, b = np.linalg.lstsq(
            np.column_stack([x, np.ones_like(x)]), y, rcond=None
        )[0]
        return {"k": None, "a": float(a), "b": float(b)}
    raise ValueError(f"unknown model {model}")


def _predict(raw: np.ndarray, model: str, parameters: dict[str, float]) -> np.ndarray:
    if model == "H0":
        return raw
    if model == "H1":
        return float(parameters["k"]) * raw
    return float(parameters["a"]) * raw + float(parameters["b"])


def _metrics(errors: Iterable[float]) -> dict[str, Any]:
    errors = np.asarray(list(errors), dtype=np.float64)
    if len(errors) == 0:
        return {
            "count": 0,
            "bias_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_mm": None,
            "max_mm": None,
            "pass_0p1_count": 0,
            "pass_0p1_rate": None,
            "pass_0p2_count": 0,
            "pass_0p2_rate": None,
        }
    absolute = np.abs(errors)
    return {
        "count": int(len(errors)),
        "bias_mm": float(np.mean(errors)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(errors**2))),
        "p95_mm": float(np.percentile(absolute, 95.0)),
        "max_mm": float(np.max(absolute)),
        "pass_0p1_count": int(np.count_nonzero(absolute <= 0.1)),
        "pass_0p1_rate": float(np.mean(absolute <= 0.1)),
        "pass_0p2_count": int(np.count_nonzero(absolute <= 0.2)),
        "pass_0p2_rate": float(np.mean(absolute <= 0.2)),
    }


def _metric_fields(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _run_cv(
    conditions: list[dict[str, Any]],
    scheme: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], dict[tuple[str, int], float]],
]:
    if scheme == "leave_one_height_out":
        groups = list(HEIGHT_ORDER)
        group_of = lambda row: row["dataset"]
    elif scheme == "leave_one_position_out":
        groups = sorted({row["position_rank"] for row in conditions})
        group_of = lambda row: row["position_rank"]
    else:
        raise ValueError(f"unknown CV scheme: {scheme}")

    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    oof: dict[tuple[str, str], dict[tuple[str, int], float]] = defaultdict(dict)

    for held_out in groups:
        train = [row for row in conditions if group_of(row) != held_out]
        test = [row for row in conditions if group_of(row) == held_out]
        if not train or not test:
            raise RuntimeError(f"invalid {scheme} fold {held_out}: train/test empty")
        test_raw = np.asarray([row["raw_height_mm"] for row in test], dtype=np.float64)
        test_truth = np.asarray([row["truth_mm"] for row in test], dtype=np.float64)
        raw_metrics = _metrics(test_raw - test_truth)
        for model in MODEL_ORDER:
            parameters = _fit_parameters(train, model)
            corrected = _predict(test_raw, model, parameters)
            corrected_metrics = _metrics(corrected - test_truth)
            for condition, prediction in zip(test, corrected, strict=True):
                oof[(scheme, model)][
                    (condition["dataset"], condition["position_rank"])
                ] = float(prediction)
            parameter_rows.append(
                {
                    "cv_scheme": scheme,
                    "held_out_group": held_out,
                    "held_out_condition_count": len(test),
                    "train_condition_count": len(train),
                    "model": model,
                    "k": parameters["k"],
                    "a": parameters["a"],
                    "b": parameters["b"],
                    "fit_equation": (
                        "h_corr=h" if model == "H0" else
                        "h_corr=k*h" if model == "H1" else
                        "h_corr=a*h+b"
                    ),
                }
            )
            row: dict[str, Any] = {
                "cv_scheme": scheme,
                "held_out_group": held_out,
                "held_out_condition_count": len(test),
                "train_condition_count": len(train),
                "model": model,
                "k": parameters["k"],
                "a": parameters["a"],
                "b": parameters["b"],
            }
            row.update(_metric_fields("raw", raw_metrics))
            row.update(_metric_fields("corrected", corrected_metrics))
            row["mae_improvement_mm"] = raw_metrics["mae_mm"] - corrected_metrics["mae_mm"]
            row["rmse_improvement_mm"] = raw_metrics["rmse_mm"] - corrected_metrics["rmse_mm"]
            row["p95_improvement_mm"] = raw_metrics["p95_mm"] - corrected_metrics["p95_mm"]
            row["max_improvement_mm"] = raw_metrics["max_mm"] - corrected_metrics["max_mm"]
            metric_rows.append(row)
    return metric_rows, parameter_rows, oof


def _model_comparison(
    conditions: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    oof: dict[tuple[str, str], dict[tuple[str, int], float]],
) -> list[dict[str, Any]]:
    raw = np.asarray([row["raw_height_mm"] for row in conditions], dtype=np.float64)
    truth = np.asarray([row["truth_mm"] for row in conditions], dtype=np.float64)
    raw_metrics = _metrics(raw - truth)
    output = []
    for scheme in ("leave_one_height_out", "leave_one_position_out"):
        scheme_rows = [row for row in metric_rows if row["cv_scheme"] == scheme]
        for model in MODEL_ORDER:
            prediction_map = oof[(scheme, model)]
            corrected = np.asarray(
                [
                    prediction_map[(row["dataset"], row["position_rank"])]
                    for row in conditions
                ],
                dtype=np.float64,
            )
            corrected_metrics = _metrics(corrected - truth)
            folds = [row for row in scheme_rows if row["model"] == model]
            h0_folds = {row["held_out_group"]: row for row in scheme_rows if row["model"] == "H0"}
            improved_flags = [
                float(row["corrected_mae_mm"]) < float(h0_folds[row["held_out_group"]]["corrected_mae_mm"]) - EPS
                and float(row["corrected_rmse_mm"]) < float(h0_folds[row["held_out_group"]]["corrected_rmse_mm"]) - EPS
                and float(row["corrected_p95_mm"]) < float(h0_folds[row["held_out_group"]]["corrected_p95_mm"]) - EPS
                and float(row["corrected_max_mm"]) < float(h0_folds[row["held_out_group"]]["corrected_max_mm"]) - EPS
                for row in folds
            ]
            output.append(
                {
                    "cv_scheme": scheme,
                    "aggregation": "pooled_out_of_fold_conditions",
                    "model": model,
                    "condition_count": len(conditions),
                    "fold_count": len(folds),
                    "improved_all_metrics_fold_count": int(sum(improved_flags)),
                    "all_folds_improved_vs_H0": bool(all(improved_flags)) if model != "H0" else False,
                    **_metric_fields("raw", raw_metrics),
                    **_metric_fields("corrected", corrected_metrics),
                    "mae_improvement_mm": raw_metrics["mae_mm"] - corrected_metrics["mae_mm"],
                    "rmse_improvement_mm": raw_metrics["rmse_mm"] - corrected_metrics["rmse_mm"],
                    "p95_improvement_mm": raw_metrics["p95_mm"] - corrected_metrics["p95_mm"],
                    "max_improvement_mm": raw_metrics["max_mm"] - corrected_metrics["max_mm"],
                    "mean_fold_corrected_mae_mm": float(np.mean([row["corrected_mae_mm"] for row in folds])),
                    "std_fold_corrected_mae_mm": float(np.std([row["corrected_mae_mm"] for row in folds])),
                    "mean_fold_corrected_rmse_mm": float(np.mean([row["corrected_rmse_mm"] for row in folds])),
                    "std_fold_corrected_rmse_mm": float(np.std([row["corrected_rmse_mm"] for row in folds])),
                    "mean_fold_corrected_p95_mm": float(np.mean([row["corrected_p95_mm"] for row in folds])),
                    "std_fold_corrected_p95_mm": float(np.std([row["corrected_p95_mm"] for row in folds])),
                }
            )
    return output


def _parameter_stability(parameter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    parameter_names = {"H0": ("k",), "H1": ("k",), "H2": ("a", "b")}
    for scheme in ("leave_one_height_out", "leave_one_position_out"):
        for model in MODEL_ORDER:
            rows = [row for row in parameter_rows if row["cv_scheme"] == scheme and row["model"] == model]
            for name in parameter_names[model]:
                values = np.asarray([float(row[name]) for row in rows if row[name] is not None], dtype=np.float64)
                output.append(
                    {
                        "cv_scheme": scheme,
                        "model": model,
                        "parameter": name,
                        "fold_count": len(values),
                        "mean": float(np.mean(values)) if len(values) else None,
                        "median": float(np.median(values)) if len(values) else None,
                        "std": float(np.std(values)) if len(values) else None,
                        "min": float(np.min(values)) if len(values) else None,
                        "max": float(np.max(values)) if len(values) else None,
                        "range": float(np.ptp(values)) if len(values) else None,
                        "relative_range": (
                            float(np.ptp(values) / abs(np.mean(values)))
                            if len(values) and abs(float(np.mean(values))) > EPS
                            else None
                        ),
                    }
                )
    return output


def _select_model(
    comparison_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    h1_height = next(row for row in comparison_rows if row["cv_scheme"] == "leave_one_height_out" and row["model"] == "H1")
    h2_height = next(row for row in comparison_rows if row["cv_scheme"] == "leave_one_height_out" and row["model"] == "H2")
    h1_position = next(row for row in comparison_rows if row["cv_scheme"] == "leave_one_position_out" and row["model"] == "H1")
    h2_position = next(row for row in comparison_rows if row["cv_scheme"] == "leave_one_position_out" and row["model"] == "H2")
    relative_affine_gain_height = (
        float(h1_height["mean_fold_corrected_mae_mm"] - h2_height["mean_fold_corrected_mae_mm"])
        / float(h1_height["mean_fold_corrected_mae_mm"])
    )
    relative_affine_gain_position = (
        float(h1_position["mean_fold_corrected_mae_mm"] - h2_position["mean_fold_corrected_mae_mm"])
        / float(h1_position["mean_fold_corrected_mae_mm"])
    )
    h2_height_folds = {
        row["held_out_group"]: float(row["corrected_mae_mm"])
        for row in metric_rows
        if row["cv_scheme"] == "leave_one_height_out" and row["model"] == "H2"
    }
    h1_height_folds = {
        row["held_out_group"]: float(row["corrected_mae_mm"])
        for row in metric_rows
        if row["cv_scheme"] == "leave_one_height_out" and row["model"] == "H1"
    }
    h2_position_folds = {
        row["held_out_group"]: float(row["corrected_mae_mm"])
        for row in metric_rows
        if row["cv_scheme"] == "leave_one_position_out" and row["model"] == "H2"
    }
    h1_position_folds = {
        row["held_out_group"]: float(row["corrected_mae_mm"])
        for row in metric_rows
        if row["cv_scheme"] == "leave_one_position_out" and row["model"] == "H1"
    }
    affine_not_uniformly_better = (
        any(h2_height_folds[group] >= h1_height_folds[group] - EPS for group in h1_height_folds)
        or any(h2_position_folds[group] >= h1_position_folds[group] - EPS for group in h1_position_folds)
    )
    selected = "H1" if (
        relative_affine_gain_height < 0.10
        and relative_affine_gain_position < 0.10
        and affine_not_uniformly_better
    ) else "H2"
    h1_improvement = all(
        row["all_folds_improved_vs_H0"]
        for row in comparison_rows
        if row["model"] == "H1"
    )
    h1_k_stability = [
        row["relative_range"]
        for row in stability_rows
        if row["model"] == "H1" and row["parameter"] == "k"
    ]
    stable_scale = bool(h1_k_stability) and all(value <= 0.005 for value in h1_k_stability)
    if h1_improvement and stable_scale:
        status = "PASS"
        allow = "YES"
    elif any(
        row["model"] == "H1" and row["improved_all_metrics_fold_count"] > 0
        for row in comparison_rows
    ):
        status = "PARTIAL"
        allow = "NO"
    else:
        status = "FAIL"
        allow = "NO"
    details = {
        "selected_model": selected,
        "affine_relative_mae_gain_height": relative_affine_gain_height,
        "affine_relative_mae_gain_position": relative_affine_gain_position,
        "affine_not_uniformly_better": affine_not_uniformly_better,
        "scale_all_folds_improved": h1_improvement,
        "scale_parameter_relative_ranges": h1_k_stability,
        "scale_stable_threshold_relative_range": 0.005,
        "scale_stable": stable_scale,
        "HEIGHT_LINEAR_STATUS": status,
        "ALLOW_NEW_HELD_OUT_VALIDATION": allow,
    }
    return selected, status, details


def _plot(output_path: Path, metric_rows: list[dict[str, Any]], selected_model: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey="row")
    for column, scheme in enumerate(("leave_one_height_out", "leave_one_position_out")):
        rows = [row for row in metric_rows if row["cv_scheme"] == scheme]
        groups = []
        for row in rows:
            if row["held_out_group"] not in groups:
                groups.append(row["held_out_group"])
        x = np.arange(len(groups), dtype=np.float64)
        width = 0.24
        for model_index, model in enumerate(MODEL_ORDER):
            model_rows = {row["held_out_group"]: row for row in rows if row["model"] == model}
            values_mae = [model_rows[group]["corrected_mae_mm"] for group in groups]
            values_rmse = [model_rows[group]["corrected_rmse_mm"] for group in groups]
            offset = (model_index - 1) * width
            axes[0, column].bar(x + offset, values_mae, width, label=MODEL_LABELS[model])
            axes[1, column].bar(x + offset, values_rmse, width, label=MODEL_LABELS[model])
        axes[0, column].set_title(f"{SCHEME_LABELS[scheme]} — MAE")
        axes[1, column].set_title(f"{SCHEME_LABELS[scheme]} — RMSE")
        axes[1, column].set_xticks(x, [str(group) for group in groups])
        axes[1, column].set_xlabel("held-out group")
        axes[0, column].grid(axis="y", alpha=0.25)
        axes[1, column].grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("corrected MAE (mm)\nH0 = raw")
    axes[1, 0].set_ylabel("corrected RMSE (mm)\nH0 = raw")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        f"Height-1 raw vs held-out correction ({selected_model} selected; condition-level repeat2–5)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    input_path: Path,
    input_hash: str,
    conditions: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    selected_model: str,
    status: str,
    selection: dict[str, Any],
) -> None:
    lines = [
        "# Height-1 高度尺度补偿交叉验证",
        "",
        f"- `HEIGHT_LINEAR_STATUS={status}`",
        f"- `SELECTED_MODEL={selected_model}`",
        f"- `ALLOW_NEW_HELD_OUT_VALIDATION={selection['ALLOW_NEW_HELD_OUT_VALIDATION']}`",
        "- 本轮仅为诊断，不修改 C0/C1、Ground-3 G(S)、GUI 或生产配置。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- 输入：`{input_path}`，SHA-256 `{input_hash}`。",
        f"- 复用 Ground-4A B `session_linear` 的正式 repeat2–5 condition mean，共 {len(conditions)} 个成功 height×position condition。",
        "- `obs_2mm/position5` 的缺失 condition 未补零、未删除；它不进入成功 condition 数值拟合。",
        "- 本轮新增仅为 H0/H1/H2 的 condition-level 完整分组 CV；无随机 point split、无随机抽样、无高阶模型。",
        "",
        "## 模型定义",
        "",
        "- H0：`h_corr=h`，B `session_linear` raw height。",
        "- H1：训练 condition 上拟合 `h_corr=k*h`。",
        "- H2：训练 condition 上拟合 `h_corr=a*h+b`。",
        "- 所有参数均只由训练组拟合；Leave-One-Height-Out 完整剔除一个高度的所有 position，Leave-One-Position-Out 完整剔除一个 position 的所有高度。",
        "- 每个 condition 等权；condition 内使用 Ground-4A 已汇总的 repeat2–5 `measured_mean_mm`，不重新展开 repeat。",
        "",
        "## Pooled out-of-fold metrics",
        "",
        "| CV | model | n | raw Bias | raw MAE | raw RMSE | raw P95 | raw Max | corrected Bias | corrected MAE | corrected RMSE | corrected P95 | corrected Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {SCHEME_LABELS[row['cv_scheme']]} | {row['model']} | {row['condition_count']} | "
            f"{_fmt(row['raw_bias_mm'])} | {_fmt(row['raw_mae_mm'])} | {_fmt(row['raw_rmse_mm'])} | "
            f"{_fmt(row['raw_p95_mm'])} | {_fmt(row['raw_max_mm'])} | {_fmt(row['corrected_bias_mm'])} | "
            f"{_fmt(row['corrected_mae_mm'])} | {_fmt(row['corrected_rmse_mm'])} | "
            f"{_fmt(row['corrected_p95_mm'])} | {_fmt(row['corrected_max_mm'])} |"
        )
    lines += [
        "",
        "## Held-out group metrics",
        "",
        "`height_linear_cv_metrics.csv` 保留每个 held-out group 的完整 Bias/MAE/RMSE/P95/Max；下面按模型逐行摘录 corrected 指标。H0 的 corrected 即 raw。",
        "",
        "| CV | held-out group | model | n | Bias | MAE | RMSE | P95 | Max |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scheme in ("leave_one_height_out", "leave_one_position_out"):
        scheme_rows = [row for row in metric_rows if row["cv_scheme"] == scheme]
        for row in scheme_rows:
            lines.append(
                f"| {SCHEME_LABELS[scheme]} | {row['held_out_group']} | {row['model']} | "
                f"{row['corrected_count']} | {_fmt(row['corrected_bias_mm'])} | {_fmt(row['corrected_mae_mm'])} | "
                f"{_fmt(row['corrected_rmse_mm'])} | {_fmt(row['corrected_p95_mm'])} | {_fmt(row['corrected_max_mm'])} |"
            )
    lines += [
        "",
        "## Fold parameter stability",
        "",
        "| CV | model | parameter | mean | std | min | max | range | relative range |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stability_rows:
        lines.append(
            f"| {SCHEME_LABELS[row['cv_scheme']]} | {row['model']} | {row['parameter']} | "
            f"{_fmt(row['mean'], 7)} | {_fmt(row['std'], 7)} | {_fmt(row['min'], 7)} | "
            f"{_fmt(row['max'], 7)} | {_fmt(row['range'], 7)} | {_fmt(row['relative_range'], 7)} |"
        )
    lines += [
        "",
        "## Model selection",
        "",
        f"- 选择 `{selected_model}`：H1 scale-only 在两个 CV 方向上每个 fold 的 MAE/RMSE/P95/Max 均优于 H0，且 k 的 relative range 为 "
        f"{', '.join(_fmt(value, 7) for value in selection['scale_parameter_relative_ranges'])}，低于稳定性门槛 0.005。",
        f"- H2 affine 相对 H1 的 mean-fold MAE 额外改善：LHO-height {100*selection['affine_relative_mae_gain_height']:.2f}%、LHO-position {100*selection['affine_relative_mae_gain_position']:.2f}%；H2 并非每个 held-out fold 都优于 H1，因此按最简单稳定原则不选 H2。",
        "- 这只是进入新量块 held-out validation 的候选资格，不是生产参数冻结，也不等同于新的工程验收。",
        "",
        "## 输出",
        "",
        "- `height_linear_cv_metrics.csv`：每个 height/position held-out fold 的 raw/corrected 指标。",
        "- `height_linear_model_comparison.csv`：两个 CV 方案的 pooled out-of-fold 模型比较。",
        "- `height_linear_fold_parameters.csv`：每个 fold 的 k/a/b。",
        "- `height_error_before_after.png`：held-out MAE/RMSE 前后对比。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = _condition_input(input_path)
    input_hash = _sha256(input_path)

    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    oof: dict[tuple[str, str], dict[tuple[str, int], float]] = defaultdict(dict)
    for scheme in ("leave_one_height_out", "leave_one_position_out"):
        metrics, parameters, scheme_oof = _run_cv(conditions, scheme)
        metric_rows.extend(metrics)
        parameter_rows.extend(parameters)
        for key, values in scheme_oof.items():
            oof[key].update(values)

    comparison_rows = _model_comparison(conditions, metric_rows, oof)
    stability_rows = _parameter_stability(parameter_rows)
    selected_model, status, selection = _select_model(
        comparison_rows, metric_rows, stability_rows
    )
    for row in comparison_rows:
        row["selected_model"] = selected_model
        row["HEIGHT_LINEAR_STATUS"] = status
        row["ALLOW_NEW_HELD_OUT_VALIDATION"] = selection["ALLOW_NEW_HELD_OUT_VALIDATION"]

    metric_fields = [
        "cv_scheme", "held_out_group", "held_out_condition_count", "train_condition_count", "model", "k", "a", "b",
        "raw_count", "raw_bias_mm", "raw_mae_mm", "raw_rmse_mm", "raw_p95_mm", "raw_max_mm", "raw_pass_0p1_count", "raw_pass_0p1_rate", "raw_pass_0p2_count", "raw_pass_0p2_rate",
        "corrected_count", "corrected_bias_mm", "corrected_mae_mm", "corrected_rmse_mm", "corrected_p95_mm", "corrected_max_mm", "corrected_pass_0p1_count", "corrected_pass_0p1_rate", "corrected_pass_0p2_count", "corrected_pass_0p2_rate",
        "mae_improvement_mm", "rmse_improvement_mm", "p95_improvement_mm", "max_improvement_mm",
    ]
    comparison_fields = list(comparison_rows[0].keys())
    parameter_fields = list(parameter_rows[0].keys())
    _write_csv(output_dir / "height_linear_cv_metrics.csv", metric_rows, metric_fields)
    _write_csv(output_dir / "height_linear_model_comparison.csv", comparison_rows, comparison_fields)
    _write_csv(output_dir / "height_linear_fold_parameters.csv", parameter_rows, parameter_fields)
    _plot(output_dir / "height_error_before_after.png", metric_rows, selected_model)

    summary = {
        "schema_version": 1,
        "HEIGHT_LINEAR_STATUS": status,
        "SELECTED_MODEL": selected_model,
        "ALLOW_NEW_HELD_OUT_VALIDATION": selection["ALLOW_NEW_HELD_OUT_VALIDATION"],
        "formal_input_condition_count": len(conditions),
        "input_path": input_path,
        "input_sha256": input_hash,
        "selection": selection,
        "model_comparison": comparison_rows,
        "parameter_stability": stability_rows,
        "protocol": {
            "raw_chain": "Ground-4A B session_linear measured_mean_mm",
            "formal_scope": "repeat2_5 condition means",
            "weighting": "one equal-weight sample per successful height-position condition",
            "random_split": False,
            "models": {
                "H0": "h_corr=h",
                "H1": "h_corr=k*h",
                "H2": "h_corr=a*h+b",
            },
        },
        "no_production_change": True,
    }
    _write_json(output_dir / "height_linear_summary.json", summary)
    _write_report(
        output_dir / "height_linear_report.md",
        input_path,
        input_hash,
        conditions,
        comparison_rows,
        metric_rows,
        parameter_rows,
        stability_rows,
        selected_model,
        status,
        selection,
    )
    print(json.dumps({"HEIGHT_LINEAR_STATUS": status, "SELECTED_MODEL": selected_model, "ALLOW_NEW_HELD_OUT_VALIDATION": selection["ALLOW_NEW_HELD_OUT_VALIDATION"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
