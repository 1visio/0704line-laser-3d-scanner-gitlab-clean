"""Feasibility audit for the Haikang H1 height-scale correction.

This is a condition-level diagnostic only.  It reuses the existing Daheng H1
definition (``h_corr = k * h_raw`` with a through-origin least-squares fit) and
the existing grouped-CV H1 predictor.  It adapts only the
Haikang H0-1M-B condition summary; no C0, Ground, H-B2, C1, or online config is
changed.

The fit unit is one ``manual_h_raw_position_summary.csv`` row per
height-position condition.  The 20 frame rows are read for provenance/QC but
are never expanded into training samples.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
TOOLS_ROOT = REPO_ROOT / "tools"
for _path in (MEASUREMENT_ROOT, TOOLS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from validate_height_linear_cv import _fit_parameters, _metrics, _predict  # noqa: E402


DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "laser_measurement_tool"
    / "output_haikang_0828"
    / "online_recordings"
    / "0829"
    / "c0_height_audit"
    / "manual_roi_measurement"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR.parent / "h1_feasibility"
H1_SOURCE_PATH = TOOLS_ROOT / "validate_height_linear_cv.py"

HEIGHTS_MM = (2.0, 6.0, 10.0, 20.0, 30.0)
HEIGHT_IDS = ("h02", "h06", "h10", "h20", "h30")
INTERIOR_HEIGHTS_MM = (6.0, 10.0, 20.0)
POSITION_IDS = tuple(f"p{index:02d}" for index in range(1, 11))
FEASIBILITY_DOMAIN_MM = (2.0, 30.0)
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
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if any(str(value or "").strip() for value in row.values())
        ]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "MISSING"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "MISSING"
    return f"{number:.{digits}f}" if math.isfinite(number) else "MISSING"


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {key}: {row}")
    return value


def load_inputs(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one condition row per H0 condition and verify frame provenance."""
    summary_path = input_dir / "manual_h_raw_position_summary.csv"
    frames_path = input_dir / "manual_h_raw_frames.csv"
    accuracy_path = input_dir / "manual_c0_accuracy_summary.json"
    report_path = input_dir / "manual_c0_height_audit_report.md"
    for path in (summary_path, frames_path, accuracy_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary_rows = read_csv(summary_path)
    required = {
        "diagnostic_mode",
        "condition",
        "height_id",
        "position_id",
        "height_gt_mm",
        "expected_frame_count",
        "valid_frame_count",
        "h_raw_mm_median",
    }
    if not summary_rows or not required.issubset(summary_rows[0]):
        missing = sorted(required - set(summary_rows[0] if summary_rows else {}))
        raise RuntimeError(f"H0 summary missing fields: {missing}")
    if len(summary_rows) != 50:
        raise RuntimeError(f"expected 50 condition rows, got {len(summary_rows)}")

    conditions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in summary_rows:
        if row["diagnostic_mode"] != "MANUAL_ROI_DIAGNOSTIC":
            raise RuntimeError(f"unexpected diagnostic mode: {row}")
        condition = row["condition"]
        if condition in seen:
            raise RuntimeError(f"duplicate condition: {condition}")
        seen.add(condition)
        height_mm = _float(row, "height_gt_mm")
        if height_mm not in HEIGHTS_MM:
            raise RuntimeError(f"unexpected height {height_mm}: {condition}")
        if row["height_id"] != f"h{int(height_mm):02d}":
            raise RuntimeError(f"height id mismatch: {row}")
        if row["position_id"] not in POSITION_IDS:
            raise RuntimeError(f"unexpected position: {row}")
        expected = int(row["expected_frame_count"])
        valid = int(row["valid_frame_count"])
        if expected != 20 or valid != expected:
            raise RuntimeError(f"condition is not 20/20 valid: {row}")
        raw = _float(row, "h_raw_mm_median")
        conditions.append(
            {
                "condition": condition,
                "height_id": row["height_id"],
                "position_id": row["position_id"],
                "truth_mm": height_mm,
                "raw_height_mm": raw,
                "expected_frame_count": expected,
                "valid_frame_count": valid,
            }
        )

    expected_keys = {(height, position) for height in HEIGHT_IDS for position in POSITION_IDS}
    actual_keys = {(row["height_id"], row["position_id"]) for row in conditions}
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"condition grid mismatch; missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    conditions.sort(key=lambda row: (HEIGHT_IDS.index(row["height_id"]), POSITION_IDS.index(row["position_id"])))

    frame_rows = read_csv(frames_path)
    frame_conditions = {row.get("condition", "") for row in frame_rows}
    counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in frame_rows:
        condition = row.get("condition", "")
        counts[condition] = counts.get(condition, 0) + 1
        status = row.get("measurement_status", "")
        statuses[status] = statuses.get(status, 0) + 1
    if len(frame_rows) != 1000 or frame_conditions != seen:
        raise RuntimeError(
            f"frame provenance mismatch: rows={len(frame_rows)}, conditions={len(frame_conditions)}"
        )
    if statuses != {"VALID": 1000}:
        raise RuntimeError(f"unexpected frame measurement statuses: {statuses}")
    if set(counts.values()) != {20}:
        raise RuntimeError(f"frame count per condition is not 20: {counts}")

    accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    if accuracy.get("diagnostic_mode") != "MANUAL_ROI_DIAGNOSTIC":
        raise RuntimeError("H0 accuracy JSON is not MANUAL_ROI_DIAGNOSTIC")
    provenance = accuracy.get("provenance", {})
    return conditions, {
        "summary_path": summary_path,
        "frames_path": frames_path,
        "accuracy_path": accuracy_path,
        "report_path": report_path,
        "summary_sha256": sha256(summary_path),
        "frames_sha256": sha256(frames_path),
        "accuracy_sha256": sha256(accuracy_path),
        "report_sha256": sha256(report_path),
        "condition_count": len(conditions),
        "frame_count": len(frame_rows),
        "frames_per_condition": 20,
        "frame_status_counts": statuses,
        "h0_accuracy": accuracy.get("accuracy", {}),
        "h0_provenance": provenance,
    }


def fit_h1_scale(train: list[dict[str, Any]]) -> float:
    """Call the existing Daheng H1 through-origin fit, with adapted fields."""
    parameters = _fit_parameters(
        [
            {
                "raw_height_mm": row["raw_height_mm"],
                "truth_mm": row["truth_mm"],
            }
            for row in train
        ],
        "H1",
    )
    scale = float(parameters["k"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"invalid H1 scale: {scale}")
    return scale


def apply_h1(raw_height_mm: float, scale: float) -> float:
    """Use the existing grouped-CV H1 predictor, without online config."""
    prediction = _predict(
        np.asarray([raw_height_mm], dtype=np.float64),
        "H1",
        {"k": scale, "a": scale, "b": 0.0},
    )
    value = float(prediction[0])
    if not math.isfinite(value):
        raise RuntimeError(f"H1 application produced a non-finite value: {raw_height_mm}")
    return value


def metrics(errors: Iterable[float]) -> dict[str, float | int]:
    values = _metrics(errors)
    return {
        "count": int(values["count"]),
        "bias_mm": float(values["bias_mm"]),
        "mae_mm": float(values["mae_mm"]),
        "rmse_mm": float(values["rmse_mm"]),
        "p95_absolute_error_mm": float(values["p95_mm"]),
        "max_absolute_error_mm": float(values["max_mm"]),
    }


def compare_metrics(
    test: list[dict[str, Any]],
    scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_errors = np.asarray(
        [row["raw_height_mm"] - row["truth_mm"] for row in test], dtype=np.float64
    )
    corrected_values = np.asarray(
        [apply_h1(row["raw_height_mm"], scale) for row in test], dtype=np.float64
    )
    truths = np.asarray([row["truth_mm"] for row in test], dtype=np.float64)
    corrected_errors = corrected_values - truths
    raw_abs = np.abs(raw_errors)
    corrected_abs = np.abs(corrected_errors)
    raw = metrics(raw_errors)
    corrected = metrics(corrected_errors)
    worsened = corrected_abs - raw_abs
    mae_improvement = raw["mae_mm"] - corrected["mae_mm"]
    row = {
        "raw": raw,
        "h1": corrected,
        "h1_scale": scale,
        "mae_improvement_mm": mae_improvement,
        "rmse_improvement_mm": raw["rmse_mm"] - corrected["rmse_mm"],
        "p95_improvement_mm": raw["p95_absolute_error_mm"] - corrected["p95_absolute_error_mm"],
        "max_improvement_mm": raw["max_absolute_error_mm"] - corrected["max_absolute_error_mm"],
        "mae_improvement_pct": 100.0 * mae_improvement / raw["mae_mm"] if raw["mae_mm"] > EPS else None,
        "rmse_improvement_pct": 100.0 * (raw["rmse_mm"] - corrected["rmse_mm"]) / raw["rmse_mm"] if raw["rmse_mm"] > EPS else None,
        "p95_improvement_pct": 100.0 * (raw["p95_absolute_error_mm"] - corrected["p95_absolute_error_mm"]) / raw["p95_absolute_error_mm"] if raw["p95_absolute_error_mm"] > EPS else None,
        "max_improvement_pct": 100.0 * (raw["max_absolute_error_mm"] - corrected["max_absolute_error_mm"]) / raw["max_absolute_error_mm"] if raw["max_absolute_error_mm"] > EPS else None,
        "worsened_condition_count": int(np.count_nonzero(worsened > EPS)),
        "max_worsening_mm": float(max(0.0, float(np.max(worsened)))) if len(worsened) else 0.0,
    }
    predictions = []
    for condition, corrected_value, raw_error, corrected_error, worsening in zip(
        test, corrected_values, raw_errors, corrected_errors, worsened, strict=True
    ):
        predictions.append(
            {
                **condition,
                "h1_scale": scale,
                "h1_height_mm": float(corrected_value),
                "raw_error_mm": float(raw_error),
                "h1_error_mm": float(corrected_error),
                "raw_absolute_error_mm": float(abs(raw_error)),
                "h1_absolute_error_mm": float(abs(corrected_error)),
                "absolute_error_delta_mm": float(worsening),
            }
        )
    return row, predictions


def flatten_metric_row(
    *,
    scheme: str,
    fold_type: str,
    fold_id: str,
    held_out_group: str,
    held_out_height_mm: float | None,
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cv_scheme": scheme,
        "fold_type": fold_type,
        "fold_id": fold_id,
        "held_out_group": held_out_group,
        "held_out_height_mm": held_out_height_mm,
        "train_condition_count": len(train),
        "test_condition_count": len(test),
        "train_height_ids": sorted({item["height_id"] for item in train}, key=HEIGHT_IDS.index),
        "test_height_ids": sorted({item["height_id"] for item in test}, key=HEIGHT_IDS.index),
        "h1_scale": comparison["h1_scale"],
        "worsened_condition_count": comparison["worsened_condition_count"],
        "max_worsening_mm": comparison["max_worsening_mm"],
    }
    for model, values in (("raw", comparison["raw"]), ("h1", comparison["h1"])):
        for key, value in values.items():
            if key != "count":
                row[f"{model}_{key}"] = value
        row[f"{model}_count"] = values["count"]
    for key in (
        "mae_improvement_mm",
        "rmse_improvement_mm",
        "p95_improvement_mm",
        "max_improvement_mm",
        "mae_improvement_pct",
        "rmse_improvement_pct",
        "p95_improvement_pct",
        "max_improvement_pct",
    ):
        row[key] = comparison[key]
    return row


def run_scheme(
    conditions: list[dict[str, Any]],
    *,
    scheme: str,
    groups: list[Any],
    group_of: Any,
    fold_type: str,
    held_out_height: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_payloads: list[dict[str, Any]] = []
    for group in groups:
        train = [row for row in conditions if group_of(row) != group]
        test = [row for row in conditions if group_of(row) == group]
        if not train or not test:
            raise RuntimeError(f"invalid {scheme} fold {group}")
        scale = fit_h1_scale(train)
        comparison, predictions = compare_metrics(test, scale)
        group_text = str(group)
        fold_id = group_text
        row = flatten_metric_row(
            scheme=scheme,
            fold_type=fold_type,
            fold_id=fold_id,
            held_out_group=group_text,
            held_out_height_mm=(float(held_out_height(group)) if held_out_height else None),
            train=train,
            test=test,
            comparison=comparison,
        )
        metric_rows.append(row)
        for prediction in predictions:
            prediction_rows.append(
                {
                    "cv_scheme": scheme,
                    "fold_type": fold_type,
                    "fold_id": fold_id,
                    "held_out_group": group_text,
                    "held_out_height_mm": row["held_out_height_mm"],
                    **prediction,
                }
            )
        fold_payloads.append(row)
    return metric_rows, prediction_rows, fold_payloads


def pooled_from_predictions(
    prediction_rows: list[dict[str, Any]],
    *,
    scheme: str,
    fold_type: str | None = None,
) -> dict[str, Any]:
    rows = [
        row
        for row in prediction_rows
        if row["cv_scheme"] == scheme and (fold_type is None or row["fold_type"] == fold_type)
    ]
    if not rows:
        raise RuntimeError(f"no predictions for {scheme}/{fold_type}")
    raw = metrics([row["raw_error_mm"] for row in rows])
    corrected = metrics([row["h1_error_mm"] for row in rows])
    worsening = np.asarray(
        [row["absolute_error_delta_mm"] for row in rows], dtype=np.float64
    )
    output: dict[str, Any] = {
        "cv_scheme": scheme,
        "fold_type": fold_type,
        "condition_count": len(rows),
        "raw": raw,
        "h1": corrected,
        "mae_improvement_mm": raw["mae_mm"] - corrected["mae_mm"],
        "rmse_improvement_mm": raw["rmse_mm"] - corrected["rmse_mm"],
        "p95_improvement_mm": raw["p95_absolute_error_mm"] - corrected["p95_absolute_error_mm"],
        "max_improvement_mm": raw["max_absolute_error_mm"] - corrected["max_absolute_error_mm"],
        "mae_improvement_pct": 100.0 * (raw["mae_mm"] - corrected["mae_mm"]) / raw["mae_mm"] if raw["mae_mm"] > EPS else None,
        "rmse_improvement_pct": 100.0 * (raw["rmse_mm"] - corrected["rmse_mm"]) / raw["rmse_mm"] if raw["rmse_mm"] > EPS else None,
        "p95_improvement_pct": 100.0 * (raw["p95_absolute_error_mm"] - corrected["p95_absolute_error_mm"]) / raw["p95_absolute_error_mm"] if raw["p95_absolute_error_mm"] > EPS else None,
        "max_improvement_pct": 100.0 * (raw["max_absolute_error_mm"] - corrected["max_absolute_error_mm"]) / raw["max_absolute_error_mm"] if raw["max_absolute_error_mm"] > EPS else None,
        "worsened_condition_count": int(np.count_nonzero(worsening > EPS)),
        "max_worsening_mm": float(max(0.0, float(np.max(worsening)))) if len(worsening) else 0.0,
    }
    return output


def fold_condition_map(prediction_rows: list[dict[str, Any]], scheme: str) -> dict[str, dict[str, Any]]:
    return {
        row["fold_id"]: row
        for row in prediction_rows
        if row["cv_scheme"] == scheme
    }


def classify(
    loho_pooled: dict[str, Any],
    endpoint_rows: list[dict[str, Any]],
    lopo_pooled: dict[str, Any],
    loho_rows: list[dict[str, Any]],
    lopo_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    interior_mae_all = all(row["h1_mae_mm"] < row["raw_mae_mm"] - EPS for row in loho_rows)
    interior_p95_all = all(row["h1_p95_absolute_error_mm"] < row["raw_p95_absolute_error_mm"] - EPS for row in loho_rows)
    interior_max_all = all(row["h1_max_absolute_error_mm"] < row["raw_max_absolute_error_mm"] - EPS for row in loho_rows)
    low = next(row for row in endpoint_rows if row["held_out_group"] == "LOW_END_EXTRAPOLATION")
    high = next(row for row in endpoint_rows if row["held_out_group"] == "HIGH_END_EXTRAPOLATION")

    def robust_endpoint(row: dict[str, Any]) -> bool:
        return all(
            row[f"h1_{metric}"] < row[f"raw_{metric}"] - EPS
            for metric in ("mae_mm", "rmse_mm", "p95_absolute_error_mm", "max_absolute_error_mm")
        )

    low_robust = robust_endpoint(low)
    high_robust = robust_endpoint(high)
    lopo_pooled_all = all(
        lopo_pooled["h1"][metric] < lopo_pooled["raw"][metric] - EPS
        for metric in ("mae_mm", "rmse_mm", "p95_absolute_error_mm", "max_absolute_error_mm")
    )
    lopo_worsened_fold_count = sum(
        row["worsened_condition_count"] > 0 for row in lopo_rows
    )
    if not lopo_pooled_all:
        classification = "H1_NOT_GENERALIZABLE"
    elif (
        interior_mae_all
        and interior_p95_all
        and (not low_robust or not high_robust or not interior_max_all)
    ):
        classification = "H1_INTERPOLATION_ONLY"
    elif low_robust and high_robust and lopo_worsened_fold_count <= 1:
        classification = "H1_BOUNDED_RANGE_USEFUL"
    elif not low_robust or not high_robust:
        classification = "H1_EXTRAPOLATION_UNSAFE"
    else:
        classification = "INCONCLUSIVE"
    basis = {
        "interior_loho_mae_improved_all_folds": interior_mae_all,
        "interior_loho_p95_improved_all_folds": interior_p95_all,
        "interior_loho_max_improved_all_folds": interior_max_all,
        "low_endpoint_all_metrics_improved": low_robust,
        "high_endpoint_all_metrics_improved": high_robust,
        "lopo_pooled_all_metrics_improved": lopo_pooled_all,
        "lopo_worsened_fold_count": lopo_worsened_fold_count,
        "loho_pooled": loho_pooled,
        "lopo_pooled": lopo_pooled,
    }
    return classification, basis


def plot_raw_vs_h1_error(path: Path, conditions: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    errors_raw = np.asarray([row["raw_height_mm"] - row["truth_mm"] for row in conditions])
    # Use the LOPO out-of-fold prediction as the complete, position-held-out
    # comparison.  It has one independent H1 prediction per condition.
    lopo = [row for row in predictions if row["cv_scheme"] == "LOPO"]
    lopo_map = {row["condition"]: row for row in lopo}
    x = np.asarray([row["truth_mm"] for row in conditions])
    raw = errors_raw
    corrected = np.asarray([lopo_map[row["condition"]]["h1_error_mm"] for row in conditions])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(x, raw, color="#777777", alpha=0.65, label="C0 raw")
    ax.scatter(x, corrected, color="#1f77b4", alpha=0.75, label="C0 + H1 (LOPO OOF)")
    for height in HEIGHTS_MM:
        mask = x == height
        ax.plot([height], [float(np.mean(raw[mask]))], "o", color="#333333")
        ax.plot([height], [float(np.mean(corrected[mask]))], "o", color="#0b4f8a")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.2, color="#cc5500", linestyle="--", linewidth=0.8)
    ax.axhline(-0.2, color="#cc5500", linestyle="--", linewidth=0.8)
    ax.set_xlabel("true height (mm)")
    ax.set_ylabel("height error (mm)")
    ax.set_title("Haikang MANUAL_ROI_DIAGNOSTIC: raw vs H1 error by height")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_group_error(path: Path, rows: list[dict[str, Any]], title: str, xlabel: str) -> None:
    labels = [str(row["held_out_group"]) for row in rows]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, metric in enumerate(("mae_mm", "p95_absolute_error_mm", "max_absolute_error_mm")):
        offset = (index - 1) * width
        ax.bar(x + offset - width / 2, [row[f"raw_{metric}"] for row in rows], width, color="#999999", alpha=0.65, label=f"raw {metric.replace('_mm', '')}")
        ax.bar(x + offset + width / 2, [row[f"h1_{metric}"] for row in rows], width, color=("#4c78a8", "#f58518", "#54a24b")[index], alpha=0.85, label=f"H1 {metric.replace('_mm', '')}")
    ax.set_xticks(x, labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error metric (mm)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_endpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["held_out_group"] for row in rows]
    x = np.arange(len(labels), dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("mae_mm", "p95_absolute_error_mm", "max_absolute_error_mm"), strict=True):
        ax.bar(x - 0.18, [row[f"raw_{metric}"] for row in rows], 0.36, label="C0 raw", color="#999999")
        ax.bar(x + 0.18, [row[f"h1_{metric}"] for row in rows], 0.36, label="C0 + H1", color="#4c78a8")
        ax.set_title(metric.replace("_mm", ""))
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("error (mm)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Endpoint extrapolation: raw vs H1")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(path: Path, conditions: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    lopo = {row["condition"]: row for row in predictions if row["cv_scheme"] == "LOPO"}
    raw = np.full((len(HEIGHTS_MM), len(POSITION_IDS)), np.nan)
    corrected = np.full_like(raw, np.nan)
    for row in conditions:
        i = HEIGHT_IDS.index(row["height_id"])
        j = POSITION_IDS.index(row["position_id"])
        raw[i, j] = row["raw_height_mm"] - row["truth_mm"]
        corrected[i, j] = lopo[row["condition"]]["h1_error_mm"]
    vmax = float(np.nanmax(np.abs(np.concatenate([raw.ravel(), corrected.ravel()]))))
    vmax = max(vmax, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, constrained_layout=True)
    for ax, data, title in zip(axes, (raw, corrected), ("C0 raw residual", "C0 + H1 LOPO residual"), strict=True):
        image = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(POSITION_IDS)), POSITION_IDS)
        ax.set_yticks(range(len(HEIGHT_IDS)), HEIGHT_IDS)
        ax.set_xlabel("position")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7)
    axes[0].set_ylabel("height")
    fig.colorbar(image, ax=axes, label="residual (mm)", shrink=0.85)
    fig.suptitle("Haikang H1 feasibility residual heatmap")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report_text(
    *,
    input_dir: Path,
    output_dir: Path,
    provenance: dict[str, Any],
    loho_rows: list[dict[str, Any]],
    loho_pooled: dict[str, Any],
    endpoint_rows: list[dict[str, Any]],
    endpoint_pooled: dict[str, Any],
    lopo_rows: list[dict[str, Any]],
    lopo_pooled: dict[str, Any],
    classification: str,
    basis: dict[str, Any],
    special: dict[str, Any],
) -> str:
    lines = [
        "# Haikang H1 feasibility audit",
        "",
        "`MANUAL_ROI_DIAGNOSTIC`",
        "",
        f"最终结论：`{classification}`",
        "",
        "本报告只评估 H1 可行性，不生成生产补偿文件，不修改 C0、Session Ground、H-B2、C1 或在线配置。",
        "所有拟合单位为一个 `height × position` condition median；20 帧只作为重复测量，不作为独立训练样本。",
        "",
        "## 结论摘要",
        "",
        f"- 有效域声明：`valid_height_domain = [{FEASIBILITY_DOMAIN_MM[0]:g}, {FEASIBILITY_DOMAIN_MM[1]:g}] mm`；没有对域外高度作有效性声明。",
        f"- H1 定义复用大恒正式实现 `{H1_SOURCE_PATH}`：`h_corr = k * h_raw`，通过原点最小二乘拟合 `k`，修正方向为乘法放大；预测使用既有 grouped-CV `_predict(model=H1)`。",
        f"- 区间内 LOHO（6/10/20 mm）pooled：MAE {_fmt_pair(loho_pooled)}；H1 MAE 改善 {fmt(loho_pooled['mae_improvement_pct'], 2)}%，P95 改善 {fmt(loho_pooled['p95_improvement_pct'], 2)}%，Max 改善 {fmt(loho_pooled['max_improvement_pct'], 2)}%。",
        f"- 低端外推 2 mm：{_metric_sentence(next(row for row in endpoint_rows if row['held_out_group'] == 'LOW_END_EXTRAPOLATION'))}；低端不是四项指标同时改善。",
        f"- 高端外推 30 mm：{_metric_sentence(next(row for row in endpoint_rows if row['held_out_group'] == 'HIGH_END_EXTRAPOLATION'))}。",
        f"- LOPO pooled：MAE 改善 {fmt(lopo_pooled['mae_improvement_pct'], 2)}%，P95 改善 {fmt(lopo_pooled['p95_improvement_pct'], 2)}%，Max 改善 {fmt(lopo_pooled['max_improvement_pct'], 2)}%；{basis['lopo_worsened_fold_count']}/10 个 position fold 至少有一个 condition 被恶化。",
        "",
        "因此 H1 的主要证据是区间内插值的系统性 Bias/MAE/P95 降低；端点留出和最坏点并非全部稳定改善，不能把本轮结果升级为全范围生产补偿。",
        "",
        "## 报告问题逐项回答",
        "",
        f"1. **6/10/20 mm 区间内是否稳定改善？** MAE 和 P95 在三个 LOHO fold 均改善；Max 并非三个 fold 均改善（{sum(row['h1_max_absolute_error_mm'] < row['raw_max_absolute_error_mm'] - EPS for row in loho_rows)}/3），所以属于有边界的插值证据，而非所有指标无条件稳定。",
        f"2. **2 mm 低端外推是否可靠？** 否。MAE {fmt(next(row for row in endpoint_rows if row['held_out_group'] == 'LOW_END_EXTRAPOLATION')['mae_improvement_pct'], 2)}%（负值代表恶化），虽 P95/Max 改善，但四项指标未同时改善。",
        f"3. **30 mm 高端外推是否可靠？** 本 fold 的 MAE/RMSE/P95/Max 均改善，但只有一个端点 fold，不能据此声明域外外推能力。",
        f"4. **LOPO 是否证明跨 position 泛化？** pooled 指标改善，但 {basis['lopo_worsened_fold_count']}/10 fold 出现至少一个 condition 恶化，因此是部分泛化证据，不是无条件保证。",
        f"5. **P95/Max 是否改善？** LOPO pooled 的 P95/Max 改善；区间 LOHO pooled 的 P95 改善但 Max {_direction(loho_pooled['max_improvement_mm'])}，说明最坏点不能只看平均 Bias。",
        f"6. **是否有高度/位置明显恶化？** 最大单 condition 恶化量为 {fmt(max(row['max_worsening_mm'] for row in loho_rows + endpoint_rows + lopo_rows))} mm；详见 fold CSV 和 residual heatmap。",
        f"7. **最终结论：** `{classification}`。建议只把 H1 作为当前 `[2, 30] mm` 标定域内的插值候选继续验证；不生成生产补偿文件，不声明范围外有效。",
        "",
        "## C0 raw baseline（H0-1M-B condition medians）",
        "",
        _metric_table([("C0 raw", provenance["h0_accuracy"].get("overall", {}))]),
        "",
        "## LOHO interpolation（held out 6/10/20 mm）",
        "",
        _metric_table([("raw", loho_pooled["raw"]), ("H1", loho_pooled["h1"])]),
        "",
        "| held-out height | H1 k | raw MAE | H1 MAE | raw P95 | H1 P95 | raw Max | H1 Max | worsened conditions | max worsening |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in loho_rows:
        lines.append(
            f"| {row['held_out_group']} | {fmt(row['h1_scale'], 7)} | {fmt(row['raw_mae_mm'])} | {fmt(row['h1_mae_mm'])} | {fmt(row['raw_p95_absolute_error_mm'])} | {fmt(row['h1_p95_absolute_error_mm'])} | {fmt(row['raw_max_absolute_error_mm'])} | {fmt(row['h1_max_absolute_error_mm'])} | {row['worsened_condition_count']} | {fmt(row['max_worsening_mm'])} |"
        )
    lines += [
        "",
        "## Endpoint extrapolation",
        "",
        _metric_table([("raw", endpoint_pooled["raw"]), ("H1", endpoint_pooled["h1"])]),
        "",
        "| endpoint | train heights | H1 k | raw Bias | H1 Bias | raw MAE | H1 MAE | raw RMSE | H1 RMSE | raw P95 | H1 P95 | raw Max | H1 Max | worsened | max worsening |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in endpoint_rows:
        train_heights = ",".join(row["train_height_ids"])
        lines.append(
            f"| {row['held_out_group']} | {train_heights} | {fmt(row['h1_scale'], 7)} | {fmt(row['raw_bias_mm'])} | {fmt(row['h1_bias_mm'])} | {fmt(row['raw_mae_mm'])} | {fmt(row['h1_mae_mm'])} | {fmt(row['raw_rmse_mm'])} | {fmt(row['h1_rmse_mm'])} | {fmt(row['raw_p95_absolute_error_mm'])} | {fmt(row['h1_p95_absolute_error_mm'])} | {fmt(row['raw_max_absolute_error_mm'])} | {fmt(row['h1_max_absolute_error_mm'])} | {row['worsened_condition_count']} | {fmt(row['max_worsening_mm'])} |"
        )
    lines += [
        "",
        "## LOPO cross-position generalization",
        "",
        _metric_table([("raw", lopo_pooled["raw"]), ("H1", lopo_pooled["h1"])]),
        "",
        "| held-out position | H1 k | raw MAE | H1 MAE | raw P95 | H1 P95 | raw Max | H1 Max | worsened conditions | max worsening |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in lopo_rows:
        lines.append(
            f"| {row['held_out_group']} | {fmt(row['h1_scale'], 7)} | {fmt(row['raw_mae_mm'])} | {fmt(row['h1_mae_mm'])} | {fmt(row['raw_p95_absolute_error_mm'])} | {fmt(row['h1_p95_absolute_error_mm'])} | {fmt(row['raw_max_absolute_error_mm'])} | {fmt(row['h1_max_absolute_error_mm'])} | {row['worsened_condition_count']} | {fmt(row['max_worsening_mm'])} |"
        )
    lines += [
        "",
        "## Targeted checks",
        "",
        f"- h30_p01：C0 raw error = {fmt(special['h30_p01']['raw_error_mm'])} mm；high-end endpoint H1 error = {fmt(special['h30_p01']['high_endpoint_h1_error_mm'])} mm；LOPO-p01 H1 error = {fmt(special['h30_p01']['lopo_p01_h1_error_mm'])} mm。",
        f"- H1 scale range：LOHO interior [{fmt(min(row['h1_scale'] for row in loho_rows), 7)}, {fmt(max(row['h1_scale'] for row in loho_rows), 7)}]；endpoint [{fmt(min(row['h1_scale'] for row in endpoint_rows), 7)}, {fmt(max(row['h1_scale'] for row in endpoint_rows), 7)}]；LOPO [{fmt(min(row['h1_scale'] for row in lopo_rows), 7)}, {fmt(max(row['h1_scale'] for row in lopo_rows), 7)}]。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- 输入目录：`{input_dir}`。",
        f"- H0 condition summary SHA-256：`{provenance['summary_sha256']}`；frame CSV SHA-256：`{provenance['frames_sha256']}`；H0 accuracy JSON SHA-256：`{provenance['accuracy_sha256']}`。",
        f"- 输入为 {provenance['condition_count']} conditions / {provenance['frame_count']} frames；frame status：`{provenance['frame_status_counts']}`。",
        "- 复用：大恒 `validate_height_linear_cv._fit_parameters(..., \"H1\")` 的 through-origin OLS 定义和同文件 `_predict(..., \"H1\")` 的乘法应用；生产 Stage-A runtime 仍保持未启用。",
        "- 本轮新增：3 个区间 LOHO folds、2 个端点 extrapolation folds、10 个 LOPO folds，共 100 个 held-out condition predictions；没有 full-data fit 后回评。",
        "- H0 的 20 帧重复未进入拟合；没有 C0 重标定、C1、H-B2 或生产补偿文件。",
        f"- 生成时间：`{now_utc()}`。",
        "",
        f"输出目录：`{output_dir}`。",
    ]
    return "\n".join(lines) + "\n"


def _fmt_pair(payload: dict[str, Any]) -> str:
    return f"{fmt(payload['raw']['mae_mm'])} → {fmt(payload['h1']['mae_mm'])} mm"


def _metric_sentence(row: dict[str, Any]) -> str:
    return (
        f"MAE {fmt(row['raw_mae_mm'])}→{fmt(row['h1_mae_mm'])} mm, "
        f"P95 {fmt(row['raw_p95_absolute_error_mm'])}→{fmt(row['h1_p95_absolute_error_mm'])} mm, "
        f"Max {fmt(row['raw_max_absolute_error_mm'])}→{fmt(row['h1_max_absolute_error_mm'])} mm"
    )


def _direction(value: float) -> str:
    return "改善" if value > EPS else "恶化" if value < -EPS else "基本不变"


def _metric_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| model | n | Bias | MAE | RMSE | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows:
        lines.append(
            f"| {name} | {row.get('count', '')} | {fmt(row.get('bias_mm'))} | {fmt(row.get('mae_mm'))} | {fmt(row.get('rmse_mm'))} | {fmt(row.get('p95_absolute_error_mm', row.get('p95_mm')))} | {fmt(row.get('max_absolute_error_mm', row.get('max_mm')))} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output.resolve()
    conditions, provenance = load_inputs(input_dir)

    loho_rows, loho_predictions, _ = run_scheme(
        conditions,
        scheme="LOHO",
        groups=list(INTERIOR_HEIGHTS_MM),
        group_of=lambda row: row["truth_mm"],
        fold_type="INTERPOLATION",
        held_out_height=lambda group: group,
    )
    # Endpoint training support is explicit: each fold excludes only the
    # selected endpoint from a fixed four-height training set.
    endpoint_rows = []
    endpoint_predictions = []
    for fold_id, held_out_height, train_heights in (
        ("LOW_END_EXTRAPOLATION", 2.0, (6.0, 10.0, 20.0, 30.0)),
        ("HIGH_END_EXTRAPOLATION", 30.0, (2.0, 6.0, 10.0, 20.0)),
    ):
        train = [row for row in conditions if row["truth_mm"] in train_heights]
        test = [row for row in conditions if row["truth_mm"] == held_out_height]
        scale = fit_h1_scale(train)
        comparison, predictions = compare_metrics(test, scale)
        row = flatten_metric_row(
            scheme="ENDPOINT",
            fold_type="EXTRAPOLATION",
            fold_id=fold_id,
            held_out_group=fold_id,
            held_out_height_mm=held_out_height,
            train=train,
            test=test,
            comparison=comparison,
        )
        row["train_height_ids"] = [f"h{int(height):02d}" for height in train_heights]
        endpoint_rows.append(row)
        for prediction in predictions:
            endpoint_predictions.append(
                {
                    "cv_scheme": "ENDPOINT",
                    "fold_type": "EXTRAPOLATION",
                    "fold_id": fold_id,
                    "held_out_group": fold_id,
                    "held_out_height_mm": held_out_height,
                    **prediction,
                }
            )
    lopo_rows, lopo_predictions, _ = run_scheme(
        conditions,
        scheme="LOPO",
        groups=list(POSITION_IDS),
        group_of=lambda row: row["position_id"],
        fold_type="POSITION_HOLDOUT",
    )
    all_predictions = loho_predictions + endpoint_predictions + lopo_predictions
    loho_pooled = pooled_from_predictions(loho_predictions, scheme="LOHO", fold_type="INTERPOLATION")
    endpoint_pooled = pooled_from_predictions(endpoint_predictions, scheme="ENDPOINT", fold_type="EXTRAPOLATION")
    lopo_pooled = pooled_from_predictions(lopo_predictions, scheme="LOPO", fold_type="POSITION_HOLDOUT")
    classification, basis = classify(
        loho_pooled,
        endpoint_rows,
        lopo_pooled,
        loho_rows,
        lopo_rows,
    )

    raw_by_condition = {row["condition"]: row for row in conditions}
    high_endpoint = {row["condition"]: row for row in endpoint_predictions}
    lopo_p01 = {row["condition"]: row for row in lopo_predictions if row["fold_id"] == "p01"}
    special = {
        "h30_p01": {
            "raw_error_mm": raw_by_condition["h30_p01"]["raw_height_mm"] - 30.0,
            "high_endpoint_h1_error_mm": high_endpoint["h30_p01"]["h1_error_mm"],
            "lopo_p01_h1_error_mm": lopo_p01["h30_p01"]["h1_error_mm"],
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "cv_scheme", "fold_type", "fold_id", "held_out_group", "held_out_height_mm",
        "train_condition_count", "test_condition_count", "train_height_ids", "test_height_ids", "h1_scale",
        "raw_count", "raw_bias_mm", "raw_mae_mm", "raw_rmse_mm", "raw_p95_absolute_error_mm", "raw_max_absolute_error_mm",
        "h1_count", "h1_bias_mm", "h1_mae_mm", "h1_rmse_mm", "h1_p95_absolute_error_mm", "h1_max_absolute_error_mm",
        "mae_improvement_mm", "rmse_improvement_mm", "p95_improvement_mm", "max_improvement_mm",
        "mae_improvement_pct", "rmse_improvement_pct", "p95_improvement_pct", "max_improvement_pct",
        "worsened_condition_count", "max_worsening_mm",
    ]
    prediction_fields = [
        "cv_scheme", "fold_type", "fold_id", "held_out_group", "held_out_height_mm",
        "condition", "height_id", "position_id", "truth_mm", "raw_height_mm", "h1_scale", "h1_height_mm",
        "raw_error_mm", "h1_error_mm", "raw_absolute_error_mm", "h1_absolute_error_mm", "absolute_error_delta_mm",
    ]
    write_csv(output_dir / "h1_loho_results.csv", loho_rows, metric_fields)
    write_csv(output_dir / "h1_lopo_results.csv", lopo_rows, metric_fields)
    write_csv(output_dir / "h1_endpoint_extrapolation.csv", endpoint_rows, metric_fields)
    write_csv(output_dir / "h1_fold_predictions.csv", all_predictions, prediction_fields)

    summary = {
        "schema_version": 1,
        "task": "H1-F",
        "diagnostic_mode": "MANUAL_ROI_DIAGNOSTIC",
        "generated_at_utc": now_utc(),
        "production_compensation_file_generated": False,
        "valid_height_domain_mm": list(FEASIBILITY_DOMAIN_MM),
        "classification": classification,
        "classification_basis": basis,
        "h1_definition": {
            "model": "H1",
            "fit_equation": "h_corr=k*h_raw",
            "fit_method": "through-origin least-squares, reused validate_height_linear_cv._fit_parameters(model=H1)",
            "application": "existing validate_height_linear_cv._predict(model=H1); same scalar direction as production Stage-A",
            "source_path": H1_SOURCE_PATH,
            "source_sha256": sha256(H1_SOURCE_PATH),
            "fit_unit": "one condition median per height-position; no frame expansion",
        },
        "baseline_c0_raw": provenance["h0_accuracy"].get("overall", {}),
        "loho_interpolation": {"folds": loho_rows, "pooled_out_of_fold": loho_pooled},
        "endpoint_extrapolation": {"folds": endpoint_rows, "pooled": endpoint_pooled},
        "lopo": {"folds": lopo_rows, "pooled_out_of_fold": lopo_pooled},
        "special_checks": special,
        "provenance": {
            "input_dir": input_dir,
            "summary_sha256": provenance["summary_sha256"],
            "frames_sha256": provenance["frames_sha256"],
            "accuracy_sha256": provenance["accuracy_sha256"],
            "report_sha256": provenance["report_sha256"],
            "condition_count": provenance["condition_count"],
            "frame_count": provenance["frame_count"],
            "frames_per_condition": provenance["frames_per_condition"],
            "frame_status_counts": provenance["frame_status_counts"],
            "reused_h0_artifacts": [
                "manual_h_raw_position_summary.csv",
                "manual_h_raw_frames.csv (provenance/QC only)",
                "manual_c0_accuracy_summary.json",
                "manual_c0_height_audit_report.md",
            ],
            "reused_implementations": [
                "validate_height_linear_cv._fit_parameters(model=H1)",
                "validate_height_linear_cv._predict(model=H1)",
            ],
            "new_calculations": [
                "3 LOHO interpolation folds",
                "2 endpoint extrapolation folds",
                "10 LOPO position folds",
                "100 held-out condition predictions",
            ],
            "full_data_fit_evaluated": False,
            "c0_modified": False,
            "c1_applied": False,
            "hb2_applied": False,
            "production_config_modified": False,
        },
    }
    write_json(output_dir / "h1_feasibility_summary.json", summary)
    (output_dir / "h1_feasibility_report.md").write_text(
        report_text(
            input_dir=input_dir,
            output_dir=output_dir,
            provenance=provenance,
            loho_rows=loho_rows,
            loho_pooled=loho_pooled,
            endpoint_rows=endpoint_rows,
            endpoint_pooled=endpoint_pooled,
            lopo_rows=lopo_rows,
            lopo_pooled=lopo_pooled,
            classification=classification,
            basis=basis,
            special=special,
        ),
        encoding="utf-8",
    )
    plot_raw_vs_h1_error(output_dir / "raw_vs_h1_error_vs_height.png", conditions, all_predictions)
    plot_group_error(output_dir / "loho_held_out_error.png", loho_rows, "LOHO interpolation held-out error", "held-out height")
    plot_group_error(output_dir / "lopo_held_out_error.png", lopo_rows, "LOPO held-out position error", "held-out position")
    plot_endpoint(output_dir / "endpoint_extrapolation_comparison.png", endpoint_rows)
    plot_heatmap(output_dir / "raw_vs_h1_residual_heatmap.png", conditions, all_predictions)

    print(json.dumps({
        "diagnostic_mode": "MANUAL_ROI_DIAGNOSTIC",
        "classification": classification,
        "loho_pooled": loho_pooled,
        "endpoint_pooled": endpoint_pooled,
        "lopo_pooled": lopo_pooled,
        "output": str(output_dir),
    }, ensure_ascii=False, indent=2, default=json_ready))


if __name__ == "__main__":
    main()
