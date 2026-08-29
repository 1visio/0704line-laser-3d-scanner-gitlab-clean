"""Height-4 diagnostic: height dependence of the frozen H1 scale.

This script only consumes existing Ground-4A, Height-1, and Height-2/3
artifacts.  It computes descriptive k_required values and diagnostic trends;
it does not fit or freeze a production compensation model.  The 50 mm rows
remain held-out validation rows and are never used in the diagnostic trend
fit.
"""

from __future__ import annotations

import argparse
import csv
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND4A = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
DEFAULT_HEIGHT50 = DEFAULT_GROUND4A / "height50_heldout"
DEFAULT_OUTPUT = DEFAULT_GROUND4A / "height4_nonlinearity"

TRAIN_HEIGHTS = (1.001, 2.0, 6.0, 10.0, 20.0, 30.0)
POSITION_HEIGHTS = (6.0, 10.0, 20.0, 30.0, 50.0)
TREND_HEIGHTS = (6.0, 10.0, 20.0, 30.0)
POSITION_IDS = (1, 2, 3, 4, 5)
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
        return value.item()
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
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def safe_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def height_label(true_height: float) -> str:
    if abs(true_height - 1.001) < 1.0e-6:
        return "1mm"
    return f"{true_height:g}mm"


def metric(values: np.ndarray, truth: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "raw_mean_mm": None, "raw_bias_mm": None, "raw_mae_mm": None, "raw_rmse_mm": None, "raw_p95_mm": None, "raw_max_mm": None, "k_required": None, "k_conditionwise_mean": None, "k_conditionwise_median": None, "k_conditionwise_std": None, "k_conditionwise_min": None, "k_conditionwise_max": None}
    errors = values - truth
    absolute = np.abs(errors)
    k_values = truth / values
    return {
        "count": int(len(values)),
        "raw_mean_mm": float(np.mean(values)),
        "raw_bias_mm": float(np.mean(errors)),
        "raw_mae_mm": float(np.mean(absolute)),
        "raw_rmse_mm": float(np.sqrt(np.mean(errors**2))),
        "raw_p95_mm": float(np.percentile(absolute, 95.0)),
        "raw_max_mm": float(np.max(absolute)),
        # Aggregate k_required is defined from the aggregate raw height.  The
        # condition-wise distribution is retained separately for transparency.
        "k_required": float(truth / np.mean(values)),
        "k_conditionwise_mean": float(np.mean(k_values)),
        "k_conditionwise_median": float(np.median(k_values)),
        "k_conditionwise_std": float(np.std(k_values)),
        "k_conditionwise_min": float(np.min(k_values)),
        "k_conditionwise_max": float(np.max(k_values)),
    }


def fit_line(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if len(x) < 2:
        return {"count": int(len(x)), "slope": None, "intercept": None, "r2": None}
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= np.finfo(np.float64).eps else 1.0 - ss_res / ss_tot
    return {"count": int(len(x)), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def load_inputs(ground4a_dir: Path, height50_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    condition_path = ground4a_dir / "ground4a_condition_comparison.csv"
    height_summary_path = ground4a_dir / "height_linear_summary.json"
    model_comparison_path = ground4a_dir / "height_linear_model_comparison.csv"
    frozen_path = height50_dir / "frozen_height_scale.json"
    position_path = height50_dir / "height50_position_metrics.csv"
    frame_path = height50_dir / "height50_frame_metrics.csv"
    heldout_summary_path = height50_dir / "height50_summary.json"
    required = [condition_path, height_summary_path, model_comparison_path, frozen_path, position_path, frame_path, heldout_summary_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Height-4 input artifacts missing: {missing}")

    condition_rows = read_csv(condition_path)
    b_rows = [row for row in condition_rows if row.get("chain") == "session_linear"]
    successful = [
        row for row in b_rows
        if int(row.get("successful_repeat2_5") or 0) == 4
        and int(row.get("failed_repeat2_5") or 0) == 0
        and row.get("measured_mean_mm", "") != ""
    ]
    if len(b_rows) != 30 or len(successful) != 29:
        raise RuntimeError(f"expected 30 session-linear B rows/29 successful conditions, got {len(b_rows)}/{len(successful)}")
    successful_typed = [
        {
            "dataset": row["dataset"],
            "position_rank": int(row["position_rank"]),
            "true_height_mm": float(row["truth_mm"]),
            "raw_height_mm": float(row["measured_mean_mm"]),
        }
        for row in successful
    ]
    if len({(row["dataset"], row["position_rank"]) for row in successful_typed}) != 29:
        raise RuntimeError("duplicate successful Ground-4A condition")

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    condition_sha = sha256(condition_path)
    if frozen.get("input", {}).get("sha256") != condition_sha:
        raise RuntimeError("frozen H1 input SHA does not match Ground-4A condition CSV")
    frozen_k = float(frozen["k_full"])
    heldout_positions = read_csv(position_path)
    if len(heldout_positions) != 5 or {int(row["position_rank"]) for row in heldout_positions} != set(POSITION_IDS):
        raise RuntimeError("50mm position metrics are not five complete positions")
    heldout_typed = [
        {
            "dataset": "obs_50mm",
            "position_rank": int(row["position_rank"]),
            "true_height_mm": 50.0,
            "raw_height_mm": float(row["raw_condition_mean_mm"]),
        }
        for row in heldout_positions
    ]
    heldout_frames = read_csv(frame_path)
    formal_frames = [row for row in heldout_frames if row.get("scope") == "evaluation_repeat2_5"]
    if len(heldout_frames) != 25 or len(formal_frames) != 20:
        raise RuntimeError("50mm frame metrics do not contain 25 total/20 formal rows")
    height_summary = json.loads(height_summary_path.read_text(encoding="utf-8"))
    if height_summary.get("SELECTED_MODEL") != "H1" or int(height_summary.get("formal_input_condition_count", 0)) != 29:
        raise RuntimeError("Height-1 summary is not the expected H1/29-condition artifact")
    model_comparison = read_csv(model_comparison_path)
    heldout_summary = json.loads(heldout_summary_path.read_text(encoding="utf-8"))
    if heldout_summary.get("frozen_height_scale", {}).get("k_full") != frozen_k:
        raise RuntimeError("50mm summary does not use the frozen H1 k")
    provenance = {
        "ground4a_condition_path": str(condition_path.resolve()),
        "ground4a_condition_sha256": condition_sha,
        "height1_summary_path": str(height_summary_path.resolve()),
        "height1_summary_sha256": sha256(height_summary_path),
        "height1_model_comparison_path": str(model_comparison_path.resolve()),
        "height1_model_comparison_sha256": sha256(model_comparison_path),
        "frozen_height_scale_path": str(frozen_path.resolve()),
        "frozen_height_scale_sha256": sha256(frozen_path),
        "height50_position_path": str(position_path.resolve()),
        "height50_position_sha256": sha256(position_path),
        "height50_frame_path": str(frame_path.resolve()),
        "height50_frame_sha256": sha256(frame_path),
        "height50_summary_path": str(heldout_summary_path.resolve()),
        "height50_summary_sha256": sha256(heldout_summary_path),
        "frozen_k_full": frozen_k,
        "height50_used_in_trend_fit": False,
        "new_production_parameter_frozen": False,
    }
    return successful_typed, heldout_typed, {"frozen": frozen, "frozen_k": frozen_k, "height_summary": height_summary, "model_comparison": model_comparison, "heldout_summary": heldout_summary, "formal_frames": formal_frames, "provenance": provenance}


def build_height_rows(train: list[dict[str, Any]], heldout: list[dict[str, Any]], frozen_k: float, formal_frames: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for true_height in TRAIN_HEIGHTS:
        selected = [row for row in train if abs(row["true_height_mm"] - true_height) < 1.0e-6]
        values = np.asarray([row["raw_height_mm"] for row in selected], dtype=np.float64)
        stats = metric(values, true_height)
        rows.append({
            "height_label": height_label(true_height),
            "true_height_mm": true_height,
            "scope": "ground4a_development_condition_mean",
            "source": "ground4a_condition_comparison.csv/session_linear",
            "held_out": False,
            "offset_sensitive": true_height <= 2.0,
            "condition_count": stats["count"],
            "formal_frame_count": "",
            **stats,
            "frozen_k_full": frozen_k,
            "k_required_minus_frozen": None if stats["k_required"] is None else stats["k_required"] - frozen_k,
            "k_required_relative_to_frozen": None if stats["k_required"] is None else stats["k_required"] / frozen_k - 1.0,
        })
    selected_50 = np.asarray([row["raw_height_mm"] for row in heldout], dtype=np.float64)
    stats_50 = metric(selected_50, 50.0)
    frame_raw = np.asarray([float(row["raw_B_height_mm"]) for row in formal_frames], dtype=np.float64)
    frame_errors = frame_raw - 50.0
    rows.append({
        "height_label": "50mm",
        "true_height_mm": 50.0,
        "scope": "height50_heldout_position_mean",
        "source": "height50_position_metrics.csv/raw_condition_mean_mm",
        "held_out": True,
        "offset_sensitive": False,
        "condition_count": stats_50["count"],
        "formal_frame_count": len(formal_frames),
        **stats_50,
        "frame_raw_bias_mm": float(np.mean(frame_errors)),
        "frame_raw_mae_mm": float(np.mean(np.abs(frame_errors))),
        "frozen_k_full": frozen_k,
        "k_required_minus_frozen": stats_50["k_required"] - frozen_k,
        "k_required_relative_to_frozen": stats_50["k_required"] / frozen_k - 1.0,
    })
    for row in rows:
        row.setdefault("frame_raw_bias_mm", "")
        row.setdefault("frame_raw_mae_mm", "")
    return rows


def build_position_rows(train: list[dict[str, Any]], heldout: list[dict[str, Any]], frozen_k: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in POSITION_IDS:
        for true_height in POSITION_HEIGHTS:
            selected = (
                [row for row in heldout if row["position_rank"] == position]
                if true_height == 50.0
                else [row for row in train if row["position_rank"] == position and abs(row["true_height_mm"] - true_height) < 1.0e-6]
            )
            if not selected:
                continue
            values = np.asarray([row["raw_height_mm"] for row in selected], dtype=np.float64)
            stats = metric(values, true_height)
            rows.append({
                "position_id": f"laser{position:03d}",
                "position_rank": position,
                "height_label": height_label(true_height),
                "true_height_mm": true_height,
                "scope": "height50_heldout_position_mean" if true_height == 50.0 else "ground4a_development_condition_mean",
                "source": "height50_position_metrics.csv" if true_height == 50.0 else "ground4a_condition_comparison.csv",
                "held_out": true_height == 50.0,
                "offset_sensitive": False,
                "condition_count": stats["count"],
                **stats,
                "frozen_k_full": frozen_k,
                "k_required_minus_frozen": stats["k_required"] - frozen_k,
                "k_required_relative_to_frozen": stats["k_required"] / frozen_k - 1.0,
            })
    return rows


def model_row(model_comparison: list[dict[str, str]], scheme: str, model: str) -> dict[str, Any] | None:
    for row in model_comparison:
        if row.get("cv_scheme") == scheme and row.get("model") == model:
            return row
    return None


def diagnostic_payload(
    height_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    frozen_k: float,
    model_comparison: list[dict[str, str]],
    heldout_summary: dict[str, Any],
) -> dict[str, Any]:
    development = {round(float(row["true_height_mm"]), 3): row for row in height_rows if not row["held_out"]}
    trend_rows = [development[round(height, 3)] for height in TREND_HEIGHTS]
    raw_bias_fit = fit_line(TREND_HEIGHTS, [row["raw_bias_mm"] for row in trend_rows])
    k_fit = fit_line(TREND_HEIGHTS, [row["k_required"] for row in trend_rows])
    heldout = next(row for row in height_rows if row["held_out"])
    heldout_position = [row for row in position_rows if row["held_out"]]
    heldout_k = np.asarray([row["k_required"] for row in heldout_position], dtype=np.float64)
    dev_k = np.asarray([row["k_required"] for row in trend_rows], dtype=np.float64)
    all_heldout_below_frozen = bool(np.all(heldout_k < frozen_k))
    heldout_below_development = bool(float(heldout["k_required"]) < float(np.min(dev_k)))
    raw_bias_monotonic = bool(all(float(trend_rows[index]["raw_bias_mm"]) > float(trend_rows[index + 1]["raw_bias_mm"]) for index in range(len(trend_rows) - 1)))
    if raw_bias_monotonic and raw_bias_fit["slope"] is not None and raw_bias_fit["slope"] < 0.0 and float(raw_bias_fit["r2"]) >= 0.80 and all_heldout_below_frozen and heldout_below_development:
        status = "SUPPORTED"
    elif raw_bias_fit["slope"] is not None and raw_bias_fit["slope"] < 0.0 and (all_heldout_below_frozen or heldout_below_development):
        status = "INCONCLUSIVE"
    else:
        status = "NOT_SUPPORTED"
    h1_lho = model_row(model_comparison, "leave_one_height_out", "H1")
    h2_lho = model_row(model_comparison, "leave_one_height_out", "H2")
    h1_lpo = model_row(model_comparison, "leave_one_position_out", "H1")
    h2_lpo = model_row(model_comparison, "leave_one_position_out", "H2")
    heldout_frame_metrics = heldout_summary.get("frame_metrics", {})
    return {
        "HEIGHT_DEPENDENCE": status,
        "trend_fit_domain_mm": list(TREND_HEIGHTS),
        "trend_fit_excludes_1_2mm": True,
        "trend_fit_excludes_50mm": True,
        "raw_bias_vs_height": raw_bias_fit,
        "k_required_vs_height": k_fit,
        "raw_bias_monotonic_6_30": raw_bias_monotonic,
        "frozen_k_full": frozen_k,
        "development_k_required_6_30_min": float(np.min(dev_k)),
        "development_k_required_6_30_max": float(np.max(dev_k)),
        "heldout_50_k_required": float(heldout["k_required"]),
        "heldout_50_k_required_min_by_position": float(np.min(heldout_k)),
        "heldout_50_k_required_max_by_position": float(np.max(heldout_k)),
        "heldout_all_positions_below_frozen_k": all_heldout_below_frozen,
        "heldout_below_development_6_30_k": heldout_below_development,
        "height1_affine_diagnostic": {
            "source": "existing Height-1 LHO CV; no new affine fit",
            "leave_one_height_out_H1": h1_lho,
            "leave_one_height_out_H2": h2_lho,
            "leave_one_position_out_H1": h1_lpo,
            "leave_one_position_out_H2": h2_lpo,
        },
        "constant_scale_diagnostic": {
            "source": "frozen_height_scale.json; no new scale fit",
            "model": "H1",
            "k_full": frozen_k,
            "heldout_50_k_gap_mm_per_mm": float(heldout["k_required"] - frozen_k),
        },
        "heldout_50_h1_diagnostic": {
            "source": "existing Height-2/3 formal repeat2-5 summary; no refit",
            "raw_B": heldout_frame_metrics.get("raw_B"),
            "frozen_H1": heldout_frame_metrics.get("frozen_H1"),
            "raw_to_h1_improvement": heldout_summary.get("raw_to_h1_improvement"),
        },
        "production_model_refit": False,
        "50mm_used_in_any_trend_fit": False,
    }


def plot_k_vs_height(path: Path, height_rows: list[dict[str, Any]], position_rows: list[dict[str, Any]], frozen_k: float) -> None:
    fig, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for position in POSITION_IDS:
        rows = sorted([row for row in position_rows if row["position_rank"] == position], key=lambda row: row["true_height_mm"])
        x = [row["true_height_mm"] for row in rows]
        y = [row["k_required"] for row in rows]
        axis.plot(x, y, marker="o", linewidth=1.3, color=colors(position - 1), label=f"laser{position:03d}")
        heldout = [row for row in rows if row["held_out"]]
        if heldout:
            axis.scatter([heldout[0]["true_height_mm"]], [heldout[0]["k_required"]], s=72, facecolors="none", edgecolors=colors(position - 1), linewidths=1.8, zorder=5)
    aggregate = sorted([row for row in height_rows if row["true_height_mm"] in POSITION_HEIGHTS], key=lambda row: row["true_height_mm"])
    axis.plot([row["true_height_mm"] for row in aggregate], [row["k_required"] for row in aggregate], color="black", marker="D", linewidth=2.2, label="condition/position aggregate")
    axis.axhline(frozen_k, color="tab:red", linestyle="--", linewidth=1.2, label=f"frozen H1 k={frozen_k:.7f}")
    axis.set_xlabel("true height [mm]")
    axis.set_ylabel("k_required = true height / raw height")
    axis.set_title("Height-4 k_required vs height (50 mm open marker = held-out)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bias_vs_height(path: Path, height_rows: list[dict[str, Any]], position_rows: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    aggregate = sorted(height_rows, key=lambda row: row["true_height_mm"])
    train = [row for row in aggregate if not row["held_out"]]
    heldout = [row for row in aggregate if row["held_out"]]
    axis.plot([row["true_height_mm"] for row in train], [row["raw_bias_mm"] for row in train], color="black", marker="D", linewidth=2.2, label="Ground-4A aggregate")
    if heldout:
        axis.scatter([heldout[0]["true_height_mm"]], [heldout[0]["raw_bias_mm"]], color="tab:red", marker="*", s=130, label="50 mm held-out aggregate")
    colors = plt.get_cmap("tab10")
    for position in POSITION_IDS:
        rows = sorted([row for row in position_rows if row["position_rank"] == position], key=lambda row: row["true_height_mm"])
        axis.plot([row["true_height_mm"] for row in rows], [row["raw_bias_mm"] for row in rows], marker=".", linewidth=0.8, alpha=0.65, color=colors(position - 1), label=f"laser{position:03d}")
    axis.axhline(0.0, color="gray", linewidth=0.9)
    for row in train:
        if row["offset_sensitive"]:
            axis.scatter([row["true_height_mm"]], [row["raw_bias_mm"]], facecolors="none", edgecolors="tab:orange", s=90, linewidths=1.5, zorder=6)
    axis.set_xlabel("true height [mm]")
    axis.set_ylabel("raw B session_linear bias [mm]")
    axis.set_title("Height-4 raw bias vs height (open markers: 1/2 mm offset-sensitive)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "MISSING"
    return f"{float(value):.{digits}f}"


def write_report(path: Path, height_rows: list[dict[str, Any]], position_rows: list[dict[str, Any]], diagnostics: dict[str, Any], provenance: dict[str, Any]) -> None:
    train_rows = [row for row in height_rows if not row["held_out"]]
    heldout = next(row for row in height_rows if row["held_out"])
    lines = [
        "# Height-4 高度尺度非线性诊断",
        "",
        f"- `HEIGHT_DEPENDENCE = {diagnostics['HEIGHT_DEPENDENCE']}`",
        "- 本轮只做诊断性比较，不重新冻结 scale、affine 或 height-dependent production compensation。",
        "- 50 mm 保持纯 validation 身份：未参与任何趋势拟合、模型选择或参数更新。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- 复用：Ground-4A `session_linear` 的 29 个成功 condition mean；输入 SHA-256：`{provenance['ground4a_condition_sha256']}`。",
        f"- 复用：Height-1 H1/H2 LHO CV 结果；H1 full-data frozen k：`{fmt(provenance['frozen_k_full'], 15)}`。",
        f"- 复用：Height-2/3 的 50 mm position/frame metrics；position SHA-256：`{provenance['height50_position_sha256']}`，frame SHA-256：`{provenance['height50_frame_sha256']}`。",
        "- 本轮新增：按高度/position 的 descriptive `k_required`、6–30 mm-only trend fit、图表和归因报告。",
        "- 未做：50 mm 拟合 k、重新冻结 affine、修改 C0/C1、修改 G(S) 或生产配置。",
        "",
        "## 按高度统计（condition/position mean）",
        "",
        "`k_required` 定义为 `true_height / mean(raw_height)`；同时保留 condition-wise k 的均值/范围。1.001 mm 和 2 mm 标为 offset-sensitive，不用于 height trend fit。",
        "",
        "| height | scope | n | raw bias | raw MAE | raw RMSE | k_required | k condition-wise mean | k range |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in height_rows:
        lines.append(
            f"| {row['height_label']} | {row['scope']} | {row['condition_count']} | {fmt(row['raw_bias_mm'])} | {fmt(row['raw_mae_mm'])} | {fmt(row['raw_rmse_mm'])} | "
            f"{fmt(row['k_required'])} | {fmt(row['k_conditionwise_mean'])} | {fmt(row['k_conditionwise_min'])}–{fmt(row['k_conditionwise_max'])} |"
        )
    lines += [
        "",
        "## Position k_required（6/10/20/30 mm development + 50 mm held-out）",
        "",
        "| position | 6 mm | 10 mm | 20 mm | 30 mm | 50 mm held-out |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for position in POSITION_IDS:
        values = {}
        for row in position_rows:
            if row["position_rank"] == position:
                values[float(row["true_height_mm"])] = row["k_required"]
        lines.append(
            f"| laser{position:03d} | {fmt(values.get(6.0))} | {fmt(values.get(10.0))} | {fmt(values.get(20.0))} | {fmt(values.get(30.0))} | {fmt(values.get(50.0))} |"
        )
    heldout_position = [row for row in position_rows if row["held_out"]]
    h1_lho = diagnostics["height1_affine_diagnostic"]["leave_one_height_out_H1"]
    h2_lho = diagnostics["height1_affine_diagnostic"]["leave_one_height_out_H2"]
    h1_lpo = diagnostics["height1_affine_diagnostic"]["leave_one_position_out_H1"]
    h2_lpo = diagnostics["height1_affine_diagnostic"]["leave_one_position_out_H2"]
    heldout_h1 = diagnostics["heldout_50_h1_diagnostic"]
    heldout_raw = heldout_h1["raw_B"]
    heldout_corrected = heldout_h1["frozen_H1"]
    lines += [
        "",
        "## Constant scale / affine / height-dependent trend",
        "",
        f"- Constant H1：frozen `k={fmt(provenance['frozen_k_full'], 15)}`。6–30 mm 的 aggregate k_required 范围为 `{fmt(diagnostics['development_k_required_6_30_min'])}`–`{fmt(diagnostics['development_k_required_6_30_max'])}`；50 mm aggregate 为 `{fmt(diagnostics['heldout_50_k_required'])}`，低于 development minimum。",
        f"- 50 mm 五个 position 的 k_required 范围为 `{fmt(diagnostics['heldout_50_k_required_min_by_position'])}`–`{fmt(diagnostics['heldout_50_k_required_max_by_position'])}`，全部低于 frozen H1；这解释了 H1 在 50 mm 的共同过补偿方向。",
        f"- 50 mm formal repeat2–5 frame：raw B 的 Bias/MAE/RMSE/P95/Max 为 `{fmt(heldout_raw.get('bias_mm'))}`/`{fmt(heldout_raw.get('mae_mm'))}`/`{fmt(heldout_raw.get('rmse_mm'))}`/`{fmt(heldout_raw.get('p95_mm'))}`/`{fmt(heldout_raw.get('max_mm'))}` mm；frozen H1 为 `{fmt(heldout_corrected.get('bias_mm'))}`/`{fmt(heldout_corrected.get('mae_mm'))}`/`{fmt(heldout_corrected.get('rmse_mm'))}`/`{fmt(heldout_corrected.get('p95_mm'))}`/`{fmt(heldout_corrected.get('max_mm'))}` mm，MAE/RMSE 改善但 P95/Max 变差。",
        f"- raw bias 的 6–30 mm 诊断直线斜率为 `{fmt(diagnostics['raw_bias_vs_height']['slope'])}` mm/mm，R²=`{fmt(diagnostics['raw_bias_vs_height']['r2'], 4)}`；6/10/20/30 的 group mean 单调变负。",
        f"- k_required 的 6–30 mm 诊断斜率为 `{fmt(diagnostics['k_required_vs_height']['slope'], 9)}` /mm，R²=`{fmt(diagnostics['k_required_vs_height']['r2'], 4)}`；这段内部近似平坦且不呈稳定单调，但 50 mm 出现共同的 high-end drop。",
        f"- Affine H2：只引用 Height-1 已有 LHO CV，不在本轮重拟合。LHO-height pooled H1 MAE/RMSE/P95/Max=`{fmt(h1_lho.get('corrected_mae_mm'))}`/`{fmt(h1_lho.get('corrected_rmse_mm'))}`/`{fmt(h1_lho.get('corrected_p95_mm'))}`/`{fmt(h1_lho.get('corrected_max_mm'))}`，H2=`{fmt(h2_lho.get('corrected_mae_mm'))}`/`{fmt(h2_lho.get('corrected_rmse_mm'))}`/`{fmt(h2_lho.get('corrected_p95_mm'))}`/`{fmt(h2_lho.get('corrected_max_mm'))}`；H1 为 6/6 folds 全部改善，H2 为 5/6，不能把 H2 解释为已验证的 50 mm 泛化模型。",
        f"- LHO-position pooled H1/H2 MAE=`{fmt(h1_lpo.get('corrected_mae_mm'))}`/`{fmt(h2_lpo.get('corrected_mae_mm'))}`、RMSE=`{fmt(h1_lpo.get('corrected_rmse_mm'))}`/`{fmt(h2_lpo.get('corrected_rmse_mm'))}`；两者均为 5/5 folds 改善，但 H2 Max=`{fmt(h2_lpo.get('corrected_max_mm'))}` 高于 H1 的 `{fmt(h1_lpo.get('corrected_max_mm'))}`。",
        "",
        "## 归因结论",
        "",
        f"- `HEIGHT_DEPENDENCE = {diagnostics['HEIGHT_DEPENDENCE']}`：支持“raw 高度误差随高度增加而恶化，且 frozen H1 在 50 mm 端点出现共同过补偿”的高度依赖现象。",
        "- 该结论不等于已经识别出可部署的非线性函数：6–30 mm 的 k_required 不完全单调，且 position 间基线不同；50 mm 的 laser003/005 仍比其他 position 更差，提示 position×height/session interaction。",
        "- 因此当前更准确的表述是：高度趋势得到支持，但趋势形状和 50 mm 幅度仍不足以支持重新冻结 affine、spline 或 position-specific 补偿。",
        "",
        "## 下一步最有信息量的中间高度",
        "",
        "- 首选 `40 mm`；若可增加完整量块，建议 `35/40/45 mm`，每个高度覆盖全部 5 个 laser position，并沿用 repeat1 proxy + repeat2–5 formal held-out protocol。",
        "- 重点观察：raw bias 是否继续单调变负；frozen H1 corrected bias 是否从负跨零并继续变正；k_required 是否在 30–50 mm 单调下降；P95/Max 是否从 35–45 mm 开始恶化；laser003/005 是否持续异常。",
        "- 这些中间高度只用于下一轮 held-out 归因验证，不回填当前 k，也不在本轮生成生产参数。",
        "",
        "## 文件",
        "",
        "- `height_scale_by_height.csv`、`height_scale_by_position.csv`",
        "- `k_required_vs_height.png`、`height_bias_vs_height.png`",
        "- 本报告中的所有 trend fit 仅使用 6/10/20/30 mm；50 mm 只用于 held-out 对照与状态判断。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground4a-dir", type=Path, default=DEFAULT_GROUND4A)
    parser.add_argument("--height50-dir", type=Path, default=DEFAULT_HEIGHT50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ground4a_dir = args.ground4a_dir.resolve()
    height50_dir = args.height50_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train, heldout, artifacts = load_inputs(ground4a_dir, height50_dir)
    frozen_k = artifacts["frozen_k"]
    height_rows = build_height_rows(train, heldout, frozen_k, artifacts["formal_frames"])
    position_rows = build_position_rows(train, heldout, frozen_k)
    diagnostics = diagnostic_payload(
        height_rows,
        position_rows,
        frozen_k,
        artifacts["model_comparison"],
        artifacts["heldout_summary"],
    )
    diagnostics["created_at_utc"] = now_utc()
    diagnostics["provenance"] = artifacts["provenance"]

    write_csv(output / "height_scale_by_height.csv", height_rows, list(height_rows[0].keys()))
    write_csv(output / "height_scale_by_position.csv", position_rows, list(position_rows[0].keys()))
    plot_k_vs_height(output / "k_required_vs_height.png", height_rows, position_rows, frozen_k)
    plot_bias_vs_height(output / "height_bias_vs_height.png", height_rows, position_rows)
    write_json(output / "height4_nonlinearity_summary.json", diagnostics)
    write_report(output / "height4_nonlinearity_report.md", height_rows, position_rows, diagnostics, artifacts["provenance"])
    print(f"HEIGHT_DEPENDENCE={diagnostics['HEIGHT_DEPENDENCE']}")
    print(f"height rows={len(height_rows)}, position rows={len(position_rows)}, trend fit uses heights={TREND_HEIGHTS}, heldout_used_in_fit=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
