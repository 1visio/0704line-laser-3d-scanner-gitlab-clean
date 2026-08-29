"""Generate visual comparisons for the 0819 C0-only gauge-block A/B run.

The script consumes the already computed CSV artifacts.  It does not reopen
TIFFs, rerun Steger extraction, or perform any reconstruction.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = TOOL_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_model_ab_c0_v5"
DEFAULT_OUTPUT = DEFAULT_INPUT / "figures_20260828"

DATASETS = ("obs_1mm", "obs_2mm", "obs_6mm", "obs_10mm", "obs_20mm", "obs_30mm")
HEIGHT_LABELS = {
    "obs_1mm": "1 mm",
    "obs_2mm": "2 mm",
    "obs_6mm": "6 mm",
    "obs_10mm": "10 mm",
    "obs_20mm": "20 mm",
    "obs_30mm": "30 mm",
}
MODE_ORDER = ("local_adjacent", "all_non_height", "fixed_zg_zero")
MODE_LABELS = {
    "local_adjacent": "local_adjacent",
    "all_non_height": "all_non_height",
    "fixed_zg_zero": "fixed_zg_zero",
}
MODEL_ORDER = ("CONE", "QUADRATIC")
MODEL_LABELS = {"CONE": "Circular cone", "QUADRATIC": "Quadratic surface"}
MODEL_COLORS = {"CONE": "#1f77b4", "QUADRATIC": "#ff7f0e"}
METRIC_ORDER = ("mae_mm", "rmse_mm", "p95_mm", "max_mm")
METRIC_LABELS = {
    "mae_mm": "MAE",
    "rmse_mm": "RMSE",
    "p95_mm": "P95",
    "max_mm": "Max",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [row for row in csv.DictReader(stream) if any(row.values())]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"missing numeric value {key!r} in row {row}")
    return float(value)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def filter_rows(
    rows: list[dict[str, str]],
    *,
    variant: str,
    mode: str,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("variant") == variant and row.get("mode") == mode
    ]
    return selected


def assert_source_consistency(input_dir: Path) -> dict[str, Any]:
    required = (
        "stats_summary.csv",
        "condition_measurements.csv",
        "paired_condition_comparison.csv",
        "provenance.json",
        "model_comparison_report.md",
    )
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing A/B source artifacts: {missing}")

    stats = read_csv_rows(input_dir / "stats_summary.csv")
    conditions = read_csv_rows(input_dir / "condition_measurements.csv")
    paired = read_csv_rows(input_dir / "paired_condition_comparison.csv")
    global_rows = [
        row
        for row in stats
        if row.get("layer") == "single_frame"
        and row.get("scope") == "ALL"
        and row.get("variant") == "native_valid"
    ]
    expected_global = len(MODE_ORDER) * len(MODEL_ORDER)
    if len(global_rows) != expected_global:
        raise ValueError(f"expected {expected_global} global rows, got {len(global_rows)}")
    if len(conditions) != 360:
        raise ValueError(f"expected 360 condition rows, got {len(conditions)}")
    if len(paired) != 180:
        raise ValueError(f"expected 180 paired rows, got {len(paired)}")

    frame_metrics = input_dir / "frame_model_metrics.csv"
    if not frame_metrics.is_file():
        raise FileNotFoundError(f"missing frame metrics: {frame_metrics}")
    frame_rows = read_csv_rows(frame_metrics)
    if len(frame_rows) != 150:
        raise ValueError(f"expected 150 frame rows, got {len(frame_rows)}")
    if not all(
        row.get("cone_valid_points")
        == row.get("quadratic_valid_points")
        == row.get("common_valid_points")
        for row in frame_rows
    ):
        raise ValueError("native/common valid point sets are not identical")

    return {
        "input_dir": str(input_dir.resolve()),
        "frame_count": len(frame_rows),
        "condition_count": len(conditions),
        "paired_count": len(paired),
        "native_equals_common": True,
        "source_files": {
            name: {"path": str((input_dir / name).resolve()), "sha256": sha256(input_dir / name)}
            for name in required
        },
    }


def plot_global_metrics(input_dir: Path, output_path: Path) -> None:
    rows = read_csv_rows(input_dir / "stats_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    x = np.arange(len(METRIC_ORDER), dtype=float)
    width = 0.36
    ymax = 0.0

    for axis, mode in zip(axes, MODE_ORDER):
        mode_rows = {
            row["model"]: row
            for row in rows
            if row["layer"] == "single_frame"
            and row["scope"] == "ALL"
            and row["variant"] == "native_valid"
            and row["mode"] == mode
        }
        if set(mode_rows) != set(MODEL_ORDER):
            raise ValueError(f"missing global metric model rows for {mode}")
        for index, model in enumerate(MODEL_ORDER):
            values = [float_value(mode_rows[model], metric) for metric in METRIC_ORDER]
            ymax = max(ymax, *values)
            axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
                alpha=0.9,
            )
        axis.axhline(0.2, color="#2ca02c", linestyle="--", linewidth=1.2, label="±0.2 mm limit")
        axis.set_title(MODE_LABELS[mode])
        axis.set_xticks(x, [METRIC_LABELS[metric] for metric in METRIC_ORDER])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("absolute error metric (mm)")
    axes[0].set_ylim(0, max(0.35, ymax * 1.15))
    fig.suptitle("0819 gauge blocks: C0-only global model comparison", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "150 frames | native_valid (= common_valid for every frame) | manual-frozen geometry-only ROI",
        ha="center",
        fontsize=9,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    save_figure(fig, output_path)


def plot_local_bias_by_position(input_dir: Path, output_path: Path) -> None:
    rows = filter_rows(
        read_csv_rows(input_dir / "condition_measurements.csv"),
        variant="native_valid",
        mode="local_adjacent",
    )
    lookup = {(row["dataset"], int(row["position_rank"]), row["model"]): row for row in rows}
    if len(rows) != len(DATASETS) * 5 * len(MODEL_ORDER):
        raise ValueError(f"expected 60 local condition rows, got {len(rows)}")

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True, sharey=True)
    positions = np.arange(1, 6)
    for axis, dataset in zip(axes.flat, DATASETS):
        axis.axhspan(-0.2, 0.2, color="#d9f0d3", alpha=0.7, zorder=0)
        axis.axhline(0.0, color="#333333", linewidth=0.9)
        axis.axhline(0.2, color="#2ca02c", linestyle="--", linewidth=0.8)
        axis.axhline(-0.2, color="#2ca02c", linestyle="--", linewidth=0.8)
        for model in MODEL_ORDER:
            values = [
                float_value(lookup[(dataset, position, model)], "signed_error_mm")
                for position in positions
            ]
            sigmas = [
                float_value(lookup[(dataset, position, model)], "repeatability_sigma_mm")
                for position in positions
            ]
            axis.errorbar(
                positions,
                values,
                yerr=sigmas,
                marker="o",
                markersize=4.5,
                capsize=2.5,
                linewidth=1.5,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.set_title(HEIGHT_LABELS[dataset])
        axis.set_xticks(positions)
        axis.grid(alpha=0.22)
        axis.set_axisbelow(True)

    axes[0, 0].set_ylabel("signed error (mm)")
    axes[1, 0].set_ylabel("signed error (mm)")
    for axis in axes[1, :]:
        axis.set_xlabel("position rank (v ascending)")
    axes[0, 0].legend(loc="lower left", fontsize=9, frameon=True)
    for axis in axes.flat:
        axis.set_ylim(-0.21, 0.21)
    fig.suptitle("0819 gauge blocks: local-adjacent C0 bias by position", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Markers are condition means across 5 repeats; whiskers show repeatability σ | green band: ±0.2 mm",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, output_path)


def matrix_from_rows(
    rows: list[dict[str, str]],
    value_key: str,
    *,
    model: str | None = None,
) -> np.ndarray:
    matrix = np.full((len(DATASETS), 5), np.nan, dtype=float)
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        dataset_index = DATASETS.index(row["dataset"])
        position_index = int(row["position_rank"]) - 1
        matrix[dataset_index, position_index] = float_value(row, value_key)
    if np.isnan(matrix).any():
        raise ValueError(f"incomplete matrix for {value_key}, model={model}")
    return matrix


def annotate_heatmap(axis: plt.Axes, matrix: np.ndarray, limit: float) -> None:
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if abs(value) < limit * 0.72 else "white",
            )


def plot_local_heatmap(input_dir: Path, output_path: Path) -> None:
    condition_rows = filter_rows(
        read_csv_rows(input_dir / "condition_measurements.csv"),
        variant="native_valid",
        mode="local_adjacent",
    )
    paired_rows = filter_rows(
        read_csv_rows(input_dir / "paired_condition_comparison.csv"),
        variant="native_valid",
        mode="local_adjacent",
    )
    cone = matrix_from_rows(condition_rows, "signed_error_mm", model="CONE")
    quadratic = matrix_from_rows(condition_rows, "signed_error_mm", model="QUADRATIC")
    delta_abs = matrix_from_rows(
        paired_rows,
        "quadratic_abs_error_minus_cone_mm",
    )
    error_norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.2)
    delta_limit = 0.2
    delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.4), gridspec_kw={"width_ratios": [1, 1, 1.08]})
    plots = (
        (axes[0], cone, "Cone signed error", error_norm, "RdBu_r"),
        (axes[1], quadratic, "Quadratic signed error", error_norm, "RdBu_r"),
        (axes[2], delta_abs, "Δ abs error (Quadratic − Cone)", delta_norm, "RdBu_r"),
    )
    for axis, matrix, title, norm, cmap in plots:
        image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axis.set_title(title)
        axis.set_xticks(range(5), [str(index) for index in range(1, 6)])
        axis.set_yticks(range(len(DATASETS)), [HEIGHT_LABELS[item] for item in DATASETS])
        axis.set_xlabel("position rank")
        axis.grid(False)
        annotate_heatmap(axis, matrix, delta_limit if axis is axes[2] else 0.2)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("mm")
    axes[0].set_ylabel("gauge-block height")
    fig.suptitle("0819 gauge blocks: local-adjacent C0 error maps", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Rows are true heights; columns are v-ordered positions | positive Δ means Quadratic has larger absolute error",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, output_path)


def write_manifest(
    output_dir: Path,
    input_dir: Path,
    source_summary: dict[str, Any],
    report_copy: Path,
) -> None:
    figure_names = (
        "model_ab_global_metrics.png",
        "model_ab_local_bias_by_position.png",
        "model_ab_local_heatmap.png",
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "0819 150 gauge blocks, Circular Cone versus Quadratic C0-only visualization",
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "reused_protocol": [
            "same 0819 processed CSV artifacts",
            "same manual-frozen geometry-only ROI",
            "same Steger-center-based C0-only measurements",
        ],
        "not_recomputed": ["TIFF loading", "Steger extraction", "model reconstruction", "C1/H1/H-B2"],
        "source_summary": source_summary,
        "report_copy": {"path": str(report_copy.resolve()), "sha256": sha256(report_copy)},
        "figures": {
            name: {"path": str((output_dir / name).resolve()), "sha256": sha256(output_dir / name)}
            for name in figure_names
        },
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    source_summary = assert_source_consistency(input_dir)
    plot_global_metrics(input_dir, output_dir / "model_ab_global_metrics.png")
    plot_local_bias_by_position(input_dir, output_dir / "model_ab_local_bias_by_position.png")
    plot_local_heatmap(input_dir, output_dir / "model_ab_local_heatmap.png")
    report_copy = output_dir / "model_comparison_report.md"
    shutil.copy2(input_dir / "model_comparison_report.md", report_copy)
    write_manifest(output_dir, input_dir, source_summary, report_copy)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "figures": [
                    "model_ab_global_metrics.png",
                    "model_ab_local_bias_by_position.png",
                    "model_ab_local_heatmap.png",
                ],
                "report_copy": str(report_copy),
                "source_summary": source_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
