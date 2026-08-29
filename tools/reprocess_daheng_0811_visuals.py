#!/usr/bin/env python3
"""Rebuild reference-style visual audits for the Daheng 0811 20 mm run.

This is intentionally a visual-only reprocessing script.  It reuses the
historical ``overlay.png`` and exported point/result files from the eight
positions that produced the original 20 mm table.  It does not read source
images, rerun Steger, rerun reconstruction, or claim that the inferred ROI
boundaries are the original GUI selections.

The ROI bands follow the geometry-only protocol used by the later Session01
ROI-freeze work:

* height: ``candidate_v +/- 45 px``;
* baseline before: ``[height_start - 220, height_start - 20]``;
* baseline after: ``[height_end + 20, height_end + 220]``.

For 0811, ``candidate_v`` is inferred from the median ``v`` of the exported
height points, and the actual historical ROI rectangles were not persisted.
The generated figures therefore distinguish inferred bands from actual CSV
point envelopes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FRAME_IDS = (
    "000974",
    "004021",
    "005772",
    "007020",
    "008310",
    "009614",
    "011317",
    "012686",
)

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 3000
OFFSET_X = 1760
ROI_HALF_WIDTH = 45
BASELINE_GAP = 20
BASELINE_HALF_WIDTH = 220
NOMINAL_HEIGHT_MM = 20.0

ROI_COLORS = {
    "baseline_before": "#f6bd60",
    "height": "#90be6d",
    "baseline_after": "#f6bd60",
}


@dataclass(frozen=True)
class PositionAudit:
    position: int
    frame_id: str
    frame_dir: Path
    overlay_path: Path
    height_points_path: Path
    baseline_points_path: Path
    result_path: Path
    height_points: np.ndarray
    baseline_points: np.ndarray
    laser_center: np.ndarray
    result: dict[str, Any]
    vblock_px: float
    vref_px: float
    delta_v_px: float
    height_mean_mm: float
    height_std_mm: float
    table_height_mm: float
    table_error_mm: float
    table_relative_error_pct: float
    height_v_min: float
    height_v_max: float
    baseline_v_min: float
    baseline_v_max: float
    height_roi: tuple[int, int]
    baseline_before: tuple[int, int]
    baseline_after: tuple[int, int]
    baseline_side: str
    baseline_before_support: int
    baseline_after_support: int
    edge_clipped: bool


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement-root",
        type=Path,
        default=repo_root / "laser_measurement_tool" / "output_daheng_0811",
        help="Historical 0811 measurement output root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs" / "daheng_0811_visual_reprocess",
        help="New output directory; it must not already exist.",
    )
    parser.add_argument(
        "--nominal-height-mm",
        type=float,
        default=NOMINAL_HEIGHT_MM,
        help="Nominal height used for the table error plot.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_points(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    points = np.asarray(
        [(float(row["u"]), float(row["v"])) for row in rows],
        dtype=np.float64,
    )
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError(f"expected non-empty u,v data: {path}")
    return points


def read_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_overlay(path: Path) -> np.ndarray:
    image = plt.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB/RGBA overlay: {path}")
    image = image[:, :, :3]
    if image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            f"unexpected overlay shape {image.shape[:2]} for {path}; "
            f"expected {(IMAGE_HEIGHT, IMAGE_WIDTH)}"
        )
    return image


def clipped_range(start: float, end: float) -> tuple[int, int]:
    lo = max(0, int(round(start)))
    hi = min(IMAGE_HEIGHT - 1, int(round(end)))
    if hi < lo:
        hi = lo
    return lo, hi


def reference_ranges(center_v: float) -> dict[str, tuple[int, int]]:
    height = clipped_range(center_v - ROI_HALF_WIDTH, center_v + ROI_HALF_WIDTH)
    return {
        "baseline_before": clipped_range(
            height[0] - BASELINE_HALF_WIDTH, height[0] - BASELINE_GAP
        ),
        "height": height,
        "baseline_after": clipped_range(
            height[1] + BASELINE_GAP, height[1] + BASELINE_HALF_WIDTH
        ),
    }


def in_range(values: np.ndarray, value_range: tuple[int, int]) -> np.ndarray:
    return (values >= value_range[0]) & (values <= value_range[1])


def baseline_side(
    baseline_v: np.ndarray,
    before: tuple[int, int],
    after: tuple[int, int],
) -> tuple[str, int, int]:
    before_count = int(np.count_nonzero(in_range(baseline_v, before)))
    after_count = int(np.count_nonzero(in_range(baseline_v, after)))
    if before_count and after_count:
        side = "BOTH_SIDES"
    elif before_count:
        side = "BEFORE_LIKE"
    elif after_count:
        side = "AFTER_LIKE"
    else:
        before_distance = float(np.min(np.abs(baseline_v - np.mean(before))))
        after_distance = float(np.min(np.abs(baseline_v - np.mean(after))))
        side = "BEFORE_LIKE" if before_distance <= after_distance else "AFTER_LIKE"
    return side, before_count, after_count


def table_value(value: float) -> float:
    """Return the numeric value represented by a three-decimal table cell."""
    return float(f"{float(value):.3f}")


def summary_metrics(
    audits: list[PositionAudit], nominal_height_mm: float
) -> dict[str, float | int]:
    """Recompute aggregate metrics from the displayed table values."""
    heights = np.asarray([audit.table_height_mm for audit in audits], dtype=np.float64)
    errors = heights - nominal_height_mm
    return {
        "count": int(len(heights)),
        "mean_height_mm": float(np.mean(heights)),
        "bias_mm": float(np.mean(errors)),
        "mae_mm": float(np.mean(np.abs(errors))),
        "rmse_mm": float(np.sqrt(np.mean(errors * errors))),
        "max_abs_error_mm": float(np.max(np.abs(errors))),
        "std_population_mm": float(np.std(heights, ddof=0)),
        "std_sample_mm": float(np.std(heights, ddof=1)) if len(heights) > 1 else 0.0,
    }


def load_audits(measurement_root: Path, nominal_height_mm: float) -> list[PositionAudit]:
    audits: list[PositionAudit] = []
    for position, frame_id in enumerate(FRAME_IDS, start=1):
        frame_dir = measurement_root / f"frame_{frame_id}_measure"
        overlay_path = frame_dir / "overlay.png"
        height_path = frame_dir / "height_points.csv"
        baseline_path = frame_dir / "baseline_points.csv"
        laser_center_path = frame_dir / "laser_center.csv"
        result_path = frame_dir / "result.json"
        required = (
            overlay_path,
            height_path,
            baseline_path,
            laser_center_path,
            result_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"frame_{frame_id}: missing required artifacts: {missing}"
            )

        height_points = read_points(height_path)
        baseline_points = read_points(baseline_path)
        laser_center = read_points(laser_center_path)
        result = read_result(result_path)
        height_result = result.get("results_mm", {})
        if not isinstance(height_result, dict):
            raise ValueError(f"missing results_mm in {result_path}")
        height_mean = float(height_result["height_mean"])
        height_std = float(height_result["height_std"])
        vblock = float(np.median(height_points[:, 1]))
        vref = float(np.median(baseline_points[:, 1]))
        ranges = reference_ranges(vblock)
        side, before_support, after_support = baseline_side(
            baseline_points[:, 1],
            ranges["baseline_before"],
            ranges["baseline_after"],
        )
        table_height = table_value(height_mean)
        table_error = table_height - nominal_height_mm
        audits.append(
            PositionAudit(
                position=position,
                frame_id=frame_id,
                frame_dir=frame_dir,
                overlay_path=overlay_path,
                height_points_path=height_path,
                baseline_points_path=baseline_path,
                result_path=result_path,
                height_points=height_points,
                baseline_points=baseline_points,
                laser_center=laser_center,
                result=result,
                vblock_px=vblock,
                vref_px=vref,
                delta_v_px=vblock - vref,
                height_mean_mm=height_mean,
                height_std_mm=height_std,
                table_height_mm=table_height,
                table_error_mm=table_error,
                table_relative_error_pct=100.0 * table_error / nominal_height_mm,
                height_v_min=float(np.min(height_points[:, 1])),
                height_v_max=float(np.max(height_points[:, 1])),
                baseline_v_min=float(np.min(baseline_points[:, 1])),
                baseline_v_max=float(np.max(baseline_points[:, 1])),
                height_roi=ranges["height"],
                baseline_before=ranges["baseline_before"],
                baseline_after=ranges["baseline_after"],
                baseline_side=side,
                baseline_before_support=before_support,
                baseline_after_support=after_support,
                edge_clipped=(
                    ranges["baseline_before"][0] == 0
                    or ranges["baseline_after"][1] == IMAGE_HEIGHT - 1
                ),
            )
        )
    return audits


def format_range(value_range: tuple[int, int]) -> str:
    return f"[{value_range[0]}, {value_range[1]}]"


def plot_band(
    axis: plt.Axes,
    roi_id: str,
    value_range: tuple[int, int],
    *,
    label_prefix: str = "inferred",
    alpha: float = 0.16,
) -> None:
    color = ROI_COLORS[roi_id]
    label = f"{label_prefix} {roi_id} {format_range(value_range)}"
    axis.axhspan(value_range[0], value_range[1], color=color, alpha=alpha, label=label)
    axis.axhline(value_range[0], color=color, linewidth=0.7, linestyle="--")
    axis.axhline(value_range[1], color=color, linewidth=0.7, linestyle="--")


def draw_position_overlay(audit: PositionAudit, output_path: Path) -> None:
    image = read_overlay(audit.overlay_path)
    local_center = audit.laser_center.copy()
    local_center[:, 0] -= OFFSET_X
    height_local = audit.height_points.copy()
    height_local[:, 0] -= OFFSET_X
    baseline_local = audit.baseline_points.copy()
    baseline_local[:, 0] -= OFFSET_X

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 16.0),
        dpi=130,
        gridspec_kw={"width_ratios": [1.0, 1.05]},
    )
    figure.suptitle(
        f"Daheng 0811 P{audit.position:02d} | frame_{audit.frame_id} | "
        f"20 mm table result={audit.table_height_mm:.3f} mm",
        fontsize=14,
    )

    zoom_lo = max(
        0,
        int(
            min(
                audit.height_roi[0],
                audit.baseline_before[0],
                audit.baseline_after[0],
                audit.baseline_v_min,
            )
            - 35
        ),
    )
    zoom_hi = min(
        IMAGE_HEIGHT - 1,
        int(
            max(
                audit.height_roi[1],
                audit.baseline_before[1],
                audit.baseline_after[1],
                audit.baseline_v_max,
            )
            + 35
        ),
    )

    for index, (axis, y_limits) in enumerate(
        zip(axes, ((IMAGE_HEIGHT - 1, 0), (zoom_hi, zoom_lo)))
    ):
        axis.imshow(image, origin="upper", aspect="auto")
        for roi_id, value_range in (
            ("baseline_before", audit.baseline_before),
            ("height", audit.height_roi),
            ("baseline_after", audit.baseline_after),
        ):
            plot_band(axis, roi_id, value_range)
        axis.plot(
            local_center[:, 0],
            local_center[:, 1],
            color="#c77dff",
            linewidth=0.55,
            alpha=0.75,
            label="saved laser_center.csv",
        )
        axis.scatter(
            baseline_local[:, 0],
            baseline_local[:, 1],
            s=6,
            color="#ffd166",
            alpha=0.7,
            label=f"saved baseline points (n={len(baseline_local)})",
        )
        axis.scatter(
            height_local[:, 0],
            height_local[:, 1],
            s=8,
            color="#ff4d6d",
            alpha=0.8,
            label=f"saved height points (n={len(height_local)})",
        )
        axis.axhline(
            audit.vblock_px,
            color="white",
            linewidth=0.9,
            linestyle=":",
            label=f"vblock median={audit.vblock_px:.1f}",
        )
        axis.set_xlim(0, IMAGE_WIDTH - 1)
        axis.set_ylim(*y_limits)
        axis.set_xlabel("local u px (saved overlay / ROI width)")
        axis.set_ylabel("full-sensor v px")
        axis.grid(alpha=0.18)
        if index == 0:
            axis.set_title("full-sensor view")
            axis.legend(loc="upper right", fontsize=7, framealpha=0.88)
        else:
            axis.set_title(
                f"ROI neighborhood | v={zoom_lo}..{zoom_hi} | "
                f"observed baseline={audit.baseline_side}"
            )

    figure.text(
        0.5,
        0.012,
        "Inferred bands use 0822 ROI-freeze geometry (±45 / gap 20 / width 220); "
        "original 0811 ROI rectangles were not persisted. Saved CSV point envelopes "
        "are shown separately.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def draw_position_coverage(audits: list[PositionAudit], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 9.5), dpi=150, sharex=True)
    positions = np.asarray([audit.position for audit in audits], dtype=float)
    vblock = np.asarray([audit.vblock_px for audit in audits], dtype=float)
    vref = np.asarray([audit.vref_px for audit in audits], dtype=float)

    axes[0].plot(positions, vblock, "o-", color="#e76f51", label="vblock median")
    axes[0].plot(positions, vref, "s--", color="#457b9d", label="vref median")
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("v median (full-sensor px)")
    axes[0].set_title("0811 position geometry from saved point exports")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for audit in audits:
        axes[0].annotate(
            f"Δv={audit.delta_v_px:+.1f}",
            (audit.position, audit.vblock_px),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=7,
        )

    for audit in audits:
        axes[1].vlines(
            audit.position,
            audit.height_roi[0],
            audit.height_roi[1],
            color="#90be6d",
            linewidth=8,
            alpha=0.75,
        )
        axes[1].vlines(
            audit.position - 0.08,
            audit.baseline_v_min,
            audit.baseline_v_max,
            color="#f6bd60",
            linewidth=5,
            alpha=0.8,
        )
        axes[1].scatter(
            audit.position,
            audit.vblock_px,
            color="#ff4d6d",
            s=26,
            zorder=3,
        )
        axes[1].annotate(
            audit.baseline_side,
            (audit.position, audit.baseline_v_max),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            rotation=90,
        )
    axes[1].set_xlabel("position in original table")
    axes[1].set_ylabel("v range (full-sensor px)")
    axes[1].set_title(
        "Inferred height ROI and observed baseline point envelope; not original ROI bounds"
    )
    axes[1].set_xticks(positions)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def draw_error_vs_position(
    audits: list[PositionAudit], output_path: Path, nominal_height_mm: float
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), dpi=150)
    positions = np.asarray([audit.position for audit in audits], dtype=float)
    vblock = np.asarray([audit.vblock_px for audit in audits], dtype=float)
    errors = np.asarray([audit.table_error_mm for audit in audits], dtype=float)
    std = np.asarray([audit.height_std_mm for audit in audits], dtype=float)
    labels = [f"P{audit.position}" for audit in audits]

    axes[0].errorbar(
        positions,
        errors,
        yerr=std,
        fmt="o-",
        color="#d1495b",
        ecolor="#457b9d",
        capsize=3,
        label="table error ± saved height std",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(positions, labels)
    axes[0].set_xlabel("position in original table")
    axes[0].set_ylabel("error (mm; displayed height − nominal)")
    axes[0].set_title(f"20 mm error by table position (nominal={nominal_height_mm:g} mm)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].errorbar(
        vblock,
        errors,
        yerr=std,
        fmt="o",
        color="#2a9d8f",
        ecolor="#264653",
        capsize=3,
    )
    for x, y, label in zip(vblock, errors, labels):
        axes[1].annotate(label, (x, y), xytext=(3, 4), textcoords="offset points", fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("vblock median (full-sensor px)")
    axes[1].set_ylabel("error (mm)")
    axes[1].set_title("Error versus longitudinal image position")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def draw_point_support(audits: list[PositionAudit], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 5.8), dpi=150)
    positions = np.asarray([audit.position for audit in audits], dtype=float)
    height_total = np.asarray([len(audit.height_points) for audit in audits], dtype=float)
    height_inliers = np.asarray(
        [audit.result.get("point_counts", {}).get("height_inliers") for audit in audits],
        dtype=float,
    )
    baseline_total = np.asarray([len(audit.baseline_points) for audit in audits], dtype=float)
    baseline_inliers = np.asarray(
        [audit.result.get("point_counts", {}).get("baseline_inliers") for audit in audits],
        dtype=float,
    )
    width = 0.19
    axis.bar(positions - 1.5 * width, height_total, width, label="height total", color="#90be6d")
    axis.bar(positions - 0.5 * width, height_inliers, width, label="height inliers", color="#43aa8b")
    axis.bar(positions + 0.5 * width, baseline_total, width, label="baseline total", color="#f6bd60")
    axis.bar(positions + 1.5 * width, baseline_inliers, width, label="baseline inliers", color="#f8961e")
    axis.set_xticks(positions, [f"P{int(p)}" for p in positions])
    axis.set_xlabel("position in original table")
    axis.set_ylabel("saved point count")
    axis.set_title("Saved point support used by the historical 0811 measurements")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def write_csv(audits: list[PositionAudit], path: Path, nominal_height_mm: float) -> None:
    fields = [
        "position",
        "frame_id",
        "frame_dir",
        "vblock_px",
        "vref_px",
        "delta_v_px",
        "height_mean_raw_mm",
        "height_std_raw_mm",
        "table_height_mm",
        "table_error_mm",
        "table_relative_error_pct",
        "height_point_count",
        "height_inlier_count",
        "height_points_v_min",
        "height_points_v_max",
        "baseline_point_count",
        "baseline_inlier_count",
        "baseline_points_v_min",
        "baseline_points_v_max",
        "inferred_height_roi_start_v",
        "inferred_height_roi_end_v",
        "inferred_baseline_before_start_v",
        "inferred_baseline_before_end_v",
        "inferred_baseline_after_start_v",
        "inferred_baseline_after_end_v",
        "observed_baseline_side",
        "baseline_before_support_count",
        "baseline_after_support_count",
        "edge_clipped_inferred_band",
        "nominal_height_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            height_counts = audit.result.get("point_counts", {})
            writer.writerow(
                {
                    "position": audit.position,
                    "frame_id": audit.frame_id,
                    "frame_dir": str(audit.frame_dir),
                    "vblock_px": f"{audit.vblock_px:.3f}",
                    "vref_px": f"{audit.vref_px:.3f}",
                    "delta_v_px": f"{audit.delta_v_px:.3f}",
                    "height_mean_raw_mm": f"{audit.height_mean_mm:.9f}",
                    "height_std_raw_mm": f"{audit.height_std_mm:.9f}",
                    "table_height_mm": f"{audit.table_height_mm:.3f}",
                    "table_error_mm": f"{audit.table_error_mm:.3f}",
                    "table_relative_error_pct": f"{audit.table_relative_error_pct:.3f}",
                    "height_point_count": len(audit.height_points),
                    "height_inlier_count": height_counts.get("height_inliers"),
                    "height_points_v_min": f"{audit.height_v_min:.3f}",
                    "height_points_v_max": f"{audit.height_v_max:.3f}",
                    "baseline_point_count": len(audit.baseline_points),
                    "baseline_inlier_count": height_counts.get("baseline_inliers"),
                    "baseline_points_v_min": f"{audit.baseline_v_min:.3f}",
                    "baseline_points_v_max": f"{audit.baseline_v_max:.3f}",
                    "inferred_height_roi_start_v": audit.height_roi[0],
                    "inferred_height_roi_end_v": audit.height_roi[1],
                    "inferred_baseline_before_start_v": audit.baseline_before[0],
                    "inferred_baseline_before_end_v": audit.baseline_before[1],
                    "inferred_baseline_after_start_v": audit.baseline_after[0],
                    "inferred_baseline_after_end_v": audit.baseline_after[1],
                    "observed_baseline_side": audit.baseline_side,
                    "baseline_before_support_count": audit.baseline_before_support,
                    "baseline_after_support_count": audit.baseline_after_support,
                    "edge_clipped_inferred_band": audit.edge_clipped,
                    "nominal_height_mm": f"{nominal_height_mm:.3f}",
                }
            )


def write_summary(
    audits: list[PositionAudit], path: Path, nominal_height_mm: float
) -> None:
    payload = {
        "source": "three_decimal_table_values_from_result_json",
        "metrics": summary_metrics(audits, nominal_height_mm),
        "note": (
            "Metrics are recomputed from the displayed three-decimal heights, "
            "matching the original screenshot rounding convention."
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_provenance(
    audits: list[PositionAudit],
    path: Path,
    measurement_root: Path,
    output_dir: Path,
    nominal_height_mm: float,
) -> None:
    source_files: list[dict[str, str]] = []
    for audit in audits:
        for source in (
            audit.overlay_path,
            audit.height_points_path,
            audit.baseline_points_path,
            audit.result_path,
        ):
            source_files.append({"path": str(source.resolve()), "sha256": sha256(source)})
        calibration = audit.result.get("calibration", {})
        if isinstance(calibration, dict):
            for key, raw_path in calibration.items():
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                candidate = Path(raw_path)
                source_files.append(
                    {
                        "role": f"result.calibration.{key}",
                        "path": raw_path,
                        "sha256": sha256(candidate) if candidate.is_file() else "MISSING",
                    }
                )
    unique_files = {item["path"]: item for item in source_files}
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "visual_only_reconstruction_of_daheng_0811_20mm_table",
        "measurement_root": str(measurement_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "frame_ids": list(FRAME_IDS),
        "position_count": len(audits),
        "nominal_height_mm": nominal_height_mm,
        "source_images_read": False,
        "laser_center_extraction_rerun": False,
        "reconstruction_rerun": False,
        "calibration_refit": False,
        "original_roi_boundaries_available": False,
        "source_artifacts": list(unique_files.values()),
        "historical_result_protocol": {
            "extraction_method": sorted(
                {str(audit.result.get("extraction_method")) for audit in audits}
            ),
            "ground_reference_mode": sorted(
                {str(audit.result.get("ground_reference_mode")) for audit in audits}
            ),
            "laser_models": sorted(
                {
                    str(audit.result.get("calibration", {}).get("laser_model"))
                    for audit in audits
                    if isinstance(audit.result.get("calibration"), dict)
                }
            ),
        },
        "borrowed_visual_protocol": {
            "height_half_width_px": ROI_HALF_WIDTH,
            "baseline_gap_px": BASELINE_GAP,
            "baseline_half_width_px": BASELINE_HALF_WIDTH,
            "candidate_v_definition": "median v of saved height_points.csv",
            "baseline_support_definition": "saved baseline points counted inside inferred before/after bands",
            "source_reference": "daheng_0822_session01_roi_freeze geometry-only ROI review",
        },
        "limitations": [
            "The original 0811 raw TIFF/PNG-to-frame mapping is not available.",
            "The original 0811 ROI rectangles were not serialized in result.json.",
            "The background is the historical overlay.png, not the original raw image.",
            "The inferred before/after bands are a reference-style reconstruction, not proof of the historical GUI ROI.",
            "The eight positions have no 20-repeat structure, so Session01 median/CV visuals are not reproduced.",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(
    audits: list[PositionAudit],
    path: Path,
    measurement_root: Path,
    output_dir: Path,
    nominal_height_mm: float,
) -> None:
    lines = [
        "# Daheng 0811 20 mm 可视化重建",
        "",
        "## 结论边界",
        "",
        "本次只对已有 0811 测量产物做可视化重建：复用历史 `overlay.png`、"
        "`laser_center.csv`、`baseline_points.csv`、`height_points.csv` 和 `result.json`。"
        "没有读取原始 TIFF/PNG，没有重新提取 Steger，没有重新进行三维重建，也没有重拟合标定。",
        "",
        "图中的 ROI 是按 0822 Session01 的 geometry-only 协议反推的示意："
        f"height 为 vblock 中心 ±{ROI_HALF_WIDTH} px，前后基准区间距 {BASELINE_GAP} px、"
        f"半宽 {BASELINE_HALF_WIDTH} px。0811 的原始 ROI 四边界没有保存，因此图中以虚线/色带明确标为 inferred。",
        "",
        "## 可复用性审计",
        "",
        "- **复用结果（历史事实）**：8 个位置的测量值、标准差、vblock/vref 点集、激光中心线和原始结果叠加图。",
        "- **本轮新增计算**：从 CSV 计算 v 中位数、点集包络、参考式三段 ROI、基准支持侧、边界裁剪标记，以及误差/支持度/位置图。",
        "- **不可复用**：0822 的 20 次重复 median、人工冻结 registry、Session Ground、local-reference、CV 和 A-13B 正式统计。",
        "- **历史协议提示**：0811 的 `result.json` 使用 `baseline_roi_profile` 和历史激光模型；本报告没有用 0822 的校准数值替换它。",
        "",
        "## 位置审计",
        "",
        "|位置|frame|vblock|vref|Δv|表中测量值|误差|height点 v范围|baseline点 v范围|观察到的基准侧|边界裁剪|",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for audit in audits:
        lines.append(
            f"|{audit.position}|`frame_{audit.frame_id}`|{audit.vblock_px:.1f}|"
            f"{audit.vref_px:.1f}|{audit.delta_v_px:+.1f}|{audit.table_height_mm:.3f} mm|"
            f"{audit.table_error_mm:+.3f} mm|"
            f"{audit.height_v_min:.0f}–{audit.height_v_max:.0f}|"
            f"{audit.baseline_v_min:.0f}–{audit.baseline_v_max:.0f}|"
            f"{audit.baseline_side}|{'是' if audit.edge_clipped else '否'}|"
        )
    summary = summary_metrics(audits, nominal_height_mm)
    lines.extend(
        [
            "",
            "## 原表汇总复核",
            "",
            "以下使用图中显示的三位小数测量值复算，因此与原表的四位小数汇总保持一致：",
            "",
            "|指标|复算值|",
            "|---|---:|",
            f"|平均测量值|{summary['mean_height_mm']:.4f} mm|",
            f"|Bias|{summary['bias_mm']:.4f} mm|",
            f"|MAE|{summary['mae_mm']:.4f} mm|",
            f"|RMSE|{summary['rmse_mm']:.4f} mm|",
            f"|最大绝对误差|{summary['max_abs_error_mm']:.3f} mm|",
            f"|8 位置测量值样本标准差|{summary['std_sample_mm']:.4f} mm|",
            "",
            "完整精度复核见 `summary_metrics.json`。原始 JSON 中未四舍五入的高度均值与图中显示值存在极小差异，这是正常的显示/汇总舍入差异。",
            "",
            "## 生成的图件",
            "",
            "- `roi_review_overlays/`：8 张逐位置图。左侧为完整 3000 px 视场，右侧为 ROI 邻域；"
            "保存的中心线和实际点集与推断的三段 ROI 分层显示。",
            "- `position_geometry_audit.png`：vblock/vref 位置关系、推断 height ROI 与实际 baseline 点包络。",
            "- `height_error_by_position.png`：原表显示值误差及 `result.json` 中的高度标准差；右侧按 vblock 位置展开。",
            "- `point_support_audit.png`：height/baseline 点数与 inlier 点数。",
            "- `reprocessed_position_audit.csv`：全部推断字段和点支持统计。",
            "- `summary_metrics.json`：按原表显示值复算的综合指标。",
            "- `provenance.json`：源文件哈希、复用边界和限制。",
            "",
            "## 可借鉴的可视化设计",
            "",
            "1. 将原图/历史 overlay、中心线、height ROI、baseline_before、baseline_after 放在同一坐标系中，"
            "避免只给一个最终高度数字。",
            "2. 将候选 ROI、人工确认/冻结状态、edge clipping 和实际点支持分开表达；不要把算法搜索 ROI 当作点选择 ROI。",
            "3. 用 height ROI 的 v 中心作为位置坐标，并单独画误差随 v 的变化，从而显式暴露纵向一致性。",
            "4. 对基准点数、height 点数、inlier 数和 before/after 支持侧做质量审计，便于解释异常位置。",
            "",
            "## 限制",
            "",
            "- 当前图件的背景是历史 `overlay.png`，不是原始采集图；因此不能证明原始 ROI 框的像素边界。",
            "- P01–P07 的 baseline 点主要落在推断的 after-like 区域，P08 主要是 before-like；不能据此宣称 0811 原始流程使用了 0822 的双侧基准。",
            "- P01/P08 的参考式基准带触及图像边界，已保留 edge clipping 标记。",
            "- 这些图用于补足结果可追溯性和可视化表达，不会改变原图表中的数值结果。",
            "",
            f"源目录：`{measurement_root.resolve()}`",
            f"输出目录：`{output_dir.resolve()}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    measurement_root = args.measurement_root.resolve()
    output_dir = args.output.resolve()
    if not measurement_root.is_dir():
        raise FileNotFoundError(f"measurement root not found: {measurement_root}")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}; "
            "choose a new --output path"
        )

    audits = load_audits(measurement_root, args.nominal_height_mm)
    output_dir.mkdir(parents=True)
    overlay_dir = output_dir / "roi_review_overlays"
    overlay_dir.mkdir()
    for audit in audits:
        draw_position_overlay(
            audit,
            overlay_dir / f"position_{audit.position:02d}_frame_{audit.frame_id}_roi_review.png",
        )
    draw_position_coverage(audits, output_dir / "position_geometry_audit.png")
    draw_error_vs_position(
        audits,
        output_dir / "height_error_by_position.png",
        args.nominal_height_mm,
    )
    draw_point_support(audits, output_dir / "point_support_audit.png")
    write_csv(audits, output_dir / "reprocessed_position_audit.csv", args.nominal_height_mm)
    write_summary(
        audits,
        output_dir / "summary_metrics.json",
        args.nominal_height_mm,
    )
    write_provenance(
        audits,
        output_dir / "provenance.json",
        measurement_root,
        output_dir,
        args.nominal_height_mm,
    )
    write_report(
        audits,
        output_dir / "reprocessed_report.md",
        measurement_root,
        output_dir,
        args.nominal_height_mm,
    )
    print(f"generated {len(audits)} position overlays in {overlay_dir}")
    print(f"generated report and summary plots in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
