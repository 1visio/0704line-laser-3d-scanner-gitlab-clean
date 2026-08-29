"""Reuse saved laser pixels to compare circular-cone and quadratic-graph models.

The script deliberately does not read source images or run a laser-centre
extractor.  It reconstructs the already saved ``laser_center.csv`` pixels and
the already selected checkerboard pixels from ``height_points.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from reconstruction.reconstructor import ReconstructionResult, reconstruct_uv_to_ground
from utils.result_io import save_ground_pointcloud_ply, save_reconstructed_points_csv


MODEL_NAMES = ("circular_cone", "quadratic_graph")
METRIC_COLUMNS = (
    "count",
    "valid_rate",
    "mean_signed_mm",
    "mae_mm",
    "rmse_mm",
    "median_abs_mm",
    "p95_abs_mm",
    "max_abs_mm",
)


def _default_paths() -> dict[str, Path]:
    tool_root = Path(__file__).resolve().parents[1]
    workspace = tool_root.parents[1]
    run_root = workspace / "calibration_tool" / "projects" / "daheng" / "outputs" / "0811"
    measurement_root = tool_root / "output_daheng_0811"
    return {
        "config": tool_root / "configs" / "measure_tool_daheng_0811.yaml",
        "measurement_root": measurement_root,
        "calibration_run": run_root,
        "output": measurement_root / "model_comparison",
    }


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="复用 0811 已保存中心点，对比圆锥与二次图光片重建"
    )
    parser.add_argument("--config", type=Path, default=defaults["config"])
    parser.add_argument(
        "--measurement-root", type=Path, default=defaults["measurement_root"]
    )
    parser.add_argument(
        "--calibration-run", type=Path, default=defaults["calibration_run"]
    )
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--v-bin-width", type=float, default=100.0)
    return parser.parse_args()


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _read_uv(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"u", "v"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} 缺少 u,v 列")
        rows = [(float(row["u"]), float(row["v"])) for row in reader]
    points = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} 包含非有限像素坐标")
    return points


def _metrics(residual: np.ndarray, total_count: int) -> dict[str, float | int]:
    values = np.asarray(residual, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "count": int(len(values)),
        "valid_rate": float(len(values) / total_count) if total_count else math.nan,
        "mean_signed_mm": float(np.mean(values)) if len(values) else math.nan,
        "mae_mm": float(np.mean(absolute)) if len(values) else math.nan,
        "rmse_mm": float(np.sqrt(np.mean(values**2))) if len(values) else math.nan,
        "median_abs_mm": float(np.median(absolute)) if len(values) else math.nan,
        "p95_abs_mm": float(np.quantile(absolute, 0.95)) if len(values) else math.nan,
        "max_abs_mm": float(np.max(absolute)) if len(values) else math.nan,
    }


def _write_dict_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_measurement_residuals(
    path: Path,
    frame: str,
    model: str,
    result: ReconstructionResult,
) -> None:
    rows = []
    for (u, v), (xc, yc, zc), (xg, yg, zg) in zip(
        result.pixels_uv, result.points_camera, result.points_ground
    ):
        rows.append(
            {
                "frame": frame,
                "model": model,
                "u_px": f"{u:.6f}",
                "v_px": f"{v:.6f}",
                "residual_zg_mm": f"{zg:.9f}",
                "Xc_mm": f"{xc:.9f}",
                "Yc_mm": f"{yc:.9f}",
                "Zc_mm": f"{zc:.9f}",
                "Xg_mm": f"{xg:.9f}",
                "Yg_mm": f"{yg:.9f}",
            }
        )
    _write_dict_rows(
        path,
        [
            "frame",
            "model",
            "u_px",
            "v_px",
            "residual_zg_mm",
            "Xc_mm",
            "Yc_mm",
            "Zc_mm",
            "Xg_mm",
            "Yg_mm",
        ],
        rows,
    )


def _load_validation_residuals(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[tuple[float, float]]] = {name: [] for name in MODEL_NAMES}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "model", "v_px", "board_error_mm", "valid"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} 缺少独立验证残差列")
        for row in reader:
            model = row["model"]
            if (
                row["split"] == "validation"
                and model in values
                and row["valid"].strip().lower() in {"true", "1", "yes"}
            ):
                values[model].append((float(row["v_px"]), float(row["board_error_mm"])))
    return {
        model: np.asarray(rows, dtype=np.float64).reshape(-1, 2)
        for model, rows in values.items()
    }


def _bin_residuals(
    datasets: dict[str, list[np.ndarray]], bin_width: float
) -> list[dict[str, Any]]:
    if bin_width <= 0.0 or not math.isfinite(bin_width):
        raise ValueError("v-bin-width 必须是有限正数")
    rows: list[dict[str, Any]] = []
    for model, chunks in datasets.items():
        data = np.vstack(chunks) if chunks else np.empty((0, 2))
        if not len(data):
            continue
        start = math.floor(float(np.min(data[:, 0])) / bin_width) * bin_width
        stop = math.ceil(float(np.max(data[:, 0])) / bin_width) * bin_width
        edges = np.arange(start, stop + bin_width * 1.01, bin_width)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (data[:, 0] >= lo) & (data[:, 0] < hi)
            if hi == edges[-1]:
                mask |= data[:, 0] == hi
            selected = data[mask, 1]
            if not len(selected):
                continue
            metric = _metrics(selected, len(selected))
            rows.append(
                {
                    "model": model,
                    "v_bin_start_px": f"{lo:.3f}",
                    "v_bin_end_px": f"{hi:.3f}",
                    "v_bin_center_px": f"{0.5 * (lo + hi):.3f}",
                    **metric,
                }
            )
    return rows


def _plot_residuals(
    path: Path,
    title: str,
    y_label: str,
    datasets: dict[str, list[np.ndarray]],
    bin_width: float,
) -> None:
    colours = {"circular_cone": "#1f77b4", "quadratic_graph": "#d62728"}
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for model in MODEL_NAMES:
        chunks = datasets.get(model, [])
        if not chunks:
            continue
        data = np.vstack(chunks)
        ax.scatter(
            data[:, 0],
            data[:, 1],
            s=4,
            alpha=0.10,
            color=colours[model],
            rasterized=True,
            label=f"{model} points",
        )
        start = math.floor(float(np.min(data[:, 0])) / bin_width) * bin_width
        stop = math.ceil(float(np.max(data[:, 0])) / bin_width) * bin_width
        edges = np.arange(start, stop + bin_width * 1.01, bin_width)
        centres: list[float] = []
        means: list[float] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (data[:, 0] >= lo) & (data[:, 0] < hi)
            if np.any(mask):
                centres.append(0.5 * (lo + hi))
                means.append(float(np.mean(data[mask, 1])))
        ax.plot(
            centres,
            means,
            linewidth=2.2,
            color=colours[model],
            label=f"{model} {bin_width:g}px-bin mean",
        )
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("image row v (px)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _comparison_markdown(
    measurement_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    frame_count: int,
) -> str:
    measurement_overall = {
        row["model"]: row
        for row in measurement_rows
        if row["frame"] == "ALL"
    }
    validation = {row["model"]: row for row in validation_rows}
    lines = [
        "# 0811 Circular Cone / Quadratic Graph 同数据重建比较",
        "",
        "## 数据边界与残差定义",
        "",
        f"- 测量帧：{frame_count} 组；只读取既有 `laser_center.csv` 和 `height_points.csv` 的 `(u,v)`，未读取原图、未运行中心提取器。",
        "- 测量 `residual(v)`：已选棋盘点重建后的 `Zg(v)`，单位 mm；0811 地面外参把棋盘参考面定义为 `Zg=0`。",
        "- 标定 validation `residual(v)`：模型交点到每幅图真实棋盘平面的有符号距离，单位 mm；直接复用 0811 `pointwise_model_errors.csv`。",
        "- 两模型共用同一内参、地面外参、工作距离、像素点和有效性约束，只替换 laser surface model。",
        "",
        "## 已选测量点的 Zg 残差（全部帧）",
        "",
        "| model | valid points | valid rate | mean signed (mm) | MAE (mm) | RMSE (mm) | P95 abs (mm) | max abs (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        row = measurement_overall[model]
        lines.append(
            f"| {model} | {row['count']} | {row['valid_rate']:.6f} | "
            f"{row['mean_signed_mm']:.6f} | {row['mae_mm']:.6f} | "
            f"{row['rmse_mm']:.6f} | {row['p95_abs_mm']:.6f} | "
            f"{row['max_abs_mm']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 标定独立验证集的棋盘平面残差",
            "",
            "| model | valid points | valid rate | mean signed (mm) | MAE (mm) | RMSE (mm) | P95 abs (mm) | max abs (mm) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODEL_NAMES:
        row = validation[model]
        lines.append(
            f"| {model} | {row['count']} | {row['valid_rate']:.6f} | "
            f"{row['mean_signed_mm']:.6f} | {row['mae_mm']:.6f} | "
            f"{row['rmse_mm']:.6f} | {row['p95_abs_mm']:.6f} | "
            f"{row['max_abs_mm']:.6f} |"
        )
    measurement_delta = (
        float(measurement_overall["quadratic_graph"]["rmse_mm"])
        - float(measurement_overall["circular_cone"]["rmse_mm"])
    )
    validation_delta = (
        float(validation["quadratic_graph"]["rmse_mm"])
        - float(validation["circular_cone"]["rmse_mm"])
    )
    lines.extend(
        [
            "",
            "## 判读",
            "",
            f"- 测量选点：Quadratic Graph 相对 Circular Cone 的 RMSE 差值为 `{measurement_delta:+.6f} mm`（负值表示 Quadratic Graph 更小）。",
            f"- 标定 validation：Quadratic Graph 相对 Circular Cone 的 RMSE 差值为 `{validation_delta:+.6f} mm`（负值表示 Quadratic Graph 更小）。",
            "- 散点和分箱均值见两张 `residual_vs_v_*.png`；逐帧与分箱数字见 CSV。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    _prepare_output(args.output)
    app = load_app_config(args.config)
    model_paths = {
        name: args.calibration_run / "laser_model" / "models" / f"{name}.yaml"
        for name in MODEL_NAMES
    }
    calibrations = {
        name: load_calibration_files(
            app.calibration.intrinsics,
            model_path,
            app.calibration.extrinsics,
            app.calibration.ground_u_compensation,
        )
        for name, model_path in model_paths.items()
    }

    frame_dirs = sorted(
        path
        for path in args.measurement_root.glob("*_measure")
        if path.is_dir()
        and (path / "laser_center.csv").is_file()
        and (path / "height_points.csv").is_file()
    )
    if not frame_dirs:
        raise FileNotFoundError(f"未找到可复用测量目录: {args.measurement_root}")

    summary_rows: list[dict[str, Any]] = []
    measurement_plot_data: dict[str, list[np.ndarray]] = {
        name: [] for name in MODEL_NAMES
    }
    total_selected = 0
    aggregate_residuals: dict[str, list[np.ndarray]] = {
        name: [] for name in MODEL_NAMES
    }
    provenance_frames: list[dict[str, Any]] = []

    for frame_dir in frame_dirs:
        frame = frame_dir.name.removesuffix("_measure")
        full_uv = _read_uv(frame_dir / "laser_center.csv")
        selected_uv = _read_uv(frame_dir / "height_points.csv")
        total_selected += len(selected_uv)
        provenance_frames.append(
            {
                "frame": frame,
                "laser_center_csv": str((frame_dir / "laser_center.csv").resolve()),
                "selected_uv_source": str((frame_dir / "height_points.csv").resolve()),
                "full_pixel_count": len(full_uv),
                "selected_pixel_count": len(selected_uv),
            }
        )

        for model in MODEL_NAMES:
            model_dir = args.output / "reconstructions" / model / frame_dir.name
            model_dir.mkdir(parents=True, exist_ok=False)
            full_result = reconstruct_uv_to_ground(
                full_uv, calibrations[model], app.reconstruction
            )
            selected_result = reconstruct_uv_to_ground(
                selected_uv, calibrations[model], app.reconstruction
            )
            save_reconstructed_points_csv(
                model_dir / "full_points.csv",
                full_result.pixels_uv,
                full_result.points_camera,
                full_result.points_ground,
            )
            save_ground_pointcloud_ply(
                model_dir / "full_laser_ground.ply", full_result.points_ground
            )
            _write_measurement_residuals(
                model_dir / "selected_points_residual.csv",
                frame,
                model,
                selected_result,
            )
            residual = selected_result.points_ground[:, 2]
            metric = _metrics(residual, len(selected_uv))
            summary_rows.append({"frame": frame, "model": model, **metric})
            pair = np.column_stack([selected_result.pixels_uv[:, 1], residual])
            measurement_plot_data[model].append(pair)
            aggregate_residuals[model].append(residual)
            with (model_dir / "reconstruction_metadata.json").open(
                "x", encoding="utf-8"
            ) as stream:
                json.dump(
                    {
                        "model": model,
                        "model_path": str(model_paths[model].resolve()),
                        "source_laser_center_csv": str(
                            (frame_dir / "laser_center.csv").resolve()
                        ),
                        "source_selected_uv_csv": str(
                            (frame_dir / "height_points.csv").resolve()
                        ),
                        "full_input_count": len(full_uv),
                        "full_valid_count": full_result.point_count,
                        "full_filtered": full_result.filtered,
                        "selected_input_count": len(selected_uv),
                        "selected_valid_count": selected_result.point_count,
                        "selected_filtered": selected_result.filtered,
                    },
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
                stream.write("\n")

    for model in MODEL_NAMES:
        residual = np.concatenate(aggregate_residuals[model])
        summary_rows.append(
            {"frame": "ALL", "model": model, **_metrics(residual, total_selected)}
        )
    _write_dict_rows(
        args.output / "measurement_residual_summary.csv",
        ["frame", "model", *METRIC_COLUMNS],
        summary_rows,
    )

    measurement_bins = _bin_residuals(measurement_plot_data, args.v_bin_width)
    _write_dict_rows(
        args.output / "measurement_residual_vs_v_bins.csv",
        ["model", "v_bin_start_px", "v_bin_end_px", "v_bin_center_px", *METRIC_COLUMNS],
        measurement_bins,
    )
    _plot_residuals(
        args.output / "residual_vs_v_measurement.png",
        "Saved checkerboard selections: residual(v)",
        "ground-plane residual Zg (mm)",
        measurement_plot_data,
        args.v_bin_width,
    )

    validation_data = _load_validation_residuals(
        args.calibration_run / "laser_model" / "pointwise_model_errors.csv"
    )
    validation_plot_data = {
        model: [values] for model, values in validation_data.items()
    }
    validation_rows = [
        {
            "model": model,
            **_metrics(values[:, 1], len(values)),
        }
        for model, values in validation_data.items()
    ]
    _write_dict_rows(
        args.output / "calibration_validation_residual_summary.csv",
        ["model", *METRIC_COLUMNS],
        validation_rows,
    )
    validation_bins = _bin_residuals(validation_plot_data, args.v_bin_width)
    _write_dict_rows(
        args.output / "calibration_validation_residual_vs_v_bins.csv",
        ["model", "v_bin_start_px", "v_bin_end_px", "v_bin_center_px", *METRIC_COLUMNS],
        validation_bins,
    )
    _plot_residuals(
        args.output / "residual_vs_v_calibration_validation.png",
        "0811 calibration validation: residual(v)",
        "signed checkerboard-plane residual (mm)",
        validation_plot_data,
        args.v_bin_width,
    )

    provenance = {
        "source_images_declared_by_existing_results": [
            str(
                (
                    Path(
                        json.loads((frame_dir / "result.json").read_text(encoding="utf-8"))[
                            "image"
                        ]
                    )
                )
            )
            for frame_dir in frame_dirs
        ],
        "source_measurement_root": str(args.measurement_root.resolve()),
        "source_calibration_run": str(args.calibration_run.resolve()),
        "intrinsics": str(app.calibration.intrinsics.resolve()),
        "extrinsics": str(app.calibration.extrinsics.resolve()),
        "models": {name: str(path.resolve()) for name, path in model_paths.items()},
        "centre_extraction_rerun": False,
        "source_images_read": False,
        "frames": provenance_frames,
    }
    with (args.output / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    report = _comparison_markdown(summary_rows, validation_rows, len(frame_dirs))
    (args.output / "comparison_report.md").write_text(report, encoding="utf-8")
    print(f"comparison written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
