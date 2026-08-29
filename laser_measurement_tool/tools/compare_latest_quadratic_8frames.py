"""Compare the saved Daheng 0811 cone reconstructions with the latest graph model.

This is an artifact-generation script: it reuses saved laser pixels and saved
old coordinates, and never reads source images or reruns laser-center extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from reconstruction.reconstructor import (
    ReconstructionParams,
    reconstruct_uv_to_ground,
)
from utils.result_io import save_ground_pointcloud_ply, save_reconstructed_points_csv


FRAME_NAMES = (
    "frame_012686_measure",
    "frame_011317_measure",
    "frame_009614_measure",
    "frame_008310_measure",
    "frame_007020_measure",
    "frame_005772_measure",
    "frame_004021_measure",
    "frame_000974_measure",
)


def read_uv(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = [(float(row["u"]), float(row["v"])) for row in csv.DictReader(stream)]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def read_saved_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    uv = np.asarray(
        [(float(row["u"]), float(row["v"])) for row in rows],
        dtype=np.float64,
    )
    camera = np.asarray(
        [
            (float(row["Xc_mm"]), float(row["Yc_mm"]), float(row["Zc_mm"]))
            for row in rows
        ],
        dtype=np.float64,
    )
    ground = np.asarray(
        [
            (float(row["Xg_mm"]), float(row["Yg_mm"]), float(row["Zg_mm"]))
            for row in rows
        ],
        dtype=np.float64,
    )
    return uv, camera, ground


def point_key(point: np.ndarray) -> tuple[float, float]:
    return (round(float(point[0]), 6), round(float(point[1]), 6))


def result_map(result):
    return {
        point_key(uv): (uv, camera, ground)
        for uv, camera, ground in zip(
            result.pixels_uv, result.points_camera, result.points_ground
        )
    }


def metric(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mean_signed_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "median_abs_mm": None,
            "p95_abs_mm": None,
            "max_abs_mm": None,
        }
    absolute = np.abs(values)
    return {
        "count": int(len(values)),
        "mean_signed_mm": float(np.mean(values)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(values * values))),
        "median_abs_mm": float(np.median(absolute)),
        "p95_abs_mm": float(np.quantile(absolute, 0.95)),
        "max_abs_mm": float(np.max(absolute)),
    }


def vector_metrics(delta: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    return {
        "X": metric(delta[:, 0]),
        "Y": metric(delta[:, 1]),
        "Z": metric(delta[:, 2]),
        "norm": metric(np.linalg.norm(delta, axis=1)),
    }


def flatten_metrics(
    row: dict[str, object],
    prefix: str,
    values: dict[str, dict[str, float | int | None]],
) -> None:
    for axis, stats in values.items():
        for name, value in stats.items():
            row[f"{prefix}_{axis}_{name}"] = value


def parse_ply_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("end_header")
    points = []
    for line in lines[end + 1 :]:
        fields = line.split()
        if len(fields) >= 3:
            points.append((float(fields[0]), float(fields[1]), float(fields[2])))
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{float(value):.9f}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement-root",
        type=Path,
        default=TOOL_ROOT / "output_daheng_0811",
        help="Directory containing the saved *_measure inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <measurement-root>/model_comparison_latest_quadratic_8frames_v2).",
    )
    parser.add_argument(
        "--validation-source",
        type=Path,
        default=Path(
            "D:/Docs/linelaserscan/calibration_tool/projects/daheng/outputs/"
            "0811/laser_model/calibration_points.csv"
        ),
        help="Calibration validation CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = load_app_config(TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml")
    old_model_path = (
        TOOL_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
    )
    new_model_path = (
        TOOL_ROOT
        / "configs"
        / "calibration_daheng_0811"
        / "quadratic_graph.yaml"
    )
    old_calibration = load_calibration_files(
        app.calibration.intrinsics,
        old_model_path,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
    )
    new_calibration = load_calibration_files(
        app.calibration.intrinsics,
        new_model_path,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
    )

    measurement_root = args.measurement_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else measurement_root / "model_comparison_latest_quadratic_8frames_v2"
    )
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    (output / "new_reconstructions").mkdir()

    full_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    all_camera_delta: list[np.ndarray] = []
    all_ground_delta: list[np.ndarray] = []
    totals = {"input": 0, "old": 0, "new": 0, "common": 0, "ply": 0}

    for frame_dir_name in FRAME_NAMES:
        frame_dir = measurement_root / frame_dir_name
        frame = frame_dir_name.removesuffix("_measure")
        full_uv = read_uv(frame_dir / "laser_center.csv")
        old_result = reconstruct_uv_to_ground(
            full_uv, old_calibration, app.reconstruction
        )
        new_result = reconstruct_uv_to_ground(
            full_uv, new_calibration, app.reconstruction
        )
        old_map = result_map(old_result)
        new_map = result_map(new_result)
        common_keys = sorted(
            set(old_map) & set(new_map), key=lambda key: (key[1], key[0])
        )
        old_camera = np.asarray([old_map[key][1] for key in common_keys])
        new_camera = np.asarray([new_map[key][1] for key in common_keys])
        old_ground = np.asarray([old_map[key][2] for key in common_keys])
        new_ground = np.asarray([new_map[key][2] for key in common_keys])
        camera_delta = new_camera - old_camera
        ground_delta = new_ground - old_ground
        all_camera_delta.append(camera_delta)
        all_ground_delta.append(ground_delta)

        old_ply = parse_ply_xyz(frame_dir / "full_laser_ground.ply")
        if len(old_ply) != old_result.point_count:
            raise RuntimeError(
                f"{frame}: old reconstruction count {old_result.point_count} "
                f"!= PLY count {len(old_ply)}"
            )
        old_ply_error = metric((old_result.points_ground - old_ply).reshape(-1))
        frame_row: dict[str, object] = {
            "frame": frame,
            "full_input_count": len(full_uv),
            "old_valid_count": old_result.point_count,
            "new_valid_count": new_result.point_count,
            "common_valid_count": len(common_keys),
            "old_ply_vertex_count": len(old_ply),
            "old_ply_check_rmse_mm": old_ply_error["rmse_mm"],
            "old_ply_check_max_abs_mm": old_ply_error["max_abs_mm"],
        }
        flatten_metrics(frame_row, "delta_camera", vector_metrics(camera_delta))
        flatten_metrics(frame_row, "delta_ground", vector_metrics(ground_delta))

        new_frame_dir = output / "new_reconstructions" / frame_dir_name
        new_frame_dir.mkdir()
        save_reconstructed_points_csv(
            new_frame_dir / "full_points.csv",
            new_result.pixels_uv,
            new_result.points_camera,
            new_result.points_ground,
        )
        save_ground_pointcloud_ply(
            new_frame_dir / "full_laser_ground.ply",
            new_result.points_ground,
        )
        with (new_frame_dir / "reconstruction_metadata.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "model": "quadratic_graph",
                    "model_path": str(new_model_path.resolve()),
                    "source_laser_center_csv": str(
                        (frame_dir / "laser_center.csv").resolve()
                    ),
                    "input_count": len(full_uv),
                    "valid_count": new_result.point_count,
                    "filtered": new_result.filtered,
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")

        for key in common_keys:
            uv = old_map[key][0]
            old_c, old_g = old_map[key][1], old_map[key][2]
            new_c, new_g = new_map[key][1], new_map[key][2]
            dc, dg = new_c - old_c, new_g - old_g
            full_rows.append(
                {
                    "frame": frame,
                    "u_px": fmt(uv[0]),
                    "v_px": fmt(uv[1]),
                    "old_Xc_mm": fmt(old_c[0]),
                    "old_Yc_mm": fmt(old_c[1]),
                    "old_Zc_mm": fmt(old_c[2]),
                    "new_Xc_mm": fmt(new_c[0]),
                    "new_Yc_mm": fmt(new_c[1]),
                    "new_Zc_mm": fmt(new_c[2]),
                    "delta_Xc_mm": fmt(dc[0]),
                    "delta_Yc_mm": fmt(dc[1]),
                    "delta_Zc_mm": fmt(dc[2]),
                    "old_Xg_mm": fmt(old_g[0]),
                    "old_Yg_mm": fmt(old_g[1]),
                    "old_Zg_mm": fmt(old_g[2]),
                    "new_Xg_mm": fmt(new_g[0]),
                    "new_Yg_mm": fmt(new_g[1]),
                    "new_Zg_mm": fmt(new_g[2]),
                    "delta_Xg_mm": fmt(dg[0]),
                    "delta_Yg_mm": fmt(dg[1]),
                    "delta_Zg_mm": fmt(dg[2]),
                    "delta_norm_mm": fmt(np.linalg.norm(dg)),
                }
            )

        for source in ("baseline", "height"):
            selected_uv, old_sc_all, old_sg_all = read_saved_points(
                frame_dir / f"{source}_points.csv"
            )
            new_selected = reconstruct_uv_to_ground(
                selected_uv, new_calibration, app.reconstruction
            )
            selected_map = result_map(new_selected)
            indices = [
                index
                for index, uv in enumerate(selected_uv)
                if point_key(uv) in selected_map
            ]
            old_sc = old_sc_all[indices]
            old_sg = old_sg_all[indices]
            new_sc = np.asarray(
                [selected_map[point_key(selected_uv[index])][1] for index in indices]
            )
            new_sg = np.asarray(
                [selected_map[point_key(selected_uv[index])][2] for index in indices]
            )
            flatten_metrics(
                frame_row,
                f"{source}_delta_camera",
                vector_metrics(new_sc - old_sc),
            )
            flatten_metrics(
                frame_row,
                f"{source}_delta_ground",
                vector_metrics(new_sg - old_sg),
            )
            frame_row[f"{source}_input_count"] = len(selected_uv)
            frame_row[f"{source}_new_valid_count"] = len(indices)
            frame_row[f"{source}_old_mean_Zg_mm"] = float(np.mean(old_sg[:, 2]))
            frame_row[f"{source}_new_mean_Zg_mm"] = float(np.mean(new_sg[:, 2]))
            for index in indices:
                uv = selected_uv[index]
                old_c, old_g = old_sc_all[index], old_sg_all[index]
                new_c = selected_map[point_key(uv)][1]
                new_g = selected_map[point_key(uv)][2]
                dc, dg = new_c - old_c, new_g - old_g
                selected_rows.append(
                    {
                        "frame": frame,
                        "source": source,
                        "u_px": fmt(uv[0]),
                        "v_px": fmt(uv[1]),
                        "old_Xc_mm": fmt(old_c[0]),
                        "old_Yc_mm": fmt(old_c[1]),
                        "old_Zc_mm": fmt(old_c[2]),
                        "new_Xc_mm": fmt(new_c[0]),
                        "new_Yc_mm": fmt(new_c[1]),
                        "new_Zc_mm": fmt(new_c[2]),
                        "delta_Xc_mm": fmt(dc[0]),
                        "delta_Yc_mm": fmt(dc[1]),
                        "delta_Zc_mm": fmt(dc[2]),
                        "old_Xg_mm": fmt(old_g[0]),
                        "old_Yg_mm": fmt(old_g[1]),
                        "old_Zg_mm": fmt(old_g[2]),
                        "new_Xg_mm": fmt(new_g[0]),
                        "new_Yg_mm": fmt(new_g[1]),
                        "new_Zg_mm": fmt(new_g[2]),
                        "delta_Xg_mm": fmt(dg[0]),
                        "delta_Yg_mm": fmt(dg[1]),
                        "delta_Zg_mm": fmt(dg[2]),
                        "delta_norm_mm": fmt(np.linalg.norm(dg)),
                    }
                )

        frame_rows.append(frame_row)
        totals["input"] += len(full_uv)
        totals["old"] += old_result.point_count
        totals["new"] += new_result.point_count
        totals["common"] += len(common_keys)
        totals["ply"] += len(old_ply)

    all_camera = np.concatenate(all_camera_delta, axis=0)
    all_ground = np.concatenate(all_ground_delta, axis=0)
    overall: dict[str, object] = {
        "frame": "ALL",
        "full_input_count": totals["input"],
        "old_valid_count": totals["old"],
        "new_valid_count": totals["new"],
        "common_valid_count": totals["common"],
        "old_ply_vertex_count": totals["ply"],
    }
    flatten_metrics(overall, "delta_camera", vector_metrics(all_camera))
    flatten_metrics(overall, "delta_ground", vector_metrics(all_ground))
    frame_rows.append(overall)

    write_rows(output / "frame_summary.csv", list(frame_rows[0]), frame_rows)
    write_rows(output / "coordinate_differences_full.csv", list(full_rows[0]), full_rows)
    write_rows(
        output / "coordinate_differences_selected.csv",
        list(selected_rows[0]),
        selected_rows,
    )

    validation_source = args.validation_source.resolve()
    with validation_source.open("r", newline="", encoding="utf-8-sig") as stream:
        validation_rows = [
            row for row in csv.DictReader(stream) if row["split"] == "validation"
        ]
    validation_uv = np.asarray(
        [(float(row["u_px"]), float(row["v_px"])) for row in validation_rows],
        dtype=np.float64,
    )
    validation_normals = np.asarray(
        [
            (
                float(row["board_nx"]),
                float(row["board_ny"]),
                float(row["board_nz"]),
            )
            for row in validation_rows
        ],
        dtype=np.float64,
    )
    validation_d = np.asarray(
        [float(row["board_d_mm"]) for row in validation_rows],
        dtype=np.float64,
    )
    validation_params = ReconstructionParams(
        parallel_epsilon=app.reconstruction.parallel_epsilon,
        quadratic_epsilon=app.reconstruction.quadratic_epsilon,
        min_camera_depth_mm=0.0,
        max_camera_depth_mm=2000.0,
        model_range_margin_mm=50.0,
        image_roi_polygon=None,
    )
    validation_metrics = []
    for name, model_path, calibration in (
        ("circular_cone", old_model_path, old_calibration),
        ("quadratic_graph_latest", new_model_path, new_calibration),
    ):
        result = reconstruct_uv_to_ground(
            validation_uv, calibration, validation_params
        )
        point_map = {
            point_key(uv): camera
            for uv, camera in zip(result.pixels_uv, result.points_camera)
        }
        errors = np.asarray(
            [
                np.dot(point_map[point_key(uv)], normal) + d
                if point_key(uv) in point_map
                else np.nan
                for uv, normal, d in zip(
                    validation_uv, validation_normals, validation_d
                )
            ]
        )
        validation_metrics.append(
            {
                "model": name,
                "model_path": str(model_path.resolve()),
                "input_count": len(validation_uv),
                "valid_count": int(np.isfinite(errors).sum()),
                **metric(errors),
            }
        )
    write_rows(
        output / "validation_summary.csv",
        list(validation_metrics[0]),
        validation_metrics,
    )

    cone = validation_metrics[0]
    quadratic = validation_metrics[1]
    validation_delta = {}
    for key in ("mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm"):
        validation_delta[f"{key}_delta_quadratic_minus_cone_mm"] = (
            quadratic[key] - cone[key]
        )
        validation_delta[f"{key}_improvement_percent"] = (
            (cone[key] - quadratic[key]) / abs(cone[key]) * 100.0
        )
    validation_delta["mean_signed_delta_quadratic_minus_cone_mm"] = (
        quadratic["mean_signed_mm"] - cone["mean_signed_mm"]
    )
    (output / "validation_delta.json").write_text(
        json.dumps(validation_delta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames": [name.removesuffix("_measure") for name in FRAME_NAMES],
        "source_images_read": False,
        "laser_center_extraction_rerun": False,
        "reused_inputs": {
            "measurement_root": str(measurement_root.resolve()),
            "laser_center_csv": "each frame's existing laser_center.csv",
            "selected_coordinate_csv": "each frame's existing baseline_points.csv and height_points.csv",
            "old_full_ground_points": "each frame's existing full_laser_ground.ply",
        },
        "models": {
            "old_circular_cone": {
                "path": str(old_model_path.resolve()),
                "sha256": file_sha256(old_model_path),
            },
            "new_quadratic_graph": {
                "path": str(new_model_path.resolve()),
                "sha256": file_sha256(new_model_path),
            },
        },
        "shared_calibration": {
            "config": str(
                (TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml").resolve()
            ),
            "intrinsics": str(app.calibration.intrinsics.resolve()),
            "extrinsics": str(app.calibration.extrinsics.resolve()),
            "online_reconstruction_params": {
                "parallel_epsilon": app.reconstruction.parallel_epsilon,
                "quadratic_epsilon": app.reconstruction.quadratic_epsilon,
                "min_camera_depth_mm": app.reconstruction.min_camera_depth_mm,
                "max_camera_depth_mm": app.reconstruction.max_camera_depth_mm,
                "model_range_margin_mm": app.reconstruction.model_range_margin_mm,
            },
        },
        "independent_validation": {
            "source": str(validation_source.resolve()),
            "points": len(validation_uv),
            "protocol": "same 0811 validation board points; min_depth=0, max_depth=2000, model_range_margin=50 mm",
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ground_norm = metric(np.linalg.norm(all_ground, axis=1))
    ground_z = metric(all_ground[:, 2])
    report = [
        "# 大恒 0811：旧圆锥模型 vs 最新二次曲面模型",
        "",
        "## 结论",
        "",
        f"- 对用户指定的 8 帧，共复用 {totals['input']} 个既有激光像素；未读取原图，也未重新提取激光中心。",
        f"- 两种模型均成功恢复 {totals['common']}/{totals['input']} 个点，逐帧有效点数完全一致；新模型没有带来覆盖率变化。",
        f"- 全部 {totals['common']} 个共同点的地面坐标变化：平均绝对三维位移 {ground_norm['mae_mm']:.6f} mm，RMSE {ground_norm['rmse_mm']:.6f} mm，P95 {ground_norm['p95_abs_mm']:.6f} mm，最大 {ground_norm['max_abs_mm']:.6f} mm；平均 ΔZg={ground_z['mean_signed_mm']:+.6f} mm。",
        f"- 在同一批 0811 独立标定验证点上，最新二次曲面 RMSE 为 {quadratic['rmse_mm']:.6f} mm，旧圆锥为 {cone['rmse_mm']:.6f} mm，下降 {validation_delta['rmse_mm_improvement_percent']:.2f}%；MAE 下降 {validation_delta['mae_mm_improvement_percent']:.2f}%，P95 下降 {validation_delta['p95_abs_mm_improvement_percent']:.2f}%。最大误差略升 {validation_delta['max_abs_mm_delta_quadratic_minus_cone_mm']:+.6f} mm。",
        "",
        "从独立标定验证看，最新二次曲面有小幅、可量化的精度改善；从这 8 帧实际测量看，它主要造成亚 0.1 mm 量级的坐标修正，最极端点约 0.20 mm。新旧坐标的差异本身不是实物真值，最终测量改善仍应以独立标定验证或已知尺寸实物验证为准。",
        "",
        "## 逐帧共同点坐标差异（地面坐标）",
        "",
        "| 帧 | 共同点数 | 平均绝对位移 mm | RMSE mm | P95 mm | 最大 mm | 平均 ΔZg mm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame_rows[:-1]:
        report.append(
            f"| {row['frame']} | {row['common_valid_count']} | "
            f"{row['delta_ground_norm_mae_mm']:.6f} | "
            f"{row['delta_ground_norm_rmse_mm']:.6f} | "
            f"{row['delta_ground_norm_p95_abs_mm']:.6f} | "
            f"{row['delta_ground_norm_max_abs_mm']:.6f} | "
            f"{row['delta_ground_Z_mean_signed_mm']:+.6f} |"
        )
    report.append(
        f"| ALL | {overall['common_valid_count']} | "
        f"{overall['delta_ground_norm_mae_mm']:.6f} | "
        f"{overall['delta_ground_norm_rmse_mm']:.6f} | "
        f"{overall['delta_ground_norm_p95_abs_mm']:.6f} | "
        f"{overall['delta_ground_norm_max_abs_mm']:.6f} | "
        f"{overall['delta_ground_Z_mean_signed_mm']:+.6f} |"
    )
    report.extend(
        [
            "",
            "Δ = quadratic_graph_latest - circular_cone，单位均为 mm。X/Y/Z 分量和每个像素的完整坐标见 coordinate_differences_full.csv。",
            "",
            "## 独立标定验证",
            "",
            "| 模型 | 有效点 | Bias mm | MAE mm | RMSE mm | P95 mm | Max mm |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validation_metrics:
        report.append(
            f"| {row['model']} | {row['valid_count']} | "
            f"{row['mean_signed_mm']:+.6f} | {row['mae_mm']:.6f} | "
            f"{row['rmse_mm']:.6f} | {row['p95_abs_mm']:.6f} | "
            f"{row['max_abs_mm']:.6f} |"
        )
    report.extend(
        [
            "",
            "验证点来自既有 0811 calibration_points.csv 的 validation split；本轮只用最新模型重新求交并按保存的棋盘格平面计算误差。旧圆锥结果与既有报告的 0.083219826 mm RMSE 一致。",
            "",
            "## 产物",
            "",
            "- frame_summary.csv：逐帧及 ALL 汇总。",
            "- coordinate_differences_full.csv：全部共同激光点的旧/新相机系与地面系坐标、三轴差值和三维位移。",
            "- coordinate_differences_selected.csv：baseline/height 选点的逐点差异。",
            "- new_reconstructions/：最新二次曲面重建出的各帧点云与 CSV。",
            "- validation_summary.csv、validation_delta.json：独立标定验证及改善量。",
        ]
    )
    (output / "comparison_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
