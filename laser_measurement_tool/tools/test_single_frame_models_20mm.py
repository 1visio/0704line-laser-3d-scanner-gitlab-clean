"""Test the Daheng 0811 cone and quadratic models on one online snapshot.

The image is processed through the same extraction and reconstruction APIs as
the online tool.  The output is intentionally a new, self-contained artifact
directory so that no previous measurement result is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_ROOT.parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from correction.stage_a_height_scale import (
    StageAHeightResult,
    resolve_stage_a_height_scale,
)
from laser.backends import create_extraction_params
from laser.laser_extractor import extract_laser_center
from measurement.height_measure import HeightLineMeasurement, measure_height_line
from reconstruction.reconstructor import reconstruct_uv_to_ground
from utils.image_io import load_grayscale_image
from utils.result_io import (
    save_ground_pointcloud_ply,
    save_laser_centers_csv,
    save_reconstructed_points_csv,
)


FRAME_NAME = "frame_000667"
TRUE_HEIGHT_MM = 20.0
HEIGHT_V_RANGE = (1600, 1693)
BASELINE_LOCAL_V_RANGES = ((1365, 1572), (1778, 1985))
OUTPUT_NAME = "frame_000667_model_test_20mm_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "tif",
        help="Directory containing frame_000667.png and its JSON metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TOOL_ROOT / "output_daheng_0811" / OUTPUT_NAME,
        help="Output directory for the generated comparison artifact.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: float | int | np.floating | np.integer) -> float:
    return float(value)


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


def point_key(point: np.ndarray) -> tuple[float, float]:
    return (round(float(point[0]), 9), round(float(point[1]), 9))


def result_map(result: Any) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        point_key(uv): (uv, camera, ground)
        for uv, camera, ground in zip(
            result.pixels_uv, result.points_camera, result.points_ground
        )
    }


def fmt(value: float | int | None) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return f"{float(value):.9f}"


def region_for_v(v: float) -> str:
    if HEIGHT_V_RANGE[0] <= v <= HEIGHT_V_RANGE[1]:
        return "height_roi"
    if BASELINE_LOCAL_V_RANGES[0][0] <= v <= BASELINE_LOCAL_V_RANGES[0][1]:
        return "baseline_local_before"
    if BASELINE_LOCAL_V_RANGES[1][0] <= v <= BASELINE_LOCAL_V_RANGES[1][1]:
        return "baseline_local_after"
    return "other"


def measurement_record(
    model_name: str,
    baseline_name: str,
    measurement: HeightLineMeasurement,
    baseline_ground: np.ndarray,
    height_ground: np.ndarray,
    stage_a: StageAHeightResult,
) -> dict[str, Any]:
    height_inliers = height_ground[measurement.height_fit.inlier_mask]
    if measurement.ground_profile_fit is None:
        local_ground = np.zeros(len(height_inliers), dtype=np.float64)
        profile_slope = 0.0
        profile_intercept = 0.0
        profile_rmse = None
    else:
        local_ground = measurement.ground_profile_fit.predict_z(height_inliers[:, :2])
        profile_slope = measurement.ground_profile_fit.slope_z_per_mm
        profile_intercept = measurement.ground_profile_fit.intercept_z_mm
        profile_rmse = measurement.ground_profile_fit.rmse_mm

    raw_top_z_mean = float(np.mean(height_inliers[:, 2]))
    local_ground_z_mean = float(np.mean(local_ground))
    return {
        "model": model_name,
        "baseline_selection": baseline_name,
        "baseline_input_count": measurement.baseline_point_count,
        "baseline_inlier_count": measurement.baseline_inlier_count,
        "height_input_count": measurement.height_point_count,
        "height_inlier_count": measurement.height_inlier_count,
        "height_mean_mm": measurement.height_mean_mm,
        "height_median_mm": measurement.height_median_mm,
        "height_std_mm": measurement.height_std_mm,
        **stage_a.as_dict(),
        "error_to_true_20mm": measurement.height_mean_mm - TRUE_HEIGHT_MM,
        "absolute_error_mm": abs(measurement.height_mean_mm - TRUE_HEIGHT_MM),
        "length_mm": measurement.length_mm,
        "ground_baseline_zg_mm": measurement.ground_baseline_zg_mm,
        "ground_noise_sigma_mm": measurement.ground_noise_sigma_mm,
        "ground_profile_slope_z_per_mm": profile_slope,
        "ground_profile_intercept_mm": profile_intercept,
        "ground_profile_rmse_mm": profile_rmse,
        "height_line_fit_rmse_mm": measurement.height_fit.rmse_mm,
        "raw_top_z_mean_mm": raw_top_z_mean,
        "local_ground_z_mean_at_height_mm": local_ground_z_mean,
        "ground_reference_mode": measurement.ground_reference_mode,
    }


def measurement_detail(
    measurement: HeightLineMeasurement,
    stage_a: StageAHeightResult,
) -> dict[str, Any]:
    return {
        "ground_baseline_zg_mm": measurement.ground_baseline_zg_mm,
        "ground_noise_sigma_mm": measurement.ground_noise_sigma_mm,
        "height_mean_mm": measurement.height_mean_mm,
        "height_median_mm": measurement.height_median_mm,
        "height_std_mm": measurement.height_std_mm,
        **stage_a.as_dict(),
        "length_mm": measurement.length_mm,
        "endpoints_ground_mm": measurement.endpoints_ground.tolist(),
        "baseline_point_count": measurement.baseline_point_count,
        "baseline_inlier_count": measurement.baseline_inlier_count,
        "height_point_count": measurement.height_point_count,
        "height_inlier_count": measurement.height_inlier_count,
        "height_line_fit_rmse_mm": measurement.height_fit.rmse_mm,
        "ground_profile_rmse_mm": (
            None
            if measurement.ground_profile_fit is None
            else measurement.ground_profile_fit.rmse_mm
        ),
    }


def main() -> int:
    args = parse_args()
    config_path = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
    app = load_app_config(config_path)
    data_root = args.data_root.resolve()
    image_path = data_root / f"{FRAME_NAME}.png"
    metadata_path = data_root / f"{FRAME_NAME}.json"
    if not image_path.is_file():
        raise FileNotFoundError(f"expected online snapshot image: {image_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"expected online snapshot metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    offset = metadata["image_offset"]
    offset_x = int(offset["u"])
    offset_y = int(offset["v"])
    image = load_grayscale_image(image_path)

    old_model_path = TOOL_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
    new_model_path = TOOL_ROOT / "configs" / "calibration_daheng_0811" / "quadratic_graph.yaml"
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

    extraction_params = create_extraction_params(
        app.extraction_method, app.extraction_options
    )
    centers_local = extract_laser_center(
        image,
        extraction_params,
        image_offset=(offset_x, offset_y),
    )
    centers_full = centers_local.copy()
    if len(centers_full):
        centers_full[:, 0] += offset_x
        centers_full[:, 1] += offset_y

    old_result = reconstruct_uv_to_ground(
        centers_full, old_calibration, app.reconstruction
    )
    new_result = reconstruct_uv_to_ground(
        centers_full, new_calibration, app.reconstruction
    )
    old_map = result_map(old_result)
    new_map = result_map(new_result)
    common_keys = sorted(set(old_map) & set(new_map), key=lambda key: (key[1], key[0]))
    if len(common_keys) != old_result.point_count or len(common_keys) != new_result.point_count:
        raise RuntimeError("the two models did not retain the same valid pixel set")
    common_uv = np.asarray([old_map[key][0] for key in common_keys], dtype=np.float64)
    old_camera = np.asarray([old_map[key][1] for key in common_keys], dtype=np.float64)
    old_ground = np.asarray([old_map[key][2] for key in common_keys], dtype=np.float64)
    new_camera = np.asarray([new_map[key][1] for key in common_keys], dtype=np.float64)
    new_ground = np.asarray([new_map[key][2] for key in common_keys], dtype=np.float64)
    camera_delta = new_camera - old_camera
    ground_delta = new_ground - old_ground

    height_mask = (common_uv[:, 1] >= HEIGHT_V_RANGE[0]) & (
        common_uv[:, 1] <= HEIGHT_V_RANGE[1]
    )
    local_baseline_mask = np.zeros(len(common_uv), dtype=bool)
    for lo, hi in BASELINE_LOCAL_V_RANGES:
        local_baseline_mask |= (common_uv[:, 1] >= lo) & (common_uv[:, 1] <= hi)
    all_baseline_mask = ~height_mask
    if height_mask.sum() < app.measurement.min_height_points:
        raise RuntimeError("height ROI has too few points")
    if local_baseline_mask.sum() < app.measurement.min_baseline_points:
        raise RuntimeError("local baseline ROI has too few points")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    save_laser_centers_csv(output / "laser_centers_local.csv", centers_local)
    save_laser_centers_csv(output / "laser_centers_full.csv", centers_full)
    save_reconstructed_points_csv(
        output / "circular_cone_points.csv",
        old_result.pixels_uv,
        old_result.points_camera,
        old_result.points_ground,
    )
    save_reconstructed_points_csv(
        output / "quadratic_graph_points.csv",
        new_result.pixels_uv,
        new_result.points_camera,
        new_result.points_ground,
    )
    save_ground_pointcloud_ply(output / "circular_cone_ground.ply", old_result.points_ground)
    save_ground_pointcloud_ply(output / "quadratic_graph_ground.ply", new_result.points_ground)

    status_rows: list[dict[str, Any]] = []
    old_keys = set(old_map)
    new_keys = set(new_map)
    for uv_local, uv_full in zip(centers_local, centers_full):
        key = point_key(uv_full)
        status_rows.append(
            {
                "u_local_px": fmt(uv_local[0]),
                "v_local_px": fmt(uv_local[1]),
                "u_full_px": fmt(uv_full[0]),
                "v_full_px": fmt(uv_full[1]),
                "region": region_for_v(float(uv_full[1])),
                "circular_cone_valid": key in old_keys,
                "quadratic_graph_valid": key in new_keys,
            }
        )
    write_rows(
        output / "laser_center_status.csv",
        list(status_rows[0]),
        status_rows,
    )

    comparison_rows: list[dict[str, Any]] = []
    for uv, old_c, old_g, new_c, new_g in zip(
        common_uv, old_camera, old_ground, new_camera, new_ground
    ):
        delta_c = new_c - old_c
        delta_g = new_g - old_g
        comparison_rows.append(
            {
                "u_full_px": fmt(uv[0]),
                "v_full_px": fmt(uv[1]),
                "region": region_for_v(float(uv[1])),
                "circular_cone_Xc_mm": fmt(old_c[0]),
                "circular_cone_Yc_mm": fmt(old_c[1]),
                "circular_cone_Zc_mm": fmt(old_c[2]),
                "quadratic_graph_Xc_mm": fmt(new_c[0]),
                "quadratic_graph_Yc_mm": fmt(new_c[1]),
                "quadratic_graph_Zc_mm": fmt(new_c[2]),
                "delta_Xc_mm": fmt(delta_c[0]),
                "delta_Yc_mm": fmt(delta_c[1]),
                "delta_Zc_mm": fmt(delta_c[2]),
                "circular_cone_Xg_mm": fmt(old_g[0]),
                "circular_cone_Yg_mm": fmt(old_g[1]),
                "circular_cone_Zg_mm": fmt(old_g[2]),
                "quadratic_graph_Xg_mm": fmt(new_g[0]),
                "quadratic_graph_Yg_mm": fmt(new_g[1]),
                "quadratic_graph_Zg_mm": fmt(new_g[2]),
                "delta_Xg_mm": fmt(delta_g[0]),
                "delta_Yg_mm": fmt(delta_g[1]),
                "delta_Zg_mm": fmt(delta_g[2]),
                "delta_norm_mm": fmt(np.linalg.norm(delta_g)),
            }
        )
    write_rows(output / "coordinate_differences.csv", list(comparison_rows[0]), comparison_rows)

    region_masks = {
        "all_common_valid": np.ones(len(common_uv), dtype=bool),
        "height_roi": height_mask,
        "baseline_local": local_baseline_mask,
        "other": ~(height_mask | local_baseline_mask),
    }
    region_rows: list[dict[str, Any]] = []
    for region, mask in region_masks.items():
        delta = ground_delta[mask]
        region_rows.append(
            {
                "region": region,
                "point_count": int(mask.sum()),
                "delta_Xg_mean_signed_mm": metric(delta[:, 0])["mean_signed_mm"],
                "delta_Yg_mean_signed_mm": metric(delta[:, 1])["mean_signed_mm"],
                "delta_Zg_mean_signed_mm": metric(delta[:, 2])["mean_signed_mm"],
                "delta_ground_mae_mm": metric(np.linalg.norm(delta, axis=1))["mae_mm"],
                "delta_ground_rmse_mm": metric(np.linalg.norm(delta, axis=1))["rmse_mm"],
                "delta_ground_p95_abs_mm": metric(np.linalg.norm(delta, axis=1))["p95_abs_mm"],
                "delta_ground_max_abs_mm": metric(np.linalg.norm(delta, axis=1))["max_abs_mm"],
            }
        )
    write_rows(output / "region_coordinate_summary.csv", list(region_rows[0]), region_rows)

    measurements: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for model_name, model_ground in (
        ("circular_cone", old_ground),
        ("quadratic_graph", new_ground),
    ):
        details[model_name] = {}
        for baseline_name, baseline_mask in (
            ("local_adjacent", local_baseline_mask),
            ("all_non_height", all_baseline_mask),
        ):
            measurement = measure_height_line(
                model_ground[baseline_mask],
                model_ground[height_mask],
                app.measurement,
            )
            stage_a = resolve_stage_a_height_scale(
                measurement.height_mean_mm,
                system=app.system,
                correction=app.correction,
            )
            measurements.append(
                measurement_record(
                    model_name,
                    baseline_name,
                    measurement,
                    model_ground[baseline_mask],
                    model_ground[height_mask],
                    stage_a,
                )
            )
            details[model_name][baseline_name] = measurement_detail(
                measurement, stage_a
            )
        fixed_measurement = measure_height_line(
            None,
            model_ground[height_mask],
            app.measurement,
        )
        fixed_stage_a = resolve_stage_a_height_scale(
            fixed_measurement.height_mean_mm,
            system=app.system,
            correction=app.correction,
        )
        measurements.append(
            measurement_record(
                model_name,
                "fixed_zg_zero",
                fixed_measurement,
                None,
                model_ground[height_mask],
                fixed_stage_a,
            )
        )
        details[model_name]["fixed_zg_zero"] = measurement_detail(
            fixed_measurement, fixed_stage_a
        )
    write_rows(output / "height_measurements.csv", list(measurements[0]), measurements)
    (output / "height_measurements.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model_metrics = {
        "camera_coordinate_delta_quadratic_minus_cone": {
            axis: metric(camera_delta[:, index])
            for index, axis in enumerate(("Xc", "Yc", "Zc"))
        },
        "ground_coordinate_delta_quadratic_minus_cone": {
            axis: metric(ground_delta[:, index])
            for index, axis in enumerate(("Xg", "Yg", "Zg"))
        },
        "ground_coordinate_delta_norm": metric(np.linalg.norm(ground_delta, axis=1)),
    }
    (output / "coordinate_delta_summary.json").write_text(
        json.dumps(model_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame_name": FRAME_NAME,
        "true_height_mm": TRUE_HEIGHT_MM,
        "source_image": {
            "path": str(image_path.resolve()),
            "sha256": sha256(image_path),
            "format_found": image_path.suffix.lower(),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        },
        "source_metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": sha256(metadata_path),
            "content": metadata,
        },
        "artifact_provenance": {
            "reused": [
                str(image_path.resolve()),
                str(metadata_path.resolve()),
                str(config_path.resolve()),
                str(app.calibration.intrinsics.resolve()),
                str(app.calibration.extrinsics.resolve()),
            ],
            "not_reused": [
                "previous saved measurement folders: no matching frame_000667_measure exists",
                "previous saved laser-center CSV: not used",
            ],
            "newly_computed": [
                "Steger laser centers from the source image",
                "circular_cone reconstruction",
                "quadratic_graph reconstruction",
                "20 mm gauge-block height statistics",
            ],
        },
        "image_offset_full_px": {"u": offset_x, "v": offset_y},
        "extraction": {
            "method": app.extraction_method,
            "options": app.extraction_options,
            "center_count": int(len(centers_local)),
        },
        "reconstruction": {
            "params": {
                "parallel_epsilon": app.reconstruction.parallel_epsilon,
                "quadratic_epsilon": app.reconstruction.quadratic_epsilon,
                "min_camera_depth_mm": app.reconstruction.min_camera_depth_mm,
                "max_camera_depth_mm": app.reconstruction.max_camera_depth_mm,
                "model_range_margin_mm": app.reconstruction.model_range_margin_mm,
            },
            "circular_cone": {
                "path": str(old_model_path.resolve()),
                "sha256": sha256(old_model_path),
                "valid_count": old_result.point_count,
                "filtered": old_result.filtered,
            },
            "quadratic_graph": {
                "path": str(new_model_path.resolve()),
                "sha256": sha256(new_model_path),
                "valid_count": new_result.point_count,
                "filtered": new_result.filtered,
            },
        },
        "height_selection": {
            "height_v_inclusive": list(HEIGHT_V_RANGE),
            "local_baseline_v_inclusive": [list(item) for item in BASELINE_LOCAL_V_RANGES],
            "all_non_height_baseline": True,
            "fixed_zg_zero_baseline": True,
            "selection_note": "The visible vertical laser-line step was used as the height ROI; adjacent uninterrupted bands were used as the local baseline.",
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    local_measurements = {
        row["model"]: row
        for row in measurements
        if row["baseline_selection"] == "local_adjacent"
    }
    all_measurements = {
        row["model"]: row
        for row in measurements
        if row["baseline_selection"] == "all_non_height"
    }
    fixed_measurements = {
        row["model"]: row
        for row in measurements
        if row["baseline_selection"] == "fixed_zg_zero"
    }
    cone_local = local_measurements["circular_cone"]
    quad_local = local_measurements["quadratic_graph"]
    cone_all = all_measurements["circular_cone"]
    quad_all = all_measurements["quadratic_graph"]
    cone_fixed = fixed_measurements["circular_cone"]
    quad_fixed = fixed_measurements["quadratic_graph"]
    full_norm = model_metrics["ground_coordinate_delta_norm"]
    local_height_delta = quad_local["height_mean_mm"] - cone_local["height_mean_mm"]
    all_height_delta = quad_all["height_mean_mm"] - cone_all["height_mean_mm"]
    local_error_delta = quad_local["absolute_error_mm"] - cone_local["absolute_error_mm"]
    fixed_error_delta = quad_fixed["absolute_error_mm"] - cone_fixed["absolute_error_mm"]

    report = [
        "# 大恒 0811 单帧 20 mm 量块测试",
        "",
        "## 结论",
        "",
        f"本轮实际读取的是 `{image_path.name}`（PNG，不是 TIFF），尺寸 {image.shape[1]}×{image.shape[0]}，按元数据加回全幅偏移 `(u={offset_x}, v={offset_y})`。复用当前实时配置的 `{app.extraction_method}` 参数/API 和在线重建门限，分别只替换激光表面模型；这是对实时链路的离线回放，不包含相机 SDK 采集本身。",
        "",
        "在图像中可见的激光线横向阶梯段作为量块顶部，采用相邻两段连续地面线拟合局部基准：",
        f"- 高度 ROI：全幅 `v={HEIGHT_V_RANGE[0]}..{HEIGHT_V_RANGE[1]}`，{int(height_mask.sum())} 点；",
        f"- 局部基准：`v={BASELINE_LOCAL_V_RANGES[0][0]}..{BASELINE_LOCAL_V_RANGES[0][1]}` 和 `v={BASELINE_LOCAL_V_RANGES[1][0]}..{BASELINE_LOCAL_V_RANGES[1][1]}`，{int(local_baseline_mask.sum())} 点。",
        "",
        "| 模型 | 测得高度 mm | 相对 20 mm | 绝对误差 mm |",
        "|---|---:|---:|---:|",
        f"| circular_cone | {cone_local['height_mean_mm']:.6f} | {cone_local['error_to_true_20mm']:+.6f} | {cone_local['absolute_error_mm']:.6f} |",
        f"| quadratic_graph | {quad_local['height_mean_mm']:.6f} | {quad_local['error_to_true_20mm']:+.6f} | {quad_local['absolute_error_mm']:.6f} |",
        "",
        f"在这一个真实采集帧上，最新二次曲面得到 **{quad_local['height_mean_mm']:.6f} mm**，比旧圆锥的 **{cone_local['height_mean_mm']:.6f} mm** 低 {abs(local_height_delta):.6f} mm；绝对误差增加 {local_error_delta:.6f} mm（约 {local_error_delta * 1000:.1f} µm）。因此，这一帧上二次曲面没有改善，反而略差。",
        "",
        "## 为什么会出现“标定验证变好、实时实物变差”",
        "",
        f"1. 这不是激光点数量或求交覆盖率造成的：两套模型都从 {len(centers_local)} 个中心点中保留 {old_result.point_count} 个有效点，均过滤 {len(centers_local) - old_result.point_count} 个无交点。",
        f"2. 二次曲面在局部基准拟合残差上并未明显变差：本帧局部 ground-profile RMSE 为 cone {cone_local['ground_profile_rmse_mm']:.6f} mm、quadratic {quad_local['ground_profile_rmse_mm']:.6f} mm；高度线自身拟合 RMSE 分别为 {cone_local['height_line_fit_rmse_mm']:.6f} mm 和 {quad_local['height_line_fit_rmse_mm']:.6f} mm。",
        f"3. 但测高使用的是“顶部 Z − 局部基准 Z”。二次曲面的顶部原始 Z 均值约为 {quad_local['raw_top_z_mean_mm']:.6f} mm（圆锥 {cone_local['raw_top_z_mean_mm']:.6f} mm），同时它给出的顶部处局部基准均值为 {quad_local['local_ground_z_mean_at_height_mm']:.6f} mm（圆锥 {cone_local['local_ground_z_mean_at_height_mm']:.6f} mm），基准上移约 {quad_local['local_ground_z_mean_at_height_mm'] - cone_local['local_ground_z_mean_at_height_mm']:+.6f} mm，最终相对高度反而降低。",
        "4. 上一轮 0811 标定板验证评价的是已知棋盘格平面上的射线求交误差；本轮评价的是新图像中的激光中心提取、局部基准拟合和量块边缘段。两者误差来源不同，前者变好不能保证每个实物帧的相对高度都变好。",
        "5. 本轮只有一帧，且高度/基准区间是依据该帧的阶梯位置选出的；它足以解释你观察到的反例，但还不足以替换多帧、多位置的实物验收结论。",
        "",
        "## 基准选择敏感性",
        "",
        "为检查结论是否完全由局部 ROI 选择造成，同时把除高度 ROI 外的所有有效点都作为基准；这不是更推荐的测量协议，只是敏感性对照。",
        "",
        "| 模型 | 相邻局部基准 mm | 全部非高度点基准 mm |",
        "|---|---:|---:|",
        f"| circular_cone | {cone_local['height_mean_mm']:.6f} | {cone_all['height_mean_mm']:.6f} |",
        f"| quadratic_graph | {quad_local['height_mean_mm']:.6f} | {quad_all['height_mean_mm']:.6f} |",
        "",
        f"使用全部非高度点时，二次曲面仍比圆锥低 {abs(all_height_delta):.6f} mm；因此“二次曲面本帧略差”的方向不依赖于这两种基准范围，但数值会随基准选取变化。",
        "",
        "如果实时工具没有框选基准 ROI，代码会使用固定 `Zg=0`；同一高度 ROI 下的结果相反：",
        "",
        "| 模型 | 固定 Zg=0 测得高度 mm | 相对 20 mm | 绝对误差 mm |",
        "|---|---:|---:|---:|",
        f"| circular_cone | {cone_fixed['height_mean_mm']:.6f} | {cone_fixed['error_to_true_20mm']:+.6f} | {cone_fixed['absolute_error_mm']:.6f} |",
        f"| quadratic_graph | {quad_fixed['height_mean_mm']:.6f} | {quad_fixed['error_to_true_20mm']:+.6f} | {quad_fixed['absolute_error_mm']:.6f} |",
        "",
        f"固定 `Zg=0` 时二次曲面为 {quad_fixed['height_mean_mm']:.6f} mm，绝对误差比圆锥减少 {abs(fixed_error_delta):.6f} mm；所以你在实时工具上看到“新模型更差”，很可能对应的是选择了基准 ROI 的相对测高模式，或实时 ROI 选择与本轮不同。",
        "",
        "## 新旧三维坐标差异",
        "",
        f"对 {len(common_keys)} 个共同有效点，定义 Δ = quadratic_graph − circular_cone。三维位移的 MAE={full_norm['mae_mm']:.6f} mm，RMSE={full_norm['rmse_mm']:.6f} mm，P95={full_norm['p95_abs_mm']:.6f} mm，最大={full_norm['max_abs_mm']:.6f} mm。完整逐点差异见 `coordinate_differences.csv`。",
        "",
        "## 复用与新增计算",
        "",
        "- 复用：本次采集的 PNG、相邻 JSON 元数据、0811 内参/外参、实时测量配置，以及用户指定的两套模型文件。",
        "- 未复用：没有找到与 `frame_000667` 对应的旧 `*_measure` 结果，因此没有把旧离线激光中心或旧三维点当作输入。",
        "- 本轮新增：从原图重新提取 Steger 中心点；用两套模型分别求交；按同一高度/基准 ROI 计算 20 mm 测量结果；写出逐点坐标与统计产物。",
        "",
        "## 产物",
        "",
        "- `height_measurements.csv/json`：两套模型、两种基准范围的测高明细。",
        "- `coordinate_differences.csv`、`coordinate_delta_summary.json`：共同点的新旧相机系/地面系坐标及差值。",
        "- `circular_cone_points.csv/.ply`、`quadratic_graph_points.csv/.ply`：两套模型的完整有效重建点。",
        "- `laser_centers_local.csv`、`laser_centers_full.csv`、`laser_center_status.csv`：本轮重新提取的中心点。",
        "- `provenance.json`：输入、配置、模型哈希和复用/新增计算记录。",
    ]
    (output / "single_frame_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(output.resolve())
    print(f"centers={len(centers_local)} valid={len(common_keys)}")
    print(
        f"local_height_cone={cone_local['height_mean_mm']:.9f} "
        f"local_height_quadratic={quad_local['height_mean_mm']:.9f}"
    )
    print(
        f"all_height_cone={cone_all['height_mean_mm']:.9f} "
        f"all_height_quadratic={quad_all['height_mean_mm']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
