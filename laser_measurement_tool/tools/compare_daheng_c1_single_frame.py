"""Shadow-compare Frozen Daheng C1 with the quadratic C0 on one real frame.

The image is passed through the configured Steger extractor exactly once.
Both reconstruction branches then consume that same full-frame center array.
This tool is an integration check; the 20 mm height difference is reported but
does not decide the shadow verdict.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
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
from reconstruction.laser_ray_correction import (
    FrozenLaserRayCorrection,
    evaluate_frozen_laser_ray_correction,
)
from reconstruction.reconstructor import (
    ReconstructionParams,
    _intersect_laser_surface,
    reconstruct_uv_to_ground,
)
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
DEFAULT_OUTPUT = (
    TOOL_ROOT / "output_daheng_0811" / "shadow_frame_000667_c1_4k"
)
CLAMP_RATE_LIMIT = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def params_dict(params: ReconstructionParams) -> dict[str, Any]:
    return {
        item.name: json_safe(getattr(params, item.name))
        for item in fields(params)
    }


def file_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": sha256(resolved) if resolved.is_file() else None,
    }


def point_key(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def result_map(
    result: Any,
) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mapped: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for uv, camera, ground in zip(
        result.pixels_uv, result.points_camera, result.points_ground
    ):
        key = point_key(uv)
        if key in mapped:
            raise RuntimeError(f"duplicate reconstructed center key: {key}")
        mapped[key] = (uv, camera, ground)
    return mapped


def csv_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.15g}"


def metric(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "count": 0,
            "min_mm": None,
            "mean_mm": None,
            "p95_mm": None,
            "max_mm": None,
            "p95_abs_mm": None,
        }
    return {
        "count": int(len(array)),
        "min_mm": float(np.min(array)),
        "mean_mm": float(np.mean(array)),
        "p95_mm": float(np.quantile(array, 0.95)),
        "max_mm": float(np.max(array)),
        "p95_abs_mm": float(np.quantile(np.abs(array), 0.95)),
    }


def norm_metric(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return metric(np.empty(0, dtype=np.float64))
    return metric(np.linalg.norm(array.reshape(-1, array.shape[-1]), axis=1))


def region_for_v(v: float) -> str:
    if HEIGHT_V_RANGE[0] <= v <= HEIGHT_V_RANGE[1]:
        return "height_roi"
    if BASELINE_LOCAL_V_RANGES[0][0] <= v <= BASELINE_LOCAL_V_RANGES[0][1]:
        return "baseline_local_before"
    if BASELINE_LOCAL_V_RANGES[1][0] <= v <= BASELINE_LOCAL_V_RANGES[1][1]:
        return "baseline_local_after"
    return "other"


def filter_reason(
    key: tuple[float, float],
    valid_keys: set[tuple[float, float]],
    index: int,
    lambdas: np.ndarray,
    stable: np.ndarray,
    params: ReconstructionParams,
) -> str:
    if key in valid_keys:
        return "valid"
    if not bool(stable[index]):
        return "no_valid_intersection"
    value = float(lambdas[index])
    if not np.isfinite(value):
        return "no_valid_intersection"
    if value <= 0.0:
        return "negative_depth"
    if not params.min_camera_depth_mm <= value <= params.max_camera_depth_mm:
        return "outside_working_distance"
    return "non_finite_ground_or_post_filter"


def coordinates_row(
    prefix: str,
    camera: np.ndarray | None,
    ground: np.ndarray | None,
) -> dict[str, str]:
    if camera is None:
        camera_values = (None, None, None)
    else:
        camera_values = tuple(float(item) for item in camera)
    if ground is None:
        ground_values = (None, None, None)
    else:
        ground_values = tuple(float(item) for item in ground)
    return {
        f"{prefix}_Xc_mm": csv_float(camera_values[0]),
        f"{prefix}_Yc_mm": csv_float(camera_values[1]),
        f"{prefix}_Zc_mm": csv_float(camera_values[2]),
        f"{prefix}_Xg_mm": csv_float(ground_values[0]),
        f"{prefix}_Yg_mm": csv_float(ground_values[1]),
        f"{prefix}_Zg_mm": csv_float(ground_values[2]),
    }


def measurement_row(
    model_name: str,
    baseline_name: str,
    measurement: HeightLineMeasurement | None,
    error: Exception | None,
    stage_a: StageAHeightResult,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "baseline_selection": baseline_name,
        "status": "success" if measurement is not None else "failed",
        "error": "" if error is None else f"{type(error).__name__}: {error}",
        "height_mean_mm": None,
        "height_median_mm": None,
        "height_std_mm": None,
        **stage_a.as_dict(),
        "error_to_true_20mm": None,
        "absolute_error_mm": None,
        "length_mm": None,
        "ground_baseline_zg_mm": None,
        "ground_reference_mode": "",
        "ground_noise_sigma_mm": None,
        "ground_profile_rmse_mm": None,
        "height_line_fit_rmse_mm": None,
        "baseline_input_count": 0,
        "baseline_inlier_count": 0,
        "height_input_count": 0,
        "height_inlier_count": 0,
    }
    if measurement is None:
        return row
    row.update(
        {
            "height_mean_mm": measurement.height_mean_mm,
            "height_median_mm": measurement.height_median_mm,
            "height_std_mm": measurement.height_std_mm,
            "error_to_true_20mm": measurement.height_mean_mm - TRUE_HEIGHT_MM,
            "absolute_error_mm": abs(measurement.height_mean_mm - TRUE_HEIGHT_MM),
            "length_mm": measurement.length_mm,
            "ground_baseline_zg_mm": measurement.ground_baseline_zg_mm,
            "ground_reference_mode": measurement.ground_reference_mode,
            "ground_noise_sigma_mm": measurement.ground_noise_sigma_mm,
            "ground_profile_rmse_mm": (
                None
                if measurement.ground_profile_fit is None
                else measurement.ground_profile_fit.rmse_mm
            ),
            "height_line_fit_rmse_mm": measurement.height_fit.rmse_mm,
            "baseline_input_count": measurement.baseline_point_count,
            "baseline_inlier_count": measurement.baseline_inlier_count,
            "height_input_count": measurement.height_point_count,
            "height_inlier_count": measurement.height_inlier_count,
        }
    )
    return row


def measure_three_baselines(
    model_name: str,
    result: Any,
    measurement_params: Any,
    system: str,
    correction: Any,
) -> list[dict[str, Any]]:
    uv = result.pixels_uv
    ground = result.points_ground
    height_mask = (uv[:, 1] >= HEIGHT_V_RANGE[0]) & (
        uv[:, 1] <= HEIGHT_V_RANGE[1]
    )
    local_mask = np.zeros(len(uv), dtype=bool)
    for low, high in BASELINE_LOCAL_V_RANGES:
        local_mask |= (uv[:, 1] >= low) & (uv[:, 1] <= high)
    baselines: tuple[tuple[str, np.ndarray | None, np.ndarray], ...] = (
        ("local_adjacent", ground[local_mask], height_mask),
        ("all_non_height", ground[~height_mask], height_mask),
        ("fixed_zg_zero", None, height_mask),
    )
    rows: list[dict[str, Any]] = []
    for baseline_name, baseline, selected_height in baselines:
        try:
            measurement = measure_height_line(
                baseline,
                ground[selected_height],
                measurement_params,
            )
        except Exception as error:
            stage_a = resolve_stage_a_height_scale(
                None,
                system=system,
                correction=correction,
            )
            rows.append(
                measurement_row(model_name, baseline_name, None, error, stage_a)
            )
        else:
            stage_a = resolve_stage_a_height_scale(
                measurement.height_mean_mm,
                system=system,
                correction=correction,
            )
            rows.append(
                measurement_row(
                    model_name, baseline_name, measurement, None, stage_a
                )
            )
    return rows


def allocate_output_directory(base: Path) -> Path:
    candidate = base
    index = 1
    while candidate.exists():
        candidate = base.parent / f"{base.name}_{index:03d}"
        index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def report_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def report_metric(values: dict[str, Any]) -> str:
    return (
        f"count={values['count']}, min={report_number(values['min_mm'])} mm, "
        f"mean={report_number(values['mean_mm'])} mm, "
        f"P95={report_number(values['p95_mm'])} mm, "
        f"max={report_number(values['max_mm'])} mm"
    )


def build_report(
    summary: dict[str, Any],
    height_rows: list[dict[str, Any]],
    image_path: Path,
    metadata_path: Path,
) -> str:
    counts = summary["point_sets"]
    clamp = summary["c1_diagnostics"]["clamp"]
    delta = summary["c1_diagnostics"]["delta_lambda_mm"]
    conditions = summary["shadow_integration"]["conditions"]
    verdict = summary["shadow_integration"]["verdict"]
    lines = [
        "# Daheng C1_4k single-frame shadow report",
        "",
        f"## SHADOW_INTEGRATION: {verdict}",
        "",
        "本轮只验证集成链路；20 mm 单帧高度误差只作记录，不参与 PASS 判定。",
        "",
        f"- 输入图像：{image_path}",
        f"- 输入 metadata：{metadata_path}",
        "- Steger 中心提取：同一张图只执行一次，C0/C1 使用同一 centers_full。",
        "- C1 数学：复用 frozen JSON evaluator；s_used 是 clamp 后的 s_eval，不做 spline extrapolation。",
        "",
        "## Point-set alignment",
        "",
        "| C0 valid | C1 valid | common | C0-only | C1-only |",
        "|---:|---:|---:|---:|---:|",
        f"| {counts['c0_valid']} | {counts['c1_valid']} | {counts['common']} | "
        f"{counts['c0_only']} | {counts['c1_only']} |",
        "",
        f"- extracted centers: {counts['extracted_centers']}",
        f"- exact valid-set equality: {counts['valid_set_same']}",
        f"- valid-set difference explanation: {counts['difference_explained']}",
        "",
        "Difference reason pairs (C0 reason -> C1 reason):",
        "",
    ]
    reason_pairs = summary["point_sets"]["difference_reason_pairs"]
    if reason_pairs:
        for pair, count in sorted(reason_pairs.items()):
            lines.append(f"- {pair}: {count}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## C1 diagnostics",
            "",
            f"- clamp count/rate over all centers: {clamp['all_centers']['count']} / "
            f"{clamp['all_centers']['rate']:.6f}",
            f"- clamp count/rate over C1-valid centers: {clamp['c1_valid']['count']} / "
            f"{clamp['c1_valid']['rate']:.6f}",
            f"- clamp count/rate over common centers: {clamp['common']['count']} / "
            f"{clamp['common']['rate']:.6f}",
            f"- delta_lambda all centers: {report_metric(delta['all_centers'])}",
            f"- delta_lambda C1-valid centers: {report_metric(delta['c1_valid'])}",
            f"- delta_lambda common centers: {report_metric(delta['common'])}",
            "",
            "Coordinate delta statistics on common points (C1 - C0, mm):",
            "",
            f"- camera norm: {report_metric(summary['coordinate_delta']['camera_norm'])}",
            f"- ground norm: {report_metric(summary['coordinate_delta']['ground_norm'])}",
            f"- ground X: {report_metric(summary['coordinate_delta']['ground_axes']['Xg'])}",
            f"- ground Y: {report_metric(summary['coordinate_delta']['ground_axes']['Yg'])}",
            f"- ground Z: {report_metric(summary['coordinate_delta']['ground_axes']['Zg'])}",
            "",
            "## 20 mm height measurements",
            "",
            "| Model | Baseline | Status | Height mm | Error mm |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in height_rows:
        lines.append(
            f"| {row['model']} | {row['baseline_selection']} | {row['status']} | "
            f"{report_number(row['height_mean_mm'])} | "
            f"{report_number(row['error_to_true_20mm'])} |"
        )
    lines.extend(
        [
            "",
            "Height results are reported for local_adjacent, all_non_height, and "
            "fixed_zg_zero using each branch's own valid points. They are not a "
            "shadow PASS criterion.",
            "",
            "## Shadow conditions",
            "",
        ]
    )
    for name, value in conditions.items():
        lines.append(f"- {name}: {value}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for name in summary["artifacts"]:
        lines.append(f"- {name}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shadow-compare Daheng quadratic C0 and Frozen C1_4k."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "data" / "tif" / f"{FRAME_NAME}.png",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=REPO_ROOT / "data" / "tif" / f"{FRAME_NAME}.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    metadata_path = args.metadata.resolve()
    config_path = args.config.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"snapshot image not found: {image_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"snapshot metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    offset = metadata["image_offset"]
    offset_x = int(offset["u"])
    offset_y = int(offset["v"])
    image = load_grayscale_image(image_path)

    app = load_app_config(config_path)
    correction_path = app.calibration.laser_ray_correction
    if correction_path is None:
        raise RuntimeError("Daheng config does not declare calibration.laser_ray_correction")
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_plane,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        laser_ray_correction=correction_path,
    )
    correction = calibration.get("laser_ray_correction")
    if not isinstance(correction, FrozenLaserRayCorrection):
        raise RuntimeError("Daheng calibration did not load a valid Frozen C1 correction")

    extraction_params = create_extraction_params(
        app.extraction_method, app.extraction_options
    )
    centers_local = extract_laser_center(
        image,
        extraction_params,
        image_offset=(offset_x, offset_y),
    )
    centers_full = np.ascontiguousarray(
        centers_local + np.array([offset_x, offset_y], dtype=np.float64)
    )
    if len(centers_full) == 0:
        raise RuntimeError("Steger returned no laser centers")

    params_c0 = replace(
        app.reconstruction,
        enable_laser_ray_correction=False,
    )
    params_c1 = replace(
        app.reconstruction,
        enable_laser_ray_correction=True,
    )
    c0_result = reconstruct_uv_to_ground(centers_full, calibration, params_c0)
    c1_result = reconstruct_uv_to_ground(centers_full, calibration, params_c1)

    normalized = cv2.undistortPoints(
        centers_full.reshape(-1, 1, 2),
        calibration["K"],
        calibration["D"],
    ).reshape(-1, 2)
    rays = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    lambda_c0, stable, model_type = _intersect_laser_surface(
        rays, calibration, params_c0
    )
    if model_type != "quadratic_graph":
        raise RuntimeError(f"expected quadratic_graph C0 model, got {model_type}")
    evaluation = evaluate_frozen_laser_ray_correction(rays, correction)
    lambda_final = lambda_c0 + evaluation.correction_mm

    c0_map = result_map(c0_result)
    c1_map = result_map(c1_result)
    c0_keys = set(c0_map)
    c1_keys = set(c1_map)
    center_keys = [point_key(uv) for uv in centers_full]
    if len(set(center_keys)) != len(center_keys):
        raise RuntimeError("Steger returned duplicate pixel centers")
    if not c0_keys.issubset(set(center_keys)) or not c1_keys.issubset(set(center_keys)):
        raise RuntimeError("reconstruction returned a pixel not present in input centers")

    c0_reasons = [
        filter_reason(
            key, c0_keys, index, lambda_c0, stable, params_c0
        )
        for index, key in enumerate(center_keys)
    ]
    c1_reasons = [
        filter_reason(
            key, c1_keys, index, lambda_final, stable, params_c1
        )
        for index, key in enumerate(center_keys)
    ]
    common_keys = sorted(c0_keys & c1_keys, key=lambda key: (key[1], key[0]))
    c0_only_keys = c0_keys - c1_keys
    c1_only_keys = c1_keys - c0_keys
    difference_reason_pairs = Counter(
        (c0_reasons[index], c1_reasons[index])
        for index, key in enumerate(center_keys)
        if key in c0_only_keys or key in c1_only_keys
    )
    common_c0_camera = np.asarray(
        [c0_map[key][1] for key in common_keys], dtype=np.float64
    ).reshape(-1, 3)
    common_c1_camera = np.asarray(
        [c1_map[key][1] for key in common_keys], dtype=np.float64
    ).reshape(-1, 3)
    common_c0_ground = np.asarray(
        [c0_map[key][2] for key in common_keys], dtype=np.float64
    ).reshape(-1, 3)
    common_c1_ground = np.asarray(
        [c1_map[key][2] for key in common_keys], dtype=np.float64
    ).reshape(-1, 3)
    camera_delta = common_c1_camera - common_c0_camera
    ground_delta = common_c1_ground - common_c0_ground

    c0_valid_mask = np.asarray([key in c0_keys for key in center_keys])
    c1_valid_mask = np.asarray([key in c1_keys for key in center_keys])
    common_mask = c0_valid_mask & c1_valid_mask
    clamp_mask = evaluation.clamped
    delta_lambda = evaluation.correction_mm
    clamp_summary = {}
    for label, mask in (
        ("all_centers", np.ones(len(center_keys), dtype=bool)),
        ("c1_valid", c1_valid_mask),
        ("common", common_mask),
    ):
        selected = clamp_mask[mask]
        denominator = int(mask.sum())
        clamp_summary[label] = {
            "count": int(np.count_nonzero(selected)),
            "total": denominator,
            "rate": (
                float(np.count_nonzero(selected) / denominator)
                if denominator
                else 0.0
            ),
        }
    delta_summary = {
        label: metric(delta_lambda[mask])
        for label, mask in (
            ("all_centers", np.ones(len(center_keys), dtype=bool)),
            ("c1_valid", c1_valid_mask),
            ("common", common_mask),
        )
    }

    diagnostics_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for index, (uv, ray, key) in enumerate(
        zip(centers_full, rays, center_keys)
    ):
        c0_item = c0_map.get(key)
        c1_item = c1_map.get(key)
        c0_valid = c0_item is not None
        c1_valid = c1_item is not None
        status = (
            "common"
            if c0_valid and c1_valid
            else "c0_only"
            if c0_valid
            else "c1_only"
            if c1_valid
            else "invalid_both"
        )
        diagnostics_rows.append(
            {
                "u": csv_float(uv[0]),
                "v": csv_float(uv[1]),
                "xn": csv_float(ray[0]),
                "yn": csv_float(ray[1]),
                "s_raw": csv_float(evaluation.s_raw[index]),
                "s_used": csv_float(evaluation.s_eval[index]),
                "c1_clamped": bool(evaluation.clamped[index]),
                "lambda_c0": csv_float(lambda_c0[index]),
                "delta_lambda": csv_float(delta_lambda[index]),
                "lambda_final": csv_float(lambda_final[index]),
                "c0_valid": c0_valid,
                "c1_valid": c1_valid,
                "status": status,
                "c0_filter_reason": c0_reasons[index],
                "c1_filter_reason": c1_reasons[index],
                "c0_stable": bool(stable[index]),
            }
        )
        comparison_row: dict[str, Any] = {
            "u": csv_float(uv[0]),
            "v": csv_float(uv[1]),
            "region": region_for_v(float(uv[1])),
            "status": status,
            "c0_valid": c0_valid,
            "c1_valid": c1_valid,
            "c0_filter_reason": c0_reasons[index],
            "c1_filter_reason": c1_reasons[index],
        }
        comparison_row.update(
            coordinates_row(
                "c0",
                None if c0_item is None else c0_item[1],
                None if c0_item is None else c0_item[2],
            )
        )
        comparison_row.update(
            coordinates_row(
                "c1",
                None if c1_item is None else c1_item[1],
                None if c1_item is None else c1_item[2],
            )
        )
        if c0_item is not None and c1_item is not None:
            comparison_row.update(
                {
                    "delta_Xc_mm": csv_float(c1_item[1][0] - c0_item[1][0]),
                    "delta_Yc_mm": csv_float(c1_item[1][1] - c0_item[1][1]),
                    "delta_Zc_mm": csv_float(c1_item[1][2] - c0_item[1][2]),
                    "delta_Xg_mm": csv_float(c1_item[2][0] - c0_item[2][0]),
                    "delta_Yg_mm": csv_float(c1_item[2][1] - c0_item[2][1]),
                    "delta_Zg_mm": csv_float(c1_item[2][2] - c0_item[2][2]),
                    "delta_camera_norm_mm": csv_float(
                        np.linalg.norm(c1_item[1] - c0_item[1])
                    ),
                    "delta_ground_norm_mm": csv_float(
                        np.linalg.norm(c1_item[2] - c0_item[2])
                    ),
                }
            )
        else:
            comparison_row.update(
                {
                    "delta_Xc_mm": "",
                    "delta_Yc_mm": "",
                    "delta_Zc_mm": "",
                    "delta_Xg_mm": "",
                    "delta_Yg_mm": "",
                    "delta_Zg_mm": "",
                    "delta_camera_norm_mm": "",
                    "delta_ground_norm_mm": "",
                }
            )
        comparison_rows.append(comparison_row)

    height_rows = (
        measure_three_baselines(
            "c0", c0_result, app.measurement, app.system, app.correction
        )
        + measure_three_baselines(
            "c1", c1_result, app.measurement, app.system, app.correction
        )
    )
    height_success = all(row["status"] == "success" for row in height_rows)

    correction_active = bool(
        len(delta_lambda[c1_valid_mask])
        and np.any(np.abs(delta_lambda[c1_valid_mask]) > 1.0e-12)
    )
    valid_coordinate_finite = bool(
        np.isfinite(c0_result.points_camera).all()
        and np.isfinite(c0_result.points_ground).all()
        and np.isfinite(c1_result.points_camera).all()
        and np.isfinite(c1_result.points_ground).all()
    )
    clamp_rate = clamp_summary["c1_valid"]["rate"]
    clamp_normal = bool(clamp_rate <= CLAMP_RATE_LIMIT)
    difference_explained = all(
        c0_reasons[index] != "unknown" and c1_reasons[index] != "unknown"
        for index, key in enumerate(center_keys)
        if key in c0_only_keys or key in c1_only_keys
    )
    conditions = {
        "same_centers_used_for_c0_and_c1": True,
        "c0_has_valid_points": bool(len(c0_result.pixels_uv) > 0),
        "c1_has_valid_points": bool(len(c1_result.pixels_uv) > 0),
        "common_points_present": bool(len(common_keys) > 0),
        "valid_coordinates_are_finite": valid_coordinate_finite,
        "c1_correction_actually_nonzero": correction_active,
        "clamp_rate_not_abnormal_le_5_percent": clamp_normal,
        "valid_set_same_or_difference_explained": difference_explained,
        "all_six_height_measurements_succeeded": height_success,
    }
    verdict = "PASS" if all(conditions.values()) else "FAIL"

    coordinate_summary = {
        "common_count": len(common_keys),
        "camera_norm": norm_metric(camera_delta),
        "ground_norm": norm_metric(ground_delta),
        "ground_axes": {
            axis: metric(ground_delta[:, index])
            for index, axis in enumerate(("Xg", "Yg", "Zg"))
        },
    }
    summary: dict[str, Any] = {
        "frame_name": FRAME_NAME,
        "extracted_centers": int(len(centers_full)),
        "point_sets": {
            "extracted_centers": int(len(centers_full)),
            "c0_valid": int(len(c0_keys)),
            "c1_valid": int(len(c1_keys)),
            "common": int(len(common_keys)),
            "c0_only": int(len(c0_only_keys)),
            "c1_only": int(len(c1_only_keys)),
            "invalid_both": int(len(centers_full) - len(c0_keys | c1_keys)),
            "valid_set_same": bool(c0_keys == c1_keys),
            "difference_explained": difference_explained,
            "difference_reason_pairs": {
                f"{left} -> {right}": int(count)
                for (left, right), count in difference_reason_pairs.items()
            },
        },
        "c0_filtered": json_safe(c0_result.filtered),
        "c1_filtered": json_safe(c1_result.filtered),
        "c1_diagnostics": {
            "clamp": clamp_summary,
            "delta_lambda_mm": delta_summary,
            "domain_policy": "s_used=s_eval=clip(s_raw, domain_min, domain_max); no spline extrapolation",
        },
        "coordinate_delta": coordinate_summary,
        "finite_checks": {
            "c0_camera": bool(np.isfinite(c0_result.points_camera).all()),
            "c0_ground": bool(np.isfinite(c0_result.points_ground).all()),
            "c1_camera": bool(np.isfinite(c1_result.points_camera).all()),
            "c1_ground": bool(np.isfinite(c1_result.points_ground).all()),
        },
        "height_measurement_success": height_success,
        "shadow_integration": {
            "verdict": verdict,
            "conditions": conditions,
            "clamp_rate_limit": CLAMP_RATE_LIMIT,
            "height_is_not_a_pass_condition": True,
        },
        "artifacts": [
            "c1_diagnostics.csv",
            "point_set_summary.json",
            "height_measurements.csv",
            "c0_points.csv",
            "c1_points.csv",
            "c0_ground.ply",
            "c1_ground.ply",
            "pointwise_comparison.csv",
            "laser_centers_local.csv",
            "laser_centers_full.csv",
            "provenance.json",
            "shadow_report.md",
        ],
    }

    output = allocate_output_directory(args.output.resolve())
    save_laser_centers_csv(output / "laser_centers_local.csv", centers_local)
    save_laser_centers_csv(output / "laser_centers_full.csv", centers_full)
    save_reconstructed_points_csv(
        output / "c0_points.csv",
        c0_result.pixels_uv,
        c0_result.points_camera,
        c0_result.points_ground,
    )
    save_reconstructed_points_csv(
        output / "c1_points.csv",
        c1_result.pixels_uv,
        c1_result.points_camera,
        c1_result.points_ground,
    )
    save_ground_pointcloud_ply(output / "c0_ground.ply", c0_result.points_ground)
    save_ground_pointcloud_ply(output / "c1_ground.ply", c1_result.points_ground)
    write_rows(
        output / "c1_diagnostics.csv",
        list(diagnostics_rows[0]),
        diagnostics_rows,
    )
    write_rows(
        output / "pointwise_comparison.csv",
        list(comparison_rows[0]),
        comparison_rows,
    )
    write_rows(
        output / "height_measurements.csv",
        list(height_rows[0]),
        height_rows,
    )

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame_name": FRAME_NAME,
        "source_image": {
            **(file_info(image_path) or {}),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        },
        "source_metadata": {
            **(file_info(metadata_path) or {}),
            "content": json_safe(metadata),
        },
        "config": file_info(config_path),
        "calibration": {
            "intrinsics": file_info(app.calibration.intrinsics),
            "quadratic_graph_c0": file_info(app.calibration.laser_plane),
            "extrinsics": file_info(app.calibration.extrinsics),
            "ground_u_compensation": file_info(
                app.calibration.ground_u_compensation
            ),
            "frozen_c1": file_info(correction_path),
        },
        "artifact_provenance": {
            "reused": [
                str(image_path),
                str(metadata_path),
                str(config_path),
                str(app.calibration.intrinsics),
                str(app.calibration.laser_plane),
                str(app.calibration.extrinsics),
                str(correction_path),
            ],
            "not_reused": [
                "previous laser-center result: Steger was run once for this shadow output",
                "previous C0/C1 point clouds and height measurements",
            ],
            "newly_computed": [
                "one Steger extraction shared by C0 and C1",
                "C0 reconstruction with enable_laser_ray_correction=false",
                "C1 reconstruction with enable_laser_ray_correction=true",
                "C1 diagnostics and C0/C1 pointwise comparison",
                "three baseline height measurements for each branch",
            ],
        },
        "image_offset_full_px": {"u": offset_x, "v": offset_y},
        "extraction": {
            "method": app.extraction_method,
            "options": json_safe(app.extraction_options),
            "center_extraction_runs": 1,
            "center_count": int(len(centers_full)),
            "same_centers_passed_to_c0_and_c1": True,
        },
        "reconstruction": {
            "model_type": model_type,
            "c0_params": params_dict(params_c0),
            "c1_params": params_dict(params_c1),
            "c0_valid_count": int(len(c0_result.pixels_uv)),
            "c1_valid_count": int(len(c1_result.pixels_uv)),
            "c0_filtered": json_safe(c0_result.filtered),
            "c1_filtered": json_safe(c1_result.filtered),
            "c1_model_id": correction.model_id,
            "c1_source_path": correction.source_path,
            "c1_formula": "lambda_final = lambda_c0 + F(s_used)",
        },
        "height_selection": {
            "true_height_mm": TRUE_HEIGHT_MM,
            "height_v_inclusive": list(HEIGHT_V_RANGE),
            "local_baseline_v_inclusive": [
                list(item) for item in BASELINE_LOCAL_V_RANGES
            ],
            "baseline_modes": [
                "local_adjacent",
                "all_non_height",
                "fixed_zg_zero",
            ],
            "height_is_not_shadow_pass_condition": True,
        },
        "output_directory": str(output.resolve()),
    }
    write_json(output / "point_set_summary.json", summary)
    write_json(output / "provenance.json", provenance)
    (output / "shadow_report.md").write_text(
        build_report(summary, height_rows, image_path, metadata_path),
        encoding="utf-8",
    )

    print(f"SHADOW_INTEGRATION: {verdict}")
    print(f"output: {output}")
    print(
        "points: "
        f"C0={len(c0_keys)} C1={len(c1_keys)} common={len(common_keys)} "
        f"C0-only={len(c0_only_keys)} C1-only={len(c1_only_keys)}"
    )
    print(
        "clamp: "
        f"{clamp_summary['c1_valid']['count']}/{clamp_summary['c1_valid']['total']} "
        f"({clamp_summary['c1_valid']['rate']:.6f}) on C1-valid centers"
    )
    print("height measurements: " + ("success" if height_success else "failure"))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
