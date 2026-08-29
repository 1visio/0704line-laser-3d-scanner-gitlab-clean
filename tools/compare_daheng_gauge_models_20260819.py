"""Compare Circular Cone and Quadratic Graph C0 on the 2026-08-19 gauge data.

This is deliberately a C0-only, same-protocol replay.  It reuses the exact
Steger center cache and manually frozen geometry-only ROI registry from the
0819 acceptance run, then replaces only the laser surface model.  Frozen C1,
H1, H-B2, and any result-driven ROI selection are excluded from this comparison.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_ROOT.parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

import evaluate_daheng_c1_gauge_blocks as gauge
from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from reconstruction.reconstructor import reconstruct_uv_to_ground


DATA_ROOT = REPO_ROOT.parent / "calibration_tool" / "projects" / "daheng" / "data"
CACHE_ROOT = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_manual_frozen_v2"
DEFAULT_CONFIG = REPO_ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_ROI = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819" / "roi_registry_manual.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_model_ab_c0_v1"

DATASETS = gauge.DATASETS
TRUTH_MM = gauge.TRUTH_MM
MODE_NAMES = gauge.MODE_NAMES
MODEL_SPECS = (
    ("CONE", "circular_cone", "circular_cone.yaml"),
    ("QUADRATIC", "quadratic_graph", "quadratic_graph.yaml"),
)
VARIANTS = ("native_valid", "common_valid")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if any(str(value or "").strip() for value in row.values())
        ]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def file_info(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": sha256(resolved) if resolved.is_file() else None,
    }


def load_cached_entries(
    data_root: Path,
    centers_csv: Path,
    frame_metrics_csv: Path,
    input_audit_csv: Path,
    roi_registry_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load and audit the exact centers/ROI used by manual_frozen_v2."""

    frame_rows = read_csv_rows(frame_metrics_csv)
    frame_rows = [row for row in frame_rows if row.get("filename")]
    if len(frame_rows) != 150:
        raise RuntimeError(f"expected 150 cached frame rows, got {len(frame_rows)}")

    audit_rows = read_csv_rows(input_audit_csv)
    audit_by_key = {
        (row.get("dataset", ""), row.get("filename", "")): row
        for row in audit_rows
        if row.get("filename") and row.get("filename") != "__dataset_summary__"
    }

    point_rows = read_csv_rows(centers_csv)
    points_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in point_rows:
        key = (
            row.get("dataset", ""),
            row.get("pose_id", ""),
            row.get("repeat_index", ""),
            row.get("filename", ""),
        )
        points_by_key[key].append(row)

    entries: list[dict[str, Any]] = []
    raw_audit_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for frame in sorted(
        frame_rows,
        key=lambda row: (
            gauge.DATASET_ORDER[row["dataset"]],
            int(row["position_rank"]),
            int(row["repeat_index"]),
        ),
    ):
        dataset = frame["dataset"]
        pose_id = frame["pose_id"]
        repeat_index = frame["repeat_index"]
        filename = frame["filename"]
        key = (dataset, pose_id, repeat_index, filename)
        if key in seen_keys:
            raise RuntimeError(f"duplicate cached frame key: {key}")
        seen_keys.add(key)

        cached_points = sorted(
            points_by_key.get(key, []), key=lambda row: int(row["point_index"])
        )
        expected_count = int(frame["center_count"])
        if len(cached_points) != expected_count:
            raise RuntimeError(
                f"cached center count mismatch for {key}: "
                f"{len(cached_points)} != {expected_count}"
            )
        point_indices = [int(row["point_index"]) for row in cached_points]
        if point_indices != list(range(expected_count)):
            raise RuntimeError(f"cached point indices are not contiguous for {key}")
        centers = np.asarray(
            [[float(row["u_px"]), float(row["v_px"])] for row in cached_points],
            dtype=np.float64,
        )
        if len(np.unique(centers, axis=0)) != len(centers):
            raise RuntimeError(f"duplicate cached center for {key}")

        dataset_folder = data_root / dataset
        raw_candidates = sorted(
            path for path in dataset_folder.rglob("*.tif") if path.name == filename
        )
        if len(raw_candidates) != 1:
            raise FileNotFoundError(
                f"expected one raw TIFF named {filename!r} under {dataset_folder}, "
                f"found {len(raw_candidates)}"
            )
        raw_path = raw_candidates[0]
        actual_sha = sha256(raw_path)
        expected_sha = frame.get("sha256") or audit_by_key.get(key[:1] + (filename,), {}).get(
            "actual_sha256", ""
        )
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(
                f"raw TIFF SHA mismatch for {key}: {actual_sha} != {expected_sha}"
            )
        audit = audit_by_key.get((dataset, filename), {})
        if audit.get("actual_sha256") and audit["actual_sha256"] != actual_sha:
            raise RuntimeError(f"input audit SHA mismatch for {key}")

        actual_height = int(float(audit.get("actual_height") or 3000))
        actual_width = int(float(audit.get("actual_width") or 4096))
        entries.append(
            {
                "dataset": dataset,
                "height_truth_mm": TRUTH_MM[dataset],
                "path": raw_path,
                "pose_id": pose_id,
                "repeat_index": int(repeat_index),
                "image_shape": (actual_height, actual_width),
                "image_dtype": audit.get("actual_dtype", "uint8"),
                "image_offset_x": int(float(audit.get("csv_offset_x") or 0)),
                "image_offset_y": int(float(audit.get("csv_offset_y") or 0)),
                "centers": centers,
                "center_count": len(centers),
                "sha256": actual_sha,
                "position_rank": int(frame["position_rank"]),
            }
        )
        raw_audit_rows.append(
            {
                "dataset": dataset,
                "pose_id": pose_id,
                "repeat_index": repeat_index,
                "filename": filename,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "sha256_match": bool(not expected_sha or actual_sha == expected_sha),
                "center_count_cached": len(centers),
                "raw_width": actual_width,
                "raw_height": actual_height,
                "offset_x": int(float(audit.get("csv_offset_x") or 0)),
                "offset_y": int(float(audit.get("csv_offset_y") or 0)),
            }
        )

    if len(entries) != 150:
        raise RuntimeError(f"expected 150 cached entries, got {len(entries)}")
    if len(points_by_key) != 150:
        raise RuntimeError(f"expected 150 cached center groups, got {len(points_by_key)}")

    registry, registry_summary = gauge.load_frozen_roi_registry(roi_registry_path, entries)
    audit_summary = {
        "data_root": str(data_root.resolve()),
        "raw_image_count": len(entries),
        "raw_sha256_all_match": all(row["sha256_match"] for row in raw_audit_rows),
        "cached_center_image_count": len(entries),
        "cached_center_point_count": int(sum(len(item["centers"]) for item in entries)),
        "steger_centers_reused": True,
        "steger_reextracted_this_run": False,
        "center_cache": file_info(centers_csv),
        "source_frame_metrics": file_info(frame_metrics_csv),
        "source_input_audit": file_info(input_audit_csv),
    }
    return entries, registry_summary, {"audit_summary": audit_summary, "raw_rows": raw_audit_rows}


def model_calibration(app: Any, model_path: Path, expected_type: str) -> dict[str, Any]:
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        model_path,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        laser_ray_correction=None,
        ground_u_optional=True,
    )
    model_type = calibration["laser_model"]["model_type"]
    if model_type != expected_type:
        raise RuntimeError(
            f"{model_path} has model_type={model_type!r}, expected {expected_type!r}"
        )
    return calibration


def metric(values: Iterable[Any], limit_mm: float = 0.2) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=np.float64,
    )
    if not len(array):
        return {
            "count": 0,
            "bias_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_mm": None,
            "max_mm": None,
            "pass_count": 0,
            "pass_rate": None,
            "limit_mm": limit_mm,
        }
    absolute = np.abs(array)
    return {
        "count": int(len(array)),
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array**2))),
        "p95_mm": float(np.quantile(absolute, 0.95)),
        "max_mm": float(np.max(absolute)),
        "pass_count": int(np.sum(absolute <= limit_mm)),
        "pass_rate": float(np.mean(absolute <= limit_mm)),
        "limit_mm": limit_mm,
    }


def distribution_metric(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return metric(array, limit_mm=0.0)


def arrays_for_keys(
    centers: np.ndarray,
    result_map: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    accepted_keys: set[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    uv_rows: list[np.ndarray] = []
    ground_rows: list[np.ndarray] = []
    for center in centers:
        key = gauge.point_key(center)
        if key in accepted_keys:
            camera, ground = result_map[key]
            uv_rows.append(center)
            ground_rows.append(ground)
    return (
        np.asarray(uv_rows, dtype=np.float64).reshape(-1, 2)
        if uv_rows
        else np.empty((0, 2), dtype=np.float64),
        np.asarray(ground_rows, dtype=np.float64).reshape(-1, 3)
        if ground_rows
        else np.empty((0, 3), dtype=np.float64),
    )


def reconstruct_and_measure(
    entries: list[dict[str, Any]],
    app: Any,
    calibrations: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    params = replace(app.reconstruction, enable_laser_ray_correction=False)
    height_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []

    for entry in entries:
        model_results: dict[str, dict[str, Any]] = {}
        for label, model_type, _ in MODEL_SPECS:
            result = reconstruct_uv_to_ground(entry["centers"], calibrations[label], params)
            mapped = gauge.result_map(result)
            model_results[label] = {"result": result, "map": mapped}

        valid_keys = {label: set(value["map"]) for label, value in model_results.items()}
        common_keys = valid_keys["CONE"] & valid_keys["QUADRATIC"]
        cone_uv, cone_ground = arrays_for_keys(
            entry["centers"], model_results["CONE"]["map"], valid_keys["CONE"]
        )
        quadratic_uv, quadratic_ground = arrays_for_keys(
            entry["centers"], model_results["QUADRATIC"]["map"], valid_keys["QUADRATIC"]
        )
        common_uv, cone_common_ground = arrays_for_keys(
            entry["centers"], model_results["CONE"]["map"], common_keys
        )
        _, quadratic_common_ground = arrays_for_keys(
            entry["centers"], model_results["QUADRATIC"]["map"], common_keys
        )
        coordinate_delta = quadratic_common_ground - cone_common_ground
        coordinate_rows.append(
            {
                "dataset": entry["dataset"],
                "pose_id": entry["pose_id"],
                "repeat_index": entry["repeat_index"],
                "position_rank": entry["position_rank"],
                "filename": entry["path"].name,
                "center_count": len(entry["centers"]),
                "cone_valid_points": len(valid_keys["CONE"]),
                "quadratic_valid_points": len(valid_keys["QUADRATIC"]),
                "common_valid_points": len(common_keys),
                "quadratic_minus_cone_mean_dx_mm": float(np.mean(coordinate_delta[:, 0]))
                if len(coordinate_delta)
                else None,
                "quadratic_minus_cone_mean_dy_mm": float(np.mean(coordinate_delta[:, 1]))
                if len(coordinate_delta)
                else None,
                "quadratic_minus_cone_mean_dz_mm": float(np.mean(coordinate_delta[:, 2]))
                if len(coordinate_delta)
                else None,
                "quadratic_minus_cone_norm_rmse_mm": float(
                    np.sqrt(np.mean(np.sum(coordinate_delta**2, axis=1)))
                )
                if len(coordinate_delta)
                else None,
            }
        )
        frame_rows.append(
            {
                "dataset": entry["dataset"],
                "truth_mm": entry["height_truth_mm"],
                "pose_id": entry["pose_id"],
                "position_rank": entry["position_rank"],
                "repeat_index": entry["repeat_index"],
                "filename": entry["path"].name,
                "center_count": len(entry["centers"]),
                "cone_valid_points": len(valid_keys["CONE"]),
                "quadratic_valid_points": len(valid_keys["QUADRATIC"]),
                "common_valid_points": len(common_keys),
                "cone_filtered_points": len(entry["centers"]) - len(valid_keys["CONE"]),
                "quadratic_filtered_points": len(entry["centers"]) - len(valid_keys["QUADRATIC"]),
            }
        )

        for label in ("CONE", "QUADRATIC"):
            model_map = model_results[label]["map"]
            model_keys = valid_keys[label]
            for variant, accepted_keys in (
                ("native_valid", model_keys),
                ("common_valid", common_keys),
            ):
                uv_array, ground_array = arrays_for_keys(
                    entry["centers"], model_map, accepted_keys
                )
                for mode in MODE_NAMES:
                    try:
                        (
                            mean_mm,
                            median_mm,
                            std_mm,
                            height_count,
                            baseline_count,
                            status,
                        ) = gauge.measure_mode(
                            ground_array,
                            uv_array,
                            entry["roi"],
                            mode,
                            app.measurement,
                        )
                        error_text = ""
                    except Exception as error:
                        mean_mm = median_mm = std_mm = None
                        height_count = baseline_count = 0
                        status = "failed"
                        error_text = f"{type(error).__name__}: {error}"
                    height_rows.append(
                        {
                            "dataset": entry["dataset"],
                            "truth_mm": entry["height_truth_mm"],
                            "pose_id": entry["pose_id"],
                            "position_rank": entry["position_rank"],
                            "v_center_px": entry["roi"]["v_center_px"],
                            "repeat_index": entry["repeat_index"],
                            "filename": entry["path"].name,
                            "model": label,
                            "model_type": dict((a, b) for a, b, _ in MODEL_SPECS)[label],
                            "variant": variant,
                            "mode": mode,
                            "height_mean_mm": mean_mm,
                            "height_median_mm": median_mm,
                            "height_std_mm": std_mm,
                            "signed_error_mm": (
                                None if mean_mm is None else mean_mm - entry["height_truth_mm"]
                            ),
                            "abs_error_mm": (
                                None
                                if mean_mm is None
                                else abs(mean_mm - entry["height_truth_mm"])
                            ),
                            "valid_point_count": len(accepted_keys),
                            "height_point_count": height_count,
                            "baseline_point_count": baseline_count,
                            "status": status,
                            "error": error_text,
                            "roi_geometry_only": True,
                            "c1_enabled": False,
                            "stage_a_enabled": False,
                        }
                    )
    return frame_rows, height_rows, coordinate_rows


def condition_rows_from_frames(height_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in height_rows:
        if row["status"] == "success" and row["height_mean_mm"] is not None:
            groups[
                (
                    row["dataset"],
                    row["position_rank"],
                    row["model"],
                    row["variant"],
                    row["mode"],
                )
            ].append(row)
    conditions: list[dict[str, Any]] = []
    for key, rows in sorted(
        groups.items(),
        key=lambda item: (
            gauge.DATASET_ORDER[item[0][0]],
            int(item[0][1]),
            item[0][2],
            item[0][3],
            MODE_NAMES.index(item[0][4]),
        ),
    ):
        dataset, position_rank, model, variant, mode = key
        measurements = np.asarray([row["height_mean_mm"] for row in rows], dtype=np.float64)
        truth = float(rows[0]["truth_mm"])
        mean_mm = float(np.mean(measurements))
        conditions.append(
            {
                "dataset": dataset,
                "truth_mm": truth,
                "position_rank": int(position_rank),
                "model": model,
                "variant": variant,
                "mode": mode,
                "repeat_count": len(measurements),
                "measured_mean_mm": mean_mm,
                "measured_median_mm": float(np.median(measurements)),
                "repeatability_sigma_mm": (
                    float(np.std(measurements, ddof=1)) if len(measurements) >= 2 else None
                ),
                "signed_error_mm": mean_mm - truth,
                "abs_error_mm": abs(mean_mm - truth),
                "pass_0p2mm": bool(abs(mean_mm - truth) <= 0.2),
            }
        )
    return conditions


def missing_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present = {
        (row["dataset"], row["position_rank"], row["model"], row["variant"], row["mode"])
        for row in conditions
    }
    missing: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for position_rank in range(1, 6):
            for model in ("CONE", "QUADRATIC"):
                for variant in VARIANTS:
                    for mode in MODE_NAMES:
                        key = (dataset, position_rank, model, variant, mode)
                        if key not in present:
                            missing.append(
                                {
                                    "dataset": dataset,
                                    "position_rank": position_rank,
                                    "model": model,
                                    "variant": variant,
                                    "mode": mode,
                                }
                            )
    return missing


def add_stats(
    stats: list[dict[str, Any]],
    layer: str,
    scope: str,
    model: str,
    variant: str,
    mode: str,
    values: Iterable[Any],
    expected_count: int,
) -> None:
    values_list = list(values)
    result = metric(values_list)
    stats.append(
        {
            "layer": layer,
            "scope": scope,
            "model": model,
            "variant": variant,
            "mode": mode,
            "expected_count": expected_count,
            "failed_count": expected_count - result["count"],
            **result,
        }
    )


def build_stats(
    height_rows: list[dict[str, Any]], conditions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats: list[dict[str, Any]] = []
    repeatability: list[dict[str, Any]] = []
    for model in ("CONE", "QUADRATIC"):
        for variant in VARIANTS:
            for mode in MODE_NAMES:
                frame_values = [
                    row["signed_error_mm"]
                    for row in height_rows
                    if row["model"] == model
                    and row["variant"] == variant
                    and row["mode"] == mode
                    and row["status"] == "success"
                ]
                add_stats(stats, "single_frame", "ALL", model, variant, mode, frame_values, 150)
                condition_values = [
                    row["signed_error_mm"]
                    for row in conditions
                    if row["model"] == model
                    and row["variant"] == variant
                    and row["mode"] == mode
                ]
                add_stats(
                    stats,
                    "global_condition",
                    "ALL",
                    model,
                    variant,
                    mode,
                    condition_values,
                    30,
                )
                for dataset in DATASETS:
                    values = [
                        row["signed_error_mm"]
                        for row in conditions
                        if row["dataset"] == dataset
                        and row["model"] == model
                        and row["variant"] == variant
                        and row["mode"] == mode
                    ]
                    add_stats(
                        stats,
                        "per_height",
                        dataset,
                        model,
                        variant,
                        mode,
                        values,
                        5,
                    )
                    for position_rank in range(1, 6):
                        values = [
                            row["signed_error_mm"]
                            for row in conditions
                            if row["dataset"] == dataset
                            and row["position_rank"] == position_rank
                            and row["model"] == model
                            and row["variant"] == variant
                            and row["mode"] == mode
                        ]
                        add_stats(
                            stats,
                            "height_position",
                            f"{dataset}/position_{position_rank}",
                            model,
                            variant,
                            mode,
                            values,
                            5,
                        )
                sigmas = [
                    row["repeatability_sigma_mm"]
                    for row in conditions
                    if row["model"] == model
                    and row["variant"] == variant
                    and row["mode"] == mode
                    and row["repeatability_sigma_mm"] is not None
                ]
                repeatability.append(
                    {
                        "model": model,
                        "variant": variant,
                        "mode": mode,
                        "condition_count": len(sigmas),
                        "median_sigma_mm": float(np.median(sigmas)) if sigmas else None,
                        "p95_sigma_mm": float(np.quantile(sigmas, 0.95)) if sigmas else None,
                        "max_sigma_mm": float(np.max(sigmas)) if sigmas else None,
                    }
                )
    return stats, repeatability


def build_paired_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["position_rank"], row["variant"], row["mode"], row["model"]): row
        for row in conditions
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for position_rank in range(1, 6):
            for variant in VARIANTS:
                for mode in MODE_NAMES:
                    cone = lookup.get((dataset, position_rank, variant, mode, "CONE"))
                    quadratic = lookup.get(
                        (dataset, position_rank, variant, mode, "QUADRATIC")
                    )
                    if cone is None or quadratic is None:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "truth_mm": cone["truth_mm"],
                            "position_rank": position_rank,
                            "variant": variant,
                            "mode": mode,
                            "cone_measured_mm": cone["measured_mean_mm"],
                            "quadratic_measured_mm": quadratic["measured_mean_mm"],
                            "cone_error_mm": cone["signed_error_mm"],
                            "quadratic_error_mm": quadratic["signed_error_mm"],
                            "quadratic_minus_cone_height_mm": quadratic["measured_mean_mm"]
                            - cone["measured_mean_mm"],
                            "quadratic_abs_error_minus_cone_mm": quadratic["abs_error_mm"]
                            - cone["abs_error_mm"],
                            "cone_pass_0p2mm": cone["pass_0p2mm"],
                            "quadratic_pass_0p2mm": quadratic["pass_0p2mm"],
                        }
                    )
    return rows


def fmt(value: Any, digits: int = 5) -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def render_report(
    path: Path,
    data_root: Path,
    config_path: Path,
    roi_path: Path,
    centers_path: Path,
    stats: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    repeatability: list[dict[str, Any]],
    coordinate_rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    stats_lookup = {
        (row["layer"], row["scope"], row["model"], row["variant"], row["mode"]): row
        for row in stats
    }

    def stat(layer: str, scope: str, model: str, variant: str, mode: str) -> dict[str, Any]:
        return stats_lookup[(layer, scope, model, variant, mode)]

    lines = [
        "# 0819 150 张量块 C0 模型 A/B 对照",
        "",
        f"- 生成时间（UTC）：{now_utc()}",
        f"- 数据根目录：`{data_root.resolve()}`",
        f"- 配置：`{config_path.resolve()}`",
        f"- ROI：`{roi_path.resolve()}`，geometry-only、manual-frozen、30/30",
        f"- Steger 中心：复用 `{centers_path.resolve()}`，本轮不重新提取",
        "- 比较范围：C0-only；关闭 Frozen C1、H1、H-B2 和任何结果驱动 ROI",
        "- ground/reference：保持原验收的 `local_adjacent`、`all_non_height`、`fixed_zg_zero` 三种模式",
        "",
        "## 结论摘要",
        "",
    ]

    for variant, title in (
        ("native_valid", "各模型自身有效点集"),
        ("common_valid", "两模型共同有效点集"),
    ):
        cone = stat("single_frame", "ALL", "CONE", variant, "local_adjacent")
        quadratic = stat("single_frame", "ALL", "QUADRATIC", variant, "local_adjacent")
        q_better_rmse = (
            quadratic["rmse_mm"] is not None
            and cone["rmse_mm"] is not None
            and quadratic["rmse_mm"] < cone["rmse_mm"]
        )
        lines.append(
            f"- **{title} / local_adjacent**："
            f"Cone RMSE={fmt(cone['rmse_mm'])} mm，"
            f"Quadratic RMSE={fmt(quadratic['rmse_mm'])} mm；"
            f"Quadratic {'更小' if q_better_rmse else '不更小'}。"
        )
        lines.append(
            f"  有效测量帧 Cone={cone['count']}/150，"
            f"Quadratic={quadratic['count']}/150；"
            f"P95 分别为 {fmt(cone['p95_mm'])}/{fmt(quadratic['p95_mm'])} mm。"
        )
    if missing:
        lines.append(f"- 缺失的 condition/model/mode 组合：{len(missing)}；详见 `missing_conditions.csv`。")
    else:
        lines.append("- 两模型的 native/common 统计组合均完整，没有缺失 condition。")
    lines.extend(["", "## C0-only 全局量块高度结果", ""])
    lines.extend(
        markdown_table(
            [
                "有效集",
                "模型",
                "模式",
                "帧数",
                "失败",
                "Bias/mm",
                "MAE/mm",
                "RMSE/mm",
                "P95/mm",
                "Max/mm",
                "±0.2 pass",
            ],
            [
                [
                    variant,
                    model,
                    mode,
                    row["count"],
                    row["failed_count"],
                    fmt(row["bias_mm"]),
                    fmt(row["mae_mm"]),
                    fmt(row["rmse_mm"]),
                    fmt(row["p95_mm"]),
                    fmt(row["max_mm"]),
                    f"{row['pass_count']}/{row['count']}",
                ]
                for variant in VARIANTS
                for model in ("CONE", "QUADRATIC")
                for mode in MODE_NAMES
                for row in [stat("single_frame", "ALL", model, variant, mode)]
            ],
        )
    )

    lines.extend(["", "## local_adjacent：按高度", ""])
    lines.extend(
        markdown_table(
            ["高度", "有效集", "Cone RMSE", "Quadratic RMSE", "Cone P95", "Quadratic P95"],
            [
                [
                    dataset,
                    variant,
                    fmt(stat("per_height", dataset, "CONE", variant, "local_adjacent")["rmse_mm"]),
                    fmt(
                        stat("per_height", dataset, "QUADRATIC", variant, "local_adjacent")[
                            "rmse_mm"
                        ]
                    ),
                    fmt(stat("per_height", dataset, "CONE", variant, "local_adjacent")["p95_mm"]),
                    fmt(
                        stat("per_height", dataset, "QUADRATIC", variant, "local_adjacent")[
                            "p95_mm"
                        ]
                    ),
                ]
                for variant in VARIANTS
                for dataset in DATASETS
            ],
        )
    )

    lines.extend(["", "## local_adjacent：按位置（condition mean 的 signed bias）", ""])
    condition_lookup = {
        (row["dataset"], row["position_rank"], row["model"], row["variant"], row["mode"]): row
        for row in conditions
    }
    position_rows: list[list[Any]] = []
    for dataset in DATASETS:
        for position_rank in range(1, 6):
            values = []
            for model in ("CONE", "QUADRATIC"):
                row = condition_lookup.get(
                    (dataset, position_rank, model, "native_valid", "local_adjacent")
                )
                values.append("-" if row is None else fmt(row["signed_error_mm"]))
            position_rows.append([dataset, position_rank, values[0], values[1]])
    lines.extend(markdown_table(["高度", "position", "Cone bias/mm", "Quadratic bias/mm"], position_rows))

    lines.extend(["", "## 重复性", ""])
    lines.extend(
        markdown_table(
            ["模型", "有效集", "模式", "conditions", "median σ/mm", "P95 σ/mm", "max σ/mm"],
            [
                [
                    row["model"],
                    row["variant"],
                    row["mode"],
                    row["condition_count"],
                    fmt(row["median_sigma_mm"]),
                    fmt(row["p95_sigma_mm"]),
                    fmt(row["max_sigma_mm"]),
                ]
                for row in repeatability
                if row["mode"] == "local_adjacent"
            ],
        )
    )

    coordinate_metric = metric(
        row["quadratic_minus_cone_norm_rmse_mm"]
        for row in coordinate_rows
        if row["quadratic_minus_cone_norm_rmse_mm"] is not None
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这次结果直接回答的是：在同一批 0819 图像、同一 Steger 中心、同一 manual ROI 和同一 ground 计算下，裸 C0 采用圆锥还是二次曲面更好。",
            "- `native_valid` 保留每个模型自身的有效点；`common_valid` 只保留两个模型都能重建的点，用来隔离有效性差异。",
            "- 本轮没有为圆锥重新拟合 Cone-specific C1，也没有把 Quadratic 的 Frozen C1 套到圆锥上；因此不能把本报告解释成最终 `Cone+C1` 与 `Quadratic+C1` 的生产链路比较。",
            f"- 同帧共同有效点的 Quadratic−Cone 三维位移，逐帧 RMSE 的汇总值为 {fmt(coordinate_metric['rmse_mm'])} mm；它是坐标变化量，不是真值误差。",
            "",
            "## Provenance",
            "",
            f"- 详细输入、配置、模型 hash、ROI hash 和复用/新增边界见 `provenance.json`。",
            f"- frame 级数据见 `frame_model_metrics.csv`、`coordinate_comparison.csv`；高度统计见 `height_measurements.csv`、`condition_measurements.csv` 和 `stats_summary.csv`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--roi-registry", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--centers-csv", type=Path, default=CACHE_ROOT / "pointwise_diagnostics.csv")
    parser.add_argument("--frame-metrics", type=Path, default=CACHE_ROOT / "frame_metrics.csv")
    parser.add_argument("--input-audit", type=Path, default=CACHE_ROOT / "input_audit.csv")
    parser.add_argument("--cone-model", type=Path, default=None)
    parser.add_argument("--quadratic-model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    roi_path = args.roi_registry.resolve()
    centers_path = args.centers_csv.resolve()
    frame_metrics_path = args.frame_metrics.resolve()
    input_audit_path = args.input_audit.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    app = load_app_config(config_path)
    q_default = (
        TOOL_ROOT
        / "laser_measurement_tool"
        / "configs"
        / "calibration_daheng_0811"
        / "quadratic_graph.yaml"
    )
    cone_default = (
        TOOL_ROOT
        / "laser_measurement_tool"
        / "configs"
        / "calibration_daheng_0811"
        / "circular_cone.yaml"
    )
    model_paths = {
        "CONE": (args.cone_model.resolve() if args.cone_model else cone_default),
        "QUADRATIC": (
            args.quadratic_model.resolve() if args.quadratic_model else q_default
        ),
    }
    entries, roi_summary, audit_payload = load_cached_entries(
        data_root,
        centers_path,
        frame_metrics_path,
        input_audit_path,
        roi_path,
    )
    calibrations = {
        label: model_calibration(app, model_paths[label], expected_type)
        for label, expected_type, _ in MODEL_SPECS
    }
    if not np.array_equal(calibrations["CONE"]["K"], calibrations["QUADRATIC"]["K"]):
        raise RuntimeError("Cone and Quadratic do not share identical K")
    if not np.array_equal(calibrations["CONE"]["D"], calibrations["QUADRATIC"]["D"]):
        raise RuntimeError("Cone and Quadratic do not share identical D")
    if not np.array_equal(calibrations["CONE"]["R"], calibrations["QUADRATIC"]["R"]):
        raise RuntimeError("Cone and Quadratic do not share identical R")
    if not np.array_equal(calibrations["CONE"]["t"], calibrations["QUADRATIC"]["t"]):
        raise RuntimeError("Cone and Quadratic do not share identical t")

    frame_rows, height_rows, coordinate_rows = reconstruct_and_measure(
        entries, app, calibrations
    )
    conditions = condition_rows_from_frames(height_rows)
    missing = missing_conditions(conditions)
    stats, repeatability = build_stats(height_rows, conditions)

    write_csv(
        output / "raw_input_audit.csv",
        audit_payload["raw_rows"],
        [
            "dataset",
            "pose_id",
            "repeat_index",
            "filename",
            "expected_sha256",
            "actual_sha256",
            "sha256_match",
            "center_count_cached",
            "raw_width",
            "raw_height",
            "offset_x",
            "offset_y",
        ],
    )
    write_csv(
        output / "frame_model_metrics.csv",
        frame_rows,
        [
            "dataset",
            "truth_mm",
            "pose_id",
            "position_rank",
            "repeat_index",
            "filename",
            "center_count",
            "cone_valid_points",
            "quadratic_valid_points",
            "common_valid_points",
            "cone_filtered_points",
            "quadratic_filtered_points",
        ],
    )
    write_csv(
        output / "coordinate_comparison.csv",
        coordinate_rows,
        [
            "dataset",
            "pose_id",
            "repeat_index",
            "position_rank",
            "filename",
            "center_count",
            "cone_valid_points",
            "quadratic_valid_points",
            "common_valid_points",
            "quadratic_minus_cone_mean_dx_mm",
            "quadratic_minus_cone_mean_dy_mm",
            "quadratic_minus_cone_mean_dz_mm",
            "quadratic_minus_cone_norm_rmse_mm",
        ],
    )
    write_csv(
        output / "height_measurements.csv",
        height_rows,
        [
            "dataset",
            "truth_mm",
            "pose_id",
            "position_rank",
            "v_center_px",
            "repeat_index",
            "filename",
            "model",
            "model_type",
            "variant",
            "mode",
            "height_mean_mm",
            "height_median_mm",
            "height_std_mm",
            "signed_error_mm",
            "abs_error_mm",
            "valid_point_count",
            "height_point_count",
            "baseline_point_count",
            "status",
            "error",
            "roi_geometry_only",
            "c1_enabled",
            "stage_a_enabled",
        ],
    )
    write_csv(
        output / "condition_measurements.csv",
        conditions,
        [
            "dataset",
            "truth_mm",
            "position_rank",
            "model",
            "variant",
            "mode",
            "repeat_count",
            "measured_mean_mm",
            "measured_median_mm",
            "repeatability_sigma_mm",
            "signed_error_mm",
            "abs_error_mm",
            "pass_0p2mm",
        ],
    )
    write_csv(
        output / "stats_summary.csv",
        stats,
        [
            "layer",
            "scope",
            "model",
            "variant",
            "mode",
            "count",
            "expected_count",
            "failed_count",
            "bias_mm",
            "mae_mm",
            "rmse_mm",
            "p95_mm",
            "max_mm",
            "pass_count",
            "pass_rate",
            "limit_mm",
        ],
    )
    write_csv(
        output / "repeatability_summary.csv",
        repeatability,
        [
            "model",
            "variant",
            "mode",
            "condition_count",
            "median_sigma_mm",
            "p95_sigma_mm",
            "max_sigma_mm",
        ],
    )
    paired = build_paired_conditions(conditions)
    write_csv(
        output / "paired_condition_comparison.csv",
        paired,
        [
            "dataset",
            "truth_mm",
            "position_rank",
            "variant",
            "mode",
            "cone_measured_mm",
            "quadratic_measured_mm",
            "cone_error_mm",
            "quadratic_error_mm",
            "quadratic_minus_cone_height_mm",
            "quadratic_abs_error_minus_cone_mm",
            "cone_pass_0p2mm",
            "quadratic_pass_0p2mm",
        ],
    )
    write_csv(
        output / "missing_conditions.csv",
        missing,
        ["dataset", "position_rank", "model", "variant", "mode"],
    )

    provenance = {
        "generated_at_utc": now_utc(),
        "script": file_info(Path(__file__)),
        "data_root": str(data_root),
        "config": file_info(config_path),
        "roi_registry": file_info(roi_path),
        "source_centers_cache": file_info(centers_path),
        "source_frame_metrics": file_info(frame_metrics_path),
        "source_input_audit": file_info(input_audit_path),
        "models": {
            label: {
                "model_type": expected_type,
                "path": file_info(model_paths[label]),
            }
            for label, expected_type, _ in MODEL_SPECS
        },
        "reused_artifacts": [
            "0819 raw TIFF files; SHA-256 rechecked",
            "0819 manual_frozen_v2 pointwise_diagnostics.csv as exact Steger center cache",
            "0819 manual_frozen_v2 frame_metrics.csv",
            "0819 manual_frozen_v2 input_audit.csv",
            "0819 manual-confirmed geometry-only ROI registry",
            "Daheng 0811 K/D/R/t and reconstruction/measurement parameters",
        ],
        "newly_computed": [
            "raw TIFF hash re-audit",
            "C0-only circular_cone reconstruction on all cached centers",
            "C0-only quadratic_graph reconstruction on all cached centers",
            "native-valid and common-valid local/all-non-height/fixed-Zg measurements",
            "paired per-frame and per-condition Cone/Quadratic statistics",
        ],
        "not_used": [
            "Frozen C1 correction",
            "H1 and H-B2",
            "automatic or result-driven ROI selection",
            "new Steger extraction",
            "any C0 refit or C1 refit",
        ],
        "protocol": {
            "scope": "C0-only same-protocol A/B",
            "steger_centers_reused": True,
            "c1_enabled": False,
            "stage_a_enabled": False,
            "modes": list(MODE_NAMES),
            "roi_summary": roi_summary,
            "audit_summary": audit_payload["audit_summary"],
            "frame_count": len(entries),
            "height_measurement_rows": len(height_rows),
            "condition_rows": len(conditions),
            "missing_condition_rows": len(missing),
        },
    }
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "comparison_summary.json",
        {
            "provenance": provenance,
            "frame_model_metrics": frame_rows,
            "coordinate_comparison": coordinate_rows,
            "height_measurements": height_rows,
            "conditions": conditions,
            "stats": stats,
            "repeatability": repeatability,
            "paired_conditions": paired,
            "missing_conditions": missing,
        },
    )
    render_report(
        output / "model_comparison_report.md",
        data_root,
        config_path,
        roi_path,
        centers_path,
        stats,
        conditions,
        repeatability,
        coordinate_rows,
        missing,
        provenance,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "raw_image_count": len(entries),
                "height_rows": len(height_rows),
                "condition_rows": len(conditions),
                "missing_conditions": len(missing),
                "local_adjacent_native": {
                    model: next(
                        row
                        for row in stats
                        if row["layer"] == "single_frame"
                        and row["scope"] == "ALL"
                        and row["model"] == model
                        and row["variant"] == "native_valid"
                        and row["mode"] == "local_adjacent"
                    )
                    for model in ("CONE", "QUADRATIC")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
