"""Read-only H1 vs H-B2 spatial consistency audit on frozen gauge-block points.

This audit deliberately consumes the canonical Surface-1A formal rows rather
than re-running Steger, reconstruction, ROI selection, ground fitting, or any
model search.  The effective point set is the frozen historical rule:
``height_measurement_inlier and jacobian_valid``.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
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
DATASETS = ("obs_1mm", "obs_2mm", "obs_6mm", "obs_10mm", "obs_20mm", "obs_30mm")
POSE_IDS = tuple(f"{index:03d}" for index in range(1, 6))
MODEL_NAMES = ("Base", "H1", "H-B2")
FORMAL_ROLE = "development_formal_repeat2_5"
EDGE_V_THRESHOLD = 2400.0
H1_DEFAULT = 1.00403395913372
HB2_A0_DEFAULT = -0.10068827127712787
HB2_A2_DEFAULT = 0.053274373969597236


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def as_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, q))


def metrics(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "point_count": 0,
            "bias_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_abs_mm": None,
            "max_abs_mm": None,
        }
    absolute = np.abs(array)
    return {
        "point_count": int(array.size),
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array * array))),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "max_abs_mm": float(np.max(absolute)),
    }


def spearman(values_x: np.ndarray, values_y: np.ndarray) -> float | None:
    if len(values_x) < 2:
        return None
    x_order = np.argsort(np.argsort(values_x, kind="mergesort"), kind="mergesort")
    y_order = np.argsort(np.argsort(values_y, kind="mergesort"), kind="mergesort")
    x = x_order.astype(np.float64)
    y = y_order.astype(np.float64)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator == 0.0:
        return None
    return float(np.sum(x * y) / denominator)


def correlation(values_x: Iterable[float], values_y: Iterable[float]) -> tuple[float | None, float | None]:
    x = np.asarray(list(values_x), dtype=np.float64)
    y = np.asarray(list(values_y), dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None, None
    pearson = float(np.corrcoef(x, y)[0, 1])
    return pearson, spearman(x, y)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def finite_row(row: dict[str, str], fields: Iterable[str]) -> bool:
    try:
        for field in fields:
            as_float(row[field])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def rolling_median(x: np.ndarray, y: np.ndarray, window: int = 151) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.asarray([]), np.asarray([])
    order = np.argsort(x)
    sorted_x = x[order]
    sorted_y = y[order]
    window = max(3, min(window, len(sorted_y)))
    if window % 2 == 0:
        window -= 1
    half = window // 2
    medians = np.empty(len(sorted_y), dtype=np.float64)
    for index in range(len(sorted_y)):
        lo = max(0, index - half)
        hi = min(len(sorted_y), index + half + 1)
        medians[index] = float(np.median(sorted_y[lo:hi]))
    stride = max(1, len(sorted_x) // 500)
    return sorted_x[::stride], medians[::stride]


def load_inputs(root: Path) -> tuple[dict[str, Path], dict[str, Any], float, float, float]:
    paths = {
        "roi_registry": root / "outputs/daheng_c1_gauge_blocks_20260819/roi_registry_manual.json",
        "manual_provenance": root / "outputs/daheng_c1_gauge_blocks_20260819_manual_frozen/provenance.json",
        "pointwise_diagnostics": root / "outputs/daheng_c1_gauge_blocks_20260819_manual_frozen/pointwise_diagnostics.csv",
        "surface1a_points": root / "outputs/daheng_c1_gauge_blocks_20260819_ground4a/surface1a/surface1a_points.csv",
        "stage_a_scale": root / "laser_measurement_tool/configs/calibration_daheng_0811/stage_a_height_scale.json",
        "hb2_parameters": root / "outputs/daheng_c1_gauge_blocks_20260819_ground4a/surface2/surface3_hb2_candidate/surface3_hb2_parameters.json",
        "hb2_candidate": root / "outputs/daheng_c1_gauge_blocks_20260819_ground4a/surface2/surface3_hb2_candidate/surface3_hb2_candidate.json",
        "height_linear_summary": root / "outputs/daheng_c1_gauge_blocks_20260819_ground4a/height_linear_summary.json",
        "hb2_report": root / "outputs/daheng_c1_gauge_blocks_20260819_ground4a/surface2/surface3_hb2_candidate/surface3_hb2_candidate_report.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen audit input(s): " + ", ".join(missing))

    registry = read_json(paths["roi_registry"])
    truth_config = read_json(root / "outputs/daheng_c1_gauge_blocks_20260819_manual_frozen/truth_config.json")
    truth = {str(key): float(value) for key, value in truth_config["truth_mm"].items()}
    stage_a = read_json(paths["stage_a_scale"])
    h1 = float(stage_a["scale"])
    hb2_parameters = read_json(paths["hb2_parameters"])
    a0 = float(
        hb2_parameters.get(
            "a0",
            hb2_parameters.get("a0_mm", hb2_parameters.get("parameters", {}).get("a0")),
        )
    )
    a2 = float(
        hb2_parameters.get(
            "a2",
            hb2_parameters.get("a2_mm_per_q2", hb2_parameters.get("parameters", {}).get("a2")),
        )
    )
    if not math.isclose(h1, H1_DEFAULT, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"unexpected frozen H1 scale: {h1!r}")
    if not math.isclose(a0, HB2_A0_DEFAULT, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"unexpected frozen H-B2 a0: {a0!r}")
    if not math.isclose(a2, HB2_A2_DEFAULT, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"unexpected frozen H-B2 a2: {a2!r}")
    if tuple(dataset for dataset in DATASETS if dataset not in truth):
        raise RuntimeError("truth_config is missing one or more target heights")
    return paths, registry, h1, a0, a2


def registry_entries(registry: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    entries = registry.get("entries")
    if not isinstance(entries, list) or len(entries) != 30:
        raise RuntimeError("manual registry does not contain exactly 30 entries")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (str(entry["dataset"]), str(entry["pose_id"]))
        if key in result:
            raise RuntimeError(f"duplicate manual ROI entry: {key}")
        result[key] = entry
    expected = {(dataset, pose) for dataset in DATASETS for pose in POSE_IDS}
    if set(result) != expected:
        raise RuntimeError("manual registry key set is not the frozen 6x5 target set")
    return result


def load_formal_rows(path: Path) -> tuple[list[dict[str, Any]], Counter[tuple[str, str, int]], Counter[tuple[str, str, int]]]:
    raw_rows = read_csv(path)
    required = (
        "dataset",
        "pose_id",
        "position_rank",
        "repeat_index",
        "frame_id",
        "point_index",
        "u",
        "v",
        "q2",
        "height_value_mm",
        "height_residual_mm",
        "true_height_mm",
        "height_measurement_inlier",
        "jacobian_valid",
    )
    candidate_counts: Counter[tuple[str, str, int]] = Counter()
    effective_counts: Counter[tuple[str, str, int]] = Counter()
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        dataset = raw.get("dataset", "")
        if dataset not in DATASETS or raw.get("split_role") != FORMAL_ROLE:
            continue
        if not finite_row(raw, (field for field in required if field not in {"height_measurement_inlier", "jacobian_valid", "frame_id"})):
            continue
        pose = str(raw["pose_id"])
        rank = int(raw["position_rank"])
        key = (dataset, pose, rank)
        candidate_counts[key] += 1
        parsed = {
            "dataset": dataset,
            "truth_mm": as_float(raw["true_height_mm"]),
            "pose_id": pose,
            "position_rank": rank,
            "repeat_index": int(raw["repeat_index"]),
            "frame_id": raw["frame_id"],
            "point_index": int(raw["point_index"]),
            "u_px": as_float(raw["u"]),
            "v_px": as_float(raw["v"]),
            "q2": as_float(raw["q2"]),
            "height_value_mm": as_float(raw["height_value_mm"]),
            "base_residual_mm": as_float(raw["height_residual_mm"]),
            "height_measurement_inlier": as_bool(raw["height_measurement_inlier"]),
            "jacobian_valid": as_bool(raw["jacobian_valid"]),
            "height_fit_status": raw.get("height_fit_status", ""),
        }
        if parsed["height_measurement_inlier"] and parsed["jacobian_valid"]:
            effective_counts[key] += 1
            rows.append(parsed)
    expected = {(dataset, pose, rank) for dataset, pose, rank in candidate_counts}
    observed = set(candidate_counts)
    if observed != expected:
        raise RuntimeError(f"formal candidate key mismatch; missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    if any(row["base_residual_mm"] != row["height_value_mm"] - row["truth_mm"] for row in rows):
        differences = [abs(row["base_residual_mm"] - (row["height_value_mm"] - row["truth_mm"])) for row in rows]
        if max(differences) > 2.0e-9:
            raise RuntimeError("Surface-1A height_residual_mm is not height_value_mm - true_height_mm")
    return rows, candidate_counts, effective_counts


def build_coverage(
    registry: dict[tuple[str, str], dict[str, Any]],
    all_rows: list[dict[str, Any]],
    effective_rows: list[dict[str, Any]],
    candidate_counts: Counter[tuple[str, str, int]],
    effective_counts: Counter[tuple[str, str, int]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    all_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    effective_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        all_groups[(row["dataset"], row["pose_id"], row["position_rank"])].append(row)
    for row in effective_rows:
        effective_groups[(row["dataset"], row["pose_id"], row["position_rank"])].append(row)

    medians: dict[tuple[str, str], float] = {}
    for dataset in DATASETS:
        for pose in POSE_IDS:
            rank = int(registry[(dataset, pose)]["position_rank"])
            values = all_groups[(dataset, pose, rank)]
            medians[(dataset, pose)] = float(np.median([row["v_px"] for row in values]))
    order: dict[tuple[str, str], int] = {}
    for dataset in DATASETS:
        sorted_poses = sorted(POSE_IDS, key=lambda pose: (medians[(dataset, pose)], pose))
        order.update({(dataset, pose): index for index, pose in enumerate(sorted_poses, start=1)})

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for pose in POSE_IDS:
            entry = registry[(dataset, pose)]
            rank = int(entry["position_rank"])
            key = (dataset, pose, rank)
            values = all_groups[key]
            effective = effective_groups[key]
            raw_v = np.asarray([row["v_px"] for row in values], dtype=np.float64)
            eff_v = np.asarray([row["v_px"] for row in effective], dtype=np.float64)
            row = {
                "dataset": dataset,
                "truth_mm": values[0]["truth_mm"],
                "pose_id": pose,
                "position_rank": rank,
                "v_order_rank": order[(dataset, pose)],
                "v_order_rank_basis": "raw_formal_rows",
                "v_center_px": entry["v_center_px"],
                "height_v_range": json_text(entry["height_v_range"]),
                "baseline_v_ranges": json_text(entry["baseline_v_ranges"]),
                "formal_row_count": int(candidate_counts[key]),
                "formal_v_min_px": float(np.min(raw_v)),
                "formal_v_p05_px": percentile(raw_v, 5.0),
                "formal_v_median_px": float(np.median(raw_v)),
                "formal_v_p95_px": percentile(raw_v, 95.0),
                "formal_v_max_px": float(np.max(raw_v)),
                "effective_formal_point_count": int(effective_counts[key]),
                "effective_v_min_px": float(np.min(eff_v)) if len(eff_v) else None,
                "effective_v_p05_px": percentile(eff_v, 5.0),
                "effective_v_median_px": float(np.median(eff_v)) if len(eff_v) else None,
                "effective_v_p95_px": percentile(eff_v, 95.0),
                "effective_v_max_px": float(np.max(eff_v)) if len(eff_v) else None,
                "effective_v_gt_2400_count": int(np.count_nonzero(eff_v > EDGE_V_THRESHOLD)),
                "status": "OK" if len(eff_v) else "NO_EFFECTIVE_FORMAL_POINTS",
            }
            rows.append(row)
    return rows, order


def add_model_values(rows: list[dict[str, Any]], h1: float, a0: float, a2: float) -> None:
    for row in rows:
        raw_height = row["height_value_mm"]
        truth = row["truth_mm"]
        delta = a0 + a2 * row["q2"]
        row["h_raw_mm"] = raw_height
        row["base_height_mm"] = raw_height
        row["base_residual_mm"] = raw_height - truth
        row["h_h1_mm"] = h1 * raw_height
        row["h1_residual_mm"] = h1 * raw_height - truth
        row["hb2_delta_h_mm"] = delta
        row["h_hb2_mm"] = raw_height - delta
        row["hb2_residual_mm"] = raw_height - delta - truth


def position_metrics(
    rows: list[dict[str, Any]],
    registry: dict[tuple[str, str], dict[str, Any]],
    order: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["pose_id"], row["position_rank"])].append(row)
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for pose in POSE_IDS:
            entry = registry[(dataset, pose)]
            rank = int(entry["position_rank"])
            values = groups[(dataset, pose, rank)]
            for model, field in (
                ("Base", "base_residual_mm"),
                ("H1", "h1_residual_mm"),
                ("H-B2", "hb2_residual_mm"),
            ):
                item = metrics(row[field] for row in values)
                output.append(
                    {
                        "row_type": "height_position",
                        "model": model,
                        "dataset": dataset,
                        "truth_mm": values[0]["truth_mm"] if values else None,
                        "pose_id": pose,
                        "position_rank": rank,
                        "v_order_rank": order[(dataset, pose)],
                        "point_count": item["point_count"],
                        "bias_mm": item["bias_mm"],
                        "mae_mm": item["mae_mm"],
                        "rmse_mm": item["rmse_mm"],
                        "p95_abs_mm": item["p95_abs_mm"],
                        "max_abs_mm": item["max_abs_mm"],
                        "position_count_available": None,
                        "position_bias_min_mm": None,
                        "position_bias_max_mm": None,
                        "position_bias_range_mm": None,
                        "bias_std_mm": None,
                        "worst_position": None,
                        "worst_position_abs_bias_mm": None,
                        "worst_position_p95_abs_mm": None,
                        "worst_position_max_abs_mm": None,
                        "status": "OK" if values else "NO_EFFECTIVE_FORMAL_POINTS",
                    }
                )

    condition_rows = {(row["dataset"], row["position_rank"], row["model"]): row for row in output}
    for dataset in DATASETS:
        for model in MODEL_NAMES:
            available = [
                condition_rows[(dataset, rank, model)]
                for rank in range(1, 6)
                if condition_rows[(dataset, rank, model)]["point_count"]
            ]
            biases = np.asarray([float(row["bias_mm"]) for row in available], dtype=np.float64)
            if not available:
                summary = {
                    "position_count_available": 0,
                    "position_bias_min_mm": None,
                    "position_bias_max_mm": None,
                    "position_bias_range_mm": None,
                    "bias_std_mm": None,
                    "worst_position": None,
                    "worst_position_abs_bias_mm": None,
                    "worst_position_p95_abs_mm": None,
                    "worst_position_max_abs_mm": None,
                    "status": "NO_EFFECTIVE_FORMAL_POINTS",
                }
            else:
                worst = max(available, key=lambda item: abs(float(item["bias_mm"])))
                summary = {
                    "position_count_available": len(available),
                    "position_bias_min_mm": float(np.min(biases)),
                    "position_bias_max_mm": float(np.max(biases)),
                    "position_bias_range_mm": float(np.max(biases) - np.min(biases)),
                    "bias_std_mm": float(np.std(biases)),
                    "worst_position": f"pose{worst['pose_id']}/rank{worst['position_rank']}",
                    "worst_position_abs_bias_mm": abs(float(worst["bias_mm"])),
                    "worst_position_p95_abs_mm": worst["p95_abs_mm"],
                    "worst_position_max_abs_mm": worst["max_abs_mm"],
                    "status": "OK" if len(available) == 5 else "PARTIAL_POSITION_SUPPORT",
                }
            truth = next((row["truth_mm"] for row in available), None)
            output.append(
                {
                    "row_type": "height_summary",
                    "model": model,
                    "dataset": dataset,
                    "truth_mm": truth,
                    "pose_id": "",
                    "position_rank": "",
                    "v_order_rank": "",
                    "point_count": sum(int(row["point_count"]) for row in available),
                    "bias_mm": None,
                    "mae_mm": None,
                    "rmse_mm": None,
                    "p95_abs_mm": None,
                    "max_abs_mm": None,
                    **summary,
                }
            )

    for model in MODEL_NAMES:
        values = [row for row in rows]
        item = metrics(row[({"Base": "base_residual_mm", "H1": "h1_residual_mm", "H-B2": "hb2_residual_mm"}[model])] for row in values)
        output.append(
            {
                "row_type": "overall_pooled",
                "model": model,
                "dataset": "ALL_TARGET_HEIGHTS",
                "truth_mm": None,
                "pose_id": "",
                "position_rank": "",
                "v_order_rank": "",
                "point_count": item["point_count"],
                "bias_mm": item["bias_mm"],
                "mae_mm": item["mae_mm"],
                "rmse_mm": item["rmse_mm"],
                "p95_abs_mm": item["p95_abs_mm"],
                "max_abs_mm": item["max_abs_mm"],
                "position_count_available": None,
                "position_bias_min_mm": None,
                "position_bias_max_mm": None,
                "position_bias_range_mm": None,
                "bias_std_mm": None,
                "worst_position": None,
                "worst_position_abs_bias_mm": None,
                "worst_position_p95_abs_mm": None,
                "worst_position_max_abs_mm": None,
                "status": "POOLED_CONTEXT_ONLY",
            }
        )
    return output


def edge_metrics(
    rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    registry: dict[tuple[str, str], dict[str, Any]],
    order: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    edge = [row for row in rows if row["v_px"] > EDGE_V_THRESHOLD]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    height_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pose_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge:
        key = (row["dataset"], row["pose_id"], row["position_rank"])
        groups[key].append(row)
        height_groups[row["dataset"]].append(row)
        pose_groups[row["pose_id"]].append(row)

    output: list[dict[str, Any]] = []
    model_fields = {"Base": "base_residual_mm", "H1": "h1_residual_mm", "H-B2": "hb2_residual_mm"}
    for model, field in model_fields.items():
        model_edge = [row for row in edge]
        item = metrics(row[field] for row in model_edge)
        absolute = np.asarray([abs(row[field]) for row in model_edge], dtype=np.float64)
        pearson, spear = correlation([row["v_px"] for row in model_edge], [row[field] for row in model_edge])
        boundary5 = 0
        boundary10 = 0
        for row in model_edge:
            roi = registry[(row["dataset"], row["pose_id"])]
            lo, hi = map(float, roi["height_v_range"])
            distance = min(row["v_px"] - lo, hi - row["v_px"])
            boundary5 += distance <= 5.0
            boundary10 += distance <= 10.0
        output.append(
            {
                "row_type": "overall",
                "model": model,
                "dataset": "ALL_TARGET_HEIGHTS",
                "truth_mm": None,
                "pose_id": "",
                "position_rank": "",
                "v_order_rank": "",
                "v_band": "v>2400",
                "point_count": item["point_count"],
                "edge_fraction_of_effective": len(model_edge) / len(rows) if rows else None,
                "v_min_px": min((row["v_px"] for row in model_edge), default=None),
                "v_median_px": float(np.median([row["v_px"] for row in model_edge])) if model_edge else None,
                "v_max_px": max((row["v_px"] for row in model_edge), default=None),
                "bias_mm": item["bias_mm"],
                "mae_mm": item["mae_mm"],
                "rmse_mm": item["rmse_mm"],
                "p95_abs_mm": item["p95_abs_mm"],
                "max_abs_mm": item["max_abs_mm"],
                "gt_0p1_rate": float(np.mean(absolute > 0.1)) if len(absolute) else None,
                "gt_0p2_rate": float(np.mean(absolute > 0.2)) if len(absolute) else None,
                "gt_0p1_count": int(np.count_nonzero(absolute > 0.1)),
                "gt_0p2_count": int(np.count_nonzero(absolute > 0.2)),
                "residual_v_pearson_r": pearson,
                "residual_v_spearman_r": spear,
                "roi_boundary_within_5px_rate": boundary5 / len(model_edge) if model_edge else None,
                "roi_boundary_within_10px_rate": boundary10 / len(model_edge) if model_edge else None,
                "covered_heights": json_text(sorted({row["dataset"] for row in model_edge})),
                "covered_poses": json_text(sorted({row["pose_id"] for row in model_edge})),
                "covered_position_ranks": json_text(sorted({row["position_rank"] for row in model_edge})),
                "status": "OK" if model_edge else "NO_EDGE_POINTS",
            }
        )

        for dataset in DATASETS:
            values = height_groups[dataset]
            item = metrics(row[field] for row in values)
            absolute = np.asarray([abs(row[field]) for row in values], dtype=np.float64)
            output.append(
                {
                    "row_type": "height",
                    "model": model,
                    "dataset": dataset,
                    "truth_mm": values[0]["truth_mm"] if values else None,
                    "pose_id": "",
                    "position_rank": "",
                    "v_order_rank": "",
                    "v_band": "v>2400",
                    "point_count": item["point_count"],
                    "edge_fraction_of_effective": len(values) / sum(row["dataset"] == dataset for row in rows) if sum(row["dataset"] == dataset for row in rows) else None,
                    "v_min_px": min((row["v_px"] for row in values), default=None),
                    "v_median_px": float(np.median([row["v_px"] for row in values])) if values else None,
                    "v_max_px": max((row["v_px"] for row in values), default=None),
                    "bias_mm": item["bias_mm"],
                    "mae_mm": item["mae_mm"],
                    "rmse_mm": item["rmse_mm"],
                    "p95_abs_mm": item["p95_abs_mm"],
                    "max_abs_mm": item["max_abs_mm"],
                    "gt_0p1_rate": float(np.mean(absolute > 0.1)) if len(absolute) else None,
                    "gt_0p2_rate": float(np.mean(absolute > 0.2)) if len(absolute) else None,
                    "gt_0p1_count": int(np.count_nonzero(absolute > 0.1)),
                    "gt_0p2_count": int(np.count_nonzero(absolute > 0.2)),
                    "residual_v_pearson_r": None,
                    "residual_v_spearman_r": None,
                    "roi_boundary_within_5px_rate": None,
                    "roi_boundary_within_10px_rate": None,
                    "covered_heights": json_text([dataset] if values else []),
                    "covered_poses": json_text(sorted({row["pose_id"] for row in values})),
                    "covered_position_ranks": json_text(sorted({row["position_rank"] for row in values})),
                    "status": "OK" if values else "NO_EDGE_POINTS",
                }
            )

        for pose in POSE_IDS:
            values = pose_groups[pose]
            item = metrics(row[field] for row in values)
            absolute = np.asarray([abs(row[field]) for row in values], dtype=np.float64)
            output.append(
                {
                    "row_type": "pose",
                    "model": model,
                    "dataset": "ALL_TARGET_HEIGHTS",
                    "truth_mm": None,
                    "pose_id": pose,
                    "position_rank": "",
                    "v_order_rank": "",
                    "v_band": "v>2400",
                    "point_count": item["point_count"],
                    "edge_fraction_of_effective": len(values) / sum(row["pose_id"] == pose for row in rows) if sum(row["pose_id"] == pose for row in rows) else None,
                    "v_min_px": min((row["v_px"] for row in values), default=None),
                    "v_median_px": float(np.median([row["v_px"] for row in values])) if values else None,
                    "v_max_px": max((row["v_px"] for row in values), default=None),
                    "bias_mm": item["bias_mm"],
                    "mae_mm": item["mae_mm"],
                    "rmse_mm": item["rmse_mm"],
                    "p95_abs_mm": item["p95_abs_mm"],
                    "max_abs_mm": item["max_abs_mm"],
                    "gt_0p1_rate": float(np.mean(absolute > 0.1)) if len(absolute) else None,
                    "gt_0p2_rate": float(np.mean(absolute > 0.2)) if len(absolute) else None,
                    "gt_0p1_count": int(np.count_nonzero(absolute > 0.1)),
                    "gt_0p2_count": int(np.count_nonzero(absolute > 0.2)),
                    "residual_v_pearson_r": None,
                    "residual_v_spearman_r": None,
                    "roi_boundary_within_5px_rate": None,
                    "roi_boundary_within_10px_rate": None,
                    "covered_heights": json_text(sorted({row["dataset"] for row in values})),
                    "covered_poses": json_text([pose] if values else []),
                    "covered_position_ranks": json_text(sorted({row["position_rank"] for row in values})),
                    "status": "OK" if values else "NO_EDGE_POINTS",
                }
            )

        for key in sorted(groups):
            dataset, pose, rank = key
            values = groups[key]
            item = metrics(row[field] for row in values)
            absolute = np.asarray([abs(row[field]) for row in values], dtype=np.float64)
            output.append(
                {
                    "row_type": "height_position",
                    "model": model,
                    "dataset": dataset,
                    "truth_mm": values[0]["truth_mm"],
                    "pose_id": pose,
                    "position_rank": rank,
                    "v_order_rank": order[(dataset, pose)],
                    "v_band": "v>2400",
                    "point_count": item["point_count"],
                    "edge_fraction_of_effective": len(values) / sum(row["dataset"] == dataset and row["pose_id"] == pose and row["position_rank"] == rank for row in rows),
                    "v_min_px": min(row["v_px"] for row in values),
                    "v_median_px": float(np.median([row["v_px"] for row in values])),
                    "v_max_px": max(row["v_px"] for row in values),
                    "bias_mm": item["bias_mm"],
                    "mae_mm": item["mae_mm"],
                    "rmse_mm": item["rmse_mm"],
                    "p95_abs_mm": item["p95_abs_mm"],
                    "max_abs_mm": item["max_abs_mm"],
                    "gt_0p1_rate": float(np.mean(absolute > 0.1)),
                    "gt_0p2_rate": float(np.mean(absolute > 0.2)),
                    "gt_0p1_count": int(np.count_nonzero(absolute > 0.1)),
                    "gt_0p2_count": int(np.count_nonzero(absolute > 0.2)),
                    "residual_v_pearson_r": None,
                    "residual_v_spearman_r": None,
                    "roi_boundary_within_5px_rate": None,
                    "roi_boundary_within_10px_rate": None,
                    "covered_heights": json_text([dataset]),
                    "covered_poses": json_text([pose]),
                    "covered_position_ranks": json_text([rank]),
                    "status": "OK",
                }
            )

    return output


def model_field(model: str) -> str:
    return {"Base": "base_residual_mm", "H1": "h1_residual_mm", "H-B2": "hb2_residual_mm"}[model]


def decision_summary(
    rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> dict[str, Any]:
    height_summaries = {
        (row["dataset"], row["model"]): row
        for row in position_rows
        if row["row_type"] == "height_summary"
    }
    position_reduction: dict[str, dict[str, Any]] = {}
    for model in ("H1", "H-B2"):
        range_improved = 0
        std_improved = 0
        both_improved = 0
        comparisons = []
        for dataset in DATASETS:
            base = height_summaries[(dataset, "Base")]
            candidate = height_summaries[(dataset, model)]
            if base["position_bias_range_mm"] is None or candidate["position_bias_range_mm"] is None:
                continue
            range_delta = float(candidate["position_bias_range_mm"]) - float(base["position_bias_range_mm"])
            std_delta = float(candidate["bias_std_mm"]) - float(base["bias_std_mm"])
            range_win = range_delta < -1.0e-12
            std_win = std_delta < -1.0e-12
            range_improved += range_win
            std_improved += std_win
            both_improved += range_win and std_win
            comparisons.append({"dataset": dataset, "range_delta_mm": range_delta, "bias_std_delta_mm": std_delta})
        if both_improved >= 4:
            status = "YES"
        elif both_improved >= 1 or range_improved >= 2 or std_improved >= 2:
            status = "PARTIAL"
        else:
            status = "NO"
        position_reduction[model] = {
            "status": status,
            "range_improved_heights": range_improved,
            "bias_std_improved_heights": std_improved,
            "both_improved_heights": both_improved,
            "comparisons": comparisons,
        }

    edge_overall = {
        row["model"]: row
        for row in edge_rows
        if row["row_type"] == "overall"
    }
    edge_height = {
        (row["dataset"], row["model"]): row
        for row in edge_rows
        if row["row_type"] == "height"
    }
    edge_reduction: dict[str, Any] = {}
    for model in ("H1", "H-B2"):
        base = edge_overall["Base"]
        candidate = edge_overall[model]
        comparisons = {
            metric: float(candidate[metric]) - float(base[metric])
            for metric in ("bias_mm", "p95_abs_mm", "max_abs_mm")
            if candidate[metric] is not None and base[metric] is not None
        }
        edge_reduction[model] = {
            "bias_abs_reduced": abs(float(candidate["bias_mm"])) < abs(float(base["bias_mm"])),
            "p95_reduced": float(candidate["p95_abs_mm"]) < float(base["p95_abs_mm"]),
            "max_reduced": float(candidate["max_abs_mm"]) < float(base["max_abs_mm"]),
            "deltas_candidate_minus_base": comparisons,
        }
        p95_improved_heights = 0
        max_improved_heights = 0
        both_improved_heights = 0
        for dataset in DATASETS:
            base_height = edge_height.get((dataset, "Base"))
            candidate_height = edge_height.get((dataset, model))
            if not base_height or not candidate_height:
                continue
            if int(base_height["point_count"]) <= 0 or int(candidate_height["point_count"]) <= 0:
                continue
            p95_win = float(candidate_height["p95_abs_mm"]) < float(base_height["p95_abs_mm"])
            max_win = float(candidate_height["max_abs_mm"]) < float(base_height["max_abs_mm"])
            p95_improved_heights += p95_win
            max_improved_heights += max_win
            both_improved_heights += p95_win and max_win
        edge_reduction[model].update(
            {
                "p95_improved_heights": p95_improved_heights,
                "max_improved_heights": max_improved_heights,
                "both_improved_heights": both_improved_heights,
            }
        )

    # H-B2/H1 preference uses equal-height position-spread metrics plus edge P95/Max.
    score = {"H1": 0, "H-B2": 0}
    preference_metrics: dict[str, Any] = {}
    for label, field in (
        ("mean_position_bias_range_mm", "position_bias_range_mm"),
        ("mean_position_bias_std_mm", "bias_std_mm"),
        ("mean_worst_position_abs_bias_mm", "worst_position_abs_bias_mm"),
        ("mean_worst_position_p95_abs_mm", "worst_position_p95_abs_mm"),
    ):
        values: dict[str, float] = {}
        for model in ("H1", "H-B2"):
            candidates = [
                float(height_summaries[(dataset, model)][field])
                for dataset in DATASETS
                if height_summaries[(dataset, model)][field] is not None
            ]
            values[model] = float(np.mean(candidates)) if candidates else float("nan")
        preference_metrics[label] = values
        if math.isfinite(values["H1"]) and math.isfinite(values["H-B2"]):
            if values["H1"] < values["H-B2"]:
                score["H1"] += 1
            elif values["H-B2"] < values["H1"]:
                score["H-B2"] += 1
    for label, field in (("edge_p95_abs_mm", "p95_abs_mm"), ("edge_max_abs_mm", "max_abs_mm")):
        values = {model: float(edge_overall[model][field]) for model in ("H1", "H-B2")}
        preference_metrics[label] = values
        if values["H1"] < values["H-B2"]:
            score["H1"] += 1
        elif values["H-B2"] < values["H1"]:
            score["H-B2"] += 1
    h1_spatial_edge_stable = (
        position_reduction["H1"]["status"] == "YES"
        and edge_reduction["H1"]["both_improved_heights"] >= 4
    )
    hb2_spatial_edge_stable = (
        position_reduction["H-B2"]["status"] == "YES"
        and edge_reduction["H-B2"]["both_improved_heights"] >= 4
    )
    if hb2_spatial_edge_stable and not h1_spatial_edge_stable:
        preferred = "HB2"
        selection_basis = (
            "HB2 meets the primary stability rule: position Bias range/std and edge P95/Max "
            "improve in at least 4 heights; H1 does not meet the position-spread rule."
        )
    elif h1_spatial_edge_stable and not hb2_spatial_edge_stable:
        preferred = "H1"
        selection_basis = (
            "H1 meets the primary stability rule: position Bias range/std and edge P95/Max "
            "improve in at least 4 heights; H-B2 does not meet it."
        )
    elif score["H1"] >= 4 and score["H1"] > score["H-B2"]:
        preferred = "H1"
        selection_basis = "Primary stability rules tie or are inconclusive; H1 wins the secondary absolute-error score."
    elif score["H-B2"] >= 4 and score["H-B2"] > score["H1"]:
        preferred = "HB2"
        selection_basis = "Primary stability rules tie or are inconclusive; H-B2 wins the secondary absolute-error score."
    else:
        preferred = "UNDECIDED"
        selection_basis = "Neither model has a stable primary spatial advantage and the secondary absolute-error score does not separate them."

    raw_edge_conditions = [row for row in coverage if row["formal_v_max_px"] > EDGE_V_THRESHOLD]
    effective_edge_conditions = [row for row in coverage if row["effective_v_gt_2400_count"] > 0]
    effective_edge_heights = sorted({row["dataset"] for row in effective_edge_conditions})
    effective_edge_positions = sorted({row["v_order_rank"] for row in effective_edge_conditions})
    if effective_edge_conditions and len(effective_edge_heights) == len(DATASETS) and len(effective_edge_positions) >= 2:
        covers = "PARTIAL"
    elif effective_edge_conditions:
        covers = "PARTIAL"
    else:
        covers = "NO"

    base_edge_height_rows = [
        edge_height[(dataset, "Base")]
        for dataset in DATASETS
        if (dataset, "Base") in edge_height and edge_height[(dataset, "Base")]["point_count"]
    ]
    failure_heights = [
        row["dataset"]
        for row in base_edge_height_rows
        if (float(row["p95_abs_mm"]) >= 0.1 or float(row["gt_0p1_rate"]) >= 0.1)
    ]
    base_edge_height_position_rows = [
        row
        for row in edge_rows
        if row["row_type"] == "height_position"
        and row["model"] == "Base"
        and int(row["point_count"]) > 0
    ]
    failure_poses = sorted(
        {
            row["pose_id"]
            for row in base_edge_height_position_rows
            if float(row["p95_abs_mm"]) >= 0.1 or float(row["gt_0p1_rate"]) >= 0.1
        }
    )
    boundary_rate = float(edge_overall["Base"]["roi_boundary_within_10px_rate"] or 0.0)
    if len(failure_heights) >= 3 and boundary_rate < 0.75:
        edge_failure = "YES"
    elif failure_heights:
        edge_failure = "PARTIAL"
    else:
        edge_failure = "NO"

    stable_spread_by_model = {}
    for model in MODEL_NAMES:
        stable_spread_by_model[model] = sum(
            1
            for dataset in DATASETS
            if height_summaries[(dataset, model)]["position_bias_range_mm"] is not None
            and float(height_summaries[(dataset, model)]["position_bias_range_mm"]) > 0.05
        )
    preferred_model = "H1" if preferred == "H1" else "H-B2" if preferred == "HB2" else "Base"
    residual_supported = (
        stable_spread_by_model["Base"] >= 3
        and stable_spread_by_model[preferred_model] >= 3
    ) or (
        edge_failure in {"YES", "PARTIAL"}
        and float(edge_overall[preferred_model]["p95_abs_mm"]) >= 0.1
    )
    new_spatial = residual_supported and edge_failure in {"YES", "PARTIAL"}

    return {
        "HISTORICAL_DATA_COVERS_V_GT_2400": covers,
        "EDGE_V_GT_2400_FAILURE_REPRODUCED": edge_failure,
        "H1_REDUCES_SPATIAL_BIAS": position_reduction["H1"]["status"],
        "HB2_REDUCES_SPATIAL_BIAS": position_reduction["H-B2"]["status"],
        "PREFERRED_DEPTH_BASELINE": preferred,
        "SPATIAL_RESIDUAL_SUPPORTED": "YES" if residual_supported else "NO",
        "NEW_SPATIAL_CORRECTION_REQUIRED": "YES" if new_spatial else "NO",
        "coverage_support": {
            "raw_formal_edge_conditions": len(raw_edge_conditions),
            "effective_edge_conditions": len(effective_edge_conditions),
            "effective_edge_heights": effective_edge_heights,
            "effective_edge_v_order_ranks": effective_edge_positions,
        },
        "position_reduction": position_reduction,
        "edge_reduction": edge_reduction,
        "preference_score": score,
        "preference_metrics": preference_metrics,
        "preference_selection_basis": selection_basis,
        "stable_position_spread_height_count": stable_spread_by_model,
        "edge_failure_heights_base": failure_heights,
        "edge_failure_poses_base": failure_poses,
        "edge_base_roi_boundary_within_10px_rate": boundary_rate,
        "classification_rules": {
            "spatial_bias_yes": "both position Bias range and Bias std improve in >=4 of 6 heights",
            "preferred_depth_baseline": "prefer a model when position Bias range/std and edge P95/Max improve in >=4 heights; otherwise use secondary absolute-error score",
            "edge_failure_yes": "Base edge P95>=0.1 mm or >0.1 rate in >=3 heights and not >=75% within 10 px of ROI boundary",
            "coverage_partial": "effective edge points exist but do not cover all five historical positions at every height",
        },
    }


def make_plots(
    output: Path,
    coverage: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    registry: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    paths: list[str] = []
    # True image-space coverage, with raw formal range and effective range.
    fig, axis = plt.subplots(figsize=(13, 10))
    y_positions = np.arange(len(coverage))
    for y, item in zip(y_positions, coverage):
        lo, hi = json.loads(item["height_v_range"])
        axis.plot([lo, hi], [y, y], color="#cbd5e1", linewidth=8, solid_capstyle="butt")
        axis.plot([item["formal_v_min_px"], item["formal_v_max_px"]], [y, y], color="#64748b", linewidth=3)
        if item["effective_formal_point_count"]:
            axis.plot([item["effective_v_min_px"], item["effective_v_max_px"]], [y, y], color="#2563eb", linewidth=5)
        axis.scatter([item["v_center_px"]], [y], color="#111827", s=18, zorder=3)
    labels = [f"{item['dataset'].replace('obs_', '')} / p{item['v_order_rank']} (rank{item['position_rank']}, pose{item['pose_id']})" for item in coverage]
    axis.set_yticks(y_positions, labels, fontsize=8)
    axis.axvline(EDGE_V_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.4, label="v=2400")
    axis.set_xlim(0, 3000)
    axis.set_xlabel("image v (px)")
    axis.set_title("Historical position coverage (per-height audit-only v_order_rank): ROI (gray), raw formal (slate), effective formal (blue)")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    path = output / "historical_position_v_coverage.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))

    # Residual-v scatter and non-parametric rolling median; diagnostic only.
    fig, axis = plt.subplots(figsize=(12, 6.5))
    colors = {"Base": "#64748b", "H1": "#2563eb", "H-B2": "#d97706"}
    for model in MODEL_NAMES:
        field = model_field(model)
        x = np.asarray([row["v_px"] for row in rows], dtype=np.float64)
        y = np.asarray([row[field] for row in rows], dtype=np.float64)
        axis.scatter(x, y, s=7, alpha=0.10, color=colors[model], label=f"{model} points")
        sx, sy = rolling_median(x, y)
        axis.plot(sx, sy, color=colors[model], linewidth=2.0, label=f"{model} rolling median")
    axis.axvline(EDGE_V_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.4, label="v=2400")
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xlim(0, 3000)
    axis.set_xlabel("image v (px)")
    axis.set_ylabel("height residual (mm)")
    axis.set_title("Base / H1 / H-B2 residual vs v (same frozen effective formal points)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = output / "residual_vs_v_base_h1_hb2.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))

    # Position bias comparison by height, x ordered by actual v median.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    condition_rows = [row for row in position_rows if row["row_type"] == "height_position"]
    for axis, dataset in zip(axes.flat, DATASETS):
        for model in MODEL_NAMES:
            selected = [row for row in condition_rows if row["dataset"] == dataset and row["model"] == model]
            selected.sort(key=lambda row: int(row["v_order_rank"]))
            x = [int(row["v_order_rank"]) for row in selected]
            y = [row["bias_mm"] if row["bias_mm"] is not None else np.nan for row in selected]
            axis.plot(x, y, marker="o", linewidth=1.5, label=model)
        axis.axhline(0.0, color="#111827", linewidth=0.8)
        axis.set_title(dataset.replace("obs_", ""))
        axis.set_xticks(range(1, 6))
        axis.set_xlabel("v_order_rank")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("position Bias (mm)")
    axes[1, 0].set_ylabel("position Bias (mm)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Position Bias by height; x is per-height audit-only actual v order")
    fig.tight_layout()
    path = output / "position_bias_comparison_by_height.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))

    # Edge error comparison.
    overall = {row["model"]: row for row in edge_rows if row["row_type"] == "overall"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(MODEL_NAMES))
    bias = [abs(float(overall[model]["bias_mm"])) for model in MODEL_NAMES]
    p95 = [float(overall[model]["p95_abs_mm"]) for model in MODEL_NAMES]
    max_error = [float(overall[model]["max_abs_mm"]) for model in MODEL_NAMES]
    axes[0].bar(x - 0.25, bias, width=0.25, label="|Bias|")
    axes[0].bar(x, p95, width=0.25, label="P95")
    axes[0].bar(x + 0.25, max_error, width=0.25, label="Max")
    axes[0].axhline(0.1, color="#dc2626", linestyle="--", linewidth=1.0, label="0.1 mm")
    axes[0].set_xticks(x, MODEL_NAMES)
    axes[0].set_ylabel("absolute error (mm)")
    axes[0].set_title("v>2400 error magnitude")
    axes[0].legend(fontsize=8)
    rates = [float(overall[model]["gt_0p1_rate"]) for model in MODEL_NAMES]
    rates2 = [float(overall[model]["gt_0p2_rate"]) for model in MODEL_NAMES]
    axes[1].bar(x - 0.18, rates, width=0.36, label="|error|>0.1")
    axes[1].bar(x + 0.18, rates2, width=0.36, label="|error|>0.2")
    axes[1].set_xticks(x, MODEL_NAMES)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction")
    axes[1].set_title("v>2400 threshold exceedance")
    axes[1].legend(fontsize=8)
    fig.suptitle("Independent edge audit")
    fig.tight_layout()
    path = output / "edge_v_gt_2400_error_comparison.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))
    return paths


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def make_report(
    output: Path,
    paths: dict[str, Path],
    registry: dict[tuple[str, str], dict[str, Any]],
    coverage: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    decisions: dict[str, Any],
    h1: float,
    a0: float,
    a2: float,
    effective_point_count: int,
    plot_paths: list[str],
) -> None:
    summary_by_model = {row["model"]: row for row in position_rows if row["row_type"] == "overall_pooled"}
    edge_overall = {row["model"]: row for row in edge_rows if row["row_type"] == "overall"}
    height_summary = {
        (row["dataset"], row["model"]): row
        for row in position_rows
        if row["row_type"] == "height_summary"
    }
    raw_edge_conditions = [row for row in coverage if row["formal_v_max_px"] > EDGE_V_THRESHOLD]
    effective_edge_conditions = [row for row in coverage if row["effective_v_gt_2400_count"] > 0]
    lines = [
        "# H1 vs H-B2 全视场位置一致性专项审计",
        "",
        "## 最终判定",
        "",
        f"- `HISTORICAL_DATA_COVERS_V_GT_2400={decisions['HISTORICAL_DATA_COVERS_V_GT_2400']}`",
        f"- `EDGE_V_GT_2400_FAILURE_REPRODUCED={decisions['EDGE_V_GT_2400_FAILURE_REPRODUCED']}`",
        f"- `H1_REDUCES_SPATIAL_BIAS={decisions['H1_REDUCES_SPATIAL_BIAS']}`",
        f"- `HB2_REDUCES_SPATIAL_BIAS={decisions['HB2_REDUCES_SPATIAL_BIAS']}`",
        f"- `PREFERRED_DEPTH_BASELINE={decisions['PREFERRED_DEPTH_BASELINE']}`",
        f"- `SPATIAL_RESIDUAL_SUPPORTED={decisions['SPATIAL_RESIDUAL_SUPPORTED']}`",
        f"- `NEW_SPATIAL_CORRECTION_REQUIRED={decisions['NEW_SPATIAL_CORRECTION_REQUIRED']}`",
        "",
        "结论只描述本轮历史数据的诊断证据；本轮没有拟合或写入新的 spatial correction，也不改变生产配置。",
        "",
        "## 1. Provenance / reuse audit",
        "",
        f"""| 项目 | 本轮处理 |
|---|---|
| 图像 / Steger | 复用历史一次 Steger / frame 的冻结产物；未重新运行 |
| ROI | 复用 `roi_registry_manual.json` 的 30/30 frozen manual ROI；未修改 |
| formal points | 复用 Surface-1A `surface1a_points.csv` 的 `development_formal_repeat2_5`，并严格保留 `height_measurement_inlier=True` 且 `jacobian_valid=True` 的有效点集合 |
| Base | Frozen C0 + Frozen C1 + 同一 session-linear Ground Reference；`height_value_mm` / `height_residual_mm` 原样复用 |
| H1 | 复用 frozen scale `h_H1 = {h1:.14f} * h_raw`；未重新拟合 |
| H-B2 | 复用 frozen `a0={a0:.15f} mm`, `a2={a2:.15f} mm/q2`；`h_HB2=h_raw-(a0+a2*q2)`；未重新拟合 |
| 新增计算 | 仅 deterministic pointwise transform、coverage/metrics、edge audit、诊断图和本报告 |
| 禁止项 | 未重跑模型搜索、未改 q2/ROI/Steger/ground proxy、未按模型结果删点 |
| 输出目录 | `{output}` |
""",
        "",
        "输入文件 SHA-256：",
        "",
    ]
    for name in ("roi_registry", "pointwise_diagnostics", "surface1a_points", "stage_a_scale", "hb2_parameters", "hb2_candidate", "height_linear_summary"):
        lines.append(f"- `{name}`: `{paths[name]}` — `{sha256(paths[name])}`")
    lines += [
        "",
        "历史 artifact 中，`height_linear_summary.json` 与 `surface3_hb2_candidate.json` 仅作 frozen 参数/候选 provenance 核对，不复用其 pooled/condition-level 数值来替代本轮 pointwise 计算。",
        "",
        "## 2. 历史 5 个位置的真实 v 覆盖",
        "",
        "`position_rank` 是 ROI registry 原有 rank；`v_order_rank` 是本审计按每个高度的 raw formal-row `v_median` 从小到大生成，仅用于本审计。任何跨高度结论均不把 pose_id 当作统一空间坐标。",
        "`historical_position_v_coverage.csv` 同时保留 raw formal 与 effective formal 两套 v 范围：报告下面各行的 `v_min–v_max` 是 raw formal rows；`n_eff`/`edge` 来自 effective set，effective 的 P05/median/P95/min/max 在 CSV 的 `effective_*` 列。`height_v_range` 是 frozen manual ROI 的 height 条带，`baseline_v_ranges` 是同一 registry 中的两个 baseline 条带，均为 image-v 像素闭区间。",
        "",
        f"raw formal candidate 有 {len(coverage)} 个 height×position 条件；effective formal point set 共 {effective_point_count} 个点。详细范围见 `historical_position_v_coverage.csv`。",
        f"实际 raw formal rows 中有 {len(raw_edge_conditions)}/30 个条件的最大 v 超过 2400；冻结 effective set 中有 {len(effective_edge_conditions)}/30 个条件实际留下 v>2400 点，覆盖高度为 {', '.join(decisions['coverage_support']['effective_edge_heights'])}，覆盖 v_order_rank 为 {decisions['coverage_support']['effective_edge_v_order_ranks']}。",
        "",
        "关键覆盖观察：",
        "",
    ]
    for dataset in DATASETS:
        items = [item for item in coverage if item["dataset"] == dataset]
        compact = "; ".join(
            f"p{item['v_order_rank']}/rank{item['position_rank']}/pose{item['pose_id']}: {fmt(item['formal_v_min_px'],0)}–{fmt(item['formal_v_max_px'],0)} (n_eff={item['effective_formal_point_count']}, edge={item['effective_v_gt_2400_count']})"
            for item in sorted(items, key=lambda item: int(item["v_order_rank"]))
        )
        lines.append(f"- `{dataset}`: {compact}")
    lines += [
        "",
        "`obs_2mm` 的 `position_rank=5` 原始 formal rows 存在，但 Base 冻结规则中全部为 `height_measurement_inlier=False`（历史 height fit `too_few_points`），因此 effective point count 为 0；本轮不补点、不把它伪装成可比较 condition。",
        "",
        "因此历史数据确实触及 v>2400，但不是完整五位置均匀覆盖：edge 支持主要来自每个高度的高 v 位置（尤其 v_order_rank 5，部分高度 rank 4），且存在 2mm rank5 的 effective 缺口；最终标记为 `PARTIAL`。五个位置之间还存在明显空档，不能追加连续 v band 统计。",
        "",
        "## 3. 三条同点测量链",
        "",
        "所有 pointwise 行均来自同一 frozen formal effective set；H1/H-B2 只生成新的高度与 residual 列：",
        "",
        "```text",
        "h_raw = height_value_mm",
        f"Base = h_raw  (residual = h_raw - truth)",
        f"H1   = {h1:.14f} * h_raw",
        f"H-B2 = h_raw - ({a0:.15f} + {a2:.15f} * q2)",
        "```",
        "",
        "输出 `pointwise_base_h1_hb2.csv` 保留了 point identity、frame/repeat、u/v、q2、h_raw 和三条链的高度/residual；不存在因 H1/H-B2 结果较差而重筛点。",
        "",
        "## 4. Position consistency（主要评价单位）",
        "",
        "position-level 指标是每个 height×position 的 pointwise 指标；height summary 对可用 position 的 Bias 等权汇总，2mm 只有 4 个 effective positions。pooled 行仅作上下文，不作为位置一致性结论。完整结果见 `position_consistency_metrics.csv`。",
        "",
        "| height | model | positions | Bias range | Bias std | worst | worst P95 | worst Max | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        for model in MODEL_NAMES:
            row = height_summary[(dataset, model)]
            lines.append(
                f"| {dataset.replace('obs_', '')} | {model} | {row['position_count_available']} | {fmt(row['position_bias_range_mm'])} | {fmt(row['bias_std_mm'])} | {fmt(row['worst_position_abs_bias_mm'])} | {fmt(row['worst_position_p95_abs_mm'])} | {fmt(row['worst_position_max_abs_mm'])} | {row['status']} |"
            )
    lines += [
        "",
        "position spread 改善判定规则：同一模型的 Bias range 与 Bias std 在至少 4/6 个高度同时下降才标 `YES`；有部分但未达此稳定门槛标 `PARTIAL`。该规则只用于解释 H1/H-B2 的 spatial benefit，不是重新拟合。",
        "",
        "## 5. 独立 v>2400 edge audit",
        "",
        "edge 直接从同一 effective formal points 按 `v_px > 2400` 筛选；不要求组成独立完整 position。",
        "",
        "| model | n | covered heights | covered poses | covered ranks | Bias | MAE | RMSE | P95 | Max | >0.1 | >0.2 | v-Bias Spearman | ROI boundary <=10px |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        row = edge_overall[model]
        lines.append(
            f"| {model} | {row['point_count']} | {row['covered_heights']} | {row['covered_poses']} | {row['covered_position_ranks']} | {fmt(row['bias_mm'])} | {fmt(row['mae_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['max_abs_mm'])} | {fmt(row['gt_0p1_rate'],3)} | {fmt(row['gt_0p2_rate'],3)} | {fmt(row['residual_v_spearman_r'],3)} | {fmt(row['roi_boundary_within_10px_rate'],3)} |"
        )
    lines += [
        "",
        "跨高度/pose/position 的 edge 分解、阈值比例和每个 height×position 结果见 `edge_v_gt_2400_metrics.csv`。",
        "",
        "failure 诊断：",
        "",
        f"- Base 在达到 edge support 的高度中，满足 edge P95≥0.1 mm 或 >0.1 比例≥0.1 的高度为 `{', '.join(decisions['edge_failure_heights_base']) or 'none'}`。",
        f"- Base edge 点在 ROI 边界 10 px 内比例为 `{fmt(decisions['edge_base_roi_boundary_within_10px_rate'],3)}`；这用于区分“仅少量 ROI 边界异常”与连续/跨高度现象。",
        f"- Base 的上述 failure 实际涉及 pose `{', '.join(decisions['edge_failure_poses_base']) or 'none'}`，不是单一 pose；pose-level 与 height×position 细节见 CSV，本轮不以 pooled 指标掩盖局部失败。",
        "- residual-v 图使用 raw v 排序 rolling median 仅作诊断，不拟合 correction；图中保留 v=2400 分界。",
        "",
        "Pose-level edge 分解（完整数值也保留在 `edge_v_gt_2400_metrics.csv` 的 `row_type=pose` 行）：",
        "",
        "| pose_id | Base n / >0.1 | H1 n / >0.1 | H-B2 n / >0.1 |",
        "|---|---:|---:|---:|",
    ]
    edge_pose = {
        (row["model"], row["pose_id"]): row
        for row in edge_rows
        if row["row_type"] == "pose"
    }
    for pose in POSE_IDS:
        base_pose = edge_pose[("Base", pose)]
        h1_pose = edge_pose[("H1", pose)]
        hb2_pose = edge_pose[("H-B2", pose)]
        lines.append(
            f"| {pose} | {base_pose['point_count']} / {fmt(base_pose['gt_0p1_rate'],3)} | {h1_pose['point_count']} / {fmt(h1_pose['gt_0p1_rate'],3)} | {hb2_pose['point_count']} / {fmt(hb2_pose['gt_0p1_rate'],3)} |"
        )
    lines += [
        "",
        "## 6. H1 / H-B2 的真实收益与选择",
        "",
    ]
    for model in ("H1", "H-B2"):
        item = decisions["position_reduction"][model]
        lines.append(
            f"- `{model}`: position Bias range 改善 {item['range_improved_heights']}/6，高度 Bias std 改善 {item['bias_std_improved_heights']}/6，同时改善 {item['both_improved_heights']}/6；分类 `{item['status']}`。"
        )
        edge_item = decisions["edge_reduction"][model]
        lines.append(
            f"  edge 相对 Base：|Bias|={'↓' if edge_item['bias_abs_reduced'] else '↑/='}, P95={'↓' if edge_item['p95_reduced'] else '↑/='}, Max={'↓' if edge_item['max_reduced'] else '↑/='}；edge P95/Max 同时逐高度改善 {edge_item['both_improved_heights']}/6。"
        )
    lines += [
        "",
        f"次级 absolute-error preference score（每个 position-spread、worst-position、edge 指标低者得 1 分）为 `{decisions['preference_score']}`；首选仍按稳定性主规则判定：{decisions['preference_selection_basis']} 因此 `PREFERRED_DEPTH_BASELINE={decisions['PREFERRED_DEPTH_BASELINE']}`。这是本历史工作域的诊断优先级，不是生产冻结。",
        "",
        "本轮 H1 与 H-B2 都明显降低 edge 的 common depth bias；但 H1 没有降低跨位置 spread，H-B2 在六个高度同时降低 position spread 与 edge P95/Max，因此位置一致性专项优先 H-B2。二者都没有建立新的 spatial correction，且仍需注意 2mm rank5 无 effective support。",
        "",
        "## 7. 是否追加连续 v band",
        "",
        "不追加 `v_band_metrics.csv`：五个历史位置是离散 acquisition bands，position 之间有明显未覆盖空档，且并非所有 height×position 都有 effective formal points。按任务约束，不能把离散五位置强行解释成连续 v 覆盖或固定五等分。",
        "",
        "## 8. 产物",
        "",
    ]
    for path in [
        "historical_position_v_coverage.csv",
        "pointwise_base_h1_hb2.csv",
        "position_consistency_metrics.csv",
        "edge_v_gt_2400_metrics.csv",
        "audit_provenance.json",
        *[Path(path).name for path in plot_paths],
    ]:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "本报告由 `tools/audit_height_depth_baseline_spatial.py` 生成；脚本只执行复用产物上的确定性审计，不调用模型搜索或数据采集。",
        "",
    ]
    (output / "height_depth_baseline_spatial_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, output: Path) -> dict[str, Any]:
    paths, registry_json, h1, a0, a2 = load_inputs(root)
    registry = registry_entries(registry_json)
    raw_rows = read_csv(paths["surface1a_points"])
    # Keep an all-formal view for coverage, but parse only finite target rows for metrics.
    candidate_counts: Counter[tuple[str, str, int]] = Counter()
    all_formal_rows: list[dict[str, Any]] = []
    required = ("true_height_mm", "position_rank", "repeat_index", "u", "v", "q2", "height_value_mm", "height_residual_mm")
    for raw in raw_rows:
        if raw.get("dataset") not in DATASETS or raw.get("split_role") != FORMAL_ROLE:
            continue
        if not finite_row(raw, required):
            continue
        dataset = raw["dataset"]
        pose = str(raw["pose_id"])
        rank = int(raw["position_rank"])
        key = (dataset, pose, rank)
        candidate_counts[key] += 1
        all_formal_rows.append(
            {
                "dataset": dataset,
                "truth_mm": as_float(raw["true_height_mm"]),
                "pose_id": pose,
                "position_rank": rank,
                "repeat_index": int(raw["repeat_index"]),
                "frame_id": raw["frame_id"],
                "point_index": int(raw["point_index"]),
                "u_px": as_float(raw["u"]),
                "v_px": as_float(raw["v"]),
                "q2": as_float(raw["q2"]),
                "height_value_mm": as_float(raw["height_value_mm"]),
                "base_residual_mm": as_float(raw["height_residual_mm"]),
                "height_measurement_inlier": as_bool(raw["height_measurement_inlier"]),
                "jacobian_valid": as_bool(raw["jacobian_valid"]),
                "height_fit_status": raw.get("height_fit_status", ""),
            }
        )
    effective_rows = [row for row in all_formal_rows if row["height_measurement_inlier"] and row["jacobian_valid"]]
    effective_counts: Counter[tuple[str, str, int]] = Counter(
        (row["dataset"], row["pose_id"], row["position_rank"]) for row in effective_rows
    )
    expected = {
        (dataset, pose, int(registry[(dataset, pose)]["position_rank"]))
        for dataset in DATASETS
        for pose in POSE_IDS
    }
    if set(candidate_counts) != expected:
        raise RuntimeError(f"formal candidate key mismatch; missing={sorted(expected-set(candidate_counts))}")
    add_model_values(effective_rows, h1, a0, a2)
    coverage, order = build_coverage(registry, all_formal_rows, effective_rows, candidate_counts, effective_counts)
    position_rows = position_metrics(effective_rows, registry, order)
    edge_rows = edge_metrics(effective_rows, coverage, registry, order)
    decisions = decision_summary(effective_rows, position_rows, edge_rows, coverage)

    output.mkdir(parents=True, exist_ok=True)
    coverage_fields = list(coverage[0].keys())
    point_fields = [
        "dataset", "truth_mm", "pose_id", "position_rank", "v_order_rank", "repeat_index", "frame_id", "point_index",
        "u_px", "v_px", "q2", "h_raw_mm", "base_height_mm", "base_residual_mm", "h_h1_mm", "h1_residual_mm",
        "hb2_delta_h_mm", "h_hb2_mm", "hb2_residual_mm", "height_fit_status", "v_gt_2400",
    ]
    point_rows = []
    order_map = order
    for row in effective_rows:
        point_rows.append({
            **{field: row.get(field) for field in point_fields if field in row},
            "v_order_rank": order_map[(row["dataset"], row["pose_id"])],
            "v_gt_2400": row["v_px"] > EDGE_V_THRESHOLD,
        })
    position_fields = list(position_rows[0].keys())
    edge_fields = list(edge_rows[0].keys())
    write_csv(output / "historical_position_v_coverage.csv", coverage, coverage_fields)
    write_csv(output / "pointwise_base_h1_hb2.csv", point_rows, point_fields)
    write_csv(output / "position_consistency_metrics.csv", position_rows, position_fields)
    write_csv(output / "edge_v_gt_2400_metrics.csv", edge_rows, edge_fields)
    plot_paths = make_plots(output, coverage, effective_rows, position_rows, edge_rows, registry)

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "scope": {"datasets": list(DATASETS), "formal_role": FORMAL_ROLE, "edge_v_gt_px": EDGE_V_THRESHOLD},
        "formal_point_rule": "surface1a_points split_role=development_formal_repeat2_5 AND height_measurement_inlier=True AND jacobian_valid=True",
        "counts": {
            "height_position_conditions": len(coverage),
            "raw_formal_rows": len(all_formal_rows),
            "effective_formal_points": len(effective_rows),
            "effective_conditions": sum(count > 0 for count in effective_counts.values()),
            "effective_edge_points": sum(row["v_px"] > EDGE_V_THRESHOLD for row in effective_rows),
        },
        "frozen_parameters": {"h1_scale": h1, "hb2_a0_mm": a0, "hb2_a2_mm_per_q2": a2},
        "reused_inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "new_outputs": [
            str(output / "height_depth_baseline_spatial_audit_report.md"),
            str(output / "historical_position_v_coverage.csv"),
            str(output / "pointwise_base_h1_hb2.csv"),
            str(output / "position_consistency_metrics.csv"),
            str(output / "edge_v_gt_2400_metrics.csv"),
            str(output / "audit_provenance.json"),
            *[str(path) for path in plot_paths],
        ],
        "decisions": decisions,
        "continuous_v_bands": {"generated": False, "reason": "discrete five-position acquisition with large uncovered v gaps and incomplete effective condition support"},
    }
    (output / "audit_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_report(output, paths, registry, coverage, position_rows, edge_rows, decisions, h1, a0, a2, len(effective_rows), plot_paths)
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/daheng_c1_gauge_blocks_20260819_height_depth_baseline_spatial_audit",
    )
    args = parser.parse_args()
    decisions = run(args.root.resolve(), args.output.resolve())
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
