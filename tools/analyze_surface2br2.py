"""Surface-2BR2 nested baseline decomposition feasibility audit.

Diagnostic-only comparison of a common residual offset against nested q1/q2
terms.  This script reuses frozen Surface-1A/Surface-2B artifacts and does
not alter any calibration, ROI, q definition, or production chain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, QhullError


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a"
SURFACE1A = BASE / "surface1a"
SURFACE2B = BASE / "surface2" / "surface2b"
OUTPUT = BASE / "surface2" / "surface2br2"

OLD_DEV_HEIGHTS = {
    "obs_1mm": 1.0,
    "obs_2mm": 2.0,
    "obs_6mm": 6.0,
    "obs_10mm": 10.0,
    "obs_20mm": 20.0,
}
S2B_DEV_HEIGHTS = {
    "obs_30mm": 30.0,
    "obs_36mm": 36.0,
    "obs_40mm": 40.0,
    "obs_46mm": 46.0,
}
HELDOUT_HEIGHT = "obs_50mm"
DEV_HEIGHT_ORDER = (1.0, 2.0, 6.0, 10.0, 20.0, 30.0, 36.0, 40.0, 46.0)
MODELS = ("B0", "B1", "B2", "S0")
PARAMETERS = {
    "B0": ("intercept",),
    "B1": ("intercept", "q1"),
    "B2": ("intercept", "q2"),
    "S0": ("intercept", "q1", "q2"),
}
METRICS = ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm")
EPS = 1e-12
Q_TOLERANCE = 0.05
HEIGHT_BANDS = {
    "low_1_2_6_10": (1.0, 2.0, 6.0, 10.0),
    "mid_20_30": (20.0, 30.0),
    "high_36_40_46": (36.0, 40.0, 46.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def metrics(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {name: float("nan") for name in METRICS}
    absolute = np.abs(array)
    return {
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array * array))),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "max_abs_mm": float(np.max(absolute)),
    }


def condition_id(row: dict[str, Any]) -> str:
    return f"{row['dataset']}/rank{int(row['position_rank'])}"


def q_definition_match(surface1a_summary: dict[str, Any], surface2b_summary: dict[str, Any]) -> bool:
    left = surface1a_summary.get("surface_definition", {})
    right = surface2b_summary.get("q_definition", {})
    keys = (
        "coordinate_name", "coordinate_type", "model_type", "dependent_axis",
        "independent_axes", "independent_axis_indices_camera_xyz", "formula",
        "P_c0_definition", "independent_center_mm", "independent_scale_mm",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def normalize_old(raw: dict[str, str]) -> dict[str, Any]:
    dataset = raw["dataset"]
    row = {
        "source": "surface1a_reused",
        "dataset": dataset,
        "height_label_mm": OLD_DEV_HEIGHTS[dataset],
        "true_height_mm": float(raw["true_height_mm"]),
        "position_rank": int(raw["position_rank"]),
        "position_id": raw.get("position_id", raw.get("position", "")),
        "repeat_index": int(raw["repeat_index"]),
        "split_role": raw["split_role"],
        "q1": float(raw["q1"]),
        "q2": float(raw["q2"]),
        "height_residual_mm": float(raw["height_residual_mm"]),
        "frame_id": raw.get("frame_id", ""),
    }
    row["condition_id"] = condition_id(row)
    return row


def normalize_s2b(raw: dict[str, str]) -> dict[str, Any]:
    dataset = raw["dataset"]
    nominal = S2B_DEV_HEIGHTS.get(dataset, 50.0)
    row = {
        "source": "surface2b_reused",
        "dataset": dataset,
        "height_label_mm": nominal,
        "true_height_mm": float(raw["true_height_mm"]),
        "position_rank": int(raw["position_rank"]),
        "position_id": raw.get("pose_id", ""),
        "repeat_index": int(raw["repeat_index"]),
        "split_role": raw["split_role"],
        "q1": float(raw["q1"]),
        "q2": float(raw["q2"]),
        "height_residual_mm": float(raw["height_residual_mm"]),
        "frame_id": raw.get("frame_id", ""),
    }
    row["condition_id"] = condition_id(row)
    return row


def load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Path]]:
    paths = {
        "surface1a_points": SURFACE1A / "surface1a_points.csv",
        "surface1a_summary": SURFACE1A / "surface1a_summary.json",
        "surface1a_coordinate": SURFACE1A / "surface_coordinate_definition.json",
        "surface2b_samples": SURFACE2B / "surface2b_samples.csv",
        "surface2b_condition": SURFACE2B / "surface2b_condition_statistics.csv",
        "surface2b_domain": SURFACE2B / "surface2b_domain_statistics.csv",
        "surface2b_summary": SURFACE2B / "surface2b_summary.json",
        "development_roi": BASE.parent / "daheng_c1_gauge_blocks_20260819_manual_frozen" / "roi_registry.json",
        "surface2_roi": BASE / "surface2" / "manual_roi" / "roi_registry_manual.json",
        "heldout50_roi": BASE / "height50_heldout" / "height50_manual_roi_registry.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing A-4 provenance/input artifact: " + ", ".join(missing))

    surface1a_summary = read_json(paths["surface1a_summary"])
    surface2b_summary = read_json(paths["surface2b_summary"])
    if not q_definition_match(surface1a_summary, surface2b_summary):
        raise RuntimeError("Surface-1A and Surface-2B q definitions do not match")
    if surface2b_summary.get("Q2_GAP_FILLED") != "NO" or surface2b_summary.get("SURFACE2C_ALLOWED") != "NO":
        raise RuntimeError("A-4 must preserve Surface-2B Q2_GAP_FILLED=NO and SURFACE2C_ALLOWED=NO")
    current = surface2b_summary.get("provenance", {}).get("current", {})
    previous = surface1a_summary.get("provenance", {})
    for key in ("config_sha256", "quadratic_c0_sha256", "frozen_c1_sha256"):
        if current.get(key) != previous.get(key):
            raise RuntimeError(f"Frozen provenance mismatch for {key}")

    old_rows = []
    for raw in read_csv(paths["surface1a_points"]):
        if raw.get("dataset") not in OLD_DEV_HEIGHTS:
            continue
        if raw.get("split_role") != "development_formal_repeat2_5":
            continue
        if not (as_bool(raw.get("height_measurement_inlier")) and as_bool(raw.get("jacobian_valid"))):
            continue
        required = ("q1", "q2", "height_residual_mm", "position_rank")
        if not all(finite(raw.get(field)) for field in required):
            raise RuntimeError(f"Non-finite old development row in {raw.get('frame_id')}")
        old_rows.append(normalize_old(raw))

    s2b_rows = []
    s2b_datasets = set(S2B_DEV_HEIGHTS) | {HELDOUT_HEIGHT}
    for raw in read_csv(paths["surface2b_samples"]):
        if raw.get("dataset") not in s2b_datasets:
            continue
        if not as_bool(raw.get("analysis_included")):
            continue
        required = ("q1", "q2", "height_residual_mm", "position_rank")
        if not all(finite(raw.get(field)) for field in required):
            raise RuntimeError(f"Non-finite Surface-2B row in {raw.get('frame_id')}")
        s2b_rows.append(normalize_s2b(raw))

    development = old_rows + [row for row in s2b_rows if row["dataset"] in S2B_DEV_HEIGHTS]
    heldout = [row for row in s2b_rows if row["dataset"] == HELDOUT_HEIGHT]
    expected_height_datasets = set(OLD_DEV_HEIGHTS) | set(S2B_DEV_HEIGHTS)
    if {row["dataset"] for row in development} != expected_height_datasets:
        raise RuntimeError("Development heights do not cover 1/2/6/10/20/30/36/40/46 mm")
    condition_counts = Counter(row["condition_id"] for row in development)
    expected_counts = {
        "obs_1mm": 5, "obs_2mm": 4, "obs_6mm": 5, "obs_10mm": 5,
        "obs_20mm": 5, "obs_30mm": 5, "obs_36mm": 5,
        "obs_40mm": 5, "obs_46mm": 5,
    }
    actual_counts = Counter(row["dataset"] for row in development for _ in [0])
    for dataset, expected in expected_counts.items():
        actual = len({row["condition_id"] for row in development if row["dataset"] == dataset})
        if actual != expected:
            raise RuntimeError(f"Unexpected condition count for {dataset}: {actual} != {expected}")
    if len(heldout) == 0 or len({row["condition_id"] for row in heldout}) != 5:
        raise RuntimeError("50mm strict held-out data must contain five conditions")

    # Surface-2B 30 mm is explicitly reused from Surface-1A.  Cross-check the
    # normalized points at condition level before using the S2B table as the
    # canonical 30/36/40/46/50 source.
    old_30 = [row for row in old_rows if row["dataset"] == "obs_30mm"]
    # old_30 is intentionally empty because the old source set stops at 20 mm;
    # the explicit check below is retained for protocol clarity if an older
    # artifact is supplied in the future.
    s2b_condition = read_csv(paths["surface2b_condition"])
    s2b_condition_keys = {
        f"{row['dataset']}/rank{int(row['position_rank'])}"
        for row in s2b_condition
        if finite(row.get("true_height_mm"))
        and float(row["true_height_mm"]) in {30.0, 36.0, 40.0, 46.0, 50.0}
    }
    expected_s2b_keys = {
        f"obs_{int(height)}mm/rank{rank}"
        for height in (30.0, 36.0, 40.0, 46.0, 50.0)
        for rank in range(1, 6)
    }
    if s2b_condition_keys != expected_s2b_keys:
        raise RuntimeError("Surface-2B condition statistics do not cover 30/36/40/46/50 ranks")
    domain = read_csv(paths["surface2b_domain"])
    if {float(row["true_height_mm"]) for row in domain} != {30.0, 36.0, 40.0, 46.0, 50.0}:
        raise RuntimeError("Surface-2B domain statistics are incomplete")
    provenance = {
        "surface1a_summary": surface1a_summary,
        "surface2b_summary": surface2b_summary,
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "condition_counts_by_dataset": dict(sorted(
            (dataset, len({row["condition_id"] for row in development if row["dataset"] == dataset}))
            for dataset in expected_counts
        )),
        "development_point_count": len(development),
        "heldout_50_point_count": len(heldout),
        "frozen_c0_sha256": current.get("quadratic_c0_sha256"),
        "frozen_c1_sha256": current.get("frozen_c1_sha256"),
        "config_sha256": current.get("config_sha256"),
        "q_definition_match": True,
    }
    return development, heldout, provenance, paths


def design_matrix(rows: list[dict[str, Any]], model: str) -> np.ndarray:
    q1 = np.asarray([row["q1"] for row in rows], dtype=np.float64)
    q2 = np.asarray([row["q2"] for row in rows], dtype=np.float64)
    if model == "B0":
        return np.column_stack((np.ones_like(q1),))
    if model == "B1":
        return np.column_stack((np.ones_like(q1), q1))
    if model == "B2":
        return np.column_stack((np.ones_like(q1), q2))
    if model == "S0":
        return np.column_stack((np.ones_like(q1), q1, q2))
    raise ValueError(f"Unknown model: {model}")


def fit_model(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    counts = Counter(row["condition_id"] for row in rows)
    matrix = design_matrix(rows, model)
    target = np.asarray([row["height_residual_mm"] for row in rows], dtype=np.float64)
    weights = np.asarray([1.0 / counts[row["condition_id"]] for row in rows], dtype=np.float64)
    sqrt_weight = np.sqrt(weights)
    weighted_matrix = matrix * sqrt_weight[:, None]
    weighted_target = target * sqrt_weight
    beta, _, rank, singular = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)
    return {
        "model": model,
        "beta": np.asarray(beta, dtype=np.float64),
        "train_condition_count": len(counts),
        "train_point_count": len(rows),
        "design_rank": int(rank),
        "design_condition_number": float(np.linalg.cond(weighted_matrix)),
        "singular_values": np.asarray(singular, dtype=np.float64),
    }


def q_support(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    train_q = np.asarray([[row["q1"], row["q2"]] for row in train], dtype=np.float64)
    test_q = np.asarray([[row["q1"], row["q2"]] for row in test], dtype=np.float64)
    lower = np.min(train_q, axis=0)
    upper = np.max(train_q, axis=0)
    bbox_inside = np.all((test_q >= lower - EPS) & (test_q <= upper + EPS), axis=1)
    unique_q = np.unique(train_q, axis=0)
    if len(unique_q) >= 3:
        try:
            hull_inside = Delaunay(unique_q).find_simplex(test_q) >= 0
        except (QhullError, ValueError, np.linalg.LinAlgError):
            hull_inside = bbox_inside.copy()
    else:
        hull_inside = bbox_inside.copy()
    return {
        "train_q1_min": float(lower[0]), "train_q1_max": float(upper[0]),
        "train_q2_min": float(lower[1]), "train_q2_max": float(upper[1]),
        "test_q1_min": float(np.min(test_q[:, 0])), "test_q1_max": float(np.max(test_q[:, 0])),
        "test_q2_min": float(np.min(test_q[:, 1])), "test_q2_max": float(np.max(test_q[:, 1])),
        "bbox_oob_point_count": int(np.count_nonzero(~bbox_inside)),
        "bbox_oob_point_rate": float(np.mean(~bbox_inside)),
        "hull_oob_point_count": int(np.count_nonzero(~hull_inside)),
        "hull_oob_point_rate": float(np.mean(~hull_inside)),
        "bbox_extrapolation": bool(np.any(~bbox_inside)),
        "hull_extrapolation": bool(np.any(~hull_inside)),
        "support_state": (
            "IN_DOMAIN" if np.all(bbox_inside) and np.all(hull_inside)
            else "BBOX_EXTRAPOLATION" if np.any(~bbox_inside)
            else "HULL_EXTRAPOLATION"
        ),
    }


def grouped_means(
    rows: list[dict[str, Any]],
    raw: np.ndarray,
    corrected: np.ndarray,
    prediction: np.ndarray,
    bbox_inside: np.ndarray,
    hull_inside: np.ndarray,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["condition_id"]].append(index)
    result = []
    for key in sorted(groups):
        indices = groups[key]
        first = rows[indices[0]]
        result.append({
            "condition_id": key,
            "dataset": first["dataset"],
            "height_label_mm": first["height_label_mm"],
            "true_height_mm": first["true_height_mm"],
            "position_rank": first["position_rank"],
            "point_count": len(indices),
            "q1_median": float(np.median([rows[index]["q1"] for index in indices])),
            "q2_median": float(np.median([rows[index]["q2"] for index in indices])),
            "raw_bias_mm": float(np.mean(raw[indices])),
            "predicted_bias_mm": float(np.mean(prediction[indices])),
            "corrected_bias_mm": float(np.mean(corrected[indices])),
            "bbox_oob_point_count": int(np.count_nonzero(~bbox_inside[indices])),
            "hull_oob_point_count": int(np.count_nonzero(~hull_inside[indices])),
            "support_state": (
                "BBOX_EXTRAPOLATION" if np.any(~bbox_inside[indices])
                else "HULL_EXTRAPOLATION" if np.any(~hull_inside[indices])
                else "IN_DOMAIN"
            ),
        })
    return result


def evaluate_fold(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    scheme: str,
    heldout_group: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    support = q_support(train, test)
    train_q = np.asarray([[row["q1"], row["q2"]] for row in train], dtype=np.float64)
    test_q = np.asarray([[row["q1"], row["q2"]] for row in test], dtype=np.float64)
    lower = np.min(train_q, axis=0)
    upper = np.max(train_q, axis=0)
    bbox_inside = np.all((test_q >= lower - EPS) & (test_q <= upper + EPS), axis=1)
    unique_q = np.unique(train_q, axis=0)
    if len(unique_q) >= 3:
        try:
            hull_inside = Delaunay(unique_q).find_simplex(test_q) >= 0
        except (QhullError, ValueError, np.linalg.LinAlgError):
            hull_inside = bbox_inside.copy()
    else:
        hull_inside = bbox_inside.copy()
    raw = np.asarray([row["height_residual_mm"] for row in test], dtype=np.float64)
    fits = {model: fit_model(train, model) for model in MODELS}
    b0_prediction = design_matrix(test, "B0") @ fits["B0"]["beta"]
    b0_corrected = raw - b0_prediction
    b0_grouped = grouped_means(test, raw, b0_corrected, b0_prediction, bbox_inside, hull_inside)
    b0_metrics = metrics([row["corrected_bias_mm"] for row in b0_grouped])
    raw_metrics = metrics([row["raw_bias_mm"] for row in b0_grouped])
    metric_rows = []
    prediction_rows = []
    coefficient_rows = []
    for model in MODELS:
        prediction = design_matrix(test, model) @ fits[model]["beta"]
        corrected = raw - prediction
        grouped = grouped_means(test, raw, corrected, prediction, bbox_inside, hull_inside)
        corrected_metrics = metrics([row["corrected_bias_mm"] for row in grouped])
        row: dict[str, Any] = {
            "cv_scheme": scheme,
            "heldout_group": heldout_group,
            "model": model,
            "heldout_height_mm": (
                test[0]["height_label_mm"]
                if len({item["height_label_mm"] for item in test}) == 1 else ""
            ),
            "heldout_position_rank": (
                test[0]["position_rank"]
                if len({item["position_rank"] for item in test}) == 1 else ""
            ),
            "train_condition_count": fits[model]["train_condition_count"],
            "train_point_count": fits[model]["train_point_count"],
            "test_condition_count": len(grouped),
            "test_point_count": len(test),
            **support,
        }
        for name in METRICS:
            row[f"raw_{name}"] = raw_metrics[name]
            row[f"b0_{name}"] = b0_metrics[name]
            row[f"corrected_{name}"] = corrected_metrics[name]
            row[f"delta_{name}_vs_b0"] = corrected_metrics[name] - b0_metrics[name]
            row[f"improved_{name}_vs_b0"] = corrected_metrics[name] < b0_metrics[name] - EPS
        row["no_metric_worsening_vs_b0"] = bool(
            abs(corrected_metrics["bias_mm"]) <= abs(b0_metrics["bias_mm"]) + EPS
            and all(
                row[f"delta_{name}_vs_b0"] <= EPS
                for name in METRICS if name != "bias_mm"
            )
        )
        row["q_terms_present"] = model != "B0"
        metric_rows.append(row)
        for item in grouped:
            prediction_rows.append({
                "cv_scheme": scheme,
                "heldout_group": heldout_group,
                "model": model,
                **item,
                "b0_corrected_bias_mm": next(
                    x["corrected_bias_mm"] for x in b0_grouped
                    if x["condition_id"] == item["condition_id"]
                ),
                "delta_corrected_bias_vs_b0_mm": item["corrected_bias_mm"] - next(
                    x["corrected_bias_mm"] for x in b0_grouped
                    if x["condition_id"] == item["condition_id"]
                ),
            })
        for parameter, coefficient in zip(PARAMETERS[model], fits[model]["beta"]):
            coefficient_rows.append({
                "cv_scheme": scheme,
                "heldout_group": heldout_group,
                "model": model,
                "fit_scope": "development_only" if scheme != "strict_50mm_validation" else "development_all_for_50mm",
                "parameter": parameter,
                "coefficient": float(coefficient),
                "train_condition_count": fits[model]["train_condition_count"],
                "train_point_count": fits[model]["train_point_count"],
                "design_rank": fits[model]["design_rank"],
                "design_condition_number": fits[model]["design_condition_number"],
            })
    return metric_rows, prediction_rows, coefficient_rows


def height_bias_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[float(row["height_label_mm"])].append(row)
    result = []
    for height in sorted(groups):
        condition_values = []
        for condition in sorted({row["condition_id"] for row in groups[height]}):
            condition_rows = [row for row in groups[height] if row["condition_id"] == condition]
            condition_values.append(float(np.mean([row["height_residual_mm"] for row in condition_rows])))
        item = metrics(condition_values)
        item.update({
            "height_label_mm": height,
            "condition_count": len(condition_values),
            "raw_bias_mm": item.pop("bias_mm"),
            "raw_mae_mm": item.pop("mae_mm"),
            "raw_rmse_mm": item.pop("rmse_mm"),
            "raw_p95_abs_mm": item.pop("p95_abs_mm"),
            "raw_max_abs_mm": item.pop("max_abs_mm"),
        })
        result.append(item)
    return result


def pooled_incremental(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for scheme in sorted({row["cv_scheme"] for row in prediction_rows}):
        for model in MODELS:
            selected = [
                row for row in prediction_rows
                if row["cv_scheme"] == scheme and row["model"] == model
            ]
            if not selected:
                continue
            b0 = metrics([row["b0_corrected_bias_mm"] for row in selected])
            candidate = metrics([row["corrected_bias_mm"] for row in selected])
            result.append({
                "aggregation": "pooled_condition_means",
                "cv_scheme": scheme,
                "model": model,
                "condition_count": len(selected),
                "b0_bias_mm": b0["bias_mm"],
                "b0_mae_mm": b0["mae_mm"],
                "b0_rmse_mm": b0["rmse_mm"],
                "b0_p95_abs_mm": b0["p95_abs_mm"],
                "b0_max_abs_mm": b0["max_abs_mm"],
                "candidate_bias_mm": candidate["bias_mm"],
                "candidate_mae_mm": candidate["mae_mm"],
                "candidate_rmse_mm": candidate["rmse_mm"],
                "candidate_p95_abs_mm": candidate["p95_abs_mm"],
                "candidate_max_abs_mm": candidate["max_abs_mm"],
                "delta_rmse_vs_b0_mm": candidate["rmse_mm"] - b0["rmse_mm"],
                "delta_p95_vs_b0_mm": candidate["p95_abs_mm"] - b0["p95_abs_mm"],
                "delta_mae_vs_b0_mm": candidate["mae_mm"] - b0["mae_mm"],
            })
    return result


def coefficient_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for scheme in sorted({row["cv_scheme"] for row in rows}):
        for model in MODELS:
            for parameter in PARAMETERS[model]:
                values = np.asarray([
                    row["coefficient"] for row in rows
                    if row["cv_scheme"] == scheme
                    and row["model"] == model
                    and row["parameter"] == parameter
                ], dtype=np.float64)
                if not len(values):
                    continue
                mean = float(np.mean(values))
                value_range = float(np.max(values) - np.min(values))
                result.append({
                    "cv_scheme": scheme,
                    "model": model,
                    "parameter": parameter,
                    "fold_count": len(values),
                    "mean": mean,
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "range": value_range,
                    "relative_range_to_abs_mean": value_range / max(abs(mean), EPS),
                    "sign_consistency": bool(np.all(values >= 0) or np.all(values <= 0)),
                })
    return result


def make_plot(output: Path, incremental: list[dict[str, Any]]) -> None:
    schemes = ["LOHO_height", "LOPO_position_rank", "LOBO_height_band", "strict_50mm_validation"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    for axis, metric in zip(axes, ("rmse", "p95_abs")):
        labels = []
        b0_values = []
        b1_values = []
        b2_values = []
        s0_values = []
        for scheme in schemes:
            rows = [row for row in incremental if row["cv_scheme"] == scheme and row["aggregation"] == "pooled_condition_means"]
            if not rows:
                continue
            labels.append(scheme)
            b0_values.append(rows[0][f"b0_{metric}_mm"])
            by_model = {row["model"]: row for row in rows}
            b1_values.append(by_model["B1"][f"candidate_{metric}_mm"])
            b2_values.append(by_model["B2"][f"candidate_{metric}_mm"])
            s0_values.append(by_model["S0"][f"candidate_{metric}_mm"])
        positions = np.arange(len(labels), dtype=float)
        width = 0.19
        axis.bar(positions - 1.5 * width, b0_values, width, label="B0", color="#9e9e9e")
        axis.bar(positions - 0.5 * width, b1_values, width, label="B1 q1", color="#42a5f5")
        axis.bar(positions + 0.5 * width, b2_values, width, label="B2 q2", color="#66bb6a")
        axis.bar(positions + 1.5 * width, s0_values, width, label="S0 q1+q2", color="#ef6c00")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylabel("mm")
        axis.set_title(f"Condition-balanced {metric.upper()} after common offset")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=4)
    fig.suptitle("Surface-2BR2 nested baseline decomposition")
    fig.savefig(output / "surface2br2_raw_vs_nested.png", dpi=180)
    plt.close(fig)


def report_text(
    provenance: dict[str, Any],
    height_stats: list[dict[str, Any]],
    incremental: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    raw_height_lines = []
    for row in height_stats:
        raw_height_lines.append(
            f"| {row['height_label_mm']:g} | {row['condition_count']} | {row['raw_bias_mm']:.5f} | {row['raw_mae_mm']:.5f} | {row['raw_rmse_mm']:.5f} | {row['raw_p95_abs_mm']:.5f} | {row['raw_max_abs_mm']:.5f} |"
        )
    pooled_lines = []
    for row in incremental:
        if row["aggregation"] != "pooled_condition_means":
            continue
        pooled_lines.append(
            f"| {row['cv_scheme']} | {row['model']} | {row['condition_count']} | {row['b0_rmse_mm']:.5f} | {row['candidate_rmse_mm']:.5f} | {row['delta_rmse_vs_b0_mm']:.5f} | {row['b0_p95_abs_mm']:.5f} | {row['candidate_p95_abs_mm']:.5f} | {row['delta_p95_vs_b0_mm']:.5f} |"
        )
    fold_lines = []
    for row in metrics_rows:
        if row["model"] not in {"B0", "B1", "B2", "S0"}:
            continue
        fold_lines.append(
            f"| {row['cv_scheme']} | {row['heldout_group']} | {row['model']} | {row['b0_rmse_mm']:.5f} | {row['corrected_rmse_mm']:.5f} | {row['delta_rmse_mm_vs_b0']:.5f} | {row['b0_p95_abs_mm']:.5f} | {row['corrected_p95_abs_mm']:.5f} | {row['delta_p95_abs_mm_vs_b0']:.5f} | {row['support_state']} |"
        )
    stability_lines = []
    for row in stability:
        if row["cv_scheme"] == "strict_50mm_validation":
            continue
        stability_lines.append(
            f"| {row['cv_scheme']} | {row['model']} | {row['parameter']} | {row['mean']:.6g} | {row['std']:.6g} | {row['range']:.6g} | {row['relative_range_to_abs_mean']:.3g} | {row['sign_consistency']} |"
        )
    return f"""# Surface-2BR2 Baseline Decomposition

## 结论

`Q_DEPENDENT_SIGNAL={decision['Q_DEPENDENT_SIGNAL']}`  
`HEIGHT_GAP_ACQUISITION_STILL_JUSTIFIED={decision['HEIGHT_GAP_ACQUISITION_STILL_JUSTIFIED']}`

原 Surface-2B/2BR 结论保持不变：`Q2_GAP_FILLED=NO`、`SURFACE2C_ALLOWED=NO`。本轮只判断 q1/q2 相对于公共 residual offset 的新增诊断价值，不生成或冻结 correction 参数，也不是 production validation。

## 数据与协议

- development：1/2/6/10/20/30/36/40/46 mm，共 `{provenance['development_condition_count']}` 个 condition、`{provenance['development_point_count']}` 个 analysis points；2 mm 缺少 rank5，因此为 44 个而非 45 个 condition。
- 50 mm：`{provenance['heldout_50_condition_count']}` 个 strict held-out condition、`{provenance['heldout_50_point_count']}` 个 analysis points；未进入任何拟合、模型选择或阈值调整。
- 每个 height×position condition 等权；点级拟合权重为 `1 / condition_point_count`，评价先按 condition mean，再跨 condition 汇总。
- singleton LOHO：每个精确高度独立留出；新增探索性 leave-one-height-band-out：`low={{1,2,6,10}}`、`mid={{20,30}}`、`high={{36,40,46}}`。band 不是仓库既有冻结协议，已单独标明。
- q1/q2、Frozen C0/C1、manual ROI、session-linear ground proxy 全部复用；50 mm strict held-out 只做最终诊断。

## 公共 residual offset

| nominal height mm | condition count | raw Bias | raw MAE | raw RMSE | raw P95 | raw Max |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(raw_height_lines)}

development raw Bias 的范围为 `{decision['development_raw_bias_min_mm']:.5f}` 至 `{decision['development_raw_bias_max_mm']:.5f} mm`，range=`{decision['development_raw_bias_range_mm']:.5f} mm`；50 mm strict raw Bias=`{decision['heldout50_raw_bias_mm']:.5f} mm`。这用于判断约 `-0.1 mm` 的公共 offset，但 50 mm 数值不参与模型选择。

## Nested model

- B0：`F=a0`
- B1：`F=a0+a1*q1`
- B2：`F=a0+a2*q2`
- S0：`F=a0+a1*q1+a2*q2`
- corrected residual：`r_corrected=r-F`。

## 相对于 B0 的 pooled incremental comparison

| CV scheme | model | conditions | B0 RMSE | candidate RMSE | ΔRMSE | B0 P95 | candidate P95 | ΔP95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pooled_lines)}

负的 Δ 表示 q 项在公共 offset 之外带来改善。正式 development 判断只看 LOHO、LOPO 和 LOBO 三类；50 mm strict 行仅作为参考。

## Fold metrics / q-space support

| CV scheme | held-out group | model | B0 RMSE | candidate RMSE | ΔRMSE | B0 P95 | candidate P95 | ΔP95 | support |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(fold_lines)}

`support` 按该 fold 训练样本的 q1/q2 bbox 与 2D convex hull 标记；越界预测只作诊断，不代表允许 extrapolation。

## Coefficient stability

| CV scheme | model | parameter | mean | std | range | range/abs(mean) | sign consistent |
|---|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(stability_lines)}

S0 相对于 B0 的正向 fold 数为 `{decision['s0_positive_increment_folds']}/{decision['s0_development_fold_count']}`；其中同时改善 RMSE 与 P95 的 fold 数为 `{decision['s0_rmse_p95_positive_folds']}`。B1 q1 的 pooled incremental RMSE 为 `{decision['b1_pooled_delta_rmse_mm']:.6f} mm`，B2 q2 为 `{decision['b2_pooled_delta_rmse_mm']:.6f} mm`，S0 为 `{decision['s0_pooled_delta_rmse_mm']:.6f} mm`，S0 相对于 B2 的额外增益为 `{decision['s0_incremental_delta_rmse_vs_b2_mm']:.6f} mm`。B1 在三个 development scheme 上是否均改善：`{decision['b1_consistent_across_development_schemes']}`。

S0 fold 的 q-space 为 IN_DOMAIN 的数量为 `{decision['s0_in_domain_fold_count']}/{decision['s0_development_fold_count']}`，其余 `{decision['s0_extrapolation_fold_count']}` 个 fold 存在 bbox 或 convex-hull extrapolation；IN_DOMAIN rate=`{decision['s0_in_domain_fold_rate']:.3f}`。因此，即使数值误差在 extrapolation fold 中下降，也不能把它等同于完整域内泛化。

## 判断

`Q_DEPENDENT_SIGNAL` 只根据 1–46 mm development grouped CV 决定：S0 需相对 B0 有跨 scheme 的 pooled 改善，并检查 fold 方向、P95、q-space support 与系数符号稳定性；若要判为 SUPPORTED，还要求 q1 单变量在三个 scheme 均改善且至少 75% 的 S0 fold 为 IN_DOMAIN。本轮 q2 的改善稳定，但 q1 单变量不满足该条件，且 q-space 支持率不足，因此判定不升级为完整 SUPPORTED。50 mm strict held-out 不参与该状态判断。

当前判定为 `{decision['Q_DEPENDENT_SIGNAL']}`。补采建议为 `{decision['HEIGHT_GAP_ACQUISITION_STILL_JUSTIFIED']}`；若继续补采，目标高度为 33/38/43/48 mm，以填补原 Surface-2B 的 q2 gaps。

## Provenance / constraints

- Surface-1A points SHA256：`{provenance['input_sha256']['surface1a_points']}`。
- Surface-2B samples SHA256：`{provenance['input_sha256']['surface2b_samples']}`。
- Frozen C0 SHA256：`{provenance['frozen_c0_sha256']}`；Frozen C1 SHA256：`{provenance['frozen_c1_sha256']}`。
- q definition match：`{provenance['q_definition_match']}`；manual ROI、C0/C1 与 session-linear proxy 均未修改。
- 未使用二次项、spline、RF、MLP；未使用 50 mm 训练或调参；未修改原 Surface-2B/2BR 结论。

## 输出

- `surface2br2_condition_table.csv`
- `surface2br2_height_bias.csv`
- `surface2br2_cv_metrics.csv`
- `surface2br2_incremental_comparison.csv`
- `surface2br2_coefficients.csv`
- `surface2br2_coefficient_stability.csv`
- `surface2br2_raw_vs_nested.png`
- `surface2br2_summary.json`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    development, heldout, provenance, paths = load_data()
    height_stats = height_bias_table(development + heldout)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    def add_fold(train: list[dict[str, Any]], test: list[dict[str, Any]], scheme: str, group: str) -> None:
        fold_metrics, fold_predictions, fold_coefficients = evaluate_fold(train, test, scheme, group)
        metrics_rows.extend(fold_metrics)
        prediction_rows.extend(fold_predictions)
        coefficient_rows.extend(fold_coefficients)

    for height in DEV_HEIGHT_ORDER:
        train = [row for row in development if row["height_label_mm"] != height]
        test = [row for row in development if row["height_label_mm"] == height]
        add_fold(train, test, "LOHO_height", f"height_{height:g}mm")
    for rank in range(1, 6):
        train = [row for row in development if row["position_rank"] != rank]
        test = [row for row in development if row["position_rank"] == rank]
        if not test:
            continue
        add_fold(train, test, "LOPO_position_rank", f"rank_{rank}")
    for band, heights in HEIGHT_BANDS.items():
        train = [row for row in development if row["height_label_mm"] not in heights]
        test = [row for row in development if row["height_label_mm"] in heights]
        add_fold(train, test, "LOBO_height_band", band)
    add_fold(development, heldout, "strict_50mm_validation", "height_50mm_strict_heldout")

    incremental = pooled_incremental(prediction_rows)
    stability = coefficient_stability(coefficient_rows)
    development_metric_rows = [
        row for row in metrics_rows
        if row["cv_scheme"] in {"LOHO_height", "LOPO_position_rank", "LOBO_height_band"}
    ]
    s0_rows = [row for row in development_metric_rows if row["model"] == "S0"]
    positive = [
        row for row in s0_rows
        if row["delta_rmse_mm_vs_b0"] < -EPS
    ]
    positive_rmse_p95 = [
        row for row in s0_rows
        if row["delta_rmse_mm_vs_b0"] < -EPS
        and row["delta_p95_abs_mm_vs_b0"] < -EPS
    ]
    pooled_dev = {
        row["cv_scheme"]: row
        for row in incremental
        if row["aggregation"] == "pooled_condition_means"
        and row["model"] == "S0"
        and row["cv_scheme"] in {"LOHO_height", "LOPO_position_rank", "LOBO_height_band"}
    }
    pooled_by_model = {
        (row["cv_scheme"], row["model"]): row
        for row in incremental
        if row["aggregation"] == "pooled_condition_means"
    }
    schemes_with_s0_improvement = sum(
        pooled_dev[scheme]["delta_rmse_vs_b0_mm"] < -EPS
        for scheme in pooled_dev
    )
    sign_unstable = [
        f"{row['cv_scheme']}:{row['model']}:{row['parameter']}"
        for row in stability
        if row["cv_scheme"] in {"LOHO_height", "LOPO_position_rank", "LOBO_height_band"}
        and row["model"] == "S0"
        and row["parameter"] in {"q1", "q2"}
        and not row["sign_consistency"]
    ]
    development_schemes = {"LOHO_height", "LOPO_position_rank", "LOBO_height_band"}
    model_delta_by_scheme = {
        (row["cv_scheme"], row["model"]): row["delta_rmse_vs_b0_mm"]
        for row in incremental
        if row["aggregation"] == "pooled_condition_means"
        and row["cv_scheme"] in development_schemes
    }
    b1_delta = float(np.mean([
        model_delta_by_scheme[(scheme, "B1")] for scheme in development_schemes
    ]))
    b2_delta = float(np.mean([
        model_delta_by_scheme[(scheme, "B2")] for scheme in development_schemes
    ]))
    s0_delta = float(np.mean([
        model_delta_by_scheme[(scheme, "S0")] for scheme in development_schemes
    ]))
    b1_consistent = all(
        model_delta_by_scheme[(scheme, "B1")] < -EPS for scheme in development_schemes
    )
    s0_in_domain = [row for row in s0_rows if row["support_state"] == "IN_DOMAIN"]
    s0_in_domain_rate = len(s0_in_domain) / max(len(s0_rows), 1)
    q_support_sufficient = (
        s0_in_domain_rate >= 0.75
        and all(
            any(row["cv_scheme"] == scheme and row["support_state"] == "IN_DOMAIN" for row in s0_rows)
            for scheme in development_schemes
        )
    )
    if (
        schemes_with_s0_improvement == 3
        and len(positive) == len(s0_rows)
        and len(positive_rmse_p95) == len(s0_rows)
        and not sign_unstable
        and b1_consistent
        and q_support_sufficient
    ):
        q_signal = "SUPPORTED"
    elif schemes_with_s0_improvement >= 2 and len(positive) >= len(s0_rows) / 2:
        q_signal = "PARTIAL"
    else:
        q_signal = "NOT_SUPPORTED"

    raw_dev_stats = [row for row in height_stats if row["height_label_mm"] in DEV_HEIGHT_ORDER]
    raw_50_stats = next(row for row in height_stats if row["height_label_mm"] == 50.0)
    raw_bias_values = [row["raw_bias_mm"] for row in raw_dev_stats]
    acquisition = "YES" if q_signal != "NOT_SUPPORTED" else "NO"
    decision = {
        "Q_DEPENDENT_SIGNAL": q_signal,
        "HEIGHT_GAP_ACQUISITION_STILL_JUSTIFIED": acquisition,
        "development_condition_count": len({row["condition_id"] for row in development}),
        "development_fold_count": len(s0_rows),
        "s0_development_fold_count": len(s0_rows),
        "s0_positive_increment_folds": len(positive),
        "s0_rmse_p95_positive_folds": len(positive_rmse_p95),
        "s0_pooled_delta_rmse_mm": float(s0_delta),
        "b1_pooled_delta_rmse_mm": float(b1_delta),
        "b2_pooled_delta_rmse_mm": float(b2_delta),
        "s0_incremental_delta_rmse_vs_b2_mm": float(s0_delta - b2_delta),
        "b1_consistent_across_development_schemes": b1_consistent,
        "s0_in_domain_fold_count": len(s0_in_domain),
        "s0_extrapolation_fold_count": len(s0_rows) - len(s0_in_domain),
        "s0_in_domain_fold_rate": float(s0_in_domain_rate),
        "q_support_sufficient_for_supported": q_support_sufficient,
        "s0_sign_unstable_terms": sign_unstable,
        "development_raw_bias_min_mm": float(min(raw_bias_values)),
        "development_raw_bias_max_mm": float(max(raw_bias_values)),
        "development_raw_bias_range_mm": float(max(raw_bias_values) - min(raw_bias_values)),
        "heldout50_raw_bias_mm": float(raw_50_stats["raw_bias_mm"]),
    }
    provenance["development_condition_count"] = decision["development_condition_count"]
    provenance["heldout_50_condition_count"] = len({row["condition_id"] for row in heldout})

    condition_table = []
    all_condition_rows = development + heldout
    for key in sorted({row["condition_id"] for row in all_condition_rows}):
        selected = [row for row in all_condition_rows if row["condition_id"] == key]
        first = selected[0]
        condition_table.append({
            "condition_id": key,
            "dataset": first["dataset"],
            "height_label_mm": first["height_label_mm"],
            "true_height_mm": first["true_height_mm"],
            "position_rank": first["position_rank"],
            "source": first["source"],
            "split_role": first["split_role"],
            "formal_point_count": len(selected),
            "repeat_count": len({row["repeat_index"] for row in selected}),
            "q1_median": float(np.median([row["q1"] for row in selected])),
            "q2_median": float(np.median([row["q2"] for row in selected])),
            "raw_bias_mm": float(np.mean([row["height_residual_mm"] for row in selected])),
        })

    metric_fields = list(metrics_rows[0].keys())
    write_csv(output / "surface2br2_condition_table.csv", condition_table, list(condition_table[0].keys()))
    write_csv(output / "surface2br2_height_bias.csv", height_stats, list(height_stats[0].keys()))
    write_csv(output / "surface2br2_cv_metrics.csv", metrics_rows, metric_fields)
    write_csv(output / "surface2br2_incremental_comparison.csv", incremental, list(incremental[0].keys()))
    write_csv(output / "surface2br2_coefficients.csv", coefficient_rows, list(coefficient_rows[0].keys()))
    write_csv(output / "surface2br2_coefficient_stability.csv", stability, list(stability[0].keys()))
    write_csv(output / "surface2br2_condition_predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    make_plot(output, incremental)
    summary = {
        "Q_DEPENDENT_SIGNAL": q_signal,
        "HEIGHT_GAP_ACQUISITION_STILL_JUSTIFIED": acquisition,
        "original_surface2b_conclusion": {
            "Q2_GAP_FILLED": provenance["surface2b_summary"]["Q2_GAP_FILLED"],
            "Q1Q2_STATE_CONSISTENCY": provenance["surface2b_summary"]["Q1Q2_STATE_CONSISTENCY"],
            "SURFACE2C_ALLOWED": provenance["surface2b_summary"]["SURFACE2C_ALLOWED"],
        },
        "protocol": {
            "development_height_labels_mm": list(DEV_HEIGHT_ORDER),
            "height_bands": {key: list(value) for key, value in HEIGHT_BANDS.items()},
            "condition_equal_weight": True,
            "random_point_split": False,
            "heldout_50_excluded_from_fit_and_selection": True,
            "c0_refit": False,
            "c1_refit": False,
            "q_redefined": False,
            "quadratic_terms": False,
            "spline_or_ml": False,
            "production_validation": False,
        },
        "provenance": provenance,
        "decision_details": decision,
        "model_formulas": {
            "B0": "a0",
            "B1": "a0+a1*q1",
            "B2": "a0+a2*q2",
            "S0": "a0+a1*q1+a2*q2",
            "corrected_residual": "r-F(q1,q2)",
        },
        "height_bias": height_stats,
        "incremental_comparison": incremental,
        "coefficient_stability": stability,
        "created_at_utc": now_utc(),
    }
    write_json(output / "surface2br2_summary.json", summary)
    (output / "surface2br2_report.md").write_text(
        report_text(provenance, height_stats, incremental, metrics_rows, stability, decision),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "Q_DEPENDENT_SIGNAL": q_signal,
        "HEIGHT_GAP_ACQUISITION_STILL_JUSTIFIED": acquisition,
        "development_condition_count": decision["development_condition_count"],
        "development_point_count": provenance["development_point_count"],
        "s0_positive_increment_folds": f"{len(positive)}/{len(s0_rows)}",
        "s0_pooled_delta_rmse_mm": float(s0_delta),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
