"""Task A-3B: Local-reference edge residual versus Frozen C0 ray geometry.

Analysis-only audit.  It reuses A-3 features, A-13A/A-13B baseline support,
the Frozen Steger cache, Frozen V2 ROI and the existing reconstruction chain.
It derives C0 surface geometry from the Frozen quadratic graph without fitting
or writing any production correction.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "laser_measurement_tool"
TOOLS_ROOT = REPO_ROOT / "tools"
BASE_OUTPUT_ROOT = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze"
A3_OUTPUT_DIR = BASE_OUTPUT_ROOT / "spatial_attribution_local_reference"
OUTPUT_DIR = BASE_OUTPUT_ROOT / "ray_surface_geometry_local_reference"
A13B_BASELINE = REPO_ROOT / "reports" / "experiments" / "daheng_0822" / "session01_roi_freeze" / "session01_a13b_v2_baseline_diagnostics.csv"
A13B_CONDITION_METRICS = REPO_ROOT / "reports" / "experiments" / "daheng_0822" / "session01_roi_freeze" / "session01_a13b_condition_metrics.csv"
A3_FEATURES = A3_OUTPUT_DIR / "a3_spatial_features.csv"
A3_GROUPED_VALIDATION = A3_OUTPUT_DIR / "a3_grouped_validation.csv"
A3_MAP = A3_OUTPUT_DIR / "a3_height_position_residual_map.png"

sys.path.insert(0, str(TOOL_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from audit_local_spatial_attribution import (  # noqa: E402
    A13B_FRAMES,
    A13B_MANIFEST,
    CACHE_NPZ,
    CONFIG_PATH,
    GROUND_PATH,
    REGISTRY_PATH,
    TARGET_FIELDS,
    finite,
    load_runtime,
    mean,
    median,
    parse_bool,
    percentile,
    read_csv,
    reconstruct_uv_to_ground,
    roi_mask,
    write_csv,
    write_json,
)


HEIGHT_ORDER = ("h10", "h20", "h30")
POSITION_ORDER = tuple(f"p{i:02d}" for i in range(1, 11))
EDGE_POSITION = {"p01": "upper", "p10": "lower"}
GEOMETRY_CORRELATION_FIELDS = (
    "ray_surface_normal_angle_deg",
    "abs_c0_intersection_dF_dlambda",
    "camera_ray_optical_axis_angle_deg",
)
VALIDATION_MODELS = {
    "M0": ("q2",),
    "G1": ("q2", "ray_surface_normal_angle_deg"),
    "G2": ("q2", "abs_c0_intersection_dF_dlambda"),
}
VALIDATION_FORMULAS = {
    "M0": "residual ~ q2",
    "G1": "residual ~ q2 + ray_surface_normal_angle_deg",
    "G2": "residual ~ q2 + abs_c0_intersection_dF_dlambda",
}
VALIDATION_METRICS = (
    "rmse_mm",
    "p95_abs_mm",
    "position_bias_range_mm",
    "worst_position_error_mm",
)
TARGET_LABELS = {
    "base": "Local Base residual (mm)",
    "h1": "Local H1 residual (mm)",
    "hb2": "Local H-B2 residual (mm)",
}
COLORS = {"h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}
EPSILON = 1.0e-12


def correlation(x: Iterable[float], y: Iterable[float], method: str) -> tuple[float | None, float | None]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if len(x_array) < 3 or len(x_array) != len(y_array):
        return None, None
    if np.all(x_array == x_array[0]) or np.all(y_array == y_array[0]):
        return None, None
    result = pearsonr(x_array, y_array) if method == "pearson" else spearmanr(x_array, y_array)
    return float(result.statistic), float(result.pvalue)


def abs_percentile(values: Iterable[float], q: float) -> float | None:
    finite_values = [abs(float(value)) for value in values if finite(value) is not None]
    return float(np.percentile(np.asarray(finite_values, dtype=np.float64), q)) if finite_values else None


def condition_repeat_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("condition_id")), str(row.get("repeat_index"))


def axis_index(name: str) -> int:
    mapping = {"X": 0, "Y": 1, "Z": 2}
    try:
        return mapping[str(name).upper()]
    except KeyError as error:
        raise RuntimeError(f"Unsupported C0 axis: {name}") from error


def validate_inputs(
    a3_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    condition_rows: list[dict[str, str]],
) -> None:
    if len(a3_rows) != 600 or len(baseline_rows) != 600:
        raise RuntimeError(f"Expected 600 A-3/baseline rows, got {len(a3_rows)}/{len(baseline_rows)}")
    if len(condition_rows) != 90:
        raise RuntimeError(f"Expected 90 A-13B condition metric rows, got {len(condition_rows)}")
    required_a3 = {"cache_key", "condition_id", "height_label", "position_id", "q2", *TARGET_FIELDS.values()}
    required_baseline = {
        "condition_id",
        "repeat_index",
        "baseline_support_type",
        "local_baseline_support",
        "local_baseline_extrapolation",
    }
    required_condition = {"condition_id", "model", "edge_baseline_clipped"}
    if missing := required_a3 - set(a3_rows[0]):
        raise RuntimeError(f"A-3 feature fields missing: {sorted(missing)}")
    if missing := required_baseline - set(baseline_rows[0]):
        raise RuntimeError(f"A-13A/A-13B baseline fields missing: {sorted(missing)}")
    if missing := required_condition - set(condition_rows[0]):
        raise RuntimeError(f"A-13B condition fields missing: {sorted(missing)}")
    if len({str(row.get("cache_key")) for row in a3_rows}) != 600:
        raise RuntimeError("A-3 cache_key is not unique")
    if len({condition_repeat_key(row) for row in baseline_rows}) != 600:
        raise RuntimeError("Baseline diagnostics condition/repeat key is not unique")
    if len({str(row.get("condition_id")) for row in a3_rows}) != 30:
        raise RuntimeError("A-3 feature rows do not contain 30 conditions")


def build_baseline_maps(
    baseline_rows: list[dict[str, str]],
    condition_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, bool]]:
    baseline_by_key = {condition_repeat_key(row): row for row in baseline_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in condition_rows:
        grouped[str(row["condition_id"])].append(row)
    clipped_by_condition: dict[str, bool] = {}
    for condition_id, rows in grouped.items():
        flags = {parse_bool(row.get("edge_baseline_clipped")) for row in rows}
        models = {str(row.get("model")) for row in rows}
        if flags != {True, False} and len(flags) != 1:
            raise RuntimeError(f"Inconsistent edge_baseline_clipped for {condition_id}: {flags}")
        if not {"base", "h1", "hb2"}.issubset(models):
            raise RuntimeError(f"Missing A-13B model rows for {condition_id}: {models}")
        clipped_by_condition[condition_id] = next(iter(flags))
    if len(clipped_by_condition) != 30:
        raise RuntimeError(f"Expected 30 clipped-condition flags, got {len(clipped_by_condition)}")
    return baseline_by_key, clipped_by_condition


def frozen_c0_geometry(
    points_camera_c0: np.ndarray,
    rays: np.ndarray,
    model: dict[str, Any],
) -> tuple[dict[str, np.ndarray], float]:
    """Evaluate Frozen quadratic-graph geometry at C0 ray intersections.

    For F(P)=D-H(p,q), the gradient is the local surface normal before
    normalization.  ``dot(ray, gradient)`` is dF/dlambda, the true implicit
    ray-surface intersection denominator for this graph.
    """
    if model.get("model_type") != "quadratic_graph":
        raise RuntimeError("A-3B requires Frozen C0 quadratic_graph")
    dep_axis = axis_index(str(model["dependent_axis"]))
    independent_axes = tuple(axis_index(str(axis)) for axis in model["independent_axes"])
    if len(independent_axes) != 2 or dep_axis in independent_axes:
        raise RuntimeError("Frozen C0 axes do not form a quadratic graph")
    normalization = model["normalization"]
    center = np.asarray(normalization["independent_center_mm"], dtype=np.float64).reshape(2)
    scale = np.asarray(normalization["independent_scale_mm"], dtype=np.float64).reshape(2)
    beta = np.asarray(model["coefficients"], dtype=np.float64).reshape(6)
    b0, b1, b2, b3, b4, b5 = beta

    p = (points_camera_c0[:, independent_axes[0]] - center[0]) / scale[0]
    q = (points_camera_c0[:, independent_axes[1]] - center[1]) / scale[1]
    dH_dind0 = (b1 + 2.0 * b3 * p + b4 * q) / scale[0]
    dH_dind1 = (b2 + b4 * p + 2.0 * b5 * q) / scale[1]
    gradient = np.zeros_like(points_camera_c0, dtype=np.float64)
    gradient[:, dep_axis] = 1.0
    gradient[:, independent_axes[0]] = -dH_dind0
    gradient[:, independent_axes[1]] = -dH_dind1
    gradient_norm = np.linalg.norm(gradient, axis=1)
    normal = gradient / np.maximum(gradient_norm[:, None], EPSILON)

    ray_norm = np.linalg.norm(rays, axis=1)
    ray_unit = rays / np.maximum(ray_norm[:, None], EPSILON)
    ray_normal_abs_cos = np.clip(np.abs(np.einsum("ij,ij->i", ray_unit, normal)), 0.0, 1.0)
    ray_surface_angle = np.degrees(np.arccos(ray_normal_abs_cos))
    optical_angle = np.degrees(np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2]))
    denominator = np.einsum("ij,ij->i", rays, gradient)
    lambda_c0 = points_camera_c0[:, 2] / rays[:, 2]

    # Reproduce the quadratic expansion used by reconstructor.py as a
    # numerical cross-check of the implicit-gradient denominator.
    rp = rays[:, independent_axes[0]]
    rq = rays[:, independent_axes[1]]
    rd = rays[:, dep_axis]
    ap = rp / scale[0]
    aq = rq / scale[1]
    bp = -center[0] / scale[0]
    bq = -center[1] / scale[1]
    quad_rhs = b3 * ap * ap + b4 * ap * aq + b5 * aq * aq
    linear_rhs = b1 * ap + b2 * aq + 2.0 * b3 * ap * bp + b4 * (ap * bq + aq * bp) + 2.0 * b5 * aq * bq
    aa = -quad_rhs
    bb = rd - linear_rhs
    denominator_expanded = 2.0 * aa * lambda_c0 + bb
    denominator_delta = float(np.max(np.abs(denominator - denominator_expanded))) if len(denominator) else 0.0

    values = {
        "c0_lambda_mm": lambda_c0,
        "c0_surface_x_mm": points_camera_c0[:, 0],
        "c0_surface_y_mm": points_camera_c0[:, 1],
        "c0_surface_z_mm": points_camera_c0[:, 2],
        "c0_surface_normal_x": normal[:, 0],
        "c0_surface_normal_y": normal[:, 1],
        "c0_surface_normal_z": normal[:, 2],
        "c0_surface_gradient_norm": gradient_norm,
        "ray_surface_normal_abs_cos": ray_normal_abs_cos,
        "ray_surface_normal_angle_deg": ray_surface_angle,
        "c0_intersection_dF_dlambda": denominator,
        "abs_c0_intersection_dF_dlambda": np.abs(denominator),
        "c0_intersection_condition_inv_abs_dF_dlambda": 1.0 / np.maximum(np.abs(denominator), EPSILON),
        "camera_ray_optical_axis_angle_deg": optical_angle,
    }
    return values, denominator_delta


def compute_geometry_rows(
    a3_rows: list[dict[str, str]],
    baseline_by_key: dict[tuple[str, str], dict[str, str]],
    clipped_by_condition: dict[str, bool],
    registry: dict[str, Any],
    app: Any,
    calibration: dict[str, Any],
    centers_by_key: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry_by_condition = {str(entry["condition_id"]): entry for entry in registry["entries"]}
    model = calibration.get("laser_model")
    if not isinstance(model, dict):
        raise RuntimeError("Calibration laser_model is not a mapping")
    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)
    output: list[dict[str, Any]] = []
    denominator_deltas: list[float] = []
    point_counts: list[int] = []
    a3_by_key = {str(row["cache_key"]): row for row in a3_rows}
    for index, key in enumerate(sorted(a3_by_key), start=1):
        a3_row = a3_by_key[key]
        condition_id = str(a3_row["condition_id"])
        roi = registry_by_condition.get(condition_id)
        if roi is None:
            raise RuntimeError(f"Frozen V2 registry missing {condition_id}")
        baseline = baseline_by_key.get(condition_repeat_key(a3_row))
        if baseline is None:
            raise RuntimeError(f"Baseline diagnostic missing for {condition_repeat_key(a3_row)}")
        reconstruction = reconstruct_uv_to_ground(centers_by_key[key], calibration, app.reconstruction)
        pixels = np.asarray(reconstruction.pixels_uv, dtype=np.float64)
        points_c0 = np.asarray(reconstruction.points_camera_c0, dtype=np.float64)
        if len(pixels) == 0 or points_c0.shape != (len(pixels), 3):
            raise RuntimeError(f"C0 reconstruction is incomplete for {key}")
        normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
        height_mask = roi_mask(pixels, [list(roi["height_v_range"])])
        if not np.any(height_mask):
            raise RuntimeError(f"No Frozen V2 height ROI points for {key}")
        geometry, denominator_delta = frozen_c0_geometry(points_c0[height_mask], rays[height_mask], model)
        denominator_deltas.append(denominator_delta)
        point_counts.append(int(np.count_nonzero(height_mask)))
        edge_clipped = clipped_by_condition[condition_id]
        support = str(baseline.get("local_baseline_support", ""))
        strict = (
            not edge_clipped
            and str(baseline.get("baseline_support_type", "")) == "BOTH_SIDES"
            and support == "BOTH_SIDES"
            and not parse_bool(baseline.get("local_baseline_extrapolation"))
        )
        item: dict[str, Any] = {
            "dataset": a3_row.get("dataset"),
            "height_label": a3_row.get("height_label"),
            "position_id": a3_row.get("position_id"),
            "condition_id": a3_row.get("condition_id"),
            "repeat_index": a3_row.get("repeat_index"),
            "cache_key": key,
            "camera_frame_number": a3_row.get("camera_frame_number"),
            "v_order_rank": a3_row.get("v_order_rank"),
            "raw_v": finite(a3_row.get("raw_v")),
            "q2": finite(a3_row.get("q2")),
            "edge_baseline_clipped": edge_clipped,
            "baseline_support_type": baseline.get("baseline_support_type"),
            "local_baseline_support": support,
            "local_baseline_extrapolation": parse_bool(baseline.get("local_baseline_extrapolation")),
            "strict_reference_valid": strict,
            "height_roi_geometry_point_count": int(np.count_nonzero(height_mask)),
            "reconstruction_valid_point_count": int(len(pixels)),
        }
        for field, values in geometry.items():
            item[field] = mean(values)
            item[f"{field}_median"] = median(values)
        for target, field in TARGET_FIELDS.items():
            item[f"residual_{target}_local_diag"] = finite(a3_row.get(field))
        output.append(item)
        if index % 50 == 0:
            print(f"A-3B C0 geometry extraction {index}/{len(a3_rows)}")
    diagnostics = {
        "frame_count": len(output),
        "height_roi_point_count_min": min(point_counts) if point_counts else None,
        "height_roi_point_count_max": max(point_counts) if point_counts else None,
        "quadratic_denominator_crosscheck_max_abs_delta": max(denominator_deltas) if denominator_deltas else None,
        "strict_reference_frame_count": sum(int(bool(row["strict_reference_valid"])) for row in output),
        "strict_reference_condition_count": len({row["condition_id"] for row in output if row["strict_reference_valid"]}),
        "c0_geometry_definition": "F=D-H(p,q), gradient=[dF/dX,dF/dY,dF/dZ], dF/dlambda=dot(ray,gradient); C0 point is pre-C1 intersection point",
        "ray_surface_normal_angle_definition": "acos(abs(dot(unit_camera_ray, unit_C0_surface_normal))) in degrees; 0 deg is normal incidence, 90 deg is grazing",
        "optical_axis_angle_definition": "atan2(sqrt(xn^2+yn^2),1) in degrees; diagnostic camera-ray angle, not laser-plane incidence",
    }
    return output, diagnostics


def condition_mean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    output: list[dict[str, Any]] = []
    scalar_fields = (
        "q2",
        *GEOMETRY_CORRELATION_FIELDS,
        "c0_intersection_condition_inv_abs_dF_dlambda",
        *TARGET_FIELDS.values(),
    )
    for condition_id, group in sorted(grouped.items()):
        first = group[0]
        item: dict[str, Any] = {
            "condition_id": condition_id,
            "height_label": first["height_label"],
            "position_id": first["position_id"],
            "v_order_rank": first.get("v_order_rank"),
            "n_frames": len(group),
        }
        for field in scalar_fields:
            item[field] = mean(row.get(field) for row in group)
        output.append(item)
    return output


def geometry_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for reference, reference_rows in (("all", rows), ("strict", [row for row in rows if row["strict_reference_valid"]])):
        level_rows = condition_mean_rows(reference_rows)
        for scope, scoped in (
            ("pooled", level_rows),
            *[(f"height:{height}", [row for row in level_rows if row["height_label"] == height]) for height in HEIGHT_ORDER],
        ):
            for geometry_field in GEOMETRY_CORRELATION_FIELDS:
                for target, target_field in TARGET_FIELDS.items():
                    pairs = [(finite(row.get(geometry_field)), finite(row.get(target_field))) for row in scoped]
                    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                    pearson_r, pearson_p = correlation((pair[0] for pair in pairs), (pair[1] for pair in pairs), "pearson")
                    spearman_rho, spearman_p = correlation((pair[0] for pair in pairs), (pair[1] for pair in pairs), "spearman")
                    output.append(
                        {
                            "reference": reference,
                            "scope": scope,
                            "geometry": geometry_field,
                            "target": target,
                            "n_conditions": len(pairs),
                            "pearson_r": pearson_r,
                            "pearson_pvalue": pearson_p,
                            "spearman_rho": spearman_rho,
                            "spearman_pvalue": spearman_p,
                        }
                    )
    return output


def metrics_from_errors(rows: list[dict[str, Any]], errors: np.ndarray) -> dict[str, Any]:
    per_position: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, errors, strict=True):
        per_position[str(row["position_id"])].append(float(error))
    position_bias = {key: float(np.mean(values)) for key, values in per_position.items() if values}
    bias_values = list(position_bias.values())
    return {
        "bias_mm": float(np.mean(errors)),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "p95_abs_mm": float(np.percentile(np.abs(errors), 95.0)),
        "max_abs_mm": float(np.max(np.abs(errors))),
        "position_bias_range_mm": float(max(bias_values) - min(bias_values)) if bias_values else None,
        "position_bias_std_mm": float(np.std(bias_values, ddof=0)) if bias_values else None,
        "worst_position_error_mm": float(max(abs(value) for value in bias_values)) if bias_values else None,
    }


def valid_model_rows(rows: list[dict[str, Any]], target_field: str, predictors: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if finite(row.get(target_field)) is not None and all(finite(row.get(predictor)) is not None for predictor in predictors)
    ]


def fit_standardized(train_rows: list[dict[str, Any]], target_field: str, predictors: tuple[str, ...]) -> dict[str, Any]:
    raw = np.asarray([[float(row[predictor]) for predictor in predictors] for row in train_rows], dtype=np.float64)
    center = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0, ddof=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    design = np.column_stack([np.ones(len(train_rows)), (raw - center) / scale])
    target = np.asarray([float(row[target_field]) for row in train_rows], dtype=np.float64)
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    return {"predictors": predictors, "center": center, "scale": scale, "coef": coef, "rank": int(rank)}


def predict(fit: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    raw = np.asarray([[float(row[predictor]) for predictor in fit["predictors"]] for row in rows], dtype=np.float64)
    design = np.column_stack([np.ones(len(rows)), (raw - fit["center"]) / fit["scale"]])
    return design @ fit["coef"]


def grouped_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for reference, reference_rows in (("all", rows), ("strict", [row for row in rows if row["strict_reference_valid"]])):
        for scheme, group_field, heldout_groups in (
            ("LOHO", "height_label", HEIGHT_ORDER),
            ("LOPO", "position_id", POSITION_ORDER),
        ):
            for target, target_field in TARGET_FIELDS.items():
                for model, predictors in VALIDATION_MODELS.items():
                    all_test_rows: list[dict[str, Any]] = []
                    all_errors: list[float] = []
                    for heldout in heldout_groups:
                        train = [row for row in reference_rows if str(row[group_field]) != heldout]
                        test = [row for row in reference_rows if str(row[group_field]) == heldout]
                        train = valid_model_rows(train, target_field, predictors)
                        test = valid_model_rows(test, target_field, predictors)
                        base = {
                            "reference": reference,
                            "validation_scheme": scheme,
                            "heldout_group": heldout,
                            "model": model,
                            "formula": VALIDATION_FORMULAS[model],
                            "target": target,
                            "train_n": len(train),
                            "test_n": len(test),
                        }
                        if len(train) < len(predictors) + 2 or not test:
                            base.update({"status": "NO_TEST", **{field: None for field in ("bias_mm", *VALIDATION_METRICS, "max_abs_mm")}})
                            output.append(base)
                            continue
                        fit = fit_standardized(train, target_field, predictors)
                        actual = np.asarray([float(row[target_field]) for row in test], dtype=np.float64)
                        errors = actual - predict(fit, test)
                        metrics = metrics_from_errors(test, errors)
                        base.update({"status": "FOLD", **metrics})
                        output.append(base)
                        all_test_rows.extend(test)
                        all_errors.extend(float(error) for error in errors)
                    if all_test_rows:
                        aggregate = metrics_from_errors(all_test_rows, np.asarray(all_errors, dtype=np.float64))
                        output.append(
                            {
                                "reference": reference,
                                "validation_scheme": scheme,
                                "heldout_group": "ALL_HELDOUT",
                                "model": model,
                                "formula": VALIDATION_FORMULAS[model],
                                "target": target,
                                "train_n": None,
                                "test_n": len(all_test_rows),
                                "status": "AGGREGATE",
                                **aggregate,
                            }
                        )
    return output


def lookup(rows: list[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    return next((row for row in rows if all(row.get(key) == value for key, value in keys.items())), {})


def aggregate_model_stable(validation_rows: list[dict[str, Any]], reference: str, target: str, model: str) -> dict[str, Any]:
    baseline = {scheme: lookup(validation_rows, reference=reference, validation_scheme=scheme, heldout_group="ALL_HELDOUT", model="M0", target=target) for scheme in ("LOHO", "LOPO")}
    result: dict[str, Any] = {}
    statuses: list[bool] = []
    for scheme in ("LOHO", "LOPO"):
        candidate = lookup(validation_rows, reference=reference, validation_scheme=scheme, heldout_group="ALL_HELDOUT", model=model, target=target)
        base = baseline[scheme]
        deltas = {field: (float(candidate[field]) - float(base[field]) if candidate.get(field) is not None and base.get(field) is not None else None) for field in VALIDATION_METRICS}
        improved = sum(int(value is not None and value <= 0.0) for value in deltas.values())
        acceptable = all(value is None or value <= 0.01 * max(abs(float(base[field])), 1.0e-9) for field, value in deltas.items())
        stable = bool(acceptable and improved >= 3)
        statuses.append(stable)
        result[f"{scheme.lower()}_stable"] = stable
        result[f"{scheme.lower()}_deltas"] = deltas
    result["stable_both"] = bool(all(statuses))
    return result


def edge_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    output: list[dict[str, Any]] = []
    for condition_id, group in sorted(grouped.items()):
        first = group[0]
        strict_group = [row for row in group if row["strict_reference_valid"]]
        support_types = sorted({str(row["baseline_support_type"]) for row in group})
        local_support_types = sorted({str(row["local_baseline_support"]) for row in group})
        item: dict[str, Any] = {
            "condition_id": condition_id,
            "height_label": first["height_label"],
            "position_id": first["position_id"],
            "edge_side": EDGE_POSITION.get(str(first["position_id"]), "interior"),
            "v_order_rank": first.get("v_order_rank"),
            "edge_baseline_clipped": first["edge_baseline_clipped"],
            "baseline_support_types": ";".join(support_types),
            "local_baseline_support_types": ";".join(local_support_types),
            "local_baseline_extrapolation_any": any(bool(row["local_baseline_extrapolation"]) for row in group),
            "n_total_frames": len(group),
            "n_strict_frames": len(strict_group),
            "strict_reference_available": bool(strict_group),
        }
        for target in TARGET_FIELDS:
            field = f"residual_{target}_local_diag"
            item[f"all_{target}_mean_mm"] = mean(row.get(field) for row in group)
            item[f"all_{target}_p95_abs_mm"] = abs_percentile((row.get(field) for row in group), 95.0)
            item[f"strict_{target}_mean_mm"] = mean(row.get(field) for row in strict_group)
            item[f"strict_{target}_p95_abs_mm"] = abs_percentile((row.get(field) for row in strict_group), 95.0)
        output.append(item)
    return output


def plot_edge_map(audit_rows: list[dict[str, Any]], path: Path) -> None:
    by_condition = {str(row["condition_id"]): row for row in audit_rows}
    all_values = [finite(row.get(f"all_{target}_mean_mm")) for row in audit_rows for target in TARGET_FIELDS]
    strict_values = [finite(row.get(f"strict_{target}_mean_mm")) for row in audit_rows for target in TARGET_FIELDS]
    finite_values = [abs(value) for value in (*all_values, *strict_values) if value is not None]
    limit = max(finite_values) if finite_values else 0.1
    limit = max(limit, 0.05)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    for row_index, target in enumerate(TARGET_FIELDS):
        for col_index, reference in enumerate(("all", "strict")):
            data = np.full((3, 10), np.nan, dtype=np.float64)
            for h_index, height in enumerate(HEIGHT_ORDER):
                for p_index, position in enumerate(POSITION_ORDER):
                    item = by_condition.get(f"{height}_{position}")
                    if item is None:
                        continue
                    value = finite(item.get(f"{reference}_{target}_mean_mm"))
                    if value is not None:
                        data[h_index, p_index] = value
            axis = axes[row_index, col_index]
            image = axis.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
            for h_index, height in enumerate(HEIGHT_ORDER):
                for p_index, position in enumerate(POSITION_ORDER):
                    item = by_condition.get(f"{height}_{position}")
                    if item is None:
                        continue
                    value = finite(item.get(f"{reference}_{target}_mean_mm"))
                    if value is None:
                        text = "—"
                    else:
                        text = f"{value:+.2f}"
                    if reference == "all" and bool(item["edge_baseline_clipped"]):
                        text += "\nC"
                    axis.text(p_index, h_index, text, ha="center", va="center", fontsize=8)
            axis.set_title(f"{target.upper()} · {'all condition' if reference == 'all' else 'strict double-side, non-clipped'}")
            axis.set_xticks(range(10), POSITION_ORDER, rotation=45)
            axis.set_yticks(range(3), HEIGHT_ORDER)
            axis.grid(False)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="residual (mm)")
    fig.suptitle("A-3B Local residual: all conditions versus strict reference", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_geometry(rows: list[dict[str, Any]], path: Path) -> None:
    all_mean = {row["condition_id"]: row for row in condition_mean_rows(rows)}
    strict_mean = {row["condition_id"]: row for row in condition_mean_rows([row for row in rows if row["strict_reference_valid"]])}
    plot_fields = (
        "ray_surface_normal_angle_deg",
        "abs_c0_intersection_dF_dlambda",
        "camera_ray_optical_axis_angle_deg",
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    for row_index, target in enumerate(("h1", "hb2")):
        target_field = TARGET_FIELDS[target]
        for col_index, geometry_field in enumerate(plot_fields):
            axis = axes[row_index, col_index]
            for height in HEIGHT_ORDER:
                all_items = [item for item in all_mean.values() if item["height_label"] == height]
                strict_items = [item for item in strict_mean.values() if item["height_label"] == height]
                x_all = [finite(item.get(geometry_field)) for item in all_items]
                y_all = [finite(item.get(target_field)) for item in all_items]
                pairs_all = [(x, y) for x, y in zip(x_all, y_all, strict=True) if x is not None and y is not None]
                if pairs_all:
                    axis.scatter([pair[0] for pair in pairs_all], [pair[1] for pair in pairs_all], color=COLORS[height], alpha=0.75, s=34, label=height if col_index == 0 and row_index == 0 else None)
                x_strict = [finite(item.get(geometry_field)) for item in strict_items]
                y_strict = [finite(item.get(target_field)) for item in strict_items]
                pairs_strict = [(x, y) for x, y in zip(x_strict, y_strict, strict=True) if x is not None and y is not None]
                if pairs_strict:
                    axis.scatter([pair[0] for pair in pairs_strict], [pair[1] for pair in pairs_strict], facecolors="none", edgecolors=COLORS[height], marker="s", s=48, linewidths=1.0)
            axis.axhline(0.0, color="#777777", linewidth=0.7)
            axis.set_xlabel(geometry_field.replace("_", " "))
            axis.set_ylabel(TARGET_LABELS[target])
            axis.set_title(f"{target.upper()} residual vs {geometry_field}")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("A-3B Local residual versus real C0 ray–surface geometry\nfilled: all conditions; open squares: strict reference", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def consistent_height_pattern(corr_rows: list[dict[str, Any]], reference: str, target: str, geometry: str) -> bool:
    values = [
        finite(lookup(corr_rows, reference=reference, scope=f"height:{height}", geometry=geometry, target=target).get("pearson_r"))
        for height in HEIGHT_ORDER
    ]
    if any(value is None or abs(value) < 0.25 for value in values):
        return False
    return len({value > 0.0 for value in values}) == 1


def edge_survival(audit_rows: list[dict[str, Any]], position: str) -> str:
    edge_rows = [row for row in audit_rows if row["position_id"] == position]
    strict_available = [row for row in edge_rows if int(row["n_strict_frames"]) > 0]
    if not strict_available:
        return "PARTIAL"
    negative_heights = 0
    for row in strict_available:
        values = [finite(row.get(f"strict_{target}_mean_mm")) for target in ("h1", "hb2")]
        if any(value is not None and value < 0.0 for value in values):
            negative_heights += 1
    if negative_heights >= 2 and len(strict_available) >= 2:
        return "YES"
    if negative_heights == 0:
        return "NO"
    return "PARTIAL"


def derive_decisions(
    audit_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    upper = edge_survival(audit_rows, "p01")
    lower = edge_survival(audit_rows, "p10")
    pattern_all = any(consistent_height_pattern(corr_rows, "all", target, geometry) for target in ("h1", "hb2") for geometry in ("ray_surface_normal_angle_deg", "abs_c0_intersection_dF_dlambda"))
    pattern_strict = any(consistent_height_pattern(corr_rows, "strict", target, geometry) for target in ("h1", "hb2") for geometry in ("ray_surface_normal_angle_deg", "abs_c0_intersection_dF_dlambda"))
    validation_all = any(aggregate_model_stable(validation_rows, "all", target, model)["stable_both"] for target in ("h1", "hb2") for model in ("G1", "G2"))
    validation_strict = any(aggregate_model_stable(validation_rows, "strict", target, model)["stable_both"] for target in ("h1", "hb2") for model in ("G1", "G2"))
    if upper == "YES" and lower == "YES" and pattern_all and pattern_strict and validation_all and validation_strict:
        geometry_explains = "YES"
    elif pattern_all or pattern_strict or validation_all or validation_strict:
        geometry_explains = "PARTIAL"
    else:
        geometry_explains = "NO"
    return {
        "UPPER_EDGE_RESIDUAL_SURVIVES_STRICT_REFERENCE": upper,
        "LOWER_EDGE_RESIDUAL_SURVIVES_STRICT_REFERENCE": lower,
        "RAY_SURFACE_GEOMETRY_EXPLAINS_EDGE": geometry_explains,
        "TRUE_TWO_EDGE_EFFECT_SUPPORTED": "YES" if upper == "YES" and lower == "YES" else "NO",
        "pattern_all": pattern_all,
        "pattern_strict": pattern_strict,
        "validation_all": validation_all,
        "validation_strict": validation_strict,
    }


def report_text(
    audit_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    decisions: dict[str, Any],
    a3_grouped_rows: list[dict[str, str]],
) -> str:
    clipped = [row for row in audit_rows if row["edge_baseline_clipped"]]
    one_side = [row for row in audit_rows if "ONE_SIDE" in str(row["local_baseline_support_types"])]
    lines = [
        "# Task A-3B｜FOV edge residual 与真实 ray–surface 几何归因",
        "",
        *[f"- `{key} = {value}`" for key, value in decisions.items() if key.isupper()],
        "",
        "本轮 target 仍仅使用 A-13B-v2 的 `residual_base_local_diag`、`residual_h1_local_diag`、`residual_hb2_local_diag`。Local-reference 只是 diagnostic reference，不是 production truth。",
        "",
        "## Provenance / reference audit",
        "",
        f"- A-3 feature frames=`{sum(int(row['n_total_frames']) for row in audit_rows)}`；A-3B condition audit rows=`{len(audit_rows)}`；A-3 grouped-validation input rows=`{len(a3_grouped_rows)}`；Frozen cache 与正式 reconstruction 链路复用，未读取 PNG、未运行 Steger、未拟合 C0/C1/H1/H-B2。",
        f"- A-13A/A-13B baseline diagnostics：BOTH_SIDES=`{sum(int(row['local_baseline_support_types'] == 'BOTH_SIDES') for row in audit_rows) * 20}` frames，ONE_SIDE=`{sum(int('ONE_SIDE' in str(row['local_baseline_support_types'])) for row in audit_rows) * 20}` frames。",
        f"- `edge_baseline_clipped=True` conditions=`{len(clipped)}`：`{','.join(row['condition_id'] for row in clipped)}`；p01/p10 三个高度均 clipped。",
        f"- ONE_SIDE conditions=`{len(one_side)}`：`{','.join(row['condition_id'] for row in one_side)}`；这些帧也位于 clipped 集合内。",
        f"- strict 定义：`edge_baseline_clipped=False AND baseline_support_type=BOTH_SIDES AND local_baseline_support=BOTH_SIDES AND local_baseline_extrapolation=False`；strict frames=`{diagnostics['strict_reference_frame_count']}`/600，strict conditions=`{diagnostics['strict_reference_condition_count']}`/30；p01/p10 strict frames 均为 0。",
        "- 实际逐条件字段名为 `edge_baseline_clipped`；`FROZEN_EDGE_BASELINE_CLIPPED` 不是本数据的逐条件枚举名。",
        "",
        "## C0 ray–surface geometry",
        "",
        "- Frozen C0 是 `D=H(p,q)` quadratic graph；在 C0 intersection point 计算 `F=D-H(p,q)` 的 local gradient，并归一化为 camera-frame surface normal。",
        "- `c0_intersection_dF_dlambda = dot(camera_ray, grad(F))` 是隐式 ray–surface intersection denominator；其绝对值越小，交会越接近 grazing/conditioning 较差。quadratic expansion cross-check 最大差=`" + fmt(diagnostics["quadratic_denominator_crosscheck_max_abs_delta"], 10) + "`。",
        "- `ray_surface_normal_angle_deg=acos(abs(dot(unit_ray,unit_normal)))`；`camera_ray_optical_axis_angle_deg` 仍只是 optical-axis proxy，未当作 laser incidence angle。",
        "- C0 geometry 使用 `points_camera_c0`（C1 前的 Frozen C0 intersection）；C1 仅随正式 reconstruction 用于最终有效性，不改变本轮法向定义。",
        "",
        "## Edge reference audit",
        "",
        "| condition | clipped | support | strict n | all H1 | strict H1 | all HB2 | strict HB2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        if row["position_id"] in {"p01", "p02", "p09", "p10"}:
            lines.append(
                f"| {row['condition_id']} | {row['edge_baseline_clipped']} | {row['local_baseline_support_types']} | {row['n_strict_frames']} | {fmt(row['all_h1_mean_mm'])} | {fmt(row['strict_h1_mean_mm'])} | {fmt(row['all_hb2_mean_mm'])} | {fmt(row['strict_hb2_mean_mm'])} |"
            )
    base_upper = [lookup(audit_rows, condition_id=f"{height}_p01").get("all_base_mean_mm") for height in HEIGHT_ORDER]
    base_lower = [lookup(audit_rows, condition_id=f"{height}_p10").get("all_base_mean_mm") for height in HEIGHT_ORDER]
    lines += [
        "",
        f"Base all-condition p01 means (h10/h20/h30)=`{','.join(fmt(value) for value in base_upper)} mm`；p10=`{','.join(fmt(value) for value in base_lower)} mm`。",
        "p01/p10 没有 strict non-clipped condition，因此其 all-condition 大负 residual 不能在本数据内被称为真实双边缘效应；strict map 中的空白是 reference 不可评估，不是 residual 被证明为 0。",
        "",
        "## Condition-mean geometry correlations",
        "",
        "| reference | target | geometry | h10 Pearson | h20 | h30 | pooled Spearman |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for reference in ("all", "strict"):
        for target in ("h1", "hb2"):
            for geometry in ("ray_surface_normal_angle_deg", "abs_c0_intersection_dF_dlambda"):
                h_values = [lookup(corr_rows, reference=reference, scope=f"height:{height}", geometry=geometry, target=target) for height in HEIGHT_ORDER]
                pooled = lookup(corr_rows, reference=reference, scope="pooled", geometry=geometry, target=target)
                lines.append(
                    f"| {reference} | {target} | {geometry} | {fmt(h_values[0].get('pearson_r'))} | {fmt(h_values[1].get('pearson_r'))} | {fmt(h_values[2].get('pearson_r'))} | {fmt(pooled.get('spearman_rho'))} |"
                )
    lines += [
        "",
        "## LOHO / LOPO read-only explanatory validation",
        "",
        "G1/G2 是预先指定的物理几何候选；M0 只含现有 q2 baseline。所有模型均为 diagnostic fit，未写入 production。strict LOPO 的 p01/p10 fold 没有 test rows，aggregate 只覆盖仍有 strict reference 的 positions。",
        "",
        "| reference | target | scheme | model | n test | RMSE | P95 | pos-range | worst-position |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for reference in ("all", "strict"):
        for target in ("h1", "hb2"):
            for scheme in ("LOHO", "LOPO"):
                for model in ("M0", "G1", "G2"):
                    row = lookup(validation_rows, reference=reference, target=target, validation_scheme=scheme, heldout_group="ALL_HELDOUT", model=model)
                    lines.append(
                        f"| {reference} | {target} | {scheme} | {model} | {row.get('test_n', 'NA')} | {fmt(row.get('rmse_mm'))} | {fmt(row.get('p95_abs_mm'))} | {fmt(row.get('position_bias_range_mm'))} | {fmt(row.get('worst_position_error_mm'))} |"
                    )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Upper p01 strict survival=`{decisions['UPPER_EDGE_RESIDUAL_SURVIVES_STRICT_REFERENCE']}`；lower p10 strict survival=`{decisions['LOWER_EDGE_RESIDUAL_SURVIVES_STRICT_REFERENCE']}`。两端均因 clipped 而无 strict non-clipped observations，属于不可证实的 PARTIAL，而不是 YES。",
        f"- `RAY_SURFACE_GEOMETRY_EXPLAINS_EDGE={decisions['RAY_SURFACE_GEOMETRY_EXPLAINS_EDGE']}`：真实几何变量可对部分 condition/interior pattern 提供 diagnostic 解释时最多判 PARTIAL；当前没有足够的严格边缘 reference 支持 YES。",
        f"- `TRUE_TWO_EDGE_EFFECT_SUPPORTED={decisions['TRUE_TWO_EDGE_EFFECT_SUPPORTED']}`：p01 与 p10 都没有 strict non-clipped 数据，不能把两端 all-condition pattern 归因为真实双边缘几何。",
        "",
        "## Boundaries",
        "",
        "- 不重新拟合 C0/C1/H1/H-B2；不修改 ROI、Ground 或 production reconstruction。",
        "- Local-reference 仍是 diagnostic；baseline clipped/one-side artifact 与 reconstruction geometry residual 分开统计。",
        "- 所有相关和 LOHO/LOPO fit 仅用于归因审计，不是 correction freeze 或 deployment evidence。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    required = [
        A13B_FRAMES,
        A13B_MANIFEST,
        CACHE_NPZ,
        REGISTRY_PATH,
        CONFIG_PATH,
        GROUND_PATH,
        A13B_BASELINE,
        A13B_CONDITION_METRICS,
        A3_FEATURES,
        A3_GROUPED_VALIDATION,
        A3_MAP,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"A-3B required artifact missing: {missing}")
    a3_rows = read_csv(A3_FEATURES)
    baseline_rows = read_csv(A13B_BASELINE)
    condition_rows = read_csv(A13B_CONDITION_METRICS)
    a3_grouped_rows = read_csv(A3_GROUPED_VALIDATION)
    validate_inputs(a3_rows, baseline_rows, condition_rows)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("dataset") != "session01" or registry.get("frozen") is not True or registry.get("manual_confirmed") is not True or len(registry.get("entries", [])) != 30:
        raise RuntimeError("Frozen V2 ROI registry provenance is invalid")
    baseline_by_key, clipped_by_condition = build_baseline_maps(baseline_rows, condition_rows)
    app, calibration, ground_payload, centers_by_key, cache_manifest, cache_info = load_runtime()
    a3_keys = {str(row["cache_key"]) for row in a3_rows}
    if a3_keys != set(centers_by_key):
        raise RuntimeError("A-3/Frozen cache key mismatch")
    geometry_rows, diagnostics = compute_geometry_rows(a3_rows, baseline_by_key, clipped_by_condition, registry, app, calibration, centers_by_key)
    audit_rows = edge_audit_rows(geometry_rows)
    corr_rows = geometry_correlations(geometry_rows)
    validation_rows = grouped_validation(geometry_rows)
    decisions = derive_decisions(audit_rows, corr_rows, validation_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "a3b_edge_reference_audit.csv", audit_rows, list(audit_rows[0].keys()))
    write_csv(OUTPUT_DIR / "a3b_ray_surface_geometry.csv", geometry_rows, list(geometry_rows[0].keys()))
    write_csv(OUTPUT_DIR / "a3b_geometry_correlations.csv", corr_rows, list(corr_rows[0].keys()))
    write_csv(OUTPUT_DIR / "a3b_geometry_validation.csv", validation_rows, list(validation_rows[0].keys()))
    plot_edge_map(audit_rows, OUTPUT_DIR / "a3b_strict_edge_residual_map.png")
    plot_geometry(geometry_rows, OUTPUT_DIR / "a3b_residual_vs_ray_surface_geometry.png")
    provenance = {
        "task": "A-3B Local-reference edge residual versus real C0 ray-surface geometry",
        "inputs": {
            "a3_features": str(A3_FEATURES),
            "a3_grouped_validation": str(A3_GROUPED_VALIDATION),
            "a3_map": str(A3_MAP),
            "a13b_frames": str(A13B_FRAMES),
            "a13b_baseline_diagnostics": str(A13B_BASELINE),
            "a13b_condition_metrics": str(A13B_CONDITION_METRICS),
            "frozen_cache": str(CACHE_NPZ),
            "frozen_v2_registry": str(REGISTRY_PATH),
            "config": str(CONFIG_PATH),
            "session_ground": str(GROUND_PATH),
        },
        "reuse": {
            "png_read": False,
            "steger_rerun": False,
            "c0_c1_refit": False,
            "production_reconstruction_modified": False,
            "local_reference_is_diagnostic_only": True,
        },
        "strict_reference_definition": "edge_baseline_clipped=False AND baseline_support_type=BOTH_SIDES AND local_baseline_support=BOTH_SIDES AND local_baseline_extrapolation=False",
        "baseline_counts": {
            "frames_total": len(baseline_rows),
            "conditions_total": len(clipped_by_condition),
            "clipped_conditions": [condition for condition, clipped in clipped_by_condition.items() if clipped],
            "strict_frames": diagnostics["strict_reference_frame_count"],
            "strict_conditions": diagnostics["strict_reference_condition_count"],
        },
        "geometry": {
            "c0_model": str(calibration.get("laser_model", {}).get("model_type")),
            "c0_denominator": "dot(ray, grad(F)) = dF/dlambda",
            "c0_surface_normal": "normalized grad(F), F=D-H(p,q), camera frame",
            "ray_surface_normal_angle": "acos(abs(dot(unit_ray, unit_normal)))",
            "optical_axis_angle_not_incidence": True,
        },
        "diagnostics": diagnostics,
        "decisions": decisions,
        "cache_info": cache_info,
        "ground_status": ground_payload.get("status"),
    }
    write_json(OUTPUT_DIR / "a3b_provenance.json", provenance)
    (OUTPUT_DIR / "report.md").write_text(report_text(audit_rows, corr_rows, validation_rows, diagnostics, decisions, a3_grouped_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "decisions": decisions, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
