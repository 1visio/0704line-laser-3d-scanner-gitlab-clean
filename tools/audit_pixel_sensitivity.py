#!/usr/bin/env python3
"""Task A-1: pointwise pixel-to-Base-height sensitivity audit.

This audit reuses the frozen Session01 Steger centers and ROI registry, then
calls the production ``reconstruct_uv_to_ground`` chain with the current
Frozen-C0/Frozen-C1 configuration and Session Ground reference.  It does not
fit or modify any calibration/correction model.

The audited pointwise Base height is the final Session-Ground-levelled Zg of
one measurement-ROI point.  For epsilon in {0.05, 0.10} px::

    dh/du = (h(u + epsilon, v) - h(u - epsilon, v)) / (2 * epsilon)

The five vectorized reconstruction calls (unperturbed, +/-0.05, +/-0.10)
are mathematically pointwise: no point uses another point's value and no
height-line fit is performed.  The vectorization only avoids Python call
overhead while preserving the formal reconstruction function and parameters.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "laser_measurement_tool"
SESSION_OUTPUT = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze"
DEFAULT_OUTPUT_DIR = SESSION_OUTPUT / "pixel_sensitivity_audit"
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_CACHE = SESSION_OUTPUT / "session01_steger_centers.npz"
DEFAULT_CACHE_MANIFEST = SESSION_OUTPUT / "session01_steger_centers_manifest.json"
DEFAULT_REGISTRY = SESSION_OUTPUT / "session01_roi_registry_manual_v2.json"
DEFAULT_GROUND = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0822\session01\session_ground_calibration.json"
)

HEIGHT_VALUES = {"h10": 10.0, "h20": 20.0, "h30": 30.0}
HEIGHT_ORDER = ("h10", "h20", "h30")
POSITION_ORDER = tuple(f"p{i:02d}" for i in range(1, 11))
V_BANDS = (
    ("v<1800", lambda value: value < 1800.0),
    ("1800<=v<2200", lambda value: 1800.0 <= value < 2200.0),
    ("2200<=v<2400", lambda value: 2200.0 <= value < 2400.0),
    ("2400<=v<=2600", lambda value: 2400.0 <= value <= 2600.0),
    ("v>2600", lambda value: value > 2600.0),
)
V_BAND_LABELS = tuple(item[0] for item in V_BANDS)
EPSILONS = (0.05, 0.10)

sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from measurement.ground_reference import SessionGroundReference  # noqa: E402
from reconstruction.reconstructor import (  # noqa: E402
    ReconstructionInputError,
    reconstruct_uv_to_ground,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def load_frozen_cache(
    cache_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = list(manifest.get("frames", []))
    with np.load(cache_path, allow_pickle=False) as bundle:
        centers = np.asarray(bundle["centers_full"], dtype=np.float64)
        offsets = np.asarray(bundle["frame_offsets"], dtype=np.int64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise RuntimeError("Frozen Steger centers_full must have shape (N, 2)")
    if offsets.ndim != 1 or len(offsets) != len(frames) + 1:
        raise RuntimeError("Frozen Steger frame_offsets shape is invalid")
    if len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != len(centers):
        raise RuntimeError("Frozen Steger frame_offsets bounds are invalid")
    centers_by_key: dict[str, np.ndarray] = {}
    for index, frame in enumerate(frames):
        start, end = int(offsets[index]), int(offsets[index + 1])
        frame_centers = np.ascontiguousarray(centers[start:end], dtype=np.float64)
        if not len(frame_centers) or not np.isfinite(frame_centers).all():
            raise RuntimeError(f"Frozen Steger frame has unusable centers: {index}")
        centers_by_key[str(frame["cache_key"])] = frame_centers
    info = {
        "frames_total": len(frames),
        "centers_total": int(len(centers)),
        "frame_offsets_shape": list(offsets.shape),
        "one_steger_per_frame": manifest.get("one_steger_per_frame"),
        "reused_existing_cache": True,
        "manifest_protocol_key": manifest.get("protocol_key", {}),
    }
    return frames, centers_by_key, info


def roi_mask(points_uv: np.ndarray, ranges: list[list[float]]) -> np.ndarray:
    """Use the same inclusive (+0.5 pixel) formal ROI rule as A-13B-v2."""
    points = np.asarray(points_uv, dtype=np.float64)
    mask = np.zeros(len(points), dtype=bool)
    if not len(points):
        return mask
    point_u = points[:, 0] + 0.5
    point_v = points[:, 1] + 0.5
    for top, bottom in ranges:
        mask |= (
            (point_u >= 0.0)
            & (point_u <= 4096.0)
            & (point_v >= float(top))
            & (point_v <= float(bottom))
        )
    return mask


def load_session_ground(path: Path) -> tuple[dict[str, Any], SessionGroundReference]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "VALID" or payload.get("valid") is not True:
        raise RuntimeError("session_ground_calibration.json is not VALID")
    if payload.get("runtime", {}).get("ground_extrinsic_source") != "session":
        raise RuntimeError("session ground runtime source is not session")
    ground = payload.get("session_ground_reference", {})
    if ground.get("status") != "VALID":
        raise RuntimeError("Session Ground Reference is not VALID")
    reference = SessionGroundReference(
        origin_xy=np.asarray(ground["origin_xy"], dtype=np.float64),
        direction_xy=np.asarray(ground["direction_xy"], dtype=np.float64),
        slope_z_per_mm=float(ground["slope_z_per_mm"]),
        intercept_z_mm=float(ground["intercept_z_mm"]),
        rmse_mm=float(ground["rmse_mm"]),
        valid_s_range_mm=tuple(float(item) for item in ground["valid_s_range_mm"]),
        status=str(ground["status"]),
        source=str(ground.get("fit_source", "session_laser_ground")),
        point_count=int(ground.get("point_count", 0)),
        inlier_count=int(ground.get("inlier_count", 0)),
        support_source=str(ground.get("support_source", ground.get("source", ""))),
        active_ground_extrinsic_source=str(
            ground.get("active_ground_extrinsic_source", "session")
        ),
        ground_extrinsic_generation=int(ground.get("ground_extrinsic_generation", 0)),
        frame_host_monotonic_ns=int(ground.get("frame_host_monotonic_ns", 0)),
        mask_inset_mm=float(ground.get("mask_inset_mm", 0.0)),
        support_metadata=dict(ground.get("support", {})),
    )
    return payload, reference


def build_point_metadata(
    frames: list[dict[str, Any]],
    centers_by_key: dict[str, np.ndarray],
    registry: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    entries = registry.get("entries", [])
    if len(entries) != 30 or not registry.get("frozen") or not registry.get("manual_confirmed"):
        raise RuntimeError("Session01 V2 ROI registry is not the frozen 30-entry registry")
    by_condition = {str(entry["condition_id"]): entry for entry in entries}
    all_points: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for frame in frames:
        height_label = str(frame["height_label"])
        position_id = str(frame["position_id"])
        condition_id = f"{height_label}_{position_id}"
        roi = by_condition.get(condition_id)
        if roi is None:
            raise RuntimeError(f"Frozen V2 ROI registry missing {condition_id}")
        centers = centers_by_key[str(frame["cache_key"])]
        selected = roi_mask(centers, [[float(v) for v in roi["height_v_range"]]])
        for point_index in np.flatnonzero(selected):
            point = np.asarray(centers[point_index], dtype=np.float64)
            all_points.append(point)
            metadata.append(
                {
                    "dataset": "session01",
                    "height_label": height_label,
                    "true_height_mm": HEIGHT_VALUES[height_label],
                    "position_id": position_id,
                    "v_order_rank": int(roi.get("v_order_rank", 0)),
                    "condition_id": condition_id,
                    "repeat_index": int(frame.get("repeat_index", 0)),
                    "filename": str(frame.get("filename", "")),
                    "cache_key": str(frame["cache_key"]),
                    "camera_frame_number": frame.get("camera_frame_number"),
                    "point_index": int(point_index),
                    "height_v_range": roi["height_v_range"],
                    "u_px": float(point[0]),
                    "v_px": float(point[1]),
                }
            )
    if not all_points:
        raise RuntimeError("No Steger points fall inside the frozen measurement ROIs")
    return np.asarray(all_points, dtype=np.float64), metadata


def v_band_for(value: float) -> str:
    for label, predicate in V_BANDS:
        if predicate(float(value)):
            return label
    raise RuntimeError(f"v value did not match a band: {value}")


def align_reconstruction(
    input_pixels: np.ndarray,
    returned_pixels: np.ndarray,
) -> np.ndarray:
    """Map returned valid pixels back to input indices without assuming all valid."""
    if len(returned_pixels) == len(input_pixels) and np.allclose(
        returned_pixels, input_pixels, rtol=0.0, atol=1.0e-9
    ):
        return np.arange(len(input_pixels), dtype=np.int64)
    matched: list[int] = []
    used: set[int] = set()
    for pixel in np.asarray(returned_pixels, dtype=np.float64):
        distance = np.max(np.abs(input_pixels - pixel[None, :]), axis=1)
        if used:
            distance[list(used)] = np.inf
        index = int(np.argmin(distance))
        if not math.isfinite(float(distance[index])) or float(distance[index]) > 1.0e-7:
            raise RuntimeError("Could not align reconstruct_uv_to_ground output to input pixels")
        used.add(index)
        matched.append(index)
    return np.asarray(matched, dtype=np.int64)


def evaluate_base_height(
    pixels_uv: np.ndarray,
    calibration: dict[str, Any],
    reconstruction_params: Any,
    ground_reference: SessionGroundReference,
) -> dict[str, np.ndarray | int | str | None]:
    try:
        result = reconstruct_uv_to_ground(pixels_uv, calibration, reconstruction_params)
    except (ReconstructionInputError, ValueError, FloatingPointError) as error:
        raise RuntimeError(f"formal reconstruction failed: {type(error).__name__}: {error}") from error
    heights = np.full(len(pixels_uv), np.nan, dtype=np.float64)
    raw_z = np.full(len(pixels_uv), np.nan, dtype=np.float64)
    ground_valid = np.zeros(len(pixels_uv), dtype=bool)
    reconstructed = np.zeros(len(pixels_uv), dtype=bool)
    c1_clamped = np.zeros(len(pixels_uv), dtype=bool)
    indices = align_reconstruction(pixels_uv, result.pixels_uv)
    levelled, valid = ground_reference.apply_to_points(result.points_ground)
    for output_index, input_index in enumerate(indices):
        reconstructed[input_index] = True
        raw_z[input_index] = float(result.points_ground[output_index, 2])
        heights[input_index] = float(levelled[output_index, 2])
        ground_valid[input_index] = bool(valid[output_index])
        if result.c1_clamped is not None:
            c1_clamped[input_index] = bool(result.c1_clamped[output_index])
    return {
        "height": heights,
        "raw_z": raw_z,
        "ground_valid": ground_valid,
        "reconstructed": reconstructed,
        "c1_clamped": c1_clamped,
        "valid_count": int(np.count_nonzero(reconstructed)),
        "filtered": result.filtered,
    }


def build_point_rows(
    points: np.ndarray,
    metadata: list[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    base = evaluations["base"]
    plus005 = evaluations["u_plus_0p05"]
    minus005 = evaluations["u_minus_0p05"]
    plus010 = evaluations["u_plus_0p10"]
    minus010 = evaluations["u_minus_0p10"]
    rows: list[dict[str, Any]] = []
    for index, (point, item) in enumerate(zip(points, metadata, strict=True)):
        values = {
            "base_height_mm": base["height"][index],
            "base_raw_zg_mm": base["raw_z"][index],
            "u_plus_0p05_height_mm": plus005["height"][index],
            "u_minus_0p05_height_mm": minus005["height"][index],
            "u_plus_0p10_height_mm": plus010["height"][index],
            "u_minus_0p10_height_mm": minus010["height"][index],
        }
        sensitivity_values = {
            "dh_du_0p05_mm_per_px": (
                float(values["u_plus_0p05_height_mm"] - values["u_minus_0p05_height_mm"]) / 0.10
                if np.isfinite(values["u_plus_0p05_height_mm"])
                and np.isfinite(values["u_minus_0p05_height_mm"])
                else np.nan
            ),
            "dh_du_0p10_mm_per_px": (
                float(values["u_plus_0p10_height_mm"] - values["u_minus_0p10_height_mm"]) / 0.20
                if np.isfinite(values["u_plus_0p10_height_mm"])
                and np.isfinite(values["u_minus_0p10_height_mm"])
                else np.nan
            ),
        }
        all_reconstruction = all(
            bool(evaluations[name]["reconstructed"][index])
            for name in evaluations
        )
        all_ground_valid = all(
            bool(evaluations[name]["ground_valid"][index])
            for name in evaluations
        )
        finite_sensitivity = all(
            np.isfinite(sensitivity_values[field])
            for field in sensitivity_values
        )
        if not all_reconstruction:
            status = "RECONSTRUCTION_INVALID"
        elif not finite_sensitivity:
            status = "FINITE_DIFFERENCE_INVALID"
        elif not all_ground_valid:
            status = "GROUND_REFERENCE_PARTIAL"
        else:
            status = "OK"
        row = {
            **item,
            "u_px": float(point[0]),
            "v_px": float(point[1]),
            "v_band": v_band_for(float(point[1])),
            **values,
            **sensitivity_values,
            "abs_dh_du_0p05_mm_per_px": abs(float(sensitivity_values["dh_du_0p05_mm_per_px"]))
            if np.isfinite(sensitivity_values["dh_du_0p05_mm_per_px"])
            else np.nan,
            "abs_dh_du_0p10_mm_per_px": abs(float(sensitivity_values["dh_du_0p10_mm_per_px"]))
            if np.isfinite(sensitivity_values["dh_du_0p10_mm_per_px"])
            else np.nan,
            "finite_difference_delta_abs_mm_per_px": abs(
                float(sensitivity_values["dh_du_0p10_mm_per_px"])
                - float(sensitivity_values["dh_du_0p05_mm_per_px"])
            )
            if finite_sensitivity
            else np.nan,
            "base_reconstruction_valid": bool(base["reconstructed"][index]),
            "base_ground_reference_valid": bool(base["ground_valid"][index]),
            "base_c1_clamped": bool(base["c1_clamped"][index]),
            "u_plus_0p05_ground_reference_valid": bool(plus005["ground_valid"][index]),
            "u_minus_0p05_ground_reference_valid": bool(minus005["ground_valid"][index]),
            "u_plus_0p10_ground_reference_valid": bool(plus010["ground_valid"][index]),
            "u_minus_0p10_ground_reference_valid": bool(minus010["ground_valid"][index]),
            "u_plus_0p05_c1_clamped": bool(plus005["c1_clamped"][index]),
            "u_minus_0p05_c1_clamped": bool(minus005["c1_clamped"][index]),
            "u_plus_0p10_c1_clamped": bool(plus010["c1_clamped"][index]),
            "u_minus_0p10_c1_clamped": bool(minus010["c1_clamped"][index]),
            "sensitivity_status": status,
        }
        rows.append(row)
    return rows


def stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {
            "n": 0,
            "median_signed": None,
            "median_abs": None,
            "p95_abs": None,
            "max_abs": None,
            "mean_abs": None,
            "min_signed": None,
            "max_signed": None,
        }
    absolute = np.abs(array)
    return {
        "n": int(len(array)),
        "median_signed": float(np.median(array)),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95.0)),
        "max_abs": float(np.max(absolute)),
        "mean_abs": float(np.mean(absolute)),
        "min_signed": float(np.min(array)),
        "max_signed": float(np.max(array)),
    }


def statistically_eligible(row: dict[str, Any]) -> bool:
    """Keep finite formal outputs; expose, rather than hide, Ground OOD points.

    ``SessionGroundReference.apply_to_points`` is the production behavior: a
    point outside the frozen S domain is returned unchanged and flagged.  It
    must remain visible in the lower-edge audit because otherwise the very
    region under test would be preferentially removed from the statistics.
    """
    return row["sensitivity_status"] in {"OK", "GROUND_REFERENCE_PARTIAL"}


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [row for row in rows if statistically_eligible(row)]
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []

    def add_group(group_type: str, group: str, selected: list[dict[str, Any]]) -> None:
        groups.append((group_type, group, selected))

    for label in V_BAND_LABELS:
        add_group("v_band", label, [row for row in usable if row["v_band"] == label])
    for label in HEIGHT_ORDER:
        add_group("true_height", label, [row for row in usable if row["height_label"] == label])
    for label in POSITION_ORDER:
        add_group("fov_position", label, [row for row in usable if row["position_id"] == label])
    for height in HEIGHT_ORDER:
        for label in V_BAND_LABELS:
            add_group(
                "height_and_v",
                f"{height}:{label}",
                [
                    row
                    for row in usable
                    if row["height_label"] == height and row["v_band"] == label
                ],
            )
    for height in HEIGHT_ORDER:
        for position in POSITION_ORDER:
            add_group(
                "height_and_position",
                f"{height}:{position}",
                [
                    row
                    for row in usable
                    if row["height_label"] == height and row["position_id"] == position
                ],
            )

    output: list[dict[str, Any]] = []
    for group_type, group, selected in groups:
        for epsilon, field in (
            (0.05, "dh_du_0p05_mm_per_px"),
            (0.10, "dh_du_0p10_mm_per_px"),
        ):
            item = stats([float(row[field]) for row in selected])
            output.append(
                {
                    "group_type": group_type,
                    "group": group,
                    "epsilon_px": epsilon,
                    "n": item["n"],
                    "median_dh_du_mm_per_px": item["median_signed"],
                    "median_abs_dh_du_mm_per_px": item["median_abs"],
                    "p95_abs_dh_du_mm_per_px": item["p95_abs"],
                    "max_abs_dh_du_mm_per_px": item["max_abs"],
                    "mean_abs_dh_du_mm_per_px": item["mean_abs"],
                    "min_dh_du_mm_per_px": item["min_signed"],
                    "max_dh_du_mm_per_px": item["max_signed"],
                    "ground_valid_fraction": (
                        float(np.mean([all(bool(row[name]) for name in (
                            "base_ground_reference_valid",
                            "u_plus_0p05_ground_reference_valid",
                            "u_minus_0p05_ground_reference_valid",
                            "u_plus_0p10_ground_reference_valid",
                            "u_minus_0p10_ground_reference_valid",
                        )) for row in selected]))
                        if selected
                        else None
                    ),
                    "c1_clamped_fraction": (
                        float(np.mean([any(bool(row[name]) for name in (
                            "base_c1_clamped",
                            "u_plus_0p05_c1_clamped",
                            "u_minus_0p05_c1_clamped",
                            "u_plus_0p10_c1_clamped",
                            "u_minus_0p10_c1_clamped",
                        )) for row in selected]))
                        if selected
                        else None
                    ),
                }
            )
    return output


def summary_lookup(summary: list[dict[str, Any]], group_type: str, group: str, epsilon: float) -> dict[str, Any]:
    for row in summary:
        if (
            row["group_type"] == group_type
            and row["group"] == group
            and float(row["epsilon_px"]) == epsilon
        ):
            return row
    raise KeyError((group_type, group, epsilon))


def ratio(high: Any, low: Any) -> float | None:
    high_value = finite(high)
    low_value = finite(low)
    if high_value is None or low_value is None or low_value <= 0.0:
        return None
    return high_value / low_value


def conclusion(summary: list[dict[str, Any]]) -> dict[str, Any]:
    pooled_ratios = {
        f"epsilon_{epsilon:g}": ratio(
            summary_lookup(summary, "v_band", "v>2600", epsilon)["p95_abs_dh_du_mm_per_px"],
            summary_lookup(summary, "v_band", "v<1800", epsilon)["p95_abs_dh_du_mm_per_px"],
        )
        for epsilon in EPSILONS
    }
    height_ratios: dict[str, dict[str, float | None]] = {}
    consistent_count = 0
    for height in HEIGHT_ORDER:
        height_ratios[height] = {
            f"epsilon_{epsilon:g}": ratio(
                summary_lookup(summary, "height_and_v", f"{height}:v>2600", epsilon)["p95_abs_dh_du_mm_per_px"],
                summary_lookup(summary, "height_and_v", f"{height}:v<1800", epsilon)["p95_abs_dh_du_mm_per_px"],
            )
            for epsilon in EPSILONS
        }
        if all(
            height_ratios[height][f"epsilon_{epsilon:g}"] is not None
            and float(height_ratios[height][f"epsilon_{epsilon:g}"]) >= 1.10
            for epsilon in EPSILONS
        ):
            consistent_count += 1
    pooled_strong = all(
        pooled_ratios[f"epsilon_{epsilon:g}"] is not None
        and float(pooled_ratios[f"epsilon_{epsilon:g}"]) >= 1.25
        for epsilon in EPSILONS
    )
    if pooled_strong and consistent_count == len(HEIGHT_ORDER):
        flag = "YES"
    elif pooled_strong or consistent_count >= 2:
        flag = "PARTIAL"
    else:
        flag = "NO"
    return {
        "LOWER_EDGE_GEOMETRIC_SENSITIVITY": flag,
        "pooled_edge_to_upper_p95_abs_ratio": pooled_ratios,
        "height_edge_to_upper_p95_abs_ratio": height_ratios,
        "height_consistent_count_threshold_1p10": consistent_count,
        "height_consistency_total": len(HEIGHT_ORDER),
        "decision_rule": {
            "strong_pooled_ratio": ">=1.25 at both epsilon values",
            "height_consistent_ratio": ">=1.10 at both epsilon values",
            "yes": "strong pooled ratio and all three heights consistent",
            "partial": "strong pooled ratio or at least two heights consistent",
        },
    }


def fmt(value: Any, digits: int = 6) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def plot_by_v(summary: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path) -> None:
    usable = [row for row in rows if statistically_eligible(row)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(V_BAND_LABELS))
    colors = {0.05: "#1f77b4", 0.10: "#d62728"}
    for epsilon in EPSILONS:
        values = [
            summary_lookup(summary, "v_band", label, epsilon)["median_abs_dh_du_mm_per_px"]
            for label in V_BAND_LABELS
        ]
        p95 = [
            summary_lookup(summary, "v_band", label, epsilon)["p95_abs_dh_du_mm_per_px"]
            for label in V_BAND_LABELS
        ]
        axes[0].plot(x, values, marker="o", color=colors[epsilon], label=f"median |dh/du|, ±{epsilon:g} px")
        axes[0].plot(x, p95, marker="^", linestyle="--", color=colors[epsilon], alpha=0.75, label=f"P95 |dh/du|, ±{epsilon:g} px")
    axes[0].set_xticks(x, V_BAND_LABELS, rotation=25, ha="right")
    axes[0].set_ylabel("|dh/du| (mm/px)")
    axes[0].set_title("Sensitivity summary by image-v band")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    for height, color in zip(HEIGHT_ORDER, ("#2ca02c", "#9467bd", "#ff7f0e"), strict=True):
        height_rows = [row for row in usable if row["height_label"] == height]
        axes[1].scatter(
            [row["v_px"] for row in height_rows],
            [row["abs_dh_du_0p10_mm_per_px"] for row in height_rows],
            s=5,
            alpha=0.20,
            color=color,
            label=height,
        )
        medians = []
        p95s = []
        for label in V_BAND_LABELS:
            group = [row for row in height_rows if row["v_band"] == label]
            values = np.asarray([row["abs_dh_du_0p10_mm_per_px"] for row in group], dtype=np.float64)
            medians.append(float(np.median(values)) if len(values) else np.nan)
            p95s.append(float(np.percentile(values, 95.0)) if len(values) else np.nan)
        centers = np.asarray([900.0, 2000.0, 2300.0, 2500.0, 2800.0])
        axes[1].plot(centers, medians, marker="o", color=color, linewidth=1.8)
        axes[1].plot(centers, p95s, marker="^", linestyle="--", color=color, alpha=0.8)
    axes[1].set_xlabel("full-sensor v (row)")
    axes[1].set_ylabel("|dh/du| at ±0.10 px (mm/px)")
    axes[1].set_title("Pointwise sensitivity versus v")
    axes[1].set_xlim(0, 3000)
    # Keep the broad point cloud readable; extreme tails remain available in
    # the CSV/report max column and are not used for this visual y-limit.
    axes[1].set_ylim(0.0, 0.33)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, title="nominal height")
    fig.suptitle("Task A-1 · Daheng row-scan u-direction sensitivity", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_height_and_v(summary: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row_index, epsilon in enumerate(EPSILONS):
        for column_index, metric in enumerate(("median_abs_dh_du_mm_per_px", "p95_abs_dh_du_mm_per_px")):
            axis = axes[row_index, column_index]
            matrix = np.full((len(HEIGHT_ORDER), len(V_BAND_LABELS)), np.nan, dtype=np.float64)
            counts = np.zeros_like(matrix, dtype=np.int64)
            for height_index, height in enumerate(HEIGHT_ORDER):
                for band_index, band in enumerate(V_BAND_LABELS):
                    item = summary_lookup(summary, "height_and_v", f"{height}:{band}", epsilon)
                    value = finite(item[metric])
                    matrix[height_index, band_index] = np.nan if value is None else value
                    counts[height_index, band_index] = int(item["n"])
            image = axis.imshow(matrix, aspect="auto", cmap="viridis")
            axis.set_xticks(np.arange(len(V_BAND_LABELS)), V_BAND_LABELS, rotation=25, ha="right")
            axis.set_yticks(np.arange(len(HEIGHT_ORDER)), HEIGHT_ORDER)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        axis.text(j, i, f"{matrix[i,j]:.3g}\n(n={counts[i,j]})", ha="center", va="center", color="white", fontsize=8)
            axis.set_title(f"{metric.replace('_', ' ')} · ±{epsilon:g} px")
            fig.colorbar(image, ax=axis, shrink=0.84, label="mm/px")
    fig.suptitle("Task A-1 · height consistency of lower-edge sensitivity", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report_text(
    output_dir: Path,
    provenance: dict[str, Any],
    summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    usable = [row for row in rows if statistically_eligible(row)]
    strict_valid = [row for row in rows if row["sensitivity_status"] == "OK"]
    lines = [
        "# Task A-1｜下边缘局部三角测量敏感度审计",
        "",
        f"生成时间（UTC）：`{provenance['generated_at_utc']}`",
        "",
        "## 明确结论",
        "",
        f"`LOWER_EDGE_GEOMETRIC_SENSITIVITY = {decision['LOWER_EDGE_GEOMETRIC_SENSITIVITY']}`",
        "",
        "这里的判定只回答：正式 C0+C1+Session Ground 逐点几何链是否在下边缘对 `u` 中心误差表现出明显更大的局部高度响应。它不等同于证明实际 Steger 误差幅度已经变大，也不单独证明所有下边缘 truth residual 都由该机制造成。",
        "",
        f"- 原始 measurement-ROI Steger 点：`{len(rows)}`；五个正式链路结果均有限且 reconstruction 有效、可进入敏感度统计的点：`{len(usable)}`；其中五次 Ground reference 均在有效域的点：`{len(strict_valid)}`。",
        f"- 采集覆盖：`{provenance['frames']['frames_total']}` 帧、`{provenance['frames']['conditions']}` 个 condition、`h10/h20/h30 × p01...p10`；高度标签沿用已有 nominal `10/20/30 mm`，没有把 truth 反向用于筛点或优化。",
        f"- 判定依据：`v>2600` 相对 `v<1800` 的 P95 `|dh/du|` 比值，在两个扰动步长均至少 `1.25` 且三个高度均至少 `1.10` 时判 `YES`；否则按报告中规则给出 `PARTIAL/NO`。",
        "",
        "## 计算定义与方向",
        "",
        "- Daheng 当前实际语义是 full-sensor `(u,v)=(column,row)`；本审计扰动的是 `u`，即横向列坐标。原始缓存已经是 full-sensor 坐标，不再重复加 ROI offset。",
        "- 对每个原始 Steger 点，先用正式 `reconstruct_uv_to_ground()` 得到 C0 Quadratic + Frozen C1 的 ground 点，再用当前 Session Ground `Zg=a*S+b` 做 leveling；该逐点 leveled `Zg` 记为 `Base height h`。没有调用 H1/H-B2，也没有调用 truth residual 进行任何选择。",
        "- 对 `epsilon=0.05, 0.10 px`：`dh/du=(h(u+epsilon,v)-h(u-epsilon,v))/(2*epsilon)`；统计主指标为 `|dh/du|` 的 median、P95、max。",
        "- 五组输入使用同一批原始 ROI 点；扰动后不重新选择 ROI、不重新提取 Steger、不重新拟合 Ground/C0/C1/H1/H-B2。五个批次是向量化执行，点与点之间没有拟合或聚合依赖。",
        "",
        "## 按 v 分组",
        "",
        "| v band | n @0.05 | median @0.05 | P95 @0.05 | max @0.05 | n @0.10 | median @0.10 | P95 @0.10 | max @0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for band in V_BAND_LABELS:
        a = summary_lookup(summary, "v_band", band, 0.05)
        b = summary_lookup(summary, "v_band", band, 0.10)
        lines.append(
            f"| {band} | {a['n']} | {fmt(a['median_abs_dh_du_mm_per_px'])} | {fmt(a['p95_abs_dh_du_mm_per_px'])} | {fmt(a['max_abs_dh_du_mm_per_px'])} | {b['n']} | {fmt(b['median_abs_dh_du_mm_per_px'])} | {fmt(b['p95_abs_dh_du_mm_per_px'])} | {fmt(b['max_abs_dh_du_mm_per_px'])} |"
        )
    lines.extend(["", "## 10/20/30 mm 高度一致性", "", "| height | P95 ratio high/low @0.05 | P95 ratio high/low @0.10 |", "|---|---:|---:|"])
    for height in HEIGHT_ORDER:
        values = decision["height_edge_to_upper_p95_abs_ratio"][height]
        lines.append(f"| {height} | {fmt(values['epsilon_0.05'], 3)} | {fmt(values['epsilon_0.1'], 3)} |")
    lines.extend([
        "",
        "## Pooled FOV position",
        "",
        "`p01...p10` 是每个 height 内由 frozen height-ROI v 排序得到的离散 FOV position label；不同 height 的同名 position 只按既有 condition 语义配对，不把它解释成连续 v 坐标。完整两种 epsilon、position 及 height×position 统计见 `pixel_sensitivity_summary.csv`。",
        "",
        "| position | n | P95 @0.05 (mm/px) | P95 @0.10 (mm/px) |",
        "|---|---:|---:|---:|",
    ])
    for position in POSITION_ORDER:
        a = summary_lookup(summary, "fov_position", position, 0.05)
        b = summary_lookup(summary, "fov_position", position, 0.10)
        lines.append(f"| {position} | {a['n']} | {fmt(a['p95_abs_dh_du_mm_per_px'])} | {fmt(b['p95_abs_dh_du_mm_per_px'])} |")
    lines.extend([
        "",
        "## Provenance / reuse audit",
        "",
        "本轮复用：",
        "",
        f"- Frozen Steger centers：`{provenance['inputs']['cache']['path']}`；600 帧、`{provenance['inputs']['cache']['centers_total']}` 个缓存中心，manifest 标记 `one_steger_per_frame=true`。",
        f"- Frozen V2 ROI registry：`{provenance['inputs']['registry']['path']}`；30/30 frozen、geometry-only、manual confirmed。",
        f"- Frozen C0/C1/config：当前 Daheng config、manifest、Quadratic C0、`C1_4k`，C1 runtime enabled=`{provenance['formal_chain']['c1_enabled']}`。",
        f"- Session R/t 与 Ground：`{provenance['inputs']['ground']['path']}`；status=`{provenance['formal_chain']['ground_status']}`、runtime extrinsic source=`{provenance['formal_chain']['ground_extrinsic_source']}`、Ground RMSE=`{fmt(provenance['formal_chain']['ground_rmse_mm'])} mm`。",
        "",
        "本轮新增：",
        "",
        "- 读取每个 frozen measurement-ROI 点并进行 `u±0.05/0.10 px` 的 deterministic forward reconstruction 与有限差分统计；没有重新运行 Steger、没有拟合或修改任何 correction/calibration。",
        "- 生成 `pixel_sensitivity_audit.csv`（逐点）、`pixel_sensitivity_summary.csv`（分组）、两张 PNG 和本报告。",
        "",
        "## 边界与解释",
        "",
        "- `h10/h20/h30` 是已有 Session01 的 nominal condition labels，不是本轮重新认证的 truth height。",
        "- `v` 分组是原始点的 full-sensor row；扰动只改 `u`，因此结果不回答 `dv` 敏感度。",
        "- Session Ground 有效域外的点没有被静默外推：按正式 `apply_to_points()` 语义保留 raw Zg 并标记 `GROUND_REFERENCE_PARTIAL`，同时进入敏感度统计；Ground-valid fraction 在 summary 中单独报告。C1 clamp 状态也保留在逐点 CSV 和 summary 中。",
        "- 结果是 local geometric sensitivity，不是实际高度误差预测。要把它转换成预期误差，还需要独立的 Steger `u` localization error envelope；本审计不从 truth residual 反推该 envelope。",
        "",
        "## 输出",
        "",
        f"输出目录：`{output_dir}`",
        "",
        "- `pixel_sensitivity_audit.csv`",
        "- `pixel_sensitivity_by_v.png`",
        "- `pixel_sensitivity_by_height_and_v.png`",
        "- `report.md`",
    ])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [args.config, args.cache, args.cache_manifest, args.registry, args.ground]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"A-1 required artifact missing: {missing}")

    app = load_app_config(args.config)
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_model,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        app.calibration.laser_ray_correction,
        ground_u_optional=True,
    )
    ground_payload, ground_reference = load_session_ground(args.ground)
    calibration["R"] = np.asarray(
        ground_payload["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64
    )
    calibration["t"] = np.asarray(
        ground_payload["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64
    )
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    frames, centers_by_key, cache_info = load_frozen_cache(args.cache, args.cache_manifest)
    points, metadata = build_point_metadata(frames, centers_by_key, registry)

    evaluations: dict[str, dict[str, Any]] = {}
    evaluations["base"] = evaluate_base_height(
        points, calibration, app.reconstruction, ground_reference
    )
    for label, delta in (
        ("u_plus_0p05", 0.05),
        ("u_minus_0p05", -0.05),
        ("u_plus_0p10", 0.10),
        ("u_minus_0p10", -0.10),
    ):
        perturbed = points.copy()
        perturbed[:, 0] += delta
        evaluations[label] = evaluate_base_height(
            perturbed, calibration, app.reconstruction, ground_reference
        )
    rows = build_point_rows(points, metadata, evaluations)
    summary = summary_rows(rows)
    decision = conclusion(summary)

    relative_inputs = {
        "config": args.config,
        "manifest": TOOL_ROOT / "configs" / "calibration_daheng_0811" / "manifest.yaml",
        "c0_quadratic": app.calibration.laser_model,
        "c1_4k": app.calibration.laser_ray_correction,
        "extrinsics_config": app.calibration.extrinsics,
        "cache": args.cache,
        "cache_manifest": args.cache_manifest,
        "registry": args.registry,
        "ground": args.ground,
    }
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "A-1 pixel sensitivity audit",
        "formal_chain": {
            "reconstruction_function": "laser_measurement_tool.reconstruction.reconstructor.reconstruct_uv_to_ground",
            "pixel_semantics": "u=full-sensor column, v=full-sensor row",
            "c0_model_type": calibration.get("laser_model", {}).get("model_type"),
            "c1_enabled": bool(app.reconstruction.enable_laser_ray_correction),
            "c1_model": getattr(calibration.get("laser_ray_correction"), "model_id", "C1_4k"),
            "ground_status": ground_payload.get("session_ground_reference", {}).get("status"),
            "ground_extrinsic_source": ground_payload.get("runtime", {}).get("ground_extrinsic_source"),
            "ground_rmse_mm": ground_payload.get("session_ground_reference", {}).get("rmse_mm"),
            "ground_valid_s_range_mm": ground_payload.get("session_ground_reference", {}).get("valid_s_range_mm"),
            "base_height_definition": "Session-Ground-levelled final pointwise Zg; no H1/H-B2",
            "roi_selection": "original frozen height_v_range, inclusive A-13B (+0.5 pixel) rule",
            "finite_difference": "central difference at u +/- 0.05 and u +/- 0.10 px",
        },
        "frames": {
            "frames_total": cache_info["frames_total"],
            "conditions": len({row["condition_id"] for row in metadata}),
            "points_in_measurement_roi": len(metadata),
            "points_statistically_eligible": sum(statistically_eligible(row) for row in rows),
            "points_ground_reference_strictly_valid": sum(row["sensitivity_status"] == "OK" for row in rows),
            "status_counts": {
                status: sum(row["sensitivity_status"] == status for row in rows)
                for status in sorted({row["sensitivity_status"] for row in rows})
            },
            "cache_info": cache_info,
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                **({"centers_total": cache_info["centers_total"]} if name == "cache" else {}),
            }
            for name, path in relative_inputs.items()
            if path.is_file()
        },
        "outputs": {
            "pointwise_csv": str((output_dir / "pixel_sensitivity_audit.csv").resolve()),
            "summary_csv": str((output_dir / "pixel_sensitivity_summary.csv").resolve()),
            "by_v_png": str((output_dir / "pixel_sensitivity_by_v.png").resolve()),
            "height_and_v_png": str((output_dir / "pixel_sensitivity_by_height_and_v.png").resolve()),
            "report": str((output_dir / "report.md").resolve()),
        },
        "decision": decision,
    }

    point_fields = [
        "dataset", "height_label", "true_height_mm", "position_id", "v_order_rank", "condition_id",
        "repeat_index", "filename", "cache_key", "camera_frame_number", "point_index",
        "u_px", "v_px", "v_band", "height_v_range", "base_height_mm", "base_raw_zg_mm",
        "u_plus_0p05_height_mm", "u_minus_0p05_height_mm", "u_plus_0p10_height_mm", "u_minus_0p10_height_mm",
        "dh_du_0p05_mm_per_px", "abs_dh_du_0p05_mm_per_px", "dh_du_0p10_mm_per_px", "abs_dh_du_0p10_mm_per_px",
        "finite_difference_delta_abs_mm_per_px", "base_reconstruction_valid", "base_ground_reference_valid",
        "base_c1_clamped", "u_plus_0p05_ground_reference_valid", "u_minus_0p05_ground_reference_valid",
        "u_plus_0p10_ground_reference_valid", "u_minus_0p10_ground_reference_valid", "u_plus_0p05_c1_clamped",
        "u_minus_0p05_c1_clamped", "u_plus_0p10_c1_clamped", "u_minus_0p10_c1_clamped", "sensitivity_status",
    ]
    summary_fields = [
        "group_type", "group", "epsilon_px", "n", "median_dh_du_mm_per_px",
        "median_abs_dh_du_mm_per_px", "p95_abs_dh_du_mm_per_px", "max_abs_dh_du_mm_per_px",
        "mean_abs_dh_du_mm_per_px", "min_dh_du_mm_per_px", "max_dh_du_mm_per_px",
        "ground_valid_fraction", "c1_clamped_fraction",
    ]
    write_csv(output_dir / "pixel_sensitivity_audit.csv", rows, point_fields)
    write_csv(output_dir / "pixel_sensitivity_summary.csv", summary, summary_fields)
    plot_by_v(summary, rows, output_dir / "pixel_sensitivity_by_v.png")
    plot_height_and_v(summary, output_dir / "pixel_sensitivity_by_height_and_v.png")
    write_json(output_dir / "pixel_sensitivity_provenance.json", provenance)
    (output_dir / "report.md").write_text(
        report_text(output_dir, provenance, summary, rows, decision), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "decision": decision}, ensure_ascii=False, indent=2))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--ground", type=Path, default=DEFAULT_GROUND)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
