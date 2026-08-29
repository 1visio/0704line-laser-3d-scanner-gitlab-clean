#!/usr/bin/env python3
"""Thermal-A3-2 paired cold-vs-hot Session configuration replay.

Only the saved Session configuration is changed between conditions.  The PNGs,
Frozen ROI registry, Steger/C0/C1, and H1/H-B2 configuration remain identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import thermal_a2_analysis as a2


ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = ROOT / "laser_measurement_tool"
MORNING_DIR = TOOL_ROOT / "output_daheng_0811" / "online_recordings" / "0827上午热漂_2000"
AFTERNOON_DIR = TOOL_ROOT / "output_daheng_0811" / "online_recordings" / "0827下午热漂_2000"
HOT_CONFIG_DIR = TOOL_ROOT / "output_daheng_0811" / "online_recordings" / "0827下午热漂结束时配置"
A3_DIR = ROOT / "projects" / "daheng" / "analysis" / "thermal_a3_full_day_0827"
A2_DIR = ROOT / "projects" / "daheng" / "analysis" / "thermal_a2_0827"
OUTPUT_DIR = ROOT / "projects" / "daheng" / "analysis" / "thermal_a3_cold_hot_0827"
TERMINAL_K = 5
FRACTION_DENOMINATOR_MIN_MM = 0.001
MATERIAL_DRIFT_MM = 0.03


class ThermalA32Error(RuntimeError):
    """Raised when a paired replay invariant is not satisfied."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morning-dir", type=Path, default=MORNING_DIR)
    parser.add_argument("--afternoon-dir", type=Path, default=AFTERNOON_DIR)
    parser.add_argument("--cold-session", type=Path, default=MORNING_DIR / "session_ground_calibration.json")
    parser.add_argument("--hot-session", type=Path, default=HOT_CONFIG_DIR / "session_ground_calibration.json")
    parser.add_argument("--a3-dir", type=Path, default=A3_DIR)
    parser.add_argument("--a2-dir", type=Path, default=A2_DIR)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "projects" / "daheng" / "analysis" / "thermal_a2a_roi_v2_0827" / "thermal_roi_registry_v2_frozen.json",
    )
    parser.add_argument("--measure-config", type=Path, default=TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--terminal-k", type=int, default=TERMINAL_K)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ThermalA32Error(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: a2.json_safe(row.get(key)) for key in fields} for row in rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(a2.json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite(value: Any) -> float | None:
    return a2.finite(value)


def values(values_in: Iterable[Any]) -> np.ndarray:
    return np.asarray([float(item) for value in values_in if (item := finite(value)) is not None], dtype=np.float64)


def stats(values_in: Iterable[Any]) -> dict[str, Any]:
    array = values(values_in)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "std": None, "range": None, "p95_abs": None, "max_abs": None}
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "range": float(np.ptp(array)),
        "p95_abs": float(np.percentile(np.abs(array), 95)),
        "max_abs": float(np.max(np.abs(array))),
    }


def schema_signature(value: Any, path: str = "$") -> set[tuple[str, str]]:
    output = {(path, type(value).__name__)}
    if isinstance(value, dict):
        for key, child in value.items():
            output |= schema_signature(child, f"{path}.{key}")
    elif isinstance(value, list) and value:
        output |= schema_signature(value[0], f"{path}[]")
    return output


def verify_a3(a3_dir: Path) -> dict[str, Any]:
    manifest_path = a3_dir / "thermal_a3_run_manifest.json"
    manifest = a2.load_json(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise ThermalA32Error("Thermal-A3-1 manifest is not COMPLETE")
    for name, expected in manifest.get("output_sha256", {}).items():
        actual = a2.sha256_file(a3_dir / name)
        if actual.lower() != str(expected).lower():
            raise ThermalA32Error(f"A3-1 artifact changed after completion: {name}")
    if manifest.get("cardinality", {}).get("formal_frames") != 760:
        raise ThermalA32Error("A3-1 formal cardinality is not 760 frames")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": a2.sha256_file(manifest_path),
        "verified_output_count": len(manifest.get("output_sha256", {})),
        "status": manifest["status"],
    }


def validate_session_pair(cold_path: Path, hot_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cold = a2.load_json(cold_path)
    hot = a2.load_json(hot_path)
    for label, payload in (("cold", cold), ("hot", hot)):
        if payload.get("status") != "VALID" or payload.get("valid") is not True:
            raise ThermalA32Error(f"{label} Session configuration is not VALID")
        if payload.get("runtime", {}).get("ground_extrinsic_source") != "session":
            raise ThermalA32Error(f"{label} Session does not use session extrinsic")
        if payload.get("session_ground_reference", {}).get("status") != "VALID":
            raise ThermalA32Error(f"{label} Session Ground reference is not VALID")
    if schema_signature(cold) != schema_signature(hot):
        raise ThermalA32Error("Cold/hot Session JSON schemas differ")
    if cold.get("reference_extrinsic") != hot.get("reference_extrinsic"):
        raise ThermalA32Error("Reference extrinsic unexpectedly changed between Session files")
    if cold.get("board") != hot.get("board"):
        raise ThermalA32Error("Session board definition unexpectedly changed")

    rc = np.asarray(cold["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64)
    rh = np.asarray(hot["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64)
    tc = np.asarray(cold["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64)
    th = np.asarray(hot["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64)
    relative = rh @ rc.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = math.degrees(math.acos(cosine))
    translation_delta = th - tc
    translation_norm = float(np.linalg.norm(translation_delta))
    gc = cold["session_ground_reference"]
    gh = hot["session_ground_reference"]
    common_lo = max(float(gc["valid_s_range_mm"][0]), float(gh["valid_s_range_mm"][0]))
    common_hi = min(float(gc["valid_s_range_mm"][1]), float(gh["valid_s_range_mm"][1]))
    if common_hi <= common_lo:
        raise ThermalA32Error("Cold/hot Ground references have no common S domain")
    delta_slope = float(gh["slope_z_per_mm"]) - float(gc["slope_z_per_mm"])
    delta_intercept = float(gh["intercept_z_mm"]) - float(gc["intercept_z_mm"])
    return cold, hot, {
        "cold_sha256": a2.sha256_file(cold_path),
        "hot_sha256": a2.sha256_file(hot_path),
        "relative_rotation_deg": rotation_deg,
        "relative_rotation_matrix": relative,
        "translation_delta_mm": translation_delta,
        "translation_delta_norm_mm": translation_norm,
        "ground_delta_slope_mm_per_mm": delta_slope,
        "ground_delta_intercept_mm": delta_intercept,
        "ground_common_s_min_mm": common_lo,
        "ground_common_s_max_mm": common_hi,
        "ground_delta_tilt_across_common_span_mm": delta_slope * (common_hi - common_lo),
        "schema_equal": True,
        "reference_extrinsic_equal": True,
        "board_equal": True,
    }


def select_terminal_rows(a3_rows: list[dict[str, str]], afternoon_dir: Path, terminal_k: int) -> tuple[list[dict[str, str]], list[str]]:
    by_recording: dict[str, dict[str, str]] = {}
    for row in a3_rows:
        if row.get("source_period") == "afternoon":
            by_recording.setdefault(row["recording_id"], row)
    ordered = sorted(by_recording.values(), key=lambda row: int(row["recording_order"]))
    if len(ordered) < terminal_k:
        raise ThermalA32Error(f"Need {terminal_k} formal afternoon recordings, found {len(ordered)}")
    selected = ordered[-terminal_k:]
    ids = [row["recording_id"] for row in selected]
    output = []
    for row in selected:
        path = (afternoon_dir / row["recording_id"]).resolve()
        if not path.is_dir() or row["recording_id"].startswith("."):
            raise ThermalA32Error(f"Invalid formal terminal recording: {path}")
        frame_rows = read_csv(path / "frames.csv")
        if len(frame_rows) != 20 or len(list(path.glob("*.png"))) != 20:
            raise ThermalA32Error(f"Terminal recording is not 20 frames: {path}")
        output.append(
            {
                "recording_id": row["recording_id"],
                "relative_path": str(path),
                "segment": row["segment"],
                "frame_count": "20",
                "first_frame_time_local": row["recording_time_local"],
                "elapsed_from_power_start_s": str(float(row["elapsed_from_power_min"]) * 60.0),
                "elapsed_from_reference_start_s": str(float(row["elapsed_from_reference_min"]) * 60.0),
            }
        )
    return output, ids


def source_set_sha256(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        path = Path(row["relative_path"])
        digest.update(row["recording_id"].encode("utf-8"))
        for file_path in [path / "frames.csv", *sorted(path.glob("*.png"))]:
            digest.update(str(file_path.name).encode("utf-8"))
            digest.update(bytes.fromhex(a2.sha256_file(file_path)))
    return digest.hexdigest()


def ground_session_replay(
    inventory: list[dict[str, str]],
    rois: dict[str, a2.ObjectRoi],
    app: Any,
    calibration: dict[str, Any],
    session_reference: Any,
) -> list[dict[str, Any]]:
    output = []
    for order, item in enumerate(inventory, start=1):
        raw_s: list[np.ndarray] = []
        raw_z: list[np.ndarray] = []
        leveled_s: list[np.ndarray] = []
        leveled_z: list[np.ndarray] = []
        frames = a2.roi_gui.load_source_frames(Path(item["relative_path"]), app)
        for frame in frames:
            reconstruction = a2.reconstruct_uv_to_ground(frame.centers_uv_full, calibration, app.reconstruction)
            pixels = np.asarray(reconstruction.pixels_uv, dtype=np.float64)
            points_raw = np.asarray(reconstruction.points_ground, dtype=np.float64)
            points_leveled, session_valid = session_reference.apply_to_points(points_raw)
            session_valid = np.asarray(session_valid, dtype=bool)
            ground_mask = np.zeros(len(pixels), dtype=bool)
            for roi in rois.values():
                ground_mask |= a2.v_mask(pixels, roi.baseline_before)
                ground_mask |= a2.v_mask(pixels, roi.baseline_after)
            selected_raw = points_raw[ground_mask]
            selected_s = session_reference.project_s(selected_raw[:, :2])
            raw_s.append(selected_s)
            raw_z.append(selected_raw[:, 2])
            valid_mask = ground_mask & session_valid
            selected_leveled = np.asarray(points_leveled, dtype=np.float64)[valid_mask]
            leveled_s.append(session_reference.project_s(points_raw[valid_mask, :2]))
            leveled_z.append(selected_leveled[:, 2])
        raw_fit = a2.fit_ground(np.concatenate(raw_s), np.concatenate(raw_z))
        leveled_fit = a2.fit_ground(np.concatenate(leveled_s), np.concatenate(leveled_z))
        output.append(
            {
                "recording_order": order,
                "recording_id": item["recording_id"],
                "segment": item["segment"],
                "recording_time_local": item["first_frame_time_local"],
                "elapsed_from_power_min": float(item["elapsed_from_power_start_s"]) / 60.0,
                "raw_ground_offset_b_mm": raw_fit.intercept,
                "raw_ground_slope_a_mm_per_mm": raw_fit.slope,
                "raw_ground_detrended_rmse_mm": raw_fit.rmse,
                "raw_ground_detrended_p95_mm": raw_fit.p95,
                "raw_ground_point_count": raw_fit.point_count,
                "session_leveled_ground_offset_b_mm": leveled_fit.intercept,
                "session_leveled_ground_slope_a_mm_per_mm": leveled_fit.slope,
                "session_leveled_ground_detrended_rmse_mm": leveled_fit.rmse,
                "session_leveled_ground_detrended_p95_mm": leveled_fit.p95,
                "session_leveled_ground_point_count": leveled_fit.point_count,
            }
        )
    return output


def replay_condition(
    label: str,
    session_path: Path,
    measure_config: Path,
    inventory: list[dict[str, str]],
    rois: dict[str, a2.ObjectRoi],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    app, calibration, _, session_reference = a2.load_chain(
        measure_config,
        session_path,
    )
    frame_rows, recording_rows, ground_rows, _ = a2.process_all(
        inventory, rois, app, calibration, session_reference, 0
    )
    height_rows = a2.build_height_summary(frame_rows, recording_rows)
    leveled_ground = ground_session_replay(inventory, rois, app, calibration, session_reference)
    session_sha = a2.sha256_file(session_path)
    for rows in (frame_rows, recording_rows, ground_rows, height_rows, leveled_ground):
        for row in rows:
            row["condition"] = label
            row["session_configuration_sha256"] = session_sha
    return frame_rows, height_rows, ground_rows, leveled_ground


def assert_paired_invariants(
    cold_frames: list[dict[str, Any]], hot_frames: list[dict[str, Any]], cold_heights: list[dict[str, Any]], a3_rows: list[dict[str, str]]
) -> None:
    key = lambda row: (row["recording_id"], int(row["frame_index"]), row["object_id"])
    cold_map = {key(row): row for row in cold_frames}
    hot_map = {key(row): row for row in hot_frames}
    if cold_map.keys() != hot_map.keys():
        raise ThermalA32Error("Cold/hot frame-object keys differ")
    invariant_fields = (
        "filename", "camera_frame_number", "host_timestamp_ns", "exposure_us", "gain_db",
        "pixel_format", "steger_point_count", "object_id", "roi_baseline_before", "roi_height",
        "roi_baseline_after", "reconstructed_point_count", "reconstruction_valid_ratio",
        "session_q1", "session_q2", "local_q1", "local_q2",
    )
    for item_key in cold_map:
        cold, hot = cold_map[item_key], hot_map[item_key]
        for field in invariant_fields:
            left, right = cold.get(field), hot.get(field)
            lf, rf = finite(left), finite(right)
            equal = abs(lf - rf) <= 1e-12 if lf is not None and rf is not None else left == right
            if not equal:
                raise ThermalA32Error(f"Paired invariant changed: {item_key} {field}: {left} != {right}")
    if any(row["frame_status"] != "VALID" for row in cold_frames + hot_frames):
        raise ThermalA32Error("Invalid reconstruction in paired replay")
    if any(row["session_status"] != "VALID" or row["local_status"] != "VALID" for row in cold_frames + hot_frames):
        raise ThermalA32Error("Frozen ROI support failure in paired replay")

    a3_lookup = {
        (row["recording_id"], row["object_id"], row["reference"], row["algorithm"]): row
        for row in a3_rows
    }
    for row in cold_heights:
        original = a3_lookup[(row["recording_id"], row["object_id"], row["reference"], row["algorithm"])]
        for field in ("mean_mm", "median_mm", "repeatability_std_mm", "bias_mm", "rmse_mm", "p95_abs_error_mm"):
            if abs(float(row[field]) - float(original[field])) > 1e-10:
                raise ThermalA32Error(f"Cold replay does not reproduce A3-1: {row['recording_id']} {field}")


def first_height_lookup(a3_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    first_order = min(int(row["recording_order"]) for row in a3_rows)
    return {
        (row["object_id"], row["reference"], row["algorithm"]): float(row["mean_mm"])
        for row in a3_rows if int(row["recording_order"]) == first_order
    }


def build_height_delta_rows(
    cold: list[dict[str, Any]], hot: list[dict[str, Any]],
    cold_frames: list[dict[str, Any]], hot_frames: list[dict[str, Any]],
    initial: dict[tuple[str, str, str], float]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in cold + hot:
        grouped[(row["object_id"], row["reference"], row["algorithm"])][row["condition"]].append(row)
    output = []
    for object_id in a2.OBJECT_IDS:
        for reference in a2.REFERENCES:
            for algorithm in a2.ALGORITHMS:
                key = (object_id, reference, algorithm)
                cold_rows = grouped[key]["cold_session"]
                hot_rows = grouped[key]["hot_session"]
                cold_stats = stats(row["mean_mm"] for row in cold_rows)
                hot_stats = stats(row["mean_mm"] for row in hot_rows)
                initial_mean = initial[key]
                cold_accum = float(cold_stats["mean"]) - initial_mean
                hot_residual = float(hot_stats["mean"]) - initial_mean
                denominator = abs(cold_accum)
                fraction = 1.0 - abs(hot_residual) / denominator if denominator >= FRACTION_DENOMINATOR_MIN_MM else None
                value_field = (
                    f"{reference}_base_mean_mm"
                    if algorithm == "base"
                    else f"{reference}_{algorithm}_mm"
                )
                nominal = float(a2.OBJECT_META[object_id]["height_mm"])
                cold_frame_values = values(
                    row.get(value_field) for row in cold_frames if row["object_id"] == object_id
                )
                hot_frame_values = values(
                    row.get(value_field) for row in hot_frames if row["object_id"] == object_id
                )
                cold_errors = cold_frame_values - nominal
                hot_errors = hot_frame_values - nominal
                output.append(
                    {
                        "row_type": "height_replay_summary",
                        "metric": "terminal_window_mean_height",
                        "object_id": object_id,
                        "position": a2.OBJECT_META[object_id]["position"],
                        "nominal_height_mm": a2.OBJECT_META[object_id]["height_mm"],
                        "reference": reference,
                        "algorithm": algorithm,
                        "terminal_recording_count": len(cold_rows),
                        "initial_cold_mean_mm": initial_mean,
                        "cold_terminal_mean_mm": cold_stats["mean"],
                        "hot_terminal_mean_mm": hot_stats["mean"],
                        "hot_minus_cold_same_png_mm": float(hot_stats["mean"]) - float(cold_stats["mean"]),
                        "cold_accumulated_drift_mm": cold_accum,
                        "hot_residual_drift_mm": hot_residual,
                        "cold_terminal_range_mm": cold_stats["range"],
                        "hot_terminal_range_mm": hot_stats["range"],
                        "cold_terminal_frame_rmse_mm": float(np.sqrt(np.mean(cold_errors**2))),
                        "hot_terminal_frame_rmse_mm": float(np.sqrt(np.mean(hot_errors**2))),
                        "cold_terminal_frame_p95_abs_error_mm": float(np.percentile(np.abs(cold_errors), 95)),
                        "hot_terminal_frame_p95_abs_error_mm": float(np.percentile(np.abs(hot_errors), 95)),
                        "cold_terminal_frame_observed_max_abs_error_mm": float(np.max(np.abs(cold_errors))),
                        "hot_terminal_frame_observed_max_abs_error_mm": float(np.max(np.abs(hot_errors))),
                        "cold_repeatability_std_median_mm": float(np.median([float(row["repeatability_std_mm"]) for row in cold_rows])),
                        "hot_repeatability_std_median_mm": float(np.median([float(row["repeatability_std_mm"]) for row in hot_rows])),
                        "reference_geometry_drift_fraction": fraction,
                        "reference_geometry_drift_fraction_status": "VALID" if fraction is not None else "DENOMINATOR_TOO_SMALL",
                        "fraction_denominator_abs_mm": denominator,
                        "fraction_definition": "1 - abs(hot_terminal-initial_cold)/abs(cold_terminal-initial_cold)",
                    }
                )
    return output


def build_ground_delta_row(
    cold_ground: list[dict[str, Any]], hot_ground: list[dict[str, Any]], initial_raw_b: float,
    initial_leveled_b: float,
) -> dict[str, Any]:
    cold_raw = stats(row["raw_ground_offset_b_mm"] for row in cold_ground)
    hot_raw = stats(row["raw_ground_offset_b_mm"] for row in hot_ground)
    cold_leveled = stats(row["session_leveled_ground_offset_b_mm"] for row in cold_ground)
    hot_leveled = stats(row["session_leveled_ground_offset_b_mm"] for row in hot_ground)
    cold_accum = float(cold_leveled["mean"]) - initial_leveled_b
    hot_residual = float(hot_leveled["mean"]) - initial_leveled_b
    denominator = abs(cold_accum)
    fraction = 1.0 - abs(hot_residual) / denominator if denominator >= FRACTION_DENOMINATOR_MIN_MM else None
    return {
        "row_type": "ground_replay_summary",
        "metric": "ground_offset_b",
        "object_id": "ground",
        "reference": "session_ground",
        "algorithm": "base",
        "terminal_recording_count": len(cold_ground),
        "initial_cold_raw_ground_offset_b_mm": initial_raw_b,
        "cold_terminal_raw_ground_offset_b_mm": cold_raw["mean"],
        "hot_terminal_raw_ground_offset_b_mm": hot_raw["mean"],
        "hot_minus_cold_raw_ground_offset_b_mm": float(hot_raw["mean"]) - float(cold_raw["mean"]),
        "initial_cold_session_leveled_ground_offset_b_mm": initial_leveled_b,
        "cold_terminal_session_leveled_ground_offset_b_mm": cold_leveled["mean"],
        "hot_terminal_session_leveled_ground_offset_b_mm": hot_leveled["mean"],
        "hot_minus_cold_session_leveled_ground_offset_b_mm": float(hot_leveled["mean"]) - float(cold_leveled["mean"]),
        "cold_accumulated_drift_mm": cold_accum,
        "hot_residual_drift_mm": hot_residual,
        "reference_geometry_drift_fraction": fraction,
        "reference_geometry_drift_fraction_status": "VALID" if fraction is not None else "DENOMINATOR_TOO_SMALL",
        "fraction_denominator_abs_mm": denominator,
        "fraction_definition": "Session-leveled Ground: 1 - abs(hot_terminal-initial_cold)/abs(cold_terminal-initial_cold)",
    }


def configuration_delta_rows(cold: dict[str, Any], hot: dict[str, Any], delta: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "row_type": "configuration_delta",
            "metric": "session_saved_at_utc",
            "cold_value": cold.get("saved_at_utc"),
            "hot_value": hot.get("saved_at_utc"),
            "unit": "ISO8601",
        }
    ]
    rc = np.asarray(cold["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64)
    rh = np.asarray(hot["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64)
    tc = np.asarray(cold["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64)
    th = np.asarray(hot["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rows.append({"row_type": "configuration_delta", "metric": f"R_camera_to_ground[{i},{j}]", "cold_value": rc[i, j], "hot_value": rh[i, j], "hot_minus_cold": rh[i, j] - rc[i, j], "unit": "dimensionless"})
    for i, axis in enumerate("xyz"):
        rows.append({"row_type": "configuration_delta", "metric": f"t_camera_to_ground_{axis}", "cold_value": tc[i], "hot_value": th[i], "hot_minus_cold": th[i] - tc[i], "unit": "mm"})
    rows.extend(
        [
            {"row_type": "configuration_delta", "metric": "relative_rotation_angle", "cold_value": 0.0, "hot_value": delta["relative_rotation_deg"], "hot_minus_cold": delta["relative_rotation_deg"], "unit": "deg"},
            {"row_type": "configuration_delta", "metric": "translation_delta_norm", "cold_value": 0.0, "hot_value": delta["translation_delta_norm_mm"], "hot_minus_cold": delta["translation_delta_norm_mm"], "unit": "mm"},
        ]
    )
    gc, gh = cold["session_ground_reference"], hot["session_ground_reference"]
    scalar_fields = [
        ("ground_slope", "slope_z_per_mm", "mm/mm"),
        ("ground_intercept", "intercept_z_mm", "mm"),
        ("ground_rmse", "rmse_mm", "mm"),
        ("ground_point_count", "point_count", "points"),
        ("ground_inlier_count", "inlier_count", "points"),
    ]
    for metric, field, unit in scalar_fields:
        rows.append({"row_type": "configuration_delta", "metric": metric, "cold_value": gc[field], "hot_value": gh[field], "hot_minus_cold": float(gh[field]) - float(gc[field]), "unit": unit})
    for index, name in enumerate(("valid_s_min", "valid_s_max")):
        rows.append({"row_type": "configuration_delta", "metric": name, "cold_value": gc["valid_s_range_mm"][index], "hot_value": gh["valid_s_range_mm"][index], "hot_minus_cold": float(gh["valid_s_range_mm"][index]) - float(gc["valid_s_range_mm"][index]), "unit": "mm"})
    rows.append({"row_type": "configuration_delta", "metric": "ground_tilt_delta_across_common_span", "cold_value": 0.0, "hot_value": delta["ground_delta_tilt_across_common_span_mm"], "hot_minus_cold": delta["ground_delta_tilt_across_common_span_mm"], "unit": "mm"})
    return rows


def decorate_replay_rows(
    heights: list[dict[str, Any]], ground: list[dict[str, Any]], initial: dict[tuple[str, str, str], float]
) -> list[dict[str, Any]]:
    output = []
    for row in heights:
        key = (row["object_id"], row["reference"], row["algorithm"])
        item = dict(row)
        item["result_type"] = "height"
        item["initial_cold_mean_mm"] = initial[key]
        item["accumulated_drift_vs_initial_cold_mm"] = float(row["mean_mm"]) - initial[key]
        output.append(item)
    for row in ground:
        item = dict(row)
        item["result_type"] = "ground"
        item["object_id"] = "ground"
        item["reference"] = "session_ground"
        item["algorithm"] = "base"
        output.append(item)
    return output


def derive_conclusions(height_delta: list[dict[str, Any]], ground_delta: dict[str, Any]) -> dict[str, Any]:
    session_base = [row for row in height_delta if row["reference"] == "session" and row["algorithm"] == "base"]
    denominator = sum(abs(float(row["cold_accumulated_drift_mm"])) for row in session_base)
    numerator = sum(abs(float(row["hot_residual_drift_mm"])) for row in session_base)
    fraction = 1.0 - numerator / denominator if denominator >= FRACTION_DENOMINATOR_MIN_MM else None
    if fraction is None:
        removal = "NO"
    elif fraction >= 0.67 and float(ground_delta["reference_geometry_drift_fraction"] or -math.inf) >= 0.67:
        removal = "YES"
    elif fraction >= 0.20 or float(ground_delta["reference_geometry_drift_fraction"] or -math.inf) >= 0.20:
        removal = "PARTIAL"
    else:
        removal = "NO"
    local_base = [row for row in height_delta if row["reference"] == "local" and row["algorithm"] == "base"]
    max_local = max(abs(float(row["hot_residual_drift_mm"])) for row in local_base)
    residual = "YES" if max_local >= MATERIAL_DRIFT_MM else "PARTIAL" if max_local >= 0.5 * MATERIAL_DRIFT_MM else "NO"
    session_cold_rmse = sum(float(row["cold_terminal_frame_rmse_mm"]) for row in session_base)
    session_hot_rmse = sum(float(row["hot_terminal_frame_rmse_mm"]) for row in session_base)
    local_cold_rmse = sum(float(row["cold_terminal_frame_rmse_mm"]) for row in local_base)
    local_hot_rmse = sum(float(row["hot_terminal_frame_rmse_mm"]) for row in local_base)
    return {
        "HOT_SESSION_REMOVES_GLOBAL_DRIFT": removal,
        "RESIDUAL_LOCAL_THERMAL_DRIFT_REMAINS": residual,
        "REFERENCE_GEOMETRY_DRIFT_FRACTION": fraction,
        "REFERENCE_GEOMETRY_DRIFT_FRACTION_DEFINITION": "object-equal L1 over 20/30/10 mm Base Session accumulated drift",
        "SESSION_BASE_COLD_ACCUMULATED_L1_MM": denominator,
        "SESSION_BASE_HOT_RESIDUAL_L1_MM": numerator,
        "GROUND_REFERENCE_GEOMETRY_DRIFT_FRACTION": ground_delta["reference_geometry_drift_fraction"],
        "MAX_HOT_LOCAL_BASE_RESIDUAL_MM": max_local,
        "MATERIAL_LOCAL_DRIFT_THRESHOLD_MM": MATERIAL_DRIFT_MM,
        "SESSION_BASE_TERMINAL_RMSE_REDUCTION_FRACTION": 1.0 - session_hot_rmse / session_cold_rmse,
        "LOCAL_BASE_TERMINAL_RMSE_REDUCTION_FRACTION": 1.0 - local_hot_rmse / local_cold_rmse,
    }


def save_plot(output_dir: Path, delta: dict[str, Any], height_delta: list[dict[str, Any]], ground_delta: dict[str, Any], conclusions: dict[str, Any]) -> None:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].axis("off")
    text = (
        "Cold → Hot Session geometry\n\n"
        f"Relative rotation: {delta['relative_rotation_deg']:.6f}°\n"
        f"Translation norm: {delta['translation_delta_norm_mm']:.6f} mm\n"
        f"Δt xyz: {', '.join(f'{value:+.6f}' for value in delta['translation_delta_mm'])} mm\n"
        f"Ground Δintercept: {delta['ground_delta_intercept_mm']:+.6f} mm\n"
        f"Ground Δslope: {delta['ground_delta_slope_mm_per_mm']:+.9f} mm/mm\n"
        f"Equivalent Δtilt over common span: {delta['ground_delta_tilt_across_common_span_mm']:+.6f} mm"
    )
    axes[0, 0].text(0.03, 0.95, text, va="top", family="monospace", fontsize=10)

    labels = ["Raw Ground b", "Session-leveled b"]
    cold_values = [
        float(ground_delta["cold_terminal_raw_ground_offset_b_mm"]) - float(ground_delta["initial_cold_raw_ground_offset_b_mm"]),
        ground_delta["cold_accumulated_drift_mm"],
    ]
    hot_values = [
        float(ground_delta["hot_terminal_raw_ground_offset_b_mm"]) - float(ground_delta["initial_cold_raw_ground_offset_b_mm"]),
        ground_delta["hot_residual_drift_mm"],
    ]
    x = np.arange(2)
    axes[0, 1].bar(x - 0.18, cold_values, 0.36, label="Cold Session replay")
    axes[0, 1].bar(x + 0.18, hot_values, 0.36, label="Hot Session replay")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Accumulated offset vs initial cold (mm)")
    axes[0, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.2)

    for ax, reference in zip(axes[1], ("session", "local")):
        rows = [row for row in height_delta if row["reference"] == reference and row["algorithm"] == "base"]
        rows.sort(key=lambda row: a2.OBJECT_IDS.index(row["object_id"]))
        x = np.arange(3)
        ax.bar(x - 0.18, [row["cold_accumulated_drift_mm"] for row in rows], 0.36, label="Cold Session replay")
        ax.bar(x + 0.18, [row["hot_residual_drift_mm"] for row in rows], 0.36, label="Hot Session replay")
        ax.set_xticks(x, ["20-upper", "30-middle", "10-lower"])
        ax.set_ylabel(f"{reference.title()} Base drift vs initial (mm)")
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Cold-vs-Hot Session replay: same terminal PNGs, only Session configuration changed\n"
        f"Reference geometry fraction={conclusions['REFERENCE_GEOMETRY_DRIFT_FRACTION']:.3f}; "
        f"global removal={conclusions['HOT_SESSION_REMOVES_GLOBAL_DRIFT']}; "
        f"local residual={conclusions['RESIDUAL_LOCAL_THERMAL_DRIFT_REMAINS']}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a3_cold_vs_hot.png", bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 6) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def build_report(
    provenance: dict[str, Any], delta: dict[str, Any], terminal_ids: list[str],
    height_delta: list[dict[str, Any]], ground_delta: dict[str, Any], conclusions: dict[str, Any],
) -> str:
    base_lines = [
        "| height | reference | cold accumulated drift | hot residual drift | same-PNG hot−cold | explained fraction |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in height_delta:
        if row["algorithm"] == "base":
            base_lines.append(
                f"| {int(row['nominal_height_mm'])} mm / {row['position']} | {row['reference']} | "
                f"{fmt(row['cold_accumulated_drift_mm'])} | {fmt(row['hot_residual_drift_mm'])} | "
                f"{fmt(row['hot_minus_cold_same_png_mm'])} | {fmt(row['reference_geometry_drift_fraction'], 4)} |"
            )
    algorithm_lines = [
        "| height | reference | algorithm | cold terminal mean | hot terminal mean | hot−cold | cold RMSE | hot RMSE |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in height_delta:
        algorithm_lines.append(
            f"| {int(row['nominal_height_mm'])} mm | {row['reference']} | {row['algorithm']} | "
            f"{fmt(row['cold_terminal_mean_mm'])} | {fmt(row['hot_terminal_mean_mm'])} | "
            f"{fmt(row['hot_minus_cold_same_png_mm'])} | "
            f"{fmt(row['cold_terminal_frame_rmse_mm'])} | "
            f"{fmt(row['hot_terminal_frame_rmse_mm'])} |"
        )
    return f"""# Thermal-A3-2｜Cold-vs-Hot Session 热漂来源闭环

## 明确结论

- `HOT_SESSION_REMOVES_GLOBAL_DRIFT = {conclusions['HOT_SESSION_REMOVES_GLOBAL_DRIFT']}`
- `RESIDUAL_LOCAL_THERMAL_DRIFT_REMAINS = {conclusions['RESIDUAL_LOCAL_THERMAL_DRIFT_REMAINS']}`
- `REFERENCE_GEOMETRY_DRIFT_FRACTION = {fmt(conclusions['REFERENCE_GEOMETRY_DRIFT_FRACTION'], 4)}`
- `GROUND_REFERENCE_GEOMETRY_DRIFT_FRACTION = {fmt(conclusions['GROUND_REFERENCE_GEOMETRY_DRIFT_FRACTION'], 4)}`
- `SESSION_BASE_TERMINAL_RMSE_REDUCTION_FRACTION = {fmt(conclusions['SESSION_BASE_TERMINAL_RMSE_REDUCTION_FRACTION'], 4)}`

主 fraction 使用 20/30/10 mm 三个量块等权的 Base Session accumulated-drift L1：`1 - Σ|hot residual| / Σ|cold accumulated|`。不把 Ground 与高度混成一个数；Ground fraction 独立报告。负值不截断，表示 hot Session 加重残差。

## Artifact provenance / paired protocol

- A3-1 manifest SHA256：`{provenance['a3']['manifest_sha256']}`；全部 A3-1 输出 hash 复核通过。
- Frozen ROI：`FROZEN_USER_CONFIRMED`，SHA256 `{provenance['registry_sha256']}`；没有移动或重选 ROI。
- cold Session SHA256：`{provenance['cold_session_sha256']}`；hot Session SHA256：`{provenance['hot_session_sha256']}`。两者 schema、board、reference extrinsic 相同，且均为 `VALID`。
- 末端窗口预注册为最后 {len(terminal_ids)} 个正式 recording：`{', '.join(terminal_ids)}`。选择只依据 recording order，不使用高度、误差或漂移结果。
- 两个 condition 使用完全相同的 {len(terminal_ids) * 20} 张 PNG；source-set SHA256 `{provenance['source_set_sha256']}`。Steger point count、q1/q2、ROI、exposure/gain/pixel format 和 reconstruction support 已逐 frame 配对核验。
- cold replay 与 A3-1 原结果逐组合复现到 `1e-10 mm`；hot replay 只替换 `session_extrinsic R/t` 与 `session_ground_reference`。没有重拟 C0/C1/H1/H-B2/Ground 或外参。

## Cold/hot Session geometry

- camera→ground relative rotation：{delta['relative_rotation_deg']:.9f}°。
- translation delta：`[{', '.join(f'{value:+.9f}' for value in delta['translation_delta_mm'])}] mm`，norm {delta['translation_delta_norm_mm']:.9f} mm。
- Ground intercept：Δ {delta['ground_delta_intercept_mm']:+.9f} mm；slope：Δ {delta['ground_delta_slope_mm_per_mm']:+.12f} mm/mm；在共同 valid S span 上等效 tilt delta {delta['ground_delta_tilt_across_common_span_mm']:+.9f} mm。
- valid S common domain：[{delta['ground_common_s_min_mm']:.6f}, {delta['ground_common_s_max_mm']:.6f}] mm。

## Ground before/after

- cold terminal raw Ground b：{fmt(ground_delta['cold_terminal_raw_ground_offset_b_mm'])} mm；hot terminal raw Ground b：{fmt(ground_delta['hot_terminal_raw_ground_offset_b_mm'])} mm。该项只反映 R/t 坐标变化。
- cold terminal Session-leveled Ground b：{fmt(ground_delta['cold_terminal_session_leveled_ground_offset_b_mm'])} mm；hot terminal：{fmt(ground_delta['hot_terminal_session_leveled_ground_offset_b_mm'])} mm。
- 相对首个 cold recording，cold accumulated / hot residual 分别为 {fmt(ground_delta['cold_accumulated_drift_mm'])} / {fmt(ground_delta['hot_residual_drift_mm'])} mm；Ground explained fraction {fmt(ground_delta['reference_geometry_drift_fraction'], 4)}。

## 量块 Base 来源闭环

{chr(10).join(base_lines)}

`REFERENCE_GEOMETRY_DRIFT_FRACTION` 的分母 L1 为 {fmt(conclusions['SESSION_BASE_COLD_ACCUMULATED_L1_MM'])} mm，hot residual L1 为 {fmt(conclusions['SESSION_BASE_HOT_RESIDUAL_L1_MM'])} mm。Local hot residual 最大值为 {fmt(conclusions['MAX_HOT_LOCAL_BASE_RESIDUAL_MM'])} mm，超过预注册 material threshold {MATERIAL_DRIFT_MM:.3f} mm 时判定残余局部热漂仍存在。

这里必须区分两个问题：相对首个 cold recording 的 accumulated-drift fraction 为 {fmt(conclusions['REFERENCE_GEOMETRY_DRIFT_FRACTION'], 4)}，说明 hot Session 使部分量块跨过初始基线并产生 over-correction；但相对 nominal 的 Session Base terminal frame RMSE 总量下降 {conclusions['SESSION_BASE_TERMINAL_RMSE_REDUCTION_FRACTION']:.1%}。因此 hot geometry 显著改善绝对高度误差，却不能作为一个统一 additive reference correction 还原整场 accumulated drift。

## Base/H1/H-B2 对照

{chr(10).join(algorithm_lines)}

H1/H-B2 只使用同一正式 correction resolver；表中的变化源自输入 Base 在 hot Session geometry 下变化，不是重新训练或新补偿。

## 解释边界

- 这是同一批末端 PNG 的 counterfactual replay，能够分离“更换 Session geometry 能解释多少”；不是第二次独立 cold-start。
- hot Session 同时包含新的 R/t 与新的 Session Ground reference，主结论解释的是二者合并的 reference geometry contribution，不能由本实验再唯一拆分 PnP 与 Ground-fit 各自比例。
- 20/30/10 mm 与 upper/middle/lower 共线，残余差异不能唯一归因于高度或位置。
- 所有数值仍是本次单 Session observed result，不是系统理论 worst case。
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("morning_dir", "afternoon_dir", "cold_session", "hot_session", "a3_dir", "a2_dir", "registry", "measure_config", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.terminal_k != TERMINAL_K:
        raise ThermalA32Error(f"Formal protocol freezes terminal-k at {TERMINAL_K}")

    a3_audit = verify_a3(args.a3_dir)
    _, rois, registry_provenance = a2.validate_registry(
        args.registry, a2.EXPECTED_REGISTRY_SHA256, args.measure_config, args.cold_session
    )
    cold_payload, hot_payload, session_delta = validate_session_pair(args.cold_session, args.hot_session)
    a3_rows = read_csv(args.a3_dir / "thermal_a3_full_day_results.csv")
    inventory, terminal_ids = select_terminal_rows(a3_rows, args.afternoon_dir, args.terminal_k)
    if args.smoke:
        inventory = inventory[-1:]

    cold_frames, cold_heights, cold_raw_ground, cold_leveled_ground = replay_condition(
        "cold_session", args.cold_session, args.measure_config, inventory, rois
    )
    hot_frames, hot_heights, hot_raw_ground, hot_leveled_ground = replay_condition(
        "hot_session", args.hot_session, args.measure_config, inventory, rois
    )
    if args.smoke:
        assert_paired_invariants(cold_frames, hot_frames, cold_heights, a3_rows)
        print("A3_2_SMOKE_RUN_COMPLETE: formal outputs were not written", flush=True)
        return 0
    assert_paired_invariants(cold_frames, hot_frames, cold_heights, a3_rows)
    if not (
        len(cold_frames) == len(hot_frames) == TERMINAL_K * 20 * 3
        and len(cold_heights) == len(hot_heights) == TERMINAL_K * 3 * 2 * 3
    ):
        raise ThermalA32Error("Paired replay cardinality mismatch")

    first_lookup = first_height_lookup(a3_rows)
    height_delta = build_height_delta_rows(
        cold_heights, hot_heights, cold_frames, hot_frames, first_lookup
    )
    first_id = min(a3_rows, key=lambda row: int(row["recording_order"]))["recording_id"]
    initial_inventory = [
        {
            "recording_id": first_id,
            "relative_path": str((args.morning_dir / first_id).resolve()),
            "segment": "pre_reconnect",
            "frame_count": "20",
            "first_frame_time_local": next(row["recording_time_local"] for row in a3_rows if row["recording_id"] == first_id),
            "elapsed_from_power_start_s": str(float(next(row["elapsed_from_power_min"] for row in a3_rows if row["recording_id"] == first_id)) * 60.0),
            "elapsed_from_reference_start_s": str(float(next(row["elapsed_from_reference_min"] for row in a3_rows if row["recording_id"] == first_id)) * 60.0),
        }
    ]
    cold_app, cold_calibration, _, cold_reference = a2.load_chain(args.measure_config, args.cold_session)
    initial_ground = ground_session_replay(initial_inventory, rois, cold_app, cold_calibration, cold_reference)[0]
    ground_delta = build_ground_delta_row(
        cold_leveled_ground,
        hot_leveled_ground,
        float(initial_ground["raw_ground_offset_b_mm"]),
        float(initial_ground["session_leveled_ground_offset_b_mm"]),
    )
    conclusions = derive_conclusions(height_delta, ground_delta)
    conclusion_rows = [
        {
            "row_type": "conclusion",
            "metric": "REFERENCE_GEOMETRY_DRIFT_FRACTION",
            "cold_value": conclusions["SESSION_BASE_COLD_ACCUMULATED_L1_MM"],
            "hot_value": conclusions["SESSION_BASE_HOT_RESIDUAL_L1_MM"],
            "hot_minus_cold": conclusions["REFERENCE_GEOMETRY_DRIFT_FRACTION"],
            "unit": "fraction",
            "conclusion_value": conclusions["REFERENCE_GEOMETRY_DRIFT_FRACTION"],
        },
        {
            "row_type": "conclusion",
            "metric": "GROUND_REFERENCE_GEOMETRY_DRIFT_FRACTION",
            "hot_minus_cold": conclusions["GROUND_REFERENCE_GEOMETRY_DRIFT_FRACTION"],
            "unit": "fraction",
            "conclusion_value": conclusions["GROUND_REFERENCE_GEOMETRY_DRIFT_FRACTION"],
        },
        {
            "row_type": "conclusion",
            "metric": "HOT_SESSION_REMOVES_GLOBAL_DRIFT",
            "unit": "enum",
            "conclusion_value": conclusions["HOT_SESSION_REMOVES_GLOBAL_DRIFT"],
        },
        {
            "row_type": "conclusion",
            "metric": "RESIDUAL_LOCAL_THERMAL_DRIFT_REMAINS",
            "unit": "enum",
            "conclusion_value": conclusions["RESIDUAL_LOCAL_THERMAL_DRIFT_REMAINS"],
        },
    ]
    delta_rows = (
        configuration_delta_rows(cold_payload, hot_payload, session_delta)
        + height_delta
        + [ground_delta]
        + conclusion_rows
    )
    replay_rows = decorate_replay_rows(cold_heights + hot_heights, cold_leveled_ground + hot_leveled_ground, first_lookup)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "thermal_a3_cold_hot_session_delta.csv", delta_rows)
    write_csv(args.output_dir / "thermal_a3_hot_replay_results.csv", replay_rows)
    save_plot(args.output_dir, session_delta, height_delta, ground_delta, conclusions)
    provenance = {
        "a3": a3_audit,
        "registry_sha256": registry_provenance["registry_sha256"],
        "registry_status": registry_provenance["registry_status"],
        "cold_session_sha256": session_delta["cold_sha256"],
        "hot_session_sha256": session_delta["hot_sha256"],
        "measure_config_sha256": a2.sha256_file(args.measure_config),
        "source_set_sha256": source_set_sha256(inventory),
        "terminal_window_rule": "LAST_5_FORMAL_RECORDINGS_BY_RECORDING_ORDER",
        "terminal_recording_ids": terminal_ids,
        "same_pngs_both_conditions": True,
        "cold_replay_matches_a3_tolerance_mm": 1e-10,
        "models_refit": [],
        "roi_reselected": False,
        "height_shadow_used": False,
    }
    report = build_report(provenance, session_delta, terminal_ids, height_delta, ground_delta, conclusions)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    outputs = ["thermal_a3_cold_hot_session_delta.csv", "thermal_a3_hot_replay_results.csv", "thermal_a3_cold_vs_hot.png", "report.md"]
    write_json(
        args.output_dir / "thermal_a3_cold_hot_run_manifest.json",
        {
            "status": "COMPLETE",
            "protocol": "Thermal-A3-2 paired cold-vs-hot Session-only replay",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance": provenance,
            "session_configuration_delta": session_delta,
            "conclusions": conclusions,
            "cardinality": {
                "terminal_recordings": TERMINAL_K,
                "unique_pngs": TERMINAL_K * 20,
                "replay_conditions": 2,
                "frame_object_rows_internal": len(cold_frames) + len(hot_frames),
                "height_recording_rows": len(cold_heights) + len(hot_heights),
                "hot_replay_output_rows": len(replay_rows),
                "delta_output_rows": len(delta_rows),
            },
            "output_sha256": {name: a2.sha256_file(args.output_dir / name) for name in outputs},
        },
    )
    print(json.dumps(a2.json_safe(conclusions), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ThermalA32Error, a2.ThermalA2Error) as error:
        print(f"ERROR: {error}", flush=True)
        raise SystemExit(2)
