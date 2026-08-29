#!/usr/bin/env python3
"""Task A-13B: replay Session01 with the frozen PNG/Steger/ROI artifacts.

This is a validation-only script.  It never runs Steger, never reads
``height_shadow.csv`` for formal geometry, and never fits C0/C1/Ground/H1/H-B2
or a new spatial correction.  The only centerline input is the A-13A NPZ cache;
the only ROI input is the manually frozen geometry-only registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "laser_measurement_tool"
DATA_ROOT = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0822\session01"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze"
CONFIG_PATH = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
MANIFEST_PATH = TOOL_ROOT / "configs" / "calibration_daheng_0811" / "manifest.yaml"
CACHE_NPZ = OUTPUT_DIR / "session01_steger_centers.npz"
CACHE_MANIFEST = OUTPUT_DIR / "session01_steger_centers_manifest.json"
REGISTRY_PATH = OUTPUT_DIR / "session01_roi_registry_manual.json"
GROUND_PATH = DATA_ROOT / "session_ground_calibration.json"

HEIGHT_LABELS = ("h10", "h20", "h30")
POSITION_IDS = tuple(f"p{i:02d}" for i in range(1, 11))
REPEAT_COUNT = 20
CONDITION_COUNT = len(HEIGHT_LABELS) * len(POSITION_IDS)
FRAME_COUNT = CONDITION_COUNT * REPEAT_COUNT
FULL_SENSOR_WIDTH = 4096.0
MODEL_NAMES = ("base", "h1", "hb2")
EDGE_THRESHOLDS = (2200.0, 2400.0, 2600.0)

sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from correction.stage_a_height_scale import (  # noqa: E402
    resolve_height_correction,
)
from measurement.ground_reference import (  # noqa: E402
    MeasurementError,
    SessionGroundReference,
)
from measurement.height_measure import measure_height_line  # noqa: E402
from reconstruction.reconstructor import (  # noqa: E402
    ReconstructionInputError,
    reconstruct_uv_to_ground,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def bool_text(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    return "YES" if bool(value) else "NO"


def parse_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def parse_float(value: Any) -> float | None:
    return finite(value)


def condition_id(height_label: str, position_id: str) -> str:
    return f"{height_label}_{position_id}"


def model_error_values(rows: Iterable[dict[str, Any]], model: str) -> np.ndarray:
    values = [finite(row.get(f"residual_{model}")) for row in rows]
    return np.asarray([value for value in values if value is not None], dtype=np.float64)


def basic_metrics(rows: Iterable[dict[str, Any]], model: str) -> dict[str, Any]:
    row_list = list(rows)
    errors = model_error_values(row_list, model)
    abs_error = np.abs(errors)
    result: dict[str, Any] = {
        "n_total": len(row_list),
        "n_valid": int(len(errors)),
        "n_invalid": int(len(row_list) - len(errors)),
        "invalid_rate": (
            float((len(row_list) - len(errors)) / len(row_list)) if row_list else None
        ),
        "bias_mm": None,
        "mae_mm": None,
        "rmse_mm": None,
        "p95_abs_mm": None,
        "max_abs_mm": None,
        "repeatability_std_mm": None,
    }
    if len(errors):
        result.update(
            {
                "bias_mm": float(np.mean(errors)),
                "mae_mm": float(np.mean(abs_error)),
                "rmse_mm": float(np.sqrt(np.mean(errors * errors))),
                "p95_abs_mm": float(np.percentile(abs_error, 95)),
                "max_abs_mm": float(np.max(abs_error)),
                "repeatability_std_mm": float(np.std(errors, ddof=1))
                if len(errors) > 1
                else 0.0,
            }
        )
    return result


def clamp_status(flags: np.ndarray | None) -> str:
    if flags is None:
        return "NOT_APPLICABLE"
    values = np.asarray(flags, dtype=bool).reshape(-1)
    if not len(values):
        return "NOT_APPLICABLE"
    if bool(np.all(values)):
        return "CLAMPED"
    if bool(np.any(values)):
        return "MIXED"
    return "IN_DOMAIN"


def roi_mask(points_uv: np.ndarray, ranges: list[list[float]]) -> np.ndarray:
    """RoiManager-compatible full-sensor rectangular mask."""
    points = np.asarray(points_uv, dtype=np.float64)
    mask = np.zeros(len(points), dtype=bool)
    if not len(points):
        return mask
    point_x = points[:, 0] + 0.5
    point_y = points[:, 1] + 0.5
    for top, bottom in ranges:
        mask |= (
            (point_x >= 0.0)
            & (point_x <= FULL_SENSOR_WIDTH)
            & (point_y >= float(top))
            & (point_y <= float(bottom))
        )
    return mask


def read_frames_csv(condition_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = condition_dir / "frames.csv"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    by_filename: dict[str, dict[str, Any]] = {}
    duplicate_filenames: list[str] = []
    camera_ids: list[int] = []
    for row in rows:
        filename = str(row.get("filename", "")).strip()
        if filename in by_filename:
            duplicate_filenames.append(filename)
        by_filename[filename] = row
        camera_number = parse_int(row.get("camera_frame_number"))
        if camera_number is not None:
            camera_ids.append(camera_number)
    duplicate_camera_ids = sorted(
        {number for number in camera_ids if camera_ids.count(number) > 1}
    )
    actual_gaps: list[int] = []
    sorted_ids = sorted(camera_ids)
    for previous, current in zip(sorted_ids, sorted_ids[1:]):
        if current - previous - 1:
            actual_gaps.append(current - previous - 1)
    reported_gaps = [
        parse_int(row.get("frame_gap"))
        for row in rows
        if parse_int(row.get("frame_gap")) not in (None, 0)
    ]
    audit = {
        "frames_csv_exists": path.is_file(),
        "frames_csv_row_count": len(rows),
        "duplicate_filename_count": len(set(duplicate_filenames)),
        "duplicate_camera_frame_id_count": len(duplicate_camera_ids),
        "actual_frame_gap_values": sorted(set(actual_gaps)),
        "reported_frame_gap_values": sorted(set(reported_gaps)),
        "camera_frame_min": min(camera_ids) if camera_ids else None,
        "camera_frame_max": max(camera_ids) if camera_ids else None,
        "offset_values": sorted(
            {
                (
                    parse_int(row.get("offset_x")),
                    parse_int(row.get("offset_y")),
                )
                for row in rows
            }
        ),
        "size_values": sorted(
            {
                (
                    parse_int(row.get("width")),
                    parse_int(row.get("height")),
                )
                for row in rows
            }
        ),
    }
    return by_filename, audit


def load_cache_and_registry() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]
]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    with np.load(CACHE_NPZ, allow_pickle=False) as bundle:
        centers = np.asarray(bundle["centers_full"], dtype=np.float64)
        offsets = np.asarray(bundle["frame_offsets"], dtype=np.int64)
    frames = cache_manifest.get("frames", [])
    if offsets.ndim != 1 or len(offsets) != len(frames) + 1:
        raise RuntimeError("A-13A center cache frame_offsets shape is invalid")
    if offsets[0] != 0 or offsets[-1] != len(centers):
        raise RuntimeError("A-13A center cache frame_offsets bounds are invalid")
    centers_by_key: dict[str, np.ndarray] = {}
    for index, frame in enumerate(frames):
        start, end = int(offsets[index]), int(offsets[index + 1])
        frame_centers = np.ascontiguousarray(centers[start:end], dtype=np.float64)
        if frame_centers.ndim != 2 or frame_centers.shape[1] != 2 or not len(frame_centers):
            raise RuntimeError(f"A-13A cache frame has no usable centerline: {index}")
        centers_by_key[str(frame["cache_key"])] = frame_centers
    cache_info = {
        "centers_total": int(len(centers)),
        "frames_total": len(frames),
        "frame_offsets_shape": list(offsets.shape),
        "one_steger_per_frame": cache_manifest.get("one_steger_per_frame"),
        "reused_existing_cache_manifest_value": cache_manifest.get("reused_existing_cache"),
    }
    return (
        frames,
        cache_manifest,
        registry,
        centers_by_key,
        cache_info,
    )


def load_session_reference(payload: dict[str, Any]) -> SessionGroundReference:
    """Hydrate the already-saved Session Ground fit; this function never fits."""
    if payload.get("status") != "VALID" or payload.get("valid") is not True:
        raise RuntimeError("session_ground_calibration.json top-level status is not VALID")
    runtime = payload.get("runtime", {})
    if runtime.get("ground_extrinsic_source") != "session":
        raise RuntimeError("Session Ground runtime source is not session")
    ground = payload.get("session_ground_reference", {})
    if ground.get("status") != "VALID":
        raise RuntimeError("Session Ground Reference status is not VALID")
    return SessionGroundReference(
        origin_xy=np.asarray(ground["origin_xy"], dtype=np.float64),
        direction_xy=np.asarray(ground["direction_xy"], dtype=np.float64),
        slope_z_per_mm=float(ground["slope_z_per_mm"]),
        intercept_z_mm=float(ground["intercept_z_mm"]),
        rmse_mm=float(ground["rmse_mm"]),
        valid_s_range_mm=tuple(float(value) for value in ground["valid_s_range_mm"]),
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


def provenance_audit(
    frames: list[dict[str, Any]],
    cache_manifest: dict[str, Any],
    registry: dict[str, Any],
    ground_payload: dict[str, Any],
    app: Any,
    cache_info: dict[str, Any],
) -> dict[str, Any]:
    protocol = cache_manifest.get("protocol_key", {})
    current_manifest_sha = sha256_file(MANIFEST_PATH)
    current_config_sha = sha256_file(CONFIG_PATH)
    artifact_paths = {
        "frozen_intrinsics": app.calibration.intrinsics,
        "frozen_c0_laser_model": app.calibration.laser_model,
        "frozen_reference_extrinsics": app.calibration.extrinsics,
        "frozen_c1": app.calibration.laser_ray_correction,
        "frozen_h1": app.correction.stage_a_height_scale_config,
        "frozen_hb2": app.correction.hb2_height_correction_config,
    }
    artifact_hashes = {
        name: {
            "path": str(path.resolve()) if path is not None else None,
            "sha256": sha256_file(path) if path is not None and path.is_file() else None,
        }
        for name, path in artifact_paths.items()
    }
    registry_entries = registry.get("entries", [])
    expected_cache_keys = {
        f"{frame.get('height_label')}_{frame.get('position_id')}/frame_{int(frame.get('repeat_index', 0)):06d}.png"
        for frame in frames
    }
    raw_hash_matches = 0
    raw_hash_missing = 0
    frames_csv_mismatches = 0
    condition_audits: dict[str, Any] = {}
    for frame in frames:
        height_label = str(frame.get("height_label"))
        position_id = str(frame.get("position_id"))
        cid = condition_id(height_label, position_id)
        source_path = DATA_ROOT / height_label / cid / str(frame.get("filename"))
        source_exists = source_path.is_file()
        source_hash = sha256_file(source_path) if source_exists else None
        if source_hash and source_hash == frame.get("source_sha256"):
            raw_hash_matches += 1
        else:
            raw_hash_missing += 1
        by_filename, audit = condition_audits.get(cid, (None, None))
        if by_filename is None:
            by_filename, audit = read_frames_csv(DATA_ROOT / height_label / cid)
            condition_audits[cid] = (by_filename, audit)
        csv_row = by_filename.get(str(frame.get("filename")))
        match = bool(csv_row is not None)
        if csv_row is not None:
            expected = {
                "camera_frame_number": parse_int(frame.get("camera_frame_number")),
                "offset_x": parse_int((frame.get("offset_xy") or [None, None])[0]),
                "offset_y": parse_int((frame.get("offset_xy") or [None, None])[1]),
            }
            actual = {
                "camera_frame_number": parse_int(csv_row.get("camera_frame_number")),
                "offset_x": parse_int(csv_row.get("offset_x")),
                "offset_y": parse_int(csv_row.get("offset_y")),
            }
            match = expected == actual
        if not match:
            frames_csv_mismatches += 1
    condition_audit_json = {}
    for cid, (by_filename, audit) in condition_audits.items():
        pngs = sorted((DATA_ROOT / cid.split("_")[0] / cid).glob("frame_*.png"))
        condition_audit_json[cid] = {
            **audit,
            "raw_png_count": len(pngs),
            "raw_png_missing_from_frames_csv": sorted(
                path.name for path in pngs if path.name not in by_filename
            ),
        }
    session = ground_payload.get("session_ground_reference", {})
    pnp = ground_payload.get("detection", {})
    pnp_valid = bool(
        ground_payload.get(
            "pnp_valid",
            pnp.get("corner_count") == 88
            and finite(pnp.get("reprojection_rmse_px")) is not None
            and float(pnp.get("reprojection_rmse_px")) <= 0.5
            and bool(ground_payload.get("reference_extrinsic"))
            and bool(ground_payload.get("session_extrinsic")),
        )
    )
    ground_valid = bool(
        ground_payload.get(
            "ground_valid",
            ground_payload.get("session_ground_reference_status") == "VALID"
            and session.get("status") == "VALID"
            and session.get("support", {}).get("status") == "applied",
        )
    )
    registry_ok = bool(
        registry.get("dataset") == "session01"
        and registry.get("frozen") is True
        and registry.get("manual_confirmed") is True
        and int(registry.get("manual_confirmed_count", -1)) == CONDITION_COUNT
        and len(registry_entries) == CONDITION_COUNT
        and all(entry.get("manual_confirmed") is True for entry in registry_entries)
        and all(str(entry.get("review_status", "")).startswith("FROZEN") for entry in registry_entries)
        and registry.get("geometry_only") is True
        and registry.get("height_shadow_used_for_formal_geometry") is False
    )
    cache_ok = bool(
        cache_info["frames_total"] == FRAME_COUNT
        and cache_info["one_steger_per_frame"] is True
        and len(expected_cache_keys) == FRAME_COUNT
        and protocol.get("extraction_method") == "steger"
        and protocol.get("full_sensor_coordinate_system") is True
        and protocol.get("height_shadow_used_for_formal_geometry") is False
        and protocol.get("frozen_manifest_sha256") == current_manifest_sha
        and all(int(frame.get("steger_run_count", 0)) == 1 for frame in frames)
    )
    config_values_ok = bool(
        app.system == "daheng"
        and app.reconstruction.min_camera_depth_mm == 630.0
        and app.reconstruction.max_camera_depth_mm == 715.0
        and app.reconstruction.model_range_margin_mm == 2.0
        and app.reconstruction.enable_laser_ray_correction is True
        and app.measurement.min_baseline_points == 20
        and app.measurement.min_height_points == 20
        and app.correction.stage_a_height_scale is not None
        and app.correction.hb2_height_correction is not None
    )
    session_ground_ok = bool(
        ground_payload.get("status") == "VALID"
        and ground_payload.get("valid") is True
        and pnp_valid
        and ground_valid
        and ground_payload.get("session_ground_reference_status") == "VALID"
        and ground_payload.get("runtime", {}).get("ground_extrinsic_source") == "session"
        and session.get("status") == "VALID"
        and session.get("support_source") == "pnp_board_mask"
    )
    source_ok = raw_hash_matches == FRAME_COUNT and raw_hash_missing == 0 and frames_csv_mismatches == 0
    return {
        "dataset_root": str(DATA_ROOT),
        "config_path": str(CONFIG_PATH.resolve()),
        "config_sha256": current_config_sha,
        "frozen_artifact_hashes": artifact_hashes,
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "manifest_sha256": current_manifest_sha,
        "registry_path": str(REGISTRY_PATH.resolve()),
        "cache_npz_path": str(CACHE_NPZ.resolve()),
        "cache_manifest_path": str(CACHE_MANIFEST.resolve()),
        "registry_ok": registry_ok,
        "cache_ok": cache_ok,
        "config_values_ok": config_values_ok,
        "session_ground_ok": session_ground_ok,
        "raw_hash_matches": raw_hash_matches,
        "raw_hash_missing_or_mismatch": raw_hash_missing,
        "frames_csv_mismatches": frames_csv_mismatches,
        "source_identity_ok": source_ok,
        "replay_provenance_match": bool(registry_ok and cache_ok and config_values_ok and session_ground_ok and source_ok),
        "registry_entry_count": len(registry_entries),
        "cache_info": cache_info,
        "cache_manifest_reused_existing_cache": cache_manifest.get("reused_existing_cache"),
        "a13a_report_cache_reuse_note": "The A-13A report may show cache_reused=True after reuse while the immutable cache manifest still says reused_existing_cache=False; A-13B uses the actual manifest/NPZ and does not rewrite them.",
        "pnp": {
            "status": pnp_valid,
            "corner_count": pnp.get("corner_count"),
            "reprojection_rmse_px": pnp.get("reprojection_rmse_px"),
            "reference_R": ground_payload.get("reference_extrinsic", {}).get("R_camera_to_ground"),
            "reference_t": ground_payload.get("reference_extrinsic", {}).get("t_camera_to_ground_mm"),
        },
        "session_extrinsic": {
            "R": ground_payload.get("session_extrinsic", {}).get("R_camera_to_ground"),
            "t": ground_payload.get("session_extrinsic", {}).get("t_camera_to_ground_mm"),
            "delta": ground_payload.get("delta"),
            "generation": ground_payload.get("runtime", {}).get("ground_extrinsic_generation"),
        },
        "session_ground": {
            "status": ground_payload.get("session_ground_reference_status"),
            "source": session.get("source"),
            "fit_source": session.get("fit_source"),
            "support_source": session.get("support_source"),
            "slope": session.get("slope"),
            "intercept": session.get("intercept"),
            "rmse_mm": session.get("rmse_mm"),
            "valid_s_range_mm": session.get("valid_s_range_mm"),
            "point_count": session.get("point_count"),
            "inlier_count": session.get("inlier_count"),
            "origin_xy": session.get("origin_xy"),
            "direction_xy": session.get("direction_xy"),
            "support": session.get("support"),
        },
        "condition_audit": condition_audit_json,
    }


def reconstruct_one_frame(
    frame: dict[str, Any],
    centers: np.ndarray,
    roi: dict[str, Any],
    calibration: dict[str, Any],
    app: Any,
    ground_reference: SessionGroundReference,
    source_qc: dict[str, Any],
) -> dict[str, Any]:
    height_range = [[float(v) for v in roi["height_v_range"]]]
    baseline_ranges = [[float(v) for v in pair] for pair in roi["baseline_v_ranges"]]
    before_range = [baseline_ranges[0]]
    after_range = [baseline_ranges[-1]]
    baseline_before_cache = roi_mask(centers, before_range)
    height_cache = roi_mask(centers, height_range)
    baseline_after_cache = roi_mask(centers, after_range)
    baseline_cache = baseline_before_cache | baseline_after_cache
    try:
        reconstruction = reconstruct_uv_to_ground(centers, calibration, app.reconstruction)
        points_ground, ground_valid = ground_reference.apply_to_points(
            reconstruction.points_ground
        )
        pixels = reconstruction.pixels_uv
        before_mask = roi_mask(pixels, before_range)
        height_mask = roi_mask(pixels, height_range)
        after_mask = roi_mask(pixels, after_range)
        baseline_mask = before_mask | after_mask

        baseline_ground = points_ground[baseline_mask]
        height_ground = points_ground[height_mask]
        measurement = measure_height_line(
            baseline_ground,
            height_ground,
            app.measurement,
            ground_correction_mode="session_reference",
        )
        h_base = float(measurement.height_mean_mm)

        height_q1 = (
            None
            if reconstruction.q1_c0 is None
            else np.asarray(reconstruction.q1_c0)[height_mask]
        )
        height_q2 = (
            None
            if reconstruction.q2_c0 is None
            else np.asarray(reconstruction.q2_c0)[height_mask]
        )
        q1_values = (
            np.asarray(height_q1, dtype=np.float64)
            if height_q1 is not None
            else np.empty(0, dtype=np.float64)
        )
        q2_values = (
            np.asarray(height_q2, dtype=np.float64)
            if height_q2 is not None
            else np.empty(0, dtype=np.float64)
        )
        q1_finite = q1_values[np.isfinite(q1_values)]
        q2_finite = q2_values[np.isfinite(q2_values)]
        q1 = float(np.mean(q1_finite)) if len(q1_finite) else None
        q2 = float(np.mean(q2_finite)) if len(q2_finite) else None
        hb2_config = app.correction.hb2_height_correction
        if hb2_config is not None and len(q2_values):
            lower, upper = hb2_config.q2_domain
            q2_in_domain = bool(
                np.isfinite(q2_values).all()
                and np.all((q2_values >= lower) & (q2_values <= upper))
            )
        else:
            q2_in_domain = False
        h1_result = resolve_height_correction(
            h_base,
            q1=q1,
            q2=q2,
            q2_in_domain=q2_in_domain,
            system=app.system,
            correction=app.correction,
            mode_override="h1",
        )
        hb2_result = resolve_height_correction(
            h_base,
            q1=q1,
            q2=q2,
            q2_in_domain=q2_in_domain,
            system=app.system,
            correction=app.correction,
            mode_override="hb2",
        )
        height_pixels = pixels[height_mask]
        ground_ood = ~np.asarray(ground_valid, dtype=bool)
        ground_status = (
            "NO_POINTS"
            if not len(ground_valid)
            else "VALID"
            if not np.any(ground_ood)
            else "OUT_OF_VALID_S_DOMAIN"
            if np.all(ground_ood)
            else "PARTIAL_OUT_OF_VALID_S_DOMAIN"
        )
        ground_ood_count = int(np.count_nonzero(ground_ood))
        def status_for(mask: np.ndarray) -> str:
            selected = np.asarray(ground_ood[mask], dtype=bool)
            if not len(selected):
                return "NO_POINTS"
            if not np.any(selected):
                return "VALID"
            if np.all(selected):
                return "OUT_OF_VALID_S_DOMAIN"
            return "PARTIAL_OUT_OF_VALID_S_DOMAIN"

        ground_status_before = status_for(before_mask)
        ground_status_height = status_for(height_mask)
        ground_status_after = status_for(after_mask)
        ground_status_formal = status_for(baseline_mask | height_mask)
        ground_s = ground_reference.project_s(reconstruction.points_ground[:, :2]) if len(reconstruction.points_ground) else np.empty(0)
        height_s = ground_s[height_mask]
        truth_height = {"h10": 10.0, "h20": 20.0, "h30": 30.0}[frame["height_label"]]
        record: dict[str, Any] = {
            "dataset": "session01",
            "height_label": frame["height_label"],
            "position_id": frame["position_id"],
            "condition_id": f"{frame['height_label']}_{frame['position_id']}",
            "v_order_rank": roi.get("v_order_rank"),
            "repeat_index": frame.get("repeat_index"),
            "filename": frame.get("filename"),
            "camera_frame_number": frame.get("camera_frame_number"),
            "true_height_mm": truth_height,
            "truth_kind": "nominal_truth",
            "height_roi_center_v": roi.get("height_roi_center_v"),
            "height_roi_formal_v_median": float(np.median(centers[height_cache, 1])) if np.any(height_cache) else None,
            "height_roi_formal_v_min": float(np.min(centers[height_cache, 1])) if np.any(height_cache) else None,
            "height_roi_formal_v_max": float(np.max(centers[height_cache, 1])) if np.any(height_cache) else None,
            "height_v_range": roi.get("height_v_range"),
            "baseline_before_v_range": roi.get("baseline_v_ranges", [None, None])[0],
            "baseline_after_v_range": roi.get("baseline_v_ranges", [None, None])[-1],
            "edge_baseline_clipped": bool(roi.get("edge_baseline_clipped", False)),
            "cached_baseline_before_point_count": int(np.count_nonzero(baseline_before_cache)),
            "cached_height_point_count": int(np.count_nonzero(height_cache)),
            "cached_baseline_after_point_count": int(np.count_nonzero(baseline_after_cache)),
            "reconstructed_all_point_count": int(len(pixels)),
            "reconstructed_baseline_before_point_count": int(np.count_nonzero(before_mask)),
            "reconstructed_height_point_count": int(np.count_nonzero(height_mask)),
            "reconstructed_baseline_after_point_count": int(np.count_nonzero(after_mask)),
            "baseline_point_count": int(measurement.baseline_point_count),
            "baseline_inlier_count": int(measurement.baseline_inlier_count),
            "height_point_count": int(measurement.height_point_count),
            "height_inlier_count": int(measurement.height_inlier_count),
            "height_std_mm": float(measurement.height_std_mm),
            "h_base": h_base,
            "h_h1": h1_result.height_h1,
            "height_raw": h_base,
            "height_h1": h1_result.height_h1,
            "q1": q1,
            "q2": q2,
            "q2_in_domain": q2_in_domain,
            "h_hb2": hb2_result.height_hb2,
            "height_hb2": hb2_result.height_hb2,
            "residual_base": h_base - truth_height,
            "residual_h1": (
                None
                if h1_result.height_h1 is None
                else float(h1_result.height_h1) - truth_height
            ),
            "residual_hb2": (
                None
                if hb2_result.height_hb2 is None
                else float(hb2_result.height_hb2) - truth_height
            ),
            "h1_status": h1_result.h1_status,
            "hb2_status": hb2_result.hb2_q2_status,
            "hb2_q2_status": hb2_result.hb2_q2_status,
            "c1_clamp_status_all": clamp_status(reconstruction.c1_clamped),
            "c1_clamp_status_baseline_before": clamp_status(
                None if reconstruction.c1_clamped is None else reconstruction.c1_clamped[before_mask]
            ),
            "c1_clamp_status_height": clamp_status(
                None if reconstruction.c1_clamped is None else reconstruction.c1_clamped[height_mask]
            ),
            "c1_clamp_status_baseline_after": clamp_status(
                None if reconstruction.c1_clamped is None else reconstruction.c1_clamped[after_mask]
            ),
            "ground_reference_status": ground_reference.status,
            "ground_status_all": ground_status,
            "ground_status_formal": ground_status_formal,
            "ground_status_baseline_before": ground_status_before,
            "ground_status_height": ground_status_height,
            "ground_status_baseline_after": ground_status_after,
            "ground_ood_count_all": ground_ood_count,
            "ground_ood_count_baseline_before": int(np.count_nonzero(ground_ood[before_mask])),
            "ground_ood_count_height": int(np.count_nonzero(ground_ood[height_mask])),
            "ground_ood_count_baseline_after": int(np.count_nonzero(ground_ood[after_mask])),
            "ground_s_min_height": float(np.min(height_s)) if len(height_s) else None,
            "ground_s_max_height": float(np.max(height_s)) if len(height_s) else None,
            "reconstruction_filtered": reconstruction.filtered,
            "invalid_status": "NONE" if h_base is not None else "MEASUREMENT_INVALID",
            "source_png_exists": source_qc.get("source_png_exists"),
            "source_hash_match": source_qc.get("source_hash_match"),
            "frames_csv_match": source_qc.get("frames_csv_match"),
            "frame_gap": source_qc.get("frame_gap"),
            "frames_csv_row_count": source_qc.get("frames_csv_row_count"),
            "raw_png_count_in_condition": source_qc.get("raw_png_count_in_condition"),
            "ground_correction_applied": True,
            "formal_height_measurement_mode": measurement.ground_reference_mode,
        }
        if len(height_pixels):
            record["height_roi_reconstructed_v_median"] = float(np.median(height_pixels[:, 1]))
            record["height_roi_reconstructed_v_min"] = float(np.min(height_pixels[:, 1]))
            record["height_roi_reconstructed_v_max"] = float(np.max(height_pixels[:, 1]))
        else:
            record["height_roi_reconstructed_v_median"] = None
            record["height_roi_reconstructed_v_min"] = None
            record["height_roi_reconstructed_v_max"] = None
        return record
    except (ReconstructionInputError, MeasurementError, ValueError, FloatingPointError) as error:
        truth = {"h10": 10.0, "h20": 20.0, "h30": 30.0}[frame["height_label"]]
        return {
            "dataset": "session01",
            "height_label": frame["height_label"],
            "position_id": frame["position_id"],
            "condition_id": f"{frame['height_label']}_{frame['position_id']}",
            "v_order_rank": roi.get("v_order_rank"),
            "repeat_index": frame.get("repeat_index"),
            "filename": frame.get("filename"),
            "camera_frame_number": frame.get("camera_frame_number"),
            "true_height_mm": truth,
            "truth_kind": "nominal_truth",
            "height_roi_center_v": roi.get("height_roi_center_v"),
            "height_v_range": roi.get("height_v_range"),
            "baseline_before_v_range": roi.get("baseline_v_ranges", [None, None])[0],
            "baseline_after_v_range": roi.get("baseline_v_ranges", [None, None])[-1],
            "edge_baseline_clipped": bool(roi.get("edge_baseline_clipped", False)),
            "cached_baseline_before_point_count": int(np.count_nonzero(baseline_before_cache)),
            "cached_height_point_count": int(np.count_nonzero(height_cache)),
            "cached_baseline_after_point_count": int(np.count_nonzero(baseline_after_cache)),
            "h_base": None,
            "h_h1": None,
            "height_raw": None,
            "height_h1": None,
            "h_hb2": None,
            "q1": None,
            "q2": None,
            "q2_in_domain": False,
            "residual_base": None,
            "residual_h1": None,
            "residual_hb2": None,
            "height_hb2": None,
            "h1_status": "not_measured",
            "hb2_status": "not_measured",
            "hb2_q2_status": "not_measured",
            "c1_clamp_status_all": "NOT_APPLICABLE",
            "c1_clamp_status_height": "NOT_APPLICABLE",
            "ground_reference_status": ground_reference.status,
            "ground_status_all": "NOT_AVAILABLE",
            "ground_ood_count_all": None,
            "invalid_status": f"{type(error).__name__}:{error}",
            "source_png_exists": source_qc.get("source_png_exists"),
            "source_hash_match": source_qc.get("source_hash_match"),
            "frames_csv_match": source_qc.get("frames_csv_match"),
            "frame_gap": source_qc.get("frame_gap"),
            "frames_csv_row_count": source_qc.get("frames_csv_row_count"),
            "raw_png_count_in_condition": source_qc.get("raw_png_count_in_condition"),
            "ground_correction_applied": False,
            "formal_height_measurement_mode": "invalid",
        }


def build_source_qc(
    frame: dict[str, Any], audit: dict[str, Any], by_filename: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    row = by_filename.get(str(frame.get("filename")))
    source_path = (
        DATA_ROOT
        / str(frame["height_label"])
        / f"{frame['height_label']}_{frame['position_id']}"
        / str(frame["filename"])
    )
    source_exists = source_path.is_file()
    source_hash_match = bool(
        source_exists
        and frame.get("source_sha256")
        and sha256_file(source_path) == frame.get("source_sha256")
    )
    frames_csv_match = False
    frame_gap = None
    if row is not None:
        frame_gap = parse_int(row.get("frame_gap"))
        frames_csv_match = (
            parse_int(row.get("camera_frame_number")) == parse_int(frame.get("camera_frame_number"))
            and parse_int(row.get("offset_x")) == parse_int((frame.get("offset_xy") or [None, None])[0])
            and parse_int(row.get("offset_y")) == parse_int((frame.get("offset_xy") or [None, None])[1])
        )
    return {
        "source_png_exists": source_exists,
        "source_hash_match": source_hash_match,
        "frames_csv_match": frames_csv_match,
        "frame_gap": frame_gap,
        "frames_csv_row_count": audit.get("frames_csv_row_count"),
        "raw_png_count_in_condition": len(
            list((source_path.parent).glob("frame_*.png"))
        ),
    }


def condition_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    output: list[dict[str, Any]] = []
    for cid in sorted(grouped):
        group = grouped[cid]
        for model in MODEL_NAMES:
            metrics = basic_metrics(group, model)
            row = {
                "dataset": "session01",
                "condition_id": cid,
                "height_label": group[0]["height_label"],
                "position_id": group[0]["position_id"],
                "v_order_rank": group[0].get("v_order_rank"),
                "height_roi_center_v": group[0].get("height_roi_center_v"),
                "true_height_mm": group[0].get("true_height_mm"),
                "truth_kind": "nominal_truth",
                "edge_baseline_clipped": group[0].get("edge_baseline_clipped"),
                "model": model,
                **metrics,
                "ground_ood_frame_count": sum(
                    1 for item in group if (item.get("ground_ood_count_all") or 0) > 0
                ),
                "q2_ood_frame_count": sum(
                    1 for item in group if item.get("q2_in_domain") is False
                ),
            }
            output.append(row)
    return output


def height_spatial_metrics(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for height in HEIGHT_LABELS:
        for model in MODEL_NAMES:
            items = [
                row
                for row in condition_rows
                if row["height_label"] == height and row["model"] == model and row["bias_mm"] is not None
            ]
            biases = np.asarray([float(row["bias_mm"]) for row in items], dtype=np.float64)
            if len(biases):
                abs_bias = np.abs(biases)
                worst_bias_row = items[int(np.argmax(abs_bias))]
                worst_p95_row = max(items, key=lambda row: float(row["p95_abs_mm"] or -np.inf))
                worst_max_row = max(items, key=lambda row: float(row["max_abs_mm"] or -np.inf))
                bias_range = float(np.max(biases) - np.min(biases)) if len(biases) > 1 else 0.0
                bias_std = float(np.std(biases))
            else:
                worst_bias_row = worst_p95_row = worst_max_row = {}
                bias_range = bias_std = None
            output.append(
                {
                    "dataset": "session01",
                    "height_label": height,
                    "model": model,
                    "truth_height_mm": {"h10": 10.0, "h20": 20.0, "h30": 30.0}[height],
                    "valid_position_count": len(items),
                    "position_bias_range_mm": bias_range,
                    "position_bias_std_mm": bias_std,
                    "worst_position_abs_bias_mm": (
                        float(worst_bias_row.get("bias_mm")) if worst_bias_row else None
                    ),
                    "worst_position_abs_bias_position_id": worst_bias_row.get("position_id"),
                    "worst_position_p95_abs_mm": (
                        float(worst_p95_row.get("p95_abs_mm")) if worst_p95_row else None
                    ),
                    "worst_position_p95_position_id": worst_p95_row.get("position_id"),
                    "worst_position_max_abs_mm": (
                        float(worst_max_row.get("max_abs_mm")) if worst_max_row else None
                    ),
                    "worst_position_max_position_id": worst_max_row.get("position_id"),
                    "position_bias_values_mm": {
                        str(row["position_id"]): row["bias_mm"] for row in items
                    },
                }
            )
    return output


def spearman_residual_v(rows: list[dict[str, Any]], model: str) -> tuple[float | None, float | None]:
    pairs = [
        (finite(row.get("height_roi_center_v")), finite(row.get(f"residual_{model}")))
        for row in rows
    ]
    pairs = [(v, e) for v, e in pairs if v is not None and e is not None]
    if len(pairs) < 3:
        return None, None
    v = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    e = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.ptp(v) <= 0.0 or np.ptp(e) <= 0.0:
        return None, None
    result = spearmanr(v, e)
    return float(result.statistic), float(result.pvalue)


def edge_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes: list[tuple[str, Any]] = [("pooled", None)]
    scopes.extend((f"height:{height}", {"height_label": height}) for height in HEIGHT_LABELS)
    scopes.extend(
        (f"position:{height}:{position}", {"height_label": height, "position_id": position})
        for height in HEIGHT_LABELS
        for position in POSITION_IDS
    )
    regions: list[tuple[str, float | None]] = [("all", None)]
    regions.extend((f"v_gt_{int(threshold)}", threshold) for threshold in EDGE_THRESHOLDS)
    output: list[dict[str, Any]] = []
    for region, threshold in regions:
        region_rows = [
            row
            for row in rows
            if threshold is None or float(row["height_roi_center_v"]) > float(threshold)
        ]
        for scope, selector in scopes:
            scoped_rows = region_rows
            if selector is not None:
                scoped_rows = [
                    row
                    for row in region_rows
                    if all(row.get(key) == value for key, value in selector.items())
                ]
            for model in MODEL_NAMES:
                metrics = basic_metrics(scoped_rows, model)
                errors = model_error_values(scoped_rows, model)
                rho, pvalue = spearman_residual_v(scoped_rows, model)
                valid_scoped = [row for row in scoped_rows if finite(row.get(f"residual_{model}")) is not None]
                conditions = sorted({str(row["condition_id"]) for row in valid_scoped})
                heights = sorted({str(row["height_label"]) for row in valid_scoped})
                if valid_scoped:
                    worst = max(
                        valid_scoped,
                        key=lambda row: abs(float(row.get(f"residual_{model}") or 0.0)),
                    )
                    worst_condition = worst["condition_id"]
                else:
                    worst_condition = None
                output.append(
                    {
                        "dataset": "session01",
                        "region": region,
                        "threshold_v": threshold,
                        "scope": scope,
                        "model": model,
                        **metrics,
                        "covered_heights": heights,
                        "covered_positions": conditions,
                        "covered_height_count": len(heights),
                        "covered_position_count": len(conditions),
                        "error_gt_0_1_fraction": (
                            float(np.mean(np.abs(errors) > 0.1)) if len(errors) else None
                        ),
                        "error_gt_0_2_fraction": (
                            float(np.mean(np.abs(errors) > 0.2)) if len(errors) else None
                        ),
                        "residual_v_spearman_rho": rho,
                        "residual_v_spearman_pvalue": pvalue,
                        "worst_condition": worst_condition,
                        "v_min": (
                            float(min(row["height_roi_center_v"] for row in scoped_rows))
                            if scoped_rows
                            else None
                        ),
                        "v_max": (
                            float(max(row["height_roi_center_v"] for row in scoped_rows))
                            if scoped_rows
                            else None
                        ),
                    }
                )
    return output


def clipped_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group_name, group_rows in (
        ("clipped", [row for row in rows if row.get("edge_baseline_clipped")]),
        ("normal", [row for row in rows if not row.get("edge_baseline_clipped")]),
    ):
        for scope, selector in [("pooled", None), *[(f"height:{height}", {"height_label": height}) for height in HEIGHT_LABELS]]:
            scoped = group_rows if selector is None else [
                row for row in group_rows if row.get("height_label") == selector["height_label"]
            ]
            for model in MODEL_NAMES:
                output.append(
                    {
                        "dataset": "session01",
                        "group": group_name,
                        "scope": scope,
                        "model": model,
                        **basic_metrics(scoped, model),
                        "condition_count": len({row["condition_id"] for row in scoped}),
                        "edge_position_concentration": sorted(
                            {row["position_id"] for row in scoped}
                        ),
                    }
                )
    return output


def paired_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        e1, e2 = finite(row.get("residual_h1")), finite(row.get("residual_hb2"))
        pair = e1 is not None and e2 is not None
        output.append(
            {
                "row_type": "frame",
                "dataset": "session01",
                "height_label": row.get("height_label"),
                "position_id": row.get("position_id"),
                "condition_id": row.get("condition_id"),
                "v_order_rank": row.get("v_order_rank"),
                "repeat_index": row.get("repeat_index"),
                "height_roi_center_v": row.get("height_roi_center_v"),
                "edge_baseline_clipped": row.get("edge_baseline_clipped"),
                "h1_valid": e1 is not None,
                "hb2_valid": e2 is not None,
                "paired_valid": pair,
                "residual_h1": e1,
                "residual_hb2": e2,
                "abs_error_h1": abs(e1) if e1 is not None else None,
                "abs_error_hb2": abs(e2) if e2 is not None else None,
                "abs_error_delta_hb2_minus_h1": abs(e2) - abs(e1) if pair else None,
                "squared_error_delta_hb2_minus_h1": e2 * e2 - e1 * e1 if pair else None,
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    for cid in sorted(grouped):
        group = grouped[cid]
        pairs = [
            (float(row["residual_h1"]), float(row["residual_hb2"]))
            for row in group
            if finite(row.get("residual_h1")) is not None and finite(row.get("residual_hb2")) is not None
        ]
        e1 = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        e2 = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        stats1 = basic_metrics(
            [{"residual_h1": value} for value in e1], "h1"
        )
        stats2 = basic_metrics(
            [{"residual_hb2": value} for value in e2], "hb2"
        )
        output.append(
            {
                "row_type": "condition",
                "dataset": "session01",
                "height_label": group[0]["height_label"],
                "position_id": group[0]["position_id"],
                "condition_id": cid,
                "v_order_rank": group[0].get("v_order_rank"),
                "height_roi_center_v": group[0].get("height_roi_center_v"),
                "edge_baseline_clipped": group[0].get("edge_baseline_clipped"),
                "n_pair": len(pairs),
                "h1_bias_mm": stats1.get("bias_mm"),
                "hb2_bias_mm": stats2.get("bias_mm"),
                "bias_diff_hb2_minus_h1": (
                    stats2["bias_mm"] - stats1["bias_mm"]
                    if stats1.get("bias_mm") is not None and stats2.get("bias_mm") is not None
                    else None
                ),
                "h1_p95_abs_mm": stats1.get("p95_abs_mm"),
                "hb2_p95_abs_mm": stats2.get("p95_abs_mm"),
                "p95_diff_hb2_minus_h1": (
                    stats2["p95_abs_mm"] - stats1["p95_abs_mm"]
                    if stats1.get("p95_abs_mm") is not None and stats2.get("p95_abs_mm") is not None
                    else None
                ),
                "h1_max_abs_mm": stats1.get("max_abs_mm"),
                "hb2_max_abs_mm": stats2.get("max_abs_mm"),
                "max_diff_hb2_minus_h1": (
                    stats2["max_abs_mm"] - stats1["max_abs_mm"]
                    if stats1.get("max_abs_mm") is not None and stats2.get("max_abs_mm") is not None
                    else None
                ),
                "mean_abs_error_delta_hb2_minus_h1": (
                    float(np.mean(np.abs(e2) - np.abs(e1))) if len(pairs) else None
                ),
                "mean_squared_error_delta_hb2_minus_h1": (
                    float(np.mean(e2 * e2 - e1 * e1)) if len(pairs) else None
                ),
                "hb2_abs_error_better_fraction": (
                    float(np.mean(np.abs(e2) < np.abs(e1))) if len(pairs) else None
                ),
            }
        )
    return output


def flag_from_support(condition_rows: list[dict[str, Any]], model: str) -> str:
    rows = [row for row in condition_rows if row["model"] == model]
    complete = sum(int(row["n_valid"] == REPEAT_COUNT) for row in rows)
    valid = sum(int(row["n_valid"] > 0) for row in rows)
    if complete == CONDITION_COUNT:
        return "SUPPORTED"
    if valid:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def find_edge_failure(edge_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    region = f"v_gt_{int(threshold)}"
    findings: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for height in HEIGHT_LABELS:
            edge = next(
                (
                    row
                    for row in edge_rows
                    if row["region"] == region
                    and row["scope"] == f"height:{height}"
                    and row["model"] == model
                ),
                None,
            )
            interior_rows = [
                row
                for row in edge_rows
                if row["region"] == "all"
                and row["scope"] == f"height:{height}"
                and row["model"] == model
            ]
            interior = interior_rows[0] if interior_rows else None
            if edge is None or interior is None or edge.get("n_valid", 0) == 0:
                continue
            degraded = bool(
                (edge.get("p95_abs_mm") is not None and interior.get("p95_abs_mm") is not None and edge["p95_abs_mm"] > interior["p95_abs_mm"] + 0.02)
                or (edge.get("max_abs_mm") is not None and interior.get("max_abs_mm") is not None and edge["max_abs_mm"] > interior["max_abs_mm"] + 0.05)
                or (edge.get("error_gt_0_2_fraction") is not None and interior.get("error_gt_0_2_fraction") is not None and edge["error_gt_0_2_fraction"] > interior["error_gt_0_2_fraction"] + 0.05)
            )
            findings.append({"model": model, "height": height, "degraded": degraded})
    degraded_count = sum(int(item["degraded"]) for item in findings)
    if degraded_count >= 2:
        flag = "YES"
    elif degraded_count == 1:
        flag = "PARTIAL"
    else:
        flag = "NO"
    return {"flag": flag, "degraded_count": degraded_count, "findings": findings}


def derive_flags(
    rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    support = {model: flag_from_support(condition_rows, model) for model in MODEL_NAMES}
    edge_2400 = find_edge_failure(edge_rows, 2400.0)
    edge_2600 = find_edge_failure(edge_rows, 2600.0)
    spread_results: list[dict[str, Any]] = []
    for height in HEIGHT_LABELS:
        h1_by_position = {
            row["position_id"]: row
            for row in condition_rows
            if row["height_label"] == height
            and row["model"] == "h1"
            and int(row.get("n_valid") or 0) == REPEAT_COUNT
        }
        hb2_by_position = {
            row["position_id"]: row
            for row in condition_rows
            if row["height_label"] == height
            and row["model"] == "hb2"
            and int(row.get("n_valid") or 0) == REPEAT_COUNT
        }
        common_positions = sorted(set(h1_by_position) & set(hb2_by_position))
        h1_biases = np.asarray(
            [float(h1_by_position[position]["bias_mm"]) for position in common_positions],
            dtype=np.float64,
        )
        hb2_biases = np.asarray(
            [float(hb2_by_position[position]["bias_mm"]) for position in common_positions],
            dtype=np.float64,
        )
        comparable = len(common_positions) >= 2
        h1_range = float(np.ptp(h1_biases)) if comparable else None
        hb2_range = float(np.ptp(hb2_biases)) if comparable else None
        h1_std = float(np.std(h1_biases)) if comparable else None
        hb2_std = float(np.std(hb2_biases)) if comparable else None
        spread_results.append(
            {
                "height": height,
                "comparable": comparable,
                "common_position_count": len(common_positions),
                "hb2_full_height_coverage": len(hb2_by_position) == len(POSITION_IDS),
                "common_positions": common_positions,
                "range_hb2_lower": comparable and hb2_range < h1_range,
                "std_hb2_lower": comparable and hb2_std < h1_std,
                "h1_range": h1_range,
                "hb2_range": hb2_range,
                "h1_std": h1_std,
                "hb2_std": hb2_std,
            }
        )
    spread_count = sum(
        int(item["range_hb2_lower"] and item["std_hb2_lower"])
        for item in spread_results
        if item["hb2_full_height_coverage"]
    )
    common_spread_count = sum(
        int(item["range_hb2_lower"] and item["std_hb2_lower"])
        for item in spread_results
    )
    if spread_count >= 2:
        spread_flag = "YES"
    elif common_spread_count >= 1:
        spread_flag = "PARTIAL"
    else:
        spread_flag = "NO"

    edge_tail_results: list[dict[str, Any]] = []
    for region in ("v_gt_2400", "v_gt_2600"):
        for height in HEIGHT_LABELS:
            h1 = next((row for row in edge_rows if row["region"] == region and row["scope"] == f"height:{height}" and row["model"] == "h1"), None)
            hb2 = next((row for row in edge_rows if row["region"] == region and row["scope"] == f"height:{height}" and row["model"] == "hb2"), None)
            comparable = bool(h1 and hb2 and h1.get("p95_abs_mm") is not None and hb2.get("p95_abs_mm") is not None)
            edge_tail_results.append(
                {
                    "region": region,
                    "height": height,
                    "comparable": comparable,
                    "hb2_p95_worse": comparable and hb2["p95_abs_mm"] > h1["p95_abs_mm"],
                    "hb2_max_worse": comparable and hb2.get("max_abs_mm") is not None and h1.get("max_abs_mm") is not None and hb2["max_abs_mm"] > h1["max_abs_mm"],
                    "h1_p95": h1.get("p95_abs_mm") if h1 else None,
                    "hb2_p95": hb2.get("p95_abs_mm") if hb2 else None,
                    "h1_max": h1.get("max_abs_mm") if h1 else None,
                    "hb2_max": hb2.get("max_abs_mm") if hb2 else None,
                }
            )
    comparable_tails = [item for item in edge_tail_results if item["comparable"]]
    tail_worse_count = sum(int(item["hb2_p95_worse"] and item["hb2_max_worse"]) for item in comparable_tails)
    if tail_worse_count >= 2:
        tail_flag = "YES"
    elif tail_worse_count == 1:
        tail_flag = "PARTIAL"
    else:
        tail_flag = "NO"

    clip_rows = [row for row in rows if row.get("edge_baseline_clipped")]
    normal_rows = [row for row in rows if not row.get("edge_baseline_clipped")]
    clip_effects: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        clipped = basic_metrics(clip_rows, model)
        normal = basic_metrics(normal_rows, model)
        clip_effects.append(
            {
                "model": model,
                "clipped_p95": clipped.get("p95_abs_mm"),
                "normal_p95": normal.get("p95_abs_mm"),
                "clipped_abs_bias": abs(clipped["bias_mm"]) if clipped.get("bias_mm") is not None else None,
                "normal_abs_bias": abs(normal["bias_mm"]) if normal.get("bias_mm") is not None else None,
            }
        )
    clip_supported_count = sum(
        int(
            item["clipped_p95"] is not None
            and item["normal_p95"] is not None
            and item["clipped_p95"] > item["normal_p95"] + 0.02
            and item["clipped_abs_bias"] is not None
            and item["normal_abs_bias"] is not None
            and item["clipped_abs_bias"] > item["normal_abs_bias"] + 0.02
        )
        for item in clip_effects
    )
    clip_effect_flag = "SUPPORTED" if clip_supported_count >= 2 else "WEAK" if clip_supported_count == 1 else "NOT_SUPPORTED"

    spatial_evidence: list[dict[str, Any]] = []
    for row in edge_rows:
        if row["region"] == "all" and str(row["scope"]).startswith("height:") and row["model"] in MODEL_NAMES:
            rho = row.get("residual_v_spearman_rho")
            if rho is not None:
                spatial_evidence.append(row)
    strong_spatial_count = sum(
        int(abs(float(row["residual_v_spearman_rho"])) >= 0.5 and (row.get("residual_v_spearman_pvalue") or 1.0) <= 0.05)
        for row in spatial_evidence
    )
    range_evidence = sum(
        int(row.get("position_bias_range_mm") is not None and row["position_bias_range_mm"] >= 0.1)
        for row in spatial_rows
        if row["model"] in MODEL_NAMES
    )
    spatial_flag = "YES" if strong_spatial_count >= 2 or range_evidence >= 2 else "PARTIAL" if strong_spatial_count or range_evidence else "NO"

    if support["hb2"] == "SUPPORTED" and spread_flag == "YES" and tail_flag == "NO":
        preferred = "HB2"
    elif support["h1"] == "SUPPORTED" and (tail_flag in {"YES", "PARTIAL"} or support["hb2"] != "SUPPORTED"):
        preferred = "H1"
    else:
        preferred = "UNDECIDED"
    return {
        "A13B_REPLAY_PROVENANCE_MATCH": "YES" if provenance.get("replay_provenance_match") else "NO",
        "A13B_MEASUREMENT_COMPLETE": "YES" if len(rows) == FRAME_COUNT and all(row.get("residual_base") is not None for row in rows) else "NO",
        "BASE_FULL_FOV_VALID": support["base"],
        "H1_FULL_FOV_VALID": support["h1"],
        "HB2_FULL_FOV_VALID": support["hb2"],
        "EDGE_V2400_FAILURE_REPRODUCED": edge_2400["flag"],
        "EDGE_V2600_FAILURE_REPRODUCED": edge_2600["flag"],
        "EDGE_BASELINE_CLIPPING_EFFECT": clip_effect_flag,
        "HB2_POSITION_SPREAD_ADVANTAGE_REPRODUCED": spread_flag,
        "HB2_EDGE_TAIL_PENALTY_REPRODUCED": tail_flag,
        "PREFERRED_DEPTH_BASELINE_AFTER_SESSION01": preferred,
        "SPATIAL_RESIDUAL_REPRODUCED_IN_NEW_SESSION": spatial_flag,
        "SPATIAL_SOURCE_ATTRIBUTION_ALLOWED": "YES" if spatial_flag in {"YES", "PARTIAL"} and provenance.get("replay_provenance_match") else "NO",
        "NEW_SPATIAL_CORRECTION_ALLOWED": "NO",
        "SECOND_SESSION_REQUIRED_BEFORE_MODEL_CHANGE": "YES",
        "support_by_model": support,
        "edge_2400_detail": edge_2400,
        "edge_2600_detail": edge_2600,
        "spread_detail": spread_results,
        "edge_tail_detail": edge_tail_results,
        "clip_effect_detail": clip_effects,
        "spatial_evidence": spatial_evidence,
    }


def plot_position_bias(condition_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    colors = {"base": "#4c78a8", "h1": "#f58518", "hb2": "#54a24b"}
    for ax, height in zip(axes, HEIGHT_LABELS):
        for model in MODEL_NAMES:
            items = sorted(
                [row for row in condition_rows if row["height_label"] == height and row["model"] == model],
                key=lambda row: int(row["v_order_rank"] or 0),
            )
            ax.plot(
                [row["v_order_rank"] for row in items],
                [row["bias_mm"] for row in items],
                marker="o",
                label=model.upper(),
                color=colors[model],
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(height)
        ax.set_xlabel("v_order_rank (height ROI v)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Bias (mm; nominal truth)")
    axes[-1].legend(loc="best")
    fig.suptitle("Session01 A-13B position bias by nominal height")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residual_v(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    colors = {"base": "#4c78a8", "h1": "#f58518", "hb2": "#54a24b"}
    for ax, model in zip(axes, MODEL_NAMES):
        x = np.asarray([row["height_roi_center_v"] for row in rows if finite(row.get(f"residual_{model}")) is not None], dtype=float)
        y = np.asarray([row[f"residual_{model}"] for row in rows if finite(row.get(f"residual_{model}")) is not None], dtype=float)
        ax.scatter(x, y, s=9, alpha=0.35, color=colors[model], edgecolors="none")
        for threshold in EDGE_THRESHOLDS:
            ax.axvline(threshold, color="#999999", linestyle="--", linewidth=0.7)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(model.upper())
        ax.set_xlabel("height ROI center v")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Residual (mm; nominal truth)")
    fig.suptitle("Session01 A-13B residual versus true height-ROI v")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_paired(paired_rows: list[dict[str, Any]], path: Path) -> None:
    condition = [row for row in paired_rows if row.get("row_type") == "condition" and row.get("mean_abs_error_delta_hb2_minus_h1") is not None]
    condition.sort(key=lambda row: (row.get("height_label"), int(row.get("v_order_rank") or 0)))
    labels = [f"{row['height_label']}-{row['position_id']}" for row in condition]
    values = [float(row["mean_abs_error_delta_hb2_minus_h1"]) for row in condition]
    colors = ["#d62728" if value > 0 else "#2ca02c" for value in values]
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_ylabel("mean |e| delta: H-B2 − H1 (mm)")
    ax.set_title("Same-frame paired H1 vs H-B2; positive means H-B2 is worse")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_edge(rows: list[dict[str, Any]], threshold: float, path: Path) -> None:
    region = f"v_gt_{int(threshold)}"
    data: list[list[float]] = []
    labels: list[str] = []
    for model in MODEL_NAMES:
        values = [
            abs(float(row[f"residual_{model}"]))
            for row in rows
            if float(row["height_roi_center_v"]) > threshold and finite(row.get(f"residual_{model}")) is not None
        ]
        data.append(values or [0.0])
        labels.append(model.upper())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    except TypeError:  # Matplotlib < 3.9
        ax.boxplot(data, labels=labels, showfliers=True)
    ax.axhline(0.1, color="#f58518", linestyle="--", linewidth=0.9, label="0.1 mm")
    ax.axhline(0.2, color="#d62728", linestyle="--", linewidth=0.9, label="0.2 mm")
    ax.set_ylabel("absolute error (mm)")
    ax.set_title(f"Edge audit: {region} (height ROI v)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_clipped(rows: list[dict[str, Any]], path: Path) -> None:
    groups = ["normal", "clipped"]
    x = np.arange(len(MODEL_NAMES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, group in enumerate(groups):
        values = []
        for model in MODEL_NAMES:
            values.append(
                basic_metrics(
                    [row for row in rows if bool(row.get("edge_baseline_clipped")) == (group == "clipped")],
                    model,
                ).get("p95_abs_mm")
                or 0.0
            )
        ax.bar(x + (index - 0.5) * width, values, width, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels([model.upper() for model in MODEL_NAMES])
    ax.set_ylabel("P95 |error| (mm)")
    ax.set_title("Baseline-clipped versus normal frozen ROIs")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def report_text(
    provenance: dict[str, Any],
    ground_payload: dict[str, Any],
    registry: dict[str, Any],
    cache_manifest: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    clipped_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    flags: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# Session01 A-13B Frozen ROI 正式全 FOV 验证",
        "",
        "本报告是 validation-only replay。没有重新运行 Steger、没有读取 `height_shadow.csv` 做正式高度/FOV 计算、没有重新拟合 C0/C1/Ground/H1/H-B2，也没有新增 correction。",
        "",
        "## Final flags",
        "",
        "```text",
    ]
    for key in (
        "A13B_REPLAY_PROVENANCE_MATCH",
        "A13B_MEASUREMENT_COMPLETE",
        "BASE_FULL_FOV_VALID",
        "H1_FULL_FOV_VALID",
        "HB2_FULL_FOV_VALID",
        "EDGE_V2400_FAILURE_REPRODUCED",
        "EDGE_V2600_FAILURE_REPRODUCED",
        "EDGE_BASELINE_CLIPPING_EFFECT",
        "HB2_POSITION_SPREAD_ADVANTAGE_REPRODUCED",
        "HB2_EDGE_TAIL_PENALTY_REPRODUCED",
        "PREFERRED_DEPTH_BASELINE_AFTER_SESSION01",
        "SPATIAL_RESIDUAL_REPRODUCED_IN_NEW_SESSION",
        "SPATIAL_SOURCE_ATTRIBUTION_ALLOWED",
        "NEW_SPATIAL_CORRECTION_ALLOWED",
        "SECOND_SESSION_REQUIRED_BEFORE_MODEL_CHANGE",
    ):
        lines.append(f"{key}={flags[key]}")
    lines.extend(["```", "", "## Provenance / reuse lock", ""])
    lines.extend(
        [
            f"- Dataset root: `{DATA_ROOT}`; PNG source identity checked: `{provenance['raw_hash_matches']}/{FRAME_COUNT}` SHA256 matches.",
            f"- Frozen cache: `{CACHE_NPZ}`; frames `{provenance['cache_info']['frames_total']}`; concatenated centers `{provenance['cache_info']['centers_total']}`; `one_steger_per_frame={cache_manifest.get('one_steger_per_frame')}`.",
            f"- Actual cache manifest field `reused_existing_cache={cache_manifest.get('reused_existing_cache')}`. A-13A report/cache-reuse display discrepancy is retained as a note; A-13B used the immutable manifest and NPZ, and did not rewrite either.",
            f"- Frozen manifest SHA256: `{provenance['manifest_sha256']}`; cache protocol SHA256: `{cache_manifest.get('protocol_key', {}).get('frozen_manifest_sha256')}`; match `{bool_text(provenance['manifest_sha256'] == cache_manifest.get('protocol_key', {}).get('frozen_manifest_sha256'))}`.",
            f"- Current formal config SHA256: `{provenance['config_sha256']}`; semantic lock values: depth `630–715 mm`, model margin `2 mm`, C1 enabled, measurement minimum baseline/height `20/20`.",
            f"- Frozen C0/C1/H1/H-B2 artifact hashes are recorded in `session01_a13b_provenance_audit.json`: `{json.dumps(provenance.get('frozen_artifact_hashes', {}), ensure_ascii=False)}`; none were modified or refit.",
            f"- Frozen registry: `{provenance['registry_entry_count']}` entries; manual confirmed/frozen/geometry-only lock `{bool_text(provenance['registry_ok'])}`; edge baseline-clipped entries `{registry.get('edge_baseline_clipped_entry_count')}`.",
            f"- No whole-frame v median, truth height, residual, q1/q2 or correction output was used to select/alter any ROI. Formal coordinate is `height_roi_center_v`.",
        ]
    )
    lines.extend(["", "## Session PnP / Ground provenance", ""])
    pnp = provenance["pnp"]
    session = provenance["session_extrinsic"]
    ground = provenance["session_ground"]
    lines.extend(
        [
            f"- `session_ground_calibration.json`: top `status={ground_payload.get('status')}`, `valid={ground_payload.get('valid')}`; PnP `pnp_valid={bool_text(pnp.get('status'))}`, corners `{pnp.get('corner_count')}`, reprojection RMSE `{fmt(pnp.get('reprojection_rmse_px'), 6)} px`.",
            f"- PnP reference R: `{json.dumps(pnp.get('reference_R'), ensure_ascii=False)}`; t (mm): `{json.dumps(pnp.get('reference_t'), ensure_ascii=False)}`.",
            f"- Session R: `{json.dumps(session.get('R'), ensure_ascii=False)}`; t (mm): `{json.dumps(session.get('t'), ensure_ascii=False)}`; PnP/session delta: `{json.dumps(session.get('delta'), ensure_ascii=False)}`.",
            f"- Session Ground Reference: status `{ground.get('status')}`, source `{ground.get('source')}`, support `{ground.get('support_source')}`, slope/intercept `{fmt(ground.get('slope'), 9)}` / `{fmt(ground.get('intercept'), 9)}`, RMSE `{fmt(ground.get('rmse_mm'), 6)} mm`, valid S `{ground.get('valid_s_range_mm')}`, support points `{ground.get('point_count')}/{ground.get('inlier_count')}`.",
            f"- Formal replay chain: cached Frozen Steger full-sensor `(u,v)` → Frozen C0 → Frozen C1 → Session R/t → saved Session Ground Reference → GUI `measure_height_line` with `ground_correction_mode=session_reference`.",
            "- `session_ground_calibration.json` 中 Ground VALID 与 `height_shadow.csv` 的 `ground_reference_status=inactive` 不矛盾：后者是独立 shadow-logging 路径的当时应用状态；本轮没有使用该文件，也没有因 inactive 重拟或修改 Ground。",
        ]
    )
    lines.extend(["", "## ROI / spatial coverage reuse", ""])
    coverage = registry.get("coverage_summary", {}).get("overall", {})
    per_height_coverage = registry.get("coverage_summary", {}).get("per_height", {})
    max_adjacent_gap = max(
        (float(item.get("max_adjacent_gap_v")) for item in per_height_coverage.values()),
        default=None,
    )
    lines.extend(
        [
            f"- 30 个 frozen geometry-only entries，位置坐标只使用 `height_roi_center_v`；全局 v 范围 `{coverage.get('min_height_roi_center_v')}–{coverage.get('max_height_roi_center_v')}`，最大相邻 gap `{max_adjacent_gap}` px。",
            f"- v>2200 `{coverage.get('covers_v_gt_2200')}`；v>2400 `{coverage.get('covers_v_gt_2400')}`；v>2600 `{coverage.get('covers_v_gt_2600')}`。六个 edge baseline clipped entry 保留原 ROI，没有删除或重尺寸。",
            "- h10/h20/h30 仅按 nominal truth 10/20/30 mm 计算 residual；未发现/未使用 certified height，因此结果不是标准件认证声明。",
        ]
    )
    lines.extend(["", "## Frame status audit", ""])
    def count_field(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in frame_rows:
            value = str(item.get(field))
            counts[value] = counts.get(value, 0) + 1
        return counts
    lines.extend(
        [
            f"- Measurement rows: `{len(frame_rows)}`; invalid status: `{json.dumps(count_field('invalid_status'), ensure_ascii=False)}`.",
            f"- q2 hard gate: `{json.dumps(count_field('q2_in_domain'), ensure_ascii=False)}`; H-B2 status: `{json.dumps(count_field('hb2_q2_status'), ensure_ascii=False)}`. H-B2 OOD rows are rejected, never clamped.",
            f"- C1 clamp: full cached centerline `{json.dumps(count_field('c1_clamp_status_all'), ensure_ascii=False)}`; height ROI formal points `{json.dumps(count_field('c1_clamp_status_height'), ensure_ascii=False)}`.",
            f"- Session Ground: full centerline `{json.dumps(count_field('ground_status_all'), ensure_ascii=False)}`; formal baseline+height `{json.dumps(count_field('ground_status_formal'), ensure_ascii=False)}`; height ROI `{json.dumps(count_field('ground_status_height'), ensure_ascii=False)}`. Ground-OOD points remain raw per the frozen `apply_to_points` contract and are explicitly counted; no extrapolation is performed.",
        ]
    )
    lines.extend(["", "## Condition-level metrics", "", "|height|model|conditions|valid frames|Bias (mm)|MAE|RMSE|P95|Max|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for height in HEIGHT_LABELS:
        for model in MODEL_NAMES:
            items = [row for row in frame_rows if row["height_label"] == height]
            metrics = basic_metrics(items, model)
            lines.append(
                f"|{height}|{model.upper()}|{CONDITION_COUNT // len(HEIGHT_LABELS)}|{metrics['n_valid']}|{fmt(metrics['bias_mm'])}|{fmt(metrics['mae_mm'])}|{fmt(metrics['rmse_mm'])}|{fmt(metrics['p95_abs_mm'])}|{fmt(metrics['max_abs_mm'])}|"
            )
    lines.extend(["", "## Height spatial metrics", "", "|height|model|valid positions|Bias range|Bias std|worst |Bias||worst P95|worst Max|", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in spatial_rows:
        lines.append(
            f"|{row['height_label']}|{row['model'].upper()}|{row['valid_position_count']}|{fmt(row['position_bias_range_mm'])}|{fmt(row['position_bias_std_mm'])}|{fmt(row['worst_position_abs_bias_mm'])} ({row.get('worst_position_abs_bias_position_id') or '—'})|{fmt(row['worst_position_p95_abs_mm'])} ({row.get('worst_position_p95_position_id') or '—'})|{fmt(row['worst_position_max_abs_mm'])} ({row.get('worst_position_max_position_id') or '—'})|"
        )
    lines.extend(["", "## Edge audit", "", "Edge regions use `height_roi_center_v`, not whole-frame v median. `edge_metrics.csv` includes pooled, per-height, and per-position scopes.", "", "|region|scope|model|n valid|P95|Max|>|0.1|>|0.2|rho(v,residual)|worst condition|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in edge_rows:
        if row["scope"] in {"pooled", *[f"height:{height}" for height in HEIGHT_LABELS]} and row["region"] in {"v_gt_2400", "v_gt_2600"}:
            lines.append(
                f"|{row['region']}|{row['scope']}|{row['model'].upper()}|{row['n_valid']}|{fmt(row['p95_abs_mm'])}|{fmt(row['max_abs_mm'])}|{fmt(row['error_gt_0_1_fraction'], 3)}|{fmt(row['error_gt_0_2_fraction'], 3)}|{fmt(row['residual_v_spearman_rho'], 3)}|{row.get('worst_condition') or '—'}|"
            )
    lines.extend(["", "## Paired H1 vs H-B2", "", "- Pairing is same processed frame. `abs_error_delta_hb2_minus_h1 < 0` means H-B2 is better; the condition rows also contain Bias/P95/Max differences and squared-error differences.", ""])
    condition_pairs = [row for row in paired_rows if row.get("row_type") == "condition"]
    paired_values = [row["mean_abs_error_delta_hb2_minus_h1"] for row in condition_pairs if row.get("mean_abs_error_delta_hb2_minus_h1") is not None]
    squared_values = [row["mean_squared_error_delta_hb2_minus_h1"] for row in condition_pairs if row.get("mean_squared_error_delta_hb2_minus_h1") is not None]
    lines.extend(
        [
            f"- Paired condition rows: `{len(condition_pairs)}`; condition mean |error| delta (pooled mean): `{fmt(np.mean(paired_values) if paired_values else None)} mm`; squared-error delta: `{fmt(np.mean(squared_values) if squared_values else None)} mm²`.",
            f"- H-B2 position spread: `{flags['HB2_POSITION_SPREAD_ADVANTAGE_REPRODUCED']}`; detail `{json.dumps(flags['spread_detail'], ensure_ascii=False)}`.",
            f"- H-B2 edge tail: `{flags['HB2_EDGE_TAIL_PENALTY_REPRODUCED']}`; detail `{json.dumps(flags['edge_tail_detail'], ensure_ascii=False)}`.",
            "- Historical A-11 relation is therefore judged from the frozen same-frame results: H-B2 spread advantage and H1 edge-tail advantage are not assumed; the flags above state whether each relation reappears.",
        ]
    )
    lines.extend(["", "## Baseline clipping audit", "", "Clipping is an A-13A frozen geometry property, not a reason to delete or resize an ROI.", "", "|group|scope|model|n|Bias|P95|Max|", "|---|---|---:|---:|---:|---:|---:|"])
    for row in clipped_rows:
        if row["scope"] == "pooled":
            lines.append(
                f"|{row['group']}|{row['scope']}|{row['model'].upper()}|{row['n_valid']}|{fmt(row['bias_mm'])}|{fmt(row['p95_abs_mm'])}|{fmt(row['max_abs_mm'])}|"
            )
    lines.extend(["", "## Artifacts", ""])
    for filename in (
        "session01_a13b_frame_measurements.csv",
        "session01_a13b_condition_metrics.csv",
        "session01_a13b_height_spatial_metrics.csv",
        "session01_a13b_edge_metrics.csv",
        "session01_a13b_clipped_vs_normal_metrics.csv",
        "session01_a13b_paired_h1_hb2.csv",
        "session01_a13b_position_bias_by_height.png",
        "session01_a13b_residual_vs_v.png",
        "session01_a13b_h1_vs_hb2_paired_delta.png",
        "session01_a13b_edge_v2400.png",
        "session01_a13b_edge_v2600.png",
        "session01_a13b_clipped_baseline_audit.png",
    ):
        lines.append(f"- `{OUTPUT_DIR / filename}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "`SPATIAL_SOURCE_ATTRIBUTION_ALLOWED` 仅表示可以把新 session 中的 residual-v / position spread 作为 frozen stack 的空间残差现象进行诊断归因；不表示已经允许拟合或部署 spatial correction。`NEW_SPATIAL_CORRECTION_ALLOWED=NO`，且本轮没有删 position、改 ROI、改 Ground 或改 H1/H-B2。",
            "",
            "`SECOND_SESSION_REQUIRED_BEFORE_MODEL_CHANGE=YES`：Session01 可完成跨 session validation，但在模型变更或新 spatial correction 前仍需另一独立 session。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not all(path.is_file() for path in (CACHE_NPZ, CACHE_MANIFEST, REGISTRY_PATH, GROUND_PATH, CONFIG_PATH, MANIFEST_PATH)):
        missing = [str(path) for path in (CACHE_NPZ, CACHE_MANIFEST, REGISTRY_PATH, GROUND_PATH, CONFIG_PATH, MANIFEST_PATH) if not path.is_file()]
        raise RuntimeError(f"A-13B required frozen artifact missing: {missing}")
    app = load_app_config(CONFIG_PATH)
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_model,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        app.calibration.laser_ray_correction,
        ground_u_optional=True,
    )
    ground_payload = json.loads(GROUND_PATH.read_text(encoding="utf-8"))
    ground_reference = load_session_reference(ground_payload)
    calibration["R"] = np.asarray(ground_payload["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64)
    calibration["t"] = np.asarray(ground_payload["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64)
    frames, cache_manifest, registry, centers_by_key, cache_info = load_cache_and_registry()
    provenance = provenance_audit(frames, cache_manifest, registry, ground_payload, app, cache_info)
    write_json(output_dir / "session01_a13b_provenance_audit.json", provenance)

    registry_by_condition = {str(entry["condition_id"]): entry for entry in registry.get("entries", [])}
    source_cache: dict[str, tuple[dict[str, dict[str, Any]], dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames, start=1):
        cid = f"{frame['height_label']}_{frame['position_id']}"
        if cid not in registry_by_condition:
            raise RuntimeError(f"Frozen ROI registry missing {cid}")
        if cid not in source_cache:
            source_cache[cid] = read_frames_csv(DATA_ROOT / str(frame["height_label"]) / cid)
        by_filename, audit = source_cache[cid]
        source_qc = build_source_qc(frame, audit, by_filename)
        key = str(frame["cache_key"])
        record = reconstruct_one_frame(
            frame,
            centers_by_key[key],
            registry_by_condition[cid],
            calibration,
            app,
            ground_reference,
            source_qc,
        )
        records.append(record)
        if index % 50 == 0:
            print(f"A-13B replay {index}/{len(frames)} frames")

    condition_rows = condition_metrics(records)
    spatial_rows = height_spatial_metrics(condition_rows)
    edge_rows = edge_metrics(records)
    clipped_rows = clipped_metrics(records)
    paired_rows = paired_metrics(records)
    flags = derive_flags(records, condition_rows, spatial_rows, edge_rows, paired_rows, provenance)
    write_csv(output_dir / "session01_a13b_frame_measurements.csv", records)
    write_csv(output_dir / "session01_a13b_condition_metrics.csv", condition_rows)
    write_csv(output_dir / "session01_a13b_height_spatial_metrics.csv", spatial_rows)
    write_csv(output_dir / "session01_a13b_edge_metrics.csv", edge_rows)
    write_csv(output_dir / "session01_a13b_clipped_vs_normal_metrics.csv", clipped_rows)
    write_csv(output_dir / "session01_a13b_paired_h1_hb2.csv", paired_rows)
    plot_position_bias(condition_rows, output_dir / "session01_a13b_position_bias_by_height.png")
    plot_residual_v(records, output_dir / "session01_a13b_residual_vs_v.png")
    plot_paired(paired_rows, output_dir / "session01_a13b_h1_vs_hb2_paired_delta.png")
    plot_edge(records, 2400.0, output_dir / "session01_a13b_edge_v2400.png")
    plot_edge(records, 2600.0, output_dir / "session01_a13b_edge_v2600.png")
    plot_clipped(records, output_dir / "session01_a13b_clipped_baseline_audit.png")
    (output_dir / "session01_a13b_validation_report.md").write_text(
        report_text(
            provenance,
            ground_payload,
            registry,
            cache_manifest,
            records,
            condition_rows,
            spatial_rows,
            edge_rows,
            clipped_rows,
            paired_rows,
            flags,
        ),
        encoding="utf-8",
    )
    write_json(output_dir / "session01_a13b_flags.json", flags)
    print(json.dumps({"output_dir": str(output_dir), "flags": flags}, ensure_ascii=False, indent=2))
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
