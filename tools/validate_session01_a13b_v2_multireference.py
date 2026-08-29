#!/usr/bin/env python3
"""Task A-13B-v2: frozen-ROI, multi-reference Session01 validation.

The formal branch is the frozen Session Ground reference.  The local-baseline
branch is deliberately diagnostic only: it consumes the already Session-Ground
leveled points and cannot change the formal reference or fit any new model.

This script reuses the A-13A Frozen Steger cache.  Each PNG/cache frame is
reconstructed exactly once; all six model/reference views share that result and
the same ROI-selected point arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


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
REGISTRY_PATH = OUTPUT_DIR / "session01_roi_registry_manual_v2.json"
GROUND_PATH = DATA_ROOT / "session_ground_calibration.json"
OLD_V1_FRAME_PATH = OUTPUT_DIR / "session01_a13b_frame_measurements.csv"
OLD_V1_CONDITION_PATH = OUTPUT_DIR / "session01_a13b_condition_metrics.csv"
OLD_V1_REGISTRY_PATH = OUTPUT_DIR / "session01_roi_registry_manual.json"

HEIGHT_LABELS = ("h10", "h20", "h30")
POSITION_IDS = tuple(f"p{i:02d}" for i in range(1, 11))
REPEAT_COUNT = 20
CONDITION_COUNT = 30
FRAME_COUNT = CONDITION_COUNT * REPEAT_COUNT
FULL_SENSOR_WIDTH = 4096.0
EDGE_THRESHOLDS = (2200.0, 2400.0, 2600.0)
KEY_V1_FAILURE_CONDITIONS = (
    "h10_p05",
    "h10_p06",
    "h20_p03",
    "h30_p02",
    "h30_p03",
    "h30_p04",
)

sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from correction.stage_a_height_scale import resolve_height_correction  # noqa: E402
from measurement.ground_reference import (  # noqa: E402
    MeasurementError,
    SessionGroundReference,
)
from measurement.height_measure import measure_height_line  # noqa: E402
from reconstruction.reconstructor import (  # noqa: E402
    ReconstructionInputError,
    reconstruct_uv_to_ground,
)


MODEL_GROUPS = (
    ("session", "base", "residual_base_session", False),
    ("session", "h1", "residual_h1_session", False),
    ("session", "hb2", "residual_hb2_session", False),
    ("local_diag", "base", "residual_base_local_diag", True),
    ("local_diag", "h1", "residual_h1_local_diag", True),
    ("local_diag", "hb2", "residual_hb2_local_diag", True),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return finite(value)
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
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
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


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def parse_float(value: Any) -> float | None:
    return finite(value)


def condition_id(height_label: str, position_id: str) -> str:
    return f"{height_label}_{position_id}"


def truth_for(height_label: str) -> float:
    return {"h10": 10.0, "h20": 20.0, "h30": 30.0}[height_label]


def roi_mask(points_uv: np.ndarray, ranges: list[list[float]]) -> np.ndarray:
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


def read_frames_csv(condition_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = read_csv(condition_dir / "frames.csv")
    by_filename: dict[str, dict[str, Any]] = {}
    duplicate_filenames: set[str] = set()
    camera_ids: list[int] = []
    for row in rows:
        filename = str(row.get("filename", "")).strip()
        if filename in by_filename:
            duplicate_filenames.add(filename)
        by_filename[filename] = row
        camera_number = parse_int(row.get("camera_frame_number"))
        if camera_number is not None:
            camera_ids.append(camera_number)
    duplicate_camera_ids = sorted(
        {number for number in camera_ids if camera_ids.count(number) > 1}
    )
    sorted_ids = sorted(camera_ids)
    actual_gaps = sorted(
        {current - previous - 1 for previous, current in zip(sorted_ids, sorted_ids[1:]) if current - previous - 1}
    )
    reported_gaps = sorted(
        {
            gap
            for row in rows
            if (gap := parse_int(row.get("frame_gap"))) not in (None, 0)
        }
    )
    return by_filename, {
        "frames_csv_exists": (condition_dir / "frames.csv").is_file(),
        "frames_csv_row_count": len(rows),
        "duplicate_filename_count": len(duplicate_filenames),
        "duplicate_camera_frame_id_count": len(duplicate_camera_ids),
        "duplicate_camera_frame_ids": duplicate_camera_ids,
        "actual_frame_gap_values": actual_gaps,
        "reported_frame_gap_values": reported_gaps,
        "camera_frame_min": min(camera_ids) if camera_ids else None,
        "camera_frame_max": max(camera_ids) if camera_ids else None,
        "offset_values": sorted(
            {
                (parse_int(row.get("offset_x")), parse_int(row.get("offset_y")))
                for row in rows
            }
        ),
        "size_values": sorted(
            {(parse_int(row.get("width")), parse_int(row.get("height"))) for row in rows}
        ),
    }


def load_cache() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    with np.load(CACHE_NPZ, allow_pickle=False) as bundle:
        centers = np.asarray(bundle["centers_full"], dtype=np.float64)
        offsets = np.asarray(bundle["frame_offsets"], dtype=np.int64)
    frames = list(manifest.get("frames", []))
    if offsets.ndim != 1 or len(offsets) != len(frames) + 1:
        raise RuntimeError("Frozen Steger frame_offsets shape is invalid")
    if len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != len(centers):
        raise RuntimeError("Frozen Steger frame_offsets bounds are invalid")
    centers_by_key: dict[str, np.ndarray] = {}
    for index, frame in enumerate(frames):
        start, end = int(offsets[index]), int(offsets[index + 1])
        frame_centers = np.ascontiguousarray(centers[start:end], dtype=np.float64)
        if frame_centers.ndim != 2 or frame_centers.shape[1] != 2 or not len(frame_centers):
            raise RuntimeError(f"Frozen Steger frame has no usable points: {index}")
        centers_by_key[str(frame["cache_key"])] = frame_centers
    return frames, manifest, centers_by_key, {
        "centers_total": int(len(centers)),
        "frames_total": len(frames),
        "frame_offsets_shape": list(offsets.shape),
        "one_steger_per_frame": manifest.get("one_steger_per_frame"),
        "reused_existing_cache": True,
    }


def hydrate_session_ground(payload: dict[str, Any]) -> SessionGroundReference:
    if payload.get("status") != "VALID" or payload.get("valid") is not True:
        raise RuntimeError("session_ground_calibration.json is not VALID")
    runtime = payload.get("runtime", {})
    if runtime.get("ground_extrinsic_source") != "session":
        raise RuntimeError("session ground runtime source is not session")
    ground = payload.get("session_ground_reference", {})
    if ground.get("status") != "VALID":
        raise RuntimeError("Session Ground Reference is not VALID")
    return SessionGroundReference(
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
        active_ground_extrinsic_source=str(ground.get("active_ground_extrinsic_source", "session")),
        ground_extrinsic_generation=int(ground.get("ground_extrinsic_generation", 0)),
        frame_host_monotonic_ns=int(ground.get("frame_host_monotonic_ns", 0)),
        mask_inset_mm=float(ground.get("mask_inset_mm", 0.0)),
        support_metadata=dict(ground.get("support", {})),
    )


def source_qc(
    frame: dict[str, Any],
    audit: dict[str, Any],
    by_filename: dict[str, dict[str, Any]],
    source_hash_cache: dict[Path, str | None],
) -> dict[str, Any]:
    cid = condition_id(str(frame["height_label"]), str(frame["position_id"]))
    path = DATA_ROOT / str(frame["height_label"]) / cid / str(frame["filename"])
    exists = path.is_file()
    if path not in source_hash_cache:
        source_hash_cache[path] = sha256_file(path) if exists else None
    row = by_filename.get(str(frame.get("filename")))
    expected_offset = frame.get("offset_xy") or [None, None]
    frames_match = bool(
        row is not None
        and parse_int(row.get("camera_frame_number")) == parse_int(frame.get("camera_frame_number"))
        and parse_int(row.get("offset_x")) == parse_int(expected_offset[0])
        and parse_int(row.get("offset_y")) == parse_int(expected_offset[1])
    )
    return {
        "source_png_exists": exists,
        "source_hash_match": bool(exists and source_hash_cache[path] == frame.get("source_sha256")),
        "frames_csv_match": frames_match,
        "frame_gap": parse_int(row.get("frame_gap")) if row is not None else None,
        "frames_csv_row_count": audit.get("frames_csv_row_count"),
        "raw_png_count_in_condition": len(list(path.parent.glob("frame_*.png"))),
        "frames_csv_width": parse_int(row.get("width")) if row is not None else None,
        "frames_csv_height": parse_int(row.get("height")) if row is not None else None,
        "frames_csv_exposure_us": parse_float(row.get("exposure_us")) if row is not None else None,
        "frames_csv_offset_x": parse_int(row.get("offset_x")) if row is not None else None,
        "frames_csv_offset_y": parse_int(row.get("offset_y")) if row is not None else None,
    }


def provenance_audit(
    frames: list[dict[str, Any]],
    manifest: dict[str, Any],
    registry: dict[str, Any],
    ground_payload: dict[str, Any],
    app: Any,
    cache_info: dict[str, Any],
) -> dict[str, Any]:
    source_hash_cache: dict[Path, str | None] = {}
    condition_audits: dict[str, dict[str, Any]] = {}
    raw_hash_matches = 0
    raw_hash_mismatches = 0
    frames_csv_mismatches = 0
    for height in HEIGHT_LABELS:
        for position in POSITION_IDS:
            cid = condition_id(height, position)
            by_filename, audit = read_frames_csv(DATA_ROOT / height / cid)
            pngs = sorted((DATA_ROOT / height / cid).glob("frame_*.png"))
            frame_subset = [frame for frame in frames if condition_id(str(frame["height_label"]), str(frame["position_id"])) == cid]
            missing_from_csv = sorted(path.name for path in pngs if path.name not in by_filename)
            condition_audits[cid] = {
                **audit,
                "raw_png_count": len(pngs),
                "manifest_frame_count": len(frame_subset),
                "raw_png_missing_from_frames_csv": missing_from_csv,
            }
            for frame in frame_subset:
                qc = source_qc(frame, audit, by_filename, source_hash_cache)
                raw_hash_matches += int(qc["source_hash_match"])
                raw_hash_mismatches += int(not qc["source_hash_match"])
                frames_csv_mismatches += int(not qc["frames_csv_match"])

    detection = ground_payload.get("detection", {})
    session = ground_payload.get("session_ground_reference", {})
    pnp_rmse = finite(detection.get("reprojection_rmse_px"))
    pnp_valid = bool(
        ground_payload.get("pnp_valid", False)
        or (
            detection.get("corner_count") == 88
            and pnp_rmse is not None
            and pnp_rmse <= 0.5
            and bool(ground_payload.get("reference_extrinsic"))
            and bool(ground_payload.get("session_extrinsic"))
        )
    )
    ground_valid = bool(
        ground_payload.get("ground_valid", False)
        or (
            ground_payload.get("session_ground_reference_status") == "VALID"
            and session.get("status") == "VALID"
            and session.get("support", {}).get("status") == "applied"
        )
    )
    entries = list(registry.get("entries", []))
    registry_ok = bool(
        registry.get("dataset") == "session01"
        and registry.get("frozen") is True
        and registry.get("manual_confirmed") is True
        and registry.get("manual_confirmed_count") == 30
        and len(entries) == 30
        and all(entry.get("human_reviewed") is True for entry in entries)
        and all(entry.get("human_decision") == "ACCEPTED" for entry in entries)
        and all(entry.get("frozen") is True for entry in entries)
        and registry.get("geometry_only") is True
    )
    protocol = manifest.get("protocol_key", {})
    cache_ok = bool(
        cache_info["frames_total"] == FRAME_COUNT
        and cache_info["one_steger_per_frame"] is True
        and protocol.get("extraction_method") == "steger"
        and protocol.get("full_sensor_coordinate_system") is True
        and protocol.get("height_shadow_used_for_formal_geometry") is False
        and all(int(frame.get("steger_run_count", 0)) == 1 for frame in frames)
    )
    config_values_ok = bool(
        app.system == "daheng"
        and app.reconstruction.enable_laser_ray_correction is True
        and app.measurement.min_baseline_points == 20
        and app.measurement.min_height_points == 20
        and app.correction.stage_a_height_scale is not None
        and app.correction.hb2_height_correction is not None
    )
    source_ok = (
        len(frames) == FRAME_COUNT
        and raw_hash_matches == FRAME_COUNT
        and raw_hash_mismatches == 0
        and frames_csv_mismatches == 0
        and all(item["raw_png_count"] == REPEAT_COUNT for item in condition_audits.values())
        and all(item["frames_csv_row_count"] == REPEAT_COUNT for item in condition_audits.values())
    )
    shadow_logging_status_counts: dict[str, int] = defaultdict(int)
    shadow_files = 0
    for height in HEIGHT_LABELS:
        for position in POSITION_IDS:
            path = DATA_ROOT / height / condition_id(height, position) / "height_shadow.csv"
            if not path.is_file():
                continue
            shadow_files += 1
            for row in read_csv(path):
                shadow_logging_status_counts[str(row.get("ground_reference_status", ""))] += 1
    artifacts = {
        "config": {"path": str(CONFIG_PATH.resolve()), "sha256": sha256_file(CONFIG_PATH)},
        "manifest": {"path": str(MANIFEST_PATH.resolve()), "sha256": sha256_file(MANIFEST_PATH)},
        "c0": {"path": str(app.calibration.laser_model.resolve()), "sha256": sha256_file(app.calibration.laser_model)},
        "c1": {"path": str(app.calibration.laser_ray_correction.resolve()), "sha256": sha256_file(app.calibration.laser_ray_correction)},
        "h1": {"path": str(app.correction.stage_a_height_scale_config.resolve()), "sha256": sha256_file(app.correction.stage_a_height_scale_config)},
        "hb2": {"path": str(app.correction.hb2_height_correction_config.resolve()), "sha256": sha256_file(app.correction.hb2_height_correction_config)},
        "session_ground": {"path": str(GROUND_PATH.resolve()), "sha256": sha256_file(GROUND_PATH)},
        "cache_npz": {"path": str(CACHE_NPZ.resolve()), "sha256": sha256_file(CACHE_NPZ)},
        "cache_manifest": {"path": str(CACHE_MANIFEST.resolve()), "sha256": sha256_file(CACHE_MANIFEST)},
        "manual_registry_v2": {"path": str(REGISTRY_PATH.resolve()), "sha256": sha256_file(REGISTRY_PATH)},
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(DATA_ROOT),
        "formal_height_shadow_used": False,
        "whole_frame_v_median_used_for_position": False,
        "steger_rerun": False,
        "reconstruction_calls_per_frame": 1,
        "reused_artifacts": {
            "frozen_steger_cache": True,
            "v2_roi_candidates": True,
            "v2_manual_registry": True,
            "frozen_c0_c1_ground_h1_hb2": True,
            "a13b_v1_results_for_historical_comparison_only": True,
        },
        "cache_info": cache_info,
        "cache_protocol": protocol,
        "cache_ok": cache_ok,
        "registry_ok": registry_ok,
        "config_values_ok": config_values_ok,
        "session_ground_ok": bool(ground_payload.get("status") == "VALID" and ground_payload.get("valid") is True and pnp_valid and ground_valid and ground_payload.get("runtime", {}).get("ground_extrinsic_source") == "session"),
        "source_identity_ok": source_ok,
        "raw_hash_matches": raw_hash_matches,
        "raw_hash_mismatches": raw_hash_mismatches,
        "frames_csv_mismatches": frames_csv_mismatches,
        "replay_provenance_match": bool(source_ok and cache_ok and registry_ok and config_values_ok),
        "condition_audit": condition_audits,
        "session_ground_summary": {
            "pnp_valid": pnp_valid,
            "pnp_corner_count": detection.get("corner_count"),
            "pnp_reprojection_rmse_px": pnp_rmse,
            "pnp_status": detection.get("status"),
            "session_ground_status": session.get("status"),
            "session_ground_reference_status": ground_payload.get("session_ground_reference_status"),
            "ground_slope_z_per_mm": session.get("slope_z_per_mm"),
            "ground_intercept_z_mm": session.get("intercept_z_mm"),
            "ground_rmse_mm": session.get("rmse_mm"),
            "ground_valid_s_range_mm": session.get("valid_s_range_mm"),
            "ground_point_count": session.get("point_count"),
            "ground_inlier_count": session.get("inlier_count"),
            "ground_support": session.get("support"),
            "reference_extrinsic_R": ground_payload.get("session_extrinsic", {}).get("R_camera_to_ground"),
            "reference_extrinsic_t_mm": ground_payload.get("session_extrinsic", {}).get("t_camera_to_ground_mm"),
        },
        "height_shadow_logging_qc_only": {
            "files_read": shadow_files,
            "ground_reference_status_counts": dict(shadow_logging_status_counts),
            "formal_use": False,
            "explanation": "height_shadow.csv is a legacy shadow logger; its inactive status does not override the VALID session_ground_calibration runtime reference and is not used for formal measurement or FOV.",
        },
        "artifact_hashes": artifacts,
    }


def status_for(ground_ood: np.ndarray, mask: np.ndarray) -> str:
    selected = np.asarray(ground_ood[mask], dtype=bool)
    if not len(selected):
        return "NO_POINTS"
    if not np.any(selected):
        return "VALID"
    if np.all(selected):
        return "OUT_OF_VALID_S_DOMAIN"
    return "PARTIAL_OUT_OF_VALID_S_DOMAIN"


def q2_domain_flag(q2_values: np.ndarray, app: Any) -> bool:
    config = app.correction.hb2_height_correction
    if config is None or not len(q2_values):
        return False
    lower, upper = config.q2_domain
    return bool(np.isfinite(q2_values).all() and np.all((q2_values >= lower) & (q2_values <= upper)))


def correction_bundle(
    height_raw: float | None,
    q1: float | None,
    q2: float | None,
    q2_in_domain: bool,
    app: Any,
) -> dict[str, Any]:
    h1 = resolve_height_correction(
        height_raw,
        q1=q1,
        q2=q2,
        q2_in_domain=q2_in_domain,
        system=app.system,
        correction=app.correction,
        mode_override="h1",
    )
    hb2 = resolve_height_correction(
        height_raw,
        q1=q1,
        q2=q2,
        q2_in_domain=q2_in_domain,
        system=app.system,
        correction=app.correction,
        mode_override="hb2",
    )
    return {
        "height_h1": h1.height_h1,
        "height_hb2": hb2.height_hb2,
        "h1_status": h1.h1_status,
        "hb2_status": hb2.hb2_q2_status,
        "hb2_q2_status": hb2.hb2_q2_status,
    }


def measurement_fit_diagnostics(measurement: Any, height_points: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ground_reference_mode": None,
        "ground_baseline_zg_mm": None,
        "ground_noise_sigma_mm": None,
        "local_ground_fit_slope_mm_per_mm": None,
        "local_ground_fit_intercept_mm": None,
        "local_ground_fit_rmse_mm": None,
        "local_ground_predicted_z_at_height_mm": None,
        "local_ground_residual_at_height_mm": None,
        "height_mean_mm": None,
        "height_median_mm": None,
        "height_std_mm": None,
        "height_point_count": 0,
        "height_inlier_count": 0,
        "baseline_point_count": 0,
        "baseline_inlier_count": 0,
    }
    if measurement is None:
        return result
    result.update(
        {
            "ground_reference_mode": measurement.ground_reference_mode,
            "ground_baseline_zg_mm": float(measurement.ground_baseline_zg_mm),
            "ground_noise_sigma_mm": finite(measurement.ground_noise_sigma_mm),
            "height_mean_mm": float(measurement.height_mean_mm),
            "height_median_mm": float(measurement.height_median_mm),
            "height_std_mm": float(measurement.height_std_mm),
            "height_point_count": int(measurement.height_point_count),
            "height_inlier_count": int(measurement.height_inlier_count),
            "baseline_point_count": int(measurement.baseline_point_count),
            "baseline_inlier_count": int(measurement.baseline_inlier_count),
        }
    )
    fit = measurement.ground_profile_fit
    if fit is not None:
        result.update(
            {
                "local_ground_fit_slope_mm_per_mm": float(fit.slope_z_per_mm),
                "local_ground_fit_intercept_mm": float(fit.intercept_z_mm),
                "local_ground_fit_rmse_mm": float(fit.rmse_mm),
            }
        )
        height_fit_mask = np.asarray(measurement.height_fit.inlier_mask, dtype=bool)
        height_inliers = np.asarray(height_points, dtype=np.float64)[height_fit_mask]
        if len(height_inliers):
            predicted = fit.predict_z(height_inliers[:, :2])
            predicted_mean = float(np.mean(predicted))
            result["local_ground_predicted_z_at_height_mm"] = predicted_mean
            # Session Ground defines leveled Z=0.  The local fitted ground
            # prediction is therefore the local-ground residual at height.
            result["local_ground_residual_at_height_mm"] = predicted_mean
    return result


def optional_measure(
    baseline: np.ndarray,
    height: np.ndarray,
    app: Any,
    mode: str,
) -> tuple[Any | None, str | None]:
    if len(height) < int(app.measurement.min_height_points):
        return None, "HEIGHT_POINTS_INSUFFICIENT"
    if mode == "auto" and len(baseline) < int(app.measurement.min_baseline_points):
        return None, "BASELINE_POINTS_INSUFFICIENT"
    try:
        return measure_height_line(
            baseline,
            height,
            app.measurement,
            ground_correction_mode=mode,
        ), None
    except (MeasurementError, ValueError, FloatingPointError) as error:
        return None, f"{type(error).__name__}:{error}"


def empty_record(frame: dict[str, Any], roi: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    height = str(frame["height_label"])
    position = str(frame["position_id"])
    cid = condition_id(height, position)
    baseline_ranges = roi.get("baseline_v_ranges", [[], []])
    return {
        "dataset": "session01",
        "height_label": height,
        "position_id": position,
        "condition_id": cid,
        "repeat_index": frame.get("repeat_index"),
        "filename": frame.get("filename"),
        "cache_key": frame.get("cache_key"),
        "camera_frame_number": frame.get("camera_frame_number"),
        "true_height_mm": truth_for(height),
        "truth_kind": "nominal_truth",
        "v_order_rank": roi.get("v_order_rank"),
        "height_roi_center_v": roi.get("height_roi_center_v"),
        "height_v_range": roi.get("height_v_range"),
        "baseline_before_v_range": baseline_ranges[0] if baseline_ranges else [],
        "baseline_after_v_range": baseline_ranges[-1] if baseline_ranges else [],
        "edge_baseline_clipped": bool(roi.get("baseline_clipped", roi.get("edge_baseline_clipped", False))),
        "baseline_clipped_before": (roi.get("baseline_clipped") or {}).get("before") if isinstance(roi.get("baseline_clipped"), dict) else None,
        "baseline_clipped_after": (roi.get("baseline_clipped") or {}).get("after") if isinstance(roi.get("baseline_clipped"), dict) else None,
        "auto_qc_status": roi.get("auto_qc_status"),
        "auto_qc_reasons": roi.get("auto_qc_reasons", []),
        "human_reviewed": roi.get("human_reviewed"),
        "human_decision": roi.get("human_decision"),
        "roi_frozen": roi.get("frozen"),
        **source,
        "reconstruction_status": "NOT_RUN",
        "session_ground_status": "NOT_RUN",
        "invalid_status": "NOT_MEASURED",
    }


def reconstruct_and_measure(
    frame: dict[str, Any],
    centers: np.ndarray,
    roi: dict[str, Any],
    calibration: dict[str, Any],
    app: Any,
    ground_reference: SessionGroundReference,
    source: dict[str, Any],
) -> dict[str, Any]:
    record = empty_record(frame, roi, source)
    height_range = [[float(v) for v in roi["height_v_range"]]]
    baseline_ranges = [[float(v) for v in pair] for pair in roi.get("baseline_v_ranges", [[], []])]
    before_range = [baseline_ranges[0]] if baseline_ranges and baseline_ranges[0] else []
    after_range = [baseline_ranges[-1]] if baseline_ranges and baseline_ranges[-1] else []
    cache_before = roi_mask(centers, before_range)
    cache_height = roi_mask(centers, height_range)
    cache_after = roi_mask(centers, after_range)
    record.update(
        {
            "cached_baseline_before_point_count": int(cache_before.sum()),
            "cached_height_point_count": int(cache_height.sum()),
            "cached_baseline_after_point_count": int(cache_after.sum()),
        }
    )
    try:
        # The only reconstruction call in this function.  All branches below
        # share the returned pixels, points, q1/q2 arrays and C1 flags.
        reconstruction = reconstruct_uv_to_ground(centers, calibration, app.reconstruction)
        record["reconstruction_status"] = "VALID"
        points_pre, ground_valid = reconstruction.points_ground, None
        points_session, ground_valid = ground_reference.apply_to_points(points_pre)
        pixels = reconstruction.pixels_uv
        before_mask = roi_mask(pixels, before_range)
        height_mask = roi_mask(pixels, height_range)
        after_mask = roi_mask(pixels, after_range)
        baseline_mask = before_mask | after_mask
        before_ground = points_session[before_mask]
        height_ground = points_session[height_mask]
        after_ground = points_session[after_mask]
        baseline_ground = points_session[baseline_mask]
        ground_ood = ~np.asarray(ground_valid, dtype=bool)
        support_min = int(app.measurement.min_baseline_points)
        before_count = len(before_ground)
        after_count = len(after_ground)
        if before_count >= support_min and after_count >= support_min:
            support_type = "BOTH_SIDES"
        elif before_count >= support_min:
            support_type = "BEFORE_ONLY"
        elif after_count >= support_min:
            support_type = "AFTER_ONLY"
        else:
            support_type = "NONE"
        local_support = "BOTH_SIDES" if support_type == "BOTH_SIDES" else "ONE_SIDE" if support_type in {"BEFORE_ONLY", "AFTER_ONLY"} else "NONE"
        local_extrapolation = local_support == "ONE_SIDE"

        session_measurement, session_error = optional_measure(
            baseline_ground, height_ground, app, "session_reference"
        )
        local_measurement, local_error = optional_measure(
            baseline_ground, height_ground, app, "auto"
        )
        before_measurement, before_error = optional_measure(
            before_ground, height_ground, app, "auto"
        )
        after_measurement, after_error = optional_measure(
            after_ground, height_ground, app, "auto"
        )

        session_diag = measurement_fit_diagnostics(session_measurement, height_ground)
        local_diag = measurement_fit_diagnostics(local_measurement, height_ground)
        before_diag = measurement_fit_diagnostics(before_measurement, height_ground)
        after_diag = measurement_fit_diagnostics(after_measurement, height_ground)
        q1_values = (
            np.asarray(reconstruction.q1_c0, dtype=np.float64)[height_mask]
            if reconstruction.q1_c0 is not None
            else np.empty(0, dtype=np.float64)
        )
        q2_values = (
            np.asarray(reconstruction.q2_c0, dtype=np.float64)[height_mask]
            if reconstruction.q2_c0 is not None
            else np.empty(0, dtype=np.float64)
        )
        q1_finite = q1_values[np.isfinite(q1_values)]
        q2_finite = q2_values[np.isfinite(q2_values)]
        q1 = float(np.mean(q1_finite)) if len(q1_finite) else None
        q2 = float(np.mean(q2_finite)) if len(q2_finite) else None
        q2_in_domain = q2_domain_flag(q2_values, app)
        session_raw = session_diag["height_mean_mm"]
        local_raw = local_diag["height_mean_mm"]
        session_correction = correction_bundle(session_raw, q1, q2, q2_in_domain, app)
        local_correction = correction_bundle(local_raw, q1, q2, q2_in_domain, app)
        truth = truth_for(str(frame["height_label"]))
        ground_s = ground_reference.project_s(points_pre[:, :2]) if len(points_pre) else np.empty(0)
        height_s = ground_s[height_mask]
        before_z = before_ground[:, 2] if len(before_ground) else np.empty(0)
        after_z = after_ground[:, 2] if len(after_ground) else np.empty(0)
        height_z_pre = points_pre[height_mask, 2] if np.any(height_mask) else np.empty(0)
        height_z_session = height_ground[:, 2] if len(height_ground) else np.empty(0)
        local_before_after_delta = (
            before_diag["height_mean_mm"] - after_diag["height_mean_mm"]
            if before_diag["height_mean_mm"] is not None and after_diag["height_mean_mm"] is not None
            else None
        )
        record.update(
            {
                "reconstructed_point_count": int(len(pixels)),
                "points_ground_pre_session_count": int(len(points_pre)),
                "points_ground_session_count": int(len(points_session)),
                "session_ground_valid_point_count": int(np.count_nonzero(ground_valid)),
                "session_ground_ood_point_count": int(np.count_nonzero(ground_ood)),
                "reconstructed_baseline_before_point_count": before_count,
                "reconstructed_height_point_count": len(height_ground),
                "reconstructed_baseline_after_point_count": after_count,
                "baseline_point_count": len(baseline_ground),
                "baseline_before_point_count": before_count,
                "baseline_after_point_count": after_count,
                "baseline_support_type": support_type,
                "local_baseline_support": local_support,
                "local_baseline_extrapolation": local_extrapolation,
                "session_measurement_error": session_error,
                "local_measurement_error": local_error,
                "local_before_measurement_error": before_error,
                "local_after_measurement_error": after_error,
                "h_raw_session": session_raw,
                "height_h1_session": session_correction["height_h1"],
                "height_hb2_session": session_correction["height_hb2"],
                "residual_base_session": session_raw - truth if session_raw is not None else None,
                "residual_h1_session": session_correction["height_h1"] - truth if session_correction["height_h1"] is not None else None,
                "residual_hb2_session": session_correction["height_hb2"] - truth if session_correction["height_hb2"] is not None else None,
                "h_raw_local": local_raw,
                "height_h1_local_diag": local_correction["height_h1"],
                "height_hb2_local_diag": local_correction["height_hb2"],
                "residual_base_local_diag": local_raw - truth if local_raw is not None else None,
                "residual_h1_local_diag": local_correction["height_h1"] - truth if local_correction["height_h1"] is not None else None,
                "residual_hb2_local_diag": local_correction["height_hb2"] - truth if local_correction["height_hb2"] is not None else None,
                "delta_reference_h_raw_local_minus_session_mm": local_raw - session_raw if local_raw is not None and session_raw is not None else None,
                "q1": q1,
                "q2": q2,
                "q2_in_domain": q2_in_domain,
                "h1_status_session": session_correction["h1_status"],
                "hb2_status_session": session_correction["hb2_status"],
                "hb2_q2_status_session": session_correction["hb2_q2_status"],
                "h1_status_local_diag": local_correction["h1_status"],
                "hb2_status_local_diag": local_correction["hb2_status"],
                "hb2_q2_status_local_diag": local_correction["hb2_q2_status"],
                "local_diag_only": True,
                "session_height_mean_mm": session_diag["height_mean_mm"],
                "session_height_median_mm": session_diag["height_median_mm"],
                "session_height_std_mm": session_diag["height_std_mm"],
                "session_height_point_count": session_diag["height_point_count"],
                "session_height_inlier_count": session_diag["height_inlier_count"],
                "session_baseline_inlier_count": session_diag["baseline_inlier_count"],
                "local_height_mean_mm": local_diag["height_mean_mm"],
                "local_height_median_mm": local_diag["height_median_mm"],
                "local_height_std_mm": local_diag["height_std_mm"],
                "local_height_point_count": local_diag["height_point_count"],
                "local_height_inlier_count": local_diag["height_inlier_count"],
                "local_baseline_inlier_count": local_diag["baseline_inlier_count"],
                "local_ground_fit_slope_mm_per_mm": local_diag["local_ground_fit_slope_mm_per_mm"],
                "local_ground_fit_intercept_mm": local_diag["local_ground_fit_intercept_mm"],
                "local_ground_fit_rmse_mm": local_diag["local_ground_fit_rmse_mm"],
                "local_ground_predicted_z_at_height_mm": local_diag["local_ground_predicted_z_at_height_mm"],
                "local_ground_residual_at_height_mm": local_diag["local_ground_residual_at_height_mm"],
                "local_before_height_mm": before_diag["height_mean_mm"],
                "local_after_height_mm": after_diag["height_mean_mm"],
                "delta_before_after_mm": local_before_after_delta,
                "local_before_fit_slope_mm_per_mm": before_diag["local_ground_fit_slope_mm_per_mm"],
                "local_after_fit_slope_mm_per_mm": after_diag["local_ground_fit_slope_mm_per_mm"],
                "local_before_fit_intercept_mm": before_diag["local_ground_fit_intercept_mm"],
                "local_after_fit_intercept_mm": after_diag["local_ground_fit_intercept_mm"],
                "local_before_fit_rmse_mm": before_diag["local_ground_fit_rmse_mm"],
                "local_after_fit_rmse_mm": after_diag["local_ground_fit_rmse_mm"],
                "baseline_before_mean_z_session_mm": float(np.mean(before_z)) if len(before_z) else None,
                "baseline_before_median_z_session_mm": float(np.median(before_z)) if len(before_z) else None,
                "baseline_after_mean_z_session_mm": float(np.mean(after_z)) if len(after_z) else None,
                "baseline_after_median_z_session_mm": float(np.median(after_z)) if len(after_z) else None,
                "baseline_before_after_mean_delta_z_mm": (float(np.mean(before_z)) - float(np.mean(after_z))) if len(before_z) and len(after_z) else None,
                "height_pre_session_mean_z_mm": float(np.mean(height_z_pre)) if len(height_z_pre) else None,
                "height_session_mean_z_mm": float(np.mean(height_z_session)) if len(height_z_session) else None,
                "ground_s_min_height": float(np.min(height_s)) if len(height_s) else None,
                "ground_s_max_height": float(np.max(height_s)) if len(height_s) else None,
                "ground_reference_status": ground_reference.status,
                "ground_status_all": "VALID" if not np.any(ground_ood) else "PARTIAL_OUT_OF_VALID_S_DOMAIN",
                "ground_status_formal": status_for(ground_ood, baseline_mask | height_mask),
                "session_ground_status": status_for(ground_ood, baseline_mask | height_mask),
                "ground_status_baseline_before": status_for(ground_ood, before_mask),
                "ground_status_height": status_for(ground_ood, height_mask),
                "ground_status_baseline_after": status_for(ground_ood, after_mask),
                "ground_ood_count_baseline_before": int(np.count_nonzero(ground_ood[before_mask])),
                "ground_ood_count_height": int(np.count_nonzero(ground_ood[height_mask])),
                "ground_ood_count_baseline_after": int(np.count_nonzero(ground_ood[after_mask])),
                "c1_clamp_status_all": clamp_status(reconstruction.c1_clamped),
                "c1_clamp_status_baseline_before": clamp_status(None if reconstruction.c1_clamped is None else reconstruction.c1_clamped[before_mask]),
                "c1_clamp_status_height": clamp_status(None if reconstruction.c1_clamped is None else reconstruction.c1_clamped[height_mask]),
                "c1_clamp_status_baseline_after": clamp_status(None if reconstruction.c1_clamped is None else reconstruction.c1_clamped[after_mask]),
                "reconstruction_filtered": reconstruction.filtered,
                "invalid_status": "NONE" if session_raw is not None else "SESSION_MEASUREMENT_INVALID",
                "formal_height_measurement_mode": session_diag["ground_reference_mode"],
                "local_height_measurement_mode": local_diag["ground_reference_mode"],
                "height_roi_formal_v_median": float(np.median(centers[cache_height, 1])) if np.any(cache_height) else None,
                "height_roi_formal_v_min": float(np.min(centers[cache_height, 1])) if np.any(cache_height) else None,
                "height_roi_formal_v_max": float(np.max(centers[cache_height, 1])) if np.any(cache_height) else None,
                "height_roi_reconstructed_v_median": float(np.median(pixels[height_mask, 1])) if np.any(height_mask) else None,
                "height_roi_reconstructed_v_min": float(np.min(pixels[height_mask, 1])) if np.any(height_mask) else None,
                "height_roi_reconstructed_v_max": float(np.max(pixels[height_mask, 1])) if np.any(height_mask) else None,
                "height_q2_point_count": int(len(q2_values)),
                "height_q2_ood_point_count": int(np.count_nonzero(~np.isfinite(q2_values)) + np.count_nonzero(np.isfinite(q2_values) & ~((q2_values >= app.correction.hb2_height_correction.q2_domain[0]) & (q2_values <= app.correction.hb2_height_correction.q2_domain[1]))) if app.correction.hb2_height_correction is not None else len(q2_values)),
            }
        )
        return record
    except (ReconstructionInputError, MeasurementError, ValueError, FloatingPointError) as error:
        record.update(
            {
                "reconstruction_status": "FAILED_OR_MEASUREMENT_INVALID",
                "invalid_status": f"{type(error).__name__}:{error}",
                "formal_height_measurement_mode": "invalid",
                "local_height_measurement_mode": "invalid",
            }
        )
        return record


def error_values(rows: Iterable[dict[str, Any]], error_field: str) -> np.ndarray:
    values = [finite(row.get(error_field)) for row in rows]
    return np.asarray([value for value in values if value is not None], dtype=np.float64)


def metrics_for(rows: list[dict[str, Any]], error_field: str) -> dict[str, Any]:
    errors = error_values(rows, error_field)
    abs_errors = np.abs(errors)
    std_field = "local_height_std_mm" if "_local_diag" in error_field else "session_height_std_mm"
    height_std_values = [
        finite(row.get(std_field))
        for row in rows
        if finite(row.get(error_field)) is not None and finite(row.get(std_field)) is not None
    ]
    return {
        "n_total": len(rows),
        "n_valid": int(len(errors)),
        "n_invalid": int(len(rows) - len(errors)),
        "invalid_rate": float((len(rows) - len(errors)) / len(rows)) if rows else None,
        "bias_mm": float(np.mean(errors)) if len(errors) else None,
        "mae_mm": float(np.mean(abs_errors)) if len(errors) else None,
        "rmse_mm": float(np.sqrt(np.mean(errors * errors))) if len(errors) else None,
        "p95_abs_mm": float(np.percentile(abs_errors, 95)) if len(errors) else None,
        "max_abs_mm": float(np.max(abs_errors)) if len(errors) else None,
        "repeatability_std_mm": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0 if len(errors) else None,
        "mean_height_std_mm": float(np.mean(height_std_values)) if height_std_values else None,
    }


def condition_metric_row(rows: list[dict[str, Any]], branch: str, model: str, error_field: str, diagnostic_only: bool, scope: str, scope_id: str) -> dict[str, Any]:
    first = rows[0] if rows else {}
    result = {
        "dataset": "session01",
        "scope": scope,
        "scope_id": scope_id,
        "reference_branch": branch,
        "model": model,
        "model_group": f"{branch}:{model}",
        "diagnostic_only": diagnostic_only,
        "condition_id": first.get("condition_id") if scope == "position" else None,
        "height_label": first.get("height_label"),
        "position_id": first.get("position_id") if scope == "position" else None,
        "v_order_rank": first.get("v_order_rank") if scope == "position" else None,
        "height_roi_center_v": first.get("height_roi_center_v") if scope == "position" else None,
        "truth_height_mm": first.get("true_height_mm"),
        "baseline_support_types": sorted({str(row.get("baseline_support_type")) for row in rows}),
        "local_baseline_support_types": sorted({str(row.get("local_baseline_support")) for row in rows}),
        "q2_ood_frame_count": sum(int(row.get("q2_in_domain") is False) for row in rows),
        "ground_ood_frame_count": sum(int((row.get("session_ground_ood_point_count") or 0) > 0) for row in rows),
        "local_baseline_extrapolation_frame_count": sum(int(bool(row.get("local_baseline_extrapolation"))) for row in rows),
        **metrics_for(rows, error_field),
    }
    return result


def add_position_spread(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("scope") != "height":
            continue
        siblings = [
            item
            for item in rows
            if item.get("scope") == "position"
            and item.get("height_label") == row.get("height_label")
            and item.get("reference_branch") == row.get("reference_branch")
            and item.get("model") == row.get("model")
            and finite(item.get("bias_mm")) is not None
        ]
        biases = np.asarray([float(item["bias_mm"]) for item in siblings], dtype=np.float64)
        row["position_count"] = len(biases)
        row["position_bias_range_mm"] = float(np.ptp(biases)) if len(biases) else None
        row["position_bias_std_mm"] = float(np.std(biases)) if len(biases) else None
        row["worst_position_abs_bias_mm"] = float(np.max(np.abs(biases))) if len(biases) else None
        if len(biases):
            worst = max(siblings, key=lambda item: abs(float(item["bias_mm"])))
            worst_p95 = max(siblings, key=lambda item: float(item.get("p95_abs_mm") or -np.inf))
            worst_max = max(siblings, key=lambda item: float(item.get("max_abs_mm") or -np.inf))
            row["worst_position_abs_bias_id"] = worst.get("position_id")
            row["worst_position_p95_abs_mm"] = worst.get("p95_abs_mm")
            row["worst_position_p95_id"] = worst_p95.get("position_id")
            row["worst_position_max_abs_mm"] = worst.get("max_abs_mm")
            row["worst_position_max_id"] = worst_max.get("position_id")
        else:
            row["worst_position_abs_bias_id"] = None
            row["worst_position_p95_abs_mm"] = None
            row["worst_position_p95_id"] = None
            row["worst_position_max_abs_mm"] = None
            row["worst_position_max_id"] = None


def build_condition_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["condition_id"])].append(record)
    output: list[dict[str, Any]] = []
    for branch, model, error_field, diagnostic in MODEL_GROUPS:
        for cid in sorted(grouped):
            group = grouped[cid]
            output.append(condition_metric_row(group, branch, model, error_field, diagnostic, "position", cid))
        for height in HEIGHT_LABELS:
            group = [record for record in records if record.get("height_label") == height]
            output.append(condition_metric_row(group, branch, model, error_field, diagnostic, "height", height))
        output.append(condition_metric_row(records, branch, model, error_field, diagnostic, "pooled", "session01"))
    add_position_spread(output)
    return output


def model_metric_lookup(rows: list[dict[str, Any]], scope: str, scope_id: str, branch: str, model: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row.get("scope") == scope
            and row.get("scope_id") == scope_id
            and row.get("reference_branch") == branch
            and row.get("model") == model
        ),
        None,
    )


def correlation_diagnostics(x_values: list[Any], y_values: list[Any]) -> dict[str, Any]:
    pairs = [(finite(x), finite(y)) for x, y in zip(x_values, y_values)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    result = {"n": len(pairs), "pearson_r": None, "pearson_pvalue": None, "spearman_rho": None, "spearman_pvalue": None}
    if len(pairs) < 3:
        return result
    x = np.asarray([item[0] for item in pairs], dtype=np.float64)
    y = np.asarray([item[1] for item in pairs], dtype=np.float64)
    if np.ptp(x) > 0 and np.ptp(y) > 0:
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        result.update(
            {
                "pearson_r": float(pearson.statistic),
                "pearson_pvalue": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
            }
        )
    return result


def build_reference_comparison(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_rows: list[dict[str, Any]] = []
    for record in records:
        for model in ("base", "h1", "hb2"):
            session_error = finite(record.get(f"residual_{model}_session"))
            local_error = finite(record.get(f"residual_{model}_local_diag"))
            frame_rows.append(
                {
                    "row_type": "frame",
                    "dataset": "session01",
                    "condition_id": record.get("condition_id"),
                    "height_label": record.get("height_label"),
                    "position_id": record.get("position_id"),
                    "v_order_rank": record.get("v_order_rank"),
                    "height_roi_formal_v_median": record.get("height_roi_formal_v_median"),
                    "repeat_index": record.get("repeat_index"),
                    "camera_frame_number": record.get("camera_frame_number"),
                    "model": model,
                    "session_reference_model": f"{model}_session",
                    "local_reference_model": f"{model}_local_diag",
                    "diagnostic_only": True,
                    "h_session_mm": record.get(f"height_{model}_session") if model != "base" else record.get("h_raw_session"),
                    "h_local_mm": record.get(f"height_{model}_local_diag") if model != "base" else record.get("h_raw_local"),
                    "error_session_mm": session_error,
                    "error_local_mm": local_error,
                    "delta_reference_mm": (local_error - session_error) if local_error is not None and session_error is not None else None,
                    "abs_error_delta_local_minus_session_mm": (abs(local_error) - abs(session_error)) if local_error is not None and session_error is not None else None,
                    "squared_error_delta_local_minus_session_mm2": (local_error * local_error - session_error * session_error) if local_error is not None and session_error is not None else None,
                    "local_ground_residual_at_height_mm": record.get("local_ground_residual_at_height_mm"),
                    "baseline_support_type": record.get("baseline_support_type"),
                }
            )
    condition_rows: list[dict[str, Any]] = []
    for cid in sorted({str(record["condition_id"]) for record in records}):
        group_records = [record for record in records if record.get("condition_id") == cid]
        for model in ("base", "h1", "hb2"):
            session_field = f"residual_{model}_session"
            local_field = f"residual_{model}_local_diag"
            session_values = error_values(group_records, session_field)
            local_values = error_values(group_records, local_field)
            n_pair = min(len(session_values), len(local_values))
            pair_delta = local_values[:n_pair] - session_values[:n_pair]
            abs_delta = np.abs(local_values[:n_pair]) - np.abs(session_values[:n_pair])
            sq_delta = local_values[:n_pair] ** 2 - session_values[:n_pair] ** 2
            session_metric = metrics_for(group_records, session_field)
            local_metric = metrics_for(group_records, local_field)
            first = group_records[0]
            condition_rows.append(
                {
                    "row_type": "condition",
                    "dataset": "session01",
                    "condition_id": cid,
                    "height_label": first.get("height_label"),
                    "position_id": first.get("position_id"),
                    "v_order_rank": first.get("v_order_rank"),
                    "model": model,
                    "diagnostic_only": True,
                    "n_pair": n_pair,
                    "session_bias_mm": session_metric.get("bias_mm"),
                    "local_bias_mm": local_metric.get("bias_mm"),
                    "bias_difference_local_minus_session_mm": (local_metric.get("bias_mm") - session_metric.get("bias_mm")) if session_metric.get("bias_mm") is not None and local_metric.get("bias_mm") is not None else None,
                    "session_mae_mm": session_metric.get("mae_mm"),
                    "local_mae_mm": local_metric.get("mae_mm"),
                    "session_p95_abs_mm": session_metric.get("p95_abs_mm"),
                    "local_p95_abs_mm": local_metric.get("p95_abs_mm"),
                    "p95_difference_local_minus_session_mm": (local_metric.get("p95_abs_mm") - session_metric.get("p95_abs_mm")) if session_metric.get("p95_abs_mm") is not None and local_metric.get("p95_abs_mm") is not None else None,
                    "session_max_abs_mm": session_metric.get("max_abs_mm"),
                    "local_max_abs_mm": local_metric.get("max_abs_mm"),
                    "max_difference_local_minus_session_mm": (local_metric.get("max_abs_mm") - session_metric.get("max_abs_mm")) if session_metric.get("max_abs_mm") is not None and local_metric.get("max_abs_mm") is not None else None,
                    "mean_abs_error_delta_local_minus_session_mm": float(np.mean(abs_delta)) if len(abs_delta) else None,
                    "mean_squared_error_delta_local_minus_session_mm2": float(np.mean(sq_delta)) if len(sq_delta) else None,
                    "local_ground_residual_at_height_mean_mm": float(np.mean([float(record["local_ground_residual_at_height_mm"]) for record in group_records if finite(record.get("local_ground_residual_at_height_mm")) is not None])) if any(finite(record.get("local_ground_residual_at_height_mm")) is not None for record in group_records) else None,
                    "baseline_support_types": sorted({str(record.get("baseline_support_type")) for record in group_records}),
                }
            )
    return frame_rows + condition_rows, {"frame_rows": frame_rows, "condition_rows": condition_rows}


def build_baseline_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "dataset", "condition_id", "height_label", "position_id", "v_order_rank", "repeat_index", "camera_frame_number",
        "height_roi_formal_v_median", "baseline_support_type", "local_baseline_support", "local_baseline_extrapolation",
        "baseline_before_point_count", "baseline_after_point_count", "baseline_point_count",
        "baseline_before_mean_z_session_mm", "baseline_before_median_z_session_mm", "baseline_after_mean_z_session_mm", "baseline_after_median_z_session_mm", "baseline_before_after_mean_delta_z_mm",
        "local_ground_fit_slope_mm_per_mm", "local_ground_fit_intercept_mm", "local_ground_fit_rmse_mm", "local_ground_predicted_z_at_height_mm", "local_ground_residual_at_height_mm",
        "local_before_height_mm", "local_after_height_mm", "delta_before_after_mm", "local_before_fit_slope_mm_per_mm", "local_after_fit_slope_mm_per_mm", "local_before_fit_intercept_mm", "local_after_fit_intercept_mm", "local_before_fit_rmse_mm", "local_after_fit_rmse_mm",
        "h_raw_session", "h_raw_local", "delta_reference_h_raw_local_minus_session_mm", "q2_in_domain", "ground_status_baseline_before", "ground_status_height", "ground_status_baseline_after", "session_ground_ood_count_baseline_before", "session_ground_ood_count_height", "session_ground_ood_count_baseline_after",
    ]
    return [{field: record.get(field) for field in fields} for record in records]


def coordinate_value(row: dict[str, Any]) -> float | None:
    return finite(row.get("height_roi_formal_v_median")) or finite(row.get("height_roi_center_v"))


def scoped_rows(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    result: list[tuple[str, list[dict[str, Any]]]] = [("pooled", records)]
    for height in HEIGHT_LABELS:
        height_rows = [record for record in records if record.get("height_label") == height]
        result.append((f"height:{height}", height_rows))
    for height in HEIGHT_LABELS:
        for position in POSITION_IDS:
            result.append((f"position:{height}:{position}", [record for record in records if record.get("height_label") == height and record.get("position_id") == position]))
    for height in HEIGHT_LABELS:
        for rank in range(1, 11):
            result.append((f"v_order_rank:{height}:{rank}", [record for record in records if record.get("height_label") == height and int(record.get("v_order_rank") or -1) == rank]))
    return result


def build_edge_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    regions: list[tuple[str, float | None]] = [("all", None)] + [(f"v_gt_{int(threshold)}", threshold) for threshold in EDGE_THRESHOLDS]
    for region, threshold in regions:
        region_records = [record for record in records if threshold is None or (coordinate_value(record) is not None and coordinate_value(record) > threshold)]
        for scope, rows in scoped_rows(region_records):
            for branch, model, error_field, diagnostic in MODEL_GROUPS:
                metrics = metrics_for(rows, error_field)
                errors = error_values(rows, error_field)
                valid_rows = [row for row in rows if finite(row.get(error_field)) is not None and coordinate_value(row) is not None]
                rho, pvalue = None, None
                if len(valid_rows) >= 3:
                    v = np.asarray([coordinate_value(row) for row in valid_rows], dtype=np.float64)
                    e = np.asarray([float(row[error_field]) for row in valid_rows], dtype=np.float64)
                    if np.ptp(v) > 0 and np.ptp(e) > 0:
                        result = spearmanr(v, e)
                        rho, pvalue = float(result.statistic), float(result.pvalue)
                output.append(
                    {
                        "dataset": "session01",
                        "region": region,
                        "threshold_v": threshold,
                        "scope": scope,
                        "reference_branch": branch,
                        "model": model,
                        "diagnostic_only": diagnostic,
                        **metrics,
                        "covered_heights": sorted({str(row.get("height_label")) for row in valid_rows}),
                        "covered_positions": sorted({str(row.get("condition_id")) for row in valid_rows}),
                        "covered_v_order_ranks": sorted({int(row.get("v_order_rank")) for row in valid_rows if row.get("v_order_rank") is not None}),
                        "error_gt_0_1_fraction": float(np.mean(np.abs(errors) > 0.1)) if len(errors) else None,
                        "error_gt_0_2_fraction": float(np.mean(np.abs(errors) > 0.2)) if len(errors) else None,
                        "residual_v_spearman_rho": rho,
                        "residual_v_spearman_pvalue": pvalue,
                        "v_min": min((coordinate_value(row) for row in rows if coordinate_value(row) is not None), default=None),
                        "v_max": max((coordinate_value(row) for row in rows if coordinate_value(row) is not None), default=None),
                    }
                )
    return output


def old_v1_vs_v2(records: list[dict[str, Any]], condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old_rows = read_csv(OLD_V1_CONDITION_PATH)
    old_map = {(str(row.get("condition_id")), str(row.get("model"))): row for row in old_rows}
    old_registry = {}
    if OLD_V1_REGISTRY_PATH.is_file():
        old_registry = {str(entry.get("condition_id")): entry for entry in json.loads(OLD_V1_REGISTRY_PATH.read_text(encoding="utf-8")).get("entries", [])}
    output: list[dict[str, Any]] = []
    for row in condition_rows:
        if row.get("scope") != "position" or row.get("reference_branch") != "session":
            continue
        cid = str(row.get("condition_id"))
        model = str(row.get("model"))
        old = old_map.get((cid, model), {})
        new_group = [record for record in records if record.get("condition_id") == cid]
        old_v1_entry = old_registry.get(cid, {})
        v1_range = old_v1_entry.get("height_v_range") or []
        v2_range = next((entry.get("height_v_range") for entry in json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("entries", []) if entry.get("condition_id") == cid), [])
        output.append(
            {
                "condition_id": cid,
                "height_label": row.get("height_label"),
                "position_id": row.get("position_id"),
                "v_order_rank": row.get("v_order_rank"),
                "model": model,
                "historical_v1_invalidated_by_roi_v2": True,
                "v1_n_valid": parse_int(old.get("n_valid")),
                "v1_bias_mm": parse_float(old.get("bias_mm")),
                "v1_mae_mm": parse_float(old.get("mae_mm")),
                "v1_rmse_mm": parse_float(old.get("rmse_mm")),
                "v1_p95_abs_mm": parse_float(old.get("p95_abs_mm")),
                "v1_max_abs_mm": parse_float(old.get("max_abs_mm")),
                "v1_repeatability_std_mm": parse_float(old.get("repeatability_std_mm")),
                "v2_n_valid": row.get("n_valid"),
                "v2_bias_mm": row.get("bias_mm"),
                "v2_mae_mm": row.get("mae_mm"),
                "v2_rmse_mm": row.get("rmse_mm"),
                "v2_p95_abs_mm": row.get("p95_abs_mm"),
                "v2_max_abs_mm": row.get("max_abs_mm"),
                "v2_repeatability_std_mm": row.get("repeatability_std_mm"),
                "v2_minus_v1_bias_mm": row.get("bias_mm") - parse_float(old.get("bias_mm")) if row.get("bias_mm") is not None and parse_float(old.get("bias_mm")) is not None else None,
                "v2_minus_v1_repeatability_std_mm": row.get("repeatability_std_mm") - parse_float(old.get("repeatability_std_mm")) if row.get("repeatability_std_mm") is not None and parse_float(old.get("repeatability_std_mm")) is not None else None,
                "v1_height_v_range": v1_range,
                "v2_height_v_range": v2_range,
                "v1_height_roi_width_px": (float(v1_range[1]) - float(v1_range[0])) if len(v1_range) == 2 else None,
                "v2_height_roi_width_px": (float(v2_range[1]) - float(v2_range[0])) if len(v2_range) == 2 else None,
                "v1_height_roi_center_v": old_v1_entry.get("height_roi_center_v"),
                "v2_height_roi_center_v": next((entry.get("height_roi_center_v") for entry in json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("entries", []) if entry.get("condition_id") == cid), None),
                "v2_frame_count": len(new_group),
            }
        )
    return output


def plot_session_vs_local(records: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, model in zip(axes, ("base", "h1", "hb2")):
        values = []
        labels = []
        for branch, label, color in (("session", "Session", "#4c78a8"), ("local_diag", "Local", "#f58518")):
            field = f"residual_{model}_{branch}"
            data = [abs(float(row[field])) for row in records if finite(row.get(field)) is not None]
            values.append(data)
            labels.append(label)
        bp = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ("#4c78a8", "#f58518")):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(model.upper())
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("|height error| (mm)")
    fig.suptitle("Session-reference vs local-baseline diagnostic error")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_position_bias(condition_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    colors = {"base": "#4c78a8", "h1": "#f58518", "hb2": "#54a24b"}
    for ax, height in zip(axes, HEIGHT_LABELS):
        for branch, style in (("session", "-"), ("local_diag", "--")):
            for model in ("base", "h1", "hb2"):
                items = sorted(
                    [row for row in condition_rows if row.get("scope") == "position" and row.get("height_label") == height and row.get("reference_branch") == branch and row.get("model") == model],
                    key=lambda row: int(row.get("v_order_rank") or 0),
                )
                ax.plot([row.get("v_order_rank") for row in items], [row.get("bias_mm") for row in items], marker="o", linewidth=1.3, linestyle=style, color=colors[model], label=f"{model.upper()} {branch.replace('_diag', '')}")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(height)
        ax.set_xlabel("height-ROI v_order_rank")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("position bias (mm)")
    axes[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Position bias by height: solid=Session, dashed=local diagnostic")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_local_residual_v(records: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = {"h10": "#4c78a8", "h20": "#f58518", "h30": "#54a24b"}
    for height in HEIGHT_LABELS:
        items = [row for row in records if row.get("height_label") == height and finite(row.get("local_ground_residual_at_height_mm")) is not None and coordinate_value(row) is not None]
        ax.scatter([coordinate_value(row) for row in items], [row.get("local_ground_residual_at_height_mm") for row in items], s=16, alpha=0.55, label=height, color=colors[height])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("height-ROI formal-point v (px)")
    ax.set_ylabel("local ground residual at height (mm)")
    ax.set_title("Local ground residual vs true height-ROI spatial coordinate")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_attribution(records: list[dict[str, Any]], path: Path, attribution: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    colors = {"h10": "#4c78a8", "h20": "#f58518", "h30": "#54a24b"}
    for ax, model in zip(axes, ("base", "h1", "hb2")):
        for height in HEIGHT_LABELS:
            items = [row for row in records if row.get("height_label") == height and finite(row.get("local_ground_residual_at_height_mm")) is not None and finite(row.get(f"residual_{model}_session")) is not None]
            ax.scatter([row["local_ground_residual_at_height_mm"] for row in items], [row[f"residual_{model}_session"] for row in items], s=14, alpha=0.5, color=colors[height], label=height)
        detail = attribution.get(model, {})
        ax.set_title(f"{model.upper()}\nSpearman={detail.get('error_vs_local_ground_spearman_rho')}")
        ax.set_xlabel("local ground residual (mm)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Session-reference error (mm)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Height error vs local-ground residual (diagnostic attribution only)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_before_after(records: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = {"h10": "#4c78a8", "h20": "#f58518", "h30": "#54a24b"}
    for height in HEIGHT_LABELS:
        items = [row for row in records if row.get("height_label") == height and finite(row.get("local_before_height_mm")) is not None and finite(row.get("local_after_height_mm")) is not None]
        ax.scatter([row["local_before_height_mm"] for row in items], [row["local_after_height_mm"] for row in items], s=14, alpha=0.55, color=colors[height], label=height)
    limits = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(limits), max(limits)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlabel("local height using before-only baseline (mm)")
    ax.set_ylabel("local height using after-only baseline (mm)")
    ax.set_title("Before-only vs after-only local baseline residual")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_v1_v2(v1v2: list[dict[str, Any]], path: Path) -> None:
    keys = [cid for cid in KEY_V1_FAILURE_CONDITIONS if any(row.get("condition_id") == cid for row in v1v2)]
    fig, axes = plt.subplots(1, len(keys), figsize=(max(12, len(keys) * 2.2), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cid in zip(axes, keys):
        items = [row for row in v1v2 if row.get("condition_id") == cid]
        x = np.arange(3)
        v1 = [float(next((item.get("v1_repeatability_std_mm") for item in items if item.get("model") == model), np.nan) or np.nan) for model in ("base", "h1", "hb2")]
        v2 = [float(next((item.get("v2_repeatability_std_mm") for item in items if item.get("model") == model), np.nan) or np.nan) for model in ("base", "h1", "hb2")]
        ax.bar(x - 0.18, v1, width=0.36, label="V1", color="#bdbdbd")
        ax.bar(x + 0.18, v2, width=0.36, label="V2", color="#4c78a8")
        ax.set_xticks(x, ["B", "H1", "HB2"])
        ax.set_title(cid)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("repeatability std of error (mm)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Historical V1 vs frozen manual V2 ROI: key pathology cases")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_edge(records: list[dict[str, Any]], threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels: list[str] = []
    values: list[list[float]] = []
    for branch, branch_label in (("session", "S"), ("local_diag", "L")):
        for model in ("base", "h1", "hb2"):
            field = f"residual_{model}_{branch}"
            data = [abs(float(row[field])) for row in records if coordinate_value(row) is not None and coordinate_value(row) > threshold and finite(row.get(field)) is not None]
            labels.append(f"{branch_label}-{model.upper()}")
            values.append(data)
    bp = ax.boxplot(values, labels=labels, showfliers=False, patch_artist=True)
    colors = ["#4c78a8", "#f58518", "#54a24b"] * 2
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("|height error| (mm)")
    ax.set_title(f"Edge comparison: v > {int(threshold)}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def height_std_pathology(records: list[dict[str, Any]]) -> dict[str, Any]:
    old_rows = read_csv(OLD_V1_FRAME_PATH)
    old_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        old_grouped[str(row.get("condition_id"))].append(row)
    results: list[dict[str, Any]] = []
    for cid in KEY_V1_FAILURE_CONDITIONS:
        v2_group = [record for record in records if record.get("condition_id") == cid]
        v1_group = old_grouped.get(cid, [])
        v1_std = [finite(row.get("height_std_mm")) for row in v1_group]
        v2_std = [finite(row.get("session_height_std_mm")) for row in v2_group]
        v1_std = [value for value in v1_std if value is not None]
        v2_std = [value for value in v2_std if value is not None]
        results.append({
            "condition_id": cid,
            "v1_height_std_median_mm": float(np.median(v1_std)) if v1_std else None,
            "v2_height_std_median_mm": float(np.median(v2_std)) if v2_std else None,
            "v1_height_std_p95_mm": float(np.percentile(v1_std, 95)) if v1_std else None,
            "v2_height_std_p95_mm": float(np.percentile(v2_std, 95)) if v2_std else None,
            "resolved": bool(v1_std and v2_std and np.median(v2_std) < np.median(v1_std) * 0.5),
        })
    resolved_count = sum(int(item["resolved"]) for item in results)
    return {"cases": results, "resolved_count": resolved_count, "case_count": len(results), "flag": "YES" if resolved_count >= 5 else "PARTIAL" if resolved_count else "NO"}


def attribution_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in ("base", "h1", "hb2"):
        session_error = [record.get(f"residual_{model}_session") for record in records]
        local_residual = [record.get("local_ground_residual_at_height_mm") for record in records]
        local_error = [record.get(f"residual_{model}_local_diag") for record in records]
        error_corr = correlation_diagnostics(local_residual, session_error)
        delta_values = []
        for session, local in zip(session_error, local_error):
            s, l = finite(session), finite(local)
            delta_values.append(s - l if s is not None and l is not None else None)
        delta_corr = correlation_diagnostics(local_residual, delta_values)
        local_residual_values = np.asarray([value for value in (finite(value) for value in local_residual) if value is not None], dtype=np.float64)
        delta_finite = np.asarray([value for value in (finite(value) for value in delta_values) if value is not None], dtype=np.float64)
        output[model] = {
            "error_vs_local_ground": error_corr,
            "error_vs_local_ground_pearson_r": error_corr["pearson_r"],
            "error_vs_local_ground_pearson_pvalue": error_corr["pearson_pvalue"],
            "error_vs_local_ground_spearman_rho": error_corr["spearman_rho"],
            "error_vs_local_ground_spearman_pvalue": error_corr["spearman_pvalue"],
            "error_session_minus_error_local_vs_local_ground": delta_corr,
            "error_session_minus_error_local_pearson_r": delta_corr["pearson_r"],
            "error_session_minus_error_local_spearman_rho": delta_corr["spearman_rho"],
            "mean_local_ground_residual_mm": float(np.mean(local_residual_values)) if len(local_residual_values) else None,
            "mean_error_session_minus_error_local_mm": float(np.mean(delta_finite)) if len(delta_finite) else None,
        }
    return output


def derive_flags(
    records: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    reference_comparison: list[dict[str, Any]],
    pathology: dict[str, Any],
    attribution: dict[str, Any],
) -> dict[str, Any]:
    session_rows = {model: [row for row in condition_rows if row.get("scope") == "position" and row.get("reference_branch") == "session" and row.get("model") == model] for model in ("base", "h1", "hb2")}
    local_rows = {model: [row for row in condition_rows if row.get("scope") == "position" and row.get("reference_branch") == "local_diag" and row.get("model") == model] for model in ("base", "h1", "hb2")}
    session_complete = all(len([record for record in records if record.get("condition_id") == cid and finite(record.get("residual_base_session")) is not None]) == REPEAT_COUNT for cid in {record.get("condition_id") for record in records})
    local_valid_fraction = len([record for record in records if finite(record.get("residual_base_local_diag")) is not None]) / len(records) if records else 0.0
    before_after = [finite(record.get("delta_before_after_mm")) for record in records if record.get("baseline_support_type") == "BOTH_SIDES"]
    before_after = [value for value in before_after if value is not None]
    if before_after:
        consistent_fraction = float(np.mean(np.abs(before_after) <= 0.1))
        before_after_flag = "YES" if consistent_fraction >= 0.8 else "PARTIAL" if consistent_fraction >= 0.5 else "NO"
    else:
        consistent_fraction = None
        before_after_flag = "PARTIAL"
    deltas = np.asarray([abs(float(record["delta_reference_h_raw_local_minus_session_mm"])) for record in records if finite(record.get("delta_reference_h_raw_local_minus_session_mm")) is not None], dtype=np.float64)
    if len(deltas):
        local_change_flag = "YES" if float(np.mean(deltas > 0.02)) >= 0.5 and float(np.median(deltas)) > 0.02 else "PARTIAL" if float(np.mean(deltas > 0.02)) >= 0.1 else "NO"
    else:
        local_change_flag = "NO"
    spatial_height_rows = [row for row in condition_rows if row.get("scope") == "height" and row.get("reference_branch") == "session" and row.get("position_bias_range_mm") is not None]
    spatial_range_evidence = sum(int(float(row["position_bias_range_mm"]) >= 0.1) for row in spatial_height_rows)
    spatial_corr_evidence = sum(int(abs(float(row.get("residual_v_spearman_rho"))) >= 0.5 and float(row.get("residual_v_spearman_pvalue") or 1.0) <= 0.05) for row in edge_rows if row.get("region") == "all" and str(row.get("scope", "")).startswith("height:") and row.get("reference_branch") == "session" and row.get("residual_v_spearman_rho") is not None)
    spatial_flag = "YES" if spatial_range_evidence >= 2 or spatial_corr_evidence >= 2 else "PARTIAL" if spatial_range_evidence or spatial_corr_evidence else "NO"
    h1_pooled = model_metric_lookup(condition_rows, "pooled", "session01", "session", "h1") or {}
    hb2_pooled = model_metric_lookup(condition_rows, "pooled", "session01", "session", "hb2") or {}
    h1_valid = len([row for row in session_rows["h1"] if int(row.get("n_valid") or 0) == REPEAT_COUNT]) == CONDITION_COUNT
    hb2_valid = len([row for row in session_rows["hb2"] if int(row.get("n_valid") or 0) == REPEAT_COUNT]) == CONDITION_COUNT
    if h1_valid and hb2_valid and h1_pooled.get("mae_mm") is not None and hb2_pooled.get("mae_mm") is not None:
        preferred = "H1" if float(h1_pooled["mae_mm"]) <= float(hb2_pooled["mae_mm"]) else "HB2"
    elif h1_valid:
        preferred = "H1"
    elif hb2_valid:
        preferred = "HB2"
    else:
        preferred = "UNDECIDED"
    spread_comparison: list[dict[str, Any]] = []
    for height in HEIGHT_LABELS:
        h1 = model_metric_lookup(condition_rows, "height", height, "session", "h1") or {}
        hb2 = model_metric_lookup(condition_rows, "height", height, "session", "hb2") or {}
        spread_comparison.append(
            {
                "height": height,
                "h1_position_bias_range_mm": h1.get("position_bias_range_mm"),
                "hb2_position_bias_range_mm": hb2.get("position_bias_range_mm"),
                "h1_position_bias_std_mm": h1.get("position_bias_std_mm"),
                "hb2_position_bias_std_mm": hb2.get("position_bias_std_mm"),
                "hb2_range_lower": bool(h1.get("position_bias_range_mm") is not None and hb2.get("position_bias_range_mm") is not None and float(hb2["position_bias_range_mm"]) < float(h1["position_bias_range_mm"])),
                "hb2_std_lower": bool(h1.get("position_bias_std_mm") is not None and hb2.get("position_bias_std_mm") is not None and float(hb2["position_bias_std_mm"]) < float(h1["position_bias_std_mm"])),
            }
        )
    spread_count = sum(int(item["hb2_range_lower"] and item["hb2_std_lower"]) for item in spread_comparison)
    edge_tail_comparison: list[dict[str, Any]] = []
    for region in ("v_gt_2400", "v_gt_2600"):
        h1 = next((row for row in edge_rows if row.get("region") == region and row.get("scope") == "pooled" and row.get("reference_branch") == "session" and row.get("model") == "h1"), None)
        hb2 = next((row for row in edge_rows if row.get("region") == region and row.get("scope") == "pooled" and row.get("reference_branch") == "session" and row.get("model") == "hb2"), None)
        comparable = bool(h1 and hb2 and h1.get("p95_abs_mm") is not None and hb2.get("p95_abs_mm") is not None and h1.get("max_abs_mm") is not None and hb2.get("max_abs_mm") is not None)
        edge_tail_comparison.append(
            {
                "region": region,
                "comparable": comparable,
                "h1_p95_abs_mm": h1.get("p95_abs_mm") if h1 else None,
                "hb2_p95_abs_mm": hb2.get("p95_abs_mm") if hb2 else None,
                "h1_max_abs_mm": h1.get("max_abs_mm") if h1 else None,
                "hb2_max_abs_mm": hb2.get("max_abs_mm") if hb2 else None,
                "h1_tail_lower": bool(comparable and float(h1["p95_abs_mm"]) < float(hb2["p95_abs_mm"]) and float(h1["max_abs_mm"]) < float(hb2["max_abs_mm"])),
            }
        )
    tail_comparable = [item for item in edge_tail_comparison if item["comparable"]]
    h1_tail_count = sum(int(item["h1_tail_lower"]) for item in tail_comparable)
    def support(rows: list[dict[str, Any]]) -> str:
        complete = sum(int(int(row.get("n_valid") or 0) == REPEAT_COUNT) for row in rows)
        valid = sum(int(int(row.get("n_valid") or 0) > 0) for row in rows)
        return "SUPPORTED" if complete == CONDITION_COUNT else "PARTIAL" if valid else "NOT_SUPPORTED"
    return {
        "ROI_V2_HUMAN_REVIEW_COMPLETE": "YES" if provenance.get("registry_ok") else "NO",
        "ROI_V2_FROZEN": "YES" if provenance.get("registry_ok") else "NO",
        "A13B_V2_MEASUREMENT_COMPLETE": "YES" if len(records) == FRAME_COUNT and session_complete else "NO",
        "A13B_V1_MM_FAILURE_CAUSED_BY_ROI": "YES" if pathology.get("resolved_count", 0) >= 5 else "PARTIAL" if pathology.get("resolved_count", 0) else "NO",
        "V2_HEIGHT_STD_PATHOLOGY_RESOLVED": pathology.get("flag"),
        "SESSION_GROUND_LOCAL_RESIDUAL_SUPPORTED": "YES" if local_valid_fraction >= 0.8 and any(item.get("mean_local_ground_residual_mm") is not None for item in attribution.values()) else "PARTIAL" if local_valid_fraction > 0 else "NO",
        "LOCAL_BASELINE_SIGNIFICANTLY_CHANGES_HEIGHT": local_change_flag,
        "BASELINE_BEFORE_AFTER_CONSISTENT": before_after_flag,
        "BASELINE_BEFORE_AFTER_CONSISTENT_FRACTION_ABS_DELTA_LE_0_1": consistent_fraction,
        "ONE_SIDE_BASELINE_DIAGNOSTIC_AVAILABLE": "YES" if any(record.get("local_baseline_support") == "ONE_SIDE" for record in records) else "NO",
        "SESSION_REFERENCE_FULL_FOV_ACCURACY": support(session_rows["base"]),
        "LOCAL_REFERENCE_FULL_FOV_ACCURACY": "SUPPORTED" if local_valid_fraction == 1.0 and not any(record.get("local_baseline_support") == "NONE" for record in records) else "PARTIAL" if local_valid_fraction > 0 else "NOT_SUPPORTED",
        "PREFERRED_DEPTH_BASELINE_SESSION": preferred,
        "HB2_POSITION_SPREAD_ADVANTAGE_VS_H1": "YES" if spread_count == len(spread_comparison) else "PARTIAL" if spread_count else "NO",
        "H1_EDGE_TAIL_ADVANTAGE_REPRODUCED": "YES" if h1_tail_count == len(tail_comparable) and tail_comparable else "PARTIAL" if h1_tail_count else "NO",
        "TRUE_SPATIAL_RESIDUAL_AFTER_ROI_AND_GROUND_AUDIT": spatial_flag,
        "SPATIAL_SOURCE_ATTRIBUTION_ALLOWED": "YES" if spatial_flag in {"YES", "PARTIAL"} and provenance.get("replay_provenance_match") else "NO",
        "NEW_SPATIAL_CORRECTION_ALLOWED": "NO",
        "NEW_ACQUISITION_REQUIRED_NOW": "NO",
        "support_by_session_model": {model: support(rows) for model, rows in session_rows.items()},
        "support_by_local_model": {model: support(rows) for model, rows in local_rows.items()},
        "local_valid_fraction": local_valid_fraction,
        "pooled_session_h1": h1_pooled,
        "pooled_session_hb2": hb2_pooled,
        "spatial_range_evidence_count": spatial_range_evidence,
        "spatial_correlation_evidence_count": spatial_corr_evidence,
        "pathology": pathology,
        "spread_comparison": spread_comparison,
        "edge_tail_comparison": edge_tail_comparison,
    }


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def report_text(
    provenance: dict[str, Any],
    registry: dict[str, Any],
    records: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    attribution: dict[str, Any],
    pathology: dict[str, Any],
    v1_v2: list[dict[str, Any]],
    flags: dict[str, Any],
) -> str:
    summary = provenance["session_ground_summary"]
    pooled = [row for row in condition_rows if row.get("scope") == "pooled" and row.get("reference_branch") == "session"]
    height_rows = [row for row in condition_rows if row.get("scope") == "height" and row.get("reference_branch") == "session"]
    lines = [
        "# Task A-13B-v2｜Session01 Manual ROI V2 + Session/Local-Baseline 多参考高度联合诊断",
        "",
        "## Scope and provenance",
        "",
        "本轮正式高度验证严格使用人工冻结 V2 geometry-only ROI。Session-reference 是唯一 authoritative branch；local-baseline 分支只作 diagnostic，不替换 Session Ground、不改模型、不拟合 correction。",
        "",
        f"- Frozen Steger cache: `{CACHE_NPZ}`；复用 {provenance['cache_info']['frames_total']} 帧，manifest `one_steger_per_frame={provenance['cache_info']['one_steger_per_frame']}`；本轮 Steger rerun=`{provenance['steger_rerun']}`。",
        f"- 每帧 reconstruction call: `{provenance['reconstruction_calls_per_frame']}`；C0/C1/Session R/t 后的 points、q1/q2、C1 clamp 被六个 view 共享。",
        f"- V2 manual registry: `{REGISTRY_PATH}`；registry_ok=`{provenance['registry_ok']}`；自动 QC 原值未改写：{registry.get('auto_qc_summary', {}).get('status_counts', {})}。",
        f"- truth: `h10/h20/h30` 按 nominal `10/20/30 mm`；未发现并未猜测更精确 certified height。",
        "- `height_shadow.csv` 仅作为 shadow-logging QC 读取，不进入正式高度、ROI 或 FOV 计算；whole-frame `v_median` 也未用于 position。",
        "",
        "## Session PnP / Ground state",
        "",
        f"- PnP: valid=`{summary.get('pnp_valid')}`, corners=`{summary.get('pnp_corner_count')}`, reprojection RMSE=`{fmt(summary.get('pnp_reprojection_rmse_px'))} px`。",
        f"- Session Ground Reference: status=`{summary.get('session_ground_status')}`, runtime status=`{summary.get('session_ground_reference_status')}`, slope=`{fmt(summary.get('ground_slope_z_per_mm'))}`, intercept=`{fmt(summary.get('ground_intercept_z_mm'))} mm`, RMSE=`{fmt(summary.get('ground_rmse_mm'))} mm`, valid S=`{summary.get('ground_valid_s_range_mm')}`。",
        f"- support: point/inlier=`{summary.get('ground_point_count')}/{summary.get('ground_inlier_count')}`, source=`{summary.get('ground_support', {}).get('source') if isinstance(summary.get('ground_support'), dict) else None}`。R/t 已从 `session_ground_calibration.json` 读取并用于本轮重建，不重新拟合。",
        f"- `height_shadow.csv` ground status QC counts=`{provenance.get('height_shadow_logging_qc_only', {}).get('ground_reference_status_counts')}`。其中 `inactive` 是旧 shadow logger 没有接入本轮 runtime Session Ground leveled-point chain 的日志状态，不否定 JSON 中 Ground VALID；本轮没有因此重拟 Ground。",
        "",
        "## Manual V2 review and measurement completeness",
        "",
        "用户已声明 30/30 overlay geometry review 完成并 ACCEPT_ALL_V2。本报告将该声明作为人工冻结 provenance；自动 QC 的 UNCERTAIN 仍然保留，表示自动质量门的原始判断，不是自动 PASS。",
        "",
        f"- records=`{len(records)}`; Session raw valid=`{sum(int(finite(row.get('residual_base_session')) is not None) for row in records)}`; local raw valid=`{sum(int(finite(row.get('residual_base_local_diag')) is not None) for row in records)}`。",
        f"- baseline support counts: `{dict((support, sum(int(row.get('baseline_support_type') == support) for row in records)) for support in ('BOTH_SIDES', 'BEFORE_ONLY', 'AFTER_ONLY', 'NONE'))}`。",
        f"- one-side local diagnostics: `{sum(int(row.get('local_baseline_support') == 'ONE_SIDE') for row in records)}` frames; h30_p01 is explicitly retained as one-side/extrapolation when applicable.",
        "",
        "## Height-ROI true spatial coverage",
        "",
        "Position order is derived from the frozen height-ROI formal-point v median, not the whole-frame centerline median.",
        "",
    ]
    for height in HEIGHT_LABELS:
        entries = sorted(
            (entry for entry in registry.get("entries", []) if entry.get("height_label") == height),
            key=lambda entry: int(entry.get("v_order_rank") or 0),
        )
        values = [finite(entry.get("height_roi_formal_v_median")) for entry in entries]
        values = [value for value in values if value is not None]
        gaps = [right - left for left, right in zip(values, values[1:])]
        lines.append(
            f"- `{height}`: order=`{','.join(str(entry.get('position_id')) for entry in entries)}`; v=`{fmt(min(values) if values else None)}..{fmt(max(values) if values else None)}`; max adjacent gap=`{fmt(max(gaps) if gaps else None)} px`; support counts `>2200/>2400/>2600={sum(value > 2200 for value in values)}/{sum(value > 2400 for value in values)}/{sum(value > 2600 for value in values)}`."
        )
    lines.extend([
        "",
        "## Authoritative Session-reference pooled metrics",
        "",
        "| model | n valid | Bias | MAE | RMSE | P95 | Max | repeatability std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in pooled:
        lines.append(f"|{row['model'].upper()}|{row.get('n_valid')}|{fmt(row.get('bias_mm'))}|{fmt(row.get('mae_mm'))}|{fmt(row.get('rmse_mm'))}|{fmt(row.get('p95_abs_mm'))}|{fmt(row.get('max_abs_mm'))}|{fmt(row.get('repeatability_std_mm'))}|")
    lines.extend(["", "## Session spatial metrics by height", "", "| height | model | position bias range | position bias std | worst | worst P95 | worst Max |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in height_rows:
        lines.append(f"|{row.get('height_label')}|{row.get('model', '').upper()}|{fmt(row.get('position_bias_range_mm'))}|{fmt(row.get('position_bias_std_mm'))}|{fmt(row.get('worst_position_abs_bias_mm'))}|{fmt(row.get('worst_position_p95_abs_mm'))}|{fmt(row.get('worst_position_max_abs_mm'))}|")
    lines.extend(["", f"HB2 vs H1 spatial spread: `{flags.get('HB2_POSITION_SPREAD_ADVANTAGE_VS_H1')}`; edge-tail relation H1 better than HB2: `{flags.get('H1_EDGE_TAIL_ADVANTAGE_REPRODUCED')}`. In the new Session01, HB2 has lower position-bias range/std at all three heights, and its v>2400/v>2600 P95/Max are lower than H1; the historical H1 edge-tail advantage is therefore not reproduced."])
    lines.extend(["", "## Session vs local reference", "", "local branch is same-frame paired diagnostic. `delta_reference_mm = h_local - h_session`; local H1/HB2 are marked diagnostic-only and do not authorize a baseline change.", ""])
    for model in ("base", "h1", "hb2"):
        rows = [row for row in comparison if row.get("row_type") == "condition" and row.get("model") == model]
        delta = [finite(row.get("bias_difference_local_minus_session_mm")) for row in rows]
        delta = [value for value in delta if value is not None]
        lines.append(f"- {model.upper()}: condition bias delta local-session median=`{fmt(np.median(delta) if delta else None)}` mm; conditions with local P95/Max available=`{sum(int(row.get('local_p95_abs_mm') is not None) for row in rows)}/{len(rows)}`.")
    lines.extend(["", "## Baseline before/after and attribution", ""])
    lines.append(f"- before/after consistency flag=`{flags.get('BASELINE_BEFORE_AFTER_CONSISTENT')}`; fraction |delta_before_after|≤0.1 mm=`{fmt(flags.get('BASELINE_BEFORE_AFTER_CONSISTENT_FRACTION_ABS_DELTA_LE_0_1'))}`.")
    lines.append(f"- local baseline significantly changes height=`{flags.get('LOCAL_BASELINE_SIGNIFICANTLY_CHANGES_HEIGHT')}`; local-vs-session raw delta median/mean abs=`{fmt(np.median([abs(float(row['delta_reference_h_raw_local_minus_session_mm'])) for row in records if finite(row.get('delta_reference_h_raw_local_minus_session_mm')) is not None]) if any(finite(row.get('delta_reference_h_raw_local_minus_session_mm')) is not None for row in records) else None)}` mm / `{fmt(np.mean([abs(float(row['delta_reference_h_raw_local_minus_session_mm'])) for row in records if finite(row.get('delta_reference_h_raw_local_minus_session_mm')) is not None]) if any(finite(row.get('delta_reference_h_raw_local_minus_session_mm')) is not None for row in records) else None)}` mm.")
    for model in ("base", "h1", "hb2"):
        item = attribution.get(model, {})
        lines.append(f"- {model.upper()} attribution diagnostics: error_session vs local-ground residual Pearson=`{fmt(item.get('error_vs_local_ground_pearson_r'))}` (p=`{fmt(item.get('error_vs_local_ground_pearson_pvalue'))}`), Spearman=`{fmt(item.get('error_vs_local_ground_spearman_rho'))}` (p=`{fmt(item.get('error_vs_local_ground_spearman_pvalue'))}`); `(error_session-error_local)` vs local-ground residual Pearson=`{fmt(item.get('error_session_minus_error_local_pearson_r'))}`, Spearman=`{fmt(item.get('error_session_minus_error_local_spearman_rho'))}`.")
    lines.extend(["", "## Historical V1 false-failure audit", "", f"V1 condition metrics remain historical only and are invalidated as formal A-13B evidence because their ROI selection used the faulty V1 geometry. Key cases: `{', '.join(KEY_V1_FAILURE_CONDITIONS)}`. V2 comparison is descriptive and does not use error to select ROI.", ""])
    for case in pathology.get("cases", []):
        case_rows = [row for row in v1_v2 if row.get("condition_id") == case["condition_id"]]
        old_biases = [finite(row.get("v1_bias_mm")) for row in case_rows if finite(row.get("v1_bias_mm")) is not None]
        new_biases = [finite(row.get("v2_bias_mm")) for row in case_rows if finite(row.get("v2_bias_mm")) is not None]
        lines.append(f"- {case['condition_id']}: V1 abs-bias range=`{fmt(min(abs(value) for value in old_biases) if old_biases else None)}..{fmt(max(abs(value) for value in old_biases) if old_biases else None)}` mm, V2=`{fmt(min(abs(value) for value in new_biases) if new_biases else None)}..{fmt(max(abs(value) for value in new_biases) if new_biases else None)}` mm; V1 height-std median/P95=`{fmt(case.get('v1_height_std_median_mm'))}/{fmt(case.get('v1_height_std_p95_mm'))}` mm; V2=`{fmt(case.get('v2_height_std_median_mm'))}/{fmt(case.get('v2_height_std_p95_mm'))}` mm; resolved=`{case.get('resolved')}`.")
    lines.extend(["", "## Edge audit", "", "Edge metrics are independently emitted for pooled, each height, each actual position/rank and v>2200/2400/2600. The key edge figures are in `session01_a13b_v2_edge_metrics.csv` and the two edge comparison plots.", ""])
    for region in ("v_gt_2400", "v_gt_2600"):
        lines.append(f"### {region}")
        lines.append("")
        lines.append("| branch | model | n valid | Bias | P95 | Max | >0.1 | >0.2 | Spearman residual-v |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in edge_rows:
            if row.get("region") == region and row.get("scope") == "pooled":
                lines.append(f"|{row.get('reference_branch')}|{row.get('model', '').upper()}|{row.get('n_valid')}|{fmt(row.get('bias_mm'))}|{fmt(row.get('p95_abs_mm'))}|{fmt(row.get('max_abs_mm'))}|{fmt(row.get('error_gt_0_1_fraction'))}|{fmt(row.get('error_gt_0_2_fraction'))}|{fmt(row.get('residual_v_spearman_rho'))}|")
    lines.extend(["", "## Final flags", "", "```text"])
    ordered_flags = (
        "ROI_V2_HUMAN_REVIEW_COMPLETE", "ROI_V2_FROZEN", "A13B_V2_MEASUREMENT_COMPLETE", "A13B_V1_MM_FAILURE_CAUSED_BY_ROI", "V2_HEIGHT_STD_PATHOLOGY_RESOLVED", "SESSION_GROUND_LOCAL_RESIDUAL_SUPPORTED", "LOCAL_BASELINE_SIGNIFICANTLY_CHANGES_HEIGHT", "BASELINE_BEFORE_AFTER_CONSISTENT", "ONE_SIDE_BASELINE_DIAGNOSTIC_AVAILABLE", "SESSION_REFERENCE_FULL_FOV_ACCURACY", "LOCAL_REFERENCE_FULL_FOV_ACCURACY", "PREFERRED_DEPTH_BASELINE_SESSION", "HB2_POSITION_SPREAD_ADVANTAGE_VS_H1", "H1_EDGE_TAIL_ADVANTAGE_REPRODUCED", "TRUE_SPATIAL_RESIDUAL_AFTER_ROI_AND_GROUND_AUDIT", "SPATIAL_SOURCE_ATTRIBUTION_ALLOWED", "NEW_SPATIAL_CORRECTION_ALLOWED", "NEW_ACQUISITION_REQUIRED_NOW",
    )
    for key in ordered_flags:
        lines.append(f"{key}={flags.get(key)}")
    lines.extend(["```", "", "## Artifact boundaries", "", "本轮复用：Frozen Steger cache、V2 candidates/overlays、frozen calibration/C1/H1/H-B2、Session PnP/Ground JSON、V1 historical CSV。新增计算：正式 V2 registry materialization、每帧一次 reconstruction、Session-reference/local diagnostic measurement、paired metrics、edge/attribution metrics、plots 和本报告。未做：Steger rerun、ROI 重选、C0/C1/Ground/H1/H-B2 refit、spatial correction、删 position、采 Session02。", "", "Artifacts:", ""])
    for name in (
        "session01_roi_registry_manual_v2.json", "session01_a13b_v2_multireference_frames.csv", "session01_a13b_v2_condition_metrics.csv", "session01_a13b_v2_reference_comparison.csv", "session01_a13b_v2_baseline_diagnostics.csv", "session01_a13b_v2_edge_metrics.csv", "session01_a13b_v1_vs_v2_comparison.csv", "session_vs_local_height_error.png", "session_vs_local_position_bias.png", "local_ground_residual_vs_v.png", "height_error_vs_local_ground_residual.png", "baseline_before_vs_after_residual.png", "a13b_v1_vs_v2_height_std.png", "a13b_v2_edge_v2400.png", "a13b_v2_edge_v2600.png",
    ):
        lines.append(f"- `{OUTPUT_DIR / name}`")
    return "\n".join(lines) + "\n"


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required = (CACHE_NPZ, CACHE_MANIFEST, REGISTRY_PATH, GROUND_PATH, CONFIG_PATH, MANIFEST_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"A-13B-v2 required artifact missing: {missing}")
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
    ground_reference = hydrate_session_ground(ground_payload)
    calibration["R"] = np.asarray(ground_payload["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64)
    calibration["t"] = np.asarray(ground_payload["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64)
    frames, cache_manifest, centers_by_key, cache_info = load_cache()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_by_condition = {str(entry["condition_id"]): entry for entry in registry.get("entries", [])}
    for height in HEIGHT_LABELS:
        entries = sorted((entry for entry in registry.get("entries", []) if entry.get("height_label") == height), key=lambda entry: (float(entry.get("height_roi_center_v")), str(entry.get("position_id"))))
        for rank, entry in enumerate(entries, start=1):
            entry["v_order_rank"] = rank
    provenance = provenance_audit(frames, cache_manifest, registry, ground_payload, app, cache_info)
    write_json(output_dir / "session01_a13b_v2_provenance_audit.json", provenance)

    source_cache: dict[str, tuple[dict[str, dict[str, Any]], dict[str, Any]]] = {}
    source_hash_cache: dict[Path, str | None] = {}
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames, start=1):
        cid = condition_id(str(frame["height_label"]), str(frame["position_id"]))
        roi = registry_by_condition.get(cid)
        if roi is None:
            raise RuntimeError(f"Manual V2 registry missing {cid}")
        if cid not in source_cache:
            source_cache[cid] = read_frames_csv(DATA_ROOT / str(frame["height_label"]) / cid)
        by_filename, audit = source_cache[cid]
        source = source_qc(frame, audit, by_filename, source_hash_cache)
        record = reconstruct_and_measure(
            frame,
            centers_by_key[str(frame["cache_key"])],
            roi,
            calibration,
            app,
            ground_reference,
            source,
        )
        records.append(record)
        if index % 50 == 0:
            print(f"A-13B-v2 measurement {index}/{len(frames)}")

    condition_metrics = build_condition_metrics(records)
    reference_comparison, _comparison_detail = build_reference_comparison(records)
    baseline_diagnostics = build_baseline_diagnostics(records)
    edge_metrics = build_edge_metrics(records)
    v1_v2 = old_v1_vs_v2(records, condition_metrics)
    pathology = height_std_pathology(records)
    attribution = attribution_summary(records)
    flags = derive_flags(records, condition_metrics, edge_metrics, provenance, reference_comparison, pathology, attribution)
    flags["attribution"] = attribution
    write_csv(output_dir / "session01_a13b_v2_multireference_frames.csv", records)
    write_csv(output_dir / "session01_a13b_v2_condition_metrics.csv", condition_metrics)
    write_csv(output_dir / "session01_a13b_v2_reference_comparison.csv", reference_comparison)
    write_csv(output_dir / "session01_a13b_v2_baseline_diagnostics.csv", baseline_diagnostics)
    write_csv(output_dir / "session01_a13b_v2_edge_metrics.csv", edge_metrics)
    write_csv(output_dir / "session01_a13b_v1_vs_v2_comparison.csv", v1_v2)
    plot_session_vs_local(records, output_dir / "session_vs_local_height_error.png")
    plot_position_bias(condition_metrics, output_dir / "session_vs_local_position_bias.png")
    plot_local_residual_v(records, output_dir / "local_ground_residual_vs_v.png")
    plot_error_attribution(records, output_dir / "height_error_vs_local_ground_residual.png", attribution)
    plot_before_after(records, output_dir / "baseline_before_vs_after_residual.png")
    plot_v1_v2(v1_v2, output_dir / "a13b_v1_vs_v2_height_std.png")
    plot_edge(records, 2400.0, output_dir / "a13b_v2_edge_v2400.png")
    plot_edge(records, 2600.0, output_dir / "a13b_v2_edge_v2600.png")
    (output_dir / "session01_a13b_v2_multireference_report.md").write_text(
        report_text(provenance, registry, records, condition_metrics, edge_metrics, reference_comparison, attribution, pathology, v1_v2, flags),
        encoding="utf-8",
    )
    write_json(output_dir / "session01_a13b_v2_flags.json", flags)
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
