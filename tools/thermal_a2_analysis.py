#!/usr/bin/env python3
"""Thermal-A2 frozen reconstruction and Session/Local reference analysis.

Protocol invariants:

* the only ROI source is the exact user-confirmed frozen registry;
* every PNG is rerun through configured Frozen Steger, C0/C1 and saved Session R/t;
* no calibration/correction model is fitted or changed;
* Session and Local height references are evaluated independently;
* pre/post camera reconnect remain separate segments;
* height_shadow.csv is never read.

The per-recording ground fits in this file are diagnostics of observed drift,
not replacement calibration models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = ROOT / "laser_measurement_tool"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from correction.stage_a_height_scale import resolve_height_correction  # noqa: E402
from measurement.ground_reference import SessionGroundReference  # noqa: E402
from measurement.height_measure import measure_height_line  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402

import thermal_a2a_r2_human_roi_gui as roi_gui  # noqa: E402


EXPECTED_REGISTRY_SHA256 = (
    "770df9948fb049c2596623847d592f1a55d8fb0f7e9dca94c09dac8a149912dd"
)
OBJECT_IDS = ("upper", "middle", "lower")
OBJECT_META = {
    "upper": {"height_mm": 20.0, "position": "upper", "color": "#2563eb"},
    "middle": {"height_mm": 30.0, "position": "middle", "color": "#db2777"},
    "lower": {"height_mm": 10.0, "position": "lower", "color": "#059669"},
}
REFERENCES = ("session", "local")
ALGORITHMS = ("base", "h1", "hb2")
MIN_SUPPORT = 20
POWER_ON = datetime.fromisoformat("2026-08-27T09:50:00+08:00")
REFERENCE_COMPLETE = datetime.fromisoformat("2026-08-27T09:57:00+08:00")
FORMAL_START = datetime.fromisoformat("2026-08-27T10:00:00+08:00")
PAUSE_START = datetime.fromisoformat("2026-08-27T11:30:00+08:00")
PAUSE_END = datetime.fromisoformat("2026-08-27T11:45:00+08:00")
NO_RECORD_START = datetime.fromisoformat("2026-08-27T12:13:00+08:00")
RECONNECT = datetime.fromisoformat("2026-08-27T13:01:00+08:00")


class ThermalA2Error(RuntimeError):
    """Raised when a frozen protocol invariant is not satisfied."""


@dataclass(frozen=True, slots=True)
class ObjectRoi:
    object_id: str
    position: str
    nominal_height_mm: float
    baseline_before: tuple[int, int]
    height: tuple[int, int]
    baseline_after: tuple[int, int]


@dataclass(slots=True)
class GroundFit:
    slope: float
    intercept: float
    rmse: float
    p95: float
    point_count: int
    inlier_count: int
    s_min: float
    s_max: float
    s: np.ndarray
    z: np.ndarray
    inlier: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    input_dir = (
        TOOL_ROOT
        / "output_daheng_0811"
        / "online_recordings"
        / "0827上午热漂_2000"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=input_dir)
    parser.add_argument(
        "--a1-dir",
        type=Path,
        default=ROOT / "projects" / "daheng" / "analysis" / "thermal_a1_0827",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            ROOT
            / "projects"
            / "daheng"
            / "analysis"
            / "thermal_a2a_roi_v2_0827"
            / "thermal_roi_registry_v2_frozen.json"
        ),
    )
    parser.add_argument(
        "--measure-config",
        type=Path,
        default=TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml",
    )
    parser.add_argument(
        "--session-ground",
        type=Path,
        default=input_dir / "session_ground_calibration.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "projects" / "daheng" / "analysis" / "thermal_a2_0827",
    )
    parser.add_argument(
        "--expected-registry-sha256",
        default=EXPECTED_REGISTRY_SHA256,
        help="Exact user-confirmed registry identity required by this experiment.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=0,
        help="Development smoke-run only; 0 means the formal 29-recording run.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ThermalA2Error(f"Cannot load JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ThermalA2Error(f"JSON root must be an object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ThermalA2Error(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ThermalA2Error(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fields})


def inclusive_range(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ThermalA2Error(f"{label} must be a two-value list")
    lo, hi = (int(value[0]), int(value[1]))
    if lo < 0 or hi < lo:
        raise ThermalA2Error(f"Invalid {label}: {value}")
    return lo, hi


def resolve_registry_path(raw: Any, registry_path: Path) -> Path:
    candidate = Path(str(raw))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    local = registry_path.parent / candidate.name
    if local.exists():
        return local
    return candidate


def validate_registry(
    path: Path,
    expected_sha256: str,
    measure_config: Path,
    session_ground: Path,
) -> tuple[dict[str, Any], dict[str, ObjectRoi], dict[str, Any]]:
    path = path.resolve()
    actual_sha = sha256_file(path)
    if actual_sha.lower() != expected_sha256.strip().lower():
        raise ThermalA2Error(
            f"Frozen ROI registry SHA256 mismatch: {actual_sha} != {expected_sha256}"
        )
    payload = load_json(path)
    required = {
        "status": "FROZEN_USER_CONFIRMED",
        "valid": True,
        "frozen": True,
        "thermal_a2_roi_frozen": True,
        "human_reviewed": True,
        "manual_confirmed": True,
        "manual_decision": "ACCEPTED_BY_USER",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ThermalA2Error(
                f"Frozen registry gate failed: {key}={payload.get(key)!r}, "
                f"expected {expected!r}"
            )
    confirmation = payload.get("manual_confirmation", {})
    if (
        confirmation.get("gui_freeze_button_clicked") is not True
        or confirmation.get("typed_confirmation_token") != "FREEZE"
    ):
        raise ThermalA2Error("Registry lacks explicit user GUI Freeze confirmation")
    support_gate = payload.get("support_gate", {})
    if (
        support_gate.get("status") != "PASS"
        or support_gate.get("steger_support_checked") is not True
        or support_gate.get("frozen_reconstruction_support_checked") is not True
    ):
        raise ThermalA2Error("Frozen registry support gate is not PASS")

    referenced_hashes: list[dict[str, str]] = []
    for path_key, hash_key in (
        ("summary_csv", "summary_csv_sha256"),
        ("frame_csv", "frame_csv_sha256"),
        ("qc_json", "qc_json_sha256"),
    ):
        ref_path = resolve_registry_path(support_gate.get(path_key), path)
        expected = str(support_gate.get(hash_key, ""))
        actual = sha256_file(ref_path)
        if actual.lower() != expected.lower():
            raise ThermalA2Error(f"Support artifact hash mismatch: {ref_path}")
        referenced_hashes.append(
            {"kind": path_key, "path": str(ref_path), "sha256": actual}
        )

    for key, actual_path in (
        ("measure_config", measure_config),
        ("session_ground_calibration", session_ground),
    ):
        info = payload.get(key, {})
        expected = str(info.get("sha256", ""))
        actual = sha256_file(actual_path)
        if actual.lower() != expected.lower():
            raise ThermalA2Error(f"{key} hash mismatch: {actual_path}")
        referenced_hashes.append(
            {"kind": key, "path": str(actual_path), "sha256": actual}
        )

    invalid_info = payload.get("invalid_codex_attempt", {})
    invalid_path = resolve_registry_path(invalid_info.get("path"), path)
    invalid_payload = load_json(invalid_path)
    invalid_sha = sha256_file(invalid_path)
    if (
        invalid_payload.get("status") != "INVALID_ATTEMPT"
        or invalid_payload.get("valid") is not False
        or invalid_payload.get("invalid_attempt") is not True
        or invalid_info.get("formal_input") is not False
        or invalid_sha.lower() != str(invalid_info.get("sha256", "")).lower()
    ):
        raise ThermalA2Error("Old Codex invalid attempt was not safely excluded")
    referenced_hashes.append(
        {"kind": "invalid_attempt_excluded", "path": str(invalid_path), "sha256": invalid_sha}
    )

    objects = payload.get("objects")
    if not isinstance(objects, list) or len(objects) != 3:
        raise ThermalA2Error("Frozen registry must contain exactly three objects")
    rois: dict[str, ObjectRoi] = {}
    for item in objects:
        object_id = str(item.get("object_id"))
        if object_id not in OBJECT_META or item.get("selection_source") != "USER_GUI":
            raise ThermalA2Error(f"Invalid object provenance: {object_id}")
        if item.get("manual_confirmed") is not True or item.get("frozen") is not True:
            raise ThermalA2Error(f"Object is not user-frozen: {object_id}")
        roi = item.get("manual_roi", {})
        height_label = str(item.get("height_label", "")).replace(" ", "")
        expected_label = f"{int(OBJECT_META[object_id]['height_mm'])}mm"
        if height_label != expected_label:
            raise ThermalA2Error(
                f"Object mapping mismatch for {object_id}: {height_label} != {expected_label}"
            )
        rois[object_id] = ObjectRoi(
            object_id=object_id,
            position=str(OBJECT_META[object_id]["position"]),
            nominal_height_mm=float(OBJECT_META[object_id]["height_mm"]),
            baseline_before=inclusive_range(roi.get("baseline_before"), f"{object_id}.before"),
            height=inclusive_range(roi.get("height"), f"{object_id}.height"),
            baseline_after=inclusive_range(roi.get("baseline_after"), f"{object_id}.after"),
        )
        frozen = rois[object_id]
        ordered = (
            frozen.baseline_before[1] < frozen.height[0]
            and frozen.height[1] < frozen.baseline_after[0]
        )
        within_sensor = all(
            0 <= lo <= hi <= 2999
            for lo, hi in (
                frozen.baseline_before,
                frozen.height,
                frozen.baseline_after,
            )
        )
        if not ordered or not within_sensor:
            raise ThermalA2Error(f"Frozen ROI geometry is invalid: {object_id}")
    if set(rois) != set(OBJECT_IDS):
        raise ThermalA2Error("Registry object IDs must be upper/middle/lower")
    provenance = {
        "registry_path": str(path),
        "registry_sha256": actual_sha,
        "registry_status": payload["status"],
        "manual_reviewer": payload.get("manual_reviewer"),
        "manual_confirmation": confirmation,
        "support_gate": support_gate,
        "referenced_hashes": referenced_hashes,
        "invalid_attempt_excluded": True,
    }
    return payload, rois, provenance


def load_a1(a1_dir: Path, input_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    index_path = a1_dir / "thermal_a1_recording_index.csv"
    timeline_path = a1_dir / "thermal_a1_event_timeline.csv"
    qc_path = a1_dir / "thermal_a1_qc_summary.csv"
    rows = read_csv(index_path)
    timeline = read_csv(timeline_path)
    qc = read_csv(qc_path)
    if len(rows) != 29 or sum(int(row["frame_count"]) for row in rows) != 580:
        raise ThermalA2Error("A1 inventory must contain 29 recordings / 580 frames")
    if Counter(row["segment"] for row in rows) != {
        "pre_reconnect": 22,
        "post_reconnect": 7,
    }:
        raise ThermalA2Error("A1 reconnect segmentation mismatch")
    for row in rows:
        if (
            row.get("raw_recording_integrity") != "PASS"
            or row.get("frame_count_ok") != "true"
            or row.get("exposure_config_match") != "true"
            or row.get("gain_config_match") != "true"
            or row.get("pixel_format_config_match") != "true"
            or row.get("roi_config_match") != "true"
        ):
            raise ThermalA2Error(f"A1 formal gate failed: {row.get('recording_id')}")
        expected = input_dir / row["recording_id"]
        if expected.resolve() != Path(row["relative_path"]).resolve():
            raise ThermalA2Error(f"A1 path mismatch for {row['recording_id']}")
        first_time = datetime.fromisoformat(row["first_frame_time_local"])
        if row["segment"] == "pre_reconnect" and not first_time < RECONNECT:
            raise ThermalA2Error(f"A1 pre-reconnect time crosses boundary: {row['recording_id']}")
        if row["segment"] == "post_reconnect" and not first_time >= RECONNECT:
            raise ThermalA2Error(f"A1 post-reconnect time precedes boundary: {row['recording_id']}")
    first_post = next(row for row in rows if row["segment"] == "post_reconnect")
    first_post_delay_s = (
        datetime.fromisoformat(first_post["first_frame_time_local"]) - RECONNECT
    ).total_seconds()
    if first_post_delay_s < 0 or first_post_delay_s > 60:
        raise ThermalA2Error(
            f"First post recording is not adjacent to reconnect: {first_post_delay_s:.3f} s"
        )
    audit = {
        "index_path": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "timeline_path": str(timeline_path.resolve()),
        "timeline_sha256": sha256_file(timeline_path),
        "qc_path": str(qc_path.resolve()),
        "qc_sha256": sha256_file(qc_path),
        "recording_count": len(rows),
        "frame_count": sum(int(row["frame_count"]) for row in rows),
        "segments": dict(Counter(row["segment"] for row in rows)),
        "timeline_event_ids": sorted(
            {row["event_id"] for row in timeline if row.get("event_id")}
        ),
        "shadow_height_used": False,
        "first_post_recording": first_post["recording_id"],
        "first_post_delay_from_reconnect_s": first_post_delay_s,
    }
    return rows, audit


def hydrate_session_reference(payload: dict[str, Any]) -> SessionGroundReference:
    if payload.get("status") != "VALID" or payload.get("valid") is not True:
        raise ThermalA2Error("session_ground_calibration.json is not VALID")
    if payload.get("runtime", {}).get("ground_extrinsic_source") != "session":
        raise ThermalA2Error("Saved ground extrinsic source is not session")
    ground = payload.get("session_ground_reference", {})
    if ground.get("status") != "VALID":
        raise ThermalA2Error("Saved Session Ground Reference is not VALID")
    return SessionGroundReference(
        origin_xy=np.asarray(ground["origin_xy"], dtype=np.float64),
        direction_xy=np.asarray(ground["direction_xy"], dtype=np.float64),
        slope_z_per_mm=float(ground["slope_z_per_mm"]),
        intercept_z_mm=float(ground["intercept_z_mm"]),
        rmse_mm=float(ground["rmse_mm"]),
        valid_s_range_mm=tuple(float(v) for v in ground["valid_s_range_mm"]),
        status=str(ground["status"]),
        source=str(ground.get("fit_source", "session_laser_ground")),
        point_count=int(ground.get("point_count", 0)),
        inlier_count=int(ground.get("inlier_count", 0)),
        support_source=str(ground.get("support_source", ground.get("source", ""))),
        active_ground_extrinsic_source="session",
        ground_extrinsic_generation=int(
            ground.get("ground_extrinsic_generation", payload.get("runtime", {}).get("ground_extrinsic_generation", 0))
        ),
        frame_host_monotonic_ns=int(ground.get("frame_host_monotonic_ns", 0)),
        mask_inset_mm=float(ground.get("mask_inset_mm", 0.0)),
        support_metadata=dict(ground.get("support", {})),
    )


def load_chain(
    measure_config: Path, session_ground: Path
) -> tuple[Any, dict[str, Any], dict[str, Any], SessionGroundReference]:
    app = load_app_config(measure_config)
    if app.extraction_method != "steger":
        raise ThermalA2Error("Formal extraction method is not steger")
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_model,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        app.calibration.laser_ray_correction,
        ground_u_optional=True,
    )
    ground_payload = load_json(session_ground)
    calibration["R"] = np.asarray(
        ground_payload["session_extrinsic"]["R_camera_to_ground"], dtype=np.float64
    )
    calibration["t"] = np.asarray(
        ground_payload["session_extrinsic"]["t_camera_to_ground_mm"], dtype=np.float64
    )
    if calibration.get("laser_model") is None:
        raise ThermalA2Error("Frozen C0 did not load")
    if (
        app.reconstruction.enable_laser_ray_correction
        and calibration.get("laser_ray_correction") is None
    ):
        raise ThermalA2Error("Frozen C1 is enabled but did not load")
    return app, calibration, ground_payload, hydrate_session_reference(ground_payload)


def v_mask(pixels: np.ndarray, value_range: tuple[int, int]) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float64)
    lo, hi = value_range
    return (values[:, 1] >= lo) & (values[:, 1] <= hi)


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return math.nan
    mad = float(np.median(np.abs(values - np.median(values))))
    sigma = 1.4826 * mad
    if sigma <= np.finfo(np.float64).eps:
        sigma = float(np.std(values))
    return sigma


def fit_ground(s: np.ndarray, z: np.ndarray) -> GroundFit:
    s = np.asarray(s, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(s) & np.isfinite(z)
    s, z = s[valid], z[valid]
    if len(s) < 40 or np.ptp(s) <= np.finfo(np.float64).eps:
        raise ThermalA2Error(f"Insufficient ground fit support: {len(s)}")
    inlier = np.ones(len(s), dtype=bool)
    for _ in range(5):
        design = np.column_stack([s[inlier], np.ones(np.count_nonzero(inlier))])
        slope, intercept = np.linalg.lstsq(design, z[inlier], rcond=None)[0]
        residual = z - (slope * s + intercept)
        sigma = robust_sigma(residual[inlier])
        if not math.isfinite(sigma) or sigma <= np.finfo(np.float64).eps:
            break
        candidate = np.abs(residual - np.median(residual[inlier])) <= 2.5 * sigma
        if np.count_nonzero(candidate) < 40 or np.array_equal(candidate, inlier):
            break
        inlier = candidate
    design = np.column_stack([s[inlier], np.ones(np.count_nonzero(inlier))])
    slope, intercept = np.linalg.lstsq(design, z[inlier], rcond=None)[0]
    residual = z - (slope * s + intercept)
    residual_in = residual[inlier]
    return GroundFit(
        slope=float(slope),
        intercept=float(intercept),
        rmse=float(np.sqrt(np.mean(residual_in**2))),
        p95=float(np.percentile(np.abs(residual_in), 95)),
        point_count=len(s),
        inlier_count=int(np.count_nonzero(inlier)),
        s_min=float(np.min(s[inlier])),
        s_max=float(np.max(s[inlier])),
        s=s,
        z=z,
        inlier=inlier,
    )


def summarize(values: Iterable[Any]) -> dict[str, float | int | None]:
    array = np.asarray(
        [float(v) for v in values if finite(v) is not None], dtype=np.float64
    )
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def error_stats(values: Iterable[Any], nominal: float) -> dict[str, Any]:
    array = np.asarray(
        [float(v) for v in values if finite(v) is not None], dtype=np.float64
    )
    if not len(array):
        return {
            "valid_frame_count": 0,
            "mean_mm": None,
            "median_mm": None,
            "repeatability_std_mm": None,
            "bias_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_abs_error_mm": None,
            "observed_max_abs_error_mm": None,
        }
    error = array - nominal
    return {
        "valid_frame_count": len(array),
        "mean_mm": float(np.mean(array)),
        "median_mm": float(np.median(array)),
        "repeatability_std_mm": float(np.std(array)),
        "bias_mm": float(np.mean(error)),
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "p95_abs_error_mm": float(np.percentile(np.abs(error), 95)),
        "observed_max_abs_error_mm": float(np.max(np.abs(error))),
    }


def correction_values(
    height_base: float | None,
    reconstruction: Any,
    height_mask: np.ndarray,
    app: Any,
) -> dict[str, Any]:
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
    hb2_config = app.correction.hb2_height_correction
    if hb2_config is not None and len(q2_values):
        lower, upper = hb2_config.q2_domain
        q2_in_domain = bool(
            np.isfinite(q2_values).all()
            and np.all((q2_values >= lower) & (q2_values <= upper))
        )
    else:
        q2_in_domain = False
    h1 = resolve_height_correction(
        height_base,
        q1=q1,
        q2=q2,
        q2_in_domain=q2_in_domain,
        system=app.system,
        correction=app.correction,
        mode_override="h1",
    )
    hb2 = resolve_height_correction(
        height_base,
        q1=q1,
        q2=q2,
        q2_in_domain=q2_in_domain,
        system=app.system,
        correction=app.correction,
        mode_override="hb2",
    )
    return {
        "q1": q1,
        "q2": q2,
        "q2_in_domain": q2_in_domain,
        "height_h1_mm": h1.height_h1,
        "h1_valid": h1.h1_valid,
        "h1_status": h1.h1_status,
        "height_hb2_mm": hb2.height_hb2,
        "hb2_valid": hb2.active_height_valid,
        "hb2_status": hb2.hb2_q2_status,
    }


def empty_reference(prefix: str, status: str, error: str = "") -> dict[str, Any]:
    return {
        f"{prefix}_status": status,
        f"{prefix}_error": error,
        f"{prefix}_base_mean_mm": None,
        f"{prefix}_base_median_mm": None,
        f"{prefix}_base_within_line_std_mm": None,
        f"{prefix}_h1_mm": None,
        f"{prefix}_h1_valid": False,
        f"{prefix}_h1_status": "not_measured",
        f"{prefix}_hb2_mm": None,
        f"{prefix}_hb2_valid": False,
        f"{prefix}_hb2_status": "not_measured",
        f"{prefix}_q1": None,
        f"{prefix}_q2": None,
        f"{prefix}_q2_in_domain": False,
    }


def measure_reference(
    prefix: str,
    points: np.ndarray,
    pixels: np.ndarray,
    reconstruction: Any,
    roi: ObjectRoi,
    app: Any,
    *,
    valid_mask: np.ndarray | None,
) -> dict[str, Any]:
    before = v_mask(pixels, roi.baseline_before)
    height = v_mask(pixels, roi.height)
    after = v_mask(pixels, roi.baseline_after)
    if valid_mask is not None:
        before &= valid_mask
        height &= valid_mask
        after &= valid_mask
    counts = {
        f"{prefix}_baseline_before_count": int(np.count_nonzero(before)),
        f"{prefix}_height_count": int(np.count_nonzero(height)),
        f"{prefix}_baseline_after_count": int(np.count_nonzero(after)),
        f"{prefix}_both_sides": bool(
            np.count_nonzero(before) >= MIN_SUPPORT
            and np.count_nonzero(after) >= MIN_SUPPORT
        ),
    }
    if not counts[f"{prefix}_both_sides"]:
        return counts | empty_reference(prefix, "INVALID_BOTH_SIDES_SUPPORT")
    if np.count_nonzero(height) < MIN_SUPPORT:
        return counts | empty_reference(prefix, "INVALID_HEIGHT_SUPPORT")
    baseline = np.concatenate([points[before], points[after]], axis=0)
    try:
        measured = measure_height_line(
            baseline,
            points[height],
            app.measurement,
            ground_correction_mode=(
                "session_reference" if prefix == "session" else "auto"
            ),
        )
    except Exception as error:  # noqa: BLE001 - retain invalid frame in output
        return counts | empty_reference(
            prefix, "INVALID_MEASUREMENT", f"{type(error).__name__}: {error}"
        )
    correction = correction_values(
        float(measured.height_mean_mm), reconstruction, height, app
    )
    return counts | {
        f"{prefix}_status": "VALID",
        f"{prefix}_error": "",
        f"{prefix}_base_mean_mm": finite(measured.height_mean_mm),
        f"{prefix}_base_median_mm": finite(measured.height_median_mm),
        f"{prefix}_base_within_line_std_mm": finite(measured.height_std_mm),
        f"{prefix}_ground_at_height_mm": finite(measured.ground_baseline_zg_mm),
        f"{prefix}_ground_profile_slope_mm_per_mm": (
            finite(measured.ground_profile_fit.slope_z_per_mm)
            if measured.ground_profile_fit is not None
            else 0.0
        ),
        f"{prefix}_ground_profile_rmse_mm": (
            finite(measured.ground_profile_fit.rmse_mm)
            if measured.ground_profile_fit is not None
            else None
        ),
        f"{prefix}_baseline_inlier_count": int(measured.baseline_inlier_count),
        f"{prefix}_height_inlier_count": int(measured.height_inlier_count),
        f"{prefix}_h1_mm": correction["height_h1_mm"],
        f"{prefix}_h1_valid": correction["h1_valid"],
        f"{prefix}_h1_status": correction["h1_status"],
        f"{prefix}_hb2_mm": correction["height_hb2_mm"],
        f"{prefix}_hb2_valid": correction["hb2_valid"],
        f"{prefix}_hb2_status": correction["hb2_status"],
        f"{prefix}_q1": correction["q1"],
        f"{prefix}_q2": correction["q2"],
        f"{prefix}_q2_in_domain": correction["q2_in_domain"],
    }


def local_time_from_ns(host_timestamp_ns: int) -> datetime:
    return datetime.fromtimestamp(host_timestamp_ns / 1.0e9, tz=timezone.utc).astimezone(
        POWER_ON.tzinfo
    )


def elapsed_min(moment: datetime) -> float:
    return (moment - POWER_ON).total_seconds() / 60.0


def process_all(
    a1_rows: list[dict[str, str]],
    rois: dict[str, ObjectRoi],
    app: Any,
    calibration: dict[str, Any],
    session_reference: SessionGroundReference,
    max_recordings: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, GroundFit],
]:
    if max_recordings > 0:
        a1_rows = a1_rows[:max_recordings]
    frame_rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    ground_rows: list[dict[str, Any]] = []
    ground_fits: dict[str, GroundFit] = {}

    for recording_order, a1 in enumerate(a1_rows, start=1):
        recording_id = a1["recording_id"]
        recording_path = Path(a1["relative_path"])
        source_frames = roi_gui.load_source_frames(recording_path, app)
        if len(source_frames) != 20 or len(source_frames) != int(a1["frame_count"]):
            raise ThermalA2Error(
                f"Frame cardinality mismatch for {recording_id}: "
                f"loaded={len(source_frames)}, A1={a1['frame_count']}"
            )
        rec_ground_s: list[np.ndarray] = []
        rec_ground_z: list[np.ndarray] = []
        rec_frame_fits: list[GroundFit] = []
        rec_frame_rows: list[dict[str, Any]] = []
        for source in source_frames:
            moment = local_time_from_ns(source.host_timestamp_ns)
            common = {
                "recording_order": recording_order,
                "recording_id": recording_id,
                "segment": a1["segment"],
                "frame_index": source.row_index,
                "filename": source.filename,
                "camera_frame_number": source.camera_frame_number,
                "host_timestamp_ns": source.host_timestamp_ns,
                "frame_time_local": moment.isoformat(timespec="milliseconds"),
                "elapsed_from_power_min": elapsed_min(moment),
                "elapsed_from_reference_min": (
                    moment - REFERENCE_COMPLETE
                ).total_seconds()
                / 60.0,
                "exposure_us": source.exposure_us,
                "gain_db": source.gain_db,
                "pixel_format": source.pixel_format,
                "steger_point_count": len(source.centers_uv_full),
                "frame_status": "VALID",
                "frame_error": "",
            }
            try:
                reconstruction = reconstruct_uv_to_ground(
                    source.centers_uv_full, calibration, app.reconstruction
                )
                pixels = np.asarray(reconstruction.pixels_uv, dtype=np.float64)
                points_raw = np.asarray(reconstruction.points_ground, dtype=np.float64)
                points_session, session_valid = session_reference.apply_to_points(
                    points_raw
                )
                session_valid = np.asarray(session_valid, dtype=bool)
                if pixels.ndim != 2 or pixels.shape[1] != 2:
                    raise ThermalA2Error(f"Invalid reconstructed pixels shape: {pixels.shape}")
                if points_raw.ndim != 2 or points_raw.shape[1] != 3:
                    raise ThermalA2Error(f"Invalid reconstructed Ground shape: {points_raw.shape}")
                if points_session.ndim != 2 or points_session.shape[1] != 3:
                    raise ThermalA2Error(f"Invalid Session-leveled shape: {points_session.shape}")
                if not (
                    len(pixels)
                    == len(points_raw)
                    == len(points_session)
                    == len(session_valid)
                ):
                    raise ThermalA2Error("Reconstruction/Session point alignment mismatch")
            except Exception as error:  # noqa: BLE001 - keep all bad frames
                message = f"{type(error).__name__}: {error}"
                for object_id in OBJECT_IDS:
                    roi = rois[object_id]
                    frame_rows.append(
                        common
                        | {
                            "frame_status": "INVALID_RECONSTRUCTION",
                            "frame_error": message,
                            "object_id": object_id,
                            "position": roi.position,
                            "nominal_height_mm": roi.nominal_height_mm,
                            "reconstructed_point_count": 0,
                            "reconstruction_valid_ratio": 0.0,
                        }
                        | empty_reference("session", "INVALID_RECONSTRUCTION", message)
                        | empty_reference("local", "INVALID_RECONSTRUCTION", message)
                    )
                continue

            global_ground_mask = np.zeros(len(pixels), dtype=bool)
            for roi in rois.values():
                global_ground_mask |= v_mask(pixels, roi.baseline_before)
                global_ground_mask |= v_mask(pixels, roi.baseline_after)
            ground_points = points_raw[global_ground_mask]
            if len(ground_points) >= 40:
                ground_s = session_reference.project_s(ground_points[:, :2])
                frame_fit = fit_ground(ground_s, ground_points[:, 2])
                rec_ground_s.append(ground_s)
                rec_ground_z.append(ground_points[:, 2])
                rec_frame_fits.append(frame_fit)
            else:
                frame_fit = None

            c1_clamped = (
                int(np.count_nonzero(reconstruction.c1_clamped))
                if reconstruction.c1_clamped is not None
                else 0
            )
            for object_id in OBJECT_IDS:
                roi = rois[object_id]
                base_row = common | {
                    "object_id": object_id,
                    "position": roi.position,
                    "nominal_height_mm": roi.nominal_height_mm,
                    "roi_baseline_before": f"{roi.baseline_before[0]}-{roi.baseline_before[1]}",
                    "roi_height": f"{roi.height[0]}-{roi.height[1]}",
                    "roi_baseline_after": f"{roi.baseline_after[0]}-{roi.baseline_after[1]}",
                    "reconstructed_point_count": len(pixels),
                    "reconstruction_valid_ratio": (
                        len(pixels) / len(source.centers_uv_full)
                        if len(source.centers_uv_full)
                        else 0.0
                    ),
                    "session_ground_valid_count": int(np.count_nonzero(session_valid)),
                    "session_ground_valid_ratio": float(np.mean(session_valid)) if len(session_valid) else 0.0,
                    "c1_clamped_count": c1_clamped,
                    "frame_ground_point_count": len(ground_points),
                    "frame_ground_slope_mm_per_mm": frame_fit.slope if frame_fit else None,
                    "frame_ground_offset_b_mm": frame_fit.intercept if frame_fit else None,
                    "frame_ground_detrended_rmse_mm": frame_fit.rmse if frame_fit else None,
                    "frame_ground_detrended_p95_mm": frame_fit.p95 if frame_fit else None,
                }
                local_result = measure_reference(
                    "local",
                    points_raw,
                    pixels,
                    reconstruction,
                    roi,
                    app,
                    valid_mask=None,
                )
                session_result = measure_reference(
                    "session",
                    points_session,
                    pixels,
                    reconstruction,
                    roi,
                    app,
                    valid_mask=session_valid,
                )
                row = base_row | session_result | local_result
                frame_rows.append(row)
                rec_frame_rows.append(row)

        if not rec_ground_s:
            raise ThermalA2Error(f"No valid Ground support in {recording_id}")
        fit = fit_ground(np.concatenate(rec_ground_s), np.concatenate(rec_ground_z))
        ground_fits[recording_id] = fit
        frame_b = summarize(item.intercept for item in rec_frame_fits)
        frame_a = summarize(item.slope for item in rec_frame_fits)
        first_time = datetime.fromisoformat(a1["first_frame_time_local"])
        ground_row = {
            "recording_order": recording_order,
            "recording_id": recording_id,
            "segment": a1["segment"],
            "recording_time_local": a1["first_frame_time_local"],
            "elapsed_from_power_min": float(a1["elapsed_from_power_start_s"]) / 60.0,
            "elapsed_from_reference_min": float(a1["elapsed_from_reference_start_s"]) / 60.0,
            "ground_point_count": fit.point_count,
            "ground_inlier_count": fit.inlier_count,
            "ground_inlier_ratio": fit.inlier_count / fit.point_count,
            "s_min_mm": fit.s_min,
            "s_max_mm": fit.s_max,
            "s_span_mm": fit.s_max - fit.s_min,
            "slope_a_mm_per_mm": fit.slope,
            "offset_b_mm": fit.intercept,
            "detrended_rmse_mm": fit.rmse,
            "detrended_p95_mm": fit.p95,
            "frame_offset_b_std_mm": frame_b["std"],
            "frame_slope_a_std_mm_per_mm": frame_a["std"],
            "diagnostic_fit_only_not_calibration": True,
            "recording_ground_weighting": "pooled_points_across_20_frames_and_6_frozen_ground_intervals",
            "frame_equal_offset_b_mm": frame_b["mean"],
            "frame_equal_slope_a_mm_per_mm": frame_a["mean"],
            "pooled_minus_frame_equal_offset_b_mm": (
                fit.intercept - float(frame_b["mean"])
                if frame_b["mean"] is not None
                else None
            ),
            "pooled_minus_frame_equal_slope_a_mm_per_mm": (
                fit.slope - float(frame_a["mean"])
                if frame_a["mean"] is not None
                else None
            ),
        }
        ground_rows.append(ground_row)

        object_rows = defaultdict(list)
        for row in rec_frame_rows:
            object_rows[row["object_id"]].append(row)
        recording_row: dict[str, Any] = {
            "recording_order": recording_order,
            "recording_id": recording_id,
            "segment": a1["segment"],
            "recording_time_local": first_time.isoformat(timespec="milliseconds"),
            "elapsed_from_power_min": float(a1["elapsed_from_power_start_s"]) / 60.0,
            "elapsed_from_reference_min": float(a1["elapsed_from_reference_start_s"]) / 60.0,
            "frame_count": len(source_frames),
            "valid_reconstruction_frame_count": len(
                {
                    row["frame_index"]
                    for row in rec_frame_rows
                    if row["frame_status"] == "VALID"
                }
            ),
            "steger_point_count_min": min(len(frame.centers_uv_full) for frame in source_frames),
            "steger_point_count_mean": float(np.mean([len(frame.centers_uv_full) for frame in source_frames])),
            "ground_offset_b_mm": fit.intercept,
            "ground_slope_a_mm_per_mm": fit.slope,
            "ground_detrended_rmse_mm": fit.rmse,
            "ground_detrended_p95_mm": fit.p95,
            "ground_frame_offset_repeatability_std_mm": frame_b["std"],
        }
        for object_id in OBJECT_IDS:
            rows = object_rows[object_id]
            for reference in REFERENCES:
                values = [row.get(f"{reference}_base_mean_mm") for row in rows]
                stats = summarize(values)
                recording_row[f"{object_id}_{reference}_base_valid_frames"] = stats["count"]
                recording_row[f"{object_id}_{reference}_base_mean_mm"] = stats["mean"]
                recording_row[f"{object_id}_{reference}_repeatability_std_mm"] = stats["std"]
                recording_row[f"{object_id}_{reference}_support_pass_all"] = all(
                    row.get(f"{reference}_status") == "VALID" for row in rows
                )
        recording_rows.append(recording_row)
        print(
            f"[{recording_order:02d}/{len(a1_rows):02d}] {recording_id} "
            f"frames={len(source_frames)} ground_b={fit.intercept:.6f} mm",
            flush=True,
        )
    return frame_rows, recording_rows, ground_rows, ground_fits


def add_ground_profile_metrics(
    ground_rows: list[dict[str, Any]], ground_fits: dict[str, GroundFit]
) -> dict[str, dict[str, np.ndarray]]:
    first = ground_fits[ground_rows[0]["recording_id"]]
    common_min = max(fit.s_min for fit in ground_fits.values())
    common_max = min(fit.s_max for fit in ground_fits.values())
    if common_max <= common_min:
        raise ThermalA2Error("Ground recordings have no common S support")
    edges = np.linspace(common_min, common_max, 61)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def profile(fit: GroundFit) -> tuple[np.ndarray, np.ndarray]:
        raw = np.full(len(centers), np.nan)
        shape = np.full(len(centers), np.nan)
        residual = fit.z - (fit.slope * fit.s + fit.intercept)
        for index in range(len(centers)):
            mask = (
                fit.inlier
                & (fit.s >= edges[index])
                & (fit.s < edges[index + 1] if index < len(centers) - 1 else fit.s <= edges[index + 1])
            )
            if np.count_nonzero(mask) >= 5:
                raw[index] = float(np.median(fit.z[mask]))
                shape[index] = float(np.median(residual[mask]))
        return raw, shape

    first_raw, first_shape = profile(first)
    profiles: dict[str, dict[str, np.ndarray]] = {}
    first_b = ground_rows[0]["offset_b_mm"]
    first_a = ground_rows[0]["slope_a_mm_per_mm"]
    for row in ground_rows:
        fit = ground_fits[row["recording_id"]]
        raw, shape = profile(fit)
        eligible = np.isfinite(first_raw)
        shape_eligible = np.isfinite(first_shape)
        matched = np.isfinite(raw) & eligible
        shape_matched = np.isfinite(shape) & shape_eligible
        raw_delta = raw - first_raw
        shape_delta = shape - first_shape
        row["delta_offset_b_vs_first_mm"] = row["offset_b_mm"] - first_b
        row["delta_slope_a_vs_first_mm_per_mm"] = row["slope_a_mm_per_mm"] - first_a
        row["tilt_delta_across_common_span_mm"] = (
            row["delta_slope_a_vs_first_mm_per_mm"] * (common_max - common_min)
        )
        row["profile_common_s_min_mm"] = common_min
        row["profile_common_s_max_mm"] = common_max
        row["profile_matched_bin_count"] = int(np.count_nonzero(matched))
        row["profile_reference_supported_bin_count"] = int(np.count_nonzero(eligible))
        row["shape_reference_supported_bin_count"] = int(np.count_nonzero(shape_eligible))
        row["profile_matched_bin_fraction"] = (
            float(np.count_nonzero(matched) / np.count_nonzero(eligible))
            if np.any(eligible)
            else 0.0
        )
        row["shape_matched_bin_fraction"] = (
            float(np.count_nonzero(shape_matched) / np.count_nonzero(shape_eligible))
            if np.any(shape_eligible)
            else 0.0
        )
        row["profile_delta_mean_mm"] = (
            float(np.mean(raw_delta[matched])) if np.any(matched) else None
        )
        row["profile_delta_rmse_mm"] = (
            float(np.sqrt(np.mean(raw_delta[matched] ** 2))) if np.any(matched) else None
        )
        row["profile_delta_p95_mm"] = (
            float(np.percentile(np.abs(raw_delta[matched]), 95)) if np.any(matched) else None
        )
        shape_coverage_ok = row["shape_matched_bin_fraction"] >= 0.8
        row["shape_profile_coverage_ok"] = shape_coverage_ok
        row["shape_delta_rmse_mm"] = (
            float(np.sqrt(np.mean(shape_delta[shape_matched] ** 2)))
            if np.any(shape_matched) and shape_coverage_ok
            else None
        )
        row["shape_delta_p95_mm"] = (
            float(np.percentile(np.abs(shape_delta[shape_matched]), 95))
            if np.any(shape_matched) and shape_coverage_ok
            else None
        )
        profiles[row["recording_id"]] = {
            "s": centers,
            "raw_delta": raw_delta,
            "shape_delta": shape_delta,
        }
    return profiles


def build_height_summary(
    frame_rows: list[dict[str, Any]], recording_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(row["recording_id"], row["object_id"])].append(row)
    output: list[dict[str, Any]] = []
    recording_lookup = {row["recording_id"]: row for row in recording_rows}
    for recording_id, object_id in sorted(
        grouped,
        key=lambda key: (recording_lookup[key[0]]["recording_order"], OBJECT_IDS.index(key[1])),
    ):
        rows = grouped[(recording_id, object_id)]
        nominal = float(OBJECT_META[object_id]["height_mm"])
        recording = recording_lookup[recording_id]
        for reference in REFERENCES:
            for algorithm in ALGORITHMS:
                field = (
                    f"{reference}_base_mean_mm"
                    if algorithm == "base"
                    else f"{reference}_{algorithm}_mm"
                )
                stats = error_stats([row.get(field) for row in rows], nominal)
                statuses = Counter(
                    str(
                        row.get(f"{reference}_status")
                        if algorithm == "base"
                        else row.get(f"{reference}_{algorithm}_status")
                    )
                    for row in rows
                )
                output.append(
                    {
                        "recording_order": recording["recording_order"],
                        "recording_id": recording_id,
                        "segment": recording["segment"],
                        "recording_time_local": recording["recording_time_local"],
                        "elapsed_from_power_min": recording["elapsed_from_power_min"],
                        "elapsed_from_reference_min": recording["elapsed_from_reference_min"],
                        "object_id": object_id,
                        "position": OBJECT_META[object_id]["position"],
                        "nominal_height_mm": nominal,
                        "reference": reference,
                        "algorithm": algorithm,
                        **stats,
                        "status_counts": ";".join(
                            f"{key}:{value}" for key, value in sorted(statuses.items())
                        ),
                    }
                )
    first_means: dict[tuple[str, str, str], float] = {}
    segment_first: dict[tuple[str, str, str, str], float] = {}
    for row in output:
        mean = finite(row["mean_mm"])
        key = (row["object_id"], row["reference"], row["algorithm"])
        segment_key = (row["segment"],) + key
        if mean is not None and key not in first_means:
            first_means[key] = mean
        if mean is not None and segment_key not in segment_first:
            segment_first[segment_key] = mean
        row["delta_h_vs_first_recording_mm"] = (
            mean - first_means[key] if mean is not None and key in first_means else None
        )
        row["delta_h_vs_segment_first_mm"] = (
            mean - segment_first[segment_key]
            if mean is not None and segment_key in segment_first
            else None
        )
    all_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        all_groups[(row["object_id"], row["reference"], row["algorithm"])].append(row)
    for rows in all_groups.values():
        all_values = [finite(row["mean_mm"]) for row in rows]
        valid = [value for value in all_values if value is not None]
        for segment in ("pre_reconnect", "post_reconnect"):
            segment_values = [
                float(row["mean_mm"])
                for row in rows
                if row["segment"] == segment and finite(row["mean_mm"]) is not None
            ]
            segment_range = max(segment_values) - min(segment_values) if segment_values else None
            for row in rows:
                row[f"thermal_range_{segment}_mm"] = segment_range
        full_range = max(valid) - min(valid) if valid else None
        for row in rows:
            row["observed_thermal_range_all_recordings_mm"] = full_range
    return output


def pearson(x: Iterable[Any], y: Iterable[Any]) -> float | None:
    pairs = [(finite(a), finite(b)) for a, b in zip(x, y)]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xa = np.asarray([a for a, _ in pairs], dtype=np.float64)
    ya = np.asarray([b for _, b in pairs], dtype=np.float64)
    if np.std(xa) <= 0 or np.std(ya) <= 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def reconnect_audit(
    ground_rows: list[dict[str, Any]], height_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    ground_series = [
        {
            "recording_id": row["recording_id"],
            "segment": row["segment"],
            "x": row["elapsed_from_power_min"],
            "value": row["offset_b_mm"],
            "repeatability": row["frame_offset_b_std_mm"],
        }
        for row in ground_rows
    ]
    series.append({"metric": "ground_offset_b", "object_id": "ground", "reference": "session_coordinate", "algorithm": "base", "rows": ground_series})
    for object_id in OBJECT_IDS:
        for reference in REFERENCES:
            for algorithm in ALGORITHMS:
                rows = [
                    row
                    for row in height_rows
                    if row["object_id"] == object_id
                    and row["reference"] == reference
                    and row["algorithm"] == algorithm
                ]
                series.append(
                    {
                        "metric": "height",
                        "object_id": object_id,
                        "reference": reference,
                        "algorithm": algorithm,
                        "rows": [
                            {
                                "recording_id": row["recording_id"],
                                "segment": row["segment"],
                                "x": row["elapsed_from_power_min"],
                                "value": row["mean_mm"],
                                "repeatability": row["repeatability_std_mm"],
                            }
                            for row in rows
                        ],
                    }
                )
    output: list[dict[str, Any]] = []
    for item in series:
        pre = [row for row in item["rows"] if row["segment"] == "pre_reconnect" and finite(row["value"]) is not None]
        post = [row for row in item["rows"] if row["segment"] == "post_reconnect" and finite(row["value"]) is not None]
        if len(pre) < 5 or not post:
            continue
        stable_pre = pre[-5:]
        x = np.asarray([row["x"] for row in stable_pre], dtype=np.float64)
        y = np.asarray([row["value"] for row in stable_pre], dtype=np.float64)
        design = np.column_stack([x, np.ones(len(x))])
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - (slope * x + intercept)
        trend_rmse = float(np.sqrt(np.mean(residual**2)))
        first_post = post[0]
        post_delay_from_reconnect_min = first_post["x"] - elapsed_min(RECONNECT)
        if post_delay_from_reconnect_min < 0 or post_delay_from_reconnect_min > 1:
            raise ThermalA2Error(
                f"Reconnect audit post-first point is not adjacent: "
                f"{post_delay_from_reconnect_min:.3f} min"
            )
        predicted = float(slope * first_post["x"] + intercept)
        observed = float(first_post["value"])
        step = observed - predicted
        repeat = [
            finite(row["repeatability"])
            for row in stable_pre + [first_post]
            if finite(row["repeatability"]) is not None
        ]
        repeat_sigma = float(np.median(repeat)) if repeat else 0.0
        threshold = max(0.03, 3.0 * math.sqrt(trend_rmse**2 + repeat_sigma**2))
        ratio = abs(step) / threshold if threshold > 0 else math.inf
        detected = "YES" if ratio >= 1.0 else "PARTIAL" if ratio >= 0.5 else "NO"
        output.append(
            {
                "metric": item["metric"],
                "object_id": item["object_id"],
                "position": OBJECT_META.get(item["object_id"], {}).get("position", "ground"),
                "reference": item["reference"],
                "algorithm": item["algorithm"],
                "pre_trend_recording_count": len(stable_pre),
                "pre_trend_first_recording": stable_pre[0]["recording_id"],
                "pre_trend_last_recording": stable_pre[-1]["recording_id"],
                "pre_trend_slope_mm_per_min": float(slope),
                "pre_trend_rmse_mm": trend_rmse,
                "post_first_recording": first_post["recording_id"],
                "post_first_elapsed_from_power_min": first_post["x"],
                "post_first_delay_from_reconnect_min": post_delay_from_reconnect_min,
                "predicted_post_first_mm": predicted,
                "observed_post_first_mm": observed,
                "extra_step_mm": step,
                "detection_threshold_mm": threshold,
                "step_to_threshold_ratio": ratio,
                "extra_step_detected": detected,
                "boundary_policy": "pre-only trend prediction; no cross-boundary continuous fit",
            }
        )
    return output


def derive_conclusions(
    frame_rows: list[dict[str, Any]],
    recording_rows: list[dict[str, Any]],
    ground_rows: list[dict[str, Any]],
    height_rows: list[dict[str, Any]],
    reconnect_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = [row for row in height_rows if row["algorithm"] == "base"]

    session_ranges: dict[str, float] = {}
    local_ranges: dict[str, float] = {}
    paired_counts: dict[str, int] = {}
    per_object_suppression: dict[str, float] = {}
    for object_id in OBJECT_IDS:
        session_by_recording = {
            row["recording_id"]: finite(row["mean_mm"])
            for row in base
            if row["object_id"] == object_id and row["reference"] == "session"
        }
        local_by_recording = {
            row["recording_id"]: finite(row["mean_mm"])
            for row in base
            if row["object_id"] == object_id and row["reference"] == "local"
        }
        common = [
            recording_id
            for recording_id in session_by_recording.keys() & local_by_recording.keys()
            if session_by_recording[recording_id] is not None
            and local_by_recording[recording_id] is not None
        ]
        if not common:
            raise ThermalA2Error(f"No paired Session/Local recordings for {object_id}")
        paired_counts[object_id] = len(common)
        session_values = [float(session_by_recording[key]) for key in common]
        local_values = [float(local_by_recording[key]) for key in common]
        session_ranges[object_id] = max(session_values) - min(session_values)
        local_ranges[object_id] = max(local_values) - min(local_values)
        per_object_suppression[object_id] = (
            1.0 - local_ranges[object_id] / session_ranges[object_id]
            if session_ranges[object_id] > 0
            else math.nan
        )
    session_overall = max(session_ranges.values())
    local_overall = max(local_ranges.values())
    suppression = (
        1.0 - local_overall / session_overall if session_overall > 0 else math.nan
    )
    improved_count = sum(local_ranges[obj] < session_ranges[obj] for obj in OBJECT_IDS)
    if suppression >= 0.5 and improved_count >= 2:
        suppression_status = "YES"
    elif suppression >= 0.1 and improved_count >= 1:
        suppression_status = "PARTIAL"
    else:
        suppression_status = "NO"

    offset_values = np.asarray([row["delta_offset_b_vs_first_mm"] for row in ground_rows])
    tilt_values = np.asarray([row["tilt_delta_across_common_span_mm"] for row in ground_rows])
    shape_values = np.asarray([row["shape_delta_p95_mm"] for row in ground_rows], dtype=np.float64)
    ground_repeat = np.asarray(
        [finite(row["frame_offset_b_std_mm"]) or 0.0 for row in ground_rows]
    )
    ground_threshold = max(0.03, 3.0 * float(np.median(ground_repeat)))
    offset_component = float(np.ptp(offset_values))
    tilt_component = float(np.ptp(tilt_values))
    shape_component = float(np.nanmax(shape_values))
    significant = {
        "OFFSET": offset_component >= ground_threshold,
        "TILT": tilt_component >= ground_threshold,
        "SHAPE": shape_component >= ground_threshold,
    }
    active_modes = [key for key, value in significant.items() if value]
    if len(active_modes) > 1:
        ground_mode = "MIXED"
    elif active_modes:
        ground_mode = active_modes[0]
    else:
        components = {"OFFSET": offset_component, "TILT": tilt_component, "SHAPE": shape_component}
        ground_mode = max(components, key=components.get)
    ground_present = "YES" if active_modes else "PARTIAL" if max(offset_component, tilt_component, shape_component) >= 0.5 * ground_threshold else "NO"

    differential = max(local_ranges.values()) - min(local_ranges.values())
    typical_repeat = np.median(
        [
            float(row["repeatability_std_mm"])
            for row in base
            if row["reference"] == "local" and finite(row["repeatability_std_mm"]) is not None
        ]
    )
    differential_threshold = max(0.03, 3.0 * float(typical_repeat))
    confounded_status = "PARTIAL" if differential >= differential_threshold else "NO"

    reconnect_base = [
        row
        for row in reconnect_rows
        if row["algorithm"] == "base"
        and (row["metric"] == "ground_offset_b" or row["reference"] in REFERENCES)
    ]
    reconnect_yes = sum(row["extra_step_detected"] == "YES" for row in reconnect_base)
    reconnect_partial = sum(row["extra_step_detected"] == "PARTIAL" for row in reconnect_base)
    reconnect_status = "YES" if reconnect_yes >= 2 else "PARTIAL" if reconnect_yes or reconnect_partial else "NO"

    invalid_frames = len(
        {
            (row["recording_id"], row["frame_index"])
            for row in frame_rows
            if row["frame_status"] != "VALID"
        }
    )
    support_failures = sum(
        row.get("session_status") != "VALID" or row.get("local_status") != "VALID"
        for row in frame_rows
    )
    counts = [row["steger_point_count_mean"] for row in recording_rows]
    extraction_range_ratio = (max(counts) - min(counts)) / np.median(counts)
    local_drift = []
    extraction = []
    for recording in recording_rows:
        local_values = [
            row["delta_h_vs_first_recording_mm"]
            for row in base
            if row["recording_id"] == recording["recording_id"]
            and row["reference"] == "local"
        ]
        finite_values = [float(v) for v in local_values if finite(v) is not None]
        if finite_values:
            local_drift.append(max(abs(v) for v in finite_values))
            extraction.append(recording["steger_point_count_mean"])
    drift_extraction_r = pearson(local_drift, extraction)
    extraction_degradation = bool(
        invalid_frames
        or support_failures
        or extraction_range_ratio >= 0.05
        or (drift_extraction_r is not None and abs(drift_extraction_r) >= 0.7)
    )

    # A single cold-start with a reconnect cannot establish a general steady
    # state.  Detect an observed pre-reconnect plateau, but cap the conclusion
    # at PARTIAL and do not force a formal warmup time.
    pre_ground = [row for row in ground_rows if row["segment"] == "pre_reconnect"]
    pre_height = {
        obj: [
            row
            for row in base
            if row["segment"] == "pre_reconnect"
            and row["object_id"] == obj
            and row["reference"] == "local"
        ]
        for obj in OBJECT_IDS
    }
    plateau_found = False
    plateau_candidate_min: float | None = None
    for index in range(max(0, len(pre_ground) - 10), len(pre_ground) - 3):
        ground_tail = pre_ground[index:]
        if ground_tail[-1]["elapsed_from_power_min"] - ground_tail[0]["elapsed_from_power_min"] < 30:
            continue
        if np.ptp([row["offset_b_mm"] for row in ground_tail]) > 2 * ground_threshold:
            continue
        object_ok = True
        for obj in OBJECT_IDS:
            lookup = {row["recording_id"]: row for row in pre_height[obj]}
            values = [
                lookup[row["recording_id"]]["mean_mm"]
                for row in ground_tail
                if row["recording_id"] in lookup and finite(lookup[row["recording_id"]]["mean_mm"]) is not None
            ]
            repeat = [
                lookup[row["recording_id"]]["repeatability_std_mm"]
                for row in ground_tail
                if row["recording_id"] in lookup and finite(lookup[row["recording_id"]]["repeatability_std_mm"]) is not None
            ]
            band = max(0.03, 3.0 * float(np.median(repeat)) if repeat else 0.03)
            if len(values) < 4 or np.ptp(values) > 2 * band:
                object_ok = False
                break
        if object_ok:
            plateau_found = True
            plateau_candidate_min = ground_tail[0]["elapsed_from_power_min"]
            break
    steady_status = "PARTIAL" if plateau_found else "NO"

    return {
        "GROUND_THERMAL_DRIFT_PRESENT": ground_present,
        "GROUND_DRIFT_MODE": ground_mode,
        "SESSION_REFERENCE_THERMAL_DRIFT": session_overall,
        "LOCAL_REFERENCE_THERMAL_DRIFT": local_overall,
        "LOCAL_REFERENCE_SUPPRESSION_RATIO": suppression,
        "LOCAL_REFERENCE_SUPPRESSES_THERMAL_DRIFT": suppression_status,
        "HEIGHT_DEPENDENT_THERMAL_DRIFT": confounded_status,
        "POSITION_DEPENDENT_THERMAL_DRIFT": confounded_status,
        "RECONNECT_EXTRA_STEP_DETECTED": reconnect_status,
        "THERMAL_STEADY_STATE_REACHED": steady_status,
        "ESTIMATED_WARMUP_TIME_MIN": None,
        "observed_pre_reconnect_plateau_candidate_min": plateau_candidate_min,
        "height_position_confounding": True,
        "session_ranges_mm": session_ranges,
        "local_ranges_mm": local_ranges,
        "paired_recording_counts": paired_counts,
        "per_object_suppression_ratio": per_object_suppression,
        "overall_range_definition": "max object range on paired valid Session/Local recordings",
        "ground_component_ranges_mm": {
            "offset": offset_component,
            "tilt_across_common_span": tilt_component,
            "shape_p95_max": shape_component,
            "significance_threshold": ground_threshold,
        },
        "quality": {
            "invalid_reconstruction_frames": invalid_frames,
            "reference_support_failure_rows": support_failures,
            "steger_count_range_ratio": extraction_range_ratio,
            "local_drift_vs_steger_count_pearson_r": drift_extraction_r,
            "extraction_degradation_detected": extraction_degradation,
        },
    }


def plot_events(ax: Any) -> None:
    ax.axvline(elapsed_min(REFERENCE_COMPLETE), color="#64748b", ls=":", lw=1, label="Session reference")
    ax.axvline(elapsed_min(RECONNECT), color="#b91c1c", ls="--", lw=1.2, label="Reconnect")
    ax.axvspan(elapsed_min(PAUSE_START), elapsed_min(PAUSE_END), color="#f59e0b", alpha=0.08)
    ax.axvspan(elapsed_min(NO_RECORD_START), elapsed_min(RECONNECT), color="#64748b", alpha=0.08)


def segmented_plot(ax: Any, rows: list[dict[str, Any]], y_key: str, **kwargs: Any) -> None:
    for segment, marker in (("pre_reconnect", "o"), ("post_reconnect", "s")):
        selected = [row for row in rows if row["segment"] == segment and finite(row.get(y_key)) is not None]
        ax.plot(
            [row["elapsed_from_power_min"] for row in selected],
            [row[y_key] for row in selected],
            marker=marker,
            ms=4,
            lw=1.2,
            label=segment,
            **kwargs,
        )


def save_plots(
    output_dir: Path,
    ground_rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, np.ndarray]],
    height_rows: list[dict[str, Any]],
    recording_rows: list[dict[str, Any]],
    reconnect_rows: list[dict[str, Any]],
) -> None:
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.22, "font.size": 9})

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    segmented_plot(axes[0], ground_rows, "delta_offset_b_vs_first_mm")
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_ylabel("Δ Ground offset b (mm)")
    segmented_plot(axes[1], ground_rows, "tilt_delta_across_common_span_mm")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Δ tilt across support (mm)")
    axes[1].set_xlabel("Elapsed from power-on (min)")
    for ax in axes:
        plot_events(ax)
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a2_ground_offset_slope.png", bbox_inches="tight")
    plt.close(fig)

    selected_ids = [ground_rows[0]["recording_id"], ground_rows[-8]["recording_id"], ground_rows[-7]["recording_id"], ground_rows[-1]["recording_id"]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for recording_id in selected_ids:
        item = profiles[recording_id]
        axes[0].plot(item["s"], item["raw_delta"], label=recording_id.replace("recording_20260827_", ""))
        axes[1].plot(item["s"], item["shape_delta"], label=recording_id.replace("recording_20260827_", ""))
    axes[0].set_ylabel("Ground profile ΔZ vs first (mm)")
    axes[1].set_ylabel("Detrended shape Δr (mm)")
    axes[1].set_xlabel("Frozen Session coordinate S (mm)")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a2_ground_profiles_over_time.png", bbox_inches="tight")
    plt.close(fig)

    base = [row for row in height_rows if row["algorithm"] == "base"]
    for reference, filename in (("session", "thermal_a2_height_session.png"), ("local", "thermal_a2_height_local.png")):
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for ax, object_id in zip(axes, OBJECT_IDS):
            rows = [row for row in base if row["object_id"] == object_id and row["reference"] == reference]
            for segment, marker in (("pre_reconnect", "o"), ("post_reconnect", "s")):
                selected = [row for row in rows if row["segment"] == segment and finite(row["delta_h_vs_first_recording_mm"]) is not None]
                ax.errorbar(
                    [row["elapsed_from_power_min"] for row in selected],
                    [row["delta_h_vs_first_recording_mm"] for row in selected],
                    yerr=[row["repeatability_std_mm"] for row in selected],
                    marker=marker, ms=4, lw=1.1, capsize=2, label=segment,
                    color=OBJECT_META[object_id]["color"],
                )
            ax.axhline(0, color="black", lw=0.7)
            ax.set_ylabel(f"{object_id}\nΔh (mm)")
            plot_events(ax)
        axes[-1].set_xlabel("Elapsed from power-on (min)")
        axes[0].legend(ncol=3, fontsize=8)
        fig.suptitle(f"Base height drift — {reference.capitalize()} reference")
        fig.tight_layout()
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(3)
    session_ranges = []
    local_ranges = []
    for object_id in OBJECT_IDS:
        rows_s = [row for row in base if row["object_id"] == object_id and row["reference"] == "session"]
        rows_l = [row for row in base if row["object_id"] == object_id and row["reference"] == "local"]
        session_ranges.append(rows_s[0]["observed_thermal_range_all_recordings_mm"])
        local_ranges.append(rows_l[0]["observed_thermal_range_all_recordings_mm"])
    axes[0].bar(x - 0.18, session_ranges, 0.36, label="Session")
    axes[0].bar(x + 0.18, local_ranges, 0.36, label="Local")
    axes[0].set_xticks(x, [f"{obj}\n{int(OBJECT_META[obj]['height_mm'])} mm" for obj in OBJECT_IDS])
    axes[0].set_ylabel("Recording-mean thermal range (mm)")
    axes[0].legend()
    axes[1].bar(x, [1 - l / s if s else np.nan for s, l in zip(session_ranges, local_ranges)], color=[OBJECT_META[obj]["color"] for obj in OBJECT_IDS])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x, OBJECT_IDS)
    axes[1].set_ylabel("Suppression ratio = 1 - Local/Session")
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a2_session_vs_local.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for object_id in OBJECT_IDS:
        rows = [row for row in base if row["object_id"] == object_id and row["reference"] == "local"]
        for segment, marker in (("pre_reconnect", "o"), ("post_reconnect", "s")):
            selected = [row for row in rows if row["segment"] == segment]
            axes[0].plot(
                [row["elapsed_from_power_min"] for row in selected],
                [row["repeatability_std_mm"] for row in selected],
                marker=marker, ms=4, lw=1, color=OBJECT_META[object_id]["color"],
                label=f"{object_id}-{segment}" if segment == "pre_reconnect" else None,
            )
    for segment, marker in (("pre_reconnect", "o"), ("post_reconnect", "s")):
        selected = [row for row in recording_rows if row["segment"] == segment]
        axes[1].plot(
            [row["elapsed_from_power_min"] for row in selected],
            [row["steger_point_count_mean"] for row in selected],
            marker=marker, ms=4, lw=1.1, label=segment,
        )
    axes[0].set_ylabel("20-frame Local repeatability std (mm)")
    axes[1].set_ylabel("Mean Steger point count")
    axes[1].set_xlabel("Elapsed from power-on (min)")
    for ax in axes:
        plot_events(ax)
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a2_repeatability.png", bbox_inches="tight")
    plt.close(fig)

    base_reconnect = [
        row
        for row in reconnect_rows
        if row["algorithm"] == "base"
        and (row["metric"] == "ground_offset_b" or row["reference"] in REFERENCES)
    ]
    labels = [
        "Ground" if row["metric"] == "ground_offset_b" else f"{row['object_id']}-{row['reference']}"
        for row in base_reconnect
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(base_reconnect))
    colors = ["#64748b" if row["metric"] == "ground_offset_b" else OBJECT_META[row["object_id"]]["color"] for row in base_reconnect]
    ax.bar(x, [row["extra_step_mm"] for row in base_reconnect], color=colors)
    ax.errorbar(x, np.zeros(len(x)), yerr=[row["detection_threshold_mm"] for row in base_reconnect], fmt="none", ecolor="black", capsize=3, label="± detection threshold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Post first − pre-trend prediction (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a2_reconnect.png", bbox_inches="tight")
    plt.close(fig)


def performance_rows(
    frame_rows: list[dict[str, Any]], height_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for object_id in OBJECT_IDS:
        local_base = [
            row
            for row in height_rows
            if row["object_id"] == object_id
            and row["reference"] == "local"
            and row["algorithm"] == "base"
        ]
        session_base = [
            row
            for row in height_rows
            if row["object_id"] == object_id
            and row["reference"] == "session"
            and row["algorithm"] == "base"
        ]
        local_frames = [
            row["local_base_mean_mm"]
            for row in frame_rows
            if row["object_id"] == object_id and finite(row.get("local_base_mean_mm")) is not None
        ]
        nominal = float(OBJECT_META[object_id]["height_mm"])
        errors = np.abs(np.asarray(local_frames) - nominal)
        session_map = {
            row["recording_id"]: finite(row["mean_mm"]) for row in session_base
        }
        local_map = {
            row["recording_id"]: finite(row["mean_mm"]) for row in local_base
        }
        common = [
            key
            for key in session_map.keys() & local_map.keys()
            if session_map[key] is not None and local_map[key] is not None
        ]
        if not common:
            raise ThermalA2Error(f"No paired performance rows for {object_id}")
        session_paired = [float(session_map[key]) for key in common]
        local_paired = [float(local_map[key]) for key in common]
        output.append(
            {
                "height": f"{int(nominal)} mm",
                "position": OBJECT_META[object_id]["position"],
                "paired_recording_count": len(common),
                "session_drift_range_mm": max(session_paired) - min(session_paired),
                "local_drift_range_mm": max(local_paired) - min(local_paired),
                "local_p95_abs_error_mm": float(np.percentile(errors, 95)),
                "repeatability_std_mm": float(np.median([row["repeatability_std_mm"] for row in local_base if finite(row["repeatability_std_mm"]) is not None])),
                "observed_max_abs_error_mm": float(np.max(errors)),
            }
        )
    return output


def fmt(value: Any, digits: int = 6) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def build_report(
    provenance: dict[str, Any],
    a1_audit: dict[str, Any],
    ground_rows: list[dict[str, Any]],
    height_rows: list[dict[str, Any]],
    reconnect_rows: list[dict[str, Any]],
    conclusions: dict[str, Any],
    performance: list[dict[str, Any]],
) -> str:
    ranges = conclusions["ground_component_ranges_mm"]
    reconnect_base = [row for row in reconnect_rows if row["algorithm"] == "base"]
    perf_lines = [
        "| height | position | Session drift range | Local drift range | Local P95 | repeatability std | observed max |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in performance:
        perf_lines.append(
            f"| {row['height']} | {row['position']} | {fmt(row['session_drift_range_mm'])} mm | "
            f"{fmt(row['local_drift_range_mm'])} mm | {fmt(row['local_p95_abs_error_mm'])} mm | "
            f"{fmt(row['repeatability_std_mm'])} mm | {fmt(row['observed_max_abs_error_mm'])} mm |"
        )
    reconnect_lines = [
        "| metric | object | reference | extra step (mm) | threshold (mm) | result |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in reconnect_base:
        reconnect_lines.append(
            f"| {row['metric']} | {row['object_id']} | {row['reference']} | "
            f"{fmt(row['extra_step_mm'])} | {fmt(row['detection_threshold_mm'])} | {row['extra_step_detected']} |"
        )
    quality = conclusions["quality"]
    return f"""# Thermal-A2｜Frozen reconstruction 与 Session/Local 双基准热漂分析

## 结论

- `GROUND_THERMAL_DRIFT_PRESENT = {conclusions['GROUND_THERMAL_DRIFT_PRESENT']}`
- `GROUND_DRIFT_MODE = {conclusions['GROUND_DRIFT_MODE']}`
- `SESSION_REFERENCE_THERMAL_DRIFT = {fmt(conclusions['SESSION_REFERENCE_THERMAL_DRIFT'])} mm`
- `LOCAL_REFERENCE_THERMAL_DRIFT = {fmt(conclusions['LOCAL_REFERENCE_THERMAL_DRIFT'])} mm`
- `LOCAL_REFERENCE_SUPPRESSION_RATIO = {fmt(conclusions['LOCAL_REFERENCE_SUPPRESSION_RATIO'], 4)}`
- `LOCAL_REFERENCE_SUPPRESSES_THERMAL_DRIFT = {conclusions['LOCAL_REFERENCE_SUPPRESSES_THERMAL_DRIFT']}`
- `HEIGHT_DEPENDENT_THERMAL_DRIFT = {conclusions['HEIGHT_DEPENDENT_THERMAL_DRIFT']}`
- `POSITION_DEPENDENT_THERMAL_DRIFT = {conclusions['POSITION_DEPENDENT_THERMAL_DRIFT']}`
- `RECONNECT_EXTRA_STEP_DETECTED = {conclusions['RECONNECT_EXTRA_STEP_DETECTED']}`
- `THERMAL_STEADY_STATE_REACHED = {conclusions['THERMAL_STEADY_STATE_REACHED']}`
- `ESTIMATED_WARMUP_TIME_MIN = NA`

这些数值是 0827 单次 cold-start Session 的 **observed thermal envelope**，不是系统理论最坏值。

## Artifact provenance / reuse audit

- 复用 A1 inventory/timeline/QC：29 recording、580 PNG；22 个 pre-reconnect、7 个 post-reconnect；所有正式帧为 2000 μs / Gain 0 / Mono8 / 固定硬件 ROI。
- 正式 ROI 仅来自用户 GUI Freeze：`{provenance['registry_path']}`；SHA256 `{provenance['registry_sha256']}`；状态 `{provenance['registry_status']}`；support gate `PASS`。
- 旧 Codex registry 已验证为 `INVALID_ATTEMPT`，且未作为输入。
- 本轮新增计算：全部 580 PNG 的 Frozen Steger + C0/C1 + Session R/t reconstruction，以及本文全部 Ground/height/reconnect/steady-state 统计与图表。
- 没有复用 `height_shadow.csv`；没有重拟 C0/C1、Session R/t、Session Ground、H1 或 H-B2。
- ROI registry 是唯一权威范围。旧 `thermal_roi_v2_manual_review.md` 中残留的过时数字不参与计算。

## 方法

- Session-reference：Frozen Session Ground 只在保存的有效 S 域内应用，不外推；Height ROI 的有效点直接相对该 09:57 reference 测量。
- Local-reference：在未做 Session leveling 的重建点上，使用每个量块人工冻结的双侧 Ground，按正式 `measure_height_line(..., ground_correction_mode='auto')` 拟局部线性 Ground；两侧任一少于 20 点即保留该帧并标 invalid。
- Base 是主分析量。H1/H-B2 仅对每帧最终 Base 标量调用正式 correction resolver；H-B2 保留 q2 domain/status，不以诊断外推值冒充 active valid。
- Ground：六段人工 Ground ROI 合并后，在 Frozen Session 的 S 坐标中按 recording robust 拟合 `Zg=aS+b+r`。这些是漂移诊断，不是新 Ground calibration。
- Thermal range：在每个 object 的 Session/Local 共同有效 recording 集合上，recording mean 的最大值减最小值；总值取三个 object 中的最大 range。`LOCAL_REFERENCE_SUPPRESSION_RATIO = 1 - Local_max_range / Session_max_range`，是 conservative worst-object envelope ratio。
- reconnect：只对最后 5 个 pre recording 拟趋势并外推 post 首点；阈值为 `max(0.03 mm, 3×sqrt(pre-trend RMSE² + repeatability²))`。没有跨边界连续拟合。

## Ground 漂移

- offset range：{fmt(ranges['offset'])} mm；tilt 在公共 S span 上的等效 range：{fmt(ranges['tilt_across_common_span'])} mm；最大 detrended shape P95：{fmt(ranges['shape_p95_max'])} mm。
- heuristic QC threshold：{fmt(ranges['significance_threshold'])} mm。模式按超过该 repeatability-based heuristic 的 OFFSET/TILT/SHAPE component 判定；多个 component 同时超过则为 MIXED。这不是跨 Session 统计显著性检验。
- recording Ground fit 使用 20 帧 × 六段 frozen Ground ROI 的 pooled-point weighting；同时在 CSV 中给出 frame-equal a/b sensitivity，避免把 pooled weighting 误作新 calibration。
- 首 recording 与末 recording 的 offset b 分别为 {fmt(ground_rows[0]['offset_b_mm'])} mm、{fmt(ground_rows[-1]['offset_b_mm'])} mm。

## 最终阶段性性能表

表中 drift range 使用 Base recording means；Local P95/observed max 是全部有效 Local Base 帧相对 nominal 的绝对误差；repeatability std 是各 recording 20-frame std 的中位数。

{chr(10).join(perf_lines)}

20/30/10 mm 各自固定在 upper/middle/lower，因此高度和位置完全共线；本 Session 只能报告差异存在的部分证据，不能把差异唯一归因于 height 或 position。

## Reconnect 独立审计

{chr(10).join(reconnect_lines)}

13:01 前后始终作为独立 segment。post 首点只与 pre-only 趋势预测比较；未把相机重连当 thermal reset，也未将前后拟成单条连续曲线。

## 稳态与提取质量

- 单次 cold-start 且存在 reconnect，不足以确认系统通用稳态。pre-reconnect plateau candidate：{fmt(conclusions['observed_pre_reconnect_plateau_candidate_min'], 2)} min；正式 warmup 保持 NA。
- invalid reconstruction frames：{quality['invalid_reconstruction_frames']}；reference/support failure rows：{quality['reference_support_failure_rows']}。
- Steger recording-mean count range/median：{fmt(quality['steger_count_range_ratio'], 4)}；Local drift magnitude 与 Steger count Pearson r：{fmt(quality['local_drift_vs_steger_count_pearson_r'], 4)}。
- extraction degradation detected：{'YES' if quality['extraction_degradation_detected'] else 'NO'}。该判断只用于识别 observed drift 是否伴随提取质量变化，不删除坏帧。

## 时间轴边界

- 09:50 power on；09:57 Session calibration complete；10:00 formal recording start。
- exposure sweep 的准确时刻未记录在 A1 timeline；A1 已确认全部正式 recording 仍为 2000 μs，因此不伪造时间标记。
- 11:30–11:45 pause 与 12:13–13:01 no-record gap 都只是 observation gap，不是 thermal reset。
- ≈13:01 camera reconnect 为独立事件；post-reconnect 结论保持分段解释。

## 输出说明

- `thermal_a2_frame_results.csv`：每帧 × 三目标，含 Session/Local Base/H1/H-B2、q1/q2、支持与 extraction QC。
- `thermal_a2_recording_summary.csv`：recording-level extraction、Ground 与 Base repeatability。
- `thermal_a2_ground_drift.csv`：a/b、detrended RMSE/P95 和相对首 recording 的 profile/shape 变化。
- `thermal_a2_height_session_local.csv`：recording × object × reference × algorithm 长表。
- `thermal_a2_reconnect_audit.csv`：pre-only prediction 与额外 step。
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("input_dir", "a1_dir", "registry", "measure_config", "session_ground", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry, rois, provenance = validate_registry(
        args.registry,
        args.expected_registry_sha256,
        args.measure_config,
        args.session_ground,
    )
    a1_rows, a1_audit = load_a1(args.a1_dir, args.input_dir)
    if args.max_recordings > 0:
        a1_rows = a1_rows[: args.max_recordings]
    app, calibration, ground_payload, session_reference = load_chain(
        args.measure_config, args.session_ground
    )
    frame_rows, recording_rows, ground_rows, ground_fits = process_all(
        a1_rows,
        rois,
        app,
        calibration,
        session_reference,
        0,
    )
    if args.max_recordings > 0:
        print("SMOKE_RUN_COMPLETE: formal outputs were not written", flush=True)
        return 0
    if len(recording_rows) != 29 or len(frame_rows) != 580 * 3:
        raise ThermalA2Error(
            f"Formal output cardinality mismatch: recordings={len(recording_rows)}, frame_rows={len(frame_rows)}"
        )
    profiles = add_ground_profile_metrics(ground_rows, ground_fits)
    height_rows = build_height_summary(frame_rows, recording_rows)
    reconnect_rows = reconnect_audit(ground_rows, height_rows)
    expected_cardinality = {
        "recording_rows": 29,
        "ground_rows": 29,
        "frame_object_rows": 580 * 3,
        "height_summary_rows": 29 * 3 * 2 * 3,
        "reconnect_rows": 1 + 3 * 2 * 3,
    }
    actual_cardinality = {
        "recording_rows": len(recording_rows),
        "ground_rows": len(ground_rows),
        "frame_object_rows": len(frame_rows),
        "height_summary_rows": len(height_rows),
        "reconnect_rows": len(reconnect_rows),
    }
    if actual_cardinality != expected_cardinality:
        raise ThermalA2Error(
            f"Formal table cardinality mismatch: {actual_cardinality} != {expected_cardinality}"
        )
    conclusions = derive_conclusions(
        frame_rows, recording_rows, ground_rows, height_rows, reconnect_rows
    )
    performance = performance_rows(frame_rows, height_rows)

    write_csv(args.output_dir / "thermal_a2_frame_results.csv", frame_rows)
    write_csv(args.output_dir / "thermal_a2_recording_summary.csv", recording_rows)
    write_csv(args.output_dir / "thermal_a2_ground_drift.csv", ground_rows)
    write_csv(args.output_dir / "thermal_a2_height_session_local.csv", height_rows)
    write_csv(args.output_dir / "thermal_a2_reconnect_audit.csv", reconnect_rows)
    save_plots(
        args.output_dir,
        ground_rows,
        profiles,
        height_rows,
        recording_rows,
        reconnect_rows,
    )
    report = build_report(
        provenance,
        a1_audit,
        ground_rows,
        height_rows,
        reconnect_rows,
        conclusions,
        performance,
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    write_json(
        args.output_dir / "thermal_a2_run_manifest.json",
        {
            "status": "COMPLETE",
            "protocol": "Thermal-A2 Frozen reconstruction / Session-Local",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance": provenance,
            "a1_audit": a1_audit,
            "formal_chain": {
                "measure_config": str(args.measure_config),
                "measure_config_sha256": sha256_file(args.measure_config),
                "session_ground": str(args.session_ground),
                "session_ground_sha256": sha256_file(args.session_ground),
                "frozen_c1_enabled": bool(app.reconstruction.enable_laser_ray_correction),
                "ground_u_compensation": app.calibration.ground_u_compensation,
                "height_shadow_used": False,
                "models_refit": [],
            },
            "cardinality": {
                "recordings": len(recording_rows),
                "frames": 580,
                "frame_object_rows": len(frame_rows),
                "height_summary_rows": len(height_rows),
            },
            "rois": {key: json_safe(asdict(value)) for key, value in rois.items()},
            "conclusions": conclusions,
            "performance_table": performance,
            "output_sha256": {
                path.name: sha256_file(path)
                for path in sorted(args.output_dir.iterdir())
                if path.is_file() and path.name != "thermal_a2_run_manifest.json"
            },
        },
    )
    print(json.dumps(json_safe(conclusions), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ThermalA2Error as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
