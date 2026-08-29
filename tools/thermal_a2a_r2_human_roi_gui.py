#!/usr/bin/env python3
"""Human-in-the-loop ROI selection and Freeze gate for Thermal-A2a-R2-Fix.

The GUI displays only:

- the median of the 20 raw Mono8 frames,
- the median Frozen Steger centreline,
- the original A2a Auto ROI candidates,
- ranges explicitly entered or dragged by the user.

No default Manual ROI is supplied.  The program never accepts an Auto range
on behalf of the user.  After the user completes all nine ranges, it saves a
draft and checks only per-frame Steger and Frozen reconstruction point
support.  It does not call height measurement code and does not calculate
height, error, residual, thermal drift, or Auto-vs-Manual results.

A formal frozen registry is created only after every support gate passes and
the user presses Freeze, supplies a human reviewer name, and types FREEZE in
the confirmation dialog.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = ROOT / "laser_measurement_tool"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from laser.backends import create_extraction_params  # noqa: E402
from laser.laser_extractor import extract_laser_center  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402
from utils.image_io import load_grayscale_image  # noqa: E402


FRAME_FIELDS = [
    "filename",
    "camera_frame_number",
    "camera_timestamp_ticks",
    "host_timestamp_ns",
    "host_monotonic_ns",
    "frame_gap",
    "exposure_us",
    "gain_db",
    "pixel_format",
    "offset_x",
    "offset_y",
    "width",
    "height",
]
OBJECT_IDS = ("upper", "middle", "lower")
HEIGHT_LABELS = {"upper": "20 mm", "middle": "30 mm", "lower": "10 mm"}
OBJECT_COLORS = {"upper": "#386cb0", "middle": "#f0027f", "lower": "#1b9e77"}
ROLES = ("baseline_before", "height", "baseline_after")
ROLE_LABELS = {
    "baseline_before": "Ground before",
    "height": "Height",
    "baseline_after": "Ground after",
}
ROLE_COLORS = {
    "baseline_before": "#5c6bc0",
    "height": "#ef6c00",
    "baseline_after": "#43a047",
}
MIN_SUPPORT_POINTS = 20
RANGE_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:-|,|:|\s)\s*(\d+(?:\.\d+)?)\s*$"
)


class HumanRoiError(RuntimeError):
    """Raised when provenance, geometry, or support gates fail."""


@dataclass(frozen=True, slots=True)
class SourceFrame:
    row_index: int
    filename: str
    image_path: Path
    image: np.ndarray
    camera_frame_number: int
    host_timestamp_ns: int
    exposure_us: float
    gain_db: float
    pixel_format: str
    offset_x: int
    offset_y: int
    width: int
    height: int
    centers_uv_full: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedContext:
    args: argparse.Namespace
    frames: list[SourceFrame]
    median_image: np.ndarray
    median_centerline_full: np.ndarray
    auto_registry: dict[str, Any]
    auto_by_id: dict[str, dict[str, Any]]
    reconstructed_pixels: list[np.ndarray]
    app: Any
    calibration: dict[str, Any]
    session_payload: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_input = (
        ROOT
        / "laser_measurement_tool"
        / "output_daheng_0811"
        / "online_recordings"
        / "0827上午热漂_2000"
    )
    default_output = (
        ROOT
        / "projects"
        / "daheng"
        / "analysis"
        / "thermal_a2a_roi_v2_0827"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--measure-config",
        type=Path,
        default=ROOT
        / "laser_measurement_tool"
        / "configs"
        / "measure_tool_daheng_0811.yaml",
    )
    parser.add_argument(
        "--auto-draft",
        type=Path,
        default=default_output / "thermal_roi_v2_registry_v2_draft.json",
    )
    parser.add_argument(
        "--session-ground",
        type=Path,
        default=default_input / "session_ground_calibration.json",
    )
    parser.add_argument(
        "--recording-id",
        default="recording_20260827_100021",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Load all evidence and Frozen reconstruction without opening a GUI.",
    )
    return parser.parse_args(argv)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(json_safe(row.get(field)), ensure_ascii=False)
                        if isinstance(row.get(field), (dict, list, tuple))
                        else row.get(field)
                    )
                    for field in fieldnames
                }
            )


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HumanRoiError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise HumanRoiError(f"JSON root must be an object: {path}")
    return payload


def parse_int(value: Any, name: str) -> int:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise HumanRoiError(f"{name} is not an integer: {value!r}") from error
    if not math.isfinite(number) or number != int(number):
        raise HumanRoiError(f"{name} is not an integer: {value!r}")
    return int(number)


def parse_float(value: Any, name: str) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise HumanRoiError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise HumanRoiError(f"{name} is not finite: {value!r}")
    return number


def read_frames_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise HumanRoiError(f"Missing frames.csv: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != FRAME_FIELDS:
            raise HumanRoiError(
                f"frames.csv schema mismatch: expected {FRAME_FIELDS}, got {header}"
            )
        rows = []
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise HumanRoiError(
                    f"frames.csv column count mismatch at line {line_number}"
                )
            rows.append(
                {
                    field: str(values[index]).strip()
                    for index, field in enumerate(header)
                }
            )
    if len(rows) != 20:
        raise HumanRoiError(f"Expected 20 frame rows, got {len(rows)}")
    return rows


def load_source_frames(
    recording_path: Path,
    app: Any,
) -> list[SourceFrame]:
    if app.extraction_method != "steger":
        raise HumanRoiError(
            f"Configured extraction must be Frozen Steger, got {app.extraction_method!r}"
        )
    extraction_params = create_extraction_params(
        app.extraction_method,
        app.extraction_options_by_method.get(app.extraction_method, {}),
    )
    rows = read_frames_csv(recording_path / "frames.csv")
    frames: list[SourceFrame] = []
    for row_index, row in enumerate(rows, start=1):
        image_path = recording_path / row["filename"]
        image = load_grayscale_image(image_path)
        width = parse_int(row["width"], "width")
        height = parse_int(row["height"], "height")
        if tuple(image.shape) != (height, width):
            raise HumanRoiError(
                f"Image shape mismatch for {image_path.name}: "
                f"{image.shape} != {(height, width)}"
            )
        offset_x = parse_int(row["offset_x"], "offset_x")
        offset_y = parse_int(row["offset_y"], "offset_y")
        centers_local = extract_laser_center(
            image,
            extraction_params,
            image_offset=(offset_x, offset_y),
        )
        centers_full = np.ascontiguousarray(centers_local, dtype=np.float64).copy()
        if centers_full.size:
            if centers_full.ndim != 2 or centers_full.shape[1] != 2:
                raise HumanRoiError(
                    f"Invalid Steger centre shape for {image_path.name}: "
                    f"{centers_full.shape}"
                )
            centers_full[:, 0] += offset_x
            centers_full[:, 1] += offset_y
        frame = SourceFrame(
            row_index=row_index,
            filename=row["filename"],
            image_path=image_path,
            image=np.asarray(image, dtype=np.uint8),
            camera_frame_number=parse_int(
                row["camera_frame_number"], "camera_frame_number"
            ),
            host_timestamp_ns=parse_int(
                row["host_timestamp_ns"], "host_timestamp_ns"
            ),
            exposure_us=parse_float(row["exposure_us"], "exposure_us"),
            gain_db=parse_float(row["gain_db"], "gain_db"),
            pixel_format=row["pixel_format"],
            offset_x=offset_x,
            offset_y=offset_y,
            width=width,
            height=height,
            centers_uv_full=centers_full,
        )
        if abs(frame.exposure_us - 2000.0) > 1.0e-9:
            raise HumanRoiError(
                f"{frame.filename} exposure is not 2000 us: {frame.exposure_us}"
            )
        if abs(frame.gain_db) > 1.0e-9:
            raise HumanRoiError(
                f"{frame.filename} gain is not 0: {frame.gain_db}"
            )
        if frame.pixel_format.strip().lower() != "mono8":
            raise HumanRoiError(
                f"{frame.filename} is not Mono8: {frame.pixel_format!r}"
            )
        frames.append(frame)
    return frames


def median_raw_image(frames: list[SourceFrame]) -> np.ndarray:
    shapes = {tuple(frame.image.shape) for frame in frames}
    if len(shapes) != 1:
        raise HumanRoiError(f"Inconsistent source image shapes: {shapes}")
    stack = np.stack([frame.image for frame in frames], axis=0)
    return np.asarray(np.rint(np.median(stack, axis=0)), dtype=np.uint8)


def median_centerline(frames: list[SourceFrame]) -> np.ndarray:
    by_v: dict[int, list[float]] = {}
    for frame in frames:
        values = np.asarray(frame.centers_uv_full, dtype=np.float64)
        finite = values[np.isfinite(values).all(axis=1)]
        for u, v in finite:
            by_v.setdefault(int(round(float(v))), []).append(float(u))
    points = [
        (float(np.median(by_v[v])), float(v))
        for v in sorted(by_v)
        if by_v[v]
    ]
    if len(points) < 50:
        raise HumanRoiError(
            f"Median Frozen Steger centreline has too few rows: {len(points)}"
        )
    return np.asarray(points, dtype=np.float64)


def load_auto_registry(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = load_json(path)
    if payload.get("status") != "DRAFT" or payload.get("frozen") is not False:
        raise HumanRoiError(
            "R2-Fix requires the original unfrozen A2a Auto draft"
        )
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise HumanRoiError("Auto draft objects must be a list")
    by_id = {
        str(item.get("object_id")): item
        for item in objects
        if isinstance(item, dict)
    }
    if set(by_id) != set(OBJECT_IDS):
        raise HumanRoiError(
            f"Auto draft must contain upper/middle/lower, got {sorted(by_id)}"
        )
    expected_labels = {"upper": "20mm", "middle": "30mm", "lower": "10mm"}
    for object_id in OBJECT_IDS:
        if str(by_id[object_id].get("height_label_hint")) != expected_labels[object_id]:
            raise HumanRoiError(
                f"Physical mapping mismatch for {object_id}: "
                f"{by_id[object_id].get('height_label_hint')!r}"
            )
    return payload, by_id


def load_formal_calibration(
    args: argparse.Namespace,
    app: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_model,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        app.calibration.laser_ray_correction,
        ground_u_optional=True,
    )
    session_payload = load_json(args.session_ground)
    if (
        session_payload.get("status") != "VALID"
        or session_payload.get("valid") is not True
    ):
        raise HumanRoiError("session_ground_calibration.json is not VALID")
    runtime = session_payload.get("runtime", {})
    if runtime.get("ground_extrinsic_source") != "session":
        raise HumanRoiError("Session R/t source is not session")
    try:
        calibration["R"] = np.asarray(
            session_payload["session_extrinsic"]["R_camera_to_ground"],
            dtype=np.float64,
        )
        calibration["t"] = np.asarray(
            session_payload["session_extrinsic"]["t_camera_to_ground_mm"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HumanRoiError(f"Saved Session R/t is invalid: {error}") from error
    if app.reconstruction.enable_laser_ray_correction and calibration.get(
        "laser_ray_correction"
    ) is None:
        raise HumanRoiError("Frozen C1 is enabled but did not load")
    return calibration, session_payload


def reconstruct_support_pixels(
    frames: list[SourceFrame],
    calibration: dict[str, Any],
    app: Any,
) -> list[np.ndarray]:
    results: list[np.ndarray] = []
    for frame in frames:
        reconstructed = reconstruct_uv_to_ground(
            frame.centers_uv_full,
            calibration,
            app.reconstruction,
        )
        pixels = np.asarray(reconstructed.pixels_uv, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise HumanRoiError(
                f"Invalid reconstruction pixels for {frame.filename}: {pixels.shape}"
            )
        results.append(np.ascontiguousarray(pixels))
    return results


def prepare_context(args: argparse.Namespace) -> PreparedContext:
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.measure_config = args.measure_config.resolve()
    args.auto_draft = args.auto_draft.resolve()
    args.session_ground = args.session_ground.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recording_path = args.input_dir / args.recording_id
    if not recording_path.is_dir():
        raise HumanRoiError(f"Recording directory is missing: {recording_path}")
    app = load_app_config(args.measure_config)
    frames = load_source_frames(recording_path, app)
    auto_registry, auto_by_id = load_auto_registry(args.auto_draft)
    calibration, session_payload = load_formal_calibration(args, app)
    reconstructed_pixels = reconstruct_support_pixels(frames, calibration, app)
    return PreparedContext(
        args=args,
        frames=frames,
        median_image=median_raw_image(frames),
        median_centerline_full=median_centerline(frames),
        auto_registry=auto_registry,
        auto_by_id=auto_by_id,
        reconstructed_pixels=reconstructed_pixels,
        app=app,
        calibration=calibration,
        session_payload=session_payload,
    )


def parse_range_text(text: str, name: str, image_height: int) -> list[int]:
    match = RANGE_PATTERN.match(text)
    if match is None:
        raise HumanRoiError(
            f"{name} must be entered as start-end, for example 120-180"
        )
    first = int(round(float(match.group(1))))
    second = int(round(float(match.group(2))))
    lo, hi = sorted((first, second))
    if lo == hi:
        raise HumanRoiError(f"{name} has zero width: {lo}-{hi}")
    if not (0 <= lo < hi < image_height):
        raise HumanRoiError(
            f"{name} is outside image rows 0-{image_height - 1}: {lo}-{hi}"
        )
    return [lo, hi]


def validate_selection_geometry(
    selection: dict[str, dict[str, list[int]]],
    image_height: int,
) -> None:
    occupied: list[tuple[int, int, str, str]] = []
    for object_id in OBJECT_IDS:
        if object_id not in selection:
            raise HumanRoiError(f"Missing object: {object_id}")
        object_ranges = selection[object_id]
        for role in ROLES:
            if role not in object_ranges:
                raise HumanRoiError(f"Missing {object_id}.{role}")
            lo, hi = object_ranges[role]
            if not (0 <= lo < hi < image_height):
                raise HumanRoiError(
                    f"{object_id}.{role} is outside 0-{image_height - 1}"
                )
            occupied.append((lo, hi, object_id, role))
        before = object_ranges["baseline_before"]
        height = object_ranges["height"]
        after = object_ranges["baseline_after"]
        if not (before[1] < height[0] and height[1] < after[0]):
            raise HumanRoiError(
                f"{object_id} must satisfy Ground before < Height < Ground after"
            )
    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if current[0] <= previous[1]:
            raise HumanRoiError(
                "ROI overlap: "
                f"{previous[2]}.{previous[3]}={previous[0]}-{previous[1]} and "
                f"{current[2]}.{current[3]}={current[0]}-{current[1]}"
            )


def interval_count(points_uv: np.ndarray, value_range: list[int]) -> int:
    lo, hi = value_range
    values = np.asarray(points_uv, dtype=np.float64)
    if not len(values):
        return 0
    return int(
        np.count_nonzero(
            np.isfinite(values[:, 1])
            & (values[:, 1] >= lo)
            & (values[:, 1] <= hi)
        )
    )


def compute_support_qc(
    context: PreparedContext,
    selection: dict[str, dict[str, list[int]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for object_id in OBJECT_IDS:
        for role in ROLES:
            value_range = selection[object_id][role]
            role_rows: list[dict[str, Any]] = []
            for frame, reconstructed_pixels in zip(
                context.frames, context.reconstructed_pixels
            ):
                steger_count = interval_count(frame.centers_uv_full, value_range)
                reconstruction_count = interval_count(
                    reconstructed_pixels, value_range
                )
                row = {
                    "object_id": object_id,
                    "height_label": HEIGHT_LABELS[object_id],
                    "role": role,
                    "v_start": value_range[0],
                    "v_end": value_range[1],
                    "frame_index": frame.row_index,
                    "filename": frame.filename,
                    "camera_frame_number": frame.camera_frame_number,
                    "steger_point_count": steger_count,
                    "frozen_reconstruction_point_count": reconstruction_count,
                    "steger_support_ok": steger_count >= MIN_SUPPORT_POINTS,
                    "frozen_reconstruction_support_ok": (
                        reconstruction_count >= MIN_SUPPORT_POINTS
                    ),
                    "support_ok": (
                        steger_count >= MIN_SUPPORT_POINTS
                        and reconstruction_count >= MIN_SUPPORT_POINTS
                    ),
                    "height_computation": "NOT_RUN",
                }
                frame_rows.append(row)
                role_rows.append(row)
            failed_frames = [
                row["filename"] for row in role_rows if not row["support_ok"]
            ]
            summary_rows.append(
                {
                    "object_id": object_id,
                    "height_label": HEIGHT_LABELS[object_id],
                    "role": role,
                    "v_start": value_range[0],
                    "v_end": value_range[1],
                    "frame_count": len(role_rows),
                    "minimum_required_points": MIN_SUPPORT_POINTS,
                    "steger_min_point_count": min(
                        row["steger_point_count"] for row in role_rows
                    ),
                    "steger_mean_point_count": float(
                        np.mean([row["steger_point_count"] for row in role_rows])
                    ),
                    "frozen_reconstruction_min_point_count": min(
                        row["frozen_reconstruction_point_count"]
                        for row in role_rows
                    ),
                    "frozen_reconstruction_mean_point_count": float(
                        np.mean(
                            [
                                row["frozen_reconstruction_point_count"]
                                for row in role_rows
                            ]
                        )
                    ),
                    "failed_frame_count": len(failed_frames),
                    "failed_frames": failed_frames,
                    "support_ok": not failed_frames,
                    "height_computation": "NOT_RUN",
                }
            )
    all_pass = all(bool(row["support_ok"]) for row in summary_rows)
    return summary_rows, frame_rows, all_pass


def support_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "draft": output_dir / "thermal_roi_manual_selection_draft.json",
        "summary_csv": output_dir / "thermal_roi_manual_support_qc.csv",
        "frame_csv": output_dir / "thermal_roi_manual_support_qc_frames.csv",
        "qc_json": output_dir / "thermal_roi_manual_support_qc.json",
        "frozen": output_dir / "thermal_roi_registry_v2_frozen.json",
        "invalid_attempt": (
            output_dir / "thermal_roi_registry_v2_invalid_attempt_codex.json"
        ),
    }


def qc_provenance(
    context: PreparedContext,
    summary_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    all_pass: bool,
) -> dict[str, Any]:
    args = context.args
    return {
        "schema_version": 1,
        "qc_type": "steger_and_frozen_reconstruction_support_only",
        "status": "PASS" if all_pass else "FAIL",
        "generated_at_utc": now_utc(),
        "source_recording": args.recording_id,
        "source_frame_count": len(context.frames),
        "minimum_required_points_per_frame": MIN_SUPPORT_POINTS,
        "height_computation": "NOT_RUN",
        "height_values": None,
        "error_computation": "NOT_RUN",
        "auto_vs_manual": "NOT_RUN",
        "input_config": {
            "path": str(args.measure_config),
            "sha256": sha256_file(args.measure_config),
        },
        "auto_draft": {
            "path": str(args.auto_draft),
            "sha256": sha256_file(args.auto_draft),
            "display_only": True,
        },
        "session_ground_calibration": {
            "path": str(args.session_ground),
            "sha256": sha256_file(args.session_ground),
            "status": context.session_payload.get("status"),
            "valid": context.session_payload.get("valid"),
            "ground_extrinsic_generation": context.session_payload.get(
                "runtime", {}
            ).get("ground_extrinsic_generation"),
        },
        "reconstruction": {
            "function": "reconstruct_uv_to_ground",
            "c0_model_type": context.calibration.get("laser_model", {}).get(
                "model_type"
            ),
            "c1_enabled": bool(
                context.app.reconstruction.enable_laser_ray_correction
            ),
            "min_camera_depth_mm": float(
                context.app.reconstruction.min_camera_depth_mm
            ),
            "max_camera_depth_mm": float(
                context.app.reconstruction.max_camera_depth_mm
            ),
        },
        "summary": summary_rows,
        "frames": frame_rows,
    }


def build_draft(
    context: PreparedContext,
    selection: dict[str, dict[str, list[int]]],
    summary_rows: list[dict[str, Any]],
    all_pass: bool,
    reviewer: str,
) -> dict[str, Any]:
    paths = support_paths(context.args.output_dir)
    summary_by_key = {
        (row["object_id"], row["role"]): row for row in summary_rows
    }
    return {
        "schema_version": 4,
        "registry_type": "thermal_multi_object_roi_v2",
        "status": "DRAFT_SUPPORT_PASS" if all_pass else "DRAFT_SUPPORT_FAIL",
        "valid": False,
        "draft": True,
        "frozen": False,
        "thermal_a2_roi_frozen": False,
        "human_interaction_completed": True,
        "human_reviewer_input": reviewer or None,
        "selection_action": "GUI_DRAG_OR_TYPED_INPUT_AND_SAVE_DRAFT",
        "support_qc_status": "PASS" if all_pass else "FAIL",
        "support_qc_path": str(paths["summary_csv"]),
        "support_qc_json_path": str(paths["qc_json"]),
        "height_computation": "NOT_RUN",
        "height_values": None,
        "error_computation": "NOT_RUN",
        "auto_vs_manual": "NOT_RUN",
        "created_at_utc": now_utc(),
        "source_recording": context.args.recording_id,
        "source_recording_path": str(
            context.args.input_dir / context.args.recording_id
        ),
        "source_frame_count": len(context.frames),
        "source_frame_files": [frame.filename for frame in context.frames],
        "auto_draft": {
            "path": str(context.args.auto_draft),
            "sha256": sha256_file(context.args.auto_draft),
            "display_only": True,
        },
        "invalid_codex_attempt": {
            "path": str(paths["invalid_attempt"]),
            "sha256": sha256_file(paths["invalid_attempt"]),
            "formal_input": False,
        },
        "objects": [
            {
                "object_id": object_id,
                "object_order": index + 1,
                "height_label": HEIGHT_LABELS[object_id],
                "height_label_basis": "user-confirmed physical placement metadata",
                "manual_roi": {
                    role: list(selection[object_id][role]) for role in ROLES
                },
                "selection_source": "USER_GUI",
                "auto_roi_snapshot": {
                    "height_v_range": context.auto_by_id[object_id].get(
                        "height_v_range"
                    ),
                    "baseline_v_ranges": context.auto_by_id[object_id].get(
                        "baseline_v_ranges"
                    ),
                    "display_only": True,
                },
                "support_qc": {
                    role: summary_by_key[(object_id, role)] for role in ROLES
                },
            }
            for index, object_id in enumerate(OBJECT_IDS)
        ],
    }


def build_frozen_registry(
    context: PreparedContext,
    draft: dict[str, Any],
    reviewer: str,
    confirmation_time: str,
) -> dict[str, Any]:
    paths = support_paths(context.args.output_dir)
    if draft.get("support_qc_status") != "PASS":
        raise HumanRoiError("Cannot Freeze: draft support QC is not PASS")
    return {
        "schema_version": 4,
        "registry_type": "thermal_multi_object_roi_v2",
        "status": "FROZEN_USER_CONFIRMED",
        "valid": True,
        "frozen": True,
        "thermal_a2_roi_frozen": True,
        "human_reviewed": True,
        "manual_confirmed": True,
        "manual_confirmed_count": 3,
        "manual_decision": "ACCEPTED_BY_USER",
        "manual_reviewer": reviewer,
        "manual_confirmation": {
            "gui_freeze_button_clicked": True,
            "typed_confirmation_token": "FREEZE",
            "confirmed_at_utc": confirmation_time,
            "confirmed_by": reviewer,
        },
        "selection_basis": (
            "User-selected image-v intervals in interactive GUI displaying "
            "median raw Mono8, Frozen Steger centreline, and Auto candidates"
        ),
        "support_gate": {
            "status": "PASS",
            "minimum_points_per_frame": MIN_SUPPORT_POINTS,
            "steger_support_checked": True,
            "frozen_reconstruction_support_checked": True,
            "summary_csv": str(paths["summary_csv"]),
            "summary_csv_sha256": sha256_file(paths["summary_csv"]),
            "frame_csv": str(paths["frame_csv"]),
            "frame_csv_sha256": sha256_file(paths["frame_csv"]),
            "qc_json": str(paths["qc_json"]),
            "qc_json_sha256": sha256_file(paths["qc_json"]),
        },
        "height_computation": "NOT_RUN",
        "height_values": None,
        "error_computation": "NOT_RUN",
        "auto_vs_manual": "NOT_RUN",
        "thermal_a2_execution": "NOT_RUN",
        "source_recording": draft["source_recording"],
        "source_recording_path": draft["source_recording_path"],
        "source_frame_count": draft["source_frame_count"],
        "source_frame_files": draft["source_frame_files"],
        "source_frame_sha256": {
            frame.filename: sha256_file(frame.image_path)
            for frame in context.frames
        },
        "measure_config": {
            "path": str(context.args.measure_config),
            "sha256": sha256_file(context.args.measure_config),
        },
        "session_ground_calibration": {
            "path": str(context.args.session_ground),
            "sha256": sha256_file(context.args.session_ground),
            "status": context.session_payload.get("status"),
            "valid": context.session_payload.get("valid"),
        },
        "auto_draft": draft["auto_draft"],
        "invalid_codex_attempt": draft["invalid_codex_attempt"],
        "objects": [
            {
                **item,
                "manual_confirmed": True,
                "frozen": True,
            }
            for item in draft["objects"]
        ],
        "freeze_policy": {
            "authoritative_for_all_580_frames": True,
            "same_registry_for_pre_post_reconnect": True,
            "adaptive_reselection_forbidden": True,
            "comparison_may_modify_registry": False,
        },
    }


def selection_signature(
    selection: dict[str, dict[str, list[int]]],
) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (
            object_id,
            role,
            int(selection[object_id][role][0]),
            int(selection[object_id][role][1]),
        )
        for object_id in OBJECT_IDS
        for role in ROLES
    )


def display_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(values, [1.0, 99.5])
    if high <= low:
        return values
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def object_windows(
    auto_by_id: dict[str, dict[str, Any]],
    image_height: int,
) -> dict[str, tuple[int, int]]:
    centres = [
        float(np.mean(auto_by_id[object_id]["height_v_range"]))
        for object_id in OBJECT_IDS
    ]
    boundaries = [
        0,
        int(round((centres[0] + centres[1]) / 2.0)),
        int(round((centres[1] + centres[2]) / 2.0)),
        image_height - 1,
    ]
    return {
        object_id: (boundaries[index], boundaries[index + 1])
        for index, object_id in enumerate(OBJECT_IDS)
    }


def launch_gui(context: PreparedContext) -> None:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, simpledialog, ttk

    import matplotlib

    matplotlib.use("TkAgg", force=True)
    from matplotlib.backends.backend_tkagg import (  # noqa: E402
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure  # noqa: E402
    from matplotlib.lines import Line2D  # noqa: E402
    from matplotlib.patches import Patch  # noqa: E402
    from matplotlib.widgets import SpanSelector  # noqa: E402

    class ManualRoiApp:
        def __init__(self) -> None:
            self.context = context
            self.paths = support_paths(context.args.output_dir)
            self.root = tk.Tk()
            self.root.title(
                "Thermal-A2a-R2-Fix — User Manual ROI Selection and Freeze"
            )
            self.root.geometry("1780x1040")
            self.root.minsize(1450, 850)
            self._loading = False
            self.current_drag_key: tuple[str, str] | None = None
            self.last_pass_signature: tuple[
                tuple[str, str, int, int], ...
            ] | None = None
            self.last_draft: dict[str, Any] | None = None
            self.vars: dict[tuple[str, str], tk.StringVar] = {}
            self.entries: dict[tuple[str, str], tk.Entry] = {}
            self.drag_buttons: dict[tuple[str, str], ttk.Button] = {}
            self.axes: dict[str, Any] = {}
            self.selectors: dict[str, Any] = {}
            self.display = display_image(context.median_image)
            self.windows = object_windows(
                context.auto_by_id, context.frames[0].height
            )
            self.offset_x = context.frames[0].offset_x
            self.reviewer_var = tk.StringVar()
            self._build_layout(
                ttk,
                tk,
                scrolledtext,
                Figure,
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
                SpanSelector,
            )
            self._load_existing_draft_if_any()
            self._redraw(Line2D, Patch)
            self._status(
                "请由用户本人逐项拖动或输入 9 个区间。Auto ROI 仅作为虚线候选显示，"
                "不会被程序自动接受。\n"
                "完成后点击 Save Draft + Check Support。QC FAIL 时窗口保持打开，"
                "请人工修改后再次检查。"
            )

        def _build_layout(
            self,
            ttk_module: Any,
            tk_module: Any,
            scrolledtext_module: Any,
            figure_class: Any,
            canvas_class: Any,
            toolbar_class: Any,
            selector_class: Any,
        ) -> None:
            self.root.columnconfigure(0, weight=4)
            self.root.columnconfigure(1, weight=2)
            self.root.rowconfigure(0, weight=1)

            plot_frame = ttk_module.Frame(self.root)
            plot_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
            plot_frame.rowconfigure(0, weight=1)
            plot_frame.columnconfigure(0, weight=1)
            self.figure = figure_class(figsize=(11.5, 10.5), constrained_layout=True)
            plot_axes = self.figure.subplots(3, 1)
            self.axes = {
                object_id: plot_axes[index]
                for index, object_id in enumerate(OBJECT_IDS)
            }
            self.canvas = canvas_class(self.figure, master=plot_frame)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            toolbar = toolbar_class(self.canvas, plot_frame, pack_toolbar=False)
            toolbar.grid(row=1, column=0, sticky="ew")

            for object_id in OBJECT_IDS:
                selector = selector_class(
                    self.axes[object_id],
                    lambda first, second, oid=object_id: self._span_selected(
                        oid, first, second
                    ),
                    "vertical",
                    useblit=True,
                    props={"alpha": 0.28, "facecolor": OBJECT_COLORS[object_id]},
                    interactive=True,
                    drag_from_anywhere=True,
                )
                selector.set_active(False)
                self.selectors[object_id] = selector

            controls = ttk_module.Frame(self.root)
            controls.grid(row=0, column=1, sticky="nsew", padx=8, pady=6)
            controls.columnconfigure(1, weight=1)
            controls.rowconfigure(9, weight=1)

            title = ttk_module.Label(
                controls,
                text=(
                    "Human ROI Controls\n"
                    "upper=20 mm · middle=30 mm · lower=10 mm"
                ),
                font=("Segoe UI", 13, "bold"),
                justify="center",
            )
            title.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))

            ttk_module.Label(
                controls,
                text=(
                    "输入格式：start-end。也可点击 Drag 后在对应图中纵向拖动。\n"
                    "所有字段初始为空，必须由用户本人输入或拖动。"
                ),
                wraplength=520,
                justify="left",
            ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))

            row = 2
            for object_id in OBJECT_IDS:
                auto = self.context.auto_by_id[object_id]
                ttk_module.Label(
                    controls,
                    text=(
                        f"{object_id} = {HEIGHT_LABELS[object_id]}    "
                        f"Auto: before={auto['baseline_v_ranges'][0]}, "
                        f"height={auto['height_v_range']}, "
                        f"after={auto['baseline_v_ranges'][1]}"
                    ),
                    foreground=OBJECT_COLORS[object_id],
                    font=("Segoe UI", 10, "bold"),
                    wraplength=530,
                ).grid(
                    row=row,
                    column=0,
                    columnspan=3,
                    sticky="w",
                    pady=(8, 2),
                )
                row += 1
                for role in ROLES:
                    key = (object_id, role)
                    variable = tk_module.StringVar()
                    variable.trace_add(
                        "write",
                        lambda *_args, selected_key=key: self._selection_changed(
                            selected_key
                        ),
                    )
                    self.vars[key] = variable
                    ttk_module.Label(
                        controls, text=ROLE_LABELS[role], width=16
                    ).grid(row=row, column=0, sticky="w", padx=(8, 2), pady=2)
                    entry = tk_module.Entry(
                        controls,
                        textvariable=variable,
                        width=20,
                        background="white",
                    )
                    entry.grid(row=row, column=1, sticky="ew", padx=2, pady=2)
                    self.entries[key] = entry
                    button = ttk_module.Button(
                        controls,
                        text="Drag",
                        command=lambda selected_key=key: self._activate_drag(
                            selected_key
                        ),
                    )
                    button.grid(row=row, column=2, sticky="ew", padx=(2, 0), pady=2)
                    self.drag_buttons[key] = button
                    row += 1

            ttk_module.Button(
                controls,
                text="Apply typed values / Redraw",
                command=self._apply_typed,
            ).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 4))
            row += 1

            reviewer_frame = ttk_module.Frame(controls)
            reviewer_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
            reviewer_frame.columnconfigure(1, weight=1)
            ttk_module.Label(
                reviewer_frame, text="Human reviewer name/initials:"
            ).grid(row=0, column=0, sticky="w")
            ttk_module.Entry(
                reviewer_frame, textvariable=self.reviewer_var
            ).grid(row=0, column=1, sticky="ew", padx=(6, 0))
            row += 1

            self.check_button = ttk_module.Button(
                controls,
                text="Save Draft + Check Support",
                command=self._check_support,
            )
            self.check_button.grid(
                row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4)
            )
            row += 1

            self.freeze_button = ttk_module.Button(
                controls,
                text="Freeze (requires QC PASS + typed FREEZE)",
                command=self._freeze,
                state="disabled",
            )
            self.freeze_button.grid(
                row=row, column=0, columnspan=3, sticky="ew", pady=4
            )
            row += 1

            self.status_text = scrolledtext_module.ScrolledText(
                controls,
                height=15,
                wrap=tk_module.WORD,
                font=("Consolas", 9),
            )
            self.status_text.grid(
                row=row,
                column=0,
                columnspan=3,
                sticky="nsew",
                pady=(8, 0),
            )
            controls.rowconfigure(row, weight=1)

        def _status(self, message: str, *, clear: bool = False) -> None:
            if clear:
                self.status_text.delete("1.0", "end")
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_text.insert("end", f"[{timestamp}] {message}\n")
            self.status_text.see("end")

        def _selection_changed(self, key: tuple[str, str]) -> None:
            if self._loading:
                return
            self.last_pass_signature = None
            self.freeze_button.configure(state="disabled")
            self.entries[key].configure(background="white")

        def _load_existing_draft_if_any(self) -> None:
            draft_path = self.paths["draft"]
            if not draft_path.is_file():
                return
            try:
                draft = load_json(draft_path)
                if draft.get("frozen") is True:
                    return
                objects = {
                    str(item.get("object_id")): item
                    for item in draft.get("objects", [])
                    if isinstance(item, dict)
                }
                self._loading = True
                for object_id in OBJECT_IDS:
                    manual = objects.get(object_id, {}).get("manual_roi", {})
                    for role in ROLES:
                        value = manual.get(role)
                        if isinstance(value, list) and len(value) == 2:
                            self.vars[(object_id, role)].set(
                                f"{int(value[0])}-{int(value[1])}"
                            )
                reviewer = draft.get("human_reviewer_input")
                if reviewer:
                    self.reviewer_var.set(str(reviewer))
                self._loading = False
                self._status(
                    f"Loaded existing non-frozen draft: {draft_path.name}"
                )
            except Exception as error:  # noqa: BLE001
                self._loading = False
                self._status(f"Draft resume skipped: {error}")

        def _parse_all(self) -> dict[str, dict[str, list[int]]]:
            selection: dict[str, dict[str, list[int]]] = {}
            image_height = self.context.frames[0].height
            for object_id in OBJECT_IDS:
                selection[object_id] = {}
                for role in ROLES:
                    text = self.vars[(object_id, role)].get().strip()
                    if not text:
                        raise HumanRoiError(
                            f"User input required: {object_id}.{role} is blank"
                        )
                    selection[object_id][role] = parse_range_text(
                        text,
                        f"{object_id}.{role}",
                        image_height,
                    )
            validate_selection_geometry(selection, image_height)
            return selection

        def _activate_drag(self, key: tuple[str, str]) -> None:
            object_id, role = key
            self.current_drag_key = key
            for oid, selector in self.selectors.items():
                selector.set_active(oid == object_id)
            self._status(
                f"Drag active: {object_id}.{role}. "
                "在对应图中沿 v 方向拖动选择区间。"
            )

        def _span_selected(
            self, object_id: str, first: float, second: float
        ) -> None:
            if self.current_drag_key is None:
                return
            selected_object, role = self.current_drag_key
            if selected_object != object_id:
                return
            image_height = self.context.frames[0].height
            lo = max(0, min(image_height - 1, int(round(min(first, second)))))
            hi = max(0, min(image_height - 1, int(round(max(first, second)))))
            if lo == hi:
                messagebox.showwarning(
                    "Invalid span", "Dragged ROI has zero height; please drag again."
                )
                return
            self.vars[(object_id, role)].set(f"{lo}-{hi}")
            self.current_drag_key = None
            for selector in self.selectors.values():
                selector.set_active(False)
            self._apply_typed(show_errors=False)
            self._status(f"User dragged {object_id}.{role} = {lo}-{hi}")

        def _apply_typed(self, *, show_errors: bool = True) -> None:
            try:
                for object_id in OBJECT_IDS:
                    for role in ROLES:
                        text = self.vars[(object_id, role)].get().strip()
                        if text:
                            parse_range_text(
                                text,
                                f"{object_id}.{role}",
                                self.context.frames[0].height,
                            )
                self._redraw(Line2D, Patch)
            except HumanRoiError as error:
                if show_errors:
                    messagebox.showerror("Invalid ROI input", str(error))

        def _redraw(self, line_class: Any, patch_class: Any) -> None:
            center_u = (
                self.context.median_centerline_full[:, 0] - self.offset_x
            )
            center_v = self.context.median_centerline_full[:, 1]
            image_height, image_width = self.context.median_image.shape
            for object_id in OBJECT_IDS:
                ax = self.axes[object_id]
                ax.clear()
                ax.imshow(
                    self.display,
                    cmap="gray",
                    origin="upper",
                    extent=(0, image_width, image_height, 0),
                    aspect="auto",
                )
                window_lo, window_hi = self.windows[object_id]
                centre_mask = (
                    np.isfinite(center_u)
                    & np.isfinite(center_v)
                    & (center_u >= 0)
                    & (center_u <= image_width)
                    & (center_v >= window_lo)
                    & (center_v <= window_hi)
                )
                ax.plot(
                    center_u[centre_mask],
                    center_v[centre_mask],
                    color="#00e5ff",
                    linewidth=0.9,
                )
                auto = self.context.auto_by_id[object_id]
                auto_ranges = {
                    "baseline_before": auto["baseline_v_ranges"][0],
                    "height": auto["height_v_range"],
                    "baseline_after": auto["baseline_v_ranges"][1],
                }
                for role in ROLES:
                    lo, hi = auto_ranges[role]
                    ax.axhspan(
                        lo,
                        hi,
                        facecolor="none",
                        edgecolor=ROLE_COLORS[role],
                        linewidth=1.1,
                        linestyle="--",
                    )
                    text = self.vars[(object_id, role)].get().strip()
                    if text:
                        try:
                            manual_range = parse_range_text(
                                text,
                                f"{object_id}.{role}",
                                image_height,
                            )
                        except HumanRoiError:
                            continue
                        ax.axhspan(
                            manual_range[0],
                            manual_range[1],
                            facecolor=ROLE_COLORS[role],
                            edgecolor=ROLE_COLORS[role],
                            linewidth=1.4,
                            alpha=0.28,
                        )
                edge_pair = auto.get("edge_pair", {})
                if edge_pair.get("edge1_v") is not None:
                    ax.axhline(
                        edge_pair["edge1_v"],
                        color=OBJECT_COLORS[object_id],
                        linewidth=0.75,
                        alpha=0.7,
                    )
                if edge_pair.get("edge2_v") is not None:
                    ax.axhline(
                        edge_pair["edge2_v"],
                        color=OBJECT_COLORS[object_id],
                        linewidth=0.75,
                        alpha=0.7,
                    )
                ax.set_xlim(0, image_width)
                ax.set_ylim(window_hi, window_lo)
                ax.set_ylabel("v")
                ax.set_title(
                    f"{object_id} = {HEIGHT_LABELS[object_id]}  "
                    "Auto dashed · User solid"
                )
                ax.grid(False)
            self.axes["lower"].set_xlabel("raw image u (recording ROI column)")
            legend = [
                line_class(
                    [0],
                    [0],
                    color="#00e5ff",
                    linewidth=1.0,
                    label="median Frozen Steger",
                ),
                line_class(
                    [0],
                    [0],
                    color="#777777",
                    linestyle="--",
                    label="Auto candidate",
                ),
                patch_class(
                    facecolor="#777777",
                    alpha=0.28,
                    label="User selection",
                ),
            ]
            self.figure.legend(
                handles=legend,
                loc="lower center",
                ncol=3,
                frameon=True,
            )
            self.figure.suptitle(
                "Thermal-A2a-R2-Fix — raw Mono8 + Frozen Steger geometry only",
                fontsize=14,
            )
            self.canvas.draw_idle()

        def _mark_qc_entries(self, summary_rows: list[dict[str, Any]]) -> None:
            for entry in self.entries.values():
                entry.configure(background="white")
            for row in summary_rows:
                if not row["support_ok"]:
                    key = (str(row["object_id"]), str(row["role"]))
                    self.entries[key].configure(background="#ffd6d6")

        def _qc_message(self, summary_rows: list[dict[str, Any]]) -> str:
            lines = [
                "Support QC (height/error NOT RUN)",
                "object role range Steger_min Reconstruction_min status",
            ]
            for row in summary_rows:
                lines.append(
                    f"{row['object_id']:6s} {row['role']:15s} "
                    f"{row['v_start']:4d}-{row['v_end']:<4d} "
                    f"{row['steger_min_point_count']:4d} "
                    f"{row['frozen_reconstruction_min_point_count']:4d} "
                    f"{'PASS' if row['support_ok'] else 'FAIL'}"
                )
                if row["failed_frames"]:
                    lines.append(
                        "  failed frames: " + ", ".join(row["failed_frames"])
                    )
            return "\n".join(lines)

        def _check_support(self) -> None:
            try:
                selection = self._parse_all()
            except HumanRoiError as error:
                messagebox.showerror("Incomplete or invalid Manual ROI", str(error))
                return
            self.check_button.configure(state="disabled")
            self.freeze_button.configure(state="disabled")
            self._status("Running Steger/Frozen reconstruction support count only...")
            self.root.update_idletasks()
            try:
                summary_rows, frame_rows, all_pass = compute_support_qc(
                    self.context, selection
                )
                qc_payload = qc_provenance(
                    self.context, summary_rows, frame_rows, all_pass
                )
                write_csv(
                    self.paths["summary_csv"],
                    [
                        "object_id",
                        "height_label",
                        "role",
                        "v_start",
                        "v_end",
                        "frame_count",
                        "minimum_required_points",
                        "steger_min_point_count",
                        "steger_mean_point_count",
                        "frozen_reconstruction_min_point_count",
                        "frozen_reconstruction_mean_point_count",
                        "failed_frame_count",
                        "failed_frames",
                        "support_ok",
                        "height_computation",
                    ],
                    summary_rows,
                )
                write_csv(
                    self.paths["frame_csv"],
                    [
                        "object_id",
                        "height_label",
                        "role",
                        "v_start",
                        "v_end",
                        "frame_index",
                        "filename",
                        "camera_frame_number",
                        "steger_point_count",
                        "frozen_reconstruction_point_count",
                        "steger_support_ok",
                        "frozen_reconstruction_support_ok",
                        "support_ok",
                        "height_computation",
                    ],
                    frame_rows,
                )
                write_json(self.paths["qc_json"], qc_payload)
                reviewer = self.reviewer_var.get().strip()
                draft = build_draft(
                    self.context,
                    selection,
                    summary_rows,
                    all_pass,
                    reviewer,
                )
                write_json(self.paths["draft"], draft)
                self.last_draft = draft
                self._mark_qc_entries(summary_rows)
                message = self._qc_message(summary_rows)
                self._status(message)
                if all_pass:
                    self.last_pass_signature = selection_signature(selection)
                    self.freeze_button.configure(state="normal")
                    messagebox.showinfo(
                        "Support QC PASS",
                        "All nine ROI intervals pass Steger and Frozen "
                        "reconstruction support on all 20 frames.\n\n"
                        "Draft saved. Freeze is still NOT performed. "
                        "Enter a human reviewer name and press Freeze only "
                        "when you explicitly confirm these ranges.",
                    )
                else:
                    self.last_pass_signature = None
                    self.freeze_button.configure(state="disabled")
                    failed = [
                        f"{row['object_id']}.{row['role']} "
                        f"{row['v_start']}-{row['v_end']} "
                        f"(Steger min={row['steger_min_point_count']}, "
                        f"Recon min={row['frozen_reconstruction_min_point_count']})"
                        for row in summary_rows
                        if not row["support_ok"]
                    ]
                    messagebox.showwarning(
                        "Support QC FAIL — GUI remains open",
                        "The following user-selected regions failed support:\n\n"
                        + "\n".join(failed)
                        + "\n\nNo automatic correction was applied. "
                        "Please modify the highlighted fields manually and "
                        "run support QC again.",
                    )
            except Exception as error:  # noqa: BLE001
                logging.exception("Support QC failed")
                messagebox.showerror("Support QC error", str(error))
            finally:
                self.check_button.configure(state="normal")

        def _freeze(self) -> None:
            try:
                selection = self._parse_all()
            except HumanRoiError as error:
                messagebox.showerror("Cannot Freeze", str(error))
                return
            signature = selection_signature(selection)
            if (
                self.last_pass_signature is None
                or signature != self.last_pass_signature
                or self.last_draft is None
                or self.last_draft.get("support_qc_status") != "PASS"
            ):
                self.freeze_button.configure(state="disabled")
                messagebox.showerror(
                    "Cannot Freeze",
                    "Current ROI values do not match the last PASS support QC. "
                    "Run Save Draft + Check Support again.",
                )
                return
            reviewer = self.reviewer_var.get().strip()
            if not reviewer:
                messagebox.showerror(
                    "Human reviewer required",
                    "Enter your name or initials before Freeze.",
                )
                return
            if "codex" in reviewer.lower() or reviewer.lower() in {
                "ai",
                "assistant",
                "auto",
            }:
                messagebox.showerror(
                    "Human reviewer required",
                    "Codex/AI cannot be the reviewer. Enter the human user's "
                    "name or initials.",
                )
                return
            if self.paths["frozen"].exists():
                messagebox.showerror(
                    "Frozen registry already exists",
                    f"Refusing to overwrite:\n{self.paths['frozen']}",
                )
                return
            token = simpledialog.askstring(
                "Explicit Freeze confirmation",
                "Type FREEZE exactly to create the formal frozen registry.\n\n"
                "This action records your current 9 ROI intervals as the "
                "single registry for all 580 frames.",
                parent=self.root,
            )
            if token != "FREEZE":
                self._status("Freeze cancelled: confirmation token was not FREEZE.")
                return
            confirmation_time = now_utc()
            frozen = build_frozen_registry(
                self.context,
                self.last_draft,
                reviewer,
                confirmation_time,
            )
            try:
                write_json(self.paths["frozen"], frozen, exclusive=True)
            except FileExistsError:
                messagebox.showerror(
                    "Frozen registry already exists",
                    f"Refusing to overwrite:\n{self.paths['frozen']}",
                )
                return
            self.freeze_button.configure(state="disabled")
            self.check_button.configure(state="disabled")
            for entry in self.entries.values():
                entry.configure(state="disabled")
            for button in self.drag_buttons.values():
                button.configure(state="disabled")
            self._status(
                "USER FREEZE COMPLETED. Formal registry created:\n"
                f"{self.paths['frozen']}\n"
                f"SHA256={sha256_file(self.paths['frozen'])}"
            )
            messagebox.showinfo(
                "Freeze completed",
                "Formal user-confirmed ROI registry was created.\n\n"
                f"{self.paths['frozen']}\n\n"
                "No Auto-vs-Manual or Thermal-A2 analysis was run.",
            )

        def run(self) -> None:
            self.root.mainloop()

    ManualRoiApp().run()


def self_test(context: PreparedContext) -> None:
    paths = support_paths(context.args.output_dir)
    print(f"recording={context.args.recording_id}")
    print(f"frames={len(context.frames)}")
    print(f"median_image_shape={tuple(context.median_image.shape)}")
    print(f"median_centerline_rows={len(context.median_centerline_full)}")
    print(
        "steger_points_min="
        f"{min(len(frame.centers_uv_full) for frame in context.frames)}"
    )
    print(
        "reconstructed_points_min="
        f"{min(len(points) for points in context.reconstructed_pixels)}"
    )
    print(
        "c1_enabled="
        f"{bool(context.app.reconstruction.enable_laser_ray_correction)}"
    )
    print(f"formal_frozen_exists={paths['frozen'].exists()}")
    print(f"invalid_attempt_exists={paths['invalid_attempt'].exists()}")
    print("height_computation=NOT_RUN")
    print("self_test=PASS")


def show_startup_error(message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Thermal ROI GUI startup error", message)
        root.destroy()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.output_dir / "thermal_roi_manual_gui.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        context = prepare_context(args)
        if args.self_test:
            self_test(context)
            return 0
        paths = support_paths(context.args.output_dir)
        if paths["frozen"].exists():
            raise HumanRoiError(
                "A formal frozen registry already exists. The GUI refuses to "
                f"overwrite it: {paths['frozen']}"
            )
        launch_gui(context)
        return 0
    except Exception as error:  # noqa: BLE001
        logging.exception("Thermal manual ROI GUI failed")
        if args.self_test:
            print(f"self_test=FAIL: {error}", file=sys.stderr)
        else:
            show_startup_error(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
