#!/usr/bin/env python3
"""Generate geometry-only multi-object ROI V2 candidates for Thermal-A2a.

This command intentionally stops before a frozen registry is created.  It
reuses the repository Auto ROI V2 detector on the first reliable 0827
recording, using only Frozen Steger centerlines and image-v geometry.  It does
not load ``height_shadow.csv``, reconstruct 3-D points, or evaluate height or
thermal-drift results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = ROOT / "laser_measurement_tool"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from app_config import load_app_config  # noqa: E402
from laser.backends import create_extraction_params  # noqa: E402
from laser.laser_extractor import extract_laser_center  # noqa: E402
from utils.image_io import load_grayscale_image  # noqa: E402

import auto_roi_v2_session01 as roi_v2  # noqa: E402


LOCAL_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
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
# User-confirmed physical placement for this experiment: top -> bottom is
# 20 mm, 30 mm, 10 mm.  The detector itself only assigns v-order; this tuple
# is metadata applied after geometry-only selection.
HEIGHT_LABEL_HINTS = ("20mm", "30mm", "10mm")
COLORS = {"upper": "#386cb0", "middle": "#f0027f", "lower": "#1b9e77"}


class RoiCandidateError(RuntimeError):
    """Raised when the geometry-only candidate run cannot be audited."""


@dataclass(frozen=True, slots=True)
class SourceFrame:
    recording_id: str
    recording_path: Path
    row_index: int
    filename: str
    image_path: Path
    image: np.ndarray
    camera_frame_number: int
    camera_timestamp_ticks: int | None
    host_timestamp_ns: int
    host_monotonic_ns: int
    frame_gap: int
    exposure_us: float
    gain_db: float
    pixel_format: str
    offset_x: int
    offset_y: int
    width: int
    height: int
    centers_uv_full: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_input = (
        ROOT
        / "laser_measurement_tool"
        / "output_daheng_0811"
        / "online_recordings"
        / "0827上午热漂_2000"
    )
    default_output = ROOT / "projects" / "daheng" / "analysis" / "thermal_a2a_roi_v2_0827"
    default_parameters = (
        ROOT
        / "reports"
        / "experiments"
        / "daheng_0822"
        / "session01_roi_freeze"
        / "auto_roi_v2_parameters.json"
    )
    default_a1_index = (
        ROOT
        / "projects"
        / "daheng"
        / "analysis"
        / "thermal_a1_0827"
        / "thermal_a1_recording_index.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--measure-config",
        type=Path,
        default=ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml",
    )
    parser.add_argument("--parameters", type=Path, default=default_parameters)
    parser.add_argument("--a1-index", type=Path, default=default_a1_index)
    return parser.parse_args(argv)


def as_int(value: Any, name: str) -> int:
    text = str(value).strip()
    if not text:
        raise RoiCandidateError(f"{name} is blank")
    try:
        number = float(text)
    except (TypeError, ValueError) as error:
        raise RoiCandidateError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(number) or number != int(number):
        raise RoiCandidateError(f"{name} is not an integer: {value!r}")
    return int(number)


def as_float(value: Any, name: str) -> float:
    text = str(value).strip()
    if not text:
        raise RoiCandidateError(f"{name} is blank")
    try:
        number = float(text)
    except (TypeError, ValueError) as error:
        raise RoiCandidateError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise RoiCandidateError(f"{name} is not finite: {value!r}")
    return number


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return "" if not math.isfinite(number) else f"{number:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fieldnames})


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_time(host_timestamp_ns: int) -> str:
    return datetime.fromtimestamp(host_timestamp_ns / 1e9, tz=timezone.utc).astimezone(LOCAL_TZ).isoformat(
        timespec="milliseconds"
    )


def read_csv_exact(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise RoiCandidateError(f"missing source table: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != expected_fields:
            raise RoiCandidateError(
                f"schema mismatch in {path}: expected {expected_fields}, got {header}"
            )
        rows: list[dict[str, str]] = []
        for line_number, values in enumerate(reader, start=2):
            if not values or all(not str(value).strip() for value in values):
                raise RoiCandidateError(f"blank row in {path}:{line_number}")
            if len(values) != len(header):
                raise RoiCandidateError(f"column count mismatch in {path}:{line_number}")
            rows.append({field: str(values[index]).strip() for index, field in enumerate(header)})
    return rows


def load_a1_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return [dict(row) for row in reader if any(str(value or "").strip() for value in row.values())]


def choose_first_recording(input_dir: Path, a1_index: Path) -> tuple[str, Path, dict[str, str]]:
    a1_rows = load_a1_rows(a1_index)
    for row in a1_rows:
        if str(row.get("raw_recording_integrity", "")).upper() != "PASS":
            continue
        recording_id = str(row.get("recording_id", "")).strip()
        candidate = input_dir / recording_id
        if candidate.is_dir():
            return recording_id, candidate, row
    candidates = sorted(path for path in input_dir.glob("recording_*") if path.is_dir())
    if not candidates:
        raise RoiCandidateError(f"no recording_* directory under {input_dir}")
    return candidates[0].name, candidates[0], {}


def load_source_frames(recording_id: str, recording_path: Path, extraction_params: Any) -> list[SourceFrame]:
    rows = read_csv_exact(recording_path / "frames.csv", FRAME_FIELDS)
    frames: list[SourceFrame] = []
    for row_index, row in enumerate(rows, start=1):
        filename = row["filename"]
        image_path = recording_path / filename
        image = load_grayscale_image(image_path)
        width = as_int(row["width"], "width")
        height = as_int(row["height"], "height")
        if tuple(image.shape) != (height, width):
            raise RoiCandidateError(
                f"image shape mismatch for {image_path.name}: {image.shape} != {(height, width)}"
            )
        offset_x = as_int(row["offset_x"], "offset_x")
        offset_y = as_int(row["offset_y"], "offset_y")
        centers_local = extract_laser_center(
            image,
            extraction_params,
            image_offset=(offset_x, offset_y),
        )
        centers_full = np.ascontiguousarray(centers_local, dtype=np.float64).copy()
        if centers_full.size:
            if centers_full.ndim != 2 or centers_full.shape[1] != 2:
                raise RoiCandidateError(f"invalid centerline shape for {image_path.name}: {centers_full.shape}")
            centers_full[:, 0] += offset_x
            centers_full[:, 1] += offset_y
        frames.append(
            SourceFrame(
                recording_id=recording_id,
                recording_path=recording_path,
                row_index=row_index,
                filename=filename,
                image_path=image_path,
                image=image,
                camera_frame_number=as_int(row["camera_frame_number"], "camera_frame_number"),
                camera_timestamp_ticks=(
                    None
                    if not row["camera_timestamp_ticks"]
                    else as_int(row["camera_timestamp_ticks"], "camera_timestamp_ticks")
                ),
                host_timestamp_ns=as_int(row["host_timestamp_ns"], "host_timestamp_ns"),
                host_monotonic_ns=as_int(row["host_monotonic_ns"], "host_monotonic_ns"),
                frame_gap=as_int(row["frame_gap"], "frame_gap"),
                exposure_us=as_float(row["exposure_us"], "exposure_us"),
                gain_db=as_float(row["gain_db"], "gain_db"),
                pixel_format=row["pixel_format"],
                offset_x=offset_x,
                offset_y=offset_y,
                width=width,
                height=height,
                centers_uv_full=centers_full,
            )
        )
    if not frames:
        raise RoiCandidateError(f"no frames in {recording_path / 'frames.csv'}")
    return frames


def median_centerline(frame_arrays: list[np.ndarray]) -> np.ndarray:
    by_v: dict[int, list[float]] = {}
    for centers in frame_arrays:
        values = np.asarray(centers, dtype=np.float64)
        if values.size == 0:
            continue
        finite = values[np.isfinite(values).all(axis=1)]
        for u, v in finite:
            by_v.setdefault(int(round(float(v))), []).append(float(u))
    points = [
        (float(np.median(by_v[v])), float(v))
        for v in sorted(by_v)
        if 0 <= v < roi_v2.FULL_SENSOR_HEIGHT and by_v[v]
    ]
    if len(points) < 50:
        raise RoiCandidateError(f"median centerline has too few valid rows: {len(points)}")
    return np.asarray(points, dtype=np.float64)


def unique_reasons(reasons: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason and reason not in seen:
            result.append(reason)
            seen.add(reason)
    return result


def assess_pair(pair: dict[str, Any], frame_arrays: list[np.ndarray]) -> dict[str, Any]:
    item = dict(pair)
    item["pair_gate_reasons"] = list(pair.get("pair_gate_reasons", []))
    item["height_support"] = roi_v2.support_stats(
        frame_arrays,
        item["height_v_range"],
        int(roi_v2.PARAMETERS["height_support"]["minimum_points_per_repeat"]),
        float(roi_v2.PARAMETERS["height_support"]["minimum_support_fraction"]),
    )
    for side, index in (("before", 0), ("after", 1)):
        item[f"{side}_support"] = roi_v2.support_stats(
            frame_arrays,
            item["baseline_v_ranges"][index],
            int(roi_v2.PARAMETERS["baseline"]["minimum_points_per_repeat"]),
            float(roi_v2.PARAMETERS["baseline"]["minimum_support_fraction"]),
        )
    reasons = list(item["pair_gate_reasons"])
    if not item["height_support"]["support_ok"]:
        reasons.append("height_20_repeat_formal_support_insufficient")
    if not item["before_support"]["support_ok"]:
        reasons.append("baseline_before_20_repeat_support_insufficient")
    if not item["after_support"]["support_ok"]:
        reasons.append("baseline_after_20_repeat_support_insufficient")
    clipped = item.get("baseline_clipped", {})
    if clipped.get("before") or clipped.get("after"):
        reasons.append("baseline_clipped_at_sensor_boundary")
    before, after = item["baseline_v_ranges"]
    height = item["height_v_range"]
    if not before or not after:
        reasons.append("baseline_range_unavailable")
    else:
        if before[1] >= height[0] or height[1] >= after[0]:
            reasons.append("baseline_overlaps_height")
    item["multi_geometry_reasons"] = unique_reasons(reasons)
    item["multi_geometry_ok"] = not item["multi_geometry_reasons"]
    item["auto_qc_status"] = "PASS" if item["multi_geometry_ok"] else "UNCERTAIN"
    return item


def intervals_overlap(first: list[int], second: list[int]) -> bool:
    return bool(first and second and max(first[0], second[0]) <= min(first[1], second[1]))


def candidate_footprint(candidate: dict[str, Any]) -> list[list[int]]:
    return [
        list(candidate["height_v_range"]),
        list(candidate["baseline_v_ranges"][0]),
        list(candidate["baseline_v_ranges"][1]),
    ]


def candidates_conflict(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return any(
        intervals_overlap(left, right)
        for left in candidate_footprint(first)
        for right in candidate_footprint(second)
    )


def select_three_objects(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    geometry_candidates = [item for item in candidates if item.get("edge_pair_geometry_ok")]
    combinations: list[tuple[tuple[int, int, int], tuple[dict[str, Any], ...]]] = []
    for combo in itertools.combinations(geometry_candidates, 3):
        if any(candidates_conflict(left, right) for left, right in itertools.combinations(combo, 2)):
            continue
        score = (
            sum(bool(item.get("multi_geometry_ok")) for item in combo),
            -sum(len(item.get("multi_geometry_reasons", [])) for item in combo),
            sum(float(item.get("pair_score") or 0.0) for item in combo),
        )
        combinations.append((score, combo))
    if not combinations:
        return []
    combinations.sort(key=lambda item: item[0], reverse=True)
    selected = sorted(combinations[0][1], key=lambda item: int(item["edge1_v"]))
    return selected


def build_candidates(
    frames: list[SourceFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame_arrays = [frame.centers_uv_full for frame in frames]
    median = median_centerline(frame_arrays)
    _, raw, interpolated = roi_v2.integer_profile(median)
    pairs, detector = roi_v2.build_edge_pairs(raw, interpolated)
    assessed = [assess_pair(pair, frame_arrays) for pair in pairs]
    for rank, item in enumerate(assessed, start=1):
        item["candidate_rank"] = rank
    selected = select_three_objects(assessed)
    for order, item in enumerate(selected, start=1):
        item["object_order"] = order
        item["object_id"] = OBJECT_IDS[order - 1]
        item["height_label_hint"] = HEIGHT_LABEL_HINTS[order - 1]
    detector = dict(detector)
    detector["selected_object_count"] = len(selected)
    detector["selected_object_ids"] = [item["object_id"] for item in selected]
    return median, raw, interpolated, assessed, selected, detector


def half_open(interval: list[int]) -> list[int]:
    return [int(interval[0]), int(interval[1]) + 1] if interval else []


def object_registry_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": item["object_id"],
        "object_order": item["object_order"],
        "height_label_hint": item["height_label_hint"],
        "height_label_basis": "v_order_only; pending human confirmation; not used by detector",
        "automatic_candidate_rank": item["candidate_rank"],
        "edge_pair": {
            "orientation": item["orientation"],
            "edge1_v": item["edge1_v"],
            "edge2_v": item["edge2_v"],
            "transition_v_ranges": item["transition_v_ranges"],
        },
        "transition_exclusion_margin_px": item["transition_exclusion_margin_px"],
        "height_v_range": item["height_v_range"],
        "height_v_range_half_open": half_open(item["height_v_range"]),
        "baseline_v_ranges": item["baseline_v_ranges"],
        "baseline_v_ranges_half_open": [half_open(interval) for interval in item["baseline_v_ranges"]],
        "baseline_safety_gap_px": roi_v2.PARAMETERS["baseline"]["safety_gap_px"],
        "baseline_clipped": item["baseline_clipped"],
        "auto_qc_status": item["auto_qc_status"],
        "auto_qc_reasons": item["multi_geometry_reasons"],
        "geometry_only": True,
    }


def build_draft(
    args: argparse.Namespace,
    first_frame: SourceFrame,
    frames: list[SourceFrame],
    median: np.ndarray,
    assessed: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    detector: dict[str, Any],
    a1_row: dict[str, str],
    parameter_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "analysis_id": "thermal_a2a_roi_v2_0827",
        "status": "DRAFT_CANDIDATES_ONLY",
        "geometry_only": True,
        "human_review_required": True,
        "human_reviewed": False,
        "manual_decision": "PENDING",
        "frozen": False,
        "thermal_a2_roi_frozen": False,
        "forbidden_inputs": list(roi_v2.PARAMETERS["forbidden_inputs"]),
        "input": {
            "recording_root": str(args.input_dir.resolve()),
            "source_recording": first_frame.recording_id,
            "source_recording_path": str(first_frame.recording_path.resolve()),
            "source_frame_count": len(frames),
            "source_frame_files": [frame.filename for frame in frames],
            "a1_index_path": str(args.a1_index.resolve()),
            "a1_index_sha256": sha256_file(args.a1_index),
            "a1_raw_recording_integrity": a1_row.get("raw_recording_integrity"),
        },
        "algorithm": {
            "name": roi_v2.PARAMETERS["algorithm"],
            "source_script": str((ROOT / "tools" / "auto_roi_v2_session01.py").resolve()),
            "source_script_sha256": sha256_file(ROOT / "tools" / "auto_roi_v2_session01.py"),
            "parameters_path": str(args.parameters.resolve()),
            "parameters_sha256": sha256_file(args.parameters),
            "parameters": parameter_payload,
            "extraction_method": "steger",
            "selection_basis": "first reliable recording median Frozen Steger u(v), image-v geometry only",
        },
        "detector_summary": detector,
        "median_centerline_summary": {
            "point_count": int(len(median)),
            "v_min": float(np.min(median[:, 1])),
            "v_max": float(np.max(median[:, 1])),
            "u_full_min": float(np.min(median[:, 0])),
            "u_full_max": float(np.max(median[:, 0])),
            "first_frame_number": first_frame.camera_frame_number,
            "first_frame_host_time_local": local_time(first_frame.host_timestamp_ns),
        },
        "candidates": assessed,
        "selected_objects": [object_registry_entry(item) for item in selected],
        "automatic_conclusions": {
            "three_objects_detected": len(selected) == 3,
            "all_height_rois_valid": bool(selected) and len(selected) == 3 and all(
                item["multi_geometry_ok"] for item in selected
            ),
            "all_local_baselines_both_sides": bool(selected) and len(selected) == 3 and all(
                bool(item["multi_geometry_ok"])
                and not item["baseline_clipped"]["before"]
                and not item["baseline_clipped"]["after"]
                for item in selected
            ),
            "human_review_required": True,
            "thermal_a2_roi_frozen": False,
        },
        "no_height_or_3d_input_used": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def qc_rows(
    selected: list[dict[str, Any]],
    assessed: list[dict[str, Any]],
    first_frame: SourceFrame,
    frames: list[SourceFrame],
    a1_row: dict[str, str],
) -> list[dict[str, Any]]:
    all_three = len(selected) == 3
    all_valid = all_three and all(item["multi_geometry_ok"] for item in selected)
    both_sides = all_three and all(
        item["multi_geometry_ok"]
        and not item["baseline_clipped"]["before"]
        and not item["baseline_clipped"]["after"]
        for item in selected
    )
    pairwise_nonoverlap = all(
        not candidates_conflict(left, right)
        for left, right in itertools.combinations(selected, 2)
    ) if selected else False
    rows: list[dict[str, Any]] = [
        {
            "scope": "global",
            "object_id": "",
            "check_id": "source_recording",
            "severity": "INFO",
            "status": "PASS" if a1_row.get("raw_recording_integrity", "PASS") == "PASS" else "REVIEW",
            "observed": f"{first_frame.recording_id}; frames={len(frames)}",
            "expected": "first A1 raw_recording_integrity=PASS; 20 frames",
            "evidence": str(first_frame.recording_path / "frames.csv"),
            "notes": "A1 index is reused only to choose the first reliable source recording.",
        },
        {
            "scope": "global",
            "object_id": "",
            "check_id": "geometry_only",
            "severity": "INFO",
            "status": "PASS",
            "observed": "Frozen Steger centerline + image-v geometry",
            "expected": "no nominal height, Z, shadow, residual, or thermal result",
            "evidence": "thermal_a2a_roi_v2.py; auto_roi_v2_parameters.json",
            "notes": "No height_shadow.csv is opened by this command.",
        },
        {
            "scope": "global",
            "object_id": "",
            "check_id": "three_objects_detected",
            "severity": "REVIEW",
            "status": "PASS" if all_three else "FAIL",
            "observed": f"selected_objects={len(selected)}; all_edge_pairs={len(assessed)}",
            "expected": "3 non-conflicting edge pairs",
            "evidence": "thermal_roi_v2_candidates.json",
            "notes": "Object IDs are assigned by ascending v only.",
        },
        {
            "scope": "global",
            "object_id": "",
            "check_id": "all_height_rois_valid",
            "severity": "REVIEW",
            "status": "PASS" if all_valid else "FAIL",
            "observed": f"{sum(item['multi_geometry_ok'] for item in selected)}/{len(selected)} selected candidates pass geometry/support gates",
            "expected": "3/3 pass V2 geometry/support gates",
            "evidence": "thermal_roi_v2_candidates.json",
            "notes": "This is an automatic candidate gate, not a freeze decision.",
        },
        {
            "scope": "global",
            "object_id": "",
            "check_id": "all_local_baselines_both_sides",
            "severity": "REVIEW",
            "status": "PASS" if both_sides else "FAIL",
            "observed": f"{sum(bool(item['baseline_v_ranges'][0]) and bool(item['baseline_v_ranges'][1]) for item in selected)}/{len(selected)} have two non-clipped ranges",
            "expected": "3/3 non-clipped before/after Ground ranges",
            "evidence": "thermal_roi_v2_candidates.json",
            "notes": "Ground is defined only by image-v stable intervals here; no Z is used.",
        },
        {
            "scope": "global",
            "object_id": "",
            "check_id": "objects_non_overlapping",
            "severity": "REVIEW",
            "status": "PASS" if all_three and pairwise_nonoverlap else "FAIL",
            "observed": f"pairwise_nonoverlap={pairwise_nonoverlap}",
            "expected": "all selected height/baseline intervals non-overlapping",
            "evidence": "thermal_roi_v2_candidates.json",
            "notes": "Automatic conflict gate; human must still inspect overlay.",
        },
        {
            "scope": "global",
            "object_id": "",
            "check_id": "human_review_required",
            "severity": "REVIEW",
            "status": "PENDING",
            "observed": "draft only",
            "expected": "manual confirmation/micro-adjustment before freeze",
            "evidence": "report.md",
            "notes": "Automatic output is never a frozen registry.",
        },
        {
            "scope": "global",
            "object_id": "",
            "check_id": "thermal_a2_roi_frozen",
            "severity": "INFO",
            "status": "NO",
            "observed": "thermal_roi_registry_v2_frozen.json not generated",
            "expected": "NO before human review",
            "evidence": "thermal_roi_v2_registry_v2_draft.json",
            "notes": "Thermal-A2 height analysis is blocked until the frozen registry exists.",
        },
    ]
    for item in selected:
        for check_id, status, observed, expected, notes in (
            (
                "edge_pair_geometry",
                "PASS" if item["edge_pair_geometry_ok"] else "FAIL",
                f"width={item['object_width_px']:.3f}; step={item.get('step_amplitude_px')}",
                "edge pair width 50..180 px and polarity/stability gates",
                "Reused Auto ROI V2 pair gate.",
            ),
            (
                "height_interior_support",
                "PASS" if item["height_support"]["support_ok"] else "FAIL",
                f"min_points={item['height_support']['min_points']}; median_support={item['height_support']['median_support_fraction']:.4f}",
                "20 repeats; >=20 points/repeat; median support >=0.50",
                "Image-v support only.",
            ),
            (
                "baseline_before_support",
                "PASS" if item["before_support"]["support_ok"] else "FAIL",
                f"min_points={item['before_support']['min_points']}; median_support={item['before_support']['median_support_fraction']:.4f}",
                "20 repeats; >=20 points/repeat; median support >=0.25",
                "Image-v support only.",
            ),
            (
                "baseline_after_support",
                "PASS" if item["after_support"]["support_ok"] else "FAIL",
                f"min_points={item['after_support']['min_points']}; median_support={item['after_support']['median_support_fraction']:.4f}",
                "20 repeats; >=20 points/repeat; median support >=0.25",
                "Image-v support only.",
            ),
            (
                "baseline_non_clipped",
                "PASS"
                if not item["baseline_clipped"]["before"] and not item["baseline_clipped"]["after"]
                else "FAIL",
                json.dumps(item["baseline_clipped"], ensure_ascii=False),
                "before/after intervals do not touch v=0 or v=2999",
                "A boundary-touching interval is not a formal local baseline.",
            ),
            (
                "height_baseline_nonoverlap",
                "PASS"
                if not candidates_conflict(
                    {"height_v_range": item["height_v_range"], "baseline_v_ranges": [[], []]},
                    {"height_v_range": [], "baseline_v_ranges": item["baseline_v_ranges"]},
                )
                else "FAIL",
                f"height={item['height_v_range']}; baseline={item['baseline_v_ranges']}",
                "height interior separated from both baselines by safety gap",
                "Transition exclusion and baseline safety gap are fixed.",
            ),
        ):
            rows.append(
                {
                    "scope": "object",
                    "object_id": item["object_id"],
                    "check_id": check_id,
                    "severity": "REVIEW",
                    "status": status,
                    "observed": observed,
                    "expected": expected,
                    "evidence": "thermal_roi_v2_candidates.json",
                    "notes": notes,
                }
            )
    return rows


def save_overlay(
    path: Path,
    first_frame: SourceFrame,
    median: np.ndarray,
    raw: np.ndarray,
    interpolated: np.ndarray,
    selected: list[dict[str, Any]],
) -> None:
    v_grid = np.arange(len(interpolated), dtype=np.float64)
    fig, (ax_image, ax_profile) = plt.subplots(1, 2, figsize=(15, 11), constrained_layout=True)
    ax_image.imshow(first_frame.image, cmap="gray", origin="upper", aspect="auto")
    ax_image.plot(
        median[:, 0] - first_frame.offset_x,
        median[:, 1],
        color="white",
        linewidth=0.8,
        alpha=0.95,
        label="median Frozen Steger u(v)",
    )
    for item in selected:
        color = COLORS[item["object_id"]]
        start, end = item["height_v_range"]
        ax_image.axhspan(start, end, color=color, alpha=0.28)
        for base in item["baseline_v_ranges"]:
            if base:
                ax_image.axhspan(base[0], base[1], color=color, alpha=0.10)
        for edge in (item["edge1_v"], item["edge2_v"]):
            ax_image.axhline(edge, color=color, linestyle="--", linewidth=0.9)
        ax_image.text(
            4,
            (start + end) / 2,
            f"{item['object_id']} ({item['height_label_hint']})",
            color=color,
            fontsize=9,
            va="center",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
        )
    ax_image.set_title("First reliable raw PNG + frozen candidate bands")
    ax_image.set_xlabel("local ROI u (px)")
    ax_image.set_ylabel("v (px)")
    ax_image.set_xlim(0, first_frame.width)
    ax_image.set_ylim(first_frame.height - 1, 0)
    ax_image.legend(loc="lower right", fontsize=8)

    ax_profile.plot(
        interpolated - first_frame.offset_x,
        v_grid,
        color="#222222",
        linewidth=1.0,
        label="median u(v), interpolated for detector",
    )
    ax_profile.plot(
        median[:, 0] - first_frame.offset_x,
        median[:, 1],
        ".",
        color="#aaaaaa",
        markersize=1.0,
        alpha=0.35,
        label="median centerline samples",
    )
    for item in selected:
        color = COLORS[item["object_id"]]
        for edge in (item["edge1_v"], item["edge2_v"]):
            ax_profile.axhline(edge, color=color, linestyle="--", linewidth=0.9)
        start, end = item["height_v_range"]
        ax_profile.axhspan(start, end, color=color, alpha=0.20)
        for base in item["baseline_v_ranges"]:
            if base:
                ax_profile.axhspan(base[0], base[1], color=color, alpha=0.07)
    ax_profile.set_title("Auto ROI V2 edge / height / baseline candidate overlay")
    ax_profile.set_xlabel("local ROI u (px)")
    ax_profile.set_ylabel("v (px)")
    ax_profile.set_xlim(0, first_frame.width)
    ax_profile.set_ylim(first_frame.height - 1, 0)
    handles = [
        Line2D([0], [0], color="white", linewidth=1, label="median Frozen Steger"),
        Line2D([0], [0], color="#222222", linewidth=1, label="interpolated detector profile"),
    ] + [Patch(facecolor=COLORS[obj], alpha=0.3, label=f"{obj} / {label}") for obj, label in zip(OBJECT_IDS, HEIGHT_LABEL_HINTS)]
    ax_profile.legend(handles=handles, loc="lower right", fontsize=8)
    fig.suptitle(
        "Thermal-A2a Auto ROI V2 draft — geometry-only; NOT frozen; no Z/height input",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_report(
    path: Path,
    args: argparse.Namespace,
    frames: list[SourceFrame],
    first_frame: SourceFrame,
    assessed: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    detector: dict[str, Any],
    a1_row: dict[str, str],
    parameter_payload: dict[str, Any],
) -> None:
    all_valid = len(selected) == 3 and all(item["multi_geometry_ok"] for item in selected)
    both_sides = len(selected) == 3 and all(
        item["multi_geometry_ok"]
        and not item["baseline_clipped"]["before"]
        and not item["baseline_clipped"]["after"]
        for item in selected
    )
    pairwise_nonoverlap = len(selected) == 3 and all(
        not candidates_conflict(left, right)
        for left, right in itertools.combinations(selected, 2)
    )
    lines = [
        "# Thermal-A2a｜热漂实验 Multi-object ROI V2 自动候选与人工 Freeze",
        "",
        "> 本报告停在人工审核前。自动候选不是 Frozen registry；没有生成 `thermal_roi_registry_v2_frozen.json`。",
        "",
        "## 自动结论",
        "",
        f"- `THREE_OBJECTS_DETECTED = {'YES' if len(selected) == 3 else 'NO'}`（selected={len(selected)}；all edge pairs={len(assessed)}）",
        f"- `ALL_HEIGHT_ROIS_VALID = {'YES' if all_valid else 'NO'}`",
        f"- `ALL_LOCAL_BASELINES_BOTH_SIDES = {'YES' if both_sides else 'NO'}`",
        "- `HUMAN_REVIEW_REQUIRED = YES`",
        "- `THERMAL_A2_ROI_FROZEN = NO`",
        "",
        "## 处理边界",
        "",
        f"首个可靠 recording：`{first_frame.recording_id}`，20 帧，输入目录 `{first_frame.recording_path}`。A1 raw integrity：`{a1_row.get('raw_recording_integrity', 'UNKNOWN')}`。",
        f"本轮只调用正式 `{parameter_payload.get('algorithm', 'ground_object_top_ground_interval_v2')}` 的 Frozen Steger centerline 和 Auto ROI V2 image-v detector；中心线点数中位候选输入来自这 20 帧。",
        "没有读取 `height_shadow.csv`，没有使用 nominal height、C0/C1、Session/Local Z、Base/H1/H-B2、residual、thermal drift 或误差结果。",
        "三个对象按 v 从小到大命名为 upper/middle/lower；本实验的人工确认映射为 upper=20mm、middle=30mm、lower=10mm；这些标签不是 detector gate。",
        "",
        "## Candidate ROI（inclusive image-v；后续实现同时保留 half-open 表达）",
        "",
    ]
    if selected:
        lines.extend([
            "| object | label hint | edge1–edge2 | height interior | baseline before | baseline after | auto QC | reasons |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ])
        for item in selected:
            lines.append(
                f"| {item['object_id']} | {item['height_label_hint']} | {item['edge1_v']}–{item['edge2_v']} | "
                f"{item['height_v_range']} | {item['baseline_v_ranges'][0]} | {item['baseline_v_ranges'][1]} | "
                f"{item['auto_qc_status']} | {', '.join(item['multi_geometry_reasons']) or '—'} |"
            )
    else:
        lines.append("未选出三个互不冲突的候选对象；不能进入人工 Freeze。")
    lines.extend([
        "",
        f"组合 non-overlap 自动检查：`{'PASS' if pairwise_nonoverlap else 'FAIL'}`。检测器摘要：`{json.dumps(json_safe(detector), ensure_ascii=False)}`。",
        "",
        "## Provenance / reuse audit",
        "",
        "- 复用：既有 Auto ROI V2 的参数、transition exclusion、stable segment、双侧 baseline support、non-clipped 和 non-overlap 规则。",
        "- 本轮新增：对 0827 首个可靠 recording 的 20 张 PNG 重新运行 Frozen Steger，生成 median `u(v)`、multi-object candidates、QC 和 overlay。",
        "- 不复用：既有 0822/0824/0819 的具体 v 坐标、历史高度数值和 `height_shadow.csv`；它们不作为本轮 ROI 选择输入。",
        "",
        "## 人工审核动作",
        "",
        "请查看 `thermal_roi_v2_overlay.png`，逐一确认：",
        "",
        "1. upper/middle/lower 是否分别对应实验记录中的 20/30/10 mm；",
        "2. 每个 height interior 是否避开两个 transition；",
        "3. before/after 两侧 Ground 是否为稳定、完整、非 clipped 区间；",
        "4. 三组 height/baseline 区间是否互不重叠；必要时只对 draft registry 做人工微调。",
        "",
        "人工确认后，才可由人工流程生成唯一 `thermal_roi_registry_v2_frozen.json`，并将 `human_reviewed=true`、`frozen=true`、确认者和确认时间写入 `thermal_roi_v2_manual_review.md`。",
        "在该文件出现并通过审计前，Thermal-A2 高度精度分析禁止运行。",
        "",
        "## 输出",
        "",
        "- `thermal_roi_v2_candidates.json`：全部 edge-pair candidates 与 selected objects。",
        "- `thermal_roi_v2_qc.csv`：自动 QC，不是人工批准。",
        "- `thermal_roi_v2_overlay.png`：原始首帧、median centerline、edge、height、baseline overlay。",
        "- `thermal_roi_v2_registry_v2_draft.json`：draft registry，明确 `frozen=false`。",
        "- `report.md`：本报告。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.measure_config = args.measure_config.resolve()
    args.parameters = args.parameters.resolve()
    args.a1_index = args.a1_index.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        parameter_payload = json.loads(args.parameters.read_text(encoding="utf-8"))
        if not isinstance(parameter_payload, dict):
            raise RoiCandidateError("Auto ROI V2 parameters must be an object")
        if parameter_payload != json_safe(roi_v2.PARAMETERS):
            raise RoiCandidateError("Auto ROI V2 parameter JSON differs from source script PARAMETERS")
        app_config = load_app_config(args.measure_config)
        if app_config.extraction_method != "steger":
            raise RoiCandidateError(
                f"formal Thermal-A2a extraction must be steger, got {app_config.extraction_method!r}"
            )
        extraction_params = create_extraction_params(
            app_config.extraction_method,
            app_config.extraction_options_by_method.get(app_config.extraction_method, {}),
        )
        recording_id, recording_path, a1_row = choose_first_recording(args.input_dir, args.a1_index)
        frames = load_source_frames(recording_id, recording_path, extraction_params)
        first_frame = frames[0]
        median, raw, interpolated, assessed, selected, detector = build_candidates(frames)
        draft = build_draft(
            args,
            first_frame,
            frames,
            median,
            assessed,
            selected,
            detector,
            a1_row,
            parameter_payload,
        )
        candidate_path = args.output_dir / "thermal_roi_v2_candidates.json"
        draft_path = args.output_dir / "thermal_roi_v2_registry_v2_draft.json"
        qc_path = args.output_dir / "thermal_roi_v2_qc.csv"
        overlay_path = args.output_dir / "thermal_roi_v2_overlay.png"
        write_json(candidate_path, draft)
        write_json(
            draft_path,
            {
                "schema_version": 2,
                "registry_type": "thermal_multi_object_roi_v2",
                "status": "DRAFT",
                "geometry_only": True,
                "human_reviewed": False,
                "manual_decision": "PENDING",
                "frozen": False,
                "thermal_a2_roi_frozen": False,
                "source_recording": recording_id,
                "source_frame_count": len(frames),
                "selection_basis": "first reliable recording median Frozen Steger u(v), image-v geometry only",
                "objects": [object_registry_entry(item) for item in selected],
                "review_required": [
                    "confirm upper/middle/lower identity against physical 10/20/30 mm blocks",
                    "confirm transition exclusion and stable Ground on both sides",
                    "confirm non-overlap and non-clipped intervals",
                ],
            },
        )
        write_csv(
            qc_path,
            ["scope", "object_id", "check_id", "severity", "status", "observed", "expected", "evidence", "notes"],
            qc_rows(selected, assessed, first_frame, frames, a1_row),
        )
        save_overlay(overlay_path, first_frame, median, raw, interpolated, selected)
        render_report(
            args.output_dir / "report.md",
            args,
            frames,
            first_frame,
            assessed,
            selected,
            detector,
            a1_row,
            parameter_payload,
        )
        print(f"Thermal-A2a source recording: {recording_id}")
        print(f"Frames: {len(frames)}; edge candidates: {len(assessed)}; selected objects: {len(selected)}")
        print(f"Output: {args.output_dir}")
        print("THERMAL_A2_ROI_FROZEN = NO")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"thermal_a2a_roi_v2: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
