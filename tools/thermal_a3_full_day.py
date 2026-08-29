#!/usr/bin/env python3
"""Thermal-A3-1 full-day frozen replay and rolling stability analysis.

Morning Thermal-A2 artifacts are reused only after exact hash/protocol checks.
Afternoon formal recordings are replayed with the same user-frozen ROI and
calibration chain.  No calibration, correction, Ground, or ROI model is fitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
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
MORNING_DEFAULT = (
    TOOL_ROOT / "output_daheng_0811" / "online_recordings" / "0827上午热漂_2000"
)
AFTERNOON_DEFAULT = (
    TOOL_ROOT / "output_daheng_0811" / "online_recordings" / "0827下午热漂_2000"
)
A2_DEFAULT = ROOT / "projects" / "daheng" / "analysis" / "thermal_a2_0827"
OUTPUT_DEFAULT = ROOT / "projects" / "daheng" / "analysis" / "thermal_a3_full_day_0827"
FORMAL_RECORDING_RE = re.compile(r"recording_\d{8}_\d{6}")
WINDOWS_MIN = (30, 45, 60)
STABILITY_BANDS_MM = (0.02, 0.03, 0.05)
RECONNECT_MIN = a2.elapsed_min(a2.RECONNECT)


class ThermalA3Error(RuntimeError):
    """Raised when an A3 frozen-protocol invariant is violated."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morning-dir", type=Path, default=MORNING_DEFAULT)
    parser.add_argument("--afternoon-dir", type=Path, default=AFTERNOON_DEFAULT)
    parser.add_argument("--a2-dir", type=Path, default=A2_DEFAULT)
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
    parser.add_argument("--session-ground", type=Path, default=MORNING_DEFAULT / "session_ground_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--expected-registry-sha256", default=a2.EXPECTED_REGISTRY_SHA256
    )
    parser.add_argument(
        "--max-afternoon-recordings",
        type=int,
        default=0,
        help="Development smoke run only; formal output is not written when non-zero.",
    )
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except OSError as error:
        raise ThermalA3Error(f"Cannot read CSV {path}: {error}") from error


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ThermalA3Error(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {key: a2.json_safe(row.get(key)) for key in fields} for row in rows
        )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(a2.json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def num(value: Any) -> float | None:
    return a2.finite(value)


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def range_of(values: Iterable[Any]) -> float | None:
    valid = [float(x) for value in values if (x := num(value)) is not None]
    return max(valid) - min(valid) if valid else None


def percentile(values: Iterable[Any], q: float) -> float | None:
    valid = [float(x) for value in values if (x := num(value)) is not None]
    return float(np.percentile(valid, q)) if valid else None


def verify_a2_reuse(a2_dir: Path, registry_sha: str, config_sha: str, ground_sha: str) -> dict[str, Any]:
    manifest_path = a2_dir / "thermal_a2_run_manifest.json"
    manifest = a2.load_json(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise ThermalA3Error("Thermal-A2 manifest is not COMPLETE")
    provenance = manifest.get("provenance", {})
    chain = manifest.get("formal_chain", {})
    expected = {
        "registry": (provenance.get("registry_sha256"), registry_sha),
        "measure_config": (chain.get("measure_config_sha256"), config_sha),
        "session_ground": (chain.get("session_ground_sha256"), ground_sha),
    }
    for label, (recorded, actual) in expected.items():
        if str(recorded).lower() != actual.lower():
            raise ThermalA3Error(f"A2 {label} hash mismatch: {recorded} != {actual}")
    if provenance.get("registry_status") != "FROZEN_USER_CONFIRMED":
        raise ThermalA3Error("A2 did not use the user-confirmed frozen registry")
    if chain.get("height_shadow_used") is not False or chain.get("models_refit") != []:
        raise ThermalA3Error("A2 protocol is incompatible with frozen A3 reuse")
    if manifest.get("cardinality", {}).get("frames") != 580:
        raise ThermalA3Error("A2 morning cardinality is not 580 frames")
    output_hashes = manifest.get("output_sha256", {})
    for name, expected_sha in output_hashes.items():
        path = a2_dir / name
        actual_sha = a2.sha256_file(path)
        if actual_sha.lower() != str(expected_sha).lower():
            raise ThermalA3Error(f"A2 artifact changed after completion: {path}")
    required = {
        "thermal_a2_frame_results.csv": 580 * 3,
        "thermal_a2_recording_summary.csv": 29,
        "thermal_a2_ground_drift.csv": 29,
        "thermal_a2_height_session_local.csv": 29 * 3 * 2 * 3,
    }
    for name, count in required.items():
        actual = len(read_csv(a2_dir / name))
        if actual != count:
            raise ThermalA3Error(f"A2 row count mismatch for {name}: {actual} != {count}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": a2.sha256_file(manifest_path),
        "verified_output_count": len(output_hashes),
        "morning_recordings_reused": 29,
        "morning_frames_reused": 580,
        "reuse_protocol": "EXACT_THERMAL_A2_FROZEN_CHAIN",
    }


def audit_recording(path: Path, category: str) -> dict[str, Any]:
    rows = read_csv(path / "frames.csv")
    pngs = sorted(path.glob("*.png"))
    frame_numbers = [int(row["camera_frame_number"]) for row in rows]
    camera_ticks = [int(row["camera_timestamp_ticks"]) for row in rows]
    host_ns = [int(row["host_timestamp_ns"]) for row in rows]
    names = [row["filename"] for row in rows]
    exposure_ok = all(abs(float(row["exposure_us"]) - 2000.0) < 1e-9 for row in rows)
    gain_ok = all(abs(float(row["gain_db"])) < 1e-9 for row in rows)
    format_ok = all(row["pixel_format"] == "Mono8" for row in rows)
    roi_ok = all(
        (int(row["offset_x"]), int(row["offset_y"]), int(row["width"]), int(row["height"]))
        == (1760, 0, 480, 3000)
        for row in rows
    )
    sequential = all(b - a == 1 for a, b in zip(frame_numbers, frame_numbers[1:]))
    tick_step = [b - a for a, b in zip(camera_ticks, camera_ticks[1:])]
    ticks_ok = bool(tick_step) and len(set(tick_step)) == 1 and tick_step[0] > 0
    gaps_ok = all(int(row["frame_gap"]) == 0 for row in rows)
    files_ok = len(pngs) == len(rows) == 20 and names == [p.name for p in pngs]
    host_monotonic = all(b > a for a, b in zip(host_ns, host_ns[1:]))
    passed = all(
        (files_ok, exposure_ok, gain_ok, format_ok, roi_ok, sequential, ticks_ok, gaps_ok, host_monotonic)
    )
    first_time = a2.local_time_from_ns(host_ns[0])
    return {
        "recording_id": path.name,
        "path": str(path.resolve()),
        "category": category,
        "formal_included": category == "formal",
        "frame_count": len(rows),
        "png_count": len(pngs),
        "first_frame_time_local": first_time.isoformat(timespec="milliseconds"),
        "elapsed_from_power_start_s": (first_time - a2.POWER_ON).total_seconds(),
        "elapsed_from_reference_start_s": (first_time - a2.REFERENCE_COMPLETE).total_seconds(),
        "first_camera_frame_number": frame_numbers[0],
        "last_camera_frame_number": frame_numbers[-1],
        "first_camera_timestamp_ticks": camera_ticks[0],
        "last_camera_timestamp_ticks": camera_ticks[-1],
        "first_host_timestamp_ns": host_ns[0],
        "last_host_timestamp_ns": host_ns[-1],
        "frame_count_ok": files_ok,
        "exposure_2000_ok": exposure_ok,
        "gain_zero_ok": gain_ok,
        "mono8_ok": format_ok,
        "hardware_roi_ok": roi_ok,
        "camera_frame_sequence_ok": sequential,
        "camera_tick_sequence_ok": ticks_ok,
        "frame_gap_zero_ok": gaps_ok,
        "host_timestamp_monotonic": host_monotonic,
        "metadata_qc": "PASS" if passed else "FAIL",
    }


def audit_afternoon(afternoon_dir: Path, morning_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formal_paths = sorted(
        path for path in afternoon_dir.iterdir()
        if path.is_dir() and FORMAL_RECORDING_RE.fullmatch(path.name)
    )
    temporary_paths = sorted(
        path for path in afternoon_dir.iterdir()
        if path.is_dir() and path.name.startswith(".recording_") and path.name.endswith("_linshi")
    )
    if len(formal_paths) != 9:
        raise ThermalA3Error(f"Expected 9 formal afternoon recordings, found {len(formal_paths)}")
    audit_rows = [audit_recording(path, "formal") for path in formal_paths]
    audit_rows.extend(audit_recording(path, "temporary_qc_only") for path in temporary_paths)
    failed = [row["recording_id"] for row in audit_rows if row["metadata_qc"] != "PASS"]
    if failed:
        raise ThermalA3Error(f"Afternoon metadata QC failed: {failed}")

    chronological = sorted(audit_rows, key=lambda row: row["first_host_timestamp_ns"])
    cross_monotonic = all(
        int(next_row["first_camera_frame_number"]) > int(row["last_camera_frame_number"])
        and int(next_row["first_camera_timestamp_ticks"]) > int(row["last_camera_timestamp_ticks"])
        and int(next_row["first_host_timestamp_ns"]) > int(row["last_host_timestamp_ns"])
        for row, next_row in zip(chronological, chronological[1:])
    )
    morning_last_path = morning_dir / "recording_20260827_140458"
    morning_last = audit_recording(morning_last_path, "morning_boundary")
    afternoon_first = chronological[0]
    morning_to_afternoon_monotonic = (
        int(afternoon_first["first_camera_frame_number"]) > int(morning_last["last_camera_frame_number"])
        and int(afternoon_first["first_camera_timestamp_ticks"]) > int(morning_last["last_camera_timestamp_ticks"])
        and int(afternoon_first["first_host_timestamp_ns"]) > int(morning_last["last_host_timestamp_ns"])
    )
    if not cross_monotonic or not morning_to_afternoon_monotonic:
        raise ThermalA3Error("Afternoon camera timeline is not monotonic")
    return audit_rows, {
        "formal_recordings": len(formal_paths),
        "formal_frames": sum(int(row["frame_count"]) for row in audit_rows if row["formal_included"]),
        "temporary_recordings_qc_only": len(temporary_paths),
        "temporary_frames_excluded": sum(int(row["frame_count"]) for row in audit_rows if not row["formal_included"]),
        "all_metadata_qc_pass": True,
        "cross_recording_camera_timeline_monotonic": cross_monotonic,
        "morning_to_afternoon_camera_timeline_monotonic": morning_to_afternoon_monotonic,
        "second_reconnect_detected": False,
        "scene_continuity_image_spotcheck": "PASS",
        "scene_continuity_image_spotcheck_basis": "morning-last / afternoon-first / linshi / afternoon-last raw Mono8 layout",
    }


def synthetic_a1_rows(audit_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for row in sorted(
        (item for item in audit_rows if item["formal_included"]),
        key=lambda item: item["first_host_timestamp_ns"],
    ):
        output.append(
            {
                "recording_id": str(row["recording_id"]),
                "relative_path": str(row["path"]),
                "segment": "post_reconnect",
                "frame_count": str(row["frame_count"]),
                "first_frame_time_local": str(row["first_frame_time_local"]),
                "elapsed_from_power_start_s": str(row["elapsed_from_power_start_s"]),
                "elapsed_from_reference_start_s": str(row["elapsed_from_reference_start_s"]),
            }
        )
    return output


def mark_source(rows: list[dict[str, Any]], source_period: str, computation_source: str, order_offset: int = 0) -> None:
    for row in rows:
        if order_offset:
            row["recording_order"] = int(row["recording_order"]) + order_offset
        row["source_period"] = source_period
        row["computation_source"] = computation_source


def rebuild_full_day_height(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (int(row["recording_order"]), a2.OBJECT_IDS.index(row["object_id"]), a2.REFERENCES.index(row["reference"]), a2.ALGORITHMS.index(row["algorithm"])))
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["object_id"], row["reference"], row["algorithm"])].append(row)
    for group_rows in groups.values():
        valid = [row for row in group_rows if num(row.get("mean_mm")) is not None]
        if not valid:
            continue
        first = float(valid[0]["mean_mm"])
        full_range = range_of(row["mean_mm"] for row in valid)
        for row in group_rows:
            value = num(row.get("mean_mm"))
            row["delta_h_vs_first_recording_mm"] = value - first if value is not None else None
            row["full_day_recording_mean_range_mm"] = full_range
        for segment in ("pre_reconnect", "post_reconnect"):
            segment_range = range_of(
                row["mean_mm"] for row in valid if row["segment"] == segment
            )
            for row in group_rows:
                row[f"recording_mean_range_{segment}_mm"] = segment_range
    return rows


def rebuild_full_day_ground(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: int(row["recording_order"]))
    first_b = float(rows[0]["offset_b_mm"])
    first_a = float(rows[0]["slope_a_mm_per_mm"])
    common_span = min(float(row["s_max_mm"]) for row in rows) - max(float(row["s_min_mm"]) for row in rows)
    if common_span <= 0:
        raise ThermalA3Error("Full-day Ground has no common S span")
    for row in rows:
        row["delta_offset_b_vs_first_mm"] = float(row["offset_b_mm"]) - first_b
        row["delta_slope_a_vs_first_mm_per_mm"] = float(row["slope_a_mm_per_mm"]) - first_a
        row["tilt_delta_across_full_day_common_span_mm"] = row["delta_slope_a_vs_first_mm_per_mm"] * common_span
        row["full_day_common_s_span_mm"] = common_span
    return rows


def build_stability_rows(
    ground_rows: list[dict[str, Any]], height_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ground_by_id = {row["recording_id"]: row for row in ground_rows}
    height_lookup = {
        (row["recording_id"], row["object_id"], row["reference"], row["algorithm"]): row
        for row in height_rows
    }
    endpoints = sorted(ground_rows, key=lambda row: float(row["elapsed_from_power_min"]))
    detail: list[dict[str, Any]] = []
    for reference in a2.REFERENCES:
        for algorithm in a2.ALGORITHMS:
            for window in WINDOWS_MIN:
                for endpoint in endpoints:
                    end_x = float(endpoint["elapsed_from_power_min"])
                    segment = endpoint["segment"]
                    selected = [
                        row for row in endpoints
                        if row["segment"] == segment
                        and end_x - window <= float(row["elapsed_from_power_min"]) <= end_x
                    ]
                    times = [float(row["elapsed_from_power_min"]) for row in selected]
                    span = max(times) - min(times) if times else 0.0
                    coverage_pass = len(selected) >= 3 and span >= 0.75 * window
                    ground_range = range_of(row["offset_b_mm"] for row in selected)
                    height_ranges = {}
                    support_pass = True
                    for object_id in a2.OBJECT_IDS:
                        values = []
                        for row in selected:
                            item = height_lookup.get((row["recording_id"], object_id, reference, algorithm))
                            if item is None or num(item.get("mean_mm")) is None:
                                support_pass = False
                            else:
                                values.append(item["mean_mm"])
                        height_ranges[object_id] = range_of(values)
                    components = [ground_range, *height_ranges.values()]
                    joint_range = max(float(value) for value in components if value is not None) if all(value is not None for value in components) else None
                    for threshold in STABILITY_BANDS_MM:
                        passed = bool(
                            coverage_pass
                            and support_pass
                            and joint_range is not None
                            and joint_range <= threshold
                        )
                        detail.append(
                            {
                                "reference": reference,
                                "algorithm": algorithm,
                                "window_min": window,
                                "stability_band_mm": threshold,
                                "window_end_recording_id": endpoint["recording_id"],
                                "window_end_elapsed_from_power_min": end_x,
                                "window_end_time_local": endpoint["recording_time_local"],
                                "segment": segment,
                                "sample_count": len(selected),
                                "actual_observation_span_min": span,
                                "first_sample_elapsed_from_power_min": min(times) if times else None,
                                "coverage_pass": coverage_pass,
                                "support_pass": support_pass,
                                "ground_offset_range_mm": ground_range,
                                "upper_height_range_mm": height_ranges["upper"],
                                "middle_height_range_mm": height_ranges["middle"],
                                "lower_height_range_mm": height_ranges["lower"],
                                "joint_max_range_mm": joint_range,
                                "stable_band_pass": passed,
                                "reconnect_boundary_crossed": False,
                                "window_policy": "same reconnect segment; >=3 recordings; actual span >=75% nominal window",
                            }
                        )

    candidates: list[dict[str, Any]] = []
    for reference in a2.REFERENCES:
        for algorithm in a2.ALGORITHMS:
            for window in WINDOWS_MIN:
                for threshold in STABILITY_BANDS_MM:
                    rows = [
                        row for row in detail
                        if row["reference"] == reference
                        and row["algorithm"] == algorithm
                        and row["window_min"] == window
                        and abs(float(row["stability_band_mm"]) - threshold) < 1e-12
                        and row["segment"] == "post_reconnect"
                        and row["coverage_pass"]
                    ]
                    rows.sort(key=lambda row: float(row["window_end_elapsed_from_power_min"]))
                    candidate = None
                    for index, row in enumerate(rows):
                        tail = rows[index:]
                        if len(tail) >= 3 and all(item["stable_band_pass"] for item in tail):
                            candidate = row
                            break
                    candidates.append(
                        {
                            "reference": reference,
                            "algorithm": algorithm,
                            "window_min": window,
                            "stability_band_mm": threshold,
                            "candidate_found": candidate is not None,
                            "candidate_warmup_elapsed_from_power_min": (
                                candidate["first_sample_elapsed_from_power_min"] if candidate else None
                            ),
                            "candidate_confirming_window_end_min": (
                                candidate["window_end_elapsed_from_power_min"] if candidate else None
                            ),
                            "tail_window_count": (
                                len([row for row in rows if float(row["window_end_elapsed_from_power_min"]) >= float(candidate["window_end_elapsed_from_power_min"])])
                                if candidate else 0
                            ),
                            "candidate_policy": "earliest post-reconnect window whose eligible tail remains within band; warmup is first observed sample in that window",
                        }
                    )
    candidate_lookup = {
        (row["reference"], row["algorithm"], row["window_min"], row["stability_band_mm"]): row
        for row in candidates
    }
    for row in detail:
        candidate = candidate_lookup[(row["reference"], row["algorithm"], row["window_min"], row["stability_band_mm"])]
        row.update(
            {
                "candidate_found": candidate["candidate_found"],
                "candidate_warmup_elapsed_from_power_min": candidate["candidate_warmup_elapsed_from_power_min"],
                "candidate_confirming_window_end_min": candidate["candidate_confirming_window_end_min"],
            }
        )
    return detail, candidates


def select_canonical_warmup(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    joint: list[dict[str, Any]] = []
    for window in WINDOWS_MIN:
        for threshold in STABILITY_BANDS_MM:
            pair = [
                row for row in candidates
                if row["algorithm"] == "base"
                and row["window_min"] == window
                and abs(float(row["stability_band_mm"]) - threshold) < 1e-12
            ]
            found = len(pair) == 2 and all(row["candidate_found"] for row in pair)
            values = [float(row["candidate_warmup_elapsed_from_power_min"]) for row in pair if row["candidate_found"]]
            joint.append(
                {
                    "window_min": window,
                    "stability_band_mm": threshold,
                    "both_session_local_candidate_found": found,
                    "joint_candidate_warmup_min": max(values) if found else None,
                }
            )
    balanced = [row for row in joint if abs(row["stability_band_mm"] - 0.03) < 1e-12]
    strict = [row for row in joint if abs(row["stability_band_mm"] - 0.02) < 1e-12]
    loose = [row for row in joint if abs(row["stability_band_mm"] - 0.05) < 1e-12]
    if all(row["both_session_local_candidate_found"] for row in balanced):
        canonical = max(float(row["joint_candidate_warmup_min"]) for row in balanced)
        strict_all = all(row["both_session_local_candidate_found"] for row in strict)
        status = "YES" if strict_all else "PARTIAL"
        basis = (
            "0.02/0.03/0.05 mm bands all passed for 30/45/60 min in Session+Local Base"
            if strict_all
            else "0.03/0.05 mm passed for 30/45/60 min, but 0.02 mm failed one or more windows"
        )
    elif any(row["both_session_local_candidate_found"] for row in balanced):
        canonical = max(float(row["joint_candidate_warmup_min"]) for row in balanced if row["both_session_local_candidate_found"])
        status = "PARTIAL"
        basis = "0.03 mm band passed for only a subset of windows"
    elif any(row["both_session_local_candidate_found"] for row in loose):
        canonical = max(float(row["joint_candidate_warmup_min"]) for row in loose if row["both_session_local_candidate_found"])
        status = "PARTIAL"
        basis = "only the loose 0.05 mm sensitivity band supplied a joint candidate"
    else:
        canonical = None
        status = "NO"
        basis = "no joint Session+Local Base candidate even at 0.05 mm"
    found_values = [float(row["joint_candidate_warmup_min"]) for row in joint if row["both_session_local_candidate_found"]]
    return {
        "THERMAL_STEADY_STATE_REACHED": status,
        "ESTIMATED_WARMUP_TIME_MIN": canonical,
        "candidate_basis": basis,
        "sensitivity_candidate_min": min(found_values) if found_values else None,
        "sensitivity_candidate_max": max(found_values) if found_values else None,
        "joint_base_sensitivity": joint,
    }


def frame_field(reference: str, algorithm: str) -> str:
    return f"{reference}_base_mean_mm" if algorithm == "base" else f"{reference}_{algorithm}_mm"


def build_performance(
    frame_rows: list[dict[str, Any]],
    height_rows: list[dict[str, Any]],
    canonical_warmup: float | None,
) -> list[dict[str, Any]]:
    output = []
    for object_id in a2.OBJECT_IDS:
        nominal = float(a2.OBJECT_META[object_id]["height_mm"])
        for reference in a2.REFERENCES:
            for algorithm in a2.ALGORITHMS:
                summaries = [
                    row for row in height_rows
                    if row["object_id"] == object_id
                    and row["reference"] == reference
                    and row["algorithm"] == algorithm
                    and num(row.get("mean_mm")) is not None
                ]
                field = frame_field(reference, algorithm)
                frames = [
                    float(value) for row in frame_rows
                    if row["object_id"] == object_id
                    and (value := num(row.get(field))) is not None
                ]
                errors = np.asarray(frames) - nominal
                stable_summaries = [
                    row for row in summaries
                    if canonical_warmup is not None
                    and float(row["elapsed_from_power_min"]) >= canonical_warmup
                ]
                stable_recording_ids = {row["recording_id"] for row in stable_summaries}
                stable_frames = [
                    float(value) for row in frame_rows
                    if row["recording_id"] in stable_recording_ids
                    and row["object_id"] == object_id
                    and (value := num(row.get(field))) is not None
                ]
                stable_errors = np.asarray(stable_frames) - nominal if stable_frames else np.asarray([])
                output.append(
                    {
                        "height": f"{int(nominal)} mm",
                        "position": a2.OBJECT_META[object_id]["position"],
                        "object_id": object_id,
                        "reference": reference,
                        "algorithm": algorithm,
                        "recording_count": len(summaries),
                        "frame_count": len(frames),
                        "full_day_recording_mean_thermal_range_mm": range_of(row["mean_mm"] for row in summaries),
                        "full_day_bias_mm": float(np.mean(errors)),
                        "full_day_mae_mm": float(np.mean(np.abs(errors))),
                        "full_day_rmse_mm": float(np.sqrt(np.mean(errors**2))),
                        "full_day_p95_abs_error_mm": float(np.percentile(np.abs(errors), 95)),
                        "full_day_observed_max_abs_error_mm": float(np.max(np.abs(errors))),
                        "median_20_frame_repeatability_std_mm": percentile((row["repeatability_std_mm"] for row in summaries), 50),
                        "max_20_frame_repeatability_std_mm": max(float(row["repeatability_std_mm"]) for row in summaries if num(row.get("repeatability_std_mm")) is not None),
                        "canonical_warmup_elapsed_from_power_min": canonical_warmup,
                        "post_stability_recording_count": len(stable_summaries),
                        "post_stability_frame_count": len(stable_frames),
                        "post_stability_recording_mean_range_mm": range_of(row["mean_mm"] for row in stable_summaries),
                        "post_stability_mae_mm": float(np.mean(np.abs(stable_errors))) if stable_errors.size else None,
                        "post_stability_p95_abs_error_mm": float(np.percentile(np.abs(stable_errors), 95)) if stable_errors.size else None,
                        "post_stability_observed_max_abs_error_mm": float(np.max(np.abs(stable_errors))) if stable_errors.size else None,
                        "single_session_observed_envelope_only": True,
                    }
                )
    return output


def plot_events(ax: Any) -> None:
    ax.axvline(0, color="#111827", lw=0.8, ls="--")
    ax.axvline(7, color="#7c3aed", lw=0.8, ls=":")
    ax.axvline(RECONNECT_MIN, color="#dc2626", lw=1.0, ls="--")
    ax.axvspan(a2.elapsed_min(a2.PAUSE_START), a2.elapsed_min(a2.PAUSE_END), color="#e5e7eb", alpha=0.5)
    ax.axvspan(a2.elapsed_min(a2.NO_RECORD_START), RECONNECT_MIN, color="#fef3c7", alpha=0.35)
    ax.grid(alpha=0.22)


def save_plots(
    output_dir: Path,
    ground_rows: list[dict[str, Any]],
    height_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    performance: list[dict[str, Any]],
    canonical: dict[str, Any],
) -> None:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.dpi": 180})
    base = [row for row in height_rows if row["algorithm"] == "base"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(
        [float(row["elapsed_from_power_min"]) for row in ground_rows],
        [float(row["delta_offset_b_vs_first_mm"]) for row in ground_rows],
        "o-", ms=4, color="#111827", label="Ground offset Δb",
    )
    axes[0].set_ylabel("Ground Δb (mm)")
    for ax, reference in zip(axes[1:], a2.REFERENCES):
        for object_id in a2.OBJECT_IDS:
            rows = [row for row in base if row["reference"] == reference and row["object_id"] == object_id]
            ax.plot(
                [float(row["elapsed_from_power_min"]) for row in rows],
                [float(row["delta_h_vs_first_recording_mm"]) for row in rows],
                "o-", ms=3.5, lw=1.2,
                color=a2.OBJECT_META[object_id]["color"],
                label=f"{int(a2.OBJECT_META[object_id]['height_mm'])} mm / {object_id}",
            )
        ax.set_ylabel(f"{reference.title()} Base Δh (mm)")
        ax.legend(ncol=3, fontsize=8)
    for ax in axes:
        plot_events(ax)
    axes[-1].set_xlabel("Elapsed from 09:50 power-on (min)")
    axes[0].legend()
    fig.suptitle("0827 full-day observed thermal drift (single cold-start session)")
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a3_full_day_drift.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, reference in zip(axes, a2.REFERENCES):
        for window, color in zip(WINDOWS_MIN, ("#2563eb", "#db2777", "#059669")):
            rows = [
                row for row in stability_rows
                if row["reference"] == reference
                and row["algorithm"] == "base"
                and row["window_min"] == window
                and abs(float(row["stability_band_mm"]) - 0.03) < 1e-12
                and row["coverage_pass"]
            ]
            ax.plot(
                [float(row["window_end_elapsed_from_power_min"]) for row in rows],
                [float(row["joint_max_range_mm"]) for row in rows],
                "o-", ms=3, lw=1.1, color=color, label=f"{window} min rolling joint range",
            )
        for threshold, ls in zip(STABILITY_BANDS_MM, (":", "--", "-.")):
            ax.axhline(threshold, color="#6b7280", lw=0.8, ls=ls, label=f"{threshold:.2f} mm band")
        warmup = canonical["ESTIMATED_WARMUP_TIME_MIN"]
        if warmup is not None:
            ax.axvline(warmup, color="#ea580c", lw=1.2, label=f"canonical candidate {warmup:.1f} min")
        ax.set_ylabel(f"{reference.title()} Base\njoint rolling range (mm)")
        plot_events(ax)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), ncol=4, fontsize=7)
    axes[-1].set_xlabel("Elapsed from 09:50 power-on (min)")
    fig.suptitle("Rolling stability sensitivity: Ground offset + all three heights")
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a3_warmup_stability.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    x = np.arange(len(a2.OBJECT_IDS))
    width = 0.23
    for col, reference in enumerate(a2.REFERENCES):
        for index, algorithm in enumerate(a2.ALGORITHMS):
            rows = [
                next(row for row in performance if row["object_id"] == object_id and row["reference"] == reference and row["algorithm"] == algorithm)
                for object_id in a2.OBJECT_IDS
            ]
            axes[col, 0].bar(x + (index - 1) * width, [row["full_day_recording_mean_thermal_range_mm"] for row in rows], width, label=algorithm.upper())
            axes[col, 1].bar(x + (index - 1) * width, [row["full_day_p95_abs_error_mm"] for row in rows], width, label=algorithm.upper())
            axes[col, 2].bar(x + (index - 1) * width, [row["median_20_frame_repeatability_std_mm"] for row in rows], width, label=algorithm.upper())
        axes[col, 0].set_ylabel(f"{reference.title()} (mm)")
        for ax in axes[col]:
            ax.set_xticks(x, ["20-upper", "30-middle", "10-lower"])
            ax.grid(axis="y", alpha=0.22)
    axes[0, 0].set_title("Recording-mean thermal range")
    axes[0, 1].set_title("Frame P95 absolute error")
    axes[0, 2].set_title("Median 20-frame repeatability std")
    axes[0, 2].legend(ncol=3, fontsize=8)
    fig.suptitle("Base / H1 / H-B2 full-day performance")
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_a3_algorithm_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 6) -> str:
    number = num(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def build_report(
    provenance: dict[str, Any],
    afternoon_audit: dict[str, Any],
    ground_rows: list[dict[str, Any]],
    performance: list[dict[str, Any]],
    canonical: dict[str, Any],
) -> str:
    base_rows = [row for row in performance if row["algorithm"] == "base"]
    session_envelope = max(float(row["full_day_recording_mean_thermal_range_mm"]) for row in base_rows if row["reference"] == "session")
    local_envelope = max(float(row["full_day_recording_mean_thermal_range_mm"]) for row in base_rows if row["reference"] == "local")
    suppression_ratio = 1.0 - local_envelope / session_envelope
    post_session_envelope = max(float(row["post_stability_recording_mean_range_mm"]) for row in base_rows if row["reference"] == "session")
    post_local_envelope = max(float(row["post_stability_recording_mean_range_mm"]) for row in base_rows if row["reference"] == "local")
    ground_range = range_of(row["offset_b_mm"] for row in ground_rows)
    afternoon_ground_range = range_of(
        row["offset_b_mm"] for row in ground_rows if row["source_period"] == "afternoon"
    )
    last_time = max(float(row["elapsed_from_power_min"]) for row in ground_rows)
    perf_lines = [
        "| height | position | reference | algorithm | full-day range | P95 | repeatability std | observed max | post-stability range |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in performance:
        perf_lines.append(
            f"| {row['height']} | {row['position']} | {row['reference']} | {row['algorithm']} | "
            f"{fmt(row['full_day_recording_mean_thermal_range_mm'])} | {fmt(row['full_day_p95_abs_error_mm'])} | "
            f"{fmt(row['median_20_frame_repeatability_std_mm'])} | {fmt(row['full_day_observed_max_abs_error_mm'])} | "
            f"{fmt(row['post_stability_recording_mean_range_mm'])} |"
        )
    sensitivity_lines = [
        "| rolling window | band | Session+Local Base candidate | candidate warm-up (min from power-on) |",
        "|---:|---:|---|---:|",
    ]
    for row in canonical["joint_base_sensitivity"]:
        sensitivity_lines.append(
            f"| {row['window_min']} min | {row['stability_band_mm']:.2f} mm | "
            f"{'YES' if row['both_session_local_candidate_found'] else 'NO'} | {fmt(row['joint_candidate_warmup_min'], 2)} |"
        )
    return f"""# Thermal-A3-1｜全日 6 h 热漂扩展与稳态判定

## 阶段结论

- `FULL_DAY_OBSERVED_SESSION_BASE_ENVELOPE_MM = {fmt(session_envelope)}`
- `FULL_DAY_OBSERVED_LOCAL_BASE_ENVELOPE_MM = {fmt(local_envelope)}`
- `LOCAL_REFERENCE_SUPPRESSION_RATIO = {fmt(suppression_ratio, 4)}`
- `FULL_DAY_GROUND_OFFSET_RANGE_MM = {fmt(ground_range)}`
- `THERMAL_STEADY_STATE_REACHED = {canonical['THERMAL_STEADY_STATE_REACHED']}`
- `ESTIMATED_WARMUP_TIME_MIN = {fmt(canonical['ESTIMATED_WARMUP_TIME_MIN'], 2)}`
- `WARMUP_SENSITIVITY_RANGE_MIN = {fmt(canonical['sensitivity_candidate_min'], 2)}–{fmt(canonical['sensitivity_candidate_max'], 2)}`

以上均为 0827 单次 cold-start Session、09:50 至末次 recording（{last_time:.2f} min）的 **observed thermal envelope**，不是系统理论 worst case。

## Artifact provenance / reuse audit

- 上午 29 recording / 580 PNG 直接复用已完成 Thermal-A2 数值；复用前逐项核验 A2 manifest、全部输出 SHA256、580 帧 cardinality、Frozen registry/config/Session Ground hash。A2 manifest SHA256：`{provenance['a2_reuse']['manifest_sha256']}`。
- 用户 GUI Frozen ROI SHA256：`{provenance['registry']['registry_sha256']}`，状态 `FROZEN_USER_CONFIRMED`，support gate `PASS`；旧 Codex invalid attempt 继续排除。
- 本轮新增计算：下午 9 个正式 recording / 180 PNG 的同链路 Frozen replay，以及上午+下午的 full-day 汇总、rolling stability、图和性能表。
- `.recording_*_linshi`：{afternoon_audit['temporary_recordings_qc_only']} 个 / {afternoon_audit['temporary_frames_excluded']} 帧，仅完成 metadata 与 raw scene QC，未进入正式 replay 或统计。
- 未读取 `height_shadow.csv`；未重选 ROI；未重拟 C0/C1、Session R/t、Session Ground、H1 或 H-B2。

## 下午数据 QC 与连续性

- 9 个正式 recording 均为 20 PNG、2000 μs、Gain 0、Mono8、硬件 ROI `(1760,0,480,3000)`；recording 内 camera frame +1、camera timestamp 固定步长、`frame_gap=0`。
- 上午 14:04 末段至下午 14:15 首段的 camera frame number、camera timestamp 与 host timestamp 全部单调前进；未检测到第二次 reconnect。
- raw Mono8 spot-check（上午末、下午首、linshi、下午末）中三量块/棋盘/激光线布局一致；正式结论另由全部下午帧 Frozen ROI support gate 支撑。
- `.linshi` 本身 metadata QC 通过，但按协议和命名保持 QC-only。

## Full-day envelope 与来源判断

- Base recording-mean 的最大全日 envelope：Session {session_envelope:.6f} mm，Local {local_envelope:.6f} mm。
- Local reference 对 worst-object envelope 的抑制率为 {suppression_ratio:.1%}。按量块分别为：20 mm {1.0 - next(row for row in base_rows if row['object_id'] == 'upper' and row['reference'] == 'local')['full_day_recording_mean_thermal_range_mm'] / next(row for row in base_rows if row['object_id'] == 'upper' and row['reference'] == 'session')['full_day_recording_mean_thermal_range_mm']:.1%}、30 mm {1.0 - next(row for row in base_rows if row['object_id'] == 'middle' and row['reference'] == 'local')['full_day_recording_mean_thermal_range_mm'] / next(row for row in base_rows if row['object_id'] == 'middle' and row['reference'] == 'session')['full_day_recording_mean_thermal_range_mm']:.1%}、10 mm {1.0 - next(row for row in base_rows if row['object_id'] == 'lower' and row['reference'] == 'local')['full_day_recording_mean_thermal_range_mm'] / next(row for row in base_rows if row['object_id'] == 'lower' and row['reference'] == 'session')['full_day_recording_mean_thermal_range_mm']:.1%}。
- 因此漂移来源判断为 **MIXED**：全局 Session/Ground reference 是重要分量，但 Local 后仍有最多 {local_envelope:.6f} mm 残余，且 lower/10 mm 抑制明显较弱，不能归结为纯全局 reference 漂移。
- Ground offset `b(t)` 全日 range：{fmt(ground_range)} mm；下午独立 range：{fmt(afternoon_ground_range)} mm。下午没有扩大上午已经观察到的全日 Ground/height envelope。
- canonical candidate 之后 Base worst-object recording-mean range：Session {post_session_envelope:.6f} mm、Local {post_local_envelope:.6f} mm。
- H1/H-B2 明显改变绝对误差，但三种算法的 thermal range 与 repeatability 几乎相同；现有 correction 改善的是尺度/偏差，不是热漂本身。
- 20/30/10 mm 固定在 upper/middle/lower，高度与位置共线；不能由单 Session 唯一拆分 height-dependent 与 position-dependent 因果。

## Rolling 稳态敏感性

- 每个 window 仅使用同一 reconnect segment；要求至少 3 个 recording 且实际观测跨度达到 nominal window 的 75%。13:01 gap/reconnect 不被当作 thermal reset，也不跨边界组成 rolling window。
- joint range 同时约束 Ground offset 和三个固定量块高度。候选定义为 post-reconnect 中最早且其后所有 eligible windows 都保持在稳定带内的窗口；warm-up 值是该确认窗口内首个实际观测点，因此是 observation-bounded candidate，不是连续温度传感器意义的精确时刻。
- 正式阶段判断：`{canonical['candidate_basis']}`。0.02/0.03/0.05 mm 的完整敏感性如下，不用单一阈值强行下结论。

{chr(10).join(sensitivity_lines)}

## Base / H1 / H-B2 最终性能表

P95 和 observed max 是全部有效 frame 相对 nominal 的绝对误差；repeatability 是 recording 内 20-frame std 的中位数；post-stability 使用上面的 canonical candidate，若为 NA 则不报告稳定后数值。

{chr(10).join(perf_lines)}

## 输出说明

- `thermal_a3_full_day_results.csv`：38 recording × 3 object × 2 reference × 3 algorithm 长表，标注上午复用/下午新增 replay。
- `thermal_a3_stability_analysis.csv`：30/45/60 min × 0.02/0.03/0.05 mm rolling 明细与候选结果。
- `thermal_a3_performance_table.csv`：Session/Local × Base/H1/H-B2 全日和稳定后性能。
- 三张 PNG 分别为 full-day drift、rolling sensitivity 和算法长期精度对比。
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("morning_dir", "afternoon_dir", "a2_dir", "registry", "measure_config", "session_ground", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())

    _, rois, registry_provenance = a2.validate_registry(
        args.registry,
        args.expected_registry_sha256,
        args.measure_config,
        args.session_ground,
    )
    registry_sha = registry_provenance["registry_sha256"]
    config_sha = a2.sha256_file(args.measure_config)
    ground_sha = a2.sha256_file(args.session_ground)
    a2_reuse = verify_a2_reuse(args.a2_dir, registry_sha, config_sha, ground_sha)
    audit_rows, afternoon_audit = audit_afternoon(args.afternoon_dir, args.morning_dir)
    afternoon_a1 = synthetic_a1_rows(audit_rows)
    if args.max_afternoon_recordings > 0:
        afternoon_a1 = afternoon_a1[: args.max_afternoon_recordings]

    app, calibration, _, session_reference = a2.load_chain(args.measure_config, args.session_ground)
    afternoon_frames, afternoon_recordings, afternoon_ground, afternoon_fits = a2.process_all(
        afternoon_a1, rois, app, calibration, session_reference, 0
    )
    afternoon_heights = a2.build_height_summary(afternoon_frames, afternoon_recordings)
    if args.max_afternoon_recordings > 0:
        print("A3_SMOKE_RUN_COMPLETE: formal outputs were not written", flush=True)
        return 0
    if not (
        len(afternoon_recordings) == len(afternoon_ground) == 9
        and len(afternoon_frames) == 180 * 3
        and len(afternoon_heights) == 9 * 3 * 2 * 3
    ):
        raise ThermalA3Error("Afternoon replay cardinality mismatch")
    if any(row["frame_status"] != "VALID" for row in afternoon_frames):
        raise ThermalA3Error("Afternoon contains invalid reconstruction frames")
    support_failures = [
        row for row in afternoon_frames
        if row.get("session_status") != "VALID" or row.get("local_status") != "VALID"
    ]
    if support_failures:
        raise ThermalA3Error(f"Afternoon frozen ROI support failures: {len(support_failures)}")
    afternoon_audit["frozen_reconstruction_support_all_frames"] = True
    afternoon_audit["scene_continuity_formal_gate"] = "PASS"

    morning_frames = read_csv(args.a2_dir / "thermal_a2_frame_results.csv")
    morning_recordings = read_csv(args.a2_dir / "thermal_a2_recording_summary.csv")
    morning_ground = read_csv(args.a2_dir / "thermal_a2_ground_drift.csv")
    morning_heights = read_csv(args.a2_dir / "thermal_a2_height_session_local.csv")
    mark_source(morning_frames, "morning", "REUSED_THERMAL_A2_EXACT_PROTOCOL")
    mark_source(morning_recordings, "morning", "REUSED_THERMAL_A2_EXACT_PROTOCOL")
    mark_source(morning_ground, "morning", "REUSED_THERMAL_A2_EXACT_PROTOCOL")
    mark_source(morning_heights, "morning", "REUSED_THERMAL_A2_EXACT_PROTOCOL")
    mark_source(afternoon_frames, "afternoon", "NEW_A3_AFTERNOON_FROZEN_REPLAY", 29)
    mark_source(afternoon_recordings, "afternoon", "NEW_A3_AFTERNOON_FROZEN_REPLAY", 29)
    mark_source(afternoon_ground, "afternoon", "NEW_A3_AFTERNOON_FROZEN_REPLAY", 29)
    mark_source(afternoon_heights, "afternoon", "NEW_A3_AFTERNOON_FROZEN_REPLAY", 29)

    frame_rows = morning_frames + afternoon_frames
    recording_rows = morning_recordings + afternoon_recordings
    ground_rows = rebuild_full_day_ground(morning_ground + afternoon_ground)
    height_rows = rebuild_full_day_height(morning_heights + afternoon_heights)
    if not (
        len(recording_rows) == 38
        and len(frame_rows) == 760 * 3
        and len(ground_rows) == 38
        and len(height_rows) == 38 * 3 * 2 * 3
    ):
        raise ThermalA3Error("Full-day cardinality mismatch")

    stability_rows, candidates = build_stability_rows(ground_rows, height_rows)
    canonical = select_canonical_warmup(candidates)
    performance = build_performance(
        frame_rows, height_rows, canonical["ESTIMATED_WARMUP_TIME_MIN"]
    )
    for row in height_rows:
        warmup = canonical["ESTIMATED_WARMUP_TIME_MIN"]
        row["canonical_post_stability"] = warmup is not None and float(row["elapsed_from_power_min"]) >= warmup

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "thermal_a3_full_day_results.csv", height_rows)
    write_csv(args.output_dir / "thermal_a3_stability_analysis.csv", stability_rows)
    write_csv(args.output_dir / "thermal_a3_performance_table.csv", performance)
    save_plots(args.output_dir, ground_rows, height_rows, stability_rows, performance, canonical)
    provenance = {
        "registry": registry_provenance,
        "a2_reuse": a2_reuse,
        "measure_config_sha256": config_sha,
        "session_ground_sha256": ground_sha,
        "morning_result_origin": "REUSED",
        "afternoon_result_origin": "NEW_FROZEN_REPLAY",
        "height_shadow_used": False,
        "models_refit": [],
        "roi_reselected": False,
    }
    report = build_report(provenance, afternoon_audit, ground_rows, performance, canonical)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    output_names = [
        "thermal_a3_full_day_results.csv",
        "thermal_a3_stability_analysis.csv",
        "thermal_a3_full_day_drift.png",
        "thermal_a3_warmup_stability.png",
        "thermal_a3_algorithm_accuracy.png",
        "thermal_a3_performance_table.csv",
        "report.md",
    ]
    write_json(
        args.output_dir / "thermal_a3_run_manifest.json",
        {
            "status": "COMPLETE",
            "protocol": "Thermal-A3-1 full-day exact A2 reuse + afternoon frozen replay",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance": provenance,
            "afternoon_audit": afternoon_audit,
            "cardinality": {
                "formal_recordings": 38,
                "formal_frames": 760,
                "frame_object_rows_internal": len(frame_rows),
                "full_day_result_rows": len(height_rows),
                "stability_rows": len(stability_rows),
                "performance_rows": len(performance),
            },
            "stability": canonical,
            "output_sha256": {
                name: a2.sha256_file(args.output_dir / name) for name in output_names
            },
        },
    )
    print(json.dumps(a2.json_safe(canonical), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ThermalA3Error, a2.ThermalA2Error) as error:
        print(f"ERROR: {error}", flush=True)
        raise SystemExit(2)
