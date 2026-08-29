#!/usr/bin/env python3
"""Create a geometry-only Thermal-A2a-R1 baseline texture-exclusion draft.

The three Thermal-A2a object identities and height ROIs are read from the
existing A2a draft and checked against the user-confirmed mapping.  This
command deliberately does not run object detection again.  It samples only
raw Mono8 background pixels on the two sides of the Frozen Steger stripe,
detects repeat-stable reflection transitions, subtracts fixed +/-15 px
exclusion bands from the existing baselines, and stops before any frozen
registry or manual-review file is created.

No height_shadow.csv, nominal height, 3-D reconstruction, calibration result,
thermal result, residual, or error result is opened by this command.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import thermal_a2a_roi_v2 as a2a  # noqa: E402


OBJECT_IDS = ("upper", "middle", "lower")
EXPECTED_OBJECTS: dict[str, dict[str, Any]] = {
    "upper": {"object_order": 1, "height_label_hint": "20mm", "height_v_range": [312, 380]},
    "middle": {"object_order": 2, "height_label_hint": "30mm", "height_v_range": [1496, 1560]},
    "lower": {"object_order": 3, "height_label_hint": "10mm", "height_v_range": [2627, 2691]},
}
RECORDING_ID = "recording_20260827_100021"
REPEAT_COUNT = 20
COLORS = {"upper": "#386cb0", "middle": "#f0027f", "lower": "#1b9e77"}

# These are fixed image-only detector rules.  They are intentionally not CLI
# parameters so a result cannot be silently tuned against a height/error curve.
BACKGROUND_INNER_GAP_PX = 24
BACKGROUND_OUTER_DISTANCE_PX = 64
TEXTURE_SMOOTHING_SIGMA_PX = 2.0
TRANSITION_PEAK_DISTANCE_PX = 16
TRANSITION_PEAK_MIN_PROMINENCE_DN_PER_V = 0.18
TRANSITION_CONTRAST_WINDOW_PX = 6
TRANSITION_MIN_CONTRAST_DN = 1.5
TRANSITION_MIN_STABLE_FRACTION = 0.75
TEXTURE_EXCLUSION_MARGIN_PX = 15
TRANSITION_FUSION_TOLERANCE_PX = 20
MIN_RETAINED_INTERVAL_PX = 20
PREFERRED_BASELINE_SUPPORT_PX = 80
MIN_POINTS_PER_REPEAT = 20
MIN_SUPPORT_FRACTION = 0.25


class TextureExclusionError(RuntimeError):
    """Raised when the immutable A2a draft/source cannot be audited."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    input_dir = ROOT / "laser_measurement_tool" / "output_daheng_0811" / "online_recordings" / "0827上午热漂_2000"
    output_dir = ROOT / "projects" / "daheng" / "analysis" / "thermal_a2a_roi_v2_0827"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=input_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--draft",
        type=Path,
        default=output_dir / "thermal_roi_v2_registry_v2_draft.json",
    )
    parser.add_argument(
        "--a1-index",
        type=Path,
        default=ROOT / "projects" / "daheng" / "analysis" / "thermal_a1_0827" / "thermal_a1_recording_index.csv",
    )
    parser.add_argument(
        "--measure-config",
        type=Path,
        default=ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml",
    )
    return parser.parse_args(argv)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return "" if not math.isfinite(number) else f"{number:.12g}"
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fieldnames})


def interval(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TextureExclusionError(f"{name} must be [start,end], got {value!r}")
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError) as error:
        raise TextureExclusionError(f"{name} is not integer-valued: {value!r}") from error
    if start > end:
        raise TextureExclusionError(f"{name} is reversed: {value!r}")
    return [start, end]


def half_open(value: list[int]) -> list[int]:
    return [value[0], value[1] + 1]


def interval_length(value: list[int]) -> int:
    return value[1] - value[0] + 1


def overlaps(first: list[int], second: list[int]) -> bool:
    return bool(first and second and max(first[0], second[0]) <= min(first[1], second[1]))


def merge_intervals(values: Iterable[list[int]]) -> list[list[int]]:
    ordered = sorted((interval(value, "merge interval") for value in values), key=lambda item: (item[0], item[1]))
    merged: list[list[int]] = []
    for value in ordered:
        if not merged or value[0] > merged[-1][1] + 1:
            merged.append(list(value))
        else:
            merged[-1][1] = max(merged[-1][1], value[1])
    return merged


def intersect(first: list[int], second: list[int]) -> list[int] | None:
    start, end = max(first[0], second[0]), min(first[1], second[1])
    return [start, end] if start <= end else None


def subtract_intervals(original: list[int], exclusions: list[list[int]]) -> list[list[int]]:
    pieces = [list(original)]
    for exclusion in merge_intervals(exclusions):
        next_pieces: list[list[int]] = []
        for piece in pieces:
            if not overlaps(piece, exclusion):
                next_pieces.append(piece)
                continue
            if piece[0] < exclusion[0]:
                next_pieces.append([piece[0], exclusion[0] - 1])
            if exclusion[1] < piece[1]:
                next_pieces.append([exclusion[1] + 1, piece[1]])
        pieces = next_pieces
    return [piece for piece in pieces if piece[0] <= piece[1]]


def validate_draft(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("frozen") or payload.get("human_reviewed"):
        raise TextureExclusionError("R1 accepts only the unfrozen A2a draft registry")
    objects = payload.get("objects")
    if not isinstance(objects, list) or len(objects) != 3:
        raise TextureExclusionError("A2a draft must contain exactly three objects")
    by_id = {str(item.get("object_id")): item for item in objects if isinstance(item, dict)}
    if tuple(by_id) != OBJECT_IDS:
        raise TextureExclusionError(f"A2a object order must be upper/middle/lower, got {tuple(by_id)}")
    checked: list[dict[str, Any]] = []
    for object_id in OBJECT_IDS:
        item = copy.deepcopy(by_id[object_id])
        expected = EXPECTED_OBJECTS[object_id]
        if int(item.get("object_order", -1)) != expected["object_order"]:
            raise TextureExclusionError(f"{object_id}: object_order changed in A2a draft")
        if str(item.get("height_label_hint")) != expected["height_label_hint"]:
            raise TextureExclusionError(f"{object_id}: user-confirmed height label changed")
        actual_height = interval(item.get("height_v_range"), f"{object_id}.height_v_range")
        if actual_height != expected["height_v_range"]:
            raise TextureExclusionError(
                f"{object_id}: height ROI changed; expected {expected['height_v_range']}, got {actual_height}"
            )
        baselines = item.get("baseline_v_ranges")
        if not isinstance(baselines, list) or len(baselines) != 2:
            raise TextureExclusionError(f"{object_id}: A2a draft must contain two original baselines")
        item["height_v_range"] = actual_height
        item["baseline_v_ranges"] = [
            interval(baselines[0], f"{object_id}.baseline_before"),
            interval(baselines[1], f"{object_id}.baseline_after"),
        ]
        checked.append(item)
    return checked


def centerline_u_by_v(frame: a2a.SourceFrame) -> np.ndarray:
    by_v: dict[int, list[float]] = defaultdict(list)
    values = np.asarray(frame.centers_uv_full, dtype=np.float64)
    for u, v in values:
        if not (math.isfinite(float(u)) and math.isfinite(float(v))):
            continue
        row = int(round(float(v)))
        if 0 <= row < frame.height:
            by_v[row].append(float(u) - frame.offset_x)
    if len(by_v) < 100:
        raise TextureExclusionError(f"{frame.filename}: Frozen Steger has too few v rows: {len(by_v)}")
    rows = np.asarray(sorted(by_v), dtype=np.int32)
    medians = np.asarray([np.median(by_v[int(row)]) for row in rows], dtype=np.float64)
    result = np.full(frame.height, np.nan, dtype=np.float64)
    result[rows[0] : rows[-1] + 1] = np.interp(
        np.arange(rows[0], rows[-1] + 1, dtype=np.float64), rows.astype(np.float64), medians
    )
    return result


def sample_texture_profiles(frames: list[a2a.SourceFrame]) -> dict[str, Any]:
    left_profiles: list[np.ndarray] = []
    right_profiles: list[np.ndarray] = []
    u_profiles: list[np.ndarray] = []
    for frame in frames:
        if frame.image.dtype != np.uint8:
            raise TextureExclusionError(f"{frame.filename}: expected raw Mono8, got {frame.image.dtype}")
        u_by_v = centerline_u_by_v(frame)
        left = np.full(frame.height, np.nan, dtype=np.float64)
        right = np.full(frame.height, np.nan, dtype=np.float64)
        for row, u_value in enumerate(u_by_v):
            if not math.isfinite(float(u_value)):
                continue
            center = int(round(float(u_value)))
            left_start = max(0, center - BACKGROUND_OUTER_DISTANCE_PX)
            left_end = min(frame.width, center - BACKGROUND_INNER_GAP_PX)
            right_start = max(0, center + BACKGROUND_INNER_GAP_PX)
            right_end = min(frame.width, center + BACKGROUND_OUTER_DISTANCE_PX)
            if left_end - left_start >= 10:
                left[row] = float(np.median(frame.image[row, left_start:left_end]))
            if right_end - right_start >= 10:
                right[row] = float(np.median(frame.image[row, right_start:right_end]))
        left_profiles.append(left)
        right_profiles.append(right)
        u_profiles.append(u_by_v)
    left_array = np.asarray(left_profiles, dtype=np.float64)
    right_array = np.asarray(right_profiles, dtype=np.float64)
    return {
        "left": left_array,
        "right": right_array,
        "u_by_v": np.asarray(u_profiles, dtype=np.float64),
        "left_median": np.nanmedian(left_array, axis=0),
        "right_median": np.nanmedian(right_array, axis=0),
    }


def local_contrast(row: np.ndarray, center: int) -> float:
    radius = TRANSITION_CONTRAST_WINDOW_PX
    before = row[max(0, center - radius) : center]
    after = row[center : min(len(row), center + radius)]
    if before.size == 0 or after.size == 0 or not np.isfinite(before).any() or not np.isfinite(after).any():
        return float("nan")
    return float(np.nanmedian(after) - np.nanmedian(before))


def detect_side_transitions(profiles: np.ndarray, side: str) -> dict[str, Any]:
    median = np.nanmedian(profiles, axis=0)
    valid = np.isfinite(median)
    if int(valid.sum()) < 100:
        raise TextureExclusionError(f"{side}: texture profile has too few valid rows")
    filled = median.copy()
    rows = np.arange(len(median), dtype=np.float64)
    filled[~valid] = np.interp(rows[~valid], rows[valid], median[valid])
    smoothed = gaussian_filter1d(filled, TEXTURE_SMOOTHING_SIGMA_PX)
    gradient = np.gradient(smoothed)
    abs_gradient = np.abs(gradient)
    derivative_noise = float(1.4826 * np.median(np.abs(gradient - np.median(gradient))))
    min_prominence = max(TRANSITION_PEAK_MIN_PROMINENCE_DN_PER_V, 6.0 * derivative_noise)
    peaks, properties = find_peaks(
        abs_gradient,
        distance=TRANSITION_PEAK_DISTANCE_PX,
        prominence=min_prominence,
    )
    candidates: list[dict[str, Any]] = []
    for index, peak in enumerate(peaks):
        if peak < TRANSITION_CONTRAST_WINDOW_PX or peak >= len(median) - TRANSITION_CONTRAST_WINDOW_PX:
            continue
        differences = np.asarray([local_contrast(row, int(peak)) for row in profiles], dtype=np.float64)
        finite = differences[np.isfinite(differences)]
        if finite.size != profiles.shape[0]:
            continue
        signed_contrast = float(np.median(finite))
        absolute_contrast = float(np.median(np.abs(finite)))
        direction = 1 if signed_contrast >= 0 else -1
        stable_threshold = max(TRANSITION_MIN_CONTRAST_DN, 0.5 * absolute_contrast)
        stable_mask = (np.abs(finite) >= stable_threshold) & (np.sign(finite) == direction)
        stable_count = int(stable_mask.sum())
        stable_fraction = stable_count / float(len(finite))
        accepted = bool(
            absolute_contrast >= TRANSITION_MIN_CONTRAST_DN
            and stable_fraction >= TRANSITION_MIN_STABLE_FRACTION
            and abs(signed_contrast) >= TRANSITION_MIN_CONTRAST_DN
        )
        candidates.append(
            {
                "source_side": side,
                "candidate_center_v": int(peak),
                "candidate_transition_v_range": [int(peak - 3), int(peak + 3)],
                "gradient_dn_per_v": float(gradient[peak]),
                "gradient_abs_dn_per_v": float(abs_gradient[peak]),
                "peak_prominence_dn_per_v": float(properties["prominences"][index]),
                "derivative_noise_dn_per_v": derivative_noise,
                "min_prominence_dn_per_v": min_prominence,
                "signed_contrast_dn": signed_contrast,
                "median_abs_contrast_dn": absolute_contrast,
                "stable_repeat_count": stable_count,
                "stable_repeat_fraction": stable_fraction,
                "accepted": accepted,
                "rejection_reason": ""
                if accepted
                else "contrast_or_repeat_stability_below_fixed_image_only_gate",
            }
        )
    accepted = [item for item in candidates if item["accepted"]]
    return {
        "side": side,
        "profile_median": median,
        "profile_smoothed": smoothed,
        "gradient": gradient,
        "derivative_noise_dn_per_v": derivative_noise,
        "min_prominence_dn_per_v": min_prominence,
        "candidates": candidates,
        "accepted": accepted,
    }


def fuse_transitions(side_results: dict[str, dict[str, Any]], height: int) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for side in ("left", "right"):
        accepted.extend(copy.deepcopy(side_results[side]["accepted"]))
    accepted.sort(key=lambda item: int(item["candidate_center_v"]))
    groups: list[list[dict[str, Any]]] = []
    for candidate in accepted:
        if not groups:
            groups.append([candidate])
            continue
        previous_center = int(groups[-1][-1]["candidate_center_v"])
        if int(candidate["candidate_center_v"]) - previous_center <= TRANSITION_FUSION_TOLERANCE_PX:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    fused: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        centers = [int(item["candidate_center_v"]) for item in group]
        center = int(round(float(np.median(centers))))
        exclusion = merge_intervals(
            [
                [
                    max(0, int(item["candidate_center_v"]) - TEXTURE_EXCLUSION_MARGIN_PX),
                    min(height - 1, int(item["candidate_center_v"]) + TEXTURE_EXCLUSION_MARGIN_PX),
                ]
                for item in group
            ]
        )
        by_side = {side: [item for item in group if item["source_side"] == side] for side in ("left", "right")}
        fused.append(
            {
                "transition_group_id": f"texture_t{group_index:02d}",
                "transition_center_v": center,
                "source_sides": [side for side in ("left", "right") if by_side[side]],
                "detected_centers_by_side": {
                    side: [int(item["candidate_center_v"]) for item in by_side[side]]
                    for side in ("left", "right")
                    if by_side[side]
                },
                "texture_exclusion_v_ranges": exclusion,
                "source_side_evidence": group,
                "max_median_abs_contrast_dn": max(float(item["median_abs_contrast_dn"]) for item in group),
                "min_stable_repeat_fraction": min(float(item["stable_repeat_fraction"]) for item in group),
                "stable_repeat_count_min": min(int(item["stable_repeat_count"]) for item in group),
            }
        )
    return fused


def support_for_intervals(frames: list[a2a.SourceFrame], ranges: list[list[int]]) -> dict[str, Any]:
    lengths = [interval_length(value) for value in ranges]
    total_width = int(sum(lengths))
    per_interval: list[dict[str, Any]] = []
    for interval_index, value in enumerate(ranges, start=1):
        counts: list[int] = []
        for frame in frames:
            centers = np.asarray(frame.centers_uv_full, dtype=np.float64)
            counts.append(int(np.sum((centers[:, 1] >= value[0]) & (centers[:, 1] <= value[1]))))
        fractions = [count / float(interval_length(value)) for count in counts]
        per_interval.append(
            {
                "interval_index": interval_index,
                "v_range": list(value),
                "length_px": interval_length(value),
                "repeat_point_counts": counts,
                "min_points": min(counts) if counts else 0,
                "median_support_fraction": float(np.median(fractions)) if fractions else 0.0,
                "all_repeats_min_points_ok": bool(counts) and all(count >= MIN_POINTS_PER_REPEAT for count in counts),
                "support_ok": bool(counts)
                and interval_length(value) >= MIN_RETAINED_INTERVAL_PX
                and all(count >= MIN_POINTS_PER_REPEAT for count in counts)
                and float(np.median(fractions)) >= MIN_SUPPORT_FRACTION,
            }
        )
    total_counts: list[int] = []
    for frame in frames:
        centers = np.asarray(frame.centers_uv_full, dtype=np.float64)
        total_counts.append(
            int(
                sum(
                    np.sum((centers[:, 1] >= value[0]) & (centers[:, 1] <= value[1]))
                    for value in ranges
                )
            )
        )
    total_fractions = [count / float(total_width) for count in total_counts] if total_width else []
    return {
        "intervals": per_interval,
        "total_width_px": total_width,
        "repeat_point_counts": total_counts,
        "min_points": min(total_counts) if total_counts else 0,
        "median_support_fraction": float(np.median(total_fractions)) if total_fractions else 0.0,
        "all_repeats_min_points_ok": bool(total_counts)
        and all(count >= MIN_POINTS_PER_REPEAT for count in total_counts),
        "support_ok": bool(ranges)
        and total_width >= PREFERRED_BASELINE_SUPPORT_PX
        and bool(total_counts)
        and all(interval_result["support_ok"] for interval_result in per_interval)
        and all(count >= MIN_POINTS_PER_REPEAT for count in total_counts)
        and float(np.median(total_fractions)) >= MIN_SUPPORT_FRACTION,
    }


def build_object_results(
    objects: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    frames: list[a2a.SourceFrame],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in objects:
        object_id = str(item["object_id"])
        edge_pair = item.get("edge_pair", {})
        edge1 = int(edge_pair.get("edge1_v"))
        edge2 = int(edge_pair.get("edge2_v"))
        original = [list(value) for value in item["baseline_v_ranges"]]
        by_side: dict[str, Any] = {}
        object_exclusions: list[list[int]] = []
        for side, original_range in zip(("before", "after"), original):
            applied: list[list[int]] = []
            groups: list[str] = []
            for transition in transitions:
                intersections = [intersect(original_range, value) for value in transition["texture_exclusion_v_ranges"]]
                clipped = [value for value in intersections if value is not None]
                if clipped:
                    applied.extend(clipped)
                    groups.append(str(transition["transition_group_id"]))
            applied = merge_intervals(applied)
            raw_revised = subtract_intervals(original_range, applied)
            short_dropped = [value for value in raw_revised if interval_length(value) < MIN_RETAINED_INTERVAL_PX]
            revised = [value for value in raw_revised if interval_length(value) >= MIN_RETAINED_INTERVAL_PX]
            support = support_for_intervals(frames, revised)
            non_clipped = bool(revised) and all(value[0] > 0 and value[1] < frames[0].height - 1 for value in revised)
            no_height_edge_overlap = bool(revised) and not any(
                overlaps(value, item["height_v_range"]) or overlaps(value, [edge1, edge2]) for value in revised
            )
            by_side[side] = {
                "original_v_range": original_range,
                "texture_exclusion_v_ranges": applied,
                "applied_transition_group_ids": groups,
                "raw_revised_v_ranges_before_min_length_qc": raw_revised,
                "short_subsegments_dropped_by_fixed_min_length_qc": short_dropped,
                "revised_v_ranges": revised,
                "non_clipped": non_clipped,
                "no_height_or_transition_overlap": no_height_edge_overlap,
                "support": support,
                "sufficient_support": bool(support["support_ok"]),
            }
            object_exclusions.extend(applied)
        object_exclusions = merge_intervals(object_exclusions)
        object_ranges = [value for side in ("before", "after") for value in by_side[side]["revised_v_ranges"]]
        results.append(
            {
                "object_id": object_id,
                "object_order": int(item["object_order"]),
                "height_label_hint": item["height_label_hint"],
                "height_v_range": list(item["height_v_range"]),
                "edge_pair": {"edge1_v": edge1, "edge2_v": edge2},
                "baseline_original_v_ranges": original,
                "texture_exclusion_v_ranges": object_exclusions,
                "texture_exclusion_v_ranges_by_side": {
                    side: by_side[side]["texture_exclusion_v_ranges"] for side in ("before", "after")
                },
                "baseline_v_ranges": [
                    by_side["before"]["revised_v_ranges"],
                    by_side["after"]["revised_v_ranges"],
                ],
                "baseline_v_ranges_half_open": [
                    [half_open(value) for value in by_side["before"]["revised_v_ranges"]],
                    [half_open(value) for value in by_side["after"]["revised_v_ranges"]],
                ],
                "by_side": by_side,
                "object_revised_footprint": object_ranges,
            }
        )
    return results


def object_pairwise_nonoverlap(results: list[dict[str, Any]]) -> bool:
    footprints: list[tuple[str, list[list[int]]]] = []
    for result in results:
        footprint = [list(result["height_v_range"])] + list(result["object_revised_footprint"])
        footprints.append((result["object_id"], footprint))
    for index, (_, first) in enumerate(footprints):
        for _, second in footprints[index + 1 :]:
            if any(overlaps(left, right) for left in first for right in second):
                return False
    return True


def applied_texture_exclusions(results: list[dict[str, Any]]) -> list[list[int]]:
    """Return only exclusion ranges that intersect an original local baseline."""
    return merge_intervals(
        value
        for result in results
        for side in ("before", "after")
        for value in result["by_side"][side]["texture_exclusion_v_ranges"]
    )


def conclusions(objects: list[dict[str, Any]], results: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> dict[str, str]:
    height_unchanged = all(
        result["height_v_range"] == EXPECTED_OBJECTS[result["object_id"]]["height_v_range"] for result in results
    ) and [result["object_id"] for result in results] == list(OBJECT_IDS)
    detected = bool(transitions)
    excluded = any(result["texture_exclusion_v_ranges"] for result in results)
    both_sides = len(results) == 3 and all(
        all(
            bool(result["by_side"][side]["revised_v_ranges"])
            and result["by_side"][side]["non_clipped"]
            and result["by_side"][side]["no_height_or_transition_overlap"]
            for side in ("before", "after")
        )
        for result in results
    )
    sufficient = len(results) == 3 and all(
        all(result["by_side"][side]["sufficient_support"] for side in ("before", "after"))
        for result in results
    )
    return {
        "HEIGHT_ROIS_UNCHANGED": "YES" if height_unchanged else "NO",
        "TEXTURE_TRANSITIONS_DETECTED": "YES" if detected else "NO",
        "TEXTURE_TRANSITIONS_EXCLUDED": "YES" if excluded else "NO",
        "ALL_LOCAL_BASELINES_BOTH_SIDES": "YES" if both_sides else "NO",
        "ALL_BASELINES_HAVE_SUFFICIENT_SUPPORT": "YES" if sufficient else "NO",
        "HUMAN_REVIEW_REQUIRED": "YES",
        "THERMAL_A2_ROI_FROZEN": "NO",
    }


def transition_csv_rows(transitions: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memberships: dict[str, list[str]] = defaultdict(list)
    for result in results:
        for side in ("before", "after"):
            for group_id in result["by_side"][side]["applied_transition_group_ids"]:
                memberships[group_id].append(f"{result['object_id']}:{side}")
    rows: list[dict[str, Any]] = []
    for transition in transitions:
        evidence = transition["source_side_evidence"]
        by_side = {side: next((item for item in evidence if item["source_side"] == side), None) for side in ("left", "right")}
        rows.append(
            {
                "transition_group_id": transition["transition_group_id"],
                "transition_center_v": transition["transition_center_v"],
                "transition_exclusion_v_ranges": transition["texture_exclusion_v_ranges"],
                "source_sides": transition["source_sides"],
                "left_center_v": by_side["left"]["candidate_center_v"] if by_side["left"] else "",
                "right_center_v": by_side["right"]["candidate_center_v"] if by_side["right"] else "",
                "left_stable_repeat_count": by_side["left"]["stable_repeat_count"] if by_side["left"] else "",
                "right_stable_repeat_count": by_side["right"]["stable_repeat_count"] if by_side["right"] else "",
                "left_stable_repeat_fraction": by_side["left"]["stable_repeat_fraction"] if by_side["left"] else "",
                "right_stable_repeat_fraction": by_side["right"]["stable_repeat_fraction"] if by_side["right"] else "",
                "left_median_abs_contrast_dn": by_side["left"]["median_abs_contrast_dn"] if by_side["left"] else "",
                "right_median_abs_contrast_dn": by_side["right"]["median_abs_contrast_dn"] if by_side["right"] else "",
                "stable_repeat_count_min": transition["stable_repeat_count_min"],
                "min_stable_repeat_fraction": transition["min_stable_repeat_fraction"],
                "max_median_abs_contrast_dn": transition["max_median_abs_contrast_dn"],
                "applied_to_baselines": memberships.get(transition["transition_group_id"], []),
                "status": "ACCEPTED_IMAGE_ONLY_STABLE",
                "rule": "fixed image-only contrast/stability gate; exclusion margin +/-15 px",
            }
        )
    return rows


def all_transition_candidate_rows(side_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in ("left", "right"):
        for candidate in side_results[side]["candidates"]:
            rows.append(
                {
                    "transition_group_id": "",
                    "transition_center_v": candidate["candidate_center_v"],
                    "transition_exclusion_v_ranges": "",
                    "source_sides": side,
                    "left_center_v": candidate["candidate_center_v"] if side == "left" else "",
                    "right_center_v": candidate["candidate_center_v"] if side == "right" else "",
                    "left_stable_repeat_count": candidate["stable_repeat_count"] if side == "left" else "",
                    "right_stable_repeat_count": candidate["stable_repeat_count"] if side == "right" else "",
                    "left_stable_repeat_fraction": candidate["stable_repeat_fraction"] if side == "left" else "",
                    "right_stable_repeat_fraction": candidate["stable_repeat_fraction"] if side == "right" else "",
                    "left_median_abs_contrast_dn": candidate["median_abs_contrast_dn"] if side == "left" else "",
                    "right_median_abs_contrast_dn": candidate["median_abs_contrast_dn"] if side == "right" else "",
                    "stable_repeat_count_min": candidate["stable_repeat_count"],
                    "min_stable_repeat_fraction": candidate["stable_repeat_fraction"],
                    "max_median_abs_contrast_dn": candidate["median_abs_contrast_dn"],
                    "applied_to_baselines": "",
                    "status": "ACCEPTED_SIDE_CANDIDATE" if candidate["accepted"] else "REJECTED_FIXED_IMAGE_ONLY_GATE",
                    "rule": candidate["rejection_reason"] or "fixed image-only contrast/stability gate",
                }
            )
    return rows


def baseline_qc_rows(results: list[dict[str, Any]], conclusion: dict[str, str], pairwise_ok: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(scope: str, object_id: str, side: str, check_id: str, status: str, observed: Any, expected: str, notes: str) -> None:
        rows.append(
            {
                "scope": scope,
                "object_id": object_id,
                "baseline_side": side,
                "check_id": check_id,
                "severity": "REVIEW" if status in {"FAIL", "PENDING"} else "INFO",
                "status": status,
                "observed": observed,
                "expected": expected,
                "evidence": "thermal_roi_v2_registry_v2_draft.json + first 20 raw Mono8 PNGs",
                "notes": notes,
            }
        )

    add("global", "", "", "height_rois_unchanged", conclusion["HEIGHT_ROIS_UNCHANGED"], "upper=20mm[312,380]; middle=30mm[1496,1560]; lower=10mm[2627,2691]", "exact user-confirmed height ROIs", "R1 did not run object detection or modify height ROIs.")
    add("global", "", "", "object_detection_reused", "PASS", "A2a selected object/edge geometry loaded from draft", "no new object detection", "The A2a object order, labels, edges, and original baselines are inputs to R1.")
    add("global", "", "", "geometry_only_texture_detector", "PASS", "raw Mono8 background bands; Frozen Steger only for u(v) sampling", "no nominal/Z/calibration/thermal/error input", "No height_shadow.csv is opened.")
    add("global", "", "", "texture_transitions_detected", "PASS" if conclusion["TEXTURE_TRANSITIONS_DETECTED"] == "YES" else "REVIEW", "see thermal_roi_texture_transitions.csv/json", "at least one repeat-stable raw-image transition", "No transition passed the fixed image-only gate." if conclusion["TEXTURE_TRANSITIONS_DETECTED"] != "YES" else "Accepted transitions are listed with side evidence.")
    add("global", "", "", "texture_transitions_excluded", "PASS" if conclusion["TEXTURE_TRANSITIONS_EXCLUDED"] == "YES" else "REVIEW", conclusion["TEXTURE_TRANSITIONS_EXCLUDED"], "accepted transition intersects at least one original baseline", "Exclusions use fixed +/-15 px margins and are not tuned from height/error results.")
    add("global", "", "", "objects_non_overlapping_after_revision", "PASS" if pairwise_ok else "FAIL", pairwise_ok, "all height/revised-baseline footprints non-overlapping", "Automatic gate; human must inspect overlay.")
    add("global", "", "", "all_local_baselines_both_sides", conclusion["ALL_LOCAL_BASELINES_BOTH_SIDES"], conclusion["ALL_LOCAL_BASELINES_BOTH_SIDES"], "three objects have nonempty, non-clipped, non-overlapping before/after baselines", "Image-v geometry/QC only.")
    add("global", "", "", "all_baselines_have_sufficient_support", conclusion["ALL_BASELINES_HAVE_SUFFICIENT_SUPPORT"], conclusion["ALL_BASELINES_HAVE_SUFFICIENT_SUPPORT"], f"each retained side has >= {PREFERRED_BASELINE_SUPPORT_PX}px and 20-repeat support", "All retained subsegments are checked separately; gaps are never re-included.")
    add("global", "", "", "human_review_required", "PENDING", "draft revised baseline only", "manual overlay review before freeze", "Automatic output is never a frozen registry.")
    add("global", "", "", "thermal_a2_roi_frozen", "NO", "thermal_roi_registry_v2_frozen.json not generated", "NO before human approval", "Thermal-A2 height analysis remains blocked.")

    for result in results:
        for side in ("before", "after"):
            item = result["by_side"][side]
            support = item["support"]
            add("object_side", result["object_id"], side, "original_baseline_non_clipped", "PASS" if item["original_v_range"][0] > 0 and item["original_v_range"][1] < 2999 else "FAIL", item["original_v_range"], "original range does not touch v=0/2999", "Original A2a baseline is preserved for audit.")
            add("object_side", result["object_id"], side, "texture_exclusion_applied", "PASS" if item["texture_exclusion_v_ranges"] else "INFO", item["texture_exclusion_v_ranges"], "all accepted overlapping texture bands are listed", "No exclusion is selected using any 3-D or error quantity.")
            add("object_side", result["object_id"], side, "retained_subsegments_min_20px", "PASS" if item["revised_v_ranges"] and all(interval_length(value) >= MIN_RETAINED_INTERVAL_PX for value in item["revised_v_ranges"]) else "FAIL", {"revised": item["revised_v_ranges"], "short_dropped": item["short_subsegments_dropped_by_fixed_min_length_qc"]}, f"every retained subsegment >= {MIN_RETAINED_INTERVAL_PX}px", "The retained list, not a dropped residual fragment, is the formal baseline candidate.")
            add("object_side", result["object_id"], side, "short_residual_fragment_dropped", "REVIEW" if item["short_subsegments_dropped_by_fixed_min_length_qc"] else "INFO", item["short_subsegments_dropped_by_fixed_min_length_qc"], "any residual shorter than the fixed 20px minimum is explicitly disclosed", "A short residual is not re-included; human should confirm the visible exclusion boundary.")
            add("object_side", result["object_id"], side, "revised_baseline_non_clipped", "PASS" if item["non_clipped"] else "FAIL", item["revised_v_ranges"], "all retained ranges are non-clipped", "Sensor boundary is not a formal local baseline.")
            add("object_side", result["object_id"], side, "revised_baseline_no_height_transition_overlap", "PASS" if item["no_height_or_transition_overlap"] else "FAIL", item["revised_v_ranges"], "no overlap with fixed height/edge transition footprint", "Height ROI and edge positions were not changed.")
            add("object_side", result["object_id"], side, "revised_baseline_support", "PASS" if item["sufficient_support"] else "FAIL", {"total_width_px": support["total_width_px"], "min_points": support["min_points"], "median_support_fraction": support["median_support_fraction"], "intervals": support["intervals"]}, f"total width >= {PREFERRED_BASELINE_SUPPORT_PX}px; each interval and every repeat >= {MIN_POINTS_PER_REPEAT} points; median support >= {MIN_SUPPORT_FRACTION}", "Frozen Steger support is measured per retained interval and over the disjoint union.")
    return rows


def build_exclusion_payload(
    args: argparse.Namespace,
    draft: dict[str, Any],
    frames: list[a2a.SourceFrame],
    a1_row: dict[str, str],
    profiles: dict[str, Any],
    side_results: dict[str, dict[str, Any]],
    transitions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    conclusion: dict[str, str],
) -> dict[str, Any]:
    recording_path = frames[0].recording_path
    return {
        "schema_version": 1,
        "analysis_id": "thermal_a2a_r1_baseline_texture_exclusion_0827",
        "status": "DRAFT_REVIEW_REQUIRED",
        "geometry_only": True,
        "human_review_required": True,
        "human_reviewed": False,
        "frozen": False,
        "thermal_a2_roi_frozen": False,
        "input": {
            "recording_root": str(args.input_dir.resolve()),
            "source_recording": frames[0].recording_id,
            "source_recording_path": str(recording_path.resolve()),
            "source_frame_count": len(frames),
            "source_frame_files": [frame.filename for frame in frames],
            "a1_index_path": str(args.a1_index.resolve()),
            "a1_index_sha256": a2a.sha256_file(args.a1_index),
            "a1_raw_recording_integrity": a1_row.get("raw_recording_integrity"),
            "a2a_draft_path": str(args.draft.resolve()),
            "a2a_draft_sha256": a2a.sha256_file(args.draft),
            "frames_csv_sha256": a2a.sha256_file(recording_path / "frames.csv"),
            "raw_png_sha256": {frame.filename: a2a.sha256_file(frame.image_path) for frame in frames},
        },
        "reuse_audit": {
            "reused": [
                "Thermal-A1 first reliable recording selection and raw integrity provenance",
                "Thermal-A2a object order, user-confirmed 20/30/10 mm labels, edge positions, height ROIs, and original baseline intervals",
                "Frozen Steger extraction configuration and raw PNG/frames.csv source",
                "Auto ROI V2 image-v support and non-overlap concepts",
            ],
            "new_this_run": [
                "20-frame raw Mono8 two-sided background texture profiles",
                "repeat-stable image-only transition candidates and unified +/-15 px exclusion ranges",
                "disjoint baseline subtraction, interval-level Frozen Steger support QC, revised draft, and overlay",
            ],
            "not_used": [
                "height_shadow.csv",
                "nominal height",
                "C0/C1 XYZ",
                "Session/Local Z",
                "Base/H1/H-B2",
                "thermal drift",
                "residual/error result",
            ],
        },
        "sampling": {
            "image_format": "Mono8 raw PNG",
            "centerline": "Frozen Steger u(v) from each of the same 20 frames",
            "background_bands": {
                "left": [f"center_u-{BACKGROUND_OUTER_DISTANCE_PX}", f"center_u-{BACKGROUND_INNER_GAP_PX}"],
                "right": [f"center_u+{BACKGROUND_INNER_GAP_PX}", f"center_u+{BACKGROUND_OUTER_DISTANCE_PX}"],
                "stripe_avoidance_inner_gap_px": BACKGROUND_INNER_GAP_PX,
                "outer_distance_px": BACKGROUND_OUTER_DISTANCE_PX,
                "per_row_statistic": "median intensity in each side band",
            },
            "frames_with_valid_left_profile_fraction": float(np.mean(np.isfinite(profiles["left"]))),
            "frames_with_valid_right_profile_fraction": float(np.mean(np.isfinite(profiles["right"]))),
        },
        "fixed_image_only_detector": {
            "smoothing_sigma_px": TEXTURE_SMOOTHING_SIGMA_PX,
            "peak_distance_px": TRANSITION_PEAK_DISTANCE_PX,
            "peak_min_prominence_dn_per_v": TRANSITION_PEAK_MIN_PROMINENCE_DN_PER_V,
            "contrast_window_px": TRANSITION_CONTRAST_WINDOW_PX,
            "minimum_contrast_dn": TRANSITION_MIN_CONTRAST_DN,
            "minimum_stable_fraction": TRANSITION_MIN_STABLE_FRACTION,
            "transition_fusion_tolerance_px": TRANSITION_FUSION_TOLERANCE_PX,
            "exclusion_margin_px_each_side": TEXTURE_EXCLUSION_MARGIN_PX,
            "retained_interval_min_length_px": MIN_RETAINED_INTERVAL_PX,
            "selection_inputs": ["raw Mono8 side-band intensity", "Frozen Steger stripe u(v) only"],
        },
        "side_detector_summary": {
            side: {
                "candidate_count": len(side_results[side]["candidates"]),
                "accepted_count": len(side_results[side]["accepted"]),
                "derivative_noise_dn_per_v": side_results[side]["derivative_noise_dn_per_v"],
                "min_prominence_dn_per_v": side_results[side]["min_prominence_dn_per_v"],
                "candidates": side_results[side]["candidates"],
            }
            for side in ("left", "right")
        },
        "transitions": transitions,
        "detected_global_texture_exclusion_v_ranges": merge_intervals(
            value for transition in transitions for value in transition["texture_exclusion_v_ranges"]
        ),
        "applied_texture_exclusion_v_ranges": applied_texture_exclusions(results),
        "texture_exclusion_v_ranges": applied_texture_exclusions(results),
        "objects": results,
        "conclusions": conclusion,
        "review_required": [
            "confirm all three height ROIs remain exactly unchanged",
            "inspect raw/median-centerline overlay for every exclusion and retained baseline subsegment",
            "confirm black-white texture transition interpretation and both-side Ground support",
            "after explicit human confirmation only, create the single frozen registry and manual review record",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_a2a_status": draft.get("status"),
    }


def build_revised_registry(draft: dict[str, Any], payload: dict[str, Any], results: list[dict[str, Any]], conclusion: dict[str, str]) -> dict[str, Any]:
    original_by_id = {str(item["object_id"]): item for item in draft["objects"]}
    objects: list[dict[str, Any]] = []
    result_by_id = {result["object_id"]: result for result in results}
    for object_id in OBJECT_IDS:
        original = copy.deepcopy(original_by_id[object_id])
        result = result_by_id[object_id]
        original_baselines = [list(value) for value in result["baseline_original_v_ranges"]]
        original["height_v_range"] = list(result["height_v_range"])
        original["height_v_range_half_open"] = half_open(result["height_v_range"])
        original["baseline_original_v_ranges"] = original_baselines
        original["baseline_original_v_ranges_half_open"] = [half_open(value) for value in original_baselines]
        original["baseline_v_ranges"] = result["baseline_v_ranges"]
        original["baseline_v_ranges_half_open"] = result["baseline_v_ranges_half_open"]
        original["texture_exclusion_v_ranges"] = result["texture_exclusion_v_ranges"]
        original["texture_exclusion_v_ranges_half_open"] = [half_open(value) for value in result["texture_exclusion_v_ranges"]]
        original["texture_exclusion_v_ranges_by_side"] = result["texture_exclusion_v_ranges_by_side"]
        original["baseline_revision_rule"] = "subtract unified raw-image transition bands from original before/after intervals; preserve disjoint subsegments"
        original["baseline_revision_qc"] = result["by_side"]
        original["height_roi_unchanged"] = result["height_v_range"] == EXPECTED_OBJECTS[object_id]["height_v_range"]
        original["auto_qc_status"] = "REVIEW_REQUIRED"
        original["auto_qc_reasons"] = ["R1 revised baseline draft requires human overlay review before freeze"]
        original["geometry_only"] = True
        objects.append(original)
    return {
        "schema_version": 3,
        "analysis_id": "thermal_a2a_roi_v2_0827_r1_baseline_texture_exclusion",
        "registry_type": "thermal_multi_object_roi_v2",
        "status": "DRAFT_REVISED_BASELINE_REVIEW_REQUIRED",
        "geometry_only": True,
        "human_review_required": True,
        "human_reviewed": False,
        "manual_decision": "PENDING",
        "frozen": False,
        "thermal_a2_roi_frozen": False,
        "source_recording": payload["input"]["source_recording"],
        "source_frame_count": payload["input"]["source_frame_count"],
        "selection_basis": "A2a object/height geometry reused unchanged; R1 baseline revision from raw Mono8 texture only",
        "height_rois_unchanged": conclusion["HEIGHT_ROIS_UNCHANGED"] == "YES",
        "detected_global_texture_exclusion_v_ranges": payload["detected_global_texture_exclusion_v_ranges"],
        "texture_exclusion_v_ranges": payload["applied_texture_exclusion_v_ranges"],
        "texture_exclusion_source": {
            "artifact": "thermal_roi_texture_exclusion.json",
            "rule": "two side background bands; fixed image-only repeat-stable transition gate; +/-15 px margin",
            "forbidden_inputs": payload["reuse_audit"]["not_used"],
        },
        "objects": objects,
        "automatic_conclusions": {key: value == "YES" for key, value in conclusion.items() if key != "HUMAN_REVIEW_REQUIRED"},
        "conclusions": conclusion,
        "review_required": payload["review_required"],
        "generated_at_utc": payload["generated_at_utc"],
    }


def save_overlay(
    path: Path,
    first_frame: a2a.SourceFrame,
    median_centerline: np.ndarray,
    profiles: dict[str, Any],
    transitions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    v_grid = np.arange(first_frame.height, dtype=np.float64)
    fig, (ax_image, ax_texture) = plt.subplots(
        1, 2, figsize=(16, 13), gridspec_kw={"width_ratios": [1.1, 1.0]}, constrained_layout=True
    )
    ax_image.imshow(first_frame.image, cmap="gray", origin="upper", aspect="auto")
    ax_image.plot(
        median_centerline[:, 0] - first_frame.offset_x,
        median_centerline[:, 1],
        color="white",
        linewidth=0.8,
        alpha=0.95,
    )
    for result in results:
        object_id = result["object_id"]
        color = COLORS[object_id]
        height_start, height_end = result["height_v_range"]
        ax_image.axhspan(height_start, height_end, color=color, alpha=0.28)
        for side in ("before", "after"):
            side_result = result["by_side"][side]
            original = side_result["original_v_range"]
            ax_image.axhspan(original[0], original[1], facecolor="none", edgecolor=color, hatch="//", linewidth=0.0, alpha=0.40)
            for revised in side_result["revised_v_ranges"]:
                ax_image.axhspan(revised[0], revised[1], color="#2ca25f", alpha=0.18)
            for exclusion in side_result["texture_exclusion_v_ranges"]:
                ax_image.axhspan(exclusion[0], exclusion[1], color="#e31a1c", alpha=0.32, hatch="xx")
        edge1, edge2 = result["edge_pair"]["edge1_v"], result["edge_pair"]["edge2_v"]
        ax_image.axhline(edge1, color=color, linestyle="--", linewidth=0.9)
        ax_image.axhline(edge2, color=color, linestyle="--", linewidth=0.9)
        ax_image.text(
            5,
            (height_start + height_end) / 2,
            f"{object_id} ({result['height_label_hint']})",
            color="white",
            fontsize=9,
            va="center",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
        )
    ax_image.set_title("First reliable raw Mono8 PNG + R1 baseline revision")
    ax_image.set_xlabel("local ROI u (px)")
    ax_image.set_ylabel("v (px)")
    ax_image.set_xlim(0, first_frame.width)
    ax_image.set_ylim(first_frame.height - 1, 0)

    ax_texture.plot(profiles["left_median"], v_grid, color="#1f78b4", linewidth=0.9, label="left background median")
    ax_texture.plot(profiles["right_median"], v_grid, color="#ff7f00", linewidth=0.9, label="right background median")
    for result in results:
        color = COLORS[result["object_id"]]
        for side in ("before", "after"):
            side_result = result["by_side"][side]
            original = side_result["original_v_range"]
            ax_texture.axhspan(original[0], original[1], facecolor="none", edgecolor=color, hatch="//", linewidth=0.0, alpha=0.30)
            for revised in side_result["revised_v_ranges"]:
                ax_texture.axhspan(revised[0], revised[1], color="#2ca25f", alpha=0.14)
            for exclusion in side_result["texture_exclusion_v_ranges"]:
                ax_texture.axhspan(exclusion[0], exclusion[1], color="#e31a1c", alpha=0.24, hatch="xx")
    for transition in transitions:
        for exclusion in transition["texture_exclusion_v_ranges"]:
            ax_texture.axhspan(exclusion[0], exclusion[1], color="#e31a1c", alpha=0.08)
        ax_texture.axhline(transition["transition_center_v"], color="#b2182b", linewidth=0.8, linestyle=":")
    ax_texture.set_title("Two-sided raw background texture profile and exclusions")
    ax_texture.set_xlabel("median Mono8 intensity (DN)")
    ax_texture.set_ylabel("v (px)")
    ax_texture.set_xlim(0, max(10.0, float(np.nanmax([profiles["left_median"], profiles["right_median"]])) + 2.0))
    ax_texture.set_ylim(first_frame.height - 1, 0)
    ax_texture.grid(alpha=0.15)

    handles = [
        Line2D([0], [0], color="white", linewidth=1, label="median Frozen Steger u(v)"),
        Patch(facecolor="none", edgecolor="#555555", hatch="//", label="original baseline"),
        Patch(facecolor="#2ca25f", alpha=0.25, label="revised baseline subsegment"),
        Patch(facecolor="#e31a1c", alpha=0.35, hatch="xx", label="texture exclusion"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1, label="fixed edge"),
    ]
    ax_image.legend(handles=handles, loc="lower right", fontsize=8)
    ax_texture.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Thermal-A2a-R1 baseline texture-transition exclusion — DRAFT / HUMAN REVIEW REQUIRED / NOT FROZEN",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_report(path: Path, payload: dict[str, Any], results: list[dict[str, Any]], conclusion: dict[str, str], pairwise_ok: bool) -> None:
    lines = [
        "# Thermal-A2a-R1｜Baseline texture-transition exclusion",
        "",
        "> 本报告停在人工审核前。自动结果是 revised baseline draft，不是 frozen registry；本轮未生成 `thermal_roi_registry_v2_frozen.json`，也未生成 `thermal_roi_v2_manual_review.md`。",
        "",
        "## 结论",
        "",
    ]
    lines.extend(f"- `{key} = {value}`" for key, value in conclusion.items())
    lines.extend(
        [
            f"- `OBJECTS_NON_OVERLAPPING_AFTER_REVISION = {'YES' if pairwise_ok else 'NO'}`",
            "",
            "## 固定边界",
            "",
            f"首个可靠 recording 为 `{payload['input']['source_recording']}`，使用其 20 张 raw Mono8 PNG。沿每帧 Frozen Steger stripe 的左右两侧采样背景带，内侧避让 {BACKGROUND_INNER_GAP_PX}px，外侧边界为 {BACKGROUND_OUTER_DISTANCE_PX}px；只用原始图像强度与 stripe u(v)。",
            f"transition gate 固定为平滑 sigma={TEXTURE_SMOOTHING_SIGMA_PX}px、peak distance={TRANSITION_PEAK_DISTANCE_PX}px、contrast window={TRANSITION_CONTRAST_WINDOW_PX}px、median contrast≥{TRANSITION_MIN_CONTRAST_DN} DN、20 repeats 稳定比例≥{TRANSITION_MIN_STABLE_FRACTION:.2f}；每个接受的 transition 使用统一 ±{TEXTURE_EXCLUSION_MARGIN_PX}px exclusion。",
            "本轮保持 upper=20 mm、middle=30 mm、lower=10 mm 以及 height ROI `312–380`、`1496–1560`、`2627–2691` 原值；不重新做 object detection，不读 height_shadow.csv，不使用 nominal height、XYZ、calibration、thermal、residual/error 结果。",
            "",
            "## Revised baseline",
            "",
        "| object | side | original | texture exclusion | retained intervals | short residual dropped | total support px | min points/repeat | median support | status |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for result in results:
        for side in ("before", "after"):
            item = result["by_side"][side]
            support = item["support"]
            status = "PASS" if item["sufficient_support"] and item["non_clipped"] and item["no_height_or_transition_overlap"] else "REVIEW/FAIL"
            lines.append(
                f"| {result['object_id']} ({result['height_label_hint']}) | {side} | `{item['original_v_range']}` | `{item['texture_exclusion_v_ranges']}` | `{item['revised_v_ranges']}` | `{item['short_subsegments_dropped_by_fixed_min_length_qc']}` | {support['total_width_px']} | {support['min_points']} | {support['median_support_fraction']:.4f} | {status} |"
            )
    lines.extend(
        [
            "",
            "## Texture transition audit",
            "",
            f"接受的 unified transition 数量：`{len(payload['transitions'])}`；其中与原始 local baseline 相交并实际应用的 exclusion ranges：`{payload['applied_texture_exclusion_v_ranges']}`。全帧检测到但未落入三个原始 baseline 的 exclusion 另列在 `thermal_roi_texture_exclusion.json` 的 `detected_global_texture_exclusion_v_ranges`；没有按 Z/height/error 选择或扩大 exclusion。",
            "",
            "## Provenance / reuse audit",
            "",
            "复用：Thermal-A1 的首个可靠 recording provenance、Thermal-A2a 的 object/edge/height/original-baseline draft、Frozen Steger 配置和 Auto ROI V2 的 image-v QC 语义。",
            "本轮新增：20 帧两侧 raw Mono8 texture profile、repeat-stable transition detection、统一 exclusion、disjoint baseline subtraction、interval-level Frozen Steger support QC、revised registry draft 和 overlay。",
            "明确未使用：`height_shadow.csv`、nominal height、C0/C1 XYZ、Session/Local Z、Base/H1/H-B2、thermal drift、residual/error。",
            "",
            "## 人工审核门槛",
            "",
            "请先检查 `thermal_roi_v2_overlay_revised.png`：height ROI 必须不变；红色 hatch 必须对应可见的黑白纹理交界；绿色区段必须是 exclusion 后实际使用的全部 baseline 子段；before/after 两侧均应完整且不与 transition/height 重叠。",
            "审核者确认前禁止生成 `thermal_roi_registry_v2_frozen.json`，也禁止进入 Thermal-A2 高度精度分析。确认时应把每侧最终 interval、exclusion 来源、审核者和时间写入单一 frozen registry 与 manual review record。",
            "",
            "## 输出",
            "",
            "- `thermal_roi_texture_transitions.csv`：accepted unified transition 与所有 side candidates 的图像审计记录。",
            "- `thermal_roi_texture_exclusion.json`：detector、transition、exclusion、source hash 与结论。",
            "- `thermal_roi_v2_baseline_revised_qc.csv`：object/side interval-level support/QC。",
            "- `thermal_roi_v2_overlay_revised.png`：原始图、median centerline、edge、height、original/revised baseline 与 exclusion overlay。",
            "- `thermal_roi_registry_v2_revised_draft.json`：仅 draft，`frozen=false`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.draft = args.draft.resolve()
    args.a1_index = args.a1_index.resolve()
    args.measure_config = args.measure_config.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / "thermal_roi_registry_v2_frozen.json"
    manual_path = args.output_dir / "thermal_roi_v2_manual_review.md"
    if frozen_path.exists():
        raise TextureExclusionError(f"existing frozen registry found; R1 will not modify it: {frozen_path}")

    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    if not isinstance(draft, dict):
        raise TextureExclusionError("A2a draft JSON must be an object")
    objects = validate_draft(draft)

    recording_id, recording_path, a1_row = a2a.choose_first_recording(args.input_dir, args.a1_index)
    if recording_id != RECORDING_ID:
        raise TextureExclusionError(f"expected first reliable recording {RECORDING_ID}, got {recording_id}")
    app_config = a2a.load_app_config(args.measure_config)
    if app_config.extraction_method != "steger":
        raise TextureExclusionError(f"Frozen Steger extraction required, got {app_config.extraction_method!r}")
    extraction_params = a2a.create_extraction_params(
        app_config.extraction_method,
        app_config.extraction_options_by_method.get(app_config.extraction_method, {}),
    )
    frames = a2a.load_source_frames(recording_id, recording_path, extraction_params)
    if len(frames) != REPEAT_COUNT:
        raise TextureExclusionError(f"expected exactly {REPEAT_COUNT} source frames, got {len(frames)}")
    if a1_row and str(a1_row.get("raw_recording_integrity", "")).upper() != "PASS":
        raise TextureExclusionError("A1 source recording is not raw_recording_integrity=PASS")

    median_centerline = a2a.median_centerline([frame.centers_uv_full for frame in frames])
    profiles = sample_texture_profiles(frames)
    side_results = {side: detect_side_transitions(profiles[side], side) for side in ("left", "right")}
    transitions = fuse_transitions(side_results, frames[0].height)
    revised_results = build_object_results(objects, transitions, frames)
    conclusion = conclusions(objects, revised_results, transitions)
    pairwise_ok = object_pairwise_nonoverlap(revised_results)

    exclusion_payload = build_exclusion_payload(
        args, draft, frames, a1_row, profiles, side_results, transitions, revised_results, conclusion
    )
    revised_registry = build_revised_registry(draft, exclusion_payload, revised_results, conclusion)
    transition_rows = transition_csv_rows(transitions, revised_results)
    transition_rows = all_transition_candidate_rows(side_results) + transition_rows
    transition_fields = [
        "transition_group_id", "transition_center_v", "transition_exclusion_v_ranges", "source_sides",
        "left_center_v", "right_center_v", "left_stable_repeat_count", "right_stable_repeat_count",
        "left_stable_repeat_fraction", "right_stable_repeat_fraction", "left_median_abs_contrast_dn",
        "right_median_abs_contrast_dn", "stable_repeat_count_min", "min_stable_repeat_fraction",
        "max_median_abs_contrast_dn", "applied_to_baselines", "status", "rule",
    ]
    qc_fields = [
        "scope", "object_id", "baseline_side", "check_id", "severity", "status", "observed", "expected", "evidence", "notes"
    ]
    write_csv(args.output_dir / "thermal_roi_texture_transitions.csv", transition_fields, transition_rows)
    write_json(args.output_dir / "thermal_roi_texture_exclusion.json", exclusion_payload)
    write_csv(args.output_dir / "thermal_roi_v2_baseline_revised_qc.csv", qc_fields, baseline_qc_rows(revised_results, conclusion, pairwise_ok))
    write_json(args.output_dir / "thermal_roi_registry_v2_revised_draft.json", revised_registry)
    save_overlay(args.output_dir / "thermal_roi_v2_overlay_revised.png", frames[0], median_centerline, profiles, transitions, revised_results)
    render_report(args.output_dir / "thermal_roi_v2_baseline_revised_report.md", exclusion_payload, revised_results, conclusion, pairwise_ok)

    print(f"Thermal-A2a-R1 source recording: {recording_id}; frames={len(frames)}")
    print(f"Accepted unified texture transitions: {len(transitions)}")
    print(json.dumps(conclusion, ensure_ascii=False, sort_keys=True))
    print(f"Draft outputs: {args.output_dir}")
    print(f"Frozen registry created: {frozen_path.exists()}")
    print(f"Manual review file existed before/after: {manual_path.exists()} (R1 does not create it)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, TextureExclusionError) as error:
        print(f"thermal_a2a_r1_texture_exclusion: {error}", file=sys.stderr)
        raise SystemExit(1)
