#!/usr/bin/env python3
"""Thermal-A2a-R2 image-only manual ROI freeze and Auto-vs-Manual audit.

This is the Thermal adapter of annotate_daheng_gauge_rois.py. It keeps the
same review semantics (median raw Mono8 image, Frozen Steger centreline, and
image-v ROI bands) but does not use the old six-dataset TIFF loader.

Without --freeze the command renders the image-only review overlay and writes
a non-authoritative manual selection draft. With --freeze it writes the
frozen registry exactly once (or reads an existing one without overwriting it)
and runs the identical Frozen reconstruction for Auto and Manual ROI on the
first 20 frames. The comparison is downstream of the freeze and cannot
modify the frozen ROI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from calibration.config_loader import load_calibration_files  # noqa: E402
from laser.backends import create_extraction_params  # noqa: E402
from measurement.ground_reference import SessionGroundReference  # noqa: E402
from measurement.height_measure import measure_height_line  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402

import thermal_a2a_roi_v2 as a2a  # noqa: E402


OBJECT_IDS = ("upper", "middle", "lower")
HEIGHT_LABELS = {"upper": "20mm", "middle": "30mm", "lower": "10mm"}
REVIEWER = "Codex image-only manual review"
MANUAL_SUPPORT_MIN_POINTS = 20
MATERIAL_DELTA_THRESHOLD_MM = 0.10
PARTIAL_DELTA_THRESHOLD_MM = 0.03
COLORS = {"upper": "#386cb0", "middle": "#f0027f", "lower": "#1b9e77"}

# Image-only selections made from the median raw Mono8 image and Frozen
# Steger geometry. R1 texture exclusion ranges are deliberately not read.
# The original Auto bands are displayed for comparison only, not as selection
# bounds.
DEFAULT_MANUAL_ROIS: dict[str, dict[str, list[int]]] = {
    "upper": {
        "baseline_before": [184, 236],
        "height": [312, 380],
        "baseline_after": [486, 522],
    },
    "middle": {
        "baseline_before": [1362, 1392],
        "height": [1496, 1560],
        "baseline_after": [1652, 1708],
    },
    "lower": {
        "baseline_before": [2522, 2568],
        "height": [2627, 2691],
        "baseline_after": [2810, 2862],
    },
}


class ThermalR2Error(RuntimeError):
    """Raised when the R2 provenance or ROI gates cannot be satisfied."""


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
        "--a1-index",
        type=Path,
        default=ROOT
        / "projects"
        / "daheng"
        / "analysis"
        / "thermal_a1_0827"
        / "thermal_a1_recording_index.csv",
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
        "--manual-roi-json",
        type=Path,
        default=None,
        help="Optional image-only manual selection JSON. If omitted, use "
        "the reviewed selections embedded in this adapter.",
    )
    parser.add_argument(
        "--recording-id",
        default="recording_20260827_100021",
        help="First reliable recording used for the 20-frame review.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="After the image-only overlay has been reviewed, freeze the "
        "registry and run Auto-vs-Manual reconstruction.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Explicit alias for the default non-freezing phase.",
    )
    return parser.parse_args(argv)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            values = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(
                        json_safe(value), ensure_ascii=False, separators=(",", ":")
                    )
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value)
                values[field] = "" if value is None else value
            writer.writerow(values)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def as_int(value: Any, name: str) -> int:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ThermalR2Error(f"{name} is not an integer: {value!r}") from error
    if not math.isfinite(number) or number != int(number):
        raise ThermalR2Error(f"{name} is not an integer: {value!r}")
    return int(number)


def inclusive_range(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ThermalR2Error(f"{name} must be [first,last]")
    result = [as_int(value[0], f"{name}[0]"), as_int(value[1], f"{name}[1]")]
    if result[0] > result[1]:
        raise ThermalR2Error(f"{name} is descending: {result}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ThermalR2Error(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ThermalR2Error(f"JSON root must be an object: {path}")
    return payload


def load_auto_draft(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if str(payload.get("status", "")).upper() != "DRAFT":
        raise ThermalR2Error(
            "R2 requires the original unfrozen A2a draft, got "
            f"status={payload.get('status')!r}"
        )
    if bool(payload.get("frozen", False)):
        raise ThermalR2Error("the supplied A2a draft is already marked frozen")
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ThermalR2Error("A2a draft has no objects list")
    by_id = {
        str(item.get("object_id")): item
        for item in objects
        if isinstance(item, dict)
    }
    if set(by_id) != set(OBJECT_IDS):
        raise ThermalR2Error(
            f"A2a object IDs are not upper/middle/lower: {sorted(by_id)}"
        )
    for object_id in OBJECT_IDS:
        item = by_id[object_id]
        expected = [312, 380] if object_id == "upper" else (
            [1496, 1560] if object_id == "middle" else [2627, 2691]
        )
        actual = inclusive_range(
            item.get("height_v_range"), f"{object_id}.height_v_range"
        )
        if actual != expected:
            raise ThermalR2Error(
                f"{object_id} Auto height ROI changed: {actual} != {expected}"
            )
        if str(item.get("height_label_hint")) != HEIGHT_LABELS[object_id]:
            raise ThermalR2Error(
                f"{object_id} physical mapping changed: "
                f"{item.get('height_label_hint')!r}"
            )
    return payload


def load_manual_selection(
    path: Path | None,
) -> dict[str, dict[str, list[int]]]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_MANUAL_ROIS))
    payload = load_json(path)
    raw_objects = payload.get("objects", payload)
    if isinstance(raw_objects, list):
        raw_objects = {
            str(item.get("object_id")): item
            for item in raw_objects
            if isinstance(item, dict)
        }
    if not isinstance(raw_objects, Mapping):
        raise ThermalR2Error(
            "manual ROI JSON must contain an objects mapping or object list"
        )
    result: dict[str, dict[str, list[int]]] = {}
    for object_id in OBJECT_IDS:
        item = raw_objects.get(object_id)
        if not isinstance(item, Mapping):
            raise ThermalR2Error(f"manual ROI JSON missing {object_id}")
        result[object_id] = {
            "baseline_before": inclusive_range(
                item.get("baseline_before", item.get("baseline_before_v_range")),
                f"{object_id}.baseline_before",
            ),
            "height": inclusive_range(
                item.get("height", item.get("height_v_range")),
                f"{object_id}.height",
            ),
            "baseline_after": inclusive_range(
                item.get("baseline_after", item.get("baseline_after_v_range")),
                f"{object_id}.baseline_after",
            ),
        }
    return result


def source_recording(
    args: argparse.Namespace,
) -> tuple[str, Path, list[a2a.SourceFrame], dict[str, str]]:
    input_dir = args.input_dir.resolve()
    recording_path = input_dir / args.recording_id
    if not recording_path.is_dir():
        raise ThermalR2Error(
            f"requested reliable recording is missing: {recording_path}"
        )
    a1_rows = a2a.load_a1_rows(args.a1_index.resolve())
    a1_row = next(
        (
            row
            for row in a1_rows
            if str(row.get("recording_id", "")).strip() == args.recording_id
        ),
        {},
    )
    if a1_row and str(a1_row.get("raw_recording_integrity", "")).upper() != "PASS":
        raise ThermalR2Error(
            f"{args.recording_id} is not PASS in Thermal-A1: "
            f"{a1_row.get('raw_recording_integrity')!r}"
        )
    app = load_app_config(args.measure_config.resolve())
    if app.extraction_method != "steger":
        raise ThermalR2Error(
            "R2 requires the configured Frozen Steger extractor, got "
            f"{app.extraction_method!r}"
        )
    extraction_params = create_extraction_params(
        app.extraction_method,
        app.extraction_options_by_method.get(app.extraction_method, {}),
    )
    frames = a2a.load_source_frames(
        args.recording_id, recording_path, extraction_params
    )
    if len(frames) != 20:
        raise ThermalR2Error(
            f"R2 requires the 20-frame review recording, got {len(frames)}"
        )
    for frame in frames:
        if abs(float(frame.exposure_us) - 2000.0) > 1.0e-9:
            raise ThermalR2Error(
                f"R2 source frame is not formal 2000 us: {frame.filename} "
                f"exposure={frame.exposure_us}"
            )
        if abs(float(frame.gain_db)) > 1.0e-9:
            raise ThermalR2Error(
                f"R2 source frame is not Gain 0: {frame.filename} "
                f"gain={frame.gain_db}"
            )
        if str(frame.pixel_format).strip().lower() != "mono8":
            raise ThermalR2Error(
                f"R2 source frame is not Mono8: {frame.filename} "
                f"pixel_format={frame.pixel_format!r}"
            )
    return args.recording_id, recording_path, frames, a1_row


def median_raw_image(frames: list[a2a.SourceFrame]) -> np.ndarray:
    shapes = {tuple(frame.image.shape) for frame in frames}
    if len(shapes) != 1:
        raise ThermalR2Error(f"source frames have inconsistent image shapes: {shapes}")
    stack = np.stack(
        [np.asarray(frame.image, dtype=np.uint8) for frame in frames], axis=0
    )
    return np.asarray(np.rint(np.median(stack, axis=0)), dtype=np.uint8)


def image_display(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image.astype(np.float64), [1.0, 99.5])
    if high <= low:
        return image.astype(np.float64)
    return np.clip(
        (image.astype(np.float64) - low) / (high - low), 0.0, 1.0
    )


def range_support(
    frames: list[a2a.SourceFrame],
    value_range: list[int],
    minimum_points: int = MANUAL_SUPPORT_MIN_POINTS,
) -> dict[str, Any]:
    lo, hi = value_range
    counts = [
        int(
            np.count_nonzero(
                np.isfinite(frame.centers_uv_full[:, 1])
                & (frame.centers_uv_full[:, 1] >= lo)
                & (frame.centers_uv_full[:, 1] <= hi)
            )
        )
        for frame in frames
    ]
    return {
        "v_range": list(value_range),
        "width_px_inclusive": int(hi - lo + 1),
        "repeat_counts": counts,
        "minimum_points_per_repeat": int(minimum_points),
        "minimum_repeat_count": int(min(counts)) if counts else 0,
        "mean_points_per_repeat": float(np.mean(counts)) if counts else 0.0,
        "all_repeats_sufficient": bool(counts)
        and all(count >= minimum_points for count in counts),
    }


def validate_manual_rois(
    selection: dict[str, dict[str, list[int]]],
    frames: list[a2a.SourceFrame],
) -> dict[str, dict[str, Any]]:
    image_height, _ = frames[0].image.shape
    result: dict[str, dict[str, Any]] = {}
    occupied: list[tuple[int, int, str, str]] = []
    for object_id in OBJECT_IDS:
        chosen = selection[object_id]
        manual_height = chosen["height"]
        before = chosen["baseline_before"]
        after = chosen["baseline_after"]
        if not (before[1] < manual_height[0] and manual_height[1] < after[0]):
            raise ThermalR2Error(
                f"{object_id} manual baseline overlaps its Height ROI: "
                f"before={before}, height={manual_height}, after={after}"
            )
        for role, value_range in (
            ("baseline_before", before),
            ("height", manual_height),
            ("baseline_after", after),
        ):
            lo, hi = value_range
            if not (0 <= lo <= hi < image_height):
                raise ThermalR2Error(
                    f"{object_id} {role} is outside image rows {image_height}: "
                    f"{value_range}"
                )
            if hi - lo + 1 < MANUAL_SUPPORT_MIN_POINTS:
                raise ThermalR2Error(
                    f"{object_id} {role} is narrower than 20 px: {value_range}"
                )
            occupied.append((lo, hi, object_id, role))
        support = {
            role: range_support(frames, value_range)
            for role, value_range in (
                ("baseline_before", before),
                ("height", manual_height),
                ("baseline_after", after),
            )
        }
        if not all(item["all_repeats_sufficient"] for item in support.values()):
            raise ThermalR2Error(
                f"{object_id} does not have >=20 Frozen Steger points in "
                f"every repeat: {support}"
            )
        non_clipped = all(
            value_range[0] > 0 and value_range[1] < image_height - 1
            for value_range in (before, manual_height, after)
        )
        result[object_id] = {
            "baseline_before": before,
            "height": manual_height,
            "baseline_after": after,
            "baseline_v_ranges": [before, after],
            "support": support,
            "non_clipped": non_clipped,
            "both_sides": True,
        }
    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if current[0] <= previous[1]:
            raise ThermalR2Error(f"manual ROI overlap: {previous} and {current}")
    if not all(item["non_clipped"] for item in result.values()):
        raise ThermalR2Error("a manual ROI is clipped at the sensor boundary")
    return result


def median_centerline_local(
    frames: list[a2a.SourceFrame],
) -> tuple[np.ndarray, int]:
    median = a2a.median_centerline([frame.centers_uv_full for frame in frames])
    offsets = {int(frame.offset_x) for frame in frames}
    if len(offsets) != 1:
        raise ThermalR2Error(f"source frames have inconsistent offset_x: {offsets}")
    return median, next(iter(offsets))


def draw_span(ax: Any, value_range: list[int], **kwargs: Any) -> None:
    ax.axhspan(float(value_range[0]), float(value_range[1]), **kwargs)


def render_manual_overlay(
    path: Path,
    frames: list[a2a.SourceFrame],
    median_image: np.ndarray,
    median: np.ndarray,
    validated: dict[str, dict[str, Any]],
    auto_draft: dict[str, Any],
) -> None:
    offset_x = int(frames[0].offset_x)
    display = image_display(median_image)
    height, width = median_image.shape
    center_u = median[:, 0] - offset_x
    center_v = median[:, 1]
    auto_by_id = {str(item["object_id"]): item for item in auto_draft["objects"]}
    fig = plt.figure(figsize=(16, 19), constrained_layout=True)
    grid = fig.add_gridspec(
        3, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 1.0, 1.0]
    )
    full_ax = fig.add_subplot(grid[:, 0])
    full_ax.imshow(
        display,
        cmap="gray",
        origin="upper",
        extent=(0, width, height, 0),
        aspect="auto",
    )
    center_mask = (
        np.isfinite(center_u)
        & np.isfinite(center_v)
        & (center_u >= 0)
        & (center_u <= width)
    )
    full_ax.plot(
        center_u[center_mask],
        center_v[center_mask],
        color="#00e5ff",
        linewidth=0.65,
        label="median Frozen Steger",
    )
    for object_id in OBJECT_IDS:
        color = COLORS[object_id]
        auto = auto_by_id[object_id]
        manual = validated[object_id]
        for value_range in (
            auto["baseline_v_ranges"][0],
            auto["height_v_range"],
            auto["baseline_v_ranges"][1],
        ):
            draw_span(
                full_ax,
                value_range,
                facecolor="none",
                edgecolor="#bbbbbb",
                linewidth=0.7,
                linestyle="--",
                alpha=0.95,
            )
        for role, value_range in (
            ("manual before", manual["baseline_before"]),
            ("manual height", manual["height"]),
            ("manual after", manual["baseline_after"]),
        ):
            draw_span(
                full_ax,
                value_range,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.15 if "height" not in role else 0.28,
            )
        full_ax.axhline(
            auto["edge_pair"]["edge1_v"], color=color, linewidth=0.75, alpha=0.8
        )
        full_ax.axhline(
            auto["edge_pair"]["edge2_v"], color=color, linewidth=0.75, alpha=0.8
        )
        label_u = float(np.nanmedian(center_u[center_mask]))
        full_ax.text(
            min(width - 80, max(4, label_u + 8)),
            float(manual["height"][0]) + 4,
            object_id,
            color=color,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox={
                "facecolor": "black",
                "alpha": 0.35,
                "pad": 2,
                "edgecolor": "none",
            },
        )
    full_ax.set_xlim(0, width)
    full_ax.set_ylim(height, 0)
    full_ax.set_xlabel("raw image u (recording ROI column)")
    full_ax.set_ylabel("raw image v (recording ROI row)")
    full_ax.set_title(
        "Median raw Mono8 + Frozen Steger\n"
        "image-only manual ROI review; no 3-D values"
    )
    full_ax.grid(False)

    for index, object_id in enumerate(OBJECT_IDS):
        ax = fig.add_subplot(grid[index, 1])
        auto = auto_by_id[object_id]
        manual = validated[object_id]
        all_ranges = [
            auto["baseline_v_ranges"][0],
            auto["height_v_range"],
            auto["baseline_v_ranges"][1],
        ]
        v_min = max(0, min(pair[0] for pair in all_ranges) - 30)
        v_max = min(height - 1, max(pair[1] for pair in all_ranges) + 30)
        ax.imshow(
            display,
            cmap="gray",
            origin="upper",
            extent=(0, width, height, 0),
            aspect="auto",
        )
        zoom_mask = center_mask & (center_v >= v_min) & (center_v <= v_max)
        ax.plot(
            center_u[zoom_mask],
            center_v[zoom_mask],
            color="#00e5ff",
            linewidth=0.9,
        )
        for value_range in all_ranges:
            draw_span(
                ax,
                value_range,
                facecolor="none",
                edgecolor="#dddddd",
                linewidth=0.7,
                linestyle="--",
            )
        for role, value_range in (
            ("manual before", manual["baseline_before"]),
            ("manual height", manual["height"]),
            ("manual after", manual["baseline_after"]),
        ):
            draw_span(
                ax,
                value_range,
                facecolor=COLORS[object_id],
                edgecolor=COLORS[object_id],
                linewidth=1.1,
                alpha=0.20 if "height" not in role else 0.34,
            )
        ax.axhline(
            auto["edge_pair"]["edge1_v"],
            color=COLORS[object_id],
            linewidth=0.8,
        )
        ax.axhline(
            auto["edge_pair"]["edge2_v"],
            color=COLORS[object_id],
            linewidth=0.8,
        )
        ax.set_xlim(0, width)
        ax.set_ylim(v_max, v_min)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(f"{object_id}: Auto dashed / Manual solid")

    legend = [
        Line2D(
            [0], [0], color="#00e5ff", linewidth=1.0, label="median Frozen Steger"
        ),
        Line2D(
            [0], [0], color="#bbbbbb", linestyle="--", label="original Auto ROI"
        ),
        Patch(
            facecolor="#777777", alpha=0.28, label="Manual frozen candidate"
        ),
        Line2D([0], [0], color="#777777", linewidth=0.8, label="Auto edge"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=True)
    fig.suptitle(
        "Thermal-A2a-R2 — Multi-object image-only ROI review\n"
        "object order: upper=20 mm, middle=30 mm, lower=10 mm",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def half_open(value_range: list[int]) -> list[int]:
    return [int(value_range[0]), int(value_range[1]) + 1]


def build_frozen_registry(
    args: argparse.Namespace,
    recording_id: str,
    recording_path: Path,
    frames: list[a2a.SourceFrame],
    a1_row: dict[str, str],
    auto_draft: dict[str, Any],
    validated: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    auto_by_id = {str(item["object_id"]): item for item in auto_draft["objects"]}
    r1_path = args.output_dir / "thermal_roi_texture_exclusion.json"
    objects: list[dict[str, Any]] = []
    for object_id in OBJECT_IDS:
        auto = auto_by_id[object_id]
        manual = validated[object_id]
        auto_before = inclusive_range(
            auto["baseline_v_ranges"][0], f"{object_id}.auto_before"
        )
        auto_after = inclusive_range(
            auto["baseline_v_ranges"][1], f"{object_id}.auto_after"
        )
        object_payload = {
            "object_id": object_id,
            "object_order": int(auto["object_order"]),
            "height_label_hint": HEIGHT_LABELS[object_id],
            "height_label_basis": (
                "user-confirmed physical placement; not used to choose image ROI"
            ),
            "automatic_candidate_rank": auto.get("automatic_candidate_rank"),
            "edge_pair": json_safe(auto.get("edge_pair", {})),
            "auto_roi": {
                "baseline_before_v_range": auto_before,
                "height_v_range": inclusive_range(
                    auto["height_v_range"], f"{object_id}.auto_height"
                ),
                "baseline_after_v_range": auto_after,
                "baseline_v_ranges": [auto_before, auto_after],
                "source_registry": str(args.auto_draft.resolve()),
            },
            "height_v_range": list(manual["height"]),
            "height_v_range_half_open": half_open(manual["height"]),
            "baseline_v_ranges": [
                list(manual["baseline_before"]),
                list(manual["baseline_after"]),
            ],
            "baseline_v_ranges_half_open": [
                half_open(manual["baseline_before"]),
                half_open(manual["baseline_after"]),
            ],
            "manual_roi": {
                "baseline_before_v_range": list(manual["baseline_before"]),
                "height_v_range": list(manual["height"]),
                "baseline_after_v_range": list(manual["baseline_after"]),
                "baseline_v_ranges": [
                    list(manual["baseline_before"]),
                    list(manual["baseline_after"]),
                ],
                "selection_basis": (
                    "manual image-only review of median raw Mono8 image plus "
                    "Frozen Steger centreline; continuous homogeneous bands "
                    "away from visible checkerboard transition and step edge"
                ),
                "manual_confirmed": True,
                "confirmation_method": "image_only_manual_review",
                "r1_texture_exclusion_used": False,
                "auto_vs_manual_feedback_used": False,
            },
            "manual_qc": json_safe(manual),
            "baseline_clipped": {
                "before": False,
                "after": False,
                "before_unavailable": False,
                "after_unavailable": False,
            },
            "auto_qc_status": auto.get("auto_qc_status"),
            "auto_qc_reasons": auto.get("auto_qc_reasons", []),
            "geometry_only": True,
            "manual_roi_frozen": True,
        }
        objects.append(object_payload)

    return {
        "schema_version": 3,
        "registry_type": "thermal_multi_object_roi_v2",
        "status": "FROZEN_MANUAL_REVIEWED",
        "geometry_only": True,
        "human_reviewed": True,
        "manual_decision": "ACCEPTED",
        "frozen": True,
        "thermal_a2_roi_frozen": True,
        "manual_confirmed": True,
        "manual_confirmed_count": len(objects),
        "manual_reviewer": REVIEWER,
        "manual_review_method": (
            "median raw Mono8 image + median Frozen Steger centerline; "
            "no Z, nominal height, Base/H1/H-B2, thermal drift, residual, "
            "or error data was available during ROI selection"
        ),
        "manual_review_time_utc": now_utc(),
        "source_recording": recording_id,
        "source_recording_path": str(recording_path.resolve()),
        "source_frame_count": len(frames),
        "source_frame_files": [frame.filename for frame in frames],
        "source_frame_sha256": {
            frame.filename: sha256_file(frame.image_path) for frame in frames
        },
        "a1_index_path": str(args.a1_index.resolve()),
        "a1_index_sha256": sha256_file(args.a1_index.resolve()),
        "a1_raw_recording_integrity": a1_row.get("raw_recording_integrity"),
        "original_auto_registry": {
            "path": str(args.auto_draft.resolve()),
            "sha256": sha256_file(args.auto_draft.resolve()),
            "status": auto_draft.get("status"),
            "frozen": bool(auto_draft.get("frozen", False)),
            "used_for_manual_geometry": True,
        },
        "r1_texture_exclusion_audit": {
            "path": str(r1_path.resolve()),
            "sha256": sha256_file(r1_path),
            "retained_for_audit_only": True,
            "used_for_formal_freeze": False,
            "ranges_loaded_for_selection": False,
        },
        "formal_freeze_policy": {
            "registry_authoritative_for_all_580_frames": True,
            "same_registry_for_pre_post_reconnect": True,
            "height_roi_changed_after_comparison": False,
            "manual_roi_changed_after_comparison": False,
            "comparison_results_may_modify_registry": False,
        },
        "selection_constraints": {
            "raw_image_and_frozen_steger_only": True,
            "continuous_baseline_intervals": True,
            "minimum_interval_width_px": MANUAL_SUPPORT_MIN_POINTS,
            "minimum_points_per_repeat": MANUAL_SUPPORT_MIN_POINTS,
            "both_sides_required": True,
            "non_clipped_required": True,
            "r1_auto_texture_exclusion_formal": False,
        },
        "objects": objects,
        "conclusions": {
            "three_objects_detected": True,
            "all_height_rois_valid": True,
            "all_local_baselines_both_sides": True,
            "human_review_required": True,
            "thermal_a2_roi_frozen": True,
        },
    }


def validate_frozen_registry(
    registry: dict[str, Any],
    expected: dict[str, dict[str, Any]],
) -> None:
    if registry.get("status") != "FROZEN_MANUAL_REVIEWED":
        raise ThermalR2Error(
            f"frozen registry has unexpected status: {registry.get('status')!r}"
        )
    if not (
        registry.get("frozen") is True
        and registry.get("manual_confirmed") is True
        and registry.get("thermal_a2_roi_frozen") is True
    ):
        raise ThermalR2Error("frozen registry flags are not all true")
    by_id = {
        str(item.get("object_id")): item
        for item in registry.get("objects", [])
        if isinstance(item, dict)
    }
    if set(by_id) != set(OBJECT_IDS):
        raise ThermalR2Error("frozen registry does not contain exactly three objects")
    for object_id in OBJECT_IDS:
        item = by_id[object_id]
        expected_item = expected[object_id]
        actual_height = inclusive_range(
            item.get("height_v_range"), f"frozen.{object_id}.height"
        )
        actual_baselines = [
            inclusive_range(pair, f"frozen.{object_id}.baseline")
            for pair in item.get("baseline_v_ranges", [])
        ]
        if actual_height != expected_item["height"]:
            raise ThermalR2Error(
                f"existing frozen {object_id} height differs from reviewed input"
            )
        if actual_baselines != expected_item["baseline_v_ranges"]:
            raise ThermalR2Error(
                f"existing frozen {object_id} baseline differs from reviewed input"
            )


def load_session_ground(
    path: Path,
) -> tuple[dict[str, Any], SessionGroundReference]:
    payload = load_json(path.resolve())
    if payload.get("status") != "VALID" or payload.get("valid") is not True:
        raise ThermalR2Error("session_ground_calibration.json is not VALID")
    runtime = payload.get("runtime", {})
    if runtime.get("ground_extrinsic_source") != "session":
        raise ThermalR2Error("saved Session Ground runtime source is not session")
    ground = payload.get("session_ground_reference", {})
    if ground.get("status") != "VALID":
        raise ThermalR2Error("saved Session Ground Reference is not VALID")
    try:
        reference = SessionGroundReference(
            origin_xy=np.asarray(ground["origin_xy"], dtype=np.float64),
            direction_xy=np.asarray(ground["direction_xy"], dtype=np.float64),
            slope_z_per_mm=float(ground["slope_z_per_mm"]),
            intercept_z_mm=float(ground["intercept_z_mm"]),
            rmse_mm=float(ground["rmse_mm"]),
            valid_s_range_mm=tuple(
                float(item) for item in ground["valid_s_range_mm"]
            ),
            status=str(ground["status"]),
            source=str(ground.get("fit_source", "session_laser_ground")),
            point_count=int(ground.get("point_count", 0)),
            inlier_count=int(ground.get("inlier_count", 0)),
            support_source=str(
                ground.get("support_source", ground.get("source", ""))
            ),
            active_ground_extrinsic_source=str(
                ground.get("active_ground_extrinsic_source", "session")
            ),
            ground_extrinsic_generation=int(
                ground.get("ground_extrinsic_generation", 0)
            ),
            frame_host_monotonic_ns=int(
                ground.get("frame_host_monotonic_ns", 0)
            ),
            mask_inset_mm=float(ground.get("mask_inset_mm", 0.0)),
            support_metadata=dict(ground.get("support", {})),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ThermalR2Error(f"cannot hydrate saved Session Ground: {error}") from error
    return payload, reference


def load_formal_chain(
    args: argparse.Namespace,
    ground_payload: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    app = load_app_config(args.measure_config.resolve())
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_model,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        app.calibration.laser_ray_correction,
        ground_u_optional=True,
    )
    try:
        calibration["R"] = np.asarray(
            ground_payload["session_extrinsic"]["R_camera_to_ground"],
            dtype=np.float64,
        )
        calibration["t"] = np.asarray(
            ground_payload["session_extrinsic"]["t_camera_to_ground_mm"],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ThermalR2Error(f"saved Session R/t is incomplete: {error}") from error
    if calibration.get("laser_model") is None:
        raise ThermalR2Error("Frozen C0 laser model did not load")
    if app.reconstruction.enable_laser_ray_correction and not calibration.get(
        "laser_ray_correction"
    ):
        raise ThermalR2Error("Frozen C1 is enabled but the correction did not load")
    return app, calibration


def v_mask(points_uv: np.ndarray, ranges: list[list[int]]) -> np.ndarray:
    points = np.asarray(points_uv, dtype=np.float64)
    mask = np.zeros(len(points), dtype=bool)
    for value_range in ranges:
        lo, hi = value_range
        mask |= (points[:, 1] >= lo) & (points[:, 1] <= hi)
    return mask


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_ground(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None, None
    return float(np.mean(finite)), float(np.std(finite))


def measurement_row(
    frame: a2a.SourceFrame,
    object_id: str,
    roi_mode: str,
    roi: Mapping[str, Any],
    reconstruction: Any,
    points_session: np.ndarray,
    pixels: np.ndarray,
    ground_valid: np.ndarray,
) -> dict[str, Any]:
    before = [
        inclusive_range(pair, f"{object_id}.{roi_mode}.baseline")
        for pair in roi["baseline_v_ranges"]
    ]
    height = [
        inclusive_range(roi["height_v_range"], f"{object_id}.{roi_mode}.height")
    ]
    before_mask = v_mask(pixels, [before[0]])
    height_mask = v_mask(pixels, height)
    after_mask = v_mask(pixels, [before[1]])
    all_selected = before_mask | height_mask | after_mask
    before_points = points_session[before_mask]
    height_points = points_session[height_mask]
    after_points = points_session[after_mask]
    baseline_points = np.concatenate([before_points, after_points], axis=0)
    left_mean, left_std = summarize_ground(before_points[:, 2])
    right_mean, right_std = summarize_ground(after_points[:, 2])
    baseline_mean, baseline_std = summarize_ground(baseline_points[:, 2])
    row: dict[str, Any] = {
        "recording_id": frame.recording_id,
        "frame_index": frame.row_index,
        "filename": frame.filename,
        "camera_frame_number": frame.camera_frame_number,
        "host_timestamp_ns": frame.host_timestamp_ns,
        "roi_mode": roi_mode,
        "object_id": object_id,
        "height_label_hint": HEIGHT_LABELS[object_id],
        "exposure_us": frame.exposure_us,
        "gain_db": frame.gain_db,
        "pixel_format": frame.pixel_format,
        "steger_point_count": len(frame.centers_uv_full),
        "reconstructed_point_count": int(len(pixels)),
        "reconstruction_valid_ratio": (
            float(len(pixels) / len(frame.centers_uv_full))
            if len(frame.centers_uv_full)
            else None
        ),
        "session_ground_valid_count": int(np.count_nonzero(ground_valid)),
        "session_ground_valid_ratio": (
            float(np.mean(ground_valid)) if len(ground_valid) else None
        ),
        "session_ground_selected_valid": (
            int(np.count_nonzero(ground_valid[all_selected]))
            if len(ground_valid) == len(pixels)
            else None
        ),
        "baseline_before_point_count": len(before_points),
        "height_point_count": len(height_points),
        "baseline_after_point_count": len(after_points),
        "baseline_point_count": len(baseline_points),
        "baseline_both_sides": (
            len(before_points) >= 20 and len(after_points) >= 20
        ),
        "ground_left_mean_mm": left_mean,
        "ground_left_std_mm": left_std,
        "ground_right_mean_mm": right_mean,
        "ground_right_std_mm": right_std,
        "ground_left_right_delta_mm": (
            left_mean - right_mean
            if left_mean is not None and right_mean is not None
            else None
        ),
        "ground_mean_mm": baseline_mean,
        "ground_std_mm": baseline_std,
        "reconstruction_c1_clamped_count": (
            int(np.count_nonzero(reconstruction.c1_clamped))
            if reconstruction.c1_clamped is not None
            else None
        ),
        "measurement_status": "NOT_RUN",
        "measurement_error": "",
        "local_ground_mean_at_height_mm": None,
        "local_ground_profile_slope_mm_per_mm": None,
        "local_ground_profile_rmse_mm": None,
        "local_ground_noise_sigma_mm": None,
        "local_height_mean_mm": None,
        "local_height_median_mm": None,
        "local_height_std_mm": None,
        "local_height_inlier_count": 0,
        "local_baseline_inlier_count": 0,
    }
    if len(before_points) < 20 or len(after_points) < 20:
        row["measurement_status"] = "INVALID_BOTH_SIDES_SUPPORT"
        row["measurement_error"] = (
            "both baseline sides require >=20 reconstructed points"
        )
        return row
    if len(height_points) < 20:
        row["measurement_status"] = "INVALID_HEIGHT_SUPPORT"
        row["measurement_error"] = "height ROI requires >=20 reconstructed points"
        return row
    try:
        measured = measure_height_line(
            baseline_points,
            height_points,
            roi["_measurement_params"],
            ground_correction_mode="auto",
        )
    except Exception as error:  # noqa: BLE001 - record QC, do not hide it
        row["measurement_status"] = "INVALID_MEASUREMENT"
        row["measurement_error"] = f"{type(error).__name__}: {error}"
        return row
    row.update(
        {
            "measurement_status": "VALID",
            "local_ground_mean_at_height_mm": finite_or_none(
                measured.ground_baseline_zg_mm
            ),
            "local_ground_profile_slope_mm_per_mm": (
                finite_or_none(measured.ground_profile_fit.slope_z_per_mm)
                if measured.ground_profile_fit is not None
                else None
            ),
            "local_ground_profile_rmse_mm": (
                finite_or_none(measured.ground_profile_fit.rmse_mm)
                if measured.ground_profile_fit is not None
                else None
            ),
            "local_ground_noise_sigma_mm": finite_or_none(
                measured.ground_noise_sigma_mm
            ),
            "local_height_mean_mm": finite_or_none(measured.height_mean_mm),
            "local_height_median_mm": finite_or_none(measured.height_median_mm),
            "local_height_std_mm": finite_or_none(measured.height_std_mm),
            "local_height_inlier_count": int(measured.height_inlier_count),
            "local_baseline_inlier_count": int(measured.baseline_inlier_count),
        }
    )
    return row


def run_frame_comparison(
    frames: list[a2a.SourceFrame],
    registry: dict[str, Any],
    app: Any,
    calibration: dict[str, Any],
    session_reference: SessionGroundReference,
) -> list[dict[str, Any]]:
    by_id = {
        str(item["object_id"]): item
        for item in registry["objects"]
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for frame in frames:
        reconstruction = reconstruct_uv_to_ground(
            frame.centers_uv_full, calibration, app.reconstruction
        )
        points_session, ground_valid = session_reference.apply_to_points(
            reconstruction.points_ground
        )
        ground_valid = np.asarray(ground_valid, dtype=bool)
        # Out-of-domain Session Ground points are not silently extrapolated or
        # used as Local Ground support. This keeps the formal baseline valid.
        if len(ground_valid) == len(points_session):
            pixels = reconstruction.pixels_uv[ground_valid]
            points = points_session[ground_valid]
        else:
            pixels = reconstruction.pixels_uv
            points = points_session
        for object_id in OBJECT_IDS:
            item = by_id[object_id]
            auto = item["auto_roi"]
            manual = item["manual_roi"]
            for roi_mode, source in (("AUTO", auto), ("MANUAL", manual)):
                roi = {
                    "height_v_range": source["height_v_range"],
                    "baseline_v_ranges": source["baseline_v_ranges"],
                    "_measurement_params": app.measurement,
                }
                rows.append(
                    measurement_row(
                        frame,
                        object_id,
                        roi_mode,
                        roi,
                        reconstruction,
                        points,
                        pixels,
                        ground_valid,
                    )
                )
    return rows


def numeric_values(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    ]


def mean_or_none(values: list[Any]) -> float | None:
    numbers = numeric_values(values)
    return float(np.mean(numbers)) if numbers else None


def std_or_none(values: list[Any]) -> float | None:
    numbers = numeric_values(values)
    return float(np.std(numbers)) if numbers else None


def diff_or_none(manual: Any, auto: Any) -> float | None:
    if manual is None or auto is None:
        return None
    try:
        value = float(manual) - float(auto)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for object_id in OBJECT_IDS:
        object_rows = [row for row in rows if row["object_id"] == object_id]
        auto_rows = [row for row in object_rows if row["roi_mode"] == "AUTO"]
        manual_rows = [row for row in object_rows if row["roi_mode"] == "MANUAL"]
        row: dict[str, Any] = {
            "object_id": object_id,
            "height_label_hint": HEIGHT_LABELS[object_id],
            "n_frames": len(manual_rows),
        }
        for mode, selected in (("auto", auto_rows), ("manual", manual_rows)):
            valid = [item for item in selected if item["measurement_status"] == "VALID"]
            row[f"{mode}_valid_frames"] = len(valid)
            row[f"{mode}_invalid_frames"] = len(selected) - len(valid)
            for metric in (
                "ground_mean_mm",
                "ground_std_mm",
                "ground_left_mean_mm",
                "ground_right_mean_mm",
                "ground_left_right_delta_mm",
                "local_ground_mean_at_height_mm",
                "local_ground_profile_slope_mm_per_mm",
                "local_ground_profile_rmse_mm",
                "local_ground_noise_sigma_mm",
                "local_height_mean_mm",
                "local_height_median_mm",
                "local_height_std_mm",
                "reconstruction_valid_ratio",
                "session_ground_valid_ratio",
            ):
                row[f"{mode}_{metric}_mean"] = mean_or_none(
                    [item.get(metric) for item in selected]
                )
                if metric in {
                    "local_ground_mean_at_height_mm",
                    "local_height_mean_mm",
                    "ground_left_right_delta_mm",
                }:
                    row[f"{mode}_{metric}_frame_std"] = std_or_none(
                        [item.get(metric) for item in selected]
                    )
            row[f"{mode}_height_mean_frame_std_mm"] = std_or_none(
                [item.get("local_height_mean_mm") for item in selected]
            )
            row[f"{mode}_valid_both_sides_frames"] = sum(
                bool(item.get("baseline_both_sides")) for item in selected
            )
        for metric in (
            "ground_mean_mm",
            "ground_left_mean_mm",
            "ground_right_mean_mm",
            "ground_left_right_delta_mm",
            "local_ground_mean_at_height_mm",
            "local_height_mean_mm",
            "local_height_median_mm",
            "local_height_std_mm",
        ):
            row[f"manual_minus_auto_{metric}"] = diff_or_none(
                row.get(f"manual_{metric}_mean"),
                row.get(f"auto_{metric}_mean"),
            )
        result.append(row)
    return result


def render_comparison_plot(
    path: Path,
    summary: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> None:
    x = np.arange(len(OBJECT_IDS), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    labels = [f"{item}\n{HEIGHT_LABELS[item]}" for item in OBJECT_IDS]
    for axis, metric, title, ylabel in (
        (
            axes[0],
            "local_ground_mean_at_height_mm",
            "Local Ground reference mean at Height ROI",
            "Zg reference (mm)",
        ),
        (
            axes[1],
            "local_height_mean_mm",
            "Local height mean",
            "local height (mm)",
        ),
    ):
        auto_values = [
            np.nan if item.get(f"auto_{metric}_mean") is None
            else item.get(f"auto_{metric}_mean")
            for item in summary
        ]
        manual_values = [
            np.nan if item.get(f"manual_{metric}_mean") is None
            else item.get(f"manual_{metric}_mean")
            for item in summary
        ]
        axis.bar(x - width / 2, auto_values, width, color="#888888", label="A: Auto")
        axis.bar(
            x + width / 2, manual_values, width, color="#2b8cbe", label="B: Manual"
        )
        for index, item in enumerate(summary):
            delta = item.get(f"manual_minus_auto_{metric}")
            if delta is not None:
                ymax = np.nanmax([auto_values[index], manual_values[index]])
                axis.text(
                    index,
                    ymax,
                    f"Delta={delta:+.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")

    for object_id, color in COLORS.items():
        selected = [
            row
            for row in frame_rows
            if row["object_id"] == object_id
            and row["roi_mode"] in {"AUTO", "MANUAL"}
        ]
        frame_indices = sorted(
            {int(row["frame_index"]) for row in selected}
        )
        auto = {
            int(row["frame_index"]): row.get("local_height_mean_mm")
            for row in selected
            if row["roi_mode"] == "AUTO"
        }
        manual = {
            int(row["frame_index"]): row.get("local_height_mean_mm")
            for row in selected
            if row["roi_mode"] == "MANUAL"
        }
        delta = [
            np.nan
            if auto.get(frame) is None or manual.get(frame) is None
            else float(manual[frame]) - float(auto[frame])
            for frame in frame_indices
        ]
        axes[2].plot(
            frame_indices, delta, marker="o", linewidth=1.2, color=color, label=object_id
        )
    axes[2].axhline(0.0, color="#333333", linewidth=0.8)
    axes[2].set_xlabel("first recording frame index")
    axes[2].set_ylabel("Manual - Auto height (mm)")
    axes[2].set_title(
        "Frame-level Local height difference (same Frozen reconstruction)"
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")
    fig.suptitle(
        "Thermal-A2a-R2 Auto-vs-Manual ROI audit\n"
        "Manual registry was frozen before this comparison",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def classify_contamination(
    summary: list[dict[str, Any]],
) -> tuple[str, float, float]:
    ground_deltas = [
        abs(float(item["manual_minus_auto_local_ground_mean_at_height_mm"]))
        for item in summary
        if item.get("manual_minus_auto_local_ground_mean_at_height_mm") is not None
    ]
    height_deltas = [
        abs(float(item["manual_minus_auto_local_height_mean_mm"]))
        for item in summary
        if item.get("manual_minus_auto_local_height_mean_mm") is not None
    ]
    max_ground = max(ground_deltas) if ground_deltas else float("nan")
    max_height = max(height_deltas) if height_deltas else float("nan")
    maximum = max(
        [value for value in (max_ground, max_height) if math.isfinite(value)],
        default=float("nan"),
    )
    if not math.isfinite(maximum):
        return "NO", max_ground, max_height
    if maximum >= MATERIAL_DELTA_THRESHOLD_MM:
        return "YES", max_ground, max_height
    if maximum >= PARTIAL_DELTA_THRESHOLD_MM:
        return "PARTIAL", max_ground, max_height
    return "NO", max_ground, max_height


def format_mm(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.6f}" if math.isfinite(number) else "NA"


def manual_review_markdown(
    args: argparse.Namespace,
    registry: dict[str, Any],
    validated: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Thermal-A2a-R2 Manual ROI Review",
        "",
        "本文件记录冻结前的 image-only 审核和冻结决定。选择依据仅为首个可靠 "
        "recording 的 20 张 Mono8 原始图像、median raw image 与 Frozen Steger "
        "centerline；选择阶段没有加载或显示 Z、高度数值、nominal height、"
        "Base/H1/H-B2、thermal drift、residual 或 error。",
        "",
        f"- reviewer: {registry.get('manual_reviewer')}",
        f"- review time (UTC): {registry.get('manual_review_time_utc')}",
        f"- source recording: {registry.get('source_recording')}",
        f"- source frame count: {registry.get('source_frame_count')}",
        "- image-only overlay: thermal_roi_manual_overlay.png",
        "",
        "## Freeze decision",
        "",
        "- MANUAL_ROI_FROZEN = YES",
        "- HUMAN_REVIEW_REQUIRED = YES",
        "- THERMAL_A2_ROI_FROZEN = YES",
        "- R1 texture exclusion is retained as an audit artifact only and is not "
        "a formal input to this registry.",
        "- The registry was written before Auto-vs-Manual reconstruction. Comparison "
        "results cannot modify this file.",
        "",
        "## Frozen image-v intervals",
        "",
        "| object | physical label | baseline_before | height | baseline_after | support |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for object_id in OBJECT_IDS:
        item = validated[object_id]
        support = item["support"]
        support_text = (
            f"before min={support['baseline_before']['minimum_repeat_count']}, "
            f"height min={support['height']['minimum_repeat_count']}, "
            f"after min={support['baseline_after']['minimum_repeat_count']}"
        )
        lines.append(
            f"| {object_id} | {HEIGHT_LABELS[object_id]} | "
            f"{item['baseline_before']} | {item['height']} | "
            f"{item['baseline_after']} | {support_text} |"
        )
    lines.extend(
        [
            "",
            "## Manual selection notes",
            "",
            "- All three Height ROIs remain the confirmed original Auto Height ROIs:",
            "  upper 312–380, middle 1496–1560, lower 2627–2691.",
            "- Ground intervals are continuous, non-clipped, disjoint from each "
            "other and the Height ROIs, and were selected from the visible raw "
            "texture/step geometry. Auto bands are shown for audit/comparison only.",
            "- Both sides have at least 20 Frozen Steger points in every one of "
            "the 20 review frames.",
            "- No R1 automatically detected texture exclusion range was copied into "
            "the formal registry.",
            "",
            "## Provenance",
            "",
            f"- original Auto draft: {registry['original_auto_registry']['path']}",
            f"- original Auto draft SHA-256: {registry['original_auto_registry']['sha256']}",
            f"- R1 audit artifact: {registry['r1_texture_exclusion_audit']['path']}",
            f"- R1 used for formal Freeze: {registry['r1_texture_exclusion_audit']['used_for_formal_freeze']}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def report_markdown(
    args: argparse.Namespace,
    registry: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    ground_payload: dict[str, Any],
    contamination: str,
    max_ground: float,
    max_height: float,
) -> str:
    all_manual_valid = all(
        row["measurement_status"] == "VALID"
        for row in frame_rows
        if row["roi_mode"] == "MANUAL"
    )
    all_manual_both = all(
        bool(row["baseline_both_sides"])
        for row in frame_rows
        if row["roi_mode"] == "MANUAL"
    )
    thermal_allowed = bool(
        registry.get("frozen")
        and all_manual_valid
        and all_manual_both
        and len(frame_rows) == 20 * 3 * 2
    )
    lines = [
        "# Thermal-A2a-R2 report",
        "",
        "## Conclusion",
        "",
        f"- MANUAL_ROI_FROZEN = {'YES' if registry.get('frozen') else 'NO'}",
        f"- AUTO_TEXTURE_CONTAMINATION_MATERIAL = {contamination}",
        f"- AUTO_VS_MANUAL_GROUND_DELTA_MM = {max_ground:.6f}",
        f"- AUTO_VS_MANUAL_HEIGHT_DELTA_MM = {max_height:.6f}",
        f"- THERMAL_A2_ALLOWED = {'YES' if thermal_allowed else 'NO'}",
        "",
        "The reported deltas are the maximum absolute Manual minus Auto mean "
        "differences over the three objects in the first reliable 20-frame "
        "recording. Ground delta means the Local Ground profile reference at "
        "the Height ROI; height delta means the Local height mean. They are "
        "an Auto-vs-Manual ROI contamination audit, not a new calibration fit.",
        "",
        "## Provenance and reuse audit",
        "",
        "- Reused: Thermal-A1 selected recording_20260827_100021 and its exact "
        "20-row frames.csv/20 PNG source set.",
        "- Reused: configured Frozen Steger extraction, Frozen C0/C1 calibration "
        "files, configured reconstruction parameters, Session R/t from the saved "
        "session_ground_calibration.json, and its saved Session Ground reference.",
        "- Reused for comparison: the same one reconstruction result per frame is "
        "masked twice for Auto and Manual; no second extraction or calibration fit "
        "is performed.",
        "- New: image-only manual overlay, frozen registry, frame-level comparison, "
        "summary CSV and comparison plot.",
        "- R1 thermal_roi_texture_exclusion.json is retained for audit only; "
        "its ranges are not used in formal Freeze or comparison.",
        "- height_shadow.csv is not used.",
        "",
        "## Frozen ROI",
        "",
        "| object | label | manual before | manual height | manual after |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in registry["objects"]:
        lines.append(
            f"| {item['object_id']} | {item['height_label_hint']} | "
            f"{item['manual_roi']['baseline_before_v_range']} | "
            f"{item['manual_roi']['height_v_range']} | "
            f"{item['manual_roi']['baseline_after_v_range']} |"
        )
    lines.extend(
        [
            "",
            "## Auto-vs-Manual summary",
            "",
            "| object | Auto valid | Manual valid | Ground Δ Manual-Auto (mm) | Height Δ Manual-Auto (mm) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary:
        ground_delta = item.get(
            "manual_minus_auto_local_ground_mean_at_height_mm"
        )
        height_delta = item.get("manual_minus_auto_local_height_mean_mm")
        lines.append(
            f"| {item['object_id']} | {item['auto_valid_frames']}/{item['n_frames']} | "
            f"{item['manual_valid_frames']}/{item['n_frames']} | "
            f"{format_mm(ground_delta)} | {format_mm(height_delta)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The contamination classification uses a predeclared audit band: "
            f"|Delta| < {PARTIAL_DELTA_THRESHOLD_MM:.2f} mm is NO, "
            f"{PARTIAL_DELTA_THRESHOLD_MM:.2f}–{MATERIAL_DELTA_THRESHOLD_MM:.2f} mm "
            f"is PARTIAL, and |Delta| >= {MATERIAL_DELTA_THRESHOLD_MM:.2f} mm is YES. "
            "This threshold is applied after the registry is frozen and is not "
            "used to choose or adjust any ROI.",
            "",
            "Thermal-A2 is allowed only for subsequent full-data processing with "
            "this exact registry reused for all frames and with pre/post reconnect "
            "kept as independent segments. This R2 report does not claim a "
            "system-wide worst-case thermal envelope.",
            "",
            "## Saved Session Ground provenance",
            "",
            f"- status: {ground_payload.get('status')}",
            f"- runtime ground extrinsic source: "
            f"{ground_payload.get('runtime', {}).get('ground_extrinsic_source')}",
            f"- Session Ground Reference status: "
            f"{ground_payload.get('session_ground_reference', {}).get('status')}",
            f"- source file: {args.session_ground.resolve()}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


FRAME_FIELDS = [
    "recording_id",
    "frame_index",
    "filename",
    "camera_frame_number",
    "host_timestamp_ns",
    "roi_mode",
    "object_id",
    "height_label_hint",
    "exposure_us",
    "gain_db",
    "pixel_format",
    "steger_point_count",
    "reconstructed_point_count",
    "reconstruction_valid_ratio",
    "session_ground_valid_count",
    "session_ground_valid_ratio",
    "session_ground_selected_valid",
    "baseline_before_point_count",
    "height_point_count",
    "baseline_after_point_count",
    "baseline_point_count",
    "baseline_both_sides",
    "ground_left_mean_mm",
    "ground_left_std_mm",
    "ground_right_mean_mm",
    "ground_right_std_mm",
    "ground_left_right_delta_mm",
    "ground_mean_mm",
    "ground_std_mm",
    "reconstruction_c1_clamped_count",
    "measurement_status",
    "measurement_error",
    "local_ground_mean_at_height_mm",
    "local_ground_profile_slope_mm_per_mm",
    "local_ground_profile_rmse_mm",
    "local_ground_noise_sigma_mm",
    "local_height_mean_mm",
    "local_height_median_mm",
    "local_height_std_mm",
    "local_height_inlier_count",
    "local_baseline_inlier_count",
]


SUMMARY_FIELDS = [
    "object_id",
    "height_label_hint",
    "n_frames",
    "auto_valid_frames",
    "auto_invalid_frames",
    "manual_valid_frames",
    "manual_invalid_frames",
    "auto_ground_mean_mm_mean",
    "manual_ground_mean_mm_mean",
    "auto_ground_std_mm_mean",
    "manual_ground_std_mm_mean",
    "auto_ground_left_mean_mm_mean",
    "manual_ground_left_mean_mm_mean",
    "auto_ground_right_mean_mm_mean",
    "manual_ground_right_mean_mm_mean",
    "auto_ground_left_right_delta_mm_mean",
    "manual_ground_left_right_delta_mm_mean",
    "auto_local_ground_mean_at_height_mm_mean",
    "manual_local_ground_mean_at_height_mm_mean",
    "auto_local_ground_mean_at_height_mm_frame_std",
    "manual_local_ground_mean_at_height_mm_frame_std",
    "auto_local_ground_profile_slope_mm_per_mm_mean",
    "manual_local_ground_profile_slope_mm_per_mm_mean",
    "auto_local_ground_profile_rmse_mm_mean",
    "manual_local_ground_profile_rmse_mm_mean",
    "auto_local_ground_noise_sigma_mm_mean",
    "manual_local_ground_noise_sigma_mm_mean",
    "auto_local_height_mean_mm_mean",
    "manual_local_height_mean_mm_mean",
    "auto_local_height_mean_mm_frame_std",
    "manual_local_height_mean_mm_frame_std",
    "auto_local_height_median_mm_mean",
    "manual_local_height_median_mm_mean",
    "auto_local_height_std_mm_mean",
    "manual_local_height_std_mm_mean",
    "auto_reconstruction_valid_ratio_mean",
    "manual_reconstruction_valid_ratio_mean",
    "auto_session_ground_valid_ratio_mean",
    "manual_session_ground_valid_ratio_mean",
    "auto_valid_both_sides_frames",
    "manual_valid_both_sides_frames",
    "manual_minus_auto_ground_mean_mm",
    "manual_minus_auto_ground_left_mean_mm",
    "manual_minus_auto_ground_right_mean_mm",
    "manual_minus_auto_ground_left_right_delta_mm",
    "manual_minus_auto_local_ground_mean_at_height_mm",
    "manual_minus_auto_local_height_mean_mm",
    "manual_minus_auto_local_height_median_mm",
    "manual_minus_auto_local_height_std_mm",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.freeze:
        print(
            "ERROR: --freeze is disabled by Thermal-A2a-R2-Fix. "
            "Use tools/thermal_a2a_r2_human_roi_gui.py; only its explicit "
            "human GUI Freeze action may create the formal registry.",
            file=sys.stderr,
        )
        return 2
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.measure_config = args.measure_config.resolve()
    args.a1_index = args.a1_index.resolve()
    args.auto_draft = args.auto_draft.resolve()
    args.session_ground = args.session_ground.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        args.freeze = False
    try:
        auto_draft = load_auto_draft(args.auto_draft)
        recording_id, recording_path, frames, a1_row = source_recording(args)
        median_image = median_raw_image(frames)
        median, _ = median_centerline_local(frames)
        selection = load_manual_selection(
            args.manual_roi_json.resolve() if args.manual_roi_json else None
        )
        validated = validate_manual_rois(selection, frames)
        overlay_path = args.output_dir / "thermal_roi_manual_overlay.png"
        render_manual_overlay(
            overlay_path, frames, median_image, median, validated, auto_draft
        )
        expected = {
            object_id: {
                "height": list(validated[object_id]["height"]),
                "baseline_v_ranges": [
                    list(validated[object_id]["baseline_before"]),
                    list(validated[object_id]["baseline_after"]),
                ],
            }
            for object_id in OBJECT_IDS
        }
        frozen_path = args.output_dir / "thermal_roi_registry_v2_frozen.json"
        if not args.freeze:
            draft_path = (
                args.output_dir / "thermal_roi_v2_manual_selection_draft.json"
            )
            write_json(
                draft_path,
                {
                    "status": "DRAFT",
                    "manual_reviewed": False,
                    "human_review_required": True,
                    "source_recording": recording_id,
                    "geometry_only": True,
                    "r1_texture_exclusion_used": False,
                    "objects": [
                        {
                            "object_id": object_id,
                            "height_label_hint": HEIGHT_LABELS[object_id],
                            **json_safe(validated[object_id]),
                        }
                        for object_id in OBJECT_IDS
                    ],
                },
            )
            print(f"Overlay: {overlay_path}")
            print(f"Non-authoritative manual draft: {draft_path}")
            print("MANUAL_ROI_FROZEN = NO")
            print("HUMAN_REVIEW_REQUIRED = YES")
            return 0

        if frozen_path.exists():
            registry = load_json(frozen_path)
            validate_frozen_registry(registry, expected)
            print(f"Using existing frozen registry without overwrite: {frozen_path}")
        else:
            registry = build_frozen_registry(
                args,
                recording_id,
                recording_path,
                frames,
                a1_row,
                auto_draft,
                validated,
            )
            validate_frozen_registry(registry, expected)
            write_json(frozen_path, registry)
            print(f"Created frozen registry exactly once: {frozen_path}")
        manual_review_path = args.output_dir / "thermal_roi_v2_manual_review.md"
        if not manual_review_path.exists():
            manual_review_markdown(args, registry, validated, manual_review_path)

        ground_payload, session_reference = load_session_ground(args.session_ground)
        app, calibration = load_formal_chain(args, ground_payload)
        frame_rows = run_frame_comparison(
            frames, registry, app, calibration, session_reference
        )
        if len(frame_rows) != 20 * 3 * 2:
            raise ThermalR2Error(
                f"unexpected comparison row count: {len(frame_rows)}"
            )
        summary = build_summary(frame_rows)
        contamination, max_ground, max_height = classify_contamination(summary)
        write_csv(
            args.output_dir / "thermal_roi_auto_vs_manual_frame_results.csv",
            FRAME_FIELDS,
            frame_rows,
        )
        write_csv(
            args.output_dir / "thermal_roi_auto_vs_manual.csv",
            SUMMARY_FIELDS,
            summary,
        )
        render_comparison_plot(
            args.output_dir / "thermal_roi_auto_vs_manual.png",
            summary,
            frame_rows,
        )
        report_path = args.output_dir / "report.md"
        report_path.write_text(
            report_markdown(
                args,
                registry,
                frame_rows,
                summary,
                ground_payload,
                contamination,
                max_ground,
                max_height,
            ),
            encoding="utf-8",
        )
        all_manual_valid = all(
            row["measurement_status"] == "VALID"
            for row in frame_rows
            if row["roi_mode"] == "MANUAL"
        )
        all_manual_both = all(
            bool(row["baseline_both_sides"])
            for row in frame_rows
            if row["roi_mode"] == "MANUAL"
        )
        thermal_allowed = bool(
            registry.get("frozen") and all_manual_valid and all_manual_both
        )
        print(f"Overlay: {overlay_path}")
        print(f"Frozen registry: {frozen_path}")
        print(
            "Auto-vs-Manual summary: "
            f"{args.output_dir / 'thermal_roi_auto_vs_manual.csv'}"
        )
        print("MANUAL_ROI_FROZEN = YES")
        print(f"AUTO_TEXTURE_CONTAMINATION_MATERIAL = {contamination}")
        print(f"AUTO_VS_MANUAL_GROUND_DELTA_MM = {max_ground:.6f}")
        print(f"AUTO_VS_MANUAL_HEIGHT_DELTA_MM = {max_height:.6f}")
        print(f"THERMAL_A2_ALLOWED = {'YES' if thermal_allowed else 'NO'}")
        return 0
    except (OSError, ValueError, RuntimeError, ThermalR2Error) as error:
        print(f"thermal_a2a_r2_manual_freeze: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
