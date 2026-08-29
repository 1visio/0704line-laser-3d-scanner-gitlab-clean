"""Human-confirm and freeze Daheng gauge-block v-axis ROIs.

The annotation window deliberately contains only image geometry:

* a five-repeat median TIFF image;
* a five-repeat median Steger centerline;
* the current automatic candidate bands;
* editable baseline-before, height, and baseline-after v ranges.

It never displays gauge truth, C0/C1 heights, errors, or any reconstruction
result.  The final registry is written only after every selected height x
position entry has been manually confirmed.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
import warnings

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
if str(MEASUREMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_ROOT))

from app_config import load_app_config
from laser.backends import create_extraction_params
from laser.laser_extractor import extract_laser_center
from utils.image_io import load_grayscale_image


DEFAULT_DATA_ROOT = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data"
)
DEFAULT_CONFIG = (
    REPO_ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819"
DEFAULT_AUTO_REGISTRY = DEFAULT_OUTPUT / "roi_registry.json"
DEFAULT_POINTWISE_CACHE = DEFAULT_OUTPUT / "pointwise_diagnostics.csv"

DATASETS = (
    "obs_1mm",
    "obs_2mm",
    "obs_6mm",
    "obs_10mm",
    "obs_20mm",
    "obs_30mm",
)
DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}
POSE_IDS = tuple(f"{index:03d}" for index in range(1, 6))
ROI_IDS = ("baseline_before", "height", "baseline_after")
ROI_COLORS = {
    "baseline_before": "#42a5f5",
    "height": "#ffca28",
    "baseline_after": "#66bb6a",
}
TIFF_PATTERN = re.compile(
    r"^laser\s+(?P<pose>\d{3})(?:_(?P<repeat>\d{2}))?\.tif$",
    re.IGNORECASE,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_tiff_name(path: Path) -> tuple[str, int]:
    match = TIFF_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid TIFF name: {path.name}")
    return match.group("pose"), int(match.group("repeat") or "01")


def read_auto_registry(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, list):
        entries = document
        protocol = {}
        summary = {}
    elif isinstance(document, dict):
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise ValueError("auto registry JSON must contain an entries list")
        protocol = document.get("protocol") or {}
        summary = document.get("summary") or {}
    else:
        raise ValueError("auto registry must be a JSON object or list")
    expected_count = len(DATASETS) * len(POSE_IDS)
    if len(entries) != expected_count:
        raise ValueError(
            f"auto registry must contain {expected_count} entries for "
            f"{len(DATASETS)} selected datasets, got {len(entries)}"
        )
    for entry in entries:
        for field in (
            "dataset",
            "pose_id",
            "position_rank",
            "v_center_px",
            "height_v_range",
            "baseline_v_ranges",
        ):
            if field not in entry:
                raise ValueError(f"auto registry entry missing {field}")
    return (
        {
            "protocol": protocol,
            "summary": summary,
            "path": str(path.resolve()),
            "sha256": sha256(path),
        },
        entries,
    )


def read_auto_candidates(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not path.is_file():
        return defaultdict(list)
    output: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset"], row["pose_id"])
            parsed: dict[str, Any] = dict(row)
            for field in (
                "candidate_rank_by_score",
                "v_center_px",
                "depth_px",
                "prominence_px",
                "support_width_px",
                "score",
            ):
                if parsed.get(field) not in (None, ""):
                    parsed[field] = float(parsed[field])
            parsed["selected"] = str(parsed.get("selected")).lower() == "true"
            output[key].append(parsed)
    return output


def read_center_cache(
    path: Path,
) -> dict[tuple[str, str, int], np.ndarray]:
    if not path.is_file():
        return {}
    groups: dict[tuple[str, str, int], list[tuple[int, float, float]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                key = (
                    row["dataset"],
                    row["pose_id"],
                    int(row["repeat_index"]),
                )
                groups[key].append(
                    (
                        int(row["point_index"]),
                        float(row["u_px"]),
                        float(row["v_px"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    output: dict[tuple[str, str, int], np.ndarray] = {}
    for key, values in groups.items():
        values.sort(key=lambda item: item[0])
        output[key] = np.asarray(
            [(item[1], item[2]) for item in values], dtype=np.float64
        )
    expected = {
        (dataset, pose_id, repeat)
        for dataset in DATASETS
        for pose_id in POSE_IDS
        for repeat in range(1, 6)
    }
    if set(output) != expected:
        return {}
    return output


def locate_images(data_root: Path) -> dict[tuple[str, str, int], Path]:
    output: dict[tuple[str, str, int], Path] = {}
    for dataset in DATASETS:
        for path in sorted((data_root / dataset).rglob("*.tif")):
            pose_id, repeat = parse_tiff_name(path)
            key = (dataset, pose_id, repeat)
            if key in output:
                raise ValueError(f"duplicate image for {key}: {path}")
            output[key] = path
    expected = {
        (dataset, pose_id, repeat)
        for dataset in DATASETS
        for pose_id in POSE_IDS
        for repeat in range(1, 6)
    }
    missing = sorted(expected - set(output))
    if missing:
        raise FileNotFoundError(f"missing gauge-block images: {missing[:5]}")
    return output


def read_frame_offsets(data_root: Path) -> dict[tuple[str, str, int], tuple[int, int]]:
    """Read the captured frame coordinate offsets used by the online path."""
    output: dict[tuple[str, str, int], tuple[int, int]] = {}
    for dataset in DATASETS:
        frames_csv = data_root / dataset / "frames.csv"
        if not frames_csv.is_file():
            continue
        with frames_csv.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                filename = str(row.get("filename") or "")
                if not filename:
                    continue
                try:
                    pose_id, repeat = parse_tiff_name(Path(filename))
                    offset = (
                        int(float(row.get("offset_x") or 0)),
                        int(float(row.get("offset_y") or 0)),
                    )
                except (TypeError, ValueError):
                    continue
                key = (dataset, pose_id, repeat)
                previous = output.get(key)
                if previous is not None and previous != offset:
                    raise ValueError(
                        f"conflicting frame offsets for {key}: {previous} vs {offset}"
                    )
                output[key] = offset
    return output


def extract_centers_if_needed(
    data_root: Path,
    config_path: Path,
    image_paths: dict[tuple[str, str, int], Path],
    cached: dict[tuple[str, str, int], np.ndarray],
    frame_offsets: dict[tuple[str, str, int], tuple[int, int]],
) -> tuple[dict[tuple[str, str, int], np.ndarray], str]:
    if cached:
        return cached, "reused_pointwise_diagnostics"
    app = load_app_config(config_path)
    extraction_params = create_extraction_params(
        app.extraction_method,
        app.extraction_options_by_method.get(app.extraction_method, {}),
    )
    output: dict[tuple[str, str, int], np.ndarray] = {}
    for key, path in sorted(image_paths.items()):
        row_offset = frame_offsets.get(key, (0, 0))
        centers_local = np.asarray(
            extract_laser_center(
                load_grayscale_image(path),
                extraction_params,
                image_offset=row_offset,
            ),
            dtype=np.float64,
        )
        if centers_local.size == 0:
            centers_local = np.empty((0, 2), dtype=np.float64)
        output[key] = np.ascontiguousarray(
            centers_local.reshape(-1, 2)
            + np.asarray(row_offset, dtype=np.float64)
        )
    return output, "fresh_single_pass_steger_with_frame_offsets"


def median_image(paths: list[Path]) -> np.ndarray:
    images = [load_grayscale_image(path) for path in paths]
    shapes = {image.shape for image in images}
    dtypes = {str(image.dtype) for image in images}
    if len(shapes) != 1 or len(dtypes) != 1:
        raise ValueError(f"five-repeat image stack is not homogeneous: {shapes}, {dtypes}")
    stack = np.stack(images, axis=0)
    return np.rint(np.median(stack, axis=0)).astype(images[0].dtype)


def binned_centerline(
    centers_by_repeat: list[np.ndarray],
    image_height: int,
    bin_width: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    bin_v = np.arange(bin_width / 2.0, image_height, bin_width)
    profiles = np.full((len(centers_by_repeat), len(bin_v)), np.nan, dtype=np.float64)
    for repeat_index, centers in enumerate(centers_by_repeat):
        bin_index = np.floor(centers[:, 1] / bin_width).astype(int)
        for index in range(len(bin_v)):
            values = centers[bin_index == index, 0]
            if len(values):
                profiles[repeat_index, index] = float(np.median(values))
    median_u = np.full(len(bin_v), np.nan, dtype=np.float64)
    valid_columns = np.any(np.isfinite(profiles), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_u[valid_columns] = np.nanmedian(
            profiles[:, valid_columns], axis=0
        )
    return bin_v, median_u


def display_windows(
    image_shape: tuple[int, int],
    auto_ranges: dict[str, list[int]],
    center_u: np.ndarray,
) -> list[tuple[int, int, int, int, str]]:
    """Return two equal-pixel local views centered on the automatic ROI bands."""
    image_height, image_width = image_shape
    range_values = [value for item in auto_ranges.values() for value in item]
    focus_v0 = max(0, min(range_values))
    focus_v1 = min(image_height, max(range_values) + 1)
    span_v = max(1, focus_v1 - focus_v0)
    margin_v = max(80, int(round(span_v * 0.28)))
    focus_v0 = max(0, focus_v0 - margin_v)
    focus_v1 = min(image_height, focus_v1 + margin_v)

    valid = np.isfinite(center_u)
    u_center = float(np.nanmedian(center_u[valid])) if np.any(valid) else image_width / 2.0

    def make_x_window(half_width: int) -> tuple[int, int]:
        x0 = max(0, int(round(u_center)) - half_width)
        x1 = min(image_width, int(round(u_center)) + half_width)
        if x1 <= x0:
            x1 = min(image_width, x0 + 1)
        return x0, x1

    context_x0, context_x1 = make_x_window(360)
    detail_x0, detail_x1 = make_x_window(130)
    return [
        (context_x0, context_x1, focus_v0, focus_v1, "ROI context (auto bands)"),
        (detail_x0, detail_x1, focus_v0, focus_v1, "ROI detail (auto bands)"),
    ]


def draw_overlay_axis(
    axis: Any,
    rendered: np.ndarray,
    display_vmax: float,
    center_v: np.ndarray,
    center_u: np.ndarray,
    auto_ranges: dict[str, list[int]],
    manual_ranges: dict[str, list[int]] | None,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    view_label: str,
) -> None:
    axis.imshow(
        rendered[y0:y1, x0:x1],
        cmap="gray",
        vmin=0.0,
        vmax=display_vmax,
        aspect="equal",
        interpolation="nearest",
        extent=(x0, x1, y1, y0),
    )
    valid = np.isfinite(center_u)
    axis.scatter(
        center_u[valid],
        center_v[valid],
        s=4,
        color="#e040fb",
        alpha=0.9,
        linewidths=0.0,
        label="median of 5 Steger point samples (no line connection)",
    )
    for roi_id in ROI_IDS:
        color = ROI_COLORS[roi_id]
        v0, v1 = auto_ranges[roi_id]
        axis.axhspan(
            v0,
            v1,
            color=color,
            alpha=0.12,
            linestyle="--",
            label=f"auto {roi_id}",
        )
        if manual_ranges and roi_id in manual_ranges:
            m0, m1 = manual_ranges[roi_id]
            axis.axhspan(
                m0,
                m1,
                color=color,
                alpha=0.26,
                label=f"manual {roi_id}",
            )
            axis.axhline(m0, color=color, linewidth=1.3)
            axis.axhline(m1, color=color, linewidth=1.3)
    axis.set_xlim(x0, x1)
    axis.set_ylim(y1, y0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("u [px]")
    axis.set_ylabel("v [px]")
    axis.set_title(view_label)
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)


def prepare_display_image(
    image: np.ndarray,
    contrast_stretch: bool,
) -> tuple[np.ndarray, float]:
    """Prepare display-only pixels; Steger always receives the raw image."""
    if not contrast_stretch:
        if np.issubdtype(image.dtype, np.integer):
            display_vmax = float(np.iinfo(image.dtype).max)
        else:
            display_vmax = max(1.0, float(np.nanmax(image)))
        return image.astype(np.float32, copy=False), display_vmax
    low, high = np.percentile(image, [1.0, 99.8])
    rendered = np.clip(
        (image.astype(np.float32) - low) * 255.0 / max(1.0, high - low),
        0,
        255,
    )
    return rendered, 255.0


def render_overlay(
    path: Path,
    image: np.ndarray,
    center_v: np.ndarray,
    center_u: np.ndarray,
    auto_entry: dict[str, Any],
    manual_ranges: dict[str, list[int]] | None,
    image_offset_x: int,
    title: str,
    status_message: str = "",
    contrast_stretch: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    rendered, display_vmax = prepare_display_image(image, contrast_stretch)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 10),
        gridspec_kw={"width_ratios": (3, 1)},
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes[0])
    auto_ranges = {
        "baseline_before": auto_entry["baseline_v_ranges"][0],
        "height": auto_entry["height_v_range"],
        "baseline_after": auto_entry["baseline_v_ranges"][1],
    }
    for axis, (x0, x1, y0, y1, view_label) in zip(
        axes, display_windows(image.shape, auto_ranges, center_u)
    ):
        draw_overlay_axis(
            axis,
            rendered,
            display_vmax,
            center_v,
            center_u,
            auto_ranges,
            manual_ranges,
            x0,
            x1,
            y0,
            y1,
            view_label,
        )
    figure.suptitle(title)
    if status_message:
        figure.text(0.01, 0.01, status_message, ha="left", va="bottom", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def validate_range(
    roi_id: str,
    candidate: tuple[int, int],
    image_height: int,
    selected: dict[str, list[int]],
    auto_ranges: dict[str, list[int]],
) -> None:
    left, right = candidate
    if left < 0 or right >= image_height or left > right:
        raise ValueError(f"{roi_id} range must be inside [0, {image_height - 1}]")
    context = dict(auto_ranges)
    context.update(selected)
    if roi_id == "baseline_before" and right >= context["height"][0]:
        raise ValueError("baseline_before must end above height")
    if roi_id == "height":
        if left <= context["baseline_before"][1]:
            raise ValueError("height must start below baseline_before")
        if right >= context["baseline_after"][0]:
            raise ValueError("height must end above baseline_after")
    if roi_id == "baseline_after" and left <= context["height"][1]:
        raise ValueError("baseline_after must start below height")


def make_manual_entry(
    auto_entry: dict[str, Any],
    manual_ranges: dict[str, list[int]],
    center_v: np.ndarray,
) -> dict[str, Any]:
    height_range = manual_ranges["height"]
    mask = (center_v >= height_range[0]) & (center_v <= height_range[1])
    manual_v_center = (
        float(np.nanmedian(center_v[mask])) if np.any(mask) else float(auto_entry["v_center_px"])
    )
    auto_copy = json_safe(auto_entry)
    return {
        "dataset": auto_entry["dataset"],
        "pose_id": auto_entry["pose_id"],
        "auto_position_rank": auto_entry["position_rank"],
        "position_rank": auto_entry["position_rank"],
        "auto_v_center_px": float(auto_entry["v_center_px"]),
        "v_center_px": manual_v_center,
        "auto_roi": {
            "height_v_range": list(auto_entry["height_v_range"]),
            "baseline_v_ranges": [list(item) for item in auto_entry["baseline_v_ranges"]],
        },
        "manual_roi": {
            "height_v_range": list(manual_ranges["height"]),
            "baseline_v_ranges": [
                list(manual_ranges["baseline_before"]),
                list(manual_ranges["baseline_after"]),
            ],
        },
        "height_v_range": list(manual_ranges["height"]),
        "baseline_v_ranges": [
            list(manual_ranges["baseline_before"]),
            list(manual_ranges["baseline_after"]),
        ],
        "manual_confirmed": True,
        "confirmation_method": "interactive_enter_or_two_clicks",
        "geometry_only": True,
        "auto_entry_snapshot": auto_copy,
        "updated_at": now_utc(),
    }


def diff_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        auto = entry["auto_roi"]
        manual = entry["manual_roi"]
        auto_ranges = {
            "baseline_before": auto["baseline_v_ranges"][0],
            "height": auto["height_v_range"],
            "baseline_after": auto["baseline_v_ranges"][1],
        }
        manual_ranges = {
            "baseline_before": manual["baseline_v_ranges"][0],
            "height": manual["height_v_range"],
            "baseline_after": manual["baseline_v_ranges"][1],
        }
        row: dict[str, Any] = {
            "dataset": entry["dataset"],
            "pose_id": entry["pose_id"],
            "position_rank": entry["position_rank"],
            "manual_confirmed": entry["manual_confirmed"],
            "auto_v_center_px": entry["auto_v_center_px"],
            "manual_v_center_px": entry["v_center_px"],
            "v_center_delta_px": entry["v_center_px"] - entry["auto_v_center_px"],
        }
        for roi_id in ROI_IDS:
            a0, a1 = auto_ranges[roi_id]
            m0, m1 = manual_ranges[roi_id]
            row.update(
                {
                    f"auto_{roi_id}_v0": a0,
                    f"auto_{roi_id}_v1": a1,
                    f"manual_{roi_id}_v0": m0,
                    f"manual_{roi_id}_v1": m1,
                    f"{roi_id}_v0_delta": m0 - a0,
                    f"{roi_id}_v1_delta": m1 - a1,
                }
            )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (DATASET_ORDER[row["dataset"]], row["position_rank"]),
    )


def write_diff(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "pose_id",
        "position_rank",
        "manual_confirmed",
        "auto_v_center_px",
        "manual_v_center_px",
        "v_center_delta_px",
    ]
    for roi_id in ROI_IDS:
        fields.extend(
            [
                f"auto_{roi_id}_v0",
                f"auto_{roi_id}_v1",
                f"manual_{roi_id}_v0",
                f"manual_{roi_id}_v1",
                f"{roi_id}_v0_delta",
                f"{roi_id}_v1_delta",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reorder_positions(entries: list[dict[str, Any]]) -> None:
    for dataset in DATASETS:
        group = [entry for entry in entries if entry["dataset"] == dataset]
        group.sort(key=lambda entry: float(entry["v_center_px"]))
        for rank, entry in enumerate(group, 1):
            entry["position_rank"] = rank


class AnnotationAborted(RuntimeError):
    pass


def annotate_entry(
    ordinal: int,
    total: int,
    entry: dict[str, Any],
    image: np.ndarray,
    center_v: np.ndarray,
    center_u: np.ndarray,
    output_dir: Path,
    image_offset_x: int,
    contrast_stretch: bool,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    auto_ranges = {
        "baseline_before": list(entry["baseline_v_ranges"][0]),
        "height": list(entry["height_v_range"]),
        "baseline_after": list(entry["baseline_v_ranges"][1]),
    }
    current = {
        "roi_index": 0,
        "clicks": [],
        "selected": {},
        "completed": False,
        "status": "",
        "press_v": None,
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 10),
        gridspec_kw={"width_ratios": (3, 1)},
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes[0])
    windows = display_windows(image.shape, auto_ranges, center_u)
    try:
        figure.canvas.manager.set_window_title(
            f"Daheng ROI annotation {ordinal}/{total} pose {entry['pose_id']}"
        )
    except AttributeError:
        pass
    status_text = figure.text(0.01, 0.01, "", ha="left", va="bottom", fontsize=9)

    def redraw() -> None:
        rendered, display_vmax = prepare_display_image(image, contrast_stretch)
        for axis, (x0, x1, y0, y1, view_label) in zip(axes, windows):
            axis.clear()
            draw_overlay_axis(
                axis,
                rendered,
                display_vmax,
                center_v,
                center_u,
                auto_ranges,
                current["selected"],
                x0,
                x1,
                y0,
                y1,
                (
                    f"{view_label} | ROI annotation {ordinal}/{total} | "
                    f"select {ROI_IDS[current['roi_index']]} "
                    f"({current['roi_index'] + 1}/3)"
                ),
            )
            pending_roi_id = ROI_IDS[current["roi_index"]]
            if current["clicks"]:
                pending_color = ROI_COLORS[pending_roi_id]
                pending_v0, pending_v1 = sorted(current["clicks"])
                if len(current["clicks"]) == 2:
                    axis.axhspan(
                        pending_v0,
                        pending_v1,
                        color=pending_color,
                        alpha=0.35,
                        hatch="//",
                        label=f"pending {pending_roi_id}",
                    )
            for value in current["clicks"]:
                axis.axhline(
                    value,
                    color="#ffffff",
                    linestyle=":" if len(current["clicks"]) == 2 else "-",
                    linewidth=1.8,
                )
        status_text.set_text(current["status"])
        figure.canvas.draw_idle()

    def set_status(message: str) -> None:
        current["status"] = message
        redraw()

    def accept_current() -> None:
        roi_id = ROI_IDS[current["roi_index"]]
        if len(current["clicks"]) == 0:
            candidate = tuple(auto_ranges[roi_id])
            method = "enter_accept_auto"
        elif len(current["clicks"]) == 2:
            candidate = tuple(sorted(current["clicks"]))
            method = "two_clicks_then_enter"
        else:
            set_status("请先点击第二条边界，或按 Esc 清除当前选择。")
            return
        try:
            validate_range(
                roi_id,
                candidate,
                image.shape[0],
                current["selected"],
                auto_ranges,
            )
        except ValueError as error:
            current["clicks"] = []
            set_status(f"未保存：{error}；请重新选择。")
            return
        current["selected"][roi_id] = [int(candidate[0]), int(candidate[1])]
        current["clicks"] = []
        current["status"] = f"{roi_id} 已确认（{method}）。"
        if current["roi_index"] == len(ROI_IDS) - 1:
            current["completed"] = True
            plt.close(figure)
            return
        current["roi_index"] += 1
        redraw()

    def on_press(event: Any) -> None:
        if event.inaxes not in axes or event.button != 1 or event.ydata is None:
            return
        current["press_v"] = int(round(float(event.ydata)))

    def on_release(event: Any) -> None:
        press_v = current.get("press_v")
        current["press_v"] = None
        if (
            press_v is None
            or event.inaxes not in axes
            or event.button != 1
            or event.ydata is None
        ):
            return
        value = int(round(float(event.ydata)))
        if value != press_v:
            current["clicks"] = [press_v, value]
            set_status(
                f"已形成临时 {ROI_IDS[current['roi_index']]} 选区 "
                f"{sorted(current['clicks'])}；按 Enter 提交，Esc 清除。"
            )
            return
        current["clicks"].append(value)
        if len(current["clicks"]) > 2:
            current["clicks"] = [value]
        if len(current["clicks"]) == 2:
            set_status(
                f"已形成临时 {ROI_IDS[current['roi_index']]} 选区 "
                f"{sorted(current['clicks'])}；按 Enter 提交，Esc 清除。"
            )
        else:
            set_status(
                f"当前 {ROI_IDS[current['roi_index']]} 已点击 1/2 条边界；"
                "再点击一条 v 边界，或按 Esc 清除。"
            )

    def on_key(event: Any) -> None:
        if event.key in {"enter", "return"}:
            accept_current()
        elif event.key in {"escape", "esc"}:
            current["clicks"] = []
            set_status("当前点击已清除；直接 Enter 可接受 auto candidate。")
        elif event.key in {"q", "Q"}:
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_press)
    figure.canvas.mpl_connect("button_release_event", on_release)
    figure.canvas.mpl_connect("key_press_event", on_key)
    current["status"] = (
        "本条件需确认 3 个 ROI；当前第 1/3 段。"
        "可左键拖拽形成选区，也可分别点击上下 v 边界；形成后按 Enter；"
        "直接 Enter 可接受当前 auto，Esc 清除，Q 退出。"
    )
    redraw()
    plt.show()
    if not current["completed"]:
        raise AnnotationAborted(
            f"annotation aborted before all three ROI ranges for pose {entry['pose_id']}"
        )
    manual = make_manual_entry(entry, current["selected"], center_v)
    render_overlay(
        output_dir
        / "manual_overlays"
        / f"{entry['dataset']}_pose{entry['pose_id']}_position{entry['position_rank']}.png",
        image,
        center_v,
        center_u,
        entry,
        current["selected"],
        image_offset_x,
        f"manual ROI {ordinal}/{total} | pose {entry['pose_id']}",
        "manual_confirmed=true",
        contrast_stretch=contrast_stretch,
    )
    return manual


def build_draft_payload(
    auto_info: dict[str, Any],
    entries: list[dict[str, Any]],
    center_source: str,
    pointwise_cache: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_date": "2026-08-19",
        "roi_stage": "manual_confirmation",
        "manual_confirmed": False,
        "manual_confirmed_count": sum(
            bool(entry.get("manual_confirmed")) for entry in entries
        ),
        "manual_confirmed_expected": len(DATASETS) * len(POSE_IDS),
        "source_auto_registry": auto_info,
        "centerline_source": {
            "method": center_source,
            "pointwise_cache": str(pointwise_cache.resolve()),
            "pointwise_cache_sha256": (
                sha256(pointwise_cache) if pointwise_cache.is_file() else None
            ),
        },
        "geometry_only": True,
        "entries": entries,
        "updated_at": now_utc(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--auto-registry", type=Path, default=DEFAULT_AUTO_REGISTRY)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help="geometry-only candidate CSV; defaults to roi_candidates.csv beside --auto-registry",
    )
    parser.add_argument("--pointwise-cache", type=Path, default=DEFAULT_POINTWISE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help=(
            "dataset directory to annotate; repeat for multiple datasets. "
            "Omit to retain the original obs_1/2/6/10/20/30mm scope."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只准备 median evidence 与 auto preview，不启动人工窗口。",
    )
    parser.add_argument(
        "--contrast-stretch",
        action="store_true",
        help="仅用于显示的 1%%~99.8%% 对比度拉伸；默认使用原始灰度范围。",
    )
    parser.add_argument(
        "--reselect",
        action="append",
        default=[],
        metavar="DATASET:POSE",
        help=(
            "只重新打开指定 ROI，例如 --reselect obs_2mm:001；"
            "其他已确认条目从 draft 复用。可重复指定多个条目。"
        ),
    )
    return parser.parse_args()


def parse_reselect_keys(values: list[str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for value in values:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"invalid --reselect value {value!r}; expected DATASET:POSE"
            )
        dataset, pose_id = parts
        if dataset not in DATASETS or pose_id not in POSE_IDS:
            raise ValueError(
                f"invalid --reselect key {value!r}; expected one of "
                f"{', '.join(DATASETS)}:001..005"
            )
        keys.add((dataset, pose_id))
    return keys


def main() -> int:
    global DATASETS, DATASET_ORDER
    args = parse_args()
    # Select the backend before any function can import pyplot.  The
    # prepare-only path must remain headless; the annotation path requires a
    # real interactive Qt backend for click/Enter editing.
    backend = "Agg" if args.prepare_only else "qtagg"
    try:
        matplotlib.use(backend, force=True)
    except (ImportError, ModuleNotFoundError) as error:
        if args.prepare_only:
            raise
        raise RuntimeError(
            "无法启动 ROI 编辑窗口：需要可用的 Qt Matplotlib backend（qtagg）。"
        ) from error
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()
    auto_path = args.auto_registry.resolve()
    if args.datasets:
        selected = tuple(dict.fromkeys(str(value) for value in args.datasets))
        if len(selected) != len(args.datasets):
            raise ValueError("--dataset values must be unique")
        DATASETS = selected
        DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}
    pointwise_path = args.pointwise_cache.resolve()
    reselect_keys = parse_reselect_keys(args.reselect)
    if args.prepare_only and reselect_keys:
        raise ValueError("--reselect cannot be combined with --prepare-only")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manual_overlays").mkdir(exist_ok=True)
    (output_dir / "manual_median_images").mkdir(exist_ok=True)

    auto_info, auto_entries = read_auto_registry(auto_path)
    candidate_path = (
        args.candidate_csv.resolve()
        if args.candidate_csv is not None
        else auto_path.parent / "roi_candidates.csv"
    )
    candidate_map = read_auto_candidates(candidate_path)
    image_paths = locate_images(data_root)
    frame_offsets = read_frame_offsets(data_root)
    center_cache = read_center_cache(pointwise_path)
    centers, center_source = extract_centers_if_needed(
        data_root, config_path, image_paths, center_cache, frame_offsets
    )
    app = load_app_config(config_path)
    image_offset_x = int(app.camera.offset_x) if app.camera is not None else 0

    # All image/centerline evidence bundles are constructed before any
    # human interaction, so the median shown in each window is fixed.
    bundles: list[dict[str, Any]] = []
    for auto_entry in sorted(
        auto_entries,
        key=lambda item: (DATASET_ORDER[item["dataset"]], item["position_rank"]),
    ):
        dataset = auto_entry["dataset"]
        pose_id = auto_entry["pose_id"]
        repeat_paths = [
            image_paths[(dataset, pose_id, repeat)] for repeat in range(1, 6)
        ]
        image = median_image(repeat_paths)
        repeat_centers = [
            centers[(dataset, pose_id, repeat)] for repeat in range(1, 6)
        ]
        center_v, center_u = binned_centerline(repeat_centers, image.shape[0])
        auto_entry = dict(auto_entry)
        auto_entry["auto_candidates"] = candidate_map.get((dataset, pose_id), [])
        bundles.append(
            {
                "auto_entry": auto_entry,
                "image": image,
                "center_v": center_v,
                "center_u": center_u,
                "repeat_paths": repeat_paths,
            }
        )
        median_path = (
            output_dir
            / "manual_median_images"
            / f"{dataset}_pose{pose_id}.png"
        )
        render_overlay(
            median_path,
            image,
            center_v,
            center_u,
            auto_entry,
            None,
            image_offset_x,
            f"auto ROI evidence {len(bundles)}/{len(auto_entries)} | pose {pose_id}",
            "geometry preview; Enter accepts the shown bands",
            contrast_stretch=args.contrast_stretch,
        )

    draft_path = output_dir / "roi_registry_manual_draft.json"
    draft_entries: list[dict[str, Any]] = []
    existing_entries: dict[tuple[str, str], dict[str, Any]] = {}
    if draft_path.is_file():
        try:
            draft_document = json.loads(draft_path.read_text(encoding="utf-8"))
            for existing in draft_document.get("entries", []):
                if (
                    isinstance(existing, dict)
                    and existing.get("manual_confirmed") is True
                ):
                    existing_entries[
                        (existing["dataset"], existing["pose_id"])
                    ] = existing
        except (OSError, TypeError, ValueError, KeyError):
            existing_entries = {}
    if reselect_keys:
        existing_entries = {
            key: entry
            for key, entry in existing_entries.items()
            if key not in reselect_keys
        }
    if args.prepare_only:
        write_json(
            draft_path,
            build_draft_payload(auto_info, draft_entries, center_source, pointwise_path),
        )
        print(
            json.dumps(
                {
                    "prepared": len(bundles),
                    "draft": str(draft_path),
                    "centerline_source": center_source,
                    "message": "Run without --prepare-only to open the manual editor.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    # q/closing a window leaves the draft intact; rerunning starts from the
    # first incomplete entry and never silently marks a group confirmed.
    import matplotlib.pyplot as plt

    try:
        for ordinal, bundle in enumerate(bundles, start=1):
            auto_entry = bundle["auto_entry"]
            existing = existing_entries.get(
                (auto_entry["dataset"], auto_entry["pose_id"])
            )
            if existing is not None:
                draft_entries.append(existing)
                continue
            manual = annotate_entry(
                ordinal,
                len(bundles),
                auto_entry,
                bundle["image"],
                bundle["center_v"],
                bundle["center_u"],
                output_dir,
                image_offset_x,
                args.contrast_stretch,
            )
            draft_entries.append(manual)
            write_json(
                draft_path,
                build_draft_payload(
                    auto_info, draft_entries, center_source, pointwise_path
                ),
            )
    except AnnotationAborted as error:
        print(str(error), file=sys.stderr)
        return 2
    finally:
        plt.close("all")

    expected_count = len(DATASETS) * len(POSE_IDS)
    if len(draft_entries) != expected_count or not all(
        entry.get("manual_confirmed") is True for entry in draft_entries
    ):
        raise RuntimeError(
            f"manual registry is not {expected_count}/{expected_count} confirmed; "
            "final file not written"
        )
    reorder_positions(draft_entries)
    draft_entries.sort(
        key=lambda item: (DATASET_ORDER[item["dataset"]], item["position_rank"])
    )
    final_payload = build_draft_payload(
        auto_info, draft_entries, center_source, pointwise_path
    )
    final_payload["manual_confirmed"] = True
    final_payload["manual_confirmed_count"] = expected_count
    final_payload["frozen_at"] = now_utc()
    final_payload["freeze_policy"] = {
        "all_entries_manual_confirmed": True,
        "auto_roi_retained_for_comparison": True,
        "c0_c1_values_used": False,
        "truth_values_used": False,
    }
    final_path = output_dir / "roi_registry_manual.json"
    write_json(final_path, final_payload)
    differences = diff_rows(draft_entries)
    write_diff(output_dir / "roi_auto_vs_manual.csv", differences)
    write_json(
        output_dir / "roi_auto_vs_manual.json",
        {
            "auto_registry": auto_info,
            "manual_registry": str(final_path.resolve()),
            "manual_confirmed_count": expected_count,
            "entries": differences,
        },
    )
    print(
        json.dumps(
            {
                "manual_registry": str(final_path),
                "manual_confirmed": expected_count,
                "auto_vs_manual_csv": str(output_dir / "roi_auto_vs_manual.csv"),
                "centerline_source": center_source,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
