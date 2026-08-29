#!/usr/bin/env python3
"""Freeze geometry-only manual ROIs for the Haikang 2026-08-29 recordings.

This is a thin Haikang adapter for the interaction and registry semantics used
by ``annotate_daheng_gauge_rois.py``.  The operator selects, in order,
``baseline_before``, ``height`` and ``baseline_after``.  The GUI deliberately
shows only a representative source image and the 20-frame median Steger
centerline.  It never reads or displays height truth, reconstructed heights,
errors, Session Ground results, or automatic ROI candidates.

All saved intervals are inclusive full-sensor ``u`` ranges.  Captured-image
offsets are applied before the centerline is fused or displayed.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
TOOLS_ROOT = REPO_ROOT / "tools"
for import_root in (MEASUREMENT_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from app_config import load_app_config  # noqa: E402
from laser.backends import create_extraction_params  # noqa: E402
from laser.laser_extractor import extract_laser_center  # noqa: E402
from utils.image_io import load_grayscale_image  # noqa: E402

import annotate_daheng_gauge_rois as daheng_roi  # noqa: E402
import thermal_a2a_roi_v2 as roi_v2_wrapper  # noqa: E402


DEFAULT_DATA_ROOT = (
    MEASUREMENT_ROOT
    / "output_haikang_0828"
    / "online_recordings"
    / "0829"
)
DEFAULT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_haikang_0828.yaml"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "c0_height_audit" / "manual_roi"

HEIGHT_IDS = ("h02", "h06", "h10", "h20", "h30")
POSITION_IDS = tuple(f"p{index:02d}" for index in range(1, 11))
ROI_IDS = daheng_roi.ROI_IDS
ROI_COLORS = daheng_roi.ROI_COLORS
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
CONDITION_RE = re.compile(r"^(h\d+)_(p\d+)$")
TERMINAL_STATUSES = {"selected", "unusable"}
CSV_FIELDS = [
    "height_id",
    "position_id",
    "condition",
    "coordinate_system",
    "baseline_before_u0",
    "baseline_before_u1",
    "height_u0",
    "height_u1",
    "baseline_after_u0",
    "baseline_after_u1",
    "source_frame",
    "selection_status",
    "selection_mode",
    "notes",
    "operator",
    "selected_at_utc",
]


class ManualRoiError(RuntimeError):
    """Raised when source or selection contracts are not satisfied."""


@dataclass(frozen=True, slots=True)
class Condition:
    height_id: str
    position_id: str
    path: Path

    @property
    def condition(self) -> str:
        return f"{self.height_id}_{self.position_id}"


@dataclass(frozen=True, slots=True)
class FrameSpec:
    path: Path
    filename: str
    offset_x: int
    offset_y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Evidence:
    condition: Condition
    representative: FrameSpec
    source_frames: tuple[FrameSpec, ...]
    centerline_uv_full: np.ndarray
    evidence_npz: Path
    selection_view: Path
    reused: bool


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
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


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field)
                    for field in CSV_FIELDS
                }
            )


def read_frame_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ManualRoiError(f"missing frames.csv: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != FRAME_FIELDS:
            raise ManualRoiError(
                f"frames.csv schema mismatch in {path}: {header!r}"
            )
        rows = [
            {name: str(value).strip() for name, value in zip(header, values)}
            for values in reader
            if values and any(str(value).strip() for value in values)
        ]
    if len(rows) != 20:
        raise ManualRoiError(f"{path} must contain exactly 20 frames, got {len(rows)}")
    return rows


def parse_frame_spec(condition: Condition, row: dict[str, str]) -> FrameSpec:
    path = (condition.path / row["filename"]).resolve()
    try:
        path.relative_to(condition.path.resolve())
    except ValueError as error:
        raise ManualRoiError(f"frame path escapes condition directory: {path}") from error
    if not path.is_file():
        raise ManualRoiError(f"missing source frame: {path}")
    try:
        values = {
            name: int(float(row[name]))
            for name in ("offset_x", "offset_y", "width", "height")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ManualRoiError(f"invalid frame geometry in {condition.condition}") from error
    if values["offset_x"] < 0 or values["offset_y"] < 0:
        raise ManualRoiError(f"negative frame offset in {condition.condition}")
    if values["width"] <= 0 or values["height"] <= 0:
        raise ManualRoiError(f"non-positive frame dimensions in {condition.condition}")
    return FrameSpec(
        path=path,
        filename=path.name,
        offset_x=values["offset_x"],
        offset_y=values["offset_y"],
        width=values["width"],
        height=values["height"],
    )


def discover_conditions(root: Path) -> list[Condition]:
    conditions: list[Condition] = []
    for height_id in HEIGHT_IDS:
        for position_id in POSITION_IDS:
            path = root / height_id / f"{height_id}_{position_id}"
            if not path.is_dir():
                raise ManualRoiError(f"missing target condition directory: {path}")
            conditions.append(Condition(height_id, position_id, path.resolve()))
    return conditions


def full_sensor_centerline(centers_by_frame: list[np.ndarray]) -> np.ndarray:
    """Reuse the existing median-centerline implementation via an axis adapter."""
    swapped = []
    for centers in centers_by_frame:
        values = np.asarray(centers, dtype=np.float64)
        if values.size == 0:
            swapped.append(np.empty((0, 2), dtype=np.float64))
        elif values.ndim == 2 and values.shape[1] == 2:
            swapped.append(np.ascontiguousarray(values[:, [1, 0]]))
        else:
            raise ManualRoiError(f"unexpected center array shape: {values.shape}")
    # Existing ROI-V2 fuses row-scan (u(v), v).  Swapping before and after
    # produces the required Haikang column-scan (u, median v(u)).
    median_swapped = roi_v2_wrapper.median_centerline(swapped)
    return np.ascontiguousarray(median_swapped[:, [1, 0]], dtype=np.float64)


def validate_ranges(
    ranges: dict[str, list[int] | None], sensor_u_min: int, sensor_u_max: int
) -> None:
    resolved: dict[str, tuple[int, int]] = {}
    for roi_id in ROI_IDS:
        value = ranges.get(roi_id)
        if not isinstance(value, list) or len(value) != 2:
            raise ManualRoiError(f"{roi_id} has not been selected")
        u0, u1 = int(value[0]), int(value[1])
        if u0 > u1:
            raise ManualRoiError(f"{roi_id} range is reversed: {value}")
        if u0 < sensor_u_min or u1 > sensor_u_max:
            raise ManualRoiError(
                f"{roi_id} must stay inside full-sensor u "
                f"[{sensor_u_min}, {sensor_u_max}]"
            )
        resolved[roi_id] = (u0, u1)
    before = resolved["baseline_before"]
    height = resolved["height"]
    after = resolved["baseline_after"]
    if before[1] >= height[0]:
        raise ManualRoiError("baseline_before must end before height starts")
    if height[1] >= after[0]:
        raise ManualRoiError("height must end before baseline_after starts")


def blank_entry(condition: Condition, evidence: Evidence, operator: str) -> dict[str, Any]:
    representative = evidence.representative
    return {
        "height_id": condition.height_id,
        "position_id": condition.position_id,
        "condition": condition.condition,
        "coordinate_system": "full_sensor",
        "baseline_before_u0": None,
        "baseline_before_u1": None,
        "height_u0": None,
        "height_u1": None,
        "baseline_after_u0": None,
        "baseline_after_u1": None,
        "source_frame": str(representative.path),
        "representative_source": {
            "mode": "original_middle_frame_plus_20_frame_median_centerline",
            "original_frame": representative.filename,
            "frame_count": len(evidence.source_frames),
            "centerline_method": "existing_steger_then_thermal_a2a_roi_v2.median_centerline_axis_swapped",
            "offset_x": representative.offset_x,
            "offset_y": representative.offset_y,
            "width": representative.width,
            "height": representative.height,
            "selection_view": str(evidence.selection_view),
            "evidence_npz": str(evidence.evidence_npz),
        },
        "selection_status": "unreviewed",
        "selection_mode": "manual",
        "notes": "",
        "operator": operator,
        "selected_at_utc": None,
        "manual_provenance": {
            "geometry_only": True,
            "truth_values_used": False,
            "height_results_used": False,
            "automatic_roi_used": False,
            "board_polygon_used": False,
        },
    }


def entry_ranges(entry: dict[str, Any]) -> dict[str, list[int] | None]:
    output: dict[str, list[int] | None] = {}
    for roi_id in ROI_IDS:
        u0 = entry.get(f"{roi_id}_u0")
        u1 = entry.get(f"{roi_id}_u1")
        output[roi_id] = None if u0 is None or u1 is None else [int(u0), int(u1)]
    return output


def apply_ranges(entry: dict[str, Any], ranges: dict[str, list[int] | None]) -> None:
    for roi_id in ROI_IDS:
        value = ranges.get(roi_id)
        entry[f"{roi_id}_u0"] = None if value is None else int(value[0])
        entry[f"{roi_id}_u1"] = None if value is None else int(value[1])


def evidence_paths(output: Path, condition: Condition) -> tuple[Path, Path]:
    evidence_dir = output / "representative_views"
    return (
        evidence_dir / f"{condition.condition}_evidence.npz",
        evidence_dir / f"{condition.condition}_selection_view.png",
    )


def evidence_fingerprint(
    condition: Condition,
    config_path: Path,
    frames_csv: Path,
    source_frames: tuple[FrameSpec, ...],
) -> dict[str, Any]:
    implementation_paths = {
        "laser_extractor": MEASUREMENT_ROOT / "laser" / "laser_extractor.py",
        "laser_backends": MEASUREMENT_ROOT / "laser" / "backends.py",
        "realtime_steger": MEASUREMENT_ROOT / "laser" / "realtime_steger.py",
        "realtime_steger_profile": MEASUREMENT_ROOT / "configs" / "realtime_steger.yaml",
        "median_centerline": TOOLS_ROOT / "thermal_a2a_roi_v2.py",
    }
    return {
        "condition": condition.condition,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "frames_csv": str(frames_csv.resolve()),
        "frames_csv_sha256": sha256_file(frames_csv),
        "frame_count": 20,
        "representative_index_one_based": 10,
        "centerline_protocol": "20_frame_steger_full_sensor_then_existing_median_centerline_axis_swapped_v1",
        "source_frames": [
            {
                "filename": frame.filename,
                "size_bytes": frame.path.stat().st_size,
                "sha256": sha256_file(frame.path),
            }
            for frame in source_frames
        ],
        "implementation_sha256": {
            name: sha256_file(path) for name, path in implementation_paths.items()
        },
    }


def compatible_fingerprint(cached: Any, current: dict[str, Any]) -> bool:
    """Accept the pre-content-hash cache once, then persist the stronger contract."""
    if not isinstance(cached, dict):
        return False
    required = (
        "condition",
        "config_path",
        "config_sha256",
        "frames_csv",
        "frames_csv_sha256",
        "frame_count",
        "representative_index_one_based",
        "centerline_protocol",
    )
    if any(cached.get(key) != current.get(key) for key in required):
        return False
    for strengthened_key in ("source_frames", "implementation_sha256"):
        if strengthened_key in cached and cached.get(strengthened_key) != current.get(
            strengthened_key
        ):
            return False
    return True


def render_view(
    path: Path,
    representative: FrameSpec,
    centerline: np.ndarray,
    condition_id: str,
    ranges: dict[str, list[int] | None] | None = None,
    status: str = "selection evidence",
) -> None:
    import matplotlib.pyplot as plt

    image = load_grayscale_image(representative.path)
    rendered, display_vmax = daheng_roi.prepare_display_image(image, False)
    x0 = representative.offset_x
    x1 = representative.offset_x + representative.width
    y0 = representative.offset_y
    y1 = representative.offset_y + representative.height
    center_u = centerline[:, 0]
    center_v = centerline[:, 1]
    finite = np.isfinite(center_u) & np.isfinite(center_v)
    if not np.any(finite):
        raise ManualRoiError(f"{condition_id}: median centerline is empty")
    detail_margin = 45
    detail_y0 = max(y0, int(math.floor(float(np.nanmin(center_v[finite])))) - detail_margin)
    detail_y1 = min(y1, int(math.ceil(float(np.nanmax(center_v[finite])))) + detail_margin)
    if detail_y1 <= detail_y0:
        detail_y0, detail_y1 = y0, y1

    figure, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    for axis, bounds, title in (
        (axes[0], (x0, x1, y0, y1), "original representative frame (full captured extent)"),
        (axes[1], (x0, x1, detail_y0, detail_y1), "20-frame median Steger centerline detail"),
    ):
        bx0, bx1, by0, by1 = bounds
        local_y0 = by0 - representative.offset_y
        local_y1 = by1 - representative.offset_y
        axis.imshow(
            rendered[local_y0:local_y1, :],
            cmap="gray",
            vmin=0.0,
            vmax=display_vmax,
            extent=(bx0, bx1, by1, by0),
            aspect="auto",
            interpolation="nearest",
        )
        axis.scatter(
            center_u[finite],
            center_v[finite],
            s=3.0,
            color="#e040fb",
            alpha=0.95,
            linewidths=0.0,
            label="20-frame median Steger centerline",
        )
        if ranges:
            for roi_id in ROI_IDS:
                value = ranges.get(roi_id)
                if value is None:
                    continue
                axis.axvspan(
                    value[0], value[1], color=ROI_COLORS[roi_id], alpha=0.25,
                    label=f"manual {roi_id} [{value[0]}, {value[1]}]",
                )
        axis.set_xlim(bx0, bx1)
        axis.set_ylim(by1, by0)
        axis.set_xlabel("full-sensor u [px]")
        axis.set_ylabel("full-sensor v [px]")
        axis.set_title(title)
        axis.grid(alpha=0.15)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    figure.suptitle(f"{condition_id} | geometry-only manual ROI | {status}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def prepare_evidence(
    conditions: list[Condition],
    config_path: Path,
    output: Path,
    refresh: bool,
    refresh_conditions: set[str] | None = None,
) -> tuple[list[Evidence], dict[str, Any]]:
    refresh_conditions = refresh_conditions or set()
    app = load_app_config(config_path)
    extraction_params = create_extraction_params(
        app.extraction_method,
        app.extraction_options_by_method.get(app.extraction_method, {}),
    )
    manifest_path = output / "representative_view_manifest.json"
    old_manifest: dict[str, Any] = {}
    if manifest_path.is_file() and not refresh:
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            old_manifest = {}
    old_by_condition = {
        str(item.get("condition")): item
        for item in old_manifest.get("conditions", [])
        if isinstance(item, dict)
    }

    evidence_items: list[Evidence] = []
    manifest_rows: list[dict[str, Any]] = []
    fresh_count = 0
    reused_count = 0
    for ordinal, condition in enumerate(conditions, start=1):
        frames_csv = condition.path / "frames.csv"
        rows = read_frame_rows(frames_csv)
        specs = tuple(parse_frame_spec(condition, row) for row in rows)
        geometry = {
            (spec.offset_x, spec.offset_y, spec.width, spec.height) for spec in specs
        }
        if len(geometry) != 1:
            raise ManualRoiError(
                f"{condition.condition}: inconsistent capture geometry: {geometry}"
            )
        representative = specs[9]
        fingerprint = evidence_fingerprint(condition, config_path, frames_csv, specs)
        npz_path, view_path = evidence_paths(output, condition)
        cached = old_by_condition.get(condition.condition)
        can_reuse = (
            not refresh
            and condition.condition not in refresh_conditions
            and cached is not None
            and compatible_fingerprint(cached.get("fingerprint"), fingerprint)
            and npz_path.is_file()
            and view_path.is_file()
        )
        if can_reuse:
            try:
                with np.load(npz_path, allow_pickle=False) as bundle:
                    centerline = np.asarray(bundle["centerline_uv_full"], dtype=np.float64)
                if centerline.ndim != 2 or centerline.shape[1] != 2:
                    raise ValueError("bad cached centerline shape")
                reused = True
                reused_count += 1
            except (OSError, ValueError, KeyError):
                can_reuse = False
        if not can_reuse:
            centers_by_frame: list[np.ndarray] = []
            for spec in specs:
                image = load_grayscale_image(spec.path)
                if image.shape != (spec.height, spec.width):
                    raise ManualRoiError(
                        f"{spec.path}: image shape {image.shape} does not match frames.csv"
                    )
                local = np.asarray(
                    extract_laser_center(
                        image,
                        extraction_params,
                        image_offset=(spec.offset_x, spec.offset_y),
                    ),
                    dtype=np.float64,
                ).reshape(-1, 2)
                full = local + np.asarray([spec.offset_x, spec.offset_y], dtype=np.float64)
                centers_by_frame.append(np.ascontiguousarray(full))
            centerline = full_sensor_centerline(centers_by_frame)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, centerline_uv_full=centerline)
            render_view(
                view_path,
                representative,
                centerline,
                condition.condition,
                status="prepared; no automatic ROI shown",
            )
            reused = False
            fresh_count += 1
        evidence = Evidence(
            condition=condition,
            representative=representative,
            source_frames=specs,
            centerline_uv_full=centerline,
            evidence_npz=npz_path,
            selection_view=view_path,
            reused=reused,
        )
        evidence_items.append(evidence)
        manifest_rows.append(
            {
                "condition": condition.condition,
                "ordinal": ordinal,
                "fingerprint": fingerprint,
                "source_frame": str(representative.path),
                "source_frame_offset": [representative.offset_x, representative.offset_y],
                "source_frame_shape": [representative.height, representative.width],
                "centerline_point_count": int(len(centerline)),
                "evidence_npz": str(npz_path.resolve()),
                "selection_view": str(view_path.resolve()),
                "calculation": "reused" if reused else "fresh_this_run",
            }
        )
        print(
            f"[{ordinal:02d}/{len(conditions)}] {condition.condition}: "
            f"{'reused' if reused else 'prepared'} ({len(centerline)} median points)",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "task": "H0-1M-A",
        "generated_at_utc": now_utc(),
        "geometry_only": True,
        "coordinate_system": "full_sensor",
        "scan_axis": "column",
        "condition_count": len(evidence_items),
        "frames_per_condition": 20,
        "artifact_reuse": {
            "reused_condition_count": reused_count,
            "fresh_condition_count": fresh_count,
            "prior_manifest_available": bool(old_manifest),
            "compatible_representative_artifact_found_before_run": reused_count > 0,
            "forced_refresh_conditions": sorted(refresh_conditions),
        },
        "reused_implementations": {
            "manual_roi_reference": {
                "path": str((TOOLS_ROOT / "annotate_daheng_gauge_rois.py").resolve()),
                "sha256": sha256_file(TOOLS_ROOT / "annotate_daheng_gauge_rois.py"),
                "semantics": "baseline_before, height, baseline_after; ordered non-overlap; geometry-only",
            },
            "center_extraction": {
                "config": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "method": app.extraction_method,
                "entry": "laser.laser_extractor.extract_laser_center",
            },
            "median_centerline": {
                "path": str((TOOLS_ROOT / "thermal_a2a_roi_v2.py").resolve()),
                "sha256": sha256_file(TOOLS_ROOT / "thermal_a2a_roi_v2.py"),
                "entry": "median_centerline",
                "adapter": "swap full-sensor (u,v) to (u'=v,v'=u), fuse, then swap back",
            },
        },
        "forbidden_inputs": {
            "height_shadow_read": False,
            "h_raw_read": False,
            "height_truth_used": False,
            "height_or_error_computed": False,
            "session_ground_read": False,
            "board_polygon_used": False,
            "automatic_roi_used": False,
            "h1_hb2_c1_called": False,
        },
        "conditions": manifest_rows,
    }
    manifest["path"] = str(manifest_path.resolve())
    write_json(manifest_path, manifest)
    return evidence_items, manifest


def registry_payload(
    entries: list[dict[str, Any]], manifest: dict[str, Any], operator: str, frozen: bool
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "H0-1M-A",
        "purpose": "frozen geometry-only manual ROI diagnostic baseline",
        "coordinate_system": "full_sensor",
        "scan_axis": "column",
        "selection_mode": "manual",
        "operator": operator,
        "manual_confirmed": frozen,
        "frozen": frozen,
        "updated_at_utc": now_utc(),
        "frozen_at_utc": now_utc() if frozen else None,
        "condition_count": len(entries),
        "terminal_condition_count": sum(
            entry.get("selection_status") in TERMINAL_STATUSES for entry in entries
        ),
        "representative_view_manifest": {
            "path": manifest.get("path"),
            "generated_at_utc": manifest.get("generated_at_utc"),
        },
        "freeze_policy": {
            "all_conditions_terminal": frozen,
            "selected_or_explicitly_unusable_only": True,
            "one_roi_per_condition_shared_by_20_frames": True,
            "truth_values_used": False,
            "height_results_used": False,
            "automatic_roi_used": False,
        },
        "entries": entries,
    }


def report_text(
    entries: list[dict[str, Any]], manifest: dict[str, Any], frozen: bool
) -> str:
    counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("selection_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    status_lines = "\n".join(
        f"- `{status}`: {count}" for status, count in sorted(counts.items())
    )
    reuse = manifest.get("artifact_reuse") or {}
    state = "FROZEN" if frozen else "AWAITING_MANUAL_SELECTION"
    return f"""# Haikang 0829 manual ROI selection report

## State

`{state}`

{status_lines}

## Reuse audit

- Reused the established Daheng three-band manual ROI semantics and Matplotlib interaction pattern from `tools/annotate_daheng_gauge_rois.py`.
- Reused production `extract_laser_center` with `measure_tool_haikang_0828.yaml` and the existing `thermal_a2a_roi_v2.median_centerline` implementation through the documented Haikang axis swap.
- Before this run, compatible cached representative/centerline evidence was {'found' if reuse.get('compatible_representative_artifact_found_before_run') else 'not found'}. This evidence run reused {reuse.get('reused_condition_count', 0)} cached conditions and newly calculated {reuse.get('fresh_condition_count', 0)} conditions.
- Existing C0/Session Ground/ROI-V2 measurement artifacts were audited for provenance only. Their height values, automatic ROI candidates, board polygon and Session Ground were not read into the selection tool.

## Frozen protocol

- Scope: `h02/h06/h10/h20/h30 × p01...p10` (50 conditions), 20 frames per condition.
- View: original frame 10 plus the 20-frame median Steger centerline; axes are full-sensor `(u,v)`.
- Selection order: `baseline_before | height | baseline_after`.
- Registry intervals are inclusive full-sensor `u` ranges. The three intervals must be ordered and non-overlapping; widths may differ.
- Each condition is selected once and the frozen ranges are shared by its 20 frames.
- The GUI and output contain no `h_raw`, truth height, residual, MAE/RMSE, H1, H-B2 or C1 result.

## Interaction

- Drag horizontally in either image panel, then press `Enter` or click `Confirm` to commit the current band.
- After all three bands are present, press `Enter`/`Confirm` once more to confirm the condition.
- `1/2/3`: choose a band for re-selection; `Esc`/`Reselect`: clear the current pending span; `U`/`Undo`: restore the preceding edit.
- `S`/`Skip`: mark the condition unusable; `P`/`Left`: previous; `N`/`Right`: next; `Q`: save draft and quit.

## Outputs

- `manual_roi_registry_draft.json`: resumable working state.
- `manual_roi_registry.json` and `manual_roi_registry.csv`: written only when all 50 conditions are selected or explicitly unusable.
- `overlays/<condition>_manual_roi.png`: frozen geometry overlay.
- `representative_views/<condition>_selection_view.png`: immutable selection evidence.
"""


def load_entries(
    output: Path, evidence: list[Evidence], operator: str, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = [
        output / "manual_roi_registry_draft.json",
        output / "manual_roi_registry.json",
    ]
    by_condition: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for entry in payload.get("entries", []):
                if isinstance(entry, dict):
                    by_condition[str(entry.get("condition"))] = entry
            break
        except (OSError, ValueError, TypeError):
            by_condition = {}
    entries: list[dict[str, Any]] = []
    for item in evidence:
        previous = by_condition.get(item.condition.condition)
        entry = blank_entry(item.condition, item, operator)
        if previous is not None:
            # Source/evidence provenance always comes from the current verified
            # bundle.  Copying an older draft's representative_source would
            # preserve stale offsets after a source-frame correction.
            preserved_fields = [
                field for field in CSV_FIELDS if field != "source_frame"
            ] + ["manual_provenance"]
            for field in preserved_fields:
                if field in previous:
                    entry[field] = previous[field]
            entry["operator"] = previous.get("operator") or operator
        entries.append(entry)
    write_json(
        output / "manual_roi_registry_draft.json",
        registry_payload(entries, manifest, operator, frozen=False),
    )
    return entries


def finalize_outputs(
    output: Path,
    entries: list[dict[str, Any]],
    evidence: list[Evidence],
    manifest: dict[str, Any],
    operator: str,
) -> bool:
    frozen = all(
        entry.get("selection_status") in TERMINAL_STATUSES for entry in entries
    )
    write_json(
        output / "manual_roi_registry_draft.json",
        registry_payload(entries, manifest, operator, frozen=False),
    )
    for entry, item in zip(entries, evidence):
        if entry.get("selection_status") == "selected":
            ranges = entry_ranges(entry)
            validate_ranges(
                ranges,
                item.representative.offset_x,
                item.representative.offset_x + item.representative.width - 1,
            )
            render_view(
                output / "overlays" / f"{item.condition.condition}_manual_roi.png",
                item.representative,
                item.centerline_uv_full,
                item.condition.condition,
                ranges,
                status="selection_status=selected",
            )
        elif entry.get("selection_status") == "unusable":
            render_view(
                output / "overlays" / f"{item.condition.condition}_manual_roi.png",
                item.representative,
                item.centerline_uv_full,
                item.condition.condition,
                None,
                status="selection_status=unusable",
            )
    if frozen:
        payload = registry_payload(entries, manifest, operator, frozen=True)
        write_json(output / "manual_roi_registry.json", payload)
        write_csv(output / "manual_roi_registry.csv", entries)
    (output / "manual_roi_selection_report.md").write_text(
        report_text(entries, manifest, frozen), encoding="utf-8"
    )
    return frozen


class ManualRoiEditor:
    """Small Matplotlib adapter retaining the established three-band workflow."""

    def __init__(
        self,
        output: Path,
        evidence: list[Evidence],
        entries: list[dict[str, Any]],
        manifest: dict[str, Any],
        operator: str,
        start_index: int,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, SpanSelector

        self.plt = plt
        self.output = output
        self.evidence = evidence
        self.entries = entries
        self.manifest = manifest
        self.operator = operator
        self.index = start_index
        self.roi_index = 0
        self.pending: list[int] | None = None
        self.history: list[tuple[int, dict[str, Any], int]] = []
        self.message = "Select baseline_before by horizontal drag."
        self.figure, self.axes = plt.subplots(2, 1, figsize=(15, 9))
        self.figure.subplots_adjust(bottom=0.14, hspace=0.32)
        labels = ["Confirm", "Reselect", "Undo", "Skip", "Previous", "Next"]
        callbacks = [
            self.confirm,
            self.reselect,
            self.undo,
            self.skip,
            self.previous,
            self.next,
        ]
        for button_index, (label, callback) in enumerate(zip(labels, callbacks)):
            axis = self.figure.add_axes([0.08 + button_index * 0.145, 0.035, 0.12, 0.045])
            button = Button(axis, label)
            button.on_clicked(callback)
            setattr(self, f"button_{button_index}", button)
        props = dict(alpha=0.25, facecolor=ROI_COLORS[ROI_IDS[0]])
        self.selectors = [
            SpanSelector(axis, self.span_selected, "horizontal", useblit=True, props=props)
            for axis in self.axes
        ]
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("close_event", self.on_close)
        self.redraw()

    def current(self) -> tuple[Evidence, dict[str, Any]]:
        return self.evidence[self.index], self.entries[self.index]

    def snapshot(self) -> None:
        self.history.append((self.index, copy.deepcopy(self.entries[self.index]), self.roi_index))
        if len(self.history) > 100:
            self.history.pop(0)

    def save_draft(self) -> None:
        write_json(
            self.output / "manual_roi_registry_draft.json",
            registry_payload(self.entries, self.manifest, self.operator, frozen=False),
        )
        (self.output / "manual_roi_selection_report.md").write_text(
            report_text(self.entries, self.manifest, frozen=False), encoding="utf-8"
        )

    def span_selected(self, first: float, second: float) -> None:
        item, _ = self.current()
        u_min = item.representative.offset_x
        u_max = u_min + item.representative.width - 1
        lo, hi = sorted((int(round(first)), int(round(second))))
        lo, hi = max(u_min, lo), min(u_max, hi)
        if hi < lo:
            self.message = f"Pending span is outside [{u_min}, {u_max}]."
            self.pending = None
        else:
            self.pending = [lo, hi]
            self.message = f"Pending {ROI_IDS[self.roi_index]}=[{lo}, {hi}]; Confirm to commit."
        self.redraw()

    def confirm(self, _event: Any = None) -> None:
        item, entry = self.current()
        ranges = entry_ranges(entry)
        if self.pending is not None:
            self.snapshot()
            ranges[ROI_IDS[self.roi_index]] = list(self.pending)
            apply_ranges(entry, ranges)
            entry["selection_status"] = "in_progress"
            entry["selected_at_utc"] = None
            self.pending = None
            if self.roi_index < len(ROI_IDS) - 1:
                self.roi_index += 1
                self.message = f"Committed; now select {ROI_IDS[self.roi_index]}."
            else:
                self.message = "All three bands exist; Confirm once more to accept condition."
            self.save_draft()
            self.redraw()
            return
        try:
            validate_ranges(
                ranges,
                item.representative.offset_x,
                item.representative.offset_x + item.representative.width - 1,
            )
        except ManualRoiError as error:
            self.message = str(error)
            self.redraw()
            return
        self.snapshot()
        entry["selection_status"] = "selected"
        entry["selection_mode"] = "manual"
        entry["operator"] = self.operator
        entry["selected_at_utc"] = now_utc()
        entry["notes"] = str(entry.get("notes") or "")
        self.save_draft()
        render_view(
            self.output / "overlays" / f"{item.condition.condition}_manual_roi.png",
            item.representative,
            item.centerline_uv_full,
            item.condition.condition,
            ranges,
            status="selection_status=selected",
        )
        self.message = f"Confirmed {item.condition.condition}."
        self.move(1)

    def reselect(self, _event: Any = None) -> None:
        self.pending = None
        self.message = f"Pending span cleared; reselect {ROI_IDS[self.roi_index]}."
        self.redraw()

    def undo(self, _event: Any = None) -> None:
        if not self.history:
            self.message = "Nothing to undo."
            self.redraw()
            return
        index, entry, roi_index = self.history.pop()
        self.index = index
        self.entries[index] = entry
        self.roi_index = roi_index
        self.pending = None
        self.message = "Previous edit restored."
        self.save_draft()
        self.redraw()

    def skip(self, _event: Any = None) -> None:
        item, entry = self.current()
        self.snapshot()
        apply_ranges(entry, {roi_id: None for roi_id in ROI_IDS})
        entry["selection_status"] = "unusable"
        entry["selection_mode"] = "manual"
        entry["operator"] = self.operator
        entry["selected_at_utc"] = now_utc()
        entry["notes"] = "operator_marked_unusable_in_gui"
        self.save_draft()
        render_view(
            self.output / "overlays" / f"{item.condition.condition}_manual_roi.png",
            item.representative,
            item.centerline_uv_full,
            item.condition.condition,
            None,
            status="selection_status=unusable",
        )
        self.message = f"Marked {item.condition.condition} unusable."
        self.move(1)

    def previous(self, _event: Any = None) -> None:
        self.move(-1)

    def next(self, _event: Any = None) -> None:
        self.move(1)

    def move(self, delta: int) -> None:
        self.pending = None
        self.index = (self.index + delta) % len(self.entries)
        ranges = entry_ranges(self.entries[self.index])
        missing = [index for index, roi_id in enumerate(ROI_IDS) if ranges[roi_id] is None]
        self.roi_index = missing[0] if missing else 0
        self.redraw()

    def on_key(self, event: Any) -> None:
        key = str(event.key or "")
        if key in {"enter", "return"}:
            self.confirm()
        elif key in {"escape", "esc", "r"}:
            self.reselect()
        elif key.lower() == "u":
            self.undo()
        elif key.lower() == "s":
            self.skip()
        elif key in {"left", "p"}:
            self.previous()
        elif key in {"right", "n"}:
            self.next()
        elif key in {"1", "2", "3"}:
            self.roi_index = int(key) - 1
            self.pending = None
            self.message = f"Reselect mode: {ROI_IDS[self.roi_index]}."
            self.redraw()
        elif key.lower() == "q":
            self.save_draft()
            self.plt.close(self.figure)

    def on_close(self, _event: Any) -> None:
        self.save_draft()

    def redraw(self) -> None:
        item, entry = self.current()
        image = load_grayscale_image(item.representative.path)
        rendered, display_vmax = daheng_roi.prepare_display_image(image, False)
        rep = item.representative
        x0, x1 = rep.offset_x, rep.offset_x + rep.width
        y0, y1 = rep.offset_y, rep.offset_y + rep.height
        centerline = item.centerline_uv_full
        center_u, center_v = centerline[:, 0], centerline[:, 1]
        finite = np.isfinite(center_u) & np.isfinite(center_v)
        detail_margin = 45
        detail_y0 = max(y0, int(np.floor(np.nanmin(center_v[finite]))) - detail_margin)
        detail_y1 = min(y1, int(np.ceil(np.nanmax(center_v[finite]))) + detail_margin)
        ranges = entry_ranges(entry)
        for axis, bounds, title in (
            (self.axes[0], (x0, x1, y0, y1), "original representative frame"),
            (self.axes[1], (x0, x1, detail_y0, detail_y1), "20-frame median centerline detail"),
        ):
            axis.clear()
            bx0, bx1, by0, by1 = bounds
            axis.imshow(
                rendered[by0 - rep.offset_y : by1 - rep.offset_y, :],
                cmap="gray", vmin=0.0, vmax=display_vmax,
                extent=(bx0, bx1, by1, by0), aspect="auto", interpolation="nearest",
            )
            axis.scatter(center_u[finite], center_v[finite], s=3.0, color="#e040fb", linewidths=0.0)
            for roi_id in ROI_IDS:
                value = ranges[roi_id]
                if value is not None:
                    axis.axvspan(value[0], value[1], color=ROI_COLORS[roi_id], alpha=0.25, label=roi_id)
            if self.pending is not None:
                axis.axvspan(
                    self.pending[0], self.pending[1], color=ROI_COLORS[ROI_IDS[self.roi_index]],
                    alpha=0.42, hatch="//", label=f"pending {ROI_IDS[self.roi_index]}",
                )
            axis.set_xlim(bx0, bx1)
            axis.set_ylim(by1, by0)
            axis.set_xlabel("full-sensor u [px]")
            axis.set_ylabel("full-sensor v [px]")
            axis.set_title(title)
            axis.grid(alpha=0.15)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                unique = dict(zip(labels, handles))
                axis.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
        terminal = sum(
            value.get("selection_status") in TERMINAL_STATUSES for value in self.entries
        )
        self.figure.suptitle(
            f"{item.condition.condition} | {self.index + 1}/{len(self.entries)} | "
            f"status={entry['selection_status']} | current={ROI_IDS[self.roi_index]} | "
            f"terminal={terminal}/{len(self.entries)}\n{self.message}"
        )
        self.figure.canvas.draw_idle()

    def run(self) -> None:
        self.plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--operator", default=getpass.getuser())
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="prepare/reuse the 50 geometry-only representative views without opening GUI",
    )
    parser.add_argument(
        "--refresh-evidence", action="store_true",
        help="ignore compatible representative-view cache and rerun Steger for all 1000 frames",
    )
    parser.add_argument(
        "--refresh-condition", action="append", default=[], metavar="CONDITION",
        help="refresh one condition's 20-frame evidence; repeatable",
    )
    parser.add_argument(
        "--reselect", action="append", default=[], metavar="CONDITION",
        help="reopen a completed condition, e.g. --reselect h06_p03; repeatable",
    )
    parser.add_argument(
        "--start", default=None, metavar="CONDITION",
        help="condition initially shown; otherwise first non-terminal condition",
    )
    return parser.parse_args(argv)


def condition_index(value: str, evidence: list[Evidence]) -> int:
    for index, item in enumerate(evidence):
        if item.condition.condition == value:
            return index
    raise ManualRoiError(f"unknown condition: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend = "Agg" if args.prepare_only else "qtagg"
    try:
        matplotlib.use(backend, force=True)
    except (ImportError, ModuleNotFoundError) as error:
        if args.prepare_only:
            raise
        raise ManualRoiError(
            "cannot start ROI editor: a Qt Matplotlib backend is required"
        ) from error
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    operator = str(args.operator).strip() or "unspecified"
    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)

    conditions = discover_conditions(data_root)
    valid_condition_ids = {condition.condition for condition in conditions}
    refresh_conditions = set(args.refresh_condition)
    unknown_refresh = sorted(refresh_conditions - valid_condition_ids)
    if unknown_refresh:
        raise ManualRoiError(f"unknown --refresh-condition values: {unknown_refresh}")
    evidence, manifest = prepare_evidence(
        conditions,
        config_path,
        output,
        bool(args.refresh_evidence),
        refresh_conditions,
    )
    entries = load_entries(output, evidence, operator, manifest)
    for value in args.reselect:
        index = condition_index(value, evidence)
        entries[index]["selection_status"] = "in_progress"
        entries[index]["selected_at_utc"] = None
        entries[index]["notes"] = "reselection_requested"
    frozen = finalize_outputs(output, entries, evidence, manifest, operator)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared": len(evidence),
                    "reused": manifest["artifact_reuse"]["reused_condition_count"],
                    "fresh": manifest["artifact_reuse"]["fresh_condition_count"],
                    "frozen": frozen,
                    "output": str(output),
                    "next": "run without --prepare-only to open the manual editor",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.start:
        start_index = condition_index(args.start, evidence)
    else:
        start_index = next(
            (
                index
                for index, entry in enumerate(entries)
                if entry.get("selection_status") not in TERMINAL_STATUSES
            ),
            0,
        )
    editor = ManualRoiEditor(output, evidence, entries, manifest, operator, start_index)
    editor.run()
    frozen = finalize_outputs(output, entries, evidence, manifest, operator)
    print(
        json.dumps(
            {
                "terminal": sum(
                    entry.get("selection_status") in TERMINAL_STATUSES for entry in entries
                ),
                "expected": len(entries),
                "frozen": frozen,
                "draft": str(output / "manual_roi_registry_draft.json"),
                "registry": str(output / "manual_roi_registry.json") if frozen else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if frozen else 2


if __name__ == "__main__":
    raise SystemExit(main())
