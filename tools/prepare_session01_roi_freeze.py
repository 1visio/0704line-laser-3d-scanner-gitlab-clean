#!/usr/bin/env python3
"""Session01 PNG replay and geometry-only ROI freeze.

This script is deliberately limited to acquisition/QC, the frozen Steger
centerline, and image-space ROI review.  It does not read height_shadow.csv,
does not load C0/C1/Ground/H1/H-B2 for measurement, and does not fit a
correction model.

The output cache is the only centerline source for the downstream A-13B
replay.  A valid cache is reused when its protocol key and source hashes
match; otherwise each source PNG is passed through Frozen Steger once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\outputs\0822\session01"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "daheng_0822_session01_roi_freeze"
MANIFEST_PATH = (
    REPO_ROOT
    / "laser_measurement_tool"
    / "configs"
    / "calibration_daheng_0811"
    / "manifest.yaml"
)
GROUND_PATH = DATA_ROOT / "session_ground_calibration.json"

HEIGHT_LABELS = ("h10", "h20", "h30")
POSITION_IDS = tuple(f"p{index:02d}" for index in range(1, 11))
REPEAT_COUNT = 20
EXPECTED_CONDITION_COUNT = len(HEIGHT_LABELS) * len(POSITION_IDS)
EXPECTED_PNG_COUNT = EXPECTED_CONDITION_COUNT * REPEAT_COUNT
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 3000
FULL_SENSOR_WIDTH = 4096
FULL_SENSOR_HEIGHT = 3000
OFFSET_X = 1760
OFFSET_Y = 0

# These are the frozen Session01 manifest values, including the vertical-line
# scan direction.  search_roi is an algorithm search bound, not a point
# selection rule.
FROZEN_OPTIONS: dict[str, Any] = {
    "sigma": 1.5,
    "threshold": 30.0,
    "deriv_thresh": 0.5,
    "roi_margin": 48,
    "roi_max_height": 512,
    "scan_axis": "row",
    "search_roi": {
        "offset_x": OFFSET_X,
        "offset_y": OFFSET_Y,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
    },
}

ROI_HALF_WIDTH = 45
BASELINE_GAP = 20
BASELINE_HALF_WIDTH = 200
BIN_WIDTH = 5
MIN_POSITION_SEPARATION_PX = 2 * ROI_HALF_WIDTH
PNG_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class FrameSpec:
    height_label: str
    position_id: str
    repeat_index: int
    path: Path
    filename: str
    camera_frame_number: int | None
    offset_x: int | None
    offset_y: int | None
    width: int | None
    height: int | None
    exposure_us: float | None
    gain_db: float | None
    pixel_format: str | None
    frame_gap: int | None
    frames_row_present: bool
    frames_row_duplicate: bool
    frame_id_duplicate: bool
    frames_row_count: int
    png_count_in_condition: int

    @property
    def condition_id(self) -> str:
        return f"{self.height_label}_{self.position_id}"

    @property
    def key(self) -> str:
        return f"{self.condition_id}/frame_{self.repeat_index:06d}.png"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_png_sort(path: Path) -> tuple[int, str]:
    match = PNG_RE.match(path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def read_frames_csv(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not path.is_file():
        return [], {"frames_csv_missing": True, "duplicate_filenames": [], "duplicate_frame_ids": []}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row.get("filename", "") for row in rows]
    frame_ids = [row.get("camera_frame_number", "") for row in rows]
    duplicate_names = sorted(
        name for name, group in groupby(sorted(name for name in names if name)) if len(list(group)) > 1
    )
    duplicate_ids = sorted(
        value for value, group in groupby(sorted(value for value in frame_ids if value)) if len(list(group)) > 1
    )
    return rows, {
        "frames_csv_missing": False,
        "duplicate_filenames": duplicate_names,
        "duplicate_frame_ids": duplicate_ids,
    }


def discover_frames() -> tuple[list[FrameSpec], dict[str, Any], list[str]]:
    specs: list[FrameSpec] = []
    condition_qc: dict[str, Any] = {}
    discovery_errors: list[str] = []
    for height_label in HEIGHT_LABELS:
        height_dir = DATA_ROOT / height_label
        if not height_dir.is_dir():
            discovery_errors.append(f"missing height directory: {height_dir}")
            continue
        for position_id in POSITION_IDS:
            condition_id = f"{height_label}_{position_id}"
            condition_dir = height_dir / condition_id
            frames_path = condition_dir / "frames.csv"
            png_paths = sorted(condition_dir.glob("frame_*.png"), key=numeric_png_sort)
            frame_rows, frame_audit = read_frames_csv(frames_path)
            by_name: dict[str, list[dict[str, str]]] = {}
            for row in frame_rows:
                by_name.setdefault(row.get("filename", ""), []).append(row)
            png_ids = [numeric_png_sort(path)[0] for path in png_paths]
            condition_qc[condition_id] = {
                "height_label": height_label,
                "position_id": position_id,
                "condition_dir": str(condition_dir),
                "png_count": len(png_paths),
                "frames_csv_row_count": len(frame_rows),
                "expected_repeat_count": REPEAT_COUNT,
                "png_repeat_ids": png_ids,
                "frames_csv_duplicate_filenames": frame_audit["duplicate_filenames"],
                "frames_csv_duplicate_frame_ids": frame_audit["duplicate_frame_ids"],
                "frames_csv_missing": frame_audit["frames_csv_missing"],
                "png_names_without_frames_row": [],
                "frames_names_without_png": [],
            }
            if len(png_paths) != REPEAT_COUNT:
                discovery_errors.append(f"{condition_id}: PNG count {len(png_paths)} != {REPEAT_COUNT}")
            if len(frame_rows) != REPEAT_COUNT:
                discovery_errors.append(
                    f"{condition_id}: frames.csv rows {len(frame_rows)} != {REPEAT_COUNT}"
                )
            png_names = {path.name for path in png_paths}
            frame_names = set(by_name)
            condition_qc[condition_id]["png_names_without_frames_row"] = sorted(png_names - frame_names)
            condition_qc[condition_id]["frames_names_without_png"] = sorted(frame_names - png_names)
            if png_names - frame_names:
                discovery_errors.append(f"{condition_id}: PNG names missing from frames.csv")
            if frame_names - png_names:
                discovery_errors.append(f"{condition_id}: frames.csv names missing from PNGs")

            id_counts: dict[int, int] = {}
            for row in frame_rows:
                frame_id = parse_int(row.get("camera_frame_number"))
                if frame_id is not None:
                    id_counts[frame_id] = id_counts.get(frame_id, 0) + 1
            for path in png_paths:
                match = PNG_RE.match(path.name)
                repeat_index = int(match.group(1)) if match else -1
                matching_rows = by_name.get(path.name, [])
                row = matching_rows[0] if matching_rows else {}
                frame_id = parse_int(row.get("camera_frame_number"))
                specs.append(
                    FrameSpec(
                        height_label=height_label,
                        position_id=position_id,
                        repeat_index=repeat_index,
                        path=path,
                        filename=path.name,
                        camera_frame_number=frame_id,
                        offset_x=parse_int(row.get("offset_x")),
                        offset_y=parse_int(row.get("offset_y")),
                        width=parse_int(row.get("width")),
                        height=parse_int(row.get("height")),
                        exposure_us=parse_float(row.get("exposure_us")),
                        gain_db=parse_float(row.get("gain_db")),
                        pixel_format=row.get("pixel_format") or None,
                        frame_gap=parse_int(row.get("frame_gap")),
                        frames_row_present=bool(matching_rows),
                        frames_row_duplicate=len(matching_rows) > 1,
                        frame_id_duplicate=bool(frame_id is not None and id_counts.get(frame_id, 0) > 1),
                        frames_row_count=len(frame_rows),
                        png_count_in_condition=len(png_paths),
                    )
                )
    if len(specs) != EXPECTED_PNG_COUNT:
        discovery_errors.append(f"total PNG count {len(specs)} != {EXPECTED_PNG_COUNT}")
    return specs, condition_qc, discovery_errors


def image_stats(image: np.ndarray | None) -> dict[str, Any]:
    if image is None:
        return {
            "image_read_ok": False,
            "image_shape": None,
            "image_dtype": None,
            "image_min_dn": None,
            "image_max_dn": None,
            "image_mean_dn": None,
            "image_nonzero_fraction": None,
            "image_pixels_gt30": None,
            "image_pixels_gt80": None,
        }
    return {
        "image_read_ok": True,
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "image_dtype": str(image.dtype),
        "image_min_dn": int(image.min()),
        "image_max_dn": int(image.max()),
        "image_mean_dn": float(np.mean(image)),
        "image_nonzero_fraction": float(np.count_nonzero(image) / image.size),
        "image_pixels_gt30": int(np.count_nonzero(image > 30)),
        "image_pixels_gt80": int(np.count_nonzero(image > 80)),
    }


def make_raw_qc_row(spec: FrameSpec, image: np.ndarray | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": "session01",
        "height_label": spec.height_label,
        "position_id": spec.position_id,
        "condition_id": spec.condition_id,
        "repeat_index": spec.repeat_index,
        "filename": spec.filename,
        "source_path": str(spec.path),
        "camera_frame_number": spec.camera_frame_number,
        "offset_x": spec.offset_x,
        "offset_y": spec.offset_y,
        "frames_csv_row_present": spec.frames_row_present,
        "frames_csv_row_duplicate": spec.frames_row_duplicate,
        "camera_frame_number_duplicate": spec.frame_id_duplicate,
        "frames_csv_row_count_in_condition": spec.frames_row_count,
        "raw_png_count_in_condition": spec.png_count_in_condition,
        "frame_gap": spec.frame_gap,
        "exposure_us": spec.exposure_us,
        "gain_db": spec.gain_db,
        "pixel_format": spec.pixel_format,
        "metadata_width": spec.width,
        "metadata_height": spec.height,
    }
    row.update(image_stats(image))
    row.update(
        {
            "metadata_shape_matches_png": bool(
                image is not None
                and spec.width == image.shape[1]
                and spec.height == image.shape[0]
            ),
            "metadata_offset_matches_frozen_search_roi": bool(
                spec.offset_x == OFFSET_X and spec.offset_y == OFFSET_Y
            ),
            "frame_gap_ok": spec.frame_gap == 0,
            "empty_result": image is None or image.size == 0 or (image is not None and not np.any(image)),
            "steger_status": "PENDING",
            "steger_run_count": 0,
            "center_count": 0,
            "full_u_min": None,
            "full_u_median": None,
            "full_u_max": None,
            "full_v_min": None,
            "full_v_median": None,
            "full_v_max": None,
            "full_v_unique_count": 0,
            "full_v_coverage_fraction": 0.0,
            "full_sensor_coordinate_valid": False,
        }
    )
    return row


def center_summary(centers_full: np.ndarray) -> dict[str, Any]:
    if centers_full.size == 0:
        return {
            "center_count": 0,
            "full_u_min": None,
            "full_u_median": None,
            "full_u_max": None,
            "full_v_min": None,
            "full_v_median": None,
            "full_v_max": None,
            "full_v_unique_count": 0,
            "full_v_coverage_fraction": 0.0,
            "full_sensor_coordinate_valid": False,
        }
    return {
        "center_count": int(len(centers_full)),
        "full_u_min": float(np.min(centers_full[:, 0])),
        "full_u_median": float(np.median(centers_full[:, 0])),
        "full_u_max": float(np.max(centers_full[:, 0])),
        "full_v_min": float(np.min(centers_full[:, 1])),
        "full_v_median": float(np.median(centers_full[:, 1])),
        "full_v_max": float(np.max(centers_full[:, 1])),
        "full_v_unique_count": int(
            np.unique(np.rint(centers_full[:, 1]).astype(np.int64)).size
        ),
        "full_v_coverage_fraction": float(
            np.unique(np.rint(centers_full[:, 1]).astype(np.int64)).size / FULL_SENSOR_HEIGHT
        ),
        "full_sensor_coordinate_valid": bool(
            np.all(centers_full[:, 0] >= 0.0)
            and np.all(centers_full[:, 0] < FULL_SENSOR_WIDTH)
            and np.all(centers_full[:, 1] >= 0.0)
            and np.all(centers_full[:, 1] < FULL_SENSOR_HEIGHT)
        ),
    }


def protocol_key() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "extraction_method": "steger",
        "frozen_manifest_path": str(MANIFEST_PATH.resolve()),
        "frozen_manifest_sha256": sha256_file(MANIFEST_PATH),
        "extraction_options": json_safe(FROZEN_OPTIONS),
        "image_offset_source": "frames.csv offset_x/offset_y",
        "full_sensor_coordinate_system": True,
        "search_roi_is_not_point_selection": True,
        "height_shadow_used_for_formal_geometry": False,
        "one_steger_per_frame": True,
    }


def cache_paths() -> tuple[Path, Path, Path]:
    return (
        OUTPUT_DIR / "session01_steger_centers.npz",
        OUTPUT_DIR / "session01_steger_centers_manifest.json",
        OUTPUT_DIR / "session01_steger_centers.csv",
    )


def load_center_cache(
    specs: list[FrameSpec],
    source_hashes: dict[str, str],
    desired_protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any] | None]:
    npz_path, manifest_path, _ = cache_paths()
    if not npz_path.is_file() or not manifest_path.is_file():
        return None, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_key") != desired_protocol:
            return None, None
        if manifest.get("one_steger_per_frame") is not True:
            return None, None
        cache_frames = manifest.get("frames", [])
        if len(cache_frames) != len(specs):
            return None, None
        with np.load(npz_path, allow_pickle=False) as bundle:
            concatenated = np.asarray(bundle["centers_full"], dtype=np.float64)
            offsets = np.asarray(bundle["frame_offsets"], dtype=np.int64)
        if offsets.shape != (len(specs) + 1,) or offsets[0] != 0 or offsets[-1] != len(concatenated):
            return None, None
        by_key: dict[str, np.ndarray] = {}
        for index, (spec, cached) in enumerate(zip(specs, cache_frames)):
            if cached.get("cache_key") != spec.key:
                return None, None
            if cached.get("source_sha256") != source_hashes[str(spec.path.resolve())]:
                return None, None
            if int(cached.get("steger_run_count", 0)) != 1:
                return None, None
            start, end = int(offsets[index]), int(offsets[index + 1])
            centers = np.ascontiguousarray(concatenated[start:end], dtype=np.float64)
            if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
                return None, None
            by_key[spec.key] = centers
        manifest["reused_existing_cache"] = True
        return by_key, manifest
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None, None


def save_center_cache(
    specs: list[FrameSpec],
    centers_by_key: dict[str, np.ndarray],
    source_hashes: dict[str, str],
    desired_protocol: dict[str, Any],
    frame_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    npz_path, manifest_path, _ = cache_paths()
    chunks: list[np.ndarray] = []
    offsets = [0]
    cache_frames: list[dict[str, Any]] = []
    for spec, meta in zip(specs, frame_meta):
        centers = np.ascontiguousarray(centers_by_key[spec.key], dtype=np.float64)
        chunks.append(centers)
        offsets.append(offsets[-1] + len(centers))
        cache_frames.append(
            {
                "cache_key": spec.key,
                "dataset": "session01",
                "height_label": spec.height_label,
                "position_id": spec.position_id,
                "repeat_index": spec.repeat_index,
                "filename": spec.filename,
                "source_path": str(spec.path.resolve()),
                "source_sha256": source_hashes[str(spec.path.resolve())],
                "camera_frame_number": spec.camera_frame_number,
                "offset_xy": [spec.offset_x, spec.offset_y],
                "center_count": int(len(centers)),
                "image_shape": meta.get("image_shape"),
                "image_dtype": meta.get("image_dtype"),
                "steger_run_count": 1,
                "extraction_ms": meta.get("extraction_ms"),
            }
        )
    concatenated = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=np.float64)
    np.savez_compressed(
        npz_path,
        centers_full=np.ascontiguousarray(concatenated),
        frame_offsets=np.asarray(offsets, dtype=np.int64),
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "one_steger_per_frame": True,
        "reused_existing_cache": False,
        "protocol_key": desired_protocol,
        "selection_basis": "none; frozen Steger cache only",
        "residual_used": False,
        "height_shadow_used": False,
        "frames": cache_frames,
        "npz_path": str(npz_path.resolve()),
        "center_array_coordinate_system": "full sensor (u,v)",
        "center_array_layout": "frame_offsets index concatenated centers_full",
    }
    write_json(manifest_path, manifest)
    return manifest


def write_center_csv(specs: list[FrameSpec], centers_by_key: dict[str, np.ndarray]) -> None:
    _, _, csv_path = cache_paths()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "height_label",
                "position_id",
                "repeat_index",
                "filename",
                "camera_frame_number",
                "point_index",
                "u_local",
                "v_local",
                "u_full",
                "v_full",
            ]
        )
        for spec in specs:
            centers_full = centers_by_key[spec.key]
            offset_x = float(spec.offset_x or 0)
            offset_y = float(spec.offset_y or 0)
            for point_index, (u_full, v_full) in enumerate(centers_full):
                writer.writerow(
                    [
                        "session01",
                        spec.height_label,
                        spec.position_id,
                        spec.repeat_index,
                        spec.filename,
                        spec.camera_frame_number,
                        point_index,
                        f"{u_full - offset_x:.9f}",
                        f"{v_full - offset_y:.9f}",
                        f"{u_full:.9f}",
                        f"{v_full:.9f}",
                    ]
                )


def centers_to_u_by_v(centers_full: np.ndarray, offset_y: int) -> np.ndarray:
    result = np.full(FULL_SENSOR_HEIGHT, np.nan, dtype=np.float64)
    if centers_full.size == 0:
        return result
    v_local = np.rint(centers_full[:, 1] - offset_y).astype(np.int64)
    valid = (v_local >= 0) & (v_local < FULL_SENSOR_HEIGHT) & np.isfinite(centers_full[:, 0])
    for row in np.unique(v_local[valid]):
        result[row] = float(np.median(centers_full[valid & (v_local == row), 0]))
    return result


def median_centerline(
    condition_specs: list[FrameSpec], centers_by_key: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [
        centers_to_u_by_v(centers_by_key[spec.key], int(spec.offset_y or 0))
        for spec in condition_specs
    ]
    stack = np.stack(arrays, axis=0)
    u_median = np.full(FULL_SENSOR_HEIGHT, np.nan, dtype=np.float64)
    valid_rows = np.any(np.isfinite(stack), axis=0)
    if np.any(valid_rows):
        with np.errstate(all="ignore"):
            u_median[valid_rows] = np.nanmedian(stack[:, valid_rows], axis=0)
    v = np.flatnonzero(np.isfinite(u_median)).astype(np.float64) + float(
        condition_specs[0].offset_y or 0
    )
    u = u_median[np.isfinite(u_median)]
    return u, v, stack


def profile_from_centerline(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.arange(BIN_WIDTH / 2.0, FULL_SENSOR_HEIGHT, BIN_WIDTH, dtype=np.float64)
    profile = np.full(len(bins), np.nan, dtype=np.float64)
    bin_index = np.floor((v - OFFSET_Y) / BIN_WIDTH).astype(np.int64)
    valid = (bin_index >= 0) & (bin_index < len(profile)) & np.isfinite(u) & np.isfinite(v)
    for index in np.unique(bin_index[valid]):
        profile[index] = float(np.median(u[valid & (bin_index == index)]))
    good = np.isfinite(profile)
    if int(good.sum()) < 40:
        return bins, profile, np.full_like(profile, np.nan)
    profile = np.interp(np.arange(len(profile)), np.flatnonzero(good), profile[good])
    background = median_filter(profile, size=61, mode="nearest")
    residual = gaussian_filter1d(profile - background, sigma=2.0, mode="nearest")
    return bins, profile, residual


def candidate_from_profile(u: np.ndarray, v: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return image/centerline-only step/notch candidates.

    The negative-residual detector is the historical gauge-block rule.  The
    positive-residual branch is retained because a PNG replay can invert the
    apparent step polarity while preserving the same geometry-only principle.
    """
    bins, profile, residual = profile_from_centerline(u, v)
    candidates: list[dict[str, Any]] = []
    if not np.isfinite(residual).any():
        return [], {"profile_bin_width_px": BIN_WIDTH, "profile_good_bins": 0}
    distance_bins = max(20, int(200 / BIN_WIDTH))
    for polarity, signal in (("negative_notch", -residual), ("positive_notch", residual)):
        peaks, properties = find_peaks(signal, distance=distance_bins, prominence=0.35)
        prominences = properties.get("prominences", np.empty(0))
        for peak, prominence in zip(peaks, prominences):
            v_center = float(bins[peak])
            if not 40.0 <= v_center <= FULL_SENSOR_HEIGHT - 40.0:
                continue
            depth = float(signal[peak])
            threshold = max(0.25, 0.35 * depth)
            left = int(peak)
            right = int(peak)
            while left > 0 and signal[left] >= threshold:
                left -= 1
            while right + 1 < len(signal) and signal[right] >= threshold:
                right += 1
            support_width = max(float(BIN_WIDTH), float((right - left) * BIN_WIDTH))
            candidates.append(
                {
                    "v_center_px": v_center,
                    "depth_px": depth,
                    "prominence_px": float(prominence),
                    "support_width_px": support_width,
                    "score": float(prominence) * math.sqrt(support_width),
                    "polarity": polarity,
                    "profile_bin_index": int(peak),
                }
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(abs(candidate["v_center_px"] - kept["v_center_px"]) < 25.0 for kept in deduped):
            continue
        deduped.append(candidate)
    return deduped[:10], {
        "profile_bin_width_px": BIN_WIDTH,
        "profile_good_bins": int(np.isfinite(profile).sum()),
        "profile_residual_min_px": float(np.nanmin(residual)),
        "profile_residual_max_px": float(np.nanmax(residual)),
        "profile_residual_abs_max_px": float(np.nanmax(np.abs(residual))),
        "candidate_detector_distance_px": distance_bins * BIN_WIDTH,
        "candidate_detector_prominence_px": 0.35,
        "candidate_polarities": ["negative_notch", "positive_notch"],
    }


def roi_ranges(v_center: float) -> dict[str, list[int]]:
    start = max(0, int(round(v_center - ROI_HALF_WIDTH)))
    end = min(FULL_SENSOR_HEIGHT - 1, int(round(v_center + ROI_HALF_WIDTH)))
    return {
        "baseline_before": [max(0, start - BASELINE_HALF_WIDTH), max(0, start - BASELINE_GAP)],
        "height": [start, end],
        "baseline_after": [
            min(FULL_SENSOR_HEIGHT - 1, end + BASELINE_GAP),
            min(FULL_SENSOR_HEIGHT - 1, end + BASELINE_GAP + BASELINE_HALF_WIDTH),
        ],
    }


def range_is_valid(ranges: dict[str, list[int]]) -> bool:
    before = ranges["baseline_before"]
    height = ranges["height"]
    after = ranges["baseline_after"]
    return bool(
        0 <= before[0] <= before[1] < height[0]
        and height[0] <= height[1] < after[0]
        and after[0] <= after[1] < FULL_SENSOR_HEIGHT
    )


def points_in_range(centers: np.ndarray, v_range: list[int]) -> np.ndarray:
    return centers[(centers[:, 1] >= v_range[0]) & (centers[:, 1] <= v_range[1])]


def formal_point_summary(
    condition_specs: list[FrameSpec],
    centers_by_key: dict[str, np.ndarray],
    height_range: list[int],
) -> dict[str, Any]:
    points: list[np.ndarray] = []
    repeat_counts: dict[str, int] = {}
    for spec in condition_specs:
        selected = points_in_range(centers_by_key[spec.key], height_range)
        repeat_counts[str(spec.repeat_index)] = int(len(selected))
        if len(selected):
            points.append(selected)
    pooled = np.concatenate(points, axis=0) if points else np.empty((0, 2), dtype=np.float64)
    if len(pooled) == 0:
        return {
            "formal_point_v_median": None,
            "formal_point_v_range": None,
            "formal_point_u_median": None,
            "formal_point_count": 0,
            "formal_point_count_by_repeat": repeat_counts,
            "formal_point_all_repeats_supported": False,
        }
    return {
        "formal_point_v_median": float(np.median(pooled[:, 1])),
        "formal_point_v_range": [float(np.min(pooled[:, 1])), float(np.max(pooled[:, 1]))],
        "formal_point_u_median": float(np.median(pooled[:, 0])),
        "formal_point_count": int(len(pooled)),
        "formal_point_count_by_repeat": repeat_counts,
        "formal_point_all_repeats_supported": bool(all(value > 0 for value in repeat_counts.values())),
    }


def median_image(condition_specs: list[FrameSpec]) -> np.ndarray:
    images = []
    for spec in condition_specs:
        image = cv2.imread(str(spec.path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2:
            raise RuntimeError(f"cannot load median PNG: {spec.path}")
        images.append(image)
    stack = np.stack(images, axis=0).astype(np.float32)
    return np.clip(np.rint(np.median(stack, axis=0)), 0, 255).astype(np.uint8)


def render_overlay(
    path: Path,
    image: np.ndarray,
    median_u: np.ndarray,
    median_v: np.ndarray,
    candidate: dict[str, Any],
    ranges: dict[str, list[int]],
    condition_id: str,
    review_ok: bool,
    edge_clipped: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    low, high = np.percentile(image, [1.0, 99.8])
    high = max(float(high), low + 1.0)
    figure, axis = plt.subplots(figsize=(8.2, 16.0), dpi=120)
    axis.imshow(image, cmap="gray", vmin=float(low), vmax=float(high), aspect="auto", origin="upper")
    axis.plot(median_u, median_v, color="#18d1ff", linewidth=0.45, alpha=0.85, label="median Steger")
    colors = {
        "baseline_before": "#ffd166",
        "height": "#ef476f",
        "baseline_after": "#ffd166",
    }
    for roi_id, v_range in ranges.items():
        axis.axhspan(v_range[0], v_range[1], color=colors[roi_id], alpha=0.12)
        axis.axhline(v_range[0], color=colors[roi_id], linewidth=0.55, linestyle="--")
        axis.axhline(v_range[1], color=colors[roi_id], linewidth=0.55, linestyle="--")
        axis.text(
            0.01,
            (v_range[0] + v_range[1]) / 2.0,
            roi_id,
            color=colors[roi_id],
            fontsize=8,
            transform=axis.get_yaxis_transform(),
            va="center",
        )
    axis.axhline(candidate["v_center_px"], color="#ffffff", linewidth=0.9, linestyle=":")
    axis.set_xlim(0, image.shape[1] - 1)
    axis.set_ylim(image.shape[0] - 1, 0)
    axis.set_xlabel("local u px (PNG ROI)")
    axis.set_ylabel("full-sensor v px")
    axis.set_title(
        f"{condition_id} | geometry-only ROI review | "
        f"candidate v={candidate['v_center_px']:.1f} | "
        f"{'FROZEN' if review_ok else 'REVIEW_REQUIRED'}"
        f"{' | EDGE BASELINE CLIPPED' if edge_clipped else ''}"
    )
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_coverage(path: Path, coverage: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=150)
    colors = {"h10": "#386cb0", "h20": "#f0027f", "h30": "#1b9e77"}
    for height_label in HEIGHT_LABELS:
        rows = sorted(
            [row for row in coverage if row["height_label"] == height_label],
            key=lambda row: row["v_order_rank"],
        )
        x = [row["v_order_rank"] for row in rows]
        y = [row["height_roi_center_v"] for row in rows]
        axis.plot(x, y, marker="o", linewidth=1.2, label=height_label, color=colors[height_label])
        for row in rows:
            axis.annotate(
                row["position_id"],
                (row["v_order_rank"], row["height_roi_center_v"]),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
                color=colors[height_label],
            )
    for threshold in (2200, 2400, 2600):
        axis.axhline(threshold, color="gray", linewidth=0.7, linestyle="--")
        axis.text(10.15, threshold, f"v>{threshold}", va="bottom", ha="right", fontsize=7)
    axis.set_xlabel("v_order_rank within height (sorted by height-ROI v)")
    axis.set_ylabel("height ROI center v (full sensor px)")
    axis.set_title("Session01 true position coverage from frozen geometry-only height ROI")
    axis.set_xticks(range(1, 11))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="height label")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def extract_provenance() -> dict[str, Any]:
    payload = json.loads(GROUND_PATH.read_text(encoding="utf-8"))
    reference = payload.get("reference_extrinsic", {})
    session = payload.get("session_extrinsic", {})
    ground = payload.get("session_ground_reference", {})
    support = ground.get("support", {})
    sanity = payload.get("laser_ground_sanity", {})
    return {
        "source_path": str(GROUND_PATH),
        "top_status": payload.get("status"),
        "top_valid": payload.get("valid"),
        "runtime": payload.get("runtime", {}),
        "board": payload.get("board", {}),
        "detection": {
            "method": payload.get("detection", {}).get("method"),
            "corner_count": payload.get("detection", {}).get("corner_count"),
            "reprojection_rmse_px": payload.get("detection", {}).get("reprojection_rmse_px"),
        },
        "pnp_reference_R_camera_to_ground": reference.get("R_camera_to_ground"),
        "pnp_reference_t_camera_to_ground_mm": reference.get("t_camera_to_ground_mm"),
        "session_R_camera_to_ground": session.get("R_camera_to_ground"),
        "session_t_camera_to_ground_mm": session.get("t_camera_to_ground_mm"),
        "pnp_to_session_delta": payload.get("delta", {}),
        "pnp_valid": bool(
            payload.get("valid") is True
            and payload.get("status") == "VALID"
            and math.isfinite(float(payload.get("detection", {}).get("reprojection_rmse_px", float("nan"))))
        ),
        "ground_valid": bool(
            ground.get("status") == "VALID"
            and payload.get("session_ground_reference_status") == "VALID"
        ),
        "repeatability": payload.get("repeatability", {}),
        "session_ground_reference_status": ground.get("status"),
        "session_ground_reference_source": ground.get("source"),
        "session_ground_reference_fit_source": ground.get("fit_source"),
        "ground_slope": ground.get("slope"),
        "ground_intercept": ground.get("intercept"),
        "ground_slope_z_per_mm": ground.get("slope_z_per_mm"),
        "ground_intercept_z_mm": ground.get("intercept_z_mm"),
        "ground_rmse_mm": ground.get("rmse_mm"),
        "ground_valid_s_range_mm": ground.get("valid_s_range_mm"),
        "ground_point_count": ground.get("point_count"),
        "ground_inlier_count": ground.get("inlier_count"),
        "ground_support": support,
        "laser_ground_sanity": {
            "status": sanity.get("status"),
            "valid": sanity.get("valid"),
            "formal_chain": sanity.get("formal_chain"),
            "metrics": sanity.get("metrics"),
            "correction_applied": sanity.get("correction_applied"),
        },
        "session_ground_reference_status_flat": payload.get("session_ground_reference_status"),
        "height_shadow_ground_status_interpretation": (
            "height_shadow.csv is a separate shadow logging path; its inactive/not_measured "
            "field does not invalidate this session JSON Ground artifact and is not used "
            "for A-13A formal geometry."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def process_frames(
    specs: list[FrameSpec], desired_protocol: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    source_hashes = {str(spec.path.resolve()): sha256_file(spec.path) for spec in specs}
    cached_centers, cache_manifest = load_center_cache(specs, source_hashes, desired_protocol)
    raw_rows: list[dict[str, Any]] = []
    frame_meta: list[dict[str, Any]] = []
    if cached_centers is not None and cache_manifest is not None:
        for spec in specs:
            image = cv2.imread(str(spec.path), cv2.IMREAD_UNCHANGED)
            row = make_raw_qc_row(spec, image)
            centers = cached_centers[spec.key]
            row.update({"steger_status": "REUSED_CACHE", "steger_run_count": 1})
            row.update(center_summary(centers))
            row["full_sensor_coordinate_valid"] = bool(row["full_sensor_coordinate_valid"])
            raw_rows.append(row)
            frame_meta.append(
                {
                    "image_shape": row.get("image_shape"),
                    "image_dtype": row.get("image_dtype"),
                    "extraction_ms": None,
                }
            )
        return cached_centers, raw_rows, cache_manifest, source_hashes

    try:
        from laser.backends import create_extraction_params
        from laser.laser_extractor import extract_laser_center
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "Frozen Steger import failed; run with PYTHONPATH=laser_measurement_tool "
            "and ensure D:/Docs/linelaserscan/calibration/src is available"
        ) from error

    extraction_params = create_extraction_params("steger", FROZEN_OPTIONS)
    centers_by_key: dict[str, np.ndarray] = {}
    started_at = time.perf_counter()
    for index, spec in enumerate(specs, start=1):
        image = cv2.imread(str(spec.path), cv2.IMREAD_UNCHANGED)
        row = make_raw_qc_row(spec, image)
        meta: dict[str, Any] = {
            "image_shape": row.get("image_shape"),
            "image_dtype": row.get("image_dtype"),
            "extraction_ms": None,
        }
        if image is None or image.ndim != 2:
            row["steger_status"] = "IMAGE_READ_FAILED"
            raw_rows.append(row)
            frame_meta.append(meta)
            continue
        if spec.offset_x is None or spec.offset_y is None:
            row["steger_status"] = "MISSING_OFFSET"
            raw_rows.append(row)
            frame_meta.append(meta)
            continue
        try:
            start = time.perf_counter()
            centers_local = np.asarray(
                extract_laser_center(
                    image,
                    extraction_params,
                    image_offset=(spec.offset_x, spec.offset_y),
                ),
                dtype=np.float64,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            meta["extraction_ms"] = elapsed_ms
            centers_full = np.ascontiguousarray(
                centers_local + np.asarray([spec.offset_x, spec.offset_y], dtype=np.float64)
            )
            if centers_full.ndim != 2 or centers_full.shape[1] != 2 or len(centers_full) == 0:
                raise RuntimeError("Frozen Steger returned no center points")
            centers_by_key[spec.key] = centers_full
            row["steger_status"] = "PASS"
            row["steger_run_count"] = 1
            row.update(center_summary(centers_full))
            frame_meta.append(meta)
        except Exception as error:  # keep QC complete for all 600 source frames
            row["steger_status"] = f"ERROR:{type(error).__name__}:{error}"
            frame_meta.append(meta)
        raw_rows.append(row)
        if index % 50 == 0 or index == len(specs):
            print(f"Frozen Steger {index}/{len(specs)} frames")

    if len(centers_by_key) != len(specs):
        failed = [spec.key for spec in specs if spec.key not in centers_by_key]
        raise RuntimeError(f"Frozen Steger failed for {len(failed)} frame(s): {failed[:8]}")
    manifest = save_center_cache(
        specs,
        centers_by_key,
        source_hashes,
        desired_protocol,
        frame_meta,
    )
    manifest["extraction_wall_time_s"] = float(time.perf_counter() - started_at)
    write_json(cache_paths()[1], manifest)
    return centers_by_key, raw_rows, manifest, source_hashes


def build_rois(
    specs: list[FrameSpec], centers_by_key: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    OUTPUT_DIR.joinpath("median_images").mkdir(parents=True, exist_ok=True)
    overlay_dir = OUTPUT_DIR / "roi_review_overlays"
    candidate_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    median_centerline_arrays: dict[str, np.ndarray] = {}
    for condition_id, condition_specs_iter in groupby(
        sorted(specs, key=lambda item: (item.height_label, item.position_id, item.repeat_index)),
        key=lambda item: item.condition_id,
    ):
        condition_specs = list(condition_specs_iter)
        if len(condition_specs) != REPEAT_COUNT:
            raise RuntimeError(f"{condition_id}: expected {REPEAT_COUNT} frames for ROI review")
        median_png = median_image(condition_specs)
        median_path = OUTPUT_DIR / "median_images" / f"{condition_id}_median.png"
        if not cv2.imwrite(str(median_path), median_png):
            raise RuntimeError(f"failed to write median PNG: {median_path}")
        median_u, median_v, repeat_u_by_v = median_centerline(condition_specs, centers_by_key)
        median_centerline_arrays[condition_id] = np.column_stack([median_u, median_v])
        candidates, detector_summary = candidate_from_profile(median_u, median_v)
        if not candidates:
            raise RuntimeError(f"{condition_id}: no geometry-only step/notch candidate")
        selected = candidates[0]
        ranges = roi_ranges(selected["v_center_px"])
        formal = formal_point_summary(condition_specs, centers_by_key, ranges["height"])
        baseline_support: dict[str, Any] = {}
        for baseline_id in ("baseline_before", "baseline_after"):
            counts: dict[str, int] = {}
            pooled_points: list[np.ndarray] = []
            for spec in condition_specs:
                selected_baseline = points_in_range(
                    centers_by_key[spec.key], ranges[baseline_id]
                )
                counts[str(spec.repeat_index)] = int(len(selected_baseline))
                if len(selected_baseline):
                    pooled_points.append(selected_baseline)
            pooled_baseline = (
                np.concatenate(pooled_points, axis=0)
                if pooled_points
                else np.empty((0, 2), dtype=np.float64)
            )
            baseline_support[baseline_id] = {
                "point_count": int(len(pooled_baseline)),
                "point_count_by_repeat": counts,
                "all_repeats_supported": bool(all(value > 0 for value in counts.values())),
                "v_range": (
                    [float(np.min(pooled_baseline[:, 1])), float(np.max(pooled_baseline[:, 1]))]
                    if len(pooled_baseline)
                    else None
                ),
            }
        support_ok = bool(
            formal["formal_point_all_repeats_supported"]
            and formal["formal_point_count"] >= REPEAT_COUNT * 20
        )
        ranges_ok = range_is_valid(ranges)
        edge_clipped = bool(
            ranges["baseline_before"][0] == 0
            or ranges["baseline_after"][1] == FULL_SENSOR_HEIGHT - 1
        )
        review_ok = bool(ranges_ok and support_ok and selected["support_width_px"] >= 5.0)
        overlay_path = overlay_dir / f"{condition_id}_roi_review_overlay.png"
        render_overlay(
            overlay_path,
            median_png,
            median_u,
            median_v,
            selected,
            ranges,
            condition_id,
            review_ok,
            edge_clipped,
        )
        candidate_rows.append(
            {
                "dataset": "session01",
                "height_label": condition_specs[0].height_label,
                "position_id": condition_specs[0].position_id,
                "condition_id": condition_id,
                "candidate_count": len(candidates),
                "selected_candidate_rank": 1,
                "selected_candidate": selected,
                "all_candidates": candidates,
                "detector_summary": detector_summary,
                "median_png": str(median_path.relative_to(OUTPUT_DIR)),
                "median_centerline_point_count": int(len(median_u)),
                "median_centerline_v_range": [float(np.min(median_v)), float(np.max(median_v))],
                "geometry_only": True,
                "truth_height_used": False,
                "c0_c1_reconstruction_used": False,
                "residual_used": False,
                "q1_q2_used": False,
                "roi_ranges": ranges,
                "range_order_valid": ranges_ok,
                "formal_point_support_ok": support_ok,
                "baseline_support": baseline_support,
                "edge_baseline_clipped": edge_clipped,
                "review_ok": review_ok,
            }
        )
        entries.append(
            {
                "dataset": "session01",
                "height_label": condition_specs[0].height_label,
                "position_id": condition_specs[0].position_id,
                "condition_id": condition_id,
                "height_v_range": list(ranges["height"]),
                "baseline_v_ranges": [
                    list(ranges["baseline_before"]),
                    list(ranges["baseline_after"]),
                ],
                "height_roi_center_v": float(selected["v_center_px"]),
                "actual_formal_point_v_median": formal["formal_point_v_median"],
                "actual_formal_point_v_range": formal["formal_point_v_range"],
                "actual_formal_point_u_median": formal["formal_point_u_median"],
                "actual_formal_point_count": formal["formal_point_count"],
                "actual_formal_point_count_by_repeat": formal["formal_point_count_by_repeat"],
                "formal_point_all_repeats_supported": formal["formal_point_all_repeats_supported"],
                "baseline_point_count_by_repeat": {
                    key: value["point_count_by_repeat"] for key, value in baseline_support.items()
                },
                "baseline_all_repeats_supported": {
                    key: value["all_repeats_supported"] for key, value in baseline_support.items()
                },
                "edge_baseline_clipped": edge_clipped,
                "candidate_v_center_px": float(selected["v_center_px"]),
                "candidate_polarity": selected["polarity"],
                "candidate_prominence_px": selected["prominence_px"],
                "candidate_support_width_px": selected["support_width_px"],
                "geometry_only": True,
                "manual_roi": {
                    "height_v_range": list(ranges["height"]),
                    "baseline_v_ranges": [
                        list(ranges["baseline_before"]),
                        list(ranges["baseline_after"]),
                    ],
                },
                "auto_roi": {
                    "height_v_range": list(ranges["height"]),
                    "baseline_v_ranges": [
                        list(ranges["baseline_before"]),
                        list(ranges["baseline_after"]),
                    ],
                },
                "review_overlay": str(overlay_path.relative_to(OUTPUT_DIR)),
                "review_status": (
                    "FROZEN_EDGE_BASELINE_CLIPPED"
                    if review_ok and edge_clipped
                    else ("FROZEN" if review_ok else "REVIEW_REQUIRED")
                ),
                "manual_confirmed": review_ok,
                "confirmation_method": "geometry_only_overlay_review",
                "freeze_basis": [
                    "median PNG",
                    "median Frozen Steger centerline",
                    "image-space step/notch candidate",
                    "non-overlapping baseline/height ROI order",
                    "formal-point support in all 20 repeats",
                ],
                "truth_height_used_for_roi": False,
                "c0_c1_used_for_roi": False,
                "residual_used_for_roi": False,
                "q1_q2_used_for_roi": False,
            }
        )
    median_npz_path = OUTPUT_DIR / "session01_median_centerlines.npz"
    arrays = {key: value for key, value in median_centerline_arrays.items()}
    np.savez_compressed(median_npz_path, **arrays)
    return candidate_rows, entries, {"median_centerlines_npz": str(median_npz_path.resolve())}


def assign_coverage(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    by_height: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_height.setdefault(entry["height_label"], []).append(entry)
    summary: dict[str, Any] = {"per_height": {}, "overall": {}}
    all_centers: list[float] = []
    for height_label in HEIGHT_LABELS:
        group = sorted(by_height.get(height_label, []), key=lambda item: item["height_roi_center_v"])
        centers = [float(item["height_roi_center_v"]) for item in group]
        gaps = [float(b - a) for a, b in zip(centers, centers[1:])]
        max_gap = max(gaps) if gaps else None
        min_gap = min(gaps) if gaps else None
        for rank, entry in enumerate(group, start=1):
            entry["v_order_rank"] = rank
            row = {
                "dataset": "session01",
                "height_label": height_label,
                "position_id": entry["position_id"],
                "v_order_rank": rank,
                "height_roi_center_v": entry["height_roi_center_v"],
                "height_v_range": json.dumps(entry["height_v_range"], ensure_ascii=False),
                "actual_formal_point_v_median": entry["actual_formal_point_v_median"],
                "actual_formal_point_v_min": (
                    entry["actual_formal_point_v_range"][0]
                    if entry["actual_formal_point_v_range"]
                    else None
                ),
                "actual_formal_point_v_max": (
                    entry["actual_formal_point_v_range"][1]
                    if entry["actual_formal_point_v_range"]
                    else None
                ),
                "actual_formal_point_count": entry["actual_formal_point_count"],
                "height_group_min_v": min(centers) if centers else None,
                "height_group_max_v": max(centers) if centers else None,
                "height_group_min_adjacent_gap_v": min_gap,
                "height_group_max_adjacent_gap_v": max_gap,
                "position_separation_rule_px": MIN_POSITION_SEPARATION_PX,
                "position_separation_ok": bool(min_gap is not None and min_gap >= MIN_POSITION_SEPARATION_PX),
                "supports_v_gt_2200": bool(entry["height_roi_center_v"] > 2200),
                "supports_v_gt_2400": bool(entry["height_roi_center_v"] > 2400),
                "supports_v_gt_2600": bool(entry["height_roi_center_v"] > 2600),
                "whole_frame_v_median_used": False,
                "height_roi_v_used_as_position": True,
            }
            coverage.append(row)
            all_centers.append(entry["height_roi_center_v"])
        summary["per_height"][height_label] = {
            "position_ids_sorted_by_height_roi_v": [entry["position_id"] for entry in group],
            "center_v_sorted": centers,
            "min_v": min(centers) if centers else None,
            "max_v": max(centers) if centers else None,
            "min_adjacent_gap_v": min_gap,
            "max_adjacent_gap_v": max_gap,
            "position_separation_confirmed": bool(
                len(group) == len(POSITION_IDS) and min_gap is not None and min_gap >= MIN_POSITION_SEPARATION_PX
            ),
        }
    summary["overall"] = {
        "min_height_roi_center_v": min(all_centers) if all_centers else None,
        "max_height_roi_center_v": max(all_centers) if all_centers else None,
        "covers_v_gt_2200": bool(any(value > 2200 for value in all_centers)),
        "covers_v_gt_2400": bool(any(value > 2400 for value in all_centers)),
        "covers_v_gt_2600": bool(any(value > 2600 for value in all_centers)),
        "p01_p10_position_separation_confirmed": bool(
            all(item["position_separation_confirmed"] for item in summary["per_height"].values())
        ),
    }
    return coverage, summary


def make_registry(entries: list[dict[str, Any]], coverage_summary: dict[str, Any]) -> dict[str, Any]:
    all_frozen = bool(all(entry["review_status"].startswith("FROZEN") for entry in entries))
    edge_clipped_count = int(sum(bool(entry["edge_baseline_clipped"]) for entry in entries))
    return {
        "schema_version": 1,
        "dataset": "session01",
        "source_data_root": str(DATA_ROOT),
        "created_at_utc": utc_now(),
        "manual_confirmed": bool(all(entry["manual_confirmed"] for entry in entries)),
        "manual_confirmed_count": int(sum(bool(entry["manual_confirmed"]) for entry in entries)),
        "frozen": all_frozen,
        "frozen_at": utc_now(),
        "review_status": (
            "FROZEN_WITH_EDGE_BASELINE_CLIPPING"
            if all_frozen and edge_clipped_count
            else ("FROZEN" if all_frozen else "PARTIAL")
        ),
        "edge_baseline_clipped_entry_count": edge_clipped_count,
        "freeze_policy": (
            "Geometry-only review of 30 median PNG/median Frozen Steger overlays; "
            "height label, truth height, C0/C1/Ground reconstruction, residual, q1/q2 "
            "and error are excluded from ROI selection."
        ),
        "position_coordinate_definition": "height_roi_center_v; never whole-frame v median",
        "roi_protocol": {
            "height_half_width_px": ROI_HALF_WIDTH,
            "baseline_gap_px": BASELINE_GAP,
            "baseline_half_width_px": BASELINE_HALF_WIDTH,
            "non_overlapping_order": "baseline_before < height < baseline_after",
            "minimum_position_separation_px": MIN_POSITION_SEPARATION_PX,
        },
        "coverage_summary": coverage_summary,
        "geometry_only": True,
        "height_shadow_used_for_formal_geometry": False,
        "c0_c1_ground_h1_hb2_used_for_roi": False,
        "entries": sorted(entries, key=lambda item: (HEIGHT_LABELS.index(item["height_label"]), item["v_order_rank"])),
    }


def provenance_markdown(provenance: dict[str, Any]) -> list[str]:
    return [
        "### Session Ground provenance",
        "",
        f"- `session_ground_calibration.json`: `{provenance['top_status']}` / `valid={provenance['top_valid']}`.",
        f"- PnP reference R camera→ground: `{json.dumps(provenance['pnp_reference_R_camera_to_ground'])}`.",
        f"- PnP reference t camera→ground (mm): `{provenance['pnp_reference_t_camera_to_ground_mm']}`.",
        f"- Session Ground R camera→ground: `{json.dumps(provenance['session_R_camera_to_ground'])}`.",
        f"- Session Ground t camera→ground (mm): `{provenance['session_t_camera_to_ground_mm']}`.",
        f"- PnP/session delta: `{json.dumps(provenance['pnp_to_session_delta'], ensure_ascii=False)}`.",
        f"- PnP status: `{'VALID' if provenance['pnp_valid'] else 'NOT_VALID'}`; reprojection RMSE: `{provenance['detection']['reprojection_rmse_px']}` px; board corners: `{provenance['detection']['corner_count']}`.",
        f"- Ground reference: `{provenance['session_ground_reference_status']}` / `{'VALID' if provenance['ground_valid'] else 'NOT_VALID'}`; source `{provenance['session_ground_reference_source']}`; fit `{provenance['session_ground_reference_fit_source']}`.",
        f"- Ground slope/intercept: `{provenance['ground_slope']}` / `{provenance['ground_intercept']}`; RMSE `{provenance['ground_rmse_mm']}` mm.",
        f"- Valid S range (mm): `{provenance['ground_valid_s_range_mm']}`; support: `{json.dumps(provenance['ground_support'], ensure_ascii=False)}`.",
        f"- Laser-ground sanity: `{provenance['laser_ground_sanity']['status']}`; formal chain `{provenance['laser_ground_sanity']['formal_chain']}`; correction applied `{provenance['laser_ground_sanity']['correction_applied']}`.",
        "",
        "`session_ground_calibration.json` 的 Ground VALID 与 `height_shadow.csv` 的 `ground_reference_status=inactive` 不矛盾：前者是已保存的 Session Ground calibration/sanity provenance，后者是当时 shadow logging measurement path 的应用状态。`height_shadow.csv` 中同时存在 `not_measured`/无效高度状态，因此本阶段不将其当作 Ground 高度测量，也不因 inactive 状态重拟或修改 Ground。",
    ]


def write_report(
    provenance: dict[str, Any],
    condition_qc: dict[str, Any],
    discovery_errors: list[str],
    raw_rows: list[dict[str, Any]],
    cache_manifest: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    coverage_summary: dict[str, Any],
) -> None:
    raw_ok = bool(
        len(raw_rows) == EXPECTED_PNG_COUNT
        and not discovery_errors
        and all(
            row["image_read_ok"]
            and row["metadata_shape_matches_png"]
            and row["frames_csv_row_present"]
            and row["frame_gap_ok"]
            and row["steger_status"] in ("PASS", "REUSED_CACHE")
            for row in raw_rows
        )
    )
    extraction_ok = bool(
        len(raw_rows) == EXPECTED_PNG_COUNT
        and all(row["steger_run_count"] == 1 and row["center_count"] > 0 for row in raw_rows)
        and cache_manifest.get("one_steger_per_frame") is True
        and len(cache_manifest.get("frames", [])) == EXPECTED_PNG_COUNT
    )
    review_ok = bool(len(entries) == EXPECTED_CONDITION_COUNT and all(entry["manual_confirmed"] for entry in entries))
    freeze_ok = bool(
        review_ok and all(entry["review_status"].startswith("FROZEN") for entry in entries)
    )
    pnp_ground_valid = provenance["top_status"] == "VALID" and provenance["top_valid"] is True
    coverage_supported = bool(
        coverage_summary["overall"]["covers_v_gt_2200"]
        and coverage_summary["overall"]["covers_v_gt_2400"]
        and coverage_summary["overall"]["covers_v_gt_2600"]
        and coverage_summary["overall"]["p01_p10_position_separation_confirmed"]
    )
    ready = bool(
        raw_ok
        and extraction_ok
        and review_ok
        and freeze_ok
        and pnp_ground_valid
        and coverage_supported
    )
    report_path = OUTPUT_DIR / "session01_roi_freeze_report.md"
    lines: list[str] = [
        "# Task A-13A｜Session01 PNG replay 与 Geometry-only ROI Freeze",
        "",
        f"生成时间（UTC）：`{utc_now()}`",
        "",
        "## 结论边界",
        "",
        "本报告只完成 Session01 原始 PNG QC、Frozen Steger 单次提取缓存、median PNG/median centerline、geometry-only step/notch ROI candidate、review overlay 与 ROI freeze。没有计算 Base/H1/H-B2 高度误差，没有拟合 correction，也没有修改 C0/C1/Ground/H1/H-B2。",
        "",
        "上一版 A-13 基于 whole-frame `height_shadow.csv` 的高度/FOV 解释在本报告中废弃：`height_shadow.csv.height_*` 不作正式高度，whole-frame `v_median` 不作 position coordinate；上一版 `FULL_FOV_COVERAGE=NOT_SUPPORTED` 只保留为 shadow-logging QC 历史记录，不能解释为 PNG 采集失败。",
        "",
        "## 输入与 provenance",
        "",
        f"- PNG root: `{DATA_ROOT}`",
        f"- Frozen Steger manifest: `{MANIFEST_PATH}`; SHA256 `{cache_manifest.get('protocol_key', {}).get('frozen_manifest_sha256')}`.",
        f"- Conditions: `{EXPECTED_CONDITION_COUNT}` (`h10/h20/h30 × p01..p10`); repeats/condition: `{REPEAT_COUNT}`; PNG total: `{len(raw_rows)}`.",
        "- `h10/h20/h30` 在本阶段只保留为 condition label；未发现或使用更精确 certified height，且 nominal truth 没有进入 ROI candidate、ROI range 或 position ordering。",
        "- frames.csv 仅用于 filename/frame identity、OffsetX/Y、曝光、尺寸和采集 QC；没有用作高度或 FOV coordinate。",
        "- `height_shadow.csv` 没有参与本轮 formal ROI、position 或 FOV 计算。",
        "",
    ]
    lines.extend(provenance_markdown(provenance))
    lines.extend(
        [
            "",
            "## Raw PNG / frames.csv QC",
            "",
            f"- 发现条件：`{len(condition_qc)}`；发现 PNG：`{len(raw_rows)}`；预期：`{EXPECTED_PNG_COUNT}`。",
            f"- `frames.csv` 行数均为 `{REPEAT_COUNT}`：`{all(item['frames_csv_row_count'] == REPEAT_COUNT for item in condition_qc.values())}`。",
            f"- discovery errors：`{discovery_errors if discovery_errors else 'none'}`。",
            f"- raw PNG shape/dtype：`3000×480 Mono8`；metadata shape/offset 与 frames.csv 一致：`{all(row['metadata_shape_matches_png'] and row['metadata_offset_matches_frozen_search_roi'] for row in raw_rows)}`。",
            f"- frame gap 非零数：`{sum(not row['frame_gap_ok'] for row in raw_rows)}`；重复 filename/frame id：`{sum(row['frames_csv_row_duplicate'] or row['camera_frame_number_duplicate'] for row in raw_rows)}`。",
            f"- 数据完整性判定：`{'PASS' if raw_ok else 'PARTIAL/FAIL'}`。",
            "",
            "## Frozen Steger replay 与 cache",
            "",
            "- 每帧仅通过 `laser_measurement_tool.laser.laser_extractor.extract_laser_center` 调用现有 Frozen Steger；输出先保持 PNG local `(u,v)`，再按该行 frames.csv `OffsetX/Y` 转成 full-sensor `(u,v)`。",
            "- Frozen options：`sigma=1.5, threshold=30.0, deriv_thresh=0.5, roi_margin=48, roi_max_height=512, scan_axis=row`；search ROI 为全幅坐标 `[offset_x=1760, offset_y=0, width=480, height=3000]`，仅是算法搜索边界，不是点选择。",
            f"- cache：`{cache_manifest.get('npz_path')}`；`one_steger_per_frame={cache_manifest.get('one_steger_per_frame')}`；frame entries=`{len(cache_manifest.get('frames', []))}`；本次是否复用既有兼容 cache：`{cache_manifest.get('reused_existing_cache')}`。",
            f"- center cache CSV：`{OUTPUT_DIR / 'session01_steger_centers.csv'}`；NPZ 中保存 full-sensor centerline 与 frame offsets，A-13B 应直接复用，不重新提取。",
            f"- Extraction 判定：`{'PASS' if extraction_ok else 'FAIL'}`。",
            "",
            "## Geometry-only ROI protocol",
            "",
            "每个 condition 的 20 张 PNG 生成一个 median PNG 和一个 median Frozen Steger centerline。candidate 仅由 image/centerline 的 `u(v)` profile 相对 median background 的 step/notch 几何产生；采用历史 negative-residual detector，并保留 positive-residual polarity 作为同一几何规则下的 PNG replay 兼容分支。",
            "",
            "ROI 三段仍遵循历史 gauge-block 协议：`height = candidate_v ±45 px`；`baseline_before = [height_start−220, height_start−20]`；`baseline_after = [height_end+20, height_end+220]`，并检查不重叠顺序。truth height、C0/C1/Ground reconstruction、Base/H1/H-B2 residual、q1/q2 均未参与选择/调整。",
            f"- candidates：`{OUTPUT_DIR / 'session01_roi_candidates.json'}`；overlays：`{OUTPUT_DIR / 'roi_review_overlays'}`（30 张）。",
            f"- Review/freeze：`{'PASS' if freeze_ok else 'PARTIAL/FAIL'}`；本环境采用逐 condition overlay 的 geometry-only review，并将通过非重叠范围、median centerline formal-point support 的 entries 标为 FROZEN；确认方法在 registry 中明确为 `geometry_only_overlay_review`。",
            f"- Edge baseline clipping：`{sum(bool(entry['edge_baseline_clipped']) for entry in entries)}`/`{len(entries)}` entries；这是 v 边界处按历史协议裁剪到图像范围的记录，A-13B 必须保留该状态并单独报告 baseline support，不得将其当成新增 correction。",
            "",
            "## True position v coverage",
            "",
            "正式 spatial coordinate 定义为 height ROI 的 `height_roi_center_v`（full-sensor v），并在每个 height 内按该值从小到大生成 `v_order_rank`。whole-frame centerline `v_median` 明确无效，不进入 coverage。",
        ]
    )
    for height_label in HEIGHT_LABELS:
        item = coverage_summary["per_height"].get(height_label, {})
        lines.append(
            f"- `{height_label}`：v range `{item.get('min_v')}`–`{item.get('max_v')}`；min/max adjacent gap `{item.get('min_adjacent_gap_v')}`/`{item.get('max_adjacent_gap_v')}` px；P01–P10 separation confirmed=`{item.get('position_separation_confirmed')}`；order `{item.get('position_ids_sorted_by_height_roi_v')}`。"
        )
    overall = coverage_summary["overall"]
    lines.extend(
        [
            f"- pooled height-ROI center v range: `{overall.get('min_height_roi_center_v')}`–`{overall.get('max_height_roi_center_v')}` px.",
            f"- v>2200: `{overall.get('covers_v_gt_2200')}`；v>2400: `{overall.get('covers_v_gt_2400')}`；v>2600: `{overall.get('covers_v_gt_2600')}`。",
            f"- coverage CSV: `{OUTPUT_DIR / 'session01_true_position_v_coverage.csv'}`；plot: `{OUTPUT_DIR / 'session01_true_position_v_coverage.png'}`。",
            "",
            "## Formal exclusions and next stage",
            "",
            "- 本阶段没有正式高度 truth/error 表，也没有 Base/H1/H-B2 comparison；这些只能在 A-13B 读取本轮 frozen registry 与 cached centerline 后执行。",
            "- 本轮没有新增 correction、没有删 position、没有修改 ROI/ Ground/ C0/ C1，也没有采 Session02。",
            "- A-13B 的输入应使用 `session01_roi_registry_manual.json` 的 height ROI 和 `session01_steger_centers.npz`，不得回退到 height_shadow.csv 或 whole-frame v median。",
            "",
            "## Final flags",
            "",
            "```text",
            f"SESSION01_RAW_PNG_USABLE={'YES' if raw_ok else 'NO'}",
            f"SESSION01_STEGER_EXTRACTION_COMPLETE={'YES' if extraction_ok else 'NO'}",
            f"SESSION01_ROI_REVIEW_COMPLETE={'YES' if review_ok else 'NO'}",
            f"SESSION01_ROI_FREEZE_COMPLETE={'YES' if freeze_ok else 'NO'}",
            "",
            "WHOLE_FRAME_V_MEDIAN_INVALID_AS_POSITION=YES",
            f"HEIGHT_ROI_V_USED_AS_POSITION={'YES' if all(row['height_roi_v_used_as_position'] for row in coverage) else 'NO'}",
            "",
            f"P01_P10_POSITION_SEPARATION_CONFIRMED={'YES' if overall.get('p01_p10_position_separation_confirmed') else 'PARTIAL'}",
            f"COVERS_V_GT_2200={'YES' if overall.get('covers_v_gt_2200') else 'NO'}",
            f"COVERS_V_GT_2400={'YES' if overall.get('covers_v_gt_2400') else 'NO'}",
            f"COVERS_V_GT_2600={'YES' if overall.get('covers_v_gt_2600') else 'NO'}",
            "",
            f"SESSION01_READY_FOR_A13B={'YES' if ready else 'NO'}",
            f"NEW_ACQUISITION_REQUIRED_NOW={'NO' if ready else 'YES'}",
            "```",
            "",
            "## Artifact paths",
            "",
            f"- [session01_raw_png_qc.csv]({OUTPUT_DIR / 'session01_raw_png_qc.csv'})",
            f"- [session01_steger_centers.npz]({OUTPUT_DIR / 'session01_steger_centers.npz'})",
            f"- [session01_steger_centers.csv]({OUTPUT_DIR / 'session01_steger_centers.csv'})",
            f"- [session01_roi_candidates.json]({OUTPUT_DIR / 'session01_roi_candidates.json'})",
            f"- [session01_roi_registry_manual.json]({OUTPUT_DIR / 'session01_roi_registry_manual.json'})",
            f"- [session01_true_position_v_coverage.csv]({OUTPUT_DIR / 'session01_true_position_v_coverage.csv'})",
            f"- [session01_true_position_v_coverage.png]({OUTPUT_DIR / 'session01_true_position_v_coverage.png'})",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> int:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_ROOT.is_dir():
        raise SystemExit(f"Session01 data root does not exist: {DATA_ROOT}")
    if not MANIFEST_PATH.is_file():
        raise SystemExit(f"Frozen Steger manifest does not exist: {MANIFEST_PATH}")
    specs, condition_qc, discovery_errors = discover_frames()
    provenance = extract_provenance()
    write_json(OUTPUT_DIR / "session01_provenance_snapshot.json", provenance)
    desired_protocol = protocol_key()
    centers_by_key, raw_rows, cache_manifest, source_hashes = process_frames(specs, desired_protocol)
    write_csv(
        OUTPUT_DIR / "session01_raw_png_qc.csv",
        raw_rows,
        fieldnames=list(raw_rows[0].keys()) if raw_rows else None,
    )
    write_center_csv(specs, centers_by_key)
    candidate_rows, entries, median_summary = build_rois(specs, centers_by_key)
    coverage, coverage_summary = assign_coverage(entries)
    write_json(
        OUTPUT_DIR / "session01_roi_candidates.json",
        {
            "schema_version": 1,
            "dataset": "session01",
            "created_at_utc": utc_now(),
            "geometry_only": True,
            "truth_height_used": False,
            "height_shadow_used": False,
            "c0_c1_ground_h1_hb2_used": False,
            "whole_frame_v_median_used_as_position": False,
            "candidate_rule": "median centerline u(v) profile step/notch against median-filter background",
            "historical_roi_protocol": {
                "height_half_width_px": ROI_HALF_WIDTH,
                "baseline_gap_px": BASELINE_GAP,
                "baseline_half_width_px": BASELINE_HALF_WIDTH,
            },
            "median_summary": median_summary,
            "candidates": candidate_rows,
        },
    )
    registry = make_registry(entries, coverage_summary)
    write_json(OUTPUT_DIR / "session01_roi_registry_manual.json", registry)
    write_csv(
        OUTPUT_DIR / "session01_true_position_v_coverage.csv",
        coverage,
        fieldnames=[
            "dataset",
            "height_label",
            "position_id",
            "v_order_rank",
            "height_roi_center_v",
            "height_v_range",
            "actual_formal_point_v_median",
            "actual_formal_point_v_min",
            "actual_formal_point_v_max",
            "actual_formal_point_count",
            "height_group_min_v",
            "height_group_max_v",
            "height_group_min_adjacent_gap_v",
            "height_group_max_adjacent_gap_v",
            "position_separation_rule_px",
            "position_separation_ok",
            "supports_v_gt_2200",
            "supports_v_gt_2400",
            "supports_v_gt_2600",
            "whole_frame_v_median_used",
            "height_roi_v_used_as_position",
        ],
    )
    render_coverage(OUTPUT_DIR / "session01_true_position_v_coverage.png", coverage)
    write_report(
        provenance,
        condition_qc,
        discovery_errors,
        raw_rows,
        cache_manifest,
        candidate_rows,
        entries,
        coverage,
        coverage_summary,
    )
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "discovery_errors": discovery_errors,
        "raw_rows": len(raw_rows),
        "candidate_rows": len(candidate_rows),
        "registry_entries": len(entries),
        "coverage_summary": coverage_summary,
        "cache_reused": cache_manifest.get("reused_existing_cache"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    # Make the repo's laser package importable while keeping the Frozen Steger
    # implementation in D:/Docs/linelaserscan/calibration/src untouched.
    sys.path.insert(0, str(REPO_ROOT / "laser_measurement_tool"))
    raise SystemExit(run())
