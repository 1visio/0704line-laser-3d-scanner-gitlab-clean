"""Independent Frozen C0/C1 evaluation for the 2026-08-19 Daheng gauge blocks.

The ROI registry is generated from image-space centerline geometry only.  Each
image is passed through Steger once; C0 and C1 then consume the same centers.
This script never changes production configuration, Frozen correction files, or
measurement algorithms.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
if str(MEASUREMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_ROOT))

from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from laser.backends import create_extraction_params
from laser.laser_extractor import extract_laser_center
from measurement.height_measure import measure_height_line
from reconstruction.laser_ray_correction import (
    FrozenLaserRayCorrection,
    evaluate_frozen_laser_ray_correction,
)
from reconstruction.reconstructor import (
    ReconstructionParams,
    _intersect_laser_surface,
    reconstruct_uv_to_ground,
)
from utils.image_io import load_grayscale_image


DEFAULT_DATA_ROOT = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data"
)
DEFAULT_CONFIG = (
    REPO_ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819"

DATASETS = (
    "obs_1mm",
    "obs_2mm",
    "obs_6mm",
    "obs_10mm",
    "obs_20mm",
    "obs_30mm",
)
POSE_IDS = tuple(f"{index:03d}" for index in range(1, 6))
TRUTH_MM = {
    "obs_1mm": 1.001,
    "obs_2mm": 2.0,
    "obs_6mm": 6.0,
    "obs_10mm": 10.0,
    "obs_20mm": 20.0,
    "obs_30mm": 30.0,
}
DATASET_ORDER = {name: index for index, name in enumerate(DATASETS)}
MODEL_NAMES = ("C0", "C1")
MODE_NAMES = ("local_adjacent", "all_non_height", "fixed_zg_zero")
POSITION_SEPARATION_PX = 260.0
BIN_WIDTH_PX = 5.0
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


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.15g}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore", restval=""
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def file_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": sha256(resolved) if resolved.is_file() else None,
    }


def params_dict(params: ReconstructionParams) -> dict[str, Any]:
    return {
        item.name: json_safe(getattr(params, item.name))
        for item in fields(params)
    }


def parse_tiff_name(path: Path) -> tuple[str, int]:
    match = TIFF_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid gauge-block filename: {path.name}")
    return match.group("pose"), int(match.group("repeat") or "01")


def row_filename_key(value: str) -> str:
    return Path(value.replace("\\", "/")).as_posix()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if any(str(value or "").strip() for value in row.values())
        ]


def manifest_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    try:
        import yaml

        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"exists": True, "parse_error": f"{type(error).__name__}: {error}"}
    if not isinstance(document, dict):
        return {"exists": True, "top_level_type": type(document).__name__}
    plan = document.get("plan")
    if not isinstance(plan, dict):
        plan = {}
    plan_camera = plan.get("camera")
    if not isinstance(plan_camera, dict):
        plan_camera = {}
    return {
        "exists": True,
        "dataset_id": document.get("dataset_id"),
        "status": document.get("status"),
        "created_at": document.get("created_at"),
        "frame_count": document.get("frame_count"),
        "image_format": document.get("image_format"),
        "image_width": document.get("image_width"),
        "image_height": document.get("image_height"),
        "quality_passed": document.get("quality_passed"),
        "quality_warnings": document.get("quality_warnings"),
        "task_count": len(plan.get("tasks", []))
        if isinstance(plan.get("tasks"), list)
        else None,
        "plan_camera": plan_camera,
    }


def load_image_and_centers(
    path: Path,
    extraction_params: Any,
    offset_x: int,
    offset_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one image and invoke Steger exactly once."""
    image = load_grayscale_image(path)
    centers_local = np.asarray(
        extract_laser_center(
            image,
            extraction_params,
            image_offset=(offset_x, offset_y),
        ),
        dtype=np.float64,
    )
    if centers_local.size == 0:
        centers_local = np.empty((0, 2), dtype=np.float64)
    centers_local = centers_local.reshape(-1, 2)
    centers_full = np.ascontiguousarray(
        centers_local + np.array([offset_x, offset_y], dtype=np.float64)
    )
    return image, centers_full


def audit_and_extract(
    data_root: Path,
    extraction_params: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    global_sha_to_paths: dict[str, list[str]] = defaultdict(list)

    for dataset in DATASETS:
        folder = data_root / dataset
        frames_csv = folder / "frames.csv"
        manifest_path = folder / "dataset_manifest.yaml"
        csv_rows = read_csv_rows(frames_csv) if frames_csv.is_file() else []
        rows_by_file = {
            row_filename_key(str(row.get("filename", ""))): row for row in csv_rows
        }
        rows_by_basename = {
            Path(row_filename_key(str(row.get("filename", "")))).name: row
            for row in csv_rows
        }
        tiffs = sorted(folder.rglob("*.tif"))
        group_counts: dict[str, int] = defaultdict(int)
        dataset_errors: list[str] = []
        if len(csv_rows) != 25:
            dataset_errors.append(f"frames.csv row count {len(csv_rows)} != 25")
        if len(tiffs) != 25:
            dataset_errors.append(f"TIFF count {len(tiffs)} != 25")

        for image_path in tiffs:
            pose_id, repeat_index = parse_tiff_name(image_path)
            group_counts[pose_id] += 1
            csv_row = rows_by_file.get(image_path.name)
            if csv_row is None:
                csv_row = rows_by_file.get(row_filename_key(str(image_path)))
            if csv_row is None:
                csv_row = rows_by_basename.get(image_path.name)
            if csv_row is None:
                dataset_errors.append(f"missing frames.csv row: {image_path.name}")
                csv_row = {}
            offset_x = int(float(csv_row.get("offset_x") or 0))
            offset_y = int(float(csv_row.get("offset_y") or 0))
            expected_width = int(float(csv_row.get("width") or 0))
            expected_height = int(float(csv_row.get("height") or 0))
            expected_format = str(csv_row.get("pixel_format") or "")
            image, centers = load_image_and_centers(
                image_path, extraction_params, offset_x, offset_y
            )
            actual_sha = sha256(image_path)
            csv_sha = str(csv_row.get("sha256") or "")
            if csv_sha and csv_sha != actual_sha:
                dataset_errors.append(f"sha256 mismatch: {image_path.name}")
            global_sha_to_paths[actual_sha].append(str(image_path.resolve()))
            shape_ok = image.shape == (expected_height, expected_width)
            dtype_ok = (
                expected_format.lower() in {"mono8", "mono12", "mono12p"}
                and (
                    (expected_format.lower() == "mono8" and image.dtype == np.uint8)
                    or (
                        expected_format.lower() != "mono8"
                        and image.dtype in {np.uint16, np.uint8}
                    )
                )
            )
            if not shape_ok:
                dataset_errors.append(
                    f"shape mismatch {image_path.name}: {image.shape} "
                    f"vs ({expected_height}, {expected_width})"
                )
            if not dtype_ok:
                dataset_errors.append(
                    f"dtype/format mismatch {image_path.name}: "
                    f"{image.dtype}/{expected_format}"
                )
            entries.append(
                {
                    "dataset": dataset,
                    "height_truth_mm": TRUTH_MM[dataset],
                    "path": image_path,
                    "pose_id": pose_id,
                    "repeat_index": repeat_index,
                    "frames_csv": csv_row,
                    "image_shape": tuple(int(item) for item in image.shape),
                    "image_dtype": str(image.dtype),
                    "image_min": int(np.min(image)),
                    "image_max": int(np.max(image)),
                    "image_offset_x": offset_x,
                    "image_offset_y": offset_y,
                    "centers": centers,
                    "center_count": int(len(centers)),
                    "sha256": actual_sha,
                }
            )
            audit_rows.append(
                {
                    "dataset": dataset,
                    "filename": image_path.name,
                    "pose_id": pose_id,
                    "repeat_index": repeat_index,
                    "csv_row_present": bool(csv_row),
                    "csv_sha256": csv_sha,
                    "actual_sha256": actual_sha,
                    "sha256_match": bool(csv_sha) and csv_sha == actual_sha,
                    "csv_offset_x": offset_x,
                    "csv_offset_y": offset_y,
                    "csv_width": expected_width,
                    "csv_height": expected_height,
                    "csv_pixel_format": expected_format,
                    "actual_width": int(image.shape[1]),
                    "actual_height": int(image.shape[0]),
                    "actual_dtype": str(image.dtype),
                    "shape_match": shape_ok,
                    "dtype_match": dtype_ok,
                    "center_count": int(len(centers)),
                    "steger_called_once": True,
                    "quality": csv_row.get("quality_passed"),
                    "quality_warnings": csv_row.get("quality_warnings"),
                }
            )
        expected_poses = {f"{index:03d}" for index in range(1, 6)}
        if set(group_counts) != expected_poses:
            dataset_errors.append(
                f"pose groups {dict(group_counts)} != five laser001~005 groups"
            )
        for pose_id in expected_poses:
            if group_counts.get(pose_id, 0) != 5:
                dataset_errors.append(
                    f"{pose_id} repeat count {group_counts.get(pose_id, 0)} != 5"
                )
        audit_rows.append(
            {
                "dataset": dataset,
                "filename": "__dataset_summary__",
                "csv_row_present": len(csv_rows) == 25,
                "sha256_match": not dataset_errors,
                "shape_match": not any("shape mismatch" in item for item in dataset_errors),
                "dtype_match": not any(
                    "dtype/format mismatch" in item for item in dataset_errors
                ),
                "steger_called_once": True,
                "dataset_error_count": len(dataset_errors),
                "dataset_errors": " | ".join(dataset_errors),
                "manifest_json": json.dumps(
                    manifest_summary(manifest_path), ensure_ascii=False
                ),
            }
        )
    duplicate_shas = {
        digest: paths
        for digest, paths in global_sha_to_paths.items()
        if len(paths) > 1
    }
    summary = {
        "data_root": str(data_root.resolve()),
        "dataset_count": len(DATASETS),
        "image_count": len(entries),
        "expected_image_count": 150,
        "all_image_count_ok": len(entries) == 150,
        "duplicate_sha256_groups": duplicate_shas,
        "duplicate_sha256_group_count": len(duplicate_shas),
        "steger_call_count": len(entries),
        "steger_call_count_matches_images": len(entries) == 150,
    }
    return entries, audit_rows, summary


def profile_candidates(centers: np.ndarray, image_height: int) -> list[dict[str, float]]:
    """Find image-space centerline features; reconstruction is not involved."""
    if len(centers) < 50:
        return []
    v = centers[:, 1]
    u = centers[:, 0]
    bins = np.arange(BIN_WIDTH_PX / 2, image_height, BIN_WIDTH_PX)
    bin_index = np.floor(v / BIN_WIDTH_PX).astype(int)
    profile = np.full(len(bins), np.nan, dtype=np.float64)
    for index in range(len(bins)):
        values = u[bin_index == index]
        if len(values):
            profile[index] = float(np.median(values))
    good = np.isfinite(profile)
    if int(good.sum()) < 40:
        return []
    profile = np.interp(np.arange(len(profile)), np.flatnonzero(good), profile[good])
    background = median_filter(profile, size=61, mode="nearest")
    residual = gaussian_filter1d(profile - background, sigma=2.0, mode="nearest")
    peaks, properties = find_peaks(
        -residual,
        distance=max(20, int(200 / BIN_WIDTH_PX)),
        prominence=0.35,
    )
    candidates: list[dict[str, float]] = []
    for peak, prominence in zip(peaks, properties.get("prominences", [])):
        depth = float(-residual[peak])
        if not 40 <= bins[peak] <= image_height - 40:
            continue
        threshold = max(0.25, 0.35 * depth)
        left = int(peak)
        right = int(peak)
        while left > 0 and -residual[left] >= threshold:
            left -= 1
        while right + 1 < len(residual) and -residual[right] >= threshold:
            right += 1
        width = max(BIN_WIDTH_PX, (right - left) * BIN_WIDTH_PX)
        candidates.append(
            {
                "v_center_px": float(bins[peak]),
                "depth_px": depth,
                "prominence_px": float(prominence),
                "support_width_px": float(width),
                "score": float(prominence) * math.sqrt(width),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:10]


def assign_position_candidates(
    candidates_by_pose: dict[str, list[dict[str, float]]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    pose_ids = [f"{index:03d}" for index in range(1, 6)]
    missing = [pose_id for pose_id in pose_ids if not candidates_by_pose.get(pose_id)]
    if missing:
        raise RuntimeError(f"no geometry-only ROI candidates for {missing}")
    scored: list[tuple[float, tuple[dict[str, float], ...]]] = []
    for combination in itertools.product(
        *(candidates_by_pose[pose_id] for pose_id in pose_ids)
    ):
        centers = sorted(item["v_center_px"] for item in combination)
        if min(np.diff(centers)) < POSITION_SEPARATION_PX:
            continue
        score = sum(item["score"] for item in combination)
        score += 0.0001 * (max(centers) - min(centers))
        scored.append((float(score), combination))
    if not scored:
        raise RuntimeError("no five-position geometry-only ROI assignment")
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None
    gap = (
        None
        if second_score is None or best_score == 0
        else (best_score - second_score) / abs(best_score)
    )
    return (
        {pose_id: dict(candidate) for pose_id, candidate in zip(pose_ids, best)},
        {
            "candidate_count_by_pose": {
                pose_id: len(candidates_by_pose[pose_id]) for pose_id in pose_ids
            },
            "feasible_assignment_count": len(scored),
            "best_score": best_score,
            "second_score": second_score,
            "relative_score_gap": gap,
            "minimum_position_separation_px": POSITION_SEPARATION_PX,
            "manual_review_required": gap is None or gap < 0.10,
        },
    )


def build_roi_registry(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "geometry_only": True,
        "c0_c1_values_used": False,
        "height_half_width_px": 45,
        "baseline_gap_px": 20,
        "baseline_half_width_px": 200,
        "position_separation_px": POSITION_SEPARATION_PX,
        "datasets": {},
    }
    for dataset in DATASETS:
        dataset_entries = [item for item in entries if item["dataset"] == dataset]
        first_by_pose = {
            pose_id: sorted(
                [
                    item
                    for item in dataset_entries
                    if item["pose_id"] == pose_id
                ],
                key=lambda item: item["repeat_index"],
            )[0]
            for pose_id in (f"{index:03d}" for index in range(1, 6))
        }
        candidates_by_pose = {
            pose_id: profile_candidates(
                first_by_pose[pose_id]["centers"],
                first_by_pose[pose_id]["image_shape"][0],
            )
            for pose_id in first_by_pose
        }
        assignment, diagnostics = assign_position_candidates(candidates_by_pose)
        for pose_id, candidates in candidates_by_pose.items():
            for rank, candidate in enumerate(candidates, start=1):
                candidate_rows.append(
                    {
                        "dataset": dataset,
                        "pose_id": pose_id,
                        "candidate_rank_by_score": rank,
                        **candidate,
                        "selected": abs(
                            candidate["v_center_px"]
                            - assignment[pose_id]["v_center_px"]
                        )
                        < 1.0e-9,
                    }
                )
        pose_centers: dict[str, float] = {}
        for pose_id, candidate in assignment.items():
            height_range = (
                max(0, int(round(candidate["v_center_px"] - 45))),
                min(
                    first_by_pose[pose_id]["image_shape"][0] - 1,
                    int(round(candidate["v_center_px"] + 45)),
                ),
            )
            pooled: list[float] = []
            repeat_counts: dict[str, int] = {}
            for item in dataset_entries:
                if item["pose_id"] != pose_id:
                    continue
                mask = (
                    (item["centers"][:, 1] >= height_range[0])
                    & (item["centers"][:, 1] <= height_range[1])
                )
                repeat_counts[str(item["repeat_index"])] = int(mask.sum())
                pooled.extend(item["centers"][mask, 1].tolist())
            pose_center = (
                float(np.median(np.asarray(pooled, dtype=np.float64)))
                if pooled
                else float(candidate["v_center_px"])
            )
            start, end = height_range
            baseline_ranges = (
                (max(0, start - 220), max(0, start - 20)),
                (
                    min(first_by_pose[pose_id]["image_shape"][0] - 1, end + 20),
                    min(first_by_pose[pose_id]["image_shape"][0] - 1, end + 220),
                ),
            )
            registry.append(
                {
                    "dataset": dataset,
                    "pose_id": pose_id,
                    "candidate_v_center_px": float(candidate["v_center_px"]),
                    "v_center_px": pose_center,
                    "height_v_range": list(height_range),
                    "baseline_v_ranges": [list(item) for item in baseline_ranges],
                    "height_point_count_by_repeat": repeat_counts,
                    "manual_review_required": bool(
                        diagnostics["manual_review_required"]
                        or not pooled
                        or any(count < 20 for count in repeat_counts.values())
                    ),
                    "candidate_evidence": candidate,
                    "assignment_diagnostics": diagnostics,
                    "geometry_only": True,
                }
            )
            pose_centers[pose_id] = pose_center
        sorted_poses = sorted(pose_centers.items(), key=lambda item: item[1])
        rank_by_pose = {pose_id: rank for rank, (pose_id, _) in enumerate(sorted_poses, 1)}
        for item in registry:
            if item["dataset"] == dataset:
                item["position_rank"] = rank_by_pose[item["pose_id"]]
        summary["datasets"][dataset] = {
            "position_order_pose_ids": [pose_id for pose_id, _ in sorted_poses],
            "position_order_v_center_px": [value for _, value in sorted_poses],
            "assignment": assignment,
            "assignment_diagnostics": diagnostics,
        }
    summary["manual_review_required"] = any(
        item["manual_review_required"] for item in registry
    )
    roi_by_key = {(item["dataset"], item["pose_id"]): item for item in registry}
    for item in entries:
        item["roi"] = roi_by_key[(item["dataset"], item["pose_id"])]
        item["position_rank"] = item["roi"]["position_rank"]
    registry.sort(key=lambda item: (DATASET_ORDER[item["dataset"]], item["position_rank"]))
    return registry, candidate_rows, summary


def load_frozen_roi_registry(
    path: Path,
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate a fully manually confirmed registry.

    This function intentionally does not call any automatic candidate detector.
    The supplied ranges are the only ROI values used downstream.
    """
    source = path.resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(
            "frozen ROI registry must be a JSON object with manual freeze metadata"
        )
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 30:
        raise ValueError("frozen ROI registry must contain exactly 30 entries")
    if document.get("manual_confirmed") is not True:
        raise ValueError("frozen ROI registry top-level manual_confirmed must be true")
    if int(document.get("manual_confirmed_count", 0)) != 30:
        raise ValueError("frozen ROI registry manual_confirmed_count must be 30")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    expected_keys = {
        (dataset, pose_id) for dataset in DATASETS for pose_id in POSE_IDS
    }
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("each frozen ROI registry entry must be an object")
        key = (str(raw.get("dataset")), str(raw.get("pose_id")))
        if key in seen:
            raise ValueError(f"duplicate frozen ROI entry: {key}")
        seen.add(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected frozen ROI entry: {key}")
        if raw.get("manual_confirmed") is not True:
            raise ValueError(f"ROI {key} is not manually confirmed")
        try:
            height_range = [
                int(raw["height_v_range"][0]),
                int(raw["height_v_range"][1]),
            ]
            baseline_ranges = [
                [int(raw["baseline_v_ranges"][0][0]), int(raw["baseline_v_ranges"][0][1])],
                [int(raw["baseline_v_ranges"][1][0]), int(raw["baseline_v_ranges"][1][1])],
            ]
            position_rank = int(raw["position_rank"])
            v_center = float(raw["v_center_px"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f"invalid frozen ROI fields for {key}: {error}") from error
        image_height = next(
            int(item["image_shape"][0]) for item in entries if item["dataset"] == key[0]
        )
        if not 0 <= height_range[0] <= height_range[1] < image_height:
            raise ValueError(f"height_v_range outside image for {key}: {height_range}")
        if not all(
            0 <= item[0] <= item[1] < image_height for item in baseline_ranges
        ):
            raise ValueError(f"baseline_v_ranges outside image for {key}: {baseline_ranges}")
        if not (
            baseline_ranges[0][1] < height_range[0]
            and height_range[1] < baseline_ranges[1][0]
        ):
            raise ValueError(f"baseline/height ranges overlap or are unordered for {key}")
        normalized_entry = dict(raw)
        normalized_entry["height_v_range"] = height_range
        normalized_entry["baseline_v_ranges"] = baseline_ranges
        normalized_entry["position_rank"] = position_rank
        normalized_entry["v_center_px"] = v_center
        normalized.append(normalized_entry)
    if seen != expected_keys:
        raise ValueError(f"frozen ROI registry missing entries: {sorted(expected_keys - seen)}")
    for dataset in DATASETS:
        ranks = sorted(
            entry["position_rank"]
            for entry in normalized
            if entry["dataset"] == dataset
        )
        if ranks != [1, 2, 3, 4, 5]:
            raise ValueError(f"{dataset} frozen position ranks must be 1..5, got {ranks}")
    normalized.sort(
        key=lambda item: (DATASET_ORDER[item["dataset"]], item["position_rank"])
    )
    roi_by_key = {(item["dataset"], item["pose_id"]): item for item in normalized}
    for entry in entries:
        roi = roi_by_key[(entry["dataset"], entry["pose_id"])]
        entry["roi"] = roi
        entry["position_rank"] = roi["position_rank"]
    summary = {
        "source": "manual_frozen",
        "registry_path": str(source),
        "registry_sha256": sha256(source),
        "geometry_only": True,
        "c0_c1_values_used": False,
        "manual_confirmed": True,
        "manual_confirmed_count": 30,
        "manual_review_required": False,
        "datasets": {
            dataset: {
                "position_order_pose_ids": [
                    item["pose_id"]
                    for item in normalized
                    if item["dataset"] == dataset
                ],
                "position_order_v_center_px": [
                    item["v_center_px"]
                    for item in normalized
                    if item["dataset"] == dataset
                ],
            }
            for dataset in DATASETS
        },
    }
    return normalized, summary


def save_overlay(
    output_path: Path,
    image_path: Path,
    roi: dict[str, Any],
    image_offset_x: int,
    centers: np.ndarray,
) -> None:
    image = load_grayscale_image(image_path)
    low, high = np.percentile(image, [1.0, 99.8])
    rendered = np.clip(
        (image.astype(np.float32) - low) * 255.0 / max(1.0, high - low),
        0,
        255,
    ).astype(np.uint8)
    x0 = max(0, image_offset_x)
    x1 = min(rendered.shape[1], image_offset_x + 480)
    canvas = cv2.cvtColor(rendered[:, x0:x1], cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(canvas, (max(1, canvas.shape[1] // 2), 900))
    scale_v = canvas.shape[0] / rendered.shape[0]
    crop_width = max(1, x1 - x0)
    point_step = max(1, len(centers) // 2500)
    for center in centers[::point_step]:
        u, v = float(center[0]), float(center[1])
        if x0 <= u < x1 and 0 <= v < rendered.shape[0]:
            x = int(round((u - x0) * canvas.shape[1] / crop_width))
            y = int(round(v * scale_v))
            cv2.circle(canvas, (x, y), 1, (220, 0, 220), -1, cv2.LINE_AA)
    for center in centers:
        u, v = float(center[0]), float(center[1])
        if (
            x0 <= u < x1
            and roi["height_v_range"][0] <= v <= roi["height_v_range"][1]
        ):
            x = int(round((u - x0) * canvas.shape[1] / crop_width))
            y = int(round(v * scale_v))
            cv2.circle(canvas, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)
    regions = (
        ("height", roi["height_v_range"], (0, 220, 255)),
        ("baseline_before", roi["baseline_v_ranges"][0], (255, 150, 0)),
        ("baseline_after", roi["baseline_v_ranges"][1], (70, 220, 70)),
    )
    for label, (v0, v1), color in regions:
        y0, y1 = int(v0 * scale_v), int((v1 + 1) * scale_v)
        cv2.rectangle(canvas, (0, y0), (canvas.shape[1] - 1, y1), color, 2)
        cv2.putText(
            canvas,
            f"{label}: {v0}-{v1}",
            (8, max(18, y0 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"{roi['dataset']} pose {roi['pose_id']} position {roi['position_rank']} "
        f"v={roi['v_center_px']:.1f}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write overlay: {output_path}")


def save_registry_plot(output_path: Path, registry: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for axis, dataset in zip(axes, DATASETS):
        rows = sorted(
            [item for item in registry if item["dataset"] == dataset],
            key=lambda item: item["v_center_px"],
        )
        for item in rows:
            v0, v1 = item["height_v_range"]
            b0, b1 = item["baseline_v_ranges"][0]
            c0, c1 = item["baseline_v_ranges"][1]
            axis.axvspan(v0, v1, color="tab:orange", alpha=0.35)
            axis.axvspan(b0, b1, color="tab:blue", alpha=0.18)
            axis.axvspan(c0, c1, color="tab:green", alpha=0.18)
            axis.axvline(item["v_center_px"], color="black", linewidth=0.8)
            axis.text(
                item["v_center_px"],
                0.5,
                str(item["position_rank"]),
                transform=axis.get_xaxis_transform(),
            )
        axis.set_ylabel(dataset.replace("obs_", ""))
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("full-frame v; orange=height, blue/green=adjacent baseline")
    fig.suptitle("Geometry-only fixed ROI registry")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def point_key(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def result_map(result: Any) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    mapped: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for uv, camera, ground in zip(
        result.pixels_uv, result.points_camera, result.points_ground
    ):
        key = point_key(uv)
        if key in mapped:
            raise RuntimeError(f"duplicate reconstructed center key: {key}")
        mapped[key] = (camera, ground)
    return mapped


def valid_filter_reason(
    index: int,
    key: tuple[float, float],
    valid_keys: set[tuple[float, float]],
    lambdas: np.ndarray,
    stable: np.ndarray,
    params: ReconstructionParams,
) -> str:
    if key in valid_keys:
        return "valid"
    if not bool(stable[index]) or not np.isfinite(lambdas[index]):
        return "no_valid_intersection"
    value = float(lambdas[index])
    if value <= 0:
        return "negative_depth"
    if not params.min_camera_depth_mm <= value <= params.max_camera_depth_mm:
        return "outside_working_distance"
    return "non_finite_ground_or_post_filter"


def roi_region(v: float, roi: dict[str, Any]) -> str:
    if roi["height_v_range"][0] <= v <= roi["height_v_range"][1]:
        return "height"
    if roi["baseline_v_ranges"][0][0] <= v <= roi["baseline_v_ranges"][0][1]:
        return "baseline_before"
    if roi["baseline_v_ranges"][1][0] <= v <= roi["baseline_v_ranges"][1][1]:
        return "baseline_after"
    return "other"


def measure_mode(
    ground: np.ndarray,
    pixels_uv: np.ndarray,
    roi: dict[str, Any],
    mode: str,
    params: Any,
) -> tuple[float | None, float | None, float | None, int, int, str]:
    v = pixels_uv[:, 1]
    height_mask = (
        (v >= roi["height_v_range"][0]) & (v <= roi["height_v_range"][1])
    )
    baseline_mask = (
        (
            (v >= roi["baseline_v_ranges"][0][0])
            & (v <= roi["baseline_v_ranges"][0][1])
        )
        | (
            (v >= roi["baseline_v_ranges"][1][0])
            & (v <= roi["baseline_v_ranges"][1][1])
        )
    )
    if mode == "local_adjacent":
        height_points = ground[height_mask]
        baseline_points = ground[baseline_mask]
        baseline_count = int(baseline_mask.sum())
    elif mode == "all_non_height":
        height_points = ground[height_mask]
        baseline_points = ground[~height_mask]
        baseline_count = int((~height_mask).sum())
    elif mode == "fixed_zg_zero":
        height_points = ground[height_mask]
        baseline_points = None
        baseline_count = 0
    else:
        raise ValueError(f"unknown height mode: {mode}")
    if not len(height_points):
        return None, None, None, int(height_mask.sum()), baseline_count, "insufficient_points"
    if mode != "fixed_zg_zero" and not len(baseline_points):
        return None, None, None, int(height_mask.sum()), baseline_count, "insufficient_points"
    measurement = measure_height_line(baseline_points, height_points, params)
    return (
        float(measurement.height_mean_mm),
        float(measurement.height_median_mm),
        float(measurement.height_std_mm),
        int(height_mask.sum()),
        baseline_count,
        "success",
    )


def reconstruct_all(
    entries: list[dict[str, Any]],
    calibration: dict[str, Any],
    app: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    params_c0 = replace(app.reconstruction, enable_laser_ray_correction=False)
    params_c1 = replace(app.reconstruction, enable_laser_ray_correction=True)
    correction = calibration.get("laser_ray_correction")
    if not isinstance(correction, FrozenLaserRayCorrection):
        raise RuntimeError("calibration did not load a Frozen C1 correction")
    frame_rows: list[dict[str, Any]] = []
    height_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for entry in entries:
        centers = entry["centers"]
        if len(centers) == 0:
            raise RuntimeError(f"Steger returned no points: {entry['path']}")
        c0_result = reconstruct_uv_to_ground(centers, calibration, params_c0)
        c1_result = reconstruct_uv_to_ground(centers, calibration, params_c1)
        c0_map = result_map(c0_result)
        c1_map = result_map(c1_result)
        center_keys = [point_key(point) for point in centers]
        if len(set(center_keys)) != len(center_keys):
            raise RuntimeError(f"duplicate Steger centers: {entry['path']}")

        normalized = cv2.undistortPoints(
            centers.reshape(-1, 1, 2), calibration["K"], calibration["D"]
        ).reshape(-1, 2)
        rays = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        lambda_c0, stable, model_type = _intersect_laser_surface(
            rays, calibration, params_c0
        )
        if model_type != "quadratic_graph":
            raise RuntimeError(f"expected quadratic_graph C0 model, got {model_type}")
        correction_eval = evaluate_frozen_laser_ray_correction(rays, correction)
        delta_lambda = np.asarray(correction_eval.correction_mm, dtype=np.float64)
        lambda_c1 = lambda_c0 + delta_lambda
        c0_valid_keys = set(c0_map)
        c1_valid_keys = set(c1_map)
        c0_reasons = [
            valid_filter_reason(
                index, key, c0_valid_keys, lambda_c0, stable, params_c0
            )
            for index, key in enumerate(center_keys)
        ]
        c1_reasons = [
            valid_filter_reason(
                index, key, c1_valid_keys, lambda_c1, stable, params_c1
            )
            for index, key in enumerate(center_keys)
        ]
        clamped = np.asarray(correction_eval.clamped, dtype=bool)
        frame_rows.append(
            {
                "dataset": entry["dataset"],
                "truth_mm": entry["height_truth_mm"],
                "pose_id": entry["pose_id"],
                "position_rank": entry["position_rank"],
                "repeat_index": entry["repeat_index"],
                "filename": entry["path"].name,
                "sha256": entry["sha256"],
                "center_count": len(centers),
                "c0_valid_points": len(c0_map),
                "c1_valid_points": len(c1_map),
                "c0_filtered_points": len(centers) - len(c0_map),
                "c1_filtered_points": len(centers) - len(c1_map),
                "common_valid_points": len(c0_valid_keys & c1_valid_keys),
                "c0_filter_reasons_json": json.dumps(
                    {reason: c0_reasons.count(reason) for reason in sorted(set(c0_reasons))}
                ),
                "c1_filter_reasons_json": json.dumps(
                    {reason: c1_reasons.count(reason) for reason in sorted(set(c1_reasons))}
                ),
                "clamp_points": int(clamped.sum()),
                "clamp_rate": float(clamped.mean()) if len(clamped) else None,
                "delta_lambda_mean_mm": float(np.mean(delta_lambda)),
                "delta_lambda_median_mm": float(np.median(delta_lambda)),
                "delta_lambda_p95_abs_mm": float(
                    np.quantile(np.abs(delta_lambda), 0.95)
                ),
                "c0_model_type": model_type,
                "roi_geometry_only": True,
                "steger_called_once": True,
            }
        )

        for index, center in enumerate(centers):
            key = center_keys[index]
            c0_camera, c0_ground = c0_map.get(key, (None, None))
            c1_camera, c1_ground = c1_map.get(key, (None, None))
            point_rows.append(
                {
                    "dataset": entry["dataset"],
                    "truth_mm": entry["height_truth_mm"],
                    "pose_id": entry["pose_id"],
                    "position_rank": entry["position_rank"],
                    "repeat_index": entry["repeat_index"],
                    "filename": entry["path"].name,
                    "point_index": index,
                    "u_px": float(center[0]),
                    "v_px": float(center[1]),
                    "image_region": roi_region(float(center[1]), entry["roi"]),
                    "lambda_c0_mm": float(lambda_c0[index]),
                    "delta_lambda_mm": float(delta_lambda[index]),
                    "lambda_c1_mm": float(lambda_c1[index]),
                    "clamped": bool(clamped[index]),
                    "stable_intersection": bool(stable[index]),
                    "c0_status": c0_reasons[index],
                    "c1_status": c1_reasons[index],
                    "c0_Xg_mm": None if c0_ground is None else float(c0_ground[0]),
                    "c0_Yg_mm": None if c0_ground is None else float(c0_ground[1]),
                    "c0_Zg_mm": None if c0_ground is None else float(c0_ground[2]),
                    "c1_Xg_mm": None if c1_ground is None else float(c1_ground[0]),
                    "c1_Yg_mm": None if c1_ground is None else float(c1_ground[1]),
                    "c1_Zg_mm": None if c1_ground is None else float(c1_ground[2]),
                }
            )

        for model, result_map_value in (("C0", c0_map), ("C1", c1_map)):
            for mode in MODE_NAMES:
                valid_ground: list[np.ndarray] = []
                valid_uv: list[np.ndarray] = []
                for uv in centers:
                    mapped = result_map_value.get(point_key(uv))
                    if mapped is not None:
                        valid_uv.append(uv)
                        valid_ground.append(mapped[1])
                ground_array = (
                    np.asarray(valid_ground, dtype=np.float64)
                    if valid_ground
                    else np.empty((0, 3), dtype=np.float64)
                )
                uv_array = (
                    np.asarray(valid_uv, dtype=np.float64)
                    if valid_uv
                    else np.empty((0, 2), dtype=np.float64)
                )
                try:
                    (
                        mean_mm,
                        median_mm,
                        std_mm,
                        height_count,
                        baseline_count,
                        measure_status,
                    ) = measure_mode(
                        ground_array, uv_array, entry["roi"], mode, app.measurement
                    )
                    error_text = ""
                except Exception as error:
                    mean_mm = median_mm = std_mm = None
                    height_count = baseline_count = 0
                    measure_status = "failed"
                    error_text = f"{type(error).__name__}: {error}"
                height_rows.append(
                    {
                        "dataset": entry["dataset"],
                        "height_truth_mm": entry["height_truth_mm"],
                        "pose_id": entry["pose_id"],
                        "position_rank": entry["position_rank"],
                        "v_center_px": entry["roi"]["v_center_px"],
                        "repeat_index": entry["repeat_index"],
                        "filename": entry["path"].name,
                        "model": model,
                        "mode": mode,
                        "height_mean_mm": mean_mm,
                        "height_median_mm": median_mm,
                        "height_std_mm": std_mm,
                        "signed_error_mm": (
                            None if mean_mm is None else mean_mm - entry["height_truth_mm"]
                        ),
                        "abs_error_mm": (
                            None if mean_mm is None else abs(mean_mm - entry["height_truth_mm"])
                        ),
                        "height_point_count": height_count,
                        "baseline_point_count": baseline_count,
                        "status": measure_status,
                        "error": error_text,
                        "steger_called_once": True,
                        "same_roi_c0_c1": True,
                    }
                )
    return frame_rows, height_rows, point_rows


def finite_numbers(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ],
        dtype=np.float64,
    )


def error_metrics(values: Iterable[Any], limit_mm: float = 0.2) -> dict[str, Any]:
    errors = finite_numbers(values)
    if len(errors) == 0:
        return {
            "count": 0,
            "bias_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_mm": None,
            "max_mm": None,
            "pass_count": 0,
            "pass_rate": None,
            "limit_mm": limit_mm,
        }
    absolute = np.abs(errors)
    return {
        "count": int(len(errors)),
        "bias_mm": float(np.mean(errors)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(errors * errors))),
        "p95_mm": float(np.quantile(absolute, 0.95)),
        "max_mm": float(np.max(absolute)),
        "pass_count": int(np.sum(absolute <= limit_mm)),
        "pass_rate": float(np.mean(absolute <= limit_mm)),
        "limit_mm": limit_mm,
    }


def condition_rows_from_frames(
    height_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in height_rows:
        if row["status"] == "success" and row["height_mean_mm"] is not None:
            groups[
                (row["dataset"], row["position_rank"], row["model"], row["mode"])
            ].append(row)
    conditions: list[dict[str, Any]] = []
    for key, rows in sorted(
        groups.items(),
        key=lambda item: (
            DATASET_ORDER[item[0][0]],
            int(item[0][1]),
            MODEL_NAMES.index(item[0][2]),
            MODE_NAMES.index(item[0][3]),
        ),
    ):
        dataset, position_rank, model, mode = key
        measurements = finite_numbers(row["height_mean_mm"] for row in rows)
        truth = float(rows[0]["height_truth_mm"])
        errors = measurements - truth
        conditions.append(
            {
                "dataset": dataset,
                "height_truth_mm": truth,
                "position_rank": int(position_rank),
                "model": model,
                "mode": mode,
                "repeat_count": len(measurements),
                "measured_mean_mm": float(np.mean(measurements)),
                "measured_median_mm": float(np.median(measurements)),
                "repeatability_sigma_mm": (
                    float(np.std(measurements, ddof=1))
                    if len(measurements) >= 2
                    else None
                ),
                "signed_error_mm": float(np.mean(measurements) - truth),
                "abs_error_mm": float(abs(np.mean(measurements) - truth)),
                "repeat_errors_json": json.dumps(errors.tolist()),
                "status": "success",
                "pass_0p2mm": bool(abs(np.mean(measurements) - truth) <= 0.2),
            }
        )
    return conditions


def missing_condition_rows(
    height_rows: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe expected height-position conditions with no successful aggregate.

    A failed ROI measurement is deliberately not converted into a fabricated
    numeric condition.  The diagnostic is persisted so the report can finish
    and point back to the frame-level failure instead of raising StopIteration.
    """
    expected = {
        (dataset, position_rank, model, mode)
        for dataset in DATASETS
        for position_rank in range(1, 6)
        for model in MODEL_NAMES
        for mode in MODE_NAMES
    }
    present = {
        (row["dataset"], row["position_rank"], row["model"], row["mode"])
        for row in conditions
    }
    missing: list[dict[str, Any]] = []
    for dataset, position_rank, model, mode in sorted(
        expected - present,
        key=lambda key: (
            DATASET_ORDER[key[0]],
            int(key[1]),
            MODEL_NAMES.index(key[2]),
            MODE_NAMES.index(key[3]),
        ),
    ):
        frame_failures = [
            row
            for row in height_rows
            if row["dataset"] == dataset
            and row["position_rank"] == position_rank
            and row["model"] == model
            and row["mode"] == mode
        ]
        reasons = sorted(
            {
                str(row.get("error") or row.get("status") or "unknown")
                for row in frame_failures
            }
        )
        missing.append(
            {
                "dataset": dataset,
                "position_rank": int(position_rank),
                "model": model,
                "mode": mode,
                "frame_count": len(frame_failures),
                "failure_reasons": reasons,
            }
        )
    return missing


def group_stats(
    conditions: list[dict[str, Any]],
    height_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for model in MODEL_NAMES:
        for mode in MODE_NAMES:
            frame_values = [
                row["signed_error_mm"]
                for row in height_rows
                if row["model"] == model
                and row["mode"] == mode
                and row["status"] == "success"
            ]
            frame_metrics = error_metrics(frame_values)
            rows.append(
                {
                    "layer": "single_frame",
                    "height": "",
                    "position_rank": "",
                    "model": model,
                    "mode": mode,
                    "expected_count": 150,
                    "failed_count": 150 - frame_metrics["count"],
                    **frame_metrics,
                }
            )
            summary[f"single_frame/{model}/{mode}"] = frame_metrics

            condition_values = [
                row["signed_error_mm"]
                for row in conditions
                if row["model"] == model and row["mode"] == mode
            ]
            global_metrics = error_metrics(condition_values)
            rows.append(
                {
                    "layer": "global",
                    "height": "",
                    "position_rank": "",
                    "model": model,
                    "mode": mode,
                    "expected_count": 30,
                    "failed_count": 30 - global_metrics["count"],
                    **global_metrics,
                }
            )
            summary[f"global/{model}/{mode}"] = global_metrics

            for dataset in DATASETS:
                values = [
                    row["signed_error_mm"]
                    for row in conditions
                    if row["dataset"] == dataset
                    and row["model"] == model
                    and row["mode"] == mode
                ]
                metrics = error_metrics(values)
                rows.append(
                    {
                        "layer": "per_height",
                        "height": dataset,
                        "position_rank": "",
                        "model": model,
                        "mode": mode,
                        "expected_count": 5,
                        "failed_count": 5 - metrics["count"],
                        **metrics,
                    }
                )
                summary[f"per_height/{dataset}/{model}/{mode}"] = metrics

            for dataset in DATASETS:
                for position_rank in range(1, 6):
                    values = [
                        row["signed_error_mm"]
                        for row in height_rows
                        if row["dataset"] == dataset
                        and row["position_rank"] == position_rank
                        and row["model"] == model
                        and row["mode"] == mode
                        and row["status"] == "success"
                    ]
                    metrics = error_metrics(values)
                    condition_key = f"{dataset}/position_{position_rank}"
                    row = {
                        "layer": "height_position",
                        "height": condition_key,
                        "position_rank": position_rank,
                        "model": model,
                        "mode": mode,
                        "expected_count": 5,
                        "failed_count": 5 - metrics["count"],
                        **metrics,
                    }
                    rows.append(row)
                    summary[
                        f"height_position/{condition_key}/{model}/{mode}"
                    ] = metrics
    return rows, summary


def position_bias_ranges(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for mode in MODE_NAMES:
            for dataset in DATASETS:
                values = np.asarray(
                    [
                        row["signed_error_mm"]
                        for row in conditions
                        if row["dataset"] == dataset
                        and row["model"] == model
                        and row["mode"] == mode
                    ],
                    dtype=np.float64,
                )
                values = values[np.isfinite(values)]
                output.append(
                    {
                        "dataset": dataset,
                        "truth_mm": TRUTH_MM[dataset],
                        "model": model,
                        "mode": mode,
                        "position_count": len(values),
                        "position_bias_min_mm": (
                            None if not len(values) else float(np.min(values))
                        ),
                        "position_bias_max_mm": (
                            None if not len(values) else float(np.max(values))
                        ),
                        "position_bias_range_mm": (
                            None
                            if not len(values)
                            else float(np.max(values) - np.min(values))
                        ),
                        "position_bias_mae_mm": (
                            None if not len(values) else float(np.mean(np.abs(values)))
                        ),
                        "pass_positions_0p2mm": (
                            int(np.sum(np.abs(values) <= 0.2)) if len(values) else 0
                        ),
                    }
                )
    return output


def repeatability_summary(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for mode in MODE_NAMES:
            sigmas = finite_numbers(
                row["repeatability_sigma_mm"]
                for row in conditions
                if row["model"] == model and row["mode"] == mode
            )
            output.append(
                {
                    "model": model,
                    "mode": mode,
                    "condition_count": len(sigmas),
                    "repeatability_median_sigma_mm": (
                        None if not len(sigmas) else float(np.median(sigmas))
                    ),
                    "repeatability_p95_sigma_mm": (
                        None if not len(sigmas) else float(np.quantile(sigmas, 0.95))
                    ),
                    "repeatability_max_sigma_mm": (
                        None if not len(sigmas) else float(np.max(sigmas))
                    ),
                }
            )
    return output


def adjacent_height_difference(
    conditions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["dataset"], row["position_rank"], row["model"], row["mode"]): row
        for row in conditions
    }
    detail: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for mode in MODE_NAMES:
            for position_rank in range(1, 6):
                for lower, upper in zip(DATASETS[:-1], DATASETS[1:]):
                    lower_row = lookup.get((lower, position_rank, model, mode))
                    upper_row = lookup.get((upper, position_rank, model, mode))
                    if lower_row is None or upper_row is None:
                        continue
                    observed = (
                        upper_row["measured_mean_mm"] - lower_row["measured_mean_mm"]
                    )
                    truth_difference = TRUTH_MM[upper] - TRUTH_MM[lower]
                    detail.append(
                        {
                            "model": model,
                            "mode": mode,
                            "position_rank": position_rank,
                            "lower_height": lower,
                            "upper_height": upper,
                            "truth_difference_mm": truth_difference,
                            "observed_difference_mm": observed,
                            "signed_error_mm": observed - truth_difference,
                            "abs_error_mm": abs(observed - truth_difference),
                        }
                    )
    summary: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for mode in MODE_NAMES:
            errors = [
                row["signed_error_mm"]
                for row in detail
                if row["model"] == model and row["mode"] == mode
            ]
            summary.append({"model": model, "mode": mode, **error_metrics(errors)})
    return detail, summary


def paired_comparison(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["position_rank"], row["mode"], row["model"]): row
        for row in conditions
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for position_rank in range(1, 6):
            for mode in MODE_NAMES:
                c0 = lookup.get((dataset, position_rank, mode, "C0"))
                c1 = lookup.get((dataset, position_rank, mode, "C1"))
                if c0 is None or c1 is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "truth_mm": TRUTH_MM[dataset],
                        "position_rank": position_rank,
                        "mode": mode,
                        "c0_measured_mm": c0["measured_mean_mm"],
                        "c1_measured_mm": c1["measured_mean_mm"],
                        "c0_error_mm": c0["signed_error_mm"],
                        "c1_error_mm": c1["signed_error_mm"],
                        "c1_minus_c0_mm": (
                            c1["measured_mean_mm"] - c0["measured_mean_mm"]
                        ),
                        "c0_pass_0p2mm": c0["pass_0p2mm"],
                        "c1_pass_0p2mm": c1["pass_0p2mm"],
                    }
                )
    return rows


def plot_outputs(
    output_dir: Path,
    conditions: list[dict[str, Any]],
    stats_summary: dict[str, Any],
    ranges: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    for axis, dataset in zip(axes.flat, DATASETS):
        for model, color in (("C0", "tab:blue"), ("C1", "tab:orange")):
            rows = sorted(
                [
                    row
                    for row in conditions
                    if row["dataset"] == dataset
                    and row["model"] == model
                    and row["mode"] == "local_adjacent"
                ],
                key=lambda row: row["position_rank"],
            )
            axis.plot(
                [row["position_rank"] for row in rows],
                [row["signed_error_mm"] for row in rows],
                "o-",
                label=model,
                color=color,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axhspan(-0.2, 0.2, color="tab:green", alpha=0.12)
        axis.set_title(dataset)
        axis.set_xlabel("position rank by v_center")
        axis.set_ylabel("condition error (mm)")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Local-adjacent condition bias by position")
    fig.tight_layout()
    fig.savefig(figures / "height_bias_by_position.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(DATASETS))
    width = 0.36
    for offset, model, color in (
        (-width / 2, "C0", "tab:blue"),
        (width / 2, "C1", "tab:orange"),
    ):
        values = [
            next(
                row["position_bias_range_mm"]
                for row in ranges
                if row["dataset"] == dataset
                and row["model"] == model
                and row["mode"] == "local_adjacent"
            )
            for dataset in DATASETS
        ]
        axis.bar(x + offset, values, width, label=model, color=color)
    axis.set_xticks(x, DATASETS)
    axis.set_ylabel("position bias range (mm)")
    axis.set_title("Position bias range by height, local-adjacent")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "position_bias_range.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    rows = [row for row in paired if row["mode"] == "local_adjacent"]
    axis.scatter(
        [row["c0_error_mm"] for row in rows],
        [row["c1_error_mm"] for row in rows],
        color="tab:orange",
        alpha=0.8,
    )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set_xlabel("C0 condition error (mm)")
    axis.set_ylabel("C1 condition error (mm)")
    axis.set_title("Paired C0/C1 condition errors, local-adjacent")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "c0_c1_paired_errors.png", dpi=160)
    plt.close(fig)

    metric_names = ("mae_mm", "rmse_mm", "p95_mm", "max_mm")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis, mode in zip(axes, MODE_NAMES):
        for model, color in (("C0", "tab:blue"), ("C1", "tab:orange")):
            values = [
                stats_summary[f"global/{model}/{mode}"][name] for name in metric_names
            ]
            axis.plot(metric_names, values, "o-", label=model, color=color)
        axis.axhline(0.2, color="tab:green", linestyle="--", linewidth=0.8)
        axis.set_title(mode)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("error metric (mm)")
    fig.suptitle("Global condition error metrics")
    fig.tight_layout()
    fig.savefig(figures / "global_error_metrics.png", dpi=160)
    plt.close(fig)


def status_payload(
    audit_summary: dict[str, Any],
    registry_summary: dict[str, Any],
    stats_summary: dict[str, Any],
    conditions: list[dict[str, Any]],
    height_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def status_for(model: str, mode: str) -> dict[str, Any]:
        frame = stats_summary[f"single_frame/{model}/{mode}"]
        condition = stats_summary[f"global/{model}/{mode}"]
        return {
            "single_frame": frame,
            "global_condition": condition,
            "all_single_frames_pass_0p2mm": bool(
                frame["count"] == 150 and frame["pass_count"] == frame["count"]
            ),
            "all_30_conditions_pass_0p2mm": bool(
                condition["count"] == 30 and condition["pass_count"] == condition["count"]
            ),
        }

    primary = status_for("C1", "local_adjacent")
    audit_ok = (
        audit_summary["image_count"] == 150
        and audit_summary["steger_call_count"] == 150
        and not audit_summary["duplicate_sha256_groups"]
    )
    all_success = len(
        [
            row
            for row in height_rows
            if row["model"] == "C1"
            and row["mode"] == "local_adjacent"
            and row["status"] == "success"
        ]
    ) == 150
    mechanically_pass = (
        audit_ok
        and all_success
        and primary["all_single_frames_pass_0p2mm"]
        and primary["all_30_conditions_pass_0p2mm"]
    )
    if not mechanically_pass:
        engineering_status = "FAIL"
    elif registry_summary["manual_review_required"]:
        engineering_status = "CONDITIONAL_REVIEW"
    else:
        engineering_status = "PASS"
    return {
        "SYSTEM_HEIGHT_ACCURACY": {
            "criterion": "absolute height error <= 0.2 mm",
            "primary_model": "C1",
            "primary_mode": "local_adjacent",
            "C0": {mode: status_for("C0", mode) for mode in MODE_NAMES},
            "C1": {mode: status_for("C1", mode) for mode in MODE_NAMES},
            "primary_result": primary,
        },
        "C1_ENGINEERING_STATUS": engineering_status,
        "C1_ENGINEERING_STATUS_REASON": (
            "accuracy and audit gates passed; ROI registry still needs human "
            "confirmation from generated overlays"
            if engineering_status == "CONDITIONAL_REVIEW"
            else "one or more audit or accuracy gates failed"
            if engineering_status == "FAIL"
            else "audit, shared-center, ROI, and primary accuracy gates passed"
        ),
        "audit_gate_pass": audit_ok,
        "registry_manual_review_required": registry_summary["manual_review_required"],
        "all_primary_height_rows_success": all_success,
        "condition_count": len(
            [
                row
                for row in conditions
                if row["model"] == "C1" and row["mode"] == "local_adjacent"
            ]
        ),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def render_report(
    path: Path,
    data_root: Path,
    config_path: Path,
    audit_summary: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    registry_summary: dict[str, Any],
    conditions: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    ranges: list[dict[str, Any]],
    repeatability: list[dict[str, Any]],
    adjacent_summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    status: dict[str, Any],
) -> None:
    primary = status["SYSTEM_HEIGHT_ACCURACY"]["primary_result"]
    roi_source = registry_summary.get("source", "auto_geometry")
    roi_description = (
        "本报告使用 30/30 人工确认后冻结的 v-axis ROI；自动候选仅作为边界差异对照保留。"
        if roi_source == "manual_frozen"
        else "ROI 由图像几何自动生成，尚未加载人工冻结 registry。"
    )
    lines: list[str] = [
        "# Daheng C1 量块工程精度验收报告",
        "",
        f"- 生成时间（UTC）：{now_utc()}",
        f"- 数据根目录：{data_root.resolve()}",
        f"- 生产配置（只读）：{config_path.resolve()}",
        "- 数据协议：6 高度 × 5 v 位置（laser001~005）× 5 重复",
        "- 真值：obs_1mm = 1.001 mm；其余为独立 truth 配置中的标称值",
        "",
        "## 结论",
        "",
        f"- SYSTEM_HEIGHT_ACCURACY：{primary['all_single_frames_pass_0p2mm']} "
        "（主判据：C1 + local_adjacent，150 个单帧）",
        f"- C1_ENGINEERING_STATUS：{status['C1_ENGINEERING_STATUS']}",
        f"- 状态说明：{status['C1_ENGINEERING_STATUS_REASON']}",
        "- 验收阈值：绝对高度误差不超过 0.2 mm。",
        "",
        "## Provenance 与审计",
        "",
        "- 复用：仓库现有 Steger backend、Daheng 0811 配置解析、quadratic C0 重建、Frozen C1 correction loader/evaluator。",
        "- 本轮新增：150 张 TIFF 的尺寸/dtype/offset/SHA/重复审计、150 次 Steger、30 个 geometry-only ROI、C0/C1 双分支重建、全部统计和图表。",
        "- 不复用：历史单帧/旧 PNG ROI、历史量块测量值、任何依据 C0/C1 结果调节的 ROI。",
        f"- 图像数：{audit_summary['image_count']}/150；Steger 调用：{audit_summary['steger_call_count']}/150；重复 SHA 组：{audit_summary['duplicate_sha256_group_count']}。",
        "- 每行 CSV 都记录 steger_called_once；height_measurements.csv 还记录 same_roi_c0_c1。",
        "",
        "## 输入组审计",
        "",
        "| 数据组 | 25 张 | 5 pose × 5 repeat | shape/dtype/offset | manifest quality |",
        "|---|---:|---:|---|---|",
    ]
    for dataset in DATASETS:
        rows = [
            row
            for row in audit_rows
            if row["dataset"] == dataset and row["filename"] != "__dataset_summary__"
        ]
        shape_ok = all(bool(row.get("shape_match")) for row in rows)
        dtype_ok = all(bool(row.get("dtype_match")) for row in rows)
        offsets = sorted(
            {(row.get("csv_offset_x"), row.get("csv_offset_y")) for row in rows}
        )
        manifest = manifest_summary(data_root / dataset / "dataset_manifest.yaml")
        frame_quality = sorted({str(row.get("quality")) for row in rows})
        frame_warnings = sorted(
            {
                str(row.get("quality_warnings"))
                for row in rows
                if str(row.get("quality_warnings") or "")
            }
        )
        quality = (
            f"frames quality={frame_quality}; warnings={frame_warnings}; "
            f"manifest={manifest.get('status')}"
        )
        lines.append(
            f"| {dataset} | {len(rows)}/25 | "
            f"{'yes' if len(rows) == 25 else 'no'} | "
            f"{'yes' if shape_ok and dtype_ok and offsets == [(0, 0)] else 'no'} "
            f"({offsets}) | {quality} |"
        )
    lines.extend(
        [
            "",
            "## Geometry-only ROI registry",
            "",
            roi_description,
            "ROI 没有读取或计算 C0/C1 高度来选取；每个位置使用固定 height window 与相邻两侧 baseline window。",
            "",
            "| 高度 | position | pose | v_center | height_v_range | baseline_before | baseline_after |",
            "|---|---:|---|---:|---|---|---|",
        ]
    )
    for item in registry:
        lines.append(
            f"| {item['dataset']} | {item['position_rank']} | {item['pose_id']} | "
            f"{fmt(item['v_center_px'], 1)} | {item['height_v_range']} | "
            f"{item['baseline_v_ranges'][0]} | {item['baseline_v_ranges'][1]} |"
        )
    lines.extend(
        [
            "",
            f"- ROI source：{roi_source}；人工确认标记：{'REQUIRED' if registry_summary['manual_review_required'] else 'complete'}。",
            "- 预览：figures/roi_registry_overview.png 与 overlays/ 下每个高度×pose 一张图。",
            "",
            "## 缺失条件诊断",
            "",
        ]
    )
    missing_conditions = status.get("missing_conditions", [])
    if missing_conditions:
        lines.extend(
            [
                "以下条件没有成功的数值汇总；报告保留为 MISSING，不用失败结果伪造精度数据。",
                "",
                "| 高度 | pos | model | mode | frame rows | failure reason |",
                "|---|---:|---|---|---:|---|",
            ]
        )
        for item in missing_conditions:
            reasons = "<br>".join(item.get("failure_reasons", [])) or "unknown"
            lines.append(
                f"| {item['dataset']} | {item['position_rank']} | {item['model']} | "
                f"{item['mode']} | {item['frame_count']} | {reasons} |"
            )
    else:
        lines.append("本轮没有缺失的 height × position × model × mode 条件。")
    lines.extend(
        [
            "",
            "## 30 个 height × position 条件（local_adjacent）",
            "",
            "| 高度 | pos | v_center | truth | C0 mean | C0 err | C0 pass | C1 mean | C1 err | C1 pass |",
            "|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|",
        ]
    )
    for dataset in DATASETS:
        for position in range(1, 6):
            c0 = next(
                (
                    row
                    for row in conditions
                    if row["dataset"] == dataset
                    and row["position_rank"] == position
                    and row["model"] == "C0"
                    and row["mode"] == "local_adjacent"
                ),
                None,
            )
            c1 = next(
                (
                    row
                    for row in conditions
                    if row["dataset"] == dataset
                    and row["position_rank"] == position
                    and row["model"] == "C1"
                    and row["mode"] == "local_adjacent"
                ),
                None,
            )
            v_center = next(
                (
                    row["v_center_px"]
                    for row in registry
                    if row["dataset"] == dataset
                    and row["position_rank"] == position
                ),
                None,
            )
            c0_mean = "MISSING" if c0 is None else fmt(c0["measured_mean_mm"])
            c0_error = "MISSING" if c0 is None else fmt(c0["signed_error_mm"])
            c0_pass = "MISSING" if c0 is None else (
                "PASS" if c0["pass_0p2mm"] else "FAIL"
            )
            c1_mean = "MISSING" if c1 is None else fmt(c1["measured_mean_mm"])
            c1_error = "MISSING" if c1 is None else fmt(c1["signed_error_mm"])
            c1_pass = "MISSING" if c1 is None else (
                "PASS" if c1["pass_0p2mm"] else "FAIL"
            )
            lines.append(
                f"| {dataset} | {position} | {fmt(v_center, 1)} | {fmt(TRUTH_MM[dataset], 3)} | "
                f"{c0_mean} | {c0_error} | {c0_pass} | "
                f"{c1_mean} | {c1_error} | {c1_pass} |"
            )
    lines.extend(
        [
            "",
            "## 六个高度的 position bias range（local_adjacent）",
            "",
            "| 高度 | truth | C0 range | C1 range | C0 position MAE | C1 position MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        c0 = next(
            row
            for row in ranges
            if row["dataset"] == dataset
            and row["model"] == "C0"
            and row["mode"] == "local_adjacent"
        )
        c1 = next(
            row
            for row in ranges
            if row["dataset"] == dataset
            and row["model"] == "C1"
            and row["mode"] == "local_adjacent"
        )
        lines.append(
            f"| {dataset} | {fmt(TRUTH_MM[dataset], 3)} | "
            f"{fmt(c0['position_bias_range_mm'])} | {fmt(c1['position_bias_range_mm'])} | "
            f"{fmt(c0['position_bias_mae_mm'])} | {fmt(c1['position_bias_mae_mm'])} |"
        )
    lines.extend(
        [
            "",
            "## 四层 Bias / MAE / RMSE / P95 / Max",
            "",
            "Bias 为 signed error 的均值；MAE、RMSE、P95、Max 均以绝对误差作为幅值口径。",
            "",
            "| layer | height/position | model | mode | n | failed | Bias | MAE | RMSE | P95 | Max | pass |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stats_rows:
        if row["layer"] not in {"single_frame", "height_position", "global"}:
            continue
        lines.append(
            f"| {row['layer']} | {row.get('height', '')} | {row['model']} | {row['mode']} | "
            f"{row['count']} | {row.get('failed_count', 0)} | "
            f"{fmt(row['bias_mm'])} | {fmt(row['mae_mm'])} | {fmt(row['rmse_mm'])} | "
            f"{fmt(row['p95_mm'])} | {fmt(row['max_mm'])} | "
            f"{row['pass_count']}/{row['count']} |"
        )
    lines.extend(
        [
            "",
            "## Repeatability",
            "",
            "repeatability 是每个 height × position 条件内五次重复高度均值的样本标准差（ddof=1）；不是单帧 height_std_mm。",
            "",
            "| model | mode | condition n | median sigma | P95 sigma | max sigma |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODEL_NAMES:
        for mode in MODE_NAMES:
            row = next(
                item
                for item in repeatability
                if item["model"] == model and item["mode"] == mode
            )
            lines.append(
                f"| {model} | {mode} | {row['condition_count']} | "
                f"{fmt(row['repeatability_median_sigma_mm'])} | "
                f"{fmt(row['repeatability_p95_sigma_mm'])} | "
                f"{fmt(row['repeatability_max_sigma_mm'])} |"
            )
    lines.extend(
        [
            "",
            "## Adjacent-height-difference MAE",
            "",
            "| model | mode | n | Bias | MAE | RMSE | P95 | Max |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in adjacent_summary:
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['count']} | {fmt(row['bias_mm'])} | "
            f"{fmt(row['mae_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['p95_mm'])} | "
            f"{fmt(row['max_mm'])} |"
        )
    lines.extend(
        [
            "",
            "## C0/C1 配对",
            "",
            f"配对 condition 行数：{len(paired)}；每个 mode 期望 30 行。",
            "两模型复用同一 Steger center array 与同一固定 ROI；详细结果见 paired_comparison.csv。",
            "",
            "## 输出文件",
            "",
            "- input_audit.csv / audit_summary.json",
            "- roi_registry.json / roi_candidates.csv / figures/roi_registry_overview.png / overlays/*.png",
            "- frame_metrics.csv / pointwise_diagnostics.csv",
            "- height_measurements.csv / condition_measurements.csv",
            "- stats_summary.csv / position_bias_ranges.csv / repeatability_summary.csv",
            "- adjacent_height_difference.csv / adjacent_height_difference_summary.csv / paired_comparison.csv",
            "- truth_config.json / provenance.json / acceptance_status.json / evaluation_summary.json",
            "- figures/*.png",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--roi-registry",
        type=Path,
        default=None,
        help="fully manual-confirmed roi_registry_manual.json; skips automatic ROI discovery",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    app = load_app_config(config_path)
    extraction_params = create_extraction_params(
        app.extraction_method,
        app.extraction_options_by_method.get(app.extraction_method, {}),
    )
    entries, audit_rows, audit_summary = audit_and_extract(
        data_root, extraction_params
    )
    if len(entries) != 150:
        raise RuntimeError(f"expected 150 input images, got {len(entries)}")

    if args.roi_registry is None:
        registry, candidate_rows, registry_summary = build_roi_registry(entries)
        roi_source_path = None
    else:
        roi_source_path = args.roi_registry.resolve()
        registry, registry_summary = load_frozen_roi_registry(
            roi_source_path, entries
        )
        candidate_rows = []
    write_json(
        output_dir / "roi_registry.json",
        {
            "protocol": {
                "roi_source": registry_summary.get("source", "auto_geometry"),
                "geometry_only": True,
                "c0_c1_values_used": False,
                "position_order": "actual v_center ascending",
            },
            "summary": registry_summary,
            "entries": registry,
        },
    )
    write_csv(
        output_dir / "roi_candidates.csv",
        candidate_rows,
        [
            "dataset",
            "pose_id",
            "candidate_rank_by_score",
            "v_center_px",
            "depth_px",
            "prominence_px",
            "support_width_px",
            "score",
            "selected",
        ],
    )
    save_registry_plot(output_dir / "figures" / "roi_registry_overview.png", registry)
    overlay_offset_x = int(app.camera.offset_x) if app.camera is not None else 0
    for item in registry:
        first = next(
            entry
            for entry in entries
            if entry["dataset"] == item["dataset"]
            and entry["pose_id"] == item["pose_id"]
            and entry["repeat_index"] == 1
        )
        save_overlay(
            output_dir
            / "overlays"
            / f"{item['dataset']}_pose{item['pose_id']}_position{item['position_rank']}.png",
            first["path"],
            item,
            overlay_offset_x,
            first["centers"],
        )

    correction_path = app.calibration.laser_ray_correction
    if correction_path is None:
        raise RuntimeError("config has no Frozen C1 correction path")
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_plane,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        laser_ray_correction=correction_path,
    )
    frame_rows, height_rows, point_rows = reconstruct_all(entries, calibration, app)
    conditions = condition_rows_from_frames(height_rows)
    missing_conditions = missing_condition_rows(height_rows, conditions)
    stats_rows, stats_summary = group_stats(conditions, height_rows)
    ranges = position_bias_ranges(conditions)
    repeatability = repeatability_summary(conditions)
    adjacent_rows, adjacent_summary = adjacent_height_difference(conditions)
    paired = paired_comparison(conditions)
    status = status_payload(
        audit_summary, registry_summary, stats_summary, conditions, height_rows
    )
    status.update(
        {
            "height_measurement_row_count": len(height_rows),
            "height_measurement_expected_count": 900,
            "height_measurement_failed_count": sum(
                row["status"] != "success" for row in height_rows
            ),
            "condition_row_count": len(conditions),
            "condition_expected_count": 180,
            "condition_missing_count": 180 - len(conditions),
            "missing_conditions": missing_conditions,
            "paired_condition_row_count": len(paired),
            "paired_condition_expected_count": 90,
            "paired_condition_missing_count": 90 - len(paired),
        }
    )

    write_csv(
        output_dir / "input_audit.csv",
        audit_rows,
        [
            "dataset",
            "filename",
            "pose_id",
            "repeat_index",
            "csv_row_present",
            "csv_sha256",
            "actual_sha256",
            "sha256_match",
            "csv_offset_x",
            "csv_offset_y",
            "csv_width",
            "csv_height",
            "csv_pixel_format",
            "actual_width",
            "actual_height",
            "actual_dtype",
            "shape_match",
            "dtype_match",
            "center_count",
            "steger_called_once",
            "quality",
            "quality_warnings",
            "dataset_error_count",
            "dataset_errors",
            "manifest_json",
        ],
    )
    write_json(output_dir / "audit_summary.json", audit_summary)
    write_json(
        output_dir / "truth_config.json",
        {
            "truth_mm": TRUTH_MM,
            "source": "task instruction; obs_1mm is the previously used 1.001 mm block",
            "protocol_date": "2026-08-19",
        },
    )
    write_csv(
        output_dir / "frame_metrics.csv",
        frame_rows,
        [
            "dataset",
            "truth_mm",
            "pose_id",
            "position_rank",
            "repeat_index",
            "filename",
            "sha256",
            "center_count",
            "c0_valid_points",
            "c1_valid_points",
            "c0_filtered_points",
            "c1_filtered_points",
            "common_valid_points",
            "c0_filter_reasons_json",
            "c1_filter_reasons_json",
            "clamp_points",
            "clamp_rate",
            "delta_lambda_mean_mm",
            "delta_lambda_median_mm",
            "delta_lambda_p95_abs_mm",
            "c0_model_type",
            "roi_geometry_only",
            "steger_called_once",
        ],
    )
    write_csv(
        output_dir / "pointwise_diagnostics.csv",
        point_rows,
        [
            "dataset",
            "truth_mm",
            "pose_id",
            "position_rank",
            "repeat_index",
            "filename",
            "point_index",
            "u_px",
            "v_px",
            "image_region",
            "lambda_c0_mm",
            "delta_lambda_mm",
            "lambda_c1_mm",
            "clamped",
            "stable_intersection",
            "c0_status",
            "c1_status",
            "c0_Xg_mm",
            "c0_Yg_mm",
            "c0_Zg_mm",
            "c1_Xg_mm",
            "c1_Yg_mm",
            "c1_Zg_mm",
        ],
    )
    write_csv(
        output_dir / "height_measurements.csv",
        height_rows,
        [
            "dataset",
            "height_truth_mm",
            "pose_id",
            "position_rank",
            "v_center_px",
            "repeat_index",
            "filename",
            "model",
            "mode",
            "height_mean_mm",
            "height_median_mm",
            "height_std_mm",
            "signed_error_mm",
            "abs_error_mm",
            "height_point_count",
            "baseline_point_count",
            "status",
            "error",
            "steger_called_once",
            "same_roi_c0_c1",
        ],
    )
    write_csv(
        output_dir / "condition_measurements.csv",
        conditions,
        [
            "dataset",
            "height_truth_mm",
            "position_rank",
            "model",
            "mode",
            "repeat_count",
            "measured_mean_mm",
            "measured_median_mm",
            "repeatability_sigma_mm",
            "signed_error_mm",
            "abs_error_mm",
            "repeat_errors_json",
            "status",
            "pass_0p2mm",
        ],
    )
    stats_fields = [
        "layer",
        "height",
        "position_rank",
        "model",
        "mode",
        "count",
        "expected_count",
        "failed_count",
        "bias_mm",
        "mae_mm",
        "rmse_mm",
        "p95_mm",
        "max_mm",
        "pass_count",
        "pass_rate",
        "limit_mm",
    ]
    write_csv(output_dir / "stats_summary.csv", stats_rows, stats_fields)
    write_csv(
        output_dir / "position_bias_ranges.csv",
        ranges,
        [
            "dataset",
            "truth_mm",
            "model",
            "mode",
            "position_count",
            "position_bias_min_mm",
            "position_bias_max_mm",
            "position_bias_range_mm",
            "position_bias_mae_mm",
            "pass_positions_0p2mm",
        ],
    )
    write_csv(
        output_dir / "repeatability_summary.csv",
        repeatability,
        [
            "model",
            "mode",
            "condition_count",
            "repeatability_median_sigma_mm",
            "repeatability_p95_sigma_mm",
            "repeatability_max_sigma_mm",
        ],
    )
    write_csv(
        output_dir / "adjacent_height_difference.csv",
        adjacent_rows,
        [
            "model",
            "mode",
            "position_rank",
            "lower_height",
            "upper_height",
            "truth_difference_mm",
            "observed_difference_mm",
            "signed_error_mm",
            "abs_error_mm",
        ],
    )
    write_csv(
        output_dir / "adjacent_height_difference_summary.csv",
        adjacent_summary,
        [
            "model",
            "mode",
            "count",
            "bias_mm",
            "mae_mm",
            "rmse_mm",
            "p95_mm",
            "max_mm",
            "pass_count",
            "pass_rate",
            "limit_mm",
        ],
    )
    write_csv(
        output_dir / "paired_comparison.csv",
        paired,
        [
            "dataset",
            "truth_mm",
            "position_rank",
            "mode",
            "c0_measured_mm",
            "c1_measured_mm",
            "c0_error_mm",
            "c1_error_mm",
            "c1_minus_c0_mm",
            "c0_pass_0p2mm",
            "c1_pass_0p2mm",
        ],
    )
    write_json(output_dir / "acceptance_status.json", status)
    write_json(
        output_dir / "evaluation_summary.json",
        {
            "protocol_date": "2026-08-19",
            "truth_mm": TRUTH_MM,
            "audit_summary": audit_summary,
            "roi_summary": registry_summary,
            "frame_metrics": frame_rows,
            "height_measurements": height_rows,
            "condition_measurements": conditions,
            "stats": stats_summary,
            "position_bias_ranges": ranges,
            "repeatability": repeatability,
            "adjacent_height_difference_summary": adjacent_summary,
            "paired_comparison": paired,
            "acceptance_status": status,
        },
    )

    provenance = {
        "generated_at_utc": now_utc(),
        "script": file_info(Path(__file__)),
        "data_root": str(data_root),
        "config": file_info(config_path),
        "roi_registry_input": file_info(roi_source_path),
        "calibration": {
            "intrinsics": file_info(app.calibration.intrinsics),
            "laser_model": file_info(app.calibration.laser_plane),
            "extrinsics": file_info(app.calibration.extrinsics),
            "laser_ray_correction_frozen_c1": file_info(correction_path),
        },
        "reconstruction_params_c0": params_dict(
            replace(app.reconstruction, enable_laser_ray_correction=False)
        ),
        "reconstruction_params_c1": params_dict(
            replace(app.reconstruction, enable_laser_ray_correction=True)
        ),
        "reused_artifacts": [
            "configured Steger backend and profile",
            "Daheng 0811 calibration files",
            "quadratic_graph C0 model",
            "Frozen C1_4k correction JSON",
            *(
                ["30/30 manually confirmed frozen ROI registry"]
                if roi_source_path is not None
                else []
            ),
        ],
        "newly_computed": [
            "input image audit",
            "one Steger extraction per 150 images",
            *(
                ["C0/C1 reconstruction against loaded frozen manual ROI"]
                if roi_source_path is not None
                else ["geometry-only automatic ROI registry and overlays"]
            ),
            "C0/C1 reconstruction and lambda diagnostics",
            "single-frame, condition, per-height, global statistics",
        ],
        "not_reused": [
            "historical single-frame PNG ROI/measurements",
            "historical gauge repeatability outputs",
            "any result-driven ROI or re-fit",
        ],
        "roi_summary": registry_summary,
    }
    write_json(output_dir / "provenance.json", provenance)
    plot_outputs(output_dir, conditions, stats_summary, ranges, paired)
    render_report(
        output_dir / "gauge_block_acceptance_report.md",
        data_root,
        config_path,
        audit_summary,
        audit_rows,
        registry,
        registry_summary,
        conditions,
        stats_rows,
        ranges,
        repeatability,
        adjacent_summary,
        paired,
        status,
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "SYSTEM_HEIGHT_ACCURACY": status["SYSTEM_HEIGHT_ACCURACY"],
                "C1_ENGINEERING_STATUS": status["C1_ENGINEERING_STATUS"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
