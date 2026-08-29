"""Ground-2R image-geometry audit and ground-only residual re-analysis.

The preview phase runs Steger exactly once per new TIFF, writes pose median
images/Steger overlays, and caches the resulting sub-pixel (u, v) centers.
After a human selects pose-level physical ground ranges from those images, the
analysis phase loads the cache, applies only those image-geometry ranges, and
reuses Ground-1's frozen origin/direction/bin edges for C0/C1 reconstruction
and residual-profile comparison.

No residual value is used by the image mask.  No C0/C1 refit, interpolation,
spline/LUT, height compensation, pose-local S redefinition, or cross-pose
point pooling is performed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config
from calibration.manifest import load_calibration_package
from laser.backends import create_extraction_params
from laser.laser_extractor import extract_laser_center
from reconstruction.reconstructor import reconstruct_uv_to_ground
from tools.fit_ground_pose_invariance import (
    EXPECTED_POSES,
    MIN_COVERAGE_FRACTION,
    _bin_specs,
    _classify,
    _compare_profiles,
    _consensus_profile_rows,
    _float,
    _ground1_profile_rows,
    _json_ready,
    _load_ground1,
    _moving_average,
    _natural_key,
    _pose_from_metadata,
    _read_csv,
    _sha256_file,
    _write_csv,
)
from tools.fit_ground_reference_20frames import (
    FrameFit,
    FrameRun,
    _build_frame_fits,
    _metric_values,
    _load_dataset_metadata,
)


DEFAULT_DATA_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\.chessboard_v2.inprogress\fit"
)
DEFAULT_GROUND1_DIR = TOOL_ROOT / "output_daheng_0811" / "ground_reference_20frames"
DEFAULT_OUTPUT_DIR = (
    TOOL_ROOT / "output_daheng_0811" / "ground_pose_invariance_ground_only"
)
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise RuntimeError(f"cannot read 2-D image: {path}")
    if image.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise RuntimeError(f"unsupported image dtype {image.dtype}: {path}")
    return image


def _preview_u8(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    low, high = np.percentile(values, [0.1, 99.9])
    if high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip((values - low) * 255.0 / (high - low), 0.0, 255.0).astype(
        np.uint8
    )


def _pose_paths(data_dir: Path) -> tuple[list[Path], dict[str, dict[str, Any]], dict[str, str]]:
    _, metadata_by_name, _, _ = _load_dataset_metadata(data_dir)
    paths = sorted(data_dir.glob("*.tif"), key=_natural_key)
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 TIFFs, found {len(paths)} in {data_dir}")
    pose_by_name: dict[str, str] = {}
    for path in paths:
        metadata = metadata_by_name.get(path.name)
        if metadata is None:
            raise RuntimeError(f"missing manifest metadata for {path.name}")
        pose_by_name[path.name] = _pose_from_metadata(metadata, path)
    counts = {pose: sum(value == pose for value in pose_by_name.values()) for pose in EXPECTED_POSES}
    if counts != {pose: 5 for pose in EXPECTED_POSES}:
        raise RuntimeError(f"expected five frames per pose, got {counts}")
    return paths, metadata_by_name, pose_by_name


def _save_geometry_overlay(
    path: Path,
    median_u8: np.ndarray,
    centers_by_frame: list[tuple[str, np.ndarray]],
    title: str,
) -> None:
    height, width = median_u8.shape
    fig, axis = plt.subplots(figsize=(16, 11))
    axis.imshow(median_u8, cmap="gray", origin="upper", vmin=0, vmax=255)
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(centers_by_frame)))
    for colour, (label, centers) in zip(colours, centers_by_frame, strict=True):
        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            s=1.8,
            linewidths=0.0,
            alpha=0.42,
            color=colour,
            label=label,
        )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("image u / column (px)")
    axis.set_ylabel("image v / row (px)")
    axis.legend(markerscale=4, loc="upper right", fontsize=8)
    axis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_geometry_trace(
    path: Path,
    centers_by_frame: list[tuple[str, np.ndarray]],
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(16, 9))
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(centers_by_frame)))
    for colour, (label, centers) in zip(colours, centers_by_frame, strict=True):
        order = np.argsort(centers[:, 0])
        axis.plot(
            centers[order, 0],
            centers[order, 1],
            ".",
            markersize=1.4,
            alpha=0.55,
            color=colour,
            label=label,
        )
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("image u / column (px)")
    axis.set_ylabel("image v / row (px)")
    axis.legend(markerscale=4, loc="best", fontsize=8)
    axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_preview(
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_document, metadata_by_name, manifest_path, frames_csv_path = _load_dataset_metadata(
        data_dir
    )
    paths, _, pose_by_name = _pose_paths(data_dir)
    app = load_app_config(config_path)
    if app.extraction_method != "steger":
        raise RuntimeError(f"expected steger extraction, got {app.extraction_method!r}")
    extraction_params = create_extraction_params(app.extraction_method, app.extraction_options)

    pose_images: dict[str, list[np.ndarray]] = {pose: [] for pose in EXPECTED_POSES}
    pose_centers: dict[str, list[tuple[str, np.ndarray]]] = {pose: [] for pose in EXPECTED_POSES}
    cache_records: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        image = _read_image(path)
        extraction_start = time.perf_counter()
        centers = np.ascontiguousarray(
            extract_laser_center(image, extraction_params, image_offset=(0, 0)),
            dtype=np.float64,
        )
        extraction_ms = (time.perf_counter() - extraction_start) * 1000.0
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
            raise RuntimeError(f"invalid Steger centers for {path}: {centers.shape}")
        pose_id = pose_by_name[path.name]
        frame_id = f"frame_{index:02d}"
        center_path = output_dir / f"{frame_id}_pose{pose_id}_centers.npy"
        np.save(center_path, centers)
        pose_images[pose_id].append(image)
        pose_centers[pose_id].append((frame_id, centers))
        metadata = metadata_by_name[path.name]
        cache_records.append(
            {
                "frame_id": frame_id,
                "pose_id": pose_id,
                "source_file": path.name,
                "source_path": str(path),
                "source_sha256": _sha256_file(path),
                "centers_path": str(center_path),
                "center_count": int(len(centers)),
                "image_shape": [int(image.shape[0]), int(image.shape[1])],
                "image_dtype": str(image.dtype),
                "camera_frame_number": metadata.get("camera_frame_number"),
                "quality": metadata.get("quality", {}),
                "extraction_ms": float(extraction_ms),
                "steger_run_count": 1,
            }
        )
        print(f"{frame_id} pose{pose_id} {path.name}: centers={len(centers)}")

    pose_summaries: dict[str, Any] = {}
    for pose_id in EXPECTED_POSES:
        median = np.median(np.stack(pose_images[pose_id], axis=0), axis=0)
        median = np.asarray(median, dtype=np.uint8)
        median_path = output_dir / f"pose{pose_id}_median_image.png"
        cv2.imwrite(str(median_path), _preview_u8(median))
        overlay_path = output_dir / f"pose{pose_id}_median_steger_overlay.png"
        _save_geometry_overlay(
            overlay_path,
            _preview_u8(median),
            pose_centers[pose_id],
            f"pose {pose_id}: median image + Steger centers (geometry only)",
        )
        trace_path = output_dir / f"pose{pose_id}_steger_geometry.png"
        _save_geometry_trace(
            trace_path,
            pose_centers[pose_id],
            f"pose {pose_id}: Steger image geometry (u, v)",
        )
        all_centers = np.concatenate([centers for _, centers in pose_centers[pose_id]])
        pose_summaries[pose_id] = {
            "frame_count": len(pose_centers[pose_id]),
            "median_image_path": str(median_path),
            "overlay_path": str(overlay_path),
            "trace_path": str(trace_path),
            "center_bbox_uv": {
                "u_min": float(np.min(all_centers[:, 0])),
                "u_max": float(np.max(all_centers[:, 0])),
                "v_min": float(np.min(all_centers[:, 1])),
                "v_max": float(np.max(all_centers[:, 1])),
            },
        }

    cache = {
        "schema_version": 1,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_dir": str(data_dir),
        "dataset_manifest": str(manifest_path),
        "frames_csv": str(frames_csv_path),
        "dataset_status": dataset_document.get("status"),
        "one_steger_per_frame": True,
        "frames": cache_records,
        "pose_summaries": pose_summaries,
        "image_preview_scaling": "per-pose median image, 0.1/99.9 percentiles mapped to uint8 PNG",
        "selection_rule": "human image geometry only; no residual values or residual thresholds",
    }
    (output_dir / "ground_pose_geometry_cache.json").write_text(
        json.dumps(_json_ready(cache), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"preview_output_dir={output_dir}")


def _load_geometry_cache(output_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_path = output_dir / "ground_pose_geometry_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    centers: dict[str, np.ndarray] = {}
    for record in cache["frames"]:
        path = Path(record["centers_path"])
        values = np.load(path)
        if values.ndim != 2 or values.shape[1] != 2:
            raise RuntimeError(f"invalid cached centers: {path}")
        centers[record["frame_id"]] = np.ascontiguousarray(values, dtype=np.float64)
    return cache, centers


def _load_ranges(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "poses" not in document:
        raise RuntimeError(f"ground-only range file must contain poses: {path}")
    poses = document["poses"]
    if not isinstance(poses, dict) or set(poses) != set(EXPECTED_POSES):
        raise RuntimeError(f"range file must define exactly poses {EXPECTED_POSES}: {path}")
    for pose_id in EXPECTED_POSES:
        spec = poses[pose_id]
        if not isinstance(spec, dict):
            raise RuntimeError(f"pose {pose_id} range must be a mapping")
        if spec.get("selection_basis") != "image_geometry_only":
            raise RuntimeError(f"pose {pose_id} selection_basis must be image_geometry_only")
        if spec.get("residual_used", False):
            raise RuntimeError(f"pose {pose_id} range claims residual-based selection")
        if spec.get("residual_threshold_used", False):
            raise RuntimeError(f"pose {pose_id} range uses a residual threshold")
        ranges = spec.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            raise RuntimeError(f"pose {pose_id} needs at least one image range")
        for item in ranges:
            if not isinstance(item, dict):
                raise RuntimeError(f"pose {pose_id} range item must be a mapping")
            axis = item.get("axis")
            if axis not in {"u", "v"}:
                raise RuntimeError(f"pose {pose_id} range axis must be u or v")
            lo = float(item["min"])
            hi = float(item["max"])
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                raise RuntimeError(f"invalid pose {pose_id} range {item}")
    return document


def _point_mask(centers_uv: np.ndarray, range_spec: dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(centers_uv), dtype=bool)
    for item in range_spec["ranges"]:
        coordinate = centers_uv[:, 0 if item["axis"] == "u" else 1]
        lo = float(item["min"])
        hi = float(item["max"])
        include_max = bool(item.get("include_max", False))
        if include_max:
            mask &= (coordinate >= lo) & (coordinate <= hi)
        else:
            mask &= (coordinate >= lo) & (coordinate < hi)
    return mask


def _build_frame_runs_from_cache(
    cache: dict[str, Any],
    centers_by_frame: dict[str, np.ndarray],
    data_dir: Path,
    ranges: dict[str, Any],
    config_path: Path,
) -> tuple[list[FrameRun], dict[str, str], dict[str, dict[str, Any]]]:
    dataset_document, metadata_by_name, _, _ = _load_dataset_metadata(data_dir)
    app = load_app_config(config_path)
    if not app.reconstruction.enable_laser_ray_correction:
        raise RuntimeError("enable_laser_ray_correction must be true")
    if app.reconstruction.image_roi_polygon is not None:
        raise RuntimeError("analytical image_roi_polygon must be null")
    package = load_calibration_package(app.calibration.manifest)
    params_c0 = replace(app.reconstruction, enable_laser_ray_correction=False)
    params_c1 = app.reconstruction
    frames: list[FrameRun] = []
    pose_by_frame: dict[str, str] = {}
    record_by_frame = {record["frame_id"]: record for record in cache["frames"]}
    for index, record in enumerate(cache["frames"], start=1):
        frame_id = record["frame_id"]
        pose_id = record["pose_id"]
        source_path = Path(record["source_path"])
        centers = centers_by_frame[frame_id]
        if _sha256_file(source_path) != record["source_sha256"]:
            raise RuntimeError(f"source image changed after preview: {source_path}")
        mask = _point_mask(centers, ranges["poses"][pose_id])
        ground_centers = np.ascontiguousarray(centers[mask], dtype=np.float64)
        if len(ground_centers) < 30:
            raise RuntimeError(f"too few geometry-selected centers in {frame_id}: {len(ground_centers)}")
        c0_start = time.perf_counter()
        c0_result = reconstruct_uv_to_ground(ground_centers, package.calibration, params_c0)
        c0_ms = (time.perf_counter() - c0_start) * 1000.0
        c1_start = time.perf_counter()
        c1_result = reconstruct_uv_to_ground(ground_centers, package.calibration, params_c1)
        c1_ms = (time.perf_counter() - c1_start) * 1000.0
        points_ground = np.ascontiguousarray(c1_result.points_ground, dtype=np.float64)
        if len(points_ground) == 0 or not np.isfinite(points_ground).all():
            raise RuntimeError(f"C1 has no finite ground-only points in {frame_id}")
        metadata = metadata_by_name[source_path.name]
        quality = metadata.get("quality", {})
        frames.append(
            FrameRun(
                index=index,
                frame_id=frame_id,
                path=source_path,
                camera_frame_number=(
                    int(metadata["camera_frame_number"])
                    if metadata.get("camera_frame_number") is not None
                    else None
                ),
                centers_uv=ground_centers,
                valid_pixels_uv=np.ascontiguousarray(c1_result.pixels_uv, dtype=np.float64),
                points_ground=points_ground,
                c0_point_count=int(c0_result.point_count),
                c1_point_count=int(c1_result.point_count),
                c0_filtered={str(key): int(value) for key, value in c0_result.filtered.items()},
                c1_filtered={str(key): int(value) for key, value in c1_result.filtered.items()},
                image_shape=tuple(int(value) for value in record["image_shape"]),
                image_dtype=str(record["image_dtype"]),
                file_sha256=str(record["source_sha256"]),
                quality={
                    "passed": quality.get("passed"),
                    "warnings": quality.get("warnings", []),
                    "dynamic_range_u8": quality.get("dynamic_range_u8"),
                    "dark_fraction": quality.get("dark_fraction"),
                    "laser_coverage": quality.get("laser_coverage"),
                },
                extraction_ms=float(record["extraction_ms"]),
                c0_reconstruction_ms=float(c0_ms),
                c1_reconstruction_ms=float(c1_ms),
            )
        )
        pose_by_frame[frame_id] = pose_id
        print(
            f"{frame_id} pose{pose_id}: image_ground_centers={len(ground_centers)} "
            f"C0={c0_result.point_count} C1={c1_result.point_count}"
        )
    return frames, pose_by_frame, metadata_by_name


def _ground_only_profile_rows(
    pose_id: str,
    fits: list[FrameFit],
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        per_frame_s: list[float] = []
        per_frame_residual: list[float] = []
        point_count = 0
        for fit in fits:
            left = spec["s_left_mm"]
            right = spec["s_right_mm"]
            in_bin = (fit.s >= left) & (
                fit.s <= right
                if spec["bin_index"] == specs[-1]["bin_index"]
                else fit.s < right
            )
            count = int(np.count_nonzero(in_bin))
            if count == 0:
                continue
            per_frame_s.append(float(np.median(fit.s[in_bin])))
            per_frame_residual.append(float(np.median(fit.residual[in_bin])))
            point_count += count
        rows.append(
            {
                "profile_group": f"pose{pose_id}",
                "pose_id": pose_id,
                "profile_source": f"5-frame-balanced ground-only residual profile for pose {pose_id}",
                "bin_index": spec["bin_index"],
                "s_left_mm": spec["s_left_mm"],
                "s_right_mm": spec["s_right_mm"],
                "s_center_mm": spec["s_center_mm"],
                "frame_count": len(per_frame_residual),
                "coverage_fraction": len(per_frame_residual) / len(fits),
                "point_count": point_count,
                "frame_balanced_s_median_mm": float(np.mean(per_frame_s)) if per_frame_s else None,
                "residual_mean_mm": float(np.mean(per_frame_residual)) if per_frame_residual else None,
                "residual_median_mm": float(np.median(per_frame_residual)) if per_frame_residual else None,
                "residual_std_mm": float(np.std(per_frame_residual)) if per_frame_residual else None,
                "profile_available": bool(per_frame_residual),
            }
        )
    return rows


def _plot_profiles(path: Path, profile_rows: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(12, 7))
    colours = {"ground1": "0.25", "pose002": "tab:blue", "pose003": "tab:orange", "pose004": "tab:green"}
    labels = {"ground1": "Ground-1 R_A(S)", "pose002": "pose 002 ground-only", "pose003": "pose 003 ground-only", "pose004": "pose 004 ground-only"}
    for group in ("ground1", "pose002", "pose003", "pose004"):
        rows = [row for row in profile_rows if row["profile_group"] == group and row["profile_available"]]
        rows.sort(key=lambda row: row["bin_index"])
        axis.plot(
            [row["s_center_mm"] for row in rows],
            [row["residual_median_mm"] for row in rows],
            marker=".",
            linewidth=1.2,
            markersize=4,
            color=colours[group],
            label=labels[group],
        )
    axis.axhline(0.0, color="0.3", linestyle=":", linewidth=0.8)
    axis.set_title("Ground-2R ground-only residual profiles on frozen Ground-1 S")
    axis.set_xlabel("Frozen S (mm)")
    axis.set_ylabel("Frame-balanced median residual (mm)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_comparison(path: Path, comparisons: list[dict[str, Any]], profile_rows: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(12, 7))
    ground1 = {int(row["bin_index"]): row for row in profile_rows if row["profile_group"] == "ground1"}
    for row in comparisons:
        if row["comparison_type"] != "ground1_vs_pose":
            continue
        target = {int(item["bin_index"]): item for item in profile_rows if item["profile_group"] == row["target_group"]}
        bins = sorted(set(ground1) & set(target))
        bins = [index for index in bins if ground1[index]["profile_available"] and target[index]["profile_available"]]
        axis.plot(
            [ground1[index]["s_center_mm"] for index in bins],
            [target[index]["residual_median_mm"] - ground1[index]["residual_median_mm"] for index in bins],
            marker=".",
            linewidth=1.1,
            markersize=4,
            label=f"{row['target_group']} - Ground-1",
        )
    axis.axhline(0.0, color="0.3", linestyle="--", linewidth=0.8)
    axis.set_title("Ground-2R ground-only residual difference on common frozen S bins")
    axis.set_xlabel("Frozen S (mm)")
    axis.set_ylabel("Target residual - Ground-1 residual (mm)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_ground_only_mask_overlay(
    path: Path,
    median_path: Path,
    centers_by_frame: list[tuple[str, np.ndarray]],
    range_spec: dict[str, Any],
    title: str,
) -> None:
    median_u8 = cv2.imread(str(median_path), cv2.IMREAD_GRAYSCALE)
    if median_u8 is None:
        raise RuntimeError(f"cannot read median preview: {median_path}")
    height, width = median_u8.shape
    fig, axis = plt.subplots(figsize=(16, 11))
    axis.imshow(median_u8, cmap="gray", origin="upper", vmin=0, vmax=255)
    kept_label_added = False
    excluded_label_added = False
    for _frame_id, centers in centers_by_frame:
        kept = _point_mask(centers, range_spec)
        if np.any(~kept):
            axis.scatter(
                centers[~kept, 0],
                centers[~kept, 1],
                s=2.2,
                color="red",
                alpha=0.55,
                linewidths=0.0,
                label="excluded by image geometry" if not excluded_label_added else None,
            )
            excluded_label_added = True
        if np.any(kept):
            axis.scatter(
                centers[kept, 0],
                centers[kept, 1],
                s=1.8,
                color="lime",
                alpha=0.32,
                linewidths=0.0,
                label="retained ground-only points" if not kept_label_added else None,
            )
            kept_label_added = True
    min_v = min(
        float(item["min"])
        for item in range_spec["ranges"]
        if item["axis"] == "v"
    )
    max_v = max(
        float(item["max"])
        for item in range_spec["ranges"]
        if item["axis"] == "v"
    )
    axis.axhspan(0.0, min_v, color="red", alpha=0.12, label="excluded v band")
    axis.axhline(min_v, color="red", linestyle="--", linewidth=1.4)
    axis.axhline(max_v, color="lime", linestyle=":", linewidth=1.0)
    axis.text(
        0.01,
        0.02,
        f"retained v=[{min_v:g}, {max_v:g}] px",
        transform=axis.transAxes,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("image u / column (px)")
    axis.set_ylabel("image v / row (px)")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _report_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6g}"


def _write_report(
    path: Path,
    summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    output_files: list[str],
) -> None:
    classification = summary["classification"]
    lines = [
        "# Ground-2R ground-only pose invariance report",
        "",
        f"## GROUND_POSE_INVARIANCE: {classification['GROUND_POSE_INVARIANCE']}",
        "",
        f"- CAMERA_SPACE_STRUCTURE: `{classification['CAMERA_SPACE_STRUCTURE']}`",
        f"- BOARD_DEPENDENT_STRUCTURE: `{classification['BOARD_DEPENDENT_STRUCTURE']}`",
        f"- MIXED_STRUCTURE: `{classification['MIXED_STRUCTURE']}`",
        f"- Ground-3 recommendation: `{classification['ground3_recommendation']}`",
        "",
        "## Geometry-only selection protocol",
        "",
        "- Ground-only ranges were selected once per pose from the pose median image and Steger overlay, then shared by all five frames in that pose.",
        "- Selection basis is image geometry only: outer board edge/frame/protrusion exclusion. No residual value, residual threshold, or residual ranking was used.",
        "- The preview cache records one Steger run per source frame; analysis reuses those cached centers and does not run Steger again.",
        "- The exact range/mask definition is in `ground_only_ranges.yaml`; the corresponding overlay/median images are listed below.",
        "",
        "## Frozen Ground-1 S definition",
        "",
        f"- origin_xy: `{summary['ground1_reuse']['frozen_origin_xy']}`",
        f"- direction_xy: `{summary['ground1_reuse']['frozen_direction_xy']}`",
        "- Formula: `S=(XY-origin_xy) dot direction_xy`.",
        f"- 50 frozen bins: [{summary['ground1_reuse']['s_min_mm']:.6g}, {summary['ground1_reuse']['s_max_mm']:.6g}] mm.",
        "- No new PCA, S origin/direction, C0/C1 fit, spline/LUT, height compensation, black-cell interpolation, or cross-pose point pooling.",
        "",
        "## Ground-1 versus ground-only pose profiles",
        "",
        "| pose | common bins | correlation | low-frequency correlation | profile RMSE difference (mm) | median abs difference (mm) | peak S offset (mm) | valley S offset (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["comparison_type"] != "ground1_vs_pose":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["target_group"],
                    str(row["common_bin_count"]),
                    _report_number(row["profile_correlation"]),
                    _report_number(row["low_frequency_correlation"]),
                    _report_number(row["profile_rmse_difference_mm"]),
                    _report_number(row["profile_median_abs_difference_mm"]),
                    _report_number(row["peak_s_offset_mm"]),
                    _report_number(row["valley_s_offset_mm"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Pose-level image geometry ranges", ""])
    for pose_id in EXPECTED_POSES:
        range_spec = summary["geometry_selection"]["ranges_by_pose"][pose_id]
        v_ranges = [item for item in range_spec["ranges"] if item["axis"] == "v"]
        for item in v_ranges:
            upper_bracket = "]" if item.get("include_max", False) else ")"
            lines.append(
                f"- pose{pose_id}: `v` in "
                f"[{float(item['min']):g}, {float(item['max']):g}{upper_bracket} px; "
                "shared by all five frames."
            )

    lines.extend(
        [
            "",
            "## Provenance and protocol caveat",
            "",
            f"- New frames: `{summary['new_data_audit']['pose_counts']}`; all retained.",
            f"- New dataset quality summary: `{summary['new_data_audit']['quality_summary']}`.",
            f"- Exposure by pose: `{summary['new_data_audit']['exposure_us_by_pose']}` µs; this remains a Ground-1/new-data protocol difference.",
            "- `enable_laser_ray_correction=true`; Ground-1 frozen C1 calibration is reused.",
            "- This is a diagnostic attribution result only; it does not authorize a new correction model.",
            "",
            "## Outputs",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in output_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_analysis(
    data_dir: Path,
    ground1_dir: Path,
    config_path: Path,
    output_dir: Path,
    ranges_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = _load_ranges(ranges_path)
    cache, centers_by_frame = _load_geometry_cache(output_dir)
    ground1_summary, ground1_source_rows, ground1_metric_rows = _load_ground1(ground1_dir)
    frozen_origin = np.asarray(ground1_summary["shared_s_definition"]["origin_xy"], dtype=np.float64)
    frozen_direction = np.asarray(ground1_summary["shared_s_definition"]["direction_xy"], dtype=np.float64)
    bin_specs = _bin_specs(ground1_source_rows)
    frames, pose_by_frame, metadata_by_name = _build_frame_runs_from_cache(
        cache,
        centers_by_frame,
        data_dir,
        ranges,
        config_path,
    )
    app = load_app_config(config_path)
    package = load_calibration_package(app.calibration.manifest)
    fits_by_pose: dict[str, list[FrameFit]] = {}
    for pose_id in EXPECTED_POSES:
        pose_frames = [frame for frame in frames if pose_by_frame[frame.frame_id] == pose_id]
        fits_by_pose[pose_id] = _build_frame_fits(
            pose_frames,
            frozen_origin,
            frozen_direction,
            app.measurement,
        )

    profile_rows = _ground1_profile_rows(ground1_source_rows)
    pose_metric_rows: list[dict[str, Any]] = []
    pose_profile_rows: dict[str, list[dict[str, Any]]] = {}
    for pose_id in EXPECTED_POSES:
        fits = fits_by_pose[pose_id]
        pose_profile_rows[pose_id] = _ground_only_profile_rows(pose_id, fits, bin_specs)
        for fit in fits:
            frame = fit.frame
            rmse, p95, max_abs = _metric_values(fit.residual)
            original_record = next(record for record in cache["frames"] if record["frame_id"] == frame.frame_id)
            range_spec = ranges["poses"][pose_id]
            selected_count = len(frame.centers_uv)
            pose_metric_rows.append(
                {
                    "frame_index": frame.index,
                    "frame_id": frame.frame_id,
                    "pose_id": pose_id,
                    "source_file": frame.path.name,
                    "camera_frame_number": frame.camera_frame_number,
                    "original_center_count": original_record["center_count"],
                    "ground_only_center_count": selected_count,
                    "c0_point_count": frame.c0_point_count,
                    "c1_point_count": frame.c1_point_count,
                    "point_count": len(frame.points_ground),
                    "filtered_c0_total": sum(frame.c0_filtered.values()),
                    "filtered_c1_total": sum(frame.c1_filtered.values()),
                    "a": fit.slope,
                    "b": fit.intercept,
                    "fit_rmse_mm": fit.fit_rmse,
                    "rmse_mm": rmse,
                    "p95_abs_mm": p95,
                    "max_abs_mm": max_abs,
                    "s_min_mm": float(np.min(fit.s)),
                    "s_max_mm": float(np.max(fit.s)),
                    "s_span_mm": float(np.ptp(fit.s)),
                    "image_geometry_mask": json.dumps(range_spec, ensure_ascii=False, sort_keys=True),
                    "quality_passed": frame.quality.get("passed"),
                    "quality_dynamic_range_u8": frame.quality.get("dynamic_range_u8"),
                    "quality_warnings": ";".join(map(str, frame.quality.get("warnings", []))),
                    "extraction_ms_reused": frame.extraction_ms,
                    "c0_reconstruction_ms": frame.c0_reconstruction_ms,
                    "c1_reconstruction_ms": frame.c1_reconstruction_ms,
                }
            )

    consensus_rows = _consensus_profile_rows(pose_profile_rows, bin_specs)
    profile_rows.extend(row for pose_id in EXPECTED_POSES for row in pose_profile_rows[pose_id])
    profile_rows.extend(consensus_rows)
    profile_rows_by_group = {
        group: [row for row in profile_rows if row["profile_group"] == group]
        for group in ("ground1", "pose002", "pose003", "pose004")
    }
    comparisons: list[dict[str, Any]] = []
    for pose_id in EXPECTED_POSES:
        comparisons.append(
            _compare_profiles(
                "ground1",
                f"pose{pose_id}",
                profile_rows_by_group["ground1"],
                profile_rows_by_group[f"pose{pose_id}"],
                ground1_metric_rows,
                [row for row in pose_metric_rows if row["pose_id"] == pose_id],
            )
        )
    for index, reference_pose in enumerate(EXPECTED_POSES):
        for target_pose in EXPECTED_POSES[index + 1 :]:
            comparisons.append(
                _compare_profiles(
                    f"pose{reference_pose}",
                    f"pose{target_pose}",
                    profile_rows_by_group[f"pose{reference_pose}"],
                    profile_rows_by_group[f"pose{target_pose}"],
                    [row for row in pose_metric_rows if row["pose_id"] == reference_pose],
                    [row for row in pose_metric_rows if row["pose_id"] == target_pose],
                )
            )
    classification = _classify(comparisons)

    mask_overlay_files: list[str] = []
    for pose_id in EXPECTED_POSES:
        pose_centers = [
            (record["frame_id"], centers_by_frame[record["frame_id"]])
            for record in cache["frames"]
            if record["pose_id"] == pose_id
        ]
        mask_path = output_dir / f"pose{pose_id}_ground_only_mask_overlay.png"
        _save_ground_only_mask_overlay(
            mask_path,
            output_dir / f"pose{pose_id}_median_image.png",
            pose_centers,
            ranges["poses"][pose_id],
            f"pose {pose_id}: Ground-2R image-geometry ground-only mask",
        )
        mask_overlay_files.append(mask_path.name)

    dataset_document, metadata_by_name, manifest_path, frames_csv_path = _load_dataset_metadata(data_dir)
    exposure_by_pose: dict[str, list[float]] = {pose: [] for pose in EXPECTED_POSES}
    for record in cache["frames"]:
        metadata = metadata_by_name[record["source_file"]]
        task_id = str(metadata.get("task_id", ""))
        for task in dataset_document.get("plan", {}).get("tasks", []):
            if isinstance(task, dict) and task.get("task_id") == task_id:
                camera = task.get("camera", {})
                if isinstance(camera, dict) and camera.get("exposure_us") is not None:
                    exposure_by_pose[record["pose_id"]].append(float(camera["exposure_us"]))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ground1_reuse": {
            "summary_path": str(ground1_dir / "ground_reference_summary.json"),
            "summary_sha256": _sha256_file(ground1_dir / "ground_reference_summary.json"),
            "profile_path": str(ground1_dir / "ground_profile_pooled.csv"),
            "profile_sha256": _sha256_file(ground1_dir / "ground_profile_pooled.csv"),
            "frozen_origin_xy": frozen_origin,
            "frozen_direction_xy": frozen_direction,
            "bin_count": len(bin_specs),
            "s_min_mm": bin_specs[0]["s_left_mm"],
            "s_max_mm": bin_specs[-1]["s_right_mm"],
            "ground1_profile_status": ground1_summary["conclusions"],
        },
        "new_data_audit": {
            "data_dir": str(data_dir),
            "dataset_manifest": str(manifest_path),
            "frames_csv": str(frames_csv_path),
            "dataset_status": dataset_document.get("status"),
            "quality_summary": dataset_document.get("quality_summary"),
            "pose_counts": {pose: sum(value == pose for value in pose_by_frame.values()) for pose in EXPECTED_POSES},
            "exposure_us_by_pose": {pose: sorted(set(values)) for pose, values in exposure_by_pose.items()},
        },
        "configuration": {
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "enable_laser_ray_correction": bool(app.reconstruction.enable_laser_ray_correction),
            "calibration_manifest": str(app.calibration.manifest),
            "calibration_package_id": package.package_id,
            "calibration_package_manifest_sha256": package.manifest_sha256,
            "analysis_code_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "geometry_selection": {
            "ranges_path": str(ranges_path),
            "ranges_sha256": _sha256_file(ranges_path),
            "cache_path": str(output_dir / "ground_pose_geometry_cache.json"),
            "cache_sha256": _sha256_file(output_dir / "ground_pose_geometry_cache.json"),
            "selection_basis": "image_geometry_only",
            "residual_used": False,
            "residual_threshold_used": False,
            "shared_within_pose": True,
            "ranges_by_pose": {pose: ranges["poses"][pose] for pose in EXPECTED_POSES},
            "mask_overlay_files": mask_overlay_files,
        },
        "protocol": {
            "one_steger_per_frame": True,
            "steger_centers_reused_from_preview_cache": True,
            "same_ground_only_centers_to_c0_and_c1": True,
            "all_c1_valid_points_retained_after_image_geometry_mask": True,
            "analytical_roi_used": False,
            "black_cell_interpolation": False,
            "new_xy_pca": False,
            "new_origin_or_direction": False,
            "cross_pose_point_pooling_before_fit": False,
            "spline_or_lut": False,
            "height_linear_compensation": False,
            "fixed_s_formula": "S=(XY-Ground1_origin_xy) dot Ground1_direction_xy",
            "profile_binning": "Ground-1 frozen 50-bin edges; no extrapolation outside common coverage",
        },
        "classification": classification,
        "comparison_rows": comparisons,
        "output_files": [
            "ground_only_ranges.yaml",
            "ground_pose_geometry_cache.json",
            "pose002_median_image.png",
            "pose002_median_steger_overlay.png",
            "pose002_steger_geometry.png",
            "pose002_ground_only_mask_overlay.png",
            "pose003_median_image.png",
            "pose003_median_steger_overlay.png",
            "pose003_steger_geometry.png",
            "pose003_ground_only_mask_overlay.png",
            "pose004_median_image.png",
            "pose004_median_steger_overlay.png",
            "pose004_steger_geometry.png",
            "pose004_ground_only_mask_overlay.png",
            "ground_pose_frame_metrics_ground_only.csv",
            "ground_pose_profiles_ground_only.csv",
            "ground_pose_comparison_ground_only.csv",
            "ground_pose_residual_overlay_ground_only.png",
            "ground_pose_pairwise_difference_ground_only.png",
            "ground_pose_invariance_ground_only_report.md",
            "ground_pose_invariance_ground_only_summary.json",
        ],
        "artifact_provenance": {
            "reused": [
                "Ground-1 frozen origin_xy/direction_xy and 50 S-bin edges",
                "Ground-1 pooled residual profile R_A(S)",
                "Ground-2 source images, pose grouping, and one-Steger geometry cache",
                "existing Daheng C1-enabled calibration/reconstruction and robust Zg=a*S+b kernel",
            ],
            "newly_computed": [
                "pose median images and Steger geometry overlays",
                "pose-level image-geometry ground-only masks/ranges",
                "ground-only C0/C1 reconstruction, frame fits, balanced profiles, comparisons, and report",
            ],
        },
    }

    _write_csv(output_dir / "ground_pose_frame_metrics_ground_only.csv", pose_metric_rows, list(pose_metric_rows[0]))
    _write_csv(output_dir / "ground_pose_profiles_ground_only.csv", profile_rows, list(profile_rows[0]))
    _write_csv(output_dir / "ground_pose_comparison_ground_only.csv", comparisons, list(comparisons[0]))
    _plot_profiles(output_dir / "ground_pose_residual_overlay_ground_only.png", profile_rows)
    _plot_comparison(
        output_dir / "ground_pose_pairwise_difference_ground_only.png",
        comparisons,
        profile_rows,
    )
    summary_path = output_dir / "ground_pose_invariance_ground_only_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "ground_pose_invariance_ground_only_report.md",
        _json_ready(summary),
        comparisons,
        summary["output_files"],
    )
    print(f"analysis_output_dir={output_dir}")
    print(f"GROUND_POSE_INVARIANCE={classification['GROUND_POSE_INVARIANCE']}")
    print(f"CAMERA_SPACE_STRUCTURE={classification['CAMERA_SPACE_STRUCTURE']}")
    print(f"BOARD_DEPENDENT_STRUCTURE={classification['BOARD_DEPENDENT_STRUCTURE']}")
    print(f"GROUND3={classification['ground3_recommendation']}")
    for row in comparisons:
        if row["comparison_type"] == "ground1_vs_pose":
            print(
                f"{row['target_group']}: n={row['common_bin_count']} "
                f"corr={row['profile_correlation']} "
                f"low_corr={row['low_frequency_correlation']} "
                f"rmse_diff={row['profile_rmse_difference_mm']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preview", "analyze"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ground1-dir", type=Path, default=DEFAULT_GROUND1_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ranges", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.phase == "preview":
        _run_preview(data_dir, args.config.resolve(), output_dir)
        return 0
    ranges_path = args.ranges.resolve() if args.ranges is not None else output_dir / "ground_only_ranges.yaml"
    _run_analysis(
        data_dir,
        args.ground1_dir.resolve(),
        args.config.resolve(),
        output_dir,
        ranges_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
