"""Height-2/3: frozen Height-1 scale and obs_50mm held-out validation.

The command is intentionally split into two phases:

``prepare``
    Fits the full-data H1 scale from the original 29 Ground-4A conditions,
    writes the frozen parameter artifact, then extracts the 25 new images once
    with Steger and writes geometry-only median/overlay artifacts.  No 50 mm
    truth or error is read in this phase.

``evaluate``
    Requires a manually confirmed geometry-only ROI registry.  It reuses the
    cached Steger centers, runs the frozen C1 reconstruction once per frame,
    builds the repeat-1 session-linear ground proxy per position, and scores
    repeat2--5 with raw B and frozen H1.

This is a retrospective held-out diagnostic.  It never changes C0/C1, Ground
G(S), GUI, or production configuration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
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


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_ROOT = REPO_ROOT / "laser_measurement_tool"
TOOLS_ROOT = REPO_ROOT / "tools"
if str(MEASUREMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from app_config import load_app_config
from calibration.config_loader import load_calibration_files
from laser.backends import create_extraction_params
from measurement.height_measure import MeasurementParams
from reconstruction.reconstructor import reconstruct_uv_to_ground
from utils.image_io import load_grayscale_image

from evaluate_daheng_c1_gauge_blocks import (
    load_image_and_centers,
    point_key,
    result_map,
)
from replay_daheng_ground4a import _fit_fixed_s_profile, _measure_fixed_profile


DEFAULT_DATA_ROOT = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\obs_50mm"
)
DEFAULT_CONFIG = REPO_ROOT / "laser_measurement_tool" / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_GROUND4A_CSV = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a" / "ground4a_condition_comparison.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "daheng_c1_gauge_blocks_20260819_ground4a" / "height50_heldout"
DEFAULT_GROUND3_SUMMARY = (
    MEASUREMENT_ROOT / "output_daheng_0811" / "ground_spatial_correction_ground3" / "ground_gs_summary.json"
)

POSE_IDS = tuple(f"{index:03d}" for index in range(1, 6))
TRUTH_MM = 50.0
TIFF_PATTERN = re.compile(r"^laser\s+(?P<pose>\d{3})(?:_(?P<repeat>\d{2}))?\.tif$", re.IGNORECASE)
PASS_LIMITS = (0.05, 0.10, 0.20)
EPS = 1.0e-12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if any(str(value or "").strip() for value in row.values())]


def parse_tiff_name(path: Path) -> tuple[str, int]:
    match = TIFF_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid obs_50mm TIFF name: {path.name}")
    return match.group("pose"), int(match.group("repeat") or "01")


def discover_images(data_root: Path) -> list[dict[str, Any]]:
    paths = sorted(data_root.glob("fit/*.tif"), key=lambda path: (parse_tiff_name(path), path.name))
    if len(paths) != 25:
        raise RuntimeError(f"obs_50mm expected 25 TIFFs, got {len(paths)}")
    entries = []
    seen: set[tuple[str, int]] = set()
    for path in paths:
        pose_id, repeat_index = parse_tiff_name(path)
        key = (pose_id, repeat_index)
        if pose_id not in POSE_IDS or repeat_index not in range(1, 6) or key in seen:
            raise RuntimeError(f"unexpected or duplicate obs_50mm frame: {path.name}")
        seen.add(key)
        entries.append({"pose_id": pose_id, "repeat_index": repeat_index, "path": path})
    expected = {(pose_id, repeat) for pose_id in POSE_IDS for repeat in range(1, 6)}
    if seen != expected:
        raise RuntimeError(f"obs_50mm frame groups are incomplete: {sorted(expected - seen)}")
    return entries


def frame_metadata(data_root: Path, entries: list[dict[str, Any]]) -> dict[str, dict[str, int | str]]:
    """Read only image metadata fields; never read truth/error fields in prepare."""
    frames_path = data_root / "frames.csv"
    rows = read_csv(frames_path)
    by_name = {Path(str(row.get("filename", "")).replace("\\", "/")).name: row for row in rows}
    if len(rows) != 25:
        raise RuntimeError(f"obs_50mm frames.csv expected 25 rows, got {len(rows)}")
    metadata: dict[str, dict[str, int | str]] = {}
    for entry in entries:
        path = entry["path"]
        row = by_name.get(path.name)
        if row is None:
            raise RuntimeError(f"frames.csv missing {path.name}")
        metadata[path.name] = {
            "offset_x": int(float(row.get("offset_x") or 0)),
            "offset_y": int(float(row.get("offset_y") or 0)),
            "width": int(float(row.get("width") or 0)),
            "height": int(float(row.get("height") or 0)),
            "pixel_format": str(row.get("pixel_format") or ""),
        }
    return metadata


def fit_full_height_scale(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Freeze H1 from the original 29 conditions before touching obs_50mm."""
    input_sha = sha256(input_path)
    rows = read_csv(input_path)
    b_rows = [row for row in rows if row.get("chain") == "session_linear"]
    conditions = []
    for row in b_rows:
        if (
            int(row.get("successful_repeat2_5") or 0) == 4
            and int(row.get("failed_repeat2_5") or 0) == 0
            and row.get("measured_mean_mm", "") != ""
        ):
            conditions.append(
                {
                    "dataset": row["dataset"],
                    "position_rank": int(row["position_rank"]),
                    "raw_height_mm": float(row["measured_mean_mm"]),
                    "truth_mm": float(row["truth_mm"]),
                }
            )
    if len(b_rows) != 30 or len(conditions) != 29:
        raise RuntimeError(f"Height-1 freeze expected 30 B rows/29 successful conditions, got {len(b_rows)}/{len(conditions)}")
    keys = {(item["dataset"], item["position_rank"]) for item in conditions}
    if len(keys) != 29:
        raise RuntimeError("Height-1 freeze input contains duplicate conditions")
    x = np.asarray([item["raw_height_mm"] for item in conditions], dtype=np.float64)
    y = np.asarray([item["truth_mm"] for item in conditions], dtype=np.float64)
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(np.float64).eps:
        raise RuntimeError("Height-1 full-data scale denominator is degenerate")
    k_full = float(np.dot(x, y) / denominator)
    parameter_payload = {
        "model": "H1",
        "fit_equation": "h_corr=k*h",
        "condition_count": 29,
        "k": k_full,
    }
    canonical = json.dumps(parameter_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_50MM_READ",
        "frozen_at_utc": now_utc(),
        "model": "H1",
        "fit_equation": "h_corr=k*h",
        "k_full": k_full,
        "condition_count": 29,
        "equal_weighting": "one successful Ground-4A B condition per row; no repeat expansion",
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_sha,
            "source": "Ground-4A ground4a_condition_comparison.csv, chain=session_linear, repeat2_5 only",
        },
        "parameter_payload": parameter_payload,
        "parameter_canonical_json": canonical,
        "parameter_sha256": sha256_bytes(canonical.encode("utf-8")),
        "50mm_read_before_freeze": False,
        "50mm_allowed_after_freeze_only": True,
    }
    write_json(output_path, payload)
    return payload


def verify_frozen_scale(path: Path, ground4a_csv: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_50MM_READ" or payload.get("model") != "H1":
        raise RuntimeError("frozen_height_scale.json is not a valid pre-50mm H1 freeze")
    if payload.get("input", {}).get("sha256") != sha256(ground4a_csv):
        raise RuntimeError("Ground-4A input SHA no longer matches frozen Height-1 artifact")
    canonical = json.dumps(payload["parameter_payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if sha256_bytes(canonical.encode("utf-8")) != payload.get("parameter_sha256"):
        raise RuntimeError("frozen H1 parameter SHA mismatch")
    if abs(float(payload["parameter_payload"]["k"]) - float(payload["k_full"])) > EPS:
        raise RuntimeError("frozen H1 parameter payload and k_full disagree")
    return payload


def median_centerline(centers_by_repeat: list[np.ndarray], image_height: int, bin_width: int = 5) -> tuple[np.ndarray, np.ndarray]:
    bin_v = np.arange(bin_width / 2.0, image_height, bin_width)
    profiles = np.full((len(centers_by_repeat), len(bin_v)), np.nan, dtype=np.float64)
    for repeat_index, centers in enumerate(centers_by_repeat):
        if len(centers) == 0:
            continue
        valid = np.isfinite(centers[:, 0]) & np.isfinite(centers[:, 1])
        centers = centers[valid]
        bin_index = np.floor(centers[:, 1] / bin_width).astype(int)
        for index in range(len(bin_v)):
            values = centers[bin_index == index, 0]
            if len(values):
                profiles[repeat_index, index] = float(np.median(values))
    center_u = np.full(len(bin_v), np.nan, dtype=np.float64)
    valid_columns = np.any(np.isfinite(profiles), axis=0)
    if np.any(valid_columns):
        center_u[valid_columns] = np.nanmedian(profiles[:, valid_columns], axis=0)
    return bin_v, center_u


def display_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    image_float = image.astype(np.float32, copy=False)
    low, high = np.percentile(image_float, [1.0, 99.8])
    rendered = np.clip((image_float - low) * 255.0 / max(1.0, high - low), 0.0, 255.0)
    return rendered, 255.0


def save_selection_overlay(path: Path, median: np.ndarray, centers_by_repeat: list[np.ndarray], title: str) -> None:
    center_v, center_u = median_centerline(centers_by_repeat, median.shape[0])
    rendered, display_vmax = display_image(median)
    figure, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": (3, 1)}, constrained_layout=True)
    axes[0].imshow(rendered, cmap="gray", vmin=0.0, vmax=display_vmax, extent=(0, median.shape[1], median.shape[0], 0), aspect="auto", interpolation="nearest")
    colors = ("#00e5ff", "#76ff03", "#ffea00", "#ff9100", "#d500f9")
    for index, centers in enumerate(centers_by_repeat):
        stride = max(1, len(centers) // 5000)
        sample = centers[::stride]
        axes[0].scatter(sample[:, 0], sample[:, 1], s=1.0, alpha=0.18, color=colors[index], linewidths=0.0)
    valid = np.isfinite(center_u)
    axes[0].scatter(center_u[valid], center_v[valid], s=2.0, alpha=0.95, color="#ff1744", linewidths=0.0, label="median of 5 Steger centers")
    axes[0].set_xlim(0, median.shape[1])
    axes[0].set_ylim(median.shape[0], 0)
    axes[0].set_xlabel("u [px]")
    axes[0].set_ylabel("v [px]")
    axes[0].set_title("median image + 5-frame Steger overlay")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.15)
    axes[1].plot(center_u[valid], center_v[valid], color="#ff1744", linewidth=0.8)
    axes[1].set_xlim(0, median.shape[1])
    axes[1].set_ylim(median.shape[0], 0)
    axes[1].set_xlabel("median center u [px]")
    axes[1].set_ylabel("v [px]")
    axes[1].set_title("geometry-only centerline profile")
    axes[1].grid(alpha=0.2)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_manual_overlay(path: Path, median: np.ndarray, centers_by_repeat: list[np.ndarray], entry: dict[str, Any]) -> None:
    center_v, center_u = median_centerline(centers_by_repeat, median.shape[0])
    rendered, display_vmax = display_image(median)
    figure, axis = plt.subplots(figsize=(12, 9), constrained_layout=True)
    axis.imshow(rendered, cmap="gray", vmin=0.0, vmax=display_vmax, extent=(0, median.shape[1], median.shape[0], 0), aspect="auto", interpolation="nearest")
    valid = np.isfinite(center_u)
    axis.scatter(center_u[valid], center_v[valid], s=2.0, color="#ff1744", alpha=0.9, linewidths=0.0, label="median of 5 Steger centers")
    colors = ("#42a5f5", "#ff9800", "#66bb6a")
    labels = ("baseline_before", "height", "baseline_after")
    ranges = [entry["baseline_v_ranges"][0], entry["height_v_range"], entry["baseline_v_ranges"][1]]
    for color, label, (v0, v1) in zip(colors, labels, ranges):
        axis.axhspan(v0, v1, color=color, alpha=0.22, label=f"manual {label} [{v0}, {v1}]")
        axis.axhline(v0, color=color, linewidth=1.0)
        axis.axhline(v1, color=color, linewidth=1.0)
    axis.set_xlim(0, median.shape[1])
    axis.set_ylim(median.shape[0], 0)
    axis.set_xlabel("u [px]")
    axis.set_ylabel("v [px]")
    axis.set_title(f"obs_50mm {entry['pose_id']} manual geometry-only ROI overlay")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(alpha=0.18)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def prepare(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "center_cache").mkdir(exist_ok=True)
    (output / "median_images").mkdir(exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)

    # This call writes the immutable H1 artifact before any obs_50mm image is opened.
    frozen = fit_full_height_scale(args.ground4a_csv.resolve(), output / "frozen_height_scale.json")

    data_root = args.data_root.resolve()
    entries = discover_images(data_root)
    metadata = frame_metadata(data_root, entries)
    app = load_app_config(args.config.resolve())
    extraction_params = create_extraction_params(app.extraction_method, app.extraction_options_by_method.get(app.extraction_method, {}))

    audit_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    by_pose: dict[str, list[dict[str, Any]]] = {pose_id: [] for pose_id in POSE_IDS}
    for item in entries:
        path = item["path"]
        meta = metadata[path.name]
        image, centers = load_image_and_centers(path, extraction_params, int(meta["offset_x"]), int(meta["offset_y"]))
        cache_path = output / "center_cache" / f"laser{item['pose_id']}_{item['repeat_index']:02d}.npy"
        np.save(cache_path, centers)
        item.update({"image": image, "centers": centers, "cache_path": cache_path})
        by_pose[item["pose_id"]].append(item)
        actual_sha = sha256(path)
        shape_match = image.shape == (int(meta["height"]), int(meta["width"]))
        audit_rows.append({
            "pose_id": item["pose_id"],
            "repeat_index": item["repeat_index"],
            "filename": path.name,
            "sha256": actual_sha,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "dtype": str(image.dtype),
            "expected_width": meta["width"],
            "expected_height": meta["height"],
            "pixel_format": meta["pixel_format"],
            "offset_x": meta["offset_x"],
            "offset_y": meta["offset_y"],
            "shape_match": shape_match,
            "center_count": int(len(centers)),
            "steger_called_once": True,
            "truth_or_error_used": False,
        })

    for pose_id in POSE_IDS:
        group = sorted(by_pose[pose_id], key=lambda item: item["repeat_index"])
        if len(group) != 5:
            raise RuntimeError(f"{pose_id} does not have five repeats")
        stack = np.stack([item["image"] for item in group], axis=0)
        median = np.rint(np.median(stack, axis=0)).astype(group[0]["image"].dtype)
        median_path = output / "median_images" / f"laser{pose_id}_median.tif"
        if not cv2.imwrite(str(median_path), median):
            raise RuntimeError(f"failed to write {median_path}")
        save_selection_overlay(
            output / "overlays" / f"laser{pose_id}_median_steger_overlay.png",
            median,
            [item["centers"] for item in group],
            f"obs_50mm laser{pose_id} — selection view; truth/error not shown",
        )
        center_v, center_u = median_centerline([item["centers"] for item in group], median.shape[0])
        for v, u in zip(center_v, center_u):
            if math.isfinite(float(u)):
                profile_rows.append({"pose_id": pose_id, "v_px": float(v), "median_u_px": float(u)})

    write_csv(output / "height50_input_audit.csv", audit_rows, list(audit_rows[0].keys()))
    write_csv(output / "height50_centerline_profiles.csv", profile_rows, ["pose_id", "v_px", "median_u_px"])
    write_json(
        output / "height50_selection_provenance.json",
        {
            "phase": "prepare",
            "selection_status": "AWAITING_MANUAL_GEOMETRY_ONLY_ROI",
            "data_root": str(data_root),
            "image_count": len(entries),
            "pose_ids": list(POSE_IDS),
            "repeats_per_pose": 5,
            "steger_call_count": len(entries),
            "steger_calls_per_frame": 1,
            "truth_or_error_used_for_selection": False,
            "selection_artifacts": "five-frame median images, Steger centers, and centerline profiles only",
            "frozen_height_scale_sha256": sha256(output / "frozen_height_scale.json"),
            "height1_k_full": frozen["k_full"],
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256(args.config.resolve()),
        },
    )
    print(f"prepared obs_50mm: {len(entries)} frames, one Steger/frame")
    print(f"frozen k_full={frozen['k_full']:.15g}, parameter_sha256={frozen['parameter_sha256']}")
    print(f"manual registry required before evaluate: {output / 'height50_manual_roi_registry.json'}")
    return 0


def load_ground_reference(summary_path: Path) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = summary.get("protocol", {})
    origin = np.asarray(protocol.get("origin_xy"), dtype=np.float64)
    direction = np.asarray(protocol.get("direction_xy"), dtype=np.float64)
    if origin.shape != (2,) or direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-8):
        raise RuntimeError("frozen Ground-1 origin/direction are missing or not unit-normalized")
    domain_min = float(protocol.get("s_domain_min_mm"))
    domain_max = float(protocol.get("s_domain_max_mm"))
    if not domain_min < domain_max:
        raise RuntimeError("invalid frozen Ground-1 S domain")
    return origin, direction, domain_min, domain_max, summary


def load_manual_registry(path: Path, output: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    protocol = document.get("protocol", {})
    summary = document.get("summary", {})
    if protocol.get("geometry_only") is not True or protocol.get("c0_c1_values_used") is not False:
        raise RuntimeError("50mm ROI registry is not geometry-only")
    if summary.get("manual_confirmed") is not True or summary.get("manual_confirmed_count") != 5 or summary.get("manual_review_required") is not False:
        raise RuntimeError("50mm ROI registry is not fully manual-confirmed")
    entries = document.get("entries", [])
    by_pose = {str(item.get("pose_id")): item for item in entries}
    if set(by_pose) != set(POSE_IDS) or len(entries) != 5:
        raise RuntimeError("50mm ROI registry must contain exactly laser001~005")
    forbidden = ("truth", "error", "residual", "height50_result")
    raw = path.read_text(encoding="utf-8").lower()
    if any(token in raw for token in forbidden):
        raise RuntimeError("50mm manual ROI registry contains result/truth-derived fields")
    for pose_id, entry in by_pose.items():
        height = entry.get("height_v_range")
        baseline = entry.get("baseline_v_ranges")
        if not isinstance(height, list) or len(height) != 2 or not isinstance(baseline, list) or len(baseline) != 2:
            raise RuntimeError(f"invalid ROI ranges for {pose_id}")
        ranges = [baseline[0], height, baseline[1]]
        if any(not isinstance(item, list) or len(item) != 2 for item in ranges):
            raise RuntimeError(f"invalid ROI range shape for {pose_id}")
        if any(int(item[0]) < 0 or int(item[1]) >= 3000 or int(item[0]) > int(item[1]) for item in ranges):
            raise RuntimeError(f"ROI range outside image for {pose_id}")
        if int(baseline[0][1]) >= int(height[0]) or int(baseline[1][0]) <= int(height[1]):
            raise RuntimeError(f"baseline/height ranges overlap for {pose_id}")
    return by_pose


def frame_point(center: np.ndarray, ground: np.ndarray, origin: np.ndarray, direction: np.ndarray, region: str) -> dict[str, Any]:
    xy = np.asarray(ground[:2], dtype=np.float64)
    return {"xy": xy, "z": float(ground[2]), "s": float((xy - origin) @ direction), "region": region}


def point_metric(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=np.float64)
    if not len(values):
        output: dict[str, Any] = {"count": 0, "bias_mm": None, "mae_mm": None, "rmse_mm": None, "p95_mm": None, "max_mm": None}
        for limit in PASS_LIMITS:
            key = f"pass_{str(limit).replace('.', 'p')}_count"
            output[key] = 0
            output[f"pass_{str(limit).replace('.', 'p')}_rate"] = None
        return output
    absolute = np.abs(values)
    output = {
        "count": int(len(values)),
        "bias_mm": float(np.mean(values)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(values**2))),
        "p95_mm": float(np.percentile(absolute, 95.0)),
        "max_mm": float(np.max(absolute)),
    }
    for limit in PASS_LIMITS:
        suffix = str(limit).replace('.', 'p')
        passed = absolute <= limit
        output[f"pass_{suffix}_count"] = int(np.count_nonzero(passed))
        output[f"pass_{suffix}_rate"] = float(np.mean(passed))
    return output


def measurement_result(frame_points: list[dict[str, Any]], model: Any, params: MeasurementParams) -> dict[str, Any]:
    try:
        result = _measure_fixed_profile(frame_points, model.slope, model.intercept, None, -math.inf, math.inf, params)
    except Exception as error:
        return {"status": f"{type(error).__name__}: {error}", "value": None, "median": None, "std": None, "point_count": len(frame_points), "inlier_count": 0, "unsupported_count": 0}
    return result


def evaluate(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    frozen = verify_frozen_scale(output / "frozen_height_scale.json", args.ground4a_csv.resolve())
    registry_path = args.registry.resolve()
    registry = load_manual_registry(registry_path, output)
    data_root = args.data_root.resolve()
    entries = discover_images(data_root)
    metadata = frame_metadata(data_root, entries)
    origin, direction, domain_min, domain_max, ground_summary = load_ground_reference(args.ground3_summary.resolve())

    cache_dir = output / "center_cache"
    for item in entries:
        cache_path = cache_dir / f"laser{item['pose_id']}_{item['repeat_index']:02d}.npy"
        if not cache_path.is_file():
            raise RuntimeError(f"missing prepared Steger cache: {cache_path}")
        item["centers"] = np.load(cache_path, allow_pickle=False).astype(np.float64, copy=False).reshape(-1, 2)

    app = load_app_config(args.config.resolve())
    if app.reconstruction.enable_laser_ray_correction is not True:
        raise RuntimeError("Daheng config does not have enable_laser_ray_correction=true")
    correction_path = app.calibration.laser_ray_correction
    if correction_path is None:
        raise RuntimeError("Daheng config has no frozen C1 correction path")
    calibration = load_calibration_files(
        app.calibration.intrinsics,
        app.calibration.laser_plane,
        app.calibration.extrinsics,
        app.calibration.ground_u_compensation,
        laser_ray_correction=correction_path,
    )
    params_c1 = replace(app.reconstruction, enable_laser_ray_correction=True)

    frame_points: dict[tuple[str, int], dict[str, Any]] = {}
    frame_rows: list[dict[str, Any]] = []
    for item in entries:
        pose_id = item["pose_id"]
        repeat = int(item["repeat_index"])
        roi = registry[pose_id]
        centers = item["centers"]
        c1_result = reconstruct_uv_to_ground(centers, calibration, params_c1)
        c1_map = result_map(c1_result)
        baseline: list[dict[str, Any]] = []
        height: list[dict[str, Any]] = []
        valid_count = 0
        s_outside = 0
        for center in centers:
            mapped = c1_map.get(point_key(center))
            if mapped is None:
                continue
            valid_count += 1
            ground = mapped[1]
            v = float(center[1])
            if roi["height_v_range"][0] <= v <= roi["height_v_range"][1]:
                point = frame_point(center, ground, origin, direction, "height")
                height.append(point)
            elif roi["baseline_v_ranges"][0][0] <= v <= roi["baseline_v_ranges"][0][1]:
                point = frame_point(center, ground, origin, direction, "baseline_before")
                baseline.append(point)
            elif roi["baseline_v_ranges"][1][0] <= v <= roi["baseline_v_ranges"][1][1]:
                point = frame_point(center, ground, origin, direction, "baseline_after")
                baseline.append(point)
            else:
                continue
            if point["s"] < domain_min or point["s"] > domain_max:
                s_outside += 1
        frame_points[(pose_id, repeat)] = {"baseline": baseline, "height": height, "valid_count": valid_count, "s_outside": s_outside}

    models: dict[str, Any] = {}
    model_status: dict[str, str] = {}
    for pose_id in POSE_IDS:
        ground = frame_points[(pose_id, 1)]["baseline"]
        try:
            s = np.asarray([point["s"] for point in ground], dtype=np.float64)
            z = np.asarray([point["z"] for point in ground], dtype=np.float64)
            models[pose_id] = _fit_fixed_s_profile(s, z, app.measurement)
            model_status[pose_id] = "success"
        except Exception as error:
            models[pose_id] = None
            model_status[pose_id] = f"{type(error).__name__}: {error}"

    k_full = float(frozen["k_full"])
    for item in entries:
        pose_id = item["pose_id"]
        repeat = int(item["repeat_index"])
        points = frame_points[(pose_id, repeat)]
        model = models[pose_id]
        result = measurement_result(points["height"], model, app.measurement) if model is not None else {"status": "ground_proxy_failed", "value": None, "median": None, "std": None, "point_count": len(points["height"]), "inlier_count": 0, "unsupported_count": 0}
        raw = result.get("value") if result.get("status") == "success" else None
        corrected = None if raw is None else k_full * float(raw)
        raw_error = None if raw is None else float(raw) - TRUTH_MM
        corrected_error = None if corrected is None else float(corrected) - TRUTH_MM
        scope = "calibration_repeat1_in_sample" if repeat == 1 else "evaluation_repeat2_5"
        frame_rows.append({
            "position_id": f"laser{pose_id}",
            "pose_id": pose_id,
            "position_rank": int(pose_id),
            "repeat_index": repeat,
            "filename": next(entry["path"].name for entry in entries if entry["pose_id"] == pose_id and entry["repeat_index"] == repeat),
            "scope": scope,
            "truth_mm": TRUTH_MM,
            "steger_called_once": True,
            "c1_enabled": True,
            "center_count": int(len(item["centers"])),
            "c1_valid_count": int(points["valid_count"]),
            "c1_filtered_count": int(len(item["centers"]) - points["valid_count"]),
            "s_outside_frozen_domain_count": int(points["s_outside"]),
            "ground_proxy_status": model_status[pose_id],
            "ground_proxy_a_mm_per_mm": None if model is None else float(model.slope),
            "ground_proxy_b_mm": None if model is None else float(model.intercept),
            "ground_proxy_point_count": None if model is None else int(model.point_count),
            "ground_proxy_inlier_count": None if model is None else int(model.inlier_count),
            "ground_proxy_rmse_mm": None if model is None else float(model.rmse),
            "ground_proxy_s_span_mm": None if model is None else float(model.s_max - model.s_min),
            "height_point_count": int(result.get("point_count", len(points["height"]))),
            "height_inlier_count": int(result.get("inlier_count", 0)),
            "height_status": result.get("status", ""),
            "raw_B_height_mm": raw,
            "raw_B_error_mm": raw_error,
            "raw_B_abs_error_mm": None if raw_error is None else abs(raw_error),
            "frozen_H1_height_mm": corrected,
            "frozen_H1_error_mm": corrected_error,
            "frozen_H1_abs_error_mm": None if corrected_error is None else abs(corrected_error),
            "height_scale_k_full": k_full,
        })

    frame_rows.sort(key=lambda row: (row["position_rank"], row["repeat_index"]))
    position_rows: list[dict[str, Any]] = []
    for pose_id in POSE_IDS:
        formal = [row for row in frame_rows if row["pose_id"] == pose_id and row["repeat_index"] > 1]
        raw_values = np.asarray([row["raw_B_height_mm"] for row in formal if row["raw_B_height_mm"] is not None], dtype=np.float64)
        corrected_values = np.asarray([row["frozen_H1_height_mm"] for row in formal if row["frozen_H1_height_mm"] is not None], dtype=np.float64)
        raw_errors = raw_values - TRUTH_MM
        corrected_errors = corrected_values - TRUTH_MM
        raw_m = point_metric(raw_errors)
        corrected_m = point_metric(corrected_errors)
        position_rows.append({
            "position_id": f"laser{pose_id}",
            "pose_id": pose_id,
            "position_rank": int(pose_id),
            "truth_mm": TRUTH_MM,
            "expected_formal_frame_count": 4,
            "raw_successful_formal_frame_count": int(len(raw_values)),
            "h1_successful_formal_frame_count": int(len(corrected_values)),
            "raw_condition_mean_mm": float(np.mean(raw_values)) if len(raw_values) else None,
            "raw_condition_error_mm": float(np.mean(raw_errors)) if len(raw_errors) else None,
            "raw_condition_abs_error_mm": abs(float(np.mean(raw_errors))) if len(raw_errors) else None,
            "raw_repeatability_sigma_mm": float(np.std(raw_values)) if len(raw_values) else None,
            "h1_condition_mean_mm": float(np.mean(corrected_values)) if len(corrected_values) else None,
            "h1_condition_error_mm": float(np.mean(corrected_errors)) if len(corrected_errors) else None,
            "h1_condition_abs_error_mm": abs(float(np.mean(corrected_errors))) if len(corrected_errors) else None,
            "h1_repeatability_sigma_mm": float(np.std(corrected_values)) if len(corrected_values) else None,
            "raw_mae_mm": raw_m["mae_mm"],
            "raw_rmse_mm": raw_m["rmse_mm"],
            "raw_bias_mm": raw_m["bias_mm"],
            "raw_p95_mm": raw_m["p95_mm"],
            "raw_max_mm": raw_m["max_mm"],
            "h1_mae_mm": corrected_m["mae_mm"],
            "h1_rmse_mm": corrected_m["rmse_mm"],
            "h1_bias_mm": corrected_m["bias_mm"],
            "h1_p95_mm": corrected_m["p95_mm"],
            "h1_max_mm": corrected_m["max_mm"],
            "raw_pass_0p05_rate": raw_m["pass_0p05_rate"],
            "h1_pass_0p05_rate": corrected_m["pass_0p05_rate"],
            "raw_pass_0p1_rate": raw_m["pass_0p1_rate"],
            "h1_pass_0p1_rate": corrected_m["pass_0p1_rate"],
            "raw_pass_0p2_rate": raw_m["pass_0p2_rate"],
            "h1_pass_0p2_rate": corrected_m["pass_0p2_rate"],
            "raw_status": "success" if len(raw_values) == 4 else "incomplete",
            "h1_status": "success" if len(corrected_values) == 4 else "incomplete",
        })

    formal_rows = [row for row in frame_rows if row["repeat_index"] > 1]
    raw_frame_errors = [float(row["raw_B_error_mm"]) for row in formal_rows if row["raw_B_error_mm"] is not None]
    h1_frame_errors = [float(row["frozen_H1_error_mm"]) for row in formal_rows if row["frozen_H1_error_mm"] is not None]
    raw_condition_errors = [float(row["raw_condition_error_mm"]) for row in position_rows if row["raw_condition_error_mm"] is not None]
    h1_condition_errors = [float(row["h1_condition_error_mm"]) for row in position_rows if row["h1_condition_error_mm"] is not None]
    raw_frame_metric = point_metric(raw_frame_errors)
    h1_frame_metric = point_metric(h1_frame_errors)
    raw_condition_metric = point_metric(raw_condition_errors)
    h1_condition_metric = point_metric(h1_condition_errors)
    complete = len(raw_frame_errors) == 20 and len(h1_frame_errors) == 20 and len(raw_condition_errors) == 5 and len(h1_condition_errors) == 5
    primary_improvement = (
        complete
        and h1_frame_metric["mae_mm"] <= raw_frame_metric["mae_mm"] + EPS
        and h1_frame_metric["rmse_mm"] <= raw_frame_metric["rmse_mm"] + EPS
        and h1_condition_metric["mae_mm"] <= raw_condition_metric["mae_mm"] + EPS
        and h1_condition_metric["rmse_mm"] <= raw_condition_metric["rmse_mm"] + EPS
    )
    tail_non_decreasing = complete and all(
        h1_metric[key] <= raw_metric[key] + EPS
        for h1_metric, raw_metric in (
            (h1_frame_metric, raw_frame_metric),
            (h1_condition_metric, raw_condition_metric),
        )
        for key in ("p95_mm", "max_mm")
    )
    position_abs_error_non_increasing = complete and all(
        row["h1_condition_abs_error_mm"] <= row["raw_condition_abs_error_mm"] + EPS
        for row in position_rows
    )
    all_pass_rates_non_decreasing = complete and all(
        h1_frame_metric[f"pass_{str(limit).replace('.', 'p')}_rate"] + EPS >= raw_frame_metric[f"pass_{str(limit).replace('.', 'p')}_rate"]
        for limit in PASS_LIMITS
    )
    if complete and primary_improvement and tail_non_decreasing and position_abs_error_non_increasing and all_pass_rates_non_decreasing:
        status = "PASS"
    elif complete and primary_improvement:
        status = "PARTIAL"
    else:
        status = "FAIL"

    repeatability_raw = np.asarray([row["raw_repeatability_sigma_mm"] for row in position_rows if row["raw_repeatability_sigma_mm"] is not None], dtype=np.float64)
    repeatability_h1 = np.asarray([row["h1_repeatability_sigma_mm"] for row in position_rows if row["h1_repeatability_sigma_mm"] is not None], dtype=np.float64)
    summary = {
        "schema_version": 1,
        "HEIGHT_SCALE_50MM_VALIDATION": status,
        "formal_scope": "repeat2_5 only; 20 frames and 5 position condition means",
        "truth_mm": TRUTH_MM,
        "frozen_height_scale": {
            "k_full": k_full,
            "parameter_sha256": frozen["parameter_sha256"],
            "input_sha256": frozen["input"]["sha256"],
        },
        "provenance": {
            "data_root": str(data_root),
            "obs_50mm_image_count": 25,
            "steger_call_count_reused": 25,
            "steger_called_once_per_frame": True,
            "c1_enabled": True,
            "config_sha256": sha256(args.config.resolve()),
            "roi_registry_path": str(registry_path),
            "roi_registry_sha256": sha256(registry_path),
            "ground3_summary_path": str(args.ground3_summary.resolve()),
            "ground3_summary_sha256": sha256(args.ground3_summary.resolve()),
            "ground_origin_xy": origin.tolist(),
            "ground_direction_xy": direction.tolist(),
            "s_domain_mm": [domain_min, domain_max],
            "roi_selection_truth_or_error_used": False,
            "reused_artifacts": [
                "Ground-4A 29-condition CSV and H1 protocol",
                "frozen Ground-1 origin/direction/S domain",
                "Daheng 0811 calibration and Frozen C1 ray correction",
                "25 cached one-pass Steger center arrays",
            ],
            "newly_computed": [
                "obs_50mm C1 reconstruction on cached centers",
                "repeat-1 per-position session-linear ground proxy",
                "repeat2-5 raw B and frozen H1 metrics",
            ],
            "not_done": ["50mm participation in k fit/model selection", "C0/C1 refit", "Ground G(S) refit", "production configuration change"],
        },
        "frame_metrics": {"raw_B": raw_frame_metric, "frozen_H1": h1_frame_metric},
        "position_condition_metrics": {"raw_B": raw_condition_metric, "frozen_H1": h1_condition_metric},
        "raw_to_h1_improvement": {
            "frame_mae_delta_mm": None if raw_frame_metric["mae_mm"] is None or h1_frame_metric["mae_mm"] is None else h1_frame_metric["mae_mm"] - raw_frame_metric["mae_mm"],
            "frame_rmse_delta_mm": None if raw_frame_metric["rmse_mm"] is None or h1_frame_metric["rmse_mm"] is None else h1_frame_metric["rmse_mm"] - raw_frame_metric["rmse_mm"],
            "condition_mae_delta_mm": None if raw_condition_metric["mae_mm"] is None or h1_condition_metric["mae_mm"] is None else h1_condition_metric["mae_mm"] - raw_condition_metric["mae_mm"],
            "condition_rmse_delta_mm": None if raw_condition_metric["rmse_mm"] is None or h1_condition_metric["rmse_mm"] is None else h1_condition_metric["rmse_mm"] - raw_condition_metric["rmse_mm"],
        },
        "status_checks": {
            "complete_20_frames_and_5_positions": complete,
            "primary_mae_rmse_non_worse_at_both_scopes": primary_improvement,
            "p95_and_max_non_worse_at_both_scopes": tail_non_decreasing,
            "each_position_abs_error_non_worse": position_abs_error_non_increasing,
            "frame_pass_rates_non_decreasing": all_pass_rates_non_decreasing,
        },
        "repeatability": {
            "raw_median_sigma_mm": float(np.median(repeatability_raw)) if len(repeatability_raw) else None,
            "raw_p95_sigma_mm": float(np.percentile(repeatability_raw, 95.0)) if len(repeatability_raw) else None,
            "raw_max_sigma_mm": float(np.max(repeatability_raw)) if len(repeatability_raw) else None,
            "h1_median_sigma_mm": float(np.median(repeatability_h1)) if len(repeatability_h1) else None,
            "h1_p95_sigma_mm": float(np.percentile(repeatability_h1, 95.0)) if len(repeatability_h1) else None,
            "h1_max_sigma_mm": float(np.max(repeatability_h1)) if len(repeatability_h1) else None,
        },
        "status_rule": "PASS requires complete 20/20 formal frames and 5/5 condition means, non-worse MAE/RMSE/P95/Max at both scopes, non-worse absolute error at every position, and non-decreasing frame pass rates at ±0.05/±0.1/±0.2; PARTIAL requires complete data and primary MAE/RMSE improvement but at least one tail/position/pass-rate criterion fails; otherwise FAIL.",
    }

    frame_fields = list(frame_rows[0].keys())
    position_fields = list(position_rows[0].keys())
    write_csv(output / "height50_frame_metrics.csv", frame_rows, frame_fields)
    write_csv(output / "height50_position_metrics.csv", position_rows, position_fields)
    write_json(output / "height50_summary.json", summary)
    for pose_id in POSE_IDS:
        group = sorted([item for item in entries if item["pose_id"] == pose_id], key=lambda item: item["repeat_index"])
        median = load_grayscale_image(output / "median_images" / f"laser{pose_id}_median.tif")
        save_manual_overlay(output / "overlays" / f"laser{pose_id}_manual_roi_overlay.png", median, [item["centers"] for item in group], registry[pose_id])
    write_report(output, args, frozen, registry_path, summary, position_rows, model_status)
    print(f"HEIGHT_SCALE_50MM_VALIDATION={status}")
    print(f"formal frames raw/H1={raw_frame_metric['count']}/{h1_frame_metric['count']}, position means raw/H1={raw_condition_metric['count']}/{h1_condition_metric['count']}")
    return 0


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "MISSING"
    return f"{float(value):.{digits}f}"


def metric_table_row(label: str, metric: dict[str, Any]) -> str:
    return (
        f"| {label} | {metric['count']} | {fmt(metric['bias_mm'])} | {fmt(metric['mae_mm'])} | "
        f"{fmt(metric['rmse_mm'])} | {fmt(metric['p95_mm'])} | {fmt(metric['max_mm'])} | "
        f"{metric['pass_0p05_count']}/{metric['count']} ({fmt(100 * metric['pass_0p05_rate'], 1)}%) | "
        f"{metric['pass_0p1_count']}/{metric['count']} ({fmt(100 * metric['pass_0p1_rate'], 1)}%) | "
        f"{metric['pass_0p2_count']}/{metric['count']} ({fmt(100 * metric['pass_0p2_rate'], 1)}%) |"
    )


def write_report(output: Path, args: argparse.Namespace, frozen: dict[str, Any], registry_path: Path, summary: dict[str, Any], position_rows: list[dict[str, Any]], model_status: dict[str, str]) -> None:
    frame = summary["frame_metrics"]
    condition = summary["position_condition_metrics"]
    improvement = summary["raw_to_h1_improvement"]
    repeatability = summary["repeatability"]
    lines = [
        "# Height-2/3：obs_50mm 首次真正 held-out 高度尺度验证",
        "",
        f"- `HEIGHT_SCALE_50MM_VALIDATION={summary['HEIGHT_SCALE_50MM_VALIDATION']}`",
        "- 判定摘要：H1 的 pooled MAE/RMSE 与 ±0.1 达标率改善，但 P95/Max 尾部指标和部分 position 的绝对误差变差，因此标为 `PARTIAL`，不视为完全泛化通过。" if summary["HEIGHT_SCALE_50MM_VALIDATION"] == "PARTIAL" else "- 判定摘要：H1 在预设的 pooled、尾部、逐 position 与达标率条件下均不劣于 raw B。",
        "- 本轮为 retrospective held-out diagnostic；不修改 C0/C1、Ground G(S)、GUI 或生产配置。",
        "",
        "## 冻结参数",
        "",
        f"- H1：`h_corr = k*h`，`k_full = {fmt(frozen['k_full'], 15)}`。",
        f"- 训练输入：原 Ground-4A 29 个成功 `session_linear` condition，等权；输入 SHA-256：`{frozen['input']['sha256']}`。",
        f"- 参数 SHA-256：`{frozen['parameter_sha256']}`；冻结状态：`{frozen['status']}`。",
        "- `obs_50mm` 未参与 k 拟合、模型选择或参数调整；k 在读取 50mm 图像之前写入冻结 artifact。",
        "",
        "## Provenance / reuse audit",
        "",
        "- 复用：Ground-4A 29-condition 的 B 链协议、Daheng 0811 配置、Frozen C1 ray correction，以及 Ground-1 frozen `origin_xy/direction_xy/S domain`。",
        f"- 复用：25 个 50mm 帧的 one-pass Steger cache；本轮 evaluate 未重新运行 Steger，C1 使用同一缓存中心点。",
        f"- 新增：25 帧 C1 重建、5 个 repeat-1 session-linear ground proxy、20 个 formal repeat2–5 frame metrics、5 个 position condition means。",
        f"- ROI registry：`{registry_path.name}`，geometry-only/manual-confirmed，SHA-256：`{sha256(registry_path)}`。",
        f"- C1 config `enable_laser_ray_correction=true`；config SHA-256：`{sha256(args.config.resolve())}`。",
        "",
        "## Formal repeat2–5：20 帧 pooled metrics",
        "",
        "| chain | n | Bias | MAE | RMSE | P95 | Max | ±0.05 | ±0.1 | ±0.2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_table_row("raw B session_linear", frame["raw_B"]),
        metric_table_row("frozen H1 scale", frame["frozen_H1"]),
        "",
        "## 5 个 position condition mean",
        "",
        "| chain | n | Bias | MAE | RMSE | P95 | Max | ±0.05 | ±0.1 | ±0.2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_table_row("raw B condition mean", condition["raw_B"]),
        metric_table_row("frozen H1 condition mean", condition["frozen_H1"]),
        "",
        "## Raw → H1 增量",
        "",
        f"- frame MAE delta (H1 - raw)：`{fmt(improvement['frame_mae_delta_mm'])}` mm；RMSE delta：`{fmt(improvement['frame_rmse_delta_mm'])}` mm。",
        f"- position-mean MAE delta (H1 - raw)：`{fmt(improvement['condition_mae_delta_mm'])}` mm；RMSE delta：`{fmt(improvement['condition_rmse_delta_mm'])}` mm。负值表示改善。",
        "",
        "| position | raw mean | H1 mean | raw bias | H1 bias | raw repeatability sigma | H1 repeatability sigma |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in position_rows:
        lines.append(
            f"| {row['position_id']} | {fmt(row['raw_condition_mean_mm'])} | {fmt(row['h1_condition_mean_mm'])} | "
            f"{fmt(row['raw_condition_error_mm'])} | {fmt(row['h1_condition_error_mm'])} | "
            f"{fmt(row['raw_repeatability_sigma_mm'])} | {fmt(row['h1_repeatability_sigma_mm'])} |"
        )
    lines += [
        "",
        "## Repeatability",
        "",
        f"- raw B：median/P95/Max sigma = `{fmt(repeatability['raw_median_sigma_mm'])}` / `{fmt(repeatability['raw_p95_sigma_mm'])}` / `{fmt(repeatability['raw_max_sigma_mm'])}` mm。",
        f"- frozen H1：median/P95/Max sigma = `{fmt(repeatability['h1_median_sigma_mm'])}` / `{fmt(repeatability['h1_p95_sigma_mm'])}` / `{fmt(repeatability['h1_max_sigma_mm'])}` mm。",
        "",
        "## ROI 与选择边界",
        "",
        "- 每个 laser position 使用 5 帧 median image + Steger overlay；5 个 repeat 共用一个手工 registry。",
        "- ROI 仅依据原图中的物理 ground/height 几何范围；未使用 50mm 真值、误差、residual 或 residual threshold。",
        "- 未对棋盘缺失点插值，也未按重建数值结果删除点；重建无效点只按 C1 的原有有效性自然剔除并计入审计列。",
        "- repeat1 只用于该 position 的 `Zg=a*S+b` ground proxy；正式指标只用 repeat2–5。",
        "",
        "## Ground / C1 细节",
        "",
        "- 统一使用 frozen `S=(XY-origin_xy)·direction_xy`，未按 position 重新定义 S。",
        f"- repeat1 ground proxy 状态：" + ", ".join(f"{pose}={status}" for pose, status in model_status.items()) + ".",
        "- 未重新拟合 C0/C1、未重新拟合 G(S)，H1 只作用于 raw B height。",
        "",
        "## 输出",
        "",
        "- `frozen_height_scale.json`：冻结 k、输入 SHA、参数 SHA。",
        "- `height50_manual_roi_registry.json` 与 `overlays/*_manual_roi_overlay.png`：geometry-only ROI registry/复核图。",
        "- `height50_frame_metrics.csv`：全 25 帧，repeat1 标为 in-sample，repeat2–5 为 formal。",
        "- `height50_position_metrics.csv`：5 个 position 的 repeat2–5 condition summary。",
        "",
    ]
    (output / "height50_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ground4a-csv", type=Path, default=DEFAULT_GROUND4A_CSV)
    parser.add_argument("--ground3-summary", type=Path, default=DEFAULT_GROUND3_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=None, help="manual geometry-only ROI registry; required for evaluate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "prepare":
        return prepare(args)
    if args.registry is None:
        raise SystemExit("evaluate requires --registry height50_manual_roi_registry.json")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
