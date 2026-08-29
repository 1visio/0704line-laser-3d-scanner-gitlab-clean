"""Ground-5A frozen factory spatial-profile diagnostic for chessboard_0821.

This is an analysis-only extension around the existing calibration helpers.
It deliberately keeps the protocol boundary explicit:

* fit poses 001--005 are the only data used for coordinate/model selection;
* validation poses 006--007 are loaded and evaluated only after the coordinate
  and Factory Profile candidate have been frozen;
* one Steger extraction is cached per laser TIFF;
* PnP uses the existing Session-ground implementation;
* the existing full physical-board polygon selector is the only point mask;
* the same cached centers are passed to frozen C0 and frozen C1 reconstruction;
* Session Ground Reference, H1/Stage-A, Ground-3 G(S), and production writes
  are disabled for this diagnostic.

The Factory Profile is fitted to per-frame detrended residuals
``r = Zg - (a*x + b)``.  This makes the held-out chains interpretable:

``A: Zg``
``B: Zg - F(x)``
``C: Zg - F(x) - (a_pose*x + b_pose)``

No point is removed using a Z residual.  Robust linear fitting is used only to
estimate the diagnostic frame line; all board-mask-selected points remain in
the reported residuals and profile bins.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from app_config import load_app_config
from calibration.manifest import load_calibration_package
from calibration.session_ground import estimate_session_ground_extrinsic
from laser.backends import create_extraction_params
from laser.laser_extractor import extract_laser_center
from measurement.board_mask import select_board_ground_points
from measurement.ground_reference import fit_ground_profile
from reconstruction.laser_ray_correction import (
    evaluate_frozen_laser_ray_correction,
)
from reconstruction.reconstructor import reconstruct_uv_to_ground
from tools.fit_ground_reference_20frames import (
    _load_dataset_metadata,
    _sha256_file,
)
from tools.fit_ground_spatial_correction_ground3 import (
    PoseProfiles,
    ResidualFrame,
    _basis_matrix,
    _bin_frame_residuals,
    _common_bins,
)


FIT_POSES = ("001", "002", "003", "004", "005")
HELDOUT_POSES = ("006", "007")
ALL_POSES = FIT_POSES + HELDOUT_POSES
COORDINATES = ("full_v", "c1_s")

DEFAULT_FIT_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\chessboard_0821\fit"
)
DEFAULT_VALIDATION_DIR = Path(
    r"D:\Docs\linelaserscan\calibration_tool\projects\daheng\data\chessboard_0821\validation"
)
DEFAULT_CONFIG = TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_OUTPUT_DIR = TOOL_ROOT.parent / "outputs" / "ground5a_factory_profile_0821"

# These are this experiment's parameters, not Ground-3 G(S) parameters.
PROFILE_BIN_COUNT = 40
FACTORY_INTERIOR_KNOT_COUNTS = (1, 2, 3)
FACTORY_SPLINE_DEGREE = 3
FACTORY_SMOOTHNESS_LAMBDA = 0.01
FACTORY_MIN_FRAME_FRACTION = 0.8
FACTORY_MIN_COMMON_BIN_FRACTION = 0.8
LOW_FREQUENCY_WINDOW_BINS = 5
COORDINATE_TIE_TOLERANCE_MM = 1.0e-6

# Predeclared diagnostic classification thresholds.  They are never adjusted
# after reading pose006/007.
HELDOUT_IMPROVEMENT_PASS = 0.10
HELDOUT_SHAPE_CORRELATION_PASS = 0.80
SESSION_LINEAR_IMPROVEMENT_YES = 0.10
SESSION_LINEAR_IMPROVEMENT_NO = 0.05


@dataclass(slots=True)
class PoseRecord:
    pose_id: str
    split: str
    chess_path: Path
    laser_paths: list[Path]


@dataclass(slots=True)
class PnPRecord:
    pose_id: str
    split: str
    chess_path: Path
    result: Any
    reprojection_rmse_px: float
    detection_method: str


@dataclass(slots=True)
class CoordinateFit:
    x: np.ndarray
    z: np.ndarray
    predicted_z: np.ndarray
    residual: np.ndarray
    inlier_mask: np.ndarray
    slope: float
    intercept: float
    fit_rmse_mm: float
    sigma_mm: float
    linear_fit_coordinate_scale: float


@dataclass(slots=True)
class GroundFrame:
    pose_id: str
    split: str
    frame_id: str
    path: Path
    camera_frame_number: int | None
    centers_all: np.ndarray
    valid_pixels_uv: np.ndarray
    points_ground: np.ndarray
    c0_point_count: int
    c1_point_count: int
    c0_selected_count: int
    c1_selected_count: int
    c0_filtered: dict[str, int]
    c1_filtered: dict[str, int]
    image_shape: tuple[int, int]
    image_dtype: str
    file_sha256: str
    quality: dict[str, Any]
    mask_metadata: dict[str, Any]
    c1_s_raw: np.ndarray
    c1_s_eval: np.ndarray
    c1_clamped_count: int
    coordinates: dict[str, CoordinateFit]
    extraction_ms: float | None
    c0_reconstruction_ms: float
    c1_reconstruction_ms: float


@dataclass(slots=True)
class CoordinateProfile:
    coordinate: str
    specs: list[dict[str, float]]
    detrended: dict[str, PoseProfiles]
    absolute: dict[str, PoseProfiles]
    common_mask: np.ndarray
    support_mask: np.ndarray
    support_domain: tuple[float, float]


@dataclass(slots=True)
class FactorySpline:
    interior_knot_count: int
    knots: np.ndarray
    coefficients: np.ndarray
    domain_min: float
    domain_max: float
    degree: int
    smoothness_lambda: float
    train_pose_ids: tuple[str, ...]
    observation_count: int
    fit_rmse_mm: float
    cv_rmse_mm: float | None


def _natural_key(value: str | Path) -> list[str | int]:
    text = Path(value).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fields})


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise RuntimeError(f"cannot read 2-D image: {path}")
    if image.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise RuntimeError(f"unsupported image dtype {image.dtype}: {path}")
    return image


def _discover_pose_records(data_dir: Path, split: str, pose_ids: Iterable[str]) -> list[PoseRecord]:
    records: list[PoseRecord] = []
    for pose_id in pose_ids:
        chess_candidates = sorted(
            [
                path
                for path in data_dir.glob("*.tif")
                if re.fullmatch(rf"chess[ _]?{pose_id}\.tif", path.name, flags=re.IGNORECASE)
            ],
            key=_natural_key,
        )
        laser_candidates = sorted(
            [
                path
                for path in data_dir.glob("*.tif")
                if re.fullmatch(
                    rf"laser[ _]?{pose_id}(?:_\d+)?\.tif",
                    path.name,
                    flags=re.IGNORECASE,
                )
            ],
            key=_natural_key,
        )
        if len(chess_candidates) != 1:
            raise RuntimeError(f"pose{pose_id} needs exactly one chess TIFF in {data_dir}")
        if len(laser_candidates) != 5:
            raise RuntimeError(f"pose{pose_id} needs exactly five laser TIFFs in {data_dir}; got {len(laser_candidates)}")
        records.append(PoseRecord(pose_id, split, chess_candidates[0], laser_candidates))
    return records


def _load_manifest_context(fit_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path, Path]:
    return _load_dataset_metadata(fit_dir)


def _cache_key(app: Any, config_path: Path) -> dict[str, Any]:
    return {
        "config_sha256": _sha256_file(config_path),
        "extraction_method": app.extraction_method,
        "extraction_options": _json_ready(app.extraction_options),
        "image_offset_xy": [0, 0],
        "full_sensor_coordinate_system": True,
    }


def _cache_records(
    records: list[PoseRecord],
    output_dir: Path,
    app: Any,
    config_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a protocol-compatible center cache or run Steger once per TIFF."""
    cache_path = output_dir / "steger_geometry_cache.json"
    center_dir = output_dir / "center_cache"
    center_dir.mkdir(parents=True, exist_ok=True)
    desired: list[tuple[str, str, Path]] = []
    for pose in records:
        for index, path in enumerate(pose.laser_paths, start=1):
            desired.append((pose.split, pose.pose_id, path))
    source_hashes = {str(path.resolve()): _sha256_file(path) for _, _, path in desired}
    key = _cache_key(app, config_path)
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_ok = cache.get("one_steger_per_frame") is True and cache.get("protocol_key") == key
            cached_by_path = {str(item["source_path"]): item for item in cache.get("frames", [])}
            cache_ok = cache_ok and set(cached_by_path) == set(source_hashes)
            if cache_ok:
                centers_by_path: dict[str, np.ndarray] = {}
                for source_path, source_sha in source_hashes.items():
                    item = cached_by_path[source_path]
                    if item.get("source_sha256") != source_sha or int(item.get("steger_run_count", 0)) != 1:
                        cache_ok = False
                        break
                    center_path = Path(item["centers_path"])
                    centers = np.asarray(np.load(center_path), dtype=np.float64)
                    if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
                        cache_ok = False
                        break
                    centers_by_path[source_path] = np.ascontiguousarray(centers)
                if cache_ok:
                    cache["reused_existing_cache"] = True
                    cache_path.write_text(
                        json.dumps(_json_ready(cache), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    return centers_by_path, cache
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    if app.extraction_method != "steger":
        raise RuntimeError(f"Ground-5A requires extraction.method=steger, got {app.extraction_method!r}")
    extraction_params = create_extraction_params(app.extraction_method, app.extraction_options)
    centers_by_path = {}
    cache_frames: list[dict[str, Any]] = []
    for split, pose_id, path in desired:
        image = _read_image(path)
        start = time.perf_counter()
        centers = np.asarray(
            extract_laser_center(image, extraction_params, image_offset=(0, 0)),
            dtype=np.float64,
        )
        extraction_ms = (time.perf_counter() - start) * 1000.0
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
            raise RuntimeError(f"Steger returned no valid centers for {path}")
        centers = np.ascontiguousarray(centers)
        source_path = str(path.resolve())
        center_path = center_dir / f"{split}_pose{pose_id}_{path.stem.replace(' ', '_')}_centers.npy"
        np.save(center_path, centers)
        centers_by_path[source_path] = centers
        cache_frames.append(
            {
                "split": split,
                "pose_id": pose_id,
                "source_file": path.name,
                "source_path": source_path,
                "source_sha256": source_hashes[source_path],
                "centers_path": str(center_path.resolve()),
                "center_count": int(len(centers)),
                "image_shape": [int(image.shape[0]), int(image.shape[1])],
                "image_dtype": str(image.dtype),
                "extraction_ms": float(extraction_ms),
                "steger_run_count": 1,
            }
        )
        print(f"Steger {split} pose{pose_id} {path.name}: centers={len(centers)}")
    cache = {
        "schema_version": 1,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "one_steger_per_frame": True,
        "protocol_key": key,
        "selection_basis": "none; Steger cache only",
        "residual_used": False,
        "frames": cache_frames,
        "reused_existing_cache": False,
    }
    cache_path.write_text(
        json.dumps(_json_ready(cache), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return centers_by_path, cache


def _run_pnp(
    records: list[PoseRecord],
    calibration: dict[str, Any],
    board_config: Any,
) -> dict[str, PnPRecord]:
    results: dict[str, PnPRecord] = {}
    intrinsics = {"K": calibration["K"], "D": calibration["D"]}
    for record in records:
        image = _read_image(record.chess_path)
        result = estimate_session_ground_extrinsic(image, intrinsics, board_config)
        if result.status != "success" or result.R is None or result.t is None:
            raise RuntimeError(
                f"PnP failed for pose{record.pose_id}: {result.status}: {result.message}"
            )
        results[record.pose_id] = PnPRecord(
            pose_id=record.pose_id,
            split=record.split,
            chess_path=record.chess_path,
            result=result,
            reprojection_rmse_px=float(result.reprojection_rmse_px),
            detection_method=str(result.detection_method),
        )
        print(
            f"PnP {record.split} pose{record.pose_id}: "
            f"rmse={result.reprojection_rmse_px:.6f}px method={result.detection_method}"
        )
    return results


def _board_selected_indices(
    pixels_uv: np.ndarray,
    points_ground: np.ndarray,
    pnp: PnPRecord,
    calibration: dict[str, Any],
    board_config: Any,
    inset_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected, metadata = select_board_ground_points(
        pixels_uv,
        points_ground,
        rvec=pnp.result.rvec,
        tvec=pnp.result.tvec,
        camera_matrix=calibration["K"],
        dist_coeffs=calibration["D"],
        pattern_cols=board_config.pattern_cols,
        pattern_rows=board_config.pattern_rows,
        square_size_mm=board_config.square_size_mm,
        image_offset=(0, 0),
        inset_mm=inset_mm,
        detected_corners=pnp.result.detected_corners,
    )
    count = len(points_ground)
    index_points = np.column_stack(
        [np.arange(count, dtype=np.float64), np.zeros((count, 2), dtype=np.float64)]
    )
    selected_index_points, _ = select_board_ground_points(
        pixels_uv,
        index_points,
        rvec=pnp.result.rvec,
        tvec=pnp.result.tvec,
        camera_matrix=calibration["K"],
        dist_coeffs=calibration["D"],
        pattern_cols=board_config.pattern_cols,
        pattern_rows=board_config.pattern_rows,
        square_size_mm=board_config.square_size_mm,
        image_offset=(0, 0),
        inset_mm=inset_mm,
        detected_corners=pnp.result.detected_corners,
    )
    indices = np.rint(selected_index_points[:, 0]).astype(np.int64)
    if len(indices) != len(selected) or not np.array_equal(selected, points_ground[indices]):
        raise RuntimeError("board-mask selector index order is not reproducible")
    metadata["selected_indices"] = indices.tolist()
    return indices, metadata


def _coordinate_fit(x: np.ndarray, z: np.ndarray, params: Any) -> CoordinateFit:
    values_x = np.asarray(x, dtype=np.float64)
    values_z = np.asarray(z, dtype=np.float64)
    # The shared kernel's ``ptp < 1`` guard is expressed in mm.  Frozen C1
    # s_ray is dimensionless and spans roughly 0.2, so use a fixed numerical
    # scale only inside the adapter, then convert the fitted slope back to
    # Zg per original coordinate.  Residuals and inlier decisions are unchanged.
    coordinate_scale = 1000.0 if float(np.ptp(values_x)) < 1.0 else 1.0
    fit_x = values_x * coordinate_scale
    # Reuse the public robust ground-profile kernel through a synthetic
    # 1-D ground line; no new robust fitter is implemented here.
    synthetic_points = np.column_stack(
        [fit_x, np.zeros(len(fit_x), dtype=np.float64), values_z]
    )
    profile, sigma = fit_ground_profile(
        synthetic_points,
        params,
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
    )
    predicted = profile.slope_z_per_mm * fit_x + profile.intercept_z_mm
    residual = values_z - predicted
    return CoordinateFit(
        x=np.ascontiguousarray(values_x),
        z=np.ascontiguousarray(values_z),
        predicted_z=np.ascontiguousarray(predicted),
        residual=np.ascontiguousarray(residual),
        inlier_mask=np.ascontiguousarray(profile.inlier_mask),
        slope=float(profile.slope_z_per_mm * coordinate_scale),
        intercept=float(profile.intercept_z_mm),
        fit_rmse_mm=float(profile.rmse_mm),
        sigma_mm=float(sigma),
        linear_fit_coordinate_scale=coordinate_scale,
    )


def _process_frames(
    records: list[PoseRecord],
    pnp_by_pose: dict[str, PnPRecord],
    centers_by_path: dict[str, np.ndarray],
    metadata_by_name: dict[str, dict[str, Any]],
    base_calibration: dict[str, Any],
    params_c0: Any,
    params_c1: Any,
    board_config: Any,
    mask_inset_mm: float,
    measurement_params: Any,
) -> list[GroundFrame]:
    frames: list[GroundFrame] = []
    for record in records:
        pnp = pnp_by_pose[record.pose_id]
        pnp_calibration = dict(base_calibration)
        pnp_calibration["R"] = np.asarray(pnp.result.R, dtype=np.float64)
        pnp_calibration["t"] = np.asarray(pnp.result.t, dtype=np.float64).reshape(3)
        # Explicitly disable the historical H1/ground-u layer for this run.
        pnp_calibration["ground_u_compensation"] = None
        source_metadata = metadata_by_name.get(record.laser_paths[0].name, {})
        for laser_index, path in enumerate(record.laser_paths, start=1):
            source_path = str(path.resolve())
            centers = centers_by_path[source_path]
            start = time.perf_counter()
            c0_result = reconstruct_uv_to_ground(centers, pnp_calibration, params_c0)
            c0_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            c1_result = reconstruct_uv_to_ground(centers, pnp_calibration, params_c1)
            c1_ms = (time.perf_counter() - start) * 1000.0
            if len(c1_result.points_ground) == 0:
                raise RuntimeError(f"C1 reconstructed zero points for {path}")
            c1_indices, mask_metadata = _board_selected_indices(
                c1_result.pixels_uv,
                c1_result.points_ground,
                pnp,
                base_calibration,
                board_config,
                mask_inset_mm,
            )
            if len(c1_indices) < 20:
                raise RuntimeError(
                    f"pose{record.pose_id} frame {path.name} has too few physical-board points: {len(c1_indices)}"
                )
            c0_selected_count = 0
            if len(c0_result.points_ground):
                c0_indices, _ = _board_selected_indices(
                    c0_result.pixels_uv,
                    c0_result.points_ground,
                    pnp,
                    base_calibration,
                    board_config,
                    mask_inset_mm,
                )
                c0_selected_count = len(c0_indices)
            pixels = np.ascontiguousarray(c1_result.pixels_uv[c1_indices], dtype=np.float64)
            points = np.ascontiguousarray(c1_result.points_ground[c1_indices], dtype=np.float64)
            normalized = cv2.undistortPoints(
                pixels.reshape(-1, 1, 2),
                np.asarray(base_calibration["K"], dtype=np.float64),
                np.asarray(base_calibration["D"], dtype=np.float64),
            ).reshape(-1, 2)
            rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
            correction = base_calibration.get("laser_ray_correction")
            if correction is None:
                raise RuntimeError("Frozen C1 correction is missing")
            evaluation = evaluate_frozen_laser_ray_correction(rays, correction)
            coordinates = {
                "full_v": pixels[:, 1],
                "c1_s": evaluation.s_raw,
            }
            coordinate_fits = {
                name: _coordinate_fit(values, points[:, 2], measurement_params)
                for name, values in coordinates.items()
            }
            metadata = metadata_by_name.get(path.name, source_metadata)
            quality = metadata.get("quality", {}) if isinstance(metadata, dict) else {}
            if not isinstance(quality, dict):
                quality = {}
            frame = GroundFrame(
                pose_id=record.pose_id,
                split=record.split,
                frame_id=f"{record.split}_pose{record.pose_id}_frame{laser_index:02d}",
                path=path,
                camera_frame_number=(
                    int(metadata["camera_frame_number"])
                    if metadata.get("camera_frame_number") is not None
                    else None
                ),
                centers_all=centers,
                valid_pixels_uv=pixels,
                points_ground=points,
                c0_point_count=int(c0_result.point_count),
                c1_point_count=int(c1_result.point_count),
                c0_selected_count=int(c0_selected_count),
                c1_selected_count=int(len(points)),
                c0_filtered={str(k): int(v) for k, v in c0_result.filtered.items()},
                c1_filtered={str(k): int(v) for k, v in c1_result.filtered.items()},
                image_shape=(3000, 4096),
                image_dtype="uint8",
                file_sha256=_sha256_file(path),
                quality=quality,
                mask_metadata=mask_metadata,
                c1_s_raw=np.ascontiguousarray(evaluation.s_raw),
                c1_s_eval=np.ascontiguousarray(evaluation.s_eval),
                c1_clamped_count=int(np.count_nonzero(evaluation.clamped)),
                coordinates=coordinate_fits,
                extraction_ms=None,
                c0_reconstruction_ms=float(c0_ms),
                c1_reconstruction_ms=float(c1_ms),
            )
            frames.append(frame)
            print(
                f"{frame.frame_id}: selected={len(points)} C0={frame.c0_point_count} "
                f"C1={frame.c1_point_count} c1_clamped={frame.c1_clamped_count}"
            )
    return frames


def _metric_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"bias_mm": math.nan, "rmse_mm": math.nan, "p95_abs_mm": math.nan, "peak_to_peak_mm": math.nan}
    return {
        "bias_mm": float(np.mean(values)),
        "rmse_mm": float(np.sqrt(np.mean(values**2))),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95.0)),
        "peak_to_peak_mm": float(np.ptp(values)),
    }


def _simple_linear_diagnostic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or float(np.ptp(x)) <= np.finfo(np.float64).eps:
        return 0.0, float(np.mean(y)) if len(y) else math.nan
    slope, intercept = np.linalg.lstsq(
        np.column_stack([x, np.ones_like(x)]), y, rcond=None
    )[0]
    return float(slope), float(intercept)


def _profile_specs(frames: list[GroundFrame], coordinate: str) -> list[dict[str, float]]:
    by_pose: dict[str, list[GroundFrame]] = {pose: [] for pose in FIT_POSES}
    for frame in frames:
        by_pose.setdefault(frame.pose_id, []).append(frame)
    lows = [float(np.min(frame.coordinates[coordinate].x)) for pose in FIT_POSES for frame in by_pose[pose]]
    highs = [float(np.max(frame.coordinates[coordinate].x)) for pose in FIT_POSES for frame in by_pose[pose]]
    if not lows or max(lows) >= min(highs):
        raise RuntimeError(f"no common coordinate support for {coordinate}")
    low = max(float(np.max([np.min(frame.coordinates[coordinate].x) for frame in by_pose[pose]])) for pose in FIT_POSES)
    high = min(float(np.min([np.max(frame.coordinates[coordinate].x) for frame in by_pose[pose]])) for pose in FIT_POSES)
    if not low < high:
        raise RuntimeError(f"no common pose support for {coordinate}: {low}..{high}")
    edges = np.linspace(low, high, PROFILE_BIN_COUNT + 1, dtype=np.float64)
    return [
        {
            "bin_index": int(index),
            "s_left_mm": float(edges[index]),
            "s_right_mm": float(edges[index + 1]),
            "s_center_mm": float((edges[index] + edges[index + 1]) / 2.0),
        }
        for index in range(PROFILE_BIN_COUNT)
    ]


def _residual_frames(
    frames: list[GroundFrame], coordinate: str, value: str
) -> dict[str, list[ResidualFrame]]:
    result: dict[str, list[ResidualFrame]] = {pose: [] for pose in ALL_POSES}
    for frame in frames:
        fit = frame.coordinates[coordinate]
        array = fit.residual if value == "detrended" else fit.z
        result.setdefault(frame.pose_id, []).append(
            ResidualFrame(
                pose_id=frame.pose_id,
                frame_id=frame.frame_id,
                source_file=frame.path.name,
                source_sha256=frame.file_sha256,
                camera_frame_number=frame.camera_frame_number,
                s=np.asarray(fit.x, dtype=np.float64),
                residual=np.asarray(array, dtype=np.float64),
                quality_passed=frame.quality.get("passed"),
                quality_warnings=[str(item) for item in frame.quality.get("warnings", [])],
                extraction_ms=frame.extraction_ms,
                c0_reconstruction_ms=frame.c0_reconstruction_ms,
                c1_reconstruction_ms=frame.c1_reconstruction_ms,
            )
        )
    return result


def _build_pose_profiles_for_ids(
    frames_by_pose: dict[str, list[ResidualFrame]],
    specs: list[dict[str, float]],
    pose_ids: Iterable[str],
) -> dict[str, PoseProfiles]:
    """Reuse Ground-3's frame/bin semantics for arbitrary pose identifiers."""
    output: dict[str, PoseProfiles] = {}
    for pose_id in pose_ids:
        frames = sorted(
            frames_by_pose.get(pose_id, []),
            key=lambda frame: _natural_key(frame.frame_id),
        )
        if not frames:
            continue
        frame_profiles = np.full(
            (len(frames), len(specs)), np.nan, dtype=np.float64
        )
        frame_counts = np.zeros((len(frames), len(specs)), dtype=np.int64)
        for frame_index, frame in enumerate(frames):
            profile, counts = _bin_frame_residuals(frame.s, frame.residual, specs)
            frame_profiles[frame_index] = profile
            frame_counts[frame_index] = counts
        required = int(math.ceil(FACTORY_MIN_FRAME_FRACTION * len(frames)))
        frame_count_valid = np.sum(np.isfinite(frame_profiles), axis=0).astype(np.int64)
        profile_available = frame_count_valid >= required
        pose_profile = np.full(len(specs), np.nan, dtype=np.float64)
        for bin_index in range(len(specs)):
            values = frame_profiles[:, bin_index]
            values = values[np.isfinite(values)]
            if len(values) >= required:
                pose_profile[bin_index] = float(np.median(values))
        output[pose_id] = PoseProfiles(
            pose_id=pose_id,
            frame_ids=[frame.frame_id for frame in frames],
            frame_profiles=frame_profiles,
            frame_point_counts=frame_counts,
            pose_profile=pose_profile,
            frame_count_valid=frame_count_valid,
            profile_available=profile_available & np.isfinite(pose_profile),
            point_count_total=np.sum(frame_counts, axis=0).astype(np.int64),
        )
    return output


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    best_start = best_end = -1
    start = None
    for index, present in enumerate(np.r_[values, False]):
        if present and start is None:
            start = index
        if not present and start is not None:
            if best_start < 0 or index - start > best_end - best_start:
                best_start, best_end = start, index
            start = None
    result = np.zeros_like(values)
    if best_start >= 0:
        result[best_start:best_end] = True
    return result


def _build_coordinate_profile(frames: list[GroundFrame], coordinate: str) -> CoordinateProfile:
    specs = _profile_specs(frames, coordinate)
    detrended = _build_pose_profiles_for_ids(
        _residual_frames(frames, coordinate, "detrended"), specs, FIT_POSES
    )
    absolute = _build_pose_profiles_for_ids(
        _residual_frames(frames, coordinate, "absolute"), specs, FIT_POSES
    )
    common = _common_bins(detrended, FIT_POSES)
    support = _longest_true_run(common)
    if not np.any(support):
        raise RuntimeError(f"{coordinate} has no contiguous common profile support")
    selected = np.flatnonzero(support)
    domain = (specs[int(selected[0])]["s_left_mm"], specs[int(selected[-1])]["s_right_mm"])
    return CoordinateProfile(
        coordinate=coordinate,
        specs=specs,
        detrended=detrended,
        absolute=absolute,
        common_mask=common,
        support_mask=support,
        support_domain=(float(domain[0]), float(domain[1])),
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return math.nan
    left = left[finite]
    right = right[finite]
    if np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _low_frequency(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = min(LOW_FREQUENCY_WINDOW_BINS, len(values))
    if window < 2:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _coordinate_comparison(profile: CoordinateProfile) -> dict[str, Any]:
    pose_profiles = {
        pose: profile.detrended[pose].pose_profile[profile.support_mask]
        for pose in FIT_POSES
    }
    correlations: list[float] = []
    low_correlations: list[float] = []
    pairwise_rmse: list[float] = []
    for left_index, left_pose in enumerate(FIT_POSES):
        for right_pose in FIT_POSES[left_index + 1 :]:
            left = pose_profiles[left_pose]
            right = pose_profiles[right_pose]
            correlations.append(_safe_corr(left, right))
            low_correlations.append(_safe_corr(_low_frequency(left), _low_frequency(right)))
            finite = np.isfinite(left) & np.isfinite(right)
            pairwise_rmse.append(
                float(np.sqrt(np.mean((left[finite] - right[finite]) ** 2)))
                if np.count_nonzero(finite)
                else math.nan
            )
    frame_repeatability: list[float] = []
    frame_repeatability_std: list[float] = []
    for pose in FIT_POSES:
        pose_profile = profile.detrended[pose].pose_profile
        for row in profile.detrended[pose].frame_profiles:
            finite = profile.support_mask & np.isfinite(row) & np.isfinite(pose_profile)
            difference = row[finite] - pose_profile[finite]
            if len(difference):
                frame_repeatability.append(float(np.sqrt(np.mean(difference**2))))
                frame_repeatability_std.append(float(np.std(difference)))
    coverage = [
        float(np.mean(profile.detrended[pose].profile_available))
        for pose in FIT_POSES
    ]
    finite_corr = [value for value in correlations if math.isfinite(value)]
    finite_low_corr = [value for value in low_correlations if math.isfinite(value)]
    finite_rmse = [value for value in pairwise_rmse if math.isfinite(value)]
    return {
        "coordinate": profile.coordinate,
        "common_support_min": profile.support_domain[0],
        "common_support_max": profile.support_domain[1],
        "common_support_span": profile.support_domain[1] - profile.support_domain[0],
        "common_bin_count": int(np.count_nonzero(profile.support_mask)),
        "raw_common_bin_count": int(np.count_nonzero(profile.common_mask)),
        "profile_bin_count": len(profile.specs),
        "common_support_coverage": float(np.count_nonzero(profile.support_mask) / len(profile.specs)),
        "mean_pose_bin_coverage": float(np.mean(coverage)),
        "min_pose_bin_coverage": float(np.min(coverage)),
        "cross_pose_correlation_mean": float(np.mean(finite_corr)) if finite_corr else math.nan,
        "cross_pose_correlation_min": float(np.min(finite_corr)) if finite_corr else math.nan,
        "low_frequency_correlation_mean": float(np.mean(finite_low_corr)) if finite_low_corr else math.nan,
        "low_frequency_correlation_min": float(np.min(finite_low_corr)) if finite_low_corr else math.nan,
        "profile_rmse_difference_mean_mm": float(np.mean(finite_rmse)) if finite_rmse else math.nan,
        "profile_rmse_difference_max_mm": float(np.max(finite_rmse)) if finite_rmse else math.nan,
        "frame_repeatability_rmse_mean_mm": float(np.mean(frame_repeatability)) if frame_repeatability else math.nan,
        "frame_repeatability_rmse_max_mm": float(np.max(frame_repeatability)) if frame_repeatability else math.nan,
        "frame_repeatability_std_mean_mm": float(np.mean(frame_repeatability_std)) if frame_repeatability_std else math.nan,
        "selection_rule_key": [
            -float(np.mean(finite_low_corr)) if finite_low_corr else math.inf,
            -float(np.mean(finite_corr)) if finite_corr else math.inf,
            float(np.mean(finite_rmse)) if finite_rmse else math.inf,
            float(np.mean(frame_repeatability)) if frame_repeatability else math.inf,
        ],
    }


def _select_coordinate(comparisons: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    ranked = sorted(
        comparisons,
        key=lambda row: tuple(float(value) for value in row["selection_rule_key"]),
    )
    if not ranked:
        raise RuntimeError("no coordinate comparison available")
    chosen = ranked[0]
    chosen_coordinate = str(chosen["coordinate"])
    for row in comparisons:
        row["selected"] = row["coordinate"] == chosen_coordinate
        row["selection_frozen_before_heldout"] = True
    return chosen_coordinate, {
        "rule": [
            "fit-only common-support coverage is computed first",
            "maximize mean low-frequency cross-pose correlation",
            "then maximize cross-pose correlation",
            "then minimize pairwise profile RMSE",
            "then minimize within-pose frame repeatability RMSE",
            "ties use the first coordinate in the fixed order full_v,c1_s",
        ],
        "coordinate_order": list(COORDINATES),
        "chosen_coordinate": chosen_coordinate,
        "comparisons": comparisons,
        "heldout_read_for_selection": False,
    }


def _factory_basis(x: np.ndarray, interior_knot_count: int, domain_min: float, domain_max: float, derivative: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # Ground-3's basis construction is reused as a code framework.  Its G(S)
    # coefficients and smoothness constants are not reused.
    return _basis_matrix(
        np.asarray(x, dtype=np.float64),
        interior_knot_count,
        domain_min,
        domain_max,
        derivative=derivative,
    )


def _factory_observations(
    profile: CoordinateProfile,
    pose_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    weights: list[float] = []
    selected_bins = np.flatnonzero(profile.support_mask)
    for pose_id in pose_ids:
        pose_profile = profile.detrended[pose_id].pose_profile
        finite_bins = [index for index in selected_bins if math.isfinite(float(pose_profile[index]))]
        if not finite_bins:
            continue
        per_observation = 1.0 / (len(pose_ids) * len(finite_bins))
        for index in finite_bins:
            x_values.append(float(profile.specs[index]["s_center_mm"]))
            y_values.append(float(pose_profile[index]))
            weights.append(per_observation)
    if not x_values:
        raise RuntimeError("Factory Profile has no fit observations")
    return (
        np.asarray(x_values, dtype=np.float64),
        np.asarray(y_values, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _fit_factory_spline(
    profile: CoordinateProfile,
    pose_ids: tuple[str, ...],
    interior_knot_count: int,
) -> FactorySpline:
    x, y, weights = _factory_observations(profile, pose_ids)
    domain_min, domain_max = profile.support_domain
    basis, knots = _factory_basis(x, interior_knot_count, domain_min, domain_max)
    grid = np.linspace(domain_min, domain_max, 161, dtype=np.float64)
    curvature, _ = _factory_basis(grid, interior_knot_count, domain_min, domain_max, derivative=2)
    curvature *= (domain_max - domain_min) ** 2
    sqrt_weights = np.sqrt(weights)
    design = np.vstack(
        [sqrt_weights[:, None] * basis, math.sqrt(FACTORY_SMOOTHNESS_LAMBDA) * curvature]
    )
    target = np.concatenate([sqrt_weights * y, np.zeros(len(grid), dtype=np.float64)])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    prediction = basis @ coefficients
    return FactorySpline(
        interior_knot_count=int(interior_knot_count),
        knots=np.asarray(knots, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        domain_min=float(domain_min),
        domain_max=float(domain_max),
        degree=FACTORY_SPLINE_DEGREE,
        smoothness_lambda=FACTORY_SMOOTHNESS_LAMBDA,
        train_pose_ids=pose_ids,
        observation_count=len(x),
        fit_rmse_mm=float(np.sqrt(np.average((prediction - y) ** 2, weights=weights))),
        cv_rmse_mm=None,
    )


def _predict_factory(model: FactorySpline, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    supported = np.isfinite(values) & (values >= model.domain_min) & (values <= model.domain_max)
    prediction = np.full(len(values), np.nan, dtype=np.float64)
    if np.any(supported):
        spline = BSpline(model.knots, model.coefficients, model.degree, extrapolate=False)
        prediction[supported] = np.asarray(spline(values[supported]), dtype=np.float64)
    return prediction, supported


def _fit_factory_candidates(profile: CoordinateProfile) -> tuple[FactorySpline, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for knot_count in FACTORY_INTERIOR_KNOT_COUNTS:
        fold_errors: list[float] = []
        for heldout_pose in FIT_POSES:
            train_poses = tuple(pose for pose in FIT_POSES if pose != heldout_pose)
            model = _fit_factory_spline(profile, train_poses, knot_count)
            x = np.asarray(
                [profile.specs[index]["s_center_mm"] for index in np.flatnonzero(profile.support_mask)],
                dtype=np.float64,
            )
            y = profile.detrended[heldout_pose].pose_profile[profile.support_mask]
            prediction, supported = _predict_factory(model, x)
            finite = supported & np.isfinite(y) & np.isfinite(prediction)
            if np.count_nonzero(finite) < 3:
                fold_errors.append(math.nan)
            else:
                fold_errors.append(float(np.sqrt(np.mean((y[finite] - prediction[finite]) ** 2))))
        finite_folds = [value for value in fold_errors if math.isfinite(value)]
        final_model = _fit_factory_spline(profile, FIT_POSES, knot_count)
        final_model.cv_rmse_mm = float(np.mean(finite_folds)) if finite_folds else math.nan
        candidates.append(
            {
                "interior_knot_count": knot_count,
                "degree": FACTORY_SPLINE_DEGREE,
                "smoothness_lambda": FACTORY_SMOOTHNESS_LAMBDA,
                "fit_rmse_mm": final_model.fit_rmse_mm,
                "leave_one_fit_pose_out_rmse_mm": final_model.cv_rmse_mm,
                "leave_one_fit_pose_out_rmse_by_pose_mm": {
                    pose: value for pose, value in zip(FIT_POSES, fold_errors, strict=True)
                },
                "coefficients_mm": final_model.coefficients,
                "knots": final_model.knots,
                "domain_min": final_model.domain_min,
                "domain_max": final_model.domain_max,
                "observation_count": final_model.observation_count,
            }
        )
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["leave_one_fit_pose_out_rmse_mm"])
            if math.isfinite(float(row["leave_one_fit_pose_out_rmse_mm"]))
            else math.inf,
            int(row["interior_knot_count"]),
        ),
    )
    if not ranked or not math.isfinite(float(ranked[0]["leave_one_fit_pose_out_rmse_mm"])):
        raise RuntimeError("all Factory Profile candidates failed fit-only CV")
    chosen_knot_count = int(ranked[0]["interior_knot_count"])
    chosen = _fit_factory_spline(profile, FIT_POSES, chosen_knot_count)
    chosen.cv_rmse_mm = float(ranked[0]["leave_one_fit_pose_out_rmse_mm"])
    for row in candidates:
        row["selected"] = int(row["interior_knot_count"]) == chosen_knot_count
        row["selection_data_split"] = "fit_001_005_only"
        row["selection_rule"] = "minimum leave-one-fit-pose-out RMSE; tie -> lower knot count"
    return chosen, candidates


def _repeatability_by_frame(profile: CoordinateProfile, frames: list[GroundFrame]) -> dict[tuple[str, str], tuple[float, float]]:
    output: dict[tuple[str, str], tuple[float, float]] = {}
    for pose in sorted({frame.pose_id for frame in frames}):
        if pose not in profile.detrended:
            continue
        pose_profiles = profile.detrended[pose]
        pose_profile = pose_profiles.pose_profile
        for frame_id, frame_values in zip(pose_profiles.frame_ids, pose_profiles.frame_profiles, strict=True):
            finite = profile.support_mask & np.isfinite(pose_profile) & np.isfinite(frame_values)
            difference = frame_values[finite] - pose_profile[finite]
            output[(pose, frame_id)] = (
                float(np.sqrt(np.mean(difference**2))) if len(difference) else math.nan,
                float(np.std(difference)) if len(difference) else math.nan,
            )
    return output


def _frame_metric_rows(
    frames: list[GroundFrame],
    profile_by_coordinate: dict[str, CoordinateProfile],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coordinate, profile in profile_by_coordinate.items():
        repeatability = _repeatability_by_frame(profile, frames)
        for frame in frames:
            fit = frame.coordinates[coordinate]
            raw_metrics = _metric_values(fit.z)
            residual_metrics = _metric_values(fit.residual)
            repeat_rmse, repeat_std = repeatability.get((frame.pose_id, frame.frame_id), (math.nan, math.nan))
            rows.append(
                {
                    "split": frame.split,
                    "pose_id": frame.pose_id,
                    "frame_id": frame.frame_id,
                    "source_file": frame.path.name,
                    "coordinate": coordinate,
                    "point_count": len(fit.x),
                    "c0_point_count": frame.c0_point_count,
                    "c1_point_count": frame.c1_point_count,
                    "board_mask_point_count": frame.c1_selected_count,
                    "support_min": float(np.min(fit.x)),
                    "support_max": float(np.max(fit.x)),
                    "support_span": float(np.ptp(fit.x)),
                    "bias_mm": raw_metrics["bias_mm"],
                    "linear_slope_mm_per_coordinate": fit.slope,
                    "linear_intercept_mm": fit.intercept,
                    "linear_fit_coordinate_scale": fit.linear_fit_coordinate_scale,
                    "fit_rmse_mm": fit.fit_rmse_mm,
                    "fit_inlier_count": int(np.count_nonzero(fit.inlier_mask)),
                    "rmse_mm": raw_metrics["rmse_mm"],
                    "p95_abs_mm": raw_metrics["p95_abs_mm"],
                    "peak_to_peak_mm": raw_metrics["peak_to_peak_mm"],
                    "detrended_bias_mm": residual_metrics["bias_mm"],
                    "detrended_rmse_mm": residual_metrics["rmse_mm"],
                    "detrended_p95_abs_mm": residual_metrics["p95_abs_mm"],
                    "detrended_peak_to_peak_mm": residual_metrics["peak_to_peak_mm"],
                    "repeatability_rmse_mm": repeat_rmse,
                    "repeatability_std_mm": repeat_std,
                    "c1_s_raw_clamped_by_frozen_c1_count": frame.c1_clamped_count if coordinate == "c1_s" else "",
                    "quality_passed": frame.quality.get("passed"),
                    "quality_warnings": ";".join(map(str, frame.quality.get("warnings", []))),
                    "extraction_reused_from_cache": True,
                    "c0_reconstruction_ms": frame.c0_reconstruction_ms,
                    "c1_reconstruction_ms": frame.c1_reconstruction_ms,
                    "mask_source": frame.mask_metadata.get("source"),
                    "mask_residual_used": False,
                }
            )
    return rows


def _profile_rows(
    split: str,
    frames: list[GroundFrame],
    coordinate: str,
    specs: list[dict[str, float]],
    fit_profile: CoordinateProfile,
    factory: FactorySpline | None,
) -> list[dict[str, Any]]:
    pose_ids = tuple(sorted({frame.pose_id for frame in frames}))
    detrended = _build_pose_profiles_for_ids(
        _residual_frames(frames, coordinate, "detrended"), specs, pose_ids
    )
    absolute = _build_pose_profiles_for_ids(
        _residual_frames(frames, coordinate, "absolute"), specs, pose_ids
    )
    rows: list[dict[str, Any]] = []
    for pose in pose_ids:
        for index, spec in enumerate(specs):
            x = float(spec["s_center_mm"])
            factory_value = math.nan
            factory_supported = False
            if factory is not None:
                factory_prediction, factory_mask = _predict_factory(factory, np.array([x], dtype=np.float64))
                factory_value = float(factory_prediction[0]) if factory_mask[0] else math.nan
                factory_supported = bool(factory_mask[0])
            rows.append(
                {
                    "split": split,
                    "pose_id": pose,
                    "coordinate": coordinate,
                    "bin_index": index,
                    "x_left": spec["s_left_mm"],
                    "x_right": spec["s_right_mm"],
                    "x_center": x,
                    "frame_count_valid": int(detrended[pose].frame_count_valid[index]),
                    "frame_coverage_fraction": float(detrended[pose].frame_count_valid[index] / max(1, len(detrended[pose].frame_ids))),
                    "point_count_total": int(detrended[pose].point_count_total[index]),
                    "profile_available": bool(detrended[pose].profile_available[index]),
                    "absolute_profile_zg_mm": absolute[pose].pose_profile[index],
                    "detrended_profile_mm": detrended[pose].pose_profile[index],
                    "factory_profile_mm": factory_value,
                    "factory_support": factory_supported,
                    "fit_common_support_bin": bool(
                        coordinate == fit_profile.coordinate
                        and index < len(fit_profile.support_mask)
                        and fit_profile.support_mask[index]
                    ),
                }
            )
    return rows


def _fit_frame_balanced_linear(frames: list[GroundFrame], coordinate: str, factory: FactorySpline) -> tuple[float, float, int]:
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for frame in frames:
        fit = frame.coordinates[coordinate]
        factory_values, supported = _predict_factory(factory, fit.x)
        supported &= np.isfinite(fit.z) & np.isfinite(factory_values)
        if np.count_nonzero(supported) < 2:
            continue
        x_values.append(fit.x[supported])
        y_values.append(fit.z[supported] - factory_values[supported])
        weights.append(np.full(np.count_nonzero(supported), 1.0 / np.count_nonzero(supported), dtype=np.float64))
    if not x_values:
        raise RuntimeError("held-out pose has no Factory-supported points for C")
    x = np.concatenate(x_values)
    y = np.concatenate(y_values)
    w = np.concatenate(weights)
    w /= float(len(x_values))
    slope, intercept = np.linalg.lstsq(
        np.column_stack([x, np.ones_like(x)]) * np.sqrt(w)[:, None],
        y * np.sqrt(w),
        rcond=None,
    )[0]
    return float(slope), float(intercept), int(len(x))


def _validation_abc_rows(
    frames: list[GroundFrame], coordinate: str, factory: FactorySpline
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pose_linear: dict[str, tuple[float, float, int]] = {}
    for pose in HELDOUT_POSES:
        pose_frames = [frame for frame in frames if frame.pose_id == pose]
        pose_linear[pose] = _fit_frame_balanced_linear(pose_frames, coordinate, factory)
    for frame in frames:
        fit = frame.coordinates[coordinate]
        factory_values, supported = _predict_factory(factory, fit.x)
        supported &= np.isfinite(fit.z) & np.isfinite(factory_values)
        a_pose, b_pose, _ = pose_linear[frame.pose_id]
        chain_values = {
            "A": fit.z[supported],
            "B": fit.z[supported] - factory_values[supported],
            "C": fit.z[supported] - factory_values[supported] - (a_pose * fit.x[supported] + b_pose),
        }
        for chain, values in chain_values.items():
            metrics = _metric_values(values)
            slope, intercept = _simple_linear_diagnostic(fit.x[supported], values)
            rows.append(
                {
                    "aggregation_level": "frame",
                    "split": "validation",
                    "pose_id": frame.pose_id,
                    "frame_id": frame.frame_id,
                    "coordinate": coordinate,
                    "chain": chain,
                    "point_count_board_mask": len(fit.x),
                    "evaluation_point_count": int(np.count_nonzero(supported)),
                    "unsupported_point_count": int(len(fit.x) - np.count_nonzero(supported)),
                    "factory_support_fraction": float(np.mean(supported)),
                    "factory_domain_min": factory.domain_min,
                    "factory_domain_max": factory.domain_max,
                    "bias_mm": metrics["bias_mm"],
                    "linear_slope_mm_per_coordinate": slope,
                    "linear_intercept_mm": intercept,
                    "rmse_mm": metrics["rmse_mm"],
                    "p95_abs_mm": metrics["p95_abs_mm"],
                    "peak_to_peak_mm": metrics["peak_to_peak_mm"],
                    "pose_linear_a_mm_per_coordinate": a_pose if chain == "C" else "",
                    "pose_linear_b_mm": b_pose if chain == "C" else "",
                    "pose_linear_fit_point_count": pose_linear[frame.pose_id][2] if chain == "C" else "",
                    "fit_source_for_C": "heldout_pose_laser_ground_only" if chain == "C" else "",
                    "residual_or_truth_used_for_mask": False,
                }
            )
    for pose in HELDOUT_POSES:
        pose_rows = [row for row in rows if row["pose_id"] == pose]
        for chain in ("A", "B", "C"):
            selected = [row for row in pose_rows if row["chain"] == chain]
            rows.append(
                {
                    "aggregation_level": "pose",
                    "split": "validation",
                    "pose_id": pose,
                    "frame_id": "pose_mean_equal_frame_weight",
                    "coordinate": coordinate,
                    "chain": chain,
                    "point_count_board_mask": int(round(np.mean([row["point_count_board_mask"] for row in selected]))),
                    "evaluation_point_count": int(round(np.mean([row["evaluation_point_count"] for row in selected]))),
                    "unsupported_point_count": int(round(np.mean([row["unsupported_point_count"] for row in selected]))),
                    "factory_support_fraction": float(np.mean([row["factory_support_fraction"] for row in selected])),
                    "factory_domain_min": factory.domain_min,
                    "factory_domain_max": factory.domain_max,
                    "bias_mm": float(np.mean([row["bias_mm"] for row in selected])),
                    "linear_slope_mm_per_coordinate": float(np.mean([row["linear_slope_mm_per_coordinate"] for row in selected])),
                    "linear_intercept_mm": float(np.mean([row["linear_intercept_mm"] for row in selected])),
                    "rmse_mm": float(np.mean([row["rmse_mm"] for row in selected])),
                    "p95_abs_mm": float(np.mean([row["p95_abs_mm"] for row in selected])),
                    "peak_to_peak_mm": float(np.mean([row["peak_to_peak_mm"] for row in selected])),
                    "pose_linear_a_mm_per_coordinate": pose_linear[pose][0] if chain == "C" else "",
                    "pose_linear_b_mm": pose_linear[pose][1] if chain == "C" else "",
                    "pose_linear_fit_point_count": pose_linear[pose][2] if chain == "C" else "",
                    "fit_source_for_C": "heldout_pose_laser_ground_only" if chain == "C" else "",
                    "residual_or_truth_used_for_mask": False,
                }
            )
    return rows, {
        pose: {"a": values[0], "b": values[1], "fit_point_count": values[2]}
        for pose, values in pose_linear.items()
    }


def _profile_prediction_metrics(
    profile_rows: list[dict[str, Any]],
    pose: str,
    coordinate: str,
    factory: FactorySpline,
) -> dict[str, float]:
    rows = [
        row for row in profile_rows
        if row["split"] == "validation"
        and row["pose_id"] == pose
        and row["coordinate"] == coordinate
        and row["factory_support"]
        and row.get("detrended_profile_mm") not in (None, "")
        and row.get("factory_profile_mm") not in (None, "")
    ]
    observed = np.asarray([row["detrended_profile_mm"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["factory_profile_mm"] for row in rows], dtype=np.float64)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if np.count_nonzero(finite) < 3:
        return {"common_bin_count": int(np.count_nonzero(finite)), "shape_correlation": math.nan, "rmse_mm": math.nan, "bias_mm": math.nan}
    error = observed[finite] - predicted[finite]
    return {
        "common_bin_count": int(np.count_nonzero(finite)),
        "shape_correlation": _safe_corr(observed[finite], predicted[finite]),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "bias_mm": float(np.mean(error)),
    }


def _save_board_overlay(
    path: Path,
    pose: PoseRecord,
    frames: list[GroundFrame],
    pnp: PnPRecord,
    title: str,
) -> None:
    image = _read_image(pose.chess_path)
    fig, axis = plt.subplots(figsize=(16, 11))
    low, high = np.percentile(image.astype(np.float32), [0.1, 99.9])
    axis.imshow(np.clip((image - low) * 255.0 / max(high - low, 1.0), 0, 255), cmap="gray", vmin=0, vmax=255)
    polygon = np.asarray(frames[0].mask_metadata["polygon_full_uv"], dtype=np.float64)
    closed = np.vstack([polygon, polygon[0]])
    axis.plot(closed[:, 0], closed[:, 1], "r-", linewidth=2.0, label="PnP full physical-board polygon")
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(frames)))
    for colour, frame in zip(colours, frames, strict=True):
        axis.scatter(frame.valid_pixels_uv[:, 0], frame.valid_pixels_uv[:, 1], s=2.0, alpha=0.35, color=colour, label=frame.frame_id)
        axis.scatter(frame.valid_pixels_uv[:, 0], frame.valid_pixels_uv[:, 1], s=3.0, alpha=0.70, color=colour)
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("full-sensor u (px)")
    axis.set_ylabel("full-sensor v (px)")
    axis.set_title(f"{title}; PnP reprojection RMSE={pnp.reprojection_rmse_px:.4f}px")
    axis.legend(loc="upper right", fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_profile_overlay(
    path: Path,
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    factory: FactorySpline | None = None,
    coordinate: str | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(13, 7))
    styles = {"fit": "-", "validation": "--"}
    colours = {pose: colour for pose, colour in zip(ALL_POSES, plt.cm.tab10(np.linspace(0, 0.9, len(ALL_POSES))), strict=True)}
    plotted: set[tuple[str, str]] = set()
    for row in rows:
        if coordinate is not None and row["coordinate"] != coordinate:
            continue
        value = row.get(value_key)
        if value in (None, "") or not math.isfinite(float(value)):
            continue
        key = (str(row["split"]), str(row["pose_id"]))
        label = f"{row['split']} pose{row['pose_id']}"
        axis.plot(
            [float(row["x_center"])], [float(value)],
            marker="o", markersize=2.5, linestyle="none", color=colours[row["pose_id"]],
            alpha=0.7, label=label if key not in plotted else None,
        )
        plotted.add(key)
    if factory is not None:
        grid = np.linspace(factory.domain_min, factory.domain_max, 300)
        prediction, supported = _predict_factory(factory, grid)
        axis.plot(grid[supported], prediction[supported], "k-", linewidth=2.0, label="Frozen Factory F(x)")
    axis.set_title(title)
    axis.set_xlabel("Factory coordinate x (full_v px or C1 s_ray)")
    axis.set_ylabel("Zg (mm)" if value_key == "absolute_profile_zg_mm" else "detrended residual r (mm)")
    axis.grid(True, alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_abc(path: Path, rows: list[dict[str, Any]], coordinate: str) -> None:
    pose_rows = [row for row in rows if row["aggregation_level"] == "pose"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    chains = ("A", "B", "C")
    colours = {"A": "tab:gray", "B": "tab:blue", "C": "tab:orange"}
    x = np.arange(len(HELDOUT_POSES))
    width = 0.23
    for offset, chain in enumerate(chains):
        values = [
            next(row["rmse_mm"] for row in pose_rows if row["pose_id"] == pose and row["chain"] == chain)
            for pose in HELDOUT_POSES
        ]
        axes[0].bar(x + (offset - 1) * width, values, width, label=chain, color=colours[chain])
        values_p95 = [
            next(row["p95_abs_mm"] for row in pose_rows if row["pose_id"] == pose and row["chain"] == chain)
            for pose in HELDOUT_POSES
        ]
        axes[1].bar(x + (offset - 1) * width, values_p95, width, label=chain, color=colours[chain])
    for axis, ylabel in zip(axes, ("RMSE (mm)", "P95 abs (mm)"), strict=True):
        axis.set_xticks(x, [f"pose{pose}" for pose in HELDOUT_POSES])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle(f"Held-out A/B/C comparison ({coordinate}; common Factory support)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_abc_residuals(path: Path, frames: list[GroundFrame], coordinate: str, factory: FactorySpline, pose_linear: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(len(HELDOUT_POSES), 1, figsize=(13, 8), sharex=False)
    if len(HELDOUT_POSES) == 1:
        axes = [axes]
    for axis, pose in zip(axes, HELDOUT_POSES, strict=True):
        for frame in [item for item in frames if item.pose_id == pose]:
            fit = frame.coordinates[coordinate]
            f, supported = _predict_factory(factory, fit.x)
            supported &= np.isfinite(fit.z) & np.isfinite(f)
            x = fit.x[supported]
            a = pose_linear[pose]["a"]
            b = pose_linear[pose]["b"]
            axis.plot(x, fit.z[supported], color="0.65", alpha=0.35, linewidth=0.7)
            axis.plot(x, fit.z[supported] - f[supported], color="tab:blue", alpha=0.55, linewidth=0.8)
            axis.plot(x, fit.z[supported] - f[supported] - (a * x + b), color="tab:orange", alpha=0.65, linewidth=0.8)
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set_title(f"pose{pose}: A raw Zg / B Zg-F / C Zg-F-session-linear")
        axis.set_ylabel("residual (mm)")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Factory coordinate x")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _classification(
    pnp_by_pose: dict[str, PnPRecord],
    frames: list[GroundFrame],
    comparisons: list[dict[str, Any]],
    selected_coordinate: str,
    factory: FactorySpline,
    validation_rows: list[dict[str, Any]],
    validation_profile_metrics: dict[str, dict[str, float]],
) -> dict[str, str]:
    selection_ok = (
        len(pnp_by_pose) == len(ALL_POSES)
        and all(math.isfinite(item.reprojection_rmse_px) and item.reprojection_rmse_px <= 0.5 for item in pnp_by_pose.values())
        and len(frames) == 35
        and all(frame.c1_selected_count >= 20 and frame.mask_metadata.get("source") == "pnp_board_mask" for frame in frames)
    )
    selection_status = "PASS" if selection_ok else "FAIL"
    selected_comparison = next(row for row in comparisons if row["coordinate"] == selected_coordinate)
    profile_ok = (
        float(selected_comparison["common_support_coverage"]) >= FACTORY_MIN_COMMON_BIN_FRACTION
        and math.isfinite(float(factory.cv_rmse_mm))
        and math.isfinite(float(selected_comparison["low_frequency_correlation_mean"]))
        and float(selected_comparison["low_frequency_correlation_mean"]) >= HELDOUT_SHAPE_CORRELATION_PASS
        and math.isfinite(float(selected_comparison["frame_repeatability_rmse_mean_mm"]))
    )
    profile_partial = (
        math.isfinite(float(factory.cv_rmse_mm))
        and float(selected_comparison["common_support_coverage"]) > 0.0
    )
    profile_status = "PASS" if profile_ok else "PARTIAL" if profile_partial else "FAIL"
    pose_summary = {
        pose: {chain: next(row for row in validation_rows if row["aggregation_level"] == "pose" and row["pose_id"] == pose and row["chain"] == chain) for chain in ("A", "B", "C")}
        for pose in HELDOUT_POSES
    }
    improvements_b: list[float] = []
    improvements_c: list[float] = []
    for pose in HELDOUT_POSES:
        a_rmse = float(pose_summary[pose]["A"]["rmse_mm"])
        b_rmse = float(pose_summary[pose]["B"]["rmse_mm"])
        c_rmse = float(pose_summary[pose]["C"]["rmse_mm"])
        improvements_b.append(1.0 - b_rmse / a_rmse if a_rmse > 0.0 else math.nan)
        improvements_c.append(1.0 - c_rmse / b_rmse if b_rmse > 0.0 else math.nan)
    support_ok = all(float(pose_summary[pose]["B"]["factory_support_fraction"]) >= FACTORY_MIN_COMMON_BIN_FRACTION for pose in HELDOUT_POSES)
    shape_ok = all(
        math.isfinite(validation_profile_metrics[pose]["shape_correlation"])
        and validation_profile_metrics[pose]["shape_correlation"] >= HELDOUT_SHAPE_CORRELATION_PASS
        for pose in HELDOUT_POSES
    )
    b_ok = all(math.isfinite(value) and value >= HELDOUT_IMPROVEMENT_PASS for value in improvements_b)
    b_partial = any(math.isfinite(value) and value > 0.0 for value in improvements_b)
    heldout_status = "PASS" if support_ok and shape_ok and b_ok else "PARTIAL" if support_ok and b_partial else "FAIL"
    c_yes = all(math.isfinite(value) and value >= SESSION_LINEAR_IMPROVEMENT_YES for value in improvements_c)
    c_no = all(math.isfinite(value) and value <= SESSION_LINEAR_IMPROVEMENT_NO for value in improvements_c)
    session_linear = "YES" if c_yes else "NO" if c_no else "UNCERTAIN"
    return {
        "GROUND_POINT_SELECTION": selection_status,
        "RECOMMENDED_COORDINATE": selected_coordinate,
        "FACTORY_PROFILE_FIT_STABILITY": profile_status,
        "HELDOUT_FACTORY_PROFILE": heldout_status,
        "SESSION_LINEAR_NEEDED": session_linear,
    }


def _report_number(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "—" if not math.isfinite(number) else f"{number:.{digits}g}"


def _write_report(path: Path, summary: dict[str, Any], output_files: list[str]) -> None:
    classification = summary["classification"]
    steger_cache_reused = bool(summary["protocol"].get("steger_cache_reused_existing", False))
    steger_provenance_line = (
        "- Steger centers were loaded from the protocol-compatible cache; no Steger rerun was needed for this final generation."
        if steger_cache_reused
        else "- One Steger extraction per laser TIFF was newly computed and stored in the local cache."
    )
    lines = [
        "# Ground-5A｜Frozen Factory Ground Profile 同 Session Held-out 验证",
        "",
        f"数据集：`chessboard_0821`；本轮生成时间：`{summary['created_at_local']}`。",
        "",
        "## 最终结论",
        "",
        f"- `GROUND_POINT_SELECTION = {classification['GROUND_POINT_SELECTION']}`",
        f"- `RECOMMENDED_COORDINATE = {classification['RECOMMENDED_COORDINATE']}`",
        f"- `FACTORY_PROFILE_FIT_STABILITY = {classification['FACTORY_PROFILE_FIT_STABILITY']}`",
        f"- `HELDOUT_FACTORY_PROFILE = {classification['HELDOUT_FACTORY_PROFILE']}`",
        f"- `SESSION_LINEAR_NEEDED = {classification['SESSION_LINEAR_NEEDED']}`",
        "",
        "## Protocol lock",
        "",
        "- Fit poses: `001–005`; held-out poses: `006–007`; no validation value was read for coordinate/model selection.",
        "- Full-sensor image coordinates are retained. `full_v` is raw sensor row `v`; `c1_s` is Frozen C1 raw PCA ray coordinate `s_raw`.",
        "- PnP: existing Session PnP implementation with board pattern `[11, 8]`, square size `20 mm`.",
        "- Mask: existing `pnp_board_mask/full_board_physical`; selection is polygon-only and uses no Z residual.",
        "- Reconstruction: same cached centers to frozen C0 and frozen C1; per-pose PnP `R,t` is used as camera-to-ground transform.",
        "- H1/Stage-A: disabled for analysis; Session Ground Reference: not fitted/applied; Ground-3 G(S) parameters: not reused.",
        "- Factory profile target: frame-detrended `r=Zg-(a*x+b)`, cubic B-spline candidates with 1/2/3 interior knots, equal pose weighting, no extrapolation and no clamp.",
        "",
        "## Artifact provenance / reuse audit",
        "",
        "### Reused implementation (not reused results)",
        "",
        "- Existing Steger adapter and full-sensor reconstruction entry points.",
        "- Existing Session PnP and checkerboard physical-board mask.",
        "- Existing robust linear ground-profile kernel through a one-dimensional adapter.",
        "- Ground-3 frame-median/pose-median bin aggregation and spline-basis construction as code framework only.",
        "",
        "### Reused artifacts",
        "",
        "- `chessboard_0821` manifest/frame split, source TIFFs and their recorded SHA-256 values.",
        "- Frozen Daheng C0/C1 calibration package and its provenance; no C0/C1 refit.",
        "- On reruns, the local one-Steger cache is reused only when source SHA, extraction options and config SHA match.",
        "",
        "### This generation",
        "",
        "- Per-pose PnP, board-mask selections, PnP-ground C0/C1 points.",
        steger_provenance_line,
        "- Fit-only coordinate comparison, frozen coordinate decision, fit-only Factory candidates and final profile.",
        "- Held-out A/B/C metrics and plots for 006/007.",
        "",
        "## PnP and point selection",
        "",
        "| split | pose | chess PnP RMSE (px) | detection | selected frames | min selected points |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for pose in ALL_POSES:
        pnp = summary["pnp"][pose]
        frame_rows = [row for row in summary["frame_metrics"] if row["pose_id"] == pose and row["coordinate"] == summary["coordinate_freeze"]["chosen_coordinate"]]
        lines.append(
            f"| {pnp['split']} | {pose} | {_report_number(pnp['reprojection_rmse_px'])} | {pnp['detection_method']} | "
            f"{len(frame_rows)} | {min(int(row['board_mask_point_count']) for row in frame_rows)} |"
        )
    lines.extend(["", "## Fit-only coordinate comparison", "", "| coordinate | common coverage | cross-pose corr | low-frequency corr | profile RMSE diff (mm) | frame repeatability RMSE (mm) | selected |", "|---|---:|---:|---:|---:|---:|---|"])
    for row in summary["coordinate_comparison"]:
        lines.append(
            f"| {row['coordinate']} | {_report_number(row['common_support_coverage'])} | {_report_number(row['cross_pose_correlation_mean'])} | "
            f"{_report_number(row['low_frequency_correlation_mean'])} | {_report_number(row['profile_rmse_difference_mean_mm'])} | "
            f"{_report_number(row['frame_repeatability_rmse_mean_mm'])} | {'YES' if row.get('selected') else 'NO'} |"
        )
    lines.extend([
        "",
        "Selection was frozen after this table was produced from 001–005 only; `006/007` did not enter any coordinate or Factory candidate rule.",
        "",
        "## Factory Profile candidate",
        "",
        f"- Coordinate: `{summary['factory_profile']['coordinate']}`",
        f"- Fit support domain: `[{_report_number(summary['factory_profile']['domain_min'])}, {_report_number(summary['factory_profile']['domain_max'])}]`",
        f"- Support bins: `{summary['factory_profile']['support_bin_count']}/{summary['factory_profile']['profile_bin_count']}`",
        f"- Selected interior knots: `{summary['factory_profile']['interior_knot_count']}`",
        f"- Fit-only leave-one-pose-out RMSE: `{_report_number(summary['factory_profile']['cv_rmse_mm'])} mm`",
        "- Model selection: minimum fit-only leave-one-pose-out RMSE; tie goes to lower knot count.",
        "- Unsupported coordinates are rejected/marked unsupported; no interpolation, extrapolation or clamp is used by Factory F(x).",
        "",
        "| candidate knots | fit RMSE (mm) | fit-pose LPO CV RMSE (mm) | selected |",
        "|---:|---:|---:|---|"])
    for row in summary["factory_profile"]["candidates"]:
        lines.append(f"| {row['interior_knot_count']} | {_report_number(row['fit_rmse_mm'])} | {_report_number(row['leave_one_fit_pose_out_rmse_mm'])} | {'YES' if row.get('selected') else 'NO'} |")
    lines.extend([
        "",
        "## Held-out A/B/C",
        "",
        "RMSE/P95 improvements are relative reductions on the same strict Factory support domain.",
        "",
        "| pose | A RMSE | B RMSE | C RMSE | B vs A | C vs B | A P95 | B P95 | C P95 | B vs A P95 | C vs B P95 | B support | detrended-shape corr |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for pose in HELDOUT_POSES:
        pose_rows = {row["chain"]: row for row in summary["validation_abc"] if row["aggregation_level"] == "pose" and row["pose_id"] == pose}
        a = pose_rows["A"]["rmse_mm"]
        b = pose_rows["B"]["rmse_mm"]
        c = pose_rows["C"]["rmse_mm"]
        a_p95 = pose_rows["A"]["p95_abs_mm"]
        b_p95 = pose_rows["B"]["p95_abs_mm"]
        c_p95 = pose_rows["C"]["p95_abs_mm"]
        lines.append(
            f"| {pose} | {_report_number(a)} | {_report_number(b)} | {_report_number(c)} | "
            f"{_report_number(1-b/a if a else math.nan)} | {_report_number(1-c/b if b else math.nan)} | "
            f"{_report_number(a_p95)} | {_report_number(b_p95)} | {_report_number(c_p95)} | "
            f"{_report_number(1-b_p95/a_p95 if a_p95 else math.nan)} | {_report_number(1-c_p95/b_p95 if b_p95 else math.nan)} | "
            f"{_report_number(pose_rows['B']['factory_support_fraction'])} | {_report_number(summary['validation_profile_prediction'][pose]['shape_correlation'])} |"
        )
    lines.extend([
        "",
        "A and B are compared on the same strict Factory support domain. C's `a,b` are fitted separately for each held-out pose using only that pose's supported laser ground points; they are not fed back into F(x).",
        "",
        "## Interpretation",
        "",
        f"- Absolute profile prediction: see A→B RMSE/P95 deltas above and `absolute_profile_overlay.png`. The Factory model is a detrended spatial profile, so pose zero/tilt remains visible in B when present.",
        f"- Nonlinear detrended shape: pose006 correlation `{_report_number(summary['validation_profile_prediction']['006']['shape_correlation'])}`, pose007 correlation `{_report_number(summary['validation_profile_prediction']['007']['shape_correlation'])}` against frozen F(x).",
        f"- Session linear diagnostic: C→B improvement is reported per held-out pose; classification uses predeclared thresholds `{SESSION_LINEAR_IMPROVEMENT_YES}` / `{SESSION_LINEAR_IMPROVEMENT_NO}`.",
        "",
        "## Outputs",
        "",
    ])
    lines.extend(f"- `{name}`" for name in output_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    fit_dir = args.fit_dir.resolve()
    validation_dir = args.validation_dir.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_records = _discover_pose_records(fit_dir, "fit", FIT_POSES)
    validation_records = _discover_pose_records(validation_dir, "validation", HELDOUT_POSES)
    all_records = fit_records + validation_records

    dataset_document, metadata_by_name, manifest_path, frames_csv_path = _load_manifest_context(fit_dir)
    app = load_app_config(config_path)
    if app.reconstruction.image_roi_polygon is not None:
        raise RuntimeError("Ground-5A requires reconstruction.image_roi_polygon=null")
    if not app.reconstruction.enable_laser_ray_correction:
        raise RuntimeError("Ground-5A requires the existing frozen C1 path to be enabled")
    if app.calibration.manifest is None:
        raise RuntimeError("Ground-5A requires calibration.manifest")
    package = load_calibration_package(app.calibration.manifest)
    base_calibration = dict(package.calibration)
    params_c0 = replace(app.reconstruction, enable_laser_ray_correction=False)
    params_c1 = app.reconstruction
    board_config = app.session_ground_calibration.board_config()
    mask_inset_mm = float(app.session_ground_calibration.sanity.mask_inset_mm)
    if mask_inset_mm != float(app.session_ground_calibration.ground_reference.mask_inset_mm):
        raise RuntimeError("Ground-5A requires Session sanity and ground-reference mask inset to agree")

    centers_by_path, cache = _cache_records(all_records, output_dir, app, config_path)

    # Phase 1: fit-only PnP, reconstruction and coordinate selection.
    fit_pnp = _run_pnp(fit_records, base_calibration, board_config)
    fit_frames = _process_frames(
        fit_records,
        fit_pnp,
        centers_by_path,
        metadata_by_name,
        base_calibration,
        params_c0,
        params_c1,
        board_config,
        mask_inset_mm,
        app.measurement,
    )
    fit_profiles = {coordinate: _build_coordinate_profile(fit_frames, coordinate) for coordinate in COORDINATES}
    coordinate_comparison = [_coordinate_comparison(fit_profiles[coordinate]) for coordinate in COORDINATES]
    chosen_coordinate, coordinate_freeze = _select_coordinate(coordinate_comparison)
    factory, factory_candidates = _fit_factory_candidates(fit_profiles[chosen_coordinate])

    # Phase 2 starts only after coordinate and Factory candidate are frozen.
    validation_pnp = _run_pnp(validation_records, base_calibration, board_config)
    validation_frames = _process_frames(
        validation_records,
        validation_pnp,
        centers_by_path,
        metadata_by_name,
        base_calibration,
        params_c0,
        params_c1,
        board_config,
        mask_inset_mm,
        app.measurement,
    )
    validation_profiles: dict[str, CoordinateProfile] = {}
    for coordinate in COORDINATES:
        validation_profiles[coordinate] = CoordinateProfile(
            coordinate=coordinate,
            specs=fit_profiles[coordinate].specs,
            detrended=_build_pose_profiles_for_ids(
                _residual_frames(validation_frames, coordinate, "detrended"),
                fit_profiles[coordinate].specs,
                HELDOUT_POSES,
            ),
            absolute=_build_pose_profiles_for_ids(
                _residual_frames(validation_frames, coordinate, "absolute"),
                fit_profiles[coordinate].specs,
                HELDOUT_POSES,
            ),
            common_mask=np.zeros(len(fit_profiles[coordinate].specs), dtype=bool),
            support_mask=fit_profiles[coordinate].support_mask,
            support_domain=fit_profiles[coordinate].support_domain,
        )
    all_frames = fit_frames + validation_frames
    frame_metrics = _frame_metric_rows(
        fit_frames,
        fit_profiles,
    ) + _frame_metric_rows(
        validation_frames,
        validation_profiles,
    )
    profile_rows = []
    for coordinate in COORDINATES:
        profile_rows.extend(_profile_rows("fit", fit_frames, coordinate, fit_profiles[coordinate].specs, fit_profiles[coordinate], factory if coordinate == chosen_coordinate else None))
    for coordinate in COORDINATES:
        profile_rows.extend(
            _profile_rows(
                "validation",
                validation_frames,
                coordinate,
                fit_profiles[coordinate].specs,
                fit_profiles[coordinate],
                factory if coordinate == chosen_coordinate else None,
            )
        )
    validation_abc, pose_linear = _validation_abc_rows(validation_frames, chosen_coordinate, factory)
    validation_profile_prediction = {
        pose: _profile_prediction_metrics(profile_rows, pose, chosen_coordinate, factory)
        for pose in HELDOUT_POSES
    }
    pnp_all = {**fit_pnp, **validation_pnp}
    pnp_json = {
        pose: {
            "split": pnp_all[pose].split,
            "chess_path": pnp_all[pose].chess_path,
            "reprojection_rmse_px": pnp_all[pose].reprojection_rmse_px,
            "detection_method": pnp_all[pose].detection_method,
            "rvec_board_to_camera": pnp_all[pose].result.rvec,
            "tvec_board_to_camera": pnp_all[pose].result.tvec,
            "R_camera_to_ground": pnp_all[pose].result.R,
            "t_camera_to_ground": pnp_all[pose].result.t,
            "corner_count": len(pnp_all[pose].result.detected_corners),
        }
        for pose in ALL_POSES
    }
    classification = _classification(
        pnp_all,
        all_frames,
        coordinate_comparison,
        chosen_coordinate,
        factory,
        validation_abc,
        validation_profile_prediction,
    )

    candidate_json = {
        "schema_version": 1,
        "status": "diagnostic_frozen_candidate",
        "coordinate": chosen_coordinate,
        "coordinate_freeze": coordinate_freeze,
        "fit_pose_ids": list(FIT_POSES),
        "heldout_pose_ids": list(HELDOUT_POSES),
        "profile_target": "per_frame_detrended_residual_r=Zg-(a*x+b)",
        "profile_bin_count": PROFILE_BIN_COUNT,
        "support_domain": [factory.domain_min, factory.domain_max],
        "support_bin_count": int(np.count_nonzero(fit_profiles[chosen_coordinate].support_mask)),
        "support_bin_indices": np.flatnonzero(fit_profiles[chosen_coordinate].support_mask),
        "degree": factory.degree,
        "interior_knot_count": factory.interior_knot_count,
        "knots": factory.knots,
        "coefficients_mm": factory.coefficients,
        "smoothness_lambda": factory.smoothness_lambda,
        "fit_rmse_mm": factory.fit_rmse_mm,
        "leave_one_fit_pose_out_cv_rmse_mm": factory.cv_rmse_mm,
        "candidates": factory_candidates,
        "train_pose_weighting": "each pose total weight=1/5; each pose bin observations share its total weight",
        "frame_weighting": "frame median per bin, then pose median across frames; minimum frame fraction=0.8",
        "extrapolation_policy": "reject_and_mark_unsupported",
        "interpolation_policy": "none",
        "clamp_policy": "forbidden",
        "ground3_numeric_parameters_reused": False,
    }
    (output_dir / "factory_profile_candidate.json").write_text(
        json.dumps(_json_ready(candidate_json), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "coordinate_freeze.json").write_text(
        json.dumps(_json_ready(coordinate_freeze), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    output_files = [
        "ground5a_report.md",
        "frame_metrics.csv",
        "pose_metrics.csv",
        "coordinate_comparison.csv",
        "fit_profile_by_pose.csv",
        "validation_abc_comparison.csv",
        "factory_profile_candidate.json",
        "coordinate_freeze.json",
        "absolute_profile_overlay.png",
        "detrended_profile_overlay.png",
        "heldout_abc_comparison.png",
        "heldout_abc_residual_overlay.png",
        "steger_geometry_cache.json",
    ]
    for pose in ALL_POSES:
        output_files.append(f"pose{pose}_board_mask_overlay.png")

    frame_fields = list(frame_metrics[0])
    _write_csv(output_dir / "frame_metrics.csv", frame_metrics, frame_fields)
    pose_metric_rows: list[dict[str, Any]] = []
    for split, frames_for_split in (("fit", fit_frames), ("validation", validation_frames)):
        for pose in (FIT_POSES if split == "fit" else HELDOUT_POSES):
            pose_frames = [frame for frame in frames_for_split if frame.pose_id == pose]
            for coordinate in COORDINATES:
                selected = [row for row in frame_metrics if row["split"] == split and row["pose_id"] == pose and row["coordinate"] == coordinate]
                pose_metric_rows.append(
                    {
                        "split": split,
                        "pose_id": pose,
                        "coordinate": coordinate,
                        "frame_count": len(selected),
                        "point_count_mean": float(np.mean([row["point_count"] for row in selected])),
                        "support_min_min": float(np.min([row["support_min"] for row in selected])),
                        "support_max_max": float(np.max([row["support_max"] for row in selected])),
                        "bias_mean_mm": float(np.mean([row["bias_mm"] for row in selected])),
                        "linear_slope_mean": float(np.mean([row["linear_slope_mm_per_coordinate"] for row in selected])),
                        "linear_intercept_mean_mm": float(np.mean([row["linear_intercept_mm"] for row in selected])),
                        "rmse_mean_mm": float(np.mean([row["rmse_mm"] for row in selected])),
                        "p95_abs_mean_mm": float(np.mean([row["p95_abs_mm"] for row in selected])),
                        "peak_to_peak_mean_mm": float(np.mean([row["peak_to_peak_mm"] for row in selected])),
                        "detrended_rmse_mean_mm": float(np.mean([row["detrended_rmse_mm"] for row in selected])),
                        "detrended_p95_abs_mean_mm": float(np.mean([row["detrended_p95_abs_mm"] for row in selected])),
                        "repeatability_rmse_mean_mm": float(np.mean([row["repeatability_rmse_mm"] for row in selected if math.isfinite(float(row["repeatability_rmse_mm"]))])),
                        "board_mask_selected_min": int(np.min([row["board_mask_point_count"] for row in selected])),
                    }
                )
    _write_csv(output_dir / "pose_metrics.csv", pose_metric_rows, list(pose_metric_rows[0]))
    _write_csv(output_dir / "coordinate_comparison.csv", coordinate_comparison, list(coordinate_comparison[0]))
    _write_csv(output_dir / "fit_profile_by_pose.csv", profile_rows, list(profile_rows[0]))
    _write_csv(output_dir / "validation_abc_comparison.csv", validation_abc, list(validation_abc[0]))

    for pose in ALL_POSES:
        pose_record = next(record for record in all_records if record.pose_id == pose)
        pose_frames = [frame for frame in all_frames if frame.pose_id == pose]
        _save_board_overlay(
            output_dir / f"pose{pose}_board_mask_overlay.png",
            pose_record,
            pose_frames,
            pnp_all[pose],
            f"pose{pose}: physical-board mask and C1-valid laser points",
        )
    _plot_profile_overlay(
        output_dir / "absolute_profile_overlay.png",
        profile_rows,
        "absolute_profile_zg_mm",
        f"Absolute PnP-ground profile overlay ({chosen_coordinate})",
        None,
        chosen_coordinate,
    )
    _plot_profile_overlay(
        output_dir / "detrended_profile_overlay.png",
        profile_rows,
        "detrended_profile_mm",
        f"Detrended profile overlay and frozen Factory F(x) ({chosen_coordinate})",
        factory,
        chosen_coordinate,
    )
    _plot_abc(output_dir / "heldout_abc_comparison.png", validation_abc, chosen_coordinate)
    _plot_abc_residuals(output_dir / "heldout_abc_residual_overlay.png", validation_frames, chosen_coordinate, factory, pose_linear)

    frame_sha = {
        str(frame.path.resolve()): frame.file_sha256 for frame in all_frames
    }
    steger_cache_reused = bool(cache.get("reused_existing_cache", False))
    reused_artifacts = [
        "chessboard_0821 manifest and source TIFF SHA-256 records",
        "frozen Daheng calibration package and C1 parameters",
        "compatible prior center cache on rerun only when provenance key matches",
    ]
    newly_computed = [
        "PnP and physical-board masks for poses001-007",
        "PnP-ground Frozen C0/C1 points and frame metrics",
        "fit-only coordinate comparison and Factory candidate",
        "held-out A/B/C evaluation and plots",
    ]
    if steger_cache_reused:
        reused_artifacts.append("protocol-compatible one-Steger center cache for 35 laser TIFFs")
    else:
        newly_computed.append("one-Steger center cache for 35 laser TIFFs")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "dataset_id": dataset_document.get("dataset_id"),
            "manifest_path": manifest_path,
            "frames_csv_path": frames_csv_path,
            "dataset_status": dataset_document.get("status"),
            "quality_summary": dataset_document.get("quality_summary"),
            "fit_dir": fit_dir,
            "validation_dir": validation_dir,
            "fit_pose_ids": FIT_POSES,
            "heldout_pose_ids": HELDOUT_POSES,
            "source_sha256_by_path": frame_sha,
        },
        "configuration": {
            "config_path": config_path,
            "config_sha256": _sha256_file(config_path),
            "calibration_manifest": app.calibration.manifest,
            "calibration_package_id": package.package_id,
            "calibration_package_manifest_sha256": package.manifest_sha256,
            "frozen_c1_source": app.calibration.laser_ray_correction,
            "frozen_c1_enabled": True,
            "c0_refit": False,
            "c1_refit": False,
            "h1_or_stage_a_applied": False,
            "ground_u_compensation_applied": False,
            "session_ground_reference_applied": False,
            "ground3_gs_parameters_reused": False,
            "analysis_code_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "protocol": {
            "one_steger_per_laser_tiff": True,
            "steger_cache_path": output_dir / "steger_geometry_cache.json",
            "steger_cache_reused_existing": steger_cache_reused,
            "same_centers_to_frozen_c0_and_c1": True,
            "pnp_board_config": {
                "pattern_cols": board_config.pattern_cols,
                "pattern_rows": board_config.pattern_rows,
                "square_size_mm": board_config.square_size_mm,
                "detector": board_config.detector,
            },
            "physical_mask": "full_board_physical_polygon",
            "mask_inset_mm": mask_inset_mm,
            "mask_uses_z_residual": False,
            "full_sensor_v": True,
            "c1_s_definition": "Frozen C1 evaluate_frozen_laser_ray_correction.s_raw from full-sensor undistorted rays",
            "factory_profile_no_extrapolation": True,
            "factory_profile_no_clamp": True,
            "factory_profile_no_interpolation": True,
            "validation_formal_metrics_after_coordinate_freeze": True,
        },
        "classification": classification,
        "pnp": pnp_json,
        "coordinate_freeze": coordinate_freeze,
        "coordinate_comparison": coordinate_comparison,
        "factory_profile": {
            "coordinate": chosen_coordinate,
            "degree": factory.degree,
            "interior_knot_count": factory.interior_knot_count,
            "knots": factory.knots,
            "coefficients_mm": factory.coefficients,
            "domain_min": factory.domain_min,
            "domain_max": factory.domain_max,
            "support_bin_count": int(np.count_nonzero(fit_profiles[chosen_coordinate].support_mask)),
            "profile_bin_count": len(fit_profiles[chosen_coordinate].specs),
            "fit_rmse_mm": factory.fit_rmse_mm,
            "cv_rmse_mm": factory.cv_rmse_mm,
            "candidates": factory_candidates,
            "candidate_json_path": output_dir / "factory_profile_candidate.json",
        },
        "validation_profile_prediction": validation_profile_prediction,
        "validation_pose_linear": pose_linear,
        "frame_metrics": frame_metrics,
        "validation_abc": validation_abc,
        "output_files": output_files,
        "artifact_provenance": {
            "reused_implementation": [
                "existing Steger adapter",
                "existing Session PnP",
                "existing physical-board mask",
                "existing Frozen C0/C1 reconstruction",
                "existing robust linear ground-profile kernel",
                "Ground-3 frame/pose-balanced profile and spline-basis code framework",
            ],
            "reused_artifacts": [
                *reused_artifacts,
            ],
            "newly_computed": newly_computed,
            "not_reused_numeric_results": [
                "Ground-1/Ground-2R/Ground-3 residuals or masks",
                "Ground-3 G(S) coefficients or model-selection result",
                "Session Ground Reference/H1/Stage-A parameters",
            ],
        },
    }
    (output_dir / "ground5a_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "ground5a_report.md", _json_ready(summary), output_files)
    print(f"output_dir={output_dir}")
    for key, value in classification.items():
        print(f"{key}={value}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    _run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
