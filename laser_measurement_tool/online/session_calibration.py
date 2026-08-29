"""Runtime/session metadata helpers for online ground calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from calibration.session_ground import (
    SessionGroundBoardConfig,
    SessionGroundExtrinsic,
    build_camera_to_ground_transform,
    estimate_session_ground_extrinsic_from_corners,
)


def compare_ground_extrinsics(
    reference_R: np.ndarray,
    reference_t: np.ndarray,
    session_R: np.ndarray,
    session_t: np.ndarray,
) -> tuple[float, float]:
    """Return ``(translation_delta_mm, rotation_delta_deg)``.

    Both rotations are camera-to-ground rotations.  The relative rotation is
    therefore ``session_R @ reference_R.T``.
    """
    reference_rotation = _rotation_matrix(reference_R, "reference_R")
    session_rotation = _rotation_matrix(session_R, "session_R")
    reference_translation = _translation_vector(reference_t, "reference_t")
    session_translation = _translation_vector(session_t, "session_t")
    translation_delta = float(
        np.linalg.norm(session_translation - reference_translation)
    )
    relative = session_rotation @ reference_rotation.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation_delta = float(np.degrees(np.arccos(cosine)))
    return translation_delta, rotation_delta


@dataclass(frozen=True, slots=True)
class SessionGroundRepeatability:
    """Repeatability summary for the accepted same-pose PnP frames."""

    required_frames: int
    accepted_frames: int
    translation_deltas_mm: tuple[float, ...]
    rotation_deltas_deg: tuple[float, ...]
    translation_mean_mm: float
    translation_std_mm: float
    translation_max_mm: float
    rotation_mean_deg: float
    rotation_std_deg: float
    rotation_max_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_frames": self.required_frames,
            "accepted_frames": self.accepted_frames,
            "translation_deltas_mm": list(self.translation_deltas_mm),
            "rotation_deltas_deg": list(self.rotation_deltas_deg),
            "translation_mean_mm": self.translation_mean_mm,
            "translation_std_mm": self.translation_std_mm,
            "translation_max_mm": self.translation_max_mm,
            "rotation_mean_deg": self.rotation_mean_deg,
            "rotation_std_deg": self.rotation_std_deg,
            "rotation_max_deg": self.rotation_max_deg,
        }


@dataclass(frozen=True, slots=True)
class SessionGroundPnPQA:
    """Leave-one-frame-out QA for the formal five-frame Session PnP result.

    This is a stability diagnostic only.  It never replaces the formal
    five-frame median solution and must not be interpreted as an absolute
    extrinsic-accuracy estimate.
    """

    method: str
    fold_count: int
    successful_folds: int
    heldout_reprojection_rmse_px: tuple[float | None, ...]
    translation_delta_mm: tuple[float | None, ...]
    rotation_delta_deg: tuple[float | None, ...]
    plane_distance_delta_mm: tuple[float | None, ...]
    plane_normal_delta_deg: tuple[float | None, ...]
    jackknife_extrinsics: tuple[dict[str, Any] | None, ...]
    zg_propagation: dict[str, Any]
    sensitivity: dict[str, float | None]
    status: str
    stability: str
    errors: tuple[str, ...] = ()

    @classmethod
    def failure(cls, message: str, *, fold_count: int = 5) -> "SessionGroundPnPQA":
        """Create a JSON-safe failed QA record without affecting formal PnP."""
        count = max(int(fold_count), 0)
        empty = (None,) * count
        return cls(
            method="leave_one_frame_out",
            fold_count=count,
            successful_folds=0,
            heldout_reprojection_rmse_px=empty,
            translation_delta_mm=empty,
            rotation_delta_deg=empty,
            plane_distance_delta_mm=empty,
            plane_normal_delta_deg=empty,
            jackknife_extrinsics=empty,
            zg_propagation=_empty_zg_propagation(),
            sensitivity={
                "translation_std_mm": None,
                "translation_max_mm": None,
                "rotation_std_deg": None,
                "rotation_max_deg": None,
                "plane_distance_std_mm": None,
                "plane_distance_max_mm": None,
                "plane_normal_std_deg": None,
                "plane_normal_max_deg": None,
            },
            status="FAIL",
            stability="LOW",
            errors=(str(message),),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON schema used by session calibration output."""
        def indexed(values: Sequence[Any]) -> dict[str, Any]:
            return {
                f"fold_{index}": value
                for index, value in enumerate(values, start=1)
            }

        final_metrics = {
            "SESSION_PNP_JACKKNIFE": self.status,
            "HELDOUT_REPROJECTION_RMSE_P95_PX": self.zg_propagation.get(
                "heldout_reprojection_rmse_p95_px"
            ),
            "TRANSLATION_JACKKNIFE_MAX_MM": self.sensitivity.get(
                "translation_max_mm"
            ),
            "ROTATION_JACKKNIFE_MAX_DEG": self.sensitivity.get(
                "rotation_max_deg"
            ),
            "PLANE_DISTANCE_JACKKNIFE_MAX_MM": self.sensitivity.get(
                "plane_distance_max_mm"
            ),
            "PLANE_NORMAL_JACKKNIFE_MAX_DEG": self.sensitivity.get(
                "plane_normal_max_deg"
            ),
            "PREDICTED_ZG_RMSE_MM": self.zg_propagation.get("rmse_mm"),
            "PREDICTED_ZG_P95_MM": self.zg_propagation.get("p95_abs_mm"),
            "PREDICTED_ZG_EDGE_P95_MM": self.zg_propagation.get(
                "edge_p95_abs_mm"
            ),
            "FIVE_FRAME_SESSION_PNP_STABILITY": self.stability,
        }
        return {
            "method": self.method,
            "scope": "leave_one_frame_out_qa",
            "fold_count": self.fold_count,
            "successful_fold_count": self.successful_folds,
            "heldout_reprojection_rmse_px": indexed(
                self.heldout_reprojection_rmse_px
            ),
            "heldout_reprojection_rmse_p95_px": self.zg_propagation.get(
                "heldout_reprojection_rmse_p95_px"
            ),
            "translation_delta_mm": indexed(self.translation_delta_mm),
            "rotation_delta_deg": indexed(self.rotation_delta_deg),
            "plane_distance_delta_mm": indexed(self.plane_distance_delta_mm),
            "plane_normal_delta_deg": indexed(self.plane_normal_delta_deg),
            "jackknife_extrinsics": indexed(self.jackknife_extrinsics),
            "zg_propagation": self.zg_propagation,
            "sensitivity": self.sensitivity,
            "translation_jackknife_std_mm": self.sensitivity.get(
                "translation_std_mm"
            ),
            "translation_jackknife_max_mm": self.sensitivity.get(
                "translation_max_mm"
            ),
            "rotation_jackknife_std_deg": self.sensitivity.get(
                "rotation_std_deg"
            ),
            "rotation_jackknife_max_deg": self.sensitivity.get(
                "rotation_max_deg"
            ),
            "plane_distance_jackknife_std_mm": self.sensitivity.get(
                "plane_distance_std_mm"
            ),
            "plane_distance_jackknife_max_mm": self.sensitivity.get(
                "plane_distance_max_mm"
            ),
            "plane_normal_jackknife_std_deg": self.sensitivity.get(
                "plane_normal_std_deg"
            ),
            "plane_normal_jackknife_max_deg": self.sensitivity.get(
                "plane_normal_max_deg"
            ),
            "predicted_zg_rmse_mm": self.zg_propagation.get("rmse_mm"),
            "predicted_zg_p95_abs_mm": self.zg_propagation.get("p95_abs_mm"),
            "predicted_zg_edge_p95_abs_mm": self.zg_propagation.get(
                "edge_p95_abs_mm"
            ),
            "status": self.status,
            "stability": self.stability,
            "interpretation": "jackknife_stability_only_not_absolute_extrinsic_accuracy",
            "stability_policy": {
                "high": dict(_STABILITY_HIGH_LIMITS),
                "moderate": dict(_STABILITY_MODERATE_LIMITS),
                "status_mapping": {
                    "HIGH": "PASS",
                    "MODERATE": "PARTIAL",
                    "LOW": "FAIL",
                },
            },
            "errors": list(self.errors),
            "final_metrics": final_metrics,
        }


def checkerboard_physical_polygon(
    corners: np.ndarray,
    *,
    pattern_cols: int,
    pattern_rows: int,
) -> np.ndarray:
    """Return the complete physical-board boundary from inner corners.

    The returned polygon extends one corner spacing beyond the detected inner
    corner grid on all four sides.  It is a display/quality-mask helper only;
    Runtime ground-reference fitting continues to use the PnP physical-board mask.
    """
    array = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    expected = int(pattern_cols) * int(pattern_rows)
    if array.shape != (expected, 2) or not np.isfinite(array).all():
        raise ValueError("corners 必须是完整且有限的棋盘角点")
    grid = array.reshape(int(pattern_rows), int(pattern_cols), 2)
    dx_top = grid[0, 1] - grid[0, 0]
    dx_bottom = grid[-1, -1] - grid[-1, -2]
    dy_left = grid[1, 0] - grid[0, 0]
    dy_right = grid[-1, -1] - grid[-2, -1]
    dx = (dx_top + dx_bottom) * 0.5
    dy = (dy_left + dy_right) * 0.5
    return np.ascontiguousarray(
        np.asarray(
            [
                grid[0, 0] - dx - dy,
                grid[0, -1] + dx - dy,
                grid[-1, -1] + dx + dy,
                grid[-1, 0] - dx + dy,
            ],
            dtype=np.float64,
        )
    )


def assess_checkerboard_image_quality(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    pattern_cols: int,
    pattern_rows: int,
    saturation_ratio_warn: float,
    dynamic_range_p95_p5_warn: float,
    edge_margin_warn_px: float,
) -> dict[str, Any]:
    """Measure configurable warning-only image-quality indicators."""
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.ndim != 2 or not gray.size:
        raise ValueError("image 必须是非空灰度图像")
    polygon = checkerboard_physical_polygon(
        corners,
        pattern_cols=pattern_cols,
        pattern_rows=pattern_rows,
    )
    height, width = gray.shape[:2]
    polygon_i = np.rint(polygon).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon_i, 1)
    values = gray[mask.astype(bool)]
    if not len(values):
        raise ValueError("棋盘物理 mask 没有覆盖当前图像")
    maximum = float(np.iinfo(gray.dtype).max) if np.issubdtype(gray.dtype, np.integer) else float(np.max(values))
    saturation_level = maximum * 0.99
    saturation_ratio = float(np.mean(values.astype(np.float64) >= saturation_level))
    p95_p5 = float(np.percentile(values, 95.0) - np.percentile(values, 5.0))
    edge_margin = float(
        min(
            np.min(polygon[:, 0]),
            np.min(polygon[:, 1]),
            width - 1 - np.max(polygon[:, 0]),
            height - 1 - np.max(polygon[:, 1]),
        )
    )
    warnings: list[str] = []
    if saturation_ratio > float(saturation_ratio_warn):
        warnings.append("checkerboard_saturation_ratio_high")
    if p95_p5 < float(dynamic_range_p95_p5_warn):
        warnings.append("checkerboard_dynamic_range_low")
    if edge_margin < float(edge_margin_warn_px):
        warnings.append("checkerboard_too_close_to_image_edge")
    return {
        "saturation_ratio": saturation_ratio,
        "dynamic_range_p95_p5": p95_p5,
        "edge_margin_px": edge_margin,
        "warnings": warnings,
        "mask": "complete_physical_board_from_corners",
    }


def aggregate_session_ground_extrinsic(
    results: Sequence[SessionGroundExtrinsic],
    intrinsics: Any,
    board_config: SessionGroundBoardConfig,
    *,
    required_frames: int = 5,
) -> tuple[SessionGroundExtrinsic, SessionGroundRepeatability]:
    """Median-aggregate same-order corners, then reuse the shared PnP solve."""
    if len(results) != int(required_frames):
        raise ValueError(
            f"需要 {int(required_frames)} 个有效 PnP 帧，实际 {len(results)} 个"
        )
    expected_corners = board_config.pattern_cols * board_config.pattern_rows
    corner_arrays: list[np.ndarray] = []
    for index, result in enumerate(results, start=1):
        if result.status != "success" or result.detected_corners is None:
            raise ValueError(f"第 {index} 个 PnP 帧无效：{result.message}")
        corners = np.asarray(result.detected_corners, dtype=np.float32).reshape(-1, 2)
        if len(corners) != expected_corners:
            raise ValueError(
                f"第 {index} 个 PnP 帧角点数不是 {expected_corners}：{len(corners)}"
            )
        if not np.isfinite(corners).all():
            raise ValueError(f"第 {index} 个 PnP 帧包含非有限角点")
        corner_arrays.append(corners)

    stack = np.stack(corner_arrays, axis=0)
    # The detector/object-point protocol is row-major.  Reject a detector
    # reversal, while allowing ordinary sub-pixel frame-to-frame movement.
    reference = stack[0]
    rows = int(board_config.pattern_rows)
    cols = int(board_config.pattern_cols)
    reference_area = _corner_grid_area(reference, rows, cols)
    reference_dx, reference_dy = _corner_grid_directions(reference, rows, cols)
    for index, corners in enumerate(stack[1:], start=2):
        area = _corner_grid_area(corners, rows, cols)
        dx, dy = _corner_grid_directions(corners, rows, cols)
        if (
            reference_area * area < 0.0
            or float(reference_dx @ dx) <= 0.0
            or float(reference_dy @ dy) <= 0.0
        ):
            raise ValueError(f"第 {index} 个 PnP 帧角点顺序发生反向")

    aggregate_corners = np.ascontiguousarray(
        np.median(stack.astype(np.float64), axis=0).astype(np.float32)
    )
    final = estimate_session_ground_extrinsic_from_corners(
        aggregate_corners,
        intrinsics,
        board_config,
        detection_method="5_frame_median",
    )
    if final.status != "success" or final.R is None or final.t is None:
        raise ValueError(f"聚合角点 PnP 失败：{final.message}")

    translation_deltas: list[float] = []
    rotation_deltas: list[float] = []
    for result in results:
        if result.R is None or result.t is None:
            raise ValueError("有效 PnP 帧缺少 R/t")
        delta_t, delta_r = compare_ground_extrinsics(
            final.R,
            final.t,
            result.R,
            result.t,
        )
        translation_deltas.append(delta_t)
        rotation_deltas.append(delta_r)
    translation_array = np.asarray(translation_deltas, dtype=np.float64)
    rotation_array = np.asarray(rotation_deltas, dtype=np.float64)
    repeatability = SessionGroundRepeatability(
        required_frames=int(required_frames),
        accepted_frames=len(results),
        translation_deltas_mm=tuple(float(value) for value in translation_array),
        rotation_deltas_deg=tuple(float(value) for value in rotation_array),
        translation_mean_mm=float(np.mean(translation_array)),
        translation_std_mm=float(np.std(translation_array)),
        translation_max_mm=float(np.max(translation_array)),
        rotation_mean_deg=float(np.mean(rotation_array)),
        rotation_std_deg=float(np.std(rotation_array)),
        rotation_max_deg=float(np.max(rotation_array)),
    )
    return final, repeatability


def assess_session_pnp_qa(
    results: Sequence[SessionGroundExtrinsic],
    final: SessionGroundExtrinsic,
    intrinsics: Any,
    board_config: SessionGroundBoardConfig,
    *,
    required_frames: int = 5,
    max_heldout_reprojection_rmse_px: float = 0.5,
    grid_columns: int = 21,
    grid_rows: int = 17,
) -> SessionGroundPnPQA:
    """Evaluate the formal Session PnP with five leave-one-out folds.

    Each fold takes the ordered corners from the other four accepted frames,
    computes their median, and runs the same solve-only PnP API.  The formal
    ``final`` result is used only as the comparison reference; it is never
    replaced by a jackknife result.

    Zg propagation uses a fixed grid covering the physical checkerboard area
    (extended by one square on each side, matching the existing board-mask
    convention).  Ground correction/reference fitting is intentionally not
    involved.
    """
    fold_count = len(results)
    if fold_count != int(required_frames):
        return SessionGroundPnPQA.failure(
            f"leave-one-frame-out 需要 {int(required_frames)} 帧，实际 {fold_count} 帧",
            fold_count=fold_count,
        )
    if final.status != "success" or final.R is None or final.t is None:
        return SessionGroundPnPQA.failure(
            f"正式 5-frame PnP 无效：{final.message}",
            fold_count=fold_count,
        )
    try:
        K, D = _normalise_qa_intrinsics(intrinsics)
    except (TypeError, ValueError) as error:
        return SessionGroundPnPQA.failure(
            f"QA intrinsics 无效：{error}",
            fold_count=fold_count,
        )
    if int(grid_columns) < 3 or int(grid_rows) < 3:
        return SessionGroundPnPQA.failure(
            "Zg QA grid 至少需要 3 x 3 个点",
            fold_count=fold_count,
        )

    heldout_rmse: list[float | None] = []
    translation_delta: list[float | None] = []
    rotation_delta: list[float | None] = []
    plane_distance_delta: list[float | None] = []
    plane_normal_delta: list[float | None] = []
    jackknife_extrinsics: list[dict[str, Any] | None] = []
    jackknife_results: list[SessionGroundExtrinsic | None] = []
    errors: list[str] = []

    try:
        final_normal = _ground_normal(final)
        final_plane_distance = _ground_plane_distance(final)
    except ValueError as error:
        return SessionGroundPnPQA.failure(
            f"正式 ground plane 无效：{error}",
            fold_count=fold_count,
        )

    grid: dict[str, Any] | None
    try:
        grid = _build_zg_test_grid(
            final,
            board_config,
            grid_columns=int(grid_columns),
            grid_rows=int(grid_rows),
        )
    except ValueError as error:
        grid = None
        errors.append(f"Zg propagation grid unavailable: {error}")

    for heldout_index in range(fold_count):
        heldout_rmse.append(None)
        translation_delta.append(None)
        rotation_delta.append(None)
        plane_distance_delta.append(None)
        plane_normal_delta.append(None)
        jackknife_extrinsics.append(None)
        jackknife_results.append(None)
        try:
            training = [
                result
                for index, result in enumerate(results)
                if index != heldout_index
            ]
            aggregate_corners = _median_corners_for_qa(training, board_config)
            jackknife = estimate_session_ground_extrinsic_from_corners(
                aggregate_corners,
                {"K": K, "D": D},
                board_config,
                detection_method="leave_one_frame_out",
            )
            if jackknife.status != "success" or jackknife.R is None or jackknife.t is None:
                raise ValueError(f"jackknife PnP 失败：{jackknife.message}")
            heldout_corners = _corners_for_qa(results[heldout_index], board_config)
            heldout_rmse[heldout_index] = _reprojection_rmse_for_pose(
                jackknife,
                heldout_corners,
                K,
                D,
                board_config,
            )
            delta_t, delta_r = compare_ground_extrinsics(
                final.R,
                final.t,
                jackknife.R,
                jackknife.t,
            )
            jack_normal = _ground_normal(jackknife)
            jack_plane_distance = _ground_plane_distance(jackknife)
            translation_delta[heldout_index] = delta_t
            rotation_delta[heldout_index] = delta_r
            plane_distance_delta[heldout_index] = abs(
                jack_plane_distance - final_plane_distance
            )
            plane_normal_delta[heldout_index] = _angle_between_normals_deg(
                final_normal,
                jack_normal,
            )
            jackknife_extrinsics[heldout_index] = {
                "R_camera_to_ground": np.asarray(jackknife.R, dtype=np.float64).tolist(),
                "t_camera_to_ground_mm": np.asarray(
                    jackknife.t, dtype=np.float64
                )
                .reshape(3)
                .tolist(),
                "plane_distance_mm": jack_plane_distance,
                "plane_normal_in_camera": jack_normal.tolist(),
            }
            jackknife_results[heldout_index] = jackknife
        except (TypeError, ValueError, cv2.error) as error:
            errors.append(f"fold_{heldout_index + 1}: {error}")

    per_fold_zg: dict[str, dict[str, Any]] = {}
    all_zg: list[np.ndarray] = []
    center_zg: list[np.ndarray] = []
    edge_zg: list[np.ndarray] = []
    if grid is not None:
        camera_points = np.asarray(grid["camera_points"], dtype=np.float64)
        final_z = np.asarray(grid["final_z"], dtype=np.float64)
        center_mask = np.asarray(grid["center_mask"], dtype=bool)
        edge_mask = np.asarray(grid["edge_mask"], dtype=bool)
        for fold_index, jackknife in enumerate(jackknife_results, start=1):
            if jackknife is None or jackknife.T_ground_from_camera is None:
                continue
            jack_z = _transform_points(
                jackknife.T_ground_from_camera,
                camera_points,
            )[:, 2]
            delta_z = jack_z - final_z
            all_zg.append(delta_z)
            center_zg.append(delta_z[center_mask])
            edge_zg.append(delta_z[edge_mask])
            per_fold_zg[f"fold_{fold_index}"] = {
                "full_fov": _summarize_zg(delta_z),
                "center": _summarize_zg(delta_z[center_mask]),
                "edge": _summarize_zg(delta_z[edge_mask]),
            }

    if grid is None:
        zg_propagation = _empty_zg_propagation()
    else:
        full_values = (
            np.concatenate(all_zg) if all_zg else np.empty(0, dtype=np.float64)
        )
        center_values = (
            np.concatenate(center_zg)
            if center_zg
            else np.empty(0, dtype=np.float64)
        )
        edge_values = (
            np.concatenate(edge_zg) if edge_zg else np.empty(0, dtype=np.float64)
        )
        full_summary = _summarize_zg(full_values)
        center_summary = _summarize_zg(center_values)
        edge_summary = _summarize_zg(edge_values)
        zg_propagation = {
            **full_summary,
            "heldout_reprojection_rmse_p95_px": _percentile_or_none(
                [value for value in heldout_rmse if value is not None],
                95.0,
            ),
            "center": center_summary,
            "edge": edge_summary,
            "full_fov": full_summary,
            "center_p95_abs_mm": center_summary["p95_abs_mm"],
            "edge_p95_abs_mm": edge_summary["p95_abs_mm"],
            "per_fold": per_fold_zg,
            "grid": grid["metadata"],
        }
    heldout_p95 = _percentile_or_none(
        [value for value in heldout_rmse if value is not None],
        95.0,
    )
    zg_propagation["heldout_reprojection_rmse_p95_px"] = heldout_p95

    sensitivity = {
        "translation_std_mm": _std_or_none(translation_delta),
        "translation_max_mm": _max_or_none(translation_delta),
        "rotation_std_deg": _std_or_none(rotation_delta),
        "rotation_max_deg": _max_or_none(rotation_delta),
        "plane_distance_std_mm": _std_or_none(plane_distance_delta),
        "plane_distance_max_mm": _max_or_none(plane_distance_delta),
        "plane_normal_std_deg": _std_or_none(plane_normal_delta),
        "plane_normal_max_deg": _max_or_none(plane_normal_delta),
    }
    successful_folds = sum(result is not None for result in jackknife_results)
    zg_full = zg_propagation.get("full_fov", {})
    pose_complete = successful_folds == fold_count and not errors
    grid_complete = (
        grid is not None
        and int(zg_full.get("count", 0)) > 0
        and zg_full.get("rmse_mm") is not None
    )
    stability = _classify_session_pnp_stability(
        heldout_p95,
        sensitivity,
        zg_full.get("p95_abs_mm"),
    )
    if successful_folds == 0:
        status = "FAIL"
    elif heldout_p95 is not None and heldout_p95 > float(
        max_heldout_reprojection_rmse_px
    ):
        status = "FAIL"
    elif not pose_complete or not grid_complete:
        status = "PARTIAL"
    elif stability == "LOW":
        status = "FAIL"
    elif stability == "MODERATE":
        status = "PARTIAL"
    else:
        status = "PASS"
    return SessionGroundPnPQA(
        method="leave_one_frame_out",
        fold_count=fold_count,
        successful_folds=successful_folds,
        heldout_reprojection_rmse_px=tuple(heldout_rmse),
        translation_delta_mm=tuple(translation_delta),
        rotation_delta_deg=tuple(rotation_delta),
        plane_distance_delta_mm=tuple(plane_distance_delta),
        plane_normal_delta_deg=tuple(plane_normal_delta),
        jackknife_extrinsics=tuple(jackknife_extrinsics),
        zg_propagation=zg_propagation,
        sensitivity=sensitivity,
        status=status,
        stability=stability,
        errors=tuple(errors),
    )


def _normalise_qa_intrinsics(intrinsics: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(intrinsics, Mapping):
        matrix = intrinsics.get("K", intrinsics.get("camera_matrix"))
        distortion = intrinsics.get("D", intrinsics.get("dist_coeffs"))
    elif isinstance(intrinsics, tuple) and len(intrinsics) == 2:
        matrix, distortion = intrinsics
    else:
        matrix = getattr(intrinsics, "camera_matrix", None)
        distortion = getattr(intrinsics, "dist_coeffs", None)
    if matrix is None or distortion is None:
        raise ValueError("intrinsics must provide camera matrix K and distortion D")
    K = np.asarray(matrix, dtype=np.float64)
    D = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if K.shape != (3, 3) or D.size == 0:
        raise ValueError("intrinsics must provide a 3x3 K and non-empty D")
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("intrinsics must contain finite values")
    return np.ascontiguousarray(K), np.ascontiguousarray(D)


def _corners_for_qa(
    result: SessionGroundExtrinsic,
    board_config: SessionGroundBoardConfig,
) -> np.ndarray:
    if result.status != "success" or result.detected_corners is None:
        raise ValueError(f"PnP frame is invalid: {result.message}")
    corners = np.asarray(result.detected_corners, dtype=np.float32).reshape(-1, 2)
    expected = board_config.pattern_cols * board_config.pattern_rows
    if corners.shape != (expected, 2) or not np.isfinite(corners).all():
        raise ValueError(f"PnP frame must contain {expected} finite ordered corners")
    return np.ascontiguousarray(corners)


def _median_corners_for_qa(
    results: Sequence[SessionGroundExtrinsic],
    board_config: SessionGroundBoardConfig,
) -> np.ndarray:
    if not results:
        raise ValueError("jackknife training set is empty")
    stack = np.stack(
        [_corners_for_qa(result, board_config) for result in results],
        axis=0,
    )
    reference = stack[0]
    rows = int(board_config.pattern_rows)
    cols = int(board_config.pattern_cols)
    reference_area = _corner_grid_area(reference, rows, cols)
    reference_dx, reference_dy = _corner_grid_directions(reference, rows, cols)
    for index, corners in enumerate(stack[1:], start=2):
        area = _corner_grid_area(corners, rows, cols)
        dx, dy = _corner_grid_directions(corners, rows, cols)
        if (
            reference_area * area < 0.0
            or float(reference_dx @ dx) <= 0.0
            or float(reference_dy @ dy) <= 0.0
        ):
            raise ValueError(f"jackknife training frame {index} corner order reversed")
    return np.ascontiguousarray(
        np.median(stack.astype(np.float64), axis=0).astype(np.float32)
    )


def _reprojection_rmse_for_pose(
    result: SessionGroundExtrinsic,
    corners: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    board_config: SessionGroundBoardConfig,
) -> float:
    if result.rvec is None or result.tvec is None:
        raise ValueError("jackknife PnP pose has no rvec/tvec")
    projected, _ = cv2.projectPoints(
        board_config.object_points(),
        result.rvec,
        result.tvec,
        K,
        D,
    )
    residual = np.asarray(corners, dtype=np.float64) - projected.reshape(-1, 2)
    value = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    if not np.isfinite(value):
        raise ValueError("held-out reprojection RMSE is not finite")
    return value


def _ground_normal(result: SessionGroundExtrinsic) -> np.ndarray:
    if result.ground_normal_in_camera is not None:
        normal = np.asarray(result.ground_normal_in_camera, dtype=np.float64).reshape(3)
    elif result.R_board_to_camera is not None:
        normal = np.asarray(result.R_board_to_camera, dtype=np.float64)[:, 2]
        if normal[2] > 0.0:
            normal = -normal
    else:
        raise ValueError("ground plane normal is unavailable")
    norm = float(np.linalg.norm(normal))
    if norm <= np.finfo(np.float64).eps or not np.isfinite(norm):
        raise ValueError("ground plane normal is degenerate")
    return np.ascontiguousarray(normal / norm)


def _ground_plane_distance(result: SessionGroundExtrinsic) -> float:
    normal = _ground_normal(result)
    if result.ground_origin_in_camera is not None:
        origin = np.asarray(result.ground_origin_in_camera, dtype=np.float64).reshape(3)
    elif result.T_camera_from_ground is not None:
        transform = np.asarray(result.T_camera_from_ground, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("T_camera_from_ground must have shape (4, 4)")
        origin = transform[:3, 3]
    elif result.R_board_to_camera is not None and result.tvec is not None:
        _, _, _, camera_from_ground = build_camera_to_ground_transform(
            result.R_board_to_camera,
            result.tvec,
        )
        origin = camera_from_ground[:3, 3]
    else:
        raise ValueError("ground plane origin is unavailable")
    if not np.isfinite(origin).all():
        raise ValueError("ground plane origin is not finite")
    value = abs(float(normal @ origin))
    if not np.isfinite(value):
        raise ValueError("ground plane distance is not finite")
    return value


def _angle_between_normals_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.asarray(first) @ np.asarray(second), -1.0, 1.0))
    value = float(np.degrees(np.arccos(cosine)))
    if not np.isfinite(value):
        raise ValueError("ground plane angular delta is not finite")
    return value


def _ground_transform(result: SessionGroundExtrinsic) -> np.ndarray:
    if result.T_ground_from_camera is not None:
        transform = np.asarray(result.T_ground_from_camera, dtype=np.float64)
        if transform.shape == (4, 4):
            return np.ascontiguousarray(transform)
    if result.R is None or result.t is None:
        raise ValueError("camera-to-ground transform is unavailable")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(result.R, dtype=np.float64)
    transform[:3, 3] = np.asarray(result.t, dtype=np.float64).reshape(3)
    return transform


def _camera_from_ground_transform(result: SessionGroundExtrinsic) -> np.ndarray:
    if result.T_camera_from_ground is not None:
        transform = np.asarray(result.T_camera_from_ground, dtype=np.float64)
        if transform.shape == (4, 4):
            return np.ascontiguousarray(transform)
    return np.ascontiguousarray(np.linalg.inv(_ground_transform(result)))


def _board_to_camera_transform(result: SessionGroundExtrinsic) -> np.ndarray:
    if result.T_camera_from_board is not None:
        transform = np.asarray(result.T_camera_from_board, dtype=np.float64)
        if transform.shape == (4, 4):
            return np.ascontiguousarray(transform)
    if result.R_board_to_camera is None or result.tvec is None:
        raise ValueError("board-to-camera pose is unavailable")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(result.R_board_to_camera, dtype=np.float64)
    transform[:3, 3] = np.asarray(result.tvec, dtype=np.float64).reshape(3)
    return transform


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack((array, np.ones(len(array), dtype=np.float64)))
    transformed = (matrix @ homogeneous.T).T
    weights = transformed[:, 3]
    if not np.isfinite(transformed).all() or np.any(np.abs(weights) <= 1.0e-12):
        raise ValueError("transform produced invalid points")
    return np.ascontiguousarray(transformed[:, :3] / weights[:, None])


def _build_zg_test_grid(
    final: SessionGroundExtrinsic,
    board_config: SessionGroundBoardConfig,
    *,
    grid_columns: int,
    grid_rows: int,
) -> dict[str, Any]:
    square = float(board_config.square_size_mm)
    cols = int(board_config.pattern_cols)
    rows = int(board_config.pattern_rows)
    # One square outside the inner-corner rectangle on all sides, matching
    # measurement.board_mask's physical-board convention.
    boundary_board = np.asarray(
        [
            [-square, -square, 0.0],
            [cols * square, -square, 0.0],
            [cols * square, rows * square, 0.0],
            [-square, rows * square, 0.0],
        ],
        dtype=np.float64,
    )
    boundary_camera = _transform_points(
        _board_to_camera_transform(final),
        boundary_board,
    )
    boundary_ground = _transform_points(
        _ground_transform(final),
        boundary_camera,
    )
    if not np.isfinite(boundary_ground).all():
        raise ValueError("checkerboard ground boundary is not finite")
    x_min, x_max = float(np.min(boundary_ground[:, 0])), float(np.max(boundary_ground[:, 0]))
    y_min, y_max = float(np.min(boundary_ground[:, 1])), float(np.max(boundary_ground[:, 1]))
    if not x_max > x_min or not y_max > y_min:
        raise ValueError("checkerboard ground boundary is degenerate")
    x_values = np.linspace(x_min, x_max, int(grid_columns), dtype=np.float64)
    y_values = np.linspace(y_min, y_max, int(grid_rows), dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    ground_points = np.column_stack(
        (
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.zeros(grid_x.size, dtype=np.float64),
        )
    )
    camera_points = _transform_points(
        _camera_from_ground_transform(final),
        ground_points,
    )
    final_ground_points = _transform_points(_ground_transform(final), camera_points)
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_norm = (ground_points[:, 0] - x_min) / x_span
    y_norm = (ground_points[:, 1] - y_min) / y_span
    center_mask = (
        (x_norm >= 0.25)
        & (x_norm <= 0.75)
        & (y_norm >= 0.25)
        & (y_norm <= 0.75)
    )
    edge_mask = (
        (x_norm <= 0.15)
        | (x_norm >= 0.85)
        | (y_norm <= 0.15)
        | (y_norm >= 0.85)
    )
    return {
        "camera_points": camera_points,
        "final_z": final_ground_points[:, 2],
        "center_mask": center_mask,
        "edge_mask": edge_mask,
        "metadata": {
            "source": "checkerboard_physical_board",
            "coordinate": "final_session_ground",
            "columns": int(grid_columns),
            "rows": int(grid_rows),
            "point_count": int(len(ground_points)),
            "x_range_mm": [x_min, x_max],
            "y_range_mm": [y_min, y_max],
            "center_definition": "normalized_x_y in [0.25, 0.75]",
            "edge_definition": "normalized_x_y outside [0.15, 0.85] on either axis",
            "ground_correction_applied": False,
        },
    }


def _summarize_zg(values: np.ndarray | Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return {
            "count": 0,
            "bias_mm": None,
            "rmse_mm": None,
            "p95_abs_mm": None,
            "max_abs_mm": None,
        }
    absolute = np.abs(array)
    return {
        "count": int(array.size),
        "bias_mm": float(np.mean(array)),
        "rmse_mm": float(np.sqrt(np.mean(array**2))),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "max_abs_mm": float(np.max(absolute)),
    }


def _empty_zg_propagation() -> dict[str, Any]:
    empty = _summarize_zg(np.empty(0, dtype=np.float64))
    return {
        **empty,
        "heldout_reprojection_rmse_p95_px": None,
        "center": dict(empty),
        "edge": dict(empty),
        "full_fov": dict(empty),
        "center_p95_abs_mm": None,
        "edge_p95_abs_mm": None,
        "per_fold": {},
        "grid": None,
    }


def _percentile_or_none(values: Sequence[float], percentile: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return float(np.percentile(array, percentile))


def _std_or_none(values: Sequence[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return float(np.std(array))


def _max_or_none(values: Sequence[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return float(np.max(array))


_STABILITY_HIGH_LIMITS = {
    "heldout_reprojection_rmse_px": 0.15,
    "translation_max_mm": 0.25,
    "rotation_max_deg": 0.05,
    "plane_distance_max_mm": 0.25,
    "plane_normal_max_deg": 0.05,
    "zg_p95_abs_mm": 0.25,
}
_STABILITY_MODERATE_LIMITS = {
    "heldout_reprojection_rmse_px": 0.5,
    "translation_max_mm": 1.0,
    "rotation_max_deg": 0.2,
    "plane_distance_max_mm": 1.0,
    "plane_normal_max_deg": 0.2,
    "zg_p95_abs_mm": 1.0,
}


def _classify_session_pnp_stability(
    heldout_p95: float | None,
    sensitivity: Mapping[str, float | None],
    zg_p95: float | None,
) -> str:
    values = {
        "heldout_reprojection_rmse_px": heldout_p95,
        "translation_max_mm": sensitivity.get("translation_max_mm"),
        "rotation_max_deg": sensitivity.get("rotation_max_deg"),
        "plane_distance_max_mm": sensitivity.get("plane_distance_max_mm"),
        "plane_normal_max_deg": sensitivity.get("plane_normal_max_deg"),
        "zg_p95_abs_mm": zg_p95,
    }
    if any(value is None or not np.isfinite(float(value)) for value in values.values()):
        return "LOW"
    if all(
        float(values[key]) <= limit
        for key, limit in _STABILITY_HIGH_LIMITS.items()
    ):
        return "HIGH"
    if all(
        float(values[key]) <= limit
        for key, limit in _STABILITY_MODERATE_LIMITS.items()
    ):
        return "MODERATE"
    return "LOW"


def _corner_grid_area(corners: np.ndarray, rows: int, cols: int) -> float:
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    top_left = grid[0, 0]
    top_right = grid[0, -1]
    bottom_left = grid[-1, 0]
    right = top_right - top_left
    down = bottom_left - top_left
    return float(right[0] * down[1] - right[1] * down[0])


def _corner_grid_directions(
    corners: np.ndarray, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    return grid[0, -1] - grid[0, 0], grid[-1, 0] - grid[0, 0]


def build_session_ground_payload(
    result: SessionGroundExtrinsic,
    board_config: SessionGroundBoardConfig,
    *,
    frame_number: int | None,
    frame_offset: tuple[int, int] | None,
    reference_R: np.ndarray,
    reference_t: np.ndarray,
    runtime_source: str,
    frame_host_monotonic_ns: int | None = None,
    session_generation: int | None = None,
    repeatability: SessionGroundRepeatability | Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
    session_pnp_qa: SessionGroundPnPQA | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe record for the latest calibration attempt."""
    reference_rotation = _rotation_matrix(reference_R, "reference_R")
    reference_translation = _translation_vector(reference_t, "reference_t")
    valid = result.status == "success" and result.R is not None and result.t is not None
    if valid:
        assert result.R is not None and result.t is not None
        delta_translation, delta_rotation = compare_ground_extrinsics(
            reference_rotation,
            reference_translation,
            result.R,
            result.t,
        )
        session_extrinsic: dict[str, Any] | None = {
            "R_camera_to_ground": np.asarray(result.R, dtype=np.float64).tolist(),
            "t_camera_to_ground_mm": np.asarray(result.t, dtype=np.float64)
            .reshape(3)
            .tolist(),
            "T_ground_from_camera": np.asarray(
                result.T_ground_from_camera, dtype=np.float64
            ).tolist()
            if result.T_ground_from_camera is not None
            else None,
        }
        delta: dict[str, float] | None = {
            "translation_mm": delta_translation,
            "rotation_deg": delta_rotation,
        }
    else:
        session_extrinsic = None
        delta = None

    frame_repeatability = (
        repeatability.as_dict()
        if isinstance(repeatability, SessionGroundRepeatability)
        else None if repeatability is None else dict(repeatability)
    )
    leave_one_frame_out_qa = (
        session_pnp_qa.as_dict()
        if isinstance(session_pnp_qa, SessionGroundPnPQA)
        else None if session_pnp_qa is None else dict(session_pnp_qa)
    )
    return {
        "schema_version": 2,
        "source": "session_ground_calibration",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VALID" if valid else "INVALID",
        "valid": valid,
        "message": result.message,
        "runtime": {
            "ground_extrinsic_source": runtime_source,
            "ground_extrinsic_generation": session_generation,
        },
        "board": {
            "pattern_cols": board_config.pattern_cols,
            "pattern_rows": board_config.pattern_rows,
            "square_size_mm": board_config.square_size_mm,
            "detector": board_config.detector,
        },
        "frame": {
            "camera_frame_number": frame_number,
            "host_monotonic_ns": frame_host_monotonic_ns,
            "session_generation": session_generation,
            "offset_x": frame_offset[0] if frame_offset is not None else None,
            "offset_y": frame_offset[1] if frame_offset is not None else None,
        },
        "detection": {
            "method": result.detection_method,
            "corner_count": (
                int(len(result.detected_corners))
                if result.detected_corners is not None
                else 0
            ),
            "corners": (
                np.asarray(result.detected_corners, dtype=np.float64).tolist()
                if result.detected_corners is not None
                else None
            ),
            "reprojection_rmse_px": result.reprojection_rmse_px,
        },
        "reference_extrinsic": {
            "R_camera_to_ground": reference_rotation.tolist(),
            "t_camera_to_ground_mm": reference_translation.tolist(),
        },
        "session_extrinsic": session_extrinsic,
        "delta": delta,
        # Keep the legacy key for existing readers, while naming the two
        # protocols explicitly for new readers.
        "repeatability": frame_repeatability,
        "frame_repeatability": frame_repeatability,
        "session_pnp_qa": leave_one_frame_out_qa,
        "quality": None if quality is None else dict(quality),
    }


def save_session_ground_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically save the latest session record.

    PnP retries must not erase an independently acquired Session ground-reference
    record that is already in the same JSON file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    if target.is_file():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"无法读取 Session 标定 JSON: {target}: {error}") from error
        if isinstance(previous, dict):
            for key in (
                "session_ground_reference",
                "session_ground_reference_status",
            ):
                if key not in record and key in previous:
                    record[key] = previous[key]
            previous_runtime = previous.get("runtime")
            current_runtime = record.get("runtime")
            if isinstance(previous_runtime, dict) and isinstance(current_runtime, dict):
                record["runtime"] = {**previous_runtime, **current_runtime}
            previous_reference = previous.get("session_ground_reference")
            current_generation = record.get("runtime", {}).get(
                "ground_extrinsic_generation"
            )
            if (
                isinstance(previous_reference, dict)
                and current_generation is not None
                and previous_reference.get("ground_extrinsic_generation") is not None
                and previous_reference.get("ground_extrinsic_generation")
                != current_generation
            ):
                # Keep the old record as history, but make its runtime state
                # explicit: a new ground-extrinsic generation cannot reuse it.
                record["session_ground_reference_status"] = (
                    "STALE_EXTRINSIC_GENERATION"
                )
                record["session_ground_reference_runtime_valid"] = False
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def merge_session_ground_reference(
    path: str | Path,
    reference_payload: Mapping[str, Any],
    *,
    ground_extrinsic_source: str,
) -> Path:
    """Merge the fitted Session ground reference into the Session JSON."""
    target = Path(path)
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"无法读取 Session 标定 JSON: {target}: {error}") from error
        if not isinstance(existing, dict):
            raise ValueError(f"Session 标定 JSON 根节点必须是对象: {target}")
    else:
        existing = {
            "schema_version": 2,
            "source": "session_ground_calibration",
        }

    existing["session_ground_reference"] = dict(reference_payload)
    existing["session_ground_reference_status"] = reference_payload.get(
        "status", "INVALID"
    )
    runtime = existing.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        existing["runtime"] = runtime
    runtime["ground_extrinsic_source"] = ground_extrinsic_source
    runtime["ground_reference_source"] = reference_payload.get("source")
    if "ground_extrinsic_generation" in reference_payload:
        runtime["ground_extrinsic_generation"] = reference_payload[
            "ground_extrinsic_generation"
        ]
    existing["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    return save_session_ground_payload(target, existing)


def _rotation_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return np.ascontiguousarray(array)


def _translation_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return np.ascontiguousarray(array)


__all__ = [
    "SessionGroundPnPQA",
    "SessionGroundRepeatability",
    "aggregate_session_ground_extrinsic",
    "assess_session_pnp_qa",
    "assess_checkerboard_image_quality",
    "build_session_ground_payload",
    "checkerboard_physical_polygon",
    "compare_ground_extrinsics",
    "merge_session_ground_reference",
    "save_session_ground_payload",
]
