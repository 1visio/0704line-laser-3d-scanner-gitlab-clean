"""Shared ground-support point selection helpers.

The checkerboard selector is the single implementation used by both the
Laser Ground Sanity diagnostic and Session Laser Ground Reference.  It keeps
the calibration-tool semantic of projecting the complete physical board
boundary, including the optional configurable physical inset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BoardGroundPointSelection:
    """Board-mask result with the exact source-row identity preserved."""

    selected_points: np.ndarray
    selected_indices: np.ndarray
    selected_mask: np.ndarray
    metadata: dict[str, Any]


def full_board_physical_polygon(
    rvec: np.ndarray,
    tvec: np.ndarray,
    *,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_offset: tuple[int, int] = (0, 0),
    inset_mm: float = 0.0,
) -> np.ndarray:
    """Project the complete physical checkerboard boundary into full pixels.

    The PnP object-point origin is the first detected inner corner. Therefore
    an 11x8, 20 mm board extends one square outside that grid on every side:
    X=[-20, 220] and Y=[-20, 160] mm. The default inset is exactly zero.
    """
    if isinstance(pattern_cols, bool) or int(pattern_cols) != pattern_cols:
        raise ValueError("pattern_cols 必须是整数")
    if isinstance(pattern_rows, bool) or int(pattern_rows) != pattern_rows:
        raise ValueError("pattern_rows 必须是整数")
    columns = int(pattern_cols)
    rows = int(pattern_rows)
    if columns < 2 or rows < 2:
        raise ValueError("pattern_cols 和 pattern_rows 必须至少为 2")
    square = float(square_size_mm)
    inset = float(inset_mm)
    if not math.isfinite(square) or square <= 0.0:
        raise ValueError("square_size_mm 必须是有限正数")
    if not math.isfinite(inset) or inset < 0.0:
        raise ValueError("inset_mm 必须是有限非负数")
    x_min = -square + inset
    x_max = columns * square - inset
    y_min = -square + inset
    y_max = rows * square - inset
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("inset_mm 过大，无法形成有效棋盘边界")
    object_corners = np.asarray(
        [
            [x_min, y_min, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
            [x_min, y_max, 0.0],
        ],
        dtype=np.float64,
    )
    try:
        offset = np.asarray(image_offset, dtype=np.float64).reshape(2)
    except (TypeError, ValueError) as error:
        raise ValueError("image_offset 必须是两个坐标") from error
    if not np.isfinite(offset).all():
        raise ValueError("image_offset 必须是有限数值")
    rotation = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    camera = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(dist_coeffs, dtype=np.float64)
    if (
        not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or camera.shape != (3, 3)
        or not np.isfinite(camera).all()
        or not np.isfinite(distortion).all()
    ):
        raise ValueError("PnP pose、内参和畸变参数必须是有限数值")
    try:
        projected, _ = cv2.projectPoints(
            object_corners,
            rotation,
            translation,
            camera,
            distortion,
        )
    except cv2.error as error:
        raise ValueError(f"完整棋盘物理边界投影失败：{error}") from error
    return np.ascontiguousarray(projected.reshape(-1, 2) + offset, dtype=np.float64)


def select_board_ground_points(
    pixels_uv_full: np.ndarray,
    points_ground: np.ndarray,
    *,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    image_offset: tuple[int, int] = (0, 0),
    inset_mm: float = 0.0,
    detected_corners: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select points inside the physical board mask.

    This compatibility wrapper intentionally keeps the historical two-value
    return contract.  Callers that need point-level audit linkage should use
    :func:`select_board_ground_points_with_mask` so the exact source indices
    are retained without reconstructing or guessing them later.
    """
    selection = select_board_ground_points_with_mask(
        pixels_uv_full,
        points_ground,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        pattern_cols=pattern_cols,
        pattern_rows=pattern_rows,
        square_size_mm=square_size_mm,
        image_offset=image_offset,
        inset_mm=inset_mm,
        detected_corners=detected_corners,
    )
    return selection.selected_points, selection.metadata


def select_board_ground_points_with_mask(
    pixels_uv_full: np.ndarray,
    points_ground: np.ndarray,
    *,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    image_offset: tuple[int, int] = (0, 0),
    inset_mm: float = 0.0,
    detected_corners: np.ndarray | None = None,
) -> BoardGroundPointSelection:
    """Select reconstructed points inside the complete physical board mask.

    ``pixels_uv_full`` uses full-sensor calibration coordinates, while
    ``rvec/tvec`` and the camera matrix use the local PnP ROI coordinates.
    Returned points retain reconstruction order and remain uncorrected.
    """
    pixels = np.asarray(pixels_uv_full, dtype=np.float64)
    points = np.asarray(points_ground, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_uv_full 必须是形状为 (N, 2) 的数组")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ground 必须是形状为 (N, 3) 的数组")
    if len(pixels) != len(points):
        raise ValueError("pixels_uv_full 与 points_ground 必须逐点对齐")
    polygon = full_board_physical_polygon(
        rvec,
        tvec,
        pattern_cols=pattern_cols,
        pattern_rows=pattern_rows,
        square_size_mm=square_size_mm,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_offset=image_offset,
        inset_mm=inset_mm,
    )
    selected_mask = _points_inside_convex_polygon(pixels, polygon)
    selected_points = np.ascontiguousarray(points[selected_mask], dtype=np.float64)
    selected_indices = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
    selected_count = int(np.count_nonzero(selected_mask))
    metadata: dict[str, Any] = {
        "enabled": True,
        "status": "applied",
        "source": "pnp_board_mask",
        "mask_mode": "full_board_physical",
        "corner_count": (
            int(len(np.asarray(detected_corners).reshape(-1, 2)))
            if detected_corners is not None
            else None
        ),
        "pattern_cols": int(pattern_cols),
        "pattern_rows": int(pattern_rows),
        "square_size_mm": float(square_size_mm),
        "inset_mm": float(inset_mm),
        "input_point_count": int(len(points)),
        "selected_point_count": selected_count,
        "rejected_point_count": int(len(points) - selected_count),
        "polygon_full_uv": polygon.tolist(),
    }
    return BoardGroundPointSelection(
        selected_points=selected_points,
        selected_indices=np.ascontiguousarray(selected_indices),
        selected_mask=np.ascontiguousarray(selected_mask, dtype=bool),
        metadata=metadata,
    )


def select_manual_ground_roi_points(
    pixels_uv_full: np.ndarray,
    points_ground: np.ndarray,
    roi_rects_full: Sequence[Sequence[float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select points inside user-confirmed rectangular ground ROIs.

    The rectangles are expressed in full-sensor pixels.  This helper is
    intentionally separate from the checkerboard projection because a manual
    ROI is an explicit user support source, not an inferred board mask.
    """
    pixels = np.asarray(pixels_uv_full, dtype=np.float64)
    points = np.asarray(points_ground, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_uv_full 必须是形状为 (N, 2) 的数组")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ground 必须是形状为 (N, 3) 的数组")
    if len(pixels) != len(points):
        raise ValueError("pixels_uv_full 与 points_ground 必须逐点对齐")

    selected_mask = np.zeros(len(points), dtype=bool)
    normalized_rects: list[list[float]] = []
    for raw_rect in roi_rects_full:
        values = np.asarray(tuple(raw_rect), dtype=np.float64).reshape(-1)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("manual_ground_roi 矩形必须包含四个有限坐标")
        left, right = sorted((float(values[0]), float(values[2])))
        top, bottom = sorted((float(values[1]), float(values[3])))
        if right <= left or bottom <= top:
            raise ValueError("manual_ground_roi 矩形必须具有正宽度和正高度")
        normalized_rects.append([left, top, right, bottom])
        selected_mask |= (
            (pixels[:, 0] >= left)
            & (pixels[:, 0] <= right)
            & (pixels[:, 1] >= top)
            & (pixels[:, 1] <= bottom)
        )

    selected_count = int(np.count_nonzero(selected_mask))
    return np.ascontiguousarray(points[selected_mask], dtype=np.float64), {
        "enabled": True,
        "status": "applied" if normalized_rects and selected_count else "unavailable",
        "source": "manual_ground_roi",
        "mask_mode": "manual_rectangles_full",
        "roi_count": len(normalized_rects),
        "roi_rects_full": normalized_rects,
        "input_point_count": int(len(points)),
        "selected_point_count": selected_count,
        "rejected_point_count": int(len(points) - selected_count),
    }


def _points_inside_convex_polygon(
    points: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    candidates = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = candidates[:, None, :] - vertices[None, :, :]
    cross = (
        edges[None, :, 0] * relative[:, :, 1]
        - edges[None, :, 1] * relative[:, :, 0]
    )
    tolerance = 1.0e-9
    same_positive = np.all(cross >= -tolerance, axis=1)
    same_negative = np.all(cross <= tolerance, axis=1)
    return np.isfinite(candidates).all(axis=1) & (same_positive | same_negative)


__all__ = [
    "BoardGroundPointSelection",
    "full_board_physical_polygon",
    "select_board_ground_points",
    "select_board_ground_points_with_mask",
    "select_manual_ground_roi_points",
]
