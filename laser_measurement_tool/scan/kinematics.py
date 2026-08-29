"""Rigid-body kinematics for the first-stage pitch scan."""

from __future__ import annotations

import math

import numpy as np


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限数值") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number


def _as_points(points_camera: np.ndarray) -> np.ndarray:
    try:
        points = np.asarray(points_camera, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("points_camera 必须是数值 ndarray") from error
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera 必须是形状为 (N, 3) 的数组")
    if not np.isfinite(points).all():
        raise ValueError("points_camera 包含 NaN 或无穷值")
    return np.ascontiguousarray(points)


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是数值 ndarray") from error
    if vector.shape != (3,):
        raise ValueError(f"{name} 必须是形状为 (3,) 的数组")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} 包含 NaN 或无穷值")
    return vector


def _as_homogeneous_transform(value: np.ndarray) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("T_scan_from_camera_zero 必须是数值 ndarray") from error
    if transform.shape != (4, 4):
        raise ValueError("T_scan_from_camera_zero 必须是形状为 (4, 4) 的数组")
    if not np.isfinite(transform).all():
        raise ValueError("T_scan_from_camera_zero 包含 NaN 或无穷值")
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0
    ):
        raise ValueError("T_scan_from_camera_zero 必须是齐次变换矩阵")
    return transform


def _rodrigues_rotation(axis_direction: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return the right-handed rotation matrix for a unit axis."""
    x, y, z = axis_direction
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return (
        cosine * identity
        + (1.0 - cosine) * np.outer(axis_direction, axis_direction)
        + sine * skew
    )


def transform_points_camera_to_scan(
    points_camera: np.ndarray,
    angle_deg: float,
    axis_point_scan_mm: np.ndarray,
    axis_direction_scan: np.ndarray,
    zero_offset_deg: float,
    T_scan_from_camera_zero: np.ndarray,
) -> np.ndarray:
    """Transform camera-frame points into the fixed scan frame.

    ``T_scan_from_camera_zero`` first maps points from the camera frame into
    the scan frame at the zero pose.  The resulting points are then rotated
    around the arbitrary scan-frame axis through ``axis_point_scan_mm`` by
    ``angle_deg + zero_offset_deg`` using the right-hand convention.
    """
    points = _as_points(points_camera)
    axis_point = _as_vector(axis_point_scan_mm, "axis_point_scan_mm")
    axis_direction = _as_vector(axis_direction_scan, "axis_direction_scan")
    direction_norm = float(np.linalg.norm(axis_direction))
    if (
        not math.isfinite(direction_norm)
        or direction_norm <= np.finfo(np.float64).eps
    ):
        raise ValueError("axis_direction_scan 不能是零向量")
    axis_direction = axis_direction / direction_norm
    transform = _as_homogeneous_transform(T_scan_from_camera_zero)
    angle = _finite_scalar(angle_deg, "angle_deg")
    zero_offset = _finite_scalar(zero_offset_deg, "zero_offset_deg")
    total_angle_deg = angle + zero_offset
    if not math.isfinite(total_angle_deg):
        raise ValueError("angle_deg + zero_offset_deg 必须是有限数值")

    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)),
        axis=1,
    )
    points_zero = (homogeneous @ transform.T)[:, :3]

    rotation = _rodrigues_rotation(
        axis_direction,
        math.radians(total_angle_deg),
    )
    relative = points_zero - axis_point
    return np.ascontiguousarray(relative @ rotation.T + axis_point)


__all__ = ["transform_points_camera_to_scan"]
