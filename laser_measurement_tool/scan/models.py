"""Immutable data contracts for the first-stage offline scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_frame_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("frame_index 必须是整数")
    index = int(value)
    if index < 0:
        raise ValueError("frame_index 不能为负数")
    return index


def _validate_angle(value: float, name: str) -> float:
    try:
        angle = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限数值") from error
    if not np.isfinite(angle):
        raise ValueError(f"{name} 必须是有限数值")
    return angle


def _validate_points(value: np.ndarray, name: str, columns: int) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是数值 ndarray") from error
    if points.ndim != 2 or points.shape[1] != columns:
        raise ValueError(f"{name} 必须是形状为 (N, {columns}) 的数组")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} 包含 NaN 或无穷值")
    return np.ascontiguousarray(points)


@dataclass(frozen=True, slots=True)
class ScanPose:
    """俯仰扫描帧的指令角和实测角。"""

    frame_index: int
    angle_command_deg: float
    angle_measured_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", _validate_frame_index(self.frame_index))
        object.__setattr__(
            self,
            "angle_command_deg",
            _validate_angle(self.angle_command_deg, "angle_command_deg"),
        )
        object.__setattr__(
            self,
            "angle_measured_deg",
            _validate_angle(self.angle_measured_deg, "angle_measured_deg"),
        )


@dataclass(frozen=True, slots=True)
class ScanProfile:
    """单帧扫描剖面的像素、相机系点和扫描系点。"""

    frame_index: int
    angle_deg: float
    pixels_uv: np.ndarray
    points_camera: np.ndarray
    points_scan: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", _validate_frame_index(self.frame_index))
        object.__setattr__(self, "angle_deg", _validate_angle(self.angle_deg, "angle_deg"))
        pixels_uv = _validate_points(self.pixels_uv, "pixels_uv", 2)
        points_camera = _validate_points(self.points_camera, "points_camera", 3)
        points_scan = _validate_points(self.points_scan, "points_scan", 3)
        if not (len(pixels_uv) == len(points_camera) == len(points_scan)):
            raise ValueError(
                "pixels_uv、points_camera 和 points_scan 的点数必须一致"
            )
        object.__setattr__(self, "pixels_uv", pixels_uv)
        object.__setattr__(self, "points_camera", points_camera)
        object.__setattr__(self, "points_scan", points_scan)


@dataclass(frozen=True, slots=True)
class ScanResult:
    """多帧扫描剖面及累积后的扫描系点云。"""

    profiles: tuple[ScanProfile, ...]
    points_scan: np.ndarray

    def __post_init__(self) -> None:
        try:
            profiles = tuple(self.profiles)
        except TypeError as error:
            raise ValueError("profiles 必须是 ScanProfile 序列") from error
        if not all(isinstance(profile, ScanProfile) for profile in profiles):
            raise ValueError("profiles 中的元素必须是 ScanProfile")
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(
            self,
            "points_scan",
            _validate_points(self.points_scan, "points_scan", 3),
        )
