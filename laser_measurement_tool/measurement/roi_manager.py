"""ROI 区域管理与激光中心点筛选。"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np


class RoiKind(str, Enum):
    """工具支持的 ROI 类型。"""

    BASELINE = "baseline"
    OBSTACLE = "obstacle"


@dataclass(frozen=True, slots=True)
class RoiRegion:
    """使用图像像素边界坐标表示的矩形 ROI。"""

    kind: RoiKind
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


class RoiManager:
    """按添加顺序管理多个 ROI，并筛选亚像素中心点。"""

    def __init__(self) -> None:
        self._regions: list[RoiRegion] = []

    @property
    def regions(self) -> tuple[RoiRegion, ...]:
        """返回按添加顺序排列的只读区域序列。"""
        return tuple(self._regions)

    def add_region(
        self,
        kind: RoiKind | str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> RoiRegion:
        """归一化矩形坐标并添加一个 ROI。"""
        values = tuple(float(value) for value in (x0, y0, x1, y1))
        if not all(isfinite(value) for value in values):
            raise ValueError("ROI 坐标必须是有限数值")

        left, right = sorted((values[0], values[2]))
        top, bottom = sorted((values[1], values[3]))
        if right <= left or bottom <= top:
            raise ValueError("ROI 必须具有正宽度和正高度")

        region = RoiRegion(RoiKind(kind), left, top, right, bottom)
        self._regions.append(region)
        return region

    def remove_last(self) -> RoiRegion | None:
        """删除并返回最后添加的 ROI；没有区域时返回 ``None``。"""
        if not self._regions:
            return None
        return self._regions.pop()

    def clear(self) -> None:
        """删除全部 ROI。"""
        self._regions.clear()

    def filter_points(
        self,
        centers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(baseline_points, obstacle_points)``，保持原点顺序。"""
        points = np.asarray(centers, dtype=np.float64)
        if points.size == 0:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty.copy(), empty.copy()
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("激光中心点必须是形状为 (N, 2) 的数组")
        if not np.isfinite(points).all():
            raise ValueError("激光中心点包含 NaN 或无穷值")

        baseline_mask = self._mask_for_kind(points, RoiKind.BASELINE)
        obstacle_mask = self._mask_for_kind(points, RoiKind.OBSTACLE)
        return (
            np.ascontiguousarray(points[baseline_mask]),
            np.ascontiguousarray(points[obstacle_mask]),
        )

    def filter_points_by_region(
        self,
        centers: np.ndarray,
        kind: RoiKind | str,
    ) -> list[np.ndarray]:
        """按同类 ROI 添加顺序返回各自点集，不合并重叠区域。"""
        points = np.asarray(centers, dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float64)
        elif points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("激光中心点必须是形状为 (N, 2) 的数组")
        elif not np.isfinite(points).all():
            raise ValueError("激光中心点包含 NaN 或无穷值")

        resolved_kind = RoiKind(kind)
        return [
            np.ascontiguousarray(points[self._mask_for_region(points, region)])
            for region in self._regions
            if region.kind is resolved_kind
        ]

    def _mask_for_kind(self, points: np.ndarray, kind: RoiKind) -> np.ndarray:
        mask = np.zeros(len(points), dtype=bool)
        point_x = points[:, 0] + 0.5
        point_y = points[:, 1] + 0.5
        for region in self._regions:
            if region.kind is not kind:
                continue
            mask |= self._mask_for_region(points, region)
        return mask

    @staticmethod
    def _mask_for_region(points: np.ndarray, region: RoiRegion) -> np.ndarray:
        point_x = points[:, 0] + 0.5
        point_y = points[:, 1] + 0.5
        return (
            (point_x >= region.left)
            & (point_x <= region.right)
            & (point_y >= region.top)
            & (point_y <= region.bottom)
        )
