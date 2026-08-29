"""Hardware-neutral scan-axis abstractions for offline/simulated scans."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


def _finite_angle(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限角度")
    try:
        angle = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限角度") from error
    if not math.isfinite(angle):
        raise ValueError(f"{name} 必须是有限角度")
    return angle


@runtime_checkable
class ScanAxis(Protocol):
    """扫描轴的最小硬件无关协议。"""

    def move_to(self, angle_deg: float) -> None:
        """移动到指定角度。"""
        ...

    def get_angle(self) -> float:
        """返回当前角度。"""
        ...

    def home(self) -> None:
        """回到零位。"""
        ...

    def stop(self) -> None:
        """停止运动。"""
        ...


class SimulatedScanAxis:
    """不连接硬件、立即完成命令的扫描轴模拟器。"""

    __slots__ = ("min_angle_deg", "max_angle_deg", "_angle_deg")

    def __init__(
        self,
        min_angle_deg: float = -90.0,
        max_angle_deg: float = 90.0,
    ) -> None:
        minimum = _finite_angle(min_angle_deg, "min_angle_deg")
        maximum = _finite_angle(max_angle_deg, "max_angle_deg")
        if minimum > maximum:
            raise ValueError("min_angle_deg 不能大于 max_angle_deg")
        if not minimum <= 0.0 <= maximum:
            raise ValueError("角度范围必须包含回零位置 0 度")
        self.min_angle_deg = minimum
        self.max_angle_deg = maximum
        self._angle_deg = 0.0

    def move_to(self, angle_deg: float) -> None:
        angle = _finite_angle(angle_deg, "angle_deg")
        if angle < self.min_angle_deg or angle > self.max_angle_deg:
            raise ValueError(
                f"angle_deg={angle} 超出范围 "
                f"[{self.min_angle_deg}, {self.max_angle_deg}]"
            )
        self._angle_deg = angle

    def get_angle(self) -> float:
        """返回最后一次命令角；模拟器中实测角与命令角相同。"""
        return self._angle_deg

    def home(self) -> None:
        self._angle_deg = 0.0

    def stop(self) -> None:
        """模拟器无异步运动，停止操作保持当前角度。"""
        return None
